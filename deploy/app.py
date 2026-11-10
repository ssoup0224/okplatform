"""okchat guard API: the production wrapper around OkChatGuard (one model per process).

    uvicorn deploy.app:create_app --factory --host 0.0.0.0 --port 8080 --workers 1

Settings come from environment variables (see deploy/README.md). Endpoints, all JSON:

    GET  /healthz              the process is up (200 while the model is still loading)
    GET  /readyz               200 once the model is loaded; 503 while loading or after a failed load
    GET  /v1/info              release, serving system, mode, per-language thresholds
    GET  /v1/stats             request counts, latency percentiles, queue, score cache
    POST /v1/assess            {"messages": [...], "conv_id"?, "user_id"?} -> action for the newest user turn
    POST /v1/assess_response   {"messages": [...], "draft": "..."} -> allow | replace

Design notes:
* One worker thread owns the model and runs every model call (prompt check, reply check, log writes),
  so calls never overlap. That matters: the reply check switches the LoRA adapter off while it runs.
  The request deadline (OKG_TIMEOUT_MS) includes the time spent waiting for that thread. A request
  that times out while still waiting is cancelled; with OKG_MAX_QUEUE requests already in flight the
  server answers with the fail action at once instead of building a backlog.
* Scoring settings are pinned to the ones the run 2 release was scored and calibrated with
  (artifacts/full/S07_score: 2,048-token window, batch 4, 8B in 4-bit NF4). Config.apply_hardware_profile
  is deliberately not called: on a >= 60 GB GPU it would pick bf16 and a 4,096-token window, which shifts
  the scores away from the release thresholds.
* Shadow mode (OKG_SHADOW=1, the default): `action` is always "allow" and the model's verdict is in
  `model_action`. The chatbot always follows `action`, so enforcement is switched on here, not there.
* Server logs never contain conversation text or user ids. Logging conversations to feedback/
  (okg.feedback.FeedbackLog, PII-scrubbed, salted user hashes) is opt-in: OKG_LOG_CONVERSATIONS=1.
"""
from __future__ import annotations

import logging
import math
import os
import secrets
import sys
import threading
import time
import uuid
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("okg.serve")

FAIL_MODES = ("safe_answer", "block", "allow")
DECISION_SYSTEMS = ("", "stack", "guard", "guard_hate")
LANGS = ("ko", "en", "mixed")


# ======================================================================================
# settings
# ======================================================================================
def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or not v.strip():
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return default if v is None or not v.strip() else int(v)


def _env_str(name: str, default: str | None) -> str | None:
    v = os.environ.get(name)
    return default if v is None or not v.strip() else v.strip()


@dataclass
class Settings:
    project_dir: str = str(ROOT)       # holds releases/<mode>/ and feedback/<mode>/
    run_mode: str = "full"             # "smoke" serves the tiny CPU release (wiring checks only)
    release: str | None = None         # None -> releases/<mode>/CURRENT
    decision_system: str = ""          # "" -> what the release says (run 2: stack); "guard" = strict, no hate OR
    shadow: bool = True                # never enforce; report the verdict in model_action
    response_check: bool = False       # reply check lowers recall as calibrated in run 2 (63.3% -> 36.7%)
    timeout_ms: int = 2000             # per request, queue wait included; 0 = no deadline
    fail_mode: str = "safe_answer"     # action on timeout / overload / guard error
    max_queue: int = 8                 # requests in flight (running + waiting) before answering "busy"
    api_key: str | None = None         # required in X-API-Key or "Authorization: Bearer" when set
    log_conversations: bool = False    # write PII-scrubbed conversations to feedback/<mode>/log
    live_memory: bool = True           # add reviewer-confirmed entries from feedback/<mode>/memory_live
    max_seq_len: int = 2048            # \
    score_batch_size: int = 4          #  > run 2 scoring settings (full mode only)
    load_in_4bit: bool = True          # /
    max_messages: int = 200
    max_chars: int = 20000             # per message

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            project_dir=_env_str("OKG_PROJECT_DIR", d.project_dir),
            run_mode=_env_str("OKG_RUN_MODE", d.run_mode),
            release=_env_str("OKG_RELEASE", None),
            decision_system=_env_str("OKG_DECISION_SYSTEM", "") or "",
            shadow=_env_bool("OKG_SHADOW", d.shadow),
            response_check=_env_bool("OKG_RESPONSE_CHECK", d.response_check),
            timeout_ms=_env_int("OKG_TIMEOUT_MS", d.timeout_ms),
            fail_mode=_env_str("OKG_FAIL_MODE", d.fail_mode),
            max_queue=_env_int("OKG_MAX_QUEUE", d.max_queue),
            api_key=_env_str("OKG_API_KEY", None),
            log_conversations=_env_bool("OKG_LOG_CONVERSATIONS", d.log_conversations),
            live_memory=_env_bool("OKG_LIVE_MEMORY", d.live_memory),
            max_seq_len=_env_int("OKG_MAX_SEQ_LEN", d.max_seq_len),
            score_batch_size=_env_int("OKG_SCORE_BATCH_SIZE", d.score_batch_size),
            load_in_4bit=_env_bool("OKG_LOAD_IN_4BIT", d.load_in_4bit),
            max_messages=_env_int("OKG_MAX_MESSAGES", d.max_messages),
            max_chars=_env_int("OKG_MAX_CHARS", d.max_chars),
        )

    def validate(self) -> "Settings":
        if self.run_mode not in ("smoke", "full"):
            raise ValueError("OKG_RUN_MODE must be 'smoke' or 'full'")
        if self.fail_mode not in FAIL_MODES:
            raise ValueError(f"OKG_FAIL_MODE must be one of {FAIL_MODES}")
        if self.decision_system not in DECISION_SYSTEMS:
            raise ValueError(f"OKG_DECISION_SYSTEM must be empty or one of {DECISION_SYSTEMS[1:]}")
        if self.max_queue < 1:
            raise ValueError("OKG_MAX_QUEUE must be >= 1")
        if self.log_conversations and not os.environ.get("OKG_USER_SALT"):
            raise ValueError("OKG_LOG_CONVERSATIONS=1 needs OKG_USER_SALT (a long random secret that never "
                             "changes), or user ids would be hashed with the public default salt")
        return self

    def public(self) -> dict:
        d = asdict(self)
        d["api_key"] = bool(self.api_key)
        return d


def serving_config(s: Settings):
    from okg.config import Config
    cfg = Config(run_mode=s.run_mode, project_dir=s.project_dir)
    if s.run_mode == "full":
        cfg.max_seq_len, cfg.score_batch_size, cfg.load_in_4bit = s.max_seq_len, s.score_batch_size, s.load_in_4bit
    cfg.response_moderation = s.response_check
    cfg.guard_fail_mode = s.fail_mode
    cfg.guard_timeout_ms = 0          # the server enforces the deadline itself (queue wait included)
    return cfg


def load_guard(cfg, s: Settings):
    """The release's own serving system, or an override (same rules as tools/demo_server.py)."""
    from okg.inference import OkChatGuard
    from okg.release import current_name
    name = s.release or current_name(cfg)
    if not name:
        raise RuntimeError(f"no release: set OKG_RELEASE or write the release name to {cfg.releases_dir / 'CURRENT'}")
    d = cfg.releases_dir / name
    if not (d / "release.json").exists():
        raise FileNotFoundError(f"release folder not found or incomplete: {d}")
    if not s.decision_system:
        return OkChatGuard.from_release(cfg, name, include_live_memory=s.live_memory)
    mems = [d / "memory"] + ([cfg.feedback_dir / "memory_live"] if s.live_memory else [])
    ds = "stack" if s.decision_system == "stack" else "guard"
    hate = None if s.decision_system == "guard" else d / "hate"
    return OkChatGuard(cfg, d / "adapter", d / "fold_ensemble.pkl", d / "thresholds.json", mems, name,
                       hate_dir=hate, decision_system=ds)


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, 6)


# ======================================================================================
# the model thread
# ======================================================================================
class GuardService:
    def __init__(self, settings: Settings, factory: Callable):
        self.s, self.factory = settings, factory
        self.guard = self.cfg = self.error = None
        self.load_s: float | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="guard")
        self._lock = threading.Lock()
        self._pending = 0
        self._feedback = None
        self._last_flush = time.monotonic()
        self.counts: Counter = Counter()
        self._lat: deque = deque(maxlen=10000)

    # ---- lifecycle ----
    def start_loading(self) -> None:
        self._pool.submit(self._load)

    def _load(self) -> None:
        t0 = time.perf_counter()
        try:
            cfg = serving_config(self.s)
            guard = self.factory(cfg, self.s)
            if self.s.log_conversations:
                from okg.feedback import FeedbackLog
                self._feedback = FeedbackLog(cfg.feedback_dir)
            self.cfg, self.guard = cfg, guard
            self.load_s = time.perf_counter() - t0
            log.info("guard ready: release=%s system=%s load_s=%.1f shadow=%s",
                     getattr(guard, "version", "?"), getattr(guard, "system", "?"), self.load_s, self.s.shadow)
        except Exception as e:  # noqa: BLE001  report through /readyz instead of killing the process
            self.error = f"{type(e).__name__}: {str(e)[:300]}"
            log.exception("guard failed to load")

    @property
    def ready(self) -> bool:
        return self.guard is not None

    def close(self) -> None:
        if self._feedback is not None:
            try:
                self._pool.submit(self._flush, True).result(timeout=30)
            except Exception:  # noqa: BLE001
                log.exception("final feedback flush failed")
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---- running work on the model thread ----
    def _run(self, fn, *args):
        """-> ("ok", result) | ("busy", None) | ("timeout", None) | ("error", message)"""
        with self._lock:
            if self._pending >= self.s.max_queue:
                return "busy", None
            self._pending += 1
        fut = self._pool.submit(self._job, fn, args)
        try:
            return "ok", fut.result(timeout=self.s.timeout_ms / 1000 if self.s.timeout_ms > 0 else None)
        except FutureTimeout:
            if fut.cancel():                      # still waiting: it will never run
                with self._lock:
                    self._pending -= 1
            return "timeout", None                # already running: it finishes, the result is dropped
        except Exception as e:  # noqa: BLE001
            log.exception("guard call failed")
            return "error", f"{type(e).__name__}: {str(e)[:120]}"

    def _job(self, fn, args):
        try:
            return fn(*args)
        finally:
            with self._lock:
                self._pending -= 1

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending

    def _flush(self, force: bool = False) -> None:
        fb = self._feedback
        if fb is None or not fb.buf:
            return
        if force or len(fb.buf) >= 50 or time.monotonic() - self._last_flush > 60:
            fb.flush()
            self._last_flush = time.monotonic()

    def _fail_action(self) -> str:
        return self.s.fail_mode

    # ---- prompt check ----
    def _assess_job(self, messages, conv_id, user_id):
        df = self.guard.assess(messages, log=self._feedback, conv_id=conv_id, user_id=user_id or "anonymous")
        try:
            self._flush()
        except Exception:  # noqa: BLE001  logging must never change the decision
            log.exception("feedback flush failed")
        last = df.iloc[-1]
        get = lambda k: last[k] if k in last.index else None  # noqa: E731
        return {"turn_idx": int(last["turn_idx"]), "lang": str(get("lang") or ""), "action": str(last["action"]),
                "p_block": _num(get("p_block")), "p_unsafe": _num(get("p_unsafe")), "p_contro": _num(get("p_contro")),
                "threshold": _num(get("threshold")), "guard_error": str(get("guard_error") or "")}

    def assess(self, messages: list[dict], conv_id: str | None, user_id: str | None) -> dict:
        t0 = time.perf_counter()
        status, res = self._run(self._assess_job, messages, conv_id, user_id)
        if status == "ok":
            out = res
        else:
            err = {"busy": "busy", "timeout": f"timeout>{self.s.timeout_ms}ms"}.get(status, f"error:{res}")
            out = {"turn_idx": sum(m["role"] == "user" for m in messages) - 1, "lang": None,
                   "action": self._fail_action(), "p_block": None, "p_unsafe": None, "p_contro": None,
                   "threshold": None, "guard_error": err}
        ms = (time.perf_counter() - t0) * 1000
        model_action = out.pop("action")
        self._count("assess", status if status != "ok" else ("guard_error" if out["guard_error"] else "ok"),
                    model_action, ms)
        return {"action": "allow" if self.s.shadow else model_action, "model_action": model_action,
                "shadow": self.s.shadow, **out, "latency_ms": round(ms, 1)}

    # ---- reply check ----
    def check_response(self, messages: list[dict], draft: str) -> dict:
        t0 = time.perf_counter()
        if not self.s.response_check:
            return {"action": "allow", "model_action": "allow", "shadow": self.s.shadow, "checked": False,
                    "p_unsafe": None, "threshold": None, "guard_error": "", "latency_ms": 0.0}
        status, res = self._run(self.guard.assess_response, messages, draft)
        if status == "ok":
            model_action, p, thr, err = str(res["action"]), _num(res.get("p_unsafe")), _num(res.get("threshold")), ""
        else:
            # The prompt check already ran; a failed second check only replaces the draft under fail_mode=block.
            model_action = "replace" if self.s.fail_mode == "block" else "allow"
            p = thr = None
            err = {"busy": "busy", "timeout": f"timeout>{self.s.timeout_ms}ms"}.get(status, f"error:{res}")
        ms = (time.perf_counter() - t0) * 1000
        self._count("response", status, model_action, ms)
        return {"action": "allow" if self.s.shadow else model_action, "model_action": model_action,
                "shadow": self.s.shadow, "checked": status == "ok", "p_unsafe": p, "threshold": thr,
                "guard_error": err, "latency_ms": round(ms, 1)}

    # ---- stats ----
    def _count(self, kind: str, status: str, action: str, ms: float) -> None:
        with self._lock:
            self.counts[f"{kind}.status.{status}"] += 1
            self.counts[f"{kind}.action.{action}"] += 1
            if kind == "assess":
                self._lat.append(ms)

    def stats(self) -> dict:
        with self._lock:
            lat = sorted(self._lat)
            counts = dict(self.counts)
        pct = (lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 1) if lat else None)
        cache = None
        try:
            cache = self.guard.cache.stats() if self.guard is not None and self.guard.cache is not None else None
        except Exception:  # noqa: BLE001
            pass
        return {"counts": counts, "pending": self.pending,
                "assess_latency_ms": {"n": len(lat), "p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99),
                                      "max": round(lat[-1], 1) if lat else None},
                "cache": cache}

    def info(self) -> dict:
        from okg.stacker import threshold_for
        g = self.guard
        safe = g.thresholds.get("safe_answer")
        return {"release": g.version, "run_mode": self.s.run_mode, "decision_system": g.decision_system,
                "system": g.system, "load_s": round(self.load_s or 0.0, 1),
                "thresholds": {l: {"block": _num(threshold_for(g.thresholds["block"], l)),
                                   "safe_answer": _num(threshold_for(safe, l)) if safe is not None else None}
                               for l in LANGS},
                "scoring": {"guard_model": self.cfg.guard_model_id, "load_in_4bit": bool(self.cfg.load_in_4bit),
                            "max_seq_len": self.cfg.max_seq_len, "score_batch_size": self.cfg.score_batch_size},
                "settings": self.s.public()}


# ======================================================================================
# HTTP
# ======================================================================================
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class AssessIn(BaseModel):
    messages: list[Message] = Field(min_length=1)
    conv_id: str | None = Field(default=None, max_length=200)
    user_id: str | None = Field(default=None, max_length=200)


class ResponseIn(BaseModel):
    messages: list[Message] = Field(min_length=1)
    draft: str


def _clean(messages: list[Message], s: Settings) -> list[dict]:
    """Drop system prompts (the guard scores user/assistant turns); the newest turn must be the user's."""
    msgs = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
    if not msgs or msgs[-1]["role"] != "user":
        raise HTTPException(422, "messages must end with a user turn (system messages are ignored)")
    if len(msgs) > s.max_messages:
        raise HTTPException(413, f"more than {s.max_messages} messages")
    if any(len(m["content"]) > s.max_chars for m in msgs):
        raise HTTPException(413, f"a message is longer than {s.max_chars} characters")
    return msgs


def create_app(settings: Settings | None = None, factory: Callable = load_guard) -> FastAPI:
    s = (settings or Settings.from_env()).validate()
    logging.basicConfig(level=os.environ.get("OKG_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    svc = GuardService(s, factory)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc.start_loading()
        yield
        svc.close()

    app = FastAPI(title="okchat guard", version="1.0", lifespan=lifespan)
    app.state.svc = svc

    def auth(request: Request) -> None:
        if not s.api_key:
            return
        got = request.headers.get("x-api-key") or ""
        bearer = request.headers.get("authorization") or ""
        if not got and bearer.lower().startswith("bearer "):
            got = bearer[7:].strip()
        if not got or not secrets.compare_digest(got.encode(), s.api_key.encode()):
            raise HTTPException(401, "missing or wrong API key")

    def ready() -> GuardService:
        if not svc.ready:
            raise HTTPException(503, "guard is loading" if svc.error is None else f"guard failed to load: {svc.error}")
        return svc

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        if request.url.path not in ("/healthz", "/readyz"):
            log.info("rid=%s %s %s %d %.1fms", rid, request.method, request.url.path, response.status_code,
                     (time.perf_counter() - t0) * 1000)
        return response

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/readyz")
    def readyz():
        if svc.ready:
            return {"ready": True, "release": svc.guard.version}
        return JSONResponse({"ready": False, "loading": svc.error is None, "error": svc.error}, status_code=503)

    @app.get("/v1/info", dependencies=[Depends(auth)])
    def info():
        return ready().info()

    @app.get("/v1/stats", dependencies=[Depends(auth)])
    def stats():
        return svc.stats()

    @app.post("/v1/assess", dependencies=[Depends(auth)])
    def assess(req: AssessIn):
        g = ready()
        out = g.assess(_clean(req.messages, s), req.conv_id, req.user_id)
        log.info("assess model_action=%s action=%s lang=%s turn=%s err=%s ms=%s", out["model_action"],
                 out["action"], out["lang"], out["turn_idx"], out["guard_error"] or "-", out["latency_ms"])
        return {**out, "release": g.guard.version}

    @app.post("/v1/assess_response", dependencies=[Depends(auth)])
    def assess_response(req: ResponseIn):
        g = ready()
        out = g.check_response(_clean(req.messages, s), req.draft)
        if out["checked"] or out["guard_error"]:
            log.info("response model_action=%s action=%s err=%s ms=%s", out["model_action"], out["action"],
                     out["guard_error"] or "-", out["latency_ms"])
        return {**out, "release": g.guard.version}

    return app

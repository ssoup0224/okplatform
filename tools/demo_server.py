"""Live chat demo of the okchat_guardrail release, for the capstone presentation video.

A local web page (tools/demo/index.html) where each user turn is scored by the CURRENT release
(`OkChatGuard.assess`), and the page shows the graded action, the per-turn risk curve against the
per-language thresholds, and the turn's guard / hate / memory scores. Standard library only: no extra
packages beyond what the notebook already needs.

    python tools/demo_server.py --mode full                       # GPU server, CURRENT full release
    python tools/demo_server.py --mode full --release v2026...    # a specific release
    python tools/demo_server.py --mode full --decision-system guard   # serve the plain fine-tuned guard
    python tools/demo_server.py --mode smoke                      # CPU, tiny smoke release (wiring check)
    python tools/demo_server.py --backend mock                    # no model at all (UI work only)

Then open http://127.0.0.1:7860 (on a remote GPU box: `ssh -L 7860:127.0.0.1:7860 <server>` first).

From the Jupyter notebook (after the "live conversation" cell has loaded `guard`), no second model load:

    import sys; sys.path.append("tools")     # append: tools/okg is an old copy, keep the real okg first
    import demo_server
    srv = demo_server.start(guard)       # background thread; the kernel stays usable
    # srv.stop()

and open <jupyter url>/proxy/7860/ (needs jupyter-server-proxy) or tunnel the port. JSON API for a
chatbot: POST /api/assess {"messages": [...]} -> {"action", "turns", "latency_ms"}.

Assistant replies: scenario turns carry a scripted draft (harmful drafts are redacted placeholders).
For free typing, pass `--chat-api http://127.0.0.1:8000/v1 --chat-model <name>` (any OpenAI-compatible
endpoint, e.g. vLLM serving Qwen2.5-7B-Instruct); the guard then also checks the draft
(`assess_response`). Without it, free-typed turns get a placeholder reply. The mock backend is a
keyword heuristic and the page labels it "MOCK" in red, so it cannot be mistaken for the model.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_okg = sys.modules.get("okg")
if _okg is not None and Path(_okg.__file__).resolve().parent != ROOT / "okg":
    raise ImportError(f"okg was imported from {Path(_okg.__file__).parent}, not {ROOT / 'okg'} (tools/okg is an old "
                      "copy). Restart the kernel and add tools with sys.path.append, not insert(0, ...).")
DEMO_DIR = Path(__file__).resolve().parent / "demo"

REPLIES = {
    "block": "죄송하지만 이 요청은 도와드릴 수 없습니다. 다른 질문이 있으시면 말씀해 주세요.",
    "safe_answer": "이 주제는 안전을 위해 일반적인 정보만 안내해 드릴게요. 위험할 수 있는 구체적인 방법은 "
                   "알려 드릴 수 없습니다. 도움이 필요하시면 관련 전문 기관에 문의해 주세요.",
    "fallback": "(데모: 자유 입력에는 챗봇 응답이 연결되어 있지 않습니다. --chat-api 를 지정하면 실제 응답이 나옵니다.)",
}
CAREFUL_NOTE = "\n\n※ 민감할 수 있는 주제라 신중하게 답변드렸습니다."


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else round(x, 4)


# ======================================================================================
# backends
# ======================================================================================
class RealBackend:
    """Wraps a loaded OkChatGuard. In a notebook, pass the `guard` you already have (no second 8B load)."""
    name = "release"

    def __init__(self, guard, load_s: float = 0.0):
        from okg.stacker import threshold_for
        self._thr_for = threshold_for
        self.guard, self.cfg, self.load_s = guard, guard.cfg, load_s
        self.lock = threading.Lock()          # one model, one request at a time (also vs. notebook cells)

    @classmethod
    def load(cls, mode: str, release: str | None, decision_system: str | None = None, cfg=None):
        from okg.config import Config
        from okg.inference import OkChatGuard
        from okg.release import current_name
        if cfg is None:
            cfg = Config(run_mode=mode, project_dir=str(ROOT))
            cfg.apply_hardware_profile(verbose=False)
        t0 = time.perf_counter()
        if decision_system:
            # Override the release's serving system. "guard" = fine-tuned guard, strict score, WITHOUT the hate
            # OR-rule (the system tools/analyze_run.py recommends after run 2); "guard_hate" = with it.
            name = release or current_name(cfg)
            if name is None:
                raise RuntimeError("no CURRENT release")
            d = cfg.releases_dir / name
            mems = [d / "memory", cfg.feedback_dir / "memory_live"]
            ds = "stack" if decision_system == "stack" else "guard"
            hate = None if decision_system == "guard" else d / "hate"
            guard = OkChatGuard(cfg, d / "adapter", d / "fold_ensemble.pkl", d / "thresholds.json", mems, name,
                                hate_dir=hate, decision_system=ds)
        else:
            guard = OkChatGuard.from_release(cfg, release)
        return cls(guard, time.perf_counter() - t0)

    def info(self) -> dict:
        g = self.guard
        safe = g.thresholds.get("safe_answer")
        langs = ("ko", "en", "mixed")
        return {"backend": self.name, "mode": self.cfg.run_mode, "version": g.version,
                "decision_system": g.decision_system, "system": g.system,
                "thresholds": {l: {"block": _num(self._thr_for(g.thresholds["block"], l)),
                                   "safe_answer": _num(self._thr_for(safe, l)) if safe is not None else None}
                               for l in langs},
                "response_check": bool(self.cfg.response_moderation), "load_s": round(self.load_s, 1)}

    def assess(self, messages: list[dict]) -> dict:
        with self.lock:
            t0 = time.perf_counter()
            df = self.guard.assess(messages)
            ms = (time.perf_counter() - t0) * 1000
        safe = self.guard.thresholds.get("safe_answer")
        turns = []
        for r in df.to_dict("records"):
            turns.append({"turn": int(r["turn_idx"]), "lang": r["lang"], "action": r["action"],
                          "p_block": _num(r.get("p_block")), "p_unsafe": _num(r.get("p_unsafe")),
                          "p_contro": _num(r.get("p_contro")), "hate": _num(r.get("hate_p_cur")),
                          "mem_attack": _num(r.get("mem_attack_max")), "mem_benign": _num(r.get("mem_benign_max")),
                          "threshold": _num(r.get("threshold")),
                          "safe_threshold": _num(self._thr_for(safe, r["lang"])) if safe is not None else None,
                          "error": r.get("guard_error") or ""})
        return {"turns": turns, "latency_ms": round(ms, 1)}

    def check_response(self, messages: list[dict], draft: str) -> dict:
        with self.lock:
            r = self.guard.assess_response(messages, draft)
        return {"action": r["action"], "p_unsafe": _num(r.get("p_unsafe")), "threshold": _num(r.get("threshold"))}


class MockBackend:
    """Keyword heuristic for building the page without a model. NOT the guardrail."""
    name = "mock"
    RISKY = {"해치": .9, "죽이": .9, "몰래": .5, "일부러": .5, "독": .4, "폭탄": .9, "malware": .9, "ransomware": .9,
             "bypass": .5, "no rules": .6, "dan": .3, "ㄷㅓ": .7, "쫓아내": .6, "없어져야": .7, "세제": .2}

    def info(self) -> dict:
        return {"backend": self.name, "mode": "mock", "version": "MOCK (실제 모델 아님)", "decision_system": "mock",
                "system": "keyword-heuristic", "thresholds": {l: {"block": .7, "safe_answer": .45}
                                                              for l in ("ko", "en", "mixed")},
                "response_check": False, "load_s": 0}

    def assess(self, messages: list[dict]) -> dict:
        from okg.policy import conversation_lang
        users = [m["content"] for m in messages if m["role"] == "user"]
        lang = conversation_lang(users)
        turns, prev = [], 0.0
        for u, t in enumerate(users):
            low = t.lower()
            p = min(.97, .03 + sum(v for k, v in self.RISKY.items() if k in low))
            prev = max(p, .8 * prev)
            act = "block" if prev > .7 else "safe_answer" if prev > .45 else "careful_answer" if p > .25 else "allow"
            turns.append({"turn": u, "lang": lang, "action": act, "p_block": round(prev, 4), "p_unsafe": round(p, 4),
                          "p_contro": round(min(p, .3), 4), "hate": None, "mem_attack": None, "mem_benign": None,
                          "threshold": .7, "safe_threshold": .45, "error": ""})
        time.sleep(0.15)
        return {"turns": turns, "latency_ms": 150.0}

    def check_response(self, messages, draft):
        return {"action": "allow", "p_unsafe": None, "threshold": None}


# ======================================================================================
# assistant reply
# ======================================================================================
def chat_draft(api: str, model: str, messages: list[dict]) -> str:
    body = json.dumps({"model": model, "messages": messages, "max_tokens": 400, "temperature": 0.3}).encode()
    req = urllib.request.Request(api.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def make_reply(backend, args, messages: list[dict], action: str, draft: str | None, guard_on: bool) -> dict:
    """The assistant turn the user sees, given the guard's action on the last user turn."""
    source = "scripted" if draft else None
    if not guard_on or action in ("allow", "careful_answer"):
        if draft is None and args.chat_api:
            try:
                draft, source = chat_draft(args.chat_api, args.chat_model, messages), "llm"
            except Exception as e:  # noqa: BLE001
                draft, source = f"(챗봇 응답 오류: {type(e).__name__})", "error"
        if draft is None:
            return {"text": REPLIES["fallback"], "source": "placeholder", "response_check": None}
        rc = None
        if guard_on and source == "llm" and getattr(backend, "name", "") != "mock":
            rc = backend.check_response(messages, draft)
            if rc["action"] == "replace":
                return {"text": REPLIES["safe_answer"], "source": "replaced", "response_check": rc}
        text = draft + (CAREFUL_NOTE if guard_on and action == "careful_answer" else "")
        return {"text": text, "source": source, "response_check": rc}
    return {"text": REPLIES[action], "source": "guard", "response_check": None}


# ======================================================================================
# http
# ======================================================================================
# GET  /                 demo page            GET  /api/health   {"ok": true}
# GET  /api/info         release / thresholds
# POST /api/assess       {"messages": [...]}  -> {"action", "turns", "latency_ms"}   (guard only: for a chatbot)
# POST /api/turn         {"messages", "draft"?, "guard_on"?} -> assess + the reply the demo page shows
# Routes match on the path suffix, so the page also works behind Jupyter's /proxy/<port>/ prefix.
def _route(path: str) -> str:
    path = path.split("?", 1)[0]
    for r in ("/api/health", "/api/info", "/api/assess", "/api/turn", "/scenarios.json"):
        if path.endswith(r):
            return r
    return "/" if path.endswith("/") or path.endswith("/index.html") else path


def _messages(req: dict) -> list[dict]:
    msgs = [{"role": m["role"], "content": str(m["content"])} for m in req.get("messages", [])
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    if not msgs or msgs[-1]["role"] != "user":
        raise ValueError("messages must be [{role: user|assistant, content}, ...] ending with a user turn")
    return msgs


def make_handler(backend, args):
    static = {"/": ("index.html", "text/html; charset=utf-8"),
              "/scenarios.json": ("scenarios.json", "application/json; charset=utf-8")}

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode())

        def log_message(self, fmt, *a):          # keep the console quiet (no conversation text in logs)
            pass

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            r = _route(self.path)
            if r == "/api/health":
                return self._json(200, {"ok": True})
            if r == "/api/info":
                return self._json(200, {**backend.info(), "chat_api": bool(args.chat_api)})
            if r in static:
                name, ctype = static[r]
                return self._send(200, (DEMO_DIR / name).read_bytes(), ctype)
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            r = _route(self.path)
            if r not in ("/api/assess", "/api/turn"):
                return self._send(404, b"not found", "text/plain")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                msgs = _messages(req)
            except (ValueError, KeyError, TypeError) as e:
                return self._json(400, {"error": str(e)})
            try:
                res = backend.assess(msgs)
                action = res["turns"][-1]["action"]
                if r == "/api/assess":
                    return self._json(200, {"action": action, **res})
                reply = make_reply(backend, args, msgs, action, req.get("draft"), bool(req.get("guard_on", True)))
                self._json(200, {**res, "reply": reply})
            except Exception as e:  # noqa: BLE001
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

    return H


# ======================================================================================
# start from a notebook (background thread in the kernel) or from the command line
# ======================================================================================
_RUNNING: dict[int, "DemoServer"] = {}


class DemoServer:
    def __init__(self, backend, args, host: str, port: int):
        self.backend, self.host, self.port = backend, host, port
        self.httpd = ThreadingHTTPServer((host, port), make_handler(backend, args))
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, name=f"okchat-demo-{port}", daemon=True)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        _RUNNING.pop(self.port, None)
        print(f"stopped {self.url}")

    def __repr__(self):
        return f"<DemoServer {self.url} running={self.thread.is_alive()}>"


def _jupyter_hint(port: int) -> str:
    try:
        import jupyter_server_proxy  # noqa: F401
        proxy = f"Jupyter 브라우저 주소 뒤에 /proxy/{port}/ (끝의 / 포함) — 예: http://<jupyter 주소>/proxy/{port}/"
    except ImportError:
        proxy = ("jupyter-server-proxy 미설치: `pip install jupyter-server-proxy` 후 Jupyter 서버를 재시작하면 "
                 f"<jupyter 주소>/proxy/{port}/ 로 열 수 있습니다")
    return (f"  · 이 서버의 브라우저: http://127.0.0.1:{port}/\n"
            f"  · Jupyter 프록시: {proxy}\n"
            f"  · SSH 터널: Mac에서 `ssh -N -L {port}:127.0.0.1:{port} <서버>` 후 http://127.0.0.1:{port}/\n"
            f"  · VS Code 원격 Jupyter: 포트 탭에서 {port} 전달")


def start(guard=None, *, cfg=None, mode: str = "full", release: str | None = None,
          decision_system: str | None = None, port: int = 7860, host: str = "127.0.0.1",
          chat_api: str | None = None, chat_model: str = "Qwen/Qwen2.5-7B-Instruct",
          mock: bool = False, warmup: bool = True) -> DemoServer:
    """Start the demo page + JSON API in a background thread and return at once (Jupyter-friendly).

        import sys; sys.path.append("tools")     # append: tools/okg is an old copy, keep the real okg first
        import demo_server
        srv = demo_server.start(guard)          # reuse the notebook's loaded OkChatGuard
        # srv.stop()                            # when done

    guard=None loads a release (`mode`/`release`/`decision_system`, or `cfg`). Calling start() again on the
    same port replaces the running server."""
    from types import SimpleNamespace
    if port in _RUNNING:
        _RUNNING[port].stop()
    if mock:
        backend = MockBackend()
    elif guard is not None:
        if decision_system:
            raise ValueError("decision_system only applies when start() loads the release (guard=None)")
        backend = RealBackend(guard)
    else:
        print(f"loading {mode} release {release or 'CURRENT'} …", flush=True)
        backend = RealBackend.load(mode, release, decision_system, cfg=cfg)
    if warmup and not mock:
        backend.assess([{"role": "user", "content": "안녕하세요"}])      # CUDA kernels, allocator
    srv = DemoServer(backend, SimpleNamespace(chat_api=chat_api, chat_model=chat_model), host, port)
    srv.thread.start()
    _RUNNING[port] = srv
    i = backend.info()
    print(f"okchat 데모 실행 중: backend={i['backend']} version={i['version']} decision={i['decision_system']}\n"
          + _jupyter_hint(port), flush=True)
    return srv


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", default="full", choices=["full", "smoke"])
    ap.add_argument("--release", default=None, help="release name (default: CURRENT)")
    ap.add_argument("--backend", default="release", choices=["release", "mock"])
    ap.add_argument("--decision-system", default=None, choices=["stack", "guard", "guard_hate"],
                    help="override the release's serving system (default: the one the release was made with)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--chat-api", default=None, help="OpenAI-compatible base URL for free-typed replies")
    ap.add_argument("--chat-model", default="Qwen/Qwen2.5-7B-Instruct")
    a = ap.parse_args()
    srv = start(mode=a.mode, release=a.release, decision_system=a.decision_system, port=a.port, host=a.host,
                chat_api=a.chat_api, chat_model=a.chat_model, mock=a.backend == "mock")
    try:
        srv.thread.join()
    except KeyboardInterrupt:
        srv.stop()


if __name__ == "__main__":
    main()

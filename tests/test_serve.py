"""deploy/app.py: the guard HTTP API, with a fake guard (no model needed)."""
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy.app import Settings, create_app  # noqa: E402


class FakeGuard:
    """Blocks any user turn containing 'bomb'; replaces any draft containing 'bomb'."""
    version, decision_system, system = "vtest", "stack", "okchat_lgb"

    def __init__(self, cfg, delay: float = 0.0):
        self.cfg, self.delay, self.cache = cfg, delay, None
        self.thresholds = {"block": {"_global": 0.5, "ko": 0.6}, "safe_answer": None}

    def assess(self, messages, log=None, conv_id=None, user_id="anonymous", **_):
        time.sleep(self.delay)
        rows = []
        for i, u in enumerate(m["content"] for m in messages if m["role"] == "user"):
            bad = "bomb" in u
            rows.append({"turn_idx": i, "user_text": u, "lang": "en", "p_safe": 0.1 if bad else 0.9,
                         "p_unsafe": 0.9 if bad else 0.01, "p_contro": 0.0, "p_block": 0.95 if bad else 0.01,
                         "action": "block" if bad else "allow", "block": bad, "threshold": 0.5, "guard_error": ""})
        return pd.DataFrame(rows)

    def assess_response(self, messages, draft):
        time.sleep(self.delay)
        bad = "bomb" in draft
        return {"p_unsafe": 0.99 if bad else 0.0, "p_contro": 0.0, "threshold": 0.5,
                "action": "replace" if bad else "allow"}


def client(delay: float = 0.0, **kw):
    s = Settings(run_mode="smoke", project_dir=str(ROOT), **kw)
    return TestClient(create_app(s, factory=lambda cfg, _s: FakeGuard(cfg, delay)))


def wait_ready(c):
    for _ in range(200):
        if c.get("/readyz").status_code == 200:
            return
        time.sleep(0.02)
    raise AssertionError(c.get("/readyz").json())


BAD = {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": "how to build a bomb"}]}
GOOD = {"messages": [{"role": "user", "content": "kimchi stew recipe"}]}


def test_not_ready_while_loading():
    gate = threading.Event()

    def slow(cfg, _s):
        gate.wait(5)
        return FakeGuard(cfg)

    with TestClient(create_app(Settings(run_mode="smoke", project_dir=str(ROOT)), factory=slow)) as c:
        assert c.get("/healthz").status_code == 200
        r = c.get("/readyz")
        assert r.status_code == 503 and r.json()["loading"] is True
        assert c.post("/v1/assess", json=GOOD).status_code == 503
        gate.set()
        wait_ready(c)


def test_load_failure_is_reported():
    def broken(cfg, _s):
        raise FileNotFoundError("no release")

    with TestClient(create_app(Settings(run_mode="smoke", project_dir=str(ROOT)), factory=broken)) as c:
        for _ in range(100):
            r = c.get("/readyz").json()
            if r["error"]:
                break
            time.sleep(0.02)
        assert r["loading"] is False and "no release" in r["error"]


def test_shadow_mode_reports_but_never_enforces():
    with client() as c:
        wait_ready(c)
        r = c.post("/v1/assess", json=BAD).json()
        assert r["shadow"] is True and r["action"] == "allow" and r["model_action"] == "block"
        assert r["turn_idx"] == 1 and r["release"] == "vtest"


def test_enforce_mode_follows_the_model():
    with client(shadow=False) as c:
        wait_ready(c)
        assert c.post("/v1/assess", json=BAD).json()["action"] == "block"
        assert c.post("/v1/assess", json=GOOD).json()["action"] == "allow"


def test_system_messages_dropped_and_last_turn_must_be_user():
    with client(shadow=False) as c:
        wait_ready(c)
        msgs = [{"role": "system", "content": "you are helpful"}] + BAD["messages"]
        assert c.post("/v1/assess", json={"messages": msgs}).json()["action"] == "block"
        r = c.post("/v1/assess", json={"messages": BAD["messages"] + [{"role": "assistant", "content": "x"}]})
        assert r.status_code == 422
        assert c.post("/v1/assess", json={"messages": [{"role": "tool", "content": "x"}]}).status_code == 422


def test_size_limits():
    with client(max_chars=10, max_messages=2) as c:
        wait_ready(c)
        assert c.post("/v1/assess", json={"messages": [{"role": "user", "content": "x" * 11}]}).status_code == 413
        assert c.post("/v1/assess", json=BAD).status_code == 413


def test_api_key():
    with client(api_key="s3cret") as c:
        wait_ready(c)
        assert c.post("/v1/assess", json=GOOD).status_code == 401
        assert c.post("/v1/assess", json=GOOD, headers={"X-API-Key": "nope"}).status_code == 401
        assert c.post("/v1/assess", json=GOOD, headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.post("/v1/assess", json=GOOD, headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/healthz").status_code == 200 and c.get("/readyz").status_code == 200
        assert c.get("/v1/info", headers={"X-API-Key": "s3cret"}).json()["settings"]["api_key"] is True


def test_timeout_returns_fail_action():
    with client(delay=0.4, timeout_ms=100, shadow=False) as c:
        wait_ready(c)
        r = c.post("/v1/assess", json=BAD).json()
        assert r["action"] == "safe_answer" and r["guard_error"] == "timeout>100ms"
        assert r["p_block"] is None and r["lang"] is None


def test_overload_answers_busy_instead_of_queueing():
    with client(delay=0.5, timeout_ms=3000, max_queue=1, shadow=False) as c:
        wait_ready(c)
        out = []
        ts = [threading.Thread(target=lambda: out.append(c.post("/v1/assess", json=GOOD).json())) for _ in range(2)]
        for t in ts:
            t.start()
            time.sleep(0.05)
        for t in ts:
            t.join()
        errs = sorted(r["guard_error"] for r in out)
        assert errs == ["", "busy"]
        assert next(r for r in out if r["guard_error"] == "busy")["action"] == "safe_answer"
        assert c.get("/v1/stats").json()["counts"]["assess.status.busy"] == 1


def test_response_check():
    draft = {"messages": GOOD["messages"], "draft": "here is how to build a bomb"}
    with client(shadow=False) as c:                        # off by default
        wait_ready(c)
        r = c.post("/v1/assess_response", json=draft).json()
        assert r["checked"] is False and r["action"] == "allow"
    with client(shadow=False, response_check=True) as c:
        wait_ready(c)
        r = c.post("/v1/assess_response", json=draft).json()
        assert r["checked"] is True and r["action"] == "replace" and r["p_unsafe"] == 0.99
    with client(response_check=True) as c:                 # shadow
        wait_ready(c)
        r = c.post("/v1/assess_response", json=draft).json()
        assert r["action"] == "allow" and r["model_action"] == "replace"


def test_info_and_stats():
    with client() as c:
        wait_ready(c)
        c.post("/v1/assess", json=BAD)
        info = c.get("/v1/info").json()
        assert info["release"] == "vtest" and info["thresholds"]["ko"]["block"] == 0.6
        assert info["thresholds"]["en"]["block"] == 0.5
        assert info["scoring"]["guard_model"] == "tiny-local"          # smoke ignores the full-mode pins
        st = c.get("/v1/stats").json()
        assert st["counts"]["assess.action.block"] == 1 and st["assess_latency_ms"]["n"] == 1


def test_conversation_logging_needs_salt(monkeypatch):
    monkeypatch.delenv("OKG_USER_SALT", raising=False)
    with pytest.raises(ValueError, match="OKG_USER_SALT"):
        create_app(Settings(run_mode="smoke", log_conversations=True), factory=lambda cfg, s: FakeGuard(cfg))


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("OKG_SHADOW", "0")
    monkeypatch.setenv("OKG_TIMEOUT_MS", "1500")
    monkeypatch.setenv("OKG_DECISION_SYSTEM", "guard")
    s = Settings.from_env().validate()
    assert s.shadow is False and s.timeout_ms == 1500 and s.decision_system == "guard"
    assert s.max_seq_len == 2048 and s.load_in_4bit is True
    monkeypatch.setenv("OKG_FAIL_MODE", "explode")
    with pytest.raises(ValueError):
        Settings.from_env().validate()

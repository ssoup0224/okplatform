"""Per-turn serving latency of a release on THIS machine's GPU (run it on the target GPU). [D45][D49]

Replays test-role conversations turn by turn, as a live chat would call the guard, and reports p50 / p95 /
p99 per assess call, with the per-turn cache on and off (off = the pre-D45 behaviour of re-scoring every
earlier turn), plus the response check. Suggests guard_timeout_ms from the measured p99.

    python tools/bench_latency.py --mode full                     # CURRENT release
    python tools/bench_latency.py --mode full --release v2026...  --convs 200 --min-turns 4
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from okg.cache import LRUCache  # noqa: E402
from okg.config import Config  # noqa: E402
from okg.inference import OkChatGuard  # noqa: E402


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def replay(guard, convs, response: bool):
    per_call, per_resp, rows = [], [], []
    for c in convs:
        users = [i for i, m in enumerate(c) if m["role"] == "user"]
        for u, i in enumerate(users):
            t0 = time.perf_counter()
            out = guard.assess(c[: i + 1], timeout_ms=0)                 # no timeout while measuring
            ms = (time.perf_counter() - t0) * 1000
            per_call.append(ms)
            rows.append({"turn": u, "ms": ms, "error": out["guard_error"].iloc[-1]})
            if response and i + 1 < len(c) and c[i + 1]["role"] == "assistant":
                t0 = time.perf_counter()
                guard.assess_response(c[: i + 1], c[i + 1]["content"])
                per_resp.append((time.perf_counter() - t0) * 1000)
    return np.array(per_call), np.array(per_resp), pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full")
    ap.add_argument("--release", default=None)
    ap.add_argument("--convs", type=int, default=100)
    ap.add_argument("--min-turns", type=int, default=3)
    ap.add_argument("--no-uncached", action="store_true", help="skip the slow cache-off comparison")
    a = ap.parse_args()
    cfg = Config(run_mode=a.mode, project_dir=str(ROOT))
    cfg.apply_hardware_profile(verbose=False)
    guard = OkChatGuard.from_release(cfg, a.release)
    convs = pd.read_parquet(cfg.artifacts / "S05_prefixes" / "convs.parquet", columns=["role", "n_user_turns", "messages"])
    convs = convs[(convs["role"] == "test") & (convs["n_user_turns"] >= a.min_turns)]
    convs = [json.loads(m) for m in convs.sample(min(a.convs, len(convs)), random_state=cfg.seed)["messages"]]
    replay(guard, convs[:3], False)                                       # warm-up (CUDA kernels, allocator)
    guard.cache = LRUCache(cfg.score_cache_size)
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    cached, resp, rows = replay(guard, convs, cfg.response_moderation)
    res = {"gpu": gpu, "release": guard.version, "decision_system": guard.decision_system, "calls": len(cached),
           "cached": {"p50_ms": pct(cached, 50), "p95_ms": pct(cached, 95), "p99_ms": pct(cached, 99)},
           "response_check": {"p50_ms": pct(resp, 50), "p95_ms": pct(resp, 95)} if len(resp) else None,
           "p95_by_turn": rows.groupby("turn")["ms"].quantile(0.95).round(1).to_dict()}
    if not a.no_uncached:
        guard.cache = LRUCache(0)                                         # pre-D45: every earlier turn re-scored
        unc, _, _ = replay(guard, convs, False)
        res["uncached"] = {"p50_ms": pct(unc, 50), "p95_ms": pct(unc, 95), "p99_ms": pct(unc, 99)}
    res["suggested_guard_timeout_ms"] = int(np.ceil(1.5 * res["cached"]["p99_ms"] / 100) * 100)
    out = cfg.feedback_dir / "latency"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"latency-{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()

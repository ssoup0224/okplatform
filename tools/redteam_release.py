"""Scheduled red-team pass against a release (cron / CI). [D55]

Builds a FRESH suite each run (date-seeded): structural padding / long-paste cases and attacker-LLM
dialogues, all from TEST-role data; runs it against the release; writes
feedback/<mode>/redteam/<date>-<release>.json and exits 1 if the attack success rate rose by more than
cfg.redteam_asr_tol versus the previous run's result for the same release line.

    python tools/redteam_release.py --mode full                         # CURRENT
    python tools/redteam_release.py --mode full --release v20261001-...

Example crontab entry (weekly, Monday 03:00):
    0 3 * * 1  cd /path/to/okplatform-main && python tools/redteam_release.py --mode full
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from okg import redteam as rt  # noqa: E402
from okg.config import Config  # noqa: E402
from okg.inference import OkChatGuard  # noqa: E402
from okg.release import current_name  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full")
    ap.add_argument("--release", default=None)
    ap.add_argument("--n", type=int, default=None, help="cases per strategy (default cfg.redteam_n_per_strategy)")
    a = ap.parse_args()
    cfg = Config(run_mode=a.mode, project_dir=str(ROOT))
    cfg.apply_hardware_profile(verbose=False)
    name = a.release or current_name(cfg)
    n = a.n or cfg.redteam_n_per_strategy
    convs = pd.read_parquet(cfg.artifacts / "S05_prefixes" / "convs.parquet")
    seed = int(time.strftime("%Y%m%d"))
    model = tok = None
    if cfg.run_mode == "full" and cfg.attack_gen:
        from okg import guard as G
        tok = G.load_tokenizer(cfg.attack_gen_model, cfg.hf_token)
        model = G.load_model(cfg.attack_gen_model, cfg, cfg.hf_token)
    suite = rt.suite_from_data(convs, cfg, seed, n) + rt.suite_generated(convs, cfg, seed, n, model, tok)
    del model
    guard = OkChatGuard.from_release(cfg, name)
    summ, det = rt.run_suite(guard, suite)
    out = cfg.feedback_dir / "redteam"
    out.mkdir(parents=True, exist_ok=True)
    prev = sorted(out.glob(f"*-{name}.json"))
    p = out / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}.json"
    p.write_text(json.dumps({"release": name, "seed": seed, "summary": summ}, indent=1))
    det.to_csv(p.with_suffix(".csv"), index=False)
    print(json.dumps(summ, indent=1))
    if prev:
        last = json.loads(prev[-1].read_text())["summary"]
        ok, rows = rt.redteam_gate(summ, last, cfg)
        print(rows)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

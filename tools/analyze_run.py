"""Offline analyses of a finished run -> artifacts/<mode>/analysis/ (CSV + JSON + REPORT.md).

    python tools/analyze_run.py --mode full            # all analyses (LightGBM refits take minutes)
    python tools/analyze_run.py --mode full --skip-ablation
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from okg import analysis as A  # noqa: E402
from okg.config import Config  # noqa: E402


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full")
    ap.add_argument("--skip-ablation", action="store_true")
    a = ap.parse_args()
    cfg = Config(run_mode=a.mode, project_dir=str(ROOT))
    art = cfg.artifacts
    out = art / "analysis"
    out.mkdir(exist_ok=True)
    df, meta = A.load(art)
    rep = [f"# Offline analysis of `{art}`\n", f"_generated {time.strftime('%Y-%m-%d %H:%M')}_\n"]
    summary = {}

    t0 = time.time()
    th = A.thresholds_and_actions(df, cfg)
    th["thresholds"].to_csv(out / "thresholds_variants.csv", index=False)
    rep += ["## Thresholds: global vs per-language + Clopper-Pearson 95%\n",
            md_table(th["thresholds"][th["thresholds"]["group"].isin(["ALL", "lang=ko", "lang=en", "lang=mixed"])]), "\n"]
    rep.append("## Graded actions (conversation level, per-language thresholds)\n")
    for name, d in th["actions"].items():
        for suite, tab in d.items():
            rep += [f"**{name} — {suite}**\n", md_table(tab), "\n"]
    print(f"thresholds/actions {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    sv = A.stack_vs_guard(df, cfg)
    summary["stack_vs_guard"] = sv
    rep += ["## Stack vs guard-only (P(Unsafe)+P(Controversial), per-language thresholds)\n",
            f"Recommended: **{sv['recommended']}**\n", md_table(pd.DataFrame(sv["suites"]).T.reset_index()), "\n"]
    if "nomem_vs_guard" in sv:
        rep += ["Stack without memory vs guard:\n", md_table(pd.DataFrame(sv["nomem_vs_guard"]["suites"]).T.reset_index()), "\n"]
    print(f"stack vs guard {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    fl = A.flip_analysis(df, art)
    fl["examples"].to_csv(out / "flip_examples.csv", index=False)
    summary["flips"] = fl["summary"]
    rep += ["## English flips (zero-shot caught, fine-tuned missed; strict score at 0.5)\n",
            "```json\n" + json.dumps(fl["summary"], indent=1, ensure_ascii=False) + "\n```\n",
            "Examples with nearest guard_train neighbours: `flip_examples.csv`.\n"]
    print(f"flips {time.time() - t0:.0f}s", flush=True)

    if not a.skip_ablation:
        t0 = time.time()
        ab = A.base_feature_ablation(df, meta, cfg)
        ab["metrics"].to_csv(out / "base_feature_ablation.csv", index=False)
        summary["base_ablation"] = {k: v for k, v in ab.items() if k != "metrics"}
        m = ab["metrics"]
        rep += ["## Base-feature ablation (same protocol, global threshold at the FPR budget)\n",
                json.dumps(summary["base_ablation"]) + "\n\n",
                md_table(m[m["group"].isin(["ALL", "suite=external", "suite=human_multiturn", "lang=en", "lang=ko"])]), "\n"]
        print(f"ablation {time.time() - t0:.0f}s", flush=True)

    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str, ensure_ascii=False))
    (out / "REPORT.md").write_text("\n".join(rep), encoding="utf-8")
    print("wrote", out / "REPORT.md")


if __name__ == "__main__":
    main()

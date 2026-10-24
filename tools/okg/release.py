"""Versioned releases, promotion gate, rollback, and the live (between-release) memory.

releases/<mode>/<name>/            adapter/, fold_ensemble.pkl, thresholds.json, memory/, metrics_by_source.parquet,
                                   release.json   — written to a temp dir and renamed into place [R9]
releases/<mode>/CURRENT            name of the serving release (atomic write)          [D35]
feedback/<mode>/memory_live/       reviewer confirmations added since the current release (instant effect) [D27]
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import atomic_write_bytes, atomic_write_json


# ======================================================================================
# promotion gate
# ======================================================================================
def promotion_gate(candidate: pd.DataFrame, current: pd.DataFrame | None, cfg, system: str = "okchat_lgb"):
    """Candidate may replace current only if, on every dataset both were evaluated on (with at
    least gate_min_n positives/negatives), recall does not drop and FPR does not rise beyond the
    tolerance BY MORE THAN SAMPLING NOISE: a check fails only when the candidate's 95% bootstrap
    interval lies entirely beyond the tolerance (recall_ci_hi < current - tol, or
    fpr_ci_lo > current + tol) [R18]. With small test sets one conversation moves recall by
    several points, so point estimates alone would reject good releases. Overall FPR must stay
    within budget + tol. New datasets are reported, not gated. [D35]"""
    cand = candidate[candidate["system"] == system].set_index("group")
    rows, ok = [], True
    all_fpr = cand.at["ALL", "fpr"] if "ALL" in cand.index else np.nan
    budget_ok = not (all_fpr > cfg.fpr_budget + cfg.gate_fpr_tol)
    rows.append({"group": "ALL", "check": "fpr <= budget+tol", "candidate": all_fpr, "current": cfg.fpr_budget,
                 "passed": budget_ok})
    ok &= budget_ok
    if current is not None and len(current):
        cur = current[current["system"] == system].set_index("group")
        for g in cand.index.intersection(cur.index):
            if g == "ALL":
                continue
            c, o = cand.loc[g], cur.loc[g]
            if min(c["n_pos"], o["n_pos"]) >= cfg.gate_min_n and not np.isnan(c["recall"]):
                hi = c.get("recall_ci_hi", c["recall"])
                hi = c["recall"] if pd.isna(hi) else hi
                passed = not (hi < o["recall"] - cfg.gate_recall_tol)
                rows.append({"group": g, "check": "recall", "candidate": c["recall"], "candidate_ci": hi,
                             "current": o["recall"], "passed": passed})
                ok &= passed
            if min(c["n_neg"], o["n_neg"]) >= cfg.gate_min_n and not np.isnan(c["fpr"]):
                lo = c.get("fpr_ci_lo", c["fpr"])
                lo = c["fpr"] if pd.isna(lo) else lo
                passed = not (lo > o["fpr"] + cfg.gate_fpr_tol)
                rows.append({"group": g, "check": "fpr", "candidate": c["fpr"], "candidate_ci": lo,
                             "current": o["fpr"], "passed": passed})
                ok &= passed
        for g in cand.index.difference(cur.index):
            rows.append({"group": g, "check": "new dataset (report only)", "candidate": cand.at[g, "recall"],
                         "current": np.nan, "passed": True})
    return bool(ok), pd.DataFrame(rows)


# ======================================================================================
# releases
# ======================================================================================
def current_name(cfg) -> str | None:
    p = cfg.releases_dir / "CURRENT"
    return p.read_text().strip() if p.exists() else None


def set_current(cfg, name: str) -> None:
    if not (cfg.releases_dir / name / "release.json").exists():
        raise FileNotFoundError(f"release '{name}' does not exist")
    atomic_write_bytes(cfg.releases_dir / "CURRENT", name.encode())


def list_releases(cfg) -> pd.DataFrame:
    rows = []
    if cfg.releases_dir.exists():
        for d in sorted(cfg.releases_dir.iterdir()):
            if (d / "release.json").exists():
                rows.append(json.loads((d / "release.json").read_text()))
    cur = current_name(cfg)
    df = pd.DataFrame(rows)
    if len(df):
        df["current"] = df["name"] == cur
    return df


def make_release(cfg, runner, name: str, promote: bool = True, force: bool = False) -> dict:
    """Package the finished pipeline as release <name>; promote it to CURRENT if the gate passes
    (or force=True). The release memory = evaluation memory + test-window confirmations + live
    confirmations, so everything reviewers confirmed is active in the new release. [D27][D35]"""
    from . import memory as mm
    for st in ("S06_sft", "S08_stacker", "S10_eval"):
        if runner.done_hash(st) is None:
            raise RuntimeError(f"stage {st} has not finished; cannot release")
    final = cfg.releases_dir / name
    if final.exists():
        raise FileExistsError(f"release '{name}' already exists")
    tmp = cfg.releases_dir / f"{name}.__tmp__"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copytree(runner.out_dir("S06_sft") / "adapter", tmp / "adapter")
    for f in ("fold_ensemble.pkl", "thresholds.json", "shap_importance.parquet"):
        shutil.copy2(runner.out_dir("S08_stacker") / f, tmp / f)
    metrics = pd.read_parquet(runner.out_dir("S10_eval") / "metrics_by_source.parquet")
    metrics.to_parquet(tmp / "metrics_by_source.parquet", index=False)
    mem_info = {}
    if cfg.use_attack_memory:
        enc = mm.make_encoder(cfg)
        mem = mm.Memory.load(runner.out_dir("S07b_memory") / "memory_eval", enc)
        add = pd.read_parquet(runner.out_dir("SF1_feedback_import") / "memory_additions.parquet")
        test_add = add[add["role"] == "test"] if len(add) else add
        for kind in mm.KINDS:
            a = test_add[test_add["kind"] == kind] if len(test_add) else test_add
            if len(a):
                mem.add(a["text"].tolist(), kind, a["group"].tolist(), "feedback:test")
        live = cfg.feedback_dir / "memory_live"
        if (live / "entries.parquet").exists():
            lm = mm.Memory.load(live, enc)
            for kind in mm.KINDS:
                e = lm.entries[(lm.entries["kind"] == kind) & lm.entries["active"]]
                have = set(zip(mem.entries["text"], mem.entries["group"]))
                e = e[[(t, g) not in have for t, g in zip(e["text"], e["group"])]]
                if len(e):
                    mem.add(e["text"].tolist(), kind, e["group"].tolist(), "feedback:live")
        mem.save(tmp / "memory")
        mem_info = mem.entries["kind"].value_counts().to_dict()
    cur = current_name(cfg)
    cur_metrics = pd.read_parquet(cfg.releases_dir / cur / "metrics_by_source.parquet") if cur else None
    passed, gate = promotion_gate(metrics, cur_metrics, cfg)
    gate.to_parquet(tmp / "gate.parquet", index=False)
    info = {"name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "run_mode": cfg.run_mode,
            "guard_model": cfg.guard_model_id, "compared_with": cur, "gate_passed": passed,
            "memory_entries": mem_info, "stage_hashes": {s: runner.done_hash(s) for s in
                                                         ("S06_sft", "S07b_memory", "S08_stacker", "S10_eval")}}
    atomic_write_json(tmp / "release.json", info)
    os.replace(tmp, final)                                                                        # [R9]
    if promote and (passed or force):
        set_current(cfg, name)
        # confirmations are now inside the release memory -> start a fresh live memory
        shutil.rmtree(cfg.feedback_dir / "memory_live", ignore_errors=True)
        info["promoted"] = True
    else:
        info["promoted"] = False
    return {"info": info, "gate": gate}


# ======================================================================================
# live memory: reviewer confirmations take effect before the next release
# ======================================================================================
def apply_reviews_to_live_memory(cfg) -> dict:
    """Import all reviewed files and add confirmations not yet in the CURRENT release's memory
    (or live memory) to feedback/<mode>/memory_live. OkChatGuard.from_release picks them up. [D27]"""
    from . import feedback as fb
    from . import memory as mm
    cur = current_name(cfg)
    if cur is None:
        raise RuntimeError("no CURRENT release yet — make_release first")
    enc = mm.make_encoder(cfg)
    _, add, rep = fb.import_reviews(cfg.review_files, fb.FeedbackLog(cfg.feedback_dir).load(), cfg)
    rel = mm.Memory.load(cfg.releases_dir / cur / "memory", enc)
    live_dir = cfg.feedback_dir / "memory_live"
    live = mm.Memory.load(live_dir, enc) if (live_dir / "entries.parquet").exists() else mm.Memory(enc)
    have = set(zip(rel.entries["text"], rel.entries["group"])) | set(zip(live.entries["text"], live.entries["group"]))
    added = {}
    for kind in mm.KINDS:
        a = add[(add["kind"] == kind)] if len(add) else add
        a = a[[(t, g) not in have for t, g in zip(a["text"], a["group"])]] if len(a) else a
        if len(a):
            live.add(a["text"].tolist(), kind, a["group"].tolist(), "feedback:live")
        added[kind] = int(len(a))
    live.save(live_dir)
    return {"added": added, "live_entries": int(len(live.entries)), "review_report": rep}

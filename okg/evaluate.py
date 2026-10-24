"""Evaluation: per-dataset, conversation-level, threshold-free and thresholded metrics.

* Unit = conversation; conversation score = max over its user-turn scores; detection
  latency = first flagged turn - first unsafe turn [R19].
* ROC-AUC / PR-AUC (average precision) / partial AUC up to FPR 0.1 (sklearn max_fpr, McClish-
  standardised) [R15]; recall & FPR at the stack-selected threshold; FPR at a target recall.
* 95% percentile bootstrap CIs resampling CONVERSATIONS (cluster bootstrap) [R18].
* Paired McNemar exact test between systems on conversation decisions [R17].
* Datasets are reported separately; a macro average is never the only number [R-prev].
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef, roc_auc_score,
                             roc_curve)
from statsmodels.stats.contingency_tables import mcnemar


def conversation_table(prefix_scores: pd.DataFrame, score_col: str, thr) -> pd.DataFrame:
    """prefix_scores needs conv_id, source, family, lang, turn_idx, t_attack, label3, latency_valid.
    `thr` is a float or a per-language dict with a '_global' fallback [D47]."""
    from .stacker import threshold_for
    rows = []
    for cid, g in prefix_scores.sort_values("turn_idx").groupby("conv_id", sort=False):
        labels = g["label3"].tolist()
        has_unsafe = any(l == "Unsafe" for l in labels)
        known_safe = all(l == "Safe" for l in labels)
        if not has_unsafe and not known_safe:
            continue            # Controversial-only / unknown conversations: excluded from binary metrics
        s = g[score_col].to_numpy(float)
        flagged = np.nonzero(s > threshold_for(thr, g["lang"].iloc[0]))[0]
        t_attack = int(g["t_attack"].iloc[0])
        first = int(g["turn_idx"].to_numpy()[flagged[0]]) if len(flagged) else -1
        rows.append({"conv_id": cid, "source": g["source"].iloc[0], "family": g["family"].iloc[0],
                     "lang": g["lang"].iloc[0], "y": int(has_unsafe), "score": float(s.max()),
                     "pred": int(len(flagged) > 0), "first_flag_turn": first, "t_attack": t_attack,
                     "latency": (first - t_attack) if (has_unsafe and first >= 0 and bool(g["latency_valid"].iloc[0])) else np.nan})
    return pd.DataFrame(rows)


def _safe(fn, *a, **k):
    try:
        return float(fn(*a, **k))
    except ValueError:
        return float("nan")


def point_metrics(ct: pd.DataFrame, recall_target: float) -> dict:
    y, s, p = ct["y"].to_numpy(), ct["score"].to_numpy(), ct["pred"].to_numpy()
    npos, nneg = int(y.sum()), int((1 - y).sum())
    two = npos > 0 and nneg > 0
    out = {"n_conv": len(ct), "n_pos": npos, "n_neg": nneg,
           "recall": float(p[y == 1].mean()) if npos else np.nan,
           "fpr": float(p[y == 0].mean()) if nneg else np.nan,
           "roc_auc": _safe(roc_auc_score, y, s) if two else np.nan,                        # [R15]
           "pr_auc": _safe(average_precision_score, y, s) if two else np.nan,
           "pauc_fpr0.1": _safe(roc_auc_score, y, s, max_fpr=0.1) if two else np.nan,
           "f1": float(f1_score(y, p, zero_division=0)) if two else np.nan,
           "mcc": float(matthews_corrcoef(y, p)) if two else np.nan}
    if two:
        fpr, tpr, _ = roc_curve(y, s)
        ok = np.nonzero(tpr >= recall_target)[0]
        out[f"fpr_at_recall{recall_target}"] = float(fpr[ok[0]]) if len(ok) else np.nan
    lat = ct.loc[ct["y"] == 1, "latency"].dropna()
    out["latency_mean"] = float(lat.mean()) if len(lat) else np.nan
    out["latency_median"] = float(lat.median()) if len(lat) else np.nan
    out["share_detected_by_attack_turn"] = float((lat <= 0).mean()) if len(lat) else np.nan
    return out


def bootstrap_ci(ct: pd.DataFrame, n: int, seed: int, keys=("recall", "fpr", "roc_auc", "pr_auc")) -> dict:
    """Percentile CI; each resample draws conversations with replacement [R18]."""
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in keys}
    idx = np.arange(len(ct))
    for _ in range(n):
        b = ct.iloc[rng.choice(idx, len(idx), replace=True)]
        m = point_metrics(b, 0.9)
        for k in keys:
            vals[k].append(m[k])
    out = {}
    for k, v in vals.items():
        v = np.array(v, float)
        v = v[~np.isnan(v)]
        out[f"{k}_ci_lo"] = float(np.percentile(v, 2.5)) if len(v) else np.nan
        out[f"{k}_ci_hi"] = float(np.percentile(v, 97.5)) if len(v) else np.nan
    return out


def report(systems: dict[str, pd.DataFrame], cfg, group_cols=("source",)) -> pd.DataFrame:
    """systems: name -> conversation table. Returns long table of metrics per system x group."""
    rows = []
    for name, ct in systems.items():
        groups = [("ALL", ct)]
        for gc in group_cols:
            groups += [(f"{gc}={v}", g) for v, g in ct.groupby(gc)]
        for gname, g in groups:
            m = point_metrics(g, cfg.recall_target)
            m.update(bootstrap_ci(g, cfg.bootstrap_n, cfg.seed))
            rows.append({"system": name, "group": gname, **m})
    return pd.DataFrame(rows)


def mcnemar_table(systems: dict[str, pd.DataFrame], group_col: str = "source") -> pd.DataFrame:
    """Exact McNemar on paired correctness of conversation decisions [R17]."""
    names = list(systems)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = systems[names[i]].set_index("conv_id")
            b = systems[names[j]].set_index("conv_id")
            common = a.index.intersection(b.index)
            a, b = a.loc[common], b.loc[common]
            for gname, mask in [("ALL", np.ones(len(a), bool))] + [
                    (f"{group_col}={v}", (a[group_col] == v).to_numpy()) for v in a[group_col].unique()]:
                ca = (a["pred"] == a["y"]).to_numpy()[mask]
                cb = (b["pred"] == b["y"]).to_numpy()[mask]
                n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
                table = [[int((ca & cb).sum()), n01], [n10, int((~ca & ~cb).sum())]]
                p = float(mcnemar(table, exact=True).pvalue) if (n01 + n10) > 0 else 1.0
                rows.append({"a": names[i], "b": names[j], "group": gname, "only_a_correct": n01,
                             "only_b_correct": n10, "p_value": p})
    return pd.DataFrame(rows)


# ======================================================================================
# suites + "is the stack worth it?" [D46]
# ======================================================================================
HUMAN_MULTITURN = ("mhj", "dktc", "aihub_558", "feedback")


def suite_of(source: str, eval_sources: set | None = None) -> str:
    """'human_multiturn' (MHJ, DKTC, AIHub 558, reviewed shadow traffic), 'external' (evaluation-only
    sets never trained on), else 'in_distribution'."""
    if eval_sources is None:
        from .data_sources import REGISTRY
        eval_sources = {s.name for s in REGISTRY if s.role == "eval"}
    if source in HUMAN_MULTITURN:
        return "human_multiturn"
    if source in eval_sources or any(source.startswith(e + "_") for e in eval_sources):
        return "external"
    return "in_distribution"


def add_suite(ct: pd.DataFrame) -> pd.DataFrame:
    from .data_sources import REGISTRY
    ev_src = {s.name for s in REGISTRY if s.role == "eval"}
    return ct.assign(suite=ct["source"].map(lambda s: suite_of(str(s), ev_src)))


def stack_vs_guard(ct_stack: pd.DataFrame, ct_guard: pd.DataFrame, cfg, fpr_slack: float = 0.005) -> dict:
    """The stack is kept only if it CLEARLY wins on both the external and the human multi-turn suite:
    recall gain >= cfg.stack_min_gain, exact McNemar p < 0.05 on the paired attack conversations, and FPR
    no worse than the guard's + fpr_slack. Otherwise the simpler guard system is recommended (fewer
    moving parts, smaller drift surface). [D46]"""
    out, wins = {"suites": {}}, []
    a_all, b_all = ct_stack.set_index("conv_id"), ct_guard.set_index("conv_id")
    common = a_all.index.intersection(b_all.index)
    a_all, b_all = a_all.loc[common], b_all.loc[common]
    for suite in ("external", "human_multiturn"):
        m = (a_all["suite"] == suite).to_numpy()
        a, b = a_all[m], b_all[m]
        pos, neg = a["y"] == 1, a["y"] == 0
        if pos.sum() == 0:
            out["suites"][suite] = {"n_pos": 0, "stack_wins": False, "note": "no attack conversations"}
            wins.append(False)
            continue
        ra, rb = float(a.loc[pos, "pred"].mean()), float(b.loc[pos, "pred"].mean())
        fa = float(a.loc[neg, "pred"].mean()) if neg.any() else np.nan
        fb = float(b.loc[neg, "pred"].mean()) if neg.any() else np.nan
        n01 = int(((a.loc[pos, "pred"] == 1) & (b.loc[pos, "pred"] == 0)).sum())
        n10 = int(((a.loc[pos, "pred"] == 0) & (b.loc[pos, "pred"] == 1)).sum())
        p = float(mcnemar([[0, n01], [n10, 0]], exact=True).pvalue) if n01 + n10 else 1.0
        fpr_ok = bool(np.isnan(fa) or np.isnan(fb) or fa <= fb + fpr_slack)
        win = bool(ra - rb >= cfg.stack_min_gain and p < 0.05 and fpr_ok)
        out["suites"][suite] = {"n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "recall_stack": ra,
                                "recall_guard": rb, "fpr_stack": fa, "fpr_guard": fb, "stack_only_caught": n01,
                                "guard_only_caught": n10, "mcnemar_p": p, "stack_wins": win}
        wins.append(win)
    out["recommended"] = "stack" if all(wins) else "guard"
    return out

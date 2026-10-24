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


def conversation_table(prefix_scores: pd.DataFrame, score_col: str, thr: float) -> pd.DataFrame:
    """prefix_scores needs conv_id, source, family, lang, turn_idx, t_attack, label3, latency_valid."""
    rows = []
    for cid, g in prefix_scores.sort_values("turn_idx").groupby("conv_id", sort=False):
        labels = g["label3"].tolist()
        has_unsafe = any(l == "Unsafe" for l in labels)
        known_safe = all(l == "Safe" for l in labels)
        if not has_unsafe and not known_safe:
            continue            # Controversial-only / unknown conversations: excluded from binary metrics
        s = g[score_col].to_numpy(float)
        flagged = np.nonzero(s > thr)[0]
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

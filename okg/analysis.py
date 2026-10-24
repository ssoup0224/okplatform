"""Offline analyses on a FINISHED run (no GPU): answers the full-run review's measurable questions from
artifacts/<mode>/S05..S10 without re-running the pipeline. Used by tools/analyze_run.py.

1. base_feature_ablation   LightGBM with vs without base_p_* features, same protocol            [D45]
2. thresholds_and_actions  global vs per-language thresholds with the Clopper-Pearson margin;
                           graded-action shares                                                  [D47][D48]
3. stack_vs_guard          full stack vs P(Unsafe)+P(Controversial) with per-language thresholds,
                           on the external and human multi-turn suites                           [D46]
4. flip_analysis           English examples the zero-shot guard caught and the fine-tuned guard
                           missed, with their nearest guard_train neighbours and those labels    [D52]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import features as fe
from . import policy as pol
from . import stacker as st


def load(art: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(art / "S08_stacker" / "scored.parquet")
    meta = json.loads((art / "S08_stacker" / "thresholds.json").read_text())
    df["p_strict"] = df["p_unsafe"] + df["p_contro"]
    return df, meta


def _systems_eval(df, cols_thr: dict, cfg) -> pd.DataFrame:
    test = df[df["role"] == "test"]
    rows = []
    for name, (col, thr) in cols_thr.items():
        ct = ev.add_suite(ev.conversation_table(test, col, thr))
        for g, sub in [("ALL", ct)] + [(f"suite={k}", v) for k, v in ct.groupby("suite")] + \
                      [(f"lang={k}", v) for k, v in ct.groupby("lang")]:
            m = ev.point_metrics(sub, cfg.recall_target)
            rows.append({"system": name, "group": g, "n_pos": m["n_pos"], "n_neg": m["n_neg"],
                         "recall": m["recall"], "fpr": m["fpr"], "roc_auc": m["roc_auc"]})
    return pd.DataFrame(rows)


def _thr(df, col, cfg, budget, conf, per_lang):
    stack = df[(df["role"] == "stack") & df["y"].notna()]
    ct = ev.conversation_table(stack, col, np.inf)
    return st.thresholds_by_lang(ct, budget, conf, cfg.threshold_min_benign, per_lang)


def base_feature_ablation(df: pd.DataFrame, meta: dict, cfg) -> dict:
    cols_all = meta["feature_columns"]
    cols_nb = [c for c in cols_all if c not in fe.BASE_FEATURES]
    stack = df[(df["role"] == "stack") & df["y"].notna()].reset_index(drop=True)
    y, groups = stack["y"].astype(int).to_numpy(), stack["seed_id"].to_numpy()
    out, ev_cols = {}, {}
    for tag, cols in (("with_base", cols_all), ("without_base", cols_nb)):
        res = st.fit(stack[cols], y, groups, cfg)
        col = f"lgb_{tag}"
        df[col] = res["model"].predict(df[cols])
        is_st = (df["role"] == "stack") & df["y"].notna()
        df.loc[is_st, col] = df.loc[is_st, "prefix_id"].map(dict(zip(stack["prefix_id"], res["oof_cal"])))
        thr = _thr(df, col, cfg, cfg.fpr_budget, None, False)
        ev_cols[tag] = (col, thr)
        from sklearn.metrics import roc_auc_score
        out[tag] = {"oof_auc": float(roc_auc_score(y, res["oof_raw"])), "n_features": len(cols)}
    tab = _systems_eval(df, ev_cols, cfg)
    out["metrics"] = tab
    return out


def thresholds_and_actions(df: pd.DataFrame, cfg) -> dict:
    variants = {
        "lgb global (old rule)": ("lgb", _thr(df, "lgb", cfg, cfg.fpr_budget, None, False)),
        "lgb per-lang + CP95": ("lgb", _thr(df, "lgb", cfg, cfg.fpr_budget, cfg.fpr_confidence, True)),
        "guard_strict global (old rule)": ("p_strict", _thr(df, "p_strict", cfg, cfg.fpr_budget, None, False)),
        "guard_strict per-lang + CP95": ("p_strict", _thr(df, "p_strict", cfg, cfg.fpr_budget, cfg.fpr_confidence, True)),
    }
    tab = _systems_eval(df, variants, cfg)
    actions = {}
    test = df[df["role"] == "test"]
    contro = test.groupby("conv_id")["p_contro"].max()
    for name, col in (("okchat_lgb", "lgb"), ("guard_strict", "p_strict")):
        thr = {"block": _thr(df, col, cfg, cfg.fpr_budget, cfg.fpr_confidence, True),
               "safe_answer": _thr(df, col, cfg, cfg.action_safe_fpr, cfg.fpr_confidence, True)}
        ct = ev.add_suite(ev.conversation_table(test, col, thr["block"]))
        ct["contro"] = ct["conv_id"].map(contro).fillna(0)
        actions[name] = {"ALL": pol.action_table(ct, thr, cfg)}
        for suite, g in ct.groupby("suite"):
            actions[name][suite] = pol.action_table(g, thr, cfg)
    return {"thresholds": tab, "actions": actions}


def stack_vs_guard(df: pd.DataFrame, cfg) -> dict:
    test = df[df["role"] == "test"]
    t_lgb = _thr(df, "lgb", cfg, cfg.fpr_budget, cfg.fpr_confidence, True)
    t_g = _thr(df, "p_strict", cfg, cfg.fpr_budget, cfg.fpr_confidence, True)
    a = ev.add_suite(ev.conversation_table(test, "lgb", t_lgb))
    b = ev.add_suite(ev.conversation_table(test, "p_strict", t_g))
    v = ev.stack_vs_guard(a, b, cfg)
    if "lgb_nomem" in df.columns:
        c = ev.add_suite(ev.conversation_table(test, "lgb_nomem", _thr(df, "lgb_nomem", cfg, cfg.fpr_budget,
                                                                        cfg.fpr_confidence, True)))
        v["nomem_vs_guard"] = ev.stack_vs_guard(c, b, cfg)
    return v


def flip_analysis(df: pd.DataFrame, art: Path, sources=("orbench_toxic", "xstest_ko"), k: int = 3,
                  max_rows: int = 400) -> dict:
    """Zero-shot caught (strict > 0.5) and fine-tuned missed (strict <= 0.5) on English positives; for each,
    the k nearest guard_train turns (char n-gram cosine) with their labels and sources."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    pre = pd.read_parquet(art / "S05_prefixes" / "prefixes.parquet",
                          columns=["role", "lang", "label3", "source", "user_text", "n_user_turns"])
    tr = pre[(pre["role"] == "guard_train") & (pre["lang"] == "en") & (pre["n_user_turns"] == 1)]
    tr = tr[tr["label3"].isin(["Safe", "Unsafe", "Controversial"])].drop_duplicates("user_text").reset_index(drop=True)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=300000)
    Xtr = vec.fit_transform(tr["user_text"])
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(Xtr)
    out, rows = {}, []
    test = df[(df["role"] == "test") & (df["y"] == 1)]
    for src in sources:
        g = test[(test["source"] == src) & (test["lang"] == "en")]
        zs = (g["base_p_unsafe"] + g["base_p_contro"]) > 0.5
        ft = g["p_strict"] > 0.5
        flips = g[zs & ~ft]
        out[src] = {"positives_en": len(g), "zs_caught": int(zs.sum()), "ft_caught": int(ft.sum()),
                    "flipped_to_missed": len(flips), "gained": int((~zs & ft).sum())}
        if not len(flips):
            continue
        sample = flips.head(max_rows)
        dist, idx = nn.kneighbors(vec.transform(sample["user_text"]))
        nb_labels = tr["label3"].to_numpy()[idx]
        nb_src = tr["source"].to_numpy()[idx]
        out[src]["neighbour_label_share"] = pd.Series(nb_labels.ravel()).value_counts(normalize=True).round(3).to_dict()
        out[src]["neighbour_source_share"] = pd.Series(nb_src.ravel()).value_counts(normalize=True).head(6).round(3).to_dict()
        out[src]["nearest_is_safe_and_close"] = float(((nb_labels[:, 0] == "Safe") & (dist[:, 0] < 0.5)).mean())
        for (_, r), d, i in zip(sample.iterrows(), dist, idx):
            rows.append({"source": src, "text": r["user_text"][:300], "zs_strict": r["base_p_unsafe"] + r["base_p_contro"],
                         "ft_strict": r["p_strict"], "ft_p_safe": r["p_safe"],
                         "nn1_label": tr.at[i[0], "label3"], "nn1_source": tr.at[i[0], "source"], "nn1_dist": d[0],
                         "nn1_text": tr.at[i[0], "user_text"][:300]})
    return {"summary": out, "examples": pd.DataFrame(rows)}

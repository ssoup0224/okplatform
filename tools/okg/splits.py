"""Leak-free role assignment.

* Every record belongs to a group (`seed_id`): translations, variants and every
  conversation synthesised from a seed inherit it, so a group never spans two roles.
  Group-wise splitting follows StratifiedGroupKFold [R11]; row-wise splitting leaked in
  the previous project (KoreanGuardrail 505 seeds x 11 variants).
* Splitting happens BEFORE multi-turn synthesis, and synthesis only combines material
  from the same role, so benign histories cannot leak across roles either. [D14]
* Exact duplicates of any evaluation text are removed from the training pools. [D15]
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import Config
from .schema import conv_is_unsafe, text_key

ROLES = ("guard_train", "guard_val", "stack", "test")


def _user_text_keys(messages_json: str) -> list[str]:
    return [text_key(m["content"]) for m in json.loads(messages_json) if m["role"] == "user"]


ORIGIN_RANK = {"human": 3, "model": 2, "template": 1, "derived": 0, "unset": -1}


def harmonize_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Across sources, the same single-turn text can appear several times (e.g. Aegis / Do-Not-Answer /
    HH-RLHF share prompts; WildGuardMix is partly WildJailbreak). [D43]
      * conflicting labels: keep the rows with the most trustworthy label_origin
        (human > model > template); if still conflicting, drop the text entirely;
      * seeds: all rows with the same text are merged into one group (union-find), so a duplicated
        text can never sit in train and test at once."""
    df = df.reset_index(drop=True)
    single = df["n_user_turns"].eq(1) & (df["pool"] != "history")
    not_test = df["fixed_role"].fillna("") != "test"             # evaluation sets are never altered
    key = pd.Series("", index=df.index)
    key[single] = df.loc[single, "messages"].map(lambda m: text_key(json.loads(m)[-1]["content"]))
    last = df["label3"].map(lambda s: json.loads(s)[-1])
    rank = df["label_origin"].map(ORIGIN_RANK).fillna(-1)
    drop = pd.Series(False, index=df.index)
    rep = {"texts_with_conflicting_labels": 0, "rows_dropped_conflict": 0}
    sub = pd.DataFrame({"k": key, "l": last, "r": rank})[single & not_test]
    for k, g in sub.groupby("k"):
        if g["l"].nunique() <= 1:
            continue
        rep["texts_with_conflicting_labels"] += 1
        best = g[g["r"] == g["r"].max()]
        if best["l"].nunique() > 1:
            drop[g.index] = True
        else:
            drop[g.index.difference(best.index)] = True
    rep["rows_dropped_conflict"] = int(drop.sum())
    df = df[~drop].reset_index(drop=True)
    key = key[~drop].reset_index(drop=True)
    # union-find over seeds that share a text
    parent: dict[str, str] = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x
    for k, g in df[key != ""].groupby(key[key != ""]):
        seeds = sorted(set(g["seed_id"]))
        for a in seeds[1:]:
            ra, rb = find(a), find(seeds[0])
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    merged = df["seed_id"].map(find)
    rep["seeds_merged"] = int((merged != df["seed_id"]).sum())
    df["seed_id"] = merged
    return df, rep


def assign_roles(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    df, report = harmonize_duplicates(df.copy())
    fixed = df["fixed_role"].fillna("") != ""

    # ---- 1. de-duplicate training pools against evaluation texts [D15]
    eval_keys = set()
    for m in df.loc[fixed & (df["fixed_role"] == "test"), "messages"]:
        eval_keys.update(_user_text_keys(m))
    eval_seeds = set(df.loc[fixed & (df["fixed_role"] == "test"), "seed_id"])
    free = df["fixed_role"].fillna("") != "test"          # every non-evaluation row
    clash = df.loc[free, "messages"].map(lambda m: any(k in eval_keys for k in _user_text_keys(m)))
    clash |= df.loc[free, "seed_id"].isin(eval_seeds)
    report["dropped_train_rows_overlapping_eval"] = int(clash.sum())
    df = df.drop(index=clash[clash].index).reset_index(drop=True)
    fixed = df["fixed_role"].fillna("") != ""

    # ---- 2. group-wise stratified folds for the free rows [R11]
    df["role"] = df["fixed_role"].where(fixed, "")
    free_idx = df.index[~fixed]
    if len(free_idx):
        y = df.loc[free_idx, "label3"].map(lambda s: conv_is_unsafe(json.loads(s))).to_numpy()
        groups = df.loc[free_idx, "seed_id"].to_numpy()
        n_groups = len(set(groups))
        n_splits = min(cfg.n_group_folds, n_groups)
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)  # [R11]
        fold = np.empty(len(free_idx), dtype=int)
        for k, (_, te) in enumerate(sgkf.split(np.zeros(len(free_idx)), y, groups)):
            fold[te] = k
        # map folds -> roles in proportion to cfg.role_folds
        weights = np.array([cfg.role_folds[r] for r in ROLES], dtype=float)
        bounds = np.cumsum(weights / weights.sum() * n_splits)
        fold_role = [ROLES[int(np.searchsorted(bounds, k + 0.5))] for k in range(n_splits)]
        df.loc[free_idx, "role"] = [fold_role[f] for f in fold]

    # ---- 2b. a seed group that contains a fixed guard_train row (e.g. PolyGuardMix, XGuard) is
    #          entirely guard_train, so merged duplicates cannot leak into stack/test [D43]
    gt_seeds = set(df.loc[df["fixed_role"] == "guard_train", "seed_id"])
    df.loc[df["seed_id"].isin(gt_seeds) & (df["fixed_role"] != "test"), "role"] = "guard_train"

    # ---- 3. held-out generator families never enter training [R19-style family holdout]
    if cfg.holdout_families:
        df.loc[df["family"].isin(cfg.holdout_families), "role"] = "test"

    # ---- 4. audit: a seed must live in exactly one role
    per_seed = df.groupby("seed_id")["role"].nunique()
    bad = per_seed[per_seed > 1]
    report["seeds_in_multiple_roles"] = int(len(bad))
    if len(bad):
        raise AssertionError(f"group leakage: {len(bad)} seeds span several roles, e.g. {list(bad.index[:5])}")
    report["rows_per_role"] = df["role"].value_counts().to_dict()
    return df, report

"""Causal trajectory features for the LightGBM stage (computed from turns 1..t only).

* risk statistics over the per-turn guard scores: max, mean, last, delta, slope, EWMA,
  count/consecutive above 0.5 — ordered trajectory information matters: mean-pooling turns
  loses signal [R38]; peak + accumulated risk is a proposed multi-turn score [R37];
  consecutive-above-threshold mirrors Qwen3Guard-Stream's debouncing [R2].
* base (zero-shot) guard scores as an independent side signal [D20].
* refusal-then-rephrase: previous assistant turn matches refusal strings (string matching
  as in XSTest / GCG refusal lists [R33][R55]) x character n-gram similarity between
  consecutive user turns (HashingVectorizer [R48]).
* script features for code-switching / obfuscation (Unicode classes) [R35][D13].
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from .obfuscate import dominant_script, script_stats

REFUSAL_RE = re.compile(
    r"(I can(?:no|')t|I cannot|I'm sorry|I am sorry|I apologi[sz]e|I won't|I will not|I'm not able|"
    r"I am not able|As an AI|I must decline|not able to (?:help|provide|assist)|"
    r"죄송|도와드릴 수 없|제공할 수 없|답변(?:드릴|할) 수 없|할 수 없습니다|어렵습니다|불가능합니다)",
    re.IGNORECASE)

_HV = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=2 ** 18,
                        alternate_sign=False, norm="l2")                                # [R48]


def _cos(a: list[str], b: list[str]) -> np.ndarray:
    A, B = _HV.transform(a), _HV.transform(b)
    return np.asarray(A.multiply(B).sum(axis=1)).ravel()                               # rows are L2-normalised


def _slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _ewma(y: np.ndarray, alpha: float) -> float:
    v = y[0]
    for z in y[1:]:
        v = alpha * z + (1 - alpha) * v
    return float(v)


FEATURE_COLUMNS: list[str] = []   # filled on first call


def build_features(df: pd.DataFrame, thr: float = 0.5) -> pd.DataFrame:
    """df: one row per prefix with p_unsafe, p_contro, cand_mass, base_p_unsafe, base_p_contro,
    user_text, prev_user_text, prev_assistant_text, turn_idx, conv_id."""
    df = df.sort_values(["conv_id", "turn_idx"]).reset_index(drop=True)
    sim = _cos(df["user_text"].fillna("").tolist(), df["prev_user_text"].fillna("").tolist())
    refusal = df["prev_assistant_text"].fillna("").map(lambda s: int(bool(REFUSAL_RE.search(s)))).to_numpy()
    feats = []
    for cid, g in df.groupby("conv_id", sort=False):
        p = g["p_unsafe"].to_numpy(float)
        pc = g["p_contro"].to_numpy(float)
        bp = g["base_p_unsafe"].to_numpy(float)
        scripts = [dominant_script(t) for t in g["user_text"].fillna("")]
        switches = 0
        for t, ix in enumerate(g.index):
            h = p[: t + 1]
            above = h > thr
            consec = 0
            for a in above[::-1]:
                if not a:
                    break
                consec += 1
            if t > 0 and scripts[t] != scripts[t - 1] and "other" not in (scripts[t], scripts[t - 1]):
                switches += 1
            ss = script_stats(str(g.at[ix, "user_text"]))
            prev_ss = script_stats(str(g.at[ix, "prev_user_text"] or "")) if t > 0 else ss
            feats.append({
                "prefix_id": g.at[ix, "prefix_id"],
                "turn_pos": t,
                "p_cur": h[-1], "p_contro_cur": pc[t], "p_strict_cur": h[-1] + pc[t], "cand_mass_cur": g.at[ix, "cand_mass"],
                "p_max": h.max(), "p_mean": h.mean(),
                "p_second_max": np.sort(h)[-2] if t > 0 else 0.0,
                "p_delta": h[-1] - h[-2] if t > 0 else 0.0,
                "p_slope3": _slope(h[-3:]),
                "p_ewma03": _ewma(h, 0.3), "p_ewma06": _ewma(h, 0.6),
                "n_above": int(above.sum()), "consec_above": consec,
                "peak_accum": float(h.max() + np.clip(h - thr, 0, None).sum()),                # [R37]
                "contro_max": pc[: t + 1].max(),
                "base_p_cur": bp[t], "base_p_max": bp[: t + 1].max(),
                "base_p_contro_cur": g["base_p_contro"].to_numpy(float)[t],
                # Shortcut guard [D26]: raw language identity (hangul/latin ratio), message length and
                # plain turn-to-turn similarity are NOT used — in synthesised data they separate the
                # generators rather than safe from unsafe (the previous project lost ~20 recall points
                # to a length shortcut). Only change/interaction signals are kept.
                "refusal_prev": refusal[ix],
                "refusal_then_rephrase": refusal[ix] * (sim[ix] if t > 0 else 0.0),
                "jamo_ratio": ss["jamo_ratio"],
                "zero_width_count": ss["zero_width_count"], "fullwidth_ratio": ss["fullwidth_ratio"],
                "nfkc_changed": ss["nfkc_changed"],
                "ko_share_delta": abs(ss["ko_share_of_letters"] - prev_ss["ko_share_of_letters"]),
                "lang_switches": switches,
            })
    out = pd.DataFrame(feats)
    global FEATURE_COLUMNS
    FEATURE_COLUMNS = [c for c in out.columns if c != "prefix_id"]
    return out


def feature_columns() -> list[str]:
    if not FEATURE_COLUMNS:
        build_features(pd.DataFrame({
            "conv_id": ["x"], "turn_idx": [0], "prefix_id": ["x#0"], "p_unsafe": [0.1], "p_contro": [0.1],
            "cand_mass": [1.0], "base_p_unsafe": [0.1], "base_p_contro": [0.0], "user_text": ["a"],
            "prev_user_text": [""], "prev_assistant_text": [""]}))
    return FEATURE_COLUMNS

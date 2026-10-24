"""Decision policy: graded actions, per-language thresholds, carried risk and the failure mode.

* Graded actions instead of one block/allow cut [D48]:
    block           score > block threshold (benign FPR budget, `fpr_budget`, with a confidence margin)
    safe_answer     score in the band down to the `action_safe_fpr` threshold: the chatbot answers with a
                    conservative safety prompt and the turn is queued for human review
    careful_answer  below both bands but P(Controversial) >= `action_careful_contro`: answer with care
    allow           everything else
  A single cut at 1% FPR forces a choice between missing attacks and refusing benign users; the middle
  band recovers recall without hard refusals.
* Thresholds are per language with a '_global' fallback [D47] (see stacker.thresholds_by_lang).
* Carried risk for the guard-only system: s_t = max(p_t, decay * s_{t-1}), so a high-risk setup keeps
  influencing later turns after the context window has dropped it [D44]. (The LightGBM path gets the
  same from its p_max / EWMA features over the cached per-turn scores.)
* Failure mode when the guard times out or errors [D49]: default "safe_answer" — the reply is produced
  under the conservative safety prompt and flagged, which keeps the product available while never
  serving an unguarded answer. "block" (strict fail-closed) and "allow" (fail-open) are configurable.
"""
from __future__ import annotations

import numpy as np

from .obfuscate import dominant_script
from .stacker import threshold_for

ACTIONS = ("allow", "careful_answer", "safe_answer", "block")
SEVERITY = {a: i for i, a in enumerate(ACTIONS)}
REVIEW_ACTIONS = ("safe_answer",)
FAIL_MODES = {"safe_answer": "safe_answer", "block": "block", "allow": "allow"}


def conversation_lang(user_texts: list[str]) -> str:
    """'ko' / 'en' / 'mixed', matching the dataset `lang` tags the thresholds were fitted on."""
    scripts = {dominant_script(t) for t in user_texts if str(t).strip()} - {"other"}
    if len(scripts) > 1:
        return "mixed"
    return scripts.pop() if scripts else "en"


def decide(score: float, p_contro: float, lang: str, thresholds: dict, cfg) -> str:
    """thresholds: {'block': float|{lang: tau}, 'safe_answer': float|{lang: tau}}."""
    if score > threshold_for(thresholds["block"], lang):
        return "block"
    safe = thresholds.get("safe_answer")
    if safe is not None and score > threshold_for(safe, lang):
        return "safe_answer"
    if p_contro >= cfg.action_careful_contro:
        return "careful_answer"
    return "allow"


def decide_many(scores, p_contro, langs, thresholds: dict, cfg) -> list[str]:
    return [decide(float(s), float(c), l, thresholds, cfg) for s, c, l in zip(scores, p_contro, langs)]


def carried(scores, decay: float = 0.8) -> np.ndarray:
    """s_t = max(p_t, decay * s_{t-1}); conversation max is unchanged, later turns keep earlier risk."""
    out, prev = [], 0.0
    for p in np.asarray(scores, float):
        prev = max(p, decay * prev)
        out.append(prev)
    return np.asarray(out)


def scaled_or(df, parts: list) -> np.ndarray:
    """OR of several thresholded scores as ONE score: max_i score_i / tau_i(lang) (langs=None: all langs).
    Its own threshold is then chosen on stack data, so the combination stays within the FPR budget. [D54]
    parts: [(column, threshold float|{lang: tau}, langs tuple|None), ...]"""
    out = np.zeros(len(df))
    lang = df["lang"].astype(str).to_numpy()
    for col, thr, langs in parts:
        tau = np.array([threshold_for(thr, l) for l in lang])
        v = df[col].to_numpy(float) / np.maximum(tau, 1e-9)
        if langs is not None:
            v = np.where(np.isin(lang, list(langs)), v, 0.0)
        out = np.maximum(out, np.nan_to_num(v))
    return out


def fail_action(cfg) -> str:
    if cfg.guard_fail_mode not in FAIL_MODES:
        raise ValueError(f"guard_fail_mode must be one of {list(FAIL_MODES)}")
    return FAIL_MODES[cfg.guard_fail_mode]


def action_table(ct_scores, thresholds: dict, cfg):
    """Conversation-level action counts for evaluation: ct_scores has y, score, contro, lang."""
    import pandas as pd
    acts = decide_many(ct_scores["score"], ct_scores.get("contro", np.zeros(len(ct_scores))),
                       ct_scores["lang"], thresholds, cfg)
    d = ct_scores.assign(action=acts)
    rows = []
    for y, g in d.groupby("y"):
        vc = g["action"].value_counts(normalize=True)
        rows.append({"y": int(y), "n": len(g), **{a: float(vc.get(a, 0.0)) for a in ACTIONS}})
    return pd.DataFrame(rows)

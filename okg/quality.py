"""Per-source quality filters and the data audit ("how each dataset looks").

Rules (each dropped row is counted by reason in the audit report) [D38]:
  empty/short      target user turn < 2 (ko) / 3 (en) non-space characters
  too_long         target user turn > 8,000 characters (extreme outliers, e.g. PolyGuardMix rows
                   up to 408k chars) — history turns are clipped later, not dropped
  repetition       translated rows: zlib compression ratio < 0.25 on texts > 200 chars, or one
                   20-character chunk repeated >= 4 times (machine-translation loops) [R61];
                   native rows: only extreme compression (< 0.15)
  script_mismatch  lang=ko but Hangul share of letters < 0.3 (untranslated / wrong language)
  mt_ratio         translated rows whose Korean/English length ratio is outside [0.25, 1.5]
                   (checked inside the parsers that have both texts; see data_sources.mt_ok)
  duplicate        identical normalised target text within the source (first kept)
  label_conflict   identical text with different labels within one source (all dropped)
"""
from __future__ import annotations

import json
import re
import zlib
from collections import Counter

import numpy as np
import pandas as pd

from .obfuscate import script_stats
from .schema import norm_text

MAX_CHARS = 8000


def is_repetitive(text: str) -> bool:
    t = text.strip()
    if len(t) > 200 and len(zlib.compress(t.encode("utf-8"))) / len(t.encode("utf-8")) < 0.25:   # [R61]
        return True
    if len(t) >= 80:
        chunks = Counter(t[i:i + 20] for i in range(0, len(t) - 20, 10))
        if chunks and chunks.most_common(1)[0][1] >= 4:
            return True
    return False


def mt_ok(ko: str, en: str) -> bool:
    """Translation sanity: non-empty, plausible KO/EN character-length ratio, no loops. [D38]"""
    if not ko or not str(ko).strip():
        return False
    if en and len(str(en).strip()) >= 20:
        r = len(str(ko)) / max(len(str(en)), 1)
        if not 0.25 <= r <= 1.5:
            return False
    return not is_repetitive(str(ko))


def _target(messages_json: str) -> str:
    msgs = json.loads(messages_json)
    return [m["content"] for m in msgs if m["role"] == "user"][-1]


def filter_source(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the rules above to one source's conversations; returns (kept, drop counts)."""
    if df.empty:
        return df, {}
    tgt = df["messages"].map(_target)
    reason = pd.Series("", index=df.index)
    nonspace = tgt.str.replace(r"\s+", "", regex=True).str.len()
    min_len = np.where(df["lang"] == "ko", 2, 3)
    reason[nonspace < min_len] = "empty_or_short"
    reason[(reason == "") & (tgt.str.len() > MAX_CHARS)] = "too_long"
    # loop detection targets machine translation; native text only fails on extreme compression
    rep = pd.Series([is_repetitive(t) if tr else (len(t) > 200 and len(zlib.compress(t.encode())) / len(t.encode()) < 0.15)
                     for t, tr in zip(tgt, df["translated"])], index=df.index)
    reason[(reason == "") & rep] = "repetition"
    ko_share = tgt.map(lambda t: script_stats(t)["ko_share_of_letters"])
    has_letters = tgt.map(lambda t: bool(re.search(r"[A-Za-z가-힣]", t)))
    reason[(reason == "") & (df["lang"] == "ko") & has_letters & (ko_share < 0.3)] = "script_mismatch"
    key = tgt.map(norm_text)
    ok = reason == ""
    # label conflicts within the source (same text, different label sequence) -> drop all copies
    lab = df["label3"]
    conflict_keys = set(pd.DataFrame({"k": key[ok], "l": lab[ok]}).groupby("k")["l"].nunique().loc[lambda s: s > 1].index)
    reason[ok & key.isin(conflict_keys)] = "label_conflict"
    ok = reason == ""
    dup = ok & key.duplicated() & df["n_user_turns"].eq(1)
    reason[dup] = "duplicate"
    counts = reason[reason != ""].value_counts().to_dict()
    return df[reason == ""].reset_index(drop=True), counts


def audit_source(df: pd.DataFrame, name: str, raw_rows: int, drops: dict, n_examples: int = 2) -> dict:
    """Numbers + examples that describe what a dataset looks like after conversion."""
    if df.empty:
        return {"source": name, "raw_rows": raw_rows, "kept": 0, "drops": drops}
    tgt = df["messages"].map(_target)
    last = df["label3"].map(lambda s: json.loads(s)[-1])
    cats = df["category"].map(lambda s: json.loads(s)[-1])
    ex = {}
    for l in ("Unsafe", "Controversial", "Safe", "Unknown"):
        t = tgt[last == l]
        if len(t):
            ex[l] = [s[:160] for s in t.sample(min(n_examples, len(t)), random_state=0)]
    return {
        "source": name, "raw_rows": int(raw_rows), "kept": int(len(df)), "drops": {k: int(v) for k, v in drops.items()},
        "lang": df["lang"].value_counts().to_dict(),
        "last_turn_label": last.value_counts().to_dict(),
        "last_turn_category": cats[last != "Safe"].value_counts().head(9).to_dict(),
        "multi_turn_share": round(float((df["n_user_turns"] > 1).mean()), 3),
        "mean_user_turns": round(float(df["n_user_turns"].mean()), 2),
        "target_chars_p50_p99": [int(tgt.str.len().median()), int(tgt.str.len().quantile(0.99))],
        "label_origin": df["label_origin"].value_counts().to_dict(),
        "translated_share": round(float(df["translated"].mean()), 3),
        "mt_model": df["mt_model"].value_counts().to_dict(),
        "license": df["license"].iloc[0], "commercial_ok": bool(df["commercial_ok"].all()),
        "examples": ex,
    }


def audit_markdown(audits: list[dict]) -> str:
    """Human-readable dataset report written next to the S01 manifest."""
    lines = ["# Dataset audit (generated by S01)", "",
             "| source | raw | kept | dropped (reason:n) | lang | last-turn labels | multi-turn | origin | translated | license |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for a in audits:
        if "lang" not in a:
            lines.append(f"| {a['source']} | {a.get('raw_rows', 0)} | 0 | {a.get('drops')} | | | | | | |")
            continue
        drops = ", ".join(f"{k}:{v}" for k, v in a["drops"].items()) or "–"
        lines.append(f"| {a['source']} | {a['raw_rows']} | {a['kept']} | {drops} | {a['lang']} | {a['last_turn_label']} | "
                     f"{a['multi_turn_share']} | {a['label_origin']} | {a['translated_share']} | "
                     f"{a['license']}{'' if a['commercial_ok'] else ' (NC/unknown)'} |")
    lines += ["", "## Examples (first 160 characters)", ""]
    for a in audits:
        for lab, xs in a.get("examples", {}).items():
            for x in xs:
                lines.append(f"- **{a['source']} / {lab}**: {x}")
    return "\n".join(lines) + "\n"

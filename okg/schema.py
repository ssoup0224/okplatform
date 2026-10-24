"""Unified conversation schema shared by every stage. [D2]

One row = one conversation.  Per-USER-turn labels follow the multi-turn formulation
"is the conversation unsafe as of user turn t" [R19]:
  * label3[t] in {"Safe","Unsafe","Controversial","Unknown"} — "Unknown" is an explicit sentinel,
    never a null (the stored table contains no nulls at all, see validate_frame) [D37]
  * category[t] in the Qwen3Guard categories, "None" (Safe turns) or "Unknown" (to be filled by
    the teacher before training) [D19]
  * t_attack = first user turn whose label is "Unsafe" (-1 for benign conversations)
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Iterable

import pandas as pd

COLUMNS = [
    "conv_id", "source", "family", "lang", "seed_id", "messages", "label3", "category",
    "t_attack", "n_user_turns", "latency_valid", "pool", "fixed_role", "license", "commercial_ok",
    "label_origin", "translated", "mt_model",
]
UNKNOWN, NO_CAT = "Unknown", "None"
LABELS = ("Safe", "Unsafe", "Controversial", UNKNOWN)
LABEL_ORIGINS = ("human", "model", "template", "derived", "unset")

LANGS = ("ko", "en", "mixed")


def norm_text(s: str) -> str:
    """NFKC + whitespace collapse, used only for de-duplication keys. [R35]"""
    s = unicodedata.normalize("NFKC", str(s))
    return " ".join(s.split()).lower()


def text_key(s: str) -> str:
    return hashlib.sha1(norm_text(s).encode("utf-8")).hexdigest()[:16]


def _norm_labels(label3: list, category: list) -> tuple[list, list]:
    """None -> explicit sentinels so that nothing null is ever stored. [D37]"""
    from .config import QG_CATEGORIES
    labs, cats = [], []
    for l, c in zip(label3, category):
        l = UNKNOWN if l is None else l
        if l not in LABELS:
            raise ValueError(f"bad label {l}")
        if l == "Safe":
            c = NO_CAT
        elif c is None or c == NO_CAT:
            c = UNKNOWN
        if c not in QG_CATEGORIES and c not in (NO_CAT, UNKNOWN):
            raise ValueError(f"bad category {c}")
        labs.append(l)
        cats.append(c)
    return labs, cats


def make_conv(*, source: str, messages: list[dict], label3: list, category: list,
              lang: str, seed_id: str, family: str, pool: str, license: str,
              commercial_ok: bool, fixed_role: str | None = None, latency_valid: bool = True,
              conv_id: str | None = None, label_origin: str = "unset", translated: bool = False,
              mt_model: str = "none") -> dict:
    """Build + validate one conversation record."""
    msgs = [{"role": m["role"], "content": str(m["content"])} for m in messages]
    n_user = sum(m["role"] == "user" for m in msgs)
    if n_user == 0:
        raise ValueError("conversation without a user turn")
    if len(label3) != n_user or len(category) != n_user:
        raise ValueError(f"labels ({len(label3)}) must match user turns ({n_user})")
    if msgs[-1]["role"] != "user" and pool != "history":
        # Records scored by the guard end on the user turn being moderated [R1].
        # Only benign 'history' material (used to build conversations) may end on an assistant turn.
        raise ValueError("conversations must end on a user turn (prompt moderation) [R1]")
    if any(m["role"] not in ("user", "assistant") for m in msgs):
        raise ValueError("roles must be user/assistant")
    if any(not m["content"].strip() for m in msgs):
        raise ValueError("empty message")
    label3, category = _norm_labels(list(label3), list(category))
    if lang not in LANGS:
        raise ValueError(f"bad lang {lang}")
    if label_origin not in LABEL_ORIGINS:
        raise ValueError(f"bad label_origin {label_origin}")
    t_attack = next((i for i, l in enumerate(label3) if l == "Unsafe"), -1)
    if conv_id is None:
        conv_id = f"{source}:{text_key(json.dumps(msgs, ensure_ascii=False))}"
    return {
        "conv_id": conv_id, "source": source, "family": family, "lang": lang, "seed_id": str(seed_id),
        "messages": json.dumps(msgs, ensure_ascii=False), "label3": json.dumps(label3),
        "category": json.dumps(category), "t_attack": int(t_attack), "n_user_turns": int(n_user),
        "latency_valid": bool(latency_valid), "pool": pool, "fixed_role": fixed_role or "",
        "license": license, "commercial_ok": bool(commercial_ok), "label_origin": label_origin,
        "translated": bool(translated), "mt_model": mt_model,
    }


def single_turn(source: str, text: str, label: str | None, cat: str | None, lang: str, seed_id: str,
                pool: str, license: str, commercial_ok: bool, family: str = "single",
                fixed_role: str | None = None, **kw) -> dict:
    return make_conv(source=source, messages=[{"role": "user", "content": text}], label3=[label],
                     category=[cat if label in ("Unsafe", "Controversial") else None], lang=lang,
                     seed_id=seed_id, family=family, pool=pool, license=license,
                     commercial_ok=commercial_ok, fixed_role=fixed_role, **kw)


def to_frame(rows: Iterable[dict]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows), columns=COLUMNS)
    return df.drop_duplicates("conv_id").reset_index(drop=True)


def validate_frame(df: pd.DataFrame, allow_unset_origin: bool = False) -> pd.DataFrame:
    """Hard checks that every stored table satisfies the unified schema: all columns present,
    no nulls anywhere (including inside the JSON lists), labels/categories in their enums,
    list lengths == number of user turns, t_attack consistent. Raises with examples. [D37]"""
    from .config import QG_CATEGORIES
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    nulls = df[COLUMNS].isna().sum()
    if nulls.any():
        raise ValueError(f"null values: {nulls[nulls > 0].to_dict()}")
    errs = []
    for r in df.itertuples(index=False):
        msgs, labs, cats = json.loads(r.messages), json.loads(r.label3), json.loads(r.category)
        n_user = sum(m.get("role") == "user" for m in msgs)
        if any(x is None for x in labs + cats) or any(m.get("content") in (None, "") for m in msgs):
            errs.append((r.conv_id, "null inside lists"))
        elif len(labs) != n_user or len(cats) != n_user or n_user != r.n_user_turns:
            errs.append((r.conv_id, "label/turn count mismatch"))
        elif any(l not in LABELS for l in labs) or any(c not in QG_CATEGORIES and c not in (NO_CAT, UNKNOWN) for c in cats):
            errs.append((r.conv_id, "label/category outside enum"))
        elif any((l == "Safe") != (c == NO_CAT) for l, c in zip(labs, cats)):
            errs.append((r.conv_id, "Safe<->None mismatch"))
        elif r.t_attack != next((i for i, l in enumerate(labs) if l == "Unsafe"), -1):
            errs.append((r.conv_id, "t_attack inconsistent"))
        elif not allow_unset_origin and r.label_origin == "unset":
            errs.append((r.conv_id, "label_origin unset"))
        elif r.lang not in LANGS:
            errs.append((r.conv_id, "bad lang"))
        if len(errs) >= 20:
            break
    if errs:
        raise ValueError(f"{len(errs)}+ invalid rows, e.g. {errs[:5]}")
    return df


def messages_of(row) -> list[dict]:
    return json.loads(row["messages"])


def labels_of(row) -> list:
    return json.loads(row["label3"])


def categories_of(row) -> list:
    return json.loads(row["category"])


def user_turn_indices(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if m["role"] == "user"]


def conv_is_unsafe(label3: list) -> int:
    return int(any(l == "Unsafe" for l in label3))

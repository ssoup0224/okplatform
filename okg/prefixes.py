"""Expand conversations into per-user-turn prefixes: the unit the guard scores.

Prefix t = messages up to and including user turn t (history + latest user message), the
multi-turn moderation setting of Qwen3Guard (whole dialogue as context, verdict on the
last user message) [R1][R3]. Scoring every prefix yields a per-turn risk trajectory [R19].
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

META = ["conv_id", "source", "family", "lang", "seed_id", "role", "t_attack", "n_user_turns", "latency_valid"]


def expand(convs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in convs.itertuples(index=False):
        msgs = json.loads(r.messages)
        labels = json.loads(r.label3)
        cats = json.loads(r.category)
        u = 0
        prev_assistant = ""
        prev_user = ""
        for i, m in enumerate(msgs):
            if m["role"] == "assistant":
                prev_assistant = m["content"]
                continue
            lab = labels[u]
            rows.append({
                **{k: getattr(r, k) for k in META},
                "turn_idx": u, "msg_end": i, "label3": lab, "category": cats[u],
                "y": (1.0 if lab == "Unsafe" else 0.0 if lab == "Safe" else np.nan),   # Controversial/Unknown -> excluded
                "user_text": m["content"], "prev_user_text": prev_user, "prev_assistant_text": prev_assistant,
            })
            prev_user = m["content"]
            u += 1
    df = pd.DataFrame(rows)
    df["prefix_id"] = df["conv_id"] + "#" + df["turn_idx"].astype(str)
    return df

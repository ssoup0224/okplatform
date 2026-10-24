"""SMOKE / DEMO ONLY: fake live traffic and fake reviewers so the feedback loop can be exercised
end to end without real users. Never use in production — real reviews come from people.

Traffic = conversations of the guard_val role (never used by stack/test, so no leakage).
Reviewer = the known dataset labels of those conversations.
"""
from __future__ import annotations

import json
import random

import pandas as pd

from . import feedback as fb


def simulate_traffic(guard, convs: pd.DataFrame, log: fb.FeedbackLog, n: int, seed: int = 0, days: int = 28):
    rng = random.Random(seed)
    sample = convs.sample(min(n, len(convs)), random_state=seed)
    for k, r in enumerate(sample.itertuples(index=False)):
        msgs = json.loads(r.messages)
        day = 1 + (k * days) // max(len(sample), 1)
        guard.assess(msgs, log=log, conv_id=r.conv_id, user_id=f"user{rng.randrange(8)}",
                     user_report=rng.random() < 0.05, timestamp=f"2026-09-{day:02d}T12:00:00")
    return log.flush()


def simulate_reviewers(queue: pd.DataFrame, convs: pd.DataFrame, double_review_share: float = 0.2,
                       seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    truth = convs.set_index("conv_id")
    rows = []
    for r in queue.itertuples(index=False):
        if r.conv_id not in truth.index:      # e.g. hand-typed demo conversations: left unreviewed
            continue
        labels = json.loads(truth.at[r.conv_id, "label3"])
        cats = json.loads(truth.at[r.conv_id, "category"])
        d = r._asdict()
        if "Unsafe" in labels:
            t = labels.index("Unsafe")
            from .config import QG_CATEGORIES
            d.update(verdict="unsafe", unsafe_from_turn=str(t + 1),
                     category=cats[t] if cats[t] in QG_CATEGORIES else "Unethical Acts")
        elif all(l == "Safe" for l in labels):
            d.update(verdict="safe", unsafe_from_turn="", category="")
        else:
            d.update(verdict="unsure", unsafe_from_turn="", category="")
        d["reviewer"] = "sim_a"
        rows.append(d)
        if rng.random() < double_review_share:
            rows.append(dict(d, reviewer="sim_b"))
    return pd.DataFrame(rows)

"""Feedback loop: log live assessments -> pick what humans should review -> import verdicts.

Principles (see REFERENCES.md):
* Nothing reaches training or memory without a human verdict (poisoning defence). [D30]
* Review budget goes to the most informative items: uncertain scores (uncertainty sampling
  [R57]), model disagreements (query-by-committee [R57]), user/downstream signals, plus a
  fixed random slice of ALLOWED traffic — the only unbiased way to find missed attacks. [D31]
* Per-user caps and near-duplicate removal limit how much one account can steer the data. [D30]
* Reviews are validated; double-reviewed items must agree; Cohen's kappa is reported [R58].
* Recent reviewed traffic is a time-based test set (evaluate on data newer than training). [D32]
* Logs are scrubbed of obvious personal data before storage (best effort, not a compliance
  guarantee) and user ids are salted SHA-256 hashes [R59][R60].
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .config import QG_CATEGORIES
from .obfuscate import dominant_script
from .schema import make_conv, text_key
from .utils import atomic_write_bytes, atomic_write_parquet

# --------------------------------------------------------------------------------------
# PII scrubbing (order matters: specific patterns first) [R59]
# --------------------------------------------------------------------------------------
_PII = [
    ("[EMAIL]", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("[RRN]", re.compile(r"(?<!\d)\d{6}\s?-\s?[1-8]\d{6}(?!\d)")),        # KR resident/foreigner reg. no. [R59]
    ("[CARD]", re.compile(r"(?<!\d)(?:\d{4}[- ]?){3}\d{4}(?!\d)")),
    ("[PHONE]", re.compile(r"(?<!\d)01[016789][- ]?\d{3,4}[- ]?\d{4}(?!\d)")),   # KR mobile
    ("[PHONE]", re.compile(r"(?<![\w])\+\d{1,3}[- ]?\(?\d{1,4}\)?(?:[- ]?\d{2,4}){2,3}(?!\d)")),
    ("[PHONE]", re.compile(r"(?<!\d)0\d{1,2}-\d{3,4}-\d{4}(?!\d)")),     # KR landline
    ("[IP]", re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")),
]


def scrub(text: str) -> str:
    for tag, rx in _PII:
        text = rx.sub(tag, text)
    return text


def hash_user(user_id: str) -> str:
    """Salted SHA-256 (salt from OKG_USER_SALT) so raw ids are never stored. [R60]"""
    salt = os.environ.get("OKG_USER_SALT", "okchat-default-salt-change-me")
    return hashlib.sha256((salt + str(user_id)).encode("utf-8")).hexdigest()[:20]


# ======================================================================================
# 1. logging
# ======================================================================================
LOG_COLUMNS = ["log_id", "conv_id", "turn_idx", "timestamp", "user_hash", "messages", "p_unsafe",
               "base_p_unsafe", "p_block", "block", "max_p_block_sofar", "threshold", "guard_version",
               "user_report", "downstream_refusal"]


class FeedbackLog:
    """Append-only log. Each flush writes one parquet part atomically; a crash loses at most
    the unflushed buffer, never corrupts stored parts. [R9][D1]"""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory) / "log"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.buf: list[dict] = []

    def record(self, conv_id: str, messages: list[dict], assessment: pd.DataFrame, *, user_id: str,
               guard_version: str, threshold: float, user_report: bool = False,
               downstream_refusal: bool = False, timestamp: str | None = None) -> None:
        """Log the newest user turn of an assessed conversation (full scrubbed history kept)."""
        last = assessment.iloc[-1]
        msgs = [{"role": m["role"], "content": scrub(str(m["content"]))} for m in messages]
        self.buf.append({
            "log_id": uuid.uuid4().hex, "conv_id": str(conv_id), "turn_idx": int(last["turn_idx"]),
            "timestamp": timestamp or time.strftime("%Y-%m-%dT%H:%M:%S"), "user_hash": hash_user(user_id),
            "messages": json.dumps(msgs, ensure_ascii=False), "p_unsafe": float(last["p_unsafe"]),
            "base_p_unsafe": float(last["base_p_unsafe"]), "p_block": float(last["p_block"]),
            "block": bool(last["block"]), "max_p_block_sofar": float(assessment["p_block"].max()),
            "threshold": float(threshold), "guard_version": guard_version,
            "user_report": bool(user_report), "downstream_refusal": bool(downstream_refusal)})

    def flush(self) -> Path | None:
        if not self.buf:
            return None
        p = self.dir / f"part-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}.parquet"
        atomic_write_parquet(p, pd.DataFrame(self.buf, columns=LOG_COLUMNS))
        self.buf = []
        return p

    def load(self) -> pd.DataFrame:
        parts = sorted(self.dir.glob("part-*.parquet"))
        if not parts:
            return pd.DataFrame(columns=LOG_COLUMNS)
        return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


# ======================================================================================
# 2. review queue
# ======================================================================================
REVIEW_FIELDS = ["verdict", "unsafe_from_turn", "category", "reviewer", "notes"]


def transcript(messages: list[dict]) -> str:
    lines, u = [], 0
    for m in messages:
        if m["role"] == "user":
            u += 1
            lines.append(f"[{u}] USER: {m['content']}")
        else:
            lines.append(f"    ASSISTANT: {m['content']}")
    return "\n".join(lines)


def build_review_queue(log: pd.DataFrame, cfg, already_reviewed: set[str] | None = None,
                       guard_threshold: float = 0.5) -> pd.DataFrame:
    """Latest state of each conversation -> prioritised, capped queue. [D31]"""
    if log.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(cfg.seed)
    df = (log.sort_values(["conv_id", "turn_idx", "timestamp"]).groupby("conv_id", as_index=False).tail(1)
             .reset_index(drop=True))
    if already_reviewed:
        df = df[~df["conv_id"].isin(already_reviewed)]
    last_user = df["messages"].map(lambda m: [x for x in json.loads(m) if x["role"] == "user"][-1]["content"])
    df = df.assign(_key=last_user.map(text_key)).drop_duplicates("_key")          # near-dup guard [D30]
    band = cfg.review_uncertainty_band
    df["signal"] = df["user_report"] | df["downstream_refusal"]
    df["disagree"] = ((df["base_p_unsafe"] > guard_threshold) != (df["p_unsafe"] > guard_threshold)) | \
                     ((df["p_unsafe"] > guard_threshold) != df["block"])
    df["uncertain"] = (df["max_p_block_sofar"] - df["threshold"]).abs() <= band
    budget = cfg.review_daily_budget
    n_random = int(round(budget * cfg.review_random_share))
    picked, used = [], set()

    def take(frame, n, bucket):
        frame = frame[~frame.index.isin(used)]
        frame = frame.groupby("user_hash", group_keys=False).head(cfg.review_max_per_user)   # per-user cap [D30]
        sel = frame.head(n)
        used.update(sel.index)
        picked.append(sel.assign(bucket=bucket))

    # fixed random slice of ALLOWED traffic first (unbiased miss-rate estimate) [D31]
    allowed = df[~df["block"]]
    take(allowed.sample(frac=1.0, random_state=int(rng.integers(1 << 31))), n_random, "random_allowed")
    rest = budget - n_random
    for bucket, frame in [("signal", df[df["signal"]]),
                          ("disagreement", df[df["disagree"]]),
                          ("uncertain", df[df["uncertain"]].assign(
                              _d=(df["max_p_block_sofar"] - df["threshold"]).abs()).sort_values("_d"))]:
        if rest <= 0:
            break
        before = len(used)
        take(frame, rest, bucket)
        rest -= len(used) - before
    q = pd.concat(picked, ignore_index=True) if picked else pd.DataFrame()
    if q.empty:
        return q
    # every reason that applies (the random slice is drawn first so it stays an unbiased sample
    # of allowed traffic; items it contains may also carry signal/disagreement flags) [D31]
    q["flags"] = [",".join(n for n, v in (("signal", a), ("disagreement", b), ("uncertain", c)) if v)
                  for a, b, c in zip(q["signal"], q["disagree"], q["uncertain"])]
    q["review_id"] = [f"r-{text_key(c + str(t))}" for c, t in zip(q["conv_id"], q["timestamp"])]
    q["transcript"] = q["messages"].map(lambda m: transcript(json.loads(m)))
    q["n_user_turns"] = q["messages"].map(lambda m: sum(x["role"] == "user" for x in json.loads(m)))
    for f in REVIEW_FIELDS:
        q[f] = ""
    cols = ["review_id", "conv_id", "log_id", "bucket", "flags", "n_user_turns", "transcript"] + REVIEW_FIELDS
    if cfg.review_show_model_decision:       # off by default to avoid anchoring reviewers [D33]
        cols.insert(4, "model_block")
        q["model_block"] = q["block"]
    return q[cols]


def write_queue_csv(queue: pd.DataFrame, path: str | Path) -> Path:
    """UTF-8 with BOM so Excel opens Korean text correctly; atomic write. [R9]"""
    buf = io.StringIO()
    queue.to_csv(buf, index=False)
    atomic_write_bytes(path, ("﻿" + buf.getvalue()).encode("utf-8"))
    return Path(path)


# ======================================================================================
# 3. import reviewed files
# ======================================================================================
def read_reviews(paths) -> pd.DataFrame:
    frames = []
    for p in paths:
        d = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        d["_file"] = Path(p).name
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def validate_reviews(rev: pd.DataFrame) -> list[str]:
    errs = []
    for i, r in rev.iterrows():
        where = f"{r.get('_file', '?')} row {i + 2} ({r.get('review_id', '?')})"
        v = str(r.get("verdict", "")).strip().lower()
        if v not in ("safe", "unsafe", "unsure", ""):
            errs.append(f"{where}: verdict must be safe/unsafe/unsure, got '{v}'")
        if v == "unsafe":
            try:
                t = int(float(r.get("unsafe_from_turn", "")))
                if not 1 <= t <= int(r.get("n_user_turns", 10 ** 6)):
                    errs.append(f"{where}: unsafe_from_turn {t} outside 1..{r.get('n_user_turns')}")
            except ValueError:
                errs.append(f"{where}: unsafe needs unsafe_from_turn (1-based user turn)")
            if r.get("category", "") not in QG_CATEGORIES:
                errs.append(f"{where}: category must be one of {list(QG_CATEGORIES)}")
        if v and not str(r.get("reviewer", "")).strip():
            errs.append(f"{where}: reviewer is empty")
    return errs


def resolve_reviews(rev: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One verdict per review_id. Double reviews must agree (verdict + turn); conflicts are
    dropped and counted; Cohen's kappa over double-reviewed verdicts [R58]."""
    from sklearn.metrics import cohen_kappa_score
    rev = rev[rev["verdict"].str.strip() != ""].copy()
    rev["verdict"] = rev["verdict"].str.strip().str.lower()
    rev["turn"] = pd.to_numeric(rev["unsafe_from_turn"], errors="coerce")
    rep = {"reviews": len(rev), "unsure": int((rev["verdict"] == "unsure").sum())}
    multi = rev.groupby("review_id").filter(lambda g: len(g) >= 2)
    if len(multi):
        pairs = multi.groupby("review_id").head(2).groupby("review_id")["verdict"].apply(list)
        a, b = zip(*[(x[0], x[1]) for x in pairs])
        rep["double_reviewed"] = len(pairs)
        rep["cohen_kappa"] = float(cohen_kappa_score(a, b)) if len(set(a) | set(b)) > 1 else 1.0
    keep, conflicts = [], 0
    for rid, g in rev.groupby("review_id"):
        sig = set(zip(g["verdict"], g["turn"].fillna(-1)))
        if len(sig) > 1:
            conflicts += 1
            continue
        if g["verdict"].iloc[0] == "unsure":
            continue
        keep.append(g.iloc[0])
    rep["conflicts_dropped"] = conflicts
    out = pd.DataFrame(keep).reset_index(drop=True) if keep else pd.DataFrame(columns=rev.columns)
    rep["accepted"] = len(out)
    return out, rep


def reviews_to_convs(accepted: pd.DataFrame, log: pd.DataFrame, cfg) -> tuple[list[dict], pd.DataFrame, dict]:
    """Accepted verdicts -> schema conversations (time-based roles) + memory additions."""
    logi = log.set_index("log_id")
    rows, mem, per_user = [], [], {}
    ts_all = pd.to_datetime(logi.loc[accepted["log_id"], "timestamp"]) if len(accepted) else pd.Series(dtype="datetime64[ns]")
    cutoff = (ts_all.max() - pd.Timedelta(days=cfg.feedback_test_days)) if len(ts_all) else None
    for r in accepted.sort_values("log_id").itertuples(index=False):
        lg = logi.loc[r.log_id]
        msgs = json.loads(lg["messages"])
        ts = pd.to_datetime(lg["timestamp"])
        role = "test" if ts >= cutoff else "stack"                                         # [D32]
        u = lg["user_hash"]
        if role != "test":
            per_user[u] = per_user.get(u, 0) + 1
            if per_user[u] > cfg.feedback_max_per_user:                                     # [D30]
                continue
        seed = f"feedback:{lg['conv_id']}"
        user_idx = [i for i, m in enumerate(msgs) if m["role"] == "user"]
        if r.verdict == "unsafe":
            k = int(float(r.unsafe_from_turn))
            msgs = msgs[: user_idx[k - 1] + 1]
            labels = ["Safe"] * (k - 1) + ["Unsafe"]
            cats = [None] * (k - 1) + [r.category]
            mem.append({"text": msgs[-1]["content"], "kind": "attack", "group": seed, "role": role,
                        "timestamp": str(lg["timestamp"])})
        else:
            n = len(user_idx)
            labels, cats = ["Safe"] * n, [None] * n
            if bool(lg["block"]):          # confirmed false positive -> benign memory [D27]
                mem.append({"text": msgs[user_idx[-1]]["content"], "kind": "benign", "group": seed,
                            "role": role, "timestamp": str(lg["timestamp"])})
        users = " ".join(m["content"] for m in msgs if m["role"] == "user")
        s = dominant_script(users)
        lang = s if s in ("ko", "en") else "mixed"
        rows.append(make_conv(source="feedback", messages=msgs, label3=labels, category=cats, lang=lang,
                              seed_id=seed, family="real_traffic", pool="feedback", license="own-traffic",
                              commercial_ok=True, fixed_role=role, conv_id=f"feedback:{r.review_id}",
                              label_origin="human"))
    report = {"convs": len(rows), "test": sum(c["fixed_role"] == "test" for c in rows),
              "stack": sum(c["fixed_role"] == "stack" for c in rows),
              "memory_attack": sum(m["kind"] == "attack" for m in mem),
              "memory_benign": sum(m["kind"] == "benign" for m in mem),
              "time_cutoff": str(cutoff) if cutoff is not None else None}
    return rows, pd.DataFrame(mem, columns=["text", "kind", "group", "role", "timestamp"]), report


def import_reviews(paths, log: pd.DataFrame, cfg):
    rev = read_reviews(paths)
    if rev.empty:
        return [], pd.DataFrame(columns=["text", "kind", "group", "role", "timestamp"]), {"reviews": 0}
    errs = validate_reviews(rev)
    if errs:
        raise ValueError("Review file problems (fix and re-run):\n  " + "\n  ".join(errs[:30]))
    missing = set(rev["log_id"]) - set(log["log_id"])
    if missing:
        raise ValueError(f"{len(missing)} reviewed log_id(s) not found in the feedback log, e.g. {list(missing)[:3]}")
    accepted, rep = resolve_reviews(rev)
    convs, mem, rep2 = reviews_to_convs(accepted, log, cfg)
    rep.update(rep2)
    return convs, mem, rep


def files_signature(paths) -> list:
    """Content hashes of the reviewed files -> part of the stage hash (new reviews re-run it)."""
    return sorted((Path(p).name, hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]) for p in paths)

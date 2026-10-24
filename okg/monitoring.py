"""Production monitoring from the feedback log, review files and the serving release. [D56]

* block / safe-answer rate by language and by hour (alert: an hour > `monitor_block_rate_alert` x the
  language's median hourly rate over the last 7 days, with at least 30 assessments in that hour)
* score distribution over time: daily population-stability index (PSI) of the logged serving score
  against the release's stack distribution (`score_reference.json`) [R83]; PSI > 0.2 is the usual
  "significant shift" rule of thumb
* review-queue backlog: queued items not yet reviewed, and the age of the oldest
* reviewer agreement: Cohen's kappa of double-reviewed items per week [R58]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def block_rates(log: pd.DataFrame) -> pd.DataFrame:
    if log.empty:
        return pd.DataFrame()
    d = log.assign(hour=pd.to_datetime(log["timestamp"]).dt.floor("h"),
                   lang=log["lang"].replace("", "unknown") if "lang" in log else "unknown")
    act = d["action"] if "action" in d else np.where(d["block"], "block", "allow")
    d = d.assign(is_block=(act == "block"), is_safe=(act == "safe_answer"),
                 is_error=d.get("guard_error", pd.Series("", index=d.index)).fillna("").ne(""))
    return (d.groupby(["lang", "hour"]).agg(n=("log_id", "size"), block_rate=("is_block", "mean"),
                                            safe_answer_rate=("is_safe", "mean"), guard_error_rate=("is_error", "mean"))
             .reset_index())


def psi(ref_share, cur_share, eps: float = 1e-4) -> float:
    p = np.clip(np.asarray(ref_share, float), eps, None)
    q = np.clip(np.asarray(cur_share, float), eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def score_drift(log: pd.DataFrame, reference: dict) -> pd.DataFrame:
    if log.empty or not reference:
        return pd.DataFrame()
    edges = np.asarray(reference["edges"], float)
    d = log.dropna(subset=["p_block"]).assign(day=pd.to_datetime(log["timestamp"]).dt.floor("D"))
    rows = []
    for day, g in d.groupby("day"):
        share = np.histogram(g["p_block"].to_numpy(float), bins=edges)[0] / max(len(g), 1)
        rows.append({"day": day, "n": len(g), "psi": psi(reference["share"], share),
                     "p50": float(g["p_block"].median()), "p95": float(g["p_block"].quantile(0.95))})
    return pd.DataFrame(rows)


def review_backlog(cfg) -> dict:
    qdir, rdir = cfg.feedback_dir / "queue", cfg.feedback_dir / "reviewed"
    q = [pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig") for p in sorted(qdir.glob("*.csv"))] \
        if qdir.exists() else []
    r = [pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig") for p in sorted(rdir.glob("*.csv"))] \
        if rdir.exists() else []
    if not q:
        return {"queued": 0, "pending": 0, "oldest_pending_days": 0.0}
    q = pd.concat(q, ignore_index=True)
    done = set()
    if r:
        r = pd.concat(r, ignore_index=True)
        done = set(r.loc[r.get("verdict", pd.Series("", index=r.index)).str.strip() != "", "review_id"])
    pend = q[~q["review_id"].isin(done)].drop_duplicates("review_id")
    age = 0.0
    if len(pend) and "queued_at" in pend:
        ts = pd.to_datetime(pend["queued_at"], errors="coerce").dropna()
        age = float((pd.Timestamp.now() - ts.min()).total_seconds() / 86400) if len(ts) else 0.0
    return {"queued": int(q["review_id"].nunique()), "pending": int(len(pend)), "oldest_pending_days": round(age, 2),
            "pending_by_bucket": pend["bucket"].value_counts().to_dict() if "bucket" in pend else {}}


def reviewer_agreement(cfg) -> pd.DataFrame:
    from sklearn.metrics import cohen_kappa_score
    from .feedback import read_reviews
    rev = read_reviews(cfg.review_files)
    if rev.empty:
        return pd.DataFrame()
    rev = rev[rev["verdict"].str.strip() != ""].copy()
    rev["verdict"] = rev["verdict"].str.strip().str.lower()
    ts = pd.to_datetime(rev["queued_at"], errors="coerce") if "queued_at" in rev else pd.Series(pd.NaT, index=rev.index)
    rev["week"] = ts.dt.to_period("W").astype(str).where(ts.notna(), rev["_file"])
    rows = []
    for wk, g in rev.groupby("week"):
        pairs = g.groupby("review_id").filter(lambda x: len(x) >= 2).groupby("review_id")["verdict"].apply(list)
        if len(pairs) == 0:
            continue
        a, b = zip(*[(x[0], x[1]) for x in pairs])
        k = float(cohen_kappa_score(a, b)) if len(set(a) | set(b)) > 1 else 1.0
        rows.append({"week": wk, "double_reviewed": len(pairs), "kappa": k,
                     "raw_agreement": float(np.mean([x == y for x, y in zip(a, b)]))})
    return pd.DataFrame(rows)


def report(cfg, log: pd.DataFrame, release_dir: str | Path | None = None) -> dict:
    """All monitoring tables + a list of alerts (strings)."""
    alerts = []
    br = block_rates(log)
    if len(br):
        recent = br[br["hour"] >= br["hour"].max() - pd.Timedelta(days=7)]
        med = recent.groupby("lang")["block_rate"].median()
        last = br[br["hour"] == br["hour"].max()]
        for r in last.itertuples(index=False):
            m = med.get(r.lang, np.nan)
            if r.n >= 30 and m > 0 and r.block_rate > cfg.monitor_block_rate_alert * m:
                alerts.append(f"block rate {r.lang} {r.hour}: {r.block_rate:.3f} > {cfg.monitor_block_rate_alert}x median {m:.3f}")
            if r.n >= 30 and r.guard_error_rate > 0.01:
                alerts.append(f"guard errors/timeouts {r.lang} {r.hour}: {r.guard_error_rate:.3f}")
    ref = {}
    if release_dir is not None and (Path(release_dir) / "score_reference.json").exists():
        ref = json.loads((Path(release_dir) / "score_reference.json").read_text())
    drift = score_drift(log, ref)
    if len(drift) and drift["psi"].iloc[-1] > cfg.monitor_psi_alert:
        alerts.append(f"score drift: PSI {drift['psi'].iloc[-1]:.3f} on {drift['day'].iloc[-1]:%Y-%m-%d}")
    backlog = review_backlog(cfg)
    if backlog["oldest_pending_days"] > cfg.monitor_backlog_alert_days:
        alerts.append(f"review backlog: {backlog['pending']} pending, oldest {backlog['oldest_pending_days']} days")
    agree = reviewer_agreement(cfg)
    if len(agree) and agree["kappa"].iloc[-1] < cfg.monitor_kappa_alert:
        alerts.append(f"reviewer agreement: kappa {agree['kappa'].iloc[-1]:.2f} in {agree['week'].iloc[-1]}")
    return {"block_rates": br, "score_drift": drift, "backlog": backlog, "reviewer_agreement": agree, "alerts": alerts}

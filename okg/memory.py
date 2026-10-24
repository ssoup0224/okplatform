"""Attack / benign memory: a nearest-neighbour store of reviewer-confirmed turns.

* Confirmed attack turns and confirmed false positives (benign turns the system blocked)
  are embedded and stored. Similarity of a new user turn to the store becomes LightGBM
  features, so a newly confirmed jailbreak template affects decisions within minutes and a
  wrong entry can be deactivated instantly — no weight update involved. [D27]
* Encoder (full mode): intfloat/multilingual-e5-small — average pooling over the last hidden
  state, "query: " prefix for symmetric similarity, L2-normalised; its cosine scores sit around
  0.7-1.0 by design, so only relative values matter [R56]. Smoke mode uses a stateless
  character n-gram HashingVectorizer [R48] so everything runs offline.
* Leakage guard: a query never matches entries from its own seed group (group-excluded
  similarity), otherwise every stored attack would match itself at similarity 1.0. [D28]
* It is a FEATURE, not a blocklist: a raw similarity cut would block benign messages that share
  wording with attacks (the surface-cue problem, [D26]); LightGBM learns how much to trust it.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import text_key
from .utils import atomic_write_parquet

KINDS = ("attack", "benign")
FEATURES = ["mem_attack_max", "mem_attack_top3", "mem_attack_n_close", "mem_benign_max",
            "mem_margin", "mem_attack_max_sofar"]


# ======================================================================================
# encoders
# ======================================================================================
class HashingEncoder:
    """Offline encoder: char_wb 2-4 grams, L2-normalised, dense float32. [R48]"""
    name = "hashing"
    close_sim = 0.5          # [D29] cut-off for mem_attack_n_close; validate on your data

    def __init__(self, n_features: int = 2 ** 14):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.v = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=n_features,
                                   alternate_sign=False, norm="l2")

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.v.transform([str(t) for t in texts]).astype(np.float32).toarray()


class E5Encoder:
    """intfloat/multilingual-e5-small usage from its model card [R56]."""
    name = "e5"
    close_sim = 0.90         # e5 cosine scores cluster in 0.7-1.0 [R56]; [D29]

    def __init__(self, model_id: str = "intfloat/multilingual-e5-small", token=None, batch_size: int = 64,
                 max_length: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id, token=token)
        self.model = AutoModel.from_pretrained(model_id, token=token)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.dev).eval()
        self.bs, self.max_length = batch_size, max_length

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch
        out = []
        for i in range(0, len(texts), self.bs):
            batch = ["query: " + str(t) for t in texts[i:i + self.bs]]                      # [R56]
            enc = self.tok(batch, max_length=self.max_length, padding=True, truncation=True,
                           return_tensors="pt").to(self.dev)
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state
            m = enc["attention_mask"][..., None].bool()
            emb = h.masked_fill(~m, 0.0).sum(1) / enc["attention_mask"].sum(1)[..., None]   # average_pool [R56]
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.append(emb.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 384), np.float32)


def make_encoder(cfg):
    if cfg.memory_encoder == "hashing" or cfg.run_mode == "smoke":
        return HashingEncoder()
    return E5Encoder(cfg.memory_encoder_model, token=cfg.hf_token)


# ======================================================================================
# memory store
# ======================================================================================
class Memory:
    """entries: text, kind (attack|benign), group (seed_id), source, added_at, active."""

    def __init__(self, encoder, entries: pd.DataFrame | None = None, vectors: np.ndarray | None = None):
        self.enc = encoder
        cols = ["entry_id", "text", "kind", "group", "source", "added_at", "active"]
        self.entries = entries.reset_index(drop=True) if entries is not None else pd.DataFrame(columns=cols)
        self.vectors = vectors if vectors is not None else np.zeros((0, 1), np.float32)

    # ---- editing ------------------------------------------------------------------
    def add(self, texts: list[str], kind: str, groups: list[str], source: str) -> list[str]:
        assert kind in KINDS
        if not texts:
            return []
        vec = self.enc.encode(texts)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        start = len(self.entries)
        ids = [f"{kind}-{start + i:07d}-{text_key(str(g) + chr(31) + str(t))[:8]}" for i, (t, g) in enumerate(zip(texts, groups))]
        new = pd.DataFrame({"entry_id": ids, "text": texts, "kind": kind, "group": groups, "source": source,
                            "added_at": now, "active": True})
        self.entries = pd.concat([self.entries, new], ignore_index=True)
        self.vectors = vec if self.vectors.shape[0] == 0 else np.vstack([self.vectors, vec])
        return ids

    def deactivate(self, entry_ids: list[str]) -> int:
        """Instant undo of a wrong confirmation (kept for audit, ignored in queries)."""
        m = self.entries["entry_id"].isin(entry_ids)
        self.entries.loc[m, "active"] = False
        return int(m.sum())

    # ---- persistence (atomic) ----------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(d / "entries.parquet", self.entries)
        tmp = d / f"vectors.tmp{os.getpid()}.npy"
        np.save(tmp, self.vectors)
        os.replace(tmp, d / "vectors.npy")                                                  # [R9]

    @classmethod
    def load(cls, directory: str | Path, encoder) -> "Memory":
        d = Path(directory)
        return cls(encoder, pd.read_parquet(d / "entries.parquet"), np.load(d / "vectors.npy"))

    # ---- querying ------------------------------------------------------------------
    def similarities(self, texts: list[str], groups: list[str], chunk: int = 2048) -> pd.DataFrame:
        """Group-excluded similarity features for each query text. [D28]"""
        n = len(texts)
        res = {k: np.zeros(n, np.float32) for k in
               ("mem_attack_max", "mem_attack_top3", "mem_attack_n_close", "mem_benign_max")}
        act = self.entries["active"].to_numpy(bool) if len(self.entries) else np.zeros(0, bool)
        if n == 0 or not act.any():
            out = pd.DataFrame(res)
            out["mem_margin"] = 0.0
            return out
        kinds = self.entries["kind"].to_numpy()
        grp = self.entries["group"].to_numpy()
        for s in range(0, n, chunk):
            q = self.enc.encode(texts[s:s + chunk])
            sim = q @ self.vectors.T                                    # cosine (both L2-normalised)
            qg = np.asarray(groups[s:s + chunk])[:, None]
            sim = np.where((grp[None, :] == qg) | ~act[None, :], -1.0, sim)
            for kind in KINDS:
                cols = kinds == kind
                if not cols.any():
                    continue
                sk = sim[:, cols]
                if kind == "attack":
                    top = -np.sort(-sk, axis=1)[:, :3]
                    res["mem_attack_max"][s:s + chunk] = np.clip(top[:, 0], 0, None)
                    res["mem_attack_top3"][s:s + chunk] = np.clip(top, 0, None).mean(1)
                    res["mem_attack_n_close"][s:s + chunk] = (sk >= self.enc.close_sim).sum(1)
                else:
                    res["mem_benign_max"][s:s + chunk] = np.clip(sk.max(1), 0, None)
        out = pd.DataFrame(res)
        out["mem_margin"] = out["mem_attack_max"] - out["mem_benign_max"]
        return out


def memory_features(rows: pd.DataFrame, mem: Memory) -> pd.DataFrame:
    """rows: prefix_id, conv_id, turn_idx, user_text, seed_id. Adds the causal running max over
    the conversation (an attack spread over several turns keeps its strongest match). [D27]"""
    rows = rows[["prefix_id", "conv_id", "turn_idx", "user_text", "seed_id"]].reset_index(drop=True)
    f = mem.similarities(rows["user_text"].fillna("").tolist(), rows["seed_id"].astype(str).tolist())
    f.insert(0, "prefix_id", rows["prefix_id"].to_numpy())
    f["conv_id"], f["turn_idx"] = rows["conv_id"].to_numpy(), rows["turn_idx"].to_numpy()
    f = f.sort_values(["conv_id", "turn_idx"])
    f["mem_attack_max_sofar"] = f.groupby("conv_id")["mem_attack_max"].cummax()
    return f.drop(columns=["conv_id", "turn_idx"]).reset_index(drop=True)

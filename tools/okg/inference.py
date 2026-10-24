"""Deployment-style wrapper: score a live conversation with the full okchat_guardrail stack.

guard (base + LoRA) per-turn scores -> causal trajectory features (+ attack/benign memory
features) -> calibrated LightGBM fold ensemble -> threshold chosen for the FPR budget.
Uses exactly the functions of S07/S07b/S08 (render_prompt, fit_messages, score_safety,
build_features, memory_features). [D24] Optionally logs every assessment for review. [D30]
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as fe
from . import guard as G


class OkChatGuard:
    def __init__(self, cfg, adapter_dir, stacker_pkl, thresholds_json, memory_dirs=(), version: str = "dev"):
        from peft import PeftModel
        from . import memory as mm
        self.cfg, self.version = cfg, version
        path = G.resolve_model_path(cfg)
        self.tok = G.load_tokenizer(path, cfg.hf_token)
        base = G.load_model(path, cfg, cfg.hf_token)
        self.model = PeftModel.from_pretrained(base, adapter_dir)                                  # [R6]
        self.model.eval()
        with open(stacker_pkl, "rb") as f:
            self.stacker = pickle.load(f)
        meta = json.loads(Path(thresholds_json).read_text())
        self.threshold = meta["thresholds"]["okchat_lgb"]
        self.guard_threshold = meta["thresholds"]["guard_finetuned"]
        self.cols = meta["feature_columns"]
        self.memory = None
        if meta.get("use_attack_memory"):
            enc = mm.make_encoder(cfg)
            mems = [mm.Memory.load(d, enc) for d in memory_dirs if (Path(d) / "entries.parquet").exists()]
            if not mems:
                raise FileNotFoundError("stacker expects memory features but no memory was found")
            m = mems[0]
            for extra in mems[1:]:                       # e.g. release memory + live confirmations
                m.entries = pd.concat([m.entries, extra.entries], ignore_index=True)
                m.vectors = extra.vectors if m.vectors.shape[0] == 0 else np.vstack([m.vectors, extra.vectors])
            if m.vectors.shape[1] != enc.encode(["x"]).shape[1]:
                raise ValueError("memory was built with a different encoder than cfg.memory_encoder")
            self.memory = m

    # ---- constructors -------------------------------------------------------------
    @classmethod
    def from_artifacts(cls, cfg, artifacts_dir=None):
        art = Path(artifacts_dir or cfg.artifacts)
        return cls(cfg, art / "S06_sft" / "adapter", art / "S08_stacker" / "fold_ensemble.pkl",
                   art / "S08_stacker" / "thresholds.json", [art / "S07b_memory" / "memory_eval"], "artifacts")

    @classmethod
    def from_release(cls, cfg, name: str | None = None, include_live_memory: bool = True):
        from .release import current_name
        name = name or current_name(cfg)
        if name is None:
            raise RuntimeError("no CURRENT release")
        d = cfg.releases_dir / name
        mems = [d / "memory"] + ([cfg.feedback_dir / "memory_live"] if include_live_memory else [])
        return cls(cfg, d / "adapter", d / "fold_ensemble.pkl", d / "thresholds.json", mems, name)

    # ---- scoring ------------------------------------------------------------------
    def assess(self, messages: list[dict], *, log=None, conv_id: str | None = None, user_id: str = "anonymous",
               user_report: bool = False, downstream_refusal: bool = False, timestamp: str | None = None) -> pd.DataFrame:
        """messages: [{'role': 'user'|'assistant', 'content': str}, ...] ending with a user turn.
        Returns one row per user turn with guard probabilities and the final decision."""
        from . import memory as mm
        rows, prompts = [], []
        prev_user, prev_asst, u = "", "", 0
        for i, m in enumerate(messages):
            if m["role"] == "assistant":
                prev_asst = m["content"]
                continue
            msgs, _ = G.fit_messages(self.tok, messages[: i + 1], self.cfg.max_seq_len - 16)
            prompts.append(G.render_prompt(self.tok, msgs))
            rows.append({"conv_id": "live", "seed_id": "live", "turn_idx": u, "prefix_id": f"live#{u}",
                         "user_text": m["content"], "prev_user_text": prev_user, "prev_assistant_text": prev_asst})
            prev_user, u = m["content"], u + 1
        df = pd.DataFrame(rows)
        tuned = G.score_safety(self.model, self.tok, prompts, self.cfg.score_batch_size)
        with self.model.disable_adapter():                                                          # [R6]
            base = G.score_safety(self.model, self.tok, prompts, self.cfg.score_batch_size)
        df = pd.concat([df, tuned], axis=1)
        df["base_p_unsafe"], df["base_p_contro"] = base["p_unsafe"].to_numpy(), base["p_contro"].to_numpy()
        df = df.merge(fe.build_features(df), on="prefix_id")
        if self.memory is not None:
            df = df.merge(mm.memory_features(df, self.memory), on="prefix_id")
        df = df.sort_values("turn_idx").reset_index(drop=True)
        df["p_block"] = self.stacker.predict(df[self.cols])
        df["block"] = df["p_block"] > self.threshold
        keep = ["turn_idx", "user_text", "p_safe", "p_unsafe", "p_contro", "base_p_unsafe", "p_block", "block"]
        keep += [c for c in ("mem_attack_max", "mem_benign_max") if c in df.columns]
        out = df[keep]
        if log is not None:
            log.record(conv_id or "unknown", messages, out, user_id=user_id, guard_version=self.version,
                       threshold=self.threshold, user_report=user_report,
                       downstream_refusal=downstream_refusal, timestamp=timestamp)
        return out

"""Deployment-style wrapper: score a live conversation with the okchat_guardrail release.

Per user turn: fine-tuned guard scores over the (pinned, chunked) context window -> serving score ->
graded action. The serving score is either
  * "stack": causal trajectory features (+ memory, + hate classifier) -> calibrated LightGBM ensemble, or
  * "guard": P(Unsafe)+P(Controversial) OR the Korean hate classifier, carried across turns,
whichever S10 recommended (`decision_system`) [D46]. Uses the same functions as S07/S07b/S08
(turn_prompts, score_prompt_groups, build_features, memory_features, hate_features) [D24].

Serving cost [D45]: each user turn is scored ONCE (per-turn cache keyed by the conversation prefix), and
there is no base-model pass (its LightGBM features were dropped), so a 20-turn conversation costs ~20
guard passes over its life instead of ~400.

`assess_response` is the second check on the assistant's draft reply (response moderation) [D51].
Timeouts and errors fall back to `cfg.guard_fail_mode` [D49]. Optionally logs every assessment. [D30]
"""
from __future__ import annotations

import json
import pickle
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as fe
from . import guard as G
from . import policy as pol
from .cache import LRUCache, chain_keys, text_key
from .stacker import threshold_for


class OkChatGuard:
    def __init__(self, cfg, adapter_dir, stacker_pkl, thresholds_json, memory_dirs=(), version: str = "dev",
                 hate_dir=None, decision_system: str | None = None):
        from peft import PeftModel
        self.cfg, self.version = cfg, version
        path = G.resolve_model_path(cfg)
        self.tok = G.load_tokenizer(path, cfg.hf_token)
        base = G.load_model(path, cfg, cfg.hf_token)
        self.model = PeftModel.from_pretrained(base, adapter_dir)                                  # [R6]
        self.model.eval()
        self.meta = json.loads(Path(thresholds_json).read_text())
        thr = self.meta["thresholds"]
        ds = decision_system or (cfg.decision_system if cfg.decision_system != "auto" else "stack")
        self.decision_system = ds
        self.use_base = bool(self.meta.get("use_base_features", True))     # releases before D45 used base features
        self.hate = None
        or_parts = self.meta.get("guard_hate_or")
        if hate_dir is not None and Path(hate_dir, "hate_meta.json").exists():
            from . import hate as hz
            self.hate = hz.load(hate_dir)
        if ds == "guard":
            self.system = "guard_strict_hate" if (or_parts and self.hate is not None and "guard_strict_hate" in thr) \
                else "guard_strict"
        else:
            self.system = "okchat_lgb"
        self.thresholds = {"block": thr[self.system],
                           "safe_answer": self.meta.get("safe_answer_thresholds", {}).get(self.system)}
        self.threshold = threshold_for(thr[self.system], "_global")                # logged; per-language in use
        self.guard_threshold = threshold_for(thr["guard_finetuned"], "_global")
        self.response_threshold = self.meta.get("response_threshold") or {"_global": 0.5}
        self.stacker, self.cols, self.memory = None, self.meta.get("feature_columns", []), None
        if ds == "stack":
            with open(stacker_pkl, "rb") as f:
                self.stacker = pickle.load(f)
            if self.meta.get("use_attack_memory"):
                self.memory = self._load_memory(memory_dirs)
        self.cache = LRUCache(cfg.score_cache_size)
        self.latency_ms: deque = deque(maxlen=10000)       # (total ms, newly scored turns) per assess call
        self._pool = ThreadPoolExecutor(max_workers=1) if cfg.guard_timeout_ms and cfg.guard_timeout_ms > 0 else None

    def _load_memory(self, memory_dirs):
        from . import memory as mm
        enc = mm.make_encoder(self.cfg)
        mems = [mm.Memory.load(d, enc) for d in memory_dirs if (Path(d) / "entries.parquet").exists()]
        if not mems:
            raise FileNotFoundError("stacker expects memory features but no memory was found")
        m = mems[0]
        for extra in mems[1:]:                       # e.g. release memory + live confirmations
            m.entries = pd.concat([m.entries, extra.entries], ignore_index=True)
            m.vectors = extra.vectors if m.vectors.shape[0] == 0 else np.vstack([m.vectors, extra.vectors])
        if m.vectors.shape[1] != enc.encode(["x"]).shape[1]:
            raise ValueError("memory was built with a different encoder than cfg.memory_encoder")
        return m

    # ---- constructors -------------------------------------------------------------
    @classmethod
    def from_artifacts(cls, cfg, artifacts_dir=None):
        art = Path(artifacts_dir or cfg.artifacts)
        summ = art / "S10_eval" / "summary.json"
        ds = json.loads(summ.read_text()).get("decision_system") if summ.exists() else None
        return cls(cfg, art / "S06_sft" / "adapter", art / "S08_stacker" / "fold_ensemble.pkl",
                   art / "S08_stacker" / "thresholds.json", [art / "S07b_memory" / "memory_eval"], "artifacts",
                   hate_dir=art / "S06h_hate" / "model", decision_system=ds)

    @classmethod
    def from_release(cls, cfg, name: str | None = None, include_live_memory: bool = True):
        from .release import current_name
        name = name or current_name(cfg)
        if name is None:
            raise RuntimeError("no CURRENT release")
        d = cfg.releases_dir / name
        info = json.loads((d / "release.json").read_text())
        mems = [d / "memory"] + ([cfg.feedback_dir / "memory_live"] if include_live_memory else [])
        return cls(cfg, d / "adapter", d / "fold_ensemble.pkl", d / "thresholds.json", mems, name,
                   hate_dir=d / "hate", decision_system=info.get("decision_system"))

    @classmethod
    def from_release_dir(cls, cfg, d: Path, decision_system: str | None = None):
        """A packaged (possibly not yet promoted) release folder, e.g. for the red-team gate."""
        d = Path(d)
        info = json.loads((d / "release.json").read_text()) if (d / "release.json").exists() else {}
        return cls(cfg, d / "adapter", d / "fold_ensemble.pkl", d / "thresholds.json", [d / "memory"], d.name,
                   hate_dir=d / "hate", decision_system=decision_system or info.get("decision_system"))

    # ---- per-turn guard scores (cached) ----------------------------------------------
    def _guard_scores(self, messages: list[dict]) -> tuple[list[int], list[dict], int]:
        users = [i for i, m in enumerate(messages) if m["role"] == "user"]
        keys = chain_keys(messages, f"guard:{self.version}")
        vals = [self.cache.get(keys[i]) for i in users]
        todo = [(u, i) for u, (i, v) in enumerate(zip(users, vals)) if v is None]
        if todo:
            groups = [G.turn_prompts(self.tok, messages[: i + 1], self.cfg.max_seq_len - 16, self.cfg) for _, i in todo]
            sc = G.score_prompt_groups(self.model, self.tok, groups, self.cfg.score_batch_size)
            base = None
            if self.use_base:                      # only for releases trained with base features (pre-D45)
                with self.model.disable_adapter():                                                # [R6]
                    base = G.score_prompt_groups(self.model, self.tok, groups, self.cfg.score_batch_size)
            for j, (u, i) in enumerate(todo):
                v = {c: float(sc.at[j, c]) for c in ("p_safe", "p_unsafe", "p_contro", "cand_mass")}
                v["n_prompts"] = int(sc.at[j, "n_prompts"])
                if base is not None:
                    v["base_p_unsafe"], v["base_p_contro"] = float(base.at[j, "p_unsafe"]), float(base.at[j, "p_contro"])
                self.cache.put(keys[i], v)
                vals[u] = v
        return users, vals, len(todo)

    def _text_scores(self, ns: str, texts: list[str], fn) -> list:
        keys = [text_key(ns, t) for t in texts]
        out = [self.cache.get(k) for k in keys]
        miss = [j for j, v in enumerate(out) if v is None]
        if miss:
            new = fn([texts[j] for j in miss])
            for j, v in zip(miss, new):
                self.cache.put(keys[j], v)
                out[j] = v
        return out

    def _memory_feats(self, df: pd.DataFrame) -> pd.DataFrame:
        from . import memory as mm
        m = self.memory
        ns = f"mem:{len(m.entries)}:{int(m.entries['active'].sum()) if len(m.entries) else 0}"
        texts = df["user_text"].fillna("").astype(str).tolist()
        rows = self._text_scores(ns, texts, lambda t: m.similarities(t, ["live"] * len(t)).to_dict("records"))
        f = pd.DataFrame(rows)
        f.insert(0, "prefix_id", df["prefix_id"].to_numpy())
        f["mem_attack_max_sofar"] = f["mem_attack_max"].cummax()                     # one conversation, in order
        return f[["prefix_id"] + mm.FEATURES]

    def _hate_feats(self, df: pd.DataFrame) -> pd.DataFrame:
        texts = df["user_text"].fillna("").astype(str).tolist()
        p = self._text_scores("hate", texts, lambda t: [float(x) for x in self.hate.predict_proba(t)])
        return pd.DataFrame({"prefix_id": df["prefix_id"].to_numpy(), "hate_p_cur": p,
                             "hate_p_max": np.maximum.accumulate(np.asarray(p, float)) if p else []})

    # ---- scoring ------------------------------------------------------------------
    def _assess_core(self, messages: list[dict]) -> tuple[pd.DataFrame, int]:
        users, vals, n_new = self._guard_scores(messages)
        rows, prev_user, prev_asst, u_texts = [], "", "", []
        for u, i in enumerate(users):
            prev_asst = next((m["content"] for m in reversed(messages[:i]) if m["role"] == "assistant"), "")
            u_texts.append(messages[i]["content"])
            rows.append({"conv_id": "live", "seed_id": "live", "turn_idx": u, "prefix_id": f"live#{u}",
                         "user_text": messages[i]["content"], "prev_user_text": prev_user,
                         "prev_assistant_text": prev_asst, "lang": pol.conversation_lang(u_texts), **vals[u]})
            prev_user = messages[i]["content"]
        df = pd.DataFrame(rows)
        if self.decision_system == "stack":
            df = df.merge(fe.build_features(df, use_base=self.use_base), on="prefix_id")
            if self.memory is not None:
                df = df.merge(self._memory_feats(df), on="prefix_id")
            if self.hate is not None and "hate_p_cur" in self.cols:
                df = df.merge(self._hate_feats(df), on="prefix_id")
            df = df.sort_values("turn_idx").reset_index(drop=True)
            df["p_block"] = self.stacker.predict(df[self.cols])
        else:
            df["p_strict"] = df["p_unsafe"] + self.meta.get("strict_lambda", 1.0) * df["p_contro"]
            col = "p_strict"
            if self.system == "guard_strict_hate":
                df = df.merge(self._hate_feats(df), on="prefix_id")
                o = self.meta["guard_hate_or"]
                df["guard_hate"] = pol.scaled_or(df, [("p_strict", o["guard_strict"], None),
                                                      ("hate_p_cur", o["hate_clf"], tuple(o["hate_langs"]))])
                col = "guard_hate"
            df["p_block"] = pol.carried(df[col].to_numpy(), self.cfg.guard_carry_decay)
        df["action"] = pol.decide_many(df["p_block"], df["p_contro"], df["lang"], self.thresholds, self.cfg)
        df["block"] = df["action"] == "block"
        df["threshold"] = [threshold_for(self.thresholds["block"], l) for l in df["lang"]]
        df["guard_error"] = ""
        return df, n_new

    def _fallback(self, messages: list[dict], err: str) -> pd.DataFrame:
        users = [m["content"] for m in messages if m["role"] == "user"]
        act = pol.fail_action(self.cfg)
        df = pd.DataFrame({"turn_idx": range(len(users)), "user_text": users, "p_safe": np.nan, "p_unsafe": np.nan,
                           "p_contro": np.nan, "p_block": np.nan, "lang": pol.conversation_lang(users),
                           "action": "allow", "block": False, "threshold": self.threshold, "guard_error": ""})
        df.loc[df.index[-1], ["action", "block", "guard_error"]] = [act, act == "block", err]
        return df

    def assess(self, messages: list[dict], *, log=None, conv_id: str | None = None, user_id: str = "anonymous",
               user_report: bool = False, downstream_refusal: bool = False, timestamp: str | None = None,
               timeout_ms: int | None = None) -> pd.DataFrame:
        """messages: [{'role': 'user'|'assistant', 'content': str}, ...] ending with a user turn.
        Returns one row per user turn; `action` of the LAST row is the decision for the new message:
        allow | careful_answer | safe_answer (+ review) | block. On timeout/error the last row carries
        `guard_error` and the configured fail action [D49]."""
        t0 = time.perf_counter()
        timeout_ms = self.cfg.guard_timeout_ms if timeout_ms is None else timeout_ms
        n_new = 0
        try:
            if self._pool is not None and timeout_ms and timeout_ms > 0:
                df, n_new = self._pool.submit(self._assess_core, messages).result(timeout=timeout_ms / 1000)
            else:
                df, n_new = self._assess_core(messages)
        except FutureTimeout:
            df = self._fallback(messages, f"timeout>{timeout_ms}ms")
        except Exception as e:  # noqa: BLE001  the chatbot must get a decision even when the guard fails
            df = self._fallback(messages, f"error:{type(e).__name__}:{str(e)[:120]}")
        self.latency_ms.append(((time.perf_counter() - t0) * 1000, n_new))
        keep = ["turn_idx", "user_text", "lang", "p_safe", "p_unsafe", "p_contro", "p_block", "action", "block",
                "threshold", "guard_error"]
        keep += [c for c in ("base_p_unsafe", "hate_p_cur", "mem_attack_max", "mem_benign_max", "n_prompts")
                 if c in df.columns]
        out = df[keep].reset_index(drop=True)
        if log is not None:
            log.record(conv_id or "unknown", messages, out, user_id=user_id, guard_version=self.version,
                       threshold=float(out["threshold"].iloc[-1]), user_report=user_report,
                       downstream_refusal=downstream_refusal, timestamp=timestamp,
                       decision_system=self.decision_system)
        return out

    def assess_response(self, messages: list[dict], draft: str) -> dict:
        """Second check: moderate the assistant's draft reply in context (Qwen3Guard response
        moderation [R1]). action 'replace' -> do not send the draft; answer with the safe fallback. [D51]"""
        if not self.cfg.response_moderation:
            return {"p_unsafe": np.nan, "action": "allow"}
        msgs = list(messages) + [{"role": "assistant", "content": draft}]
        key = chain_keys(msgs, f"resp:{self.version}")[-1]
        v = self.cache.get(key)
        if v is None:
            groups = [G.turn_prompts(self.tok, msgs, self.cfg.max_seq_len - 16, self.cfg)]
            if self.cfg.response_use_adapter:
                sc = G.score_prompt_groups(self.model, self.tok, groups, 1)
            else:
                with self.model.disable_adapter():                    # adapter only saw prompt moderation
                    sc = G.score_prompt_groups(self.model, self.tok, groups, 1)
            v = {"p_unsafe": float(sc.at[0, "p_unsafe"]), "p_contro": float(sc.at[0, "p_contro"])}
            self.cache.put(key, v)
        lang = pol.conversation_lang([m["content"] for m in messages if m["role"] == "user"])
        tau = threshold_for(self.response_threshold, lang)
        return {**v, "threshold": tau, "action": "replace" if v["p_unsafe"] > tau else "allow"}

    def latency_report(self) -> dict:
        if not self.latency_ms:
            return {}
        a = np.array(self.latency_ms, float)
        return {"calls": len(a), "p50_ms": float(np.percentile(a[:, 0], 50)), "p95_ms": float(np.percentile(a[:, 0], 95)),
                "max_ms": float(a[:, 0].max()), "mean_new_turns": float(a[:, 1].mean()), "cache": self.cache.stats()}

"""Dedicated Korean hate-speech classifier that runs next to the guard. [D54]

The full run left Korean hate speech at 22-57% recall (K-HATERS, K-MHaS, BEEP, UnSmile, APEACH): a general
safety LLM tuned on a 1:1 mix of many harms under-weights comment-style hate. Hate speech is in scope for
blocking, so a small encoder classifier trained only on Korean hate corpora is added. It is cheap
(~110M parameters, one forward pass per user turn, cached) and is used two ways:

* features `hate_p_cur` / `hate_p_max` for the LightGBM stacker, and
* an OR-rule beside the guard-only system (its own Korean threshold at the same benign FPR budget).

Backends
* "transformer": beomi/KcELECTRA-base-v2022 (trained on Korean news comments, MIT) [R80], or
  klue/roberta-base [R81]; standard sequence-classification fine-tuning.
* "ngram": character 1-4-gram hashing + logistic regression [R48]; offline smoke runs and a fallback.

Data: Unsafe vs not-Unsafe user turns of `hate_train_sources` (guard_train role; guard_val for early
checks) plus Safe Korean chat turns from `hate_negative_sources`, so ordinary chatbot traffic is
represented among the negatives. Rude-but-not-hateful (Controversial) turns are negatives, matching the
guard's label semantics [D4].
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HATE_FEATURES = ["hate_p_cur", "hate_p_max"]


def training_frame(pre: pd.DataFrame, role: str, cfg, rng_seed: int = 0) -> pd.DataFrame:
    """user_text, y for one role (one row per distinct Korean user turn)."""
    d = pre[(pre["role"] == role) & (pre["lang"] == "ko") & (pre["label3"] != "Unknown")]
    core = d[d["source"].isin(cfg.hate_train_sources)]
    neg = d[d["source"].isin(cfg.hate_negative_sources) & (d["label3"] == "Safe")]
    n_core_neg = int((core["label3"] != "Unsafe").sum())
    if len(neg) > max(n_core_neg, 1):
        neg = neg.sample(max(n_core_neg, 1), random_state=rng_seed)
    out = pd.concat([core, neg])
    out = pd.DataFrame({"user_text": out["user_text"].astype(str), "y": (out["label3"] == "Unsafe").astype(int)})
    return out.drop_duplicates("user_text").sample(frac=1.0, random_state=rng_seed).reset_index(drop=True)


# ======================================================================================
# backends
# ======================================================================================
class NgramHateClassifier:
    backend = "ngram"

    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = HashingVectorizer(analyzer="char_wb", ngram_range=(1, 4), n_features=2 ** 20,
                                     alternate_sign=False, norm="l2")
        self.clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")

    def fit(self, texts, y, val_texts=None, val_y=None, **_):
        self.clf.fit(self.vec.transform(list(texts)), np.asarray(y, int))
        return {}

    def predict_proba(self, texts) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros(0, np.float32)
        return self.clf.predict_proba(self.vec.transform([str(t) for t in texts]))[:, 1].astype(np.float32)

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "ngram.pkl", "wb") as f:
            pickle.dump(self.clf, f)
        (d / "hate_meta.json").write_text(json.dumps({"backend": self.backend}))

    @classmethod
    def load(cls, d: Path):
        obj = cls()
        with open(Path(d) / "ngram.pkl", "rb") as f:
            obj.clf = pickle.load(f)
        return obj


class TransformerHateClassifier:
    backend = "transformer"

    def __init__(self, model_id: str, max_len: int = 128, token=None, model=None, tok=None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.max_len = max_len
        self.tok = tok or AutoTokenizer.from_pretrained(model_id, token=token)
        self.model = model or AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2, token=token)

    def fit(self, texts, y, val_texts=None, val_y=None, cfg=None, out_dir: Path | None = None):
        import torch
        from datasets import Dataset
        from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed
        set_seed(cfg.seed)

        def enc(b):
            return self.tok(b["text"], truncation=True, max_length=self.max_len)
        tr = Dataset.from_dict({"text": list(map(str, texts)), "label": list(map(int, y))}).map(enc, batched=True)
        va = (Dataset.from_dict({"text": list(map(str, val_texts)), "label": list(map(int, val_y))}).map(enc, batched=True)
              if val_texts is not None and len(val_texts) else None)
        cuda = torch.cuda.is_available()
        args = TrainingArguments(
            output_dir=str(out_dir or "hate_ckpt"), per_device_train_batch_size=cfg.hate_batch_size,
            per_device_eval_batch_size=cfg.hate_batch_size * 2, learning_rate=cfg.hate_lr,
            num_train_epochs=cfg.hate_epochs, warmup_steps=0.06, weight_decay=0.01, logging_steps=50,
            eval_strategy="epoch" if va is not None else "no", save_strategy="no", report_to="none",
            bf16=bool(cuda and cfg.bf16), seed=cfg.seed, use_cpu=not cuda, dataloader_num_workers=0)
        trainer = Trainer(model=self.model, args=args, train_dataset=tr, eval_dataset=va,
                          data_collator=DataCollatorWithPadding(self.tok))
        res = trainer.train()
        self.model = trainer.model.eval()
        return dict(res.metrics)

    def predict_proba(self, texts, batch_size: int = 64) -> np.ndarray:
        import torch
        if len(texts) == 0:
            return np.zeros(0, np.float32)
        dev = next(self.model.parameters()).device
        out = []
        for i in range(0, len(texts), batch_size):
            b = self.tok([str(t) for t in texts[i:i + batch_size]], truncation=True, max_length=self.max_len,
                         padding=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                logits = self.model(**b).logits.float()
            out.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(d)
        self.tok.save_pretrained(d)
        (d / "hate_meta.json").write_text(json.dumps({"backend": self.backend, "max_len": self.max_len}))

    @classmethod
    def load(cls, d: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        meta = json.loads((Path(d) / "hate_meta.json").read_text())
        tok = AutoTokenizer.from_pretrained(d)
        model = AutoModelForSequenceClassification.from_pretrained(d)
        if torch.cuda.is_available():
            model = model.to("cuda")
        return cls("", meta.get("max_len", 128), model=model.eval(), tok=tok)


def make(cfg, token=None):
    if cfg.hate_backend == "ngram":
        return NgramHateClassifier()
    return TransformerHateClassifier(cfg.hate_model, cfg.hate_max_len, token=token)


def load(d: str | Path):
    meta = json.loads((Path(d) / "hate_meta.json").read_text())
    return NgramHateClassifier.load(d) if meta["backend"] == "ngram" else TransformerHateClassifier.load(d)


def hate_features(rows: pd.DataFrame, clf) -> pd.DataFrame:
    """rows: prefix_id, conv_id, turn_idx, user_text -> hate_p_cur and its causal running max."""
    rows = rows[["prefix_id", "conv_id", "turn_idx", "user_text"]].reset_index(drop=True)
    texts = rows["user_text"].fillna("").astype(str)
    uniq = texts.drop_duplicates()
    p = dict(zip(uniq, clf.predict_proba(uniq.tolist())))
    f = pd.DataFrame({"prefix_id": rows["prefix_id"], "conv_id": rows["conv_id"], "turn_idx": rows["turn_idx"],
                      "hate_p_cur": texts.map(p).astype(float)})
    f = f.sort_values(["conv_id", "turn_idx"])
    f["hate_p_max"] = f.groupby("conv_id")["hate_p_cur"].cummax()
    return f.drop(columns=["conv_id", "turn_idx"]).reset_index(drop=True)

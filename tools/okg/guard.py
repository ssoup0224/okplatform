"""LLM guard: prompt rendering, logit scoring, generation probe, LoRA SFT.

Key choices and their sources
* Prompt = the model's OWN chat template, called exactly as the Qwen3Guard README does:
  apply_chat_template(messages, tokenize=False) [R1]. The template already appends the
  assistant header + empty think block [R3]. One function (render_prompt) is used for
  training AND inference, fixing the previous project's train/inference template skew.
* Verdict = probability of the first label token after "Safety:" read from the logits of a
  single forward pass — no free-text parsing [R45][R46][R47]. Safe/Unsafe/Controversial are
  Qwen3Guard's three severity levels [R1][R2].
* Training target = Qwen3Guard's native output "Safety: X\\nCategories: Y" [R1]; loss only on
  the completion via TRL prompt-completion datasets (completion_only_loss) [R4]; LoRA via
  TRL's peft_config [R4][R6].
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import QG_CATEGORIES, SAFETY_LABELS
from .taxonomy import completion_text

TEMPLATE_KWARGS: dict = {}          # identical at train and inference time (asserted below) [R1]
SAFETY_LEAD = "Safety:"             # the model's output starts with "Safety: <label>" [R1]
CATEGORY_LEAD = "Safety: Unsafe\nCategories:"
_SAFE_RE = re.compile(r"Safety: (Safe|Unsafe|Controversial)")   # official extraction regex [R1]


# ======================================================================================
# model / tokenizer loading
# ======================================================================================
def resolve_model_path(cfg) -> str:
    if cfg.guard_model_id == "tiny-local":
        from .tiny_model import build_tiny_guard
        return str(build_tiny_guard(cfg.cache_dir / "tiny-local"))
    return cfg.guard_model_id


def load_tokenizer(path: str, token=None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(path: str, cfg, token=None, for_training: bool = False):
    """bf16 on GPU (fp32 on CPU); optional 4-bit NF4 QLoRA [R51]."""
    import torch
    from transformers import AutoModelForCausalLM
    cuda = torch.cuda.is_available()
    kwargs = {"token": token}
    kwargs["dtype"] = torch.bfloat16 if (cuda and cfg.bf16) else torch.float32
    if cuda:
        kwargs["device_map"] = "auto"
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)                                      # [R51]
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if cfg.load_in_4bit and for_training:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    model.eval()
    return model


# ======================================================================================
# prompt rendering
# ======================================================================================
def render_prompt(tok, messages: list[dict]) -> str:
    return tok.apply_chat_template(messages, tokenize=False, **TEMPLATE_KWARGS)   # [R1]


def n_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def fit_messages(tok, messages: list[dict], max_tokens: int) -> tuple[list[dict], int]:
    """Sliding window: drop the OLDEST user/assistant pair until the prompt fits; the latest
    user turn is always kept (its head is trimmed as a last resort). [R36][D17]"""
    msgs = [dict(m) for m in messages]
    dropped = 0
    while n_tokens(tok, render_prompt(tok, msgs)) > max_tokens and len(msgs) > 1:
        msgs = msgs[2:] if len(msgs) > 2 else msgs[1:]
        while msgs and msgs[0]["role"] != "user":
            msgs = msgs[1:]
        dropped += 1
    while n_tokens(tok, render_prompt(tok, msgs)) > max_tokens and len(msgs[-1]["content"]) > 20:
        c = msgs[-1]["content"]
        msgs[-1]["content"] = c[len(c) // 4:]        # keep the end of the message
        dropped += 1
    return msgs, dropped


# ======================================================================================
# candidate-token scoring
# ======================================================================================
def first_token_ids(tok, lead: str, candidates, probe_prompt: str) -> dict:
    """Token id that starts each candidate right after `lead` (e.g. ' Unsafe' after 'Safety:').
    Checks that (a) prompt+lead tokenises as a prefix of prompt+lead+' '+cand and
    (b) candidates start with distinct tokens; otherwise raises. [D18]"""
    base = tok(probe_prompt + lead, add_special_tokens=False)["input_ids"]
    ids = {}
    for c in candidates:
        full = tok(probe_prompt + lead + " " + c, add_special_tokens=False)["input_ids"]
        if full[: len(base)] != base or len(full) <= len(base):
            raise ValueError(f"tokenisation of '{lead}' + ' {c}' is not prefix-stable")
        ids[c] = full[len(base)]
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"candidates share a first token: {ids}")
    return ids


def _device(model):
    import torch
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def score_next_token(model, tok, prompts: list[str], lead: str, cand_ids: dict, batch_size: int = 8) -> pd.DataFrame:
    """P(candidate) for the token following prompt+lead.

    Left padding + explicit position_ids so the last position is the real last token [R53];
    logits_to_keep=1 avoids materialising full-vocabulary logits for every position.
    Returns per-candidate probability renormalised over the candidates, plus the raw mass
    the model puts on them (a format-health signal). On CUDA OOM the batch is halved and
    retried [R52].
    """
    import torch
    tok.padding_side = "left"
    names = list(cand_ids)
    idx = torch.tensor([cand_ids[n] for n in names])
    order = np.argsort([len(p) for p in prompts])[::-1]        # long first -> OOM early
    res = np.zeros((len(prompts), len(names) + 1), dtype=np.float32)
    dev = _device(model)
    i, bs = 0, batch_size
    while i < len(order):
        sel = order[i:i + bs]
        batch = [prompts[j] + lead for j in sel]
        try:
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
            with torch.no_grad():
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                            position_ids=pos, logits_to_keep=1)
            logp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1).cpu()
            cand = logp[:, idx]
            res[sel, :len(names)] = torch.softmax(cand, dim=-1).numpy()
            res[sel, len(names)] = torch.exp(cand).sum(-1).numpy()
            i += len(sel)
        except torch.cuda.OutOfMemoryError:            # [R52]
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"  OOM -> batch size {bs}")
    cols = [f"p_{n}" for n in names] + ["cand_mass"]
    return pd.DataFrame(res, columns=cols)


def score_safety(model, tok, prompts: list[str], batch_size: int) -> pd.DataFrame:
    ids = first_token_ids(tok, SAFETY_LEAD, SAFETY_LABELS, prompts[0] if prompts else "")
    df = score_next_token(model, tok, prompts, SAFETY_LEAD, ids, batch_size)
    return df.rename(columns={"p_Safe": "p_safe", "p_Unsafe": "p_unsafe", "p_Controversial": "p_contro"})


def teacher_categories(model, tok, prompts: list[str], batch_size: int) -> list[str]:
    """Category for Unsafe examples whose source gives none: argmax of the base model's
    first category token after a forced 'Safety: Unsafe\\nCategories:' (self-distillation,
    in the spirit of Qwen3Guard's model-assisted labelling [R2]). [D19]"""
    if not prompts:
        return []
    ids = first_token_ids(tok, CATEGORY_LEAD, QG_CATEGORIES, prompts[0])
    df = score_next_token(model, tok, prompts, CATEGORY_LEAD, ids, batch_size)
    cols = [f"p_{c}" for c in QG_CATEGORIES]
    return [QG_CATEGORIES[k] for k in df[cols].to_numpy().argmax(1)]


def generation_probe(model, tok, prompts: list[str], max_new_tokens: int = 24) -> pd.DataFrame:
    """Greedy generation + the official regex: measures parse rate (the failure mode of the
    previous project, where fine-tuned models stopped emitting a verdict on multi-turn input). [R1]"""
    import torch
    tok.padding_side = "left"
    dev = _device(model)
    rows = []
    for p in prompts:
        enc = tok([p], return_tensors="pt", add_special_tokens=False).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        m = _SAFE_RE.search(text)
        rows.append({"generated": text, "parsed": m.group(1) if m else None})
    return pd.DataFrame(rows)


# ======================================================================================
# SFT dataset
# ======================================================================================
def select_training_prefixes(prefixes: pd.DataFrame, cfg, rng: random.Random) -> pd.DataFrame:
    """Choose which (conversation, user-turn) prefixes become SFT examples.

    1. drop unknown labels (pre-attack turns of native attacks) unless configured [D6]
    2. at most `max_prefixes_per_conv` per conversation, always keeping the attack turn,
       so long dialogues do not dominate [D5]
    3. cap hate-comment corpora at `comment_corpora_ko_share` of Korean examples [D4]
    4. balance Unsafe vs not-Unsafe to `target_unsafe_ratio` within each language [R2]
    5. cap the total
    """
    df = prefixes[prefixes["label3"] != "Unknown"].copy() if not cfg.train_on_ambiguous_prefixes else prefixes.copy()
    df.loc[df["label3"] == "Unknown", ["label3", "category"]] = ["Safe", "None"]     # only if explicitly enabled [D6]
    keep = []
    for cid, g in df.groupby("conv_id", sort=False):
        idx = list(g.index)
        must = [i for i in idx if g.at[i, "turn_idx"] == g.at[i, "t_attack"]]
        rest = [i for i in idx if i not in must]
        rng.shuffle(rest)
        keep += (must + rest)[: cfg.max_prefixes_per_conv]
    df = df.loc[sorted(keep)]
    # comment-corpus cap (Korean) [D4]
    ko = df["lang"] == "ko"
    com = ko & (df["family"] == "comment")
    max_com = int(cfg.comment_corpora_ko_share / (1 - cfg.comment_corpora_ko_share) * (ko & ~com).sum())
    if com.sum() > max_com:
        drop = rng.sample(list(df.index[com]), int(com.sum() - max_com))
        df = df.drop(index=drop)
    # class balance per language [R2]
    parts = []
    for lang, g in df.groupby("lang"):
        pos, neg = g[g["label3"] == "Unsafe"], g[g["label3"] != "Unsafe"]
        r = cfg.target_unsafe_ratio
        if len(pos) == 0 or len(neg) == 0:
            parts.append(g)
            continue
        want_neg = int(len(pos) * (1 - r) / r)
        if len(neg) > want_neg:
            neg = neg.sample(want_neg, random_state=cfg.seed)
        else:
            want_pos = int(len(neg) * r / (1 - r))
            pos = pos.sample(min(len(pos), want_pos), random_state=cfg.seed)
        parts.append(pd.concat([pos, neg]))
    df = pd.concat(parts)
    if len(df) > cfg.max_train_examples:
        df = df.sample(cfg.max_train_examples, random_state=cfg.seed)
    return df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)


def build_sft_records(sel: pd.DataFrame, convs: pd.DataFrame, tok, cfg, prompts_only: bool = False) -> list[dict]:
    """prompt = rendered conversation prefix (fitted to max_seq_len); completion = native format."""
    by_id = convs.set_index("conv_id")
    reserve = 48                                     # completion + EOS budget [D17]
    recs = []
    for r in sel.itertuples(index=False):
        msgs = json.loads(by_id.at[r.conv_id, "messages"])[: r.msg_end + 1]
        msgs, _ = fit_messages(tok, msgs, cfg.max_seq_len - reserve)
        prompt = render_prompt(tok, msgs)
        if prompts_only:                                    # teacher pass: category still unknown
            recs.append({"prompt": prompt, "conv_id": r.conv_id})
            continue
        if r.label3 != "Safe" and r.category not in QG_CATEGORIES:
            raise ValueError(f"{r.prefix_id}: {r.label3} example without a category (teacher step missing)")
        comp = completion_text(r.label3, r.category)
        assert n_tokens(tok, prompt + comp) + 1 <= cfg.max_seq_len, "example would be truncated"
        recs.append({"prompt": prompt, "completion": comp, "conv_id": r.conv_id, "lang": r.lang,
                     "label3": r.label3})
    return recs


def train_lora(cfg, model, tok, train_recs: list[dict], val_recs: list[dict], ckpt_dir: Path, resume: bool):
    """LoRA SFT with TRL. prompt/completion columns -> completion-only loss [R4]; LoRA on all
    linear layers [R6]; Trainer checkpointing/resume [R7]; seeding [R8]."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(cfg.seed)                                                                    # [R8]
    cols = ["prompt", "completion"]
    train_ds = Dataset.from_list([{k: r[k] for k in cols} for r in train_recs])
    val_ds = Dataset.from_list([{k: r[k] for k in cols} for r in val_recs]) if val_recs else None
    peft_cfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                          target_modules=cfg.lora_target_modules, task_type="CAUSAL_LM")  # [R6]
    cuda = torch.cuda.is_available()
    args = SFTConfig(
        output_dir=str(ckpt_dir),
        max_length=cfg.max_seq_len, packing=False,
        completion_only_loss=True,                                                        # [R4]
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate, num_train_epochs=cfg.num_train_epochs, max_steps=cfg.max_steps,
        # transformers>=5: a float in [0,1) passed as warmup_steps is a ratio of total steps [R7]
        warmup_steps=cfg.warmup_ratio, lr_scheduler_type=cfg.lr_scheduler_type,
        bf16=bool(cfg.bf16 and cuda), gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=10, save_strategy="steps", save_steps=cfg.save_steps, save_total_limit=2,   # [R7]
        eval_strategy="steps" if val_ds is not None else "no", eval_steps=cfg.save_steps,
        report_to="none", seed=cfg.seed, use_cpu=not cuda, dataloader_num_workers=0,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tok, peft_config=peft_cfg)
    has_ckpt = resume and any(Path(ckpt_dir).glob("checkpoint-*"))
    result = trainer.train(resume_from_checkpoint=True if has_ckpt else None)            # [R7]
    metrics = dict(result.metrics)
    if val_ds is not None:
        metrics.update({f"final_{k}": v for k, v in trainer.evaluate().items()})
    return trainer, metrics

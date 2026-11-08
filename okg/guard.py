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
    try:
        return tok.apply_chat_template(messages, tokenize=False, **TEMPLATE_KWARGS)   # [R1]
    except Exception:  # noqa: BLE001  a template that insists on alternating roles (e.g. MHJ's user-only turns)
        merged = []
        for m in messages:
            if merged and merged[-1]["role"] == m["role"]:
                merged[-1] = {"role": m["role"], "content": merged[-1]["content"] + "\n\n" + m["content"]}
            else:
                merged.append(dict(m))
        return tok.apply_chat_template(merged, tokenize=False, **TEMPLATE_KWARGS)


def n_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def trim_tokens(tok, text: str, max_tokens: int, keep: str = "head_tail") -> str:
    """Shorten `text` to about `max_tokens` tokens. keep="head_tail" keeps both ends (a harmful
    request at the START of a long paste must survive, so tail-only trimming is not used) [D44]."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    max_tokens = max(max_tokens, 2)
    if keep == "head":
        return tok.decode(ids[:max_tokens]).rstrip() + " …"
    h = max_tokens // 2
    return tok.decode(ids[:h]).rstrip() + " … " + tok.decode(ids[len(ids) - (max_tokens - h):]).lstrip()


MIN_LAST_TOKENS = 256      # below this budget for the latest message, the pinned opening pair is dropped


def fit_window(tok, messages: list[dict], max_tokens: int, pin_first: bool = True,
               pin_tokens: int = 384) -> tuple[list[dict], dict]:
    """Fit a conversation prefix into `max_tokens` (prompt tokens). [R36][D17][D44]

    1. If the prefix fits, nothing changes.
    2. The opening user/assistant pair is PINNED (each message cut to its head within `pin_tokens`),
       so an attacker cannot pad the middle of a conversation until the setup falls out of the window.
    3. The oldest messages after the pinned pair are dropped first; the latest user turn is always kept.
    4. If the latest message alone is still too long, it keeps its head AND tail; callers that want full
       coverage score the message in chunks (`turn_prompts`).
    Returns (messages, info) with info = {dropped, last_trimmed, last_budget}."""
    msgs = [dict(m) for m in messages]
    info = {"dropped": 0, "last_trimmed": False, "last_budget": max_tokens}

    def size(m):
        return n_tokens(tok, render_prompt(tok, m))

    if size(msgs) <= max_tokens:
        return msgs, info
    head, body = [], msgs
    if pin_first and len(msgs) >= 2:
        k = 2 if len(msgs) >= 3 and msgs[1]["role"] == "assistant" else 1
        head = [dict(m, content=trim_tokens(tok, m["content"], max(16, pin_tokens // k), keep="head")) for m in msgs[:k]]
        body = msgs[k:]
    while size(head + body) > max_tokens and len(body) > 1:
        body = body[2:] if len(body) > 2 else body[1:]
        while len(body) > 1 and body[0]["role"] != "user":
            body = body[1:]
        info["dropped"] += 1
    if size(head + body) <= max_tokens:
        return head + body, info
    ctx, last = head + body[:-1], body[-1]

    def budget(c):
        return max_tokens - size(c + [dict(last, content="")])

    if head and budget(ctx) < MIN_LAST_TOKENS:          # the latest message matters more than the opening
        ctx = body[:-1]
        info["dropped"] += 1
    avail = budget(ctx)
    while avail > 8:
        out = ctx + [dict(last, content=trim_tokens(tok, last["content"], avail))]
        if size(out) <= max_tokens:
            info.update(last_trimmed=True, last_budget=avail)
            return out, info
        avail = int(avail * 0.9)
    info.update(last_trimmed=True, last_budget=8, dropped=info["dropped"] + len(ctx))   # context cannot fit at all
    return [dict(last, content=trim_tokens(tok, last["content"], 8))], info


def fit_messages(tok, messages: list[dict], max_tokens: int, pin_first: bool = True,
                 pin_tokens: int = 384) -> tuple[list[dict], int]:
    """Backwards-compatible wrapper: (messages, number of dropped/trimmed units)."""
    msgs, info = fit_window(tok, messages, max_tokens, pin_first, pin_tokens)
    return msgs, info["dropped"] + int(info["last_trimmed"])


def message_chunks(tok, text: str, chunk_tokens: int, overlap: int, max_chunks: int) -> list[str]:
    """Overlapping token windows covering the WHOLE message; if more than `max_chunks` are needed,
    evenly spaced windows are kept (first and last always included). [D44]"""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    chunk_tokens = max(chunk_tokens, 16)
    step = max(1, chunk_tokens - max(0, min(overlap, chunk_tokens // 2)))
    starts = list(range(0, max(1, len(ids) - overlap), step))
    if len(starts) > max_chunks:
        starts = sorted({starts[round(i * (len(starts) - 1) / (max_chunks - 1))] for i in range(max_chunks)}) \
            if max_chunks > 1 else starts[:1]
    return [tok.decode(ids[s:s + chunk_tokens]) for s in starts]


def turn_prompts(tok, messages: list[dict], max_tokens: int, cfg=None) -> list[str]:
    """All prompts that together score the LAST user turn of `messages`: the fitted window, plus one
    prompt per chunk of the latest message when it had to be trimmed. Score = max over them. [D44]"""
    pin = getattr(cfg, "pin_first_turn", True)
    pin_tokens = getattr(cfg, "pin_first_max_tokens", 384)
    msgs, info = fit_window(tok, messages, max_tokens, pin, pin_tokens)
    prompts = [render_prompt(tok, msgs)]
    if info["last_trimmed"] and getattr(cfg, "chunk_long_messages", True):
        ctx = msgs[:-1]
        for ch in message_chunks(tok, messages[-1]["content"], info["last_budget"],
                                 getattr(cfg, "chunk_overlap_tokens", 64), getattr(cfg, "max_chunks_per_message", 8)):
            p = render_prompt(tok, ctx + [dict(messages[-1], content=ch)])
            if n_tokens(tok, p) <= max_tokens + 8:          # decode/re-encode can shift a few tokens
                prompts.append(p)
    return prompts


# ======================================================================================
# candidate-token scoring
# ======================================================================================
def first_token_ids(tok, lead: str, candidates, probe_prompt: str, sep: str = " ") -> dict:
    """Token id that starts each candidate right after `lead` (e.g. ' Unsafe' after 'Safety:').
    `sep` is what separates lead and candidate (" " for Qwen3Guard's 'Safety: X'; "" for Llama
    Guard, whose answer starts right after the assistant header). Checks that (a) prompt+lead
    tokenises as a prefix of prompt+lead+sep+cand and (b) candidates start with distinct tokens;
    otherwise raises. [D18]"""
    base = tok(probe_prompt + lead, add_special_tokens=False)["input_ids"]
    ids = {}
    for c in candidates:
        full = tok(probe_prompt + lead + sep + c, add_special_tokens=False)["input_ids"]
        if full[: len(base)] != base or len(full) <= len(base):
            raise ValueError(f"tokenisation of '{lead}' + '{sep}{c}' is not prefix-stable")
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


def score_prompt_groups(model, tok, groups: list[list[str]], batch_size: int) -> pd.DataFrame:
    """One row per group (= one user turn, possibly several chunk prompts): the scores of the prompt
    with the highest P(Unsafe)+P(Controversial), plus n_prompts. [D44]"""
    flat = [p for g in groups for p in g]
    owner = np.repeat(np.arange(len(groups)), [len(g) for g in groups])
    cols = ["p_safe", "p_unsafe", "p_contro", "cand_mass"]
    if not flat:
        return pd.DataFrame(columns=cols + ["n_prompts"])
    sc = score_safety(model, tok, flat, batch_size)
    sc["_g"], sc["_r"] = owner, sc["p_unsafe"] + sc["p_contro"]
    best = sc.sort_values("_r", ascending=False).drop_duplicates("_g").sort_values("_g")
    out = best[cols].reset_index(drop=True)
    out["n_prompts"] = [len(g) for g in groups]
    return out


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


def select_replay(prefixes: pd.DataFrame, chosen_ids: set, n: int, cfg) -> pd.DataFrame:
    """Candidates for base-model replay [D52]: guard_train prefixes in `cfg.replay_langs` that were not
    selected for SFT. Their targets are set later from the BASE model's own verdict (self-distillation),
    so fine-tuning cannot silently overwrite the zero-shot conventions there (e.g. OR-Bench toxic)."""
    if n <= 0 or not len(prefixes):
        return prefixes.iloc[0:0]
    c = prefixes[prefixes["lang"].isin(cfg.replay_langs) & ~prefixes["prefix_id"].isin(chosen_ids)
                 & (prefixes["label3"] != "Unknown")]
    return c.sample(min(n, len(c)), random_state=cfg.seed) if len(c) else c


def apply_replay_labels(rep: pd.DataFrame, base_scores: pd.DataFrame) -> pd.DataFrame:
    """Replace dataset labels with the base model's argmax verdict. A dataset-Unsafe row the base calls
    Safe is dropped: replay must never teach 'Safe' against a human 'Unsafe' label. [D52]"""
    rep = rep.reset_index(drop=True).copy()
    probs = base_scores[["p_safe", "p_unsafe", "p_contro"]].to_numpy()
    lab = np.array(["Safe", "Unsafe", "Controversial"])[probs.argmax(1)]
    keep = ~((lab == "Safe") & (rep["label3"].to_numpy() == "Unsafe"))
    rep["label3"] = lab
    rep.loc[rep["label3"] == "Safe", "category"] = "None"
    rep.loc[rep["label3"] != "Safe", "category"] = "Unknown"          # filled by the teacher pass
    rep["replay"] = True
    return rep[keep].reset_index(drop=True)


def build_sft_records(sel: pd.DataFrame, convs: pd.DataFrame, tok, cfg, prompts_only: bool = False) -> list[dict]:
    """prompt = rendered conversation prefix (fitted to max_seq_len); completion = native format."""
    by_id = convs.set_index("conv_id")
    reserve = 48                                     # completion + EOS budget [D17]
    recs = []
    for r in sel.itertuples(index=False):
        msgs = json.loads(by_id.at[r.conv_id, "messages"])[: r.msg_end + 1]
        msgs, _ = fit_messages(tok, msgs, cfg.max_seq_len - reserve, cfg.pin_first_turn, cfg.pin_first_max_tokens)
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


def latest_complete_checkpoint(ckpt_dir) -> str | None:
    """Newest Trainer checkpoint that finished writing. A session killed mid-save leaves a partial
    checkpoint-N; Trainer writes trainer_state.json and then rng_state*.pth last, so a folder without
    both is incomplete and is removed rather than resumed from. [D57]"""
    import shutil
    good = []
    for p in Path(ckpt_dir).glob("checkpoint-*"):
        if not p.is_dir() or not p.name.split("-")[-1].isdigit():
            continue
        if (p / "trainer_state.json").exists() and any(p.glob("rng_state*.pth")):
            good.append(p)
        else:
            print(f"[resume] discarding incomplete {p.name}")
            shutil.rmtree(p, ignore_errors=True)
    return str(max(good, key=lambda p: int(p.name.split("-")[-1]))) if good else None


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
        eval_strategy="steps" if val_ds is not None else "no", eval_steps=cfg.eval_steps,
        report_to="none", seed=cfg.seed, use_cpu=not cuda, dataloader_num_workers=0,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tok, peft_config=peft_cfg)
    ckpt = latest_complete_checkpoint(ckpt_dir) if resume else None
    if ckpt:
        print(f"[resume] SFT from {ckpt}")
    result = trainer.train(resume_from_checkpoint=ckpt)                                  # [R7][D57]
    metrics = dict(result.metrics)
    if val_ds is not None:
        metrics.update({f"final_{k}": v for k, v in trainer.evaluate().items()})
    return trainer, metrics

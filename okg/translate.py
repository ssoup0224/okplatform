"""EN -> KO translation of native multi-turn dialogues.

Uses the same EN->KO model that produced WildGuardMix-KO (nayohan/llama3-instrucTrans-enko-8b)
[R24][R41], with the system prompt and chat-template call from its model card [R41]. Both
attack AND benign conversations are translated, so machine-translation artefacts appear
in both classes and cannot become a label shortcut (lesson from the previous project's
length shortcut) [D23].
"""
from __future__ import annotations

import json

import pandas as pd

from .schema import make_conv

MAX_CHARS = 1500


def translate_texts(model, tok, texts: list[str], system_prompt: str, max_new_tokens: int, batch_size: int = 8) -> list[str]:
    import torch
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = next(model.parameters()).device
    # Llama-3 chat turns end with <|eot_id|>, not <|end_of_text|>; stop on either so a batch
    # does not run on to max_new_tokens when the checkpoint's generation_config omits it
    stop = [tok.eos_token_id] + [i for i in [tok.convert_tokens_to_ids("<|eot_id|>")]
                                 if isinstance(i, int) and i != tok.unk_token_id and i != tok.eos_token_id]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # similar lengths per batch -> less padding
    out_sorted = []
    n_batches = (len(order) + batch_size - 1) // batch_size
    for b, i in enumerate(range(0, len(order), batch_size)):
        chunk = [texts[j][:MAX_CHARS] for j in order[i:i + batch_size]]
        prompts = [tok.apply_chat_template([{"role": "system", "content": system_prompt},
                                            {"role": "user", "content": t}],
                                           tokenize=False, add_generation_prompt=True) for t in chunk]   # [R41]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id,
                                 eos_token_id=stop)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            out_sorted.append(tok.decode(row, skip_special_tokens=True).strip())
        print(f"    batch {b + 1}/{n_batches} ({len(out_sorted)}/{len(texts)} messages)", flush=True)
    out = [""] * len(texts)
    for j, t in zip(order, out_sorted):
        out[j] = t
    return out


def translate_convs(model, tok, convs: pd.DataFrame, cfg) -> list[dict]:
    """Translate every message of each conversation; seed_id/role/labels are inherited."""
    msgs_all = [json.loads(m) for m in convs["messages"]]
    flat = [m["content"] for msgs in msgs_all for m in msgs]
    tr = translate_texts(model, tok, flat, cfg.translator_system_prompt, cfg.translate_max_new_tokens,
                         batch_size=cfg.score_batch_size)
    out, k = [], 0
    for r, msgs in zip(convs.itertuples(index=False), msgs_all):
        new = []
        for m in msgs:
            new.append({"role": m["role"], "content": tr[k] or m["content"]})
            k += 1
        out.append(make_conv(source=f"{r.source}_ko_mt", messages=new, label3=json.loads(r.label3),
                             category=json.loads(r.category), lang="ko", seed_id=r.seed_id,
                             family=f"{r.family}_mt", pool=r.pool, license=r.license,
                             commercial_ok=bool(r.commercial_ok), fixed_role=r.role,
                             latency_valid=bool(r.latency_valid),
                             label_origin=r.label_origin if r.label_origin != "unset" else "derived",
                             translated=True, mt_model=cfg.translator_model))
    return out

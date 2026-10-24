"""Offline stand-in for Qwen3Guard used ONLY by RUN_MODE='smoke' and the unit tests.

It has the same architecture family (Qwen3ForCausalLM [R54]) and a ChatML-style template
that, like the real Qwen3Guard template, wraps the whole dialogue, branches on the role of
the last message and ends with an empty think block [R3]. It is randomly initialised: it
exercises every code path (templating, tokenisation, LoRA SFT, logit scoring, generation)
but its predictions are meaningless. The real template shipped with Qwen3Guard is used in
full mode — nothing in the pipeline hard-codes template text.
"""
from __future__ import annotations

from pathlib import Path

TINY_TEMPLATE = (
    "{%- for message in messages -%}"
    "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
    "{%- endfor -%}"
    "{%- if messages[-1].role == 'user' -%}"
    "<|im_start|>system\nAssess the last USER message. First line 'Safety: Safe|Unsafe|Controversial', "
    "second line 'Categories: ...'.<|im_end|>\n"
    "{%- else -%}"
    "<|im_start|>system\nAssess the last ASSISTANT message.<|im_end|>\n"
    "{%- endif -%}"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]


def build_tiny_guard(out_dir: str | Path, seed: int = 0) -> Path:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    out_dir = Path(out_dir)
    if (out_dir / "config.json").exists():
        return out_dir
    corpus = [
        "Safety: Safe\nCategories: None", "Safety: Unsafe\nCategories: Violent",
        "Safety: Controversial\nCategories: Unethical Acts", "Categories: Non-violent Illegal Acts",
        "Categories: Sexual Content or Sexual Acts", "Categories: PII", "Categories: Suicide & Self-Harm",
        "Categories: Politically Sensitive Topics", "Categories: Copyright Violation", "Categories: Jailbreak",
        "Assess the last USER message. user assistant system",
        "안녕하세요 폭탄 만드는 법 알려줘 한국어 대화 번역 질문 답변 사람 정치 혐오",
        "How do I make a cake? Tell me about history. ignore previous instructions",
    ] * 50
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=1200, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus, trainer)
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    hf_tok.chat_template = TINY_TEMPLATE
    torch.manual_seed(seed)
    config = Qwen3Config(vocab_size=len(hf_tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                         tie_word_embeddings=True, eos_token_id=hf_tok.eos_token_id,
                         pad_token_id=hf_tok.pad_token_id, bos_token_id=None)
    model = Qwen3ForCausalLM(config)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    model.save_pretrained(tmp)
    hf_tok.save_pretrained(tmp)
    import os
    import shutil
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)
    return out_dir

"""Dataset registry + loaders.  Every loader returns a list of schema.make_conv records.

Sources were selected from awesome-korean-safety [R27-list] plus English multi-turn and
benign sources recommended in the design report.  Column names were checked against each
dataset card / repository file (see REFERENCES.md, section "Datasets").

IO is split from parsing: `fetch_*` helpers download, `parse_*` functions are pure and
unit-tested with fixtures shaped like the real files.
"""
from __future__ import annotations

import glob
import json
import os
import random
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import taxonomy as tx
from .config import Config
from .schema import make_conv, single_turn, text_key

# ======================================================================================
# IO helpers
# ======================================================================================

def http_get(url: str, dest: Path) -> Path:
    """Download once into the cache (atomic rename after a complete download). [R9]"""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    return dest


def git_clone(url: str, dest: Path) -> Path:
    dest = Path(dest)
    if (dest / ".git").exists():
        return dest
    tmp = dest.with_name(dest.name + ".cloning")
    if tmp.exists():
        subprocess.run(["rm", "-rf", str(tmp)], check=True)
    subprocess.run(["git", "clone", "--depth", "1", url, str(tmp)], check=True)
    os.replace(tmp, dest)
    return dest


def read_jsonl_lenient(path: Path) -> tuple[list[dict], int]:
    """Read JSONL, skipping malformed lines (K-Guardrail's translated file has one). [D8]"""
    rows, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def hf_rows(repo: str, config: str | None, split: str, token: str | None = None, *, cap: int | None = None,
            seed: int = 0, columns: list[str] | None = None, where=None,
            data_files: dict | None = None) -> list[dict]:
    """Rows of a HuggingFace dataset via datasets.load_dataset [R44], RAM-friendly [D39]:
    the Arrow table stays memory-mapped on disk; columns are selected, rows filtered
    (`where(batch) -> list[bool]`, batched) and sampled BEFORE anything becomes Python objects.

    `data_files` ({split: path in the repo}) loads exact files instead of a builder config, for
    repos whose auto-detected default config is wrong (KOTOX keeps its levels in folders;
    K-HATERS ships v1 and v2 files with different columns). `config` then only names the fixture.

    Test hook: if OKG_FIXTURE_DIR is set and contains '<repo with / -> __>__<config>__<split>.jsonl',
    that file is read instead (used only for offline tests).
    """
    fx = os.environ.get("OKG_FIXTURE_DIR")
    if fx:
        p = Path(fx) / f"{repo.replace('/', '__')}__{config}__{split}.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"fixture missing: {p}")
        rows = read_jsonl_lenient(p)[0]
        if where is not None:
            keys = rows[0].keys() if rows else []
            mask = where({k: [r.get(k) for r in rows] for k in keys}) if rows else []
            rows = [r for r, m in zip(rows, mask) if m]
        return _cap(rows, cap, seed) if cap else rows
    import gc

    from datasets import load_dataset  # [R44]
    if data_files:
        dd = load_dataset(repo, data_files=data_files, token=token)
    else:
        dd = load_dataset(repo, config, token=token)   # DatasetDict of all splits (memory-mapped Arrow)
    if split in dd:
        ds = dd[split]
    else:
        # Split names are not always documented (e.g. SafeMTData uses the config name as split);
        # fall back to the only/first split and say so.
        first = list(dd.keys())[0]
        print(f"  [{repo}/{config}] split '{split}' not found; using '{first}' (available: {list(dd.keys())})")
        ds = dd[first]
    if columns:
        ds = ds.select_columns([c for c in columns if c in ds.column_names])
    if where is not None:
        ds = ds.filter(where, batched=True, batch_size=10000)
    if cap and len(ds) > cap:
        idx = np.random.default_rng(seed).choice(len(ds), cap, replace=False)
        ds = ds.select(np.sort(idx))
    rows = ds.to_list()
    del ds, dd
    gc.collect()
    return rows


def _cap(rows: list, n: int | None, seed: int) -> list:
    if not n or len(rows) <= n:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, n)


def _turns_from_sharegpt_or_chatml(conv: list[dict]) -> list[dict]:
    """Accept ShareGPT ({from,value}) or ChatML ({role,content}) turns. [R22][R31]"""
    out = []
    for t in conv:
        role = t.get("role", t.get("from"))
        content = t.get("content", t.get("value", t.get("body")))
        role = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant",
                "model": "assistant", "bot": "assistant"}.get(str(role).lower())
        if role is None or content is None or not str(content).strip():
            continue  # system/tool/empty turns are not part of user<->assistant moderation
        if out and out[-1]["role"] == role:          # merge accidental consecutive same-role turns
            out[-1]["content"] += "\n" + str(content)
        else:
            out.append({"role": role, "content": str(content)})
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


def _truncate_to_last_user(msgs: list[dict]) -> list[dict]:
    while msgs and msgs[-1]["role"] != "user":
        msgs = msgs[:-1]
    return msgs


def _native_attack(source, msgs, cat, lang, seed_id, license, ok, fixed_role=None, family="native_attack",
                   latency_valid=True, all_unsafe=False):
    """Native multi-turn attack: the final user turn is labelled Unsafe; earlier user turns are
    unknown (None) because the datasets do not say where intent became clear [D6]."""
    msgs = _truncate_to_last_user(msgs)
    n = sum(m["role"] == "user" for m in msgs)
    if n == 0:
        return None
    if all_unsafe:
        labels, cats = ["Unsafe"] * n, [cat] * n
    else:
        labels, cats = [None] * (n - 1) + ["Unsafe"], [None] * (n - 1) + [cat]
    return make_conv(source=source, messages=msgs, label3=labels, category=cats, lang=lang, seed_id=seed_id,
                     family=family, pool="native_multi", license=license, commercial_ok=ok,
                     fixed_role=fixed_role, latency_valid=latency_valid)


# ======================================================================================
# Parsers (pure)
# ======================================================================================

INSTRUCTRANS = "nayohan/llama3-instrucTrans-enko-8b"


def parse_wildguardmix_ko(rows, source, pool, fixed_role=None, license="ODC-BY", ok=True, origin="model"):
    """WildGuardMix-KO: columns prompt, prompt_ko, prompt_harm_label, adversarial, subcategory [R24].
    Emits a Korean AND an English record sharing one seed_id (so splits keep them together).
    Rows with a null/disputed harm label are dropped, never coerced to Safe; the Korean side is
    kept only if the translation passes mt_ok. Train labels are GPT-4 (origin=model), test labels
    are human [R24]."""
    from .quality import mt_ok
    out, unknown, bad_mt = [], set(), 0
    for i, r in enumerate(rows):
        hl = r.get("prompt_harm_label")
        if hl not in ("harmful", "unharmful"):
            continue
        label = "Unsafe" if hl == "harmful" else "Safe"
        sub = r.get("subcategory")
        cat = tx.WILDGUARD_SUBCAT.get(sub) or tx.keyword_category(sub) or "Unethical Acts"
        if label == "Unsafe" and sub not in tx.WILDGUARD_SUBCAT:
            unknown.add(sub)
        en, ko = r.get("prompt") or "", r.get("prompt_ko") or ""
        seed = f"wild:{text_key(en or ko)}"                  # shared with WildJailbreak rows of the same text
        fam = "single_adversarial" if r.get("adversarial") else "single"
        if en.strip():
            out.append(single_turn(source, en, label, cat, "en", seed, pool, license, ok, family=fam,
                                   fixed_role=fixed_role, label_origin=origin))
        if mt_ok(ko, en):
            out.append(single_turn(source, ko, label, cat, "ko", seed, pool, license, ok, family=fam,
                                   fixed_role=fixed_role, label_origin=origin, translated=True, mt_model=INSTRUCTRANS))
        elif ko:
            bad_mt += 1
    if unknown:
        print(f"  [{source}] unmapped WildGuard subcategories -> keyword/default: {sorted(map(str, unknown))[:10]}")
    if bad_mt:
        print(f"  [{source}] {bad_mt} Korean translations failed the MT sanity check (dropped)")
    return out


def parse_aegis2(rows, source="aegis2"):
    """Aegis 2.0: prompt, prompt_label (safe/unsafe), violated_categories; drop REDACTED [R34]."""
    out = []
    for r in rows:
        p = r.get("prompt")
        if not p or str(p).strip().upper() == "REDACTED":
            continue
        lab = str(r.get("prompt_label", "")).lower()
        if lab not in ("safe", "unsafe"):
            continue
        label = "Unsafe" if lab == "unsafe" else "Safe"
        cat = tx.map_category(r.get("violated_categories"), tx.AEGIS_CAT)
        out.append(single_turn(source, p, label, cat, "en", f"aegis:{r.get('id') or text_key(p)}",
                               "train_single", "CC-BY-4.0", True))
    return out


def parse_kguardrail_kmhas(rows, source="kmhas"):
    """K-MHaS via K-Guardrail JSONL (query, topic list) [R27]. Protected-attribute
    discrimination -> Unsafe; profanity/political-leaning only -> Controversial [D4]."""
    out = []
    for r in rows:
        q, topics = r.get("query"), set(r.get("topic") or [])
        if not q:
            continue
        if topics & tx.KMHAS_PROTECTED:
            label = "Unsafe"
        elif topics & tx.KMHAS_NONPROTECTED:
            label = "Controversial"
        else:
            label = "Safe"
        out.append(single_turn(source, q, label, "Unethical Acts", "ko", f"kmhas:{r.get('id')}", "comment",
                               "CC-BY-SA-4.0", True, family="comment"))
    return out


def parse_k_haters(rows, source="k_haters"):
    """K-HATERS labels: normal / offensive / L1_hate / L2_hate [R29]."""
    m = {"L2_hate": "Unsafe", "L1_hate": "Controversial", "offensive": "Controversial", "normal": "Safe"}
    out = []
    for i, r in enumerate(rows):
        lab = m.get(r.get("label"))
        if lab is None or not r.get("text"):
            continue
        out.append(single_turn(source, r["text"], lab, "Unethical Acts", "ko", f"khaters:{text_key(r['text'])}",
                               "comment", "CC-BY-4.0", True, family="comment"))
    return out


def parse_kguardrail_blocked(rows, source, *, blocked_label, blocked_cat, pool, license, ok,
                             fixed_role=None, family="single", topic_table=None, lang="ko"):
    """Generic K-Guardrail JSONL parser: 'blocked' True/False -> label [R27]."""
    out = []
    for r in rows:
        q = r.get("query")
        if not q or not str(q).strip():
            continue
        if r.get("blocked"):
            label = blocked_label
            cat = tx.map_category(r.get("topic"), topic_table, default=blocked_cat) if topic_table else blocked_cat
        else:
            label, cat = "Safe", None
        out.append(single_turn(source, q, label, cat, lang, f"{source}:{r.get('id') or text_key(q)}", pool,
                               license, ok, family=family, fixed_role=fixed_role))
    return out


def parse_kotox_cls(rows, source, fixed_role=None):
    """KOTOX-classification: text, label (1 = hate) [R25]. Obfuscated Korean toxic text."""
    out = []
    for r in rows:
        if r.get("text") is None or r.get("label") not in (0, 1):
            continue
        lab = "Unsafe" if int(r["label"]) == 1 else "Safe"
        out.append(single_turn(source, r["text"], lab, "Unethical Acts", "ko", f"kotox:{text_key(r['text'])}",
                               "comment" if fixed_role is None else "eval", "MIT", True,
                               family="obfuscated_native", fixed_role=fixed_role))
    return out


def parse_square_questions(rows, source, fixed_role=None):
    """SQuARe questions: sensitive-but-acceptable Korean questions -> Safe hard negatives [R30]."""
    out = []
    for r in rows:
        q = r.get("question")
        if q:
            out.append(single_turn(source, q, "Safe", None, "ko", f"square_q:{text_key(q)}",
                                   "train_single" if fixed_role is None else "eval", "MIT", True,
                                   family="benign_sensitive", fixed_role=fixed_role))
    return out


def parse_square_responses(rows, source="square_resp"):
    """SQuARe (question, response, acceptable?==1) pairs -> benign KO history material [R30]."""
    out = []
    for r in rows:
        if r.get("acceptable?") == 1 and r.get("question") and r.get("response"):
            msgs = [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["response"]}]
            out.append(make_conv(source=source, messages=msgs, label3=["Safe"], category=[None], lang="ko",
                                 seed_id=f"square_r:{text_key(r['question'])}", family="history", pool="history",
                                 license="MIT", commercial_ok=True))
    return out


def parse_koalpaca(rows, source="koalpaca"):
    """KoAlpaca v1.1a (instruction, output) -> benign KO history pairs [R42]."""
    out = []
    for r in rows:
        if r.get("instruction") and r.get("output"):
            msgs = [{"role": "user", "content": r["instruction"]}, {"role": "assistant", "content": r["output"]}]
            out.append(make_conv(source=source, messages=msgs, label3=["Safe"], category=[None], lang="ko",
                                 seed_id=f"koalpaca:{text_key(r['instruction'])}", family="history",
                                 pool="history", license="unspecified", commercial_ok=False))
    return out


def parse_ultrachat(rows, source, pool, fixed_role=None):
    """UltraChat 200k: messages [{role, content}] -> benign multi-turn EN (all user turns Safe) [R31]."""
    out = []
    for r in rows:
        msgs = _turns_from_sharegpt_or_chatml(r.get("messages") or [])
        if pool != "history":
            msgs = _truncate_to_last_user(msgs)
        n = sum(m["role"] == "user" for m in msgs)
        if n == 0:
            continue
        out.append(make_conv(source=source, messages=msgs, label3=["Safe"] * n, category=[None] * n, lang="en",
                             seed_id=f"ultrachat:{r.get('prompt_id') or text_key(msgs[0]['content'])}",
                             family="benign_native" if pool != "history" else "history", pool=pool,
                             license="MIT", commercial_ok=True, fixed_role=fixed_role))
    return out


def parse_orbench(rows, source, label, pool, fixed_role=None):
    """OR-Bench: prompt, category. 80k/hard-1k are safe-but-seemingly-toxic; 'toxic' is unsafe [R32]."""
    out = []
    for r in rows:
        p = r.get("prompt")
        if p:
            cat = tx.keyword_category(r.get("category")) or "Unethical Acts"
            out.append(single_turn(source, p, label, cat, "en", f"orbench:{text_key(p)}", pool, "CC-BY-4.0", True,
                                   family="benign_sensitive" if label == "Safe" else "single", fixed_role=fixed_role))
    return out


def parse_csrt(rows, source="csrt"):
    """CSRT: prompt (code-switched), original, harms [R26]. All harmful; eval only."""
    out = []
    for r in rows:
        if r.get("prompt"):
            cat = tx.map_category(r.get("harms"), tx.CSRT_CAT)
            out.append(single_turn(source, r["prompt"], "Unsafe", cat, "mixed",
                                   f"csrt:{text_key(r.get('original') or r['prompt'])}", "eval", "unverified",
                                   False, family="codeswitch_native", fixed_role="test"))
    return out


def parse_xstest(rows, source="xstest"):
    """XSTest: prompt + label ('safe'/'unsafe'); column names accepted defensively [R33]."""
    out = []
    for r in rows:
        p = r.get("prompt") or r.get("text")
        lab = str(r.get("label", r.get("type", ""))).lower()
        if not p:
            continue
        label = "Unsafe" if lab.startswith("unsafe") or lab.startswith("contrast") else "Safe"
        out.append(single_turn(source, p, label, "Unethical Acts", "en", f"xstest:{text_key(p)}", "eval",
                               "CC-BY-4.0", True, family="benign_sensitive" if label == "Safe" else "single",
                               fixed_role="test"))
    return out


def parse_safemt_1k(rows, cat_by_query: dict, source="safemt_1k"):
    """SafeMTData_1K: query_id + conversations (ActorAttack dialogues ending in refusals) [R21]."""
    out = []
    for r in rows:
        msgs = _turns_from_sharegpt_or_chatml(r.get("conversations") or [])
        qid = r.get("query_id")
        cat = cat_by_query.get(qid) or tx.keyword_category(r.get("plain_query")) or "Non-violent Illegal Acts"
        c = _native_attack(source, msgs, cat, "en", f"harmbench:{qid}", "MIT", True)
        if c:
            out.append(c)
    return out


def parse_xguard(rows, source="xguard"):
    """XGuard-Train: ShareGPT conversations, ~5 turns, attack-refusal pairs [R22].
    No behaviour id is published, so all XGuard rows are forced into guard_train (never
    stack/test) to rule out cross-split leakage between plans of one behaviour. [D9]"""
    out = []
    for i, r in enumerate(rows):
        msgs = _turns_from_sharegpt_or_chatml(r.get("conversations") or [])
        c = _native_attack(source, msgs, None, "en", f"xguard:{i}", "unverified", False, fixed_role="guard_train")
        if c:
            out.append(c)
    return out


def parse_mhj(rows, source="mhj"):
    """MHJ: columns message_0..message_N, each a JSON {'body','role'} [R23]. Eval only (CC-BY-NC-4.0)."""
    out = []
    for r in rows:
        keys = sorted((k for k in r if str(k).startswith("message_")), key=lambda k: int(str(k).split("_")[1]))
        turns = []
        for k in keys:
            v = r.get(k)
            if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
                continue
            try:
                d = json.loads(v) if isinstance(v, str) else v
            except json.JSONDecodeError:
                continue
            turns.append(d)
        msgs = _turns_from_sharegpt_or_chatml(turns)
        c = _native_attack(source, msgs, None, "en", f"mhj:{r.get('question_id', text_key(json.dumps(msgs)))}",
                           "CC-BY-NC-4.0", False, fixed_role="test", family="human_attack")
        if c:
            out.append(c)
    return out


DKTC_CAT = {"협박 대화": "Violent", "갈취 대화": "Non-violent Illegal Acts",
            "직장 내 괴롭힘 대화": "Unethical Acts", "기타 괴롭힘 대화": "Unethical Acts"}


def parse_dktc(df: pd.DataFrame, source="dktc"):
    """DKTC train.csv: class, conversation (utterances separated by newlines) [README of DKTC].
    Human-human threat dialogues; speaker A mapped to 'user'. Whole conversation harmful ->
    every user turn Unsafe; detection latency is not meaningful (latency_valid=False). [D10]"""
    out = []
    for _, r in df.iterrows():
        utts = [u.strip() for u in str(r["conversation"]).split("\n") if u.strip()]
        msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": u} for i, u in enumerate(utts)]
        c = _native_attack(source, msgs, DKTC_CAT.get(r["class"], "Unethical Acts"), "ko", f"dktc:{r['idx']}",
                           "CC-BY-NC-SA-4.0", False, fixed_role="test", family="human_dialogue",
                           latency_valid=False, all_unsafe=True)
        if c:
            out.append(c)
    return out


# ======================================================================================
# Parsers added in the dataset expansion (2026-09-29); column names verified with the
# Hugging Face dataset-viewer API / raw files (see DATASETS.md)
# ======================================================================================
WJB_MAP = {"vanilla_harmful": ("Unsafe", None), "adversarial_harmful": ("Unsafe", "Jailbreak"),
           "vanilla_benign": ("Safe", None), "adversarial_benign": ("Safe", None)}


def parse_wildjailbreak_ko(rows, source, pool, fixed_role=None):
    """iknow-lab/wildjailbreak-{train,eval}-ko: prompt_ko, vanilla, adversarial, data_type [R62].
    adversarial_harmful -> Unsafe/Jailbreak; vanilla_harmful -> Unsafe (category from the teacher);
    *_benign -> Safe (over-refusal negatives). seed = vanilla text, so every adversarial rewrite
    stays in the split of its source intent; the English side is emitted with the same seed."""
    from .quality import mt_ok
    out, bad = [], 0
    for r in rows:
        dt = r.get("data_type")
        if dt not in WJB_MAP:
            continue
        label, cat = WJB_MAP[dt]
        en = (r.get("adversarial") if dt.startswith("adversarial") else r.get("vanilla")) or ""
        van = r.get("vanilla") or en
        ko = r.get("prompt_ko") or ""
        seed = f"wild:{text_key(van or ko)}"
        fam = "single_adversarial" if dt.startswith("adversarial") else "single"
        if en.strip():
            out.append(single_turn(source, en, label, cat, "en", seed, pool, "ODC-BY", True, family=fam,
                                   fixed_role=fixed_role, label_origin="template"))
        if mt_ok(ko, en):
            out.append(single_turn(source, ko, label, cat, "ko", seed, pool, "ODC-BY", True, family=fam,
                                   fixed_role=fixed_role, label_origin="template", translated=True,
                                   mt_model=INSTRUCTRANS))
        elif ko:
            bad += 1
    if bad:
        print(f"  [{source}] {bad} Korean translations failed the MT sanity check (dropped)")
    return out


def parse_polyguardmix(rows, source="polyguardmix"):
    """PolyGuardMix: prompt, prompt_harm_label yes/no, prompt_safety_categories (S-codes),
    metadata{language, source} [R63]. Korean + English only. Labels are model ensembles tuned for
    recall (origin=model). Korean is machine-translated (TowerInstruct/NLLB) and cannot be linked
    back to its English source, so all rows are fixed to guard_train (never stack/test). [D9]"""
    out = []
    for r in rows:
        meta = r.get("metadata") or {}
        lang = {"Korean": "ko", "English": "en"}.get(meta.get("language"))
        p = r.get("prompt")
        hl = str(r.get("prompt_harm_label", "")).lower()
        if lang is None or not p or hl not in ("yes", "no"):
            continue
        label = "Unsafe" if hl == "yes" else "Safe"
        cat = tx.map_scodes(r.get("prompt_safety_categories")) if label == "Unsafe" else None
        out.append(single_turn(source, p, label, cat, lang, f"pgm:{text_key(p)}", "train_single", "unverified", False,
                               fixed_role="guard_train", label_origin="model", translated=(lang == "ko"),
                               mt_model="TowerInstruct-7B-v0.2/NLLB" if lang == "ko" else "none"))
    return out


def parse_polyguard_prompts(rows, source="polyguard_prompts"):
    """PolyGuardPrompts (test): language, id, prompt, prompt_label safe/unsafe, subcategory [R63].
    Korean = human-verified GPT-4o translation of WildGuardTest; English = WildGuardTest itself.
    seed = id, so both languages of one item stay together. Eval only."""
    out = []
    for r in rows:
        lang = {"Korean": "ko", "English": "en"}.get(r.get("language"))
        lab = str(r.get("prompt_label", "")).lower()
        if lang is None or lab not in ("safe", "unsafe") or not r.get("prompt"):
            continue
        label = "Unsafe" if lab == "unsafe" else "Safe"
        sub = r.get("subcategory")
        cat = tx.WILDGUARD_SUBCAT.get(sub) or tx.keyword_category(sub) or "Unethical Acts"
        out.append(single_turn(source, r["prompt"], label, cat, lang, f"wgtest:{r.get('id')}", "eval", "unverified", False,
                               fixed_role="test", label_origin="human", translated=(lang == "ko"),
                               mt_model="gpt-4o-agentic" if lang == "ko" else "none"))
    return out


def parse_toxic_chat(rows, source, fixed_role=None):
    """lmsys/toxic-chat: user_input, toxicity, jailbreaking, human_annotation [R64]. Real user
    prompts. jailbreaking -> Unsafe/Jailbreak; toxicity -> Unsafe (category from the teacher)."""
    out = []
    for r in rows:
        t = r.get("user_input")
        if not t:
            continue
        if int(r.get("jailbreaking") or 0) == 1:
            label, cat = "Unsafe", "Jailbreak"
        elif int(r.get("toxicity") or 0) == 1:
            label, cat = "Unsafe", None
        else:
            label, cat = "Safe", None
        out.append(single_turn(source, t, label, cat, "en", f"toxicchat:{r.get('conv_id') or text_key(t)}",
                               "eval" if fixed_role else "train_single", "CC-BY-NC-4.0", False, fixed_role=fixed_role,
                               label_origin="human" if r.get("human_annotation") else "model"))
    return out


def parse_binary_injection(rows, source, text_col, label_col, positive, fixed_role=None, license="unverified", ok=False):
    """Generic binary jailbreak / prompt-injection sets (jackhhao, xTRam1) [R65]."""
    out = []
    for r in rows:
        t = r.get(text_col)
        if not t or not str(t).strip():
            continue
        s = dominant_script_safe(t)
        if s is None:
            continue
        label = "Unsafe" if str(r.get(label_col)) == str(positive) else "Safe"
        out.append(single_turn(source, t, label, "Jailbreak", s, f"{source}:{text_key(t)}",
                               "eval" if fixed_role else "train_single", license, ok, fixed_role=fixed_role))
    return out


def dominant_script_safe(text: str):
    from .obfuscate import dominant_script
    d = dominant_script(str(text))
    return d if d in ("ko", "en") else None


def parse_itw_jailbreak(rows, source="itw_jailbreak"):
    """TrustAIRLab in-the-wild jailbreak templates (jailbreak=True configs) [R66]. Near-duplicate
    DAN variants are collapsed on the first 300 normalised characters."""
    from .schema import norm_text
    out, seen = [], set()
    for r in rows:
        t = r.get("prompt")
        if not t or r.get("jailbreak") is False:
            continue
        k = norm_text(t)[:300]
        if k in seen:
            continue
        seen.add(k)
        lang = dominant_script_safe(t)
        if lang is None:
            continue
        out.append(single_turn(source, t, "Unsafe", "Jailbreak", lang, f"itw:{text_key(k)}", "train_single",
                               "unverified", False, family="single_adversarial"))
    return out


DNA_MAP = {"Information Hazards": ("Unsafe", "PII"), "Malicious Uses": ("Unsafe", "Non-violent Illegal Acts"),
           "Discrimination, Exclusion, Toxicity, Hateful, Offensive": ("Unsafe", "Unethical Acts"),
           "Misinformation Harms": ("Unsafe", "Politically Sensitive Topics"),
           "Human-Chatbot Interaction Harms": ("Controversial", "Suicide & Self-Harm")}


def parse_do_not_answer(rows, source="do_not_answer"):
    """LibrAI/do-not-answer: question, risk_area, types_of_harm [R67]."""
    out = []
    for r in rows:
        q = r.get("question")
        lab = DNA_MAP.get(r.get("risk_area"))
        if not q or lab is None:
            continue
        cat = lab[1]
        th = str(r.get("types_of_harm", "")).lower()
        if "self-harm" not in th and "mental" not in th and lab[0] == "Controversial":
            cat = "Unethical Acts"
        out.append(single_turn(source, q, lab[0], cat, "en", f"dna:{r.get('id')}", "train_single", "Apache-2.0", True))
    return out


UNSMILE_GROUP = ["여성/가족", "남성", "성소수자", "인종/국적", "연령", "지역", "종교", "기타 혐오"]


def parse_unsmile(rows, source, fixed_role=None):
    """smilegate-ai/kor_unsmile: 문장 + multi-label ints [R68]. group hate -> Unsafe; only
    악플/욕설 or 개인지칭 -> Controversial; clean -> Safe; no positive column -> dropped. [D4]"""
    out = []
    for r in rows:
        t = r.get("문장")
        if not t:
            continue
        if any(int(r.get(c) or 0) for c in UNSMILE_GROUP):
            label = "Unsafe"
        elif int(r.get("악플/욕설") or 0) or int(r.get("개인지칭") or 0):
            label = "Controversial"
        elif int(r.get("clean") or 0):
            label = "Safe"
        else:
            continue
        out.append(single_turn(source, t, label, "Unethical Acts", "ko", f"unsmile:{text_key(t)}",
                               "eval" if fixed_role else "comment", "CC-BY-NC-ND-4.0", False, family="comment",
                               fixed_role=fixed_role))
    return out


def parse_beep(df: pd.DataFrame, source, fixed_role=None):
    """BEEP! (kocohub/korean-hate-speech labeled/*.tsv): comments, hate {hate, offensive, none},
    bias {gender, others, none} [R69]. hate -> Unsafe; offensive -> Controversial; biased but
    not hateful -> Controversial; otherwise Safe. [D4]"""
    out = []
    for r in df.itertuples(index=False):
        t = str(r.comments)
        if r.hate == "hate":
            label = "Unsafe"
        elif r.hate == "offensive" or r.bias in ("gender", "others"):
            label = "Controversial"
        else:
            label = "Safe"
        out.append(single_turn(source, t, label, "Unethical Acts", "ko", f"beep:{text_key(t)}",
                               "eval" if fixed_role else "comment", "CC-BY-SA-4.0", True, family="comment",
                               fixed_role=fixed_role))
    return out


def parse_curse(lines: list[str], source="curse"):
    """2runo/Curse-detection-data dataset.txt: 'text|label' (1 = profanity) [R70] -> Controversial."""
    out = []
    for line in lines:
        if "|" not in line:
            continue
        t, lab = line.rsplit("|", 1)
        if lab.strip() not in ("0", "1") or not t.strip():
            continue
        label = "Controversial" if lab.strip() == "1" else "Safe"
        out.append(single_turn(source, t.strip(), label, "Unethical Acts", "ko", f"curse:{text_key(t)}", "comment",
                               "MIT", True, family="comment"))
    return out


def parse_kosbi(rows, source="kosbi"):
    """KoSBi v2: context, sentence, sentence_label safe/unsafe [R30-KoSBi]. Statements about social
    groups (not requests): unsafe -> Controversial/Unethical Acts, safe -> Safe. [D4]"""
    out = []
    for r in rows:
        lab = r.get("sentence_label")
        if lab not in ("safe", "unsafe") or not r.get("sentence"):
            continue
        t = f"{r.get('context', '')} {r['sentence']}".strip()
        out.append(single_turn(source, t, "Controversial" if lab == "unsafe" else "Safe", "Unethical Acts", "ko",
                               f"kosbi:{text_key(t)}", "comment", "MIT", True, family="comment"))
    return out


def parse_sharegpt_benign(rows, source, lang="ko", license="MIT", ok=True):
    """Benign multi-turn chat (maywell/koVast: conversations [{from: user|gpt, value}]) [R71] ->
    history material. Screened later by the zero-shot guard (S02b) before use. [D40]"""
    out = []
    for i, r in enumerate(rows):
        msgs = _turns_from_sharegpt_or_chatml(r.get("conversations") or [])
        n = sum(m["role"] == "user" for m in msgs)
        if n == 0:
            continue
        out.append(make_conv(source=source, messages=msgs, label3=["Safe"] * n, category=[None] * n, lang=lang,
                             seed_id=f"{source}:{text_key(msgs[0]['content'])}", family="history", pool="history",
                             license=license, commercial_ok=ok, label_origin="model"))
    return out


def parse_xstest_ko(rows, source="xstest_ko"):
    """heegyu/XSTest_ko: prompt (EN), prompt_ko, label safe/unsafe, type [R33][R72]. Both languages,
    shared seed. Eval only (over-refusal: 250 safe prompts that look unsafe)."""
    out = []
    for i, r in enumerate(rows):
        lab = str(r.get("label", "")).lower()
        if lab not in ("safe", "unsafe"):
            continue
        label = "Unsafe" if lab == "unsafe" else "Safe"
        fam = "benign_sensitive" if label == "Safe" else "single"
        seed = f"xstest:{text_key(r.get('prompt') or r.get('prompt_ko'))}"
        for lang, col, tr in (("en", "prompt", False), ("ko", "prompt_ko", True)):
            if r.get(col):
                out.append(single_turn(source, r[col], label, "Unethical Acts", lang, seed, "eval", "CC-BY-4.0", True,
                                       family=fam, fixed_role="test", label_origin="human", translated=tr,
                                       mt_model="unknown" if tr else "none"))
    return out


def parse_jbb(rows, label, source="jbb"):
    """JailbreakBench JBB-Behaviors: Goal, Category (harmful / benign splits) [R73]."""
    return [single_turn(source, r["Goal"], label, tx.keyword_category(r.get("Category")) or "Unethical Acts", "en",
                        f"jbb:{text_key(r['Goal'])}", "eval", "MIT", True, fixed_role="test", label_origin="human")
            for r in rows if r.get("Goal")]


OAI_MOD = {"S": "Sexual Content or Sexual Acts", "S3": "Sexual Content or Sexual Acts", "H": "Unethical Acts",
           "H2": "Unethical Acts", "V": "Violent", "V2": "Violent", "HR": "Unethical Acts", "SH": "Suicide & Self-Harm"}


def parse_openai_mod(rows, source="openai_mod"):
    """mmathys/openai-moderation-api-evaluation: prompt + category columns; null = not annotated
    (never read as 0). Unsafe if any annotated column is 1; Safe only if all annotated are 0 [R74]."""
    out = []
    for r in rows:
        vals = {k: r.get(k) for k in OAI_MOD if r.get(k) is not None and not (isinstance(r.get(k), float) and np.isnan(r.get(k)))}
        if not vals or not r.get("prompt"):
            continue
        pos = [k for k, v in vals.items() if int(v) == 1]
        label = "Unsafe" if pos else "Safe"
        out.append(single_turn(source, r["prompt"], label, OAI_MOD[pos[0]] if pos else None, "en",
                               f"oaimod:{text_key(r['prompt'])}", "eval", "MIT", True, fixed_role="test",
                               label_origin="human"))
    return out


def parse_korset(rows, source="korset"):
    """datumo/KorSET (gated, CC-BY-NC-4.0): prompt, lv1/lv2/lv3_name [R75]. All rows are red-team
    prompts -> Unsafe; 'Toxic Language' -> Controversial. Eval only."""
    out = []
    for r in rows:
        p = r.get("prompt")
        if not p:
            continue
        names = " ".join(str(r.get(k, "")) for k in ("lv1_name", "lv2_name", "lv3_name"))
        label = "Controversial" if "toxic language" in names.lower() else "Unsafe"
        cat = tx.keyword_category(names) or ("PII" if "privacy" in names.lower() else "Unethical Acts")
        out.append(single_turn(source, p, label, cat, "ko", f"korset:{text_key(p)}", "eval", "CC-BY-NC-4.0", False,
                               fixed_role="test", label_origin="model"))
    return out


# ---- AIHub 558 (텍스트 윤리검증 데이터) -------------------------------------------------
AIHUB_SEVERITY = [("VIOLENCE", "Violent"), ("SEXUAL", "Sexual Content or Sexual Acts"),
                  ("CRIME", "Non-violent Illegal Acts"), ("HATE", "Unethical Acts"),
                  ("DISCRIMINATION", "Unethical Acts"), ("CENSURE", "Unethical Acts"), ("ABUSE", "Unethical Acts")]


def aihub_sentence_label(sent: dict) -> tuple[str, str | None]:
    """is_immoral=false -> Safe; only CENSURE/ABUSE with intensity < 2 -> Controversial; otherwise
    Unsafe with the most severe type as category [R76][D41]."""
    if not sent.get("is_immoral"):
        return "Safe", None
    types = {str(t).upper() for t in (sent.get("types") or [])} - {"IMMORAL_NONE"}
    if not types:
        return "Unknown", None                       # inconsistent row (immoral without a type)
    if types <= {"CENSURE", "ABUSE"} and float(sent.get("intensity") or 0) < 2:
        return "Controversial", "Unethical Acts"
    for t, cat in AIHUB_SEVERITY:
        if t in types:
            return "Unsafe", cat
    return "Unsafe", "Unethical Acts"


def parse_aihub_talksets(talksets: list[dict], source="aihub_558", max_prefixes: int | None = None) -> list[dict]:
    """One conversation per talkset: speakers alternate user/assistant from the first speaker; each
    user turn carries its sentence label. De-identified `text` is used, never `origin_text`.
    seed = talkset id (all prefixes of a talkset stay in one split). [D41]"""
    out = []
    for ts in talksets:
        sents = ts.get("sentences") or []
        if not sents:
            continue
        first = sents[0].get("speaker")
        msgs, labels, cats = [], [], []
        for s in sents:
            txt = str(s.get("text") or "").strip()
            if not txt:
                continue
            role = "user" if s.get("speaker") == first else "assistant"
            if msgs and msgs[-1]["role"] == role:
                msgs[-1]["content"] += " " + txt
                if role == "user":                       # merged user sentences: keep the worse label
                    l2, c2 = aihub_sentence_label(s)
                    order = {"Unsafe": 3, "Controversial": 2, "Unknown": 1, "Safe": 0}
                    if order[l2] > order[labels[-1]]:
                        labels[-1], cats[-1] = l2, c2
                continue
            msgs.append({"role": role, "content": txt})
            if role == "user":
                l, c = aihub_sentence_label(s)
                labels.append(l)
                cats.append(c)
        while msgs and msgs[-1]["role"] != "user":
            msgs.pop()
        n = sum(m["role"] == "user" for m in msgs)
        if n == 0:
            continue
        labels, cats = labels[:n], cats[:n]
        # a conversation ends at its first Unsafe turn ("unsafe as of turn t") [R19]
        if "Unsafe" in labels:
            k = labels.index("Unsafe")
            user_idx = [i for i, m in enumerate(msgs) if m["role"] == "user"]
            msgs = msgs[: user_idx[k] + 1]
            labels, cats = labels[: k + 1], cats[: k + 1]
        out.append(make_conv(source=source, messages=msgs, label3=labels, category=cats, lang="ko",
                             seed_id=f"aihub558:{ts.get('id')}", family="native_dialogue", pool="native_multi",
                             license="AIHub-policy", commercial_ok=False, label_origin="human"))
    return out


def read_aihub_dir(directory: Path) -> list[dict]:
    """Every *.json under the directory; each file is a talkset, a list of talksets, or {'data': [...]}. """
    talksets = []
    for p in sorted(Path(directory).rglob("*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(d, dict) and "sentences" in d:
            talksets.append(d)
        elif isinstance(d, list):
            talksets += [x for x in d if isinstance(x, dict) and "sentences" in x]
        elif isinstance(d, dict):
            for v in d.values():
                if isinstance(v, list):
                    talksets += [x for x in v if isinstance(x, dict) and "sentences" in x]
    return talksets


# ======================================================================================
# Registry
# ======================================================================================

@dataclass
class Source:
    name: str
    loader: Callable[[Config], list]
    role: str               # train | eval | history
    commercial_ok: bool
    enabled_by_default: bool = True
    note: str = ""
    origin: str = "human"           # default label_origin for rows the parser left "unset" [D37]
    translated_langs: tuple = ()     # languages in this source that are machine-translated
    mt_model: str = "unknown"
    gated: bool = False


def _gh(cfg: Config, repo_path: str) -> Path:
    return http_get(f"{cfg.github_raw}/{repo_path}", cfg.cache_dir / "github" / repo_path.replace("/", "__"))


def _kg(cfg, fname: str, sub="processed") -> list[dict]:
    rows, bad = read_jsonl_lenient(_gh(cfg, f"skan0779/korean-guardrail-dataset/main/data/{sub}/{fname}"))
    if bad:
        print(f"  skipped {bad} malformed JSON line(s) in {fname}")
    return rows


def _square_dir(cfg) -> Path:
    return git_clone("https://github.com/naver-ai/korean-safety-benchmarks", cfg.cache_dir / "git" / "korean-safety-benchmarks") / "data" / "SQuARe"


def _json_list(p: Path) -> list:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def L_wgm_train(cfg):
    rows = hf_rows("iknow-lab/wildguardmix-train-ko", None, "train", cfg.hf_token, cap=cfg.rows_per_source_cap, seed=cfg.seed)
    return parse_wildguardmix_ko(rows, "wgm_ko_train", "train_single")


def L_wgm_test(cfg):
    rows = hf_rows("iknow-lab/wildguardmix-test-ko", None, "test", cfg.hf_token)
    return parse_wildguardmix_ko(_cap(rows, cfg.rows_per_source_cap, cfg.seed), "wgm_ko_test", "eval", fixed_role="test", origin="human")


def L_aegis(cfg):
    return parse_aegis2(hf_rows("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", None, "train", cfg.hf_token,
                                cap=cfg.rows_per_source_cap, seed=cfg.seed))


def L_kmhas(cfg):
    return parse_kguardrail_kmhas(_cap(_kg(cfg, "KMHaS.jsonl"), cfg.rows_per_source_cap, cfg.seed))


def L_khaters(cfg):
    # train.jsonl and train_v2.jsonl have different columns, so the default config fails to build
    return parse_k_haters(hf_rows("humane-lab/K-HATERS", None, "train", cfg.hf_token, cap=cfg.rows_per_source_cap,
                                  seed=cfg.seed, columns=["text", "label"], data_files={"train": "train.jsonl"}))


def L_kold(cfg):
    rows = _cap(_kg(cfg, "KOLD.jsonl"), cfg.rows_per_source_cap, cfg.seed)
    return parse_kguardrail_blocked(rows, "kold", blocked_label="Controversial", blocked_cat="Unethical Acts",
                                    pool="comment", license="unknown", ok=False, family="comment")


def L_kotox_train(cfg):
    # levels are folders (data/<level>/<split>.csv), not builder configs; the repo's only
    # config ('default') globs every level together, and 'total' is the union of the other three
    return parse_kotox_cls(hf_rows("ssgyejin/KOTOX-classification", "total", "train", cfg.hf_token,
                                   cap=cfg.rows_per_source_cap, seed=cfg.seed,
                                   data_files={"train": "data/total/train.csv"}), "kotox_train")


def L_kotox_test(cfg):
    out = []
    for level in ("easy", "normal", "hard"):
        out += parse_kotox_cls(hf_rows("ssgyejin/KOTOX-classification", level, "test", cfg.hf_token,
                                       data_files={"test": f"data/{level}/test.csv"}),
                               f"kotox_test_{level}", fixed_role="test")
    return out


def L_pi_kr(cfg):
    rows = _cap(_kg(cfg, "prompt-injections-benchmark-kr.jsonl", "translated"), cfg.rows_per_source_cap, cfg.seed)
    return parse_kguardrail_blocked(rows, "pi_bench_kr", blocked_label="Unsafe", blocked_cat="Jailbreak",
                                    pool="train_single", license="CC-BY-NC-4.0 (qualifire original)", ok=False)


def L_redteam_kr(cfg):
    return parse_kguardrail_blocked(_kg(cfg, "llm-red-teaming-dataset-kr.jsonl", "translated"), "redteam_kr",
                                    blocked_label="Unsafe", blocked_cat="Unethical Acts", pool="eval", license="MIT",
                                    ok=True, fixed_role="test", topic_table=tx.REDTEAM_KR_CAT)


def L_raccoon_kr(cfg):
    return parse_kguardrail_blocked(_kg(cfg, "raccoonbench-kr.jsonl", "translated"), "raccoon_kr",
                                    blocked_label="Unsafe", blocked_cat="Jailbreak", pool="eval",
                                    license="GPL-3.0 (original)", ok=False, fixed_role="test")


def L_apeach(cfg):
    return parse_kguardrail_blocked(_kg(cfg, "APEACH.jsonl"), "apeach", blocked_label="Unsafe",
                                    blocked_cat="Unethical Acts", pool="eval", license="CC-BY-SA-4.0", ok=True,
                                    fixed_role="test", family="comment")


def L_square_q(cfg):
    return parse_square_questions(_cap(_json_list(_square_dir(cfg) / "question_train.json"),
                                       cfg.rows_per_source_cap, cfg.seed), "square_q")


def L_square_test(cfg):
    return parse_square_questions(_cap(_json_list(_square_dir(cfg) / "question_test_ood.json"),
                                       cfg.rows_per_source_cap, cfg.seed), "square_test_ood", fixed_role="test")


def L_square_resp(cfg):
    return parse_square_responses(_cap(_json_list(_square_dir(cfg) / "response_train.json"),
                                       cfg.rows_per_source_cap, cfg.seed))


def L_koalpaca(cfg):
    return parse_koalpaca(hf_rows("beomi/KoAlpaca-v1.1a", None, "train", cfg.hf_token, cap=cfg.rows_per_source_cap, seed=cfg.seed))


def L_ultra_hist(cfg):
    rows = hf_rows("HuggingFaceH4/ultrachat_200k", None, "train_sft", cfg.hf_token, cap=cfg.ultrachat_max_conversations,
                   seed=cfg.seed, columns=["prompt_id", "messages"])
    return parse_ultrachat(rows, "ultrachat_train", "history")


def L_ultra_test(cfg):
    rows = hf_rows("HuggingFaceH4/ultrachat_200k", None, "test_sft", cfg.hf_token, cap=min(2000, cfg.rows_per_source_cap),
                   seed=cfg.seed, columns=["prompt_id", "messages"])
    return parse_ultrachat(rows, "ultrachat_test", "eval", fixed_role="test")


def L_orbench80k(cfg):
    return parse_orbench(hf_rows("bench-llm/or-bench", "or-bench-80k", "train", cfg.hf_token, cap=cfg.rows_per_source_cap,
                                 seed=cfg.seed), "orbench_80k", "Safe", "train_single")


def L_orbench_hard(cfg):
    return parse_orbench(hf_rows("bench-llm/or-bench", "or-bench-hard-1k", "train", cfg.hf_token)[: cfg.rows_per_source_cap],
                         "orbench_hard", "Safe", "eval", fixed_role="test")


def L_orbench_toxic(cfg):
    return parse_orbench(hf_rows("bench-llm/or-bench", "or-bench-toxic", "train", cfg.hf_token)[: cfg.rows_per_source_cap],
                         "orbench_toxic", "Unsafe", "eval", fixed_role="test")


def L_csrt(cfg):
    return parse_csrt(hf_rows("walledai/CSRT", None, "train", cfg.hf_token))


def L_xstest(cfg):
    return parse_xstest(hf_rows("walledai/XSTest", None, "test", cfg.hf_token))


def L_safemt(cfg):
    a600 = hf_rows("SafeMTData/SafeMTData", "Attack_600", "Attack_600", cfg.hf_token)
    cat_by_q = {r["query_id"]: tx.HARMBENCH_CAT.get(r.get("category"), tx.keyword_category(r.get("category")))
                for r in a600}
    rows = hf_rows("SafeMTData/SafeMTData", "SafeMTData_1K", "SafeMTData_1K", cfg.hf_token)
    return parse_safemt_1k(_cap(rows, cfg.rows_per_source_cap, cfg.seed), cat_by_q)


def L_xguard(cfg):
    return parse_xguard(hf_rows("marslabucla/XGuard-Train", None, "train", cfg.hf_token,
                                cap=cfg.xguard_max_conversations, seed=cfg.seed))


def L_mhj(cfg):
    """MHJ ships CSV file(s); if datasets cannot build one table from them (e.g. differing
    columns), read each CSV directly from the Hub [R44]."""
    try:
        return parse_mhj(hf_rows("ScaleAI/mhj", None, "train", cfg.hf_token))
    except FileNotFoundError:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"  mhj: load_dataset failed ({type(e).__name__}); reading CSV files directly")
        from huggingface_hub import HfApi, hf_hub_download
        files = [f for f in HfApi().list_repo_files("ScaleAI/mhj", repo_type="dataset", token=cfg.hf_token)
                 if f.endswith(".csv")]
        rows = []
        for f in files:
            rows += pd.read_csv(hf_hub_download("ScaleAI/mhj", f, repo_type="dataset", token=cfg.hf_token)).to_dict("records")
        return parse_mhj(rows)


def L_dktc(cfg):
    d = git_clone("https://github.com/tunib-ai/DKTC", cfg.cache_dir / "git" / "DKTC")
    df = pd.read_csv(d / "data" / "train.csv")
    return parse_dktc(df.sample(min(len(df), cfg.rows_per_source_cap), random_state=cfg.seed))


def L_kotox_instr(cfg):
    """KoTox (39K instruction/refusal pairs) [R27-list]. Disabled by default: many rows ask for
    the model's political opinion, which is not unsafe under the Qwen3Guard/MLCommons policy,
    so a blanket 'Unsafe' label would teach over-refusal. [D11]"""
    p = _gh(cfg, "byunsj/KoTox-Korean-Toxic-Instruction-Dataset/main/KoTox_dataset.csv")
    df = pd.read_csv(p).dropna(subset=["instruction"])
    rows = _cap(df.to_dict("records"), cfg.rows_per_source_cap, cfg.seed)
    return [single_turn("kotox_instr", r["instruction"], "Controversial", "Unethical Acts", "ko",
                        f"kotox_i:{text_key(r['instruction'])}", "train_single", "MIT", True) for r in rows]


def L_wjb_train(cfg):
    rows = hf_rows("iknow-lab/wildjailbreak-train-ko", None, "train", cfg.hf_token, cap=cfg.rows_per_source_cap,
                   seed=cfg.seed, columns=["prompt_ko", "vanilla", "adversarial", "data_type"])
    return parse_wildjailbreak_ko(rows, "wjb_ko_train", "train_single")


def L_wjb_eval(cfg):
    rows = hf_rows("iknow-lab/wildjailbreak-eval-ko", None, "train", cfg.hf_token)
    return parse_wildjailbreak_ko(rows, "wjb_ko_eval", "eval", fixed_role="test")


def L_polyguardmix(cfg):
    def ko_en(batch):
        return [(m or {}).get("language") in ("Korean", "English") for m in batch["metadata"]]
    rows = hf_rows("ToxicityPrompts/PolyGuardMix", None, "train", cfg.hf_token, cap=cfg.rows_per_source_cap, seed=cfg.seed,
                   columns=["prompt", "prompt_harm_label", "prompt_safety_categories", "metadata"], where=ko_en)
    return parse_polyguardmix(rows)


def L_polyguard_prompts(cfg):
    rows = hf_rows("ToxicityPrompts/PolyGuardPrompts", None, "test", cfg.hf_token,
                   columns=["id", "language", "prompt", "prompt_label", "subcategory"],
                   where=lambda b: [l in ("Korean", "English") for l in b["language"]])
    return parse_polyguard_prompts(_cap(rows, cfg.rows_per_source_cap, cfg.seed))


def L_toxic_chat_train(cfg):
    return parse_toxic_chat(hf_rows("lmsys/toxic-chat", "toxicchat0124", "train", cfg.hf_token,
                                    cap=cfg.rows_per_source_cap, seed=cfg.seed), "toxic_chat_train")


def L_toxic_chat_test(cfg):
    return parse_toxic_chat(hf_rows("lmsys/toxic-chat", "toxicchat0124", "test", cfg.hf_token,
                                    cap=cfg.rows_per_source_cap, seed=cfg.seed), "toxic_chat_test", fixed_role="test")


def L_jailbreak_cls(cfg):
    return parse_binary_injection(hf_rows("jackhhao/jailbreak-classification", None, "train", cfg.hf_token),
                                  "jailbreak_cls", "prompt", "type", "jailbreak")


def L_safeguard_pi(cfg):
    return parse_binary_injection(hf_rows("xTRam1/safe-guard-prompt-injection", None, "train", cfg.hf_token,
                                          cap=min(cfg.rows_per_source_cap, 3000), seed=cfg.seed),
                                  "safeguard_pi", "text", "label", 1)


def L_safeguard_pi_test(cfg):
    return parse_binary_injection(hf_rows("xTRam1/safe-guard-prompt-injection", None, "test", cfg.hf_token),
                                  "safeguard_pi_test", "text", "label", 1, fixed_role="test")


def L_itw(cfg):
    rows = []
    for c in ("jailbreak_2023_05_07", "jailbreak_2023_12_25"):
        rows += hf_rows("TrustAIRLab/in-the-wild-jailbreak-prompts", c, "train", cfg.hf_token, columns=["prompt", "jailbreak"])
    return parse_itw_jailbreak(_cap(rows, cfg.rows_per_source_cap, cfg.seed))


def L_dna(cfg):
    return parse_do_not_answer(hf_rows("LibrAI/do-not-answer", None, "train", cfg.hf_token,
                                       columns=["id", "question", "risk_area", "types_of_harm"]))


def L_unsmile(cfg):
    return parse_unsmile(hf_rows("smilegate-ai/kor_unsmile", None, "train", cfg.hf_token, cap=cfg.rows_per_source_cap,
                                 seed=cfg.seed), "unsmile")


def L_unsmile_valid(cfg):
    return parse_unsmile(hf_rows("smilegate-ai/kor_unsmile", None, "valid", cfg.hf_token), "unsmile_valid", fixed_role="test")


def L_beep(cfg):
    df = pd.read_csv(_gh(cfg, "kocohub/korean-hate-speech/master/labeled/train.tsv"), sep="\t")
    return parse_beep(df.sample(min(len(df), cfg.rows_per_source_cap), random_state=cfg.seed), "beep")


def L_beep_dev(cfg):
    return parse_beep(pd.read_csv(_gh(cfg, "kocohub/korean-hate-speech/master/labeled/dev.tsv"), sep="\t"), "beep_dev",
                      fixed_role="test")


def L_curse(cfg):
    lines = Path(_gh(cfg, "2runo/Curse-detection-data/master/dataset.txt")).read_text(encoding="utf-8").splitlines()
    return parse_curse(_cap(lines, cfg.rows_per_source_cap, cfg.seed))


def L_kosbi(cfg):
    d = git_clone("https://github.com/naver-ai/korean-safety-benchmarks", cfg.cache_dir / "git" / "korean-safety-benchmarks")
    rows = _json_list(d / "data" / "KoSBi" / "kosbi_v2_train.json")
    return parse_kosbi(_cap(rows, min(cfg.rows_per_source_cap, 10000), cfg.seed))


def L_kovast(cfg):
    rows = hf_rows("maywell/koVast", None, "train", cfg.hf_token, cap=cfg.kovast_max_conversations, seed=cfg.seed)
    return parse_sharegpt_benign(rows, "kovast")


def L_xstest_ko(cfg):
    return parse_xstest_ko(hf_rows("heegyu/XSTest_ko", None, "test", cfg.hf_token))


def L_jbb(cfg):
    return (parse_jbb(hf_rows("JailbreakBench/JBB-Behaviors", "behaviors", "harmful", cfg.hf_token), "Unsafe") +
            parse_jbb(hf_rows("JailbreakBench/JBB-Behaviors", "behaviors", "benign", cfg.hf_token), "Safe"))


def L_openai_mod(cfg):
    return parse_openai_mod(hf_rows("mmathys/openai-moderation-api-evaluation", None, "train", cfg.hf_token))


def L_korset(cfg):
    """KorSET ships one split per category; load them all [R75]."""
    from datasets import get_dataset_split_names
    fx = os.environ.get("OKG_FIXTURE_DIR")
    splits = ["all"] if fx else get_dataset_split_names("datumo/KorSET", token=cfg.hf_token)
    rows = []
    for sp_ in splits:
        rows += hf_rows("datumo/KorSET", None, sp_, cfg.hf_token)
    return parse_korset(rows)


def L_aihub558(cfg):
    """AIHub 558 text-ethics data, downloaded on YOUR server with aihubshell (see okg/aihub.py)."""
    d = cfg.aihub_dir / str(cfg.aihub_datasetkey)
    if os.environ.get("OKG_FIXTURE_DIR"):                      # offline tests only
        d = Path(os.environ["OKG_FIXTURE_DIR"]) / "aihub" / str(cfg.aihub_datasetkey)
    talksets = read_aihub_dir(d)
    if not talksets:
        raise FileNotFoundError(f"no AIHub 558 JSON under {d} — run the AIHub download cell first")
    import random as _r
    if len(talksets) > cfg.aihub_max_talksets:
        talksets = _r.Random(cfg.seed).sample(talksets, cfg.aihub_max_talksets)
    return parse_aihub_talksets(talksets)


REGISTRY: list[Source] = [
    # ---- training material: Korean ----
    Source("wgm_ko_train", L_wgm_train, "train", True, origin="model", note="KO+EN pairs, WildGuardMix-KO [R24]"),
    Source("wjb_ko_train", L_wjb_train, "train", True, origin="template", note="KO+EN WildJailbreak [R62]"),
    Source("polyguardmix", L_polyguardmix, "train", False, origin="model", note="KO+EN slice, guard_train only [R63]"),
    Source("aihub_558", L_aihub558, "train", False, origin="human", note="native KO multi-turn, needs aihubshell [R76]"),
    Source("kmhas", L_kmhas, "train", True, note="KO hate comments [R27]"),
    Source("k_haters", L_khaters, "train", True, note="KO hate comments [R29]"),
    Source("kold", L_kold, "train", False, note="KO offensive comments, license unknown [R28]"),
    Source("beep", L_beep, "train", True, note="KO hate/offensive/bias comments [R69]"),
    Source("unsmile", L_unsmile, "train", False, note="KO multi-label hate, CC-BY-NC-ND [R68]"),
    Source("curse", L_curse, "train", True, note="KO profanity -> Controversial [R70]"),
    Source("kosbi", L_kosbi, "train", True, note="KO social-bias statements -> Controversial [R30]"),
    Source("kotox_train", L_kotox_train, "train", True, origin="template", note="KO obfuscated toxic [R25]"),
    Source("pi_bench_kr", L_pi_kr, "train", False, translated_langs=("ko",), note="KO prompt injection, CC-BY-NC [R27]"),
    Source("square_q", L_square_q, "train", True, note="KO sensitive-but-safe questions [R30]"),
    # ---- training material: English ----
    Source("aegis2", L_aegis, "train", True, note="EN single-turn, human labels [R34]"),
    Source("toxic_chat_train", L_toxic_chat_train, "train", False, note="EN real user prompts, CC-BY-NC [R64]"),
    Source("jailbreak_cls", L_jailbreak_cls, "train", False, origin="template", note="EN jailbreak vs benign [R65]"),
    Source("safeguard_pi", L_safeguard_pi, "train", False, origin="template", note="EN synthetic injection, capped [R65]"),
    Source("itw_jailbreak", L_itw, "train", False, origin="template", note="EN in-the-wild jailbreak templates [R66]"),
    Source("do_not_answer", L_dna, "train", True, note="EN should-refuse questions [R67]"),
    Source("orbench_80k", L_orbench80k, "train", True, origin="model", note="EN over-refusal hard negatives [R32]"),
    Source("safemt_1k", L_safemt, "train", True, origin="template", note="EN multi-turn ActorAttack dialogues [R21]"),
    Source("xguard", L_xguard, "train", False, origin="template", gated=True, note="EN multi-turn X-Teaming [R22]"),
    Source("kotox_instr", L_kotox_instr, "train", True, enabled_by_default=False, origin="template", note="see D11"),
    # ---- benign history material ----
    Source("ultrachat_train", L_ultra_hist, "history", True, origin="model", note="EN benign multi-turn [R31]"),
    Source("kovast", L_kovast, "history", True, origin="model", note="KO benign multi-turn, screened in S02b [R71]"),
    Source("koalpaca", L_koalpaca, "history", False, note="KO benign pairs, license unspecified [R42]"),
    Source("square_resp", L_square_resp, "history", True, note="KO acceptable responses [R30]"),
    # ---- evaluation only ----
    Source("polyguard_prompts", L_polyguard_prompts, "eval", False, note="KO+EN WildGuardTest (human-verified MT) [R63]"),
    Source("wgm_ko_test", L_wgm_test, "eval", True, enabled_by_default=False, note="superseded by polyguard_prompts"),
    Source("wjb_ko_eval", L_wjb_eval, "eval", True, origin="template", note="KO+EN adversarial eval [R62]"),
    Source("xstest_ko", L_xstest_ko, "eval", True, note="KO+EN over-refusal [R72]"),
    Source("toxic_chat_test", L_toxic_chat_test, "eval", False, note="EN real users [R64]"),
    Source("safeguard_pi_test", L_safeguard_pi_test, "eval", False, origin="template"),
    Source("jbb", L_jbb, "eval", True, note="EN JailbreakBench [R73]"),
    Source("openai_mod", L_openai_mod, "eval", True, note="EN OpenAI moderation eval [R74]"),
    Source("korset", L_korset, "eval", False, origin="model", gated=True, note="KO native red-team, gated, NC [R75]"),
    Source("unsmile_valid", L_unsmile_valid, "eval", False),
    Source("beep_dev", L_beep_dev, "eval", True),
    Source("kotox_test", L_kotox_test, "eval", True, origin="template"),
    Source("csrt", L_csrt, "eval", False),
    Source("redteam_kr", L_redteam_kr, "eval", True, translated_langs=("ko",)),
    Source("raccoon_kr", L_raccoon_kr, "eval", False, translated_langs=("ko",)),
    Source("apeach", L_apeach, "eval", True),
    Source("square_test_ood", L_square_test, "eval", True),
    Source("orbench_hard", L_orbench_hard, "eval", True, origin="model"),
    Source("orbench_toxic", L_orbench_toxic, "eval", True, origin="model"),
    Source("xstest", L_xstest, "eval", True, enabled_by_default=False, gated=True, note="superseded by xstest_ko"),
    Source("ultrachat_test", L_ultra_test, "eval", True, origin="model"),
    Source("mhj", L_mhj, "eval", False, gated=True, note="gated, CC-BY-NC-4.0 [R23]"),
    Source("dktc", L_dktc, "eval", False, note="CC-BY-NC-SA [DKTC README]"),
]


def load_custom(spec: dict, cfg: Config) -> list:
    """Previous-project data (e.g. KoreanGuardrail, wildjailbreak-train-ko) or AIHub exports
    converted to a flat file. spec keys: name, path, text_col, label_col, lang, [seed_col],
    [unsafe_values], [category_col], [role: train|eval]. [D12]"""
    p = Path(spec["path"])
    df = pd.read_parquet(p) if p.suffix == ".parquet" else (
        pd.read_json(p, lines=True) if p.suffix == ".jsonl" else pd.read_csv(p))
    unsafe_vals = set(map(str, spec.get("unsafe_values", ["1", "unsafe", "Unsafe", "harmful", "True"])))
    fixed_role = "test" if spec.get("role") == "eval" else None
    out = []
    for i, r in df.iterrows():
        txt = r.get(spec["text_col"])
        if not isinstance(txt, str) or not txt.strip():
            continue
        label = "Unsafe" if str(r.get(spec["label_col"])) in unsafe_vals else "Safe"
        cat = tx.keyword_category(r.get(spec.get("category_col", ""), "")) or spec.get("default_category", "Jailbreak")
        seed = f"{spec['name']}:{r[spec['seed_col']]}" if spec.get("seed_col") else f"{spec['name']}:{text_key(txt)}"
        out.append(single_turn(spec["name"], txt, label, cat, spec.get("lang", "ko"), seed,
                               "eval" if fixed_role else "train_single", spec.get("license", "own"), True,
                               fixed_role=fixed_role))
    return out

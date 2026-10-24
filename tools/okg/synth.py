"""Deterministic multi-turn conversation synthesis (no LLM needed).

Generators (each creates conversations inside ONE role, see splits.py [D14]):
  late_insertion  benign history (1..k turns) + harmful turn          [R36][R19]
  codeswitch      history in one language, harmful turn in the other  [R26]
  obfuscated      late_insertion with an obfuscated harmful turn      [R25]
  benign_mirror   benign history + benign-but-sensitive final turn    (topic-matched hard negatives) [R2][R19]
  benign_native   genuine benign multi-turn dialogue, cut at a user turn [R31]

Per-turn labels: history user turns = Safe, attack turn = Unsafe; conversations end at the
attack turn (label is "unsafe as of turn t") [R19]. Every synthesised conversation
inherits the seed_id of its final user turn so group splits stay leak-free [R11].
Crescendo/ActorAttack-style gradual escalation is covered by the native SafeMTData / XGuard
dialogues (and their Korean translations) rather than synthesised here [R20][R21][R22].
"""
from __future__ import annotations

import json
import random

import pandas as pd

from .obfuscate import obfuscate
from .schema import make_conv

MAX_CHARS = 1500   # per history message, keeps prompts inside max_seq_len [D16]


def _clip(s: str) -> str:
    return s if len(s) <= MAX_CHARS else s[:MAX_CHARS] + " …"


class Pools:
    """Material of one role, split by language."""

    def __init__(self, role_df: pd.DataFrame):
        self.harm = {"ko": [], "en": []}      # (text, category, seed_id, source)
        self.benign = {"ko": [], "en": []}    # benign-but-sensitive single turns
        self.hist = {"ko": [], "en": []}      # list of (user, assistant) pair lists
        self.pairs = {}                       # seed_id -> {lang: (text, cat)} for codeswitch
        for r in role_df.itertuples(index=False):
            msgs = json.loads(r.messages)
            labels = json.loads(r.label3)
            cats = json.loads(r.category)
            if r.lang not in ("ko", "en"):
                continue
            if r.pool == "history":
                pairs = [(msgs[i]["content"], msgs[i + 1]["content"]) for i in range(0, len(msgs) - 1, 2)
                         if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant"]
                if pairs:
                    self.hist[r.lang].append(pairs)
            elif r.pool == "train_single" and len(msgs) == 1:
                text = msgs[0]["content"]
                if labels[0] == "Unsafe":
                    self.harm[r.lang].append((text, cats[0], r.seed_id, r.source))
                    self.pairs.setdefault(r.seed_id, {})[r.lang] = (text, cats[0])
                elif labels[0] == "Safe":
                    self.benign[r.lang].append((text, None, r.seed_id, r.source))

    def history(self, lang: str, k: int, rng: random.Random) -> list[dict]:
        """k user/assistant pairs; consecutive pairs from one dialogue when possible."""
        out = []
        guard = 0
        while len(out) < k and self.hist[lang] and guard < 50:
            guard += 1
            conv = rng.choice(self.hist[lang])
            start = rng.randrange(len(conv))
            for u, a in conv[start:start + (k - len(out))]:
                out += [(u, a)]
        msgs = []
        for u, a in out[:k]:
            msgs += [{"role": "user", "content": _clip(u)}, {"role": "assistant", "content": _clip(a)}]
        return msgs


def _conv(source, family, lang, history, final_text, final_label, final_cat, seed_id, role):
    n_hist = sum(m["role"] == "user" for m in history)
    msgs = history + [{"role": "user", "content": final_text}]
    return make_conv(source=source, messages=msgs, label3=["Safe"] * n_hist + [final_label],
                     category=[None] * n_hist + [final_cat], lang=lang, seed_id=seed_id, family=family,
                     pool="synth", license="derived", commercial_ok=True, fixed_role=role, label_origin="derived")


def generate(role: str, role_df: pd.DataFrame, family: str, n: int, cfg, rng: random.Random) -> list[dict]:
    P = Pools(role_df)
    out = []
    lo, hi = cfg.synth_min_history, cfg.synth_max_history
    for i in range(n):
        lang = ("ko", "en")[i % 2]
        k = rng.randint(lo, hi)
        if family == "late_insertion":
            if not P.harm[lang] or not P.hist[lang]:
                continue
            text, cat, seed, src = rng.choice(P.harm[lang])
            out.append(_conv(f"synth_{family}", family, lang, P.history(lang, k, rng), text, "Unsafe", cat, seed, role))
        elif family == "obfuscated":
            if not P.harm[lang] or not P.hist[lang]:
                continue
            text, cat, seed, src = rng.choice(P.harm[lang])
            ob, _ = obfuscate(text, lang, rng)
            out.append(_conv(f"synth_{family}", family, lang, P.history(lang, k, rng), ob, "Unsafe", cat, seed, role))
        elif family == "codeswitch":
            hist_lang, atk_lang = lang, ("en" if lang == "ko" else "ko")
            both = [s for s, d in P.pairs.items() if atk_lang in d]
            if not both or not P.hist[hist_lang]:
                continue
            seed = rng.choice(both)
            text, cat = P.pairs[seed][atk_lang]
            out.append(_conv(f"synth_{family}", family, "mixed", P.history(hist_lang, k, rng), text, "Unsafe", cat,
                             seed, role))
        elif family == "benign_mirror":
            if not P.benign[lang] or not P.hist[lang]:
                continue
            text, _, seed, src = rng.choice(P.benign[lang])
            # Mirror the attack generators' surface transformations on benign turns, so that
            # obfuscation / language switching alone never identifies an attack
            # (paired surface-similar negatives, Qwen3Guard report §3 [R2]) [D26].
            u, hist_lang, conv_lang = rng.random(), lang, lang
            if u < 0.25:
                text, _ = obfuscate(text, lang, rng)
            elif u < 0.5:
                hist_lang = "en" if lang == "ko" else "ko"
                conv_lang = "mixed" if P.hist[hist_lang] else lang
                hist_lang = hist_lang if P.hist[hist_lang] else lang
            out.append(_conv(f"synth_{family}", family, conv_lang, P.history(hist_lang, k, rng), text, "Safe", None,
                             seed, role))
        elif family == "benign_native":
            if not P.hist[lang]:
                continue
            hist = P.history(lang, k + 1, rng)
            if len(hist) < 2:
                continue
            final = hist[-2]["content"]
            seed = f"benign_native:{role}:{lang}:{i}"
            out.append(_conv(f"synth_{family}", family, lang, hist[:-2], final, "Safe", None, seed, role))
        else:
            raise ValueError(family)
    return out


FAMILIES = ("late_insertion", "codeswitch", "obfuscated", "benign_mirror", "benign_native")

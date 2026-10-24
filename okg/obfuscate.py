"""Text obfuscation used to synthesise evasive attack turns, and script statistics used as
LightGBM features.

Korean rules are simplified versions of KOTOX transformation families (jamo splitting,
spacing, visually similar characters) [R25]; character-level facts come from the Unicode
Standard: Hangul syllable decomposition (Ch. 3.12) and compatibility jamo, NFKC (UAX #15),
and invisible/confusable characters (UTS #39) [R35].
"""
from __future__ import annotations

import random
import unicodedata

# --- Unicode Hangul constants (Unicode Standard §3.12 "Conjoining Jamo Behavior") [R35] ---
S_BASE, L_COUNT, V_COUNT, T_COUNT = 0xAC00, 19, 21, 28
N_COUNT = V_COUNT * T_COUNT
S_COUNT = L_COUNT * N_COUNT
# Compatibility jamo (U+3131..U+318E) for the leading consonants, vowels and trailing consonants.
_L = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_V = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_T = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ",
      "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
assert len(_L) == L_COUNT and len(_V) == V_COUNT and len(_T) == T_COUNT

ZERO_WIDTH = ["​", "‌", "‍", "⁠"]  # default-ignorable, invisible [R35]
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}


def is_hangul_syllable(ch: str) -> bool:
    return S_BASE <= ord(ch) < S_BASE + S_COUNT


def decompose_syllable(ch: str) -> str:
    """'한' -> 'ㅎㅏㄴ' using the arithmetic decomposition of §3.12. [R35]"""
    s = ord(ch) - S_BASE
    l, v, t = s // N_COUNT, (s % N_COUNT) // T_COUNT, s % T_COUNT
    return _L[l] + _V[v] + _T[t]


# ------------------------------- rules ------------------------------------------------
def rule_jamo_split(text: str, rng: random.Random, p: float = 0.3) -> str:
    return "".join(decompose_syllable(c) if is_hangul_syllable(c) and rng.random() < p else c for c in text)


def rule_spacing(text: str, rng: random.Random, p: float = 0.3) -> str:
    return "".join(c + (" " if c.strip() and rng.random() < p else "") for c in text)


def rule_zero_width(text: str, rng: random.Random, p: float = 0.3) -> str:
    return "".join(c + (rng.choice(ZERO_WIDTH) if c.strip() and rng.random() < p else "") for c in text)


def rule_fullwidth(text: str, rng: random.Random, p: float = 0.5) -> str:
    """ASCII -> fullwidth forms (U+FF01..FF5E); NFKC maps them back [R35]."""
    return "".join(chr(ord(c) - 0x21 + 0xFF01) if "!" <= c <= "~" and rng.random() < p else c for c in text)


def rule_leet(text: str, rng: random.Random, p: float = 0.5) -> str:
    return "".join(LEET.get(c.lower(), c) if rng.random() < p else c for c in text)


KO_RULES = [rule_jamo_split, rule_spacing, rule_zero_width, rule_fullwidth]
EN_RULES = [rule_leet, rule_spacing, rule_zero_width, rule_fullwidth]


def obfuscate(text: str, lang: str, rng: random.Random, n_rules: int | None = None) -> tuple[str, list[str]]:
    """Apply 2-4 random rules (KOTOX difficulty levels use 2/3/4 rules) [R25]."""
    rules = KO_RULES if lang == "ko" else EN_RULES
    n = n_rules or rng.choice([2, 3, 4])
    chosen = rng.sample(rules, min(n, len(rules)))
    for r in chosen:
        text = r(text, rng)
    return text, [r.__name__ for r in chosen]


# ------------------------------- statistics (features) --------------------------------
def script_stats(text: str) -> dict:
    """Character-class ratios for code-switching / obfuscation features. [R35][D13]"""
    n = max(len(text), 1)
    hangul = sum(is_hangul_syllable(c) for c in text)
    jamo = sum(0x3131 <= ord(c) <= 0x318E for c in text)                 # compatibility jamo
    latin = sum(("a" <= c.lower() <= "z") for c in text)
    zw = sum(c in ZERO_WIDTH or unicodedata.category(c) == "Cf" for c in text)   # format chars
    fullwidth = sum(0xFF01 <= ord(c) <= 0xFF5E for c in text)
    nfkc_changed = int(unicodedata.normalize("NFKC", text) != text)
    letters = max(hangul + latin, 1)
    return {
        "hangul_ratio": hangul / n, "latin_ratio": latin / n, "jamo_ratio": jamo / n,
        "zero_width_count": zw, "fullwidth_ratio": fullwidth / n, "nfkc_changed": nfkc_changed,
        "ko_share_of_letters": hangul / letters, "n_chars": len(text),
    }


def dominant_script(text: str) -> str:
    s = script_stats(text)
    if s["hangul_ratio"] + s["jamo_ratio"] > s["latin_ratio"]:
        return "ko"
    return "en" if s["latin_ratio"] > 0 else "other"

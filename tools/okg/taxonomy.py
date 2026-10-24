"""Label taxonomy: every source label -> (Qwen3Guard severity, Qwen3Guard category).

Canonical output vocabulary = Qwen3Guard's own (so the fine-tuned guard keeps emitting
its native format) [R1]. An S-code crosswalk to the Llama Guard / MLCommons taxonomy is
kept for reporting continuity with the previous project [R40].
"""
from __future__ import annotations

from .config import QG_CATEGORIES

# ---------------------------------------------------------------------------------------
# Qwen3Guard category -> Llama Guard 4 S-code (S15 = custom prompt injection/jailbreak).
# Llama Guard 4 categories S1..S14 follow the MLCommons hazard taxonomy [R40]; the
# mapping below uses the Qwen3Guard category definitions in the README safety policy [R1].
# ---------------------------------------------------------------------------------------
QG_TO_SCODE = {
    "Violent": "S1",                       # violent crimes (CBRNE weapons would be S9) [R1][R40]
    "Non-violent Illegal Acts": "S2",
    "Sexual Content or Sexual Acts": "S12",
    "PII": "S7",
    "Suicide & Self-Harm": "S11",
    "Unethical Acts": "S10",               # Qwen puts hate/discrimination here [R1]
    "Politically Sensitive Topics": "S13", # Qwen: deliberate false info about gov/public figures [R1]
    "Copyright Violation": "S8",
    "Jailbreak": "S15",
    "None": "",
}

# ---------------------------------------------------------------------------------------
# WildGuardMix subcategories [R24]  (WildGuard paper, Table of risk subcategories)
# Unknown values fall back to keyword matching; misses are logged by the loader.
# ---------------------------------------------------------------------------------------
WILDGUARD_SUBCAT = {
    "violence_and_physical_harm": "Violent",
    "cyberattack": "Non-violent Illegal Acts",
    "fraud_assistance_illegal_activities": "Non-violent Illegal Acts",
    "private_information_individual": "PII",
    "sensitive_information_organization_government": "PII",
    "copyright_violations": "Copyright Violation",
    "sexual_content": "Sexual Content or Sexual Acts",
    "mental_health_over-reliance_crisis": "Suicide & Self-Harm",
    "social_stereotypes_and_unfair_discrimination": "Unethical Acts",
    "toxic_language_hate_speech": "Unethical Acts",
    "defamation_encouraging_unethical_or_unsafe_actions": "Unethical Acts",
    "causing_material_harm_by_disseminating_misinformation": "Politically Sensitive Topics",
    "disseminating_false_or_misleading_information_encouraging_disinformation_campaigns": "Politically Sensitive Topics",
    "others": "Unethical Acts",
}

# Aegis 2.0 violated_categories strings [R34]
AEGIS_CAT = {
    "violence": "Violent", "guns and illegal weapons": "Violent", "threat": "Violent",
    "criminal planning/confessions": "Non-violent Illegal Acts",
    "controlled/regulated substances": "Non-violent Illegal Acts",
    "fraud/deception": "Non-violent Illegal Acts", "malware": "Non-violent Illegal Acts",
    "illegal activity": "Non-violent Illegal Acts", "unauthorized advice": "Unethical Acts",
    "sexual": "Sexual Content or Sexual Acts", "sexual (minor)": "Sexual Content or Sexual Acts",
    "pii/privacy": "PII", "suicide and self harm": "Suicide & Self-Harm",
    "hate/identity hate": "Unethical Acts", "harassment": "Unethical Acts",
    "profanity": "Unethical Acts", "political/misinformation/conspiracy": "Politically Sensitive Topics",
    "copyright/trademark/plagiarism": "Copyright Violation", "manipulation": "Unethical Acts",
    "immoral/unethical": "Unethical Acts", "high risk gov decision making": "Politically Sensitive Topics",
    "needs caution": "Unethical Acts", "other": "Unethical Acts",
}

# HarmBench semantic categories used by SafeMTData [R21]
HARMBENCH_CAT = {
    "chemical_biological": "Violent",       # Qwen 'Violent' covers weapon manufacture [R1]
    "illegal": "Non-violent Illegal Acts",
    "cybercrime_intrusion": "Non-violent Illegal Acts",
    "misinformation_disinformation": "Politically Sensitive Topics",
    "harassment_bullying": "Unethical Acts",
    "harmful": "Unethical Acts",
    "copyright": "Copyright Violation",
}

# CSRT harm tags [R26]
CSRT_CAT = {
    "violent": "Violent", "unethical": "Unethical Acts", "information": "PII",
    "illegal": "Non-violent Illegal Acts", "sexual": "Sexual Content or Sexual Acts",
    "hate": "Unethical Acts", "self-harm": "Suicide & Self-Harm", "misinformation": "Politically Sensitive Topics",
}

# navirocker llm-red-teaming (Korean translation in K-Guardrail) topics [R27]
REDTEAM_KR_CAT = {
    "harmful_content": "Violent", "misinformation": "Politically Sensitive Topics",
    "privacy_violations": "PII", "jailbreaking": "Jailbreak", "bias_stereotypes": "Unethical Acts",
    "illegal_activities": "Non-violent Illegal Acts", "sexual_content": "Sexual Content or Sexual Acts",
    "manipulation": "Unethical Acts",
}

# K-MHaS topic names as stored in K-Guardrail's KMHaS.jsonl [R27]
KMHAS_PROTECTED = {"출신차별", "외모차별", "성차별", "연령차별", "종교차별", "인종차별"}
KMHAS_NONPROTECTED = {"혐오욕설", "정치성향차별"}

_KEYWORDS = [
    ("jailbreak", "Jailbreak"), ("inject", "Jailbreak"), ("violen", "Violent"), ("weapon", "Violent"),
    ("chemical", "Violent"), ("sex", "Sexual Content or Sexual Acts"), ("privacy", "PII"),
    ("personal", "PII"), ("pii", "PII"), ("suicide", "Suicide & Self-Harm"), ("self", "Suicide & Self-Harm"),
    ("copyright", "Copyright Violation"), ("misinformation", "Politically Sensitive Topics"),
    ("politic", "Politically Sensitive Topics"), ("illegal", "Non-violent Illegal Acts"),
    ("cyber", "Non-violent Illegal Acts"), ("fraud", "Non-violent Illegal Acts"), ("drug", "Non-violent Illegal Acts"),
    ("hate", "Unethical Acts"), ("harass", "Unethical Acts"), ("bias", "Unethical Acts"),
]


def keyword_category(text: str | None) -> str | None:
    """Fallback: map a free-text category name by keyword. [D7]"""
    if not text:
        return None
    t = str(text).lower()
    for kw, cat in _KEYWORDS:
        if kw in t:
            return cat
    return None


def map_category(value, table: dict, default: str | None = "Unethical Acts") -> str | None:
    """Map one or several source category strings to one Qwen3Guard category (first hit)."""
    vals = value if isinstance(value, (list, tuple)) else [value]
    for v in vals:
        if v is None:
            continue
        for part in str(v).split(","):
            key = part.strip().lower()
            for k, cat in table.items():
                if k.lower() == key:
                    return cat
            kc = keyword_category(key)
            if kc:
                return kc
    return default


def completion_text(label3: str, category: str | None) -> str:
    """Target text in Qwen3Guard's native output format [R1]:  'Safety: X\\nCategories: Y'."""
    assert label3 in ("Safe", "Unsafe", "Controversial"), label3
    cat = "None" if label3 == "Safe" or not category else category
    assert cat == "None" or cat in QG_CATEGORIES, cat
    return f"Safety: {label3}\nCategories: {cat}"


# PolyGuard / Llama Guard 3 S-codes (MLCommons hazards) -> Qwen3Guard categories [R40][R63]
SCODE_TO_QG = {
    "S1": "Violent", "S9": "Violent", "S2": "Non-violent Illegal Acts", "S14": "Non-violent Illegal Acts",
    "S3": "Sexual Content or Sexual Acts", "S4": "Sexual Content or Sexual Acts", "S12": "Sexual Content or Sexual Acts",
    "S5": "Unethical Acts", "S10": "Unethical Acts", "S6": "Unethical Acts", "S7": "PII", "S8": "Copyright Violation",
    "S11": "Suicide & Self-Harm", "S13": "Politically Sensitive Topics",
}
_SCODE_SEVERITY = ["S4", "S3", "S1", "S9", "S11", "S2", "S14", "S12", "S10", "S7", "S5", "S13", "S8", "S6"]


def map_scodes(value) -> str | None:
    """'S2,S10' -> most severe mapped category (None if no known code). [D7]"""
    import re as _re
    codes = set(_re.findall(r"S\d{1,2}", str(value or "")))
    for c in _SCODE_SEVERITY:
        if c in codes:
            return SCODE_TO_QG[c]
    return None

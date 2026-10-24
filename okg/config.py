"""Single source of truth for every tunable value. [D2]

`RUN_MODE="smoke"` shrinks every stage (tiny model, few rows, few steps) so the whole
pipeline can be validated in minutes before the full run on the GPU server.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path


# Qwen3Guard label and category vocabulary, spelled exactly as in the official
# extraction regex of the Qwen3Guard README. [R1]
SAFETY_LABELS = ("Safe", "Unsafe", "Controversial")
QG_CATEGORIES = (
    "Violent", "Non-violent Illegal Acts", "Sexual Content or Sexual Acts", "PII",
    "Suicide & Self-Harm", "Unethical Acts", "Politically Sensitive Topics",
    "Copyright Violation", "Jailbreak",
)


def _local_secret(name: str) -> str | None:
    """Value from local_secrets.py in the project folder (hard-coded credentials), if present."""
    try:
        import importlib
        mod = importlib.import_module("local_secrets")
        return getattr(mod, name, None) or None
    except ImportError:
        return None


@dataclass
class Config:
    # ---------------- run mode ----------------
    run_mode: str = "smoke"            # "smoke" | "full"
    project_dir: str = "."
    seed: int = 20260927
    hf_token_env: str = "HF_TOKEN"     # gated sets (XGuard-Train, MHJ, XSTest) need a token [R44]

    # ---------------- models ----------------
    guard_model: str = "Qwen/Qwen3Guard-Gen-8B"          # [R1][R2] base guard
    smoke_guard_model: str = "tiny-local"                 # built offline by okg.tiny_model
    llamaguard_model: str = "meta-llama/Llama-Guard-4-12B"  # baseline [R40]
    run_llamaguard_baseline: bool = True      # restored comparison; needs the gated model's terms accepted [D25]
    llamaguard_max_stack_benign: int = 4000   # benign stack convs scored to set its threshold (12B is slow)
    translator_model: str = "nayohan/llama3-instrucTrans-enko-8b"  # used by WildGuardMix-KO [R41][R24]
    translator_system_prompt: str = "당신은 번역기 입니다. 영어를 한국어로 번역하세요."  # verbatim from model card [R41]
    translate_multiturn: bool = True
    translate_max_conversations: int = 200    # attacks; + as many benign -> ~400 convs, ~1 GPU-hour with HF generate
    translate_max_new_tokens: int = 512       # inputs are capped at 1500 chars (translate.MAX_CHARS)
    translate_shard_size: int = 50            # conversations per saved S04 part (separate from shard_size so other stage hashes are untouched)

    # ---------------- data ----------------
    github_raw: str = "https://raw.githubusercontent.com"
    disabled_sources: tuple = ()        # names from okg.data_sources.REGISTRY to skip
    allow_noncommercial: bool = True    # False -> drop NC / research-only sources [D3]
    rows_per_source_cap: int = 20000    # cap per raw source (full mode)
    comment_corpora_ko_share: float = 0.15   # cap hate-comment corpora in KO train [D4]
    xguard_max_conversations: int = 8000
    ultrachat_max_conversations: int = 12000
    custom_sources: list = field(default_factory=list)   # previous-project CSVs etc. (see okg.data_sources.load_custom)

    # ---------------- synthesis ----------------
    synth_per_generator: int = 6000     # conversations per generator per language (full)
    synth_min_history: int = 1          # benign turns before the attack turn
    synth_max_history: int = 6
    shard_size: int = 1000              # conversations per saved part

    # ---------------- splits ----------------
    n_group_folds: int = 20             # StratifiedGroupKFold folds, assigned to roles [R11]
    role_folds: dict = field(default_factory=lambda: {"guard_train": 11, "guard_val": 1, "stack": 4, "test": 4})
    holdout_families: tuple = ()        # generator families kept out of training entirely

    # ---------------- attacker-LLM multi-turn attacks (S03b) [D50] ----------------
    attack_gen: bool = True
    attack_gen_model: str = "Qwen/Qwen2.5-7B-Instruct"     # attacker + simulated assistant (Apache-2.0)
    attack_gen_strategies: tuple = ("crescendo", "decomposition", "roleplay")
    attack_gen_langs: tuple = ("ko",)
    attack_gen_per_strategy: int = 800   # attack conversations per strategy per language (+ as many benign twins)
    attack_gen_turns: tuple = (3, 6)     # user turns per generated conversation (min, max)
    attack_gen_max_new_tokens: int = 1200
    attack_gen_batch_size: int = 8

    # ---------------- guard SFT ----------------
    max_seq_len: int = 4096             # multi-turn convs avg ~3.2k tokens [R36]
    max_prefixes_per_conv: int = 3      # cap on training prefixes from one conversation [D5]
    train_on_ambiguous_prefixes: bool = False   # pre-attack turns of native attacks [D6]
    target_unsafe_ratio: float = 0.5    # 1:1 Safe:Unsafe [R2]
    max_train_examples: int = 200000    # was 60k of ~217k available; balanced selection usually yields fewer [D53]
    replay_share: float = 0.10          # share of SFT examples labelled by the BASE model (forgetting guard) [D52]
    replay_langs: tuple = ("en",)       # languages whose zero-shot behaviour the replay preserves
    lora_r: int = 16                    # [R6] starting point, validate
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"   # PEFT shortcut for all linear layers [R6]
    learning_rate: float = 1e-4
    num_train_epochs: float = 2.0       # was 1.0 [D53]
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16     # effective batch 32
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    gradient_checkpointing: bool = True       # [R50]
    load_in_4bit: bool = False                # QLoRA option [R51] (needs bitsandbytes + CUDA)
    save_steps: int = 500
    sft_resume_from_checkpoint: bool = False  # user rule: an interrupted stage is re-run [D1]
    max_steps: int = -1

    # ---------------- scoring ----------------
    score_batch_size: int = 8
    strict_lambda: float = 1.0          # p_unsafe_strict = P(Unsafe)+λ·P(Controversial) [R2]
    parse_probe_n: int = 200            # generations used for the output-format health gate
    parse_rate_gate: float = 0.99

    # ---------------- context window (anti-padding) [D44] ----------------
    pin_first_turn: bool = True         # keep the first user/assistant pair when the window slides
    pin_first_max_tokens: int = 384     # token budget for that pinned pair (head of each message kept)
    chunk_long_messages: bool = True    # a user message too long for the window is scored in chunks, max taken
    chunk_overlap_tokens: int = 64
    max_chunks_per_message: int = 8     # beyond this the chunks are spread evenly over the message

    # ---------------- serving [D45][D49] ----------------
    use_base_features: bool = False     # base_p_* LightGBM features (SHAP 0.027) cost a second guard pass
    score_cache_size: int = 200000      # per-process LRU of per-turn scores keyed by conversation prefix
    guard_timeout_ms: int = 2000        # per-assessment budget; p95 should sit well below it
    guard_fail_mode: str = "safe_answer"   # on timeout/error: "safe_answer" | "block" | "allow" [D49]
    guard_carry_decay: float = 0.8      # guard-only serving: s_t = max(p_t, decay * s_{t-1}) [D44]
    response_moderation: bool = True    # second check on the assistant's draft reply [D51]
    response_use_adapter: bool = False  # the adapter is trained on prompt moderation only -> base model
    response_eval_max_convs: int = 6000 # multi-turn convs per role whose assistant replies S07c scores

    # ---------------- decision policy [D46][D47][D48] ----------------
    decision_system: str = "auto"       # "stack" | "guard" | "auto" (S10 picks; guard unless the stack clearly wins)
    stack_min_gain: float = 0.02        # recall the stack must add on external AND human multi-turn sets
    per_language_thresholds: bool = True
    threshold_min_benign: int = 300     # fewer benign convs for a language -> global threshold
    fpr_confidence: float = 0.95        # threshold keeps the one-sided Clopper-Pearson FPR bound <= budget
    action_safe_fpr: float = 0.05       # lower edge of the "answer safely + review" band (benign FPR)
    action_careful_contro: float = 0.5  # P(Controversial) above this (below the bands) -> careful answer

    # ---------------- Korean hate-speech classifier (S06h) [D54] ----------------
    hate_classifier: bool = True
    hate_backend: str = "transformer"   # "transformer" (full) | "ngram" (offline smoke / fallback)
    hate_model: str = "beomi/KcELECTRA-base-v2022"   # alternative: "klue/roberta-base" [R80][R81]
    hate_train_sources: tuple = ("k_haters", "kmhas", "beep")
    hate_negative_sources: tuple = ("square_q", "aihub_558", "synth_benign_native", "kosbi", "curse")
    hate_max_len: int = 128
    hate_epochs: float = 2.0
    hate_lr: float = 3e-5
    hate_batch_size: int = 32

    # ---------------- LightGBM ----------------
    lgb_params: dict = field(default_factory=lambda: {
        "objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 40,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
        "verbosity": -1,
    })                                    # [R12] starting point, validate
    lgb_num_boost_round: int = 2000
    lgb_early_stopping_rounds: int = 100  # [R13]
    lgb_cv_folds: int = 5                 # [R11]
    fpr_budget: float = 0.01              # threshold = max recall s.t. benign FPR <= budget [R15]

    # ---------------- data expansion / quality ----------------
    kovast_max_conversations: int = 12000       # KO benign multi-turn history material
    aihub_max_talksets: int = 120000            # AIHub 558 talksets used (of ~132k); was 30k [D50]
    aihub_datasetkey: int = 558
    aihub_filekeys: list = field(default_factory=list)   # empty -> auto-pick label files from the listing
    screen_benign: bool = True                  # zero-shot guard screen of benign pools (S02b) [D40]
    screen_threshold: float = 0.5               # drop a benign conversation if any user turn P(Unsafe)+P(Contro) > this
    min_free_ram_gb: float = 4.0                # warn before a stage if less RAM is available [D39]

    # ---------------- attack / benign memory ----------------
    use_attack_memory: bool = True
    memory_encoder: str = "e5"                                   # "e5" (full) | "hashing" (offline)
    memory_encoder_model: str = "intfloat/multilingual-e5-small" # MIT, 384-d [R56]
    memory_seed_cap: int = 50000                                 # max seed entries per kind [D29]

    # ---------------- feedback loop ----------------
    review_daily_budget: int = 200            # items per review batch [D31]
    review_random_share: float = 0.25         # share reserved for a random slice of ALLOWED traffic [D31]
    review_uncertainty_band: float = 0.15     # |score - threshold| counted as uncertain [R57]
    review_max_per_user: int = 5              # per-user cap in one queue [D30]
    review_show_model_decision: bool = False  # hide model verdict from reviewers (anchoring) [D33]
    feedback_test_days: int = 14              # newest N days of reviewed traffic = time-based test [D32]
    feedback_max_per_user: int = 20           # per-user cap on reviewed items used for training [D30]
    feedback_to_guard_train: bool = False     # monthly: also fine-tune the adapter on older reviews [D34]

    # ---------------- promotion gate ----------------
    gate_min_n: int = 30                      # only compare groups with >= n positives / negatives
    gate_recall_tol: float = 0.02             # allowed recall drop per dataset [D35]
    gate_fpr_tol: float = 0.01                # allowed FPR increase per dataset [D35]
    # named checks that must exist and must not get worse; "feedback" = reviewed shadow traffic [D55]
    gate_required_groups: tuple = ("source=mhj", "source=dktc", "source=aihub_558", "source=feedback")
    redteam_on_release: bool = True           # run the red-team suite against every candidate [D55]
    redteam_asr_tol: float = 0.02             # allowed rise in attack success rate vs CURRENT
    redteam_n_per_strategy: int = 40          # fresh generated attacks per strategy per run (+ fixed suite)

    # ---------------- monitoring [D56] ----------------
    monitor_psi_alert: float = 0.2            # population-stability index of the score distribution
    monitor_block_rate_alert: float = 2.0     # alert when an hourly block rate is > x times its 7-day median
    monitor_backlog_alert_days: float = 2.0   # oldest unreviewed queue item older than this -> alert
    monitor_kappa_alert: float = 0.6          # reviewer agreement below this -> alert

    # ---------------- evaluation ----------------
    bootstrap_n: int = 1000               # conversation-level (cluster) bootstrap [R18]
    recall_target: float = 0.90

    # ======================================================================
    def __post_init__(self):
        if self.run_mode not in ("smoke", "full"):
            raise ValueError("run_mode must be 'smoke' or 'full'")
        if self.run_mode == "smoke":
            self.rows_per_source_cap = 120
            self.xguard_max_conversations = 60
            self.ultrachat_max_conversations = 120
            self.synth_per_generator = 40
            self.shard_size = 25
            self.translate_max_conversations = 6
            self.translate_max_new_tokens = 16
            self.max_seq_len = 512
            self.max_train_examples = 300
            self.per_device_train_batch_size = 4
            self.gradient_accumulation_steps = 1
            self.max_steps = 20
            self.save_steps = 10
            self.bf16 = False
            self.gradient_checkpointing = False
            self.score_batch_size = 8
            self.parse_probe_n = 8
            self.parse_rate_gate = 0.0          # a random tiny model cannot follow the format
            self.lgb_num_boost_round = 200
            self.lgb_early_stopping_rounds = 20
            self.lgb_cv_folds = 3
            self.bootstrap_n = 50
            self.n_group_folds = 10
            self.role_folds = {"guard_train": 5, "guard_val": 1, "stack": 2, "test": 2}
            self.fpr_budget = 0.05
            self.review_daily_budget = 40
            self.feedback_test_days = 7
            self.memory_encoder = "hashing"
            self.kovast_max_conversations = 60
            self.aihub_max_talksets = 60
            self.screen_threshold = 0.99            # tiny random model: exercise the code, drop nothing
            self.gate_min_n = 5
            self.num_train_epochs = 1.0
            self.attack_gen_per_strategy = 6
            self.attack_gen_max_new_tokens = 32
            self.hate_backend = "ngram"
            self.response_eval_max_convs = 40
            self.threshold_min_benign = 20
            self.redteam_n_per_strategy = 3
            self.run_llamaguard_baseline = False    # the 12B model cannot run offline

    # ---- derived paths ----
    @property
    def root(self) -> Path:
        return Path(self.project_dir).resolve()

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts" / self.run_mode

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"        # downloads shared by smoke/full

    @property
    def aihub_dir(self) -> Path:
        return self.cache_dir / "aihub"

    def apply_hardware_profile(self, verbose: bool = True) -> dict:
        """Fit the run to the detected GPU (full mode only). Measured budgets: 8B bf16 weights ~16 GB,
        8B NF4 ~5.5 GB, 4B bf16 ~8 GB; activations grow with sequence length. Below 30 GB the 8B model
        is trained as 4-bit QLoRA instead of falling back to the 4B model [R51][D39][D53]."""
        from .memutil import gpu_memory_gb, _meminfo_gb
        vram, ram = gpu_memory_gb(), _meminfo_gb("MemTotal") or 0.0
        prof = {"vram_gb": round(vram, 1), "ram_gb": round(ram, 1)}
        if self.run_mode != "full":
            prof["profile"] = "smoke (no change)"
        elif vram == 0:
            prof["profile"] = "NO GPU detected — full mode will be extremely slow"
        elif vram < 30:
            self.max_seq_len, self.per_device_train_batch_size, self.gradient_accumulation_steps = 2048, 1, 32
            self.score_batch_size = 4
            try:
                import bitsandbytes  # noqa: F401
                self.load_in_4bit = True
                prof["profile"] = "<30 GB GPU: Qwen3Guard-Gen-8B as 4-bit QLoRA, 2k context, batch 1x32"
            except ImportError:
                self.guard_model = "Qwen/Qwen3Guard-Gen-4B"
                prof["profile"] = ("<30 GB GPU without bitsandbytes: Qwen3Guard-Gen-4B, 2k context. "
                                   "pip install bitsandbytes to train the 8B model in 4-bit instead.")
        elif vram < 60:
            self.max_seq_len, self.per_device_train_batch_size, self.gradient_accumulation_steps = 2048, 1, 32
            self.score_batch_size = 4
            prof["profile"] = "30-60 GB GPU: Qwen3Guard-Gen-8B bf16, 2k context, batch 1x32"
        else:
            prof["profile"] = ">=60 GB GPU: defaults (8B, 4k context)"
        if ram and ram < 32 and self.run_mode == "full":
            self.rows_per_source_cap = min(self.rows_per_source_cap, 20000)
            self.shard_size = min(self.shard_size, 500)
            self.aihub_max_talksets = min(self.aihub_max_talksets, 60000)
            prof["ram_note"] = "<32 GB RAM: per-source cap 20k, shard 500, AIHub 60k talksets"
        if verbose:
            print(prof)
        return prof

    @property
    def feedback_dir(self) -> Path:
        return self.root / "feedback" / self.run_mode

    @property
    def releases_dir(self) -> Path:
        return self.root / "releases" / self.run_mode

    @property
    def review_files(self) -> list:
        d = self.feedback_dir / "reviewed"
        return sorted(d.glob("*.csv")) if d.exists() else []

    @property
    def guard_model_id(self) -> str:
        return self.smoke_guard_model if self.run_mode == "smoke" else self.guard_model

    @property
    def hf_token(self) -> str | None:
        return os.environ.get(self.hf_token_env) or _local_secret("HF_TOKEN")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def subset(self, *keys: str) -> dict:
        d = self.to_dict()
        return {k: d[k] for k in keys}

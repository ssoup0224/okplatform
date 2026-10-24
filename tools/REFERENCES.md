# okchat_guardrail — references and design decisions

Every non-trivial line in `okg/` and in the notebook carries a tag:
- **[R#]**: an external source (listed below) that the code follows.
- **[D#]**: a design decision of this project, with its rationale. These are *not* claims backed by a paper, and they are labelled that way on purpose.

Sources marked † came from the design-research report and were not re-opened line-by-line during the build. Check them before citing them in a paper.

## External sources

| Tag | Source | Used for |
|---|---|---|
| R1 | Qwen3Guard README (usage code, output format, extraction regex, safety policy). https://github.com/QwenLM/Qwen3Guard | prompt rendering call, `Safety:`/`Categories:` target format, label and category vocabulary |
| R2 | Zhao et al., *Qwen3Guard Technical Report*, arXiv:2510.14276. https://arxiv.org/abs/2510.14276 | base-model choice (Korean share, benchmark F1), 1:1 Safe:Unsafe balance (§3.3), strict/loose (Controversial) modes, model-assisted labelling, paired surface-similar negatives |
| R3 | Qwen3Guard-Gen model card and `tokenizer_config.json` chat template. https://huggingface.co/Qwen/Qwen3Guard-Gen-8B | the template appends `<|im_start|>assistant\n<think>\n\n</think>\n\n` and branches on `messages[-1].role`; the whole dialogue is included |
| R4 | TRL `SFTTrainer` documentation (prompt-completion datasets, `completion_only_loss`, `peft_config`). https://huggingface.co/docs/trl/sft_trainer | loss on the verdict only, LoRA integration |
| R5 | TRL issue #3781 (assistant-only loss mask dropped with the Liger kernel). https://github.com/huggingface/trl/issues/3781 | reason Liger is not enabled |
| R6 | PEFT docs: `LoraConfig` (`target_modules="all-linear"`), `PeftModel.from_pretrained`, `disable_adapter`. https://huggingface.co/docs/peft/en/package_reference/peft_model, https://huggingface.co/docs/peft/en/package_reference/lora. LoRA: Hu et al., arXiv:2106.09685 | adapter training and loading; baseline without adapter |
| R7 | Transformers `Trainer` / `TrainingArguments` docs (checkpointing, `resume_from_checkpoint`, `warmup_steps` ratio in v5). https://huggingface.co/docs/transformers/main_classes/trainer | SFT checkpoints and resume |
| R8 | Transformers `set_seed`. https://huggingface.co/docs/transformers/internal/trainer_utils | seeding |
| R9 | Python `os.replace` (atomic on POSIX, same filesystem) and `os.fsync`. https://docs.python.org/3/library/os.html#os.replace | atomic saves: nothing partial is ever committed |
| R10 | † CPython issue #143909 (atomicity covers the destination name). https://github.com/python/cpython/issues/143909 | temp file placed in the destination directory |
| R11 | scikit-learn `StratifiedGroupKFold`. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html | group-wise (seed-level) splits, LightGBM CV |
| R12 | LightGBM parameters (`scale_pos_weight`, `is_unbalance`, and the warning that these distort probabilities). https://lightgbm.readthedocs.io/en/latest/Parameters.html | imbalance handling, then calibration |
| R13 | LightGBM `early_stopping` callback. https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html | early stopping on an inner group fold |
| R14 | scikit-learn probability calibration user guide and `CalibratedClassifierCV` (isotonic overfits below ~1000 samples; the ensemble of per-fold calibrated models). https://scikit-learn.org/stable/modules/calibration.html | per-fold Platt/isotonic, `FoldEnsemble` |
| R15 | scikit-learn `roc_auc_score` (incl. `max_fpr` McClish partial AUC), `average_precision_score`, `roc_curve`. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html | metrics, FPR-budget threshold |
| R16 | SHAP `TreeExplainer`. https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html | feature importance |
| R17 | statsmodels `mcnemar` (exact binomial). https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html | paired system comparison |
| R18 | Efron & Tibshirani, *An Introduction to the Bootstrap* (1993). Field & Welsh, "Bootstrapping clustered data", JRSS-B 69(3), 2007 | percentile CIs that resample conversations (clusters), not turns |
| R19 | † SafeDream, arXiv:2604.16824 | per-turn formulation, conversation score = max over turns, detection latency = first flagged turn, topic-matched benign negatives |
| R20 | Russinovich et al., *Crescendo*, arXiv:2404.01833 | gradual-escalation attack family (covered by native dialogues) |
| R21 | Ren et al., *ActorAttack / ActorBreaker*, arXiv:2410.10700, and the SafeMTData card (MIT). https://huggingface.co/datasets/SafeMTData/SafeMTData | English multi-turn attacks, HarmBench categories |
| R22 | XGuard-Train card (X-Teaming). https://huggingface.co/datasets/marslabucla/XGuard-Train | English multi-turn attacks (ShareGPT format) |
| R23 | Li et al., *LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks* (MHJ), arXiv:2408.15221. https://huggingface.co/datasets/ScaleAI/mhj | held-out human attack family (CC-BY-NC-4.0, evaluation only) |
| R24 | Han et al., *WildGuard*, arXiv:2406.18495. WildGuardMix-KO train/test cards (iknow-lab, ODC-BY) | Korean+English paired single-turn data; translator provenance |
| R25 | KOTOX (repo, paper and HF `ssgyejin/KOTOX-classification`). https://github.com/leeyejin1231/KOTOX | Korean obfuscation rules (2/3/4 rules per difficulty), obfuscated train/test |
| R26 | CSRT (code-switching red-teaming). https://huggingface.co/datasets/walledai/CSRT | code-switched evaluation set |
| R27 | awesome-korean-safety list https://github.com/Marker-Inc-Korea/awesome-korean-safety; K-Guardrail dataset repo https://github.com/skan0779/korean-guardrail-dataset (processed K-MHaS, KOLD, APEACH, translated prompt-injection / red-teaming / RaccoonBench); K-MHaS card https://huggingface.co/datasets/jeanlee/kmhas_korean_hate_speech | Korean dataset selection and loaders |
| R28 | KOLD. https://github.com/boychaboy/KOLD | Korean offensive comments |
| R29 | K-HATERS. https://github.com/ssu-humane/K-HATERS, https://huggingface.co/datasets/humane-lab/K-HATERS | Korean hate levels (L2 / L1 / offensive / normal) |
| R30 | SQuARe (NAVER). https://github.com/naver-ai/korean-safety-benchmarks | sensitive-but-safe Korean questions and acceptable responses |
| R31 | UltraChat 200k (MIT). https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k | benign English multi-turn dialogues |
| R32 | OR-Bench. https://huggingface.co/datasets/bench-llm/or-bench | over-refusal hard negatives and evaluation |
| R33 | Röttger et al., *XSTest*, arXiv:2308.01263 | exaggerated-safety evaluation; string-match refusal detection |
| R34 | Aegis 2.0 (CC-BY-4.0). https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0 | English single-turn data, category names |
| R35 | Unicode Standard §3.12 (Hangul syllable decomposition), UAX #15 (NFKC), UTS #39 (security / confusables), code charts U+3130 and U+FF00. https://www.unicode.org/versions/latest/ , https://unicode.org/reports/tr15/ , https://unicode.org/reports/tr39/ | obfuscation rules and script features |
| R36 | † *Defensive M2S*, arXiv:2601.00454 | multi-turn prompts average ~3.2k tokens (hence `max_seq_len=4096`); sliding-window context |
| R37 | † *Peak + Accumulation* multi-turn scoring, arXiv:2602.11247 | `peak_accum` feature |
| R38 | † Hierarchical turn/conversation model, arXiv:2606.21082 | ordered trajectory features beat mean pooling |
| R40 | Llama Guard 4 model card (https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard4/12B/MODEL_CARD.md). MLCommons AILuminate taxonomy, arXiv:2503.05731 | S-code crosswalk, optional baseline |
| R41 | `nayohan/llama3-instrucTrans-enko-8b` model card. https://huggingface.co/nayohan/llama3-instrucTrans-enko-8b | EN→KO translation (system prompt quoted verbatim) |
| R42 | KoAlpaca v1.1a. https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a | benign Korean history pairs (license unspecified, so flagged non-commercial) |
| R43 | pandas `DataFrame.to_parquet`. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html | artifact format |
| R44 | Hugging Face `datasets.load_dataset` and gated-dataset access. https://huggingface.co/docs/datasets/loading , https://huggingface.co/docs/hub/datasets-gated | loaders; `HF_TOKEN` |
| R45 | Kanana Safeguard model card (evaluation by first output token). https://huggingface.co/kakaocorp/kanana-safeguard-8b | logit scoring instead of parsing |
| R46 | Zeng et al., *ShieldGemma*, arXiv:2407.21772 | probability from verdict-token logits |
| R47 | Inan et al., *Llama Guard*, arXiv:2312.06674 | P(unsafe) from the first generated token; Llama Guard baseline scoring |
| R48 | scikit-learn `HashingVectorizer`. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html | stateless character n-gram similarity |
| R50 | Chen et al., *Training Deep Nets with Sublinear Memory Cost*, arXiv:1604.06174 | gradient checkpointing |
| R51 | Dettmers et al., *QLoRA*, arXiv:2305.14314. Transformers bitsandbytes docs | optional 4-bit NF4 training |
| R52 | PyTorch `torch.cuda.OutOfMemoryError`, `torch.cuda.empty_cache`. https://docs.pytorch.org/docs/stable/generated/torch.cuda.OutOfMemoryError.html | OOM back-off in scoring |
| R53 | Transformers LLM tutorial, "wrong padding side". https://huggingface.co/docs/transformers/llm_tutorial | left padding for decoder-only batching |
| R54 | Transformers Qwen3 docs. https://huggingface.co/docs/transformers/model_doc/qwen3 | tiny offline stand-in model |
| R55 | Zou et al., *Universal and Transferable Adversarial Attacks* (GCG refusal-prefix list), arXiv:2307.15043 | refusal string patterns |
| R56 | `intfloat/multilingual-e5-small` model card (average pooling, `query:` prefix for similarity tasks, cosine scores clustered around 0.7–1.0, MIT). https://huggingface.co/intfloat/multilingual-e5-small. Wang et al., *Multilingual E5 Text Embeddings*, arXiv:2402.05672 | memory encoder |
| R57 | Settles, *Active Learning Literature Survey*, Univ. of Wisconsin–Madison CS Tech. Report 1648 (2009) | uncertainty sampling and query-by-committee in the review queue |
| R58 | Cohen (1960), coefficient of agreement; scikit-learn `cohen_kappa_score`. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html | reviewer agreement |
| R59 | Structure of the Korean resident registration number (6-digit birth date + 7 digits, first digit 1–8). https://en.wikipedia.org/wiki/Resident_registration_number | PII scrub patterns (best effort, not a legal compliance guarantee) |
| R60 | Python `hashlib` (SHA-256). https://docs.python.org/3/library/hashlib.html | salted user-id hashing |
| R61 | Rae et al., *Scaling Language Models: Methods, Analysis & Insights from Training Gopher*, arXiv:2112.11446, Appendix A (repetition / quality filters) | repetition-loop filter |
| R62 | `iknow-lab/wildjailbreak-train-ko` / `-eval-ko` (column stats via the HF dataset viewer); Jiang et al., *WildTeaming / WildJailbreak*, arXiv:2406.18510 | KO+EN WildJailbreak |
| R63 | Kumar et al., *PolyGuard*, arXiv:2504.04377; `ToxicityPrompts/PolyGuardMix`, `ToxicityPrompts/PolyGuardPrompts` | multilingual KO/EN train and test |
| R64 | Lin et al., *ToxicChat*, arXiv:2310.17389; `lmsys/toxic-chat` (CC-BY-NC-4.0) | real-user prompts |
| R65 | `jackhhao/jailbreak-classification`, `xTRam1/safe-guard-prompt-injection` dataset cards | binary jailbreak / injection |
| R66 | Shen et al., *"Do Anything Now"*, arXiv:2308.03825; `TrustAIRLab/in-the-wild-jailbreak-prompts` | in-the-wild jailbreak templates |
| R67 | Wang et al., *Do-Not-Answer*, arXiv:2308.13387; `LibrAI/do-not-answer` | should-refuse questions |
| R68 | Smilegate AI, *Korean UnSmile dataset* (https://github.com/smilegate-ai/korean_unsmile_dataset); `smilegate-ai/kor_unsmile` | KO multi-label hate |
| R69 | Moon et al., *BEEP! Korean corpus of online news comments*, arXiv:2005.12503; https://github.com/kocohub/korean-hate-speech | KO hate / offensive / bias |
| R70 | https://github.com/2runo/Curse-detection-data (MIT) | KO profanity |
| R71 | `maywell/koVast` card (MIT, ShareGPT-style `conversations`) | KO benign multi-turn |
| R72 | `heegyu/XSTest_ko` (XSTest with `prompt_ko`) | KO/EN over-refusal eval |
| R73 | Chao et al., *JailbreakBench*, arXiv:2404.01318; `JailbreakBench/JBB-Behaviors` | EN eval |
| R74 | Markov et al., *A Holistic Approach to Undesired Content Detection*, arXiv:2208.03274; `mmathys/openai-moderation-api-evaluation` | EN eval (null = not annotated) |
| R75 | `datumo/KorSET` card (gated, CC-BY-NC-4.0, one split per category); CAGE, arXiv:2602.20170 | KO native red-team eval |
| R76 | AIHub 텍스트 윤리검증 데이터 (dataSetSn=558). https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=558 | KO native multi-turn, per-sentence immorality labels |
| R77 | AIHub API shell guide. https://aihub.or.kr/devsport/apishell/list.do?currMenu=403&topMenu=100 | selective download with `-filekey` |
| R78 | glibc `malloc_trim(3)`. https://man7.org/linux/man-pages/man3/malloc_trim.3.html | returning freed heap to the OS |
| R79 | PyTorch CUDA semantics: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. https://docs.pytorch.org/docs/stable/notes/cuda.html | fewer fragmentation OOMs |


## Design decisions (project-specific, not external claims)

| Tag | Decision | Rationale |
|---|---|---|
| D1 | Stage-level commit: write to `<stage>.__tmp__/`, rename into place, and write the `_DONE.json` marker **last**. Long loops save "parts" with their own markers. Stage hash = config + upstream hashes + code hash. SFT checkpoint resume is off by default. | This is your requirement: only finished work is saved; an interrupted part is re-run; a changed config or code never reuses stale results. |
| D2 | One `Config` dataclass with a `smoke` mode that shrinks everything. | The pipeline can be validated in minutes before a multi-hour run. |
| D3 | `allow_noncommercial` switch. | MHJ, DKTC, KoAlpaca, the prompt-injection-KR data and others are NC, unknown or GPL. A commercial model must be trained without them. |
| D4 | K-MHaS: protected-attribute discrimination → Unsafe; profanity or political leaning only → Controversial. K-HATERS L1 / offensive → Controversial. KOLD offensive → Controversial. Comment corpora capped at 15% of Korean SFT data. | Comment-toxicity corpora are not LLM-request data. Treating all rudeness as Unsafe teaches over-refusal. |
| D5 | At most 3 training prefixes per conversation, always including the attack turn. | Stops long dialogues from dominating (approximates 1/T weighting, which TRL lacks). |
| D6 | Native multi-turn attacks (SafeMT, XGuard, MHJ): the last user turn is Unsafe, and earlier turns are unknown (excluded from SFT and from FPR). | The datasets do not mark where intent becomes clear. Labelling those turns Safe or Unsafe would be a guess. |
| D7 | Keyword fallback for unmapped source categories. | Keeps every Unsafe example inside Qwen3Guard's category vocabulary. Misses are printed. |
| D8 | Lenient JSONL reader. | K-Guardrail's `prompt-injections-benchmark-kr.jsonl` contains one malformed line (verified). |
| D9 | XGuard forced into `guard_train`. | No behaviour id is published, so plans of one behaviour cannot be grouped. Keeping them out of stack and test rules out leakage. |
| D10 | DKTC: speaker A → user, every user turn Unsafe, latency not reported. | These are human-human threat dialogues, so conversation-level recall is meaningful and per-turn latency is not. |
| D11 | KoTox disabled by default. | Many rows ask the model's political opinion, which is not unsafe under the policy. |
| D12 | `custom_sources` hook. | Brings in the previous project's KoreanGuardrail / wildjailbreak-train-ko data, or AIHub exports, without code changes. |
| D13 | Script features from Unicode classes. | Cheap signals for code-switching and obfuscation. |
| D14 | Split before synthesis; synthesise within a role. | Benign histories and seeds can never cross roles. |
| D15 | Drop training rows whose user text exactly matches any evaluation text. | Exact-duplicate leakage guard. |
| D16 | History messages clipped to 1,500 characters. | Keeps prompts within `max_seq_len`. |
| D17 | Sliding window that drops the oldest pair first, with a 48-token completion reserve. | The latest user turn is never lost; SFT examples are never truncated (asserted). |
| D18 | Verdict-token ids derived by tokenising `prompt + "Safety:" + " <label>"`, with prefix stability and distinctness asserted. | No hard-coded token ids; the check fails loudly on a different tokenizer. |
| D19 | Teacher categories for Unsafe rows without a category (XGuard, MHJ) come from the base model's first category token. | Keeps the native `Categories:` line meaningful. |
| D20 | Zero-shot base-guard scores are used as a LightGBM feature and as a baseline. | An independent second opinion at no extra model cost. |
| D21 | LightGBM trained only on the `stack` role, with calibrated per-fold ensemble inference. | Honest out-of-sample guard scores; test scores come from the same models that picked the threshold. |
| D22 | Threshold = smallest score whose conversation-level benign FPR is within the budget. | Deployment decision: at most X% of benign conversations blocked. |
| D23 | Translate benign dialogues too. | Machine-translation style must not become an "attack" cue (the lesson of the previous project's length shortcut). |
| D24 | Score the base model **before** attaching the adapter; `inference.py` reuses the same functions. | Rules out the previous project's baseline contamination. |
| D25 | Llama Guard 4 baseline optional and untested offline. | Needs the gated 12B model and GPU. |
| D26 | Shortcut guard: no raw language-identity, length or plain-similarity features. Benign mirror turns receive the same obfuscation and language switches as attacks. | In synthetic data these features identify the generator, not the label. The smoke run showed them ranking top before the fix. |
| D27 | Attack/benign memory used as LightGBM features (max, top-3, count above cutoff, benign max, margin, running max), not as a blocklist. Confirmed false positives go to the benign memory. | A reviewer confirmation takes effect in minutes and can be undone by deactivating the entry. LightGBM learns how much to trust similarity, which avoids surface-match false positives. |
| D28 | Group-excluded similarity; test-window confirmations are withheld until release. | Without this, every stored item matches itself at similarity 1.0 and the evaluation becomes non-causal. |
| D29 | `close_sim` 0.90 (e5) or 0.5 (hashing); `memory_seed_cap` of 50k per kind. | Starting points: e5 scores are compressed into 0.7–1.0 [R56]. Validate on your own traffic. |
| D30 | Nothing trains or enters memory without a human verdict; per-user caps in the queue and in training; duplicates removed. | Defence against poisoning by attackers who are also users. |
| D31 | The queue draws a random slice of allowed traffic first, then signals, disagreements and uncertain items. Every reason that applies is listed in `flags`. | The random slice is the only unbiased estimate of missed attacks; the rest spends the review budget where it is most informative [R57]. |
| D32 | The newest `feedback_test_days` of reviewed traffic form the test set; older reviews go to stack (or to training in the monthly path). | Evaluate on data newer than the training data. |
| D33 | The model's decision is hidden from reviewers by default. | Avoids anchoring reviewers to the model's answer. |
| D34 | Weekly path: SF1 → S10 (LightGBM + memory only). Monthly path: `feedback_to_guard_train=True` feeds older reviews into S01 so the adapter is retrained. | The cheap path runs often; the expensive path runs rarely and is gated. |
| D35 | Versioned releases with an atomic `CURRENT` pointer. The promotion gate fails a dataset only when the candidate's 95% bootstrap CI lies entirely beyond the tolerance (recall 0.02, FPR 0.01; minimum n = 30). | Point estimates on small sets would reject good releases because of noise; the tolerances are project choices. |
| D36 | Memory ablation (`okchat_lgb_nomem`) trained with the same protocol. | Shows whether the memory generalizes or only matches near-duplicate templates. In smoke fixtures it dominated SHAP because the fixture texts are templated. |
| D37 | Unified schema with no nulls. Unknown labels use the sentinel `"Unknown"`, Safe turns use the category `"None"`, and every row carries provenance (`label_origin`, `translated`, `mt_model`). `validate_frame` hard-fails on nulls, count mismatches or out-of-enum values in every source and in the combined table. | This is your requirement that all datasets share columns with no null or mismatch. |
| D38 | Quality filters per source, with every drop counted in `DATASET_AUDIT.md`: empty or short, > 8k characters, MT loops, script mismatch, KO/EN length ratio outside 0.25–1.5, duplicates, and label conflicts within a source. | Keeps only usable, consistently labelled samples. The thresholds are heuristics; inspect the drop counts. |
| D39 | RAM hygiene for about 23 GB:<br>• datasets stay memory-mapped, and columns are selected, rows filtered and rows sampled before conversion to Python;<br>• one source at a time, released after each;<br>• `after_stage()` (gc, CUDA cache, `malloc_trim`) after every stage;<br>• `expandable_segments`;<br>• the hardware profile switches to the 4B guard below 30 GB of VRAM. | Avoids out-of-memory errors without losing progress: stages resume. |
| D40 | Benign chat pools (koVast, UltraChat, KoAlpaca, SQuARe responses) are screened by the zero-shot guard before use (S02b). | Unlabelled chat data can hide jailbreaks or NSFW role-play that would be mislabelled Safe. |
| D41 | AIHub 558 mapping:<br>• speaker 1 → user;<br>• `is_immoral=false` → Safe;<br>• only CENSURE/ABUSE with intensity < 2 → Controversial;<br>• otherwise Unsafe, using the most severe type (VIOLENCE > SEXUAL > CRIME > HATE/DISCRIMINATION > CENSURE/ABUSE);<br>• cut at the first Unsafe turn;<br>• seed = talkset id. | Uses the dataset's own human intensity votes to separate rude from harmful. |
| D42 | AIHub is downloaded with `aihubshell` on your server, label files only; the key comes from `AIHUB_API_KEY` and is never printed. | Access is restricted to approved users; no need to download the whole dataset. |
| D43 | Cross-source harmonization:<br>• identical texts with conflicting labels are resolved by origin (human > model > template), or dropped on a tie;<br>• identical texts share one seed group (union-find);<br>• a group containing a fixed `guard_train` row stays in guard_train;<br>• evaluation rows are never altered. | Stops the same prompt from appearing in train and test, and stops contradictory labels. |


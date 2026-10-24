# Datasets used by okchat_guardrail

Every source is converted to the **same columns, with no nulls** (see `okg/schema.py`, `validate_frame`):

| column | meaning |
|---|---|
| `conv_id` | unique id |
| `source`, `family` | dataset name, and the kind of sample (single, comment, native_attack, synth…) |
| `lang` | `ko`, `en` or `mixed` |
| `seed_id` | grouping key: translations, variants and duplicates of one item share it, so they never cross train/test |
| `messages` | JSON list of `{role: user\|assistant, content}`, ending on the user turn being moderated |
| `label3` | one label per user turn: `Safe` / `Unsafe` / `Controversial` / `Unknown` (Unknown = not labelled, never trained on) |
| `category` | one per user turn: a Qwen3Guard category, `None` (for Safe), or `Unknown` (filled by the teacher before training) |
| `t_attack` | first Unsafe user turn, or -1 |
| `n_user_turns`, `latency_valid` | turn count; whether detection latency is meaningful |
| `pool`, `fixed_role` | how the row is used (train_single, comment, history, native_multi, eval…); forced role if any |
| `license`, `commercial_ok` | licence string; False for NC, ND, unknown, GPL or AIHub |
| `label_origin` | `human` / `model` / `template` / `derived` |
| `translated`, `mt_model` | machine-translated row and the translation model |

Rows are filtered per source: empty or too short; longer than 8k characters; MT loops; script mismatch; bad KO/EN length ratio; duplicates; label conflicts. Every drop is counted in `artifacts/<mode>/S01_sources/DATASET_AUDIT.md`, which is regenerated from the real data on every run and includes example rows per label.

Labels map to Qwen3Guard's `Safe / Unsafe / Controversial` plus one of its 9 categories. **Rude but not hateful → Controversial** [D4].

## Training sources
| source | where / how loaded | columns used | label mapping | licence (commercial?) |
|---|---|---|---|---|
| wgm_ko_train | HF `iknow-lab/wildguardmix-train-ko` (86k) | prompt, prompt_ko, prompt_harm_label, subcategory, adversarial | harmful → Unsafe (WildGuard subcategory → category); unharmful → Safe; null labels dropped; KO kept only if the MT check passes | ODC-BY (yes) |
| wjb_ko_train | HF `iknow-lab/wildjailbreak-train-ko` (261k, capped) | prompt_ko, vanilla, adversarial, data_type | adversarial_harmful → Unsafe/Jailbreak; vanilla_harmful → Unsafe (teacher category); *_benign → Safe; seed = vanilla text | ODC-BY (yes) |
| polyguardmix | HF `ToxicityPrompts/PolyGuardMix`, Korean + English rows only (filtered on disk) | prompt, prompt_harm_label, prompt_safety_categories, metadata.language | yes → Unsafe (S-codes → category); no → Safe; fixed to guard_train | unverified (treated NC) |
| aihub_558 | AIHub 558 via `aihubshell`, label JSON only (needs `AIHUB_API_KEY`) | talkset id, sentences[].speaker/text/is_immoral/types/intensity | see D41 (native KO multi-turn, human labels) | AIHub policy (no) |
| kmhas | GitHub K-Guardrail `KMHaS.jsonl` | query, topic | protected-group discrimination → Unsafe; profanity or politics only → Controversial; none → Safe | CC-BY-SA-4.0 (yes) |
| k_haters | HF `humane-lab/K-HATERS` | text, label | L2_hate → Unsafe; L1_hate and offensive → Controversial; normal → Safe | CC-BY-4.0 (yes) |
| kold | GitHub K-Guardrail `KOLD.jsonl` | query, blocked | offensive → Controversial; else Safe | unknown (no) |
| beep | GitHub `kocohub/korean-hate-speech` labeled/train.tsv | comments, hate, bias | hate → Unsafe; offensive, or biased-but-not-hateful → Controversial; none → Safe | CC-BY-SA-4.0 (yes) |
| unsmile | HF `smilegate-ai/kor_unsmile` | 문장 + label columns | group hate → Unsafe; 악플/욕설 or 개인지칭 only → Controversial; clean → Safe | CC-BY-NC-ND (no) |
| curse | GitHub `2runo/Curse-detection-data` dataset.txt | text\|label | 1 → Controversial; 0 → Safe | MIT (yes) |
| kosbi | GitHub naver-ai KoSBi v2 | context, sentence, sentence_label | unsafe → Controversial; safe → Safe (statements, not requests; capped at 10k) | MIT (yes) |
| kotox_train | HF `ssgyejin/KOTOX-classification` total/train | text, label | 1 → Unsafe; 0 → Safe (obfuscated KO) | MIT (yes) |
| pi_bench_kr | GitHub K-Guardrail translated injection benchmark | query, blocked | blocked → Unsafe/Jailbreak | CC-BY-NC (no) |
| square_q | GitHub naver-ai SQuARe question_train.json | question | Safe (sensitive but acceptable questions) | MIT (yes) |
| aegis2 | HF `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` | prompt, prompt_label, violated_categories | human labels; REDACTED and empty rows dropped | CC-BY-4.0 (yes) |
| toxic_chat_train | HF `lmsys/toxic-chat` toxicchat0124/train | user_input, toxicity, jailbreaking, human_annotation | jailbreaking → Unsafe/Jailbreak; toxicity → Unsafe; else Safe | CC-BY-NC-4.0 (no) |
| jailbreak_cls | HF `jackhhao/jailbreak-classification` | prompt, type | jailbreak → Unsafe/Jailbreak; benign → Safe | unverified (no) |
| safeguard_pi | HF `xTRam1/safe-guard-prompt-injection` (capped at 3k) | text, label | 1 → Unsafe/Jailbreak | unverified (no) |
| itw_jailbreak | HF `TrustAIRLab/in-the-wild-jailbreak-prompts` jailbreak_* configs | prompt | Unsafe/Jailbreak; near-duplicate templates collapsed | unverified (no) |
| do_not_answer | HF `LibrAI/do-not-answer` | question, risk_area | Unsafe by risk area; chatbot-interaction harms → Controversial | Apache-2.0 (yes) |
| orbench_80k | HF `bench-llm/or-bench` or-bench-80k | prompt | Safe (over-refusal hard negatives) | CC-BY-4.0 (yes) |
| safemt_1k | HF `SafeMTData/SafeMTData` SafeMTData_1K | conversations, query_id | last user turn Unsafe; earlier turns Unknown | MIT (yes) |
| xguard | HF `marslabucla/XGuard-Train` (gated) | conversations | last turn Unsafe; fixed to guard_train | unverified (no) |

## Benign multi-turn history (screened by the zero-shot guard in S02b)
| source | where | notes |
|---|---|---|
| ultrachat_train | HF `HuggingFaceH4/ultrachat_200k` train_sft | EN, MIT |
| kovast | HF `maywell/koVast` | KO, MIT, `conversations[{from,value}]` |
| koalpaca | HF `beomi/KoAlpaca-v1.1a` | KO single pairs, licence unspecified |
| square_resp | SQuARe response_train.json (acceptable?=1) | KO |

## Evaluation only
| source | notes |
|---|---|
| polyguard_prompts | KO+EN WildGuardTest (human-verified GPT-4o translation). Replaces wgm_ko_test, which covers the same items. |
| wjb_ko_eval | KO+EN adversarial harmful/benign |
| xstest_ko | KO+EN over-refusal (heegyu/XSTest_ko). Replaces the gated walledai/XSTest. |
| toxic_chat_test | EN real users (NC) |
| safeguard_pi_test, jbb, openai_mod | EN. In openai_mod, null category cells mean "not annotated" and are never read as 0. |
| korset | KO native red-team (gated, NC) |
| unsmile_valid, beep_dev, apeach, kotox_test (easy/normal/hard) | KO hate/offensive, including obfuscated text |
| csrt, redteam_kr, raccoon_kr, square_test_ood | code-switching, KO red-team, KO sensitive-safe |
| orbench_hard, orbench_toxic, ultrachat_test | EN over-refusal / toxic / benign multi-turn |
| mhj, dktc | human multi-turn attacks (EN) and threatening dialogues (KO); NC. MHJ ships the attacker's turns only (no model replies); since D44 consecutive user turns stay separate (before, each attack was merged into one message) |

## Considered and excluded
| dataset | why |
|---|---|
| heegyu/PKU-SafeRLHF-ko | response-level labels only; no prompt label |
| reshabhs/SPML | the injection label depends on a system prompt this guard does not see |
| Anthropic hh-rlhf red-team | ratings measure attack success, not prompt harm; would need relabelling |
| deepset/prompt-injections | small; mixed German; known label errors (see project label-quality audit) |
| KorNAT, KoBBQ, K-HALU, KoreaGEO, multimodal sets | not moderation data |
| K-Guardrail processed copies of UnSmile / APEACH-train / selectstar | duplicates of the originals |
| EN-KO jailbreak prompts (koreascience 2025), KoTox | no public download; KoTox's political-opinion rows would teach over-refusal |
| KDPII (PII dialogues) | whether a user sharing their own personal data is "unsafe" is a policy decision; not included by default |

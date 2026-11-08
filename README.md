# okchat_guardrail

This is a multi-turn **Korean + English** LLM guardrail: a LoRA-fine-tuned **Qwen3Guard-Gen-8B** scores every user turn in its dialogue context, a small **Korean hate-speech classifier** runs beside it, and the serving decision comes either from the guard (+ hate classifier) or from a calibrated **LightGBM** trajectory model, whichever S10 shows is worth it. Decisions are graded (block / answer safely + review / answer with care / allow) with per-language thresholds, and the assistant's draft reply gets a second check. All of it runs from one notebook: `okchat_guardrail.ipynb`.

Run outputs (`artifacts/`, `releases/`, `feedback/`, `cache/`) are not in git; they are too large and partly licensed. This README therefore records the results of the latest full run.

## Contents
- [Results of full run 2 (2026-11-05)](#results-of-full-run-2-2026-11-05)
- [What changed between run 1 and run 2 (D44–D62)](#what-changed-between-run-1-and-run-2-d44d62)
- [Quick start](#quick-start-server) · [Saving and resume](#saving-and-resume-rules) · [Data](#data-see-datasetsmd) · [Memory](#memory-about-23-gb-ram) · [Feedback loop](#continual-improvement-feedback-loop) · [Layout](#layout)
- [Tests](#tests) · [Run cost](#run-cost) · [Known limitations](#known-limitations)

## Results of full run 2 (2026-11-05)

### Setup
| | |
|---|---|
| Hardware | 1 × NVIDIA GeForce RTX 4090 (23.5 GB VRAM), 1 TB RAM → `<30 GB` hardware profile |
| Guard | `Qwen/Qwen3Guard-Gen-8B`, 4-bit QLoRA, LoRA r=16, 2k context |
| SFT data | 108,801 examples (100k + 8,801 base-model replay), 1 epoch, 500 validation examples |
| Hate classifier | `beomi/KcELECTRA-base-v2022`, 90,753 rows (12,810 hateful), 2 epochs |
| Attack generator (S03b) | `Qwen/Qwen2.5-7B-Instruct` 4-bit |
| Baseline (S09) | `meta-llama/Llama-Guard-4-12B` 4-bit, zero-shot |
| Data | 52 sources loaded, **0 failed** (25 train, 5 benign history pools, 22 evaluation); 563,896 rows |
| Test set | 101,229 conversations (48,797 unsafe, 52,432 safe), 198,145 scored prefixes |
| Operating point | benign FPR ≤ 1% (`fpr_budget=0.01`), per-language thresholds with a Clopper–Pearson 95% margin |
| Release | `v20261105-005127-weekly` is `CURRENT`; serving system = LightGBM stack (`okchat_lgb`) |

"Recall" is the share of unsafe conversations flagged at any turn; "FPR" is the share of safe conversations flagged at any turn. All numbers are conversation-level on the held-out test split unless the row says otherwise.

### Headline
- **The served stack catches 60.9% of unsafe conversations at 0.95% FPR** (ROC AUC 0.962). Fine-tuning lifts Qwen3Guard-8B from 35.5% to 53.1–58.8% recall at the same FPR; zero-shot Llama Guard 4 reaches 13.4%.
- **English is much easier than Korean:** stack recall 75.1% (en) vs 55.0% (ko).
- **Human multi-turn attacks remain the weak spot:** 40.0% recall (AIHub 558 43.1%, MHJ 30.1%, DKTC 27.1%); the external multi-turn MTID test set reaches only 11.4%. 45.5% of red-team attacks get through (crescendo 85%).
- **The stack's win is smaller than S10 reports.** S10 compared the stack with the guard + hate OR-rule (44.6% / 27.7% recall on external / human multi-turn) and picked the stack. Against the plain fine-tuned guard with the strict score (53.7% / 38.4%), the stack adds only +1.0 and +1.6 points, below the 2-point `stack_min_gain`; `tools/analyze_run.py` therefore recommends **guard**. The stack *without* memory clears the bar (+3.0 / +2.4).
- **The hate-classifier OR-rule hurts:** guard + hate 50.9% vs guard alone 58.8% overall. To stay inside the FPR budget, its combined score ends up with a threshold above 1.0 (1.013 global).
- **The response check hurts as configured:** on the 6,000 conversations with assistant replies, adding the reply score drops recall from 63.3% to 36.7% (FPR 1.30% → 0.85%).

### Overall (all 101,229 test conversations)
| System | Recall % [95% CI] | FPR % | ROC AUC | PR AUC | FPR % at 90% recall | Turn-level ROC AUC |
|---|---|---|---|---|---|---|
| Qwen3Guard-8B zero-shot | 35.5 [35.1–35.9] | 1.19 | 0.868 | 0.874 | 42.9 | 0.880 |
| Llama Guard 4 zero-shot | 13.4 [13.1–13.7] | 0.74 | 0.796 | 0.785 | 54.5 | 0.839 |
| Hate classifier alone | 18.2 [17.9–18.5] | 0.79 | 0.696 | 0.730 | 77.8 | 0.720 |
| Fine-tuned guard, P(Unsafe) | 53.1 [52.7–53.5] | 1.15 | 0.954 | 0.953 | 13.0 | 0.947 |
| Fine-tuned guard, P(Unsafe)+P(Contr.) | 58.8 [58.4–59.2] | 1.09 | 0.958 | 0.959 | 11.6 | 0.951 |
| Fine-tuned guard + hate OR-rule | 50.9 [50.4–51.3] | 1.12 | 0.956 | 0.950 | 11.2 | 0.949 |
| LightGBM stack, no memory | 62.2 [61.7–62.6] | 1.06 | 0.961 | 0.962 | 10.2 | 0.951 |
| **LightGBM stack (served)** | 60.9 [60.5–61.3] | 0.95 | 0.962 | 0.963 | 10.1 | 0.953 |

### By language
| System | en recall | FPR | ko recall | FPR | mixed recall | FPR |
|---|---|---|---|---|---|---|
| Qwen3Guard-8B zero-shot | 56.9 | 1.67 | 25.1 | 0.86 | 57.1 | 0.51 |
| Llama Guard 4 zero-shot | 23.9 | 0.58 | 9.1 | 0.87 | 14.7 | 0.17 |
| Hate classifier alone | 3.0 | 0.68 | 25.8 | 0.88 | 0.7 | 0.00 |
| Fine-tuned guard, P(Unsafe) | 75.7 | 1.25 | 43.6 | 1.09 | 58.7 | 0.34 |
| Fine-tuned guard, P(Unsafe)+P(Contr.) | 75.6 | 1.47 | 51.8 | 0.84 | 61.8 | 0.17 |
| Fine-tuned guard + hate OR-rule | 75.6 | 1.47 | 40.6 | 0.89 | 54.7 | 0.17 |
| LightGBM stack, no memory | 76.3 | 1.27 | 56.1 | 0.92 | 67.2 | 0.51 |
| **LightGBM stack (served)** | 75.1 | 1.16 | 55.0 | 0.82 | 63.8 | 0.34 |
| _unsafe / safe conversations_ | 13,244 | 21,525 | 32,839 | 30,313 | 2,714 | 594 |

### By evaluation suite
*human multi-turn* = MHJ, DKTC, AIHub 558 (native human dialogues); *external* = evaluation-only sets never trained on; *in distribution* = held-out test rows of training sources.

| System | external recall | FPR | human multiturn recall | FPR | in distribution recall | FPR |
|---|---|---|---|---|---|---|
| Qwen3Guard-8B zero-shot | 44.3 | 2.40 | 8.9 | 1.39 | 44.8 | 0.69 |
| Llama Guard 4 zero-shot | 18.9 | 0.76 | 6.7 | 0.72 | 14.4 | 0.74 |
| Hate classifier alone | 11.2 | 0.48 | 26.4 | 0.68 | 17.2 | 0.93 |
| Fine-tuned guard, P(Unsafe) | 55.1 | 1.77 | 27.5 | 0.81 | 64.8 | 0.98 |
| Fine-tuned guard, P(Unsafe)+P(Contr.) | 53.7 | 1.89 | 38.4 | 1.08 | 71.0 | 0.80 |
| Fine-tuned guard + hate OR-rule | 44.6 | 1.87 | 27.7 | 0.65 | 65.0 | 0.93 |
| LightGBM stack, no memory | 56.7 | 1.59 | 40.8 | 1.12 | 75.0 | 0.85 |
| **LightGBM stack (served)** | 54.7 | 1.43 | 40.0 | 1.10 | 73.9 | 0.74 |
| _unsafe / safe conversations_ | 10,918 | 12,506 | 12,487 | 7,059 | 25,392 | 32,867 |

### Stack vs guard decision
| Comparison (per-language thresholds) | Suite | Stack recall | Guard recall | Stack FPR | Guard FPR | Only stack caught | Only guard caught | Stack clears +2 pts? |
|---|---|---|---|---|---|---|---|---|
| S10: stack vs guard + hate OR-rule | external | 54.7 | 44.6 | 1.43 | 1.87 | 1,375 | 269 | yes |
| S10: stack vs guard + hate OR-rule | human multi-turn | 40.0 | 27.7 | 1.10 | 0.65 | 1,901 | 370 | yes |
| Offline: stack vs guard (strict score) | external | 54.7 | 53.7 | 1.43 | 1.89 | 381 | 265 | no (+1.0) |
| Offline: stack vs guard (strict score) | human multi-turn | 40.0 | 38.4 | 1.10 | 1.08 | 702 | 508 | no (+1.6) |
| Offline: stack without memory vs guard (strict) | external | 56.7 | 53.7 | 1.59 | 1.89 | 457 | 128 | yes (+3.0) |
| Offline: stack without memory vs guard (strict) | human multi-turn | 40.8 | 38.4 | 1.12 | 1.08 | 749 | 447 | yes (+2.4) |

All McNemar p-values are below 1e-4. Top stack features by mean |SHAP|: `p_strict_cur` 2.10, `p_cur` 1.65, `mem_margin` 0.27, `p_ewma06` 0.26, `p_max` 0.16, `hate_p_cur` 0.15. LightGBM out-of-fold AUC 0.986 on 130,015 stack rows.

### Graded actions (served stack)
Share of conversations by final action, per-language thresholds.

| Suite | Label | n | allow | answer with care | answer safely + review | block |
|---|---|---|---|---|---|---|
| all | safe | 52,432 | 93.4% | 1.1% | 4.5% | 1.0% |
| all | unsafe | 48,797 | 16.0% | 1.3% | 21.7% | 60.9% |
| external | safe | 12,506 | 92.7% | 0.1% | 5.7% | 1.4% |
| external | unsafe | 10,918 | 23.7% | 0.3% | 21.3% | 54.7% |
| human multi-turn | safe | 7,059 | 92.4% | 1.9% | 4.6% | 1.1% |
| human multi-turn | unsafe | 12,487 | 25.8% | 3.4% | 30.8% | 40.0% |
| in distribution | safe | 32,867 | 93.8% | 1.4% | 4.1% | 0.7% |
| in distribution | unsafe | 25,392 | 7.9% | 0.7% | 17.5% | 73.9% |

So 84.0% of unsafe conversations get something other than a plain answer (block, safe answer or careful answer), and so do 6.6% of safe ones.

### Threshold rule: global vs per-language + CP95
| System | Rule | All recall | All FPR | en recall | en FPR | ko recall | ko FPR | mixed recall | mixed FPR |
|---|---|---|---|---|---|---|---|---|---|
| stack | one global threshold (old rule) | 63.8 | 1.0 | 69.3 | 0.7 | 59.8 | 1.3 | 85.6 | 2.2 |
| stack | per-language + CP95 (served) | 60.9 | 1.0 | 75.1 | 1.2 | 55.0 | 0.8 | 63.8 | 0.3 |
| guard (strict) | one global threshold (old rule) | 60.0 | 0.9 | 65.2 | 0.6 | 56.2 | 1.1 | 81.0 | 1.9 |
| guard (strict) | per-language + CP95 | 58.8 | 1.1 | 75.6 | 1.5 | 51.8 | 0.8 | 61.8 | 0.2 |

The global threshold leaves Korean (1.3%) and mixed (2.2%) over the 1% FPR budget. Per-language thresholds bring both under it, but English rises to 1.2% (guard 1.5%), and overall stack recall drops by 2.9 points.

### Red team and release gate
- **Red-team pass** (200 attacks, 200 benign twins, `okg/redteam.py`): attack success rate **45.5%**: crescendo 85%, decomposition 75%, role-play 55%, padding 10%, long paste 2.5%. Benign intervention rate 25%.
- **Release gate:** both `v20261105-004331` and the weekly refresh `v20261105-005127-weekly` passed (89 checks, overall FPR 0.95% ≤ 1%). The weekly release has the same stage hashes, so its metrics are identical.
- **Serving smoke check** (RTX 4090, 2 calls, cache cold): p50 226 ms, p95 262 ms per `assess` call. Run `tools/bench_latency.py` for real p50/p95/p99.

### English flips (zero-shot caught, fine-tuned missed)
| Set | English positives | Zero-shot caught | Fine-tuned caught | Lost | Gained | Nearest training neighbours labelled Safe |
|---|---|---|---|---|---|---|
| orbench_toxic | 655 | 652 | 533 | 119 | 0 | 86.6% (74.8% from OR-Bench-80k) |
| xstest_ko | 200 | 195 | 181 | 19 | 5 | 61.4% |

The OR-Bench-80k label convention (borderline prompts marked Safe) still pulls toxic English prompts towards Safe despite the base-model replay (8,801 of 108,801 SFT examples). Examples: `artifacts/full/analysis/flip_examples.csv`.

### By attack family
| Family | Unsafe | Safe | Zero-shot | Llama Guard 4 | Guard (strict) | Stack recall | Stack FPR |
|---|---|---|---|---|---|---|---|
| benign_mirror | 0 | 2,400 | – | – | – | – | 2.2 |
| benign_native | 0 | 5,039 | – | – | – | – | 0.5 |
| benign_native_mt | 0 | 23 | – | – | – | – | 0.0 |
| benign_sensitive | 0 | 10,018 | – | – | – | – | 0.1 |
| codeswitch | 2,400 | 0 | 59.3 | 11.8 | 69.1 | 71.0 | – |
| codeswitch_native | 314 | 0 | 40.1 | 36.3 | 6.1 | 8.6 | – |
| comment | 8,558 | 10,033 | 12.8 | 6.0 | 50.1 | 58.2 | 0.9 |
| gen_crescendo | 42 | 0 | 7.1 | 4.8 | 50.0 | 33.3 | – |
| gen_crescendo_benign | 0 | 42 | – | – | – | – | 4.8 |
| gen_decomposition | 38 | 0 | 15.8 | 2.6 | 52.6 | 42.1 | – |
| gen_decomposition_benign | 0 | 38 | – | – | – | – | 5.3 |
| gen_roleplay | 46 | 0 | 26.1 | 2.2 | 58.7 | 52.2 | – |
| gen_roleplay_benign | 0 | 46 | – | – | – | – | 0.0 |
| human_attack | 521 | 0 | 6.0 | 4.0 | 36.1 | 30.1 | – |
| human_dialogue | 2,000 | 0 | 4.0 | 21.2 | 28.3 | 27.1 | – |
| late_insertion | 2,400 | 0 | 62.5 | 8.4 | 90.7 | 91.0 | – |
| native_attack | 937 | 0 | 5.2 | 2.3 | 37.0 | 36.8 | – |
| native_attack_mt | 5 | 0 | 0.0 | 0.0 | 20.0 | 20.0 | – |
| native_dialogue | 9,966 | 7,059 | 10.0 | 3.9 | 40.6 | 43.1 | 1.1 |
| obfuscated | 2,400 | 0 | 10.7 | 7.9 | 79.4 | 82.3 | – |
| obfuscated_native | 1,777 | 1,764 | 0.1 | 1.9 | 21.9 | 29.9 | 0.3 |
| single | 11,370 | 12,055 | 66.8 | 30.7 | 69.1 | 67.7 | 1.6 |
| single_adversarial | 6,023 | 3,915 | 68.8 | 14.3 | 86.2 | 87.0 | 1.1 |

### By source
Recall % per source (FPR % for the stack). *Run 1 stack* is the 2026-10-01 run (Qwen3Guard-Gen-4B, 60k examples, older code and a different split), shown for direction only.

| Source | Suite | Unsafe | Safe | Zero-shot | Llama Guard 4 | Guard (strict) | **Stack** | Stack FPR | Stack ROC AUC | Run 1 stack |
|---|---|---|---|---|---|---|---|---|---|---|
| aihub_558 | human multiturn | 9,966 | 7,059 | 10.0 | 3.9 | 40.6 | **43.1** | 1.1 | 0.940 | 40.1 |
| dktc | human multiturn | 2,000 | 0 | 4.0 | 21.2 | 28.3 | **27.1** | – | – | 29.4 |
| mhj | human multiturn | 521 | 0 | 6.0 | 4.0 | 36.1 | **30.1** | – | – | 28.4 |
| apeach | external | 988 | 1,012 | 11.2 | 3.3 | 37.6 | **44.3** | 1.1 | 0.936 | 40.7 |
| beep_dev | external | 122 | 149 | 5.7 | 3.3 | 25.4 | **32.8** | 0.0 | 0.981 | 22.1 |
| csrt | external | 314 | 0 | 40.1 | 36.3 | 6.1 | **8.6** | – | – | 41.7 |
| jbb | external | 100 | 100 | 90.0 | 73.0 | 94.0 | **93.0** | 8.0 | 0.980 | 88.0 |
| korset | external | 1,935 | 0 | 39.4 | 18.4 | 35.5 | **33.5** | – | – | – |
| kotox_test_easy | external | 230 | 229 | 0.0 | 0.9 | 26.5 | **30.0** | 0.9 | 0.925 | 51.3 |
| kotox_test_hard | external | 230 | 228 | 0.0 | 0.9 | 17.0 | **25.2** | 0.0 | 0.867 | 48.3 |
| kotox_test_normal | external | 230 | 230 | 0.0 | 1.3 | 21.3 | **28.7** | 0.0 | 0.912 | 53.0 |
| mtid_test | external | 298 | 299 | 2.0 | 0.7 | 11.4 | **11.4** | 2.7 | 0.741 | – |
| openai_mod | external | 513 | 1,149 | 66.5 | 42.9 | 87.1 | **80.3** | 6.6 | 0.950 | 73.5 |
| orbench_hard | external | 0 | 1,319 | – | – | – | **–** | 0.8 | – | – |
| orbench_toxic | external | 655 | 0 | 83.5 | 43.7 | 44.3 | **35.3** | – | – | 35.7 |
| polyguard_prompts | external | 881 | 1,119 | 69.9 | 33.5 | 75.9 | **75.4** | 2.8 | 0.960 | 76.4 |
| raccoon_kr | external | 56 | 0 | 3.6 | 0.0 | 26.8 | **25.0** | – | – | 30.4 |
| redteam_kr | external | 320 | 0 | 65.0 | 16.9 | 58.8 | **58.8** | – | – | 68.8 |
| safeguard_pi_test | external | 627 | 1,373 | 62.0 | 16.3 | 87.6 | **88.0** | 0.2 | 0.999 | 84.4 |
| square_test_ood | external | 0 | 255 | – | – | – | **–** | 0.0 | – | – |
| toxic_chat_test | external | 142 | 1,858 | 54.2 | 24.6 | 64.1 | **64.1** | 0.4 | 0.989 | 69.7 |
| ultrachat_test | external | 0 | 2,000 | – | – | – | **–** | 0.1 | – | – |
| unsmile_valid | external | 1,077 | 488 | 15.8 | 5.2 | 54.0 | **63.3** | 0.8 | 0.956 | 56.9 |
| wjb_ko_eval | external | 1,802 | 198 | 62.8 | 13.9 | 82.1 | **83.1** | 7.6 | 0.957 | 80.2 |
| xstest_ko | external | 398 | 500 | 63.8 | 43.7 | 41.7 | **43.0** | 0.2 | 0.965 | 46.2 |
| aegis2 | in distribution | 1,673 | 1,473 | 63.5 | 36.5 | 66.6 | **65.0** | 3.5 | 0.931 | 64.7 |
| aihub_71833 | in distribution | 2,377 | 0 | 14.6 | 9.8 | 55.0 | **61.5** | – | – | – |
| attackgen_crescendo | in distribution | 42 | 42 | 7.1 | 4.8 | 50.0 | **33.3** | 4.8 | 0.860 | – |
| attackgen_decomposition | in distribution | 38 | 38 | 15.8 | 2.6 | 52.6 | **42.1** | 5.3 | 0.886 | – |
| attackgen_roleplay | in distribution | 46 | 46 | 26.1 | 2.2 | 58.7 | **52.2** | 0.0 | 0.934 | – |
| beep | in distribution | 363 | 615 | 2.8 | 1.9 | 26.7 | **34.2** | 0.0 | 0.960 | 27.7 |
| curse | in distribution | 0 | 776 | – | – | – | **–** | 0.8 | – | – |
| do_not_answer | in distribution | 173 | 0 | 36.4 | 19.1 | 43.4 | **43.9** | – | – | 43.3 |
| itw_jailbreak | in distribution | 192 | 0 | 64.6 | 10.4 | 75.0 | **74.5** | – | – | 70.1 |
| jailbreak_cls | in distribution | 62 | 103 | 82.3 | 9.7 | 85.5 | **85.5** | 0.0 | 0.991 | 81.2 |
| k_haters | in distribution | 850 | 998 | 10.6 | 8.0 | 52.2 | **49.1** | 0.9 | 0.957 | 44.6 |
| kmhas | in distribution | 1,201 | 2,181 | 6.0 | 1.8 | 49.5 | **68.7** | 2.2 | 0.957 | 50.6 |
| kold | in distribution | 0 | 1,968 | – | – | – | **–** | 0.2 | – | – |
| kosbi | in distribution | 0 | 1,053 | – | – | – | **–** | 0.0 | – | – |
| kotox_train | in distribution | 1,087 | 1,077 | 0.1 | 2.4 | 22.2 | **31.2** | 0.4 | 0.898 | 50.1 |
| mtid | in distribution | 361 | 343 | 3.9 | 1.1 | 19.7 | **17.2** | 3.8 | 0.825 | – |
| mtid_ko_mt | in distribution | 4 | 0 | 0.0 | 0.0 | 0.0 | **0.0** | – | – | – |
| orbench_80k | in distribution | 0 | 3,936 | – | – | – | **–** | 0.1 | – | – |
| pi_bench_kr | in distribution | 418 | 567 | 56.2 | 12.9 | 78.0 | **79.4** | 0.4 | 0.991 | 80.0 |
| safeguard_pi | in distribution | 178 | 401 | 60.7 | 15.2 | 87.6 | **88.2** | 0.5 | 0.996 | 84.1 |
| safemt_1k | in distribution | 278 | 0 | 10.4 | 5.8 | 87.1 | **89.6** | – | – | 96.9 |
| safemt_1k_ko_mt | in distribution | 1 | 0 | 0.0 | 0.0 | 100.0 | **100.0** | – | – | 83.3 |
| square_q | in distribution | 0 | 4,008 | – | – | – | **–** | 0.0 | – | – |
| synth_benign_mirror | in distribution | 0 | 2,400 | – | – | – | **–** | 2.2 | – | – |
| synth_benign_native | in distribution | 0 | 2,397 | – | – | – | **–** | 0.0 | – | – |
| synth_codeswitch | in distribution | 2,400 | 0 | 59.3 | 11.8 | 69.1 | **71.0** | – | – | 91.4 |
| synth_late_insertion | in distribution | 2,400 | 0 | 62.5 | 8.4 | 90.7 | **91.0** | – | – | 89.4 |
| synth_obfuscated | in distribution | 2,400 | 0 | 10.7 | 7.9 | 79.4 | **82.3** | – | – | 82.3 |
| toxic_chat_train | in distribution | 84 | 900 | 70.2 | 27.4 | 72.6 | **71.4** | 0.2 | 0.995 | 69.9 |
| ultrachat_train_ko_mt | in distribution | 0 | 23 | – | – | – | **–** | 0.0 | – | – |
| unsmile | in distribution | 1,580 | 793 | 18.0 | 5.7 | 54.3 | **62.7** | 1.3 | 0.952 | 54.9 |
| wgm_ko_train | in distribution | 3,288 | 2,848 | 84.0 | 31.6 | 89.9 | **90.1** | 0.5 | 0.994 | 89.5 |
| wjb_ko_train | in distribution | 3,896 | 3,881 | 73.5 | 17.7 | 89.7 | **89.9** | 0.5 | 0.995 | 89.2 |

### Training and stage statistics
| Stage | Result |
|---|---|
| S01 sources | 563,896 rows from 52 sources, 0 failed |
| S02 split | guard_train 319,097 · guard_val 23,926 · stack 95,622 · test 122,237 rows; 1,972 rows dropped for conflicting labels, 1,042 train rows dropped for overlapping evaluation sets |
| S02b screen | 57,740 benign history conversations kept, 4,378 dropped (WildChat-KO 1,548, KoVast 1,105, SQuARe resp 1,000, KoAlpaca 470, UltraChat 255) |
| S03 synth | 59,992 synthetic conversations (12k each: code-switch, late insertion, obfuscated, benign mirror, benign native) |
| S03b attack gen | 1,210 Korean dialogues kept of 4,800 jobs (crescendo 206, decomposition 173, role-play 226, each with a benign twin); main losses: bad JSON 1,830, unpaired 1,164, wrong turn count 462 |
| S05 prefixes | 1,002,300 prefixes from 564,744 conversations |
| S06 SFT | validation loss 0.067, token accuracy 97.5% (the reported train loss 0.006 covers only the last resumed session) |
| S06h hate | validation ROC AUC 0.923 |
| S07 scoring | 353,965 prefixes, parse rate 100%, 419 long messages scored in chunks |
| S07b memory | 50,000 attack + 50,000 benign entries (e5 encoder) |
| S07c response | 28,538 replies in 12,000 conversations |
| S08 stacker | 130,015 rows, out-of-fold AUC 0.986 |
| S09 Llama Guard 4 | 204,356 prefixes |

Regenerate the offline analyses with `python tools/analyze_run.py --mode full` → `artifacts/full/analysis/REPORT.md` (no GPU, about 3 minutes). The base-feature ablation in that report is a no-op for this run, because base features were already off.

### What run 2 says to fix next
- **Human multi-turn recall (40%)** and the red-team ASR (crescendo 85%) are the main gaps.
- **Regressions vs run 1:** KOTOX 48–53% → 25–30%, CSRT 41.7% → 8.6%, synthetic code-switch 91.4% → 71.0%, red-teaming-KR 68.8% → 58.8%. Run 1 used a different model and split, so check these sets before attributing the drop.
- **Gains vs run 1:** Korean hate/comment sets (K-MHaS 50.6 → 68.7, UnSmile 54.9 → 62.7, BEEP 27.7 → 34.2, K-HATERS 44.6 → 49.1, APEACH 40.7 → 44.3), JBB 88 → 93, OpenAI moderation 73.5 → 80.3.
- **Decision system:** either serve the guard with the strict score, or the stack without memory, which is the only stack variant that clears `stack_min_gain` against it.
- **Drop or re-tune** the hate OR-rule and the response check; both lower recall at the FPR budget.

## What changed between run 1 and run 2 (D44–D62)
Run 1 (2026-10-01) used Qwen3Guard-Gen-4B, 60k SFT examples and the code before these changes: 67.7% recall at 1.12% FPR on its own test split, human multi-turn 34.7%. Its artifacts are kept locally in `artifacts/full_run1_20261001/`.

| Review finding | Fix | Where |
|---|---|---|
| Cost grows with the square of conversation length | per-turn score cache keyed by conversation prefix; each turn scored once | `okg/cache.py`, `okg/inference.py` [D45] |
| Base-model pass doubles GPU cost for SHAP 0.027 | base features off (`use_base_features=False`); no base pass at serving; ablation in `tools/analyze_run.py` | `okg/features.py` [D45] |
| Padding pushes the setup out of the window; long pastes lose their head | opening pair pinned; head+tail trimming; long messages scored in chunks (max); carried risk for guard-only | `okg/guard.py` `fit_window` / `turn_prompts` [D44] |
| MHJ (human multi-turn) recall 28% | **bug:** MHJ's user-only turns were merged into one message (1.0 turns/conv); now kept separate | `okg/data_sources.py` [D44] |
| Latency unknown, no failure policy | `tools/bench_latency.py` (p50/p95/p99, cache on/off); timeout → `guard_fail_mode="safe_answer"` | [D49] |
| Only user turns scored | response moderation of the assistant draft (`assess_response`), S07c offline evaluation | [D51] |
| Synthetic multi-turn data too easy | S03b: attacker-LLM Korean crescendo / decomposition / role-play with benign twins; AIHub 558 30k → 120k talksets | `okg/attackgen.py` [D50] |
| English regression after tuning | flip analysis (OR-Bench-80k label-convention conflict); 10% base-model replay in SFT | `okg/analysis.py`, `okg/guard.py` [D52] |
| Korean hate speech 22–57% | KcELECTRA hate classifier (S06h) as features and OR-rule | `okg/hate.py` [D54] |
| Stack not clearly better than the guard | `decision_system="auto"`: stack only if it clearly wins on external AND human multi-turn sets | `okg/evaluate.py` [D46] |
| Training under-used | ≤100k examples, 1 epoch (D53 had 200k × 2, ~60 h; cut for 10 h sessions [D57]), 8B as 4-bit QLoRA under 30 GB | `okg/config.py` [D53][D57] |
| One threshold, 1.12% test FPR on a 1% budget | per-language thresholds with a Clopper–Pearson 95% margin; graded actions | `okg/stacker.py`, `okg/policy.py` [D47][D48] |
| No monitoring | block rate by language/hour, PSI drift, review backlog, reviewer κ, alerts | `okg/monitoring.py` [D56] |
| Weak promotion gate | named checks MHJ / DKTC / AIHub 558 / shadow traffic; red-team pass vs CURRENT; `tools/redteam_release.py` for cron | `okg/release.py`, `okg/redteam.py` [D55] |
| Missing comparisons | Llama Guard 4 baseline on by default (fixed verdict-token detection, config fix; 4-bit on <30 GB GPUs [D58]); KorSET loaded (gated) | S09 [D25][D58] |
| More multi-turn and Korean data | XGuard-Train in full (30.7k); WildChat Korean real-user chats as benign history; MTID multi-turn attacks with turn-level labels (+ `mtid_test`); AIHub 71833 Korean harmful-expression sentences (also trains the hate classifier) | `okg/data_sources.py` [D59–D62] |

## Quick start (server)
```bash
pip install torch  # build matching your CUDA (tested: 2.14.0)
pip install -r requirements.txt
# HF and AIHub keys are hard-coded in local_secrets.py (env vars HF_TOKEN / AIHUB_API_KEY override them).
# Accept the terms of the gated HF datasets (XGuard-Train, MHJ, KorSET) and meta-llama/Llama-Guard-4-12B with the same account first.
# Request AIHub access to datasets 558 and 71833. Install bitsandbytes on GPUs under 30 GB.
jupyter notebook okchat_guardrail.ipynb
```
1. Run everything once with `RUN_MODE = "smoke"`. It takes a few minutes, even on CPU, and exercises every stage.
2. Set `RUN_MODE = "full"` and run top to bottom again. Smoke and full artifacts are kept apart (`artifacts/smoke`, `artifacts/full`).

The first full S01 downloads WildChat-4.8M shard by shard (~15 GB transferred once, a few MB kept).

## Saving and resume rules
- A stage's results are committed only when the whole stage finishes. It writes to `artifacts/<mode>/<stage>.__tmp__/`, renames that into place, and writes `_DONE.json` last.
- Long stages save each finished part: sources, synthesis shards, translation shards, scoring shards. If a run stops, re-running the cell skips the finished parts and re-runs the interrupted one.
- When a stage re-runs, its old marker is removed first. Downstream stages therefore refuse to run until the stage finishes again.
- A changed config value, changed stage code, or changed upstream result invalidates the stage automatically.
- SFT is the one exception to "an interrupted stage is re-run": `sft_resume_from_checkpoint=True` (default since D57) keeps Trainer checkpoints in `S06_sft.__parts__/checkpoints` every `save_steps=100` steps and resumes from the newest complete one, so training can span several 10 h sessions. A checkpoint cut off mid-write is discarded. Do not change the config or `okg/` code between sessions: the stage hash includes both, and a changed hash starts the stage over.

## Data (see DATASETS.md)
- **52 sources are enabled by default** (all 52 loaded in run 2):
  - 25 train, including native Korean multi-turn AIHub 558 and AIHub 71833 via `aihubshell`
  - 5 benign multi-turn history pools, including WildChat Korean
  - 22 evaluation sets
- **Unified schema:** every source is converted to one set of columns, with no nulls, and validated.
- **Quality filters:** each source is filtered, and the per-source audit, with drop reasons and examples, is written to `S01_sources/DATASET_AUDIT.md`.
- **AIHub:** the key is in `local_secrets.py`; the AIHub cell downloads the label files only.

## Memory (about 23 GB RAM)
- Datasets are filtered and sampled on disk before they are loaded, and one source is processed at a time.
- `after_stage()` runs after every stage: gc, CUDA cache, `malloc_trim`.
- `expandable_segments` is set before torch is imported.
- `cfg.apply_hardware_profile()` uses a 2k context on GPUs under 30 GB and trains the 8B model as 4-bit QLoRA (Qwen3Guard-Gen-4B only if `bitsandbytes` is missing). The Llama Guard 4 baseline is loaded in 4-bit there too [D58].
- If memory runs low, restart the kernel and re-run: finished stages are skipped.

## Continual improvement (feedback loop)
- **Log:** `OkChatGuard.from_release(cfg).assess(msgs, log=FeedbackLog(...))` records each assessment. Logs are PII-scrubbed and user ids are salted hashes; set `OKG_USER_SALT` on the server.
- **Daily:** `build_review_queue`, then reviewers fill in the CSV, which is saved to `feedback/<mode>/reviewed/`. Then `apply_reviews_to_live_memory` makes confirmations effective at once, with no training.
- **Weekly:** re-run SF1 → S10, then `make_release`. This refreshes LightGBM and the memory; the adapter is untouched.
- **Monthly:** set `feedback_to_guard_train=True` and re-run from S01 to retrain the adapter too.
- **Promotion gate:** a release becomes `CURRENT` only if no shared test set gets worse beyond noise. Roll back with `release.set_current(cfg, name)`.

## Layout
```
okchat_guardrail.ipynb   the pipeline (S00 sanity … S10 evaluation + live inference)
okg/                     code used by the notebook (every choice tagged [R#]/[D#])
                         feedback.py, memory.py, release.py = continual-improvement loop
                         cache.py, policy.py, inference.py = serving (cache, graded actions, fail mode)
                         attackgen.py, hate.py, redteam.py, monitoring.py, analysis.py = D50–D56
REFERENCES.md            sources [R#] and design decisions [D#]
DATASETS.md              every source: loader, fields, label mapping, licence
tests/                   unit tests + offline fixtures (pytest -q tests)
tools/build_notebook.py  regenerates the notebook (the checked-in notebook also has hand edits; diff before overwriting)
tools/analyze_run.py     offline analyses of a finished run (no GPU)
tools/bench_latency.py   per-turn latency on the target GPU
tools/redteam_release.py scheduled red-team pass (cron)
```

## Tests
- **Unit tests:** 78 test cases in `tests/` (CPU, offline fixtures; `pytest -q tests`). On 2026-11-08, 77 passed on a Mac; `test_stacker_fit_and_calibration` needs `shap` installed.
- **Full run 2 on real models (RTX 4090):** every stage S00 → S10 finished, including the ones not run before: the attacker LLM (S03b), KcELECTRA (S06h), response scoring (S07c), 4-bit 8B training, Llama Guard 4 (S09), serving from a release, the review queue, live memory, the weekly refresh (SF1 → S10 skipped as unchanged) and monitoring.
- **Earlier checks (smoke mode, CPU):**
  - The whole notebook executes end to end with zero errors in about 2 minutes, including two releases (the second correctly blocked by the gate), the response check and the timeout fallback.
  - Killing the scoring stage midway commits nothing, and the next run resumes at the first unfinished shard. Re-running the notebook skips finished stages.
  - Serving scores match the S08 offline scores (guard probabilities to 1e-7; stack exactly).
  - The LoRA loss is applied only to `Safety: …\nCategories: …<|im_end|>`, and `disable_adapter()` equals the pristine base model.
- **Not measured yet:** real p50/p95/p99 latency under load (`tools/bench_latency.py`), and the response check on MHJ (MHJ ships no model replies; replay it against the chatbot with `okg.redteam`).

## Run cost
Measured in run 2 on one RTX 4090 (24 GB). Stages that span several sessions record only their final session; the run took 2026-10-29 → 2026-11-05 in 10 h sessions.

| Stage | Recorded time | Note |
|---|---|---|
| S01 sources | 46 min | includes the WildChat-4.8M scan |
| S02 split + S02b screen | 8 min + 1.9 h | |
| S03b attack generation | 3.9 h | 4,800 jobs, 7B 4-bit |
| S04 translation | 1.8 h | 400 dialogues |
| S06 SFT | 2.5 h last session | 108.8k examples × 1 epoch, 8B 4-bit; spanned 2026-10-29 10:03 → 2026-10-30 16:14 |
| S06h hate classifier | 5 min | |
| S07 scoring | 6.9 h last session | 354k prefixes |
| S07b memory + S07c response | 23 min + 1.0 h | |
| S09 Llama Guard 4 | 5.0 h | 204k prefixes, 12B 4-bit |
| S08 stacker + S10 eval | 10 min + 2 min | |

On a GPU with 30 GB or more, the hardware profile trains the 8B model in bf16 instead of 4-bit.

## Known limitations
- The AIHub sets need manual approval (558, 71833; 71833 is for Korean nationals only), and KorSET and Llama Guard 4 are gated on Hugging Face.
- Several licences are non-commercial or unverified; see the S01 manifest. Train with `allow_noncommercial=False` for a commercial model.
- Synthetic multi-turn attacks can carry generator artefacts. Judge the model on the human (MHJ, DKTC, AIHub 558) and external Korean sets (KOTOX, CSRT, red-teaming-KR, KorSET), not on the synthetic test split alone.
- Several evaluation sets are small (attack-gen families 38–46 conversations, RACCOON 56, `mtid_ko_mt` 4), so their per-source numbers are noisy.

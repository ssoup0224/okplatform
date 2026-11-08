# okchat_guardrail

This is a multi-turn **Korean + English** LLM guardrail: a LoRA-fine-tuned **Qwen3Guard-Gen-8B** scores every user turn in its dialogue context, a small **Korean hate-speech classifier** runs beside it, and the serving decision comes either from the guard (+ hate classifier) or from a calibrated **LightGBM** trajectory model, whichever S10 shows is worth it. Decisions are graded (block / answer safely + review / answer with care / allow) with per-language thresholds, and the assistant's draft reply gets a second check. All of it runs from one notebook: `okchat_guardrail.ipynb`.

## Changes after the first full run (D44–D56)
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
| Missing comparisons | Llama Guard 4 baseline on by default (fixed verdict-token detection, config fix; 4-bit on <30 GB GPUs [D58]); KorSET needs gated access (below) | S09 [D25][D58] |
| More multi-turn and Korean data | XGuard-Train in full (30.7k); WildChat Korean real-user chats as benign history; MTID multi-turn attacks with turn-level labels (+ `mtid_test`); AIHub 71833 Korean harmful-expression sentences (also trains the hate classifier) | `okg/data_sources.py` [D59–D62] |

**Before the next full run, on your side:** accept the terms of `datumo/KorSET` and `meta-llama/Llama-Guard-4-12B` on Hugging Face with the account whose token is in `local_secrets.py` (both are gated; KorSET failed with `DatasetNotFoundError` in the last run). Request AIHub access to dataset 71833 as well (the AIHub cell downloads it next to 558). The first S01 also downloads WildChat-4.8M shard by shard (~15 GB transferred once, a few MB kept). Install `bitsandbytes` if the GPU has < 30 GB, so the 8B model trains in 4-bit.

Offline results on the existing full-run artifacts: `python tools/analyze_run.py --mode full` → `artifacts/full/analysis/REPORT.md`.

## Quick start (server)
```bash
pip install torch  # build matching your CUDA (tested: 2.14.0)
pip install -r requirements.txt
# HF and AIHub keys are hard-coded in local_secrets.py (env vars HF_TOKEN / AIHUB_API_KEY override them).
# Accept the terms of the gated HF datasets (XGuard-Train, MHJ, KorSET) with the same account first.
jupyter notebook okchat_guardrail.ipynb
```
1. Run everything once with `RUN_MODE = "smoke"`. It takes a few minutes, even on CPU, and exercises every stage.
2. Set `RUN_MODE = "full"` and run top to bottom again. Smoke and full artifacts are kept apart (`artifacts/smoke`, `artifacts/full`).

## Saving and resume rules
- A stage's results are committed only when the whole stage finishes. It writes to `artifacts/<mode>/<stage>.__tmp__/`, renames that into place, and writes `_DONE.json` last.
- Long stages save each finished part: sources, synthesis shards, translation shards, scoring shards. If a run stops, re-running the cell skips the finished parts and re-runs the interrupted one.
- When a stage re-runs, its old marker is removed first. Downstream stages therefore refuse to run until the stage finishes again.
- A changed config value, changed stage code, or changed upstream result invalidates the stage automatically.
- SFT is the one exception to "an interrupted stage is re-run": `sft_resume_from_checkpoint=True` (default since D57) keeps Trainer checkpoints in `S06_sft.__parts__/checkpoints` every `save_steps=100` steps and resumes from the newest complete one, so training can span several 10 h sessions. A checkpoint cut off mid-write is discarded. Do not change the config or `okg/` code between sessions: the stage hash includes both, and a changed hash starts the stage over.

## Data (see DATASETS.md)
- **48 sources are enabled by default:**
  - 23 train, including native Korean multi-turn AIHub 558 via `aihubshell`
  - 4 benign multi-turn pools
  - 21 evaluation sets
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
tests/                   unit tests + offline fixtures (pytest -q tests)
tools/build_notebook.py  regenerates the notebook (the checked-in notebook also has hand edits; diff before overwriting)
tools/analyze_run.py     offline analyses of a finished run (no GPU)
tools/bench_latency.py   per-turn latency on the target GPU
tools/redteam_release.py scheduled red-team pass (cron)
```

## What was tested and what wasn't
- **After the D44–D56 changes (CPU, offline fixtures):** 51 unit tests pass (`tests/test_fixes.py` adds window pinning, chunking, MHJ parsing, thresholds, policy, cache, gate, red-team runner, monitoring). The full smoke pipeline S00 → S10, two releases (the second correctly blocked by the gate), serving with cache, response check, timeout fallback, review queue, live memory and monitoring ran end to end in 2 minutes. Serving scores match S08 offline scores (guard probabilities to 1e-7; stack exactly).
- **Not yet run with real models:** the attacker LLM (S03b), KcELECTRA (S06h), response scoring on Qwen3Guard, 4-bit 8B training, the Llama Guard 4 stage, and GPU latency. Run `tools/bench_latency.py` on the target GPU.
- **Tested (CPU, offline build machine):**
  - All 31 unit tests pass.
  - The whole notebook executes end to end in smoke mode with zero errors.
  - Killing the scoring stage midway commits nothing, and the next run resumes at the first unfinished shard.
  - Re-running the notebook skips finished stages.
  - The full feedback loop in smoke mode:
    - Simulated traffic is logged, the review queue is built, simulated reviewers' CSV is imported, and a confirmed attack reaches similarity 1.0 in the live memory immediately.
    - The weekly refresh skipped S01–S07, including the adapter.
    - The release gate blocked a candidate that regressed beyond noise.
  - The LoRA loss is applied only to `Safety: …\nCategories: …<|im_end|>`, and `disable_adapter()` equals the pristine base model.
- **Loaded for real during testing:** the GitHub-hosted Korean sources (K-Guardrail processed/translated files, SQuARe, DKTC).
- **Tested against schema-faithful fixtures only:** the HuggingFace datasets. The build machine could not reach huggingface.co, so these first load for real on your server. A source that fails is listed in the S01 manifest and skipped, so it does not stop the run.
- **Not tested here:**
  - The real Qwen3Guard-Gen-8B. The S00 sanity cell checks its template, token ids and zero-shot verdicts before any expensive work.
  - The Llama Guard 4 baseline (optional).
  - 4-bit QLoRA.

## Rough full-run cost (single 80 GB GPU; estimates, not measured)
| Stage | Time |
|---|---|
| Translation | about 4 h (1,000 dialogues) |
| Attack generation (S03b) | about 3–5 h (≈4.8k Korean dialogues incl. twins, 7B model) |
| SFT | ≈ 6–10 h per 60k examples per epoch: the default 100k × 1 epoch is roughly 10–17 h, resumable across sessions (200k × 2 epochs would be 40–65 h) |
| Hate classifier (S06h) | under 30 min |
| Scoring | about 3 h (base + tuned) + response scoring ≈ 1 h |
| Llama Guard 4 baseline | about 2–4 h (test + 4k benign stack conversations) |
| LightGBM + eval | minutes |

With a 48 GB GPU, set `load_in_4bit=True` or use `Qwen/Qwen3Guard-Gen-4B`.

## Known limitations
- KoRSET (no public download) and the AIHub sets (manual access) are not loaded automatically. Use `custom_sources` once you have the files.
- Several licences are non-commercial or unverified; see the S01 manifest. Train with `allow_noncommercial=False` for a commercial model.
- Synthetic multi-turn attacks can carry generator artefacts. Judge the model on the human (MHJ, DKTC) and external Korean sets (KOTOX, CSRT, red-teaming-KR), not on the synthetic test split alone.

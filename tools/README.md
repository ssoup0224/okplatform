# okchat_guardrail

This is a multi-turn **Korean + English** LLM guardrail: a LoRA-fine-tuned **Qwen3Guard-Gen-8B** scores every user turn in its dialogue context, and a calibrated **LightGBM** trajectory model makes the final block/allow decision. All of it runs from one notebook: `okchat_guardrail.ipynb`.

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
- SFT checkpoints are not resumed by default, per your rule that interrupted work is re-run. Set `sft_resume_from_checkpoint=True` to resume Trainer checkpoints instead.

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
- `cfg.apply_hardware_profile()` switches to Qwen3Guard-Gen-4B and a 2k context on GPUs under 30 GB.
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
REFERENCES.md            sources [R#] and design decisions [D#]
tests/                   unit tests + offline fixtures (pytest -q tests)
tools/build_notebook.py  regenerates the notebook
```

## What was tested and what wasn't
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
| SFT | about 6–10 h (≤60k examples, 1 epoch) |
| Scoring | about 3 h (base + tuned) |
| LightGBM + eval | minutes |

With a 48 GB GPU, set `load_in_4bit=True` or use `Qwen/Qwen3Guard-Gen-4B`.

## Known limitations
- KoRSET (no public download) and the AIHub sets (manual access) are not loaded automatically. Use `custom_sources` once you have the files.
- Several licences are non-commercial or unverified; see the S01 manifest. Train with `allow_noncommercial=False` for a commercial model.
- Synthetic multi-turn attacks can carry generator artefacts. Judge the model on the human (MHJ, DKTC) and external Korean sets (KOTOX, CSRT, red-teaming-KR), not on the synthetic test split alone.

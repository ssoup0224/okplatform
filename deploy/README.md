# Deploying the guard API

`deploy/app.py` serves a release (run 2: `v20261105-005127-weekly`) over HTTP so a chatbot backend can check
each user turn before calling its LLM. One process holds one model on one GPU.

> **Before real traffic:** the run 2 release was trained on 13 non-commercial or unlicensed sources, and
> `memory/` stores 41,021 raw texts from them. Legal review is pending. Keep `OKG_SHADOW=1` (the default)
> until it is cleared.

## What the server needs

| Item | Where it comes from |
|---|---|
| NVIDIA GPU with at least 24 GB (L4, A10G, RTX 4090, A100 …) | the host |
| NVIDIA driver; for Docker also the NVIDIA Container Toolkit | the host |
| ~40 GB disk: base model 16 GB, release 0.85 GB, image ~10 GB | the host |
| Release folder `releases/full/v20261105-005127-weekly/` + `releases/full/CURRENT` | copied from the training machine (not in git) |
| `Qwen/Qwen3Guard-Gen-8B`, `intfloat/multilingual-e5-small` | downloaded from Hugging Face on first start (neither is gated) |

The server pins the scoring settings the release was calibrated with: 8B in 4-bit NF4, 2,048-token window,
batch 4. Do not swap in bf16 or a 4k window on a bigger GPU without re-checking the thresholds.

For a RunPod Pod (RTX 4090), follow [runpod/README.md](runpod/README.md) instead of steps 1–2 below.

## 1. Copy the release to the host

```bash
rsync -av --progress releases/full/CURRENT releases/full/v20261105-005127-weekly <host>:/srv/okg/releases/full/
```

## 2a. Run with Docker (any Linux GPU VM)

Build on an x86-64 machine (or with `--platform linux/amd64`) and either build on the host or push to a registry:

```bash
docker build --platform linux/amd64 -f deploy/Dockerfile -t okchat-guard:1 .
```

`.dockerignore` lets only `okg/` and `deploy/` into the image: no releases, data, caches or `local_secrets.py`.

```bash
docker run -d --name okchat-guard --gpus all --restart unless-stopped -p 8080:8080 \
  -v /srv/okg/releases:/app/releases:ro \
  -v /srv/okg/feedback:/app/feedback \
  -v /srv/okg/hf-cache:/data/hf-cache \
  -e OKG_API_KEY="$(cat /srv/okg/api_key)" \
  okchat-guard:1
```

The first start downloads 16 GB and loads the model (several minutes); `/readyz` returns 503 until then.
On hosts that run your image directly (RunPod pods, ECS, Kubernetes), use the same image, mounts and
variables; give the container a GPU and point the health check at `/readyz`.

## 2b. Run without Docker

```bash
python3.12 -m venv /srv/okg/venv && . /srv/okg/venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.14.0   # match the driver
pip install -r deploy/requirements.txt
OKG_PROJECT_DIR=/srv/okg OKG_API_KEY=... HF_HOME=/srv/okg/hf-cache \
  uvicorn deploy.app:create_app --factory --host 0.0.0.0 --port 8080 --workers 1
```

`OKG_PROJECT_DIR` is the folder that holds `releases/full/`. Keep it alive with systemd
(`Restart=always`), not a notebook or a terminal session.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `OKG_SHADOW` | `1` | `1`: `action` is always `allow`; the verdict is in `model_action`. `0`: enforce. |
| `OKG_API_KEY` | unset | When set, every `/v1/*` call needs `X-API-Key: <key>` or `Authorization: Bearer <key>`. Set it on any host reachable from outside. |
| `OKG_RELEASE` | `CURRENT` | Release folder name under `releases/<mode>/`. |
| `OKG_DECISION_SYSTEM` | release's (`stack`) | `guard` = fine-tuned guard, strict score, no hate OR-rule (what `tools/analyze_run.py` recommends after run 2); `guard_hate`; `stack`. |
| `OKG_RESPONSE_CHECK` | `0` | Reply check. Off: as calibrated in run 2 it cuts recall from 63.3% to 36.7%. |
| `OKG_TIMEOUT_MS` | `2000` | Per-request deadline, time waiting for the model included. Set it from measured p99 (`tools/bench_latency.py`). |
| `OKG_FAIL_MODE` | `safe_answer` | Action on timeout, overload or guard error: `safe_answer`, `block` or `allow`. |
| `OKG_MAX_QUEUE` | `8` | Requests in flight before the server answers `busy` (with the fail action) at once. |
| `OKG_LOG_CONVERSATIONS` | `0` | `1`: write PII-scrubbed conversations to `feedback/full/log/` for the review loop. Needs `OKG_USER_SALT`. |
| `OKG_USER_SALT` | — | Secret for hashing user ids. Never change it once logging has started. |
| `OKG_LIVE_MEMORY` | `1` | Add reviewer-confirmed entries from `feedback/full/memory_live/`. |
| `OKG_RUN_MODE` | `full` | `smoke` serves the tiny CPU release (wiring checks only). |

Server logs hold request ids, status, latency and actions, never conversation text or user ids.

## API

`POST /v1/assess` with the whole conversation; the newest message must be the user's. System messages are ignored.

```json
{"conv_id": "c-123", "user_id": "u-42",
 "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."},
              {"role": "user", "content": "..."}]}
```

```json
{"action": "allow", "model_action": "block", "shadow": true, "turn_idx": 1, "lang": "ko",
 "p_block": 0.97, "p_unsafe": 0.91, "p_contro": 0.05, "threshold": 0.9266, "guard_error": "",
 "latency_ms": 180.4, "release": "v20261105-005127-weekly"}
```

`POST /v1/assess_response` with `{"messages": [...], "draft": "..."}` returns `action` `allow` or `replace`.
`GET /healthz`, `/readyz`, `/v1/info` (release, thresholds, settings) and `/v1/stats` (counts, latency, cache).

## Calling it from the chatbot

| `action` | Chatbot does |
|---|---|
| `allow` | answer normally |
| `careful_answer` | answer with a caution instruction in the system prompt |
| `safe_answer` | answer with a safe-answer prompt (general information, no specifics); queue for review |
| `block` | send a refusal; do not call the LLM |

```python
import requests

FALLBACK = {"action": "safe_answer"}           # what to do when the guard cannot answer

def guard_check(messages, conv_id, user_id):
    try:
        r = requests.post(GUARD_URL + "/v1/assess", timeout=2.5,      # a bit above OKG_TIMEOUT_MS
                          headers={"X-API-Key": GUARD_KEY},
                          json={"messages": messages, "conv_id": conv_id, "user_id": user_id})
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return FALLBACK
```

Always follow `action`, never `model_action`: switching `OKG_SHADOW` on the server is then the only change
needed to start enforcing. The per-turn score cache lives in the process, so with several replicas route
each `conv_id` to the same one (consistent hashing); otherwise every request re-scores the whole history.

## Rollout

1. Shadow (`OKG_SHADOW=1`) on real traffic; review `model_action=block|safe_answer` and a sample of `allow`
   to measure the real false-positive and miss rates per language.
2. Enforce `block` first, then `safe_answer`; ramp traffic 10% → 50% → 100%.
3. Keep the other defenses (LLM safety prompt, rate limits, user reports): recall on the test sets is ~61%.

## Checks

```bash
pytest tests/test_serve.py                    # API behaviour with a fake guard (needs fastapi, httpx)
OKG_RUN_MODE=smoke uvicorn deploy.app:create_app --factory --port 8080   # real OkChatGuard, tiny CPU release
```

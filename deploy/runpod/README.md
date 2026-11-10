# Guard API on a RunPod Pod (RTX 4090)

The release was trained and calibrated on an RTX 4090 (24 GB, 8B in 4-bit), so a 4090 Pod scores exactly as
in run 2. This uses an always-on **Pod**, not Serverless: a cold start loads 16 GB of weights, far over the
2-second budget. General settings, API and chatbot integration are in [../README.md](../README.md).

## 1. Create the Pod (RunPod console)

1. Settings → SSH Public Keys: add the public half of a key on the training machine (`~/.ssh/id_ed25519.pub`).
2. Deploy a Pod:
   - GPU: **RTX 4090 × 1**. Secure Cloud is more reliable than Community Cloud for an always-on service.
   - Template: the official **RunPod PyTorch** template (SSH preconfigured). Its own torch is not used:
     `setup.sh` builds a separate Python 3.12 environment on the volume.
   - Container disk 30 GB; **volume disk 60 GB at `/workspace`** (models, venv and release live there and
     survive restarts).
   - Expose HTTP port **8080** (keep the template's ports) and TCP port **22**.
3. When it runs, open Connect and note the IP and port under "SSH over exposed TCP"
   (`ssh root@<ip> -p <port>`) and the Pod ID.

## 2. Copy code and release (on the training machine)

```bash
deploy/runpod/push.sh <ip> <port>
```

Copies `okg/`, `deploy/` and the CURRENT full release (845 MB) to `/workspace/okg`. Re-run it after code
changes; it only sends what changed. Neither `local_secrets.py` nor any training data is copied.

## 3. Set up and start (on the Pod)

```bash
ssh root@<ip> -p <port>
bash /workspace/okg/deploy/runpod/setup.sh        # once: venv, torch, packages, API key, 16 GB model download
nohup bash /workspace/okg/deploy/runpod/start.sh >> /workspace/okg/serve.log 2>&1 &
tail -f /workspace/okg/serve.log                  # "guard ready: release=v20261105-005127-weekly ..."
cat /workspace/okg/api_key                        # give this to the chatbot backend
```

`start.sh` restarts the server if it exits and refuses to start a second copy. Settings go in
`/workspace/okg/okg.env` (shadow mode is on); restart after editing:

```bash
pkill -f deploy/runpod/start.sh; pkill -f deploy.app:create_app
nohup bash /workspace/okg/deploy/runpod/start.sh >> /workspace/okg/serve.log 2>&1 &
```

## 4. Check it (from anywhere)

```bash
python deploy/check.py https://<pod-id>-8080.proxy.runpod.net --key <api key>
```

It waits for `/readyz`, prints the release and scoring settings (expect `4bit True`, `max_seq_len 2048`), then
sends benign and harmful Korean and English conversations and prints each `model_action` and latency.

## 5. Survive Pod restarts

A restart kills processes but keeps `/workspace`. Once the manual start works, set the Pod's
**Container Start Command** so the server comes back by itself, while the template's own start script still
runs SSH:

```bash
bash -c 'nohup bash /workspace/okg/deploy/runpod/start.sh >> /workspace/okg/serve.log 2>&1 & /start.sh'
```

If SSH does not come back after that change, the image has no `/start.sh`: clear the field and start manually.

## Notes

- The proxy URL `https://<pod-id>-8080.proxy.runpod.net` is public HTTPS. The API key is the only lock; keep it
  out of client-side code and rotate it by replacing `/workspace/okg/api_key` and restarting.
- The RunPod proxy cuts requests at 100 s. Guard calls finish within `OKG_TIMEOUT_MS` (2 s), so this only
  matters if the timeout is raised a lot.
- Stopping the Pod keeps the volume (still billed) but the same GPU may not be free when you resume.
  Terminating deletes the volume: re-run steps 2 and 3.
- User conversations are processed on RunPod's machines. Confirm that is acceptable before sending real traffic,
  and keep `OKG_LOG_CONVERSATIONS=0` until the logging policy is agreed.
- Licensing of the run 2 release is still under review: stay in shadow mode until it is cleared.

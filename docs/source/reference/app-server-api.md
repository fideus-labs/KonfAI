# App server HTTP API

The `konfai-apps-server` command launches a FastAPI server
(`konfai_apps.app_server:app`) that exposes packaged apps as **remote,
asynchronous jobs**. This page is the complete endpoint contract. Start the
server with {doc}`cli` and drive it with the {doc}`python-api` (`KonfAIAppClient`)
or plain HTTP.

## Authentication

Every route — including `/health` — is behind a bearer-token dependency:

- The server reads the token from the env var **`KONFAI_API_TOKEN`**.
- If it is **unset or empty, authentication is disabled** and all endpoints are
  open. (`konfai-apps-server --auth off` explicitly clears it.)
- If set, every request needs `Authorization: Bearer <token>`; a missing or
  non-bearer header returns **401 "Missing bearer token"**, a wrong value **401
  "Invalid token"**.

```{warning}
The server speaks **plain HTTP** — the token and the uploaded medical volumes
travel unencrypted. Terminate TLS with a reverse proxy before exposing it beyond
localhost, and keep the `--apps` allowlist tightly scoped (it is the trust
boundary that stops a token holder from making the resolver fetch arbitrary
repos). See {doc}`python-api`.
```

## Endpoints

### Health & system

| Method | Path | Params | Response |
| --- | --- | --- | --- |
| `GET` | `/health` | — | `{"status":"ok"}` |
| `GET` | `/available_devices` | — | `{"devices_index":[int], "devices_name":[str]}` |
| `GET` | `/ram` | — | `{"used_gb":float, "total_gb":float}` |
| `GET` | `/vram` | `devices` (repeated int, required) | `{"used_gb":float, "total_gb":float}` |

### App repository

| Method | Path | Response |
| --- | --- | --- |
| `GET` | `/repo_apps_list` | `{"apps":[str,…]}` — the configured allowlist |
| `GET` | `/repo_apps/{app_id}` | App metadata & capabilities (display name, description, `checkpoints_name`, `maximum_tta`, `mc_dropout`, `patch_size`, `inputs`/`outputs`, `terminology`, …). 404 if not allowlisted. |
| `GET` | `/repo_apps_config/{app_id}` | `application/zip` of the app's config files (no `.pt`). |

### Job submission

All five accept a **multipart form** and return the same job envelope:

| Method | Path | Runs |
| --- | --- | --- |
| `POST` | `/apps/{app_name}/infer` | `konfai-apps infer` |
| `POST` | `/apps/{app_name}/evaluate` | `konfai-apps eval` |
| `POST` | `/apps/{app_name}/uncertainty` | `konfai-apps uncertainty` |
| `POST` | `/apps/{app_name}/pipeline` | `konfai-apps pipeline` |
| `POST` | `/apps/{app_name}/fine_tune` | `konfai-apps fine-tune` |

```json
{
  "job_id": "<12 hex>",
  "status_url": "/jobs/<id>",
  "logs_url":   "/jobs/<id>/logs",
  "result_url": "/jobs/<id>/result"
}
```

**Multipart fields** (files via `File`, scalars via `Form`; `*_groups` are CSV
group-size lists used to re-split the flat file list into per-group folders):

- **infer** — `inputs` (files, required), `inputs_groups`, `ensemble` (0),
  `ensemble_models` (CSV), `tta` (0), `mc` (0), `uncertainty` (false),
  `prediction_file` (`Prediction.yml`), `gpu` (CSV), `cpu` (1), `quiet` (false).
- **evaluate** — `inputs`, `gt` (files, required), `mask` (files), the matching
  `*_groups`, `evaluation_file` (`Evaluation.yml`), `gpu`/`cpu`/`quiet`.
- **uncertainty** — `inputs`, `inputs_groups`, `uncertainty_file`
  (`Uncertainty.yml`), `gpu`/`cpu`/`quiet`.
- **pipeline** — union of the above; here `gt` is **required** and `uncertainty`
  defaults to **true**.
- **fine_tune** — `dataset` (single zip, required — extracted with zip-slip
  protection), `name` (`Finetune`), `epochs` (10), `it_validation` (1000),
  `models` (CSV), `config_file` (`Config.yml`), `lr` (optional), `gpu`/`cpu`/`quiet`.

### Job control

| Method | Path | Response |
| --- | --- | --- |
| `GET` | `/jobs/{job_id}` | `{"job_id","status","error"}` — status ∈ `queued/waiting/running/done/error/killed`. 404 unknown. |
| `GET` | `/jobs/{job_id}/logs` | `text/event-stream` (SSE) — see below. |
| `GET` | `/jobs/{job_id}/result` | `application/zip` (`result.zip`); **202** while running; **500** on error. |
| `POST` | `/jobs/{job_id}/kill` | `{"job_id","status","message"}` — SIGTERM → SIGKILL the process group. |

**SSE log stream** — each event is `data: <line>\n\n`; a `: keepalive` comment is
sent every 15 s of silence. Terminal markers are `__DONE__` and `__ERROR__ <msg>`.
Admission control: at most one stream per job (else 429), and a global cap of 200.

## Server limits & lifecycle

| Limit | Value | Effect |
| --- | --- | --- |
| Active jobs | `MAX_ACTIVE_JOBS = 32` | **429 "Server busy"** beyond it |
| Per-file upload | 2 GB | **413** on overflow |
| Total upload | 6 GB | **413** on overflow |
| GPU scheduling | one semaphore per visible GPU | auto mode waits for any free GPU; explicit mode acquires all requested (400 unknown id, 503 if none) |
| Result grace period | 120 s after completion | workspace and job are then removed — **download promptly** |

Jobs run in an isolated temp workspace; the command is built as an argv list and
launched with `subprocess.Popen` (no shell), so there is no shell injection.

## curl example

```bash
TOKEN=changeme
BASE=http://127.0.0.1:8000

curl -H "Authorization: Bearer $TOKEN" $BASE/health
curl -H "Authorization: Bearer $TOKEN" $BASE/repo_apps_list

# submit an inference job (one input group of 2 files → inputs_groups=2)
JOB=$(curl -s -H "Authorization: Bearer $TOKEN" \
  -F "inputs=@case_0000.nii.gz" -F "inputs=@case_0001.nii.gz" \
  -F "inputs_groups=2" -F "tta=4" -F "ensemble=3" -F "gpu=0" \
  "$BASE/apps/VBoussot%2FImpactSynth:MR/infer" | jq -r .job_id)

curl -N -H "Authorization: Bearer $TOKEN" $BASE/jobs/$JOB/logs   # stream logs (SSE)
curl -H "Authorization: Bearer $TOKEN" -o result.zip $BASE/jobs/$JOB/result   # 202 until done
curl -X POST -H "Authorization: Bearer $TOKEN" $BASE/jobs/$JOB/kill
```

## Next steps

- {doc}`python-api` — `KonfAIAppClient` wraps all of this for you
- {doc}`cli` — `konfai-apps-server` options
- {doc}`../usage/apps` — what an app is and how to run the server day-to-day

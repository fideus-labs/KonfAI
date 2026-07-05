# The apps layer (`konfai-apps`)

Once a KonfAI workflow is stable, package it as an **app** for a clean inference interface.
`konfai-apps` is a separate package (own `pyproject.toml`, tests, CI) that layers on KonfAI's
public API. Use it when you want inference/evaluation/packaging rather than authoring raw YAML.

Two console scripts: `konfai-apps` (local or remote execution) and `konfai-apps-server`
(host apps behind an HTTP API). There is also a Python API under `konfai_apps`.

## What an app is, and where it resolves from

An **app bundle** contains a KonfAI config + custom `.py` module(s) + model weights
(+ an optional `requirements.txt`). The app id (and `--host`) decides the source:

| How you name the app | Source |
|---|---|
| a filesystem path or bare local name | **Local directory** |
| `repo_id:app_name` (one `:`, e.g. `VBoussot/ImpactSynth:sCT`) | **HuggingFace repo** (weights/config pulled via `huggingface_hub`) |
| any app id **with `--host`/`--port`/`--token`** | **Remote server** (`konfai-apps-server`) |

## ⚠️ Trust model — read before resolving an app

Resolving or running an app is **not** a pure data download. It:

- **copies the app's `.py` files into a run workspace and imports them** (the workspace is put
  on `sys.path`, and KonfAI imports the custom modules) — i.e. it **runs arbitrary code**;
- **pip-installs `requirements.txt` by default** on every local resolution — only missing or
  version-mismatched packages, core packages (`torch`, `konfai`, …) are never touched, and
  non-PEP 508 lines are skipped; opt out with `KONFAI_APPS_INSTALL_REQUIREMENTS=0`;
- for HuggingFace apps, **downloads weights/config over the network**;
- in remote mode, **uploads your inputs and config** to the server.

**Only resolve/run apps from sources you trust.** This is the same trust boundary as any
"download and execute" tool.

## `konfai-apps` subcommands

Inputs use `-i/--inputs` (repeatable; a file or a dataset dir), output goes to `-o/--output`
(default `./Output`), device is `--gpu` XOR `--cpu`.

| Command | Purpose | Example |
|---|---|---|
| `infer` | Run inference with an app | `konfai-apps infer VBoussot/ImpactSynth:sCT -i mr.nii.gz -o ./Output --gpu 0 --tta 4 --ensemble 3` |
| `eval` | Inference + evaluation vs ground truth | `konfai-apps eval my_app -i ./inputs --gt ./gt --mask ./masks -o ./Eval` |
| `uncertainty` | Uncertainty estimation | `konfai-apps uncertainty my_app -i mr.nii.gz -o ./Output` |
| `pipeline` | Infer, then eval and optionally uncertainty in one run | `konfai-apps pipeline my_app -i ./in --gt ./gt -o ./Out --ensemble 3 -uncertainty` |
| `fine-tune` | Fine-tune an app's checkpoint(s) into a new named app | `konfai-apps fine-tune my_app MyFT -d ./Dataset --models CV_0 CV_1 --epochs 20 --lr 1e-4` |
| `bundle` | Assemble a bundle (config + checkpoints + optional `Model.py`), optionally export ONNX | `konfai-apps bundle sCT --out ./bundles --app-json app.json --config Prediction.yml --checkpoint CV_0.pt --onnx` |
| `download` | Fetch a HuggingFace app's files into the local cache | `konfai-apps download VBoussot/ImpactSynth:sCT Prediction.yml` |

Inference knobs on `infer`/`pipeline`: `--tta N` (test-time augmentations), `--ensemble N` /
`--ensemble-models ...` (checkpoint ensembling), `--mc N` (Monte-Carlo dropout),
`--patch-size` / `--batch-size` (override inference patch/batch), `-uncertainty` (also write
the inference stack).

### Serving apps over HTTP

`konfai-apps-server` hosts a FastAPI app (uvicorn `konfai_apps.app_server:app`) and, unlike
the internal MCP server, **defaults to bearer auth** (`--auth bearer`, token from
`KONFAI_API_TOKEN`):

```bash
KONFAI_API_TOKEN=secret konfai-apps-server --apps ./apps.json --host 0.0.0.0 --port 8000
# clients then run konfai-apps with --host/--port/--token to execute remotely
konfai-apps infer my_app -i mr.nii.gz -o ./Out --host 127.0.0.1 --port 8000 --token secret
```

`--apps` is a required JSON file listing the app ids to serve; `--check` validates them
(no download) and `--download` pre-fetches them. Relevant env vars: `KONFAI_API_TOKEN`
(bearer token, client and server) and `KONFAI_IMPACTREG_REPO` (used by the IMPACT-Reg app).

## Published app bundles (`apps/`)

The `apps/` directory holds ready-to-use bundles — each an **independent pip package** that
layers on `konfai` + `konfai-apps` (excluded from the `konfai` wheel):

| Bundle | Task | Entry / usage |
|---|---|---|
| `impact_synth` | Synthesis (e.g. MR→CT) | `pip install impact-synth-konfai` → `impact-synth-konfai synthesize MR -i input.nii.gz -o ./Output/` |
| `impact_seg` | Segmentation | `pip install impact-seg-konfai` → `impact-seg-konfai segment body -i image.nii.gz -o ./Output/` |
| `impact_reg` | Registration | IMPACT-Reg orchestrator (`KONFAI_IMPACTREG_REPO`) |
| `mrsegmentator` | MR segmentation | thin wrapper over `konfai-apps` |
| `totalsegmentator` | CT segmentation | thin wrapper over `konfai-apps` |

The thin wrappers (`impact_synth`, `impact_seg`, `mrsegmentator`, `totalsegmentator`) just
call `konfai-apps` with a fixed app id, giving a task-named command
(`impact-synth-konfai synthesize ...`) instead of `konfai-apps infer <id> ...`.

## Where this fits

- Author + train a workflow → raw `konfai` CLI (this skill's main path).
- Ship a stable workflow for others to run inference → package as a `konfai-apps` app.
- Integrate into an external tool (e.g. 3D Slicer) or a lightweight client → `konfai-apps-server`.

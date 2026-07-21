# KonfAI Apps

`konfai-apps` is the standalone package for **packaged KonfAI workflows**.

It is the layer you use when a low-level KonfAI workflow is already stable and
you want a cleaner interface for:

- local inference and evaluation
- reusable app packaging
- remote execution through a FastAPI server
- integration in external tools such as 3D Slicer or lightweight clients

In the KonfAI repository, the split is intentional:

- `konfai` = the low-level workflow engine
- `konfai-apps` = the packaged app runtime
- [`apps/`](../apps) = a collection of concrete published apps

## What This Package Provides

`konfai-apps` ships three interfaces:

- `konfai-apps` for local or remote app execution
- `konfai-apps-server` for hosting apps behind an HTTP API
- the Python API under `konfai_apps`

This package depends on the core `konfai` framework, but it is versioned and
tested as a separate runtime surface.

## Installation

From PyPI:

```bash
python -m pip install konfai-apps
```

From a local checkout of this monorepo:

```bash
git clone https://github.com/fideus-labs/KonfAI.git
cd KonfAI
python -m pip install -e .
python -m pip install -e ./konfai-apps
```

Check the entrypoints:

```bash
konfai-apps --help
konfai-apps-server --help
```

## App Identifiers

The `app` argument accepted by `konfai-apps` supports three main forms:

- Hugging Face app: `repo_id:app_name`
- local app directory: `/path/to/my_app`
- remote app execution: same app identifier, plus `--host ...`

Examples:

```bash
konfai-apps infer VBoussot/ImpactSynth:CBCT ...
konfai-apps infer /data/apps/MyApp ...
konfai-apps infer VBoussot/ImpactSynth:CBCT --host 127.0.0.1 --port 8000 ...
```

## Quick Start

### Local inference

```bash
konfai-apps infer VBoussot/ImpactSynth:CBCT \
  -i input.mha \
  -o ./Output \
  --gpu 0
```

### Evaluation

```bash
konfai-apps eval VBoussot/ImpactSynth:CBCT \
  -i input.mha \
  --gt gt.mha \
  --mask mask.mha \
  -o ./Evaluation \
  --gpu 0
```

### Full pipeline

```bash
konfai-apps pipeline VBoussot/ImpactSynth:CBCT \
  -i input.mha \
  --gt gt.mha \
  --mask mask.mha \
  -o ./Output \
  -uncertainty \
  --gpu 0
```

### Fine-tuning

```bash
konfai-apps fine-tune my_app MyCustomApp \
  -d ./Dataset \
  -o ./FineTune \
  --epochs 10 \
  --gpu 0
```

## Main Commands

The CLI is organized around the packaged workflows available inside an app:

- `infer`
- `eval`
- `uncertainty`
- `pipeline`
- `fine-tune`

Common execution options:

- `--gpu 0` or `--gpu 0 1`
- `--cpu N`
- `--download`
- `--force_update`
- `--quiet`

Workflow-specific config overrides:

- `--prediction-file`
- `--evaluation-file`
- `--uncertainty-file`

## Python API

The same runtime is available from Python:

```python
from pathlib import Path

from konfai_apps import KonfAIApp

app = KonfAIApp("VBoussot/ImpactSynth:CBCT", download=False, force_update=False)
app.infer(
    inputs=[[Path("input.mha")]],
    output=Path("./Output"),
    gpu=[0],
    cpu=None,
    quiet=False,
    tmp_dir=None,
    ensemble=0,
    ensemble_models=[],
    tta=0,
    mc=0,
    uncertainty=False,
    prediction_file="Prediction.yml",
)
```

For remote execution:

```python
from konfai import RemoteServer
from konfai_apps import KonfAIAppClient

client = KonfAIAppClient(
    "VBoussot/ImpactSynth:CBCT",
    RemoteServer("127.0.0.1", 8000, "secret"),
)
```

## Remote Server

`konfai-apps-server` exposes packaged apps through a FastAPI service.

Minimal example:

```bash
export KONFAI_API_TOKEN="secret"
konfai-apps-server \
  --host 0.0.0.0 \
  --port 8000 \
  --apps ./konfai-apps/tests/assets/apps.json
```

Once the server is running, the client switches to remote mode as soon as
`--host` is provided:

```bash
konfai-apps infer VBoussot/ImpactSynth:CBCT \
  -i input.mha \
  -o ./Output \
  --host 127.0.0.1 \
  --port 8000 \
  --token secret \
  --cpu 1
```

The remote flow is:

1. upload inputs and parameters
2. schedule the job on the server
3. stream logs back to the client
4. download the packaged result archive
5. extract the final outputs locally

## What a KonfAI App Contains

A typical app directory contains:

```text
my_app/
├── app.json
├── Prediction.yml
├── Evaluation.yml            # optional
├── Uncertainty.yml           # optional
├── checkpoint.pt             # one or more checkpoints
└── custom Python modules     # optional
```

`app.json` stores metadata such as:

- display name
- description
- short description
- TTA capability
- MC-dropout capability
- optional UI-oriented metadata

The exact app authoring model is documented in the main repository under
[`apps/`](../apps) and in the Sphinx docs.

## Repository Layout

Inside this monorepo:

- [`konfai-apps/konfai_apps`](./konfai_apps) contains the package runtime
- [`konfai-apps/tests`](./tests) contains package-specific tests
- [`apps/`](../apps) contains published packaged workflows built on top of this runtime

This is why the folder is named `konfai-apps` and not just `apps`:
the package is the **runtime**, while `apps/` is the **catalog of concrete apps**.

## Development

Install both packages in editable mode:

```bash
python -m pip install -e .
python -m pip install -e ./konfai-apps
```

Run the package test suite:

```bash
pytest -q konfai-apps/tests
```

The tests are split into:

- `konfai-apps/tests/unit`
- `konfai-apps/tests/integration`

GitHub Actions also separates the package from the core repository workflow:

- `.github/workflows/konfai_ci.yml`
- `.github/workflows/konfai_apps_ci.yml`

## Related Documentation

Useful entry points in the main repository:

- [`../README.md`](../README.md)
- [`../apps/README.md`](../apps/README.md)
- [`../docs/source/usage/apps.md`](../docs/source/usage/apps.md)
- [`../docs/source/usage/remote-server.md`](../docs/source/usage/remote-server.md)
- [`../docs/source/reference/cli.md`](../docs/source/reference/cli.md)

## Current Scope

`konfai-apps` is intentionally focused on packaged workflow execution.

It is not meant to replace the low-level `konfai` workflow authoring interface

Its job is narrower and more operational:
turn a mature KonfAI workflow into a reproducible, distributable application
surface.

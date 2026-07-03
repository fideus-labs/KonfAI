# CLI reference

This page lists the main command-line entrypoints used in the repository. Use
it as the quick map of "which command should I run?".

KonfAI uses four main command-line entrypoints:

- `konfai`
- `konfai-apps`
- `konfai-apps-server`
- `konfai-cluster`

## `konfai`

Low-level workflow runner for training, prediction, and evaluation.

Use `konfai` when you are still designing a workflow directly from YAML.

### Commands

| Command | Purpose |
| --- | --- |
| `TRAIN` | Train a model from scratch. |
| `RESUME` | Resume training from a checkpoint. |
| `PREDICTION` | Run inference using one or more checkpoints. |
| `EVALUATION` | Compute metrics on saved outputs. |

### Common options

| Option | Meaning |
| --- | --- |
| `-c`, `--config` | YAML file to use. |
| `-y`, `--overwrite` | Overwrite existing outputs without prompting. |
| `--gpu` | One or more GPU ids. |
| `--cpu` | Number of CPU workers when not using GPUs. |
| `-q`, `--quiet` | Reduce console output. |
| `-tb`, `--tensorboard` | Launch TensorBoard. |

### Default config file per command

If `-c/--config` is omitted, each command falls back to a **fixed filename in the
current directory**:

| Command | Default config | Root key |
| --- | --- | --- |
| `TRAIN` / `RESUME` | `./Config.yml` | `Trainer:` |
| `PREDICTION` | `./Prediction.yml` | `Predictor:` |
| `EVALUATION` | `./Evaluation.yml` | `Evaluator:` |

```{note}
The `--config` help text mentions `Train.yml`, but the real TRAIN default is
**`./Config.yml`**. Also remember that **reading a config rewrites it on disk** —
after a run your YAML will contain the resolved defaults. See
{doc}`../concepts/configuration`.
```

### Command-specific options

`TRAIN`

- `--checkpoints-dir` / `--checkpoints_dir` (default `./Checkpoints/`)
- `--statistics-dir` / `--statistics_dir` (default `./Statistics/`)

`RESUME`

- `--model` — checkpoint path to resume from (**required**)
- `--lr` — override the learning rate on resume (omit to keep the checkpoint LR)
- `-checkpoints-dir` / `-statistics-dir` — note the **single leading dash** here
  (an inconsistency with TRAIN's `--` forms; both parse, but invoke them exactly
  as written)

`PREDICTION`

- `--models` — one or more checkpoint paths (**required**); multiple = ensemble
- `--predictions-dir` / `--predictions_dir` (default `./Predictions/`)

`EVALUATION`

- `--evaluations-dir` / `--evaluations_dir` (default `./Evaluations/`)

```{note}
**Device selection quirks.** The CLI default is **CPU** (`--gpu` defaults to an
empty list); pass `--gpu 0` to use a GPU. Valid `--gpu` ids are frozen at startup
from the visible CUDA devices, so an id that isn't visible is rejected by argparse.
`--cpu` must be `> 0`. `--version` works on the root parser (`konfai --version`)
but not on a subcommand.
```

## `konfai-apps`

Higher-level packaged workflow runner.

Use `konfai-apps` when a workflow is already packaged as a KonfAI App and you
want a simpler interface than the low-level YAML CLI.

This command is provided by the standalone `konfai-apps` package.

### Commands

| Command | Purpose |
| --- | --- |
| `infer` | Run inference for an app. |
| `eval` | Run evaluation for an app. |
| `uncertainty` | Run uncertainty estimation for an app. |
| `pipeline` | Chain inference, evaluation, and optional uncertainty. |
| `fine-tune` | Fine-tune an app on a dataset. |

### Shared options

| Option | Meaning |
| --- | --- |
| `app` | App identifier or repository path. |
| `--host`, `--port`, `--token` | Switch from local app execution to remote server mode. |
| `-i`, `--inputs` | Input paths, grouped by repeated flag occurrences. |
| `-o`, `--output` | Output directory. |
| `--gpu` / `--cpu` | Device selection. |
| `-q`, `--quiet` | Reduce console output. |
| `--download` | Pre-download the full app locally. |
| `--force_update` | Force an updated app download. |

### Important command-specific options

`infer`

- `--ensemble`
- `--ensemble-models`
- `--tta`
- `--mc`
- `-uncertainty`
- `--prediction-file` (alias: `--prediction_file`)

`eval`

- `--gt`
- `--mask`
- `--evaluation-file` (alias: `--evaluation_file`)

`uncertainty`

- `--uncertainty-file` (alias: `--uncertainty_file`)

`pipeline`

- combines the options from `infer`, `eval`, and `uncertainty`

`fine-tune`

- positional `name`
- `-d`, `--dataset`
- `--models` — checkpoint name(s) to fine-tune, e.g. `CV_0 CV_1` (default: first available)
- `--epochs`
- `--it-validation`
- `--config` (aliases: `--config-file`, `--config_file`)

## `konfai-apps-server`

FastAPI server exposing packaged apps remotely.

This command is the server-side counterpart of `konfai-apps --host ...`.
It is also provided by the standalone `konfai-apps` package.

Important options:

| Option | Meaning |
| --- | --- |
| `--host` | Bind address. |
| `--port` | Bind port. |
| `--auth` | `off` or `bearer`. |
| `--token-env` | Environment variable holding the token. |
| `--token` | Development-only token override. |
| `--apps` | JSON file listing the available apps. |
| `--download` | Pre-download configured apps at startup. |
| `--check` | Validate configured apps without downloading them. |

## `konfai-cluster`

Cluster-oriented wrapper around the low-level `konfai` commands.

It adds job-submission options such as:

- `--name`
- `--num-nodes`
- `--memory`
- `--time-limit`
- `--resubmit`

The cluster command depends on the optional `cluster` extra.

## ONNX export is not a subcommand

`konfai/export.py` can export a trained model to ONNX (+ a manifest) for the
`konfai-rs` portable-inference path, but it is a **Python-API-only** feature —
there is no `konfai export` subcommand. See {doc}`python-api`.

## See also

- {doc}`components/index` — the components a config can reference
- {doc}`environment`
- {doc}`app-server-api` — the `konfai-apps-server` HTTP contract
- {doc}`python-api` — the `konfai_apps` Python API
- {doc}`../usage/apps`

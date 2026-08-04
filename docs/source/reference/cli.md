# CLI reference

This page lists the main command-line entrypoints used in the repository. Use
it as the quick map of "which command should I run?".

KonfAI uses four main command-line entrypoints:

- `konfai`
- `konfai-apps`
- `konfai-apps-server`
- `konfai-cluster`

## `konfai`

Low-level workflow runner for training, prediction, evaluation, and transformation.

Use `konfai` when you are still designing a workflow directly from YAML.

### Commands

| Command | Purpose |
| --- | --- |
| `TRAIN` | Train a model from scratch. |
| `RESUME` | Resume training from a checkpoint. |
| `PREDICTION` | Run inference using one or more checkpoints. |
| `EVALUATION` | Compute metrics on saved outputs. |
| `TRANSFORM` | Apply a transform chain to a dataset — no model. |

### Common options

These apply to `TRAIN`, `RESUME`, `PREDICTION`, and `EVALUATION`. `TRANSFORM`
builds its own parser: it takes `-c`, `-y`, `--gpu`, `--cpu` and `-q` with the
meanings noted below, and has **no** `-tb`.

| Option | Meaning |
| --- | --- |
| `-c`, `--config` | YAML file to use. |
| `-y`, `--overwrite` | Overwrite existing outputs without prompting. Under `TRANSFORM`: recompute cases whose output exists — without it such a case is skipped, and nothing prompts. |
| `--gpu` | One or more GPU ids. |
| `--cpu` | Number of CPU workers when not using GPUs. Under `TRANSFORM`: shard the cases over N worker processes (default 1). |
| `-q`, `--quiet` | Reduce console output. |
| `-tb`, `--tensorboard` | Launch TensorBoard. Not accepted by `TRANSFORM`. |

### Default config file per command

If `-c/--config` is omitted, each command falls back to a **fixed filename in the
current directory**:

| Command | Default config | Root key |
| --- | --- | --- |
| `TRAIN` / `RESUME` | `./Config.yml` | `Trainer:` |
| `PREDICTION` | `./Prediction.yml` | `Predictor:` |
| `EVALUATION` | `./Evaluation.yml` | `Evaluator:` |
| `TRANSFORM` | `./Transform.yml` | `Transformer:` |

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
- `-checkpoints-dir` / `-statistics-dir` — note the **single leading dash** here, an
  inconsistency with TRAIN's `--` forms. Only the single-dash spelling parses on
  RESUME (`--checkpoints-dir` gives *unrecognized arguments*), so invoke them exactly
  as written; the underscore variants (`-checkpoints_dir`) also work.

`PREDICTION`

- `--models` — one or more checkpoint paths (**required**); multiple = ensemble
- `--predictions-dir` / `--predictions_dir` (default `./Predictions/`)

`EVALUATION`

- `--evaluations-dir` / `--evaluations_dir` (default `./Evaluations/`)

`TRANSFORM`

- `--plan` — print the per-case streaming plan and exit. The plan probes each
  destination with a real region-write open (created, then removed), so its
  verdict is the run's own — and even plan mode touches the output directories.
- `--transforms-dir` / `--transforms_dir` (default `./Transforms/`) — run logs
  and the plan; the outputs go where each `Write:` says.
- `--gpu` exists for one reason: a `KonfAIInference` stage runs a nested
  inference that does use the device. Plain read transforms run on CPU either
  way. There is no `-tb`: the workflow emits no scalars.
- `--plan` short-circuits before the distributed wrapper, so it runs in one
  process and spawns no ranks. `--cpu` is still read: it is the rank count the
  plan divides the `memory_budget` by, so `--plan --cpu 4` reports the per-rank
  budget a four-process run would actually get.

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
| `pipeline` | Chain inference, evaluation, and optional uncertainty. **`--gt` is required** — it always evaluates. |
| `fine-tune` | Fine-tune an app on a dataset. |
| `bundle` | Assemble an app bundle (HF layout), optionally with a portable ONNX model. |
| `download` | Pre-fetch an app's files from Hugging Face into the local cache (offline use). |

```{note}
`bundle` and `download` take neither `app` nor the shared options below — they have
their own signatures. `bundle NAME` **requires** `--out`, `--app-json`, `--config`
and `--checkpoint` (and its `--patch-size` sizes the *ONNX export*, not inference).
`download APP [FILES…]` takes `--no-force-update`. Run `--help` on either for the
full list.
```

### Shared options

| Option | Meaning |
| --- | --- |
| `app` | App identifier or repository path. |
| `--host`, `--port`, `--token` | Switch from local app execution to remote server mode. |
| `-i`, `--inputs` | Input paths, grouped by repeated flag occurrences. |
| `-o`, `--output` | Output directory. |
| `--gpu` / `--cpu` | Device selection — **mutually exclusive**, as on the `konfai` CLI. |
| `--tmp-dir` (alias: `--tmp_dir`) | Where intermediate artifacts are written. On `infer`, `eval`, `uncertainty` and `pipeline` only. |
| `-q`, `--quiet` | Reduce console output. |
| `--download` | Pre-download the full app locally. |
| `--force_update` | Force an updated app download. |

### Important command-specific options

`infer`

- `--ensemble` / `--ensemble-models` — **mutually exclusive**
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
- `--gt` is **required** here (unlike the per-app `pipeline` shims, where it is optional)

`fine-tune`

- positional `name`
- `-d`, `--dataset`
- `--models` — checkpoint name(s) to fine-tune, e.g. `CV_0 CV_1` (default: first available)
- `--epochs`
- `--it-validation`
- `--lr` — override the learning rate; omitted, the checkpoint's is resumed
- `--set` — the same config overrides as `infer` (see below)
- `--config` (aliases: `--config-file`, `--config_file`)

### Tuning a preset (`--set`, `--patch-size`, `--batch-size`)

`infer` and `pipeline` accept all three overrides below; `fine-tune` accepts
`--set` (plus its own `--lr`, `--epochs` and `--it-validation`). They let you adapt
a published App without editing its bundled config:

| Option | Meaning |
| --- | --- |
| `--set NAME=VALUE` | Override any config value (repeatable). A bare `NAME` tunes a model parameter (`--set iterations=300`); a dotted `NAME` is a full path from the config root (`--set Predictor.Dataset.batch_size=2`). The value is parsed as YAML (int / float / bool / list / string). |
| `--patch-size` | Override the inference `Patch.patch_size` (one value = an isotropic cube; else per-axis). Overrides the App's auto `vram_plan` choice. |
| `--batch-size` | Override the inference batch size. |

These are the same knobs SlicerKonfAI drives through its ⚙ **Advanced** dialog.

```{note}
These overrides are honoured in **remote** mode too (`--host …`). Each operation
declares which tunables the server must carry — `infer` and `pipeline` forward
`patch_size`, `batch_size` and `config_overrides`; `fine-tune` forwards
`config_overrides` — and the client **refuses the submission** if the server does
not echo them back in `accepted_options`. A server too old to honour a tunable
fails loudly rather than ignoring it silently.
```

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

## Next steps

- {doc}`components/index` — the component names those YAML configs can reference
- {doc}`environment` — the variables these wrappers read and set
- {doc}`../usage/apps` — the guided workflow behind `konfai-apps`

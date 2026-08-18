# CLI reference

This page lists the main command-line entrypoints used in the repository. Use
it as the quick map of "which command should I run?".

KonfAI ships six command-line entrypoints, across four packages:

| Command | Package | Purpose |
| --- | --- | --- |
| `konfai` | `konfai` | run a YAML workflow: train, predict, evaluate, transform |
| `konfai-cluster` | `konfai` (`cluster` extra) | submit those workflows to SLURM |
| `konfai-apps` | `konfai-apps` | run a packaged App |
| `konfai-apps-server` | `konfai-apps` | serve Apps over HTTP |
| `konfai-mcp` | `konfai-mcp` | expose KonfAI to an LLM agent |
| `konfai-studio` | `konfai-studio` | the web UI over `konfai-mcp` |

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
| `TRANSFORM` | Prepare a dataset: apply a transform chain and write the result. |

### Common options

These apply to `TRAIN`, `RESUME`, `PREDICTION`, and `EVALUATION`. `TRANSFORM`
builds its own parser: it takes `-c`, `-y`, `--gpu`, `--cpu` and `-q` with the
meanings noted below, and has **no** `-tb`.

| Option | Meaning |
| --- | --- |
| `-c`, `--config` | YAML file to use. |
| `-y`, `--overwrite` | Overwrite existing outputs without prompting. Under `TRANSFORM`: recompute cases whose output exists, without it such a case is skipped, and nothing prompts. |
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

Reading a config rewrites it on disk: after a run your YAML holds the resolved
defaults. See {doc}`../concepts/configuration`.

### Command-specific options

`TRAIN`

- `--checkpoints-dir` / `--checkpoints_dir` (default `./Checkpoints/`)
- `--statistics-dir` / `--statistics_dir` (default `./Statistics/`)

`RESUME`

- `--model`: checkpoint path to resume from (**required**)
- `--lr`: override the learning rate on resume (omit to keep the checkpoint LR)
- `-checkpoints-dir` / `-statistics-dir`: note the **single leading dash** here, an
  inconsistency with TRAIN's `--` forms. Only the single-dash spelling parses on
  RESUME (`--checkpoints-dir` gives *unrecognized arguments*), so invoke them exactly
  as written; the underscore variants (`-checkpoints_dir`) also work.

`PREDICTION`

- `--models`: one or more checkpoint paths (**required**); multiple = ensemble
- `--predictions-dir` / `--predictions_dir` (default `./Predictions/`)

`EVALUATION`

- `--evaluations-dir` / `--evaluations_dir` (default `./Evaluations/`)

`TRANSFORM`

- `--plan`: print the per-case streaming plan and exit. The plan probes each
  destination with a real region-write open (created, then removed), so its
  verdict is the run's own, and even plan mode touches the output directories.
  It also reads the config the way a run does, which resolves the defaults back
  into `Transform.yml`; copy the file first to keep the text you wrote.
- `--transforms-dir` / `--transforms_dir` (default `./Transforms/`): run logs;
  the outputs go where each `Write:` says. `--plan` prints and writes nothing
  there.
- `--gpu` exists for one reason: a `KonfAIInference` stage runs a nested
  inference that does use the device. Plain read transforms run on CPU either
  way. There is no `-tb`: the workflow emits no scalars.
- `--plan` short-circuits before the distributed wrapper, so it runs in one
  process and spawns no ranks. `--cpu` and `--gpu` are still read: the plan is
  sized for the run's world size (one rank per GPU, else `--cpu` ranks), and an
  `auto` budget is the node's memory split across that many ranks, so
  `--plan --cpu 4` reports the per-rank budget a four-process run would actually
  get. An explicit `memory_budget` is already per rank and is not divided. The
  plan is the requested output, so `-q` does not silence it. `konfai-cluster`
  refuses `--plan`: a plan submits nothing.

The default is **CPU**: `--gpu` defaults to an empty list, so pass `--gpu 0` to
use a card. An id that is not among the visible CUDA devices is a usage error
(exit code 2), checked once the command is dispatched so that `--help` never
loads torch. `--cpu` must be greater than 0.
`--version` works on the root parser, `konfai --version`, not on a subcommand.

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
| `pipeline` | Chain inference, evaluation, and optional uncertainty. **`--gt` is required**: it always evaluates. |
| `fine-tune` | Fine-tune an app on a dataset. |
| `bundle` | Assemble an app bundle (HF layout), optionally with a portable ONNX model. |
| `download` | Pre-fetch an app's files from Hugging Face into the local cache (offline use). |

`bundle` and `download` have their own signatures: neither takes `app` nor the
shared options below. `bundle NAME` requires `--out`, `--app-json`, `--config` and
`--checkpoint`, and its `--patch-size` sizes the ONNX export rather than
inference. `download APP [FILES…]` takes `--no-force-update`. Run `--help` on either for
the full signature.

### Shared options

| Option | Meaning |
| --- | --- |
| `app` | App identifier or repository path. |
| `--host`, `--port`, `--token` | Switch from local app execution to remote server mode. |
| `-i`, `--inputs` | Input paths, grouped by repeated flag occurrences. |
| `-o`, `--output` | Output directory. |
| `--gpu` / `--cpu` | Device selection: **mutually exclusive**, as on the `konfai` CLI. |
| `--tmp-dir` (alias: `--tmp_dir`) | Where intermediate artifacts are written. On `infer`, `eval`, `uncertainty` and `pipeline` only. |
| `-q`, `--quiet` | Reduce console output. |
| `--download` | Pre-download the full app locally. |
| `--force_update` | Force an updated app download. |

### Important command-specific options

`infer`

- `--ensemble` / `--ensemble-models`: **mutually exclusive**
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
- `--models`: checkpoint name(s) to fine-tune, e.g. `CV_0 CV_1` (default: first available)
- `--epochs`
- `--it-validation`
- `--lr`: override the learning rate; omitted, the checkpoint's is resumed
- `--batch-size`: override the training batch size (`Trainer.Dataset.batch_size`)
- `--set`: the same config overrides as `infer` (see below)
- `--config` (aliases: `--config-file`, `--config_file`)

### Tuning a preset (`--set`, `--patch-size`, `--batch-size`)

`infer` and `pipeline` accept all three overrides below; `fine-tune` accepts
`--set` and `--batch-size` (plus its own `--lr`, `--epochs` and `--it-validation`,
with `--batch-size` writing the training `Trainer.Dataset.batch_size`). They let
you adapt a published App without editing its bundled config:

| Option | Meaning |
| --- | --- |
| `--set NAME=VALUE` | Override any config value (repeatable). A bare `NAME` tunes a model parameter (`--set iterations=300`); a dotted `NAME` is a full path from the config root (`--set Predictor.Dataset.batch_size=2`). The value is parsed as YAML (int / float / bool / list / string). |
| `--patch-size` | Override the inference `Patch.patch_size` (one value = an isotropic cube; else per-axis). Overrides the App's auto `vram_plan` choice. |
| `--batch-size` | Override the inference batch size. |

These are the same knobs SlicerKonfAI drives through its ⚙ **Advanced** dialog.

These overrides work in remote mode too (`--host …`). Each operation declares
which tunables the server must carry: `infer` and `pipeline` forward `patch_size`,
`batch_size` and `config_overrides`, `fine-tune` forwards `batch_size` and
`config_overrides`. The
client refuses the submission when the server does not echo them back in
`accepted_options`, so a server too old to honour a tunable fails loudly instead
of ignoring it.

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

Cluster-oriented wrapper around the low-level `konfai` commands: it takes the
same workflow arguments and submits them to SLURM through `submitit`. Depends on
the optional `cluster` extra.

| Option | Default | Meaning |
| --- | --- | --- |
| `--name` | **required** | SLURM job name. |
| `--num-nodes` | `1` | Nodes to request. |
| `--memory` | `16` | Memory per node, in GB. |
| `--time-limit` | `1440` | Wall-clock limit, in minutes. |
| `--resubmit` | off | Accepted, but **not implemented**: the run warns and does not requeue. |

Otherwise `konfai-cluster` takes the same subcommands and arguments as `konfai`.
**The cluster options come before the subcommand**: they sit on the top-level
parser, so putting them after it fails with `the following arguments are
required: --name`:

```bash
konfai-cluster --name my_job --num-nodes 2 TRAIN -y --config Config.yml
```

## `konfai-mcp`

Runs the MCP server that exposes KonfAI to an LLM agent. Every option also reads
an environment variable, so a client that can only set `env` can configure the
server without arguments: see {doc}`environment`.

| Option | Meaning |
| --- | --- |
| `--transport` | `stdio` (default), `sse`, or `streamable-http`. |
| `--session` | Default session name for this server process. |
| `--workspace-root` | Directory holding MCP sessions and datasets. |
| `--log-tail-lines` | Default maximum lines returned by log-tail helpers. |
| `--host` / `--port` | Bind address and port, for the SSE/HTTP transports. |
| `--path` | HTTP path prefix, for the SSE/HTTP transports. |
| `--log-level` | FastMCP/Uvicorn log level, where the transport supports it. |
| `--bearer-token` | Token required by the SSE/HTTP transports. |

## `konfai-studio`

Launches the Studio web UI and its BFF. Binds loopback by default; anything else
requires authentication, because Studio drives arbitrary host compute.

| Option | Meaning |
| --- | --- |
| `--host` / `--port` | Bind address (default `127.0.0.1`) and port (default `8730`). |
| `--proxy-headers` | Trust `X-Forwarded-*`; set this behind nginx or Caddy. |
| `--forwarded-allow-ips` | Proxy IPs allowed to set those headers (default `127.0.0.1`). |
| `--ssl-certfile` / `--ssl-keyfile` | Serve HTTPS directly; the two go together. |
| `--i-know-this-is-insecure` | Bind a public address with no `KONFAI_STUDIO_TOKEN`. |

```{warning}
Binding a non-loopback address without `KONFAI_STUDIO_TOKEN` is refused, not
warned about: an unauthenticated Studio is a shell on the host. Set a token and
serve over TLS: see `studio/docs/REMOTE.md`.
```

## ONNX export is not a subcommand

`konfai/export.py` can export a trained model to ONNX (+ a manifest) for the
`konfai-rs` portable-inference path, but it is a **Python-API-only** feature: there is no `konfai export` subcommand. See {doc}`python-api`.

## Next steps

- {doc}`components/index`: the component names those YAML configs can reference
- {doc}`environment`: the variables these wrappers read and set
- {doc}`../usage/apps`: the guided workflow behind `konfai-apps`

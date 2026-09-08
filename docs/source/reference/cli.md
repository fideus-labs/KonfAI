# CLI reference

This page lists the main command-line entrypoints used in the repository. Use
it as the quick map of "which command should I run?".

KonfAI ships six command-line entrypoints, across four packages:

| Command | Package | Purpose |
| --- | --- | --- |
| `konfai` | `konfai` | run a YAML workflow: train, predict, evaluate, transform |
| `konfai-cluster` | `konfai` (submission needs the `cluster` extra) | submit those workflows to SLURM |
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
| `list` | Print the components a YAML config can reference (see below). |

### Common options

These apply to `TRAIN`, `RESUME`, `PREDICTION`, and `EVALUATION`. `TRANSFORM`
builds its own parser: it takes `-c`, `-y`, `--gpu`, `--cpu` and `-q` with the
meanings noted below, and has **no** `-tb`.

| Option | Meaning |
| --- | --- |
| `-c`, `--config` | YAML file to use. |
| `-y`, `--overwrite` | Overwrite existing outputs without prompting. Under `TRANSFORM`: recompute cases whose output exists, without it such a case is skipped, and nothing prompts. |
| `--gpu` | One or more GPU ids. |
| `--cpu` | Number of CPU worker processes when no `--gpu` is given; the run stays on CPU unless `--gpu` is passed. Under `TRANSFORM`: shard the cases over N worker processes (default 1). |
| `-q`, `--quiet` | Reduce console output. |
| `-tb`, `--tensorboard` | Launch TensorBoard. Not accepted by `TRANSFORM`. |
| `--init` | Create the config file if missing, resolve every default into it, and exit without running. |

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
defaults. See {doc}`../config_guide/index`.

### Generating a config: `--init`

`konfai <COMMAND> --init` is how a config file is generated: it creates the
command's config file when missing (seeded with its root key), binds the
workflow once so every default resolves into the file, and exits without
running anything. `-c` picks the filename. A binding error after partial
resolution still leaves what resolved on disk, plus the error naming the key.

```bash
konfai TRAIN --init -c Config.yml
```

### `konfai list`

`konfai list {transforms,augmentations,criteria,reductions,models,blocks}`
prints one component family: the exact spelling a YAML config references, and
each component's one-line doc. `konfai list models` covers both the Python
catalog (`segmentation.UNet.UNet`) and the declarative catalog
(`default|UNet.yml`). `list` takes none of the run flags and loads no torch for
`--help`.

### Command-specific options

`TRAIN`

- `--checkpoints-dir` / `--checkpoints_dir` (default `./Checkpoints/`)
- `--statistics-dir` / `--statistics_dir` (default `./Statistics/`)

`RESUME`

- `--model`: checkpoint path to resume from (**required**)
- `--lr`: override the learning rate on resume (omit to keep the checkpoint LR)
- `--checkpoints-dir` / `--statistics-dir`: as for TRAIN

`PREDICTION`

- `--models`: one or more checkpoint paths (**required**); multiple = ensemble
- `--predictions-dir` / `--predictions_dir` (default `./Predictions/`)

`EVALUATION`

- `--evaluations-dir` / `--evaluations_dir` (default `./Evaluations/`)

`TRANSFORM`

- `--plan`: print the per-case streaming plan and exit. The plan probes each
  destination with a real region-write open, so its verdict is the run's own,
  then takes back what the probe created: the entry, and the store itself when
  it did not exist before. It also reads the config the way a run does, which
  resolves the defaults back into `Transform.yml`; copy the file first to keep
  the text you wrote.
- `--transforms-dir` / `--transforms_dir` (default `./Transforms/`): run logs;
  the outputs go where each `Write:` says. `--plan` prints and writes nothing
  there.
- `--gpu`: each rank runs its chain on its device, in taller slabs than on a
  CPU, and writes the same bytes; a `KonfAIInference` stage runs its nested
  inference there too. There is no `-tb`: the workflow emits no scalars.
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
loads torch. `--cpu` must be greater than 0. Unless `-q` is passed, every run
prints one startup line naming the resolved devices (`[KonfAI] Running on
cuda:0`, or `[KonfAI] Running on CPU (4 workers)`), so a silent CPU fallback on
a GPU machine is visible.
`--version` works on the root parser, `konfai --version`, not on a subcommand.

### How a run is launched

Every workflow runs under the distributed runtime in `konfai.utils.runtime`: it
sets `CUDA_VISIBLE_DEVICES` from `--gpu`, handles the overwrite and verbosity
flags, launches TensorBoard when requested, spawns one worker process per
device with `torch.multiprocessing.spawn` and initializes `torch.distributed`
on a free local TCP port. Even a local multi-process run uses that bootstrap,
for the workflows that need a process group: TRAIN, PREDICTION and EVALUATION.
TRANSFORM ranks are independent, each writing its own shard, so they are
spawned without one: no port, no rendezvous, and a rank that fails takes down
nothing but its own shard. The `KONFAI_*` variables the wrappers set on the way
are listed under [Environment variables](#environment-variables).

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
same workflow arguments and submits them to SLURM through `submitit`. The
command ships with the core package; submitting needs `submitit`, which the
`cluster` extra installs.

| Option | Default | Meaning |
| --- | --- | --- |
| `--name` | **required** | SLURM job name. |
| `--num-nodes` | `1` | Nodes to request. |
| `--memory` | `16` | Memory per node, in GB. |
| `--time-limit` | `1440` | Wall-clock limit, in minutes. |

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
server without arguments: see [Environment variables](#environment-variables).

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
`konfai-rs` portable-inference path, but it is a **Python-API-only** feature: there is no `konfai export` subcommand. See {doc}`../usage/python-api`.

## Environment variables

This section catalogues the environment variables KonfAI reads or sets: the
user-facing ones you may set yourself, and the `KONFAI_*` runtime variables the
CLI wrappers manage. Reach for it when a run behaves differently across shells
or machines, or when you are debugging the runtime wrappers themselves.

### User-facing variables

#### `CUDA_VISIBLE_DEVICES`

Controls which GPUs are visible to PyTorch and therefore to KonfAI.

KonfAI also rewrites this variable internally when you pass `--gpu`.

#### `KONFAI_API_TOKEN`

Bearer token used by:

- `konfai-apps` in remote mode
- `konfai-apps-server` in bearer-auth mode

#### `KONFAI_APPS_INSTALL_REQUIREMENTS`

Set to `0` to stop `konfai-apps` from pip-installing a resolved app's
`requirements.txt` (installed by default; core packages are never touched).
This is a **trust-model** switch: see the apps guide.

#### Streaming and write-path switches

Diagnostic kill-switches for the streamed prediction writer. Defaults are the
streamed behavior; set to `0`/a value only to compare against the whole-volume
path or to tune the gate.

| Variable | Effect |
| --- | --- |
| `KONFAI_STREAMED_WRITES` | `0` disables streamed writes entirely (whole-volume reference path). |
| `KONFAI_STREAM_WORTH_THRESHOLD` | Overrides the "worth streaming" accumulator-size threshold (fraction of the per-rank memory budget). Test harnesses set `0` to force the streamed machinery on toy volumes. |
| `KONFAI_ASYNC_WRITES` | Controls the background writer for disjoint-file sinks. |
| `KONFAI_INLINE_SINGLE_RANK` | Default on. `0` forces a single rank through the spawn path instead of running it in-process: useful when a host process must keep its own CUDA context. |

#### Hugging Face authentication

The repository and CI also rely on Hugging Face-hosted assets. KonfAI itself
uses `huggingface_hub`, so standard Hugging Face authentication variables may be
relevant in practice, but they are not KonfAI-specific.

### Runtime variables set by KonfAI

**These variables are normally set by the CLI wrappers and are not expected to
be managed manually in day-to-day usage.**

| Variable | Set by | Purpose |
| --- | --- | --- |
| `KONFAI_config_file` | workflow wrappers | Active YAML file path. |
| `KONFAI_ROOT` | workflow wrappers | Root config object: `Trainer`, `Predictor`, `Evaluator`, or `Transformer`. |
| `KONFAI_STATE` | workflow wrappers | Active workflow state: `TRAIN`, `RESUME`, `PREDICTION`, `EVALUATION`, or `TRANSFORM`. |
| `KONFAI_CHECKPOINTS_DIRECTORY` | training wrapper | Checkpoint output directory. |
| `KONFAI_STATISTICS_DIRECTORY` | training wrapper | Statistics output directory. |
| `KONFAI_PREDICTIONS_DIRECTORY` | prediction wrapper | Prediction output directory. |
| `KONFAI_EVALUATIONS_DIRECTORY` | evaluation wrapper | Evaluation output directory. |
| `KONFAI_TRANSFORMS_DIRECTORY` | transform wrapper | Transform run logs and plan directory. |
| `KONFAI_OVERWRITE` | distributed wrapper | Mirrors the `--overwrite` flag. |
| `KONFAI_TENSORBOARD_PORT` | distributed wrapper | Selected TensorBoard port. |
| `KONFAI_VERBOSE` | distributed wrapper | Mirrors the inverse of `--quiet`. |
| `KONFAI_CLUSTER` | cluster wrapper | Marks cluster execution. |

### Internal debug/config variables

The codebase also references internal variables such as:

- `KONFAI_CONFIG_MODE`, `KONFAI_CONFIG_PATH`: the config binder's mode machine
- `KONFAI_APPS_CONFIG`
- `KONFAI_DEBUG`: `1` re-attaches the framework traceback to a designed refusal (a
  `KonfAIError`), which otherwise prints its message and remedy alone
- `KONFAI_DEBUG_LAST_LAYER`: set it (empty) before a run and the network appends each module
  it enters, so after a crash it names the last layer reached
- `KONFAI_MASTER_PORT`: distributed rendezvous bookkeeping
- `KONFAI_LOCAL_RANKS`: how many ranks share one node's RAM, published by the
  launcher so a node-scoped `memory_budget` is divided before the spawn. It changes
  the cache-versus-stream decision, so it is not mere bookkeeping.
- `KONFAI_ATTR_KEY`, `KONFAI_DEPS`, `KONFAI_COMPONENT_BASES`, `KONFAI_VERSION`

These are part of KonfAI's internal execution model and are best treated as
implementation details unless you are actively extending the framework.

### konfai-mcp

Every `konfai-mcp` command-line option has a matching variable, so an MCP client
that can only set `env` configures the server without arguments. The option wins
when both are given.

| Variable | Equivalent option | Effect |
| --- | --- | --- |
| `KONFAI_MCP_WORKSPACES_ROOT` | `--workspace-root` | Directory holding MCP sessions and datasets. |
| `KONFAI_MCP_SESSION` | `--session` | Default session name for this server process. |
| `KONFAI_MCP_TRANSPORT` | `--transport` | `stdio` (default), `sse`, or `streamable-http`. |
| `KONFAI_MCP_HOST` / `KONFAI_MCP_PORT` | `--host` / `--port` | Bind address and port, for the SSE/HTTP transports. |
| `KONFAI_MCP_PATH` | `--path` | HTTP path prefix, for those same transports. |
| `KONFAI_MCP_BEARER_TOKEN` | `--bearer-token` | Token required by the SSE/HTTP transports. |
| `KONFAI_MCP_LOG_LEVEL` | `--log-level` | FastMCP/Uvicorn log level. |
| `KONFAI_MCP_LOG_TAIL_LINES` | `--log-tail-lines` | Default maximum lines returned by log-tail helpers. |

An invalid `KONFAI_MCP_TRANSPORT` is rejected at startup rather than passed
through. A few further `KONFAI_MCP_*` names configure internals with no option of
their own (the app catalog, the subprocess timeout, the validation root) and
are covered in {doc}`../usage/mcp`.

### KonfAI Studio

`konfai-studio` reads its own family. The first two are security-relevant: Studio
drives arbitrary host compute, so binding a non-loopback address without a token is
refused unless you override it. See `studio/docs/REMOTE.md`.

| Variable | Effect |
| --- | --- |
| `KONFAI_STUDIO_TOKEN` | Shared bearer token. **Unset means no authentication**, which is why a non-loopback bind is refused without it. |
| `KONFAI_STUDIO_INSECURE_COOKIE` | Drops the `Secure` flag on the session cookie, for plain-HTTP testing only. |
| `KONFAI_STUDIO_LLM` | Which backend drives the agent (for example `anthropic`, or an OpenAI-compatible server). |
| `KONFAI_STUDIO_LLM_API_KEY` | Key for that backend. |
| `KONFAI_STUDIO_LLM_BASE_URL` | Base URL of an OpenAI-compatible server (vLLM / Ollama / LM Studio). |
| `KONFAI_STUDIO_MODEL` | Main model id. |
| `KONFAI_STUDIO_SIDE_MODEL` | Model used for the cheaper side calls. |
| `KONFAI_STUDIO_MAX_TOKENS` | Per-response token ceiling. |
| `KONFAI_STUDIO_MAX_TURNS` | Agent-loop turn ceiling. |
| `KONFAI_STUDIO_TERMINAL` | Enables the in-app terminal. |

## Next steps

- {doc}`components/models`: the component names those YAML configs can reference
- {doc}`../usage/apps`: the guided workflow behind `konfai-apps`
- {doc}`../usage/python-api`: the same workflows as Python callables

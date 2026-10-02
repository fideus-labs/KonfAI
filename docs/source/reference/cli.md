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
meanings noted below, and has **no** `-tb`. `EVALUATION` has no `-tb` either:
it writes no TensorBoard events.

| Option | Meaning |
| --- | --- |
| `-c`, `--config` | YAML file to use. |
| `-y`, `--overwrite` | Overwrite existing outputs without prompting. Under `TRANSFORM`: recompute cases whose output exists, without it such a case is skipped, and nothing prompts. |
| `--gpu` | One or more GPU ids. |
| `--cpu` | Number of CPU worker processes when no `--gpu` is given; the run stays on CPU unless `--gpu` is passed. Under `TRANSFORM`: shard the cases over N worker processes (default 1). |
| `-q`, `--quiet` | Reduce console output. |
| `-tb`, `--tensorboard` | Launch TensorBoard on `127.0.0.1` (`KONFAI_TENSORBOARD_HOST` names another address). Needs the `tensorboard` extra, checked before the run starts. Not accepted by `EVALUATION` or `TRANSFORM`. |
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
An error that is not a KonfAI refusal (a bug in your own `Model.py`, say) prints
its traceback, as a run does.

```bash
konfai TRAIN --init -c Config.yml
```

### `konfai list`

`konfai list {transforms,augmentations,criteria,reductions,schedulers,models,blocks}`
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

- `--model`: checkpoint path to resume from (**required**, except with `--init`)
- `--lr`: override the learning rate on resume (omit to keep the checkpoint LR)
- `--checkpoints-dir` / `--statistics-dir`: as for TRAIN

`PREDICTION`

- `--models`: one or more checkpoint paths (**required**, except with `--init`); multiple = ensemble
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
a GPU machine is visible, and a finished `TRAIN`, `RESUME`, `PREDICTION` or
`EVALUATION` closes on the absolute path of what it wrote (`[KonfAI] outputs in
.../Predictions/<train_name>/Dataset`), so a `train_name` that differs between two
configs is visible too.
`--version` works on the root parser, `konfai --version`, not on a subcommand.

### How a run is launched

Every workflow runs under the distributed runtime in `konfai.utils.runtime`: it
sets `CUDA_VISIBLE_DEVICES` from `--gpu`, handles the overwrite and verbosity
flags, launches TensorBoard when requested, spawns one worker process per
device with `torch.multiprocessing.spawn` and initializes `torch.distributed`
on a free local TCP port. Even a local multi-process run uses that bootstrap,
for the workflows whose ranks talk to each other: TRAIN and EVALUATION.
PREDICTION and TRANSFORM ranks are independent, each writing its own cases, so
they are spawned without one: no port, no rendezvous, and a TRANSFORM rank that
fails takes down nothing but its own shard. Windows gets no process group, so
there TRAIN, RESUME and EVALUATION refuse more than one process. No workflow
lets several processes write a single-file output (`h5`): every rank would
write into the same file. The `KONFAI_*` variables the wrappers set on the way
are listed under [Environment variables](#environment-variables). Ctrl+C stops
a run with exit code 130, so `konfai TRAIN && konfai PREDICTION` stops there.
A designed refusal raised on a spawned rank reads as it would on a single rank:
its message alone, exit code 1 (the Python API raises it as itself). A rank
killed by `SIGKILL`, the signal the kernel's out-of-memory killer sends, ends
the run with a message naming the likely lack of RAM.

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
| `--tmp-dir` (alias: `--tmp_dir`) | Where intermediate artifacts are written. On `infer`, `eval`, `uncertainty` and `pipeline` only. The inputs are staged in its `Dataset`, so a directory holding a `Dataset` konfai-apps did not stage is refused. |
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
| `--patch-size` | Override the inference `Patch.patch_size` (one value = an isotropic cube; else per-axis). |
| `--batch-size` | Override the inference batch size. Without it the app's config decides: `batch_size: 0` measures the batch on the GPU. |

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
| `--check` | Validate configured apps without downloading them, then exit without starting the server (with `--download`, download them and serve). |

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
konfai-cluster --name my_job --num-nodes 2 TRAIN -y --gpu 0 1 --config Config.yml
```

`--gpu` is required: the job runs one rank per listed GPU on each node, and
`--cpu` does not apply. A job over several nodes names its rendezvous host with `scontrol`: a rank
that cannot run it (a container without the host's Slurm client) refuses at
startup rather than waiting on its own node until the rendezvous timeout.

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
| `--i-know-this-is-insecure` | Bind a non-loopback address with no bearer token. |

```{warning}
An SSE/HTTP server that binds a non-loopback address with no bearer token is
refused: it would let anyone on the network run jobs and read files on the host.
Bound to loopback with no token, it answers only requests addressed to
`127.0.0.1`, `localhost` or `::1`, which keeps a DNS-rebound web page out.
```

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
serve over TLS: see `konfai-studio/docs/REMOTE.md`.
```

## Exit codes

The workflow commands (`konfai`, `konfai-cluster`, `konfai-apps` and the app
CLIs built on it, such as `impact-seg-konfai`) share one convention, so a shell
script or a notebook can decide on the code alone:

| Code | Meaning |
| --- | --- |
| `0` | Success. |
| `1` | The command ran and failed: a refusal (a configuration, dataset or App it cannot run) or an unexpected error. |
| `2` | Usage error, reported by the argument parser before any work starts: an unknown or missing argument, an invalid value, or a combination of options the command refuses. |
| `130` | Interrupted by Ctrl+C. |

A command stopped by a signal ends with that signal: a shell reports 128 plus
the signal number (`143` for `SIGTERM`), Python's `subprocess` a negative code
(`-15`).

`impact-reg-konfai` does not follow the `130` code yet: Ctrl+C stops it as the
signal would, after a Python traceback, so a shell reports `130` but Python's
`subprocess` reports `-2`.

The servers (`konfai-apps-server`, `konfai-studio`) run until they are stopped:
Ctrl+C shuts them down and they exit with `0`. On its default `stdio`
transport, `konfai-mcp` exits with `130` on Ctrl+C, and on its own once its
client closes stdin. A server that refuses to start exits with `1` or `2`, as
above.

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

KonfAI also rewrites this variable internally when you pass `--gpu`. It reads
the entries as integer indices, the ones `--gpu` takes: a GPU named by UUID
(`GPU-...`, `MIG-...`) is refused with a message naming the variable.

#### `KONFAI_API_TOKEN`

Bearer token used by:

- `konfai-apps` in remote mode
- `konfai-apps-server` in bearer-auth mode

#### `KONFAI_APPS_INSTALL_REQUIREMENTS`

Set to `0` to stop `konfai-apps` from pip-installing a resolved app's
`requirements.txt` (installed by default; core packages are never touched).
This is a **trust-model** switch: see the apps guide.

#### `KONFAI_APPS_MAX_DATASET_BYTES`

Bound, in bytes, on the `dataset` zip that `konfai-apps-server` receives for a
fine-tune job (default 64 GiB): the archive, each member and the total extracted
bytes. Past it the server answers **413**. See the limits of the
[app server API](app-server-api.md).

#### `KONFAI_DECOMPRESSED_DIRECTORY`

Where runs keep the uncompressed twins of the compressed files they read by
region (`.nii.gz`, a compressed MetaImage). Default `~/.cache/konfai/decompressed`
(`$XDG_CACHE_HOME/konfai/decompressed` when that is set). See
[compressed files](components/storage-backends.md#compressed-files).

#### `KONFAI_TENSORBOARD_HOST`

The address TensorBoard binds under `-tb`. Unset or empty, it is `127.0.0.1`:
only this machine reaches it. TensorBoard has no authentication, so from another
machine forward the port over SSH (`ssh -N -L <port>:127.0.0.1:<port> user@server`)
and open `http://127.0.0.1:<port>/` on your own computer. `0.0.0.0` serves every
interface, and KonfAI prints the machine's network address.

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
  `KonfAIError`), which otherwise prints its message and remedy alone, and walks the graph
  eagerly to fill `KONFAI_DEBUG_LAST_LAYER`; `0`, `false` or unset leave it off
- `KONFAI_DEBUG_LAST_LAYER`: under `KONFAI_DEBUG`, the network sets it to each module it enters,
  as `name:memory:device`, so after a crash it names the last layer reached
- `KONFAI_MASTER_PORT`: distributed rendezvous bookkeeping
- `KONFAI_DECOMPRESSED_RUN`: the run's own directory under `KONFAI_DECOMPRESSED_DIRECTORY`,
  shared by every process of the run (its ranks and their loader workers inherit it) and removed
  when the run ends
- `KONFAI_LOCAL_RANKS`: how many ranks share one node's RAM, published by the
  launcher so a node-scoped `memory_budget` is divided before the spawn. It changes
  the cache-versus-stream decision, so it is not mere bookkeeping.

These are part of KonfAI's internal execution model and are best treated as
implementation details unless you are actively extending the framework.

### konfai-mcp

Every `konfai-mcp` command-line option but `--i-know-this-is-insecure` has a
matching variable, so an MCP client that can only set `env` configures the server
without arguments. The option wins when both are given.

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
through. A few further `KONFAI_MCP_*` names have no option of their own:

| Variable | Effect |
| --- | --- |
| `KONFAI_MCP_APP_CATALOG` | A JSON file of app sources layered over the shipped catalogue, see {doc}`../usage/mcp`. |
| `KONFAI_MCP_SUBPROCESS_TIMEOUT` | Seconds a subprocess the server runs outside a job (validation, smoke test, plan) may take before it is stopped. Default `1800`; `0` waits without bound. |
| `KONFAI_MCP_VALIDATE_ROOT` | Scratch root of `konfai_mcp.runner.validate_workflow_api` when it is called without one. The server never reads it: each validation builds in a fresh temporary directory. |

### KonfAI Studio

`konfai-studio` reads its own family. The first two are security-relevant: Studio
drives arbitrary host compute, so binding a non-loopback address without a token is
refused unless you override it. See `konfai-studio/docs/REMOTE.md`.

| Variable | Effect |
| --- | --- |
| `KONFAI_STUDIO_TOKEN` | Shared bearer token. **Unset means no authentication**, which is why a non-loopback bind is refused without it; a loopback bind then answers only to `127.0.0.1`, `localhost` and `::1`. |
| `KONFAI_STUDIO_INSECURE_COOKIE` | Drops the `Secure` flag on the session cookie, for plain-HTTP testing only. |
| `KONFAI_STUDIO_LLM` | Which backend drives the agent (for example `anthropic`, or an OpenAI-compatible server). |
| `KONFAI_STUDIO_LLM_API_KEY` | Key for that backend. |
| `KONFAI_STUDIO_LLM_BASE_URL` | Base URL of an OpenAI-compatible server (vLLM / Ollama / LM Studio). |
| `KONFAI_STUDIO_MODEL` | Main model id. |
| `KONFAI_STUDIO_SIDE_MODEL` | Model used for the cheaper side calls. |
| `KONFAI_STUDIO_MAX_TOKENS` | Per-response token ceiling. |
| `KONFAI_STUDIO_MAX_TURNS` | Agent-loop turn ceiling. |
| `KONFAI_STUDIO_TERMINAL` | Enables the in-app terminal. |
| `KONFAI_STUDIO_SLICER` | The 3D Slicer executable the viewer launches when no Slicer is listening (default: `Slicer` on `PATH`). |
| `KONFAI_STUDIO_PROXY_HEADERS` / `KONFAI_STUDIO_LOOPBACK` | Set by `konfai-studio` for the server it starts, from `--proxy-headers` and from whether `--host` is a loopback address: a value in your shell is overwritten. |

## Next steps

- {doc}`components/models`: the component names those YAML configs can reference
- {doc}`../usage/apps`: the guided workflow behind `konfai-apps`
- {doc}`../usage/python-api`: the same workflows as Python callables

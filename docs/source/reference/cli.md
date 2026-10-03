# CLI reference

| Command | Package | Purpose |
| --- | --- | --- |
| `konfai` | `konfai` | run a YAML workflow: train, predict, evaluate, transform |
| `konfai-cluster` | `konfai` (needs the `cluster` extra to submit) | submit those workflows to SLURM |
| `konfai-apps` | `konfai-apps` | run a packaged app |
| `konfai-apps-server` | `konfai-apps` | serve apps over HTTP |
| `konfai-mcp` | `konfai-mcp` | expose KonfAI to an MCP client |
| `konfai-studio` | `konfai-studio` | the web UI over `konfai-mcp` |

## `konfai`

| Command | Purpose | Default config | Root key |
| --- | --- | --- | --- |
| `TRAIN` | train a model | `./Config.yml` | `Trainer:` |
| `RESUME` | continue training from a checkpoint | `./Config.yml` | `Trainer:` |
| `PREDICTION` | run inference with one or more checkpoints | `./Prediction.yml` | `Predictor:` |
| `EVALUATION` | compute metrics on saved outputs | `./Evaluation.yml` | `Evaluator:` |
| `TRANSFORM` | prepare a dataset | `./Transform.yml` | `Transformer:` |
| `list` | print the components a config can name | | |

A run writes the resolved defaults back into its config file. A run that fails while building its workflow
leaves the file as you wrote it ({doc}`../config_guide/index`).

### Options

| Option | Meaning |
| --- | --- |
| `-c`, `--config` | The YAML file. |
| `-y`, `--overwrite` | Overwrite existing outputs without asking. Under `TRANSFORM` and `PREDICTION`, recompute cases that are already written. Without it, `PREDICTION` skips them, and refuses a `Prediction.yml` or checkpoints other than the ones they were written with. |
| `--gpu` | GPU ids (`--gpu 0 1`). Without it, the run is on the CPU. |
| `--cpu` | Number of CPU processes when no `--gpu` is given. Under `TRANSFORM`, the number of processes sharing the cases. |
| `-q`, `--quiet` | Less console output. |
| `-tb`, `--tensorboard` | Start TensorBoard on `127.0.0.1` (needs the `tensorboard` extra). Not for `EVALUATION` or `TRANSFORM`. |
| `--init` | Write the config file with every default resolved, and exit without running. |

Per command:

| Command | Options |
| --- | --- |
| `TRAIN` | `--checkpoints-dir` (`./Checkpoints/`), `--statistics-dir` (`./Statistics/`) |
| `RESUME` | `--model` (required): the checkpoint; `--lr` to change the learning rate; the `TRAIN` directories |
| `PREDICTION` | `--models` (required): one or more checkpoints, several make an ensemble; `--predictions-dir` (`./Predictions/`) |
| `EVALUATION` | `--evaluations-dir` (`./Evaluations/`) |
| `TRANSFORM` | `--plan` prints the plan and stops; `--transforms-dir` (`./Transforms/`) holds the run logs |

The underscore spellings (`--checkpoints_dir`) also work. `konfai --version` prints the version.

Each run prints the devices it uses (`[KonfAI] Running on cuda:0`) and, when it ends, where it wrote its
outputs. An unknown GPU id is a usage error.

### Generating a config

```bash
konfai TRAIN --init -c Config.yml
```

`--init` creates the file if needed, fills in every default, and exits. It reads no data, so it works in an
empty folder. What the defaults cannot decide (an output name, a `Write` destination) is printed as what is
left to complete.

### Listing components

```bash
konfai list transforms
```

`konfai list {transforms,augmentations,criteria,reductions,schedulers,models,blocks}` prints the exact names
a config can use, with one line of description each. `konfai list models` covers the Python models
(`segmentation.UNet.UNet`) and the YAML catalog (`default|UNet.yml`).

### How a run is launched

A run with one device runs in the current process. With several, KonfAI starts one process per device.
Training and evaluation connect them through `torch.distributed`; prediction and transform processes work
on their own cases independently. On Windows, `TRAIN`, `RESUME` and `EVALUATION` run on one process only.

No workflow lets several processes write to a single-file format (`h5`). Ctrl+C stops a run with exit code
130. A process killed by the system for lack of memory ends the run with a message saying so.

## `konfai-apps`

Runs a packaged app ({doc}`../usage/apps`).

| Command | Purpose |
| --- | --- |
| `infer` | run inference |
| `eval` | evaluate against a ground truth |
| `uncertainty` | estimate uncertainty |
| `pipeline` | inference, evaluation, and optionally uncertainty; `--gt` is required |
| `fine-tune` | fine-tune the app on a dataset |
| `bundle` | build an app bundle, optionally with an ONNX model |
| `download` | download an app's files for offline use |

Options shared by `infer`, `eval`, `uncertainty`, `pipeline` and `fine-tune`:

| Option | Meaning |
| --- | --- |
| `app` | The app: a Hugging Face `repo:app`, a local folder, or a server's app. |
| `--host`, `--port`, `--token` | Run on a `konfai-apps-server` instead of locally. |
| `-i`, `--inputs` | Input files; repeat the flag for each input group. |
| `-o`, `--output` | Output directory. |
| `--gpu` / `--cpu` | The device (one or the other). |
| `--tmp-dir` | Where intermediate files go. |
| `-q`, `--quiet` | Less output. |
| `--download` | Download the whole app first. |
| `--force_update` | Download the app again. |

Per command:

| Command | Options |
| --- | --- |
| `infer` | `--ensemble` or `--ensemble-models`, `--tta`, `-uncertainty`, `--prediction-file` |
| `eval` | `--gt`, `--mask`, `--evaluation-file` |
| `uncertainty` | `--uncertainty-file` |
| `pipeline` | those of `infer`, `eval` and `uncertainty` |
| `fine-tune` | `name`, `-d/--dataset`, `--models` (which checkpoints), `--epochs`, `--it-validation`, `--lr`, `--batch-size`, `--config` |

`bundle NAME` takes `--out`, `--app-json`, `--config` and `--checkpoint`. `download APP [FILES…]` takes
`--no-force-update`. Run `--help` on either for details.

### Tuning an app without editing it

`infer` and `pipeline` take the three options below; `fine-tune` takes `--set` and `--batch-size`:

| Option | Meaning |
| --- | --- |
| `--set NAME=VALUE` | Change a config value (repeatable). A bare name changes a model parameter (`--set iterations=300`), a dotted one a full path (`--set Predictor.Dataset.batch_size=2`). The value is read as YAML. |
| `--patch-size` | The inference patch size (one value for a cube). |
| `--batch-size` | The batch size. Without it, the app decides (`batch_size: 0` measures it on the GPU). |

They also work on a remote server; a server that does not support one refuses the job.

## `konfai-apps-server`

Serves apps over HTTP, for `konfai-apps --host …` ({doc}`app-server-api`).

| Option | Meaning |
| --- | --- |
| `--host`, `--port` | Bind address and port. |
| `--auth` | `off` or `bearer`. |
| `--token-env` | The environment variable holding the token. |
| `--token` | A token on the command line, for development. |
| `--apps` | A JSON file listing the apps. |
| `--download` | Download the apps at startup. |
| `--check` | Check the apps and exit without serving. |

## `konfai-cluster`

Submits a `konfai` workflow to SLURM. Its options go **before** the workflow command:

```bash
konfai-cluster --name my_job --num-nodes 2 TRAIN -y --gpu 0 1 --config Config.yml
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--name` | required | Job name. |
| `--num-nodes` | `1` | Nodes. |
| `--memory` | `16` | Memory per node, in GB. |
| `--time-limit` | `1440` | Time limit, in minutes. |

`--gpu` is required: the job runs one process per listed GPU on each node. `--plan` is refused (a plan
submits nothing).

## `konfai-mcp`

Runs the MCP server ({doc}`../usage/mcp`). Every option but `--i-know-this-is-insecure` also has an
environment variable ([below](#konfai-mcp-variables)).

| Option | Meaning |
| --- | --- |
| `--transport` | `stdio` (default), `sse` or `streamable-http`. |
| `--session` | Default session name. |
| `--workspace-root` | Where sessions and datasets live. |
| `--log-tail-lines` | Default number of log lines returned. |
| `--host`, `--port`, `--path` | Bind address, port and path, for SSE and HTTP. |
| `--log-level` | Server log level. |
| `--bearer-token` | Token required over SSE and HTTP. |
| `--stateless-http` | Streamable HTTP without sessions, for replicas behind a load balancer. |
| `--json-response` | Answer in plain JSON instead of an event stream, for buffering proxies. |
| `--i-know-this-is-insecure` | Allow a network address without a token. |

```{warning}
Over SSE or HTTP, a network address without a bearer token is refused: anyone on the network could run jobs
and read files. On loopback without a token, the server only answers `127.0.0.1`, `localhost` and `::1`.
```

## `konfai-studio`

Runs the Studio web UI ({doc}`../usage/studio`).

| Option | Meaning |
| --- | --- |
| `--host`, `--port` | Bind address (`127.0.0.1`) and port (`8730`). |
| `--proxy-headers` | Trust `X-Forwarded-*`, behind nginx or Caddy. |
| `--forwarded-allow-ips` | Proxies allowed to set them (`127.0.0.1`). |
| `--ssl-certfile`, `--ssl-keyfile` | Serve HTTPS. |
| `--i-know-this-is-insecure` | Allow a network address without `KONFAI_STUDIO_TOKEN`. |

```{warning}
Studio runs code on the host. A network address without `KONFAI_STUDIO_TOKEN` is refused. Set a token and
serve over TLS (`konfai-studio/docs/REMOTE.md`).
```

## Exit codes

`konfai`, `konfai-cluster`, `konfai-apps` and the app CLIs share one convention:

| Code | Meaning |
| --- | --- |
| `0` | Success. |
| `1` | The command ran and failed (a refusal or an error). |
| `2` | Usage error: a missing or invalid argument. |
| `130` | Interrupted by Ctrl+C. |

A process stopped by a signal ends with it (a shell shows 128 plus the signal number). `impact-reg-konfai`
does not use `130` yet. The servers exit with `0` on Ctrl+C; `konfai-mcp` on `stdio` exits with `130`, or on
its own when its client closes.

ONNX export has no subcommand: it is a Python function ({doc}`../usage/python-api`).

## Environment variables

### Variables you may set

| Variable | Effect |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | Which GPUs are visible. KonfAI sets it from `--gpu`; GPUs named by UUID are refused. |
| `KONFAI_API_TOKEN` | Bearer token of `konfai-apps` in remote mode and of `konfai-apps-server`. |
| `KONFAI_APPS_INSTALL_REQUIREMENTS` | `0` stops `konfai-apps` from installing an app's `requirements.txt` ({doc}`../usage/apps`). |
| `KONFAI_APPS_MAX_DATASET_BYTES` | Largest fine-tune dataset `konfai-apps-server` accepts (default 64 GiB; above, HTTP 413). |
| `KONFAI_DECOMPRESSED_DIRECTORY` | Where compressed images are decompressed for region reads (default `~/.cache/konfai/decompressed`). |
| `KONFAI_TENSORBOARD_HOST` | The address TensorBoard binds (default `127.0.0.1`). TensorBoard has no authentication: from another machine, forward the port with `ssh -N -L <port>:127.0.0.1:<port> user@server`. |
| `KONFAI_DEBUG` | `1` shows the full traceback of a KonfAI refusal, and records the last layer reached in `KONFAI_DEBUG_LAST_LAYER`. |
| `KONFAI_STREAMED_WRITES` | `0` turns off streamed prediction writes, to compare against. |
| `KONFAI_STREAM_WORTH_THRESHOLD` | The output size above which prediction streams, as a share of the budget (`0` always streams). |
| `KONFAI_ASYNC_WRITES` | Turns the background writer on or off. |
| `KONFAI_INLINE_SINGLE_RANK` | `0` runs a single process through the multi-process path, for a host that keeps its own CUDA context. |

Hugging Face's own variables (`HF_TOKEN`, `HF_HUB_OFFLINE`, …) apply to app downloads.

### Variables KonfAI sets

The CLI sets these for the run; you do not set them yourself.

| Variable | Purpose |
| --- | --- |
| `KONFAI_config_file`, `KONFAI_ROOT`, `KONFAI_STATE` | The config file, its root key and the command. |
| `KONFAI_CHECKPOINTS_DIRECTORY`, `KONFAI_STATISTICS_DIRECTORY`, `KONFAI_PREDICTIONS_DIRECTORY`, `KONFAI_EVALUATIONS_DIRECTORY`, `KONFAI_TRANSFORMS_DIRECTORY` | The output directories. |
| `KONFAI_OVERWRITE`, `KONFAI_VERBOSE` | `-y` and the inverse of `-q`. |
| `KONFAI_TENSORBOARD_PORT` | The TensorBoard port. |
| `KONFAI_CLUSTER` | Set under `konfai-cluster`. |
| `KONFAI_CONFIG_MODE`, `KONFAI_CONFIG_PATH` | The config binder's state. |
| `KONFAI_APPS_CONFIG` | The config an app runs. |
| `KONFAI_MASTER_PORT` | The port processes meet on. |
| `KONFAI_LOCAL_RANKS` | How many processes share the machine's memory budget. |
| `KONFAI_DECOMPRESSED_RUN` | This run's folder of decompressed images, removed at the end. |

### konfai-mcp variables

| Variable | Option | Effect |
| --- | --- | --- |
| `KONFAI_MCP_WORKSPACES_ROOT` | `--workspace-root` | Where sessions and datasets live. |
| `KONFAI_MCP_SESSION` | `--session` | Default session name. |
| `KONFAI_MCP_TRANSPORT` | `--transport` | `stdio`, `sse` or `streamable-http`. |
| `KONFAI_MCP_HOST`, `KONFAI_MCP_PORT` | `--host`, `--port` | Bind address and port. |
| `KONFAI_MCP_PATH` | `--path` | HTTP path. |
| `KONFAI_MCP_BEARER_TOKEN` | `--bearer-token` | Token over SSE and HTTP. |
| `KONFAI_MCP_LOG_LEVEL` | `--log-level` | Log level. |
| `KONFAI_MCP_LOG_TAIL_LINES` | `--log-tail-lines` | Default number of log lines. |
| `KONFAI_MCP_STATELESS_HTTP`, `KONFAI_MCP_JSON_RESPONSE` | `--stateless-http`, `--json-response` | `1` turns them on. |
| `KONFAI_MCP_APP_CATALOG` | | A JSON file of app sources added to the catalogue ({doc}`../usage/mcp`). |
| `KONFAI_MCP_SUBPROCESS_TIMEOUT` | | Seconds a validation or plan may take (default `1800`, `0` for no limit). |
| `KONFAI_MCP_VALIDATE_ROOT` | | Scratch folder of `validate_workflow_api` when called directly. |

The option wins over the variable.

### KonfAI Studio variables

| Variable | Effect |
| --- | --- |
| `KONFAI_STUDIO_TOKEN` | Shared bearer token. Unset means no authentication, so a network address is refused. |
| `KONFAI_STUDIO_INSECURE_COOKIE` | Drop the cookie's `Secure` flag, for plain-HTTP tests. |
| `KONFAI_STUDIO_LLM` | Model backend: `claude-code` (default), `anthropic`, or `openai` for a compatible endpoint. |
| `KONFAI_STUDIO_LLM_API_KEY` | Its key. |
| `KONFAI_STUDIO_LLM_BASE_URL` | The URL of an OpenAI-compatible server (vLLM, Ollama, LM Studio). |
| `KONFAI_STUDIO_MODEL`, `KONFAI_STUDIO_SIDE_MODEL` | The main model and the one for cheaper side calls. |
| `KONFAI_STUDIO_MAX_TOKENS`, `KONFAI_STUDIO_MAX_TURNS` | Limits per answer and per tool loop. |
| `KONFAI_STUDIO_TERMINAL` | Turn on the in-app terminal. |
| `KONFAI_STUDIO_SLICER` | The 3D Slicer executable to launch (default: `Slicer` on `PATH`). |
| `KONFAI_STUDIO_PROXY_HEADERS`, `KONFAI_STUDIO_LOOPBACK` | Set by `konfai-studio` itself from its options. |

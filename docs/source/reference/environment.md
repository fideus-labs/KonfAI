# Environment variables

This page catalogues the environment variables KonfAI reads or sets — the
user-facing ones you may set yourself, and the `KONFAI_*` runtime variables the
CLI wrappers manage. Reach for it when a run behaves differently across shells
or machines, or when you are debugging the runtime wrappers themselves.

## User-facing variables

### `CUDA_VISIBLE_DEVICES`

Controls which GPUs are visible to PyTorch and therefore to KonfAI.

KonfAI also rewrites this variable internally when you pass `--gpu`.

### `KONFAI_API_TOKEN`

Bearer token used by:

- `konfai-apps` in remote mode
- `konfai-apps-server` in bearer-auth mode

### `KONFAI_APPS_INSTALL_REQUIREMENTS`

Set to `0` to stop `konfai-apps` from pip-installing a resolved app's
`requirements.txt` (installed by default; core packages are never touched).
This is a **trust-model** switch — see the apps guide.

### Streaming and write-path switches

Diagnostic kill-switches for the streamed prediction writer. Defaults are the
streamed behavior; set to `0`/a value only to compare against the whole-volume
path or to tune the gate.

| Variable | Effect |
| --- | --- |
| `KONFAI_STREAMED_WRITES` | `0` disables streamed writes entirely (whole-volume reference path). |
| `KONFAI_STREAM_LINEAR_RESAMPLE` | `0` restores bit-exact (non-streamed) linear resample inverses. |
| `KONFAI_STREAM_WORTH_THRESHOLD` | Overrides the "worth streaming" accumulator-size threshold (fraction of allocatable memory). |
| `KONFAI_ASYNC_WRITES` | Controls the background writer for disjoint-file sinks. |

### Hugging Face authentication

The repository and CI also rely on Hugging Face-hosted assets. KonfAI itself
uses `huggingface_hub`, so standard Hugging Face authentication variables may be
relevant in practice, but they are not KonfAI-specific.

## Runtime variables set by KonfAI

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

## Internal debug/config variables

The codebase also references internal variables such as:

- `KONFAI_CONFIG_MODE`, `KONFAI_CONFIG_PATH` — the config binder's mode machine
- `KONFAI_APPS_CONFIG`
- `KONFAI_DEBUG`, `KONFAI_DEBUG_LAST_LAYER`
- `KONFAI_MASTER_PORT`, `KONFAI_LOCAL_RANKS` — distributed rendezvous bookkeeping
- `KONFAI_ATTR_KEY`, `KONFAI_DEPS`, `KONFAI_COMPONENT_BASES`, `KONFAI_VERSION`

These are part of KonfAI's internal execution model and are best treated as
implementation details unless you are actively extending the framework.

The `konfai-mcp` server has its own `KONFAI_MCP_*` family (workspace root,
transport, host/port, bearer token, log level, subprocess timeout, session and
app-catalog selection) — documented in the MCP guide.

## Next steps

- {doc}`cli` — the wrappers that set the `KONFAI_*` runtime variables
- {doc}`../concepts/execution-flow` — where in the launch sequence they are set

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
| `KONFAI_STREAM_WORTH_THRESHOLD` | Overrides the "worth streaming" accumulator-size threshold (fraction of the per-rank memory budget). Test harnesses set `0` to force the streamed machinery on toy volumes. |
| `KONFAI_ASYNC_WRITES` | Controls the background writer for disjoint-file sinks. |
| `KONFAI_INLINE_SINGLE_RANK` | Default on. `0` forces a single rank through the spawn path instead of running it in-process — useful when a host process must keep its own CUDA context. |

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
- `KONFAI_DEBUG` — `1` re-attaches the framework traceback to a designed refusal (a
  `KonfAIError`), which otherwise prints its message and remedy alone; `KONFAI_DEBUG_LAST_LAYER`
- `KONFAI_MASTER_PORT` — distributed rendezvous bookkeeping
- `KONFAI_LOCAL_RANKS` — how many ranks share one node's RAM, published by the
  launcher so a node-scoped `memory_budget` is divided before the spawn. It changes
  the cache-versus-stream decision, so it is not mere bookkeeping.
- `KONFAI_ATTR_KEY`, `KONFAI_DEPS`, `KONFAI_COMPONENT_BASES`, `KONFAI_VERSION`

These are part of KonfAI's internal execution model and are best treated as
implementation details unless you are actively extending the framework.

## konfai-mcp

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
their own — the app catalog, the subprocess timeout, the validation root — and
are covered in {doc}`../usage/mcp`.

## KonfAI Studio

`konfai-studio` reads its own family. The first two are security-relevant: Studio
drives arbitrary host compute, so binding a non-loopback address without a token is
refused unless you override it. See `studio/docs/REMOTE.md`.

| Variable | Effect |
| --- | --- |
| `KONFAI_STUDIO_TOKEN` | Shared bearer token. **Unset means no authentication**, which is why a non-loopback bind is refused without it. |
| `KONFAI_STUDIO_INSECURE_COOKIE` | Drops the `Secure` flag on the session cookie — for plain-HTTP testing only. |
| `KONFAI_STUDIO_LLM` | Which backend drives the agent (for example `anthropic`, or an OpenAI-compatible server). |
| `KONFAI_STUDIO_LLM_API_KEY` | Key for that backend. |
| `KONFAI_STUDIO_LLM_BASE_URL` | Base URL of an OpenAI-compatible server (vLLM / Ollama / LM Studio). |
| `KONFAI_STUDIO_MODEL` | Main model id. |
| `KONFAI_STUDIO_SIDE_MODEL` | Model used for the cheaper side calls. |
| `KONFAI_STUDIO_MAX_TOKENS` | Per-response token ceiling. |
| `KONFAI_STUDIO_MAX_TURNS` | Agent-loop turn ceiling. |
| `KONFAI_STUDIO_TERMINAL` | Enables the in-app terminal. |

## Next steps

- {doc}`cli` — the wrappers that set the `KONFAI_*` runtime variables
- {doc}`../concepts/execution-flow` — where in the launch sequence they are set

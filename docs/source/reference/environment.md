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

### Hugging Face authentication

The repository and CI also rely on Hugging Face-hosted assets. KonfAI itself
uses `huggingface_hub`, so standard Hugging Face authentication variables may be
relevant in practice, but they are not KonfAI-specific.

## Runtime variables set by KonfAI

**These variables are normally set by the CLI wrappers and are not expected to
be managed manually in day-to-day usage.**

| Variable | Set by | Purpose |
| --- | --- | --- |
| `KONFAI_config_file` | train/predict/evaluate wrappers | Active YAML file path. |
| `KONFAI_ROOT` | train/predict/evaluate wrappers | Root config object: `Trainer`, `Predictor`, or `Evaluator`. |
| `KONFAI_STATE` | train/predict/evaluate wrappers | Active workflow state. |
| `KONFAI_CHECKPOINTS_DIRECTORY` | training wrapper | Checkpoint output directory. |
| `KONFAI_STATISTICS_DIRECTORY` | training wrapper | Statistics output directory. |
| `KONFAI_PREDICTIONS_DIRECTORY` | prediction wrapper | Prediction output directory. |
| `KONFAI_EVALUATIONS_DIRECTORY` | evaluation wrapper | Evaluation output directory. |
| `KONFAI_OVERWRITE` | distributed wrapper | Mirrors the `--overwrite` flag. |
| `KONFAI_TENSORBOARD_PORT` | distributed wrapper | Selected TensorBoard port. |
| `KONFAI_VERBOSE` | distributed wrapper | Mirrors the inverse of `--quiet`. |
| `KONFAI_CLUSTER` | cluster wrapper | Marks cluster execution. |

## Internal debug/config variables

The codebase also references internal variables such as:

- `KONFAI_CONFIG_MODE`
- `KONFAI_CONFIG_PATH`
- `KONFAI_CONFIG_VARIABLE`
- `KONFAI_APPS_CONFIG`
- `KONFAI_DEBUG`
- `KONFAI_DEBUG_LAST_LAYER`

These are part of KonfAI's internal execution model and are best treated as
implementation details unless you are actively extending the framework.

## Next steps

- {doc}`cli` — the wrappers that set the `KONFAI_*` runtime variables
- {doc}`../concepts/execution-flow` — where in the launch sequence they are set

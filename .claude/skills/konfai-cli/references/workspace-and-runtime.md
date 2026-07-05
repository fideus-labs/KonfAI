# Workspace, outputs, and the runtime

## You don't set `KONFAI_*` yourself

The `konfai` CLI is a thin front-end. When you run a subcommand, the runtime
(`run_distributed_app` / `execute_distributed_object`) sets every `KONFAI_*` environment
variable, **forces `KONFAI_CONFIG_MODE='Done'`**, and spawns the workflow. You configure
runs through the **YAML + CLI flags**, not env vars. The variables below matter only for
debugging or advanced setups.

| Variable | Set by runtime to | Notes |
|---|---|---|
| `KONFAI_config_file` | resolved absolute path of the active config | read by every `Config()`; the file is rewritten on exit |
| `KONFAI_CONFIG_MODE` | `Done` | Trainer/Predictor/Evaluator raise `ConfigError` unless it is `Done` |
| `KONFAI_ROOT` | `Trainer` / `Predictor` / `Evaluator` | anchors reflection paths |
| `KONFAI_STATE` | `TRAIN` / `RESUME` / `PREDICTION` / `EVALUATION` | the subcommand token |
| `KONFAI_CHECKPOINTS_DIRECTORY` | `./Checkpoints/` (or `--checkpoints-dir`) | |
| `KONFAI_STATISTICS_DIRECTORY` | `./Statistics/` (or `--statistics-dir`) | |
| `KONFAI_PREDICTIONS_DIRECTORY` | `./Predictions/` (or `--predictions-dir`) | |
| `KONFAI_EVALUATIONS_DIRECTORY` | `./Evaluations/` (or `--evaluations-dir`) | |
| `KONFAI_OVERWRITE` | `True` when `-y/--overwrite` | skips the overwrite prompt |
| `CUDA_VISIBLE_DEVICES` | frozen at startup | valid `--gpu` ids are validated against it |
| `KONFAI_MASTER_PORT` / `KONFAI_TENSORBOARD_PORT` | DDP rendezvous / TensorBoard port | |
| `KONFAI_VERBOSE` / `KONFAI_CLUSTER` | from `--quiet` / SLURM submission | verbosity and per-rank logging |

## Workspace layout — keyed by `train_name`

Every output directory is namespaced by the `train_name` set in the config (default
`TRAIN_01`; the Segmentation example uses `SEG_BASELINE`). A full train→predict→evaluate
loop produces:

```text
Checkpoints/<train_name>/    # timestamped <YYYY_MM_DD_HH_MM_SS>.pt (each holds epoch, it, loss, Model state)
Statistics/<train_name>/     # tb/ TensorBoard logdir + the resolved-config snapshot of the training run
Predictions/<train_name>/    # the Prediction.yml snapshot + one output-dataset subfolder per exported output
Evaluations/<train_name>/    # the Evaluation.yml snapshot + Metric_<split>.json (e.g. Metric_TRAIN.json)
```

The **resolved-config snapshot** written next to each run is the reproducible record — it is
the config after defaults were materialised. (Prediction and evaluation write their snapshot
into `Predictions/` and `Evaluations/`; there is no separate top-level snapshot directory.)

`train_name` is the single most important handle: **`Prediction.yml` and `Evaluation.yml`
must use the same `train_name`** as the training run whose checkpoints/predictions they
consume, or evaluation won't find its inputs.

## Config modes (mostly invisible, occasionally sharp)

Workflows always run in `KONFAI_CONFIG_MODE='Done'` (resolve + write back, no prompts). Other
modes exist in the engine and are worth knowing so you don't trip on them:

- `default` — materialise defaults non-interactively; may create a missing config file.
- `interactive` — prompt for every `default`-marked field.
- `remove` — **deletes** the config file on context exit (destructive).
- `Import` — suppresses config reading (used internally around imports).

Practical consequence: **there is no read-only path** — entering and exiting any `Config`
context rewrites the file (materialising defaults; `None` round-trips as the literal string
`"None"`). Keep configs under version control and expect a post-run diff.

## Parallelism and clusters

- **DDP — one process per GPU.** The runtime computes `world_size = len(gpu_ids)` (or the CPU
  worker count when no GPU) and `mp.spawn`s that many processes; each runs one rank.
  Disk/log side effects are gated on `global_rank == 0`.
- **SLURM.** `konfai-cluster` submits the same subcommands to a scheduler via `submitit`
  (adds `--name`, `--num-nodes`, `--memory`, `--time-limit`, `--resubmit`). Needs the
  `cluster` extra.

## Install note

Reading the `.mha` demo data needs SimpleITK — install the imaging extra:
`pip install "konfai[imaging]"` (a bare `pip install konfai` fails on first read). Other
extras: `itk`, `hdf5`, `dicom`, `omezarr`, `tensorboard`, `lpips`, `ssim`, `fid`, `cluster`.

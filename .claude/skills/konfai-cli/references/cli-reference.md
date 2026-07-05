# The `konfai` command-line reference

KonfAI installs two console scripts (`konfai` and `konfai-cluster`, entry points
`konfai.main:main` / `konfai.main:cluster`). Everything runs through four subcommands.

```
konfai <TRAIN|RESUME|PREDICTION|EVALUATION> [options]
konfai --version
```

The subcommand (`dest="command"`) is **required** and maps to the KonfAI `State`. TRAIN and
RESUME dispatch to `konfai.trainer.train`, PREDICTION to `konfai.predictor.predict`,
EVALUATION to `konfai.evaluator.evaluate`.

## Common options (every subcommand)

| Option | Meaning |
|---|---|
| `-c`, `--config PATH` | Path to the workflow YAML. If omitted, a command-specific default filename is used — **always pass it explicitly** to avoid ambiguity. |
| `-y`, `--overwrite` | Overwrite existing outputs (checkpoints, logs, predictions) without prompting. |
| `--gpu ID [ID ...]` | GPU device ids, constrained to the visible devices, e.g. `--gpu 0` or `--gpu 0 1 2`. Omit to run on CPU. |
| `--cpu N` | Run on CPU with `N` (>0) worker processes. **Mutually exclusive with `--gpu`.** |
| `-q`, `--quiet` | Suppress console output. |
| `-tb`, `--tensorboard` | Launch TensorBoard. |

`--gpu` and `--cpu` are a mutually-exclusive group. With neither, execution falls back to CPU.

## `TRAIN` — train from scratch

Reads a `Trainer:` config and runs the full training loop.

| Extra option | Default | Meaning |
|---|---|---|
| `--checkpoints-dir DIR` | `./Checkpoints/` | Where checkpoints are saved. |
| `--statistics-dir DIR` | `./Statistics/` | Where training statistics / TensorBoard logs are saved. |

```bash
konfai TRAIN -y --gpu 0 --config Config.yml
```

## `RESUME` — continue an existing run

Same as TRAIN plus checkpoint reload.

| Extra option | Default | Meaning |
|---|---|---|
| `--model PATH` | *(required)* | Checkpoint to resume from. |
| `-checkpoints-dir DIR` | `./Checkpoints/` | Checkpoints directory. |
| `-statistics-dir DIR` | `./Statistics/` | Statistics directory. |
| `--lr FLOAT` | *(unset)* | Override the learning rate. If omitted, the checkpoint LR resumes and the scheduler continues; if set, LR restarts from this value. |

```bash
konfai RESUME -y --gpu 0 --config Config.yml --model Checkpoints/TRAIN_01/last.pt
```

## `PREDICTION` — inference with a trained model

Reads a `Predictor:` config. The `--config` value is passed as `prediction_file`.

| Extra option | Default | Meaning |
|---|---|---|
| `--models PATH [PATH ...]` | *(required)* | One or more checkpoints. Passing several enables **ensembling**. |
| `--predictions-dir DIR` | `./Predictions/` | Where predictions are written. |

```bash
konfai PREDICTION -y --gpu 0 --config Prediction.yml --models Checkpoints/TRAIN_01/best.pt
```

## `EVALUATION` — score predictions against ground truth

Reads an `Evaluator:` config. The `--config` value is passed as `evaluations_file`.

| Extra option | Default | Meaning |
|---|---|---|
| `--evaluations-dir DIR` | `./Evaluations/` | Where per-case + aggregate metric JSON is written. |

```bash
konfai EVALUATION -y --config Evaluation.yml
```

## `konfai-cluster` — SLURM submission

Same four subcommands, plus a "Cluster manager arguments" group that submits via `submitit`
instead of running locally:

| Option | Default | Meaning |
|---|---|---|
| `--name NAME` | *(required)* | Job name. |
| `--num-nodes N` | `1` | Number of nodes. |
| `--memory GB` | `16` | Memory per node. |
| `--time-limit MIN` | `1440` | Job time limit (minutes). |
| `--resubmit` | off | Auto-resubmit just before timeout. |

```bash
konfai-cluster TRAIN --name seg_run --num-nodes 1 --gpu 0 --config Config.yml
```

Requires the `cluster` extra (`pip install konfai[cluster]`).

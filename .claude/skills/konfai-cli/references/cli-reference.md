# The `konfai` command-line reference

KonfAI installs two console scripts (`konfai` and `konfai-cluster`, entry points
`konfai.main:main` / `konfai.main:cluster`). Everything runs through five workflow subcommands,
plus `list`, which prints the components a config can reference.

```text
konfai <TRAIN|RESUME|PREDICTION|EVALUATION|TRANSFORM> [options]
konfai list <transforms|augmentations|criteria|reductions|models|blocks>
konfai --version
```

The subcommand (`dest="command"`) is **required** and maps to the KonfAI `State`. TRAIN and
RESUME dispatch to `konfai.trainer.train`, PREDICTION to `konfai.predictor.predict`,
EVALUATION to `konfai.evaluator.evaluate`, TRANSFORM to `konfai.transformer.transform`
(`konfai.transformer.plan_transform` under `--plan`).

## Common options (TRAIN, RESUME, PREDICTION, EVALUATION)

| Option | Meaning |
|---|---|
| `-c`, `--config PATH` | Path to the workflow YAML. If omitted, a command-specific default filename is used: **always pass it explicitly** to avoid ambiguity. |
| `-y`, `--overwrite` | Overwrite existing outputs (checkpoints, logs, predictions) without prompting. |
| `--gpu ID [ID ...]` | GPU device ids, constrained to the visible devices, e.g. `--gpu 0` or `--gpu 0 1 2`. Omit to run on CPU. |
| `--cpu N` | Run on CPU with `N` (>0) worker processes. **Mutually exclusive with `--gpu`.** |
| `-q`, `--quiet` | Suppress console output. |
| `-tb`, `--tensorboard` | Launch TensorBoard (needs the `tensorboard` extra). Not accepted by `EVALUATION` or `TRANSFORM`. |
| `--init` | Create the config file if missing, resolve every default into it, and exit without running. |

`--gpu` and `--cpu` are a mutually-exclusive group. With neither, execution falls back to CPU.
TRANSFORM declares its own set (see below): no `-tb`.

## `TRAIN`: train from scratch

Reads a `Trainer:` config and runs the full training loop.

| Extra option | Default | Meaning |
|---|---|---|
| `--checkpoints-dir DIR` | `./Checkpoints/` | Where checkpoints are saved. |
| `--statistics-dir DIR` | `./Statistics/` | Where training statistics / TensorBoard logs are saved. |

```bash
konfai TRAIN -y --gpu 0 --config Config.yml
```

Checkpoints are named after the moment they were written, `<YYYY_MM_DD_HH_MM_SS>.pt`, in
`Checkpoints/<train_name>/`. Beside them, `resume_latest.pt` is the training continuation and
`crash_<date>.pt` a save on an exceptional exit; neither is a model to predict with.

## `RESUME`: continue an existing run

Same as TRAIN plus checkpoint reload.

| Extra option | Default | Meaning |
|---|---|---|
| `--model PATH` | *(required)* | Checkpoint to resume from. |
| `--checkpoints-dir DIR` | `./Checkpoints/` | Checkpoints directory. |
| `--statistics-dir DIR` | `./Statistics/` | Statistics directory. |
| `--lr FLOAT` | *(unset)* | Override the learning rate. If omitted, the checkpoint LR resumes and the scheduler continues; if set, LR restarts from this value. |

```bash
konfai RESUME -y --gpu 0 --config Config.yml --model Checkpoints/TRAIN_01/resume_latest.pt
```

## `PREDICTION`: inference with a trained model

Reads a `Predictor:` config. The `--config` value is passed as `prediction_file`.

| Extra option | Default | Meaning |
|---|---|---|
| `--models PATH [PATH ...]` | *(required)* | One or more checkpoints. Passing several enables **ensembling**. |
| `--predictions-dir DIR` | `./Predictions/` | Where predictions are written. |

```bash
konfai PREDICTION -y --gpu 0 --config Prediction.yml --models Checkpoints/TRAIN_01/<checkpoint>.pt
```

## `EVALUATION`: score predictions against ground truth

Reads an `Evaluator:` config. The `--config` value is passed as `evaluations_file`.

| Extra option | Default | Meaning |
|---|---|---|
| `--evaluations-dir DIR` | `./Evaluations/` | Where per-case + aggregate metric JSON is written. |

```bash
konfai EVALUATION -y --config Evaluation.yml
```

## `TRANSFORM`: prepare a dataset

Reads a `Transformer:` config. The `--config` value is passed as `transform_file`. It takes
`-c`, `-y`, `--gpu`, `--cpu`, `-q` and `--init` as above, with two differences: `-y` recomputes
the cases whose output exists (without it such a case is skipped), and `--cpu N` shards the
cases over `N` processes. There is no `-tb`.

| Extra option | Default | Meaning |
|---|---|---|
| `--plan` | off | Print the per-case streaming plan and exit without transforming. Printed even with `-q`. |
| `--transforms-dir DIR` | `./Transforms/` | Run logs; the outputs go where each `Write:` says. `--plan` writes nothing there. |

```bash
konfai TRANSFORM --config Transform.yml --plan
konfai TRANSFORM --config Transform.yml
```

## `list`: the components a config can reference

`konfai list {transforms,augmentations,criteria,reductions,models,blocks}` prints the spelling a
YAML config uses for each component of that family, with its one-line doc. It takes none of the
run flags.

```bash
konfai list criteria
```

## `konfai-cluster`: SLURM submission

Same subcommands, plus a "Cluster manager arguments" group that submits via `submitit`
instead of running locally. **The cluster options come before the subcommand**: they sit on the
top-level parser, so putting them after it fails with `the following arguments are required: --name`.

| Option | Default | Meaning |
|---|---|---|
| `--name NAME` | *(required)* | Job name. |
| `--num-nodes N` | `1` | Number of nodes. |
| `--memory GB` | `16` | Memory per node. |
| `--time-limit MIN` | `1440` | Job time limit (minutes). |

```bash
konfai-cluster --name seg_run --num-nodes 1 TRAIN -y --gpu 0 --config Config.yml
```

`--gpu` is required (one rank per listed GPU on each node) and `TRANSFORM --plan` is refused: a
plan submits nothing. Submitting requires the `cluster` extra (`pip install konfai[cluster]`).

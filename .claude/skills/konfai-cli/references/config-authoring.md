# Authoring KonfAI YAML configs

A KonfAI experiment is **fully described in YAML** and mapped onto Python objects by a
reflection engine — no experiment-specific Python for standard tasks. **The config is the
experiment**, and a finished run leaves a fully-resolved config on disk.

The fastest way to author a config is to **copy a runnable template from `examples/`**
(Segmentation or Synthesis) and adapt it, rather than writing from a blank file.

## The three files and their root keys

| Command | File (by convention) | Root key | Holds |
|---|---|---|---|
| `TRAIN` / `RESUME` | `Config.yml` | `Trainer:` | model + dataset + losses + augmentations + optimizer/schedulers + patch + training params |
| `PREDICTION` | `Prediction.yml` | `Predictor:` | model load(s), patch/TTA/ensemble inference, output post-processing |
| `EVALUATION` | `Evaluation.yml` | `Evaluator:` | predictions vs ground truth → per-case + aggregate metric JSON |

The root key is mandatory and load-bearing: a `Prediction.yml` must open with `Predictor:`.

## Reading a config rewrites it — the mutation invariant

`apply_config` reads a callable's signature and fills its arguments from the YAML subtree it
owns (`@config("Key")`), recursing into nested `@config` objects. **Resolved defaults are
written back to the file**, so every `konfai` run rewrites its config in place, materialising
defaults. Consequences for a CLI user:

- After a run, the on-disk YAML is the fully-resolved snapshot — commit it as the record of
  the experiment; a re-run is reproducible from it.
- Keep a pristine copy (or use version control) if you want to preserve exactly what you
  hand-wrote before defaults were expanded.
- Workflows run with `KONFAI_CONFIG_MODE='Done'` (forced), which is what triggers the
  write-back at the end of the run.

## Non-negotiable conventions

- **Arrays are channel-first**: `[C, (Z), Y, X]`.
- **Geometry/spacing is `(x, y, z)`** (SimpleITK order), keys `Origin` / `Spacing` / `Direction`.
- **Datasets are lazy and patch-based** — patch size, overlap, and streaming are config, not
  code. Never assume a whole volume is in RAM.

## Classpath resolution (how to name any object in YAML)

- A **bare name** resolves inside that kind's package: `Dice` (a criterion), `Flip`
  (an augmentation), `CosineAnnealing` (a scheduler). **Models are the exception** — a model
  is referenced by its **dotted classpath under `konfai.models`**, e.g. `segmentation.UNet.UNet`,
  not a bare name.
- `module:Class` imports **anything** — a **local file beside the config** (`Model:UNetpp5`,
  `UnNormalize:UnNormalize`) or an installed library (`monai.losses:DiceLoss`, `torch:nn:L1Loss`).
  KonfAI prepends the current working directory to `sys.path`, so a `.py` next to the config
  resolves. This is the pattern the Synthesis example uses.

Component kinds: `loss`/`metric` (both → a `Criterion`), `transform`, `augmentation`,
`scheduler`, `model`, `block`. To discover what exists, read the runnable `examples/`, the
authoritative config catalogue under `docs/source/config_guide/`
(`training.md` = `Trainer:`, `prediction.md` = `Predictor:`, `evaluation.md` = `Evaluator:`,
`patterns.md` = `dataset_filenames`/`groups_src`/`subset`/`validation` conventions), and the
base classes in `konfai/` (e.g. `konfai/metric/measure.py` for criteria,
`konfai/data/augmentation.py` for augmentations). The CLI is documented at
`docs/source/reference/cli.md`.

## The shape of each file (top-level keys)

- **`Trainer:`** → `Model` (`classpath` + selected-class args + `optimizer` + `outputs_criterions`
  + `schedulers`), `Dataset` (`dataset_filenames`, `groups_src`, `augmentations`, `Patch`),
  plus training params (epochs, `train_name`, etc.).
- **`Predictor:`** → `Model` (same `classpath` convention, inference-simplified),
  `Dataset` (`dataset_filenames`, `groups_src`, TTA `augmentations`, `Patch`), and
  `outputs_dataset` (which model output → which on-disk group).
- **`Evaluator:`** → `metrics` (predicted group → target group(s), optionally composed with
  `;` like `CT;MASK` → criteria) and `Dataset` (`DataMetric`: `dataset_filenames`, `groups_src`,
  `subset`, `validation`).

## Losses, metrics, and named module outputs

- Losses and metrics are both `Criterion` subclasses, attached under `outputs_criterions` /
  `metrics` to a **named module output**.
- **An `outputs_criterions` key equals a module's dotted path** — e.g. `UNetBlock_0:Head:Softmax`,
  or with the patch-accumulation marker `Generator_A_to_B:;accu;Head:Tanh`. The `:` / `.` /
  `;accu;` separators are **load-bearing**; do not rewrite them.
- `;accu;` refers to the patch-wise output **before overlap-blended re-assembly** — used when a
  downstream module (e.g. a 3D discriminator) must see the reassembled chunk.
- A `forward` returns a `Tensor` for a loss or a `(value, dict)` tuple for a metric.

## What to adapt first for a new project

From either example: `dataset_filenames`, `train_name` (keys every output folder), patch size,
batch size, model class + hyperparameters, losses/monitored metrics, preprocessing transforms,
and (segmentation) `nb_class` + `Dice.labels` in `Evaluation.yml`.

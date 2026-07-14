# Authoring KonfAI YAML configs

A KonfAI experiment is **fully described in YAML**. There is no experiment-specific
Python to write for standard tasks — you describe the model, data pipeline,
losses/metrics, optimizer/schedulers, augmentations, and the whole workflow, and a
reflection engine maps them onto Python objects. **The config is the experiment.**

Use the MCP discovery tools to author configs instead of guessing: `list_components`
answers *what exists*, `inspect_object_signature` and `describe_config_schema` answer
*how to configure it*, and `describe_extension_points` / `check_external_dependency`
answer *what you can add* and *what is installed*.

## The three files and their root keys

Each CLI state maps to one file with one mandatory root key. Author the file that
matches the workflow you are running.

| Workflow (CLI) | File | Root key | Holds |
|---|---|---|---|
| `TRAIN` / `RESUME` | `Config.yml` | `Trainer:` | model + dataset + losses + augmentations + optimizer/schedulers + training params |
| `PREDICTION` | `Prediction.yml` | `Predictor:` | model load(s), patch/TTA/ensemble inference, output post-processing |
| `EVALUATION` | `Evaluation.yml` | `Evaluator:` | predictions vs ground truth → per-case + aggregate metric JSON |

`write_workflow_config` validates that the top-level root key matches the workflow
before writing, so a `Prediction.yml` whose root is `Trainer:` is rejected early.

> **Reading a config mutates it.** `apply_config` writes resolved defaults *back* to the
> file. After a validate or run, the on-disk YAML is the fully-resolved snapshot, not
> what you originally wrote. `validate_config_semantics` is the exception — it validates
> on a snapshot and restores your authored file, so validating never rewrites your work.

## Non-negotiable conventions

- **Arrays are channel-first**: `[C, (Z), Y, X]`.
- **Geometry/spacing is `(x, y, z)`** (SimpleITK order), keys `Origin` / `Spacing` / `Direction`.
- **Datasets are lazy and patch-based** — never assume a whole volume is in RAM. Patch
  size, overlap, and streaming are config, not code.

## Classpath resolution (how to name any object in YAML)

- A **bare name** resolves inside that kind's package: `Dice` (a criterion), `Flip`
  (an augmentation), `CosineAnnealing` (a scheduler). **Models are the exception** —
  a model is referenced by its **dotted classpath under `konfai.models`**, e.g.
  `segmentation.UNet.UNet`, not a bare name (a bare `UNet` does not resolve).
- `module:Class` imports **anything** — a local session file (`Loss:MyWrapper`) or an
  installed library (`monai.losses:DiceLoss`, `torch:nn:L1Loss`).
- Local `Module:Object` classpaths resolve **inside the current session workspace**, so a
  custom `.py` you added via `write_session_file` is referenceable immediately.

Component kinds understood by `list_components`: `loss` / `metric` (both → criterion),
`transform`, `augmentation`, `scheduler`, `model`, `block`. Get the exact reference string
for any component from the `config_reference` field of `list_components` — do not guess it.

## Losses, metrics, and named module outputs

- Losses and metrics are both `Criterion` subclasses, attached under
  `outputs_criterions` / `metrics` to a **named module output**.
- **An `outputs_criterions` key equals a module's dotted path** — e.g.
  `UNetBlock_0:Head:Softmax`. The `:` and `.` separators are **load-bearing**; do not
  rewrite them. A terminal/deep-supervision head is marked with `out_branch: [-1]`.
- `forward` returns a `Tensor` for a loss or a `(value, dict)` tuple for a metric;
  consumers `isinstance`-branch on that, so keep the contract.

## Dataset onboarding → aliases

Datasets map raw group names to KonfAI roles via **aliases** such as `IMG -> CT`.
The typical path:

1. `browse_dataset` (or `inspect_dataset(include_stats=False)`) to see roots / cohorts / groups.
2. `inspect_dataset` on the chosen root for group structure + sampled stats +
   missing-by-case + a suggested `dataset_entry`.
3. `inspect_dataset(groups=[...])` when you need intensity ranges to choose normalization.
4. `prepare_dataset_aliases` to fix the group→role mapping the config will use.

Feed the resulting `dataset_entry` / aliases into `design_config_strategy`, then into the
YAML you write with `write_workflow_config`.

## Authoring loop that avoids invalid configs

1. `design_config_strategy` — turn task + dataset roots + group roles into a config plan.
2. `initialize_session` — create the workspace, optionally seeding from an example template.
3. `list_components` / `inspect_object_signature` / `describe_config_schema` — resolve
   every object name and its parameters *before* writing them.
4. `write_workflow_config` — write one workflow YAML (root key checked).
5. `review_config_semantics` — cheap, no runtime instantiation; fix any `blocking_issues`.
6. `validate_config_semantics` — instantiates KonfAI objects on a snapshot to catch
   runtime-facing errors; side-effect-free.

Only after step 6 is clean do you launch a run. See `troubleshooting.md` for what to do
when a step fails.

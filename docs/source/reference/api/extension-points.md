# Extension points

KonfAI is designed to be extended mostly through configuration-aware Python
classes rather than through a plugin registry with explicit manifests.

This page documents the extension mechanisms that are clearly visible in the
codebase and examples.

## Local Python modules next to YAML files

The most practical extension mechanism is to place Python modules next to your
configuration files and refer to them through `classpath`.

The shipped synthesis example does this with:

- `examples/Synthesis/Model.py`
- `examples/Synthesis/UnNormalize.py`

This pattern is appropriate for:

- custom models
- post-processing transforms
- local research prototypes

## `@config(...)`

Use `konfai.utils.config.config` to bind a class to a configuration key.

Example use cases visible in the codebase:

- `Trainer`
- `Predictor`
- `Evaluator`
- `EarlyStopping`
- `OptimizerLoader`

Why it exists:

- it lets `apply_config(...)` instantiate the object from the right YAML branch
- it keeps the YAML structure aligned with constructor signatures

For local custom classes next to YAML files, do not use `@config()` by default.
Without any decorator, the class reads its constructor parameters directly from
the current YAML branch, which is usually the most readable layout.

In the current codebase, when you do use a decorator:

- `@config("SomeKey")` binds the object to `SomeKey`
- `@config()` defaults to the class name

Use `@config("SomeKey")` only when you intentionally want that extra nesting.

## `classpath`

Use `classpath` when a YAML branch must resolve to a concrete implementation at
runtime.

This appears in the examples for:

- models
- transforms
- losses and metrics

Why it exists:

- it keeps the core framework generic
- it lets projects mix built-in and local modules

## Dataset transforms and augmentations

Transforms and augmentations are also extension points.

Relevant modules:

- `konfai.data.transform`
- `konfai.data.augmentation`

Use this path when you need custom preprocessing, postprocessing, or data
augmentation behavior.

Runtime contracts:

- transforms should inherit `konfai.data.transform.Transform` or
  `TransformInverse`
- augmentations should inherit `konfai.data.augmentation.DataAugmentation`

## Criteria and schedulers

KonfAI lets you attach multiple losses and metrics to multiple outputs and
targets. The relevant extension points live in:

- `konfai.metric.measure`
- `konfai.metric.schedulers`
- `konfai.network.network.TargetCriterionsLoader`

This is the mechanism used by the examples to define reconstruction losses,
Dice-based evaluation, adversarial losses, and scheduled weights.

Runtime contracts:

- simple criteria should inherit `konfai.metric.measure.Criterion`
- criteria that need model graph initialization should inherit
  `CriterionWithInit`
- criteria that need per-sample metadata should inherit
  `CriterionWithAttribute`

## Quick contract table

| Extension point | Recommended base class | Typical YAML entry point |
| --- | --- | --- |
| Custom model | `konfai.network.network.Network` | `Trainer.Model.classpath` |
| Custom transform | `konfai.data.transform.Transform` or `TransformInverse` | `groups_dest.<group>.transforms` |
| Custom augmentation | `konfai.data.augmentation.DataAugmentation` | `Dataset.augmentations.*.data_augmentations` |
| Custom loss / metric | `konfai.metric.measure.Criterion` family | `outputs_criterions.*.targets_criterions.*.criterions_loader` |

For a practical, contract-oriented guide with code snippets, see
{doc}`../../usage/custom-models`.

## KonfAI Apps

At a higher level, an entire workflow can be packaged as a KonfAI App. This is
the preferred extension path when a workflow is already mature and should be
reused through a stable interface.

See {doc}`../../usage/apps`.

## Caveat

KonfAI is highly configurable, but not every internal helper is a stable public
extension API. Prefer the extension mechanisms already exercised by the shipped
examples and package code.

## See also

- {doc}`index`
- {doc}`../../usage/custom-models`
- {doc}`../../examples/synthesis`

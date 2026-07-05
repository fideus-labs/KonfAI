# Built-in component catalogue

Almost everything in a KonfAI config is a **component referenced by name**: a
model, a loss, a metric, a transform, an augmentation, a scheduler, or a storage
backend. This section is the catalogue of what ships in the box — **these names
are exactly what you reference in YAML**: copy them into your config verbatim,
with the constructor arguments listed alongside.

```{note}
These pages are generated from a source-level read of `konfai/`: class names,
constructor signatures and defaults are taken directly from the code.
```

## The pages

Start with {doc}`models` (the network you train), then {doc}`losses-metrics`
(what you attach to its named outputs). {doc}`transforms` and
{doc}`augmentations` cover the data pipeline, {doc}`schedulers` the loss-weight
and learning-rate schedules, and {doc}`storage-backends` the on-disk formats.

```{toctree}
:maxdepth: 1

models
losses-metrics
transforms
augmentations
schedulers
storage-backends
```

## How a name is resolved

Every component name in a config is resolved by `konfai.utils.utils.get_module`
in one of two ways:

| Form | Example | Resolves to |
| --- | --- | --- |
| **bare name** | `Dice`, `Standardize`, `Flip` | inside that kind's package (`konfai.metric.measure`, `konfai.data.transform`, `konfai.data.augmentation`, …) |
| **`module:Class`** | `torch:nn:L1Loss`, `monai.losses:DiceLoss`, `Loss:MyWrapper` | *any* importable module — an installed library **or** a local `.py` file next to your config (the current working directory is on `sys.path`) |

So the tables below list the **bare name** for built-ins; you are never limited
to them — any importable class that satisfies the same contract works via the
`module:Class` form. See {doc}`../../concepts/configuration` for the full
resolution rules and {doc}`../../reference/api/extension-points` for how to write
your own.

## How to discover a component's parameters

The tables give the **key** constructor arguments and defaults, but the exact,
always-current parameter set is whatever the class's `__init__` declares — the
reflection engine binds YAML keys directly to constructor parameter names. Two
ways to get the exhaustive list for any component:

1. **Let KonfAI materialise the defaults.** Reference the component in a config
   and run the workflow (or run with `KONFAI_CONFIG_MODE=default`). KonfAI writes
   every resolved default back into the YAML file, giving you a complete,
   fully-expanded subtree to edit. (This is the same
   [config-mutation behaviour](../../concepts/configuration.md) that surprises
   new users — here it is a feature.)
2. **Read the signature.** Bare names map to `konfai/metric/measure.py`,
   `konfai/data/transform.py`, `konfai/data/augmentation.py`,
   `konfai/models/**`, and `konfai/metric/schedulers.py`.

## Next steps

- {doc}`../../concepts/configuration` — how names and `classpath` are resolved
- {doc}`../api/extension-points` — writing your own model / loss / transform

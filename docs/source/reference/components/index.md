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

Most component names in a config are resolved by `konfai.utils.utils.get_module` in
one of two ways. Three kinds do **not** go through it: loss-weight schedulers and
optimizers are looked up directly inside `konfai.metric.schedulers` and `torch.optim`
(so `module:Class` is not accepted for them), and a storage backend is never named at
all — you pick a format token in `dataset_filenames`.

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
2. **Read the signature.** Where a bare name is looked up depends on the kind:

   | Kind | Bare name resolves in |
   | --- | --- |
   | criteria | `konfai/metric/measure.py` |
   | transforms | `konfai/data/transform.py`, then `konfai/data/augmentation.py` |
   | augmentations | `konfai/data/augmentation.py` |
   | models | `konfai/models/python/**` |
   | learning-rate schedulers | `torch.optim.lr_scheduler` **first**, then `konfai/metric/schedulers.py` |
   | loss-weight schedulers | `konfai/metric/schedulers.py` only |
   | patch blending (`patch_combine`) | `konfai/data/patching.py` |
   | prediction reduction | `konfai/predictor.py` |
   | case reduction | `konfai/data/reduction.py` |

   So a bare `StepLR` resolves *outside* KonfAI, in torch.

## Next steps

- {doc}`../../concepts/configuration` — how names and `classpath` are resolved
- {doc}`../api/extension-points` — writing your own model / loss / transform

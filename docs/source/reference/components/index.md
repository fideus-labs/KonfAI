# Built-in component catalogue

Almost everything in a KonfAI config is a **component referenced by name**: a
model, a loss, a metric, a transform, an augmentation, a scheduler, or a storage
backend. This section is the catalogue of what ships in the box — the names you
can drop into a YAML file today, their constructor arguments, and, just as
important, **how mature each one is**.

```{note}
These pages are generated from a source-level audit of `konfai/` (class names,
constructor signatures, and stability are taken from the code, not from
marketing). Where a component is experimental, partially implemented, or crashes
with its default arguments, it is labelled as such — see {doc}`../stability` for
the aggregated maturity matrix.
```

## The pages

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

## Reading the stability column

| Label | Meaning |
| --- | --- |
| **Stable** | Implemented and exercised by the test suite and/or the shipped examples. Safe to rely on. |
| **Usable** | Implemented and functional, but not covered by tests or examples. Works; verify on your data. |
| **Optional-dep** | Stable, but needs an extra installed (`pip install "konfai[…]"`); raises an actionable error otherwise. |
| **Experimental** | Research / undocumented code. May need a very specific config; interfaces can change. |
| **Broken by default** | The default constructor raises or fails — usable only with specific arguments (or not at all yet). Flagged explicitly. |

## See also

- {doc}`../stability` — the aggregated maturity matrix across every kind
- {doc}`../../concepts/configuration` — how names and `classpath` are resolved
- {doc}`../api/extension-points` — writing your own model / loss / transform

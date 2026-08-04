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

For transforms and criteria, an undecorated class reads its constructor parameters
directly from the branch the loader already appended for the classpath, which is
usually the most readable layout. **Models are different**: the model loader appends
the class name when the class has no `_key`, so an undecorated `UNetpp5` still reads
from `Trainer.Model.UNetpp5` — exactly where `@config()` would put it. For a model a
decorator only *renames* the subtree.

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

Inheriting the base class is the way to state a contract, not a formality — but it
is not required. The loader type-checks what it built: a `Transform` is prepared and
returned, an augmentation is handed over as itself, and anything else is wrapped in
`konfai.data.transform.Foreign` (augmentations have their own `Foreign` in
`konfai.data.augmentation`). The wrapper reads the whole volume, leaves geometry
alone, and **checks** the returned shape — a foreign class that changes it raises a
named `TransformError` telling you to subclass `Transform` and implement
`transform_shape()`. Subclass the base when you need to declare a locality, an
inverse, or a shape change; name the foreign class directly when you do not.

## Patch locality

A transform declares how its output at one voxel depends on its input. The
patch-streaming dispatcher (`konfai.data.patching`) reads that declaration and
reads only the source region a target patch needs, instead of materialising the
whole volume.

The safe default is to declare nothing:

- `Transform.patch_locality` returns `WHOLE_VOLUME`
- `DataAugmentation._patch_locality` returns `WHOLE_VOLUME`

A transform that overrides only `__call__` therefore takes the whole-volume path.
The case is loaded, your `__call__` sees the tensor it always would, and patches
are cut from the result. Custom transforms never have to know streaming exists.

To opt in, override `patch_locality(cache_attribute)` and return a
`PatchLocality`. Augmentations override `_patch_locality(index, a,
cache_attribute)`: an augmentation declares per case *and* per copy, because the
halo of a geometric draw is that draw's own.

| Declared kind | Meaning | What you must also implement |
| --- | --- | --- |
| `POINTWISE` | output voxel depends only on the same voxel, across channels | nothing |
| `HALO` | bounded neighbourhood, radius `halo` per axis in array order (Z, Y, X) | nothing — the dispatcher reads the enlarged region and crops |
| `ORIENTATION` | flip or permute | `stream_region_source` |
| `CROP` | source region is the target region translated | `stream_region_source` |
| `GLOBAL_STAT` | needs whole-volume statistics, `stat_keys` a subset of Min/Max/Mean/Std (or their `…PerChannel` forms) | nothing — the dispatcher seeds the statistic from disk |
| `RESCALE` | resample by a ratio | subclass `Resample` |
| `REGRID` | resample onto a grid declared elsewhere — a stored reference, not a ratio — so the source region is computed from the two geometries | subclass `Resample`; declare a halo when a displacement field is composed in |
| `SLAB` | a per-voxel value map plus a side effect that needs the slab's place in the volume | `stream_slab(name, tensor, region, spatial_shape, cache_attribute)`, and optionally `stream_abort`. The **read** dispatcher has no slab context and treats it as `WHOLE_VOLUME`; the gain is on the write side |
| `WHOLE_VOLUME` | needs the whole volume | nothing — this is the default |

A sampler you write yourself should take its arithmetic from
`konfai.data.transform`'s own: `sampling_dtype` (what to accumulate a weighted
sum in — an integer input and a CPU half both need float32), `nearest_index`
(ITK's round-half-up, which `torch.round` and `F.interpolate` each get wrong in
their own way) and `window_index` (a global source index clamped into the
sub-region that was actually read). A sampler that is only *nearly* the same as
the ones shipped here makes every comparison against them a negotiation.

A declaration is bound by three rules:

- **read-only** — never write to `cache_attribute`. A declaration is made once for
  the whole case, so anything it wrote would be one patch's answer imposed on
  every other. The dispatcher hands over a private copy, so a write is contained
  and silently lost.
- **no I/O** — read the attribute in hand, nothing else. Whether the outside world
  can honour the declaration is the dispatcher's call.
- **total** — answer for any case, including one with no metadata. A missing key
  must return `WHOLE_VOLUME`, never raise. The config-time checks probe with an
  empty `Attribute`.

`ORIENTATION` and `CROP` are the kinds that need the remap, and declaring one
without it is a loud failure rather than a wrong answer:
`Transform.stream_region_source` raises `TransformError` and
`DataAugmentation._stream_region_source` raises `AugmentationError`. A `HALO`
never calls it — the dispatcher derives the enlarged region from the radius.

Any region kind that rewrites geometry must also implement
`write_stream_cache_attribute(cache_attribute, source_spatial_shape)`. A region
stage runs on the region, so the `Origin`, `Spacing` or `Direction` its `__call__`
records describe that region rather than the case; those writes land on a throwaway
`Attribute` and are dropped. `write_stream_cache_attribute` is called once per
case, on the persistent attribute, with the full source spatial shape: write the
case-level geometry there. Omitting it is **refused**, not silent: a region stage that records geometry on the
throwaway scope and implements no `write_stream_cache_attribute()` raises a
`PatchError` naming the keys it recorded. `Canonical` is an `ORIENTATION`
transform and implements it: its new origin is the corner the volume mirrors onto,
which only the full extent gives. The base is a no-op, for a transform that leaves
geometry alone.

The rule stops at the region stage. A pointwise stage needs no extra method: a key
it adds reaches the case on its own, which is how `TensorCast` keeps the source
dtype its `inverse()` reads.

What a declaration costs you:

- A streamed patch must equal what the whole-volume path produces on the same
  grid. This is what the declaration promises and what the test suite checks for
  every built-in.
- A halo is paid on every side of every patch, so streaming reads
  `prod(1 + 2 * halo_k / patch_k)` times the case's bytes. A radius above half the
  patch — or half the case, whichever is smaller — on any axis is rejected, and
  the case falls back to a full load.
- A chain streams when every stage is pointwise or a region kind, where
  `GLOBAL_STAT` counts as pointwise. Region stages compose — each pulls through
  the one before it — so their number is not limited; any `WHOLE_VOLUME` falls
  back.

`patch_transforms` is stricter than `transforms`: only `POINTWISE` and
`GLOBAL_STAT` are accepted, and any other kind raises a `ConfigError` pointing at
`transforms` instead.

## `preserves_statistics`

`PatchLocality.preserves_statistics` overrides the kind's own answer to "does this
leave every whole-volume statistic of its input untouched". Only `ORIENTATION`
says yes by default: a flip or a permute is a bijection on the voxels, so the
multiset of values — and therefore Min/Max/Mean/Std — is exactly the input's.

It exists because it decides whether a later `GLOBAL_STAT` stage may seed from the
stored volume's statistics. `[Canonical(), Normalize()]` streams.
`[Clip(-200., 400.), Normalize()]` falls back, because the clip moves the
statistics the normalise would then read.

One built-in overrides it: `TensorCast` declares `POINTWISE` and preserves the
statistics only for a target that holds every value a volume is read as —
`float32` and `float64`. A later `Standardize` may then still seed from disk. A
half cast is not one of them: `float16` runs out of mantissa at 2048, where a CT
reaches 3000.

Declaring `preserves_statistics=True` on a transform that is not a bijection is a
silent-correctness bug, not an error. Nothing validates the claim against what
your `__call__` does. The chain streams, every patch is seeded with a statistic
taken before your transform ran, and the result quietly disagrees with the
whole-volume answer. Set it only when your transform permutes voxels or maps
values one-to-one.

## Transforms from another framework

**Name it directly.** A class that is not a `Transform` is wrapped in `Foreign`, so
the short form works:

```yaml
transforms:
  monai.transforms:ScaleIntensity:
    minv: 0.0
    maxv: 1.0
```

`Foreign` reads the whole volume — what a class saying nothing about where its
output comes from is owed — leaves geometry as it stands, and verifies that the
returned shape matches the input. A class that resamples, crops or reorients owns
both shape and geometry, and it will be refused with a `TransformError` pointing at
`transform_shape()`.

Type defaults are not a problem either: the config writer records a `type` value as
its `__name__`, so `ScaleIntensity`'s `dtype=np.float32` is written back as
`float32` rather than failing to serialise.

**Write a wrapper when you need more than that** — a locality declaration, an
inverse, a shape change, or to keep a large foreign signature out of the config
file. The wrapper's own signature is what YAML binds:

```python
# MonaiTransform.py
import torch
from monai.transforms import ScaleIntensity

from konfai.data.transform import Transform
from konfai.utils.dataset import Attribute


class MonaiScaleIntensity(Transform):
    """Adapt a MONAI array transform to the KonfAI transform contract."""

    def __init__(self, minv: float = 0.0, maxv: float = 1.0) -> None:
        super().__init__()
        # Expose only the arguments YAML should bind, and pass them on.
        self.transform = ScaleIntensity(minv=minv, maxv=maxv)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return self.transform(tensor)
```

Referenced as `MonaiTransform:MonaiScaleIntensity`. KonfAI tensors are
channel-first `[C, (Z), Y, X]`.

Both routes are safe by default: neither declares a locality, so both take the
whole-volume path and see exactly the tensor the foreign transform expects. Add a
`patch_locality` only once you can state which kind is honest for it.

Two traps:

- **A random per-voxel augmentation is not `POINTWISE`.** The kind is about the
  voxel's position, not the arithmetic's shape. A field drawn per call is a
  different field on every call, so overlapping patches sample unrelated fields
  and the overlap blend suppresses the variance the augmentation exists to add.
  The built-in `Noise` declares `WHOLE_VOLUME` for this reason. Declared
  `POINTWISE`, two reads of the same patch return different values.
- **Wrap the array transform, not the dict one.** A MONAI `*d` transform takes a
  dict and pairs image and label through its `keys`. `__call__` is handed one
  tensor and returns one tensor, so there is no dict for `keys` to select from.
  Let the group configuration do the pairing.

A transform reads another group through `self.datasets`, by group name — the
built-in `Mask` does, and so can yours. Inside `__call__` it reads that group whole: `__call__` is not told where its tensor
sits in the volume, so it cannot ask for the matching region — which is why a wrapper
that reaches for a second group from `__call__` alone should declare nothing and take
the whole-volume path. `Mask` itself goes one step further: it declares `SLAB` and
implements `stream_slab`, so on the write path it reads only the slab's rows of the
mask. That is the kind to reach for when your second-group read *can* be located.

## Criteria and schedulers

KonfAI lets you attach multiple losses and metrics to multiple outputs and
targets. The relevant extension points live in:

- `konfai.metric.measure`
- `konfai.metric.schedulers`
- `konfai.network.network.TargetCriterionsLoader`

This is the mechanism used by the examples to define reconstruction losses,
Dice-based evaluation, adversarial losses, and scheduled weights.

Runtime contracts — each base class names exactly what you must implement:

| Base class | You must implement | Also worth setting |
| --- | --- | --- |
| `konfai.metric.measure.Criterion` | `forward(output, *targets) -> Tensor` | `maximize` (higher-is-better, drives ranking) and `reducible` (whether streamed evaluation may accumulate it) — both default `False` |
| `CriterionWithInit` | `forward`, plus `init(model, output_group, target_group) -> str` | — |
| `CriterionWithAttribute` | `forward(output, *targets, attributes: list[list[Attribute]])` — `attributes` is **keyword-only** | — |
| `konfai.data.transform.Transform` | `__call__(name, tensor, cache_attribute)` | `transform_shape()` if the shape changes, `patch_locality()` to stream |
| `konfai.data.transform.TransformInverse` | the above, plus `inverse(name, tensor, cache_attribute)` | — |
| `konfai.data.augmentation.DataAugmentation` | three: `_state_init(index, shapes, caches_attribute)`, `_compute(name, index, a, tensor)`, `_inverse(index, a, tensor)` | — |

## Quick contract table

| Extension point | Recommended base class | Typical YAML entry point |
| --- | --- | --- |
| Custom model | `konfai.network.network.Network` — recommended, not required: a plain `torch.nn.Module` is wrapped in `MinimalModel` automatically | `Trainer.Model.classpath` |
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

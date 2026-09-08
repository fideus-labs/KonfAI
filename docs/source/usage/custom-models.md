# Extending KonfAI

When the built-in components are not enough, you extend KonfAI by writing
regular Python classes and selecting them from YAML: no core edits, no plugin
registry, no manifest. A custom object is selected through `classpath` and
instantiated through `apply_config()`, so it must satisfy two contracts:

- the **configuration contract**: constructor argument names and YAML keys must match
- the **runtime contract**: the class must inherit the right KonfAI base class
  and implement the expected methods

This page gives both contracts for every kind you can add: models, transforms,
augmentations, losses and metrics, storage backends and reduction operators,
with a minimal example of each. The most practical way to ship one is a Python
module next to your configuration files, referenced through `classpath`
(`examples/Synthesis` does it with `Model.py` and `UnNormalize.py`).

**A local classpath such as `Model:UNetpp5` imports a Python file resolved
from the directory you launch `konfai` in**: keep the custom `.py` next to
the YAML and run the command from that directory.

## General rules

- Keep project-specific Python files next to the YAML when possible.
- Use explicit local classpaths such as `Model:UNetpp5` or `MyLoss:BoundaryDice`.
- Keep constructor argument names in `snake_case`.
- Do not use `@config()` on a local transform or criterion: it inserts an
  extra YAML subtree named after the class and makes the config harder to
  read and generate. A model is nested under its class name either way (see
  below).
- Use `@config("...")` only when you intentionally want a fixed explicit
  subtree name.
- Prefer concrete defaults for nested custom objects when they define the
  natural baseline behavior of the class. This makes the default wiring visible
  to KonfAI and helps YAML generation stay explicit.
- Start from a shipped example and change one layer at a time.

For example, this pattern is usually a good fit:

```python
class Gan(network.Network):
    def __init__(
        self,
        generator: UNetpp5 = UNetpp5(),
        discriminator: Discriminator = Discriminator(),
    ) -> None:
        super().__init__()
        self.add_module("Generator", generator)
        self.add_module("Discriminator", discriminator)
```

This style is often preferable in KonfAI because the constructor already shows
the effective default model or loss stack that will appear in YAML.

Use `None` defaults only when the nested object must be created dynamically,
depends on runtime information, or would be too expensive or stateful to
instantiate eagerly.

## How `classpath` and `@config(...)` interact

`classpath` selects the Python implementation:

```yaml
Trainer:
  Model:
    classpath: Model:UNetpp5
```

Where the class then reads its constructor arguments depends on its kind.
A transform or a criterion loaded through `classpath` reads them directly from
the branch the loader already appended for it (`transforms: {MyTransforms:Clamp01: {...}}`),
which is the most readable layout, and `@config()` would add an extra subtree
named after the class. **Models are different**: the model loader appends the
class name when the class has no `_key`, so an undecorated `UNetpp5` reads
from `Trainer.Model.UNetpp5`, exactly where `@config()` would put it. For a
model a decorator only *renames* that subtree:

```yaml
Trainer:
  Model:
    classpath: Model:UNetpp5
    UNetpp5:
      ...                 # the model's own arguments, decorated or not
      outputs_criterions:
        ...
```

In the current codebase:

- `@config("SomeKey")` binds the class to the `SomeKey` subtree
- `@config()` defaults to the class name

Use `@config("SomeKey")` only when you intentionally want that nesting.

## Contract summary

| Extension point | Recommended base class | Required methods | Typical YAML entry point |
| --- | --- | --- | --- |
| Model | `konfai.network.network.Network` (recommended, not required: a plain `torch.nn.Module` is wrapped in `MinimalModel` automatically) | `__init__`, then build the graph with `add_module(...)` | `Trainer.Model.classpath` |
| Transform | `konfai.data.transform.Transform` or `TransformInverse` | `__call__`, `transform_shape` if the shape changes, `inverse` for `TransformInverse` | `groups_dest.<group>.transforms` |
| Augmentation | `konfai.data.augmentation.DataAugmentation` | `_state_init`, `_compute`, `_inverse` | `Dataset.augmentations.*.data_augmentations` |
| Loss / metric | `konfai.metric.measure.Criterion`, `CriterionWithInit`, or `CriterionWithAttribute` | `forward`, plus `init` when using `CriterionWithInit` | `outputs_criterions.*.targets_criterions.*.criterions_loader` |
| Storage backend | `konfai.utils.dataset.AbstractFile` (plus a `BACKENDS` entry) | see [A storage backend](#a-storage-backend) | the `:format` token in a group's `path` |
| Reduction operator | `konfai.data.reduction.Reduction` | `__call__(list[Tensor]) -> Tensor` | `reduction` (predictor ensemble/TTA, TRANSFORM `Reduce`) |

## Custom models

For a real KonfAI model, inherit from
`konfai.network.network.Network`.

This is the right choice when you need:

- a named graph built with `add_module(...)`
- patch-aware inference
- multiple outputs
- `outputs_criterions`
- full compatibility with KonfAI training, prediction, and evaluation workflows

Minimal example:

```python
import torch

from konfai.data.patching import ModelPatch
from konfai.network import blocks, network
class MySegNet(network.Network):
    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {
            "Argmax": network.TargetCriterionsLoader()
        },
        patch: ModelPatch | None = None,
        dim: int = 2,
    ) -> None:
        super().__init__(
            in_channels=1,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            patch=patch,
            dim=dim,
        )
        self.add_module("Backbone", torch.nn.Conv2d(1, 8, kernel_size=3, padding=1))
        self.add_module("Head", torch.nn.Conv2d(8, 2, kernel_size=1))
        self.add_module("Argmax", blocks.ArgMax(dim=1))
```

Matching YAML:

```yaml
Trainer:
  Model:
    classpath: Model:MySegNet
    MySegNet:                     # the model's own subtree, decorated or not
      dim: 2
      outputs_criterions:
        Argmax:
          targets_criterions:
            SEG:
              criterions_loader:
                Dice:
                  is_loss: true
                  group: 0
                  schedulers:
                    Constant:
                      nb_step: 0
                      value: 1
```

### Model contract details

- Call `super().__init__(...)` so KonfAI can attach the optimizer, scheduler,
  patching, and criterion machinery.
- Build the graph with `self.add_module(...)`, not only with raw PyTorch
  attributes.
- The keys in `outputs_criterions` must match the actual model output path,
  built from `add_module(...)` names.
- The output path is a graph name, not a dataset name. For example, in the
  shipped `examples/Segmentation/Config.yml` the cross-entropy loss is attached to
  `UNetBlock_0:Head:Conv` (the pre-softmax logits) while a Dice loss is attached to
  `UNetBlock_0:Head:Softmax` on the same head, each criterion reads the output whose
  form it expects.

KonfAI can wrap a simpler module internally in some situations, but if you want
reliable custom behavior, inheriting from `Network` is the supported path.

## Custom transforms

Use `konfai.data.transform.Transform` for one-way transforms and
`TransformInverse` when KonfAI must be able to invert the operation later.

The contract is tiered, and tier 0 is all a correct transform owes:

- **Tier 0, correct**: `__call__(name, tensor, cache_attribute)`, plus
  `transform_shape(...)` if the transform changes the spatial shape (and
  `inverse(...)` if you inherit from `TransformInverse`). The stage then runs
  on the whole volume and nothing silently breaks.
- **Tier 1, streaming**: set the `locality` class attribute (a `LocalityKind`;
  plus `halo` for a bounded neighbourhood) and the stage streams.
- **Tier 2, streaming-aware**: method overrides, only where the answer depends
  on the case or the stage owns a region's geometry or reads beside it. See
  [Patch locality](#patch-locality) below.

`cache_attribute` is where you should save anything needed later by the inverse
transform.

Minimal invertible transform:

```python
import torch

from konfai.data.transform import TransformInverse
from konfai.utils.dataset import Attribute

class Clamp01(TransformInverse):
    def __init__(self, inverse: bool = False) -> None:
        super().__init__(inverse)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        cache_attribute["original_min"] = tensor.min()
        cache_attribute["original_max"] = tensor.max()
        return tensor.clamp(0, 1)

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor
```

Matching YAML:

```yaml
groups_dest:
  CT:
    transforms:
      MyTransforms:Clamp01:
        inverse: false
```

### Transform contract details

- The transform receives one tensor at a time.
- `name` is the case identifier.
- `cache_attribute` stores per-case metadata and is the right place for values
  needed by `inverse(...)`.
- `self.datasets` is populated by KonfAI and can be used when the transform
  needs to read another group, as built-in transforms such as `Clip` and
  `Standardize` do.

### A class that is not a `Transform`

Inheriting the base class is the way to state a contract, not a formality, but it
is not required. The loader type-checks what it built: a `Transform` is prepared and
returned, an augmentation is handed over as itself, and anything else is wrapped in
`konfai.data.transform.Foreign` (augmentations have their own `Foreign` in
`konfai.data.augmentation`). The wrapper reads the whole volume, leaves geometry
alone, and **checks** the returned shape: a foreign class that changes it raises a
named `TransformError` telling you to subclass `Transform` and implement
`transform_shape()`. Subclass the base when you need to declare a locality, an
inverse, or a shape change; name the foreign class directly when you do not.

### Patch locality

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
That is the whole tier-0 contract: `__call__`, plus `transform_shape()` when the
spatial shape changes.

To opt in to streaming (tier 1), set the `locality` class attribute to a
`LocalityKind` (plus the `halo` attribute for a bounded neighbourhood):

```python
class Threshold(Transform):
    locality = LocalityKind.POINTWISE

    def __call__(self, name, tensor, cache_attribute):
        return (tensor > 0.5).to(tensor.dtype)
```

The base `patch_locality` answers from the attribute. Override the method
`patch_locality(cache_attribute)` itself (tier 2) only when the answer depends
on the case (read off the header) or carries `stat_keys` or a `reason`; the
other tier-2 methods (`stream_region_source`, `stream_region`,
`plan_region_reads`, `stream_slab`, `write_stream_cache_attribute`) are owed
only where the table below says so. Augmentations declare the same way
(`_patch_locality(index, a, cache_attribute)` as the method form): an
augmentation declares per case *and* per copy, because the halo of a geometric
draw is that draw's own.

| Declared kind | Meaning | What you must also implement |
| --- | --- | --- |
| `POINTWISE` | output voxel depends only on the same voxel, across channels | nothing |
| `HALO` | bounded neighbourhood, radius `halo` per axis in array order (Z, Y, X) | nothing: the dispatcher reads the enlarged region and crops |
| `ORIENTATION` | flip or permute | `stream_region_source` |
| `CROP` | source region is the target region translated | `stream_region_source` |
| `GLOBAL_STAT` | needs whole-volume statistics, `stat_keys` a subset of Min/Max/Mean/Std (or their `…PerChannel` forms) | nothing: the dispatcher seeds the statistic from disk |
| `REGRID` | resample onto a grid declared elsewhere: a stored reference, not a ratio, so the source region is computed from the two geometries | subclass `Resample`; declare a halo when a displacement field is composed in |
| `SLAB` | a per-voxel value map plus a side effect that needs the slab's place in the volume | `stream_slab(name, tensor, region, spatial_shape, cache_attribute)`, and optionally `stream_abort`. The **read** dispatcher has no slab context and treats it as `WHOLE_VOLUME`; the gain is on the write side |
| `WHOLE_VOLUME` | needs the whole volume | nothing: this is the default |

A sampler you write yourself should take its arithmetic from
`konfai.data.transform`'s own: `sampling_dtype` (what to accumulate a weighted
sum in: an integer input and a CPU half both need float32), `nearest_index`
(ITK's round-half-up, which `torch.round` and `F.interpolate` each get wrong in
their own way) and `window_index` (a global source index clamped into the
sub-region that was actually read). A sampler that is only *nearly* the same as
the ones shipped here makes every comparison against them a negotiation.

A declaration is bound by three rules:

- **read-only**: never write to `cache_attribute`. A declaration is made once for
  the whole case, so anything it wrote would be one patch's answer imposed on
  every other. The dispatcher hands over a private copy, so a write is contained
  and silently lost.
- **no I/O**: read the attribute in hand, nothing else. Whether the outside world
  can honour the declaration is the dispatcher's call.
- **total**: answer for any case, including one with no metadata. A missing key
  must return `WHOLE_VOLUME`, never raise. The config-time checks probe with an
  empty `Attribute`.

`ORIENTATION` and `CROP` are the kinds that need the remap, and declaring one
without it is a loud failure rather than a wrong answer:
`Transform.stream_region_source` raises `TransformError` and
`DataAugmentation._stream_region_source` raises `AugmentationError`. A `HALO`
never calls it: the dispatcher derives the enlarged region from the radius.

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

A stage that reads a second volume beside its region (`Mask` reads its mask where
the region sits) overrides `stream_region(name, tensor, context, cache_attribute)`:
`context` says which part of the input the tensor covers and which part of the
output is due. It may also override `plan_region_reads(name, contexts)`: it is
called once, before a case's first region, with the contexts `stream_region` will
then be handed in that order, and the stage declares the windows it will read to
the dataset holding them (`Dataset.plan_region_reads`), so a store that caches
decoded chunks evicts by next use rather than by recency. A sweep declares its
blocks where their decomposition is fixed (a cubic block, or no budget to grow
under); a sweep whose regions grow with what they hold declares nothing, since
its second region already deviates from any order declared at the first. The
patch route declares the case's patches in the DataLoader's own order, on the
process that reads them. A hint: neither what is read nor its values depend on
it.

What a declaration costs you:

- A streamed patch must equal what the whole-volume path produces on the same
  grid. This is what the declaration promises and what the test suite checks for
  every built-in.
- A halo is paid on every side of every patch, so streaming reads
  `prod(1 + 2 * halo_k / patch_k)` times the case's bytes. A radius above half the
  patch (or half the case, whichever is smaller), on any axis is rejected, and
  the case falls back to a full load.
- A chain streams when every stage is pointwise or a region kind, where
  `GLOBAL_STAT` counts as pointwise. Region stages compose (each pulls through
  the one before it), so their number is not limited; any `WHOLE_VOLUME` falls
  back.

`patch_transforms` is stricter than `transforms`: only `POINTWISE` and
`GLOBAL_STAT` are accepted, and any other kind raises a `ConfigError` pointing at
`transforms` instead.

### `preserves_statistics`

`PatchLocality.preserves_statistics` overrides the kind's own answer to "does this
leave every whole-volume statistic of its input untouched". Only `ORIENTATION`
says yes by default: a flip or a permute is a bijection on the voxels, so the
multiset of values (and therefore Min/Max/Mean/Std) is exactly the input's.

It exists because it decides whether a later `GLOBAL_STAT` stage may seed from the
stored volume's statistics. `[Canonical(), Normalize()]` streams.
`[Clip(-200., 400.), Normalize()]` falls back, because the clip moves the
statistics the normalise would then read.

One built-in overrides it: `TensorCast` declares `POINTWISE` and preserves the
statistics only for a target that holds every value a volume is read as: `float32` and `float64`. A later `Standardize` may then still seed from disk. A
half cast is not one of them: `float16` runs out of mantissa at 2048, where a CT
reaches 3000.

Declaring `preserves_statistics=True` on a transform that is not a bijection is a
silent-correctness bug, not an error. Nothing validates the claim against what
your `__call__` does. The chain streams, every patch is seeded with a statistic
taken before your transform ran, and the result quietly disagrees with the
whole-volume answer. Set it only when your transform permutes voxels or maps
values one-to-one.

### Transforms from another framework

**Name it directly.** A class that is not a `Transform` is wrapped in `Foreign`, so
the short form works:

```yaml
transforms:
  monai.transforms:ScaleIntensity:
    minv: 0.0
    maxv: 1.0
```

`Foreign` reads the whole volume (what a class saying nothing about where its
output comes from is owed) leaves geometry as it stands, and verifies that the
returned shape matches the input. A class that resamples, crops or reorients owns
both shape and geometry, and it will be refused with a `TransformError` pointing at
`transform_shape()`.

Type defaults are not a problem either: the config writer records a `type` value as
its `__name__`, so `ScaleIntensity`'s `dtype=np.float32` is written back as
`float32` rather than failing to serialise.

**Write a wrapper when you need more than that**: a locality declaration, an
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

- **A random per-voxel augmentation is `POINTWISE` only if its draw is a
  function of the voxel's position.** The built-in `Noise` hashes a per-copy
  seed with the absolute voxel position, so two reads of the same patch and
  two overlapping patches see the same field. A field drawn per call is a
  different field on every call: overlapping patches sample unrelated fields
  and the overlap blend suppresses the variance the augmentation exists to add.
  Such a draw declares `WHOLE_VOLUME`.
- **Wrap the array transform, not the dict one.** A MONAI `*d` transform takes a
  dict and pairs image and label through its `keys`. `__call__` is handed one
  tensor and returns one tensor, so there is no dict for `keys` to select from.
  Let the group configuration do the pairing.

A transform reads another group through `self.datasets`, by group name: the
built-in `Mask` does, and so can yours. Inside `__call__` it reads that group whole: `__call__` is not told where its tensor
sits in the volume, so it cannot ask for the matching region, which is why a wrapper
that reaches for a second group from `__call__` alone should declare nothing and take
the whole-volume path. `Mask` itself goes one step further: it declares `SLAB` and
implements `stream_slab`, so on the write path it reads only the slab's rows of the
mask. That is the kind to reach for when your second-group read *can* be located.

## Custom augmentations

Use `konfai.data.augmentation.DataAugmentation` for training-time
augmentations and test-time augmentation building blocks.

The augmentation contract has three stages:

- `_state_init(...)` samples randomness once and can update the expected shapes
- `_compute(...)` applies the augmentation to the selected tensors
- `_inverse(...)` inverts the augmentation when needed

Minimal example:

```python
import torch

from konfai.data.augmentation import DataAugmentation
from konfai.utils.dataset import Attribute

class AddNoise(DataAugmentation):
    def __init__(self, sigma: float = 0.1, groups: list[str] | None = None) -> None:
        super().__init__(groups=groups)
        self.sigma = sigma

    def _state_init(
        self,
        index: int,
        shapes: list[list[int]],
        caches_attribute: list[Attribute],
    ) -> list[list[int]]:
        return shapes

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor.float()) * self.sigma

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return tensor
```

Matching YAML:

```yaml
augmentations:
  DataAugmentation_0:
    data_augmentations:
      MyAugmentations:AddNoise:
        sigma: 0.1
        prob: 0.5
    nb: 1
```

### Augmentation contract details

- `_compute(name, index, a, tensor)` receives **one tensor at a time**: `a`
  identifies the augmented copy within the case; KonfAI applies it across the
  group in `__call__`, so each call returns a single tensor.
- `prob` lives in the same YAML branch as the augmentation-specific parameters.
- `_state_init(...)` is the right place to sample random state that must stay
  consistent across groups.
- Implement `_inverse(...)` even for a no-op, because KonfAI may call it during
  inverse augmentation workflows.

## Custom losses and metrics

For custom criteria, use one of these base classes:

- `Criterion` for the common case
- `CriterionWithInit` when the criterion needs access to the model graph before
  training starts
- `CriterionWithAttribute` when the criterion needs per-sample attributes

Minimal loss:

```python
import torch

from konfai.metric.measure import Criterion

class BoundaryMAE(Criterion):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        return self.weight * torch.nn.functional.l1_loss(output.float(), targets[0].float())
```

Matching YAML:

```yaml
outputs_criterions:
  Argmax:
    targets_criterions:
      SEG:
        criterions_loader:
          MyLosses:BoundaryMAE:
            is_loss: true
            group: 0
            schedulers:
              Constant:
                nb_step: 0
                value: 1
            weight: 1.0
```

### Loss contract details

- `forward(output, *targets)` is the standard signature.
- `CriterionWithAttribute` uses
  `forward(output, *targets, attributes=...)`.
- `CriterionWithInit` adds `init(model, output_group, target_group)`.
- A criterion can return either a tensor or a tuple such as
  `(loss_tensor, scalar_value_for_logging)`.
- The YAML branch lives under
  `outputs_criterions -> <output_group> -> targets_criterions -> <target_group> -> criterions_loader`.
- The same YAML branch can contain both KonfAI runtime fields such as
  `is_loss`, `group`, and `schedulers`, and the constructor arguments of the
  criterion itself. Each object reads only the keys it needs.

What each base class asks of you, and what is worth setting beside it:

| Base class | You must implement | Also worth setting |
| --- | --- | --- |
| `konfai.metric.measure.Criterion` | `forward(output, *targets) -> Tensor` | `maximize` (higher-is-better, drives ranking) and `reducible` (whether streamed evaluation may accumulate it): both default `False` |
| `CriterionWithInit` | `forward`, plus `init(model, output_group, target_group) -> str` | |
| `CriterionWithAttribute` | `forward(output, *targets, attributes: list[list[Attribute]])`: `attributes` is **keyword-only** | |

The loss-weight schedulers a criterion's `schedulers:` block names resolve
against `konfai.metric.schedulers` only: see {doc}`../reference/components/losses-metrics`.

## A storage backend

An imaging format is one class plus one registry entry. Subclass
`konfai.utils.dataset.AbstractFile`, declare the backend's facts as class
attributes, and register the format token in
`konfai.utils.dataset.BACKENDS`; `backend_for(file_format)` then dispatches to
it and nothing else needs a format branch. A token that is also a file suffix
(like `h5`) additionally belongs in `SUPPORTED_EXTENSIONS`
(`konfai.utils.utils`); a token no file on disk ever carries (`:itktransform`
writes `<group>.h5`) goes in `SUPPORTED_BACKEND_FORMATS` instead, because only
extensions are probed on disk.

```python
# chunked_backend.py
from konfai.utils.dataset import AbstractFile, Attribute, BACKENDS
from konfai.utils.errors import DatasetManagerError

class ChunkedFile(AbstractFile):
    """One case per store, decoded in blocks."""

    single_store = False          # True: one store holds every case (like one .h5 file)
    concurrent_write_safe = False # entries share handles/metadata, so writes stay serial
    case_file_suffix = None      # what a case carries implicitly on disk (H5File: ".h5")
    reads_remote = False          # True: the backend opens URI roots (OME-Zarr does)
    writes_pyramid = False        # True: a written store can hold multiscale levels
    lists_case_entries = False    # True: a case is a directory the backend enumerates

    def __init__(self, filename: str, read: bool) -> None:
        try:
            import mychunklib  # noqa: F401
        except ImportError as e:
            raise DatasetManagerError(
                "mychunklib is required to read '.blk' stores.",
                "Install it with: pip install mychunklib",
            ) from e
        ...

    def __enter__(self): ...
    def __exit__(self, exc_type, value, traceback): ...
    def file_to_data(self, group, name): ...                  # whole entry + Attribute
    def file_to_data_slice(self, group, name, slices): ...    # one region
    def data_to_file(self, name, data, attributes=None): ...  # whole write

    def read_granularity(self, name):
        # The stored block a region read is served in, as a C[Z]YX shape.
        return (1, 64, 64, 64)

BACKENDS["blk"] = ChunkedFile
```

Two of the contract's methods matter more than they look:

- **`read_granularity(name)`**: the block a region read is actually served in.
  A chunked store decodes whole blocks, so a window costs the block-aligned
  hull covering it; a memory-mapped one is served band by band (`SitkFile`
  answers `(1, 1, Y, X)` for a `.mha`: one step along the outermost axis a
  window spans, every axis below it whole, because those are the pages the
  read touches). The streaming sweep is priced and cut on this grid, so the
  grain need not be isotropic and need not come from a compressor. A backend
  that stays silent (`None`) is priced at what its reads ask for, which is
  right only when a read costs exactly that.
- **`bounded_region_reads(name)`**: whether a region read decodes only the
  region. The base answers `False`, which is the safe direction: a wrong
  `False` costs speed (the plan prefers one ordered whole read), never
  correctness.

Import-guard the heavy library and raise a `DatasetManagerError` naming the
install, at the point of use, never a bare `ImportError` at import time: a bare
install must still import `konfai.utils.dataset`. Declare
`can_stream(file_format, attributes)` `True` (and implement
`open_data_stream`) only when the backend serves incremental region writes;
the default routes writes whole through `data_to_file`.

## A reduction operator

A reduction folds N tensors into one, and one vocabulary serves two engines:
the predictor folds one case's copies (ensemble, TTA), and the TRANSFORM
`Reduce` stage folds N cases into one. Subclass `konfai.data.reduction.Reduction`
and reference it by classpath wherever a `reduction` is configured.

```python
# my_reduction.py
import torch

from konfai.data.reduction import Reduction

class TrimmedMean(Reduction):
    """Mean of the members left after dropping each voxel's min and max."""

    voxel_local = True     # every output voxel reads only the SAME voxel of each member
    incremental = False    # __call__ needs all members at once
    working_multiple = 4.0 # the stacked float copy plus the reduction buffers

    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        if len(tensors) < 3:
            raise ValueError("TrimmedMean needs at least three members")
        stack = torch.stack([tensor.float() for tensor in tensors])
        trimmed = stack.sum(dim=0) - stack.amax(dim=0) - stack.amin(dim=0)
        return trimmed / (len(tensors) - 2)
```

The list is the fold axis: one tensor per member (a model of an ensemble, a
case of a cohort), each in the `[1, K, C, *spatial]` layout both engines hand
over. The declarations:

- **`voxel_local`**: declare `True` only if every output voxel depends on the
  same voxel of each input. **The streamed gates trust this flag and check
  nothing else: a wrong `True` corrupts a streamed output** (each region is
  reduced with its own members only), while a wrong `False` merely costs the
  whole-volume path. The TRANSFORM `Reduce` stage refuses a non-`voxel_local`
  operator outright.
- **`incremental`**: `True` when the operator can fold members one at a time;
  then override the `start` / `accumulate` / `finalize` protocol and the
  working set stays two regions whatever N is. `Mean` and `Std` do; `Median`
  cannot.
- **`working_multiple`** and **`working_multiple_for(cases)`**: the
  buffers-worth the operator allocates on top of what it is handed; the plan
  multiplies it into the peak it sizes regions against. The attribute is the
  worst case; an operator whose route depends on the member count overrides
  the method (`Median` selects the middle through element-wise min/max
  networks up to five members and sorts the stack past that, so it answers
  per count).
- **`output_channels(channels, cases)`**: override when the fold changes the
  channel count (`Concat` returns `channels * cases`; the default returns
  `channels`).

## Common failure modes

- `classpath` imports the wrong file or class.
- Constructor argument names do not match YAML keys.
- The object inherits from the wrong base class.
- `outputs_criterions` points to a graph name that does not exist in the
  model.
- `@config()` inserted an extra class-name subtree, but the YAML was edited at
  the parent level.
- `@config("...")` points to a different subtree than the one you edited in
  YAML.

## Two things that are not extension points

At a higher level, an entire workflow can be packaged as a KonfAI App: the
preferred path when a workflow is mature and should be reused through a stable
interface. See {doc}`apps`.

KonfAI is highly configurable, but not every internal helper is a stable
public extension API. Prefer the mechanisms on this page, which the shipped
examples and the package code exercise.

## Next steps

- {doc}`../config_guide/index`: how `classpath` and `apply_config()` bind YAML
  keys to constructor arguments.
- {doc}`../reference/components/models`: how `add_module(...)` names become
  the output paths that `outputs_criterions` points at.
- {doc}`large-images`: what a locality declaration buys, and how close a
  streamed patch is to a whole-volume one.
- {doc}`../reference/api/index`: the base classes' full signatures.

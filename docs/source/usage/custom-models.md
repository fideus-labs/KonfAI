# Extending KonfAI

When the built-in components are not enough, write a Python class and name it in the YAML. There is no
registry and no core edit. A class works when two things hold:

- its constructor arguments have the names of the YAML keys that configure it;
- it subclasses the right KonfAI base class and implements its methods.

Put the `.py` file next to the YAML and run `konfai` from that directory: a classpath such as
`Model:UNetpp5` imports `Model.py` from the working directory (`examples/Synthesis` does this with
`Model.py` and `UnNormalize.py`).

| To add | Subclass | Implement | Named in the YAML under |
| --- | --- | --- | --- |
| a model | `konfai.network.network.Network` | `__init__`, building the graph with `add_module` | `Model.classpath` |
| a transform | `konfai.data.transform.Transform` (or `TransformInverse`) | `__call__`, `transform_shape` if the shape changes, `inverse` for `TransformInverse` | a group's `transforms` |
| an augmentation | `konfai.data.augmentation.DataAugmentation` | `_state_init`, `_compute`, `_inverse` | `augmentations.*.data_augmentations` |
| a loss or metric | `konfai.metric.measure.Criterion` | `forward` | `outputs_criterions.*.targets_criterions.*.criterions_loader` |
| a storage format | `konfai.utils.dataset.AbstractFile` | reads and writes, plus a `BACKENDS` entry | the `:format` of a dataset path |
| a reduction | `konfai.data.reduction.Reduction` | `__call__(list[Tensor]) -> Tensor` | `combine` (ensembles, TTA), TRANSFORM's `Reduce` |

Give nested objects a concrete default when there is a natural one: the constructor then shows the default
setup, and KonfAI writes it into the resolved config.

```python
class Gan(network.Network):
    def __init__(self, generator: UNetpp5 = UNetpp5(), discriminator: Discriminator = Discriminator()) -> None:
        super().__init__()
        self.add_module("Generator", generator)
        self.add_module("Discriminator", discriminator)
```

## Where a class reads its arguments

A transform or a criterion reads its arguments right under its own name:

```yaml
transforms:
  MyTransforms:Clamp01:
    inverse: false
```

A model reads them under its class name, below `Model`:

```yaml
Trainer:
  Model:
    classpath: Model:UNetpp5
    UNetpp5:
      ...
```

`@config("SomeKey")` on a class makes it read from `SomeKey` instead. Do not put `@config()` on a local
transform or criterion: it adds a level named after the class.

## A model

Subclass `Network`, call `super().__init__(...)`, and build the graph with `add_module`. The names you give
become the output paths that losses and metrics attach to.

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
            "Softmax": network.TargetCriterionsLoader()
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
        self.add_module("Softmax", torch.nn.Softmax(dim=1))
        self.add_module("Argmax", blocks.ArgMax(dim=1))
```

```yaml
Trainer:
  Model:
    classpath: Model:MySegNet
    MySegNet:
      dim: 2
      outputs_criterions:
        Softmax:                  # a loss needs an output with a gradient: not Argmax
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

An `outputs_criterions` key is a path in the graph, not a group name. A nested block's outputs are joined
with `:`, as in `UNetBlock_0:Head:Softmax` in `examples/Segmentation`. A plain `torch.nn.Module` also works,
but only a `Network` gets named outputs, patching and criteria.

## A transform

A transform receives one tensor at a time, channel first (`[C, (Z), Y, X]`), with the case name and an
`Attribute` holding the case's metadata. Store what `inverse` will need in that `Attribute`.

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

That is enough for a correct transform: it runs on the whole volume. To let it stream (read and write the
volume region by region), declare what it reads with the `locality` class attribute:

```python
class Threshold(Transform):
    locality = LocalityKind.POINTWISE

    def __call__(self, name, tensor, cache_attribute):
        return (tensor > 0.5).to(tensor.dtype)
```

| `__call__` reads | Declare | Also implement |
| --- | --- | --- |
| the same voxel | `POINTWISE` | nothing |
| a neighbourhood | `HALO`, with `halo` (radius per axis) | nothing |
| the volume flipped or permuted | `ORIENTATION` | `stream_region_source` |
| a shifted sub-box | `CROP` | `stream_region_source` |
| another grid | `REGRID` | subclass `Resample` |
| a whole-volume statistic | `GLOBAL_STAT`, with `stat_keys` | nothing |
| the whole volume | nothing (the default) | nothing |

The declaration must be true: nothing checks it against `__call__`, and a wrong one gives seams at region
borders. When unsure, declare nothing. The {doc}`transform guide <../config_guide/transform>` has a full
streaming example. Other points:

- Override the method `patch_locality(cache_attribute)` instead of the attribute when the answer depends on
  the case. It must only read the attribute, and answer `WHOLE_VOLUME` when a key is missing.
- A region stage that changes the geometry writes it in `write_stream_cache_attribute`, called once per
  case with the full shape.
- `self.datasets` gives access to other groups, as `Mask` and `Clip` use it.
- `single_process = True` marks a stage that cannot run in a DataLoader worker.
- `patch_transforms` accepts only `POINTWISE` and `GLOBAL_STAT` stages.

### Transforms from another library

A class that is not a `Transform` can be named directly. It runs on the whole volume, and KonfAI checks that
it keeps the shape:

```yaml
transforms:
  monai.transforms:ScaleIntensity:
    minv: 0.0
    maxv: 1.0
```

Write a wrapper when you need a locality, an inverse, a shape change, or a shorter set of arguments:

```python
# MonaiTransform.py
import torch
from monai.transforms import ScaleIntensity

from konfai.data.transform import Transform
from konfai.utils.dataset import Attribute

class MonaiScaleIntensity(Transform):
    def __init__(self, minv: float = 0.0, maxv: float = 1.0) -> None:
        super().__init__()
        self.transform = ScaleIntensity(minv=minv, maxv=maxv)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return self.transform(tensor)
```

Wrap the array transform, not the dictionary one (`ScaleIntensity`, not `ScaleIntensityd`): KonfAI pairs
images and labels through its groups.

## An augmentation

`_state_init` draws the random parameters once per case, `_compute` applies them to one tensor, `_inverse`
undoes them (implement it, even as a no-op).

```python
import torch

from konfai.data.augmentation import DataAugmentation
from konfai.utils.dataset import Attribute

class AddNoise(DataAugmentation):
    def __init__(self, sigma: float = 0.1, groups: list[str] | None = None) -> None:
        super().__init__(groups=groups)
        self.sigma = sigma

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        return shapes

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor.float()) * self.sigma

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return tensor
```

```yaml
augmentations:
  DataAugmentation_0:
    data_augmentations:
      MyAugmentations:AddNoise:
        sigma: 0.1
        prob: 0.5
    nb: 1
```

`a` is the copy number within the case. An augmentation declares its locality like a transform
(`_patch_locality`). A random draw per voxel is `POINTWISE` only if the noise depends on the voxel's
position, as the built-in `Noise` does; a field drawn on each call must stay `WHOLE_VOLUME`.

## A loss or a metric

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

```yaml
outputs_criterions:
  Head:Tanh:
    targets_criterions:
      CT:
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

The same block holds KonfAI's own keys (`is_loss`, `group`, `schedulers`) and your constructor's arguments.
`forward` returns a tensor, or a `(tensor, value_for_the_log)` tuple.

| Base class | Use it when | Signature |
| --- | --- | --- |
| `Criterion` | the usual case | `forward(output, *targets)` |
| `CriterionWithInit` | it needs the model before training | adds `init(model, output_group, target_group)` |
| `CriterionWithAttribute` | it needs the targets' geometry | `forward(output, *targets, attributes=...)` |

Class attributes worth setting: `maximize` (higher is better), `reducible` (can be accumulated patch by
patch), `batch_mean` (the value is a mean over the batch's patches), `default_is_loss` and `loss_capable`
(what `is_loss` means when the config leaves it out, and whether it may be a loss at all).

## A storage format

Subclass `AbstractFile`, implement its reads and writes, and register the format name:

```python
# chunked_backend.py
from konfai.utils.dataset import AbstractFile, Attribute, BACKENDS
from konfai.utils.errors import DatasetManagerError

class ChunkedFile(AbstractFile):
    """One case per store, decoded in blocks."""

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
    def file_to_data(self, group, name): ...                  # read a whole entry and its Attribute
    def file_to_data_slice(self, group, name, slices): ...    # read one region
    def data_to_file(self, name, data, attributes=None): ...  # write a whole entry

    def read_granularity(self, name):
        return (1, 64, 64, 64)                               # the block a region read decodes

BACKENDS["blk"] = ChunkedFile
```

- Add the extension to `SUPPORTED_EXTENSIONS` (`konfai.utils.utils`) so files are found on disk.
- Raise a `DatasetManagerError` naming the missing package when the library is not installed, at use and not
  at import.
- `read_granularity` tells the planner the block a region read really costs.
- `bounded_region_reads` answers whether a region read decodes only that region (default `False`).
- `can_stream` and `open_data_stream` add region writes; without them, writes go whole through
  `data_to_file`.

## A reduction

A reduction combines several tensors into one: the copies of a case in an ensemble or TTA, or the cases of a
cohort in TRANSFORM's `Reduce`.

```python
# my_reduction.py
import torch

from konfai.data.reduction import Reduction

class TrimmedMean(Reduction):
    """Mean after dropping each voxel's min and max."""

    voxel_local = True      # each output voxel reads only the same voxel of each input
    incremental = False     # needs every input at once
    working_multiple = 4.0  # buffers allocated on top of the inputs, in inputs-worth

    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        if len(tensors) < 3:
            raise ValueError("TrimmedMean needs at least three members")
        stack = torch.stack([tensor.float() for tensor in tensors])
        trimmed = stack.sum(dim=0) - stack.amax(dim=0) - stack.amin(dim=0)
        return trimmed / (len(tensors) - 2)
```

- `voxel_local = True` lets the reduction stream. Set it only if it is true: a wrong `True` corrupts a
  streamed output.
- `incremental = True` with `start` / `accumulate` / `finalize` folds the inputs one at a time.
- `working_multiple` (or `working_multiple_for(cases)`) lets the planner size its regions.
- Override `output_channels(channels, cases)` if the reduction changes the channel count, as `Concat` does.

## When it does not work

- The classpath imports another file or class than you meant (check the working directory).
- A constructor argument and its YAML key have different names.
- An `outputs_criterions` key names no output of the graph.
- `@config` moved the class's arguments to another level than the one you edited.

A mature workflow can be packaged as an app and reused as is: see {doc}`apps`.

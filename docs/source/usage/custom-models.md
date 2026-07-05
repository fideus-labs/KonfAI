# Using custom models, transforms, augmentations, and losses

When the built-in components are not enough, you extend KonfAI by writing
regular Python classes and selecting them from YAML — no core edits. Custom
objects are selected through `classpath` and instantiated through
`apply_config()`; this page gives the contract each object type must satisfy.

To integrate cleanly, a custom object must satisfy two contracts:

- the **configuration contract**: constructor argument names and YAML keys must match
- the **runtime contract**: the class must inherit the right KonfAI base class
  and implement the expected methods

This page focuses on the custom object types that matter most in practice:

- models
- transforms
- augmentations
- losses and metrics

**A local classpath such as `Model:UNetpp5` imports a Python file resolved
from the directory you launch `konfai` in** — keep the custom `.py` next to
the YAML and run the command from that directory.

## General rules

- Keep project-specific Python files next to the YAML when possible.
- Use explicit local classpaths such as `Model:UNetpp5` or `MyLoss:BoundaryDice`.
- Keep constructor argument names in `snake_case`.
- Do not use `@config()` for local custom classes by default. It inserts an
  extra YAML subtree named after the class and makes custom configs harder to
  read and generate.
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

`classpath` selects the Python implementation.

Example:

```yaml
Trainer:
  Model:
    classpath: Model:UNetpp5
```

Without `@config(...)`, a local class loaded through `classpath` reads its
constructor arguments directly from the current YAML branch.

For local custom classes, this is usually what you want:

```yaml
Trainer:
  Model:
    classpath: Model:UNetpp5
    outputs_criterions:
      ...
```

`@config(...)` is only needed when you deliberately want an extra nested
subtree.

In the current codebase:

- `@config("SomeKey")` binds the class to the `SomeKey` subtree
- `@config()` defaults to the class name

That means `@config()` on a local class loaded as `Model:UNetpp5` would force a
less convenient YAML shape such as:

```yaml
Trainer:
  Model:
    classpath: Model:UNetpp5
    UNetpp5:
      ...
```

Prefer avoiding that implicit nesting unless you explicitly need it.

## Contract summary

| Custom object | Recommended base class | Required methods |
| --- | --- | --- |
| Model | `konfai.network.network.Network` | `__init__`, then build the graph with `add_module(...)` |
| Transform | `konfai.data.transform.Transform` or `TransformInverse` | `__call__`, optionally `transform_shape`, and `inverse` for invertible transforms |
| Augmentation | `konfai.data.augmentation.DataAugmentation` | `_state_init`, `_compute`, `_inverse` |
| Loss / metric | `konfai.metric.measure.Criterion`, `CriterionWithInit`, or `CriterionWithAttribute` | `forward`, plus `init` when using `CriterionWithInit` |

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
  shipped `examples/Segmentation/Config.yml` the loss is attached to
  `UNetBlock_0:Head:Conv` (the pre-softmax logits).

KonfAI can wrap a simpler module internally in some situations, but if you want
reliable custom behavior, inheriting from `Network` is the supported path.

## Custom transforms

Use `konfai.data.transform.Transform` for one-way transforms and
`TransformInverse` when KonfAI must be able to invert the operation later.

The key methods are:

- `__call__(name, tensor, cache_attribute)` to transform the tensor
- `transform_shape(...)` if the transform changes the tensor shape
- `inverse(...)` if you inherit from `TransformInverse`

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

    def _compute(self, name: str, index: int, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        return [tensor + torch.randn_like(tensor.float()) * self.sigma for tensor in tensors]

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

- Augmentations operate on a list of tensors, not on a single tensor.
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

## Next steps

- {doc}`../concepts/configuration` — how `classpath` and `apply_config()` bind
  YAML keys to constructor arguments.
- {doc}`../concepts/model-graph` — how `add_module(...)` names become the
  output paths that `outputs_criterions` points at.
- {doc}`../reference/api/extension-points` — the full API of the base classes
  used on this page.

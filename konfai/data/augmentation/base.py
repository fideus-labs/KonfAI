# Copyright (c) 2025 Valentin Boussot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


"""The draw contract, the copies list, the affine matrix builders, a foreign framework's draw."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]
from konfai import konfai_root
from konfai.data.transform import LocalityKind, PatchLocality, RegionContext
from konfai.utils.config import _escape_key_component, apply_config, record_given_arguments
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import AugmentationError
from konfai.utils.runtime import NeedDevice, preserved_rng, seed_all
from konfai.utils.utils import get_module


def _require_simpleitk() -> None:
    """Raise a clear project error when an augmentation requires SimpleITK."""
    if sitk is None:
        raise AugmentationError(
            "SimpleITK is required for this augmentation. Install it with `pip install konfai[itk]`."
        )


def _translate_matrix(t: torch.Tensor) -> torch.Tensor:
    """The homogeneous matrix of the translation ``t`` (one entry per axis)."""
    matrix = torch.eye(t.shape[0] + 1)
    matrix[:-1, -1] = t
    return matrix


def _scale_matrix(s: torch.Tensor) -> torch.Tensor:
    """The homogeneous matrix of the per-axis scaling ``s`` (one entry per axis)."""
    matrix = torch.eye(s.shape[0] + 1)
    matrix[:-1, :-1] = torch.diag(s)
    return matrix


def _rotation_3d_matrix(rotation: torch.Tensor, center: torch.Tensor | None = None) -> torch.Tensor:
    a = torch.tensor(
        [
            [torch.cos(rotation[2]), -torch.sin(rotation[2]), 0],
            [torch.sin(rotation[2]), torch.cos(rotation[2]), 0],
            [0, 0, 1],
        ]
    )
    b = torch.tensor(
        [
            [torch.cos(rotation[1]), 0, torch.sin(rotation[1])],
            [0, 1, 0],
            [-torch.sin(rotation[1]), 0, torch.cos(rotation[1])],
        ]
    )
    c = torch.tensor(
        [
            [1, 0, 0],
            [0, torch.cos(rotation[0]), -torch.sin(rotation[0])],
            [0, torch.sin(rotation[0]), torch.cos(rotation[0])],
        ]
    )
    rotation_matrix = torch.cat(
        (
            torch.cat((a.mm(b).mm(c), torch.zeros((3, 1))), dim=1),
            torch.tensor([[0, 0, 0, 1]]),
        ),
        dim=0,
    )
    if center is not None:
        translation_before = torch.eye(4)
        translation_before[:-1, -1] = -center
        rotation_matrix = translation_before.mm(rotation_matrix)
    if center is not None:
        translation_after = torch.eye(4)
        translation_after[:-1, -1] = center
        rotation_matrix = rotation_matrix.mm(translation_after)
    return rotation_matrix


def _axis_rotation_matrix(theta: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation of a colour vector about ``axis`` by ``theta``, as a 4x4 homogeneous matrix.

    About the luma axis (1, 1, 1)/sqrt(3) it preserves luma and is the identity at theta = 0. The
    4th (alpha) channel is left untouched.
    """
    k = (axis[:3] / torch.linalg.norm(axis[:3])).to(torch.float32)
    cross = torch.zeros((3, 3))
    cross[0, 1], cross[0, 2] = -k[2], k[1]
    cross[1, 0], cross[1, 2] = k[2], -k[0]
    cross[2, 0], cross[2, 1] = -k[1], k[0]
    rot3 = torch.eye(3) * torch.cos(theta) + (1 - torch.cos(theta)) * torch.outer(k, k) + torch.sin(theta) * cross
    matrix = torch.eye(4)
    matrix[:3, :3] = rot3
    return matrix


def _rotation_2d_matrix(rotation: torch.Tensor, center: torch.Tensor | None = None) -> torch.Tensor:
    return torch.cat(
        (
            torch.cat(
                (
                    torch.tensor(
                        [
                            [torch.cos(rotation[0]), -torch.sin(rotation[0])],
                            [torch.sin(rotation[0]), torch.cos(rotation[0])],
                        ]
                    ),
                    torch.zeros((2, 1)),
                ),
                dim=1,
            ),
            torch.tensor([[0, 0, 1]]),
        ),
        dim=0,
    )


class Prob:
    def __init__(self, prob: float = 1.0) -> None:
        self.prob = prob


class DataAugmentationsList:
    def __init__(
        self,
        nb: int = 10,
        data_augmentations: dict[str, Prob] = {"default|Flip": Prob(1)},
    ) -> None:
        self.nb = nb
        self.data_augmentations: list[DataAugmentation] = []
        self.data_augmentationsLoader = data_augmentations

    def prepare(self, key: str) -> None:
        self.data_augmentations = []
        for augmentation, prob in self.data_augmentationsLoader.items():
            module, name = get_module(augmentation, "konfai.data.augmentation")
            # A key is read as a dotted path, and a classpath carries dots of its own.
            drawn = apply_config(
                f"{konfai_root()}.Dataset.augmentations.{key}.data_augmentations.{_escape_key_component(augmentation)}"
            )(getattr(module, name))()
            # A foreign class is handed over wrapped, the wrapper reading its own parameters from
            # the subtree the class read its arguments from.
            data_augmentation: DataAugmentation = (
                drawn
                if isinstance(drawn, DataAugmentation)
                else apply_config(
                    f"{konfai_root()}.Dataset.augmentations.{key}"
                    f".data_augmentations.{_escape_key_component(augmentation)}"
                )(partial(Foreign, drawn, augmentation))()
            )
            # A foreign class brings its own gate (`prob`, `p`), which a second gate would compose with.
            data_augmentation.load(1.0 if isinstance(data_augmentation, Foreign) else prob.prob)
            self.data_augmentations.append(data_augmentation)

    def set_datasets(self, datasets: list[Dataset]) -> None:
        for data_augmentation in self.data_augmentations:
            data_augmentation.set_datasets(datasets)


class DataAugmentation(NeedDevice, ABC):
    #: The one :class:`LocalityKind` every draw of this class makes, when it is unconditional.
    #: ``None`` keeps the fail-safe ``WHOLE_VOLUME``; a declaration that depends on the draw
    #: overrides ``_patch_locality``.
    locality: LocalityKind | None = None

    #: Companion to a ``HALO`` :attr:`locality`: per-spatial-axis radius in array order.
    halo: tuple[int, ...] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        # A draw is a chain stage too: konfai.api writes the config tree back from live objects.
        super().__init_subclass__(**kwargs)
        record_given_arguments(cls)

    def __init__(self, groups: list[str] | None = None) -> None:
        self.who_index: dict[int, list[int]] = {}
        self.shape_index: dict[int, list[list[int]]] = {}
        self._prob: float = 0
        self.groups = groups
        self.datasets: list[Dataset] = []

    def load(self, prob: float):
        self._prob = prob

    def set_datasets(self, datasets: list[Dataset]):
        self.datasets = datasets

    def reset_state(self, index: int | None = None) -> None:
        """Drop the cached sampling for *index* so the next ``state_init`` re-samples.

        Augmentation parameters are drawn once per case index and cached, so every patch of that
        case shares one transform within an epoch, and re-drawn at each epoch reset, which
        ``DatasetManager.reset_augmentation`` calls this from. ``None`` clears every cached index.
        """
        if index is None:
            self.who_index.clear()
            self.shape_index.clear()
        else:
            self.who_index.pop(index, None)
            self.shape_index.pop(index, None)

    def state_init(
        self,
        index: None | int,
        shapes: list[list[int]],
        caches_attribute: list[Attribute],
    ) -> list[list[int]]:
        if index is not None:
            if index not in self.who_index:
                self.who_index[index] = torch.where(torch.rand(len(shapes)) < self._prob)[0].tolist()
            else:
                return self.shape_index[index]
        else:
            index = 0
            self.who_index[index] = torch.where(torch.rand(len(shapes)) < self._prob)[0].tolist()

        if len(self.who_index[index]) > 0:
            for i, shape in enumerate(
                self._state_init(
                    index,
                    [shapes[i] for i in self.who_index[index]],
                    [caches_attribute[i] for i in self.who_index[index]],
                )
            ):
                shapes[self.who_index[index][i]] = shape
        self.shape_index[index] = shapes
        return self.shape_index[index]

    @abstractmethod
    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        pass

    def _slot(self, index: int, a: int) -> int:
        """The slot copy *a*'s draw is kept at: state is stored for the selected copies only, in
        selection order, and the ``_``-prefixed methods are handed that slot."""
        return self.who_index[index].index(a)

    def patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        """Declare how the draw of copy *a* makes its output depend on its input, for patch streaming.

        The same contract as :meth:`konfai.data.transform.Transform.patch_locality` (read-only, no
        I/O, total, ``WHOLE_VOLUME`` by default), asked of one copy of one case.
        """
        if a not in self.who_index[index]:
            return PatchLocality(LocalityKind.POINTWISE)
        return self._patch_locality(index, self._slot(index, a), cache_attribute)

    def _patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        if self.locality is not None:
            return PatchLocality(self.locality, halo=self.halo)
        return PatchLocality(LocalityKind.WHOLE_VOLUME)

    def stream_region_source(
        self,
        index: int,
        a: int,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
    ) -> list[slice]:
        """Map a target patch's spatial slices to the source region copy *a* reads (region kinds)."""
        return self._stream_region_source(index, self._slot(index, a), target_slices, source_spatial_shape)

    def stream_shape(self, index: int, a: int, shape: list[int]) -> list[int]:
        """The spatial shape copy *a*'s draw produces from ``shape``, the counterpart of
        ``Transform.transform_shape``. A shape-changing draw restates what ``state_init`` did."""
        return self._stream_shape(index, self._slot(index, a), shape)

    def _stream_shape(self, index: int, a: int, shape: list[int]) -> list[int]:
        return shape

    def _stream_region_source(
        self,
        index: int,
        a: int,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
    ) -> list[slice]:
        raise AugmentationError(
            f"{type(self).__name__} declared a region patch-locality but does not implement _stream_region_source().",
            "Implement _stream_region_source() or declare a non-region _patch_locality().",
        )

    def compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the draw of copy *a* to one tensor: the forward counterpart of :meth:`inverse`."""
        if a in self.who_index[index]:
            tensor = self._compute(name, index, self._slot(index, a), tensor)
        return tensor

    def stream_region(
        self, name: str, index: int, a: int, tensor: torch.Tensor, context: RegionContext
    ) -> torch.Tensor:
        """Apply the draw of copy *a* to one region, told where it sits (the same contract as
        :meth:`konfai.data.transform.Transform.stream_region`). The default is the draw itself; a
        draw parameterised by the place overrides ``_stream_region``."""
        if a not in self.who_index[index]:
            return tensor
        return self._stream_region(name, index, self._slot(index, a), tensor, context)

    def _stream_region(
        self, name: str, index: int, a: int, tensor: torch.Tensor, context: RegionContext
    ) -> torch.Tensor:
        del context
        return self._compute(name, index, a, tensor)

    def __call__(
        self,
        name: str,
        index: int,
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return [self.compute(name, index, a, tensor) for a, tensor in enumerate(tensors)]

    @abstractmethod
    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        """Copy ``a`` of case ``index`` drawn from ``tensor``: a fresh tensor or a view of it.

        Every copy the draw did not select aliases ``tensor``, so nothing may be written into it.
        """

    def inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        if a in self.who_index[index]:
            tensor = self._inverse(index, self._slot(index, a), tensor)
        return tensor

    @abstractmethod
    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        pass


def _hashed_normal_field(
    seed: int, shape: tuple[int, ...], offsets: tuple[int, ...], full: tuple[int, ...], device: torch.device
) -> torch.Tensor:
    """A standard-normal field over ``shape`` (channel first) whose value at a voxel is a function of
    ``(seed, channel, position in the full volume)``, so a region holds its part of the field."""
    channels, spatial = int(shape[0]), tuple(int(extent) for extent in shape[1:])
    positions = [
        torch.arange(start, start + extent, device=device, dtype=torch.int64)
        for start, extent in zip(offsets, spatial, strict=True)
    ]
    linear = torch.zeros((), device=device, dtype=torch.int64)
    for axis, (position, extent) in enumerate(zip(positions, full, strict=True)):
        view = [1] * len(spatial)
        view[axis] = -1
        linear = linear * int(extent) + position.reshape(view)
    voxels = int(np.prod(full, dtype=np.int64))
    channel = torch.arange(channels, device=device, dtype=torch.int64).reshape(-1, *[1] * len(spatial))
    key = (channel * voxels + linear).expand(channels, *spatial) * 2 + torch.tensor(
        seed, device=device, dtype=torch.int64
    ) * (2 * voxels * channels + 1)

    def shift(value: torch.Tensor, bits: int) -> torch.Tensor:
        return (value >> bits) & ((1 << (64 - bits)) - 1)  # a logical shift on the two's-complement int64

    def mix(value: torch.Tensor) -> torch.Tensor:
        value = value + (-7046029254386353131)  # 0x9E3779B97F4A7C15 as a signed 64-bit constant
        value = (value ^ shift(value, 30)) * (-4658895280553007687)  # 0xBF58476D1CE4E5B9
        value = (value ^ shift(value, 27)) * (-7723592293110705685)  # 0x94D049BB133111EB
        return value ^ shift(value, 31)

    def uniform(value: torch.Tensor) -> torch.Tensor:
        # The top 53 bits of the hash, in (0, 1): never exactly 0, which the log needs.
        return shift(value, 11).to(torch.float64) * (1.0 / (1 << 53)) + (0.5 / (1 << 53))

    first, second = uniform(mix(key)), uniform(mix(key + 1))
    normal = torch.sqrt(-2.0 * torch.log(first)) * torch.cos(2.0 * torch.pi * second)
    return normal.to(torch.float32)


def _reflect_interval(low: float, high: float, span: float) -> tuple[float, float]:
    """The interval ``[low, high]`` after ``padding_mode='reflection'`` folds it into ``[0, span]``:
    the hull of the folded endpoints and the fold points inside the interval."""
    if span <= 0:
        return 0.0, 0.0

    def fold(value: float) -> float:
        magnitude = abs(value)
        flips = np.floor(magnitude / span)
        extra = magnitude - flips * span
        return extra if flips % 2 == 0 else span - extra

    points = [fold(low), fold(high)]
    for k in range(int(np.ceil(low / span)), int(np.floor(high / span)) + 1):
        points.append(0.0 if k % 2 == 0 else span)
    return max(0.0, min(points)), min(span, max(points))


class Foreign(DataAugmentation):
    """Draw an augmentation from another framework.

    ``classpath`` is ``module:Class`` and ``args`` are the arguments that class takes::

        augmentations:
          Foreign:
            classpath: monai.transforms:RandGaussianNoise
            args: {prob: 1.0, std: 12.0}
            groups: [CT]

    The class must be callable on one tensor, return the transformed tensor, and keep its shape. A
    draw belongs to the case, and each group of the case is handed the same copy of it: the seed is
    drawn once and the global state is set from it before every group.

    Name the one group a foreign draw belongs to: a single draw suits several groups only when the
    class consumes its random state identically whatever it is given and does not sample values.
    Subclass ``DataAugmentation`` for a draw that must span groups.
    """

    def __init__(self, transform, classpath: str, groups: list[str] | None = None) -> None:
        super().__init__(groups)
        self.classpath = classpath
        self.transform = transform
        self.seeds: dict[int, list[int]] = {}

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        # One seed per copy, drawn once for the case: every group is handed these same seeds.
        self.seeds[index] = torch.randint(0, 2**31 - 1, (len(shapes),)).tolist()
        return shapes

    @contextmanager
    def _seeded(self, seed: int):
        """Put the class's random state where the seed says, and give the process back what it had.
        Both the interpreter's global state and the class's own ``set_random_state`` are set."""
        with preserved_rng():
            seed_all(seed)
            set_random_state = getattr(self.transform, "set_random_state", None)
            if callable(set_random_state):
                set_random_state(seed=seed)
            yield

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        # A class from another framework may write into what it is handed, and the tensor is shared.
        with self._seeded(self.seeds[index][a]):
            result = self.transform(tensor.clone())
        if not isinstance(result, torch.Tensor):
            result = torch.as_tensor(np.asarray(result))
        if list(result.shape) != list(tensor.shape):
            raise AugmentationError(
                f"'{self.classpath}' returned the shape {list(result.shape)} for an input of {list(tensor.shape)}.",
                "Subclass DataAugmentation and return the shape from _state_init to draw onto another grid.",
            )
        return result

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        raise AugmentationError(
            f"'{self.classpath}' cannot be undone.",
            "Subclass DataAugmentation and implement _inverse(), or drop the augmentation from a"
            " workflow that inverts it.",
        )

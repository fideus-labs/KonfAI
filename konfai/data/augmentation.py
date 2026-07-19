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

"""Data augmentation primitives applied by KonfAI datasets."""

import random
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]

from konfai import konfai_root
from konfai.data.transform import LocalityKind, PatchLocality
from konfai.utils.config import _escape_key_component, apply_config
from konfai.utils.dataset import Attribute, Dataset, data_to_image
from konfai.utils.errors import AugmentationError
from konfai.utils.runtime import NeedDevice
from konfai.utils.utils import get_module


def _require_simpleitk() -> None:
    """Raise a clear project error when an augmentation requires SimpleITK."""
    if sitk is None:
        raise AugmentationError(
            "SimpleITK is required for this augmentation. Install it with `pip install konfai[itk]`."
        )


def _translate_2d_matrix(t: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            torch.cat((torch.eye(2), torch.tensor([[t[0]], [t[1]]])), dim=1),
            torch.Tensor([[0, 0, 1]]),
        ),
        dim=0,
    )


def _translate_3d_matrix(t: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            torch.cat((torch.eye(3), torch.tensor([[t[0]], [t[1]], [t[2]]])), dim=1),
            torch.Tensor([[0, 0, 0, 1]]),
        ),
        dim=0,
    )


def _scale_2d_matrix(s: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            torch.cat((torch.eye(2) * s, torch.tensor([[0], [0]])), dim=1),
            torch.tensor([[0, 0, 1]]),
        ),
        dim=0,
    )


def _scale_3d_matrix(s: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            torch.cat((torch.eye(3) * s, torch.tensor([[0], [0], [0]])), dim=1),
            torch.tensor([[0, 0, 0, 1]]),
        ),
        dim=0,
    )


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

    Hue rotation is a rotation of the RGB vector about the luma axis (1, 1, 1)/sqrt(3): it preserves luma
    (a grey pixel stays grey) and is identity at theta = 0. The 4th (alpha) channel is left untouched.
    Using Euler XYZ angles about the coordinate axes instead — as ``_rotation_3d_matrix(theta.repeat(3), v)``
    did — is not a rotation about the luma axis and recolours grey pixels.
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
            # A key is read as a dotted path, and a classpath naming its module carries dots of its own.
            drawn = apply_config(
                f"{konfai_root()}.Dataset.augmentations.{key}.data_augmentations.{_escape_key_component(augmentation)}"
            )(getattr(module, name))()
            # A foreign class is handed over wrapped, and the wrapper reads its own parameters from
            # the same subtree the class read its arguments from -- as MinimalModel does for a model.
            data_augmentation: DataAugmentation = (
                drawn
                if isinstance(drawn, DataAugmentation)
                else apply_config(
                    f"{konfai_root()}.Dataset.augmentations.{key}"
                    f".data_augmentations.{_escape_key_component(augmentation)}"
                )(partial(Foreign, drawn, augmentation))()
            )
            # A foreign class brings all of its randomness, including whether it applies at all, and
            # names that gate itself (`prob`, `p`). A second gate here would compose with it, so a
            # probability of one half would be one quarter. The one it declares is the one that runs.
            data_augmentation.load(1.0 if isinstance(data_augmentation, Foreign) else prob.prob)
            self.data_augmentations.append(data_augmentation)

    def set_datasets(self, datasets: list[Dataset]) -> None:
        for data_augmentation in self.data_augmentations:
            data_augmentation.set_datasets(datasets)


class DataAugmentation(NeedDevice, ABC):
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

        Augmentation parameters are drawn once per case index and cached so that
        every patch of that case shares a consistent transform within an epoch
        (see ``state_init``). They must, however, be re-drawn at the start of each
        epoch; otherwise a case keeps identical augmentation parameters for the
        whole run. ``DatasetManager.reset_augmentation`` calls this before
        ``state_init`` on every epoch reset. Subclass-specific caches (e.g.
        ``matrix``/``flip``) are keyed by the same index and are overwritten by
        the subsequent ``_state_init``; when the re-draw selects nothing they are
        left untouched but never read (``__call__``/``inverse`` gate on
        ``who_index``). Passing ``None`` clears every cached index.
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

    def patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        """Declare how the draw of copy *a* makes its output depend on its input, for patch streaming.

        The same contract as :meth:`konfai.data.transform.Transform.patch_locality` -- read-only, no
        I/O, total, ``WHOLE_VOLUME`` by default -- asked of one copy of one case, because that is the
        grain an augmentation is parameterised at: the halo of a geometric draw is the draw's own, so
        two copies of the same case answer differently and the same copy answers differently next
        epoch. A copy the draw did not select is the identity, which the base answers for.
        """
        if a not in self.who_index[index]:
            return PatchLocality(LocalityKind.POINTWISE)
        return self._patch_locality(index, self.who_index[index].index(a), cache_attribute)

    def _patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.WHOLE_VOLUME)

    def stream_region_source(
        self,
        index: int,
        a: int,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
    ) -> list[slice]:
        """Map a target patch's spatial slices to the source region copy *a* reads (region kinds)."""
        return self._stream_region_source(index, self.who_index[index].index(a), target_slices, source_spatial_shape)

    def stream_shape(self, index: int, a: int, shape: list[int]) -> list[int]:
        """The spatial shape copy *a*'s draw produces from ``shape`` (the shape-fold counterpart of
        ``Transform.transform_shape``). The identity default covers every draw but a shape-changing
        one, which restates here what its ``state_init`` did to the copy's grid."""
        return self._stream_shape(index, self.who_index[index].index(a), shape)

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
        """Apply the draw of copy *a* to one tensor -- the forward counterpart of :meth:`inverse`."""
        if a in self.who_index[index]:
            tensor = self._compute(name, index, self.who_index[index].index(a), tensor)
        return tensor

    def __call__(
        self,
        name: str,
        index: int,
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return [self.compute(name, index, a, tensor) for a, tensor in enumerate(tensors)]

    @abstractmethod
    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        pass

    def inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        if a in self.who_index[index]:
            tensor = self._inverse(index, self.who_index[index].index(a), tensor)
        return tensor

    @abstractmethod
    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        pass


class EulerTransform(DataAugmentation):
    def __init__(self) -> None:
        super().__init__()
        self.matrix: dict[int, list[torch.Tensor]] = {}

    def _grid_matrix(self, index: int, a: int, shape: list[int]) -> torch.Tensor:
        """Copy *a*'s affine, in the normalised coordinates ``affine_grid`` spans over ``shape``."""
        return self.matrix[index][a]

    def _sample(self, matrix: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        # Integer tensors are label maps: interpolating them blends class ids into
        # non-existent labels, so resample them with nearest-neighbour instead.
        mode = "nearest" if not tensor.dtype.is_floating_point else "bilinear"
        return (
            F.grid_sample(
                tensor.unsqueeze(0).type(torch.float32),
                F.affine_grid(matrix[:, :-1, ...], [1, *list(tensor.shape)], align_corners=True).to(tensor.device),
                align_corners=True,
                mode=mode,
                padding_mode="reflection",
            )
            .type(tensor.dtype)
            .squeeze(0)
        )

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return self._sample(self._grid_matrix(index, a, list(tensor.shape[1:])), tensor)

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return self._sample(self._grid_matrix(index, a, list(tensor.shape[1:])).inverse(), tensor)


class Translate(EulerTransform):
    def __init__(self, t_min: float = -10, t_max=10, is_int: bool = False):
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max
        self.is_int = is_int
        self.translate: dict[int, torch.Tensor] = {}

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        dim = len(shapes[0])
        translate = torch.rand((len(shapes), dim)) * torch.tensor(self.t_max - self.t_min) + torch.tensor(self.t_min)
        self.translate[index] = torch.round(translate) if self.is_int else translate
        return shapes

    def _grid_matrix(self, index: int, a: int, shape: list[int]) -> torch.Tensor:
        # The draw is a shift in VOXELS, in (x, y, z). ``affine_grid`` spans [-1, 1] over whatever
        # extent it is given, so the same shift is a different matrix on a patch than on the volume:
        # normalise it against the extent it is about to be applied to, never against a fixed one.
        func = _translate_3d_matrix if len(shape) == 3 else _translate_2d_matrix
        sizes = torch.tensor(list(reversed(shape)), dtype=torch.float32)
        return torch.unsqueeze(func(self.translate[index][a] * 2.0 / (sizes - 1)), dim=0)

    def _patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        # A uniform shift sends a target patch to that same patch displaced by the draw, so the source
        # is a bounded neighbourhood of it. One voxel past the ceiling covers the far tap a fractional
        # shift interpolates from. The draw is in (x, y, z); a halo is in array order.
        radius = (torch.ceil(self.translate[index][a].abs()).to(torch.int64) + 1).tolist()
        return PatchLocality(LocalityKind.HALO, halo=tuple(int(r) for r in reversed(radius)))


class Rotate(EulerTransform):
    """Rotate a copy of the case about its centre.

    A quarter draw is a signed permutation of the axes: an exact index remap (permute + flip), never an
    interpolation, and it transposes the extents it swaps, so the copy is cut on its own grid. A free
    angle displaces a voxel by 2 * R * sin(theta / 2) from the centre, which no constant halo bounds --
    it stays whole-volume.
    """

    # A quarter angle's cosines are computed in float32, so an entry of the composed matrix lands within
    # ~1e-7 of the 0 or +/-1 it stands for rather than on it.
    _QUARTER_ATOL = 1e-6

    def __init__(self, a_min: float = 0, a_max: float = 360, is_quarter: bool = False):
        super().__init__()
        self.a_min = a_min
        self.a_max = a_max
        self.is_quarter = is_quarter

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        dim = len(shapes[0])
        func = _rotation_3d_matrix if dim == 3 else _rotation_2d_matrix
        angles = []

        if self.is_quarter:
            quarter_angles = torch.tensor([90.0, 180.0, 270.0])
            choices = torch.randint(0, quarter_angles.numel(), (len(shapes), dim))
            angles = torch.deg2rad(quarter_angles[choices])
        else:
            angles = torch.deg2rad(
                torch.rand((len(shapes), dim)) * torch.tensor(self.a_max - self.a_min) + torch.tensor(self.a_min)
            )

        self.matrix[index] = [torch.unsqueeze(func(value), dim=0) for value in angles]
        # A quarter turn transposes the extents it swaps, so a copy whose draw is one is cut on the grid
        # that draw lands on. A sampled draw keeps the grid it was applied to.
        return [Rotate._draw_shape(self.matrix[index][a], shape) for a, shape in enumerate(shapes)]

    @classmethod
    def _index_remap(cls, matrix: torch.Tensor) -> tuple[list[int], list[int]] | None:
        """The permute dims and flip axes reproducing a rotation exactly, or ``None`` if it must be sampled.

        ``matrix`` maps an output coordinate onto the input it comes from, so it is a signed permutation
        exactly when every row and column has a single +/-1: output axis ``pi(k)`` then reads input axis
        ``k``, mirrored where the sign is negative. An orthonormal row of L1 norm 1 has one such entry,
        which is what separates a quarter turn from any other angle.

        Dims and axes are channel-first, where physical axis ``k`` is dim ``n - k``.
        """
        linear = matrix[0, :-1, :-1]
        n = linear.shape[0]
        unit = torch.ones(n)
        if not torch.allclose(linear.abs().sum(0), unit, atol=cls._QUARTER_ATOL):
            return None
        if not torch.allclose(linear.abs().sum(1), unit, atol=cls._QUARTER_ATOL):
            return None

        dims = [0] * (n + 1)
        flips: list[int] = []
        for k in range(n):
            source = int(linear[k].abs().argmax())
            dims[n - source] = n - k
            if linear[k, source] < 0:
                flips.append(n - source)
        return dims, flips

    @classmethod
    def _draw_shape(cls, matrix: torch.Tensor, shape: list[int]) -> list[int]:
        """The spatial extents a draw lands on, given the ones it is applied to.

        Output dim ``i`` reads input dim ``dims[i]``, so it carries that axis's extent with it -- what a
        turn preserves is the volume, not which axis holds an extent. A sampled draw spans the extent it
        is given. ``dims`` is channel-first, so spatial axis k is dim k + 1.
        """
        remap = cls._index_remap(matrix)
        if remap is None:
            return list(shape)
        dims, _ = remap
        return [shape[dim - 1] for dim in dims[1:]]

    def _reorient(self, index: int, a: int, matrix: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        remap = Rotate._index_remap(matrix)
        if remap is None:
            return self._sample(matrix, tensor)
        dims, flips = remap
        # flip materialises the permuted view, so the copy never aliases the tensor it was drawn from.
        return tensor.permute(dims).flip(flips)

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return self._reorient(index, a, self._grid_matrix(index, a, list(tensor.shape[1:])), tensor)

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return self._reorient(index, a, self._grid_matrix(index, a, list(tensor.shape[1:])).inverse(), tensor)

    def _patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        # Permuting and mirroring voxels is a bijection on them, which is what ORIENTATION promises and
        # what LocalityKind.preserves_statistics lets a later stage trust. Only the draw can say whether
        # this one is that, and the draw is a property of the copy rather than of the case.
        if Rotate._index_remap(self.matrix[index][a]) is None:
            return PatchLocality(LocalityKind.WHOLE_VOLUME)
        return PatchLocality(LocalityKind.ORIENTATION)

    def _stream_shape(self, index: int, a: int, shape: list[int]) -> list[int]:
        # The same extent carry state_init applied to the copy's grid.
        return Rotate._draw_shape(self.matrix[index][a], list(shape))

    def _stream_region_source(
        self,
        index: int,
        a: int,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
    ) -> list[slice]:
        # Output axis o reads input axis dims[o], so placing o's target slice at that input axis yields
        # the region whose remap reproduces the patch; a MIRRORED output axis reads the mirror region
        # ``[n - stop, n - start)`` of it. Dims and flips are channel-first, so spatial axis k is dim k + 1.
        remap = Rotate._index_remap(self.matrix[index][a])
        if remap is None:
            raise AugmentationError(
                "Rotate declared a region patch-locality for a draw it cannot remap exactly.",
                "Report this: _patch_locality() and _stream_region_source() disagree about the draw.",
            )
        dims, flips = remap
        source_slices = [slice(0, n) for n in source_spatial_shape]
        for out_dim in range(1, len(dims)):
            in_axis = dims[out_dim] - 1
            sl = target_slices[out_dim - 1]
            n = source_spatial_shape[in_axis]
            source_slices[in_axis] = slice(n - sl.stop, n - sl.start) if out_dim in flips else slice(sl.start, sl.stop)
        return source_slices


class Scale(EulerTransform):
    # WHOLE_VOLUME on purpose: a scale about the volume centre displaces a voxel by |s - 1| * its
    # distance from that centre, so the source region depends on where the patch sits -- no constant
    # halo is both correct at the border and cheap in the middle.
    def __init__(self, s_std: float = 0.2):
        super().__init__()
        self.s_std = s_std

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        func = _scale_3d_matrix if len(shapes[0]) == 3 else _scale_2d_matrix
        scale = torch.Tensor.repeat(
            torch.exp2(torch.randn(len(shapes)) * self.s_std).unsqueeze(1),
            [1, len(shapes[0])],
        )
        self.matrix[index] = [torch.unsqueeze(func(value), dim=0) for value in scale]
        return shapes


class Foreign(DataAugmentation):
    """Draw an augmentation from another framework.

    ``classpath`` is ``module:Class`` and ``args`` are the arguments that class takes::

        augmentations:
          Foreign:
            classpath: monai.transforms:RandGaussianNoise
            args: {prob: 1.0, std: 12.0}
            groups: [CT]

    The class must be callable on one tensor, return the transformed tensor, and keep its shape --
    which is what torchvision's transforms, TorchIO's and MONAI's array transforms all do.

    A draw belongs to the case, and each group of the case is handed the same copy of it. The seed
    of the copy is drawn once and the global state is set from it before every group, so the class
    draws the same way for the label as for the image.

    Name the ONE group a foreign draw belongs to. A single draw suits several groups only when the
    class consumes its random state identically whatever it is given and the draw does not SAMPLE:
    a rotation of the image is a rotation of the label, but a label interpolated between two ids is
    neither. Subclass ``DataAugmentation`` for a draw that must span groups -- the draw is then a
    value this framework holds, rather than a random state two libraries agree about.
    """

    def __init__(self, transform, classpath: str, groups: list[str] | None = None) -> None:
        super().__init__(groups)
        self.classpath = classpath
        self.transform = transform
        self.seeds: dict[int, list[int]] = {}

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        # One seed per copy, drawn once for the case: every group of it is handed these same seeds.
        self.seeds[index] = torch.randint(0, 2**31 - 1, (len(shapes),)).tolist()
        return shapes

    @contextmanager
    def _seeded(self, seed: int):
        """Put the class's random state where the seed says, and give the process back what it had.

        A class draws either from the interpreter's global state, which torchvision's transforms and
        TorchIO's draw from, or from a state of its own, which MONAI's Randomizable holds and reaches
        through ``set_random_state``. Both are set: which one a class uses is not something it says.

        The global state belongs to the run, not to this draw. Left where the class stopped, the two
        groups of one case would leave it in the same place and whatever drew next would draw twice
        the same -- and torch's seed reaches the devices, where the model draws its own.
        """
        states = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
        # torch.manual_seed also (re)seeds every CUDA generator, so snapshot those too -- but only when
        # CUDA is already initialised, so a CPU data-loader worker is never forced to spin CUDA up.
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        try:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            set_random_state = getattr(self.transform, "set_random_state", None)
            if callable(set_random_state):
                set_random_state(seed=seed)
            yield
        finally:
            random.setstate(states[0])
            np.random.set_state(states[1])
            torch.random.set_rng_state(states[2])
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        with self._seeded(self.seeds[index][a]):
            result = self.transform(tensor)
        if not isinstance(result, torch.Tensor):
            result = torch.as_tensor(np.asarray(result))
        if list(result.shape) != list(tensor.shape):
            raise AugmentationError(
                f"'{self.classpath}' returned the shape {list(result.shape)} for an input of {list(tensor.shape)}.",
                "Subclass DataAugmentation and return the shape from _state_init to draw onto another grid.",
            )
        return result

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        # Undoing a draw is a second thing a class must expose, and the convention this reads covers
        # applying one alone.
        raise AugmentationError(
            f"'{self.classpath}' cannot be undone.",
            "Subclass DataAugmentation and implement _inverse(), or drop the augmentation from a"
            " workflow that inverts it.",
        )


class Flip(DataAugmentation):
    def __init__(self, f_prob: list[float] = [0.33, 0.33, 0.33], vector_field: bool = False) -> None:
        super().__init__()
        self.f_prob = f_prob
        self.vector_field = vector_field
        self.flip: dict[int, list[list[int]]] = {}

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        prob = torch.rand((len(shapes), len(self.f_prob))) < torch.tensor(self.f_prob)
        dims = torch.tensor([1, 2, 3][: len(self.f_prob)])
        self.flip[index] = [dims[mask].tolist() for mask in prob]
        return shapes

    def _flip(self, tensor: torch.Tensor, dims: list[int]) -> torch.Tensor:
        result = torch.flip(tensor, dims=dims)
        # A displacement/vector field (one channel per spatial axis, channel-first [C=(dx,dy,dz),(D),H,W])
        # is not mirror-invariant: flipping a spatial axis must also negate its component channel
        # (channel = tensor.dim() - 1 - dim, as channels are in (x,y,z) order and axes are reversed).
        # Enable ``vector_field`` only in configs whose augmented tensors are single-channel (scalars/masks,
        # left untouched) or genuine vector fields: any OTHER multi-channel tensor whose channel count
        # equals the spatial rank (e.g. a 3-contrast volume in 3D) would be wrongly negated by this guard.
        if self.vector_field and tensor.shape[0] == tensor.dim() - 1:
            for dim in dims:
                result[tensor.dim() - 1 - dim] = -result[tensor.dim() - 1 - dim]
        return result

    def _patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        # A mirror is a bijection on the voxels (ORIENTATION). Negating a component channel is not: it
        # maps values, so a later GLOBAL_STAT could no longer seed from the stored volume -- and only
        # the tensor's channel count says whether it fires, which a header-time declaration cannot see.
        if self.vector_field:
            return PatchLocality(LocalityKind.WHOLE_VOLUME)
        return PatchLocality(LocalityKind.ORIENTATION)

    def _stream_region_source(
        self,
        index: int,
        a: int,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
    ) -> list[slice]:
        # A flipped spatial axis reads the mirror region ``[n - stop, n - start)``; flipping that
        # sub-region reproduces the target patch. Non-flipped axes read the identity region. ``flip``
        # holds channel-first tensor dims, so spatial axis k is dim k + 1.
        dims = self.flip[index][a]
        return [
            (
                slice(source_spatial_shape[k] - sl.stop, source_spatial_shape[k] - sl.start)
                if (k + 1) in dims
                else slice(sl.start, sl.stop)
            )
            for k, sl in enumerate(target_slices)
        ]

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return self._flip(tensor, self.flip[index][a])

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return self._flip(tensor, self.flip[index][a])


class ColorTransform(DataAugmentation):
    def __init__(self, groups: list[str] | None = None) -> None:
        super().__init__(groups)
        self.matrix: dict[int, list[torch.Tensor]] = {}

    def _patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        # The draw is a colour matrix applied to each voxel on its own: no neighbour, no coordinate,
        # no extent. Whatever region a voxel is read in, it comes out the same.
        return PatchLocality(LocalityKind.POINTWISE)

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        matrix = self.matrix[index][a]
        result = tensor.reshape([*tensor.shape[:1], int(np.prod(tensor.shape[1:]))])
        if tensor.shape[0] == 3:
            matrix = matrix.to(tensor.device)
            result = matrix[:, :3, :3] @ result.float() + matrix[:, :3, 3:]
        elif tensor.shape[0] == 1:
            matrix = matrix[:, :3, :].mean(dim=1, keepdims=True).to(tensor.device)
            result = result.float() * matrix[:, :, :3].sum(dim=2, keepdims=True) + matrix[:, :, 3:]
        else:
            raise AugmentationError("Image must be RGB (3 channels) or L (1 channel)")
        return result.reshape(tensor.shape)

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class Brightness(ColorTransform):
    def __init__(self, b_std: float, groups: list[str] | None = None) -> None:
        super().__init__(groups)
        self.b_std = b_std

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        brightness = torch.Tensor.repeat((torch.randn(len(shapes)) * self.b_std).unsqueeze(1), [1, 3])
        self.matrix[index] = [torch.unsqueeze(_translate_3d_matrix(value), dim=0) for value in brightness]
        return shapes


class Contrast(ColorTransform):
    def __init__(self, c_std: float, groups: list[str] | None = None) -> None:
        super().__init__(groups)
        self.c_std = c_std

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        contrast = torch.exp2(torch.randn(len(shapes)) * self.c_std)
        self.matrix[index] = [torch.unsqueeze(_scale_3d_matrix(value), dim=0) for value in contrast]
        return shapes


class LumaFlip(ColorTransform):
    def __init__(self, groups: list[str] | None = None) -> None:
        super().__init__(groups)
        self.v = torch.tensor([1, 1, 1, 0]) / torch.sqrt(torch.tensor(3))

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        luma = torch.floor(torch.rand([len(shapes), 1, 1]) * 2)
        self.matrix[index] = [torch.unsqueeze((torch.eye(4) - 2 * self.v.ger(self.v) * value), dim=0) for value in luma]
        return shapes


class HUE(ColorTransform):
    def __init__(self, hue_max: float, groups: list[str] | None = None) -> None:
        super().__init__(groups)
        self.hue_max = hue_max
        self.v = torch.tensor([1, 1, 1]) / torch.sqrt(torch.tensor(3))

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        theta = (torch.rand([len(shapes)]) * 2 - 1) * np.pi * self.hue_max
        self.matrix[index] = [torch.unsqueeze(_axis_rotation_matrix(value, self.v), dim=0) for value in theta]
        return shapes


class Saturation(ColorTransform):
    def __init__(self, s_std: float, groups: list[str] | None = None) -> None:
        super().__init__(groups)
        self.s_std = s_std
        self.v = torch.tensor([1, 1, 1, 0]) / torch.sqrt(torch.tensor(3))

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        saturation = torch.exp2(torch.randn(len(shapes)) * self.s_std)
        # Keep the luma component (v vT) at unit gain and scale only the orthogonal chroma component
        # (I - v vT) by the saturation factor. The previous parenthesisation scaled the whole matrix,
        # i.e. (v vT + (I - v vT)) * s = I * s, a uniform per-channel gain (contrast) that never mixes
        # toward luma. With this form s=1 is identity, s=0 collapses to greyscale, s>1 boosts saturation.
        self.matrix[index] = [
            (self.v.ger(self.v) + (torch.eye(4) - self.v.ger(self.v)) * value).unsqueeze(0) for value in saturation
        ]
        return shapes


class Noise(DataAugmentation):
    def __init__(
        self,
        n_std: float,
        noise_step: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        groups: list[str] | None = None,
    ) -> None:
        super().__init__(groups)
        self.n_std = n_std
        self.noise_step = noise_step

        self.ts: dict[int, list[torch.Tensor]] = {}
        self.betas = torch.linspace(beta_start, beta_end, noise_step)
        self.betas = Noise.enforce_zero_terminal_snr(self.betas)
        self.alphas = 1 - self.betas
        self.alpha_hat = torch.concat((torch.ones(1), torch.cumprod(self.alphas, dim=0)))
        self.max_T = 0.0

        self.C = 1
        self.n = 4
        self.d = 0.25
        self._prob = 1

    @staticmethod
    def enforce_zero_terminal_snr(betas: torch.Tensor):
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_t = alphas_bar_sqrt[-1].clone()
        alphas_bar_sqrt -= alphas_bar_sqrt_t
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_t)
        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1 - alphas
        return betas

    def load(self, prob: float):
        self.max_T = prob * self.noise_step

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        if int(self.max_T) == 0:
            self.ts[index] = [0 for _ in shapes]
        else:
            self.ts[index] = [torch.randint(0, int(self.max_T), (1,)) for _ in shapes]
        return shapes

    # WHOLE_VOLUME on purpose: the noise field is drawn per call, not per voxel position, so two
    # overlapping patches would sample unrelated fields and the overlap blend would suppress the
    # variance this exists to add.
    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        alpha_hat_t = self.alpha_hat[self.ts[index][a]].to(tensor.device).reshape(*[1 for _ in tensor.shape])
        return (
            alpha_hat_t.sqrt() * tensor
            + (1 - alpha_hat_t).sqrt() * torch.randn_like(tensor.float()).to(tensor.device) * self.n_std
        )

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class CutOUT(DataAugmentation):
    def __init__(
        self,
        c_prob: float,
        cutout_size: int,
        value: float,
        groups: list[str] | None = None,
    ) -> None:
        super().__init__(groups)
        self.c_prob = c_prob
        self.cutout_size = cutout_size
        self.centers: dict[int, list[torch.Tensor]] = {}
        self.value = value

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        self.centers[index] = [torch.rand((3) if len(shape) == 3 else (2)) for shape in shapes]
        return shapes

    # WHOLE_VOLUME on purpose: the cutout box is normalised to the extent of the tensor in hand, so
    # applied per patch it would land in every patch instead of once in the volume.
    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        center = self.centers[index][a]
        masks = []
        for i, w in enumerate(tensor.shape[1:]):
            re = [1] * i + [-1] + [1] * (len(tensor.shape[1:]) - i - 1)
            masks.append(
                ((torch.arange(w).reshape(re) + 0.5) / w - center[i].reshape([1, 1])).abs()
                >= torch.tensor(self.cutout_size).reshape([1, 1]) / 2
            )
        result = masks[0]
        for mask in masks[1:]:
            result = torch.logical_or(result, mask)
        result = result.unsqueeze(0).repeat([tensor.shape[0], *[1 for _ in range(len(tensor.shape) - 1)]])
        return torch.where(
            result.to(tensor.device) == 1,
            tensor,
            torch.tensor(self.value).to(tensor.device),
        )

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class Elastix(DataAugmentation):
    def __init__(self, grid_spacing: int = 16, max_displacement: int = 16) -> None:
        _require_simpleitk()
        super().__init__()
        self.grid_spacing = grid_spacing
        self.max_displacement = max_displacement
        self.displacement_fields: dict[int, list[torch.Tensor]] = {}
        self.displacement_fields_true: dict[int, list[torch.Tensor]] = {}

    @staticmethod
    def _format_loc(new_locs, shape):
        for i in range(len(shape)):
            new_locs[..., i] = 2 * (new_locs[..., i] / (shape[i] - 1) - 0.5)
        new_locs = new_locs[..., list(reversed(range(len(shape))))]
        return new_locs

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        print(f"[KonfAI] Compute Displacement Field for index {index}")
        self.displacement_fields[index] = []
        self.displacement_fields_true[index] = []
        for i, (shape, cache_attribute) in enumerate(zip(shapes, caches_attribute, strict=False)):
            shape = shape
            dim = len(shape)
            if "Spacing" not in cache_attribute:
                spacing = np.array([1.0 for _ in range(dim)])
            else:
                spacing = cache_attribute.get_np_array("Spacing")

            grid_physical_spacing = [self.grid_spacing] * dim
            image_physical_size = [size * spacing for size, spacing in zip(shape, spacing, strict=False)]
            mesh_size = [
                int(image_size / grid_spacing + 0.5)
                for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing, strict=False)
            ]
            if "Spacing" not in cache_attribute:
                cache_attribute["Spacing"] = np.array([1.0 for _ in range(dim)])
            if "Origin" not in cache_attribute:
                cache_attribute["Origin"] = np.array([1.0 for _ in range(dim)])
            if "Direction" not in cache_attribute:
                cache_attribute["Direction"] = np.eye(dim).flatten()

            ref_image = data_to_image(np.expand_dims(np.zeros(shape), 0), cache_attribute)

            bspline_transform = sitk.BSplineTransformInitializer(
                image1=ref_image, transformDomainMeshSize=mesh_size, order=3
            )
            displacement_filter = sitk.TransformToDisplacementFieldFilter()
            displacement_filter.SetReferenceImage(ref_image)

            vectors = [torch.arange(0, s) for s in shape]
            grids = torch.meshgrid(vectors, indexing="ij")
            grid = torch.stack(grids)
            grid = torch.unsqueeze(grid, 0)
            grid = grid.type(torch.float).permute([0] + [i + 2 for i in range(len(shape))] + [1])

            control_points = torch.rand(*[size + 3 for size in mesh_size], dim)
            control_points -= 0.5
            control_points *= 2 * self.max_displacement
            bspline_transform.SetParameters(control_points.flatten().tolist())
            displacement = sitk.GetArrayFromImage(displacement_filter.Execute(bspline_transform))
            self.displacement_fields_true[index].append(displacement)
            new_locs = grid + torch.unsqueeze(torch.from_numpy(displacement), 0).type(torch.float32)
            self.displacement_fields[index].append(Elastix._format_loc(new_locs, shape))
            print(f"[KonfAI] Compute in progress : {(i + 1) / len(shapes) * 100:.2f} %")
        return shapes

    # WHOLE_VOLUME on purpose: _state_init materialises the displacement field at the full shape and
    # indexes it by absolute position, so streaming the image saves nothing while the field is
    # resident.
    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        # Integer tensors are label maps: nearest-neighbour keeps class ids intact.
        mode = "nearest" if not tensor.dtype.is_floating_point else "bilinear"
        return (
            F.grid_sample(
                tensor.type(torch.float32).unsqueeze(0),
                self.displacement_fields[index][a].to(tensor.device),
                align_corners=True,
                mode=mode,
                padding_mode="border",
            )
            .type(tensor.dtype)
            .squeeze(0)
        )

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Elastix augmentation has no inverse; do not use it for invertible TTA.")


class Permute(DataAugmentation):
    def __init__(self, prob_permute: list[float] | None = [0.5, 0.5]) -> None:
        super().__init__()
        self._permute_dims = torch.tensor([[0, 2, 1, 3], [0, 3, 1, 2]])
        self.prob_permute = prob_permute
        self.permute: dict[int, torch.Tensor] = {}

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        if len(shapes):
            dim = len(shapes[0])
            if dim != 3:
                raise ValueError("The permute augmentation only support 3D images")
            if self.prob_permute:
                if len(self.prob_permute) != 2:
                    raise ValueError("Size of prob_permute must be equal 2")
                self.permute[index] = torch.rand((len(shapes), len(self.prob_permute))) < torch.tensor(
                    self.prob_permute
                )
            else:
                if len(shapes) != 2:
                    raise ValueError("The number of augmentation images must be equal to 2")
                self.permute[index] = torch.eye(2, dtype=torch.bool)
            for i in range(len(shapes)):
                shapes[i] = [shapes[i][axis] for axis in self._source_axes(index, i)]
        return shapes

    def _source_axes(self, index: int, a: int) -> list[int]:
        """Which source spatial axis each output spatial axis is drawn from, for copy *a*."""
        axes = list(range(3))
        for permute in self._permute_dims[self.permute[index][a]]:
            axes = [axes[dim - 1] for dim in permute[1:]]
        return axes

    def _patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        # Reordering axes moves every voxel and touches none, so the multiset of values is the input's:
        # a bijection, which is what ORIENTATION promises.
        return PatchLocality(LocalityKind.ORIENTATION)

    def _stream_shape(self, index: int, a: int, shape: list[int]) -> list[int]:
        # The same reorder state_init applied to the copy's grid.
        return [shape[axis] for axis in self._source_axes(index, a)]

    def _stream_region_source(
        self,
        index: int,
        a: int,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
    ) -> list[slice]:
        # Output axis k is source axis ``_source_axes()[k]``, so placing each target slice back on its
        # source axis gives the region whose permutation is the target patch.
        source_slices = [slice(0, n) for n in source_spatial_shape]
        for k, sl in enumerate(target_slices):
            source_slices[self._source_axes(index, a)[k]] = slice(sl.start, sl.stop)
        return source_slices

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        for permute in self._permute_dims[self.permute[index][a]]:
            tensor = tensor.permute(tuple(permute))
        return tensor

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        for permute in reversed(self._permute_dims[self.permute[index][a]]):
            tensor = tensor.permute(tuple(np.argsort(permute)))
        return tensor


class Mask(DataAugmentation):
    def __init__(self, mask: str, value: float, groups: list[str] | None = None) -> None:
        _require_simpleitk()
        super().__init__(groups)
        self.mask_path = Path(mask)
        if not self.mask_path.is_file():
            raise AugmentationError(f"Mask file '{self.mask_path}' does not exist.")
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(self.mask_path))
        reader.ReadImageInformation()
        self.mask_shape = tuple(reversed(reader.GetSize()))
        self._mask: torch.Tensor | None = None
        self.positions: dict[int, list[torch.Tensor]] = {}
        self.value = value

    def _load_mask(self) -> torch.Tensor:
        if self._mask is None:
            self._mask = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(str(self.mask_path))))
        return self._mask

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        self.positions[index] = [
            torch.rand((3) if len(shape) == 3 else (2))
            * (torch.tensor([max(s1 - s2, 0) for s1, s2 in zip(torch.tensor(shape), self.mask_shape, strict=False)]))
            for shape in shapes
        ]
        return [list(self.mask_shape) for _ in shapes]

    # WHOLE_VOLUME on purpose: the output grid is the mask's, and the mask volume is already resident
    # at that extent -- there is no whole-volume read left for a declaration to save.
    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        mask = self._load_mask()
        position = self.positions[index][a]
        slices = [slice(None, None)] + [
            slice(int(s1), int(s1) + s2) for s1, s2 in zip(position, mask.shape, strict=False)
        ]
        padding = []
        for s1, s2 in zip(reversed(tensor.shape), reversed(mask.shape), strict=False):
            padding.append(0)
            padding.append(s2 - s1 if s1 < s2 else 0)
        value = (
            torch.tensor(0, dtype=torch.uint8)
            if tensor.dtype == torch.uint8
            else torch.tensor(self.value).to(tensor.device)
        )
        return torch.where(
            mask.to(tensor.device) == 1,
            torch.nn.functional.pad(tensor, tuple(padding), mode="constant", value=value)[tuple(slices)],
            value,
        )

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Mask augmentation has no inverse; do not use it for invertible TTA.")

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

from abc import ABC, abstractmethod
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
from konfai.utils.config import apply_config
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
            data_augmentation: DataAugmentation = apply_config(
                f"{konfai_root()}.Dataset.augmentations.{key}.data_augmentations.{augmentation}"
            )(getattr(module, name))()
            data_augmentation.load(prob.prob)
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
        """
        Initialize cached augmentation state for the selected copies and update their shapes.
        
        Parameters:
        	index (int): Dataset case index.
        	shapes (list[list[int]]): Spatial shapes for the copies being initialized.
        	caches_attribute (list[Attribute]): Cached dataset attributes associated with the copies.
        
        Returns:
        	list[list[int]]: Updated spatial shapes.
        """
        pass

    def patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        """
        Determine the input locality of an augmentation for a dataset copy.
        
        Args:
            index (int): Case index.
            a (int): Dataset copy index.
            cache_attribute (Attribute): Cached attributes for the case.
        
        Returns:
            PatchLocality: The locality of the selected augmentation, or pointwise locality when the copy is unchanged.
        """
        if a not in self.who_index[index]:
            return PatchLocality(LocalityKind.POINTWISE)
        return self._patch_locality(index, self.who_index[index].index(a), cache_attribute)

    def _patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        """Return the locality classification for a selected augmentation copy.
        
        Returns:
            PatchLocality: Whole-volume locality.
        """
        return PatchLocality(LocalityKind.WHOLE_VOLUME)

    def stream_region_source(
        self,
        index: int,
        a: int,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
    ) -> list[slice]:
        """
        Map target patch slices to the source region required to produce them.
        
        Parameters:
            index (int): Case index.
            a (int): Copy index.
            target_slices (tuple[slice, ...]): Spatial slices of the target patch.
            source_spatial_shape (list[int]): Spatial dimensions of the source volume.
        
        Returns:
            list[slice]: Source-region slices corresponding to the target patch.
        """
        return self._stream_region_source(index, self.who_index[index].index(a), target_slices, source_spatial_shape)

    def _stream_region_source(
        self,
        index: int,
        a: int,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
    ) -> list[slice]:
        """
        Map target patch slices to source-region slices for region-local augmentations.
        
        Parameters:
        	index (int): Case index associated with the cached augmentation state.
        	a (int): Copy index.
        	target_slices (tuple[slice, ...]): Spatial slices of the requested target patch.
        	source_spatial_shape (list[int]): Spatial dimensions of the source volume.
        
        Raises:
        	AugmentationError: If the augmentation declares region locality without providing a source-region mapping.
        """
        raise AugmentationError(
            f"{type(self).__name__} declared a region patch-locality but does not implement _stream_region_source().",
            "Implement _stream_region_source() or declare a non-region _patch_locality().",
        )

    def compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the augmentation to one tensor for a selected copy.
        
        Args:
            name: Name of the tensor being augmented.
            index: Dataset case index.
            a: Copy index.
            tensor: Input tensor.
        
        Returns:
            The augmented tensor, or the input tensor if the copy is not selected.
        """
        if a in self.who_index[index]:
            tensor = self._compute(name, index, self.who_index[index].index(a), tensor)
        return tensor

    def __call__(
        self,
        name: str,
        index: int,
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Apply the augmentation independently to each tensor in a collection.
        
        Parameters:
        	name (str): Name identifying the tensor collection.
        	index (int): Case or volume index used for cached augmentation state.
        	tensors (list[torch.Tensor]): Tensors to transform.
        
        Returns:
        	list[torch.Tensor]: Transformed tensors in the original order.
        """
        return [self.compute(name, index, a, tensor) for a, tensor in enumerate(tensors)]

    @abstractmethod
    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the augmentation to one selected tensor.
        
        Parameters:
        	name (str): Name of the tensor being augmented.
        	index (int): Dataset case or volume index.
        	a (int): Copy index within the case.
        	tensor (torch.Tensor): Tensor to transform.
        
        Returns:
        	torch.Tensor: Transformed tensor.
        """
        pass

    def inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the inverse transformation to a selected augmentation copy.
        
        Parameters:
        	index (int): Case index whose cached augmentation state is used.
        	a (int): Copy index to transform.
        	tensor (torch.Tensor): Tensor to transform.
        
        Returns:
        	torch.Tensor: The inversely transformed tensor, or the input tensor if the copy was not selected.
        """
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
        """
        Retrieve the cached affine transformation matrix for a dataset case and copy.
        
        Parameters:
            index (int): Dataset case index.
            a (int): Copy index.
        
        Returns:
            torch.Tensor: Cached affine transformation matrix.
        """
        return self.matrix[index][a]

    def _sample(self, matrix: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        # Integer tensors are label maps: interpolating them blends class ids into
        # non-existent labels, so resample them with nearest-neighbour instead.
        """Resample a tensor using an affine transformation matrix.
        
        Parameters:
            matrix (torch.Tensor): Homogeneous affine transformation matrix.
            tensor (torch.Tensor): Tensor to resample.
        
        Returns:
            torch.Tensor: The resampled tensor with the input shape and data type.
        """
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
        """
        Resample a tensor using the cached transformation for a dataset copy.
        
        Returns:
            torch.Tensor: The transformed tensor.
        """
        return self._sample(self._grid_matrix(index, a, list(tensor.shape[1:])), tensor)

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the inverse spatial transformation to a tensor.
        
        Parameters:
        	index (int): Case index associated with the cached transformation.
        	a (int): Copy index associated with the cached transformation.
        	tensor (torch.Tensor): Tensor to transform.
        
        Returns:
        	torch.Tensor: Tensor resampled using the inverse transformation.
        """
        return self._sample(self._grid_matrix(index, a, list(tensor.shape[1:])).inverse(), tensor)


class Translate(EulerTransform):
    def __init__(self, t_min: float = -10, t_max=10, is_int: bool = False):
        """Initialize a translation augmentation with the specified displacement range.
        
        Parameters:
        	t_min (float): Minimum translation value for each spatial axis.
        	t_max (float): Maximum translation value for each spatial axis.
        	is_int (bool): Whether to restrict sampled translations to integer values.
        """
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max
        self.is_int = is_int
        self.translate: dict[int, torch.Tensor] = {}

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        """
        Sample and cache per-copy translation vectors for a dataset case.
        
        Parameters:
            index (int): Case index whose translations are cached.
            shapes (list[list[int]]): Spatial shapes used to determine the translation dimensionality.
            caches_attribute (list[Attribute]): Cached attributes associated with the selected copies.
        
        Returns:
            list[list[int]]: The input shapes unchanged.
        """
        dim = len(shapes[0])
        translate = torch.rand((len(shapes), dim)) * torch.tensor(self.t_max - self.t_min) + torch.tensor(self.t_min)
        self.translate[index] = torch.round(translate) if self.is_int else translate
        return shapes

    def _grid_matrix(self, index: int, a: int, shape: list[int]) -> torch.Tensor:
        # The draw is a shift in VOXELS, in (x, y, z). ``affine_grid`` spans [-1, 1] over whatever
        # extent it is given, so the same shift is a different matrix on a patch than on the volume:
        # normalise it against the extent it is about to be applied to, never against a fixed one.
        """
        Build the normalized affine translation matrix for a selected augmentation copy.
        
        Parameters:
            index (int): Case index containing the cached translation.
            a (int): Copy index whose translation is used.
            shape (list[int]): Spatial extent to which the translation is applied.
        
        Returns:
            torch.Tensor: Batched 2D or 3D affine translation matrix normalized to the given spatial extent.
        """
        func = _translate_3d_matrix if len(shape) == 3 else _translate_2d_matrix
        sizes = torch.tensor(list(reversed(shape)), dtype=torch.float32)
        return torch.unsqueeze(func(self.translate[index][a] * 2.0 / (sizes - 1)), dim=0)

    def _patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        # A uniform shift sends a target patch to that same patch displaced by the draw, so the source
        # is a bounded neighbourhood of it. One voxel past the ceiling covers the far tap a fractional
        # shift interpolates from. The draw is in (x, y, z); a halo is in array order.
        """
        Return the bounded source-region locality for a translated patch.
        
        Parameters:
        	index (int): Case index identifying the cached translation.
        	a (int): Copy index identifying the translation to use.
        	cache_attribute (Attribute): Cached dataset attribute associated with the patch.
        
        Returns:
        	PatchLocality: A halo locality whose radius covers the absolute translation and interpolation extent.
        """
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
        """
        Initialize per-copy rotation matrices and resulting spatial shapes.
        
        Parameters:
        	index (int): Case index used to store the sampled rotation matrices.
        	shapes (list[list[int]]): Spatial shapes for the selected copies.
        	caches_attribute (list[Attribute]): Cached attributes associated with the copies.
        
        Returns:
        	list[list[int]]: Spatial shapes after applying the sampled rotations.
        """
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
        """
        Determine whether a rotation can be represented by axis permutation and reflection.
        
        Parameters:
            matrix (torch.Tensor): Homogeneous rotation matrix to analyze.
        
        Returns:
            tuple[list[int], list[int]] | None: Permutation dimensions and flipped axes
            when the matrix is an exact signed permutation; otherwise, ``None``.
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
        """
        Determine the spatial shape produced by applying a rotation matrix.
        
        Parameters:
        	matrix (torch.Tensor): Rotation matrix used to determine axis remapping.
        	shape (list[int]): Input spatial dimensions.
        
        Returns:
        	list[int]: Remapped spatial dimensions for an exact axis permutation, or the
        		input shape when interpolation is required.
        """
        remap = cls._index_remap(matrix)
        if remap is None:
            return list(shape)
        dims, _ = remap
        return [shape[dim - 1] for dim in dims[1:]]

    def _reorient(self, index: int, a: int, matrix: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply a rotation to a tensor using exact axis remapping when possible and resampling otherwise.
        
        Parameters:
        	index (int): Case index associated with the cached transformation.
        	a (int): Copy index associated with the cached transformation.
        	matrix (torch.Tensor): Rotation matrix to apply.
        	tensor (torch.Tensor): Tensor to transform.
        
        Returns:
        	torch.Tensor: The reoriented tensor.
        """
        remap = Rotate._index_remap(matrix)
        if remap is None:
            return self._sample(matrix, tensor)
        dims, flips = remap
        # flip materialises the permuted view, so the copy never aliases the tensor it was drawn from.
        return tensor.permute(dims).flip(flips)

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the cached forward spatial transform to a tensor.
        
        Parameters:
            index (int): Case index associated with the cached transform.
            a (int): Copy index associated with the cached transform.
        
        Returns:
            torch.Tensor: The transformed tensor.
        """
        return self._reorient(index, a, self._grid_matrix(index, a, list(tensor.shape[1:])), tensor)

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the inverse rotation or spatial reorientation to a tensor.
        
        Parameters:
            index (int): Case index associated with the cached transformation.
            a (int): Copy index associated with the cached transformation.
            tensor (torch.Tensor): Tensor to reorient.
        
        Returns:
            torch.Tensor: Tensor transformed using the inverse cached rotation.
        """
        return self._reorient(index, a, self._grid_matrix(index, a, list(tensor.shape[1:])).inverse(), tensor)

    def _patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        # Permuting and mirroring voxels is a bijection on them, which is what ORIENTATION promises and
        # what LocalityKind.preserves_statistics lets a later stage trust. Only the draw can say whether
        # this one is that, and the draw is a property of the copy rather than of the case.
        """
        Determine the locality required for a rotation applied to a specific copy.
        
        Parameters:
            index (int): Case index.
            a (int): Copy index.
            cache_attribute (Attribute): Cached dataset attribute associated with the copy.
        
        Returns:
            PatchLocality: Orientation locality for exact axis permutations and reflections;
                whole-volume locality for other rotations.
        """
        if Rotate._index_remap(self.matrix[index][a]) is None:
            return PatchLocality(LocalityKind.WHOLE_VOLUME)
        return PatchLocality(LocalityKind.ORIENTATION)

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
        """
        Map a target patch to the source slices required for an exactly remappable rotation.
        
        Parameters:
            index (int): Case index for the cached rotation.
            a (int): Copy index.
            target_slices (tuple[slice, ...]): Spatial slices of the requested output patch.
            source_spatial_shape (list[int]): Spatial dimensions of the source tensor.
        
        Returns:
            list[slice]: Source slices that produce the target patch after rotation.
        
        Raises:
            AugmentationError: If the cached rotation cannot be represented as an exact axis permutation and flip.
        """
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
        """
        Flip the tensor along the specified spatial dimensions, negating corresponding vector components when enabled.
        
        Parameters:
        	tensor (torch.Tensor): Tensor to transform.
        	dims (list[int]): Spatial dimensions along which to flip the tensor.
        
        Returns:
        	torch.Tensor: Flipped tensor, with affected vector components negated when `vector_field` is enabled.
        """
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
        """Classify the locality required by the flip operation.
        
        Returns:
            PatchLocality: Whole-volume locality when vector-field components are
                negated; orientation locality otherwise.
        """
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
        """
        Map target patch slices to the source regions required by spatial flipping.
        
        Parameters:
            index (int): Case index identifying the cached flip configuration.
            a (int): Copy index within the case.
            target_slices (tuple[slice, ...]): Spatial slices of the requested target patch.
            source_spatial_shape (list[int]): Sizes of the source spatial dimensions.
        
        Returns:
            list[slice]: Source slices, mirrored along flipped axes and unchanged along other axes.
        """
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
        """Apply the configured axis flips to a tensor.
        
        Parameters:
        	name (str): Name associated with the tensor.
        	index (int): Case index for the cached augmentation state.
        	a (int): Copy index within the case.
        	tensor (torch.Tensor): Tensor to transform.
        
        Returns:
        	torch.Tensor: Tensor after applying the configured flips.
        """
        return self._flip(tensor, self.flip[index][a])

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the configured flips to restore the tensor orientation.
        
        Parameters:
        	index (int): Case index associated with the cached flip configuration.
        	a (int): Copy index within the case.
        	tensor (torch.Tensor): Tensor to transform.
        
        Returns:
        	torch.Tensor: Tensor with the configured spatial flips applied.
        """
        return self._flip(tensor, self.flip[index][a])


class ColorTransform(DataAugmentation):
    def __init__(self, groups: list[str] | None = None) -> None:
        super().__init__(groups)
        self.matrix: dict[int, list[torch.Tensor]] = {}

    def _patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        # The draw is a colour matrix applied to each voxel on its own: no neighbour, no coordinate,
        # no extent. Whatever region a voxel is read in, it comes out the same.
        """Classifies the augmentation as pointwise because each output voxel depends only on the corresponding input voxel.
        
        Returns:
        	PatchLocality: Pointwise locality.
        """
        return PatchLocality(LocalityKind.POINTWISE)

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the cached color transformation to an image tensor.
        
        Returns:
            torch.Tensor: The transformed image tensor with the same shape as the input.
        
        Raises:
            AugmentationError: If the input does not have one or three channels.
        """
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
        """
        Initialize a noise timestep for each selected copy.
        
        Parameters:
        	index (int): Dataset case index used to store the sampled timesteps.
        	shapes (list[list[int]]): Shapes associated with the selected copies.
        
        Returns:
        	list[list[int]]: The input shapes unchanged.
        """
        if int(self.max_T) == 0:
            self.ts[index] = [0 for _ in shapes]
        else:
            self.ts[index] = [torch.randint(0, int(self.max_T), (1,)) for _ in shapes]
        return shapes

    # WHOLE_VOLUME on purpose: the noise field is drawn per call, not per voxel position, so two
    # overlapping patches would sample unrelated fields and the overlap blend would suppress the
    # variance this exists to add.
    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        """Apply scheduled Gaussian noise to a tensor.
        
        Parameters:
            name (str): Name associated with the computation.
            index (int): Dataset case index.
            a (int): Selected copy index.
            tensor (torch.Tensor): Input tensor to augment.
        
        Returns:
            torch.Tensor: Tensor combined with Gaussian noise according to the cached timestep.
        """
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
        """
        Initialize random cutout centers for each selected copy.
        
        Parameters:
            shapes (list[list[int]]): Spatial shapes used to determine the center dimensionality.
        
        Returns:
            list[list[int]]: The input shapes unchanged.
        """
        self.centers[index] = [torch.rand((3) if len(shape) == 3 else (2)) for shape in shapes]
        return shapes

    # WHOLE_VOLUME on purpose: the cutout box is normalised to the extent of the tensor in hand, so
    # applied per patch it would land in every patch instead of once in the volume.
    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the configured cutout mask to a tensor.
        
        Parameters:
        	name (str): Name associated with the computation.
        	index (int): Dataset or case index identifying cached augmentation state.
        	a (int): Copy index identifying the cached cutout center.
        	tensor (torch.Tensor): Tensor to transform.
        
        Returns:
        	torch.Tensor: Tensor with values outside the retained region replaced by the configured cutout value.
        """
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
        """
        Generate and cache a random displacement field for each selected shape.
        
        Parameters:
            index (int): Case index used to identify the cached displacement fields.
            shapes (list[list[int]]): Spatial shapes for the selected copies.
            caches_attribute (list[Attribute]): Metadata used to define image spacing, origin, and direction.
        
        Returns:
            list[list[int]]: The input shapes unchanged.
        """
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
        """Resample a tensor using the cached displacement field for the selected augmentation copy.
        
        Parameters:
            name (str): Name associated with the tensor.
            index (int): Dataset case index.
            a (int): Augmentation copy index.
            tensor (torch.Tensor): Tensor to resample.
        
        Returns:
            torch.Tensor: The resampled tensor with the original shape and data type.
        """
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
        """Indicate that Elastix does not support inverse transformation.
        
        Raises:
            NotImplementedError: Always raised because Elastix transformations are not invertible.
        """
        raise NotImplementedError("Elastix augmentation has no inverse; do not use it for invertible TTA.")


class Permute(DataAugmentation):
    def __init__(self, prob_permute: list[float] | None = [0.5, 0.5]) -> None:
        super().__init__()
        self._permute_dims = torch.tensor([[0, 2, 1, 3], [0, 3, 1, 2]])
        self.prob_permute = prob_permute
        self.permute: dict[int, torch.Tensor] = {}

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        """
        Initialize permutation choices and update shapes for selected 3D augmentations.
        
        Parameters:
            index (int): Case index used to store the permutation choices.
            shapes (list[list[int]]): Spatial shapes to update according to the selected axis permutations.
            caches_attribute (list[Attribute]): Cached dataset attributes associated with the shapes.
        
        Returns:
            list[list[int]]: The shapes after applying the selected axis permutations.
        
        Raises:
            ValueError: If the shapes are not 3D, `prob_permute` does not contain two probabilities, or exactly two augmentation shapes are not provided when probabilities are unset.
        """
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
        """
        Determine which input spatial axis supplies each output spatial axis.
        
        Parameters:
            index (int): Case index used to select the stored permutation.
            a (int): Copy index within the case.
        
        Returns:
            list[int]: Input spatial-axis indices corresponding to the three output axes.
        """
        axes = list(range(3))
        for permute in self._permute_dims[self.permute[index][a]]:
            axes = [axes[dim - 1] for dim in permute[1:]]
        return axes

    def _patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        # Reordering axes moves every voxel and touches none, so the multiset of values is the input's:
        # a bijection, which is what ORIENTATION promises.
        """Return the locality classification for an axis-reordering augmentation.
        
        Returns:
        	PatchLocality: Orientation locality, because the transformation reorders voxels without changing their values.
        """
        return PatchLocality(LocalityKind.ORIENTATION)

    def _stream_region_source(
        self,
        index: int,
        a: int,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
    ) -> list[slice]:
        # Output axis k is source axis ``_source_axes()[k]``, so placing each target slice back on its
        # source axis gives the region whose permutation is the target patch.
        """
        Map a target patch region to the corresponding source region for the selected permutation.
        
        Parameters:
            index (int): Case index identifying the cached permutation.
            a (int): Copy index.
            target_slices (tuple[slice, ...]): Spatial slices defining the target patch.
            source_spatial_shape (list[int]): Spatial dimensions of the source tensor.
        
        Returns:
            list[slice]: Source-axis slices whose permutation produces the target region.
        """
        source_slices = [slice(0, n) for n in source_spatial_shape]
        for k, sl in enumerate(target_slices):
            source_slices[self._source_axes(index, a)[k]] = slice(sl.start, sl.stop)
        return source_slices

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the configured axis permutation to a tensor.
        
        Parameters:
        	name (str): Name associated with the computation.
        	index (int): Dataset or case index.
        	a (int): Copy index.
        
        Returns:
        	torch.Tensor: The tensor with its spatial axes permuted.
        """
        for permute in self._permute_dims[self.permute[index][a]]:
            tensor = tensor.permute(tuple(permute))
        return tensor

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        """Restore a tensor to its original axis order after the stored permutation.
        
        Parameters:
            index (int): Case index identifying the cached permutation.
            a (int): Copy index identifying the permutation to reverse.
            tensor (torch.Tensor): Permuted tensor.
        
        Returns:
            torch.Tensor: Tensor with the stored axis permutations reversed.
        """
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
        """
        Initialize mask placement positions and return the resulting spatial shapes.
        
        Parameters:
        	index (int): Case index used to store the sampled positions.
        	shapes (list[list[int]]): Input spatial shapes used to determine valid mask positions.
        
        Returns:
        	list[list[int]]: One mask-shaped spatial dimension list for each input shape.
        """
        self.positions[index] = [
            torch.rand((3) if len(shape) == 3 else (2))
            * (torch.tensor([max(s1 - s2, 0) for s1, s2 in zip(torch.tensor(shape), self.mask_shape, strict=False)]))
            for shape in shapes
        ]
        return [list(self.mask_shape) for _ in shapes]

    # WHOLE_VOLUME on purpose: the output grid is the mask's, and the mask volume is already resident
    # at that extent -- there is no whole-volume read left for a declaration to save.
    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        """
        Overlay the tensor onto the configured mask region.
        
        Parameters:
            name (str): Name of the tensor being transformed.
            index (int): Dataset case index associated with the cached mask position.
            a (int): Copy index associated with the cached mask position.
            tensor (torch.Tensor): Tensor to place within the mask extent.
        
        Returns:
            torch.Tensor: Tensor with original values where the mask equals 1 and the replacement value elsewhere.
        """
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
        """Raise an error because mask overlay cannot be inverted.
        
        Parameters:
            index (int): Case index associated with the augmentation state.
            a (int): Copy index associated with the augmentation state.
            tensor (torch.Tensor): Tensor to be inverse-transformed.
        
        Raises:
            NotImplementedError: Always raised because the mask overlay operation is not invertible.
        """
        raise NotImplementedError("Mask augmentation has no inverse; do not use it for invertible TTA.")

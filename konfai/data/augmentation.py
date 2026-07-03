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
        pass

    def __call__(
        self,
        name: str,
        index: int,
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if len(self.who_index[index]) > 0:
            for i, result in enumerate(self._compute(name, index, [tensors[i] for i in self.who_index[index]])):
                tensors[self.who_index[index][i]] = result
        return tensors

    @abstractmethod
    def _compute(
        self,
        name: str,
        index: int,
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
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

    def _compute(
        self,
        name: str,
        index: int,
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        results = []
        for tensor, matrix in zip(tensors, self.matrix[index], strict=False):
            # Integer tensors are label maps: interpolating them blends class ids into
            # non-existent labels, so resample them with nearest-neighbour instead.
            mode = "nearest" if not tensor.dtype.is_floating_point else "bilinear"
            results.append(
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
        return results

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        mode = "nearest" if not tensor.dtype.is_floating_point else "bilinear"
        return (
            F.grid_sample(
                tensor.unsqueeze(0).type(torch.float32),
                F.affine_grid(
                    self.matrix[index][a].inverse()[:, :-1, ...],
                    [1, *list(tensor.shape)],
                    align_corners=True,
                ).to(tensor.device),
                align_corners=True,
                mode=mode,
                padding_mode="reflection",
            )
            .type(tensor.dtype)
            .squeeze(0)
        )


class Translate(EulerTransform):
    def __init__(self, t_min: float = -10, t_max=10, is_int: bool = False):
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max
        self.is_int = is_int

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        dim = len(shapes[0])
        func = _translate_3d_matrix if dim == 3 else _translate_2d_matrix
        translate = torch.rand((len(shapes), dim)) * torch.tensor(self.t_max - self.t_min) + torch.tensor(self.t_min)
        if self.is_int:
            translate = torch.round(translate)
        matrices = []
        for value, shape in zip(translate, shapes, strict=False):
            sizes = torch.tensor(list(reversed(shape)), dtype=torch.float32)
            normalized = value * 2.0 / (sizes - 1)
            matrices.append(torch.unsqueeze(func(normalized), dim=0))
        self.matrix[index] = matrices
        return shapes


class Rotate(EulerTransform):
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
        return shapes


class Scale(EulerTransform):
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

    def _compute(
        self,
        name: str,
        index: int,
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        results = []
        for tensor, flip in zip(tensors, self.flip[index], strict=False):
            results.append(self._flip(tensor, flip))
        return results

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return self._flip(tensor, self.flip[index][a])


class ColorTransform(DataAugmentation):
    def __init__(self, groups: list[str] | None = None) -> None:
        super().__init__(groups)
        self.matrix: dict[int, list[torch.Tensor]] = {}

    def _compute(
        self,
        name: str,
        index: int,
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        results = []
        for tensor, matrix in zip(tensors, self.matrix[index], strict=False):
            result = tensor.reshape([*tensor.shape[:1], int(np.prod(tensor.shape[1:]))])
            if tensor.shape[0] == 3:
                matrix = matrix.to(tensor.device)
                result = matrix[:, :3, :3] @ result.float() + matrix[:, :3, 3:]
            elif tensor.shape[0] == 1:
                matrix = matrix[:, :3, :].mean(dim=1, keepdims=True).to(tensor.device)
                result = result.float() * matrix[:, :, :3].sum(dim=2, keepdims=True) + matrix[:, :, 3:]
            else:
                raise AugmentationError("Image must be RGB (3 channels) or L (1 channel)")
            results.append(result.reshape(tensor.shape))
        return results

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
        self.matrix[index] = [torch.unsqueeze(_rotation_3d_matrix(value.repeat(3), self.v), dim=0) for value in theta]
        return shapes


class Saturation(ColorTransform):
    def __init__(self, s_std: float, groups: list[str] | None = None) -> None:
        super().__init__(groups)
        self.s_std = s_std
        self.v = torch.tensor([1, 1, 1, 0]) / torch.sqrt(torch.tensor(3))

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        saturation = torch.exp2(torch.randn(len(shapes)) * self.s_std)
        self.matrix[index] = [
            (self.v.ger(self.v) + (torch.eye(4) - self.v.ger(self.v))).unsqueeze(0) * value for value in saturation
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

    def _compute(
        self,
        name: str,
        index: int,
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        results = []
        for tensor, t in zip(tensors, self.ts[index], strict=False):
            alpha_hat_t = self.alpha_hat[t].to(tensor.device).reshape(*[1 for _ in range(len(tensor.shape))])
            results.append(
                alpha_hat_t.sqrt() * tensor
                + (1 - alpha_hat_t).sqrt() * torch.randn_like(tensor.float()).to(tensor.device) * self.n_std
            )
        return results

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

    def _compute(
        self,
        name: str,
        index: int,
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        results = []
        for tensor, center in zip(tensors, self.centers[index], strict=False):
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
            results.append(
                torch.where(
                    result.to(tensor.device) == 1,
                    tensor,
                    torch.tensor(self.value).to(tensor.device),
                )
            )
        return results

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

    def _compute(
        self,
        name: str,
        index: int,
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        results = []
        for tensor, displacement_field in zip(tensors, self.displacement_fields[index], strict=False):
            # Integer tensors are label maps: nearest-neighbour keeps class ids intact.
            mode = "nearest" if not tensor.dtype.is_floating_point else "bilinear"
            results.append(
                F.grid_sample(
                    tensor.type(torch.float32).unsqueeze(0),
                    displacement_field.to(tensor.device),
                    align_corners=True,
                    mode=mode,
                    padding_mode="border",
                )
                .type(tensor.dtype)
                .squeeze(0)
            )
        return results

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        pass


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
            for i, prob in enumerate(self.permute[index]):
                for permute in self._permute_dims[prob]:
                    shapes[i] = [shapes[i][dim - 1] for dim in permute[1:]]
        return shapes

    def _compute(
        self,
        name: str,
        index: int,
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        results = []
        for tensor, prob in zip(tensors, self.permute[index], strict=False):
            res = tensor
            for permute in self._permute_dims[prob]:
                res = res.permute(tuple(permute))
            results.append(res)
        return results

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

    def _compute(
        self,
        name: str,
        index: int,
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        mask = self._load_mask()
        results = []
        for tensor, position in zip(tensors, self.positions[index], strict=False):
            slices = [slice(None, None)] + [
                slice(int(s1), int(s1) + s2) for s1, s2 in zip(position, mask.shape, strict=False)
            ]
            padding = []
            for s1, s2 in zip(reversed(tensor.shape), reversed(mask.shape), strict=False):
                if s1 < s2:
                    pad = s2 - s1
                else:
                    pad = 0
                padding.append(0)
                padding.append(pad)
            value = (
                torch.tensor(0, dtype=torch.uint8)
                if tensor.dtype == torch.uint8
                else torch.tensor(self.value).to(tensor.device)
            )
            results.append(
                torch.where(
                    mask.to(tensor.device) == 1,
                    torch.nn.functional.pad(tensor, tuple(padding), mode="constant", value=value)[tuple(slices)],
                    value,
                )
            )
        return results

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        pass

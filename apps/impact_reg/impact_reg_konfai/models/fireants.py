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
#
# This wrapper does NOT copy any FireANTs source: it only calls the public FireANTs API of the
# separately-installed ``fireants`` wheel (PyPI). FireANTs is distributed under the FireANTs License
# v1.0 and must be cited: see the NOTICE file in this directory for the license, copyright and
# bibliography that ship with this app.

"""FireANTs registration as a self-contained KonfAI model (shared by the FireANTs presets).

Same idiomatic ``add_module`` graph and the same output contract as the ConvexAdam preset
(``DisplacementField`` on the FIXED grid), so the
orchestrator / app.json / ensemble / uncertainty are unchanged. The engine chains FireANTs' own
composable stages (GPU, Riemannian Adam), each seeding the next like ANTs' ``-t`` stages:

    Rigid (MI, seeded by 'moments_init') -> Affine (MI, seeded by the rigid) -> deformable

Two mirrored knobs specialise this shared module into the different presets (as ConvexAdam's shared
module is specialised by ``stages``). ``deformable_method`` picks the deformable stage:

    "syn"    symmetric diffeomorphic SyN (CC) (invertible, higher quality, averages cleanly for ensembling
    "greedy" greedy diffeomorphic (CC)) one-directional, faster / lower VRAM
    "none"   no deformable: the linear stage IS the transform (FireANTs_Affine)

and ``linear_method`` picks the linear one:

    "rigid_affine"  Rigid then Affine, as ANTs (the default
    "rigid"         Rigid only) no free scale or shear
    "none"          deformable from identity, for a pair already globally aligned

They compose, so ``deformable_method="none"`` runs whichever linear stage was asked for rather than
always Rigid+Affine: with ``linear_method="rigid"`` it is a rigid-only registration. The one
combination refused is both set to ``"none"``: that leaves nothing to optimise, and the engine
raises ``ValueError`` at build time rather than returning an identity transform.

Masks: the optional Fixed/Moving masks restrict the metric to a region. FireANTs implements this by
carrying the mask as the last image channel and prefixing the metric with ``masked_``; a mask is only
honoured when it actually restricts (some voxels in, some out), so the common mask-free path is
unchanged (an absent optional mask arrives as a whole-image default and is treated as no mask).

The deformable stages produce the single TOTAL displacement field on the fixed grid (the linear
pre-align is baked in via ``init_affine``, ANTs convention); ``none`` uses the affine matrix directly.
the emitted ``DisplacementField`` is rebuilt from that transform with SimpleITK (the same output path as the
ConvexAdam engine), so all presets/engines are interchangeable in an
ensemble. FireANTs' output-transform writer only serialises to a file, so the deformable field is
round-tripped through a temporary NIfTI (no FireANTs internals are reimplemented here).

NOTE: do NOT add ``from __future__ import annotations``: KonfAI's config engine relies on
runtime-evaluated annotations (``get_origin``); PEP 563 stringized annotations break binding.
"""

import functools
import gc
import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from functools import reduce
from pathlib import Path
from typing import Annotated, Literal, cast
from unittest import mock

import numpy as np
import SimpleITK as sitk
import torch
import torch.utils.checkpoint
from konfai.metric.measure import ImpactFeatureModel
from konfai.metric.measure.impact import (
    MIN_TILE,
    _patch_views,
    _statistics,
    channel_subset,
    distance,
    draw_centres,
    grid_size,
    ncc_of_sums,
    ncc_sums,
    no_texpr_fuser,
    normalized_features,
    onto_image_grid,
    pca_project,
    resampled,
    swept_order,
)
from konfai.network import network
from konfai.utils.config import Choices, Range
from konfai.utils.dataset import Attribute, image_to_data
from konfai.utils.errors import ConfigError, MeasureError
from konfai.utils.vram import halve_on_oom, out_of_memory_as_torch, peak_bytes

from ..ooc import free_bytes, resized
from .impact_loss import (
    FeatureMapUpdateInterval,
    LevelSpec,
    LNCCKernel,
    MixedPrecision,
    Mode,
    ModelSpec,
    Normalize,
    VoxelSampling,
    check_models,
    layer_weights,
    level_models,
)
from .intensity import (
    EngineRegistration,
    center_of_mass_translation,
    check_transform_type,
    is_partial_mask,
    winsorized,
)
from .orientation import world_aligned_pair

DIM = 3


def _fitting_tile(core, images: dict[str, torch.Tensor]) -> int:
    """The tile a Static extraction takes ``images`` in on their CUDA device: 0, each whole, when one pass over the
    larger fits half the free memory beside the layers both leave, else the widest cube that does, a whole number of
    the input multiple; measured on a corner of the larger, 64 voxels a side. 0 off CUDA."""
    larger = max(images.values(), key=lambda image: image.numel())
    if not larger.is_cuda:
        return 0
    probe = larger[(slice(None), slice(None), *[slice(0, max(64, core.model.multiple))] * (larger.dim() - 2))]
    layers, taken = peak_bytes(lambda: core.model.volume(probe, core.normalization, 0, 0.0), larger.device)
    voxels = probe[0, 0].numel()
    left = sum(layer.numel() * layer.element_size() for layer in layers) / voxels
    room = free_bytes(larger.device) / 2 - left * sum(image[0, 0].numel() for image in images.values())
    if taken / voxels * larger[0, 0].numel() <= room:
        return 0
    step = max(MIN_TILE, core.model.multiple)
    return max(step, int((max(room, 0.0) * voxels / taken) ** (1 / (larger.dim() - 2))) // step * step)


#: Linear stages a preset may ask for. The engine checks against this rather than trusting the
#: ``Literal`` annotation, which only binds a config-driven call and not a direct Python one.
_LINEAR_METHODS = ("rigid_affine", "rigid", "none")


class _StepPerElement:
    """A FireANTs linear stage's Adam with a step size per element: FireANTs gives the translation (a length, in
    voxels here) and the rotation or the matrix (unitless) one. Adam's update does not depend on its step size until it
    is applied, so the optimizer runs at a step of 1 and each element's update is scaled after it: exactly a step size
    per element. ``steps`` pairs each parameter with its step size, a number or one per element."""

    def __init__(self, optimizer, steps: list) -> None:
        self._optimizer, self._steps = optimizer, steps

    @property
    def param_groups(self) -> list:
        return self._optimizer.param_groups

    def zero_grad(self, *args, **kwargs) -> None:
        self._optimizer.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs) -> None:
        starts = [parameter.detach().clone() for parameter, _ in self._steps]
        self._optimizer.step(*args, **kwargs)
        with torch.no_grad():
            for (parameter, size), start in zip(self._steps, starts, strict=True):
                parameter.copy_(start + (parameter - start) * size)


def _on_model_grid(
    moved: torch.Tensor, fixed: torch.Tensor, mask: torch.Tensor | None, size: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """The pair resampled on a model's grid (``resampled``), its mask by nearest neighbour."""
    return resampled(moved, size), resampled(fixed, size), None if mask is None else resampled(mask, size, "nearest")


def _model_cells(low: int, high: int, voxels: int, cells: int) -> tuple[int, int]:
    """The cells ``[first, last)`` of a model grid of ``cells`` over a level axis of ``voxels`` whose positions
    (``resampled``: cell j at the level's continuous index j * voxels / cells) a box ``[low, high)`` holds with both
    voxels it interpolates between; at the level's end, every cell left, which reads zeros past it as the whole does."""
    first = -(-low * cells // voxels)
    last = cells if high == voxels else ((high - 1) * cells) // voxels + 1
    return first, max(first, last)


def _box_on_model_grid(
    moved: torch.Tensor,
    fixed: torch.Tensor,
    mask: torch.Tensor | None,
    place: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    size: tuple[int, ...],
    stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """A box of the pair (``place``: its ``low`` and ``high`` in a level of ``grid`` voxels) on the cells of the
    level's model grid of ``size`` it holds (``_model_cells``), from one on ``stride`` to one as far from the grid's end
    as a whole number of it, so that a strided layer meets the whole grid's cells; interpolated as ``resampled``
    interpolates the whole level: linearly, zero past the level's end, its mask at the level's nearest voxel, ties
    broken alike in every box."""
    low, high, grid = place
    axes, nearest = [], []
    for a, b, voxels, cells in zip(low, high, grid, size, strict=True):
        first, last = _model_cells(a, b, voxels, cells)
        first = -(-first // stride) * stride
        last = max(first, last - (cells - last) % stride)
        position = torch.arange(first, last, dtype=torch.float64) * voxels / cells
        axes.append(2 * (position - a) / max(b - a - 1, 1) - 1)
        nearest.append((torch.floor(position + 0.5).long() - a).clamp(0, b - a))  # b - a: past the level's end
    coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij")[::-1], dim=-1)[None]

    def sample(tensor: torch.Tensor) -> torch.Tensor:
        source = tensor if tensor.is_floating_point() else tensor.float()
        grid_ = coordinates.to(source.device, source.dtype).expand(source.shape[0], *coordinates.shape[1:])
        return torch.nn.functional.grid_sample(source, grid_, mode="bilinear", padding_mode="zeros", align_corners=True)

    def pick(tensor: torch.Tensor) -> torch.Tensor:
        for axis, index in enumerate(nearest):
            padded = torch.nn.functional.pad(tensor, [0, 0] * (tensor.dim() - 3 - axis) + [0, 1])  # zeros past it
            tensor = padded.index_select(2 + axis, index.to(tensor.device))
        return tensor

    return sample(moved), sample(fixed), None if mask is None else pick(mask.float())


def _require_fireants() -> None:
    """Fail before any compute when FireANTs' deformable registrations do not import. ``import fireants`` proves
    nothing (the package is empty): scipy, and ``fcntl``, which Windows lacks, are reached by the SyN and greedy
    modules only, which a registration would otherwise first import after minutes of linear stages."""
    import importlib

    try:
        for module in ("fireants.registration.syn", "fireants.registration.greedy"):
            importlib.import_module(module)
    except ModuleNotFoundError as error:
        if error.name == "fcntl":
            raise RuntimeError(
                "FireANTs imports fcntl, which only Linux and macOS have: the FireANTs presets cannot run on this "
                "system. The elastix and ConvexAdam presets can."
            ) from error
        raise RuntimeError(
            f"FireANTs cannot be imported ({error}). Its preset installs it through konfai-apps (app.json "
            "requirements_no_deps); with KONFAI_APPS_INSTALL_REQUIREMENTS=0, install 'fireants' with pip --no-deps and "
            "the preset's requirements.txt yourself."
        ) from error


def _patch_distances(
    model: torch.nn.Module,
    moved: torch.Tensor,
    fixed: torch.Tensor,
    mask: torch.Tensor | None,
    moved_rest: list[torch.Tensor],
    fixed_rest: list[torch.Tensor],
    kept: list[int],
    terms: list[tuple[str, int]],
    normalization: str,
    project: Callable[[int, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None,
    seed: int,
    kernel: int,
    pooled: "dict[int, tuple[torch.Tensor, float, float]] | None" = None,
) -> torch.Tensor:
    """Each kept layer's distance on one patch, stacked: the network run on both images, each layer normalised, reduced
    by PCA, cut to its channel subset and compared, in itk-impact's order. A layer in ``pooled`` (an NCC whose level
    sums ``ImpactFeatureLoss.pool`` gave: their derivative, the level's value, and the inverse of the stage's weight of
    the piece) takes the level's value, and through the patch's sums the level's gradient."""
    moved_layers, fixed_layers = model(moved, *moved_rest), model(fixed, *fixed_rest)
    values = []
    for index, (layer, (name, subset)) in enumerate(zip(kept, terms, strict=True)):
        moved_features = normalized_features(moved_layers[layer].float(), normalization)
        fixed_features = normalized_features(fixed_layers[layer].float(), normalization)
        if project is not None:
            moved_features, fixed_features = project(index, moved_features, fixed_features)
        moved_features, fixed_features = channel_subset(moved_features, fixed_features, subset, seed + index)
        layer_mask = None
        if mask is not None:
            layer_mask = torch.nn.functional.interpolate(mask.float(), size=moved_features.shape[2:], mode="nearest")
        if pooled is not None and index in pooled:
            derivative, value, scale = pooled[index]
            weight = layer_mask if layer_mask is not None else torch.ones_like(moved_features[:, :1])
            surrogate = (derivative.to(moved_features.device) * ncc_sums(moved_features, fixed_features, weight)).sum()
            values.append((value + (surrogate - surrogate.detach()) * scale).float())
        else:
            values.append(distance(name, moved_features, fixed_features, layer_mask, kernel, 0))
    return torch.stack(values)


def _on_basis(feature: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """``feature`` ``[B, C, *spatial]`` on the PCA ``basis`` ``[C, K]``, uncentred: a piece centred by its own mean
    would not add up to the level, and an NCC takes each component's mean off itself."""
    return torch.einsum("bc...,ck->bk...", feature, basis.to(feature.device, feature.dtype))


def _level_basis(total: tuple[torch.Tensor, torch.Tensor, torch.Tensor], components: int) -> torch.Tensor:
    """The basis ``[C, K]`` of the ``components`` principal components of a layer from its level sums (``pca_sums``:
    its cells, their channels' sums and cross-products), as ``pca_project`` fits it on the whole map."""
    count, sums, products = total
    mean = sums / count
    covariance = (products - count * torch.outer(mean, mean)) / max(float(count) - 1, 1.0)
    channels = covariance.shape[0]
    return torch.linalg.eigh(covariance)[1][:, channels - min(components, channels) :].float()


def _level_ncc(total: torch.Tensor) -> tuple[torch.Tensor, float]:
    """The NCC of a layer's level sums and its derivative with respect to them."""
    total.requires_grad_()
    value = ncc_of_sums(total)
    (derivative,) = torch.autograd.grad(value, total)
    return derivative, float(value.detach())


class _Intensities:
    """An image's intensity statistics over the voxels pieces of it own, added piece by piece: what ``_statistics``
    reads off the whole image (its unbiased standard deviation, over every channel)."""

    def __init__(self) -> None:
        self.count, self.total, self.squares = 0, 0.0, 0.0
        self.low, self.high = math.inf, -math.inf

    def add(self, image: torch.Tensor, region: torch.Tensor) -> None:
        values = image[0][:, region[0, 0] > 0.5].double()
        if values.numel() == 0:
            return
        self.count += values.numel()
        self.total += float(values.sum())
        self.squares += float((values * values).sum())
        self.low, self.high = min(self.low, float(values.min())), max(self.high, float(values.max()))

    def attribute(self) -> Attribute:
        mean = self.total / self.count
        variance = max(self.squares - self.count * mean * mean, 0.0) / max(self.count - 1, 1)
        return Attribute(
            {"ImageMin": self.low, "ImageMax": self.high, "ImageMean": mean, "ImageStd": math.sqrt(variance)}
        )


class _ImpactCore(torch.nn.Module):
    """One IMPACT feature model: its network (KonfAI's ``ImpactFeatureModel``, which prepares the inputs as itk-impact
    does) and its kept layers' distances between two images, for Jacobian mode, or its feature volumes, for Static.
    Each layer is normalised and weighed on its own, as in the other engines."""

    def __init__(self, spec: "ModelSpec", mixed_precision: bool, gradient: bool = False) -> None:
        super().__init__()
        self.extent: list[float] | None = None  # the fixed image's size in mm along (L, P, S), for the model grids
        self.pca = int(spec.pca)
        self.normalization = spec.feature_normalization
        weights = [1.0 if char == "1" else 0.0 for char in spec.layers_mask]
        # Fetched, shaped by the registry and probed once on the CPU; the whole (downsampled) tensor is scored. The
        # engine hands it copies with their voxel axes in LPS order (``world_aligned``): their direction is the
        # identity, from which a TotalSegmentator model reorients them as it was trained.
        self.model = ImpactFeatureModel.from_ref(spec.ref, weights)
        self.model.check(gradient)
        self.model.direction = torch.eye(DIM, dtype=torch.int16)
        self.model.half = mixed_precision
        self.dimension = self.model.dim  # 2 for a network swept slice by slice
        self.kept = self.model.kept
        # The tile each pyramid level is scored in once its whole image has run out of the card's memory.
        self._tiles: dict[tuple[int, ...], list[int]] = {}
        self._layer_rules: list[list[tuple[int, bool]]] | None = None  # each kept layer's stride and rounding
        self._last_shape: tuple[int, ...] | None = None  # the level scored last, for narrow()

    def _pca_project(
        self, output_feature: torch.Tensor, target_feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return pca_project(output_feature, target_feature, self.pca, self.dimension)

    def _scored(
        self,
        moved: torch.Tensor,
        fixed: torch.Tensor,
        mask: torch.Tensor | None,
        statistics: tuple[dict, dict],
        terms: list[tuple[str, int]],
        seed: int,
        kernel: int,
        pooled: dict[int, tuple[torch.Tensor, float, float]] | None = None,
        bases: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Each kept layer's distance, averaged over the patches the mask reaches (the whole image when untiled);
        ``bases``, each layer's PCA basis when the level's sums gave it (``ImpactFeatureLoss.pool``)."""
        network = self.model.network(moved.device)
        moved_inputs, fixed_inputs = self.model.inputs(moved, statistics[0]), self.model.inputs(fixed, statistics[1])
        project = None
        if bases is not None:
            project = lambda index, moved, fixed: (_on_basis(moved, bases[index]), _on_basis(fixed, bases[index]))  # noqa: E731
        elif self.pca > 0:
            project = lambda index, moved, fixed: self._pca_project(moved, fixed)  # noqa: E731
        patches = [
            (moved_patch, fixed_patch, mask_patch)
            for moved_patch, fixed_patch, mask_patch in _patch_views(
                moved_inputs[0], fixed_inputs[0], mask, self.model.shape
            )
            if mask_patch is None or torch.any(mask_patch == 1)
        ]
        # The patches' sums add up to the piece's: each patch's surrogate counts whole in their average.
        pooled = {layer: (d, v, scale * len(patches)) for layer, (d, v, scale) in (pooled or {}).items()} or None
        total: torch.Tensor | None = None
        count = 0
        for moved_patch, fixed_patch, mask_patch in patches:
            args = (network, moved_patch, fixed_patch, mask_patch, moved_inputs[1:], fixed_inputs[1:], self.kept, terms)
            args += (self.normalization, project, seed, kernel, pooled)
            if self.model.checkpoint:
                values = torch.utils.checkpoint.checkpoint(_patch_distances, *args, use_reentrant=False)
            else:
                values = _patch_distances(*args)
            total = values if total is None else total + values
            count += 1
        if total is None:  # a mask no patch reaches: nothing to score
            return torch.zeros(len(self.kept), device=moved.device)
        return total / count

    def distances(
        self,
        moved: torch.Tensor,
        fixed: torch.Tensor,
        mask: torch.Tensor | None,
        terms: list[tuple[str, int]],
        seed: int,
        kernel: int,
        statistics: tuple[Attribute, Attribute] | None = None,
        pooled: dict[int, tuple[torch.Tensor, float, float]] | None = None,
        place: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None = None,
        bases: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Each kept layer's distance over the whole image, or over tiles once the whole image has run out of memory.
        ``statistics`` are the moved and fixed images' intensity statistics when the two tensors are pieces of them
        (``ImpactFeatureLoss.pool``), their own otherwise; ``pooled``, the NCC layers whose level sums it gave;
        ``place``, where a piece lies in its level (``_prepared``); ``bases``, the layers' PCA bases the level's
        sums gave.

        A level that runs out is scored again in tiles half as wide, and in tiles half as wide again if it
        still does not fit; the tile is kept for that level, so the retry happens once per registration.
        This keeps the registration one global registration, where cutting the volume would estimate its
        rigid and affine stages patch by patch.
        """
        moved, fixed, mask = self._prepared(moved, fixed, mask, seed, place)
        shape = self._last_shape = tuple(moved.shape[2:])
        statistics = statistics or (_statistics(moved)[0], _statistics(fixed)[0])

        def run() -> torch.Tensor:
            tile = self._tiles.get(shape)
            # Each tile under a checkpoint: the optimiser differentiates through the network for BOTH warped images
            # (SyN warps the fixed one too), so a whole-image pass keeps two networks' activations until the backward.
            # The intensity statistics stay the whole image's, so every tile is normalised alike.
            self.model.shape, self.model.checkpoint = tile, tile is not None
            return self._scored(moved, fixed, mask, statistics, terms, seed, kernel, pooled, bases)

        with no_texpr_fuser():
            return halve_on_oom(run, self._narrow_tile, moved.is_cuda)

    def ncc_sums(
        self,
        moved: torch.Tensor,
        fixed: torch.Tensor,
        weight: torch.Tensor,
        statistics: tuple[Attribute, Attribute],
        terms: list[tuple[str, int]],
        seed: int,
        place: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None = None,
        bases: list[torch.Tensor] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Each NCC layer's ``ncc_sums`` over one piece, the voxels ``weight`` keeps on its grid, its features on its
        PCA basis when ``bases`` gives it: what the level's sums add up from it (``ImpactFeatureLoss.pool``). The
        network runs on the whole piece, as ``distances`` does."""
        moved, fixed, weight = self._prepared(moved, fixed, weight, seed, place)
        network = self.model.network(moved.device)
        moved_inputs, fixed_inputs = self.model.inputs(moved, statistics[0]), self.model.inputs(fixed, statistics[1])
        with torch.no_grad(), no_texpr_fuser():
            moved_layers, fixed_layers = network(*moved_inputs), network(*fixed_inputs)
        sums = {}
        for index, (layer, (name, subset)) in enumerate(zip(self.kept, terms, strict=True)):
            if name != "NCC":
                continue
            moved_features = normalized_features(moved_layers[layer].float(), self.normalization)
            fixed_features = normalized_features(fixed_layers[layer].float(), self.normalization)
            if bases is not None:
                moved_features, fixed_features = (_on_basis(f, bases[index]) for f in (moved_features, fixed_features))
            moved_features, fixed_features = channel_subset(moved_features, fixed_features, subset, seed + index)
            layer_weight = torch.nn.functional.interpolate(
                weight.float(), size=moved_features.shape[2:], mode="nearest"
            )
            sums[index] = ncc_sums(moved_features, fixed_features, layer_weight).cpu()
        return sums

    def pca_sums(
        self,
        fixed: torch.Tensor,
        region: torch.Tensor,
        statistics: Attribute,
        seed: int,
        place: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Each kept layer's fixed features over the cells of one piece ``region`` covers on its grid: their count,
        their channels' sums and cross-products, in double. Over the pieces of a level they add up to what ``pca_project``
        fits its basis on, every cell of the whole map (``_level_basis``)."""
        fixed, _, region = self._prepared(fixed, fixed, region, seed, place)
        with torch.no_grad(), no_texpr_fuser():
            layers = self.model.network(fixed.device)(*self.model.inputs(fixed, statistics))
        sums = []
        for layer in self.kept:
            features = normalized_features(layers[layer].float(), self.normalization)
            weight = torch.nn.functional.interpolate(region.float(), size=features.shape[2:], mode="nearest")
            flat, kept = features[0].flatten(1).double(), weight.flatten().double()
            sums.append(((kept.sum()).cpu(), (flat @ kept).cpu(), ((flat * kept) @ flat.t()).cpu()))
        return sums

    def layer_cells(self, weight: torch.Tensor) -> list[float]:
        """Each kept layer's cells over a piece ``weight`` ``[1, 1, *spatial]`` covers as the network reads it (after
        ``_prepared``): the cells its nearest voxel puts in, counted on the layer's own grid (``layer_shapes``)."""
        return [
            float(torch.nn.functional.interpolate(weight.float(), size=shape, mode="nearest").sum())
            for shape in self.layer_shapes(tuple(weight.shape[2:]), weight.device)
        ]

    def layer_shapes(self, shape: tuple[int, ...], device: torch.device | str) -> list[tuple[int, ...]]:
        """Each kept layer's grid for an input of ``shape``: its stride along each axis, and whether it rounds a
        remainder up, read once off a probe of an odd side."""
        if self._layer_rules is None:
            side = 2 * math.lcm(max(1, int(self.model.multiple)), self.model.block_stride(device)) + 1
            probe = torch.rand((1, self.model.in_channels, *[side] * len(shape)), device=device)
            with torch.no_grad(), no_texpr_fuser():
                outputs = self.model.network(probe.device)(*self.model.inputs(probe, _statistics(probe)[0]))
            self._layer_rules = [
                [
                    (stride, int(cells) * stride >= side)
                    for stride, cells in zip(strides, outputs[layer].shape[2:], strict=True)
                ]
                for layer, strides in zip(self.kept, self.model.layer_strides(device), strict=True)
            ]
        return [
            tuple(-(-n // stride) if up else max(1, n // stride) for n, (stride, up) in zip(shape, rule, strict=True))
            for rule in self._layer_rules
        ]

    def model_size(self, level: tuple[int, ...], seed: int) -> tuple[int, ...] | None:
        """The grid a model with a voxel_size reads a level of ``level`` voxels on (None without one): the level's
        extent at its voxel_size, a network of one dimension fewer keeping the level's voxels along the axis it sweeps."""
        voxel_size = self.model.voxel_size
        if voxel_size is None:
            return None
        if self.dimension == DIM:
            return grid_size(self.extent, voxel_size, level)
        order = swept_order(seed)
        # The two values are the slices' own: the swept axis keeps this level's, the other two the model's.
        sides = list(reversed(self.extent)) if self.extent is not None else [float(n) for n in level]
        size = [0, 0, 0]
        size[order[2] - 2] = level[order[2] - 2]
        for axis, step in zip(order[3:], voxel_size, strict=True):
            size[axis - 2] = max(1, int(sides[axis - 2] / step + 0.5))
        return tuple(size)

    def _prepared(
        self,
        moved: torch.Tensor,
        fixed: torch.Tensor,
        mask: torch.Tensor | None,
        seed: int,
        place: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """The two images and the mask as the network reads them: on its grid at its voxel_size, and for a network
        of one dimension fewer, with the axis it sweeps (drawn at each evaluation, ``swept_order``) first. ``place``
        (a box's ``low`` and ``high`` in a level of ``grid`` voxels) has a box take the cells of the level's model grid
        that fall in it, as the whole level's resampling places them."""
        size = self.model_size(place[2] if place is not None else tuple(moved.shape[2:]), seed)
        if size is not None:
            if place is None:
                moved, fixed, mask = _on_model_grid(moved, fixed, mask, size)
            else:
                stride = self.model.block_stride(moved.device)
                moved, fixed, mask = _box_on_model_grid(moved, fixed, mask, place, size, stride)
        if self.dimension < DIM:
            order = swept_order(seed)
            moved, fixed, mask = (
                moved.permute(order),
                fixed.permute(order),
                None if mask is None else mask.permute(order),
            )
        return moved, fixed, mask

    def sampled_distances(
        self,
        moved: torch.Tensor,
        fixed: torch.Tensor,
        centres: torch.Tensor,
        patch: int,
        terms: list[tuple[str, int]],
        seed: int,
        kernel: int,
    ) -> torch.Tensor:
        """Each kept layer's distance at ``centres`` [N, 3], between the centre features of the ``patch`` around each
        point (``ImpactFeatureModel.sampled``, elastix's Jacobian scheme)."""
        layers = self.model.sampled(moved, fixed, centres, patch, self.normalization, self.extent, seed)
        values = []
        for index, ((moved_features, fixed_features), (name, subset)) in enumerate(zip(layers, terms, strict=True)):
            if self.pca > 0:
                moved_features, fixed_features = self._pca_project(moved_features, fixed_features)
            moved_features, fixed_features = channel_subset(moved_features, fixed_features, subset, seed + index)
            values.append(distance(name, moved_features, fixed_features, None, kernel, 0))
        return torch.stack(values)

    def _narrow_tile(self) -> bool:
        """The level scored last in tiles half as wide; False once they are ``MIN_TILE`` a side."""
        assert self._last_shape is not None
        tile = self._tiles.get(self._last_shape)
        if tile is not None and max(tile) <= MIN_TILE:
            return False
        self._tiles[self._last_shape] = [max(MIN_TILE, (size + 1) // 2) for size in (tile or self._last_shape)]
        return True

    def narrow(self) -> bool:
        """Smaller pieces for the evaluation scored last (tiles, or patches a batch when sampled), once it has run out
        of memory where ``distances`` cannot retry it: in the backward, which FireANTs runs itself. False once they go
        no smaller."""
        if self.model.batch:
            return self.model.narrow_batch()
        return self._last_shape is not None and self._narrow_tile()


class ImpactFeatureLoss(torch.nn.Module):
    """FireANTs ``custom_loss``: the IMPACT loss of the other engines, layer by layer.

    Each kept layer of each model is compared with its model's ``distance``; with ``normalize`` it is divided by its
    value at the first evaluation of every FireANTs scale (FireANTs announces each through
    ``set_current_scale_and_iterations``), so every layer starts the scale at 1; the layers are summed weighed by
    ``layers_weight``. ``levels`` gives each scale its own models.

    Jacobian mode runs each network on both warped images at every step. Static mode extracts every model's layers
    once per image (``extract``), FireANTs then warps those feature volumes, and this compares their channels layer by
    layer.

    ``voxel_sampling`` below 1 reads that share of the voxels at each evaluation, drawn anew: Static compares the
    warped volumes there only; Jacobian runs each network on the patch of its receptive field around every drawn
    point instead of on the whole images, and compares the patches' centre voxels, as elastix does.

    ``masked`` mirrors the engine's own decision to run FireANTs' masked mode: the images then carry the mask as one
    extra trailing channel (``apply_mask_to_image``), which nothing about the tensors themselves announces; the
    distances are then averaged where the fixed mask and the warped moving mask both hold.
    """

    def __init__(
        self,
        levels: list[list["ModelSpec"]],
        mode: str,
        normalize: bool,
        lncc_kernel: int,
        chunk: int,
        seed: int,
        mixed_precision: bool,
        masked: bool = False,
        voxel_sampling: float = 1.0,
    ) -> None:
        super().__init__()
        self._specs: list[ModelSpec] = []  # each distinct model once, whatever the levels it serves
        for spec in (spec for specs in levels for spec in specs):
            if spec not in self._specs:
                self._specs.append(spec)
        self._levels = [[self._specs.index(spec) for spec in specs] for specs in levels]
        self._cores = torch.nn.ModuleList()
        for spec in self._specs:
            core = _ImpactCore(spec, mixed_precision, gradient=mode == "Jacobian")
            if spec.voxel_size is not None:
                # As elastix reads it: the image's three axes, but a 2D network's own two in Jacobian mode (its slices
                # or planes), where the swept axis keeps the level's resolution.
                axes = 2 if core.dimension < DIM and mode == "Jacobian" else DIM
                if len(spec.voxel_size) != axes:
                    raise ValueError(
                        f"voxel_size of '{spec.ref}' has {len(spec.voxel_size)} values, expected {axes} "
                        f"({'the 2D network slices in Jacobian mode' if axes == 2 else 'x y z in mm'})."
                    )
                core.model.voxel_size = [float(v) for v in spec.voxel_size]
            self._cores.append(core)
        # Sampled Jacobian: each model's patch, its receptive field, as elastix's PatchSize.
        self._patch: list[int] = []
        if voxel_sampling < 1 and mode == "Jacobian":
            for spec, core in zip(self._specs, self.cores, strict=True):
                try:
                    self._patch.append(core.model.receptive_field)
                except MeasureError as error:
                    raise ValueError(
                        f"voxel_sampling in Jacobian mode runs '{spec.ref}' on the patch of its receptive field around "
                        f"each drawn point, which cannot be sized: {error.args[0]} Use mode Static, or voxel_sampling 1."
                    ) from error
        self._sampling = float(voxel_sampling)
        self._mode = mode
        self._normalize = normalize
        self._kernel = int(lncc_kernel)
        self._chunk = max(0, int(chunk))  # the LNCC's channels a pass in Static mode, 0 for all
        self._seed = int(seed)
        # The draws of one registration: reset() seeds it again, so a case draws the same channels and points
        # whatever registrations this cached loss ran before it, and a restarted stage replays its own.
        self._generator = torch.Generator().manual_seed(self._seed)
        self._masked = masked
        self._channels: list[list[int]] = []  # Static: each model's kept layers' channels in the volumes
        self._level = -1  # the FireANTs scale running, advanced by set_current_scale_and_iterations
        self._factors: list[float] | None = None  # each layer's normalization at this level
        self._draw: int | None = None  # the draw every piece of an iteration takes (``draw``)
        self._pooled: dict[int, tuple[torch.Tensor, float, float]] | None = None  # NCC layers' level sums (``pool``)
        # Jacobian: each model's intensities over the level, the cells it weighs, and where a piece lies (``pool``)
        self._statistics: dict[int, tuple[Attribute, Attribute]] = {}
        self._cells: dict[int, list[float]] = {}  # each model's kept layers' cells the level weighs
        self._place: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None = None
        # Jacobian: each model's NCC layers' derivative and value from the level's sums, and the level's weight
        self._layer_sums: dict[int, dict[int, tuple[torch.Tensor, float]]] = {}
        self._bases: dict[int, list[torch.Tensor]] = {}  # each model's layers' PCA bases over the level (``pool``)
        self._level_weight = 0.0

    @property
    def cores(self) -> list[_ImpactCore]:
        return [cast("_ImpactCore", core) for core in self._cores]

    def reset(self) -> None:
        """A registration (or a restarted stage) begins: the next scale FireANTs announces is the first level, and
        the draws start again from the seed."""
        self._level, self._factors, self._draw, self._pooled, self._place = -1, None, None, None, None
        self._statistics, self._cells, self._layer_sums, self._bases = {}, {}, {}, {}
        self._generator.manual_seed(self._seed)

    def draw(self) -> None:
        """An iteration evaluated by pieces of the level begins: every piece takes this draw (the channels a layer
        keeps, the axis a 2-D network sweeps), as one evaluation of the whole level does."""
        self._draw = int(torch.randint(2**31 - 1, (1,), generator=self._generator))

    def set_current_scale_and_iterations(self, scale: int, iterations: int) -> None:
        """FireANTs' hook at the start of every scale: the next level, whose layers start at 1 again."""
        self._level, self._factors, self._pooled = self._level + 1, None, None
        self._statistics, self._cells, self._layer_sums, self._bases = {}, {}, {}, {}

    def start_at(self, level: int) -> None:
        """A run of a refreshed Static stage (``FireANTsEngine._refreshed_stage``) registers FireANTs scale ``level``
        alone: the next scale FireANTs announces is that level, whose layers start at 1 again."""
        self._level, self._factors = level - 1, None

    def release(self) -> None:
        """Move the feature models off the card: Static mode is done with them once the volumes are extracted."""
        for core in self.cores:
            if core.model.model is not None:
                core.model.model.cpu()

    def narrow(self) -> bool:
        """Smaller pieces, once the backward has run out of memory: tiles for every network (Jacobian), half as many
        channels a LNCC pass (Static). False when none can go smaller."""
        if self._mode == "Jacobian":
            narrowed = [core.narrow() for core in self.cores]  # every one, not up to the first that can
            return any(narrowed)
        if not any(spec.distance == "LNCC" for spec in self._specs):
            return False  # the chunk splits LNCC's pass only: the other distances need no less memory in pieces
        widest = max((c for layers in self._channels for c in layers), default=1)
        chunk = self._chunk if 0 < self._chunk < widest else widest
        if chunk <= 1:
            return False
        self._chunk = (chunk + 1) // 2
        return True

    def extract(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        patch: int,
        extents: tuple[list[float] | None, list[float] | None] = (None, None),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Static mode: the fixed and moving volumes of every model's kept layers, concatenated along the channels.

        A model with a ``voxel_size`` extracts on the image resampled at that resolution (``extents``: each image's
        size in mm along L, P, S) and its features are brought back onto the image's grid, as elastix interpolates
        its feature maps at its points: FireANTs registers them at the image's resolution."""
        resampled_images = []
        for core in self.cores:
            images = {"fixed": fixed, "moving": moving}
            if core.model.voxel_size is not None:
                images = {
                    side: resampled(image, grid_size(extent, core.model.voxel_size, tuple(image.shape[2:])))
                    for (side, image), extent in zip(images.items(), extents, strict=True)
                }
            resampled_images.append(images)
        # One pass over each image, or, where a model does not fit one (or the preset gives a patch), every model by
        # blocks as a segmentation is predicted, which gives the one pass's features whatever each image's extent.
        try:
            with out_of_memory_as_torch(fixed.is_cuda):
                tiles = [
                    patch or _fitting_tile(core, images)
                    for core, images in zip(self.cores, resampled_images, strict=True)
                ]
                if not any(tiles):
                    return self._one_pass(fixed, moving, resampled_images)
        except torch.cuda.OutOfMemoryError:
            tiles = [64]  # the probe's side: the blocks halve from it while one does not fit
        # Out of the handler, the tensors its traceback held are released.
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[FireANTs] feature extraction does not fit whole: extracted by blocks of {max(tiles)} voxels.")
        return self._by_blocks(fixed, moving, max(tiles), extents)

    def _one_pass(
        self, fixed: torch.Tensor, moving: torch.Tensor, resampled_images: list[dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``extract`` in one pass of every model over each image (``resampled_images``: as each model reads them)."""
        fixed_sides, moving_sides, self._channels = [], [], []
        for core, images in zip(self.cores, resampled_images, strict=True):
            fixed_layers, moving_layers = (
                core.model.volume(images[side], core.normalization, 0, 0.0) for side in ("fixed", "moving")
            )
            if core.pca > 0:  # onto the fixed image's basis, as the other engines fit it on the reference side
                for index, (fixed_layer, moving_layer) in enumerate(zip(fixed_layers, moving_layers, strict=True)):
                    moving_layers[index], fixed_layers[index] = core._pca_project(moving_layer, fixed_layer)
            # Normalised and reduced on its own grid, each layer is read at the image's voxels as elastix reads it.
            fixed_layers = [onto_image_grid(layer, tuple(images["fixed"].shape[2:])) for layer in fixed_layers]
            moving_layers = [onto_image_grid(layer, tuple(images["moving"].shape[2:])) for layer in moving_layers]
            if core.model.voxel_size is not None:
                fixed_layers = [resampled(layer, tuple(fixed.shape[2:]), padding="border") for layer in fixed_layers]
                moving_layers = [resampled(layer, tuple(moving.shape[2:]), padding="border") for layer in moving_layers]
            self._channels.append([layer.shape[1] for layer in fixed_layers])
            fixed_sides += fixed_layers
            moving_sides += moving_layers
        # A single layer is returned as it is: concatenating it would copy the whole volume.
        return tuple(sides[0] if len(sides) == 1 else torch.cat(sides, dim=1) for sides in (fixed_sides, moving_sides))

    def _by_blocks(
        self, fixed: torch.Tensor, moving: torch.Tensor, block: int, extents: tuple[list[float] | None, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both images' feature volumes predicted by blocks of ``block`` voxels a side at least, as the sampled route
        predicts them (``static_feature_sources``): the one pass's features, the fixed image's PCA basis projecting
        both."""
        from ..ooc import TensorSource
        from .fireants_sampled import static_feature_sources

        with tempfile.TemporaryDirectory(prefix="impact_static_") as directory:
            sources = static_feature_sources(
                self, TensorSource(fixed[0]), TensorSource(moving[0]), Path(directory), block=block, extents=extents
            )
            volumes = [source.read([0] * len(source.shape), list(source.shape)).to(fixed.device) for source in sources]
            del sources  # their memmaps: Windows removes no file still mapped
        return volumes[0], volumes[1]

    def _reach(self, index: int, shape: tuple[int, ...]) -> int:
        """The cube, in voxels of this level, that model ``index``'s sampled patch spans: its receptive field, wider when
        its voxel_size is coarser than the level's voxels (odd, so it has a centre)."""
        core, patch = self.cores[index], self._patch[index]
        if core.model.voxel_size is None:
            return patch
        extent = core.extent if core.extent is not None else [float(size) for size in reversed(shape)]
        finest = min(side / size for side, size in zip(extent, reversed(shape), strict=True))
        return math.ceil(patch * max(core.model.voxel_size) / finest) | 1

    def _terms(self, index: int) -> list[tuple[str, int]]:
        """Each kept layer of model ``index``: its distance and the channels it draws (0 = all)."""
        spec = self._specs[index]
        return [(spec.distance, int(spec.subset_features))] * len(self.cores[index].kept)

    def _static_layers(self, level: list[int], moved: torch.Tensor, fixed: torch.Tensor, seed: int):
        """Each kept layer of the warped feature volumes, read off their channels: its distance's name and its moved
        and fixed maps."""
        for index in level:
            start = sum(sum(layers) for layers in self._channels[:index])
            for layer, (channels, (name, subset)) in enumerate(
                zip(self._channels[index], self._terms(index), strict=True)
            ):
                moved_layer, fixed_layer = channel_subset(
                    moved[:, start : start + channels],
                    fixed[:, start : start + channels],
                    subset,
                    seed + 1000 * index + layer,
                )
                yield name, moved_layer, fixed_layer
                start += channels

    def _static_distances(
        self, level: list[int], moved: torch.Tensor, fixed: torch.Tensor, mask: torch.Tensor | None, seed: int
    ) -> list[torch.Tensor]:
        """Each kept layer's distance between the warped feature volumes, read off their channels; an NCC layer whose
        level sums ``pool`` gave, the level's (see ``pool``)."""
        values = []
        pooled = self._pooled or {}
        for layer, (name, moved_layer, fixed_layer) in enumerate(self._static_layers(level, moved, fixed, seed)):
            if layer in pooled:
                # The level's NCC, and through this piece's sums the level's gradient at its voxels, weighed against
                # the stage's weight of a piece (its voxels over the level's) so that the two cancel.
                derivative, counted, value = pooled[layer]
                weight = mask if mask is not None else torch.ones_like(moved_layer[:, :1])
                piece = ncc_sums(moved_layer, fixed_layer, weight)
                surrogate = (derivative.to(piece.device) * piece).sum()
                scale = counted / max(float(piece[0, 0]), 1e-8)
                values.append((value + (surrogate - surrogate.detach()) * scale).float())
            else:
                values.append(distance(name, moved_layer, fixed_layer, mask, self._kernel, self._chunk))
        return values

    @property
    def pools(self) -> bool:
        """Whether an evaluation by pieces of the level needs the level's sums first (``pool``): Jacobian mode, whose
        networks normalise their inputs by the whole images' intensity statistics, and Static mode at a level with an
        NCC layer, an NCC being no mean of its voxels' values."""
        if self._sampling < 1 or not self._levels:
            return False
        if self._mode == "Jacobian":
            return True
        level = self._levels[min(max(self._level, 0), len(self._levels) - 1)]
        return any(name == "NCC" for index in level for name, _ in self._terms(index))

    def stride(self, device: torch.device | str) -> int:
        """The stride a piece of the level must start on, and end as far from its end as a whole number of, for every
        network run on it to meet the whole level's cells (``ImpactFeatureModel.block_stride``, the input multiple):
        Jacobian mode on whole pieces; 1 otherwise, and for a model read on its own grid (a voxel_size)."""
        if self._mode != "Jacobian" or self._sampling < 1:
            return 1
        strides = [
            math.lcm(max(1, int(core.model.multiple)), core.model.block_stride(device))
            for core in self.cores
            if core.model.voxel_size is None
        ]
        return max([1, *strides])

    def place(self, low: Sequence[int], high: Sequence[int], grid: Sequence[int]) -> None:
        """Where the next piece lies: ``[low, high)`` of a level of ``grid`` voxels (a registration by boxes says so
        before each), for a model that reads the level on its own grid (``_ImpactCore._prepared``)."""
        self._place = (tuple(low), tuple(high), tuple(grid))

    def _split(
        self, moved: torch.Tensor, fixed: torch.Tensor, region: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """The two images and the voxels they are compared over, ``region`` (all when None). Masked, the images lose
        their mask channel and a voxel counts where the fixed mask and the warped moving mask both hold it, as in
        FireANTs' own masked metrics."""
        if not self._masked:
            return moved, fixed, region
        weight = (fixed[:, -1:] * moved[:, -1:] >= 0.5).to(moved.dtype) * (1.0 if region is None else region)
        return moved[:, :-1], fixed[:, :-1], weight

    def pool(
        self,
        pieces: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | Callable[[], Iterable[tuple]],
    ) -> None:
        """The level's sums over the voxels the pieces own (each a moved and a fixed tensor read with its halo, and
        ones on its own voxels), under this iteration's draw. Static mode: each NCC layer's, its NCC, the voxels it
        counts and its derivative with respect to the sums, which give every piece's evaluation the level's value and
        gradient. Jacobian mode: the intensity statistics each network normalises by, then its NCC layers'."""
        if self._draw is None:
            self.draw()
        assert self._draw is not None
        level = self._levels[min(max(self._level, 0), len(self._levels) - 1)]
        read = pieces if callable(pieces) else (lambda: pieces)  # a callable reads them again for a second pass
        self._pooled = {}
        if self._mode == "Jacobian":
            self._pool_intensities(read, level)
            self._pool_layers(read, level)
            return
        totals: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for pieces in read():
                moved, fixed, mask = self._split(*pieces)
                for layer, (name, moved_layer, fixed_layer) in enumerate(
                    self._static_layers(level, moved, fixed, self._draw)
                ):
                    if name == "NCC":
                        piece = ncc_sums(moved_layer, fixed_layer, mask).cpu()
                        totals[layer] = piece if layer not in totals else totals[layer] + piece
        for layer, total in totals.items():
            derivative, value = _level_ncc(total)
            self._pooled[layer] = (derivative, float(total[0, 0]), value)

    def _pool_intensities(self, read: Callable[[], Iterable[tuple]], level: list[int]) -> None:
        """Jacobian mode: the intensity statistics of the warped images as each network reads them, over the voxels
        the pieces own (on its grid, for a model with a voxel_size), and the cells of that grid the level weighs
        (those both masks keep, when masked): what the whole level's evaluation computes on the whole images."""
        assert self._draw is not None
        sized = [index for index in level if self.cores[index].model.voxel_size is not None]
        intensities = {index: (_Intensities(), _Intensities()) for index in [None, *sized]}
        self._cells = {index: [0.0] * len(self.cores[index].kept) for index in level}
        self._level_weight = 0.0
        with torch.no_grad():
            for moved, fixed, region in read():
                moved, fixed, weight = self._split(moved, fixed, region)
                self._level_weight += float(weight.sum())  # as the stage weighs a piece: its voxels, or its support
                intensities[None][0].add(moved, region)
                intensities[None][1].add(fixed, region)
                for index in level:
                    seed = self._draw + 1000 * index
                    core = self.cores[index]
                    if index in sized:
                        on_grid, fixed_on_grid, own = core._prepared(moved, fixed, region, seed, self._place)
                        intensities[index][0].add(on_grid, own)
                        intensities[index][1].add(fixed_on_grid, own)
                    # The cells of each layer the piece owns (those both masks keep), on the layer's own grid.
                    own_weight = core._prepared(weight, weight, weight, seed, self._place)[2]
                    for layer, cells in enumerate(core.layer_cells(own_weight)):
                        self._cells[index][layer] += cells
        self._statistics = {}
        for index in level:
            moved_side, fixed_side = intensities[index if index in sized else None]
            if moved_side.count > 0:
                self._statistics[index] = (moved_side.attribute(), fixed_side.attribute())

    def _pool_layers(self, read: "Callable[[], Iterable[tuple]]", level: list[int]) -> None:
        """Jacobian mode: each NCC layer's sums over the voxels the pieces own, their networks run on the level's
        intensities (a second pass, once those are known), and the level's weight as the stage weighs a piece (its
        voxels, or those both masks keep)."""
        self._layer_sums, self._bases = {}, {}
        assert self._draw is not None
        models = [index for index in level if any(name == "NCC" for name, _ in self._terms(index))]
        if not models or any(index not in self._statistics for index in models):
            return
        # A PCA is fitted on the whole fixed map, every cell (``pca_project``): its basis from the pieces' sums.
        reduced = [index for index in models if self.cores[index].pca > 0]
        if reduced:
            fitted: dict[int, list] = {}
            for _, fixed, region in read():
                fixed = fixed[:, :-1] if self._masked else fixed
                for index in reduced:
                    seed = self._draw + 1000 * index
                    piece = self.cores[index].pca_sums(fixed, region, self._statistics[index][1], seed, self._place)
                    fitted[index] = (
                        piece
                        if index not in fitted
                        else [
                            tuple(a + b for a, b in zip(total, part, strict=True))
                            for total, part in zip(fitted[index], piece, strict=True)
                        ]
                    )
            self._bases = {
                index: [_level_basis(total, self.cores[index].pca) for total in totals]
                for index, totals in fitted.items()
            }
        totals: dict[tuple[int, int], torch.Tensor] = {}
        for pieces in read():
            moved, fixed, weight = self._split(*pieces)
            for index in models:
                seed = self._draw + 1000 * index
                core, terms, bases = self.cores[index], self._terms(index), self._bases.get(index)
                for layer, piece in core.ncc_sums(
                    moved, fixed, weight, self._statistics[index], terms, seed, self._place, bases
                ).items():
                    totals[(index, layer)] = piece if (index, layer) not in totals else totals[(index, layer)] + piece
        for (index, layer), total in totals.items():
            self._layer_sums.setdefault(index, {})[layer] = _level_ncc(total)

    def _values(
        self, moved: torch.Tensor, fixed: torch.Tensor, region: torch.Tensor | None = None
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Each kept layer's weight and its distance between the two images, at the current level, over the voxels
        ``region`` holds (``[1, 1, *spatial]``, all when None)."""
        moved, fixed, mask = self._split(moved, fixed, region)
        level = self._levels[min(max(self._level, 0), len(self._levels) - 1)]
        # One draw per evaluation, the same channels in every patch and in the checkpoints' second pass.
        seed = self._draw if self._draw is not None else int(torch.randint(2**31 - 1, (1,), generator=self._generator))
        weights = [weight for index in level for weight in layer_weights([self._specs[index]])]
        if self._mode == "Jacobian" and self._sampling < 1:
            values = []
            for index in level:
                core, terms = self.cores[index], self._terms(index)
                centres = draw_centres(
                    tuple(moved.shape[2:]),
                    mask,
                    self._reach(index, tuple(moved.shape[2:])),
                    self._sampling,
                    self._generator,
                    moved.device,
                )
                if centres is None:  # a mask that keeps no point: nothing to score
                    values += [moved.sum() * 0.0] * len(terms)
                    continue
                values += list(
                    core.sampled_distances(
                        moved, fixed, centres, self._patch[index], terms, seed + 1000 * index, self._kernel
                    )
                )
        elif self._mode == "Jacobian":
            # An NCC layer pooled over the level: the inverse of the stage's weight of this piece, so that its
            # surrogate's gradient is the level's.
            piece = float(mask.sum()) if mask is not None else float(math.prod(moved.shape[2:]))
            scale = self._level_weight / max(piece, 1e-8)
            values = []
            for index in level:
                core, pooled = self.cores[index], self._layer_sums.get(index, {})
                distances = core.distances(
                    moved,
                    fixed,
                    mask,
                    self._terms(index),
                    seed + 1000 * index,
                    self._kernel,
                    self._statistics.get(index),
                    {layer: (d, v, scale) for layer, (d, v) in pooled.items()} or None,
                    self._place,
                    self._bases.get(index),
                )
                if index in self._cells:
                    # A layer's mean over the piece's cells, weighed as the stage weighs the piece (its voxels of the
                    # level, or its support): brought to the cells' share of the level's, which the whole level
                    # averages, on the layer's own grid (strided, or a model's at its voxel_size).
                    cells = mask if mask is not None else torch.ones_like(moved[:, :1])
                    held = core.layer_cells(core._prepared(cells, cells, cells, seed + 1000 * index, self._place)[2])
                    distances = torch.stack(
                        [
                            value if layer in pooled else value * held[layer] / max(total, 1e-8) * scale
                            for layer, (value, total) in enumerate(zip(distances, self._cells[index], strict=True))
                        ]
                    )
                values += list(distances)
        else:
            if self._sampling < 1:
                # The same voxels for every layer, compared as [1, C, N]: the mask is already in the draw.
                # A mask that keeps no voxel stays on the whole maps, where the masked mean is 0.
                centres = draw_centres(tuple(moved.shape[2:]), mask, 1, self._sampling, self._generator, moved.device)
                if centres is not None:
                    voxels = centres[:, 0]  # each centre's index in the flattened grid, a plane's or a volume's
                    for axis, extent in enumerate(moved.shape[3:], start=1):
                        voxels = voxels * extent + centres[:, axis]
                    moved, fixed, mask = moved.flatten(2)[:, :, voxels], fixed.flatten(2)[:, :, voxels], None
            # Half as many channels a LNCC pass when it does not fit, kept for the registration.
            values = halve_on_oom(
                lambda: self._static_distances(level, moved, fixed, mask, seed), self.narrow, moved.is_cuda
            )
        return weights, values

    def latch(self, pieces: "Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]") -> None:
        """Latch the level's normalisation on pieces of the pair instead of on the first evaluation: a registration
        by boxes hands each box read with its halo (a moved and a fixed tensor) and ones on the box's own voxels.
        Each layer's start is its mean over the voxels the boxes own, weighed by the voxels each counts (those both
        masks keep, when masked): the level's mean, however its boxes cut it. Masks that meet nowhere leave every
        layer at 1, as a layer at 0 is."""
        if not self._normalize:
            return
        totals: list[float] = []
        voxels = 0.0
        with torch.no_grad():
            for moved, fixed, region in pieces:
                if self._masked:
                    region = region * (fixed[:, -1:] * moved[:, -1:] >= 0.5).to(region.dtype)
                size = float(region.sum())
                if size == 0:  # a box the masks miss counts no voxel
                    continue
                values = [float(value) * size for value in self._values(moved, fixed, region)[1]]
                totals = [total + value for total, value in zip(totals or [0.0] * len(values), values, strict=True)]
                voxels += size
        self._latch([total / max(voxels, 1.0) for total in totals])

    def _latch(self, starts: list[float]) -> None:
        self._factors = [1.0 / start if start > 0 and np.isfinite(start) else 1.0 for start in starts]

    def forward(self, moved: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        weights, values = self._values(moved, fixed)
        if self._normalize:
            if self._factors is None:
                # Latched once a level, at the transform the scale starts from; a layer already at 0 keeps 1.
                self._latch([float(value.detach()) for value in values])
            factors = self._factors or [1.0] * len(values)  # none latched: no voxel counted
            values = [value * factor for value, factor in zip(values, factors, strict=True)]
        return reduce(torch.add, [weight * value for weight, value in zip(weights, values, strict=True)])


#: The top of the range ``_unit_range`` rescales to: under 1, so that no interpolated value reaches past 1.
_UNIT_TOP = 0.999


def _unit_range(image: sitk.Image) -> sitk.Image:
    """``image`` winsorised to its 0.5-99.5 percentiles and rescaled to [0, 0.999], for FireANTs' mutual information.

    That MI divides both images by their shared maximum and clamps below 0 before binning: each image on its own
    range, winsorised as ANTs' antsRegistrationSyN does, spreads both over the bins. Not quite 1: past 1, which the
    moved image's interpolation of the top plateau can reach, FireANTs divides both images by the moved image's
    maximum, which carries a gradient and pulls the fixed image's Parzen windowing into the backward graph.
    """
    return sitk.RescaleIntensity(winsorized(sitk.Cast(image, sitk.sitkFloat32), 0.5, 99.5), 0.0, _UNIT_TOP)


def _extent(image: "sitk.Image") -> list[float]:
    """``image``'s size in mm along its voxel axes (L, P, S once world-aligned)."""
    return [size * step for size, step in zip(image.GetSize(), image.GetSpacing(), strict=True)]


def _mask_on_grid(mask: "sitk.Image", image: "sitk.Image", device: str):
    """A FireANTs ``Image`` of ``mask`` carrying ``image``'s geometry.

    The mask is defined on the image's grid, but its header can differ by float rounding once it has
    crossed KonfAI's Attribute round-trip; FireANTs' ``concatenate`` then rejects it with a spurious
    "different physical spaces" (surfacing as a ``TypeError`` in its ``check_and_raise_cond``). Reusing
    the image's spacing/origin/direction makes the two spaces identical by construction.
    """
    from fireants.io import Image

    return Image(
        mask,
        device=device,
        spacing=image.GetSpacing(),
        origin=image.GetOrigin(),
        direction=image.GetDirection(),
    )


def _inverse_on(shape: tuple[int, ...]) -> Callable:
    """FireANTs' ``compositive_warp_inverse`` with its result brought onto ``shape`` by linear interpolation: SyN's
    ``get_warp_parameters``, handed an inverse of another size (an axis under 32 voxels, which FireANTs computes at
    32), interpolates the reverse warp itself in its place, not its inverse."""
    from fireants.utils.warputils import compositive_warp_inverse

    def inverse(image, ref_disp: torch.Tensor, **kwargs) -> torch.Tensor:
        warp = compositive_warp_inverse(image, ref_disp, **kwargs)
        if tuple(warp.shape[1:-1]) == shape:
            return warp
        return resized(warp.movedim(-1, 1), shape).movedim(1, -1)

    return inverse


class FireANTsEngine:
    """Register a fixed/moving pair with FireANTs (Rigid -> Affine -> [SyN | Greedy | none]); return
    the displacement field on the fixed grid.

    ``fireants`` is imported lazily inside :meth:`register` so this module can be imported for config
    /signature introspection (SlicerImpactReg reads the tuning knobs off the ``RegistrationNet``
    annotations) on a machine without a GPU or without FireANTs installed.
    """

    def __init__(
        self,
        scales: list[int],
        affine_iterations: list[int],
        deformable_iterations: list[int],
        cc_kernel: int,
        affine_metric: str,
        affine_lr: float,
        moments_init: str,
        linear_method: str,
        deformable_method: str,
        deformable_metric: str,
        deformable_lr: float,
        smooth_warp_sigma: float,
        smooth_grad_sigma: float,
        seed: int,
        impact_levels: list[list["ModelSpec"]],
        mode: str = "Static",
        feature_patch: int = 0,
        feature_chunk: int = 0,
        deformable_masked: bool = True,
        normalize: bool = True,
        feature_map_update_interval: int = -1,
        lncc_kernel: int = 5,
        mixed_precision: bool = False,
        voxel_sampling: float = 1.0,
        translation_lr: float = 0.015,
        transform_type: str = "native",
    ) -> None:
        """Hold one preset's registration settings; nothing is imported or allocated until ``register``.

        The settings that name a choice rather than a number -- the linear and deformable methods, the
        mode, the models -- are checked here rather than at the first run, so a typo in a preset stops before a
        card is taken.
        """
        self._scales = [int(s) for s in scales]
        self._affine_iterations = [int(i) for i in affine_iterations]
        self._deformable_iterations = [int(i) for i in deformable_iterations]
        self._cc_kernel = int(cc_kernel)
        self._affine_metric = affine_metric
        self._affine_lr = float(affine_lr)
        self._translation_lr = float(translation_lr)
        self._moments_init = moments_init
        self._linear_method = linear_method
        self._deformable_method = deformable_method
        # Checked at BUILD time: the stages they guard run for minutes.
        if linear_method not in _LINEAR_METHODS:
            # An unrecognised value would otherwise fall through to the rigid-then-affine branch.
            raise ValueError(
                f"Unknown linear_method '{linear_method}' (expected {', '.join(map(repr, _LINEAR_METHODS))})."
            )
        if cc_kernel % 2 == 0:
            # FireANTs' own cross-correlation refuses an even window, which has no centre voxel.
            raise ValueError(f"cc_kernel must be odd, got {cc_kernel}: an even window has no centre voxel.")
        if deformable_method not in ("syn", "greedy", "none"):
            raise ValueError(f"Unknown deformable_method '{deformable_method}' (expected 'syn', 'greedy' or 'none').")
        if linear_method == "none" and deformable_method == "none":
            # It would optimise nothing and return an identity no downstream check tells from a result.
            raise ValueError("linear_method='none' with deformable_method='none' leaves nothing to optimise.")
        check_transform_type(transform_type)
        self._transform_type = transform_type
        self._deformable_metric = deformable_metric
        self._deformable_lr = float(deformable_lr)
        self._smooth_warp_sigma = float(smooth_warp_sigma)
        self._smooth_grad_sigma = float(smooth_grad_sigma)
        self._seed = int(seed)
        # IMPACT deformable metric (only used when deformable_metric == "impact"): the IMPACT feature models of
        # each scale drive the SyN/greedy stage instead of the analytic CC/MI/MSE.
        self._impact_levels = impact_levels
        self._feature_loss: ImpactFeatureLoss | None = None  # built at the first registration, see _impact_loss
        self._deformable_masked = bool(deformable_masked)
        self._mode = mode
        self._feature_patch = int(feature_patch)
        self._feature_chunk = int(feature_chunk)
        self._normalize = bool(normalize)
        self._lncc_kernel = int(lncc_kernel)
        self._mixed_precision = bool(mixed_precision)
        self._voxel_sampling = float(voxel_sampling)
        if not 0 < voxel_sampling <= 1:
            raise ValueError(f"voxel_sampling is a share of the voxels, in (0, 1]: got {voxel_sampling}.")
        if mode not in ("Static", "Jacobian"):
            raise ValueError(f"Unknown mode '{mode}' (expected 'Static' or 'Jacobian').")
        if lncc_kernel % 2 == 0:
            raise ValueError(f"lncc_kernel must be odd, got {lncc_kernel}: an even window has no centre voxel.")
        for specs in impact_levels:
            # Sampled, the loss reads points: no window for an LNCC.
            check_models(specs, "FireANTs", dense=voxel_sampling >= 1)
        if feature_map_update_interval > 0 and mode != "Static":
            raise ValueError(
                "FireANTs: feature_map_update_interval refreshes the features Static mode extracts once; Jacobian mode "
                "extracts them at every step already. Leave it at -1, or set mode: Static."
            )
        self._update_interval = int(feature_map_update_interval)

    def _impact_loss(self) -> ImpactFeatureLoss:
        """The IMPACT feature models, fetched, probed and loaded by the first registration and kept: register runs
        for every case and every native tile of a run, all in one process."""
        if self._feature_loss is None:
            self._feature_loss = ImpactFeatureLoss(
                self._impact_levels,
                self._mode,
                self._normalize,
                self._lncc_kernel,
                self._feature_chunk,
                self._seed,
                self._mixed_precision,
                voxel_sampling=self._voxel_sampling,
            )
        return self._feature_loss

    @property
    def _writes_affine(self) -> bool:
        """The linear stages' affine is the result where no deformable stage follows."""
        return self._transform_type == "native" and self._deformable_method == "none"

    @staticmethod
    def _affine_to_sitk(affine_matrix: "torch.Tensor") -> sitk.AffineTransform:
        """FireANTs' physical (LPS) linear matrix -> SimpleITK AffineTransform (fixed -> moving points),
        the same convention FireANTs writes into an ANTs ``0GenericAffine.mat``."""
        matrix = affine_matrix.double().cpu().numpy()[0]
        dimension = matrix.shape[1] - 1
        affine = sitk.AffineTransform(dimension)
        affine.SetMatrix(matrix[:dimension, :dimension].flatten().astype(np.float64))
        affine.SetTranslation(matrix[:dimension, dimension].astype(np.float64))
        return affine

    @staticmethod
    def _field_on(transform: sitk.Transform, grid: sitk.Image) -> sitk.Image:
        """``transform`` as a float32 displacement field sampled on ``grid``."""
        return sitk.TransformToDisplacementField(
            transform, sitk.sitkVectorFloat32, grid.GetSize(), grid.GetOrigin(), grid.GetSpacing(), grid.GetDirection()
        )

    def _refreshed_stage(
        self,
        deformable: type,
        fixed: sitk.Image,
        moving: sitk.Image,
        fixed_mask: "sitk.Image | None",
        moving_mask: "sitk.Image | None",
        masked: bool,
        affine_matrix: "torch.Tensor | None",
        device: str,
    ) -> sitk.Image:
        """Static mode with ``feature_map_update_interval``: the deformable stage in runs of that many iterations, the
        moving features extracted again between two runs from the moving image warped by the transform so far; the
        total field on the fixed grid.

        elastix and ConvexAdam extract the moving features again under the current transform every N iterations of one
        optimisation. FireANTs' loop hands the loss neither its warp nor its images, so the stage is cut instead: each
        run starts a new SyN (or greedy) from the identity at one scale, on the moving image
        resampled through everything before it, and the fields are composed, the new run's first. Adam's state and SyN's
        midpoint start again at each run, and the layers are normalised again at the first evaluation of each run."""
        from fireants.io import BatchedImages, Image
        from fireants.io.imagemask import apply_mask_to_image, generate_image_mask_allones

        loss = self._impact_loss()
        loss._masked = masked
        loss.reset()  # this registration's draws; each run then goes on from the last one's
        runs = [
            (level, scale, min(self._update_interval, iterations - done))
            for level, (scale, iterations) in enumerate(zip(self._scales, self._deformable_iterations, strict=True))
            for done in range(0, iterations, self._update_interval)
        ]
        total: sitk.Transform | None = None
        for level, scale, iterations in runs:
            warped, warped_mask, init = moving, moving_mask, affine_matrix
            if total is not None:
                warped = sitk.Resample(moving, fixed, total, sitk.sitkLinear, 0.0, moving.GetPixelID())
                if moving_mask is not None:
                    warped_mask = sitk.Resample(moving_mask, fixed, total, sitk.sitkNearestNeighbor, 0)
                init = None
            pair = []
            for image, mask in ((fixed, fixed_mask), (warped, warped_mask)):
                stage_image = Image(image, device=device)
                if masked:
                    region = _mask_on_grid(mask, image, device) if mask is not None else None
                    if region is None:
                        region = generate_image_mask_allones(stage_image)
                    stage_image = apply_mask_to_image(stage_image, region)
                pair.append(stage_image)
            volumes = loss.extract(
                pair[0].array[:, :1],
                pair[1].array[:, :1],
                self._feature_patch,
                (_extent(fixed), _extent(warped)),
            )
            for stage_image, features in zip(pair, volumes, strict=True):
                if masked:
                    features = torch.cat([features, stage_image.array[:, -1:]], dim=1)
                stage_image.array = features
                stage_image.channels = features.shape[1]
            del volumes
            loss.release()
            loss.start_at(level)
            reg = deformable(
                scales=[scale],
                iterations=[iterations],
                fixed_images=BatchedImages([pair[0]]),
                moving_images=BatchedImages([pair[1]]),
                loss_type="custom",
                custom_loss=loss,
                cc_kernel_size=self._cc_kernel,
                deformation_type="compositive",
                smooth_warp_sigma=self._smooth_warp_sigma,
                smooth_grad_sigma=self._smooth_grad_sigma,
                optimizer="Adam",
                optimizer_lr=self._deformable_lr,
                init_affine=init,
            )
            with out_of_memory_as_torch(device != "cpu"):
                step = sitk.DisplacementFieldTransform(sitk.Cast(self._total_field(reg), sitk.sitkVectorFloat64))
            del reg, pair
            gc.collect()
            if total is None:
                total = step
            else:  # one dense field again, so that the next resampling does not walk a growing chain
                chain = sitk.CompositeTransform([total, step])
                total = sitk.DisplacementFieldTransform(sitk.Cast(self._field_on(chain, fixed), sitk.sitkVectorFloat64))
        assert total is not None
        return self._field_on(total, fixed)

    def _total_field(self, reg) -> sitk.Image:
        """Optimise a deformable stage and return its TOTAL displacement field (affine baked in), float32, on
        the fixed image FireANTs registered.

        FireANTs serialises the total field (ANTs convention, fixed grid) only to a file, so it is
        round-tripped through a temporary NIfTI: its public API, no internals reimplemented."""
        from fireants.registration import syn

        reg.optimize()
        inverse = _inverse_on(tuple(reg.fixed_images.shape[2:]))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(syn, "compositive_warp_inverse", inverse):
            # Uncompressed: the file is read back at once.
            warp_path = os.path.join(tmp, "total_warp.nii")
            if reg.dims == 2:
                # FireANTs' writer takes a 2-D field's (H, W, 2) array for a scalar volume: only 4 axes read as vectors.
                as_vectors = functools.partial(sitk.GetImageFromArray, isVector=True)
                with mock.patch.object(sitk, "GetImageFromArray", as_vectors):
                    reg.save_as_ants_transforms(warp_path)
            else:
                reg.save_as_ants_transforms(warp_path)
            return sitk.ReadImage(warp_path, sitk.sitkVectorFloat32)

    def register(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        device_index: int,
        fixed_mask: sitk.Image | None = None,
        moving_mask: sitk.Image | None = None,
    ) -> np.ndarray | sitk.Transform:
        """Register ``moving`` onto ``fixed``; return the displacement field, channel-first float32, on the fixed
        grid, or with ``transform_type: native`` and no deformable stage the linear transform the stages found, fixed points to moving points.
        """
        # A mask is a region whatever values it holds: a label map (SlicerImpactReg exports segments as labels
        # 1..N) would otherwise weight FireANTs' masked cc/mse by the product of label values, and the stages
        # below threshold it at 0, 0.5 or >= 0.5, which only agree on a 0/1 image.
        fixed_mask = None if fixed_mask is None else fixed_mask > 0
        moving_mask = None if moving_mask is None else moving_mask > 0
        if fixed_mask is not None and not sitk.GetArrayViewFromImage(fixed_mask).any():
            # A fixed mask with no voxel in it leaves nothing to register: a zero field, as the elastix engine
            # returns (in a tiled run, a tile the tissue does not reach). With a linear stage it is no such tile,
            # but a wrong mask, and said so.
            if self._linear_method != "none":
                print("[FireANTs] the fixed mask is empty: nothing registered, the field is zero.", flush=True)
            if self._writes_affine:
                return sitk.AffineTransform(fixed.GetDimension())
            return np.zeros((fixed.GetDimension(), *fixed.GetSize()[::-1]), dtype=np.float32)
        # Only a mask that restricts (some voxels in, some out) is one, decided on the mask as given: konfai-apps
        # writes an all-ones default on the FIXED grid for both sides, which would sit on another grid than the moving
        # image.
        fixed_mask = fixed_mask if is_partial_mask(fixed_mask) else None
        moving_mask = moving_mask if is_partial_mask(moving_mask) else None
        grid = fixed  # the field is sampled on the fixed image as it came
        if self._deformable_metric == "impact" and fixed.GetDimension() == 2:
            # ponytail: ImpactFeatureLoss lays its grids and patches out in 3-D; ConvexAdam swaps the models
            raise ValueError(
                "The FireANTs IMPACT presets are 3-D: register a 2-D image with ConvexAdam or FireANTs#syn."
            )
        if self._deformable_metric == "impact":
            # Feature networks compute their channels along the voxel axes they are given: both images go in with
            # their voxel axes in one order, so the fixed and moving features compare the same descriptors.
            fixed, moving, fixed_mask, moving_mask = world_aligned_pair(fixed, moving, fixed_mask, moving_mask)
        _require_fireants()
        self._check_thin(fixed.GetSize())
        if self._deformable_metric == "impact" and self._deformable_method != "none":
            loss = self._impact_loss()  # before any compute: a model that cannot be had or probed fails here
            for core in loss.cores:  # the model grids and the sampled patches are laid out in mm
                core.extent = _extent(fixed)
        from fireants.io import BatchedImages, Image
        from fireants.io.imagemask import apply_mask_to_image, generate_image_mask_allones
        from fireants.registration.affine import AffineRegistration
        from fireants.registration.rigid import RigidRegistration

        torch.manual_seed(self._seed)
        device = f"cuda:{device_index}" if device_index >= 0 else "cpu"
        # Masked metric only when a mask genuinely restricts the region; the plain path is untouched when no
        # real mask is present.
        use_fixed_mask = fixed_mask is not None
        use_moving_mask = moving_mask is not None
        masked = use_fixed_mask or use_moving_mask

        def images(metric: str, with_masks: bool) -> tuple["Image", "Image"]:
            """The fixed and moving FireANTs images of a stage scoring ``metric``: rescaled for MI (see
            ``_unit_range``), as they are for the others. FireANTs' Image takes a SimpleITK image directly, so
            they cross in memory with their geometry. Masked mode wants the mask as the last channel of BOTH
            images (all ones where one side has none) and a ``masked_`` metric prefix."""
            pair = []
            for image, mask, partial in ((fixed, fixed_mask, use_fixed_mask), (moving, moving_mask, use_moving_mask)):
                stage_image = Image(_unit_range(image) if metric == "mi" else image, device=device)
                if with_masks:
                    region = _mask_on_grid(mask, image, device) if partial else generate_image_mask_allones(stage_image)
                    stage_image = apply_mask_to_image(stage_image, region)
                pair.append(stage_image)
            return pair[0], pair[1]

        # The linear stages' images, for them alone: a tile (linear_method "none") goes straight to the deformable
        # stage, which builds its own.
        if self._linear_method != "none":
            fixed_img, moving_img = images(self._affine_metric, masked)
            bf, bm = BatchedImages([fixed_img]), BatchedImages([moving_img])
        affine_loss = f"masked_{self._affine_metric}" if masked else self._affine_metric

        # Linear: Rigid(MI) -> Affine(MI, seeded by the rigid), mirroring ANTs. The affine seeds the
        # deformable stage (or is the whole transform when deformable_method == "none").
        #
        # ``linear_method`` is ``deformable_method``'s mirror. "none" skips the linear stage entirely
        # and starts the deformable from identity: for a pair already globally aligned: a tiled
        # refinement at full resolution, where each patch sees only local anatomy, so a per-patch
        # rigid has no global meaning and neighbouring patches would each estimate a different one
        # and tear the blended field at the seams.
        affine_matrix: torch.Tensor | None
        if self._linear_method == "none":
            affine_matrix = None  # the deformable stage builds its own identity init
        else:
            # The rigid's starting translation mirrors ANTs' ``-r [fixed,moving,N]``: "cof" is the
            # centre of FRAME (N=0), "com" the centre of MASS (N=1). "cof" aligns the image frames,
            # not the subjects, so a subject sitting off its frame centre starts the chain misplaced.
            init_translation: str | torch.Tensor = "cof"
            if self._moments_init == "none":
                # Identity, not a seed: the pair arrives centred and the rigid starts where it is.
                init_translation = torch.zeros((1, fixed.GetDimension()), device=device, dtype=torch.float32)
            elif self._moments_init == "com":
                # Not FireANTs' MomentsRegistration, which mis-estimates the centre of mass on anisotropic spacings.
                init_translation = torch.tensor(
                    center_of_mass_translation(fixed, moving, fixed_mask, moving_mask),
                    device=device,
                    dtype=torch.float32,
                ).reshape(1, -1)
            rigid = RigidRegistration(
                scales=self._scales,
                iterations=self._affine_iterations,
                fixed_images=bf,
                moving_images=bm,
                loss_type=affine_loss,
                optimizer="Adam",
                optimizer_lr=1.0,
                cc_kernel_size=self._cc_kernel,
                init_translation=init_translation,
            )
            rigid.optimizer = _StepPerElement(
                rigid.optimizer, [(rigid.rotation, self._affine_lr), (rigid.transl, self._translation_lr)]
            )
            rigid.optimize()
            rigid_matrix = rigid.get_rigid_matrix().detach()
            del rigid  # and its images, once the deformable stage has built its own

            if self._linear_method == "rigid":
                # The rigid (rotation and translation, no scale or shear) IS the linear transform.
                # For inspecting the linear stage without the affine's free scale, and for a pair whose
                # scale difference is known to be nothing the affine should be free to invent.
                affine_matrix = rigid_matrix
            else:
                affine = AffineRegistration(
                    scales=self._scales,
                    iterations=self._affine_iterations,
                    fixed_images=bf,
                    moving_images=bm,
                    loss_type=affine_loss,
                    optimizer="Adam",
                    optimizer_lr=1.0,
                    cc_kernel_size=self._cc_kernel,
                    init_rigid=rigid_matrix,
                )
                sizes = torch.full_like(affine.affine, self._affine_lr)
                sizes[..., -1] = self._translation_lr  # the matrix's last column is its translation
                affine.optimizer = _StepPerElement(affine.optimizer, [(affine.affine, sizes)])
                affine.optimize()
                affine_matrix = affine.get_affine_matrix().detach()
                del affine
            # What the linear stages found, where a stage capped by its step size shows as a short shift and a
            # determinant away from 1 (rotation and shear standing in for the translation it could not reach).
            matrix = affine_matrix.double().cpu().numpy()[0]
            dimension = fixed.GetDimension()
            centre = np.asarray(fixed.TransformContinuousIndexToPhysicalPoint([(n - 1) / 2 for n in fixed.GetSize()]))
            shift = matrix[:dimension, :dimension] @ centre + matrix[:dimension, dimension] - centre
            print(
                f"[FireANTs] linear stage: {np.round(shift, 2).tolist()} voxels at the fixed image centre, "
                f"determinant {np.linalg.det(matrix[:dimension, :dimension]):.3f}.",
                flush=True,
            )

        # Deformable stage (or none). SyN and Greedy share the same constructor surface; both warm-start
        # from the affine so their TOTAL transform already bakes in the linear pre-align.
        if self._deformable_method == "none":
            linear = self._affine_to_sitk(affine_matrix)
            if self._writes_affine:
                return linear
            field = self._field_on(linear, grid)
        else:
            if self._deformable_method == "syn":
                from fireants.registration.syn import SyNRegistration as Deformable
            else:
                from fireants.registration.greedy import GreedyRegistration as Deformable
            # The linear stage's images go before this stage builds its own: FireANTs keeps its tensors in
            # reference cycles, which only the cyclic collector frees.
            if self._linear_method != "none":
                del bf, bm, fixed_img, moving_img
                gc.collect()
            # "impact" swaps the analytic metric for a KonfAI IMPACT feature loss on the deformable stage
            # (the linear pre-align keeps its own affine_metric); the fixed mask restricts it too.
            loss_type: str
            # The masks drive the deformable stage only when the caller asks: a tight mask hides the
            # outline the stage needs to pull an end into place.
            deformable_masked = masked and self._deformable_masked
            if self._update_interval > 0 and self._deformable_metric == "impact":
                field = self._refreshed_stage(
                    Deformable, fixed, moving, fixed_mask, moving_mask, deformable_masked, affine_matrix, device
                )
            else:
                custom_loss: ImpactFeatureLoss | None = None
                if self._deformable_metric == "impact" and self._mode == "Static":
                    # Static: extract once from the images' intensity channel, then register the feature volumes,
                    # with the mask channel concatenated back so ``masked_`` still means what it says. The loss then
                    # compares the volumes layer by layer, as Jacobian mode compares the networks' outputs.
                    custom_loss = self._impact_loss()
                    fixed_img, moving_img = images("impact", deformable_masked)
                    volumes = custom_loss.extract(
                        fixed_img.array[:, :1],
                        moving_img.array[:, :1],
                        self._feature_patch,
                        tuple(_extent(image) for image in (fixed, moving)),
                    )
                    for image, features in ((fixed_img, volumes[0]), (moving_img, volumes[1])):
                        if deformable_masked:
                            features = torch.cat([features, image.array[:, -1:]], dim=1)
                        image.array = features
                        image.channels = features.shape[1]
                    del volumes, features  # the unmasked volumes, a copy each once the mask channel was appended
                    custom_loss.release()
                    custom_loss._masked = deformable_masked  # this registration's masks, not the last one's
                    bf, bm = BatchedImages([fixed_img]), BatchedImages([moving_img])
                    loss_type = "custom"
                else:
                    # Its own images: the metric may want other intensities than the linear stage's, and the masks
                    # may be the linear stage's alone.
                    fixed_img, moving_img = images(self._deformable_metric, deformable_masked)
                    bf, bm = BatchedImages([fixed_img]), BatchedImages([moving_img])
                    if self._deformable_metric == "impact":
                        loss_type = "custom"
                        feature_loss = self._impact_loss()
                        feature_loss._masked = deformable_masked  # this registration's masks, not the last one's
                        custom_loss = feature_loss
                    else:
                        loss_type = (
                            f"masked_{self._deformable_metric}" if deformable_masked else self._deformable_metric
                        )
                restarts = 0

                def run() -> sitk.Image:
                    if custom_loss is not None:
                        custom_loss.reset()  # the stage starts from its first scale, a restart included
                    reg = Deformable(
                        scales=self._scales,
                        iterations=self._deformable_iterations,
                        fixed_images=bf,
                        moving_images=bm,
                        loss_type=loss_type,
                        custom_loss=custom_loss,
                        cc_kernel_size=self._cc_kernel,
                        deformation_type="compositive",
                        smooth_warp_sigma=self._smooth_warp_sigma,
                        smooth_grad_sigma=self._smooth_grad_sigma,
                        optimizer="Adam",
                        optimizer_lr=self._deformable_lr,
                        init_affine=affine_matrix,
                    )
                    return self._total_field(reg)

                def narrow() -> bool:
                    # Our losses retry a forward that runs out of memory; the backward runs in FireANTs' own loop, out
                    # of their reach. The stage starts over with smaller pieces rather than leaving konfai to cut the
                    # registration into patches, each with its own linear stages. Twice at most: an out of memory the
                    # loss does not cause (SyN's inversion, the warp) would otherwise rerun the whole stage until the
                    # loss's pieces are 16 voxels.
                    nonlocal restarts
                    if custom_loss is None or restarts == 2 or not custom_loss.narrow():
                        return False
                    restarts += 1
                    print("[FireANTs] out of memory in the deformable stage: starting it over in smaller pieces.")
                    return True

                field = halve_on_oom(run, narrow, device != "cpu")
            if fixed is not grid:
                # FireANTs registered a reordered copy (the IMPACT metric): the field is sampled back on the
                # fixed image as it came. Otherwise it is already on it, and taken as it is, in float32.
                field = self._field_on(sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64)), grid)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dvf_np, _ = image_to_data(field)
        return dvf_np

    def check_sampled(self, fixed_shape, moving_shape) -> None:
        """Refuse what ``register_sampled`` cannot register, from the pair's voxel shapes ((z, y, x)) alone, before
        anything is read."""
        _require_fireants()
        # A volume one voxel thick along one axis registers as its plane (``register_sampled``), as on the whole pair.
        plane = len(fixed_shape) == 3 and list(fixed_shape).count(1) == 1
        if 1 in (*fixed_shape, *moving_shape) and not plane:
            raise NotImplementedError(
                "an image with an axis of one voxel has no extent along it for FireANTs' normalized coordinates:"
                " by boxes, a volume one voxel thick along one axis registers as its plane, and nothing thinner."
            )
        if self._deformable_method == "none":
            return
        if self._deformable_metric == "impact" and self._mode == "Jacobian" and self._voxel_sampling >= 1:
            # By boxes, a Jacobian NCC layer reduced by PCA takes the basis of the whole fixed map from the boxes'
            # sums; a 2-D network's is fitted on slices spread over the map, which no box holds.
            for spec, core in zip(self._impact_loss()._specs, self._impact_loss().cores, strict=True):
                if spec.pca > 0 and spec.distance == "NCC" and core.dimension < DIM:
                    raise ConfigError(
                        f"FireANTs by boxes cannot score '{spec.ref}' in Jacobian mode: a 2-D network's NCC layer with"
                        f" pca {spec.pca} fits its basis on slices spread over the whole level.",
                        "Set mode: Static or pca: 0, or choose --out-of-core whole.",
                    )
        self._check_thin([extent for extent in fixed_shape if extent != 1] if plane else fixed_shape)

    def _check_thin(self, size) -> None:
        """FireANTs sizes its warp no smaller than MIN_IMG_SIZE a side: greedy then scores that warp against a fixed
        image with a thinner axis, and SyN cannot invert a warp larger than the image on every axis. Both fail only
        once the optimisation is over, so the sizes are checked before it starts, whole or by boxes."""
        from fireants.utils.globals import MIN_IMG_SIZE

        thin = [extent < MIN_IMG_SIZE for extent in size]
        if self._deformable_method == "greedy" and any(thin):
            raise ValueError(
                f"FireANTs' greedy registration needs every axis of the fixed image to span at least {MIN_IMG_SIZE} "
                f"voxels, got {tuple(size)}: use deformable_method 'syn', which takes a thin axis."
            )
        if self._deformable_method == "syn" and all(thin):
            raise ValueError(
                f"FireANTs' SyN needs one axis of the fixed image to span at least {MIN_IMG_SIZE} voxels, got "
                f"{tuple(size)}: register the whole image, or tiles of at least {MIN_IMG_SIZE} a side."
            )

    def register_sampled(
        self,
        fixed: Path,
        moving: Path,
        fixed_mask: Path | None,
        moving_mask: Path | None,
        device_index: int,
        scratch: Path,
        field: Path,
        box: int,
        samples: int | None = None,
        epochs: int | None = None,
        plane: int | None = None,
    ) -> dict:
        """Register the uncompressed MetaImage ``moving`` onto ``fixed`` (float32, their masks binary, on their
        image's grid) without ever holding an image or a warp whole: the images are memory mapped and read by boxes,
        the pyramid levels and the warps live in files of ``scratch``.

        ``samples`` is how many voxels an iteration draws, in random boxes drawn anew at every iteration, as elastix
        draws its points: an iteration's memory and time follow them and not the image. Unset, every stage takes
        every box of a level at every iteration, on boxes as large as the device holds, which is FireANTs' own
        iteration. Drawn, the rigid and affine stages run the preset's iterations on their draws, their step falling
        as elastix's gain does, which takes them to the whole pair's optimum as the iterations grow; the deformable
        stage steps the boxes it draws, each box as often on average as the preset gives a scale iterations
        (``epochs`` for every scale, when given). A level the samples hold whole is registered as FireANTs registers
        it. The displacement field, physical, on the fixed grid, is written to ``field`` block by block. Returns what
        the run measured.

        A fixed volume one voxel thick along an axis is registered as its plane (``plane``, that numpy axis), the
        moving one's first slice along it, as ``EngineRegistration`` registers it on the whole pair: the plane's field
        comes back one voxel thick, along the dropped axis the shift from one plane to the other."""
        from ..native import (
            PlaneHeader,
            field_map,
            lift_plane_field,
            mapped_centre,
            mapped_window,
            mmap_image,
            plane_of,
            read_information,
            write_field,
        )
        from ..ooc import ArraySource, HeaderImages, OnesSource, RegionSource, RescaledSource, StackedSource, tiles
        from .fireants_sampled import (
            _TOGETHER,
            read_through,
            sampled_deformable_class,
            sampled_linear_alignment,
            static_feature_sources,
        )

        shape = mmap_image(fixed).array.shape
        self.check_sampled(shape, mmap_image(moving).array.shape)
        if plane is None and len(shape) == 3 and shape.count(1) == 1:
            flat = scratch / "plane_field.mha"
            scratch.mkdir(parents=True, exist_ok=True)
            arguments = (fixed, moving, fixed_mask, moving_mask, device_index, scratch, flat, box, samples, epochs)
            measured = self.register_sampled(*arguments, plane=shape.index(1))
            lift_plane_field(flat, field, mmap_image(fixed), mmap_image(moving), shape.index(1))
            flat.unlink()
            return measured
        torch.manual_seed(self._seed)
        device = f"cuda:{device_index}" if device_index >= 0 else "cpu"
        mapped = {"fixed": mmap_image(fixed), "moving": mmap_image(moving)}
        masks = {
            side: mmap_image(path) if path is not None else None
            for side, path in (("fixed", fixed_mask), ("moving", moving_mask))
        }
        masked = any(mask is not None for mask in masks.values())
        readers = {"fixed": read_information(fixed), "moving": read_information(moving)}
        if plane is not None:
            mapped = {side: plane_of(image, plane) for side, image in mapped.items()}
            masks = {side: None if mask is None else plane_of(mask, plane) for side, mask in masks.items()}
            readers = {side: PlaneHeader(reader, plane) for side, reader in readers.items()}

        def mask_source(side: str):
            """A side's mask, read by boxes: all ones where it has none (masked mode wants one beside both images)."""
            if masks[side] is None:
                return OnesSource(mapped[side].array.shape, device)
            return ArraySource(masks[side].array[None], device)

        def sources(metric: str, with_masks: bool) -> list:
            """The fixed and moving images of a stage scoring ``metric``, read by boxes: rescaled for MI (see
            ``_unit_range``, the percentiles taken by passes over the twin), with the mask as a last channel of both in
            masked mode (all ones where a side has none)."""
            pair = []
            for side in ("fixed", "moving"):
                source = ArraySource(mapped[side].array[None], device)
                if metric == "mi":
                    source = RescaledSource(source, *mapped_window(mapped[side]), _UNIT_TOP)
                pair.append(StackedSource([source, mask_source(side)]) if with_masks else source)
            return pair

        scratch.mkdir(parents=True, exist_ok=True)
        started, measured = time.perf_counter(), {}
        affine_matrix: torch.Tensor | None = None
        if self._linear_method != "none":
            init_translation: str | torch.Tensor = "cof"
            if self._moments_init == "none":
                init_translation = torch.zeros((1, mapped["fixed"].array.ndim), device=device, dtype=torch.float32)
            elif self._moments_init == "com":
                centres = [mapped_centre(mapped[side], masks[side], "com") for side in ("moving", "fixed")]
                init_translation = torch.tensor(np.subtract(*centres), device=device, dtype=torch.float32).reshape(
                    1, -1
                )
            pair = sources(self._affine_metric, masked)
            affine_matrix = sampled_linear_alignment(
                readers["fixed"],
                readers["moving"],
                *pair,
                scales=self._scales,
                iterations=self._affine_iterations,
                loss_type=f"masked_{self._affine_metric}" if masked else self._affine_metric,
                optimizer_lr=self._affine_lr,
                translation_lr=self._translation_lr,
                cc_kernel_size=self._cc_kernel,
                init_translation=init_translation,
                device=device,
                # Every box of a level, unless told how many voxels to draw: FireANTs' own iteration. Drawn, small
                # boxes in many places: a global metric is estimated on them, and boxes side by side tell it less than
                # boxes apart.
                box=box if samples is None else max(box // 3, 8),
                boxes=len(tiles(mapped["fixed"].array.shape, box))
                if samples is None
                else max(1, samples // max(box // 3, 8) ** mapped["fixed"].array.ndim),
                together=samples or _TOGETHER,
                seed=self._seed,
                affine=self._linear_method != "rigid",
                scratch=scratch,
            )
            measured["linear_seconds"] = time.perf_counter() - started
            measured["bytes_read"] = sum(source.bytes_read for source in pair)

        if self._deformable_method == "none":
            write_field(field, mapped["fixed"], self._affine_to_sitk(affine_matrix))
            return measured
        deformable_masked = masked and self._deformable_masked
        custom_loss, loss_halo = None, None
        if self._deformable_metric == "impact":
            custom_loss = self._impact_loss()
            custom_loss._masked = deformable_masked
            custom_loss.reset()
            for core in custom_loss.cores:  # a 2-D pair: planes, which a 3-D network lifts to volumes one voxel thick
                core.model.planes = mapped["fixed"].array.ndim == 2
            extent = _extent(readers["fixed"])
            for core in custom_loss.cores:
                core.extent = extent
            loss_type = "custom"
            if self._mode == "Static":
                # The loss compares the volumes' channels: twice the radius of a local correlation's window.
                loss_halo = 2 * ((self._lncc_kernel - 1) // 2)
            else:
                # A network's feature at a voxel reads its receptive field around it, and so does the feature's
                # gradient: twice its radius; half a box for a model that sees whole images, as patches overlap.
                # A model with a voxel_size reads its field in its own cells, a stride more either side for a box to
                # start on it (``_box_on_model_grid``): counted in the image's voxels.
                loss_halo = 0
                for core in custom_loss.cores:
                    try:
                        reach = core.model.receptive_field - 1
                        if core.model.voxel_size is not None:
                            sizes, spacing = core.model.voxel_size, readers["fixed"].GetSpacing()
                            # A network of another dimension than the pair's (a 2-D one sweeping any axis of a volume)
                            # has its values along any of the pair's axes.
                            ratio = (
                                max(size / step for size, step in zip(sizes, spacing, strict=True))
                                if len(sizes) == len(spacing)
                                else max(sizes) / min(spacing)
                            )
                            reach = math.ceil((reach + 2 * core.model.block_stride(device)) * max(ratio, 1.0))
                        loss_halo = max(loss_halo, reach)
                    except MeasureError:
                        loss_halo = max(loss_halo, box // 2)
        else:
            loss_type = f"masked_{self._deformable_metric}" if deformable_masked else self._deformable_metric
        deformable_started = time.perf_counter()
        iterations = [int(epochs)] * len(self._scales) if epochs else list(self._deformable_iterations)

        def stage(scales: list[int], counts: list[int], pair: list, moving_reader, init, out: Path, seed: int):
            """The deformable stage over boxes of ``pair``, its field written to ``out`` on the fixed grid, its draws
            from ``seed``."""
            registration = sampled_deformable_class()(
                scales,
                counts,
                HeaderImages(readers["fixed"], pair[0].channels, device),
                HeaderImages(moving_reader, pair[0].channels, device),
                *pair,
                scratch=scratch,
                symmetric=self._deformable_method == "syn",
                loss_type=loss_type,
                custom_loss=custom_loss,
                cc_kernel_size=self._cc_kernel,
                loss_halo=loss_halo,
                smooth_warp_sigma=self._smooth_warp_sigma,
                smooth_grad_sigma=self._smooth_grad_sigma,
                optimizer_lr=self._deformable_lr,
                init_affine=init,
                box=box,
                samples=samples,
                seed=seed,
                # The points of a sampled Jacobian loss read a patch of the receptive field around them, which a box
                # at the image's edge, short and with no halo past it, would not hold.
                least=max(getattr(custom_loss, "_patch", None) or [1]),
            )
            try:
                registration.optimize()
                written = field_map(out, mapped["fixed"])
                registration.write_field(written)
                written.flush()
            finally:
                registration.close()
            return registration, written

        def static_pair(moving_source: RegionSource, moving_mask: RegionSource, moving_reader) -> list:
            """Static: the feature volumes extracted by blocks into files, with the mask channel beside them so that
            masked still means what it says."""
            features = static_feature_sources(
                custom_loss,
                sources("impact", False)[0],
                moving_source,
                scratch,
                block=max(box, 64),
                extents=(_extent(readers["fixed"]), _extent(moving_reader)),
            )
            custom_loss.release()
            if not deformable_masked:
                return list(features)
            return [StackedSource([features[0], mask_source("fixed")]), StackedSource([features[1], moving_mask])]

        refresh = custom_loss is not None and self._mode == "Static" and self._update_interval > 0
        if not refresh:
            pair = sources(self._deformable_metric, deformable_masked)
            if custom_loss is not None and self._mode == "Static":
                pair = static_pair(sources("impact", False)[1], mask_source("moving"), readers["moving"])
            registration, _ = stage(self._scales, iterations, pair, readers["moving"], affine_matrix, field, self._seed)
        else:
            # feature_map_update_interval, as a run on the whole pair (``_refreshed_stage``): runs of that many
            # iterations at one scale, each from the identity on the moving image read through the field so far,
            # its features extracted again, and the fields composed, the new run's first.
            runs = [
                (level, scale, min(self._update_interval, count - done))
                for level, (scale, count) in enumerate(zip(self._scales, iterations, strict=True))
                for done in range(0, count, self._update_interval)
            ]
            fixed_geometry, extents = mapped["fixed"].geometry, mapped["fixed"].array.shape
            total: np.memmap | None = None
            held_files: list[Path] = []  # what the field so far rests on, gone once the next one stands

            def read_moved(image, field_so_far: np.ndarray, path: Path, nearest: bool = False) -> ArraySource:
                """``image`` read through the field so far onto the fixed grid, into a float32 file at ``path``."""
                out = np.lib.format.open_memmap(str(path), mode="w+", dtype=np.float32, shape=(1, *extents))
                source = ArraySource(image.array[None], device)
                read_through(source, image.geometry, field_so_far, fixed_geometry, out, block=box, nearest=nearest)
                return ArraySource(out, device)

            for number, (level, scale, count) in enumerate(runs):
                moving_source, moving_mask, moving_reader, init = (
                    sources("impact", False)[1],
                    mask_source("moving"),
                    readers["moving"],
                    affine_matrix,
                )
                if total is not None:
                    moving_source = read_moved(mapped["moving"], total, scratch / f"moved_{number}.npy")
                    moving_reader, init = readers["fixed"], None
                    moving_mask = OnesSource(extents, device)
                    if masks["moving"] is not None:
                        path = scratch / f"moved_mask_{number}.npy"
                        moving_mask = read_moved(masks["moving"], total, path, nearest=True)
                custom_loss.start_at(level)
                pair = static_pair(moving_source, moving_mask, moving_reader)
                # Each run draws boxes of its own, as one stage's iterations draw anew.
                step_path = scratch / f"run_{number}.mha"
                registration, step = stage([scale], [count], pair, moving_reader, init, step_path, self._seed + number)
                if total is None:
                    total, held_files = step, [step_path]
                    continue
                composed = field_map(scratch / f"total_{number}.mha", mapped["fixed"])
                previous = ArraySource(np.moveaxis(total, -1, 0), device)
                read_through(previous, fixed_geometry, step, fixed_geometry, np.moveaxis(composed, -1, 0), block=box)
                for start, stop in tiles(extents, box):
                    region = tuple(slice(a, b) for a, b in zip(start, stop, strict=True))
                    composed[region] += step[region]
                composed.flush()
                del total, step, moving_source, moving_mask
                for path in [*held_files, step_path, *scratch.glob(f"moved*_{number}.npy")]:
                    path.unlink(missing_ok=True)
                total, held_files = composed, [scratch / f"total_{number}.mha"]
            assert total is not None
            del total
            # The last field, a field_map of the fixed grid as ``field`` is: its file is the result.
            (last,) = held_files
            shutil.move(last, field)
        measured["deformable_seconds"] = time.perf_counter() - deformable_started
        measured["bytes_read"] = measured.get("bytes_read", 0) + sum(source.bytes_read for source in pair)
        measured["loss"] = [registration.losses[0], registration.losses[-1]] if registration.losses else []
        measured["warps_resident"] = registration.resident
        measured["boxes_per_scale"] = registration.boxes_per_scale
        return measured


class RegistrationNet(network.Network):
    """Pairwise FireANTs registration as an ``add_module`` graph (fixed = branch 0, moving = branch 1,
    fixed mask = 2, moving mask = 3; masks restrict the metric, whole-image = no restriction).

    Output on the fixed grid: ``DisplacementField``
    (the DIM-component displacement field, in mm). Geometry is attached by the predictor via
    ``same_as_group: Volume_0:Fixed``. The knobs below are read straight from these annotations by the
    UI: ``Annotated[.., Range]`` gives numeric spin bounds; ``Literal`` a dropdown. ``deformable_method``
    is the knob that specialises this shared model into each FireANTs preset.
    """

    #: Its output is a displacement field in world units: KonfAI blends it in float32, not float16.
    full_precision_outputs = True

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default:ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        scales: Annotated[
            list[int],
            "Multi-resolution pyramid: downsampling factor per level, coarse to fine (e.g. [4,2,1]); the "
            "affine/deformable iteration lists are indexed by these levels.",
        ] = [4, 2, 1],
        affine_iterations: Annotated[
            list[int],
            "Iterations per pyramid level (one entry per 'scales' level) of the rigid stage, and again of the "
            "affine stage.",
        ] = [200, 100, 50],
        deformable_iterations: Annotated[
            list[int], "Deformable-stage iterations per pyramid level (one entry per 'scales' level)."
        ] = [200, 100, 50],
        cc_kernel: Annotated[
            int,
            Choices(list(range(1, 22, 2))),
            "Side (voxels, odd) of the local cross-correlation window when a 'cc' metric is used; larger = more "
            "spatial context, slower.",
        ] = 5,
        affine_metric: Annotated[
            Literal["mi", "cc", "mse"], "Similarity metric optimised by the rigid and affine (global) stages."
        ] = "mi",
        affine_lr: Annotated[
            float,
            Range(0.0, 10.0),
            "Adam's step per iteration for the rotation (rigid stage) and the matrix (affine stage), unitless; "
            "the translation has its own, translation_lr.",
        ] = 0.03,
        translation_lr: Annotated[
            float,
            Range(0.0, 10.0),
            "Adam's step per iteration for the translation of the rigid and affine stages, in voxels of the fixed "
            "image. Each stage moves it at most about translation_lr x sum(affine_iterations): 5 voxels at 0.015 over "
            "350 iterations.",
        ] = 0.015,
        moments_init: Annotated[
            Literal["cof", "com", "none"],
            "Initial translation seeding the rigid stage, mirroring ANTs' -r [fixed,moving,N]: 'cof' = centre "
            "of frame (N=0), 'com' = intensity-weighted centre of mass (N=1). Prefer 'com' when the subject "
            "sits off its frame centre, where aligning frames starts the chain misplaced. 'none' seeds "
            "nothing and starts the rigid from identity: for a pair the caller has already centred, where "
            "'com' re-estimates a translation that is zero and 'cof' undoes the centring by aligning "
            "frames instead of subjects.",
        ] = "cof",
        linear_method: Annotated[
            Literal["rigid_affine", "rigid", "none"],
            "Linear stage before the deformable: 'rigid_affine' (rigid then affine, as ANTs), 'rigid' to stop "
            "after the rigid and leave scale and shear alone, or 'none' to skip it and start the deformable "
            "from identity. Use 'none' only on a pair already globally aligned: a tiled pass at full "
            "resolution, where a patch sees no global context and a per-patch linear has no global meaning.",
        ] = "rigid_affine",
        deformable_method: Annotated[
            Literal["none", "syn", "greedy"],
            "Deformable algorithm: 'syn' (symmetric diffeomorphic), 'greedy', or 'none' to stop after the linear "
            "stage, whichever 'linear_method' selected, with linear_method='rigid' that is a rigid-only "
            "registration. Both set to 'none' is refused: it would optimise nothing.",
        ] = "syn",
        transform_type: Annotated[
            Literal["native", "field"],
            "The form the transform is written in: 'native', the one the stages found (the linear stages' affine for "
            "deformable_method 'none', written as an ITK AffineTransform file or an RFC-5 'affine' store, twelve "
            "numbers where a field weighs megabytes; a field otherwise); 'field', its displacement on the fixed grid. "
            "A mean (TTA, an ensemble) or a tiled run writes the field whatever the form.",
        ] = "native",
        deformable_metric: Annotated[
            Literal["cc", "mi", "mse", "impact"],
            "Similarity metric for the deformable stage; 'impact' uses the IMPACT feature models under 'models'.",
        ] = "cc",
        deformable_lr: Annotated[float, Range(0.0, 10.0), "Gradient step size of the deformable optimisation."] = 0.25,
        smooth_warp_sigma: Annotated[
            float,
            Range(0.0, 100.0),
            "Gaussian sigma (voxels) smoothing the displacement/warp field each step; higher = smoother, more "
            "regular deformation.",
        ] = 0.5,
        smooth_grad_sigma: Annotated[
            float,
            Range(0.0, 100.0),
            "Gaussian sigma (voxels) smoothing the update gradient each step; higher = more stable but slower "
            "convergence.",
        ] = 1.0,
        seed: Annotated[
            int,
            "Seed of what the IMPACT loss draws: the channels of subset_features, the voxels of voxel_sampling, a 2D "
            "network's swept axis (dense Jacobian) or planes (sampled Jacobian). FireANTs' own registrations draw "
            "nothing at random.",
        ] = 42,
        mode: Mode = "Static",
        feature_patch: Annotated[
            int,
            "Static only: the cube of voxels each feature extraction pass sees, 0 for the whole image "
            "at once, or on a GPU the widest tile its free memory holds when the whole image does not fit.",
            Range(0, 1024),
        ] = 0,
        feature_chunk: Annotated[
            int,
            "Static only: how many channels of a layer the LNCC distance compares at a time, 0 "
            "for all of them at once. A smaller chunk trades a little time for a peak that follows the chunk "
            "instead of the channel count, which is what lets several feature models share one card; a pass "
            "that runs out of memory is retried with half as many channels.",
            Range(0, 64),
        ] = 0,
        deformable_masked: Annotated[
            bool,
            "Restrict the deformable metric to the masks as well. False keeps them for the centre of mass, rigid "
            "and affine only: a tight or ragged mask hides the subject's outline, and the deformable stage "
            "cannot pull into place an end it does not see.",
        ] = True,
        models: dict[str, ModelSpec] = {},
        levels: Annotated[
            dict[str, LevelSpec],
            "The IMPACT models of each 'scales' level ('0', '1', ...), in place of 'models' there; empty = 'models' "
            "at every level.",
        ] = {},
        normalize: Normalize = True,
        feature_map_update_interval: FeatureMapUpdateInterval = -1,
        lncc_kernel: LNCCKernel = 5,
        mixed_precision: MixedPrecision = False,
        voxel_sampling: VoxelSampling = 1.0,
    ) -> None:
        """Build the graph a FireANTs preset runs: registration, then the moved image and the field.

        Every argument is a preset knob; the annotations beside them are what SlicerKonfAI renders.
        """
        super().__init__(
            in_channels=1,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            dim=3,
        )
        # Fail at build time: with no feature model the IMPACT loss would surface as a None-loss crash
        # deep in the deformable stage, minutes after the rigid/affine stages already ran. With
        # deformable_method 'none' the metric is never consumed, so a stale 'impact' stays harmless.
        impact = deformable_method != "none" and deformable_metric == "impact"
        if impact and not models and not levels:
            raise ValueError("deformable_metric='impact' requires at least one feature model under 'models'.")
        engine = FireANTsEngine(
            scales,
            affine_iterations,
            deformable_iterations,
            cc_kernel,
            affine_metric,
            affine_lr,
            moments_init,
            linear_method,
            deformable_method,
            deformable_metric,
            deformable_lr,
            smooth_warp_sigma,
            smooth_grad_sigma,
            seed,
            level_models(models, levels, len(scales), "FireANTs") if impact else [],
            mode,
            feature_patch,
            feature_chunk,
            deformable_masked,
            normalize,
            feature_map_update_interval,
            lncc_kernel,
            mixed_precision,
            voxel_sampling,
            translation_lr=translation_lr,
            transform_type=transform_type,
        )
        self.add_module("Registration", EngineRegistration(engine), in_branch=[0, 1, 2, 3], out_branch=["registration"])
        # The output module the presets name.
        self.add_module("DisplacementField", torch.nn.Identity(), in_branch=["registration"], out_branch=["dvf"])

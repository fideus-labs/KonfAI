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

import gc
import math
import os
import tempfile
from collections.abc import Callable
from functools import reduce
from typing import Annotated, Literal, cast

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
    no_texpr_fuser,
    normalized_features,
    onto_image_grid,
    pca_project,
    resampled,
    swept_order,
)
from konfai.network import network
from konfai.utils.config import Choices, Range
from konfai.utils.dataset import image_to_data
from konfai.utils.errors import MeasureError
from konfai.utils.vram import halve_on_oom, out_of_memory_as_torch

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
from .intensity import EngineRegistration, is_partial_mask, winsorized
from .orientation import world_aligned_pair

DIM = 3

#: Linear stages a preset may ask for. The engine checks against this rather than trusting the
#: ``Literal`` annotation, which only binds a config-driven call and not a direct Python one.
_LINEAR_METHODS = ("rigid_affine", "rigid", "none")


def _on_model_grid(
    moved: torch.Tensor, fixed: torch.Tensor, mask: torch.Tensor | None, size: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """The pair resampled on a model's grid (``resampled``), its mask by nearest neighbour."""
    return resampled(moved, size), resampled(fixed, size), None if mask is None else resampled(mask, size, "nearest")


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
    project: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None,
    seed: int,
    kernel: int,
) -> torch.Tensor:
    """Each kept layer's distance on one patch, stacked: the network run on both images, each layer normalised, reduced
    by PCA, cut to its channel subset and compared, in itk-impact's order."""
    moved_layers, fixed_layers = model(moved, *moved_rest), model(fixed, *fixed_rest)
    values = []
    for index, (layer, (name, subset)) in enumerate(zip(kept, terms, strict=True)):
        moved_features = normalized_features(moved_layers[layer].float(), normalization)
        fixed_features = normalized_features(fixed_layers[layer].float(), normalization)
        if project is not None:
            moved_features, fixed_features = project(moved_features, fixed_features)
        moved_features, fixed_features = channel_subset(moved_features, fixed_features, subset, seed + index)
        layer_mask = None
        if mask is not None:
            layer_mask = torch.nn.functional.interpolate(mask.float(), size=moved_features.shape[2:], mode="nearest")
        values.append(distance(name, moved_features, fixed_features, layer_mask, kernel, 0))
    return torch.stack(values)


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
    ) -> torch.Tensor:
        """Each kept layer's distance, averaged over the patches the mask reaches (the whole image when untiled)."""
        network = self.model.network(moved.device)
        moved_inputs, fixed_inputs = self.model.inputs(moved, statistics[0]), self.model.inputs(fixed, statistics[1])
        project = self._pca_project if self.pca > 0 else None
        total: torch.Tensor | None = None
        count = 0
        for moved_patch, fixed_patch, mask_patch in _patch_views(
            moved_inputs[0], fixed_inputs[0], mask, self.model.shape
        ):
            if mask_patch is not None and not torch.any(mask_patch == 1):
                continue
            args = (network, moved_patch, fixed_patch, mask_patch, moved_inputs[1:], fixed_inputs[1:], self.kept, terms)
            args += (self.normalization, project, seed, kernel)
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
    ) -> torch.Tensor:
        """Each kept layer's distance over the whole image, or over tiles once the whole image has run out of memory.

        A level that runs out is scored again in tiles half as wide, and in tiles half as wide again if it
        still does not fit; the tile is kept for that level, so the retry happens once per registration.
        This keeps the registration one global registration, where cutting the volume would estimate its
        rigid and affine stages patch by patch.
        """
        voxel_size = self.model.voxel_size
        if voxel_size is not None and self.dimension == DIM:
            moved, fixed, mask = _on_model_grid(
                moved, fixed, mask, grid_size(self.extent, voxel_size, tuple(moved.shape[2:]))
            )
        if self.dimension < DIM:
            # Swept along an axis drawn at each evaluation (``swept_order``).
            order = swept_order(seed)
            if voxel_size is not None:
                # The two values are the slices' own: the swept axis keeps this level's, the other two the model's.
                sides = list(reversed(self.extent)) if self.extent is not None else [float(n) for n in moved.shape[2:]]
                size = [0, 0, 0]
                size[order[2] - 2] = moved.shape[order[2]]
                for axis, step in zip(order[3:], voxel_size, strict=True):
                    size[axis - 2] = max(1, int(sides[axis - 2] / step + 0.5))
                moved, fixed, mask = _on_model_grid(moved, fixed, mask, tuple(size))
            moved, fixed, mask = (
                moved.permute(order),
                fixed.permute(order),
                None if mask is None else mask.permute(order),
            )
        shape = self._last_shape = tuple(moved.shape[2:])
        statistics = (_statistics(moved)[0], _statistics(fixed)[0])

        def run() -> torch.Tensor:
            tile = self._tiles.get(shape)
            # Each tile under a checkpoint: the optimiser differentiates through the network for BOTH warped images
            # (SyN warps the fixed one too), so a whole-image pass keeps two networks' activations until the backward.
            # The intensity statistics stay the whole image's, so every tile is normalised alike.
            self.model.shape, self.model.checkpoint = tile, tile is not None
            return self._scored(moved, fixed, mask, statistics, terms, seed, kernel)

        with no_texpr_fuser():
            return halve_on_oom(run, self._narrow_tile, moved.is_cuda)

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
            if mode == "Jacobian":
                # Loaded here once, on the CPU, and kept: the registration moves it to its device.
                core.model.model = torch.jit.load(core.model.model_path, map_location="cpu").eval()  # nosec B614
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

    @property
    def cores(self) -> list[_ImpactCore]:
        return [cast("_ImpactCore", core) for core in self._cores]

    def reset(self) -> None:
        """A registration (or a restarted stage) begins: the next scale FireANTs announces is the first level, and
        the draws start again from the seed."""
        self._level, self._factors = -1, None
        self._generator.manual_seed(self._seed)

    def set_current_scale_and_iterations(self, scale: int, iterations: int) -> None:
        """FireANTs' hook at the start of every scale: the next level, whose layers start at 1 again."""
        self._level, self._factors = self._level + 1, None

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
        overlap: float,
        extents: tuple[list[float] | None, list[float] | None] = (None, None),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Static mode: the fixed and moving volumes of every model's kept layers, concatenated along the channels.

        A model with a ``voxel_size`` extracts on the image resampled at that resolution (``extents``: each image's
        size in mm along L, P, S) and its features are brought back onto the image's grid, as elastix interpolates
        its feature maps at its points: FireANTs registers them at the image's resolution."""
        fixed_sides, moving_sides, self._channels = [], [], []
        for core in self.cores:
            images = {"fixed": fixed, "moving": moving}
            if core.model.voxel_size is not None:
                images = {
                    side: resampled(image, grid_size(extent, core.model.voxel_size, tuple(image.shape[2:])))
                    for (side, image), extent in zip(images.items(), extents, strict=True)
                }
            # Both images through the same tile, the larger first: two volumes assembled alike (a model that sees the
            # whole image, as anatomix does, gives other features in tiles).
            layers, tile = {}, patch
            for side, image in sorted(images.items(), key=lambda item: -item[1].numel()):
                layers[side], tile = core.model.volume(image, core.normalization, tile, overlap)
            if tile != patch:
                print(
                    f"[FireANTs] feature extraction did not fit whole: extracted in tiles of {tile} voxels.", flush=True
                )
            fixed_layers, moving_layers = layers["fixed"], layers["moving"]
            if core.pca > 0:  # onto the fixed image's basis, as the other engines fit it on the reference side
                for index, (fixed_layer, moving_layer) in enumerate(zip(fixed_layers, moving_layers, strict=True)):
                    moving_layers[index], fixed_layers[index] = core._pca_project(moving_layer, fixed_layer)
            # Normalised and reduced on its own grid, each layer is read at the image's voxels as elastix reads it.
            fixed_layers = [onto_image_grid(layer, tuple(images["fixed"].shape[2:])) for layer in fixed_layers]
            moving_layers = [onto_image_grid(layer, tuple(images["moving"].shape[2:])) for layer in moving_layers]
            if core.model.voxel_size is not None:
                fixed_layers = [resampled(layer, tuple(fixed.shape[2:]), padding="border") for layer in fixed_layers]
                moving_layers = [resampled(layer, tuple(moving.shape[2:]), padding="border") for layer in moving_layers]
            del images
            self._channels.append([layer.shape[1] for layer in fixed_layers])
            fixed_sides += fixed_layers
            moving_sides += moving_layers
        # A single layer is returned as it is: concatenating it would copy the whole volume.
        return tuple(sides[0] if len(sides) == 1 else torch.cat(sides, dim=1) for sides in (fixed_sides, moving_sides))

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

    def _static_distances(
        self, level: list[int], moved: torch.Tensor, fixed: torch.Tensor, mask: torch.Tensor | None, seed: int
    ) -> list[torch.Tensor]:
        """Each kept layer's distance between the warped feature volumes, read off their channels."""
        values = []
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
                values.append(distance(name, moved_layer, fixed_layer, mask, self._kernel, self._chunk))
                start += channels
        return values

    def forward(self, moved: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        mask: torch.Tensor | None = None
        if self._masked:
            # A voxel counts where the fixed mask and the warped moving mask both hold it, as in FireANTs' own
            # masked metrics.
            mask = (fixed[:, -1:] * moved[:, -1:] >= 0.5).to(moved.dtype)
            moved, fixed = moved[:, :-1], fixed[:, :-1]
        level = self._levels[min(max(self._level, 0), len(self._levels) - 1)]
        # One draw per evaluation, the same channels in every patch and in the checkpoints' second pass.
        seed = int(torch.randint(2**31 - 1, (1,), generator=self._generator))
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
            values = [
                value
                for index in level
                for value in self.cores[index].distances(
                    moved, fixed, mask, self._terms(index), seed + 1000 * index, self._kernel
                )
            ]
        else:
            if self._sampling < 1:
                # The same voxels for every layer, compared as [1, C, N]: the mask is already in the draw.
                # A mask that keeps no voxel stays on the whole maps, where the masked mean is 0.
                centres = draw_centres(tuple(moved.shape[2:]), mask, 1, self._sampling, self._generator, moved.device)
                if centres is not None:
                    _, height, width = moved.shape[2:]
                    voxels = (centres[:, 0] * height + centres[:, 1]) * width + centres[:, 2]
                    moved, fixed, mask = moved.flatten(2)[:, :, voxels], fixed.flatten(2)[:, :, voxels], None
            # Half as many channels a LNCC pass when it does not fit, kept for the registration.
            values = halve_on_oom(
                lambda: self._static_distances(level, moved, fixed, mask, seed), self.narrow, moved.is_cuda
            )
        if self._normalize:
            if self._factors is None:
                # Latched once a level, at the transform the scale starts from; a layer already at 0 keeps 1.
                starts = [float(value.detach()) for value in values]
                self._factors = [1.0 / start if start > 0 and np.isfinite(start) else 1.0 for start in starts]
            values = [value * factor for value, factor in zip(values, self._factors, strict=True)]
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
        integrator_n: int,
        smooth_warp_sigma: float,
        smooth_grad_sigma: float,
        seed: int,
        impact_levels: list[list["ModelSpec"]],
        mode: str = "Static",
        feature_patch: int = 0,
        feature_chunk: int = 0,
        feature_overlap: float = 0.25,
        deformable_masked: bool = True,
        normalize: bool = True,
        feature_map_update_interval: int = -1,
        lncc_kernel: int = 5,
        mixed_precision: bool = False,
        voxel_sampling: float = 1.0,
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
        if linear_method == "none" and deformable_method == "none":
            # It would optimise nothing and return an identity no downstream check tells from a result.
            raise ValueError("linear_method='none' with deformable_method='none' leaves nothing to optimise.")
        self._deformable_metric = deformable_metric
        self._deformable_lr = float(deformable_lr)
        self._integrator_n = int(integrator_n)
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
        self._feature_overlap = float(feature_overlap)
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

    @staticmethod
    def _center_of_mass_translation(
        fixed: sitk.Image,
        moving: sitk.Image,
        fixed_mask: "sitk.Image | None",
        moving_mask: "sitk.Image | None",
        device: str,
    ) -> torch.Tensor:
        """Seed translation ``com_moving - com_fixed`` from intensity-weighted centres of mass.

        Computed here rather than through FireANTs' ``MomentsRegistration``, which mis-estimates the centre of
        mass on anisotropic-spacing volumes.

        Each subject's centre is taken in its OWN physical space (mask-restricted when a real mask is
        given), so origin/spacing/direction differences between the two frames are handled exactly, and on
        its intensities winsorised and rescaled to [0, 1] (``_unit_range``): raw CT clipped at 0 weighs bone
        tens of times over soft tissue and a stray bright voxel over whole organs.
        Only the centroid is used: a near-symmetric subject has ambiguous principal axes.

        Returns the ``[1, 3]`` physical-space translation ``RigidRegistration(init_translation=...)``
        expects.
        """

        def com_phys(img: sitk.Image, mask: "sitk.Image | None") -> np.ndarray:
            array = sitk.GetArrayFromImage(_unit_range(img)).astype(np.float64)  # (z, y, x), in [0, 1]
            if mask is not None:
                array = array * (sitk.GetArrayFromImage(mask).astype(np.float64) > 0.5)
            positive = array > 0
            if not positive.any():
                # A constant subject (all 0 once rescaled) has no centre of mass to speak of; the frame
                # centre is the only defensible answer and matches what "cof" would have done.
                size = img.GetSize()
                return np.asarray(img.TransformContinuousIndexToPhysicalPoint([(extent - 1) / 2.0 for extent in size]))
            index = np.array(np.nonzero(positive))  # (3, N) in z, y, x
            weight = array[positive]
            centre_voxel = (index * weight).sum(axis=1) / weight.sum()  # z, y, x
            return np.asarray(
                img.TransformContinuousIndexToPhysicalPoint(
                    [float(centre_voxel[2]), float(centre_voxel[1]), float(centre_voxel[0])]
                )
            )

        translation = com_phys(moving, moving_mask) - com_phys(fixed, fixed_mask)
        return torch.tensor(translation, device=device, dtype=torch.float32).reshape(1, 3)

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

    @staticmethod
    def _affine_to_sitk(affine_matrix: "torch.Tensor") -> sitk.AffineTransform:
        """FireANTs' physical (LPS) linear matrix -> SimpleITK AffineTransform (fixed -> moving points),
        the same convention FireANTs writes into an ANTs ``0GenericAffine.mat``."""
        matrix = affine_matrix.float().cpu().numpy()[0]
        affine = sitk.AffineTransform(DIM)
        affine.SetMatrix(matrix[:DIM, :DIM].flatten().astype(np.float64))
        affine.SetTranslation(matrix[:DIM, DIM].astype(np.float64))
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
                self._feature_overlap,
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
                integrator_n=self._integrator_n,
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
        reg.optimize()
        with tempfile.TemporaryDirectory() as tmp:
            # Uncompressed: the file is read back at once.
            warp_path = os.path.join(tmp, "total_warp.nii")
            reg.save_as_ants_transforms(warp_path)
            return sitk.ReadImage(warp_path, sitk.sitkVectorFloat32)

    def register(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        device_index: int,
        fixed_mask: sitk.Image | None = None,
        moving_mask: sitk.Image | None = None,
    ) -> np.ndarray:
        """Register ``moving`` onto ``fixed``; return the displacement field, channel-first float32, on the fixed
        grid."""
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
            return np.zeros((DIM, *fixed.GetSize()[::-1]), dtype=np.float32)
        # Only a mask that restricts (some voxels in, some out) is one, decided on the mask as given: konfai-apps
        # writes an all-ones default on the FIXED grid for both sides, which would sit on another grid than the moving
        # image.
        fixed_mask = fixed_mask if is_partial_mask(fixed_mask) else None
        moving_mask = moving_mask if is_partial_mask(moving_mask) else None
        grid = fixed  # the field is sampled on the fixed image as it came
        if self._deformable_metric == "impact":
            # Feature networks compute their channels along the voxel axes they are given: both images go in with
            # their voxel axes in one order, so the fixed and moving features compare the same descriptors.
            fixed, moving, fixed_mask, moving_mask = world_aligned_pair(fixed, moving, fixed_mask, moving_mask)
        _require_fireants()
        from fireants.utils.globals import MIN_IMG_SIZE

        # FireANTs sizes its warp no smaller than MIN_IMG_SIZE a side: greedy then scores that warp against a
        # fixed image with a thinner axis, and SyN cannot invert a warp larger than the image on every axis.
        # Both fail only once the optimisation is over, so the sizes are checked before it starts.
        thin = [extent < MIN_IMG_SIZE for extent in fixed.GetSize()]
        if self._deformable_method == "greedy" and any(thin):
            raise ValueError(
                f"FireANTs' greedy registration needs every axis of the fixed image to span at least {MIN_IMG_SIZE} "
                f"voxels, got {fixed.GetSize()}: use deformable_method 'syn', which takes a thin axis."
            )
        if self._deformable_method == "syn" and all(thin):
            raise ValueError(
                f"FireANTs' SyN needs one axis of the fixed image to span at least {MIN_IMG_SIZE} voxels, got "
                f"{fixed.GetSize()}: register the whole image, or tiles of at least {MIN_IMG_SIZE} a side."
            )
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
                init_translation = torch.zeros((1, 3), device=device, dtype=torch.float32)
            elif self._moments_init == "com":
                init_translation = self._center_of_mass_translation(
                    fixed,
                    moving,
                    fixed_mask if use_fixed_mask else None,
                    moving_mask if use_moving_mask else None,
                    device,
                )
            rigid = RigidRegistration(
                scales=self._scales,
                iterations=self._affine_iterations,
                fixed_images=bf,
                moving_images=bm,
                loss_type=affine_loss,
                optimizer="Adam",
                optimizer_lr=self._affine_lr,
                cc_kernel_size=self._cc_kernel,
                init_translation=init_translation,
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
                    optimizer_lr=self._affine_lr,
                    cc_kernel_size=self._cc_kernel,
                    init_rigid=rigid_matrix,
                )
                affine.optimize()
                affine_matrix = affine.get_affine_matrix().detach()
                del affine
            # What the linear stages found, where a stage capped by its step size shows as a short shift and a
            # determinant away from 1 (rotation and shear standing in for the translation it could not reach).
            matrix = affine_matrix.double().cpu().numpy()[0]
            centre = np.asarray(fixed.TransformContinuousIndexToPhysicalPoint([(n - 1) / 2 for n in fixed.GetSize()]))
            shift = matrix[:DIM, :DIM] @ centre + matrix[:DIM, DIM] - centre
            print(
                f"[FireANTs] linear stage: {np.round(shift, 2).tolist()} mm at the fixed image centre, "
                f"determinant {np.linalg.det(matrix[:DIM, :DIM]):.3f}.",
                flush=True,
            )

        # Deformable stage (or none). SyN and Greedy share the same constructor surface; both warm-start
        # from the affine so their TOTAL transform already bakes in the linear pre-align.
        if self._deformable_method == "none":
            field = self._field_on(self._affine_to_sitk(affine_matrix), grid)
        else:
            if self._deformable_method == "syn":
                from fireants.registration.syn import SyNRegistration as Deformable
            elif self._deformable_method == "greedy":
                from fireants.registration.greedy import GreedyRegistration as Deformable
            else:
                raise ValueError(
                    f"Unknown deformable_method '{self._deformable_method}' (expected 'syn', 'greedy' or 'none')."
                )
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
                        self._feature_overlap,
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
                        integrator_n=self._integrator_n,
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
            "Adam's step per iteration in the rigid and affine stages: millimetres for the translation (the image's "
            "physical unit), unitless for the rotation and the matrix. Each stage moves the translation at most about "
            "affine_lr x sum(affine_iterations): 10 mm at 0.03 over 350 iterations, 1 mm at 0.003.",
        ] = 0.03,
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
        deformable_metric: Annotated[
            Literal["cc", "mi", "mse", "impact"],
            "Similarity metric for the deformable stage; 'impact' uses the IMPACT feature models under 'models'.",
        ] = "cc",
        deformable_lr: Annotated[float, Range(0.0, 10.0), "Gradient step size of the deformable optimisation."] = 0.25,
        integrator_n: Annotated[
            int,
            Range(1, 100),
            "No effect: FireANTs integrates a velocity field only for its geodesic deformation, and this engine "
            "runs the compositive one. Accepted so the presets that set it still load.",
        ] = 10,
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
            "at once. Tiles share the 'feature_overlap' share of their width and are blended by a cosine "
            "window, so a volume larger than the card still goes through; a pass that runs out of memory is "
            "retried in tiles half as wide (256 voxels after a whole image).",
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
        feature_overlap: Annotated[
            float,
            "Static only: the share of its width two neighbouring extraction tiles have in common. More "
            "overlap costs time and blends the seams further; it is ignored when the whole image goes "
            "through in one pass.",
            Range(0.0, 0.9),
        ] = 0.25,
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
            integrator_n,
            smooth_warp_sigma,
            smooth_grad_sigma,
            seed,
            level_models(models, levels, len(scales), "FireANTs") if impact else [],
            mode,
            feature_patch,
            feature_chunk,
            feature_overlap,
            deformable_masked,
            normalize,
            feature_map_update_interval,
            lncc_kernel,
            mixed_precision,
            voxel_sampling,
        )
        self.add_module("Registration", EngineRegistration(engine), in_branch=[0, 1, 2, 3], out_branch=["registration"])
        # The output module the presets name.
        self.add_module("DisplacementField", torch.nn.Identity(), in_branch=["registration"], out_branch=["dvf"])

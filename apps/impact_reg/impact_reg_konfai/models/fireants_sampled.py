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

"""FireANTs' rigid and affine registrations over boxes of the images, read as they are needed.

FireANTs' linear stages warp the whole moving image onto the whole fixed grid at every iteration (``rigid.py``,
``affine.py``: ``affine_grid`` + ``grid_sample`` over the fixed shape, then the loss). Here the same parametrisation,
optimizer, pyramid and losses run on boxes of the fixed grid: a box's normalized coordinates are a linear map of the
whole image's (``-1 + 2 i / (N - 1)``, align_corners), so the moving box a fixed box reads is the affine image of its
corners widened by a voxel, and the interpolation inside it is the whole image's to the bit. A point-wise loss (MSE,
and mutual information, which bins the points it is given) takes every box's points together, an estimate of the
whole-image loss over the boxes drawn; local cross-correlation takes each box with its window's halo. A scale's
draw grows to every box of the level, where the iteration is FireANTs' own; the boxes go by packets, each scored
and forgotten. The pyramid
levels are FireANTs' (:func:`~impact_reg_konfai.ooc.build_level`), built by blocks to a memmap.

Nothing of FireANTs is changed: the linear classes below override ``optimize`` alone.

FireANTs' deformable stages (greedy and SyN, compositive, Adam) hold the warp, its Adam moments and both images on
the device and step every voxel at every iteration. :func:`sampled_deformable_class` runs the same iteration on
boxes of the warp, which lives off the device with its moments when the device cannot hold them: a box is read with
a halo wide enough for what the iteration reads around a voxel (the loss window, the gradient and warp smoothing),
the images under it are read where the box's warp sends it, and the step is FireANTs' own (smoothed gradient, Adam
direction, normalisation by the largest step of the boxes drawn, composition, smoothing). Every box is visited once
an epoch, a few at a time: with every box in one iteration the stage is FireANTs' on the whole image.
"""

from __future__ import annotations

import contextlib
import functools
import math
import random
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from konfai.utils.vram import max_voxels, peak_bytes

from ..ooc import (
    ArraySource,
    HeaderImages,
    RegionSource,
    ResampledSource,
    StackedSource,
    TensorSource,
    blur_level,
    box_region,
    build_level,
    fireants_gaussians,
    fireants_level_sizes,
    free_bytes,
    held,
    on_device,
    resized,
    tiles,
    to_array,
    to_device,
    voxel_frame,
)


def _box_to_global(start: Sequence[int], stop: Sequence[int], shape: Sequence[int], device, dtype) -> torch.Tensor:
    """The homogeneous map from a box's normalized coordinates to the whole image's, (x, y, z) order: along an axis
    of ``N`` voxels, voxel ``i`` sits at ``-1 + 2 i / (N - 1)``, and the box ``[i0, i1)`` spans ``b = i1 - i0``."""
    dims = len(shape)
    matrix = torch.eye(dims + 1, device=device, dtype=dtype)
    for axis, (i0, i1, extent) in enumerate(zip(reversed(start), reversed(stop), reversed(shape), strict=True)):
        b = i1 - i0
        if extent > 1:
            matrix[axis, axis] = (b - 1) / (extent - 1)
            matrix[axis, -1] = -1.0 + (2 * i0 + b - 1) / (extent - 1)
        else:
            matrix[axis, axis] = 0.0
            matrix[axis, -1] = 0.0
    return matrix


def _box_coordinates(low: Sequence[int], high: Sequence[int], shape: Sequence[int], device, dtype) -> torch.Tensor:
    """The normalized coordinates on a ``shape`` grid of the voxels of its box ``[low, high)``, ``[1, *size, 3]``."""
    theta = _box_to_global(low, high, shape, device, dtype)[:-1].unsqueeze(0)
    return F.affine_grid(theta, (1, 1, *(b - a for a, b in zip(low, high, strict=True))), align_corners=True)


def _voxels(regions) -> int:
    return sum(math.prod(b - a for a, b in zip(*region, strict=True)) for region in regions)


#: The voxels a packet of boxes holds at most, unless told: what is scored at once, then forgotten; and the voxels the
#: linear stages draw an iteration, unless told.
_TOGETHER = 8 * 64**3
#: elastix's gain sequence a / (A + k + 1)^alpha, with the constants its documentation recommends (Spall).
_GAIN_A, _GAIN_ALPHA = 50.0, 0.602


def _moved_and_fixed(tiles, halo: int, fixed_level: RegionSource, moving_level: RegionSource, matrix, together):
    """The moving image through ``matrix`` (normalized coordinates of the fixed level to the moving one's) and the
    fixed image over ``tiles`` read with ``halo`` voxels around them, by packets of ``[K, C, *size]`` each, and the
    voxels of each packet that are a tile's own (``[K, 1, *size]`` of ones and zeros; None without a halo, where
    all are). Levels in files are read a tile at a time. Levels on the device give the tiles of one size together,
    ``together`` voxels at most, in one sampling of the moving level: a draw of many small boxes then costs a few
    calls."""
    shape = fixed_level.shape

    def own(size, voxels) -> torch.Tensor | None:
        if halo == 0:
            return None
        cores = torch.zeros((1, *size), device=matrix.device, dtype=matrix.dtype)
        cores[(slice(None), *voxels)] = 1
        return cores

    regions = [box_region(tile, halo, shape) for tile in tiles]
    if not (isinstance(fixed_level, TensorSource) and isinstance(moving_level, TensorSource)):
        for low, high, voxels in regions:
            at = _box_coordinates(low, high, shape, matrix.device, matrix.dtype)
            moved = _sample_region(moving_level, at @ matrix[:-1, :-1].T + matrix[:-1, -1])
            cores = own([b - a for a, b in zip(low, high, strict=True)], voxels)
            yield moved, fixed_level.read(low, high), None if cores is None else cores[None]
        return
    by_size: dict[tuple[int, ...], list] = {}
    for low, high, voxels in regions:
        by_size.setdefault(tuple(b - a for a, b in zip(low, high, strict=True)), []).append((voxels, low))
    # A voxel further along an axis is 2 / (N - 1) further in normalized coordinates, (x, y, z) order.
    step = torch.tensor([2.0 / (n - 1) if n > 1 else 0.0 for n in reversed(shape)], device=matrix.device)
    moving = moving_level.tensor[None].float()
    for size, boxes in by_size.items():
        origin = _box_coordinates([0] * len(size), size, shape, matrix.device, matrix.dtype)
        packet = max(1, together // math.prod(size))
        for first in range(0, len(boxes), packet):
            chunk = boxes[first : first + packet]
            lows = [low for _, low in chunk]
            shifts = torch.tensor([low[::-1] for low in lows], device=matrix.device, dtype=matrix.dtype) * step
            at = origin + shifts.view(len(chunk), *([1] * len(size)), len(size))
            at = at @ matrix[:-1, :-1].T + matrix[:-1, -1]
            moved = F.grid_sample(
                moving,
                at.reshape(1, len(chunk) * size[0], *size[1:], len(size)),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            fixed = torch.stack(
                [
                    fixed_level.tensor[
                        (slice(None), *(slice(a, a + extent) for a, extent in zip(low, size, strict=True)))
                    ]
                    for low in lows
                ]
            )
            cores = None if halo == 0 else torch.stack([own(size, voxels) for voxels, _ in chunk])
            yield moved.reshape(moved.shape[1], len(chunk), *size).transpose(0, 1), fixed.float(), cores


def _joint_counts(
    loss, moved: torch.Tensor, fixed: torch.Tensor, region: torch.Tensor | None = None
) -> tuple[torch.Tensor, float]:
    """The joint histogram ``[bins, bins]`` of a packet of boxes ``[K, C, *size]`` as FireANTs' mutual information
    builds it, before its division by the points it counts, and how many they are; over the voxels ``region``
    (``[K, 1, *size]``) holds, when given (a box's own, its halo being another box's)."""
    moved, fixed = moved.flatten(2), fixed.flatten(2)
    mask = None
    if loss.masked:
        from fireants.losses.maskedutils import get_tensors_and_mask

        moved, fixed, mask = get_tensors_and_mask(moved, fixed, loss.mask_mode)
        mask = (mask.flatten(1) >= 0.5).to(moved.dtype)
    if region is not None:
        held = (region.flatten(1) >= 0.5).to(moved.dtype)
        mask = held if mask is None else mask * held
    largest = max(moved.max(), fixed.max())
    if largest > 1:
        moved, fixed = moved / largest, fixed / largest
    moved_weight, _, fixed_weight, _ = loss.parzen_windowing(moved, fixed)
    counted = float(moved_weight.shape[0] * moved_weight.shape[1])
    if mask is not None:
        moved_weight, counted = moved_weight * mask[:, :, None], float(mask.sum())
    return torch.bmm(moved_weight.permute(0, 2, 1), fixed_weight.to(moved_weight)).sum(0), counted


class _Information(NamedTuple):
    """A mutual information over the level, as an iteration of the deformable stage reads it: the derivative of the
    information at the level's joint histogram, the points the histogram counts, and the information."""

    weight: torch.Tensor
    points: float
    value: float


def _information(loss, joint: torch.Tensor) -> torch.Tensor:
    """FireANTs' mutual information loss of a joint histogram ``[bins, bins]`` that sums to 1."""
    independent = joint.sum(1, keepdim=True) @ joint.sum(0, keepdim=True)
    ratio = (joint + loss.smooth_nr) / (independent + loss.smooth_dr)
    return -(joint * torch.log(ratio + loss.smooth_dr)).sum()


def _tracked(loss, tracked: torch.Tensor | None, joint, points: float, share: float):
    """The level's joint histogram ``tracked`` moved by a draw's ``joint`` counts over its ``points``, the draw
    holding ``share`` of the level (every box drawn: the draw's own histogram), and the information read off it
    with its derivative there; None for the information before any draw held a point."""
    if points > 0:  # a draw the masks miss holds no point: the level's histogram stays as it was
        histogram = (joint / points).float()
        tracked = histogram if tracked is None or share >= 1 else tracked.lerp(histogram, share)
    if tracked is None:
        return None, None
    histogram = tracked.clone().requires_grad_()
    value = _information(loss, histogram)
    (weight,) = torch.autograd.grad(value, histogram)
    return tracked, _Information(weight, points, float(value.detach()))


class SampledLinearMixin:
    """``optimize`` over boxes for a FireANTs rigid or affine registration built on :class:`HeaderImages`.

    ``sample(fixed, moving, box, boxes, halo, seed)`` sets the sources and the schedule before ``optimize``: ``fixed``
    and ``moving`` are :class:`RegionSource` of the native images (their mask as a last channel in masked mode),
    ``box`` the side of a box in voxels of a level, ``boxes`` how many an iteration draws, anew at each one (see
    ``optimize``), ``halo`` the voxels read around a box for a local loss, ``together`` the voxels scored at once:
    the boxes of a draw go by packets of that many, each scored and forgotten. The levels above the finest are built
    once per scale, as FireANTs computes them."""

    def sample(
        self,
        fixed: RegionSource,
        moving: RegionSource,
        *,
        box: int = 64,
        boxes: int = 8,
        halo: int = 0,
        seed: int = 0,
        scratch: Path | None = None,
        together: int = _TOGETHER,
    ) -> None:
        self._sources = (on_device(fixed), on_device(moving))
        self._together = int(together)
        self._scratch = scratch
        self._box = int(box)
        self._boxes = int(boxes)
        self._halo = int(halo)
        self._draws = random.Random(int(seed))
        self._levels: dict[int, tuple[RegionSource, RegionSource]] = {}

    # The rigid and affine classes expose their matrix under different names.
    def _matrix(self) -> torch.Tensor:
        raise NotImplementedError

    _extra_moving_blur = False

    def _level_sources(self, scale: int) -> tuple[RegionSource, RegionSource]:
        fixed, moving = self._sources
        if scale == 1:
            return self._sources
        if scale not in self._levels:
            fixed_size = fireants_level_sizes(fixed.shape, scale)
            moving_size = fireants_level_sizes(moving.shape, scale)
            settings = {"mask_channel": self.masked, "directory": self._scratch}
            if not self.blur:  # FireANTs interpolates the fixed image onto the level, unsmoothed, and keeps the moving
                self._levels[scale] = (on_device(build_level(fixed, fixed_size, None, **settings)), moving)
                return self._levels[scale]
            gaussians = fireants_gaussians(fixed.shape, fixed_size, fixed.device)  # FireANTs smooths both with these
            moving_level = build_level(moving, moving_size, gaussians, **settings)
            if self._extra_moving_blur:
                moving_level = blur_level(moving_level, gaussians, **settings)
            fixed_level = build_level(fixed, fixed_size, gaussians, **settings)
            self._levels[scale] = (on_device(fixed_level), on_device(moving_level))
        return self._levels[scale]

    def _sums(self, moved, fixed, cores, masked: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """What a packet adds to the loss's mean: the sum of the loss over its own voxels (``cores``), and how many
        they are, as the loss counts them (its channels unmasked, the voxels both masks keep when masked)."""
        mask = cores
        if masked:
            from fireants.losses.maskedutils import get_tensors_and_mask

            moved, fixed, mask = get_tensors_and_mask(moved, fixed, self.loss_fn.mask_mode)
            mask = mask if cores is None else mask * cores
        values = self.loss_fn.forward_util(moved, fixed)
        if mask is None:
            return values.sum(), torch.tensor(float(values.numel()))
        return (values * mask).sum(), mask.sum() * (1 if masked else values.shape[1])

    def _information_step(self, packets, share: float) -> float:
        """Mutual information of a draw holding ``share`` of the level: its value, and its gradient left on what the
        packets were sampled through. The information is no sum over the points but a function of their joint
        histogram, which is one: a draw's own information is a biased estimate of the level's, and so is its
        gradient, whatever the iterations. The level's histogram is tracked from draw to draw instead, each moving
        it by its share; the information is read off it, and the draw gives its gradient through its own counts,
        weighed by the information's derivative at the tracked histogram (a stochastic compositional gradient).
        With every box drawn this is FireANTs' loss. A draw that one packet holds keeps its counts from the first
        pass; a larger one counts each packet again, so that no more than a packet is held."""
        joint, points, kept = None, 0.0, []
        with torch.set_grad_enabled(self._keeps_draw):
            for moved, fixed, _ in packets():
                counts, counted = _joint_counts(self.loss_fn, moved, fixed)
                joint = counts.detach().double() if joint is None else joint + counts.detach().double()
                points += counted
                kept += [counts] if self._keeps_draw else []
        self._histogram, information = _tracked(self.loss_fn, self._histogram, joint, points, share)
        if information is None:
            return 0.0
        if points == 0:
            return information.value
        for counts in kept or (_joint_counts(self.loss_fn, moved, fixed)[0] for moved, fixed, _ in packets()):
            piece = (counts * information.weight).sum() / points
            if piece.requires_grad:  # not when the boxes fall outside the moving image
                piece.backward()
        return information.value

    def optimize(self) -> None:
        from fireants.losses.mi import GlobalMutualInformationLoss

        fixed_t2p = self.fixed_images.get_torch2phy().to(self.dtype)
        moving_p2t = self.moving_images.get_phy2torch().to(self.dtype)
        for scale, iters in zip(self.scales, self.iterations, strict=True):
            self.convergence_monitor.reset()
            if hasattr(self.loss_fn, "set_current_scale_and_iterations"):
                self.loss_fn.set_current_scale_and_iterations(scale, iters)
            fixed_level, moving_level = self._level_sources(scale)
            fixed_shape = fixed_level.shape
            # A box and its halo are in voxels of the level: the coarse levels fit in a few boxes, read whole.
            box, halo = self._box, self._halo
            information = type(self.loss_fn) is GlobalMutualInformationLoss
            masked = bool(getattr(self.loss_fn, "masked", False))
            if not information:
                self.loss_fn.reduction = "none"  # the loss at each voxel: its mean is taken here, over the packets
            boxes = tiles(fixed_level.shape, box)
            count = min(self._boxes, len(boxes))
            whole = count == len(boxes)  # the samples hold the level: FireANTs' iteration, its step and its stop
            self._histogram = None  # the level's joint histogram, as the draws of a mutual information tell it
            rates = [group["lr"] for group in self.optimizer.param_groups]
            parameters = [parameter for group in self.optimizer.param_groups for parameter in group["params"]]
            means = [torch.zeros_like(parameter) for parameter in parameters]
            for iteration in range(iters):
                # Every iteration draws its boxes anew, as elastix draws its points. A draw's gradient is the
                # level's plus a noise of zero mean, which a constant step never lets the transform settle under:
                # the step falls as elastix's gain does, a / (A + k + 1)^alpha, whose sum diverges while the sum of
                # its squares does not (Robbins and Monro), and the scale ends on the mean of the transforms of its
                # second half (Polyak and Ruppert), which the noise of the draws leaves at the level's optimum as
                # the iterations grow.
                gain = 1.0 if whole else ((_GAIN_A + 1) / (_GAIN_A + iteration + 1)) ** _GAIN_ALPHA
                for group, rate in zip(self.optimizer.param_groups, rates, strict=True):
                    group["lr"] = rate * gain
                drawn = boxes if whole else [boxes[index] for index in self._draws.sample(range(len(boxes)), count)]
                # The voxels a draw holds on average: divided by them and not by its own, the sum over a draw is an
                # unbiased estimate of the level's mean, whatever the sizes of the boxes drawn.
                expected = math.prod(fixed_shape) * count / len(boxes)
                self._keeps_draw = _voxels(drawn) <= self._together
                self.optimizer.zero_grad()
                matrix = (moving_p2t @ self._matrix() @ fixed_t2p).to(self.dtype)  # [1, D+1, D+1], torch to torch
                # The gradients of the packets gather on a copy of the matrix and go through it once: a packet is
                # scored and forgotten, and an iteration holds one of them whatever it draws.
                gathered = matrix.detach().requires_grad_()
                packets = partial(_moved_and_fixed, drawn, halo, fixed_level, moving_level, gathered[0], self._together)
                if information:
                    cur_loss = self._information_step(packets, count / len(boxes))
                    gradient = gathered.grad
                else:
                    # The loss is a mean over the level's voxels (those both masks keep, when masked): a sum over
                    # them, divided by how many they are. Each packet adds its own voxels to both and its gradient
                    # to theirs; the quotient is taken once.
                    sums, gradients = [0.0, 0.0], [torch.zeros_like(gathered), torch.zeros_like(gathered)]
                    for moved, fixed_boxes, cores in packets():
                        for part, term in enumerate(self._sums(moved, fixed_boxes, cores, masked)):
                            sums[part] += float(term)
                            if term.requires_grad:  # not when the boxes fall outside the moving image
                                gradients[part] += torch.autograd.grad(term, gathered, retain_graph=part == 0)[0]
                    if not masked:  # an unbiased estimate of the level's mean: see ``expected``
                        sums[1], gradients[1] = expected * sums[1] / max(_voxels(drawn), 1), gradients[1] * 0
                    held = sums[1] + 1e-8
                    cur_loss = sums[0] / held
                    gradient = gradients[0] / held - gradients[1] * (sums[0] / held**2)
                if gradient is not None:
                    matrix.backward(gradient)
                self.optimizer.step()
                # FireANTs leaves a scale once its loss stops falling, which the loss of a draw is too noisy to
                # tell: on draws a scale runs its iterations, as elastix does.
                if whole and self.convergence_monitor.converged(cur_loss):
                    break
                if not whole and iteration >= iters // 2:
                    for parameter, mean in zip(parameters, means, strict=True):
                        mean += (parameter.detach() - mean) / (iteration - iters // 2 + 1)
            if not whole and iters > 0:
                with torch.no_grad():
                    for parameter, mean in zip(parameters, means, strict=True):
                        parameter.copy_(mean)
            for group, rate in zip(self.optimizer.param_groups, rates, strict=True):
                group["lr"] = rate


def sampled_rigid_class():
    """``RigidRegistration`` reading boxes; imported lazily, FireANTs being optional at import time."""
    from fireants.registration.rigid import RigidRegistration

    class SampledRigidRegistration(SampledLinearMixin, RigidRegistration):
        def _matrix(self) -> torch.Tensor:
            return self.get_rigid_matrix()

        _extra_moving_blur = True  # rigid.py smooths its moving image a second time at every level above the finest

    return SampledRigidRegistration


def sampled_affine_class():
    """``AffineRegistration`` reading boxes."""
    from fireants.registration.affine import AffineRegistration

    class SampledAffineRegistration(SampledLinearMixin, AffineRegistration):
        def _matrix(self) -> torch.Tensor:
            return self.get_affine_matrix()

    return SampledAffineRegistration


def sampled_linear_alignment(
    fixed_image,
    moving_image,
    fixed: RegionSource,
    moving: RegionSource,
    *,
    scales: Sequence[int],
    iterations: Sequence[int],
    loss_type: str,
    optimizer_lr: float,
    translation_lr: float,
    cc_kernel_size: int,
    init_translation,
    device: str,
    box: int,
    boxes: int,
    seed: int,
    affine: bool = True,
    scratch: Path | None = None,
    together: int = _TOGETHER,
) -> torch.Tensor:
    """FireANTs' linear chain, rigid then affine, over boxes of ``fixed`` and ``moving``: the ``[1, D+1, D+1]`` matrix
    (physical, fixed to moving) the engine seeds its deformable stage with. ``fixed_image`` and ``moving_image`` give
    the geometry (SimpleITK images, their pixels never read). The translation steps by ``translation_lr``, the
    rotation or the matrix by ``optimizer_lr``, as the engine's whole-image stages step them."""
    from .fireants import _StepPerElement

    channels = fixed.channels
    header_fixed, header_moving = (HeaderImages(image, channels, device) for image in (fixed_image, moving_image))
    halo = (cc_kernel_size - 1) // 2 if loss_type.endswith("cc") else 0
    common = {
        "scales": list(scales),
        "iterations": list(iterations),
        "fixed_images": header_fixed,
        "moving_images": header_moving,
        "loss_type": loss_type,
        "optimizer": "Adam",
        "optimizer_lr": 1.0,  # each element's step is _StepPerElement's
        "cc_kernel_size": cc_kernel_size,
        "progress_bar": False,
    }
    drawn = {"box": box, "boxes": boxes, "halo": halo, "scratch": scratch, "together": together}
    rigid = sampled_rigid_class()(**common, init_translation=init_translation)
    rigid.optimizer = _StepPerElement(rigid.optimizer, [(rigid.rotation, optimizer_lr), (rigid.transl, translation_lr)])
    rigid.sample(fixed, moving, seed=seed, **drawn)
    rigid.optimize()
    matrix = rigid.get_rigid_matrix().detach()
    if not affine:
        return matrix
    del rigid
    affine_registration = sampled_affine_class()(**common, init_rigid=matrix)
    sizes = torch.full_like(affine_registration.affine, optimizer_lr)
    sizes[..., -1] = translation_lr  # the matrix's last column is its translation
    affine_registration.optimizer = _StepPerElement(
        affine_registration.optimizer, [(affine_registration.affine, sizes)]
    )
    affine_registration.sample(fixed, moving, seed=seed + 1, **drawn)
    affine_registration.optimize()
    return affine_registration.get_affine_matrix().detach()


class _Warp:
    """A warp of the deformable stage on a level's grid with its Adam moments, channel-first ``[d, *grid]`` (a component
    an axis), read and written by boxes (channel-last, as FireANTs holds a warp): tensors on the device when
    ``resident``, float32 arrays in host memory when ``host``, float32 memmaps in files of ``directory`` otherwise.
    ``next`` is the warp an iteration steps to, box by box, while every box still reads the iteration's own: ``swap``
    makes it the warp, ``keep`` the boxes of it a draw stepped."""

    KINDS = ("warp", "mean", "square")

    def __init__(
        self, directory: Path, name: str, grid: Sequence[int], device: str, resident: bool, host: bool
    ) -> None:
        self.directory, self.name, self.device, self.resident, self.generation = directory, name, device, resident, 0
        self.host = host
        self.arrays: dict = {}
        self.paths: dict[str, Path] = {}
        for kind in (*self.KINDS, "next"):
            self._new(kind, grid)

    def _new(self, kind: str, grid: Sequence[int]) -> None:
        if self.resident:
            self.arrays[kind] = torch.zeros((len(grid), *grid), device=self.device)
            return
        if self.host:
            self.arrays[kind] = np.zeros((len(grid), *grid), dtype=np.float32)
            return
        self.paths[kind] = self.directory / f"{self.name}_{kind}_{self.generation}.npy"
        self.arrays[kind] = np.lib.format.open_memmap(
            str(self.paths[kind]), mode="w+", dtype=np.float32, shape=(len(grid), *grid)
        )

    @property
    def grid(self) -> tuple[int, ...]:
        return tuple(self.arrays["warp"].shape[1:])

    def resize(self, grid: Sequence[int], resident: bool, host: bool) -> None:
        """The warp and its moments on another grid, by linear interpolation (``align_corners``): what FireANTs
        does to them between two scales, block by block when they live in files, before or after."""
        if tuple(grid) == self.grid and (resident, host) == (self.resident, self.host):
            return
        paths: dict[str, Path] = {}  # the new generation's files: the old ones stay owned until it is whole
        self.generation += 1
        try:
            for kind in self.KINDS:
                array = self.arrays[kind]
                if self.resident and resident:
                    self.arrays[kind] = resized(array[None], grid)[0]
                    continue
                source = TensorSource(array) if self.resident else ArraySource(array, self.device)
                if resident or host:
                    level = build_level(source, grid, None, memory=True).array
                    self.arrays[kind] = torch.from_numpy(level).to(self.device) if resident else level
                else:
                    paths[kind] = self.directory / f"{self.name}_{kind}_{self.generation}.npy"
                    self.arrays[kind] = build_level(source, grid, None, path=paths[kind]).array
        except BaseException:
            for path in paths.values():
                path.unlink(missing_ok=True)
            raise
        previous, self.paths = self.paths, paths
        self.resident, self.host = resident, host
        try:
            self._new("next", grid)
        finally:  # the old generation is no longer in ``paths``: close() would not find it
            for path in previous.values():
                path.unlink(missing_ok=True)

    def swap(self) -> None:
        """The stepped warp becomes the warp."""
        self.arrays["warp"], self.arrays["next"] = self.arrays["next"], self.arrays["warp"]
        if self.paths:
            self.paths["warp"], self.paths["next"] = self.paths["next"], self.paths["warp"]

    def keep(self, start: Sequence[int], stop: Sequence[int]) -> None:
        """The box ``[start, stop)`` of the stepped warp becomes the warp's."""
        self.write("warp", start, self.read("next", start, stop))

    def read(self, kind: str, start: Sequence[int], stop: Sequence[int]) -> torch.Tensor:
        region = self.arrays[kind][(slice(None), *(slice(a, b) for a, b in zip(start, stop, strict=True)))]
        # A copy: the caller's in-place steps must not write the store.
        tensor = region.clone() if self.resident else to_device(region, self.device)
        return tensor.movedim(0, -1).unsqueeze(0)

    def write(self, kind: str, start: Sequence[int], values: torch.Tensor) -> None:
        stop = [a + extent for a, extent in zip(start, values.shape[1:-1], strict=True)]
        box = (slice(None), *(slice(a, b) for a, b in zip(start, stop, strict=True)))
        values = values[0].movedim(-1, 0).detach()
        if self.resident:
            self.arrays[kind][box] = values
        else:
            to_array(self.arrays[kind][box], values)

    def source(self) -> RegionSource:
        """The warp, read by boxes."""
        return TensorSource(self.arrays["warp"]) if self.resident else ArraySource(self.arrays["warp"], self.device)

    def peak(self, channel: int) -> float:
        """The largest absolute value of one channel of the warp."""
        values = self.arrays["warp"][channel]
        if self.resident:
            return float(values.abs().max())
        return max(float(np.abs(values[first : first + 16]).max()) for first in range(0, values.shape[0], 16))

    def close(self) -> None:
        self.arrays.clear()
        for path in self.paths.values():
            path.unlink(missing_ok=True)
        self.paths.clear()


def _nodes_from(index: int, extent: int, nodes: int) -> int:
    """The first node of a ``nodes``-node grid at or past node ``index`` of an ``extent``-node grid over the same
    span (``align_corners``), ``nodes`` past the end."""
    if index >= extent:
        return nodes
    return min(nodes, -((-index * (nodes - 1)) // max(extent - 1, 1)))


def _local_frame(low: Sequence[int], high: Sequence[int], shape: Sequence[int], device, dtype):
    """The scale and offset, (x, y, z) order, taking the normalized coordinates of a ``shape`` grid to those of its
    box ``[low, high)``: ``local = global * scale + offset``."""
    scale, offset = [], []
    for a, b, extent in zip(reversed(low), reversed(high), reversed(shape), strict=True):
        span = max(b - a - 1, 1)
        scale.append((extent - 1) / span)
        offset.append((extent - 1 - 2 * a) / span - 1.0)
    return torch.tensor(scale, device=device, dtype=dtype), torch.tensor(offset, device=device, dtype=dtype)


def _interpolate(box: torch.Tensor, low, high, shape, coordinates: torch.Tensor) -> torch.Tensor:
    """The box ``[low, high)`` of a ``shape`` grid (``[1, C, *size]``) read by trilinear interpolation at
    ``coordinates``, normalized to the grid; zero outside the box."""
    scale, offset = _local_frame(low, high, shape, coordinates.device, coordinates.dtype)
    return F.grid_sample(box, coordinates * scale + offset, mode="bilinear", padding_mode="zeros", align_corners=True)


#: The most voxels of a source one read takes per coordinate it samples: SyN's moving level stays at the image's own
#: resolution, 64 times the coordinates of a box at scale 4.
_FOOTPRINT = 512


def _sample_region(source: RegionSource, coordinates: torch.Tensor) -> torch.Tensor:
    """``source`` read by trilinear interpolation at ``coordinates`` (``[1, *size, 3]``, normalized to the source's
    grid, (x, y, z) last), zero outside it: the voxels under the coordinates are read as one box, or, where it would
    hold over ``_FOOTPRINT`` of them a coordinate (a source far finer than the grid of the coordinates), the
    coordinates halved along their longest axis, each half by itself."""
    dims = len(source.shape)
    # The voxels under the coordinates, (x, y, z) last; their bounds on every axis come back in one trip.
    extents = torch.tensor([extent - 1 for extent in reversed(source.shape)], device=coordinates.device)
    voxels = (coordinates.detach().reshape(-1, dims) + 1.0) * 0.5 * extents
    lowest, highest = voxels.amin(0).tolist(), voxels.amax(0).tolist()
    low, high = [], []
    for axis, extent in enumerate(source.shape):  # tensor axis (z, y, x) = coordinate (dims - 1 - axis)
        low.append(max(0, math.floor(lowest[dims - 1 - axis]) - 1))
        high.append(min(extent, math.ceil(highest[dims - 1 - axis]) + 2))
    if any(b <= a for a, b in zip(low, high, strict=True)):
        return torch.zeros((1, source.channels, *coordinates.shape[1:-1]), device=coordinates.device)
    for axis, extent in enumerate(source.shape):
        # Two voxels at least: on an axis of one, grid_sample reads that voxel wherever the coordinate lies, where the
        # whole grid blends it with the zero beyond the image.
        if high[axis] - low[axis] < 2:
            low[axis], high[axis] = (
                (low[axis] - 1, high[axis]) if low[axis] > 0 else (low[axis], min(extent, high[axis] + 1))
            )
    size = coordinates.shape[1:-1]
    if math.prod(b - a for a, b in zip(low, high, strict=True)) > _FOOTPRINT * math.prod(size) and max(size) > 1:
        axis = max(range(len(size)), key=size.__getitem__)  # of the coordinates' grid: 1 + axis in them, 2 + in out
        halves = coordinates.split([size[axis] // 2, size[axis] - size[axis] // 2], dim=1 + axis)
        return torch.cat([_sample_region(source, half) for half in halves], dim=2 + axis)
    return _interpolate(source.read(low, high), low, high, source.shape, coordinates)


def _smooth(field: torch.Tensor, gaussians) -> torch.Tensor:
    """A channel-last field smoothed by FireANTs' separable kernels (``'same'``, zero outside)."""
    from fireants.losses.cc import separable_filtering

    if gaussians is None:
        return field
    return separable_filtering(field.movedim(-1, 1), gaussians).movedim(1, -1)


def read_through(
    source: RegionSource,
    source_geometry,
    field: np.ndarray,
    geometry,
    out: np.ndarray,
    *,
    block: int = 64,
    nearest: bool = False,
) -> None:
    """``source`` (its voxels on ``source_geometry``) read at ``x + u(x)`` for every voxel ``x`` of the grid of
    ``geometry``, ``u`` the physical displacement ``field`` (``[*grid, dims]``, (x, y, z) components) on that grid,
    zero outside the source, written block by block to ``out`` (``[C, *grid]``): linearly, as SimpleITK's resampler
    reads an image, or at the nearest voxel, as it reads a mask (``nearest``); a point within half a voxel past the
    source's last reads that voxel, as ITK's interpolators read the buffer's edge. The field is never held whole."""
    grid = field.shape[:-1]
    dims = len(grid)
    to_world, origin = voxel_frame(geometry)
    source_matrix, source_origin = voxel_frame(source_geometry)
    to_source = np.linalg.inv(source_matrix)
    extents = np.asarray(list(reversed(source.shape)), dtype=np.float64)  # (x, y, z)
    for start, stop in tiles(grid, block):
        box = tuple(slice(a, b) for a, b in zip(start, stop, strict=True))
        index = np.stack(np.meshgrid(*(np.arange(a, b) for a, b in zip(start, stop, strict=True)), indexing="ij"))
        index = index[::-1].reshape(dims, -1).astype(np.float64)  # (x, y, z) rows, one column a voxel
        world = to_world @ index + origin[:, None] + np.asarray(field[box], dtype=np.float64).reshape(-1, dims).T
        voxel = to_source @ (world - source_origin[:, None])
        inside = ((voxel >= -0.5) & (voxel < extents[:, None] - 0.5)).all(0)
        voxel = np.clip(np.floor(voxel + 0.5) if nearest else voxel, 0.0, extents[:, None] - 1.0)
        normalized = -1.0 + 2.0 * voxel / np.maximum(extents[:, None] - 1.0, 1.0)
        size = tuple(b - a for a, b in zip(start, stop, strict=True))
        coordinates = torch.from_numpy(normalized.T.reshape(1, *size, dims)).to(source.device, torch.float32)
        values = _sample_region(source, coordinates)
        held = torch.from_numpy(inside.reshape(1, 1, *size)).to(values.device, values.dtype)
        to_array(out[(slice(None), *box)], (values * held)[0])


def static_feature_sources(
    loss,
    fixed: RegionSource,
    moving: RegionSource,
    directory: Path,
    *,
    block: int = 96,
    extents: tuple[list[float] | None, list[float] | None] = (None, None),
) -> tuple[ArraySource, ArraySource]:
    """Static mode's feature volumes of ``loss`` (the engine's ``ImpactFeatureLoss``), predicted by blocks as a
    segmentation is into memmaps of ``directory``: every model's kept layers over the fixed and the moving image,
    concatenated along the channels as ``ImpactFeatureLoss.extract`` lays them out. Each block is read with half the
    model's receptive field around it (as much of it as the device holds with a block) and run through the network,
    which normalises it by the whole image's statistics (its intensities', its normalisations'); each layer keeps the cells of its own grid the
    block owns, assembled on that grid as the whole image's layer. A PCA is then fitted on the fixed image's layer and
    each image centred by its own mean, and every layer brought onto the image's voxels, block by block, as the whole
    image's is (``onto_image_grid``). The network runs in float16 where the preset asks for it (``mixed_precision``,
    on a GPU), as the whole run does, and the blocks are as large as half the free memory holds them with their
    halos, ``block`` voxels a side at least: halved, and the side done again, when one does not fit."""
    from konfai.metric.measure.impact import (
        SWEEP_BATCH,
        block_region,
        grid_size,
        layer_window,
        onto_image_grid,
        own_cells,
    )
    from konfai.utils.dataset import Attribute
    from konfai.utils.dataset.statistics import _finalize_running_statistics, _update_running_statistics
    from konfai.utils.errors import MeasureError
    from konfai.utils.vram import halve_on_oom, out_of_memory_as_torch

    def statistics_of(source: RegionSource) -> Attribute:
        """The image's ``ImageMin``/``ImageMax``/``ImageMean``/``ImageStd`` (unbiased), by one pass over its blocks."""
        state = None
        for start, stop in tiles(source.shape, block):
            state = _update_running_statistics(state, source.read(start, stop)[0, :1].cpu().numpy())
        found = _finalize_running_statistics(state)
        return Attribute({f"Image{key.capitalize()}": found[key] for key in ("min", "max", "mean", "std")})

    def read_block(source: RegionSource, tile, blocks: list[int], halo: int, stride: int, slabs: bool):
        """The block ``tile`` read with ``halo`` around it (``block_region``), and the slices of its own voxels in it.
        The last block of an axis, short of ``blocks[0]``, is read back to a whole one, as a prediction's last patch
        is, its start kept on ``stride``; a slab of slices (``slabs``) is read as it is."""
        start, stop = tile
        halos = [
            0 if slabs else halo + -(-(blocks[0] - (b - a)) // stride) * stride
            for a, b in zip(start, stop, strict=True)
        ]
        low, high, own = block_region(start, stop, halos, source.shape, stride)
        return source.read(low, high)[:, :1], own

    def boxes_of(core, shape: Sequence[int], blocks: list[int]) -> list:
        """The blocks a source of ``shape`` is predicted by: cubes of ``blocks[0]`` voxels a side, or, for a 2-D network
        swept over a volume's slices, which reads each slice alone, slabs of whole slices as many as a block holds,
        a whole number of the sweep's batches (whose slices a network may normalise together)."""
        if core.dimension >= len(shape):
            return tiles(shape, blocks[0])
        plane = math.prod(shape[1:])
        thick = max(SWEEP_BATCH, blocks[0] ** len(shape) // plane // SWEEP_BATCH * SWEEP_BATCH)
        return [((z, *[0] * len(shape[1:])), (min(z + thick, shape[0]), *shape[1:])) for z in range(0, shape[0], thick)]

    def owned_layers(
        core, box: torch.Tensor, own, statistics: Attribute, norms
    ) -> list[tuple[list[int], torch.Tensor]]:
        """The layers of ``box``, each on its own grid, its stride along every axis and its cells covering ``own``
        (slices of the block's voxels)."""
        layers = core.model.volume(box, core.normalization, 0, 0.0, statistics, norms)
        return [
            (
                [2 ** round(math.log2(b / n)) for b, n in zip(box.shape[2:], layer.shape[2:], strict=True)],
                layer[own_cells(own, box.shape[2:], layer.shape[2:])].float(),
            )
            for layer in layers
        ]

    def block_cost(core, statistics: Attribute) -> float:
        """The device bytes a voxel of a block takes through the network, measured on a probe of 64 voxels a side."""
        side = max(64, core.model.multiple)
        probe = torch.zeros((1, 1, *[side] * len(fixed.shape)), device=fixed.device)
        own = (slice(0, side),) * len(fixed.shape)
        norms = core.model.norms(lambda: [(probe, own)], statistics)
        _, taken = peak_bytes(lambda: owned_layers(core, probe, own, statistics, norms), fixed.device)
        return taken / probe.numel()

    def basis_of(array: np.ndarray, side: int, components: int) -> tuple[torch.Tensor, torch.Tensor]:
        """A layer's channel mean over its cells and the basis of its ``components`` principal ones (``[C, K]``, the
        largest last), by boxes of ``side`` cells a side, in double."""
        count, total, outer = 0, 0.0, 0.0
        for start, stop in tiles(array.shape[1:], side):
            box = array[(slice(None), *(slice(a, b) for a, b in zip(start, stop, strict=True)))]
            flat = to_device(box, str(fixed.device)).flatten(1).double()
            count, total, outer = count + flat.shape[1], total + flat.sum(1), outer + flat @ flat.t()
        mean = (total / count).cpu()
        covariance = (outer.cpu() - count * torch.outer(mean, mean)) / max(count - 1, 1)
        return mean.float(), torch.linalg.eigh(covariance)[1][:, -min(components, len(mean)) :].float()

    def extract(
        core,
        index: int,
        side: str,
        source: RegionSource,
        stats: Attribute,
        blocks: list[int],
        halo: int,
        stride: int,
        bases,
    ):
        """``source``'s features of ``core`` by blocks of ``blocks[0]`` voxels a side, written to memmaps of
        ``directory``, and the PCA bases that project both sides, fitted on the fixed one's (``bases`` otherwise)."""
        boxes = boxes_of(core, source.shape, blocks)
        slabs = core.dimension < len(source.shape)  # a 2-D network swept over a volume's slices
        # The network's normalisations by the whole image's statistics, one pass over the blocks each.
        norms = core.model.norms(
            lambda: (read_block(source, tile, blocks, halo, stride, slabs) for tile in boxes), stats
        )
        cells: list[np.ndarray] = []  # each layer on its own grid, as a prediction's output is assembled
        ends: list[list[int]] = []  # how far along each axis its blocks reached: the whole image's layer
        for start, stop in boxes:
            box, own = read_block(source, (start, stop), blocks, halo, stride, slabs)
            for layer, (strides, features) in enumerate(owned_layers(core, box, own, stats, norms)):
                at = [a // k for a, k in zip(start, strides, strict=True)]
                if layer == len(cells):
                    cells.append(
                        np.lib.format.open_memmap(
                            str(directory / f"{side}_{index}_{layer}_cells.npy"),
                            mode="w+",
                            dtype=np.float32,
                            shape=(
                                features.shape[1],
                                *(-(-x // k) for x, k in zip(source.shape, strides, strict=True)),
                            ),
                        )
                    )
                    ends.append([0] * len(source.shape))
                reach = [a + n for a, n in zip(at, features.shape[2:], strict=True)]
                to_array(
                    cells[layer][(slice(None), *(slice(a, b) for a, b in zip(at, reach, strict=True)))], features[0]
                )
                ends[layer] = [max(end, b) for end, b in zip(ends[layer], reach, strict=True)]
        cells = [
            array[(slice(None), *(slice(0, end) for end in reach))] for array, reach in zip(cells, ends, strict=True)
        ]
        means: list[torch.Tensor] = []
        if core.pca > 0:  # fitted on the fixed image's layer, on its own grid, each side centred by its own mean
            fitted = [basis_of(array, blocks[0], core.pca) for array in cells]
            means = [mean for mean, _ in fitted]
            bases = [basis for _, basis in fitted] if side == "fixed" else bases
        arrays: list[np.ndarray] = []
        for layer, array in enumerate(cells):
            if tuple(array.shape[1:]) == tuple(source.shape) and core.pca == 0:  # on the image's grid already
                arrays.append(array)
                continue
            projection = (means[layer], bases[layer]) if core.pca > 0 else None
            path = directory / f"{side}_{index}_{layer}.npy"
            arrays.append(onto_image(array, source.shape, path, boxes, projection))
        return arrays, bases

    def onto_image(array: np.ndarray, shape: Sequence[int], path: Path, boxes, projection=None) -> np.ndarray:
        """Features on a model's grid brought onto the image's voxels over ``boxes``, as a run on the whole pair
        brings them (``resampled`` with the border past the last feature, which ``onto_image_grid`` is), each
        centred by the ``projection``'s mean and projected on its basis when given."""
        channels = array.shape[0] if projection is None else projection[1].shape[1]
        placed = np.lib.format.open_memmap(str(path), mode="w+", dtype=np.float32, shape=(channels, *shape))
        grid = tuple(array.shape[1:])
        for start, stop in boxes:
            box = list(zip(start, stop, strict=True))
            window = layer_window(box, grid, shape)
            features = to_device(array[(slice(None), *(slice(a, b) for a, b in window))], str(fixed.device))[None]
            if projection is not None:
                mean, basis = projection
                centred = features - mean.to(features.device).view(1, -1, *[1] * len(grid))
                features = torch.einsum("bc...,ck->bk...", centred, basis.to(features.device))
            to_array(
                placed[(slice(None), *(slice(a, b) for a, b in box))],
                onto_image_grid(features, tuple(shape), box, grid)[0],
            )
        return placed

    def halved(blocks: list[int], stride: int) -> bool:
        """``blocks[0]`` halved, a whole number of ``stride``: whether it could be."""
        if blocks[0] <= stride:
            return False
        blocks[0] = max(stride, blocks[0] // 2 // stride * stride)
        return True

    sources = {"fixed": fixed, "moving": moving}
    statistics = {side: statistics_of(source) for side, source in sources.items()}
    volumes: dict[str, list[tuple[np.ndarray, int]]] = {"fixed": [], "moving": []}
    cuda = torch.device(fixed.device).type == "cuda"
    loss._channels = []
    with torch.no_grad():
        for index, core in enumerate(loss.cores):
            # A model with a voxel_size reads each image on a grid at it over the image's extent, as a run on the
            # whole pair resamples it (``resampled``), normalised by that image's statistics; its features come back
            # onto the image's voxels as the whole run brings them (``onto_image``).
            read, read_statistics = sources, statistics
            if core.model.voxel_size is not None:
                read = {
                    side: ResampledSource(source, grid_size(extent, core.model.voxel_size, source.shape))
                    for (side, source), extent in zip(sources.items(), extents, strict=True)
                }
                read_statistics = {side: statistics_of(source) for side, source in read.items()}
            # A side at a time, the fixed one first: its features' basis projects both.
            bases: list[torch.Tensor] = []
            widths: list[int] = []
            try:
                # Blocks and halos on the input multiple and on the stride of the layers and normalisations gathered
                # over them, so that a block owns whole cells of each.
                stride = max(1, int(core.model.multiple), core.model.block_stride(fixed.device))
                blocks = [math.ceil(block / stride) * stride]
                try:  # half the receptive field around a block: its layers are then the whole image's
                    halo = (core.model.receptive_field - 1) // 2
                except MeasureError:  # a model that sees whole images: a quarter of a block around each
                    halo = blocks[0] // 4
                halo = math.ceil(halo / stride) * stride if core.dimension >= len(fixed.shape) else 0  # slabs: none
                # The larger the blocks, the less of the network's work goes to their halos; a probe the device does
                # not hold leaves them at ``block``, which the extraction halves while they do not fit.
                with contextlib.suppress(torch.cuda.OutOfMemoryError), out_of_memory_as_torch(cuda):
                    if cuda:
                        cost = block_cost(core, read_statistics["fixed"])
                        widest = int((free_bytes(fixed.device) / 2 / cost) ** (1 / len(fixed.shape)))
                        held = max(0, (widest - stride) // 2 // stride * stride)
                        if held < halo:  # no block with the whole field around it fits: the widest halo that does
                            print(
                                f"[FireANTs] a block with half the receptive field around it ({halo} voxels) does"
                                f" not fit the device: its features read {held} voxels around it.",
                                flush=True,
                            )
                            halo = held
                        blocks[0] = max(blocks[0], (widest - 2 * halo) // stride * stride)
                for side, source in read.items():
                    arrays, bases = halve_on_oom(
                        functools.partial(
                            extract, core, index, side, source, read_statistics[side], blocks, halo, stride, bases
                        ),
                        functools.partial(halved, blocks, stride),
                        cuda,
                    )
                    if source is not sources[side]:
                        shape = sources[side].shape
                        arrays = [
                            onto_image(
                                array, shape, directory / f"{side}_{index}_{layer}_image.npy", tiles(shape, block)
                            )
                            for layer, array in enumerate(arrays)
                        ]
                    widths = [int(array.shape[0]) for array in arrays]
                    volumes[side] += [(array, index) for array in arrays]
            finally:
                if cuda and core.model.half:  # a float16 cast rounds the cached weights in place: reloaded for later
                    core.model.reload()
            loss._channels.append(widths)
    stacked = []
    for side, source in sources.items():
        layers = [ArraySource(array, source.device) for array, _ in volumes[side]]
        stacked.append(layers[0] if len(layers) == 1 else StackedSource(layers))
    return stacked[0], stacked[1]


def sampled_deformable_class():
    """FireANTs' compositive greedy and SyN registrations over boxes of the warp; imported lazily, FireANTs being
    optional at import time."""
    from fireants.losses.cc import gaussian_1d
    from fireants.registration.abstract import AbstractRegistration

    class SampledDeformableRegistration(AbstractRegistration):
        """``symmetric`` picks SyN (a forward and a reverse warp meeting halfway) over greedy (one warp). ``fixed``
        and ``moving`` are :class:`RegionSource` of the native images (their mask as a last channel in masked mode);
        ``fixed_images`` and ``moving_images`` their :class:`HeaderImages`. ``box`` is the side of a box in voxels of
        a level: at most with ``samples``, at least without (``_fitting_box``).
        ``loss_halo`` is how far around a voxel the loss reads to give its gradient (twice the window's radius for
        a local correlation); ``least`` the voxels a box's region holds at least along an axis, the patch a sampled
        Jacobian loss runs its network on around each point it draws in it.

        With ``samples``, an iteration draws boxes of the grid at random, that many voxels of them, and steps those
        alone, as elastix steps on the points it draws: their gradients at the warps as they are, then their steps under the
        largest Adam direction the boxes last had. A box's gradient is the whole grid's over its voxels, so the
        warps go where FireANTs' own iteration takes them as the iterations grow, each box stepping as often as it
        is drawn; a draw of every box is FireANTs' iteration on the whole grid, whatever the boxes, and what the
        stage does without ``samples``, on boxes as large as the device holds. An iteration holds its boxes and no
        more, whatever the image. ``iterations`` counts, per scale, the steps a box takes
        on average: a scale runs the draws that hold that many steps of every box, rounded up to a whole draw.

        The warps and their Adam moments stay on the device when ``resident``; otherwise in host memory while this
        rank's budget holds them, in files of ``scratch`` beyond. By default each scale keeps them on the device when
        they fit three quarters of its free memory, and the sources join them there while that room lasts at the finest
        scale. ``resident`` tells whether they were on the device at the finest scale."""

        def __init__(
            self,
            scales,
            iterations,
            fixed_images: HeaderImages,
            moving_images: HeaderImages,
            fixed: RegionSource,
            moving: RegionSource,
            *,
            scratch: Path,
            symmetric: bool,
            loss_type: str = "cc",
            custom_loss=None,
            cc_kernel_size: int = 3,
            loss_halo: int | None = None,
            smooth_warp_sigma: float = 0.5,
            smooth_grad_sigma: float = 1.0,
            optimizer_lr: float = 0.1,
            init_affine: torch.Tensor | None = None,
            box: int = 64,
            samples: int | None = None,
            seed: int = 0,
            resident: bool | None = None,
            least: int = 1,
        ) -> None:
            super().__init__(
                scales=scales,
                iterations=iterations,
                fixed_images=fixed_images,
                moving_images=moving_images,
                loss_type=loss_type,
                custom_loss=custom_loss,
                cc_kernel_size=cc_kernel_size,
                loss_params={},
                progress_bar=False,
            )
            self._scratch, self._symmetric = Path(scratch), symmetric
            # Three quarters of the device's free memory hold what every box reads: the warps, their moments and
            # directions at the finest scale first, which a step also writes, then the two sources; the rest is the
            # boxes' work (``_fitting_box``). What does not fit stays in files, where each step moves it through the
            # host.
            room = free_bytes(fixed.device) * 3 / 4
            needed = 4 * len(fixed.shape) * 4 * math.prod(fixed.shape) * (2 if symmetric else 1)
            self._forced = resident
            if resident is None:
                resident = needed <= room
            self._sources = tuple(held((fixed, moving), room - needed if resident else room))
            self.resident = bool(resident)
            self._lr, self._betas, self._eps = float(optimizer_lr), (0.9, 0.99), 1e-8  # FireANTs' Adam
            self._box, self._samples, self._least = int(box), samples, int(least)
            self._draws = random.Random(int(seed))
            device = self.device

            def kernels(sigma: float):
                if sigma <= 0:
                    return None
                return [gaussian_1d(s, truncated=2) for s in torch.zeros(self.dims, device=device) + sigma]

            self._grad_gaussians, self._warp_gaussians = kernels(smooth_grad_sigma), kernels(smooth_warp_sigma)
            grad_tail, warp_tail = (
                (int(kernel[0].numel()) - 1) // 2 if kernel else 0
                for kernel in (self._grad_gaussians, self._warp_gaussians)
            )
            if loss_halo is None:
                loss_halo = 2 * ((cc_kernel_size - 1) // 2) if loss_type.endswith("cc") else 0
            # What the two halves of a step read around a voxel: the loss and the smoothing of its gradient; then the
            # smoothing of the warp, and for the composition how far a step moves a voxel, half the rate in voxels at
            # most (``_step``).
            # A Jacobian IMPACT loss runs its networks on the boxes: they start, and end as far from the level's end,
            # on a whole number of the networks' stride, so that a strided layer meets the whole level's cells.
            stride = getattr(custom_loss, "stride", None)
            self._stride = int(stride(device)) if callable(stride) else 1
            self._box = max(self._stride, self._box - self._box % self._stride)
            self._halo = -(-(int(loss_halo) + grad_tail) // self._stride) * self._stride
            self._step_halo = warp_tail + max(1, math.ceil(float(optimizer_lr) / 2))
            affine = (
                torch.eye(self.dims + 1, device=device)
                if init_affine is None
                else init_affine.reshape(-1, self.dims + 1, self.dims + 1)[0]
            )
            self._affine = affine.to(device, self.dtype)
            self._warps: dict[str, _Warp] = {}
            self._levels: dict[int, tuple[RegionSource, RegionSource]] = {}
            self.losses: list[float] = []
            self.boxes_per_scale: list[int] = []  # how many boxes each scale's grid was cut into

        # ------------------------------------------------------------------ levels

        def _level_sources(self, scale: int) -> tuple[RegionSource, RegionSource]:
            """The images of a scale as FireANTs computes them: the fixed one smoothed and downsampled; the moving
            one smoothed on its own grid for SyN, smoothed and downsampled for greedy. At the last scale SyN
            interpolates the fixed image onto its grid, unsmoothed: the image itself, unless an axis is under
            MIN_IMG_SIZE, where the grid is finer than the image."""
            fixed, moving = self._sources
            # One scale's levels at a time: SyN's moving one is as large as the image at every scale.
            for other in [held_scale for held_scale in self._levels if held_scale != scale]:
                for level in self._levels.pop(other):
                    level.close()
            size = fireants_level_sizes(fixed.shape, scale)
            if scale == 1 and size == list(fixed.shape):
                return self._sources
            if scale not in self._levels:
                settings = {"mask_channel": self.masked, "directory": self._scratch}
                gaussians = fireants_gaussians(fixed.shape, size, fixed.device) if scale > 1 else None
                if scale == 1:
                    levels = (build_level(fixed, size, None, **settings),)  # the moving image as it is
                elif self._symmetric:
                    levels = (
                        build_level(fixed, size, gaussians, **settings),
                        blur_level(moving, gaussians, **settings),
                    )
                else:
                    moving_size = fireants_level_sizes(moving.shape, scale)
                    levels = (
                        build_level(fixed, size, gaussians, **settings),
                        build_level(moving, moving_size, gaussians, **settings),
                    )
                # On the device while they fit half of what it has left, as the sources are.
                self._levels[scale] = tuple(held(levels, free_bytes(fixed.device) / 2))
            levels = self._levels[scale]
            return (levels[0], moving) if len(levels) == 1 else levels

        def _placement(self, grid: Sequence[int], count: int) -> tuple[bool, bool]:
            """Where ``count`` warps on ``grid`` live: on the device while they fit three quarters of it (a coarser
            scale than one the device cannot hold may fit it), off it in host memory while this rank's budget holds
            them, in files beyond, where every step's writes go to the disk."""
            voxels, a_voxel = math.prod(grid), 4 * len(grid) * 4 * count  # 4 arrays of a float32 channel an axis
            resident = self.resident or (self._forced is None and voxels * a_voxel <= free_bytes(self.device) * 3 / 4)
            return resident, not resident and voxels <= (max_voxels(None, a_voxel, None) or 0)

        def _set_grid(self, grid: Sequence[int]) -> None:
            names = ("fwd", "rev") if self._symmetric else ("fwd",)
            resident, host = self._placement(grid, len(names))
            for name in names:
                if name in self._warps:
                    self._warps[name].resize(grid, resident, host)
                else:
                    self._warps[name] = _Warp(self._scratch, name, grid, self.device, resident, host)

        def _matrix(self) -> torch.Tensor:
            """The affine between the normalized coordinates of the fixed and the moving grids, whatever the scale."""
            fixed_t2p = self.fixed_images.get_torch2phy().to(self.dtype)[0]
            moving_p2t = self.moving_images.get_phy2torch().to(self.dtype)[0]
            return moving_p2t @ self._affine @ fixed_t2p

        # ------------------------------------------------------------------ one iteration

        def _region(self, tile, grid, halo: int, aligned: bool = True):
            """A box with ``halo`` around, at least the patch a sampled Jacobian loss runs its network on; ``aligned``,
            ending as far from the level's end as a whole number of the loss's stride (``block_region``)."""
            return box_region(tile, halo, grid, self._least, self._stride if aligned else 1)

        def _warped(self, low, high, identity, levels, warps) -> tuple[torch.Tensor, torch.Tensor]:
            """The two images the loss compares over a box of coordinates ``identity``: the moving one through the
            affine and the forward warp, and the fixed one (through the reverse warp for SyN)."""
            fixed_level, moving_level = levels
            matrix = self._matrix()
            moved = _sample_region(moving_level, identity @ matrix[:-1, :-1].T + matrix[:-1, -1] + warps["fwd"])
            if self._symmetric:
                return moved, _sample_region(fixed_level, identity + warps["rev"])
            return moved, fixed_level.read(low, high)

        def _boxes(self, boxes, grid, levels):
            """Each of ``boxes`` read with the loss's halo, at the warps as they are: the two images the loss compares
            over it, and ones on the box's own voxels."""
            for tile in boxes:
                low, high, core = self._region(tile, grid, self._halo)
                self._place(low, high, grid)
                warps = {name: warp.read("warp", low, high) for name, warp in self._warps.items()}
                identity = _box_coordinates(low, high, grid, self.device, self.dtype)
                with torch.no_grad():
                    moved, target = self._warped(low, high, identity, levels, warps)
                region = torch.zeros((1, 1, *moved.shape[2:]), device=moved.device, dtype=moved.dtype)
                region[(slice(None), slice(None), *core)] = 1
                yield moved, target, region

        def _place(self, low, high, grid) -> None:
            """Tell a loss that reads a level on its own grid (an IMPACT model with a voxel_size) where the next box lies."""
            place = getattr(self.loss_fn, "place", None)
            if place is not None:
                place(low, high, grid)

        def _support(self, moved: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """The voxels a masked loss averages over, weighed as it weighs them: FireANTs' metrics take the product
            of the two masks, which moves with the warp; IMPACT's the voxels both masks keep."""
            if hasattr(self.loss_fn, "forward_util"):
                from fireants.losses.maskedutils import get_tensors_and_mask

                return get_tensors_and_mask(moved, target, self.loss_fn.mask_mode)[2]
            return (moved[:, -1:] * target[:, -1:] >= 0.5).to(moved.dtype)

        def _information_weight(self, boxes, grid, levels, share: float) -> _Information:
            """Mutual information over the voxels of ``boxes`` (each counted once, in its own box), a draw holding
            ``share`` of the level: the level's joint histogram is tracked from draw to draw, each moving it by its
            share (every box drawn: the level's own), and the information and its derivative are read off it, as the
            linear stages read theirs (``SampledLinearMixin._information_step``)."""
            joint, points = None, 0.0
            with torch.no_grad():
                for moved, target, region in self._boxes(boxes, grid, levels):
                    counts, counted = _joint_counts(self.loss_fn, moved, target, region)
                    joint = counts.double() if joint is None else joint + counts.double()
                    points += counted
            self._histogram, information = _tracked(self.loss_fn, self._histogram, joint, points, share)
            if information is None:
                return _Information(torch.zeros(()), 1.0, 0.0)
            if points == 0:  # a draw that holds no point of the loss moves nothing
                return _Information(torch.zeros_like(information.weight), 1.0, information.value)
            return information

        def _level_totals(self, boxes, grid, levels) -> tuple[float, float]:
            """A masked loss over the voxels of ``boxes`` (each counted once, in its own box): its support, and its
            mean where the loss gives its value at each voxel (FireANTs' metrics), 0 otherwise."""
            support = total = 0.0
            for moved, target, region in self._boxes(boxes, grid, levels):
                with torch.no_grad():
                    if hasattr(self.loss_fn, "forward_util"):
                        from fireants.losses.maskedutils import get_tensors_and_mask

                        moved, target, mask = get_tensors_and_mask(moved, target, self.loss_fn.mask_mode)
                        total += float((self.loss_fn.forward_util(moved, target) * mask * region).sum())
                    else:
                        mask = self._support(moved, target)
                    support += float((mask * region).sum())
            return support, total / max(support, 1e-8)

        def _loss_gradients(self, tile, grid, levels, totals=None) -> tuple[float, list[torch.Tensor]]:
            """The loss of one box and its smoothed gradient with respect to each warp, over the box. ``totals``: a
            masked loss's support and mean over the level (``_level_totals``)."""
            low, high, core = self._region(tile, grid, self._halo)
            self._place(low, high, grid)
            warps = [warp.read("warp", low, high).requires_grad_() for warp in self._warps.values()]
            identity = _box_coordinates(low, high, grid, self.device, self.dtype)
            moved, target = self._warped(low, high, identity, levels, dict(zip(self._warps, warps, strict=True)))
            # The loss reported counts the box's own voxels alone, its halo being another box's: summed over the
            # boxes, the grid's.
            owned = (slice(None), slice(None), *core)
            if isinstance(totals, _Information):
                # Mutual information is a function of the level's joint histogram: a voxel's counts weighed by the
                # information's derivative there have its gradient, every voxel of the box, its halo included (which
                # the gradient's smoothing reads); the box reports its own voxels' share of the information.
                counts, _ = _joint_counts(self.loss_fn, moved, target)
                loss = (counts * totals.weight).sum() / totals.points
                region = torch.zeros_like(moved[:, :1])
                region[owned] = 1
                with torch.no_grad():
                    counted = _joint_counts(self.loss_fn, moved, target, region)[1]
                reported = torch.tensor(totals.value * counted / totals.points)
            elif totals is None:
                value = self.loss_fn(moved, target)
                # The loss is a mean over the box: weighed as the box's share of the grid, its gradient is the whole
                # grid's wherever the box holds what the loss reads.
                loss = value * (math.prod(moved.shape[2:]) / math.prod(grid))
                reported = value * (math.prod(moved[owned].shape[2:]) / math.prod(grid))
            else:
                # A masked mean over the box is its sum over the box's support S_b. Against the level's support S
                # and mean c, the box adds S_b (d_b - c) / S, whose gradient is the level's masked mean's wherever the
                # box holds what the loss reads, the support moving with the warp as it does on the whole grid.
                value = self.loss_fn(moved, target)
                support, mean = totals
                mask = self._support(moved, target)
                # A draw the masks miss (its boxes outside them) holds no voxel of the loss: no gradient.
                weight = mask.sum() if support > 0 else value * 0.0
                support = support if support > 0 else 1.0
                loss, reported = weight * (value - mean) / support, value * mask[owned].sum().detach() / support
            inner = (slice(None), *core)
            # A box the warp sends wholly outside an image reads zeros there, which hold no gradient.
            raw = torch.autograd.grad(loss, warps, allow_unused=True) if loss.requires_grad else [None] * len(warps)
            gradients = [
                _smooth(torch.zeros_like(warp) if gradient is None else gradient, self._grad_gaussians)[inner]
                for warp, gradient in zip(warps, raw, strict=True)
            ]
            return float(reported.detach()), gradients

        def _direction(self, mean: torch.Tensor, square: torch.Tensor, steps) -> torch.Tensor:
            """Adam's direction after ``steps`` steps, from a warp's moments."""
            beta1, beta2 = self._betas
            return (mean / (1 - beta1**steps)) / ((square / (1 - beta2**steps)).sqrt() + self._eps)

        def _gradient(self, tile, grid, levels, steps: int, totals=None) -> tuple[float, dict[str, float]]:
            """The first half of a step on one box, its ``steps``-th: its loss, and the smoothed gradient of each
            warp folded into the warp's moments over the box. Returns the loss and each warp's largest Adam
            direction over the box."""
            loss, gradients = self._loss_gradients(tile, grid, levels, totals)
            return loss, self._moments(tile, gradients, steps)

        def _moments(self, tile, gradients: list[torch.Tensor], steps: int) -> dict[str, float]:
            """Each warp's gradient over a box folded into the warp's moments, its ``steps``-th: each warp's largest
            Adam direction over the box."""
            beta1, beta2 = self._betas
            peaks = {}
            for (name, store), gradient in zip(self._warps.items(), gradients, strict=True):
                mean = store.read("mean", *tile).mul_(beta1).add_(gradient, alpha=1 - beta1)
                square = store.read("square", *tile).mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                store.write("mean", tile[0], mean)
                store.write("square", tile[0], square)
                peaks[name] = float(self._direction(mean, square, steps).norm(p=2, dim=-1).max())
            return peaks

        def _fitting_box(self, grid, levels) -> int:
            """The side of the boxes of a scale: ``box`` at least, and as large as half the device's free memory
            holds. A box's first half is run on a small one to tell what a voxel takes; the larger the boxes, the
            less of their work goes to their halos. The result does not depend on them."""
            device = torch.device(self.device)
            if device.type != "cuda" or self._box >= max(grid):
                return self._box
            tile = ([0] * len(grid), [min(32, extent) for extent in grid])
            low, high, _ = self._region(tile, grid, self._halo)
            # The gradients are let go before the free memory is read.
            taken = peak_bytes(lambda: self._loss_gradients(tile, grid, levels), device)[1]
            a_voxel = max(taken / math.prod(b - a for a, b in zip(low, high, strict=True)), 1.0)
            side = int((free_bytes(self.device) / 2 / a_voxel) ** (1 / len(grid))) - 2 * self._halo
            return max(self._box, min(side - side % 8, max(grid)))

        def _step(self, tile, grid, largest: dict[str, float], steps: np.ndarray, side: int) -> None:
            """The second half: FireANTs' compositive update of each warp from Adam's direction, normalised by the
            largest of the grid, then smoothed. The box is written to the warp the iteration steps to. The smoothing
            reads the update around the box: each voxel's direction is its box's, after the steps that box has taken
            (``steps``, one count per box of ``side`` voxels)."""
            half_resolution = 1.0 / (max(grid) - 1)
            low, high, core = self._region(tile, grid, self._step_halo, aligned=False)
            identity = _box_coordinates(low, high, grid, self.device, self.dtype)
            counts = steps[np.ix_(*[np.arange(a, b) // side for a, b in zip(low, high, strict=True)])]
            # A box that has not stepped holds no moment: one step's correction leaves its direction at zero.
            counts = torch.from_numpy(np.maximum(counts, 1)).to(self.device, self.dtype)[None, ..., None]
            for name, store in self._warps.items():
                update = self._direction(store.read("mean", low, high), store.read("square", low, high), counts)
                update = update / max(largest[name] + self._eps, 1.0) * half_resolution * (-self._lr)
                # The warp read where the update sends each voxel.
                warp = store.read("warp", low, high).movedim(-1, 1)
                composed = _interpolate(warp, low, high, grid, identity + update).movedim(1, -1)
                store.write("next", tile[0], _smooth(update + composed, self._warp_gaussians)[(slice(None), *core)])

        def optimize(self) -> None:
            from fireants.losses.mi import GlobalMutualInformationLoss

            fixed, _ = self._sources
            for scale, epochs in zip(self.scales, self.iterations, strict=True):
                if hasattr(self.loss_fn, "set_current_scale_and_iterations"):
                    self.loss_fn.set_current_scale_and_iterations(scale, epochs)
                self.convergence_monitor.reset()
                grid = fireants_level_sizes(fixed.shape, scale)
                levels = self._level_sources(scale)
                self._set_grid(grid)
                draw = getattr(self.loss_fn, "draw", None)  # one draw an evaluation of the level, as FireANTs makes
                if hasattr(self.loss_fn, "latch"):
                    # A loss that normalises itself on its first evaluation of a scale is given every box of the
                    # scale, one at a time: what the whole grid would give it.
                    if draw is not None:
                        draw()
                    if getattr(self.loss_fn, "pools", False):  # the level's sums first: an NCC layer's start
                        self.loss_fn.pool(functools.partial(self._boxes, tiles(grid, self._box), grid, levels))
                    self.loss_fn.latch(self._boxes(tiles(grid, self._box), grid, levels))
                if self._samples is None:
                    # Not told how many samples an iteration draws, it takes what the device holds: every box, as
                    # large as its memory takes them, which is FireANTs' iteration on the whole grid.
                    side = self._fitting_box(grid, levels)
                    side = max(self._stride, side - side % self._stride)
                    boxes = tiles(grid, side)
                    count = len(boxes)
                else:
                    side = min(self._box, max(8, round(self._samples ** (1 / len(grid)))))
                    side = max(self._stride, side - side % self._stride)
                    boxes = tiles(grid, side)
                    count = min(len(boxes), max(1, self._samples // side ** len(grid)))
                self.boxes_per_scale.append(len(boxes))
                # Over several boxes, a mutual information is the level's histogram's and a masked mean is weighed
                # against the level's: one pass more an iteration.
                information = len(boxes) > 1 and type(self.loss_fn) is GlobalMutualInformationLoss
                # An IMPACT loss's level sums (its NCC layers', Jacobian mode's intensities), read over the boxes'
                # own voxels however many there are: a box read with its halo is more than the level it holds.
                pools = bool(getattr(self.loss_fn, "pools", False))
                masked = (
                    not information
                    and len(boxes) > 1
                    and bool(getattr(self.loss_fn, "masked", getattr(self.loss_fn, "_masked", False)))
                )
                self._histogram = None  # the level's joint histogram, as the draws of a mutual information tell it
                # The steps each box has taken at this scale, laid out as the boxes are (tiles' order is C order).
                steps = np.zeros([math.ceil(extent / side) for extent in grid], dtype=np.int64)
                peaks = {name: np.zeros(len(boxes)) for name in self._warps}  # each box's largest direction, last seen
                for _ in range(math.ceil(epochs * len(boxes) / count)):
                    # Every box, or a draw of ``count`` that costs ``count``, not the boxes.
                    drawn = (
                        list(range(len(boxes))) if count == len(boxes) else self._draws.sample(range(len(boxes)), count)
                    )
                    if draw is not None:
                        draw()
                    if pools:  # the level's sums over the boxes drawn, under this iteration's draw
                        self.loss_fn.pool(
                            functools.partial(self._boxes, [boxes[index] for index in drawn], grid, levels)
                        )
                    totals: _Information | tuple[float, float] | None = None
                    if information:
                        totals = self._information_weight(
                            [boxes[index] for index in drawn], grid, levels, count / len(boxes)
                        )
                    elif masked:  # a draw's support stands for the level's in proportion to the boxes it holds
                        support, mean = self._level_totals([boxes[index] for index in drawn], grid, levels)
                        totals = (support * len(boxes) / count, mean)
                    total = 0.0
                    for index in drawn:
                        steps.flat[index] += 1
                        loss, seen = self._gradient(boxes[index], grid, levels, int(steps.flat[index]), totals)
                        total += loss
                        for name, peak in seen.items():
                            peaks[name][index] = peak
                    largest = {name: float(values.max()) for name, values in peaks.items()}
                    # Every drawn box steps from the warps of the iteration into the warp it steps to, which then
                    # takes their place: whole when every box is drawn, box by box otherwise.
                    for index in drawn:
                        self._step(boxes[index], grid, largest, steps, side)
                    for warp in self._warps.values():
                        if count == len(boxes):
                            warp.swap()
                        else:
                            for index in drawn:
                                warp.keep(*boxes[index])
                    # The level's loss: a masked metric's mean over the level where the pass gave it, the boxes'
                    # shares otherwise.
                    exact = isinstance(totals, tuple) and hasattr(self.loss_fn, "forward_util")
                    if isinstance(totals, _Information):
                        level = totals.value
                    else:
                        level = totals[1] if exact else total * len(boxes) / count
                    self.losses.append(level)
                    # FireANTs leaves a scale once its loss stops falling, which a draw's loss is too noisy to tell.
                    if count == len(boxes) and self.convergence_monitor.converged(level):
                        break

        # ------------------------------------------------------------------ the transform

        def _scaled(self, warp: _Warp, size: Sequence[int]) -> RegionSource:
            """``warp`` resampled to ``size`` by linear interpolation (``align_corners``), as ``F.interpolate`` does."""
            if warp.resident:
                return TensorSource(resized(warp.arrays["warp"][None], size)[0])
            return build_level(warp.source(), size, None, directory=self._scratch, memory=warp.host)

        def _consistency(self, tile, grid, reference: RegionSource, halo, margins) -> tuple[float, torch.Tensor]:
            """One box's share of the loss ``compositive_warp_inverse`` minimises, ``InverseConsistencyOperator``'s: the
            mean over the reference's grid of u(y + r(y)) + r(y) squared, plus the mean over u's grid of r(x + u(x)) +
            u(x) squared. Every term of the box's window counts, ``halo`` covering how far r sends a term and
            ``margins`` (normalized, an axis) how far u does: the gradient over the box's own voxels is the whole
            grid's. Returns the terms the box owns and that gradient."""
            low, high, core = box_region(tile, halo, grid)
            inverse = self._warps["inv"].read("warp", low, high).requires_grad_()
            shape, dims = reference.shape, len(grid)
            x = _box_coordinates(low, high, grid, self.device, self.dtype)
            # r read where u sends the window's voxels, from the reference's nodes around them.
            span = [(r - 1) / max(g - 1, 1) for g, r in zip(grid, shape, strict=True)]
            qlow = [
                max(0, math.floor(a * k - m * 0.5 * (r - 1)) - 1)
                for a, k, m, r in zip(low, span, margins, shape, strict=True)
            ]
            qhigh = [
                min(r, math.ceil((b - 1) * k + m * 0.5 * (r - 1)) + 2)
                for b, k, m, r in zip(high, span, margins, shape, strict=True)
            ]
            second = _interpolate(reference.read(qlow, qhigh), qlow, qhigh, shape, x + inverse).movedim(1, -1) + inverse
            total = (second**2).sum() / (dims * math.prod(grid))
            # u read where r sends the reference's nodes in the window, which each box owns from its first voxel on.
            nodes = [
                (_nodes_from(a, g, r), _nodes_from(b, g, r)) for a, b, g, r in zip(low, high, grid, shape, strict=True)
            ]
            owned = [
                slice(_nodes_from(a, g, r) - n, _nodes_from(b, g, r) - n)
                for a, b, g, r, (n, _) in zip(*tile, grid, shape, nodes, strict=True)
            ]
            first = None
            if all(b > a for a, b in nodes):
                rlow, rhigh = [a for a, _ in nodes], [b for _, b in nodes]
                r = reference.read(rlow, rhigh).movedim(1, -1)
                y = _box_coordinates(rlow, rhigh, shape, self.device, self.dtype)
                first = _interpolate(inverse.movedim(-1, 1), low, high, grid, y + r).movedim(1, -1) + r
                total = total + (first**2).sum() / (dims * math.prod(shape))
            (gradient,) = torch.autograd.grad(total, inverse)
            with torch.no_grad():
                own = (second[(slice(None), *core)] ** 2).sum() / (dims * math.prod(grid))
                if first is not None:
                    own = own + (first[(slice(None), *owned)] ** 2).sum() / (dims * math.prod(shape))
            return float(own), gradient[(slice(None), *core)]

        def _inverse(self, block: int) -> _Warp:
            """The reverse warp's inverse as FireANTs computes it (``compositive_warp_inverse``), through which its SyN
            field reads the forward warp: a greedy registration of a warp alone from minus the reverse one, at the
            stage's scales and iterations, Adam at a rate of 0.5 with nothing smoothed and no image compared, its loss
            the two warps' inverse consistency. Both of its means are sums over voxels, which each box adds its own to
            (``_consistency``): an iteration over every box is the whole grid's."""
            fixed, _ = self._sources
            full, dims = list(fixed.shape), len(fixed.shape)
            # On its own grid, the last scale's, as SyN hands it over (``rev_warp.get_warp()``): an axis under 32
            # voxels, or a last scale above 1, keeps it off the fixed grid.
            reverse = self._warps["rev"]
            last = list(reverse.grid)
            inverse = _Warp(self._scratch, "inv", last, self.device, *self._placement(last, 1))
            for start, stop in tiles(last, block):
                inverse.write("warp", start, -reverse.read("warp", start, stop))
            peaks = [reverse.peak(channel) for channel in range(dims)]  # r at a scale, interpolated, is no larger
            saved = (self._warps, self._lr, self._warp_gaussians, self._step_halo)
            self._warps, self._lr, self._warp_gaussians, self._step_halo = {"inv": inverse}, 0.5, None, 1
            try:
                for scale, iterations in zip(self.scales, self.iterations, strict=True):
                    self.convergence_monitor.reset()
                    grid = fireants_level_sizes(full, scale)
                    inverse.resize(grid, *self._placement(grid, 1))
                    # The reference as the operator reads it: its own grid scaled down by the largest ratio of the
                    # image to u's grid.
                    ratio = max(n / g for n, g in zip(full, grid, strict=True))
                    reference = (
                        self._scaled(reverse, [math.floor(float(n) * (1.0 / ratio)) for n in last])
                        if ratio > 1
                        else reverse.source()
                    )
                    halo = [
                        math.ceil(peaks[dims - 1 - axis] * 0.5 * (extent - 1)) + 2 for axis, extent in enumerate(grid)
                    ]
                    boxes = tiles(grid, block)
                    for iteration in range(iterations):
                        margins = [inverse.peak(dims - 1 - axis) for axis in range(dims)]
                        loss, largest = 0.0, 0.0
                        for tile in boxes:
                            own, gradient = self._consistency(tile, grid, reference, halo, margins)
                            loss += own
                            largest = max(largest, self._moments(tile, [gradient], iteration + 1)["inv"])
                        steps = np.full([math.ceil(extent / block) for extent in grid], iteration + 1)
                        for tile in boxes:
                            self._step(tile, grid, {"inv": largest}, steps, block)
                        inverse.swap()
                        if self.convergence_monitor.converged(loss):
                            break
                    if ratio > 1 and isinstance(reference, ArraySource):
                        reference.close()
            finally:
                self._warps, self._lr, self._warp_gaussians, self._step_halo = saved
            inverse.resize(full, *self._placement(full, 1))  # a last scale coarser than the image
            return inverse

        @torch.no_grad()
        def write_field(self, field: np.ndarray, block: int = 96) -> None:
            """The registration's displacement field, physical, on the fixed grid, written block by block to
            ``field`` (``[*grid, 3]``, (x, y, z) components): the affine and the warp for greedy; for SyN the
            forward warp read through FireANTs' inverse of the reverse one (``_inverse``), added to the affine as
            it is, as FireANTs' own ``SyNRegistration.get_warp_parameters`` composes them: the field of the
            whole-image run."""
            fixed, _ = self._sources
            grid = list(fixed.shape)
            fixed_t2p = self.fixed_images.get_torch2phy().to(self.dtype)[0]
            moving_t2p = self.moving_images.get_torch2phy().to(self.dtype)[0]
            linear = moving_t2p @ self._matrix() - fixed_t2p
            with torch.enable_grad():  # the reverse warp on its own grid, before the forward one goes to the fixed grid
                inverse = self._inverse(block) if self._symmetric else None
            self._set_grid(grid)
            forward = self._warps["fwd"].source()
            try:
                for start, stop in tiles(grid, block):
                    identity = _box_coordinates(start, stop, grid, self.device, self.dtype)
                    if inverse is not None:
                        inverted = inverse.read("warp", start, stop)
                        composed = inverted + _sample_region(forward, identity + inverted).movedim(1, -1)
                    else:
                        composed = forward.read(start, stop).movedim(1, -1)
                    displacement = composed @ moving_t2p[:-1, :-1].T + identity @ linear[:-1, :-1].T + linear[:-1, -1]
                    to_array(field[tuple(slice(a, b) for a, b in zip(start, stop, strict=True))], displacement[0])
            finally:
                if inverse is not None:
                    inverse.close()

        def close(self) -> None:
            for warp in self._warps.values():
                warp.close()
            for pair in self._levels.values():
                for level in pair:
                    level.close()

        def get_warp_parameters(self, *args, **kwargs):
            raise NotImplementedError("a sampled registration writes its field by blocks (write_field).")

        get_inverse_warp_parameters = get_warp_parameters

    return SampledDeformableRegistration

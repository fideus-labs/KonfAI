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

"""FireANTs' stages over boxes against FireANTs on the whole image, on synthetic volumes, on the CPU."""

import functools
import math
from types import SimpleNamespace
from typing import Optional
from unittest import mock

import numpy as np
import pytest
import SimpleITK as sitk
import torch

fireants = pytest.importorskip("fireants")

from impact_reg_konfai.models.fireants_sampled import (  # noqa: E402
    _box_to_global,
    _local_frame,
    sampled_affine_class,
    sampled_deformable_class,
    sampled_linear_alignment,
    sampled_rigid_class,
)
from impact_reg_konfai.ooc import (  # noqa: E402
    ArraySource,
    HeaderImages,
    OnesSource,
    ResampledSource,
    RescaledSource,
    StackedSource,
    blur_level,
    box_region,
    build_level,
    fireants_gaussians,
    fireants_level_sizes,
    tiles,
)

SIZE = (34, 40, 44)  # z, y, x
PLANE = SIZE[1:]  # y, x: the sampled stages take a plane as they take a volume
SHAPES = pytest.mark.parametrize("size", [SIZE, PLANE], ids=["volume", "plane"])


def texture(shift=(0.0, 0.0, 0.0), size=SIZE) -> np.ndarray:
    """A smooth texture f(x + shift), in [0, 1): waves a dozen voxels long and a blob; on a plane, its section z = 8."""
    grids = np.meshgrid(*(np.arange(n, dtype=np.float64) for n in size), indexing="ij")
    z = grids[0] + shift[2] if len(size) == 3 else np.full_like(grids[-1], 8.0)
    x, y = grids[-1] + shift[0], grids[-2] + shift[1]
    waves = np.sin(x / 2.6) * np.cos(y / 3.1) + 0.5 * np.sin((x + y + 2 * z) / 4.2) + 0.7 * np.cos(z / 2.9 + x / 5.3)
    blob = 2.0 * np.exp(-((x - 22) ** 2 + (y - 20) ** 2 + (z - 17) ** 2) / 60.0)
    values = waves + blob
    return ((values - values.min()) / (values.max() - values.min()) * 0.98).astype(np.float32)


def image_of(array: np.ndarray) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0,) * array.ndim)
    return image


def whole_registration(cls, fixed: np.ndarray, moving: np.ndarray, **kwargs):
    from fireants.io import BatchedImages, Image

    batched = [BatchedImages([Image(image_of(array), device="cpu")]) for array in (fixed, moving)]
    return cls(fixed_images=batched[0], moving_images=batched[1], progress_bar=False, **kwargs)


def sampled_registration(cls, fixed: np.ndarray, moving: np.ndarray, *, box, boxes, seed=0, together=10**9, **kwargs):
    headers = [HeaderImages(image_of(array), 1, "cpu") for array in (fixed, moving)]
    registration = cls(fixed_images=headers[0], moving_images=headers[1], progress_bar=False, **kwargs)
    registration.sample(
        ArraySource(fixed[None], "cpu"),
        ArraySource(moving[None], "cpu"),
        box=box,
        boxes=boxes,
        seed=seed,
        together=together,
        halo=1 if kwargs.get("loss_type") == "cc" else 0,  # the radius of the correlation's window (3 voxels)
    )
    return registration


def test_box_frames_are_inverse_of_each_other():
    shape = (34, 40, 44)
    start, stop = (3, 8, 10), (19, 24, 42)
    forward = _box_to_global(start, stop, shape, "cpu", torch.float64)
    scale, offset = _local_frame(start, stop, shape, "cpu", torch.float64)
    corners = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [0.3, -0.2, 0.7]], dtype=torch.float64)
    homogeneous = torch.cat([corners, torch.ones(3, 1, dtype=torch.float64)], dim=1)
    assert torch.allclose((homogeneous @ forward.T)[:, :3] * scale + offset, corners, atol=1e-12)
    # Voxel i of the box, at its own -1 + 2 k / (b - 1), lands at the image's -1 + 2 i / (N - 1).
    k, b, n, i0 = 5, 16, 34, 3
    point = torch.tensor([0.0, 0.0, -1.0 + 2 * k / (b - 1), 1.0], dtype=torch.float64)  # z is the last normalized axis
    mapped = forward @ point
    assert mapped[2].item() == pytest.approx(-1.0 + 2 * (i0 + k) / (n - 1))


def test_a_box_s_region_holds_the_patch_its_points_read():
    """A sampled Jacobian loss runs its network on a patch around each point a box draws: the last box along an axis,
    short and at the image's edge, is read wide enough for one where the grid is."""
    low, high, core = box_region(((60, 0, 30), (64, 16, 40)), 2, (64, 20, 40), least=23)
    assert (low, high) == ([41, 0, 17], [64, 20, 40])  # 23 voxels where the axis has them, its 20 otherwise
    assert core == (slice(19, 23), slice(0, 16), slice(13, 23))


def test_a_source_far_finer_than_its_coordinates_is_read_by_parts_as_it_is_whole():
    """Coordinates of a few voxels over a whole finer source are read in parts, each a box of its own: the values
    and the gradient of one read of the whole source."""
    from impact_reg_konfai.models.fireants_sampled import _interpolate, _sample_region

    source = ArraySource(texture()[None], "cpu")
    grid = torch.stack(torch.meshgrid(*(torch.linspace(-0.95, 0.9, n) for n in (3, 4, 5)), indexing="ij"), -1)
    coordinates = grid.flip(-1)[None].requires_grad_()  # (x, y, z) last
    parts = _sample_region(source, coordinates)
    whole = _interpolate(source.read([0, 0, 0], list(SIZE)), [0, 0, 0], list(SIZE), SIZE, coordinates)
    assert torch.allclose(parts, whole, atol=1e-6)
    gradients = [torch.autograd.grad(values.sum(), coordinates)[0] for values in (parts, whole)]
    assert torch.allclose(*gradients, atol=1e-5)


def test_level_equals_fireants_downsample():
    from fireants.losses.cc import separable_filtering
    from fireants.utils.imageutils import downsample

    array = texture()
    source = ArraySource(array[None], "cpu")
    size = fireants_level_sizes(SIZE, 2)
    gaussians = fireants_gaussians(SIZE, size, "cpu")
    level = build_level(source, size, gaussians, block=13)
    expected = downsample(torch.from_numpy(array)[None, None], size=size, mode="trilinear", gaussians=gaussians)
    assert np.abs(level.array - expected[0].numpy()).max() < 1e-5
    blurred = blur_level(level, gaussians, block=11)
    expected_blurred = separable_filtering(expected, gaussians)
    assert np.abs(blurred.array - expected_blurred[0].numpy()).max() < 1e-5
    # The mask channel is interpolated, not smoothed.
    mask = (texture((5.0, 0.0, 0.0)) > 0.5).astype(np.float32)
    stacked = StackedSource([source, ArraySource(mask[None], "cpu")])
    level_masked = build_level(stacked, size, gaussians, mask_channel=True, block=13)
    expected_mask = torch.nn.functional.interpolate(
        torch.from_numpy(mask)[None, None], size=size, mode="trilinear", align_corners=True
    )
    assert np.abs(level_masked.array[1] - expected_mask[0, 0].numpy()).max() < 1e-5
    assert np.abs(level_masked.array[0] - expected[0, 0].numpy()).max() < 1e-5


def test_a_level_far_coarser_than_its_source_reads_bounded_blocks():
    """A level 64 times coarser than its image (a 4096^3 volume at its 64^3 level) is built by blocks whose reads stay
    within four blocks and the kernels' tails a side, and is FireANTs' level."""
    from fireants.utils.imageutils import downsample

    array = np.random.default_rng(0).random((1024, 1024)).astype(np.float32)
    size = [16, 16]
    gaussians = fireants_gaussians(array.shape, size, "cpu")
    tail = (int(gaussians[0].numel()) - 1) // 2
    source, spans = ArraySource(array[None], "cpu"), []
    read = source.read

    def recorded(start, stop):
        spans.append(max(b - a for a, b in zip(start, stop, strict=True)))
        return read(start, stop)

    source.read = recorded  # type: ignore[method-assign]
    level = build_level(source, size, gaussians, block=64)
    assert max(spans) <= 4 * 64 + 2 * tail + 3
    expected = downsample(torch.from_numpy(array)[None, None], size=size, mode="bilinear", gaussians=gaussians)
    assert np.abs(level.array - expected[0].numpy()).max() < 1e-5


# Starting off the voxel centres: at an exact centre the trilinear interpolation's derivative is one-sided and float
# round-off decides which side, so two exact runs would take different gradients there.
OFF_GRID = torch.tensor([[0.3, -0.2, 0.1]])


@SHAPES
@pytest.mark.parametrize("scales,iterations,blur", [([1], [6], True), ([2, 1], [4, 3], True), ([2, 1], [4, 3], False)])
def test_every_box_reproduces_the_whole_image_rigid(scales, iterations, blur, size):
    fixed, moving = texture(size=size), texture((1.5, -1.0, 0.5), size)
    settings = {
        "scales": scales,
        "iterations": iterations,
        "blur": blur,  # unsmoothed, FireANTs interpolates the fixed image onto a coarse level and keeps the moving
        "loss_type": "mse",
        "optimizer": "Adam",
        "optimizer_lr": 0.02,
        "optimizer_params": {"eps": 1.0},  # the step proportional to the gradient: float noise does not flip it
        "init_translation": OFF_GRID[:, : len(size)],
    }
    from fireants.registration.rigid import RigidRegistration

    whole = whole_registration(RigidRegistration, fixed, moving, **settings)
    whole.optimize()
    sampled = sampled_registration(sampled_rigid_class(), fixed, moving, box=16, boxes=10_000, **settings)
    sampled.optimize()
    # Unsmoothed, a level is FireANTs' interpolation of the image, whose coordinates F.interpolate rounds in float32:
    # 2e-5 apart after the iterations, where another pyramid moves the matrix by 1e-2.
    assert torch.allclose(whole.get_rigid_matrix(), sampled.get_rigid_matrix(), atol=1e-5 if blur else 5e-5)


@SHAPES
@pytest.mark.parametrize("loss", ["mse", "cc", "mi"])
def test_every_box_reproduces_the_whole_image_affine(loss, size):
    fixed, moving = texture(size=size), texture((1.5, -1.0, 0.5), size)
    dims = len(size)
    start = torch.eye(dims, dims + 1).unsqueeze(0)
    start[:, :, dims] = OFF_GRID[:, :dims]
    settings = {
        "scales": [2, 1],
        "iterations": [4, 3],
        "loss_type": loss,
        "optimizer": "Adam",
        "optimizer_lr": 0.01,
        "optimizer_params": {"eps": 1.0},
        "init_rigid": start,
    }
    from fireants.registration.affine import AffineRegistration

    whole = whole_registration(AffineRegistration, fixed, moving, **settings)
    whole.optimize()
    # Packets of two boxes: the level is never scored whole.
    sampled = sampled_registration(
        sampled_affine_class(), fixed, moving, box=16, boxes=10_000, together=2 * 16**dims, **settings
    )
    sampled.optimize()
    assert torch.allclose(whole.get_affine_matrix(), sampled.get_affine_matrix(), atol=1e-5)


@pytest.mark.parametrize("loss", ["mse", "cc"])
def test_a_drawn_box_gives_an_unbiased_gradient(loss):
    """Over every box a draw of one may return, the mean of the gradients is the whole level's: boxes of several
    sizes, and a loss that sums over the voxels."""
    fixed, moving = texture(), texture((1.5, -1.0, 0.5))
    start = torch.eye(3, 4).unsqueeze(0)
    start[:, :, 3] = OFF_GRID
    settings = {"scales": [1], "iterations": [1], "loss_type": loss, "optimizer_lr": 0.0, "init_rigid": start}

    def gradient(only: int | None) -> torch.Tensor:
        boxes = 10_000 if only is None else 1
        registration = sampled_registration(sampled_affine_class(), fixed, moving, box=16, boxes=boxes, **settings)
        if only is not None:
            registration._draws = SimpleNamespace(sample=lambda population, count: [only])
        registration.optimize()
        return torch.cat([p.grad.flatten() for group in registration.optimizer.param_groups for p in group["params"]])

    whole = gradient(None)
    draws = torch.stack([gradient(tile) for tile in range(len(tiles(SIZE, 16)))])
    assert len({tuple(b - a for a, b in zip(*tile, strict=True)) for tile in tiles(SIZE, 16)}) > 1
    assert torch.allclose(draws.mean(0), whole, rtol=1e-3, atol=1e-6 * float(whole.abs().max()))


@pytest.mark.parametrize("loss", ["mse", "cc", "mi"])
def test_levels_on_the_device_score_their_boxes_together(loss, monkeypatch):
    """A draw of boxes of several sizes gives the same steps scored one box at a time, from files, and every box of
    one size together, from levels on the device."""
    import impact_reg_konfai.ooc as ooc

    fixed, moving = texture(), texture((1.5, -1.0, 0.5))
    settings = {"scales": [2, 1], "iterations": [4, 3], "loss_type": loss, "optimizer_lr": 0.01}
    settings["optimizer_params"] = {"eps": 1.0}  # steps that follow the gradient, not its sign where it is near zero
    matrices = []
    for free in (0, 2**40):
        monkeypatch.setattr(ooc, "free_bytes", lambda device, free=free: free)
        start = torch.eye(3, 4).unsqueeze(0)  # FireANTs steps the tensor it is handed
        start[:, :, 3] = OFF_GRID
        sampled = sampled_registration(
            sampled_affine_class(), fixed, moving, box=8, boxes=40, seed=3, init_rigid=start, **settings
        )
        sampled.optimize()
        matrices.append(sampled.get_affine_matrix())
    assert torch.allclose(*matrices, atol=1e-4)  # of matrices stepped by 1e-2; the histogram sums in another order


def test_more_iterations_bring_drawn_boxes_to_the_whole_image_run():
    """Four boxes of the thirty drawn anew at each iteration, a noisy estimate of the whole image's gradient: under
    the falling step and the mean of the scale's second half, the run nears the whole-image run as its iterations
    grow."""
    from fireants.registration.rigid import RigidRegistration

    fixed, moving = texture(), texture((1.5, -1.0, 0.5))
    centre = torch.tensor([(n - 1) / 2 for n in reversed(SIZE)], dtype=torch.float32)

    def gap(iterations: int) -> float:
        settings = {
            "scales": [2, 1],
            "iterations": [60, iterations],
            "loss_type": "mse",
            "optimizer": "Adam",
            "optimizer_lr": 0.05,
            "init_translation": torch.zeros((1, 3)),
            "tolerance": float("inf"),  # every iteration is run
        }
        whole = whole_registration(RigidRegistration, fixed, moving, **settings)
        whole.optimize()
        sampled = sampled_registration(sampled_rigid_class(), fixed, moving, box=16, boxes=4, **settings)
        sampled.optimize()
        matrices = [registration.get_rigid_matrix()[0].detach() for registration in (sampled, whole)]
        moved = [matrix[:3, :3] @ centre + matrix[:3, 3] for matrix in matrices]
        return float((moved[0] - moved[1]).abs().max())

    near, nearer = gap(40), gap(640)
    assert near < 0.25
    assert nearer < 0.05 and nearer < near / 2


def test_the_information_of_every_point_is_fireants_loss():
    from fireants.losses.mi import GlobalMutualInformationLoss
    from impact_reg_konfai.models.fireants_sampled import _information, _joint_counts

    generator = torch.Generator().manual_seed(0)
    fixed = torch.rand((1, 1, 500, 1, 1), generator=generator)
    moved = (0.7 * fixed + 0.3 * torch.rand(fixed.shape, generator=generator)).requires_grad_()
    loss = GlobalMutualInformationLoss()
    expected = loss(moved, fixed)
    counts, points = _joint_counts(loss, moved, fixed)
    ours = _information(loss, counts / points)
    assert points == 500 and torch.allclose(ours, expected, atol=1e-6)
    gradients = [torch.autograd.grad(value, moved)[0] for value in (ours, expected)]
    assert torch.allclose(*gradients, atol=1e-7, rtol=1e-4)


@pytest.mark.parametrize("loss", ["mse", "cc", "mi"])
def test_a_draw_scored_by_packets_is_the_draw_scored_at_once(loss):
    """Packets of one box against one packet for the whole draw: the same steps, the level never held whole."""
    fixed, moving = texture(), texture((1.5, -1.0, 0.5))
    settings = {"scales": [2, 1], "iterations": [4, 3], "loss_type": loss, "optimizer_lr": 0.01}
    settings["optimizer_params"] = {"eps": 1.0}  # steps that follow the gradient, not its sign where it is near zero
    matrices = []
    for together in (8**3, 10**9):
        start = torch.eye(3, 4).unsqueeze(0)  # FireANTs steps the tensor it is handed
        start[:, :, 3] = OFF_GRID
        sampled = sampled_registration(
            sampled_affine_class(),
            fixed,
            moving,
            box=8,
            boxes=40,
            seed=3,
            together=together,
            init_rigid=start,
            **settings,
        )
        sampled.optimize()
        matrices.append(sampled.get_affine_matrix())
    assert torch.allclose(*matrices, atol=1e-4)


def test_a_draw_the_masks_miss_leaves_the_tracked_histogram_whole():
    """The level's joint histogram, tracked from draw to draw, sums to one: a draw whose masks keep no point moves
    it nowhere."""
    fixed, moving = texture(), texture((1.0, 0.0, 0.0))
    headers = [HeaderImages(image_of(array), 2, "cpu") for array in (fixed, moving)]
    registration = sampled_affine_class()(
        scales=[1], iterations=[1], fixed_images=headers[0], moving_images=headers[1], loss_type="masked_mi"
    )
    registration._histogram, registration._keeps_draw = None, True
    image = torch.from_numpy(fixed[None, None, :8, :8, :8].copy())
    for mask in (torch.ones_like(image), torch.zeros_like(image)):
        packet = torch.cat([image, mask], dim=1)
        registration._information_step(lambda packet=packet: iter([(packet, packet, None)]), 0.5)
        assert float(registration._histogram.sum()) == pytest.approx(1.0)


def test_mutual_information_runs_on_the_sampled_points():
    fixed, moving = texture(), texture((1.0, 0.0, 0.0))
    matrix = sampled_linear_alignment(
        image_of(fixed),
        image_of(moving),
        ArraySource(fixed[None], "cpu"),
        ArraySource(moving[None], "cpu"),
        scales=[2, 1],
        iterations=[20, 10],
        loss_type="mi",
        optimizer_lr=0.05,
        translation_lr=0.05,
        cc_kernel_size=3,
        init_translation=torch.zeros((1, 3)),
        device="cpu",
        box=16,
        boxes=6,
        seed=1,
        affine=False,
    )
    assert matrix.shape == (1, 4, 4)
    assert torch.isfinite(matrix).all()
    centre = torch.tensor([(n - 1) / 2 for n in reversed(SIZE)], dtype=torch.float32)
    mapped = matrix[0, :3, :3] @ centre + matrix[0, :3, 3]
    assert (mapped - centre)[0].item() == pytest.approx(-1.0, abs=0.4)


class _Pointwise(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        nb_layer: torch.Tensor,
        stats: torch.Tensor | None = None,
        direction: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        return [x]


@pytest.mark.parametrize("distance", ["LNCC", "NCC"])
@pytest.mark.parametrize("masked", [False, True])
def test_a_level_s_normalisation_is_its_mean_however_its_boxes_cut_it(tmp_path, masked, distance):
    """The deformable stage latches a layer's normalisation on every box of the level, each read with the window's
    halo: the halos are read, never counted, and a masked loss weighs each box by the voxels both masks keep. An NCC,
    no mean of its voxels' values, is the level's from the sums the boxes add up (``pool``)."""
    from impact_reg_konfai.models import fireants as engine

    path = tmp_path / "features.pt"
    torch.jit.script(_Pointwise()).save(str(path))
    spec = engine.ModelSpec(ref=str(path), distance=distance)
    loss = engine.ImpactFeatureLoss([[spec]], "Static", True, 5, 0, 0, False, masked=masked)
    loss._channels, loss._level = [[2]], 0
    generator = torch.Generator().manual_seed(0)
    shape = (20, 24, 28)
    moved, fixed = (torch.rand((1, 3, *shape), generator=generator) for _ in range(2))
    for volume, cut in ((moved, 9), (fixed, 13)):
        volume[:, -1] = 0.0
        volume[:, -1, :, :cut] = 1.0  # masks that keep part of the level

    def factors(box: int) -> list[float]:
        pieces = []
        for tile in tiles(shape, box):
            low, high, core = box_region(tile, 4, shape)
            region = torch.zeros((1, 1, *(b - a for a, b in zip(low, high, strict=True))))
            region[(slice(None), slice(None), *core)] = 1
            crop = (slice(None), slice(None), *(slice(a, b) for a, b in zip(low, high, strict=True)))
            pieces.append((moved[crop], fixed[crop], region))
        if loss.pools:
            loss.pool(pieces)
        loss.latch(pieces)
        return loss._factors

    assert np.allclose(factors(64), factors(7), rtol=1e-5)


@pytest.mark.parametrize("masked", [False, True])
def test_an_ncc_layer_by_boxes_has_the_level_s_gradient(tmp_path, masked):
    """Evaluated box by box, each box weighed as the deformable stage weighs it (its voxels over the level's), an NCC
    layer has the level's gradient wherever a box holds its voxels: the boxes' sums (``pool``) give each the level's
    value and derivative."""
    from impact_reg_konfai.models import fireants as engine

    path = tmp_path / "features.pt"
    torch.jit.script(_Pointwise()).save(str(path))
    spec = engine.ModelSpec(ref=str(path), distance="NCC")
    loss = engine.ImpactFeatureLoss([[spec]], "Static", False, 5, 0, 0, False, masked=masked)
    loss._channels, loss._level = [[2]], 0
    generator = torch.Generator().manual_seed(1)
    shape = (20, 24, 28)
    moved, fixed = (torch.rand((1, 3, *shape), generator=generator) for _ in range(2))
    for volume, cut in ((moved, 9), (fixed, 13)):
        volume[:, -1] = 0.0
        volume[:, -1, :, :cut] = 1.0
    if not masked:
        moved, fixed = moved[:, :2], fixed[:, :2]
    whole = moved.clone().requires_grad_()
    (expected,) = torch.autograd.grad(loss(whole, fixed), whole)
    boxes = []
    for tile in tiles(shape, 9):
        low, high, core = box_region(tile, 2, shape)
        region = torch.zeros((1, 1, *(b - a for a, b in zip(low, high, strict=True))))
        region[(slice(None), slice(None), *core)] = 1
        crop = (slice(None), slice(None), *(slice(a, b) for a, b in zip(low, high, strict=True)))
        boxes.append((low, core, crop, region))
    loss.pool([(moved[crop], fixed[crop], region) for _, _, crop, region in boxes])
    support = sum(
        float(((moved[crop][:, -1:] * fixed[crop][:, -1:] >= 0.5) * region).sum()) for *_, crop, region in boxes
    )
    gathered = torch.zeros_like(moved)
    for low, core, crop, _ in boxes:
        box = moved[crop].clone().requires_grad_()
        value = loss(box, fixed[crop])
        if masked:  # the stage's masked weight: the box's support over the level's
            weight = (box[:, -1:] * fixed[crop][:, -1:] >= 0.5).sum()
            share = value * weight / support
        else:
            share = value * math.prod(box.shape[2:]) / math.prod(shape)
        (gradient,) = torch.autograd.grad(share, box)
        own = tuple(slice(lo + c.start, lo + c.stop) for lo, c in zip(low, core, strict=True))
        gathered[(slice(None), slice(None), *own)] = gradient[(slice(None), slice(None), *core)]
    channels = slice(None, 2)  # the features: a mask channel holds no gradient of the distance
    assert torch.allclose(gathered[:, channels], expected[:, channels], atol=1e-6)


class _Strided(torch.nn.Module):
    """A convolution, then its average over 2^3 voxels: a layer at the image's resolution and one at half of it. With
    ``flipped``, on the image read backwards along its last axis, as TotalSegmentator reorients what it is handed,
    so that the coarse layer's cells are anchored at the end of that axis."""

    def __init__(self, flipped: bool = False, pool: int = 2) -> None:
        super().__init__()
        self.flipped, self.pool = flipped, pool
        self.conv = torch.nn.Conv3d(1, 4, 3, padding=1)
        for parameter in self.conv.parameters():
            torch.nn.init.normal_(parameter, generator=torch.Generator().manual_seed(1))

    def forward(
        self,
        x: torch.Tensor,
        nb_layer: torch.Tensor,
        stats: torch.Tensor | None = None,
        direction: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if self.flipped:
            y = self.conv(x.flip(-1))
            return [y.flip(-1), torch.nn.functional.avg_pool3d(y, self.pool).flip(-1)]
        y = self.conv(x)
        return [y, torch.nn.functional.avg_pool3d(y, self.pool)]


@pytest.mark.parametrize("pool", [2, 3])
@pytest.mark.parametrize("flipped", [False, True], ids=["as-is", "flipped"])
@pytest.mark.parametrize("multiple", [0, 2], ids=["any-size", "multiple"])
@pytest.mark.parametrize("layers", ["10", "01"], ids=["full", "strided"])
@pytest.mark.parametrize("shape", [(20, 24, 28), (21, 25, 27)])
def test_static_features_by_blocks_are_the_whole_image_s(tmp_path, shape, layers, multiple, flipped, pool):
    """Blocks read with half the receptive field around them, predicted as a segmentation is, give a layer as the
    whole image does, at its resolution or at a ``pool``-th of it (a power of two or not), and the PCA fitted on it
    its components: an image that is no whole number of the layer's cells, or of the input multiple the network pads
    it to, and a network that anchors its cells at the end of an axis, included."""
    from impact_reg_konfai.models import fireants as engine
    from impact_reg_konfai.models.fireants_sampled import static_feature_sources

    path = tmp_path / "strided.pt"
    torch.jit.script(_Strided(flipped, pool)).save(str(path))
    spec = engine.ModelSpec(ref=str(path), layers_mask=layers, pca=2, distance="LNCC", feature_normalization="l2")
    loss = engine.ImpactFeatureLoss([[spec]], "Static", True, 5, 0, 0, False)
    loss.cores[0].model.fov, loss.cores[0].model.multiple = [3, 4], multiple
    crop = tuple(slice(0, extent) for extent in shape)
    fixed, moving = texture()[crop], texture((1.5, -1.0, 0.5))[crop]
    whole = loss.extract(*(torch.from_numpy(np.ascontiguousarray(a))[None, None] for a in (fixed, moving)), 0)
    sources = static_feature_sources(
        loss,
        ArraySource(np.ascontiguousarray(fixed)[None], "cpu"),
        ArraySource(np.ascontiguousarray(moving)[None], "cpu"),
        tmp_path,
        block=8,
    )
    blocks = [torch.from_numpy(np.asarray(source.array))[None] for source in sources]
    # A component's sign is free, and the same for both images: the fixed one's basis projects them.
    signs = torch.sign((whole[0] * blocks[0]).sum((0, 2, 3, 4))).view(1, -1, 1, 1, 1)
    for reference, found in zip(whole, blocks, strict=True):
        assert torch.allclose(found * signs, reference, atol=1e-5)


@pytest.mark.parametrize("layers", ["10", "01"], ids=["full", "strided"])
def test_static_features_at_a_voxel_size_by_blocks_are_the_whole_image_s(tmp_path, layers):
    """A model with a voxel_size reads the image resampled at it, by blocks of that grid, and its features come back
    onto the image's voxels: the whole image's, the moving image on its own extent."""
    from impact_reg_konfai.models import fireants as engine
    from impact_reg_konfai.models.fireants_sampled import static_feature_sources

    path = tmp_path / "strided.pt"
    torch.jit.script(_Strided()).save(str(path))
    spec = engine.ModelSpec(
        ref=str(path),
        layers_mask=layers,
        pca=2,
        distance="LNCC",
        feature_normalization="l2",
        voxel_size=[1.4, 0.7, 1.9],
    )
    loss = engine.ImpactFeatureLoss([[spec]], "Static", True, 5, 0, 0, False)
    loss.cores[0].model.fov, loss.cores[0].model.multiple = [3, 4], 2
    fixed, moving = texture()[:21, :25, :27], texture((1.5, -1.0, 0.5))[:20, :24, :26]
    extents = ([27.0, 25.0, 21.0], [26.0, 24.0, 20.0])  # mm along L, P, S: one voxel a mm
    whole = loss.extract(*(torch.from_numpy(np.ascontiguousarray(a))[None, None] for a in (fixed, moving)), 0, extents)
    sources = static_feature_sources(
        loss,
        ArraySource(np.ascontiguousarray(fixed)[None], "cpu"),
        ArraySource(np.ascontiguousarray(moving)[None], "cpu"),
        tmp_path,
        block=8,
        extents=extents,
    )
    blocks = [torch.from_numpy(np.asarray(source.array))[None] for source in sources]
    signs = torch.sign((whole[0] * blocks[0]).sum((0, 2, 3, 4))).view(1, -1, 1, 1, 1)
    for reference, found in zip(whole, blocks, strict=True):
        assert found.shape == reference.shape and torch.allclose(found * signs, reference, atol=1e-5)


class _Counted(ArraySource):
    """An image source that keeps the size of every box it was asked."""

    def __init__(self, array):
        super().__init__(array, "cpu")
        self.reads = []

    def read(self, start, stop):
        self.reads.append(math.prod(b - a for a, b in zip(start, stop, strict=True)))
        return super().read(start, stop)


@pytest.mark.parametrize(
    "shape, size", [((97, 83, 71), (40, 33, 30)), ((40, 31, 29), (77, 70, 61)), ((61, 47), (13, 50))]
)
def test_a_resampled_box_is_the_whole_resampled_image_s_and_reads_at_most_2_to_the_dims_times_it(shape, size):
    """A box of an image resampled at a model's voxel size is the whole resampled image's, read in runs of the planes
    it interpolates from: downsampling reads no more than 2**dims times the box at once."""
    from konfai.metric.measure.impact import resampled

    image = torch.rand(2, *shape, generator=torch.Generator().manual_seed(0))
    source, whole = _Counted(image.numpy()), resampled(image[None], size)
    for box in [(0, s) for s in size], [(s // 3, s - 1) for s in size]:
        source.reads.clear()
        found = ResampledSource(source, size).read([a for a, _ in box], [b for _, b in box])
        assert torch.allclose(found, whole[(slice(None), slice(None), *(slice(a, b) for a, b in box))], atol=1e-5)
        assert max(source.reads) <= 2 ** len(size) * math.prod(b - a for a, b in box)


class _Dilated(torch.nn.Module):
    """A convolution dilated by 6: a receptive field of 13 voxels, wider than half a block of 8."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 4, 3, padding=6, dilation=6)
        for parameter in self.conv.parameters():
            torch.nn.init.normal_(parameter, generator=torch.Generator().manual_seed(1))

    def forward(
        self,
        x: torch.Tensor,
        nb_layer: torch.Tensor,
        stats: torch.Tensor | None = None,
        direction: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        return [self.conv(x)]


def test_static_blocks_read_the_whole_receptive_field_around_them(tmp_path):
    # A layer reads half its receptive field around a voxel, here wider than half a block: blocks read all of it, so
    # their features are the whole image's.
    from impact_reg_konfai.models import fireants as engine
    from impact_reg_konfai.models.fireants_sampled import static_feature_sources

    path = tmp_path / "dilated.pt"
    torch.jit.script(_Dilated()).save(str(path))
    spec = engine.ModelSpec(ref=str(path), layers_mask="1", pca=0, distance="LNCC", feature_normalization="l2")
    loss = engine.ImpactFeatureLoss([[spec]], "Static", True, 5, 0, 0, False)
    loss.cores[0].model.fov = [13]
    fixed, moving = texture()[:21, :25, :27], texture((1.5, -1.0, 0.5))[:21, :25, :27]
    whole = loss.extract(*(torch.from_numpy(np.ascontiguousarray(a))[None, None] for a in (fixed, moving)), 0)
    sources = static_feature_sources(
        loss,
        ArraySource(np.ascontiguousarray(fixed)[None], "cpu"),
        ArraySource(np.ascontiguousarray(moving)[None], "cpu"),
        tmp_path,
        block=8,
    )
    for reference, source in zip(whole, sources, strict=True):
        assert torch.allclose(torch.from_numpy(np.asarray(source.array))[None], reference, atol=1e-5)


class _Convolved(torch.nn.Module):
    """A convolution of 3 voxels a side, 3-D or 2-D: a layer at the image's resolution."""

    def __init__(self, dims: int = 3) -> None:
        super().__init__()
        self.conv = (torch.nn.Conv3d if dims == 3 else torch.nn.Conv2d)(1, 4, 3, padding=1)
        for parameter in self.conv.parameters():
            torch.nn.init.normal_(parameter, generator=torch.Generator().manual_seed(1))

    def forward(
        self,
        x: torch.Tensor,
        nb_layer: torch.Tensor,
        stats: torch.Tensor | None = None,
        direction: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        return [self.conv(x)]


@pytest.mark.parametrize("dims", [3, 2], ids=["3d-network", "2d-network"])
def test_static_features_of_a_plane_by_blocks_are_the_whole_plane_s(tmp_path, dims):
    """A 2-D pair is handed to a 3-D network as volumes one voxel thick, as a run on the whole pair lifts it, and to a
    2-D network as planes: block by block either gives the whole plane's features, and the PCA fitted on them its
    components."""
    from impact_reg_konfai.models import fireants as engine
    from impact_reg_konfai.models.fireants_sampled import static_feature_sources

    path = tmp_path / "convolved.pt"
    torch.jit.script(_Convolved()).save(str(path))
    spec = engine.ModelSpec(ref=str(path), layers_mask="1", pca=2, distance="LNCC", feature_normalization="l2")
    loss = engine.ImpactFeatureLoss([[spec]], "Static", True, 5, 0, 0, False)
    core = loss.cores[0]
    if dims == 2:  # a local file is a 3-D model to the registry: the 2-D network handed in, run as it is
        core.model.model, core.model.dim, core.dimension = torch.jit.script(_Convolved(2)), 2, 2
    core.model.fov, core.model.planes = [3], True
    fixed, moving = texture(size=PLANE)[:21, :25], texture((1.5, -1.0, 0.5), PLANE)[:21, :25]
    whole = loss.extract(*(torch.from_numpy(np.ascontiguousarray(a))[None, None] for a in (fixed, moving)), 0)
    sources = static_feature_sources(
        loss,
        ArraySource(np.ascontiguousarray(fixed)[None], "cpu"),
        ArraySource(np.ascontiguousarray(moving)[None], "cpu"),
        tmp_path,
        block=8,
    )
    blocks = [torch.from_numpy(np.asarray(source.array))[None] for source in sources]
    signs = torch.sign((whole[0] * blocks[0]).sum((0, 2, 3))).view(1, -1, 1, 1)
    for reference, found in zip(whole, blocks, strict=True):
        assert found.shape == reference.shape and torch.allclose(found * signs, reference, atol=1e-5)


@pytest.mark.parametrize("voxel_size", [None, [1.4, 0.7, 1.9]], ids=["native", "voxel_size"])
def test_static_features_of_a_volume_swept_by_a_2d_network_are_the_whole_volume_s(tmp_path, voxel_size):
    """A 2-D network on a volume reads each slice alone, as the whole run sweeps them: predicted by slabs of whole
    slices, of the volume resampled at its voxel_size when it has one, it gives the whole volume's features."""
    from impact_reg_konfai.models import fireants as engine
    from impact_reg_konfai.models.fireants_sampled import static_feature_sources

    path = tmp_path / "convolved.pt"
    torch.jit.script(_Convolved()).save(str(path))
    spec = engine.ModelSpec(
        ref=str(path), layers_mask="1", pca=2, distance="LNCC", feature_normalization="l2", voxel_size=voxel_size
    )
    loss = engine.ImpactFeatureLoss([[spec]], "Static", True, 5, 0, 0, False)
    core = loss.cores[0]  # a local file is a 3-D model to the registry: the 2-D network handed in
    core.model.model, core.model.dim, core.dimension, core.model.fov = torch.jit.script(_Convolved(2)), 2, 2, [3]
    fixed, moving = texture()[:21, :25, :27], texture((1.5, -1.0, 0.5))[:21, :25, :27]
    extents = ([27.0, 25.0, 21.0], [27.0, 25.0, 21.0])  # mm along L, P, S: one voxel a mm
    whole = loss.extract(*(torch.from_numpy(np.ascontiguousarray(a))[None, None] for a in (fixed, moving)), 0, extents)
    sources = static_feature_sources(
        loss,
        ArraySource(np.ascontiguousarray(fixed)[None], "cpu"),
        ArraySource(np.ascontiguousarray(moving)[None], "cpu"),
        tmp_path,
        block=8,
        extents=extents,
    )
    blocks = [torch.from_numpy(np.asarray(source.array))[None] for source in sources]
    signs = torch.sign((whole[0] * blocks[0]).sum((0, 2, 3, 4))).view(1, -1, 1, 1, 1)
    for reference, found in zip(whole, blocks, strict=True):
        assert found.shape == reference.shape and torch.allclose(found * signs, reference, atol=1e-5)


def test_a_2d_jacobian_model_at_a_voxel_size_reads_a_halo_for_every_axis_it_sweeps(tmp_path, monkeypatch):
    """A 2-D network sweeps a volume along any of its axes: its two voxel_size values fall on any two of the pair's,
    and the halo the deformable stage reads around a box is the widest they give."""
    from impact_reg_konfai.models import fireants as engine
    from impact_reg_konfai.models import fireants_sampled

    spacing = (1.0, 2.0, 0.5)
    paths = []
    for name, array in (("fixed", texture()), ("moving", texture((1.0, 0.0, 0.0)))):
        image = image_of(array)
        image.SetSpacing(spacing)
        paths.append(tmp_path / f"{name}.mha")
        sitk.WriteImage(image, str(paths[-1]))
    path = tmp_path / "convolved.pt"
    torch.jit.script(_Convolved()).save(str(path))
    settings = {"scales": [1], "affine_iterations": [1], "deformable_iterations": [1], "cc_kernel": 3}
    settings |= {"affine_metric": "mse", "affine_lr": 0.01, "moments_init": "none", "linear_method": "none"}
    settings |= {"deformable_lr": 0.25, "smooth_warp_sigma": 0.5, "smooth_grad_sigma": 1.0, "seed": 0}
    spec = engine.ModelSpec(ref=str(path), layers_mask="1", distance="L2", voxel_size=[1.5, 3.0, 1.0])
    fireants = engine.FireANTsEngine(
        **settings, deformable_method="greedy", deformable_metric="impact", mode="Jacobian", impact_levels=[[spec]]
    )
    core = fireants._impact_loss().cores[0]  # a local file is a 3-D model to the registry: the 2-D network handed in
    core.model.model, core.model.dim, core.dimension = torch.jit.script(_Convolved(2)), 2, 2
    core.model.fov, core.model.voxel_size = [3], [1.5, 3.0]
    halos = []

    class Stage:
        def __init__(self, *args, loss_halo, **kwargs) -> None:
            halos.append(loss_halo)
            raise InterruptedError

    monkeypatch.setattr(fireants_sampled, "sampled_deformable_class", lambda: Stage)
    with pytest.raises(InterruptedError):
        fireants.register_sampled(*paths, None, None, -1, tmp_path / "work", tmp_path / "field.mha", 16)
    reach = core.model.receptive_field - 1 + 2 * core.model.block_stride("cpu")
    assert halos == [math.ceil(reach * max(core.model.voxel_size) / min(spacing))]


def test_a_rescaled_source_and_a_mask_of_ones_read_by_boxes():
    source = RescaledSource(ArraySource(np.arange(27, dtype=np.float32).reshape(1, 3, 3, 3), "cpu"), 5.0, 20.0, 0.5)
    box = source.read([0, 0, 0], [3, 3, 3])
    assert box.min() == 0.0 and box.max() == 0.5 and torch.isclose(box[0, 0, 1, 1, 1], torch.tensor((13 - 5) / 30))
    assert OnesSource((4, 5, 6), "cpu").read([1, 0, 2], [3, 5, 6]).shape == (1, 1, 2, 5, 4)


#: A warp the deformable stages start from, in voxels: at zero the fixed image is read exactly at its voxels, where
#: the derivative of the interpolation is one-sided and its side the rounding's.
def off_centres(size=SIZE) -> torch.Tensor:
    return torch.tensor([0.31, -0.23, 0.17][: len(size)]) * 2 / (max(size) - 1)


def start_affine(size=SIZE) -> torch.Tensor:
    """The affine the deformable stages start from: a translation of a few tenths of a voxel."""
    dims = len(size)
    affine = torch.eye(dims + 1)[None].clone()
    affine[0, :dims, dims] = torch.tensor([0.4, -0.3, 0.2][:dims])
    return affine


OFF_CENTRES = off_centres()
AFFINE = start_affine()
DEFORMABLE = {"scales": [2, 1], "iterations": [4, 3], "cc_kernel_size": 5, "optimizer_lr": 0.25}


def deformable_pair(tmp_path, symmetric: bool, loss: str, samples: int | None = None, mask=None, size=SIZE):
    """FireANTs' compositive greedy or SyN on the whole pair, and the same stage over boxes of 16 voxels. With
    ``mask``, FireANTs' masked mode: that mask on the fixed image, ones on the moving one."""
    from fireants.io import BatchedImages, Image
    from fireants.io.imagemask import apply_mask_to_image
    from fireants.registration.greedy import GreedyRegistration
    from fireants.registration.syn import SyNRegistration

    fixed, moving = texture(size=size), texture((1.3, -0.8, 0.6), size)
    images = {}
    if mask is not None:
        loss = f"masked_{loss}"
        pair = []
        for array, region in ((fixed, mask), (moving, np.ones_like(mask))):
            image = Image(image_of(array), device="cpu")
            pair.append(BatchedImages([apply_mask_to_image(image, Image(image_of(region), device="cpu"))]))
        images = {"fixed_images": pair[0], "moving_images": pair[1]}
    cls = SyNRegistration if symmetric else GreedyRegistration
    whole = (
        cls(
            progress_bar=False,
            **images,
            loss_type=loss,
            deformation_type="compositive",
            optimizer="Adam",
            init_affine=start_affine(size),
            **DEFORMABLE,
        )
        if images
        else whole_registration(
            cls,
            fixed,
            moving,
            loss_type=loss,
            deformation_type="compositive",
            optimizer="Adam",
            init_affine=start_affine(size),
            **DEFORMABLE,
        )
    )
    whole_warps = {"fwd": whole.fwd_warp, "rev": whole.rev_warp} if symmetric else {"fwd": whole.warp}
    for warp in whole_warps.values():
        warp.warp.data += off_centres(size)
    whole.optimize()
    sources = [ArraySource(fixed[None], "cpu"), ArraySource(moving[None], "cpu")]
    if mask is not None:
        sources = [
            StackedSource([sources[0], ArraySource(mask[None], "cpu")]),
            StackedSource([sources[1], OnesSource(size, "cpu")]),
        ]
    channels = 2 if mask is not None else 1
    sampled = sampled_deformable_class()(
        fixed_images=HeaderImages(image_of(fixed), channels, "cpu"),
        moving_images=HeaderImages(image_of(moving), channels, "cpu"),
        fixed=sources[0],
        moving=sources[1],
        scratch=tmp_path,
        symmetric=symmetric,
        loss_type=loss,
        init_affine=start_affine(size),
        box=16,
        samples=samples,
        **DEFORMABLE,
    )
    sampled._set_grid(fireants_level_sizes(size, DEFORMABLE["scales"][0]))
    for warp in sampled._warps.values():
        warp.arrays["warp"][:] = off_centres(size).numpy().reshape(-1, *[1] * len(size))
    sampled.optimize()
    return whole, {name: warp.warp.detach() for name, warp in whole_warps.items()}, sampled


def warp_gap(sampled, whole_warps, size=SIZE) -> float:
    """The largest difference, in voxels, between the warps of the two runs."""
    return (
        max(
            float(
                (torch.from_numpy(np.asarray(sampled._warps[name].arrays["warp"])).movedim(0, -1)[None] - warp)
                .abs()
                .max()
            )
            for name, warp in whole_warps.items()
        )
        * 0.5
        * (max(size) - 1)
    )


@pytest.mark.parametrize("symmetric,loss", [(False, "mse"), (False, "cc"), (True, "cc"), (False, "mi"), (True, "mi")])
@SHAPES
def test_every_box_reproduces_the_whole_image_deformable(tmp_path, symmetric, loss, size):
    whole, whole_warps, sampled = deformable_pair(tmp_path, symmetric, loss, size=size)
    assert warp_gap(sampled, whole_warps, size) < 1e-3
    if symmetric:  # SyN's reads the forward warp through FireANTs' own inverse of the reverse one
        gap, tolerance = syn_field_gap(tmp_path, whole, sampled, size)
        assert gap < tolerance
    else:  # the field FireANTs writes
        field = np.zeros((*size, len(size)), dtype=np.float32)
        sampled.write_field(field, block=20)
        assert np.abs(field - written_field(tmp_path, whole, "whole.nii")).max() < 1e-3
    sampled.close()
    assert not list(tmp_path.glob("*.npy"))


def written_field(tmp_path, whole, name: str) -> np.ndarray:
    """The field FireANTs writes for ``whole``, as the whole route has it written: SyN's inverse brought onto the
    fixed grid (``_inverse_on``)."""
    from fireants.registration import syn
    from impact_reg_konfai.models.fireants import _inverse_on

    vectors = functools.partial(sitk.GetImageFromArray, isVector=True)  # a plane's (H, W, 2) read as vectors
    inverse = _inverse_on(tuple(whole.fixed_images.shape[2:]))
    with (
        mock.patch.object(sitk, "GetImageFromArray", vectors),
        mock.patch.object(syn, "compositive_warp_inverse", inverse),
    ):
        whole.save_as_ants_transforms(str(tmp_path / name))
    return sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / name)))


def syn_field_gap(tmp_path, whole, sampled, size) -> tuple[float, float]:
    """How far the boxes' SyN field lies from FireANTs' field of the same warps, and how far it may: twice what
    FireANTs' own field moves when its reverse warp moves by 1e-7 of it, its inverse being an optimisation (a
    thousandth of a voxel at least)."""
    for name, warp in (("fwd", whole.fwd_warp), ("rev", whole.rev_warp)):
        warp.warp.data.copy_(torch.from_numpy(np.asarray(sampled._warps[name].arrays["warp"])).movedim(0, -1)[None])
    reverse = whole.rev_warp.warp.data
    noise = torch.randn(reverse.shape, generator=torch.Generator().manual_seed(0)) * 1e-7 * reverse.abs().max()
    reverse += noise
    moved = written_field(tmp_path, whole, "moved.nii")
    reverse -= noise
    expected = written_field(tmp_path, whole, "whole.nii")
    field = np.zeros((*size, len(size)), dtype=np.float32)
    sampled.write_field(field, block=20)
    return float(np.abs(field - expected).max()), max(1e-3, 2 * float(np.abs(moved - expected).max()))


THIN = (20, 40, 44)  # z under FireANTs' MIN_IMG_SIZE: its grid has 32 nodes along it


@pytest.mark.parametrize("size, scales", [(THIN, [2, 1]), (SIZE, [4, 2])], ids=["thin", "last-scale-coarse"])
def test_the_syn_field_inverts_the_reverse_warp_on_its_own_grid(tmp_path, size, scales):
    """SyN's inverse reads the reverse warp on its own grid, the last scale's, as compositive_warp_inverse does: an
    axis under 32 voxels or a last scale above 1 keeps it off the fixed grid. The field is the whole run's."""
    with mock.patch.dict(DEFORMABLE, {"scales": scales}):
        whole, _, sampled = deformable_pair(tmp_path, True, "cc", size=size)
    gap, tolerance = syn_field_gap(tmp_path, whole, sampled, size)
    assert gap < tolerance
    sampled.close()


@pytest.mark.parametrize("loss", ["cc", "mi"])
def test_every_box_reproduces_the_whole_image_syn_on_a_thin_axis(tmp_path, loss):
    """An axis under MIN_IMG_SIZE: SyN's grid is finer than the image along it, the fixed image interpolated onto it at
    the last scale as FireANTs interpolates it; every box at every iteration gives FireANTs' warps."""
    _, whole_warps, sampled = deformable_pair(tmp_path, True, loss, size=THIN)
    assert warp_gap(sampled, whole_warps, THIN) < 1e-3
    sampled.close()


@pytest.mark.parametrize("loss", ["cc", "mi"])
def test_every_box_reproduces_the_whole_image_masked_deformable(tmp_path, loss):
    """With a mask that keeps some boxes more than others, every box at every iteration is still FireANTs' masked
    iteration: each box weighed against the level's support, not its own, or its voxels counted in the level's
    joint histogram."""
    z, y, x = np.meshgrid(*(np.arange(n) for n in SIZE), indexing="ij")
    mask = (((x - 30) ** 2 + (y - 14) ** 2 + (z - 10) ** 2) < 15**2).astype(np.float32)  # off-centre: uneven boxes
    _, whole_warps, sampled = deformable_pair(tmp_path, False, loss, mask=mask)
    assert sampled.boxes_per_scale[-1] > 1
    assert warp_gap(sampled, whole_warps) < 1e-3
    sampled.close()


@pytest.mark.parametrize("loss", ["mse", "cc", "masked_cc"])
def test_a_level_s_loss_by_boxes_is_its_loss_in_one_box(tmp_path, loss):
    """The level loss the deformable stage reports, which tells it when a scale has converged, is the whole grid's
    however many boxes cut the level: a box's halo is read, never counted."""
    fixed, moving = texture(), texture((0.6, -0.4, 0.3))
    pair = [ArraySource(fixed[None], "cpu"), ArraySource(moving[None], "cpu")]
    channels = 1
    if loss.startswith("masked"):
        z, y, x = np.meshgrid(*(np.arange(n) for n in SIZE), indexing="ij")
        mask = (((x - 30) ** 2 + (y - 14) ** 2 + (z - 10) ** 2) < 15**2).astype(np.float32)
        pair = [
            StackedSource([pair[0], ArraySource(mask[None], "cpu")]),
            StackedSource([pair[1], OnesSource(SIZE, "cpu")]),
        ]
        channels = 2

    def first_loss(box: int) -> tuple[int, float]:
        sampled = sampled_deformable_class()(
            [1],
            [1],
            HeaderImages(image_of(fixed), channels, "cpu"),
            HeaderImages(image_of(moving), channels, "cpu"),
            *pair,
            scratch=tmp_path,
            symmetric=False,
            loss_type=loss,
            cc_kernel_size=5,
            box=box,
        )
        sampled.optimize()
        sampled.close()
        return sampled.boxes_per_scale[0], sampled.losses[0]

    (boxes, by_boxes), (one, whole) = first_loss(16), first_loss(64)
    assert boxes > 1 and one == 1
    assert math.isclose(by_boxes, whole, rel_tol=1e-5)


def test_a_draw_the_mask_misses_keeps_the_warps_finite(tmp_path):
    """With a mask in one corner, most draws of one box miss it: they hold no voxel of the loss, and the warps stay
    finite."""
    mask = np.zeros(SIZE, dtype=np.float32)
    mask[:10, :10, :10] = 1.0
    _, _, sampled = deformable_pair(tmp_path, False, "cc", samples=16**3, mask=mask)
    assert all(np.isfinite(np.asarray(warp.arrays["warp"])).all() for warp in sampled._warps.values())
    sampled.close()


def test_an_iteration_steps_the_boxes_it_draws(tmp_path):
    """With the samples of one box an iteration, a scale draws a box at a time, each as often on average as the
    whole-grid run iterates, and stays near that run."""
    _, whole_warps, sampled = deformable_pair(tmp_path, True, "cc", samples=16**3)
    boxes = [len(tiles(fireants_level_sizes(SIZE, scale), 16)) for scale in DEFORMABLE["scales"]]
    assert sampled.boxes_per_scale == boxes
    steps = zip(DEFORMABLE["iterations"], boxes, strict=True)
    assert len(sampled.losses) == sum(epochs * count for epochs, count in steps)
    assert warp_gap(sampled, whole_warps) < 0.5
    sampled.close()


def test_warps_held_on_the_device_step_as_warps_held_in_files(tmp_path, monkeypatch):
    """The homes of the warps and their moments give the same stage: files, host memory, the device, and the device
    at the coarser scale then files at the finest, which the device does not hold."""
    import impact_reg_konfai.models.fireants_sampled as module

    homes = []
    resize = module._Warp.resize
    monkeypatch.setattr(
        module._Warp,
        "resize",
        lambda warp, grid, home, host: (homes.append((warp.resident, home)), resize(warp, grid, home, host))[1],
    )
    fixed, moving = texture(), texture((1.3, -0.8, 0.6))
    warps = []
    for resident, host, kind in (
        (False, False, np.memmap),
        (False, True, np.ndarray),
        (True, False, torch.Tensor),
        (None, False, np.memmap),  # the finest scale in files
    ):
        monkeypatch.setattr(module, "max_voxels", lambda vram, ram, gpu, host=host: 2**40 if host else 1)
        if resident is None:  # three quarters of it hold the finest scale's warps of SyN, not the coarser one's
            monkeypatch.setattr(module, "free_bytes", lambda device: 4 * 3 * 4 * math.prod(SIZE) * 2)
        (tmp_path / f"{resident}{host}").mkdir()
        sampled = sampled_deformable_class()(
            fixed_images=HeaderImages(image_of(fixed), 1, "cpu"),
            moving_images=HeaderImages(image_of(moving), 1, "cpu"),
            fixed=ArraySource(fixed[None], "cpu"),
            moving=ArraySource(moving[None], "cpu"),
            scratch=tmp_path / f"{resident}{host}",
            symmetric=True,
            loss_type="cc",
            init_affine=AFFINE.clone(),
            box=16,
            resident=resident,
            **DEFORMABLE,
        )
        sampled.optimize()
        stored = sampled._warps["fwd"].arrays["warp"]
        assert type(stored) is kind
        stage = np.stack([np.asarray(sampled._warps[name].arrays["warp"]) for name in ("fwd", "rev")])
        field = np.zeros((*SIZE, 3), dtype=np.float32)
        sampled.write_field(field, block=20)
        warps.append((stage, field))
        sampled.close()
    assert (True, False) in homes and not sampled.resident  # the mixed run went from the device to files
    assert all(np.abs(warps[0][0] - stage).max() < 1e-4 for stage, _ in warps[1:])
    # FireANTs' inverse, an optimisation, takes the rounding of the homes' resizes to thousandths of a voxel.
    assert all(np.abs(warps[0][1] - field).max() < 5e-3 for _, field in warps[1:])


def test_the_syn_inverse_goes_on_the_device_only_where_it_fits_beside_the_warps(tmp_path, monkeypatch):
    """SyN's field reads a third warp, the reverse one's inverse, beside the two the device was measured for: it goes
    on the device only where what is free then holds it, and the field is the same wherever it lives."""
    import impact_reg_konfai.models.fireants_sampled as module

    fixed, moving = texture(), texture((1.3, -0.8, 0.6))
    two_warps = 4 * 3 * 4 * math.prod(SIZE) * 2
    fields, homes = [], []
    # Room for many warps, then for the two alone, which leave none once they are on the device.
    for free, left in ((two_warps * 10, two_warps * 10), (two_warps * 4 / 3, 0)):
        monkeypatch.setattr(module, "free_bytes", lambda device, free=free: free)
        (tmp_path / str(free)).mkdir()
        sampled = sampled_deformable_class()(
            fixed_images=HeaderImages(image_of(fixed), 1, "cpu"),
            moving_images=HeaderImages(image_of(moving), 1, "cpu"),
            fixed=ArraySource(fixed[None], "cpu"),
            moving=ArraySource(moving[None], "cpu"),
            scratch=tmp_path / str(free),
            symmetric=True,
            loss_type="cc",
            init_affine=AFFINE.clone(),
            box=16,
            **DEFORMABLE,
        )
        assert sampled.resident
        sampled.optimize()
        monkeypatch.setattr(module, "free_bytes", lambda device, left=left: left)
        sampled._inverse = lambda block, inverse=sampled._inverse: (
            (warp := inverse(block)),
            homes.append(warp.resident),
        )[0]
        fields.append(np.zeros((*SIZE, 3), dtype=np.float32))
        sampled.write_field(fields[-1], block=20)
        sampled.close()
    assert homes == [True, False]
    assert np.abs(fields[0] - fields[1]).max() < 5e-3


@SHAPES
def test_a_warp_held_on_the_device_is_resized_as_one_held_in_files(tmp_path, size):
    """Between two scales a warp goes to the finer grid by linear interpolation, a plane's as a volume's, wherever it
    lives."""
    from impact_reg_konfai.models.fireants_sampled import _Warp

    coarse = [extent // 2 for extent in size]
    values = torch.from_numpy(np.random.default_rng(0).random((len(size), *coarse)).astype(np.float32))
    resized = []
    for resident in (True, False):
        warp = _Warp(tmp_path, f"w{resident}", coarse, "cpu", resident, False)
        warp.arrays["warp"][:] = values if resident else values.numpy()
        warp.resize(size, resident, False)
        resized.append(torch.as_tensor(np.asarray(warp.arrays["warp"])))
    assert resized[0].shape == (len(size), *size) and torch.allclose(resized[0], resized[1], atol=1e-5)


@pytest.mark.parametrize("nearest", [False, True], ids=["linear", "nearest"])
@SHAPES
def test_a_source_read_through_a_field_by_blocks_is_simpleitk_s_resampling(tmp_path, size, nearest):
    """Read through a displacement field block by block, a source on its own grid gives SimpleITK's resampling of
    it through that field onto the field's grid, linear for an image, at the nearest voxel for a mask."""
    from impact_reg_konfai.models.fireants_sampled import read_through

    dims = len(size)
    source = sitk.GetImageFromArray(texture((0.7, -0.4, 0.3), size))
    source.SetSpacing([0.9, 1.1, 1.3][:dims])
    source.SetOrigin([-2.0, 1.5, 0.5][:dims])
    grid = sitk.Image(list(reversed(size)), sitk.sitkFloat32)
    angle = np.deg2rad(7.0)
    rotation = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    grid.SetDirection(np.asarray(rotation).ravel() if dims == 2 else sitk.VersorTransform((0, 0, 1), angle).GetMatrix())
    grid.SetSpacing([1.0, 0.8, 1.2][:dims])
    rng = np.random.default_rng(0)
    field = (rng.random((*size, dims)) * 3.0 - 1.5).astype(np.float64)
    displacement = sitk.GetImageFromArray(field, isVector=True)
    displacement.CopyInformation(grid)
    interpolator = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
    transform = sitk.DisplacementFieldTransform(sitk.Image(displacement))
    expected = sitk.GetArrayFromImage(sitk.Resample(source, grid, transform, interpolator, 0.0))
    out = np.zeros((1, *size), dtype=np.float32)
    array = sitk.GetArrayFromImage(source)[None]
    read_through(ArraySource(array, "cpu"), source, field.astype(np.float32), grid, out, block=16, nearest=nearest)
    assert np.abs(out[0] - expected).max() < 1e-4


def test_a_source_kept_on_its_device_reads_the_same_boxes_as_copies(monkeypatch):
    import impact_reg_konfai.ooc as ooc

    source = ooc.ArraySource(np.arange(2 * 4 * 5 * 6, dtype=np.float32).reshape(2, 4, 5, 6), "cpu")
    monkeypatch.setattr(ooc, "free_bytes", lambda device: 0)
    assert ooc.on_device(source) is source
    monkeypatch.setattr(ooc, "free_bytes", lambda device: 2**40)
    kept = ooc.on_device(source)
    assert isinstance(kept, ooc.TensorSource) and (kept.shape, kept.channels) == (source.shape, source.channels)
    box = kept.read([1, 0, 2], [3, 5, 6])
    assert torch.equal(box, source.read([1, 0, 2], [3, 5, 6]))
    box.zero_()  # a reader may change its box in place
    assert torch.equal(kept.read([1, 0, 2], [3, 5, 6]), source.read([1, 0, 2], [3, 5, 6]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the pinned route is the GPU's")
def test_a_box_of_a_memmap_crosses_to_the_gpu_and_back_unchanged(tmp_path):
    import impact_reg_konfai.ooc as ooc

    array = np.lib.format.open_memmap(str(tmp_path / "a.npy"), mode="w+", dtype=np.float32, shape=(3, 8, 9, 10))
    array[:] = np.random.default_rng(0).random(array.shape)
    before = np.array(array)
    box = (slice(None), slice(2, 7), slice(1, 8), slice(3, 9))
    read = ooc.to_device(array[box], "cuda")
    assert read.is_cuda and np.array_equal(read.cpu().numpy(), before[box])
    ooc.to_array(array[box], read.movedim(0, -1).contiguous().movedim(-1, 0) * 2)  # a strided tensor, as a warp's
    expected = before.copy()
    expected[box] *= 2
    assert np.array_equal(np.array(array), expected)


def test_a_box_read_at_the_image_edge_is_the_whole_grid_s():
    """A box that keeps one voxel along an axis at the edge of the image still blends it with the zero beyond, as
    grid_sample on the whole grid does (on an axis of one voxel, align_corners reads that voxel everywhere)."""
    from impact_reg_konfai.models.fireants_sampled import _sample_region

    source = torch.rand(1, 1, 8, 8, 8)
    for x in (1.3, 1.1, -1.2, 0.95):
        point = torch.tensor([x, 0.1, -0.2]).view(1, 1, 1, 1, 3)
        whole = torch.nn.functional.grid_sample(source, point, padding_mode="zeros", align_corners=True)
        assert abs(float(_sample_region(ArraySource(source[0].numpy(), "cpu"), point)) - float(whole)) < 1e-6


def test_an_unknown_deformable_method_is_refused_before_any_stage():
    from impact_reg_konfai.models.fireants import FireANTsEngine

    settings = {"scales": [2, 1], "affine_iterations": [2, 2], "deformable_iterations": [2, 2], "cc_kernel": 5}
    settings |= {"affine_metric": "mi", "affine_lr": 0.03, "moments_init": "cof", "linear_method": "rigid_affine"}
    settings |= {"deformable_metric": "cc", "deformable_lr": 0.25, "smooth_warp_sigma": 0.5, "smooth_grad_sigma": 1.0}
    with pytest.raises(ValueError, match="deformable_method"):  # by boxes it would have run greedy
        FireANTsEngine(**settings, deformable_method="syyn", seed=0, impact_levels=[])


class _Standardized(torch.nn.Module):
    """A layer that is its input standardised by the statistics it is handed: what a model normalising by the whole
    image's intensities (TotalSegmentator) does before its first convolution."""

    def forward(
        self,
        x: torch.Tensor,
        nb_layers: torch.Tensor,
        stats: Optional[torch.Tensor] = None,  # noqa: UP045 (TorchScript)
        direction: Optional[torch.Tensor] = None,  # noqa: UP045 (TorchScript)
    ) -> list[torch.Tensor]:
        if stats is None or stats.numel() != 4:  # as itk-impact's models: the tensor's own statistics
            return [(x[:, :1] - x[:, :1].mean()) / x[:, :1].std()]
        return [(x[:, :1] - stats[2]) / stats[3]]


def test_jacobian_boxes_normalise_by_the_level_s_intensities(tmp_path):
    """Evaluated box by box, a Jacobian loss whose network normalises by its input's intensity statistics runs it on
    the whole warped images' (``pool``), as the whole level's evaluation does: the level's value and gradient."""
    from impact_reg_konfai.models import fireants as engine

    path = tmp_path / "standardized.pt"
    torch.jit.script(_Standardized()).save(str(path))
    spec = engine.ModelSpec(ref=str(path), distance="L2")
    loss = engine.ImpactFeatureLoss([[spec]], "Jacobian", False, 5, 0, 0, False)
    loss._level = 0
    generator = torch.Generator().manual_seed(2)
    shape = (20, 24, 28)
    moved, fixed = torch.rand((1, 1, *shape), generator=generator), torch.rand((1, 1, *shape), generator=generator)
    whole = moved.clone().requires_grad_()
    expected_value = loss(whole, fixed)
    (expected,) = torch.autograd.grad(expected_value, whole)
    boxes = []
    for tile in tiles(shape, 9):
        low, high, core = box_region(tile, 2, shape)
        region = torch.zeros((1, 1, *(b - a for a, b in zip(low, high, strict=True))))
        region[(slice(None), slice(None), *core)] = 1
        crop = (slice(None), slice(None), *(slice(a, b) for a, b in zip(low, high, strict=True)))
        boxes.append((low, core, crop, region))
    assert loss.pools
    loss.pool([(moved[crop], fixed[crop], region) for _, _, crop, region in boxes])
    gathered, value = torch.zeros_like(moved), 0.0
    for low, core, crop, region in boxes:
        box = moved[crop].clone().requires_grad_()
        own = (slice(None), slice(None), *core)
        voxels = math.prod(c.stop - c.start for c in core)
        # The box's own voxels' share of the level's mean, as the stage weighs a box.
        share = loss._values(box, fixed[crop], region)[1][0]
        share = share * voxels / math.prod(shape)
        (gradient,) = torch.autograd.grad(share, box)
        value += float(share)
        placed = tuple(slice(lo + c.start, lo + c.stop) for lo, c in zip(low, core, strict=True))
        gathered[(slice(None), slice(None), *placed)] = gradient[own]
    assert math.isclose(value, float(expected_value), rel_tol=1e-5)
    assert torch.allclose(gathered, expected, atol=1e-7)


class _Averaged(torch.nn.Module):
    """Two channels with a receptive field of 3 voxels: the input averaged over 3^3 voxels, and its square."""

    def forward(
        self,
        x: torch.Tensor,
        nb_layers: torch.Tensor,
        stats: Optional[torch.Tensor] = None,  # noqa: UP045 (TorchScript)
        direction: Optional[torch.Tensor] = None,  # noqa: UP045 (TorchScript)
    ) -> list[torch.Tensor]:
        y = x[:, :1]
        return [torch.cat([torch.nn.functional.avg_pool3d(y, 3, 1, 1), y * y], 1)]


@pytest.mark.parametrize("pca", [0, 1])
@pytest.mark.parametrize("masked", [False, True])
def test_a_jacobian_ncc_layer_by_boxes_has_the_level_s_value_and_gradient(tmp_path, masked, pca):
    """Evaluated box by box, each box read with the receptive field's halo and weighed as the stage weighs it, a
    Jacobian NCC layer has the level's value and gradient: the boxes' sums (``pool``, a second pass over them once the
    level's intensities are known) give each box the level's NCC and its derivative, on the PCA basis the whole fixed
    map gives when the model reduces its layers."""
    from impact_reg_konfai.models import fireants as engine

    path = tmp_path / "averaged.pt"
    torch.jit.script(_Averaged()).save(str(path))
    spec = engine.ModelSpec(ref=str(path), distance="NCC", pca=pca)
    loss = engine.ImpactFeatureLoss([[spec]], "Jacobian", False, 5, 0, 0, False, masked=masked)
    loss._level = 0
    generator = torch.Generator().manual_seed(3)
    shape = (20, 24, 28)
    moved, fixed = (torch.rand((1, 2, *shape), generator=generator) for _ in range(2))
    for volume, cut in ((moved, 9), (fixed, 13)):
        volume[:, -1] = 0.0
        volume[:, -1, :, :cut] = 1.0
    if not masked:
        moved, fixed = moved[:, :1], fixed[:, :1]
    loss.draw()
    whole = moved.clone().requires_grad_()
    expected_value = loss(whole, fixed)
    (expected,) = torch.autograd.grad(expected_value, whole)
    boxes = []
    for tile in tiles(shape, 9):
        low, high, core = box_region(tile, 2, shape)
        region = torch.zeros((1, 1, *(b - a for a, b in zip(low, high, strict=True))))
        region[(slice(None), slice(None), *core)] = 1
        crop = (slice(None), slice(None), *(slice(a, b) for a, b in zip(low, high, strict=True)))
        boxes.append((low, core, crop, region))
    loss.pool(lambda: [(moved[crop], fixed[crop], region) for _, _, crop, region in boxes])
    support = sum(
        float(((moved[crop][:, -1:] * fixed[crop][:, -1:] >= 0.5) * region).sum()) for *_, crop, region in boxes
    )
    gathered = torch.zeros_like(moved)
    for low, core, crop, _ in boxes:
        box = moved[crop].clone().requires_grad_()
        value = loss(box, fixed[crop])
        # The stage's weight of the box: its support over the level's (masked), its voxels over the level's.
        weight = (
            (box[:, -1:] * fixed[crop][:, -1:] >= 0.5).sum() / support
            if masked
            else box[0, 0].numel() / whole[0, 0].numel()
        )
        if weight == 0:  # a box the masks miss scores nothing, which the stage weighs by 0
            continue
        assert math.isclose(float(value), float(expected_value), rel_tol=1e-5)
        share = value * weight
        (gradient,) = torch.autograd.grad(share, box)
        own = tuple(slice(lo + c.start, lo + c.stop) for lo, c in zip(low, core, strict=True))
        gathered[(slice(None), slice(None), *own)] = gradient[(slice(None), slice(None), *core)]
    assert torch.allclose(gathered[:, :1], expected[:, :1], atol=1e-7)


@pytest.mark.parametrize("distance", ["L2", "NCC"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_a_jacobian_model_at_a_voxel_size_by_boxes_has_the_level_s_gradient(tmp_path, masked, distance, strided):
    """A Jacobian model with a voxel_size reads the level on its own grid: a box takes the cells of that grid it
    holds (``place``), from one on its layers' stride (a layer at half the grid, anchored at the end of an axis its
    network flips), normalised by the level's intensities there, its mean weighed by its cells' share of the level's,
    and the boxes' gradients are the whole level's."""
    from impact_reg_konfai.models import fireants as engine

    path = tmp_path / "model.pt"
    torch.jit.script(_Strided(flipped=True) if strided else _Averaged()).save(str(path))
    spec = engine.ModelSpec(
        ref=str(path), distance=distance, voxel_size=[1.4, 1.3, 1.7], layers_mask="01" if strided else "1"
    )
    loss = engine.ImpactFeatureLoss([[spec]], "Jacobian", False, 5, 0, 0, False, masked=masked)
    loss._level, loss.cores[0].extent = 0, [28.0, 24.0, 20.0]  # one voxel a mm along L, P, S
    generator = torch.Generator().manual_seed(4)
    shape = (20, 24, 28)
    moved, fixed = (torch.rand((1, 2, *shape), generator=generator) for _ in range(2))
    for volume, cut in ((moved, 9), (fixed, 13)):
        volume[:, -1] = 0.0
        volume[:, -1, :, :cut] = 1.0
    if not masked:
        moved, fixed = moved[:, :1], fixed[:, :1]
    loss.draw()
    whole = moved.clone().requires_grad_()
    (expected,) = torch.autograd.grad(loss(whole, fixed), whole)
    boxes = []
    for tile in tiles(shape, 9):
        low, high, core = box_region(tile, 10, shape)
        region = torch.zeros((1, 1, *(b - a for a, b in zip(low, high, strict=True))))
        region[(slice(None), slice(None), *core)] = 1
        crop = (slice(None), slice(None), *(slice(a, b) for a, b in zip(low, high, strict=True)))
        boxes.append((low, high, core, crop, region))

    def pieces():
        for low, high, _, crop, region in boxes:
            loss.place(low, high, shape)
            yield moved[crop], fixed[crop], region

    loss.pool(pieces)
    support = sum(
        float(((moved[crop][:, -1:] * fixed[crop][:, -1:] >= 0.5) * region).sum()) for *_, crop, region in boxes
    )
    gathered = torch.zeros_like(moved)
    for low, high, core, crop, _ in boxes:
        loss.place(low, high, shape)
        box = moved[crop].clone().requires_grad_()
        value = loss(box, fixed[crop])
        # The stage's weight of the box: its support over the level's (masked), its voxels over the level's.
        weight = (
            (box[:, -1:] * fixed[crop][:, -1:] >= 0.5).sum() / support
            if masked
            else box[0, 0].numel() / whole[0, 0].numel()
        )
        if weight == 0:
            continue
        (gradient,) = torch.autograd.grad(value * weight, box)
        own = tuple(slice(lo + c.start, lo + c.stop) for lo, c in zip(low, core, strict=True))
        gathered[(slice(None), slice(None), *own)] = gradient[(slice(None), slice(None), *core)]
    assert torch.allclose(gathered[:, :1], expected[:, :1], atol=1e-7)


def test_the_whole_syn_field_takes_the_inverse_onto_the_fixed_grid(monkeypatch):
    """On an axis under 32 voxels FireANTs computes SyN's inverse at 32. The whole route brings that inverse onto the
    fixed grid, as the boxes do, where SyN's get_warp_parameters would put the reverse warp itself in its place."""
    from fireants.utils import warputils
    from impact_reg_konfai.models.fireants import _inverse_on

    inverse = torch.rand(1, 32, 44, 2, generator=torch.Generator().manual_seed(0))
    monkeypatch.setattr(warputils, "compositive_warp_inverse", lambda *args, **kwargs: inverse)
    found = _inverse_on((20, 44))(None, None)
    expected = torch.nn.functional.interpolate(inverse.movedim(-1, 1), (20, 44), mode="bilinear", align_corners=True)
    assert torch.equal(found, expected.movedim(1, -1))


@pytest.mark.parametrize("masked", [False, True])
def test_a_jacobian_strided_layer_by_aligned_boxes_has_the_level_s_gradient(tmp_path, masked):
    """A Jacobian layer at half the image's resolution, anchored at the end of an axis its network flips (as
    TotalSegmentator's), evaluated on boxes that start on its stride and end as far from the level's end as a whole
    number of it, each box's mean brought to its cells' share of the level's, has the whole level's gradient, the
    level's last axis odd."""
    from impact_reg_konfai.models import fireants as engine
    from konfai.metric.measure.impact import block_region

    path = tmp_path / "strided.pt"
    torch.jit.script(_Strided(flipped=True)).save(str(path))
    spec = engine.ModelSpec(ref=str(path), layers_mask="01", distance="L2")
    loss = engine.ImpactFeatureLoss([[spec]], "Jacobian", False, 5, 0, 0, False, masked=masked)
    loss._level = 0
    assert loss.stride("cpu") == 2
    generator = torch.Generator().manual_seed(5)
    shape = (20, 24, 27)
    moved, fixed = (torch.rand((1, 2, *shape), generator=generator) for _ in range(2))
    for volume, cut in ((moved, 9), (fixed, 13)):
        volume[:, -1] = 0.0
        volume[:, -1, :, :cut] = 1.0
    if not masked:
        moved, fixed = moved[:, :1], fixed[:, :1]
    loss.draw()
    whole = moved.clone().requires_grad_()
    (expected,) = torch.autograd.grad(loss(whole, fixed), whole)
    boxes = []
    for start, stop in tiles(shape, 8):
        low, high, core = block_region(start, stop, [4] * 3, shape, 2)
        region = torch.zeros((1, 1, *(b - a for a, b in zip(low, high, strict=True))))
        region[(slice(None), slice(None), *core)] = 1
        crop = (slice(None), slice(None), *(slice(a, b) for a, b in zip(low, high, strict=True)))
        boxes.append((low, high, core, crop, region))

    def pieces():
        for low, high, _, crop, region in boxes:
            loss.place(low, high, shape)
            yield moved[crop], fixed[crop], region

    loss.pool(pieces)
    support = sum(
        float(((moved[crop][:, -1:] * fixed[crop][:, -1:] >= 0.5) * region).sum()) for *_, crop, region in boxes
    )
    gathered = torch.zeros_like(moved)
    for low, high, core, crop, _ in boxes:
        loss.place(low, high, shape)
        box = moved[crop].clone().requires_grad_()
        value = loss(box, fixed[crop])
        weight = (
            (box[:, -1:] * fixed[crop][:, -1:] >= 0.5).sum() / support
            if masked
            else box[0, 0].numel() / whole[0, 0].numel()
        )
        if weight == 0:
            continue
        (gradient,) = torch.autograd.grad(value * weight, box)
        own = tuple(slice(lo + c.start, lo + c.stop) for lo, c in zip(low, core, strict=True))
        gathered[(slice(None), slice(None), *own)] = gradient[(slice(None), slice(None), *core)]
    assert torch.allclose(gathered[:, :1], expected[:, :1], atol=1e-7)


@pytest.mark.parametrize("distance", ["L2", "NCC"])
def test_a_jacobian_level_s_normalisation_is_the_whole_level_s_whatever_the_box(tmp_path, distance):
    """The deformable stage latches a strided Jacobian layer's normalisation on boxes of any side, a multiple of the
    network's stride or not: the whole level's."""
    from impact_reg_konfai.models import fireants as engine

    path = tmp_path / "strided.pt"
    torch.jit.script(_Strided()).save(str(path))
    generator = torch.Generator().manual_seed(2)
    fixed, moving = (torch.rand(SIZE, generator=generator).numpy() for _ in range(2))

    def latched(box: int) -> list[float]:
        spec = engine.ModelSpec(ref=str(path), layers_mask="01", distance=distance)
        loss = engine.ImpactFeatureLoss([[spec]], "Jacobian", True, 5, 0, 0, False)
        sampled = sampled_deformable_class()(
            [1],
            [0],
            HeaderImages(image_of(fixed), 1, "cpu"),
            HeaderImages(image_of(moving), 1, "cpu"),
            ArraySource(fixed[None], "cpu"),
            ArraySource(moving[None], "cpu"),
            scratch=tmp_path,
            symmetric=False,
            loss_type="custom",
            custom_loss=loss,
            loss_halo=4,
            box=box,
        )
        sampled.optimize()
        sampled.close()
        return loss._factors

    assert np.allclose(latched(9), latched(64), rtol=1e-5)


def test_a_level_of_several_jacobian_models_starts_its_boxes_on_all_their_strides(tmp_path):
    """Boxes of a level whose networks pool by 2 and by 3 start on 6: on the larger stride alone, every other box
    starts off the first network's cells."""
    from impact_reg_konfai.models import fireants as engine

    specs = []
    for pool in (2, 3):
        path = tmp_path / f"strided{pool}.pt"
        torch.jit.script(_Strided(pool=pool)).save(str(path))
        specs.append(engine.ModelSpec(ref=str(path), layers_mask="01", distance="L2"))
    loss = engine.ImpactFeatureLoss([specs], "Jacobian", True, 5, 0, 0, False)
    assert loss.stride("cpu") == 6


@pytest.mark.parametrize("voxel_size", [None, [1.5, 2.0]], ids=["native", "voxel_size"])
def test_a_2d_jacobian_model_scores_a_plane_by_boxes_as_it_scores_it_whole(tmp_path, voxel_size):
    """A 2-D network on a 2-D pair reads the plane as it is, no axis swept: by boxes, read with the receptive field
    around them and weighed as the stage weighs them, it gives the level's value, at its voxel_size too."""
    from impact_reg_konfai.models import fireants as engine

    path = tmp_path / "convolved.pt"
    torch.jit.script(_Convolved()).save(str(path))
    spec = engine.ModelSpec(ref=str(path), distance="L2")
    loss = engine.ImpactFeatureLoss([[spec]], "Jacobian", False, 5, 0, 0, False)
    core = loss.cores[0]  # a local file is a 3-D model to the registry: the 2-D network handed in
    core.model.model, core.model.dim, core.dimension = torch.jit.script(_Convolved(2)), 2, 2
    core.model.fov, core.model.planes, core.model.voxel_size = [3], True, voxel_size
    loss._level = 0
    generator = torch.Generator().manual_seed(4)
    shape = (40, 44)
    moved, fixed = (torch.rand((1, 1, *shape), generator=generator) for _ in range(2))
    loss.draw()
    expected = float(loss(moved, fixed))
    pieces = []
    for tile in tiles(shape, 16):
        low, high, core_slices = box_region(tile, 8, shape)
        region = torch.zeros((1, 1, *(b - a for a, b in zip(low, high, strict=True))))
        region[(slice(None), slice(None), *core_slices)] = 1
        crop = (slice(None), slice(None), *(slice(a, b) for a, b in zip(low, high, strict=True)))
        pieces.append((low, high, moved[crop], fixed[crop], region))

    def read():
        for low, high, box_moved, box_fixed, region in pieces:
            loss.place(low, high, shape)
            yield box_moved, box_fixed, region

    loss.pool(read)
    value = 0.0
    for low, high, box_moved, box_fixed, region in pieces:
        loss.place(low, high, shape)
        # The box's own voxels' share, as the stage's normalisation latches it.
        value += float(loss._values(box_moved, box_fixed, region)[1][0]) * float(region.sum()) / math.prod(shape)
    assert math.isclose(value, expected, rel_tol=1e-4)

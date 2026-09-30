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

"""When a pass does not fit the card, the FireANTs engine retries it in smaller pieces rather than failing or
letting konfai cut the registration into patches: the IMPACT metric in checkpointed tiles, the feature
correlation a few channels at a time. Both are the same value with the same gradient."""

from pathlib import Path

import pytest
import torch
from impact_reg_konfai.models import fireants
from konfai.metric.measure.impact import _statistics


class _PointwiseFeatures(torch.nn.Module):
    """A feature model with no receptive field, so a tiled score is exactly the whole-image one."""

    def forward(
        self,
        x: torch.Tensor,
        nb_layer: torch.Tensor,
        stats: torch.Tensor | None = None,
        direction: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        return [x * 2.0, x * x]


def _core(tmp_path: Path) -> "fireants._ImpactCore":
    path = tmp_path / "features.pt"
    torch.jit.script(_PointwiseFeatures()).save(str(path))
    return fireants._ImpactCore(fireants.ModelSpec(ref=str(path), layers_mask="11", distance="L1"), False)


TERMS = [("L1", 0), ("L1", 0)]


def test_the_impact_metric_falls_back_to_checkpointed_tiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    monkeypatch.setattr(fireants, "MIN_TILE", 2)  # an 8-voxel test volume, far below the real floor
    core = _core(tmp_path)
    moved, fixed = torch.rand(1, 1, 8, 8, 8, requires_grad=True), torch.rand(1, 1, 8, 8, 8)
    whole = core.distances(moved, fixed, None, TERMS, 0, 3)
    (whole_gradient,) = torch.autograd.grad(whole.sum(), moved)

    scored = core._scored

    def out_of_memory_on_the_whole_image(*args):
        if core.model.shape is None:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.85 GiB.")
        return scored(*args)

    monkeypatch.setattr(core, "_scored", out_of_memory_on_the_whole_image)
    tiled = core.distances(moved, fixed, None, TERMS, 0, 3)
    (tiled_gradient,) = torch.autograd.grad(tiled.sum(), moved)
    assert core._tiles[(8, 8, 8)] == [4, 4, 4]
    assert core.model.shape == [4, 4, 4] and core.model.checkpoint

    # The checkpoint changes the memory, not the result: the same tiles scored without it give the same, and a
    # pointwise model's tiles give the whole image's.
    monkeypatch.setattr(core, "_scored", scored)
    core.model.checkpoint = False
    statistics = (_statistics(moved)[0], _statistics(fixed)[0])
    plain = scored(moved, fixed, None, statistics, TERMS, 0, 3)
    (plain_gradient,) = torch.autograd.grad(plain.sum(), moved)
    torch.testing.assert_close(tiled, plain)
    torch.testing.assert_close(tiled_gradient, plain_gradient)
    torch.testing.assert_close(tiled, whole)
    assert whole_gradient.abs().sum() > 0


def test_the_static_lncc_falls_back_to_fewer_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    torch.manual_seed(0)
    moved, fixed = torch.rand(1, 6, 6, 6, 6, requires_grad=True), torch.rand(1, 6, 6, 6, 6)
    spec = fireants.ModelSpec(ref="model.pt", distance="LNCC")
    loss = fireants.ImpactFeatureLoss.__new__(fireants.ImpactFeatureLoss)
    torch.nn.Module.__init__(loss)  # the model download in __init__ is not what is tested
    loss._masked, loss._mode, loss._normalize, loss._kernel, loss._chunk = False, "Static", False, 3, 0
    loss._specs, loss._levels, loss._channels, loss._level, loss._factors = [spec], [[0]], [[6]], -1, None
    loss._cores = [SimpleNamespace(kept=[0])]
    loss._generator = torch.Generator().manual_seed(0)
    loss._sampling, loss._patch = 1.0, []
    whole = loss(moved, fixed)
    (whole_gradient,) = torch.autograd.grad(whole, moved)

    distance = fireants.distance

    def out_of_memory_above_two_channels(name, m, f, mask, kernel, chunk):
        if not 0 < chunk <= 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 830.00 MiB.")
        return distance(name, m, f, mask, kernel, chunk)

    monkeypatch.setattr(fireants, "distance", out_of_memory_above_two_channels)
    chunked = loss(moved, fixed)
    (chunked_gradient,) = torch.autograd.grad(chunked, moved)

    assert loss._chunk == 2
    torch.testing.assert_close(chunked, whole)
    torch.testing.assert_close(chunked_gradient, whole_gradient)


def _static_loss(tmp_path: Path) -> "fireants.ImpactFeatureLoss":
    path = tmp_path / "features.pt"
    torch.jit.script(_PointwiseFeatures()).save(str(path))
    spec = fireants.ModelSpec(ref=str(path), layers_mask="11", distance="L1")
    return fireants.ImpactFeatureLoss([[spec]], "Static", False, 5, 0, 0, False)


def test_a_static_extraction_that_does_not_fit_tiles_both_images_alike(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The whole-image pass runs out of memory: the larger image finds the tile that fits, the other goes straight
    # through the same one, and a thin axis is one tile of its own rather than padded up to a 256-voxel cube
    # (an overlap of 64 voxels did not even fit a 40-voxel axis).
    from konfai.metric.measure.impact import ImpactFeatureModel

    torch.manual_seed(0)
    tiles: list[tuple[tuple[int, ...], int]] = []
    volume = ImpactFeatureModel._volume

    def out_of_memory_on_the_whole_image(self, image, normalization, patch, overlap):
        tiles.append((tuple(image.shape[2:]), patch))
        if patch == 0:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 9.10 GiB.")
        return volume(self, image, normalization, patch, overlap)

    monkeypatch.setattr(ImpactFeatureModel, "_volume", out_of_memory_on_the_whole_image)
    loss = _static_loss(tmp_path)
    loss.cores[0].model.multiple = 16
    fixed, moving = torch.rand(1, 1, 20, 40, 300), torch.rand(1, 1, 24, 40, 280)
    fixed_volume, moving_volume = loss.extract(fixed, moving, 0, 0.25)
    assert tiles == [((24, 40, 280), 0), ((24, 40, 280), 256), ((20, 40, 300), 256)]
    torch.testing.assert_close(fixed_volume, torch.cat([fixed * 2.0, fixed * fixed], dim=1))
    assert moving_volume.shape == (1, 2, 24, 40, 280)


def test_static_pca_reduces_each_layer_on_the_fixed_basis(tmp_path: Path) -> None:
    # Each kept layer is projected on its own, onto principal components fitted on the fixed image, as the other
    # engines fit their PCA per layer on the reference side.
    torch.manual_seed(0)
    loss = _static_loss(tmp_path)
    loss.cores[0].pca = 1
    volumes = loss.extract(torch.rand(1, 1, 8, 8, 8), torch.rand(1, 1, 8, 8, 8), 0, 0.25)
    assert [volume.shape[1] for volume in volumes] == [2, 2] and loss._channels == [[1, 1]]


class _Identity(torch.nn.Module):
    """A one-layer feature model taking the published models' inputs."""

    def forward(
        self,
        x: torch.Tensor,
        nb_layer: torch.Tensor,
        stats: torch.Tensor | None = None,
        direction: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        return [x]


def test_an_out_of_memory_past_the_forward_restarts_the_stage_in_tiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FireANTs runs the backward in its own loop, out of the losses' reach: the deformable stage starts over with
    # the level that ran out scored in tiles, instead of escaping to konfai's per-patch re-plan.
    import numpy as np
    import SimpleITK as sitk

    pytest.importorskip("fireants")
    monkeypatch.setattr(fireants, "MIN_TILE", 8)
    path = tmp_path / "features.pt"
    torch.jit.script(_Identity()).save(str(path))
    runs: list[int] = []
    total_field = fireants.FireANTsEngine._total_field

    def backward_out_of_memory_once(engine, reg):
        field = total_field(engine, reg)
        runs.append(1)
        if len(runs) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 3.20 GiB.")
        return field

    monkeypatch.setattr(fireants.FireANTsEngine, "_total_field", backward_out_of_memory_once)
    engine = fireants.FireANTsEngine(
        [1], [1], [1], 3, "mse", 0.01, "none", "none", "syn", "impact", 0.1, 1, 0.5, 1.0, 0,
        [[fireants.ModelSpec(ref=str(path))]], mode="Jacobian",
    )  # fmt: skip
    image = sitk.GetImageFromArray(np.random.default_rng(0).random((32, 32, 32)).astype(np.float32))
    field = engine.register(image, image, -1)
    assert len(runs) == 2 and field.shape == (3, 32, 32, 32)
    assert engine._feature_loss.cores[0]._tiles == {(32, 32, 32): [16, 16, 16]}


def test_an_out_of_memory_that_recurs_reaches_konfai_after_two_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One the loss does not cause (SyN's inversion, the warp) recurs whatever its pieces: the stage reran until they
    # were 16 voxels, five to seven whole SyN runs, before konfai's re-plan saw it.
    import numpy as np
    import SimpleITK as sitk

    pytest.importorskip("fireants")
    monkeypatch.setattr(fireants, "MIN_TILE", 2)
    path = tmp_path / "features.pt"
    torch.jit.script(_Identity()).save(str(path))
    runs: list[int] = []
    total_field = fireants.FireANTsEngine._total_field

    def always_out_of_memory(engine, reg):
        total_field(engine, reg)  # the loss scores its level, which it could narrow again
        runs.append(1)
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 3.20 GiB.")

    monkeypatch.setattr(fireants.FireANTsEngine, "_total_field", always_out_of_memory)
    engine = fireants.FireANTsEngine(
        [1], [1], [1], 3, "mse", 0.01, "none", "none", "syn", "impact", 0.1, 1, 0.5, 1.0, 0,
        [[fireants.ModelSpec(ref=str(path))]], mode="Jacobian",
    )  # fmt: skip
    image = sitk.GetImageFromArray(np.random.default_rng(0).random((32, 32, 32)).astype(np.float32))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        engine.register(image, image, -1)
    assert len(runs) == 3

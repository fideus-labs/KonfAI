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
from typing import Optional

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
    loss._sampling, loss._patch, loss._draw, loss._pooled = 1.0, [], None, None
    loss._statistics, loss._cells, loss._layer_sums, loss._level_weight, loss._place = {}, {}, {}, 0.0, None
    loss._bases = {}
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


class _Averaged(torch.nn.Module):
    """Two layers with a receptive field of 3 voxels: the input averaged over 3^3 voxels, and its square."""

    def forward(
        self,
        x: torch.Tensor,
        nb_layers: torch.Tensor,
        stats: Optional[torch.Tensor] = None,  # noqa: UP045 (TorchScript)
        direction: Optional[torch.Tensor] = None,  # noqa: UP045 (TorchScript)
    ) -> list[torch.Tensor]:
        y = x[:, :1]
        return [torch.nn.functional.avg_pool3d(y, 3, 1, 1), y * y]


def test_a_static_extraction_in_patches_is_the_one_pass_s_for_both_images(tmp_path: Path) -> None:
    # Too large for one pass, or given a patch, every model is predicted by blocks as a segmentation is: both images
    # come back as the one pass gives them, whatever their extents (a fixed image tiled where the moving one is not).
    path = tmp_path / "averaged.pt"
    torch.jit.script(_Averaged()).save(str(path))
    spec = fireants.ModelSpec(ref=str(path), layers_mask="11", distance="L1")
    loss = fireants.ImpactFeatureLoss([[spec]], "Static", False, 5, 0, 0, False)
    generator = torch.Generator().manual_seed(0)
    fixed, moving = torch.rand(1, 1, 40, 36, 30, generator=generator), torch.rand(1, 1, 14, 12, 10, generator=generator)
    whole = loss.extract(fixed, moving, 0)
    patched = loss.extract(fixed, moving, 16)
    for reference, found in zip(whole, patched, strict=True):
        assert found.shape == reference.shape
        torch.testing.assert_close(found, reference, atol=1e-6, rtol=1e-5)


def test_a_static_extraction_the_device_cannot_hold_whole_goes_by_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One pass that runs out of the card's memory is predicted by blocks, which halve until one fits: the one pass's
    # features all the same.
    path = tmp_path / "averaged.pt"
    torch.jit.script(_Averaged()).save(str(path))
    spec = fireants.ModelSpec(ref=str(path), layers_mask="11", distance="L1")
    loss = fireants.ImpactFeatureLoss([[spec]], "Static", False, 5, 0, 0, False)
    generator = torch.Generator().manual_seed(0)
    fixed, moving = torch.rand(1, 1, 40, 36, 30, generator=generator), torch.rand(1, 1, 14, 12, 10, generator=generator)
    model = loss.cores[0].model
    model.fov = [3, 3]  # each layer's field, which the blocks read around them
    whole = loss.extract(fixed, moving, 0)
    volume = model.volume

    def out_of_memory_above_20_cubed(image, *args, **kwargs):
        if image[0, 0].numel() > 20**3:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.10 GiB.")
        return volume(image, *args, **kwargs)

    monkeypatch.setattr(model, "volume", out_of_memory_above_20_cubed)
    for reference, found in zip(whole, loss.extract(fixed, moving, 0), strict=True):
        assert found.shape == reference.shape
        torch.testing.assert_close(found, reference, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not Path("/proc/self/maps").exists(), reason="the process's mappings are read off /proc")
def test_a_static_extraction_by_blocks_maps_no_file_its_directory_removes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Windows cannot remove a file still mapped: the blocks' memmaps are gone before their directory is.
    import tempfile

    directory = tempfile.TemporaryDirectory

    class Checked(directory):  # type: ignore[valid-type, misc]
        def __exit__(self, *args):
            mapped = [line for line in Path("/proc/self/maps").read_text().splitlines() if self.name in line]
            assert not mapped, mapped
            return super().__exit__(*args)

    monkeypatch.setattr(fireants.tempfile, "TemporaryDirectory", Checked)
    path = tmp_path / "averaged.pt"
    torch.jit.script(_Averaged()).save(str(path))
    spec = fireants.ModelSpec(ref=str(path), layers_mask="11", distance="L1")
    loss = fireants.ImpactFeatureLoss([[spec]], "Static", False, 5, 0, 0, False)
    generator = torch.Generator().manual_seed(0)
    loss.extract(
        torch.rand(1, 1, 40, 36, 30, generator=generator), torch.rand(1, 1, 14, 12, 10, generator=generator), 16
    )


def test_static_pca_reduces_each_layer_on_the_fixed_basis(tmp_path: Path) -> None:
    # Each kept layer is projected on its own, onto principal components fitted on the fixed image, as the other
    # engines fit their PCA per layer on the reference side.
    torch.manual_seed(0)
    loss = _static_loss(tmp_path)
    loss.cores[0].pca = 1
    volumes = loss.extract(torch.rand(1, 1, 8, 8, 8), torch.rand(1, 1, 8, 8, 8), 0)
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
        [1], [1], [1], 3, "mse", 0.01, "none", "none", "syn", "impact", 0.1, 0.5, 1.0, 0,
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
        [1], [1], [1], 3, "mse", 0.01, "none", "none", "syn", "impact", 0.1, 0.5, 1.0, 0,
        [[fireants.ModelSpec(ref=str(path))]], mode="Jacobian",
    )  # fmt: skip
    image = sitk.GetImageFromArray(np.random.default_rng(0).random((32, 32, 32)).astype(np.float32))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        engine.register(image, image, -1)
    assert len(runs) == 3

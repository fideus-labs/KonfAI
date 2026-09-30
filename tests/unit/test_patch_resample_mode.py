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


"""``Patch.mode: resample``: a case too large for memory runs whole on a coarser grid, and its outputs come back
onto the case's own grid (linear for images and fields, nearest for labels)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from konfai.data.patching import DatasetPatch
from konfai.data.transform import Resample
from konfai.data.transform.resample import CoarseLabelDilate, coarse_spacing
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import ConfigError
from oracle_support import geometry

_CASE = "CASE_000"
_SHAPE = [20, 24, 28]


def _roundtrip(stage: Resample, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Attribute]:
    attribute = Attribute(geometry([1.0, -2.0, 3.0], [0.5, 0.5, 1.0]))
    assert stage.transform_shape("Volume", _CASE, list(tensor.shape[1:]), attribute) != list(tensor.shape[1:])
    forward = stage(_CASE, tensor, attribute)
    return forward, stage.inverse(_CASE, forward, attribute), attribute


def test_the_coarse_grid_holds_the_budget_and_coarsens_the_finest_axes_first() -> None:
    assert coarse_spacing([10, 10, 10], [1.0, 1.0, 1.0], 1000) is None
    spacing = coarse_spacing(_SHAPE, [0.5, 0.5, 1.0], 20 * 24 * 28 // 8)
    assert spacing[0] == pytest.approx(spacing[1]) and spacing[0] > 0.5
    counts = [round(n * s / t) for n, s, t in zip(reversed(_SHAPE), [0.5, 0.5, 1.0], spacing, strict=True)]
    assert np.prod(counts) <= 20 * 24 * 28 // 8


def test_a_field_comes_back_on_the_case_grid_with_its_values() -> None:
    field = torch.empty(3, *_SHAPE)
    for channel, value in enumerate((2.5, -1.25, 7.0)):
        field[channel] = value  # a displacement in mm: resampling changes its grid, never its values
    forward, back, attribute = _roundtrip(Resample.coarsened(2000), field)
    assert np.prod(forward.shape[1:]) <= 2000
    assert list(back.shape[1:]) == _SHAPE
    torch.testing.assert_close(back, field)
    np.testing.assert_allclose(attribute.get_np_array("Spacing"), [0.5, 0.5, 1.0])
    np.testing.assert_allclose(attribute.get_np_array("Origin"), [1.0, -2.0, 3.0])


def test_a_label_map_comes_back_through_nearest_neighbour() -> None:
    labels = torch.zeros(1, *_SHAPE, dtype=torch.uint8)
    labels[:, :, :, 14:] = 3
    _, back, _ = _roundtrip(Resample.coarsened(2000), labels)
    assert back.dtype == torch.uint8 and set(back.unique().tolist()) <= {0, 3}


def test_a_case_that_fits_is_not_resampled() -> None:
    attribute = Attribute(geometry([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    assert Resample.coarsened(10**6).transform_shape("Volume", _CASE, _SHAPE, attribute) == _SHAPE


def test_the_patch_mode_is_tile_or_resample() -> None:
    assert DatasetPatch([0, 0, 0]).mode == "tile"
    assert DatasetPatch([0, 0, 0], mode="resample", max_voxels=1000).max_voxels == 1000
    with pytest.raises(ConfigError):
        DatasetPatch([0, 0, 0], mode="shrink")
    with pytest.raises(ConfigError):
        DatasetPatch([0, 0, 0], max_voxels=0)


def test_an_unset_max_voxels_is_what_the_device_holds_at_the_declared_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """The GPU's free VRAM (NVML, the card ``--gpu`` names) under KonfAI's margin, or on the CPU its automatic
    memory budget, divided by the peak bytes per voxel the prediction declares for that device."""
    import konfai
    from konfai.utils import budget, vram

    asked: list = []
    monkeypatch.setattr(konfai, "get_vram", lambda devices: asked.append(devices) or (2.0, 10.0))
    monkeypatch.setattr(budget, "available_memory_bytes", lambda: (10 * 2**30, "test"))
    patch = DatasetPatch([0, 0, 0], mode="resample", vram_bytes_per_voxel=1024, ram_bytes_per_voxel=512)
    assert patch.voxel_budget(3) == int(8 * 2**30 * vram.VRAM_BUDGET_SAFETY_FRACTION / 1024) and asked == [[3]]
    assert patch.voxel_budget(None) == int(10 * 2**30 * budget.AUTO_MEMORY_SAFETY_FRACTION / 512)
    assert DatasetPatch([0, 0, 0], max_voxels=1000, vram_bytes_per_voxel=1024).voxel_budget(3) == 1000
    assert DatasetPatch([0, 0, 0], vram_bytes_per_voxel=1024).voxel_budget(None) is None


def test_a_label_thinner_than_the_coarse_spacing_survives_the_coarsening() -> None:
    """Sampled by nearest neighbour at 4x, a one-voxel sheet missed every sample: the coarse mask was empty, and the
    registration engines return a zero field for an empty mask."""
    sheet = torch.zeros(1, 16, 16, 16, dtype=torch.uint8)
    sheet[..., 0] = 1  # x = 0, never the nearest voxel of a coarse sample (1.5, 5.5, ...)
    attribute = Attribute(geometry([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    dilate, stage = CoarseLabelDilate(64), Resample.coarsened(64)
    for transform in (dilate, stage):
        transform.transform_shape("Volume", _CASE, [16, 16, 16], attribute)
    coarse = stage(_CASE, dilate(_CASE, sheet, attribute), attribute)
    assert list(coarse.shape[1:]) == [4, 4, 4]
    assert coarse[..., 0].all() and not coarse[..., 1:].any()
    image = torch.rand(1, 16, 16, 16)
    assert dilate(_CASE, image, attribute) is image  # an image is not a label map


def test_a_store_is_read_from_its_finest_level_still_finer_than_the_target_and_answered_on_level_0(
    tmp_path: Path,
) -> None:
    pytest.importorskip("zarr")
    from konfai.predictor.workflow import Predictor

    root = tmp_path / "Dataset"
    written = Dataset(root, "omezarr", scale_factors=[2, 2])
    written.write("Volume", _CASE, np.zeros((1, 16, 16, 16), np.float32), geometry([0.0] * 3, [1.0] * 3))
    dataset = Dataset(root, "omezarr")
    fake = SimpleNamespace(
        dataset=SimpleNamespace(datasets={str(root): dataset}, groups_src={"Volume": {}}), _declared_levels={}
    )
    natives = Predictor._read_coarse_levels(fake, 16**3)  # fits: level 0
    assert dataset.level == 0 and not natives
    natives = Predictor._read_coarse_levels(fake, 5**3)  # a 2.9 mm target: level 1 (2 mm), not level 2 (4 mm)
    assert dataset.level == 1 and list(natives[("Volume", _CASE)].size_zyx) == [16, 16, 16]
    Predictor._read_coarse_levels(fake, 3**3)  # a 4.6 mm target: level 2
    assert dataset.level == 2
    # Read at level 2, the case comes back onto its level-0 grid.
    shape, attribute = dataset.get_infos("Volume", _CASE)
    stage = Resample.coarsened(3**3)
    stage._native = {_CASE: natives[("Volume", _CASE)]}
    stage.transform_shape("Volume", _CASE, list(shape[-3:]), attribute)
    forward = stage(_CASE, torch.ones(1, *shape[-3:]), attribute)
    back = stage.inverse(_CASE, forward, attribute)
    assert list(back.shape[1:]) == [16, 16, 16]
    np.testing.assert_allclose(attribute.get_np_array("Spacing"), [1.0, 1.0, 1.0])

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

"""``Crop(source=...)``: the box of another group of the case, grown by a margin in mm."""

import gc
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

sitk = pytest.importorskip("SimpleITK")

from konfai import api  # noqa: E402
from konfai.data.transform import Crop, Write  # noqa: E402
from konfai.utils.dataset import Attribute, Dataset  # noqa: E402
from konfai.utils.errors import TransformError  # noqa: E402

# The synthetic case of the fixture, in nibabel's (i, j, k) order: a label map and its CT on one grid.
SHAPE_IJK = (30, 36, 40)
ZOOMS = (0.8, 0.7, 1.25)  # 20 / float32(0.8) truncates to 24 voxels where float64 gives 25

# The boxes of the mask ``np.isin(labels, ...)`` (all non-zero for None) on that case, grown by
# ``int(m / float32(zoom))`` voxels per axis and clipped (half-open, (i, j, k)), computed outside KonfAI.
# Pasting ones back into zeros fills exactly the box's voxels.
MASK_BOXES = [
    ([3, 7], 20.0, [[0, 30], [0, 36], [0, 40]]),
    ([3, 7], 10.0, [[0, 30], [0, 34], [0, 33]]),
    ([9], 20.0, [[0, 27], [2, 36], [14, 40]]),
    ([3], 3.0, [[1, 12], [4, 24], [3, 14]]),
    (None, 0.0, [[1, 28], [2, 34], [5, 33]]),
]


def _labels_ijk() -> np.ndarray:
    labels = np.zeros(SHAPE_IJK, np.uint8)
    labels[4:9, 8:20, 5:12] = 3
    labels[10:28, 2:6, 20:25] = 7
    labels[1:3, 30:34, 30:33] = 9
    return labels


def _image(array_ijk: np.ndarray) -> "sitk.Image":
    image = sitk.GetImageFromArray(np.ascontiguousarray(array_ijk.transpose(2, 1, 0)))  # sitk reads (k, j, i)
    image.SetSpacing(ZOOMS)
    image.SetOrigin((12.5, -30.25, 7.0))
    image.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))  # RAS identity affine, in LPS
    return image


def _write_case(root: Path, stored: str) -> "sitk.Image":
    ct_ijk = (np.arange(np.prod(SHAPE_IJK)).reshape(SHAPE_IJK) % 1000 - 500).astype(np.int16)
    (root / "P000").mkdir(parents=True)
    ct = _image(ct_ijk)
    sitk.WriteImage(ct, str(root / "P000" / f"CT.{stored}"))
    sitk.WriteImage(_image(_labels_ijk()), str(root / "P000" / f"total.{stored}"))
    return ct


@pytest.mark.parametrize(("labels", "margin_mm", "box_ijk"), MASK_BOXES)
@pytest.mark.parametrize(("stored", "verdict"), [("mha", "STREAM"), ("mhd", "WHOLE-VOLUME")])
def test_crop_to_another_groups_box_grown_by_a_margin_in_mm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, labels, margin_mm, box_ijk, stored, verdict
) -> None:
    """Streamed or whole, the CT is cut to the box of the other group's labels grown by the margin,
    with ITK's ``RegionOfInterest`` header."""
    monkeypatch.chdir(tmp_path)
    ct = _write_case(tmp_path / "Raw", stored)
    crop = Crop(source="total", labels=labels, margin_mm=margin_mm, inverse=False)
    chains = {"CT": {"CT": [crop, Write(dataset=f"./Out:{stored}")]}}

    plan = api.plan_transform("PLAN", f"./Raw:{stored}", chains, transforms_dir=tmp_path / "T", quiet=True)
    assert [entry.verdict for entry in plan.entries] == [verdict]
    api.transform("CROP", f"./Raw:{stored}", chains, transforms_dir=tmp_path / "T", quiet=True)

    expected = sitk.RegionOfInterest(ct, [b - a for a, b in box_ijk], [a for a, _ in box_ijk])
    written = sitk.ReadImage(str(tmp_path / "Out" / "P000" / f"CT.{stored}"))
    assert written.GetSize() == expected.GetSize()
    assert written.GetOrigin() == pytest.approx(expected.GetOrigin(), abs=1e-5)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(written), sitk.GetArrayFromImage(expected))


def test_a_source_on_another_grid_is_carried_through_physical_space(tmp_path: Path) -> None:
    """A mask at 2 mm whose origin is half a voxel off the CT's 1 mm grid: its box's outer faces, carried
    into the CT's index space, take the CT voxels whose centres they hold."""
    ct = sitk.GetImageFromArray(np.zeros((20, 22, 24), np.float32))
    mask = np.zeros((10, 11, 12), np.uint8)
    mask[1:3, 2:5, 3:6] = 1  # (z, y, x) indices 1..2, 2..4, 3..5
    total = sitk.GetImageFromArray(mask)
    total.SetSpacing((2.0, 2.0, 2.0))
    total.SetOrigin((0.5, 0.5, 0.5))  # centre n at 0.5 + 2n, faces 2n - 0.5 .. 2n + 1.5
    (tmp_path / "P000").mkdir()
    sitk.WriteImage(ct, str(tmp_path / "P000" / "CT.mha"))
    sitk.WriteImage(total, str(tmp_path / "P000" / "total.mha"))
    dataset = Dataset(tmp_path, "mha")
    _, attribute = dataset.read_data("CT", "P000")

    crop = Crop(source="total", margin_mm=2.0)
    crop.set_datasets([dataset])
    shape = crop.transform_shape("CT", "P000", [20, 22, 24], attribute)

    # Faces 1.5..5.5, 3.5..9.5, 5.5..11.5 hold the CT centres 2..5, 4..9, 6..11; 2 mm more each side.
    near = Crop._parse_box(attribute["box"])[:, 0]
    assert near.tolist() == [0, 2, 4]
    assert shape == [8, 10, 10]


def test_a_tilted_source_on_the_same_grid_keeps_its_box_index_for_index(tmp_path: Path) -> None:
    """A CT tilted 12.5 degrees, as MandibleCT Pat_2: a hull in the world between the two grids would grow the box."""
    angle = np.deg2rad(12.5)
    direction = (1.0, 0.0, 0.0, 0.0, np.cos(angle), -np.sin(angle), 0.0, np.sin(angle), np.cos(angle))
    mask = np.zeros((20, 40, 30), np.uint8)
    mask[5:9, 10:25, 8:12] = 1
    (tmp_path / "P000").mkdir()
    for name, array in (("CT", np.zeros(mask.shape, np.float32)), ("total", mask)):
        image = sitk.GetImageFromArray(array)
        image.SetDirection(direction)
        image.SetSpacing((0.5, 0.5, 2.0))
        sitk.WriteImage(image, str(tmp_path / "P000" / f"{name}.mha"))
    dataset = Dataset(tmp_path, "mha")
    _, attribute = dataset.read_data("CT", "P000")

    crop = Crop(source="total")
    crop.set_datasets([dataset])
    assert crop.transform_shape("CT", "P000", list(mask.shape), attribute) == [4, 15, 4]
    assert Crop._parse_box(attribute["box"])[:, 0].tolist() == [5, 10, 8]


def test_the_inverse_pastes_back_into_the_fill_and_the_stored_header(tmp_path: Path) -> None:
    """Forward then inverse: the case's own extent and header, the cropped values in place and the
    ``fill`` around them."""
    _write_case(tmp_path, "mha")
    dataset = Dataset(tmp_path, "mha")
    data, attribute = dataset.read_data("CT", "P000")
    stored = Attribute(attribute)
    crop = Crop(source="total", labels=[3], margin_mm=3.0, fill=0)
    crop.set_datasets([dataset])

    crop.transform_shape("CT", "P000", list(data.shape[1:]), attribute)
    out = crop("P000", torch.ones((1, *data.shape[1:]), dtype=torch.uint8), attribute)
    back = crop.inverse("P000", out, attribute)

    expected = np.zeros(data.shape, np.uint8)
    box_ijk = MASK_BOXES[3][2]
    expected[(slice(None), *(slice(a, b) for a, b in reversed(box_ijk)))] = 1
    np.testing.assert_array_equal(back.numpy(), expected)
    assert attribute == stored


def test_crop_source_refusals(tmp_path: Path) -> None:
    _write_case(tmp_path, "mha")
    dataset = Dataset(tmp_path, "mha")
    _, attribute = dataset.read_data("CT", "P000")
    with pytest.raises(TransformError, match="margin"):
        Crop(margin_mm=-1.0)
    missing = Crop(source="lungs")
    missing.set_datasets([dataset])
    with pytest.raises(TransformError, match="no dataset of the run holds it"):
        missing.transform_shape("CT", "P000", [40, 36, 30], Attribute(attribute))
    empty = Crop(source="total", labels=[42])
    empty.set_datasets([dataset])
    with pytest.raises(TransformError, match=r"no voxel holding one of the labels \[42\]\.\n.*empty: fill"):
        empty.transform_shape("CT", "P000", [40, 36, 30], Attribute(attribute))


@pytest.mark.parametrize("stored", ["mha", "mhd"])
def test_an_empty_crop_under_empty_fill_hands_on_the_near_corner_voxel_streamed_or_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stored: str
) -> None:
    """No voxel of the source holds a label: under ``empty: fill`` the stages after it get the box of the near
    corner voxel grown by the margin (16, 28 and 24 voxels along k, j, i), with the case's own header, on either
    route."""
    monkeypatch.chdir(tmp_path)
    ct = _write_case(tmp_path / "Raw", stored)
    crop = Crop(source="total", labels=[42], margin_mm=20.0, fill=0, empty="fill", inverse=False)
    chains = {"CT": {"CT": [crop, Write(dataset=f"./Out:{stored}")]}}
    api.transform("CROP", f"./Raw:{stored}", chains, transforms_dir=tmp_path / "T", quiet=True)

    written = sitk.ReadImage(str(tmp_path / "Out" / "P000" / f"CT.{stored}"))
    assert written.GetSize() == (25, 29, 17)
    assert written.GetOrigin() == pytest.approx(ct.GetOrigin(), abs=1e-5)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(written), sitk.GetArrayFromImage(ct)[:17, :29, :25])


def test_the_inverse_of_an_empty_crop_under_empty_fill_is_the_fill_on_the_whole_grid(tmp_path: Path) -> None:
    """An empty crop mask under ``empty: fill`` comes back as the fill on the CT's grid: the inverse drops what
    the stages after the Crop computed on its voxel."""
    _write_case(tmp_path, "mha")
    dataset = Dataset(tmp_path, "mha")
    data, attribute = dataset.read_data("CT", "P000")
    stored = Attribute(attribute)
    crop = Crop(source="total", labels=[42], margin_mm=20.0, fill=0, empty="fill")
    crop.set_datasets([dataset])

    assert crop.transform_shape("CT", "P000", list(data.shape[1:]), attribute) == [17, 29, 25]
    out = crop("P000", torch.from_numpy(data), attribute)
    np.testing.assert_array_equal(out.numpy(), data[:, :17, :29, :25])
    back = crop.inverse("P000", torch.full((2, 17, 29, 25), 7, dtype=torch.uint8), attribute)
    np.testing.assert_array_equal(back.numpy(), np.zeros((2, *data.shape[1:]), np.uint8))
    assert attribute == stored
    with pytest.raises(TransformError, match="needs fill to be a value"):
        Crop(source="total", empty="fill")
    with pytest.raises(TransformError, match="empty is refuse"):
        Crop(source="total", fill=0, empty="zeros")


def test_a_handed_box_is_its_own_chains_when_two_chains_share_their_stage_names() -> None:
    # CT -> CT_1mm and CT -> CT_2mm: the same class names before the Crop, configured otherwise. The box is
    # remembered per chain, not per sequence of class names, so each chain crops its own volume's foreground.
    class KeepHalf:
        def __init__(self, first: bool) -> None:
            self.first = first

        def transform_shape(self, group: str, name: str, shape: list[int], attribute: object) -> list[int]:
            return shape

        def case_working_multiple(self, name: str) -> float:
            return 1.0

        def __call__(self, name: str, tensor: torch.Tensor, attribute: object) -> torch.Tensor:
            kept = torch.zeros_like(tensor)
            half = tensor.shape[1] // 2
            span = slice(0, half) if self.first else slice(half, None)
            kept[:, span] = tensor[:, span]
            return kept

    volume = np.ones((1, 8, 4, 4), dtype=np.float32)
    holder = SimpleNamespace(
        case_facts={},
        read_data=lambda group, name: (volume, Attribute()),
        get_infos=lambda group, name: ([1, 8, 4, 4], Attribute()),
    )
    # A freed chain's stage often lends its address to the next one's: remembered by id, one pair in four got the
    # first chain's box.
    for _ in range(30):
        holder.case_facts.clear()
        boxes = []
        for first in (True, False):
            crop = Crop(nonzero=True, handed=True)
            crop.preceding = [KeepHalf(first)]
            boxes.append(crop._handed_box(cast(Any, holder), "CT", "P000", [8, 4, 4]))
            del crop
            gc.collect()
        assert boxes[0][0].tolist() == [0, 3] and boxes[1][0].tolist() == [4, 7]


def test_a_handed_replay_over_the_memory_budget_is_refused_before_the_read(monkeypatch: pytest.MonkeyPatch) -> None:
    import konfai.utils.budget as budget

    monkeypatch.setattr(budget, "per_rank_budget_bytes", lambda: 64.0**3 * 4)  # one float32 64^3 volume

    def unread(group: str, name: str) -> None:
        raise AssertionError("read before it was priced")

    holder = SimpleNamespace(case_facts={}, read_data=unread, get_infos=lambda group, name: ([1, 8, 8, 8], Attribute()))
    crop = Crop(nonzero=True, handed=True)
    crop.preceding = []
    # Stored at 8^3, handed at 64^3 (a Resample before it enlarged it): the larger extent is what the replay holds.
    with pytest.raises(TransformError, match="memory budget"):
        crop._handed_box(cast(Any, holder), "CT", "P000", [64, 64, 64])

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

"""``Warp`` resamples a case through a displacement field, region by region.

The claim under test is the one that matters for a volume larger than memory: the streamed result
equals the whole-volume one, and the declared displacement bound is verified rather than trusted."""

from pathlib import Path

import numpy as np
import pytest
import torch
from konfai.data.patching import DatasetManager
from konfai.data.transform import LocalityKind, RegionContext, Save, Warp
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import TransformError

pytest.importorskip("SimpleITK")

SPACING = (2.0, 1.0, 1.0)  # (x, y, z) SimpleITK order


def _attributes() -> Attribute:
    attribute = Attribute()
    attribute["Origin"] = np.asarray([0.0, 0.0, 0.0])
    attribute["Spacing"] = np.asarray(list(SPACING))
    attribute["Direction"] = np.eye(3, dtype=np.float64).reshape(-1)
    return attribute


def _fixture(
    tmp_path: Path, shift_um: tuple[float, float, float] = (0.0, 0.0, 4.0)
) -> tuple[Dataset, Dataset, np.ndarray]:
    """A case and a CONSTANT displacement field: a constant shift is the one warp whose answer can
    be written down independently, which is what makes it a reference rather than a re-run."""
    rng = np.random.default_rng(0)
    volume = (rng.random((1, 10, 12, 14)) * 100).astype(np.float32)
    source = Dataset(tmp_path / "src", "h5")
    source.write("CT", "CASE_000", volume, _attributes())

    field = np.zeros((3, 10, 12, 14), dtype=np.float32)
    for component, value in enumerate(shift_um):  # component order is (x, y, z)
        field[component] = value
    fields = Dataset(tmp_path / "dvf", "h5")
    fields.write("DVF", "CASE_000", field, _attributes())
    return source, fields, volume


def _manager(source: Dataset, transforms: list) -> DatasetManager:
    return DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=source,
        patch=None,
        transforms=transforms,
        data_augmentations_list=[],
    )


def test_declares_a_halo_sized_by_the_declared_displacement() -> None:
    warp = Warp(field="./x:h5", group="DVF", max_displacement=4.0)
    locality = warp.patch_locality(_attributes())
    # Spacing is (x=2, y=1, z=1) so in array order (z, y, x) it is (1, 1, 2): 4 um is 4, 4 and 2 voxels.
    assert locality.kind is LocalityKind.HALO
    assert locality.halo == (4, 4, 2)


def test_without_a_declared_bound_it_refuses_to_stream() -> None:
    """The safety net: an undeclared reach is an unbounded read, so the whole volume it is."""
    assert Warp(field="./x:h5", group="DVF").patch_locality(_attributes()).kind is LocalityKind.WHOLE_VOLUME
    assert Warp(field="./x:h5", group="DVF", max_displacement=4.0).patch_locality(Attribute()).kind is (
        LocalityKind.WHOLE_VOLUME
    )


def test_a_constant_shift_moves_the_volume_by_that_many_voxels(tmp_path: Path) -> None:
    _source, _fields, volume = _fixture(tmp_path, shift_um=(0.0, 0.0, 3.0))
    warp = Warp(field=f"{tmp_path / 'dvf'}:h5", group="DVF", max_displacement=3.0)

    moved = warp("CASE_000", torch.from_numpy(volume), _attributes()).numpy()

    # d = +3 um along z, spacing z = 1 um: output(z) = input(z + 3).
    np.testing.assert_allclose(moved[:, :-3], volume[:, 3:], rtol=1e-5, atol=1e-4)


def test_streamed_equals_whole_volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The claim that matters: the same answer, region by region, with several regions."""
    from konfai.data import patching as patching_module

    monkeypatch.setattr(patching_module, "_SWEEP_SLAB_ROWS", 3)
    source, _fields, volume = _fixture(tmp_path, shift_um=(1.0, 2.0, 3.0))
    warp = Warp(field=f"{tmp_path / 'dvf'}:h5", group="DVF", max_displacement=4.0)

    reference = warp("CASE_000", torch.from_numpy(volume), _attributes()).numpy()

    manager = _manager(source, [warp, Save(f"{tmp_path / 'out'}:h5")])
    assert manager.can_stream_patch(0, apply_augmentations=False)
    assert manager.materialize() is True
    streamed, _ = Dataset(tmp_path / "out", "h5").read_data("CT", "CASE_000")

    np.testing.assert_allclose(streamed, reference, rtol=1e-5, atol=1e-4)


def test_a_field_beyond_the_declared_bound_raises(tmp_path: Path) -> None:
    """Declared, then verified: sampling outside what was read would show as a dark rim and nothing
    else, so the mismatch is raised instead."""
    _source, _fields, volume = _fixture(tmp_path, shift_um=(0.0, 0.0, 9.0))
    warp = Warp(field=f"{tmp_path / 'dvf'}:h5", group="DVF", max_displacement=1.0)

    with pytest.raises(TransformError, match="beyond the declared"):
        whole = (slice(0, 10), slice(0, 12), slice(0, 14))
        warp.stream_region(
            "CASE_000",
            torch.from_numpy(volume),
            RegionContext(whole, whole, (10, 12, 14), (10, 12, 14)),
            _attributes(),
        )


def test_a_field_with_the_wrong_component_count_is_named(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    Dataset(tmp_path / "src", "h5").write("CT", "CASE_000", rng.random((1, 4, 4, 4)).astype(np.float32), _attributes())
    Dataset(tmp_path / "dvf", "h5").write("DVF", "CASE_000", np.zeros((2, 4, 4, 4), np.float32), _attributes())
    warp = Warp(field=f"{tmp_path / 'dvf'}:h5", group="DVF", max_displacement=1.0)

    with pytest.raises(TransformError, match="component"):
        warp("CASE_000", torch.zeros(1, 4, 4, 4), _attributes())


def test_warp_without_a_field_is_refused_at_construction() -> None:
    with pytest.raises(TransformError, match="needs a 'field'"):
        Warp(field="")


def test_unknown_interpolation_is_refused_at_construction() -> None:
    with pytest.raises(TransformError, match="unknown interpolation"):
        Warp(field="./x:h5", interpolation="cubic")

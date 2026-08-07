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

"""``Resample`` through a displacement field alone: a warp on the case's own grid, region by region.

The claim under test is the one that matters for a volume larger than memory: the streamed result
equals the whole-volume one, and each region's window is sized from the field values it reads."""

from pathlib import Path

import numpy as np
import pytest
import torch
from konfai.data.patching import DatasetManager
from konfai.data.transform import LocalityKind, Resample, Save
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import TransformError
from konfai.utils.ome_zarr import _zarr_v3_available

pytest.importorskip("SimpleITK")

# A field records its bound in an RFC-5 store, which is zarr v3 (NGFF >= 0.5): and zarr 2.x, the
# newest release for Python 3.10, cannot write one. Everything else here reads an h5 field.
_needs_rfc5 = pytest.mark.skipif(
    not _zarr_v3_available(),
    reason="a displacement field's bound is recorded in a zarr v3 store (zarr>=3, Python>=3.11)",
)

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


def _recorded(warp: Resample, attribute: Attribute | None = None, shape: tuple[int, ...] = (10, 12, 14)) -> Resample:
    """A stage that has met its case, which is when a region can be asked about at all."""
    warp.transform_shape("CT", "CASE_000", list(shape), attribute if attribute is not None else _attributes())
    return warp


def test_the_source_region_is_the_target_grown_by_the_field_reach(tmp_path: Path) -> None:
    """A warp is a regrid onto the case's own grid, and its window is the field's reach in voxels.

    Spacing is (x=2, y=1, z=1), so in array order (z, y, x) 4 um of displacement is 4, 4 and 2
    voxels: plus the one voxel the linear taps reach. Declared as REGRID and not HALO because the
    window is derived from the case's GEOMETRY: see the oblique case below, which a per-axis halo
    cannot express at all.
    """
    _source, _fields, _volume = _fixture(tmp_path, shift_um=(4.0, 4.0, 4.0))
    warp = _recorded(Resample(field=f"{tmp_path / 'dvf'}:h5", field_group="DVF"))
    assert warp.patch_locality(_attributes()).kind is LocalityKind.REGRID

    target = (slice(4, 6), slice(4, 6), slice(4, 6))
    window = warp.measured_region_source("CASE_000", target, [10, 12, 14], _attributes())

    # The rule, written out: the region's OUTER faces (start - 0.5 .. stop - 0.5) in world units,
    # grown by the field's 4 um, back to indices, floor/ceil, one voxel of margin for the taps.
    extents, per_voxel = (10, 12, 14), (1.0, 1.0, 2.0)  # array order (z, y, x)
    expected = []
    for axis, extent in enumerate(extents):
        reach = 4.0 / per_voxel[axis]
        low, high = 4 - 0.5 - reach, 6 - 0.5 + reach
        expected.append((max(0, int(np.floor(low)) - 1), min(extent, int(np.ceil(high)) + 2)))
    assert [(part.start, part.stop) for part in window] == expected


def test_an_oblique_case_grows_its_window_on_every_axis(tmp_path: Path) -> None:
    """The bug a per-axis halo hid: a displacement along x reaches into y and z when the axes turn.

    A world bound converted to a halo per ARRAY axis silently assumes the direction cosines are
    the identity, on a turned case the window is short on the axes the displacement actually
    reaches, and a short window returns the border value rather than raising.
    """
    turned = _attributes()
    angle = np.deg2rad(35.0)
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    turned["Direction"] = np.asarray([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]]).reshape(-1)

    _source, _fields, _volume = _fixture(tmp_path, shift_um=(4.0, 0.0, 0.0))
    warp = _recorded(Resample(field=f"{tmp_path / 'dvf'}:h5", field_group="DVF"), turned)
    target = (slice(5, 6), slice(5, 6), slice(5, 6))
    window = warp.measured_region_source("CASE_000", target, [10, 12, 14], turned)

    widths = [part.stop - part.start for part in window]
    assert all(width > 1 for width in widths), f"a turned case reaches on every axis, got {widths}"


def test_the_header_scan_survives_an_unreadable_entry_in_the_field_group(tmp_path: Path) -> None:
    """The whole group is header-read, including entries this run never warps.

    A directory store lists its entries from the filesystem alone, so a corrupt one is only met at
    its header, and the planner reads them all before any case is chosen. Left to raise, one bad
    file anywhere under the field root aborts the run with a SimpleITK error instead of the
    whole-volume answer the same guard promises for a field dataset it cannot read.
    """
    fields = Dataset(tmp_path / "dvf", "mha")
    field = np.zeros((3, 4, 5, 6), dtype=np.float32)
    for case in ("CASE_000", "CASE_001"):
        fields.write("DVF", case, field, _attributes())
    # The first entry the scan meets, so the read reaches it before any bound-less entry ends the scan.
    (tmp_path / "dvf" / "CASE_000" / "DVF.mha").write_bytes(b"not an image")

    locality = Resample(field=f"{tmp_path / 'dvf'}:mha", field_group="DVF").patch_locality(_attributes())

    assert locality.kind is LocalityKind.WHOLE_VOLUME


def test_a_field_with_no_bound_still_streams_with_windows_measured_at_run(tmp_path: Path) -> None:
    """A bound-less field is not a whole-volume answer: the field window a region samples is read
    for sampling regardless, and the sup of those very values sizes that region's source pull: per region, so a quiet slab pays a quiet halo where the shifted one pays its shift."""
    _source, _fields, _volume = _fixture(tmp_path, shift_um=(4.0, 0.0, 0.0))  # 4 um along x alone
    warp = _recorded(Resample(field=f"{tmp_path / 'dvf'}:h5", field_group="DVF"))

    assert warp.patch_locality(_attributes()).kind is LocalityKind.REGRID

    target = (slice(4, 6), slice(4, 6), slice(4, 6))
    priced = warp.stream_region_source("CASE_000", target, [10, 12, 14], _attributes())
    measured = warp.measured_region_source("CASE_000", target, [10, 12, 14], _attributes())

    # The plan prices as if the field were zero: the target's outer faces plus the taps' voxel.
    assert [(part.start, part.stop) for part in priced] == [(2, 8), (2, 8), (2, 8)]
    # The run pays the shift the values actually hold: 4 um at spacing 2 is 2 voxels, on x alone.
    assert [(part.start, part.stop) for part in measured] == [(2, 8), (2, 8), (0, 10)]


def test_sizing_and_sampling_share_one_field_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The window that sizes a region's pull is the window the sampler needs next: one read."""
    _source, _fields, _volume = _fixture(tmp_path)
    warp = _recorded(Resample(field=f"{tmp_path / 'dvf'}:h5", field_group="DVF"))
    displacement = warp.displacement
    assert displacement is not None
    reads: list[int] = []
    original = type(displacement).read
    monkeypatch.setattr(
        type(displacement), "read", lambda self, *args, **kwargs: (reads.append(1), original(self, *args, **kwargs))[1]
    )

    target = (slice(2, 5), slice(0, 12), slice(0, 14))
    warp.measured_region_source("CASE_000", target, [10, 12, 14], _attributes())
    _source_grid, target_grid = warp._grids_of("CASE_000")
    warp._stages("CASE_000", target_grid.sub_grid(target))

    assert len(reads) == 1


def test_a_constant_shift_moves_the_volume_by_that_many_voxels(tmp_path: Path) -> None:
    _source, _fields, volume = _fixture(tmp_path, shift_um=(0.0, 0.0, 3.0))
    warp = Resample(field=f"{tmp_path / 'dvf'}:h5", field_group="DVF")

    moved = warp("CASE_000", torch.from_numpy(volume), _attributes()).numpy()

    # d = +3 um along z, spacing z = 1 um: output(z) = input(z + 3).
    np.testing.assert_allclose(moved[:, :-3], volume[:, 3:], rtol=1e-5, atol=1e-4)


def test_streamed_equals_whole_volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The claim that matters: the same answer, region by region, with several regions: and
    nothing was declared to make it true, the windows being measured from the field itself."""
    from konfai.data import patching as patching_module

    monkeypatch.setattr(patching_module, "_SWEEP_SLAB_ROWS", 3)
    source, _fields, volume = _fixture(tmp_path, shift_um=(1.0, 2.0, 3.0))
    warp = Resample(field=f"{tmp_path / 'dvf'}:h5", field_group="DVF")

    reference = warp("CASE_000", torch.from_numpy(volume), _attributes()).numpy()

    manager = _manager(source, [warp, Save(f"{tmp_path / 'out'}:h5")])
    assert manager.can_stream_patch(0, apply_augmentations=False)
    assert manager.materialize() is True
    streamed, _ = Dataset(tmp_path / "out", "h5").read_data("CT", "CASE_000")

    np.testing.assert_allclose(streamed, reference, rtol=1e-5, atol=1e-4)


def test_a_field_with_the_wrong_component_count_is_named(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    Dataset(tmp_path / "src", "h5").write("CT", "CASE_000", rng.random((1, 4, 4, 4)).astype(np.float32), _attributes())
    Dataset(tmp_path / "dvf", "h5").write("DVF", "CASE_000", np.zeros((2, 4, 4, 4), np.float32), _attributes())
    warp = Resample(field=f"{tmp_path / 'dvf'}:h5", field_group="DVF")

    with pytest.raises(TransformError, match="component"):
        warp("CASE_000", torch.zeros(1, 4, 4, 4), _attributes())


def test_a_case_with_no_field_entry_is_refused_at_plan_time(tmp_path: Path) -> None:
    """The cohort scan proves what is on disk; only the per-case probe knows what the plan asks for.

    Without it a missing entry surfaces mid-run, in the field read, after bytes are written.
    """
    _fixture(tmp_path)  # writes a field for CASE_000, and for no other case
    warp = Resample(field=f"{tmp_path / 'dvf'}:h5", field_group="DVF")
    with pytest.raises(TransformError, match="CASE_MISSING"):
        warp.transform_shape("CT", "CASE_MISSING", [10, 12, 14], _attributes())


def test_an_empty_field_path_declares_no_field() -> None:
    """An empty ``field`` with no group is the identity map, not a broken declaration."""
    assert Resample(field="").displacement is None


def test_unknown_interpolation_is_refused_at_construction() -> None:
    with pytest.raises(TransformError, match="unknown interpolation"):
        Resample(field="./x:h5", interpolation="cubic")

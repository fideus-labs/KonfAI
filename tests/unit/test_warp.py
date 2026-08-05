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
from konfai.utils.dataset import DISPLACEMENT_FIELD_ATTRIBUTE, Attribute, Dataset
from konfai.utils.errors import TransformError
from konfai.utils.ome_zarr import _zarr_v3_available

pytest.importorskip("SimpleITK")

# A field records its bound in an RFC-5 store, which is zarr v3 (NGFF >= 0.5) -- and zarr 2.x, the
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


def _recorded(warp: Warp, attribute: Attribute | None = None, shape: tuple[int, ...] = (10, 12, 14)) -> Warp:
    """A stage that has met its case — which is when a region can be asked about at all."""
    warp.transform_shape("CT", "CASE_000", list(shape), attribute if attribute is not None else _attributes())
    return warp


def test_the_source_region_is_the_target_grown_by_the_declared_displacement() -> None:
    """A warp is a regrid onto the case's own grid, and its window is the bound in voxels.

    Spacing is (x=2, y=1, z=1), so in array order (z, y, x) 4 um of displacement is 4, 4 and 2
    voxels -- plus the one voxel the linear taps reach. Declared as REGRID and not HALO because the
    window is derived from the case's GEOMETRY: see the oblique case below, which a per-axis halo
    cannot express at all.
    """
    warp = _recorded(Warp(field="./x:h5", group="DVF", max_displacement=4.0))
    assert warp.patch_locality(_attributes()).kind is LocalityKind.REGRID

    target = (slice(4, 6), slice(4, 6), slice(4, 6))
    window = warp.stream_region_source("CASE_000", target, [10, 12, 14], _attributes())

    # The rule, written out: the region's OUTER faces (start - 0.5 .. stop - 0.5) in world units,
    # grown by the declared 4 um, back to indices, floor/ceil, one voxel of margin for the taps.
    extents, per_voxel = (10, 12, 14), (1.0, 1.0, 2.0)  # array order (z, y, x)
    expected = []
    for axis, extent in enumerate(extents):
        reach = 4.0 / per_voxel[axis]
        low, high = 4 - 0.5 - reach, 6 - 0.5 + reach
        expected.append((max(0, int(np.floor(low)) - 1), min(extent, int(np.ceil(high)) + 2)))
    assert [(part.start, part.stop) for part in window] == expected


def test_an_oblique_case_grows_its_window_on_every_axis() -> None:
    """The bug a per-axis halo hid: a displacement along x reaches into y and z when the axes turn.

    ``Warp`` used to convert a world bound to a halo per ARRAY axis, which silently assumed the
    direction cosines were the identity -- on a turned case the window was short on the axes the
    displacement actually reached, and a short window returns the border value rather than raising.
    """
    turned = _attributes()
    angle = np.deg2rad(35.0)
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    turned["Direction"] = np.asarray([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]]).reshape(-1)

    warp = _recorded(Warp(field="./x:h5", group="DVF", max_displacement=4.0), turned)
    target = (slice(5, 6), slice(5, 6), slice(5, 6))
    window = warp.stream_region_source("CASE_000", target, [10, 12, 14], turned)

    widths = [part.stop - part.start for part in window]
    assert all(width > 1 for width in widths), f"a turned case reaches on every axis, got {widths}"


@_needs_rfc5
def test_auto_reads_the_bound_the_fields_recorded_when_they_were_written(tmp_path: Path) -> None:
    """``max_displacement: auto`` is the number the producer already knew.

    A field records its own per-component bound at write time, so asking the user to measure it is
    asking for something the store can answer from a header. Per component and not one collapsed
    maximum: these grids are anisotropic, and one number over-reads the fine axes.
    """
    fields = Dataset(tmp_path / "dvf", "omezarr")
    for case, shift in (("CASE_000", (1.0, 2.0, 3.0)), ("CASE_001", (0.5, 6.0, 1.0))):
        field = np.zeros((3, 4, 5, 6), dtype=np.float32)
        for component, value in enumerate(shift):
            field[component] = value
        attribute = _attributes()
        attribute[DISPLACEMENT_FIELD_ATTRIBUTE] = "true"
        fields.write("DVF", case, field, attribute)

    warp = Warp(field=f"{tmp_path / 'dvf'}:omezarr", group="DVF", max_displacement="auto")
    locality = warp.patch_locality(_attributes())

    # The cohort's bound is (x=1.0, y=6.0, z=3.0); spacing in array order (z, y, x) is (1, 1, 2), so
    # the window grows by 3, 6 and 1 voxels (plus the linear taps' one) around its target.
    assert locality.kind is LocalityKind.REGRID
    window = _recorded(warp).stream_region_source("CASE_000", (slice(4, 6),) * 3, [10, 12, 14], _attributes())
    reaches = (3.0, 6.0, 0.5)  # array order (z, y, x): the bound divided by that axis's spacing
    starts = [max(0, int(np.floor(4 - 0.5 - reach)) - 1) for reach in reaches]
    assert [part.start for part in window] == starts


def test_auto_survives_an_unreadable_entry_in_the_field_group(tmp_path: Path) -> None:
    """The whole group is header-read, including entries this run never warps.

    A directory store lists its entries from the filesystem alone, so a corrupt one is only met at
    its header -- and the planner reads them all before any case is chosen. Left to raise, one bad
    file anywhere under the field root aborts the run with a SimpleITK error instead of the
    whole-volume answer the same guard promises for a field dataset it cannot read.
    """
    fields = Dataset(tmp_path / "dvf", "mha")
    field = np.zeros((3, 4, 5, 6), dtype=np.float32)
    for case in ("CASE_000", "CASE_001"):
        fields.write("DVF", case, field, _attributes())
    # The first entry the scan meets, so the read reaches it before any bound-less entry ends the scan.
    (tmp_path / "dvf" / "CASE_000" / "DVF.mha").write_bytes(b"not an image")

    locality = Warp(field=f"{tmp_path / 'dvf'}:mha", group="DVF", max_displacement="auto").patch_locality(_attributes())

    assert locality.kind is LocalityKind.WHOLE_VOLUME


def test_auto_falls_back_to_the_whole_volume_when_no_field_recorded_a_bound(tmp_path: Path) -> None:
    """Only an OME-Zarr field KonfAI wrote carries the bound, so `auto` must answer for the rest."""
    _source, _fields, _volume = _fixture(tmp_path)  # an h5 field: no bound recorded

    locality = Warp(field=f"{tmp_path / 'dvf'}:h5", group="DVF", max_displacement="auto").patch_locality(_attributes())

    assert locality.kind is LocalityKind.WHOLE_VOLUME
    assert locality.reason is not None and "no recorded bound" in locality.reason


def test_a_max_displacement_that_is_neither_a_number_nor_auto_is_refused() -> None:
    with pytest.raises(TransformError, match="neither a number nor 'auto'"):
        Warp(field="./x:h5", group="DVF", max_displacement="lots")


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
    else, so the mismatch is raised instead.

    Checked per component, the way the halo is derived: a field under the collapsed maximum can
    still exceed the bound on one axis, and that axis is the one whose halo was too small.
    """
    _source, _fields, volume = _fixture(tmp_path, shift_um=(0.0, 0.0, 9.0))
    warp = Warp(field=f"{tmp_path / 'dvf'}:h5", group="DVF", max_displacement=1.0)

    _recorded(warp)
    with pytest.raises(TransformError, match=r"on component 2, beyond the 1\.000"):
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

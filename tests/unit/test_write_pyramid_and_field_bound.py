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

"""The write side of the data surface: a declared pyramid, and a field that records its own bound.

Both are properties a CONSUMER depends on and cannot re-derive cheaply. A pyramid is indexed by
position (``:omezarr@1``), so a producer that writes one level where two were promised does not fail
-- it resolves ``@1`` to something else, or to nothing. A field's bound sizes the region a warp must
read, and measuring it after the fact means scanning a volume that exists region by region precisely
because it does not fit.

So each test here asserts the two write paths agree: assembled in memory, and streamed region by
region. They are different code (``write_ome_zarr`` against ``create_ome_zarr_store`` +
``append_ome_zarr_levels``), and the streamed one is the path a real volume takes.
"""

from pathlib import Path

import numpy as np
import pytest
import zarr
from konfai.utils.dataset import DISPLACEMENT_FIELD_ATTRIBUTE, Attribute, Dataset
from konfai.utils.errors import DatasetManagerError
from konfai.utils.ome_zarr import (
    DISPLACEMENT_BOUND_ATTRIBUTE,
    _read_konfai_attributes,
    _zarr_v3_available,
    append_ome_zarr_levels,
    get_ome_zarr_info,
    is_displacement_field,
    write_ome_zarr,
)


def _levels(store) -> int:
    attributes = dict(zarr.open_group(str(store), mode="r").attrs)
    return len(attributes.get("ome", attributes)["multiscales"][0]["datasets"])


def _only_store(root):
    return next(root.rglob("*.ome.zarr"))


def _geometry() -> Attribute:
    attributes = Attribute()
    attributes["Spacing"] = np.array([2.0, 1.0, 1.0])
    attributes["Origin"] = np.array([0.0, 0.0, 0.0])
    return attributes


def test_a_declared_pyramid_is_written_by_both_paths_and_they_agree(tmp_path):
    """``scale_factors`` reaches the store from the dataset, streamed or not, with the same pixels.

    The streamed path cannot take ``scale_factors`` at creation -- no level exists until the last
    region lands -- so it derives them at finalize instead. That is different code, and the levels it
    produces have to be the same ones, or a chain's output would depend on whether it happened to
    stream.
    """
    data = np.arange(1 * 16 * 16 * 16, dtype=np.float32).reshape(1, 16, 16, 16)

    Dataset(tmp_path / "whole", "omezarr", scale_factors=[4]).write("G", "case", data, _geometry())
    whole = _only_store(tmp_path / "whole")

    streamed_dataset = Dataset(tmp_path / "streamed", "omezarr", scale_factors=[4])
    stream = streamed_dataset.open_data_stream("G", "case", [1, 16, 16, 16], np.dtype("float32"), _geometry())
    assert stream is not None
    with stream:
        for start in range(0, 16, 4):
            stream.write_slice(
                (slice(0, 1), slice(start, start + 4), slice(0, 16), slice(0, 16)), data[:, start : start + 4]
            )
    streamed = _only_store(tmp_path / "streamed")

    assert _levels(whole) == 2
    assert _levels(streamed) == 2
    # Level 0 must survive deriving the levels above it: append rewrites the whole multiscales, and an
    # earlier version of that truncated the store before dask had pulled a single tile, leaving a
    # uniformly zero pyramid with correct metadata and no error.
    back, _ = Dataset(tmp_path / "streamed", "omezarr").read_data("G", "case")
    assert np.array_equal(np.asarray(back, dtype=np.float32).reshape(data.shape), data)

    coarse_whole, _ = Dataset(tmp_path / "whole", "omezarr@1").read_data("G", "case")
    coarse_streamed, _ = Dataset(tmp_path / "streamed", "omezarr@1").read_data("G", "case")
    assert np.array_equal(np.asarray(coarse_whole), np.asarray(coarse_streamed))
    # Each level carries its OWN scale; the coarse one is the factor times the fine one. NGFF scale is
    # (c, z, y, x) where Spacing is (x, y, z) -- the reversal is the point of asserting it here.
    assert get_ome_zarr_info(streamed, 0)["scale"] == [1.0, 1.0, 1.0, 2.0]
    assert get_ome_zarr_info(streamed, 1)["scale"] == [1.0, 4.0, 4.0, 8.0]


def test_an_interrupted_level_append_leaves_the_original_store_readable(tmp_path, monkeypatch):
    """Deriving the coarse levels rewrites level 0, so a failure here is a failure over the only copy.

    The safety is bought with a sibling store and a rename: a rename that does not happen has to leave
    the original where its readers expect it, not a gap between two deletes.
    """
    data = np.arange(1 * 16 * 16 * 16, dtype=np.float32).reshape(1, 16, 16, 16)
    Dataset(tmp_path / "out", "omezarr").write("G", "case", data, _geometry())
    store = _only_store(tmp_path / "out")
    real_rename = Path.rename

    def interrupt_the_publishing_rename(self, target):
        if self.name.endswith(".appending"):
            raise KeyboardInterrupt
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", interrupt_the_publishing_rename)
    with pytest.raises(KeyboardInterrupt):
        append_ome_zarr_levels(store, [4])
    monkeypatch.undo()

    back, _ = Dataset(tmp_path / "out", "omezarr").read_data("G", "case")
    assert np.array_equal(np.asarray(back, dtype=np.float32).reshape(data.shape), data)


def test_a_pyramid_asked_of_a_format_without_levels_is_refused(tmp_path):
    """Refused at construction, not ignored: only OME-NGFF has levels, and silently writing one would
    leave a consumer's ``@1`` resolving to a level that was never written."""
    with pytest.raises(DatasetManagerError, match="no levels"):
        Dataset(tmp_path / "out", "mha", scale_factors=[4])


def test_a_scale_factor_below_two_is_refused():
    """A 'pyramid' whose level does not shrink is a second copy of level 0."""
    from konfai.data.transform import Write
    from konfai.utils.errors import TransformError

    with pytest.raises(TransformError, match="scale factor below 2"):
        Write(dataset="./Out:omezarr", scale_factors=[1])


# An RFC-5 field is a zarr v3 store (NGFF >= 0.5), which zarr 2.x -- the newest release for Python
# 3.10 -- cannot write. The pyramid tests above need no such store and run everywhere.
_needs_rfc5 = pytest.mark.skipif(
    not _zarr_v3_available(),
    reason="a displacement field's bound is recorded in a zarr v3 store (zarr>=3, Python>=3.11)",
)


@_needs_rfc5
def test_a_field_records_its_own_bound_on_both_write_paths(tmp_path):
    """The largest displacement per component, in world units, written by whoever holds the samples.

    Per component and not one scalar: a consumer divides each bound by the spacing of its OWN axis,
    and these grids are anisotropic, so one collapsed maximum over-reads two axes out of three.
    """
    rng = np.random.default_rng(0)
    field = rng.normal(0.0, 30.0, size=(3, 8, 8, 8)).astype(np.float32)
    field[0, 4, 4, 4], field[1, 2, 2, 2] = 917.5, -640.25
    expected = [float(np.abs(field[component]).max()) for component in range(3)]

    whole = tmp_path / "whole" / "case" / "DVF.ome.zarr"
    whole.parent.mkdir(parents=True)
    write_ome_zarr(whole, field, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), displacement_field=True)
    assert _read_konfai_attributes(whole)[DISPLACEMENT_BOUND_ATTRIBUTE] == expected

    shape, attributes = Dataset(tmp_path / "whole", "omezarr").get_infos("DVF", "case")
    # The read side must carry the RFC-5 type on the HEADERS path too, or a field read region by
    # region and written region by region comes out an ordinary 3-channel image.
    assert DISPLACEMENT_FIELD_ATTRIBUTE in attributes

    stream = Dataset(tmp_path / "streamed", "omezarr").open_data_stream(
        "DVF", "case", list(shape), np.dtype("float32"), attributes
    )
    assert stream is not None
    with stream:
        for start in range(0, 8, 2):
            stream.write_slice(
                (slice(0, 3), slice(start, start + 2), slice(0, 8), slice(0, 8)), field[:, start : start + 2]
            )
    streamed = _only_store(tmp_path / "streamed")

    # Accumulated across regions: the streamed path never sees the field whole, which is the whole
    # reason the bound has to be recorded rather than measured later.
    assert _read_konfai_attributes(streamed)[DISPLACEMENT_BOUND_ATTRIBUTE] == expected
    assert is_displacement_field(streamed)
    assert DISPLACEMENT_BOUND_ATTRIBUTE in Dataset(tmp_path / "streamed", "omezarr").get_infos("DVF", "case")[1]


def tmp_field_store(field: np.ndarray) -> Path:
    import tempfile

    root = Path(tempfile.mkdtemp())
    store = root / "fields" / "case" / "DVF.ome.zarr"
    store.parent.mkdir(parents=True)
    write_ome_zarr(store, field, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), displacement_field=True)
    return root / "fields"


@_needs_rfc5
def test_the_recorded_bound_reaches_each_axis_by_its_own_spacing_under_anisotropy() -> None:
    """What the bound is FOR, read by the stage that consumes it.

    Component ``i`` of a displacement field is world axis (x, y, z)[i], while array axes are
    (z, y, x). So the reach of a region on each array axis reads the components reversed, each
    against its own spacing. Getting the pairing wrong is a warp that raises nothing and reads the
    wrong neighbourhood -- and under this anisotropy the three numbers are far enough apart (3, 22
    and 31 voxels) that any permutation of them is visible.
    """
    from konfai.data.transform import LocalityKind, Warp

    field = np.zeros((3, 8, 8, 8), dtype=np.float32)
    field[0, 4, 4, 4], field[1, 2, 2, 2], field[2, 1, 1, 1] = 917.5, -640.25, 96.0
    store = tmp_field_store(field)

    warp = Warp(field=f"{store}:omezarr", group="DVF", max_displacement="auto")
    attribute = Attribute()
    attribute["Spacing"] = np.array([30.08, 30.08, 40.0])  # stored (x, y, z)
    attribute["Origin"] = np.zeros(3)
    attribute["Direction"] = np.eye(3).reshape(-1)

    assert warp.patch_locality(attribute).kind is LocalityKind.REGRID

    shape = [64, 128, 128]
    warp.transform_shape("CT", "CASE_000", shape, attribute)
    target = tuple(slice(30, 32) for _ in shape)
    window = warp.stream_region_source("CASE_000", target, shape, attribute)

    # z takes the z component (96 um over a 40 um voxel), x the x component (917.5 over 30.08).
    per_axis = [(96.0, 40.0), (640.25, 30.08), (917.5, 30.08)]  # array order (z, y, x)
    expected = [
        (
            max(0, int(np.floor(30 - 0.5 - reach / spacing)) - 1),
            min(extent, int(np.ceil(32 - 0.5 + reach / spacing)) + 2),
        )
        for (reach, spacing), extent in zip(per_axis, shape, strict=True)
    ]
    assert [(part.start, part.stop) for part in window] == expected

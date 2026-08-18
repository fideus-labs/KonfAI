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

"""The write side of the data surface: a declared pyramid, written by both paths identically.

A pyramid is indexed by position (``:omezarr@1``), so a producer that writes one level where two
were promised does not fail, it resolves ``@1`` to something else, or to nothing. So each test
here asserts the two write paths agree: assembled in memory, and streamed region by region. They
are different code (``write_ome_zarr`` against ``create_ome_zarr_store`` +
``append_ome_zarr_levels``), and the streamed one is the path a real volume takes.
"""

from pathlib import Path

import numpy as np
import pytest
import zarr
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import DatasetManagerError
from konfai.utils.ome_zarr import (
    append_ome_zarr_levels,
    get_ome_zarr_info,
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

    The streamed path cannot take ``scale_factors`` at creation (no level exists until the last
    region lands), so it derives them at finalize instead. That is different code, and the levels it
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
    # (c, z, y, x) where Spacing is (x, y, z): the reversal is the point of asserting it here.
    assert get_ome_zarr_info(streamed, 0)["scale"] == [1.0, 1.0, 1.0, 2.0]
    assert get_ome_zarr_info(streamed, 1)["scale"] == [1.0, 4.0, 4.0, 8.0]


def test_each_scale_factor_shrinks_the_level_above_it(tmp_path):
    """``[2, 2]`` is 16 / 8 / 4, on both write paths: each factor is relative to the previous level,
    as the Save docstring says. ngff-zarr takes factors relative to level 0, where ``[2, 2]`` writes
    two identical levels; a user following the doc would get two quarters and no eighth."""
    data = np.arange(1 * 16 * 16 * 16, dtype=np.float32).reshape(1, 16, 16, 16)
    Dataset(tmp_path / "whole", "omezarr", scale_factors=[2, 2]).write("G", "case", data, _geometry())
    streamed_dataset = Dataset(tmp_path / "streamed", "omezarr", scale_factors=[2, 2])
    stream = streamed_dataset.open_data_stream("G", "case", [1, 16, 16, 16], np.dtype("float32"), _geometry())
    assert stream is not None
    with stream:
        stream.write_slice((slice(0, 1), slice(0, 16), slice(0, 16), slice(0, 16)), data)
    for root in ("whole", "streamed"):
        shapes = [
            list(Dataset(tmp_path / root, f"omezarr@{level}").read_data("G", "case")[0].shape) for level in range(3)
        ]
        assert shapes == [[1, 16, 16, 16], [1, 8, 8, 8], [1, 4, 4, 4]], root


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

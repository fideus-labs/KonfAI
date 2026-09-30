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

"""One ``Resample``: which grid to write on, what map to write it through, and what that fixed.

One stage, one sampler, one idea of where a voxel is: the tests here are for what is only
checkable because there is a single answer to check.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from konfai.data.transform import (
    Resample,
)
from konfai.data.transform.resample import _itk_picks_as_the_walk
from konfai.utils.dataset import Attribute
from oracle_support import geometry

sitk = pytest.importorskip("SimpleITK")

_ORIGIN, _SPACING = [11.5, -3.25, 40.0], [1.3, 0.9, 0.9]
_SHAPE = (24, 30, 32)


def _attributes(origin=None, spacing=None, direction=None) -> Attribute:
    return geometry(_ORIGIN if origin is None else origin, _SPACING if spacing is None else spacing, direction)


def _volume(shape=_SHAPE, seed: int = 0) -> np.ndarray:
    # Noise, not a smooth field: a smooth volume resampled onto a grid that is half a voxel off is
    # still nearly right, so a smooth fixture would pass a map that is wrong by exactly the amount
    # this file exists to catch.
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)[None]


def _as_image(volume: np.ndarray, attribute: Attribute) -> sitk.Image:
    image = sitk.GetImageFromArray(volume[0])
    image.SetOrigin(attribute.get_np_array("Origin").tolist())
    image.SetSpacing(attribute.get_np_array("Spacing").tolist())
    image.SetDirection(attribute.get_np_array("Direction").tolist())
    return image


def test_an_axis_the_map_leaves_alone_is_left_alone() -> None:
    """A resample of one axis reads the other two, it does not blend them, and says so in the values.

    ``spacing: [-1, -1, 3]`` keeps x and y exactly. Blending them anyway would be two gathers and a
    lerp over the largest tensor in flight, for a result equal to the input; skipping them has to be
    exactly that, not nearly.
    """
    attribute = _attributes(origin=[0.0] * 3, spacing=[1.0, 1.0, 1.0])
    volume = torch.from_numpy(_volume())
    kept = Resample(spacing=[-1.0, -1.0, 1.0])("case", volume.clone(), Attribute(attribute))
    torch.testing.assert_close(kept, volume, rtol=0, atol=0)


def test_a_case_recorded_again_with_the_same_header_is_not_judged_again(monkeypatch) -> None:
    """Every plan records the case again; its coverage is counted once for the grid it was found on."""
    from konfai.data.transform import Resample
    from konfai.utils.dataset import Attribute

    judged = []
    count = Resample._coverage
    monkeypatch.setattr(Resample, "_coverage", classmethod(lambda cls, *args: judged.append(args) or count(*args)))
    stage = Resample(spacing=[2.0, 2.0, 2.0])
    header = {"Origin": np.zeros(3), "Spacing": np.ones(3), "Direction": np.eye(3).ravel()}
    for _ in range(3):
        assert stage.transform_shape("CT", "CASE", [8, 8, 8], Attribute(header)) == [4, 4, 4]
    assert len(judged) == 1

    moved = dict(header, Origin=np.full(3, 1.0))
    stage.transform_shape("CT", "CASE", [8, 8, 8], Attribute(moved))
    assert len(judged) == 2


def test_a_header_that_left_the_geometry_unsaid_is_not_the_identity_header_it_reads_as() -> None:
    """Both read as the identity grid, and only the second cannot place a target grid: the memo of the
    first must not answer for it."""
    from konfai.data.transform import Resample
    from konfai.utils.dataset import Attribute
    from konfai.utils.errors import TransformError

    stage = Resample(spacing=[2.0, 2.0, 2.0])
    explicit = {"Origin": np.zeros(3), "Spacing": np.ones(3), "Direction": np.eye(3).ravel()}
    stage.transform_shape("CT", "CASE", [8, 8, 8], Attribute(explicit))
    stage._grids_of("CASE")

    stage._record("CASE", [8, 8, 8], Attribute())
    with pytest.raises(TransformError):
        stage._grids_of("CASE")


# The axes of a sagittal or coronal acquisition are a signed permutation of the world's: a label map
# resampled at half its spacing puts every other target voxel EXACTLY half-way between two source
# voxels, and the voxel ITK picks there depends on the last bit of the continuous index.
_PERMUTED = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_PERMUTED_ORIGIN, _PERMUTED_SPACING = [-11.3, 5.9, 3.3], [0.77, 1.31, 2.05]
_PERMUTED_SHAPE = (19, 23, 29)
# The same axes tilted by three degrees about x: an oblique acquisition. Resample's target keeps the
# source's direction, so the map between their indices is still a scale and a shift per axis in
# exact arithmetic, with the same ties to decide.
_TILT = np.deg2rad(3.0)
_OBLIQUE = (
    np.array([[1.0, 0.0, 0.0], [0.0, np.cos(_TILT), -np.sin(_TILT)], [0.0, np.sin(_TILT), np.cos(_TILT)]]) @ _PERMUTED
)
# An axial grid whose direction carries float noise off the diagonal: not exactly diagonal, so off the
# separable path, and read as an oblique grid is.
_NOISY_AXIAL = np.diag([-1.0, -1.0, 1.0]) + 1e-9 * np.random.default_rng(1).normal(size=(3, 3))
_DIRECTIONS = {"permuted": _PERMUTED, "oblique": _OBLIQUE, "noisy-axial": _NOISY_AXIAL}


def _permuted_case(tmp_path, volume: np.ndarray, direction: np.ndarray = _PERMUTED):
    from konfai.utils.dataset import Dataset

    dataset = Dataset(tmp_path / "Dataset", "mha")
    dataset.write("Case", "CASE", volume, geometry(_PERMUTED_ORIGIN, _PERMUTED_SPACING, direction))
    return dataset


def _streamed(dataset, stage: Resample, rows: int) -> np.ndarray:
    """The case read region by region, ``rows`` target slices at a time, as a small budget sweeps it."""
    from konfai.data.patching import DatasetManager

    manager = DatasetManager(
        index=0,
        group_src="Case",
        group_dest="Case",
        name="CASE",
        dataset=dataset,
        patch=None,
        transforms=[stage],
        data_augmentations_list=[],
    )
    assert manager.stream_refusal(0) is None
    extent = manager.spatial_shape
    parts = [
        manager.read_region((slice(start, min(start + rows, extent[0])), slice(0, extent[1]), slice(0, extent[2])))
        for start in range(0, extent[0], rows)
    ]
    return torch.cat(parts, dim=1).numpy()


@pytest.mark.parametrize("route", ["host", "device"])
@pytest.mark.parametrize("interpolation", ["nearest", "linear"])
@pytest.mark.parametrize(
    "request_grid",
    [{"spacing": [0.385, 0.655, 1.025], "align": "origin"}, {"shape": [40, 9, 0], "align": "origin"}],
    ids=["half-spacing", "shape"],
)
@pytest.mark.parametrize("direction", list(_DIRECTIONS))
def test_a_permuted_or_oblique_grid_streams_what_simpleitk_resamples(
    tmp_path, monkeypatch, direction: str, request_grid: dict, interpolation: str, route: str
) -> None:
    """Region by region, a case on a permuted or oblique grid holds what the whole volume holds, which
    is what ``sitk.Resample`` holds: the same voxel picked on every half-voxel tie, the same fill on
    the far face, the same blend, whatever the budget. The device route (the torch walk, which a host
    without ITK's filter also takes) holds it too."""
    if route == "device":
        monkeypatch.setattr("konfai.data.transform.resample.sitk", None)
    rng = np.random.default_rng(7)
    if interpolation == "nearest":
        volume = rng.integers(0, 250, size=_PERMUTED_SHAPE).astype(np.uint8)[None]
    else:
        volume = (rng.normal(size=_PERMUTED_SHAPE) * 1000).astype(np.float32)[None]
    dataset = _permuted_case(tmp_path, volume, _DIRECTIONS[direction])
    attribute = geometry(_PERMUTED_ORIGIN, _PERMUTED_SPACING, _DIRECTIONS[direction])

    landed = Attribute(attribute)
    stage = Resample(**request_grid, interpolation=interpolation, fill=9.0)
    whole = stage("CASE", torch.from_numpy(volume.copy()), landed).numpy()
    got = _streamed(dataset, Resample(**request_grid, interpolation=interpolation, fill=9.0), rows=3)

    differ = int((got != whole).sum())
    assert differ == 0, f"{differ} of {got.size} voxels streamed differ from the whole volume"
    _as_simpleitk(whole[0], lambda: _permuted_reference(volume, attribute, landed, whole.shape[1:], interpolation))


@pytest.mark.parametrize(
    ("header", "shape", "request_grid"),
    [
        (
            (_PERMUTED_ORIGIN, _PERMUTED_SPACING, np.diag([-1.0, -1.0, 1.0])),
            _PERMUTED_SHAPE,
            {"spacing": [0.385, 0.655, 1.025], "align": "origin"},
        ),
        (
            ([38.409, 8.445, -142.821], [1.691, 1.072, 0.446], np.diag([-1.0, 1.0, -1.0])),
            (36, 24, 38),
            {"spacing": [3.382, 2.144, 0.892]},
        ),
    ],
    ids=["half-spacing-origin", "double-spacing-extent"],
)
def test_an_axis_aligned_label_map_streams_its_ties_as_the_whole_volume(
    tmp_path, header: tuple, shape: tuple, request_grid: dict
) -> None:
    """At half the spacing on ``align: origin``, or twice it on the default extent, target voxels lie
    EXACTLY half-way between source voxels. A region reads them on the whole grid's own indices, so
    it picks the voxel the whole volume picks, whatever the budget cuts."""
    from konfai.utils.dataset import Dataset

    labels = np.random.default_rng(11).integers(0, 250, size=shape).astype(np.uint8)[None]
    dataset = Dataset(tmp_path / "Dataset", "mha")
    dataset.write("Case", "CASE", labels, geometry(*header))

    whole = Resample(**request_grid, fill=9.0)("CASE", torch.from_numpy(labels.copy()), Attribute(geometry(*header)))
    got = _streamed(dataset, Resample(**request_grid, fill=9.0), rows=3)

    differ = int((got != whole.numpy()).sum())
    assert differ == 0, f"{differ} of {got.size} voxels streamed differ from the whole volume"


def _as_simpleitk(actual: np.ndarray, reference) -> None:
    """``actual`` is what ``sitk.Resample`` holds (``reference()``) where this platform's ITK picks the
    half-voxel ties KonfAI's walk picks; elsewhere ITK's own arithmetic moves them, and the walk, which a
    region and the whole volume share, is the reference."""
    if _itk_picks_as_the_walk():
        np.testing.assert_array_equal(actual, reference())


def _permuted_reference(
    volume: np.ndarray, attribute: Attribute, landed: Attribute, shape, interpolation: str, fill: float = 9.0
):
    """``sitk.Resample`` of ``volume`` onto the grid ``landed`` describes, through no map."""
    return sitk.GetArrayFromImage(
        sitk.Resample(
            _as_image(volume, attribute),
            [int(extent) for extent in reversed(shape)],
            sitk.Transform(),
            sitk.sitkNearestNeighbor if interpolation == "nearest" else sitk.sitkLinear,
            landed.get_np_array("Origin").tolist(),
            landed.get_np_array("Spacing").tolist(),
            landed.get_np_array("Direction").tolist(),
            fill,
        )
    )


@pytest.mark.parametrize("precision", ["exact", "fast"])
@pytest.mark.parametrize("interpolation", ["nearest", "linear"])
def test_a_permuted_grid_streams_what_simpleitk_resamples_whatever_the_values(
    tmp_path, interpolation: str, precision: str
) -> None:
    """A NaN or an infinity on a tap the blend gives no weight stays out of the value, as ITK's
    branches leave it out, and an infinity lands on float32's largest value, as ITK's cast puts it.
    On the host a region stands in for ITK's filter over the whole grid, so 'fast', a trade the
    device walk makes, keeps it in float64 there."""
    rng = np.random.default_rng(3)
    volume = (rng.normal(size=_PERMUTED_SHAPE) * 1000).astype(np.float32)[None]
    spots = rng.choice(volume.size, 120, replace=False)
    volume[0].flat[spots[:40]] = np.nan
    volume[0].flat[spots[40:80]] = np.inf
    volume[0].flat[spots[80:]] = -np.inf
    dataset = _permuted_case(tmp_path, volume)
    attribute = geometry(_PERMUTED_ORIGIN, _PERMUTED_SPACING, _PERMUTED)
    request = {"spacing": [0.385, 0.655, 1.025], "align": "origin", "interpolation": interpolation}

    landed = Attribute(attribute)
    whole = Resample(**request, fill=9.0, precision=precision)("CASE", torch.from_numpy(volume.copy()), landed).numpy()
    got = _streamed(dataset, Resample(**request, fill=9.0, precision=precision), rows=3)

    np.testing.assert_array_equal(got, whole)
    _as_simpleitk(whole[0], lambda: _permuted_reference(volume, attribute, landed, whole.shape[1:], interpolation))


def test_a_permuted_mask_resamples_as_the_same_mask_in_bytes() -> None:
    """ITK has no boolean pixel, so a bool mask takes the torch walk on the host where its uint8 twin
    takes ITK's filter: on a permuted grid both pick the same voxel on every half-voxel tie, whole
    and region by region."""
    mask = torch.from_numpy(np.random.default_rng(5).random(_PERMUTED_SHAPE) > 0.5)[None]
    attribute = geometry(_PERMUTED_ORIGIN, _PERMUTED_SPACING, _PERMUTED)
    request = {"spacing": [0.385, 0.655, 1.025], "align": "origin"}

    stage = Resample(**request)
    whole = stage("CASE", mask.clone(), Attribute(attribute))
    as_bytes = Resample(**request)("CASE", mask.to(torch.uint8), Attribute(attribute))
    extent = list(whole.shape[1:])
    regions = torch.cat(
        [
            stage._sample(
                "CASE",
                mask,
                (slice(start, min(start + 3, extent[0])), slice(0, extent[1]), slice(0, extent[2])),
                [0, 0, 0],
            )
            for start in range(0, extent[0], 3)
        ],
        dim=1,
    )

    assert torch.equal(whole.to(torch.uint8), as_bytes)
    assert torch.equal(regions, whole)


# Every dtype a volume arrives in: the integer widths a scanner, a microscope or an Argmax stores, the
# float widths, and a boolean mask.
_DTYPES = [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]
_DTYPES += [np.float16, np.float32, np.float64, np.bool_]
_ALIGNED = np.diag([-1.0, -1.0, 1.0])


@pytest.mark.parametrize("fill", [9.0, -1.0])
@pytest.mark.parametrize("route", ["host", "device"])
@pytest.mark.parametrize("direction", ["aligned", "permuted", "oblique"])
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda dtype: np.dtype(dtype).name)
def test_a_nearest_resample_serves_every_dtype(monkeypatch, dtype, direction: str, route: str, fill: float) -> None:
    """A nearest pick copies voxels in their own dtype and fills the far face in it, the fill cast as
    ITK casts it (-1 is 255 in a uint8 volume). torch has no fill of its own for uint16, uint32 or
    uint64, which is what a microscope stores. Whole and region by region, every dtype resamples as
    the same values do in float64, cast; off the separable path, as ``sitk.Resample`` does, or, where
    ITK has no such pixel (bool, float16), as the same values in a wider dtype."""
    if route == "device":
        monkeypatch.setattr("konfai.data.transform.resample.sitk", None)
    shape = (7, 9, 11)
    values = np.random.default_rng(17).integers(0, 120, size=shape)[None]
    volume = values.astype(dtype)
    attribute = geometry(_PERMUTED_ORIGIN, _PERMUTED_SPACING, {"aligned": _ALIGNED, **_DIRECTIONS}[direction])
    request = {"spacing": [0.385, 0.655, 1.025], "align": "origin", "interpolation": "nearest", "fill": fill}

    stage = Resample(**request)
    landed = Attribute(attribute)
    whole = stage("CASE", torch.from_numpy(volume.copy()), landed)
    extent = list(whole.shape[1:])
    regions = torch.cat(
        [
            stage._sample(
                "CASE",
                torch.from_numpy(volume.copy()),
                (slice(start, min(start + 3, extent[0])), slice(0, extent[1]), slice(0, extent[2])),
                [0, 0, 0],
            )
            for start in range(0, extent[0], 3)
        ],
        dim=1,
    )

    wide = Resample(**request)("CASE", torch.from_numpy(values.astype(np.float64)), Attribute(attribute))

    assert whole.dtype == regions.dtype == torch.from_numpy(volume).dtype
    assert torch.equal(regions, whole)
    assert torch.equal(whole, wide.to(whole.dtype))
    if direction != "aligned":
        # Two axis-aligned grids are read by the separable path, whose ties are its own, not ITK's.
        twin = {np.bool_: np.uint8, np.float16: np.float32}.get(dtype, dtype)
        _as_simpleitk(
            whole[0].numpy(),
            lambda: _permuted_reference(values.astype(twin), attribute, landed, extent, "nearest", fill).astype(dtype),
        )


def _inverse_streamed(stage: Resample, held: torch.Tensor, case: Attribute, rows: int) -> torch.Tensor:
    """``held`` brought back onto the case's grid ``rows`` slices at a time, each slice reading only the
    window of ``held`` it pulls, as a prediction's streamed writer does."""
    from konfai.data.transform.base import RegionContext

    extent = stage.inverse_transform_shape(list(held.shape[1:]), Attribute(case))
    parts = []
    for start in range(0, extent[0], rows):
        target = (slice(start, min(start + rows, extent[0])), slice(0, extent[1]), slice(0, extent[2]))
        window = stage.stream_region_target("CASE", target, list(held.shape[1:]), case)
        context = RegionContext(source=tuple(window), target=target, source_shape=tuple(held.shape[1:]))
        parts.append(stage.stream_region_inverse("CASE", held[(slice(None), *window)], context, Attribute(case)))
    return torch.cat(parts, dim=1)


@pytest.mark.parametrize("interpolation", ["nearest", "linear"])
@pytest.mark.parametrize(
    "request_grid",
    [{"spacing": [1.54, 2.62, 4.1], "align": "origin"}, {"spacing": [1.0, 1.0, 3.0]}],
    ids=["double-spacing", "other-spacing"],
)
@pytest.mark.parametrize("direction", list(_DIRECTIONS))
def test_the_inverse_onto_a_permuted_or_oblique_grid_streams_what_simpleitk_resamples(
    direction: str, request_grid: dict, interpolation: str
) -> None:
    """A prediction brought back onto a sagittal, coronal or oblique case holds, slab by slab, what the
    whole volume holds, which is what ``sitk.Resample`` holds: the inverse reads the index ITK reads
    and blends as ITK does, as the forward does. At twice the spacing on ``align: origin`` every
    other voxel of the case lies EXACTLY half-way between two voxels of the prediction."""
    rng = np.random.default_rng(23)
    if interpolation == "nearest":
        volume = rng.integers(0, 250, size=_PERMUTED_SHAPE).astype(np.uint8)[None]
    else:
        volume = (rng.normal(size=_PERMUTED_SHAPE) * 1000).astype(np.float32)[None]
    attribute = geometry(_PERMUTED_ORIGIN, _PERMUTED_SPACING, _DIRECTIONS[direction])
    stage = Resample(**request_grid, interpolation=interpolation, fill=9.0, inverse=True)
    case = Attribute(attribute)
    held = stage("CASE", torch.from_numpy(volume.copy()), case)

    whole = stage.inverse("CASE", held.clone(), Attribute(case))
    streamed = _inverse_streamed(stage, held, case, rows=3)

    def want() -> np.ndarray:
        return sitk.GetArrayFromImage(
            sitk.Resample(
                _as_image(held.numpy(), case),
                list(reversed(_PERMUTED_SHAPE)),
                sitk.Transform(),
                sitk.sitkNearestNeighbor if interpolation == "nearest" else sitk.sitkLinear,
                list(_PERMUTED_ORIGIN),
                list(_PERMUTED_SPACING),
                _DIRECTIONS[direction].ravel().tolist(),
                9.0,
            )
        )

    differ = int((streamed != whole).sum())
    assert differ == 0, f"{differ} of {whole.numel()} voxels streamed differ from the whole volume"
    _as_simpleitk(whole[0].numpy(), want)


def test_a_permuted_cubic_resample_keeps_its_own_walk() -> None:
    """ITK has no Keys kernel to agree with: on a permuted grid a cubic resample keeps the exact index
    its own walk reads, and the far face that index puts outside."""
    from konfai.data.sampling import gather, source_index

    volume = (np.random.default_rng(9).normal(size=_PERMUTED_SHAPE) * 1000).astype(np.float32)[None]
    stage = Resample(spacing=[0.385, 0.655, 1.025], align="origin", interpolation="cubic", fill=9.0)
    whole = stage("CASE", torch.from_numpy(volume.copy()), geometry(_PERMUTED_ORIGIN, _PERMUTED_SPACING, _PERMUTED))
    source, target = stage._grids_of("CASE")
    walked = gather(
        torch.from_numpy(volume),
        source_index(target, source, (), torch.device("cpu")),
        [0] * source.rank,
        list(source.size_zyx),
        "cubic",
        9.0,
    )

    assert torch.equal(whole, walked)


def test_the_host_blends_as_itk_a_chunk_at_a_time(monkeypatch) -> None:
    """What ITK's route holds beside its output on the host is one chunk's corners and values: the
    region's size, and the channel count past the chunk's own output, never reach it."""
    from konfai.data import sampling
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    rng = np.random.default_rng(13)
    source = torch.from_numpy(rng.normal(size=(6, 9, 11, 13)).astype(np.float32))
    coordinates = torch.from_numpy(rng.uniform(-0.4, 1.0, size=(40, 30, 3)) * np.array([12.4, 10.4, 8.4]))
    expected = sampling.gather(source, coordinates, [0, 0, 0], [9, 11, 13], "linear", 9.0, True)
    storages: dict[int, int] = {}

    class Storages(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            for tensor in tree_leaves(result):
                if isinstance(tensor, torch.Tensor):
                    storage = tensor.untyped_storage()
                    storages[storage.data_ptr()] = max(storages.get(storage.data_ptr(), 0), storage.nbytes())
            return result

    chunk = 512
    monkeypatch.setattr(sampling, "_HOST_ITK_VOXELS", chunk, raising=False)
    with Storages():
        blended = sampling.gather(source, coordinates, [0, 0, 0], [9, 11, 13], "linear", 9.0, True)
    held = {tensor.untyped_storage().data_ptr() for tensor in (source, coordinates, blended)}

    assert torch.equal(blended, expected)
    assert max(nbytes for pointer, nbytes in storages.items() if pointer not in held) <= 8 * chunk * 8

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

"""The torch coordinate producer and gather, against SimpleITK.

The oracle is external on purpose: two KonfAI paths agree by construction, including on a grid
placed in the wrong place. The fixture is high-frequency (a smooth phantom would pass a wrong
map), and its direction cosines are oblique, which is the case the separable samplers refuse and
this one exists for.
"""

import numpy as np
import pytest
import torch
from konfai.data.geometry import Grid, bound_of
from konfai.data.sampling import gather, source_index, source_window

sitk = pytest.importorskip("SimpleITK")

from konfai.utils.ITK import decode_transform_stages  # noqa: E402

SIZE = (24, 30, 36)


def _phantom(size=SIZE) -> np.ndarray:
    z, y, x = np.meshgrid(*[np.arange(extent, dtype=np.float64) for extent in size], indexing="ij")
    return (100.0 * np.sin(1.7 * z) * np.cos(2.1 * y) + 80.0 * np.sin(2.9 * x)).astype(np.float32)


def _image(oblique: bool = True, size=SIZE) -> "sitk.Image":
    image = sitk.GetImageFromArray(_phantom(size))
    image.SetSpacing((0.8, 1.2, 1.5))
    image.SetOrigin((10.0, -5.0, 2.0))
    if oblique:
        a, b = np.deg2rad(20.0), np.deg2rad(15.0)
        rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
        ry = np.array([[np.cos(b), 0.0, np.sin(b)], [0.0, 1.0, 0.0], [-np.sin(b), 0.0, np.cos(b)]])
        image.SetDirection(tuple((rz @ ry).ravel()))
    return image


def _grid(image: "sitk.Image") -> Grid:
    rank = image.GetDimension()
    return Grid(
        tuple(int(extent) for extent in reversed(image.GetSize())),
        np.asarray(image.GetOrigin(), dtype=np.float64),
        np.asarray(image.GetSpacing(), dtype=np.float64),
        np.asarray(image.GetDirection(), dtype=np.float64).reshape(rank, rank),
    )


def _euler(image):
    transform = sitk.Euler3DTransform()
    transform.SetCenter(image.TransformContinuousIndexToPhysicalPoint([(s - 1) / 2 for s in image.GetSize()]))
    transform.SetRotation(0.11, -0.2, 0.31)
    transform.SetTranslation((3.0, -2.0, 1.0))
    return transform


def _affine(image):
    transform = sitk.AffineTransform(3)
    transform.SetCenter(image.TransformContinuousIndexToPhysicalPoint([(s - 1) / 2 for s in image.GetSize()]))
    transform.SetMatrix(np.array([[1.1, 0.05, 0.0], [0.0, 0.9, 0.07], [0.02, 0.0, 1.2]]).ravel())
    transform.SetTranslation((4.0, -3.0, 2.0))
    return transform


def _bspline(image, mesh: int = 5, amplitude: float = 7.0):
    transform = sitk.BSplineTransformInitializer(image, [mesh] * 3, 3)
    size = np.asarray(transform.GetParameters()).size
    transform.SetParameters(list(np.random.RandomState(2).uniform(-amplitude, amplitude, size)))
    return transform


def _field(image):
    filt = sitk.TransformToDisplacementFieldFilter()
    filt.SetReferenceImage(image)
    return sitk.DisplacementFieldTransform(sitk.Cast(filt.Execute(_bspline(image)), sitk.sitkVectorFloat64))


def _transforms(image) -> list[tuple[str, "sitk.Transform"]]:
    return [
        ("euler", _euler(image)),
        ("affine", _affine(image)),
        ("bspline", _bspline(image)),
        ("field", _field(image)),
        ("composite", sitk.CompositeTransform([_affine(image), _bspline(image)])),
    ]


def _resample_whole(image, transform, interpolator=sitk.sitkLinear, fill: float = 0.0) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.Resample(image, image, transform, interpolator, fill))


def _konfai_region(
    image, transform, region: tuple[slice, ...], device: torch.device, mode: str = "linear", fill: float = 0.0
) -> np.ndarray:
    """One target region through the torch path, reading only the bounded source window."""
    grid = _grid(image)
    stages = decode_transform_stages(transform)
    bound = bound_of(stages, 3)
    target = grid.sub_grid(region)
    window = source_window(target, grid, bound)
    volume = torch.from_numpy(sitk.GetArrayFromImage(image)).unsqueeze(0)
    block = volume[(slice(None), *window)].to(device)
    coordinates = source_index(target, grid, stages, device)
    out = gather(
        block,
        coordinates,
        [part.start for part in window],
        list(grid.size_zyx),
        mode,
        fill,
    )
    return out.squeeze(0).cpu().numpy()


DEVICES = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
DEVICE_IDS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def test_the_host_resampler_and_the_walk_agree_on_an_integer_payload():
    """A CPU run takes ITK's resampler and a CUDA run takes the walk, so what separates them is
    what separates a laptop's answer from a card's.

    Both blend in the same working dtype and both cast by truncating toward zero, so they agree to
    the ulp of the blend -- and on an INTEGER payload that ulp is a whole unit wherever the blended
    value lands on a truncation boundary. This pins the size of that seam (at most one unit, on a
    handful of voxels) so a real divergence, which moves voxels by far more, cannot hide in it.
    ``nearest`` copies voxels and must be exact, integers included.
    """
    from konfai.data.transform.resample import _resample_with_sitk

    image = _image(oblique=True)
    counts = (sitk.GetArrayFromImage(image) * 4.0).astype(np.int16)
    integer = sitk.GetImageFromArray(counts)
    integer.CopyInformation(image)
    grid = _grid(integer)
    payload = torch.from_numpy(counts).unsqueeze(0)
    for label, transform in (("euler", _euler(integer)), ("affine", _affine(integer))):
        stages = decode_transform_stages(transform)  # a rotation: no separable form, so the host path serves it
        target = grid.sub_grid((slice(4, 11), slice(0, SIZE[1]), slice(0, SIZE[2])))
        window = source_window(target, grid, bound_of(stages, 3))
        block = payload[(slice(None), *window)]
        starts = [part.start for part in window]
        for mode, tolerance in (("linear", 1), ("nearest", 0)):
            host = _resample_with_sitk(block, target, grid, stages, starts, mode, 0.0)
            walk = gather(
                block, source_index(target, grid, stages, torch.device("cpu")), starts, list(grid.size_zyx), mode, 0.0
            )
            assert host is not None and host.dtype == walk.dtype == torch.int16, label
            apart = (host.to(torch.int32) - walk.to(torch.int32)).abs()
            assert int(apart.max()) <= tolerance, f"{label}/{mode}: {int(apart.max())} units apart"
            assert int((apart > 0).sum()) <= apart.numel() // 500, f"{label}/{mode}: the seam is not a seam"


@pytest.mark.slow
@pytest.mark.parametrize("oblique", [False, True], ids=["axis-aligned", "oblique"])
def test_the_whole_grid_matches_simpleitk(oblique: bool):
    device = torch.device("cpu")
    image = _image(oblique)
    whole = (slice(0, SIZE[0]), slice(0, SIZE[1]), slice(0, SIZE[2]))
    for label, transform in _transforms(image):
        want = _resample_whole(image, transform)
        got = _konfai_region(image, transform, whole, device)
        deviation = float(np.abs(want - got).max())
        scale = float(np.abs(want).max())
        assert deviation <= 1e-3 * scale, f"{label}: {deviation:.4g} against a range of {scale:.4g}"


@pytest.mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@pytest.mark.parametrize(
    "rows", [pytest.param(3, marks=pytest.mark.slow), 7, SIZE[0]], ids=["slab-3", "slab-7", "whole"]
)
def test_the_streamed_slabs_agree_with_the_whole_volume(device: torch.device, rows: int):
    """A slab and the whole volume put every sample in the same place, to a stated bound.

    NOT bit for bit, and the reason is deliberate: a blend through a map that does not factorise
    goes to ``grid_sample``, one fused kernel worth 4x on a warp, and grid_sample takes NORMALISED
    coordinates, so it divides by the extent of the tensor handed to it, and a slab is handed a
    window. What is bit-identical is the SEPARABLE path, which is most resamples, and which
    ``test_resample.py`` pins.

    The bound below is a fraction of the data's own range, and it is roughly thirty times the worst
    disagreement measured, while a slab whose map actually moved is wrong by VOXELS, orders of
    magnitude above it. The companion test underneath keeps that end honest.
    """
    image = _image()
    whole_region = (slice(0, SIZE[0]), slice(0, SIZE[1]), slice(0, SIZE[2]))
    for label, transform in _transforms(image):
        reference = _konfai_region(image, transform, whole_region, device)
        streamed = np.empty_like(reference)
        for start in range(0, SIZE[0], rows):
            stop = min(start + rows, SIZE[0])
            region = (slice(start, stop), slice(0, SIZE[1]), slice(0, SIZE[2]))
            streamed[start:stop] = _konfai_region(image, transform, region, device)
        span = float(reference.max() - reference.min())
        np.testing.assert_allclose(streamed, reference, rtol=0, atol=1e-5 * span, err_msg=label)


def test_a_slab_left_at_the_volume_origin_is_loudly_wrong():
    device = torch.device("cpu")
    # The silent failure mode, tested first: a region whose grid keeps the VOLUME's origin replays
    # the same part of the source for every slab, and the output still looks like an image. If this
    # test ever passes quietly, sub_grid stopped placing regions.
    image = _image()
    transform = _euler(image)
    grid = _grid(image)
    stages = decode_transform_stages(transform)
    reference = _konfai_region(image, transform, (slice(0, SIZE[0]), slice(0, SIZE[1]), slice(0, SIZE[2])), device)

    rows = 6
    wrong = np.empty_like(reference)
    volume = torch.from_numpy(sitk.GetArrayFromImage(image)).unsqueeze(0).to(device)
    for start in range(0, SIZE[0], rows):
        stop = min(start + rows, SIZE[0])
        misplaced = Grid((stop - start, SIZE[1], SIZE[2]), grid.origin_xyz, grid.spacing_xyz, grid.direction_xyz)
        coordinates = source_index(misplaced, grid, stages, device)
        wrong[start:stop] = (
            gather(volume, coordinates, [0, 0, 0], list(grid.size_zyx), "linear", 0.0).squeeze(0).cpu().numpy()
        )
    assert np.abs(wrong - reference).max() > 0.1 * float(np.abs(reference).max())


def test_the_fill_mask_matches_simpleitk_voxel_for_voxel():
    device = torch.device("cpu")
    # The sharp test: where the map leaves the source, SimpleITK writes the fill and so must this.
    # A source window one voxel short does not raise (it returns background), so the only thing
    # that catches it is comparing WHICH voxels are fill, with no tolerance at all.
    image = _image()
    transform = sitk.TranslationTransform(3, (14.0, 9.0, 7.0))
    fill = -1234.0
    want = _resample_whole(image, transform, fill=fill)
    got = _konfai_region(image, transform, (slice(0, SIZE[0]), slice(0, SIZE[1]), slice(0, SIZE[2])), device, fill=fill)
    assert 0 < int((want == fill).sum()) < want.size, "the fixture must actually leave the source"
    np.testing.assert_array_equal(got == fill, want == fill)


def test_nearest_is_byte_identical_on_a_label_map():
    device = torch.device("cpu")
    labels = (np.abs(_phantom()) % 7).astype(np.uint8)
    image = sitk.GetImageFromArray(labels)
    image.SetSpacing((0.8, 1.2, 1.5))
    image.SetOrigin((10.0, -5.0, 2.0))
    transform = _euler(image)
    want = _resample_whole(image, transform, interpolator=sitk.sitkNearestNeighbor)
    got = _konfai_region(
        image, transform, (slice(0, SIZE[0]), slice(0, SIZE[1]), slice(0, SIZE[2])), device, mode="nearest"
    )
    np.testing.assert_array_equal(got, want)


class TestSourceWindow:
    def test_the_window_covers_every_coordinate_the_region_samples(self):
        image = _image()
        grid = _grid(image)
        for label, transform in _transforms(image):
            stages = decode_transform_stages(transform)
            bound = bound_of(stages, 3)
            region = (slice(6, 13), slice(0, SIZE[1]), slice(0, SIZE[2]))
            target = grid.sub_grid(region)
            window = source_window(target, grid, bound)
            coordinates = source_index(target, grid, stages, torch.device("cpu")).numpy()
            for axis in range(3):
                array_axis = 2 - axis
                sampled = coordinates[..., axis]
                # Only what actually lands on the source has to be covered; the rest takes the fill.
                inside = (sampled >= -0.5) & (sampled < grid.size_zyx[array_axis] - 0.5)
                if not inside.any():
                    continue
                low, high = float(sampled[inside].min()), float(sampled[inside].max())
                extent = int(grid.size_zyx[array_axis])
                # The taps a sample needs, clamped to the source exactly as the gather clamps them:
                # a coordinate in the half-voxel rim reads the border voxel, it does not read past it.
                need_low = min(max(int(np.floor(low)), 0), extent - 1)
                need_high = min(max(int(np.floor(high)) + 1, 0), extent - 1)
                part = window[array_axis]
                assert part.start <= need_low and need_high < part.stop, (
                    f"{label}: axis {array_axis} needs [{need_low}, {need_high}] outside {part}"
                )

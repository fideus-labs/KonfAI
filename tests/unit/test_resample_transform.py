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

"""``ResampleTransform`` streaming: against SimpleITK, and against its own whole-volume path.

The fixture is high-frequency and its direction cosines are oblique, because a smooth phantom on
an axis-aligned grid passes a map that is wrong in exactly the ways this stage can be wrong.
"""

import numpy as np
import pytest
import torch
from konfai.data.transform import LocalityKind, RegionContext, ResampleTransform
from konfai.utils.dataset import Attribute
from konfai.utils.errors import TransformError

sitk = pytest.importorskip("SimpleITK")

SIZE = (22, 28, 34)
CASE = "CASE_000"


def _phantom() -> np.ndarray:
    z, y, x = np.meshgrid(*[np.arange(extent, dtype=np.float64) for extent in SIZE], indexing="ij")
    return (100.0 * np.sin(1.7 * z) * np.cos(2.1 * y) + 80.0 * np.sin(2.9 * x)).astype(np.float32)


def _image(oblique: bool = True) -> "sitk.Image":
    image = sitk.GetImageFromArray(_phantom())
    image.SetSpacing((0.8, 1.2, 1.5))
    image.SetOrigin((10.0, -5.0, 2.0))
    if oblique:
        a, b = np.deg2rad(20.0), np.deg2rad(15.0)
        rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
        ry = np.array([[np.cos(b), 0.0, np.sin(b)], [0.0, 1.0, 0.0], [-np.sin(b), 0.0, np.cos(b)]])
        image.SetDirection(tuple((rz @ ry).ravel()))
    return image


def _attribute(image: "sitk.Image") -> Attribute:
    attribute = Attribute()
    attribute["Origin"] = np.asarray(image.GetOrigin())
    attribute["Spacing"] = np.asarray(image.GetSpacing())
    attribute["Direction"] = np.asarray(image.GetDirection())
    return attribute


class _StoredTransform:
    """The smallest thing that answers what ``ResampleTransform`` asks of a ``Dataset``."""

    def __init__(self, group: str, transform: "sitk.Transform") -> None:
        self.group = group
        self.transform = transform

    def is_dataset_exist(self, group: str, name: str) -> bool:
        del name
        return group == self.group

    def read_transform(self, group: str, name: str) -> "sitk.Transform":
        del group, name
        return self.transform


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


def _bspline(image, amplitude: float = 7.0):
    transform = sitk.BSplineTransformInitializer(image, [5] * 3, 3)
    size = np.asarray(transform.GetParameters()).size
    transform.SetParameters(list(np.random.RandomState(2).uniform(-amplitude, amplitude, size)))
    return transform


def _field(image):
    filt = sitk.TransformToDisplacementFieldFilter()
    filt.SetReferenceImage(image)
    return sitk.DisplacementFieldTransform(sitk.Cast(filt.Execute(_bspline(image)), sitk.sitkVectorFloat64))


def _families(image) -> list[tuple[str, "sitk.Transform"]]:
    return [
        ("euler", _euler(image)),
        ("affine", _affine(image)),
        ("bspline", _bspline(image)),
        ("field", _field(image)),
    ]


def _stage(image, transform, **kwargs) -> ResampleTransform:
    stage = ResampleTransform(transforms={"reg": False}, **kwargs)
    stage.set_datasets([_StoredTransform("reg", transform)])
    stage.transform_shape("", CASE, list(SIZE), _attribute(image))
    return stage


DEVICES = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
DEVICE_IDS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("oblique", [False, True], ids=["axis-aligned", "oblique"])
def test_the_whole_volume_path_matches_simpleitk(oblique: bool):
    device = torch.device("cpu")
    image = _image(oblique)
    volume = torch.from_numpy(sitk.GetArrayFromImage(image)).unsqueeze(0).to(device)
    for label, transform in _families(image):
        stage = _stage(image, transform)
        want = sitk.GetArrayFromImage(sitk.Resample(image, image, transform, sitk.sitkLinear, 0.0))
        got = stage(CASE, volume, _attribute(image)).squeeze(0).cpu().numpy()
        deviation = float(np.abs(want - got).max())
        assert deviation <= 1e-3 * float(np.abs(want).max()), f"{label}: {deviation:.4g}"


@pytest.mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@pytest.mark.parametrize("rows", [3, 8], ids=["slab-3", "slab-8"])
def test_the_streamed_slabs_agree_with_the_whole_volume(device: torch.device, rows: int):
    """A slab puts every sample where the whole volume puts it, to a fraction of the data's range.

    A stored transform never factorises, so this is the general path: the blend goes to
    ``grid_sample``, one fused kernel worth 4x, which takes NORMALISED coordinates and so divides by
    the extent of the tensor handed to it -- and a slab is handed a window. That single region-local
    number is the whole of the disagreement. Everything else in the path is global, which is why it
    stays at rounding instead of moving a sample.

    The bound is far above what is measured and far below a moved map, which is wrong by voxels.
    Bit-identity still holds on the separable path, and ``test_resample.py`` pins it there.
    """
    image = _image()
    attribute = _attribute(image)
    volume = torch.from_numpy(sitk.GetArrayFromImage(image)).unsqueeze(0).to(device)
    for label, transform in _families(image):
        stage = _stage(image, transform)
        reference = stage(CASE, volume, Attribute(attribute))
        streamed = torch.empty_like(reference)
        for start in range(0, SIZE[0], rows):
            stop = min(start + rows, SIZE[0])
            target = (slice(start, stop), slice(0, SIZE[1]), slice(0, SIZE[2]))
            source = tuple(stage.stream_region_source(CASE, target, list(SIZE), Attribute(attribute)))
            block = volume[(slice(None), *source)]
            context = RegionContext(source, target, tuple(SIZE), tuple(SIZE))
            streamed[(slice(None), *target)] = stage.stream_region(CASE, block, context, Attribute(attribute))
        span = float((reference.max() - reference.min()).item())
        torch.testing.assert_close(streamed, reference, rtol=0.0, atol=1e-5 * span, msg=label)


def _slab_reads(stage: ResampleTransform, attribute: Attribute, rows: int) -> list[int]:
    reads = []
    for start in range(0, SIZE[0], rows):
        target = (slice(start, min(start + rows, SIZE[0])), slice(0, SIZE[1]), slice(0, SIZE[2]))
        source = stage.stream_region_source(CASE, target, list(SIZE), Attribute(attribute))
        reads.append(int(np.prod([part.stop - part.start for part in source])))
    return reads


def test_a_slab_pulls_a_bounded_source_window():
    # The point of the whole exercise: on a map aligned with the storage axes a slab reads a slab,
    # not the volume.
    image = _image(oblique=False)
    stage = _stage(image, sitk.TranslationTransform(3, (2.0, -1.5, 1.0)))
    attribute = _attribute(image)
    for start in range(0, SIZE[0], 4):
        target = (slice(start, min(start + 4, SIZE[0])), slice(0, SIZE[1]), slice(0, SIZE[2]))
        source = stage.stream_region_source(CASE, target, list(SIZE), Attribute(attribute))
        assert source[0].stop - source[0].start <= 8, "a translated slab pulled more than its neighbours"
    assert sum(_slab_reads(stage, attribute, 4)) < 3 * np.prod(SIZE)


def test_an_oblique_map_does_not_bound_and_the_numbers_say_so():
    # Not a defect of the bound -- the bound is exact here. A thin slab rotated against the storage
    # axes genuinely has an axis-aligned source box covering most of the volume, and it gets worse
    # the finer the decomposition. This is the measurement a cost model exists to report, and the
    # reason streaming this map is not automatically worth it.
    image = _image(oblique=True)
    stage = _stage(image, _euler(image))
    attribute = _attribute(image)
    coarse = sum(_slab_reads(stage, attribute, SIZE[0])) / np.prod(SIZE)
    fine = sum(_slab_reads(stage, attribute, 4)) / np.prod(SIZE)
    assert fine > coarse, f"amplification must grow as slabs thin: {coarse:.2f}x then {fine:.2f}x"
    assert fine > 2.0, f"the oblique fixture is meant to be expensive, got {fine:.2f}x"


class TestLocality:
    def test_a_boundable_cohort_declares_regrid(self):
        image = _image()
        for label, transform in _families(image):
            stage = _stage(image, transform)
            assert stage.patch_locality(_attribute(image)).kind is LocalityKind.REGRID, label

    def test_a_case_without_geometry_falls_back_and_says_why(self):
        image = _image()
        stage = ResampleTransform(transforms={"reg": False})
        stage.set_datasets([_StoredTransform("reg", _euler(image))])
        stage.transform_shape("", CASE, list(SIZE), Attribute())  # no Origin/Spacing/Direction
        locality = stage.patch_locality(Attribute())  # judged on the header handed over
        assert locality.kind is LocalityKind.WHOLE_VOLUME
        assert "physical space" in (locality.reason or "")

    def test_a_missing_transform_falls_back_and_says_which_group(self):
        image = _image()
        stage = ResampleTransform(transforms={"absent": False})
        stage.set_datasets([_StoredTransform("reg", _euler(image))])
        stage.transform_shape("", CASE, list(SIZE), _attribute(image))
        locality = stage.patch_locality(_attribute(image))
        assert locality.kind is LocalityKind.WHOLE_VOLUME
        assert "absent" in (locality.reason or "")

    def test_a_spline_order_with_no_kernel_falls_back_instead_of_crashing_mid_run(self):
        """ITK writes orders 0 and 2 as readily as 3, and neither has a kernel here.

        The refusal has to happen where the value is BUILT, not where it is finally sampled: a stage
        that decodes such a spline without complaint declares REGRID, passes the plan, and raises on
        the first region -- which is halfway through a run, per case, after bytes are already
        written. Refused at decode, it is one more whole-volume line in the plan.
        """
        image = _image()
        quadratic = sitk.BSplineTransformInitializer(image, [5] * 3, 2)
        size = np.asarray(quadratic.GetParameters()).size
        quadratic.SetParameters(list(np.random.RandomState(3).uniform(-5.0, 5.0, size)))

        stage = ResampleTransform(transforms={"reg": False})
        stage.set_datasets([_StoredTransform("reg", quadratic)])
        stage.transform_shape("", CASE, list(SIZE), _attribute(image))

        locality = stage.patch_locality(_attribute(image))
        assert locality.kind is LocalityKind.WHOLE_VOLUME
        assert "order 2" in (locality.reason or "")

    def test_inverting_a_spline_falls_back_with_the_remedy(self):
        image = _image()
        stage = ResampleTransform(transforms={"reg": True})
        stage.set_datasets([_StoredTransform("reg", _bspline(image))])
        stage.transform_shape("", CASE, list(SIZE), _attribute(image))
        locality = stage.patch_locality(_attribute(image))
        assert locality.kind is LocalityKind.WHOLE_VOLUME
        assert "Store the inverse" in (locality.reason or "")

    def test_inverting_a_rigid_map_is_exact_and_still_streams(self):
        image = _image()
        transform = _euler(image)
        stage = ResampleTransform(transforms={"reg": True})
        stage.set_datasets([_StoredTransform("reg", transform)])
        stage.transform_shape("", CASE, list(SIZE), _attribute(image))
        assert stage.patch_locality(_attribute(image)).kind is LocalityKind.REGRID
        volume = torch.from_numpy(sitk.GetArrayFromImage(image)).unsqueeze(0)
        want = sitk.GetArrayFromImage(sitk.Resample(image, image, transform.GetInverse(), sitk.sitkLinear, 0.0))
        got = stage(CASE, volume, _attribute(image)).squeeze(0).numpy()
        assert float(np.abs(want - got).max()) <= 1e-3 * float(np.abs(want).max())


class TestSampling:
    def test_a_label_map_is_not_blended(self):
        labels = (np.abs(_phantom()) % 5).astype(np.uint8)
        image = sitk.GetImageFromArray(labels)
        image.SetSpacing((0.8, 1.2, 1.5))
        image.SetOrigin((10.0, -5.0, 2.0))
        stage = _stage(image, _euler(image))
        got = stage(CASE, torch.from_numpy(labels).unsqueeze(0), _attribute(image)).squeeze(0).numpy()
        want = sitk.GetArrayFromImage(sitk.Resample(image, image, _euler(image), sitk.sitkNearestNeighbor, 0.0))
        np.testing.assert_array_equal(got, want)
        assert set(np.unique(got)) <= set(np.unique(labels))

    def test_the_fill_reaches_where_the_map_leaves_the_source(self):
        image = _image(oblique=False)
        transform = sitk.TranslationTransform(3, (12.0, 9.0, 6.0))
        stage = _stage(image, transform, fill=-1234.0)
        volume = torch.from_numpy(sitk.GetArrayFromImage(image)).unsqueeze(0)
        got = stage(CASE, volume, _attribute(image)).squeeze(0).numpy()
        want = sitk.GetArrayFromImage(sitk.Resample(image, image, transform, sitk.sitkLinear, -1234.0))
        assert 0 < int((want == -1234.0).sum()) < want.size
        np.testing.assert_array_equal(got == -1234.0, want == -1234.0)


class TestRefusals:
    def test_an_unknown_interpolation_is_refused_at_construction(self):
        with pytest.raises(TransformError, match="interpolation"):
            ResampleTransform(transforms={"reg": False}, interpolation="bspline")

    def test_no_transforms_is_refused_at_construction(self):
        with pytest.raises(TransformError, match="at least one group"):
            ResampleTransform(transforms={})

    def test_the_inverse_direction_says_what_to_do_instead(self):
        stage = ResampleTransform(transforms={"reg": False})
        assert stage.inverse_patch_locality(Attribute()).kind is LocalityKind.WHOLE_VOLUME
        with pytest.raises(TransformError, match="inverse: false"):
            stage.inverse(CASE, torch.zeros(1, 2, 2, 2), Attribute())


class TestCompositeOrder:
    def test_two_groups_compose_as_they_always_did(self):
        # Declaration order has always meant SimpleITK's composite order (last declared applied
        # first), because this stage built a CompositeTransform from the declared list. Decoding
        # normalizes to application order, so the reversal has to be reinstated -- and pinned.
        image = _image(oblique=False)
        first = sitk.TranslationTransform(3, (4.0, 0.0, 0.0))
        second = sitk.ScaleTransform(3, (1.5, 1.5, 1.5))
        stage = ResampleTransform(transforms={"a": False, "b": False})

        class _Two:
            def is_dataset_exist(self, group: str, name: str) -> bool:
                del name
                return group in ("a", "b")

            def read_transform(self, group: str, name: str):
                del name
                return first if group == "a" else second

        stage.set_datasets([_Two()])
        stage.transform_shape("", CASE, list(SIZE), _attribute(image))
        volume = torch.from_numpy(sitk.GetArrayFromImage(image)).unsqueeze(0)
        got = stage(CASE, volume, _attribute(image)).squeeze(0).numpy()
        want = sitk.GetArrayFromImage(
            sitk.Resample(image, image, sitk.CompositeTransform([first, second]), sitk.sitkLinear, 0.0)
        )
        assert float(np.abs(want - got).max()) <= 1e-3 * float(np.abs(want).max())

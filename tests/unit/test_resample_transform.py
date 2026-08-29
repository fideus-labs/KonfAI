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

"""``Resample`` through stored transforms: against SimpleITK, and against its own whole-volume path.

The fixture is high-frequency and its direction cosines are oblique, because a smooth phantom on
an axis-aligned grid passes a map that is wrong in exactly the ways this stage can be wrong.
"""

import sys
from pathlib import Path

import konfai.data.transform as transform_module
import numpy as np
import pytest
import torch
from konfai.data.transform import (
    LocalityKind,
    RegionContext,
    Resample,
    _optional_image_filler,
    _SitkInput,
)
from konfai.utils.dataset import Attribute
from konfai.utils.errors import TransformError

sitk = pytest.importorskip("SimpleITK")

SIZE = (22, 28, 34)
CASE = "CASE_000"
GOLDEN = Path(__file__).resolve().parents[1] / "assets" / "golden" / "resample.npz"


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
    """The smallest thing that answers what a stored-transform ``Resample`` asks of a ``Dataset``."""

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


def _stage(image, transform, **kwargs) -> Resample:
    stage = Resample(transforms={"reg": False}, **kwargs)
    stage.set_datasets([_StoredTransform("reg", transform)])
    stage.transform_shape("", CASE, list(SIZE), _attribute(image))
    return stage


DEVICES = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
DEVICE_IDS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _label_map() -> np.ndarray:
    """The phantom as a label map: six bands of the same field, one integer per band."""
    return (np.digitize(_phantom(), np.linspace(-200.0, 200.0, 6)) * 7).astype(np.uint16)


def _golden_resamples() -> dict[str, np.ndarray]:
    """Every case the golden fixture pins: the oblique phantom, as intensities and as a label map,
    through a stored rigid and a stored affine, on the host's exact route."""
    image = _image(oblique=True)
    attribute = _attribute(image)
    resampled = {}
    for map_name, transform in (("euler", _euler(image)), ("affine", _affine(image))):
        for dtype_name, volume, interpolation in (
            ("float32", _phantom(), "linear"),
            ("uint16", _label_map(), "nearest"),
        ):
            stage = _stage(image, transform, interpolation=interpolation)
            output = stage(CASE, torch.from_numpy(volume).unsqueeze(0), Attribute(attribute))
            resampled[f"{map_name}_{dtype_name}"] = output.squeeze(0).numpy()
    return resampled


def test_a_stored_resample_reproduces_its_golden_output() -> None:
    """The values this stage produced when the fixture was stored, on the CPU's exact route.

    The oracle tests above compare against SimpleITK, so they move with it; this one holds the
    numbers still across releases of KonfAI and of its dependencies. A label map through nearest
    must come back voxel-identical: nearest picks source voxels, so a coordinate walk that moved by
    a fraction of a voxel returns another label. Intensities through linear are held to 1e-6
    relative, the band the interpolation's last bits live in.

    A deliberate value change regenerates the fixture and states the deviation in the commit:
        python tests/unit/test_resample_transform.py --regenerate
    """
    assert GOLDEN.is_file(), f"the golden fixture is missing; regenerate it: {GOLDEN}"
    with np.load(GOLDEN) as stored:
        golden = dict(stored)
    produced = _golden_resamples()
    assert sorted(produced) == sorted(golden)

    for key, want in golden.items():
        got = produced[key]
        assert got.dtype == want.dtype and got.shape == want.shape, key
        if want.dtype == np.uint16:
            moved = int(np.count_nonzero(got != want))
            assert moved == 0, f"{key}: {moved} of {want.size} labels moved"
        else:
            deviation = float(np.abs(got.astype(np.float64) - want.astype(np.float64)).max())
            assert deviation <= 1e-6 * float(np.abs(want).max()), f"{key}: {deviation:.4g}"


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
    the extent of the tensor handed to it, and a slab is handed a window. That single region-local
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


def _slab_reads(stage: Resample, attribute: Attribute, rows: int) -> list[int]:
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
    # Not a defect of the bound: the bound is exact here. A thin slab rotated against the storage
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
        stage = Resample(transforms={"reg": False})
        stage.set_datasets([_StoredTransform("reg", _euler(image))])
        stage.transform_shape("", CASE, list(SIZE), Attribute())  # no Origin/Spacing/Direction
        locality = stage.patch_locality(Attribute())  # judged on the header handed over
        assert locality.kind is LocalityKind.WHOLE_VOLUME
        assert "physical space" in (locality.reason or "")

    def test_a_missing_transform_refuses_at_plan_time_and_says_which_group(self):
        """A map no route can apply refuses as the plan is built, not per case after bytes.

        Declaring WHOLE_VOLUME instead would print a fallback the run then contradicts by dying:
        the whole-volume path needs the same decode this refusal comes from.
        """
        image = _image()
        stage = Resample(transforms={"absent": False})
        stage.set_datasets([_StoredTransform("reg", _euler(image))])
        with pytest.raises(TransformError, match="absent"):
            stage.transform_shape("", CASE, list(SIZE), _attribute(image))

    def test_a_spline_order_with_no_kernel_refuses_at_plan_time(self):
        """ITK writes orders 0 and 2 as readily as 3, and neither has a kernel here.

        The refusal has to happen where the plan is built, not where the value is finally sampled:
        a stage that decodes such a spline without complaint passes the plan and raises on the
        first region (halfway through a run, per case, after bytes are already written), and the
        whole-volume path raises the identical error, so there is no fallback to declare.
        """
        image = _image()
        quadratic = sitk.BSplineTransformInitializer(image, [5] * 3, 2)
        size = np.asarray(quadratic.GetParameters()).size
        quadratic.SetParameters(list(np.random.RandomState(3).uniform(-5.0, 5.0, size)))

        stage = Resample(transforms={"reg": False})
        stage.set_datasets([_StoredTransform("reg", quadratic)])
        with pytest.raises(TransformError, match="order 2"):
            stage.transform_shape("", CASE, list(SIZE), _attribute(image))

    def test_inverting_a_spline_refuses_at_plan_time_with_the_remedy(self):
        image = _image()
        stage = Resample(transforms={"reg": True})
        stage.set_datasets([_StoredTransform("reg", _bspline(image))])
        with pytest.raises(TransformError, match="Store the inverse"):
            stage.transform_shape("", CASE, list(SIZE), _attribute(image))

    def test_inverting_a_rigid_map_is_exact_and_still_streams(self):
        image = _image()
        transform = _euler(image)
        stage = Resample(transforms={"reg": True})
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
            Resample(transforms={"reg": False}, interpolation="bspline")

    def test_no_transforms_is_refused_at_construction(self):
        with pytest.raises(TransformError, match="empty 'transforms'"):
            Resample(transforms={})

    def test_the_inverse_direction_says_what_to_do_instead(self):
        stage = Resample(transforms={"reg": False})
        assert stage.inverse_patch_locality(Attribute()).kind is LocalityKind.WHOLE_VOLUME
        with pytest.raises(TransformError, match="inverse: false"):
            stage.inverse(CASE, torch.zeros(1, 2, 2, 2), Attribute())


class TestCompositeOrder:
    def test_two_groups_compose_as_they_always_did(self):
        # Declaration order has always meant SimpleITK's composite order (last declared applied
        # first), because this stage built a CompositeTransform from the declared list. Decoding
        # normalizes to application order, so the reversal has to be reinstated, and pinned.
        image = _image(oblique=False)
        first = sitk.TranslationTransform(3, (4.0, 0.0, 0.0))
        second = sitk.ScaleTransform(3, (1.5, 1.5, 1.5))
        stage = Resample(transforms={"a": False, "b": False})

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


class _CrossFrameStore:
    """One root answering both lookups a cross-frame resample makes: the reference's header and the
    stored transform bridging the frames."""

    def __init__(self, reference: "sitk.Image", transform: "sitk.Transform") -> None:
        self.reference = reference
        self.transform = transform

    def is_dataset_exist(self, group: str, name: str) -> bool:
        del name
        return group in ("Reference", "reg")

    def get_infos(self, group: str, name: str):
        del group, name
        return [1, *list(self.reference.GetSize())[::-1]], _attribute(self.reference)

    def read_transform(self, group: str, name: str) -> "sitk.Transform":
        del group, name
        return self.transform


def test_a_stored_map_bridging_disjoint_frames_is_not_refused_as_disjoint():
    """An MR and a CT can sit 1000 mm apart in stage coordinates with a rigid bridging them.

    The all-fill refusal gates on coverage, and coverage must be judged THROUGH the declared map:
    judged before applying it, every cross-frame registration apply (the situation the stage
    exists to serve) is refused as disjoint. The counter-assert keeps the gate alive: a map that
    leads nowhere still refuses.
    """
    from konfai.data.transform import Resample

    case = _image(oblique=False)
    case.SetOrigin((1000.0, 0.0, 0.0))
    reference = _image(oblique=False)  # origin (10, -5, 2): ~1000 mm from the case in x
    bridge = sitk.TranslationTransform(3, (990.0, 5.0, -2.0))  # target world -> case world

    stage = Resample(reference="ref", reference_group="Reference", transforms={"reg": False})
    stage.set_datasets([_CrossFrameStore(reference, bridge)])
    assert stage.transform_shape("", CASE, list(SIZE), _attribute(case)) == list(SIZE)
    assert stage.coverage(CASE) > 0.9

    astray = sitk.TranslationTransform(3, (500000.0, 0.0, 0.0))
    refused = Resample(reference="ref", reference_group="Reference", transforms={"reg": False})
    refused.set_datasets([_CrossFrameStore(reference, astray)])
    with pytest.raises(TransformError, match="nothing but 'fill'"):
        refused.transform_shape("", CASE, list(SIZE), _attribute(case))


def test_cubic_tracks_the_bspline_oracle_closer_than_linear() -> None:
    """Keys' cubic is a real cubic reconstruction: brought to another density, the phantom must
    land nearer SimpleITK's spline resample than the linear blend does."""
    image = _image(oblique=False)
    volume = torch.from_numpy(sitk.GetArrayFromImage(image)).unsqueeze(0)
    outputs: dict[str, torch.Tensor] = {}
    attribute = _attribute(image)
    for interpolation in ("linear", "cubic"):
        attribute = _attribute(image)
        stage = Resample(spacing=[0.5, 0.9, 1.1], interpolation=interpolation)
        outputs[interpolation] = stage(CASE, volume.clone(), attribute)
    reference = sitk.Image([int(extent) for extent in reversed(outputs["cubic"].shape[1:])], sitk.sitkFloat32)
    reference.SetOrigin(tuple(attribute.get_np_array("Origin")))
    reference.SetSpacing(tuple(attribute.get_np_array("Spacing")))
    reference.SetDirection(tuple(attribute.get_np_array("Direction")))
    oracle = sitk.GetArrayFromImage(sitk.Resample(image, reference, sitk.Transform(), sitk.sitkBSpline, 0.0))
    crop = (slice(2, -2),) * 3  # away from the border, where tap-clamp and spline-edge policies differ
    error = {name: float(np.abs(out.numpy()[0][crop] - oracle[crop]).mean()) for name, out in outputs.items()}
    assert error["cubic"] < error["linear"]


def test_cubic_saturates_overshoot_instead_of_wrapping() -> None:
    """Keys' negative lobes overshoot beside an edge, and an integer cast would WRAP the overshoot
    to the opposite extreme: a uint8 step must stay a step, saturated at its plateaus."""
    volume = torch.zeros((1, 4, 4, 32), dtype=torch.uint8)
    volume[..., 16:] = 255
    attribute = Attribute()
    attribute["Origin"] = np.zeros(3)
    attribute["Spacing"] = np.ones(3)
    attribute["Direction"] = np.eye(3).ravel()
    stage = Resample(spacing=[0.5, 0.0, 0.0], interpolation="cubic")
    out = stage(CASE, volume, attribute)
    assert out.dtype == torch.uint8
    dark, bright = out[..., : out.shape[-1] // 2 - 2], out[..., out.shape[-1] // 2 + 2 :]
    assert int(dark.max()) <= 50
    assert int(bright.min()) >= 200


class _FieldPerCase:
    """A dense field per case, built from the case's name: what a cohort's transform group answers."""

    def is_dataset_exist(self, group: str, name: str) -> bool:
        del name
        return group == "reg"

    def read_transform(self, group: str, name: str) -> "sitk.Transform":
        del group
        rng = np.random.RandomState(int(name.rsplit("_", 1)[1]))
        image = sitk.GetImageFromArray(rng.normal(0.0, 2.0, (*SIZE, 3)), isVector=True)
        return sitk.DisplacementFieldTransform(image)


def test_the_plan_keeps_a_case_s_bound_and_the_run_holds_one_case_s_stages():
    """Planning a cohort of dense fields keeps three vectors per case, not the fields: the object
    that crosses mp.spawn does not grow with the cohort, and a rank holds the stages of the case it
    is sampling and of no other."""
    import pickle

    image = _image(oblique=False)
    attribute = _attribute(image)
    field_bytes = int(np.prod(SIZE)) * 3 * 8

    def planned(cases: int) -> Resample:
        stage = Resample(transforms={"reg": False})
        stage.set_datasets([_FieldPerCase()])
        for case in range(cases):
            stage.transform_shape("", f"CASE_{case:03d}", list(SIZE), Attribute(attribute))
        return stage

    one, ten = planned(1), planned(10)
    assert not one._stored and not ten._stored, "the plan must not hold decoded stages"
    assert len(ten._maps) == 10
    assert len(pickle.dumps(ten)) < field_bytes, "the pickle carries a field's values"
    assert len(pickle.dumps(ten)) - len(pickle.dumps(one)) < 9 * 1024, "a case is its grid and its bound"

    volume = torch.from_numpy(sitk.GetArrayFromImage(image)).unsqueeze(0)
    first = ten("CASE_003", volume, Attribute(attribute))
    assert set(ten._stored) == {"CASE_003"}
    ten("CASE_007", volume, Attribute(attribute))
    assert set(ten._stored) == {"CASE_003", "CASE_007"}, "the last cases' stages are held, within the bound"
    ten.stored_stage_bytes = 1
    ten("CASE_005", volume, Attribute(attribute))
    assert set(ten._stored) == {"CASE_005"}, "past the bound, the slot follows the case being sampled"
    # The same values as a stage that decoded the map once and kept it.
    kept = planned(10)
    kept._stored["CASE_003"] = kept._decode_stored("CASE_003")
    torch.testing.assert_close(first, kept("CASE_003", volume, Attribute(attribute)), rtol=0.0, atol=0.0)
    # What a rank receives holds no stages either; it decodes what it samples.
    rank = pickle.loads(pickle.dumps(ten))
    assert not rank._stored and set(rank._maps) == set(ten._maps)
    torch.testing.assert_close(rank("CASE_003", volume, Attribute(attribute)), first, rtol=0.0, atol=0.0)


@pytest.mark.slow
def test_the_slabbed_walk_lands_each_slab_in_the_one_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Above the walk budget the general path gathers slab by slab, and each slab is written into
    the output as it lands: bit for bit the single pass, with no parts held for a cat."""
    from konfai.data import transform as transform_module

    image = _image(oblique=True)
    attribute = _attribute(image)
    volume = torch.from_numpy(sitk.GetArrayFromImage(image)).unsqueeze(0)
    # Cubic: the host resampler declines it, so the walk serves the region on the CPU too.
    stage = _stage(image, _euler(image), interpolation="cubic")
    whole = stage(CASE, volume, Attribute(attribute))
    monkeypatch.setattr(transform_module, "walk_rows", lambda *args, **kwargs: 5)
    joins: list[int] = []
    cat = torch.cat
    monkeypatch.setattr(torch, "cat", lambda *args, **kwargs: joins.append(1) or cat(*args, **kwargs))
    slabbed = stage(CASE, volume, Attribute(attribute))
    assert torch.equal(slabbed, whole)
    assert not joins


def _diagonal_maps(image) -> list[tuple[str, "sitk.Transform"]]:
    translation = sitk.TranslationTransform(3, (2.3, -1.7, 0.9))
    scale = sitk.ScaleTransform(3, (1.15, 0.9, 1.3))
    scale.SetCenter(image.TransformContinuousIndexToPhysicalPoint([(s - 1) / 2 for s in image.GetSize()]))
    return [
        ("translation", translation),
        ("scale", scale),
        ("translation-then-scale", sitk.CompositeTransform([translation, scale])),
    ]


@pytest.mark.slow
def test_a_diagonal_stored_map_resamples_separably_within_the_routes_it_replaces() -> None:
    """A stored translation, an axis-aligned scale or their composition factorises: the region is
    read one axis at a time, from the very coordinates the general walk computes, and no coordinate
    per voxel is built. The values stay within the rounding of the two routes that served these
    maps before, the general walk and the host resampler, both kept here as oracles: measured at
    most 1.3e-6 of the range against the walk (grid_sample's normalisation) and 4.3e-8 against ITK
    (one float32 ulp) on a float32 blend, one unit on at most 0.23% of the voxels of an int16
    blend, and nothing at all on a nearest pick, where every route copies the same voxel.
    """
    from konfai.data.sampling import gather, source_index
    from konfai.data.transform import _resample_with_sitk

    image = _image(oblique=False)
    counts = (_phantom() * 4.0).astype(np.int16)
    for array in (_phantom(), counts):
        payload = sitk.GetImageFromArray(array)
        payload.CopyInformation(image)
        attribute = _attribute(payload)
        volume = torch.from_numpy(array).unsqueeze(0)
        span = float(array.max() - array.min())
        for label, transform in _diagonal_maps(payload):
            for mode in ("linear", "nearest"):
                stage = _stage(payload, transform, interpolation=mode)
                assert stage.slab_height_sensitive(CASE) is False, label
                got = stage(CASE, volume, Attribute(attribute))
                source, target = stage._grids_of(CASE)
                stages = stage._stages(CASE, target)
                walk = gather(
                    volume, source_index(target, source, stages, torch.device("cpu")), [0, 0, 0], list(SIZE), mode, 0.0
                )
                host = _resample_with_sitk(volume, target, source, stages, [0, 0, 0], mode, 0.0)
                assert host is not None
                if mode == "nearest":
                    assert torch.equal(got, walk) and torch.equal(got, host), label
                    continue
                for name, other, band in (("walk", walk, 3e-6), ("host", host, 1e-7)):
                    apart = (got.double() - other.double()).abs()
                    if array.dtype == np.int16:
                        assert float(apart.max()) <= 1.0, f"{label}/{name}: {float(apart.max())} units apart"
                        assert int((apart > 0).sum()) <= apart.numel() // 200, f"{label}/{name}: the seam is not a seam"
                    else:
                        assert float(apart.max()) <= band * span, f"{label}/{name}: {float(apart.max()):.3g}"
    rotated = _stage(image, _euler(image))
    assert rotated.slab_height_sensitive(CASE) is True


def test_a_diagonal_stored_map_matches_simpleitk_on_a_label_map() -> None:
    """The external oracle for the separable route through a stored map: a nearest pick through a
    translation and a scale lands every label where sitk.Resample lands it, byte for byte."""
    labels = (np.abs(_phantom()) % 7).astype(np.uint8)
    image = sitk.GetImageFromArray(labels)
    image.CopyInformation(_image(oblique=False))
    volume = torch.from_numpy(labels).unsqueeze(0)
    for label, transform in _diagonal_maps(image):
        stage = _stage(image, transform)
        got = stage(CASE, volume, _attribute(image)).squeeze(0).numpy()
        want = sitk.GetArrayFromImage(sitk.Resample(image, image, transform, sitk.sitkNearestNeighbor, 0.0))
        np.testing.assert_array_equal(got, want, err_msg=label)


class TestTheHostRouteReusesItsInputImage:
    """ITK's input image is filled in place from one region to the next, replaced on a change of
    shape or dtype (a same-length array is never reinterpreted), and never held past a whole-volume
    call or across a pickle."""

    def test_one_shape_fills_the_image_in_place_and_another_replaces_it(self):
        holder = _SitkInput()
        array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        first = holder.filled(array)
        assert holder.filled(array + 1.0) is first
        assert np.array_equal(sitk.GetArrayViewFromImage(first), array + 1.0)
        for other in (array.view(np.int32), array.reshape(3, 2, 4)):  # the same 96 bytes
            image = holder.filled(other)
            assert image is not first
            assert image.GetSize() == other.shape[::-1]
            assert np.array_equal(sitk.GetArrayViewFromImage(image), other)
        holder.drop()
        assert holder.filled(array) is not first

    def test_a_simpleitk_without_the_in_place_fill_is_still_a_simpleitk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The fill is a private symbol of the wrapper: optional on its own.

        Guarded in the same clause as the package, a SimpleITK not exporting it made the whole
        module read as no SimpleITK at all, and every stage needing one refused with an install
        hint for a package that is installed.
        """
        import SimpleITK.SimpleITK as core

        monkeypatch.delattr(core, "_SetImageFromArray")
        assert _optional_image_filler() is None
        assert transform_module.sitk is not None

        monkeypatch.setattr(transform_module, "_set_image_from_array", None)
        holder = _SitkInput()
        array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        first = holder.filled(array)
        assert holder.filled(array + 1.0) is not first
        assert np.array_equal(sitk.GetArrayViewFromImage(holder.filled(array + 1.0)), array + 1.0)

    def test_a_sweep_keeps_the_image_and_a_whole_volume_call_does_not(self):
        image = _image()
        attribute = _attribute(image)
        volume = torch.from_numpy(sitk.GetArrayFromImage(image)).unsqueeze(0)
        stage = _stage(image, _euler(image))  # never separable: the host route runs ITK
        target = (slice(0, 4), slice(0, SIZE[1]), slice(0, SIZE[2]))
        source = tuple(stage.stream_region_source(CASE, target, list(SIZE), Attribute(attribute)))
        context = RegionContext(source, target, tuple(SIZE), tuple(SIZE))
        stage.stream_region(CASE, volume[(slice(None), *source)], context, Attribute(attribute))
        held = stage._sitk_input._image
        assert held is not None
        stage.stream_region(CASE, volume[(slice(None), *source)], context, Attribute(attribute))
        assert stage._sitk_input._image is held
        assert stage.__getstate__()["_sitk_input"]._image is None
        stage.stream_abort(CASE)
        assert stage._sitk_input._image is None
        stage(CASE, volume, Attribute(attribute))
        assert stage._sitk_input._image is None


if __name__ == "__main__":  # the golden fixture's regeneration entry point
    if sys.argv[1:] != ["--regenerate"]:
        raise SystemExit(f"usage: python {Path(__file__).name} --regenerate")
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN, **_golden_resamples())
    print(f"wrote {GOLDEN} ({GOLDEN.stat().st_size / 1024:.0f} KiB)")

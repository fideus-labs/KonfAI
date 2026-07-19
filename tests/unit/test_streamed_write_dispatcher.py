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

"""The write mirror of the read-side patch-streaming dispatcher.

A finalize chain streams to the written image through its region stages — composed in any number,
each pulling through the next — scheduled by :class:`SlabRegionStream` from the stages' own
declarations (``inverse_patch_locality``, ``stream_region_target``), with the read side's promise:
streaming is an optimisation, so every streamed output must equal the whole-volume result voxel for
voxel. These tests prove that per region kind and in composition over random slab partitions, pin
the window to its bound, and pin the planner's classification of chains onto the direct / region /
buffered (chain-split) modes.
"""

from types import SimpleNamespace
from typing import ClassVar, cast

import numpy as np
import pytest
import torch
from konfai.data.data_manager import DatasetIter
from konfai.data.patching import SlabRegionStream, _halo_radii
from konfai.data.transform import (
    Canonical,
    Dilate,
    Flip,
    LocalityKind,
    Normalize,
    Padding,
    Permute,
    ResampleToResolution,
    Softmax,
    Standardize,
    TensorCast,
    Transform,
    TransformInverse,
)
from konfai.predictor import Mean, OutSameAsGroupDataset, Reduction, _FinalizeStage
from konfai.utils.dataset import Attribute
from konfai.utils.errors import PatchError

# --------------------------------------------------------------------------------------
# SlabRegionStream: each region kind, streamed over random slab partitions, must equal the
# whole-volume operator — and hold only a bounded window while doing it.
# --------------------------------------------------------------------------------------

C, Z, Y, X = 2, 8, 6, 5


def _run_stream(pull, produce, in_shape, out_shape, volume, partitions):
    """Push ``volume`` slab by slab and reassemble the emitted blocks into one output tensor."""
    stream = SlabRegionStream(pull, produce, in_shape, out_shape)
    out = torch.zeros([volume.shape[0], *out_shape], dtype=volume.dtype)
    written = torch.zeros(out_shape, dtype=torch.bool)
    emitted = []
    start = 0
    for stop in partitions:
        emitted += stream.push(slice(start, stop), volume[:, start:stop].clone())
        start = stop
    emitted += stream.finalize()
    for target, tensor in emitted:
        assert not written[target].any(), "a region was emitted twice"
        written[target] = True
        out[(slice(None), *target)] = tensor
    assert written.all(), "the emitted regions do not tile the output"
    return out


def _partitions(n: int, rng: np.random.Generator) -> list[int]:
    cuts = rng.choice(range(1, n), size=int(rng.integers(0, min(4, n - 1))), replace=False)
    return [*sorted(cuts.tolist()), n]


@pytest.mark.parametrize("seed", range(4))
def test_stream_orientation_mirrored_slab_axis_matches_whole_volume(seed: int) -> None:
    # Flip on the slab axis: input slabs arrive ascending, output regions descend — the scheduler must
    # discover the mirrored direction from the pull map alone and still tile the output exactly.
    rng = np.random.default_rng(seed)
    volume = torch.from_numpy(rng.standard_normal((C, Z, Y, X)).astype(np.float32))
    flip = Flip("0")
    reference = flip.inverse("case", volume, Attribute())
    got = _run_stream(
        lambda target: flip.stream_region_target(target, [Z, Y, X], Attribute()),
        lambda window, target, source: flip.inverse("case", window, Attribute()),
        [Z, Y, X],
        [Z, Y, X],
        volume,
        _partitions(Z, rng),
    )
    assert torch.equal(got, reference)


@pytest.mark.parametrize("seed", range(4))
def test_stream_orientation_permuted_slab_axis_matches_whole_volume(seed: int) -> None:
    # Permute z<->y: the input slab axis feeds output axis 1, so the emitted regions are slabs of a
    # NON-first output axis, written by random access.
    rng = np.random.default_rng(seed)
    volume = torch.from_numpy(rng.standard_normal((C, Z, Y, X)).astype(np.float32))
    permute = Permute("1|0|2")
    permuted = permute("case", volume, Attribute())
    reference = permute.inverse("case", permuted, Attribute())
    in_shape = [Y, Z, X]
    out_shape = permute.inverse_transform_shape(in_shape, Attribute())
    assert out_shape == [Z, Y, X]
    got = _run_stream(
        lambda target: permute.stream_region_target(target, in_shape, Attribute()),
        lambda window, target, source: permute.inverse("case", window, Attribute()),
        in_shape,
        out_shape,
        permuted,
        _partitions(Y, rng),
    )
    assert torch.equal(got, reference)


@pytest.mark.parametrize("seed", range(4))
def test_stream_orientation_canonical_inverse_matches_whole_volume(seed: int) -> None:
    # An identity-direction case: Canonical's forward remap mirrors x and y, so the inverse mirrors
    # them back INSIDE each slab while the slab axis maps identically.
    rng = np.random.default_rng(seed)
    volume = torch.from_numpy(rng.standard_normal((C, Z, Y, X)).astype(np.float32))
    canonical = Canonical()
    attribute = Attribute()
    attribute["Direction"] = np.eye(3).flatten()
    attribute["Spacing"] = np.ones(3)
    attribute["Origin"] = np.zeros(3)
    canonical_volume = canonical("case", volume, attribute)  # stacks the canonical geometry
    assert canonical.inverse_patch_locality(attribute).kind is LocalityKind.ORIENTATION
    reference = canonical.inverse("case", canonical_volume.clone(), Attribute(attribute))
    got = _run_stream(
        lambda target: canonical.stream_region_target(target, [Z, Y, X], Attribute(attribute)),
        lambda window, target, source: canonical.inverse("case", window, Attribute(attribute)),
        [Z, Y, X],
        canonical.inverse_transform_shape([Z, Y, X], attribute),
        canonical_volume,
        _partitions(Z, rng),
    )
    assert torch.equal(got, reference)


def test_forward_orientation_records_case_geometry_from_the_volume_not_the_slab() -> None:
    # A FORWARD orientation in the write pipe records the output origin from the extent it is handed;
    # streamed, that is a slab window, so it must be given the FULL volume extent instead (Canonical
    # documents "the extent is the VOLUME's, never a patch's"). Otherwise the sink origin is
    # derived from the slab height and diverges from the whole-volume path.
    ds = _output_dataset()
    canonical = Canonical()
    stage = _FinalizeStage(transform=canonical, inverted=False)
    in_shape = [Z, Y, X]
    direction = np.array([[1.0, 0, 0], [0, 0, 1], [0, 1, 0]]).flatten()  # swap y/z: origin then depends on the z-extent

    def _attr() -> Attribute:
        attribute = Attribute()
        attribute["Direction"] = direction
        attribute["Spacing"] = np.ones(3)
        attribute["Origin"] = np.array([7.0, 5.0, 3.0])
        return attribute

    reference = _attr()
    canonical.write_stream_cache_attribute(reference, in_shape)  # the whole-volume path's geometry

    slab_derived = _attr()
    canonical.write_stream_cache_attribute(slab_derived, [2, Y, X])  # what the slab shape would produce

    got = _attr()
    target = (slice(0, 2), slice(0, Y), slice(0, X))  # a 2-row slab, height != Z
    ds._apply_pipe_stage(
        stage, LocalityKind.ORIENTATION, torch.zeros(C, 2, Y, X), target, list(target), in_shape, in_shape, got, "case"
    )

    assert np.allclose(got.get_np_array("Origin"), reference.get_np_array("Origin"))
    assert not np.allclose(
        reference.get_np_array("Origin"), slab_derived.get_np_array("Origin")
    )  # the test distinguishes
    # The whole stack must match, not just the top Origin: a bug that pushed the slab geometry and then
    # the full-volume one would leave a stale entry lower in the stack for the later inverse pop to hit.
    assert got == reference


@pytest.mark.parametrize("seed", range(4))
def test_stream_crop_padding_inverse_matches_whole_volume(seed: int) -> None:
    # Padding's inverse is a translation that drops the padded border: the pulled window IS the block
    # (the CROP contract), including slabs that lie entirely inside the dropped border.
    rng = np.random.default_rng(seed)
    volume = torch.from_numpy(rng.standard_normal((C, Z, Y, X)).astype(np.float32))
    padding = Padding([0, 0, 1, 1, 2, 1])
    padded = padding("case", volume, Attribute())
    in_shape = [Z + 3, Y + 2, X]
    out_shape = padding.inverse_transform_shape(in_shape, Attribute())
    reference = padding.inverse("case", padded, Attribute())
    assert list(reference.shape[1:]) == out_shape
    got = _run_stream(
        lambda target: padding.stream_region_target(target, in_shape, Attribute()),
        lambda window, target, source: window,
        in_shape,
        out_shape,
        padded,
        _partitions(in_shape[0], rng),
    )
    assert torch.equal(got, reference)


@pytest.mark.parametrize("seed", range(4))
def test_stream_rescale_nearest_is_byte_identical_to_the_whole_volume_inverse(seed: int) -> None:
    # The streamed resample takes its index map from F.interpolate itself, so nearest (uint8) is
    # byte-identical to the whole-volume inverse — the exactness the RESCALE gate relies on.
    rng = np.random.default_rng(seed)
    volume = torch.from_numpy(rng.integers(0, 7, size=(C, Z, Y, X)).astype(np.uint8))
    resample = ResampleToResolution([1.0, 1.0, 1.0])
    attribute = Attribute()
    attribute["Spacing"] = torch.tensor([1.0, 1.0, 1.0])
    attribute["Size"] = np.asarray([12, 9, 7])  # the size the inverse restores
    attribute["Size"] = np.asarray([Z, Y, X])
    in_shape = [Z, Y, X]
    out_shape = resample.inverse_transform_shape(in_shape, attribute)
    assert out_shape == [12, 9, 7]
    scales = [in_shape[k] / out_shape[k] for k in range(3)]
    reference = resample.inverse("case", volume.clone(), Attribute(attribute))
    got = _run_stream(
        lambda target: resample.stream_region_target(target, in_shape, Attribute(attribute)),
        lambda window, target, source: resample.resample_region(
            window, target, [s.start for s in source], scales, in_shape
        ),
        in_shape,
        out_shape,
        volume,
        _partitions(Z, rng),
    )
    assert torch.equal(got, reference)


@pytest.mark.parametrize("seed", range(4))
def test_stream_rescale_linear_matches_the_whole_volume_inverse_to_float_rounding(seed: int) -> None:
    # A float (linear) rescale is not byte-identical to F.interpolate windowed, but resample_region
    # computes the same linear taps, so the streamed inverse matches the whole-volume one to
    # ~float-rounding (KONFAI_STREAM_LINEAR_RESAMPLE trades exactly this for a bounded window).
    rng = np.random.default_rng(seed)
    volume = torch.from_numpy(rng.standard_normal((C, Z, Y, X)).astype(np.float32)) * 100.0
    resample = ResampleToResolution([1.0, 1.0, 1.0])
    attribute = Attribute()
    attribute["Spacing"] = torch.tensor([1.0, 1.0, 1.0])
    attribute["Size"] = np.asarray([12, 9, 7])
    attribute["Size"] = np.asarray([Z, Y, X])
    in_shape = [Z, Y, X]
    out_shape = resample.inverse_transform_shape(in_shape, attribute)
    scales = [in_shape[k] / out_shape[k] for k in range(3)]
    reference = resample.inverse("case", volume.clone(), Attribute(attribute))
    got = _run_stream(
        lambda target: resample.stream_region_target(target, in_shape, Attribute(attribute)),
        lambda window, target, source: resample.resample_region(
            window, target, [s.start for s in source], scales, in_shape
        ),
        in_shape,
        out_shape,
        volume,
        _partitions(Z, rng),
    )
    assert got.shape == reference.shape
    torch.testing.assert_close(got, reference, atol=1e-2, rtol=0)  # ~1e-4 relative on values ~100


@pytest.mark.parametrize("seed", range(4))
def test_stream_halo_dilate_matches_whole_volume(seed: int) -> None:
    # A forward HALO stage (Dilate) rides the same window: the pull enlarges by the declared radius,
    # the stage runs on the window, and the halo is cropped back — seams must agree bit for bit.
    rng = np.random.default_rng(seed)
    volume = (torch.from_numpy(rng.standard_normal((C, Z, Y, X)).astype(np.float32)) > 0.7).to(torch.float32)
    dilate = Dilate(1)
    radii = _halo_radii(dilate.patch_locality(Attribute()).halo, 3)
    in_shape = [Z, Y, X]
    reference = dilate("case", volume, Attribute())

    def pull(target):
        return [
            slice(max(0, t.start - radius), min(extent, t.stop + radius))
            for t, radius, extent in zip(target, radii, in_shape, strict=False)
        ]

    def produce(window, target, source):
        result = dilate("case", window, Attribute())
        crop = tuple(slice(t.start - s.start, t.stop - s.start) for t, s in zip(target, source, strict=False))
        return result[(slice(None), *crop)]

    got = _run_stream(pull, produce, in_shape, in_shape, volume, _partitions(Z, rng))
    assert torch.equal(got, reference)


def test_stream_window_is_bounded_by_the_pull_span() -> None:
    # The whole point: the buffer holds the pull span of the pending output rows, never the volume.
    tall = 64
    volume = (torch.arange(1 * tall * Y * X).reshape(1, tall, Y, X) % 5).to(torch.uint8)
    resample = ResampleToResolution([1.0, 1.0, 1.0])
    attribute = Attribute()
    attribute["Spacing"] = torch.tensor([1.0, 1.0, 1.0])
    attribute["Size"] = np.asarray([96, Y, X])
    attribute["Size"] = np.asarray([tall, Y, X])
    in_shape = [tall, Y, X]
    out_shape = [96, Y, X]
    scales = [in_shape[k] / out_shape[k] for k in range(3)]
    stream = SlabRegionStream(
        lambda target: resample.stream_region_target(target, in_shape, Attribute(attribute)),
        lambda window, target, source: resample.resample_region(
            window, target, [s.start for s in source], scales, in_shape
        ),
        in_shape,
        out_shape,
    )
    emitted = []
    max_buffered = 0
    for z in range(0, tall, 8):
        emitted += stream.push(slice(z, z + 8), volume[:, z : z + 8].clone())
        max_buffered = max(max_buffered, sum(slab.shape[1] for _, slab in stream._slabs))
    emitted += stream.finalize()
    assert max_buffered <= 16, f"window held {max_buffered} of {tall} rows"
    reference = resample.inverse("case", volume.clone(), Attribute(attribute))
    out = torch.zeros([1, *out_shape], dtype=torch.uint8)
    for target, block in emitted:
        out[(slice(None), *target)] = block
    assert torch.equal(out, reference)


def test_stream_rejects_non_contiguous_slabs_and_incomplete_finalize() -> None:
    identity = SlabRegionStream(lambda t: list(t), lambda w, t, s: w, [4, 2], [4, 2])
    identity.push(slice(0, 2), torch.zeros(1, 2, 2))
    with pytest.raises(PatchError):
        identity.push(slice(3, 4), torch.zeros(1, 1, 2))
    with pytest.raises(PatchError):
        identity.finalize()


# --------------------------------------------------------------------------------------
# The write-mirror declarations: how each inverse answers, and the safe defaults.
# --------------------------------------------------------------------------------------


def test_inverse_locality_defaults_and_overrides() -> None:
    empty = Attribute()
    # A per-voxel value map inverts to a per-voxel value map; an index remap to an index remap.
    assert TensorCast().inverse_patch_locality(empty).kind is LocalityKind.POINTWISE
    assert Flip("0").inverse_patch_locality(empty).kind is LocalityKind.ORIENTATION
    assert Permute("1|0|2").inverse_patch_locality(empty).kind is LocalityKind.ORIENTATION
    # Stat-based inverses only pop what the forward stacked: pointwise on the finalize-time attribute.
    assert Normalize().inverse_patch_locality(empty).kind is LocalityKind.POINTWISE
    assert Standardize().inverse_patch_locality(empty).kind is LocalityKind.POINTWISE
    # Geometry inverses declare their own kind.
    assert Padding().inverse_patch_locality(empty).kind is LocalityKind.CROP
    # A resample inverse is patch-native only when the Size stack it pops is on the case.
    assert ResampleToResolution().inverse_patch_locality(empty).kind is LocalityKind.WHOLE_VOLUME
    seeded = Attribute()
    seeded["Size"] = np.asarray([4, 4, 4])
    seeded["Size"] = np.asarray([2, 2, 2])
    assert ResampleToResolution().inverse_patch_locality(seeded).kind is LocalityKind.RESCALE
    # Canonical judges the POPPED state: without a stacked direction there is nothing to invert onto.
    assert Canonical().inverse_patch_locality(empty).kind is LocalityKind.WHOLE_VOLUME

    # Any other kind falls to the safety net, exactly like the read side's default.
    class _OpaqueInverse(TransformInverse):
        def __call__(self, name, tensor, cache_attribute):
            return tensor

        def inverse(self, name, tensor, cache_attribute):
            return tensor

    assert _OpaqueInverse(True).inverse_patch_locality(empty).kind is LocalityKind.WHOLE_VOLUME


def test_canonical_oblique_direction_refuses_the_inverse_remap() -> None:
    attribute = Attribute()
    theta = np.deg2rad(30.0)
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    attribute["Direction"] = rotation.flatten()
    attribute["Direction"] = np.diag([-1.0, -1.0, 1.0]).flatten()  # what a forward pass stacked on top
    assert Canonical().inverse_patch_locality(attribute).kind is LocalityKind.WHOLE_VOLUME


# --------------------------------------------------------------------------------------
# The planner: which chains stream, through which mode.
# --------------------------------------------------------------------------------------


def _output_dataset(
    reduction: Reduction | None = None,
    before: list[Transform] | None = None,
    after: list[Transform] | None = None,
    final: list[Transform] | None = None,
    file_format: str = "h5",
    nb_data_augmentation: int = 1,
) -> OutSameAsGroupDataset:
    ds = OutSameAsGroupDataset.__new__(OutSameAsGroupDataset)
    ds.nb_data_augmentation = nb_data_augmentation
    ds.reduction = reduction if reduction is not None else Mean()
    ds.before_reduction_transforms = before or []
    ds.after_reduction_transforms = after or []
    ds.final_transforms = final or []
    ds.group_src, ds.group_dest = "src", "dest"
    ds.file_format = file_format
    return ds


def _dataset_iter(transforms: list[Transform]) -> DatasetIter:
    return cast(
        DatasetIter,
        SimpleNamespace(groups_src={"src": {"dest": SimpleNamespace(transforms=list(transforms))}}),
    )


def _geometry_attribute() -> Attribute:
    attribute = Attribute()
    attribute["Origin"] = np.zeros(3)
    attribute["Spacing"] = np.ones(3)
    attribute["Direction"] = np.eye(3).flatten()
    return attribute


def test_plan_all_pointwise_streams_direct() -> None:
    plan = _output_dataset(final=[TensorCast("float32", inverse=False)])._plan_stream(
        _dataset_iter([]), 0, _geometry_attribute()
    )
    assert plan is not None and plan.mode == "direct"


def test_plan_one_geometry_inverse_streams_through_the_region_stage() -> None:
    plan = _output_dataset()._plan_stream(_dataset_iter([Flip("0", inverse=True)]), 0, _geometry_attribute())
    assert plan is not None and plan.mode == "region"
    assert plan.pipe_start == 0 and plan.stages[0].inverted


def test_plan_several_region_stages_compose_into_one_streamed_pipe() -> None:
    # Padding + Flip inverses: region stages compose (each pulls through the next), so a multi-inverse
    # geometry chain still streams to the write — no chain is one region too many.
    plan = _output_dataset()._plan_stream(
        _dataset_iter([Flip("0", inverse=True), Padding([0, 0, 0, 0, 2, 1], inverse=True)]),
        0,
        _geometry_attribute(),
    )
    assert plan is not None and plan.mode == "region"
    assert plan.pipe_start == 0 and plan.tail_start == len(plan.stages)


def test_plan_whole_volume_stage_falls_to_the_buffered_tail_and_swallows_the_region() -> None:
    # [region, WHOLE_VOLUME]: the tail must start at the region stage, not after it — a buffered head
    # is pointwise-only so the buffer sits on the accumulator grid.
    plan = _output_dataset(final=[Softmax(1)])._plan_stream(
        _dataset_iter([Flip("0", inverse=True)]), 0, _geometry_attribute()
    )
    assert plan is not None and plan.mode == "buffered"
    assert plan.pipe_start is None and plan.tail_start == 0


def test_plan_seeded_global_stat_counts_as_pointwise_and_unseeded_does_not() -> None:
    # Normalize forward in the finalize chain needs the volume's Min/Max: with the statistic already on
    # the case it is a per-voxel map; without it each slab would derive its own — so it must be a tail.
    seeded = _geometry_attribute()
    seeded["Min"] = 0.0
    seeded["Max"] = 1.0
    plan = _output_dataset(final=[Normalize(inverse=False)])._plan_stream(_dataset_iter([]), 0, seeded)
    assert plan is not None and plan.mode == "direct"
    plan = _output_dataset(final=[Normalize(inverse=False)])._plan_stream(_dataset_iter([]), 0, _geometry_attribute())
    assert plan is not None and plan.mode == "buffered" and plan.tail_start == 0


def test_plan_refuses_tta_custom_reduction_and_non_pointwise_before_reduction() -> None:
    class _CustomReduction(Reduction):
        def __call__(self, tensors):
            return tensors[0]

    attribute = _geometry_attribute()
    assert _output_dataset(nb_data_augmentation=2)._plan_stream(_dataset_iter([]), 0, attribute) is None
    assert _output_dataset(reduction=_CustomReduction())._plan_stream(_dataset_iter([]), 0, attribute) is None
    assert _output_dataset(before=[Softmax(1)])._plan_stream(_dataset_iter([]), 0, attribute) is None


def test_plan_non_region_writable_format_buffers_and_writes_classically() -> None:
    # nrrd cannot serve region writes: the pointwise chain still streams the accumulator into a buffer
    # (the windowed-accumulator win survives), and the volume is written through the classic writer.
    plan = _output_dataset(file_format="nrrd")._plan_stream(_dataset_iter([]), 0, _geometry_attribute())
    assert plan is not None and plan.mode == "buffered" and plan.tail_start == len(plan.stages)


# --------------------------------------------------------------------------------------
# End to end at the unit level: a geometry inverse streamed through add_layer into a real sink.
# --------------------------------------------------------------------------------------


def _drive_prediction(tmp_path, transforms, volume, monkeypatch, streamed=True, before=(), final=()):
    """Push a whole case patch by patch through add_layer against an h5 sink and read the entry back.

    The forward ``transforms`` run once on ``volume`` (chained, stacking the attribute exactly as the
    input pipeline would), and the transformed volume plays the model output: the finalize chain must
    invert it back.
    """
    from konfai.utils.dataset import Dataset

    monkeypatch.setenv("KONFAI_STREAMED_WRITES", "1" if streamed else "0")
    monkeypatch.setenv("KONFAI_config_file", "unused.yml")
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    # Toy volumes sit far below the worth gate: zero it so the streamed machinery is exercised.
    monkeypatch.setenv("KONFAI_STREAM_WORTH_THRESHOLD", "0")

    attribute = _geometry_attribute()
    model_volume = volume.clone()
    for transform in transforms:
        model_volume = transform("CASE_000", model_volume, attribute)
    z = model_volume.shape[1]
    patch_slices = [
        (slice(k, k + 1), slice(0, model_volume.shape[2]), slice(0, model_volume.shape[3])) for k in range(z)
    ]

    class DummyPatch:
        patch_size: ClassVar[list[int]] = [1, model_volume.shape[2], model_volume.shape[3]]

        @staticmethod
        def get_patch_slices(index_augmentation: int):
            del index_augmentation
            return patch_slices

    class DummyManager:
        name = "CASE_000"
        patch = DummyPatch()
        shapes: ClassVar[list[list[int]]] = [list(model_volume.shape[1:])]
        cache_attributes: ClassVar[list[Attribute]] = [Attribute()]

    class DummyDatasetIter:
        groups_src: ClassVar[dict] = {
            "src": {"dest": SimpleNamespace(transforms=list(transforms), patch_transforms=[])}
        }

        @staticmethod
        def get_dataset_from_index(group_dest: str, index: int):
            return DummyManager()

    output_dataset = OutSameAsGroupDataset(
        same_as_group="src:dest",
        dataset_filename=f"{tmp_path}/output.h5:h5",
        group="out",
        patch_combine=None,
        reduction="Mean",
    )
    output_dataset.reduction = Mean()
    output_dataset.nb_data_augmentation = 1
    output_dataset.before_reduction_transforms = list(before)
    output_dataset.final_transforms = list(final)

    for k in range(z):
        # Patch.get_data squeezes size-1 patch axes, so a [1, Y, X] patch reaches add_layer as [C, Y, X].
        output_dataset.add_layer(
            0, 0, k, model_volume[:, k].clone(), cast(DatasetIter, DummyDatasetIter()), Attribute(attribute)
        )
        if output_dataset.is_done(0):
            result = output_dataset.get_output(0, [volume.shape[0]], cast(DatasetIter, DummyDatasetIter()))
            output_dataset.write_prediction(0, "CASE_000", result)

    data, _ = Dataset(f"{tmp_path}/output.h5", "h5").read_data("out", "CASE_000")
    return torch.from_numpy(data)


def test_add_layer_streams_a_flip_inverse_through_the_region_stage(tmp_path, monkeypatch) -> None:
    # The forward Flip was applied to the model input, so the accumulator holds the flipped volume and
    # the finalize chain flips it back; the streamed sink entry must equal the original bit for bit —
    # and equal what the whole-volume path (kill-switch) writes.
    volume = torch.from_numpy(np.random.default_rng(0).standard_normal((1, 6, 4, 3)).astype(np.float32))
    transforms = [Flip("0", inverse=True)]
    streamed = _drive_prediction(tmp_path / "streamed", transforms, volume, monkeypatch, streamed=True)
    reference = _drive_prediction(tmp_path / "reference", transforms, volume, monkeypatch, streamed=False)
    assert torch.equal(streamed, reference)
    assert torch.equal(streamed, volume)


def test_add_layer_streams_a_stat_inverse_riding_the_pipe(tmp_path, monkeypatch) -> None:
    # The common synthesis finalize: the forward Normalize stacked Min/Max, so its inverse is a
    # per-voxel affine map riding the pipe behind the flip's region.
    volume = torch.from_numpy(np.random.default_rng(2).standard_normal((1, 6, 4, 3)).astype(np.float32))
    transforms = [Normalize(inverse=True), Flip("0", inverse=True)]
    streamed = _drive_prediction(tmp_path / "streamed", transforms, volume, monkeypatch, streamed=True)
    reference = _drive_prediction(tmp_path / "reference", transforms, volume, monkeypatch, streamed=False)
    assert torch.equal(streamed, reference)


def test_add_layer_streams_a_forward_region_final_transform(tmp_path, monkeypatch) -> None:
    # A region transform applied FORWARD in final_transforms (reorient the written output): the pull
    # map is stream_region_source, the same declaration the read dispatcher uses.
    volume = torch.from_numpy(np.random.default_rng(3).standard_normal((1, 6, 4, 3)).astype(np.float32))
    final = [Flip("0", inverse=False)]
    streamed = _drive_prediction(tmp_path / "streamed", [], volume, monkeypatch, streamed=True, final=final)
    reference = _drive_prediction(tmp_path / "reference", [], volume, monkeypatch, streamed=False, final=final)
    assert torch.equal(streamed, reference)
    assert torch.equal(streamed, volume.flip(1))


def test_add_layer_streams_a_full_geometry_stack_through_the_composed_pipe(tmp_path, monkeypatch) -> None:
    # The general case the composition exists for: Canonical + ResampleToResolution + Padding forward,
    # so the finalize chain carries CROP + RESCALE + ORIENTATION in sequence. With the labelmap cast
    # to uint8 before the reduction, the whole stack streams to the sink and must match the
    # whole-volume path bit for bit.
    volume = (torch.arange(1 * 6 * 4 * 3).reshape(1, 6, 4, 3) % 5).to(torch.float32)
    transforms = [
        Canonical(inverse=True),
        ResampleToResolution([0.5, 0.5, -1.0], inverse=True),
        Padding([0, 0, 0, 0, 2, 1], inverse=True),
    ]
    before = [TensorCast("uint8", inverse=False)]
    streamed = _drive_prediction(tmp_path / "streamed", transforms, volume, monkeypatch, streamed=True, before=before)
    reference = _drive_prediction(
        tmp_path / "reference", transforms, volume, monkeypatch, streamed=False, before=before
    )
    assert streamed.dtype == reference.dtype
    assert torch.equal(streamed, reference)
    assert list(streamed.shape) == list(volume.shape)

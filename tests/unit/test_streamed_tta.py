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

"""Streamed test-time augmentation: the slab-synchronized cross-copy reduce.

A TTA case streams when every copy's un-augment acts slab by slab (a draw whose declared remap fixes
the slab axis): each copy keeps its own sliding-window accumulator, a :class:`SlabAligner` holds the
copies' finalized slabs until the slowest frontier passes, and the joint interval runs the SAME
per-copy head and reduction call as the whole-volume ``get_output`` — which is why the streamed
output must equal the assembled one bit for bit, for every reduction (Mean, Median, Concat), blend,
and dtype. A draw that moves the slab axis (a z-flip) must refuse and fall back, transparently."""

from types import SimpleNamespace
from typing import ClassVar, cast

import numpy as np
import pytest
import torch
from konfai.data.augmentation import Brightness, DataAugmentationsList, Flip, Permute
from konfai.data.data_manager import DatasetIter, _interleaved_case_entries
from konfai.data.patching import DatasetPatch, Gaussian, SlabAligner
from konfai.data.transform import Flip as FlipTransform
from konfai.data.transform import InferenceStack, LocalityKind, Sum
from konfai.predictor import Concat, Mean, Median, OutSameAsGroupDataset, Reduction
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.utils import get_patch_slices_from_shape

SHAPE = [6, 4, 3]
PATCH_SIZE = [2, 4, 3]
OVERLAP = 1
C = 2


def _geometry_attribute() -> Attribute:
    attribute = Attribute()
    attribute["Origin"] = np.zeros(3)
    attribute["Spacing"] = np.ones(3)
    attribute["Direction"] = np.eye(3).flatten()
    return attribute


def _augmentations(
    augmentation, nb: int = 1, shape: list[int] | None = None, case_index: int = 0
) -> DataAugmentationsList:
    """A one-augmentation list with its draw made for ``case_index`` (prob 1, so every copy is selected)."""
    augmentation.load(1.0)
    augmentations = DataAugmentationsList(nb=nb, data_augmentations={})
    augmentations.data_augmentations = [augmentation]
    augmentation.state_init(case_index, [list(shape or SHAPE) for _ in range(nb)], [Attribute() for _ in range(nb)])
    return augmentations


def _make_patches(volume: torch.Tensor, patch_slices) -> list[torch.Tensor]:
    """Cut ``volume`` into model-sized patches, border patches zero-padded up to the patch extent."""
    patches = []
    for patch_slice in patch_slices:
        patch = volume[(slice(None), *patch_slice)]
        padding = []
        for dim, s in reversed(list(enumerate(patch_slice))):
            padding += [0, PATCH_SIZE[dim] - (s.stop - s.start)]
        patches.append(torch.nn.functional.pad(patch, padding) if any(padding) else patch)
    return patches


def _drive_tta(
    tmp_path,
    monkeypatch,
    *,
    augmentation,
    streamed: bool,
    reduction: Reduction | None = None,
    transforms=(),
    after=(),
    dtype: torch.dtype = torch.float32,
    patch_combine=None,
    case_index: int = 0,
    file_format: str = "h5",
    worth_gate: bool = False,
):
    """Push one TTA case (identity copy + one augmented copy) patch by patch through ``add_layer``
    against an h5 sink, interleaved along the slab axis exactly as the prediction mapping orders it,
    and read the written entry back.

    The forward ``transforms`` run once on the volume (stacking the attribute as the input pipeline
    would) and the result plays the model output; the augmented copy is the augmentation's own
    ``compute`` of it. Returns the written tensor and whether the whole-volume path produced it."""
    monkeypatch.setenv("KONFAI_STREAMED_WRITES", "1" if streamed else "0")
    monkeypatch.setenv("KONFAI_config_file", "unused.yml")
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    # Toy volumes are far below the production worth-streaming threshold: zero it so these tests
    # exercise the streamed machinery; ``worth_gate`` keeps the production threshold instead.
    if worth_gate:
        monkeypatch.delenv("KONFAI_STREAMED_TTA_THRESHOLD", raising=False)
    else:
        monkeypatch.setenv("KONFAI_STREAMED_TTA_THRESHOLD", "0")

    attribute = _geometry_attribute()
    volume = torch.from_numpy(np.random.default_rng(0).standard_normal((C, *SHAPE)).astype(np.float32)).to(dtype)
    for transform in transforms:
        volume = transform("CASE_000", volume, attribute)
    augmentations = _augmentations(augmentation, shape=list(volume.shape[1:]), case_index=case_index)
    copies = [volume, augmentation.compute("CASE_000", case_index, 0, volume)]
    patch_slices, _ = get_patch_slices_from_shape(PATCH_SIZE, list(volume.shape[1:]), OVERLAP)
    patches = [_make_patches(copy, patch_slices) for copy in copies]

    class DummyPatch:
        patch_size: ClassVar[list[int]] = list(PATCH_SIZE)

        @staticmethod
        def get_patch_slices(index_augmentation: int):
            del index_augmentation
            return patch_slices

    class DummyManager:
        index = case_index
        name = "CASE_000"
        patch = DummyPatch()
        shapes: ClassVar[list[list[int]]] = [list(volume.shape[1:]), list(volume.shape[1:])]
        cache_attributes: ClassVar[list[Attribute]] = [Attribute(), Attribute()]

    class DummyDatasetIter:
        data_augmentations_list: ClassVar[list[DataAugmentationsList]] = [augmentations]
        groups_src: ClassVar[dict] = {
            "src": {"dest": SimpleNamespace(transforms=list(transforms), patch_transforms=[])}
        }

        @staticmethod
        def get_dataset_from_index(group_dest: str, index: int):
            return DummyManager()

    output_dataset = OutSameAsGroupDataset(
        same_as_group="src:dest",
        dataset_filename=f"{tmp_path}/output.h5:h5" if file_format == "h5" else f"{tmp_path}/output:{file_format}",
        group="out",
        patch_combine=None,
        reduction="Mean",
    )
    output_dataset.reduction = reduction if reduction is not None else Mean()
    output_dataset.nb_data_augmentation = 2
    output_dataset.after_reduction_transforms = list(after)
    for transform in output_dataset.after_reduction_transforms:
        transform.set_datasets([output_dataset])
    if patch_combine is not None:
        patch_combine.set_patch_config(list(PATCH_SIZE), OVERLAP)
        output_dataset.patch_combine = patch_combine

    dataset_iter = cast(DatasetIter, DummyDatasetIter())
    order = sorted((patch_slices[p][0].start, a, p) for a in range(2) for p in range(len(patch_slices)))
    whole_volume = False
    for _, a, p in order:
        output_dataset.add_layer(0, a, p, patches[a][p].clone(), dataset_iter, Attribute(attribute), [C])
        if output_dataset.is_done(0):
            result = output_dataset.get_output(0, [C], dataset_iter)
            output_dataset.write_prediction(0, "CASE_000", result)
            whole_volume = True

    output_dataset.finalize_writes()
    store = f"{tmp_path}/output.h5" if file_format == "h5" else f"{tmp_path}/output"
    data, _ = Dataset(store, file_format).read_data("out", "CASE_000")
    return torch.from_numpy(data), whole_volume


# --------------------------------------------------------------------------------------
# Byte-identity: the slab-synchronized reduce equals the whole-volume path, per reduction.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reduction", "dtype", "patch_combine"),
    [
        (Mean(), torch.float32, Gaussian()),
        (Mean(), torch.float16, None),
        (Median(), torch.float32, Gaussian()),
        (Median(), torch.float16, None),
    ],
)
def test_streamed_tta_flip_matches_whole_volume(tmp_path, monkeypatch, reduction, dtype, patch_combine) -> None:
    # An in-plane flip (y and x, never z): the copies' un-augment is slab-parallel, so the case must
    # stream (the whole-volume path never fires) and still match the assembled reference bit for bit.
    kwargs = {"reduction": reduction, "dtype": dtype, "patch_combine": patch_combine}
    streamed, whole_volume = _drive_tta(
        tmp_path / "streamed", monkeypatch, augmentation=Flip(f_prob=[0, 1, 1]), streamed=True, **kwargs
    )
    assert not whole_volume, "the TTA case should have streamed"
    reference, whole_volume = _drive_tta(
        tmp_path / "reference", monkeypatch, augmentation=Flip(f_prob=[0, 1, 1]), streamed=False, **kwargs
    )
    assert whole_volume
    assert streamed.dtype == reference.dtype
    assert torch.equal(streamed, reference)


def test_streamed_tta_concat_layout_matches_whole_volume(tmp_path, monkeypatch) -> None:
    # reduce=Concat keeps the copies as leading blocks for after_reduction to merge (the documented
    # contract): the streamed prefix must hand Sum the same [T, C, ...] layout, slab by slab.
    kwargs = {"reduction": Concat(), "after": [Sum(dim=0)]}
    streamed, whole_volume = _drive_tta(
        tmp_path / "streamed", monkeypatch, augmentation=Flip(f_prob=[0, 1, 1]), streamed=True, **kwargs
    )
    assert not whole_volume
    reference, _ = _drive_tta(
        tmp_path / "reference", monkeypatch, augmentation=Flip(f_prob=[0, 1, 1]), streamed=False, **kwargs
    )
    assert torch.equal(streamed, reference)


def test_streamed_tta_composes_with_a_region_pipe(tmp_path, monkeypatch) -> None:
    # TTA reduce feeding the write dispatcher's region pipe: the forward Flip on the slab axis leaves
    # a z-mirror inverse in the finalize chain, streamed AFTER the cross-copy reduction. Both layers
    # of the machinery compose without either knowing about the other.
    kwargs = {"transforms": [FlipTransform("0", inverse=True)]}
    streamed, whole_volume = _drive_tta(
        tmp_path / "streamed", monkeypatch, augmentation=Flip(f_prob=[0, 1, 1]), streamed=True, **kwargs
    )
    assert not whole_volume
    reference, _ = _drive_tta(
        tmp_path / "reference", monkeypatch, augmentation=Flip(f_prob=[0, 1, 1]), streamed=False, **kwargs
    )
    assert torch.equal(streamed, reference)


def test_light_tta_case_takes_the_whole_volume_path_by_the_worth_gate(tmp_path, monkeypatch, capsys) -> None:
    # A TTA case whose assembled accumulators are a sliver of allocatable memory has nothing for the
    # slab-synchronized reduce to save: the worth gate routes it whole-volume, output unchanged, and
    # the fallback is said once (a silent one would hide that a large case pays it).
    streamed, whole_volume = _drive_tta(
        tmp_path / "gated", monkeypatch, augmentation=Flip(f_prob=[0, 1, 1]), streamed=True, worth_gate=True
    )
    assert whole_volume, "a toy TTA case should not pay the streamed reduce"
    reference, _ = _drive_tta(tmp_path / "reference", monkeypatch, augmentation=Flip(f_prob=[0, 1, 1]), streamed=False)
    assert torch.equal(streamed, reference)
    assert capsys.readouterr().out.count("takes the whole-volume path") == 1


def test_streamed_tta_slab_axis_flip_falls_back_to_whole_volume(tmp_path, monkeypatch) -> None:
    # A z-flip mirrors the slab order: finalizing output slab 0 would need the copy's LAST patch, so
    # the gate must refuse and the case must complete through the whole-volume path, transparently.
    streamed, whole_volume = _drive_tta(
        tmp_path / "streamed", monkeypatch, augmentation=Flip(f_prob=[1, 0, 0]), streamed=True
    )
    assert whole_volume, "a slab-axis flip must fall back to the whole-volume path"
    reference, _ = _drive_tta(tmp_path / "reference", monkeypatch, augmentation=Flip(f_prob=[1, 0, 0]), streamed=False)
    assert torch.equal(streamed, reference)


# --------------------------------------------------------------------------------------
# The gate: slab-parallelism is read from the augmentation's own declarations.
# --------------------------------------------------------------------------------------


def _gate(augmentation, nb: int = 1) -> bool:
    augmentations = _augmentations(augmentation, nb=nb)

    class DummyManager:
        index = 0
        shapes: ClassVar[list[list[int]]] = [list(SHAPE)] * (nb + 1)

    class DummyDatasetIter:
        data_augmentations_list: ClassVar[list[DataAugmentationsList]] = [augmentations]

        @staticmethod
        def get_dataset_from_index(group_dest: str, index: int):
            return DummyManager()

    output_dataset = OutSameAsGroupDataset.__new__(OutSameAsGroupDataset)
    output_dataset.nb_data_augmentation = nb + 1
    output_dataset.group_dest = "dest"
    return output_dataset._tta_streamable(cast(DatasetIter, DummyDatasetIter()), 0, _geometry_attribute())


def test_gate_reads_the_draw_declarations() -> None:
    # In-plane flips fix every slab; a z-flip mirrors them; Permute always moves the slab axis
    # (its two generators both displace z); a per-voxel colour draw inverts per voxel.
    assert _gate(Flip(f_prob=[0, 1, 1]))
    assert not _gate(Flip(f_prob=[1, 0, 0]))
    assert not _gate(Permute(prob_permute=None), nb=2)
    assert _gate(Brightness(b_std=0.5))


# --------------------------------------------------------------------------------------
# SlabAligner: pure interval arithmetic over ordered emitters.
# --------------------------------------------------------------------------------------


def test_aligner_single_stream_passes_slabs_through() -> None:
    aligner = SlabAligner(1)
    slab = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    emitted = aligner.push(0, [(slice(0, 3), slab)], finished=False)
    assert len(emitted) == 1
    interval, rows = emitted[0]
    assert interval == slice(0, 3) and list(rows) == [0]
    assert torch.equal(rows[0], slab), "a lone stream's rows must be the slab itself"
    assert not aligner.complete
    aligner.push(0, [], finished=True)
    assert aligner.complete


def test_aligner_joint_intervals_carry_every_stream_and_stay_bounded() -> None:
    z = 8
    streams = [torch.randn(1, z, 2), torch.randn(1, z, 2)]
    partitions = {0: [2, 5, 8], 1: [1, 3, 4, 8]}  # deliberately skewed emission boundaries
    aligner = SlabAligner(2)
    outputs = {0: [], 1: []}
    consumed = 0
    for step in range(max(len(p) for p in partitions.values())):
        for stream, cuts in partitions.items():
            if step >= len(cuts):
                continue
            start = 0 if step == 0 else cuts[step - 1]
            stop = cuts[step]
            emitted = aligner.push(stream, [(slice(start, stop), streams[stream][:, start:stop])], finished=stop == z)
            for interval, rows in emitted:
                assert interval.start == consumed, "intervals must tile the axis in order"
                assert sorted(rows) == [0, 1]
                for key, block in rows.items():
                    outputs[key].append(block)
                consumed = interval.stop
            # Nothing beyond the slowest frontier is buffered once emitted rows are pruned.
            for pending in aligner._pending.values():
                assert all(start + slab.shape[1] > consumed for start, slab in pending)
    assert consumed == z and aligner.complete
    for key, stream_volume in enumerate(streams):
        assert torch.equal(torch.cat(outputs[key], dim=1), stream_volume)


# --------------------------------------------------------------------------------------
# The prediction mapping: copies advance together along the slab axis.
# --------------------------------------------------------------------------------------


def test_streamed_inference_stack_writes_the_stack_region_by_region(tmp_path, monkeypatch) -> None:
    # InferenceStack declares SLAB: per-voxel member reduction plus a per-region side write of the
    # stack. Fed through the streamed prefix it must produce the same main output AND the same stack
    # entry as the whole-volume call — here on a TTA Concat chain, where the stack holds both copies.
    def run(where: str, streamed: bool):
        stack = InferenceStack("", "stack", mode="Seg")
        streamed_out, whole_volume = _drive_tta(
            tmp_path / where,
            monkeypatch,
            augmentation=Flip(f_prob=[0, 1, 1]),
            streamed=streamed,
            reduction=Concat(),
            after=[stack],
        )
        stack_data, _ = Dataset(f"{tmp_path / where}/output.h5", "h5").read_data("InferenceStack", "CASE_000")
        return streamed_out, torch.from_numpy(stack_data), whole_volume

    assert InferenceStack("", "stack").patch_locality(Attribute()).kind is LocalityKind.SLAB
    got, got_stack, whole_volume = run("streamed", streamed=True)
    assert not whole_volume, "a SLAB stage must not force the whole-volume path"
    want, want_stack, _ = run("reference", streamed=False)
    assert torch.equal(got, want)
    assert torch.equal(got_stack, want_stack)


def test_streamed_inference_stack_buffers_when_the_sink_refuses_regions(tmp_path, monkeypatch) -> None:
    # A destination that cannot serve region writes must not lose the stack: the SLAB stage buffers
    # and writes classically at the last slab — the whole-volume path's memory, never a missing file.
    stack = InferenceStack(f"{tmp_path}/stack_only.h5:h5", "stack", mode="Seg")
    stack.dataset.open_data_stream = lambda *args, **kwargs: None  # type: ignore[union-attr,method-assign]
    _, whole_volume = _drive_tta(
        tmp_path / "streamed",
        monkeypatch,
        augmentation=Flip(f_prob=[0, 1, 1]),
        streamed=True,
        reduction=Concat(),
        after=[stack],
    )
    assert not whole_volume
    stack_data, _ = Dataset(f"{tmp_path}/stack_only.h5", "h5").read_data("InferenceStack", "CASE_000")
    assert stack_data.shape == (2, *SHAPE)


def test_streamed_tta_binds_the_draw_by_the_manager_case_index(tmp_path, monkeypatch) -> None:
    # A DDP shard remaps case indices to loader-local ones, but the draw was made under the manager's
    # own index — the only key ``who_index`` holds. Both paths must un-augment through it: with the
    # local index (0 here) the state lookup has no entry at all.
    streamed, whole_volume = _drive_tta(
        tmp_path / "streamed", monkeypatch, augmentation=Flip(f_prob=[0, 1, 1]), streamed=True, case_index=7
    )
    assert not whole_volume
    reference, _ = _drive_tta(
        tmp_path / "reference", monkeypatch, augmentation=Flip(f_prob=[0, 1, 1]), streamed=False, case_index=7
    )
    assert torch.equal(streamed, reference)


def test_streamed_inference_stack_aborts_its_sink_when_the_case_dies(monkeypatch) -> None:
    # A case that dies between slabs must not leave the stack's region sink open: the dispatcher's
    # error path calls stream_abort, which closes and drops whatever the stage held for the case.
    class _Sink:
        aborted = False

        def write_slice(self, target, array):
            del target, array

        def abort(self, error=None):
            # The partial stack must be REMOVED, never finalized under its final name.
            _Sink.aborted = True

    stack = InferenceStack("", "stack", mode="Seg")
    stack._stack_sinks["CASE_000"] = cast(Dataset, _Sink())  # type: ignore[assignment]
    stack.stream_abort("CASE_000")
    assert _Sink.aborted and not stack._stack_sinks
    stack._stack_buffers["CASE_000"] = [np.zeros((1, 1, 1, 1), dtype=np.float32)]
    stack.stream_abort("CASE_000")
    assert not stack._stack_buffers


def test_interleaved_case_entries_order_copies_by_slab_start() -> None:
    patch = DatasetPatch(patch_size=list(PATCH_SIZE), overlap=OVERLAP)
    patch.load(list(SHAPE), 0)
    patch.load(list(SHAPE), 1)
    entries = [(a, p) for a in range(2) for p in range(patch.get_size(0))]
    ordered = _interleaved_case_entries([patch, patch], entries)
    assert sorted(ordered) == sorted(entries), "the interleave must be a permutation of the case"
    starts = [patch.get_patch_slices(a)[p][0].start for a, p in ordered]
    assert starts == sorted(starts), "arrival must be non-decreasing along the slab axis"
    for a in range(2):
        within_copy = [p for entry_a, p in ordered if entry_a == a]
        assert within_copy == sorted(within_copy), "within a copy the patch order must be untouched"


def test_interleaved_case_entries_keep_the_plain_order_when_groups_disagree() -> None:
    # One shared arrival order must serve every destination group: when their grids disagree on the
    # slab starts (or one cannot even index an entry), the interleave silently steps aside — it is a
    # memory bound, never a correctness requirement.
    patch = DatasetPatch(patch_size=list(PATCH_SIZE), overlap=OVERLAP)
    patch.load(list(SHAPE), 0)
    patch.load(list(SHAPE), 1)
    other = DatasetPatch(patch_size=[3, 4, 3], overlap=OVERLAP)
    other.load(list(SHAPE), 0)
    other.load(list(SHAPE), 1)
    entries = [(a, p) for a in range(2) for p in range(patch.get_size(0))]
    assert _interleaved_case_entries([patch, other], entries) == entries

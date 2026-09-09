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


"""The prediction sink: patches blended per case and copy, reduced, written whole or streamed by slabs."""

import os
import queue
import threading
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from types import EllipsisType
from typing import cast

import numpy as np
import torch

from konfai import konfai_root
from konfai.data.augmentation import DataAugmentation
from konfai.data.data_manager import (
    DatasetIter,
)
from konfai.data.patching import (
    Accumulator,
    PathCombine,
    SlabAligner,
    SlabRegionStream,
    StreamingAccumulator,
    blend_axes,
    blend_overlap,
)
from konfai.data.patching.stage import _halo_radii, _HaloPull, _RemapPull
from konfai.data.reduction import Mean, Median, Reduction
from konfai.data.transform import (
    LocalityKind,
    PatchLocality,
    RegionContext,
    Transform,
    TransformInverse,
    TransformLoader,
)
from konfai.utils.budget import node_local_ranks, resolve_memory_budget
from konfai.utils.clock import SweepClock
from konfai.utils.config import _escape_key_component, apply_config, config
from konfai.utils.dataset import Attribute, Dataset, DataStream
from konfai.utils.errors import PredictorError
from konfai.utils.runtime import (
    NeedDevice,
)
from konfai.utils.utils import env_flag, get_module, split_path_spec

#: This rank's prediction loop, phase by phase, summed over the cases it ran (see ``_prediction_report``).
PREDICTION_CLOCK = SweepClock()


class _AsyncWriter:
    """A background thread owning one output dataset's disk writes, in submission order.

    The queue is bounded, so a slow destination back-pressures the loop. The first failure is kept and
    re-raised at the next submission and at ``close``; later operations drain unexecuted.
    """

    _CAPACITY = 4

    def __init__(self) -> None:
        self._queue: queue.Queue[Callable[[], None] | None] = queue.Queue(maxsize=self._CAPACITY)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="konfai-writer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            operation = self._queue.get()
            try:
                if operation is None:
                    return
                if self._error is None:
                    with PREDICTION_CLOCK.phase("write"):
                        operation()
            except BaseException as error:  # kept and re-raised on the loop thread
                self._error = error
            finally:
                self._queue.task_done()

    def submit(self, operation: Callable[[], None]) -> None:
        self._raise_pending()
        self._queue.put(operation)

    def close(self) -> None:
        """Drain every submitted operation, stop the thread, and surface any failure."""
        self._queue.put(None)
        self._thread.join()
        self._raise_pending()

    def _raise_pending(self) -> None:
        if self._error is not None:
            error, self._error = self._error, None
            raise error


def _slab_context(region: slice, spatial: list[int]) -> RegionContext:
    """Where one z-slab of the accumulator grid sits: the same region as source and target."""
    slices = (region, *(slice(0, int(extent)) for extent in spatial[1:]))
    shape = tuple(int(extent) for extent in spatial)
    return RegionContext(slices, slices, shape)


@dataclass(frozen=True)
class _FinalizeStage:
    """One step of the finalize chain, bound to how the chain applies it (forward or inverted)."""

    transform: Transform
    inverted: bool

    def locality(self, attribute: Attribute) -> PatchLocality:
        if self.inverted:
            return cast(TransformInverse, self.transform).inverse_patch_locality(attribute)
        return self.transform.patch_locality(attribute)

    def __call__(self, name: str, tensor: torch.Tensor, attribute: Attribute) -> torch.Tensor:
        if self.inverted:
            return cast(TransformInverse, self.transform).inverse(name, tensor, attribute)
        return self.transform(name, tensor, attribute)

    def stream_region(
        self, name: str, tensor: torch.Tensor, context: RegionContext, attribute: Attribute
    ) -> torch.Tensor:
        """The call, told where the region sits (both default to the whole-volume call)."""
        if self.inverted:
            return cast(TransformInverse, self.transform).stream_region_inverse(name, tensor, context, attribute)
        return self.transform.stream_region(name, tensor, context, attribute)


@dataclass(frozen=True)
class _StreamPlan:
    """How one case streams: the post-reduction stages, split into a per-slab pointwise prefix, a
    streamed pipe of region and pointwise stages, and a whole-volume tail.

    ``to_sink`` streams straight into a region-write ``DataStream``; ``pipe_start`` is the first region
    stage (``None`` when the chain is pointwise throughout) and the pipe runs from there to the end.
    Without ``to_sink`` the prefix streams into a post-reduction buffer and ``stages[tail_start:]``
    runs once on it. Invariants: ``to_sink`` implies no tail, a pipe implies ``to_sink``, and the
    buffer sits on the accumulator grid.
    """

    stages: list[_FinalizeStage]
    pipe_start: int | None
    tail_start: int
    to_sink: bool
    # Prefix stages declared SLAB: run through ``Transform.stream_slab`` so they learn where each slab sits.
    slab_stages: frozenset[int] = frozenset()

    @property
    def boundary(self) -> int:
        """Where the per-slab pointwise prefix ends."""
        return self.pipe_start if self.pipe_start is not None else self.tail_start

    @property
    def mode(self) -> str:
        if not self.to_sink:
            return "buffered"
        return "region" if self.pipe_start is not None else "direct"


@dataclass
class _RegionState:
    """One case's live streamed pipe: its slab scheduler and the geometry its closures share.

    ``shapes[i]`` is the spatial shape between pipe stage ``i - 1`` and ``i``: ``shapes[0]`` the
    accumulator's, ``shapes[-1]`` the written image's.
    """

    shapes: list[list[int]]
    stream: SlabRegionStream | None = None
    # The attribute the latest emission ran the pipe on: what the sink opens with.
    attribute: Attribute | None = None


# Below this fraction of allocatable memory, the case takes the whole-volume path.
# KONFAI_STREAM_WORTH_THRESHOLD overrides the fraction.
_STREAM_WORTH_MIN_FRACTION = 0.05


@config("OutputDataset")
class OutputDataset(Dataset, NeedDevice):
    """The sink of one model output: accumulates a case's patches across its copies and models,
    mirrors the geometry and transform chain of the input group ``same_as_group``, and writes the
    result, streamed by slabs whenever that is byte-identical to the assembled path."""

    def __init__(
        self,
        same_as_group: str = "default",
        dataset_filename: str = "default|./Dataset:mha",
        group: str = "default",
        before_reduction_transforms: dict[str, TransformLoader] = {"default|Normalize": TransformLoader()},
        after_reduction_transforms: dict[str, TransformLoader] = {"default|Normalize": TransformLoader()},
        final_transforms: dict[str, TransformLoader] = {"default|Normalize": TransformLoader()},
        patch_combine: str | None = None,
        reduction: str = "Mean",
        attributes: list[str] | None = None,
    ) -> None:
        filename, _, file_format = split_path_spec(dataset_filename)
        super().__init__(filename, file_format)
        # ``Dataset.__init__`` does not forward ``super().__init__()``: initialise ``NeedDevice`` here.
        NeedDevice.__init__(self)
        self.group = group
        self._before_reduction_transforms = before_reduction_transforms
        self._after_reduction_transforms = after_reduction_transforms
        self._final_transforms = final_transforms
        self._patch_combine = patch_combine
        # "key=value" strings: the config layer accepts list[str], not dict[str, str]. Same spelling as --set.
        self._attributes = dict(entry.split("=", 1) for entry in attributes or [])
        self.reduction_classpath = reduction
        self.reduction: Reduction
        #: The per-rank budget the streamed-vs-assembled route is priced against, pushed by the predictor.
        self._per_rank_budget_bytes: float | None = None

        self.before_reduction_transforms: list[Transform] = []
        self.after_reduction_transforms: list[Transform] = []
        self.final_transforms: list[Transform] = []
        self.patch_combine: PathCombine | None = None

        self.output_layer_accumulator: dict[int, dict[int, Accumulator]] = {}
        self.attributes: dict[int, dict[int, dict[int, Attribute]]] = {}
        self.names: dict[int, str] = {}
        self.nb_data_augmentation = 0
        # One reusable page-locked buffer for GPU->CPU offload, consumed synchronously.
        self._pin_buffer: torch.Tensor | None = None
        # Per-CASE blend device, decided once at the case's first patch (see ``_accumulate_device``):
        # CUDA when every augmentation's volume fits VRAM, else CPU. Never per (case, augmentation).
        self._accum_device: dict[int, torch.device] = {}
        # Same single-decision rule for the CPU-blend reduction device (see ``_reduction_device``).
        self._reduce_device: dict[int, torch.device] = {}
        # Disk writes go to a background writer when the destination serves disjoint files per entry
        # (``Dataset.concurrent_write_safe``) AND the output runs on a GPU; anything else stays inline.
        # ``KONFAI_ASYNC_WRITES`` is tri-state: unset = automatic, ``0`` kills, ``1`` forces.
        raw = os.environ.get("KONFAI_ASYNC_WRITES", "").lower()
        self._async_writes: bool | None
        if raw in ("0", "false") or not self.concurrent_write_safe():
            self._async_writes = False
        elif raw in ("1", "true"):
            self._async_writes = True
        else:
            self._async_writes = None  # decided at the first write, once the device is placed
        self._writer: _AsyncWriter | None = None
        self.group_src, self.group_dest = same_as_group.split(":")
        # Slab streaming has no config knob: applied per case whenever it is byte-identical to the
        # assembled path (``_plan_stream``). ``KONFAI_STREAMED_WRITES=0`` is a global kill-switch.
        self._streaming_enabled = env_flag("KONFAI_STREAMED_WRITES", True)
        self._stream_plans: dict[int, _StreamPlan | None] = {}
        self._stream_sinks: dict[int, DataStream] = {}
        self._region_states: dict[int, _RegionState] = {}
        self._stream_buffers: dict[int, torch.Tensor] = {}
        self._post_prefix_attributes: dict[int, Attribute] = {}
        self._reported_paths: set[str] = set()
        # One aligner per streamed case: the finalize needs every copy's rows together for the cross-copy reduction.
        self._aligners: dict[int, SlabAligner] = {}

    def set_memory_budget(self, budget_bytes: float | None) -> None:
        """The per-rank budget the streamed-vs-assembled route is priced against."""
        self._per_rank_budget_bytes = budget_bytes

    def _torch_device(self) -> torch.device:
        """The placed device as ``torch.device`` (``NeedDevice`` may hold a bare CUDA ordinal)."""
        return torch.device("cuda", self.device) if isinstance(self.device, int) else self.device

    def _submit_write(self, operation: Callable[[], None]) -> None:
        """Run ``operation`` on the background writer, or inline when the destination must stay serial.

        Charged to ``wait(write)`` either way, and a submission blocks once the queue is full."""
        if self._async_writes is None:
            self._async_writes = self._torch_device().type == "cuda"
        with PREDICTION_CLOCK.phase("wait(write)"):
            if not self._async_writes:
                operation()
                return
            if self._writer is None:
                self._writer = _AsyncWriter()
            self._writer.submit(operation)

    def finalize_writes(self) -> None:
        """Drain and stop the background writer; every submitted write is on disk when this returns."""
        if self._writer is not None:
            writer, self._writer = self._writer, None
            writer.close()

    # Pinned staging pays off only above this size.
    _PINNED_OFFLOAD_MIN_BYTES = 64 * 1024 * 1024

    def _offload_to_cpu(self, layer: torch.Tensor) -> torch.Tensor:
        """Return a CPU patch whose storage is independent of the reusable offload buffer. Large CUDA
        patches stage through pinned memory, then copy into owned pageable storage."""
        with self._borrow_cpu_patch(layer) as patch:
            if patch is not self._pin_buffer:
                return patch
            out = torch.empty(patch.shape, dtype=patch.dtype)
            out.copy_(patch)
            return out

    @contextmanager
    def _borrow_cpu_patch(self, layer: torch.Tensor) -> Iterator[torch.Tensor]:
        """Lend one CPU patch until this context exits, for synchronous accumulation only.

        The caller must finish consuming the patch within the context: the next offload may overwrite
        its storage."""
        detached = layer.detach()
        if (
            detached.device.type != "cuda"
            or detached.numel() * detached.element_size() < self._PINNED_OFFLOAD_MIN_BYTES
        ):
            yield detached.cpu()
            return
        buffer = self._pin_buffer
        if buffer is None or buffer.shape != detached.shape or buffer.dtype != detached.dtype:
            try:
                buffer = torch.empty(detached.shape, dtype=detached.dtype, pin_memory=True)
            except RuntimeError:  # host cannot lock this much memory -> plain pageable copy
                self._pin_buffer = None
                yield detached.cpu()
                return
            self._pin_buffer = buffer
        # Blocking copy: the host data must be complete before the buffer is lent.
        buffer.copy_(detached)
        yield buffer

    def prepare(self, name_layer: str) -> None:
        konfai_args = f"{konfai_root()}.outputs_dataset.{name_layer}.OutputDataset"

        def build(name: str, loaders: dict[str, TransformLoader] | None) -> list[Transform]:
            return [
                loader.get_transform(classpath, konfai_args=f"{konfai_args}.{name}")
                for classpath, loader in (loaders or {}).items()
            ]

        self.before_reduction_transforms = build("before_reduction_transforms", self._before_reduction_transforms)
        self.after_reduction_transforms = build("after_reduction_transforms", self._after_reduction_transforms)
        self.final_transforms = build("final_transforms", self._final_transforms)

        # The overlap needs an owner whether or not a combine is declared: Trim keeps each patch's
        # central band and never averages.
        module, name = get_module(self._patch_combine or "Trim", "konfai.data.patching")
        self.patch_combine = apply_config(konfai_args)(getattr(module, name))()

        module, name = get_module(self.reduction_classpath, "konfai.predictor")
        # The classpath is one key, dots and all: escaped so the dotted path is not split through it.
        subtree = f"{konfai_args}.{_escape_key_component(self.reduction_classpath)}"
        self.reduction = apply_config(subtree)(getattr(module, name))()

    def set_datasets(self, datasets: list[Dataset]) -> None:
        for transform in self.before_reduction_transforms:
            transform.set_datasets([*datasets, self])
        for transform in self.after_reduction_transforms:
            transform.set_datasets([*datasets, self])
        for transform in self.final_transforms:
            transform.set_datasets([*datasets, self])

    def set_patch_config(
        self,
        patch_size: list[int] | None,
        overlap: int | float | str | list[int | float | str] | None,
        nb_data_augmentation: int,
    ) -> None:
        # A single patch covering the volume takes no combine. Anything else keeps EVERY axis, the
        # untiled ones as a single broadcast entry.
        if patch_size and any(size > 1 for size in patch_size) and overlap is not None:
            if self.patch_combine is not None:
                axes = blend_axes(patch_size)
                self.patch_combine.set_patch_config(axes, blend_overlap(overlap, axes))
        else:
            self.patch_combine = None
        self.nb_data_augmentation = nb_data_augmentation

    def to(self, device: int):
        super().to(device)
        for transform in [*self.before_reduction_transforms, *self.after_reduction_transforms, *self.final_transforms]:
            transform.to(device)

    def is_done(self, index: int) -> bool:
        # ``.get``: a streamed case cleans itself up inside ``add_layer``, so its index may already be gone.
        accumulators = self.output_layer_accumulator.get(index)
        if accumulators is None or len(accumulators) != self.nb_data_augmentation:
            return False
        return all(acc.is_full() for acc in accumulators.values())

    def _submit_final_write(self, name: str, tensor: torch.Tensor, attribute: Attribute) -> None:
        """Queue one whole-volume entry write (D2H copy included) on the write path."""
        write = super().write

        def operation() -> None:
            write(self.group, name, tensor.detach().cpu().numpy(), attribute)

        self._submit_write(operation)

    def write_prediction(self, index: int, name: str, layer: torch.Tensor) -> None:
        attribute = self.attributes[index][0][0]
        self.attributes.pop(index)
        self._submit_final_write(name, layer, attribute)

    def __str__(self) -> str:
        params = {
            "filename": self.filename,
            "group": self.group,
            "before_reduction_transforms": self.before_reduction_transforms,
            "after_reduction_transforms": self.after_reduction_transforms,
            "final_transforms": self.final_transforms,
            "patch_combine": self.patch_combine,
            "reduction": self.reduction,
        }
        return str(params)

    def __repr__(self) -> str:
        return str(self)

    def add_layer(
        self,
        index_dataset: int,
        index_augmentation: int,
        index_patch: int,
        layer: torch.Tensor,
        dataset: DatasetIter,
        attribute: Attribute | None = None,
        number_of_channels_per_model: list[int] | None = None,
    ):
        with PREDICTION_CLOCK.phase("blend"):
            self._ensure_case_state(
                index_dataset, index_augmentation, layer, dataset, attribute, number_of_channels_per_model
            )
            attributes = self.attributes[index_dataset][index_augmentation]
            for transform in self._patch_inverses(dataset):
                layer = transform.inverse(self.names[index_dataset], layer, attributes[index_patch])
            accumulator = self.output_layer_accumulator[index_dataset][index_augmentation]
            slabs = self._blend_patch(index_dataset, index_patch, layer, accumulator)
        if not self._stream_plans.get(index_dataset):
            return
        with PREDICTION_CLOCK.phase("finalize(stream)"):
            self._advance_stream(
                index_dataset, index_augmentation, accumulator, slabs, number_of_channels_per_model, dataset
            )

    def _ensure_case_state(
        self,
        index_dataset: int,
        index_augmentation: int,
        layer: torch.Tensor,
        dataset: DatasetIter,
        attribute: Attribute | None,
        number_of_channels_per_model: list[int] | None,
    ) -> None:
        """First patch of a (case, augmentation): build the inherited-then-declared header, decide
        the stream plan once per case, and open this augmentation's accumulator."""
        if (
            index_dataset in self.output_layer_accumulator
            and index_augmentation in self.output_layer_accumulator[index_dataset]
        ):
            return
        input_dataset = dataset.get_dataset_from_index(self.group_dest, index_dataset)
        source_attribute = (
            Attribute(attribute) if attribute is not None else Attribute(input_dataset.cache_attributes[0])
        )
        # The declared attributes are applied over the inherited ones; an empty value drops a key.
        for key, value in self._attributes.items():
            if value == "":
                source_attribute.pop(key, None)
            else:
                source_attribute[key] = value
        if index_dataset not in self.output_layer_accumulator:
            self.output_layer_accumulator[index_dataset] = {}
            self.attributes[index_dataset] = {}
            self.names[index_dataset] = input_dataset.name
            # The streamed consumers index their slabs on the first spatial axis: a grid swept along
            # another axis takes the whole-volume path. max(1, ...): the count is set by load().
            sweeps_first_axis = all(
                input_dataset.patch.get_sweep_axis(a) == 0 for a in range(max(1, self.nb_data_augmentation))
            )
            plan = (
                self._plan_stream(dataset, index_dataset, source_attribute, layer, number_of_channels_per_model)
                if self._streaming_enabled and sweeps_first_axis
                else None
            )
            self._stream_plans[index_dataset] = plan
            if self._streaming_enabled and (plan is None or not plan.to_sink):
                path = "whole-volume" if plan is None else "buffered (the prefix streams, the tail runs whole-volume)"
                self._report_once(path, f"streaming: case '{input_dataset.name}' takes the {path} path.")
        # Everything past this point reads the header at index 0; a patch-level inverse reads one copy
        # per patch, taken before any inverse ran.
        attributes = self.attributes[index_dataset][index_augmentation] = {0: source_attribute}
        if self._patch_inverses(dataset):
            for i in range(1, len(input_dataset.patch.get_patch_slices(index_augmentation))):
                attributes[i] = Attribute(source_attribute)

        accumulator_type = StreamingAccumulator if self._stream_plans[index_dataset] else Accumulator
        self.output_layer_accumulator[index_dataset][index_augmentation] = accumulator_type(
            input_dataset.patch.get_patch_slices(index_augmentation),
            input_dataset.patch.patch_size,
            self.patch_combine,
            batch=False,
            sweep_axis=input_dataset.patch.get_sweep_axis(index_augmentation),
        )

    def _patch_inverses(self, dataset: DatasetIter) -> list[TransformInverse]:
        """The patch-level transforms of the mirrored group each patch is passed back through, last first."""
        return [
            transform
            for transform in reversed(dataset.groups_src[self.group_src][self.group_dest].patch_transforms)
            if isinstance(transform, TransformInverse) and transform.apply_inverse
        ]

    def _blend_patch(
        self, index_dataset: int, index_patch: int, layer: torch.Tensor, accumulator: Accumulator
    ) -> list[tuple[slice, torch.Tensor]]:
        """Blend one patch on the case's accumulate device and return the slabs it released."""
        if index_dataset not in self._accum_device:
            self._accum_device[index_dataset] = self._accumulate_device(layer, accumulator)
            if layer.device.type != "cpu" and self._accum_device[index_dataset].type == "cpu":
                self._report_once(
                    "host-accumulate",
                    f"case '{self.names[index_dataset]}' accumulates on the host.",
                )
        target = self._accum_device[index_dataset]
        # A GPU accumulator takes the patch straight in; a CPU one takes an offloaded copy.
        if target.type == "cpu":
            if layer.device.type != "cpu":
                with self._borrow_cpu_patch(layer) as cpu_patch:
                    return accumulator.add_layer(index_patch, cpu_patch) or []
        elif str(layer.device) != str(target):
            layer = layer.to(target)
        try:
            return accumulator.add_layer(index_patch, layer) or []
        except torch.cuda.OutOfMemoryError:
            # Nothing is blended yet, so the rest of the case blends on the CPU; a mid-blend OOM is fatal.
            if layer.device.type == "cpu" or not accumulator.is_empty():
                raise
            self._accum_device[index_dataset] = torch.device("cpu")
            torch.cuda.empty_cache()
            with self._borrow_cpu_patch(layer) as cpu_patch:
                return accumulator.add_layer(index_patch, cpu_patch) or []

    def _advance_stream(
        self,
        index_dataset: int,
        index_augmentation: int,
        accumulator: Accumulator,
        slabs: list[tuple[slice, torch.Tensor]],
        number_of_channels_per_model: list[int] | None,
        dataset: DatasetIter,
    ) -> None:
        """Push the released slabs through the aligner and the finalize chain; on any error, abort
        the sink and every slab stage before re-raising."""
        copy_finished = accumulator.is_full()
        if copy_finished:
            slabs = slabs + cast(StreamingAccumulator, accumulator).finalize()
        try:
            aligner = self._aligners.setdefault(index_dataset, SlabAligner(self.nb_data_augmentation))
            joint = aligner.push(index_augmentation, slabs, copy_finished)
            self._consume_slabs(index_dataset, joint, number_of_channels_per_model, dataset)
            finished = aligner.complete
            if finished:
                self._finish_stream(index_dataset)
        except BaseException as error:

            def abort(error: BaseException = error, index: int = index_dataset) -> None:
                sink = self._stream_sinks.pop(index, None)
                if sink is not None:
                    sink.abort(error)

            with suppress(Exception):
                self._submit_write(abort)
            plan = self._stream_plans.get(index_dataset)
            if plan is not None:
                for position in plan.slab_stages:
                    with suppress(Exception):
                        plan.stages[position].transform.stream_abort(self.names[index_dataset])
            raise
        if finished:
            self._close_stream(index_dataset)

    @staticmethod
    def _voxel_local(locality: PatchLocality, attribute: Attribute) -> bool:
        """POINTWISE, or GLOBAL_STAT whose statistic the case attribute already carries."""
        if locality.kind is LocalityKind.POINTWISE:
            return True
        return locality.kind is LocalityKind.GLOBAL_STAT and all(key in attribute for key in locality.stat_keys)

    def _report_once(self, key: str, message: str) -> None:
        """One line the first time a distinct non-window-bounded outcome appears; repeats are silent."""
        if key not in self._reported_paths:
            self._reported_paths.add(key)
            print(f"[KonfAI] {message}")

    def _worth_streaming(self, dataset: DatasetIter, index: int, layer: torch.Tensor) -> bool:
        """Whether this case's accumulators are heavy enough for the per-slab machinery to pay. The
        estimate is taken before the patch-level inverses, so a dtype-widening inverse under-counts by
        at most 2x."""
        spatial = dataset.get_dataset_from_index(self.group_dest, index).shapes[0]
        assembled = int(layer.shape[0]) * int(np.prod(spatial)) * layer.element_size() * self.nb_data_augmentation
        raw = os.environ.get("KONFAI_STREAM_WORTH_THRESHOLD")
        try:
            fraction = float(raw) if raw is not None else _STREAM_WORTH_MIN_FRACTION
        except ValueError:
            warnings.warn(
                f"KONFAI_STREAM_WORTH_THRESHOLD={raw!r} is not a number; using {_STREAM_WORTH_MIN_FRACTION}.",
                stacklevel=2,
            )
            fraction = _STREAM_WORTH_MIN_FRACTION
        # The config's budget, never the machine's free memory.
        budget = self._per_rank_budget_bytes
        if budget is None:
            budget = resolve_memory_budget(None).per_rank_bytes(node_local_ranks())
        return assembled >= fraction * budget

    def _plan_stream(
        self,
        dataset: DatasetIter,
        index: int,
        attribute: Attribute,
        layer: torch.Tensor | None = None,
        number_of_channels_per_model: list[int] | None = None,
    ) -> _StreamPlan | None:
        """The streaming plan for this case, or ``None`` for the whole-volume path.

        The streamed part of the finalize chain is ``[pointwise*][region and pointwise stages]``. What
        streaming cannot honour becomes a whole-volume TAIL run once on a post-reduction buffer, as does
        a destination that cannot serve region writes. Refused outright: a reduction that is not
        voxel-local, a non-voxel-local before-reduction transform, a TTA copy whose un-augment does not
        act slab by slab (``_tta_streamable``), or a case too light to pay (``_worth_streaming``).
        """
        if self.nb_data_augmentation < 1:
            return None
        if not self.reduction.voxel_local:
            return None
        if layer is not None and not self._worth_streaming(dataset, index, layer):
            return None
        if self.nb_data_augmentation != 1 and not self._tta_streamable(dataset, index, attribute):
            return None
        for transform in self.before_reduction_transforms:
            locality = transform.patch_locality(Attribute(attribute))
            # A SLAB before-reduction transform streams through ``stream_slab``; any other
            # non-voxel-local one refuses outright.
            if not self._voxel_local(locality, attribute) and locality.kind is not LocalityKind.SLAB:
                return None
        stages = [
            *(_FinalizeStage(transform, False) for transform in self.after_reduction_transforms),
            *(
                _FinalizeStage(transform, True)
                for transform in reversed(dataset.groups_src[self.group_src][self.group_dest].transforms)
                if isinstance(transform, TransformInverse) and transform.apply_inverse
            ),
            *(_FinalizeStage(transform, False) for transform in self.final_transforms),
        ]
        pipe_start: int | None = None
        tail_start = len(stages)
        slab_stages = set()
        for position, stage in enumerate(stages):
            locality = stage.locality(Attribute(attribute))
            if self._voxel_local(locality, attribute):
                continue
            if locality.kind is LocalityKind.SLAB and not stage.inverted and pipe_start is None:
                # A SLAB stage streams through ``stream_slab`` only on the accumulator grid.
                slab_stages.add(position)
                continue
            if locality.kind.is_region:
                if pipe_start is None:
                    pipe_start = position
                continue
            tail_start = position
            break
        if tail_start < len(stages) or not self.can_stream_data(attribute):
            # A whole-volume tail swallows the region stages too: the buffer sits on the accumulator grid.
            if pipe_start is not None:
                tail_start = min(tail_start, pipe_start)
            return _StreamPlan(stages, None, tail_start, to_sink=False, slab_stages=frozenset(slab_stages))
        return _StreamPlan(stages, pipe_start, len(stages), to_sink=True, slab_stages=frozenset(slab_stages))

    @staticmethod
    def _copy_draw(dataset: DatasetIter, index_augmentation: int) -> tuple[list[DataAugmentation], int] | None:
        """The augmentations of copy ``index_augmentation`` and its index in their list, ``None`` for copy 0."""
        if index_augmentation == 0:
            return None
        i = index_augmentation - 1
        for data_augmentations in dataset.data_augmentations_list:
            if i < data_augmentations.nb:
                return data_augmentations.data_augmentations, i
            i -= data_augmentations.nb
        return None

    def _unaugment(
        self, dataset: DatasetIter, index: int, index_augmentation: int, tensor: torch.Tensor
    ) -> torch.Tensor:
        """Undo copy ``index_augmentation``'s draw on ``tensor``: the augmentations applied in reverse,
        bound by the case index the draw was made under."""
        draw = self._copy_draw(dataset, index_augmentation)
        if draw is None:
            return tensor
        augmentations, a = draw
        case = dataset.get_dataset_from_index(self.group_dest, index).index
        for data_augmentation in reversed(augmentations):
            tensor = data_augmentation.inverse(case, a, tensor)
        return tensor

    def _tta_streamable(self, dataset: DatasetIter, index: int, attribute: Attribute) -> bool:
        """Whether every copy's un-augment acts slab by slab, read from the declarations alone.

        A POINTWISE draw does; an ORIENTATION draw does when its declared region remap fixes the slab
        axis row for row and its shape fold keeps the slab extent. Any other kind refuses outright.
        """
        try:
            input_dataset = dataset.get_dataset_from_index(self.group_dest, index)
            case = input_dataset.index
            for index_augmentation in range(1, self.nb_data_augmentation):
                draw = self._copy_draw(dataset, index_augmentation)
                if draw is None:
                    continue
                augmentations, a = draw
                shape = [int(extent) for extent in input_dataset.shapes[0]]
                for augmentation in augmentations:
                    locality = augmentation.patch_locality(case, a, Attribute(attribute))
                    if locality.kind is LocalityKind.POINTWISE:
                        continue
                    if locality.kind is not LocalityKind.ORIENTATION:
                        return False
                    out_shape = [int(extent) for extent in augmentation.stream_shape(case, a, list(shape))]
                    if out_shape[0] != shape[0]:
                        return False
                    plane = tuple(slice(0, extent) for extent in out_shape[1:])
                    for row in range(out_shape[0]):
                        source = augmentation.stream_region_source(case, a, (slice(row, row + 1), *plane), shape)
                        if (source[0].start, source[0].stop) != (row, row + 1):
                            return False
                    shape = out_shape
        except Exception:  # nosec B110 - an unprobeable draw keeps the case on the whole-volume path
            return False
        return True

    def _consume_slabs(
        self,
        index: int,
        slabs: list[tuple[slice, dict[int, torch.Tensor]]],
        number_of_channels_per_model: list[int] | None,
        dataset: DatasetIter,
    ) -> None:
        """Run each jointly finalized slab through the plan: prefix per slab, then sink, region
        stream, or buffer. The first slab fixes the case's state (see ``_init_stream_state``)."""
        plan = cast(_StreamPlan, self._stream_plans[index])
        for region, copies in slabs:
            block, attribute = self._finalize_slab(index, copies, number_of_channels_per_model, plan, dataset, region)
            if index not in self._post_prefix_attributes:
                plan = self._init_stream_state(index, plan, block, attribute)
            state = self._region_states.get(index)
            if state is not None:
                for target, emitted in cast(SlabRegionStream, state.stream).push(region, block):
                    self._write_stream_block(index, target, emitted, cast(Attribute, state.attribute))
            elif plan.to_sink:
                spatial = self.output_layer_accumulator[index][0].shape
                target = (region, *(slice(0, int(extent)) for extent in spatial[1:]))
                self._write_stream_block(index, target, block, attribute)
            else:
                buffer = self._stream_buffers[index]
                lead = (slice(None),) * (block.dim() - len(self.output_layer_accumulator[index][0].shape))
                buffer[(*lead, region)] = block.to(buffer.device)

    def _init_stream_state(
        self, index: int, plan: _StreamPlan, block: torch.Tensor, attribute: Attribute
    ) -> _StreamPlan:
        """Fix the case's streaming state at its first slab, when the prefix output is known."""
        self._post_prefix_attributes[index] = Attribute(attribute)
        spatial = [int(extent) for extent in self.output_layer_accumulator[index][0].shape]
        if plan.pipe_start is not None:
            state = self._make_pipe_state(index, plan, spatial, block)
            if state is None:
                plan = _StreamPlan(plan.stages, None, plan.pipe_start, to_sink=False, slab_stages=plan.slab_stages)
                self._stream_plans[index] = plan
            else:
                self._region_states[index] = state
        if not plan.to_sink:
            lead = list(block.shape[: block.dim() - len(spatial)])
            try:
                buffer = torch.empty([*lead, *spatial], dtype=block.dtype, device=block.device)
            except torch.cuda.OutOfMemoryError:
                buffer = torch.empty([*lead, *spatial], dtype=block.dtype, device="cpu")
            self._stream_buffers[index] = buffer
        return plan

    def _make_pipe_state(
        self, index: int, plan: _StreamPlan, in_shape: list[int], block: torch.Tensor
    ) -> _RegionState | None:
        """Wire the case's streamed pipe into one :class:`SlabRegionStream`, or answer ``None`` for
        the buffered tail where streaming would not be exact.

        The pull map folds each stage's declaration backward from the written region down to the
        accumulator; ``produce`` walks the pipe forward over the pulled window. The fold is planned
        by walking a one-voxel corner of the first slab through the pipe with one evolving attribute,
        and ``produce`` replays the same transitions on a fresh copy per emission.
        """
        attr0 = self._post_prefix_attributes[index]
        pipe = plan.stages[cast(int, plan.pipe_start) :]
        name = self.names[index]

        corner: tuple[EllipsisType | slice, ...] = (Ellipsis, *([slice(0, 1)] * len(in_shape)))
        probe = block[corner].clone()
        walking = Attribute(attr0)
        shapes = [list(in_shape)]
        kinds: list[LocalityKind] = []
        pull_fns: list[Callable[[tuple[slice, ...]], list[slice]]] = []
        try:
            for stage in pipe:
                shape = shapes[-1]
                locality = stage.locality(Attribute(walking))
                kinds.append(locality.kind)
                snapshot = Attribute(walking)
                if locality.kind is LocalityKind.HALO:
                    pull_fns.append(_HaloPull(_halo_radii(locality.halo, len(shape)), shape))
                    shapes.append(list(shape))
                    probe = stage(name, probe, walking)
                elif locality.kind is LocalityKind.REGRID:
                    # A regrid states its transition instead of performing it: the one-voxel probe
                    # cannot run through its inverse.
                    transform = stage.transform
                    if stage.inverted:
                        remapper = cast(TransformInverse, transform)
                        pull_fns.append(_RemapPull(remapper.stream_region_target, shape, snapshot, name))
                        out = remapper.inverse_transform_shape(list(shape), Attribute(walking))
                        remapper.inverse_stream_cache_attribute(walking, shape)
                    else:
                        pull_fns.append(_RemapPull(transform.stream_region_source, shape, snapshot, name))
                        out = transform.transform_shape(self.group_src, name, list(shape), Attribute(walking))
                        transform.write_stream_cache_attribute(walking, shape, name)
                    shapes.append([int(extent) for extent in out])
                elif locality.kind.is_region:
                    if stage.inverted:
                        remapper = cast(TransformInverse, stage.transform)
                        pull_fns.append(_RemapPull(remapper.stream_region_target, shape, snapshot, name))
                        out = remapper.inverse_transform_shape(list(shape), Attribute(walking))
                    else:
                        pull_fns.append(_RemapPull(stage.transform.stream_region_source, shape, snapshot, name))
                        out = stage.transform.transform_shape(self.group_src, name, list(shape), Attribute(walking))
                    shapes.append([int(extent) for extent in out])
                    # The stage's attribute transition on the probe: a crop's tensor answer is dropped, its pops kept.
                    result = stage(name, probe, walking)
                    if locality.kind is not LocalityKind.CROP:
                        probe = result
                else:
                    pull_fns.append(lambda target: list(target))
                    shapes.append(list(shape))
                    probe = stage(name, probe, walking)
        except Exception:  # nosec B110 - an unplannable pipe just keeps the case on the buffered path
            return None

        state = _RegionState(shapes)

        def spans_for(target: tuple[slice, ...]) -> list[list[slice]]:
            """The region of each inter-stage space behind ``target``, folded back to the accumulator."""
            spans: list[list[slice]] = [list(target)]
            for pull_stage in reversed(pull_fns):
                spans.append(pull_stage(tuple(spans[-1])))
            spans.reverse()
            return spans

        def pull(target: tuple[slice, ...]) -> list[slice]:
            return spans_for(target)[0]

        def produce(window: torch.Tensor, target: tuple[slice, ...], source: list[slice]) -> torch.Tensor:
            attribute = Attribute(attr0)
            spans = spans_for(target)
            block = window
            for i, (stage, kind) in enumerate(zip(pipe, kinds, strict=True)):
                block = self._apply_pipe_stage(
                    stage, kind, block, tuple(spans[i + 1]), spans[i], shapes[i], shapes[i + 1], attribute, name
                )
            state.attribute = attribute
            return block

        state.stream = SlabRegionStream(pull, produce, in_shape, shapes[-1])
        return state

    def _apply_pipe_stage(
        self,
        stage: _FinalizeStage,
        kind: LocalityKind,
        block: torch.Tensor,
        target: tuple[slice, ...],
        source: list[slice],
        in_shape: list[int],
        out_shape: list[int],
        attribute: Attribute,
        name: str,
    ) -> torch.Tensor:
        """Run one pipe stage on its pulled block, by declared kind, never by stage name."""
        if kind is LocalityKind.CROP:
            # The pull already translated the region, so the block IS the answer; the stage still
            # runs for its attribute transition.
            stage(name, block, attribute)
            return block
        if kind is LocalityKind.REGRID:
            # Region-aware on both sides; the geometry is written from the FULL shape.
            context = RegionContext(tuple(source), tuple(target), tuple(in_shape))
            if stage.inverted:
                remapper = cast(TransformInverse, stage.transform)
                result = remapper.stream_region_inverse(name, block, context, Attribute(attribute))
                remapper.inverse_stream_cache_attribute(attribute, in_shape)
            else:
                result = stage.transform.stream_region(name, block, context, Attribute(attribute))
                stage.transform.write_stream_cache_attribute(attribute, in_shape, name)
            return result
        if kind is LocalityKind.ORIENTATION and not stage.inverted:
            # Run the tensor action on a throwaway scope so it does not record the SLAB's extent, then
            # write the case geometry from the full ``in_shape``.
            result = stage(name, block, Attribute(attribute))
            cast(TransformInverse, stage.transform).write_stream_cache_attribute(attribute, in_shape, name)
            return result
        result = stage(name, block, attribute)
        if kind is LocalityKind.HALO:
            lead = (slice(None),) * (result.dim() - len(target))
            crop = tuple(slice(t.start - s.start, t.stop - s.start) for t, s in zip(target, source, strict=False))
            result = result[(*lead, *crop)]
        return result

    def _write_stream_block(
        self, index: int, target: tuple[slice, ...], block: torch.Tensor, attribute: Attribute
    ) -> None:
        """Write one finalized output block into the case's sink (opened at the first block).

        The whole write is one submitted operation, so ``_stream_sinks`` is only touched in
        submission order; the attribute is snapshotted: the region state's evolves."""
        state = self._region_states.get(index)
        spatial = (
            state.shapes[-1]
            if state is not None
            else [int(extent) for extent in self.output_layer_accumulator[index][0].shape]
        )
        name = self.names[index]
        attribute = Attribute(attribute)

        def operation() -> None:
            array = block.detach().cpu().numpy()
            sink = self._stream_sinks.get(index)
            if sink is None:
                sink = self.open_data_stream(self.group, name, [array.shape[0], *spatial], array.dtype, attribute)
                if sink is None:
                    raise PredictorError(
                        f"Streamed write refused by the '{self.file_format}' backend for dtype"
                        f" '{array.dtype}' on output '{self.group}': write it to an h5 or omezarr"
                        f" dataset, or set KONFAI_STREAMED_WRITES=0 to force the whole-volume path."
                    )
                self._stream_sinks[index] = sink
            sink.write_slice((slice(0, array.shape[0]), *target), array)

        self._submit_write(operation)

    def _finish_stream(self, index: int) -> None:
        """Complete the case: flush the region scheduler, or run the whole-volume tail on the buffer
        and write it classically."""
        plan = cast(_StreamPlan, self._stream_plans[index])
        state = self._region_states.get(index)
        if state is not None:
            for target, emitted in cast(SlabRegionStream, state.stream).finalize():
                self._write_stream_block(index, target, emitted, cast(Attribute, state.attribute))
            return
        if plan.to_sink:
            return
        result = self._stream_buffers.pop(index)
        attribute = Attribute(self._post_prefix_attributes[index])
        name = self.names[index]
        for stage in plan.stages[plan.tail_start :]:
            result = stage(name, result, attribute)
        self._submit_final_write(name, result, attribute)

    def _prepare_copy_slab(
        self,
        index: int,
        index_augmentation: int,
        layer: torch.Tensor,
        number_of_channels_per_model: list[int] | None,
        dataset: DatasetIter,
        region: slice,
        spatial: list[int],
    ) -> torch.Tensor:
        """One copy's slab through the per-copy head of ``_get_output``: un-augment it, split the model
        chunks, run before_reduction on each, and stack to the copy's ``[1, M, C, ...]`` block. A
        per-voxel before-reduction transform is told where the slab sits (``stream_region``); a SLAB
        one goes through ``stream_slab``."""
        layer = self._unaugment(dataset, index, index_augmentation, layer)
        attribute = Attribute(self.attributes[index][index_augmentation][0])
        chunks = self._split_model_chunks(layer, number_of_channels_per_model, attribute)
        context = _slab_context(region, spatial)
        results = []
        for chunk in chunks:
            for transform in self.before_reduction_transforms:
                if transform.patch_locality(Attribute(attribute)).kind is LocalityKind.SLAB:
                    chunk = transform.stream_slab(self.names[index], chunk, region, spatial, Attribute(attribute))
                else:
                    chunk = transform.stream_region(self.names[index], chunk, context, Attribute(attribute))
            results.append(chunk)
        # A lone chunk stacks as a view: torch.stack would copy the slab once per slab of the case.
        if len(results) == 1:
            return results[0].unsqueeze(0).unsqueeze(0)
        return torch.stack(results, dim=0).unsqueeze(0)

    def _finalize_slab(
        self,
        index: int,
        copies: dict[int, torch.Tensor],
        number_of_channels_per_model: list[int] | None,
        plan: _StreamPlan,
        dataset: DatasetIter,
        region: slice,
    ) -> tuple[torch.Tensor, Attribute]:
        """The finalize chain of ``_get_output``/``get_output``, on one z-slab of every copy, up to
        the plan's prefix boundary.

        Each step is the whole-volume computation restricted to the slab (same ops, same order, same
        reduction call). Each slab gets its own copy of the case attribute.
        """
        spatial = [int(extent) for extent in self.output_layer_accumulator[index][0].shape]
        blocks = [
            self._prepare_copy_slab(
                index, index_augmentation, layer, number_of_channels_per_model, dataset, region, spatial
            )
            for index_augmentation, layer in copies.items()
        ]
        result = self._reduce_copies(blocks)
        attribute = Attribute(self.attributes[index][0][0])
        self._split_model_chunks(next(iter(copies.values())), number_of_channels_per_model, attribute)
        context = _slab_context(region, spatial)
        for position, stage in enumerate(plan.stages[: plan.boundary]):
            if position in plan.slab_stages:
                result = stage.transform.stream_slab(self.names[index], result, region, spatial, attribute)
            else:
                # Told where the slab sits, so a stage reading a companion volume (Mask) reads its slab region.
                result = stage.stream_region(self.names[index], result, context, attribute)
        return result, attribute

    def reset(self) -> None:
        """Drop every in-flight accumulation (the OOM-restart path re-runs the rank's cases from scratch)."""
        # Abort each open sink so the backend removes the partial entry; the restart rewrites it.
        error = PredictorError("prediction restart: the partial streamed output is discarded")
        for sink in self._stream_sinks.values():
            try:
                sink.abort(error)
            except Exception:  # nosec B110 - one sink failing to abort must not leak the others
                pass
        self._stream_sinks.clear()
        self._stream_plans.clear()
        self._region_states.clear()
        self._stream_buffers.clear()
        self._post_prefix_attributes.clear()
        self._aligners.clear()
        self.output_layer_accumulator.clear()
        self.attributes.clear()
        self.names.clear()
        self._accum_device.clear()
        self._reduce_device.clear()
        self._pin_buffer = None

    def _close_stream(self, index: int) -> None:
        """Finalize the case's sink and drop its bookkeeping (``is_done`` then reports nothing left)."""

        def operation() -> None:
            sink = self._stream_sinks.pop(index, None)
            if sink is not None:
                sink.close()

        self._submit_write(operation)
        self._stream_plans.pop(index, None)
        self._region_states.pop(index, None)
        self._stream_buffers.pop(index, None)
        self._post_prefix_attributes.pop(index, None)
        self._aligners.pop(index, None)
        self.output_layer_accumulator.pop(index, None)
        self.attributes.pop(index, None)
        self._accum_device.pop(index, None)
        self._reduce_device.pop(index, None)

    def setup(self, datasets: list[Dataset], groups: dict[str, list[str]]):
        self.set_datasets(datasets)
        if self.group_src not in groups.keys():
            raise PredictorError(f"Source group '{self.group_src}' not found. Available groups: {list(groups.keys())}.")

        if self.group_dest not in groups[self.group_src]:
            raise PredictorError(
                f"Destination group '{self.group_dest}' not found. Available groups: {groups[self.group_src]}."
            )

    @staticmethod
    def _split_model_chunks(
        layer: torch.Tensor, number_of_channels_per_model: list[int] | None, attribute: Attribute
    ) -> list[torch.Tensor]:
        """Split an ensemble layer into per-model chunk views and tag the attribute with the layout; a
        layer whose channels do not match the ensemble layout stays whole."""
        if number_of_channels_per_model and layer.shape[0] == sum(number_of_channels_per_model):
            attribute["number_of_channels_per_model_0"] = torch.tensor(number_of_channels_per_model)
            return list(torch.split(layer, number_of_channels_per_model, dim=0))
        return [layer]

    def _reduce_copies(self, copies: list[torch.Tensor]) -> torch.Tensor:
        """The cross-copy reduction, identical for a slab and a whole volume.

        Mixed devices (a mid-case OOM fallback) reconcile on the host. Reduce, then drop the singleton
        stack axis; Mean/Median also drop the singleton model axis, Concat keeps ``[M, C, ...]``."""
        if len({copy.device for copy in copies}) > 1:
            copies = [copy.cpu() if copy.device.type != "cpu" else copy for copy in copies]
        result = self.reduction(copies).squeeze(0)
        if isinstance(self.reduction, Mean | Median):
            result = result.squeeze(0)
        return result

    def _get_output(
        self, index: int, index_augmentation: int, number_of_channels_per_model: list[int], dataset: DatasetIter
    ) -> torch.Tensor:
        layer = self.output_layer_accumulator[index][index_augmentation].assemble()  # if concat then [N*C] else [C]
        layer = self._unaugment(dataset, index, index_augmentation, layer)
        base_attr = self.attributes[index][index_augmentation][0]
        chunks = self._split_model_chunks(layer, number_of_channels_per_model, base_attr)

        # The per-model channel reduction materialises a working volume on top of the resident
        # accumulator. Decided once per case whether it fits free VRAM; per augmentation would let the
        # device flip mid-case and hand the reduction a mixed-device list.
        if index not in self._reduce_device:
            self._reduce_device[index] = (
                self._reduction_device(chunks[0], len(chunks)) if chunks else torch.device("cpu")
            )
        reduce_device = self._reduce_device[index]
        results = []
        for i, layer in enumerate(chunks):
            attr = base_attr if (i == len(chunks) - 1) else Attribute(base_attr)
            layer = layer.to(reduce_device)
            for transform in self.before_reduction_transforms:
                layer = transform(self.names[index], layer, Attribute(attr))
            # The chunk stays on its current device; ``get_output`` runs the finalize where the volume is.
            results.append(layer)

        # Mean, Median -> [1, C, ...] | Concat -> [M, C, ...]. A lone chunk stacks as a view; no
        # reduction writes into what it is handed.
        if len(results) == 1:
            return results[0].unsqueeze(0)
        return torch.stack(results, dim=0)

    def _reduction_device(self, chunk: torch.Tensor, nb_chunks: int = 1) -> torch.device:
        """Device for the channel-reduction transforms: this dataset's CUDA device when every chunk (plus
        working headroom) fits free VRAM, else CPU (the memory-safe fallback)."""
        # NeedDevice stores a CUDA ordinal (int) on GPU and a torch.device on CPU; normalise to a device.
        device = torch.device("cuda", self.device) if isinstance(self.device, int) else self.device
        if device.type != "cuda":
            return torch.device("cpu")
        try:
            # Release the allocator's unused reserved cache so ``mem_get_info`` reports what is free.
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(device)
        except Exception:  # nosec B110 - any CUDA query failure just keeps the reduction on CPU
            return torch.device("cpu")
        # Every transformed chunk is parked on the reduce device until the final stack: budget all of
        # them plus a same-size working temp per chunk and one stack copy.
        needed = chunk.numel() * chunk.element_size() * (2 * max(1, nb_chunks) + 1)
        return device if needed < free else torch.device("cpu")

    # The memory queries run before a forward's allocations land: keep ~10 % of free VRAM in reserve
    # for fragmentation and a concurrent process.
    _ACCUMULATE_MARGIN = 0.9

    def _accumulate_device(self, layer: torch.Tensor, accumulator: Accumulator) -> torch.device:
        """Device on which to blend a case's patches: the GPU when the accumulator fits alongside the
        memory a forward needs, decided once per case at the first patch; else the CPU."""
        device = torch.device("cuda", self.device) if isinstance(self.device, int) else self.device
        if device.type != "cuda" or layer.device.type != "cuda":
            return torch.device("cpu")
        try:
            # Return the reserved-but-unused cache so ``mem_get_info`` reports the memory actually free.
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(device)
            # A forward's transient footprint above the resident set, from the batch that just ran;
            # ``max_memory_allocated`` is a high-water mark, so the gate errs toward the CPU.
            transient = torch.cuda.max_memory_allocated(device) - torch.cuda.memory_allocated(device)
        except Exception:  # nosec B110 - any CUDA query failure keeps the blend on CPU
            return torch.device("cpu")
        voxels = int(np.prod(accumulator.footprint_shape))
        # result [C, volume] + weight_sum [volume] at the patch dtype, for EVERY augmentation: all of
        # a case's accumulators are resident simultaneously.
        accumulator_bytes = (layer.shape[0] + 1) * voxels * layer.element_size() * max(1, self.nb_data_augmentation)
        # The channel reduction's working volume is budgeted separately by ``_reduction_device``.
        needed = accumulator_bytes + transient
        if isinstance(accumulator, StreamingAccumulator):
            # Transients on top of the resident window: the advance clone, the emission slab and its
            # weight clamp, and up to ``_AsyncWriter._CAPACITY`` emitted blocks awaiting their copy.
            # Two window footprints bound the sum.
            needed += 2 * layer.shape[0] * voxels * layer.element_size()
            if self.nb_data_augmentation > 1:
                # Slab-aligned TTA holds pending slabs per copy and reduces through a float32
                # accumulate: one window per copy plus one more for the reduction's transients.
                needed += (self.nb_data_augmentation + 2) * layer.shape[0] * voxels * layer.element_size()
        return device if needed < free * self._ACCUMULATE_MARGIN else torch.device("cpu")

    def get_output(self, index: int, number_of_channels_per_model: list[int], dataset: DatasetIter) -> torch.Tensor:
        results = [
            self._get_output(index, index_augmentation, number_of_channels_per_model, dataset).unsqueeze(0)
            for index_augmentation in self.output_layer_accumulator[index].keys()
        ]
        self.output_layer_accumulator.pop(index)
        self._accum_device.pop(index, None)
        self._reduce_device.pop(index, None)
        # The finalize runs where the volume was blended; only the final result returns to the host.
        result = self._reduce_copies(results)
        # combine = aggregation across models (M), reduce = aggregation across TTA copies (T):
        #   Mean/Median at both levels : [M, C, ...] -> [C, ...], then [T, C, ...] -> [C, ...]
        #   combine Concat, reduce Mean : [T, M, C, ...] -> [M, C, ...]
        #   reduce Concat               : [T, C, ...] stays [T, C, ...]
        #   Concat at both levels       : [M * T, C, ...]
        # With a Concat at either level, the first ``after_reduction_transforms`` entry must be
        # ``InferenceStack`` or ``Sum`` so a ``[C, ...]`` follows.
        for transform in self.after_reduction_transforms:
            result = transform(self.names[index], result, self.attributes[index][0][0])

        for transform in reversed(dataset.groups_src[self.group_src][self.group_dest].transforms):
            if isinstance(transform, TransformInverse) and transform.apply_inverse:
                result = transform.inverse(self.names[index], result, self.attributes[index][0][0])

        for transform in self.final_transforms:
            result = transform(self.names[index], result, self.attributes[index][0][0])

        return result.cpu() if result.device.type != "cpu" else result


# ``name_class: OutSameAsGroupDataset`` is what published Prediction.yml carry.
OutSameAsGroupDataset = OutputDataset


@config("OutputDataset")
class OutputDatasetLoader:
    """Builds one output's sink from ``Predictor.outputs_dataset.<layer>``: ``name_class`` is the
    sink's classpath, a bare name resolving in ``konfai.predictor``."""

    def __init__(self, name_class: str = "OutputDataset") -> None:
        self.name_class = name_class

    def get_output_dataset(self, layer_name: str) -> OutputDataset:
        module, name = get_module(self.name_class, "konfai.predictor")
        return apply_config(f"Predictor.outputs_dataset.{layer_name}")(getattr(module, name))()

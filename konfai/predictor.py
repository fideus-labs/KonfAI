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

"""Prediction workflow classes, reductions, and export helpers for KonfAI."""

import copy
import importlib
import os
import queue
import shutil
import threading
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]

from konfai import config_file, cuda_visible_devices, konfai_root, predictions_directory
from konfai.data.augmentation import DataAugmentation
from konfai.data.data_manager import BatchSample, DataPrediction, DatasetIter
from konfai.data.patching import (
    Accumulator,
    PathCombine,
    SlabAligner,
    SlabRegionStream,
    StreamingAccumulator,
    _halo_pull,
    _halo_radii,
    _remap_pull,
    _scale_pull,
    blend_overlap,
)
from konfai.data.transform import (
    LocalityKind,
    PatchLocality,
    Resample,
    Transform,
    TransformInverse,
    TransformLoader,
)
from konfai.network.network import Model, ModelLoader, NetState, Network
from konfai.utils.config import apply_config, config
from konfai.utils.dataset import Attribute, Dataset, DataStream
from konfai.utils.errors import ConfigError, PredictorError
from konfai.utils.runtime import (
    DataLog,
    DistributedObject,
    NeedDevice,
    State,
    available_memory_bytes,
    configure_workflow_environment,
    confirm_overwrite_or_raise,
    description,
    run_distributed_app,
    safe_torch_load,
)
from konfai.utils.utils import concretize_patch_size, env_flag, get_module, size_free_axes, split_path_spec
from konfai.utils.vram import next_patch_candidate, usable_vram


class Reduction(ABC):
    """Aggregate a list of predictions (one per model in an ensemble, or per TTA augmentation) into one.

    A ``Reduction`` is a KonfAI extension point: subclass it and reference it by classpath in the
    ``OutputDataset`` config. ``__call__`` receives the stacked predictions and returns the aggregate.
    """

    #: Streamed-write contract. ``True`` declares this reduction a **pure per-voxel** operation over the
    #: model/TTA (stack) axis — every output voxel depends only on the SAME voxel of each input, never on
    #: a spatial neighbour. The streamed-write gate reads this flag to decide whether the finalize chain
    #: may run slab by slab.
    #:
    #: Rules for a custom reduction:
    #: - **Default False.** Leave it False unless you are sure: an unknown reduction then takes the
    #:   whole-volume path, costing the streaming optimisation but never correctness.
    #: - **Set True only if voxel-local.** Reducing/stacking along the stack axis (dim 0) or the channel
    #:   axis (dim 1) is fine — those are orthogonal to the spatial slab axis. Anything that reads across
    #:   spatial positions (a spatial blur, a resample, a global-argmax over Z) must stay False.
    #: - **A wrong True corrupts the streamed output** (each slab would be reduced with only its own data);
    #:   the gate trusts this flag and checks nothing else.
    voxel_local: bool = False

    @abstractmethod
    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()


class Mean(Reduction):
    """Average ensemble or augmentation predictions element-wise."""

    voxel_local = True

    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        # A single element (no TTA / a lone model) is its own mean; skip the float32 clone + accumulate,
        # which for a whole-volume multi-class output is a large no-op allocation (fp16->fp32 round-trips
        # to the same values). Returns the same values as the general path.
        if len(tensors) == 1:
            return tensors[0]
        acc = tensors[0].float().clone()
        for t in tensors[1:]:
            acc.add_(t.float())
        acc.div_(len(tensors))
        return acc.to(dtype=tensors[0].dtype)


class Median(Reduction):
    """Compute the element-wise median across prediction tensors."""

    voxel_local = True

    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        # A single element is its own median; skip the float32 stack (a large no-op for whole volumes).
        if len(tensors) == 1:
            return tensors[0]
        return torch.median(torch.stack(tensors, dim=0).float(), dim=0).values.to(tensors[0].dtype)


class Concat(Reduction):
    """Concatenate prediction tensors along the channel dimension."""

    # Cats along the channel axis, orthogonal to the spatial slab axis -- per-voxel, so slab-local.
    voxel_local = True

    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tensors, dim=1)


class _AsyncWriter:
    """A background thread owning one output dataset's disk writes, in submission order.

    The prediction loop otherwise waits on every device-to-host copy and destination write between
    two forwards; submitting them here overlaps that tail with the next batch. The queue is bounded,
    so a slow destination back-pressures the loop instead of buffering the run; the first failure is
    kept, later operations drain unexecuted, and the failure re-raises at the next submission and at
    ``close`` — a run never ends with a write silently missing.
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


class OutputDataset(Dataset, NeedDevice, ABC):
    """
    Abstract prediction sink that accumulates model outputs and writes them to disk.

    Concrete subclasses define how layers are accumulated across patches,
    augmentations, and multiple models before the final prediction volume is
    materialized.
    """

    def __init__(
        self,
        filename: str,
        group: str,
        before_reduction_transforms: dict[str, TransformLoader],
        after_reduction_transforms: dict[str, TransformLoader],
        final_transforms: dict[str, TransformLoader],
        patch_combine: str | None,
        reduction: str,
    ) -> None:
        filename, _, file_format = split_path_spec(filename)
        super().__init__(filename, file_format)
        # ``Dataset.__init__`` does not forward ``super().__init__()``, so the ``NeedDevice`` mixin is
        # never initialised through the MRO; call it explicitly so ``self.device`` always has its CPU
        # default. Otherwise an output writer that is never moved (e.g. a CPU-only PREDICTION run, whose
        # device propagation is CUDA-gated) reads ``self.device`` and raises ``AttributeError``.
        NeedDevice.__init__(self)
        self.group = group
        self._before_reduction_transforms = before_reduction_transforms
        self._after_reduction_transforms = after_reduction_transforms
        self._final_transforms = final_transforms
        self._patch_combine = patch_combine
        self.reduction_classpath = reduction
        self.reduction: Reduction

        self.before_reduction_transforms: list[Transform] = []
        self.after_reduction_transforms: list[Transform] = []
        self.final_transforms: list[Transform] = []
        self.patch_combine: PathCombine | None = None

        self.output_layer_accumulator: dict[int, dict[int, Accumulator]] = {}
        self.attributes: dict[int, dict[int, dict[int, Attribute]]] = {}
        self.names: dict[int, str] = {}
        self.nb_data_augmentation = 0
        # Reusable page-locked staging buffer for the per-patch GPU->CPU offload. Prediction
        # accumulators keep every patch of a case until assembly, so patches cannot share one CPU
        # tensor; a single pinned buffer (one patch) stages each device patch instead, which is
        # copied into a fresh pageable tensor for storage. See ``_offload_to_cpu``.
        self._pin_buffer: torch.Tensor | None = None
        # Per-CASE blend device, decided once at the case's first patch (see ``_accumulate_device``):
        # CUDA when the full combined volume of EVERY augmentation fits VRAM (blend on GPU, no per-patch
        # offload, assembled volume stays on-device for the reduction), else CPU. The decision is per case,
        # not per (case, augmentation): all of a case's augmentations are reduced together in
        # ``get_output``, and a mid-case flip would hand the reduction a mixed CPU/CUDA tensor list.
        self._accum_device: dict[int, torch.device] = {}
        # Same single-decision rule for the CPU-blend reduction device (see ``_reduction_device``).
        self._reduce_device: dict[int, torch.device] = {}
        # Disk writes go to a background writer when the destination serves disjoint files per entry
        # (see ``Dataset.concurrent_write_safe``) AND the output runs on a GPU: the device-to-host
        # copy and the write then overlap the next forward, byte-identically — same operations, same
        # order. A single-store destination stays inline, so nothing ever writes one store from two
        # threads; a CPU-only loop stays inline too — its blend shares the memory bandwidth the
        # writer would consume, so there is nothing to overlap and something to lose.
        # ``KONFAI_ASYNC_WRITES`` is tri-state: unset = automatic, ``0`` kills, ``1`` forces (tests).
        raw = os.environ.get("KONFAI_ASYNC_WRITES", "").lower()
        self._async_writes: bool | None
        if raw in ("0", "false") or not self.concurrent_write_safe():
            self._async_writes = False
        elif raw in ("1", "true"):
            self._async_writes = True
        else:
            self._async_writes = None  # decided at the first write, once the device is placed
        self._writer: _AsyncWriter | None = None

    def _torch_device(self) -> torch.device:
        """The placed device as ``torch.device`` (``NeedDevice`` may hold a bare CUDA ordinal)."""
        return torch.device("cuda", self.device) if isinstance(self.device, int) else self.device

    def _submit_write(self, operation: Callable[[], None]) -> None:
        """Run ``operation`` on the background writer, or inline when the destination must stay serial."""
        if self._async_writes is None:
            self._async_writes = self._torch_device().type == "cuda"
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

    # A pageable D2H copy on a large multi-class patch is a slow, fully synchronous PCIe transfer;
    # staging through page-locked memory only pays off once the patch is large enough that the copy,
    # not the buffer bookkeeping, dominates. Small patches (e.g. single-channel synthesis) take the
    # plain path unchanged.
    _PINNED_OFFLOAD_MIN_BYTES = 64 * 1024 * 1024

    def _offload_to_cpu(self, layer: torch.Tensor) -> torch.Tensor:
        """Move a device patch to CPU, staging through a reusable pinned buffer for a faster copy.

        Prediction accumulators hold every patch of a case until assembly, so patches cannot reuse a
        single CPU tensor. A pageable ``layer.detach().cpu()`` on a large multi-class patch is a slow,
        fully synchronous PCIe copy; a page-locked staging buffer makes it DMA-fast, and the result is
        copied into a fresh pageable tensor so the one-patch pinned buffer can be reused (capping pinned
        host RAM at a single patch). Bit-identical to ``layer.detach().cpu()``; falls back to it for
        non-CUDA or small patches, or when the host cannot allocate page-locked memory.
        """
        detached = layer.detach()
        if (
            detached.device.type != "cuda"
            or detached.numel() * detached.element_size() < self._PINNED_OFFLOAD_MIN_BYTES
        ):
            return detached.cpu()
        buffer = self._pin_buffer
        if buffer is None or buffer.shape != detached.shape or buffer.dtype != detached.dtype:
            try:
                buffer = torch.empty(detached.shape, dtype=detached.dtype, pin_memory=True)
            except RuntimeError:  # host cannot lock this much memory -> plain pageable copy
                self._pin_buffer = None
                return detached.cpu()
            self._pin_buffer = buffer
        # Blocking copy into page-locked memory (fast DMA), then a CPU->CPU copy into a fresh pageable
        # tensor so the pinned buffer is free to stage the next patch.
        buffer.copy_(detached)
        out = torch.empty(detached.shape, dtype=detached.dtype)
        out.copy_(buffer)
        return out

    def prepare(self, name_layer: str) -> None:
        self.before_reduction_transforms = []
        self.after_reduction_transforms = []
        self.final_transforms = []
        transforms_type = [
            "before_reduction_transforms",
            "after_reduction_transforms",
            "final_transforms",
        ]
        for name, _transform_type, transform_type in [
            (k, getattr(self, f"_{k}"), getattr(self, k)) for k in transforms_type
        ]:
            if _transform_type is not None:
                for classpath, transform in _transform_type.items():
                    transform = transform.get_transform(
                        classpath,
                        konfai_args=f"{konfai_root()}.outputs_dataset.{name_layer}.OutputDataset.{name}",
                    )
                    transform_type.append(transform)

        if self._patch_combine is not None:
            module, name = get_module(self._patch_combine, "konfai.data.patching")
            self.patch_combine = apply_config(f"{konfai_root()}.outputs_dataset.{name_layer}.OutputDataset")(
                getattr(module, name)
            )()

        module, name = get_module(self.reduction_classpath, "konfai.predictor")
        # get_module returns the module OBJECT: compare its dotted name, not the module against the string
        # (a ``module == "konfai.predictor"`` comparison is always False and takes the custom branch).
        if module.__name__ == "konfai.predictor":
            self.reduction = getattr(module, name)()
        else:
            self.reduction = apply_config(
                f"{konfai_root()}.outputs_dataset.{name_layer}.OutputDataset.{self.reduction_classpath}"
            )(getattr(module, name))()

    def set_datasets(self, datasets: list[Dataset]) -> None:
        for transform in self.before_reduction_transforms:
            transform.set_datasets([*datasets, self])
        for transform in self.after_reduction_transforms:
            transform.set_datasets([*datasets, self])
        for transform in self.final_transforms:
            transform.set_datasets([*datasets, self])

    @abstractmethod
    def setup(self, datasets: list[Dataset], groups: dict[str, list[str]]):
        self.set_datasets(datasets)

    def set_patch_config(
        self,
        patch_size: list[int] | None,
        overlap: int | float | str | list[int | float | str] | None,
        nb_data_augmentation: int,
    ) -> None:
        if patch_size is not None and overlap is not None:
            if self.patch_combine is not None:
                self.patch_combine.set_patch_config(patch_size, blend_overlap(overlap, patch_size))
        else:
            self.patch_combine = None
        self.nb_data_augmentation = nb_data_augmentation

    def to(self, device: torch.device):
        super().to(device)
        transforms_type = [
            "before_reduction_transforms",
            "after_reduction_transforms",
            "final_transforms",
        ]
        for transform_type in [(getattr(self, k)) for k in transforms_type]:
            if transform_type is not None:
                for transform in transform_type:
                    transform.to(device)

    @abstractmethod
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
        raise NotImplementedError()

    def is_done(self, index: int) -> bool:
        # ``.get``: a streamed case cleans itself up inside ``add_layer`` (its slabs are already on
        # disk), so by the time the run loop asks, the index is gone and the answer is "nothing to do".
        accumulators = self.output_layer_accumulator.get(index)
        if accumulators is None or len(accumulators) != self.nb_data_augmentation:
            return False
        return all(acc.is_full() for acc in accumulators.values())

    @abstractmethod
    def get_output(self, index: int, number_of_channels_per_model: list[int], dataset: DatasetIter) -> torch.Tensor:
        raise NotImplementedError()

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

    def reset(self) -> None:
        """Drop every in-flight accumulation (the OOM-restart path re-runs the rank's cases from scratch)."""
        self.output_layer_accumulator.clear()
        self.attributes.clear()
        self.names.clear()
        self._accum_device.clear()
        self._reduce_device.clear()
        self._pin_buffer = None

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


# The write-side region kinds: what SlabRegionStream can carry inside a streamed finalize chain (the
# same set the read dispatcher accepts as its region stages; on both sides they compose).
_REGION_KINDS = (LocalityKind.HALO, LocalityKind.ORIENTATION, LocalityKind.CROP, LocalityKind.RESCALE)

# Streaming pays per-slab work (the pipe traversal, region writes, the TTA aligner); when the
# assembled accumulators of all copies are below this fraction of allocatable memory (a 2.5D case),
# holding them whole costs nothing and the case takes the whole-volume path.
# KONFAI_STREAM_WORTH_THRESHOLD overrides the fraction (tests set 0 to exercise the streamed
# machinery on toy volumes). See OutSameAsGroupDataset._worth_streaming.
_STREAM_WORTH_MIN_FRACTION = 0.05


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


@dataclass(frozen=True)
class _StreamPlan:
    """How one case streams: the post-reduction stages, split into a per-slab pointwise prefix, a
    streamed pipe of region and pointwise stages, and — past what streaming can honour — a
    whole-volume tail.

    ``to_sink`` streams straight into a region-write ``DataStream``; ``pipe_start`` is the first
    region stage (``None`` when the chain is pointwise throughout), and the pipe runs from there to
    the end — region stages compose, so their number is not limited. Without ``to_sink`` the prefix
    streams into a post-reduction buffer and ``stages[tail_start:]`` runs once on it (the chain
    split). The invariants: ``to_sink`` implies no tail, and a pipe implies ``to_sink`` — a tail
    swallows the region stages, so the buffer always sits on the accumulator grid.
    """

    stages: list[_FinalizeStage]
    pipe_start: int | None
    tail_start: int
    to_sink: bool
    # Prefix stages whose declaration is SLAB: per-voxel value maps with a per-region side effect,
    # run through ``Transform.stream_slab`` so they learn where each slab sits.
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

    ``shapes[i]`` is the spatial shape between pipe stage ``i - 1`` and ``i`` (``shapes[0]`` the
    accumulator's, ``shapes[-1]`` the written image's); a pointwise stage leaves it unchanged, so the
    per-stage region bookkeeping folds through the same list the pull map composes over. The pipe's
    stages themselves live in the ``produce``/``pull`` closures the stream was built on.
    """

    shapes: list[list[int]]
    stream: SlabRegionStream | None = None
    # The attribute the latest emission ran the pipe on: what the sink opens with.
    attribute: Attribute | None = None


@config("OutputDataset")
class OutSameAsGroupDataset(OutputDataset):
    """
    Output dataset that mirrors the geometry and transform chain of an input group.

    This is the default output writer used by KonfAI prediction workflows.
    """

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
    ) -> None:
        super().__init__(
            dataset_filename,
            group,
            before_reduction_transforms,
            after_reduction_transforms,
            final_transforms,
            patch_combine,
            reduction,
        )
        self.group_src, self.group_dest = same_as_group.split(":")
        # Slab streaming has no config knob: it is applied automatically, per case, whenever it is
        # byte-identical to the assembled path (see ``_plan_stream``), finalizing each z-slab as its
        # patches complete so RAM is bounded at one patch window; otherwise the whole-volume path is
        # used transparently. ``KONFAI_STREAMED_WRITES=0`` is a global ops/debug kill-switch (also how
        # a test gets the assembled reference), not a per-output option.
        self._streaming_enabled = env_flag("KONFAI_STREAMED_WRITES", True)
        self._stream_plans: dict[int, _StreamPlan | None] = {}
        self._stream_sinks: dict[int, DataStream] = {}
        self._region_states: dict[int, _RegionState] = {}
        self._stream_buffers: dict[int, torch.Tensor] = {}
        self._post_prefix_attributes: dict[int, Attribute] = {}
        self._reported_paths: set[str] = set()
        # One aligner per streamed case: the copies' accumulators emit slabs at their own pace, and
        # the finalize needs every copy's rows together (the cross-copy reduction). A single copy is
        # simply a one-stream aligner — same path, no special case.
        self._aligners: dict[int, SlabAligner] = {}

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
        if (
            index_dataset not in self.output_layer_accumulator
            or index_augmentation not in self.output_layer_accumulator[index_dataset]
        ):
            input_dataset = dataset.get_dataset_from_index(self.group_dest, index_dataset)
            source_attribute = (
                Attribute(attribute) if attribute is not None else Attribute(input_dataset.cache_attributes[0])
            )
            if index_dataset not in self.output_layer_accumulator:
                self.output_layer_accumulator[index_dataset] = {}
                self.attributes[index_dataset] = {}
                self.names[index_dataset] = input_dataset.name
                plan = (
                    self._plan_stream(dataset, index_dataset, source_attribute, layer, number_of_channels_per_model)
                    if self._streaming_enabled
                    else None
                )
                self._stream_plans[index_dataset] = plan
                if self._streaming_enabled and (plan is None or not plan.to_sink):
                    # The whole-volume fallback is a normal outcome, but a silent one hides that a
                    # large case pays it: say so, once per distinct path.
                    path = (
                        "whole-volume" if plan is None else "buffered (the prefix streams, the tail runs whole-volume)"
                    )
                    self._report_once(path, f"streaming: case '{input_dataset.name}' takes the {path} path.")
            self.attributes[index_dataset][index_augmentation] = {}

            accumulator_type = StreamingAccumulator if self._stream_plans[index_dataset] else Accumulator
            self.output_layer_accumulator[index_dataset][index_augmentation] = accumulator_type(
                input_dataset.patch.get_patch_slices(index_augmentation),
                input_dataset.patch.patch_size,
                self.patch_combine,
                batch=False,
            )

            for i in range(len(input_dataset.patch.get_patch_slices(index_augmentation))):
                self.attributes[index_dataset][index_augmentation][i] = Attribute(source_attribute)

        for transform in reversed(dataset.groups_src[self.group_src][self.group_dest].patch_transforms):
            if isinstance(transform, TransformInverse) and transform.apply_inverse:
                layer = transform.inverse(
                    self.names[index_dataset],
                    layer,
                    self.attributes[index_dataset][index_augmentation][index_patch],
                )
        accumulator = self.output_layer_accumulator[index_dataset][index_augmentation]
        if index_dataset not in self._accum_device:
            self._accum_device[index_dataset] = self._accumulate_device(layer, accumulator)
            if layer.device.type != "cpu" and self._accum_device[index_dataset].type == "cpu":
                self._report_once(
                    "host-accumulate",
                    f"case '{self.names[index_dataset]}' accumulates on the host.",
                )
        target = self._accum_device[index_dataset]
        # When the accumulator lives on the GPU, blend the patch straight in (no host round-trip);
        # otherwise offload each patch to CPU so its device memory is released after post-processing.
        if target.type == "cpu":
            if layer.device.type != "cpu":
                layer = self._offload_to_cpu(layer)
        elif str(layer.device) != str(target):
            layer = layer.to(target)
        try:
            slabs = accumulator.add_layer(index_patch, layer) or []
        except torch.cuda.OutOfMemoryError:
            # The gate samples free VRAM once per case: another process can reclaim it before this
            # volume-sized first allocation lands. Nothing is blended yet, so fall back to the
            # memory-safe CPU blend for the rest of the case; ``get_output`` reconciles augmentations
            # already blended on the GPU. A mid-blend OOM (buffer already resident) stays fatal.
            if layer.device.type == "cpu" or not accumulator.is_empty():
                raise
            self._accum_device[index_dataset] = torch.device("cpu")
            torch.cuda.empty_cache()
            slabs = accumulator.add_layer(index_patch, self._offload_to_cpu(layer)) or []
        if not self._stream_plans.get(index_dataset):
            return
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
        """Whether a finalize stage is a per-voxel map here: POINTWISE, or GLOBAL_STAT whose statistic
        the case already carries — the finalize attribute holds what the forward pass pushed, and there
        is no stored volume left to derive a missing one from."""
        if locality.kind is LocalityKind.POINTWISE:
            return True
        return locality.kind is LocalityKind.GLOBAL_STAT and all(key in attribute for key in locality.stat_keys)

    def _report_once(self, key: str, message: str) -> None:
        """One line the first time a distinct non-window-bounded outcome appears; repeats are silent."""
        if key not in self._reported_paths:
            self._reported_paths.add(key)
            print(f"[KonfAI] {message}")

    def _worth_streaming(self, dataset: DatasetIter, index: int, layer: torch.Tensor) -> bool:
        """Whether this case's accumulators are heavy enough for the per-slab machinery to pay:
        every copy holds a volume-sized accumulator, and when all of them together are a sliver of
        allocatable memory (a 2.5D case) the assembled path costs nothing to hold — streaming it
        would spend pipe traversals and region writes to save nothing.

        ``layer`` carries the accumulator's channel count and dtype whatever the ensemble combine is
        (a Concat layer arrives already concatenated); the estimate is taken before the patch-level
        inverses, so a dtype-widening inverse under-counts by at most 2x — inside the threshold's
        margin."""
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
        return assembled >= fraction * available_memory_bytes()[0]

    def _plan_stream(
        self,
        dataset: DatasetIter,
        index: int,
        attribute: Attribute,
        layer: torch.Tensor | None = None,
        number_of_channels_per_model: list[int] | None = None,
    ) -> _StreamPlan | None:
        """The streaming plan for this case, or ``None`` for the whole-volume path.

        The streamed part of the finalize chain is ``[pointwise*][region and pointwise stages]``:
        region stages compose — each pulls through the one before it — so any number of geometry
        inverses streams to the write. What streaming cannot honour — a WHOLE_VOLUME stage, a
        statistic nothing seeded — becomes a whole-volume TAIL: the prefix still streams slab by slab
        into a light post-reduction buffer and the tail runs once on that buffer. A destination that
        cannot serve region writes buffers too, and writes classically. Only a non-streamable start
        refuses outright: a reduction that is not voxel-local, a non-voxel-local before-reduction
        transform (it runs per model chunk inside the slab prefix), a TTA copy whose un-augment does
        not act slab by slab (see ``_tta_streamable``), or a case too light for the per-slab
        machinery to pay (``_worth_streaming`` — gauged from ``layer`` when the caller has one).
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
            # A SLAB before-reduction transform (e.g. Mask) streams per slab through ``stream_slab`` in
            # ``_prepare_copy_slab``, so it does not force the whole-volume path; anything else that is
            # not voxel-local (a spatial mix, an unseeded statistic) still refuses outright.
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
                # A per-voxel stage with a per-region side effect streams through ``stream_slab`` —
                # but only on the accumulator grid: past a region stage the emissions are regions of
                # ANOTHER space, so there it falls to the tail (whose whole-volume call is its
                # classic behaviour).
                slab_stages.add(position)
                continue
            if locality.kind in _REGION_KINDS:
                if pipe_start is None:
                    pipe_start = position
                continue
            tail_start = position
            break
        if tail_start < len(stages) or not self.can_stream_data(attribute):
            # A whole-volume tail swallows the region stages too: the buffer sits on the accumulator
            # grid, and the tail runs the true whole-volume operators on it — byte-identical for free.
            if pipe_start is not None:
                tail_start = min(tail_start, pipe_start)
            return _StreamPlan(stages, None, tail_start, to_sink=False, slab_stages=frozenset(slab_stages))
        return _StreamPlan(stages, pipe_start, len(stages), to_sink=True, slab_stages=frozenset(slab_stages))

    @staticmethod
    def _copy_draw(dataset: DatasetIter, index_augmentation: int) -> tuple[list[DataAugmentation], int] | None:
        """The augmentations copy ``index_augmentation`` carries and its index within their list, or
        ``None`` for the un-augmented copy."""
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
        """Undo copy ``index_augmentation``'s draw on ``tensor`` — the augmentations applied in
        reverse, bound by the case index the draw was made under (the manager's own)."""
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

        The slab-synchronized reduce applies each augmentation's ``inverse`` to a finalized z-slab,
        which equals the whole-volume inverse restricted to that slab exactly when the draw maps every
        slab onto itself: a POINTWISE draw does (a per-voxel map inverts per voxel), and an
        ORIENTATION draw does when its declared region remap fixes the slab axis row for row and its
        shape fold keeps the slab extent — probed here against ``stream_region_source``, never by
        running patches. A z-flip mirrors the rows (row 0 pulls the last), a z-moving permute
        relocates them: both fail the probe and the case falls back to the whole-volume path. Any
        other kind (a halo'd translate, a whole-volume draw) refuses outright.
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
        except Exception:  # nosec B110 - an unprobeable draw just keeps the case on the whole-volume path
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
        stream, or buffer. The first slab fixes the case's state (post-prefix attribute, region
        scheduler or buffer) and may demote a RESCALE region to the buffered tail (see
        ``_init_stream_state``)."""
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
        """Fix the case's streaming state at its first slab, when the prefix output is known.

        A RESCALE stage streams through ``resample_region``, which matches ``F.interpolate`` bit for
        bit in nearest mode (uint8) and to ~float-rounding in linear mode, so a rescale streams by
        default and bounds a large float resample to a window. ``KONFAI_STREAM_LINEAR_RESAMPLE=0``
        demotes a float rescale here to the buffered whole-volume tail for a run that needs exactness;
        the demotion leaves the prefix untouched (the pipe was never part of it).
        """
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
        """Wire the case's streamed pipe into one :class:`SlabRegionStream` — or answer ``None`` for
        the buffered tail where streaming would not be exact.

        Region stages compose: the pull map folds each stage's own declaration backward — a written
        region pulls through the last stage, whose region pulls through the one before it, down to
        the accumulator — and ``produce`` walks the pipe forward over the pulled window, handing each
        stage the region pair the same fold computed for it. Pointwise stages ride along unchanged.

        The fold is planned by walking a one-voxel corner of the real first slab through the pipe
        with one evolving attribute: each stage declares against, and remaps from, the state the
        stages before it left — a second resample pops the Size stack the first one already popped,
        a reorientation after a permute reads the moved axes — and the walk carries the dtype, so a
        float RESCALE (``resample_region`` matches ``F.interpolate`` only to ~float-rounding) answers
        ``None`` — demoting to the whole-volume tail — only under ``KONFAI_STREAM_LINEAR_RESAMPLE=0``.
        ``produce`` then replays the same transitions on a fresh copy per emission (the same
        slab-local scoping as the prefix).
        """
        attr0 = self._post_prefix_attributes[index]
        pipe = plan.stages[cast(int, plan.pipe_start) :]
        name = self.names[index]

        probe = block[(Ellipsis, *([slice(0, 1)] * len(in_shape)))].clone()
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
                    pull_fns.append(_halo_pull(_halo_radii(locality.halo, len(shape)), shape))
                    shapes.append(list(shape))
                    probe = stage(name, probe, walking)
                elif locality.kind is LocalityKind.RESCALE:
                    # A float rescale streams within a window: resample_region computes the same linear
                    # taps the read side already streams, matching the whole-volume F.interpolate to
                    # ~float-rounding (a boundary voxel or two flips after argmax; a raw float output
                    # differs by ~1 ULP) -- which bounds a large float resample (a probability volume
                    # sent back to native) instead of holding it whole. KONFAI_STREAM_LINEAR_RESAMPLE=0
                    # forces the exact whole-volume resample for a run that needs bit-identity. Nearest
                    # (uint8) is byte-identical either way.
                    if probe.dtype is not torch.uint8 and not env_flag("KONFAI_STREAM_LINEAR_RESAMPLE", True):
                        return None
                    resample = cast(Resample, stage.transform)
                    if stage.inverted:
                        pull_fns.append(_remap_pull(resample.stream_region_target, shape, snapshot))
                        out = resample._inverse_geometry(walking)
                    else:
                        out = [
                            int(extent)
                            for extent in resample.transform_shape(
                                self.group_src, name, list(shape), Attribute(walking)
                            )
                        ]
                        scales = [shape[k] / out[k] for k in range(len(shape))]
                        pull_fns.append(_scale_pull(scales, shape))
                        resample.write_stream_cache_attribute(walking, shape)
                    shapes.append([int(extent) for extent in out])
                elif locality.kind in _REGION_KINDS:
                    if stage.inverted:
                        remapper = cast(TransformInverse, stage.transform)
                        pull_fns.append(_remap_pull(remapper.stream_region_target, shape, snapshot))
                        out = remapper.inverse_transform_shape(list(shape), Attribute(walking))
                    else:
                        pull_fns.append(_remap_pull(stage.transform.stream_region_source, shape, snapshot))
                        out = stage.transform.transform_shape(self.group_src, name, list(shape), Attribute(walking))
                    shapes.append([int(extent) for extent in out])
                    # The stage's attribute transition, on a one-voxel corner: a crop's tensor answer
                    # is meaningless there (the map is the action) but its pops are the case's.
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
        """Run one pipe stage on its pulled block, by declared kind — never by stage name."""
        if kind is LocalityKind.CROP:
            # The pull already translated the region, so the block IS the answer. The stage still runs
            # for its attribute transition (a crop restores the origin it recorded); its tensor answer
            # is one window's, dropped.
            stage(name, block, attribute)
            return block
        if kind is LocalityKind.RESCALE:
            resample = cast(Resample, stage.transform)
            scales = [in_shape[k] / out_shape[k] for k in range(len(in_shape))]
            result = resample.resample_region(block, target, [s.start for s in source], scales, in_shape)
            if stage.inverted:
                resample._inverse_geometry(attribute)
            else:
                resample.write_stream_cache_attribute(attribute, in_shape)
            return result
        if kind is LocalityKind.ORIENTATION and not stage.inverted:
            # A forward orientation writes the case origin/direction from the extent it is handed; run
            # the tensor action on a throwaway scope so it does not record the SLAB's extent, then write
            # the case geometry from the full ``in_shape`` (its documented contract) -- as RESCALE does.
            result = stage(name, block, Attribute(attribute))
            cast(TransformInverse, stage.transform).write_stream_cache_attribute(attribute, in_shape)
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
        """Write one finalized output block into the case's sink (opened at the first block, once the
        chain has fixed the output's shape, channel count and dtype).

        The whole write — device-to-host copy, lazy sink open, region write — is one submitted
        operation, so ``_stream_sinks`` is only ever touched in submission order; the attribute is
        snapshotted because the region state's evolves with later emissions."""
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
        """One copy's slab through the per-copy head of ``_get_output``: un-augment it (exact on a
        slab — the gate admitted only slab-parallel draws), split the model chunks, run
        before_reduction on each, and stack to the copy's ``[1, M, C, ...]`` block. A SLAB
        before-reduction transform learns where the slab sits through ``stream_slab`` (the accumulator
        grid, where before_reduction runs), so it reads its slab region instead of the whole volume."""
        layer = self._unaugment(dataset, index, index_augmentation, layer)
        attribute = Attribute(self.attributes[index][index_augmentation][0])
        chunks = self._split_model_chunks(layer, number_of_channels_per_model, attribute)
        results = []
        for chunk in chunks:
            for transform in self.before_reduction_transforms:
                if transform.patch_locality(Attribute(attribute)).kind is LocalityKind.SLAB:
                    chunk = transform.stream_slab(self.names[index], chunk, region, spatial, Attribute(attribute))
                else:
                    chunk = transform(self.names[index], chunk, Attribute(attribute))
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

        Every prefix stage passed the gate as voxel-local (a SLAB stage additionally learns where the
        slab sits, through ``stream_slab``) and every copy as slab-parallel, so each step is the
        whole-volume computation restricted to the slab — same ops, same order, same reduction call —
        which is what makes the streamed output byte-identical. Each slab gets its own copy of the
        case attribute (transform writes stay slab-local, and case-level pops repeat identically per
        slab).
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
        for position, stage in enumerate(plan.stages[: plan.boundary]):
            if position in plan.slab_stages:
                result = stage.transform.stream_slab(self.names[index], result, region, spatial, attribute)
            else:
                result = stage(self.names[index], result, attribute)
        return result, attribute

    def reset(self) -> None:
        # Aborting an attempt mid-stream: abort each open sink so the backend removes the partial
        # entry (a reader must never see a half-written volume); the restart rewrites it.
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
        super().reset()

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
        super().setup(datasets, groups)

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
        layer whose channels do not match the ensemble layout stays whole. One splitter serves the
        whole-volume and slab paths, so the two cannot drift."""
        if number_of_channels_per_model and layer.shape[0] == sum(number_of_channels_per_model):
            attribute["number_of_channels_per_model_0"] = torch.tensor(number_of_channels_per_model)
            return list(torch.split(layer, number_of_channels_per_model, dim=0))
        return [layer]

    def _reduce_copies(self, copies: list[torch.Tensor]) -> torch.Tensor:
        """The cross-copy reduction, identical for a slab and a whole volume — the streamed path's
        byte-identity rests on the two staying in lockstep.

        Mixed devices can only come from a mid-case OOM fallback: reconcile on the host. Reduce, then
        drop the singleton stack axis; Mean/Median also drop the singleton model axis, while Concat
        keeps the ``[M, C, ...]`` model axis for after_reduction (Sum) to merge into labels."""
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

        # The per-model channel reduction (softmax/argmax over the class dimension of a whole-volume
        # multi-class output) materialises a working volume on top of the resident accumulator. Decide
        # once per case whether it fits free VRAM, with the accumulator already resident whatever device
        # it sits on: if it fits, reduce on the GPU (a no-op move when the volume is already there); else
        # move the finalize to the host. One decision per case -- deciding per augmentation would let free
        # VRAM shrinking between augmentations flip the device mid-case and hand the reduction a
        # mixed-device list.
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
            # Keep the chunk on its current device; ``get_output`` decides once (via the GPU-finalize
            # gate) whether the whole finalize chain stays on the GPU or moves back to the host.
            results.append(layer)

        # Mean, Median -> [1, C, ...] | Concat -> [M, C, ...]
        return torch.stack(results, dim=0)

    def _reduction_device(self, chunk: torch.Tensor, nb_chunks: int = 1) -> torch.device:
        """Device for the channel-reduction transforms: this dataset's CUDA device when every chunk (plus
        working headroom) fits free VRAM, else CPU (the memory-safe fallback)."""
        # NeedDevice stores a CUDA ordinal (int) on GPU and a torch.device on CPU; normalise to a device.
        device = torch.device("cuda", self.device) if isinstance(self.device, int) else self.device
        if device.type != "cuda":
            return torch.device("cpu")
        try:
            # The forward pass leaves the allocator holding a large reserved cache; release the unused part
            # back to the driver so a genuinely-free GPU is not mistaken for a full one.
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(device)
        except Exception:  # nosec B110 - any CUDA query failure just keeps the reduction on CPU
            return torch.device("cpu")
        # Every transformed chunk is parked on the reduce device until the final stack (a combine:Concat
        # ensemble keeps M of them), so budget all of them plus a same-size working temp per chunk and
        # one stack copy.
        needed = chunk.numel() * chunk.element_size() * (2 * max(1, nb_chunks) + 1)
        return device if needed < free else torch.device("cpu")

    # A forward runs alongside the resident accumulator on every patch; the memory queries below happen
    # before those allocations land, so keep ~10 % of free VRAM in reserve for fragmentation and a
    # concurrent process.
    _ACCUMULATE_MARGIN = 0.9

    def _accumulate_device(self, layer: torch.Tensor, accumulator: Accumulator) -> torch.device:
        """Device on which to blend a case's patches. On the GPU the accumulator stays resident through
        the case (no per-patch offload, no CPU blend, and the reduction runs where the volume already is).
        Decided once per case, at the first patch, when the accumulator fits alongside the memory a
        forward needs; else the accumulation runs on the CPU."""
        device = torch.device("cuda", self.device) if isinstance(self.device, int) else self.device
        if device.type != "cuda" or layer.device.type != "cuda":
            return torch.device("cpu")
        try:
            # Return the reserved-but-unused cache so ``mem_get_info`` reports the memory actually free.
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(device)
            # A forward's transient footprint above the resident set, measured on the batch that just ran
            # (its activations are already freed). ``max_memory_allocated`` is a high-water mark, so this
            # bounds the next forward from above -- the gate errs toward the CPU, never toward an OOM.
            transient = torch.cuda.max_memory_allocated(device) - torch.cuda.memory_allocated(device)
        except Exception:  # nosec B110 - any CUDA query failure keeps the blend on CPU
            return torch.device("cpu")
        voxels = int(np.prod(accumulator.footprint_shape))
        # result [C, volume] + weight_sum [volume] at the patch dtype, for EVERY augmentation of the
        # case: ``is_done`` requires all augmentations complete before ``get_output``, so their
        # accumulators are resident simultaneously and the per-case device decision must budget them all.
        accumulator_bytes = (layer.shape[0] + 1) * voxels * layer.element_size() * max(1, self.nb_data_augmentation)
        # During accumulation the resident accumulator and one forward coexist. The channel reduction's
        # working volume is budgeted separately, at ``get_output`` time, by ``_reduction_device``.
        needed = accumulator_bytes + transient
        if isinstance(accumulator, StreamingAccumulator):
            # Transients on top of the resident window (``voxels`` is the window footprint): the advance
            # clone of the retained rows, the emission slab and its weight clamp, and — when the
            # background writer engages — up to ``_AsyncWriter._CAPACITY`` emitted blocks alive on the
            # device until their device-to-host copy runs. Two window footprints bound the sum.
            needed += 2 * layer.shape[0] * voxels * layer.element_size()
            if self.nb_data_augmentation > 1:
                # Slab-aligned TTA holds pending slabs per copy (the arrival skew, ~one window) and
                # reduces the joint interval through a float32 accumulate: budget one window per copy
                # plus one more for the reduction's transients.
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
        # The volume stays on whatever device it was blended on (GPU when it fit VRAM, else CPU): the
        # reduction and every finalize transform are device- and dtype-transparent, so the whole finalize
        # simply runs where the volume already is. Only the final result is returned to the host.
        result = self._reduce_copies(results)
        # Reduction strategy overview:
        #
        # Terminology:
        #   - combine : aggregation across models (model ensembling)
        #   - reduce  : aggregation across TTA (test-time augmentation)
        #
        # Let:
        #   M = number of models
        #   T = number of TTA samples
        #   C = number of output channels
        #
        # Case 1 - combine = Mean / Median, reduce = Mean / Median:
        #   Models are aggregated first:
        #     [M, C, ...] -> combine -> [C, ...]
        #   TTA samples are then reduced:
        #     [T, C, ...] -> reduce -> [C, ...]
        #
        # Case 2 - combine = Mean / Median, reduce = Concat:
        #   Models are aggregated first:
        #     [M, C, ...] -> combine -> [C, ...]
        #   TTA samples are concatenated:
        #     [T, C, ...] -> concat -> [T, C, ...]
        #
        # Case 3 - combine = Concat, reduce = Mean / Median:
        #   Model outputs are concatenated:
        #     [M, C, ...] -> concat -> [M, C, ...]
        #   TTA samples are then reduced:
        #     [T, M, C, ...] -> reduce -> [M, C, ...]
        #
        # Case 4 - combine = Concat, reduce = Concat:
        #   No reduction is applied at either level:
        #     [M, C, ...] x T -> concat -> [M * T, C, ...]
        #
        # Important:
        #   If combine = Concat or reduce = Concat,
        #   the first transform in `after_reduction_transforms`
        #   must be either `InferenceStack` or `Sum`,
        #   to ensure a [C, ....] after
        for transform in self.after_reduction_transforms:
            result = transform(self.names[index], result, self.attributes[index][0][0])

        for transform in reversed(dataset.groups_src[self.group_src][self.group_dest].transforms):
            if isinstance(transform, TransformInverse) and transform.apply_inverse:
                result = transform.inverse(self.names[index], result, self.attributes[index][0][0])

        for transform in self.final_transforms:
            result = transform(self.names[index], result, self.attributes[index][0][0])

        return result.cpu() if result.device.type != "cpu" else result


@config("OutputDataset")
class OutputDatasetLoader:
    """Factory that instantiates output dataset classes from predictor config."""

    def __init__(self, name_class: str = "OutSameAsGroupDataset") -> None:
        self.name_class = name_class

    def get_output_dataset(self, layer_name: str) -> OutputDataset:
        return apply_config(f"Predictor.outputs_dataset.{layer_name}")(
            getattr(importlib.import_module("konfai.predictor"), self.name_class)
        )()


class _Predictor:
    """
    Internal class that runs distributed inference over a dataset using a composite model.

    This class handles patch-wise prediction, output accumulation, logging to TensorBoard, and
    writing final predictions to disk. It is designed to be used as a context manager and
    supports model ensembles via `ModelComposite`.

    Args:
        world_size (int): Total number of processes or GPUs used.
        global_rank (int): Rank of the current process across all nodes.
        local_rank (int): Local GPU index within a single node.
        autocast (bool): Whether to use automatic mixed precision (AMP).
        predict_path (str): Output directory path where predictions and metrics are saved.
        data_log (list[str] | None): List of logging targets in the format 'group/DataLogType/N'.
        outputs_dataset (dict[str, OutputDataset]): Dictionary of output datasets to store predictions.
        model_composite (Model): Model container that wraps the prediction model(s).
        dataloader_prediction (DataLoader): DataLoader that provides prediction batches.
    """

    def __init__(
        self,
        world_size: int,
        global_rank: int,
        local_rank: int,
        autocast: bool,
        predict_path: Path,
        data_log: list[str] | None,
        outputs_dataset: dict[str, OutputDataset],
        model_composite: Model,
        dataloader_prediction: DataLoader,
    ) -> None:
        self.world_size = world_size
        self.global_rank = global_rank
        self.local_rank = local_rank

        self.model_composite = model_composite

        self.dataloader_prediction = dataloader_prediction
        self.outputs_dataset = outputs_dataset
        self.autocast = autocast
        self.it = 0

        self.dataset: DatasetIter = self.dataloader_prediction.dataset
        patch_size, overlap = self.dataset.get_patch_config()
        for output_dataset in self.outputs_dataset.values():
            output_dataset.set_patch_config(
                [size for size in patch_size if size > 1] if patch_size else None,
                overlap,
                np.max(
                    [
                        int(
                            np.sum([data_augmentation.nb for data_augmentation in self.dataset.data_augmentations_list])
                            + 1
                        ),
                        1,
                    ]
                ),
            )
        self.data_log: dict[str, tuple[DataLog, int]] = {}
        if data_log is not None:
            for data in data_log:
                self.data_log[data.split("/")[0].replace(":", ".")] = (
                    DataLog[data.split("/")[1]],
                    int(data.split("/")[2]),
                )
        self._has_runtime_measures = any(
            network.measure is not None for network in self.model_composite.module.get_networks().values()
        )
        if self._has_runtime_measures or len(self.data_log):
            if SummaryWriter is None:
                raise ImportError(
                    "TensorBoard is required for prediction logging. Install it with: pip install konfai[tensorboard]"
                )
            self.tb = SummaryWriter(log_dir=predict_path / "Metric")
        else:
            self.tb = None

    def __enter__(self):
        """
        Enters the prediction context and returns the predictor instance.
        """
        return self

    def __exit__(self, exc_type, value, traceback):
        """
        Closes the TensorBoard writer upon exit.
        """
        if self.tb:
            self.tb.close()

    def run(self):
        """
        Run the full prediction loop.

        Iterates over the prediction DataLoader, performs inference using the composite model,
        applies reduction (e.g., mean), and writes the final results using each `OutputDataset`.

        Also logs intermediate data and metrics to TensorBoard if enabled.
        """

        self.model_composite.eval()
        self.model_composite.module.set_state(NetState.PREDICTION)
        self.dataloader_prediction.dataset.load("Prediction")
        try:
            self._run_batches()
        finally:
            # Every submitted write must be on disk before the run returns — including on the error
            # path, where the drain also closes the sinks the abort operations enqueued.
            for output_dataset in self.outputs_dataset.values():
                output_dataset.finalize_writes()

    def _run_batches(self) -> None:
        with tqdm.tqdm(
            iterable=enumerate(self.dataloader_prediction),
            leave=True,
            desc=f"Prediction : {description(self.model_composite)}",
            total=len(self.dataloader_prediction),
            ncols=0,
        ) as batch_iter:
            with torch.inference_mode():
                with torch.amp.autocast("cuda", enabled=self.autocast):
                    for _, batch_sample in batch_iter:
                        outputs = self.model_composite(
                            batch_sample,
                            list(self.outputs_dataset.keys()),
                        )
                        self._predict_log(batch_sample)
                        for name, number_of_channels_per_model, output in outputs:
                            output_dataset = self.outputs_dataset[name]
                            group = getattr(output_dataset, "group_dest", next(iter(batch_sample)))
                            for i, (index, patch_augmentation, patch_index) in enumerate(
                                [
                                    (int(index), int(patch_augmentation), int(patch_index))
                                    for index, patch_augmentation, patch_index in zip(
                                        batch_sample[group].x,
                                        batch_sample[group].a,
                                        batch_sample[group].p,
                                        strict=False,
                                    )
                                ]
                            ):
                                output_dataset.add_layer(
                                    index,
                                    patch_augmentation,
                                    patch_index,
                                    output[i],
                                    self.dataset,
                                    batch_sample[group].attribute[i],
                                    number_of_channels_per_model,
                                )
                                if output_dataset.is_done(index):
                                    output_dataset.write_prediction(
                                        index,
                                        batch_sample[group].name[i],
                                        output_dataset.get_output(index, number_of_channels_per_model, self.dataset),
                                    )

                        batch_iter.set_description(f"Prediction : {description(self.model_composite)}")
                        self.it += 1

    def _predict_log(
        self,
        batch_sample: BatchSample,
    ):
        """
        Log prediction results to TensorBoard, including images and metrics.

        This method handles:
        - Logging image-like data (e.g., inputs, outputs, masks) using `DataLog` instances,
        based on the `data_log` configuration.
        - Logging scalar loss and metric values (if present in the network) under the `Prediction/` namespace.
        - Dynamically retrieving additional feature maps or intermediate layers if requested via `data_log`.

        Logging is performed only on the global rank 0 process and only if `TensorBoard` is active.

        Args:
            data_dict (dict): Dictionary mapping group names to 6-tuples containing:
                - input tensor,
                - index,
                - patch_augmentation,
                - patch_index,
                - metadata (list of strings),
                - `requires_grad` flag (as a tensor).
        """
        if self.tb is None or self.global_rank != 0:
            # Prediction logging is a rank-0 progress indicator; gate before touching the measures so a
            # non-zero rank never enters a cross-rank collective the unequal shards would deadlock on.
            return

        measures: dict[str, tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]] = {}
        if self._has_runtime_measures:
            measures = DistributedObject.get_measure(
                1,
                0,
                self.local_rank,
                {"": self.model_composite.module},
                1,
                sync=False,
            )

        images_log = []
        if len(self.data_log):
            for name, data_type in self.data_log.items():
                if name in batch_sample:
                    data_type[0](
                        self.tb,
                        f"Prediction/{name}",
                        batch_sample[name].tensor[: self.data_log[name][1]].detach().cpu().numpy(),
                        self.it,
                    )
                else:
                    images_log.append(name.replace(":", "."))

        for name, network in self.model_composite.module.get_networks().items():
            if network.measure is not None:
                self.tb.add_scalars(
                    f"Prediction/{name}/Loss",
                    {k.replace(":", "."): v[1] for k, v in measures[name][0].items()},
                    self.it,
                )
                self.tb.add_scalars(
                    f"Prediction/{name}/Metric",
                    {k.replace(":", "."): v[1] for k, v in measures[name][1].items()},
                    self.it,
                )
            if len(images_log):
                for name, layer, _ in self.model_composite.module.get_layers(
                    [v.tensor for v in batch_sample.values() if v.is_input],
                    images_log,
                ):
                    self.data_log[name][0](
                        self.tb,
                        f"Prediction/{name}",
                        layer[: self.data_log[name][1]].detach().cpu().numpy(),
                        self.it,
                    )


def _colocate_loaded_modules(model: torch.nn.Module) -> None:
    """Move any still-CPU leaf module onto the model's device.

    A custom :meth:`Network.load` may append modules after the model was already placed on its
    device — e.g. a head sized from the checkpoint's class count — and those default to CPU, which
    then raises a device mismatch on the forward pass. This re-homes any fully-CPU leaf onto the
    device the rest of the model already lives on. Modules already on a device (including
    model-parallel splits across several GPUs) are left untouched.
    """
    target = next((p.device for p in model.parameters() if p.device.type != "cpu"), None)
    if target is None:
        return
    for sub in model.modules():
        # ModuleArgsDict overrides parameters()/buffers() without a ``recurse`` kwarg, so use the
        # base nn.Module methods to read each module's own (non-recursive) tensors.
        own = [
            *torch.nn.Module.parameters(sub, recurse=False),
            *torch.nn.Module.buffers(sub, recurse=False),
        ]
        if own and all(t.device.type == "cpu" for t in own):
            sub.to(target)


class ModelComposite(Network):
    """
    A composite model that replicates a given base network multiple times and combines their outputs.

    This class is designed to handle model ensembles or repeated predictions from the same architecture.
    It creates `nb_models` deep copies of the input `model`, each with its own name and output branch,
    and aggregates their outputs using a provided `Reduction` strategy (e.g., mean, median).

    Args:
        model (Network): The base network to replicate.
        nb_models (int): Number of copies of the model to create.
        combine (Reduction): The reduction method used to combine outputs from all model replicas.

    Attributes:
        combine (Reduction): The reduction method used during forward inference.
    """

    def __init__(self, model: Network, combine: Reduction):
        super().__init__(
            model.in_channels,
            model.optimizer,
            model.lr_schedulers_loader,
            model.outputs_criterions_loader,
            model.patch,
            model.nb_batch_per_step,
            model.init_type,
            model.init_gain,
            model.dim,
        )
        self.combine = combine
        self._model_name = "Model_0"
        self._base_model_name = model.get_name()
        self._state_sources: list[dict[str, Any] | Path | str] = []
        self._loaded = False  # load() has run: distinguishes "not loaded yet" from "loaded, weightless"
        self._loaded_state_index: int | None = None
        # Cache the CPU state_dict per index so a local-path ensemble is read from
        # disk once, not re-read + re-unpickled on every batch (the index cycles
        # 0..N-1 each forward, so the next batch would otherwise reload all N).
        self._state_cache: dict[int, dict[str, Any]] = {}
        self.add_module(
            self._model_name,
            copy.deepcopy(model),
            in_branch=[0],
            out_branch=["output_0"],
        )

    def _get_model(self) -> Network:
        return cast(Network, self[self._model_name])

    def _read_state_source(self, source: dict[str, Any] | Path | str) -> dict[str, Any]:
        if isinstance(source, dict):
            return source
        if isinstance(source, str) and source.startswith("https://"):
            return torch.hub.load_state_dict_from_url(url=source, map_location="cpu", check_hash=True)
        return safe_torch_load(source, torch.device("cpu"))

    def _ensure_model_loaded(self, index: int) -> Network:
        model = self._get_model()
        if self._loaded_state_index != index:
            state = self._state_cache.get(index)
            if state is None:
                state = self._read_state_source(self._state_sources[index])
                self._state_cache[index] = state
            # Checkpoints are keyed by the base model name, not by the streamed
            # ensemble suffix added after the previous load.
            model.set_name(self._base_model_name)
            model.load(state, init=False)
            # A custom load() may append checkpoint-sized modules (e.g. the head) on CPU; co-locate
            # them with the already device-placed model so the forward pass doesn't hit a mismatch.
            _colocate_loaded_modules(model)
            model.set_name(f"{self._base_model_name}_{index}")
            self._loaded_state_index = index
        return model

    def _model_for_index(self, index: int) -> Network:
        # With no checkpoint sources the model is weightless (0 parameters, e.g. a classical/optimisation
        # engine): run it as constructed, once. The Predictor guards this -- it only reaches here with empty
        # sources when the model has no parameters to load -- so there is nothing to stream.
        if not self._state_sources:
            return self._get_model()
        return self._ensure_model_loaded(index)

    def load(self, state_sources: list[dict[str, Any] | Path | str]):
        """
        Load weights for each sub-model in the composite from the corresponding state dictionaries.

        Args:
            state_sources (list): One checkpoint source per model replica. Empty ONLY for a weightless model
                (0 parameters), which is then run once with its constructed weights; empty sources for a model
                that has trainable parameters is refused here, so a caller cannot silently run random weights.
        """
        if not state_sources and any(parameter.numel() for parameter in self._get_model().parameters()):
            raise PredictorError(
                "ModelComposite.load() received no checkpoint sources for a model with trainable parameters.",
                "A weightless model (0 parameters) may run with no checkpoint; a parameterised one may not.",
                "Pass at least one checkpoint source, or wrap a model that has no parameters.",
            )
        self._state_sources = state_sources
        self._loaded = True
        self._loaded_state_index = None
        self._state_cache = {}
        if len(self._state_sources) == 1:
            self._ensure_model_loaded(0)

    @torch.inference_mode()
    def forward(  # type: ignore[override]
        self,
        data_dict: dict[tuple[str, bool], torch.Tensor],
        output_layers: list[str] = [],
    ) -> list[tuple[str, list[int], torch.Tensor]]:
        """
        Perform a forward pass on all model replicas and aggregate their outputs.

        Args:
            data_dict (dict): A dictionary mapping (group_name, requires_grad) to input tensors.
            output_layers (list): List of output layer names to extract from each sub-model.

        Returns:
            list[tuple[str, torch.Tensor]]: Aggregated output for each layer, after applying the reduction.
        """
        final_outputs: list[tuple[str, list[int], torch.Tensor]] = []
        if not self._loaded:
            raise PredictorError(
                "ModelComposite.forward() called before load().",
                "Prediction ran before the composite's checkpoint sources were set.",
                "Call load(...) first (load([]) for a weightless model).",
            )
        # A weightless model (loaded with no checkpoint sources) is a single replica: the model as constructed.
        n_replicas = len(self._state_sources) or 1
        if isinstance(self.combine, Mean):
            sum_acc: dict[str, torch.Tensor] = {}
            count: dict[str, int] = defaultdict(int)
            channels: dict[str, list[int]] = defaultdict(list)
            for model_index in range(n_replicas):
                for key, tensor in self._model_for_index(model_index)(data_dict, output_layers):
                    if tensor.dtype == torch.float32:
                        tensor = tensor.to(torch.float16)
                    channels[key].append(tensor.shape[1])
                    if key not in sum_acc:
                        sum_acc[key] = tensor
                    else:
                        sum_acc[key].add_(tensor)
                    count[key] += 1
            for key, acc in sum_acc.items():
                final_outputs.append((key, channels[key], (acc / count[key])))
        else:
            aggregated = defaultdict(list)
            for model_index in range(n_replicas):
                for key, tensor in self._model_for_index(model_index)(data_dict, output_layers):
                    if tensor.dtype == torch.float32:
                        tensor = tensor.to(torch.float16)
                    aggregated[key].append(tensor)

            for key, tensors in aggregated.items():
                # Mean, Median -> [N, C, ...] | Concat -> [N, C*M, ...]
                final_outputs.append((key, [t.shape[1] for t in tensors], self.combine(tensors)))

        return final_outputs


@config()
class Predictor(DistributedObject):
    """
    KonfAI's main prediction controller.

    This class orchestrates the prediction phase by:
    - Loading model weights from checkpoint(s) or URL(s)
    - Preparing datasets and output configurations
    - Managing distributed inference with optional multi-GPU support
    - Applying transformations and saving predictions
    - Optionally logging results to TensorBoard

    Attributes:
        model (Network): The neural network model to use for prediction.
        dataset (DataPrediction): Dataset manager for prediction data.
        combine_classpath (str): Path to the reduction strategy (e.g., "Mean").
        autocast (bool): Whether to enable AMP inference.
        outputs_dataset (dict[str, OutputDataset]): Mapping from layer names to output writers.
        data_log (list[str] | None): List of tensors to log during inference.
    """

    def __init__(
        self,
        model: ModelLoader = ModelLoader(),
        dataset: DataPrediction = DataPrediction(),
        combine: str = "Mean",
        train_name: str = "name",
        manual_seed: int | None = None,
        gpu_checkpoints: list[str] | None = None,
        autocast: bool = False,
        outputs_dataset: dict[str, OutputDatasetLoader] | None = {"default|Default": OutputDatasetLoader()},
        data_log: list[str] | None = None,
    ) -> None:
        if os.environ["KONFAI_CONFIG_MODE"] != "Done":
            raise ConfigError("Predictor requires KONFAI_CONFIG_MODE='Done' before initialization.")
        super().__init__(train_name)
        self.manual_seed = manual_seed
        self.dataset = dataset
        # Auto-patching (VRAM): a per-axis 0 in the user's patch_size marks a FREE axis and opts into
        # the OOM restart loop -- captured before any re-plan materialises concrete sizes over it.
        patch = dataset.patch
        self._vram_patch_template: list[int] | None = (
            [int(size) for size in patch.patch_size]
            if patch is not None and patch.patch_size is not None and any(size == 0 for size in patch.patch_size)
            else None
        )
        self._vram_patch_candidate: list[int] | None = None
        # Per-axis input multiple the model needs (its downsampling factor); a free axis is sized to it.
        self._downsampling_factor: list[int] | None = None
        module, name = get_module(combine, "konfai.predictor")
        if module.__name__ == "konfai.predictor":
            self.combine = getattr(module, name)()
        else:
            self.combine = apply_config(f"{konfai_root()}.{combine}")(getattr(module, name))()

        self.autocast = autocast
        self.model = model.get_model(train=False)
        self.it = 0
        self.outputs_dataset_loader = outputs_dataset if outputs_dataset else {}
        self.outputs_dataset = {
            name.replace(":", "."): value.get_output_dataset(name)
            for name, value in self.outputs_dataset_loader.items()
        }

        self.datasets_filename = []
        self.predict_path = predictions_directory() / self.name
        for output_dataset in self.outputs_dataset.values():
            self.datasets_filename.append(output_dataset.filename)
            # Rebase under the run directory, re-deriving is_directory: a bare string + "/" would flag an
            # h5 output as a directory and write the hidden dotfile Predictions/<run>/Dataset/.h5.
            output_dataset.rebase(self.predict_path)
        self.data_log = data_log
        modules = []
        for i, _ in self.model.named_modules():
            modules.append(i)
        if self.data_log is not None:
            for k in self.data_log:
                tmp = k.split("/")[0].replace(":", ".")
                if tmp not in self.dataset.get_groups_dest() and tmp not in modules:
                    raise PredictorError(
                        f"Invalid key '{tmp}' in `data_log`.",
                        f"This key is neither a destination group from the dataset ({self.dataset.get_groups_dest()})",
                        f"nor a valid module name in the model ({modules}).",
                        "Please check your `data_log` configuration,"
                        " it should reference either a model output or a dataset group.",
                    )

        self.gpu_checkpoints = gpu_checkpoints
        # Cut the grids with the model's downsampling multiple already known, so each case's free axis
        # rounds up to a valid input size (the graph -- hence the factor -- is final before init()).
        self.dataset.set_free_axis_multiple(self.model.downsampling_factor())
        self.dataset.prepare()
        self.model.init(self.autocast, State.PREDICTION, self.dataset.get_groups_dest())
        self.model.init_outputs_group()
        self.model._compute_channels_trace(self.model, self.model.in_channels, None, self.gpu_checkpoints)
        # The per-axis multiple a free patch axis rounds up to, read off the model's downsampling graph.
        self._downsampling_factor = self.model.downsampling_factor()
        self.output_modules = [name for name, _, _ in self.model.named_module_args_dict()]

        for output_group in self.outputs_dataset.keys():
            if output_group.replace(";accu;", "") not in self.output_modules:
                raise PredictorError(
                    f"The output group '{output_group}' defined in 'outputs_criterions' "
                    "does not correspond to any module in the model.",
                    f"Available modules: {self.output_modules}",
                    "Please check that the name matches exactly a submodule or output of your model architecture.",
                )

        dataset_groups = {
            group_src: list(groups_dest.keys()) for group_src, groups_dest in self.dataset.groups_src.items()
        }

        for name, output_dataset in self.outputs_dataset.items():
            output_dataset.prepare(name.replace(".", ":"))
            output_dataset.setup(
                list(self.dataset.datasets.values()),
                dataset_groups,
            )

        if len(self.outputs_dataset) == 0 and not any(
            network.measure is not None for network in self.model.get_networks().values()
        ):
            raise PredictorError(
                "No prediction outputs or runtime measures are configured.",
                "Define at least one outputs_dataset entry or enable a network measure.",
            )

    def setup(self, world_size: int):
        """
        Set up the predictor for inference.

        This method performs all necessary initialization steps before running predictions:
        - Ensures output directories exist, and optionally prompts the user before overwriting existing predictions.
        - Copies the current configuration file (Prediction.yml) into the output directory for reproducibility.
        - Dynamically loads pretrained weights from local files or remote URLs.
        - Wraps the base model into a `ModelComposite` to support ensemble inference.
        - Initializes the prediction dataloader, with proper distribution across available GPUs.

        Args:
            world_size (int): Total number of processes or GPUs used for distributed prediction.

        """
        for dataset_filename in self.datasets_filename:
            path = self.predict_path / dataset_filename
            if os.path.exists(path) and len(list(Path(path).rglob("*.yml"))):
                confirm_overwrite_or_raise(path, "prediction", PredictorError)

            if not os.path.exists(path):
                os.makedirs(path)

        shutil.copyfile(config_file(), self.predict_path / "Prediction.yml")

        self.model_composite = ModelComposite(self.model, self.combine)
        if not self.path_to_models and any(parameter.numel() for parameter in self.model.parameters()):
            # A model WITH weights but no checkpoint would run with random weights and silently produce
            # garbage -- refuse it. A WEIGHTLESS model (0 parameters, e.g. a classical/optimisation engine
            # such as registration) is legitimate with no checkpoint: it is run once as constructed.
            raise PredictorError(
                "No model checkpoint available for prediction.",
                "This model has trainable weights, so at least one '.pt' checkpoint must be provided (for "
                "KonfAI Apps, declare it via the 'models' field in app.json).",
                "Without a checkpoint its weights are random and prediction would silently produce garbage.",
            )
        self.model_composite.load(self._load())

        self.size = len(self.gpu_checkpoints) + 1 if self.gpu_checkpoints else 1

        self.dataloader, _, _ = self.dataset.get_data(world_size // self.size)

    def set_models(self, path_to_models: list[Path | str]) -> None:
        self.path_to_models = path_to_models

    def _load(self) -> list[dict[str, Any] | Path | str]:
        """
        Resolve checkpoint sources for ensemble prediction.

        This method handles both remote and local model sources:
        - If the model path is a URL (starting with "https://"), it eagerly downloads and loads the state dict
          once because re-fetching it every batch would be prohibitively slow.
        - If the model path is local:
            - it keeps only the checkpoint path and lets `ModelComposite` stream weights into a single model
              instance during prediction to reduce memory pressure.

        Returns:
            list[dict[str, dict[str, torch.Tensor]] | Path | str]: A list of checkpoint sources, one per model.

        Raises:
            Exception: If a model path does not exist or cannot be loaded.
        """
        state_dicts = []
        for path_to_model in self.path_to_models:
            if isinstance(path_to_model, str) and path_to_model.startswith("https://"):
                try:
                    state_dicts.append(
                        torch.hub.load_state_dict_from_url(url=path_to_model, map_location="cpu", check_hash=True)
                    )
                except Exception as exc:
                    raise Exception(f"Model : {path_to_model} does not exist !") from exc
            elif Path(path_to_model).exists():
                state_dicts.append(Path(path_to_model))
            else:
                raise ValueError(f"Invalid model path entry: {path_to_model}")
        return state_dicts

    def run_process(
        self,
        world_size: int,
        global_rank: int,
        local_rank: int,
        dataloaders: list[DataLoader],
    ):
        """
        Launch prediction on the given process rank.

        Args:
            world_size (int): Number of model replicas sharding the data -- the spawned process count
                already divided by the model-parallel size (``gpu_checkpoints``), NOT the GPU count.
            global_rank (int): Rank of the current process.
            local_rank (int): Local device rank.
            dataloaders (list[DataLoader]): List of data loaders for prediction.
        """

        model_composite = (
            Network.to(self.model_composite, local_rank * self.size)
            if len(cuda_visible_devices())
            else self.model_composite
        )
        if len(cuda_visible_devices()):
            # Co-locate the output writers with the model so their reduction/transforms know the GPU.
            for output_dataset in self.outputs_dataset.values():
                output_dataset.to(local_rank * self.size)
        model_composite = Model(model_composite)
        device = local_rank * self.size if len(cuda_visible_devices()) else None
        dataloader = dataloaders[0]
        # Round a free patch axis up to the model's valid input multiple before the first attempt, so
        # the network's encoder/decoder skips align instead of crashing on a non-divisible extent (the
        # border padding fills the round-up, cropped back after the forward). A whole-axis extent still
        # too large for VRAM OOMs into the shrink loop below, which keeps the size valid too.
        if self._vram_patch_candidate is None:
            sized = size_free_axes(
                self._vram_patch_template, self.dataset.worst_case_shape(), self._downsampling_factor
            )
            if sized is not None:
                self._vram_patch_candidate = sized
                self.dataset.replan_patch(sized)
                dataloader = self.dataset.get_data(world_size)[0][global_rank][0]
        while True:
            try:
                with _Predictor(
                    world_size,
                    global_rank,
                    local_rank,
                    self.autocast,
                    self.predict_path,
                    self.data_log,
                    self.outputs_dataset,
                    model_composite,
                    dataloader,
                ) as p:
                    p.run()
                return
            except torch.cuda.OutOfMemoryError:
                # The restart loop IS the sizing iteration (no probe phase): the run that just OOMed
                # already measured the step's transient for free. Read it BEFORE the reset (the peak
                # still includes the resident accumulators on both sides of the difference), free the
                # in-flight state -- open streamed sinks abort and remove their partial entries, so a
                # reader never sees a half-written volume even when the OOM is fatal -- then read the
                # honest free VRAM.
                measured = self._transient_at_oom(device)
                for output_dataset in self.outputs_dataset.values():
                    output_dataset.reset()
                if self._vram_patch_template is None:
                    raise  # no free axis declared: not auto-patched
                candidate = self._shrunken_patch(measured, device)
                if candidate is None:
                    raise
                self._reset_cuda_peak(device)
                print(
                    f"[KonfAI] VRAM: rank {global_rank} ran out of memory -> "
                    f"re-planning the free patch axes to {candidate} and restarting this rank's cases."
                )
                self._vram_patch_candidate = candidate
                self.dataset.replan_patch(candidate)
                dataloader = self.dataset.get_data(world_size)[0][global_rank][0]

    def _shrunken_patch(self, measured: int | None, device: int | None) -> list[int] | None:
        """One shrink step for the free patch axes after a CUDA OOM (``None`` = not auto, or floor).

        The first OOM starts from the worst prepared case at full extent (the size the failed grid
        effectively ran); later ones shrink the current candidate further. When the framework picks
        the size, it must also leave the blend on the GPU: the accumulation footprint is RESERVED
        beside the forward, so the sized patch passes the accumulation gate. Only when that reserve
        fits at no size (or cannot be priced) is the forward sized alone -- the gate's memory-safe
        CPU blend absorbs that case.
        """
        if self._vram_patch_template is None:
            return None
        worst = self.dataset.worst_case_shape()
        if worst is None:
            return None
        candidate = self._vram_patch_candidate or concretize_patch_size(
            self._vram_patch_template, worst, self._downsampling_factor
        )
        usable = self._usable_vram_after_oom(device)
        reserve = self._accumulation_reserve(candidate, worst)
        snap = self._downsampling_factor
        if reserve is not None:
            shrunk = next_patch_candidate(candidate, self._vram_patch_template, worst, measured, usable - reserve, snap)
            if shrunk is not None:
                return shrunk
        return next_patch_candidate(candidate, self._vram_patch_template, worst, measured, usable, snap)

    def _accumulation_reserve(self, candidate: list[int], worst: list[int]) -> float | None:
        """Bytes each case keeps resident while its patches accumulate, per output writer: the
        streamed window (one patch extent x the cross-section) when the writer will stream --
        single augmentation, voxel-local reduction -- the assembled volume otherwise. ``None``
        when a writer's channels cannot be read off the model trace (no reserve, gate decides).
        """
        trace = {name: args.out_channels for name, _, args in self.model.named_module_args_dict()}
        elem = 2  # ModelComposite casts float32 outputs to float16 before accumulation
        reserve = 0.0
        for name, writer in self.outputs_dataset.items():
            out_channels = trace.get(name.replace(";accu;", ""))
            if not out_channels:
                return None
            if isinstance(self.combine, Concat):
                out_channels *= max(1, len(self.path_to_models))
            nb_augmentation = max(1, writer.nb_data_augmentation)
            streams = nb_augmentation == 1 and writer.reduction.voxel_local
            voxels = candidate[0] * np.prod(worst[1:], dtype=np.int64) if streams else np.prod(worst, dtype=np.int64)
            reserve += float((out_channels + 1) * voxels * elem * nb_augmentation)
        return reserve

    @staticmethod
    def _reset_cuda_peak(device: int | None) -> None:
        """Drop the failed attempt's high-water mark so the rerun measures its own steps.

        ``max_memory_allocated`` only rises: left in place, the full-extent attempt's peak would
        overstate every later transient -- a second shrink would overshoot, and the accumulation
        gate would keep the rerun's blend on the CPU.
        """
        if device is None:
            return
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:  # nosec B110 - stale stats only cost precision, never correctness
            pass

    def _transient_at_oom(self, device: int | None) -> int | None:
        """The failed step's measured transient (CUDA peak over resident), ``None`` when unreadable."""
        if device is None:
            return None
        try:
            transient = int(torch.cuda.max_memory_allocated(device) - torch.cuda.memory_allocated(device))
        except Exception:  # nosec B110 - an unreadable measurement just falls back to the fixed step
            return None
        return transient if transient > 0 else None

    def _usable_vram_after_oom(self, device: int | None) -> float:
        """The VRAM budget the next attempt's step may claim, read once the failed state is freed."""
        if device is None:
            return 0.0
        try:
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(device)
        except Exception:  # nosec B110 - an unreadable budget refuses the restart (the OOM re-raises)
            return 0.0
        return usable_vram(free)

    def __str__(self) -> str:
        params = {
            "model": self.model,
            "dataset": self.dataset,
            "combine": self.combine,
            "train_name": self.name,
            "manual_seed": self.manual_seed,
            "gpu_checkpoints": self.gpu_checkpoints,
            "autocast": self.autocast,
            "outputs_dataset": self.outputs_dataset,
            "data_log": self.data_log,
        }
        return str(params)

    def __repr__(self) -> str:
        return str(self)


def build_predict(
    models: list[Path],
    prediction_file: Path | str = Path("./Prediction.yml").resolve(),
    predictions_dir: Path | str = Path("./Predictions").resolve(),
) -> DistributedObject:
    """
    Build and return the configured prediction workflow without executing it.

    Parameters
    ----------
    models : list[Path]
        One or more checkpoint files to load for prediction.
    prediction_file : Path | str, optional
        Prediction configuration file.
    predictions_dir : Path | str, optional
        Directory where prediction outputs are written.

    Returns
    -------
    DistributedObject
        Configured predictor object ready to be executed by the runtime wrapper.
    """
    configure_workflow_environment(
        config_path=prediction_file,
        root="Predictor",
        state=State.PREDICTION,
        path_env={"KONFAI_PREDICTIONS_DIRECTORY": predictions_dir},
    )
    os.environ["KONFAI_CONFIG_MODE"] = "Done"
    predictor = apply_config()(Predictor)()
    predictor.set_models(models)
    return predictor


@run_distributed_app
def predict(
    models: list[Path],
    overwrite: bool = False,
    gpu: list[int] | None = cuda_visible_devices(),
    cpu: int = 1,
    quiet: bool = False,
    tensorboard: bool = False,
    prediction_file: Path | str = Path("./Prediction.yml").resolve(),
    predictions_dir: Path | str = Path("./Predictions").resolve(),
) -> DistributedObject:
    """
    Build and execute the configured prediction workflow.

    This compatibility wrapper preserves the historical CLI-facing API while
    delegating the pure build step to :func:`build_predict`.
    """
    del overwrite, gpu, cpu, quiet, tensorboard
    return build_predict(
        models=models,
        prediction_file=prediction_file,
        predictions_dir=predictions_dir,
    )

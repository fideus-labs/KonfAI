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
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
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
from konfai.data.data_manager import BatchSample, DataPrediction, DatasetIter
from konfai.data.patching import Accumulator, PathCombine, StreamingAccumulator
from konfai.data.transform import LocalityKind, Transform, TransformInverse, TransformLoader
from konfai.network.network import Model, ModelLoader, NetState, Network
from konfai.utils.config import apply_config, config
from konfai.utils.dataset import Attribute, Dataset, DataStream
from konfai.utils.errors import ConfigError, PredictorError
from konfai.utils.runtime import (
    DataLog,
    DistributedObject,
    NeedDevice,
    State,
    configure_workflow_environment,
    confirm_overwrite_or_raise,
    description,
    run_distributed_app,
    safe_torch_load,
)
from konfai.utils.utils import get_module, split_path_spec


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
        overlap: int | None,
        nb_data_augmentation: int,
    ) -> None:
        if patch_size is not None and overlap is not None:
            if self.patch_combine is not None:
                self.patch_combine.set_patch_config(patch_size, overlap)
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
        return len(self.output_layer_accumulator.get(index, {})) == self.nb_data_augmentation and all(
            acc.is_full() for acc in self.output_layer_accumulator[index].values()
        )

    @abstractmethod
    def get_output(self, index: int, number_of_channels_per_model: list[int], dataset: DatasetIter) -> torch.Tensor:
        raise NotImplementedError()

    def write_prediction(self, index: int, name: str, layer: torch.Tensor) -> None:
        super().write(self.group, name, layer.detach().cpu().numpy(), self.attributes[index][0][0])
        self.attributes.pop(index)

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
        # byte-identical to the assembled path (``_stream_refusal is None`` -- pointwise finalize, single
        # augmentation, region-writable backend), finalizing and writing each z-slab as its patches
        # complete so RAM is bounded at one patch window; otherwise the whole-volume path is used
        # transparently. ``KONFAI_STREAMED_WRITES=0`` is a global ops/debug kill-switch (also how a test
        # gets the assembled reference), not a per-output option.
        self._streaming_enabled = os.environ.get("KONFAI_STREAMED_WRITES", "1").lower() not in ("0", "false")
        self._stream_active: dict[int, bool] = {}
        self._stream_sinks: dict[int, DataStream] = {}

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
                # Stream this case iff enabled and byte-identical to the assembled path. No warning on a
                # refusal: streaming is transparent, so the whole-volume path is a normal outcome, not a
                # misconfiguration.
                self._stream_active[index_dataset] = (
                    self._streaming_enabled and self._stream_refusal(dataset, source_attribute) is None
                )
            self.attributes[index_dataset][index_augmentation] = {}

            accumulator_type = StreamingAccumulator if self._stream_active[index_dataset] else Accumulator
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
        if not self._stream_active.get(index_dataset, False):
            return
        finished = accumulator.is_full()
        if finished:
            slabs = slabs + cast(StreamingAccumulator, accumulator).finalize()
        try:
            self._flush_slabs(index_dataset, slabs, number_of_channels_per_model, dataset)
        except BaseException as error:
            sink = self._stream_sinks.pop(index_dataset, None)
            if sink is not None:
                sink.__exit__(type(error), error, error.__traceback__)
            raise
        if finished:
            self._close_stream(index_dataset)

    def _stream_refusal(self, dataset: DatasetIter, attribute: Attribute) -> str | None:
        """Why this case cannot be written slab by slab (``None`` = it can).

        Streaming is only byte-identical to the assembled-volume path when every finalize stage is
        voxel-local: single augmentation (TTA inverses apply to the assembled volume), a Mean/Median
        reduction, and a POINTWISE ``patch_locality`` for every transform of the finalize chain — a
        pointwise transform is a per-voxel value map, so its inverse is voxel-local too. Anything else
        falls back to the proven whole-volume path.
        """
        if self.nb_data_augmentation != 1:
            return "test-time augmentation inverses apply to the assembled volume"
        # The reduction declares its own slab-locality (like a transform declares patch_locality) rather
        # than the gate hardcoding a whitelist: any voxel-local reduction streams, an unknown one does not.
        if not self.reduction.voxel_local:
            return f"reduction '{type(self.reduction).__name__}' is not voxel-local"
        chain: list[Transform] = [
            *self.before_reduction_transforms,
            *self.after_reduction_transforms,
            *[
                transform
                for transform in dataset.groups_src[self.group_src][self.group_dest].transforms
                if isinstance(transform, TransformInverse) and transform.apply_inverse
            ],
            *self.final_transforms,
        ]
        for transform in chain:
            if transform.patch_locality(Attribute(attribute)).kind is not LocalityKind.POINTWISE:
                return f"transform '{type(transform).__name__}' is not pointwise"
        if not self.can_stream_data(attribute):
            return f"write format '{self.file_format}' cannot serve region writes"
        return None

    def _flush_slabs(
        self,
        index: int,
        slabs: list[tuple[slice, torch.Tensor]],
        number_of_channels_per_model: list[int] | None,
        dataset: DatasetIter,
    ) -> None:
        """Finalize each slab and write it into the case's sink (opened at the first slab, once the
        finalize chain has fixed the output's channel count and dtype)."""
        for region, slab in slabs:
            result, attribute = self._finalize_slab(index, slab, number_of_channels_per_model, dataset)
            array = result.detach().cpu().numpy()
            if index not in self._stream_sinks:
                spatial = self.output_layer_accumulator[index][0].shape
                sink = self.open_data_stream(
                    self.group, self.names[index], [array.shape[0], *spatial], array.dtype, attribute
                )
                if sink is None:
                    raise PredictorError(
                        f"Streamed write refused by the '{self.file_format}' backend for dtype"
                        f" '{array.dtype}' on output '{self.group}': write it to an h5 or omezarr"
                        f" dataset, or set KONFAI_STREAMED_WRITES=0 to force the whole-volume path."
                    )
                self._stream_sinks[index] = sink
            self._stream_sinks[index].write_slice(
                (slice(0, array.shape[0]), region, *(slice(0, extent) for extent in array.shape[2:])),
                array,
            )

    def _finalize_slab(
        self,
        index: int,
        layer: torch.Tensor,
        number_of_channels_per_model: list[int] | None,
        dataset: DatasetIter,
    ) -> tuple[torch.Tensor, Attribute]:
        """The single-augmentation finalize chain of ``_get_output``/``get_output``, on one z-slab.

        Every stage passed ``_stream_refusal`` as voxel-local, so applying it slab by slab yields the
        same voxels as applying it to the assembled volume. Each slab gets its own copy of the case
        attribute (transform writes stay slab-local).
        """
        attribute = Attribute(self.attributes[index][0][0])
        if number_of_channels_per_model and layer.shape[0] == sum(number_of_channels_per_model):
            attribute["number_of_channels_per_model_0"] = torch.tensor(number_of_channels_per_model)
            chunks = list(torch.split(layer, number_of_channels_per_model, dim=0))
        else:
            chunks = [layer]
        results = []
        for chunk in chunks:
            for transform in self.before_reduction_transforms:
                chunk = transform(self.names[index], chunk, Attribute(attribute))
            results.append(chunk)
        # Reduce, then drop the singleton stack axis; Mean/Median also drop the singleton model axis,
        # while Concat keeps the [M, C, ...] model axis for after_reduction (Sum) to merge into labels.
        result = self.reduction([torch.stack(results, dim=0).unsqueeze(0)]).squeeze(0)
        if isinstance(self.reduction, Mean | Median):
            result = result.squeeze(0)
        for transform in self.after_reduction_transforms:
            result = transform(self.names[index], result, attribute)
        for transform in reversed(dataset.groups_src[self.group_src][self.group_dest].transforms):
            if isinstance(transform, TransformInverse) and transform.apply_inverse:
                result = transform.inverse(self.names[index], result, attribute)
        for transform in self.final_transforms:
            result = transform(self.names[index], result, attribute)
        return result, attribute

    def _close_stream(self, index: int) -> None:
        """Finalize the case's sink and drop its bookkeeping (``is_done`` then reports nothing left)."""
        sink = self._stream_sinks.pop(index, None)
        if sink is not None:
            sink.__exit__(None, None, None)
        self._stream_active.pop(index, None)
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

    def _get_output(
        self, index: int, index_augmentation: int, number_of_channels_per_model: list[int], dataset: DatasetIter
    ) -> torch.Tensor:
        layer = self.output_layer_accumulator[index][index_augmentation].assemble()  # if concat then [N*C] else [C]

        if index_augmentation > 0:
            i = 0
            index_augmentation_tmp = index_augmentation - 1
            for data_augmentations in dataset.data_augmentations_list:
                if index_augmentation_tmp >= i and index_augmentation_tmp < i + data_augmentations.nb:
                    for data_augmentation in reversed(data_augmentations.data_augmentations):
                        layer = data_augmentation.inverse(index, index_augmentation_tmp - i, layer)
                    break
                i += data_augmentations.nb

        base_attr = self.attributes[index][index_augmentation][0]
        if layer.shape[0] == sum(number_of_channels_per_model):
            base_attr["number_of_channels_per_model_0"] = torch.tensor(number_of_channels_per_model)
            chunks = list(torch.split(layer, number_of_channels_per_model, dim=0))
        else:
            chunks = [layer]

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
        # The per-case device decisions above make the list single-device; if VRAM pressure ever mixes
        # devices anyway, fall back to the host — the one device guaranteed to fit everything.
        if len({r.device for r in results}) > 1:
            results = [r.cpu() if r.device.type != "cpu" else r for r in results]
        result = self.reduction(results).squeeze(0)
        if isinstance(self.reduction, Mean) or isinstance(self.reduction, Median):
            result = result.squeeze(0)
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
        if self.tb is None:
            return

        measures: dict[str, tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]] = {}
        if self._has_runtime_measures:
            measures = DistributedObject.get_measure(
                self.world_size,
                self.global_rank,
                self.local_rank,
                {"": self.model_composite.module},
                1,
            )

        if self.global_rank != 0:
            return

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

    def load(self, state_sources: list[dict[str, Any] | Path | str]):
        """
        Load weights for each sub-model in the composite from the corresponding state dictionaries.

        Args:
            state_sources (list): One checkpoint source per model replica.
        """
        self._state_sources = state_sources
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
        if not self._state_sources:
            return final_outputs
        if isinstance(self.combine, Mean):
            sum_acc: dict[str, torch.Tensor] = {}
            count: dict[str, int] = defaultdict(int)
            channels: dict[str, list[int]] = defaultdict(list)
            for model_index in range(len(self._state_sources)):
                for key, tensor in self._ensure_model_loaded(model_index)(data_dict, output_layers):
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
            for model_index in range(len(self._state_sources)):
                for key, tensor in self._ensure_model_loaded(model_index)(data_dict, output_layers):
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
            output_dataset.filename = str(self.predict_path / output_dataset.filename) + "/"
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
        self.dataset.prepare()
        self.model.init(self.autocast, State.PREDICTION, self.dataset.get_groups_dest())
        self.model.init_outputs_group()
        self.model._compute_channels_trace(self.model, self.model.in_channels, None, self.gpu_checkpoints)
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
        if not self.path_to_models:
            raise PredictorError(
                "No model checkpoint available for prediction.",
                "At least one '.pt' checkpoint must be provided (for KonfAI Apps, declare it via the "
                "'models' field in app.json).",
                "Without a checkpoint the model is never executed and prediction would silently produce no output.",
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
            world_size (int): Total number of processes.
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
        with _Predictor(
            world_size,
            global_rank,
            local_rank,
            self.autocast,
            self.predict_path,
            self.data_log,
            self.outputs_dataset,
            model_composite,
            *dataloaders,
        ) as p:
            p.run()

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

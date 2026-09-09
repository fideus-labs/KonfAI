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


"""Label and channel transforms: masks, dilation, argmax/softmax, label merging and selection, one-hot."""

import contextlib
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from konfai.data.transform.base import LocalityKind, PatchLocality, RegionContext, Transform, TransformInverse, sitk
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import TransformError
from konfai.utils.ITK import _require_simpleitk


class Mask(Transform):
    """Set everything outside a mask to a constant.

    Per-voxel: a region needs only the part of the mask that lines up with it, which the dispatcher
    hands over. The mask is assumed aligned to the volume.
    """

    def __init__(self, path: str = "./default.mha", value_outside: int = 0) -> None:
        super().__init__()
        self.path = path
        self.value_outside = value_outside
        self._cached_mask: torch.Tensor | None = None
        #: Cases whose stored mask was checked against the chain input's extent (once per case).
        self._aligned: set[str] = set()

    # POINTWISE on the promise that the mask sits on the stage's input grid, checked in
    # stream_region once per case.
    locality = LocalityKind.POINTWISE

    def _apply(self, tensor: torch.Tensor, mask: torch.Tensor | np.ndarray) -> torch.Tensor:
        # Index on the tensor's own device so the mask works whether the volume is on CPU or GPU.
        tensor[torch.as_tensor(mask, device=tensor.device) == 0] = self.value_outside
        return tensor

    def _cached_mha(self) -> torch.Tensor:
        """The whole ``.mha`` mask, read once, for the whole-volume path only."""
        _require_simpleitk()
        if self._cached_mask is None:
            self._cached_mask = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(self.path))).unsqueeze(0)
        return self._cached_mask

    def _mha_extent(self) -> tuple[int, ...]:
        """The ``.mha`` mask's spatial extent, channel-first order, from its header alone."""
        _require_simpleitk()
        reader = sitk.ImageFileReader()
        reader.SetFileName(self.path)
        reader.ReadImageInformation()
        return tuple(int(extent) for extent in reversed(reader.GetSize()))

    def _mha_region(self, slices: tuple[slice, ...]) -> torch.Tensor:
        """One region of the ``.mha`` mask; the whole mask is never held on the streamed path."""
        _require_simpleitk()
        reader = sitk.ImageFileReader()
        reader.SetFileName(self.path)
        reader.ReadImageInformation()
        spatial = list(slices[1:])
        reader.SetExtractIndex([int(part.start or 0) for part in reversed(spatial)])
        reader.SetExtractSize([int(part.stop - (part.start or 0)) for part in reversed(spatial)])
        return torch.as_tensor(sitk.GetArrayFromImage(reader.Execute())).unsqueeze(0)

    def _mask_dataset(self, name: str) -> Dataset:
        """The dataset holding the case's mask group."""
        for dataset in self.datasets:
            if dataset.is_dataset_exist(self.path, name):
                return dataset
        raise TransformError(f"'Mask' found no mask '{self.path}' for case '{name}' in any dataset.")

    def _mask(self, name: str, slices: tuple[slice, ...] | None) -> torch.Tensor | np.ndarray:
        """The case's mask, or the ``slices`` region of it."""
        if self.path.endswith(".mha"):
            return self._cached_mha() if slices is None else self._mha_region(slices)
        dataset = self._mask_dataset(name)
        if slices is None:
            return dataset.read_data(self.path, name)[0]
        return dataset.read_data_slice(self.path, name, slices)[0]

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return self._apply(tensor, self._mask(name, None))

    def _check_aligned(self, name: str, context: RegionContext) -> None:
        """Refuse a mask whose extent is not the stage input's. Headers only, once per case."""
        if name in self._aligned:
            return
        expected = tuple(int(extent) for extent in context.source_shape)
        if self.path.endswith(".mha"):
            stored = self._mha_extent()
        else:
            stored = tuple(int(extent) for extent in self._mask_dataset(name).get_infos(self.path, name)[0][1:])
        if stored != expected:
            raise TransformError(
                f"'Mask' reads '{self.path}' for case '{name}' region by region, but the mask's extent"
                f" {list(stored)} is not the stage input's {list(expected)}.",
                "A streamed Mask needs a mask on the grid it is applied to: resample the mask onto it"
                " first, or apply the Mask before the stages that change the grid.",
            )
        self._aligned.add(name)

    @staticmethod
    def _window(context: RegionContext) -> tuple[slice, ...]:
        """The part of the mask a region reads: every channel, where the region sits in the input."""
        return (slice(None), *context.source)

    def plan_region_reads(self, name: str, contexts: Sequence[RegionContext]) -> None:
        # A missing mask is the first region's error to raise. SimpleITK plans nothing for a .mha.
        if self.path.endswith(".mha"):
            return
        with contextlib.suppress(TransformError):
            self._mask_dataset(name).plan_region_reads(self.path, name, [self._window(c) for c in contexts])

    def stream_region(
        self,
        name: str,
        tensor: torch.Tensor,
        context: RegionContext,
        cache_attribute: Attribute,
    ) -> torch.Tensor:
        # Only the region's part of the mask, checked to sit on the stage input's grid first.
        self._check_aligned(name, context)
        return self._apply(tensor, self._mask(name, self._window(context)))


class Dilate(Transform):
    working_multiple = 15.0

    def __init__(self, dilate: int = 1) -> None:
        super().__init__()
        if dilate < 0:
            raise ValueError(f"[Dilate] 'dilate' must be >= 0, got {dilate}")
        self.dilate = dilate

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # A box dilation spreads foreground by at most ``dilate`` voxels per axis: a bounded halo.
        if self.dilate == 0:
            return PatchLocality(LocalityKind.POINTWISE)
        return PatchLocality(LocalityKind.HALO, halo=(self.dilate,))

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self.dilate == 0:
            return tensor

        data = (tensor > 0).to(torch.float32)
        spatial_dims = data.dim() - 1
        d = self.dilate
        k = 2 * d + 1

        # A box structuring element is separable: a k**n box equals n successive 1-D max-pools.
        if spatial_dims == 2:
            data = F.max_pool2d(data, kernel_size=(k, 1), stride=1, padding=(d, 0))
            data = F.max_pool2d(data, kernel_size=(1, k), stride=1, padding=(0, d))
        elif spatial_dims == 3:
            data = F.max_pool3d(data, kernel_size=(k, 1, 1), stride=1, padding=(d, 0, 0))
            data = F.max_pool3d(data, kernel_size=(1, k, 1), stride=1, padding=(0, d, 0))
            data = F.max_pool3d(data, kernel_size=(1, 1, k), stride=1, padding=(0, 0, d))
        else:
            raise ValueError(
                "[Dilate] Unsupported tensor shape for "
                f"'{name}': expected [C,H,W] or [C,D,H,W], got {list(tensor.shape)}"
            )

        return data.to(tensor.dtype)


def _forget_model_channel_counts(cache_attribute: Attribute) -> None:
    """Take ``number_of_channels_per_model`` off a case's state, as folding the model axis does.

    A later ``Sum`` or ``MergeLabels`` reading it off the written store would take the ensemble
    branch on a one-channel map, and a streamed region pops it from a scope that is thrown away.
    """
    if "number_of_channels_per_model" in cache_attribute:
        cache_attribute.pop_tensor("number_of_channels_per_model")


class Sum(Transform):
    working_multiple = 3.75

    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return _axis_reduction_locality(self.dim)

    def write_stream_cache_attribute(
        self, cache_attribute: Attribute, source_spatial_shape: list[int], name: str = ""
    ) -> None:
        del source_spatial_shape, name
        _forget_model_channel_counts(cache_attribute)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "number_of_channels_per_model" in cache_attribute:
            number_of_channels = cache_attribute.pop_tensor("number_of_channels_per_model")
            result = tensor[0]
            for i, t in enumerate(tensor[1:]):
                t[t != 0] += int(number_of_channels[i]) - 1
                result += t
            return result
        else:
            return torch.sum(tensor, dim=self.dim).to(tensor.dtype)


class MergeLabels(Transform):
    """Merge the per-model argmax label maps of a ``combine: Concat`` ensemble into one global map.

    Each model's ``Argmax`` produces a local class index (``0`` = background). A model's
    non-background labels are shifted past every earlier model's foreground classes, by the
    cumulative sum of the earlier models' foreground counts (``nb_class - 1``), so the models'
    disjoint label ranges tile a single global label space. Use it when the models segment different
    structures. Requires ``number_of_channels_per_model`` in the attribute, written by the ``Concat``
    reduction. A voxel claimed by several models takes the label of the last model in ensemble order.
    """

    working_multiple = 3.75

    locality = LocalityKind.POINTWISE

    def write_stream_cache_attribute(
        self, cache_attribute: Attribute, source_spatial_shape: list[int], name: str = ""
    ) -> None:
        del source_spatial_shape, name
        _forget_model_channel_counts(cache_attribute)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "number_of_channels_per_model" not in cache_attribute:
            raise TransformError(
                "MergeLabels expects a multi-model 'combine: Concat' output: "
                "'number_of_channels_per_model' is missing from the attribute.",
            )
        number_of_channels = cache_attribute.pop_tensor("number_of_channels_per_model")
        result = tensor[0].clone()
        offset = int(number_of_channels[0]) - 1
        for i, t in enumerate(tensor[1:]):
            foreground = t != 0
            result[foreground] = (t[foreground] + offset).to(result.dtype)
            offset += int(number_of_channels[i + 1]) - 1
        return result


def _axis_reduction_locality(dim: int) -> PatchLocality:
    """POINTWISE over the channel axis (dim 0); over a spatial axis the stage takes the whole volume."""
    if dim == 0:
        return PatchLocality(LocalityKind.POINTWISE)
    return PatchLocality(
        LocalityKind.WHOLE_VOLUME,
        reason=f"dim {dim} reduces a spatial axis, which spans the whole extent; dim: 0 reduces the channels and streams",
    )


class Argmax(Transform):
    working_multiple = 0.0

    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return _axis_reduction_locality(self.dim)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return torch.argmax(tensor, dim=self.dim).unsqueeze(self.dim)


class Softmax(Transform):
    # 0.00 on a float input, 1.00 on the int16 a store serves, which widens before the kernel runs.
    working_multiple = 1.0

    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return _axis_reduction_locality(self.dim)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # torch.softmax has no integer kernel, and a store serves int16.
        return torch.softmax(tensor if tensor.is_floating_point() else tensor.float(), dim=self.dim)


class FlatLabel(Transform):
    working_multiple = 1.0

    locality = LocalityKind.POINTWISE

    def __init__(self, labels: list[int] | None = None) -> None:
        super().__init__()
        self.labels = labels

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # Filled through the mask, not through the indices of what it selects: see Clip.
        data = torch.zeros_like(tensor)
        if self.labels:
            for label in self.labels:
                data.masked_fill_(tensor == label, 1)
        else:
            data.masked_fill_(tensor > 0, 1)
        return data


class SelectLabel(Transform):
    """Relabel: each ``"(old,new)"`` pair maps label ``old`` to ``new``; every other voxel is 0."""

    working_multiple = 1.0

    locality = LocalityKind.POINTWISE

    def __init__(self, labels: list[str]) -> None:
        super().__init__()
        try:
            self.labels = [(int(old), int(new)) for old, new in (str(label).strip("()").split(",") for label in labels)]
        except ValueError:
            raise TransformError(
                f"'SelectLabel' cannot read labels={list(labels)!r}.",
                'labels is a list of "(old,new)" strings: labels: ["(1,2)", "(3,1)"].',
            ) from None

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        data = torch.zeros_like(tensor)
        for old_label, new_label in self.labels:
            data[tensor == old_label] = new_label
        return data


class OneHot(TransformInverse):
    # Half its own wider output, against the larger of input and output.
    working_multiple = 0.5

    def __init__(self, num_classes: int, inverse: bool = True) -> None:
        super().__init__(inverse)
        self.num_classes = num_classes

    locality = LocalityKind.POINTWISE

    def output_channels(self, channels: int) -> int:
        return self.num_classes * channels

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # Scattered straight into the float32 answer, where ``F.one_hot`` would build int64 first.
        labels = tensor.to(torch.int64)
        # scatter_ names no label, so an out-of-range one is checked here.
        lowest, highest = int(labels.min()), int(labels.max())
        if lowest < 0 or highest >= self.num_classes:
            raise TransformError(
                f"'OneHot' with num_classes={self.num_classes} met labels from {lowest} to {highest} in '{name}'.",
                "Labels must lie in [0, num_classes): raise num_classes or relabel (SelectLabel) first.",
            )
        result = torch.zeros(
            (int(tensor.shape[0]), self.num_classes, *tensor.shape[1:]), dtype=torch.float32, device=tensor.device
        )
        result.scatter_(1, labels.unsqueeze(1), 1.0)
        return result.squeeze(0)

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # Argmax the class axis (the one sized num_classes) and re-insert it, restoring a
        # [.., 1, *spatial] label map. A batched [B, num_classes, *spatial] is handled too.
        class_dim = 0 if tensor.shape[0] == self.num_classes else 1
        return torch.argmax(tensor, dim=class_dim).unsqueeze(class_dim)

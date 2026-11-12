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
        dataset = self.dataset_holding(self.path, name)
        if dataset is not None:
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
        mask = self._mask(name, None)
        if tuple(mask.shape) != tuple(tensor.shape[: len(mask.shape)]):
            raise TransformError(
                f"'Mask' reads '{self.path}' for case '{name}' with the shape {list(mask.shape)}, where the"
                f" tensor it masks has {list(tensor.shape)}.",
                "The mask must sit on the grid it is applied to: resample the mask onto it first, or apply"
                " the Mask before the stages that change the grid.",
            )
        return self._apply(tensor, mask)

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
            raise TransformError(f"'Dilate' was given dilate={dilate}.", "The radius in voxels is 0 or more.")
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
            raise TransformError(
                f"'Dilate' takes a [C, Y, X] or [C, Z, Y, X] tensor; case '{name}' is {list(tensor.shape)}."
            )

        return data.to(tensor.dtype)


def _forget_model_channel_counts(cache_attribute: Attribute) -> None:
    """Take ``number_of_channels_per_model`` off a case's state, as folding the model axis does.

    A later ``MergeLabels`` reading it off the written store would merge a one-channel map as an
    ensemble, and a streamed region pops it from a scope that is thrown away.
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

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return _reduced_shape(shape, self.dim, keep=False)

    def write_stream_cache_attribute(
        self, cache_attribute: Attribute, source_spatial_shape: list[int], name: str = ""
    ) -> None:
        del source_spatial_shape, name
        _forget_model_channel_counts(cache_attribute)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # The stack a Concat left is summed like any other: merging an ensemble's label maps is
        # MergeLabels' job. An integer sum is taken in its own dtype (the result is cast back to it
        # either way), not in the int64 torch promotes it to.
        _forget_model_channel_counts(cache_attribute)
        integral = not (tensor.is_floating_point() or tensor.is_complex() or tensor.dtype == torch.bool)
        return torch.sum(tensor, dim=self.dim, dtype=tensor.dtype if integral else None).to(tensor.dtype)


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


def _reduced_shape(shape: list[int], dim: int, keep: bool) -> list[int]:
    """The spatial shape a reduction over tensor axis ``dim`` of a channel-first tensor leaves, the axis
    kept at 1 or dropped. Over the channels it is the input's: the channel axis is not spatial."""
    axis = dim if dim >= 0 else dim + len(shape) + 1
    if axis == 0:
        return shape
    return [*shape[: axis - 1], *([1] if keep else []), *shape[axis:]]


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

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return _reduced_shape(shape, self.dim, keep=True)

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


def _components(foreground: np.ndarray) -> np.ndarray:
    """``foreground``'s components, face to face, numbered from 1 in the order of their first voxel in scan order,
    0 outside (ITK's ``ConnectedComponentImageFilter``)."""
    _require_simpleitk()
    labels = sitk.ConnectedComponent(sitk.GetImageFromArray(foreground.astype(np.uint8)), False)
    return sitk.GetArrayViewFromImage(labels).astype(np.int64)


def largest_component(foreground: np.ndarray, rows: int) -> np.ndarray:
    """Where ``foreground``'s largest component lies, its voxels face to face; of two of one size, the first in scan
    order, as ``scipy.ndimage.label`` numbers them. Two passes over slabs of ``rows`` planes along the first axis:
    the first numbers each slab's components, joins them across the slab's face with the slab before (a
    union-find whose root is the lowest number, the first in scan order) and sums their sizes; the second numbers
    each slab again and keeps the voxels of the largest root. Memory: one slab's numbers beside the answer."""
    parent = np.zeros(1, dtype=np.int64)
    sizes = np.zeros(1, dtype=np.int64)
    offsets, face = [], None

    def root(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = int(parent[label])
        return label

    for start in range(0, foreground.shape[0], rows):
        local = _components(foreground[start : start + rows])
        offset, count = len(parent) - 1, int(local.max())
        offsets.append(offset)
        parent = np.concatenate([parent, np.arange(offset + 1, offset + count + 1)])
        sizes = np.concatenate([sizes, np.bincount(local.ravel(), minlength=count + 1)[1:]])
        numbered = np.where(local > 0, local + offset, 0)
        if face is not None:
            touching = (face > 0) & (numbered[0] > 0)
            for a, b in {(int(x), int(y)) for x, y in zip(face[touching], numbered[0][touching], strict=True)}:
                low, high = sorted((root(a), root(b)))
                parent[high] = low
        face = numbered[-1]
    roots = np.array([root(label) for label in range(len(parent))])
    totals = np.bincount(roots[1:], weights=sizes[1:], minlength=len(parent))
    kept = np.zeros(foreground.shape, dtype=bool)
    if not totals.any():
        return kept
    largest = int(np.argmax(totals))
    for start, offset in zip(range(0, foreground.shape[0], rows), offsets, strict=True):
        local = _components(foreground[start : start + rows])
        kept[start : start + rows] = (local > 0) & (roots[np.where(local > 0, local + offset, 0)] == largest)
    return kept


class KeepLargestComponent(Transform):
    """Set every voxel outside the largest component of a foreground to 0, the component's voxels face to face; of
    two of one size, the first in scan order is kept.

    The foreground is the non-zero voxels of the channel ``mask``, which the output drops, else of any channel. It
    is found by slabs of ``rows`` planes (:func:`largest_component`), but which component is largest is known once
    the last slab is read: the stage needs the whole volume."""

    working_multiple = 1.5  # the foreground, one slab's numbers and the kept voxels, beside the output

    def __init__(self, mask: int | None = None, rows: int = 64) -> None:
        super().__init__()
        self.mask = mask
        self.rows = int(rows)

    def output_channels(self, channels: int) -> int:
        return channels - 1 if self.mask is not None else channels

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(
            LocalityKind.WHOLE_VOLUME, reason="which component is largest is known once every slab is read"
        )

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self.mask is None:
            values, foreground = tensor, (tensor != 0).any(0)
        else:
            mask = self.mask % tensor.shape[0]
            values = torch.cat([tensor[:mask], tensor[mask + 1 :]])
            foreground = tensor[mask] != 0
        kept = torch.from_numpy(largest_component(foreground.cpu().numpy(), self.rows)).to(values.device)
        return torch.where(kept, values, torch.zeros((), dtype=values.dtype, device=values.device))


class Otsu(Transform):
    """A body mask: 1 where the image is above Otsu's threshold, 0 elsewhere, one ``uint8`` channel.

    The threshold is ITK's (``OtsuThresholdImageFilter``, 128 bins) of the whole case, the image in float32 and a
    multichannel image (a colour photograph) as the mean of its channels. It needs the whole volume, which the
    loader hands it: the stage is ``WHOLE_VOLUME``.
    """

    # in volumes of an int16 case: the float32 grey copy (2), SimpleITK's own of it (2) and the uint8 mask (0.5)
    working_multiple = 4.5

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.WHOLE_VOLUME, reason="Otsu's threshold is computed on the whole volume")

    def output_channels(self, channels: int) -> int:
        return 1

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        _require_simpleitk()
        grey = tensor.float().mean(0)
        calculator = sitk.OtsuThresholdImageFilter()
        calculator.Execute(sitk.GetImageFromArray(grey.cpu().numpy()))
        # ITK compares in the image's own type: the float32 the volume is thresholded in
        return (grey > float(np.float32(calculator.GetThreshold()))).to(torch.uint8).unsqueeze(0)


#: The types isin and where have no kernel for, and the type each is compared in, which holds all its values.
_RELABEL = "Relabel the map (labels are small integers) or store it as uint32/int64."
_COMPARED_AS = {
    torch.bool: torch.uint8,
    torch.uint16: torch.int32,
    torch.uint32: torch.int64,
    torch.uint64: torch.int64,
}


class KeepLabels(Transform):
    """The labels another group of the case also holds; every other voxel 0.

    A label is then scored only when both maps hold it: on the fixed labels, ``group`` names the moving
    labels before any warp, so that a label the warp erases still counts. The other map's labels are read once per
    case, block by block; the stage is then voxel by voxel. A store that cannot serve a region alone (GIPL) is
    refused: it would be read whole, a read no plan counts.
    """

    locality = LocalityKind.POINTWISE
    # a uint16 map compared as int32 (2), the mask (0.5) and the int32 result (2)
    working_multiple = 4.5

    def __init__(self, group: str = "default") -> None:
        super().__init__()
        self.group = group
        self._held: dict[str, torch.Tensor] = {}  # per case, the labels the other map holds, background included

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if name not in self._held:
            dataset = self.dataset_holding(self.group, name)
            if dataset is None:
                raise TransformError(
                    f"'KeepLabels' found no map of group '{self.group}' for case '{name}'.",
                    "Name a group every case holds: KeepLabels: {group: MOVING_LABELS}.",
                )
            if not dataset.bounded_region_reads(self.group, name):
                raise TransformError(
                    f"'KeepLabels' would read the map of group '{self.group}' for case '{name}' whole: its store"
                    " cannot serve a region alone.",
                    "Store that group in a format read by region (.mha, .nii or OME-Zarr), so that its labels are"
                    " read block by block.",
                )
            labels: set[int] = set()
            for block in dataset.iter_data_blocks(self.group, name)():
                labels.update(int(label) for label in np.unique(block))
            if labels and max(labels) > torch.iinfo(torch.int64).max:
                raise TransformError(
                    f"'KeepLabels' found a label above 2^63-1 in the map of group '{self.group}' for case '{name}'.",
                    _RELABEL,
                )
            self._held[name] = torch.tensor(sorted(labels), dtype=torch.int64)
        held = self._held[name]
        if tensor.dtype is torch.uint64 and bool((tensor.view(torch.int64) < 0).any()):  # int64 would wrap it
            raise TransformError(f"'KeepLabels' was handed a map of case '{name}' with a label above 2^63-1.", _RELABEL)
        values = tensor.to(_COMPARED_AS.get(tensor.dtype, tensor.dtype))
        if not values.is_floating_point():  # a label this map's type cannot hold is not in it
            held = held[(held >= torch.iinfo(values.dtype).min) & (held <= torch.iinfo(values.dtype).max)]
        keep = torch.isin(values, held.to(device=values.device, dtype=values.dtype))
        return torch.where(keep, values, 0).to(tensor.dtype)


class FlatLabel(Transform):
    working_multiple = 1.0

    locality = LocalityKind.POINTWISE

    def __init__(self, labels: list[int] | None = None) -> None:
        super().__init__()
        self.labels = labels

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # Filled through the mask: indexing by it would first gather the indices of every voxel it selects.
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


class ThresholdLabels(Transform):
    """A label map from thresholded channels, a sigmoid head's postprocessing: channel ``c`` is on where
    ``sigmoid(x) >= threshold`` (``x >= threshold`` with ``sigmoid: false``), and a voxel takes ``labels[c]`` of the
    first channel on in ``priority`` order (channel indices; the channels' own order when unset), else 0. One
    channel, uint8, so each label lies in [0, 255]. In float32, as torch compares a float tensor with a number;
    NaN is on nowhere. The BraTS rule on TC, WT, ET logits, 4 where ET is on, else 1 where TC is, else 2 where WT
    is, is ``labels: [1, 2, 4], priority: [2, 0, 1]``.
    """

    # Per channel in turn: its sigmoid and the voxels on, beside the map; a one-channel input's worst.
    working_multiple = 1.5

    locality = LocalityKind.POINTWISE

    def __init__(
        self,
        labels: list[int],
        threshold: float = 0.5,
        priority: list[int] | None = None,
        sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.labels = [int(label) for label in labels]
        if not self.labels or any(not 0 <= label <= 255 for label in self.labels):
            raise TransformError(
                f"'ThresholdLabels' writes labels={list(labels)!r} into a uint8 map.",
                "Give one label per channel, each in [0, 255].",
            )
        self.priority = list(range(len(self.labels))) if priority is None else [int(channel) for channel in priority]
        if sorted(self.priority) != list(range(len(self.labels))):
            raise TransformError(
                f"'ThresholdLabels' was given priority={list(priority or [])!r} for {len(self.labels)} labels.",
                "priority orders every channel once: a permutation of 0 .. len(labels) - 1.",
            )
        self.threshold = float(threshold)
        self.sigmoid = bool(sigmoid)

    def output_channels(self, channels: int) -> int:
        return 1

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if tensor.shape[0] != len(self.labels):
            raise TransformError(
                f"'ThresholdLabels' names {len(self.labels)} labels, and '{name}' has {tensor.shape[0]} channels.",
                "Give one label per channel.",
            )
        out = torch.zeros(tensor.shape[1:], dtype=torch.uint8, device=tensor.device)
        # Lowest priority first: the channel written last is the first in priority order that is on.
        for channel in reversed(self.priority):
            logits = tensor[channel].float()
            on = (torch.sigmoid(logits) if self.sigmoid else logits) >= self.threshold
            out.masked_fill_(on, self.labels[channel])
        return out.unsqueeze(0)


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
        if tensor.shape[0] != 1:
            raise TransformError(
                f"'OneHot' encodes a one-channel label map, and '{name}' has {tensor.shape[0]} channels.",
                "Its output is [num_classes, *spatial]: store the labels as one channel.",
            )
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

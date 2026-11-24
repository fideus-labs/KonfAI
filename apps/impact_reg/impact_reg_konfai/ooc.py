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

"""Images read by regions, for the engines' out-of-core stages.

A :class:`RegionSource` hands back any box of an image as a tensor and never holds the image whole: from an array
(a NumPy memmap on disk, or a small array in memory). :func:`build_level` writes the Gaussian pyramid level
FireANTs computes on whole images, block by block, to a memmap. :class:`HeaderImages` stands in for FireANTs'
``BatchedImages`` with the geometry alone.
"""

from __future__ import annotations

import contextlib
import math
import os
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from konfai.data.patching.sweep import _sweep_targets

#: What a slab copied to the device holds on its way through the host.
_SLAB_BYTES = 64 << 20


class RegionSource:
    """A channel-first image ``[C, *shape]`` read by boxes; ``bytes_read`` counts what was asked of it."""

    shape: tuple[int, ...]
    channels: int
    device: str

    bytes_read = 0  # a reader that reads counts on its own instance

    def read(self, start: Sequence[int], stop: Sequence[int]) -> torch.Tensor:
        """The box ``[start, stop)`` along each spatial axis, as a float32 tensor ``[1, C, *size]`` on ``device``."""
        raise NotImplementedError

    def close(self) -> None:
        """Release what the source holds outside the process; nothing for most."""


def tiles(shape: Sequence[int], box: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """The boxes of ``box`` voxels a side (the last one along an axis takes what is left) tiling a grid, in C order."""
    return [
        (tuple(axis.start for axis in target), tuple(axis.stop for axis in target))
        for target in _sweep_targets(list(shape), [box] * len(shape))
    ]


def box_region(
    tile, halo: int | Sequence[int], shape: Sequence[int], least: int = 1
) -> tuple[list[int], list[int], tuple[slice, ...]]:
    """A box ``(start, stop)`` with ``halo`` voxels around (the same along every axis, or one count an axis), within a
    ``shape`` grid, and widened inside it to ``least`` voxels along an axis where the grid has them; and the slices of
    the box's own voxels in it."""
    start, stop = tile
    halos = [halo] * len(shape) if isinstance(halo, int) else list(halo)
    low = [max(0, a - h) for a, h in zip(start, halos, strict=True)]
    high = [min(extent, b + h) for b, h, extent in zip(stop, halos, shape, strict=True)]
    for axis, extent in enumerate(shape):
        side = min(least, extent)
        if high[axis] - low[axis] < side:
            high[axis] = min(extent, low[axis] + side)
            low[axis] = high[axis] - side
    return low, high, tuple(slice(a - lo, b - lo) for a, b, lo in zip(start, stop, low, strict=True))


def to_device(region: np.ndarray, device: str) -> torch.Tensor:
    """A float32 copy of the array ``region`` (a view of a memmap, say) on ``device``. For a GPU it goes through a
    pinned buffer, which PyTorch's parallel copy fills: a NumPy copy then a transfer from pageable memory runs five
    times slower."""
    if torch.device(device).type != "cuda":
        return torch.from_numpy(np.array(region, dtype=np.float32))
    staging = torch.empty(region.shape, dtype=torch.float32, pin_memory=True)
    staging.copy_(torch.from_numpy(region))
    return staging.to(device)


def to_array(target: np.ndarray, values: torch.Tensor) -> None:
    """Write ``values`` into the array view ``target``, through a pinned buffer from a GPU (see :func:`to_device`)."""
    values = values.detach()
    if values.device.type == "cuda":
        staging = torch.empty(values.shape, dtype=values.dtype, pin_memory=True)
        values = staging.copy_(values)
    torch.from_numpy(target).copy_(values)


class ArraySource(RegionSource):
    """A box of a NumPy array ``[C, *shape]``: a memmap on disk, or an array in memory. A memmap this source owns
    (``path``) is deleted with it."""

    def __init__(self, array: np.ndarray, device: str, path: Path | None = None) -> None:
        self.array = array
        self.channels = int(array.shape[0])
        self.shape = tuple(int(extent) for extent in array.shape[1:])
        self.device = device
        self.path = path

    def close(self) -> None:
        """Drop the memmap and delete the file this source owns."""
        if self.path is not None:
            self.array = np.zeros((self.channels, *([0] * len(self.shape))), dtype=np.float32)
            self.path.unlink(missing_ok=True)
            self.path = None

    def __del__(self) -> None:
        self.close()

    def read(self, start: Sequence[int], stop: Sequence[int]) -> torch.Tensor:
        region = self.array[(slice(None), *(slice(a, b) for a, b in zip(start, stop, strict=True)))]
        self.bytes_read += region.nbytes
        # A copy: a box spanning whole axes is a view of the array, which the reader may then change in place.
        return to_device(region, self.device).unsqueeze(0)


class TensorSource(RegionSource):
    """A box of a tensor ``[C, *shape]`` already on its device: a view of it, or a copy (``copies``) for a reader
    that changes its boxes in place."""

    def __init__(self, tensor: torch.Tensor, copies: bool = False) -> None:
        self.tensor, self.channels, self.copies = tensor, int(tensor.shape[0]), copies
        self.shape, self.device = tuple(int(extent) for extent in tensor.shape[1:]), str(tensor.device)

    def read(self, start: Sequence[int], stop: Sequence[int]) -> torch.Tensor:
        box = self.tensor[(slice(None), *(slice(a, b) for a, b in zip(start, stop, strict=True)))][None]
        return box.to(torch.float32, copy=True) if self.copies else box.float()


def free_bytes(device: str) -> int:
    """The memory of a CUDA ``device`` this process can still take: what the device has free, and what PyTorch keeps
    reserved without using it. 0 for any other device, which holds nothing."""
    where = torch.device(device)
    if where.type != "cuda":
        return 0
    return torch.cuda.mem_get_info(where)[0] + torch.cuda.memory_reserved(where) - torch.cuda.memory_allocated(where)


def held(sources: Sequence[RegionSource], room: float) -> list[RegionSource]:
    """``sources``, each read once and kept on its device while it still fits the ``room`` bytes left by those before
    it: its boxes are then copied from that tensor, where each read of a file goes through the host."""
    kept = []
    for source in sources:
        needed = 4 * source.channels * math.prod(source.shape)
        plane = 4 * source.channels * math.prod(source.shape[1:])
        slab = max(1, _SLAB_BYTES // max(plane, 1))
        # Filled slab by slab: the host holds one slab on its way, never the image, and the device the slab beside it.
        if not isinstance(source, TensorSource) and needed + min(slab, source.shape[0]) * plane <= room:
            room -= needed
            tensor = torch.empty((source.channels, *source.shape), dtype=torch.float32, device=source.device)
            for start in range(0, source.shape[0], slab):
                stop = min(source.shape[0], start + slab)
                tensor[:, start:stop] = source.read([start, *[0] * (len(source.shape) - 1)], [stop, *source.shape[1:]])[
                    0
                ]
            source = TensorSource(tensor, copies=True)
        kept.append(source)
    return kept


def on_device(source: RegionSource, share: float = 0.1) -> RegionSource:
    """``source`` kept on its device (``held``) when it takes no more than ``share`` of the device's free memory."""
    return held([source], free_bytes(source.device) * share)[0]


class StackedSource(RegionSource):
    """The channels of several sources on one grid, read together (an image and its mask)."""

    def __init__(self, sources: Sequence[RegionSource]) -> None:
        self.sources = list(sources)
        shapes = {source.shape for source in self.sources}
        if len(shapes) != 1:
            raise ValueError(f"the stacked sources lie on different grids: {sorted(shapes)}")
        self.shape = self.sources[0].shape
        self.channels = sum(source.channels for source in self.sources)
        self.device = self.sources[0].device

    def read(self, start: Sequence[int], stop: Sequence[int]) -> torch.Tensor:
        return torch.cat([source.read(start, stop) for source in self.sources], dim=1)

    @property
    def bytes_read(self) -> int:  # type: ignore[override]
        return sum(source.bytes_read for source in self.sources)


class OnesSource(RegionSource):
    """One channel of ones on a grid: the mask of an image that has none, beside one that has."""

    def __init__(self, shape: Sequence[int], device: str) -> None:
        self.shape, self.channels, self.device = tuple(shape), 1, device

    def read(self, start: Sequence[int], stop: Sequence[int]) -> torch.Tensor:
        return torch.ones((1, 1, *(b - a for a, b in zip(start, stop, strict=True))), device=self.device)


class RescaledSource(RegionSource):
    """``source`` clamped to ``[low, high]`` and rescaled to ``[0, top]``, box by box."""

    def __init__(self, source: RegionSource, low: float, high: float, top: float) -> None:
        self.source, self.low, self.high, self.top = source, low, high, top
        self.shape, self.channels, self.device = source.shape, source.channels, source.device

    def read(self, start: Sequence[int], stop: Sequence[int]) -> torch.Tensor:
        values = self.source.read(start, stop).clamp_(self.low, self.high)
        return values.sub_(self.low).mul_(self.top / max(self.high - self.low, 1e-12))

    @property
    def bytes_read(self) -> int:  # type: ignore[override]
        return self.source.bytes_read


class HeaderImages:
    """What FireANTs' registrations read of a ``BatchedImages`` to set themselves up, from an image's header alone:
    the torch (normalized, align_corners) to physical mapping, the shape and the device. ``__call__`` has no image to
    give: a sampled stage reads its boxes from a :class:`RegionSource` instead."""

    def __init__(self, image: sitk.Image, channels: int, device: str) -> None:
        dims = image.GetDimension()
        size = np.array(image.GetSize(), dtype=np.float64)
        spacing = np.array(image.GetSpacing(), dtype=np.float64)[None]
        direction = np.array(image.GetDirection(), dtype=np.float64).reshape(dims, dims)
        px2phy = np.eye(dims + 1)
        px2phy[:dims, -1] = np.array(image.GetOrigin(), dtype=np.float64)
        px2phy[:dims, :dims] = direction * spacing
        torch2px = np.eye(dims + 1)
        half = (size - 1) * 0.5
        torch2px[:dims, :dims] = np.diag(half)
        torch2px[:dims, -1] = half
        self.torch2phy = torch.from_numpy(px2phy @ torch2px).to(device).float().unsqueeze(0)
        self.phy2torch = torch.inverse(self.torch2phy[0]).float().unsqueeze(0)
        self.dims = dims
        self.device = device
        self.shape = (1, channels, *[int(extent) for extent in reversed(image.GetSize())])
        self.interpolate_mode = "bilinear" if dims == 2 else "trilinear"

    def get_torch2phy(self) -> torch.Tensor:
        return self.torch2phy

    def get_phy2torch(self) -> torch.Tensor:
        return self.phy2torch

    def size(self) -> int:
        return 1

    def __call__(self) -> torch.Tensor:
        raise RuntimeError("a sampled registration reads its images by regions; the whole image is never held.")


def fireants_level_sizes(shape: Sequence[int], scale: int) -> list[int]:
    """The size FireANTs gives a pyramid level: ``max(int(s / scale), 32)`` along each axis."""
    from fireants.utils.globals import MIN_IMG_SIZE

    return [max(int(extent / scale), MIN_IMG_SIZE) for extent in shape]


def fireants_gaussians(shape: Sequence[int], size: Sequence[int], device: str) -> list[torch.Tensor]:
    """FireANTs' smoothing kernels for a level: sigma ``0.5 * (N / N_level)`` along each axis, erf, truncated at 2."""
    from fireants.losses.cc import gaussian_1d

    sigmas = 0.5 * torch.tensor(
        [extent / down for extent, down in zip(shape, size, strict=True)], device=device, dtype=torch.float32
    )
    return [gaussian_1d(sigma, truncated=2) for sigma in sigmas]


@contextlib.contextmanager
def _level_array(
    shape: Sequence[int], path: Path | None, directory: Path | None, memory: bool
) -> Iterator[tuple[np.ndarray, Path | None]]:
    """The float32 array a level is written to: in memory with ``memory``, a memmap at ``path``, or one in a file of
    its own in ``directory`` (the system's temporary directory when unset), whose path comes with it and goes if the
    level fails."""
    owned = None
    if memory:
        array = np.empty(tuple(shape), dtype=np.float32)
    else:
        if path is None:
            descriptor, name = tempfile.mkstemp(suffix=".npy", prefix="impact_reg_level_", dir=directory)
            os.close(descriptor)
            path = owned = Path(name)
        array = np.lib.format.open_memmap(str(path), mode="w+", dtype=np.float32, shape=tuple(shape))
    try:
        yield array, owned
    except BaseException:
        if owned is not None:
            owned.unlink(missing_ok=True)
        raise
    if not memory:
        array.flush()


def build_level(
    source: RegionSource,
    size: Sequence[int],
    gaussians: Sequence[torch.Tensor] | None,
    *,
    mask_channel: bool = False,
    block: int = 64,
    path: Path | None = None,
    directory: Path | None = None,
    memory: bool = False,
) -> ArraySource:
    """The pyramid level FireANTs' ``downsample`` computes on the whole image, block by block from ``source``:
    the image channels smoothed by ``gaussians`` (``'same'`` convolution, zero outside the image) then read by
    trilinear interpolation at ``j * (N - 1) / (N_level - 1)`` (``align_corners``), the mask channel (the last one,
    with ``mask_channel``) interpolated without smoothing, as ``_downsample_image_and_mask`` does. Each output block
    reads the voxels it needs, with the kernels' tails around: the level equals the whole-image one wherever it is
    read. Written to a float32 memmap at ``path``, or to a file of its own in ``directory``, deleted with the source
    returned; to an array in memory with ``memory``."""
    from fireants.losses.cc import separable_filtering

    dims = len(source.shape)
    size = [int(extent) for extent in size]
    tails = [0] * dims
    if gaussians is not None:
        tails = [(int(kernel.numel()) - 1) // 2 for kernel in gaussians]
    ratios = [(extent - 1) / (down - 1) if down > 1 else 0.0 for extent, down in zip(source.shape, size, strict=True)]
    # A block reads ``block * ratio`` source voxels a side, and the kernels' tails: cut so that it reads no more than
    # 4 * block, down to one output voxel where the level is that much coarser than the source.
    block = max(1, min(block, int(4 * block / max(*ratios, 1.0))))
    with _level_array((source.channels, *size), path, directory, memory) as (level, owned):
        for starts, stops in tiles(size, block):
            # The source voxels the block's samples read, with the kernels' tails and one voxel for the interpolation.
            low = [max(0, math.floor(j0 * r) - tail - 1) for j0, r, tail in zip(starts, ratios, tails, strict=True)]
            high = [
                min(extent, math.ceil((j1 - 1) * r) + tail + 2)
                for j1, r, tail, extent in zip(stops, ratios, tails, source.shape, strict=True)
            ]
            span = source.read(low, high)
            image, mask = (span[:, :-1], span[:, -1:]) if mask_channel else (span, None)
            if gaussians is not None:
                image = separable_filtering(image, list(gaussians))
            # The samples, in the span's own normalized coordinates, (x, y, z) last as grid_sample takes them.
            axes = []
            for d in range(dims):
                positions = torch.arange(starts[d], stops[d], device=span.device, dtype=torch.float32) * ratios[d]
                extent = high[d] - low[d]
                axes.append(
                    (positions - low[d]) * (2.0 / (extent - 1)) - 1.0 if extent > 1 else torch.zeros_like(positions)
                )
            grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).flip(-1).unsqueeze(0)
            sampled = torch.nn.functional.grid_sample(
                image, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )
            if mask is not None:
                mask_sampled = torch.nn.functional.grid_sample(
                    mask, grid, mode="bilinear", padding_mode="zeros", align_corners=True
                )
                sampled = torch.cat([sampled, mask_sampled], dim=1)
            to_array(level[(slice(None), *(slice(a, b) for a, b in zip(starts, stops, strict=True)))], sampled[0])
    return ArraySource(level, source.device, owned)


def blur_level(
    source: RegionSource,
    gaussians: Sequence[torch.Tensor],
    *,
    mask_channel: bool = False,
    block: int = 64,
    directory: Path | None = None,
) -> ArraySource:
    """``source`` smoothed by ``gaussians`` on its own grid, block by block (each block read with the kernels' tails
    around, zero outside the image), the mask channel left as it is: what the rigid stage does to its moving level a
    second time (``_smooth_image_not_mask``). Written to a file of its own in ``directory``, deleted with the source
    returned."""
    from fireants.losses.cc import separable_filtering

    tails = [(int(kernel.numel()) - 1) // 2 for kernel in gaussians]
    with _level_array((source.channels, *source.shape), None, directory, False) as (level, owned):
        for start, stop in tiles(source.shape, block):
            low, high, own = box_region((start, stop), tails, source.shape)
            span = source.read(low, high)
            image, mask = (span[:, :-1], span[:, -1:]) if mask_channel else (span, None)
            image = separable_filtering(image, list(gaussians))
            if mask is not None:
                image = torch.cat([image, mask], dim=1)
            to_array(
                level[(slice(None), *(slice(s, e) for s, e in zip(start, stop, strict=True)))],
                image[0][(slice(None), *own)],
            )
    return ArraySource(level, source.device, owned)

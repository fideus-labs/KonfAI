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

"""The native pair of the sampled strategy: HDF5 twins that ITK's readers stream by chunks for the stages that read
boxes, uncompressed MetaImage twins memory-mapped for the engines that map their images, the statistics taken by
passes over them (the percentiles FireANTs rescales by, the centres a linear stage starts from), and a transform's
dense field written band by band. Nothing here reads an image whole."""

from __future__ import annotations

import contextlib
import itertools
import math
import os
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from konfai.utils.dataset.statistics import _lerp_like_numpy, _order_statistics

#: Side of a twin's chunks, in voxels. A box read with its halo touches the few chunks under it, each read whole,
#: wherever it lands: a 70^3 box touches 27 chunks, 3.4 MiB for 1.3 MiB of voxels, where the raw rows of a
#: MetaImage touch a page per row, 19 MiB (3.3 ms against 14.4 ms a box on an NVMe, cold cache).
CHUNK = 32
_BAND_BYTES = 512 << 20
#: Bytes a twin's header keeps spare, for its origin to be written again in place (``shift_origin``).
_HEADER_ROOM = 64


def _slabs(shape: Sequence[int], itemsize: int, most: int | None = None) -> Iterator[tuple[slice, slice]]:
    """The blocks of about ``_BAND_BYTES`` an array of ``shape`` (``itemsize`` bytes an item) is read or written by,
    as slices along its first two axes: slabs of whole planes, ``most`` at most, or rows of one plane where a plane
    alone is larger. A block is contiguous in C order."""
    plane = math.prod(shape[1:]) * itemsize
    if plane <= _BAND_BYTES:
        band = max(1, min(most or shape[0], _BAND_BYTES // max(1, plane)))
        for start in range(0, shape[0], band):
            yield slice(start, min(start + band, shape[0])), slice(0, shape[1])
        return
    rows = max(1, _BAND_BYTES // max(1, math.prod(shape[2:]) * itemsize))
    for depth in range(shape[0]):
        for row in range(0, shape[1], rows):
            yield slice(depth, depth + 1), slice(row, min(row + rows, shape[1]))


@dataclass(frozen=True)
class AxisMap:
    """A grid's voxels seen with its axes permuted and read backwards, in NumPy order ((z,) y, x), of any dimension:
    axis ``k`` of the view is axis ``axes[k]`` of the grid, reversed where ``flips[k]``. ``to_lps`` is the view
    SimpleITK's DICOMOrient gives a run on the whole pair: each voxel axis along the LPS axis nearest to it and pointing
    its way, nothing resampled (an oblique grid keeps the rotation left)."""

    axes: tuple[int, ...]
    flips: tuple[bool, ...]

    @classmethod
    def identity_of(cls, dimension: int) -> AxisMap:
        return cls(tuple(range(dimension)), (False,) * dimension)

    @classmethod
    def to_lps(cls, direction: Sequence[float]) -> AxisMap:
        # Column j of an ITK direction (row-major) is voxel axis j, ITK order ((x, y), z), in LPS; NumPy axis k is ITK
        # axis n - 1 - k.
        n = math.isqrt(len(direction))
        matrix = np.reshape(np.asarray(direction, dtype=float), (n, n))
        nearest = max(itertools.permutations(range(n)), key=lambda p: sum(abs(matrix[i, p[i]]) for i in range(n)))
        return cls(
            tuple(n - 1 - nearest[n - 1 - k] for k in range(n)),
            tuple(bool(matrix[n - 1 - k, nearest[n - 1 - k]] < 0) for k in range(n)),
        )

    @property
    def identity(self) -> bool:
        return self.axes == tuple(range(len(self.axes))) and not any(self.flips)

    def inverse(self) -> AxisMap:
        """The map from the view back to the grid."""
        axes, flips = [0] * len(self.axes), [False] * len(self.axes)
        for k, (axis, flip) in enumerate(zip(self.axes, self.flips, strict=True)):
            axes[axis], flips[axis] = k, flip
        return AxisMap(tuple(axes), tuple(flips))

    def extents(self, extents: Sequence[int]) -> list[int]:
        """The view's extents of a grid of ``extents``."""
        return [int(extents[axis]) for axis in self.axes]

    def box(self, low: Sequence[int], high: Sequence[int], extents: Sequence[int]) -> tuple[slice, ...]:
        """The box ``[low, high)`` of a grid of ``extents``, where the view holds it."""
        return tuple(
            slice(extents[axis] - high[axis], extents[axis] - low[axis]) if flip else slice(low[axis], high[axis])
            for axis, flip in zip(self.axes, self.flips, strict=True)
        )

    def view(self, values: np.ndarray) -> np.ndarray:
        """``values``, a block of the grid (its spatial axes first, any components after), laid out as the view lays
        it."""
        values = values.transpose(*self.axes, *range(len(self.axes), values.ndim))
        flipped = [k for k, flip in enumerate(self.flips) if flip]
        return np.ascontiguousarray(np.flip(values, flipped) if flipped else values)

    def geometry(
        self, spacing: Sequence[float], origin: Sequence[float], direction: Sequence[float], extents: Sequence[int]
    ) -> tuple[list[float], list[float], list[float]]:
        """The view's spacing, origin and direction (ITK order, the direction row-major) of a grid of ``extents`` and
        that geometry: the same voxels at the same physical points."""
        n = len(self.axes)
        matrix = np.reshape(np.asarray(direction, dtype=float), (n, n))
        corner = [0] * n  # the grid's voxel the view starts at, NumPy order
        for axis, flip in zip(self.axes, self.flips, strict=True):
            corner[axis] = extents[axis] - 1 if flip else 0
        start = np.asarray(origin, dtype=float) + matrix @ (np.asarray(spacing, dtype=float) * corner[::-1])
        columns = [n - 1 - self.axes[n - 1 - i] for i in range(n)]  # view ITK axis i is the grid's ITK axis columns[i]
        signs = [-1.0 if self.flips[n - 1 - i] else 1.0 for i in range(n)]
        view = np.stack([sign * matrix[:, column] for sign, column in zip(signs, columns, strict=True)], axis=1)
        return [float(spacing[column]) for column in columns], start.tolist(), view.reshape(-1).tolist()


def write_twin(
    dataset,
    group: str,
    name: str,
    target: Path,
    *,
    dtype: str,
    clip: tuple[float, float] | None = None,
    chunk: int = CHUNK,
    band: int | None = None,
    voxel: float = 1.0,
    positive: bool = False,
    axis_map: AxisMap | None = None,
    mirror: Sequence[int] = (),
    shift: Sequence[float] | None = None,
) -> Path:
    """The entry ``(group, name)`` written to ``target``: an HDF5 image in ITK's layout, its voxels chunked
    ``chunk``^3, where it ends in ``.h5``, an uncompressed MetaImage otherwise; from one read of the entry by bands
    through the dataset's region reads (``band`` planes at most, rows of one plane where a plane alone is larger than
    a band). ``clip`` winsorises it, as the whole-image run winsorises the images of the engine that reads it;
    ``dtype`` "uint8" makes a mask (any non-zero voxel on, or with ``positive`` the positive ones). ``axis_map``
    writes the entry as that view of it (``AxisMap``): a band of the entry lands in a band of the twin where the map
    keeps the outer axis outer. ``mirror`` reverses those axes of the entry (NumPy order), its header kept but its
    origin moved by ``shift`` (mm, ITK order), as a test-time copy's flip does. The geometry is measured in ``voxel`` (the fixed image's finest, as the engines measure
    a pair). The headers are ITK's own, written by it for a one-voxel image of the same geometry and voxel type."""
    shape, attributes = dataset.get_infos(group, name)
    extents = [int(extent) for extent in shape[1:]]
    axis_map = axis_map or AxisMap.identity_of(len(extents))
    geometry = axis_map.geometry(*(attributes.get_np_array(key) for key in ("Spacing", "Origin", "Direction")), extents)
    if shift is not None:
        geometry = (geometry[0], list(np.add(geometry[1], shift)), geometry[2])
    if mirror:  # the voxels reversed where the header is not
        axis_map = AxisMap(
            axis_map.axes, tuple(f != (a in mirror) for a, f in zip(axis_map.axes, axis_map.flips, strict=True))
        )
    with _new_twin(target, geometry, axis_map.extents(extents), dtype, chunk, voxel) as voxels:
        for depth, rows in _slabs(extents, 4, band or chunk):  # read as float32
            slices = (slice(0, 1), depth, rows, *(slice(None) for _ in extents[2:]))
            values = np.asarray(dataset.read_data_slice(group, name, slices)[0][0], dtype=np.float32)
            if dtype == "uint8":
                values = ((values > 0) if positive else (values != 0)).astype(np.uint8)
            else:
                values = values.astype(dtype, copy=False)
            if clip is not None:
                values = np.clip(values, clip[0], clip[1])
            box = axis_map.box(
                (depth.start, rows.start, *[0] * len(extents[2:])), (depth.stop, rows.stop, *extents[2:]), extents
            )
            voxels[box] = axis_map.view(values)
    return target


@contextlib.contextmanager
def _new_twin(target: Path, geometry, shape: list[int], dtype: str, chunk: int = CHUNK, voxel: float = 1.0):
    """The voxels of a new twin of ``shape`` (z, y, x) at ``target``, its ``geometry`` measured in ``voxel``: an HDF5
    image in ITK's layout, chunked ``chunk``^3, where it ends in ``.h5``, an uncompressed MetaImage memory mapped for
    writing otherwise; closed, or flushed, once written."""
    if target.suffix == ".h5":
        with h5py.File(target, "w") as file:
            for key, value in _hdf5_header(geometry, shape, dtype, voxel).items():
                file[key] = value
            file["ITKImage/0/Dimension"] = np.array(shape[::-1], dtype=np.uint64)
            file.create_group("ITKImage/0/MetaData")
            yield file.create_dataset(
                "ITKImage/0/VoxelData", shape=shape, dtype=dtype, chunks=tuple(min(chunk, extent) for extent in shape)
            )
        return
    start = _new_metaimage(target, _one_voxel(geometry, len(shape), dtype, voxel), shape, np.dtype(dtype).itemsize)
    voxels = np.memmap(target, dtype=dtype, mode="r+", offset=start, shape=tuple(shape))
    yield voxels
    voxels.flush()


def _one_voxel(geometry, dim: int, dtype: str, voxel: float = 1.0):
    """A one-voxel SimpleITK image carrying ``geometry`` (spacing, origin, direction) measured in ``voxel``, of the
    twin's voxel type."""
    import SimpleITK as sitk

    spacing, origin, direction = (np.asarray(value, dtype=float) for value in geometry)
    image = sitk.Image([1] * dim, {"uint8": sitk.sitkUInt8, "float32": sitk.sitkFloat32}[dtype])
    image.SetSpacing((spacing / voxel).tolist())
    image.SetOrigin((origin / voxel).tolist())
    image.SetDirection(direction.reshape(-1).tolist())
    return image


def _metaimage_header(geometry, extents: list[int], length: int | None = None) -> bytes:
    """The MetaImage header ITK writes for the one-voxel image ``geometry``, its DimSize set to ``extents``
    (z, y, x) and a comment padding it to a multiple of eight bytes with room to spare, or to ``length``: an
    uncompressed file of that size follows the installed ITK's conventions, and its header can be written again in
    place (``shift_origin``)."""
    import SimpleITK as sitk

    with tempfile.TemporaryDirectory() as scratch:
        mha = Path(scratch) / "header.mha"
        sitk.WriteImage(geometry, str(mha), useCompression=False)
        text = mha.read_bytes()
    last = b"ElementDataFile = LOCAL\n"
    one = b"DimSize = " + b" ".join(b"1" for _ in extents) + b"\n"
    size = b"DimSize = " + " ".join(str(extent) for extent in extents[::-1]).encode() + b"\n"
    assert text.count(one) == 1
    fields = text[: text.index(last)].replace(one, size)
    # The pixels start on a multiple of eight bytes: a reader maps them where they are, aligned for their type.
    comment = b"Comment = twin\n"
    if length is None:
        length = len(fields) + len(comment) + len(last) + _HEADER_ROOM
        length += -length % 8
    padding = length - len(fields) - len(comment) - len(last)
    if padding < 0:
        raise ValueError(f"the header needs {-padding} more bytes than the {length} it has.")
    return fields + comment[:-1] + b"-" * padding + b"\n" + last


def _hdf5_header(geometry, extents: list[int], dtype: str, voxel: float = 1.0) -> dict[str, object]:
    """The HDF5 header datasets ITK writes for a one-voxel image of this geometry and voxel type: the chunked twin
    follows the installed ITK's conventions, the direction matrix's layout among them."""
    import itk  # the engine's own dependency

    dim = len(extents)
    image = itk.Image[itk.UC if dtype == "uint8" else itk.F, dim].New()
    image.SetRegions([1] * dim)
    image.Allocate()
    spacing, origin, direction = (np.asarray(value, dtype=float) for value in geometry)
    image.SetSpacing((spacing / voxel).tolist())
    image.SetOrigin((origin / voxel).tolist())
    image.SetDirection(itk.matrix_from_array(direction.reshape(dim, dim)))
    keys = (
        "ITKVersion",
        "HDFVersion",
        "ITKImage/0/Directions",
        "ITKImage/0/Origin",
        "ITKImage/0/Spacing",
        "ITKImage/0/VoxelType",
    )
    with tempfile.TemporaryDirectory() as scratch:
        itk.imwrite(image, str(Path(scratch) / "header.h5"))
        with h5py.File(Path(scratch) / "header.h5", "r") as file:
            return {key: file[key][()] for key in keys}


@dataclass
class Mapped:
    """A twin read where it is asked: its voxels (a memmap, an HDF5 dataset or an array view) and a one-voxel
    SimpleITK image carrying its geometry."""

    array: Any
    geometry: Any


def mmap_image(path: Path) -> Mapped:
    """The uncompressed MetaImage at ``path`` with its pixel block memory mapped copy-on-write: the OS pages in what
    a reader touches, so a metric sampling points reads pages, never the image."""
    reader, dtype, shape, offset = _pixels(path)
    if shape[-1] != 1 or dtype not in (np.float32, np.uint8):
        raise ValueError(f"{path} is a {dtype} image of {shape[-1]} components; a twin is a scalar float32 or uint8.")
    return Mapped(np.memmap(path, dtype=dtype, mode="c", offset=offset, shape=shape[:-1]), geometry_of(reader))


def read_information(path: Path):
    """A SimpleITK reader of ``path`` that has read its header, the image's geometry and type, not its pixels."""
    import SimpleITK as sitk

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return reader


def geometry_of(source, origin=None, scale: float = 1.0) -> Any:
    """A one-voxel SimpleITK image of the pixel type and geometry of ``source``, an image or a reader that read the
    information: its origin ``origin`` when given, its spacing and origin times ``scale``."""
    import SimpleITK as sitk

    components = getattr(source, "GetNumberOfComponents", None) or source.GetNumberOfComponentsPerPixel
    geometry = sitk.Image([1] * source.GetDimension(), source.GetPixelID(), components())
    geometry.SetSpacing([v * scale for v in source.GetSpacing()])
    geometry.SetOrigin([v * scale for v in (source.GetOrigin() if origin is None else origin)])
    geometry.SetDirection(source.GetDirection())
    return geometry


def _bands(array) -> Iterator[tuple[tuple[int, int], np.ndarray]]:
    """The map read by the blocks of ``_slabs``, sized for float64, with the index (first axis, second axis) each
    starts at."""
    for depth, rows in _slabs(array.shape, 8):
        yield (depth.start, rows.start), np.asarray(array[depth, rows])


def mapped_quantile(mapped: Mapped, q: float) -> float:
    """``numpy.quantile`` of the map's values, as KonfAI's ``read_data_quantile`` takes it: by bounded passes over
    its bands, never the image whole."""
    low, high, weight = _order_statistics(lambda: (band for _, band in _bands(mapped.array)), q)
    return float(_lerp_like_numpy(low, high, weight) if weight else low)


def mapped_window(mapped: Mapped) -> tuple[float, float]:
    """The range a mapped image is winsorised to: its 0.5-99.5 percentiles, or its whole range where they meet (a
    small bright object on a background), which ``intensity.winsorized`` leaves alone."""
    bounds = mapped_quantile(mapped, 0.005), mapped_quantile(mapped, 0.995)
    return bounds if bounds[1] > bounds[0] else (mapped_quantile(mapped, 0.0), mapped_quantile(mapped, 1.0))


def mapped_centre(mapped: Mapped, mask: Mapped | None, kind: str) -> np.ndarray:
    """The centre a starting translation takes a mapped image from: its frame's (``'cof'``), or the centre of mass of
    its intensities winsorised to their 0.5-99.5 percentiles and counted from the lower one, inside ``mask`` when
    given (``'com'``), by one pass over bands of the map; an image with no weight gives its frame's centre.
    ``intensity.center_of_mass_translation`` takes it here too, so that a pair registered whole and its twins start
    from the same translation to the bit."""
    shape = mapped.array.shape
    if kind == "com":
        bounds = mapped_window(mapped)
        total, moments = 0.0, np.zeros(len(shape))
        for (depth, row), band in _bands(mapped.array):
            weights = np.clip(band.astype(np.float64), *bounds) - bounds[0]
            if mask is not None:
                weights *= np.asarray(mask.array[depth : depth + band.shape[0], row : row + band.shape[1]]) != 0
            total += weights.sum()
            offsets = (depth, row, *[0] * (len(shape) - 2))
            for axis in range(len(shape)):
                profile = weights.sum(axis=tuple(other for other in range(len(shape)) if other != axis))
                moments[axis] += profile @ (np.arange(profile.size) + offsets[axis])
        if total > 0:
            return np.asarray(
                mapped.geometry.TransformContinuousIndexToPhysicalPoint([float(v) for v in moments[::-1] / total])
            )
    frame = [(n - 1) / 2 for n in shape[::-1]]
    return np.asarray(mapped.geometry.TransformContinuousIndexToPhysicalPoint(frame))


def hdf5_twin(path: Path) -> Mapped:
    """The HDF5 twin at ``path`` (``write_twin``) as a :class:`Mapped`: its voxels read from the file where a box
    asks for them, chunk by chunk."""
    return Mapped(h5py.File(path, "r")["ITKImage/0/VoxelData"], geometry_of(read_information(path)))


def open_twin(path: Path) -> Mapped:
    """A twin of ``write_twin``: an HDF5 image read by chunks, or a MetaImage memory-mapped."""
    return hdf5_twin(path) if path.suffix == ".h5" else mmap_image(path)


def nonzero(mapped: Mapped) -> int:
    """The voxels of the map that are not zero, counted band by band."""
    return sum(int(np.count_nonzero(band)) for _, band in _bands(mapped.array))


def shift_field(path: Path, shift) -> None:
    """Add the vector ``shift`` to every voxel of the uncompressed displacement field at ``path`` ((x, y, z)
    components), in place, band by band: the field of a registration whose moving image was translated by it first."""
    _, dtype, shape, offset = _pixels(path)
    field = np.memmap(path, dtype=dtype, mode="r+", offset=offset, shape=shape)
    for block in _slabs(shape, dtype.itemsize):
        field[block] += np.asarray(shift, dtype=dtype)
    field.flush()


def _pixels(path: Path):
    """The uncompressed MetaImage at ``path``: its reader (information read), the NumPy type of a component, the
    shape of its pixel block ((z, y, x) then the components) and where that block starts."""
    import SimpleITK as sitk

    reader = read_information(path)
    dtype = np.dtype(sitk.GetArrayFromImage(sitk.Image([1] * reader.GetDimension(), reader.GetPixelID())).dtype)
    shape = (*(int(extent) for extent in reader.GetSize()[::-1]), reader.GetNumberOfComponents())
    offset = path.stat().st_size - math.prod(shape) * dtype.itemsize  # the pixel block ends the file
    if offset < 0:
        raise ValueError(f"{path} holds fewer bytes than its voxels: a compressed MetaImage has no pixel block.")
    return reader, dtype, shape, offset


def _new_metaimage(path: Path, geometry, extents: list[int], itemsize: int) -> int:
    """A new uncompressed MetaImage at ``path`` of the one-voxel image ``geometry``'s type and geometry, ``extents``
    (z, y, x) voxels of ``itemsize`` bytes, zero (a sparse file): where its pixels start."""
    header = _metaimage_header(geometry, extents)
    with open(path, "wb") as file:
        file.write(header)
        file.truncate(len(header) + math.prod(extents) * itemsize)
    return len(header)


def remap(path: Path, axis_map: AxisMap) -> None:
    """The uncompressed MetaImage at ``path`` as ``axis_map``'s view of it, in place: written band by band into a file
    beside it, its components as they are (a field's are physical), then put in its place."""
    reader, dtype, shape, offset = _pixels(path)
    source = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)
    extents = list(shape[:-1])
    view = axis_map.extents(extents)
    spacing, origin, direction = axis_map.geometry(
        reader.GetSpacing(), reader.GetOrigin(), reader.GetDirection(), extents
    )
    geometry = geometry_of(reader, origin)
    geometry.SetSpacing(spacing)
    geometry.SetDirection(direction)
    staged = path.with_name(f".{path.name}.view")
    start = _new_metaimage(staged, geometry, view, shape[-1] * dtype.itemsize)
    target = np.memmap(staged, dtype=dtype, mode="r+", offset=start, shape=(*view, shape[-1]))
    for depth, rows in _slabs(shape, dtype.itemsize):
        low, high = (depth.start, rows.start, *[0] * len(extents[2:])), (depth.stop, rows.stop, *extents[2:])
        target[axis_map.box(low, high, extents)] = axis_map.view(np.asarray(source[depth, rows]))
    target.flush()
    del source, target
    os.replace(staged, path)


def shift_origin(path: Path, shift) -> None:
    """The twin at ``path`` (``write_twin``) placed ``shift`` back, ``moved(p) = image(p + shift)``: its header
    written again in place, its voxels as they are."""
    reader, _, shape, offset = _pixels(path)
    origin = np.subtract(reader.GetOrigin(), shift)
    with open(path, "r+b") as file:
        file.write(_metaimage_header(geometry_of(reader, origin), list(shape[:-1]), length=offset))


def _mirrored(block: slice, extent: int, mirrored: bool) -> slice:
    return slice(extent - block.stop, extent - block.start) if mirrored else block


def fold_mirrored(target: Path, copies: Sequence[tuple[Path, Sequence[int]]]) -> None:
    """The mean, in place at ``target``, of the uncompressed field there and of ``copies``, band by band: each the
    field, on the same grid, of the pair mirrored along those of its axes (NumPy order, its header kept: a test-time
    copy's flip), brought back first as the predictor undoes the flip: mirrored again, its vectors reflected through
    the plane normal to each mirrored axis's direction."""
    reader, dtype, shape, offset = _pixels(target)
    total = np.memmap(target, dtype=dtype, mode="r+", offset=offset, shape=shape)
    dims = len(shape) - 1
    direction = np.reshape(np.asarray(reader.GetDirection(), dtype=np.float64), (dims, dims))
    sources = [
        (np.memmap(path, dtype=dtype, mode="r", offset=_pixels(path)[3], shape=shape), sorted(axes))
        for path, axes in copies
    ]
    for depth, rows in _slabs(shape, dtype.itemsize):
        folded = np.asarray(total[depth, rows], dtype=np.float64)
        for source, axes in sources:
            values = np.asarray(
                source[_mirrored(depth, shape[0], 0 in axes), _mirrored(rows, shape[1], 1 in axes)], dtype=np.float64
            )
            values = np.flip(values, axes) if axes else values
            for axis in axes:  # the ITK column of a NumPy axis: its world direction
                normal = direction[:, dims - 1 - axis]
                values = values - 2 * (values @ normal)[..., None] * normal
            folded += values
        total[depth, rows] = (folded / (len(sources) + 1)).astype(dtype)
    total.flush()


def to_millimetres(path: Path, voxel: float) -> None:
    """The field at ``path``, found on a pair measured in ``voxel`` (``write_twin``), in millimetres: its grid and
    its vectors times ``voxel``, written band by band to a file that then takes its place."""
    reader, dtype, shape, offset = _pixels(path)
    target = path.with_name(f".{path.name}.mm")
    start = _new_metaimage(target, geometry_of(reader, scale=voxel), list(shape[:-1]), shape[-1] * dtype.itemsize)
    source = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)
    world = np.memmap(target, dtype=dtype, mode="r+", offset=start, shape=shape)
    for block in _slabs(shape, dtype.itemsize):
        world[block] = source[block] * dtype.type(voxel)
    world.flush()
    del source, world
    os.replace(target, path)


class PlaneHeader:
    """A twin's header without its voxel axis ``axis`` (numpy's, of a (z, y, x) volume): what a reader of the plane
    reads, its size, spacing, origin and direction those of the other two axes."""

    def __init__(self, header, axis: int) -> None:
        self._header, self._keep = header, [a for a in range(3) if a != 2 - axis]

    def GetDimension(self) -> int:
        return 2

    def GetSize(self) -> tuple:
        return tuple(self._header.GetSize()[a] for a in self._keep)

    def GetSpacing(self) -> tuple:
        return tuple(self._header.GetSpacing()[a] for a in self._keep)

    def GetOrigin(self) -> tuple:
        return tuple(self._header.GetOrigin()[a] for a in self._keep)

    def GetDirection(self) -> tuple:
        direction = np.reshape(self._header.GetDirection(), (3, 3))
        return tuple(direction[np.ix_(self._keep, self._keep)].ravel())

    def GetPixelID(self) -> int:
        return self._header.GetPixelID()

    def GetNumberOfComponents(self) -> int:
        header = self._header
        return (getattr(header, "GetNumberOfComponents", None) or header.GetNumberOfComponentsPerPixel)()


def plane_of(mapped: Mapped, axis: int) -> Mapped:
    """A twin one voxel thick along numpy axis ``axis`` as its plane: a view of its first slice along it, with the
    plane's geometry (``PlaneHeader``)."""
    return Mapped(mapped.array[(slice(None),) * axis + (0,)], geometry_of(PlaneHeader(mapped.geometry, axis)))


def write_plane(twin: Path, axis: int, target: Path, scale: float = 1.0, rows: int = 256) -> Path:
    """The first slice along numpy axis ``axis`` of a twin (``write_twin``) as a 2-D twin of the same form at
    ``target``: the plane's geometry (``PlaneHeader``), its spacing and origin times ``scale``, written by rows."""
    source = open_twin(twin)
    plane = PlaneHeader(source.geometry, axis)
    geometry = (
        np.asarray(plane.GetSpacing()) * scale,
        np.asarray(plane.GetOrigin()) * scale,
        np.asarray(plane.GetDirection()),
    )
    shape = [extent for index, extent in enumerate(source.array.shape) if index != axis]
    with _new_twin(target, geometry, shape, np.dtype(source.array.dtype).name) as voxels:
        index: list = [slice(None)] * 3
        index[axis] = 0
        for begin in range(0, shape[0], rows):
            index[1 if axis == 0 else 0] = slice(begin, begin + rows)
            voxels[begin : begin + rows] = source.array[tuple(index)]
    return target


def lift_plane_field(plane: Path, path: Path, fixed: Mapped, moving: Mapped, axis: int, rows: int = 256) -> Path:
    """The field of a plane (``plane``, ``[y, x, 2]``) as the field of the volume one voxel thick it was taken from,
    at ``path`` on the fixed twin's grid: its two components on the plane's axes and, along the dropped one, the shift
    from the fixed plane to the moving one, as ``EngineRegistration`` brings back a plane it registered. Row by row."""
    _, dtype, shape, offset = _pixels(plane)
    source = np.memmap(plane, dtype=dtype, mode="r", offset=offset, shape=shape)
    dropped = 2 - axis  # the (x, y, z) component along the dropped voxel axis
    keep = [a for a in range(3) if a != dropped]
    shift = moving.geometry.GetOrigin()[dropped] - fixed.geometry.GetOrigin()[dropped]
    target = field_map(path, fixed)
    lifted = target[(slice(None),) * axis + (0,)]
    for start in range(0, shape[0], rows):
        lifted[start : start + rows, ..., keep] = source[start : start + rows]
        lifted[start : start + rows, ..., dropped] = shift
    target.flush()
    return path


def field_map(path: Path, fixed: Mapped) -> np.memmap:
    """A displacement field on the grid of the fixed twin as an uncompressed MetaImage at ``path``, its voxels
    (``[*grid, 3]`` float32, (x, y, z) components) memory mapped for writing: the field is written by regions."""
    import SimpleITK as sitk

    extents = list(fixed.array.shape)
    vector = sitk.Image([1] * len(extents), sitk.sitkVectorFloat32, len(extents))
    vector.CopyInformation(fixed.geometry)
    start = _new_metaimage(path, vector, extents, len(extents) * 4)
    return np.memmap(path, dtype=np.float32, mode="r+", offset=start, shape=(*extents, len(extents)))


def write_field(path: Path, fixed: Mapped, transform, band: int | None = None) -> Path:
    """``transform`` (SimpleITK, physical, fixed to moving) as a displacement field on the grid of the fixed twin,
    written to ``path`` as an uncompressed MetaImage by the blocks of ``_slabs`` (``band`` planes at most), sized for
    the float64 field SimpleITK computes: the field is never held whole."""
    import SimpleITK as sitk

    extents = list(fixed.array.shape)
    spacing, direction = fixed.geometry.GetSpacing(), fixed.geometry.GetDirection()
    field = field_map(path, fixed)
    for depth, rows in _slabs((*extents, len(extents)), 8, band):
        size = [*extents[:1:-1], rows.stop - rows.start, depth.stop - depth.start]
        index = [0.0] * (len(extents) - 2) + [float(rows.start), float(depth.start)]  # (x, y, z)
        origin = fixed.geometry.TransformContinuousIndexToPhysicalPoint(index)
        values = sitk.TransformToDisplacementField(transform, sitk.sitkVectorFloat64, size, origin, spacing, direction)
        field[depth, rows] = sitk.GetArrayFromImage(values)
    field.flush()
    return path

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


"""The ITK transform backend: displacement fields and parametric transforms in one HDF5 group."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]
from konfai.utils.dataset.abstract import AbstractFile
from konfai.utils.dataset.attribute import (
    DISPLACEMENT_FIELD_ATTRIBUTE,
    Attribute,
    _encode_transform_leaves,
    image_to_data,
    is_an_image,
)
from konfai.utils.dataset.h5 import _get_h5_file_lock, _h5_read_pool, _open_h5
from konfai.utils.dataset.staging import is_staging_entry
from konfai.utils.dataset.stream import DataStream
from konfai.utils.errors import DatasetManagerError


def _create_itk_transform_file(path: str, spatial: list[int], attributes: Attribute) -> tuple[Any, Any]:
    """An ITK displacement-transform HDF5 file with its parameters dataset still to fill.

    Three datasets, as ITK's own writer lays them out: the type (a variable-length ASCII string,
    which is what ITK's reader accepts), the fixed parameters (size, origin, spacing, direction)
    and the parameters, the field buffer with the component fastest, float64. Returns the open file
    and the parameters dataset.
    """
    fixed = np.concatenate(
        [
            np.asarray(spatial[::-1], dtype=np.float64),  # size, in (x, y, z)
            attributes.get_np_array("Origin").astype(np.float64),
            attributes.get_np_array("Spacing").astype(np.float64),
            attributes.get_np_array("Direction").astype(np.float64).reshape(-1),
        ]
    )
    file = _open_h5(path, "w")
    file.create_dataset(
        "TransformGroup/0/TransformType",
        data=[b"DisplacementFieldTransform_double_3_3"],
        dtype=h5py.string_dtype(encoding="ascii"),
    )
    file.create_dataset("TransformGroup/0/TransformFixedParameters", data=fixed)
    parameters = file.create_dataset(
        "TransformGroup/0/TransformParameters", shape=(3 * int(np.prod(spatial)),), dtype=np.float64
    )
    return file, parameters


class _ItkTransformDataStream(DataStream):
    """An ITK displacement-transform file written region by region.

    A slab of the field maps to one contiguous span of the parameters (the buffer is ``[z][y][x]``
    with the component fastest), so full-width leading-axis slabs (what the streamed write
    dispatcher emits) land with plain offset writes. Under a temporary name until the clean exit,
    like every stream.
    """

    def __init__(self, file: Any, parameters: Any, temporary_path: str, final_path: str, spatial: list[int]) -> None:
        self._h5 = file
        self._parameters = parameters
        self._temporary_path = temporary_path
        self._final_path = final_path
        self.published_path = Path(final_path)
        self._spatial = [int(extent) for extent in spatial]

    def write_slice(self, slices: tuple[slice, ...], data: np.ndarray) -> None:
        channels, leading, *rest = slices
        full = (channels.start or 0) == 0 and channels.stop in (None, 3)
        for axis, part in enumerate(rest, start=2):
            full = full and (part.start or 0) == 0 and part.stop in (None, self._spatial[axis])
        if not full:
            raise DatasetManagerError(
                "A transform file writes full-width leading-axis slabs, and this region is not one.",
                "This is a bug if it was reached: the streamed write dispatcher finalizes full rows.",
            )
        # One buffer: the cast and the transpose are the same pass. Casting first materialises the
        # slab in float64, and ravelling the transposed VIEW of that materialises it again.
        block = np.ascontiguousarray(np.moveaxis(data, 0, -1), dtype=np.float64).ravel()
        offset = 3 * int(leading.start or 0) * int(np.prod(self._spatial[2:], dtype=np.int64))
        self._parameters[offset : offset + block.size] = block

    def _close(self, success: bool) -> None:
        self._h5.close()
        if success:
            os.replace(self._temporary_path, self._final_path)
        else:
            Path(self._temporary_path).unlink(missing_ok=True)


class ItkTransformFile(AbstractFile):
    """ITK transform files, one ``<case>/<group>.h5`` per entry.

    The write side is the point: ``sitk.WriteTransform`` needs the whole field resident in
    float64, where the FILE is three HDF5 datasets that write by regions, so a displacement
    field streams into a transform any ITK consumer (Slicer first) loads. The read side hands
    back what ``Dataset.read_transform`` decodes: a displacement entry carries its field and
    the displacement marker; any other stored transform, the parameter rows and type keys of
    ``_encode_transform_leaves``.

    Needs ``h5py``, as the ``h5`` backend does: the whole point is to touch the parameters
    region by region, and a run whose peak memory turns on whether an optional import
    succeeded is a run nobody can size.
    """

    def __init__(self, filename: str, read: bool) -> None:
        if h5py is None:
            raise DatasetManagerError(
                "An ':itktransform' dataset needs h5py.",
                "Install it with: pip install konfai[hdf5]",
            )
        self.filename = filename
        self.read = read

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        pass

    def _path(self, name: str) -> str:
        for extension in ("h5", "tfm"):
            candidate = f"{self.filename}{name}.{extension}"
            if os.path.exists(candidate):
                return candidate
        return f"{self.filename}{name}.h5"

    def _read_path(self, name: str) -> str:
        """The entry's file for a READ: a missing entry is the structured refusal, never a
        synthesized path sitk.ReadTransform turns into its own RuntimeError. ``is_exist`` keeps
        ``_path``, whose answer for a missing entry is a path that does not exist."""
        path = self._path(name)
        if not os.path.exists(path):
            raise DatasetManagerError(
                f"The entry '{name}' is not in '{self.filename}'.",
                "Check the groups_src spelling and that the case carries every group it names.",
            )
        return path

    def file_to_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
        header = None
        if h5py.is_hdf5(self._read_path(name)):
            with self._field_file(name) as file:
                header = self._field_header(file)
                if header is not None:
                    # The parameters ARE the field: one span off the file, where ITK's transform
                    # reader holds the field twice before the array is even copied out (a 128^3
                    # field: +147 MiB of RSS through ITK, +100 MiB off the span). Read at the
                    # dtype a region read takes, so the two routes carry the same values: the
                    # file keeps ITK's double, the pipeline does not.
                    shape, attributes = header
                    return self._field_region(file, shape[1:], (slice(None),) * 4), attributes
        transform = sitk.ReadTransform(self._read_path(name))
        attributes = Attribute()
        if "DisplacementFieldTransform" in transform.GetName():  # a field in a text transform file
            field = sitk.DisplacementFieldTransform(transform).GetDisplacementField()
            data, attributes = image_to_data(field)
            attributes[DISPLACEMENT_FIELD_ATTRIBUTE] = "true"
            return data, attributes
        leaves = _encode_transform_leaves(transform, name, attributes)
        longest = max(len(leaf) for leaf in leaves)
        return (
            np.asarray([np.pad(leaf, (0, longest - len(leaf)), constant_values=np.nan) for leaf in leaves]),
            attributes,
        )

    def bounded_region_reads(self, name: str) -> bool:
        shape, _attributes = self.get_infos("", name)
        return len(shape) == 4 and shape[0] == 3

    @contextlib.contextmanager
    def _field_file(self, name: str) -> Iterator[Any]:
        """The entry's HDF5 file off the process's read pool: opened once per file, held while
        a region is read, replaced by the pool when the file is rewritten."""
        path = self._read_path(name)
        with _get_h5_file_lock(path):
            yield _h5_read_pool.get(path).file

    @staticmethod
    def _field_header(file: Any) -> tuple[list[int], Attribute] | None:
        """Shape and geometry of a displacement entry off its fixed parameters (size, origin,
        spacing, direction); ``None`` for a transform of another kind, which the whole read decodes."""
        kind = bytes(file["TransformGroup/0/TransformType"][0])
        if not kind.startswith(b"DisplacementFieldTransform"):
            return None
        fixed = np.asarray(file["TransformGroup/0/TransformFixedParameters"][()], dtype=np.float64)
        attributes = Attribute()
        attributes["Origin"] = fixed[3:6]
        attributes["Spacing"] = fixed[6:9]
        attributes["Direction"] = fixed[9:18]
        attributes[DISPLACEMENT_FIELD_ATTRIBUTE] = "true"
        size_xyz = [int(extent) for extent in fixed[0:3]]
        return [3, *size_xyz[::-1]], attributes

    @staticmethod
    def _covered(item: slice, extent: int) -> tuple[int, int, slice]:
        """The ``[low, high)`` range of an axis a slice touches, and the slice of that range it
        takes: what a read of the range then subsamples, whichever way the slice runs."""
        start, stop, step = item.indices(extent)
        count = len(range(start, stop, step))
        if count == 0:
            return 0, 0, slice(0, 0, 1)
        if step > 0:
            return start, start + (count - 1) * step + 1, slice(0, None, step)
        low, high = start + (count - 1) * step, start + 1
        return low, high, slice(high - low - 1, None, step)

    @classmethod
    def _field_region(
        cls, file: Any, spatial: list[int], slices: tuple[slice, ...], dtype: type = np.float32
    ) -> np.ndarray:
        """The region ``slices`` of the field, read as one HDF5 hyperslab of the parameters.

        The buffer is ``[z][y][x]`` with the component fastest, so the rows of one plane the
        region covers are one contiguous span, and the planes it covers are such spans a stride
        apart: a hyperslab of ``count`` blocks reads them into one buffer, the bytes of the
        region's planes and rows and no other (a 64^3 region of a 512^3 field reads 50 MB where
        the leading-axis rows it sits on are 403 MB). A forward step on the leading axis is the
        stride; every other step, and a reversed axis, subsamples the block after the read.
        """
        plane_low, plane_high, planes = cls._covered(slices[1], spatial[0])
        row_low, row_high, rows = cls._covered(slices[2], spatial[1])
        row_length = 3 * int(spatial[2])
        plane_length = row_length * int(spatial[1])
        if planes.step > 0:  # the hyperslab's stride: the planes in between are never read
            stride, count, planes = planes.step, len(range(plane_low, plane_high, planes.step)), slice(None)
        else:
            stride, count = 1, plane_high - plane_low
        block = (row_high - row_low) * row_length
        span = np.empty(count * block, dtype=np.float64)
        if span.size:
            parameters = file["TransformGroup/0/TransformParameters"]
            selection = parameters.id.get_space()
            selection.select_hyperslab(
                (plane_low * plane_length + row_low * row_length,), (count,), (stride * plane_length,), (block,)
            )
            parameters.id.read(h5py.h5s.create_simple((span.size,)), selection, span)
        region = span.reshape(count, row_high - row_low, int(spatial[2]), 3)[planes, rows, slices[3], slices[0]]
        return np.ascontiguousarray(np.moveaxis(region, -1, 0), dtype=dtype)

    def file_to_data_slice(self, group: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        """A region of a displacement entry, decoded from the parameters it maps to alone: the
        header and the region come off one pooled handle, so a region read opens nothing."""
        if h5py.is_hdf5(self._read_path(name)):
            with self._field_file(name) as file:
                header = self._field_header(file)
                if header is not None and len(slices) == 4:
                    shape, attributes = header
                    return self._field_region(file, shape[1:], slices), attributes
        data, attributes = self.file_to_data(group, name)
        return data[slices], attributes

    def data_to_file(
        self,
        name: str,
        data: sitk.Image | sitk.Transform | np.ndarray,
        attributes: Attribute | None = None,
    ) -> None:
        os.makedirs(self.filename, exist_ok=True)
        # Always the `.h5` name: the content is HDF5 and ITK selects its transform IO from the
        # extension, so renaming it onto a resolved existing `.tfm` would corrupt that entry.
        final = os.path.join(self.filename, f"{name}.h5")
        staging = DataStream.staging_path(final)
        if isinstance(data, sitk.Transform):
            sitk.WriteTransform(data, staging)
        else:
            if isinstance(data, sitk.Image):
                data, attributes = image_to_data(data)
            array = np.asarray(data)
            if attributes is None or array.ndim != 4 or array.shape[0] != 3:
                raise DatasetManagerError(
                    f"An ':itktransform' entry is a 3-component 3-D displacement field; '{name}' has"
                    f" shape {list(array.shape)}.",
                    "Write the field itself (channel-first, with its geometry), or a sitk.Transform.",
                )
            spatial = [int(extent) for extent in array.shape[1:]]
            file, parameters = _create_itk_transform_file(staging, spatial, attributes)
            with file:
                # One buffer, as in _ItkTransformDataStream.write_slice.
                parameters[:] = np.ascontiguousarray(np.moveaxis(array, 0, -1), dtype=np.float64).ravel()
        os.replace(staging, final)
        try:  # one entry per name: a `.tfm` left under the same stem would double it
            os.remove(os.path.join(self.filename, f"{name}.tfm"))
        except FileNotFoundError:
            pass

    def open_data_stream(
        self,
        name: str,
        shape: list[int],
        dtype: np.dtype,
        attributes: Attribute,
        region_shape: list[int] | None = None,
    ) -> DataStream | None:
        del dtype, region_shape  # the parameters are float64 whatever arrives, converted per slab
        if len(shape) != 4 or shape[0] != 3 or not is_an_image(attributes):
            return None
        os.makedirs(self.filename, exist_ok=True)
        spatial = [int(extent) for extent in shape[1:]]
        # The `.h5` name, as data_to_file: HDF5 content renamed onto a resolved `.tfm` is a
        # transform ITK reads with its text IO.
        final = os.path.join(self.filename, f"{name}.h5")
        staging = DataStream.staging_path(final)
        file, parameters = _create_itk_transform_file(staging, spatial, attributes)
        return _ItkTransformDataStream(file, parameters, staging, final, [3, *spatial])

    def _entries(self) -> list[str]:
        # Path.glob matches hidden files, so a writer's staging file is filtered out by name.
        return sorted(
            {
                path.stem
                for pattern in ("*.h5", "*.tfm")
                for path in Path(self.filename).glob(pattern)
                if not is_staging_entry(path.name)
            }
        )

    def get_names(self, group: str) -> list[str]:
        del group
        return self._entries()

    def get_group(self) -> list[str]:
        return self._entries()

    def is_exist(self, group: str, name: str | None = None) -> bool:
        return os.path.exists(self._path(name if name else group))

    def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
        # A legacy TEXT transform (`#Insight Transform File V1.0`) is served by the read side
        # too; only a real HDF5 file has the parameter datasets this fast path opens.
        header = None
        if h5py.is_hdf5(self._path(name)):
            with self._field_file(name) as file:
                header = self._field_header(file)
        if header is None:
            data, attributes = self.file_to_data(group, name)
            return [int(extent) for extent in data.shape], attributes
        return header

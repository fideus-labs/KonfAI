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

"""Dataset file abstractions and image conversion utilities for KonfAI."""

from __future__ import annotations

import ast
import copy
import csv
import glob
import math
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lxml import etree  # nosec B410

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]

from konfai import current_date
from konfai.utils.errors import DatasetManagerError
from konfai.utils.utils import SUPPORTED_EXTENSIONS, split_format_level

_h5_file_locks: dict[str, threading.RLock] = {}
_h5_file_locks_guard = threading.Lock()


def _get_h5_file_lock(filename: str) -> threading.RLock:
    """Return the process-wide lock guarding one HDF5 file across worker threads."""
    with _h5_file_locks_guard:
        lock = _h5_file_locks.get(filename)
        if lock is None:
            lock = threading.RLock()
            _h5_file_locks[filename] = lock
        return lock


class Attribute(dict[str, Any]):
    """Metadata container storing repeated values with a stack-like naming scheme."""

    def __init__(self, attributes: dict[str, Any] | None = None) -> None:
        super().__init__()
        attributes = attributes or {}
        for k, v in attributes.items():
            super().__setitem__(copy.deepcopy(k), copy.deepcopy(v))

    def _count_key(self, key: str) -> int:
        return len([k for k in super().keys() if k.startswith(key)])

    def __getitem__(self, key: str) -> Any:
        i = self._count_key(key)
        if i > 0 and f"{key}_{i - 1}" in super().keys():
            return str(super().__getitem__(f"{key}_{i - 1}"))
        if key in super().keys():
            return str(super().__getitem__(key))
        raise NameError(f"{key} not in cache_attribute")

    def __setitem__(self, key: str, value: Any) -> None:
        if "_" not in key:
            i = self._count_key(key)
            result = None
            if isinstance(value, torch.Tensor):
                result = str(value.numpy())
            else:
                result = str(value)
            result = result.replace("\n", "")
            super().__setitem__(f"{key}_{i}", result)
        else:
            result = None
            if isinstance(value, torch.Tensor):
                result = str(value.numpy())
            else:
                result = str(value)
            result = result.replace("\n", "")
            super().__setitem__(key, result)

    def pop(self, key: str, default: Any = None) -> Any:
        i = self._count_key(key)
        if i > 0 and f"{key}_{i - 1}" in super().keys():
            return super().pop(f"{key}_{i - 1}")
        if key in super().keys():
            return super().pop(key)
        raise NameError(f"{key} not in cache_attribute")

    def get_np_array(self, key: str) -> np.ndarray:
        return np.fromstring(self[key][1:-1], sep=" ", dtype=np.double)

    def get_tensor(self, key: str) -> torch.Tensor:
        return torch.tensor(self.get_np_array(key)).to(torch.float32)

    def pop_np_array(self, key: str) -> np.ndarray:
        return np.fromstring(self.pop(key)[1:-1], sep=" ", dtype=np.double)

    def pop_tensor(self, key: str) -> torch.Tensor:
        return torch.tensor(self.pop_np_array(key))

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return any(k.startswith(key) for k in super().keys())

    def is_info(self, key: str, value: str) -> bool:
        return key in self and self[key] == value


def _update_running_statistics(
    state: dict[str, float] | None,
    array: np.ndarray,
) -> dict[str, float]:
    """Update running min/max/mean/std statistics from a NumPy chunk."""
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return state or {"count": 0.0, "mean": 0.0, "m2": 0.0, "min": np.inf, "max": -np.inf}

    if state is None:
        state = {"count": 0.0, "mean": 0.0, "m2": 0.0, "min": np.inf, "max": -np.inf}

    chunk_count = float(values.size)
    chunk_mean = float(values.mean())
    chunk_m2 = float(np.square(values - chunk_mean).sum())

    total_count = state["count"] + chunk_count
    delta = chunk_mean - state["mean"]
    if total_count > 0:
        state["mean"] += delta * chunk_count / total_count
        state["m2"] += chunk_m2 + delta * delta * state["count"] * chunk_count / total_count
        state["count"] = total_count
        state["min"] = min(state["min"], float(values.min()))
        state["max"] = max(state["max"], float(values.max()))
    return state


def _finalize_running_statistics(state: dict[str, float] | None) -> dict[str, float]:
    """Convert a running-statistics state into the public stats dictionary."""
    if state is None or state["count"] == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    variance = state["m2"] / (state["count"] - 1) if state["count"] > 1 else 0.0
    return {
        "min": state["min"],
        "max": state["max"],
        "mean": state["mean"],
        "std": math.sqrt(max(variance, 0.0)),
    }


def is_an_image(attributes: Attribute) -> bool:
    """Return whether the given attribute set contains image geometry metadata."""
    return "Origin" in attributes and "Spacing" in attributes and "Direction" in attributes


def data_to_image(data: np.ndarray, attributes: Attribute) -> sitk.Image:
    """Convert a NumPy array and KonfAI attributes into a SimpleITK image."""
    if not is_an_image(attributes):
        raise NameError("Data is not an image")
    if data.shape[0] == 1:
        image = sitk.GetImageFromArray(data[0])
    else:
        data = data.transpose(tuple([i + 1 for i in range(len(data.shape) - 1)] + [0]))
        image = sitk.GetImageFromArray(data, isVector=True)
    for k, v in attributes.items():
        if v and len(v):
            image.SetMetaData(k, v)
    image.SetOrigin(attributes.get_np_array("Origin").tolist())
    image.SetSpacing(attributes.get_np_array("Spacing").tolist())
    image.SetDirection(attributes.get_np_array("Direction").tolist())
    return image


def image_to_data(image: sitk.Image) -> tuple[np.ndarray, Attribute]:
    """Convert a SimpleITK image into a channel-first NumPy array and attributes."""
    attributes = Attribute()
    attributes["Origin"] = np.asarray(image.GetOrigin())
    attributes["Spacing"] = np.asarray(image.GetSpacing())
    attributes["Direction"] = np.asarray(image.GetDirection())
    for k in image.GetMetaDataKeys():
        attributes[k] = image.GetMetaData(k)
    data = sitk.GetArrayFromImage(image)

    if image.GetNumberOfComponentsPerPixel() == 1:
        data = np.expand_dims(data, 0)
    else:
        data = np.transpose(data, (len(data.shape) - 1, *list(range(len(data.shape) - 1))))
    return data, attributes


def get_infos(filename: str | Path) -> tuple[list[int], Attribute]:
    """Read shape and metadata from an image file without loading its full pixel data."""
    attributes = Attribute()
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(str(filename))
    file_reader.ReadImageInformation()
    attributes["Origin"] = np.asarray(file_reader.GetOrigin())
    attributes["Spacing"] = np.asarray(file_reader.GetSpacing())
    attributes["Direction"] = np.asarray(file_reader.GetDirection())
    for k in file_reader.GetMetaDataKeys():
        attributes[k] = file_reader.GetMetaData(k)
    # SimpleITK GetSize() is (x, y, [z], ...); KonfAI arrays are numpy-order [C, (Z), Y, X], so the
    # spatial size must be reversed for EVERY rank, not only 3-D (2-D/4-D used to come out transposed).
    size = list(reversed(file_reader.GetSize()))
    size = [file_reader.GetNumberOfComponents(), *size]
    return size, attributes


def read_landmarks(filename: Path) -> np.ndarray | None:
    """Read Slicer-style fiducial landmarks from disk."""
    data = None
    with open(filename, newline="") as csvfile:
        reader = csv.reader(filter(lambda row: row[0] != "#", csvfile))
        lines = list(reader)
        data = np.zeros((len(list(lines)), 3), dtype=np.double)
        for i, row in enumerate(lines):
            data[i] = np.array(row[1:4], dtype=np.double)
        csvfile.close()
    return data


def write_landmarks(data: np.ndarray, filename: Path) -> None:
    """Write landmarks to the Slicer Markups fiducial CSV-like format."""
    with open(filename, "w") as f:
        f.write(
            "# Markups fiducial file version = 4.6\n# CoordinateSystem = LPS\n#"
            " columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n",
        )
        for i in range(data.shape[0]):
            f.write(
                "vtkMRMLMarkupsFiducialNode_"
                + str(i + 1)
                + ","
                + str(data[i, 0])
                + ","
                + str(data[i, 1])
                + ","
                + str(data[i, 2])
                + ",0,0,0,1,1,1,0,F-"
                + str(i + 1)
                + ",,vtkMRMLScalarVolumeNode1\n"
            )
        f.close()


class Dataset:
    """Filesystem or HDF5-backed dataset abstraction used across KonfAI."""

    class AbstractFile(ABC):
        @abstractmethod
        def __init__(self) -> None:
            pass

        @abstractmethod
        def __enter__(self):
            pass

        @abstractmethod
        def __exit__(self, exc_type, value, traceback):
            pass

        @abstractmethod
        def file_to_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
            pass

        @abstractmethod
        def file_to_data_slice(self, group: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
            pass

        @abstractmethod
        def file_to_data_statistics(
            self,
            group: str,
            name: str,
            channels: list[int] | None = None,
        ) -> dict[str, float]:
            pass

        @abstractmethod
        def data_to_file(
            self,
            name: str,
            data: sitk.Image | sitk.Transform | np.ndarray,
            attributes: Attribute | None = None,
        ) -> None:
            pass

        @abstractmethod
        def get_names(self, group: str) -> list[str]:
            pass

        @abstractmethod
        def get_group(self) -> list[str]:
            pass

        @abstractmethod
        def is_exist(self, group: str, name: str | None = None) -> bool:
            pass

        @abstractmethod
        def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
            pass

    class H5File(AbstractFile):
        def __init__(self, filename: str, read: bool) -> None:
            self.h5: h5py.File | None = None
            self.filename = filename
            if not self.filename.endswith(".h5"):
                self.filename += ".h5"
            self.read = read
            self._lock: threading.RLock | None = None

        def __enter__(self):
            # A single HDF5 file cannot be opened concurrently from several threads:
            # the whole open/use/close sequence is serialised per file so that two
            # cache workers never race between the existence check and the "w"/"r+"
            # open (which would truncate each other's data).
            self._lock = _get_h5_file_lock(self.filename)
            self._lock.acquire()
            try:
                if self.read:
                    self.h5 = h5py.File(self.filename, "r")
                else:
                    if not os.path.exists(self.filename):
                        Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
                        self.h5 = h5py.File(self.filename, "w")
                    else:
                        self.h5 = h5py.File(self.filename, "r+")
                    self.h5.attrs["Date"] = current_date()
                self.h5.__enter__()
            except BaseException:
                self._lock.release()
                self._lock = None
                raise
            return self.h5

        def __exit__(self, exc_type, value, traceback):
            try:
                if self.h5 is not None:
                    self.h5.close()
            finally:
                if self._lock is not None:
                    self._lock.release()
                    self._lock = None

        def file_to_data(self, groups: str, name: str) -> tuple[np.ndarray, Attribute]:
            dataset = self._get_dataset(groups, name)
            data = np.zeros(dataset.shape, dataset.dtype)
            dataset.read_direct(data)
            return data, Attribute({k: str(v) for k, v in dataset.attrs.items()})

        def file_to_data_slice(self, groups: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
            dataset = self._get_dataset(groups, name)
            data = np.asarray(dataset[slices])
            return data, Attribute({k: str(v) for k, v in dataset.attrs.items()})

        def file_to_data_statistics(
            self,
            groups: str,
            name: str,
            channels: list[int] | None = None,
        ) -> dict[str, float]:
            dataset = self._get_dataset(groups, name)
            if dataset is None:
                raise NameError(f"Dataset '{groups}/{name}' not found in '{self.filename}'.")

            axis = 1 if dataset.ndim > 1 else 0
            trailing_size = int(np.prod(dataset.shape[axis + 1 :], dtype=np.int64)) if axis + 1 < dataset.ndim else 1
            max_elements = 8_000_000
            chunk_length = max(1, max_elements // max(1, trailing_size))
            state: dict[str, float] | None = None

            for start in range(0, dataset.shape[axis], chunk_length):
                slices = [slice(None)] * dataset.ndim
                slices[axis] = slice(start, min(dataset.shape[axis], start + chunk_length))
                chunk = np.asarray(dataset[tuple(slices)])
                if channels is not None:
                    chunk = chunk[channels]
                state = _update_running_statistics(state, chunk)

            return _finalize_running_statistics(state)

        def data_to_file(
            self,
            name: str,
            data: sitk.Image | sitk.Transform | np.ndarray,
            attributes: Attribute | None = None,
        ) -> None:
            if self.h5 is None:
                return
            if attributes is None:
                attributes = Attribute()
            if isinstance(data, sitk.Image):
                data, attributes_tmp = image_to_data(data)
                attributes.update(attributes_tmp)
            elif isinstance(data, sitk.Transform):
                transforms = []
                if isinstance(data, sitk.CompositeTransform):
                    for i in range(data.GetNumberOfTransforms()):
                        transforms.append(data.GetNthTransform(i))
                else:
                    transforms.append(data)
                datas = []
                for i, transform in enumerate(transforms):
                    if isinstance(transform, sitk.Euler3DTransform):
                        transform_type = "Euler3DTransform_double_3_3"
                    elif isinstance(transform, sitk.AffineTransform):
                        transform_type = "AffineTransform_double_3_3"
                    elif isinstance(transform, sitk.BSplineTransform):
                        transform_type = "BSplineTransform_double_3_3"
                    else:
                        raise DatasetManagerError(
                            f"Unsupported transform type '{type(transform).__name__}' for entry '{name}'."
                        )
                    attributes[f"{i}:Transform"] = transform_type
                    attributes[f"{i}:FixedParameters"] = transform.GetFixedParameters()

                    datas.append(np.asarray(transform.GetParameters()))
                data = np.asarray(datas)

            h5_group = self.h5
            if len(name.split("/")) > 1:
                group = "/".join(name.split("/")[:-1])
                if group not in self.h5:
                    self.h5.create_group(group)
                h5_group = self.h5[group]

            name = name.split("/")[-1]
            if name in h5_group:
                del h5_group[name]

            dataset = h5_group.create_dataset(name, data=data, dtype=data.dtype, chunks=None)
            dataset.attrs.update({k: str(v) for k, v in attributes.items()})

        def is_exist(self, group: str, name: str | None = None) -> bool:
            if self.h5 is not None:
                if group in self.h5:
                    if isinstance(self.h5[group], h5py.Dataset):
                        return True
                    elif name is not None:
                        return name in self.h5[group]
                    else:
                        return False
            return False

        def get_names(self, groups: str, h5_group: h5py.Group = None) -> list[str]:
            names = []
            if h5_group is None:
                h5_group = self.h5
            group = groups.split("/")[0]
            if group == "":
                names = [
                    dataset.name.split("/")[-1] for dataset in h5_group.values() if isinstance(dataset, h5py.Dataset)
                ]
            elif group == "*":
                for k in h5_group.keys():
                    if isinstance(h5_group[k], h5py.Group):
                        names.extend(self.get_names("/".join(groups.split("/")[1:]), h5_group[k]))
            else:
                if group in h5_group:
                    names.extend(self.get_names("/".join(groups.split("/")[1:]), h5_group[group]))
            return names

        def get_group(self) -> list[str]:
            return list(self.h5.keys()) if self.h5 is not None else []

        def _get_dataset(self, groups: str, name: str, h5_group: h5py.Group = None) -> h5py.Dataset:
            if h5_group is None:
                h5_group = self.h5
            if groups != "":
                group = groups.split("/")[0]
            else:
                group = ""
            result = None
            if group == "":
                if name in h5_group:
                    result = h5_group[name]
            elif group == "*":
                for k in h5_group.keys():
                    if isinstance(h5_group[k], h5py.Group):
                        result_tmp = self._get_dataset("/".join(groups.split("/")[1:]), name, h5_group[k])
                        if result_tmp is not None:
                            result = result_tmp
            else:
                if group in h5_group:
                    result_tmp = self._get_dataset("/".join(groups.split("/")[1:]), name, h5_group[group])
                    if result_tmp is not None:
                        result = result_tmp
            return result

        def get_infos(self, groups: str, name: str) -> tuple[list[int], Attribute]:
            dataset = self._get_dataset(groups, name)
            return (
                dataset.shape,
                Attribute({k: str(v) for k, v in dataset.attrs.items()}),
            )

    class SitkFile(AbstractFile):
        def __init__(self, filename: str, read: bool, file_format: str) -> None:
            self.filename = filename
            self.read = read
            self.file_format = file_format

        @staticmethod
        def _normalize_slices(slices: tuple[slice, ...], shape: list[int]) -> tuple[slice, ...]:
            if len(slices) != len(shape):
                raise ValueError(f"Expected {len(shape)} slices, got {len(slices)}.")

            normalized = []
            for item, size in zip(slices, shape, strict=False):
                start, stop, step = item.indices(size)
                normalized.append(slice(start, stop, step))
            return tuple(normalized)

        @staticmethod
        def _supports_direct_slice(slices: tuple[slice, ...]) -> bool:
            return all(item.step in (None, 1) for item in slices)

        def _resolve_data_path(self, name: str) -> str | None:
            base = f"{self.filename}{name}"
            for suffix in (".itk.txt", ".fcsv", ".xml", ".vtk", ".npy"):
                candidate = f"{base}{suffix}"
                if os.path.exists(candidate):
                    return candidate

            direct = f"{base}.{self.file_format}"
            if os.path.exists(direct):
                return direct

            matches = glob.glob(f"{base}.*")
            return matches[0] if matches else None

        def _file_to_image_slice(self, name: str, path: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
            reader = sitk.ImageFileReader()
            reader.SetFileName(path)
            reader.ReadImageInformation()

            spatial_size_xyz = list(reader.GetSize())
            spatial_shape = list(reversed(spatial_size_xyz))
            data_shape = [reader.GetNumberOfComponents(), *spatial_shape]
            normalized = self._normalize_slices(slices, data_shape)

            if not self._supports_direct_slice(normalized):
                data, attributes = self.file_to_data("", name)
                return data[normalized], attributes

            extract_index_xyz = [item.start for item in reversed(normalized[1:])]
            extract_size_xyz = [item.stop - item.start for item in reversed(normalized[1:])]
            reader.SetExtractIndex(extract_index_xyz)
            reader.SetExtractSize(extract_size_xyz)

            image = reader.Execute()
            data, attributes = image_to_data(image)
            origin = np.asarray(reader.GetOrigin(), dtype=np.float64)
            spacing = np.asarray(reader.GetSpacing(), dtype=np.float64)
            direction = np.asarray(reader.GetDirection(), dtype=np.float64).reshape(len(spacing), len(spacing))
            attributes["Origin"] = origin + direction @ (np.asarray(extract_index_xyz, dtype=np.float64) * spacing)
            return data[normalized[:1] + tuple(slice(None) for _ in normalized[1:])], attributes

        def file_to_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
            attributes = Attribute()
            if os.path.exists(f"{self.filename}{name}.itk.txt"):
                data = sitk.ReadTransform(f"{self.filename}{name}.itk.txt")
                transforms = []
                if isinstance(data, sitk.CompositeTransform):
                    for i in range(data.GetNumberOfTransforms()):
                        transforms.append(data.GetNthTransform(i))
                else:
                    transforms.append(data)
                datas = []
                for i, transform in enumerate(transforms):
                    if isinstance(transform, sitk.Euler3DTransform):
                        transform_type = "Euler3DTransform_double_3_3"
                    elif isinstance(transform, sitk.AffineTransform):
                        transform_type = "AffineTransform_double_3_3"
                    elif isinstance(transform, sitk.BSplineTransform):
                        transform_type = "BSplineTransform_double_3_3"
                    else:
                        raise DatasetManagerError(
                            f"Unsupported transform type '{type(transform).__name__}' for entry '{name}'."
                        )
                    attributes[f"{i}:Transform"] = transform_type
                    attributes[f"{i}:FixedParameters"] = transform.GetFixedParameters()

                    datas.append(np.asarray(transform.GetParameters()))

                max_len = max(len(v) for v in datas)

                padded_datas = np.array([np.pad(v, (0, max_len - len(v)), constant_values=np.nan) for v in datas])

                data = np.asarray(padded_datas)
            elif os.path.exists(f"{self.filename}{name}.fcsv"):
                data = read_landmarks(Path(f"{self.filename}{name}.fcsv"))
            elif os.path.exists(f"{self.filename}{name}.xml"):
                with open(f"{self.filename}{name}.xml", "rb") as xml_file:
                    root = etree.parse(xml_file, etree.XMLParser(remove_blank_text=True)).getroot()  # nosec B320
                node = root
                while len(node):
                    node = node[-1]
                for key, value in node.attrib.items():
                    attributes[key] = value
                text = (node.text or "").strip()
                data = np.fromstring(text, sep=",", dtype=np.float64) if text else np.asarray([], dtype=np.float64)
            elif os.path.exists(f"{self.filename}{name}.vtk"):
                import vtk

                vtk_reader = vtk.vtkPolyDataReader()
                vtk_reader.SetFileName(f"{self.filename}{name}.vtk")
                vtk_reader.Update()
                data = []
                points = vtk_reader.GetOutput().GetPoints()
                num_points = points.GetNumberOfPoints()
                for i in range(num_points):
                    data.append(list(points.GetPoint(i)))
                data = np.asarray(data)
            elif os.path.exists(f"{self.filename}{name}.npy"):
                data = np.load(f"{self.filename}{name}.npy")
            else:
                pattern = f"{self.filename}{name}.*"
                matches = glob.glob(pattern)
                if not matches:
                    raise NameError(f"Data '{name}' not found in dataset '{self.filename}'.")
                path = matches[0]
                image = sitk.ReadImage(path)
                data, attributes_tmp = image_to_data(image)
                attributes.update(attributes_tmp)
            return data, attributes

        def file_to_data_slice(self, group: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
            path = self._resolve_data_path(name)
            if path is None:
                raise NameError(f"Data '{name}' not found in dataset '{self.filename}'.")

            if path.endswith(".npy"):
                data = np.load(path, mmap_mode="r")[slices]
                return np.asarray(data), Attribute()

            if path.endswith((".itk.txt", ".fcsv", ".xml", ".vtk")):
                data, attributes = self.file_to_data(group, name)
                return data[slices], attributes

            return self._file_to_image_slice(name, path, slices)

        def file_to_data_statistics(
            self,
            group: str,
            name: str,
            channels: list[int] | None = None,
        ) -> dict[str, float]:
            path = self._resolve_data_path(name)
            if path is None:
                raise NameError(f"Data '{name}' not found in dataset '{self.filename}'.")

            if path.endswith(".npy"):
                data = np.load(path, mmap_mode="r")
                if channels is not None:
                    data = data[channels]
                return _finalize_running_statistics(_update_running_statistics(None, data))

            if path.endswith((".itk.txt", ".fcsv", ".xml", ".vtk")):
                data, _ = self.file_to_data(group, name)
                if channels is not None:
                    data = data[channels]
                return _finalize_running_statistics(_update_running_statistics(None, data))

            image = sitk.ReadImage(path)
            data = sitk.GetArrayViewFromImage(image)
            if image.GetNumberOfComponentsPerPixel() == 1:
                data = np.expand_dims(data, 0)
            else:
                data = np.transpose(data, (len(data.shape) - 1, *list(range(len(data.shape) - 1))))
            if channels is not None:
                data = data[channels]
            return _finalize_running_statistics(_update_running_statistics(None, data))

        def is_vtk_polydata(self, obj) -> bool:
            try:
                import vtk

                return isinstance(obj, vtk.vtkPolyData)
            except ImportError:
                return False

        def __enter__(self):
            pass

        def __exit__(self, exc_type, value, traceback):
            pass

        def data_to_file(
            self,
            name: str,
            data: sitk.Image | sitk.Transform | np.ndarray,
            attributes: Attribute | None = None,
        ) -> None:
            if attributes is None:
                attributes = Attribute()
            os.makedirs(self.filename, exist_ok=True)
            if isinstance(data, sitk.Image):
                for k, v in attributes.items():
                    if v and len(v):
                        data.SetMetaData(k, v)
                sitk.WriteImage(data, f"{self.filename}{name}.{self.file_format}")
            elif isinstance(data, sitk.Transform):
                sitk.WriteTransform(data, f"{self.filename}{name}.itk.txt")
            elif self.is_vtk_polydata(data):
                import vtk

                vtk_writer = vtk.vtkPolyDataWriter()
                vtk_writer.SetFileName(f"{self.filename}{name}.vtk")
                vtk_writer.SetInputData(data)
                vtk_writer.Write()
            elif is_an_image(attributes):
                self.data_to_file(name, data_to_image(data, attributes), attributes)
            elif len(data.shape) == 2 and data.shape[1] == 3 and data.shape[0] > 0:
                data = np.round(data, 4)
                write_landmarks(data, Path(f"{self.filename}{name}.fcsv"))
            elif "path" in attributes:
                if os.path.exists(f"{self.filename}{name}.xml"):
                    with open(f"{self.filename}{name}.xml", "rb") as xml_file:
                        root = etree.parse(xml_file, etree.XMLParser(remove_blank_text=True)).getroot()  # nosec B320
                        xml_file.close()
                else:
                    root = etree.Element(name)
                node = root
                path = attributes["path"].split(":")

                for node_name in path:
                    node_tmp = node.find(node_name)
                    if node_tmp is None:
                        node_tmp = etree.SubElement(node, node_name)
                        node.append(node_tmp)
                    node = node_tmp
                if attributes is not None:
                    for attribute_tmp in attributes.keys():
                        attribute = "_".join(attribute_tmp.split("_")[:-1])
                        if attribute != "path":
                            node.set(attribute, attributes[attribute])
                if data.size > 0:
                    node.text = ", ".join(
                        map(str, data.flatten())
                    )  # np.array2string(data, separator=',')[1:-1].replace('\n','')
                with open(f"{self.filename}{name}.xml", "wb") as f:
                    f.write(etree.tostring(root, pretty_print=True, encoding="utf-8"))
                    f.close()
            else:
                np.save(f"{self.filename}{name}.npy", data)

        def is_exist(self, group: str, name: str | None = None) -> bool:
            base = f"{self.filename}{group}"
            return any(os.path.exists(base + "." + ext) for ext in SUPPORTED_EXTENSIONS)

        def get_names(self, group: str) -> list[str]:
            raise NotImplementedError()

        def get_group(self) -> list[str]:
            raise NotImplementedError()

        def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
            attributes = Attribute()
            if os.path.exists(f"{self.filename}{group if group is not None else ''}{name}.{self.file_format}"):
                file_reader = sitk.ImageFileReader()
                file_reader.SetFileName(f"{self.filename}{group if group is not None else ''}{name}.{self.file_format}")
                file_reader.ReadImageInformation()
                attributes["Origin"] = np.asarray(file_reader.GetOrigin())
                attributes["Spacing"] = np.asarray(file_reader.GetSpacing())
                attributes["Direction"] = np.asarray(file_reader.GetDirection())
                for k in file_reader.GetMetaDataKeys():
                    attributes[k] = file_reader.GetMetaData(k)
                # Reverse for every rank (see the module-level get_infos): 2-D/4-D used to be transposed.
                size = list(reversed(file_reader.GetSize()))
                size = [file_reader.GetNumberOfComponents(), *size]
            else:
                data, attributes = self.file_to_data(group if group is not None else "", name)
                size = data.shape
            return size, attributes

    class OmeZarrFile(AbstractFile):
        """OME-NGFF backend using chunked Zarr reads for KonfAI patches.

        ``level`` selects the multiscale pyramid resolution to read (0 = full
        resolution, higher = coarser); it comes from the ``omezarr@<level>``
        dataset-spec suffix.
        """

        def __init__(self, filename: str, read: bool, level: int = 0) -> None:
            self.filename = filename if filename.endswith("/") else f"{filename}/"
            self.read = read
            self.level = level

        def __enter__(self):
            return self

        def __exit__(self, exc_type, value, traceback):
            return None

        def _path(self, name: str, *, writing: bool = False) -> Path:
            base = Path(self.filename) / name
            if writing:
                return Path(f"{base}.ome.zarr")
            candidates = [Path(f"{base}.ome.zarr"), Path(f"{base}.zarr"), base]
            for candidate in candidates:
                if candidate.is_dir():
                    return candidate
            raise NameError(f"OME-Zarr group '{name}' not found in '{self.filename}'.")

        @staticmethod
        def _attributes(metadata: dict[str, Any]) -> Attribute:
            attributes = Attribute(metadata.get("attributes", {}))
            axes = metadata["axes"]
            scale = dict(zip(axes, metadata.get("scale", []), strict=False))
            translation = dict(zip(axes, metadata.get("translation", []), strict=False))
            spatial_axes = [axis for axis in ("x", "y", "z") if axis in axes]
            if "Spacing" not in attributes:
                attributes["Spacing"] = np.asarray([scale.get(axis, 1.0) for axis in spatial_axes])
            if "Origin" not in attributes:
                attributes["Origin"] = np.asarray([translation.get(axis, 0.0) for axis in spatial_axes])
            if "Direction" not in attributes:
                attributes["Direction"] = np.eye(len(spatial_axes), dtype=np.float64).flatten()
            attributes["OMEAxes"] = np.asarray(axes)
            return attributes

        def file_to_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
            info_shape, _ = self.get_infos(group, name)
            return self.file_to_data_slice(group, name, tuple(slice(None) for _ in info_shape))

        def file_to_data_slice(self, group: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
            from konfai.utils.ome_zarr import read_ome_zarr_data_slice

            path = self._path(name)
            data, metadata = read_ome_zarr_data_slice(path, slices, level=self.level)
            attributes = self._attributes(metadata)
            shape = metadata["shape"]
            normalized = tuple(slice(*item.indices(size)) for item, size in zip(slices, shape, strict=True))
            spacing = attributes.get_np_array("Spacing")
            direction = attributes.get_np_array("Direction").reshape(len(spacing), len(spacing))
            start_xyz = np.asarray([item.start for item in reversed(normalized[1:])], dtype=np.float64)
            step_xyz = np.asarray([item.step for item in reversed(normalized[1:])], dtype=np.float64)
            attributes["Origin"] = attributes.get_np_array("Origin") + direction @ (start_xyz * spacing)
            attributes["Spacing"] = spacing * step_xyz
            return data, attributes

        def file_to_data_statistics(
            self,
            group: str,
            name: str,
            channels: list[int] | None = None,
        ) -> dict[str, float]:
            shape, _ = self.get_infos(group, name)
            trailing_size = int(np.prod(shape[2:], dtype=np.int64)) if len(shape) > 2 else 1
            chunk_length = max(1, 8_000_000 // max(1, trailing_size))
            state: dict[str, float] | None = None
            for start in range(0, shape[1], chunk_length):
                slices = [slice(None)] * len(shape)
                slices[1] = slice(start, min(shape[1], start + chunk_length))
                chunk, _ = self.file_to_data_slice(group, name, tuple(slices))
                if channels is not None:
                    chunk = chunk[channels]
                state = _update_running_statistics(state, chunk)
            return _finalize_running_statistics(state)

        def data_to_file(
            self,
            name: str,
            data: sitk.Image | sitk.Transform | np.ndarray,
            attributes: Attribute | None = None,
        ) -> None:
            from konfai.utils.ome_zarr import write_ome_zarr

            attributes = attributes or Attribute()
            if sitk is not None and isinstance(data, sitk.Image):
                data, image_attributes = image_to_data(data)
                attributes.update(image_attributes)
            if not isinstance(data, np.ndarray):
                raise DatasetManagerError("OME-Zarr datasets can only store image arrays.")
            dimension = data.ndim - 1
            spacing = attributes.get_np_array("Spacing") if "Spacing" in attributes else np.ones(dimension)
            origin = attributes.get_np_array("Origin") if "Origin" in attributes else np.zeros(dimension)
            write_ome_zarr(
                self._path(name, writing=True),
                data,
                spacing=spacing,
                origin=origin,
                attributes=dict(attributes),
            )

        def get_names(self, group: str) -> list[str]:
            return self.get_group()

        def get_group(self) -> list[str]:
            root = Path(self.filename)
            if not root.is_dir():
                return []
            groups = []
            for path in root.iterdir():
                if path.name.endswith(".ome.zarr"):
                    groups.append(path.name.removesuffix(".ome.zarr"))
                elif path.name.endswith(".zarr"):
                    groups.append(path.name.removesuffix(".zarr"))
            return sorted(groups)

        def is_exist(self, group: str, name: str | None = None) -> bool:
            try:
                self._path(f"{group}/{name}" if name else group)
                return True
            except NameError:
                return False

        def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
            from konfai.utils.ome_zarr import get_ome_zarr_info

            metadata = get_ome_zarr_info(self._path(name), level=self.level)
            axes = [str(axis).lower() for axis in metadata["axes"]]
            axis_sizes = dict(zip(axes, metadata["shape"], strict=True))
            shape = [axis_sizes.get("c", 1), *[axis_sizes[axis] for axis in ("z", "y", "x") if axis in axis_sizes]]
            metadata["shape"] = shape
            return shape, self._attributes(metadata)

    class DicomFile(AbstractFile):
        """DICOM series backend with header-only metadata and slice-level reads."""

        def __init__(self, filename: str, read: bool) -> None:
            self.filename = filename if filename.endswith("/") else f"{filename}/"
            self.read = read

        def __enter__(self):
            return self

        def __exit__(self, exc_type, value, traceback):
            return None

        def _path(self, name: str) -> Path:
            return Path(self.filename) / name

        @staticmethod
        def _attributes(info: dict[str, Any]) -> Attribute:
            attributes = Attribute()
            attributes["Origin"] = np.asarray(info["origin"])
            attributes["Spacing"] = np.asarray(info["spacing"])
            attributes["Direction"] = np.asarray(info["direction"])
            attributes["SeriesInstanceUID"] = info["series_uid"]
            return attributes

        def file_to_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
            from konfai.utils.dicom import read_dicom_series

            data, origin, spacing, direction = read_dicom_series(self._path(name))
            attributes = Attribute()
            attributes["Origin"] = origin
            attributes["Spacing"] = spacing
            attributes["Direction"] = direction
            return data, attributes

        def file_to_data_slice(self, group: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
            from konfai.utils.dicom import get_dicom_info, read_dicom_series_slice

            path = self._path(name)
            info = get_dicom_info(path)
            data, origin, spacing, direction = read_dicom_series_slice(path, slices, series_uid=info["series_uid"])
            info.update(origin=origin, spacing=spacing, direction=direction)
            return data, self._attributes(info)

        def file_to_data_statistics(
            self,
            group: str,
            name: str,
            channels: list[int] | None = None,
        ) -> dict[str, float]:
            shape, _ = self.get_infos(group, name)
            state: dict[str, float] | None = None
            for index in range(shape[1]):
                chunk, _ = self.file_to_data_slice(
                    group,
                    name,
                    (slice(None), slice(index, index + 1), slice(None), slice(None)),
                )
                if channels is not None:
                    chunk = chunk[channels]
                state = _update_running_statistics(state, chunk)
            return _finalize_running_statistics(state)

        def data_to_file(
            self,
            name: str,
            data: sitk.Image | sitk.Transform | np.ndarray,
            attributes: Attribute | None = None,
        ) -> None:
            from konfai.utils.dicom import write_dicom_series

            attributes = attributes or Attribute()
            if sitk is not None and isinstance(data, sitk.Image):
                data, image_attributes = image_to_data(data)
                attributes.update(image_attributes)
            if not isinstance(data, np.ndarray):
                raise DatasetManagerError("DICOM datasets can only store scalar image arrays.")
            spacing = attributes.get_np_array("Spacing") if "Spacing" in attributes else np.ones(3)
            origin = attributes.get_np_array("Origin") if "Origin" in attributes else np.zeros(3)
            direction = attributes.get_np_array("Direction") if "Direction" in attributes else np.eye(3).flatten()
            metadata = {
                key: attributes[key]
                for key in ("PatientName", "PatientID", "Modality", "StudyInstanceUID", "SeriesInstanceUID")
                if key in attributes
            }
            write_dicom_series(
                self._path(name),
                data,
                spacing=spacing,
                origin=origin,
                direction=direction,
                metadata=metadata,
            )

        def get_names(self, group: str) -> list[str]:
            return self.get_group()

        def get_group(self) -> list[str]:
            root = Path(self.filename)
            if not root.is_dir():
                return []
            return sorted(path.name for path in root.iterdir() if path.is_dir() and self.is_exist(path.name))

        def is_exist(self, group: str, name: str | None = None) -> bool:
            from konfai.utils.dicom import get_dicom_info

            try:
                get_dicom_info(self._path(f"{group}/{name}" if name else group))
                return True
            except DatasetManagerError:
                return False

        def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
            from konfai.utils.dicom import get_dicom_info

            info = get_dicom_info(self._path(name))
            return info["shape"], self._attributes(info)

    class File:
        def __init__(self, filename: str, read: bool, file_format: str, level: int = 0) -> None:
            self.filename = filename
            self.read = read
            self.file: Dataset.AbstractFile | None = None
            self.file_format = file_format
            self.level = level

        def __enter__(self) -> Dataset.AbstractFile:
            if self.file_format == "h5":
                self.file = Dataset.H5File(self.filename, self.read)
            elif self.file_format == "omezarr":
                self.file = Dataset.OmeZarrFile(self.filename, self.read, self.level)
            elif self.file_format == "dicom":
                self.file = Dataset.DicomFile(self.filename, self.read)
            else:
                self.file = Dataset.SitkFile(self.filename + "/", self.read, self.file_format)
            self.file.__enter__()
            return self.file

        def __exit__(self, exc_type, value, traceback):
            if self.file is not None:
                self.file.__exit__(exc_type, value, traceback)

    def __init__(self, filename: str | Path, file_format: str) -> None:
        base_format, self.level = split_format_level(file_format)
        normalized_format = base_format.lower().removeprefix(".").replace("_", "-")
        file_format = {"ome-zarr": "omezarr", "zarr": "omezarr"}.get(normalized_format, normalized_format)
        if file_format != "h5" and not str(filename).endswith("/"):
            filename = f"{filename}/"
        self.is_directory = str(filename).endswith("/")
        self.filename = str(filename)
        self.file_format = file_format
        self._names_cache: dict[str, list[str]] = {}

    def _exists_on_disk(self) -> bool:
        if os.path.exists(self.filename):
            return True
        return self.file_format == "h5" and os.path.exists(f"{self.filename}.h5")

    def write(
        self,
        group: str,
        name: str,
        data: sitk.Image | sitk.Transform | np.ndarray,
        attributes: Attribute | None = None,
    ) -> None:
        self._names_cache.clear()
        if attributes is None:
            attributes = Attribute()
        if self.is_directory:
            os.makedirs(self.filename, exist_ok=True)
        if self.is_directory:
            s_group = group.split("/")
            if len(s_group) > 1:
                sub_directory = "/".join(s_group[:-1])
                name = f"{sub_directory}/{name}"
                group = s_group[-1]
            with Dataset.File(f"{self.filename}{name}", False, self.file_format, self.level) as file:
                file.data_to_file(group, data, attributes)
        else:
            with Dataset.File(self.filename, False, self.file_format, self.level) as file:
                file.data_to_file(f"{group}/{name}", data, attributes)

    def read_data(self, groups: str, name: str) -> tuple[np.ndarray, Attribute]:
        if not self._exists_on_disk():
            raise NameError(f"Dataset {self.filename} not found")
        if self.is_directory:
            for sub_directory in self._get_sub_directories(groups):
                group = groups.split("/")[-1]
                if os.path.exists(f"{self.filename}{sub_directory}{name}{'.h5' if self.file_format == 'h5' else ''}"):
                    with Dataset.File(
                        f"{self.filename}{sub_directory}{name}",
                        True,
                        self.file_format,
                        self.level,
                    ) as file:
                        return file.file_to_data("", group)
        else:
            with Dataset.File(self.filename, True, self.file_format, self.level) as file:
                return file.file_to_data(groups, name)
        raise NameError(f"Dataset entry '{groups}/{name}' not found in {self.filename}.")

    def read_data_slice(self, groups: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        if not self._exists_on_disk():
            raise NameError(f"Dataset {self.filename} not found")
        if self.is_directory:
            for sub_directory in self._get_sub_directories(groups):
                group = groups.split("/")[-1]
                if os.path.exists(f"{self.filename}{sub_directory}{name}{'.h5' if self.file_format == 'h5' else ''}"):
                    with Dataset.File(
                        f"{self.filename}{sub_directory}{name}",
                        True,
                        self.file_format,
                        self.level,
                    ) as file:
                        result = file.file_to_data_slice("", group, slices)
                        return result
        else:
            with Dataset.File(self.filename, True, self.file_format, self.level) as file:
                return file.file_to_data_slice(groups, name, slices)

        raise NameError(f"Dataset entry '{groups}/{name}' not found in {self.filename}.")

    def read_data_statistics(
        self,
        groups: str,
        name: str,
        channels: list[int] | None = None,
    ) -> dict[str, float]:
        if not self._exists_on_disk():
            raise NameError(f"Dataset {self.filename} not found")
        if self.is_directory:
            for sub_directory in self._get_sub_directories(groups):
                group = groups.split("/")[-1]
                if os.path.exists(f"{self.filename}{sub_directory}{name}{'.h5' if self.file_format == 'h5' else ''}"):
                    with Dataset.File(
                        f"{self.filename}{sub_directory}{name}",
                        True,
                        self.file_format,
                        self.level,
                    ) as file:
                        return file.file_to_data_statistics("", group, channels)
        else:
            with Dataset.File(self.filename, True, self.file_format, self.level) as file:
                return file.file_to_data_statistics(groups, name, channels)

        raise NameError(f"Dataset entry '{groups}/{name}' not found in {self.filename}.")

    def read_transform(self, group: str, name: str) -> sitk.Transform:
        if not self._exists_on_disk():
            raise NameError(f"Dataset {self.filename} not found")
        transform_parameters, attribute = self.read_data(group, name)
        transforms_type = [v for k, v in attribute.items() if k.endswith(":Transform_0")]
        transforms = []
        for i, transform_type in enumerate(transforms_type):
            if transform_type == "Euler3DTransform_double_3_3":
                transform = sitk.Euler3DTransform()
            elif transform_type == "AffineTransform_double_3_3":
                transform = sitk.AffineTransform(3)
            elif transform_type == "BSplineTransform_double_3_3":
                transform = sitk.BSplineTransform(3)
            else:
                raise DatasetManagerError(f"Unsupported transform type '{transform_type}' for entry '{name}'.")
            transform.SetFixedParameters(ast.literal_eval(attribute[f"{i}:FixedParameters"]))
            transform.SetParameters(tuple(transform_parameters[i]))
            transforms.append(transform)
        return sitk.CompositeTransform(transforms) if len(transforms) > 1 else transforms[0]

    def read_image(self, group: str, name: str) -> sitk.Image:
        data, attribute = self.read_data(group, name)
        return data_to_image(data, attribute)

    def get_size(self, group: str) -> int:
        return len(self.get_names(group))

    def is_group_exist(self, group: str) -> bool:
        return self.get_size(group) > 0

    def is_dataset_exist(self, group: str, name: str) -> bool:
        return name in self.get_names(group)

    def _get_sub_directories(self, groups: str, sub_directory: str = ""):
        group = groups.split("/")[0]
        sub_directories = []
        if len(groups.split("/")) == 1:
            sub_directories.append(sub_directory)
        elif group == "*":
            for k in os.listdir(f"{self.filename}{sub_directory}"):
                if not os.path.isfile(f"{self.filename}{sub_directory}{k}"):
                    sub_directories.extend(
                        self._get_sub_directories(
                            "/".join(groups.split("/")[1:]),
                            f"{sub_directory}{k}/",
                        )
                    )
        else:
            sub_directory = f"{sub_directory}{group}/"
            if os.path.exists(f"{self.filename}{sub_directory}"):
                sub_directories.extend(self._get_sub_directories("/".join(groups.split("/")[1:]), sub_directory))
        return sub_directories

    def get_names(self, groups: str, index: list[int] | None = None) -> list[str]:
        if index is None and groups in self._names_cache:
            return self._names_cache[groups]

        names = []
        if self.is_directory:
            for sub_directory in self._get_sub_directories(groups):
                group = groups.split("/")[-1]
                if os.path.exists(f"{self.filename}{sub_directory}"):
                    for name in sorted(os.listdir(f"{self.filename}{sub_directory}")):
                        if os.path.isfile(f"{self.filename}{sub_directory}{name}") or self.file_format != "h5":
                            with Dataset.File(
                                f"{self.filename}{sub_directory}{name}",
                                True,
                                self.file_format,
                                self.level,
                            ) as file:
                                if file.is_exist(group):
                                    names.append(name.replace(".h5", "") if self.file_format == "h5" else name)
        else:
            with Dataset.File(self.filename, True, self.file_format, self.level) as file:
                names = file.get_names(groups)

        sorted_names = sorted(names)
        if index is None:
            self._names_cache[groups] = sorted_names
            return sorted_names
        return [name for i, name in enumerate(sorted_names) if i in index]

    def get_group(self) -> list[str]:
        if self.is_directory:
            if self.file_format in {"dicom", "omezarr"}:
                groups_set = set()
                root_path = Path(self.filename)
                for case_path in root_path.iterdir() if root_path.is_dir() else []:
                    if case_path.is_dir():
                        with Dataset.File(str(case_path), True, self.file_format, self.level) as dataset_file:
                            groups_set.update(dataset_file.get_group())
                return sorted(groups_set)
            groups_set = set()
            for root_dir, _, files in os.walk(self.filename):
                for file in files:
                    path = Path(root_dir, file.split(".")[0]).relative_to(self.filename).as_posix()
                    parts = path.split("/")
                    if len(parts) >= 2:
                        del parts[-2]
                    groups_set.add("/".join(parts))
            groups = list(groups_set)
        else:
            with Dataset.File(self.filename, True, self.file_format, self.level) as dataset_file:
                groups = dataset_file.get_group()
        return list(groups)

    def get_infos(self, groups: str, name: str) -> tuple[list[int], Attribute]:
        if self.is_directory:
            for sub_directory in self._get_sub_directories(groups):
                group = groups.split("/")[-1]
                if os.path.exists(f"{self.filename}{sub_directory}{name}{'.h5' if self.file_format == 'h5' else ''}"):
                    with Dataset.File(
                        f"{self.filename}{sub_directory}{name}",
                        True,
                        self.file_format,
                        self.level,
                    ) as file:
                        return file.get_infos("", group)
        else:
            with Dataset.File(self.filename, True, self.file_format, self.level) as file:
                return file.get_infos(groups, name)
        raise NameError(f"Dataset entry '{groups}/{name}' not found in {self.filename}.")

    def get_statistics(self, groups: str) -> dict[str, dict[str, dict[str, float | list[float]]]]:
        names = self.get_names(groups)
        stats = {}
        for name in names:
            data, attr = self.read_data(groups, name)

            min_, max_ = data.min(), data.max()
            mean_ = data.mean()
            std_ = data.std()

            # Percentiles in ONE call
            p25, p50, p75 = np.percentile(data, (25, 50, 75))

            stats[name] = {
                "min": float(min_),
                "max": float(max_),
                "mean": float(mean_),
                "std": float(std_),
                "25pc": float(p25),
                "50pc": float(p50),
                "75pc": float(p75),
                "shape": list(data.shape),
                "spacing": attr.get_np_array("Spacing").tolist(),
            }

        result: dict[str, dict[str, dict[str, Any]]] = {}
        result["case"] = {}
        for name, v in stats.items():
            for metric_name, value in v.items():
                if metric_name not in result["case"]:
                    result["case"][metric_name] = {}
                result["case"][metric_name][name] = value

        result["aggregates"] = {}
        tmp: dict[str, list[float]] = {}
        for _, v in stats.items():
            for metric_name, _ in v.items():
                if metric_name not in tmp:
                    tmp[metric_name] = []
                tmp[metric_name].append(v[metric_name])
        for metric_name, values in tmp.items():
            if isinstance(values[0], float):
                result["aggregates"][metric_name] = {
                    "max": float(np.nanmax(values)) if np.any(~np.isnan(values)) else np.nan,
                    "min": float(np.nanmin(values)) if np.any(~np.isnan(values)) else np.nan,
                    "std": float(np.nanstd(values)) if np.any(~np.isnan(values)) else np.nan,
                    "25pc": float(np.nanpercentile(values, 25)) if np.any(~np.isnan(values)) else np.nan,
                    "50pc": float(np.nanpercentile(values, 50)) if np.any(~np.isnan(values)) else np.nan,
                    "75pc": float(np.nanpercentile(values, 75)) if np.any(~np.isnan(values)) else np.nan,
                    "mean": float(np.nanmean(values)) if np.any(~np.isnan(values)) else np.nan,
                    "count": float(np.count_nonzero(~np.isnan(values))) if np.any(~np.isnan(values)) else np.nan,
                }
            else:
                p25, p50, p75 = np.nanpercentile(values, (25, 50, 75))

                result["aggregates"][metric_name] = {
                    "max": np.nanmax(values, axis=0).tolist(),
                    "min": np.nanmin(values, axis=0).tolist(),
                    "std": np.nanstd(values, axis=0).tolist(),
                    "mean": np.nanmean(values, axis=0).tolist(),
                }
        return result

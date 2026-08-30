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


"""The DICOM series backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]
from konfai.utils.dataset.abstract import AbstractFile
from konfai.utils.dataset.attribute import Attribute, image_to_data
from konfai.utils.errors import DatasetManagerError


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

    def bounded_region_reads(self, name: str) -> bool:
        del name
        return True  # one file per slice: a region decodes the slices it covers and nothing else

    def file_to_data_slice(self, group: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        from konfai.utils.dicom import get_dicom_info, read_dicom_series_slice

        path = self._path(name)
        info = dict(get_dicom_info(path))  # copy: get_dicom_info is memoised, and we update it below
        data, origin, spacing, direction = read_dicom_series_slice(
            path, slices, series_uid=info["series_uid"], info=info
        )
        info.update(origin=origin, spacing=spacing, direction=direction)
        return data, self._attributes(info)

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

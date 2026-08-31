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


"""The backend a path and a format token dispatch to."""

from __future__ import annotations

from typing import TYPE_CHECKING

from konfai.utils import uri
from konfai.utils.dataset.dicom_file import DicomFile
from konfai.utils.dataset.h5 import H5File
from konfai.utils.dataset.itk_transform_file import ItkTransformFile
from konfai.utils.dataset.ome_zarr_file import OmeZarrFile
from konfai.utils.dataset.sitk_file import SitkFile
from konfai.utils.errors import DatasetManagerError

if TYPE_CHECKING:
    from konfai.utils.dataset.abstract import AbstractFile


class File:
    def __init__(
        self,
        filename: str,
        read: bool,
        file_format: str,
        level: int = 0,
        scale_factors: list[int] | None = None,
        downsample_method: str | None = None,
    ) -> None:
        self.filename = filename
        self.read = read
        self.file: AbstractFile | None = None
        self.file_format = file_format
        self.level = level
        self.scale_factors = scale_factors
        self.downsample_method = downsample_method

    def __enter__(self) -> AbstractFile:
        if self.file_format == "omezarr":
            self.file = OmeZarrFile(self.filename, self.read, self.level, self.scale_factors, self.downsample_method)
        elif uri.is_uri(self.filename):
            # OME-Zarr addresses a store; every other backend opens a path.
            raise DatasetManagerError(
                f"'{self.filename}' is a remote root, which only ':omezarr' can read.",
                "Declare the root as ':omezarr', or copy the dataset locally first.",
            )
        elif self.file_format == "h5":
            self.file = H5File(self.filename, self.read)
        elif self.file_format == "dicom":
            self.file = DicomFile(self.filename, self.read)
        elif self.file_format == "itktransform":
            self.file = ItkTransformFile(self.filename + "/", self.read)
        else:
            self.file = SitkFile(self.filename + "/", self.read, self.file_format)
        self.file.__enter__()
        return self.file

    def __exit__(self, exc_type, value, traceback):
        if self.file is not None:
            self.file.__exit__(exc_type, value, traceback)

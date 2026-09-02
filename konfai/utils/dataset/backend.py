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

#: The backend each format token dispatches to; every token not named here is a plain-file
#: extension SitkFile serves. THE token-to-class table: a new backend registers here and declares
#: its facts on the class (see ``AbstractFile``), and nothing else needs a format-name branch.
BACKENDS: dict[str, type[AbstractFile]] = {
    "h5": H5File,
    "omezarr": OmeZarrFile,
    "dicom": DicomFile,
    "itktransform": ItkTransformFile,
}


def backend_for(file_format: str) -> type[AbstractFile]:
    """The backend class serving ``file_format``: where ``File.__enter__`` and ``Dataset`` read
    the per-backend facts from."""
    return BACKENDS.get(file_format, SitkFile)


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
        backend = backend_for(self.file_format)
        if uri.is_uri(self.filename) and not backend.reads_remote:
            # OME-Zarr addresses a store; every other backend opens a path.
            raise DatasetManagerError(
                f"'{self.filename}' is a remote root, which only ':omezarr' can read.",
                "Declare the root as ':omezarr', or copy the dataset locally first.",
            )
        self.file = backend.open(
            self.filename, self.read, self.file_format, self.level, self.scale_factors, self.downsample_method
        )
        self.file.__enter__()
        return self.file

    def __exit__(self, exc_type, value, traceback):
        if self.file is not None:
            self.file.__exit__(exc_type, value, traceback)

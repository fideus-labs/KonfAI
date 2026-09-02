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


"""Dataset file abstractions and image conversion utilities for KonfAI.

The storage backends live one per module and are addressed as ``Dataset.<Backend>``; every name of
the package is re-exported here; the backends are public as ``Dataset.<Backend>``."""

from konfai.utils.dataset.abstract import AbstractFile as AbstractFile
from konfai.utils.dataset.attribute import DISPLACEMENT_FIELD_ATTRIBUTE as DISPLACEMENT_FIELD_ATTRIBUTE
from konfai.utils.dataset.attribute import Attribute as Attribute
from konfai.utils.dataset.attribute import as_channel_first as as_channel_first
from konfai.utils.dataset.attribute import data_to_image as data_to_image
from konfai.utils.dataset.attribute import data_to_transform as data_to_transform
from konfai.utils.dataset.attribute import displacement_field_to_data as displacement_field_to_data
from konfai.utils.dataset.attribute import get_infos as get_infos
from konfai.utils.dataset.attribute import image_to_data as image_to_data
from konfai.utils.dataset.attribute import is_an_image as is_an_image
from konfai.utils.dataset.attribute import ome_zarr_attributes as ome_zarr_attributes
from konfai.utils.dataset.attribute import region_geometry as region_geometry
from konfai.utils.dataset.attribute import sitk as sitk
from konfai.utils.dataset.backend import BACKENDS as BACKENDS
from konfai.utils.dataset.backend import File as File
from konfai.utils.dataset.backend import backend_for as backend_for
from konfai.utils.dataset.core import Dataset as Dataset
from konfai.utils.dataset.dicom_file import DicomFile as DicomFile
from konfai.utils.dataset.h5 import H5File as H5File
from konfai.utils.dataset.h5 import h5py as h5py
from konfai.utils.dataset.h5 import release_read_handles as release_read_handles
from konfai.utils.dataset.itk_transform_file import ItkTransformFile as ItkTransformFile
from konfai.utils.dataset.landmarks import read_landmarks as read_landmarks
from konfai.utils.dataset.landmarks import write_landmarks as write_landmarks
from konfai.utils.dataset.ome_zarr_file import OmeZarrFile as OmeZarrFile
from konfai.utils.dataset.sitk_file import SitkFile as SitkFile
from konfai.utils.dataset.staging import is_staging_entry as is_staging_entry
from konfai.utils.dataset.statistics import chunk_hull_voxels as chunk_hull_voxels
from konfai.utils.dataset.stream import DataStream as DataStream

__all__ = [
    "BACKENDS",
    "DISPLACEMENT_FIELD_ATTRIBUTE",
    "Attribute",
    "DataStream",
    "Dataset",
    "as_channel_first",
    "backend_for",
    "chunk_hull_voxels",
    "data_to_image",
    "data_to_transform",
    "displacement_field_to_data",
    "get_infos",
    "image_to_data",
    "is_an_image",
    "is_staging_entry",
    "ome_zarr_attributes",
    "read_landmarks",
    "region_geometry",
    "release_read_handles",
    "write_landmarks",
]

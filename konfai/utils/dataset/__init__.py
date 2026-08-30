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
the package is re-exported here."""

from konfai.utils.dataset.abstract import AbstractFile as AbstractFile
from konfai.utils.dataset.attribute import DISPLACEMENT_FIELD_ATTRIBUTE as DISPLACEMENT_FIELD_ATTRIBUTE
from konfai.utils.dataset.attribute import Attribute as Attribute
from konfai.utils.dataset.attribute import _attribute_text as _attribute_text
from konfai.utils.dataset.attribute import _decode_transform as _decode_transform
from konfai.utils.dataset.attribute import _encode_transform_leaves as _encode_transform_leaves
from konfai.utils.dataset.attribute import _flatten_transforms as _flatten_transforms
from konfai.utils.dataset.attribute import _is_listed_name as _is_listed_name
from konfai.utils.dataset.attribute import _transform_codec as _transform_codec
from konfai.utils.dataset.attribute import as_channel_first as as_channel_first
from konfai.utils.dataset.attribute import data_to_image as data_to_image
from konfai.utils.dataset.attribute import data_to_transform as data_to_transform
from konfai.utils.dataset.attribute import displacement_field_to_data as displacement_field_to_data
from konfai.utils.dataset.attribute import get_infos as get_infos
from konfai.utils.dataset.attribute import image_to_data as image_to_data
from konfai.utils.dataset.attribute import is_an_image as is_an_image
from konfai.utils.dataset.attribute import ome_zarr_attributes as ome_zarr_attributes
from konfai.utils.dataset.attribute import read_landmarks as read_landmarks
from konfai.utils.dataset.attribute import sitk as sitk
from konfai.utils.dataset.attribute import write_landmarks as write_landmarks
from konfai.utils.dataset.backend import File as File
from konfai.utils.dataset.core import Dataset as Dataset
from konfai.utils.dataset.dicom_file import DicomFile as DicomFile
from konfai.utils.dataset.h5 import H5File as H5File
from konfai.utils.dataset.h5 import _get_h5_file_lock as _get_h5_file_lock
from konfai.utils.dataset.h5 import _h5_file_locks as _h5_file_locks
from konfai.utils.dataset.h5 import _h5_file_locks_guard as _h5_file_locks_guard
from konfai.utils.dataset.h5 import _h5_read_pool as _h5_read_pool
from konfai.utils.dataset.h5 import _H5DataStream as _H5DataStream
from konfai.utils.dataset.h5 import _H5ReadPool as _H5ReadPool
from konfai.utils.dataset.h5 import _open_h5 as _open_h5
from konfai.utils.dataset.h5 import _PooledRead as _PooledRead
from konfai.utils.dataset.h5 import h5py as h5py
from konfai.utils.dataset.h5 import release_read_handles as release_read_handles
from konfai.utils.dataset.itk_transform_file import ItkTransformFile as ItkTransformFile
from konfai.utils.dataset.itk_transform_file import _create_itk_transform_file as _create_itk_transform_file
from konfai.utils.dataset.itk_transform_file import _ItkTransformDataStream as _ItkTransformDataStream
from konfai.utils.dataset.ome_zarr_file import OmeZarrFile as OmeZarrFile
from konfai.utils.dataset.ome_zarr_file import _OmeZarrDataStream as _OmeZarrDataStream
from konfai.utils.dataset.raw_block import _MHA_DTYPES as _MHA_DTYPES
from konfai.utils.dataset.raw_block import _MHA_HEADER_PROBE_BYTES as _MHA_HEADER_PROBE_BYTES
from konfai.utils.dataset.raw_block import _NIFTI_DTYPES as _NIFTI_DTYPES
from konfai.utils.dataset.raw_block import _T as _T
from konfai.utils.dataset.raw_block import _divisor_tile as _divisor_tile
from konfai.utils.dataset.raw_block import _mapped_band as _mapped_band
from konfai.utils.dataset.raw_block import _mha_raw_block as _mha_raw_block
from konfai.utils.dataset.raw_block import _nifti_extract_aborts as _nifti_extract_aborts
from konfai.utils.dataset.raw_block import _nifti_raw_block as _nifti_raw_block
from konfai.utils.dataset.raw_block import _pixel_block as _pixel_block
from konfai.utils.dataset.raw_block import _pixel_block_at as _pixel_block_at
from konfai.utils.dataset.raw_block import _pixel_block_attributes as _pixel_block_attributes
from konfai.utils.dataset.raw_block import _pixel_block_region as _pixel_block_region
from konfai.utils.dataset.raw_block import _PixelBlock as _PixelBlock
from konfai.utils.dataset.raw_block import _sitk_component_dtypes as _sitk_component_dtypes
from konfai.utils.dataset.raw_block import _store_chunks as _store_chunks
from konfai.utils.dataset.raw_block import _unstreamed_formats_warned as _unstreamed_formats_warned
from konfai.utils.dataset.raw_block import _warn_unstreamed_region_read as _warn_unstreamed_region_read
from konfai.utils.dataset.sitk_file import SitkFile as SitkFile
from konfai.utils.dataset.staging import _REPLACED_MARKER as _REPLACED_MARKER
from konfai.utils.dataset.staging import _STAGING_PID as _STAGING_PID
from konfai.utils.dataset.staging import _orphaned_backup_names as _orphaned_backup_names
from konfai.utils.dataset.staging import _recover_orphaned_backup as _recover_orphaned_backup
from konfai.utils.dataset.staging import _replaced_name as _replaced_name
from konfai.utils.dataset.staging import _retire_dead_debris as _retire_dead_debris
from konfai.utils.dataset.staging import _writer_is_dead as _writer_is_dead
from konfai.utils.dataset.staging import is_staging_entry as is_staging_entry
from konfai.utils.dataset.statistics import _QUANTILE_BINS as _QUANTILE_BINS
from konfai.utils.dataset.statistics import _QUANTILE_COLLECT_CAP as _QUANTILE_COLLECT_CAP
from konfai.utils.dataset.statistics import _STATISTICS_BLOCKS_IN_FLIGHT as _STATISTICS_BLOCKS_IN_FLIGHT
from konfai.utils.dataset.statistics import _STATISTICS_CHUNK_ELEMENTS as _STATISTICS_CHUNK_ELEMENTS
from konfai.utils.dataset.statistics import _STATISTICS_ELEMENT_BYTES as _STATISTICS_ELEMENT_BYTES
from konfai.utils.dataset.statistics import _STATISTICS_UPDATE_ELEMENTS as _STATISTICS_UPDATE_ELEMENTS
from konfai.utils.dataset.statistics import _binned as _binned
from konfai.utils.dataset.statistics import _empty_statistics_state as _empty_statistics_state
from konfai.utils.dataset.statistics import _finalize_running_statistics as _finalize_running_statistics
from konfai.utils.dataset.statistics import _lerp_like_numpy as _lerp_like_numpy
from konfai.utils.dataset.statistics import _max_of as _max_of
from konfai.utils.dataset.statistics import _min_of as _min_of
from konfai.utils.dataset.statistics import _order_statistics as _order_statistics
from konfai.utils.dataset.statistics import _quantile_positions as _quantile_positions
from konfai.utils.dataset.statistics import _scan_block_on_the_store_grid as _scan_block_on_the_store_grid
from konfai.utils.dataset.statistics import _statistics_block_elements as _statistics_block_elements
from konfai.utils.dataset.statistics import _statistics_chunk_length as _statistics_chunk_length
from konfai.utils.dataset.statistics import _statistics_plane_elements as _statistics_plane_elements
from konfai.utils.dataset.statistics import _update_pieces as _update_pieces
from konfai.utils.dataset.statistics import _update_running_statistics as _update_running_statistics
from konfai.utils.dataset.statistics import chunk_hull_voxels as chunk_hull_voxels
from konfai.utils.dataset.stream import _MADV_DONTNEED as _MADV_DONTNEED
from konfai.utils.dataset.stream import _MHA_ELEMENT_TYPES as _MHA_ELEMENT_TYPES
from konfai.utils.dataset.stream import _NIFTI_DATATYPES as _NIFTI_DATATYPES
from konfai.utils.dataset.stream import DataStream as DataStream
from konfai.utils.dataset.stream import _MhaDataStream as _MhaDataStream
from konfai.utils.dataset.stream import _NiftiDataStream as _NiftiDataStream
from konfai.utils.dataset.stream import _RawBlockStream as _RawBlockStream

__all__ = [
    "DISPLACEMENT_FIELD_ATTRIBUTE",
    "AbstractFile",
    "Attribute",
    "DataStream",
    "Dataset",
    "DicomFile",
    "File",
    "H5File",
    "ItkTransformFile",
    "OmeZarrFile",
    "SitkFile",
    "as_channel_first",
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
    "release_read_handles",
    "write_landmarks",
]

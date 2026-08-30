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


"""Data augmentation: draws applied per case copy, each a chain stage past the Expand marker."""

from konfai.data.augmentation.base import DataAugmentation as DataAugmentation
from konfai.data.augmentation.base import DataAugmentationsList as DataAugmentationsList
from konfai.data.augmentation.base import Foreign as Foreign
from konfai.data.augmentation.base import Prob as Prob
from konfai.data.augmentation.base import _axis_rotation_matrix as _axis_rotation_matrix
from konfai.data.augmentation.base import _hashed_normal_field as _hashed_normal_field
from konfai.data.augmentation.base import _reflect_interval as _reflect_interval
from konfai.data.augmentation.base import _require_simpleitk as _require_simpleitk
from konfai.data.augmentation.base import _rotation_2d_matrix as _rotation_2d_matrix
from konfai.data.augmentation.base import _rotation_3d_matrix as _rotation_3d_matrix
from konfai.data.augmentation.base import _scale_matrix as _scale_matrix
from konfai.data.augmentation.base import _translate_matrix as _translate_matrix
from konfai.data.augmentation.base import sitk as sitk
from konfai.data.augmentation.color import HUE as HUE
from konfai.data.augmentation.color import Brightness as Brightness
from konfai.data.augmentation.color import ColorTransform as ColorTransform
from konfai.data.augmentation.color import Contrast as Contrast
from konfai.data.augmentation.color import LumaFlip as LumaFlip
from konfai.data.augmentation.color import Saturation as Saturation
from konfai.data.augmentation.placed import CutOUT as CutOUT
from konfai.data.augmentation.placed import Mask as Mask
from konfai.data.augmentation.placed import Noise as Noise
from konfai.data.augmentation.placed import PlacedDraw as PlacedDraw
from konfai.data.augmentation.spatial import Elastix as Elastix
from konfai.data.augmentation.spatial import EulerTransform as EulerTransform
from konfai.data.augmentation.spatial import Flip as Flip
from konfai.data.augmentation.spatial import Permute as Permute
from konfai.data.augmentation.spatial import Rotate as Rotate
from konfai.data.augmentation.spatial import Scale as Scale
from konfai.data.augmentation.spatial import Translate as Translate

__all__ = [
    "HUE",
    "Brightness",
    "ColorTransform",
    "Contrast",
    "CutOUT",
    "DataAugmentation",
    "DataAugmentationsList",
    "Elastix",
    "EulerTransform",
    "Flip",
    "Foreign",
    "LumaFlip",
    "Mask",
    "Noise",
    "Permute",
    "PlacedDraw",
    "Prob",
    "Rotate",
    "Saturation",
    "Scale",
    "Translate",
]

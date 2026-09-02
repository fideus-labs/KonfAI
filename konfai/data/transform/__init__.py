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


"""Tensor and image transforms used in KonfAI preprocessing and postprocessing.

A chain stage is referenced by its bare name (``Standardize``), which resolves here: every stage of
the package is re-exported, whichever module holds it."""

from konfai.data.transform.base import Foreign as Foreign
from konfai.data.transform.base import LocalityKind as LocalityKind
from konfai.data.transform.base import PatchLocality as PatchLocality
from konfai.data.transform.base import RegionContext as RegionContext
from konfai.data.transform.base import Transform as Transform
from konfai.data.transform.base import TransformInverse as TransformInverse
from konfai.data.transform.base import TransformLoader as TransformLoader
from konfai.data.transform.base import sitk as sitk
from konfai.data.transform.base import stat_seed_valid as stat_seed_valid
from konfai.data.transform.chain import Expand as Expand
from konfai.data.transform.chain import Reduce as Reduce
from konfai.data.transform.chain import resolve_operator as resolve_operator
from konfai.data.transform.chain import split_expand as split_expand
from konfai.data.transform.ensemble import InferenceStack as InferenceStack
from konfai.data.transform.ensemble import Magnitude as Magnitude
from konfai.data.transform.ensemble import Norm as Norm
from konfai.data.transform.ensemble import Percentage as Percentage
from konfai.data.transform.ensemble import SegmentationDisagreement as SegmentationDisagreement
from konfai.data.transform.ensemble import StandardDeviation as StandardDeviation
from konfai.data.transform.ensemble import Variance as Variance
from konfai.data.transform.intensity import Clip as Clip
from konfai.data.transform.intensity import HistogramMatching as HistogramMatching
from konfai.data.transform.intensity import Normalize as Normalize
from konfai.data.transform.intensity import Standardize as Standardize
from konfai.data.transform.intensity import Statistics as Statistics
from konfai.data.transform.intensity import TensorCast as TensorCast
from konfai.data.transform.intensity import UnNormalize as UnNormalize
from konfai.data.transform.io import Save as Save
from konfai.data.transform.io import Write as Write
from konfai.data.transform.labels import Argmax as Argmax
from konfai.data.transform.labels import Dilate as Dilate
from konfai.data.transform.labels import FlatLabel as FlatLabel
from konfai.data.transform.labels import Mask as Mask
from konfai.data.transform.labels import MergeLabels as MergeLabels
from konfai.data.transform.labels import OneHot as OneHot
from konfai.data.transform.labels import SelectLabel as SelectLabel
from konfai.data.transform.labels import Softmax as Softmax
from konfai.data.transform.labels import Sum as Sum
from konfai.data.transform.resample import Resample as Resample
from konfai.data.transform.shape import Canonical as Canonical
from konfai.data.transform.shape import Crop as Crop
from konfai.data.transform.shape import Flatten as Flatten
from konfai.data.transform.shape import Flip as Flip
from konfai.data.transform.shape import Gradient as Gradient
from konfai.data.transform.shape import Padding as Padding
from konfai.data.transform.shape import Permute as Permute
from konfai.data.transform.shape import Squeeze as Squeeze

__all__ = [
    "Argmax",
    "Canonical",
    "Clip",
    "Crop",
    "Dilate",
    "Expand",
    "FlatLabel",
    "Flatten",
    "Flip",
    "Foreign",
    "Gradient",
    "HistogramMatching",
    "InferenceStack",
    "LocalityKind",
    "Magnitude",
    "Mask",
    "MergeLabels",
    "Norm",
    "Normalize",
    "OneHot",
    "Padding",
    "PatchLocality",
    "Percentage",
    "Permute",
    "Reduce",
    "RegionContext",
    "Resample",
    "Save",
    "SegmentationDisagreement",
    "SelectLabel",
    "Softmax",
    "Squeeze",
    "StandardDeviation",
    "Standardize",
    "Statistics",
    "Sum",
    "TensorCast",
    "Transform",
    "TransformInverse",
    "TransformLoader",
    "UnNormalize",
    "Variance",
    "Write",
    "resolve_operator",
    "split_expand",
    "stat_seed_valid",
]


def __getattr__(name: str):
    # ``KonfAIInference`` lives in konfai-apps (it drives a nested app run), but published configs
    # spell the bare name, which the TransformLoader resolves against this package: hand the class
    # over when konfai-apps is installed, refuse with the install otherwise.
    if name == "KonfAIInference":
        try:
            from konfai_apps.transforms import KonfAIInference
        except ImportError as exc:
            from konfai.utils.errors import TransformError

            raise TransformError(
                "KonfAIInference requires the standalone 'konfai-apps' package.",
                "Install it with 'pip install konfai-apps'.",
            ) from exc
        return KonfAIInference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

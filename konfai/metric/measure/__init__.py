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


"""Criterion and metric implementations used by KonfAI workflows: a bare name in a config resolves here."""

from konfai.metric.measure.adversarial import FID as FID
from konfai.metric.measure.adversarial import WGP as WGP
from konfai.metric.measure.adversarial import Gram as Gram
from konfai.metric.measure.adversarial import PatchGanLoss as PatchGanLoss
from konfai.metric.measure.adversarial import PerceptualLoss as PerceptualLoss
from konfai.metric.measure.base import Criterion as Criterion
from konfai.metric.measure.base import CriterionWithAttribute as CriterionWithAttribute
from konfai.metric.measure.base import CriterionWithInit as CriterionWithInit
from konfai.metric.measure.base import MaskedLoss as MaskedLoss
from konfai.metric.measure.base import _require_optional as _require_optional
from konfai.metric.measure.base import models_register as models_register
from konfai.metric.measure.impact import ImpactFeatureModel as ImpactFeatureModel
from konfai.metric.measure.impact import IMPACTReg as IMPACTReg
from konfai.metric.measure.impact import IMPACTSynth as IMPACTSynth
from konfai.metric.measure.impact import SAM_Perceptual as SAM_Perceptual
from konfai.metric.measure.impact import _check_feature_model as _check_feature_model
from konfai.metric.measure.impact import _denormalized as _denormalized
from konfai.metric.measure.impact import _feature_loss_mean as _feature_loss_mean
from konfai.metric.measure.impact import _feature_mask as _feature_mask
from konfai.metric.measure.impact import _masked_feature_loss as _masked_feature_loss
from konfai.metric.measure.impact import _patch_views as _patch_views
from konfai.metric.measure.regression import BCE as BCE
from konfai.metric.measure.regression import LPIPS as LPIPS
from konfai.metric.measure.regression import MAE as MAE
from konfai.metric.measure.regression import ME as ME
from konfai.metric.measure.regression import MSE as MSE
from konfai.metric.measure.regression import PSNR as PSNR
from konfai.metric.measure.regression import SSIM as SSIM
from konfai.metric.measure.regression import TRE as TRE
from konfai.metric.measure.regression import Accuracy as Accuracy
from konfai.metric.measure.regression import CrossEntropyLoss as CrossEntropyLoss
from konfai.metric.measure.regression import FocalLoss as FocalLoss
from konfai.metric.measure.regression import GradientImages as GradientImages
from konfai.metric.measure.regression import KLDivergence as KLDivergence
from konfai.metric.measure.regression import L1LossRepresentation as L1LossRepresentation
from konfai.metric.measure.regression import MAESaveMap as MAESaveMap
from konfai.metric.measure.regression import Mean as Mean
from konfai.metric.measure.regression import MutualInformationLoss as MutualInformationLoss
from konfai.metric.measure.regression import TripletLoss as TripletLoss
from konfai.metric.measure.regression import Variance as Variance
from konfai.metric.measure.segmentation import Dice as Dice
from konfai.metric.measure.segmentation import DiceSaveMap as DiceSaveMap
from konfai.metric.measure.segmentation import LabelSums as LabelSums

__all__ = [
    "BCE",
    "FID",
    "LPIPS",
    "MAE",
    "ME",
    "MSE",
    "PSNR",
    "SSIM",
    "TRE",
    "WGP",
    "Accuracy",
    "Criterion",
    "CriterionWithAttribute",
    "CriterionWithInit",
    "CrossEntropyLoss",
    "Dice",
    "DiceSaveMap",
    "FocalLoss",
    "GradientImages",
    "Gram",
    "IMPACTReg",
    "IMPACTSynth",
    "ImpactFeatureModel",
    "KLDivergence",
    "L1LossRepresentation",
    "LabelSums",
    "MAESaveMap",
    "MaskedLoss",
    "Mean",
    "MutualInformationLoss",
    "PatchGanLoss",
    "PerceptualLoss",
    "SAM_Perceptual",
    "TripletLoss",
    "Variance",
    "models_register",
]

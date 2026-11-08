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

from konfai.metric.measure.adversarial import Gram as Gram
from konfai.metric.measure.adversarial import PatchGanLoss as PatchGanLoss
from konfai.metric.measure.adversarial import PerceptualLoss as PerceptualLoss
from konfai.metric.measure.base import Criterion as Criterion
from konfai.metric.measure.base import CriterionOutput as CriterionOutput
from konfai.metric.measure.base import CriterionResult as CriterionResult
from konfai.metric.measure.base import CriterionValue as CriterionValue
from konfai.metric.measure.base import CriterionWithAttribute as CriterionWithAttribute
from konfai.metric.measure.base import CriterionWithInit as CriterionWithInit
from konfai.metric.measure.base import LabelledValues as LabelledValues
from konfai.metric.measure.base import MaskedLoss as MaskedLoss
from konfai.metric.measure.impact import ImpactFeatureModel as ImpactFeatureModel
from konfai.metric.measure.impact import IMPACTReg as IMPACTReg
from konfai.metric.measure.impact import IMPACTSynth as IMPACTSynth
from konfai.metric.measure.impact import SAM_Perceptual as SAM_Perceptual
from konfai.metric.measure.regression import BCE as BCE
from konfai.metric.measure.regression import LNCC as LNCC
from konfai.metric.measure.regression import LPIPS as LPIPS
from konfai.metric.measure.regression import MAE as MAE
from konfai.metric.measure.regression import ME as ME
from konfai.metric.measure.regression import MSE as MSE
from konfai.metric.measure.regression import NCC as NCC
from konfai.metric.measure.regression import NMI as NMI
from konfai.metric.measure.regression import PSNR as PSNR
from konfai.metric.measure.regression import SSIM as SSIM
from konfai.metric.measure.regression import TRE as TRE
from konfai.metric.measure.regression import Accuracy as Accuracy
from konfai.metric.measure.regression import BendingEnergy as BendingEnergy
from konfai.metric.measure.regression import CrossEntropyLoss as CrossEntropyLoss
from konfai.metric.measure.regression import Displacement as Displacement
from konfai.metric.measure.regression import FocalLoss as FocalLoss
from konfai.metric.measure.regression import GradientImages as GradientImages
from konfai.metric.measure.regression import JacobianFolding as JacobianFolding
from konfai.metric.measure.regression import JacobianLogStd as JacobianLogStd
from konfai.metric.measure.regression import JacobianMin as JacobianMin
from konfai.metric.measure.regression import KLDivergence as KLDivergence
from konfai.metric.measure.regression import MAESaveMap as MAESaveMap
from konfai.metric.measure.regression import Mean as Mean
from konfai.metric.measure.regression import PointDistance as PointDistance
from konfai.metric.measure.regression import PointSpread as PointSpread
from konfai.metric.measure.regression import Quantile as Quantile
from konfai.metric.measure.regression import SpreadCalibration as SpreadCalibration
from konfai.metric.measure.regression import Variance as Variance
from konfai.metric.measure.segmentation import ASSD as ASSD
from konfai.metric.measure.segmentation import HD95 as HD95
from konfai.metric.measure.segmentation import Dice as Dice
from konfai.metric.measure.segmentation import DiceSaveMap as DiceSaveMap
from konfai.metric.measure.segmentation import Jaccard as Jaccard
from konfai.metric.measure.segmentation import LabelSums as LabelSums
from konfai.metric.measure.segmentation import SurfaceDice as SurfaceDice
from konfai.metric.measure.segmentation import VolumeDifference as VolumeDifference

__all__ = [
    "ASSD",
    "BCE",
    "HD95",
    "LNCC",
    "LPIPS",
    "MAE",
    "ME",
    "MSE",
    "NCC",
    "NMI",
    "PSNR",
    "SSIM",
    "TRE",
    "Accuracy",
    "BendingEnergy",
    "Criterion",
    "CriterionOutput",
    "CriterionResult",
    "CriterionValue",
    "CriterionWithAttribute",
    "CriterionWithInit",
    "CrossEntropyLoss",
    "Dice",
    "DiceSaveMap",
    "Displacement",
    "FocalLoss",
    "GradientImages",
    "Gram",
    "IMPACTReg",
    "IMPACTSynth",
    "ImpactFeatureModel",
    "Jaccard",
    "JacobianFolding",
    "JacobianLogStd",
    "JacobianMin",
    "KLDivergence",
    "LabelSums",
    "LabelledValues",
    "MAESaveMap",
    "MaskedLoss",
    "Mean",
    "PatchGanLoss",
    "PerceptualLoss",
    "PointDistance",
    "PointSpread",
    "Quantile",
    "SAM_Perceptual",
    "SpreadCalibration",
    "SurfaceDice",
    "Variance",
    "VolumeDifference",
]

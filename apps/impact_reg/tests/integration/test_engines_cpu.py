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

"""One small real registration per engine, on the CPU, and one end-to-end ``register``.

Each engine registers a blob translated by a known vector: its field must be that vector inside the object, on
the fixed grid, channel-first. Built through the preset's own ``RegistrationNet``, so the build path runs as well
as the engine: every test elsewhere stubs the engines, which is how an itk-impact API rename, a missing scipy and
an elastix binary that no longer loads next to the installed torch all shipped.

Gated by ``IMPACT_REG_ENGINE_TESTS=1``, which the CI lane that installs the engines sets (itk-impact, fireants,
the elastix-IMPACT binary, network access for the feature models and the presets). There nothing skips: a missing
or broken engine fails.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("IMPACT_REG_ENGINE_TESTS"),
        reason="set IMPACT_REG_ENGINE_TESTS=1 with itk-impact, fireants and the elastix-IMPACT binary installed",
    ),
]

#: The moving blob sits here relative to the fixed one, in mm (x, y, z): the field must be this inside it.
_SHIFT = np.array([3.0, -2.0, 1.5])

_TRANSLATION_MAP = """(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(Registration "MultiResolutionRegistration")
(Interpolator "LinearInterpolator")
(ResampleInterpolator "FinalLinearInterpolator")
(Resampler "DefaultResampler")
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")
(Optimizer "AdaptiveStochasticGradientDescent")
(Transform "TranslationTransform")
(Metric "AdvancedMeanSquares")
(NumberOfResolutions 2)
(MaximumNumberOfIterations 200)
(NumberOfSpatialSamples 2000)
(ImageSampler "RandomCoordinate")
(NewSamplesEveryIteration "true")
(AutomaticTransformInitialization "true")
(WriteResultImage "false")
(ITKTransformOutputFileNameExtension "itk.txt")
(WriteITKCompositeTransform "true")
"""


def _blob(center: np.ndarray) -> sitk.Image:
    """An anisotropic blob on a 40^3 grid of 1.5 mm, centred at ``center`` (mm)."""
    z, y, x = np.mgrid[:40, :40, :40].astype(np.float32) * 1.5
    r2 = ((x - center[0]) / 12) ** 2 + ((y - center[1]) / 9) ** 2 + ((z - center[2]) / 7) ** 2
    image = sitk.GetImageFromArray((100 * np.exp(-2 * r2) + 100 * (r2 < 1)).astype(np.float32))
    image.SetSpacing((1.5, 1.5, 1.5))
    return image


@pytest.fixture(scope="module")
def pair() -> tuple[sitk.Image, sitk.Image, np.ndarray]:
    center = np.array([30.0, 30.0, 30.0])
    fixed = _blob(center)
    return fixed, _blob(center + _SHIFT), sitk.GetArrayFromImage(fixed) > 50


def _assert_shift(field: np.ndarray, inside: np.ndarray) -> None:
    assert field.shape == (3, *inside.shape)
    np.testing.assert_allclose([field[axis][inside].mean() for axis in range(3)], _SHIFT, atol=0.75)


def test_fireants_recovers_a_translation(pair: tuple) -> None:
    from impact_reg_konfai.models.fireants import RegistrationNet

    fixed, moving, inside = pair
    net = RegistrationNet(scales=[2, 1], affine_iterations=[200, 100], deformable_iterations=[30, 15])
    _assert_shift(net.Registration._engine.register(fixed, moving, -1), inside)


def test_convexadam_recovers_a_translation(pair: tuple) -> None:
    from impact_reg_konfai.models.convexadam import ModelSpec, RegistrationNet

    fixed, moving, inside = pair
    mind = ModelSpec(ref="VBoussot/impact-torchscript-models:MIND/R1D2_3D.pt", voxel_size=[1.5, 1.5, 1.5])
    # The presets' 80 fine steps: the 2-voxel step of the reference's refinement settles over them, not over 20.
    net = RegistrationNet(models={"0": mind}, iterations=80, linear_iterations=50, displacement_half_width=3)
    _assert_shift(net.Registration._engine.register(fixed, moving, -1), inside)


def test_elastix_recovers_a_translation(pair: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from impact_reg_konfai.models.elastix import RegistrationNet

    fixed, moving, inside = pair
    monkeypatch.chdir(tmp_path)  # a preset's parameter maps are staged in the working directory
    (tmp_path / "Translation.txt").write_text(_TRANSLATION_MAP)
    net = RegistrationNet(engine="elastix", parameter_maps=["Translation.txt"])
    _assert_shift(net.Registration._engine.register(fixed, moving, -1), inside)


def test_register_end_to_end_on_the_cpu(pair: tuple, tmp_path: Path) -> None:
    # The CLI as SlicerImpactReg drives it, on a published preset: the transform and the moved image come back.
    fixed, moving, inside = pair
    sitk.WriteImage(fixed, str(tmp_path / "fixed.mha"))
    sitk.WriteImage(moving, str(tmp_path / "moving.mha"))
    command = [sys.executable, "-m", "impact_reg_konfai.cli", "register", "Generic_Rigid_BSpline"]
    command += ["-f", str(tmp_path / "fixed.mha"), "-m", str(tmp_path / "moving.mha"), "-o", str(tmp_path / "out")]
    subprocess.run([*command, "--cpu", "1", "--tmp-dir", str(tmp_path / "tmp")], check=True)

    case = tmp_path / "out" / "P000"
    assert (case / "Moved.mha").is_file()
    transform = sitk.ReadTransform(str(case / "Transform.h5"))
    field = sitk.TransformToDisplacementField(
        transform, sitk.sitkVectorFloat64, fixed.GetSize(), fixed.GetOrigin(), fixed.GetSpacing()
    )
    _assert_shift(np.moveaxis(sitk.GetArrayFromImage(field), -1, 0), inside)

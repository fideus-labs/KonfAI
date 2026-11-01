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

"""Where a linear stage starts, when a preset declares ``moments_init: com``."""

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from impact_reg_konfai.models import elastix_engine
from impact_reg_konfai.models.elastix_engine import ElastixEngine
from impact_reg_konfai.models.intensity import center_of_mass_translation


def _two_sparse_subjects() -> tuple[sitk.Image, sitk.Image]:
    """Two blobs a field of view apart, on the background a microscope records."""
    fixed = np.full((20, 40, 40), 100.0, np.float32)
    moving = fixed.copy()
    fixed[8:12, 5:12, 5:12] = 1000.0
    moving[8:12, 25:32, 25:32] = 1000.0
    return sitk.GetImageFromArray(fixed), sitk.GetImageFromArray(moving)


def test_the_centres_of_mass_overlap_two_sparse_subjects_over_a_background() -> None:
    fixed, moving = _two_sparse_subjects()
    np.testing.assert_allclose(center_of_mass_translation(fixed, moving), [20.0, 20.0, 0.0], atol=1e-9)


def test_a_mask_on_another_grid_restricts_the_centre() -> None:
    fixed, moving = _two_sparse_subjects()
    both = sitk.GetArrayFromImage(moving)
    both[8:12, 5:12, 5:12] = 1000.0  # the moving image holds the fixed blob too
    region = np.zeros((10, 20, 20), np.uint8)
    region[:, 10:, 10:] = 1  # the half of the image, on a grid twice as coarse, around the moving blob
    mask = sitk.GetImageFromArray(region)
    mask.SetSpacing((2.0, 2.0, 2.0))
    translation = center_of_mass_translation(fixed, sitk.GetImageFromArray(both), None, mask)
    np.testing.assert_allclose(translation, [20.0, 20.0, 0.0], atol=1e-9)


def test_a_constant_image_starts_from_its_grid_centre() -> None:
    constant = sitk.Image([8, 8, 8], sitk.sitkFloat32) + 5.0
    assert center_of_mass_translation(constant, constant) == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(("moments_init", "shifted"), [("cof", False), ("com", True)])
def test_elastix_starts_where_the_preset_declares(
    monkeypatch, tmp_path: Path, moments_init: str, shifted: bool
) -> None:
    # 'com' goes to the moving image's origin and comes back in the field, with elastix's own start off; 'cof' leaves
    # the images and the map as they are. elastix answers with an identity, so the field is the start.
    (tmp_path / "Rigid.txt").write_text('(Transform "EulerTransform")\n(AutomaticTransformInitialization "true")\n')
    monkeypatch.chdir(tmp_path)
    fixed, moving = _two_sparse_subjects()
    seen = {}

    class Process:
        stdout = iter([])

        def __init__(self, args, **kwargs) -> None:
            work = Path(args[args.index("-out") + 1])
            seen["origin"] = sitk.ReadImage(args[args.index("-m") + 1]).GetOrigin()
            seen["map"] = Path(args[args.index("-p") + 1]).read_text()
            sitk.WriteTransform(
                sitk.Transform(3, sitk.sitkIdentity), str(work / "TransformParameters.0-Composite.itk.txt")
            )

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(elastix_engine.subprocess, "Popen", Process)
    monkeypatch.setattr(elastix_engine, "loader_env", lambda root: {})
    monkeypatch.setattr(ElastixEngine, "_ensure_binary", lambda self: Path("elastix"))
    engine = ElastixEngine(["Rigid.txt"], moments_init=moments_init)
    engine._elastix_root = Path("/opt/elastix-impact")
    field = engine.register(fixed, moving, -1)

    expected = [20.0, 20.0, 0.0] if shifted else [0.0, 0.0, 0.0]
    np.testing.assert_allclose(seen["origin"], [-v for v in expected], atol=1e-6)
    np.testing.assert_allclose([component.mean() for component in field], expected, atol=1e-5)
    assert (f'(AutomaticTransformInitialization "{"false" if shifted else "true"}")') in seen["map"]


def test_elastix_refuses_an_unknown_start() -> None:
    with pytest.raises(ValueError, match="Unknown moments_init 'auto'"):
        ElastixEngine(["Rigid.txt"], moments_init="auto")

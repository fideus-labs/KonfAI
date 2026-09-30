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

"""feature_map_update_interval in FireANTs' Static mode: the deformable stage in runs of N iterations, the moving
features extracted again between runs from the image warped so far, the fields composed."""

from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import SimpleITK as sitk
import torch
from impact_reg_konfai.models import fireants


class _Echo(torch.nn.Module):
    """A 3D feature model whose only layer is its input."""

    def forward(
        self,
        x: torch.Tensor,
        nb_layers: torch.Tensor,
        stats: Optional[torch.Tensor] = None,  # noqa: UP045 (TorchScript)
        direction: Optional[torch.Tensor] = None,  # noqa: UP045 (TorchScript)
    ) -> list[torch.Tensor]:
        return [x[:, :1] * 1.0]


def _engine(tmp_path: Path, **overrides) -> fireants.FireANTsEngine:
    path = tmp_path / "echo.pt"
    torch.jit.script(_Echo()).save(str(path))
    settings = {"mode": "Static", "feature_map_update_interval": 2, **overrides}
    return fireants.FireANTsEngine(
        [2, 1], [1, 1], [3, 2], 3, "mse", 0.01, "none", "none", "syn", "impact", 0.1, 0.5, 1.0, 0,
        [[fireants.ModelSpec(ref=str(path))], [fireants.ModelSpec(ref=str(path))]], **settings,
    )  # fmt: skip


def _blob(shift: float = 0.0) -> sitk.Image:
    z, y, x = np.mgrid[:32, :32, :32].astype(np.float32)
    return sitk.GetImageFromArray(np.exp(-((x - 15 - shift) ** 2 + (y - 16) ** 2 + (z - 16) ** 2) / 40.0))


def test_the_stage_runs_in_pieces_each_at_its_scale_and_level(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fireants")
    runs: list[tuple] = []
    total_field = fireants.FireANTsEngine._total_field

    def recorded(engine, reg):
        field = total_field(engine, reg)
        runs.append((list(reg.scales), list(reg.iterations), engine._feature_loss._level))
        return field

    monkeypatch.setattr(fireants.FireANTsEngine, "_total_field", recorded)
    field = _engine(tmp_path).register(_blob(), _blob(2.0), -1)
    # scales [2, 1] with [3, 2] iterations in runs of 2: 2 + 1 at scale 2 (level 0), then 2 at scale 1 (level 1)
    assert runs == [([2], [2], 0), ([2], [1], 0), ([1], [2], 1)]
    assert field.shape == (3, 32, 32, 32) and np.isfinite(field).all()


def test_the_runs_fields_are_composed_each_after_the_ones_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fireants")
    seen: list[float] = []

    def translation(engine, reg):
        # Each run returns a 1 mm shift along x; the moving image it registered was already shifted by the runs before.
        seen.append(float(reg.moving_images().detach().cpu().numpy().sum()))
        shift = np.zeros((32, 32, 32, 3))
        shift[..., 0] = 1.0
        return sitk.GetImageFromArray(shift, isVector=True)

    monkeypatch.setattr(fireants.FireANTsEngine, "_total_field", translation)
    field = _engine(tmp_path).register(_blob(), _blob(2.0), -1)
    assert np.allclose(field[:, 16, 16, 16], [3.0, 0.0, 0.0])  # three runs of 1 mm, composed
    assert len(seen) == 3 and not seen[0] == seen[1]  # the moving features were extracted again after a run


def test_jacobian_mode_refuses_an_update_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Jacobian mode extracts them at every step"):
        _engine(tmp_path, mode="Jacobian")

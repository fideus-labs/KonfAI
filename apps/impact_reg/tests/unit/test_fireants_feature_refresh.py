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


def test_the_runs_by_boxes_are_the_whole_run_s(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # By boxes, the runs go as the whole run's: each at its scale and level, on the moving image read through the
    # field so far, its features extracted again, the fields composed (three runs of 1 mm along x: 3 mm).
    pytest.importorskip("fireants")
    from impact_reg_konfai.models import fireants_sampled

    paths = []
    for name, image in (("fixed", _blob()), ("moving", _blob(2.0))):
        paths.append(tmp_path / f"{name}.mha")
        sitk.WriteImage(image, str(paths[-1]))
    runs: list[tuple] = []

    class Translation:
        def __init__(self, scales, iterations, fixed_images, moving_images, fixed, moving, **kwargs) -> None:
            self.scales, self.iterations, self.loss = scales, iterations, kwargs["custom_loss"]
            self.moving, self.losses, self.resident, self.boxes_per_scale = moving, [], True, [1]

        def optimize(self) -> None:
            self.loss.set_current_scale_and_iterations(self.scales[0], self.iterations[0])
            seen = float(self.moving.read([0, 0, 0], list(self.moving.shape)).sum())
            runs.append((list(self.scales), list(self.iterations), self.loss._level, seen))

        def write_field(self, field) -> None:
            field[:] = 0.0
            field[..., 0] = 1.0

        def close(self) -> None:
            pass

    monkeypatch.setattr(fireants_sampled, "sampled_deformable_class", lambda: Translation)
    (tmp_path / "scratch").mkdir()
    _engine(tmp_path).register_sampled(*paths, None, None, -1, tmp_path / "scratch", tmp_path / "field.mha", box=16)
    field = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "field.mha")))
    # scales [2, 1] with [3, 2] iterations in runs of 2: 2 + 1 at scale 2 (level 0), then 2 at scale 1 (level 1)
    assert [run[:3] for run in runs] == [([2], [2], 0), ([2], [1], 0), ([1], [2], 1)]
    assert np.allclose(field[16, 16, 16], [3.0, 0.0, 0.0])  # three runs of 1 mm, composed
    assert len({run[3] for run in runs}) == 3  # the moving features were extracted again after each run
    leftovers = [
        path.name for pattern in ("run_*", "total_*", "moved*") for path in (tmp_path / "scratch").glob(pattern)
    ]
    assert not leftovers  # each run's files go once the field after it stands


def test_jacobian_mode_refuses_an_update_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Jacobian mode extracts them at every step"):
        _engine(tmp_path, mode="Jacobian")


def test_without_samples_the_linear_stages_take_every_box(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Not told how many voxels to draw, the rigid and affine stages take every box of a level at every iteration, as
    # the deformable stage does: FireANTs' own iteration, not a draw.
    pytest.importorskip("fireants")
    from impact_reg_konfai.models import fireants_sampled
    from impact_reg_konfai.ooc import tiles

    paths = []
    for name, image in (("fixed", _blob()), ("moving", _blob(2.0))):
        paths.append(tmp_path / f"{name}.mha")
        sitk.WriteImage(image, str(paths[-1]))
    asked: dict = {}

    def alignment(*args, box, boxes, **kwargs):
        asked.update(box=box, boxes=boxes)
        return torch.eye(4)[None]

    monkeypatch.setattr(fireants_sampled, "sampled_linear_alignment", alignment)
    monkeypatch.setattr(fireants_sampled, "_TOGETHER", 8**3)  # a draw far smaller than the image
    (tmp_path / "scratch").mkdir()
    engine = fireants.FireANTsEngine(
        [2, 1], [1, 1], [0, 0], 3, "mse", 0.01, "none", "rigid_affine", "none", "cc", 0.1, 0.5, 1.0, 0, []
    )
    engine.register_sampled(*paths, None, None, -1, tmp_path / "scratch", tmp_path / "field.mha", box=16)
    assert asked["boxes"] >= len(tiles((32, 32, 32), asked["box"]))

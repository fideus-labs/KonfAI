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

"""The per-component statistic, exercised through a transform defined HERE rather than shipped.

The stage under test is deliberately a user's: KonfAI ships generic transforms, and centring a
displacement field per component then scaling it is one step of an atlas build, which belongs to the
pipeline that needs it. What the framework owes such a stage is that declaring a locality contract is
enough to stream it, and that is exactly what these tests pin.

So this doubles as the worked example: forty lines below, written against the public surface only,
is a user transform that streams a volume it never assembles.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from konfai.data import Attribute, LocalityKind, PatchLocality, Transform, Write
from konfai.data.patching import DatasetManager
from konfai.utils.dataset import Dataset

pytest.importorskip("SimpleITK")


class _CentreAndScale(Transform):
    """``-step * (field - t)``, t being the field's per-component spatial mean.

    Per-component on purpose: a translation has as many parts as the field has components, and the
    pooled mean of all of them describes nothing. Declaring ``MeanPerChannel`` is what lets the
    statistic be read once from the stored volume and the stage then be a value map, so a field of
    any size runs region by region.
    """

    def __init__(self, step: float = 0.25) -> None:
        super().__init__()
        self.step = float(step)

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.GLOBAL_STAT, stat_keys=frozenset({"MeanPerChannel"}))

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "MeanPerChannel" not in cache_attribute:
            cache_attribute["MeanPerChannel"] = tensor.reshape(int(tensor.shape[0]), -1).to(torch.float32).mean(dim=1)
        mean = torch.as_tensor(cache_attribute.get_np_array("MeanPerChannel"), dtype=torch.float32)
        centred = tensor.to(torch.float32) - mean.reshape(-1, *([1] * (tensor.dim() - 1)))
        return -self.step * centred


def _field(tmp_path: Path) -> Dataset:
    rng = np.random.default_rng(0)
    attributes = Attribute()
    attributes["Origin"] = np.asarray([0.0, 0.0, 0.0])
    attributes["Spacing"] = np.asarray([1.0, 1.0, 1.0])
    attributes["Direction"] = np.eye(3, dtype=np.float64).reshape(-1)
    dataset = Dataset(tmp_path / "src", "h5")
    # Three components with DIFFERENT means: a pooled mean would centre none of them.
    volume = rng.normal(size=(3, 10, 8, 6)).astype(np.float32)
    volume += np.asarray([5.0, -2.0, 11.0], dtype=np.float32).reshape(3, 1, 1, 1)
    dataset.write("DVF", "CASE_000", volume, attributes)
    return dataset


def _manager(source: Dataset, transforms) -> DatasetManager:
    return DatasetManager(
        index=0,
        group_src="DVF",
        group_dest="DVF",
        name="CASE_000",
        dataset=source,
        patch=None,
        transforms=transforms,
        data_augmentations_list=[],
    )


def test_the_statistic_is_per_component(tmp_path: Path) -> None:
    source = _field(tmp_path)
    stats = source.read_data_statistics("DVF", "CASE_000")
    volume = source.read_data("DVF", "CASE_000")[0]
    np.testing.assert_allclose(stats["mean_per_channel"], volume.reshape(3, -1).mean(axis=1), rtol=0, atol=1e-4)
    # The pooled figure stays what it always was.
    np.testing.assert_allclose(stats["mean"], volume.mean(), rtol=0, atol=1e-4)


def test_a_user_stage_streams_and_equals_the_whole_volume(tmp_path: Path) -> None:
    source = _field(tmp_path)
    manager = _manager(source, [_CentreAndScale(step=0.25), Write(f"{tmp_path / 'out'}:h5")])
    assert manager.stream_refusal(0, apply_augmentations=False) is None
    assert manager.materialize() is True, "a per-component centring is a value map: it must stream"

    volume = source.read_data("DVF", "CASE_000")[0].astype(np.float32)
    expected = -0.25 * (volume - volume.reshape(3, -1).mean(axis=1).reshape(3, 1, 1, 1))
    got = Dataset(tmp_path / "out", "h5").read_data("DVF", "CASE_000")[0]
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-4)
    # Centred per component: each component's mean is now zero, which a pooled mean cannot give.
    np.testing.assert_allclose(got.reshape(3, -1).mean(axis=1), np.zeros(3), rtol=0, atol=1e-4)


def test_a_streamed_user_stage_never_reads_the_volume(tmp_path: Path, monkeypatch) -> None:
    source = _field(tmp_path)

    def refuse(*args, **kwargs):
        raise AssertionError("the user stage assembled the volume")

    # Installed before the manager exists, not after: building it resolves the chain and plans the
    # reads, and a whole-volume read there would fall outside a guard armed later.
    monkeypatch.setattr(Dataset, "read_data", refuse)
    manager = _manager(source, [_CentreAndScale(step=0.25), Write(f"{tmp_path / 'out'}:h5")])
    assert manager.materialize() is True

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

"""Tests for ``konfai.data.transform.Foreign``: a transform from another framework.

The stand-ins here carry the calling convention torchvision, MONAI's array transforms and TorchIO
share -- a class callable on one tensor -- rather than any one of those libraries, none of which
KonfAI depends on.
"""

import numpy as np
import pytest
import torch
from konfai.data.transform import Foreign, LocalityKind, TransformLoader
from konfai.utils.dataset import Attribute
from konfai.utils.errors import TransformError


class Scale:
    """A foreign transform: callable on one tensor, returns the transformed tensor."""

    def __init__(self, factor: float = 1.0) -> None:
        self.factor = factor

    def __call__(self, img):
        return img * self.factor


class ToNumpy:
    """A foreign transform returning an array rather than a tensor, as several frameworks do."""

    def __call__(self, img):
        return np.asarray(img) + 1.0


class Halve:
    """A foreign transform that resizes: what Foreign's contract does not cover."""

    def __call__(self, img):
        return img[..., : img.shape[-1] // 2]


def _volume() -> torch.Tensor:
    return torch.arange(1 * 4 * 5 * 6, dtype=torch.float32).reshape(1, 4, 5, 6)


def _load(tmp_path, monkeypatch, class_name: str, args: dict):
    """Build a transform the way a run does: from a config that names the class."""
    classpath = f"{__name__}:{class_name}"
    body = "".join(f"      {key}: {value}\n" for key, value in args.items())
    config = tmp_path / "Config.yml"
    config.write_text(f"t:\n  {classpath}:\n{body or '    {}'}\n")
    monkeypatch.setenv("KONFAI_config_file", str(config))
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    return TransformLoader().get_transform(classpath, "t")


def test_the_loader_wraps_a_class_that_is_not_a_transform(tmp_path, monkeypatch) -> None:
    # The config names the class where a transform goes; nothing says it is foreign.
    transform = _load(tmp_path, monkeypatch, "Scale", {"factor": 3.0})
    assert isinstance(transform, Foreign)
    volume = _volume()
    assert torch.equal(transform("CT", volume, Attribute()), volume * 3.0)


def test_the_loader_reads_the_arguments_of_the_class(tmp_path, monkeypatch) -> None:
    transform = _load(tmp_path, monkeypatch, "Scale", {"factor": -1.0})
    assert transform.transform.factor == -1.0


def test_foreign_returns_a_tensor_whatever_the_class_returns(tmp_path, monkeypatch) -> None:
    result = _load(tmp_path, monkeypatch, "ToNumpy", {})("CT", _volume(), Attribute())
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, _volume() + 1.0)


def test_foreign_leaves_the_geometry_as_it_stands(tmp_path, monkeypatch) -> None:
    # A transform of the intensities alone does not move a voxel, so it states nothing about them.
    attributes = Attribute()
    attributes["Origin"] = np.asarray([-3.0, 5.0, 11.0])
    attributes["Spacing"] = np.asarray([1.5, 1.5, 2.0])
    before = dict(attributes)
    _load(tmp_path, monkeypatch, "Scale", {"factor": 2.0})("CT", _volume(), attributes)
    assert dict(attributes) == before


def test_foreign_refuses_a_class_that_changes_the_shape(tmp_path, monkeypatch) -> None:
    # The shape is what the patch grid is planned on, and this contract cannot state a new one.
    with pytest.raises(TransformError) as error:
        _load(tmp_path, monkeypatch, "Halve", {})("CT", _volume(), Attribute())
    assert "transform_shape" in str(error.value)


def test_foreign_reads_the_whole_volume(tmp_path, monkeypatch) -> None:
    # A foreign class states nothing about where its output reads from, so it reads everywhere.
    transform = _load(tmp_path, monkeypatch, "Scale", {"factor": 1.0})
    assert transform.patch_locality(Attribute()).kind is LocalityKind.WHOLE_VOLUME

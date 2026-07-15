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

"""Tests for ``konfai.data.augmentation.Foreign``: an augmentation from another framework.

A class draws through one of two routes, and the stand-ins here carry one each: the interpreter's
global state, which torchvision's transforms and TorchIO's draw from, and a state of the class's
own reached through ``set_random_state``, which MONAI's Randomizable holds. A stand-in for the
global route alone would pass while the other route silently drew twice.
"""

import numpy as np
import pytest
import torch
from konfai.data.augmentation import Foreign
from konfai.utils.dataset import Attribute
from konfai.utils.errors import AugmentationError
from konfai.utils.utils import get_module

_SPATIAL = [4, 5, 6]
_COPIES = 3


class GlobalNoise:
    """A foreign augmentation drawing from the interpreter's global state, as torchvision does."""

    def __init__(self, std: float = 1.0) -> None:
        self.std = std

    def __call__(self, img):
        return img + torch.randn(img.shape) * self.std


class OwnStateNoise:
    """A foreign augmentation holding its own state, as MONAI's Randomizable does."""

    def __init__(self, std: float = 1.0) -> None:
        self.std = std
        self.rng = np.random.RandomState()

    def set_random_state(self, seed=None, state=None):
        self.rng = np.random.RandomState(seed)
        return self

    def __call__(self, img):
        return img + torch.as_tensor(self.rng.normal(0.0, self.std, tuple(img.shape)), dtype=img.dtype)


class Resize:
    """A foreign augmentation that draws onto another grid: what Foreign's contract does not cover."""

    def __call__(self, img):
        return img[..., :-1]


def _volume() -> torch.Tensor:
    return torch.arange(1 * 4 * 5 * 6, dtype=torch.float32).reshape(1, *_SPATIAL)


def _draw(augmentation: Foreign) -> None:
    augmentation.state_init(0, [list(_SPATIAL)] * _COPIES, [Attribute()] * _COPIES)


def _apply(augmentation: Foreign, volume: torch.Tensor) -> list[torch.Tensor]:
    return [augmentation.compute("case", 0, a, volume.clone()) for a in range(_COPIES)]


def _foreign(cls: type, **args) -> Foreign:
    """Build one the way the loader does: the class is constructed, then handed over wrapped."""
    augmentation = Foreign(cls(**args), f"{__name__}:{cls.__name__}")
    augmentation.load(1.0)
    return augmentation


@pytest.mark.parametrize("cls", [GlobalNoise, OwnStateNoise], ids=["global-state", "own-state"])
def test_foreign_hands_every_group_of_a_case_the_same_draw(cls: type) -> None:
    # The image and its label are two groups of one case: a draw applied to one is the draw the other
    # gets, whichever state the class draws from. A second draw here is a label off its image.
    volume = _volume()
    augmentation = _foreign(cls, std=5.0)
    _draw(augmentation)
    image = _apply(augmentation, volume)
    _draw(augmentation)
    label = _apply(augmentation, volume)
    for a in range(_COPIES):
        assert torch.equal(image[a], label[a])


@pytest.mark.parametrize("cls", [GlobalNoise, OwnStateNoise], ids=["global-state", "own-state"])
def test_foreign_draws_a_copy_of_its_own_for_each_copy(cls: type) -> None:
    volume = _volume()
    augmentation = _foreign(cls, std=5.0)
    _draw(augmentation)
    copies = _apply(augmentation, volume)
    assert not torch.equal(copies[0], volume)
    for a in range(1, _COPIES):
        assert not torch.equal(copies[0], copies[a])


@pytest.mark.parametrize("cls", [GlobalNoise, OwnStateNoise], ids=["global-state", "own-state"])
def test_foreign_draws_again_on_the_next_epoch(cls: type) -> None:
    volume = _volume()
    augmentation = _foreign(cls, std=5.0)
    _draw(augmentation)
    first = _apply(augmentation, volume)
    augmentation.reset_state(0)
    _draw(augmentation)
    second = _apply(augmentation, volume)
    for a in range(_COPIES):
        assert not torch.equal(first[a], second[a])


def test_foreign_refuses_a_class_that_draws_onto_another_grid() -> None:
    augmentation = _foreign(Resize)
    _draw(augmentation)
    with pytest.raises(AugmentationError) as error:
        augmentation.compute("case", 0, 0, _volume())
    assert "_state_init" in str(error.value)


def test_foreign_cannot_be_undone() -> None:
    augmentation = _foreign(GlobalNoise)
    _draw(augmentation)
    with pytest.raises(AugmentationError) as error:
        augmentation.inverse(0, 0, _volume())
    assert "_inverse" in str(error.value)


@pytest.mark.parametrize(
    "library, classpath, args",
    [
        # Holds a random state of its own, reached through set_random_state.
        ("monai", "monai.transforms:RandGaussianNoise", {"prob": 1.0, "std": 5.0}),
        # Draws from the interpreter's global state.
        ("torchio", "torchio:RandomNoise", {"std": 5.0}),
    ],
    ids=["monai-RandGaussianNoise", "torchio-RandomNoise"],
)
def test_foreign_hands_a_real_framework_the_same_draw(library: str, classpath: str, args: dict) -> None:
    # The stand-ins model the two routes; these hold the contract to a library that ships each.
    pytest.importorskip(library)
    volume = _volume()
    module, name = get_module(classpath, "")
    augmentation = Foreign(getattr(module, name)(**args), classpath)
    augmentation.load(1.0)
    _draw(augmentation)
    image = _apply(augmentation, volume)
    _draw(augmentation)
    label = _apply(augmentation, volume)
    assert not torch.equal(image[0], volume)
    for a in range(_COPIES):
        assert torch.equal(image[a], label[a])


class GatedNoise:
    """A foreign augmentation with a gate of its own, as MONAI's Rand* classes have."""

    def __init__(self, prob: float = 0.1, std: float = 1.0) -> None:
        self.prob = prob
        self.std = std

    def __call__(self, img):
        if torch.rand(()) >= self.prob:
            return img
        return img + torch.randn(img.shape) * self.std


def test_a_foreign_gate_is_the_only_one(tmp_path, monkeypatch) -> None:
    # A foreign class brings all of its randomness, the gate included. KonfAI selecting the copies as
    # well would compose with it: one half would run as one quarter, and nothing would say so.
    config = tmp_path / "Config.yml"
    config.write_text(
        "Trainer:\n  Dataset:\n    augmentations:\n      A:\n        nb: 400\n"
        "        data_augmentations:\n"
        f"          {__name__.replace('.', '%2E')}:GatedNoise:\n"
        "            prob: 0.5\n            std: 9.0\n"
    )
    monkeypatch.setenv("KONFAI_config_file", str(config))
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    monkeypatch.setenv("KONFAI_ROOT", "Trainer")
    from konfai.data.augmentation import DataAugmentationsList
    from konfai.utils.config import apply_config

    augmentations = apply_config("Trainer.Dataset.augmentations.A")(DataAugmentationsList)()
    augmentations.prepare("A")
    augmentation = augmentations.data_augmentations[0]
    assert isinstance(augmentation, Foreign)

    copies = 400
    volume = torch.zeros(1, *_SPATIAL)
    augmentation.state_init(0, [list(_SPATIAL)] * copies, [Attribute()] * copies)
    applied = sum(
        1 for a in range(copies) if not torch.equal(augmentation.compute("case", 0, a, volume.clone()), volume)
    )
    assert 0.4 < applied / copies < 0.6


@pytest.mark.parametrize("cls", [GlobalNoise, OwnStateNoise], ids=["global-state", "own-state"])
def test_foreign_gives_the_process_back_the_random_state_it_had(cls: type) -> None:
    # The global state belongs to the run. Seeded and left where the class stopped, the two groups of
    # one case leave it in the same place, and whatever draws next draws twice the same -- the model
    # included, since torch's seed reaches the devices.
    augmentation = _foreign(cls, std=5.0)
    _draw(augmentation)
    volume = _volume()

    before = torch.random.get_rng_state()
    augmentation.compute("case", 0, 0, volume.clone())
    assert torch.equal(before, torch.random.get_rng_state())

    after_image = torch.rand(3)
    augmentation.compute("case", 0, 0, volume.clone())
    after_label = torch.rand(3)
    assert not torch.equal(after_image, after_label)

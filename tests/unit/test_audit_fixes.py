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

"""Regression tests for the audit bug-fixes (see AUDIT.md §5)."""

import inspect
import os
import subprocess
import sys

os.environ.setdefault("KONFAI_config_file", "/tmp/konfai-none.yml")
os.environ.setdefault("KONFAI_CONFIG_MODE", "Done")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from konfai.data.augmentation import Flip, Rotate  # noqa: E402
from konfai.data.data_manager import DataTrain  # noqa: E402
from konfai.data.transform import Padding, ResampleToShape, Standardize  # noqa: E402
from konfai.network.blocks import Select, Unsqueeze  # noqa: E402
from konfai.utils.dataset import Attribute  # noqa: E402
from konfai.utils.errors import ConfigError, MeasureError  # noqa: E402


def test_vae_latent_uses_gaussian_noise():
    """#3 LatentDistributionZ must sample N(0,1), not U[0,1]."""
    from konfai.network.blocks import LatentDistribution

    layer = LatentDistribution.LatentDistributionZ()
    mu = torch.zeros(200_000)
    log_std = torch.zeros(200_000)
    z = layer(mu, log_std)  # == epsilon
    assert abs(float(z.mean())) < 0.05, "mean should be ~0"
    assert abs(float(z.std()) - 1.0) < 0.05, "std should be ~1 (Gaussian), not ~0.29 (uniform)"


def test_standardize_explicit_scalar_stats():
    """#5 Standardize with explicit scalar mean/std must not crash."""
    t = Standardize(lazy=False, mean=[10.0], std=[2.0])
    x = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    out = t("c", x.clone(), Attribute())
    assert torch.allclose(out, (x - 10.0) / 2.0)


def test_standardize_explicit_per_channel_stats():
    """#5 Per-channel mean/std broadcast over the channel axis."""
    t = Standardize(lazy=False, mean=[10.0, 20.0], std=[2.0, 4.0])
    x = torch.zeros(2, 3, 4)
    x[0] = 10.0
    x[1] = 20.0
    out = t("c", x.clone(), Attribute())
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_rotate_converts_degrees_to_radians():
    """#6 A 90-degree rotation must yield [[0,-1],[1,0]], not cos/sin of 90 radians."""
    rot = Rotate(a_min=90.0, a_max=90.0, is_quarter=False)
    rot._state_init(0, [[8, 8]], [Attribute()])
    block = rot.matrix[0][0][0, :2, :2]
    assert torch.allclose(block, torch.tensor([[0.0, -1.0], [1.0, 0.0]]), atol=1e-5)


def test_predict_evaluate_expose_tensorboard_param():
    """#7 CLI -tb/--tensorboard (dest 'tensorboard') must reach predict()/evaluate()."""
    from konfai.evaluator import evaluate
    from konfai.predictor import predict

    for fn in (predict, evaluate):
        params = inspect.signature(fn).parameters
        assert "tensorboard" in params, f"{fn.__name__} must accept 'tensorboard'"
        assert "tb" not in params, f"{fn.__name__} must not use the old 'tb' name"


def test_unsqueeze_forward_accepts_tensor():
    """#8 Unsqueeze.forward(tensor) must work on a single tensor."""
    assert Unsqueeze(dim=1)(torch.randn(3, 4)).shape == (3, 1, 4)


def test_resample_to_shape_does_not_mutate_config():
    """#9 transform_shape must not write resolved dims back into the shared instance config."""
    resampler = ResampleToShape(shape=[0, 16, 16])
    before = resampler.shape.clone()
    attributes = Attribute()
    attributes["Spacing"] = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    out = resampler.transform_shape("CT", "case", [8, 16, 16], attributes)
    assert out[0] == 8  # sentinel 0 resolved to the input dim for this call
    assert torch.equal(resampler.shape, before), "self.shape must stay [0, 16, 16] for the next case"


def test_select_squeezes_size_one_dims_by_size():
    """#12 Select must squeeze dimensions whose size is 1, not the dim at index 1."""
    out = Select([slice(0, 1), slice(None), slice(None)])(torch.randn(1, 5, 6))
    assert out.shape == (5, 6)
    # a tensor with no size-1 dims is unchanged
    out2 = Select([slice(None), slice(None)])(torch.randn(4, 5))
    assert out2.shape == (4, 5)


def test_augmentation_resamples_after_reset_state():
    """#1 Augmentation parameters must be re-sampled each epoch via reset_state.

    Within an epoch ``state_init`` caches the per-case draw so every patch shares
    one transform; ``reset_state`` must clear that cache so the next epoch draws
    fresh parameters (previously ``who_index`` was never cleared → frozen forever).
    """
    aug = Flip([1.0, 1.0, 1.0])
    aug.load(1.0)

    aug.state_init(0, [[4, 4, 4]], [Attribute()])
    first = aug.flip[0]
    # Re-running state_init without a reset returns the cached draw unchanged.
    aug.state_init(0, [[4, 4, 4]], [Attribute()])
    assert aug.flip[0] is first

    aug.reset_state(0)
    assert 0 not in aug.who_index
    aug.state_init(0, [[4, 4, 4]], [Attribute()])
    assert 0 in aug.who_index
    assert aug.flip[0] is not first  # a fresh draw replaced the cached one


def test_update_scheduler_empty_raises_config_error():
    """update_scheduler on an empty schedule must raise a clear ConfigError."""
    from konfai.network.network import Measure

    with pytest.raises(ConfigError):
        Measure.update_scheduler(None, {}, 0)  # type: ignore[arg-type]


def test_update_scheduler_past_last_window_clamps_to_last():
    """Past every configured window, the last scheduler is selected (no crash)."""
    from konfai.metric.schedulers import Constant
    from konfai.network.network import Measure

    s0, s1 = Constant(1.0), Constant(2.0)
    schedulers = {s0: 3, s1: 3}  # active windows [0,3) and [3,6)
    assert Measure.update_scheduler(None, schedulers, 4) is s1  # type: ignore[arg-type]
    assert Measure.update_scheduler(None, schedulers, 100) is s1  # type: ignore[arg-type]


def test_missing_metric_dependency_raises_actionable_error():
    """Optional criterion deps must surface an actionable MeasureError, not ImportError."""
    from konfai.metric.measure import _require_optional

    with pytest.raises(MeasureError) as excinfo:
        _require_optional("konfai_definitely_missing_pkg_zzz", criterion="SSIM", extra="ssim")
    message = str(excinfo.value)
    assert "SSIM" in message
    assert "konfai[ssim]" in message


def test_load_state_dict_warm_starts_resized_layer_and_keeps_siblings():
    """#2 A resized layer must warm-start, and sibling layers must still load.

    The bug checked ``isinstance(module, Linear)`` (the parent) instead of the
    child, and used an early ``return`` that aborted loading the remaining
    siblings of a resized layer.
    """
    from konfai.network.network import Network

    class _Net(Network):
        def __init__(self, fc_out: int) -> None:
            super().__init__(in_channels=1)
            self.add_module("fc", torch.nn.Linear(4, fc_out))
            self.add_module("head", torch.nn.Linear(4, 2))

    old = _Net(fc_out=4)
    # Network.state_dict() wraps the flat params under the network name; load_state_dict
    # consumes that inner flat dict ("fc.weight", ...).
    inner = next(iter(old.state_dict().values()))
    checkpoint = {key: value.clone() for key, value in inner.items()}

    new = _Net(fc_out=6)  # fc output grows 4 -> 6 (resized); head is unchanged
    new.load_state_dict(checkpoint)  # must not raise

    fc = new["fc"]
    head = new["head"]
    assert fc.weight.shape == (6, 4)
    assert torch.equal(fc.weight[:4], checkpoint["fc.weight"])  # warm-started rows
    # The sibling after the resized layer must still be loaded (old `return` skipped it).
    assert torch.equal(head.weight, checkpoint["head.weight"])
    assert torch.equal(head.bias, checkpoint["head.bias"])


def test_adaptation_sets_requires_grad_at_construction():
    """#18 Adaptation must configure requires_grad in __init__, not on every forward."""
    from konfai.models.representation.representation import Adaptation

    adaptation = Adaptation()
    # State is correct immediately after construction, before any forward pass.
    assert all(not p.requires_grad for p in adaptation.Encoder_1.parameters())
    assert all(p.requires_grad for p in adaptation.FCT_1.parameters())


def test_linear_vae_is_parameterized_and_variational():
    """#17 LinearVAE must be parameterized (no hardcoded dims) and sample a latent."""
    from konfai.models.generation.vae import LinearVAE

    model = LinearVAE(in_features=32, hidden_features=16, latent_dim=4)
    x = torch.randn(2, 32)
    outputs = dict(model.named_forward(x))
    assert outputs["Head.Tanh"].shape == (2, 32)  # reconstruction matches input size
    assert "Latent.mu" in outputs and "Latent.log_std" in outputs  # KL-ready outputs
    # The latent is sampled: the reconstruction differs across RNG draws.
    torch.manual_seed(0)
    first = dict(model.named_forward(x))["Head.Tanh"]
    torch.manual_seed(1)
    second = dict(model.named_forward(x))["Head.Tanh"]
    assert not torch.allclose(first, second)


def _geometry(origin: list[float], spacing: list[float]) -> Attribute:
    attributes = Attribute()
    attributes["Origin"] = np.asarray(origin, dtype=np.float64)
    attributes["Spacing"] = np.asarray(spacing, dtype=np.float64)
    attributes["Direction"] = np.eye(len(origin), dtype=np.float64).reshape(-1)
    return attributes


def test_padding_shifts_origin_along_the_padded_axes():
    """Each F.pad pair (X, Y, Z) must shift the matching (x, y, z) origin component."""
    attributes = _geometry(origin=[10.0, 20.0, 30.0], spacing=[1.0, 2.0, 4.0])

    padded = Padding(padding=[1, 0, 2, 0, 3, 0])("case", torch.zeros(1, 5, 5, 5), attributes)

    assert list(padded.shape) == [1, 8, 7, 6]
    np.testing.assert_allclose(
        attributes.get_np_array("Origin"),
        [10.0 - 1 * 1.0, 20.0 - 2 * 2.0, 30.0 - 3 * 4.0],
    )


def test_padding_after_the_data_keeps_origin():
    """Padding only on the high side of each axis must leave the origin untouched."""
    attributes = _geometry(origin=[10.0, 20.0, 30.0], spacing=[1.0, 2.0, 4.0])

    padded = Padding(padding=[0, 2, 0, 0, 0, 1])("case", torch.zeros(1, 5, 5, 5), attributes)

    assert list(padded.shape) == [1, 6, 5, 7]
    np.testing.assert_allclose(attributes.get_np_array("Origin"), [10.0, 20.0, 30.0])


_SPLIT_PROBE = """
import os
import random

os.environ.setdefault("KONFAI_config_file", "/tmp/konfai-none.yml")
os.environ.setdefault("KONFAI_CONFIG_MODE", "Done")

from konfai.data.data_manager import DataTrain

names = [f"CASE_{i:03d}" for i in range(20)]
data = DataTrain(augmentations=None, validation="0:4")
data._resolve_dataset_sources = lambda: {}
data._resolve_common_names = lambda datasets: ({}, set(names))
data._get_datasets = lambda case_names, dataset_name, augmentations: ({}, [])
random.seed(1234)
data._prepare_datasets()
print(";".join(data._prepared_train_names))
print(";".join(data._prepared_validation_names))
"""


def test_train_validation_split_is_reproducible_across_interpreters():
    """Same seed → same split, whatever the interpreter's string-hash randomization."""
    outputs = []
    for hash_seed in ("0", "424242"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        result = subprocess.run(
            [sys.executable, "-c", _SPLIT_PROBE],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
    train_names, validation_names = (line.split(";") for line in outputs[0].splitlines())
    assert len(train_names) == 16
    assert len(validation_names) == 4
    assert set(train_names).isdisjoint(validation_names)


def test_train_split_shuffle_draws_from_sorted_names(monkeypatch):
    """The seeded shuffle must receive the case names in sorted order and drive the split."""
    captured: dict[str, list[str]] = {}

    def fake_sample(population, k):
        captured["population"] = list(population)
        assert k == len(population)
        return list(reversed(population))

    monkeypatch.setattr("konfai.data.data_manager.random.sample", fake_sample)

    data = DataTrain(augmentations=None, validation="0:2")
    names = {"CASE_010", "CASE_002", "CASE_001", "CASE_005", "CASE_003"}
    data._resolve_dataset_sources = lambda: {}
    data._resolve_common_names = lambda datasets: ({}, names)
    data._get_datasets = lambda case_names, dataset_name, augmentations: ({}, [])
    data._prepare_datasets()

    assert captured["population"] == sorted(names)
    assert data._prepared_validation_names == ["CASE_010", "CASE_005"]
    assert data._prepared_train_names == ["CASE_003", "CASE_002", "CASE_001"]

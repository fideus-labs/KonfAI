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

"""Tests for ``konfai_apps.transforms``: the KonfAIInference chain stage."""

import os
import sys
import types
from pathlib import Path

import konfai_apps
import numpy as np
import pytest
import torch
from konfai.utils.dataset import Attribute
from konfai_apps.transforms import (
    DEFAULT_INFERENCE_MODEL_NAME,
    DEFAULT_INFERENCE_REPO_ID,
    KonfAIInference,
)


@pytest.fixture(autouse=True)
def _ambient_ports_survive(monkeypatch: pytest.MonkeyPatch):
    """infer_entry pops both port vars from the real environment; registering them with monkeypatch
    makes teardown put an ambient value back instead of leaking the deletion into the session."""
    monkeypatch.delenv("KONFAI_MASTER_PORT", raising=False)
    monkeypatch.delenv("KONFAI_TENSORBOARD_PORT", raising=False)


def test_konfai_inference_reassembles_channels_in_sorted_order(tmp_path, monkeypatch):
    """Per-channel outputs must be stacked in deterministic (sorted) case order."""
    sitk = pytest.importorskip("SimpleITK")

    output_dir = tmp_path / "Output"
    files = []
    for i in range(3):
        case_dir = output_dir / f"P{i:03d}"
        case_dir.mkdir(parents=True)
        array = np.full((2, 2, 2), float(i * 10), dtype=np.float32)
        path = case_dir / "Volume.mha"
        sitk.WriteImage(sitk.GetImageFromArray(array), str(path))
        files.append(path)

    # Simulate an arbitrary (here reversed) filesystem enumeration order.
    scrambled = list(reversed(files))
    monkeypatch.setattr(Path, "rglob", lambda self, pattern: iter(scrambled))

    result = KonfAIInference._reassemble_output(output_dir)

    assert list(result.shape) == [3, 2, 2, 2]
    assert float(result[0].mean()) == 0.0
    assert float(result[1].mean()) == 10.0
    assert float(result[2].mean()) == 20.0


def test_konfai_inference_default_repo_and_model_preserved():
    """Constructing without arguments keeps the current published repo/model default."""
    transform = KonfAIInference()

    assert transform.repo_id == DEFAULT_INFERENCE_REPO_ID
    assert transform.model_name == DEFAULT_INFERENCE_MODEL_NAME
    assert transform.repo_id == "VBoussot/MRSegmentator-KonfAI"
    assert transform.model_name == "MRSegmentator"


def test_konfai_inference_forwards_configured_repo_and_model(monkeypatch):
    """A custom repo/model is forwarded verbatim to the KonfAIApp spec, not the default."""
    captured = {}

    class _FakeKonfAIApp:
        def __init__(self, spec, *args):
            captured["spec"] = spec

        def infer(self, *args, **kwargs):
            captured["infer"] = (args, kwargs)

    fake_module = types.ModuleType("konfai_apps")
    fake_module.KonfAIApp = _FakeKonfAIApp
    monkeypatch.setitem(sys.modules, "konfai_apps", fake_module)

    transform = KonfAIInference(
        repo_id="acme/Custom-KonfAI",
        model_name="CustomModel",
        checkpoints_name=["fold_1"],
    )
    transform.infer_entry(Path("dataset"), Path("output"), [])

    assert captured["spec"] == "acme/Custom-KonfAI:CustomModel"


def test_konfai_inference_raises_clear_error_inside_daemon_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    transform = KonfAIInference()

    class DaemonProcess:
        daemon = True

    monkeypatch.setattr("konfai_apps.transforms.current_process", lambda: DaemonProcess())

    with pytest.raises(RuntimeError, match=r"Dataset\.num_workers: 0"):
        transform("CASE_000", torch.zeros(1, 4, 4), Attribute())


def test_konfai_inference_forwards_config_overrides_to_the_nested_run(monkeypatch: pytest.MonkeyPatch) -> None:
    # The nested run is tunable from the calling code via the generic --set mechanism (not for shrinking a
    # trained patch_size (that hurts the result), but for any legitimate config knob).
    recorded: dict[str, object] = {}

    class FakeKonfAIApp:
        def __init__(self, ref: str, download: bool, force_update: bool) -> None:
            recorded["ref"] = ref

        def infer(self, *args: object, **kwargs: object) -> None:
            recorded["config_overrides"] = kwargs.get("config_overrides")

    monkeypatch.setattr(konfai_apps, "KonfAIApp", FakeKonfAIApp)
    overrides = ["iterations=300"]
    transform = KonfAIInference(repo_id="Org/Repo", model_name="tiny", config_overrides=overrides)
    transform.infer_entry(Path("/tmp/in"), Path("/tmp/out"), [0])

    assert recorded["ref"] == "Org/Repo:tiny"
    assert recorded["config_overrides"] == overrides


def test_konfai_inference_defragments_the_nested_allocator(monkeypatch: pytest.MonkeyPatch) -> None:
    # A heavy nested model (e.g. a 3D segmentation a metric relies on) can OOM on a large volume purely from
    # allocator fragmentation; the nested run enables expandable segments so it fits without config changes.
    class FakeKonfAIApp:
        def __init__(self, ref: str, download: bool, force_update: bool) -> None:
            pass

        def infer(self, *args: object, **kwargs: object) -> None:
            pass

    monkeypatch.setattr(konfai_apps, "KonfAIApp", FakeKonfAIApp)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)

    KonfAIInference(repo_id="Org/Repo", model_name="tiny").infer_entry(Path("/tmp/in"), Path("/tmp/out"), [0])
    assert "expandable_segments:True" in os.environ["PYTORCH_CUDA_ALLOC_CONF"]

    # An explicit caller setting must win (setdefault, not overwrite).
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    KonfAIInference(repo_id="Org/Repo", model_name="tiny").infer_entry(Path("/tmp/in"), Path("/tmp/out"), [0])
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"


def test_konfai_inference_targets_the_ranks_own_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """A TRANSFORM chain routes its tensors to the rank's device, and the nested inference
    must run there too, not on every device the launch was given (a two-GPU prediction per rank)."""
    import konfai_apps.transforms as transform_module

    pytest.importorskip("SimpleITK")
    launched: dict[str, list[int]] = {}

    class _Process:
        exitcode = 0

        def __init__(self, target, args):
            launched["gpu"] = list(args[2])

        def start(self) -> None:
            pass

        def join(self) -> None:
            pass

    class _Context:
        Process = _Process

    monkeypatch.setattr(transform_module, "get_context", lambda _method: _Context())
    monkeypatch.setattr(transform_module, "cuda_visible_devices", lambda: [4, 7])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)  # local rank 1
    monkeypatch.setattr(KonfAIInference, "_reassemble_output", staticmethod(lambda _dir: torch.zeros(1, 2, 2, 2)))
    attributes = Attribute()
    attributes["Origin"] = np.zeros(3)
    attributes["Spacing"] = np.ones(3)
    attributes["Direction"] = np.eye(3).reshape(-1)
    KonfAIInference()("case", torch.zeros(1, 2, 2, 2), attributes)
    assert launched["gpu"] == [7], "the rank's own device, in the launch's numbering"


def test_konfai_inference_bare_name_resolves_through_core() -> None:
    """Published bundles spell the bare name ``KonfAIInference:``; core's transform package must
    keep resolving it to this class."""
    import konfai.data.transform as transform_package

    assert transform_package.KonfAIInference is KonfAIInference

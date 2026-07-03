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

"""Regression tests for transform-pipeline fixes."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import torch
from konfai.data.transform import (
    DEFAULT_INFERENCE_MODEL_NAME,
    DEFAULT_INFERENCE_REPO_ID,
    InferenceStack,
    KonfAIInference,
    ResampleToResolution,
    ResampleToShape,
)
from konfai.utils.dataset import Attribute
from konfai.utils.errors import TransformError


def test_resample_to_resolution_transform_shape_missing_spacing_raises():
    """A tensor without 'Spacing' metadata must surface a TransformError, not fall through."""
    with pytest.raises(TransformError):
        ResampleToResolution().transform_shape("group", "case", [10, 10, 10], Attribute())


def test_resample_to_shape_transform_shape_missing_spacing_raises():
    """ResampleToShape must also raise when 'Spacing' metadata is absent."""
    with pytest.raises(TransformError):
        ResampleToShape().transform_shape("group", "case", [10, 10, 10], Attribute())


def test_resample_to_resolution_transform_shape_dimension_mismatch_message():
    """The dimension-mismatch error is raised and its message interpolates the actual shape."""
    attributes = Attribute()
    attributes["Spacing"] = np.asarray([1.0, 1.0], dtype=np.float64)
    with pytest.raises(TransformError) as excinfo:
        ResampleToResolution(spacing=[1.0, 1.0]).transform_shape("group", "case", [10, 10, 10], attributes)
    assert "shape=[10, 10, 10]" in str(excinfo.value)


def test_resample_to_shape_transform_shape_dimension_mismatch_message():
    """ResampleToShape raises a formatted (f-string) message on a shape/target mismatch."""
    attributes = Attribute()
    attributes["Spacing"] = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    with pytest.raises(TransformError) as excinfo:
        ResampleToShape(shape=[4, 4]).transform_shape("group", "case", [10, 10, 10], attributes)
    message = str(excinfo.value)
    assert "shape=[10, 10, 10]" in message
    assert "target_shape" in message


def test_resample_to_shape_inverse_without_spacing_metadata():
    """Inverting a resample must not pop a 'Spacing' the forward pass never pushed."""
    resampler = ResampleToShape(shape=[4, 4, 4])
    attributes = Attribute()  # no image metadata at all
    tensor = torch.arange(8 * 8 * 8, dtype=torch.float32).reshape(1, 8, 8, 8)

    forward = resampler("case", tensor, attributes)
    assert list(forward.shape) == [1, 4, 4, 4]

    restored = resampler.inverse("case", forward, attributes)
    assert list(restored.shape) == [1, 8, 8, 8]


def test_resample_to_shape_inverse_pops_pushed_spacing():
    """When 'Spacing' exists, the inverse removes the version the forward pass pushed."""
    resampler = ResampleToShape(shape=[4, 4, 4])
    attributes = Attribute()
    attributes["Spacing"] = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    tensor = torch.zeros(1, 8, 8, 8)

    resampler("case", tensor, attributes)
    # image_shape / shape == 2 on every axis, so the resampled spacing doubles.
    np.testing.assert_allclose(attributes.get_np_array("Spacing"), [2.0, 2.0, 2.0])

    resampler.inverse("case", torch.zeros(1, 4, 4, 4), attributes)
    # The pushed spacing is popped, restoring the original resolution.
    np.testing.assert_allclose(attributes.get_np_array("Spacing"), [1.0, 1.0, 1.0])


def test_inference_stack_super_init_enables_dataset_fallback():
    """InferenceStack must inherit Transform's 'datasets' list for the fallback write path."""
    stack = InferenceStack(dataset="", name="pred", mode="mean")
    assert stack.datasets == []

    written: dict[str, object] = {}

    class _FakeDataset:
        def write(self, group, name, data, cache_attribute):
            written["group"] = group
            written["name"] = name
            written["shape"] = tuple(data.shape)

    stack.set_datasets([object(), _FakeDataset()])

    tensors = torch.stack(
        [torch.full((1, 2, 2), 3.0), torch.full((1, 2, 2), 5.0)],
        dim=0,
    )  # shape [2, 1, 2, 2]: two stacked predictions
    out = stack("pred", tensors, Attribute())

    assert written["group"] == "InferenceStack"
    assert written["name"] == "pred"
    assert written["shape"] == (2, 2, 2)
    # 'mean' mode averages the two predictions element-wise.
    assert torch.allclose(out, torch.full((1, 2, 2), 4.0))


def test_konfai_inference_reassembles_channels_in_sorted_order(tmp_path, monkeypatch):
    """Per-channel outputs must be stacked in deterministic (sorted) case order."""
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

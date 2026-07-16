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

"""Tests for the app-bundle assembler."""

import json

import pytest
from konfai.utils.errors import AppMetadataError
from konfai_apps.bundle import assemble_bundle

VALID_META = {
    "display_name": "Synthesis: MR",
    "description": "d",
    "short_description": "s",
    "tta": 0,
    "mc_dropout": 0,
}


def _write(path, obj):
    path.write_text(json.dumps(obj))
    return path


def test_assemble_bundle_layout(tmp_path):
    app_json = _write(tmp_path / "app.json", VALID_META)
    config = tmp_path / "Prediction.yml"
    config.write_text("Predictor: {}\n")
    checkpoint = tmp_path / "CV_0.pt"
    checkpoint.write_bytes(b"weights")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("torch\n")
    model_py = tmp_path / "Model.py"
    model_py.write_text("# custom\n")

    bundle = assemble_bundle(
        "MR", tmp_path / "out", app_json, [str(config)], [str(checkpoint)],
        model_py=str(model_py), requirements=str(requirements),
    )

    assert bundle == tmp_path / "out" / "MR"
    for expected in ("app.json", "Prediction.yml", "CV_0.pt", "Model.py", "requirements.txt"):
        assert (bundle / expected).exists(), expected
    # `models` auto-filled from the provided checkpoints
    assert json.loads((bundle / "app.json").read_text())["models"] == ["CV_0.pt"]


def test_missing_required_keys_raises(tmp_path):
    app_json = _write(tmp_path / "app.json", {"display_name": "x"})
    with pytest.raises(AppMetadataError):
        assemble_bundle("MR", tmp_path / "out", app_json, [], [])


def test_models_mismatch_raises(tmp_path):
    app_json = _write(tmp_path / "app.json", {**VALID_META, "models": ["CV_0.pt", "CV_1.pt"]})
    checkpoint = tmp_path / "CV_0.pt"
    checkpoint.write_bytes(b"w")
    with pytest.raises(AppMetadataError):
        assemble_bundle("MR", tmp_path / "out", app_json, [], [str(checkpoint)])


def test_derive_requirements_keeps_only_extra(tmp_path):
    from konfai_apps.bundle import derive_requirements

    model_py = tmp_path / "Model.py"
    model_py.write_text(
        "import os\n"
        "import torch\n"
        "import numpy as np\n"
        "import segmentation_models_pytorch as smp\n"
        "from konfai.network import network\n"
        "import skimage\n"
    )
    # stdlib (os), konfai-provided (torch/numpy), and konfai itself are excluded.
    assert derive_requirements([model_py]) == ["scikit-image", "segmentation-models-pytorch"]


def test_derive_onnx_params_from_config():
    from konfai_apps.bundle import _derive_onnx_params

    config = {
        "Predictor": {
            "Model": {"classpath": "Model:UNetpp", "UNetpp": {"nb_channel": 5}},
            "Dataset": {"Patch": {"patch_size": [1, 256, 256], "extend_slice": 2}},
            "Model_unused_patch": {"ModelPatch": {"patch_size": [128, 128, 128]}},
        }
    }
    patch_size, in_channels, extend_slice = _derive_onnx_params(config, "Predictor")
    assert patch_size == [256, 256]  # singleton slice dim dropped (2.5D)
    assert in_channels == 5
    assert extend_slice == 2

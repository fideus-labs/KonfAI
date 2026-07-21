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
        "MR",
        tmp_path / "out",
        app_json,
        [str(config)],
        [str(checkpoint)],
        model_py=str(model_py),
        requirements=str(requirements),
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
            "Dataset": {"Patch": {"patch_size": [1, 256, 256], "extend_slice": 2, "pad_value": -2}},
            "Model_unused_patch": {"ModelPatch": {"patch_size": [128, 128, 128]}},
        }
    }
    patch_size, in_channels, extend_slice, pad_value = _derive_onnx_params(config, "Predictor")
    assert patch_size == [256, 256]  # singleton slice dim dropped (2.5D)
    assert in_channels == 5
    assert extend_slice == 2
    assert pad_value == -2.0  # the config's border-pad value reaches the manifest


def test_derive_overlap_broadcasts_scalar_and_drops_singleton():
    from konfai_apps.bundle import _derive_overlap

    assert _derive_overlap({"overlap": 32}, [96, 128, 160]) == [32, 32, 32]  # scalar broadcast
    assert _derive_overlap({"overlap": [8, 8]}, [64, 64]) == [8, 8]  # per-axis list kept
    # a full-rank overlap carrying the 2.5D singleton slice axis: kept axes match patch_size
    assert _derive_overlap({"overlap": [0, 16, 16], "patch_size": [1, 256, 256]}, [256, 256]) == [16, 16]
    assert _derive_overlap({}, [96, 128, 160]) is None  # no overlap declared


def test_derive_blend_reads_patch_combine():
    from konfai_apps.bundle import _derive_blend

    cfg = {"Predictor": {"outputs_dataset": {"H": {"OutputDataset": {"patch_combine": "Gaussian"}}}}}
    assert _derive_blend(cfg, "Predictor") == "Gaussian"
    assert _derive_blend({"Predictor": {"outputs_dataset": {"O": {"patch_combine": "None"}}}}, "Predictor") is None
    assert _derive_blend({"Predictor": {}}, "Predictor") is None


def test_transform_manifest_maps_the_pipeline_to_runtime_ops():
    from konfai_apps.bundle import _transform_manifest

    config = {
        "Predictor": {
            "Dataset": {
                "groups_src": {
                    "Volume_0": {
                        "groups_dest": {
                            "Volume": {
                                "transforms": {
                                    "TensorCast": {"dtype": "float32"},
                                    "ResampleToResolution": {"spacing": [3, 3, 3], "inverse": True},
                                    "Standardize": {"mean": "None", "std": "None"},
                                }
                            }
                        }
                    }
                }
            },
            "outputs_dataset": {
                "SegHead": {"OutputDataset": {"final_transforms": {"Softmax": {"dim": 0}, "Argmax": {"dim": 0}}}}
            },
        }
    }
    manifest, folds = _transform_manifest(config, "Predictor")
    assert folds == []  # every transform here is a runtime op; nothing folded into the graph
    assert manifest["preprocessing"] == [
        {"op": "cast", "dtype": "float32"},
        {"op": "resample", "spacing": [3.0, 3.0, 3.0], "inverse": True},
        {"op": "standardize"},  # mean/std unset -> computed at runtime
    ]
    assert manifest["postprocessing"] == [{"op": "softmax", "dim": 0}, {"op": "argmax", "dim": 0}]


def test_transform_manifest_reads_canonical_and_before_reduction_post():
    from konfai_apps.bundle import _transform_manifest

    # A disjoint (merge_labels) ensemble: Canonical -> a runtime op; the per-fold argmax lives in
    # before_reduction_transforms (each fold argmaxes to a label map before the merge), not final_transforms.
    config = {
        "Predictor": {
            "Dataset": {
                "g": {
                    "transforms": {
                        "Canonical": {"inverse": True},
                        "ResampleToResolution": {"spacing": [1.5, 1.5, 1.5], "inverse": True},
                    }
                }
            },
            "outputs_dataset": {
                "H": {
                    "OutputDataset": {
                        "before_reduction_transforms": {"Softmax": {"dim": 0}, "Argmax": {"dim": 0}},
                        "final_transforms": "None",
                    }
                }
            },
        }
    }
    manifest, _folds = _transform_manifest(config, "Predictor")
    assert {"op": "canonical", "inverse": True} in manifest["preprocessing"]
    assert manifest["postprocessing"] == [{"op": "softmax", "dim": 0}, {"op": "argmax", "dim": 0}]


def test_derive_reduction_reads_the_ensemble_reduction():
    from konfai_apps.bundle import _derive_reduction

    def cfg(after):
        return {"Predictor": {"outputs_dataset": {"H": {"OutputDataset": {"after_reduction_transforms": after}}}}}

    # The multi-model reduction is read from the output transforms, never hard-coded per app.
    assert _derive_reduction(cfg({"MergeLabels": {}}), "Predictor") == "merge_labels"
    assert _derive_reduction(cfg({"InferenceStack": {"mode": "mean"}}), "Predictor") == "mean"
    assert _derive_reduction(cfg("None"), "Predictor") is None
    assert _derive_reduction({"Predictor": {}}, "Predictor") is None


def test_transform_manifest_refuses_an_unportable_transform():
    import pytest
    from konfai_apps.bundle import AppMetadataError, _transform_manifest

    config = {"Predictor": {"Dataset": {"g": {"transforms": {"SomeCustomTransform": {"x": 1}}}}, "outputs_dataset": {}}}
    with pytest.raises(AppMetadataError, match="no portable runtime op"):
        _transform_manifest(config, "Predictor")

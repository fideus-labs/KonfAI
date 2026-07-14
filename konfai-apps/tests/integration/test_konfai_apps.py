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

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
KONFAI_APPS_ROOT = REPO_ROOT / "konfai-apps"
WORKFLOW_ASSETS_DIR = REPO_ROOT / "tests" / "assets" / "Workflows"
SimpleITK = pytest.importorskip("SimpleITK")


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    package_paths = f"{KONFAI_APPS_ROOT}{os.pathsep}{REPO_ROOT}"
    env["PYTHONPATH"] = package_paths if not pythonpath else f"{package_paths}{os.pathsep}{pythonpath}"
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def _write_image(path: Path, array: np.ndarray, pixel_id: int) -> None:
    image = SimpleITK.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    image = SimpleITK.Cast(image, pixel_id)
    SimpleITK.WriteImage(image, str(path))


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise AssertionError(f"Expected JSON file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _single_case_metric(path: Path, *, expected_metric_suffix: str) -> float:
    data = _read_json(path)
    assert "case" in data and len(data["case"]) == 1, path
    metric_name, case_values = next(iter(data["case"].items()))
    assert metric_name.endswith(expected_metric_suffix), metric_name
    assert case_values.keys() == {"P000"}
    value = float(case_values["P000"])
    assert np.isfinite(value), metric_name
    return value


def _write_local_synthesis_app(app_dir: Path) -> None:
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Tiny Synth",
                "description": "Tiny local synthesis app for integration testing",
                "short_description": "Tiny synth",
                "tta": 0,
                "mc_dropout": 0,
                "models": ["tiny_0.pt", "tiny_1.pt"],
                "inputs": {
                    "Volume_0": {
                        "display_name": "MR",
                        "volume_type": "VOLUME",
                        "required": True,
                    }
                },
                "outputs": {
                    "sCT": {
                        "display_name": "sCT",
                        "volume_type": "VOLUME",
                        "required": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (app_dir / "Prediction.yml").write_text(
        textwrap.dedent(
            """
            Predictor:
              Model:
                classpath: TinySynth:TinySynthNet
                TinySynthNet:
                  outputs_criterions: None
              Dataset:
                groups_src:
                  Volume_0:
                    groups_dest:
                      MR:
                        transforms: None
                        patch_transforms: None
                        is_input: true
                augmentations: None
                Patch:
                  patch_size: [1, 16, 16]
                  overlap: None
                  mask: None
                  pad_value: 0
                  extend_slice: 0
                subset: None
                filter: None
                dataset_filenames:
                  - ./Dataset:a:mha
                use_cache: false
                batch_size: 16
              outputs_dataset:
                Head:Tanh:
                  OutputDataset:
                    name_class: OutSameAsGroupDataset
                    before_reduction_transforms: None
                    after_reduction_transforms:
                      InferenceStack:
                        dataset: Predictions/TinyApp/Output:mha
                        name: InferenceStack
                        mode: mean
                    final_transforms:
                      TensorCast:
                        dtype: float32
                        inverse: false
                    dataset_filename: Dataset:mha
                    group: sCT
                    same_as_group: Volume_0:MR
                    patch_combine: None
                    inverse_transform: false
                    reduction: Mean
                    Mean: {}
              train_name: TinyApp
              manual_seed: 0
              gpu_checkpoints: None
              autocast: false
              combine: Concat
              data_log: None
              Concat: {}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (app_dir / "Evaluation.yml").write_text(
        textwrap.dedent(
            """
            Evaluator:
              metrics:
                Output:
                  targets_criterions:
                    Reference;Mask:
                      criterions_loader:
                        MAE:
                          reduction: mean
              Dataset:
                groups_src:
                  Mask_0:
                    groups_dest:
                      Mask:
                        transforms: None
                  Volume_0:
                    groups_dest:
                      Output:
                        transforms: None
                  Reference_0:
                    groups_dest:
                      Reference:
                        transforms: None
                subset: None
                dataset_filenames:
                  - ./Dataset:a:mha
                validation: None
              train_name: TinyApp
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (app_dir / "Uncertainty.yml").write_text(
        textwrap.dedent(
            """
            Evaluator:
              metrics:
                Uncertainty:
                  targets_criterions:
                    None:
                      criterions_loader:
                        Mean:
                          name: Uncertainty
              Dataset:
                groups_src:
                  Volume_0:
                    groups_dest:
                      Uncertainty:
                        transforms:
                          Variance: {}
                          Save:
                            dataset: ./Uncertainties/TinyApp/Output:mha
                            group: None
                subset: None
                dataset_filenames:
                  - ./Dataset:mha
                validation: None
              train_name: TinyApp
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (app_dir / "TinySynth.py").write_text(
        (WORKFLOW_ASSETS_DIR / "TinySynth.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    spec = importlib.util.spec_from_file_location("TinySynth", app_dir / "TinySynth.py")
    if spec is None or spec.loader is None:
        raise AssertionError("Failed to load TinySynth test module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.TinySynthNet()
    with torch.no_grad():
        model.Projection.weight.fill_(1.0)
        model.Projection.bias.zero_()

    checkpoint = {"Model": model.state_dict()}
    torch.save(checkpoint, app_dir / "tiny_0.pt")
    torch.save(checkpoint, app_dir / "tiny_1.pt")


def test_konfai_apps_pipeline_is_local_and_deterministic(tmp_path: Path) -> None:
    app_dir = tmp_path / "TinySynthesisApp"
    _write_local_synthesis_app(app_dir)

    input_array = np.linspace(-0.8, 0.8, 16 * 16, dtype=np.float32).reshape(1, 16, 16)
    expected_output = np.tanh(input_array).astype(np.float32)
    mask_array = np.ones_like(input_array, dtype=np.uint8)

    input_path = tmp_path / "input.mha"
    gt_path = tmp_path / "gt.mha"
    mask_path = tmp_path / "mask.mha"
    _write_image(input_path, input_array, SimpleITK.sitkFloat32)
    _write_image(gt_path, expected_output, SimpleITK.sitkFloat32)
    _write_image(mask_path, mask_array, SimpleITK.sitkUInt8)

    cmd = [
        sys.executable,
        "-c",
        (
            f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); "
            f"sys.path.insert(0, {str(KONFAI_APPS_ROOT)!r}); "
            "from konfai_apps.cli import main_apps; main_apps()"
        ),
        "pipeline",
        str(app_dir),
        "-i",
        str(input_path),
        "-o",
        str(tmp_path),
        "--gt",
        str(gt_path),
        "--mask",
        str(mask_path),
        "--ensemble",
        "2",
        "--tta",
        "0",
        "--mc",
        "0",
        "-uncertainty",
        "--cpu",
        "1",
    ]

    result = run(cmd)
    assert result.returncode == 0, (
        "The local 'konfai-apps pipeline' command failed.\n\n"
        f"CMD: {' '.join(cmd)}\n\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )

    predicted_path = tmp_path / "Predictions" / "TinyApp" / "Dataset" / "P000" / "sCT.mha"
    inference_stack_path = tmp_path / "Predictions" / "TinyApp" / "Output" / "P000" / "InferenceStack.mha"
    evaluation_path = tmp_path / "Evaluations" / "TinyApp" / "Metric_TRAIN.json"
    uncertainty_path = tmp_path / "Uncertainties" / "TinyApp" / "Metric_TRAIN.json"

    assert predicted_path.exists()
    assert inference_stack_path.exists()
    assert evaluation_path.exists()
    assert uncertainty_path.exists()

    predicted = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(str(predicted_path)))
    inference_stack = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(str(inference_stack_path)))

    np.testing.assert_allclose(predicted, expected_output, atol=3e-4)
    assert inference_stack.shape == (1, 16, 16, 2)
    np.testing.assert_allclose(inference_stack[0, :, :, 0], expected_output[0], atol=3e-4)
    np.testing.assert_allclose(inference_stack[0, :, :, 1], expected_output[0], atol=3e-4)

    assert _single_case_metric(evaluation_path, expected_metric_suffix="MAE") == pytest.approx(0.0, abs=3e-4)
    assert _single_case_metric(uncertainty_path, expected_metric_suffix="Uncertainty") == pytest.approx(0.0, abs=1e-6)


# Counters baked into the fixture checkpoints. They mimic a released model whose `epoch` already
# reaches the fine-tuning target: without the weights-only sanitize step, `range(epoch, epochs)` is
# empty and fine-tuning silently trains for zero steps.
_PRETRAINED_EPOCH = 10
_PRETRAINED_IT = 125134


def _write_local_finetune_app(app_dir: Path) -> None:
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Tiny Synth",
                "description": "Tiny local synthesis app for fine-tuning integration testing",
                "short_description": "Tiny synth",
                "tta": 0,
                "mc_dropout": 0,
                "models": ["tiny_0.pt", "tiny_1.pt"],
                "inputs": {"Volume_0": {"display_name": "MR", "volume_type": "VOLUME", "required": True}},
                "outputs": {"sCT": {"display_name": "sCT", "volume_type": "VOLUME", "required": True}},
            }
        ),
        encoding="utf-8",
    )
    # Reuse the core training-config template (TinySynthNet + MR/CT/MASK groups). The dataset is linked
    # into ./Dataset by fine_tune, and train_name is overridden per-model, so only the placeholders
    # need substituting here.
    config = (WORKFLOW_ASSETS_DIR / "Config.yml").read_text(encoding="utf-8")
    config = config.replace("__DATASET_DIR__", "./Dataset").replace("__TRAIN_NAME__", "FT")
    (app_dir / "Config.yml").write_text(config, encoding="utf-8")
    (app_dir / "TinySynth.py").write_text(
        (WORKFLOW_ASSETS_DIR / "TinySynth.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    spec = importlib.util.spec_from_file_location("TinySynth", app_dir / "TinySynth.py")
    if spec is None or spec.loader is None:
        raise AssertionError("Failed to load TinySynth test module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.TinySynthNet()
    checkpoint = {
        "epoch": _PRETRAINED_EPOCH,
        "it": _PRETRAINED_IT,
        "loss": 0.0,
        "Model": model.state_dict(),
    }
    torch.save(checkpoint, app_dir / "tiny_0.pt")
    torch.save(checkpoint, app_dir / "tiny_1.pt")


def _build_finetune_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(4):
        case_dir = dataset_dir / f"CASE_{idx:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        zz, yy, xx = np.meshgrid(
            np.linspace(-0.2, 0.2, 3, dtype=np.float32),
            np.linspace(-1.0, 1.0, 16, dtype=np.float32),
            np.linspace(-1.0, 1.0, 16, dtype=np.float32),
            indexing="ij",
        )
        mr = np.clip(0.45 * yy + 0.35 * xx + zz + (idx - 1.5) * 0.05, -0.9, 0.9).astype(np.float32)
        ct = np.tanh(1.25 * mr - 0.15).astype(np.float32)
        mask = np.ones_like(mr, dtype=np.uint8)
        _write_image(case_dir / "MR.mha", mr, SimpleITK.sitkFloat32)
        _write_image(case_dir / "CT.mha", ct, SimpleITK.sitkFloat32)
        _write_image(case_dir / "MASK.mha", mask, SimpleITK.sitkUInt8)


def _run_finetune(
    app_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    *,
    models: list[str] | None = None,
    epochs: int = 2,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-c",
        (
            f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); "
            f"sys.path.insert(0, {str(KONFAI_APPS_ROOT)!r}); "
            "from konfai_apps.cli import main_apps; main_apps()"
        ),
        "fine-tune",
        str(app_dir),
        "FT",
        "-d",
        str(dataset_dir),
        "--epochs",
        str(epochs),
        "--cpu",
        "1",
        "-o",
        str(output_dir),
    ]
    if models:
        cmd += ["--models", *models]
    return run(cmd)


def _assert_finetuned_from_checkpoint(checkpoint_path: Path) -> None:
    """A fine-tuned checkpoint must start fresh (counters reset) yet have actually trained."""
    assert checkpoint_path.exists(), checkpoint_path
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert state["epoch"] < _PRETRAINED_EPOCH, f"epoch not reset: {state['epoch']}"
    assert 0 < state["it"] < _PRETRAINED_IT, f"training did not run from a fresh counter: it={state['it']}"


def test_finetune_single_model_resumes_from_checkpoint_and_trains(tmp_path: Path) -> None:
    app_dir = tmp_path / "TinySynthApp"
    dataset_dir = tmp_path / "Dataset"
    output_dir = tmp_path / "Output"
    _write_local_finetune_app(app_dir)
    _build_finetune_dataset(dataset_dir)

    result = _run_finetune(app_dir, dataset_dir, output_dir, models=["tiny_0"])
    assert result.returncode == 0, f"CMD failed.\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    meta = _read_json(output_dir / "app.json")
    assert meta["display_name"] == "FT"
    assert meta["models"] == ["tiny_0.pt"]

    _assert_finetuned_from_checkpoint(output_dir / "tiny_0.pt")
    # Only the selected checkpoint is packaged, and the bundle is clean (no training artifacts).
    assert not (output_dir / "tiny_1.pt").exists()
    for artifact in ("Checkpoints", "Statistics", "Dataset"):
        assert not (output_dir / artifact).exists(), artifact


def test_finetune_defaults_to_first_checkpoint(tmp_path: Path) -> None:
    app_dir = tmp_path / "TinySynthApp"
    dataset_dir = tmp_path / "Dataset"
    output_dir = tmp_path / "Output"
    _write_local_finetune_app(app_dir)
    _build_finetune_dataset(dataset_dir)

    result = _run_finetune(app_dir, dataset_dir, output_dir)  # no --models
    assert result.returncode == 0, f"CMD failed.\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    meta = _read_json(output_dir / "app.json")
    assert meta["models"] == ["tiny_0.pt"]
    assert (output_dir / "tiny_0.pt").exists()
    assert not (output_dir / "tiny_1.pt").exists()


def test_finetune_multiple_models_produces_one_checkpoint_each(tmp_path: Path) -> None:
    app_dir = tmp_path / "TinySynthApp"
    dataset_dir = tmp_path / "Dataset"
    output_dir = tmp_path / "Output"
    _write_local_finetune_app(app_dir)
    _build_finetune_dataset(dataset_dir)

    result = _run_finetune(app_dir, dataset_dir, output_dir, models=["tiny_0", "tiny_1"])
    assert result.returncode == 0, f"CMD failed.\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    meta = _read_json(output_dir / "app.json")
    assert meta["models"] == ["tiny_0.pt", "tiny_1.pt"]
    _assert_finetuned_from_checkpoint(output_dir / "tiny_0.pt")
    _assert_finetuned_from_checkpoint(output_dir / "tiny_1.pt")

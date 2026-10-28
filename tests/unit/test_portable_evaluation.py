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

"""The EVALUATION half of the portable contract: what an ``Evaluation.yml`` compiles to, and what is refused."""

from pathlib import Path

import pytest
from konfai.utils.errors import ConfigError, EvaluatorError


def _evaluation(tmp_path: Path, metrics: str, groups: str = "", validation: str = "None") -> Path:
    """An Evaluation.yml over two small MetaImage cases, each ``Volume_0``, ``Reference_0``, ``Mask_0`` and
    ``Unread_0``."""
    import numpy as np
    import SimpleITK as sitk

    for case in ("P000", "P001"):
        (tmp_path / "Dataset" / case).mkdir(parents=True)
        for name, array in [
            ("Volume_0", np.arange(720, dtype=np.float32).reshape(8, 9, 10) % 7),
            ("Reference_0", np.arange(720, dtype=np.float32).reshape(8, 9, 10) % 5),
            ("Mask_0", np.ones((8, 9, 10), np.uint8)),
            ("Unread_0", np.zeros((8, 9, 10), np.uint8)),
        ]:
            sitk.WriteImage(sitk.GetImageFromArray(array), str(tmp_path / "Dataset" / case / f"{name}.mha"))
    config = tmp_path / "Evaluation.yml"
    config.write_text(
        "Evaluator:\n  train_name: T\n  metrics:\n    Output:\n      targets_criterions:\n"
        f"        Reference;Mask:\n          criterions_loader:\n{metrics}"
        "  Dataset:\n    groups_src:\n"
        "      Volume_0:\n        groups_dest:\n          Output:\n            transforms:\n"
        "              Statistics: {}\n              TensorCast: {dtype: float32}\n"
        "      Reference_0:\n        groups_dest:\n          Reference:\n            transforms: None\n"
        "      Mask_0:\n        groups_dest:\n          Mask:\n            transforms: None\n"
        f"{groups}"
        f"    dataset_filenames: ['{tmp_path / 'Dataset'}:mha']\n    validation: {validation}\n"
    )
    return config


def _compile(tmp_path: Path, metrics: str, **more: str) -> dict:
    from konfai.portable_evaluation import compile_evaluation

    config = _evaluation(tmp_path, metrics, **more)
    return compile_evaluation(config, tmp_path / "Evaluations", tmp_path / "program" / "evaluation.json")


def test_an_evaluation_compiles_its_groups_criteria_and_maps_under_konfais_keys(tmp_path: Path) -> None:
    from konfai.portable_schema import program_schema

    program = _compile(
        tmp_path,
        "            DiceSaveMap: {labels: None, dataset: './Evaluations/T/Output:mha', group: Seg_map}\n"
        "            Dice: {labels: [1, 2]}\n"
        "            MAE: {reduction: sum}\n"
        "            PSNR: {dynamic_range: 1000}\n"
        "            SSIM: {dynamic_range: None}\n",
        groups="      Unread_0:\n        groups_dest:\n          Unread:\n            transforms: None\n",
    )
    # Only the groups a criterion reads, each with its chain: Statistics where it records, then the cast.
    assert [(g["group_src"], g["group_dest"]) for g in program["groups"]] == [
        ("Volume_0", "Output"),
        ("Reference_0", "Reference"),
        ("Mask_0", "Mask"),
    ]
    assert program["groups"][0]["steps"] == [{"op": "statistics"}, {"op": "cast", "dtype": "float32"}]
    (metric,) = program["metrics"]
    assert (metric["output"], metric["targets"]) == ("Output", ["Reference", "Mask"])
    # A repeated class records under `Name#2`, as konfai's `criterion_keys` names it.
    assert [c["key"] for c in metric["criteria"]] == [
        "Output:Reference;Mask:Dice",
        "Output:Reference;Mask:Dice#2",
        "Output:Reference;Mask:MAE",
        "Output:Reference;Mask:PSNR",
        "Output:Reference;Mask:SSIM",
    ]
    dice, labelled, mae, psnr, ssim = metric["criteria"]
    assert dice["map"]["root"] == "Evaluations/T/Output" and dice["map"]["group"] == "Seg_map"
    assert (dice["maximize"], labelled["labels"], "map" in labelled) == (True, [1, 2], False)
    assert (mae["reduction"], mae["maximize"]) == ("sum", False)
    assert (psnr["dynamic_range"], ssim["dynamic_range"]) == (1000.0, 4095.0)
    assert (program["evaluations"], program["cases"]) == (str(tmp_path / "Evaluations" / "T"), ["P000", "P001"])
    import jsonschema

    jsonschema.validate(program, program_schema())


def test_a_criterion_with_no_portable_form_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(EvaluatorError, match="Criterion 'MSE' has no portable form"):
        _compile(tmp_path, "            MSE: {reduction: mean}\n")


def test_an_evaluation_split_into_train_and_validation_is_refused(tmp_path: Path) -> None:
    with pytest.raises(EvaluatorError, match="scores one split"):
        _compile(tmp_path, "            Dice: {labels: None}\n", validation="[P001]")


def test_a_nested_app_needs_the_exporter_that_resolves_it(tmp_path: Path) -> None:
    from konfai.portable_evaluation import _group_steps

    with pytest.raises(ConfigError, match="nested app"):
        _group_steps("Output_seg", {"KonfAIInference": {"repo_id": "a", "model_name": "b"}}, tmp_path, {}, None)
    exported = []

    def exporter(group: str, params: dict, out: Path) -> dict:
        exported.append(group)
        return {"id": group, "manifest": {"model": f"{group}.onnx"}}

    nested: dict = {}
    for group in ("Output_seg", "Reference_seg"):
        steps = _group_steps(
            group, {"KonfAIInference": {"repo_id": "a", "model_name": "b"}}, tmp_path, nested, exporter
        )
        assert steps == [{"model": "Output_seg", "manifest": {"model": "Output_seg.onnx"}}]
    # One nested app, however many groups run it, is exported once.
    assert exported == ["Output_seg"]


def test_impact_reg_has_a_portable_form_on_a_whole_volume_only(tmp_path: Path) -> None:
    from types import SimpleNamespace

    import torch
    from konfai.portable_evaluation import _impact_reg

    feature = SimpleNamespace(shape=[64, 64, 64], dim=3, in_channels=1, denormalize=False, weights=[0, 1])
    tiled = SimpleNamespace(model=feature, loss=torch.nn.MSELoss(), pca=0)
    with pytest.raises(EvaluatorError, match="IMPACTReg with a distance, a PCA, a tile"):
        _impact_reg(tiled, "Output", tmp_path)

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


def _evaluation(
    tmp_path: Path,
    metrics: str,
    groups: str = "",
    validation: str = "None",
    shape: tuple[int, ...] = (8, 9, 10),
    chain: str = "              Statistics: {}\n              TensorCast: {dtype: float32}\n",
) -> Path:
    """An Evaluation.yml over two small MetaImage cases of ``shape``, each ``Volume_0``, ``Reference_0``, ``Mask_0``
    and ``Unread_0``, the output prepared by ``chain``."""
    import numpy as np

    sitk = pytest.importorskip("SimpleITK")

    count = int(np.prod(shape))
    for case in ("P000", "P001"):
        (tmp_path / "Dataset" / case).mkdir(parents=True)
        for name, array in [
            ("Volume_0", np.arange(count, dtype=np.float32).reshape(shape) % 7),
            ("Reference_0", np.arange(count, dtype=np.float32).reshape(shape) % 5),
            ("Mask_0", np.ones(shape, np.uint8)),
            ("Unread_0", np.zeros(shape, np.uint8)),
        ]:
            sitk.WriteImage(sitk.GetImageFromArray(array), str(tmp_path / "Dataset" / case / f"{name}.mha"))
    config = tmp_path / "Evaluation.yml"
    config.write_text(
        "Evaluator:\n  train_name: T\n  metrics:\n    Output:\n      targets_criterions:\n"
        f"        Reference;Mask:\n          criterions_loader:\n{metrics}"
        "  Dataset:\n    groups_src:\n"
        "      Volume_0:\n        groups_dest:\n          Output:\n            transforms:\n"
        f"{chain}"
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


def test_a_class_of_the_app_s_is_probed_on_what_its_chain_hands_it_of_the_case(tmp_path: Path, monkeypatch) -> None:
    # A OneHot hands the class two channels, of a case of two axes: it is probed on them, not on one channel of three.
    from konfai.portable_evaluation import compile_evaluation

    (tmp_path / "PairOfPlanes.py").write_text(
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Background(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        if tuple(tensor.shape[:1]) != (2,) or tensor.dim() != 3:\n"
        "            raise ValueError(f'two channels of a plane, not {list(tensor.shape)}')\n"
        "        return tensor[:1]\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    chain = "              OneHot: {num_classes: 2}\n              PairOfPlanes:Background: {}\n"
    config = _evaluation(tmp_path, "            Dice: {labels: [1]}\n", shape=(9, 10), chain=chain)
    extension = tmp_path / "own.wasm"
    extension.write_bytes(b"\0asm")
    program = compile_evaluation(
        config, tmp_path / "Evaluations", tmp_path / "program" / "evaluation.json", extension=extension
    )
    assert [step["op"] for step in program["groups"][0]["steps"]] == ["one_hot", "extension"]


@pytest.mark.parametrize("per_channel", [False, True])
def test_a_class_of_the_app_s_past_a_nested_model_is_probed_on_what_the_model_hands_on(
    tmp_path: Path, monkeypatch, per_channel: bool
) -> None:
    # The nested app's head gives three channels, its own chain an argmax of them: one channel, or one for each of the
    # two the group holds when it runs on each.
    from konfai.portable_evaluation import _group_steps

    (tmp_path / "AfterNested.py").write_text(
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Labels(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __init__(self, channels: int = 1) -> None:\n"
        "        super().__init__()\n"
        "        self.channels = channels\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        if tensor.shape[0] != self.channels:\n"
        "            raise ValueError(f'{self.channels} channels, not {list(tensor.shape)}')\n"
        "        return tensor\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    manifest = {"model": "Seg.onnx", "output": {"channels": 3}, "postprocessing": [{"op": "argmax"}]}
    inference = {"repo_id": "a", "model_name": "b", "per_channel": per_channel}
    chain = {"KonfAIInference": inference, "AfterNested:Labels": {"channels": 2 if per_channel else 1}}
    steps = _group_steps(
        "Seg", chain, tmp_path, {}, lambda group, _params, _out: {"id": group, "manifest": manifest}, True, (2, 3, 1)
    )
    assert steps[-1]["op"] == "extension"


@pytest.mark.parametrize(
    "before",
    [{"TensorCast": {"dtype": "int16"}}, {"KonfAIInference": {"repo_id": "a", "model_name": "b"}}],
    ids=["past-a-cast", "past-a-nested-model's-argmax"],
)
def test_a_class_of_the_app_s_in_an_evaluation_chain_is_probed_in_the_dtype_it_is_handed(
    tmp_path: Path, monkeypatch, before: dict
) -> None:
    # Past a cast to int16, or a nested app whose chain ends on an argmax, konfai hands the class integer labels,
    # which a bitwise test takes and a float32 volume does not.
    from konfai.portable_evaluation import _group_steps

    (tmp_path / "EvaluatedLabels.py").write_text(
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Odd(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor & 1\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    manifest = {"model": "Seg.onnx", "output": {"channels": 3}, "postprocessing": [{"op": "argmax"}]}
    steps = _group_steps(
        "Seg",
        {**before, "EvaluatedLabels:Odd": {}},
        tmp_path,
        {},
        lambda group, _params, _out: {"id": group, "manifest": manifest},
        True,
        (1, 2, 1),
    )
    assert steps[-1]["op"] == "extension"


def test_a_class_of_the_app_s_run_again_to_count_its_channels_is_run_in_the_dtype_it_is_handed(
    tmp_path: Path, monkeypatch
) -> None:
    # Probing the last class runs the ones before it again: the bitwise one past an Argmax takes int64 labels there too.
    from konfai.portable_evaluation import _group_steps

    (tmp_path / "EvaluatedBits.py").write_text(
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Odd(Transform):\n"
        "    locality = LocalityKind.POINTWISE\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor & 1\n\n"
        "class Whole(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    chain = {"Argmax": {}, "EvaluatedBits:Odd": {}, "EvaluatedBits:Whole": {}}
    steps = _group_steps("Seg", chain, tmp_path, {}, None, True, (2, 2, 1))
    assert [step["op"] for step in steps] == ["argmax", "extension", "extension"]


@pytest.mark.parametrize(
    ("fold", "before", "members", "channels"),
    [
        ("StandardDeviation", {}, 3, 2),
        ("Variance", {}, 1, 6),
        ("SegmentationDisagreement", {}, 3, 1),
        ("Variance", {"OneHot": {"num_classes": 3}}, None, 1),
    ],
    ids=["three-members-of-two", "one-member-of-six", "a-disagreement", "one-member-a-channel-of-a-one-hot"],
)
def test_a_class_of_the_app_s_past_a_fold_of_the_members_is_probed_on_what_the_fold_leaves(
    tmp_path: Path, monkeypatch, fold: str, before: dict, members: int | None, channels: int
) -> None:
    # A stack of members cast to int16: a spread leaves one member's, a disagreement one channel, each in float32.
    # Where no header counts them, each channel the fold is handed is a member.
    from konfai.portable_evaluation import _group_steps

    (tmp_path / "AfterMembers.py").write_text(
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Count(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __init__(self, channels: int = 1) -> None:\n"
        "        super().__init__()\n"
        "        self.channels = channels\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        if tensor.shape[0] != self.channels or not tensor.is_floating_point():\n"
        "            raise ValueError(f'{self.channels} float channels, not {tensor.dtype} {list(tensor.shape)}')\n"
        "        return tensor\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    chain = {**before, "TensorCast": {"dtype": "int16"}, fold: {}, "AfterMembers:Count": {"channels": channels}}
    stored = 1 if members is None else 2 * members if members > 1 else 6
    steps = _group_steps("Spread", chain, tmp_path, {}, None, True, (stored, 3, members))
    assert steps[-1]["op"] == "extension"


@pytest.mark.parametrize(("members", "count"), [(2, 2), (None, None)])
def test_the_first_case_s_stacked_members_are_read_off_its_header(
    tmp_path: Path, image_attributes, members, count
) -> None:
    # As konfai counts them: the header's count, else none, one member a channel of what the fold is handed.
    from types import SimpleNamespace

    import numpy as np
    from konfai.portable_evaluation import _first_case
    from konfai.utils.dataset import Dataset

    attribute = image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    if members is not None:
        attribute["konfai_members"] = members
    dataset = Dataset(tmp_path / "Stacks", "mha")
    dataset.write("Stack", "P0", np.zeros((4, 3, 5, 6), np.float32), attribute)
    assert _first_case(SimpleNamespace(case_names=["P0"], datasets={"stacks": dataset}), "Stack") == (4, 3, count)


@pytest.mark.parametrize(
    "manifest",
    [
        {"output": {"channels": 3}, "blended": {"channels": 2, "dtype": "uint8", "params": {}}},
        {
            "output": {"channels": 1},
            "postprocessing": [
                {"op": "cast", "dtype": "uint8"},
                {"op": "extension", "class": "NestedChain:Repeat", "params": {}, "whole": True},
            ],
        },
    ],
    ids=["what-its-fold-leaves", "past-its-own-chain"],
)
def test_a_class_of_the_app_s_past_a_nested_model_is_probed_in_the_dtype_the_model_hands_on(
    tmp_path: Path, monkeypatch, manifest: dict
) -> None:
    # The nested app's Fold leaves two uint8 channels; or its own chain casts its one to uint8, which a class of its
    # own repeats into two, as it does integers only: two uint8 channels either way.
    from konfai.portable_evaluation import _group_steps

    (tmp_path / "NestedChain.py").write_text(
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Repeat(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor if tensor.is_floating_point() else tensor[:1].repeat(2, 1, 1, 1)\n\n"
        "class Bytes(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        if str(tensor.dtype) != 'torch.uint8' or tensor.shape[0] != 2:\n"
        "            raise ValueError(f'two uint8 channels, not {tensor.dtype} {list(tensor.shape)}')\n"
        "        return tensor\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    nested = {"model": "Seg.onnx", **manifest}
    steps = _group_steps(
        "Seg",
        {"KonfAIInference": {"repo_id": "a", "model_name": "b"}, "NestedChain:Bytes": {}},
        tmp_path,
        {},
        lambda group, _params, _out: {"id": group, "manifest": nested},
        True,
        (1, 3, 1),
    )
    assert steps[-1]["op"] == "extension"


def test_a_criterion_with_no_portable_form_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(EvaluatorError, match="Criterion 'MSE' has no portable form"):
        _compile(tmp_path, "            MSE: {reduction: mean}\n")


def test_a_criterion_of_the_apps_own_is_measured_by_its_extension(tmp_path: Path, monkeypatch) -> None:
    # Named by classpath: with the app's extension the program names the criterion by it, with the keys konfai binds
    # it with; without it, the criterion has no portable form.
    import jsonschema
    from konfai.portable_evaluation import compile_evaluation
    from konfai.portable_schema import program_schema

    (tmp_path / "OwnCriteria.py").write_text(
        "import torch\n"
        "from konfai.metric.measure import Criterion\n\n\n"
        "class Bias(Criterion):\n"
        "    def __init__(self, scale: float = 1.0) -> None:\n"
        "        super().__init__()\n"
        "        self.scale = scale\n\n"
        "    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:\n"
        "        return self.scale * (output - targets[0]).mean()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config = _evaluation(tmp_path, "            OwnCriteria:Bias: {scale: 2.0}\n")
    out = tmp_path / "program" / "evaluation.json"
    with pytest.raises(EvaluatorError, match="Criterion 'Bias' has no portable form"):
        compile_evaluation(config, tmp_path / "Evaluations", out)
    extension = tmp_path / "own.wasm"
    extension.write_bytes(b"\0asm")
    program = compile_evaluation(config, tmp_path / "Evaluations", out, extension=extension)

    assert program["extension"] == "own.wasm" and (out.parent / "own.wasm").is_file()
    (criterion,) = program["metrics"][0]["criteria"]
    assert (criterion["metric"], criterion["class"]) == ("extension", "OwnCriteria:Bias")
    assert criterion["params"]["scale"] == 2.0
    jsonschema.validate(program, program_schema())


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


def test_a_group_running_two_nested_apps_exports_each_under_its_own_name(tmp_path: Path) -> None:
    from konfai.portable_evaluation import _group_steps

    def exporter(group: str, params: dict, out: Path) -> dict:
        return {"id": group, "manifest": {"model": params["model_name"]}}

    chain = {
        "KonfAIInference": {"repo_id": "a", "model_name": "b"},
        "KonfAIInference#2": {"repo_id": "a", "model_name": "c"},
    }
    steps = _group_steps("Output_seg", chain, tmp_path, {}, exporter)
    # Named after the group, the second app would overwrite the first's `Output_seg.onnx`.
    assert [(step["model"], step["manifest"]["model"]) for step in steps] == [
        ("Output_seg", "b"),
        ("Output_seg_2", "c"),
    ]


def test_a_crop_after_a_stage_moving_the_values_is_refused_in_an_evaluation_chain(tmp_path: Path) -> None:
    from konfai.portable_evaluation import _group_steps

    # konfai's Crop finds its box in the group as stored, the runtime in the volume it is handed: one box only
    # while nothing before it moved the grid or the values (`compile_chain` refuses the same).
    steps = _group_steps(
        "Output", {"Statistics": {}, "TensorCast": {"dtype": "float32"}, "Crop": {}}, tmp_path, {}, None
    )
    assert [step["op"] for step in steps] == ["statistics", "cast", "crop"]
    with pytest.raises(EvaluatorError, match=r"(?s)'Crop' follows 'Clip'.*konfai EVALUATION"):
        _group_steps("Output", {"Clip": {"min_value": 0.0, "max_value": 400.0}, "Crop": {}}, tmp_path, {}, None)
    exporter = lambda group, _params, _out: {"id": group, "manifest": {}}  # noqa: E731
    with pytest.raises(EvaluatorError, match="'Crop' follows 'KonfAIInference'"):
        _group_steps("Output", {"KonfAIInference": {"repo_id": "a"}, "Crop": {}}, tmp_path, {}, exporter)


def test_sam_perceptual_names_its_graph_after_the_network_it_loads(tmp_path: Path, monkeypatch) -> None:
    import konfai.portable_evaluation as portable_evaluation
    from konfai.metric.measure import impact

    monkeypatch.setattr(impact, "_download_model", lambda _criterion, repo_id, filename: f"/cache/{repo_id}/{filename}")
    written: list[Path] = []
    monkeypatch.setattr(portable_evaluation, "export_sam_perceptual", lambda _model, _tile, path: written.append(path))
    steps = [
        portable_evaluation._sam_perceptual(impact.SAM_Perceptual(train=train), "Output", tmp_path)
        for train in (False, True)
    ]
    # The metric-tuned network and the raw extractor are two networks under one file name, `SAM2.1_Small.pt`.
    assert steps[0]["model"] != steps[1]["model"] and len(set(written)) == 2
    assert all(step["model"] == path.stem and path.suffix == ".onnx" for step, path in zip(steps, written, strict=True))


def test_a_remote_map_root_is_refused_as_konfai_refuses_it() -> None:
    # POSIX would read s3://bucket/maps as the local path s3:/bucket/maps.
    from konfai.portable import _portable_root
    from konfai.utils.errors import DatasetManagerError

    with pytest.raises(DatasetManagerError, match="remote root"):
        _portable_root("s3://bucket/maps")


def test_impact_reg_has_a_portable_form_on_a_whole_volume_only(tmp_path: Path) -> None:
    from types import SimpleNamespace

    import torch
    from konfai.portable_evaluation import _impact_reg

    feature = SimpleNamespace(shape=[64, 64, 64], dim=3, in_channels=1, denormalize=False, weights=[0, 1])
    tiled = SimpleNamespace(model=feature, loss=torch.nn.MSELoss(), pca=0)
    with pytest.raises(EvaluatorError, match="IMPACTReg with a distance, a PCA, a tile"):
        _impact_reg(tiled, "Output", tmp_path)


def test_an_uncertainty_compiles_the_member_folds_the_mean_and_a_nested_app_per_channel(tmp_path: Path) -> None:
    # An app's Uncertainty.yml, as konfai-apps runs it on a two-member stack: no target, a fold over the members.
    import numpy as np
    from konfai.portable_evaluation import _group_steps, compile_uncertainty
    from konfai.portable_schema import program_schema

    sitk = pytest.importorskip("SimpleITK")

    (tmp_path / "Dataset" / "P000").mkdir(parents=True)
    stack = sitk.GetImageFromArray(np.zeros((4, 5, 6, 2), np.float32), isVector=True)
    sitk.WriteImage(stack, str(tmp_path / "Dataset" / "P000" / "Volume_0.mha"))
    config = tmp_path / "Uncertainty.yml"
    config.write_text(
        "Evaluator:\n  train_name: T\n  metrics:\n    Uncertainty:\n      targets_criterions:\n"
        "        None:\n          criterions_loader:\n            Mean: {name: Uncertainty}\n"
        "  Dataset:\n    groups_src:\n      Volume_0:\n        groups_dest:\n          Uncertainty:\n"
        "            transforms:\n              Variance: {}\n              Percentage: {baseline: 1822.49}\n"
        f"              Save: {{dataset: '{tmp_path}/Uncertainties/T/Output:mha', group: None}}\n"
        f"    dataset_filenames: ['{tmp_path / 'Dataset'}:mha']\n    validation: None\n"
    )
    program = compile_uncertainty(config, tmp_path / "Uncertainties", tmp_path / "uncertainty.json")
    (group,) = program["groups"]
    assert [step["op"] for step in group["steps"]] == ["variance", "percentage", "save"]
    assert group["steps"][2]["destination"]["group"] == "Uncertainty"
    assert program["metrics"] == [
        {
            "output": "Uncertainty",
            "targets": [],
            "criteria": [{"key": "Uncertainty:None:Uncertainty", "maximize": False, "metric": "mean"}],
        }
    ]
    import jsonschema

    jsonschema.validate(program, program_schema())

    def exporter(group: str, params: dict, out: Path) -> dict:
        return {"id": group, "manifest": {"model": f"{group}.onnx"}}

    steps = _group_steps(
        "Comformity",
        {"KonfAIInference": {"repo_id": "a", "model_name": "b", "per_channel": True}, "SegmentationDisagreement": {}},
        tmp_path,
        {},
        exporter,
    )
    assert steps == [
        {"model": "Comformity", "manifest": {"model": "Comformity.onnx"}, "per_channel": True},
        {"op": "segmentation_disagreement"},
    ]

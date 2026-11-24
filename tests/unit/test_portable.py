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

"""The portable-runtime contract: what compiles to a runtime op and what is refused by name.

A chain the Python engine and the portable runtime read differently is wrong data rather than an error, so
what has no portable form must be refused at compile time and never approximated.
"""

import inspect
import json
import os
import re
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from konfai.portable import (
    _CHAIN_OPS,
    OP_MAP,
    _literal_group,
    assemble_program,
    base_name,
    chain_items,
    derive_overlap,
    derive_reduction,
    export_checkpoints,
    models_fold_first,
    program_reduction,
    source_of,
    split_post,
    transform_manifest,
    tta_passes,
)
from konfai.portable import compile_chain as steps
from konfai.utils.errors import ConfigError, KonfAIError, TransformError


def test_a_chain_binds_in_order_in_either_spelling() -> None:
    mapping = {"Clip": {"min_value": 0.0}, "Resample": {"spacing": [1.5, 1.5, 1.5]}}
    listed = [{"Clip": {"min_value": 0.0}}, {"Resample": {"spacing": [1.5, 1.5, 1.5]}}]
    assert chain_items(mapping) == chain_items(listed)
    assert [name for name, _ in chain_items(listed)] == ["Clip", "Resample"]


def test_a_repeated_stage_keeps_its_class_past_the_occurrence_suffix() -> None:
    assert base_name("Clip#2") == "Clip"
    assert base_name("Clip") == "Clip"


@pytest.mark.parametrize(
    ("name", "stage"),
    [
        ("konfai.data.transform:Clip", "Clip"),
        ("konfai.data.transform.intensity:Clip", "Clip"),
        ("konfai:data:transform:Clip", "Clip"),
        ("konfai_apps.transforms:KonfAIInference", "KonfAIInference"),
        # A module that only shares a prefix with konfai's is the caller's own: its Clip is not konfai's.
        ("konfai_apps.transforms_custom:Clip", "konfai_apps.transforms_custom:Clip"),
        ("konfai.data.transformers:Clip", "konfai.data.transformers:Clip"),
    ],
)
def test_a_stage_is_konfais_only_through_a_konfai_module(name: str, stage: str) -> None:
    assert base_name(name) == stage


def test_a_chain_compiles_to_ops_and_a_destination() -> None:
    chain = {
        "Clip": {"min_value": 0.0, "max_value": 400.0},
        "Resample": {"spacing": [1.5, 1.5, 1.5], "interpolation": "linear"},
        "Write": {"dataset": "./Prepared:nii.gz"},
    }
    ops, write = steps(chain, "CT_prepared")
    assert [op["op"] for op in ops] == ["clip", "resample"]
    assert ops[0] == {"op": "clip", "min_value": 0.0, "max_value": 400.0}
    assert ops[1] == {"op": "resample", "spacing": [1.5, 1.5, 1.5], "interpolation": "linear"}
    # The destination is the Write's own, and the group falls back to the chain's.
    assert {key: write[key] for key in ("root", "format", "group")} == {
        "root": "./Prepared",
        "format": "nii.gz",
        "group": "CT_prepared",
    }


def test_a_write_with_no_format_keeps_konfais_default() -> None:
    _, write = steps({"Write": {"dataset": "./Out"}}, "CT")
    assert write["format"] == "mha"


def test_a_stage_with_no_portable_form_is_refused_by_name() -> None:
    # A class of the caller's own module: konfai imports it, the runtime has no op for it.
    chain = {"Custom:Rescale": {}, "Write": {"dataset": "./Out:mha"}}
    with pytest.raises(TransformError, match="Custom:Rescale"):
        steps(chain, "CT")


def test_a_histogram_match_reads_its_reference_group_beside_the_case() -> None:
    # konfai reads the reference as stored and hands it to SimpleITK, whatever its extent.
    chain, _write = steps({"HistogramMatching": {"reference_group": "REF"}, "Write": {"dataset": "./Out:mha"}}, "CT")
    assert chain == [{"op": "histogram_match", "reference_group": "REF"}]
    manifest, _folds, _per_copy = transform_manifest(
        _prediction({"HistogramMatching": {"reference_group": "REF"}}, {}), "Predictor"
    )
    assert manifest["reads"] == [{"group": "REF", "channels": 1, "preprocessing": []}]


@pytest.mark.parametrize(
    ("stage", "params", "why"),
    [("Squeeze", {"dim": 1}, "writes the entry back on it"), ("Flatten", {}, "ITK refuses past five")],
)
def test_a_stage_konfai_does_not_carry_to_its_writer_is_refused_with_what_konfai_does(
    stage: str, params: dict[str, object], why: str
) -> None:
    # Measured on konfai 1.9: a Squeeze writes the entry back on the grid it kept, and a Flatten reaches
    # the writer as one long axis, which ITK refuses. Neither has a portable form that is konfai's.
    with pytest.raises(TransformError, match=why):
        steps({stage: params, "Write": {"dataset": "./Out:mha"}}, "CT")


def test_compiling_works_on_a_private_copy_and_never_writes_the_callers_config(tmp_path: Path) -> None:
    # The builders write their defaults back: a read-only config, or one saved during a compile, stays the caller's.
    import os

    from konfai.portable import config_environment

    config = tmp_path / "Prediction.yml"
    config.write_text("Predictor: {}\n")
    config.chmod(0o444)
    previous = os.environ.get("KONFAI_config_file")
    with config_environment(config, root="Predictor", state="PREDICTION") as working:
        assert Path(os.environ["KONFAI_config_file"]) == working and working.parent == tmp_path and working != config
        working.write_text("Predictor: {resolved: true}\n")
        config.chmod(0o644)
        config.write_text("Predictor: {edited: true}\n")
    assert config.read_text() == "Predictor: {edited: true}\n"
    assert os.environ.get("KONFAI_config_file") == previous
    assert [path.name for path in tmp_path.iterdir()] == ["Prediction.yml"]


def test_a_missing_config_keeps_the_builders_refusal(tmp_path: Path) -> None:
    from konfai.portable import compile_transform

    with pytest.raises(ConfigError, match="does not exist"):
        compile_transform(tmp_path / "Missing.yml", tmp_path / "Transforms")


def test_a_private_copy_that_cannot_be_made_is_refused_and_the_config_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Falling back to the caller's file would let the builders rewrite it; a copy cut short is not left behind.
    import shutil

    from konfai.portable import config_environment

    def cut_short(source: Path, target: Path) -> None:
        Path(target).write_text("Predic")
        raise OSError(28, "No space left on device")

    config = tmp_path / "Prediction.yml"
    config.write_text("Predictor: {}\n")
    monkeypatch.setattr(shutil, "copyfile", cut_short)
    with pytest.raises(ConfigError, match="private copy"):
        with config_environment(config, root="Predictor", state="PREDICTION"):
            pass
    assert config.read_text() == "Predictor: {}\n"
    assert [path.name for path in tmp_path.iterdir()] == ["Prediction.yml"]


def test_a_private_copy_takes_no_file_it_did_not_create(tmp_path: Path) -> None:
    # A file already beside the config, and the copy an overlapping compile of the same config works on, are kept.
    import os

    from konfai.portable import config_environment

    config = tmp_path / "Prediction.yml"
    config.write_text("Predictor: {}\n")
    stranger = tmp_path / f".Prediction.{os.getpid()}.yml"
    stranger.write_text("not konfai's")
    with config_environment(config) as outer:
        with config_environment(config) as inner:
            assert inner not in (outer, stranger)
        assert outer.read_text() == "Predictor: {}\n"
    assert stranger.read_text() == "not konfai's"
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted([config.name, stranger.name])


def test_a_stage_that_alters_no_value_compiles_to_no_step() -> None:
    # konfai's Statistics records Image* keys on the case and hands its input back (alters_values False):
    # the values and the grid are the chain's without it, and the keys are the attribute sidecar the
    # runtime never writes.
    chain = [
        {"Statistics": None},
        {"Clip": {"min_value": 0.0, "max_value": 400.0}},
        {"Statistics": {}},
        {"konfai.data.transform:Statistics": {}},
        {"Write": {"dataset": "./Out:mha"}},
    ]
    assert steps(chain, "CT") == steps(
        {"Clip": {"min_value": 0.0, "max_value": 400.0}, "Write": {"dataset": "./Out:mha"}}, "CT"
    )


def test_a_statistics_subclass_replacing_call_must_restate_that_it_alters_no_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import konfai.portable as portable
    from konfai.data.transform import Statistics

    class Shifted(Statistics):
        def __call__(self, name, tensors, cache_attribute):
            return tensors + 1

    class Recorded(Shifted):
        alters_values = False

    class Plain(Statistics):
        pass

    classes = {cls.__name__: cls for cls in (Shifted, Recorded, Plain)}
    monkeypatch.setattr(portable, "_stage_class", classes.get)
    assert [portable.records_only(name) for name in classes] == [False, True, True]


def test_a_save_compiles_to_a_step_naming_the_destination_a_write_would() -> None:
    # Save also hands its input back, but konfai writes what reaches it and a later run reads the last
    # write already there in place of the stages before it, so the runtime needs it at its place.
    chain = [
        {"Clip": {"min_value": 0.0, "max_value": 400.0}},
        {"Save": {"dataset": "./Work:nii.gz", "group": "CT_clipped"}},
        {"konfai.data.transform:Save": {"dataset": "./Cache", "group": "None", "scale_factors": "None"}},
        {"Write": {"dataset": "./Out:mha"}},
    ]
    ops, write = steps(chain, "CT")
    assert ops == [
        {"op": "clip", "min_value": 0.0, "max_value": 400.0},
        {"op": "save", "destination": steps({"Write": {"dataset": "./Work:nii.gz", "group": "CT_clipped"}}, "CT")[1]},
        # A Save naming no group writes the chain's, in konfai's default format.
        {"op": "save", "destination": steps({"Write": {"dataset": "./Cache:mha"}}, "CT")[1]},
    ]
    assert write == steps({"Write": {"dataset": "./Out:mha"}}, "CT")[1]


@pytest.mark.parametrize(
    "chain",
    [
        {"Save": {"dataset": "./Work:omezarr", "scale_factors": [2]}, "Write": {"dataset": "./Out:omezarr"}},
        {"Write": {"dataset": "./Out:omezarr", "scale_factors": [4, 4]}},
    ],
)
def test_a_write_asking_for_a_pyramid_is_refused_by_name(chain: dict) -> None:
    # konfai writes the extra levels into the store; the runtime writes the full resolution only.
    with pytest.raises(TransformError, match="scale_factors"):
        steps(chain, "CT")


def test_a_chain_that_does_not_end_with_a_write_is_refused() -> None:
    with pytest.raises(TransformError, match="does not end with a Write"):
        steps({"Clip": {"min_value": 0.0}}, "CT")


def test_a_write_naming_no_dataset_is_refused() -> None:
    with pytest.raises(TransformError, match="names no dataset"):
        steps({"Write": {}}, "CT")


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"reference": "CASE_000"}, "reference"),
        ({"field": "DVF"}, "field"),
        ({"spacing": [1.5, 1.5, 1.5], "transforms": {"T": False}}, "transforms"),
        ({"spacing": [1.5, 1.5, 1.5], "align": "center"}, "center"),
        ({}, "neither"),
        ({"spacing": [1.5, 1.5, 1.5], "shape": [64, 64, 64]}, "spacing and shape"),
        ({"spacing": [1.5, 1.5, 1.5], "interpolation": "lanczos"}, "lanczos"),
    ],
)
def test_a_resample_the_runtime_cannot_place_is_refused(params: dict, expected: str) -> None:
    with pytest.raises(TransformError, match=expected):
        OP_MAP["Resample"](params)


def test_a_cubic_resample_compiles_to_its_op() -> None:
    cubic = {"spacing": [1.5, 1.5, 1.5], "interpolation": "cubic"}
    assert OP_MAP["Resample"](cubic) == {"op": "resample", **cubic}


def test_a_resample_carries_its_align_off_konfais_default_only() -> None:
    # konfai places a spacing or a shape grid by its extent unless told otherwise; a program placed by extent
    # compiles as it did before the runtime read an align (konfai-edge tests/transform.rs parity fixtures).
    by_spacing = {"spacing": [1.5, 0.7, 1.2], "interpolation": "linear"}
    assert OP_MAP["Resample"]({**by_spacing, "align": "origin"}) == {"op": "resample", **by_spacing, "align": "origin"}
    assert OP_MAP["Resample"]({**by_spacing, "align": "extent"}) == {"op": "resample", **by_spacing}
    assert OP_MAP["Resample"]({"shape": [64, 0, 32], "align": "origin"}) == {
        "op": "resample",
        "shape": [64, 0, 32],
        "align": "origin",
    }


def test_a_resample_by_shape_compiles_to_konfais_extent_in_array_order() -> None:
    # konfai reads a shape in array order and a component <= 0 as its axis kept, which the runtime reads
    # the same way (konfai-edge tests/transform.rs::a_resample_by_shape_keeps_an_axis_at_zero_or_below).
    from konfai.data.transform.resample import _DerivedGrid

    shape = [64, 0, -3]
    step = OP_MAP["Resample"]({"spacing": "None", "shape": shape, "interpolation": "linear"})
    assert step == {"op": "resample", "shape": [64, 0, 0], "interpolation": "linear"}
    assert tuple(step["shape"]) == _DerivedGrid(None, shape, "extent").shape


@pytest.mark.parametrize(
    ("before", "expected"),
    [
        ({"TensorCast": {"dtype": "uint8"}}, "nearest"),
        ({"TensorCast": {"dtype": "int16"}}, "linear"),
        ({"TensorCast": {"dtype": "uint8"}, "Standardize": {"mean": 0.0, "std": 1.0}}, "linear"),
    ],
)
def test_an_unstated_interpolation_is_konfais_default_for_the_dtype_that_reaches_it(
    before: dict, expected: str
) -> None:
    ops, _ = steps({**before, "Resample": {"spacing": [1.5, 1.5, 1.5]}, "Write": {"dataset": "./Out:mha"}}, "CT")
    (resample,) = [op for op in ops if op["op"] == "resample"]
    assert resample["interpolation"] == expected


def test_a_linear_read_of_an_integer_payload_compiles_to_the_read_alone() -> None:
    # konfai blends in float32 and casts the blend back to the payload's dtype, the stored one or the one
    # a cast set. The runtime carries that dtype and truncates the read itself
    # (konfai-edge tests/transform.rs::a_linear_read_of_an_integer_stage_is_truncated_to_its_dtype).
    resample = {"spacing": [1.5, 1.5, 1.5], "interpolation": "linear"}
    ops, _ = steps({"TensorCast": {"dtype": "int16"}, "Resample": resample, "Write": {"dataset": "./Out:mha"}}, "CT")
    assert ops == [{"op": "cast", "dtype": "int16"}, {"op": "resample", **resample}]
    ops, _ = steps({"Resample": resample, "Write": {"dataset": "./Out:mha"}}, "CT")
    assert ops == [{"op": "resample", **resample}]


@pytest.mark.parametrize(("alias", "name"), [("float", "float32"), ("long", "int64"), ("uint8", "uint8")])
def test_a_cast_names_the_dtype_torch_resolves_its_alias_to(alias: str, name: str) -> None:
    assert OP_MAP["TensorCast"]({"dtype": alias}) == {"op": "cast", "dtype": name}


def test_an_unstated_interpolation_on_the_stored_dtype_is_left_to_the_runtime() -> None:
    # konfai picks one from the dtype the volume is stored in, which the runtime's reader records
    # (konfai-edge tests/transform.rs::a_resample_naming_no_interpolation_reads_with_the_default_of_its_dtype).
    chain = {"Clip": {"min_value": 0.0}, "Resample": {"spacing": [1.5, 1.5, 1.5]}, "Write": {"dataset": "./Out:mha"}}
    ops, _ = steps(chain, "CT")
    assert ops[-1] == {"op": "resample", "spacing": [1.5, 1.5, 1.5]}


def test_a_stated_interpolation_wins_over_the_dtype() -> None:
    chain = {
        "TensorCast": {"dtype": "float32"},
        "Resample": {"spacing": [1.5, 1.5, 1.5], "interpolation": "nearest"},
        "Write": {"dataset": "./Out:mha"},
    }
    ops, _ = steps(chain, "CT")
    assert ops[-1]["interpolation"] == "nearest"


def _undone_resample(outputs: dict, resample: dict | None = None) -> dict:
    """The post chain konfai's predictor runs for ``outputs`` over a float32 input it resampled."""
    resample = {"Resample": {"spacing": [1.5, 1.5, 1.5], "inverse": True, **(resample or {})}}
    return transform_manifest(_prediction(resample, outputs), "Predictor")[0]


def _input_group(transforms: dict | None, **more: object) -> dict:
    """A ``Dataset`` section as konfai resolves it: one input group ``g`` prepared by ``transforms``."""
    return {"groups_src": {"g": {"groups_dest": {"g": {"is_input": True, "transforms": transforms, **more}}}}}


_CLASS_UINT8 = {"Softmax": {"dim": 0}, "Argmax": {"dim": 0}, "TensorCast": {"dtype": "uint8"}}


@pytest.mark.parametrize(
    ("outputs", "resample", "expected"),
    [
        # The inverse runs on the model's floating output, before the final transforms decide a class.
        ({"final_transforms": {"Softmax": {"dim": 0}, "Argmax": {"dim": 0}}}, {}, "linear"),
        # Mean averages each copy's uint8 class into float32 before the inverse, which konfai reads linearly.
        ({"before_reduction_transforms": _CLASS_UINT8}, {}, "linear"),
        ({"before_reduction_transforms": _CLASS_UINT8, "reduction": "Median"}, {}, "linear"),
        # Vote keeps the copies' uint8, which konfai reads nearest.
        ({"before_reduction_transforms": _CLASS_UINT8, "reduction": "Vote"}, {}, "nearest"),
        ({}, {"interpolation": "nearest"}, "nearest"),
    ],
)
def test_an_inverse_resample_reads_the_dtype_of_what_it_undoes(outputs: dict, resample: dict, expected: str) -> None:
    manifest = _undone_resample(outputs, resample)
    (inverse,) = [step for step in manifest["postprocessing"] if step["op"] == "inverse"]
    assert inverse == {"op": "inverse", "of": "Resample", "interpolation": expected}


def test_an_inverse_of_a_class_decided_after_the_reduction_reads_it_nearest() -> None:
    # konfai's Argmax is torch's int64, a label dtype its Resample reads nearest.
    manifest = _undone_resample({"after_reduction_transforms": {"Argmax": {"dim": 0}}})
    assert manifest["postprocessing"] == [
        {"op": "argmax", "dim": 0},
        {"op": "inverse", "of": "Resample", "interpolation": "nearest"},
    ]


def test_an_inverse_of_a_resample_carries_the_fill_its_stage_states() -> None:
    # konfai undoes a resample with the stage's own fill (`_resample_between` reads `self.fill_value`);
    # the runtime fills the inverse from the step, so a stated fill travels to it.
    manifest = _undone_resample({}, {"fill": -1000.0})
    (inverse,) = [step for step in manifest["postprocessing"] if step["op"] == "inverse"]
    assert inverse == {"op": "inverse", "of": "Resample", "interpolation": "linear", "fill": -1000.0}


def test_an_edge_fill_reaches_the_resample_and_its_inverse() -> None:
    # `fill: edge` reads the source's edge pixel past it, both ways.
    manifest = _undone_resample({}, {"fill": "edge"})
    (resample,) = [step for step in manifest["preprocessing"] if step["op"] == "resample"]
    (inverse,) = [step for step in manifest["postprocessing"] if step["op"] == "inverse"]
    assert resample["fill"] == inverse["fill"] == "edge"


def test_a_linear_inverse_of_an_intensity_integer_reads_linear() -> None:
    # int16 is how scanners store intensities: konfai's Resample reads it linearly and casts the blend
    # back, which the runtime does from the dtype the cast leaves.
    manifest = _undone_resample({"after_reduction_transforms": {"TensorCast": {"dtype": "int16"}}})
    assert manifest["postprocessing"] == [
        {"op": "cast", "dtype": "int16"},
        {"op": "inverse", "of": "Resample", "interpolation": "linear"},
    ]


def test_a_reduction_whose_dtype_the_compiler_does_not_know_is_refused() -> None:
    with pytest.raises(ConfigError, match="Std"):
        _undone_resample({"reduction": "Std"})


def test_a_hoisted_inverse_resamples_the_way_its_step_names_and_keeps_its_casts() -> None:
    manifest = {
        "input": {"channels": 1},
        "output": {"channels": 2},
        "patch": {"size": [4, 4]},
        "preprocessing": [{"op": "resample", "spacing": [1.5, 1.5], "id": "r", "interpolation": "linear"}],
        "postprocessing": [
            {"op": "argmax", "dim": 0},
            {"op": "inverse", "of": "r", "interpolation": "linear"},
            {"op": "cast", "dtype": "int64"},
            {"op": "cast", "dtype": "float32"},
        ],
    }
    program = assemble_program([{"id": "a", "manifest": manifest}, {"id": "b", "manifest": manifest}], reduce="mean")
    ops = [step for step in program["steps"] if "op" in step]
    assert [step["op"] for step in ops] == ["mean", "argmax", "resample_linear", "cast", "cast"]
    # The float32 cast sets the dtype the output is written in.
    assert ops[-2:] == [
        {"op": "cast", "dtype": "int64", "in": ["t1"], "out": "t2"},
        {"op": "cast", "dtype": "float32", "in": ["t2"], "out": "output"},
    ]
    # A grid rule's inverse places the grid by that rule, which the hoisted resample does not carry.
    manifest["preprocessing"][0]["grid"] = "faces"
    with pytest.raises(ConfigError, match="faces rule"):
        assemble_program([{"id": "a", "manifest": manifest}, {"id": "b", "manifest": manifest}], reduce="mean")


def test_an_own_class_undone_past_the_reduction_is_replayed_with_every_inverse_there() -> None:
    # The app's own class fitted the grid before the resample: its inverse, and the resample's, land where they read,
    # which past the class is not the caller's grid, so both replay what the models' chain recorded.
    fit = {"op": "extension", "id": "fit", "class": "app:FitBox", "params": {}, "whole": True}
    manifest = {
        "input": {"channels": 1},
        "output": {"channels": 2},
        "patch": {"size": [4, 4]},
        "preprocessing": [fit, {"op": "resample", "spacing": [1.5, 1.5], "id": "r", "interpolation": "linear"}],
        "postprocessing": [
            {"op": "inverse", "of": "r", "interpolation": "linear"},
            {"op": "inverse", "of": "fit"},
            {"op": "argmax", "dim": 0},
        ],
    }
    program = assemble_program([{"id": "a", "manifest": manifest}, {"id": "b", "manifest": manifest}], reduce="mean")
    ops = [step for step in program["steps"] if "op" in step]
    assert [step["op"] for step in ops] == ["mean", "inverse", "inverse", "argmax"]
    assert {key: ops[1][key] for key in ("of", "interpolation")} == {"of": "r", "interpolation": "linear"}
    assert ops[2]["of"] == "fit"


def test_a_padding_the_runtime_cannot_fill_is_refused() -> None:
    with pytest.raises(TransformError, match="reflect"):
        OP_MAP["Padding"]({"padding": [1, 1], "mode": "reflect"})
    assert OP_MAP["Padding"]({"padding": [1, 1], "mode": "constant:-3.5"})["mode"] == "constant:-3.5"


def test_canonical_compiles_to_an_op_that_carries_its_grid() -> None:
    # A canonical reorientation moves the voxels and the header together: the op is portable only because the
    # runtime's grid travels with the values, as konfai's own Canonical writes it.
    ops, _ = steps({"Canonical": {}, "Write": {"dataset": "./Out:mha"}}, "CT")
    assert ops == [{"op": "canonical"}]


@pytest.mark.parametrize("dims", [None, "2", "0|2", "2|0|1"])
def test_flip_and_permute_compile_to_the_spatial_axes_konfai_reads(dims: str | None) -> None:
    from konfai.data.transform import Flip, Permute

    params = {} if dims is None else {"dims": dims}
    ops, _ = steps({"Flip": params, "Write": {"dataset": "./Out:mha"}}, "CT")
    assert ops == [{"op": "flip", "axes": [d - 1 for d in Flip(**params).dims]}]
    if dims is None or len(dims.split("|")) == 3:
        ops, _ = steps({"Permute": params, "Write": {"dataset": "./Out:mha"}}, "CT")
        assert ops == [{"op": "permute", "axes": [d - 1 for d in Permute(**params).dims[1:]]}]


def test_flip_dims_konfai_cannot_read_are_refused_by_name() -> None:
    with pytest.raises(TransformError, match="'Flip' cannot read dims"):
        OP_MAP["Flip"]({"dims": "x|y"})


def test_select_label_and_flat_label_compile_to_one_relabel_and_one_hot_to_its_classes() -> None:
    chain = [
        {"SelectLabel": {"labels": ["(1,3)", "(3,1)"]}},
        {"FlatLabel": {"labels": [1, 2]}},
        {"FlatLabel": {"labels": None}},
        {"FlatLabel": {"labels": []}},
        {"OneHot": {"num_classes": 4, "inverse": False}},
        {"Write": {"dataset": "./Out:mha"}},
    ]
    ops, _ = steps(chain, "CT")
    assert ops == [
        {"op": "relabel", "pairs": [[1, 3], [3, 1]]},
        {"op": "relabel", "pairs": [[1, 1], [2, 1]]},
        # konfai's FlatLabel listing no label marks every voxel above 0.
        {"op": "relabel", "positive": 1},
        {"op": "relabel", "positive": 1},
        {"op": "one_hot", "num_classes": 4},
    ]


def test_select_labels_konfai_cannot_read_are_refused_by_name() -> None:
    with pytest.raises(TransformError, match="'SelectLabel' cannot read labels"):
        OP_MAP["SelectLabel"]({"labels": ["1-2"]})


def test_a_resample_after_a_one_hot_reads_its_float32_classes_linearly() -> None:
    # konfai's OneHot writes float32, which konfai resamples linearly; a relabel keeps the stored dtype, which the
    # runtime settles.
    resample = {"Resample": {"spacing": [1.5, 1.5, 1.5]}}
    ops, _ = steps([{"OneHot": {"num_classes": 3}}, resample, {"Write": {"dataset": "./Out:mha"}}], "CT")
    assert ops[1]["interpolation"] == "linear"
    ops, _ = steps([{"FlatLabel": {}}, resample, {"Write": {"dataset": "./Out:mha"}}], "CT")
    assert "interpolation" not in ops[1]


def test_a_tensor_cast_the_predictor_undoes_carries_its_id() -> None:
    # konfai's TensorCast.inverse puts the volume back in the dtype it recorded before the cast, the stored
    # one: the program names the step to undo, and a resample after that inverse reads the stored dtype,
    # which the runtime alone knows.
    config = _prediction({}, {"final_transforms": {"Resample": {"spacing": [1.0, 1.0, 1.0]}}})
    chain = config["Predictor"]["Dataset"]["groups_src"]["g"]["groups_dest"]["g"]["transforms"]
    chain["TensorCast"] = {"dtype": "float32", "inverse": True}
    manifest, _, _ = transform_manifest(config, "Predictor")
    assert manifest["preprocessing"] == [{"op": "cast", "dtype": "float32", "id": "TensorCast"}]
    assert manifest["postprocessing"] == [
        {"op": "inverse", "of": "TensorCast"},
        {"op": "resample", "spacing": [1.0, 1.0, 1.0]},
    ]
    # The dtype the inverse restores is the stored one, which no program op names: past a reduction it is
    # refused rather than hoisted as a cast the compiler would have to guess.
    with pytest.raises(ConfigError, match="the inverse of 'TensorCast'"):
        assemble_program([{"id": "a", "manifest": manifest}, {"id": "b", "manifest": manifest}], reduce="mean")


def test_a_chain_of_threshold_labels_bspline_nearest_even_and_nonzero_compiles() -> None:
    # The resample on a grid rule is undone by that rule at the spacing it asked for.
    chain = {
        "Resample": {"spacing": [1.5, 1.5, 3.0], "grid": "faces", "interpolation": "bspline"},
        "Standardize": {"nonzero": True, "inverse": False},
    }
    outputs = {"final_transforms": {"ThresholdLabels": {"labels": [3, 7], "priority": [1, 0]}}}
    config = _prediction(chain, outputs)
    config["Predictor"]["Dataset"]["groups_src"]["g"]["groups_dest"]["g"]["transforms"]["Resample"][
        "inverse_interpolation"
    ] = "nearest_even"
    manifest, _, _ = transform_manifest(config, "Predictor")
    assert manifest["preprocessing"][1:] == [
        {"op": "resample", "spacing": [1.5, 1.5, 3.0], "grid": "faces", "interpolation": "bspline", "id": "Resample"},
        {"op": "standardize", "nonzero": True},
    ]
    assert manifest["postprocessing"] == [
        {
            "op": "inverse",
            "of": "Resample",
            "grid": "faces",
            "spacing": [1.5, 1.5, 3.0],
            "interpolation": "nearest_even",
        },
        {"op": "threshold_labels", "threshold": 0.5, "labels": [3, 7], "priority": [1, 0], "sigmoid": True},
    ]


def test_a_one_hot_the_predictor_undoes_is_refused_by_name() -> None:
    # konfai undoes a OneHot by an argmax over its class axis, which the runtime has no inverse op for.
    with pytest.raises(ConfigError, match="no inverse for its 'one_hot' op"):
        transform_manifest(_prediction({"OneHot": {"num_classes": 3}}, {}), "Predictor")
    manifest, _, _ = transform_manifest(_prediction({"OneHot": {"num_classes": 3, "inverse": False}}, {}), "Predictor")
    assert manifest["preprocessing"][-1] == {"op": "one_hot", "num_classes": 3}


def test_a_flip_and_a_permute_the_predictor_undoes_stay_per_copy() -> None:
    chain = {"Permute": {"dims": "2|0|1"}, "Flip": {"dims": "0"}}
    config = {"Predictor": {"Dataset": _input_group(chain), "outputs_dataset": {"H": {"OutputDataset": {}}}}}
    manifest, _, per_copy = transform_manifest(config, "Predictor")
    pre, post = manifest["preprocessing"], manifest["postprocessing"]
    assert pre == [{"op": "permute", "axes": [2, 0, 1], "id": "Permute"}, {"op": "flip", "axes": [0], "id": "Flip"}]
    assert post == [{"op": "inverse", "of": "Flip"}, {"op": "inverse", "of": "Permute"}]
    # Both inverses only move rows, which commutes with an elementwise reduction of the copies.
    assert split_post(post, pre, per_copy) == 2


def test_dilate_compiles_with_its_radius() -> None:
    assert OP_MAP["Dilate"]({"dilate": 2}) == {"op": "dilate", "radius": 2}
    # konfai's own default is one voxel.
    assert OP_MAP["Dilate"]({})["radius"] == 1


def test_a_per_channel_statistic_collapses_or_is_refused() -> None:
    # konfai states Standardize's mean and std per channel; one value is the single number the
    # runtime applies, and several have no portable form.
    assert OP_MAP["Standardize"]({"mean": [100.0], "std": [40.0]}) == {
        "op": "standardize",
        "mean": 100.0,
        "std": 40.0,
    }
    assert OP_MAP["Standardize"]({"mean": 100.0, "std": 40.0})["mean"] == 100.0
    with pytest.raises(TransformError, match="one per channel"):
        OP_MAP["Standardize"]({"mean": [10.0, 20.0], "std": [1.0, 2.0]})
    # Neither given is the measured form, which stays portable but is not pointwise.
    assert OP_MAP["Standardize"]({}) == {"op": "standardize"}


@pytest.mark.parametrize(
    ("written", "expected"),
    [("2GB", 2 * 10**9), ("512MiB", 512 * 2**20), ("4096b", 4096), (24, 24 * 2**30), ("auto", None), (None, None)],
)
def test_a_budget_reaches_the_runtime_in_the_bytes_konfai_plans_against(written: object, expected: int | None) -> None:
    # The program's budget is a byte count the runtime plans against, read from the config by konfai's own
    # grammar: a bare number is GiB there, not bytes.
    from konfai.portable import _budget_bytes

    assert _budget_bytes(written) == expected


_WRITE = {"Write": {"dataset": "./Out:nii.gz"}}


@pytest.mark.parametrize(
    ("stage", "params"),
    [
        ("Clip", {"min_value": "min", "max_value": "percentile:99.5"}),
        ("Clip", {"min_value": 0.0, "max_value": "max"}),
        ("Standardize", {}),
        ("Standardize", {"mean": 10.0}),
    ],
)
def test_a_masked_statistic_reads_its_mask_group(stage: str, params: dict[str, object]) -> None:
    # konfai takes a data-dependent bound or statistic over tensor[mask == 1] (`_masked_values`): a TRANSFORM
    # chain reads the group beside the case. Dropping it would widen the statistic to every voxel, hundreds
    # of HU on a masked Standardize.
    (step,), _ = steps({stage: {**params, "mask": "MASK"}, **_WRITE}, "CT")
    assert step["mask_group"] == "MASK"


@pytest.mark.parametrize(
    ("stage", "params"), [("Clip", {"min_value": 0.0, "max_value": 1.0}), ("Standardize", {"mean": 0.0, "std": 1.0})]
)
def test_a_stage_stating_its_numbers_leaves_its_mask_unread(stage: str, params: dict[str, object]) -> None:
    # konfai reads the mask only to measure, so either compiler compiles the stage as if unmasked.
    (step,), _ = steps({stage: {**params, "mask": "MASK"}, **_WRITE}, "CT")
    assert "mask_group" not in step
    assert OP_MAP[stage]({**params, "mask": "MASK"}) == step


def test_a_resample_onto_a_reference_names_where_konfai_looks_it_up(tmp_path: Path) -> None:
    # konfai adopts the grid of an image it looks up in reference_dataset, else in the run's roots in order,
    # under the group named or the dataset's only group (`_ReferenceGrid`); '{case}' follows the case.
    for root, groups in (("Atlas", ["GRID"]), ("A", ["CT"]), ("B", ["CT", "MASK"])):
        for group in groups:
            (tmp_path / root / "CASE_000").mkdir(parents=True, exist_ok=True)
            (tmp_path / root / "CASE_000" / f"{group}.nii.gz").write_bytes(b"")
    atlas, a, b = (str(tmp_path / name) for name in ("Atlas", "A", "B"))
    params = {"reference": "CASE_000", "reference_dataset": f"{atlas}:nii.gz", "fill": -1000}
    (step,), _ = steps({"Resample": params, **_WRITE}, "CT", [f"{a}:nii.gz"])
    assert step == {
        "op": "resample",
        "reference": {"entry": "CASE_000", "datasets": [{"root": atlas, "format": "nii.gz", "group": "GRID"}]},
        "fill": -1000.0,
    }
    params = {"reference": "{case}", "reference_group": "CT", "interpolation": "linear"}
    (step,), _ = steps({"Resample": params, **_WRITE}, "CT", [f"{a}:nii.gz", f"{b}:a:nii.gz"])
    assert step["reference"] == {
        "entry": "{case}",
        "datasets": [{"root": a, "format": "nii.gz", "group": "CT"}, {"root": b, "format": "nii.gz", "group": "CT"}],
    }
    assert step["interpolation"] == "linear"
    with pytest.raises(TransformError, match="cannot tell which group"):
        steps({"Resample": {"reference": "{case}"}, **_WRITE}, "CT", [f"{b}:nii.gz"])


def test_only_a_transform_program_resamples_onto_a_reference() -> None:
    # A prediction program has no dataset to look the reference image up in.
    with pytest.raises(TransformError, match="only a TRANSFORM program looks up"):
        OP_MAP["Resample"]({"reference": "ATLAS", "reference_group": "CT"})


def test_a_mask_reads_its_group_and_refuses_a_file() -> None:
    # konfai's Mask reads a path ending in .mha as a file of its own and anything else as a group of the case.
    (step,), _ = steps({"Mask": {"path": "MASK", "value_outside": -1024}, **_WRITE}, "CT")
    assert step == {"op": "mask", "mask_group": "MASK", "value_outside": -1024.0}
    with pytest.raises(TransformError, match=r"'\./body\.mha'"):
        steps({"Mask": {"path": "./body.mha"}, **_WRITE}, "CT")
    with pytest.raises(TransformError, match="holds a '\\*'"):
        steps({"Standardize": {"mask": "*"}, **_WRITE}, "CT")


def test_a_crop_finds_its_box_at_run_time_after_stages_that_move_nothing() -> None:
    # konfai's Crop keeps the box of the group as stored. A Statistics compiles to nothing and a Save writes what
    # reaches it, so after them the runtime crops the volume konfai finds the box in.
    chain = [
        {"Statistics": {}},
        {"TensorCast": {"dtype": "float32"}},  # a value of the store is one of float32, and keeps its box
        {"Save": {"dataset": "./Work:nii.gz"}},
        {"Crop": {"inverse": True}},
        {"Clip": {"min_value": 0.0, "max_value": 400.0}},
        _WRITE,
    ]
    ops, _ = steps(chain, "CT")
    assert [op["op"] for op in ops] == ["cast", "save", "crop", "clip"]
    assert ops[2] == {"op": "crop"}


@pytest.mark.parametrize(
    "before",
    [
        {"Clip": {"min_value": 0.0, "max_value": 400.0}},
        {"TensorCast": {"dtype": "int16"}},
        {"TensorCast": {"dtype": "float32", "round": True}},  # 0.4 rounds to 0, out of a nonzero box
        {"Mask": {"path": "MASK"}},
        {"Flip": {"dims": "0"}},
        {"Resample": {"spacing": [1.5, 1.5, 1.5], "interpolation": "linear"}},
        {"Padding": {"padding": [1, 1, 1, 1, 1, 1]}},
        {"Crop": {}},
    ],
)
def test_a_crop_after_a_stage_moving_the_grid_or_the_values_is_refused_by_name(before: dict) -> None:
    # konfai finds the box in the group as stored whatever the stage before did to the volume it crops.
    (earlier,) = before
    with pytest.raises(TransformError, match=f"follows '{earlier}'"):
        steps([before, {"Save": {"dataset": "./Work:nii.gz"}}, {"Crop": {}}, _WRITE], "CT")


def test_a_prediction_crop_compiles_and_is_undone_after_the_forward() -> None:
    # The runtime crops to the foreground box and puts back what it dropped with the nearest voxel, as
    # konfai's `Crop.inverse` does (`F.pad(..., "replicate")`).
    manifest, _folds, _per_copy = transform_manifest(_prediction({"Crop": {}}, {}), "Predictor")
    assert {"op": "crop", "id": "Crop"} in manifest["preprocessing"]
    assert {"op": "inverse", "of": "Crop"} in manifest["postprocessing"]


def test_a_prediction_crop_after_a_stage_that_moved_the_volume_is_refused_by_name() -> None:
    # konfai crops to the box of the group as stored, the runtime to the box of the volume in hand.
    with pytest.raises(ConfigError, match="'Crop' follows 'Resample'"):
        transform_manifest(_prediction({"Resample": {"spacing": [1, 1, 1]}, "Crop": {}}, {}), "Predictor")
    # A crop to a shape finds no box: both engines keep the centre of the volume they are handed.
    centred = {"Resample": {"spacing": [1, 1, 1]}, "Crop": {"shape": [4, 4, 4]}}
    manifest, _folds, _per_copy = transform_manifest(_prediction(centred, {}), "Predictor")
    assert {"op": "crop", "shape": [4, 4, 4], "id": "Crop"} in manifest["preprocessing"]
    # A handed crop measures its box on the volume in hand, as the runtime always does: the step is a plain crop.
    handed = {"Resample": {"spacing": [1, 1, 1]}, "Crop": {"handed": True, "nonzero": True}}
    manifest, _folds, _per_copy = transform_manifest(_prediction(handed, {}), "Predictor")
    assert {"op": "crop", "nonzero": True, "id": "Crop"} in manifest["preprocessing"]


@pytest.mark.parametrize("sink", ["OutputLayerDataset", "Custom:ConfidenceSelect"])
def test_a_prediction_output_written_by_another_sink_is_refused_by_name(sink: str) -> None:
    # Another sink assembles its patches its own way; the program would blend and write as OutputDataset does.
    with pytest.raises(ConfigError, match=f"written by '{sink}'"):
        transform_manifest(_prediction({}, {"name_class": sink}), "Predictor")
    transform_manifest(_prediction({}, {"name_class": "OutputDataset"}), "Predictor")


@pytest.mark.parametrize(
    ("stage", "params"),
    [("Clip", {"min_value": "min", "max_value": 1.0}), ("Standardize", {"std": 1.0, "inverse": False})],
)
def test_a_prediction_stage_reading_a_mask_group_has_the_program_read_it(stage: str, params: dict) -> None:
    # konfai hands such a stage the group as stored, whichever chain prepares it elsewhere: the program
    # reads it into a buffer of its own, beside the input, and runs nothing on it.
    assert OP_MAP[stage]({**params, "mask": "MASK"})["mask_group"] == "MASK"
    manifest, _folds, _per_copy = transform_manifest(_prediction({stage: {**params, "mask": "MASK"}}, {}), "Predictor")
    assert manifest["reads"] == [{"group": "MASK", "channels": 1, "preprocessing": []}]
    assert assemble_program([{"id": "m", "manifest": manifest}], reduce="mean")["steps"][0]["in"] == [
        "input",
        "MASK",
    ]


@pytest.mark.parametrize(
    ("stage", "params"), [("Standardize", {}), ("Standardize", {"mask": "MASK"}), ("Normalize", {})]
)
def test_a_lazy_statistic_is_refused_by_name(stage: str, params: dict[str, object]) -> None:
    # konfai's lazy stage hands its input back and records its statistic for a later stage to apply.
    for lazy in (True, "True"):
        with pytest.raises(TransformError, match=f"'{stage}' is lazy"):
            steps({stage: {**params, "lazy": lazy}, **_WRITE}, "CT")
        with pytest.raises(TransformError, match=f"'{stage}' is lazy"):
            transform_manifest(_prediction({stage: {**params, "lazy": lazy}}, {}), "Predictor")
    assert steps({stage: {**params, "lazy": False}, **_WRITE}, "CT") == steps({stage: params, **_WRITE}, "CT")


def test_a_resolved_config_spelling_no_mask_still_compiles() -> None:
    # KonfAI writes the resolved default back as the string "None".
    ops, _ = steps({"Clip": {"min_value": 0.0, "max_value": 1.0, "mask": "None"}, "Write": {"dataset": "./Out"}}, "CT")
    assert ops == [{"op": "clip", "min_value": 0.0, "max_value": 1.0}]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("./Dataset", {"root": "./Dataset", "format": "mha"}),
        ("./Dataset:nii.gz", {"root": "./Dataset", "format": "nii.gz"}),
        # The flag sits between the path and the format, so the format is still the last field.
        ("./Dataset:a:mha", {"root": "./Dataset", "format": "mha"}),
        ("./Dataset:i:nii.gz", {"root": "./Dataset", "format": "nii.gz"}),
        ("./Store:omezarr@0", {"root": "./Store", "format": "omezarr"}),
        # Every case of an h5 dataset sits in one file, which the program names as konfai opens it.
        ("./Dataset:h5", {"root": "./Dataset", "format": "h5", "store": "Dataset.h5"}),
        ("./Dataset.h5:h5", {"root": "./Dataset.h5", "format": "h5", "store": "Dataset.h5"}),
    ],
)
def test_a_dataset_spec_splits_by_konfais_grammar(spec: str, expected: dict) -> None:
    assert source_of(spec) == expected


def test_a_write_target_splits_by_konfais_grammar_and_names_its_store() -> None:
    # konfai's Save splits its dataset from the right, so a drive letter is part of the path.
    _, write = steps({"Write": {"dataset": "C:\\Prepared:nii.gz"}}, "CT")
    assert {key: write[key] for key in ("root", "format", "group")} == {
        "root": "C:\\Prepared",
        "format": "nii.gz",
        "group": "CT",
    }
    assert "store" not in write
    _, write = steps({"Write": {"dataset": "./Prepared:h5", "group": "CT_h5"}}, "CT")
    assert write == {
        "root": "./Prepared",
        "format": "h5",
        "group": "CT_h5",
        "entry_suffixes": [],
        "store": "Prepared.h5",
    }


@pytest.mark.parametrize("file_format", ["nii.gz", "mha", "omezarr", "zarr", "itktransform"])
def test_a_write_target_names_every_suffix_konfai_resumes_on(tmp_path: Path, file_format: str) -> None:
    # konfai skips a case whose entry is there under any name the format's backend probes
    # (`Dataset.is_dataset_exist`): a CT.mha is the entry of a Write to nii.gz. Each candidate is asked
    # of konfai itself, alone under a root of its own, so the list cannot drift from the backends.
    from konfai.utils.dataset import Dataset
    from konfai.utils.utils import STORE_FORMS, SUPPORTED_EXTENSIONS

    _, write = steps({"Write": {"dataset": f"{tmp_path / 'Out'}:{file_format}"}}, "CT")
    directory = "zarr" in file_format  # a store is a directory; every other entry here a file
    candidates = sorted(
        {"", ".tfm", ".OME.ZARR", *STORE_FORMS, *(f".{extension}" for extension in SUPPORTED_EXTENSIONS)}
    )
    counted = []
    for index, suffix in enumerate(candidates):
        entry = tmp_path / str(index) / "CASE" / f"CT{suffix}"
        entry.parent.mkdir(parents=True)
        if directory:
            entry.mkdir()
        else:
            entry.write_bytes(b"")
        if Dataset(tmp_path / str(index), file_format).is_dataset_exist("CT", "CASE"):
            counted.append(suffix)
    # konfai also matches a store's name in any letter case (`OmeZarrFile._listed_as`), which the runtime
    # does not.
    assert sorted(write["entry_suffixes"]) == sorted(s for s in counted if s != ".OME.ZARR")


def test_a_dcm_dataset_is_stated_as_konfai_opens_it(tmp_path: Path) -> None:
    # konfai reads and writes a 'dcm' dataset through SimpleITK, one <group>.dcm file per case, and as a DICOM
    # series once its root holds series (`Dataset.__init__` detects them): the program states which it is.
    import numpy as np
    from konfai.utils.dicom import write_dicom_series

    fresh = tmp_path / "Fresh"
    assert source_of(f"{fresh}:dcm")["format"] == "dcm"
    _, write = steps({"Write": {"dataset": f"{fresh}:dcm"}}, "CT")
    assert write["format"] == "dcm" and ".dcm" in write["entry_suffixes"]

    series = tmp_path / "Series"
    write_dicom_series(series / "CASE_000" / "CT", np.zeros((1, 2, 3, 4), dtype=np.int16))
    assert source_of(f"{series}:dcm")["format"] == "dicom"
    _, write = steps({"Write": {"dataset": f"{series}:dcm"}}, "CT")
    assert (write["format"], write["entry_suffixes"]) == ("dicom", [""])


def test_a_group_konfai_resolves_by_listing_is_refused() -> None:
    # konfai lists the dataset for a '*' component (`Dataset._get_sub_directories`, `H5File.get_names`).
    assert _literal_group("CT/Volume", TransformError) == "CT/Volume"
    with pytest.raises(TransformError, match="'\\*/CT'"):
        _literal_group("*/CT", TransformError)


def test_a_store_read_at_a_coarser_level_is_refused() -> None:
    # The runtime reads level 0; a level it would ignore gives a volume at the wrong resolution.
    with pytest.raises(TransformError, match="resolution level 2"):
        source_of("./Store:omezarr@2")


def _one_model() -> list[dict]:
    manifest = {"input": {"channels": 1}, "output": {"channels": 1}, "patch": {"size": [4, 4]}}
    return [{"id": "m", "manifest": {**manifest, "preprocessing": [], "postprocessing": []}}]


def test_a_test_time_pass_is_drawn_after_the_chain_and_undone_before_the_one_reduction() -> None:
    program = assemble_program(_one_model(), reduce="mean", passes=[[], [{"op": "flip", "axes": [0, 1]}]])
    steps = [(step.get("op", "model"), step["out"]) for step in program["steps"]]
    assert steps == [("model", "m__p0"), ("model", "m__p1"), ("mean", "output")]
    # The copy is flipped where konfai draws it, after the chain the manifest carries, and put back first.
    copy = program["steps"][1]["manifest"]
    assert copy["preprocessing"][-1] == {"op": "flip", "axes": [0, 1], "id": "__tta_0"}
    assert copy["postprocessing"][0] == {"op": "inverse", "of": "__tta_0"}
    # The identity copy runs the manifest as it is.
    assert program["steps"][0]["manifest"] == _one_model()[0]["manifest"]


#: Stages the two compilers place themselves, not through the op registries: the chain structure
#: (`compile_chain` splits an Expand's copies and a Reduce's fold, a Write and a Save name a
#: destination), a stage that records a fact and alters no value, and a base class no config names.
_PLACED_BY_THE_COMPILERS = {"Expand", "Reduce", "Save", "Statistics", "Write", "TransformInverse"}


def test_every_konfai_stage_and_draw_is_carried_or_refused_by_name() -> None:
    # The coverage that matters is konfai's own registry, not the configs that ship: a stage this
    # compiler has never heard of must still meet a refusal that names it, never a crash or a silent
    # drop. konfai growing one is what makes this fail.
    import konfai.data.augmentation as augmentation
    import konfai.data.transform as transform
    from konfai.data.augmentation import DataAugmentation
    from konfai.data.transform import Transform
    from konfai.portable import _CHAIN_OPS, PROGRAM_TRANSFORMS, REFUSED, compile_pipeline

    def named(module: object, base: type) -> set[str]:
        return {
            name
            for name, held in vars(module).items()
            if inspect.isclass(held) and issubclass(held, base) and held is not base
        }

    carried = set(OP_MAP) | set(_CHAIN_OPS) | set(PROGRAM_TRANSFORMS) | set(REFUSED)
    stages = named(transform, Transform) | named(augmentation, DataAugmentation)
    unplaced = sorted(stages - carried - _PLACED_BY_THE_COMPILERS)
    for stage in unplaced:
        # A draw konfai does not carry outside an Expand: the compilers refuse it by name where it is
        # spelled, which is what this asserts for every one of them at once.
        with pytest.raises(KonfAIError, match=re.escape(stage)):
            compile_pipeline({stage: {}}, fold=False)


def test_a_resample_on_a_grid_rule_compiles_its_antialias_and_inverse() -> None:
    # The inverse reads with the interpolation and the fill the stage states for it, not the forward's.
    resample = {
        "spacing": [1.0, 1.0, 1.0],
        "grid": "faces",
        "antialias": "gaussian",
        "fill": "edge",
        "interpolation": "linear",
        "inverse_interpolation": "nearest",
        "inverse_fill": 0.0,
    }
    manifest, _folds, _per_copy = transform_manifest(_prediction({"Resample": resample}, {}), "Predictor")
    assert {
        "op": "resample",
        "spacing": [1.0, 1.0, 1.0],
        "grid": "faces",
        "antialias": "gaussian",
        "fill": "edge",
        "interpolation": "linear",
        "id": "Resample",
    } in manifest["preprocessing"]
    inverse = {"op": "inverse", "of": "Resample", "grid": "faces", "spacing": [1.0, 1.0, 1.0]}
    assert {**inverse, "interpolation": "nearest"} in manifest["postprocessing"]


_AUG = "konfai.data.augmentation:"

#: What a resolved config spells for the stages the compilers read through `_CHAIN_OPS`, a `Write`, a `Save`,
#: and the four draws a prediction's test-time augmentation or a TRANSFORM's Expand carries. Every parameter
#: konfai binds on their constructors (plus the `prob` gate the binder reads beside a test-time draw) sits in
#: one of three tables. CARRIED_PARAMETERS holds a value that, joined to the stage's baseline, changes what
#: the compilers emit or meets a refusal naming the stage: the parameter reaches the program. `inverse` is
#: carried by the prediction chain alone, as the step's id; a TRANSFORM never undoes a stage, so there
#: konfai itself ignores it. A parameter pinned by a test of its own names that test.
CARRIED_PARAMETERS: dict[str, dict[str, object]] = {
    "Argmax": {"dim": 1},
    "Softmax": {"dim": 1},
    "Canonical": {"inverse": False, "resample": False},
    "Crop": {
        "inverse": False,
        "source": "OTHER",
        "labels": [1],
        "margin_mm": 5.0,
        "fill": 0.0,
        "empty": "fill",
        "nonzero": True,
        "shape": [4, 4, 4],
        "handed": test_a_prediction_crop_after_a_stage_that_moved_the_volume_is_refused_by_name,
    },
    "Clip": {"min_value": 3.0, "max_value": "max", "mask": "MASK"},
    "TensorCast": {"dtype": "int16", "inverse": False, "round": True},
    "Resample": {
        "spacing": [3.0, 3.0, 3.0],
        "shape": [4, 4, 4],
        "reference": "X",
        "reference_group": test_a_resample_onto_a_reference_names_where_konfai_looks_it_up,
        "reference_dataset": test_a_resample_onto_a_reference_names_where_konfai_looks_it_up,
        "align": "origin",
        "interpolation": "nearest",
        "fill": 5.0,
        "inverse": False,
        "field": "F",
        "field_group": "G",
        "grid": "zoom",
        "antialias": "gaussian",
        "inverse_interpolation": test_a_resample_on_a_grid_rule_compiles_its_antialias_and_inverse,
        "inverse_fill": test_a_resample_on_a_grid_rule_compiles_its_antialias_and_inverse,
    },
    "Padding": {"padding": [2, 2, 2, 2, 2, 2], "mode": "constant:3", "inverse": False},
    "Dilate": {"dilate": 3},
    "Flip": {"dims": "0", "inverse": False},
    "Permute": {"dims": "2|0|1", "inverse": False},
    "Standardize": {"mean": 3.0, "std": 2.0, "mask": "MASK", "inverse": False, "nonzero": True},
    "Normalize": {"min_value": 0.0, "max_value": 2.0, "inverse": False},
    "UnNormalize": {"min_value": 0.0, "max_value": 2.0},
    "SelectLabel": {"labels": ["(1,2)"]},
    "FlatLabel": {"labels": [1]},
    "ThresholdLabels": {"labels": [4, 5], "threshold": 0.3, "priority": [1, 0], "sigmoid": False},
    "OneHot": {"num_classes": 3, "inverse": False},
    "Gradient": {"per_dim": True},
    "HistogramMatching": {"reference_group": "S"},
    "Percentage": {"baseline": 2.0},
    "KeepLargestComponent": {"mask": 1, "per_label": True, "connectivity": 26},
    "Mask": {"path": "OTHER", "value_outside": 5.0},
    "Write": {"dataset": "./Other:mha", "group": "G2"},
    "Save": {"dataset": "./Other:mha", "group": "G2"},
    _AUG + "Brightness": {"b_std": 2.0},
    _AUG + "Flip": {"f_prob": [0.0, 1.0, 0.0], "vector_field": True, "prob": 0.0},
    _AUG + "Permute": {"prob_permute": None, "prob": 0.0},
    _AUG + "Rotate": {"is_quarter": False, "in_plane": True, "prob": 0.0},
}

#: A value the compilers refuse, naming the parameter: konfai reads it and the portable runtime has no form for it.
REFUSED_PARAMETERS: dict[str, dict[str, object]] = {
    "Clip": {"save_clip_min": True, "save_clip_max": True},
    "Standardize": {"lazy": True},
    "Canonical": {"orientation": "LPS"},
    "Normalize": {"lazy": True, "channels": [0]},
    "Resample": {"transforms": "T"},
    "Write": {"scale_factors": [2]},
    "Save": {"scale_factors": [2]},
    _AUG + "Brightness": {"groups": ["CT"]},
}

#: A value that leaves the compiled step as it is, and why konfai's own values do not depend on it here.
IGNORED_PARAMETERS: dict[str, dict[str, tuple[object, str]]] = {
    "KeepLargestComponent": {"rows": (8, "it sizes the slabs the components are found by, not which are kept.")},
    "Canonical": {
        "fill": (
            -1024.0,
            "it fills what an oblique direction uncovers, and the portable runtime refuses an oblique direction.",
        ),
    },
    "Resample": {
        "precision": (
            "fast",
            "konfai computes the sampling coordinates in float32 with precision: fast and the portable runtime in"
            " float64: the step is the same, the rounding of a coordinate is not.",
        ),
    },
    "Write": {
        "downsample_method": (
            "mean",
            "it averages the pyramid levels, and scale_factors, which builds them, is refused.",
        )
    },
    "Save": {
        "downsample_method": (
            "mean",
            "it averages the pyramid levels, and scale_factors, which builds them, is refused.",
        )
    },
    _AUG + "Rotate": {
        "a_min": (90.0, "a quarter turn draws its angles from 90, 180 and 270 degrees whatever the bounds."),
        "a_max": (91.0, "a quarter turn draws its angles from 90, 180 and 270 degrees whatever the bounds."),
    },
}

#: The parameters a stage needs to compile at all, which every probe joins.
_BASELINE: dict[str, dict[str, object]] = {
    "Clip": {"min_value": "min", "max_value": 400.0},
    "Resample": {"spacing": [2.0, 2.0, 2.0]},
    "Padding": {"padding": [1, 1, 1, 1, 1, 1]},
    "UnNormalize": {"min_value": -1.0, "max_value": 1.0},
    "OneHot": {"num_classes": 2},
    "ThresholdLabels": {"labels": [1, 2]},
    "HistogramMatching": {"reference_group": "R"},
    "Percentage": {"baseline": 1.0},
    "Mask": {"path": "MASK"},
    "Write": {"dataset": "./Out:mha"},
    "Save": {"dataset": "./Out:mha"},
    _AUG + "Brightness": {"b_std": 1.0},
    _AUG + "Flip": {"f_prob": [1.0, 0.0, 1.0]},
    _AUG + "Permute": {"prob_permute": [1.0, 1.0]},
    _AUG + "Rotate": {"is_quarter": True},
}


def _stage_class(key: str) -> type:
    module, _, name = key.rpartition(":")
    return getattr(import_module(module or "konfai.data.transform"), name)


def _compile_stage(key: str, params: dict[str, object]) -> object:
    """What the compilers make of ``key`` with ``params``, where a config spells it: a test-time draw as the
    passes it draws, an Expand draw and a program stage as the TRANSFORM chain's steps, a Write or a Save as
    the destination, every other stage as the prediction chain's steps, where its inverse is named."""
    from konfai.portable import PROGRAM_TRANSFORMS, compile_pipeline

    name = key.rpartition(":")[-1]
    if key.startswith(_AUG) and name != "Brightness":
        return tta_passes(_prediction({}, {}, flips=2, draws={name: params}), "Predictor")
    if name in ("Write", "Save"):
        return steps([{name: params}], "G")[1]
    if name in OP_MAP and name not in PROGRAM_TRANSFORMS:
        return compile_pipeline({name: params}, invertible=True)[0]
    return steps([{"Expand": {"nb": 2}}, {name: params}, _WRITE], "G")[0]


def test_canonical_reads_its_orientation_in_any_case() -> None:
    assert _compile_stage("Canonical", {"orientation": "ras"}) == _compile_stage("Canonical", {})


def _outcome(key: str, params: dict[str, object]) -> object:
    """The compiled form, or the refusal naming the stage, which is an outcome of its own."""
    try:
        return _compile_stage(key, params)
    except KonfAIError as refusal:
        assert key.rpartition(":")[-1] in str(refusal), refusal
        return str(refusal)


@pytest.mark.parametrize(
    "key",
    sorted(
        {_AUG + name if name == "Brightness" else name for name in _CHAIN_OPS}
        | {"Write", "Save", _AUG + "Flip", _AUG + "Permute", _AUG + "Rotate"}
        | {f"konfai.data.reduction:{name}" for name in ("Mean", "Median", "Std", "Vote", "Concat")}
    ),
)
def test_every_constructor_parameter_is_carried_refused_or_listed(key: str) -> None:
    # A name in the registry proves nothing about its parameters: `Clip(save_clip_min)` and
    # `Permute(prob_permute: None)` both compiled while the compilers read neither. konfai adding a parameter,
    # or a compiler dropping a refusal, is what makes this fail.
    signature = inspect.signature(_stage_class(key).__init__).parameters
    bound = {p for p, s in signature.items() if p != "self" and s.kind not in (s.VAR_POSITIONAL, s.VAR_KEYWORD)}
    if key.startswith(_AUG) and key != _AUG + "Brightness":
        bound.add("prob")  # the gate konfai binds through `Prob`, beside the draw's own constructor
    carried, refused, ignored = (
        table.get(key, {}) for table in (CARRIED_PARAMETERS, REFUSED_PARAMETERS, IGNORED_PARAMETERS)
    )
    assert set(carried) | set(refused) | set(ignored) == bound, "every parameter is in one table"
    assert len(carried) + len(refused) + len(ignored) == len(bound), "and in one only"
    baseline = _BASELINE.get(key, {})
    plain = _outcome(key, baseline) if carried or ignored else None
    for param, probe in carried.items():
        if callable(probe):
            continue
        assert _outcome(key, {**baseline, param: probe}) != plain, f"{key}.{param} reaches no program"
    for param, probe in refused.items():
        with pytest.raises(KonfAIError, match=param):
            _compile_stage(key, {**baseline, param: probe})
    for param, (probe, reason) in ignored.items():
        assert _outcome(key, {**baseline, param: probe}) == plain, f"{key}.{param}: {reason}"


def _two_groups(second: dict | None = None) -> dict:
    """A Predictor whose model reads MR then MASK, the second prepared by ``second``."""
    return {
        "Predictor": {
            "Model": {"classpath": "Probe"},
            "Dataset": {
                "groups_src": {
                    "MR": {"groups_dest": {"MR": {"is_input": True, "transforms": {"Normalize": {"inverse": False}}}}},
                    "MASK": {"groups_dest": {"MASK": {"is_input": True, "transforms": second}}},
                }
            },
            "outputs_dataset": {"H": {"OutputDataset": {}}},
            "combine": "Mean",
        }
    }


def test_a_model_reading_two_groups_names_the_second_and_the_chain_that_prepares_it() -> None:
    # konfai hands the model every is_input group, in the dataset's order, patched on one grid: the
    # manifest carries the second group's own chain and the step reads its buffer beside the input.
    manifest, _folds, _per_copy = transform_manifest(
        _two_groups({"Clip": {"min_value": 0, "max_value": 1}}), "Predictor"
    )
    assert manifest["reads"] == [
        {"group": "MASK", "channels": 1, "preprocessing": [{"op": "clip", "min_value": 0.0, "max_value": 1.0}]}
    ]
    # The first group's chain is the one the manifest's own preprocessing carries, and the one undone.
    assert [step["op"] for step in manifest["preprocessing"]] == ["normalize"]
    program = assemble_program([{"id": "m", "manifest": manifest}], reduce="mean")
    assert program["steps"][0]["in"] == ["input", "MASK"]


def test_an_extension_carried_is_a_file_named_apart_from_what_the_program_names(tmp_path: Path) -> None:
    # A program names its extension beside it: a file that is there (refused before the compile writes anything),
    # never one of the files it names otherwise (an exported graph copied over), nor the program itself. One already
    # beside it is reused under its own name; a string of the program that names no file is none of them.
    from konfai.portable import carry_extension, compile_destinations

    out = tmp_path / "out" / "prediction.json"
    out.parent.mkdir()
    (out.parent / "model.onnx").write_bytes(b"graph")
    program = {"models": {"model": "model.onnx"}}
    with pytest.raises(ConfigError, match=r"missing\.wasm' is no file"):
        compile_destinations(out, out.parent / "missing.wasm")
    for name in ("model.onnx", "prediction.json"):
        (tmp_path / name).write_bytes(b"\0asm")
        with pytest.raises(ConfigError, match=re.escape(name)):
            carry_extension(tmp_path / name, dict(program), out)
    assert (out.parent / "model.onnx").read_bytes() == b"graph"
    beside = out.parent / "own.wasm"
    beside.write_bytes(b"\0asm")
    named = dict(program)
    carry_extension(beside, named, out)
    assert named["extension"] == "own.wasm"
    # A name the program holds as no file (a label's, a group's) is no collision; a graph's spelled in other cases is.
    for name in ("roi.wasm", "labels.onnx"):
        (tmp_path / name).write_bytes(b"\0asm")
        named = {
            **program,
            "outputs": [{"labels": {"1": "roi.wasm"}, "steps": [{"op": "mask", "group": "labels.onnx"}]}],
        }
        carry_extension(tmp_path / name, named, out)
        assert named["extension"] == name and (out.parent / name).read_bytes() == b"\0asm"
    (tmp_path / "MODEL.ONNX").write_bytes(b"\0asm")
    with pytest.raises(ConfigError, match=r"MODEL\.ONNX"):
        carry_extension(tmp_path / "MODEL.ONNX", dict(program), out)
    (tmp_path / "fold_0.onnx").write_bytes(b"\0asm")
    with pytest.raises(ConfigError, match=r"fold_0\.onnx"):
        carry_extension(tmp_path / "fold_0.onnx", {"models": {"fold_0": "fold_0.onnx"}}, out)
    # A link left where it would land is not written through.
    (out.parent / "engine.wasm").symlink_to(out.parent / "model.onnx")
    (tmp_path / "engine.wasm").write_bytes(b"\0asm")
    with pytest.raises(ConfigError, match="link"):
        carry_extension(tmp_path / "engine.wasm", dict(program), out)
    assert (out.parent / "model.onnx").read_bytes() == b"graph"


@pytest.mark.parametrize(
    "site", ["the extension", "a graph", "a transform program", "an evaluation program", "a prediction program"]
)
def test_every_file_a_compile_writes_lands_on_nothing_but_a_file_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    # A directory where the compile writes is refused by name, as any destination it would write into or through
    # (konfai.utils.utils.owned_destination).
    from konfai.portable import carry_extension, compile_transform

    out = tmp_path / "out"
    out.mkdir()
    if site == "the extension":
        (out / "engine.wasm").mkdir()
        (tmp_path / "engine.wasm").write_bytes(b"\0asm")
        write, name = lambda: carry_extension(tmp_path / "engine.wasm", {}, out / "prediction.json"), "engine.wasm"
    elif site == "a graph":
        pytest.importorskip("onnxscript")
        from konfai.export import export_folds

        (out / "fold.onnx").mkdir()
        write, name = lambda: export_folds([lambda tensor: tensor * 2], out / "fold.onnx", 1, 2, "float32"), "fold.onnx"
    elif site == "a transform program":
        config = _reducing_config(tmp_path, "Clip: {min_value: 0.0, max_value: 1.0}\n")
        (out / "transform.json").mkdir()
        write, name = (
            lambda: compile_transform(config, tmp_path / "Transforms", out / "transform.json"),
            "transform.json",
        )
    elif site == "a prediction program":
        pytest.importorskip("onnxscript")
        import torch
        from konfai.portable import compile_prediction

        config = _tiny_prediction(tmp_path, monkeypatch, "TinySeg:TinySegNet", "Head:Softmax")
        from TinySeg import TinySegNet

        torch.save({"Model": TinySegNet().network_states()}, tmp_path / "fold_0.pt")
        (out / "prediction.json").mkdir()
        write, name = (
            lambda: compile_prediction(
                config, [tmp_path / "fold_0.pt"], tmp_path / "Predictions", out / "prediction.json"
            ),
            "prediction.json",
        )
    else:
        from konfai.portable_evaluation import compile_evaluation
        from test_portable_evaluation import _evaluation

        config = _evaluation(tmp_path, "            Dice: {labels: [1]}\n")
        (out / "evaluation.json").mkdir()
        write, name = (
            lambda: compile_evaluation(config, tmp_path / "Evaluations", out / "evaluation.json"),
            "evaluation.json",
        )
    with pytest.raises(ConfigError, match=rf"{re.escape(name)}' is a directory"):
        write()


@pytest.mark.parametrize(
    "planted",
    ["a link at the program", "a link at the manifest", "a missing extension", "a link at a statistic's finish graph"],
)
def test_a_compile_refuses_its_destinations_before_it_writes_a_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, planted: str
) -> None:
    # The program, the extension it carries, the manifest an export writes beside the graph and a statistic's second
    # graph: each is refused before a graph is written over the one a previous compile left.
    pytest.importorskip("onnxscript")
    import torch
    from konfai.export import export_case_statistic, export_to_onnx
    from konfai.portable import compile_prediction

    config = _tiny_prediction(tmp_path, monkeypatch, "TinySeg:TinySegNet", "Head:Softmax")
    from TinySeg import TinySegNet

    torch.save({"Model": TinySegNet().network_states()}, tmp_path / "fold_0.pt")
    out = tmp_path / "out"
    out.mkdir()
    (out / "model.onnx").write_bytes(b"the previous graph")
    (tmp_path / "elsewhere.json").write_text("not the program's")
    extension = None
    if planted == "a link at the program":
        (out / "prediction.json").symlink_to(tmp_path / "elsewhere.json")
    elif planted in ("a link at the manifest", "a link at a statistic's finish graph"):
        (out / ("manifest.json" if planted == "a link at the manifest" else "finish.onnx")).symlink_to(
            tmp_path / "elsewhere.json"
        )
    else:
        extension = tmp_path / "missing.wasm"

    class Mean:  # a CaseStatistic's two halves: a block's sum and count, then their quotient
        halo = 0

        def partial(self, block):
            return torch.stack([block.sum(), torch.ones_like(block).sum()])

        def finish(self, total):
            return total[0] / total[1]

    with pytest.raises(ConfigError, match=r"is a link|is no file"):
        if planted == "a link at the manifest":
            export_to_onnx(TinySegNet(), out, torch.randn(1, 1, 16, 16))
        elif planted == "a link at a statistic's finish graph":
            export_case_statistic(Mean(), out / "model.onnx", out / "finish.onnx", rank=3)
        else:
            compile_prediction(
                config, [tmp_path / "fold_0.pt"], tmp_path / "Predictions", out / "prediction.json", extension=extension
            )
    assert (out / "model.onnx").read_bytes() == b"the previous graph"
    assert (tmp_path / "elsewhere.json").read_text() == "not the program's"


def test_a_class_of_the_app_s_run_again_takes_a_dtype_it_takes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The stored dtype only the run knows, a class taking int16 alone is run again, to count its channels, in a
    # dtype it takes.
    (tmp_path / "StoredInt.py").write_text(
        "import torch\n"
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Pairs(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        if tensor.dtype != torch.int16:\n"
        "            raise TypeError(f'int16, not {tensor.dtype}')\n"
        "        return tensor.repeat(2, *[1] * (tensor.dim() - 1))\n\n"
        "class Two(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        if tensor.shape[0] != 2:\n"
        "            raise ValueError(f'two channels, not {list(tensor.shape)}')\n"
        "        return tensor\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    manifest, _folds, _per_copy = transform_manifest(
        _two_groups({"StoredInt:Pairs": {}, "StoredInt:Two": {}}), "Predictor", extension=True
    )
    assert [step["class"] for step in manifest["reads"][0]["preprocessing"]] == ["StoredInt:Pairs", "StoredInt:Two"]


@pytest.mark.parametrize(
    ("trim", "copies", "refused"),
    [(1, 1, True), (0, 1, False), (0, 2, False)],
    ids=["trims", "keeps-the-grid", "keeps-the-grid-in-two-channels"],
)
def test_a_class_of_the_app_s_moving_a_second_input_off_the_grid_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trim: int, copies: int, refused: bool
) -> None:
    # konfai patches every group of a case on the first one's grid: a whole-volume class of the app's own that trims
    # the second is refused as konfai's own regridding stages are, and one that keeps the grid compiles, whatever
    # channels it hands on.
    (tmp_path / "SecondGrid.py").write_text(
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Trim(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __init__(self, by: int = 1, copies: int = 1) -> None:\n"
        "        super().__init__()\n"
        "        self.by, self.copies = by, copies\n\n"
        "    def transform_shape(self, group_src, name, shape, cache_attribute):\n"
        "        return [extent - 2 * self.by for extent in shape]\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        kept = tensor[(slice(None), *[slice(self.by, extent - self.by) for extent in tensor.shape[1:]])]\n"
        "        return kept.repeat(self.copies, *[1] * (kept.dim() - 1))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config = _two_groups({"SecondGrid:Trim": {"by": trim, "copies": copies}})
    if refused:
        with pytest.raises(ConfigError, match="off the grid"):
            transform_manifest(config, "Predictor", extension=True)
    else:
        assert transform_manifest(config, "Predictor", extension=True)[0]["reads"][0]["preprocessing"][0]["op"] == (
            "extension"
        )


def test_a_class_of_the_app_s_is_probed_on_each_grid_as_a_case_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A class remembering a case's first spacing (konfai's case_facts) brings any grid to a unit spacing: the placed
    # grid is probed as another case, whose facts are its own, and the move is seen.
    (tmp_path / "CaseGrid.py").write_text(
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Unit(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def transform_shape(self, group_src, name, shape, cache_attribute):\n"
        "        self.datasets[0].case_facts.setdefault(name, cache_attribute.get_np_array('Spacing'))\n"
        "        return shape\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        spacing = cache_attribute.get_np_array('Spacing') / self.datasets[0].case_facts[name]\n"
        "        cache_attribute['Spacing'] = spacing\n"
        "        return tensor\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ConfigError, match="off the grid"):
        transform_manifest(_two_groups({"CaseGrid:Unit": {}}), "Predictor", extension=True)


def test_a_class_of_the_app_s_reads_the_header_of_the_grid_it_is_probed_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A class copying its group's stored spacing keeps the grid: the placed grid's header is the placed one.
    (tmp_path / "HeaderGrid.py").write_text(
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Header(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def transform_shape(self, group_src, name, shape, cache_attribute):\n"
        "        self.group = group_src\n"
        "        return shape\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        cache_attribute['Spacing'] = self.datasets[0].get_infos(self.group, name)[1].get_np_array('Spacing')\n"
        "        return tensor\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    manifest = transform_manifest(_two_groups({"HeaderGrid:Header": {}}), "Predictor", extension=True)[0]
    assert manifest["reads"][0]["preprocessing"][0]["op"] == "extension"


def test_a_class_of_the_app_s_is_probed_in_each_stored_dtype_it_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Where only the run knows the stored dtype, a class keeping the grid of an integer volume and trimming a float one
    # is refused: each stored dtype it takes is a case it may be handed.
    (tmp_path / "DtypeGrid.py").write_text(
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Trim(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor[:, 1:] if tensor.is_floating_point() else tensor\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ConfigError, match="for the stored dtypes it takes"):
        transform_manifest(_two_groups({"DtypeGrid:Trim": {}}), "Predictor", extension=True)


def test_a_class_of_the_app_s_taking_blocks_of_voxels_is_probed_on_a_volume_it_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A class folding blocks of 2 x 2 x 2 voxels takes even extents alone: the probe's are, and other along each axis.
    (tmp_path / "Blocks.py").write_text(
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Even(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        if any(extent % 2 for extent in tensor.shape[1:]):\n"
        "            raise ValueError(f'even extents, not {list(tensor.shape[1:])}')\n"
        "        return tensor\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    manifest = transform_manifest(_two_groups({"Blocks:Even": {}}), "Predictor", extension=True)[0]
    assert manifest["reads"][0]["preprocessing"][0]["op"] == "extension"


def test_a_class_of_the_app_s_swapping_a_second_input_s_axes_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A swap of two axes keeps a cube's extents: the class is probed on extents that differ along each axis.
    (tmp_path / "SwappedGrid.py").write_text(
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Swap(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def transform_shape(self, group_src, name, shape, cache_attribute):\n"
        "        return [shape[1], shape[0], *shape[2:]]\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor.transpose(1, 2)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ConfigError, match="off the grid"):
        transform_manifest(_two_groups({"SwappedGrid:Swap": {}}), "Predictor", extension=True)


def test_a_class_of_the_app_s_shifting_a_second_input_by_its_spacing_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A shift by the spacing less one voxel's millimetre is none on a grid of unit spacing: the class is probed on
    # another grid too.
    (tmp_path / "ShiftedGrid.py").write_text(
        "import numpy as np\n"
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Shift(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def transform_shape(self, group_src, name, shape, cache_attribute):\n"
        "        spacing = cache_attribute.get_np_array('Spacing')\n"
        "        cache_attribute['Origin'] = cache_attribute.get_np_array('Origin') + spacing - 1\n"
        "        return shape\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ConfigError, match="off the grid"):
        transform_manifest(_two_groups({"ShiftedGrid:Shift": {}}), "Predictor", extension=True)


def test_a_class_of_the_app_s_straightening_a_second_input_s_grid_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A class that sets the direction to the identity keeps an identity grid: it is probed on an oblique one too.
    (tmp_path / "StraightGrid.py").write_text(
        "import numpy as np\n"
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Straight(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def transform_shape(self, group_src, name, shape, cache_attribute):\n"
        "        cache_attribute['Direction'] = np.eye(len(shape)).flatten()\n"
        "        return shape\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ConfigError, match="off the grid"):
        transform_manifest(_two_groups({"StraightGrid:Straight": {}}), "Predictor", extension=True)


def test_a_group_a_second_input_s_chain_reads_beside_the_case_is_read_too() -> None:
    # The runtime reads the entries a program names: a masked bound of the second group's chain reads its mask, which
    # konfai hands the stage as stored, as the first group's chain's own.
    second = {"Clip": {"min_value": "min", "max_value": "max", "mask": "BODY"}}
    manifest, _folds, _per_copy = transform_manifest(_two_groups(second), "Predictor")
    assert manifest["reads"][0]["preprocessing"][0]["mask_group"] == "BODY"
    assert manifest["reads"][1:] == [{"group": "BODY", "channels": 1, "preprocessing": []}]


def test_a_test_time_copy_is_drawn_on_every_group_the_model_reads() -> None:
    # konfai shares one draw across the groups of a case, so a copy flips them all; only the first is
    # undone, since the output chain undoes that group's chain alone.
    manifest = {**_one_model()[0]["manifest"], "reads": [{"group": "MASK", "channels": 1, "preprocessing": []}]}
    program = assemble_program(
        [{"id": "m", "manifest": manifest}], reduce="mean", passes=[[], [{"op": "flip", "axes": [0, 1]}]]
    )
    copy = program["steps"][1]["manifest"]
    assert copy["preprocessing"][-1] == {"op": "flip", "axes": [0, 1], "id": "__tta_0"}
    assert copy["reads"] == [{"group": "MASK", "channels": 1, "preprocessing": [{"op": "flip", "axes": [0, 1]}]}]
    assert [step["op"] for step in copy["postprocessing"]] == ["inverse"]
    assert program["steps"][1]["in"] == ["input", "MASK"]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_a_quarter_turn_draw_is_the_permute_and_flip_konfai_applies(seed: int) -> None:
    # konfai reads a quarter turn as a signed permutation and applies it with `apply_remap`, a permute
    # then a flip; the compiler emits those two ops, so a copy is the volume konfai would have drawn.
    import random

    import torch
    from konfai.data.augmentation.spatial import (
        SIGNED_PERMUTATION_ATOL_FLOAT32,
        _rotation_3d_matrix,
    )
    from konfai.data.geometry import apply_remap, signed_permutation
    from konfai.portable import _quarter_turn

    rng = random.Random(seed)
    ops = _quarter_turn(random.Random(seed), 3)
    # The same draw konfai would make from this generator, read as konfai reads it.
    turns = [rng.choice([90.0, 180.0, 270.0]) for _ in range(3)]
    matrix = _rotation_3d_matrix(torch.deg2rad(torch.tensor(turns)))
    remap = signed_permutation(matrix[:-1, :-1], SIGNED_PERMUTATION_ATOL_FLOAT32)
    volume = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    theirs = apply_remap(volume, remap)

    ours = volume
    for op in ops:
        if op["op"] == "permute":
            ours = ours.permute([0, *(1 + axis for axis in op["axes"])])
        else:
            ours = ours.flip([1 + axis for axis in op["axes"]])
    assert torch.equal(ours.contiguous(), theirs.contiguous())


def test_a_permuted_test_time_pass_is_permuted_back_by_the_runtime_inverse() -> None:
    program = assemble_program(_one_model(), reduce="mean", passes=[[], [{"op": "permute", "axes": [1, 2, 0]}]])
    copy = program["steps"][1]["manifest"]
    assert copy["preprocessing"][-1] == {"op": "permute", "axes": [1, 2, 0], "id": "__tta_0"}
    assert copy["postprocessing"][0] == {"op": "inverse", "of": "__tta_0"}


def test_the_identity_pass_alone_is_the_plain_program() -> None:
    assert assemble_program(_one_model(), reduce="mean", passes=[[]]) == assemble_program(_one_model(), reduce="mean")


def test_a_label_merge_takes_no_test_time_passes() -> None:
    with pytest.raises(ConfigError, match="no meaning across flips"):
        assemble_program(_one_model(), reduce="merge_labels", passes=[[], [1]])


def _prediction(
    transforms: dict, outputs: dict, combine: str = "Mean", flips: int = 0, draws: dict | list | None = None
) -> dict:
    """A Predictor section; ``flips`` test-time copies, each drawn by ``draws`` (a Flip of every axis unnamed)."""
    chain = {"TensorCast": {"dtype": "float32", "inverse": False}, **transforms}
    dataset = _input_group(chain)
    if flips:
        draws = {"Flip": {"f_prob": [1, 1]}} if draws is None else draws
        dataset["augmentations"] = {"T": {"nb": flips, "data_augmentations": draws}}
    return {
        "Predictor": {
            "Model": {"classpath": "Probe"},
            "Dataset": dataset,
            "outputs_dataset": {"H": {"OutputDataset": outputs}},
            "combine": combine,
        }
    }


_MASK = {"Mask": {"path": "MASK", "value_outside": -1024}}
_STACK = {"InferenceStack": {"mode": "mean"}}


@pytest.mark.parametrize(
    ("transforms", "outputs", "stage"),
    [
        # A mask nothing applies: the program would write the unmasked prediction.
        ({}, {"before_reduction_transforms": _MASK}, "Mask"),
        # A list-spelled chain binds a repeated stage under an occurrence key.
        ({"Mask#2": _MASK["Mask"]}, {}, "Mask#2"),
        # konfai folds the copies at a stack that opens the after-reduction chain, and nowhere else.
        ({}, {"after_reduction_transforms": {"Softmax": {"dim": 0}, **_STACK}}, "InferenceStack"),
        ({}, {"final_transforms": _STACK}, "InferenceStack"),
        # konfai merges the models' label maps at a MergeLabels that opens the after-reduction chain.
        ({}, {"after_reduction_transforms": {"Softmax": {"dim": 0}, "MergeLabels": {}}}, "MergeLabels"),
        ({}, {"final_transforms": {"MergeLabels": {}}}, "MergeLabels"),
        ({"KonfAIInference": {"repo_id": "R/S", "model_name": "body"}}, {}, "KonfAIInference"),
        ({"Dilate": {"dilate": 5}}, {}, "Dilate"),
    ],
)
def test_a_program_stage_nothing_builds_is_refused_by_name(transforms: dict, outputs: dict, stage: str) -> None:
    with pytest.raises(ConfigError, match=f"transform '{stage}' compiles to a program step"):
        transform_manifest(_prediction(transforms, outputs), "Predictor")


def test_a_prediction_stage_that_alters_no_value_compiles_to_no_step() -> None:
    clip = {"Clip": {"min_value": 0, "max_value": 1}}
    softmax = {"Softmax": {"dim": 0}}
    statistics = {"Statistics": {}, "konfai.data.transform:Statistics": {}}
    plain = _prediction(clip, {"before_reduction_transforms": softmax})
    recorded = _prediction(
        {**clip, **statistics},
        {
            "before_reduction_transforms": {**statistics, **softmax},
            "after_reduction_transforms": statistics,
            "final_transforms": statistics,
        },
    )
    assert transform_manifest(recorded, "Predictor") == transform_manifest(plain, "Predictor")


def test_a_prediction_save_writes_no_cache() -> None:
    # In the input transforms konfai writes what reaches a Save (or a Write) and, on a later run, reads that
    # entry in place of the stages before it; the program writes nothing and runs them again.
    # The output chains call a Save directly, which hands its input back and writes nothing, as the program.
    clip = {"Clip": {"min_value": 0, "max_value": 1}}
    softmax = {"Softmax": {"dim": 0}}
    saves = {"Save": {"dataset": "./Dataset:mha"}, "konfai.data.transform:Write": {"dataset": "./Out:nii.gz"}}
    plain = _prediction(clip, {"before_reduction_transforms": softmax})
    saved = _prediction(
        {**saves, **clip},
        {
            "before_reduction_transforms": {**softmax, **saves},
            "after_reduction_transforms": saves,
            "final_transforms": saves,
        },
    )
    assert transform_manifest(saved, "Predictor") == transform_manifest(plain, "Predictor")


@pytest.mark.parametrize("spelling", [None, "None"])
@pytest.mark.parametrize(
    "chain", ["transforms", "before_reduction_transforms", "after_reduction_transforms", "final_transforms"]
)
def test_a_prediction_stage_written_with_no_parameters_is_refused_by_name(chain: str, spelling: str | None) -> None:
    # konfai binds a bare `Reduce:` key to its defaults and runs it, so it alters the prediction.
    stage = {"Reduce": spelling}
    config = _prediction(stage, {}) if chain == "transforms" else _prediction({}, {chain: stage})
    with pytest.raises(ConfigError, match="transform 'Reduce' has no portable runtime op"):
        transform_manifest(config, "Predictor")


def test_the_program_stages_a_caller_builds_are_left_to_it() -> None:
    before = {"UnNormalize": {"min_value": -1024, "max_value": 3071}, "TensorCast": {"dtype": "int16"}, **_MASK}
    outputs = {"before_reduction_transforms": before, "after_reduction_transforms": _STACK, "reduction": "Concat"}
    config = _prediction({}, outputs, combine="Concat", flips=1)
    manifest, _, per_copy = transform_manifest(config, "Predictor", masked=True)
    # The stack casts its average of the int16 copies back to int16.
    assert [step["op"] for step in manifest["postprocessing"]] == ["unnormalize", "cast", "cast"]
    assert per_copy == 2
    assert derive_reduction(config, "Predictor") == "mean"
    config["Predictor"]["outputs_dataset"]["H"]["OutputDataset"]["after_reduction_transforms"] = {
        "InferenceStack": {"mode": "median"}
    }
    assert derive_reduction(config, "Predictor") == "median_lower"
    # A stack past the first after-reduction stage folds nothing the program can name.
    later = _prediction({}, {"after_reduction_transforms": {"Softmax": {"dim": 0}, **_STACK}})
    assert derive_reduction(later, "Predictor") is None


def _label_merge(resample: dict | None = None) -> dict:
    """An ensemble of disjoint tasks: each model decides a uint8 class on the canonical 1.5 mm grid, the models
    concatenate, MergeLabels tiles their classes and the input chain is undone on the merged map."""
    transforms = {
        "Canonical": {"inverse": True},
        "Resample": {"spacing": [1.5, 1.5, 1.5], "inverse": True, **(resample or {})},
    }
    outputs = {
        "before_reduction_transforms": _CLASS_UINT8,
        "after_reduction_transforms": {"MergeLabels": {}},
        "reduction": "Concat",
    }
    return _prediction(transforms, outputs, combine="Concat")


def _merged(config: dict, classes: list[int]) -> dict:
    manifest, _, per_copy = transform_manifest(config, "Predictor")
    models = [{"id": f"m{i}", "manifest": {**manifest, "output": {"channels": c}}} for i, c in enumerate(classes)]
    return assemble_program(models, reduce=derive_reduction(config, "Predictor"), per_copy=per_copy)


def test_a_label_merge_merges_each_models_classes_where_the_undone_stages_pick_the_same_voxels() -> None:
    config = _label_merge()
    manifest, _, per_copy = transform_manifest(config, "Predictor")
    # Undoing a nearest resample and a reorientation moves voxels without changing one, in every model alike.
    assert manifest["postprocessing"][per_copy:] == [
        {"op": "inverse", "of": "Resample", "interpolation": "nearest"},
        {"op": "inverse", "of": "Canonical"},
    ]
    program = _merged(config, [25, 27])
    assert program["steps"][-1] == {"op": "merge_labels", "in": ["m0", "m1"], "out": "output", "classes": [25, 27]}
    assert [step["manifest"] for step in program["steps"][:2]] == [
        {**manifest, "output": {"channels": c}} for c in (25, 27)
    ]


def test_a_label_merge_that_a_later_stage_reads_the_values_of_is_refused_by_name() -> None:
    # konfai blends the merged labels linearly back onto the input grid; the models' maps blend differently.
    linear = _label_merge({"interpolation": "linear"})
    with pytest.raises(ConfigError, match="'Resample' runs once on konfai's merged label map"):
        _merged(linear, [25, 27])
    # One model: konfai's merge hands its map back.
    assert [step["out"] for step in _merged(linear, [25])["steps"]] == ["output"]


_MEDIAN_STACK = {"InferenceStack": {"mode": "median"}}


@pytest.mark.parametrize(
    ("outputs", "combine", "flips", "folded"),
    [
        # Mean widens the int16 copy to float32, and the runtime runs no reduce op over a single copy.
        ({}, "Mean", 0, "float32"),
        # konfai's stack casts its average of the int16 copies back to int16; the runtime's mean returns float32.
        ({"after_reduction_transforms": _STACK, "reduction": "Concat"}, "Concat", 1, "int16"),
        # Vote and a median stack keep the copies' dtype, as the runtime's reduce ops do.
        ({"reduction": "Vote"}, "Mean", 1, None),
        ({"after_reduction_transforms": _MEDIAN_STACK, "reduction": "Concat"}, "Concat", 1, None),
    ],
)
def test_the_dtype_integer_copies_fold_into_is_cast_where_they_fold(
    outputs: dict, combine: str, flips: int, folded: str | None
) -> None:
    transforms = {
        "Resample": {"spacing": [1.5, 1.5, 1.5], "inverse": True},
        "Padding": {"padding": [1, 1, 1, 1, 1, 1], "inverse": True},
    }
    before = {"before_reduction_transforms": {"TensorCast": {"dtype": "int16"}}}
    config = _prediction(transforms, {**before, **outputs}, combine, flips)
    manifest, _, per_copy = transform_manifest(config, "Predictor")
    cast = [{"op": "cast", "dtype": folded}] if folded else []
    # Past the pad's inverse, which a program leaves per copy.
    assert manifest["postprocessing"][per_copy:] == [
        {"op": "inverse", "of": "Padding"},
        *cast,
        {"op": "inverse", "of": "Resample", "interpolation": "linear"},
    ]


@pytest.mark.parametrize(
    ("combine", "copies", "flips"),
    [
        # konfai's stack reads the channel axis of the models' mean.
        ("Mean", "Concat", 0),
        # konfai's stack reads the model axis of the copies' mean.
        ("Concat", "Mean", 1),
    ],
)
def test_a_stack_the_copies_reach_unstacked_is_refused(combine: str, copies: str, flips: int) -> None:
    config = _prediction({}, {"after_reduction_transforms": _STACK, "reduction": copies}, combine, flips)
    for compile_stack in (derive_reduction, transform_manifest):
        with pytest.raises(ConfigError, match="transform 'InferenceStack' folds the first axis"):
            compile_stack(config, "Predictor")


@pytest.mark.parametrize("flips", [0, 1])
@pytest.mark.parametrize(
    ("combine", "outputs", "alone"),
    [
        # konfai takes the median or the vote of each batch of patches, before the overlap blend.
        ("Median", {}, "median"),
        ("Vote", {}, "vote"),
        # konfai's MergeLabels reads the per-model layout only a concatenation records.
        ("Mean", {"after_reduction_transforms": {"MergeLabels": {}}}, "merge_labels"),
    ],
)
def test_models_konfai_folds_before_the_blend_are_refused(combine: str, outputs: dict, alone: str, flips: int) -> None:
    config = _prediction({}, {**outputs, "reduction": combine}, combine, flips)
    passes = tta_passes(config, "Predictor")
    with pytest.raises(ConfigError, match=f"2 models combine by '{combine}'"):
        program_reduction(config, "Predictor", 2, passes)
    # One model: konfai's combine hands its output back.
    assert program_reduction(config, "Predictor", 1, passes) == alone


@pytest.mark.parametrize("copies", ["Mean", "Median"])
def test_models_konfai_averages_before_a_per_copy_chain_fold_first(copies: str) -> None:
    # konfai applies the before-reduction chain to the models' mean, once per copy; the copies then fold by
    # the output's reduction.
    softmax = {"before_reduction_transforms": {"Softmax": {"dim": 0}}, "reduction": copies}
    config = _prediction({}, softmax, "Mean", flips=1)
    passes = tta_passes(config, "Predictor")
    assert models_fold_first(config, "Predictor", 2)
    assert program_reduction(config, "Predictor", 2, passes) == copies.lower()
    # Without a per-copy chain the one mean over every copy is the same fold.
    assert not models_fold_first(_prediction({}, {}, "Mean", flips=1), "Predictor", 2)
    assert not models_fold_first(config, "Predictor", 1)


def test_models_folded_first_into_one_copy_keep_their_members() -> None:
    # konfai averages the models before a per-copy cast; on one pass that mean is the whole program, and the
    # members a runtime selects by are the models' buffers, not the fold's.
    config = _prediction({}, {"before_reduction_transforms": {"TensorCast": {"dtype": "float32"}}}, "Mean", 0)
    passes = tta_passes(config, "Predictor")
    manifest, _, per_copy = transform_manifest(config, "Predictor")
    program = assemble_program(
        [{"id": f"m{i}", "manifest": manifest} for i in range(2)],
        reduce=program_reduction(config, "Predictor", 2, passes),
        passes=passes,
        per_copy=per_copy,
        fold_models=models_fold_first(config, "Predictor", 2),
        checkpoints=["fold_0.pt", "fold_1.pt"],
    )
    assert [step["out"] for step in program["steps"]] == ["m0", "m1", "mean__p0", "output"]
    assert program["members"] == {
        "m0": {"checkpoint": "fold_0.pt", "pass": 0},
        "m1": {"checkpoint": "fold_1.pt", "pass": 0},
    }


def test_a_per_copy_chain_folded_on_the_models_average_runs_as_body_ops_or_is_refused_by_name() -> None:
    # konfai runs the before-reduction chain on the models' average; a program runs a cast, a softmax or an argmax
    # on a buffer and nothing else, so each such step takes its body form (a softmax names no axis there) and any
    # other stage is refused before export, rather than copied into a program the schema rejects.
    from konfai.portable_schema import program_schema

    lenient = _without_required(program_schema())
    manifest = {
        **_one_model()[0]["manifest"],
        "postprocessing": [{"op": "softmax", "dim": 0}, {"op": "cast", "dtype": "uint8"}],
    }
    models = [{"id": name, "manifest": manifest} for name in ("a", "b")]
    program = assemble_program(models, reduce="mean", per_copy=2, fold_models=True)
    _meets(program, "program", lenient)
    assert [step for step in program["steps"] if "op" in step][:3] == [
        {"op": "mean", "in": ["a", "b"], "out": "mean__p0"},
        {"op": "softmax", "in": ["mean__p0"], "out": "mean__p0__0"},
        {"op": "cast", "dtype": "uint8", "in": ["mean__p0__0"], "out": "output"},
    ]
    unnormalize = {**manifest, "postprocessing": [{"op": "unnormalize", "min_value": -1.0, "max_value": 1.0}]}
    with pytest.raises(ConfigError, match="'unnormalize'"):
        assemble_program([{**m, "manifest": unnormalize} for m in models], reduce="mean", per_copy=1, fold_models=True)


def test_a_cubic_resample_runs_through_a_whole_program_the_schema_admits() -> None:
    # The compiler emits `cubic` on a resample and its inverse, and `resample_cubic` once that inverse is hoisted
    # past the reduction: the schema admits each where it admits `linear`.
    from konfai.portable_schema import program_schema

    lenient = _without_required(program_schema())
    manifest = _undone_resample({}, {"interpolation": "cubic"})
    _meets(manifest, "manifest", lenient)
    assert manifest["postprocessing"] == [{"op": "inverse", "of": "Resample", "interpolation": "cubic"}]
    program = assemble_program([{"id": name, "manifest": manifest} for name in ("a", "b")], reduce="mean")
    _meets(program, "program", lenient)
    assert program["steps"][-1] == {"op": "resample_cubic", "in": ["r", "input"], "out": "output"}


def test_one_model_folds_its_test_time_copies_by_their_own_reduction() -> None:
    # konfai's combine hands one model's output back, and the copies fold by the output's `reduction`: that
    # fold is the program's only one, whatever `combine` says.
    voted = _prediction({}, {"reduction": "Vote"}, flips=1)
    assert program_reduction(voted, "Predictor", 1, tta_passes(voted, "Predictor")) == "vote"
    assert program_reduction(voted, "Predictor", 1, [[]]) == "mean"


def test_models_a_program_may_fold_after_the_blend_keep_their_reduction() -> None:
    # A mean commutes with the blend; the module-qualified spelling names the same class.
    mean = _prediction({}, {"reduction": "konfai.data.reduction:Mean"}, "konfai.data.reduction:Mean", flips=1)
    assert program_reduction(mean, "Predictor", 2, tta_passes(mean, "Predictor")) == "mean"
    # konfai splits a concatenation per model before the before-reduction transforms.
    softmax = {"before_reduction_transforms": {"Softmax": {"dim": 0}}}
    assert program_reduction(_prediction({}, softmax, "Concat"), "Predictor", 2, [[]]) == "concat"


def test_an_occurrence_key_names_its_stage() -> None:
    config = _prediction(
        {"Clip#2": {"min_value": 0, "max_value": 1}}, {}, flips=1, draws={"Flip#2": {"f_prob": [1, 1]}}
    )
    manifest, folds, _ = transform_manifest(config, "Predictor")
    assert [step["op"] for step in manifest["preprocessing"]] == ["cast", "clip"]
    assert not folds
    assert tta_passes(config, "Predictor") == [[], [{"op": "flip", "axes": [0]}]]


def test_a_prediction_program_applies_no_mask(tmp_path: Path) -> None:
    # compile_prediction exports with this default: no assembler applies a before-reduction Mask after it.
    config = _prediction({}, {"before_reduction_transforms": _MASK})
    with pytest.raises(ConfigError, match="transform 'Mask'"):
        export_checkpoints(config, "Predictor", [None], tmp_path, ["model"], patch_size=[4, 4], in_channels=1)


def _tiny_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, classpath: str, head: str, placement: str | None = None
) -> Path:
    """A Prediction.yml over two 16 x 16 cases, the model ``classpath`` writing ``head`` as a label map."""
    import numpy as np
    from konfai.utils.dataset import Attribute, Dataset

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "assets" / "Workflows"))
    monkeypatch.chdir(tmp_path)
    for case in ("CASE_000", "CASE_001"):
        Dataset(str(tmp_path / "Dataset"), "mha").write("MR", case, np.ones((1, 1, 16, 16), np.float32), Attribute())
    group = {"transforms": "None", "patch_transforms": "None", "is_input": True}
    name = classpath.split(":")[-1]
    config = tmp_path / "Prediction.yml"
    config.write_text(
        json.dumps(
            {
                "Predictor": {
                    "Model": {"classpath": classpath, name: {"outputs_criterions": "None"}},
                    "autocast": True,
                    "Dataset": {
                        "groups_src": {"MR": {"groups_dest": {"MR": group}}},
                        "augmentations": "None",
                        "Patch": {"patch_size": [1, 16, 16], "overlap": 0}
                        | ({"placement": placement} if placement else {}),
                        "dataset_filenames": ["./Dataset:a:mha"],
                        "num_workers": 0,
                    },
                    "outputs_dataset": {
                        head: {
                            "OutputDataset": {
                                "before_reduction_transforms": "None",
                                "after_reduction_transforms": "None",
                                "final_transforms": {"Argmax": {"dim": 0}, "TensorCast": {"dtype": "uint8"}},
                                "dataset_filename": "Dataset:mha",
                                "group": "SEG",
                                "same_as_group": "MR:MR",
                            }
                        }
                    },
                }
            }
        )
    )
    return config


def test_a_class_of_the_app_s_before_a_folded_graph_is_run_again_in_the_dtype_it_is_handed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The graph a pointwise class folds into is exported on the channels the steps before it leave: a whole class
    # taking the argmax's int64 labels is run again in them to count its own.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    import torch
    from konfai.portable import compile_prediction

    config = _tiny_prediction(tmp_path, monkeypatch, "TinySeg:TinySegNet", "Head:Softmax")
    (tmp_path / "FoldedAfter.py").write_text(
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Odd(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor & 1\n\n"
        "class Double(Transform):\n"
        "    locality = LocalityKind.POINTWISE\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor * 2\n"
    )
    resolved = json.loads(config.read_text())
    resolved["Predictor"]["outputs_dataset"]["Head:Softmax"]["OutputDataset"]["final_transforms"] = {
        "Argmax": {"dim": 0},
        "FoldedAfter:Odd": {},
        "FoldedAfter:Double": {},
    }
    config.write_text(json.dumps(resolved))
    from TinySeg import TinySegNet

    checkpoint = tmp_path / "fold_0.pt"
    torch.save({"Model": TinySegNet().network_states()}, checkpoint)
    extension = tmp_path / "own.wasm"
    extension.write_bytes(b"\0asm")
    program = compile_prediction(
        config, [checkpoint], tmp_path / "Predictions", tmp_path / "out" / "prediction.json", extension=extension
    )
    (model,) = program["outputs"][0]["program"]["steps"]
    assert [step["op"] for step in model["manifest"]["postprocessing"]][-3:] == ["argmax", "extension", "onnx"]


def test_an_extension_beside_the_program_named_as_a_graph_is_refused_before_a_graph_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The export would write its model.onnx over the extension: refused first, the extension as it was.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    import torch
    from konfai.portable import compile_prediction

    config = _tiny_prediction(tmp_path, monkeypatch, "TinySeg:TinySegNet", "Head:Softmax")
    from TinySeg import TinySegNet

    checkpoint = tmp_path / "fold_0.pt"
    torch.save({"Model": TinySegNet().network_states()}, checkpoint)
    out = tmp_path / "out" / "prediction.json"
    out.parent.mkdir()
    extension = out.parent / "model.onnx"
    extension.write_bytes(b"\0asm")
    with pytest.raises(ConfigError, match=r"model\.onnx"):
        compile_prediction(config, [checkpoint], tmp_path / "Predictions", out, extension=extension)
    assert extension.read_bytes() == b"\0asm" and not out.exists()


def test_a_prediction_config_and_its_checkpoints_compile_to_a_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A real Prediction.yml over the tests' tiny segmentation model and two of its checkpoints: the predictor
    # builds, each checkpoint exports beside the program, and the autouse check holds the program to the schema.
    # Each model's manifest carries the Patch.placement konfai places its windows with, and its unrounded step.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    import torch
    from konfai.portable import compile_prediction

    config = _tiny_prediction(tmp_path, monkeypatch, "TinySeg:TinySegNet", "Head:Softmax", "end")
    from TinySeg import TinySegNet

    checkpoints = [tmp_path / f"fold_{index}.pt" for index in range(2)]
    for checkpoint in checkpoints:
        torch.save({"Model": TinySegNet().network_states()}, checkpoint)
    program = compile_prediction(config, checkpoints, tmp_path / "Predictions", tmp_path / "out" / "prediction.json")

    assert json.loads((tmp_path / "out" / "prediction.json").read_text()) == program
    assert program["models"] == {"model_0": "model_0.onnx", "model_1": "model_1.onnx"}
    assert all((tmp_path / "out" / name).is_file() for name in program["models"].values())
    assert program["cases"] == ["CASE_000", "CASE_001"] and program["input_group"] == "MR"
    # konfai's float16 forward is the program's to carry, for a runtime that has float16.
    assert program["autocast"] is True
    (output,) = program["outputs"]
    assert (output["write"]["group"], output["write"]["format"]) == ("SEG", "mha")
    # Both folds run on the input, fold by the models' Mean, then decide the class once.
    steps = output["program"]["steps"]
    assert [step.get("model") for step in steps[:2]] == ["model_0", "model_1"]
    # konfai blends the folds' float32 outputs in float16, which the runtime is told to do alike.
    assert {step["manifest"]["accumulation_dtype"] for step in steps[:2]} == {"float16"}
    assert [step["op"] for step in steps[2:]] == ["mean", "argmax", "cast"]
    assert [step["manifest"]["patch"]["placement"] for step in steps[:2]] == ["end"] * 2
    assert [step["manifest"]["patch"]["stride"] for step in steps[:2]] == [[16, 16]] * 2


def test_an_apps_own_fold_compiles_to_the_patch_combine_of_its_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The app's Fold is named by classpath: the program names the extension's patch combine by it, what it enters a
    # patch as is part of the model's graph, and the manifest states what it leaves.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    onnxruntime = pytest.importorskip("onnxruntime")
    import numpy as np
    import torch
    from konfai.portable import compile_prediction

    config = _tiny_prediction(tmp_path, monkeypatch, "TinySeg:TinySegNet", "Head:Softmax")
    from TinySeg import TinySegNet

    (tmp_path / "Fusion.py").write_text(
        "import torch\n"
        "from konfai.data.patching import Fold\n\n\n"
        "class Surest(Fold):\n"
        "    def enter(self, patch):\n"
        "        score, label = patch.max(0, keepdim=True)\n"
        "        return torch.cat((label.to(torch.uint8), (score * 255).round().to(torch.uint8)))\n\n"
        "    def fold(self, held, patch):\n"
        "        return torch.where(patch[1:] >= held[1:], patch, held)\n\n"
        "    def leave(self, held):\n"
        "        return held[:1]\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    resolved = json.loads(config.read_text())
    output = resolved["Predictor"]["outputs_dataset"]["Head:Softmax"]["OutputDataset"]
    output |= {"patch_combine": "Fusion:Surest", "final_transforms": "None"}
    config.write_text(json.dumps(resolved))
    checkpoint = tmp_path / "fold_0.pt"
    torch.save({"Model": TinySegNet().network_states()}, checkpoint)
    out = tmp_path / "out" / "prediction.json"

    with pytest.raises(KonfAIError, match="'Fusion:Surest' is the app's own"):
        compile_prediction(config, [checkpoint], tmp_path / "Predictions", out)
    extension = tmp_path / "fusion.wasm"
    extension.write_bytes(b"\0asm")
    # Two models: each graph ends on the head, and their mean is entered by a graph of its own.
    folds = [tmp_path / "fold_0.pt", tmp_path / "fold_1.pt"]
    torch.save({"Model": TinySegNet().network_states()}, folds[1])
    both = compile_prediction(
        config, folds, tmp_path / "Predictions", tmp_path / "two" / "prediction.json", extension=extension
    )
    assert both["models"] == {"model_0": "model_0.onnx", "model_1": "model_1.onnx", "model_enter": "model_enter.onnx"}
    averaged = both["outputs"][0]["program"]
    assert averaged["enter"] == {"model": "model_enter"} and len(averaged["averaged"]["model"]) == 2
    assert averaged["steps"][0]["manifest"]["output"]["channels"] == 2
    enter = onnxruntime.InferenceSession(str(tmp_path / "two" / "model_enter.onnx"), providers=["CPUExecutionProvider"])
    mean = np.stack([np.full((16, 16), 0.25, np.float32), np.full((16, 16), 0.75, np.float32)])[None]
    (entered,) = enter.run(None, {"input": mean})
    assert (entered[0, 0] == 1).all() and (entered[0, 1] == 191).all()
    program = compile_prediction(config, [checkpoint], tmp_path / "Predictions", out, extension=extension)

    assert program["extension"] == "fusion.wasm" and (out.parent / "fusion.wasm").read_bytes() == b"\0asm"
    # An extension already beside the program is the one it names: compiled again, nothing is copied over itself.
    again = compile_prediction(
        config, [checkpoint], tmp_path / "Predictions", out, extension=out.parent / "fusion.wasm"
    )
    assert again["extension"] == "fusion.wasm"
    (step,) = program["outputs"][0]["program"]["steps"]
    manifest = step["manifest"]
    assert (manifest["blend"], manifest["output"]["channels"]) == ("Fusion:Surest", 2)
    assert manifest["blended"] == {"channels": 1, "dtype": "uint8", "params": {}}
    session = onnxruntime.InferenceSession(str(out.parent / "model.onnx"), providers=["CPUExecutionProvider"])
    (entered,) = session.run(None, {"input": np.ones((1, 1, 16, 16), np.float32)})
    assert entered.dtype == np.float32 and entered.shape[1] == 2
    assert set(np.unique(entered[:, 0])) <= {0.0, 1.0} and entered[:, 1].max() <= 255


def test_an_apps_own_classes_compile_to_the_steps_its_extension_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A transform, a test-time draw and a reduction of the app's own, each named by classpath: with the app's
    # extension the program names them by it, built with the keys konfai binds them with; without it, a transform is
    # folded into the model's graph and the two others are refused.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    import torch
    from konfai.portable import compile_prediction

    config = _tiny_prediction(tmp_path, monkeypatch, "TinySeg:TinySegNet", "Head:Softmax")
    from TinySeg import TinySegNet

    (tmp_path / "Own.py").write_text(
        "import torch\n"
        "from konfai.data.augmentation import DataAugmentation\n"
        "from konfai.data.reduction import Reduction\n"
        "from konfai.data.transform import LocalityKind, PatchLocality, Transform, TransformInverse\n\n\n"
        "class Smooth(Transform):\n"
        "    def patch_locality(self, cache_attribute):\n"
        "        if 'Spacing' not in cache_attribute:\n"
        "            return PatchLocality(LocalityKind.WHOLE_VOLUME, reason='a halo in millimetres')\n"
        "        return PatchLocality(LocalityKind.HALO, halo=(1,))\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return torch.nn.functional.avg_pool3d(tensor.float()[None], 3, 1, 1, count_include_pad=False)[0]\n\n\n"
        "class Centered(Transform):\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor.float() - tensor.float().mean()\n\n\n"
        "class Scale(TransformInverse):\n"
        "    def __init__(self, scale: float = 2.0, inverse: bool = True) -> None:\n"
        "        super().__init__(inverse)\n"
        "        self.scale = scale\n\n"
        "    def patch_locality(self, cache_attribute):\n"
        "        return PatchLocality(LocalityKind.POINTWISE)\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor.float() * self.scale\n\n"
        "    def inverse(self, name, tensor, cache_attribute):\n"
        "        return tensor / self.scale\n\n\n"
        "class Gain(DataAugmentation):\n"
        "    locality = LocalityKind.POINTWISE\n\n"
        "    def __init__(self, g_std: float = 0.1, groups: list[str] | None = None) -> None:\n"
        "        super().__init__(groups)\n"
        "        self.g_std = g_std\n\n"
        "    def _state_init(self, index, shapes, caches_attribute):\n"
        "        return shapes\n\n"
        "    def _compute(self, name, index, a, tensor):\n"
        "        return tensor * (1 + self.g_std)\n\n"
        "    def _inverse(self, index, a, tensor):\n"
        "        return tensor\n\n\n"
        "class Largest(Reduction):\n"
        "    voxel_local = True\n\n"
        "    def __call__(self, tensors):\n"
        "        return torch.stack(tensors).amax(0)\n\n\n"
        "class Double(Transform):\n"
        "    locality = LocalityKind.POINTWISE\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor * 2\n\n\n"
        "class Strongest(Transform):\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        if tensor.shape[0] != 2:\n"
        "            raise ValueError('Strongest reads the head and its two channels')\n"
        "        return tensor[tensor.flatten(1).sum(1).argmax()][None]\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    resolved = json.loads(config.read_text())
    predictor = resolved["Predictor"]
    # konfai undoes an input transform once the copies are folded, which a program does for a resample alone.
    # A class reads the voxel it writes, its neighbours as far as the halo it declares for a grid, or, declaring
    # nothing, the whole volume.
    own = {"Own:Scale": {"scale": 4.0, "inverse": False}, "Own:Smooth": {}, "Own:Centered": {}}
    predictor["Dataset"]["groups_src"]["MR"]["groups_dest"]["MR"]["transforms"] = own
    predictor["Dataset"]["augmentations"] = {"TTA": {"data_augmentations": {"Own:Gain": {"g_std": 0.5}}, "nb": 2}}
    # A class is compiled on what its place in the chain hands it, the head's two channels here, and one reading the
    # whole volume may write fewer.
    predictor["outputs_dataset"]["Head:Softmax"]["OutputDataset"] |= {
        "reduction": "Own:Largest",
        # A pointwise class a graph carries is folded into one, extension or not.
        "before_reduction_transforms": {"Own:Strongest": {}, "Own:Double": {}},
    }
    config.write_text(json.dumps(resolved))
    checkpoint = tmp_path / "fold_0.pt"
    torch.save({"Model": TinySegNet().network_states()}, checkpoint)

    with pytest.raises(KonfAIError, match="'Own:Gain'"):
        compile_prediction(config, [checkpoint], tmp_path / "Predictions", tmp_path / "bare" / "prediction.json")
    extension = tmp_path / "own.wasm"
    extension.write_bytes(b"\0asm")
    out = tmp_path / "out" / "prediction.json"
    program = compile_prediction(config, [checkpoint], tmp_path / "Predictions", out, extension=extension)

    assert program["extension"] == "own.wasm"
    steps = program["outputs"][0]["program"]["steps"]
    # The identity and two drawn copies, then the app's own fold of the three.
    models = [step["manifest"] for step in steps if "manifest" in step]
    assert len(models) == 3
    # The stored dtype only the run knows, Scale returns float32 of it; the classes after it are handed float32,
    # which they return.
    scale = {"op": "extension", "class": "Own:Scale", "params": own["Own:Scale"], "dtype": "float32"}
    smooth = {"op": "extension", "class": "Own:Smooth", "params": {}, "halo": 1}
    centered = {"op": "extension", "class": "Own:Centered", "params": {}, "whole": True}
    assert models[0]["preprocessing"] == [scale, smooth, centered]
    # A copy's draw closes its preprocessing with a seed of its own, and is undone first.
    draws = [manifest["preprocessing"][-1] for manifest in models[1:]]
    assert [draw["class"] for draw in draws] == ["Own:Gain"] * 2 and all(draw["id"] == "__tta_0" for draw in draws)
    # The keys konfai binds the draw with, the defaults it resolved among them, and the copy's seed.
    assert draws[0]["params"].keys() == {"g_std", "groups", "seed"} and draws[0]["params"] != draws[1]["params"]
    assert models[1]["postprocessing"][0] == {"op": "inverse", "of": "__tta_0"}
    (fold,) = [step for step in steps if step.get("op") == "extension_reduce"]
    assert fold["class"] == "Own:Largest" and len(fold["in"]) == 3
    strongest = {"op": "extension", "class": "Own:Strongest", "params": {}, "whole": True}
    assert all(strongest in model["postprocessing"] for model in models)
    assert '"Own:Double"' not in json.dumps(program)
    assert all(model["postprocessing"][-1]["op"] == "onnx" for model in models)


def test_a_final_chain_reading_groups_of_the_case_closes_the_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # konfai's final chain reads groups of the case as stored: a class of the app's (read_companion) and a Mask. The
    # program runs that end of the chain on its buffers, the first group being its input and any other one read
    # with the case.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    import numpy as np
    import torch
    from konfai.portable import compile_prediction
    from konfai.utils.dataset import Attribute, Dataset

    config = _tiny_prediction(tmp_path, monkeypatch, "TinySeg:TinySegNet", "Head:Softmax")
    from TinySeg import TinySegNet

    for case in ("CASE_000", "CASE_001"):
        Dataset(str(tmp_path / "Dataset"), "mha").write("BODY", case, np.ones((1, 1, 16, 16), np.uint8), Attribute())
    (tmp_path / "OwnGroup.py").write_text(
        "from konfai.data.transform import Transform\n\n\n"
        "class ScaleOnto(Transform):\n"
        "    def __init__(self, reference_group: str = 'MR') -> None:\n"
        "        super().__init__()\n"
        "        self.reference_group = reference_group\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        reference = self.read_companion(self.reference_group, name)\n"
        "        return tensor.float() * float((tensor.float() * reference).sum() + 1)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    resolved = json.loads(config.read_text())
    final = {
        "OwnGroup:ScaleOnto": {"reference_group": "MR"},
        "Mask": {"path": "BODY", "value_outside": -1},
        "Argmax": {"dim": 0},
    }
    resolved["Predictor"]["outputs_dataset"]["Head:Softmax"]["OutputDataset"]["final_transforms"] = final
    config.write_text(json.dumps(resolved))
    checkpoint = tmp_path / "fold_0.pt"
    torch.save({"Model": TinySegNet().network_states()}, checkpoint)
    extension = tmp_path / "own.wasm"
    extension.write_bytes(b"\0asm")

    program = compile_prediction(
        config, [checkpoint], tmp_path / "Predictions", tmp_path / "out" / "prediction.json", extension=extension
    )

    assert program["input_group"] == ["MR", "BODY"]
    model, scale, mask, argmax = program["outputs"][0]["program"]["steps"]
    assert not any("extra" in step for step in model["manifest"]["postprocessing"])
    assert scale == {
        "op": "extension",
        "class": "OwnGroup:ScaleOnto",
        "params": {"reference_group": "MR"},
        "whole": True,
        "groups": ["MR"],
        "in": [model["out"], "input"],
        "out": "t0",
    }
    assert mask == {"op": "mask", "value_outside": -1.0, "in": ["t0", "BODY"], "out": "t1"}
    assert argmax == {"op": "argmax", "in": ["t1"], "out": "output"}


def test_an_own_class_that_moves_the_grid_is_undone_by_the_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A class that reads the whole volume may leave another grid: its step is marked `whole`, and the output is
    # put back on the grid the class read by the inverse of that step.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    import torch
    from konfai.portable import compile_prediction

    config = _tiny_prediction(tmp_path, monkeypatch, "TinySeg:TinySegNet", "Head:Softmax")
    from TinySeg import TinySegNet

    (tmp_path / "OwnGrid.py").write_text(
        "import torch\n"
        "from konfai.data.transform import TransformInverse\n\n\n"
        "class Trim(TransformInverse):\n"
        "    def __init__(self, margin: int = 1, inverse: bool = True) -> None:\n"
        "        super().__init__(inverse)\n"
        "        self.margin = margin\n\n"
        "    def transform_shape(self, group_src, name, shape, cache_attribute):\n"
        "        return [*shape[:-2], *(extent - 2 * self.margin for extent in shape[-2:])]\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor[..., self.margin : -self.margin, self.margin : -self.margin]\n\n"
        "    def inverse(self, name, tensor, cache_attribute):\n"
        "        return torch.nn.functional.pad(tensor, [self.margin] * 4)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    resolved = json.loads(config.read_text())
    transforms = {"OwnGrid:Trim": {"margin": 2, "inverse": True}}
    resolved["Predictor"]["Dataset"]["groups_src"]["MR"]["groups_dest"]["MR"]["transforms"] = transforms
    config.write_text(json.dumps(resolved))
    checkpoint = tmp_path / "fold_0.pt"
    torch.save({"Model": TinySegNet().network_states()}, checkpoint)
    extension = tmp_path / "own.wasm"
    extension.write_bytes(b"\0asm")
    program = compile_prediction(
        config, [checkpoint], tmp_path / "Predictions", tmp_path / "out" / "prediction.json", extension=extension
    )

    manifest = program["outputs"][0]["program"]["steps"][0]["manifest"]
    trim = {"op": "extension", "class": "OwnGrid:Trim", "params": transforms["OwnGrid:Trim"], "whole": True}
    assert manifest["preprocessing"] == [trim | {"id": "OwnGrid:Trim"}]
    assert {"op": "inverse", "of": "OwnGrid:Trim"} in manifest["postprocessing"]


def test_a_crop_on_a_needed_apps_group_compiles_to_a_read_beside_the_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The box comes from another app's output, which no dataset of the compile holds: the plan measures no box,
    # and the program reads that group beside the case, where the runtime measures it.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    import torch
    from konfai.data.transform import Crop
    from konfai.portable import compile_prediction

    config = _tiny_prediction(tmp_path, monkeypatch, "TinySeg:TinySegNet", "Head:Softmax")
    crop = {"source": "BODY", "labels": [2, 7], "margin_mm": 10.0, "fill": 0.0, "empty": "fill"}
    resolved = json.loads(config.read_text())
    resolved["Predictor"]["Dataset"]["groups_src"]["MR"]["groups_dest"]["MR"]["transforms"] = {"Crop": crop}
    config.write_text(json.dumps(resolved))
    from TinySeg import TinySegNet

    torch.save({"Model": TinySegNet().network_states()}, tmp_path / "model.pt")
    program = compile_prediction(
        config, [tmp_path / "model.pt"], tmp_path / "Predictions", tmp_path / "out" / "prediction.json"
    )
    manifest = program["outputs"][0]["program"]["steps"][0]["manifest"]
    step = {"op": "crop", "source_group": "BODY", "labels": [2, 7], "margin_mm": 10.0, "fill": 0.0, "empty": "fill"}
    assert manifest["preprocessing"] == [step | {"id": "Crop"}]
    assert manifest["reads"] == [{"group": "BODY", "channels": 1, "preprocessing": []}]
    assert program["input_group"] == ["MR", "BODY"]
    assert Crop.plan_without_source is False


def test_a_pointwise_output_transform_with_no_op_compiles_to_a_graph_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A transform of the app's own, per voxel, after the blend: exported beside the model as one graph, which the
    # program runs where the chain places it, on the channels the steps before it leave.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    import numpy as np
    import torch
    from konfai.portable import compile_prediction

    config = _tiny_prediction(tmp_path, monkeypatch, "TinySeg:TinySegNet", "Head:Softmax")
    (tmp_path / "Post.py").write_text(
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Hottest(Transform):\n"
        "    locality = LocalityKind.POINTWISE\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return (tensor * 10).floor().to(tensor.dtype)[:1]\n"
    )
    resolved = json.loads(config.read_text())
    resolved["Predictor"]["outputs_dataset"]["Head:Softmax"]["OutputDataset"]["after_reduction_transforms"] = {
        "KeepLargestComponent": {"mask": 1},
        "Post:Hottest": {},
    }
    config.write_text(json.dumps(resolved))
    from TinySeg import TinySegNet

    torch.save({"Model": TinySegNet().network_states()}, tmp_path / "model.pt")
    program = compile_prediction(
        config, [tmp_path / "model.pt"], tmp_path / "Predictions", tmp_path / "out" / "prediction.json"
    )
    post = program["outputs"][0]["program"]["steps"][0]["manifest"]["postprocessing"]
    # keep_largest_component drops its mask channel: the graph after it takes one channel of the head's two.
    assert post[0] == {"op": "keep_largest_component", "mask": 1}
    assert post[1] == {"op": "onnx", "model": "model_post_0.onnx", "channels": 1, "dtype": "float32"}
    assert [step["op"] for step in post[2:]] == ["argmax", "cast"]
    session = ort.InferenceSession(str(tmp_path / "out" / "model_post_0.onnx"), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape[1] == 1
    # On the case's three axes, which the output chain runs on, though the patch traced has two.
    blended = np.random.default_rng(0).random((1, 1, 3, 5, 7), dtype=np.float32)
    np.testing.assert_array_equal(session.run(None, {"input": blended})[0], np.floor(blended[:, :1] * 10))


def test_a_case_statistic_of_a_second_input_compiles_to_two_float64_graphs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The model reads the image and its CaseStatistic: the second read names the two graphs beside the model, which
    # onnxruntime runs on a block of another shape to what the statistic's own functions give: the class built with
    # the mapping's scale, which a float32 constant would move.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    import numpy as np
    import torch
    from konfai.portable import compile_prediction

    config = _tiny_prediction(tmp_path, monkeypatch, "TwoInputs:TwoInputNet", "Head:Softmax")
    resolved = json.loads(config.read_text())
    statistic = {
        "transforms": {"CaseStatistic": {"statistic": "oracle_support:ScaledRoughness", "scale": 0.9}},
        "is_input": True,
    }
    resolved["Predictor"]["Dataset"]["groups_src"]["MR"]["groups_dest"]["SIGMA"] = statistic
    config.write_text(json.dumps(resolved))
    program = compile_prediction(config, [], tmp_path / "Predictions", tmp_path / "out" / "prediction.json")
    reads = program["outputs"][0]["program"]["steps"][0]["manifest"]["reads"]
    step = {"op": "case_statistic", "halo": 1, "partial": "model_statistic_0_partial.onnx"}
    assert reads == [
        {"group": "MR", "channels": 1, "preprocessing": [step | {"finish": "model_statistic_0_finish.onnx"}]}
    ]
    from oracle_support import ScaledRoughness

    partial = ort.InferenceSession(str(tmp_path / "out" / step["partial"]), providers=["CPUExecutionProvider"])
    finish = ort.InferenceSession(str(tmp_path / "out" / "model_statistic_0_finish.onnx"))
    assert [(i.name, i.type) for i in partial.get_inputs()] == [("block", "tensor(double)")]
    assert [(i.name, i.type) for i in finish.get_outputs()] == [("statistic", "tensor(double)")]
    block = torch.from_numpy(np.random.default_rng(0).random((1, 7, 12, 9)))
    total = partial.run(None, {"block": block.numpy()})[0]
    np.testing.assert_allclose(total, ScaledRoughness().partial(block).numpy(), rtol=1e-12)
    value = finish.run(None, {"total": total})[0]
    assert value.shape == () and value == ScaledRoughness(0.9).finish(torch.from_numpy(total)).item()


def test_a_whole_class_of_the_app_s_is_probed_in_the_dtype_its_chain_hands_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Past an Argmax konfai hands the class int64 labels, which a bitwise test takes and a float32 volume does not.
    (tmp_path / "Labels.py").write_text(
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Odd(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor & 1\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config = _prediction({}, {"after_reduction_transforms": {"Argmax": {}, "Labels:Odd": {}}})
    manifest, _folds, _per_copy = transform_manifest(config, "Predictor", extension=True)
    assert [step["op"] for step in manifest["postprocessing"]][-2:] == ["argmax", "extension"]


def test_a_class_of_the_app_s_past_a_case_statistic_is_probed_in_float32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # konfai's CaseStatistic hands on float32 whatever it is handed: an int16 cast before it is no longer what the
    # class is handed.
    from konfai.portable import _probe_after, compile_pipeline

    (tmp_path / "StatisticMap.py").write_text(
        "import torch\n"
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Floats(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        if tensor.dtype != torch.float32 or tensor.shape[0] != 1:\n"
        "            raise ValueError(f'one float32 channel, not {tensor.dtype} {list(tensor.shape)}')\n"
        "        return tensor\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    statistic = {"op": "case_statistic", "halo": 1, "partial": "partial.onnx", "finish": "finish.onnx"}
    chain = {"TensorCast": {"dtype": "int16"}, "CaseStatistic": {"statistic": "x:Y"}, "StatisticMap:Floats": {}}
    steps, _, _ = compile_pipeline(
        chain, extension=True, case_statistic=lambda _params: statistic, probe=_probe_after(3, lambda: 1)
    )
    assert [step["op"] for step in steps] == ["cast", "case_statistic", "extension"]


def test_a_class_of_the_app_s_handed_a_known_dtype_states_what_it_returns_of_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Past an Argmax the class is handed int64 labels, the only dtype it takes, and returns uint8: its step says so.
    # In the input chain, where only the run knows the stored dtype, a class returning what it reads states none.
    (tmp_path / "KnownLabels.py").write_text(
        "import torch\n"
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Bytes(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        if tensor.dtype != torch.int64:\n"
        "            raise TypeError(f'labels, not {tensor.dtype}')\n"
        "        return tensor.to(torch.uint8)\n\n"
        "class Same(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor\n\n"
        "class Floats(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor.float()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config = _prediction(
        {"KnownLabels:Same": {}}, {"after_reduction_transforms": {"Argmax": {}, "KnownLabels:Bytes": {}}}
    )
    manifest, _folds, _per_copy = transform_manifest(config, "Predictor", extension=True)
    (step,) = [step for step in manifest["postprocessing"] if step.get("class") == "KnownLabels:Bytes"]
    assert step["dtype"] == "uint8"
    (same,) = [step for step in manifest["preprocessing"] if step.get("class") == "KnownLabels:Same"]
    assert "dtype" not in same
    # A second input's chain starts from its stored dtype too: a class returning float32 of any says so.
    manifest, _folds, _per_copy = transform_manifest(
        _two_groups({"KnownLabels:Floats": {}}), "Predictor", extension=True
    )
    assert manifest["reads"][0]["preprocessing"][0]["dtype"] == "float32"


@pytest.mark.parametrize("phase", ["after_reduction_transforms", "final_transforms"])
def test_a_class_of_the_app_s_past_the_reduction_is_run_again_in_the_dtype_the_reduction_leaves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    # The copies' labels averaged are float32 again: the class after the reduction, run again to count what it hands
    # the next one, in its phase or the final one, splits them into two channels as konfai's run does, which it would
    # not of int64 labels.
    (tmp_path / "Phases.py").write_text(
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Split(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor.repeat(2, *[1] * (tensor.dim() - 1)) if tensor.is_floating_point() else tensor\n\n"
        "class Pair(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        if tensor.shape[0] != 2:\n"
        "            raise ValueError(f'two channels, not {list(tensor.shape)}')\n"
        "        return tensor\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    outputs = {"before_reduction_transforms": {"Argmax": {}}, "after_reduction_transforms": {"Phases:Split": {}}}
    outputs[phase] = {**outputs.get(phase, {}), "Phases:Pair": {}}
    manifest, _folds, _per_copy = transform_manifest(_prediction({}, outputs), "Predictor", extension=True)
    assert [step["op"] for step in manifest["postprocessing"]][-2:] == ["extension", "extension"]


@pytest.mark.parametrize("phase", ["after_reduction_transforms", "final_transforms"])
def test_a_class_of_the_app_s_past_a_vote_is_run_again_in_the_labels_the_vote_keeps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    # A vote of the copies' labels keeps them int64: a class taking labels only, after it, is run again in them.
    (tmp_path / "VotedPhases.py").write_text(
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Odd(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor & 1\n\n"
        "class Whole(Transform):\n"
        "    locality = LocalityKind.WHOLE_VOLUME\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    outputs = {
        "before_reduction_transforms": {"Argmax": {}},
        "after_reduction_transforms": {"VotedPhases:Odd": {}},
        "reduction": "Vote",
    }
    outputs[phase] = {**outputs.get(phase, {}), "VotedPhases:Whole": {}}
    manifest, _folds, _per_copy = transform_manifest(_prediction({}, outputs), "Predictor", extension=True)
    assert [step["op"] for step in manifest["postprocessing"]][-2:] == ["extension", "extension"]


@pytest.mark.parametrize(
    ("transforms", "phase", "module", "body", "folded"),
    [
        ({}, "before_reduction_transforms", "Integers", "tensor & 1", True),
        ({}, "after_reduction_transforms", "Votes", "tensor & 1", True),
        ({"TensorCast#2": {"dtype": "int16", "inverse": True}}, "final_transforms", "Doubles", "tensor * 2", False),
    ],
    ids=["in-the-dtype-the-fold-leaves", "in-the-dtype-a-vote-keeps", "past-an-undone-cast"],
)
def test_a_pointwise_class_of_the_app_s_folds_in_the_dtype_its_chain_hands_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transforms: dict, phase: str, module: str, body: str, folded: bool
) -> None:
    # Past the app's Fold, which leaves uint8, and a vote of the copies, which keeps it, a class taking integer labels
    # traces and folds into a graph; past the inverse of a cast only the run knows the dtype, and a class that would
    # trace in any stays its extension's step.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")  # a class folds where its graph exports and checks
    (tmp_path / f"{module}.py").write_text(
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Own(Transform):\n"
        "    locality = LocalityKind.POINTWISE\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        f"        return {body}\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    handed: list[str] = []

    def fold_out(folds: list[Any], steps: list[dict[str, Any]], dtype: str) -> dict[str, Any]:
        handed.append(dtype)
        return {"op": "onnx", "model": "post.onnx", "channels": 1, "dtype": dtype}

    config = _prediction(transforms, {phase: {f"{module}:Own": {}}, "reduction": "Vote"})
    manifest, _, _ = transform_manifest(config, "Predictor", fold_out=fold_out, blended="uint8", extension=True)
    ops = [step["op"] for step in manifest["postprocessing"]]
    assert (handed, "onnx" in ops, "extension" in ops) == ((["uint8"], True, False) if folded else ([], False, True))


def test_an_output_chain_folds_an_apps_own_class_through_a_plain_fold_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A fold_out that says nothing of the dtype it reads folds a class of the app's own as konfai hands a fold its
    # input, in float32.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")  # a class folds where its graph exports and checks
    from konfai.portable import compile_pipeline

    (tmp_path / "PlainFold.py").write_text(
        "from konfai.data.transform import LocalityKind, Transform\n\n"
        "class Twice(Transform):\n"
        "    locality = LocalityKind.POINTWISE\n\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor * 2\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    folded: list[int] = []

    def fold_out(folds: list[Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
        folded.append(len(folds))
        return {"op": "onnx", "model": "post.onnx", "channels": 1, "dtype": "float32"}

    steps, _, _ = compile_pipeline({"PlainFold:Twice": {}}, fold_out=fold_out, extension=True)
    assert folded == [1] and [step["op"] for step in steps] == ["onnx"]


def test_a_case_statistic_of_a_group_of_several_channels_is_refused_before_its_graphs_are_traced() -> None:
    # Its partial graph is traced on one channel, a stored scalar volume, and would refuse a block of two at run time.
    from konfai.portable import _probe_after, compile_pipeline

    traced: list[dict[str, Any]] = []

    def statistic(params: dict[str, Any]) -> dict[str, Any]:
        traced.append(params)
        return {"op": "case_statistic", "halo": 1, "partial": "p.onnx", "finish": "f.onnx"}

    chain = {"CaseStatistic": {"statistic": "oracle_support:Roughness"}}
    with pytest.raises(ConfigError, match="2 channels"):
        compile_pipeline(chain, case_statistic=statistic, probe=_probe_after(3, lambda: 2))
    assert not traced
    steps, _, _ = compile_pipeline(chain, case_statistic=statistic, probe=_probe_after(3, lambda: 1))
    assert [step["op"] for step in steps] == ["case_statistic"]


def test_a_model_with_weights_and_no_checkpoint_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Only a parameter-free model compiles without checkpoints: another would export its random weights.
    from konfai.portable import compile_prediction

    config = _tiny_prediction(tmp_path, monkeypatch, "TinySeg:TinySegNet", "Head:Softmax")
    with pytest.raises(ConfigError, match="needs the checkpoints to export"):
        compile_prediction(config, None, tmp_path / "Predictions", tmp_path / "out" / "prediction.json")


def test_a_weightless_model_running_python_on_the_case_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An app without checkpoints whose graph hands the case's attributes to a module (a wrapper running a
    # tool on the image): refused before any forward runs, by what the exported graph could not carry.
    from konfai.portable import compile_prediction

    (tmp_path / "Reader.py").write_text(
        "import torch\n"
        "from konfai.network import network\n\n"
        "class Geometry(torch.nn.Module):\n"
        "    accepts_attributes = True\n\n"
        "    def forward(self, x, attributes):\n"
        "        raise AssertionError('a compile ran the forward')\n\n"
        "class ReaderNet(network.Network):\n"
        "    def __init__(self, outputs_criterions: dict[str, network.TargetCriterionsLoader] | None = None) -> None:\n"
        "        super().__init__(in_channels=1, outputs_criterions=outputs_criterions, dim=2)\n"
        "        self.add_module('Tool', Geometry())\n"
    )
    config = _tiny_prediction(tmp_path, monkeypatch, "Reader:ReaderNet", "Tool")
    with pytest.raises(ConfigError, match="module 'Tool' reads the case's attributes"):
        compile_prediction(config, None, tmp_path / "Predictions", tmp_path / "out" / "prediction.json")


def _reducing_config(tmp_path: Path, transforms: str) -> Path:
    """A Transform.yml over two small MetaImage cases whose one chain runs ``transforms``."""
    import numpy as np

    sitk = pytest.importorskip("SimpleITK")

    for case in ("CASE_000", "CASE_001"):
        (tmp_path / "Raw" / case).mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(
            sitk.GetImageFromArray(np.zeros((4, 5, 6), np.float32)), str(tmp_path / "Raw" / case / "CT.mha")
        )
    config = tmp_path / "Transform.yml"
    body = "".join(f"              {line}\n" for line in transforms.strip().splitlines())
    config.write_text(
        "Transformer:\n  name: REDUCE\n  Dataset:\n"
        f"    dataset_filenames: ['{tmp_path / 'Raw'}:mha']\n"
        "    groups_src:\n      CT:\n        groups_dest:\n          CT_template:\n            transforms:\n"
        f"{body}              Write: {{dataset: '{tmp_path / 'Template'}:mha'}}\n"
    )
    return config


def test_a_reduce_compiles_to_the_fold_between_the_steps_each_case_runs_and_those_run_once(tmp_path: Path) -> None:
    # konfai runs the stages before a Reduce once per case, folds the results, and runs the stages after it
    # once on the fold, written under the output's name (`split_chain`, `CaseReduction`).
    from konfai.portable import compile_transform

    config = _reducing_config(
        tmp_path,
        """
Clip: {min_value: 0.0, max_value: 400.0}
Reduce: {operator: Median, output: template, grid: 'reference:CASE_001', grid_tolerance: 0.001}
TensorCast: {dtype: int16}
""",
    )
    (chain,) = compile_transform(config, tmp_path / "Transforms")["chains"]
    clip, _ = steps({"Clip": {"min_value": 0.0, "max_value": 400.0}, **_WRITE}, "CT_template")
    cast, _ = steps({"TensorCast": {"dtype": "int16"}, **_WRITE}, "CT_template")
    assert chain["steps"] == clip
    assert chain["reduce"] == {
        "operator": "median",
        "output": "template",
        "grid": "reference:CASE_001",
        "grid_tolerance": 0.001,
        "post": cast,
    }
    assert chain["write"]["root"] == str(tmp_path / "Template")
    assert chain["write"]["format"] == "mha"


def test_compiling_leaves_the_callers_config_as_it_was(tmp_path: Path) -> None:
    # building the workflow writes its defaults back into the config it reads; the compiler restores the
    # caller's file, and the defaults still reach the program
    from konfai.portable import compile_transform

    config = _reducing_config(tmp_path, "Reduce: {operator: Median, output: template}")
    config.write_bytes(config.read_bytes().replace(b"\n", b"\r\n"))  # its line endings too
    before = config.read_bytes()
    (chain,) = compile_transform(config, tmp_path / "Transforms")["chains"]
    assert config.read_bytes() == before
    assert chain["reduce"]["grid"] == "strict"


@pytest.mark.parametrize(
    ("operator", "op"),
    [
        ("Mean", "mean"),
        ("Std", "std"),
        ("Median", "median"),
        ("Vote", "vote"),
        ("Concat", "concat"),
        ("konfai.data.reduction:Median", "median"),
    ],
)
def test_a_reduce_folds_by_the_op_of_the_operator_konfai_resolves(tmp_path: Path, operator: str, op: str) -> None:
    from konfai.portable import compile_transform

    config = _reducing_config(tmp_path, f"Reduce: {{operator: '{operator}', output: template}}")
    (chain,) = compile_transform(config, tmp_path / "Transforms")["chains"]
    assert chain["steps"] == []
    assert chain["reduce"] == {
        "operator": op,
        "output": "template",
        "grid": "strict",
        "grid_tolerance": 1e-6,
        "post": [],
    }


def test_a_reduction_the_portable_runtime_cannot_fold_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from konfai.portable import compile_transform
    from konfai.utils.errors import ReductionError

    # An operator of the user's own, even one deriving a konfai operator, folds by code the runtime does not hold.
    (tmp_path / "trimmed_ops.py").write_text(
        "from konfai.data.reduction import Mean\n\n\nclass Trimmed(Mean):\n    pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config = _reducing_config(tmp_path, "Reduce: {operator: 'trimmed_ops:Trimmed', output: template}")
    with pytest.raises(TransformError, match="folds by 'trimmed_ops:Trimmed'"):
        compile_transform(config, tmp_path / "Transforms")
    config = _reducing_config(
        tmp_path, f"Save: {{dataset: '{tmp_path / 'Work'}:mha'}}\nReduce: {{operator: Median, output: template}}"
    )
    with pytest.raises(TransformError, match="holds a Save"):
        compile_transform(config, tmp_path / "Transforms")
    # konfai's own refusal of a stage after the fold, which it makes only when it plans the run.
    config = _reducing_config(tmp_path, "Reduce: {operator: Median, output: template}\nResample: {spacing: [2, 2, 2]}")
    with pytest.raises(ReductionError, match="reads across space"):
        compile_transform(config, tmp_path / "Transforms")


@pytest.mark.parametrize(("file_format", "refused"), [("mha", False), ("nii", False), ("nii.gz", True), ("nrrd", True)])
def test_a_reduction_is_written_only_where_konfai_writes_one_region_by_region(
    tmp_path: Path, file_format: str, refused: bool
) -> None:
    # konfai's plan probes a region write on the reduction's destination and refuses the run when the
    # format writes whole (`_probe_destination`; `SitkFile.open_data_stream` takes mha and nii).
    from konfai.portable import compile_transform

    config = _reducing_config(tmp_path, "Reduce: {operator: Median, output: template}")
    config.write_text(config.read_text().replace("Template:mha", f"Template:{file_format}"))
    if refused:
        with pytest.raises(
            TransformError, match=f"writes '{file_format}', which konfai cannot write region by region"
        ) as refusal:
            compile_transform(config, tmp_path / "Transforms")
        # KonfAI Edge writes konfai's HDF5 store, so the remedy names it.
        assert "omezarr or h5" in str(refusal.value)
    else:
        (chain,) = compile_transform(config, tmp_path / "Transforms")["chains"]
        assert chain["write"]["format"] == file_format


def test_the_expand_example_compiles_to_the_steps_each_case_runs_and_the_draw_each_copy_runs(tmp_path: Path) -> None:
    # konfai runs the stages before an Expand once per case and the rest once per copy, each copy written under
    # `Expand.entry(case, a)` and drawn from the run's manual_seed (`split_expand`, `_draw_expand_copies`).
    from konfai.data.transform import Expand
    from konfai.portable import compile_transform

    example = Path(__file__).parents[2] / "examples" / "Transform" / "Transform_expand.yml"
    config = _reducing_config(tmp_path, "Clip: {min_value: 0.0, max_value: 400.0}")
    config.write_text(
        example.read_text()
        .replace("./Raw:mha", f"{tmp_path / 'Raw'}:mha")
        .replace("./Augmented:mha", f"{tmp_path / 'Augmented'}:mha")
    )
    program = compile_transform(config, tmp_path / "Transforms")
    (chain,) = program["chains"]
    clip, _ = steps({"Clip": {"min_value": 0.0, "max_value": 400.0}, **_WRITE}, "CT_aug")
    marker = Expand(nb=4, pattern="{name}_r{a:02d}")
    assert program["konfai_edge_transform"] == 4
    assert chain["steps"] == clip
    assert chain["expand"] == {
        "seed": 7,
        "entries": {case: [marker.entry(case, a) for a in range(1, 5)] for case in ("CASE_000", "CASE_001")},
        "post": [{"op": "brightness", "b_std": 25.0}],
    }
    assert chain["expand"]["entries"]["CASE_001"] == ["CASE_001_r01", "CASE_001_r02", "CASE_001_r03", "CASE_001_r04"]
    assert chain["write"]["root"] == str(tmp_path / "Augmented")


def test_an_expand_carries_the_dtype_and_its_saves_across_the_marker(tmp_path: Path) -> None:
    # A Save before the marker is the case's and one after it the copy's, and the draw's float32 settles the
    # interpolation of a Resample after it, which konfai reads linear.
    from konfai.portable import compile_transform

    work = tmp_path / "Work"
    config = _reducing_config(
        tmp_path,
        f"""
Save: {{dataset: '{work}:mha'}}
Expand: {{nb: 2, seed: 3}}
Brightness: {{b_std: 5.0}}
Resample: {{spacing: [2.0, 2.0, 2.0]}}
konfai.data.transform:Save: {{dataset: '{work}:mha'}}
""",
    )
    (chain,) = compile_transform(config, tmp_path / "Transforms")["chains"]
    assert [step["op"] for step in chain["steps"]] == ["save"]
    expand = chain["expand"]
    assert expand["seed"] == 3
    assert expand["entries"] == {
        "CASE_000": ["CASE_000_01", "CASE_000_02"],
        "CASE_001": ["CASE_001_01", "CASE_001_02"],
    }
    assert [step["op"] for step in expand["post"]] == ["brightness", "resample", "save"]
    assert expand["post"][1]["interpolation"] == "linear"
    assert expand["post"][2]["destination"]["root"] == str(work)


@pytest.mark.parametrize("draw", ["Flip: {f_prob: [0.5, 0.5, 0.5]}", "Contrast: {c_std: 0.1}"])
def test_a_draw_the_portable_runtime_does_not_draw_is_refused_by_name(tmp_path: Path, draw: str) -> None:
    # Past the marker a bare `Flip` is konfai's draw, not the transform the runtime flips by.
    from konfai.portable import compile_transform

    config = _reducing_config(tmp_path, f"Expand: {{nb: 2}}\n{draw}")
    name = draw.split(":", 1)[0]
    with pytest.raises(TransformError, match=f"'{name}' draws a {name} for each copy"):
        compile_transform(config, tmp_path / "Transforms")


@pytest.mark.parametrize(
    ("chain", "refused"),
    [
        ({"Rotate": {"r_prob": [1.0, 1.0, 1.0]}}, "Rotate"),
        ({"Flip": {"f_prob": [1.0, 0.0]}, "Rotate": {}}, "Rotate"),
    ],
)
def test_portable_tta_refuses_a_draw_it_cannot_represent(chain: dict, refused: str) -> None:
    with pytest.raises(ConfigError, match=refused):
        tta_passes(_prediction({}, {}, flips=1, draws=chain), "Predictor")


def test_a_vector_field_draw_negates_the_component_of_every_mirrored_axis() -> None:
    # konfai's `Flip(vector_field=True)` mirrors the axes and negates each one's component channel, and
    # the draw is its own inverse, so the copy's inverse negates them again.
    config = _prediction({}, {}, flips=1, draws={"Flip": {"f_prob": [1.0, 1.0, 1.0], "vector_field": True}})
    passes = tta_passes(config, "Predictor")
    assert passes == [[], [{"op": "flip", "axes": [0], "vector_field": True}]]
    copy = assemble_program(_one_model(), reduce="mean", passes=passes)["steps"][1]["manifest"]
    assert copy["preprocessing"][-1] == {"op": "flip", "axes": [0], "vector_field": True, "id": "__tta_0"}
    assert copy["postprocessing"][0] == {"op": "inverse", "of": "__tta_0", "vector_field": True}


@pytest.mark.parametrize("as_list", [False, True])
def test_portable_tta_composes_every_flip_in_a_draw(as_list: bool) -> None:
    import torch

    first = {"f_prob": [0.0, 0.0, 1.0]}
    second = {"f_prob": [1.0, 1.0, 0.0]}
    chain = [{"Flip": first}, {"Flip": second}] if as_list else {"Flip": first, "Flip#2": second}
    passes = tta_passes(_prediction({}, {}, flips=1, draws=chain), "Predictor")
    # Both flips apply, in the chain's order, and are undone.
    image = torch.arange(24).reshape(1, 2, 3, 4)
    expected = torch.flip(torch.flip(image, [3]), [1])
    actual = image
    for op in passes[1]:
        actual = torch.flip(actual, [axis + 1 for axis in op["axes"]])
    assert passes[0] == []
    assert torch.equal(actual, expected)


def test_portable_tta_keeps_identity_copies_in_the_reduction() -> None:
    assert tta_passes(_prediction({}, {}, flips=3, draws={}), "Predictor") == [[], [], [], []]


@pytest.mark.parametrize(
    ("spec", "declared", "exported", "expected"),
    [
        (0.25, [8, 12, 16], [8, 12, 16], [2, 3, 4]),
        ("25%", [8, 12, 16], [8, 12, 16], [2, 3, 4]),
        ([0.25, 2, "50%"], [8, 12, 16], [8, 12, 16], [2, 2, 8]),
        ([0, 0.25, "50%"], [1, 8, 12], [8, 12], [2, 6]),
    ],
)
def test_portable_overlap_keeps_fractional_and_percentage_windows(
    spec: object, declared: list[int], exported: list[int], expected: list[int]
) -> None:
    result = derive_overlap({"patch_size": declared, "overlap": spec}, exported)
    assert result == expected


def test_a_pointwise_patch_transform_is_baked_after_case_preprocessing() -> None:
    import torch

    config = _prediction({"Clip": {"min_value": -2.0, "max_value": 3.0}}, {})
    config["Predictor"]["Dataset"] = _input_group(
        config["Predictor"]["Dataset"]["groups_src"]["g"]["groups_dest"]["g"]["transforms"],
        patch_transforms={"Clip": {"min_value": 0.0, "max_value": 0.01}},
    )
    manifest, folds, _ = transform_manifest(config, "Predictor")
    assert manifest["preprocessing"][-1]["max_value"] == 3.0
    assert len(folds) == 1
    actual = folds[0](torch.tensor([[[[-1.0, 0.005, 1.0]]]]))
    assert torch.equal(actual, torch.tensor([[[[0.0, 0.005, 0.01]]]]))


@pytest.mark.parametrize(
    "params",
    [
        {"mean": 0.0, "std": 2.0, "inverse": True},
        {"inverse": False},
    ],
)
def test_a_patch_transform_with_no_portable_state_is_refused(params: dict) -> None:
    config = _prediction({}, {})
    config["Predictor"]["Dataset"] = _input_group(
        config["Predictor"]["Dataset"]["groups_src"]["g"]["groups_dest"]["g"]["transforms"],
        patch_transforms={"Standardize": params},
    )
    with pytest.raises(ConfigError, match=r"patch_transforms.*Standardize"):
        transform_manifest(config, "Predictor")


def test_an_empty_model_input_chain_never_uses_an_auxiliary_groups_transform() -> None:
    config = _prediction({}, {})
    config["Predictor"]["Dataset"] = {
        "groups_src": {
            "MASK": {"groups_dest": {"MASK": {"is_input": False, "transforms": {"Clip": {"max_value": 1}}}}},
            "MR": {"groups_dest": {"MR": {"is_input": True, "transforms": None, "patch_transforms": None}}},
        }
    }
    manifest, folds, _ = transform_manifest(config, "Predictor")
    assert manifest["preprocessing"] == []
    assert folds == []


def test_a_resample_through_a_field_reads_it_beside_the_case_or_from_its_own_store(tmp_path: Path) -> None:
    # konfai reads the case's entry of the field group, in the run's roots, or in the store `field` names, whose
    # only group it takes when field_group names none (`_DisplacementSource`); a field alone keeps the case's own
    # grid and its align (`_OwnGrid`). konfai-edge tests/cli_transform.rs runs both on konfai's own fields.
    store = tmp_path / "Fields"
    (store / "CASE_000").mkdir(parents=True)
    (store / "CASE_000" / "DVF.mha").write_bytes(b"")
    beside = {"field_group": "DVF", "interpolation": "linear", "align": "origin"}
    assert steps({"Resample": beside, **_WRITE}, "CT", [f"{tmp_path}:mha"])[0] == [
        {"op": "resample", "field_group": "DVF", "interpolation": "linear"}
    ]
    by_spacing = {"spacing": [1.5, 0.7, 1.2], "field_group": "DVF", "fill": -5.0}
    assert steps({"Resample": by_spacing, **_WRITE}, "CT", [f"{tmp_path}:mha"])[0] == [
        {"op": "resample", "spacing": [1.5, 0.7, 1.2], "field_group": "DVF", "fill": -5.0}
    ]
    for named in ({}, {"field_group": "DVF"}):
        own = {"field": f"{store}:mha", **named, "interpolation": "linear"}
        assert steps({"Resample": own, **_WRITE}, "CT", [f"{tmp_path}:mha"])[0] == [
            {
                "op": "resample",
                "field_group": "DVF",
                "field_dataset": {"root": str(store), "format": "mha"},
                "interpolation": "linear",
            }
        ]
    with pytest.raises(TransformError, match="field_group: DVF"):
        (store / "CASE_000" / "MORE.mha").write_bytes(b"")
        steps({"Resample": {"field": f"{store}:mha"}, **_WRITE}, "CT", [f"{tmp_path}:mha"])


def test_a_prediction_reads_a_resample_s_field_beside_the_case() -> None:
    # The program reads the field group into a buffer of its own, one component per spatial axis the patch
    # cuts, and hands it to the resample, which the model never sees (konfai-edge tests/prediction.rs).
    warp = {"spacing": [1.5, 1.5, 1.5], "field_group": "DVF", "interpolation": "linear"}
    manifest, _folds, _per_copy = transform_manifest(_prediction({"Resample": warp}, {}), "Predictor")
    assert manifest["preprocessing"][-1] == {"op": "resample", **warp, "id": "Resample"}
    assert manifest["postprocessing"] == [{"op": "inverse", "of": "Resample", "interpolation": "linear"}]
    assert manifest["reads"] == [{"group": "DVF", "channels": 3, "preprocessing": []}]
    assert assemble_program([{"id": "m", "manifest": manifest}], reduce="mean")["steps"][0]["in"] == ["input", "DVF"]


def test_a_resample_s_field_a_prediction_cannot_read_or_undo_is_refused_by_name() -> None:
    # A store of its own is looked up by path, which only a TRANSFORM program does; a warp on the case's own grid
    # has no inverse in konfai (`Resample.inverse`), and konfai's predictor undoes the stage unless told not to.
    with pytest.raises(TransformError, match="only a TRANSFORM program looks up"):
        OP_MAP["Resample"]({"field": "./Fields:mha", "field_group": "DVF"})
    with pytest.raises(ConfigError, match="no inverse for a Resample through a field"):
        transform_manifest(_prediction({"Resample": {"field_group": "DVF"}}, {}), "Predictor")
    manifest, _folds, _per_copy = transform_manifest(
        _prediction({"Resample": {"field_group": "DVF", "interpolation": "linear", "inverse": False}}, {}),
        "Predictor",
    )
    assert manifest["preprocessing"][-1] == {"op": "resample", "field_group": "DVF", "interpolation": "linear"}
    assert manifest["postprocessing"] == []


def test_a_resample_through_stored_transforms_looks_each_group_up_in_the_run_s_datasets(tmp_path: Path) -> None:
    # konfai looks each group up in the run's datasets, in order, and applies the last declared first
    # (`Resample._decode_stored`): the program carries the groups as declared, each with whether to invert it, and
    # the datasets as the run names them. A map alone keeps the case's own grid and its align (`_OwnGrid`).
    # konfai-edge tests/cli_transform.rs runs konfai's own cases through them.
    roots = [f"{tmp_path}/Dataset:mha", f"{tmp_path}/Reg:itktransform"]
    stored = {
        "transforms": {
            "groups": [{"group": "reg", "invert": False}, {"group": "init", "invert": True}],
            "datasets": [
                {"root": f"{tmp_path}/Dataset", "format": "mha"},
                {"root": f"{tmp_path}/Reg", "format": "itktransform"},
            ],
        }
    }
    alone = {"transforms": {"reg": False, "init": True}, "interpolation": "linear", "align": "origin"}
    assert steps({"Resample": alone, **_WRITE}, "CT", roots)[0] == [
        {"op": "resample", **stored, "interpolation": "linear"}
    ]
    by_spacing = {"spacing": [1.5, 0.7, 1.2], "transforms": {"reg": False, "init": True}, "fill": -5.0}
    assert steps({"Resample": by_spacing, **_WRITE}, "CT", roots)[0] == [
        {"op": "resample", "spacing": [1.5, 0.7, 1.2], **stored, "fill": -5.0}
    ]
    with pytest.raises(TransformError, match="holds a '\\*'"):
        steps({"Resample": {"transforms": {"*": False}}, **_WRITE}, "CT", roots)


def test_a_prediction_looks_a_resample_s_stored_transforms_up_in_its_datasets() -> None:
    # konfai's predictor hands its stages every dataset it reads (`sources.py`), so the program looks each group up
    # in the Predictor's dataset_filenames; the transforms are read beside the case, no buffer of the model's. A
    # map on the case's own grid has no inverse in konfai (`Resample.inverse`), which the predictor would run.
    warp = {"spacing": [1.5, 1.5, 1.5], "transforms": {"reg": False}, "interpolation": "linear"}
    config = _prediction({"Resample": warp}, {})
    config["Predictor"]["Dataset"]["dataset_filenames"] = ["./Dataset:mha", "./Reg:itktransform"]
    manifest, _folds, _per_copy = transform_manifest(config, "Predictor")
    assert manifest["preprocessing"][-1] == {
        "op": "resample",
        "spacing": [1.5, 1.5, 1.5],
        "transforms": {
            "groups": [{"group": "reg", "invert": False}],
            "datasets": [{"root": "./Dataset", "format": "mha"}, {"root": "./Reg", "format": "itktransform"}],
        },
        "interpolation": "linear",
        "id": "Resample",
    }
    assert manifest["postprocessing"] == [{"op": "inverse", "of": "Resample", "interpolation": "linear"}]
    assert "reads" not in manifest
    with pytest.raises(ConfigError, match="no inverse for a Resample through stored transforms"):
        transform_manifest(_prediction({"Resample": {"transforms": {"reg": False}}}, {}), "Predictor")
    # A chain naming no dataset (a nested model's) has none to look the transforms up in.
    with pytest.raises(TransformError, match="names none"):
        OP_MAP["Resample"]({"spacing": [1.5, 1.5, 1.5], "transforms": {"reg": False}})


def test_an_overlap_without_a_patch_combine_is_refused_by_name() -> None:
    # konfai owns such an overlap with Trim (output.py `self._patch_combine or "Trim"`), which no runtime op does.
    from konfai.portable import check_blend

    for blend in (None, "Overwrite"):
        with pytest.raises(KonfAIError, match="Trim"):
            check_blend([16, 16, 16], blend)
    check_blend([0, 0, 0], None)
    check_blend(None, "Overwrite")
    check_blend([16, 16, 16], "Gaussian")
    with pytest.raises(KonfAIError, match="patch_combine 'mypkg:MyBlend' is no blend the portable runtime runs"):
        check_blend([16, 16, 16], "mypkg:MyBlend")


def test_an_output_mirroring_another_group_than_the_first_input_is_refused_by_name() -> None:
    # konfai undoes the mirrored group's chain (OutputDataset.same_as_group); the compiler undoes the first
    # input group's, so a second group's inverses cannot be right.
    with pytest.raises(KonfAIError, match="mirrors"):
        transform_manifest(_prediction({}, {"same_as_group": "MR:MR"}), "Predictor")
    transform_manifest(_prediction({}, {"same_as_group": "g:g"}), "Predictor")


def test_an_output_mirroring_another_destination_of_the_first_input_s_source_is_refused_by_name() -> None:
    # konfai undoes the chain of groups_src[src][dest]: 'CT:CT_low' is not the first input 'CT:CT'.
    def mirroring(group: str) -> dict:
        config = _prediction({}, {"same_as_group": group})
        config["Predictor"]["Dataset"] = {
            "groups_src": {
                "CT": {
                    "groups_dest": {
                        "CT": {"is_input": True, "transforms": {"Resample": {"spacing": [1.5, 1.5, 1.5]}}},
                        "CT_low": {"is_input": False, "transforms": {"Resample": {"spacing": [3, 3, 3]}}},
                    }
                }
            }
        }
        return config

    with pytest.raises(KonfAIError, match="mirrors"):
        transform_manifest(mirroring("CT:CT_low"), "Predictor")
    transform_manifest(mirroring("CT:CT"), "Predictor")


def test_a_nested_group_no_mask_reads_leaves_the_model_s_output_as_the_output() -> None:
    # `aux` holds every nested model of the dataset; an output whose chain masks by none of them must end
    # on its own model, not on a mask buffer the program built for nothing.
    manifest = {"preprocessing": [], "postprocessing": []}
    nested = {"id": "aux", "manifest": manifest, "ops": [{"op": "dilate", "radius": 1}]}
    program = assemble_program([{"id": "m", "manifest": manifest}], reduce="mean", aux={"aux": nested}, masks=[])
    assert program["steps"][-1].get("model") == "m", program["steps"]
    assert program["steps"][-1]["out"] == "output"
    # Nor does it run the nested model nobody reads.
    assert [step["model"] for step in program["steps"] if "model" in step] == ["m"]


def test_a_nested_model_no_output_masks_by_is_left_out_of_the_prediction() -> None:
    # compile_prediction exports and lists the nested models of these groups only.
    from konfai.portable import read_nested_groups

    nested = {"is_input": False, "transforms": {"KonfAIInference": {"repo_id": "R/S", "model_name": "body"}}}

    def config(before: dict[str, Any]) -> dict[str, Any]:
        groups = {"Volume_0": {"groups_dest": {"MASK": nested, "Volume": {"is_input": True}}}}
        outputs = {"H": {"OutputDataset": {"before_reduction_transforms": before}}}
        return {"Predictor": {"Dataset": {"groups_src": groups}, "outputs_dataset": outputs}}

    assert read_nested_groups(config({}), "Predictor", ["H"]) == {}
    assert set(read_nested_groups(config({"Mask": {"path": "MASK"}}), "Predictor", ["H"])) == {"MASK"}


@pytest.mark.parametrize("value", [None, "None"])
def test_an_explicitly_disabled_patch_combine_keeps_the_last_patch(value) -> None:
    from konfai.portable import derive_blend

    config = _prediction({}, {"patch_combine": value})
    assert derive_blend(config, "Predictor") == "Overwrite"


@pytest.mark.parametrize(("pad_value", "expected"), [(-1.0, -1.0), ("edge", "edge"), (None, "min"), ("None", "min")])
def test_an_onnx_export_carries_the_pad_value_of_a_short_axis(pad_value: float | str | None, expected) -> None:
    # A pad_value left null pads with the minimum of the window konfai reads, not with 0.
    from konfai.portable import derive_onnx_params

    config = {"Predictor": {"Dataset": {"Patch": {"patch_size": [16, 16], "pad_value": pad_value}}}}
    assert derive_onnx_params(config, "Predictor")[3] == expected


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        ({}, "pad"),
        ({"placement": "None"}, "pad"),
        ({"placement": "end"}, "end"),
        ({"placement": "spread"}, "spread"),
    ],
)
def test_the_placement_a_patch_states_reaches_the_manifest(patch: dict[str, str], expected: str) -> None:
    from konfai.portable import derive_placement

    assert derive_placement(patch) == expected


def test_a_placement_konfai_does_not_know_is_refused() -> None:
    # The runtime would place the windows otherwise than the config says.
    from konfai.portable import derive_placement

    with pytest.raises(ConfigError, match="placement 'centre'"):
        derive_placement({"patch_size": [16, 16], "placement": "centre"})


@pytest.mark.parametrize(
    ("patch", "exported", "expected"),
    [
        # Patch 5, overlap 0.25: konfai steps by 3.75 (`end` truncates it to 3); the integer overlap, 1, would step by 4.
        ({"overlap": 0.25}, [5], [3.75]),
        ({"overlap": "25%"}, [8, 16], [6.0, 12.0]),
        ({"overlap": 2}, [5], [3]),
        # Left unset, konfai's end and spread step on the default fraction of the patch: nothing is case-dependent.
        ({}, [5], [4.0]),
        # A 2.5-D axis drops out of the step as it does of the overlap.
        ({"patch_size": [1, 8, 12], "overlap": [0, 0.25, "50%"]}, [8, 12], [6.0, 6.0]),
    ],
)
def test_an_end_or_spread_placement_exports_the_step_konfai_walks(
    patch: dict[str, object], exported: list[int], expected: list[float]
) -> None:
    from konfai.portable import derive_stride

    assert derive_stride(patch, exported) == pytest.approx(expected)


@pytest.mark.parametrize("size", [[0, 16, 16], [16, 0, 16], [16, 16, -1]])
def test_an_onnx_export_refuses_some_free_patch_axes_instead_of_dropping_them(size: list[int]) -> None:
    from konfai.portable import derive_onnx_params

    config = {"Predictor": {"Dataset": {"Patch": {"patch_size": size}}}}
    with pytest.raises(ConfigError, match=r"patch_size.*frees some axes"):
        derive_onnx_params(config, "Predictor")
    # Every axis free is one window per case, exported with its axes free.
    config["Predictor"]["Dataset"]["Patch"]["patch_size"] = [0, 0, 0]
    assert derive_onnx_params(config, "Predictor")[0] == [0, 0, 0]


@pytest.mark.parametrize(("prob", "copies"), [(0.0, [[], []]), (1.0, [[0], [2]])])
def test_portable_tta_reads_the_resolved_draw_gate(prob: float, copies: list[list[int]]) -> None:
    config = _prediction(
        {}, {}, flips=2, draws={"Flip": {"f_prob": [1.0, 0.0, 1.0], "prob": prob, "vector_field": False}}
    )
    # A draw the gate closes leaves the copy as it is: an identity pass, with no op to undo.
    assert tta_passes(config, "Predictor") == [[], *([{"op": "flip", "axes": axes}] if axes else [] for axes in copies)]


@pytest.mark.parametrize(("f_prob", "nb"), [([0.0, 0.5, 0.5], 2), ([0.5, 0.5, 0.5], 7), ([1.0, 0.0, 1.0], 5)])
def test_portable_tta_flips_are_the_copies_konfai_predicts_with(
    monkeypatch: pytest.MonkeyPatch, f_prob: list[float], nb: int
) -> None:
    from konfai.data.augmentation import Flip
    from konfai.utils.dataset import Attribute
    from konfai.utils.runtime import State

    monkeypatch.setenv("KONFAI_STATE", str(State.PREDICTION))
    draw = Flip(f_prob=f_prob)
    draw._state_init(0, [[4, 4, 4] for _ in range(nb)], [Attribute() for _ in range(nb)])
    config = _prediction({}, {}, flips=nb, draws={"Flip": {"f_prob": f_prob}})
    # konfai's flip dims count the channel axis first.
    konfai = [[{"op": "flip", "axes": [dim - 1 for dim in dims]}] if dims else [] for dims in draw.flip[0]]
    assert tta_passes(config, "Predictor") == [[], *konfai]


def test_portable_tta_reads_an_unset_seed_as_konfai_resolves_it() -> None:
    # A Prediction.yml that leaves manual_seed unset resolves it to 'None', and the draws compiled from it
    # stop at int('None').
    config = _prediction({}, {}, flips=2)
    config["Predictor"]["manual_seed"] = "None"
    assert len(tta_passes(config, "Predictor")) == 3


def test_a_softmax_or_argmax_off_the_channel_axis_is_refused_by_name() -> None:
    # The runtime reads across the channels whatever dim the step names, so any other axis would run
    # silently on the wrong one.
    from konfai.portable import compile_pipeline

    for stage in ("Softmax", "Argmax"):
        with pytest.raises(KonfAIError, match=stage):
            compile_pipeline({stage: {"dim": 1}}, fold=False)
    steps, _folds, _inverted = compile_pipeline({"Softmax": {"dim": 0}, "Argmax": {}}, fold=False)
    assert steps == [{"op": "softmax", "dim": 0}, {"op": "argmax", "dim": 0}]


def test_a_normalize_over_some_channels_is_refused_by_name() -> None:
    # konfai's Normalize(channels) measures and rescales the named channels alone; the runtime rescales
    # every channel by the volume's extrema.
    from konfai.portable import compile_pipeline

    with pytest.raises(KonfAIError, match="Normalize"):
        compile_pipeline({"Normalize": {"channels": [0], "inverse": False}}, fold=False)
    steps, _folds, _inverted = compile_pipeline({"Normalize": {"channels": None, "inverse": False}}, fold=False)
    assert steps == [{"op": "normalize", "min_value": -1.0, "max_value": 1.0}]


#: What each op compiles from at the least: the arguments its konfai class cannot do without.
_LEAST = {
    "TensorCast": {"dtype": "float32"},
    "Resample": {"spacing": [1.0, 1.0, 1.0]},
    "Padding": {"padding": [1, 1, 1, 1, 1, 1]},
    "Flip": {"dims": "0"},
    "Permute": {"dims": "1|0|2"},
    "SelectLabel": {"labels": ["(1,2)"]},
    "ThresholdLabels": {"labels": [1]},
    "OneHot": {"num_classes": 2},
    "HistogramMatching": {"reference_group": "REF"},
    "Percentage": {"baseline": 1.0},
    "Mask": {"path": "BODY"},
    "Brightness": {"b_std": 0.1},
}


#: The parameters each stage of the chain vocabulary is run with beyond its least ones, where they change the dtype.
_DTYPE_VARIANTS = {
    "Gradient": [{}, {"per_dim": True}],
    "Standardize": [{}, {"nonzero": True, "inverse": False}],
    "TensorCast": [{"dtype": "int16"}, {"dtype": "float32"}],
}
_DTYPES = ("uint8", "int8", "int16", "int32", "int64", "float16", "float32", "float64")
#: The dtypes konfai's stage refuses, each a row the portable runtime is tested against: any other failure fails.
_REFUSED = {
    "Clip": ("uint8", "int8"),  # bounds past the dtype's range
    "HistogramMatching": ("uint8", "int8", "int16", "int32", "int64", "float64"),  # float16 and float32 alone
}


def _tabled(step: dict[str, Any], dtype: str, returned: str | None) -> None:
    """One row of the table of what each op of a program returns of each dtype (``None``: konfai's stage refuses it),
    appended as a JSON line to the file ``KONFAI_DTYPE_TABLE`` names, which the portable runtime tests against."""
    if table := os.environ.get("KONFAI_DTYPE_TABLE"):
        with open(table, "a", encoding="utf-8") as rows:
            rows.write(json.dumps({"step": step, "dtype": dtype, "returns": returned}) + "\n")


@pytest.mark.parametrize("stage", sorted(_CHAIN_OPS))
def test_every_op_hands_on_the_dtype_its_konfai_stage_returns(stage: str, tmp_path: Path) -> None:
    # konfai's stage run on a volume of each dtype it takes: what the compiler says it leaves (_result_dtype, which
    # settles the default interpolation and the dtype a fold is traced in after it) is what it returns.
    import numpy as np
    import torch
    from konfai.data.augmentation.color import Brightness
    from konfai.portable import _probe_attribute, _probed, _result_dtype, _stage_class
    from konfai.utils.dataset import Attribute, Dataset

    companions = Dataset(str(tmp_path / "Companions"), "mha")
    for group in ("REF", "BODY"):
        companions.write(group, "probe", np.ones((1, 6, 7, 8), np.float32), Attribute(_probe_attribute(3)))
    for params in _DTYPE_VARIANTS.get(stage, [_LEAST.get(stage, {})]):
        step, ran, refused = _CHAIN_OPS[stage](dict(params)), [], []
        for dtype in _DTYPES:
            volume = (torch.arange(6 * 7 * 8).reshape(1, 6, 7, 8) % 2).to(getattr(torch, dtype))
            try:
                if stage == "Brightness":  # a draw: run as konfai draws a copy of the case
                    draw = Brightness(**params)
                    draw.load(1.0)
                    draw.state_init(0, [list(volume.shape[1:])], [Attribute()])
                    returned = draw("probe", 0, [volume], [Attribute()])[0]
                else:
                    built = _stage_class(stage)(**params)
                    built.set_datasets([companions])
                    returned = _probed(built, volume, _probe_attribute(3))
            except (RuntimeError, TypeError, ValueError, KonfAIError):  # a dtype the stage does not take
                _tabled(step, dtype, None)
                refused.append(dtype)
                continue
            ran.append((dtype, _result_dtype(step, dtype), str(returned.dtype).removeprefix("torch.")))
            _tabled(step, dtype, ran[-1][2])
        assert ran, f"{stage} {params} takes none of {_DTYPES}"
        assert tuple(refused) == _REFUSED.get(stage, ()), (stage, params, refused)  # no other failure
        assert [(dtype, said) for dtype, said, _ in ran] == [(dtype, got) for dtype, _, got in ran], (stage, params)


@pytest.mark.parametrize(
    ("op", "fold"),
    [
        ("mean", "reduction:Mean"),
        ("median", "reduction:Median"),
        ("median_lower", "reduction:Median"),
        ("std", "reduction:Std"),
        ("variance", "ensemble:Variance"),
        ("standard_deviation", "ensemble:StandardDeviation"),
        ("segmentation_disagreement", "ensemble:SegmentationDisagreement"),
    ],
)
def test_every_fold_of_copies_or_members_hands_on_the_dtype_konfai_s_returns(op: str, fold: str) -> None:
    # Three copies of each dtype, folded by konfai's reduction, or a stack of three members by its ensemble stage.
    import torch
    from konfai.data import reduction
    from konfai.data.transform import ensemble
    from konfai.portable import _result_dtype
    from konfai.utils.dataset import Attribute

    module, name = fold.split(":")
    cls = getattr({"reduction": reduction, "ensemble": ensemble}[module], name)
    for dtype in _DTYPES:
        values = (torch.arange(3 * 6 * 7).reshape(3, 1, 6, 7) % 3).to(getattr(torch, dtype))
        if module == "reduction":
            returned = cls()([copy[None] for copy in values])
        else:
            returned = cls()("probe", values[:, 0], Attribute())
        _tabled({"op": op}, dtype, str(returned.dtype).removeprefix("torch."))
        assert _result_dtype({"op": op}, dtype) == str(returned.dtype).removeprefix("torch."), (op, dtype)


def test_a_resample_past_a_gradient_of_labels_is_linear_as_konfai_s() -> None:
    # A gradient of uint8 labels is float32 in konfai, which a Resample stating nothing reads linearly, where uint8
    # labels it reads by nearest.
    manifest, _, _ = transform_manifest(
        _prediction({"TensorCast": {"dtype": "uint8"}, "Gradient": {}, "Resample": {"spacing": [2.0, 2.0, 2.0]}}, {}),
        "Predictor",
    )
    assert [step.get("interpolation") for step in manifest["preprocessing"] if step["op"] == "resample"] == ["linear"]


@pytest.mark.parametrize("stage", sorted(_CHAIN_OPS))
def test_every_op_refuses_a_key_its_konfai_class_does_not_take(stage: str) -> None:
    # konfai builds a stage from the keys its config gives and refuses one its class does not take: the compiler
    # refuses it too, whatever its op reads, rather than compile a program that runs without it.
    least = _LEAST.get(stage, {})
    assert _CHAIN_OPS[stage](dict(least))["op"]
    with pytest.raises(TransformError, match="no_such_key"):
        _CHAIN_OPS[stage]({**least, "no_such_key": 1})


def test_a_chain_draw_s_prob_is_refused_as_konfai_s_build_refuses_it(tmp_path: Path) -> None:
    # konfai binds a draw of a chain by its constructor keys alone (a test-time draw's prob is read apart, through
    # Prob): its build refuses a prob nothing reads, and the op refuses it too, the program never drawing without it.
    from konfai.portable import compile_transform

    config = _reducing_config(tmp_path, "Expand: {nb: 2, seed: 3}\nBrightness: {b_std: 5.0, prob: 0.5}\n")
    with pytest.raises(ConfigError, match=r"Brightness\.prob"):
        compile_transform(config, tmp_path / "Transforms")
    with pytest.raises(TransformError, match="prob"):
        _CHAIN_OPS["Brightness"]({"b_std": 5.0, "prob": 0.5})


def test_a_chain_konfai_refuses_to_build_compiles_to_no_program() -> None:
    # SynthSeg-KonfAI at 728e5cb: a Crop that crops only, then one that pads only, and a Resample on a grid konfai has
    # no rule for. konfai's run refuses each as it builds it, and so does the compile, in both chains it compiles.
    from konfai.data.transform import Crop, Resample
    from konfai.portable import compile_pipeline

    crop = {"shape": [0, 0, 0], "only": "crop", "fill": 0.0, "inverse": True}
    resample = {"spacing": [1.0, 1.0, 1.0], "grid": "first_face", "interpolation": "linear"}
    with pytest.raises(TypeError, match="only"):
        Crop(**crop)
    with pytest.raises(TransformError, match="first_face"):
        Resample(**resample)
    chains = [
        {"Crop": crop, "Clip": {"min_value": "percentile:0.5", "max_value": "percentile:99.5"}, "Crop#2": crop},
        {"Resample": resample},
    ]
    for chain in chains:
        with pytest.raises(TransformError, match=r"only|first_face"):
            compile_pipeline(chain, fold=True, invertible=True)
        with pytest.raises(TransformError, match=r"only|first_face"):
            steps({**chain, "Write": {"dataset": "./Out:mha"}}, "CT")


@pytest.mark.parametrize(
    ("chain", "refused"),
    [
        # Crop.transform_shape: a Crop behind another of the chain would read the first one's box.
        (
            {
                "Crop": {"shape": [4, 4, 4]},
                "Clip": {"min_value": 0.0, "max_value": 1.0},
                "Crop#2": {"shape": [6, 6, 6]},
            },
            "earlier 'Crop'",
        ),
        # Crop.transform_shape: a source group no dataset holds, the case's to say (a needed app's output, staged
        # beside it at run time): no rule of the chain's.
        ({"Crop": {"source": "BODY"}}, None),
    ],
    ids=["a-second-crop", "a-crop-source-staged-at-run-time"],
)
def test_a_group_chain_konfai_plans_to_no_case_compiles_to_no_program(chain: dict, refused: str | None) -> None:
    # konfai binds a group's chain stage by stage, each with the stages before it, and plans it before a case runs:
    # each rule of that planning, refused at compile with konfai's own message where konfai refuses every case.
    from konfai.portable import transform_manifest

    config = {"Predictor": {"Dataset": {"groups_src": {"CT": {"groups_dest": {"CT": {"transforms": chain}}}}}}}
    compiles = [
        lambda: transform_manifest(config, "Predictor"),
        lambda: steps({**chain, "Write": {"dataset": "./Out:mha"}}, "CT"),
    ]
    for compile_it in compiles:
        if refused is None:
            compile_it()
            continue
        with pytest.raises(TransformError, match=refused):
            compile_it()


def test_a_keep_largest_component_carries_its_labels_and_connectivity_and_refuses_another() -> None:
    from konfai.portable import compile_pipeline

    with pytest.raises(KonfAIError, match="connectivity=18"):
        compile_pipeline({"KeepLargestComponent": {"connectivity": 18}}, fold=False)
    steps, _folds, _inverted = compile_pipeline(
        {"KeepLargestComponent": {"per_label": True, "connectivity": 26}}, fold=False
    )
    assert steps == [{"op": "keep_largest_component", "per_label": True, "connectivity": 26}]


@pytest.mark.parametrize("rank", [2, 3])
def test_a_per_axis_gradient_hands_on_a_channel_per_spatial_axis_of_the_case(rank: int) -> None:
    # The channels the compiler counts after the step are konfai's own Gradient's, in 2-D as in 3-D.
    import torch
    from konfai.data.transform import Gradient
    from konfai.portable import _channels_after
    from konfai.utils.dataset import Attribute

    handed = Gradient(per_dim=True)("case", torch.rand(2, *[5] * rank), Attribute())
    assert _channels_after({"op": "gradient", "per_dim": True}, 2, rank) == handed.shape[0]


def test_an_in_plane_quarter_turn_turns_about_the_slice_axis_alone() -> None:
    # konfai zeroes the angles of the first two axes of an in-plane draw, so every copy is a turn about
    # one axis: that axis reads itself and is never mirrored, whatever the seed.
    import random

    from konfai.portable import _quarter_turn

    def fixed(ops: list[dict]) -> set[int]:
        permuted = next((op["axes"] for op in ops if op["op"] == "permute"), [0, 1, 2])
        mirrored = next((op["axes"] for op in ops if op["op"] == "flip"), [])
        return {axis for axis in range(3) if permuted[axis] == axis and axis not in mirrored}

    kept = [fixed(_quarter_turn(random.Random(seed), 3, in_plane=True)) for seed in range(30)]
    assert all(len(axes) == 1 for axes in kept) and len(set(map(frozenset, kept))) == 1, kept
    free = [fixed(_quarter_turn(random.Random(seed), 3)) for seed in range(30)]
    assert any(axes != kept[0] for axes in free), "a free draw turns about every axis"


def test_a_model_tiling_by_a_model_patch_is_refused_by_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The export runs the whole inference patch through the network; a ModelPatch tiles inside it, which
    # the exported graph would drop. The loader is stubbed: the refusal comes before any forward.
    from konfai.portable import export_checkpoints

    class Tiled:
        patch = object()

    class Loader:
        def __init__(self, classpath: str) -> None:
            pass

        def get_model(self, train: bool = False) -> Tiled:
            return Tiled()

    monkeypatch.setattr("konfai.network.network.ModelLoader", Loader)
    with pytest.raises(KonfAIError, match="ModelPatch"):
        export_checkpoints(
            {"Predictor": {"Model": {"classpath": "Tiled"}, "Dataset": {"Patch": {"patch_size": [8, 8]}}}},
            "Predictor",
            [None],
            tmp_path,
            ["model"],
            in_channels=1,
        )


def test_a_permute_draw_with_no_shares_gives_each_of_two_copies_its_own_order() -> None:
    # konfai's Permute(prob_permute=None) draws torch.eye(2): the first copy takes the first order alone,
    # the second the second, and any other copy count is refused.
    config = _prediction({}, {}, flips=2, draws={"Permute": {"prob_permute": None}})
    passes = tta_passes(config, "Predictor")
    assert passes == [[], [{"op": "permute", "axes": [1, 0, 2]}], [{"op": "permute", "axes": [2, 0, 1]}]]
    config["Predictor"]["Dataset"]["augmentations"]["T"]["nb"] = 3
    with pytest.raises(ConfigError, match="Permute"):
        tta_passes(config, "Predictor")


def _warnings_of(config: dict, root: str = "Predictor") -> list[str]:
    """What the compilers warn about for ``config``, one message per key."""
    import warnings

    from konfai.portable import warn_unportable_keys
    from konfai.utils.errors import KonfAIWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_unportable_keys(config, root)
    return [str(warning.message) for warning in caught if issubclass(warning.category, KonfAIWarning)]


@pytest.mark.parametrize(
    ("section", "key"),
    [
        (
            {"channels_last": True, "Dataset": {"num_workers": 4}},
            "Predictor.channels_last, Predictor.Dataset.num_workers",
        ),
        ({"outputs_dataset": {"H": {"OutputDataset": {"attributes": ["Modality=CT"]}}}}, "OutputDataset.attributes"),
    ],
)
def test_a_prediction_key_the_program_cannot_carry_is_warned_by_key(section: dict, key: str) -> None:
    # konfai reads these keys and the program has no counterpart: a header attribute, a setting that changes
    # how konfai runs. Each is said by key, so the
    # author can decide, where a refusal would name a chain the two engines read differently.
    from konfai.portable import warn_unportable_keys
    from konfai.utils.errors import KonfAIWarning

    config = _prediction({}, {})
    config["Predictor"] = {**config["Predictor"], **section}
    with pytest.warns(KonfAIWarning, match=re.escape(key)):
        warn_unportable_keys(config, "Predictor")
    assert len(_warnings_of(config)) == 1


def test_a_config_at_konfais_defaults_compiles_without_a_warning() -> None:
    config = _prediction({}, {"attributes": None})
    config["Predictor"].update(
        {"autocast": False, "channels_last": False, "Dataset": {**config["Predictor"]["Dataset"], "num_workers": None}}
    )
    assert _warnings_of(config) == []


def test_a_fast_resample_and_a_reduce_provenance_are_warned_in_either_workflow() -> None:
    fast = {"Resample": {"spacing": [1.5, 1.5, 1.5], "precision": "fast"}}
    assert [w.split(":")[0] for w in _warnings_of(_prediction(fast, {}))] == [
        "Predictor.Dataset.groups_src.g.groups_dest.g.transforms.Resample.precision"
    ]
    assert _warnings_of(_prediction({"Resample": {"spacing": [1.5, 1.5, 1.5], "precision": "exact"}}, {})) == []
    chain = {"CT": {"groups_dest": {"T": {"transforms": {**fast, "Reduce": {"operator": "Mean", "output": "t"}}}}}}
    transformer = {"Transformer": {"Dataset": {"groups_src": chain}}}
    warned = _warnings_of(transformer, "Transformer")
    assert [w.split(":")[0] for w in warned] == [
        "Transformer.Dataset.groups_src.CT.groups_dest.T.transforms.Resample.precision",
        "Transformer.Dataset.groups_src.CT.groups_dest.T.transforms.Reduce.provenance",
    ]
    chain["CT"]["groups_dest"]["T"]["transforms"]["Reduce"]["provenance"] = False
    assert len(_warnings_of(transformer, "Transformer")) == 1


@pytest.mark.parametrize(
    ("draw", "key"),
    [
        ({"Flip": {"f_prob": [1.0, 1.0, 1.0], "prob": 0.5}}, "Flip.prob"),
        ({"Permute": {"prob_permute": [0.5, 0.5]}}, "Permute.prob_permute"),
    ],
)
def test_a_fractional_test_time_share_warns_that_the_copies_are_not_torchs(draw: dict, key: str) -> None:
    # konfai draws such a share with torch's generator and the program with Python's from the same seed: the
    # copies are reproducible, not konfai's.
    import warnings

    from konfai.utils.errors import KonfAIWarning

    with pytest.warns(KonfAIWarning, match=re.escape(key)):
        tta_passes(_prediction({}, {}, flips=2, draws=draw), "Predictor")
    # A test-time Flip's f_prob only names the axes its mirrors take: konfai draws nothing from it.
    config = _prediction({}, {}, flips=2, draws={"Flip": {"f_prob": [0.5, 0, 1]}})
    with warnings.catch_warnings():
        warnings.simplefilter("error", KonfAIWarning)
        tta_passes(config, "Predictor")


# The program contract: every program these tests compile meets the schema written once in
# `konfai.portable_schema`, and the copy committed beside the runtime's fixtures is that schema.

_SCHEMA_COPY = Path(__file__).parents[2] / "konfai-edge" / "tests" / "fixtures" / "program_schema.json"
_MRSEG_PROGRAM = _SCHEMA_COPY.with_name("mrseg_program.json")
# The public tree ships the compiler without the runtime; these checks pair the two where both are present.
_needs_the_runtime = pytest.mark.skipif(not _SCHEMA_COPY.exists(), reason="the runtime's fixtures are not in this tree")


def _meets(instance: object, definition: str, schema: dict) -> None:
    """Validate ``instance`` against ``definition`` of ``schema`` (draft 2020-12), the root's own ``oneOf`` of the
    four whole documents set aside: a fragment is none of them."""
    jsonschema.Draft202012Validator({"$defs": schema["$defs"], "$ref": f"#/$defs/{definition}"}).validate(instance)


def _without_required(node: object) -> object:
    """The schema with every ``required`` dropped: a fragment, or the skeletal manifests these tests assemble
    programs from, carry the chains alone; the keys the export writes are pinned by test_onnx_export."""
    if isinstance(node, dict):
        return {key: _without_required(value) for key, value in node.items() if key != "required"}
    return [_without_required(item) for item in node] if isinstance(node, list) else node


@pytest.fixture(autouse=True)
def _every_compiled_program_meets_the_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    import konfai.portable as portable
    from konfai.portable_schema import program_schema

    strict = program_schema()
    lenient = _without_required(strict)
    marker = {"op": "expand"}  # what compile_chain leaves for _compile_expand to split at: no program carries it

    def checked(name: str, alias: str | None, picks: Callable[[Any], list[tuple[Any, str, dict]]]) -> None:
        original = getattr(portable, name)

        def compiled(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            for fragment, definition, schema in picks(result):
                _meets(fragment, definition, schema)
            return result

        monkeypatch.setattr(portable, name, compiled)
        if alias is not None:
            monkeypatch.setattr(sys.modules[__name__], alias, compiled)

    checked("compile_transform", None, lambda program: [(program, "transform", strict)])
    checked("compile_prediction", None, lambda program: [(program, "prediction", strict)])
    checked(
        "compile_chain",
        "steps",
        lambda pair: [([s for s in pair[0] if s != marker], "chain", strict), (pair[1], "destination", strict)],
    )
    checked("compile_pipeline", None, lambda triple: [(triple[0], "chain", strict)])
    checked("transform_manifest", "transform_manifest", lambda triple: [(triple[0], "manifest", lenient)])
    checked("assemble_program", "assemble_program", lambda program: [(program, "program", lenient)])


@_needs_the_runtime
def test_the_committed_program_schema_is_what_the_code_generates() -> None:
    # The runtime's tests validate their fixtures against this copy; `python -m konfai.portable_schema
    # konfai-edge/tests/fixtures/program_schema.json` rewrites it after a change to the contract.
    from konfai.portable_schema import program_schema

    assert json.loads(_SCHEMA_COPY.read_text()) == program_schema()


@_needs_the_runtime
def test_the_published_mrseg_program_meets_the_schema() -> None:
    _meets(json.loads(_MRSEG_PROGRAM.read_text()), "program", json.loads(_SCHEMA_COPY.read_text()))


@_needs_the_runtime
def test_the_schema_is_valid_draft_2020_12_and_closed_on_the_keys_the_ops_read() -> None:
    from konfai.portable_schema import program_schema

    schema = program_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    program = json.loads(_MRSEG_PROGRAM.read_text())
    _meets(program, "program", schema)
    # A key no op reads is what both refuse: the runtime by name, the schema by its closed objects.
    program["steps"][0]["manifest"]["preprocessing"][0]["value"] = 0.0
    with pytest.raises(jsonschema.ValidationError):
        _meets(program, "program", schema)


def test_a_stage_left_without_its_parameters_compiles_konfais_own_defaults() -> None:
    """A config that leaves a key out binds the transform's default (audit of 2026-09-26, P-09 and P-10): the
    compiler reads it off konfai's signature rather than a copy of its own."""
    from konfai.data import transform as stages
    from konfai.portable import OP_MAP

    unnormalize = OP_MAP["UnNormalize"]({})
    signature = inspect.signature(stages.UnNormalize).parameters
    assert unnormalize == {
        "op": "unnormalize",
        "min_value": float(signature["min_value"].default),
        "max_value": float(signature["max_value"].default),
    }
    assert OP_MAP["Dilate"]({}) == {
        "op": "dilate",
        "radius": inspect.signature(stages.Dilate).parameters["dilate"].default,
    }
    padding = inspect.signature(stages.Padding).parameters["padding"].default
    assert OP_MAP["Padding"]({}) == {"op": "pad", "padding": list(padding), "mode": "constant"}
    assert OP_MAP["Padding"]({})["padding"] == [0, 0, 0, 0, 0, 0], "konfai pads nothing by default"


def test_every_op_the_compiler_emits_is_closed_on_the_contracts_table() -> None:
    """`_admitted` ties the emitters to `portable_schema.CHAIN_OPS`, the table the runtime's contract test
    reads: an emitted key outside it is the compiler's error."""
    from konfai.portable import _admitted

    assert _admitted({"op": "cast", "dtype": "float32"}) == {"op": "cast", "dtype": "float32"}
    with pytest.raises(AssertionError, match="outside the contract"):
        _admitted({"op": "cast", "dtype": "float32", "inverse": True})
    with pytest.raises(AssertionError, match="missing"):
        _admitted({"op": "one_hot"})


def test_the_compiler_imports_without_pyyaml() -> None:
    # PyYAML is no dependency of konfai: the compiler reads the resolved config with ruamel.yaml.
    import os
    import subprocess
    import sys

    code = "import sys; sys.modules['yaml'] = None; import konfai.portable"
    root = str(Path(__file__).resolve().parents[2])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([root, os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 0, result.stderr


def test_the_output_chain_reads_the_dtype_an_apps_fold_leaves() -> None:
    # A Fold leaving uint8 labels: the copies' Mean reads them as float32 and their Vote keeps them, so konfai's
    # Resample undoes the first linearly and the second by the nearest voxel.
    for reduction, interpolation in (("Mean", "linear"), ("Vote", "nearest")):
        config = _prediction({"Resample": {"spacing": [1, 1, 1], "inverse": True}}, {"reduction": reduction})
        manifest, _folds, _per_copy = transform_manifest(config, "Predictor", blended="uint8")
        [undone] = [step for step in manifest["postprocessing"] if step["op"] == "inverse"]
        assert undone["interpolation"] == interpolation

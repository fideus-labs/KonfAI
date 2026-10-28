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
import re
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any

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
        ({"spacing": [1.5, 1.5, 1.5], "shape": [64, 64, 64]}, "both"),
        ({"spacing": [1.5, 1.5, 1.5], "interpolation": "bspline"}, "bspline"),
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
    chain = {
        "TensorCast": {"dtype": "float32", "inverse": False},
        "Resample": {"spacing": [1.5, 1.5, 1.5], "inverse": True, **(resample or {})},
    }
    config = {"Predictor": {"Dataset": _input_group(chain), "outputs_dataset": {"H": {"OutputDataset": outputs}}}}
    manifest, _, _ = transform_manifest(config, "Predictor")
    return manifest


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
    [("2G", 2 * 1024**3), ("512M", 512 * 1024**2), ("1024K", 1024**2), (4096, 4096), ("auto", None), (None, None)],
)
def test_a_budget_reaches_the_runtime_in_bytes(written: object, expected: int | None) -> None:
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
        {"Save": {"dataset": "./Work:nii.gz"}},
        {"Crop": {"inverse": True}},
        {"Clip": {"min_value": 0.0, "max_value": 400.0}},
        _WRITE,
    ]
    ops, _ = steps(chain, "CT")
    assert [op["op"] for op in ops] == ["save", "crop", "clip"]
    assert ops[1] == {"op": "crop"}


@pytest.mark.parametrize(
    "before",
    [
        {"Clip": {"min_value": 0.0, "max_value": 400.0}},
        {"TensorCast": {"dtype": "int16"}},
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
    import inspect

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
    "Canonical": {"inverse": False},
    "Crop": {"inverse": False},
    "Clip": {"min_value": 3.0, "max_value": "max", "mask": "MASK"},
    "TensorCast": {"dtype": "int16", "inverse": False},
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
    },
    "Padding": {"padding": [2, 2, 2, 2, 2, 2], "mode": "constant:3", "inverse": False},
    "Dilate": {"dilate": 3},
    "Flip": {"dims": "0", "inverse": False},
    "Permute": {"dims": "2|0|1", "inverse": False},
    "Standardize": {"mean": 3.0, "std": 2.0, "mask": "MASK", "inverse": False},
    "Normalize": {"min_value": 0.0, "max_value": 2.0, "inverse": False},
    "UnNormalize": {"min_value": 0.0, "max_value": 2.0},
    "SelectLabel": {"labels": ["(1,2)"]},
    "FlatLabel": {"labels": [1]},
    "OneHot": {"num_classes": 3, "inverse": False},
    "Gradient": {"per_dim": True},
    "HistogramMatching": {"reference_group": "S"},
    "Percentage": {"baseline": 2.0},
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
    "Normalize": {"lazy": True, "channels": [0]},
    "Resample": {"transforms": "T"},
    "Write": {"scale_factors": [2]},
    "Save": {"scale_factors": [2]},
    _AUG + "Brightness": {"groups": ["CT"]},
}

#: A value that leaves the compiled step as it is, and why konfai's own values do not depend on it here.
IGNORED_PARAMETERS: dict[str, dict[str, tuple[object, str]]] = {
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
        config = _prediction({}, {})
        config["Predictor"]["Dataset"]["augmentations"] = {"T": {"nb": 2, "data_augmentations": {name: params}}}
        return tta_passes(config, "Predictor")
    if name in ("Write", "Save"):
        return steps([{name: params}], "G")[1]
    if name in OP_MAP and name not in PROGRAM_TRANSFORMS:
        return compile_pipeline({name: params}, invertible=True)[0]
    return steps([{"Expand": {"nb": 2}}, {name: params}, _WRITE], "G")[0]


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


def _prediction(transforms: dict, outputs: dict, combine: str = "Mean", flips: int = 0) -> dict:
    """A Predictor section; ``flips`` test-time draws that flip every axis."""
    chain = {"TensorCast": {"dtype": "float32", "inverse": False}, **transforms}
    dataset = _input_group(chain)
    if flips:
        dataset["augmentations"] = {"T": {"nb": flips, "data_augmentations": {"Flip": {"f_prob": [1, 1]}}}}
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
        ({}, {"final_transforms": _MASK}, "Mask"),
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
    """TotalSegmentator's shape: each model decides a uint8 class on the canonical 1.5 mm grid, the models
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
    config = _prediction({"Clip#2": {"min_value": 0, "max_value": 1}}, {}, flips=1)
    config["Predictor"]["Dataset"]["augmentations"]["T"]["data_augmentations"] = {"Flip#2": {"f_prob": [1, 1]}}
    manifest, folds, _ = transform_manifest(config, "Predictor")
    assert [step["op"] for step in manifest["preprocessing"]] == ["cast", "clip"]
    assert not folds
    assert tta_passes(config, "Predictor") == [[], [{"op": "flip", "axes": [0]}]]


def test_a_prediction_program_applies_no_mask(tmp_path: Path) -> None:
    # compile_prediction exports with this default: no assembler applies a before-reduction Mask after it.
    config = _prediction({}, {"before_reduction_transforms": _MASK})
    with pytest.raises(ConfigError, match="transform 'Mask'"):
        export_checkpoints(config, "Predictor", [None], tmp_path, ["model"], patch_size=[4, 4], in_channels=1)


def test_a_prediction_config_and_its_checkpoints_compile_to_a_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A real Prediction.yml over the tests' tiny segmentation model and two of its checkpoints: the predictor
    # builds, each checkpoint exports beside the program, and the autouse check holds the program to the schema.
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    import numpy as np
    import torch
    from konfai.portable import compile_prediction
    from konfai.utils.dataset import Attribute, Dataset

    assets = Path(__file__).resolve().parents[1] / "assets" / "Workflows"
    monkeypatch.syspath_prepend(str(assets))
    monkeypatch.chdir(tmp_path)
    for case in ("CASE_000", "CASE_001"):
        Dataset(str(tmp_path / "Dataset"), "mha").write("MR", case, np.ones((1, 1, 16, 16), np.float32), Attribute())
    from TinySeg import TinySegNet

    checkpoints = [tmp_path / f"fold_{index}.pt" for index in range(2)]
    for checkpoint in checkpoints:
        torch.save({"Model": TinySegNet().network_states()}, checkpoint)
    group = {"transforms": "None", "patch_transforms": "None", "is_input": True}
    config = tmp_path / "Prediction.yml"
    config.write_text(
        json.dumps(
            {
                "Predictor": {
                    "Model": {"classpath": "TinySeg:TinySegNet", "TinySegNet": {"outputs_criterions": "None"}},
                    "autocast": True,
                    "Dataset": {
                        "groups_src": {"MR": {"groups_dest": {"MR": group}}},
                        "augmentations": "None",
                        "Patch": {"patch_size": [1, 16, 16], "overlap": 0},
                        "dataset_filenames": ["./Dataset:a:mha"],
                        "num_workers": 0,
                    },
                    "outputs_dataset": {
                        "Head:Softmax": {
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
    assert [step["op"] for step in steps[2:]] == ["mean", "argmax", "cast"]


def _reducing_config(tmp_path: Path, transforms: str) -> Path:
    """A Transform.yml over two small MetaImage cases whose one chain runs ``transforms``."""
    import numpy as np
    import SimpleITK as sitk

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
    config = _prediction({}, {}, flips=1)
    config["Predictor"]["Dataset"]["augmentations"]["T"]["data_augmentations"] = chain
    with pytest.raises(ConfigError, match=refused):
        tta_passes(config, "Predictor")


def test_a_vector_field_draw_negates_the_component_of_every_mirrored_axis() -> None:
    # konfai's `Flip(vector_field=True)` mirrors the axes and negates each one's component channel, and
    # the draw is its own inverse, so the copy's inverse negates them again.
    config = _prediction({}, {}, flips=1)
    config["Predictor"]["Dataset"]["augmentations"]["T"]["data_augmentations"] = {
        "Flip": {"f_prob": [1.0, 1.0, 1.0], "vector_field": True}
    }
    passes = tta_passes(config, "Predictor")
    assert passes == [[], [{"op": "flip", "axes": [0], "vector_field": True}]]
    copy = assemble_program(_one_model(), reduce="mean", passes=passes)["steps"][1]["manifest"]
    assert copy["preprocessing"][-1] == {"op": "flip", "axes": [0], "vector_field": True, "id": "__tta_0"}
    assert copy["postprocessing"][0] == {"op": "inverse", "of": "__tta_0", "vector_field": True}


@pytest.mark.parametrize("as_list", [False, True])
def test_portable_tta_composes_every_flip_in_a_draw(as_list: bool) -> None:
    import torch

    config = _prediction({}, {}, flips=1)
    first = {"f_prob": [0.0, 0.0, 1.0]}
    second = {"f_prob": [1.0, 1.0, 0.0]}
    chain = [{"Flip": first}, {"Flip": second}] if as_list else {"Flip": first, "Flip#2": second}
    config["Predictor"]["Dataset"]["augmentations"]["T"]["data_augmentations"] = chain
    passes = tta_passes(config, "Predictor")
    # Both flips apply, in the chain's order, and are undone.
    image = torch.arange(24).reshape(1, 2, 3, 4)
    expected = torch.flip(torch.flip(image, [3]), [1])
    actual = image
    for op in passes[1]:
        actual = torch.flip(actual, [axis + 1 for axis in op["axes"]])
    assert passes[0] == []
    assert torch.equal(actual, expected)


def test_portable_tta_keeps_identity_copies_in_the_reduction() -> None:
    config = _prediction({}, {}, flips=3)
    config["Predictor"]["Dataset"]["augmentations"]["T"]["data_augmentations"] = {}
    assert tta_passes(config, "Predictor") == [[], [], [], []]


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


@pytest.mark.parametrize("size", [[0, 16, 16], [16, 0, 16], [16, 16, -1]])
def test_an_onnx_export_refuses_free_patch_axes_instead_of_dropping_them(size: list[int]) -> None:
    from konfai.portable import derive_onnx_params

    config = {"Predictor": {"Dataset": {"Patch": {"patch_size": size}}}}
    with pytest.raises(ConfigError, match=r"patch_size.*free axis"):
        derive_onnx_params(config, "Predictor")


@pytest.mark.parametrize(("prob", "copies"), [(0.0, [[], []]), (1.0, [[0], [2]])])
def test_portable_tta_reads_the_resolved_draw_gate(prob: float, copies: list[list[int]]) -> None:
    config = _prediction({}, {}, flips=2)
    chain = {"Flip": {"f_prob": [1.0, 0.0, 1.0], "prob": prob, "vector_field": False}}
    config["Predictor"]["Dataset"]["augmentations"]["T"]["data_augmentations"] = chain
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
    config = _prediction({}, {}, flips=nb)
    config["Predictor"]["Dataset"]["augmentations"]["T"]["data_augmentations"] = {"Flip": {"f_prob": f_prob}}
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


def test_a_draw_naming_the_groups_it_draws_is_refused_by_name() -> None:
    # konfai draws the named groups alone; the runtime draws the chain the draw is written in.
    from konfai.portable import compile_chain

    with pytest.raises(KonfAIError, match="Brightness"):
        compile_chain(
            [
                {"Expand": {"nb": 2}},
                {"Brightness": {"b_std": 0.1, "groups": ["CT"]}},
                {"Write": {"dataset": "./Out:mha"}},
            ],
            "CT",
        )


def test_a_clip_that_records_its_bound_is_refused_by_name() -> None:
    # konfai's Clip(save_clip_min) writes its bound as the volume's Min, and a later Normalize takes that
    # statistic over the volume's own minimum; the runtime measures the minimum, so the values differ.
    from konfai.portable import compile_pipeline

    with pytest.raises(KonfAIError, match="save_clip_min"):
        compile_pipeline({"Clip": {"min_value": 0, "max_value": 1, "save_clip_min": True}}, fold=False)
    steps, _folds, _inverted = compile_pipeline(
        {"Clip": {"min_value": 0, "max_value": 1, "save_clip_min": False, "save_clip_max": False}}, fold=False
    )
    assert steps == [{"op": "clip", "min_value": 0.0, "max_value": 1.0}]


def test_a_permute_draw_with_no_shares_gives_each_of_two_copies_its_own_order() -> None:
    # konfai's Permute(prob_permute=None) draws torch.eye(2): the first copy takes the first order alone,
    # the second the second, and any other copy count is refused.
    config = _prediction({}, {}, flips=1)
    config["Predictor"]["Dataset"]["augmentations"]["T"] = {
        "nb": 2,
        "data_augmentations": {"Permute": {"prob_permute": None}},
    }
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
        ({"outputs_dataset": {"H": {"OutputDataset": {"after_reduction_transforms": _STACK}}}}, "InferenceStack"),
    ],
)
def test_a_prediction_key_the_program_cannot_carry_is_warned_by_key(section: dict, key: str) -> None:
    # konfai reads these keys and the program has no counterpart: a header attribute, the stacked members a
    # stack writes, a setting that changes how konfai runs. Each is said by key, so the
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

    config = _prediction({}, {}, flips=2)
    config["Predictor"]["Dataset"]["augmentations"]["T"]["data_augmentations"] = draw
    with pytest.warns(KonfAIWarning, match=re.escape(key)):
        tta_passes(config, "Predictor")
    # A test-time Flip's f_prob only names the axes its mirrors take: konfai draws nothing from it.
    config["Predictor"]["Dataset"]["augmentations"]["T"]["data_augmentations"] = {"Flip": {"f_prob": [0.5, 0, 1]}}
    with warnings.catch_warnings():
        warnings.simplefilter("error", KonfAIWarning)
        tta_passes(config, "Predictor")


# The program contract: every program these tests compile meets the schema written once in
# `konfai.portable_schema`, and the copy committed beside the runtime's fixtures is that schema.

_SCHEMA_COPY = Path(__file__).parents[2] / "konfai-edge" / "tests" / "fixtures" / "program_schema.json"
_MRSEG_PROGRAM = _SCHEMA_COPY.with_name("mrseg_program.json")
# The public tree ships the compiler without the runtime; these checks pair the two where both are present.
_needs_the_runtime = pytest.mark.skipif(not _SCHEMA_COPY.exists(), reason="the runtime's fixtures are not in this tree")

_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _meets(instance: object, node: dict, schema: dict, at: str = "$") -> None:
    """Assert ``instance`` meets ``node`` of ``schema``: the subset of draft 2020-12 the schema spells."""
    if "$ref" in node:
        node = schema["$defs"][node["$ref"].rsplit("/", 1)[-1]]
    if "oneOf" in node:
        fits = [alternative for alternative in node["oneOf"] if _fits(instance, alternative, schema)]
        assert len(fits) == 1, f"{at}: {instance!r} fits {len(fits)} of the alternatives"
        return
    types = node.get("type", [])
    accepted = tuple(_TYPES[name] for name in ([types] if isinstance(types, str) else types))
    if accepted:
        assert isinstance(instance, accepted) and not (isinstance(instance, bool) and bool not in accepted), (
            f"{at}: {instance!r} is no {types}"
        )
    assert "const" not in node or instance == node["const"], f"{at}: {instance!r} is not {node['const']!r}"
    assert "enum" not in node or instance in node["enum"], f"{at}: {instance!r} is not one of {node['enum']}"
    assert "minimum" not in node or instance >= node["minimum"], f"{at}: {instance!r} is below {node['minimum']}"  # type: ignore[operator]
    assert "maximum" not in node or instance <= node["maximum"], f"{at}: {instance!r} is above {node['maximum']}"  # type: ignore[operator]
    if isinstance(instance, dict):
        for key in node.get("required", []):
            assert key in instance, f"{at}: no {key!r}"
        for key, value in instance.items():
            if key in node.get("properties", {}):
                _meets(value, node["properties"][key], schema, f"{at}.{key}")
            elif node.get("additionalProperties", True) is False:
                raise AssertionError(f"{at}: {key!r} is no key of this object")
            elif isinstance(node.get("additionalProperties"), dict):
                _meets(value, node["additionalProperties"], schema, f"{at}.{key}")
    if isinstance(instance, list):
        assert len(instance) >= node.get("minItems", 0) and len(instance) <= node.get("maxItems", len(instance)), at
        for index, item in enumerate(instance):
            _meets(item, node.get("items", {}), schema, f"{at}[{index}]")


def _fits(instance: object, node: dict, schema: dict) -> bool:
    try:
        _meets(instance, node, schema)
    except AssertionError:
        return False
    return True


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
                _meets(fragment, {"$ref": f"#/$defs/{definition}"}, schema, definition)
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
    _meets(json.loads(_MRSEG_PROGRAM.read_text()), {"$ref": "#/$defs/program"}, json.loads(_SCHEMA_COPY.read_text()))


@_needs_the_runtime
def test_the_schema_is_valid_draft_2020_12_and_the_check_here_agrees_with_it() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    from konfai.portable_schema import program_schema

    schema = program_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    program = json.loads(_MRSEG_PROGRAM.read_text())
    jsonschema.Draft202012Validator({**schema, "$ref": "#/$defs/program"}).validate(program)
    # A key no op reads is what both refuse: the runtime by name, the schema by its closed objects.
    program["steps"][0]["manifest"]["preprocessing"][0]["value"] = 0.0
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator({**schema, "$ref": "#/$defs/program"}).validate(program)
    with pytest.raises(AssertionError):
        _meets(program, {"$ref": "#/$defs/program"}, schema)


def test_a_stage_left_without_its_parameters_compiles_konfais_own_defaults() -> None:
    """A config that leaves a key out binds the transform's default (audit of 2026-09-26, P-09 and P-10): the
    compiler reads it off konfai's signature rather than a copy of its own."""
    import inspect

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

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

"""`assemble_program`: model steps and one reduction over named buffers."""

import json

import pytest
from konfai.portable import assemble_program
from konfai.utils.errors import ConfigError

M = {"input": {"channels": 1}, "output": {"channels": 3}, "patch": {"size": [8, 8], "overlap": [0, 0]}}

# A same-class ensemble fold: per-volume resample to 1.5mm, Gaussian-blended tiled forward emitting raw
# logits, then the predictor's own order: undo the resample, then decide a class.
MR = {
    "input": {"channels": 1},
    "output": {"channels": 41},
    "patch": {"size": [96, 128, 160], "overlap": [32, 32, 32], "pad_value": 0},
    "preprocessing": [
        {"op": "cast", "dtype": "float32"},
        {"op": "standardize"},
        {"op": "resample", "spacing": [1.5, 1.5, 1.5], "id": "Resample", "interpolation": "linear"},
    ],
    "postprocessing": [
        {"op": "inverse", "of": "Resample", "interpolation": "linear"},
        {"op": "softmax", "dim": 0},
        {"op": "argmax", "dim": 0},
        {"op": "cast", "dtype": "uint8"},
    ],
    "blend": "Gaussian",
}

# The same fold with a padded working grid: the pad's inverse only drops rows, so it stays per fold.
MR_PADDED = {
    **MR,
    "preprocessing": [*MR["preprocessing"], {"op": "pad", "padding": [32] * 6, "mode": "constant", "id": "Padding"}],
    "postprocessing": [{"op": "inverse", "of": "Padding"}, *MR["postprocessing"]],
}


def test_single_model_is_a_one_step_program():
    program = assemble_program([{"id": "a", "manifest": M}], reduce="mean")
    assert program == {"steps": [{"model": "a", "in": "input", "out": "output", "manifest": M}], "output": "output"}


def test_ensemble_merge_labels():
    two = {**M, "output": {"channels": 2}}
    program = assemble_program([{"id": "m291", "manifest": M}, {"id": "m292", "manifest": two}], reduce="merge_labels")
    assert program["output"] == "output"
    assert [s["model"] for s in program["steps"][:2]] == ["m291", "m292"]
    assert program["steps"][0]["in"] == "input" and program["steps"][0]["out"] == "m291"
    assert program["steps"][-1] == {"op": "merge_labels", "in": ["m291", "m292"], "out": "output", "classes": [3, 2]}
    assert json.loads(json.dumps(program)) == program  # round-trips as JSON


def test_explicit_classes_override_the_per_model_ones():
    program = assemble_program(
        [{"id": "a", "manifest": M}, {"id": "b", "manifest": M}], reduce="merge_labels", classes=[5, 7]
    )
    assert program["steps"][-1]["classes"] == [5, 7]


def test_mean_reduction_carries_no_classes():
    program = assemble_program([{"id": "a", "manifest": M}, {"id": "b", "manifest": M}], reduce="mean")
    assert program["steps"][-1] == {"op": "mean", "in": ["a", "b"], "out": "output"}


def test_unknown_reduction_is_refused():
    with pytest.raises(ConfigError, match="unknown reduction"):
        assemble_program([{"id": "a", "manifest": M}, {"id": "b", "manifest": M}], reduce="nope")


def test_no_models_is_refused():
    with pytest.raises(ConfigError, match="at least one model"):
        assemble_program([], reduce="mean")


def test_same_class_ensemble_hoists_the_run_once_tail_in_the_chains_order():
    # 5 folds mean-reduced: folds emit raw logits, and everything the predictor runs once moves past
    # the reduction IN THE CHAIN'S ORDER. The inverse resample comes first, so the class is decided on
    # the grid the caller gave and no label map is ever resampled.
    models = [{"id": f"fold_{i}", "manifest": MR} for i in range(5)]
    program = assemble_program(models, reduce="mean")

    fold_steps = program["steps"][:5]
    for i, step in enumerate(fold_steps):
        assert step["model"] == f"fold_{i}" and step["in"] == "input" and step["out"] == f"fold_{i}"
        assert step["manifest"]["postprocessing"] == []  # every post op is a run-once one here

    tail = program["steps"][5:]
    # A cast sets the dtype the output is written in, so it stays a tail op.
    assert [t["op"] for t in tail] == ["mean", "resample_linear", "softmax", "argmax", "cast"]
    assert tail[0]["in"] == [model["id"] for model in models] and tail[0]["out"] == "r"
    # Float logits, so the inverse interpolates; it never picks a neighbour of a label map.
    assert tail[1] == {"op": "resample_linear", "in": ["r", "input"], "out": "t0"}
    assert tail[2] == {"op": "softmax", "in": ["t0"], "out": "t1"}
    assert tail[3] == {"op": "argmax", "in": ["t1"], "out": "t2"}
    assert tail[4] == {"op": "cast", "dtype": "uint8", "in": ["t2"], "out": "output"}
    assert program["output"] == "output"
    assert json.loads(json.dumps(program)) == program  # round-trips as JSON


def test_models_folded_first_run_the_per_copy_ops_on_each_copys_mean():
    # konfai's combine: Mean averages a copy's models, then runs its before-reduction chain on the average.
    synth = {
        **M,
        "preprocessing": [{"op": "resample", "spacing": [1.0, 1.0], "id": "Resample", "interpolation": "linear"}],
        "postprocessing": [
            {"op": "unnormalize", "min_value": -1024.0, "max_value": 3071.0},
            {"op": "cast", "dtype": "int16"},
            {"op": "inverse", "of": "Resample", "interpolation": "linear"},
        ],
    }
    models = [{"id": f"cv_{i}", "manifest": synth} for i in range(2)]
    flip = [{"op": "flip", "axes": [1]}]
    program = assemble_program(models, reduce="mean", passes=[[], flip], per_copy=2, fold_models=True)
    steps = [(step.get("op", "model"), step["in"], step["out"]) for step in program["steps"]]
    assert steps == [
        ("model", "input", "cv_0__p0"),
        ("model", "input", "cv_1__p0"),
        ("model", "input", "cv_0__p1"),
        ("model", "input", "cv_1__p1"),
        ("mean", ["cv_0__p0", "cv_1__p0"], "mean__p0"),
        ("unnormalize", ["mean__p0"], "mean__p0__0"),
        ("cast", ["mean__p0__0"], "mean__p0__1"),
        ("mean", ["cv_0__p1", "cv_1__p1"], "mean__p1"),
        ("unnormalize", ["mean__p1"], "mean__p1__0"),
        ("cast", ["mean__p1__0"], "mean__p1__1"),
        ("mean", ["mean__p0__1", "mean__p1__1"], "r"),
        ("resample_linear", ["r", "input"], "output"),
    ]
    # Each model keeps only the undoing of its own copy.
    assert program["steps"][0]["manifest"]["postprocessing"] == []
    assert program["steps"][2]["manifest"]["postprocessing"] == [{"op": "inverse", "of": "__tta_0"}]


def test_an_index_selection_inverse_stays_per_fold():
    # Dropping a padded border commutes with an elementwise mean, so each fold crops its own buffer and
    # the reduction runs on the smaller grid.
    models = [{"id": f"fold_{i}", "manifest": MR_PADDED} for i in range(3)]
    program = assemble_program(models, reduce="mean")
    for step in program["steps"][:3]:
        assert step["manifest"]["postprocessing"] == [{"op": "inverse", "of": "Padding"}]
    assert [t["op"] for t in program["steps"][3:]] == ["mean", "resample_linear", "softmax", "argmax", "cast"]


def test_a_non_resample_inverse_past_the_reduction_is_refused():
    # Only a resample onto the caller's grid is expressible as a buffer op; anything else is named.
    bad = {**MR, "postprocessing": [{"op": "softmax", "dim": 0}, {"op": "inverse", "of": "Padding"}]}
    bad = {**bad, "preprocessing": [*MR["preprocessing"], {"op": "pad", "padding": [1] * 6, "id": "Padding"}]}
    with pytest.raises(ConfigError, match="cannot hoist the inverse of"):
        assemble_program([{"id": "a", "manifest": bad}, {"id": "b", "manifest": bad}], reduce="mean")


def test_a_hoisted_inverse_reads_the_interpolation_its_step_names():
    # The compiler writes on the inverse what konfai reads it with; the hoist emits that, whatever ran before.
    inverse = {"op": "inverse", "of": "Resample", "interpolation": "nearest"}
    manifest = {**MR, "postprocessing": [{"op": "argmax", "dim": 0}, inverse]}
    program = assemble_program(
        [{"id": "a", "manifest": manifest}, {"id": "b", "manifest": manifest}], reduce="mean", per_copy=1
    )
    assert [t["op"] for t in program["steps"][2:]] == ["mean", "resample_nearest"]
    for step in program["steps"][:2]:
        assert step["manifest"]["postprocessing"] == [{"op": "argmax", "dim": 0}]


@pytest.mark.parametrize(
    ("target", "placed"),
    [
        ({"spacing": [1.5, 1.5, 1.5]}, {"align": "origin", "spacing": [1.5, 1.5, 1.5]}),
        ({"shape": [64, 0, 64]}, {"align": "origin"}),
    ],
)
def test_a_hoisted_inverse_of_a_resample_aligned_on_the_origin_carries_its_placement(target, placed):
    # Aligned on the origin, the folds' grid is not the one an extent of the input's box gives back: the op
    # carries the align, and the spacing a count does not give back (konfai-edge tests/program.rs).
    resample = {"op": "resample", **target, "align": "origin", "id": "Resample", "interpolation": "linear"}
    manifest = {**MR, "preprocessing": [*MR["preprocessing"][:2], resample]}
    program = assemble_program([{"id": "a", "manifest": manifest}, {"id": "b", "manifest": manifest}], reduce="mean")
    assert program["steps"][3] == {"op": "resample_linear", **placed, "in": ["r", "input"], "out": "t0"}


@pytest.mark.parametrize(
    ("align", "placed"),
    [
        ({"align": "origin"}, {"align": "origin", "spacing": [1.5, 1.5, 1.5], "fill": -1000.0}),
        ({}, {}),
    ],
)
def test_a_hoisted_inverse_carries_the_fill_only_an_origin_alignment_reads(align, placed):
    # Aligned on the origin, the input's voxels past the folds' far face read outside them and konfai writes
    # the stage's fill there (konfai-edge tests/program.rs); an extent grid covers every input voxel, so its
    # op compiles as it did before.
    resample = {**MR["preprocessing"][2], **align, "fill": -1000.0}
    undo = {**MR["postprocessing"][0], "fill": -1000.0}
    manifest = {
        **MR,
        "preprocessing": [*MR["preprocessing"][:2], resample],
        "postprocessing": [undo, *MR["postprocessing"][1:]],
    }
    program = assemble_program([{"id": "a", "manifest": manifest}, {"id": "b", "manifest": manifest}], reduce="mean")
    assert program["steps"][3] == {"op": "resample_linear", **placed, "in": ["r", "input"], "out": "t0"}


def test_a_nested_resample_aligned_on_the_origin_is_refused_by_name():
    # The program puts a nested model's volume on the copies' grid from the two extents alone.
    resample = {"op": "resample", "spacing": [1.5, 1.5], "align": "origin", "interpolation": "nearest"}
    nested = {"id": "aux", "manifest": M, "ops": [resample]}
    with pytest.raises(ConfigError, match="aligns on 'origin'"):
        assemble_program(
            [{"id": "m", "manifest": M}],
            reduce="mean",
            aux={"aux": nested},
            masks=[{"group": "aux", "value_outside": 0.0}],
        )


def test_a_nested_resample_through_a_field_is_refused_by_name():
    # The program puts a nested model's volume on the copies' grid by the two extents alone: a field would be
    # dropped from it.
    resample = {"op": "resample", "spacing": [1.5, 1.5], "field_group": "DVF", "interpolation": "nearest"}
    nested = {"id": "aux", "manifest": M, "ops": [resample]}
    with pytest.raises(ConfigError, match="warps through the field in 'DVF'"):
        assemble_program(
            [{"id": "m", "manifest": M}],
            reduce="mean",
            aux={"aux": nested},
            masks=[{"group": "aux", "value_outside": 0.0}],
        )


def test_a_hoisted_inverse_of_a_resample_through_a_field_reads_no_field():
    # konfai undoes a Resample through a field by a resample back through no map (`_resample_between`), so the
    # hoisted op is the one a plain resample of the same grid compiles to.
    plain = MR["preprocessing"][2]
    warped = {**plain, "field_group": "DVF"}
    programs = [
        assemble_program(
            [
                {"id": name, "manifest": {**MR, "preprocessing": [*MR["preprocessing"][:2], resample]}}
                for name in ("a", "b")
            ],
            reduce="mean",
        )
        for resample in (plain, warped)
    ]
    assert programs[0]["steps"][2:] == programs[1]["steps"][2:]


def test_hoisting_does_not_mutate_the_input_manifest():
    before = json.dumps(MR, sort_keys=True)
    assemble_program([{"id": f"f{i}", "manifest": MR} for i in range(3)], reduce="mean")
    assert json.dumps(MR, sort_keys=True) == before, "assemble_program must not mutate the caller's manifest"


def test_mean_ensemble_without_post_or_resample_has_no_tail():
    # A plain same-class ensemble (logits already final, no resample) reduces straight to output.
    program = assemble_program([{"id": "a", "manifest": M}, {"id": "b", "manifest": M}], reduce="mean")
    assert [s.get("op") for s in program["steps"] if "op" in s] == ["mean"]
    assert program["steps"][-1] == {"op": "mean", "in": ["a", "b"], "out": "output"}


def test_non_channel_axis_post_op_cannot_be_hoisted():
    bad = {**MR, "postprocessing": [{"op": "softmax", "dim": 1}]}
    with pytest.raises(ConfigError, match="channel axis"):
        assemble_program([{"id": "a", "manifest": bad}, {"id": "b", "manifest": bad}], reduce="mean")


def test_non_softmax_argmax_post_op_cannot_be_hoisted():
    # A per-voxel intensity op in the post can't cross the channel reduction, it belongs in each fold.
    bad = {**MR, "postprocessing": [{"op": "unnormalize", "min_value": -1, "max_value": 1}]}
    with pytest.raises(ConfigError, match="cannot hoist post op"):
        assemble_program([{"id": "a", "manifest": bad}, {"id": "b", "manifest": bad}], reduce="mean")


def test_merge_labels_keeps_per_fold_post():
    # A disjoint ensemble does NOT hoist: each fold argmaxes to its own label map, undoes the resample
    # nearest, then merge_labels tiles.
    post = [*MR["postprocessing"][1:], {**MR["postprocessing"][0], "interpolation": "nearest"}]
    labels = {**MR, "postprocessing": post}
    models = [{"id": "a", "manifest": labels}, {"id": "b", "manifest": {**labels, "output": {"channels": 2}}}]
    program = assemble_program(models, reduce="merge_labels", per_copy=3)
    for step in program["steps"][:2]:
        assert step["manifest"]["postprocessing"] == post  # untouched
    assert program["steps"][-1] == {"op": "merge_labels", "in": ["a", "b"], "out": "output", "classes": [41, 2]}


def test_a_stack_takes_each_copy_before_its_padding_is_undone_and_records_its_members():
    # konfai's InferenceStack (mode Seg) labels each copy before the case's inverses: the program takes each
    # row as its copy is written, folds the copies, undoes the Padding on the fold and stacks the rows.
    models = [{"id": "a", "manifest": MR_PADDED}, {"id": "b", "manifest": MR_PADDED}]
    program = assemble_program(models, reduce="mean", stack="Seg", checkpoints=["fold_0.pt", "fold_1.pt"])
    steps = [(step.get("model") or step["op"], step["out"]) for step in program["steps"]]
    assert steps == [
        ("a", "a"),
        ("argmax", "a__s0"),
        ("cast", "a__s1"),
        ("b", "b"),
        ("argmax", "b__s0"),
        ("cast", "b__s1"),
        ("mean", "r"),
        ("pad", "t0"),
        ("resample_linear", "t1"),
        ("softmax", "t2"),
        ("argmax", "t3"),
        ("cast", "output"),
        ("concat", "stack"),
    ]
    assert program["steps"][0]["manifest"]["postprocessing"] == []
    assert program["steps"][7]["padding"] == [-32] * 6
    assert program["stack"] == "stack"
    assert program["members"] == {
        "a": {"checkpoint": "fold_0.pt", "pass": 0},
        "b": {"checkpoint": "fold_1.pt", "pass": 0},
    }

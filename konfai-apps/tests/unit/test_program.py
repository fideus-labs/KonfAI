# SPDX-License-Identifier: Apache-2.0
"""Tests for `assemble_program` (Model steps + a reduction Op over named buffers)."""

import json

import pytest
from konfai_apps.bundle import AppMetadataError, assemble_program

M = {"input": {"channels": 1}, "output": {"channels": 3}, "patch": {"size": [8, 8], "overlap": [0, 0]}}

# A same-class ensemble fold: per-volume resample to 1.5mm (with inverse), Gaussian-blended tiled
# forward emitting raw logits, softmax/argmax post.
MR = {
    "input": {"channels": 1},
    "output": {"channels": 41},
    "patch": {"size": [96, 128, 160], "overlap": [32, 32, 32], "pad_value": 0},
    "preprocessing": [
        {"op": "cast", "dtype": "float32"},
        {"op": "standardize"},
        {"op": "resample", "spacing": [1.5, 1.5, 1.5], "inverse": True},
    ],
    "postprocessing": [{"op": "softmax", "dim": 0}, {"op": "argmax", "dim": 0}, {"op": "cast", "dtype": "uint8"}],
    "blend": "Gaussian",
}


def test_single_model_is_a_one_step_program():
    program = assemble_program([{"id": "a", "manifest": M}], reduce="mean")
    assert program == {"steps": [{"model": "a", "in": "input", "out": "output", "manifest": M}], "output": "output"}


def test_ensemble_merge_labels():
    program = assemble_program(
        [{"id": "m291", "manifest": M, "classes": 3}, {"id": "m292", "manifest": M, "classes": 2}],
        reduce="merge_labels",
    )
    assert program["output"] == "output"
    assert [s["model"] for s in program["steps"][:2]] == ["m291", "m292"]
    assert program["steps"][0]["in"] == "input" and program["steps"][0]["out"] == "m0"
    assert program["steps"][-1] == {"op": "merge_labels", "in": ["m0", "m1"], "out": "output", "classes": [3, 2]}
    assert json.loads(json.dumps(program)) == program  # round-trips as JSON


def test_explicit_classes_override_the_per_model_ones():
    program = assemble_program(
        [{"id": "a", "manifest": M}, {"id": "b", "manifest": M}], reduce="merge_labels", classes=[5, 7]
    )
    assert program["steps"][-1]["classes"] == [5, 7]


def test_mean_reduction_carries_no_classes():
    program = assemble_program([{"id": "a", "manifest": M}, {"id": "b", "manifest": M}], reduce="mean")
    assert program["steps"][-1] == {"op": "mean", "in": ["m0", "m1"], "out": "output"}


def test_unknown_reduction_is_refused():
    with pytest.raises(AppMetadataError, match="unknown reduction"):
        assemble_program([{"id": "a", "manifest": M}, {"id": "b", "manifest": M}], reduce="nope")


def test_no_models_is_refused():
    with pytest.raises(AppMetadataError, match="at least one model"):
        assemble_program([], reduce="mean")


def test_same_class_ensemble_hoists_post_and_inverse_resample():
    # 5 folds mean-reduced: folds emit raw logits, so softmax/argmax + inverse resample hoist to run
    # once after the reduction.
    models = [{"id": f"fold_{i}", "manifest": MR} for i in range(5)]
    program = assemble_program(models, reduce="mean")

    fold_steps = program["steps"][:5]
    for i, step in enumerate(fold_steps):
        assert step["model"] == f"fold_{i}" and step["in"] == "input" and step["out"] == f"m{i}"
        # Each fold emits logits: no post, and its resample is forward-only (the inverse is hoisted).
        assert step["manifest"]["postprocessing"] == []
        resamples = [s for s in step["manifest"]["preprocessing"] if s.get("op") == "resample"]
        assert resamples and all(s["inverse"] is False for s in resamples)

    tail = program["steps"][5:]
    # The dtype cast(uint8) in the fold post is a no-op in the float program -> skipped, not a tail op.
    assert [t["op"] for t in tail] == ["mean", "softmax", "argmax", "resample_nearest"]
    assert tail[0]["in"] == [f"m{i}" for i in range(5)] and tail[0]["out"] == "r"
    assert tail[1] == {"op": "softmax", "in": ["r"], "out": "t0"}
    assert tail[2] == {"op": "argmax", "in": ["t0"], "out": "t1"}
    # The inverse resample lands the label map back on the input grid (nearest, since we argmaxed).
    assert tail[3] == {"op": "resample_nearest", "in": ["t1", "input"], "out": "output"}
    assert program["output"] == "output"
    assert json.loads(json.dumps(program)) == program  # round-trips as JSON


def test_hoisting_does_not_mutate_the_input_manifest():
    before = json.dumps(MR, sort_keys=True)
    assemble_program([{"id": f"f{i}", "manifest": MR} for i in range(3)], reduce="mean")
    assert json.dumps(MR, sort_keys=True) == before, "assemble_program must not mutate the caller's manifest"


def test_mean_ensemble_without_post_or_resample_has_no_tail():
    # A plain same-class ensemble (logits already final, no resample) reduces straight to output.
    program = assemble_program([{"id": "a", "manifest": M}, {"id": "b", "manifest": M}], reduce="mean")
    assert [s.get("op") for s in program["steps"] if "op" in s] == ["mean"]
    assert program["steps"][-1] == {"op": "mean", "in": ["m0", "m1"], "out": "output"}


def test_non_channel_axis_post_op_cannot_be_hoisted():
    bad = {**MR, "postprocessing": [{"op": "softmax", "dim": 1}]}
    with pytest.raises(AppMetadataError, match="channel axis"):
        assemble_program([{"id": "a", "manifest": bad}, {"id": "b", "manifest": bad}], reduce="mean")


def test_non_softmax_argmax_post_op_cannot_be_hoisted():
    # A per-voxel intensity op in the post can't cross the channel reduction -- it belongs in each fold.
    bad = {**MR, "postprocessing": [{"op": "unnormalize", "min_value": -1, "max_value": 1}]}
    with pytest.raises(AppMetadataError, match="cannot hoist post op"):
        assemble_program([{"id": "a", "manifest": bad}, {"id": "b", "manifest": bad}], reduce="mean")


def test_merge_labels_keeps_per_fold_post():
    # A disjoint ensemble does NOT hoist: each fold argmaxes to its own label map, then merge_labels tiles.
    models = [{"id": "a", "manifest": MR, "classes": 3}, {"id": "b", "manifest": MR, "classes": 2}]
    program = assemble_program(models, reduce="merge_labels")
    for step in program["steps"][:2]:
        assert step["manifest"]["postprocessing"] == MR["postprocessing"]  # untouched
    assert program["steps"][-1] == {"op": "merge_labels", "in": ["m0", "m1"], "out": "output", "classes": [3, 2]}


def test_multi_checkpoint_nested_model_is_refused(tmp_path):
    # A nested mask/condition producer is copied as `<group>.onnx` + its `manifest.json`. Several
    # checkpoints would make the nested export emit `program.json` + per-fold onnx instead, so the
    # step refuses up front rather than failing later on a missing `model.onnx`.
    from konfai_apps.bundle import _export_nested_model

    inference = {"repo_id": "org/app", "model_name": "body", "checkpoints_name": ["a.pt", "b.pt"]}
    with pytest.raises(AppMetadataError, match="single-checkpoint nested model"):
        _export_nested_model(tmp_path, "MASK", inference)

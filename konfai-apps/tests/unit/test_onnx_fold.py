# SPDX-License-Identifier: Apache-2.0
"""Auto-fold: a POINTWISE torch transform is baked into the ONNX graph via ``fold_pre`` instead of
being refused; non-pointwise / non-torch transforms are still refused. Tests the bundle-side
detection + ordering."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))  # make _fold_fixture importable by classpath

from konfai.portable import transform_manifest, try_fold
from konfai.utils.errors import ConfigError


def _cfg(transforms):
    groups = {"V": {"groups_dest": {"g": {"is_input": True, "transforms": transforms}}}}
    return {"Predictor": {"Dataset": {"groups_src": groups}, "outputs_dataset": {}}}


def test_try_fold_accepts_pointwise_refuses_global_and_geometric():
    fold = try_fold("Clip", {"min_value": -10.0, "max_value": 40.0})
    assert fold is not None
    out = fold(torch.linspace(-50.0, 50.0, 21))
    assert out.min().item() == -10.0 and out.max().item() == 40.0
    assert try_fold("Standardize", {}) is None  # GLOBAL_STAT: needs whole-volume stats
    assert try_fold("Resample", {"spacing": [1, 1, 3]}) is None  # not pointwise


def test_custom_pointwise_transform_folds_instead_of_refusing():
    manifest, folds, _ = transform_manifest(_cfg({"_fold_fixture:DoubleThenBias": {"bias": 3.0}}), "Predictor")
    assert manifest["preprocessing"] == []  # folded into the graph, not a runtime op
    assert len(folds) == 1
    assert torch.allclose(folds[0](torch.ones(4)), torch.full((4,), 5.0))  # 2*1 + 3


def test_runtime_op_after_fold_is_refused():
    cfg = _cfg({"_fold_fixture:DoubleThenBias": {}, "Clip": {"min_value": -1, "max_value": 1}})
    with pytest.raises(ConfigError, match="follows a folded transform"):
        transform_manifest(cfg, "Predictor")


def test_non_foldable_transform_is_refused():
    # A real transform that is neither a runtime op nor pointwise (Reduce folds several cases into one);
    # a program-level transform such as KonfAIInference is instead compiled into a nested model step.
    with pytest.raises(ConfigError, match="not a foldable pointwise"):
        transform_manifest(_cfg({"Reduce": {}}), "Predictor")

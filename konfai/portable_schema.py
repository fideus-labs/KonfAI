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

"""The portable program contract, as a JSON Schema (draft 2020-12) written once and checked on both sides.

A document is a TRANSFORM program, a PREDICTION program, an EVALUATION program, a bundle's assembled
``program.json`` or a single model's ``manifest.json``. Every op a step may name, the keys it may carry and the ones it must, sit in
:data:`CHAIN_OPS` and :data:`BODY_OPS`: a key outside them is one no compiler writes, and the runtime
refuses it by name rather than reading around it. The Python tests validate every program the compilers
emit against :func:`program_schema`; the runtime's tests validate their fixtures against the copy committed
beside them, which ``python -m konfai.portable_schema <path>`` writes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from konfai.export import MANIFEST_VERSION
from konfai.portable import PREDICTION_PROGRAM_VERSION, TRANSFORM_PROGRAM_VERSION
from konfai.portable_evaluation import EVALUATION_PROGRAM_VERSION

_STRING: dict[str, Any] = {"type": "string"}
_INTEGER: dict[str, Any] = {"type": "integer"}
_NUMBER: dict[str, Any] = {"type": "number"}
_BOOLEAN: dict[str, Any] = {"type": "boolean"}
#: A clip bound: a number, or "min", "max" or "percentile:<q>", resolved against the volume in hand.
_BOUND: dict[str, Any] = {"type": ["number", "string"]}
_INTERPOLATION: dict[str, Any] = {"enum": ["nearest", "linear"]}
_ALIGN: dict[str, Any] = {"enum": ["extent", "origin"]}
_AXES: dict[str, Any] = {"type": "array", "items": {"type": "integer", "minimum": 0}}
#: The name a post chain undoes the step by; written on the ops konfai's predictor can undo.
_ID: dict[str, Any] = {"id": _STRING}


def _array(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def _ref(name: str) -> dict[str, Any]:
    return {"$ref": f"#/$defs/{name}"}


#: Op name -> (the keys a chain step of it may carry, the ones it must). A chain is a manifest's preprocessing,
#: postprocessing or reads, a TRANSFORM chain's steps, a Reduce's or an Expand's post.
CHAIN_OPS: dict[str, tuple[dict[str, Any], list[str]]] = {
    "cast": ({"dtype": _STRING, **_ID}, ["dtype"]),
    "clip": ({"min_value": _BOUND, "max_value": _BOUND, "mask_group": _STRING}, ["min_value", "max_value"]),
    "resample": (
        {
            "spacing": _array(_NUMBER),
            "shape": _array(_INTEGER),
            "reference": _ref("reference"),
            "align": _ALIGN,
            "fill": _NUMBER,
            "interpolation": _INTERPOLATION,
            # The group holding the displacement field, one entry per case; beside the case, or in the store
            # field_dataset names.
            "field_group": _STRING,
            "field_dataset": _ref("source"),
            # The transforms stored beside each case, by group in the order the stage declares them, and the
            # run's datasets each is looked up in.
            "transforms": _ref("stored_transforms"),
            **_ID,
        },
        [],
    ),
    "pad": ({"padding": _array(_INTEGER), "mode": _STRING, **_ID}, ["padding", "mode"]),
    "dilate": ({"radius": _INTEGER}, ["radius"]),
    "canonical": (dict(_ID), []),
    "flip": ({"axes": _AXES, "vector_field": _BOOLEAN, **_ID}, ["axes"]),
    "permute": ({"axes": _AXES, **_ID}, ["axes"]),
    "standardize": ({"mean": _NUMBER, "std": _NUMBER, "mask_group": _STRING, **_ID}, []),
    "normalize": ({"min_value": _NUMBER, "max_value": _NUMBER, **_ID}, ["min_value", "max_value"]),
    "unnormalize": ({"min_value": _NUMBER, "max_value": _NUMBER}, ["min_value", "max_value"]),
    "softmax": ({"dim": {"const": 0}}, ["dim"]),
    "argmax": ({"dim": {"const": 0}}, ["dim"]),
    "relabel": (
        {"pairs": _array({"type": "array", "items": _INTEGER, "minItems": 2, "maxItems": 2}), "positive": _INTEGER},
        [],
    ),
    "one_hot": ({"num_classes": _INTEGER, **_ID}, ["num_classes"]),
    "crop": (dict(_ID), []),
    "gradient": ({"per_dim": _BOOLEAN}, ["per_dim"]),
    "norm": ({}, []),
    "magnitude": ({}, []),
    "histogram_match": ({"reference_group": _STRING}, ["reference_group"]),
    "percentage": ({"baseline": _NUMBER}, ["baseline"]),
    "variance": ({}, []),
    "standard_deviation": ({}, []),
    "segmentation_disagreement": ({}, []),
    "mask": ({"mask_group": _STRING, "value_outside": _NUMBER}, ["mask_group", "value_outside"]),
    "brightness": ({"b_std": _NUMBER}, ["b_std"]),
    "save": ({"destination": _ref("destination")}, ["destination"]),
    "inverse": ({"of": _STRING, "interpolation": _INTERPOLATION, "vector_field": _BOOLEAN, "fill": _NUMBER}, ["of"]),
}

#: Op name -> (the keys a program step of it carries beside ``in`` and ``out``, the ones it must). A program
#: step reads buffers and writes one: the copies' fold, a mask, a hoisted tail op.
BODY_OPS: dict[str, tuple[dict[str, Any], list[str]]] = {
    "mean": ({}, []),
    "median": ({}, []),
    "median_lower": ({}, []),
    "vote": ({}, []),
    "concat": ({}, []),
    "merge_labels": ({"classes": _array(_INTEGER)}, ["classes"]),
    # Aligned on the origin, the align, the spacing and the fill of the resample undone onto the reference buffer.
    "resample_linear": ({"align": _ALIGN, "spacing": _array(_NUMBER), "fill": _NUMBER}, []),
    "resample_nearest": ({"align": _ALIGN, "spacing": _array(_NUMBER), "fill": _NUMBER}, []),
    "softmax": ({}, []),
    "argmax": ({}, []),
    "cast": ({"dtype": _STRING}, ["dtype"]),
    "mask": ({"value_outside": _NUMBER}, ["value_outside"]),
    "dilate": ({"radius": _INTEGER}, ["radius"]),
    # A copy's Padding undone past the reduction, its margins taken back: the one program op that moves the grid.
    "pad": ({"padding": _array(_INTEGER), "mode": _STRING}, ["padding", "mode"]),
}


#: Criterion name -> (the keys an evaluation's criterion of it carries beside its ``key``, ``maximize`` and
#: ``metric``, the ones it must): the criteria ``konfai.portable_evaluation.CRITERIA`` compiles.
CRITERIA: dict[str, tuple[dict[str, Any], list[str]]] = {
    "dice": (
        {"labels": {"type": ["array", "null"], "items": _INTEGER}, "map": {"$ref": "#/$defs/destination"}},
        ["labels"],
    ),
    "mae": ({"reduction": {"enum": ["mean", "sum"]}, "map": {"$ref": "#/$defs/destination"}}, ["reduction"]),
    "psnr": ({"dynamic_range": _NUMBER}, ["dynamic_range"]),
    "mean": ({}, []),
    "ssim": ({"dynamic_range": _NUMBER}, ["dynamic_range"]),
    "sam_perceptual": (
        {
            "model": _STRING,
            "tile": _array(_INTEGER),
            "layers": _array({"type": "array", "items": _INTEGER, "minItems": 2, "maxItems": 2}),
            "weights": _array(_NUMBER),
        },
        ["model", "tile", "layers", "weights"],
    ),
    # konfai-impact's configuration, as the compiler writes it: one model and the layers it weighs.
    "impact_reg": (
        {
            "impact": _object(
                {
                    "models": {
                        "type": "object",
                        "additionalProperties": _object(
                            {
                                "ref": _STRING,
                                "layers_mask": {"type": "string", "pattern": "^[01]+$"},
                                "layers_weight": _array(_NUMBER),
                                "distance": {"enum": ["L1", "L2"]},
                                "direction": {"type": "boolean"},
                            },
                            ["ref", "layers_mask", "layers_weight", "distance", "direction"],
                        ),
                    },
                    "mode": {"const": "Jacobian"},
                    "normalize": {"type": "boolean"},
                },
                ["models", "mode", "normalize"],
            ),
            "activations": _NUMBER,
        },
        ["impact", "activations"],
    ),
}


def _step(op: str, spec: tuple[dict[str, Any], list[str]], fixed: dict[str, Any]) -> dict[str, Any]:
    properties, required = spec
    return _object({"op": {"const": op}, **fixed, **properties}, ["op", *fixed, *required])


def program_schema() -> dict[str, Any]:
    """The contract, as a JSON Schema whose ``$defs`` name every part a test may validate on its own."""
    location = {"root": _STRING, "format": _STRING, "store": _STRING}
    criterion = {"key": _STRING, "maximize": _BOOLEAN}
    buffers = {"in": _array(_STRING), "out": _STRING}
    model_step = _object(
        {
            "model": _STRING,
            "in": {"type": ["string", "array"], "items": _STRING},
            "out": _STRING,
            "manifest": _ref("manifest"),
        },
        ["model", "in", "out", "manifest"],
    )
    chain_object = _object(
        {
            "group_src": _STRING,
            "group_dest": _STRING,
            "steps": _ref("chain"),
            "write": _ref("destination"),
            "reduce": _object(
                {
                    "operator": {"enum": ["mean", "std", "median", "vote", "concat"]},
                    "output": _STRING,
                    "grid": _STRING,
                    "grid_tolerance": _NUMBER,
                    "post": _ref("chain"),
                },
                ["operator", "output", "grid", "grid_tolerance", "post"],
            ),
            "expand": _object(
                {
                    "seed": _INTEGER,
                    "entries": {"type": "object", "additionalProperties": _array(_STRING)},
                    "post": _ref("chain"),
                },
                ["seed", "entries", "post"],
            ),
        },
        ["group_src", "group_dest", "steps", "write"],
    )
    defs: dict[str, Any] = {
        "source": _object(location, ["root", "format"]),
        "destination": _object(
            {**location, "group": _STRING, "entry_suffixes": _array(_STRING)},
            ["root", "format", "group", "entry_suffixes"],
        ),
        "reference": _object(
            {
                "entry": _STRING,
                "datasets": _array(_object({**location, "group": _STRING}, ["root", "format", "group"])),
            },
            ["entry", "datasets"],
        ),
        "stored_transforms": _object(
            {
                "groups": _array(_object({"group": _STRING, "invert": _BOOLEAN}, ["group", "invert"])),
                "datasets": _array(_ref("source")),
            },
            ["groups", "datasets"],
        ),
        "chain_step": {"oneOf": [_step(op, spec, {}) for op, spec in CHAIN_OPS.items()]},
        "chain": _array(_ref("chain_step")),
        "channels": _object(
            {"name": _STRING, "group": _STRING, "channels": _INTEGER, "dtype": {"const": "f32"}},
            ["name", "group", "channels", "dtype"],
        ),
        "patch": _object(
            {
                "size": _array(_INTEGER),
                # null keeps konfai's own case-dependent overlap and last-patch assignment (manifest version 4).
                "overlap": {"type": ["array", "null"], "items": _INTEGER},
                "dim": _INTEGER,
                "extend_slice": _INTEGER,
                "pad_value": _NUMBER,
                "batch": _INTEGER,
            },
            ["size", "overlap", "dim", "extend_slice", "pad_value"],
        ),
        "read": _object(
            {"group": _STRING, "channels": _INTEGER, "preprocessing": _ref("chain")},
            ["group", "channels", "preprocessing"],
        ),
        "manifest": _object(
            {
                # The runtime reads a manifest from version 1; an explicit overlap is what the earlier ones carried.
                "konfai_edge_manifest": {"type": "integer", "minimum": 1, "maximum": MANIFEST_VERSION},
                "model": _STRING,
                "opset": _INTEGER,
                "output_module": _STRING,
                "input": _ref("channels"),
                "output": _ref("channels"),
                "patch": _ref("patch"),
                "geometry": _object({"mode": {"const": "preserve_from_input"}}, ["mode"]),
                "preprocessing": _ref("chain"),
                "postprocessing": _ref("chain"),
                "blend": _STRING,
                "reads": _array(_ref("read")),
            },
            ["konfai_edge_manifest", "model", "opset", "output_module", "input", "output", "patch", "geometry"],
        ),
        "program_step": {"oneOf": [model_step, *(_step(op, spec, buffers) for op, spec in BODY_OPS.items())]},
        "program": _object(
            {
                "steps": _array(_ref("program_step")),
                "output": _STRING,
                # The buffer of one float32 channel per copy, konfai's InferenceStack.
                "stack": _STRING,
                # Each copy's buffer -> its checkpoint and its test-time pass, what a run selects by.
                "members": {
                    "type": "object",
                    "additionalProperties": _object({"checkpoint": _STRING, "pass": _INTEGER}, ["checkpoint", "pass"]),
                },
            },
            ["steps", "output"],
        ),
        "transform": _object(
            {
                "konfai_edge_transform": {"const": TRANSFORM_PROGRAM_VERSION},
                "name": _STRING,
                "memory_budget": {"type": ["integer", "null"]},
                "on_fallback": {"enum": ["allow", "warn", "error"]},
                "sources": _array(_ref("source")),
                "cases": _array(_STRING),
                "chains": _array(chain_object),
            },
            ["konfai_edge_transform", "name", "memory_budget", "on_fallback", "sources", "cases", "chains"],
        ),
        "criterion": {
            "oneOf": [
                _object({**criterion, "metric": {"const": metric}, **fields}, ["key", "maximize", "metric", *required])
                for metric, (fields, required) in CRITERIA.items()
            ]
        },
        "evaluation": _object(
            {
                "konfai_edge_evaluation": {"const": EVALUATION_PROGRAM_VERSION},
                "name": _STRING,
                "memory_budget": {"type": ["integer", "null"]},
                "sources": _array(_ref("source")),
                "cases": _array(_STRING),
                "evaluations": _STRING,
                "groups": _array(
                    _object(
                        {
                            "group_src": _STRING,
                            "group_dest": _STRING,
                            "steps": _array(
                                {
                                    "oneOf": [
                                        _ref("chain_step"),
                                        # Where konfai's Statistics records the volume's minimum and maximum.
                                        _object({"op": {"const": "statistics"}}, ["op"]),
                                        # A KonfAIInference: the nested app's model on the whole volume, or on
                                        # each channel as a case of its own.
                                        _object(
                                            {"model": _STRING, "manifest": _ref("manifest"), "per_channel": _BOOLEAN},
                                            ["model", "manifest"],
                                        ),
                                    ]
                                }
                            ),
                        },
                        ["group_src", "group_dest", "steps"],
                    )
                ),
                "models": {"type": "object", "additionalProperties": _STRING},
                "metrics": _array(
                    _object(
                        {"output": _STRING, "targets": _array(_STRING), "criteria": _array(_ref("criterion"))},
                        ["output", "targets", "criteria"],
                    )
                ),
            },
            [
                "konfai_edge_evaluation",
                "name",
                "memory_budget",
                "sources",
                "cases",
                "evaluations",
                "groups",
                "models",
                "metrics",
            ],
        ),
        "prediction": _object(
            {
                "konfai_edge_prediction": {"const": PREDICTION_PROGRAM_VERSION},
                "name": _STRING,
                "memory_budget": {"type": ["integer", "null"]},
                "autocast": {"type": "boolean"},
                "sources": _array(_ref("source")),
                "cases": _array(_STRING),
                "input_group": {"type": ["string", "array"], "items": _STRING},
                "models": {"type": "object", "additionalProperties": _STRING},
                "outputs": _array(
                    _object(
                        {"program": _ref("program"), "write": _ref("destination"), "stack": _ref("destination")},
                        ["program", "write"],
                    )
                ),
            },
            ["konfai_edge_prediction", "name", "memory_budget", "sources", "cases", "input_group", "models", "outputs"],
        ),
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "KonfAI portable programs",
        "description": "A TRANSFORM program, a PREDICTION program, an EVALUATION program, a bundle's assembled program"
        " or one model's manifest, as konfai.portable emits them and the portable runtime reads them.",
        "oneOf": [_ref("transform"), _ref("prediction"), _ref("evaluation"), _ref("program"), _ref("manifest")],
        "$defs": defs,
    }


def write_program_schema(path: str | Path) -> Path:
    """Write the schema to ``path``, the runtime's committed copy, and return it."""
    target = Path(path)
    target.write_text(json.dumps(program_schema(), indent=2) + "\n")
    return target


if __name__ == "__main__":
    print(write_program_schema(sys.argv[1]))

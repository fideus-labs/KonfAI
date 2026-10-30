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

import codecs
import json
import locale
import sys

import pytest
from konfai.utils.errors import ConfigError
from konfai_apps.bundle import assemble_bundle
from konfai_apps.errors import AppMetadataError

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


@pytest.fixture
def ascii_locale():
    """The locale's encoding set to ASCII: a file opened without an encoding takes it, as it takes the
    ANSI code page on Windows, and any character outside it fails there the same way."""
    if sys.flags.utf8_mode:
        pytest.skip("UTF-8 mode ignores the locale's encoding")
    previous = locale.setlocale(locale.LC_CTYPE)
    locale.setlocale(locale.LC_CTYPE, "C")
    try:
        if codecs.lookup(locale.getencoding()).name == "utf-8":
            pytest.skip("this platform's C locale is UTF-8")
        yield
    finally:
        locale.setlocale(locale.LC_CTYPE, previous)


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
    from konfai.portable import derive_onnx_params

    config = {
        "Predictor": {
            "Model": {"classpath": "Model:UNetpp", "UNetpp": {"nb_channel": 5}},
            "Dataset": {"Patch": {"patch_size": [1, 256, 256], "extend_slice": 2, "pad_value": -2}},
            "Model_unused_patch": {"ModelPatch": {"patch_size": [128, 128, 128]}},
        }
    }
    patch_size, in_channels, extend_slice, pad_value = derive_onnx_params(config, "Predictor")
    assert patch_size == [256, 256]  # singleton slice dim dropped (2.5D)
    assert in_channels == 5
    assert extend_slice == 2
    assert pad_value == -2.0  # the config's border-pad value reaches the manifest


def test_derive_overlap_broadcasts_scalar_and_drops_singleton():
    from konfai.portable import derive_overlap

    assert derive_overlap({"overlap": 32}, [96, 128, 160]) == [32, 32, 32]  # scalar broadcast
    assert derive_overlap({"overlap": [8, 8]}, [64, 64]) == [8, 8]  # per-axis list kept
    # a full-rank overlap carrying the 2.5D singleton slice axis: kept axes match patch_size
    assert derive_overlap({"overlap": [0, 16, 16], "patch_size": [1, 256, 256]}, [256, 256]) == [16, 16]
    assert derive_overlap({}, [96, 128, 160]) is None  # no overlap declared


@pytest.mark.parametrize(
    "patch_size, overlap",
    [
        ([128, 64], 16),
        ([128, 64], 0.25),
        ([128, 64], "20%"),
        ([128, 64], [0.5, 0.25]),
        ([1, 256, 256], "20%"),
        ([1, 256, 256], [0, 0.2, 0.2]),
    ],
)
def test_the_manifest_overlap_is_the_one_konfai_tiles_with(patch_size, overlap):
    """A fraction, a percent string or a list of fractions is a share of the patch: the manifest carries
    the voxels KonfAI's own grid steps by on an axis it tiles, not ``int(0.25) == 0``."""
    from konfai.portable import derive_onnx_params as _derive_onnx_params
    from konfai.portable import derive_overlap as _derive_overlap
    from konfai.utils.utils import get_patch_slices_from_shape

    patch = {"patch_size": patch_size, "overlap": overlap}
    exported, *_ = _derive_onnx_params({"Predictor": {"Dataset": {"Patch": patch}}}, "Predictor")
    slices = get_patch_slices_from_shape(patch_size, [1000] * len(patch_size), overlap)
    konfai = []
    for axis, size in enumerate(patch_size):
        starts = sorted({region[axis].start for region in slices})
        if size > 1:  # the axis the manifest keeps
            konfai.append(size - (starts[1] - starts[0]))
    assert _derive_overlap(patch, exported) == konfai


def test_derive_blend_reads_patch_combine():
    from konfai.portable import derive_blend

    cfg = {"Predictor": {"outputs_dataset": {"H": {"OutputDataset": {"patch_combine": "Gaussian"}}}}}
    assert derive_blend(cfg, "Predictor") == "Gaussian"
    cfg = {"Predictor": {"outputs_dataset": {"O": {"OutputDataset": {"patch_combine": "None"}}}}}
    assert derive_blend(cfg, "Predictor") == "Overwrite"
    assert derive_blend({"Predictor": {}}, "Predictor") is None


def test_transform_manifest_maps_the_pipeline_to_runtime_ops():
    from konfai.portable import transform_manifest

    config = {
        "Predictor": {
            "Dataset": {
                "groups_src": {
                    "Volume_0": {
                        "groups_dest": {
                            "Volume": {
                                "transforms": {
                                    "TensorCast": {"dtype": "float32", "inverse": False},
                                    "Resample": {"spacing": [3, 3, 3], "inverse": True},
                                    "Standardize": {"mean": "None", "std": "None", "inverse": False},
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
    manifest, folds, per_copy = transform_manifest(config, "Predictor")
    assert folds == []  # every transform here is a runtime op; nothing folded into the graph
    assert per_copy == 0  # nothing runs before the reduction
    assert manifest["preprocessing"] == [
        {"op": "cast", "dtype": "float32"},
        # konfai reads the float32 the cast leaves linearly.
        {"op": "resample", "spacing": [3.0, 3.0, 3.0], "id": "Resample", "interpolation": "linear"},
        {"op": "standardize"},  # mean/std unset -> computed at runtime
    ]
    # The chain names where its inverse runs, and the predictor runs it BEFORE the final transforms:
    # the class is decided on the grid the caller gave.
    assert manifest["postprocessing"] == [
        {"op": "inverse", "of": "Resample", "interpolation": "linear"},
        {"op": "softmax", "dim": 0},
        {"op": "argmax", "dim": 0},
    ]


def test_transform_manifest_reads_canonical_and_before_reduction_post():
    from konfai.portable import transform_manifest

    # A disjoint (merge_labels) ensemble: Canonical -> a runtime op; the per-fold argmax lives in
    # before_reduction_transforms (each fold argmaxes to a label map before the merge), not final_transforms.
    config = {
        "Predictor": {
            "Dataset": {
                "groups_src": {
                    "V": {
                        "groups_dest": {
                            "g": {
                                "is_input": True,
                                "transforms": {
                                    "TensorCast": {"dtype": "float32", "inverse": False},
                                    "Canonical": {"inverse": True},
                                    "Resample": {"spacing": [1.5, 1.5, 1.5], "inverse": True},
                                },
                            }
                        }
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
    manifest, _folds, per_copy = transform_manifest(config, "Predictor")
    assert {"op": "canonical", "id": "Canonical"} in manifest["preprocessing"]
    assert per_copy == 2  # the per-fold argmax runs on every copy, before the merge
    # The per-fold ops come first, then the inverses of the input chain in reverse order.
    assert manifest["postprocessing"] == [
        {"op": "softmax", "dim": 0},
        {"op": "argmax", "dim": 0},
        # The Mean reduction averages each copy's int64 class into float32, which konfai's Resample reads linearly.
        {"op": "cast", "dtype": "float32"},
        {"op": "inverse", "of": "Resample", "interpolation": "linear"},
        {"op": "inverse", "of": "Canonical"},
    ]


def test_derive_reduction_reads_the_ensemble_reduction():
    from konfai.portable import derive_reduction

    def cfg(after):
        outputs = {"H": {"OutputDataset": {"after_reduction_transforms": after}}}
        return {"Predictor": {"combine": "Concat", "outputs_dataset": outputs}}

    # The multi-model reduction is read from the output transforms, never hard-coded per app.
    assert derive_reduction(cfg({"MergeLabels": {}}), "Predictor") == "merge_labels"
    assert derive_reduction(cfg({"InferenceStack": {"mode": "mean"}}), "Predictor") == "mean"
    # The stack's median is torch's lower middle, not the Median reduction's averaged pair.
    assert derive_reduction(cfg({"InferenceStack": {"mode": "median"}}), "Predictor") == "median_lower"
    assert derive_reduction(cfg("None"), "Predictor") is None
    assert derive_reduction({"Predictor": {}}, "Predictor") is None


def test_masked_tta_compiler_reads_the_config():
    from konfai.portable import assemble_program, aux_mask_groups, mask_specs, tta_passes

    config = {
        "Predictor": {
            "manual_seed": 32,
            "Dataset": {
                "groups_src": {
                    "Volume_0": {
                        "groups_dest": {
                            "MASK": {
                                "is_input": False,
                                "transforms": {
                                    "konfai_apps.transforms:KonfAIInference": {"repo_id": "R/S", "model_name": "body"},
                                    "Resample": {"spacing": [1, 1, 3]},
                                    "Dilate": {"dilate": 5},
                                    "Save": {"dataset": "x"},
                                },
                            },
                            "Volume": {"is_input": True, "transforms": {"Normalize": {}}},
                        }
                    }
                },
                "augmentations": {"DA0": {"nb": 2, "data_augmentations": {"Flip": {"f_prob": [0, 0.5, 0.5]}}}},
            },
            "outputs_dataset": {
                "H": {
                    "OutputDataset": {"before_reduction_transforms": {"Mask": {"path": "MASK", "value_outside": -1024}}}
                }
            },
        }
    }
    passes = tta_passes(config, "Predictor")
    assert passes[0] == [] and len(passes) == 3  # identity + nb draws; other augmentations would be skipped
    aux = aux_mask_groups(config, "Predictor")
    assert set(aux) == {"MASK"} and [op["op"] for op in aux["MASK"]["ops"]] == ["resample", "dilate"]
    assert mask_specs(config, "Predictor") == [{"group": "MASK", "value_outside": -1024.0}]
    # The nested body app ends its chain on a uint8 cast, so that is the dtype of the file konfai reads.
    body = {"postprocessing": [{"op": "argmax", "dim": 0}, {"op": "cast", "dtype": "uint8"}]}

    fold = {
        "preprocessing": [{"op": "resample", "id": "Resample"}],
        "postprocessing": [
            {"op": "inverse", "of": "Resample", "interpolation": "linear"},
            {"op": "cast", "dtype": "int16"},
        ],
    }
    program = assemble_program(
        [{"id": f"CV_{i}", "manifest": fold} for i in range(5)],
        reduce="mean",
        passes=passes,
        aux={"MASK": {"id": "MASK", "manifest": body, "ops": aux["MASK"]["ops"]}},
        masks=mask_specs(config, "Predictor"),
    )
    ops = [s["op"] for s in program["steps"] if "op" in s]
    assert "mask" in ops and "resample_linear" in ops  # the mask and the hoisted inverse resample
    assert program["steps"][-1]["op"] == "cast" and program["steps"][-1]["dtype"] == "int16"
    # konfai reads the uint8 mask nearest when its Resample states no interpolation.
    mask_ops = [s for s in program["steps"] if s.get("out", "").startswith("MASK__") and "op" in s]
    assert [s["op"] for s in mask_ops] == ["resample_nearest", "dilate"]

    # The hoisted inverse reads with the interpolation the fold's inverse step names.
    fold["postprocessing"][0]["interpolation"] = "nearest"
    program = assemble_program(
        [{"id": "CV_0", "manifest": fold}],
        reduce="mean",
        passes=passes,
        aux={"MASK": {"id": "MASK", "manifest": body, "ops": aux["MASK"]["ops"]}},
        masks=mask_specs(config, "Predictor"),
    )
    assert [s["op"] for s in program["steps"] if s.get("op", "").startswith("resample_")][-1] == "resample_nearest"

    # The cast the compiler writes where the copies fold runs on their reduction, before the mask.
    fold["postprocessing"].insert(0, {"op": "cast", "dtype": "int16"})
    program = assemble_program(
        [{"id": "CV_0", "manifest": fold}],
        reduce="mean",
        passes=passes,
        aux={"MASK": {"id": "MASK", "manifest": body, "ops": aux["MASK"]["ops"]}},
        masks=mask_specs(config, "Predictor"),
    )
    # The cast the copies fold into runs on their reduction, before the mask reads the nested buffer.
    assert [s["op"] for s in program["steps"] if s["out"].startswith("t") or s["out"] == "output"][:2] == [
        "cast",
        "mask",
    ]


def test_masked_tta_compiler_refuses_what_its_program_has_no_step_for():
    import pytest
    from konfai.portable import aux_mask_groups, mask_specs
    from konfai.utils.errors import ConfigError

    def config(mask_chain: dict, before: dict) -> dict:
        inference = {"KonfAIInference": {"repo_id": "R/S", "model_name": "body"}}
        groups = {"MASK": {"is_input": False, "transforms": {**inference, **mask_chain}}}
        return {
            "Predictor": {
                "Dataset": {"groups_src": {"V": {"groups_dest": groups}}},
                "outputs_dataset": {"H": {"OutputDataset": {"before_reduction_transforms": before}}},
            }
        }

    mask = {"Mask": {"path": "MASK", "value_outside": 0}}
    # A Save writes what passes through, which alters no value of the mask the program builds.
    assert set(aux_mask_groups(config({"Dilate": {"dilate": 1}, "Save": {"dataset": "x"}}, mask), "Predictor"))
    # Any other stage would be dropped from that mask.
    with pytest.raises(ConfigError, match="'TensorCast' on the nested model's group 'MASK'"):
        aux_mask_groups(config({"TensorCast": {"dtype": "uint8"}}, mask), "Predictor")
    # A Mask reading a group no nested model builds has no buffer in the program, konfai's default path included.
    for before in ({"Mask": {"path": "BODY"}}, {"Mask": None}):
        with pytest.raises(ConfigError, match="which no nested model on the config builds"):
            mask_specs(config({}, before), "Predictor")


def test_masked_tta_compiler_honours_the_mask_resample_interpolation():
    import pytest
    from konfai.portable import assemble_program, aux_mask_groups
    from konfai.utils.errors import ConfigError, TransformError

    def mask_ops(resample: dict, nested: dict) -> list[dict]:
        config = {
            "Predictor": {
                "Dataset": {
                    "groups_src": {
                        "V": {
                            "groups_dest": {
                                "MASK": {
                                    "is_input": False,
                                    "transforms": {
                                        "KonfAIInference": {"repo_id": "R/S", "model_name": "body"},
                                        "Resample": {"spacing": [1, 1, 3], **resample},
                                    },
                                }
                            }
                        }
                    }
                }
            }
        }
        aux = aux_mask_groups(config, "Predictor")
        fold = {"id": "CV_0", "manifest": {"postprocessing": []}}
        program = assemble_program(
            [fold],
            reduce="mean",
            aux={"MASK": {"id": "MASK", "manifest": nested, "ops": aux["MASK"]["ops"]}},
            masks=[{"group": "MASK", "value_outside": 0.0}],
        )
        return [
            {k: v for k, v in s.items() if k not in ("in", "out", "value_outside")}
            for s in program["steps"]
            if "op" in s and s["op"] != "mask"
        ]

    uint8 = {"postprocessing": [{"op": "cast", "dtype": "uint8"}]}
    # A stated linear read of the uint8 mask: the runtime casts the blend back to the uint8 the nested
    # app's buffer carries, as konfai does.
    assert mask_ops({"interpolation": "linear"}, uint8) == [{"op": "resample_linear"}]
    assert mask_ops({"interpolation": "cubic"}, uint8) == [{"op": "resample_cubic"}]
    with pytest.raises(TransformError, match="lanczos"):
        mask_ops({"interpolation": "lanczos"}, uint8)
    # No interpolation, and a nested chain that does not end on a cast: the dtype konfai picks from is unknown.
    with pytest.raises(ConfigError, match="the nested app writes"):
        mask_ops({}, {"postprocessing": [{"op": "argmax", "dim": 0}]})


def test_transform_manifest_undoes_a_value_transform_after_the_forward():
    from konfai.portable import transform_manifest

    # konfai's Standardize undoes itself on the output by default, with the statistics its forward
    # recorded on the case; the runtime records them too, so the chain ships with its inverse.
    config = {
        "Predictor": {
            "Dataset": {
                "groups_src": {
                    "V": {"groups_dest": {"g": {"transforms": {"Standardize": {"mean": 0, "std": 1, "inverse": True}}}}}
                }
            },
            "outputs_dataset": {},
        }
    }
    manifest, _folds, _per_copy = transform_manifest(config, "Predictor")
    assert manifest["preprocessing"] == [{"op": "standardize", "mean": 0.0, "std": 1.0, "id": "Standardize"}]
    assert {"op": "inverse", "of": "Standardize"} in manifest["postprocessing"]


def test_transform_manifest_refuses_an_unportable_transform():
    import pytest
    from konfai.portable import transform_manifest
    from konfai.utils.errors import ConfigError

    groups = {"V": {"groups_dest": {"g": {"transforms": {"SomeCustomTransform": {"x": 1}}}}}}
    config = {"Predictor": {"Dataset": {"groups_src": groups}, "outputs_dataset": {}}}
    with pytest.raises(ConfigError, match="no portable runtime op"):
        transform_manifest(config, "Predictor")


def test_the_program_reads_every_occurrence_of_a_repeated_stage():
    from konfai.portable import aux_mask_groups as _aux_mask_groups
    from konfai.portable import derive_reduction as _derive_reduction
    from konfai.portable import mask_specs as _mask_specs

    before = {
        "Mask": {"path": "BODY", "value_outside": -1024},
        "Mask#2": {"path": "BONE", "value_outside": 0},
        "konfai.data.transform:Mask": {"path": "AIR", "value_outside": 1},
    }
    config = {
        "Predictor": {
            "Dataset": {
                "groups_src": {
                    "CT": {
                        "groups_dest": {
                            "BODY": {
                                "is_input": False,
                                "transforms": {
                                    "KonfAIInference": {"repo_id": "R/S", "model_name": "body"},
                                    "Resample": {"spacing": [1, 1, 3]},
                                    "Dilate": {"dilate": 2},
                                    "Dilate#2": {"dilate": 3},
                                },
                            },
                            "BONE": {
                                "is_input": False,
                                "transforms": {"KonfAIInference": {"repo_id": "R/S", "model_name": "bone"}},
                            },
                            "AIR": {
                                "is_input": False,
                                "transforms": {"KonfAIInference": {"repo_id": "R/S", "model_name": "air"}},
                            },
                        }
                    }
                }
            },
            "outputs_dataset": {
                "H": {
                    "OutputDataset": {
                        "before_reduction_transforms": before,
                        "after_reduction_transforms": {"konfai.data.transform:MergeLabels": {}},
                    }
                }
            },
        }
    }
    assert _mask_specs(config, "Predictor") == [
        {"group": "BODY", "value_outside": -1024.0},
        {"group": "BONE", "value_outside": 0.0},
        {"group": "AIR", "value_outside": 1.0},
    ]
    assert [op["op"] for op in _aux_mask_groups(config, "Predictor")["BODY"]["ops"]] == [
        "resample",
        "dilate",
        "dilate",
    ]
    assert _derive_reduction(config, "Predictor") == "merge_labels"


def test_a_chain_spelled_as_a_list_reaches_the_manifest_and_the_program():
    """The list spelling binds in the core as its occurrence mapping; the export reads it the same way
    instead of finding no chain at all."""
    from konfai.portable import aux_mask_groups as _aux_mask_groups
    from konfai.portable import mask_specs as _mask_specs
    from konfai.portable import transform_manifest as _transform_manifest
    from konfai.portable import tta_passes as _tta_passes

    config = {
        "Predictor": {
            "Dataset": {
                "groups_src": {
                    "CT": {
                        "groups_dest": {
                            "CT": {
                                "is_input": True,
                                "transforms": [
                                    {"Clip": {"min_value": -1000, "max_value": 1000}},
                                    {"Clip": {"min_value": -500, "max_value": 500}},
                                    {"Resample": {"spacing": [1.5, 1.5, 1.5]}},
                                ],
                            },
                            "BODY": {
                                "is_input": False,
                                "transforms": [
                                    {"KonfAIInference": {"repo_id": "R/S", "model_name": "body"}},
                                    {"Dilate": {"dilate": 2}},
                                ],
                            },
                        }
                    }
                },
                "augmentations": {"DA0": {"nb": 2, "data_augmentations": [{"Flip": {"f_prob": [0, 0.5, 0.5]}}]}},
            },
            "outputs_dataset": {
                "H": {
                    "OutputDataset": {
                        "before_reduction_transforms": [{"Mask": {"path": "BODY", "value_outside": 0}}],
                        "final_transforms": [{"Argmax": {"dim": 0}}],
                    }
                }
            },
        }
    }
    manifest, folds, _per_copy = _transform_manifest(config, "Predictor", masked=True)
    assert folds == []
    assert [step["op"] for step in manifest["preprocessing"]] == ["clip", "clip", "resample"]
    assert [step["op"] for step in manifest["postprocessing"]] == ["inverse", "argmax"]
    assert [op["op"] for op in _aux_mask_groups(config, "Predictor")["BODY"]["ops"]] == ["dilate"]
    assert _mask_specs(config, "Predictor") == [{"group": "BODY", "value_outside": 0.0}]
    assert len(_tta_passes(config, "Predictor")) == 3


def _masked_config(dest: dict, section: str, chain) -> dict:
    """An input group ``CT``, the ``dest`` groups beside it, and ``chain`` as its ``section``."""
    groups = {"CT": {"is_input": True, "transforms": chain if section == "transforms" else {}}, **dest}
    outputs = {} if section == "transforms" else {"H": {"OutputDataset": {section: chain}}}
    return {"Predictor": {"Dataset": {"groups_src": {"CT": {"groups_dest": groups}}}, "outputs_dataset": outputs}}


@pytest.mark.parametrize(
    ("section", "chain"),
    [
        # A list of two masks, and the mapping it lands on disk as once the core reads it.
        ("before_reduction_transforms", [{"Mask": {"path": "BODY"}}, {"Mask": {"path": "BONE"}}]),
        ("before_reduction_transforms", {"Mask": {"path": "BODY"}, "Mask#2": {"path": "BONE"}}),
        ("final_transforms", {"Mask": {"path": "BODY", "value_outside": -1024}}),
        ("transforms", {"Dilate": {"dilate": 1}, "Dilate#2": {"dilate": 2}}),
    ],
)
def test_a_program_stage_no_program_step_applies_is_refused_not_dropped(section, chain):
    """The portable program masks the output only by a group a nested model produces: a mask read from
    the dataset, or a Dilate on the input, would be skipped and the artifact's output left unmasked."""
    from konfai.portable import transform_manifest as _transform_manifest

    config = _masked_config({"BODY": {"is_input": False}, "BONE": {"is_input": False}}, section, chain)
    with pytest.raises(ConfigError):
        _transform_manifest(config, "Predictor")


def test_every_mask_on_a_group_a_nested_model_produces_is_left_to_the_program():
    from konfai.portable import transform_manifest as _transform_manifest

    nested = {
        "is_input": False,
        "transforms": {"KonfAIInference": {"repo_id": "R/S", "model_name": "body"}, "Dilate": {"dilate": 2}},
    }
    before = {"Mask": {"path": "BODY"}, "Mask#2": {"path": "BONE"}, "Save": {"dataset": "x"}}
    manifest, _folds, _per_copy = _transform_manifest(
        _masked_config({"BODY": nested, "BONE": nested}, "before_reduction_transforms", before),
        "Predictor",
        masked=True,
    )
    assert manifest["postprocessing"] == []


@pytest.mark.parametrize("resample", [False, True])
@pytest.mark.parametrize(
    "volume",
    [{"is_input": True, "transforms": None}, {"is_input": True}, {}],
    ids=["transforms-None", "no-transforms-key", "is_input-by-default"],
)
def test_an_input_group_without_a_chain_takes_no_other_groups_chain(volume, resample):
    """The core writes an absent chain as ``transforms: None``: the model's input then has no
    preprocessing, and the chain of the mask group a nested model produces is not the input's."""
    from konfai.portable import transform_manifest as _transform_manifest

    mask_chain = {"KonfAIInference": {"repo_id": "R/S", "model_name": "body"}}
    if resample:
        mask_chain["Resample"] = {"spacing": [1, 1, 3]}
    mask_chain["Dilate"] = {"dilate": 5}
    groups = {"MASK": {"is_input": False, "transforms": mask_chain}, "Volume": volume}
    before = {"TensorCast": {"dtype": "int16"}, "Mask": {"path": "MASK", "value_outside": -1024}}
    config = {
        "Predictor": {
            "Dataset": {"groups_src": {"Volume_0": {"groups_dest": groups}}},
            "outputs_dataset": {"H": {"OutputDataset": {"before_reduction_transforms": before}}},
        }
    }
    manifest, folds, _per_copy = _transform_manifest(config, "Predictor", masked=True)
    assert manifest["preprocessing"] == [] and folds == []
    assert manifest["postprocessing"][0] == {"op": "cast", "dtype": "int16"}


def test_colliding_bundle_filenames_are_refused_before_anything_is_written(tmp_path):
    """fold0/best.pt and fold1/best.pt flattened to one best.pt: the second overwrote the first and
    app.json named both."""
    app_json = _write(tmp_path / "app.json", VALID_META)
    config = tmp_path / "Prediction.yml"
    config.write_text("Predictor: {}\n")
    (tmp_path / "fold0").mkdir()
    (tmp_path / "fold1").mkdir()
    (tmp_path / "fold0" / "best.pt").write_bytes(b"zero")
    (tmp_path / "fold1" / "best.pt").write_bytes(b"one")
    with pytest.raises(AppMetadataError, match=r"best\.pt"):
        assemble_bundle(
            "MR",
            tmp_path / "out",
            app_json,
            [str(config)],
            [str(tmp_path / "fold0/best.pt"), str(tmp_path / "fold1/best.pt")],
        )
    assert not (tmp_path / "out" / "MR").exists()
    assert not list((tmp_path / "out").glob(".MR.staging-*")) if (tmp_path / "out").exists() else True


def test_a_repackaged_bundle_drops_the_previous_checkpoint_and_keeps_user_files(tmp_path):
    """The second export into an existing bundle left the first checkpoint beside the new one, and
    the default inference enumerated the old one. The previous manifest's checkpoints go; a file the
    export does not manage stays."""
    app_json = _write(tmp_path / "app.json", VALID_META)
    config = tmp_path / "Prediction.yml"
    config.write_text("Predictor: {}\n")
    first = tmp_path / "first.pt"
    first.write_bytes(b"first")
    bundle = assemble_bundle("MR", tmp_path / "out", app_json, [str(config)], [str(first)])
    (bundle / "notes.txt").write_text("mine\n")

    new = tmp_path / "new.pt"
    new.write_bytes(b"new")
    assert assemble_bundle("MR", tmp_path / "out", app_json, [str(config)], [str(new)]) == bundle
    assert sorted(path.name for path in bundle.glob("*.pt")) == ["new.pt"]
    assert json.loads((bundle / "app.json").read_text())["models"] == ["new.pt"]
    assert (bundle / "notes.txt").read_text() == "mine\n"
    assert not list((tmp_path / "out").glob(".MR.staging-*"))


@pytest.mark.parametrize("reference_kind", ["parent", "absolute", "symlink"])
def test_repackaging_refuses_previous_checkpoints_outside_the_bundle(tmp_path, reference_kind):
    app_json = _write(tmp_path / "app.json", VALID_META)
    config = tmp_path / "Prediction.yml"
    config.write_text("Predictor: {}\n")
    checkpoint = tmp_path / "new.pt"
    checkpoint.write_bytes(b"new")
    bundle = tmp_path / "out" / "MR"
    bundle.mkdir(parents=True)
    neighbor = bundle.parent / "neighbor.pt"
    neighbor.write_bytes(b"must survive")
    previous = "../neighbor.pt" if reference_kind == "parent" else str(neighbor)
    if reference_kind == "symlink":
        (bundle / "old.pt").symlink_to(neighbor)
        previous = "old.pt"
    manifest = json.dumps({**VALID_META, "models": [previous]})
    (bundle / "app.json").write_text(manifest)
    (bundle / "Prediction.yml").write_text("old config\n")

    with pytest.raises(AppMetadataError, match="outside the bundle"):
        assemble_bundle("MR", bundle.parent, app_json, [str(config)], [str(checkpoint)])

    assert neighbor.read_bytes() == b"must survive"
    assert (bundle / "app.json").read_text() == manifest
    assert (bundle / "Prediction.yml").read_text() == "old config\n"
    assert not (bundle / "new.pt").exists()
    assert not list(bundle.parent.glob(".MR.staging-*"))


def test_portable_asset_roles_preserve_other_native_onnx_dependencies(tmp_path, monkeypatch):
    from konfai_apps.app_repository import LocalAppRepositoryFromDirectory
    from konfai_apps.bundle import _declare_portable_assets

    monkeypatch.setenv("KONFAI_APPS_INSTALL_REQUIREMENTS", "0")
    bundle = tmp_path / "MR"
    bundle.mkdir()
    metadata = {**VALID_META, "models": ["best.pt"], "portable_assets": ["legacy.onnx.data"]}
    (bundle / "app.json").write_text(json.dumps(metadata))
    (bundle / "Prediction.yml").write_text("Predictor: {}\n")
    for name in (
        "best.pt",
        "model.onnx",
        "model.onnx.data",
        "manifest.json",
        "legacy.onnx.data",
        "native_auxiliary.onnx",
    ):
        (bundle / name).write_bytes(name.encode())
    produced = {bundle / name for name in ("model.onnx", "model.onnx.data", "manifest.json")}

    assert _declare_portable_assets(bundle, bundle / "model.onnx", produced) == bundle / "model.onnx"
    roles = json.loads((bundle / "app.json").read_text())["portable_assets"]
    assert roles == ["legacy.onnx.data", "manifest.json", "model.onnx", "model.onnx.data"]
    _, _, assets = LocalAppRepositoryFromDirectory(bundle.parent, bundle.name).download_inference(
        1, [], "Prediction.yml"
    )
    assert "native_auxiliary.onnx" in {name for name, _ in assets}
    assert not set(roles) & {name for name, _ in assets}


@pytest.mark.parametrize("ensemble", [False, True])
def test_portable_export_declares_only_its_own_outputs(tmp_path, monkeypatch, ensemble):
    from types import SimpleNamespace

    import konfai.export as exporter
    import konfai.network.network as network
    import konfai.utils.runtime as runtime
    from konfai_apps.bundle import export_portable_into_bundle

    metadata = {**VALID_META, "portable_assets": ["previous.onnx"]}
    (tmp_path / "app.json").write_text(json.dumps(metadata))
    (tmp_path / "native_auxiliary.onnx").write_bytes(b"required by native custom code")
    (tmp_path / "manifest.json").write_text("previous native manifest")
    config = {
        "Predictor": {
            "Model": {"classpath": "Probe"},
            "combine": "Concat",
            "outputs_dataset": {
                "Head": {"OutputDataset": {"after_reduction_transforms": {"InferenceStack": {"mode": "mean"}}}},
            },
        }
    }
    (tmp_path / "Prediction.yml").write_text(json.dumps(config))
    model = SimpleNamespace(eval=lambda: None, load=lambda *args, **kwargs: None, patch=None)
    monkeypatch.setattr(network, "ModelLoader", lambda _: SimpleNamespace(get_model=lambda **kwargs: model))
    monkeypatch.setattr(runtime, "safe_torch_load", lambda *args: {})

    def export_stub(_model, directory, _example, _head, **kwargs):
        path = directory / kwargs["model_filename"]
        path.write_bytes(b"exported portable model")
        manifest = {"model": path.name, "input": {"channels": 1}, "output": {"channels": 1}, "patch": {}}
        if kwargs["write_manifest"]:
            (directory / "manifest.json").write_text(json.dumps(manifest))
        return path, manifest

    monkeypatch.setattr(exporter, "export_to_onnx", export_stub)
    result = export_portable_into_bundle(
        tmp_path,
        checkpoints=["fold0.pt", "fold1.pt"] if ensemble else None,
        patch_size=[4, 4],
        in_channels=1,
        output_module="Head",
    )
    expected = {"fold0.onnx", "fold1.onnx", "program.json"} if ensemble else {"model.onnx", "manifest.json"}
    assert result.name == ("program.json" if ensemble else "model.onnx")
    assert set(json.loads((tmp_path / "app.json").read_text())["portable_assets"]) == expected | {"previous.onnx"}
    assert (tmp_path / "native_auxiliary.onnx").read_bytes() == b"required by native custom code"


def test_an_unmasked_export_runs_every_checkpoint_on_every_test_time_copy(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import konfai.export as exporter
    import konfai.network.network as network
    import konfai.utils.runtime as runtime
    from konfai_apps.bundle import export_portable_into_bundle

    (tmp_path / "app.json").write_text(json.dumps(VALID_META))
    config = {
        "Predictor": {
            "Model": {"classpath": "Probe"},
            "combine": "Concat",
            "Dataset": {"augmentations": {"T": {"nb": 2, "data_augmentations": {"Flip": {"f_prob": [1, 1]}}}}},
            "outputs_dataset": {
                "Head": {
                    "OutputDataset": {
                        "after_reduction_transforms": {"InferenceStack": {"mode": "mean"}},
                        "reduction": "Concat",
                    }
                },
            },
        }
    }
    (tmp_path / "Prediction.yml").write_text(json.dumps(config))
    model = SimpleNamespace(eval=lambda: None, load=lambda *args, **kwargs: None, patch=None)
    monkeypatch.setattr(network, "ModelLoader", lambda _: SimpleNamespace(get_model=lambda **kwargs: model))
    monkeypatch.setattr(runtime, "safe_torch_load", lambda *args: {})

    def export_stub(_model, directory, _example, _head, **kwargs):
        path = directory / kwargs["model_filename"]
        path.write_bytes(b"exported portable model")
        return path, {"model": path.name, "input": {"channels": 1}, "output": {"channels": 1}, "patch": {}}

    monkeypatch.setattr(exporter, "export_to_onnx", export_stub)
    program = json.loads(
        export_portable_into_bundle(
            tmp_path, checkpoints=["fold0.pt", "fold1.pt"], patch_size=[4, 4], in_channels=1, output_module="Head"
        ).read_text()
    )
    # The identity and the two drawn copies, each run by both folds, folded once.
    models = [step for step in program["steps"] if "model" in step]
    assert len(models) == 2 * 3
    ops = [step for step in program["steps"] if "op" in step]
    means = [step for step in ops if step["op"] == "mean"]
    assert len(means) == 1 and means[0]["in"] == [step["out"] for step in models]
    # The InferenceStack writes every copy, cast to float32, as one stack.
    assert ops[-1]["op"] == "concat" and ops[-1]["out"] == "stack" and len(ops[-1]["in"]) == len(models)


def test_a_masked_export_applies_the_mask_it_leaves_out_of_the_manifest(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import konfai.export as exporter
    import konfai.network.network as network
    import konfai_apps.bundle as bundle

    config = {
        "Predictor": {
            "Model": {"classpath": "Probe"},
            "combine": "Concat",
            "Dataset": {
                "groups_src": {
                    "V": {
                        "groups_dest": {
                            "MASK": {
                                "is_input": False,
                                "transforms": {
                                    "KonfAIInference": {"repo_id": "R/S", "model_name": "body"},
                                    "Dilate": {"dilate": 1},
                                },
                            },
                            "Volume": {"is_input": True, "transforms": {"Normalize": {"inverse": False}}},
                        }
                    }
                }
            },
            "outputs_dataset": {
                "Head": {
                    "OutputDataset": {
                        "before_reduction_transforms": {"Mask": {"path": "MASK", "value_outside": 0}},
                        "after_reduction_transforms": {"InferenceStack": {"mode": "mean"}},
                    }
                }
            },
        }
    }
    (tmp_path / "Prediction.yml").write_text(json.dumps(config))
    model = SimpleNamespace(eval=lambda: None, patch=None)
    monkeypatch.setattr(network, "ModelLoader", lambda _: SimpleNamespace(get_model=lambda **kwargs: model))

    def export_stub(_model, directory, _example, _head, **kwargs):
        path = directory / kwargs["model_filename"]
        path.write_bytes(b"exported portable model")
        return path, {"model": path.name, "patch": {}, **kwargs["extra_manifest"]}

    monkeypatch.setattr(exporter, "export_to_onnx", export_stub)
    nested = {"postprocessing": [{"op": "cast", "dtype": "uint8"}]}
    monkeypatch.setattr(bundle, "export_nested_model", lambda name, _inference, _out: {"id": name, "manifest": nested})

    program = json.loads(
        bundle.export_portable_into_bundle(tmp_path, patch_size=[4, 4], in_channels=1, output_module="Head").read_text()
    )
    # The manifest leaves the Mask out, and the assembled program applies it after the reduction.
    assert program["steps"][1]["manifest"]["postprocessing"] == []
    # One copy folds to itself, so the program holds no reduction: the mask reads the model's own output.
    assert [step["op"] for step in program["steps"] if "op" in step] == ["dilate", "mask"]
    assert program["steps"][-1] == {"op": "mask", "in": ["model", "MASK__0"], "out": "output", "value_outside": 0.0}


def test_repackaging_preserves_a_checkpoint_declared_with_a_relative_prefix(tmp_path):
    app_json = _write(tmp_path / "app.json", VALID_META)
    source = tmp_path / "current.pt"
    source.write_bytes(b"replacement weights")
    bundle = tmp_path / "out" / "MR"
    bundle.mkdir(parents=True)
    (bundle / "app.json").write_text(json.dumps({**VALID_META, "models": ["./current.pt"]}))
    (bundle / "current.pt").write_bytes(b"old weights")

    assemble_bundle("MR", bundle.parent, app_json, [], [str(source)])

    assert (bundle / "current.pt").read_bytes() == b"replacement weights"
    assert json.loads((bundle / "app.json").read_text())["models"] == ["current.pt"]


def _support_workspace(tmp_path):
    root = tmp_path / "workspace"
    (root / "helpers").mkdir(parents=True)
    (root / "helpers" / "__init__.py").write_text("")
    (root / "helpers" / "util.py").write_text("SCALE = 2\n")
    (root / "assets").mkdir()
    (root / "assets" / "table.csv").write_text("a,b\n")
    (root / "Prediction.yml").write_text("Predictor: {}\n")
    (root / "CV_0.pt").write_bytes(b"weights")
    _write(root / "app.json", VALID_META)
    return root


def _assemble_with_support(root, support_files):
    return assemble_bundle(
        "Seg",
        root.parent / "out",
        root / "app.json",
        [str(root / "Prediction.yml")],
        [str(root / "CV_0.pt")],
        support_files=support_files,
        support_root=root,
    )


def test_declared_support_files_land_in_the_bundle_and_the_manifest_lists_them(tmp_path):
    root = _support_workspace(tmp_path)

    bundle = _assemble_with_support(root, {"helpers": "helpers", "assets/table.csv": "assets/table.csv"})

    assert (bundle / "helpers" / "util.py").read_text() == "SCALE = 2\n"
    assert (bundle / "assets" / "table.csv").read_text() == "a,b\n"
    meta = json.loads((bundle / "app.json").read_text())
    assert meta["support_files"] == ["assets/table.csv", "helpers/__init__.py", "helpers/util.py"]

    # Repackaging without the helpers removes the files the previous export managed, and keeps the
    # user's own notes beside them.
    (bundle / "NOTES.md").write_text("mine\n")
    _assemble_with_support(root, {"assets/table.csv": "assets/table.csv"})
    assert not (bundle / "helpers").exists() or not any((bundle / "helpers").iterdir())
    assert (bundle / "NOTES.md").read_text() == "mine\n"
    assert (bundle / "assets" / "table.csv").is_file()


def test_two_spellings_of_one_obsolete_file_unlink_it_once(tmp_path):
    root = _support_workspace(tmp_path)
    (root / "helper.py").write_text("SCALE = 1\n")
    bundle = _assemble_with_support(root, {"helpers": "helper.py"})
    meta = json.loads((bundle / "app.json").read_text())
    meta["support_files"] = ["helpers", "assets/../helpers"]
    (bundle / "app.json").write_text(json.dumps(meta))

    _assemble_with_support(root, {"assets/table.csv": "assets/table.csv"})

    assert not (bundle / "helpers").exists()
    assert (bundle / "assets" / "table.csv").is_file()


def test_a_managed_helper_file_can_become_a_support_directory(tmp_path):
    root = _support_workspace(tmp_path)
    (root / "helper.py").write_text("SCALE = 1\n")

    bundle = _assemble_with_support(root, {"helpers": "helper.py"})
    assert (bundle / "helpers").read_text() == "SCALE = 1\n"

    _assemble_with_support(root, {"helpers": "helpers"})
    assert (bundle / "helpers" / "util.py").read_text() == "SCALE = 2\n"


@pytest.mark.parametrize(
    "support_files, support_root, message",
    [
        ({"helpers": "../outside"}, "workspace", "relative path below"),
        ({"helpers": "<anchor>outside"}, "workspace", "relative path below"),
        ({"../up": "helpers"}, "workspace", "relative path below"),
        ({"helpers": "helpers"}, None, "support_root is required"),
        ({"helpers": "missing"}, "workspace", "Cannot read support path"),
    ],
)
def test_support_files_outside_their_root_are_refused_before_anything_is_written(
    tmp_path, support_files, support_root, message
):
    root = _support_workspace(tmp_path)
    out = root.parent / "out"
    # An absolute path of the host: "/x" is relative on Windows, where the anchor is a drive.
    support_files = {k: v.replace("<anchor>", tmp_path.anchor) for k, v in support_files.items()}

    with pytest.raises(AppMetadataError, match=message):
        assemble_bundle(
            "Seg",
            out,
            root / "app.json",
            [str(root / "Prediction.yml")],
            [str(root / "CV_0.pt")],
            support_files=support_files,
            support_root=None if support_root is None else root,
        )
    assert not out.exists()


def test_a_symlink_escaping_the_support_root_is_refused(tmp_path):
    root = _support_workspace(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("no\n")
    (root / "helpers" / "leak.txt").symlink_to(secret)

    with pytest.raises(AppMetadataError, match="escapes support_root"):
        _assemble_with_support(root, {"helpers": "helpers"})


def test_bundle_cli_maps_support_files_and_drops_local_packages_from_the_requirements_draft(tmp_path, capsys):
    from konfai_apps.bundle import run_bundle_cli

    root = _support_workspace(tmp_path)
    (root / "Model.py").write_text("import einops\nfrom helpers.util import SCALE\n")

    run_bundle_cli(
        {
            "name": "Seg",
            "out": str(root.parent / "out"),
            "app_json": str(root / "app.json"),
            "config": [str(root / "Prediction.yml")],
            "checkpoint": [str(root / "CV_0.pt")],
            "model_py": str(root / "Model.py"),
            "support_file": ["helpers=helpers"],
            "support_root": str(root),
        }
    )

    bundle = root.parent / "out" / "Seg"
    assert (bundle / "helpers" / "util.py").is_file()
    drafted = (bundle / "requirements.txt").read_text().split()
    assert "einops" in drafted and "helpers" not in drafted  # a declared local package is not a PyPI dependency

    with pytest.raises(AppMetadataError, match="DESTINATION=SOURCE"):
        run_bundle_cli(
            {
                "name": "Seg",
                "out": str(root.parent / "out"),
                "app_json": str(root / "app.json"),
                "config": [str(root / "Prediction.yml")],
                "checkpoint": [str(root / "CV_0.pt")],
                "support_file": ["helpers"],
                "support_root": str(root),
            }
        )


@pytest.mark.slow
def test_the_published_mrseg_program_fixture_is_what_the_compiler_emits(tmp_path):
    """The runtime's `mrseg_program.json` fixture is the program this export writes for the published
    MRSegmentator bundle: the same steps in the same order, each model's chains as compiled, the same keys.
    The names of the buffers, of the files and the versions are left to the fixture."""
    import shutil
    from pathlib import Path

    from konfai_apps.bundle import export_portable_into_bundle

    source = Path("~/Documents/hf_bundles/MRSegmentator-KonfAI/MRSegmentator").expanduser()
    folds = sorted(path.name for path in source.glob("fold_*.pt")) if source.is_dir() else []
    if not folds:
        pytest.skip(f"the MRSegmentator bundle and its checkpoints are not at {source}")
    pytest.importorskip("onnxscript")
    fixture = Path(__file__).parents[3] / "konfai-edge" / "tests" / "fixtures" / "mrseg_program.json"
    if not fixture.is_file():
        pytest.skip(f"the runtime's fixture is not in this tree: {fixture}")
    bundle = tmp_path / source.name
    bundle.mkdir()
    for path in source.iterdir():
        if path.suffix == ".pt":
            (bundle / path.name).symlink_to(path)
        elif path.is_file():
            shutil.copy(path, bundle / path.name)

    program = json.loads(export_portable_into_bundle(bundle, checkpoints=folds).read_text())

    def structure(steps: list[dict]) -> list[dict]:
        buffers = {name: f"b{index}" for index, name in enumerate(dict.fromkeys(step["out"] for step in steps))}

        def named(value: object) -> object:
            return [buffers.get(v, v) for v in value] if isinstance(value, list) else buffers.get(value, value)  # type: ignore[arg-type]

        def of(step: dict) -> dict:
            wired = {key: named(value) if key in ("in", "out") else value for key, value in step.items()}
            if "manifest" in step:
                chains = {key: step["manifest"].get(key) for key in ("preprocessing", "postprocessing", "reads")}
                wired["manifest"] = {"keys": sorted(step["manifest"]), **chains}
            return wired

        return [of(step) for step in steps]

    expected = json.loads(fixture.read_text())
    assert program["output"] == expected["output"]
    assert structure(program["steps"]) == structure(expected["steps"])


def test_a_nested_app_without_checkpoints_name_exports_the_default_fold(tmp_path, monkeypatch) -> None:
    # KonfAIInference defaults checkpoints_name to ["fold_0"]; a nested export passing None loads no weights.
    from types import SimpleNamespace

    from konfai_apps import bundle

    seen: dict[str, object] = {}

    def export(nested_dir, checkpoints=None):
        seen["checkpoints"] = checkpoints
        (nested_dir / "model.onnx").write_bytes(b"onnx")
        (nested_dir / "manifest.json").write_text("{}")
        return nested_dir / "model.onnx"

    monkeypatch.setattr(bundle, "export_portable_into_bundle", export)
    app = SimpleNamespace(app_repository=SimpleNamespace(download_app=lambda: []))
    monkeypatch.setattr("konfai_apps.app.KonfAIApp", lambda ref, install, quiet: app)
    result = bundle.export_nested_model("G", {"repo_id": "r", "model_name": "m"}, tmp_path)
    assert seen["checkpoints"] == ["fold_0"]
    assert result["id"] == "G" and (tmp_path / "G.onnx").read_bytes() == b"onnx"


def _nested_app(tmp_path, monkeypatch, files: dict[str, bytes]):
    """A nested app whose bundle holds ``files``, as ``download_app`` hands them over."""
    from types import SimpleNamespace

    source = tmp_path / "source"
    source.mkdir()
    for name, content in files.items():
        (source / name).write_bytes(content)
    repository = SimpleNamespace(download_app=lambda: [(name, source / name) for name in files])
    monkeypatch.setattr(
        "konfai_apps.app.KonfAIApp", lambda ref, install, quiet: SimpleNamespace(app_repository=repository)
    )
    out = tmp_path / "out"
    out.mkdir()
    return out


def test_a_nested_app_s_default_fold_resolves_to_its_checkpoint_file(tmp_path, monkeypatch) -> None:
    # konfai-apps finds the checkpoint 'fold_0' as 'fold_0.pt': the export must load that file's weights.
    from konfai_apps import bundle

    config = json.dumps({"Predictor": {"Model": {"classpath": "Probe"}}}).encode()
    out = _nested_app(tmp_path, monkeypatch, {"Prediction.yml": config, "fold_0.pt": b"weights"})
    seen: list = []

    def compile_output(_config, _root, checkpoints, out_dir, _ids, **_kwargs):
        seen.extend((path, path.is_file()) for path in checkpoints)
        (out_dir / "model.onnx").write_bytes(b"onnx")
        (out_dir / "manifest.json").write_text("{}")
        return [], [out_dir / "model.onnx"], {}

    monkeypatch.setattr(bundle, "compile_output", compile_output)
    bundle.export_nested_model("G", {"repo_id": "r", "model_name": "m"}, out)
    assert [(path.name, exists) for path, exists in seen] == [("fold_0.pt", True)]


@pytest.mark.parametrize("artifact", ["model.onnx", "program.json"])
def test_a_nested_export_is_judged_by_what_it_returns(tmp_path, monkeypatch, artifact: str) -> None:
    # The nested app may publish a program.json of its own: only the export's own result says it compiled to one.
    from konfai_apps import bundle

    out = _nested_app(tmp_path, monkeypatch, {"program.json": b"{}", "fold_0.pt": b"weights"})

    def export(nested_dir, checkpoints=None):
        (nested_dir / "model.onnx").write_bytes(b"onnx")
        (nested_dir / "manifest.json").write_text('{"model": "model.onnx"}')
        return nested_dir / artifact

    monkeypatch.setattr(bundle, "export_portable_into_bundle", export)
    if artifact == "program.json":
        with pytest.raises(AppMetadataError, match="compiles to a program"):
            bundle.export_nested_model("G", {"repo_id": "r", "model_name": "m"}, out)
        return
    result = bundle.export_nested_model("G", {"repo_id": "r", "model_name": "m"}, out)
    assert result == {"id": "G", "manifest": {"model": "model.onnx"}}
    assert (out / "G.onnx").read_bytes() == b"onnx"


def test_multi_checkpoint_nested_model_is_refused(tmp_path):
    # A nested mask/condition producer is copied as `<group>.onnx` + its `manifest.json`. Several
    # checkpoints would make the nested export emit `program.json` + per-fold onnx instead, so the
    # step refuses up front rather than failing later on a missing `model.onnx`.
    from konfai_apps import bundle
    from konfai_apps.errors import AppMetadataError

    inference = {"repo_id": "org/app", "model_name": "body", "checkpoints_name": ["a.pt", "b.pt"]}
    with pytest.raises(AppMetadataError, match="single-checkpoint nested model"):
        bundle.export_nested_model("MASK", inference, tmp_path)

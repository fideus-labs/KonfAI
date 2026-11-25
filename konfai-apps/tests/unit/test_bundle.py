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
import os
import re
import shutil
import sys

import pytest
import torch
from konfai.portable import (
    assemble_program,
    aux_mask_groups,
    derive_blend,
    derive_onnx_params,
    derive_overlap,
    derive_reduction,
    mask_specs,
    transform_manifest,
    tta_passes,
)
from konfai.utils.errors import ConfigError
from konfai.utils.runtime import safe_torch_load, save_weights
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


def _weights(path, value=0.0):
    save_weights({"Model": {"Net": {"weight": torch.full((1,), value)}}}, path)


def test_assemble_bundle_layout(tmp_path):
    app_json = _write(tmp_path / "app.json", VALID_META)
    config = tmp_path / "Prediction.yml"
    config.write_text("Predictor: {}\n")
    checkpoint = tmp_path / "CV_0.safetensors"
    _weights(checkpoint)
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
    for expected in ("app.json", "Prediction.yml", "CV_0.safetensors", "Model.py", "requirements.txt"):
        assert (bundle / expected).exists(), expected
    # `models` auto-filled from the provided checkpoints
    assert json.loads((bundle / "app.json").read_text())["models"] == ["CV_0.safetensors"]


def test_a_runs_checkpoint_is_published_as_weights_only(tmp_path):
    app_json = _write(tmp_path / "app.json", {**VALID_META, "models": ["CV_0.pt"]})
    weights = {"conv.weight": torch.randn(2, 3)}
    checkpoint = tmp_path / "CV_0.pt"
    torch.save({"Model": {"Net": weights}, "epoch": 3, "Net_optimizer_state_dict": {"state": {}}}, checkpoint)

    bundle = assemble_bundle("MR", tmp_path / "out", app_json, [], [str(checkpoint)])

    assert not (bundle / "CV_0.pt").exists()
    assert json.loads((bundle / "app.json").read_text())["models"] == ["CV_0.safetensors"]
    published = safe_torch_load(bundle / "CV_0.safetensors", "cpu")
    assert published.keys() == {"Model"} and torch.equal(
        published["Model"]["Net"]["conv.weight"], weights["conv.weight"]
    )


class _Pickled:
    pass


def test_a_checkpoint_holding_objects_is_refused_not_unpickled(tmp_path):
    app_json = _write(tmp_path / "app.json", VALID_META)
    checkpoint = tmp_path / "CV_0.pt"
    torch.save({"Model": {"Net": {}}, "object": _Pickled()}, checkpoint)
    with pytest.raises(AppMetadataError, match="does not load as KonfAI weights"):
        assemble_bundle("MR", tmp_path / "out", app_json, [], [str(checkpoint)])


def test_a_runs_own_checkpoint_scalars_are_published(tmp_path):
    """A KonfAI run pickles its epoch as a numpy scalar and config floats as ruamel scalars: weights only, still."""
    import numpy as np
    from ruamel.yaml.scalarfloat import ScalarFloat

    app_json = _write(tmp_path / "app.json", VALID_META)
    checkpoint = tmp_path / "CV_0.pt"
    lr = ScalarFloat(0.1)
    _ = lr.anchor  # a ruamel scalar read from a config carries its Anchor, pickled with it
    torch.save({"Model": {"Net": {"weight": torch.zeros(1)}}, "epoch": np.int64(3), "lr": lr}, checkpoint)
    bundle = assemble_bundle("MR", tmp_path / "out", app_json, [], [str(checkpoint)])
    assert bundle and any((tmp_path / "out").rglob("*.safetensors"))


def test_safetensors_weights_in_another_layout_are_refused(tmp_path):
    from safetensors.torch import save_file

    app_json = _write(tmp_path / "app.json", VALID_META)
    weights = tmp_path / "CV_0.safetensors"
    save_file({"conv.weight": torch.zeros(1)}, str(weights))
    with pytest.raises(AppMetadataError, match="not a KonfAI weight name"):
        assemble_bundle("MR", tmp_path / "out", app_json, [], [str(weights)])


def test_safetensors_weights_are_published_as_they_are(tmp_path):
    app_json = _write(tmp_path / "app.json", VALID_META)
    checkpoint = tmp_path / "CV_0.safetensors"
    save_weights({"Model": {"Net": {"weight": torch.zeros(1)}}, "epoch": 3}, checkpoint)
    bundle = assemble_bundle("MR", tmp_path / "out", app_json, [], [str(checkpoint)])
    assert (bundle / "CV_0.safetensors").read_bytes() == checkpoint.read_bytes()


def test_a_support_file_with_a_weight_suffix_is_refused(tmp_path):
    app_json = _write(tmp_path / "app.json", VALID_META)
    (tmp_path / "lut.safetensors").write_bytes(b"data")
    with pytest.raises(AppMetadataError, match="reserves for its models"):
        assemble_bundle(
            "MR",
            tmp_path / "out",
            app_json,
            [],
            [],
            support_files={"lut.safetensors": "lut.safetensors"},
            support_root=tmp_path,
        )


def test_an_app_without_weights_bundles_its_own_model_file(tmp_path):
    """A wrapped tool has no checkpoint: its ``model.py`` keeps its name, the one ``classpath: model:Net`` imports."""
    app_json = _write(tmp_path / "app.json", {**VALID_META, "models": []})
    config = tmp_path / "Prediction.yml"
    config.write_text("Predictor: {}\n")
    (tmp_path / "model.py").write_text("class Net: ...\n")
    bundle = assemble_bundle(
        "Wrapped",
        tmp_path / "dist",
        app_json,
        [str(config)],
        [],
        support_files={"model.py": "model.py"},
        support_root=tmp_path,
    )
    assert json.loads((bundle / "app.json").read_text())["models"] == []
    assert (bundle / "model.py").exists() and (bundle / "Prediction.yml").exists()


def test_missing_required_keys_raises(tmp_path):
    app_json = _write(tmp_path / "app.json", {"display_name": "x"})
    with pytest.raises(AppMetadataError):
        assemble_bundle("MR", tmp_path / "out", app_json, [], [])


def test_models_mismatch_raises(tmp_path):
    app_json = _write(tmp_path / "app.json", {**VALID_META, "models": ["CV_0.safetensors", "CV_1.safetensors"]})
    checkpoint = tmp_path / "CV_0.safetensors"
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
    # A module of the bundle's own is none, by the name it is imported under.
    assert derive_requirements([model_py], local={"skimage"}) == ["segmentation-models-pytorch"]


def test_derive_onnx_params_from_config():
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
    from konfai.utils.utils import get_patch_slices_from_shape

    patch = {"patch_size": patch_size, "overlap": overlap}
    exported, *_ = derive_onnx_params({"Predictor": {"Dataset": {"Patch": patch}}}, "Predictor")
    slices = get_patch_slices_from_shape(patch_size, [1000] * len(patch_size), overlap)
    konfai = []
    for axis, size in enumerate(patch_size):
        starts = sorted({region[axis].start for region in slices})
        if size > 1:  # the axis the manifest keeps
            konfai.append(size - (starts[1] - starts[0]))
    assert derive_overlap(patch, exported) == konfai


def test_derive_blend_reads_patch_combine():
    cfg = {"Predictor": {"outputs_dataset": {"H": {"OutputDataset": {"patch_combine": "Gaussian"}}}}}
    assert derive_blend(cfg, "Predictor") == "Gaussian"
    assert derive_blend({"Predictor": {}}, "Predictor") is None


def test_transform_manifest_maps_the_pipeline_to_runtime_ops():
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
    from konfai.utils.errors import ConfigError

    groups = {"V": {"groups_dest": {"g": {"transforms": {"SomeCustomTransform": {"x": 1}}}}}}
    config = {"Predictor": {"Dataset": {"groups_src": groups}, "outputs_dataset": {}}}
    with pytest.raises(ConfigError, match="no portable runtime op"):
        transform_manifest(config, "Predictor")


def test_the_program_reads_every_occurrence_of_a_repeated_stage():
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
    assert mask_specs(config, "Predictor") == [
        {"group": "BODY", "value_outside": -1024.0},
        {"group": "BONE", "value_outside": 0.0},
        {"group": "AIR", "value_outside": 1.0},
    ]
    assert [op["op"] for op in aux_mask_groups(config, "Predictor")["BODY"]["ops"]] == [
        "resample",
        "dilate",
        "dilate",
    ]
    assert derive_reduction(config, "Predictor") == "merge_labels"


def test_a_chain_spelled_as_a_list_reaches_the_manifest_and_the_program():
    """The list spelling binds in the core as its occurrence mapping; the export reads it the same way
    instead of finding no chain at all."""
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
    manifest, folds, _per_copy = transform_manifest(config, "Predictor", masked=True)
    assert folds == []
    assert [step["op"] for step in manifest["preprocessing"]] == ["clip", "clip", "resample"]
    assert [step["op"] for step in manifest["postprocessing"]] == ["inverse", "argmax"]
    assert [op["op"] for op in aux_mask_groups(config, "Predictor")["BODY"]["ops"]] == ["dilate"]
    assert mask_specs(config, "Predictor") == [{"group": "BODY", "value_outside": 0.0}]
    assert len(tta_passes(config, "Predictor")) == 3


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
        ("transforms", {"Dilate": {"dilate": 1}, "Dilate#2": {"dilate": 2}}),
    ],
)
def test_a_program_stage_no_program_step_applies_is_refused_not_dropped(section, chain):
    """The portable program masks the output only by a group a nested model produces: a mask read from
    the dataset, or a Dilate on the input, would be skipped and the artifact's output left unmasked."""
    config = _masked_config({"BODY": {"is_input": False}, "BONE": {"is_input": False}}, section, chain)
    with pytest.raises(ConfigError):
        transform_manifest(config, "Predictor")


def test_every_mask_on_a_group_a_nested_model_produces_is_left_to_the_program():
    nested = {
        "is_input": False,
        "transforms": {"KonfAIInference": {"repo_id": "R/S", "model_name": "body"}, "Dilate": {"dilate": 2}},
    }
    before = {"Mask": {"path": "BODY"}, "Mask#2": {"path": "BONE"}, "Save": {"dataset": "x"}}
    manifest, _folds, _per_copy = transform_manifest(
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
    manifest, folds, _per_copy = transform_manifest(config, "Predictor", masked=True)
    assert manifest["preprocessing"] == [] and folds == []
    assert manifest["postprocessing"][0] == {"op": "cast", "dtype": "int16"}


def test_colliding_bundle_filenames_are_refused_before_anything_is_written(tmp_path):
    """fold0/best.safetensors and fold1/best.safetensors flattened to one best.safetensors: the second overwrote the first and
    app.json named both."""
    app_json = _write(tmp_path / "app.json", VALID_META)
    config = tmp_path / "Prediction.yml"
    config.write_text("Predictor: {}\n")
    (tmp_path / "fold0").mkdir()
    (tmp_path / "fold1").mkdir()
    (tmp_path / "fold0" / "best.safetensors").write_bytes(b"zero")
    (tmp_path / "fold1" / "best.safetensors").write_bytes(b"one")
    with pytest.raises(AppMetadataError, match=r"best\.safetensors"):
        assemble_bundle(
            "MR",
            tmp_path / "out",
            app_json,
            [str(config)],
            [str(tmp_path / "fold0/best.safetensors"), str(tmp_path / "fold1/best.safetensors")],
        )
    assert not (tmp_path / "out" / "MR").exists()
    assert not list((tmp_path / "out").glob(".MR.staging-*")) if (tmp_path / "out").exists() else True


def test_ensemble_checkpoints_whose_stems_collide_are_refused_before_any_export(tmp_path):
    # Each model's id is its checkpoint's stem: /a/best.pt and /b/best.pt would both write best.onnx.
    from konfai_apps.bundle import export_portable_into_bundle

    (tmp_path / "Prediction.yml").write_text("Predictor:\n  Model:\n    classpath: Model:Net\n")
    with pytest.raises(AppMetadataError, match="share a file stem"):
        export_portable_into_bundle(tmp_path, checkpoints=["/a/best.pt", "/b/best.pt"])


def test_a_pt_name_resolves_to_the_safetensors_the_bundle_publishes(tmp_path):
    from konfai_apps.bundle import _bundle_checkpoint

    (tmp_path / "CV_0.safetensors").write_bytes(b"")
    (tmp_path / "older.pt").write_bytes(b"")
    assert _bundle_checkpoint(tmp_path, "CV_0.pt") == tmp_path / "CV_0.safetensors"
    assert _bundle_checkpoint(tmp_path, "older.pt") == tmp_path / "older.pt"


def test_a_repackaged_bundle_drops_the_previous_checkpoint_and_keeps_user_files(tmp_path):
    """The second export into an existing bundle left the first checkpoint beside the new one, and
    the default inference enumerated the old one. The previous manifest's checkpoints go; a file the
    export does not manage stays."""
    app_json = _write(tmp_path / "app.json", VALID_META)
    config = tmp_path / "Prediction.yml"
    config.write_text("Predictor: {}\n")
    first = tmp_path / "first.safetensors"
    _weights(first)
    bundle = assemble_bundle("MR", tmp_path / "out", app_json, [str(config)], [str(first)])
    (bundle / "notes.txt").write_text("mine\n")

    new = tmp_path / "new.safetensors"
    _weights(new)
    assert assemble_bundle("MR", tmp_path / "out", app_json, [str(config)], [str(new)]) == bundle
    assert sorted(path.name for path in bundle.glob("*.safetensors")) == ["new.safetensors"]
    assert json.loads((bundle / "app.json").read_text())["models"] == ["new.safetensors"]
    assert (bundle / "notes.txt").read_text() == "mine\n"
    assert not list((tmp_path / "out").glob(".MR.staging-*"))


@pytest.mark.parametrize("reference_kind", ["parent", "absolute", "symlink"])
def test_repackaging_refuses_previous_checkpoints_outside_the_bundle(tmp_path, reference_kind):
    app_json = _write(tmp_path / "app.json", VALID_META)
    config = tmp_path / "Prediction.yml"
    config.write_text("Predictor: {}\n")
    checkpoint = tmp_path / "new.safetensors"
    _weights(checkpoint)
    bundle = tmp_path / "out" / "MR"
    bundle.mkdir(parents=True)
    neighbor = bundle.parent / "neighbor.safetensors"
    neighbor.write_bytes(b"must survive")
    previous = "../neighbor.safetensors" if reference_kind == "parent" else str(neighbor)
    if reference_kind == "symlink":
        (bundle / "old.safetensors").symlink_to(neighbor)
        previous = "old.safetensors"
    manifest = json.dumps({**VALID_META, "models": [previous]})
    (bundle / "app.json").write_text(manifest)
    (bundle / "Prediction.yml").write_text("old config\n")

    with pytest.raises(AppMetadataError, match="outside the bundle"):
        assemble_bundle("MR", bundle.parent, app_json, [str(config)], [str(checkpoint)])

    assert neighbor.read_bytes() == b"must survive"
    assert (bundle / "app.json").read_text() == manifest
    assert (bundle / "Prediction.yml").read_text() == "old config\n"
    assert not (bundle / "new.safetensors").exists()
    assert not list(bundle.parent.glob(".MR.staging-*"))


def test_portable_asset_roles_preserve_other_native_onnx_dependencies(tmp_path, monkeypatch):
    from konfai_apps.app_repository import LocalAppRepositoryFromDirectory
    from konfai_apps.bundle import _declare_portable_assets

    monkeypatch.setenv("KONFAI_APPS_INSTALL_REQUIREMENTS", "0")
    bundle = tmp_path / "MR"
    bundle.mkdir()
    metadata = {**VALID_META, "models": ["best.safetensors"], "portable_assets": ["legacy.onnx.data"]}
    (bundle / "app.json").write_text(json.dumps(metadata))
    (bundle / "Prediction.yml").write_text("Predictor: {}\n")
    for name in (
        "best.safetensors",
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


def _stub_export(monkeypatch):
    """A model that loads from nothing, and an ONNX export that writes a stub file and its manifest."""
    from types import SimpleNamespace

    import konfai.export as exporter
    import konfai.network.network as network
    import konfai.utils.runtime as runtime

    model = SimpleNamespace(eval=lambda: None, load=lambda *args, **kwargs: None, patch=None)
    monkeypatch.setattr(network, "ModelLoader", lambda _: SimpleNamespace(get_model=lambda **kwargs: model))
    monkeypatch.setattr(runtime, "safe_torch_load", lambda *args: {})

    def export_stub(_model, directory, _example, _head, **kwargs):
        path = directory / kwargs["model_filename"]
        path.write_bytes(b"exported portable model")
        manifest = {"model": path.name, "input": {"channels": 1}, "output": {"channels": 1}, "patch": {}}
        manifest |= kwargs.get("extra_manifest") or {}
        if kwargs.get("write_manifest"):
            (directory / "manifest.json").write_text(json.dumps(manifest))
        return path, manifest

    monkeypatch.setattr(exporter, "export_to_onnx", export_stub)


@pytest.mark.parametrize("ensemble", [False, True])
def test_portable_export_declares_only_its_own_outputs(tmp_path, monkeypatch, ensemble):
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
    _stub_export(monkeypatch)
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
    _stub_export(monkeypatch)
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
    _stub_export(monkeypatch)
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
    source = tmp_path / "current.safetensors"
    _weights(source, 1.0)
    bundle = tmp_path / "out" / "MR"
    bundle.mkdir(parents=True)
    (bundle / "app.json").write_text(json.dumps({**VALID_META, "models": ["./current.safetensors"]}))
    (bundle / "current.safetensors").write_bytes(b"old weights")

    assemble_bundle("MR", bundle.parent, app_json, [], [str(source)])

    assert safe_torch_load(bundle / "current.safetensors", "cpu")["Model"]["Net"]["weight"].item() == 1.0
    assert json.loads((bundle / "app.json").read_text())["models"] == ["current.safetensors"]


def _support_workspace(tmp_path):
    root = tmp_path / "workspace"
    (root / "helpers").mkdir(parents=True)
    (root / "helpers" / "__init__.py").write_text("")
    (root / "helpers" / "util.py").write_text("SCALE = 2\n")
    (root / "assets").mkdir()
    (root / "assets" / "table.csv").write_text("a,b\n")
    (root / "Prediction.yml").write_text("Predictor: {}\n")
    _weights(root / "CV_0.safetensors")
    _write(root / "app.json", VALID_META)
    return root


def _assemble_with_support(root, support_files):
    return assemble_bundle(
        "Seg",
        root.parent / "out",
        root / "app.json",
        [str(root / "Prediction.yml")],
        [str(root / "CV_0.safetensors")],
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
            [str(root / "CV_0.safetensors")],
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
            "checkpoint": [str(root / "CV_0.safetensors")],
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
                "checkpoint": [str(root / "CV_0.safetensors")],
                "support_file": ["helpers"],
                "support_root": str(root),
            }
        )


#: The least module KonfAI Edge opens: it starts and calls the interface KonfAI Edge calls.
_MODULE = '(module (memory (export "memory") 1) (func (export "ext_abi") (result i32) (i32.const 2)))'


def _cli(root, **more):
    """``konfai-apps bundle`` of the support workspace's app into ``out/Seg``."""
    from konfai_apps.bundle import run_bundle_cli

    args = {
        "name": "Seg",
        "out": str(root.parent / "out"),
        "app_json": str(root / "app.json"),
        "config": [str(root / "Prediction.yml")],
        "checkpoint": [str(root / "CV_0.safetensors")],
    }
    run_bundle_cli(args | more)
    return root.parent / "out" / "Seg"


def _exporting(tmp_path, transforms: dict | None = None) -> tuple:
    """A workspace whose app the bundle command exports for real: the tests' two-class segmentation network, its
    checkpoint, and the arguments that ship and export it."""
    import importlib.util
    from pathlib import Path

    root = _support_workspace(tmp_path)
    shutil.copy(Path(__file__).resolve().parents[3] / "tests" / "assets" / "Workflows" / "TinySeg.py", root)
    # Under a name of its own: a TinySeg another test imported stays its.
    spec = importlib.util.spec_from_file_location("_exported_tinyseg", root / "TinySeg.py")
    assert spec is not None and spec.loader is not None
    tinyseg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tinyseg)
    torch.save({"Model": tinyseg.TinySegNet().network_states()}, root / "CV_0.pt")
    group = {"transforms": transforms or "None", "patch_transforms": "None", "is_input": True}
    config = {
        "Predictor": {
            "Model": {"classpath": "TinySeg:TinySegNet", "TinySegNet": {"outputs_criterions": "None"}},
            "Dataset": {
                "groups_src": {"MR": {"groups_dest": {"MR": group}}},
                "augmentations": "None",
                "Patch": {"patch_size": [1, 16, 16], "overlap": 0},
            },
            "outputs_dataset": {
                "Head:Softmax": {
                    "OutputDataset": {
                        "final_transforms": {"Argmax": {"dim": 0}},
                        "group": "SEG",
                        "same_as_group": "MR:MR",
                    }
                }
            },
        }
    }
    (root / "Prediction.yml").write_text(json.dumps(config))
    exporting = {
        "checkpoint": [str(root / "CV_0.pt")],
        "support_file": ["TinySeg.py=TinySeg.py"],
        "support_root": str(root),
        "onnx": True,
    }
    return root, config, exporting


def test_a_bundle_takes_an_update_only_once_its_export_and_its_extension_are_checked(tmp_path, capsys):
    # The update is exported and checked on the bundle it makes, beside the bundle: an extension that does not load
    # leaves the bundle the config and the program it ran before, the manifest the refused export wrote anew included.
    # The program is told exported once the bundle holds it.
    pytest.importorskip("onnxscript")
    wasmtime = pytest.importorskip("wasmtime")
    root, config, exporting = _exporting(tmp_path)
    (root / "own.wasm").write_bytes(wasmtime.wat2wasm(_MODULE))
    bundle = _cli(root, extension=str(root / "own.wasm"), **exporting)
    assert f"Portable model exported: {bundle / 'model.onnx'}" in capsys.readouterr().out
    before = {path.name: path.read_bytes() for path in bundle.iterdir() if path.is_file()}
    config["Predictor"]["outputs_dataset"]["Head:Softmax"]["OutputDataset"]["final_transforms"] = "None"
    (root / "Prediction.yml").write_text(json.dumps(config))
    (root / "own.wasm").write_bytes(b"\0asm, and no module")

    with pytest.raises(AppMetadataError, match="is no WebAssembly module"):
        _cli(root, extension=str(root / "own.wasm"), **exporting)
    assert "Portable model exported" not in capsys.readouterr().out
    assert "manifest.json" in before and "model.onnx" in before
    assert {path.name: path.read_bytes() for path in bundle.iterdir() if path.is_file()} == before
    assert not list(bundle.parent.glob(".Seg.*"))


def test_an_update_checks_the_program_of_the_prediction_config_the_bundle_keeps(tmp_path):
    # The update names its Evaluation.yml alone: the export compiles the Prediction.yml the bundle keeps, whose
    # class the replacement Model.py now scales by 3, which the extension does not: refused.
    from test_extension import DOUBLE, OWN

    pytest.importorskip("onnxscript")
    wasmtime = pytest.importorskip("wasmtime")
    root, _config, exporting = _exporting(tmp_path, {"Model:Scale": {}})
    (root / "Model.py").write_text(OWN)
    (root / "own.wasm").write_bytes(wasmtime.wat2wasm(DOUBLE))
    _cli(root, model_py=str(root / "Model.py"), extension=str(root / "own.wasm"), **exporting)
    (root / "Model.py").write_text(OWN.replace("scale: float = 2.0", "scale: float = 3.0"))
    (root / "Evaluation.yml").write_text("Evaluator: {}\n")

    with pytest.raises(AppMetadataError, match="'Model:Scale' does not transform a volume"):
        _cli(
            root,
            config=[str(root / "Evaluation.yml")],
            model_py=str(root / "Model.py"),
            extension=str(root / "own.wasm"),
            **exporting,
        )


def test_a_bundle_whose_update_fails_to_take_its_place_is_put_back(tmp_path, monkeypatch):
    # The update checked, the bundle moved aside, and the update refused its place: the bundle is back where it was.
    from pathlib import Path

    wasmtime = pytest.importorskip("wasmtime")
    root = _support_workspace(tmp_path)
    (root / "own.wasm").write_bytes(wasmtime.wat2wasm(_MODULE))
    bundle = _cli(root, extension=str(root / "own.wasm"))
    before = {path.name: path.read_bytes() for path in bundle.iterdir()}
    rename = Path.rename

    def refused(self, target):
        if self.name.startswith(".Seg.update-") and Path(target) == bundle:
            raise OSError("refused")
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", refused)
    with pytest.raises(OSError, match="refused"):
        _cli(root, extension=str(root / "own.wasm"))
    assert {path.name: path.read_bytes() for path in bundle.iterdir()} == before
    assert not list(bundle.parent.glob(".Seg.*"))


def test_a_bundle_whose_previous_copy_stays_is_still_updated(tmp_path, monkeypatch):
    # The update has taken the bundle's place: a copy of the old one that cannot be removed fails nothing, and the
    # next update removes it.
    import konfai_apps.bundle as bundle_module

    wasmtime = pytest.importorskip("wasmtime")
    root = _support_workspace(tmp_path)
    (root / "own.wasm").write_bytes(wasmtime.wat2wasm(_MODULE))
    bundle = _cli(root, extension=str(root / "own.wasm"))
    (root / "Prediction.yml").write_text("Predictor: {}\n# second\n")
    remove = bundle_module.shutil.rmtree

    def kept(path, *args, **kwargs):
        if not str(path).endswith(".Seg.previous"):
            return remove(path, *args, **kwargs)
        if not kwargs.get("ignore_errors"):
            raise OSError("kept")

    monkeypatch.setattr(bundle_module.shutil, "rmtree", kept)
    _cli(root, extension=str(root / "own.wasm"))
    assert (bundle / "Prediction.yml").read_text().endswith("# second\n")
    # The next update removes it first.
    monkeypatch.setattr(bundle_module.shutil, "rmtree", remove)
    assert (bundle.parent / ".Seg.previous").is_dir()
    _cli(root, extension=str(root / "own.wasm"))
    assert not list(bundle.parent.glob(".Seg.*"))


def test_an_export_does_not_write_through_a_link_beside_the_configs(tmp_path):
    # A manifest the bundle holds as a link elsewhere: the export would write the new one there, so it is refused.
    # A link the export does not write (an icon) is the bundle's own, and the export runs beside it.
    pytest.importorskip("onnxscript")
    root, _config, exporting = _exporting(tmp_path)
    bundle = _cli(root, **{key: value for key, value in exporting.items() if key != "onnx"})
    (tmp_path / "elsewhere.json").write_text("not the bundle's")
    (tmp_path / "icon.png").write_bytes(b"png")
    (bundle / "icon.png").symlink_to(tmp_path / "icon.png")
    assert (_cli(root, **exporting) / "model.onnx").is_file()
    (bundle / "manifest.json").unlink()
    (bundle / "manifest.json").symlink_to(tmp_path / "elsewhere.json")

    with pytest.raises(AppMetadataError, match=r"manifest\.json"):
        _cli(root, **exporting)
    assert (tmp_path / "elsewhere.json").read_text() == "not the bundle's"


def test_an_update_whose_export_stops_short_is_refused(tmp_path):
    # The model's module ends the interpreter cleanly before the export: no graph, so no update.
    root, config, exporting = _exporting(tmp_path)
    (root / "Exits.py").write_text("import sys\n\nsys.exit(0)\n")
    config["Predictor"]["Model"]["classpath"] = "Exits:TinySegNet"
    (root / "Prediction.yml").write_text(json.dumps(config))
    exporting["support_file"].append("Exits.py=Exits.py")

    with pytest.raises(AppMetadataError, match="did not finish"):
        _cli(root, **exporting)
    assert not (tmp_path / "out" / "Seg").exists()


def test_an_update_is_checked_with_a_module_found_on_a_relative_import_path(tmp_path, monkeypatch):
    # The process finds a module through a path relative to where it runs: so does the check, which runs elsewhere.
    from test_extension import DOUBLE, OWN

    wasmtime = pytest.importorskip("wasmtime")
    root = _support_workspace(tmp_path)
    (tmp_path / "deps").mkdir()
    (tmp_path / "deps" / "relscales.py").write_text(OWN)
    (root / "Model.py").write_text("from relscales import Scale\n")
    (root / "own.wasm").write_bytes(wasmtime.wat2wasm(DOUBLE))
    (root / "Prediction.yml").write_text("Predictor:\n  Dataset:\n    transforms:\n      Model:Scale: {}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend("deps")
    for module in ("Model", "relscales"):
        monkeypatch.delitem(sys.modules, module, raising=False)

    assert (_cli(root, model_py=str(root / "Model.py"), extension=str(root / "own.wasm")) / "own.wasm").is_file()


def test_bytecode_the_bundle_holds_is_none_the_check_reads(tmp_path):
    # An unchecked bytecode of the old Model.py beside it in the bundle: the check reads the new source, which
    # scales by 3, and refuses.
    import py_compile

    from test_extension import DOUBLE, OWN

    wasmtime = pytest.importorskip("wasmtime")
    root = _support_workspace(tmp_path)
    (root / "Model.py").write_text(OWN)
    (root / "own.wasm").write_bytes(wasmtime.wat2wasm(DOUBLE))
    (root / "Prediction.yml").write_text("Predictor:\n  Dataset:\n    transforms:\n      Model:Scale: {}\n")
    bundle = _cli(root, model_py=str(root / "Model.py"))
    cache = bundle / "__pycache__" / f"Model.{sys.implementation.cache_tag}.pyc"
    py_compile.compile(
        str(bundle / "Model.py"), cfile=str(cache), invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH
    )
    (root / "Model.py").write_text(OWN.replace("scale: float = 2.0", "scale: float = 3.0"))

    with pytest.raises(AppMetadataError, match="'Model:Scale' does not transform a volume"):
        _cli(root, model_py=str(root / "Model.py"), extension=str(root / "own.wasm"))


@pytest.mark.parametrize(
    "planted",
    ["a directory at the manifest", "a link at the requirements the bundle holds"],
)
def test_every_file_a_bundle_s_update_writes_lands_on_nothing_but_a_file_of_its_own(tmp_path, planted):
    # What the update writes, beside the configs or once published, it writes whole: into no directory, through no
    # link (konfai.utils.utils.owned_destination).
    pytest.importorskip("onnxscript")
    root, _config, exporting = _exporting(tmp_path)
    bundle = _cli(root, **{key: value for key, value in exporting.items() if key != "onnx"})
    (tmp_path / "elsewhere.txt").write_text("not the bundle's")
    if planted == "a directory at the manifest":
        (bundle / "manifest.json").mkdir()
        with pytest.raises(AppMetadataError, match=r"manifest\.json' is a directory"):
            _cli(root, **exporting)
    else:
        # The requirements the bundle holds are its author's: no draft is written over them, nor through them.
        (root / "Model.py").write_text("import einops\n")
        (bundle / "requirements.txt").symlink_to(tmp_path / "elsewhere.txt")
        _cli(root, model_py=str(root / "Model.py"))
        assert (bundle / "Model.py").is_file() and (bundle / "requirements.txt").is_symlink()
    assert (tmp_path / "elsewhere.txt").read_text() == "not the bundle's"


@pytest.mark.parametrize("group", ["aux/mask", "..", ""])
def test_a_nested_app_s_group_is_one_name(tmp_path, group):
    # Its graph lands beside the configs as <group>.onnx: a group that names a path is refused before anything is read.
    from konfai_apps.bundle import export_nested_model

    with pytest.raises(AppMetadataError, match="one name"):
        export_nested_model(group, {"repo_id": "a", "model_name": "b"}, tmp_path)


def test_the_import_path_of_a_bundle_s_check_is_the_update_s(tmp_path, monkeypatch):
    # The empty entry is the caller's folder, an entry inside the bundle its counterpart in the update, every other
    # entry absolute; the update comes first.
    from konfai_apps.bundle import _import_path

    bundle, update = tmp_path / "out" / "Seg", tmp_path / "out" / ".Seg.update-1"
    (bundle / "lib").mkdir(parents=True)
    (tmp_path / "link").symlink_to(bundle / "lib")
    monkeypatch.chdir(tmp_path)
    entries = ["", "deps", str(tmp_path / "abs"), str(bundle), str(bundle / "lib"), str(tmp_path / "link")]
    assert _import_path(entries, bundle, update) == [
        str(update.absolute()),
        str(tmp_path),
        str(tmp_path / "deps"),
        str(tmp_path / "abs"),
        str(update.absolute()),
        str(update.absolute() / "lib"),
        str(update.absolute() / "lib"),
    ]


@pytest.mark.parametrize("naming", ["Prediction.yml", "Evaluation.yml", "Uncertainty.yml", "Evaluation.yaml"])
def test_an_update_checks_its_extension_against_every_config_the_bundle_keeps(tmp_path, naming):
    # The update brings a new Model.py, which scales by 3, and no --onnx: a config KonfAI Edge runs names that class,
    # the Prediction.yml the bundle keeps or the Evaluator's config the update brings, and the extension, which
    # doubles, is refused for it.
    from test_extension import DOUBLE, OWN

    wasmtime = pytest.importorskip("wasmtime")
    root = _support_workspace(tmp_path)
    (root / "Model.py").write_text(OWN)
    (root / "own.wasm").write_bytes(wasmtime.wat2wasm(DOUBLE))
    scaled = "  Dataset:\n    transforms:\n      Model:Scale: {}\n"
    (root / "Prediction.yml").write_text("Predictor:\n" + (scaled if naming == "Prediction.yml" else "  {}\n"))
    _cli(root, model_py=str(root / "Model.py"), extension=str(root / "own.wasm"))
    (root / "Model.py").write_text(OWN.replace("scale: float = 2.0", "scale: float = 3.0"))
    brought = "Evaluation.yml" if naming == "Prediction.yml" else naming
    (root / brought).write_text("Evaluator:\n" + (scaled if brought == naming else "  {}\n"))

    with pytest.raises(AppMetadataError, match="'Model:Scale' does not transform a volume"):
        _cli(root, config=[str(root / brought)], model_py=str(root / "Model.py"), extension=str(root / "own.wasm"))


@pytest.mark.parametrize(
    "kept",
    [
        "Trainer:\n  Dataset:\n    transforms:\n      Model:Scale: {scale: 3.0}\n",
        "Trainer:\n  Dataset:\n    transforms:\n      Model:Boxed: {}\n",
        "- notes of the app's, no config\n",
        "Predictor: [a file that does not parse\n",
        b"Predictor: \xff\n",
        None,
    ],
    ids=[
        "a training class the extension does not stand for",
        "a training class bound by no value",
        "no config",
        "no YAML",
        "no text",
        "a folder named as a config",
    ],
)
def test_an_update_leaves_the_training_config_out_of_its_extension_s_check(tmp_path, kept):
    # KonfAI Edge runs a Predictor's or an Evaluator's config, never a Trainer's: the training config, its classes
    # bound by values only training gives, is the bundle's and no check's, as is a file KonfAI Edge could not read.
    from test_extension import DOUBLE, OWN

    wasmtime = pytest.importorskip("wasmtime")
    root = _support_workspace(tmp_path)
    (root / "Model.py").write_text(
        OWN + "\n\nclass Boxed(Scale):\n    def __init__(self, source: str) -> None:\n        super().__init__()\n"
    )
    (root / "own.wasm").write_bytes(wasmtime.wat2wasm(DOUBLE))
    (root / "Prediction.yml").write_text("Predictor:\n  Dataset:\n    transforms:\n      Model:Scale: {}\n")
    held = kept if isinstance(kept, bytes) else (kept or "Trainer: {}\n").encode()
    (root / "Config.yml").write_bytes(held)
    if kept is None:  # a folder the bundle keeps
        (_cli(root) / "notes.yaml").mkdir()

    bundle = _cli(
        root,
        config=[str(root / "Prediction.yml"), str(root / "Config.yml")],
        model_py=str(root / "Model.py"),
        extension=str(root / "own.wasm"),
    )
    assert (bundle / "Config.yml").read_bytes() == held
    assert (bundle / "own.wasm").exists()


@pytest.mark.parametrize(
    "made",
    [
        "a support file's requirements",
        "requirements the bundle holds",
        "a support file's Model.py",
        "a module the update drops",
        "an empty folder of the module's name",
        "a bundle left aside",
    ],
)
def test_the_requirements_drafted_are_those_of_the_bundle_the_update_makes(tmp_path, made):
    # The draft reads the bundle the update makes, where it holds no requirements (a support file's, or the author's
    # it kept): the Model.py a support file brings is read, an import of a module the update drops is a requirement,
    # and a bundle an earlier update left aside is the one updated.
    root = _support_workspace(tmp_path)
    (root / "Model.py").write_text("import einops\n")
    (root / "einops.py").write_text("")
    (root / "listed.txt").write_text("numpy-stl\n")
    if made == "a support file's requirements":
        support = {"support_file": ["requirements.txt=listed.txt"], "support_root": str(root)}
        bundle = _cli(root, model_py=str(root / "Model.py"), **support)
        assert (bundle / "requirements.txt").read_text() == "numpy-stl\n"
        return
    if made == "requirements the bundle holds":
        bundle = _cli(root, model_py=str(root / "Model.py"), requirements=str(root / "listed.txt"))
        _cli(root)
        assert (bundle / "requirements.txt").read_text() == "numpy-stl\n"
        return
    if made == "a support file's Model.py":
        bundle = _cli(root, support_file=["./Model.py=Model.py"], support_root=str(root))
        assert (bundle / "requirements.txt").read_text() == "einops\n"
        return
    if made == "a module the update drops":
        bundle = _cli(
            root, model_py=str(root / "Model.py"), support_file=["einops.py=einops.py"], support_root=str(root)
        )
        assert not (bundle / "requirements.txt").exists()  # einops is the bundle's own module
    elif made == "an empty folder of the module's name":
        bundle = _cli(root, model_py=str(root / "Model.py"))
        (bundle / "requirements.txt").unlink()
        (bundle / "einops").mkdir()  # no module: what is imported is the package
    else:
        bundle = _cli(root, model_py=str(root / "Model.py"))
        (bundle / "requirements.txt").unlink()
        bundle.rename(bundle.with_name(".Seg.previous"))
    _cli(root)
    assert (bundle / "requirements.txt").read_text() == "einops\n"


def test_a_bundle_reached_by_a_link_is_updated_where_the_link_leads(tmp_path):
    # The update takes the place of the folder the link names, which the link then names still, as a direct update
    # writes there.
    root = _support_workspace(tmp_path)
    (root / "Model.py").write_text("import einops\n")
    bundle = _cli(root, model_py=str(root / "Model.py"))
    (bundle / "requirements.txt").unlink()
    (tmp_path / "kept").mkdir()
    bundle.rename(tmp_path / "kept" / "SegV")
    bundle.symlink_to(tmp_path / "kept" / "SegV")
    (root / "Model.py").write_text("import einops\nimport scipy\n")

    _cli(root, model_py=str(root / "Model.py"))
    assert bundle.is_symlink() and (bundle / "Model.py").read_text() == "import einops\nimport scipy\n"
    assert sorted(path.name for path in bundle.parent.iterdir()) == ["Seg"]


def test_a_bundle_s_name_is_none_of_the_command_s_scratch_folders(tmp_path):
    # `.Seg.previous` is the folder an update leaves the bundle in: no bundle takes such a name.
    root = _support_workspace(tmp_path)
    with pytest.raises(AppMetadataError, match=r"starts with '\.'"):
        _cli(root, name=".Seg.previous")


def test_a_link_the_update_re_points_keeps_the_link_it_names(tmp_path):
    # Alias.py names Bridge.py by its absolute path, which names Model.py: re-pointed into the update, it names
    # Bridge.py still, and the update keeps the bundle's chain of links.
    root = _support_workspace(tmp_path)
    (root / "Model.py").write_text("import einops\n")
    bundle = _cli(root, model_py=str(root / "Model.py"))
    (bundle / "requirements.txt").unlink()
    (bundle / "Bridge.py").symlink_to("Model.py")
    (bundle / "Alias.py").symlink_to((bundle / "Bridge.py").absolute())

    _cli(root)
    assert os.readlink(bundle / "Alias.py") == "Bridge.py"


def test_the_check_of_a_bundle_holding_a_module_named_as_python_s_own_runs(tmp_path):
    # A json.py of the bundle's is the app's module, imported once the check is loaded: the check's own json is
    # Python's, as when the app runs.
    from test_extension import DOUBLE, OWN

    wasmtime = pytest.importorskip("wasmtime")
    root = _support_workspace(tmp_path)
    (root / "Model.py").write_text(OWN)
    (root / "own.wasm").write_bytes(wasmtime.wat2wasm(DOUBLE))
    (root / "Prediction.yml").write_text("Predictor:\n  Dataset:\n    transforms:\n      Model:Scale: {}\n")
    bundle = _cli(root, model_py=str(root / "Model.py"))
    (bundle / "json.py").write_text("raise ImportError('the bundle json')\n")

    _cli(root, model_py=str(root / "Model.py"), extension=str(root / "own.wasm"))
    assert (bundle / "own.wasm").is_file()


def test_requirements_are_named_requirements_txt(tmp_path):
    # konfai-apps installs requirements.txt by that name: a support file of another case would be no requirements on
    # a file system telling case apart, and the same file as a draft on one that does not.
    root = _support_workspace(tmp_path)
    (root / "listed.txt").write_text("numpy-stl\n")
    with pytest.raises(AppMetadataError, match=r"Requirements\.txt is named apart from requirements\.txt"):
        _cli(root, support_file=["Requirements.txt=listed.txt"], support_root=str(root))


@pytest.mark.parametrize("kept", ["docs/notes.txt", "cache/lookup.pt"])
def test_a_file_the_bundle_keeps_is_its_own_while_the_update_is_checked(tmp_path, kept):
    # The app's code the check runs writes a file the bundle keeps, in place, and the update is refused: the bundle's
    # file is as it was, the update holding a copy of every file but the checkpoints app.json declares.
    from test_extension import DOUBLE, OWN

    wasmtime = pytest.importorskip("wasmtime")
    root = _support_workspace(tmp_path)
    (root / "Model.py").write_text(OWN)
    (root / "own.wasm").write_bytes(wasmtime.wat2wasm(DOUBLE))
    (root / "Prediction.yml").write_text("Predictor:\n  Dataset:\n    transforms:\n      Model:Scale: {}\n")
    bundle = _cli(root, model_py=str(root / "Model.py"), extension=str(root / "own.wasm"))
    (bundle / kept).parent.mkdir()
    (bundle / kept).write_text("the bundle's")
    written = f'with open("{kept}", "r+") as notes:\n    notes.write("in place")\n'
    (root / "Model.py").write_text(written + OWN.replace("scale: float = 2.0", "scale: float = 3.0"))

    with pytest.raises(AppMetadataError, match="'Model:Scale' does not transform a volume"):
        _cli(root, model_py=str(root / "Model.py"), extension=str(root / "own.wasm"))
    assert (bundle / kept).read_text() == "the bundle's"


def test_a_config_spelled_as_a_kept_one_in_another_case_is_refused(tmp_path):
    # Prediction.yml kept, prediction.yml brought: one file on a file system telling no case apart, and the one
    # konfai reads by its name is the old one on the others.
    root = _support_workspace(tmp_path)
    bundle = _cli(root)
    (root / "prediction.yml").write_text("Predictor: {}\n")
    with pytest.raises(AppMetadataError, match="one file on a file system telling no case apart"):
        _cli(root, config=[str(root / "prediction.yml")])
    assert sorted(path.name for path in bundle.glob("*.yml")) == ["Prediction.yml"]


@pytest.mark.parametrize(
    ("name", "refusal"),
    [
        ("Manifest.json", "Manifest.json takes the name of a file"),
        ("model.onnx", "model.onnx takes the name of a file"),
        ("Requirements.txt", "Requirements.txt takes the name of a file"),
        ("APP.JSON", "'app.json' and 'APP.JSON' are one file"),
    ],
)
def test_an_extension_named_as_a_file_the_bundle_command_writes_is_refused(tmp_path, name, refusal):
    # The export would write its manifest over the extension, the draft its requirements, the command its app.json,
    # in any case: refused before anything is written, a graph's name with or without an export.
    root = _support_workspace(tmp_path)
    (root / "Model.py").write_text("import einops\n")
    (root / "built").mkdir()  # apart from the workspace's app.json, which a file system telling no case apart holds
    (root / "built" / name).write_bytes(b"\0asm")

    with pytest.raises(AppMetadataError, match=re.escape(refusal)):
        _cli(root, model_py=str(root / "Model.py"), extension=str(root / "built" / name), onnx=name == "Manifest.json")
    assert not (tmp_path / "out" / "Seg").exists()


@pytest.mark.parametrize("spelled", ["absolute", "relative"])
def test_a_link_into_the_bundle_reads_the_update_s_file(tmp_path, spelled):
    # Alias.py links to the bundle's Model.py, by its absolute path or out of the bundle and back: the check reads
    # the Model.py the update brings, which scales by 3, as the published bundle will.
    from test_extension import DOUBLE, OWN

    wasmtime = pytest.importorskip("wasmtime")
    root = _support_workspace(tmp_path)
    (root / "Model.py").write_text(OWN)
    (root / "own.wasm").write_bytes(wasmtime.wat2wasm(DOUBLE))
    (root / "Prediction.yml").write_text("Predictor:\n  Dataset:\n    transforms:\n      Alias:Scale: {}\n")
    bundle = _cli(root, model_py=str(root / "Model.py"))
    (bundle / "Alias.py").symlink_to((bundle / "Model.py").resolve() if spelled == "absolute" else "../Seg/Model.py")
    (root / "Model.py").write_text(OWN.replace("scale: float = 2.0", "scale: float = 3.0"))

    with pytest.raises(AppMetadataError, match="'Alias:Scale' does not transform a volume"):
        _cli(root, model_py=str(root / "Model.py"), extension=str(root / "own.wasm"))
    # A link reaching out of the bundle through another link reads what it read.
    (tmp_path / "out" / "shared" / "sub").mkdir(parents=True)
    (tmp_path / "out" / "shared" / "icon.png").write_bytes(b"shared")
    (bundle / "icon.png").write_bytes(b"the bundle's")
    (bundle / "pivot").symlink_to("../shared/sub")
    (bundle / "Alias.png").symlink_to("pivot/../icon.png")
    (root / "Model.py").write_text(OWN)
    _cli(root, model_py=str(root / "Model.py"), extension=str(root / "own.wasm"))
    assert (bundle / "Alias.png").read_bytes() == b"shared"


def test_a_copy_an_earlier_update_left_that_cannot_be_removed_is_named(tmp_path, monkeypatch):
    # The bundle is in place and its old copy beside it cannot be removed: the packaging says which, and stops.
    import konfai_apps.bundle as bundle_module

    root = _support_workspace(tmp_path)
    bundle = _cli(root)
    (bundle.parent / ".Seg.previous").mkdir()
    remove = bundle_module.shutil.rmtree

    def locked(path, *args, **kwargs):
        if str(path).endswith(".Seg.previous") and not kwargs.get("ignore_errors"):
            raise PermissionError("locked")
        return remove(path, *args, **kwargs)

    monkeypatch.setattr(bundle_module.shutil, "rmtree", locked)
    with pytest.raises(AppMetadataError, match=r"\.Seg\.previous is left from an earlier update"):
        _cli(root)


def test_a_bundle_left_only_as_its_previous_copy_is_put_back_first(tmp_path):
    # Both renames refused, the bundle stayed only as its copy beside it: the next packaging, in this process or
    # another, puts it back, user files and all, before it clears its scratch folders.
    root = _support_workspace(tmp_path)
    bundle = _cli(root)
    (bundle / "notes.txt").write_text("mine\n")
    bundle.rename(bundle.with_name(".Seg.previous"))

    _cli(root)
    assert (bundle / "notes.txt").read_text() == "mine\n"
    assert not list(bundle.parent.glob(".Seg.*"))


def test_an_export_whose_model_needs_what_is_missing_fails_the_update(tmp_path, capfd):
    # The extension's check alone stands down without wasmtime: an export whose model needs it fails, on that import.
    root, config, exporting = _exporting(tmp_path)
    (root / "Missing.py").write_text("raise ModuleNotFoundError(\"No module named 'wasmtime'\", name='wasmtime')\n")
    config["Predictor"]["Model"]["classpath"] = "Missing:TinySegNet"
    (root / "Prediction.yml").write_text(json.dumps(config))
    (root / "own.wasm").write_bytes(b"\0asm")
    exporting["support_file"].append("Missing.py=Missing.py")

    with pytest.raises(AppMetadataError):
        _cli(root, extension=str(root / "own.wasm"), **exporting)
    assert "Missing.py" in capfd.readouterr().err
    assert not (tmp_path / "out" / "Seg").exists()


def test_an_update_is_checked_with_the_modules_the_bundle_keeps(tmp_path, monkeypatch, capsys):
    # The Model.py a first packaging carried stays in the bundle when the next one names none: the classes the next
    # config names from it are the app's, and the staged update reads them there.
    from test_extension import DOUBLE, OWN

    wasmtime = pytest.importorskip("wasmtime")
    root = _support_workspace(tmp_path)
    # It reads a file beside it, which the next packaging ships.
    (root / "Model.py").write_text(
        f"from pathlib import Path\n\nTWICE = Path(__file__).with_name('twice.txt').read_text()\n{OWN}"
    )
    (root / "own.wasm").write_bytes(wasmtime.wat2wasm(DOUBLE))
    monkeypatch.delitem(sys.modules, "Model", raising=False)
    (root / "twice.txt").write_text("2")
    support = {"support_file": ["twice.txt=twice.txt"], "support_root": str(root)}
    bundle = _cli(root, model_py=str(root / "Model.py"), **support)
    (root / "Model.py").unlink()
    (root / "Prediction.yml").write_text("Predictor:\n  Dataset:\n    transforms:\n      Model:Scale: {}\n")

    capsys.readouterr()
    assert (_cli(root, extension=str(root / "own.wasm"), **support) / "own.wasm").is_file()
    assert "Extension checked against: Model:Scale" in capsys.readouterr().out
    assert (bundle / "Model.py").read_text().endswith(OWN)


@pytest.mark.parametrize("out", ["absolute", "relative"])
def test_an_update_is_checked_without_the_support_files_it_drops(tmp_path, monkeypatch, out):
    # The Model.py the bundle keeps imports a support module the update no longer declares, which the update removes:
    # the staged check reads the bundle as the update leaves it, and the import fails there, whatever spells the
    # output folder.
    from test_extension import DOUBLE, OWN

    wasmtime = pytest.importorskip("wasmtime")
    root = _support_workspace(tmp_path)
    (root / "Model.py").write_text("from scales import Scale\n")
    (root / "scales.py").write_text(OWN)
    (root / "own.wasm").write_bytes(wasmtime.wat2wasm(DOUBLE))
    for module in ("Model", "scales"):
        monkeypatch.delitem(sys.modules, module, raising=False)
    (root / "Prediction.yml").write_text("Predictor:\n  Dataset:\n    transforms:\n      Model:Scale: {}\n")
    # Checked once with it, which imports it.
    support = {"support_file": ["scales.py=scales.py"], "support_root": str(root)}
    monkeypatch.chdir(tmp_path)
    folder = {"out": "out"} if out == "relative" else {}
    bundle = _cli(root, model_py=str(root / "Model.py"), extension=str(root / "own.wasm"), **support, **folder)

    with pytest.raises(AppMetadataError, match="'Model:Scale' does not import"):
        _cli(root, extension=str(root / "own.wasm"), **folder)
    assert (bundle / "scales.py").read_text() == OWN


def test_an_update_is_checked_without_a_module_the_process_imported_from_the_bundle(tmp_path, monkeypatch):
    # The process imported the bundle's support module, which the update drops: the check does not take it from there.
    from test_extension import DOUBLE, OWN

    wasmtime = pytest.importorskip("wasmtime")
    root = _support_workspace(tmp_path)
    (root / "Model.py").write_text("from dropped import Scale\n")
    (root / "dropped.py").write_text(OWN)
    (root / "own.wasm").write_bytes(wasmtime.wat2wasm(DOUBLE))
    (root / "Prediction.yml").write_text("Predictor:\n  Dataset:\n    transforms:\n      Model:Scale: {}\n")
    support = {"support_file": ["dropped.py=dropped.py"], "support_root": str(root)}
    bundle = _cli(root, model_py=str(root / "Model.py"), **support)
    for module in ("Model", "dropped"):
        monkeypatch.delitem(sys.modules, module, raising=False)
    sys.path.insert(0, str(bundle))
    try:
        import dropped  # noqa: F401
    finally:
        sys.path.remove(str(bundle))

    with pytest.raises(AppMetadataError, match="'Model:Scale' does not import"):
        _cli(root, extension=str(root / "own.wasm"))
    # Nor from the bundle on the path the process imports from, which it keeps.
    monkeypatch.delitem(sys.modules, "dropped")
    monkeypatch.syspath_prepend(str(bundle))
    with pytest.raises(AppMetadataError, match="'Model:Scale' does not import"):
        _cli(root, extension=str(root / "own.wasm"))
    assert sys.path[0] == str(bundle)


def test_an_update_is_checked_without_a_folder_of_the_bundle_on_the_import_path(tmp_path, monkeypatch):
    # The process imports from a folder inside the bundle, which the update drops: the check does not read it.
    from test_extension import DOUBLE, OWN

    wasmtime = pytest.importorskip("wasmtime")
    root = _support_workspace(tmp_path)
    (root / "lib").mkdir()
    (root / "lib" / "libscales.py").write_text(OWN)
    (root / "Model.py").write_text("from libscales import Scale\n")
    (root / "own.wasm").write_bytes(wasmtime.wat2wasm(DOUBLE))
    (root / "Prediction.yml").write_text("Predictor:\n  Dataset:\n    transforms:\n      Model:Scale: {}\n")
    bundle = _cli(root, model_py=str(root / "Model.py"), support_file=["lib=lib"], support_root=str(root))
    monkeypatch.syspath_prepend(str(bundle / "lib"))
    for module in ("Model", "libscales"):
        monkeypatch.delitem(sys.modules, module, raising=False)

    with pytest.raises(AppMetadataError, match="'Model:Scale' does not import"):
        _cli(root, extension=str(root / "own.wasm"))


def test_a_second_packaging_in_one_process_checks_its_own_modules(tmp_path, monkeypatch, capsys):
    # A Model imported before is not the first packaging's, and the first packaging's is not the second's, which
    # scales by 3: each checks its own, the second is refused, and the Model imported before is the process's again.
    from types import ModuleType

    from test_extension import DOUBLE, OWN

    wasmtime = pytest.importorskip("wasmtime")
    root = _support_workspace(tmp_path)
    (root / "own.wasm").write_bytes(wasmtime.wat2wasm(DOUBLE))
    (root / "Prediction.yml").write_text("Predictor:\n  Dataset:\n    transforms:\n      Model:Scale: {}\n")
    before = ModuleType("Model")
    monkeypatch.setitem(sys.modules, "Model", before)
    for scale in (2.0, 3.0):
        (root / "Model.py").write_text(OWN.replace("scale: float = 2.0", f"scale: float = {scale}"))
        if scale == 2.0:
            _cli(root, model_py=str(root / "Model.py"), extension=str(root / "own.wasm"))
            assert "Extension checked against: Model:Scale" in capsys.readouterr().out
            continue
        with pytest.raises(
            AppMetadataError, match="'Model:Scale' does not transform a volume as the Python class does"
        ):
            _cli(root, model_py=str(root / "Model.py"), extension=str(root / "own.wasm"))
    # The process keeps the Model it held, and no check wrote bytecode into the bundle.
    assert sys.modules["Model"] is before
    assert not list((tmp_path / "out" / "Seg").rglob("__pycache__"))


def test_an_extension_is_planned_with_the_bundle_s_other_files(tmp_path):
    # Its name beside a support file of another source is refused before anything is written, a symlink in the
    # bundle under its name is not written through, and an extension of its own name lands.
    module = pytest.importorskip("wasmtime").wat2wasm(_MODULE)
    root = _support_workspace(tmp_path)
    (root / "engine.wasm").write_bytes(b"a support file")
    (tmp_path / "built").mkdir()
    (tmp_path / "built" / "engine.wasm").write_bytes(module)
    support = {"support_file": ["engine.wasm=engine.wasm"], "support_root": str(root)}

    with pytest.raises(AppMetadataError, match=r"Different bundle files would overwrite 'engine\.wasm'"):
        _cli(root, extension=str(tmp_path / "built" / "engine.wasm"), **support)
    assert not (tmp_path / "out").exists()
    bundle = _cli(root)
    (tmp_path / "elsewhere.wasm").write_bytes(b"not the bundle's")
    (bundle / "engine.wasm").symlink_to(tmp_path / "elsewhere.wasm")
    with pytest.raises(AppMetadataError, match="symlink"):
        _cli(root, extension=str(tmp_path / "built" / "engine.wasm"))
    assert (tmp_path / "elsewhere.wasm").read_bytes() == b"not the bundle's"
    (bundle / "engine.wasm").unlink()
    (tmp_path / "built" / "own.wasm").write_bytes(module)
    assert (_cli(root, extension=str(tmp_path / "built" / "own.wasm"), **support) / "own.wasm").read_bytes() == module


@pytest.mark.slow
def test_the_published_mrseg_program_fixture_is_what_the_compiler_emits(tmp_path):
    """The runtime's `mrseg_program.json` fixture is the program this export writes for the published
    MRSegmentator bundle: the same steps in the same order, each model's chains as compiled, the same keys.
    The names of the buffers, of the files and the versions are left to the fixture."""
    import os
    import shutil
    from pathlib import Path

    from konfai_apps.bundle import export_portable_into_bundle

    source = Path(os.environ.get("KONFAI_MRSEG_BUNDLE") or pytest.skip("KONFAI_MRSEG_BUNDLE names no bundle"))
    folds = sorted(path.name for path in source.glob("fold_*.pt"))
    if not folds:
        pytest.skip(f"no fold_*.pt checkpoint under {source}")
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


@pytest.mark.parametrize(
    ("weights", "resolved"),
    [(["fold_0.safetensors", "fold_0.pt"], "fold_0.safetensors"), (["fold_0.pt"], "fold_0.pt")],
)
def test_a_nested_app_s_default_fold_resolves_to_its_checkpoint_file(tmp_path, monkeypatch, weights, resolved) -> None:
    # konfai-apps finds the checkpoint 'fold_0' as the bundle's weights, .safetensors before an older .pt: the
    # export must load that file's weights.
    from konfai_apps import bundle

    config = json.dumps({"Predictor": {"Model": {"classpath": "Probe"}}}).encode()
    out = _nested_app(tmp_path, monkeypatch, {"Prediction.yml": config, **dict.fromkeys(weights, b"weights")})
    seen: list = []

    def compile_output(_config, _root, checkpoints, out_dir, _ids, **_kwargs):
        seen.extend((path, path.is_file()) for path in checkpoints)
        (out_dir / "model.onnx").write_bytes(b"onnx")
        (out_dir / "manifest.json").write_text("{}")
        return [], [out_dir / "model.onnx"], {}

    monkeypatch.setattr(bundle, "compile_output", compile_output)
    bundle.export_nested_model("G", {"repo_id": "r", "model_name": "m"}, out)
    assert [(path.name, exists) for path, exists in seen] == [(resolved, True)]


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

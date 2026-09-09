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

"""MONAI Bundles both ways: a bundle's network and weights predict through KonfAI; a KonfAI network's
head comes back as a bundle any MONAI runtime loads."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

monai = pytest.importorskip("monai")
sitk = pytest.importorskip("SimpleITK")

from konfai import api  # noqa: E402
from konfai.bundle import export_bundle, import_bundle  # noqa: E402
from konfai.utils.errors import ConfigError  # noqa: E402

NETWORK = {
    "_target_": "monai.networks.nets.UNet",
    "spatial_dims": "@spatial_dims",
    "in_channels": 1,
    "out_channels": 3,
    "channels": [4, 8, 16],
    "strides": [2, 2],
    "num_res_units": 1,
}


def _write_bundle(root: Path, network: torch.nn.Module) -> Path:
    """A bundle as the model zoo ships one: metadata, an inference config with references and
    expressions, the plain state dict."""
    (root / "configs").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "metadata.json").write_text(json.dumps({"version": "0.1.0", "task": "toy segmentation"}))
    config = {
        "imports": ["$import torch"],
        "bundle_root": ".",
        "spatial_dims": 2,
        "network_def": NETWORK,
        "network": "$@network_def.to('cpu')",
        "preprocessing": {"_target_": "Compose", "transforms": [{"_target_": "LoadImaged", "keys": "image"}]},
        "inferer": {"_target_": "SlidingWindowInferer", "roi_size": [16, 16], "sw_batch_size": 1},
    }
    (root / "configs" / "inference.json").write_text(json.dumps(config))
    torch.save(network.state_dict(), root / "models" / "model.pt")
    return root


def test_a_bundle_s_network_and_weights_predict_through_konfai(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from monai.networks.nets import UNet

    torch.manual_seed(0)
    network = UNet(spatial_dims=2, in_channels=1, out_channels=3, channels=(4, 8, 16), strides=(2, 2), num_res_units=1)
    bundle = _write_bundle(tmp_path / "toy_bundle", network)

    imported = import_bundle(bundle)

    assert imported.classpath == "monai.networks.nets:UNet" and imported.name == "UNet"
    assert imported.arguments["spatial_dims"] == 2, "the bundle's @reference is resolved, not copied"
    assert imported.arguments["channels"] == [4, 8, 16]
    assert imported.checkpoint == bundle / "models" / "konfai_model.pt" and imported.checkpoint.is_file()
    assert set(imported.untranslated) == {"preprocessing", "inferer"}, "what KonfAI does not translate is reported"
    assert imported.metadata["task"] == "toy segmentation"
    tree = imported.model_tree()
    assert tree["classpath"] == "monai.networks.nets:UNet" and tree["UNet"]["dim"] == 2

    # The imported model predicts: the checkpoint loads through the wrapper, on a KonfAI dataset.
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(1)
    for case in ("P000", "P001"):
        (tmp_path / "Raw" / case).mkdir(parents=True)
        sitk.WriteImage(
            sitk.GetImageFromArray(rng.normal(0.0, 1.0, (2, 16, 16)).astype(np.float32)),
            str(tmp_path / "Raw" / case / "CT.mha"),
        )
    prediction = {
        "Predictor": {
            "Model": tree,
            "Dataset": {
                "groups_src": {
                    "CT": {"groups_dest": {"CT": {"transforms": None, "patch_transforms": None, "is_input": True}}}
                },
                "augmentations": None,
                "Patch": {"patch_size": [1, 16, 16], "overlap": None, "pad_value": 0, "extend_slice": 0},
                "dataset_filenames": ["./Raw:mha"],
                "batch_size": 1,
            },
            "outputs_dataset": {
                "Model": {
                    "OutputDataset": {
                        "name_class": "OutputDataset",
                        # Left out, each of the three defaults to a Normalize on the output.
                        "before_reduction_transforms": None,
                        "after_reduction_transforms": None,
                        "final_transforms": None,
                        "dataset_filename": "./Pred:mha",
                        "group": "PRED",
                        "same_as_group": "CT:CT",
                        "reduction": "Mean",
                    }
                }
            },
            "train_name": "BUNDLE",
            "autocast": False,
            "combine": "Mean",
        }
    }
    workspace = api.predict(imported.checkpoint, prediction, predictions_dir=tmp_path / "Predictions", quiet=True)
    predicted = sitk.GetArrayFromImage(sitk.ReadImage(str(workspace / "Pred" / "P000" / "PRED.mha")))
    assert predicted.shape == (2, 16, 16, 3)

    # And the same as the network itself, run by hand on one slice: the weights are the bundle's.
    volume = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "Raw" / "P000" / "CT.mha")))
    with torch.no_grad():
        expected = network.eval()(torch.from_numpy(volume[0])[None, None]).numpy()[0]
    # Within float16: the prediction route accumulates and writes its output in half precision.
    np.testing.assert_allclose(np.moveaxis(predicted[0], -1, 0), expected, rtol=1e-3, atol=2e-3)


def test_a_konfai_network_s_head_becomes_a_bundle_monai_loads(tmp_path: Path) -> None:
    from konfai.network.network import MinimalModel
    from monai.networks.nets import UNet

    torch.manual_seed(0)
    network = MinimalModel(
        UNet(spatial_dims=2, in_channels=1, out_channels=3, channels=(4, 8, 16), strides=(2, 2), num_res_units=1), dim=2
    )
    example = torch.randn(1, 1, 16, 16)
    root = export_bundle(network, example, tmp_path / "out_bundle", name="toy", task="toy segmentation")

    assert (root / "models" / "model.ts").is_file() and (root / "configs" / "inference.json").is_file()
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["network_data_format"]["outputs"]["pred"]["num_channels"] == 3
    assert metadata["konfai"]["output_module"] == "Model"

    traced = torch.jit.load(str(root / "models" / "model.ts"))
    with torch.no_grad():
        expected = dict(network.eval().named_forward(example))["Model"]
        torch.testing.assert_close(traced(example), expected)
    # What MONAI's own config runtime makes of it: the bundle's network entry, evaluated.
    parser = monai.bundle.ConfigParser()
    parser.read_config(str(root / "configs" / "inference.json"))
    parser["bundle_root"] = str(root)
    module = parser.get_parsed_content("network")
    with torch.no_grad():
        torch.testing.assert_close(module(example.to(parser.get_parsed_content("device"))).cpu(), expected)


def test_a_bundle_without_its_config_or_weights_is_refused_by_name(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ConfigError, match=r"inference\.json"):
        import_bundle(tmp_path / "empty")

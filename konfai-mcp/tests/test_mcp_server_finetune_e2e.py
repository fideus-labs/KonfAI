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

"""fine_tune_app end to end, black-box through the MCP client: real app, real training, changed weights."""

import asyncio
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from mcp_test_helpers import create_synthesis_dataset, resource_to_text

fastmcp = pytest.importorskip("fastmcp")
pytest.importorskip("SimpleITK")

pytestmark = pytest.mark.slow

# The tiny synthesis training assets shared with the konfai-apps fine-tune integration suite.
WORKFLOW_ASSETS_DIR = Path(__file__).resolve().parents[2] / "tests" / "assets" / "Workflows"

_PRETRAINED_EPOCH = 10
_PRETRAINED_IT = 125134


def _write_finetunable_app(app_dir: Path) -> None:
    """A minimal REAL app bundle: manifest + train config + model code + a genuine checkpoint."""
    import torch

    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Tiny Synth",
                "description": "Tiny local synthesis app for the fine-tune e2e",
                "short_description": "Tiny synth",
                "task": "synthesis",
                "tta": 0,
                "mc_dropout": 0,
                "models": ["tiny_0.pt"],
                "inputs": {"MR": {"display_name": "MR", "volume_type": "VOLUME", "required": True}},
                "outputs": {"sCT": {"display_name": "sCT", "volume_type": "VOLUME", "required": True}},
            }
        ),
        encoding="utf-8",
    )
    config = (WORKFLOW_ASSETS_DIR / "Config.yml").read_text(encoding="utf-8")
    config = config.replace("__DATASET_DIR__", "./Dataset").replace("__TRAIN_NAME__", "FT")
    (app_dir / "Config.yml").write_text(config, encoding="utf-8")
    (app_dir / "TinySynth.py").write_text(
        (WORKFLOW_ASSETS_DIR / "TinySynth.py").read_text(encoding="utf-8"), encoding="utf-8"
    )

    spec = importlib.util.spec_from_file_location("TinySynth", app_dir / "TinySynth.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.save(
        {
            "epoch": _PRETRAINED_EPOCH,
            "it": _PRETRAINED_IT,
            "loss": 0.0,
            "Model": module.TinySynthNet().network_states(),
        },
        app_dir / "tiny_0.pt",
    )


def _model_tensors(checkpoint_path: Path) -> dict[str, "object"]:
    import torch

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    flat: dict[str, object] = {}
    for network_name, network_state in state["Model"].items():
        for key, tensor in network_state.items():
            flat[f"{network_name}.{key}"] = tensor
    return flat


@pytest.mark.usefixtures("workspace_root")
def test_fine_tune_app_end_to_end_produces_a_bundle_with_changed_weights(
    tmp_path: Path, load_mcp_server: Callable[[], ModuleType]
) -> None:
    import torch

    app_dir = tmp_path / "TinySynthApp"
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "FTBundle"
    _write_finetunable_app(app_dir)
    create_synthesis_dataset(dataset_dir)

    mcp_server = load_mcp_server()

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            described = await client.call_tool("describe_app", {"ref": str(app_dir)})
            assert described.structured_content["finetunable"] is True
            assert "fine_tune_app" in described.structured_content["next_actions"]

            job = await client.call_tool(
                "fine_tune_app",
                {
                    "ref": str(app_dir),
                    "dataset": str(dataset_dir),
                    "output": str(output_dir),
                    "name": "FT",
                    "epochs": 1,
                    "it_validation": 1,
                    "cpu": 1,
                    "allow_untrusted_code": True,
                },
            )
            payload = job.structured_content
            assert payload["kind"] == "finetune"
            assert payload["output"] == str(output_dir)

            done = await client.call_tool(
                "wait_for_job", {"job_id": payload["job_id"], "timeout_s": 300.0, "poll_interval_s": 0.5}
            )
            if done.structured_content["status"] != "done":
                log = await client.read_resource(f"job://{payload['job_id']}/log")
                raise AssertionError(f"fine_tune_app failed: {done.structured_content}\n{resource_to_text(log)}")
            assert "run_app_infer" in done.structured_content["next_actions"]

    asyncio.run(scenario())

    # The produced bundle is a resolvable app: manifest + train config + code + fine-tuned checkpoint.
    metadata = json.loads((output_dir / "app.json").read_text(encoding="utf-8"))
    assert metadata["models"] == ["tiny_0.pt"]
    assert (output_dir / "Config.yml").is_file()
    assert (output_dir / "TinySynth.py").is_file()

    # The training really ran: counters restarted from the sanitized weights-only checkpoint...
    produced = torch.load(output_dir / "tiny_0.pt", map_location="cpu", weights_only=False)
    assert produced["epoch"] < _PRETRAINED_EPOCH
    assert 0 < produced["it"] < _PRETRAINED_IT

    # ...and at least one weight tensor moved away from the input app's checkpoint.
    before = _model_tensors(app_dir / "tiny_0.pt")
    after = _model_tensors(output_dir / "tiny_0.pt")
    assert set(after) == set(before)
    assert any(not torch.equal(before[key], after[key]) for key in before), "fine-tune left every weight unchanged"

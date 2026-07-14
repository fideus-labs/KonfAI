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

import asyncio
import io
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from mcp_test_helpers import create_synthesis_dataset, install_fake_konfai_runtime, resource_to_text, run_job
from ruamel.yaml import YAML

fastmcp = pytest.importorskip("fastmcp")
pytest.importorskip("SimpleITK")

yaml = YAML()
yaml.default_flow_style = False

MODEL_SOURCE = """
import torch
from konfai.network import network


class Head(network.ModuleArgsDict):
    def __init__(self):
        super().__init__()
        self.add_module("Tanh", torch.nn.Tanh())


class TinySynthNet(network.Network):
    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {"default|ConstantLR": network.LRSchedulersLoader(0)},
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"Head:Tanh": network.TargetCriterionsLoader()},
    ) -> None:
        super().__init__(
            in_channels=1,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            dim=2,
        )
        self.add_module("Projection", torch.nn.Conv2d(1, 1, kernel_size=1, bias=True))
        self.add_module("Head", Head())
""".strip()


def _yaml_dump(data: dict[str, Any]) -> str:
    stream = io.StringIO()
    yaml.dump(data, stream)
    return stream.getvalue()


def _train_config(dataset_dir: Path, train_name: str) -> str:
    return _yaml_dump(
        {
            "Trainer": {
                "Model": {
                    "classpath": "TinySynth:TinySynthNet",
                    "TinySynthNet": {
                        "outputs_criterions": None,
                        "Patch": None,
                    },
                },
                "Dataset": {
                    "groups_src": {
                        "MR": {"groups_dest": {"MR": {"transforms": None, "patch_transforms": None, "is_input": True}}},
                        "CT": {
                            "groups_dest": {"CT": {"transforms": None, "patch_transforms": None, "is_input": False}}
                        },
                    },
                    "Patch": {
                        "patch_size": [1, 16, 16],
                        "overlap": None,
                        "mask": None,
                        "pad_value": 0,
                        "extend_slice": 0,
                    },
                    "dataset_filenames": [f"{dataset_dir}:a:mha"],
                    "augmentations": None,
                    "batch_size": 4,
                    "use_cache": False,
                    "validation": 0.25,
                },
                "train_name": train_name,
                "epochs": 1,
                "it_validation": 1,
            }
        }
    )


def _prediction_config(dataset_dir: Path, train_name: str) -> str:
    return _yaml_dump(
        {
            "Predictor": {
                "Model": {"classpath": "TinySynth:TinySynthNet", "TinySynthNet": {"outputs_criterions": None}},
                "Dataset": {
                    "groups_src": {
                        "MR": {"groups_dest": {"MR": {"transforms": None, "patch_transforms": None, "is_input": True}}}
                    },
                    "Patch": {
                        "patch_size": [1, 16, 16],
                        "overlap": None,
                        "mask": None,
                        "pad_value": 0,
                        "extend_slice": 0,
                    },
                    "dataset_filenames": [f"{dataset_dir}:a:mha"],
                    "augmentations": None,
                    "batch_size": 4,
                    "use_cache": False,
                },
                "outputs_dataset": {
                    "Head:Tanh": {
                        "OutputDataset": {
                            "name_class": "OutSameAsGroupDataset",
                            "before_reduction_transforms": None,
                            "after_reduction_transforms": None,
                            "final_transforms": None,
                            "dataset_filename": "Dataset:mha",
                            "group": "sCT",
                            "same_as_group": "MR:MR",
                            "patch_combine": None,
                            "inverse_transform": False,
                            "reduction": "Mean",
                            "Mean": {},
                        }
                    }
                },
                "train_name": train_name,
            }
        }
    )


def _evaluation_config(dataset_dir: Path, train_name: str) -> str:
    return _yaml_dump(
        {
            "Evaluator": {
                "Dataset": {"dataset_filenames": [f"{dataset_dir}:a:mha", f"./Predictions/{train_name}/Dataset:i:mha"]},
                "groups_src": {"CT": "CT", "sCT": "sCT"},
                "Metrics": {"CT:sCT": {"MAE": {}}},
                "train_name": train_name,
            }
        }
    )


def test_mcp_server_session_pipeline_with_local_support_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    workspace_root = tmp_path / "workspaces"
    dataset_dir = tmp_path / "dataset"
    create_synthesis_dataset(dataset_dir)
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(workspace_root))

    mcp_server = load_mcp_server()
    install_fake_konfai_runtime(tmp_path, monkeypatch, mcp_server)
    client_cls = fastmcp.Client

    async def scenario() -> None:
        async with client_cls(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {"overwrite": True})
            await client.call_tool(
                "write_session_file",
                {"relative_path": "TinySynth.py", "content": MODEL_SOURCE},
            )
            inspected_model = await client.call_tool(
                "inspect_object_signature",
                {"classpath": "TinySynth:TinySynthNet"},
            )
            assert inspected_model.structured_content["ok"] is True

            await client.call_tool(
                "write_workflow_config",
                {"workflow": "train", "content": _train_config(dataset_dir, "baseline")},
            )
            await client.call_tool(
                "write_workflow_config",
                {"workflow": "prediction", "content": _prediction_config(dataset_dir, "baseline")},
            )
            await client.call_tool(
                "write_workflow_config",
                {"workflow": "evaluation", "content": _evaluation_config(dataset_dir, "baseline")},
            )

            train = await run_job(
                client, "run_train", {"cpu": 1, "overwrite": True, "quiet": True, "single_process": True}
            )
            assert train["status"] == "done"
            prediction = await run_job(
                client, "run_prediction", {"cpu": 1, "overwrite": True, "quiet": True, "single_process": True}
            )
            assert prediction["status"] == "done"
            evaluation = await run_job(
                client, "run_evaluation", {"cpu": 1, "overwrite": True, "quiet": True, "single_process": True}
            )
            assert evaluation["status"] == "done"

            summary = await client.call_tool("summarize_session", {"leaderboard_metric": "MAE"})
            summary_data = summary.structured_content
            assert summary_data["latest_job"]["status"] == "done"

            metrics = await client.read_resource("session://current/metrics")
            metrics_text = resource_to_text(metrics)
            assert "PRED:SEG:MAE" in metrics_text or "MAE" in metrics_text

            leaderboard = await client.call_tool("leaderboard", {"metric": "MAE", "split": "TRAIN"})
            assert leaderboard.structured_content["best"]["run_name"] == "baseline"

    asyncio.run(scenario())

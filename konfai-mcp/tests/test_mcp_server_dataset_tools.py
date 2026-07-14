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
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

fastmcp = pytest.importorskip("fastmcp")
sitk = pytest.importorskip("SimpleITK")


def _write_image(path: Path, array: np.ndarray, pixel_id: int) -> None:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    image = sitk.Cast(image, pixel_id)
    sitk.WriteImage(image, str(path))


def _create_alias_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(2):
        case_dir = dataset_dir / f"CASE_{idx:03d}"
        case_dir.mkdir()
        img = np.full((3, 8, 8), fill_value=idx + 1, dtype=np.float32)
        seg = np.zeros((3, 8, 8), dtype=np.uint8)
        seg[:, 2:6, 2:6] = idx + 1
        _write_image(case_dir / "IMG.mha", img, sitk.sitkFloat32)
        _write_image(case_dir / "SEG.mha", seg, sitk.sitkUInt8)


def _metric_json(value: float, metric_name: str) -> str:
    return json.dumps(
        {
            "case": {metric_name: {"CASE_000": value}},
            "aggregates": {
                metric_name: {
                    "max": value,
                    "min": value,
                    "std": 0.0,
                    "25pc": value,
                    "50pc": value,
                    "75pc": value,
                    "mean": value,
                    "count": 1.0,
                }
            },
        },
        indent=2,
    )


def test_mcp_server_dataset_inspection_and_aliasing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    workspace_root = tmp_path / "workspaces"
    dataset_dir = tmp_path / "dataset"
    _create_alias_dataset(dataset_dir)
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(workspace_root))

    mcp_server = load_mcp_server()
    client_cls = fastmcp.Client

    async def scenario() -> None:
        async with client_cls(mcp_server.mcp) as client:
            inferred = await client.call_tool(
                "inspect_dataset", {"dataset_dir": str(dataset_dir), "include_stats": False}
            )
            inferred_data = inferred.structured_content
            assert {"IMG", "SEG"} == set(inferred_data["groups"])

            # Role is knowledge, not a guess: is_input is left null for every group (the agent
            # decides it from the user's task) and the payload explains what is_input means, instead
            # of forcing a name-based guess that mis-wires segmentation (CT input) vs synthesis.
            suggested = inferred_data["suggested_groups_src"]
            assert suggested["IMG"]["groups_dest"]["IMG"]["is_input"] is None
            assert suggested["SEG"]["groups_dest"]["SEG"]["is_input"] is None
            assert "is_input" in inferred_data["is_input_meaning"].lower()

            inspected = await client.call_tool(
                "inspect_dataset",
                {
                    "dataset_dir": str(dataset_dir),
                    "groups": ["IMG", "SEG"],
                    "extension": "mha",
                    "max_cases_per_group": 1,
                },
            )
            inspected_data = inspected.structured_content
            assert inspected_data["statistics"]["IMG"]["total_cases"] == 2
            assert inspected_data["statistics"]["IMG"]["sampled_cases"] == 1
            assert "design_config_strategy" in inspected_data["next_actions"]

            aliased = await client.call_tool(
                "prepare_dataset_aliases",
                {
                    "dataset_dir": str(dataset_dir),
                    "rename_map": {"IMG": "CT"},
                    "mode": "copy",
                },
            )
            aliased_data = aliased.structured_content
            assert aliased_data["created_count"] == 2
            assert all((dataset_dir / f"CASE_{idx:03d}" / "CT.mha").exists() for idx in range(2))

            ct_stats = await client.call_tool(
                "inspect_dataset",
                {
                    "dataset_dir": str(dataset_dir),
                    "groups": ["CT"],
                    "extension": "mha",
                    "max_cases_per_group": 1,
                },
            )
            ct_stats_data = ct_stats.structured_content["statistics"]["CT"]
            assert ct_stats_data["group"] == "CT"
            assert ct_stats_data["sampled_cases"] == 1

            strategy = await client.call_tool(
                "design_config_strategy",
                {
                    "dataset_dir": str(dataset_dir),
                    "task": "segmentation",
                    "group_roles": {"IMG": "input", "SEG": "target"},
                    "workflows": ["train", "prediction", "evaluation"],
                    "modeling_intent": "2d",
                    "example": "Segmentation",
                },
            )
            strategy_data = strategy.structured_content
            assert strategy_data["ok"] is True
            assert strategy_data["task"] == "segmentation"
            assert strategy_data["group_roles"]["input"] == ["IMG"]
            assert strategy_data["group_roles"]["target"] == ["SEG"]
            assert strategy_data["selected_example"]["name"] == "Segmentation"
            assert strategy_data["guidance_resources"]["overview"] == "guide://config-design"
            assert strategy_data["unresolved_questions"] == []

            created = await client.call_tool(
                "initialize_session",
                {
                    "from_example": "Segmentation",
                    "workflows": ["train", "prediction", "evaluation"],
                    "overwrite": True,
                },
            )
            created_data = created.structured_content
            assert created_data["seeded_from_example"] == "Segmentation"
            assert created_data["session"] == "default"
            assert created_data["workflows"] == ["train", "prediction", "evaluation"]
            assert "write_workflow_config" in created_data["next_actions"]

            session_dir = Path(created_data["path"])
            assert (session_dir / "Config.yml").exists()
            assert (session_dir / "Prediction.yml").exists()

    asyncio.run(scenario())


def test_mcp_server_review_config_semantics_surfaces_reasoning_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    workspace_root = tmp_path / "workspaces"
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(workspace_root))

    mcp_server = load_mcp_server()
    client_cls = fastmcp.Client
    model_source = (Path(__file__).resolve().parents[2] / "examples" / "Synthesis" / "Model.py").read_text(
        encoding="utf-8"
    )
    review_config = """
Trainer:
  Model:
    classpath: Model:UNetpp5
    UNetpp5:
      outputs_criterions: None
      Patch: None
      dim: 2
  Dataset:
    groups_src:
      MR:
        groups_dest:
          MR:
            transforms: None
            patch_transforms: None
            is_input: true
    Patch:
      patch_size:
      - 1
      - 256
      - 256
      overlap: None
      mask: None
      pad_value: 0
      extend_slice: 0
    dataset_filenames:
    - ./Dataset:a:mha
    augmentations: None
    subset: None
    filter: None
    use_cache: false
    batch_size: 1
  train_name: REVIEW
""".strip()

    async def scenario() -> None:
        async with client_cls(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {"overwrite": True})
            await client.call_tool(
                "write_session_file",
                {
                    "relative_path": "Model.py",
                    "content": model_source,
                },
            )
            await client.call_tool(
                "write_workflow_config",
                {
                    "workflow": "train",
                    "content": review_config,
                },
            )

            review = await client.call_tool(
                "review_config_semantics",
                {
                    "workflow": "train",
                },
            )
            review_data = review.structured_content
            warning_codes = {warning["code"] for warning in review_data["warnings"]}
            assert review_data["strategy_hint"] == "2d"
            assert "input_channel_context_mismatch" in warning_codes
            assert "no_non_input_groups_declared" in warning_codes
            assert review_data["summary"]["model"]["local_metadata"]["detected_contract"]["in_channels"] == 5
            assert "validate_config_semantics" in review_data["next_actions"]

    asyncio.run(scenario())


def test_mcp_server_inspect_object_signature_supports_local_custom_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    workspace_root = tmp_path / "workspaces"
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(workspace_root))

    mcp_server = load_mcp_server()
    client_cls = fastmcp.Client
    loss_source = """
class DiceFocalLoss:
    \"\"\"Hybrid overlap and hard-example loss.\"\"\"

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, smooth: float = 1e-5):
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
""".strip()

    async def scenario() -> None:
        async with client_cls(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {"overwrite": True})
            await client.call_tool(
                "write_session_file",
                {
                    "relative_path": "Loss.py",
                    "content": loss_source,
                },
            )
            inspected = await client.call_tool(
                "inspect_object_signature",
                {
                    "classpath": "Loss:DiceFocalLoss",
                },
            )
            inspected_data = inspected.structured_content
            assert inspected_data["ok"] is True
            assert inspected_data["source"] == "local"
            assert inspected_data["doc_summary"] == "Hybrid overlap and hard-example loss."
            assert inspected_data["signature"] == "DiceFocalLoss(alpha=0.25, gamma=2.0, smooth=1e-05)"
            assert inspected_data["defaults"]["gamma"] == 2.0
            assert any(parameter["name"] == "alpha" for parameter in inspected_data["parameters"])

    asyncio.run(scenario())


def test_mcp_server_leaderboard_ranks_runs_by_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    workspace_root = tmp_path / "workspaces"
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(workspace_root))

    mcp_server = load_mcp_server()
    client_cls = fastmcp.Client

    async def scenario() -> None:
        async with client_cls(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {"overwrite": True})
            metrics = {
                "run_low": _metric_json(0.61, "PRED:SEG:Dice"),
                "run_high": _metric_json(0.82, "PRED:SEG:Dice"),
                "run_mid": _metric_json(0.73, "PRED:SEG:Dice"),
            }
            for run_name, content in metrics.items():
                await client.call_tool(
                    "write_session_file",
                    {
                        "relative_path": f"Evaluations/{run_name}/Metric_TRAIN.json",
                        "content": content,
                    },
                )

            ranked = await client.call_tool(
                "leaderboard",
                {
                    "metric": "Dice",
                    "split": "TRAIN",
                },
            )
            ranked_data = ranked.structured_content
            assert ranked_data["selected_metric"] == "PRED:SEG:Dice"
            assert ranked_data["best"]["run_name"] == "run_high"
            assert [row["run_name"] for row in ranked_data["leaderboard"]] == ["run_high", "run_mid", "run_low"]

    asyncio.run(scenario())


def test_mcp_server_design_config_strategy_accepts_multiple_datasets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    workspace_root = tmp_path / "workspaces"
    dataset_a = tmp_path / "dataset_a"
    dataset_b = tmp_path / "dataset_b"
    _create_alias_dataset(dataset_a)
    _create_alias_dataset(dataset_b)
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(workspace_root))

    mcp_server = load_mcp_server()
    client_cls = fastmcp.Client

    async def scenario() -> None:
        async with client_cls(mcp_server.mcp) as client:
            strategy = await client.call_tool(
                "design_config_strategy",
                {
                    "task": "segmentation",
                    "dataset_dirs": [str(dataset_a), str(dataset_b)],
                    "group_roles": {"IMG": "input", "SEG": "target"},
                    "workflows": ["train", "evaluation"],
                    "modeling_intent": "2d",
                    "example": "Segmentation",
                },
            )
            strategy_data = strategy.structured_content
            assert strategy_data["dataset_dir"] is None
            assert strategy_data["dataset_summary"]["count"] == 2
            assert len(strategy_data["config_plan"]["dataset_entries"]) == 2
            assert any("provided datasets" in question for question in strategy_data["unresolved_questions"])

    asyncio.run(scenario())


def test_mcp_server_browse_dataset_surfaces_nested_candidate_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    workspace_root = tmp_path / "workspaces"
    dataset_root = tmp_path / "dataset"
    nested_root = dataset_root / "AB"
    nested_root.mkdir(parents=True, exist_ok=True)
    image = np.ones((2, 6, 6), dtype=np.float32)
    mask = np.ones((2, 6, 6), dtype=np.uint8)
    for index in range(2):
        case_dir = nested_root / f"CASE_{index:03d}"
        case_dir.mkdir()
        _write_image(case_dir / "MR.nii.gz", image, sitk.sitkFloat32)
        _write_image(case_dir / "CT.nii.gz", image, sitk.sitkFloat32)
        _write_image(case_dir / "MASK.nii.gz", mask, sitk.sitkUInt8)
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(workspace_root))

    mcp_server = load_mcp_server()
    client_cls = fastmcp.Client

    async def scenario() -> None:
        async with client_cls(mcp_server.mcp) as client:
            browsed = await client.call_tool(
                "browse_dataset",
                {
                    "dataset_dir": str(dataset_root),
                    "depth": 3,
                    "max_entries": 50,
                },
            )
            browsed_data = browsed.structured_content
            assert browsed_data["requested_path"] == str(dataset_root)
            assert browsed_data["root"] == str(nested_root)
            assert browsed_data["root_inferred"] is True
            assert browsed_data["case_count"] == 2
            assert browsed_data["common_groups"] == ["CT", "MASK", "MR"]
            assert browsed_data["candidate_dataset_roots"][0]["relative_path"] == "AB"

            inferred = await client.call_tool(
                "inspect_dataset", {"dataset_dir": str(dataset_root), "include_stats": False}
            )
            inferred_data = inferred.structured_content
            assert inferred_data["groups"] == {}
            assert "browse_dataset" in inferred_data["next_actions"]
            assert inferred_data["candidate_dataset_roots"][0]["relative_path"] == "AB"

    asyncio.run(scenario())


def test_mcp_server_inspect_dataset_recognizes_a_bare_zarr_store_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    """A single OME-Zarr store handed in as the root is one image, not a case tree.

    Walking it as cases would report the multiscale levels ('scale0'...) as cases and hide the
    store; the payload must say layout=single_store and carry a warning explaining the expected
    '<root>/<case>/<group>.zarr' layout instead of a bogus dataset_entry.
    """
    workspace_root = tmp_path / "workspaces"
    store = tmp_path / "brain.ome.zarr"
    for level in range(3):
        (store / f"scale{level}").mkdir(parents=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(workspace_root))

    mcp_server = load_mcp_server()
    client_cls = fastmcp.Client

    async def scenario() -> None:
        async with client_cls(mcp_server.mcp) as client:
            inferred = await client.call_tool("inspect_dataset", {"dataset_dir": str(store), "include_stats": False})
            data = inferred.structured_content
            assert data["layout"] == "single_store"
            assert data["dataset_entry"] is None
            # The warning must reach the tool payload (not just the internal scan).
            assert any("single OME-Zarr store" in warning for warning in data["warnings"])
            # The store itself is the only entry; its levels are not cases.
            assert data["total_cases"] == 1

    asyncio.run(scenario())


def test_mcp_server_read_dataset_file_previews_text_and_refuses_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    """Bounded sidecar reader: structured CSV preview, truncation flag, binary refusal."""
    workspace_root = tmp_path / "workspaces"
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(workspace_root))
    labels = tmp_path / "labels.csv"
    labels.write_text("case,grade\nCASE_000,2\nCASE_001,3\n", encoding="utf-8")
    long_txt = tmp_path / "cases.txt"
    long_txt.write_text("\n".join(f"CASE_{i:03d}" for i in range(500)), encoding="utf-8")
    binary = tmp_path / "weights.bin"
    binary.write_bytes(b"\x00\x01\x02" * 64)

    mcp_server = load_mcp_server()
    client_cls = fastmcp.Client

    async def scenario() -> None:
        async with client_cls(mcp_server.mcp) as client:
            index = await mcp_server.read_tool_index()
            registered = set(index["tools"])

            csv_read = await client.call_tool("read_dataset_file", {"path": str(labels)})
            data = csv_read.structured_content
            assert data["kind"] == "delimited"
            assert data["columns"] == ["case", "grade"]
            assert data["rows"] == [["CASE_000", "2"], ["CASE_001", "3"]]
            assert data["truncated"] is False
            # The tool's own next_actions must be registered tool names (AGENTS.md anti-drift rule).
            assert set(data["next_actions"]) <= registered

            bounded = await client.call_tool("read_dataset_file", {"path": str(long_txt), "max_lines": 10})
            bounded_data = bounded.structured_content
            assert bounded_data["returned_lines"] == 10
            assert bounded_data["truncated"] is True

            with pytest.raises(Exception, match="binary"):
                await client.call_tool("read_dataset_file", {"path": str(binary)})

    asyncio.run(scenario())

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

from konfai_mcp.server_experiments import SessionService  # noqa: E402
from konfai_mcp.server_jobs import JobRegistry  # noqa: E402
from konfai_mcp.server_support import WorkspaceLayout  # noqa: E402


def _session_service(tmp_path: Path) -> SessionService:
    repo_root = Path(__file__).resolve().parents[2]
    layout = WorkspaceLayout(tmp_path)
    layout.ensure_session_workspace()
    return SessionService(
        repo_root=repo_root,
        examples_root=repo_root / "examples",
        workspace_layout=layout,
        job_registry=JobRegistry({"queued", "running"}, workspace_layout=layout),
        max_log_tail_lines=20,
        active_job_states={"queued", "running"},
        validation_levels={"instantiate", "setup"},
        workflows={"train", "prediction", "evaluation"},
    )


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


def _write_run_metrics(workspace: Path, run_name: str, metric_name: str, value: float, split: str = "TRAIN") -> None:
    metrics = workspace / "Evaluations" / run_name / f"Metric_{split}.json"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(_metric_json(value, metric_name), encoding="utf-8")


@pytest.mark.usefixtures("workspace_root")
def test_directory_backed_dataset_entries_are_discovered(
    tmp_path: Path,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    dataset_dir = tmp_path / "dataset"
    for idx in range(2):
        case_dir = dataset_dir / f"CASE_{idx:03d}"
        zarr_store = case_dir / "CT.ome.zarr"
        zarr_store.mkdir(parents=True)
        (zarr_store / ".zattrs").write_text("{}", encoding="utf-8")
        dicom_series = case_dir / "MR"
        dicom_series.mkdir()
        (dicom_series / "slice0001.dcm").write_bytes(b"\x00")
        (case_dir / "SEG.mha").write_bytes(b"\x00")
        (case_dir / "notes").mkdir()
        (case_dir / "notes" / "readme.txt").write_text("x", encoding="utf-8")

    mcp_server = load_mcp_server()

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            inferred = await client.call_tool(
                "inspect_dataset", {"dataset_dir": str(dataset_dir), "include_stats": False}
            )
            data = inferred.structured_content
            assert set(data["groups"]) == {"CT", "MR", "SEG"}
            assert data["groups"]["CT"]["extensions"] == ["zarr"]
            assert data["groups"]["MR"]["extensions"] == ["dicom"]
            assert data["groups"]["CT"]["count"] == 2

    asyncio.run(scenario())


@pytest.mark.usefixtures("workspace_root")
def test_h5_internal_groups_are_discovered(
    tmp_path: Path,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    h5py = pytest.importorskip("h5py")

    dataset_dir = tmp_path / "dataset"
    for idx in range(2):
        case_dir = dataset_dir / f"CASE_{idx:03d}"
        case_dir.mkdir(parents=True)
        with h5py.File(case_dir / "data.h5", "w") as handle:
            handle.create_dataset("CT", data=np.zeros((2, 4, 4), dtype=np.float32))
            handle.create_dataset("SEG", data=np.zeros((2, 4, 4), dtype=np.uint8))

    mcp_server = load_mcp_server()

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            inferred = await client.call_tool(
                "inspect_dataset", {"dataset_dir": str(dataset_dir), "include_stats": False}
            )
            data = inferred.structured_content
            assert set(data["groups"]) == {"CT", "SEG"}
            assert data["groups"]["CT"]["extensions"] == ["h5"]

    asyncio.run(scenario())


@pytest.mark.usefixtures("workspace_root")
def test_mcp_server_dataset_inspection_and_aliasing(
    tmp_path: Path,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    dataset_dir = tmp_path / "dataset"
    _create_alias_dataset(dataset_dir)

    mcp_server = load_mcp_server()
    client_cls = fastmcp.Client

    async def scenario() -> None:
        async with client_cls(mcp_server.mcp) as client:
            inferred = await client.call_tool(
                "inspect_dataset", {"dataset_dir": str(dataset_dir), "include_stats": False}
            )
            inferred_data = inferred.structured_content
            assert {"IMG", "SEG"} == set(inferred_data["groups"])

            # Role is knowledge, not a guess: is_input is left null for every group and settled by
            # design_config_strategy with the user, instead of a name-based guess that mis-wires
            # segmentation (CT input) against synthesis (CT target). What it means is documented in
            # that tool's description, paid once, not restated in every payload.
            suggested = inferred_data["suggested_groups_src"]
            assert suggested["IMG"]["groups_dest"]["IMG"]["is_input"] is None
            assert suggested["SEG"]["groups_dest"]["SEG"]["is_input"] is None

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
            assert inspected_data["groups"]["IMG"]["count"] == 2
            assert inspected_data["groups"]["IMG"]["sampled_cases"] == 1
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
            ct_stats_data = ct_stats.structured_content["groups"]["CT"]
            assert ct_stats_data["sampled_cases"] == 1
            assert ct_stats_data["statistics"]["mean"]["count"] == 1

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


@pytest.mark.usefixtures("workspace_root")
def test_mcp_server_prepare_dataset_aliases_rejects_path_traversal(
    tmp_path: Path,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    # A rename_map target that climbs out of the case directory is an arbitrary-write primitive; the
    # tool must refuse it (writes never widen beyond the dataset) while a legit rename still works.
    dataset_dir = tmp_path / "dataset"
    _create_alias_dataset(dataset_dir)
    outside = tmp_path / "outside"
    outside.mkdir()

    mcp_server = load_mcp_server()
    client_cls = fastmcp.Client

    async def scenario() -> None:
        async with client_cls(mcp_server.mcp) as client:
            with pytest.raises(Exception, match=r"[Ii]nvalid target group|bare filename"):
                await client.call_tool(
                    "prepare_dataset_aliases",
                    {
                        "dataset_dir": str(dataset_dir),
                        "rename_map": {"IMG": "../../outside/PWNED"},
                        "mode": "copy",
                    },
                )
            assert not (outside / "PWNED.mha").exists()

            ok = await client.call_tool(
                "prepare_dataset_aliases",
                {"dataset_dir": str(dataset_dir), "rename_map": {"IMG": "CT"}, "mode": "copy"},
            )
            assert ok.structured_content["created_count"] == 2

    asyncio.run(scenario())


@pytest.mark.usefixtures("workspace_root")
def test_mcp_server_review_config_semantics_surfaces_reasoning_warnings(
    load_mcp_server: Callable[[], ModuleType],
) -> None:
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


@pytest.mark.usefixtures("workspace_root")
def test_mcp_server_inspect_object_signature_supports_local_custom_components(
    load_mcp_server: Callable[[], ModuleType],
) -> None:
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


@pytest.mark.usefixtures("workspace_root")
def test_mcp_server_inspect_object_signature_isolates_library_import(
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    """An installed-library classpath is imported in the spawn subprocess, not in the server process.

    AGENTS.md invariant: code that imports/executes runs isolated. A local File:Class is parsed
    statically and must stay in-process (no multi-second spawn on the common path).
    """
    mcp_server = load_mcp_server()

    subprocess_calls: list[tuple[str, dict[str, object]]] = []
    real_summarize = mcp_server.summarize_classpath_signature

    def spy_run_api_in_subprocess(target: str, kwargs: dict[str, object]) -> dict[str, object]:
        subprocess_calls.append((target, kwargs))
        # Delegate statically so the tool still returns a valid payload; the point is to record the
        # routing decision, not to spawn a real interpreter in the test.
        return real_summarize(**kwargs)

    monkeypatch.setattr(mcp_server, "_run_api_in_subprocess", spy_run_api_in_subprocess)

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {"overwrite": True})
            await client.call_tool(
                "write_session_file",
                {"relative_path": "Loss.py", "content": "class MyLoss:\n    def __init__(self, a: float = 1.0): ...\n"},
            )

            # Local File:Class is parsed statically, it must NOT be routed to the subprocess.
            local = await client.call_tool("inspect_object_signature", {"classpath": "Loss:MyLoss"})
            assert local.structured_content["source"] == "local"
            assert subprocess_calls == []

            # Installed-library classpath: its import MUST be isolated in the subprocess.
            imported = await client.call_tool("inspect_object_signature", {"classpath": "json.decoder.JSONDecoder"})
            assert imported.structured_content["source"] == "imported"
            assert imported.structured_content["ok"] is True
            assert len(subprocess_calls) == 1
            target, kwargs = subprocess_calls[0]
            assert target == "konfai_mcp.server_support:summarize_classpath_signature"
            assert kwargs["classpath"] == "json.decoder.JSONDecoder"

            # The colon form 'pkg:mod:Class' resolves to a dotted module ('json.decoder') and imports too --
            # its first token has no dot, so it must NOT be mistaken for a local File:Class and stay in-process.
            colon_form = await client.call_tool("inspect_object_signature", {"classpath": "json:decoder:JSONDecoder"})
            assert colon_form.structured_content["source"] == "imported"
            assert len(subprocess_calls) == 2
            assert subprocess_calls[1][1]["classpath"] == "json:decoder:JSONDecoder"

    asyncio.run(scenario())


@pytest.mark.usefixtures("workspace_root")
def test_mcp_server_leaderboard_ranks_runs_by_metric(
    load_mcp_server: Callable[[], ModuleType],
) -> None:
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


@pytest.mark.usefixtures("workspace_root")
def test_metric_direction_and_leaderboard_controls(
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    mcp_server = load_mcp_server()

    # A criterion named DiceLoss must rank as minimize despite the 'dice' token.
    direction, source = mcp_server.SESSION._metric_direction("PRED:SEG:DiceLoss")
    assert direction == "min"
    assert source == "heuristic:min"
    assert mcp_server.SESSION._metric_direction("PRED:SEG:Dice")[0] == "max"

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {"overwrite": True})
            workspace = Path(mcp_server.WORKSPACE_LAYOUT.workspace_dir())
            _write_run_metrics(workspace, "RUN_A", "PRED:SEG:DiceLoss", 0.2)
            _write_run_metrics(workspace, "RUN_B", "PRED:SEG:DiceLoss", 0.5)

            board = await client.call_tool("leaderboard", {"metric": "DiceLoss"})
            board_data = board.structured_content
            assert board_data["best"]["run_name"] == "RUN_A"
            assert board_data["available_splits"] == ["TRAIN"]

            flipped = await client.call_tool("leaderboard", {"metric": "DiceLoss", "direction": "max"})
            assert flipped.structured_content["best"]["run_name"] == "RUN_B"

            run_metrics = await client.call_tool("get_run_metrics", {"run_name": "RUN_B"})
            run_data = run_metrics.structured_content
            assert run_data["metrics"]["case"]["PRED:SEG:DiceLoss"]["CASE_000"] == 0.5
            assert run_data["split"] == "TRAIN"

            with pytest.raises(Exception, match=r"Available runs: .*RUN_A"):
                await client.call_tool("get_run_metrics", {"run_name": "MISSING"})

            with pytest.raises(Exception, match="Available splits"):
                await client.call_tool("leaderboard", {"split": "TEST"})

    asyncio.run(scenario())


@pytest.mark.usefixtures("workspace_root")
def test_mcp_server_design_config_strategy_accepts_multiple_datasets(
    tmp_path: Path,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    dataset_a = tmp_path / "dataset_a"
    dataset_b = tmp_path / "dataset_b"
    _create_alias_dataset(dataset_a)
    _create_alias_dataset(dataset_b)

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


@pytest.mark.usefixtures("workspace_root")
def test_mcp_server_browse_dataset_surfaces_nested_candidate_root(
    tmp_path: Path,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
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


@pytest.mark.usefixtures("workspace_root")
def test_mcp_server_inspect_dataset_recognizes_a_bare_zarr_store_root(
    tmp_path: Path,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    """A single OME-Zarr store handed in as the root is one image, not a case tree.

    Walking it as cases would report the multiscale levels ('scale0'...) as cases and hide the
    store; the payload must say layout=single_store and carry a warning explaining the expected
    '<root>/<case>/<group>.zarr' layout instead of a bogus dataset_entry.
    """
    store = tmp_path / "brain.ome.zarr"
    for level in range(3):
        (store / f"scale{level}").mkdir(parents=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")

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


@pytest.mark.usefixtures("workspace_root")
def test_mcp_server_read_dataset_file_previews_text_and_refuses_binary(
    tmp_path: Path,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    """Bounded sidecar reader: structured CSV preview, truncation flag, binary refusal."""
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


@pytest.mark.usefixtures("workspace_root")
def test_mcp_server_read_dataset_file_summarises_an_itk_transform_h5(
    tmp_path: Path,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    """A registration's Transform.h5 returns a structured summary instead of a binary refusal.

    Seen live: an agent asked to verify a known 5-voxel shift had the number in Transform.h5 and no
    tool that could open it. Linear transforms expose their parameters; dense fields a displacement
    summary."""
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")

    transform = tmp_path / "Transform.h5"
    field = np.zeros((4 * 4 * 2, 3))
    field[:, 1] = 5.0
    with h5py.File(transform, "w") as handle:
        group = handle.create_group("TransformGroup/0")
        group.create_dataset("TransformType", data=[b"DisplacementFieldTransform_double_3_3"])
        group.create_dataset("TransformFixedParameters", data=np.array([4.0, 4.0, 2.0]))
        group.create_dataset("TransformParameters", data=field.reshape(-1))
        rigid = handle.create_group("TransformGroup/1")
        rigid.create_dataset("TransformType", data=[b"Euler3DTransform_double_3_3"])
        rigid.create_dataset("TransformParameters", data=np.array([0.0, 0.0, 0.0, 0.0, 5.0, 0.0]))
    plain = tmp_path / "features.h5"
    with h5py.File(plain, "w") as handle:
        handle.create_dataset("data", data=np.ones(8))

    mcp_server = load_mcp_server()

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            read = await client.call_tool("read_dataset_file", {"path": str(transform)})
            data = read.structured_content
            assert data["kind"] == "itk_transform"
            dense, linear = data["transforms"]
            assert dense["type"] == "DisplacementFieldTransform_double_3_3"
            assert dense["displacement_summary"]["mean_xyz"] == [0.0, 5.0, 0.0]
            assert dense["displacement_summary"]["max_magnitude"] == 5.0
            assert linear["parameters"] == [0.0, 0.0, 0.0, 0.0, 5.0, 0.0]

            # An HDF5 that is NOT an ITK transform keeps the binary refusal.
            with pytest.raises(Exception, match="binary"):
                await client.call_tool("read_dataset_file", {"path": str(plain)})

    asyncio.run(scenario())


def test_a_dense_field_summary_is_computed_in_bounded_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The summary of a displacement field never loads it whole: sliced accumulation must equal
    the single-pass numbers, or a bounded read would be a wrong read."""
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    from konfai_mcp import file_io

    rng = np.random.default_rng(7)
    field = rng.normal(size=(50, 3))
    transform = tmp_path / "Transform.h5"
    with h5py.File(transform, "w") as handle:
        group = handle.create_group("TransformGroup/0")
        group.create_dataset("TransformType", data=[b"DisplacementFieldTransform_double_3_3"])
        group.create_dataset("TransformParameters", data=field.reshape(-1))

    monkeypatch.setattr(file_io, "_DENSE_CHUNK_ELEMENTS", 9)  # 3 vectors per slice: many chunks
    summary = file_io._itk_transform_summary(transform)["transforms"][0]["displacement_summary"]

    magnitudes = np.linalg.norm(field, axis=1)
    assert summary["vectors"] == 50
    assert summary["mean_xyz"] == [round(float(v), 4) for v in field.mean(axis=0)]
    assert summary["std_xyz"] == [round(float(v), 4) for v in field.std(axis=0)]
    assert summary["mean_magnitude"] == round(float(magnitudes.mean()), 4)
    assert summary["max_magnitude"] == round(float(magnitudes.max()), 4)


def test_design_config_strategy_uses_per_root_extension(tmp_path: Path) -> None:
    mha_case = tmp_path / "MhaDataset" / "case_001"
    nii_case = tmp_path / "NiiDataset" / "case_001"
    mha_case.mkdir(parents=True)
    nii_case.mkdir(parents=True)
    (mha_case / "MR.mha").write_text("", encoding="utf-8")
    (nii_case / "MR.nii.gz").write_text("", encoding="utf-8")

    payload = _session_service(tmp_path).design_config_strategy_payload(
        dataset_dir=None,
        dataset_dirs=[mha_case.parent, nii_case.parent],
        task="synthesis",
    )
    entries = payload["config_plan"]["dataset_entries"]
    by_path = {entry["path"]: entry["entry"] for entry in entries}
    assert by_path[str(mha_case.parent)].endswith(":a:mha")
    assert by_path[str(nii_case.parent)].endswith(":a:nii.gz")


def test_browse_dataset_depth_is_inclusive_and_bounded(tmp_path: Path) -> None:
    dataset = tmp_path / "root"
    (dataset / "A" / "B" / "C").mkdir(parents=True)
    service = _session_service(tmp_path)

    depth1 = service.browse_dataset_payload(dataset, depth=1)
    assert depth1["entries"], "depth=1 must still list the immediate children"
    assert max(entry["depth"] for entry in depth1["entries"]) == 1
    assert {entry["path"] for entry in depth1["entries"]} == {"A"}

    depth2 = service.browse_dataset_payload(dataset, depth=2)
    paths = {entry["path"] for entry in depth2["entries"]}
    assert max(entry["depth"] for entry in depth2["entries"]) == 2
    assert "A/B" in paths
    assert "A/B/C" not in paths

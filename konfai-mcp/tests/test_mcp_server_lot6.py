# SPDX-License-Identifier: Apache-2.0
"""Lot 6: sequential batch runs, cross-validation folds, and volume previews."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

fastmcp = pytest.importorskip("fastmcp")

from mcp_test_helpers import install_fake_konfai_runtime, yaml_dump  # noqa: E402


def test_generate_folds_and_run_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    dataset_dir = tmp_path / "dataset"
    for index in range(4):
        case_dir = dataset_dir / f"CASE_{index:03d}"
        case_dir.mkdir(parents=True)
        (case_dir / "CT.mha").write_bytes(b"\x00")

    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    monkeypatch.setenv("KONFAI_MCP_FAKE_SLEEP_S", "0.05")
    mcp_server = load_mcp_server()
    install_fake_konfai_runtime(tmp_path, monkeypatch, mcp_server)

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {"overwrite": True})

            folds = await client.call_tool("generate_folds", {"dataset_dir": str(dataset_dir), "k": 2, "seed": 1})
            folds_data = folds.structured_content
            assert folds_data["k"] == 2 and folds_data["total_cases"] == 4
            members = [case for fold in folds_data["folds"].values() for case in fold["cases"]]
            assert sorted(members) == [f"CASE_{i:03d}" for i in range(4)]
            fold_0 = folds_data["folds"]["fold_0"]
            assert fold_0["train_subset"] == "~folds/fold_0.txt"
            assert Path(fold_0["file"]).read_text(encoding="utf-8").strip().splitlines() == fold_0["cases"]

            with pytest.raises(Exception, match="cannot make k="):
                await client.call_tool("generate_folds", {"dataset_dir": str(dataset_dir), "k": 5})

            # A sequential two-config batch (fake runtime): both runs complete in order.
            await client.call_tool(
                "write_workflow_config",
                {"workflow": "train", "content": yaml_dump({"Trainer": {"train_name": "SWEEP_BASE"}})},
            )
            for name in ("FOLD_A", "FOLD_B"):
                await client.call_tool(
                    "write_session_file",
                    {"relative_path": f"Config_{name}.yml", "content": yaml_dump({"Trainer": {"train_name": name}})},
                )
            batch = await client.call_tool("run_batch", {"config_files": ["Config_FOLD_A.yml", "Config_FOLD_B.yml"]})
            batch_data = batch.structured_content
            assert batch_data["requested"] == 2 and batch_data["completed"] == 2
            assert [result["run_name"] for result in batch_data["results"]] == ["FOLD_A", "FOLD_B"]
            assert all(result["status"] == "done" for result in batch_data["results"])

            missing = await client.call_tool(
                "run_batch", {"config_files": ["Config_MISSING.yml"], "stop_on_error": True}
            )
            missing_data = missing.structured_content
            assert missing_data["completed"] == 0
            assert missing_data["results"][0]["status"] == "launch_error"

    asyncio.run(scenario())


def test_preview_volume_returns_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    pytest.importorskip("SimpleITK")
    from mcp_test_helpers import create_segmentation_dataset

    dataset_dir = tmp_path / "dataset"
    create_segmentation_dataset(dataset_dir)
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    mcp_server = load_mcp_server()

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            volume = dataset_dir / "CASE_000" / "CT.mha"
            preview = await client.call_tool("preview_volume", {"path": str(volume)})
            image_blocks = [block for block in preview.content if getattr(block, "type", "") == "image"]
            assert image_blocks, preview.content
            assert image_blocks[0].mimeType == "image/png"
            assert len(image_blocks[0].data) > 100

            with pytest.raises(Exception, match="Volume file not found"):
                await client.call_tool("preview_volume", {"path": str(dataset_dir / "nope.mha")})

    asyncio.run(scenario())


def test_preview_volume_streams_a_single_plane_without_full_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    """A 3-D scalar volume must be previewed by streaming one plane, never a full sitk.ReadImage.

    Guards the mandatory lazy invariant (AGENTS.md §7): the whole volume is never materialised in RAM.
    Also pins byte-identity -- the streamed plane must equal the full-read np.take plane on every axis.
    """
    sitk = pytest.importorskip("SimpleITK")
    import base64

    import numpy as np

    # Asymmetric shape (each axis a different length) so a wrong axis mapping changes the plane shape and
    # is caught immediately; a ramp gives each plane distinct, non-constant values.
    z, y, x = 4, 5, 6
    zz, yy, xx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing="ij")
    array = (zz * 100 + yy * 10 + xx).astype(np.float32)  # [Z, Y, X]
    volume = tmp_path / "ramp.mha"
    sitk.WriteImage(sitk.GetImageFromArray(array), str(volume))

    def window(plane: "np.ndarray") -> "np.ndarray":
        plane = plane.astype(np.float32)
        low, high = np.percentile(plane, (1.0, 99.0))
        if high <= low:
            high = low + 1.0
        plane = np.clip((plane - low) / (high - low) * 255.0, 0.0, 255.0).astype(np.uint8)
        stride = max(1, -(-max(plane.shape) // max(512, 16)))
        return plane[::stride, ::stride]

    real_read_image = sitk.ReadImage
    full = sitk.GetArrayFromImage(real_read_image(str(volume)))  # reference read, before patching
    expected = {axis: window(np.take(full, 1, axis=axis)) for axis in (0, 1, 2)}

    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    mcp_server = load_mcp_server()

    full_reads: list[str] = []

    def spy_read_image(*args, **kwargs):
        full_reads.append(str(args[0]) if args else "")
        return real_read_image(*args, **kwargs)

    monkeypatch.setattr(sitk, "ReadImage", spy_read_image)

    def decode(image_block) -> "np.ndarray":
        png = tmp_path / "decoded.png"
        png.write_bytes(base64.b64decode(image_block.data))
        return sitk.GetArrayFromImage(real_read_image(str(png)))

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            for axis in (0, 1, 2):
                preview = await client.call_tool(
                    "preview_volume", {"path": str(volume), "axis": axis, "slice_index": 1}
                )
                block = next(b for b in preview.content if getattr(b, "type", "") == "image")
                assert np.array_equal(decode(block), expected[axis]), f"axis {axis} plane differs from full read"

    asyncio.run(scenario())

    # The streaming path uses ImageFileReader.Execute(), never sitk.ReadImage on the whole volume.
    assert full_reads == [], f"preview_volume full-read the volume instead of streaming: {full_reads}"


def test_set_parameters_preserve_value_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    """export_app must encode --set values so their type survives the YAML re-parse in konfai_apps.

    A string "true"/"1" would otherwise come back as a bool/int; json.dumps keeps it a string while
    leaving genuine ints/bools/floats/lists untouched.
    """
    yaml_safe = pytest.importorskip("ruamel.yaml")

    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    mcp_server = load_mcp_server()

    captured: dict[str, list[str] | None] = {}

    def fake_export_app(ref, path, *, display_name=None, config_overrides=None, force_update=False):
        captured["config_overrides"] = config_overrides
        return {"ref": ref, "exported_to": str(path)}

    monkeypatch.setattr(mcp_server.APP_SERVICE, "export_app", fake_export_app)

    set_parameters = {"mode": "true", "code": "1", "iterations": 300, "flag": True, "lr": 2.0}

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            await client.call_tool(
                "export_app",
                {"ref": "org/App", "path": str(tmp_path / "out"), "set_parameters": set_parameters},
            )

    asyncio.run(scenario())

    overrides = captured["config_overrides"]
    assert overrides is not None
    parser = yaml_safe.YAML(typ="safe")
    round_tripped = {name: parser.load(value) for name, _, value in (item.partition("=") for item in overrides)}
    assert round_tripped == set_parameters
    assert isinstance(round_tripped["mode"], str) and isinstance(round_tripped["code"], str)

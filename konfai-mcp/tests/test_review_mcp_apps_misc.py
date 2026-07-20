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

"""ONNX export isolation, Windows-safe bundle rewrite, surfacing the app.json optional-input
``default``, paired-group case-count validation, per-root dataset extensions, browse depth, namespace
distribution resolution, and transport env validation.
"""

import importlib.metadata as _metadata
import json
import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import konfai_mcp  # noqa: E402
from konfai_mcp.extensions import _select_distribution, check_external_dependency  # noqa: E402
from konfai_mcp.server_apps import AppService  # noqa: E402
from konfai_mcp.server_experiments import SessionService  # noqa: E402
from konfai_mcp.server_jobs import JobRegistry  # noqa: E402
from konfai_mcp.server_support import WorkspaceLayout  # noqa: E402


def _app_service(tmp_path: Path) -> AppService:
    return AppService(workspace_layout=WorkspaceLayout(tmp_path / "workspaces"))


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


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return str(path)


# -- Finding 3: describe_app surfaces the optional-input "default" -------------------------------


def test_describe_app_surfaces_optional_input_default(tmp_path: Path) -> None:
    app_dir = tmp_path / "with_default"
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Masked Reg",
                "description": "Registration app with an optional mask defaulting to ones",
                "short_description": "masked reg",
                "task": "registration",
                "tta": 0,
                "mc_dropout": 0,
                "models": ["tiny.pt"],
                "inputs": {
                    "Fixed": {"display_name": "Fixed", "volume_type": "VOLUME", "required": True},
                    "Mask": {
                        "display_name": "Mask",
                        "volume_type": "SEGMENTATION",
                        "required": False,
                        "default": "ones",
                    },
                },
                "outputs": {"Moved": {"display_name": "Moved", "volume_type": "VOLUME", "required": True}},
            }
        ),
        encoding="utf-8",
    )
    (app_dir / "tiny.pt").write_bytes(b"")

    payload = _app_service(tmp_path).describe_app(str(app_dir))

    assert payload["inputs"]["Mask"]["required"] is False
    assert payload["inputs"]["Mask"]["default"] == "ones"
    # A required input carries no synthesised default, so the key stays absent.
    assert "default" not in payload["inputs"]["Fixed"]


# -- Finding 2: the bundle rewrite keeps a Windows drive path intact ----------------------------


def test_normalize_bundled_prediction_config_preserves_windows_drive_path(tmp_path: Path) -> None:
    from konfai_mcp.server_support import YAML_SAFE, yaml_dump_content

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    # A naive split(":", 1) would read the drive letter 'C' as the path and '\\...:a:mha' as the token.
    config = {"Predictor": {"Dataset": {"dataset_filenames": ["C:\\data\\Dataset:a:mha"]}}}
    (bundle / "Prediction.yml").write_text(yaml_dump_content(config), encoding="utf-8")

    _app_service(tmp_path)._normalize_bundled_prediction_configs(bundle, ["Prediction.yml"])

    data = YAML_SAFE.load((bundle / "Prediction.yml").read_text(encoding="utf-8"))
    # The staged root replaces the path; the ':a:mha' accessor/format token survives uncorrupted.
    assert data["Predictor"]["Dataset"]["dataset_filenames"] == ["./Dataset/:a:mha"]


# -- Finding 4: paired input/gt/mask groups with unequal case counts are rejected ---------------


def test_check_case_pairing_rejects_mismatched_file_counts(tmp_path: Path) -> None:
    inputs = [[_touch(tmp_path / "in" / f"a{i}.mha") for i in range(3)]]
    gt = [[_touch(tmp_path / "gt" / f"g{i}.mha") for i in range(2)]]
    with pytest.raises(ValueError, match="mismatched case counts"):
        AppService._check_case_pairing([("inputs", inputs), ("gt", gt)])

    equal_gt = [[_touch(tmp_path / "gt2" / f"g{i}.mha") for i in range(3)]]
    AppService._check_case_pairing([("inputs", inputs), ("gt", equal_gt)])  # no raise


def test_check_case_pairing_skips_directory_groups(tmp_path: Path) -> None:
    # A directory expands to an unknown case count downstream, so it is not counted here: a 1-dir
    # group beside a 3-file group must NOT falsely trip the guard.
    (tmp_path / "series").mkdir()
    inputs = [[str(tmp_path / "series")]]
    gt = [[_touch(tmp_path / "gt" / f"g{i}.mha") for i in range(3)]]
    AppService._check_case_pairing([("inputs", inputs), ("gt", gt)])  # no raise


def test_prepare_infer_rejects_unequal_channel_counts(tmp_path: Path) -> None:
    a = _touch(tmp_path / "c" / "a.mha")
    b = _touch(tmp_path / "c" / "b.mha")
    c = _touch(tmp_path / "c" / "c.mha")
    with pytest.raises(ValueError, match="mismatched case counts"):
        # Remote ref keeps the trust gate out of the way; the pairing check runs first regardless.
        _app_service(tmp_path).prepare_infer(ref="localhost:8000:MyApp", inputs=[[a, b], [c]])


# -- Finding 5: design_config_strategy resolves the read extension per root ----------------------


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


# -- Finding 6: browse depth is an inclusive cap on entry depth ----------------------------------


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


# -- Finding 7: namespace-package distribution resolution ---------------------------------------


def test_select_distribution_prefers_name_match_and_handles_single() -> None:
    assert _select_distribution("Foo_Bar", ["baz", "foo-bar"]) == "foo-bar"
    assert _select_distribution("solo", ["only-dist"]) == "only-dist"
    assert _select_distribution("missing", []) == "missing"


def test_check_external_dependency_resolves_namespace_by_owning_wheel() -> None:
    candidates = _metadata.packages_distributions().get("itk", [])
    if len(candidates) <= 1 or "itk-core" not in candidates:
        pytest.skip("environment does not split the 'itk' namespace across multiple wheels")
    payload = check_external_dependency("itk")
    # The base 'itk' import is shipped by itk-core, NOT by whichever sibling wheel sorts first.
    assert payload["distribution"] == "itk-core"


# -- Finding 8: a bad KONFAI_MCP_TRANSPORT env value is rejected instead of passed through -------


def test_main_rejects_invalid_transport_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KONFAI_MCP_TRANSPORT", "not-a-transport")
    with pytest.raises(SystemExit):
        konfai_mcp.main([])


def test_parser_default_accepts_valid_transport_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KONFAI_MCP_TRANSPORT", "streamable-http")
    args = konfai_mcp._build_parser().parse_args([])
    assert args.transport == "streamable-http"


# -- Finding 1: ONNX export runs in the spawn subprocess, never in the server process -----------


def test_package_from_session_runs_onnx_export_in_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = WorkspaceLayout(tmp_path / "workspaces")
    workspace = layout.workspace_dir()
    (workspace / "Checkpoints" / "run").mkdir(parents=True)
    (workspace / "Checkpoints" / "run" / "model.pt").write_bytes(b"")
    (workspace / "Prediction.yml").write_text("Predictor: {}\n", encoding="utf-8")

    import konfai_apps.bundle as bundle_module

    def _fail_in_process(*args: object, **kwargs: object) -> object:
        raise AssertionError("export_onnx_into_bundle must not run in the server process")

    monkeypatch.setattr(bundle_module, "export_onnx_into_bundle", _fail_in_process)

    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_subprocess(target: str, kwargs: dict[str, object], *rest: object) -> str:
        calls.append((target, kwargs))
        return kwargs["bundle"] + "/model.onnx"  # type: ignore[operator]

    # Patch by dotted path so it lands on the live konfai_mcp.runner module (some suites reload it),
    # which is the exact object package_from_session's `from . import runner` resolves at call time.
    monkeypatch.setattr("konfai_mcp.runner.run_api_in_subprocess", _fake_subprocess)

    result = AppService(workspace_layout=layout).package_from_session(
        name="OnnxBundle", display_name="Onnx App", description="exports onnx", onnx=True
    )

    assert len(calls) == 1
    target, kwargs = calls[0]
    assert target == "konfai_apps.bundle:export_onnx_into_bundle"
    assert kwargs["bundle"] == str(Path(result["bundle_path"]))
    assert kwargs["checkpoint"].endswith("model.pt")  # type: ignore[union-attr]
    assert result["onnx"].endswith("model.onnx")

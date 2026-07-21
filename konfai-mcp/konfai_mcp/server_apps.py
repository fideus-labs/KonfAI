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

"""App-facing services for the KonfAI MCP server.

The *use an app* half of the lifecycle, beside the train/predict/evaluate loop: discover published
KonfAI apps and read their ``app.json`` manifest so the agent can decide whether an existing app
already solves the user's task before authoring and training a config from scratch.

Discovery is layered over a referenced catalogue of app sources (the same ``{"apps": [...]}`` shape
the ``konfai-apps`` server consumes via ``--apps``): a shipped default, a per-workspace editable
file, and a ``KONFAI_MCP_APP_CATALOG`` env override, plus an ad-hoc per-call override.

Heavy ``konfai_apps`` imports happen lazily inside methods (like ``catalog.py`` does for ``konfai``),
so importing this module is cheap and does not hard-require the optional ``konfai-apps`` package.

This module is the SAFE tier: ``list_apps``/``describe_app`` read manifest metadata only. They never
import an app's model ``.py`` nor pip-install its requirements (only the parameter/inference tools do
that, and they live behind the explicit ``allow_untrusted_code`` gate -- see ``_require_trust``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .server_support import WorkspaceLayout

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent / "apps_catalog.json"


def _import_app_repository() -> Any:
    """Import ``konfai_apps.app_repository`` lazily, failing at point-of-use with an install hint."""
    try:
        from konfai_apps import app_repository
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "KonfAI MCP app tools require the 'konfai-apps' package. Install it with: pip install konfai-apps"
        ) from exc
    return app_repository


def _source_of(info: Any) -> str:
    """Map a resolved repository object to a coarse source label."""
    name = type(info).__name__
    if "FromHF" in name:
        return "hf"
    if "RemoteServer" in name:
        return "remote"
    return "local"


def _slots(entries: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Render a ``{key: DataEntry}`` mapping as a JSON-clean input/output descriptor.

    An optional input may declare a ``default`` in app.json (``"ones"``/``"zeros"``): the app's runtime
    synthesises that volume when the input is omitted. Surface it so the agent knows both that the input
    is optional (``required: false``) and how a missing one is filled.
    """
    slots: dict[str, dict[str, Any]] = {}
    for key, entry in entries.items():
        slot: dict[str, Any] = {
            "display_name": entry.display_name,
            "volume_type": entry.volume_type.value,
            "required": entry.required,
        }
        default = getattr(entry, "default", None)
        if default is not None:
            slot["default"] = default
        slots[key] = slot
    return slots


class AppService:
    """Discover KonfAI apps from a referenced catalogue and read their manifest metadata.

    Injected with the same ``WorkspaceLayout`` as ``SessionService`` so a per-workspace catalogue
    file lives alongside the session workspaces.
    """

    def __init__(self, workspace_layout: WorkspaceLayout, default_catalog_path: Path | None = None) -> None:
        self.workspace_layout = workspace_layout
        self.default_catalog_path = default_catalog_path or DEFAULT_CATALOG_PATH

    # -- catalogue resolution -------------------------------------------------------------------

    @staticmethod
    def _read_catalog_file(path: Path) -> list[str]:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid app catalog JSON at {path}: {exc}") from exc
        apps = data.get("apps") if isinstance(data, dict) else None
        if not isinstance(apps, list):
            raise ValueError(f"App catalog {path} must be a JSON object with an 'apps' list.")
        return [str(item) for item in apps]

    def _workspace_catalog_path(self) -> Path:
        return self.workspace_layout.apps_catalog_path()

    def resolve_catalog(self) -> tuple[list[str], dict[str, Any]]:
        """Merge the shipped default, the workspace file, and the env override (in that precedence).

        Later sources are appended; duplicates are dropped preserving first-seen order. Returns the
        merged reference list plus a per-source provenance map so the agent can see where each
        entry came from.
        """
        provenance: dict[str, Any] = {}
        merged: list[str] = []

        default_refs = self._read_catalog_file(self.default_catalog_path)
        provenance["default"] = {"path": str(self.default_catalog_path), "apps": default_refs}
        merged.extend(default_refs)

        workspace_path = self._workspace_catalog_path()
        if workspace_path.exists():
            workspace_refs = self._read_catalog_file(workspace_path)
            provenance["workspace"] = {"path": str(workspace_path), "apps": workspace_refs}
            merged.extend(workspace_refs)

        env_value = os.environ.get("KONFAI_MCP_APP_CATALOG")
        if env_value:
            env_path = Path(env_value).expanduser()
            env_refs = self._read_catalog_file(env_path) if env_path.exists() else []
            provenance["env"] = {"path": str(env_path), "exists": env_path.exists(), "apps": env_refs}
            merged.extend(env_refs)

        seen: set[str] = set()
        deduped: list[str] = []
        for ref in merged:
            if ref not in seen:
                seen.add(ref)
                deduped.append(ref)
        return deduped, provenance

    # -- reference classification ---------------------------------------------------------------

    @staticmethod
    def _classify(ref: str) -> str:
        """Classify a catalogue entry, mirroring ``get_app_repository_info`` dispatch.

        Returns one of: ``local`` (an existing path to an app folder), ``remote``
        (``host:port:name``), ``remote_server`` (``host:port[|token]`` to expand into its apps),
        ``hf_app`` (``repo_id:app_name``), ``hf_repo`` (a bare ``repo_id`` to expand into its apps),
        or ``unknown``.
        """
        if Path(ref).expanduser().exists():
            return "local"
        parts = ref.split(":")
        if len(parts) >= 3 and parts[1].isdigit():
            return "remote"
        # 'host:port[|token]' is a remote server to enumerate (not a single app).
        if len(parts) == 2 and parts[1].split("|")[0].isdigit():
            return "remote_server"
        # An HF app id is 'repo_id:app_name' where repo_id is 'org/name' -- require the '/', so a
        # bare 'host:port' is not mistaken for an HF app.
        if ":" in ref and "/" in ref.split(":", 1)[0]:
            return "hf_app"
        if ":" not in ref and "/" in ref:
            return "hf_repo"
        return "unknown"

    # -- tools ----------------------------------------------------------------------------------

    def list_apps(
        self,
        repos: list[str] | None = None,
        include_summary: bool = False,
        force_update: bool = False,
    ) -> dict[str, Any]:
        """Enumerate apps from the catalogue (or an ad-hoc ``repos`` override).

        Bare HuggingFace ``repo_id`` entries are expanded into their contained apps; single app ids,
        local paths, and remote entries are passed through. This is cheap by default (no manifest
        resolution); set ``include_summary`` to also fetch each app's display name / short
        description / modality (slower, one resolve per app, best-effort).
        """
        app_repository = _import_app_repository()

        if repos is not None:
            entries = [str(ref) for ref in repos]
            catalog: dict[str, Any] = {"override": entries}
        else:
            entries, catalog = self.resolve_catalog()

        apps: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for ref in entries:
            kind = self._classify(ref)
            if kind == "hf_repo":
                try:
                    names = app_repository.get_available_apps_on_hf_repo(ref, force_update)
                except Exception as exc:  # network / repo errors -> report, keep going
                    errors.append({"ref": ref, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                for name in names:
                    apps.append({"ref": f"{ref}:{name}", "source": "hf", "repo": ref, "app_name": name})
            elif kind == "remote_server":
                try:
                    base, _, token = ref.partition("|")
                    host, port_str = base.split(":")
                    from konfai import RemoteServer

                    server = RemoteServer(host, int(port_str), token or None)
                    names = app_repository.get_available_apps_on_remote_server(server)
                except Exception as exc:  # server unreachable / auth -> report, keep going
                    errors.append({"ref": ref, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                suffix = f"|{token}" if token else ""
                for name in names:
                    apps.append(
                        {
                            "ref": f"{host}:{port_str}:{name}{suffix}",
                            "source": "remote",
                            "repo": f"{host}:{port_str}",
                            "app_name": name,
                        }
                    )
            elif kind == "hf_app":
                repo, _, name = ref.partition(":")
                apps.append({"ref": ref, "source": "hf", "repo": repo, "app_name": name})
            elif kind == "local":
                apps.append({"ref": ref, "source": "local", "app_name": Path(ref).expanduser().name})
            elif kind == "remote":
                apps.append({"ref": ref, "source": "remote"})
            else:
                errors.append({"ref": ref, "error": "Unrecognized app reference format."})

        if include_summary:
            for app in apps:
                try:
                    summary = self.describe_app(app["ref"], force_update=False)
                except Exception as exc:  # resolution is best-effort here
                    app["summary_error"] = f"{type(exc).__name__}: {exc}"
                    continue
                app["display_name"] = summary["display_name"]
                app["short_description"] = summary["short_description"]
                app["inputs"] = list(summary["inputs"].keys())
                app["outputs"] = list(summary["outputs"].keys())

        return {
            "apps": apps,
            "count": len(apps),
            "catalog": catalog,
            "errors": errors,
            "next_actions": ["describe_app", "design_config_strategy"],
        }

    def describe_app(self, ref: str, force_update: bool = False) -> dict[str, Any]:
        """Resolve one app and return its ``app.json`` manifest as a JSON-clean payload.

        SAFE: reads manifest metadata only (display name, description, input/output modality,
        capabilities, checkpoints, terminology). It does not import the app's model code or install
        its requirements.
        """
        app_repository = _import_app_repository()
        info = app_repository.get_app_repository_info(ref, force_update)
        inference, evaluation, uncertainty = info.has_capabilities()
        finetunable = info.is_finetunable()

        # Route by what the app can actually do instead of dead-ending on describe/design. A runnable app
        # is imported into the session (import_app), then predicted / evaluated / fine-tuned through the
        # ordinary run_prediction / run_evaluation / run_resume tools -- so import_app is the single entry.
        next_actions: list[str] = []
        if inference:
            next_actions.extend(["import_app", "list_app_parameters", "export_app"])
        else:
            next_actions.extend(["describe_app", "design_config_strategy"])

        payload: dict[str, Any] = {
            "ref": ref,
            "source": _source_of(info),
            "name": info.get_name(),
            "display_name": info.get_display_name(),
            "short_description": info.get_short_description(),
            "description": info.get_description(),
            "inputs": _slots(info.get_inputs()),
            "outputs": _slots(info.get_outputs()),
            "capabilities": {
                "inference": inference,
                "evaluation": evaluation,
                "uncertainty": uncertainty,
            },
            "checkpoints": info.get_checkpoints_name(),
            "checkpoints_available": info.get_checkpoints_name_available(),
            "maximum_tta": info.get_maximum_tta(),
            "mc_dropout": info.get_mc_dropout(),
            "finetunable": finetunable,
            "next_actions": next_actions,
        }

        task = info.get_task()
        if task:
            payload["task"] = task

        terminology = info.get_terminology()
        if terminology:
            payload["terminology"] = {
                str(label): {"name": entry.name, "color": entry.color} for label, entry in terminology.items()
            }

        try:
            patch_size = info.get_patch_size()
        except Exception:  # best-effort; the LocalAppRepository override may hit the network
            patch_size = None
        if patch_size:
            payload["patch_size"] = list(patch_size)

        evaluations = info.get_evaluations_inputs()
        if evaluations:
            payload["evaluation_configs"] = [
                {
                    "display_name": key.display_name,
                    "evaluation_file": key.evaluation_file,
                    "inputs": _slots(slots),
                }
                for key, slots in evaluations.items()
            ]
        return payload

    def list_parameters(
        self, ref: str, allow_untrusted_code: bool = False, force_update: bool = False
    ) -> dict[str, Any]:
        """Read an app's tunable model parameters and their type-derived constraints.

        GATED: deriving constraints imports the app's model class (the trust boundary), so this
        requires ``allow_untrusted_code=True`` and the import runs in an isolated spawn subprocess,
        never in the server process. Returns the same ``{values, constraints}`` shape the
        ``--set`` / ``set_parameters`` overrides write into. Local/HuggingFace apps only.
        """
        if self._classify(ref) in ("remote", "remote_server"):
            raise ValueError("Reading tunable parameters is only supported for local or HuggingFace apps.")
        if not allow_untrusted_code:
            raise ValueError(
                "Reading an app's parameters imports its model code to derive constraints. Set "
                "allow_untrusted_code=True to confirm you trust this app source."
            )
        from . import runner as mcp_runner

        payload = mcp_runner.run_api_in_subprocess(
            "konfai_mcp.runner:app_parameters_api", {"ref": ref, "force_update": force_update}
        )
        return {
            "ref": ref,
            "source": payload.get("source", "local"),
            "values": payload.get("values", {}),
            "constraints": payload.get("constraints", {}),
            "next_actions": ["import_app", "export_app"],
        }

    def export_app(
        self,
        ref: str,
        path: str,
        display_name: str | None = None,
        config_overrides: list[str] | None = None,
        force_update: bool = False,
    ) -> dict[str, Any]:
        """Materialise a resolved app into a local, editable bundle (optionally baking tuned --set values).

        This is the inference-side reproducibility artifact: 'save this HuggingFace/remote-cached app,
        with my tuned parameters, as a local app'. It copies files and rewrites the config; it does not
        import the app's model code. Local/HuggingFace apps only.
        """
        if self._classify(ref) in ("remote", "remote_server"):
            raise ValueError("Exporting is only supported for local or HuggingFace apps (a remote server cannot).")
        app_repository = _import_app_repository()
        info = app_repository.get_app_repository_info(ref, force_update)
        if not isinstance(info, app_repository.LocalAppRepository):
            raise ValueError("Exporting is only supported for local or HuggingFace apps (a remote server cannot).")
        target = Path(path).expanduser().resolve()
        info.export_app(target, display_name=display_name, config_overrides=config_overrides)
        return {
            "ref": ref,
            "exported_to": str(target),
            "next_actions": ["describe_app", "import_app", "register_app_source"],
        }

    def import_app(
        self,
        ref: str,
        allow_untrusted_code: bool = False,
        display_name: str | None = None,
        config_overrides: list[str] | None = None,
        force_update: bool = False,
    ) -> dict[str, Any]:
        """Copy an app (config, code, checkpoints) into the session root and install its requirements, then
        run it via run_prediction / run_resume / run_evaluation. Local/HuggingFace apps only. Gated by
        ``allow_untrusted_code`` and run in a spawn subprocess."""
        if self._classify(ref) in ("remote", "remote_server"):
            raise ValueError(
                "Importing into the session is only for local or HuggingFace apps; a remote server keeps its "
                "code remote and cannot be imported — drive a remote app with konfai-apps directly."
            )
        self._require_trust(self._infer_mode(ref), allow_untrusted_code, "Importing")
        target = (
            self.workspace_layout.ensure_session_workspace()
        )  # session root (auto-created); in-jail by construction
        from . import runner as mcp_runner

        payload = mcp_runner.run_api_in_subprocess(
            "konfai_mcp.runner:import_app_api",
            {
                "ref": ref,
                "target": str(target),
                "config_overrides": config_overrides,
                "display_name": display_name,
                "force_update": force_update,
            },
        )
        return {
            "ref": ref,
            "imported_to": str(target),
            "files": payload["files"],
            "checkpoints": payload["checkpoints"],
            "configs": payload["configs"],
            "next_actions": ["run_resume", "run_prediction", "run_evaluation", "validate_config_semantics"],
        }

    # -- app resolution helpers (mode + trust gate, shared by import_app) ------------------------

    def _infer_mode(self, ref: str) -> str:
        """Resolve which execution path an app reference takes: ``local`` (local/HF) or ``remote``."""
        kind = self._classify(ref)
        if kind == "remote":
            return "remote"
        if kind in ("local", "hf_app"):
            return "local"
        if kind == "hf_repo":
            raise ValueError(
                f"{ref!r} is a HuggingFace repo, not a single app. Use list_apps / describe_app to pick a "
                "concrete 'repo_id:app_name'."
            )
        raise ValueError(f"Unrecognized app reference: {ref!r}")

    @staticmethod
    def _require_trust(mode: str, allow_untrusted_code: bool, verb: str) -> None:
        """Enforce the code-execution gate for local/HuggingFace apps (remote runs on the user's server)."""
        if mode == "local" and not allow_untrusted_code:
            raise ValueError(
                f"{verb} a local/HuggingFace app imports its Python code and pip-installs its requirements "
                "(the konfai-apps trust model). Set allow_untrusted_code=True to confirm you trust this app source. "
                "Set the env KONFAI_APPS_INSTALL_REQUIREMENTS=0 to skip requirement installs."
            )

    # -- packaging ------------------------------------------------------------------------------

    def package_from_session(
        self,
        name: str,
        display_name: str,
        description: str,
        short_description: str | None = None,
        tta: int = 0,
        mc_dropout: int = 0,
        checkpoints: list[str] | None = None,
        configs: list[str] | None = None,
        model_py: str | None = None,
        requirements: str | None = None,
        output: str | None = None,
        onnx: bool = False,
        onnx_patch_size: list[int] | None = None,
        onnx_in_channels: int | None = None,
    ) -> dict[str, Any]:
        """Package a session's trained model into a resolvable app bundle via ``assemble_bundle``.

        Gathers checkpoints and a config from the current session workspace (or explicit paths),
        synthesizes an ``app.json`` from the given metadata, and writes a bundle (app.json + config +
        checkpoint + optional Model.py/requirements) that ``describe_app`` / ``import_app`` can then
        consume. This closes the train-from-scratch branch onto the same bundle endpoint as fine-tuning.
        """
        from konfai_apps import bundle

        resolved_checkpoints = self._resolve_package_checkpoints(checkpoints)
        resolved_configs = self._resolve_package_configs(configs)
        if model_py is not None and not Path(model_py).expanduser().exists():
            raise ValueError(f"model_py not found: {model_py}")
        if requirements is not None and not Path(requirements).expanduser().exists():
            raise ValueError(f"requirements not found: {requirements}")

        bundle_name = self.workspace_layout.sanitize_name(name)
        out_dir = Path(output).expanduser() if output else (self.workspace_layout.workspace_dir() / "AppBundles")
        metadata = {
            "display_name": display_name,
            "description": description,
            "short_description": short_description or display_name,
            "tta": tta,
            "mc_dropout": mc_dropout,
        }
        # Derive inputs/outputs from the config so the bundle is actually runnable: describe_app reports
        # capabilities.inference from len(get_inputs()) > 0, so without these the packaged app reads as
        # non-runnable and routes the agent back to design_config_strategy instead of import_app.
        inputs, outputs = self._derive_app_io(resolved_configs)
        if inputs:
            metadata["inputs"] = inputs
        if outputs:
            metadata["outputs"] = outputs

        handle, tmp_app_json = tempfile.mkstemp(suffix="_app.json")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                json.dump(metadata, file)
            bundle_path = bundle.assemble_bundle(
                bundle_name,
                out_dir,
                tmp_app_json,
                resolved_configs,
                resolved_checkpoints,
                model_py=str(Path(model_py).expanduser()) if model_py else None,
                requirements=str(Path(requirements).expanduser()) if requirements else None,
            )
        finally:
            Path(tmp_app_json).unlink(missing_ok=True)

        # Copy every support file the config references by classpath (e.g. the UNet.yml YAML model, a local
        # Model.py/Loss.py). assemble_bundle copies the configs + model_py but NOT the files they reference,
        # so a YAML-model bundle would fail at resolve with 'Could not read model YAML file'.
        copied_support = self._copy_referenced_support_files(
            resolved_configs, bundle_path, protected={"Model.py"} if model_py else None
        )
        self._normalize_bundled_prediction_configs(bundle_path, resolved_configs)

        result: dict[str, Any] = {
            "bundle_path": str(bundle_path),
            "checkpoints": [Path(path).name for path in resolved_checkpoints],
            "configs": [Path(path).name for path in resolved_configs],
            "support_files": copied_support,
            "inputs": sorted(inputs) if inputs else [],
            "outputs": sorted(outputs) if outputs else [],
            "next_actions": ["describe_app", "import_app"],
        }
        if any(Path(path).name == "Config.yml" for path in resolved_configs):
            result["warnings"] = [
                "Config.yml is copied from the session as-is: make sure it describes the SAME architecture "
                "as the packaged checkpoints (run_resume with weights_only=True warm-starts from it)."
            ]
        if onnx:
            # The ONNX export instantiates and traces the packaged model: it imports the bundle's
            # Model.py, runs a forward pass, and may init CUDA. That must never run in the long-lived
            # server process, so route it through the spawn subprocess like every other code-executing
            # step. Without checkpoint= the export ships an untrained model.onnx, so pick the NEWEST
            # packaged checkpoint (mirroring discover_model_paths); pass checkpoints=[...] to override.
            from . import runner as mcp_runner

            onnx_path = mcp_runner.run_api_in_subprocess(
                "konfai_apps.bundle:export_onnx_into_bundle",
                {
                    "bundle": str(bundle_path),
                    "patch_size": onnx_patch_size,
                    "in_channels": onnx_in_channels,
                    "checkpoint": max(resolved_checkpoints, key=lambda path: Path(path).stat().st_mtime),
                },
            )
            result["onnx"] = str(onnx_path)
        return result

    def _resolve_package_checkpoints(self, checkpoints: list[str] | None) -> list[str]:
        if checkpoints is None:
            checkpoints_dir = self.workspace_layout.checkpoints_dir()
            run_dirs = (
                sorted((d for d in checkpoints_dir.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
                if checkpoints_dir.exists()
                else []
            )
            if run_dirs:
                # Default to the NEWEST run only: a session often holds several runs, and sweeping
                # them all silently ships foreign experiments' checkpoints in one bundle.
                checkpoints = sorted(str(path) for path in run_dirs[-1].rglob("*.pt"))
            else:
                checkpoints = (
                    sorted(str(path) for path in checkpoints_dir.rglob("*.pt")) if checkpoints_dir.exists() else []
                )
        resolved = [str(Path(path).expanduser()) for path in checkpoints]
        if not resolved:
            raise ValueError(
                "No checkpoints to package: none found under the session Checkpoints/ and none provided. "
                "Pass checkpoints=[...]."
            )
        for path in resolved:
            if not Path(path).exists():
                raise ValueError(f"Checkpoint not found: {path}")
        return resolved

    def _resolve_package_configs(self, configs: list[str] | None) -> list[str]:
        if configs is None:
            # Bundle BOTH the prediction config (to run) and the train config (so a later run_resume can
            # warm-start from the bundle) when present -- a Prediction.yml-only bundle cannot be fine-tuned.
            prediction = self.workspace_layout.config_path("prediction")
            train = self.workspace_layout.config_path("train")
            configs = [str(path) for path in (prediction, train) if path.exists()]
        resolved = [str(Path(path).expanduser()) for path in configs]
        if not resolved:
            raise ValueError(
                "No config to package: no Prediction.yml/Config.yml in the session and none provided. "
                "Pass configs=[...] (an app needs at least a Prediction.yml to run inference)."
            )
        for path in resolved:
            if not Path(path).exists():
                raise ValueError(f"Config not found: {path}")
        return resolved

    @staticmethod
    def _load_config_root(config_path: str) -> tuple[str, dict[str, Any]] | None:
        """Load a workflow config and return its (root_key, root_body); None if unreadable/unrecognized."""
        from .server_support import YAML_SAFE

        try:
            data = YAML_SAFE.load(Path(config_path).read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        for root in ("Predictor", "Trainer", "Evaluator"):
            if isinstance(data.get(root), dict):
                return root, data[root]
        return None

    @staticmethod
    def _volume_type_for(group: str) -> str:
        return "SEGMENTATION" if re.search(r"seg|label|mask|lesion", group, re.IGNORECASE) else "VOLUME"

    def _derive_app_io(self, config_paths: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Derive app.json inputs/outputs from a config: inputs = is_input groups, outputs = produced groups."""
        inputs: dict[str, Any] = {}
        outputs: dict[str, Any] = {}
        # Prefer the prediction config (it declares outputs_dataset); fall back to the train config for inputs.
        roots = [self._load_config_root(path) for path in config_paths]
        prediction = next((body for kind, body in filter(None, roots) if kind == "Predictor"), None)
        any_body = next((body for item in roots if item for body in [item[1]]), None)
        body = prediction if prediction is not None else any_body
        if not isinstance(body, dict):
            return inputs, outputs
        dataset = body.get("Dataset")
        groups_src = dataset.get("groups_src") if isinstance(dataset, dict) else None
        for group, spec in (groups_src or {}).items() if isinstance(groups_src, dict) else []:
            slot = {"display_name": group, "volume_type": self._volume_type_for(group), "required": True}
            if self._spec_is_input(spec):
                # Only a Prediction.yml makes the bundle runnable: app.json inputs drive
                # has_capabilities.inference, so a train-only bundle must not advertise them.
                if prediction is not None:
                    inputs[group] = slot
            else:
                # A non-input group in a train/eval config is the produced target (what a from-scratch model makes).
                outputs.setdefault(group, slot)
        outputs_dataset = body.get("outputs_dataset")
        for spec in outputs_dataset.values() if isinstance(outputs_dataset, dict) else []:
            output_dataset = spec.get("OutputDataset") if isinstance(spec, dict) else None
            out_group = output_dataset.get("group") if isinstance(output_dataset, dict) else None
            if isinstance(out_group, str):
                outputs[out_group] = {
                    "display_name": out_group,
                    "volume_type": self._volume_type_for(out_group),
                    "required": True,
                }
        return inputs, outputs

    @staticmethod
    def _spec_is_input(spec: Any) -> bool:
        """True if any groups_dest entry marks this source group as a model input (KonfAI default: True)."""
        dests = spec.get("groups_dest") if isinstance(spec, dict) else None
        if not isinstance(dests, dict):
            return False
        return any(d.get("is_input", True) if isinstance(d, dict) else True for d in dests.values())

    def _normalize_bundled_prediction_configs(self, bundle_path: Path, config_paths: list[str]) -> None:
        """Rewrite the bundle's prediction config(s) onto the published-app dataset contract.

        konfai_apps stages inference inputs at ``./Dataset/P{idx}/Volume_{i}.<ext>`` in a temp workspace,
        so a runnable bundle must read group ``Volume_i`` from ``./Dataset``. A session Prediction.yml
        instead points at the session's absolute dataset path with the session's group names -- packaged
        as-is it silently predicts on the TRAINING data. Only the bundle copy is rewritten (idempotent on
        already-conformant configs), never the session file.
        """
        from konfai.utils.utils import split_path_spec

        from .server_support import YAML_SAFE, yaml_dump_content

        for config_path in config_paths:
            bundled = bundle_path / Path(config_path).name
            if not bundled.exists():
                continue
            data = YAML_SAFE.load(bundled.read_text(encoding="utf-8"))
            predictor = data.get("Predictor") if isinstance(data, dict) else None
            dataset = predictor.get("Dataset") if isinstance(predictor, dict) else None
            if not isinstance(dataset, dict):
                continue
            renames: dict[str, str] = {}
            groups_src = dataset.get("groups_src")
            if isinstance(groups_src, dict):
                renames = {
                    group: f"Volume_{index}"
                    for index, group in enumerate(g for g, spec in groups_src.items() if self._spec_is_input(spec))
                }
                clobbered = set(renames.values()) & (set(groups_src) - set(renames))
                if clobbered:
                    raise ValueError(
                        f"Cannot package '{bundled.name}': the app contract renames input groups to Volume_0..n, "
                        f"but non-input group(s) {sorted(clobbered)} already use those names. Rename them in the "
                        "session config or pass configs=[...] explicitly."
                    )
                dataset["groups_src"] = {renames.get(group, group): spec for group, spec in groups_src.items()}
            filenames = dataset.get("dataset_filenames")
            if isinstance(filenames, list) and filenames:
                # Keep each entry's accessor/format token (staging symlinks inputs with their original
                # suffix), replace only the path part with the staged ./Dataset/ root. split_path_spec
                # parses from the right, so a Windows drive letter is not mistaken for the token.
                rewritten: list[str] = []
                for entry in filenames:
                    _path, flag, file_format = split_path_spec(str(entry), allowed_flags={"a", "i"})
                    suffix = ":".join(part for part in (flag, file_format) if part)
                    rewritten.append("./Dataset/" + (f":{suffix}" if suffix else ""))
                dataset["dataset_filenames"] = list(dict.fromkeys(rewritten))
            outputs_dataset = predictor.get("outputs_dataset") if isinstance(predictor, dict) else None
            for spec in outputs_dataset.values() if isinstance(outputs_dataset, dict) else []:
                output = spec.get("OutputDataset") if isinstance(spec, dict) else None
                same_as = output.get("same_as_group") if isinstance(output, dict) else None
                if isinstance(same_as, str) and isinstance(output, dict):
                    src, sep, dest = same_as.partition(":")
                    if src in renames:
                        output["same_as_group"] = renames[src] + sep + dest
            bundled.write_text(yaml_dump_content(data), encoding="utf-8")

    def _copy_referenced_support_files(
        self, config_paths: list[str], bundle_path: Path, protected: set[str] | None = None
    ) -> list[str]:
        """Copy config-referenced model/support files (classpath: X.yml, local File:Class -> File.py) into the bundle.

        Copies unconditionally so repackaging under the same name picks up edited session files;
        ``protected`` names (e.g. an explicitly passed model_py already placed as Model.py) are skipped.
        """
        session_dir = self.workspace_layout.workspace_dir().resolve()
        wanted: set[str] = set()
        for config_path in config_paths:
            try:
                text = Path(config_path).read_text(encoding="utf-8")
            except OSError:
                continue
            # classpath: UNet.yml or sub/UNet.yml (the YAML model builder file), session-relative
            wanted.update(re.findall(r"classpath:\s*([\w./-]+\.ya?ml)\b", text))
            # local File:Class references (e.g. Loss:MyWrapper) -> File.py living in the session
            for file_stem in re.findall(r"\b([A-Za-z]\w*):[A-Za-z]\w*\b", text):
                candidate = f"{file_stem}.py"
                if (session_dir / candidate).exists():
                    wanted.add(candidate)
        copied: list[str] = []
        for relative in sorted(wanted):
            if relative in (protected or set()):
                continue
            src = (session_dir / relative).resolve()
            # Containment: a classpath like ../shared/UNet.yml must not read outside the session.
            if not src.is_relative_to(session_dir) or not src.is_file():
                continue
            dst = bundle_path / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            copied.append(relative)
        return copied

    def register_app_source(self, ref: str) -> dict[str, Any]:
        """Append an app reference (app id or bare HF ``repo_id``) to the workspace catalogue file."""
        ref = str(ref).strip()
        if not ref:
            raise ValueError("App reference cannot be empty.")
        path = self._workspace_catalog_path()
        existing = self._read_catalog_file(path) if path.exists() else []
        added = ref not in existing
        if added:
            existing.append(ref)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"apps": existing}, indent=2) + "\n", encoding="utf-8")
        return {
            "ref": ref,
            "added": added,
            "catalog_path": str(path),
            "apps": existing,
            "next_actions": ["list_apps", "describe_app"],
        }

    def unregister_app_source(self, ref: str) -> dict[str, Any]:
        """Remove an app reference from the workspace catalogue file."""
        ref = str(ref).strip()
        path = self._workspace_catalog_path()
        existing = self._read_catalog_file(path) if path.exists() else []
        updated = [item for item in existing if item != ref]
        removed = len(updated) != len(existing)
        if removed:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"apps": updated}, indent=2) + "\n", encoding="utf-8")
        return {
            "ref": ref,
            "removed": removed,
            "catalog_path": str(path),
            "apps": updated,
            "next_actions": ["list_apps"],
        }

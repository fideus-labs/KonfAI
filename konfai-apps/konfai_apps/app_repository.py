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

"""Repository and metadata adapters for KonfAI Apps."""

from __future__ import annotations

import importlib.metadata
import json
import re
import shutil
import subprocess  # nosec B404
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import requests
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.hf_api import RepoFolder
from konfai import RemoteServer
from konfai.utils.errors import AppMetadataError, AppRepositoryError, ConfigError
from konfai.utils.utils import is_windows_absolute_path
from packaging.requirements import InvalidRequirement, Requirement
from ruamel.yaml import YAML


def get_available_apps_on_remote_server(remote_server: RemoteServer) -> list[str]:
    """Return the list of app identifiers exposed by a remote KonfAI app server."""
    r = requests.get(
        f"{remote_server.get_url()}/repo_apps_list",
        headers=remote_server.get_headers(),
        timeout=remote_server.timeout,
    )
    r.raise_for_status()

    data = r.json()
    apps = data.get("apps")

    if not isinstance(apps, list):
        raise ValueError("Invalid response from remote server: expected 'apps' list.")

    return [str(a) for a in apps]


def get_available_apps_on_hf_repo(repo_id: str, force_update: bool) -> list[str]:
    """List app folders available inside a Hugging Face repository."""
    api = HfApi()
    app_names: list[str] = []
    base_repo_id, revision = LocalAppRepositoryFromHF._split_repo_reference(repo_id)

    if force_update:
        try:
            tree = api.list_repo_tree(repo_id=base_repo_id, revision=revision, repo_type="model")
            for entry in tree:
                app_name = Path(entry.path).name
                if isinstance(entry, RepoFolder) and is_app_repo(
                    LocalAppRepositoryFromHF.get_filenames(repo_id, app_name, True)
                ):
                    app_names.append(app_name)
            return app_names
        except Exception as exc:
            raise AppRepositoryError(
                f"Failed to inspect Hugging Face repository '{repo_id}'. "
                "Unable to list its tree and detect valid application folders. "
                "Please check that the repository exists, that you have access to it, "
                "that your authentication is valid, and that your internet connection is working.\n"
                f"Original error: {exc}"
            ) from exc

    try:
        snapshot_dir = snapshot_download(
            repo_id=base_repo_id,
            repo_type="model",
            local_files_only=True,
            revision=revision,
        )  # nosec B615
        root = Path(snapshot_dir)
        for path in root.iterdir():
            if path.is_dir():
                app_name = path.name
                if is_app_repo(LocalAppRepositoryFromHF.get_filenames(repo_id, app_name, False)):
                    app_names.append(app_name)
        return app_names
    except Exception:
        return get_available_apps_on_hf_repo(repo_id, True)


def is_app_repo(filenames: list[str]) -> bool:
    """Return whether the given repository file list looks like a KonfAI app."""
    return "app.json" in filenames


class VolumeType(Enum):
    SEGMENTATION = "SEGMENTATION"
    VOLUME = "VOLUME"
    FIDUCIALS = "FIDUCIALS"
    TRANSFORM = "TRANSFORM"


@dataclass
class VRAMPlanEntry:
    patch_size: list[int]
    batch_size: int


@dataclass
class TerminologyEntry:
    name: str
    color: str


@dataclass
class DataEntry:
    display_name: str
    volume_type: VolumeType
    required: bool


@dataclass(frozen=True, slots=True)
class EvaluationKey:
    display_name: str
    evaluation_file: str


class AppRepositoryInfo(ABC):
    """Common interface implemented by local, HF, and remote app repositories."""

    def __init__(
        self,
        app_name: str,
        display_name: str,
        description: str,
        short_description: str,
        checkpoints_name: list[str],
        checkpoints_name_available: list[str],
        maximum_tta: int,
        mc_dropout: int,
        inputs: dict[str, DataEntry],
        outputs: dict[str, DataEntry],
        inputs_evaluations: dict[EvaluationKey, dict[str, DataEntry]],
        terminology: dict[int, TerminologyEntry] | None = None,
        vram_plan: dict[int, VRAMPlanEntry] | None = None,
        patch_size: list[int] | None = None,
    ) -> None:
        super().__init__()
        self._app_name = app_name
        self._display_name = display_name
        self._description = description
        self._short_description = short_description
        self._checkpoints_name = checkpoints_name
        self._checkpoints_name_available = checkpoints_name_available
        self._maximum_tta = maximum_tta
        self._mc_dropout = mc_dropout
        self._inputs = inputs
        self._outputs = outputs
        self._inputs_evaluations = inputs_evaluations
        self._terminology = terminology
        self._vram_plan = vram_plan
        self._patch_size = patch_size

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  app_name={self._app_name!r},\n"
            f"  display_name={self._display_name!r},\n"
            f"  description={self._description!r},\n"
            f"  short_description={self._short_description!r}\n"
            f"  checkpoints_name={self._checkpoints_name!r},\n"
            f"  maximum_tta={self._maximum_tta!r},\n"
            f"  mc_dropout={self._mc_dropout!r},\n"
            f"  inputs={self._inputs!r},\n"
            f"  outputs={self._outputs!r},\n"
            f"  inputs_evaluations={self._inputs_evaluations!r},\n"
            f"  terminology={self._terminology!r},\n"
            f"  vram_plan={self._vram_plan!r}\n"
            f")"
        )

    def get_display_name(self) -> str:
        return self._display_name

    def get_description(self) -> str:
        return self._description

    def get_short_description(self) -> str:
        return self._short_description

    def get_checkpoints_name(self) -> list[str]:
        return self._checkpoints_name

    def get_checkpoints_name_available(self) -> list[str]:
        return self._checkpoints_name_available

    def get_maximum_tta(self) -> int:
        return self._maximum_tta

    def get_patch_size(self) -> list[int] | None:
        """The app's default inference patch size (per-dim), for UI patch controls; None if undeclared."""
        return self._patch_size

    def get_mc_dropout(self) -> int:
        return self._mc_dropout

    def get_inputs(self) -> dict[str, DataEntry]:
        return self._inputs

    def get_outputs(self) -> dict[str, DataEntry]:
        return self._outputs

    def get_evaluations_inputs(self) -> dict[EvaluationKey, dict[str, DataEntry]]:
        return self._inputs_evaluations

    def get_terminology(self) -> dict[int, TerminologyEntry] | None:
        return self._terminology

    @abstractmethod
    def has_capabilities(self) -> tuple[bool, bool, bool]:
        raise NotImplementedError()

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def download_config_file(self) -> list[Path]:
        raise NotImplementedError()


class LocalAppRepository(AppRepositoryInfo):
    """Base implementation shared by local-directory and Hugging Face apps."""

    def __init__(self, app_name: str) -> None:
        self._app_name = app_name
        filenames = self._get_filenames()
        if not is_app_repo(filenames):
            raise AppRepositoryError("Missing 'app.json' in apps.")

        required_keys = ["description", "short_description", "tta", "mc_dropout", "display_name"]
        for filename in filenames:
            if not filename.endswith(".pt"):
                self._download(filename)
        metadata_file_path = self._download("app.json")
        with open(metadata_file_path, encoding="utf-8") as file:
            app_repository_metadata = json.load(file)

        missing = [key for key in required_keys if key not in app_repository_metadata]
        if missing:
            raise AppMetadataError(f"Missing keys in app.json: {', '.join(missing)}")

        inputs: dict[str, DataEntry] = {}
        if "inputs" in app_repository_metadata:
            inputs = {
                key: DataEntry(
                    display_name=value["display_name"],
                    volume_type=VolumeType(value["volume_type"]),
                    required=bool(value["required"]),
                )
                for key, value in app_repository_metadata["inputs"].items()
            }

        outputs: dict[str, DataEntry] = {}
        if "outputs" in app_repository_metadata:
            outputs = {
                key: DataEntry(
                    display_name=value["display_name"],
                    volume_type=VolumeType(value["volume_type"]),
                    required=bool(value["required"]),
                )
                for key, value in app_repository_metadata["outputs"].items()
            }

        inputs_evaluations: dict[EvaluationKey, dict[str, DataEntry]] = {}
        if "inputs_evaluations" in app_repository_metadata:
            for display_name, by_file in app_repository_metadata["inputs_evaluations"].items():
                for evaluation_file, entries in by_file.items():
                    eval_key = EvaluationKey(display_name=display_name, evaluation_file=evaluation_file)
                    inputs_evaluations[eval_key] = {
                        key: DataEntry(
                            display_name=value["display_name"],
                            volume_type=VolumeType(value["volume_type"]),
                            required=bool(value["required"]),
                        )
                        for key, value in entries.items()
                    }

        try:
            maximum_tta = int(app_repository_metadata["tta"])
        except Exception as exc:
            raise AppMetadataError("The field 'tta' must be an integer.") from exc

        try:
            mc_dropout = int(app_repository_metadata["mc_dropout"])
        except Exception as exc:
            raise AppMetadataError("The field 'mc_dropout' must be an integer.") from exc

        terminology: dict[int, TerminologyEntry] | None = None
        if "terminology" in app_repository_metadata:
            terminology = {
                int(key): TerminologyEntry(name=value["name"], color=value["color"])
                for key, value in app_repository_metadata["terminology"].items()
            }

        vram_plan: dict[int, VRAMPlanEntry] | None = None
        if "vram_plan" in app_repository_metadata:
            vram_plan = {
                int(key): VRAMPlanEntry(
                    patch_size=list(map(int, value["patch_size"])),
                    batch_size=int(value["batch_size"]),
                )
                for key, value in app_repository_metadata["vram_plan"].items()
            }

        patch_size = app_repository_metadata.get("patch_size")
        patch_size = [int(x) for x in patch_size] if isinstance(patch_size, list) else None

        checkpoints_name: list[str] = app_repository_metadata.get("models", [])
        checkpoints_name_available = self._get_available_checkpoint_names(checkpoints_name, filenames)

        super().__init__(
            app_name=app_name,
            display_name=str(app_repository_metadata["display_name"]),
            description=str(app_repository_metadata["description"]),
            short_description=str(app_repository_metadata["short_description"]),
            checkpoints_name=checkpoints_name,
            checkpoints_name_available=checkpoints_name_available,
            maximum_tta=maximum_tta,
            mc_dropout=mc_dropout,
            inputs=inputs,
            outputs=outputs,
            inputs_evaluations=inputs_evaluations,
            terminology=terminology,
            vram_plan=vram_plan,
            patch_size=patch_size,
        )

    def _get_available_checkpoint_names(self, checkpoints_name: list[str], filenames: list[str]) -> list[str]:
        return [
            checkpoint_name
            for checkpoint_name in checkpoints_name
            if self._find_repo_filename(checkpoint_name, filenames, suffix=".pt") is not None
        ]

    def _find_repo_filename(
        self, requested_filename: str, filenames: list[str], suffix: str | None = None
    ) -> str | None:
        candidates = [PurePosixPath(requested_filename).as_posix()]
        if suffix is not None and not candidates[0].endswith(suffix):
            candidates.append(candidates[0] + suffix)

        for candidate in candidates:
            if candidate in filenames:
                return candidate

        basename_matches = [
            filename
            for filename in filenames
            if PurePosixPath(filename).name in {PurePosixPath(c).name for c in candidates}
        ]
        if len(basename_matches) == 1:
            return basename_matches[0]
        if len(basename_matches) > 1:
            raise AppRepositoryError(
                f"Multiple files match '{requested_filename}' in app '{self._app_name}'. "
                "Please use an explicit relative path in the app metadata."
            )
        return None

    def _require_repo_filename(self, requested_filename: str, filenames: list[str], suffix: str | None = None) -> str:
        resolved = self._find_repo_filename(requested_filename, filenames, suffix=suffix)
        if resolved is None:
            raise AppRepositoryError(f"File '{requested_filename}' was not found in app '{self._app_name}'.")
        return resolved

    def _set_number_of_augmentation(self, inference_file_path: str, new_value: int) -> None:
        new_value = int(np.clip(new_value, 0, self._maximum_tta))
        yaml = YAML()
        with open(inference_file_path) as file:
            data = yaml.load(file)

        if new_value > 0:
            tmp = data["Predictor"]["Dataset"]["augmentations"]
            if "DataAugmentation_0" in tmp:
                tmp["DataAugmentation_0"]["nb"] = new_value
        else:
            data["Predictor"]["Dataset"]["augmentations"] = {}

        with open(inference_file_path, "w") as file:
            yaml.dump(data, file)

    def _disable_uncertainty(self, inference_file_path: str) -> None:
        yaml = YAML()
        with open(inference_file_path) as file:
            data = yaml.load(file)

        predictor = data["Predictor"]
        outputs = predictor["outputs_dataset"]

        has_inference_stack = False
        for value in outputs.values():
            after = value["OutputDataset"]["after_reduction_transforms"]
            if "InferenceStack" in after:
                has_inference_stack = True
                break
        if not has_inference_stack:
            return

        predictor["combine"] = "Mean"
        for value in outputs.values():
            value["OutputDataset"]["reduction"] = "Mean"
            if "InferenceStack" in value["OutputDataset"]["after_reduction_transforms"]:
                del value["OutputDataset"]["after_reduction_transforms"]["InferenceStack"]

        with open(inference_file_path, "w") as file:
            yaml.dump(data, file)

    def _set_patch_size_and_batch_size(
        self,
        inference_file_path: str,
        patch_size: list[int] | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Write the inference ``Patch.patch_size`` / ``batch_size``, overriding only the values given.

        A single-element ``patch_size`` is broadcast to the config's spatial dimensionality (an isotropic
        cube), so ``--patch-size 192`` works regardless of 2D/3D; a full list is written verbatim.
        """
        if patch_size is None and batch_size is None:
            return
        yaml = YAML()
        with open(inference_file_path) as file:
            data = yaml.load(file)

        tmp = data["Predictor"]["Dataset"]
        if patch_size is not None:
            if len(patch_size) == 1:
                existing = tmp.get("Patch", {}).get("patch_size")
                dim = len(existing) if isinstance(existing, list) and len(existing) > 1 else 3
                patch_size = patch_size * dim
            tmp["Patch"]["patch_size"] = patch_size
        if batch_size is not None:
            tmp["batch_size"] = batch_size

        with open(inference_file_path, "w") as file:
            yaml.dump(data, file)

    @abstractmethod
    def _get_filenames(self) -> list[str]:
        raise NotImplementedError()

    @abstractmethod
    def _download(self, filename: str) -> Path:
        raise NotImplementedError()

    def _all_repo_filenames(self) -> list[str]:
        """The complete app file list.

        For a Hugging Face repo the local snapshot is populated lazily (one file per ``hf_hub_download``),
        so a snapshot-derived listing can omit bundle assets not pulled yet (e.g. elastix parameter maps).
        Refresh it from the Hub tree once — a lightweight metadata call, no file transfer — so every asset
        is known; the per-file downloads still hit the local cache, and it falls back to the local snapshot
        when offline.
        """
        filenames = self._get_filenames()
        if isinstance(self, LocalAppRepositoryFromHF):
            try:
                filenames = LocalAppRepositoryFromHF.get_filenames(self._repo_id, self._app_name, True)
            except AppRepositoryError:
                pass
        return filenames

    def has_capabilities(self) -> tuple[bool, bool, bool]:
        filenames = self._get_filenames()
        inference_support = len(self.get_inputs()) > 0
        evaluation_support = len(self.get_evaluations_inputs()) > 0
        uncertainty_support = self._find_repo_filename("Uncertainty.yml", filenames) is not None
        return inference_support, evaluation_support, uncertainty_support

    def download_config_file(self) -> list[Path]:
        filenames = self._get_filenames()
        files_path: list[Path] = []
        for filename in filenames:
            if not filename.endswith(".pt"):
                files_path.append(self._download(filename))
        return files_path

    def download_inference(
        self,
        number_of_model: int,
        name_of_models: list[str],
        prediction_file: str,
        install_requirements: bool = False,
    ) -> tuple[list[Path], Path, list[tuple[str, Path]]]:
        filenames = self._all_repo_filenames()
        models_path: list[Path] = []
        codes_path: list[tuple[str, Path]] = []

        inference_file_path = self._download(self._require_repo_filename(prediction_file, filenames))
        available_models = [name for name in filenames if name.endswith(".pt")]
        if len(name_of_models):
            if isinstance(self, LocalAppRepositoryFromHF):
                for name in name_of_models:
                    if self._find_repo_filename(name, filenames, suffix=".pt") is None:
                        filenames = LocalAppRepositoryFromHF.get_filenames(self._repo_id, self._app_name, True)
                        break
            for name in name_of_models:
                models_path.append(self._download(self._require_repo_filename(name, filenames, suffix=".pt")))
        else:
            models_to_download = available_models
            if len(available_models) < number_of_model and isinstance(self, LocalAppRepositoryFromHF):
                remote_filenames = LocalAppRepositoryFromHF.get_filenames(self._repo_id, self._app_name, True)
                remote_models = [name for name in remote_filenames if name.endswith(".pt")]
                models_to_download = available_models + [name for name in remote_models if name not in available_models]
                filenames = remote_filenames
            if len(models_to_download) < number_of_model:
                raise AppRepositoryError(
                    f"Expected {number_of_model} model files (.pt), but found "
                    f"{len(models_to_download)} in the repository."
                )
            for name in models_to_download[:number_of_model]:
                models_path.append(self._download(name))

        # Make every bundle asset (custom .py, elastix parameter maps, lookup tables, …) available in
        # the run workspace, as documented ("files can live in the app directory and will be available
        # at runtime"). Model checkpoints (.pt) are handled separately via ``models_path``.
        for filename in filenames:
            if not filename.endswith(".pt"):
                codes_path.append((filename, self._download(filename)))

        self._install_requirements(filenames, install_requirements)

        return models_path, inference_file_path, codes_path

    def _install_requirements(self, filenames: list[str], install_requirements: bool = False) -> None:
        """Install missing/outdated packages listed in the app's requirements.txt.

        Installation is opt-in (``install_requirements=True``) and defaults to a no-op. Core
        packages (torch, konfai, …) are never installed or altered, and lines that are not
        PEP 508 requirements (``-r``, ``--extra-index-url``, ``git+https``…) are skipped.
        """
        if not install_requirements:
            return
        requirements_filename = self._find_repo_filename("requirements.txt", filenames)
        if requirements_filename is None:
            return
        protected = {"torch", "torchvision", "torchaudio", "konfai", "konfai-apps"}
        with open(self._download(requirements_filename), encoding="utf-8") as file:
            required_lines = [line.strip() for line in file if line.strip() and not line.startswith("#")]
        installed = {
            dist.metadata["Name"].lower(): dist.version
            for dist in importlib.metadata.distributions()
            if dist.metadata.get("Name")
        }
        missing_or_outdated = []
        for line in required_lines:
            try:
                req = Requirement(line)
            except InvalidRequirement:
                continue
            name = req.name.lower()
            if name in protected:
                print(f"[KonfAI-Apps] Skipping protected requirement '{line}'.")
                continue
            installed_version_str = installed.get(name)
            if installed_version_str is None:
                missing_or_outdated.append(line)
                continue
            if req.specifier and not req.specifier.contains(installed_version_str, prereleases=True):
                missing_or_outdated.append(line)

        if missing_or_outdated:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_or_outdated])  # nosec B603
            except subprocess.CalledProcessError as exc:
                raise AppRepositoryError(f"Failed to install packages: {exc}") from exc

    def download_app(self) -> list[tuple[str, Path]]:
        filenames = self._get_filenames()
        files_path: list[tuple[str, Path]] = []
        for filename in filenames:
            files_path.append((filename, self._download(filename)))
            print(f"[KonfAI-Apps] {filename} is ready.")
        return files_path

    def download_evaluation(self, evaluation_file: str) -> tuple[Path, list[tuple[str, Path]]]:
        filenames = self._all_repo_filenames()
        codes_path: list[tuple[str, Path]] = []
        config_filename = self._require_repo_filename(evaluation_file, filenames)
        evaluation_file_path = self._download(config_filename)
        for filename in filenames:
            if filename != config_filename and not filename.endswith(".pt"):
                codes_path.append((filename, self._download(filename)))
        return evaluation_file_path, codes_path

    def download_uncertainty(self, uncertainty_file: str) -> tuple[Path, list[tuple[str, Path]]]:
        filenames = self._all_repo_filenames()
        codes_path: list[tuple[str, Path]] = []
        config_filename = self._require_repo_filename(uncertainty_file, filenames)
        uncertainty_file_path = self._download(config_filename)
        for filename in filenames:
            if filename != config_filename and not filename.endswith(".pt"):
                codes_path.append((filename, self._download(filename)))
        return uncertainty_file_path, codes_path

    def install_inference(
        self,
        number_of_augmentation: int,
        number_of_model: int,
        name_of_models: list[str],
        number_of_mc_dropout: int,
        uncertainty: bool,
        prediction_file: str,
        available_vram: float | None,
        install_requirements: bool = False,
        forced_patch_size: list[int] | None = None,
        forced_batch_size: int | None = None,
    ) -> list[Path]:
        if len(name_of_models) == 0 and number_of_model == 0:
            number_of_model = len(self._checkpoints_name)

        models_path, inference_file_path, codes_path = self.download_inference(
            number_of_model, name_of_models, prediction_file, install_requirements
        )
        # Copy every bundle asset into the workspace first, then write (and tweak) the prediction config
        # last so its modifications are never clobbered by a raw copy of the same file.
        for repo_filename, code_path in codes_path:
            dest = Path(repo_filename)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(code_path, dest)

        shutil.copy2(inference_file_path, prediction_file)
        self._set_number_of_augmentation(prediction_file, number_of_augmentation)
        # NOTE: `number_of_mc_dropout` is reserved for an upcoming MC-dropout feature
        # (stochastic forward passes for models with dropout layers); it is plumbed
        # through but not yet applied to the prediction config.
        if not uncertainty:
            self._disable_uncertainty(prediction_file)
        # Patch/batch precedence: an explicit override wins; otherwise the app's VRAM plan (largest
        # threshold that fits the detected free VRAM); otherwise the config's own defaults are left as-is.
        plan_patch_size: list[int] | None = None
        plan_batch_size: int | None = None
        if self._vram_plan is not None and available_vram is not None:
            thresholds = sorted(self._vram_plan.keys())
            selected_t = thresholds[0]
            for threshold in thresholds:
                if threshold <= available_vram:
                    selected_t = threshold
                else:
                    break
            plan_patch_size = self._vram_plan[selected_t].patch_size
            plan_batch_size = self._vram_plan[selected_t].batch_size
        self._set_patch_size_and_batch_size(
            prediction_file,
            forced_patch_size if forced_patch_size is not None else plan_patch_size,
            forced_batch_size if forced_batch_size is not None else plan_batch_size,
        )

        return models_path

    def install_evaluation(self, evaluation_file: str) -> None:
        evaluation_file_path, codes_path = self.download_evaluation(evaluation_file)
        for repo_filename, code_path in codes_path:
            dest = Path(repo_filename)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(code_path, dest)
        shutil.copy2(evaluation_file_path, evaluation_file)

    def install_uncertainty(self, uncertainty_file: str) -> None:
        uncertainty_file_path, codes_path = self.download_uncertainty(uncertainty_file)
        for repo_filename, code_path in codes_path:
            dest = Path(repo_filename)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(code_path, dest)
        shutil.copy2(uncertainty_file_path, uncertainty_file)

    def _resolve_fine_tune_models(self, filenames: list[str], name_of_models: list[str]) -> list[tuple[str, Path]]:
        """
        Resolve the checkpoint(s) to fine-tune into ``(basename, source_path)`` pairs.

        If ``name_of_models`` is provided, each requested name is resolved to a ``.pt`` file.
        Otherwise the first checkpoint advertised in ``app.json`` (falling back to the first
        available ``.pt``) is selected. Missing files trigger a Hugging Face refresh, mirroring
        :meth:`download_inference`.
        """
        selected: list[str] = []
        if name_of_models:
            if isinstance(self, LocalAppRepositoryFromHF):
                for name in name_of_models:
                    if self._find_repo_filename(name, filenames, suffix=".pt") is None:
                        filenames = LocalAppRepositoryFromHF.get_filenames(self._repo_id, self._app_name, True)
                        break
            selected = [self._require_repo_filename(name, filenames, suffix=".pt") for name in name_of_models]
        else:
            default_name: str | None = None
            for candidate in self._checkpoints_name:
                resolved = self._find_repo_filename(candidate, filenames, suffix=".pt")
                if resolved is not None:
                    default_name = resolved
                    break
            if default_name is None:
                available = sorted(name for name in filenames if name.endswith(".pt"))
                if not available and isinstance(self, LocalAppRepositoryFromHF):
                    filenames = LocalAppRepositoryFromHF.get_filenames(self._repo_id, self._app_name, True)
                    available = sorted(name for name in filenames if name.endswith(".pt"))
                if not available:
                    raise AppRepositoryError(f"No checkpoint (.pt) found to fine-tune in app '{self._app_name}'.")
                default_name = available[0]
            selected = [default_name]

        return [(PurePosixPath(repo_name).name, self._download(repo_name)) for repo_name in selected]

    def install_fine_tune(
        self,
        config_file: str,
        path: Path,
        display_name: str,
        epochs: int,
        it_validation: int | None,
        name_of_models: list[str],
        install_requirements: bool = False,
    ) -> list[tuple[str, Path]]:
        """
        Install the app assets needed for fine-tuning and resolve the selected checkpoint(s).

        Shared assets (config, code, ``app.json``, ``requirements.txt``) are installed into
        ``path``; ``app.json`` is rewritten with the new ``display_name`` and a ``models`` list
        limited to the selected checkpoints, and the training config's ``epochs``/``it_validation``
        are updated. Only the selected ``.pt`` checkpoints are downloaded. Returns ``(basename,
        source_path)`` pairs for the checkpoints to fine-tune.
        """
        filenames = self._get_filenames()
        models = self._resolve_fine_tune_models(filenames, name_of_models)

        for filename in filenames:
            if filename.endswith(".pt"):
                continue
            dest = path / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self._download(filename), dest)

        self._install_requirements(filenames, install_requirements)

        metadata_file = path / "app.json"
        config_file_path = path / config_file
        if not metadata_file.exists():
            raise ConfigError(
                f"Metadata file not found: '{metadata_file}'.",
                "Ensure the metadata file exists and the provided path is correct.",
            )

        with open(metadata_file, encoding="utf-8") as file:
            app_repository_metadata = json.load(file)

        app_repository_metadata["display_name"] = display_name
        app_repository_metadata["models"] = [basename for basename, _ in models]

        with open(metadata_file, "w", encoding="utf-8") as file:
            json.dump(app_repository_metadata, file, indent=2, ensure_ascii=False)

        if not Path(config_file_path).exists():
            raise ConfigError(
                f"Configuration file not found: '{config_file_path}'.",
                "Ensure the configuration file exists and the provided path is correct.",
            )

        yaml = YAML()
        with open(config_file_path) as file:
            data = yaml.load(file)
            data["Trainer"]["epochs"] = epochs
            data["Trainer"]["it_validation"] = it_validation

        with open(config_file_path, "w") as file:
            yaml.dump(data, file)

        return models


class LocalAppRepositoryFromDirectory(LocalAppRepository):
    """KonfAI app repository loaded from a local folder."""

    def __init__(self, app_directory: Path, app_name: str):
        self._app_directory = app_directory
        super().__init__(app_name)

    @staticmethod
    def get_filenames(app_directory: Path, app_name: str) -> list[str]:
        root = app_directory / app_name
        return sorted([path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()])

    def _get_filenames(self) -> list[str]:
        return LocalAppRepositoryFromDirectory.get_filenames(self._app_directory, self._app_name)

    def _download(self, filename: str) -> Path:
        return self._app_directory / self._app_name / filename

    def get_name(self) -> str:
        return str(self._app_directory / self._app_name)


class LocalAppRepositoryFromHF(LocalAppRepository):
    """KonfAI app repository backed by a Hugging Face model repository."""

    def __init__(self, repo_id: str, app_name: str, force_update: bool):
        self._repo_id = repo_id
        self._force_update = force_update
        super().__init__(app_name)

    @staticmethod
    def _split_repo_reference(repo_id: str) -> tuple[str, str | None]:
        base_repo_id, _, revision = repo_id.partition("@")
        return base_repo_id, revision or None

    @staticmethod
    def _list_repo_tree(repo_id: str, app_name: str, recursive: bool = False) -> list[Any]:
        base_repo_id, revision = LocalAppRepositoryFromHF._split_repo_reference(repo_id)
        api = HfApi()
        return list(
            api.list_repo_tree(
                repo_id=base_repo_id,
                path_in_repo=app_name,
                recursive=recursive,
                revision=revision,
                repo_type="model",
            )
        )

    @staticmethod
    def _get_sync_patterns(repo_id: str, app_name: str, requested_filename: str) -> list[str]:
        requested_path = PurePosixPath(requested_filename)
        sync_patterns: set[str] = {requested_path.as_posix()}

        for entry in LocalAppRepositoryFromHF._list_repo_tree(repo_id, app_name, recursive=True):
            if isinstance(entry, RepoFolder):
                continue
            entry_path = PurePosixPath(entry.path)
            if entry_path.suffix != ".pt":
                sync_patterns.add(entry_path.as_posix())

        return sorted(sync_patterns)

    @staticmethod
    def get_filenames(repo_id: str, app_name: str, force_update: bool) -> list[str]:
        if force_update:
            try:
                app_root = PurePosixPath(app_name)
                tree = LocalAppRepositoryFromHF._list_repo_tree(repo_id, app_name, recursive=True)
                return sorted(
                    [
                        PurePosixPath(entry.path).relative_to(app_root).as_posix()
                        for entry in tree
                        if not isinstance(entry, RepoFolder)
                    ]
                )
            except Exception as exc:
                raise AppRepositoryError(
                    f"Failed to list contents of '{app_name}' in Hugging Face repository '{repo_id}'. "
                    "This prevents verifying whether it is a valid application folder. "
                    "Please check that the repository exists, that the path is correct, "
                    "and that you have sufficient access rights.\n"
                    f"Original error: {exc}"
                ) from exc
        try:
            base_repo_id, revision = LocalAppRepositoryFromHF._split_repo_reference(repo_id)
            app_json_path = hf_hub_download(
                repo_id=base_repo_id,
                filename=f"{app_name}/app.json",
                repo_type="model",
                local_files_only=True,
                revision=revision,
            )  # nosec B615
            folder = Path(app_json_path).parent
            return sorted([path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()])
        except Exception:
            return LocalAppRepositoryFromHF.get_filenames(repo_id, app_name, True)

    @staticmethod
    def get_cached_filenames(repo_id: str, app_name: str) -> list[str]:
        try:
            base_repo_id, revision = LocalAppRepositoryFromHF._split_repo_reference(repo_id)
            app_json_path = hf_hub_download(
                repo_id=base_repo_id,
                filename=f"{app_name}/app.json",
                repo_type="model",
                local_files_only=True,
                revision=revision,
            )  # nosec B615
            folder = Path(app_json_path).parent
            return sorted([path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()])
        except Exception:
            return []

    def _get_filenames(self) -> list[str]:
        return LocalAppRepositoryFromHF.get_filenames(self._repo_id, self._app_name, self._force_update)

    def _get_available_checkpoint_names(self, checkpoints_name: list[str], filenames: list[str]) -> list[str]:
        cached_filenames = LocalAppRepositoryFromHF.get_cached_filenames(self._repo_id, self._app_name)
        return [
            checkpoint_name
            for checkpoint_name in checkpoints_name
            if self._find_repo_filename(checkpoint_name, cached_filenames, suffix=".pt") is not None
        ]

    @staticmethod
    def download(repo_id: str, filename: str, force_update: bool) -> Path:
        base_repo_id, revision = LocalAppRepositoryFromHF._split_repo_reference(repo_id)
        if force_update:
            try:
                filename_path = PurePosixPath(filename)
                if len(filename_path.parts) < 2:
                    raise AppRepositoryError(
                        f"Invalid Hugging Face app path '{filename}'. Expected '<app_name>/<file>'."
                    )
                app_name = filename_path.parts[0]
                allow_patterns = LocalAppRepositoryFromHF._get_sync_patterns(repo_id, app_name, filename)

                snapshot_dir = snapshot_download(
                    repo_id=base_repo_id,
                    repo_type="model",
                    revision=revision,
                    allow_patterns=allow_patterns,
                )  # nosec B615
                return Path(snapshot_dir) / filename_path
            except Exception as exc:
                raise AppRepositoryError(
                    f"Failed to download '{filename}' from '{repo_id}'. "
                    "Check your internet connection or repository access."
                ) from exc
        try:
            return Path(
                hf_hub_download(
                    repo_id=base_repo_id,
                    filename=filename,
                    repo_type="model",
                    revision=revision,
                    local_files_only=True,
                )  # nosec B615
            )
        except Exception:
            return LocalAppRepositoryFromHF.download(repo_id, filename, True)

    def _download(self, filename: str) -> Path:
        if not filename.startswith(self._app_name):
            filename = self._app_name + "/" + filename
        return LocalAppRepositoryFromHF.download(self._repo_id, filename, self._force_update)

    def get_app_filenames(self) -> list[str]:
        """Return the app files as paths relative to the app folder."""
        return self._get_filenames()

    def download_files(self, filenames: list[str] | None = None, force_update: bool = True) -> list[Path]:
        """
        Download app files into the local Hugging Face cache and return their local paths.

        Parameters
        ----------
        filenames:
            Paths relative to the app folder; the whole app is downloaded when empty or None.
        force_update:
            Refresh the files from the Hub instead of reusing cached copies.
        """
        paths: list[Path] = []
        for filename in filenames or self.get_app_filenames():
            if not filename.startswith(self._app_name):
                filename = f"{self._app_name}/{filename}"
            paths.append(LocalAppRepositoryFromHF.download(self._repo_id, filename, force_update))
        return paths

    def get_name(self) -> str:
        return f"{self._repo_id}:{self._app_name}"


class AppRepositoryInfoFromRemoteServer(AppRepositoryInfo):
    """Read-only repository adapter backed by a remote KonfAI app server."""

    def __init__(self, remote_server: RemoteServer, app_name: str) -> None:
        self._remote_server = remote_server
        url = f"{remote_server.get_url()}/repo_apps/{app_name}"
        response = requests.get(url, headers=remote_server.get_headers(), timeout=remote_server.timeout)
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        if not data.get("available", False):
            raise AppRepositoryError(f"App '{app_name}' is not available on remote server.")
        self._has_capabilities = data["has_capabilities"]

        inputs = {
            key: DataEntry(
                display_name=str(value["display_name"]),
                volume_type=VolumeType(value["volume_type"]),
                required=bool(value["required"]),
            )
            for key, value in data["inputs"].items()
        }
        outputs = {
            key: DataEntry(
                display_name=str(value["display_name"]),
                volume_type=VolumeType(value["volume_type"]),
                required=bool(value["required"]),
            )
            for key, value in data["outputs"].items()
        }

        inputs_evaluations: dict[EvaluationKey, dict[str, DataEntry]] = {}
        for display_name, by_file in data["inputs_evaluations"].items():
            for evaluation_file, entries in by_file.items():
                eval_key = EvaluationKey(display_name=str(display_name), evaluation_file=str(evaluation_file))
                inputs_evaluations[eval_key] = {
                    key: DataEntry(
                        display_name=str(value["display_name"]),
                        volume_type=VolumeType(value["volume_type"]),
                        required=bool(value["required"]),
                    )
                    for key, value in entries.items()
                }

        terminology: dict[int, TerminologyEntry] | None = None
        if "terminology" in data:
            terminology = {
                int(key): TerminologyEntry(name=str(value["name"]), color=str(value["color"]))
                for key, value in data["terminology"].items()
            }

        super().__init__(
            app_name=data["app"],
            display_name=str(data["display_name"]),
            description=str(data["description"]),
            short_description=str(data["short_description"]),
            checkpoints_name=list(data["checkpoints_name"]),
            checkpoints_name_available=list(data["checkpoints_name_available"]),
            maximum_tta=int(data["maximum_tta"]),
            mc_dropout=int(data["mc_dropout"]),
            inputs=inputs,
            outputs=outputs,
            inputs_evaluations=inputs_evaluations,
            terminology=terminology,
        )

    def has_capabilities(self) -> tuple[bool, bool, bool]:
        return self._has_capabilities

    def get_name(self) -> str:
        return self._app_name

    def download_config_file(self) -> list[Path]:
        import tempfile
        import zipfile

        def safe_name(value: str) -> str:
            return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)

        url = f"{self._remote_server.get_url()}/repo_apps_config/{self._app_name}"
        response = requests.get(url, headers=self._remote_server.get_headers(), timeout=self._remote_server.timeout)
        response.raise_for_status()

        base_tmp = Path(tempfile.gettempdir())
        folder_name = safe_name(
            f"konfai_remote_app_{self._remote_server.host}_{self._remote_server.port}_{self._app_name}"
        )
        app_tmp = base_tmp / folder_name
        if app_tmp.exists():
            shutil.rmtree(app_tmp, ignore_errors=True)
        app_tmp.mkdir(parents=True, exist_ok=True)

        zip_filename = safe_name(f"{self._app_name}_configs.zip")
        zip_path = app_tmp / zip_filename
        with open(zip_path, "wb") as file:
            file.write(response.content)

        extract_dir = app_tmp / "configs"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)

        files = [path for path in extract_dir.iterdir() if path.is_file()]
        if not files:
            raise AppRepositoryError(f"No configuration files received for remote app '{self._app_name}'.")

        return files


def get_app_repository_info(app_id: str, force_update: bool) -> AppRepositoryInfo:
    """
    Resolve an app repository identifier into a concrete repository adapter.

    Supported formats:
    - ``repo_id:app_name`` -> Hugging Face repository
    - ``/path/to/app_repository`` -> local folder
    - ``host:port:app_name`` -> remote KonfAI app server
    - ``host:port:app_name|token`` -> remote KonfAI app server with bearer token
    """
    local_path = _resolve_local_app_path(app_id)
    if local_path is not None:
        if local_path.exists():
            return LocalAppRepositoryFromDirectory(local_path.parent, local_path.name)
        raise AppRepositoryError(f"Local app directory not found: {app_id!r}")

    if app_id.count(":") >= 2:
        host, port_str, name_and_token = app_id.split(":", 2)
        name_and_token_split = name_and_token.split("|")
        name = name_and_token
        token = None
        if len(name_and_token_split) == 2:
            name, token = name_and_token_split
        if port_str.isdigit():
            remote = RemoteServer(host, int(port_str), token)
            return AppRepositoryInfoFromRemoteServer(remote, name)

    if app_id.count(":") == 1:
        repo_id, name = app_id.split(":", 1)
        return LocalAppRepositoryFromHF(repo_id, name, force_update)
    raise AppRepositoryError(
        "Invalid app_id format. Expected one of:\n"
        "  - repo_id:app_name\n"
        "  - /path/to/app_repository\n"
        "  - host:port:app_name\n"
        "  - host:port:app_name|token\n"
        f"Got: {app_id!r}"
    )


def _resolve_local_app_path(app_id: str) -> Path | None:
    """Resolve *app_id* as a local path when it clearly targets the filesystem."""
    path = Path(app_id).expanduser()
    if path.exists():
        return path
    if is_windows_absolute_path(app_id):
        return path
    return None

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

"""Assemble a KonfAI app bundle (the HuggingFace layout) from trained artifacts.

App bundles were previously assembled by hand. :func:`assemble_bundle` builds the
standard bundle folder (``app.json`` + configs + checkpoints + optional ``Model.py`` /
``requirements.txt``) and validates the metadata. With ``--onnx``,
:func:`export_onnx_into_bundle` also emits the portable contract (``model.onnx`` +
``manifest.json``) so the same bundle drives both the PyTorch runtime and the
portable konfai-rs / ONNX-Runtime-Web runtimes.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from konfai.utils.errors import AppMetadataError

REQUIRED_APP_JSON_KEYS = ["display_name", "description", "short_description", "tta", "mc_dropout"]

# import name -> PyPI package name, for best-effort requirements derivation.
_IMPORT_TO_PYPI = {
    "segmentation_models_pytorch": "segmentation-models-pytorch",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
}
# Already provided by konfai / konfai-apps (and their deps): never emitted as a requirement.
_PROVIDED_MODULES = {"torch", "torchvision", "numpy", "scipy", "yaml", "konfai", "konfai_apps"}


def derive_requirements(py_files: list[str | Path]) -> list[str]:
    """Best-effort: infer the *extra* PyPI requirements from imports in custom ``.py`` files.

    Returns third-party packages imported beyond the standard library and what konfai
    already provides (so ``segmentation_models_pytorch`` is kept, ``torch``/``numpy`` are
    not). Heuristic — a draft the author should review; the import→package mapping is
    approximate.
    """
    import ast
    import sys

    stdlib = set(sys.stdlib_module_names)
    found: set[str] = set()
    for py_file in py_files:
        for node in ast.walk(ast.parse(Path(py_file).read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                top = module.split(".")[0]
                if top in stdlib or top in _PROVIDED_MODULES or top.startswith("konfai"):
                    continue
                found.add(_IMPORT_TO_PYPI.get(top, top.replace("_", "-")))
    return sorted(found)


def _derive_onnx_params(config: dict[str, Any], root: str) -> tuple[list[int] | None, int | None, int]:
    """Best-effort ``(patch_size, in_channels, extend_slice)`` read from a prediction config.

    ``patch_size`` is the inference ``Patch`` spatial size (dropping a singleton slice
    dimension for 2.5D); ``in_channels`` comes from the model's ``nb_channel`` /
    ``in_channels`` / first of ``channels``; ``extend_slice`` from the inference ``Patch``.
    Returns ``None`` for values that cannot be derived (the caller may use explicit flags).
    """

    def find_inference_patch(node: Any) -> dict[str, Any] | None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "ModelPatch":
                    continue
                if key == "Patch" and isinstance(value, dict) and "patch_size" in value:
                    return value
                found = find_inference_patch(value)
                if found is not None:
                    return found
        return None

    patch_size: list[int] | None = None
    extend_slice = 0
    patch = find_inference_patch(config)
    if patch is not None:
        raw = patch.get("patch_size")
        if isinstance(raw, list):
            dims = [int(d) for d in raw]
            patch_size = [d for d in dims if d > 1] or dims
        raw_extend = patch.get("extend_slice", 0)
        if isinstance(raw_extend, int):
            extend_slice = raw_extend
        elif isinstance(raw_extend, str) and raw_extend.lstrip("-").isdigit():
            extend_slice = int(raw_extend)

    in_channels: int | None = None
    model_cfg = config.get(root, {}).get("Model", {}) if isinstance(config.get(root), dict) else {}
    for value in model_cfg.values() if isinstance(model_cfg, dict) else []:
        if not isinstance(value, dict):
            continue
        if isinstance(value.get("nb_channel"), int):
            in_channels = value["nb_channel"]
            break
        if isinstance(value.get("in_channels"), int):
            in_channels = value["in_channels"]
            break
        channels = value.get("channels")
        if isinstance(channels, list) and channels and isinstance(channels[0], int):
            in_channels = channels[0]
            break
    return patch_size, in_channels, extend_slice


def assemble_bundle(
    name: str,
    out_dir: str | Path,
    app_json: str | Path,
    configs: list[str],
    checkpoints: list[str],
    model_py: str | None = None,
    requirements: str | None = None,
) -> Path:
    """Assemble ``<out_dir>/<name>/`` in the standard app-bundle layout.

    Validates that ``app.json`` has the required keys and that its ``models`` list (if
    present) matches the provided checkpoints; fills ``models`` from the checkpoints
    otherwise. Returns the bundle directory.
    """
    metadata: dict[str, Any] = json.loads(Path(app_json).read_text())
    missing = [key for key in REQUIRED_APP_JSON_KEYS if key not in metadata]
    if missing:
        raise AppMetadataError(f"app.json is missing required keys: {', '.join(missing)}")

    checkpoint_names = [Path(c).name for c in checkpoints]
    declared = [str(m) for m in metadata.get("models", [])]
    if declared and sorted(declared) != sorted(checkpoint_names):
        raise AppMetadataError(
            f"app.json 'models' {declared} does not match the provided checkpoints {checkpoint_names}",
        )

    bundle = Path(out_dir) / name
    bundle.mkdir(parents=True, exist_ok=True)

    if not declared:
        metadata["models"] = checkpoint_names
    (bundle / "app.json").write_text(json.dumps(metadata, indent=2))

    for config in configs:
        shutil.copy(config, bundle / Path(config).name)
    for checkpoint in checkpoints:
        shutil.copy(checkpoint, bundle / Path(checkpoint).name)
    if model_py is not None:
        shutil.copy(model_py, bundle / "Model.py")
    if requirements is not None:
        shutil.copy(requirements, bundle / "requirements.txt")
    return bundle


def export_onnx_into_bundle(
    bundle: str | Path,
    *,
    patch_size: list[int] | None = None,
    in_channels: int | None = None,
    prediction_config: str = "Prediction.yml",
    checkpoint: str | None = None,
    output_module: str | None = None,
    root: str = "Predictor",
) -> Path:
    """Add ``model.onnx`` + ``manifest.json`` to an assembled bundle.

    Loads the model declared in the bundle's prediction config (its ``Model.classpath``)
    with the given checkpoint — mirroring how ``konfai.predictor`` loads it — and exports
    it via :func:`konfai.export.export_to_onnx`. ``patch_size`` / ``in_channels`` are read
    from the config (the app already declares them) when not given explicitly. The bundle's
    custom ``Model.py`` is made importable for the duration of the call. The config file is
    restored afterwards so the defaults-materialising config reader does not mutate the bundle.
    """
    import torch
    import yaml
    from konfai.export import export_to_onnx, list_output_modules
    from konfai.network.network import ModelLoader
    from konfai.utils.runtime import safe_torch_load

    bundle = Path(bundle)
    config_path = bundle / prediction_config
    if not config_path.exists():
        raise AppMetadataError(f"prediction config '{prediction_config}' not found in bundle {bundle}")

    config = yaml.safe_load(config_path.read_text())
    try:
        classpath = config[root]["Model"]["classpath"]
    except (KeyError, TypeError) as exc:
        raise AppMetadataError(f"could not read {root}.Model.classpath from {config_path}") from exc

    derived_patch, derived_channels, extend_slice = _derive_onnx_params(config, root)
    patch_size = patch_size or derived_patch
    in_channels = in_channels if in_channels is not None else derived_channels
    if not patch_size or not in_channels:
        raise AppMetadataError(
            "could not derive patch_size/in_channels from the config; pass --patch-size and --in-channels",
        )

    config_snapshot = config_path.read_text()
    env_keys = ("KONFAI_config_file", "KONFAI_CONFIG_MODE", "KONFAI_ROOT", "KONFAI_STATE")
    env_backup = {key: os.environ.get(key) for key in env_keys}
    sys.path.insert(0, str(bundle))
    try:
        os.environ["KONFAI_config_file"] = str(config_path)
        os.environ["KONFAI_CONFIG_MODE"] = "Done"
        os.environ["KONFAI_ROOT"] = root
        os.environ["KONFAI_STATE"] = "PREDICTION"

        model = ModelLoader(classpath).get_model(train=False)
        # Disable the model's internal patch-based forward: we export the per-patch
        # network and let the portable runtime do the sliding-window tiling.
        model.patch = None
        model.eval()
        if checkpoint is not None:
            ckpt_path = Path(checkpoint)
            if not ckpt_path.is_absolute():
                ckpt_path = bundle / ckpt_path.name
            state = safe_torch_load(ckpt_path, "cpu")
            model.load(state, init=False)

        example = torch.randn(1, in_channels, *patch_size)
        head = output_module or list_output_modules(model, example)[-1][0]
        return export_to_onnx(model, bundle, example, head, extend_slice=extend_slice)
    finally:
        sys.path.remove(str(bundle))
        config_path.write_text(config_snapshot)
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_bundle_cli(args: dict[str, Any]) -> None:
    """Entry point for the ``konfai-apps bundle`` subcommand."""
    bundle = assemble_bundle(
        name=args["name"],
        out_dir=args["out"],
        app_json=args["app_json"],
        configs=args["config"],
        checkpoints=args["checkpoint"],
        model_py=args.get("model_py"),
        requirements=args.get("requirements"),
    )
    print(f"Bundle assembled at {bundle}")

    # If no requirements.txt was provided, draft one from the custom Model.py imports.
    if not args.get("requirements") and (bundle / "Model.py").exists():
        drafted = derive_requirements([bundle / "Model.py"])
        if drafted:
            (bundle / "requirements.txt").write_text("\n".join(drafted) + "\n")
            print(f"Drafted requirements.txt (review!): {', '.join(drafted)}")

    if args.get("onnx"):
        onnx_path = export_onnx_into_bundle(
            bundle,
            patch_size=args.get("patch_size"),
            in_channels=args.get("in_channels"),
            checkpoint=Path(args["checkpoint"][0]).name if args.get("checkpoint") else None,
            output_module=args.get("output_module"),
        )
        print(f"Portable model exported: {onnx_path} (+ manifest.json)")

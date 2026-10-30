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

:func:`assemble_bundle` writes the bundle folder (``app.json`` + configs + checkpoints +
optional ``Model.py`` / ``requirements.txt``) and validates the metadata. With ``--onnx``,
:func:`export_portable_into_bundle` also emits the portable runtime's artifacts: ``model.onnx`` +
``manifest.json`` for one checkpoint, one ``.onnx`` per checkpoint + ``program.json`` otherwise.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path, PureWindowsPath
from typing import Any

from konfai.portable import aux_mask_groups, compile_output, config_environment, tta_passes, warn_unportable_keys

from .errors import AppMetadataError

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
    """Best-effort: the *extra* PyPI requirements imported by custom ``.py`` files.

    Returns third-party packages beyond the standard library and what konfai provides
    (``segmentation_models_pytorch`` kept, ``torch``/``numpy`` dropped). A draft to review.
    """
    import ast

    stdlib = set(sys.stdlib_module_names)
    found: set[str] = set()
    for py_file in py_files:
        for node in ast.walk(ast.parse(Path(py_file).read_bytes())):
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


def _support_relative(value: str, role: str) -> Path:
    """One portable relative name, never a traversal or a Windows drive on a POSIX host."""
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or PureWindowsPath(value).drive
        or "\\" in value
        or ".." in path.parts
        or path == Path(".")
    ):
        raise AppMetadataError(f"Invalid {role} {value!r}: use a relative path below its declared root.")
    return path


def resolve_support_files(
    support_files: dict[str, str] | None,
    support_root: str | Path | None,
) -> dict[str, Path]:
    """Plan explicitly declared files/directories without importing code or following imports.

    Keys are bundle-relative destinations; values are paths below the required source root.
    Directories expand deterministically. In-root symlinks are copied as regular files, while
    escaping links, cycles and special files are refused before anything is written.
    """
    if not support_files:
        return {}
    if support_root is None:
        raise AppMetadataError("support_root is required when support_files are declared.")
    root = Path(support_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise AppMetadataError(f"Support root is not a directory: {root}")
    planned: dict[str, Path] = {}

    def visit(source: Path, destination: Path, ancestors: set[Path]) -> None:
        try:
            resolved = source.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise AppMetadataError(f"Cannot read support path {source}: {error}") from error
        if not resolved.is_relative_to(root):
            raise AppMetadataError(f"Support path {source} escapes support_root {root}.")
        if resolved.is_dir():
            if resolved in ancestors:
                raise AppMetadataError(f"Support directory cycle at {source}.")
            for child in sorted(resolved.iterdir()):
                visit(child, destination / child.name, ancestors | {resolved})
        elif resolved.is_file():
            name = destination.as_posix()
            if name in planned and planned[name] != resolved:
                raise AppMetadataError(f"Different support files would overwrite {name!r}.")
            planned[name] = resolved
        else:
            raise AppMetadataError(f"Support path is not a regular file or directory: {source}")

    for destination, source in sorted(support_files.items()):
        visit(
            root / _support_relative(source, "support source"),
            _support_relative(destination, "support destination"),
            set(),
        )
    _check_file_layout(planned)
    return dict(sorted(planned.items()))


def _check_file_layout(files: dict[str, Any]) -> None:
    for name in files:
        for parent in Path(name).parents:
            if parent.as_posix() in files:
                raise AppMetadataError(f"Bundle paths collide: {parent.as_posix()!r} is both a file and a directory.")


def assemble_bundle(
    name: str,
    out_dir: str | Path,
    app_json: str | Path,
    configs: list[str],
    checkpoints: list[str],
    model_py: str | None = None,
    requirements: str | None = None,
    support_files: dict[str, str] | None = None,
    support_root: str | Path | None = None,
) -> Path:
    """Assemble ``<out_dir>/<name>/`` in the standard app-bundle layout.

    Validates that ``app.json`` has the required keys and that its ``models`` list (if
    present) matches the provided checkpoints; fills ``models`` from the checkpoints
    otherwise. Two checkpoints (or two configs) that would land under one bundle filename are
    refused before anything is written: a flat copy kept the last one and the manifest named both.

    The export is assembled in a staging directory beside the bundle before its files are replaced. An
    existing bundle keeps the files the export does not write (a user's notes, an icon), except the
    checkpoints its previous ``app.json`` declared and the new export does not: those are the
    previous export's, and left in place they were what the default inference picked up.
    ``support_files`` explicitly maps bundle-relative destinations to files/directories below
    ``support_root`` (required with this option). Only these paths are copied; imports and asset
    references are not recursively inferred. The manifest records the expanded managed files.
    Returns the bundle directory.
    """
    metadata: dict[str, Any] = json.loads(Path(app_json).read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_APP_JSON_KEYS if key not in metadata]
    if missing:
        raise AppMetadataError(f"app.json is missing required keys: {', '.join(missing)}")

    checkpoint_names = [Path(c).name for c in checkpoints]
    config_names = [Path(c).name for c in configs]
    for kind, names in (("checkpoints", checkpoint_names), ("configs", config_names)):
        colliding = sorted({item for item in names if names.count(item) > 1})
        if colliding:
            raise AppMetadataError(
                f"Two {kind} would land under the same bundle filename {colliding} and overwrite each other. "
                "Rename them (fold0_best.pt, fold1_best.pt) before packaging."
            )
    declared = [str(m) for m in metadata.get("models", [])]
    if declared and sorted(declared) != sorted(checkpoint_names):
        raise AppMetadataError(
            f"app.json 'models' {declared} does not match the provided checkpoints {checkpoint_names}",
        )
    if not declared:
        metadata["models"] = checkpoint_names

    if _support_relative(name, "bundle name").name != name:
        raise AppMetadataError("Bundle name must be a single directory name.")
    support = resolve_support_files(support_files, support_root)
    files: dict[str, Path | None] = {"app.json": None}

    def add(destination: str, source: str | Path) -> None:
        path = Path(source).resolve(strict=True)
        if destination in files and files[destination] != path:
            raise AppMetadataError(f"Different bundle files would overwrite {destination!r}.")
        files[destination] = path

    for source in [*configs, *checkpoints]:
        add(Path(source).name, source)
    if model_py is not None:
        add("Model.py", model_py)
    if requirements is not None:
        add("requirements.txt", requirements)
    for destination, planned in support.items():
        add(destination, planned)
    _check_file_layout(files)
    metadata["support_files"] = sorted(support)
    bundle = Path(out_dir) / name
    staging = bundle.with_name(f".{name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "app.json").write_text(json.dumps(metadata, indent=2))
        for destination, staged in files.items():
            if staged is not None:
                target = staging / destination
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(staged, target)
        _replace_bundle(staging, bundle, metadata["models"])
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return bundle


def _replace_bundle(staging: Path, bundle: Path, models: list[str]) -> None:
    """Move the staged export over ``bundle``: a fresh bundle is the staging directory renamed; an
    existing one takes every staged file and loses the checkpoints its previous ``app.json``
    declared that ``models`` no longer does."""
    if not bundle.exists():
        staging.rename(bundle)
        return
    previous: list[str] = []
    try:
        old_metadata = json.loads((bundle / "app.json").read_text(encoding="utf-8"))
        previous = [str(m) for m in old_metadata.get("models", [])] + [
            str(path) for path in old_metadata.get("support_files", [])
        ]
    except (OSError, ValueError):
        pass
    staged_files = sorted(path for path in staging.rglob("*") if path.is_file())
    retained = {(bundle / path.relative_to(staging)).resolve() for path in staged_files}
    obsolete: dict[Path, Path] = {}  # by resolved path: two spellings of one file unlink once
    for name in sorted(set(previous)):
        target = bundle / name
        if Path(name).is_absolute() or not target.resolve().is_relative_to(bundle.resolve()):
            raise AppMetadataError(
                f"Previous app.json declares managed file {name!r} outside the bundle; "
                "correct its models/support_files lists before replacing this export."
            )
        if target.resolve() in retained:
            continue  # './model.pt' and 'model.pt' can name the same retained checkpoint.
        if target.is_file():
            obsolete.setdefault(target.resolve(), target)
    removable = set(obsolete)
    # Preflight the entire update: never write through a destination's symlink or replace a user's
    # file or directory. A managed file this export no longer declares may become a directory.
    for staged in staged_files:
        target = bundle / staged.relative_to(staging)
        for part in [target, *target.parents]:
            if part == bundle:
                break
            if part.is_symlink():
                raise AppMetadataError(f"Bundle destination contains a symlink: {part}")
            if part != target and part.exists() and not part.is_dir() and part.resolve() not in removable:
                raise AppMetadataError(f"Bundle destination parent is not a directory: {part}")
        if target.is_dir():
            raise AppMetadataError(f"Bundle destination is a directory, not a file: {target}")
    for target in obsolete.values():
        target.unlink()
    for staged in sorted(staged_files, key=lambda path: path.name == "app.json"):
        target = bundle / staged.relative_to(staging)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged.replace(target)
    shutil.rmtree(staging)


def export_nested_model(group_name: str, inference: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Resolve the app a ``KonfAIInference`` references (``repo_id:model_name``), export its single-model
    onnx + manifest, copy the onnx into the main bundle as ``<group>.onnx``, and return ``{id, manifest}``
    for the program. The nested model's own transforms (resample / argmax / inverse) live in its manifest."""
    import tempfile

    from konfai_apps.app import KonfAIApp

    # Several checkpoints would make the nested export emit `program.json` + per-fold onnx instead of the
    # `model.onnx` + `manifest.json` this step copies, so refuse it here rather than fail on a missing file.
    # KonfAIInference reads ["fold_0"] where the config names none: the same default, or the nested export
    # would load no weights at all.
    nested_checkpoints = list(inference.get("checkpoints_name") or ["fold_0"])
    if len(nested_checkpoints) > 1:
        raise AppMetadataError(
            f"nested model '{group_name}' declares {len(nested_checkpoints)} checkpoints; the portable "
            "runtime supports a single-checkpoint nested model (a mask/condition producer) only.",
        )

    app = KonfAIApp(f"{inference['repo_id']}:{inference['model_name']}", True, False)
    with tempfile.TemporaryDirectory() as tmp:
        # Materialise the bundle under its real filenames (the HF cache stores blobs under content hashes,
        # but the config's bundle-relative classpaths need the real names).
        nested_dir = Path(tmp)
        for name, path in app.app_repository.download_app():
            dest = nested_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, dest)
        artifact = export_portable_into_bundle(nested_dir, checkpoints=nested_checkpoints)
        # Test-time copies or a mask of its own make the nested export a program, which this step cannot nest.
        # The export's result says so: the app may publish a program.json of its own.
        if artifact.name == "program.json":
            raise AppMetadataError(
                f"nested model '{group_name}' compiles to a program (test-time copies or a mask of its own); the"
                " portable runtime nests one plain model."
            )
        shutil.copy(artifact, out_dir / f"{group_name}.onnx")
        manifest = json.loads((nested_dir / "manifest.json").read_text(encoding="utf-8"))
    return {"id": group_name, "manifest": manifest}


def export_portable_into_bundle(
    bundle: str | Path,
    *,
    checkpoints: list[str] | None = None,
    patch_size: list[int] | None = None,
    in_channels: int | None = None,
    prediction_config: str = "Prediction.yml",
    output_module: str | None = None,
    root: str = "Predictor",
) -> Path:
    """Write the bundle's portable-runtime artifacts from its prediction config.

    One checkpoint, no test-time copy and no mask -> ``model.onnx`` + ``manifest.json``. Otherwise one
    ``<id>.onnx`` per checkpoint plus ``program.json``: the program :func:`konfai.portable.compile_output`
    assembles, every checkpoint on every test-time pass, the nested models a ``Mask`` reads, folded by the
    reduction the config declares. The patch geometry, transforms, blend, and reduction are all read from
    the config: nothing is per-app. The config file is restored afterwards because reading it mutates it.
    """
    import yaml

    bundle = Path(bundle)
    config_path = bundle / prediction_config
    if not config_path.exists():
        raise AppMetadataError(f"prediction config '{prediction_config}' not found in bundle {bundle}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_section = (config.get(root) or {}).get("Model") if isinstance(config, dict) else None
    if not isinstance(model_section, dict) or "classpath" not in model_section:
        raise AppMetadataError(f"could not read {root}.Model.classpath from {config_path}")
    warn_unportable_keys(config, root)

    # `None` stands for "the config's own checkpoint", so the export is uniform over both cases.
    selected: list[str | None] = list(checkpoints) if checkpoints else [None]
    ensemble = len(selected) > 1
    with config_environment(config_path, root=root, state="PREDICTION", keep_file=True):
        # A relative checkpoint is the bundle's own file; `None` is the weights the model builds with.
        weights = [
            None
            if checkpoint is None
            else Path(checkpoint)
            if Path(checkpoint).is_absolute()
            else _bundle_checkpoint(bundle, checkpoint)
            for checkpoint in selected
        ]
        ids = [Path(checkpoint).stem if ensemble and checkpoint is not None else "model" for checkpoint in selected]
        # The nested models a Mask reads, exported first: a chain the program cannot run is refused before
        # the bundle's own models are.
        aux = {
            name: {**export_nested_model(name, group["inference"], bundle), "ops": group["ops"]}
            for name, group in aux_mask_groups(config, root).items()
        }
        passes = tta_passes(config, root)
        _models, written, program = compile_output(
            config,
            root,
            weights,
            bundle,
            ids,
            passes=passes,
            aux=aux,
            head=output_module,
            patch_size=patch_size,
            in_channels=in_channels,
            write_manifest=not ensemble,
        )
    portable_assets = set(written) | {bundle / f"{name}.onnx" for name in aux}
    if not ensemble:
        portable_assets.add(bundle / "manifest.json")
    # One model on the identity pass, masked by nothing, is its manifest: konfai's reduction of a single copy
    # hands it back. The test-time copies and a mask ride on a program.
    if not (ensemble or aux or len(passes) > 1):
        return _declare_portable_assets(bundle, written[-1], portable_assets)
    program_path = bundle / "program.json"
    program_path.write_text(json.dumps(program, indent=2))
    portable_assets.add(program_path)
    return _declare_portable_assets(bundle, program_path, portable_assets)


def _bundle_checkpoint(bundle: Path, name: str) -> Path:
    """The bundle's checkpoint ``name``, found as konfai-apps finds it: as spelled, else with ``.pt`` added."""
    path = bundle / Path(name).name
    return path if path.exists() or path.name.endswith(".pt") else path.with_name(f"{path.name}.pt")


def _declare_portable_assets(bundle: Path, artifact: Path, produced: set[Path]) -> Path:
    """Record in the bundle's ``app.json`` the files only the portable runtime consumes (the ONNX
    models, the manifest and the program), so a native run does not copy them into its workspace
    (``download_inference``). Only artifacts this export produced receive a role: a custom native
    app can already depend on another ONNX file. The core exporter returns self-contained models
    and removes their external-data sidecars. Returns ``artifact`` unchanged."""
    metadata_path = bundle / "app.json"
    if not metadata_path.exists():
        return artifact
    portable = {path.resolve().relative_to(bundle.resolve()).as_posix() for path in produced if path.is_file()}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return artifact
    declared = [str(name) for name in metadata.get("portable_assets", [])]
    metadata["portable_assets"] = sorted(set(declared) | set(portable))
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return artifact


def run_bundle_cli(args: dict[str, Any]) -> None:
    """Entry point for the ``konfai-apps bundle`` subcommand."""
    support: dict[str, str] = {}
    for declaration in args.get("support_file", []):
        if "=" not in declaration:
            raise AppMetadataError("--support-file expects DESTINATION=SOURCE, both relative paths.")
        destination, source = declaration.split("=", 1)
        if destination in support and support[destination] != source:
            raise AppMetadataError(f"Conflicting --support-file mappings for {destination!r}.")
        support[destination] = source
    bundle = assemble_bundle(
        name=args["name"],
        out_dir=args["out"],
        app_json=args["app_json"],
        configs=args["config"],
        checkpoints=args["checkpoint"],
        model_py=args.get("model_py"),
        requirements=args.get("requirements"),
        support_files=support,
        support_root=args.get("support_root"),
    )
    print(f"[KonfAI-Apps] Bundle assembled at {bundle}")

    # If no requirements.txt was provided, draft one from the custom Model.py imports.
    if not args.get("requirements") and (bundle / "Model.py").exists():
        drafted = derive_requirements([bundle / "Model.py"])
        # A declared local package is not a PyPI dependency. Requirements remain a reviewed draft;
        # packaging deliberately does not execute code to discover dependencies.
        local_modules = {path.stem for path in bundle.glob("*.py")} | {
            path.name for path in bundle.iterdir() if path.is_dir()
        }
        drafted = [name for name in drafted if name.replace("-", "_") not in local_modules]
        if drafted:
            (bundle / "requirements.txt").write_text("\n".join(drafted) + "\n")
            print(f"[KonfAI-Apps] Drafted requirements.txt (review!): {', '.join(drafted)}")

    if args.get("onnx"):
        artifact = export_portable_into_bundle(
            bundle,
            checkpoints=[Path(c).name for c in args["checkpoint"]] if args.get("checkpoint") else None,
            patch_size=args.get("patch_size"),
            in_channels=args.get("in_channels"),
            output_module=args.get("output_module"),
        )
        print(f"[KonfAI-Apps] Portable model exported: {artifact}")

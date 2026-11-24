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
import subprocess  # nosec B404
import sys
import tempfile
from collections.abc import Callable, Collection
from pathlib import Path, PureWindowsPath
from typing import Any

from konfai.portable import aux_mask_groups, compile_output, config_environment, tta_passes, warn_unportable_keys
from konfai.utils.utils import owned_destination

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


def derive_requirements(py_files: list[str | Path], local: Collection[str] = ()) -> list[str]:
    """Best-effort: the *extra* PyPI requirements imported by custom ``.py`` files.

    Returns third-party packages beyond the standard library, what konfai provides
    (``segmentation_models_pytorch`` kept, ``torch``/``numpy`` dropped) and the ``local`` modules, by the name they are
    imported under. A draft to review.
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
                if top in stdlib or top in _PROVIDED_MODULES or top.startswith("konfai") or top in local:
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
    weights = sorted(name for name in planned if name.endswith((".pt", ".safetensors")))
    if weights:
        raise AppMetadataError(
            f"Support files {weights} carry a weight suffix (.pt, .safetensors), which an app reserves for its "
            "models: pass them as checkpoints, or store the data under another extension."
        )
    return dict(sorted(planned.items()))


def _check_file_layout(files: dict[str, Any]) -> None:
    """Refuse two bundle files that one file system would hold as one: a file that is another's folder, or two
    names telling apart by case alone (a file system may tell no case apart)."""
    folded: dict[str, str] = {}
    for name in files:
        if folded.setdefault(name.lower(), name) != name:
            raise AppMetadataError(
                f"Bundle paths collide: {folded[name.lower()]!r} and {name!r} are one file on a file system telling"
                " no case apart."
            )
    for name in files:
        for parent in Path(name).parents:
            if parent.as_posix().lower() in folded:
                raise AppMetadataError(f"Bundle paths collide: {parent.as_posix()!r} is both a file and a directory.")


def _published(name: str) -> str:
    """The bundle filename of a checkpoint: a ``.pt`` is published as ``.safetensors`` weights."""
    return name.removesuffix(".pt") + ".safetensors" if name.endswith(".pt") else name


def _konfai_scalar_globals() -> list[Any]:
    """The non-tensor types a KonfAI run pickles into its checkpoint (epoch counters as numpy scalars, config
    floats as ruamel scalars): let through the weights-only unpickler, which holds no code of its own."""
    import numpy as np
    from numpy._core import multiarray

    allowed: list[Any] = [np.dtype, np.ndarray, multiarray.scalar, multiarray._reconstruct]
    allowed += [getattr(np.dtypes, name) for name in dir(np.dtypes) if name.endswith("DType")]
    try:
        from ruamel.yaml.anchor import Anchor  # every ruamel scalar carries one
        from ruamel.yaml.scalarbool import ScalarBoolean
        from ruamel.yaml.scalarfloat import ScalarFloat
        from ruamel.yaml.scalarint import ScalarInt
        from ruamel.yaml.scalarstring import ScalarString

        allowed += [ScalarFloat, ScalarInt, ScalarBoolean, ScalarString, Anchor]
    except ImportError:  # pragma: no cover - ruamel is a KonfAI dependency
        pass
    return allowed


def _publish_weights(checkpoint: Path, target: Path) -> None:
    """Publish a checkpoint as weights only. A run's ``.pt`` is read without unpickling objects
    (packaging may run in a server process, and a run's checkpoint holds tensors and primitives only);
    a ``.safetensors`` is copied as it is once its names are KonfAI weight names, so one in another
    layout is refused here rather than at inference."""
    import torch
    from konfai.utils.runtime import check_weights, save_weights

    try:
        if checkpoint.suffix != ".safetensors":
            # weights only, plus the scalar types a KonfAI run writes beside its tensors (numpy scalars, ruamel
            # floats): the bare unpickler refused every run's own checkpoint; any other object is still refused
            with torch.serialization.safe_globals(_konfai_scalar_globals()):
                save_weights(torch.load(checkpoint, map_location="cpu", weights_only=True), target)
            return
        check_weights(checkpoint)
    except Exception as exc:
        raise AppMetadataError(f"'{checkpoint}' does not load as KonfAI weights: {exc}") from exc
    shutil.copy(checkpoint, target)


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
    extension: str | Path | None = None,
    finish: Callable[[Path], None] | None = None,
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
    ``extension``, the app's extension of KonfAI Edge, lands under its own name. ``finish`` runs on the bundle the
    export makes, a copy beside it that then takes its place: what it writes there goes in with it, and what it raises
    leaves the bundle as it was. The copy links the checkpoints, which ``finish`` reads or replaces whole, never
    writes in place, and copies every other file. Returns the bundle directory.
    """
    metadata: dict[str, Any] = json.loads(Path(app_json).read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_APP_JSON_KEYS if key not in metadata]
    if missing:
        raise AppMetadataError(f"app.json is missing required keys: {', '.join(missing)}")

    # A run's .pt is published as weights only (.safetensors), under its own stem.
    checkpoint_names = [_published(Path(c).name) for c in checkpoints]
    config_names = [Path(c).name for c in configs]
    for kind, names in (("checkpoints", checkpoint_names), ("configs", config_names)):
        colliding = sorted({item for item in names if names.count(item) > 1})
        if colliding:
            raise AppMetadataError(
                f"Two {kind} would land under the same bundle filename {colliding} and overwrite each other. "
                "Rename them (fold0_best.pt, fold1_best.pt) before packaging."
            )
    declared = [_published(str(m)) for m in metadata.get("models", [])]
    if declared and sorted(declared) != sorted(checkpoint_names):
        raise AppMetadataError(
            f"app.json 'models' {declared} does not match the provided checkpoints {checkpoint_names}",
        )
    metadata["models"] = declared or checkpoint_names

    if _support_relative(name, "bundle name").name != name:
        raise AppMetadataError("Bundle name must be a single directory name.")
    if name.startswith("."):
        raise AppMetadataError(
            f"Bundle name {name!r} starts with '.', which names the command's own folders beside it."
        )
    support = resolve_support_files(support_files, support_root)
    files: dict[str, Path | None] = {"app.json": None}

    def add(destination: str, source: str | Path) -> None:
        path = Path(source).resolve(strict=True)
        if destination in files and files[destination] != path:
            raise AppMetadataError(f"Different bundle files would overwrite {destination!r}.")
        files[destination] = path

    for source in configs:
        add(Path(source).name, source)
    for source, published in zip(checkpoints, checkpoint_names, strict=True):
        add(published, source)
    if model_py is not None:
        add("Model.py", model_py)
    if requirements is not None:
        add("requirements.txt", requirements)
    for destination, planned in support.items():
        add(destination, planned)
    if extension is not None:
        add(Path(extension).name, extension)
    _check_file_layout(files)
    metadata["support_files"] = sorted(support)
    bundle = Path(out_dir) / name
    bundle = bundle.resolve() if bundle.is_symlink() else bundle  # a bundle reached by a link: where it leads
    staging = bundle.with_name(f".{bundle.name}.staging-{os.getpid()}")
    update = bundle.with_name(f".{bundle.name}.update-{os.getpid()}")
    previous = bundle.with_name(f".{bundle.name}.previous")  # no process's own: the next one finds it
    if previous.exists() and not bundle.exists():
        previous.rename(bundle)  # an earlier update left the bundle only there
    for scratch in (staging, update, previous):
        if scratch.exists():
            try:
                shutil.rmtree(scratch)
            except OSError as error:
                raise AppMetadataError(
                    f"{scratch} is left from an earlier update and cannot be removed ({error}).",
                    "Remove it, then package again.",
                ) from error
    staging.mkdir(parents=True)
    try:
        (staging / "app.json").write_text(json.dumps(metadata, indent=2))
        for destination, staged in files.items():
            if staged is not None:
                target = staging / destination
                target.parent.mkdir(parents=True, exist_ok=True)
                if destination in checkpoint_names:
                    _publish_weights(staged, target)
                else:
                    shutil.copy(staged, target)
        if finish is None:
            _replace_bundle(staging, bundle, metadata["models"])
            return bundle
        # The update is finished on the bundle it makes, which then takes the bundle's place: the bundle as it was,
        # every file the export reads beside the staged ones, none of those the update drops.
        if bundle.exists():
            shutil.copytree(bundle, update, symlinks=True, copy_function=lambda src, dst: _carried(bundle, src, dst))
            # A link into the bundle, by its absolute path or out of it and back, reads the update's file there, as the
            # bundle published will.
            for link in [path for path in update.rglob("*") if path.is_symlink()]:
                # What the link names, its folders read as the system reads them, its last part not followed: a chain
                # of links into the bundle is kept a chain.
                named = Path(os.readlink(link))
                held = bundle.absolute() / link.relative_to(update).parent
                last = named.name not in ("", ".", "..")
                target = Path(os.path.realpath(held / (named.parent if last else named))) / (named.name if last else "")
                for spelled in {bundle.absolute(), bundle.resolve()}:
                    if target.is_relative_to(spelled):
                        link.unlink()
                        link.symlink_to(os.path.relpath(update / target.relative_to(spelled), link.parent))
                        break
        _replace_bundle(staging, update, metadata["models"])
        finish(update)
        if bundle.exists():
            bundle.rename(previous)
            try:
                update.rename(bundle)
            except OSError:
                previous.rename(bundle)  # the update refused its place: the bundle takes its own back
                raise
            # The update has its place: a copy of the old bundle left behind fails nothing, the next one removes it.
            shutil.rmtree(previous, ignore_errors=True)
        else:
            update.rename(bundle)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(update, ignore_errors=True)
    return bundle


def _import_path(entries: list[str], bundle: Path, update: Path) -> list[str]:
    """The import path a bundle's update is checked under: the update first, then each of ``entries`` absolute, the
    empty one the caller's working folder, one inside the bundle its counterpart in the update."""
    held = Path(os.path.realpath(bundle))
    path = [str(update.absolute())]
    for entry in entries:
        real = Path(os.path.realpath(entry or "."))
        path.append(
            str(update.absolute() / real.relative_to(held))
            if real.is_relative_to(held)
            else str(Path(entry or ".").absolute())
        )
    return path


def _exported(name: str) -> bool:
    """Whether the portable export writes a file of ``name`` beside the configs: its manifest, its program, a graph;
    in any case, a file system telling no case apart."""
    return name.lower() in ("manifest.json", "program.json") or name.lower().endswith(".onnx")


def _finish_update(spec: dict[str, Any]) -> None:
    """The portable export and the extension's check of a bundle's update (:func:`run_bundle_cli`), run in an
    interpreter of its own whose working folder is the update: the app's modules are the update's, none another
    process imported. Writes to ``spec["report"]`` the artifact exported, the classes checked (``None``: not checked,
    wasmtime missing) or the refusal."""
    from konfai.utils.errors import KonfAIError

    from .extension import check_extension, classes_run

    update = Path.cwd()
    report: dict[str, Any] = {}
    try:
        runs = None
        if spec["onnx"]:
            artifact = export_portable_into_bundle(
                update,
                checkpoints=spec["checkpoints"],
                patch_size=spec["patch_size"],
                in_channels=spec["in_channels"],
                output_module=spec["output_module"],
                extension=spec["extension"] is not None,
            )
            report["artifact"] = artifact.relative_to(update).as_posix()
            if spec["extension"]:
                # The classes the program runs by the extension: a class folded into a graph is no longer its.
                # A program, or the manifest beside the one model it would run.
                written = artifact if artifact.suffix == ".json" else update / "manifest.json"
                runs = classes_run(json.loads(written.read_text()))
        if spec["extension"]:
            # Every config KonfAI Edge runs that the bundle holds once updated, passed or kept, runs with the
            # extension: the compiled program's classes of the Prediction.yml, every class of the others. A config
            # is told by its root: a Predictor's or an Evaluator's (Evaluation.yml, Uncertainty.yml) is run, a
            # Trainer's (Config.yml) never, nor a file that reads as no YAML. The bytes checked are the ones the
            # bundle takes.
            compiled = [update / "Prediction.yml"] if runs is not None else []
            others = [
                config
                for config in sorted([*update.glob("*.yml"), *update.glob("*.yaml")])
                if config not in compiled and _run_by_konfai_edge(config)
            ]
            extension = update / spec["extension"]
            try:
                report["checked"] = check_extension(extension, compiled, runs) + check_extension(extension, others)
            except ModuleNotFoundError as error:
                if error.name != "wasmtime":
                    raise
                report["checked"] = None
    except KonfAIError as error:
        report["refused"] = [str(part) for part in error.args]
        raise SystemExit(1) from error
    finally:
        Path(spec["report"]).write_text(json.dumps(report))


def _run_by_konfai_edge(config: Path) -> bool:
    """Whether ``config`` is one KonfAI Edge runs: a file, a Predictor's or an Evaluator's, told by its root."""
    import yaml

    if not config.is_file():
        return False
    try:
        loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError):  # no text, or no YAML: nothing KonfAI Edge reads
        return False
    return isinstance(loaded, dict) and ("Predictor" in loaded or "Evaluator" in loaded)


def _carried(bundle: Path, source: str, target: str) -> None:
    """A file of ``bundle`` into the update an export finishes: a link for a checkpoint, which the export and the
    check only read, a copy for any other file, which the app's code they run may write in place, or a copy where
    the system makes no link."""
    from konfai_apps.app_repository import WEIGHTS

    if Path(source).name.endswith(WEIGHTS):
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


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

    from konfai_apps.app import KonfAIApp

    # Its graph lands beside the configs as <group>.onnx: a group naming a path would land elsewhere.
    if group_name in ("", ".", "..") or Path(group_name).name != group_name:
        raise AppMetadataError(f"nested model group '{group_name}' is no one name, which its graph is written under.")
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
        shutil.copy(artifact, owned_destination(out_dir / f"{group_name}.onnx"))
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
    extension: bool = False,
) -> Path:
    """Write the bundle's portable-runtime artifacts from its prediction config.

    One checkpoint, no test-time copy and no mask -> ``model.onnx`` + ``manifest.json``. Otherwise one
    ``<id>.onnx`` per checkpoint plus ``program.json``: the program :func:`konfai.portable.compile_output`
    assembles, every checkpoint on every test-time pass, the nested models a ``Mask`` reads, folded by the
    reduction the config declares. The patch geometry, transforms, blend, and reduction are all read from
    the config: nothing is per-app. Reading a config mutates it, so the export reads a private copy of it.

    ``extension``: the bundle carries the app's extension of KonfAI Edge, so the classes of the app's own compile
    to the steps the extension runs. These artifacts name no extension: the program
    :func:`konfai.portable.compile_prediction` writes does.
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
    with config_environment(config_path, root=root, state="PREDICTION"):
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
        if len(set(ids)) < len(ids):
            raise AppMetadataError(
                f"Two checkpoints of {selected} share a file stem, the id their exported models are written under. "
                "Rename them (fold0_best.pt, fold1_best.pt) before exporting."
            )
        # The nested models a Mask reads, exported first: a chain the program cannot run is refused before
        # the bundle's own models are.
        aux = {
            name: {**export_nested_model(name, group["inference"], bundle), "ops": group["ops"]}
            for name, group in aux_mask_groups(config, root).items()
        }
        passes = tta_passes(config, root, extension)
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
            extension=extension,
        )
    portable_assets = set(written) | {bundle / f"{name}.onnx" for name in aux}
    if not ensemble:
        portable_assets.add(bundle / "manifest.json")
    # One model on the identity pass, masked by nothing, is its manifest: konfai's reduction of a single copy
    # hands it back. The test-time copies, a mask and a final chain reading groups of the case ride on a program.
    if not (ensemble or aux or len(passes) > 1) and len(program.get("steps", [])) <= 1:
        return _declare_portable_assets(bundle, written[-1], portable_assets)
    program_path = bundle / "program.json"
    owned_destination(program_path).write_text(json.dumps(program, indent=2))
    portable_assets.add(program_path)
    return _declare_portable_assets(bundle, program_path, portable_assets)


def _bundle_checkpoint(bundle: Path, name: str) -> Path:
    """The bundle's checkpoint ``name``, found as konfai-apps finds it: as spelled, else as published (a ``.pt``
    is a ``.safetensors`` in a bundle), else with a weight suffix added, ``.safetensors`` before ``.pt`` (an older
    bundle's)."""
    from konfai_apps.app_repository import WEIGHTS

    path = bundle / Path(name).name
    if path.exists():
        return path
    if path.name.endswith(WEIGHTS):
        return bundle / _published(path.name)
    candidates = [path.with_name(path.name + suffix) for suffix in WEIGHTS]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


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
    owned_destination(metadata_path).write_text(json.dumps(metadata, indent=2))
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
    out = Path(args["out"]) / args["name"]
    out = out.resolve() if out.is_symlink() else out  # as assemble_bundle reads it
    report: dict[str, Any] = {}
    named = Path(args["extension"]).name if args.get("extension") else None
    if named is not None and (named.lower() == "requirements.txt" or _exported(named)):
        raise AppMetadataError(
            f"The extension {named} takes the name of a file the bundle command writes.",
            "Name the extension apart from the requirements, the manifest, the program and the graphs.",
        )
    # The bundle the update makes, where it holds a Model.py and no requirements.txt (passed, a support file, or the
    # author's it keeps), gets one drafted from its imports, less its own modules. Packaging deliberately does not
    # execute code to discover dependencies: the draft is reviewed.
    planned = {name.lower() for name in resolve_support_files(support, args.get("support_root"))}
    left = out.with_name(f".{out.name}.previous")  # the bundle an earlier update left aside, which it takes back
    held = [folder for folder in (out, left) if folder.is_dir()][:1]
    drafting = not args.get("requirements") and "requirements.txt" not in planned
    drafting = drafting and bool(
        args.get("model_py") or "model.py" in planned or any((folder / "Model.py").is_file() for folder in held)
    )
    drafted: list[str] = []

    def draft(update: Path) -> None:
        if (update / "Model.py").is_file() and not os.path.lexists(update / "requirements.txt"):
            local = {path.stem for path in update.glob("*.py")} | {
                path.name for path in update.iterdir() if path.is_dir() and any(path.rglob("*.py"))
            }
            drafted[:] = derive_requirements([update / "Model.py"], local)
        if drafted:
            (update / "requirements.txt").write_text("\n".join(drafted) + "\n")

    def export_and_check(update: Path) -> None:
        """The portable program and the extension's check, on the bundle the update makes, in an interpreter of their
        own (:func:`_finish_update`): the bundle takes neither until both are done."""
        spec: dict[str, Any] = {
            "onnx": bool(args.get("onnx")),
            "checkpoints": [Path(c).name for c in args["checkpoint"]] if args.get("checkpoint") else None,
            "patch_size": args.get("patch_size"),
            "in_channels": args.get("in_channels"),
            "output_module": args.get("output_module"),
            "extension": Path(args["extension"]).name if args.get("extension") else None,
        }
        with tempfile.TemporaryDirectory() as scratch:
            told = Path(scratch) / "report.json"
            spec["report"] = str(told)
            # The app's modules are the update's (:func:`_import_path`); bytecode neither written into the update nor
            # read from it, but under a folder of its own.
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": str(Path(scratch) / "bytecode")}
            env["PYTHONPATH"] = os.pathsep.join(_import_path(sys.path, out, update))
            code = "import json, sys; from konfai_apps.bundle import _finish_update; _finish_update(json.loads(sys.argv[1]))"
            done = subprocess.run([sys.executable, "-c", code, json.dumps(spec)], cwd=update, env=env, check=False)  # nosec B603
            if told.is_file():
                report.update(json.loads(told.read_text()))
        if done.returncode:
            raise AppMetadataError(*report.get("refused", ["The update's export or check failed, as told above."]))
        # Each stage asked for told what it did: one that ends the interpreter early is no export, no check.
        if (spec["onnx"] and "artifact" not in report) or (spec["extension"] and "checked" not in report):
            raise AppMetadataError("The update's export or check did not finish: the bundle is as it was.")

    checking = bool(args.get("onnx") or args.get("extension"))

    def finish(update: Path) -> None:
        if drafting:
            draft(update)
        if checking:
            export_and_check(update)

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
        # The app's extension of KonfAI Edge rides in the bundle, once it holds what the classes do.
        extension=args.get("extension"),
        finish=finish if drafting or checking else None,
    )
    print(f"[KonfAI-Apps] Bundle assembled at {bundle}")
    if "artifact" in report:
        print(f"[KonfAI-Apps] Portable model exported: {bundle / report['artifact']}")
    if "checked" in report:
        names = ", ".join(dict.fromkeys(report["checked"] or [])) or "no class of the app"
        print(
            f"[KonfAI-Apps] Extension checked against: {names}"
            if report["checked"] is not None
            else "[KonfAI-Apps] Extension not checked: install konfai-apps[extension] to compare it with the classes."
        )
    if drafted:
        print(f"[KonfAI-Apps] Drafted requirements.txt (review!): {', '.join(drafted)}")

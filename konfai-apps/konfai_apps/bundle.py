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
:func:`export_onnx_into_bundle` also emits ``model.onnx`` + ``manifest.json`` for the
Python-free runtime.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Callable
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
    """Best-effort: the *extra* PyPI requirements imported by custom ``.py`` files.

    Returns third-party packages beyond the standard library and what konfai provides
    (``segmentation_models_pytorch`` kept, ``torch``/``numpy`` dropped). A draft to review.
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


def _find_inference_patch(node: Any) -> dict[str, Any] | None:
    """The inference ``Patch`` sub-dict (sliding-window geometry), skipping the model's ``ModelPatch``."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "ModelPatch":
                continue
            if key == "Patch" and isinstance(value, dict) and "patch_size" in value:
                return value
            found = _find_inference_patch(value)
            if found is not None:
                return found
    return None


def _derive_overlap(patch: dict[str, Any], patch_size: list[int]) -> list[int] | None:
    """The inference ``Patch.overlap`` as a per-kept-axis list: a scalar broadcasts; a full-rank list
    drops the same 2.5D singleton axes ``patch_size`` dropped."""
    raw = patch.get("overlap")
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        return [int(raw)] * len(patch_size)
    if isinstance(raw, list):
        vals = [int(v) for v in raw]
        if len(vals) == len(patch_size):
            return vals
        dims = [int(d) for d in patch.get("patch_size", [])]
        if len(vals) == len(dims):  # full-rank overlap incl. the singleton slice axis: keep dim>1 axes
            kept = [v for v, d in zip(vals, dims, strict=True) if d > 1]
            return kept or vals
    return None


def _derive_blend(config: dict[str, Any], root: str) -> str | None:
    """The output's ``patch_combine`` window (Gaussian/Cosinus/Mean) -> the manifest ``blend``."""

    def walk(node: Any) -> str | None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "patch_combine" and isinstance(value, str) and value not in ("None", "none", ""):
                    return value
                found = walk(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = walk(item)
                if found is not None:
                    return found
        return None

    return walk(config.get(root))


def _derive_onnx_params(config: dict[str, Any], root: str) -> tuple[list[int] | None, int | None, int, float | None]:
    """Best-effort ``(patch_size, in_channels, extend_slice, pad_value)`` from a prediction config.

    ``patch_size`` drops a singleton slice dim (2.5D); ``in_channels`` reads the model's
    ``nb_channel`` / ``in_channels`` / first ``channels``. ``None`` for anything not derivable.
    """

    patch_size: list[int] | None = None
    extend_slice = 0
    pad_value: float | None = None
    patch = _find_inference_patch(config)
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
        raw_pad = patch.get("pad_value")
        if isinstance(raw_pad, (int, float)) and not isinstance(raw_pad, bool):
            pad_value = float(raw_pad)

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
    return patch_size, in_channels, extend_slice, pad_value


# KonfAI transform name -> runtime op. A transform outside this curated map is refused by the export.
def _op_cast(p: dict[str, Any]) -> dict[str, Any]:
    return {"op": "cast", "dtype": str(p.get("dtype", "float32"))}


def _op_resample(p: dict[str, Any]) -> dict[str, Any]:
    return {"op": "resample", "spacing": [float(s) for s in p["spacing"]], "inverse": bool(p.get("inverse", False))}


def _clean(value: Any) -> Any:
    return None if value in (None, "None") else value


def _op_standardize(p: dict[str, Any]) -> dict[str, Any]:
    step: dict[str, Any] = {"op": "standardize"}
    if _clean(p.get("mean")) is not None:
        step["mean"] = float(p["mean"]) if not isinstance(p["mean"], list) else p["mean"]
    if _clean(p.get("std")) is not None:
        step["std"] = float(p["std"]) if not isinstance(p["std"], list) else p["std"]
    return step


def _op_normalize(p: dict[str, Any]) -> dict[str, Any]:
    return {"op": "normalize", "min_value": float(p.get("min_value", -1)), "max_value": float(p.get("max_value", 1))}


def _op_unnormalize(p: dict[str, Any]) -> dict[str, Any]:
    return {"op": "unnormalize", "min_value": float(p["min_value"]), "max_value": float(p["max_value"])}


def _op_clip(p: dict[str, Any]) -> dict[str, Any]:
    # Bounds are a fixed number or a data-dependent spec ("min"/"max"/"percentile:<q>") the runtime resolves.
    def bound(v: Any) -> Any:
        return v if isinstance(v, str) else float(v)

    return {"op": "clip", "min_value": bound(p.get("min_value", -1024)), "max_value": bound(p.get("max_value", 1024))}


_OP_MAP = {
    "TensorCast": _op_cast,
    "Clip": _op_clip,
    "ResampleToResolution": _op_resample,
    # Canonical reorients from the volume's own direction cosines at runtime; the manifest op is just the inverse flag.
    "Canonical": lambda p: {"op": "canonical", "inverse": bool(p.get("inverse", True))},
    "Standardize": _op_standardize,
    "Normalize": _op_normalize,
    "UnNormalize": _op_unnormalize,
    "Softmax": lambda p: {"op": "softmax", "dim": int(p.get("dim", 0))},
    "Argmax": lambda p: {"op": "argmax", "dim": int(p.get("dim", 0))},
}


def _find_transforms(node: Any, key: str) -> dict[str, Any] | None:
    """First ``key`` sub-dict of a mapping value (the input ``transforms`` / output ``final_transforms``)."""
    if isinstance(node, dict):
        found = node.get(key)
        if isinstance(found, dict):
            return found
        for value in node.values():
            nested = _find_transforms(value, key)
            if nested is not None:
                return nested
    elif isinstance(node, list):
        for item in node:
            nested = _find_transforms(item, key)
            if nested is not None:
                return nested
    return None


def _try_fold(name: str, params: Any) -> Callable[[Any], Any] | None:
    """A tensor->tensor callable for the exporter's ``fold_pre`` if ``name`` is a POINTWISE,
    torch-instantiable transform (bakeable into the ONNX graph), else ``None``."""
    from konfai.data.transform import LocalityKind, Transform
    from konfai.utils.dataset import Attribute
    from konfai.utils.utils import get_module

    base = name.split("/", 1)[0]  # drop the ``/N`` uniqueness suffix a repeated transform carries
    try:
        module, cls_name = get_module(base, "konfai.data.transform")
        cls = getattr(module, cls_name)
    except Exception:
        return None
    if not (isinstance(cls, type) and issubclass(cls, Transform)):
        return None
    kwargs = {k: _clean(v) for k, v in params.items()} if isinstance(params, dict) else {}
    try:
        inst = cls(**kwargs)
        if inst.patch_locality(Attribute()).kind is not LocalityKind.POINTWISE:
            return None
    except Exception:
        return None
    return lambda tensor: inst("fold", tensor, Attribute())


def _pipeline(
    transforms: dict[str, Any] | None, *, fold: bool = False
) -> tuple[list[dict[str, Any]], list[Callable[[Any], Any]]]:
    """Map an ordered KonfAI transform chain to runtime ops. With ``fold=True``, a POINTWISE + torch
    transform not in the registry is baked into the ONNX graph. Folds must form a SUFFIX: a runtime
    op after a fold is refused."""
    steps: list[dict[str, Any]] = []
    folds: list[Callable[[Any], Any]] = []
    for name, params in (transforms or {}).items():
        base = name.split("/", 1)[0]
        if _clean(params) is None and base not in _OP_MAP:
            continue
        if base in _OP_MAP:
            if folds:
                raise AppMetadataError(
                    f"transform '{name}' is a runtime op but follows a folded transform; move folded "
                    "(pointwise custom) transforms to the end of the inference pipeline."
                )
            steps.append(_OP_MAP[base](params if isinstance(params, dict) else {}))
            continue
        folded = _try_fold(name, params) if fold else None
        if folded is None:
            raise AppMetadataError(
                f"transform '{name}' has no portable runtime op and is not a foldable pointwise transform; "
                "the ONNX bundle is not deployable in a Python-free runtime. Remove it from the inference "
                "config or extend the runtime op registry."
            )
        folds.append(folded)
    return steps, folds


def _transform_manifest(config: dict[str, Any], root: str) -> tuple[dict[str, Any], list[Callable[[Any], Any]]]:
    """The pre/post op pipeline the portable runtime applies around the tiled forward, plus the
    ``fold_pre`` callables (POINTWISE preprocessing transforms baked into the ONNX graph)."""
    section = config.get(root, {})
    pre, folds = _pipeline(_find_transforms(section.get("Dataset"), "transforms"), fold=True)
    # Post is `before_reduction_transforms` (disjoint ensemble: argmax per fold before merge) or
    # `final_transforms` (same-class ensemble: runs once after the mean); read both.
    before, _ = _pipeline(_find_transforms(section.get("outputs_dataset"), "before_reduction_transforms"))
    final, _ = _pipeline(_find_transforms(section.get("outputs_dataset"), "final_transforms"))
    return {"preprocessing": pre, "postprocessing": before + final}, folds


def _hoist_ensemble_tail(
    manifests: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hoist the post-reduction tail out of a same-class (``mean``) ensemble's per-fold manifests.

    A mean ensemble reduces RAW LOGITS, then applies softmax/argmax and the inverse resample once. So
    each fold is stripped of its ``postprocessing`` and its resample made forward-only; the removed
    steps become program ops after the ``mean`` (softmax/argmax, then a resample back onto ``input`` --
    nearest for a label map, linear otherwise).

    Returns ``(stripped_manifests, tail_ops)``.
    """
    rep = manifests[0]
    tail: list[dict[str, Any]] = []
    for step in rep.get("postprocessing", []):
        op = step["op"]
        if op == "cast":
            continue  # a dtype cast is a no-op in the float program; the final NIfTI write handles dtype
        if op not in ("softmax", "argmax"):
            raise AppMetadataError(
                f"assemble_program: cannot hoist post op '{op}' across the reduction (only softmax/argmax "
                "are channel-axis buffer ops; a per-voxel intensity op belongs in each fold's manifest)."
            )
        if int(step.get("dim", 0)) != 0:
            raise AppMetadataError(
                f"assemble_program: ensemble post op '{op}' has dim={step['dim']}; the program reduction "
                "runs on the channel axis (dim 0), so it cannot be hoisted across the reduction."
            )
        tail.append({"op": op})
    has_argmax = any(s["op"] == "argmax" for s in rep.get("postprocessing", []))
    if any(s.get("op") == "resample" and s.get("inverse") for s in rep.get("preprocessing", [])):
        tail.append({"op": "resample_nearest" if has_argmax else "resample_linear", "extra": ["input"]})

    stripped: list[dict[str, Any]] = []
    for mf in manifests:
        fold = dict(mf)
        fold["postprocessing"] = []
        fold["preprocessing"] = [
            {**step, "inverse": False} if step.get("op") == "resample" else step for step in mf.get("preprocessing", [])
        ]
        stripped.append(fold)
    return stripped, tail


def assemble_program(models: list[dict[str, Any]], *, reduce: str, classes: list[int] | None = None) -> dict[str, Any]:
    """Assemble the multi-model program JSON (Model steps + a reduction Op over named buffers).

    Each ``models[i]`` is ``{"id": str, "manifest": <single-model manifest>}``; every model runs on
    ``input`` into its own buffer, then a reduction combines them. ``mean`` (same-class ensemble)
    hoists each fold's post nonlinearity + inverse resample past the reduction (see
    :func:`_hoist_ensemble_tail`); ``merge_labels`` (disjoint ensemble) keeps each fold's post. A single
    model is one step, no reduction.
    """
    if not models:
        raise AppMetadataError("assemble_program: at least one model is required")
    if reduce not in ("mean", "merge_labels"):
        raise AppMetadataError(f"assemble_program: unknown reduction '{reduce}' (expected 'mean' or 'merge_labels')")

    manifests = [m["manifest"] for m in models]
    tail: list[dict[str, Any]] = []
    if reduce == "mean" and len(models) > 1:
        manifests, tail = _hoist_ensemble_tail(manifests)

    buffers = [f"m{i}" for i in range(len(models))]
    steps: list[dict[str, Any]] = [
        {"model": m["id"], "in": "input", "out": buf, "manifest": mf}
        for m, buf, mf in zip(models, buffers, manifests, strict=True)
    ]
    if len(models) == 1:
        return {"steps": [{**steps[0], "out": "output"}], "output": "output"}

    reduced = "r" if tail else "output"
    op: dict[str, Any] = {"op": reduce, "in": buffers, "out": reduced}
    if reduce == "merge_labels":
        op["classes"] = classes if classes is not None else [int(m["classes"]) for m in models]
    chain: list[dict[str, Any]] = [*steps, op]

    cursor = reduced
    for i, t in enumerate(tail):
        out = "output" if i == len(tail) - 1 else f"t{i}"
        chain.append({"op": t["op"], "in": [cursor, *t.get("extra", [])], "out": out})
        cursor = out
    return {"steps": chain, "output": "output"}


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


def _derive_reduction(config: dict[str, Any], root: str) -> str | None:
    """The ensemble reduction a multi-model program applies, read from the output transforms:
    ``MergeLabels`` (models with disjoint label spaces) -> ``merge_labels``; ``InferenceStack``
    (same-class probability ensemble) -> ``mean``. ``None`` when the config declares no reduction."""
    after = _find_transforms(config.get(root, {}).get("outputs_dataset"), "after_reduction_transforms") or {}
    names = {name.split("/", 1)[0] for name in after}
    if "MergeLabels" in names:
        return "merge_labels"
    if "InferenceStack" in names:
        return "mean"
    return None


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

    One checkpoint -> ``model.onnx`` + ``manifest.json``. Several checkpoints combined by the ensemble
    reduction the config declares (``MergeLabels`` / ``InferenceStack``) -> one ``<fold>.onnx`` per
    checkpoint plus ``program.json``, the multi-model dataflow the konfai-rs runtime executes. The patch
    geometry, transforms, blend, and reduction are all read from the config -- nothing is per-app. The
    config file is restored afterwards because reading it mutates it.
    """
    import torch
    import yaml
    from konfai.export import export_to_onnx, select_inference_head
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

    derived_patch, derived_channels, extend_slice, pad_value = _derive_onnx_params(config, root)
    patch_size = patch_size or derived_patch
    in_channels = in_channels if in_channels is not None else derived_channels
    if not patch_size:
        raise AppMetadataError("could not derive patch_size from the config; pass --patch-size")

    inference_patch = _find_inference_patch(config)
    patch_overlap = _derive_overlap(inference_patch, patch_size) if inference_patch else None
    blend = _derive_blend(config, root)
    reduction = _derive_reduction(config, root)
    checkpoints = checkpoints or [None]
    ensemble = len(checkpoints) > 1
    if ensemble and reduction is None:
        raise AppMetadataError(
            "several checkpoints were given but the config declares no ensemble reduction "
            "(MergeLabels / InferenceStack in outputs_dataset); cannot assemble a multi-model program.",
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

        extra_manifest, fold_pre = _transform_manifest(config, root)
        if blend is not None:
            extra_manifest["blend"] = blend

        models: list[dict[str, Any]] = []
        onnx_path = bundle / "model.onnx"
        for checkpoint in checkpoints:
            model = ModelLoader(classpath).get_model(train=False)
            # Export the per-patch network; the runtime does the sliding-window tiling.
            model.patch = None
            model.eval()
            if checkpoint is not None:
                ckpt_path = Path(checkpoint)
                if not ckpt_path.is_absolute():
                    ckpt_path = bundle / ckpt_path.name
                model.load(safe_torch_load(ckpt_path, "cpu"), init=False)

            # The input channel count is intrinsic to the architecture; read it off the loaded model
            # when the config does not declare it.
            if not in_channels:
                in_channels = getattr(model, "in_channels", None)
            if not in_channels:
                raise AppMetadataError("could not derive in_channels from the config or model; pass --in-channels")
            example = torch.randn(1, in_channels, *patch_size)
            # Default to the terminal float head; the graph's last output is often an integer Argmax.
            head = output_module or select_inference_head(model, example)
            fold_id = Path(checkpoint).stem if ensemble else "model"
            onnx_path, manifest = export_to_onnx(
                model,
                bundle,
                example,
                head,
                patch_overlap=patch_overlap,
                extend_slice=extend_slice,
                pad_value=pad_value,
                extra_manifest=dict(extra_manifest),
                fold_pre=fold_pre,
                model_filename="model.onnx" if not ensemble else f"{fold_id}.onnx",
                write_manifest=not ensemble,
            )
            models.append({"id": fold_id, "manifest": manifest})

        if not ensemble:
            return onnx_path

        program = assemble_program(
            models,
            reduce=reduction,
            classes=[int(m["manifest"]["output"]["channels"]) for m in models],
        )
        program_path = bundle / "program.json"
        program_path.write_text(json.dumps(program, indent=2))
        return program_path
    finally:
        sys.path.remove(str(bundle))
        config_path.write_text(config_snapshot)
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
    """Single-checkpoint portable export (``model.onnx`` + ``manifest.json``); a thin wrapper over
    :func:`export_portable_into_bundle`, which also handles the ensemble (``program.json``) form."""
    return export_portable_into_bundle(
        bundle,
        checkpoints=[checkpoint] if checkpoint is not None else None,
        patch_size=patch_size,
        in_channels=in_channels,
        prediction_config=prediction_config,
        output_module=output_module,
        root=root,
    )


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
        artifact = export_portable_into_bundle(
            bundle,
            checkpoints=[Path(c).name for c in args["checkpoint"]] if args.get("checkpoint") else None,
            patch_size=args.get("patch_size"),
            in_channels=args.get("in_channels"),
            output_module=args.get("output_module"),
        )
        print(f"Portable model exported: {artifact}")

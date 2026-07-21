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

"""Extension-point guidance and external-dependency pre-flight for an LLM agent.

Two gaps this closes:

* KonfAI's extension model ("subclass a base, reference it by classpath") and especially the
  external-library classpath syntax (``lib.module:Class``) are not surfaced anywhere in the MCP
  tool surface -- the agent is biased toward copying/wrapping when it could often reference an
  installed library class directly. ``describe_extension_points`` makes the contract explicit.
* There is no way to vet an external dependency before integrating a brick. ``check_external_dependency``
  reports importability/version/license without importing the module into the server process
  (``find_spec``/``importlib.metadata`` only), so external import side-effects never run here.
"""

from __future__ import annotations

import importlib.metadata as _metadata
import importlib.util as _util
import re
from email.message import Message
from pathlib import Path
from typing import Any, cast

# How a component name/classpath is written in a KonfAI YAML config.
YAML_REFERENCE_SYNTAX = {
    "builtin_name": "Bare class name resolved inside KonfAI's package for that kind, e.g. `Dice`, `Flip`.",
    "local_file": "`File:Class` -> a class in a local `File.py` written into the session workspace, e.g. `Loss:MyDiceFocal` (write it with write_session_file).",
    "external_library": "`package.module:Class` -> a class imported directly from an installed library, e.g. `monai.losses:DiceLoss`, `torch.nn:L1Loss`, `segmentation_models_pytorch:Unet`. Works for any installed library via get_module(); no wrapper needed when the class's call/forward convention already matches the KonfAI base.",
}

# Curated, code-grounded extension contract per kind (see konfai/metric/measure.py, konfai/network/network.py,
# konfai/data/transform.py, konfai/data/augmentation.py, konfai/metric/schedulers.py, AGENTS.md section 7c).
EXTENSION_POINTS: dict[str, dict[str, Any]] = {
    "loss": {
        "base_class": "konfai.metric.measure:Criterion",
        "required_methods": ["forward(self, output, *targets) -> Tensor"],
        "return_contract": "A loss returns a Tensor. (Metrics return a (value, dict) tuple; consumers isinstance-branch.)",
        "config_location": "Trainer.Model.outputs_criterions.<output_module_path>.targets_criterions.<target_group>.criterions_loader.<Name>",
        "direct_external": "A library loss with forward(input, target) -> Tensor (e.g. torch.nn:L1Loss, most monai.losses) can be referenced directly by classpath.",
        "local_wrapper": "When the signature/return differ or you need masking, subclass Criterion and adapt in forward.",
        "gotcha": "MaskedLoss-style masking is NOT applied to a directly-referenced external loss; wrap it if you need masking. Inspect forward with inspect_object_signature before deciding.",
        "list_tool": "list_components('criterion')",
    },
    "metric": {
        "base_class": "konfai.metric.measure:Criterion",
        "required_methods": ["forward(self, output, *targets) -> tuple[value, dict]"],
        "return_contract": "Return the (value, dict) metric tuple, not a bare Tensor.",
        "config_location": "Evaluator.metrics.<target_group>.targets_criterions.<group>.criterions_loader.<Name> (also usable as a monitored loss).",
        "direct_external": "Possible if the external metric already returns the (value, dict) shape; usually needs a thin wrapper.",
        "local_wrapper": "Subclass Criterion; return the metric tuple.",
        "gotcha": "inspect_object_signature reports the forward signature -- use it to confirm Tensor vs tuple return before wiring.",
        "list_tool": "list_components('criterion')",
    },
    "model": {
        "base_class": "konfai.network.network:Network",
        "required_methods": [
            "__init__ that builds the graph via add_module(name, module, in_branch=[...], out_branch=[...], alias=[...])"
        ],
        "return_contract": "A routed ModuleArgsDict graph; named module outputs become the dotted keys used by outputs_criterions.",
        "config_location": "Trainer.Model.classpath = `segmentation.UNet.UNet` (built-in) | `Model:MyNet` (local) | `segmentation_models_pytorch:Unet` / `torchvision.models:resnet50` (external) | `UNet.yml` (declarative).",
        "direct_external": "A non-Network external nn.Module is wrapped in MinimalModel automatically -- usable as a black-box model.",
        "local_wrapper": "Subclass Network and re-add_module the external submodules (see examples/Synthesis/Model.py wrapping segmentation_models_pytorch's UnetPlusPlus).",
        "gotcha": "MinimalModel gives the external network a SINGLE 'Model' child with NO add_module graph, so outputs_criterions dotted keys (deep supervision / intermediate-feature / perceptual / multi-head) can only target the top-level output. To wire losses to internal features you MUST subclass Network and re-add_module.",
        "list_tool": "list_components('model')",
    },
    "augmentation": {
        "base_class": "konfai.data.augmentation:DataAugmentation",
        "required_methods": [
            "_state_init(self, index, shapes, caches_attribute) -> list[int]  # sample params per case, cache by index",
            "_compute(self, name, index, tensors) -> list[tensors]  # apply lazily",
        ],
        "return_contract": "One output tensor per input; same spatial shape (only Mask/Permute may change shape).",
        "config_location": "Trainer.Dataset.augmentations.DataAugmentation_*.data_augmentations.<Name>",
        "direct_external": "External aug libs (albumentations/torchvision.transforms) do not match the index-keyed lazy contract -- almost always needs a wrapper.",
        "local_wrapper": "Subclass DataAugmentation; sample params in _state_init keyed by index so all patches of a case stay consistent; apply in _compute.",
        "gotcha": "An external aug that randomizes per call breaks patch consistency silently. Keep per-case params in _state_init.",
        "list_tool": "list_components('augmentation')",
    },
    "transform": {
        "base_class": "konfai.data.transform:Transform",
        "required_methods": [
            "__call__(self, name, tensor, cache_attribute) -> Tensor",
            "transform_shape(self, group_src, name, shape, cache_attribute) -> list[int]  # MUST be exact",
        ],
        "return_contract": "transform_shape must return the EXACT output spatial shape; pair inverse() if apply_inverse.",
        "config_location": "Trainer.Dataset.groups_src.<src>.groups_dest.<dest>.transforms.<Name> (or patch_transforms).",
        "direct_external": "An external image op rarely satisfies transform_shape -- wrap it.",
        "local_wrapper": "Subclass Transform; implement __call__ AND transform_shape exactly.",
        "gotcha": "Patch planning pre-computes slicing from transform_shape BEFORE data load. A wrong shape silently corrupts reassembly (no crash). A forward smoke test alone won't catch it -- run run_component_smoke_test, which asserts transform_shape(shape) == __call__(...).shape for you.",
        "verify_tool": "run_component_smoke_test(classpath, kind='transform') -- executes the transform_shape/__call__ contract on dummy tensors in an isolated spawn subprocess.",
        "list_tool": "list_components('transform')",
    },
    "scheduler": {
        "base_class": "konfai.metric.schedulers:Scheduler",
        "required_methods": ["get_value(self) -> float"],
        "return_contract": "Scalar weight for the current iteration (self.it updated by step()).",
        "config_location": "Trainer.Model.outputs_criterions.<...>.criterions_loader.<Criterion>.schedulers.<Name> (bare name only).",
        "direct_external": "Not supported -- the ':' import path is NOT honored for weight schedulers.",
        "local_wrapper": "Not extensible without a core edit: only classes defined inside konfai.metric.schedulers resolve (a workspace-local subclass is unreachable). Pick a built-in from list_components('scheduler'). (LR schedulers are separate via LRSchedulersLoader using torch names.)",
        "gotcha": "Weight schedulers resolve by bare name inside konfai.metric.schedulers only.",
        "list_tool": "list_components('scheduler')",
    },
    "pretrained": {
        "base_class": "(none -- ad hoc, inside a Network subclass)",
        "required_methods": [
            "Load weights in the model __init__ (torch.load / torch.hub) and apply via load_state_dict"
        ],
        "return_contract": "A Network whose submodules are initialised from external weights.",
        "config_location": "No first-class config hook; the loading lives in the model's Python code.",
        "direct_external": "No fetch/cache/verify path; HF/timm/torchvision weights are downloaded out of band or pre-saved as a .pt in the workspace.",
        "local_wrapper": "Subclass Network, build the graph with add_module, then load_state_dict with positional alias remapping (Network.load_state_dict does not recurse into nested Networks).",
        "gotcha": "No provenance/version capture for fetched weights -- record the source/version yourself for reproducibility.",
        "list_tool": "list_components('model')",
    },
}

_KIND_ALIASES = {
    "losses": "loss",
    "criterion": "loss",
    "metrics": "metric",
    "models": "model",
    "network": "model",
    "networks": "model",
    "augmentations": "augmentation",
    "transforms": "transform",
    "schedulers": "scheduler",
    "pretrained_model": "pretrained",
}


def describe_extension_points(kind: str | None = None) -> dict[str, Any]:
    """Describe KonfAI's extension points: how to plug a new loss/metric/model/augmentation/transform/scheduler."""
    payload: dict[str, Any] = {
        "yaml_reference_syntax": YAML_REFERENCE_SYNTAX,
        "principle": (
            "Every extension point is 'subclass a base, reference it by classpath in YAML' -- no core edits. "
            "Prefer referencing an installed library class directly (external syntax) over copying code; "
            "write a local wrapper only when the call/forward convention does not match the KonfAI base."
        ),
        "next_actions": [
            "list_components",
            "check_external_dependency",
            "inspect_object_signature",
            "write_session_file",
            "run_component_smoke_test",
        ],
    }
    if kind is None:
        payload["extension_points"] = EXTENSION_POINTS
        payload["kinds"] = list(EXTENSION_POINTS)
        return payload
    canonical = _KIND_ALIASES.get(kind.strip().lower(), kind.strip().lower())
    if canonical not in EXTENSION_POINTS:
        raise ValueError(f"Unknown extension kind '{kind}'. Expected one of: {', '.join(EXTENSION_POINTS)}.")
    payload["kind"] = canonical
    payload["extension_point"] = EXTENSION_POINTS[canonical]
    return payload


def _normalize_distribution_name(name: str) -> str:
    """PEP 503 name normalization so ``itk_core`` / ``ITK.Core`` compare equal to ``itk-core``."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _select_distribution(top: str, distributions: list[str], full: str | None = None) -> str:
    """Pick the distribution that actually provides top-level import ``top``.

    ``packages_distributions()`` maps one import name to EVERY distribution that ships it, and a
    namespace package (``itk``, ``google``) spans several wheels -- taking the first would report a
    sibling wheel's version/license. Prefer a distribution whose normalized name matches the import
    name; otherwise pick the one whose files include the module's own ``origin`` path.

    ``full`` is the pointed module path (``google.cloud.storage`` for ``google.cloud.storage:Class``).
    A pure namespace root has no ``origin`` of its own, so when ``top`` resolves to nothing we fall back
    to the pointed submodule's origin. ``find_spec`` on a dotted name imports only the lightweight
    namespace parents, never the heavy target module, so the no-import contract still holds.
    """
    if not distributions:
        return top
    if len(distributions) == 1:
        return distributions[0]
    normalized = _normalize_distribution_name(top)
    for candidate in distributions:
        if _normalize_distribution_name(candidate) == normalized:
            return candidate
    try:
        origin = getattr(_util.find_spec(top), "origin", None)
    except (ImportError, ModuleNotFoundError, ValueError):
        origin = None
    if origin is None and full is not None and full != top:
        try:
            origin = getattr(_util.find_spec(full), "origin", None)
        except (ImportError, ModuleNotFoundError, ValueError):
            origin = None
    if origin:
        origin_path = Path(origin).resolve()
        for candidate in distributions:
            try:
                files = _metadata.distribution(candidate).files or []
            except _metadata.PackageNotFoundError:
                continue
            for entry in files:
                try:
                    if Path(entry.locate()).resolve() == origin_path:
                        return candidate
                except (OSError, ValueError):
                    continue
    return distributions[0]


def _konfai_required_distributions() -> set[str]:
    try:
        requirements = _metadata.requires("konfai") or []
    except _metadata.PackageNotFoundError:
        return set()
    names: set[str] = set()
    for requirement in requirements:
        name = re.split(r"[<>=!~;\[ (]", requirement, maxsplit=1)[0].strip()
        if name:
            names.add(name.lower())
    return names


def check_external_dependency(module: str, object_name: str | None = None) -> dict[str, Any]:
    """Pre-flight an external dependency before integrating a brick: is it installed, which version/license?

    Uses importlib.util.find_spec + importlib.metadata only, so the external module is NEVER imported into
    the server process (no registry population / CUDA probe / network fetch side effects run here).
    """
    cleaned = module.strip()
    if not cleaned:
        raise ValueError("module must not be empty.")
    full = cleaned.split(":", 1)[0]
    top = full.split(".", 1)[0]

    try:
        installed = _util.find_spec(top) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        installed = False

    version = license_name = distribution = None
    if installed:
        distributions = _metadata.packages_distributions().get(top, [])
        distribution = _select_distribution(top, distributions, full)
        try:
            # metadata() returns an email.message.Message at runtime; the PackageMetadata stub omits .get.
            meta = cast(Message, _metadata.metadata(distribution))
            version = meta.get("Version")
            license_name = meta.get("License") or next(
                (
                    classifier.split("::")[-1].strip()
                    for classifier in (meta.get_all("Classifier") or [])
                    if classifier.startswith("License")
                ),
                None,
            )
        except _metadata.PackageNotFoundError:
            pass

    konfai_deps = _konfai_required_distributions()
    is_konfai_dependency = top.lower() in konfai_deps or (distribution or "").lower() in konfai_deps

    payload: dict[str, Any] = {
        "module": top,
        "installed": installed,
        "version": version,
        "license": license_name,
        "distribution": distribution,
        "is_konfai_dependency": is_konfai_dependency,
        "install_hint": None if installed else f"pip install {distribution or top}",
        "caution": (
            "Adding a new dependency increases fragility and may carry a restrictive license. Prefer libraries with "
            "permissive licenses (Apache-2.0/MIT/BSD), record the version + source for reproducibility, and prefer "
            "referencing the class by classpath over copying its code."
        ),
        "next_actions": (
            ["inspect_object_signature", "describe_extension_points"]
            if installed
            else ["describe_extension_points", "list_components"]
        ),
    }
    if object_name:
        payload["inspect_classpath"] = f"{cleaned}:{object_name}" if ":" not in cleaned else cleaned
    return payload

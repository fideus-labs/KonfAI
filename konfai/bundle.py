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

"""MONAI Bundles, both ways: a model zoo entry becomes a KonfAI prediction, a KonfAI checkpoint
becomes a bundle.

A `MONAI Bundle <https://docs.monai.io/en/stable/mb_specification.html>`_ is a directory:
``metadata.json``, ``configs/inference.json`` (the network under ``network_def`` as a ``_target_``
class and its arguments, the preprocessing, the inferer), ``models/model.pt`` (the network's plain
state dict) and, optionally, ``models/model.ts``. :func:`import_bundle` reads the network and its
weights into the ``Model`` block a ``Prediction.yml`` holds and a checkpoint in KonfAI's own
format; the bundle's preprocessing and inferer are reported, not translated, and the caller spells
the equivalent ``transforms:`` in KonfAI's own stages.

:func:`export_bundle` goes the other way: a KonfAI checkpoint's inference head traced to
``models/model.ts``, with a ``metadata.json`` and an ``inference.json`` that load that TorchScript
module.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from konfai.utils.errors import ConfigError


def _require_monai() -> Any:
    try:
        import monai.bundle
    except ImportError as error:
        raise ConfigError("MONAI Bundles need MONAI.", "pip install konfai[monai] (or: pip install monai)") from error
    return monai.bundle


@dataclass
class BundleImport:
    """What a bundle gave KonfAI: the ``Model`` block, the checkpoint, and what it left to the caller."""

    #: The bundle directory.
    root: Path
    #: The class the bundle instantiates, as a KonfAI classpath (``monai.networks.nets:UNet``).
    classpath: str
    #: The class's arguments, resolved (every ``@ref`` and ``$expr`` of the bundle's config folded).
    arguments: dict[str, Any]
    #: The KonfAI checkpoint written from the bundle's ``models/model.pt``.
    checkpoint: Path
    #: The bundle's ``metadata.json``.
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Bundle entries KonfAI does not translate (``preprocessing``, ``inferer``, ...), as configured.
    untranslated: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """The name the ``Model`` block binds the class's arguments under: the class's own."""
        return self.classpath.rsplit(":", 1)[-1]

    @property
    def spatial_dims(self) -> int | None:
        for key in ("spatial_dims", "dimensions", "dim"):
            if key in self.arguments:
                return int(self.arguments[key])
        return None

    def model_tree(self, *, patch: Sequence[int] | None = None) -> dict[str, Any]:
        """The ``Model`` block of a ``Prediction.yml`` (or ``Config.yml``) for this bundle's network:
        the classpath, the class's arguments under its name, and what the KonfAI wrapper adds."""
        block: dict[str, Any] = {
            **self.arguments,
            "outputs_criterions": None,
            "ModelPatch": None,
        }
        if self.spatial_dims is not None:
            block["dim"] = self.spatial_dims
        if "in_channels" not in block:
            block["in_channels"] = 1
        del patch  # a Patch block belongs to the Dataset, not the model; kept for the signature's future
        return {"classpath": self.classpath, self.name: block}


def _resolve_network(parser: Any, key: str) -> tuple[str, dict[str, Any]]:
    component = parser.get_parsed_content(key, instantiate=False)
    module_name = component.resolve_module_name()
    arguments = dict(component.resolve_args())
    if not isinstance(module_name, str) or "." not in module_name:
        raise ConfigError(
            f"The bundle's '{key}' is not a class path: {module_name!r}.",
            "KonfAI imports a bundle whose network_def names a class (`_target_: monai.networks.nets.UNet`).",
        )
    module, _, class_name = module_name.rpartition(".")
    return f"{module}:{class_name}", arguments


def import_bundle(
    bundle: Path | str,
    *,
    config: str = "configs/inference.json",
    network_key: str = "network_def",
    weights: str = "models/model.pt",
    out: Path | str | None = None,
) -> BundleImport:
    """Read a MONAI Bundle into what a KonfAI prediction needs.

    ``config`` is the bundle config holding the network (``configs/inference.json`` by
    convention); ``network_key`` its entry; ``weights`` the state dict to convert. The checkpoint
    is written to ``out`` (default: ``<bundle>/models/konfai_model.pt``) in the format
    ``Network.load`` reads for a wrapped foreign class. Nothing of the bundle is modified.
    """
    monai_bundle = _require_monai()
    root = Path(bundle)
    config_path = root / config
    if not config_path.is_file():
        raise ConfigError(
            f"'{config_path}' is not in the bundle.", "A MONAI Bundle keeps its inference config under configs/."
        )
    parser = monai_bundle.ConfigParser()
    parser.read_config(str(config_path))
    parser["bundle_root"] = str(root)
    classpath, arguments = _resolve_network(parser, network_key)
    class_name = classpath.rsplit(":", 1)[-1]

    weights_path = root / weights
    if not weights_path.is_file():
        raise ConfigError(f"'{weights_path}' is not in the bundle.", "A MONAI Bundle keeps its weights under models/.")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ConfigError(
            f"'{weights_path}' is not a state dict.", "KonfAI converts the plain state dict a bundle ships as model.pt."
        )
    # The wrapped class is loaded by the wrapper's name (the class's), each key under the module
    # the wrapper adds: what Network.load reads.
    checkpoint = Path(out) if out is not None else root / "models" / "konfai_model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"Model": {class_name: {f"Model.{key}": value for key, value in state.items()}}}, checkpoint)

    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    raw = parser.get()
    untranslated = {key: raw[key] for key in ("preprocessing", "postprocessing", "inferer") if key in raw}
    return BundleImport(root, classpath, arguments, checkpoint, metadata, untranslated)


def export_bundle(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    out: Path | str,
    *,
    name: str,
    output_module: str | None = None,
    description: str = "",
    version: str = "0.1.0",
    task: str = "",
    input_channels: int | None = None,
) -> Path:
    """Write a MONAI Bundle holding ``model``'s inference head as TorchScript.

    ``model`` is a loaded KonfAI network (``ModelLoader.get_model`` then ``load``), ``example_input``
    one patch of the shape it is fed (``[1, C, *patch]``); the head is the last floating-point
    output in execution order unless ``output_module`` names one. The bundle carries
    ``models/model.ts``, a ``metadata.json`` and a ``configs/inference.json`` that loads the traced
    module. Preprocessing is not exported; the traced module expects what the KonfAI chain fed the
    network, which the bundle's metadata states.
    """
    import monai
    import numpy

    from konfai.export import _NamedHead, select_inference_head

    root = Path(out)
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    model = model.eval()
    head = output_module or select_inference_head(model, example_input)
    wrapped = _NamedHead(model, head).eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example_input)
        output = wrapped(example_input)
    traced.save(str(root / "models" / "model.ts"))
    spatial = list(example_input.shape[2:])
    metadata = {
        "version": version,
        "changelog": {version: "exported from a KonfAI checkpoint"},
        "monai_version": monai.__version__,
        "pytorch_version": torch.__version__.split("+")[0],
        "numpy_version": numpy.__version__,
        "task": task,
        "description": description,
        "authors": "",
        "copyright": "",
        "network_data_format": {
            "inputs": {
                "image": {
                    "type": "image",
                    "format": "magnitude",
                    "num_channels": input_channels or int(example_input.shape[1]),
                    "spatial_shape": spatial,
                    "dtype": str(example_input.dtype).removeprefix("torch."),
                    "value_range": [],
                    "is_patch_data": True,
                    "channel_def": {},
                }
            },
            "outputs": {
                "pred": {
                    "type": "image",
                    "format": "segmentation" if output.shape[1] > 1 else "magnitude",
                    "num_channels": int(output.shape[1]),
                    "spatial_shape": list(output.shape[2:]),
                    "dtype": str(output.dtype).removeprefix("torch."),
                    "value_range": [],
                    "is_patch_data": True,
                    "channel_def": {},
                }
            },
        },
        "konfai": {"output_module": head, "name": name},
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    inference = {
        "imports": ["$import torch"],
        "bundle_root": ".",
        "device": "$torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')",
        "network": "$torch.jit.load(@bundle_root + '/models/model.ts').to(@device)",
        "inferer": {
            "_target_": "SlidingWindowInferer",
            "roi_size": spatial,
            "sw_batch_size": 1,
            "overlap": 0.25,
        },
    }
    (root / "configs" / "inference.json").write_text(json.dumps(inference, indent=2) + "\n", encoding="utf-8")
    return root

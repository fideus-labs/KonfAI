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

"""The EVALUATION half of the portable-runtime contract: an ``Evaluation.yml`` compiled into the program the
portable runtime runs, the groups each case is read into, the chains they run and the metrics scored between
them, keyed and written as :class:`konfai.evaluator.Statistics` writes them.

A metric outside :data:`CRITERIA` has no portable form and is refused by name. ``SAM_Perceptual`` carries its
network as an ONNX graph written beside the program, ``IMPACTReg`` the configuration of konfai-impact, the IMPACT
loss of the Rust engines, which bakes its networks; a ``KonfAIInference`` stage runs the nested app's model, which
``nested_exporter`` resolves (konfai-apps).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from konfai.portable import (
    _CHAIN_OPS,
    _admitted,
    _budget_bytes,
    _built,
    _destination,
    _file_stem,
    _literal_group,
    _or_konfai,
    _portable_root,
    _write_target,
    base_name,
    chain_items,
    clean,
    is_save,
    records_only,
    refuse_a_stage_konfai_does_not_carry,
    source_of,
    warn_unportable_keys,
)
from konfai.utils.errors import ConfigError, EvaluatorError

#: The evaluation program format the portable runtime parses. Raised when a field's meaning changes.
EVALUATION_PROGRAM_VERSION = 1

_EVALUATE = "konfai EVALUATION"

#: The ONNX opset the metric graphs are exported at, the one the prediction export uses.
_OPSET = 18


def _saved(criterion: Any, output_group: str) -> dict[str, Any]:
    """Where a ``*SaveMap`` writes its per-pixel map, as konfai's ``Evaluator._write_map`` names it: the dataset the
    criterion states, the group it states or the output group; nothing for a criterion that writes none."""
    from konfai.utils.utils import split_path_spec

    spec = getattr(criterion, "dataset", None)
    if not spec:
        return {}
    root, _flag, file_format = split_path_spec(str(spec), default_format="mha")
    return {"map": _destination(_portable_root(root), file_format, criterion.group or output_group)}


def _dice(criterion: Any, output_group: str, _out: Path) -> dict[str, Any]:
    labels = criterion._labels
    return {"metric": "dice", "labels": None if labels is None else [int(label) for label in labels]} | _saved(
        criterion, output_group
    )


def _mae(criterion: Any, output_group: str, _out: Path) -> dict[str, Any]:
    if criterion._reduction not in ("mean", "sum"):
        raise EvaluatorError(
            f"MAE with reduction '{criterion._reduction}' reports a map, not a value, and the portable runtime"
            " scores 'mean' and 'sum'.",
            _or_konfai("Use reduction: mean", _EVALUATE),
        )
    return {"metric": "mae", "reduction": criterion._reduction} | _saved(criterion, output_group)


def _psnr(criterion: Any, _output_group: str, _out: Path) -> dict[str, Any]:
    return {"metric": "psnr", "dynamic_range": float(criterion._dynamic_range)}


def _ssim(criterion: Any, _output_group: str, _out: Path) -> dict[str, Any]:
    return {"metric": "ssim", "dynamic_range": float(criterion._dynamic_range)}


def _sam_perceptual(criterion: Any, _output_group: str, out: Path) -> dict[str, Any]:
    import torch

    feature = criterion.model
    if not isinstance(criterion.loss, torch.nn.L1Loss) or feature.denormalize or feature.in_channels != 3:
        raise EvaluatorError(
            "SAM_Perceptual with another distance, a denormalized input or another channel count than konfai's own"
            " has no portable graph.",
            _or_konfai("Use SAM_Perceptual as konfai builds it", _EVALUATE),
        )
    tile = [int(size) for size in feature.shape]
    model = _file_stem(f"sam_perceptual_{Path(feature.model_path).stem}")
    layers = export_sam_perceptual(feature.model_path, tile, out / f"{model}.onnx")
    return {
        "metric": "sam_perceptual",
        "model": model,
        "tile": tile,
        "layers": layers,
        "weights": [float(w) for w in feature.weights],
    }


def _impact_reg(criterion: Any, _output_group: str, _out: Path) -> dict[str, Any]:
    """``IMPACTReg`` as konfai-impact's configuration: its model and the layers it weighs, compared by its loss's
    distance on each layer's own grid (mode Jacobian), the network handed no direction, as konfai runs it."""
    import torch

    from konfai.metric.measure.impact import MODELS_REPO

    feature = criterion.model
    distance = {torch.nn.MSELoss: "L2", torch.nn.L1Loss: "L1"}.get(type(criterion.loss))
    if (
        distance is None
        or criterion.loss.reduction != "mean"
        or criterion.pca
        or feature.shape is not None
        or feature.dim != 3
        or feature.in_channels != 1
        or feature.denormalize
    ):
        raise EvaluatorError(
            "IMPACTReg with a distance, a PCA, a tile, a 2-D or multi-channel network, or a loss other than an L1 or"
            " MSE mean has no portable form.",
            _or_konfai("Score the whole volume with loss torch:nn:MSELoss or torch:nn:L1Loss", _EVALUATE),
        )
    weights = [float(weight) for weight in feature.weights]
    model = {
        "ref": f"{MODELS_REPO}:{criterion.model_name}",
        "layers_mask": "".join("1" if weight else "0" for weight in weights).rstrip("0"),
        "layers_weight": [weight for weight in weights if weight],
        "distance": distance,
        "direction": False,
    }
    return {
        "metric": "impact_reg",
        "impact": {"models": {"0": model}, "mode": "Jacobian", "normalize": False},
        "activations": impact_reg_activations(feature.model_path, weights),
    }


def _mean(_criterion: Any, _output_group: str, _out: Path) -> dict[str, Any]:
    return {"metric": "mean"}


#: Criterion class -> the step it compiles to. A subclass compiles as the class it extends (``DiceSaveMap`` is a
#: ``Dice`` that writes its map), matched exactly so a criterion of the caller's own is refused, not approximated.
CRITERIA: dict[str, Callable[[Any, str, Path], dict[str, Any]]] = {
    "Dice": _dice,
    "DiceSaveMap": _dice,
    "MAE": _mae,
    "MAESaveMap": _mae,
    "PSNR": _psnr,
    "SSIM": _ssim,
    "SAM_Perceptual": _sam_perceptual,
    "IMPACTReg": _impact_reg,
    "Mean": _mean,
}

#: The folds over a stack's members (``konfai.data.transform.ensemble``) an uncertainty chain runs, by the op each
#: compiles to: one channel per member in, one out.
_MEMBER_FOLDS: dict[str, str] = {
    "Variance": "variance",
    "StandardDeviation": "standard_deviation",
    "SegmentationDisagreement": "segmentation_disagreement",
}


def _criterion(criterion: Any, key: str, output_group: str, out: Path) -> dict[str, Any]:
    """One criterion as the runtime scores it, under the key konfai records it by."""
    name = type(criterion).__name__
    if type(criterion).__module__.split(".")[0] != "konfai" or name not in CRITERIA:
        raise EvaluatorError(
            f"Criterion '{name}' has no portable form, so this evaluation cannot run outside Python.",
            f"Portable criteria are: {', '.join(sorted(CRITERIA))}. {_or_konfai('Score it', _EVALUATE)}",
        )
    return {"key": key, "maximize": bool(getattr(criterion, "maximize", False))} | CRITERIA[name](
        criterion, output_group, out
    )


def _group_steps(
    group_dest: str,
    chain: Any,
    out: Path,
    nested: dict[str, dict[str, Any]],
    nested_exporter: Callable[[str, dict[str, Any], Path], dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """A ``groups_dest`` chain as the steps the runtime runs on the case it reads: the chain ops, a ``statistics``
    where konfai's ``Statistics`` records the volume's on the case, a model step for a ``KonfAIInference``, the
    nested app's model exported once however many groups run it, and a ``save`` where a ``Save`` writes what
    reaches it (konfai reads that entry back on a later run in place of the stages before it; the runtime runs
    them again)."""
    steps: list[dict[str, Any]] = []
    for name, params in chain_items(chain):
        stage = base_name(name)
        if stage == "Statistics":
            steps.append({"op": "statistics"})
        elif stage == "KonfAIInference":
            steps.append(_inference(group_dest, params, out, nested, nested_exporter))
        elif is_save(name):
            steps.append(_admitted({"op": "save", "destination": _write_target(name, params, group_dest)}))
        elif stage in _MEMBER_FOLDS:
            if stage == "SegmentationDisagreement" and clean((params or {}).get("ignore_background")):
                raise EvaluatorError(
                    f"SegmentationDisagreement of group '{group_dest}' ignores the background, and the portable"
                    " runtime counts every label.",
                    _or_konfai("Drop ignore_background", _EVALUATE),
                )
            steps.append(_admitted({"op": _MEMBER_FOLDS[stage]}))
        elif not records_only(name):
            refuse_a_stage_konfai_does_not_carry(name, EvaluatorError)
            if stage not in _CHAIN_OPS:
                raise EvaluatorError(
                    f"Transform '{name}' of group '{group_dest}' has no portable runtime op.",
                    f"Portable stages are: {', '.join(sorted(_CHAIN_OPS))}. {_or_konfai('Run it', _EVALUATE)}",
                )
            steps.append(_admitted(_CHAIN_OPS[stage](params)))
    return steps


def _inference(
    group_dest: str,
    params: dict[str, Any],
    out: Path,
    nested: dict[str, dict[str, Any]],
    nested_exporter: Callable[[str, dict[str, Any], Path], dict[str, Any]] | None,
) -> dict[str, Any]:
    """A ``KonfAIInference`` as one model step: its app's model, exported on its first use, run on each channel
    as a case of its own when it runs ``per_channel``, the outputs stacked as channels."""
    if clean(params.get("number_of_tta")) or clean(params.get("config_overrides")):
        raise EvaluatorError(
            f"The KonfAIInference of group '{group_dest}' runs with test-time copies or with config overrides, and"
            " the portable runtime nests one plain model.",
            _or_konfai("Drop number_of_tta and config_overrides", _EVALUATE),
        )
    key = json.dumps({k: clean(v) for k, v in params.items()}, sort_keys=True)
    if key not in nested:
        if nested_exporter is None:
            raise ConfigError(
                f"Group '{group_dest}' is produced by a nested app, which this compiler does not resolve.",
                "Pass nested_exporter=konfai_apps.bundle.export_nested_model, which resolves an app reference.",
            )
        nested[key] = nested_exporter(group_dest, params, out)
    step = {"model": nested[key]["id"], "manifest": nested[key]["manifest"]}
    return step | ({"per_channel": True} if clean(params.get("per_channel")) else {})


def compile_evaluation(
    evaluation_file: str | Path = Path("./Evaluation.yml"),
    evaluations_dir: str | Path = Path("./Evaluations"),
    output: str | Path = Path("./evaluation.json"),
    nested_exporter: Callable[[str, dict[str, Any], Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compile an ``Evaluation.yml`` into the program the portable runtime executes, written to ``output`` with
    the ONNX of every network it runs beside it. The workflow is built first, so KonfAI's own refusals come
    before a program is written, and the criteria are read as it binds them."""
    from konfai.evaluator import Evaluator, build_evaluate
    from konfai.network.network.measure import criterion_keys

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    root = "Evaluator"
    with _built(
        evaluation_file, lambda file: build_evaluate(evaluations_file=file, evaluations_dir=Path(evaluations_dir))
    ) as (built, resolved):
        evaluator = cast(Evaluator, built)
        warn_unportable_keys(resolved, root)
        section = (resolved.get(root) or {}).get("Dataset") or {}
        if clean(section.get("validation")) is not None:
            raise EvaluatorError(
                "The evaluation splits its cases into TRAIN and VALIDATION, and the portable runtime scores one split.",
                _or_konfai("Set validation: None", _EVALUATE),
            )
        read = {
            group
            for output_group, targets in evaluator.metrics.items()
            for target_group in (output_group, *targets)
            for group in target_group.split(";")
        }
        groups: list[dict[str, Any]] = []
        nested: dict[str, dict[str, Any]] = {}
        for group_src, node in (section.get("groups_src") or {}).items():
            _literal_group(group_src, EvaluatorError)
            for group_dest, spec in ((node or {}).get("groups_dest") or {}).items():
                if group_dest in read:
                    steps = _group_steps(
                        group_dest, (spec or {}).get("transforms"), output.parent, nested, nested_exporter
                    )
                    groups.append({"group_src": group_src, "group_dest": group_dest, "steps": steps})
        dests = {group["group_dest"] for group in groups}
        metrics = []
        for output_group, targets in evaluator.metrics.items():
            for target_group, criteria in targets.items():
                keys = criterion_keys(output_group, target_group, (criterion.get_name() for criterion in criteria))
                metrics.append(
                    {
                        "output": output_group,
                        "targets": [group for group in target_group.split(";") if group in dests],
                        "criteria": [
                            _criterion(criterion, key, output_group, output.parent)
                            for criterion, key in zip(criteria, keys, strict=True)
                        ],
                    }
                )
        models = {model["id"]: f"{model['id']}.onnx" for model in nested.values()}
        models |= {
            criterion["model"]: f"{criterion['model']}.onnx"
            for metric in metrics
            for criterion in cast(list[dict[str, Any]], metric["criteria"])
            if "model" in criterion
        }
        program = {
            "konfai_edge_evaluation": EVALUATION_PROGRAM_VERSION,
            "name": evaluator.name,
            "memory_budget": _budget_bytes(section.get("memory_budget")),
            "sources": [source_of(entry) for entry in section.get("dataset_filenames") or []],
            "cases": list(evaluator.dataset.case_names),
            "evaluations": _portable_root(evaluator.metric_path),
            "groups": groups,
            "models": models,
            "metrics": metrics,
        }
    output.write_text(json.dumps(program, indent=2))
    return program


def compile_uncertainty(
    uncertainty_file: str | Path = Path("./Uncertainty.yml"),
    uncertainties_dir: str | Path = Path("./Uncertainties"),
    output: str | Path = Path("./uncertainty.json"),
    nested_exporter: Callable[[str, dict[str, Any], Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compile an app's ``Uncertainty.yml``, the EVALUATION konfai-apps runs on an inference stack (one channel per
    member), into the program the portable runtime executes: :func:`compile_evaluation` writing under
    ``Uncertainties``, as ``konfai-apps uncertainty`` does."""
    return compile_evaluation(uncertainty_file, uncertainties_dir, output, nested_exporter)


def _sam_trunk(script: Any, tile: list[int]) -> Any:
    """The Hiera trunk of a SAM2 TorchScript feature model as plain torch modules carrying its weights, each
    stage end's features weighted by the model's channel weights where it has them: the graph the TorchScript
    file computes, bit for bit on the CPU, without its control flow."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    trunk = script.model

    def linear(module: Any) -> nn.Linear:
        out_features, in_features = module.weight.shape
        layer = nn.Linear(in_features, out_features)
        layer.load_state_dict(module.state_dict())
        return layer

    def norm(module: Any) -> nn.LayerNorm:
        layer = nn.LayerNorm(module.normalized_shape, eps=module.eps)
        layer.load_state_dict(module.state_dict())
        return layer

    def partition(x: torch.Tensor, window: int) -> tuple[torch.Tensor, tuple[int, int]]:
        b, h, w, c = x.shape
        pad_h, pad_w = (window - h % window) % window, (window - w % window) % window
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        hp, wp = h + pad_h, w + pad_w
        x = x.view(b, hp // window, window, wp // window, window, c)
        return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window, window, c), (hp, wp)

    def unpartition(x: torch.Tensor, window: int, padded: tuple[int, int], hw: tuple[int, int]) -> torch.Tensor:
        (hp, wp), (h, w) = padded, hw
        x = x.view(-1, hp // window, wp // window, window, window, x.shape[-1])
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, hp, wp, x.shape[-1])
        return x[:, :h, :w, :]

    class Block(nn.Module):
        def __init__(self, scripted: Any) -> None:
            super().__init__()
            self.norm1, self.norm2 = norm(scripted.norm1), norm(scripted.norm2)
            self.qkv, self.attn_proj = linear(scripted.attn.qkv), linear(scripted.attn.proj)
            first, second = (module for _, module in scripted.mlp.layers.named_children())
            self.mlp = nn.Sequential(linear(first), nn.GELU(), linear(second))
            self.heads, self.window = int(scripted.attn.num_heads), int(scripted.window_size)
            self.pooled = scripted.attn.q_pool is not None
            self.stride = int(scripted.q_stride[0]) if self.pooled else 1
            self.proj = linear(scripted.proj) if scripted.dim != scripted.dim_out else None

        def pool(self, x: torch.Tensor) -> torch.Tensor:
            # The trunk's q_pool, a max_pool2d(stride, stride), of the channels-last map.
            return F.max_pool2d(x.permute(0, 3, 1, 2), self.stride, self.stride).permute(0, 2, 3, 1)

        def attention(self, x: torch.Tensor) -> torch.Tensor:
            b, h, w, _ = x.shape
            q, k, v = torch.unbind(self.qkv(x).reshape(b, h * w, 3, self.heads, -1), 2)
            if self.pooled:
                q = self.pool(q.reshape(b, h, w, -1))
                h, w = q.shape[1:3]
                q = q.reshape(b, h * w, self.heads, -1)
            x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            return self.attn_proj(x.transpose(1, 2).reshape(b, h, w, -1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            normed = self.norm1(x)
            shortcut = x if self.proj is None else self.pool(self.proj(normed)) if self.pooled else self.proj(normed)
            hw = tuple(normed.shape[1:3])
            if self.window > 0:
                y, padded = partition(normed, self.window)
                y = self.attention(y)
                window = self.window // self.stride
                hw = tuple(shortcut.shape[1:3])
                padded = tuple(side + (window - side % window) % window for side in hw)
                y = unpartition(y, window, padded, hw)
            else:
                y = self.attention(normed)
            x = shortcut + y
            return x + self.mlp(self.norm2(x))

    class Trunk(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            embed = trunk.patch_embed.proj
            self.embed = nn.Conv2d(
                embed.in_channels, embed.out_channels, embed.kernel_size, embed.stride, embed.padding
            )
            self.embed.load_state_dict(embed.state_dict())
            grid = [side // embed.stride[0] for side in tile]
            with torch.no_grad():
                self.register_buffer("pos", trunk._get_pos_embed(tuple(grid)).detach().clone())
            self.blocks = nn.ModuleList(Block(block) for _, block in trunk.blocks.named_children())
            self.ends = [int(end) for end in trunk.stage_ends]
            names = [f"channel_weights_{i}" for i in range(len(self.ends))]
            for i, name in enumerate(names):
                weights = getattr(script, name, None)
                self.register_buffer(f"weights_{i}", None if weights is None else weights.detach().clone())

        def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
            y = self.embed(x).permute(0, 2, 3, 1) + self.pos
            layers: list[torch.Tensor] = []
            for index, block in enumerate(self.blocks):
                y = block(y)
                if index in self.ends:
                    layer = y.permute(0, 3, 1, 2)
                    weights = getattr(self, f"weights_{len(layers)}")
                    layers.append(layer if weights is None else layer * weights.view(1, -1, 1, 1))
            return layers

    return Trunk().eval()


def sam_perceptual_graph(model_path: str | Path, tile: list[int]) -> Any:
    """The per-tile half of ``SAM_Perceptual`` as one torch module: the output's and the target's tiles, already
    min-max normalized by the target's statistics (``standardizeImageNet``'s first step, which the runtime takes
    in float32 as torch does), run through the trunk together, and each layer's L1 distance over the pixels the
    mask keeps at that layer's size (``_feature_mask``: nearest, an integer ratio). It returns, layer after layer,
    the distance summed along each row of each channel (``[C, H]``), then the kept pixels of each row (``[H]``):
    the runtime adds them in float64, where a float32 sum over a whole layer loses four digits (ONNX Runtime's
    ReduceSum, measured 4e-4 off the mean)."""
    import torch
    from torch import nn

    script = torch.jit.load(str(model_path), map_location="cpu").eval()  # nosec B614
    trunk = _sam_trunk(script, tile)
    mean, std = (script.standardizeImageNet.mean.detach().clone(), script.standardizeImageNet.std.detach().clone())

    class Graph(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = trunk

        def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            x = (torch.cat([output, target]) - mean) / std
            sums = []
            for layer in self.trunk(x):
                h, w = layer.shape[2:]
                cut = mask.reshape(1, 1, h, tile[0] // h, w, tile[1] // w)[:, :, :, :1, :, :1]
                keep = (cut.reshape(1, 1, h, w) != 0).float()
                sums += [((layer[:1] - layer[1:]).abs() * keep).sum(3).reshape(-1), keep.sum(3).reshape(-1)]
            return torch.cat(sums)

    return Graph().eval()


def export_sam_perceptual(model_path: str | Path, tile: list[int], path: Path) -> list[list[int]]:
    """Write :func:`sam_perceptual_graph` as a self-contained ONNX file at ``path``, and return ``[channels, rows]`` of
    each layer it sums, in order."""
    import onnx
    import torch

    graph = sam_perceptual_graph(model_path, tile)
    examples = tuple(torch.rand(1, 1, *tile) for _ in range(3))
    with torch.no_grad():
        layers = [[int(layer.shape[1]), int(layer.shape[2])] for layer in graph.trunk(torch.zeros(1, 3, *tile))]
        torch.onnx.export(
            graph,
            examples,
            str(path),
            opset_version=_OPSET,
            input_names=["output", "target", "mask"],
            output_names=["layers"],
            dynamo=True,
        )
    exported = onnx.load(str(path))
    for node in exported.graph.node:
        kept = [prop for prop in node.metadata_props if prop.key != "pkg.torch.onnx.stack_trace"]
        del node.metadata_props[:]
        node.metadata_props.extend(kept)
    onnx.save(exported, str(path), save_as_external_data=False)
    path.with_name(path.name + ".data").unlink(missing_ok=True)
    return layers


def impact_reg_activations(model_path: str | Path, weights: list[float]) -> float:
    """The float32 values ``IMPACTReg``'s network holds per pixel of the volume at its peak, which the runtime's plan
    prices: four of its widest layer, the weighted layers of the output it keeps while it runs the target, the two
    inputs. The network is a TotalSegmentator encoder normalized by fixed CT statistics, which konfai-impact runs on
    the volume as konfai does: a network normalized by the image's own statistics would read the ones konfai records
    in the case's attributes."""
    import torch

    script = torch.jit.load(str(model_path), map_location="cpu").eval()  # nosec B614
    kept = [index for index, weight in enumerate(weights) if weight]
    normalize = getattr(script, "normalize", None)
    buffers = {} if normalize is None else dict(normalize.named_buffers())
    names = [f"DownConvBlock_{index}" for index in range(max(kept, default=0) + 1)]
    if not kept or "mean" not in buffers or not all(hasattr(script, name) for name in names):
        raise EvaluatorError(
            f"IMPACTReg's portable form runs a TotalSegmentator encoder normalized by fixed CT statistics, and"
            f" '{model_path}' with the weights {weights} is not one.",
            _or_konfai("Use a TS/ model of VBoussot/impact-torchscript-models", _EVALUATE),
        )
    stride = [1, 1, 1]
    widest = held = 0.0
    for index, name in enumerate(names):
        block = getattr(script, name)
        for convolution in (block.Conv_0, block.Conv_1):
            stride = [a * int(b) for a, b in zip(stride, convolution.stride, strict=True)]
        per_pixel = int(block.Conv_1.weight.shape[0]) / math.prod(stride)
        widest = max(widest, per_pixel)
        if index in kept:
            held += per_pixel
    return 4 * widest + held + 2

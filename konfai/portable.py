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

"""The portable-runtime contract: which KonfAI stages a Python-free runtime can execute, and what
each one carries.

Two registries, read by both compilers, the PREDICTION one (:func:`compile_prediction`, which the
``konfai_apps`` bundle export goes through) and the TRANSFORM one (:func:`compile_transform`): ``OP_MAP``
holds the ops a buffer runs, ``_CHAIN_OPS`` adds the ones a TRANSFORM chain runs on its own, and
``PROGRAM_TRANSFORMS`` names the stages a prediction program builds a step for instead. A stage outside them
has no portable form, and is refused by name rather than approximated, because a chain the two engines read
differently is wrong data and not an error. Each compiler builds the workflow first, so every refusal KonfAI
itself makes (an unbound key, a chain with no ``Write``) comes before a program is emitted.
"""

from __future__ import annotations

import json
import random
import re
import warnings
from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from itertools import chain, combinations, islice, repeat, takewhile
from pathlib import Path
from typing import Any, cast

from ruamel.yaml import YAML

from konfai.utils.errors import ConfigError, KonfAIError, KonfAIWarning, TransformError

#: The program format the portable runtime parses. Raised when a field's meaning changes.
TRANSFORM_PROGRAM_VERSION = 4

#: The prediction program format the portable runtime parses. Raised when a field's meaning changes.
PREDICTION_PROGRAM_VERSION = 2


def clean(value: Any) -> Any:
    """A YAML ``None`` in either spelling, as ``None``."""
    return None if value in (None, "None") else value


def _mapping(value: Any) -> dict[str, Any]:
    """``value`` when it is a mapping, else ``{}``: a key written with nothing under it."""
    return value if isinstance(value, dict) else {}


#: How a refusal's hint names the run konfai itself makes of what the program cannot carry.
_RUN, _PREPARE, _PREDICT = "run this config", "prepare this dataset", "predict this config"


def _or_konfai(advice: str, run: str = _RUN) -> str:
    """A refusal's hint: ``advice``, or the same run made by konfai itself."""
    return f"{advice}, or {run} with konfai itself."


def _op_cast(p: dict[str, Any]) -> dict[str, Any]:
    import torch

    # konfai's TensorCast resolves the name through torch, so an alias ("float", "long") is spelled the
    # way the runtime names the dtype.
    dtype = getattr(torch, str(p.get("dtype", "float32")))
    return {"op": "cast", "dtype": str(dtype).removeprefix("torch.")}


def _op_resample(
    p: dict[str, Any], datasets: list[str] | None = None, roots: list[str] | None = None
) -> dict[str, Any]:
    # A spacing, a shape or a reference image is portable: the runtime derives the first two from the volume
    # in hand and reads the third's header. So is a displacement field, read beside the case, and a transform
    # stored beside it, looked up in the run's datasets (`roots`).
    field = _field(p, datasets)
    stored = _stored(p, roots)
    reference = clean(p.get("reference"))
    step: dict[str, Any]
    if reference is not None:
        # konfai places a reference by its own header and ignores align.
        step = {"op": "resample", "reference": _reference(p, str(reference), datasets)}
    else:
        spacing, shape = clean(p.get("spacing")), clean(p.get("shape"))
        # A map alone moves the case on its own grid (konfai's `_OwnGrid`), which ignores align.
        if (spacing is None) == (shape is None) and (spacing is not None or not (field or stored)):
            named = "both a spacing and a shape" if spacing is not None else "neither a spacing nor a shape"
            raise TransformError(
                f"'Resample' names {named}.",
                "A portable Resample names a 'spacing', a 'shape' or a 'reference', exactly one, or a map alone"
                " (a field or stored transforms).",
            )
        # konfai reads a shape in array order and a component <= 0 as its axis kept.
        step = {"op": "resample"}
        if spacing is not None:
            step["spacing"] = [float(s) for s in spacing]
        elif shape is not None:
            step["shape"] = [max(0, int(s)) for s in shape]
        # Written off konfai's default only, so a grid placed by extent compiles as it always has.
        align = str(clean(p.get("align")) or "extent")
        if align not in ("extent", "origin"):
            raise TransformError(
                f"'Resample' aligns on '{align}', which konfai does not place a grid by.",
                "Use align: extent or align: origin.",
            )
        if align != "extent" and (spacing is not None or shape is not None):
            step["align"] = align
    step.update(field)
    step.update(stored)
    # `edge` reads the source's edge pixel past it (MONAI's padding_mode: border), a number fills with it.
    fill = clean(p.get("fill"))
    if fill == "edge":
        step["fill"] = fill
    elif float(fill or 0.0):
        step["fill"] = float(fill)
    interpolation = clean(p.get("interpolation"))
    if interpolation is not None:
        if interpolation not in ("nearest", "linear", "cubic"):
            raise TransformError(
                f"'Resample' interpolates '{interpolation}', which has no portable runtime op.",
                "A portable Resample interpolates 'nearest', 'linear' or 'cubic'.",
            )
        step["interpolation"] = interpolation
    # Left unset, konfai picks from the dtype the volume arrives in: resolve_interpolation decides it.
    return step


def _field(p: dict[str, Any], datasets: list[str] | None) -> dict[str, Any]:
    """Where konfai reads a Resample's displacement field, the case's entry of its group
    (``_DisplacementSource``): ``field_group`` beside the case, or the group of the store ``field`` names, its
    only one when ``field_group`` names none, which only a TRANSFORM program looks up. ``{}`` for no field."""
    field, group = clean(p.get("field")), clean(p.get("field_group"))
    if field is not None and not str(field).strip():
        field = None
    if field is None:
        return {} if group is None else {"field_group": _literal_group(str(group), TransformError)}
    if datasets is None:
        raise TransformError(
            f"'Resample' reads its fields from the store '{field}', which only a TRANSFORM program looks up.",
            _or_konfai("Store the fields beside the cases and name their group alone (field_group)", _PREDICT),
        )
    from konfai.data.transform.resample import _DisplacementSource

    group = _DisplacementSource(str(field), group).group_for(None)
    return {"field_group": _literal_group(group, TransformError), "field_dataset": source_of(str(field))}


def _stored(p: dict[str, Any], roots: list[str] | None) -> dict[str, Any]:
    """Where konfai reads a Resample's stored transforms (``Resample._decode_stored``): each group, in the order the
    stage declares it with whether to invert it, looked up in the run's ``dataset_filenames`` in order, which is
    what both workflows hand their stages. ``{}`` for none."""
    transforms = clean(p.get("transforms"))
    if transforms is None:
        return {}
    if roots is None:
        raise TransformError(
            "'Resample' reads stored transforms, which the program looks up in the run's datasets, and this chain"
            " names none.",
            _or_konfai("Apply them in the model's own input chain", _PREDICT),
        )
    groups = [
        {"group": _literal_group(str(group), TransformError), "invert": bool(invert)}
        for group, invert in dict(transforms).items()
    ]
    return {"transforms": {"groups": groups, "datasets": [source_of(str(root)) for root in roots]}}


def _reference(p: dict[str, Any], entry: str, datasets: list[str] | None) -> dict[str, Any]:
    """Where konfai looks a Resample's reference image up: ``reference_dataset``, else the run's roots in order,
    each with the group holding the image. The entry keeps ``{case}``, which the runtime resolves per case."""
    if datasets is None:
        raise TransformError(
            f"'Resample' adopts the grid of the reference '{entry}', which only a TRANSFORM program looks up.",
            _or_konfai("Resample onto a spacing or a shape"),
        )
    from konfai.data.transform.resample import _ReferenceGrid
    from konfai.utils.dataset import Dataset

    stated = clean(p.get("reference_dataset"))
    grid = _ReferenceGrid(entry, clean(p.get("reference_group")), stated)
    found = []
    for spec in [str(stated)] if grid.dataset is not None else datasets:
        source = source_of(spec)
        dataset = grid.dataset if grid.dataset is not None else Dataset(source["root"], source["format"])
        found.append({**source, "group": grid._group_in(dataset)})
    return {"entry": grid.entry, "datasets": found}


def _one_per_channel(value: Any, name: str) -> float:
    """A statistic konfai states per channel, as the single number the runtime applies to every channel."""
    if isinstance(value, list):
        if len(value) != 1:
            raise TransformError(
                f"'Standardize' states {len(value)} values for '{name}', one per channel, and the"
                " portable runtime applies a single one to every channel.",
                _or_konfai("State one value", _PREPARE),
            )
        return float(value[0])
    return float(value)


def _refuse_lazy(stage: str, p: dict[str, Any]) -> None:
    """A lazy ``Standardize`` or ``Normalize`` records its statistic for a later stage and hands its input back:
    as an op it would rewrite every voxel, as no step it would leave that statistic unrecorded."""
    if clean(p.get("lazy")):
        raise TransformError(
            f"'{stage}' is lazy: it records a statistic for a later stage to apply, and the portable runtime"
            " carries no recorded statistic.",
            "Run this config with konfai itself.",
        )


def _op_standardize(p: dict[str, Any]) -> dict[str, Any]:
    _refuse_lazy("Standardize", p)
    step: dict[str, Any] = {"op": "standardize"}
    for name in ("mean", "std"):
        if clean(p.get(name)) is not None:
            step[name] = _one_per_channel(p[name], name)
    return _masked(step, p, measures="mean" not in step or "std" not in step)


def _masked(step: dict[str, Any], p: dict[str, Any], *, measures: bool) -> dict[str, Any]:
    """``step`` reading the group its stage's ``mask`` names beside the case, when the stage measures: konfai
    measures over the mask, and never reads the mask of a stage that states its numbers."""
    mask = clean(p.get("mask"))
    if mask is not None and measures:
        step["mask_group"] = _literal_group(str(mask), TransformError)
    return step


def _op_mask(p: dict[str, Any]) -> dict[str, Any]:
    # konfai's Mask reads a path ending in .mha as a file of its own, anything else as a group of the case.
    path = str(p.get("path", "./default.mha"))
    if path.endswith(".mha"):
        raise TransformError(
            f"'Mask' reads the file '{path}', and the portable runtime reads a mask stored as a group of the case.",
            _or_konfai("Store the mask beside each case and name its group in path", _PREPARE),
        )
    return {
        "op": "mask",
        "mask_group": _literal_group(path, TransformError),
        "value_outside": float(p.get("value_outside", 0)),
    }


def _op_normalize(p: dict[str, Any]) -> dict[str, Any]:
    _refuse_lazy("Normalize", p)
    if clean(p.get("channels")):
        raise TransformError(
            f"'Normalize' rescales the channels {p['channels']} by their own extrema, and the portable runtime"
            " rescales every channel by the volume's.",
            _or_konfai("Drop channels"),
        )
    return {"op": "normalize", **_bounds("Normalize", p)}


def _bounds(stage: str, p: dict[str, Any]) -> dict[str, float]:
    """The ``min_value`` and ``max_value`` a ``Normalize`` or an ``UnNormalize`` rescales to."""
    return {name: float(p.get(name, _default(stage, name))) for name in ("min_value", "max_value")}


def _default(stage: str, name: str) -> Any:
    """konfai's own default for parameter ``name`` of the transform ``stage``, read off its signature: what a
    config that leaves the key out binds."""
    import inspect

    from konfai.data import transform as stages

    return inspect.signature(getattr(stages, stage)).parameters[name].default


def _op_pad(p: dict[str, Any]) -> dict[str, Any]:
    # Only a constant border is portable: a reflected one would need the runtime to mirror rows it does
    # not hold when the pad is wider than the axis.
    mode = str(p.get("mode", "constant"))
    if mode.split(":")[0] != "constant":
        raise TransformError(
            f"'Padding' fills its border with mode '{mode}', which has no portable runtime op.",
            "A portable Padding uses mode 'constant' or 'constant:<value>'.",
        )
    padding = p.get("padding", _default("Padding", "padding"))
    return {"op": "pad", "padding": [int(x) for x in padding], "mode": mode}


def _op_clip(p: dict[str, Any]) -> dict[str, Any]:
    # A bound is a fixed number or a data-dependent spec ("min"/"max"/"percentile:<q>") the runtime
    # resolves against the volume in hand.
    def bound(value: Any) -> Any:
        return value if isinstance(value, str) else float(value)

    saved = [key for key in ("save_clip_min", "save_clip_max") if clean(p.get(key))]
    if saved:
        # A saved bound becomes the volume's Min or Max, which a later Normalize takes over measuring its own.
        raise TransformError(
            f"'Clip' with {' and '.join(saved)} records its bound as the volume's statistic, which a later"
            " 'Normalize' reads in place of the volume's own extrema, and the portable runtime measures the extrema.",
            _or_konfai("Set save_clip_min and save_clip_max to false and give Normalize its bounds"),
        )
    step = {"op": "clip", "min_value": bound(p.get("min_value", -1024)), "max_value": bound(p.get("max_value", 1024))}
    return _masked(step, p, measures=isinstance(step["min_value"], str) or isinstance(step["max_value"], str))


def _op_axes(stage: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """``Flip`` or ``Permute``: ``dims`` names spatial axes in array order, ``|``-separated, which konfai
    reads as tensor dims ``d + 1``. The runtime refuses the axes torch refuses."""

    def compile_stage(p: dict[str, Any]) -> dict[str, Any]:
        dims = p.get("dims", "1|0|2")
        try:
            axes = [int(d) for d in str(dims).split("|")]
        except ValueError:
            raise TransformError(
                f"'{stage}' cannot read dims={dims!r}.",
                'dims names spatial axes as a "|"-separated string: dims: "1|0|2".',
            ) from None
        return {"op": stage.lower(), "axes": axes}

    return compile_stage


def _op_select_label(p: dict[str, Any]) -> dict[str, Any]:
    from konfai.data.transform import SelectLabel

    # konfai's own reading of the "(old,new)" strings, which refuses what it cannot read.
    return {"op": "relabel", "pairs": [list(pair) for pair in SelectLabel(clean(p.get("labels")) or []).labels]}


def _op_flat_label(p: dict[str, Any]) -> dict[str, Any]:
    # konfai's FlatLabel writes 1 where the voxel equals a label it lists, and above 0 when it lists none.
    labels = clean(p.get("labels"))
    if not labels:
        return {"op": "relabel", "positive": 1}
    return {"op": "relabel", "pairs": [[int(label), 1] for label in labels]}


def _op_brightness(p: dict[str, Any]) -> dict[str, Any]:
    # konfai draws the groups the draw names and leaves the others as they are; the runtime draws the chain
    # it is written in.
    if clean(p.get("groups")) is not None:
        raise TransformError(
            f"'Brightness' names the groups {p['groups']} it draws, and the portable runtime draws every copy of"
            " the chain it is written in.",
            _or_konfai("Drop groups", _PREPARE),
        )
    return {"op": "brightness", "b_std": float(p["b_std"])}


def _op_over_channels(stage: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """``Softmax`` or ``Argmax``: the runtime reads across the channel axis, konfai's ``dim: 0``, and has
    no kernel for another axis, so any other is refused by name rather than run on the channels."""

    def compile_stage(p: dict[str, Any]) -> dict[str, Any]:
        dim = int(p.get("dim", 0))
        if dim != 0:
            raise TransformError(
                f"'{stage}' reads across dim {dim}, and the portable runtime reads across the channel axis (dim 0) only.",
                _or_konfai("Set dim: 0"),
            )
        return {"op": stage.lower(), "dim": 0}

    return compile_stage


def _admitted(step: dict[str, Any]) -> dict[str, Any]:
    """``step`` as an op emitted it, its keys checked against the contract's table (``portable_schema.CHAIN_OPS``):
    a key the table does not admit, or one it requires and the op left out, is an error of this compiler."""
    from konfai.portable_schema import CHAIN_OPS

    admitted, required = CHAIN_OPS[step["op"]]
    unknown = sorted(set(step) - {"op"} - set(admitted))
    missing = sorted(set(required) - set(step))
    if unknown or missing:
        raise AssertionError(
            f"the compiler emitted '{step['op']}' with keys {unknown} outside the contract, missing {missing}"
        )
    return step


#: KonfAI transform name -> the runtime op it compiles to. A stage outside this map has no portable
#: form; each compiler refuses it by name. Either workflow's program reads the companion group a stage
#: names beside its case, into a buffer of its own.
OP_MAP: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "TensorCast": _op_cast,
    "Clip": _op_clip,
    "Resample": _op_resample,
    "Padding": _op_pad,
    "Dilate": lambda p: {"op": "dilate", "radius": int(p.get("dilate", _default("Dilate", "dilate")))},
    # Canonical reorients from the volume's own direction cosines at runtime, so it carries no params.
    "Canonical": lambda _p: {"op": "canonical"},
    # Flip and Permute move the voxels and, as konfai's do, leave the geometry untouched.
    "Flip": _op_axes("Flip"),
    "Permute": _op_axes("Permute"),
    "Standardize": _op_standardize,
    "Normalize": _op_normalize,
    "UnNormalize": lambda p: {"op": "unnormalize", **_bounds("UnNormalize", p)},
    "Softmax": _op_over_channels("Softmax"),
    "Argmax": _op_over_channels("Argmax"),
    # SelectLabel and FlatLabel are one relabel. OneHot's inverse, an argmax, has no runtime op (INVERTIBLE_OPS);
    # a cast's puts the volume back in the dtype it was stored in, which the runtime's reader recorded.
    "SelectLabel": _op_select_label,
    "FlatLabel": _op_flat_label,
    "OneHot": lambda p: {"op": "one_hot", "num_classes": int(p["num_classes"])},
    "Crop": lambda _p: {"op": "crop"},
    # konfai's Gradient: the first difference along each spatial axis, per axis or folded into the norm
    # of their sigmoids.
    "Gradient": lambda p: {"op": "gradient", "per_dim": bool(clean(p.get("per_dim", False)))},
    # konfai's Norm folds the volume's trailing axis into the magnitude of what it holds; its
    # channel-first sibling Magnitude folds the channel axis and leaves the grid.
    "Norm": lambda _p: {"op": "norm"},
    "Magnitude": lambda _p: {"op": "magnitude"},
    # ITK's HistogramMatchingImageFilter at konfai's settings, onto a group read as stored beside the case.
    "HistogramMatching": lambda p: {"op": "histogram_match", "reference_group": str(p["reference_group"])},
    "Percentage": lambda p: {"op": "percentage", "baseline": float(p["baseline"])},
}
# A prediction's Mask is a program step over a nested model's buffer (PROGRAM_TRANSFORMS). A Crop finds its box
# at run time (`compile_chain` says where it may); a prediction refuses it, konfai undoing it by a replicate pad.
_CHAIN_OPS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    **OP_MAP,
    "Mask": _op_mask,
    # A draw, compiled only past an Expand marker (`_compile_group`): konfai refuses one anywhere else.
    "Brightness": _op_brightness,
}


def moves_the_volume(step: dict[str, Any]) -> bool:
    """Whether a step can move the grid or the values the foreground box is found in. A cast to a float
    cannot: the runtime holds every value as float32 either way, and konfai's box is the same in any float."""
    return not (step["op"] == "cast" and str(step.get("dtype", "")).startswith("float"))


def refuse_a_late_crop(name: str, moved: str | None, error: type[KonfAIError]) -> None:
    """konfai's ``Crop`` keeps the box of the group as stored and the runtime the box of the volume it is
    handed: one box only while no stage before it moved the grid or the values."""
    if base_name(name) != "Crop" or moved is None:
        return
    raise error(
        f"'{name}' follows '{moved}', and konfai crops to the box it finds in the group as stored, whatever"
        f" '{moved}' did to the volume, where the portable runtime finds the box in the volume it is handed.",
        _or_konfai("Put the Crop first in the chain, or run what precedes it as a chain of its own", _PREDICT),
    )


def _result_dtype(step: dict[str, Any], dtype: str | None) -> str | None:
    """The dtype konfai's stage returns for an input of ``dtype`` (``None``: the stored one, known at run time),
    as the runtime's ``output_dtype`` settles it."""
    op = step["op"]
    if op == "cast":
        return str(step["dtype"])
    if op == "argmax":
        return "int64"
    if op in ("one_hot", "brightness", "norm"):
        return "float32"
    if op in ("standardize", "normalize", "unnormalize", "softmax") and dtype not in (None, "float64"):
        return "float32"
    return dtype


def _default_interpolation(dtype: str) -> str:
    """konfai's ``Resample._mode`` with no interpolation stated, for a volume of ``dtype``."""
    import torch

    from konfai.data.sampling import default_interpolation

    return default_interpolation(torch.empty(0, dtype=getattr(torch, dtype)))


def resolve_interpolation(
    steps: list[dict[str, Any]],
    dtype: str | None,
    undone: dict[str, str | None] | None = None,
    restored: Collection[str] = (),
) -> str | None:
    """Write on every ``resample`` step, and on every ``inverse`` of one, the interpolation konfai reads it with:
    the one its stage states, else the one the dtype at that point picks; return the dtype the steps leave.
    ``dtype`` ``None`` is the stored one, which only the runtime knows, and leaves an unstated interpolation
    unset. ``undone`` maps each resample a post chain undoes to the interpolation its stage states;
    ``restored`` names the casts it undoes, past which the volume is back in its stored dtype."""
    undone = undone or {}
    for step in steps:
        if step["op"] == "resample" or (step["op"] == "inverse" and step.get("of") in undone):
            stated = step.get("interpolation") if step["op"] == "resample" else undone[step["of"]]
            if stated is not None:
                step["interpolation"] = stated
            elif dtype is not None:
                step["interpolation"] = _default_interpolation(dtype)
        dtype = None if step["op"] == "inverse" and step.get("of") in restored else _result_dtype(step, dtype)
    return dtype


def _reduced_dtype(dtype: str, reduction: str) -> str:
    """The dtype konfai's reduction of the test-time copies returns: ``Mean`` and ``Median`` average an
    integer copy into float32, ``Vote`` and ``Concat`` keep the copies' dtype, as the runtime's do."""
    name = _class_name(reduction)
    if name in ("Vote", "Concat"):
        return dtype
    if name in ("Mean", "Median"):
        return dtype if dtype.startswith("float") else "float32"
    raise ConfigError(
        f"The test-time copies fold by '{reduction}', and the portable runtime folds by Mean, Median, Vote or Concat.",
        _or_konfai("Use one of those reductions", _PREDICT),
    )


#: The packages a chain key may name a konfai stage through, bare or module-qualified, ``konfai_apps.transforms``
#: among them for ``KonfAIInference``, the spelling the guide gives. A classpath into any other module is the
#: caller's own class: a ``Clip`` of their own is not konfai's, and compiling it as one would run another
#: operation entirely.
KONFAI_STAGE_MODULES = ("konfai.data.transform", "konfai.data.augmentation", "konfai_apps.transforms")


def base_name(name: str) -> str:
    """A chain key as the stage it names, without a repeat's suffix (``Clip#2``) or the konfai package a
    repeat is spelled through (``konfai.data.transform:Clip``)."""
    name = name.split("#", 1)[0].split("/", 1)[0]
    module, _, stage = name.rpartition(":")
    module = module.replace(":", ".")
    konfai = module in KONFAI_STAGE_MODULES or module.startswith(tuple(f"{m}." for m in KONFAI_STAGE_MODULES))
    return stage if konfai else name


def _class_name(classpath: str) -> str:
    """The class a classpath names, in either spelling (``konfai.data.reduction:Median`` -> ``Median``)."""
    return classpath.rsplit(":", 1)[-1].rsplit(".", 1)[-1]


def bound_chain(chain: Any) -> dict[str, Any]:
    """A chain as the core binds it: a mapping keyed by class name, the list spelling under its occurrence
    keys (``Clip``, ``Clip#2``), ``{}`` for a chain that prepares nothing."""
    from konfai.utils.config import _occurrence_mapping

    return _occurrence_mapping(chain, "chain") if isinstance(chain, list) else _mapping(chain)


def chain_items(chain: Any) -> list[tuple[str, dict[str, Any]]]:
    """A ``transforms:`` block as ordered ``(stage name, params)``, in either spelling."""
    return [(name, _mapping(params)) for name, params in bound_chain(chain).items()]


def _write_target(name: str, params: dict[str, Any], group_dest: str) -> dict[str, Any]:
    """The destination a ``Write`` or a ``Save`` names, split into what the runtime needs to open it."""
    dataset = clean(params.get("dataset"))
    if dataset is None:
        raise TransformError(
            f"The chain writing '{group_dest}' has a '{name}' that names no dataset.",
            "Spell it Write: {dataset: <path>[:format]}.",
        )
    if clean(params.get("scale_factors")):
        raise TransformError(
            f"'{name}' in the chain writing '{group_dest}' asks for a pyramid (scale_factors), and the portable"
            " runtime writes the full resolution only.",
            _or_konfai("Drop scale_factors", _PREPARE),
        )
    from konfai.utils.utils import split_path_spec

    # Split as konfai's own Save splits it, so a drive letter or a URI keeps its colon.
    root, _flag, fmt = split_path_spec(str(dataset), default_format="mha")
    return _destination(root, fmt, clean(params.get("group")) or group_dest)


def _opened_as(root: str, file_format: str) -> str:
    """The format konfai opens the dataset ``root`` in: a directory whose first case holds a DICOM series or an
    OME-Zarr store is read and written as one, whatever token names it."""
    from konfai.utils.dataset import Dataset

    return Dataset(root, file_format).file_format


def _destination(root: str, file_format: str, group: str) -> dict[str, Any]:
    """A write target as the runtime opens it and asks whether a case is already written there."""
    file_format = _opened_as(root, file_format)
    return {
        "root": root,
        "format": file_format,
        "group": group,
        "entry_suffixes": _entry_suffixes(file_format),
    } | _store_of(root, file_format)


def _backend_of(file_format: str) -> type[Any]:
    """The backend konfai serves ``file_format`` with: every store spelling is the OME-Zarr one."""
    from konfai.utils.dataset.backend import backend_for
    from konfai.utils.utils import STORE_FORMS

    token = file_format.lower().removeprefix(".")
    return backend_for("omezarr" if f".{token}" in STORE_FORMS else token)


def _entry_suffixes(file_format: str) -> list[str]:
    """Every suffix konfai counts ``<group>`` under when it asks whether a case is written (``""``: the bare
    name). A store's entries are datasets inside it, named by no suffix."""
    from konfai.utils.dataset import Dataset
    from konfai.utils.utils import STORE_FORMS, SUPPORTED_EXTENSIONS

    backend = _backend_of(file_format)
    if backend.single_store:
        return []
    if backend is Dataset.OmeZarrFile:  # every store spelling, then the bare directory
        return [*STORE_FORMS, ""]
    if backend is Dataset.ItkTransformFile:
        return [".h5", ".tfm"]
    if backend is Dataset.DicomFile:  # the series directory
        return [""]
    return [f".{extension}" for extension in SUPPORTED_EXTENSIONS]


def _store_of(root: str, file_format: str) -> dict[str, str]:
    """The ``store`` a program names for a format that keeps every case in one file (``h5``), whose
    entries the runtime addresses inside it; nothing for a format with a file per case."""
    from konfai.utils.dataset import Dataset
    from konfai.utils.dataset.backend import backend_for

    return {"store": Dataset.store_root_for(root, file_format)} if backend_for(file_format).single_store else {}


def _literal_group(group: str, error: type[KonfAIError]) -> str:
    """``group``, refused when konfai would resolve it by listing the dataset (a ``*`` component):
    the runtime reads the one entry the program names and never lists a dataset."""
    if "*" in group.split("/"):
        raise error(
            f"Group '{group}' holds a '*', which konfai resolves by listing the dataset, and the portable"
            " runtime reads only the entry a program names.",
            _or_konfai("Name the group in full"),
        )
    return group


def _warn(message: str) -> None:
    warnings.warn(message, KonfAIWarning, stacklevel=4)


def _stages_named(node: Any, stage: str, path: tuple[str, ...] = ()) -> Iterator[tuple[str, dict[str, Any]]]:
    """Every chain entry naming ``stage`` anywhere under ``node``, with the dotted path it sits at."""
    if isinstance(node, dict):
        for key, value in node.items():
            if base_name(str(key)) == stage:
                yield ".".join((*path, str(key))), _mapping(value)
            else:
                yield from _stages_named(value, stage, (*path, str(key)))
    elif isinstance(node, list):
        for item in node:
            yield from _stages_named(item, stage, path)


#: Keys of a prediction that change how konfai runs and not what it writes: the program has no counterpart,
#: and the run's values do not depend on them. Each names the class whose constructor states its default.
_NO_EFFECT_KEYS = {
    ("konfai.predictor.workflow", "Predictor", ""): (
        "channels_last",
        "cudnn_benchmark",
        "torch_compile",
        "gpu_checkpoints",
        "data_log",
        "checkpoint_cache_gib",
    ),
    ("konfai.data.data_manager.sources", "DataPrediction", "Dataset"): (
        "num_workers",
        "pin_memory",
        "prefetch_factor",
        "persistent_workers",
    ),
}


def warn_unportable_keys(config: dict[str, Any], root: str) -> None:
    """Warn, key by key, about what the config asks of konfai that the program does not carry, where the
    difference is known and the author may want it: a chain the two engines would read differently is refused."""
    import inspect
    from importlib import import_module

    section = config.get(root) or {}
    for path, params in _stages_named(section, "Resample"):
        if clean(params.get("precision")) == "fast":
            _warn(
                f"{root}.{path}.precision: fast computes the sampling coordinates in float32 where the portable"
                " runtime computes them in float64, so a voxel at a rounding boundary can read its neighbour."
            )
    for path, params in _stages_named(section, "Reduce"):
        if clean(params.get("provenance", True)) is not False:
            _warn(
                f"{root}.{path}.provenance: konfai records the operator and the cases folded in the output's header,"
                " and the portable runtime writes the geometry alone."
            )
    for output, spec in (section.get("outputs_dataset") or {}).items():
        attributes = clean(((spec or {}).get("OutputDataset") or {}).get("attributes"))
        if attributes:
            _warn(
                f"{root}.outputs_dataset.{output}.OutputDataset.attributes: konfai writes {attributes} into the"
                " output's header, and the portable runtime writes the geometry alone."
            )
    changed = []
    for (module, cls, sub), keys in _NO_EFFECT_KEYS.items():
        node = section.get(sub) if sub else section
        if not isinstance(node, dict) or not any(key in node for key in keys):
            continue
        defaults = inspect.signature(getattr(import_module(module), cls).__init__).parameters
        prefix = f"{root}.{sub}." if sub else f"{root}."
        changed += [prefix + key for key in keys if key in node and clean(node[key]) != defaults[key].default]
    if changed:
        _warn(
            f"{', '.join(changed)}: these keys change how konfai runs and nothing it writes; the program has no"
            " counterpart for them."
        )


@contextmanager
def _built(config_file: str | Path, build: Callable[[Path], Any]) -> Iterator[tuple[Any, dict[str, Any]]]:
    """The workflow ``build`` makes of ``config_file``, inside the environment its builders need, so KonfAI's own
    refusals run first, and the file as the build left it: every default the build bound is written in it. A
    TRANSFORM binds its stages as it builds; a PREDICTION binds them only when its dataset is prepared, so the
    stages of a prediction's file hold the parameters their author spelled and each op supplies the rest."""
    config_file = Path(config_file).resolve()
    with config_environment(config_file):
        workflow = build(config_file)
        yield workflow, YAML(typ="safe").load(config_file.read_text())


@contextmanager
def config_environment(
    config_file: Path, *, root: str | None = None, state: str | None = None, keep_file: bool = False
) -> Iterator[None]:
    """Read a KonfAI config with the environment its builders need, the config's directory on ``sys.path`` so
    a classpath beside it resolves, and restore every ``KONFAI_*`` variable afterwards and, with ``keep_file``,
    the file, which reading rewrites. ``root`` and ``state`` serve a caller that builds no workflow."""
    import os
    import sys

    previous = {key: value for key, value in os.environ.items() if key.startswith("KONFAI_")}
    snapshot = config_file.read_text() if keep_file else None
    os.environ["KONFAI_config_file"] = str(config_file)
    os.environ["KONFAI_CONFIG_MODE"] = "Done"
    if root is not None:
        os.environ["KONFAI_ROOT"] = root
    if state is not None:
        os.environ["KONFAI_STATE"] = state
    sys.path.insert(0, str(config_file.parent))
    try:
        yield
    finally:
        sys.path.remove(str(config_file.parent))
        if snapshot is not None:
            config_file.write_text(snapshot)
        for key in [key for key in os.environ if key.startswith("KONFAI_")]:
            if key not in previous:
                del os.environ[key]
        os.environ.update(previous)


def compile_chain(
    chain: Any, group_dest: str, datasets: list[str] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """A chain as portable steps plus its write target, the terminal ``Write``. A ``Save`` before it is a
    ``save`` step naming its own destination: konfai writes what reaches it, and a later run reads the last
    write already there in place of the stages before it. ``datasets`` are the run's ``dataset_filenames``,
    where a ``Resample`` naming no ``reference_dataset`` looks its reference image up."""
    items = chain_items(chain)
    if not items:
        raise TransformError(f"The chain writing '{group_dest}' is empty.", "A chain ends with a Write.")
    head, (last_name, last_params) = items[:-1], items[-1]
    if base_name(last_name) not in ("Write", "Save"):
        raise TransformError(
            f"The chain writing '{group_dest}' does not end with a Write, so it has no destination.",
            "End every chain with Write: {dataset: <path>[:format]}.",
        )
    steps: list[dict[str, Any]] = []
    moved = None  # the first stage that may move the grid or the values
    for name, params in head:
        stage = base_name(name)
        if is_save(name):
            steps.append({"op": "save", "destination": _write_target(name, params, group_dest)})
            continue
        if _class_name(stage) == "Expand":
            # Where the copies start: `_compile_expand` splits the steps here.
            steps.append({"op": "expand"})
            continue
        if records_only(name):
            continue
        refuse_a_stage_konfai_does_not_carry(name, TransformError)
        if stage not in _CHAIN_OPS:
            raise TransformError(
                f"Transform '{name}' has no portable runtime op, so this chain cannot run outside Python.",
                f"Portable stages are: {', '.join(sorted(_CHAIN_OPS))}.",
            )
        refuse_a_late_crop(name, moved, TransformError)
        step = _op_resample(params, datasets, datasets) if stage == "Resample" else _CHAIN_OPS[stage](params)
        moved = moved or (name if moves_the_volume(step) else None)
        steps.append(step)
    resolve_interpolation(steps, None)
    return steps, _write_target(last_name, last_params, group_dest)


def _compile_group(
    chain: Any, stages: list[Any], group_dest: str, datasets: list[str], cases: list[str]
) -> dict[str, Any]:
    """One ``groups_dest`` chain as the runtime runs it: its steps and its Write and, when it carries a
    ``Reduce``, the fold between the steps each case runs and the steps run once on the result. ``stages``
    are the chain as konfai bound it, which say where the ``Reduce`` sits and which operator it resolved."""
    from konfai.data import reduction
    from konfai.data.case_reduction import check_post_stages, split_chain
    from konfai.utils.dataset import Dataset

    _pre, reduce, post = split_chain(stages)
    if reduce is None:
        return _compile_expand(chain, stages, group_dest, datasets, cases)
    items = chain_items(chain)
    if any(is_save(name) for name, _ in items[:-1]):
        raise TransformError(
            f"The chain reducing into '{reduce.output}' holds a Save, and the portable runtime writes no member"
            " of a reduction.",
            _or_konfai("Drop the Save", _PREPARE),
        )
    # konfai checks the stages after a Reduce only when it plans the run.
    check_post_stages(post[:-1], reduce.output)
    if type(reduce.operator) not in (reduction.Mean, reduction.Std, reduction.Median, reduction.Vote, reduction.Concat):
        raise TransformError(
            f"The chain reducing into '{reduce.output}' folds by '{reduce.operator_classpath}', and the portable"
            " runtime folds by Mean, Std, Median, Vote or Concat.",
            _or_konfai("Use one of those operators", _PREPARE),
        )
    at = next(index for index, (name, _) in enumerate(items) if _class_name(base_name(name)) == "Reduce")

    def listed(part: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        return [{name: params} for name, params in part]

    steps, write = compile_chain(listed(items[:at] + items[-1:]), group_dest, datasets)
    post_steps, _ = compile_chain(listed(items[at + 1 :]), group_dest, datasets)
    # konfai folds a reduction one region at a time, into a store or an uncompressed MetaImage or NIfTI.
    token = write["format"].lower().removeprefix(".")
    backend = _backend_of(token)
    if not (backend.single_store or backend is Dataset.OmeZarrFile or token in ("mha", "nii")):
        raise TransformError(
            f"The chain reducing into '{reduce.output}' writes '{write['format']}', which konfai cannot write"
            " region by region, and konfai folds a reduction one region at a time.",
            "Write the reduction to mha, nii, omezarr or h5.",
        )
    fold = {
        "operator": type(reduce.operator).__name__.lower(),
        "output": reduce.output,
        "grid": reduce.grid,
        "grid_tolerance": reduce.grid_tolerance,
        "post": post_steps,
    }
    return {"steps": steps, "reduce": fold, "write": write}


def _compile_expand(
    chain: Any, stages: list[Any], group_dest: str, datasets: list[str], cases: list[str]
) -> dict[str, Any]:
    """A chain with no ``Reduce``: its steps and its Write, split at an ``Expand`` marker into the steps each
    case runs and ``expand``, what each copy runs, the copies' entry names and the seed they draw from. One
    pass compiles both sides, so the dtype and the stage that moved the values carry across the marker."""
    from konfai.data.augmentation import Brightness, DataAugmentation
    from konfai.data.transform import split_expand

    _pre, expand, _post = split_expand(stages)
    named = []
    # The stages as bound say which entries are draws: past the marker, a name both packages define is the draw.
    for (name, params), stage in zip(chain_items(chain), stages, strict=True):
        if isinstance(stage, DataAugmentation):
            if type(stage) is not Brightness:
                raise TransformError(
                    f"'{name}' draws a {type(stage).__name__} for each copy, and the portable runtime draws"
                    " Brightness only.",
                    _or_konfai("Drop the draw", _PREPARE),
                )
            name = "Brightness"
        named.append({name: params})
    steps, write = compile_chain(named, group_dest, datasets)
    if expand is None:
        return {"steps": steps, "write": write}
    at = next(index for index, step in enumerate(steps) if step["op"] == "expand")
    entries = {case: [expand.entry(case, a) for a in range(1, expand.nb + 1)] for case in cases}
    return {
        "steps": steps[:at],
        "expand": {"seed": expand.draw_seed, "entries": entries, "post": steps[at + 1 :]},
        "write": write,
    }


#: What a suffix multiplies a budget by, as konfai spells them.
_SCALE = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def _budget_bytes(budget: Any) -> int | None:
    """A ``memory_budget`` in bytes, or ``None`` when the config leaves it to the machine."""
    budget = clean(budget)
    if budget is None or budget == "auto":
        return None
    if isinstance(budget, (int, float)):
        return int(budget)
    text = str(budget).strip()
    scale = _SCALE.get(text[-1].upper())
    return int(float(text[:-1]) * scale) if scale else int(float(text))


def source_of(spec: str) -> dict[str, str]:
    """A ``path[:flag]:format`` dataset spec as the root and format the runtime opens, split by konfai's own
    grammar: a flag such as ``a`` sits between the path and the format."""
    from konfai.utils.utils import DATASET_FLAGS, SUPPORTED_FORMATS, split_path_spec

    root, _flag, file_format = split_path_spec(
        str(spec), default_format="mha", allowed_flags=DATASET_FLAGS, supported_formats=SUPPORTED_FORMATS
    )
    base, _, level = file_format.partition("@")
    if level not in ("", "0"):
        raise TransformError(
            f"Dataset '{spec}' reads resolution level {level}, and the portable runtime reads the full resolution only.",
            _or_konfai(f"Name the store without a level ({root}:{base})", _PREPARE),
        )
    base = _opened_as(root, base)
    return {"root": root, "format": base} | _store_of(root, base)


def compile_transform(
    transform_file: str | Path = Path("./Transform.yml"),
    transforms_dir: str | Path = Path("./Transforms"),
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Compile a ``Transform.yml`` into the program the portable runtime executes, written to ``output`` when
    given. The workflow binds every stage as it builds, so each op carries the parameters the run would use."""
    from konfai.transformer import Transformer, build_transform

    with _built(
        transform_file, lambda file: build_transform(transform_file=file, transforms_dir=Path(transforms_dir))
    ) as (built, resolved):
        workflow = cast(Transformer, built)
    warn_unportable_keys(resolved, "Transformer")
    dataset = workflow.dataset
    section = (resolved.get("Transformer") or {}).get("Dataset") or {}
    roots = [str(entry) for entry in section.get("dataset_filenames") or []]

    chains = []
    for group_src, group in (section.get("groups_src") or {}).items():
        _literal_group(group_src, TransformError)
        for group_dest, spec in ((group or {}).get("groups_dest") or {}).items():
            stages = dataset.groups_src[group_src][group_dest].transforms
            chain = _compile_group((spec or {}).get("transforms"), stages, group_dest, roots, list(dataset.case_names))
            chains.append({"group_src": group_src, "group_dest": group_dest, **chain})

    program = {
        "konfai_edge_transform": TRANSFORM_PROGRAM_VERSION,
        "name": workflow.name,
        # What the run may hold per case, and what it does when a chain cannot stream: the portable
        # runtime plans against the same two numbers konfai plans against.
        "memory_budget": _budget_bytes(section.get("memory_budget")),
        "on_fallback": str((resolved.get("Transformer") or {}).get("on_fallback", "warn")),
        "sources": [source_of(root) for root in roots],
        "cases": list(dataset.case_names),
        "chains": chains,
    }
    if output is not None:
        Path(output).write_text(json.dumps(program, indent=2))
    return program


# The PREDICTION half: what a prediction config's transforms, patch and outputs compile to. Read by
# `compile_prediction` below and by the konfai-apps bundle export, so the two cannot drift.


def inference_patch(config: dict[str, Any], root: str) -> dict[str, Any]:
    """The ``Dataset.Patch`` section, the sliding window the run tiles a case with; ``{}`` when the config
    states none. The model's own ``ModelPatch`` is another key, which the export refuses."""
    return _mapping((config.get(root) or {}).get("Dataset", {}).get("Patch"))


def derive_overlap(patch: dict[str, Any], patch_size: list[int]) -> list[int] | None:
    """Resolve KonfAI's overlap forms on the exported patch, retaining its 2.5-D axes."""
    from konfai.data.patching.blend import blend_overlap

    raw = clean(patch.get("overlap"))
    if isinstance(raw, bool) or raw is None or (isinstance(raw, int) and raw < 0):
        return None
    if isinstance(raw, list) and len(raw) != len(patch_size):
        declared = [int(size) for size in patch.get("patch_size", [])]
        if len(raw) == len(declared):
            raw = [value for value, size in zip(raw, declared, strict=True) if size > 1]
    return blend_overlap(raw, patch_size)


def _file_stem(name: str) -> str:
    """A named module output as a file name: its path separators are not a file system's."""
    return re.sub(r"\W+", "_", name).strip("_")


def outputs_node(config: dict[str, Any], root: str, output: str | None = None) -> Any:
    """The config's ``outputs_dataset``, or the subtree of the one output ``output`` names: every reader of
    an output's chain takes one output at a time, since a config may declare several."""
    outputs = config.get(root, {}).get("outputs_dataset")
    if output is None or not isinstance(outputs, dict):
        return outputs
    # A head is one module path; the config writes its separators as ':' and the predictor reads them as '.'.
    wanted = _file_stem(output)
    named = {_file_stem(key): key for key in outputs}
    if wanted not in named:
        raise ConfigError(
            f"the config declares no output '{output}' (it declares {', '.join(outputs) or 'none'}).",
        )
    return {named[wanted]: outputs[named[wanted]]}


def output_spec(config: dict[str, Any], root: str, output: str | None = None) -> dict[str, Any]:
    """The ``OutputDataset`` section of the output ``output`` names, or of the first declared when ``None``:
    the chains, the reduction and the blend that output runs. ``{}`` when the config declares none."""
    outputs = outputs_node(config, root, output)
    first = next(iter(outputs.values()), None) if isinstance(outputs, dict) else None
    return _mapping((first or {}).get("OutputDataset"))


def _chain_of(spec: dict[str, Any], key: str) -> dict[str, Any]:
    """One of an output's chains, as bound: a mapping, ``{}`` for a chain that prepares nothing."""
    return bound_chain(spec.get(key))


def derive_blend(config: dict[str, Any], root: str, output: str | None = None) -> str | None:
    """The output's ``patch_combine`` window (Gaussian/Cosinus/Mean) -> the manifest ``blend``; ``None`` when
    the config states none, which leaves the window to the export."""
    spec = output_spec(config, root, output)
    return str(clean(spec["patch_combine"]) or "Overwrite") if "patch_combine" in spec else None


def check_blend(overlap: list[int] | None, blend: str | None) -> None:
    """Refuse an overlap no ``patch_combine`` owns: konfai then keeps each voxel from the patch holding it most
    centrally (``Trim``), which no portable runtime op does, where the export's ``Overwrite`` keeps the last
    patch at an intersection."""
    if overlap and any(int(o) > 0 for o in overlap) and blend in (None, "Overwrite"):
        raise ConfigError(
            "the output declares an overlap and no patch_combine: konfai assembles it with Trim, each voxel from the"
            " patch holding it most centrally, which has no portable runtime op.",
            _or_konfai("Declare a patch_combine (Mean, Gaussian or Cosinus), set the overlap to 0", _PREDICT),
        )


def derive_onnx_params(
    config: dict[str, Any], root: str
) -> tuple[list[int] | None, int | None, int, float | str | None]:
    """Best-effort ``(patch_size, in_channels, extend_slice, pad_value)`` from a prediction config, ``None`` for
    what it does not state. ``patch_size`` drops a singleton slice dim (2.5D); ``in_channels`` reads the
    model's ``nb_channel``, else its ``in_channels``, else its first ``channels``."""
    patch_size: list[int] | None = None
    extend_slice = 0
    pad_value: float | str | None = None
    patch = inference_patch(config, root)
    if patch:
        raw = patch.get("patch_size")
        if isinstance(raw, list):
            dims = [int(d) for d in raw]
            if any(d <= 0 for d in dims):
                raise ConfigError(
                    f"patch_size {dims} contains a free axis; ONNX export requires fixed spatial dimensions.",
                    "Set an explicit positive patch_size before compiling the portable program.",
                )
            patch_size = [d for d in dims if d > 1] or dims
        raw_extend = patch.get("extend_slice", 0)
        if isinstance(raw_extend, int):
            extend_slice = raw_extend
        elif isinstance(raw_extend, str) and raw_extend.lstrip("-").isdigit():
            extend_slice = int(raw_extend)
        raw_pad = patch.get("pad_value")
        if isinstance(raw_pad, (int, float)) and not isinstance(raw_pad, bool):
            pad_value = float(raw_pad)
        elif raw_pad == "edge":
            pad_value = raw_pad
        elif raw_pad is None:
            pad_value = "min"  # konfai pads with the minimum of the window it reads (`Patch._pad_constant`)

    in_channels: int | None = None
    for model in map(_mapping, _mapping(_mapping(config.get(root)).get("Model")).values()):
        channels = model.get("channels")
        first = channels[0] if isinstance(channels, list) and channels else None
        in_channels = next(
            (n for n in (model.get("nb_channel"), model.get("in_channels"), first) if isinstance(n, int)), None
        )
        if in_channels is not None:
            break
    return patch_size, in_channels, extend_slice, pad_value


# Transforms compiled into PROGRAM steps (cross-buffer, or a buffer produced by a nested model), never
# per-buffer manifest pipeline ops. `compile_pipeline` leaves one to its caller only where the caller
# builds that step, and refuses it anywhere else: the program would run without it.
PROGRAM_TRANSFORMS = {"Mask", "KonfAIInference", "Dilate", "InferenceStack", "MergeLabels"}

#: A stacked group's leading axis read as the members to spread, which a workflow folds with a Reduce instead.
_SPREAD = (
    "the stage reads the leading axis of a stacked group as the members to spread, which a TRANSFORM or"
    " a PREDICTION folds with a Reduce or a reduction instead",
    _or_konfai("Fold the members with Reduce: Std"),
)

#: Stages whose portable form would not be konfai's, and what konfai does with them instead. Measured on
#: konfai 1.9: a Squeeze writes the entry back on the grid it kept, extent-1 axis included, and a Flatten
#: reaches the writer as one long axis, which ITK refuses ("The maximum supported Image dimension is 5").
REFUSED = {
    "Squeeze": (
        "konfai keeps the grid of the axis it drops and writes the entry back on it, so the stage changes"
        " nothing a run writes, where the portable runtime would write the volume it left",
        _or_konfai("Drop the Squeeze", _PREPARE),
    ),
    "Flatten": (
        "konfai hands the writer one long axis, which ITK refuses past five dimensions, so no run writes"
        " what this stage leaves",
        _or_konfai("Drop the Flatten", _PREPARE),
    ),
    "Variance": _SPREAD,
    "StandardDeviation": _SPREAD,
    "SegmentationDisagreement": (
        "the stage reads the leading axis of a stacked group as the members to disagree over, which a"
        " TRANSFORM or a PREDICTION folds with a Reduce or a reduction instead",
        "Run this config with konfai itself.",
    ),
    "Sum": (
        "konfai reads the axis it sums as the models' channel counts when the case carries them, and as a"
        " plain axis otherwise, so the same chain sums two different things",
        _or_konfai("Spell the label merge as MergeLabels"),
    ),
    "Foreign": (
        "the stage runs a callable from another framework, which only Python runs",
        _or_konfai("Replace it with the konfai stage that does the same"),
    ),
}


def refuse_a_stage_konfai_does_not_carry(name: str, error: type[KonfAIError]) -> None:
    """Refuse a stage whose portable form would not be konfai's, by name and with konfai's own behaviour."""
    if (refusal := REFUSED.get(base_name(name))) is not None:
        raise error(f"'{name}': {refusal[0]}.", refusal[1])


def input_groups(config: dict[str, Any], root: str) -> list[tuple[str, dict[str, Any]]]:
    """Every input group (``is_input``, true when unset as in the core), as ``(group_src, the group's spec)``,
    in the order konfai hands them to the model: its first input is branch ``'0'``, the next ``'1'``, and so on."""
    return [(group_src, group) for group_src, _dest, group in _groups(config, root) if group.get("is_input", True)]


def input_group_transforms(config: dict[str, Any], root: str, *, key: str = "transforms") -> dict[str, Any] | None:
    """The case or patch chain of the model's first input group, ``None`` when it states none: another group's
    chain never stands in."""
    groups = input_groups(config, root)
    chain = groups[0][1].get(key) if groups else None
    return bound_chain(chain) if isinstance(chain, (dict, list)) else None


def _groups(config: dict[str, Any], root: str) -> list[tuple[str, str, dict[str, Any]]]:
    """Every destination group of the dataset, as ``(group_src, group_dest, its spec)``, in the config's order."""
    dataset = config.get(root, {}).get("Dataset", {})
    return [
        (group_src, group_dest, group)
        for group_src, src in (dataset.get("groups_src") or {}).items()
        if isinstance(src, dict)
        for group_dest, group in (src.get("groups_dest") or {}).items()
        if isinstance(group, dict)
    ]


def _stage_class(name: str) -> type[Any] | None:
    """The konfai ``Transform`` class a chain entry names, ``None`` when it names none."""
    from konfai.data.transform import Transform
    from konfai.utils.utils import get_module

    try:
        module, cls_name = get_module(base_name(name), "konfai.data.transform")
        cls = getattr(module, cls_name)
    except Exception:
        return None
    return cls if isinstance(cls, type) and issubclass(cls, Transform) else None


def try_fold(name: str, params: Any) -> Callable[[Any], Any] | None:
    """A tensor->tensor callable for the exporter's ``fold_pre`` if ``name`` is a POINTWISE,
    torch-instantiable transform (bakeable into the ONNX graph), else ``None``."""
    from konfai.data.transform import LocalityKind
    from konfai.utils.dataset import Attribute

    cls = _stage_class(name)
    if cls is None:
        return None
    try:
        inst = cls(**{key: clean(value) for key, value in _mapping(params).items()})
        if inst.patch_locality(Attribute()).kind is not LocalityKind.POINTWISE:
            return None
    except Exception:
        return None
    return lambda tensor: inst("fold", tensor, Attribute())


def declares_inverse(name: str, params: Any) -> bool:
    """Whether the predictor undoes this chain entry after the forward: a ``TransformInverse`` whose
    ``apply_inverse`` is set, read from the class so a user's transform is treated like a shipped one."""
    from konfai.data.transform import TransformInverse
    from konfai.utils.utils import get_module

    kwargs = {key: clean(value) for key, value in _mapping(params).items()}
    try:
        module, cls_name = get_module(base_name(name), "konfai.data.transform")
        cls = getattr(module, cls_name)
        if not (isinstance(cls, type) and issubclass(cls, TransformInverse)):
            return False
        return bool(cls(**kwargs).apply_inverse)
    except Exception:
        return bool(kwargs.get("inverse"))


def records_only(name: str) -> bool:
    """Whether a chain entry hands its input back untouched (``alters_values``) and is no ``Save``: it compiles
    to no step, and what it records on the case is an attribute sidecar the runtime never writes."""
    from konfai.data.transform import Save

    cls = _stage_class(name)
    return cls is not None and not cls.alters_values and not issubclass(cls, Save)


def is_save(name: str) -> bool:
    """Whether a chain entry is a ``Save``, a ``Write`` included: it writes what reaches it."""
    from konfai.data.transform import Save

    cls = _stage_class(name)
    return cls is not None and issubclass(cls, Save)


def compile_pipeline(
    transforms: dict[str, Any] | None,
    *,
    fold: bool = False,
    invertible: bool = False,
    consumed: Collection[str] = (),
    roots: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[Callable[[Any], Any]], list[str]]:
    """Map an ordered KonfAI transform chain to runtime ops. With ``fold``, a POINTWISE transform with no op is
    baked into the ONNX graph, and the folds must end the chain. With ``invertible`` (the input chain, the only
    one the predictor undoes) a step the chain undoes is named, and its name returned, for the post chain to
    place its inverse. ``consumed`` names the program stages the caller builds a step for; any other is refused.
    ``roots`` are the run's ``dataset_filenames``, where a ``Resample`` looks its stored transforms up."""
    steps: list[dict[str, Any]] = []
    folds: list[Callable[[Any], Any]] = []
    inverted: list[str] = []
    moved: str | None = None
    # A chain is a mapping (konfai rewrites the list spelling into one under occurrence keys as it binds)
    # or konfai's `None`, in either spelling, for a group that prepares nothing.
    for name, params in chain_items(clean(transforms)):
        base = base_name(name)
        # A stage that alters no value is no step, and neither is a Save: the program writes no cache and runs
        # the stages before it again, and an output chain's Save hands its input back.
        if records_only(name) or is_save(name):
            continue
        refuse_a_stage_konfai_does_not_carry(name, ConfigError)
        if base in PROGRAM_TRANSFORMS:
            if base in consumed:
                continue
            raise ConfigError(
                f"transform '{name}' compiles to a program step, and nothing builds one where this chain places"
                " it, so the program would run without it.",
                "An InferenceStack or a MergeLabels folds concatenated copies as the first"
                " after_reduction_transforms stage; a Mask over a nested model's group is applied by the konfai-apps"
                " bundle export. Otherwise predict this config with konfai itself.",
            )
        if base in OP_MAP:
            refuse_a_late_crop(name, moved, ConfigError)
            if folds:
                raise ConfigError(
                    f"transform '{name}' is a runtime op but follows a folded transform; move folded "
                    "(pointwise custom) transforms to the end of the inference pipeline."
                )
            step = _admitted(_op_resample(params, roots=roots) if base == "Resample" else OP_MAP[base](params))
            moved = moved or (name if moves_the_volume(step) else None)
            if invertible and declares_inverse(name, params):
                step["id"] = name
                inverted.append(name)
            steps.append(step)
            continue
        folded = try_fold(name, params) if fold else None
        if folded is None:
            raise ConfigError(
                f"transform '{name}' has no portable runtime op and is not a foldable pointwise transform; "
                "the ONNX bundle is not deployable in a Python-free runtime. Remove it from the inference "
                "config or extend the runtime op registry."
            )
        folds.append(folded)
    return steps, folds, inverted


def _split_stack(
    config: dict[str, Any], root: str, spec: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """The params of the ``InferenceStack`` that opens the after-reduction chain of the output ``spec``, ``None``
    when none does, and the stages that follow; an opening ``MergeLabels`` is split off too, the program's
    ``merge_labels`` op applying it. konfai's stack folds the first axis of what the reduction hands it, which
    holds every copy only when the models, and the test-time copies if any, combine by ``Concat``."""
    items = list(_chain_of(spec, "after_reduction_transforms").items())
    if items and base_name(items[0][0]) == "MergeLabels":
        return None, dict(items[1:])
    if not items or base_name(items[0][0]) != "InferenceStack":
        return None, dict(items)
    combine = str(clean(config.get(root, {}).get("combine")) or "Mean")
    copies = str(clean(spec.get("reduction")) or "Mean")
    stacked = [combine, copies] if len(tta_passes(config, root)) > 1 else [combine]
    if any(_class_name(value) != "Concat" for value in stacked):
        raise ConfigError(
            f"transform '{items[0][0]}' folds the first axis of what the reduction hands it, which holds every"
            " copy only when the models and the test-time copies are concatenated; this config combines the"
            f" models by '{combine}' and the copies by '{copies}'.",
            _or_konfai(
                "Set combine: Concat, and reduction: Concat on the output when the config has test-time augmentation",
                _PREDICT,
            ),
        )
    return _mapping(items[0][1]), dict(items[1:])


# The ops that put a volume on another grid. A group past the first that runs one lands off the grid the
# model's first input lands on, which konfai's patching does not allow.
REGRIDS = ("resample", "pad", "crop", "canonical", "permute")


def _group_chain(group_src: str, group: dict[str, Any], channels: int) -> dict[str, Any]:
    """One more group the model reads: its name on disk, what it holds per voxel, and the chain that
    prepares it. konfai patches every group of a case on the first one's grid, so a chain moving this
    group off that grid is refused by name rather than discovered as a shape mismatch at run time."""
    steps, _folds, _inverted = compile_pipeline(group.get("transforms"))
    if (moved := next((step for step in steps if step["op"] in REGRIDS), None)) is not None:
        raise ConfigError(
            f"the chain of group '{group_src}' runs a '{moved['op']}', which moves it off the grid the"
            " model's first input lands on, and konfai patches every group of a case on one grid.",
            _or_konfai("Drop the stage, put the same one on the first input group", _PREDICT),
        )
    if clean(group.get("patch_transforms")):
        raise ConfigError(
            f"the group '{group_src}' carries patch_transforms, which the portable runtime runs on the"
            " model's first input alone.",
            _or_konfai("Move them to the first input group", _PREDICT),
        )
    return {"group": _literal_group(group_src, ConfigError), "channels": channels, "preprocessing": steps}


def transform_manifest(
    config: dict[str, Any],
    root: str,
    *,
    masked: bool = False,
    output: str | None = None,
    others: list[int] | None = None,
) -> tuple[dict[str, Any], list[Callable[[Any], Any]], int]:
    """The pre/post op pipeline the portable runtime applies around the tiled forward, the ``fold_pre``
    callables (POINTWISE preprocessing transforms baked into the ONNX graph), and how many post ops the
    predictor runs per copy, before any reduction. ``masked``: the caller applies the before-reduction
    ``Mask`` stages as program steps of its own. ``others`` is what each group past the first holds per
    voxel, in the order the model reads them."""
    _refuse_another_mirror(config, root, output)
    roots = [str(entry) for entry in ((config.get(root) or {}).get("Dataset") or {}).get("dataset_filenames") or []]
    pre, folds, inverted = compile_pipeline(
        input_group_transforms(config, root), fold=True, invertible=True, roots=roots
    )
    folds += _patch_folds(config, root)
    post, per_copy = _post_chain(config, root, output_spec(config, root, output), pre, inverted, masked)
    manifest = {"preprocessing": pre, "postprocessing": post}
    reads = _reads_of(config, root, pre, others)
    if reads:
        manifest["reads"] = reads
    return manifest, folds, per_copy


def _refuse_another_mirror(config: dict[str, Any], root: str, output: str | None) -> None:
    """konfai undoes the chain of the group the output mirrors (``same_as_group``), and the compiler the first
    input group's, the only one it reads: another group's inverses would be wrong without a word."""
    mirrored = clean(output_spec(config, root, output).get("same_as_group"))
    first = next(iter(input_groups(config, root)), None)
    if mirrored in (None, "default") or first is None:
        return
    src, _, dest = str(mirrored).partition(":")
    groups = (config.get(root) or {}).get("Dataset", {}).get("groups_src") or {}
    if ((groups.get(src) or {}).get("groups_dest") or {}).get(dest) is not first[1]:
        raise ConfigError(
            f"the output mirrors the group '{mirrored}' and the compiler undoes the chain of the first input"
            f" group '{first[0]}': the inverses written would be another group's.",
            _or_konfai("Let the output mirror the first input group", _PREDICT),
        )


def _patch_folds(config: dict[str, Any], root: str) -> list[Callable[[Any], Any]]:
    """The input group's ``patch_transforms``, as ``fold_pre`` callables baked after the case chain's."""
    folds = []
    for name, params in chain_items(input_group_transforms(config, root, key="patch_transforms")):
        if records_only(name):
            continue
        fold = None if declares_inverse(name, params) else try_fold(name, params)
        if fold is None:
            raise ConfigError(
                f"Input patch_transforms contains '{name}', whose patch state or inverse the portable"
                " runtime cannot represent.",
                _or_konfai(
                    "Use a pointwise patch transform with inverse: false, move the stage to transforms", _PREDICT
                ),
            )
        folds.append(fold)
    return folds


def _post_chain(
    config: dict[str, Any],
    root: str,
    spec: dict[str, Any],
    pre: list[dict[str, Any]],
    inverted: list[str],
    masked: bool,
) -> tuple[list[dict[str, Any]], int]:
    """The post chain in the predictor's own order, and how many of its ops run per copy: the per-copy
    transforms, then the ones that follow the reduction, then the inverses of the input chain in reverse, then
    the final transforms. Softmax and argmax after the inverses decide a class on the output grid; before them,
    a resample of a label map would move every class boundary by a voxel."""
    before, _, _ = compile_pipeline(
        _chain_of(spec, "before_reduction_transforms"), consumed=("Mask",) if masked else ()
    )
    # The stack or the label merge that opens the chain is the reduce op `derive_reduction` names.
    stack, after_stack = _split_stack(config, root, spec)
    after, _, _ = compile_pipeline(after_stack)
    final, _, _ = compile_pipeline(_chain_of(spec, "final_transforms"))
    op_of = _op_of(pre)
    for name in inverted:
        if op_of[name] not in INVERTIBLE_OPS:
            raise ConfigError(
                f"transform '{name}' is undone after the forward and the portable runtime has no inverse "
                f"for its '{op_of[name]}' op; set inverse: false on it, or extend the runtime.",
            )
    for step in pre:
        mapped = "a field" if "field_group" in step else "stored transforms" if "transforms" in step else None
        if "id" in step and mapped and not {"spacing", "shape", "reference"} & step.keys():
            raise ConfigError(
                f"transform '{step['id']}' is undone after the forward, and konfai has no inverse for a Resample"
                f" through {mapped} that changes no grid: undoing it is applying another map, not this one"
                " backwards.",
                "Set inverse: false on it.",
            )
    # The interpolations each resample states, before the input chain's own are resolved.
    undone = {step["id"]: step.get("interpolation") for step in pre if step["op"] == "resample" and "id" in step}
    restored = [step["id"] for step in pre if step["op"] == "cast" and "id" in step]
    resolve_interpolation(pre, None)
    # konfai undoes a resample with the fill its stage states.
    fills = {step["id"]: step["fill"] for step in pre if step["op"] == "resample" and "id" in step and "fill" in step}
    inverses = [
        {"op": "inverse", "of": name, **({"fill": fills[name]} if name in fills else {})} for name in reversed(inverted)
    ]
    # The model hands the post chain a floating tensor, and the reduction between the per-copy ops and
    # the rest settles the dtype the inverses read.
    dtype = resolve_interpolation(before, "float32") or "float32"
    folded = _reduced_dtype(dtype, str(clean(spec.get("reduction")) or "Mean"))
    rest = after + inverses + final
    resolve_interpolation(rest, folded, undone, restored)
    post = before + rest
    # konfai folds integer copies into `folded`, and a mean stack casts its average back to it. The runtime
    # does neither, so the cast is written where the copies fold, past the index selections left per copy.
    mean_stack = stack is not None and stack.get("mode") != "median"
    if not dtype.startswith("float") and (folded != dtype or mean_stack):
        post.insert(split_post(post, pre, len(before)), {"op": "cast", "dtype": folded})
    return post, len(before)


def _reads_of(
    config: dict[str, Any], root: str, pre: list[dict[str, Any]], others: list[int] | None
) -> list[dict[str, Any]]:
    """Every group the model reads past the first, with the chain that prepares it, in the order konfai hands
    them to the network; then the groups the input chain's own stages read beside the case (a masked bound or
    statistic, a histogram reference, a displacement field), which konfai hands them as stored, so the program
    runs nothing on them."""
    reads = [
        _group_chain(group_src, group, channels)
        for (group_src, group), channels in zip(
            input_groups(config, root)[1:], chain(others or [], repeat(1)), strict=False
        )
    ]
    named = {read["group"] for read in reads}
    companions: dict[str, int] = {}
    for step in pre:
        if group := step.get("mask_group") or step.get("reference_group"):
            companions.setdefault(group, 1)
        elif group := step.get("field_group"):
            # A displacement field holds one component per spatial axis of the case the patch cuts.
            companions.setdefault(group, len(inference_patch(config, root).get("patch_size") or [0, 0, 0]))
    return reads + [
        {"group": group, "channels": channels, "preprocessing": []}
        for group, channels in companions.items()
        if group not in named
    ]


# The preprocessing ops the runtime knows how to undo. A chain that asks for any other inverse is
# refused by name at export, rather than shipped and discovered at run time.
INVERTIBLE_OPS = ("cast", "resample", "pad", "canonical", "flip", "permute", "crop", "standardize", "normalize")

# Preprocessing ops whose inverse only picks and moves rows. An index selection commutes with an
# elementwise reduction, so it may stay per copy, where each buffer is still on the smaller grid.
INDEX_SELECTIONS = ("pad", "canonical", "flip", "permute", "crop")


def _op_of(preprocessing: list[dict[str, Any]]) -> dict[Any, Any]:
    """The op of every preprocessing step a post chain may undo, by the name its ``inverse`` gives."""
    return {step["id"]: step["op"] for step in preprocessing if "id" in step}


def split_post(post: list[dict[str, Any]], preprocessing: list[dict[str, Any]], per_copy: int) -> int:
    """Where a post chain stops being per copy: past the first ``per_copy`` ops, which the predictor runs on
    every copy, an op stays per copy only while it commutes with the reduction."""
    op_of = _op_of(preprocessing)
    split = per_copy
    while split < len(post) and post[split]["op"] == "inverse" and op_of.get(post[split].get("of")) in INDEX_SELECTIONS:
        split += 1
    return split


def hoist_ensemble_tail(
    manifests: list[dict[str, Any]], per_copy: int, stacked: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(stripped_manifests, tail_ops)``: a same-class ensemble reduces raw logits, so everything the predictor
    runs once moves past the reduction as program ops, in the chain's order, and each fold keeps the rest.
    ``stacked``: konfai's ``InferenceStack`` takes each copy before the case's inverses, so an undone ``Padding``
    moves past the reduction too, as the ``pad`` of the margins it added taken back."""
    post, preprocessing = manifests[0].get("postprocessing", []), manifests[0].get("preprocessing", [])
    op_of = _op_of(preprocessing)
    undone = {step["id"]: step for step in preprocessing if "id" in step}
    split = per_copy if stacked else split_post(post, preprocessing, per_copy)

    tail: list[dict[str, Any]] = []
    for step in post[split:]:
        op = step["op"]
        if op == "cast":
            # A cast sets the dtype the rest of the tail reads and the output is written in.
            tail.append({"op": "cast", "dtype": step["dtype"]})
            continue
        if op == "inverse" and stacked and op_of.get(step.get("of")) in INDEX_SELECTIONS:
            of = step.get("of")
            if op_of.get(of) != "pad":
                raise ConfigError(
                    f"assemble_program: the InferenceStack takes each copy before the inverse of '{of}'"
                    f" ({op_of.get(of)}), which a program undoes on each copy only.",
                    _or_konfai("Drop the InferenceStack", _PREDICT),
                )
            tail.append({"op": "pad", "padding": [-margin for margin in undone[of]["padding"]], "mode": "constant"})
            continue
        if op == "inverse":
            of = step.get("of")
            kind = op_of.get(of)
            if kind != "resample":
                raise ConfigError(
                    f"assemble_program: cannot hoist the inverse of '{of}' ({kind}) past the reduction; "
                    "only a resample back onto the caller's grid is a buffer op."
                )
            if any(t["op"].startswith("resample_") for t in tail):
                raise ConfigError(
                    "assemble_program: two resamples to undo past the reduction; only the outermost "
                    "lands on the grid the caller gave."
                )
            # The resample undone placed the buffer's grid against the input's: off the default align, the
            # op carries that align and the spacing it named, which an extent alone does not give back, and
            # the fill the input's voxels past the buffer's far face read, which an extent never leaves.
            forward = undone[of]
            placed = {key: forward[key] for key in ("align", "spacing") if "align" in forward and key in forward}
            if "align" in forward and "fill" in step:
                placed["fill"] = step["fill"]
            tail.append({"op": f"resample_{step['interpolation']}", **placed, "extra": ["input"]})
            continue
        if op not in ("softmax", "argmax"):
            raise ConfigError(
                f"assemble_program: cannot hoist post op '{op}' across the reduction (only softmax/argmax "
                "are channel-axis buffer ops; a per-voxel intensity op belongs in each fold's manifest)."
            )
        if int(step.get("dim", 0)) != 0:
            raise ConfigError(
                f"assemble_program: ensemble post op '{op}' has dim={step['dim']}; the program reduction "
                "runs on the channel axis (dim 0), so it cannot be hoisted across the reduction."
            )
        tail.append({"op": op})

    stripped = [{**mf, "postprocessing": mf.get("postprocessing", [])[:split]} for mf in manifests]
    return stripped, tail


def _refuse_values_past_the_merge(manifest: dict[str, Any], per_copy: int) -> None:
    """konfai merges the models' label maps before it runs the rest of the post chain once; the program
    merges each model's finished map. Both give the same map only while every op past the per-copy
    prefix undoes an index selection or a nearest resample, which picks the same voxels in every model."""
    op_of = _op_of(manifest.get("preprocessing", []))
    for step in manifest.get("postprocessing", [])[per_copy:]:
        undone = op_of.get(step.get("of")) if step["op"] == "inverse" else None
        if undone in INDEX_SELECTIONS or (undone == "resample" and step.get("interpolation") == "nearest"):
            continue
        name = step.get("of") if step["op"] == "inverse" else step["op"]
        raise ConfigError(
            f"'{name}' runs once on konfai's merged label map, and a program merges each model's finished map:"
            " only undoing an index selection or a nearest resample gives the same map either way.",
            _or_konfai(
                "Decide each model's class before the reduction and undo only such stages after the merge", _PREDICT
            ),
        )


def assemble_program(
    models: list[dict[str, Any]],
    *,
    reduce: str,
    passes: list[list[dict[str, Any]]] | None = None,
    aux: dict[str, dict[str, Any]] | None = None,
    masks: list[dict[str, Any]] | None = None,
    classes: list[int] | None = None,
    per_copy: int = 0,
    fold_models: bool = False,
    stack: str | None = None,
    checkpoints: list[str] | None = None,
) -> dict[str, Any]:
    """The program: every model (``{"id", "manifest"}``) on every test-time pass, each into a buffer of its own,
    reduced once, then the tail the predictor runs once. A pass is the ops that make a copy from the input
    (``[[]]``, the identity alone, by default), undone on each output before the reduction; a permuted copy is
    resampled back onto the input's grid there, where konfai reduces on the working grid and resamples once,
    the same value up to the rounding of a linear read. ``fold_models`` averages each pass's models first and
    runs the ``per_copy`` ops on that average (:func:`models_fold_first`). ``aux`` names the nested models a ``Mask`` reads, each
    ``{"id", "manifest", "ops"}``; ``masks`` are ``{"group", "value_outside"}``, applied to the reduced volume.

    ``mean`` hoists the run-once tail past the reduction (:func:`hoist_ensemble_tail`); ``merge_labels`` keeps
    each model's post and merges the classes its head outputs, unless ``classes`` states them. One model on one
    pass, with nothing to mask, is one step: konfai's reduction of a single copy hands it back.

    ``stack`` is the mode of the ``InferenceStack`` the reduction stands for: the program then also writes the
    buffer ``stack``, one float32 channel per copy, each the copy as konfai stacks it (its label for ``Seg``) after
    the masks, its row taken as soon as the copy is written. ``checkpoints`` names each model's checkpoint, and
    the program's ``members`` maps each copy's buffer to its checkpoint and its pass, the runtime's selection.
    """
    if not models:
        raise ConfigError("assemble_program: at least one model is required")
    if reduce not in ("mean", "median", "median_lower", "vote", "concat", "merge_labels"):
        raise ConfigError(
            f"assemble_program: unknown reduction '{reduce}' (expected 'mean', 'median', 'median_lower', 'vote',"
            " 'concat' or 'merge_labels'",
        )
    passes = passes or [[]]
    masks = masks or []
    # A nested model no mask of this output reads builds nothing: its inference would run on every case for
    # nothing, and its ops, ending the steps, would stand in for the output the one-model return below names.
    read = {spec["group"] for spec in masks}
    aux = {name: nested for name, nested in (aux or {}).items() if name in read}
    if len(passes) > 1 and reduce == "merge_labels":
        raise ConfigError(
            "test-time passes reduce by 'merge_labels', and a label merge has no meaning across flips.",
            "A label merge combines models with disjoint classes; fold flipped copies by Mean, Median, Vote or Concat.",
        )

    manifests = [model["manifest"] for model in models]
    folded: list[dict[str, Any]] = []
    if fold_models and len(models) > 1:
        post = manifests[0].get("postprocessing", [])
        folded = post[:per_copy]
        moves = [step["op"] for step in folded if step["op"] in ("resample", *INDEX_SELECTIONS)]
        if moves and split_post(post, manifests[0].get("preprocessing", []), per_copy) > per_copy:
            raise ConfigError(
                f"assemble_program: the per-copy '{moves[0]}' runs on the models' average, past the inverses each"
                " model keeps, which konfai runs after it.",
                _or_konfai("Combine the models by Concat", _PREDICT),
            )
        manifests = [{**mf, "postprocessing": mf.get("postprocessing", [])[per_copy:]} for mf in manifests]
        per_copy = 0
    tail: list[dict[str, Any]] = []
    copies_of = len(models) * len(passes)
    if reduce == "merge_labels" and copies_of > 1:
        _refuse_values_past_the_merge(manifests[0], per_copy)
    elif copies_of > 1 or masks:
        manifests, tail = hoist_ensemble_tail(manifests, per_copy, stacked=bool(stack) and copies_of > 1)

    steps, copies = _model_steps(models, manifests, passes, aux)
    members = (
        {
            copy: {"checkpoint": checkpoints[at % len(models)], "pass": at // len(models)}
            for at, copy in enumerate(copies)
        }
        if checkpoints
        else {}
    )
    # Every copy sits on the same grid, the one each mask's buffer is resampled onto.
    mask_buffers = _mask_buffers(aux, copies[0], steps)
    rows = _stack_rows(steps, copies, len(aux), stack, masks, mask_buffers) if stack and len(copies) > 1 else []
    if folded:
        copies = _fold_models(steps, copies, len(models), folded)
    if len(copies) == 1 and not tail and not masks:
        only = {**members[copies[0]]} if members else None
        return {"steps": [*steps[:-1], {**steps[-1], "out": "output"}], "output": "output"} | (
            {"members": {"output": only}} if only else {}
        )

    # A cast opening the tail is the dtype konfai folds the copies into, so it runs before the masks.
    casts = len(list(takewhile(lambda step: step["op"] == "cast", tail)))
    ordered = [*tail[:casts], *({"mask": spec} for spec in masks), *tail[casts:]]
    cursor = copies[0]
    if len(copies) > 1:
        cursor = "r" if ordered else "output"
        op: dict[str, Any] = {"op": reduce, "in": copies, "out": cursor}
        if reduce == "merge_labels":
            op["classes"] = classes if classes is not None else [int(mf["output"]["channels"]) for mf in manifests]
        steps.append(op)
    _append_tail(steps, ordered, cursor, mask_buffers)
    program: dict[str, Any] = {"steps": steps, "output": "output"}
    if rows:
        steps.append({"op": "concat", "in": rows, "out": "stack"})
        program["stack"] = "stack"
    return program | ({"members": members} if members else {})


def _stack_rows(
    steps: list[dict[str, Any]],
    copies: list[str],
    nested: int,
    mode: str,
    masks: list[dict[str, Any]],
    mask_buffers: dict[str, str],
) -> list[str]:
    """Reorder ``steps`` (the ``nested`` models, the copies, then the ops building the masks' buffers on the first
    copy's grid) so each copy's stack row follows the copy, the masks' buffers built after the first, and name the
    rows: the copy, its label for ``Seg`` (konfai's ``InferenceStack`` takes it of the blended logits), each mask, in
    float32. A copy is then held until its row is taken, not until the stack is."""
    models, built = steps[nested : nested + len(copies)], steps[nested + len(copies) :]
    del steps[nested:]
    rows = []
    for at, step in enumerate(models):
        steps.append(step)
        if at == 0:
            steps.extend(built)
        cursor = step["out"]
        ops: list[dict[str, Any]] = [{"op": "argmax"}] if mode == "Seg" else []
        ops += [
            {"op": "mask", "value_outside": spec["value_outside"], "extra": [mask_buffers[spec["group"]]]}
            for spec in masks
        ]
        for index, op in enumerate([*ops, {"op": "cast", "dtype": "float32"}]):
            out = f"{step['out']}__s{index}"
            args = {key: value for key, value in op.items() if key != "extra"}
            steps.append({**args, "in": [cursor, *op.get("extra", [])], "out": out})
            cursor = out
        rows.append(cursor)
    return rows


def _model_steps(
    models: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    passes: list[list[dict[str, Any]]],
    aux: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """The nested models first, each on the input, then every model on every pass, each into a buffer of its
    own; and the names of those copies' buffers."""
    steps = [
        {"model": nested["id"], "in": _model_inputs(nested["manifest"]), "out": name, "manifest": nested["manifest"]}
        for name, nested in aux.items()
    ]
    copies: list[str] = []
    for index, ops in enumerate(passes):
        for model, manifest in zip(models, manifests, strict=True):
            out = f"{model['id']}__p{index}" if len(passes) > 1 else str(model["id"])
            copy = _copy_manifest(manifest, ops)
            steps.append({"model": model["id"], "in": _model_inputs(copy), "out": out, "manifest": copy})
            copies.append(out)
    return steps, copies


def _fold_models(steps: list[dict[str, Any]], copies: list[str], models: int, ops: list[dict[str, Any]]) -> list[str]:
    """Append, per test-time pass, the mean of its ``models`` buffers and ``ops`` on that mean; the passes'
    buffers."""
    folded = []
    for start in range(0, len(copies), models):
        mean = cursor = f"mean__p{start // models}"
        steps.append({"op": "mean", "in": copies[start : start + models], "out": mean})
        for index, op in enumerate(ops):
            out = f"{mean}__{index}"
            steps.append({**op, "in": [cursor], "out": out})
            cursor = out
        folded.append(cursor)
    return folded


def _append_tail(
    steps: list[dict[str, Any]], ordered: list[dict[str, Any]], cursor: str, mask_buffers: dict[str, str]
) -> None:
    """Append the run-once ops and the masks, in order, each reading the buffer ``cursor`` the one before
    wrote, and the last writing ``output``. A tail op reads its ``extra`` buffers too."""
    for i, entry in enumerate(ordered):
        out = "output" if i == len(ordered) - 1 else f"t{i}"
        if "mask" in entry:
            spec = entry["mask"]
            mask = mask_buffers[spec["group"]]
            steps.append({"op": "mask", "in": [cursor, mask], "out": out, "value_outside": spec["value_outside"]})
        else:
            args = {key: value for key, value in entry.items() if key != "extra"}
            steps.append({**args, "in": [cursor, *entry.get("extra", [])], "out": out})
        cursor = out


def _model_inputs(manifest: dict[str, Any]) -> str | list[str]:
    """The buffers a model step reads: the case's input group, then the buffer each further group of the
    case is read into, in the order the model's graph takes them."""
    reads = [read["group"] for read in manifest.get("reads") or []]
    return ["input", *reads] if reads else "input"


def _copy_manifest(manifest: dict[str, Any], ops: list[dict[str, Any]]) -> dict[str, Any]:
    """The manifest of one test-time copy: the draw's ops close its preprocessing and their inverses open
    its postprocessing, where konfai makes its copies from the case its chain prepared and puts each one
    back as it assembles it."""
    if not ops:
        return manifest
    drawn = [{**op, "id": f"__tta_{index}"} for index, op in enumerate(ops)]
    # konfai shares one draw across every group of a case, so a copy flips them all; only the first
    # group's is undone, since the output chain undoes that one alone.
    reads = [{**read, "preprocessing": [*read.get("preprocessing", []), *ops]} for read in manifest.get("reads") or []]
    return {
        **manifest,
        "preprocessing": [*manifest.get("preprocessing", []), *drawn],
        "postprocessing": [
            *(
                {"op": "inverse", "of": op["id"], **({"vector_field": True} if op.get("vector_field") else {})}
                for op in reversed(drawn)
            ),
            *manifest.get("postprocessing", []),
        ],
        **({"reads": reads} if reads else {}),
    }


def _mask_buffers(aux: dict[str, dict[str, Any]], grid: str, steps: list[dict[str, Any]]) -> dict[str, str]:
    """Append the ops that put each nested model's volume on ``grid``, and name the buffer each mask reads."""
    buffers: dict[str, str] = {}
    for name, nested in aux.items():
        cursor = name
        # konfai reads the file the nested app wrote, in the dtype its chain ends on when that is a cast.
        post = nested["manifest"].get("postprocessing") or []
        written = str(post[-1]["dtype"]) if post and post[-1]["op"] == "cast" else None
        ops = json.loads(json.dumps(nested.get("ops") or []))
        resolve_interpolation(ops, written)
        for index, step in enumerate(ops):
            out = f"{name}__{index}"
            if step["op"] == "resample":
                if "field_group" in step:
                    raise ConfigError(
                        f"'Resample' on group '{name}' warps through the field in '{step['field_group']}', and the"
                        " program puts a nested model's volume on the copies' grid by the two extents alone.",
                        _or_konfai("Drop the field from that Resample", _PREDICT),
                    )
                if "align" in step:
                    raise ConfigError(
                        f"'Resample' on group '{name}' aligns on '{step['align']}', and the program puts a nested"
                        " model's volume on the copies' grid the way align: extent places it.",
                        _or_konfai("Align that Resample on extent", _PREDICT),
                    )
                if "interpolation" not in step:
                    raise ConfigError(
                        f"'Resample' on group '{name}' states no interpolation, and konfai picks one from the"
                        " dtype the nested app writes, which the program reads only from a cast its chain ends on.",
                        "State interpolation on that Resample.",
                    )
                steps.append({"op": f"resample_{step['interpolation']}", "in": [cursor, grid], "out": out})
            else:
                steps.append({**step, "in": [cursor], "out": out})
            cursor = out
        buffers[name] = cursor
    return buffers


def derive_reduction(config: dict[str, Any], root: str, output: str | None = None) -> str | None:
    """The reduction the output's after-reduction chain opens with, ``None`` for none: ``merge_labels`` for a
    ``MergeLabels``, ``mean`` for an ``InferenceStack``, ``median_lower`` for one of ``mode: median``, which
    takes torch's lower middle where the ``Median`` reduction averages the middle pair."""
    spec = output_spec(config, root, output)
    after = list(_chain_of(spec, "after_reduction_transforms"))
    if after and base_name(after[0]) == "MergeLabels":
        return "merge_labels"
    stack, _ = _split_stack(config, root, spec)
    if stack is not None:
        return "median_lower" if stack.get("mode") == "median" else "mean"
    return None


def tta_passes(config: dict[str, Any], root: str) -> list[list[dict[str, Any]]]:
    """The identity and every test-time copy, each as the ops that make it from the input: flips, permutations
    and quarter turns of the spatial axes, which the runtime undoes exactly; any other draw is refused. A flip
    copy takes konfai's mirror, the distinct mirrors in turn; any other share between 0 and 1 is drawn by
    Python's generator from ``manual_seed``, so those copies are reproducible and not torch's."""
    from konfai.data.augmentation import Flip, Permute, Rotate
    from konfai.utils.utils import get_module

    predictor = config.get(root, {})
    augmentations = predictor.get("Dataset", {}).get("augmentations")
    if not isinstance(augmentations, dict) or not augmentations:
        return [[]]
    rng = random.Random(int(clean(predictor.get("manual_seed")) or 0))  # nosec B311 - TTA sampling, not security
    passes: list[list[dict[str, Any]]] = [[]]
    for group, aug in augmentations.items():
        draws: list[tuple[float, Any]] = []
        for name, params in chain_items(clean(aug.get("data_augmentations") if isinstance(aug, dict) else None)):
            try:
                module, class_name = get_module(name, "konfai.data.augmentation")
                drawn = getattr(module, class_name, None)
            except ImportError:
                drawn = None
            # The binder reads `prob` through Prob, separately from the draw's constructor.
            kept = {key: value for key, value in params.items() if key != "prob"}
            if drawn in (Flip, Permute) or (drawn is Rotate and bool(clean(kept.get("is_quarter", False)))):
                draws.append((float(params.get("prob", 1.0)), drawn(**kept)))
            else:
                detail = " on free angles, which resamples" if drawn is Rotate else ""
                raise ConfigError(
                    f"Test-time augmentation '{group}' contains '{name}'{detail}, which the portable runtime"
                    " cannot represent as a flip or a permutation of the spatial axes.",
                    _or_konfai("Use scalar Flip or Permute draws, or a Rotate with is_quarter: true", _PREDICT),
                )
        _warn_fractional_shares(group, draws)
        copies = int(aug.get("nb", 0))
        selected = [0] * len(draws)
        for copy in range(copies):
            passes.append(_drawn_pass(draws, rng, _spatial_rank(config, root), copy, copies, selected))
    return passes


def _warn_fractional_shares(group: str, draws: list[tuple[float, Any]]) -> None:
    """A share strictly between 0 and 1 is a draw: konfai makes it with torch's generator, the program with
    Python's from the same seed, so the copies are reproducible and not the ones konfai draws."""
    from konfai.data.augmentation import Permute

    shares = []
    for prob, draw in draws:
        name = type(draw).__name__
        shares.append((f"{name}.prob", [prob]))
        if isinstance(draw, Permute) and draw.prob_permute:
            shares.append((f"{name}.prob_permute", list(draw.prob_permute)))
    fractional = [key for key, values in shares if any(0 < float(value) < 1 for value in values)]
    if fractional:
        _warn(
            f"Test-time augmentation '{group}' draws {', '.join(fractional)} at a share between 0 and 1: the program's"
            " copies are reproducible, drawn by Python's generator from manual_seed, not torch's, so they are not the"
            " copies konfai draws."
        )


def _spatial_rank(config: dict[str, Any], root: str) -> int:
    """How many spatial axes the model's patch names, which is what a draw permutes."""
    return len(inference_patch(config, root).get("patch_size") or []) or 3


def _drawn_pass(
    draws: list[tuple[float, Any]], rng: random.Random, rank: int, copy: int, copies: int, selected: list[int]
) -> list[dict[str, Any]]:
    """One test-time copy, the ``copy``-th of ``copies``: each draw of the chain, in order, as the op it makes.
    ``selected`` counts, per draw, the earlier copies its gate let through."""
    from konfai.data.augmentation import Flip, Rotate

    ops: list[dict[str, Any]] = []
    for at, (prob, draw) in enumerate(draws):
        if prob <= 0 or (prob < 1 and rng.random() >= prob):
            continue
        slot, selected[at] = selected[at], selected[at] + 1
        if isinstance(draw, Flip):
            # As konfai's Flip in prediction: the copies its gate selects take, in turn, the distinct mirrors of
            # the axes f_prob allows.
            allowed = [axis for axis, share in enumerate(draw.f_prob) if float(share) > 0]
            mirrors = [list(axes) for size in range(1, len(allowed) + 1) for axes in combinations(allowed, size)]
            axes = mirrors[slot % len(mirrors)] if mirrors else []
            if axes:
                # A vector-field draw also negates the component channel of every mirrored axis, which
                # the runtime does where the volume holds one component per spatial axis, as konfai does.
                field = {"vector_field": True} if draw.vector_field else {}
                ops.append({"op": "flip", "axes": axes, **field})
            continue
        if isinstance(draw, Rotate):
            ops.extend(_quarter_turn(rng, rank, draw.in_plane))
            continue
        # konfai's Permute composes the fixed permutations it drew, each naming the source axis of every
        # output axis; prob_permute states each one's share.
        if rank != 3:
            raise ConfigError(
                f"a Permute draw permutes the three spatial axes; this patch has rank {rank}, which konfai refuses too.",
                _or_konfai("Draw with Flip or Rotate on a 2-D patch", _PREDICT),
            )
        orders = [[2, 1, 3], [3, 1, 2]]
        axes = list(range(3))
        if draw.prob_permute:
            for order, share in zip(orders, draw.prob_permute, strict=False):
                if rng.random() < float(share):
                    axes = [axes[dim - 1] for dim in order]
        else:
            # No shares: konfai draws exactly two copies, the first permuted by the first order alone and the
            # second by the second.
            if copies != 2:
                raise ConfigError(
                    f"Test-time augmentation 'Permute' with no prob_permute draws two copies, one per order,"
                    f" and this augmentation asks for {copies}.",
                    "Set nb: 2, or give prob_permute its two shares.",
                )
            axes = [axes[dim - 1] for dim in orders[copy]]
        if axes != [0, 1, 2]:
            ops.append({"op": "permute", "axes": axes})
    return ops


def _quarter_turn(rng: random.Random, rank: int, in_plane: bool = False) -> list[dict[str, Any]]:
    """A quarter-turn ``Rotate`` as the ops it is: konfai reads its matrix as a signed permutation of the axes
    and reorders then mirrors them, a ``permute`` then a ``flip``. An ``in_plane`` draw of a 3-D case turns
    about the slice axis alone."""
    import torch

    from konfai.data.augmentation.spatial import (
        SIGNED_PERMUTATION_ATOL_FLOAT32,
        _rotation_2d_matrix,
        _rotation_3d_matrix,
    )
    from konfai.data.geometry import signed_permutation

    turns = [rng.choice([90.0, 180.0, 270.0]) for _ in range(rank)]  # nosec B311 - a test-time copy
    if in_plane and rank == 3:
        turns[:2] = [0.0, 0.0]
    rotation = _rotation_3d_matrix if rank == 3 else _rotation_2d_matrix
    matrix = rotation(torch.deg2rad(torch.tensor(turns)))
    remap = signed_permutation(matrix[:-1, :-1], SIGNED_PERMUTATION_ATOL_FLOAT32)
    if remap is None:
        raise ConfigError(f"a quarter turn of {turns} degrees is no permutation of the axes")
    ops: list[dict[str, Any]] = []
    axes = [source for source, _ in remap]
    if axes != list(range(rank)):
        ops.append({"op": "permute", "axes": axes})
    mirrored = [axis for axis, (_, flip) in enumerate(remap) if flip]
    if mirrored:
        ops.append({"op": "flip", "axes": mirrored})
    return ops


def export_checkpoints(
    config: dict[str, Any],
    root: str,
    checkpoints: list[Path | None],
    out_dir: Path,
    ids: list[str],
    *,
    head: str | None = None,
    patch_size: list[int] | None = None,
    in_channels: int | list[int] | None = None,
    write_manifest: bool = False,
    masked: bool = False,
    output: str | None = None,
) -> tuple[list[dict[str, Any]], list[Path], int]:
    """Export the config's model once per checkpoint to ``<out_dir>/<id>.onnx``, in the ``KONFAI_*`` environment
    the caller set; return each model as ``{id, manifest}``, the files written and how many post ops run per
    copy. ``patch_size`` and ``in_channels`` (one count per group the model reads) override the config; a
    ``None`` checkpoint is the weights the model builds with; ``masked``: the caller applies the Masks."""
    import torch

    from konfai.export import export_to_onnx, select_inference_head

    derived_patch, derived_channels, extend_slice, pad_value = derive_onnx_params(config, root)
    patch_size = patch_size or derived_patch
    if not patch_size:
        raise ConfigError("could not derive patch_size from the config; pass --patch-size")
    # One count per group the model reads, in the order its graph takes them. A group past the first
    # holds one value per voxel unless the caller states otherwise.
    stated = [in_channels] if isinstance(in_channels, int) else list(in_channels or [])
    inputs = input_groups(config, root)
    channels = stated[0] if stated else derived_channels
    others = list(islice(chain(stated[1:], repeat(1)), max(len(inputs), 1) - 1))
    patch = inference_patch(config, root)
    overlap = derive_overlap(patch, patch_size) if patch else None
    extra, folds, per_copy = transform_manifest(config, root, masked=masked, output=output, others=others)
    blend = derive_blend(config, root, output)
    check_blend(overlap, blend)
    if blend is not None:
        extra["blend"] = blend
    # konfai runs `batch_size` patches through each forward; the runtime batches them alike where the
    # device holds them.
    batch = int(clean((config.get(root, {}).get("Dataset") or {}).get("batch_size")) or 1)

    classpath = config[root]["Model"]["classpath"]
    models: list[dict[str, Any]] = []
    written: list[Path] = []
    for checkpoint, model_id in zip(checkpoints, ids, strict=True):
        model = _loaded_model(classpath, checkpoint)
        # The input channel count is intrinsic to the architecture when the config does not state it.
        channels = channels or getattr(model, "in_channels", None)
        if not channels:
            raise ConfigError("could not derive in_channels from the config or model; pass --in-channels")
        examples = tuple(torch.randn(1, count, *patch_size) for count in [channels, *others])
        # Default to the terminal float head; the graph's last output is often an integer Argmax.
        path, manifest = export_to_onnx(
            model,
            out_dir,
            examples,
            head or select_inference_head(model, *examples),
            input_group=inputs[0][0] if inputs else "Volume_0",
            patch_overlap=overlap,
            extend_slice=extend_slice,
            pad_value=pad_value,
            extra_manifest=dict(extra),
            fold_pre=folds,
            model_filename=f"{model_id}.onnx",
            write_manifest=write_manifest,
        )
        manifest["patch"]["batch"] = batch
        models.append({"id": model_id, "manifest": manifest})
        written.append(path)
    return models, written, per_copy


def _loaded_model(classpath: str, checkpoint: Path | None) -> Any:
    """The per-patch network the runtime tiles a case with, in eval mode, holding ``checkpoint``'s weights or,
    for ``None``, the ones it builds with."""
    from konfai.network.network import ModelLoader
    from konfai.utils.runtime import safe_torch_load

    model = ModelLoader(classpath).get_model(train=False)
    # A ModelPatch tiles inside the network, which the export would drop.
    if model.patch is not None:
        raise ConfigError(
            "the model tiles its input by a ModelPatch, which the export drops: the exported graph would"
            " run the whole inference patch through the network in one forward.",
            _or_konfai("Set ModelPatch to None", _PREDICT),
        )
    model.eval()
    if checkpoint is not None:
        model.load(safe_torch_load(checkpoint, "cpu"), init=False)
    return model


def aux_mask_groups(config: dict[str, Any], root: str) -> dict[str, dict[str, Any]]:
    """Every group a nested model builds (``KonfAIInference``, then a resample or a dilate), as ``{group:
    {"inference": <its params>, "ops": [<steps>]}}``. Any other stage altering the group's values is refused
    by name: the program would build the mask without it."""
    groups: dict[str, dict[str, Any]] = {}
    for _src, name, group in _groups(config, root):
        if group.get("is_input", True):
            continue
        inference, ops, unbuilt = None, [], []
        for tname, params in chain_items(group.get("transforms")):
            base = base_name(tname)
            if base == "KonfAIInference":
                inference = params
            elif base in ("Resample", "Dilate"):
                ops.append(_admitted(OP_MAP[base](params)))
            elif not (records_only(tname) or is_save(tname)):
                unbuilt.append(tname)
        if inference is None:
            continue
        if unbuilt:
            raise ConfigError(
                f"transform '{unbuilt[0]}' on the nested model's group '{name}' has no step in the masked"
                " program, which resamples and dilates that group only.",
                _or_konfai("Remove it from the group's transforms", _PREDICT),
            )
        groups[name] = {"inference": inference, "ops": ops}
    return groups


def read_nested_groups(config: dict[str, Any], root: str, outputs: list[str]) -> dict[str, dict[str, Any]]:
    """The nested models' groups (:func:`aux_mask_groups`) a ``Mask`` of one of ``outputs`` reads: a nested
    model no output reads would be exported, and loaded by the runtime, for nothing."""
    read = {spec["group"] for output in outputs for spec in mask_specs(config, root, output)}
    return {name: group for name, group in aux_mask_groups(config, root).items() if name in read}


def mask_specs(config: dict[str, Any], root: str, output: str | None = None) -> list[dict[str, Any]]:
    """``Mask`` ops in ``before_reduction`` (masking the primary output by an auxiliary group's buffer):
    ``[{"group": <aux group name>, "value_outside": float}]``. A Mask reading a group no nested model builds
    is refused by name: the program holds no buffer for it."""
    before = _chain_of(output_spec(config, root, output), "before_reduction_transforms")
    built = aux_mask_groups(config, root)
    specs = []
    for name, params in before.items():
        if base_name(name) != "Mask":
            continue
        params = _mapping(params)
        path = str(params.get("path", "./default.mha"))  # konfai's Mask default
        if path not in built:
            raise ConfigError(
                f"transform '{name}' masks by '{path}', which no nested model on the config builds, and the masked"
                " program holds a buffer for those groups only.",
                _or_konfai("Name a group a KonfAIInference produces", _PREDICT),
            )
        specs.append({"group": path, "value_outside": float(params.get("value_outside", 0))})
    return specs


def program_reduction(
    config: dict[str, Any], root: str, models: int, passes: list[list[dict[str, Any]]], output: str | None = None
) -> str:
    """The op a prediction program folds its copies with: the reduction :func:`derive_reduction` reads,
    else the models' ``combine``, which the test-time copies must fold by too. The program folds each
    model's blended, transformed output, and anything konfai folds elsewhere is refused by name."""
    spec = output_spec(config, root, output)
    combine = _class_name(str(clean(config.get(root, {}).get("combine")) or "Mean"))
    derived = derive_reduction(config, root, output)
    if models > 1 and combine != "Concat":
        # konfai combines the models inside the forward, on each batch of patches: before the overlap
        # blend, before the before-reduction transforms, and without the per-model layout MergeLabels
        # reads. Of those folds only a mean commutes with the blend, up to konfai's float16 rounding.
        if combine != "Mean" or derived is not None:
            raise ConfigError(
                f"{models} models combine by '{combine}', which konfai applies to each batch of patches before"
                " the overlap blend and the before-reduction transforms, and a program folds each model's"
                " blended output.",
                _or_konfai(
                    "Combine by Concat (with an InferenceStack or MergeLabels after the reduction), or by Mean",
                    _PREDICT,
                ),
            )
    copies = _class_name(str(clean(spec.get("reduction")) or "Mean"))
    if models_fold_first(config, root, models, output):
        return copies.lower()
    # One model's combine hands its output back, so the program's only fold is the copies' own.
    reduction = derived or (copies if models == 1 and len(passes) > 1 else combine).lower()
    # A derived reduction folds every copy at once: `_split_stack` refuses a stack they do not reach stacked.
    if len(passes) > 1 and derived is None and copies.lower() != reduction:
        raise ConfigError(
            f"The models combine by '{combine}' and the test-time copies by '{copies}', and a program"
            " folds all copies with one reduction.",
            _or_konfai("Use the same reduction for both", _PREDICT),
        )
    return reduction


def models_fold_first(config: dict[str, Any], root: str, models: int, output: str | None = None) -> bool:
    """Whether a program averages each test-time copy's models before that copy's before-reduction chain, as
    konfai's ``combine: Mean`` does: the chain (a cast that truncates, a softmax) does not commute with the
    mean, so it runs once per copy, on the average, and the copies then fold by the output's ``reduction``."""
    combine = _class_name(str(clean(config.get(root, {}).get("combine")) or "Mean"))
    spec = output_spec(config, root, output)
    before, _, _ = compile_pipeline(_chain_of(spec, "before_reduction_transforms"), consumed=("Mask",))
    return models > 1 and combine == "Mean" and bool(before) and derive_reduction(config, root, output) is None


def compile_output(
    config: dict[str, Any],
    root: str,
    checkpoints: list[Path | None],
    out_dir: Path,
    ids: list[str],
    *,
    passes: list[list[dict[str, Any]]],
    aux: dict[str, dict[str, Any]],
    head: str | None = None,
    output: str | None = None,
    patch_size: list[int] | None = None,
    in_channels: int | None = None,
    write_manifest: bool = False,
) -> tuple[list[dict[str, Any]], list[Path], dict[str, Any]]:
    """One output of the config, compiled: its checkpoints exported to ``<out_dir>/<id>.onnx`` and the program
    that runs them with the nested models of ``aux``. ``head`` is the module output the graph ends on and
    ``output`` the config output whose chains the program runs (``None``: the default, the only one). The
    konfai-apps bundle export reads it too, so a bundle carries the program a PREDICTION program does."""
    masks = mask_specs(config, root, output) if aux else []
    models, written, per_copy = export_checkpoints(
        config,
        root,
        checkpoints,
        out_dir,
        ids,
        head=head,
        patch_size=patch_size,
        in_channels=in_channels,
        write_manifest=write_manifest,
        masked=bool(aux),
        output=output,
    )
    reduce = program_reduction(config, root, len(checkpoints), passes, output)
    fold = models_fold_first(config, root, len(checkpoints), output)
    stack, _ = _split_stack(config, root, output_spec(config, root, output))
    program = assemble_program(
        models,
        reduce=reduce,
        passes=passes,
        aux=aux,
        masks=masks,
        per_copy=per_copy,
        fold_models=fold,
        stack=None if stack is None else str(stack.get("mode", "mean")),
        checkpoints=names if len(names := [c.name for c in checkpoints if c is not None]) == len(checkpoints) else None,
    )
    return models, written, program


def _portable_root(path: str | Path) -> str:
    """A path the runtime resolves against its ``--root``: relative to the working directory when it
    sits below it, with slashes whatever the system compiling, absolute otherwise."""
    import os

    text = str(path).rstrip("/") or "/"
    try:
        relative = os.path.relpath(text, Path.cwd())
    except ValueError:  # on another Windows drive: no relative path reaches it
        return text
    return text if relative.startswith("..") else Path(relative).as_posix()


def compile_prediction(
    prediction_file: str | Path = Path("./Prediction.yml"),
    models: list[str | Path] | None = None,
    predictions_dir: str | Path = Path("./Predictions"),
    output: str | Path = Path("./prediction.json"),
    nested_exporter: Callable[[str, dict[str, Any], Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compile a ``Prediction.yml`` and its checkpoints into the program the portable runtime executes, each
    checkpoint exported to ONNX beside ``output`` and each output landing where PREDICTION would write it. A
    nested app (a ``KonfAIInference`` group a ``Mask`` reads) needs ``nested_exporter``, which resolves that
    app and writes its ONNX: only konfai-apps resolves an app."""
    from konfai.predictor.workflow import Predictor, build_predict

    if not models:
        raise ConfigError("compile_prediction needs the checkpoints to export.", "Pass models=[...], as --models.")
    root = "Predictor"
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoints = [Path(model).resolve() for model in models]

    with _built(
        prediction_file,
        lambda file: build_predict(models=checkpoints, prediction_file=file, predictions_dir=Path(predictions_dir)),
    ) as (built, resolved):
        predictor = cast(Predictor, built)
        warn_unportable_keys(resolved, root)
        dataset_section = (resolved.get(root) or {}).get("Dataset") or {}

        # The groups the model reads, in the order it takes them: its first input is branch '0'.
        inputs = [_literal_group(group_src, ConfigError) for group_src, _group in input_groups(resolved, root)]
        if not inputs:
            raise ConfigError(
                "The config feeds the model no input group.",
                "Mark the group the model reads with is_input: true.",
            )
        passes = tta_passes(resolved, root)
        aux = _nested_models(resolved, root, list(predictor.outputs_dataset), nested_exporter, output.parent)
        exported, written = _compile_heads(
            resolved, root, predictor.outputs_dataset, checkpoints, passes, aux, output.parent
        )
        predictor.dataset.prepare()
        # What each case is read from: the model's first input group, then every group the manifest names,
        # which is the order the runtime fills the program's buffers in.
        reads = [read["group"] for read in (exported[0]["manifest"].get("reads") or [])] if exported else []
        prediction = {
            "konfai_edge_prediction": PREDICTION_PROGRAM_VERSION,
            "name": predictor.name,
            "memory_budget": _budget_bytes(dataset_section.get("memory_budget")),
            # konfai runs the forward in float16 under autocast; a runtime with float16 runs it there too.
            "autocast": bool(clean((resolved.get(root) or {}).get("autocast"))),
            "sources": [source_of(entry) for entry in dataset_section.get("dataset_filenames") or []],
            "cases": list(predictor.dataset.case_names),
            "input_group": [inputs[0], *reads] if reads else inputs[0],
            "models": {model["id"]: f"{model['id']}.onnx" for model in [*aux.values(), *exported]},
            "outputs": written,
        }

    output.write_text(json.dumps(prediction, indent=2))
    return prediction


def _nested_models(
    config: dict[str, Any],
    root: str,
    outputs: list[str],
    nested_exporter: Callable[[str, dict[str, Any], Path], dict[str, Any]] | None,
    out_dir: Path,
) -> dict[str, dict[str, Any]]:
    """The nested models a ``Mask`` of one of ``outputs`` reads, exported into ``out_dir``, each with the ops
    that build its buffer. They are read before anything is exported, so a chain the program cannot run is
    refused first."""
    groups = read_nested_groups(config, root, outputs)
    if not groups:
        return {}
    if nested_exporter is None:
        raise ConfigError(
            f"the group '{next(iter(groups))}' is produced by a nested app, which this compiler does not resolve.",
            "Pass nested_exporter=konfai_apps.bundle.export_nested_model, which resolves an app reference.",
        )
    return {
        name: {**nested_exporter(name, group["inference"], out_dir), "ops": group["ops"]}
        for name, group in groups.items()
    }


def _compile_heads(
    config: dict[str, Any],
    root: str,
    heads: dict[str, Any],
    checkpoints: list[Path],
    passes: list[list[dict[str, Any]]],
    aux: dict[str, dict[str, Any]],
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One program per output sink of ``heads``: each names the head it reads, its own models and its own
    destination. Returns every model exported and every output as its program and its write target."""
    exported: list[dict[str, Any]] = []
    written: list[dict[str, Any]] = []
    for head, sink in heads.items():
        prefix = "" if len(heads) == 1 else f"{_file_stem(head)}_"
        ids = (
            [f"{prefix}model"]
            if len(checkpoints) == 1
            else [f"{prefix}model_{index}" for index in range(len(checkpoints))]
        )
        models, _files, program = compile_output(
            config, root, list[Path | None](checkpoints), out_dir, ids, passes=passes, aux=aux, head=head, output=head
        )
        exported += models
        destination = _destination(_portable_root(sink.filename), sink.file_format, sink.group)
        written.append({"program": program, "write": destination} | _stack_write(config, root, head, sink))
    return exported, written


def _stack_write(config: dict[str, Any], root: str, head: str, sink: Any) -> dict[str, Any]:
    """Where the output's ``InferenceStack`` writes the copies, as konfai names it: its ``dataset``, else the
    output's, under the group ``InferenceStack``; nothing for an output that stacks none."""
    from konfai.utils.utils import split_path_spec

    stack, _ = _split_stack(config, root, output_spec(config, root, head))
    if stack is None:
        return {}
    dataset = clean(stack.get("dataset"))
    if dataset:
        filename, _flag, file_format = split_path_spec(str(dataset), default_format="mha")
    else:
        filename, file_format = sink.filename, sink.file_format
    return {"stack": _destination(_portable_root(filename), file_format, "InferenceStack")}

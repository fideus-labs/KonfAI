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

"""KonfAI in Python: the workflows as callables.

:func:`transform` (with :func:`plan_transform`, its dry-run twin), :func:`evaluate`,
:func:`predict` and :func:`train` build the config tree the YAML file would hold and hand it to
the same binder: a chain is a list of live stage objects (constructor arguments recorded as given,
see ``record_given_arguments``), the equivalent mapping, or a tree loaded from a YAML and modified.

The contract:

- A designed refusal raises ``KonfAIError``; only the CLI catches and exits.
- Results come back structured, read from the run's own record (``outputs.json``, ``Metric_*.json``).
- The ``KONFAI_*`` environment is restored around every call; one workflow runs at a time per
  process, a second concurrent call is refused.
- Every call materializes the resolved YAML in the run's workspace, the record of the experiment.
"""

import importlib
import json
import os
import shutil
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from konfai.utils.errors import ConfigError

if TYPE_CHECKING:
    from konfai.bundle import BundleImport
    from konfai.transformer import TransformPlan
    from konfai.utils.catalog import Component
    from konfai.utils.runtime import DistributedObject

_T = TypeVar("_T")

#: Where a bare stage name resolves, per stage family: the same rule the YAML loader applies.
_STAGE_MODULES = ("konfai.data.transform", "konfai.data.augmentation")
_CRITERION_MODULES = ("konfai.metric.measure",)

_ACTIVE = threading.Lock()


@contextmanager
def _one_workflow_at_a_time(ranks: int) -> Iterator[None]:
    """Serialize workflows within the process and leave the environment as found. Two in-process runs
    would corrupt the process-wide ``KONFAI_*`` state, so a second is refused. ``ranks`` is exported
    as ``KONFAI_LOCAL_RANKS`` for build-time budget sizing."""
    if not _ACTIVE.acquire(blocking=False):
        raise ConfigError(
            "A KonfAI workflow is already running in this process.",
            "Wait for it to return, or run concurrent workflows in separate processes: the engine"
            " keys its state on process-wide KONFAI_* variables, so two in-process runs would"
            " corrupt each other.",
        )
    saved = {key: value for key, value in os.environ.items() if key.startswith("KONFAI")}
    os.environ["KONFAI_LOCAL_RANKS"] = str(max(1, ranks))
    try:
        yield
    finally:
        try:
            from konfai.utils.dataset import release_read_handles  # lazy: api.py stays light to import

            release_read_handles()
        finally:
            # Whatever a handle's close raised, the caller gets its environment and the lock back.
            for key in [key for key in os.environ if key.startswith("KONFAI")]:
                if key not in saved:
                    del os.environ[key]
            os.environ.update(saved)
            _ACTIVE.release()


@contextmanager
def _workflow_scope(ranks: int) -> Iterator[None]:
    """Own build-time RNG draws and scratch files until execution and result extraction finish."""
    from konfai.utils.runtime.distributed import preserved_rng
    from konfai.utils.runtime.environment import _SCRATCH_CONFIGS, release_scratch_configs

    with _one_workflow_at_a_time(ranks), preserved_rng():
        mark = len(_SCRATCH_CONFIGS)
        try:
            yield
        finally:
            release_scratch_configs(mark)


def _launch(
    ranks: int,
    build: "Callable[[], DistributedObject]",
    finish: "Callable[[DistributedObject], _T]",
    *,
    gpu: Sequence[int] | None,
    cpu: int | None,
    overwrite: bool,
    quiet: bool,
) -> _T:
    """Take the workflow lock, build, execute, and read the result out through ``finish``, which runs
    inside the lock: the workspace lives in ``KONFAI_*`` variables the lock's exit restores."""
    from konfai.utils.clock import restart_startup_clock
    from konfai.utils.runtime import execute_distributed_object

    with _workflow_scope(ranks):
        with restart_startup_clock().phase("build"):  # this call's own clock, not the previous workflow's
            workflow = build()
        execute_distributed_object(workflow, gpu=list(gpu or []), cpu=cpu, overwrite=overwrite, quiet=quiet)
        return finish(workflow)


def _yaml_safe(value: object, where: str) -> object:
    """``value`` as the config file could hold it: or a refusal that names the argument."""
    # Before the Python scalars: np.float64 is a float subclass, and ruamel refuses it.
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _yaml_safe(entry, f"{where}.{key}") for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(entry, f"{where}[{index}]") for index, entry in enumerate(value)]
    raise ConfigError(
        f"'{where}' is a {type(value).__name__}, which a config tree cannot hold.",
        "A stage's constructor arguments must be YAML-spellable (numbers, strings, paths, lists,"
        " mappings): the same rule the YAML file obeys.",
    )


def _public_module(cls: type, default_modules: tuple[str, ...]) -> str:
    """The module a config names ``cls`` by: the shipped package that re-exports it, else its own."""
    return next(
        (m for m in default_modules if cls.__module__ == m or cls.__module__.startswith(m + ".")), cls.__module__
    )


def _classpath(stage: object, default_modules: tuple[str, ...]) -> str:
    """The name the config tree references ``stage`` by: bare for a shipped class, qualified else."""
    cls = type(stage)
    module = _public_module(cls, default_modules)
    return cls.__name__ if module in default_modules else f"{module}:{cls.__name__}"


def _qualified_spelling(stage: object, name: str, default_modules: tuple[str, ...]) -> str:
    """A repeated bare name's second spelling: module-qualified, resolved as the binder resolves it."""
    if not isinstance(stage, Mapping):
        return f"{_public_module(type(stage), default_modules)}:{type(stage).__name__}"
    for module in default_modules:
        if hasattr(importlib.import_module(module), name):
            return f"{module}:{name}"
    raise ConfigError(
        f"'{name}' appears twice and resolves in no default module, so its second spelling is unknown.",
        "Spell the second occurrence with its module: {'my_module:" + name + "': {...}}.",
    )


def _stage_entry(stage: object, default_modules: tuple[str, ...], where: str) -> tuple[str, object]:
    """One chain entry: ``(classpath, kwargs-subtree)`` from a live object or a mapping."""
    if isinstance(stage, Mapping):
        if len(stage) != 1:
            raise ConfigError(
                f"'{where}' is a mapping of {len(stage)} stages; a chain entry holds exactly one.",
                "Spell each stage as its own entry: [{'Clip': {...}}, {'Write': {...}}]: or"
                " instantiate the classes and pass the objects.",
            )
        ((name, kwargs),) = stage.items()
        return str(name), _yaml_safe(kwargs, f"{where}.{name}")
    given = getattr(stage, "_konfai_given", None)
    if given is None:
        raise ConfigError(
            f"'{where}' ({type(stage).__name__}) records no constructor arguments.",
            "A chain stage is a Transform, DataAugmentation or Criterion subclass instance: their"
            " bases record what the constructor was given: or a one-entry mapping"
            " {'Name': {...kwargs...}}.",
        )
    name = _classpath(stage, default_modules)
    return name, {key: _yaml_safe(value, f"{where}.{name}.{key}") for key, value in given.items()}


def _chain_tree(stages: object, default_modules: tuple[str, ...], where: str) -> dict[str, object]:
    """A chain (a sequence of stages), as the mapping the config tree holds, in order: the second
    occurrence of a class is written module-qualified, from the third on under an occurrence key
    (``Clip#3``); an already-qualified classpath keeps its module in every occurrence key."""
    if isinstance(stages, Mapping):  # a chain already spelled as its tree
        return {str(key): _yaml_safe(value, f"{where}.{key}") for key, value in stages.items()}
    tree: dict[str, object] = {}
    occurrences: dict[str, int] = {}
    for index, stage in enumerate(_stage_sequence(stages, where)):
        classpath, kwargs = _stage_entry(stage, default_modules, f"{where}[{index}]")
        occurrences[classpath] = occurrences.get(classpath, 0) + 1
        key = classpath
        if key in tree and ":" not in classpath:
            key = _qualified_spelling(stage, classpath, default_modules)
        if key in tree:
            key = f"{classpath}#{occurrences[classpath]}"
            while key in tree:
                occurrences[classpath] += 1
                key = f"{classpath}#{occurrences[classpath]}"
        tree[key] = kwargs
    return tree


def _stage_sequence(stages: object, where: str) -> Sequence[object]:
    if isinstance(stages, Sequence) and not isinstance(stages, (str, bytes)):
        return stages
    raise ConfigError(
        f"'{where}' is a {type(stages).__name__}; a chain is a sequence of stages.",
        "Pass the stages in application order: [Clip(min_value=0), Write(dataset='./Out:mha')].",
    )


def list_components(kind: str) -> "list[Component]":
    """Enumerate the shipped components of one kind, spelled as a YAML config references them.

    ``kind`` is ``transform``, ``augmentation``, ``criterion``, ``reduction``, ``model`` or ``block``
    (plural spellings accepted). Records carry ``name``, ``config_reference``, ``module`` and ``doc``.
    """
    from konfai.utils.catalog import list_components as _list_components

    return _list_components(kind)


def _dataset_filenames(datasets: str | Path | Sequence[str | Path]) -> list[str]:
    entries = [datasets] if isinstance(datasets, (str, Path)) else list(datasets)
    if not entries:
        raise ConfigError(
            "No dataset was given.",
            "Name at least one root, as the YAML would: './Dataset:mha' (path, then format).",
        )
    return [str(entry) for entry in entries]


# ------------------------------------------------------------------------------------- TRANSFORM


@dataclass(frozen=True)
class TransformResult:
    """What a TRANSFORM run produced, in the run's own terms."""

    #: The run directory (``Transforms/<name>``): logs, the resolved config, ``outputs.json``; never the data.
    workspace: Path
    #: Every chain's terminal ``Write``: ``{group_src, group_dest, dataset, path, group, format}``
    #: (``dataset`` as a config names the root, ``path`` as it is on disk: the ``.h5`` file itself).
    outputs: list[dict[str, str]]
    #: The resolved config the run kept: copy this file to version the experiment.
    config: Path


def _transform_tree(
    name: str,
    datasets: str | Path | Sequence[str | Path],
    chains: Mapping[str, Mapping[str, object]],
    memory_budget: str | int | None,
    on_fallback: str,
    manual_seed: int,
    dataset_options: Mapping[str, object] | None,
) -> dict:
    groups_src: dict[str, object] = {}
    for group_src, destinations in chains.items():
        groups_dest = {
            str(group_dest): {"transforms": _chain_tree(stages, _STAGE_MODULES, f"chains.{group_src}.{group_dest}")}
            for group_dest, stages in destinations.items()
        }
        groups_src[str(group_src)] = {"groups_dest": groups_dest}
    dataset_tree: dict[str, object] = {"dataset_filenames": _dataset_filenames(datasets), "groups_src": groups_src}
    if memory_budget is not None:
        dataset_tree["memory_budget"] = memory_budget
    dataset_tree.update(dict(dataset_options or {}))
    return {
        "Transformer": {
            "name": name,
            "on_fallback": on_fallback,
            "manual_seed": manual_seed,
            "Dataset": dataset_tree,
        }
    }


def transform(
    name: str,
    datasets: str | Path | Sequence[str | Path],
    chains: Mapping[str, Mapping[str, object]],
    *,
    memory_budget: str | int | None = None,
    on_fallback: str = "warn",
    manual_seed: int = 0,
    dataset_options: Mapping[str, object] | None = None,
    gpu: Sequence[int] | None = None,
    cpu: int = 1,
    quiet: bool = False,
    overwrite: bool = False,
    transforms_dir: Path | str = Path("./Transforms"),
) -> TransformResult:
    """Run a TRANSFORM workflow: read a dataset, apply each chain, ``Write`` the results.

    ``chains`` maps ``group_src -> group_dest -> chain``, where a chain is a list of stage objects
    (``[Resample(...), Write(dataset='./Out:mha')]``), of one-entry mappings, or the equivalent
    mapping tree. Every chain ends in a ``Write``. GPU is opt-in (``gpu=[0]``).
    """
    tree = _transform_tree(name, datasets, chains, memory_budget, on_fallback, manual_seed, dataset_options)
    from konfai.transformer import build_transform

    workspace, config_name = _launch(
        len(gpu or []) or cpu,
        lambda: build_transform(transform_file=tree, transforms_dir=transforms_dir),
        lambda workflow: (
            Path(workflow.transform_path),  # type: ignore[attr-defined]
            Path(os.environ["KONFAI_config_file"]).name,
        ),
        gpu=gpu,
        cpu=cpu,
        overwrite=overwrite,
        quiet=quiet,
    )
    outputs = json.loads((workspace / "outputs.json").read_text(encoding="utf-8"))
    return TransformResult(workspace=workspace, outputs=outputs, config=workspace / config_name)


def plan_transform(
    name: str,
    datasets: str | Path | Sequence[str | Path],
    chains: Mapping[str, Mapping[str, object]],
    *,
    memory_budget: str | int | None = None,
    on_fallback: str = "warn",
    manual_seed: int = 0,
    dataset_options: Mapping[str, object] | None = None,
    gpu: Sequence[int] | None = None,
    cpu: int = 1,
    overwrite: bool = False,
    quiet: bool = False,
    transforms_dir: Path | str = Path("./Transforms"),
) -> "TransformPlan":
    """:func:`transform`'s dry-run twin: build, plan, print, return the plan; the run never starts.
    The ``TransformPlan`` is the run's own routing (STREAM/LOAD/WHOLE-VOLUME/SKIP/REDUCE)."""
    tree = _transform_tree(name, datasets, chains, memory_budget, on_fallback, manual_seed, dataset_options)
    from konfai.transformer import plan_transform as _plan_transform

    ranks = len(gpu or []) or cpu
    with _workflow_scope(ranks):
        return _plan_transform(
            transform_file=tree,
            transforms_dir=transforms_dir,
            gpu=list(gpu) if gpu is not None else None,
            cpu=cpu,
            quiet=quiet,
            overwrite=overwrite,
        )


# ------------------------------------------------------------------------------------ EVALUATION


@dataclass(frozen=True)
class EvaluationResult:
    """What an EVALUATION run measured, parsed from its own record."""

    #: The run directory (``Evaluations/<name>``), holding the ``Metric_*.json`` files.
    workspace: Path
    #: The parsed metric reports, keyed by split (``TRAIN``, ``VALIDATION``) when present.
    metrics: dict[str, Any]


def evaluate(
    name: str,
    datasets: str | Path | Sequence[str | Path],
    metrics: Mapping[str, Mapping[str, object]],
    *,
    transforms: Mapping[str, object] | None = None,
    dataset_options: Mapping[str, object] | None = None,
    gpu: Sequence[int] | None = None,
    cpu: int = 1,
    quiet: bool = False,
    overwrite: bool = False,
    evaluations_dir: Path | str = Path("./Evaluations"),
) -> EvaluationResult:
    """Run an EVALUATION workflow and return the measured metrics.

    ``metrics`` maps ``output_group -> target_group -> criteria``, where criteria is a list of
    :class:`~konfai.metric.measure.Criterion` instances (``[MAE(), Dice(labels=[1, 2])]``) or
    one-entry mappings. ``transforms`` optionally names a pre-metric chain per group; the groups
    themselves are derived from ``metrics``.
    """
    # A composite target ("Seg;Mask") names several dataset groups: split it so each is loaded.
    groups = sorted(
        {str(group) for group in metrics}
        | {part for targets in metrics.values() for target in targets for part in str(target).split(";")}
    )
    groups_src: dict[str, object] = {}
    for group in groups:
        # An undeclared chain is spelled None, never left out: the binder materializes its own default
        # (Normalize) for an absent key, which erases the difference the metrics measure.
        declared = None if transforms is None else transforms.get(group)
        chain: object = "None" if declared is None else _chain_tree(declared, _STAGE_MODULES, f"transforms.{group}")
        groups_src[group] = {"groups_dest": {group: {"transforms": chain}}}
    metrics_tree = {
        str(output): {
            "targets_criterions": {
                str(target): {
                    "criterions_loader": _chain_tree(criteria, _CRITERION_MODULES, f"metrics.{output}.{target}")
                }
                for target, criteria in targets.items()
            }
        }
        for output, targets in metrics.items()
    }
    dataset_tree: dict[str, object] = {"dataset_filenames": _dataset_filenames(datasets), "groups_src": groups_src}
    dataset_tree.update(dict(dataset_options or {}))
    tree = {"Evaluator": {"train_name": name, "metrics": metrics_tree, "Dataset": dataset_tree}}

    from konfai.evaluator import build_evaluate

    workspace = _launch(
        len(gpu or []) or cpu,
        lambda: build_evaluate(evaluations_file=tree, evaluations_dir=evaluations_dir),
        lambda _: Path(os.environ["KONFAI_EVALUATIONS_DIRECTORY"]) / name,
        gpu=gpu,
        cpu=cpu,
        overwrite=overwrite,
        quiet=quiet,
    )
    reports = {
        split: json.loads(report.read_text(encoding="utf-8"))
        for split in ("TRAIN", "VALIDATION")
        if (report := workspace / f"Metric_{split}.json").is_file()
    }
    return EvaluationResult(workspace=workspace, metrics=reports)


def _config_copy(config: "Mapping[str, object] | Path | str") -> "dict[str, object] | Path":
    """The caller's config, in a form this call may consume.

    Reading a KonfAI config resolves and REWRITES it. A tree is passed through; a caller's FILE is
    copied to scratch (released when the API call returns), so the write-back never lands on it. A
    model YAML named by a relative path resolves next to the config file that names it
    (``ModelLoader._yaml_path``), so the path is made absolute first: against the caller's file for
    a file, the working directory for a tree.
    """
    if isinstance(config, Mapping):
        # Through _yaml_safe: an np.float64 or Path value fails as a named refusal, not a ruamel error.
        tree = _yaml_safe(dict(config), "config")
        _anchor_model_paths(tree, Path.cwd())
        return tree  # type: ignore[return-value]
    from ruamel.yaml import YAML

    from konfai.utils.runtime.environment import register_scratch_config

    source = Path(config)
    scratch = Path(tempfile.mkdtemp(prefix="konfai_config_"))
    register_scratch_config(scratch)
    copy = scratch / source.name
    yaml = YAML()
    yaml.width = 4096  # a long absolute path stays on its line
    with source.open("r", encoding="utf-8") as file:
        tree = yaml.load(file)
    if _anchor_model_paths(tree, source.resolve().parent):
        with copy.open("w", encoding="utf-8") as file:
            yaml.dump(tree, file)
    else:
        shutil.copy2(source, copy)
    return copy


def _anchor_model_paths(tree: object, base: Path) -> bool:
    """Make every relative model-YAML ``classpath`` under ``tree`` absolute against ``base``;
    whether any changed. The catalog spelling (``default|Name.yml``) is not a path."""
    changed = False
    if isinstance(tree, dict):
        classpath = tree.get("classpath")
        if (
            isinstance(classpath, str)
            and "|" not in classpath
            and Path(classpath).suffix.lower() in {".yaml", ".yml"}
            and not Path(classpath).is_absolute()
        ):
            tree["classpath"] = str((base / classpath).resolve())
            changed = True
        for value in tree.values():
            changed = _anchor_model_paths(value, base) or changed
    elif isinstance(tree, list):
        for value in tree:
            changed = _anchor_model_paths(value, base) or changed
    return changed


# ------------------------------------------------------------------------- BRING YOUR MODEL

#: The models Python callers built and handed to a workflow, by the token their config names.
_LIVE_MODELS: dict[str, object] = {}


def live_model(token: str) -> object:
    """The model a caller built in Python and registered under ``token`` (:func:`train_model`,
    :func:`predict_model`). The classpath ``konfai.api:live_model`` names it in the run's config; the
    model lives in the process that registered it, so such a run stays on one rank, inline."""
    try:
        return _LIVE_MODELS[token]
    except KeyError:
        raise ConfigError(
            f"No live model is registered under '{token}'.",
            "A config naming 'konfai.api:live_model' runs in the process that built the model, through"
            " konfai.train_model / konfai.predict_model; it cannot be replayed from its file alone.",
        ) from None


@contextmanager
def _registered_live_model(model: object) -> Iterator[str]:
    """The token the run's config names, registered for the run only."""
    token = f"{type(model).__name__}-{id(model):x}"
    _LIVE_MODELS[token] = model
    try:
        yield token
    finally:
        _LIVE_MODELS.pop(token, None)


def _one_rank_inline(gpu: Sequence[int] | None) -> None:
    from konfai.utils.utils import env_flag

    if len(gpu or []) > 1 or not env_flag("KONFAI_INLINE_SINGLE_RANK", True):
        raise ConfigError(
            "A model built in Python runs on one rank, in this process.",
            "Spawned ranks are fresh interpreters that cannot see the object: give at most one GPU and"
            " leave KONFAI_INLINE_SINGLE_RANK on. For several GPUs, spell the model as a classpath.",
        )


def _group_tree(inputs: str, targets: str | None, transforms: Mapping[str, object] | None) -> dict[str, object]:
    """The ``groups_src`` block of a run that feeds ``inputs`` to the model and scores ``targets``."""
    transforms = dict(transforms or {})
    tree: dict[str, object] = {}
    for group, is_input in ((inputs, True), (targets, False)):
        if group is None:
            continue
        chain = transforms.get(group)
        tree[group] = {
            "groups_dest": {
                group: {
                    "transforms": None if chain is None else _chain_tree(chain, _STAGE_MODULES, f"transforms.{group}"),
                    "patch_transforms": None,
                    "is_input": is_input,
                }
            }
        }
    return tree


def _patch_tree(patch: Sequence[int], overlap: int | None, pad_value: float) -> dict[str, object]:
    return {
        "patch_size": [int(extent) for extent in patch],
        "overlap": overlap,
        "pad_value": pad_value,
        "extend_slice": 0,
    }


def train_model(
    model: object,
    datasets: str | Path | Sequence[str | Path],
    *,
    inputs: str,
    targets: str,
    loss: object,
    patch: Sequence[int],
    epochs: int = 1,
    batch_size: int = 1,
    lr: float = 1e-3,
    dim: int | None = None,
    in_channels: int = 1,
    transforms: Mapping[str, object] | None = None,
    augmentations: Sequence[object] | None = None,
    validation: float | str | None = 0.2,
    autocast: bool = False,
    channels_last: bool = False,
    name: str = "MODEL",
    manual_seed: int | None = None,
    gpu: Sequence[int] | None = None,
    quiet: bool = False,
    overwrite: bool = False,
    checkpoints_dir: Path | str = Path("./Checkpoints"),
    statistics_dir: Path | str = Path("./Statistics"),
) -> Path:
    """Train a model built in Python (any ``nn.Module`` with one tensor in and one out) on a KonfAI
    dataset; return the checkpoint workspace.

    ``inputs`` and ``targets`` are the dataset's groups, ``loss`` a criterion object or a list of
    them, ``augmentations`` a list of draws applied to every case, ``patch`` the patch the model is
    fed, ``dim`` its spatial rank (the patch's non-unit axes by default). The workspace names the
    model by its token, and a RESUME must come from this same process. One rank, inline.
    """
    from konfai.trainer import build_train
    from konfai.utils.runtime import State

    _one_rank_inline(gpu)
    with _registered_live_model(model) as token:
        losses = list(loss) if isinstance(loss, (list, tuple)) else [loss]
        tree: dict[str, object] = {
            "Trainer": {
                "Model": {
                    "classpath": "konfai.api:live_model",
                    "live_model": {
                        "token": token,
                        "in_channels": in_channels,
                        "dim": dim if dim is not None else sum(1 for extent in patch if int(extent) != 1),
                        "optimizer": {"name": "AdamW", "lr": lr},
                        "schedulers": None,
                        "outputs_criterions": {
                            "Model": {
                                "targets_criterions": {
                                    targets: {
                                        "criterions_loader": {
                                            key: {"is_loss": True, **kwargs}  # type: ignore[dict-item]
                                            for key, kwargs in _chain_tree(losses, _CRITERION_MODULES, "loss").items()
                                        }
                                    }
                                }
                            }
                        },
                        "ModelPatch": None,
                    },
                },
                "Dataset": {
                    "groups_src": _group_tree(inputs, targets, transforms),
                    "augmentations": (
                        None
                        if not augmentations
                        else {
                            "DataAugmentation_0": {
                                "data_augmentations": _chain_tree(augmentations, _STAGE_MODULES, "augmentations"),
                                "nb": 1,
                            }
                        }
                    ),
                    "Patch": _patch_tree(patch, None, 0),
                    "dataset_filenames": _dataset_filenames(datasets),
                    "batch_size": batch_size,
                    "validation": validation,
                    "shuffle": True,
                    "inline_augmentations": True,
                    "pin_memory": bool(gpu),
                },
                "train_name": name,
                "manual_seed": manual_seed,
                "epochs": epochs,
                "autocast": autocast,
                "channels_last": channels_last,
                "save_checkpoint_mode": "BEST",
            }
        }
        return _launch(
            1,
            lambda: build_train(
                command=State.TRAIN,
                model=None,
                config=_config_copy(tree),
                checkpoints_dir=checkpoints_dir,
                statistics_dir=statistics_dir,
                lr=None,
            ),
            lambda workflow: Path(os.environ["KONFAI_CHECKPOINTS_DIRECTORY"]) / workflow.name,
            gpu=gpu,
            cpu=1,
            overwrite=overwrite,
            quiet=quiet,
        )


def _live_checkpoint(model: object, scratch_root: Path) -> Path:
    """The weights a live model holds, written as the checkpoint KonfAI's loader reads: under the
    ``Model`` entry, by the wrapper's name, each key under the module the wrapper adds."""
    import torch

    state = {"Model": {"live_model": {f"Model.{key}": value for key, value in model.state_dict().items()}}}  # type: ignore[attr-defined]
    path = scratch_root / "live_model.pt"
    torch.save(state, path)
    return path


def predict_model(
    model: object,
    datasets: str | Path | Sequence[str | Path],
    *,
    inputs: str,
    patch: Sequence[int],
    output: str | Path,
    checkpoints: Path | str | Sequence[Path | str] | None = None,
    group: str = "PRED",
    dim: int | None = None,
    in_channels: int = 1,
    overlap: int | None = None,
    batch_size: int = 1,
    transforms: Mapping[str, object] | None = None,
    final_transforms: object = None,
    autocast: bool = False,
    name: str = "MODEL",
    gpu: Sequence[int] | None = None,
    quiet: bool = False,
    overwrite: bool = False,
    predictions_dir: Path | str = Path("./Predictions"),
) -> Path:
    """Predict with a model built in Python over a KonfAI dataset, patch by patch with overlap
    blending, the output written slab by slab next to each case; return the workspace.

    ``checkpoints`` are KonfAI checkpoints of this model (what :func:`train_model` wrote); left
    ``None``, the weights the module holds in memory are used. ``output`` is a dataset root the way
    the YAML spells one (``./Pred:mha``), relative to the run's workspace (``Predictions/<name>/``)
    unless absolute; the prediction lands under ``group`` with the input's geometry. One rank, inline.
    """
    from konfai.predictor import build_predict
    from konfai.utils.runtime.environment import register_scratch_config

    _one_rank_inline(gpu)
    with _registered_live_model(model) as token:
        root, _, file_format = str(output).rpartition(":")
        if not root or not file_format:
            raise ConfigError(
                f"'output' must name a dataset root and its format, as the YAML does: './Pred:mha' (got {output!r}).",
                "The prediction is written into that root, one entry per case, under the given group.",
            )
        tree: dict[str, object] = {
            "Predictor": {
                "Model": {
                    "classpath": "konfai.api:live_model",
                    "live_model": {
                        "token": token,
                        "in_channels": in_channels,
                        "dim": dim if dim is not None else sum(1 for extent in patch if int(extent) != 1),
                        "outputs_criterions": None,
                        "ModelPatch": None,
                    },
                },
                "Dataset": {
                    "groups_src": _group_tree(inputs, None, transforms),
                    "augmentations": None,
                    "Patch": _patch_tree(patch, overlap, 0),
                    "dataset_filenames": _dataset_filenames(datasets),
                    "batch_size": batch_size,
                },
                "outputs_dataset": {
                    "Model": {
                        "OutputDataset": {
                            "name_class": "OutputDataset",
                            "before_reduction_transforms": None,
                            "after_reduction_transforms": None,
                            "final_transforms": (
                                None
                                if final_transforms is None
                                else _chain_tree(final_transforms, _STAGE_MODULES, "final_transforms")
                            ),
                            "dataset_filename": f"{root}:{file_format}",
                            "group": group,
                            "same_as_group": f"{inputs}:{inputs}",
                            "reduction": "Mean",
                        }
                    }
                },
                "train_name": name,
                "autocast": autocast,
                "combine": "Mean",
            }
        }

        def build() -> "DistributedObject":
            if checkpoints is None:
                scratch = Path(tempfile.mkdtemp(prefix="konfai_live_"))
                register_scratch_config(scratch)
                sources = [_live_checkpoint(model, scratch)]
            else:
                sources = (
                    [Path(checkpoints)]
                    if isinstance(checkpoints, (str, Path))
                    else [Path(entry) for entry in checkpoints]
                )
            return build_predict(models=sources, prediction_file=_config_copy(tree), predictions_dir=predictions_dir)

        return _launch(
            1,
            build,
            lambda workflow: Path(os.environ["KONFAI_PREDICTIONS_DIRECTORY"]) / workflow.name,
            gpu=gpu,
            cpu=1,
            overwrite=overwrite,
            quiet=quiet,
        )


# ------------------------------------------------------------------------------ MONAI BUNDLES


def import_bundle(bundle: "Path | str", **options: object) -> "BundleImport":
    """A MONAI Bundle's network and weights as a KonfAI ``Model`` block and checkpoint
    (:func:`konfai.bundle.import_bundle`)."""
    from konfai.bundle import import_bundle as _import_bundle

    return _import_bundle(bundle, **options)  # type: ignore[arg-type]


def export_bundle(model: object, example_input: object, out: "Path | str", **options: object) -> Path:
    """A loaded KonfAI network's inference head as a MONAI Bundle (:func:`konfai.bundle.export_bundle`)."""
    from konfai.bundle import export_bundle as _export_bundle

    return _export_bundle(model, example_input, out, **options)  # type: ignore[arg-type]


# ------------------------------------------------------------------------- PREDICTION / TRAINING


def predict(
    models: Path | str | Sequence[Path | str],
    config: Mapping[str, object] | Path | str,
    *,
    gpu: Sequence[int] | None = None,
    cpu: int = 1,
    quiet: bool = False,
    overwrite: bool = False,
    predictions_dir: Path | str = Path("./Predictions"),
) -> Path:
    """Run a PREDICTION workflow; return its workspace (``Predictions/<name>``).

    ``config`` is a ``Prediction.yml`` path or the same tree as a dict.
    """
    from konfai.predictor import build_predict

    # A bare str is a Sequence[str]: "best.pt" would expand per character.
    if isinstance(models, (str, Path)):
        models = [models]
    return _launch(
        len(gpu or []) or cpu,
        lambda: build_predict(
            models=[Path(model) for model in models],
            prediction_file=_config_copy(config),
            predictions_dir=predictions_dir,
        ),
        lambda workflow: Path(os.environ["KONFAI_PREDICTIONS_DIRECTORY"]) / workflow.name,
        gpu=gpu,
        cpu=cpu,
        overwrite=overwrite,
        quiet=quiet,
    )


def train(
    config: Mapping[str, object] | Path | str,
    *,
    resume: bool = False,
    model: Path | str | None = None,
    lr: float | None = None,
    gpu: Sequence[int] | None = None,
    cpu: int | None = None,
    quiet: bool = False,
    overwrite: bool = False,
    checkpoints_dir: Path | str = Path("./Checkpoints"),
    statistics_dir: Path | str = Path("./Statistics"),
) -> Path:
    """Run a TRAIN (or RESUME) workflow; return its checkpoint workspace.

    ``config`` is a ``Config.yml`` path or the same tree as a dict; for a sweep, load the YAML once,
    change the keys under study, and call this per run.
    """
    from konfai.trainer import build_train
    from konfai.utils.runtime import State

    return _launch(
        len(gpu or []) or (cpu or 1),
        lambda: build_train(
            command=State.RESUME if resume else State.TRAIN,
            model=model,
            config=_config_copy(config),
            checkpoints_dir=checkpoints_dir,
            statistics_dir=statistics_dir,
            lr=lr,
        ),
        lambda workflow: Path(os.environ["KONFAI_CHECKPOINTS_DIRECTORY"]) / workflow.name,
        gpu=gpu,
        cpu=cpu,
        overwrite=overwrite,
        quiet=quiet,
    )

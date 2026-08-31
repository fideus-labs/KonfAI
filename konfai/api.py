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

The four CLI commands, callable: :func:`transform` (with :func:`plan_transform`, its dry-run
twin), :func:`evaluate`, :func:`predict` and :func:`train`. One engine, two spellings: everything
here builds the same config tree the YAML file would hold and hands it to the same binder: a
chain is a list of live stage objects (their constructor arguments are recorded as given, see
``record_given_arguments``), or the equivalent mapping, or a whole tree loaded from an existing
YAML and modified in place.

The contract, and how it differs from the CLI:

- **A designed refusal raises** ``KonfAIError``: the message and the remedy are the exception;
  the caller decides. Only the CLI catches and exits.
- **Results come back structured**: what a run wrote and where, read from the run's own record
  (``outputs.json``, ``Metric_*.json``) instead of leaving the caller to fish files out.
- **The process is left as found**: the ``KONFAI_*`` environment is restored around every call,
  and one workflow runs at a time per process: a second concurrent call is refused with the
  remedy (subprocesses), not allowed to corrupt the first.
- **The record remains.** Every call materializes the resolved YAML in the run's workspace: a
  notebook run is promoted to a versioned experiment by copying ``result.config``: nothing to
  rewrite.
"""

import atexit
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
    from konfai.transformer import TransformPlan
    from konfai.utils.runtime import DistributedObject

_T = TypeVar("_T")

#: Where a bare stage name resolves, per stage family: the same rule the YAML loader applies.
_STAGE_MODULES = ("konfai.data.transform", "konfai.data.augmentation")
_CRITERION_MODULES = ("konfai.metric.measure",)

_ACTIVE = threading.Lock()


@contextmanager
def _one_workflow_at_a_time(ranks: int) -> Iterator[None]:
    """Serialize workflows within the process and leave the environment as found.

    The engine keys its state on process-wide ``KONFAI_*`` variables, so two in-process runs would
    corrupt each other: refused with the remedy rather than allowed. ``ranks`` is exported as
    ``KONFAI_LOCAL_RANKS`` for build-time budget sizing, exactly as the CLI launcher does.
    """
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
    """Take the workflow lock, build, execute, and read the result out through ``finish``.

    ``finish`` runs inside the lock on purpose: the workspace lives in ``KONFAI_*`` variables the
    lock's exit restores to the caller's. ``ranks`` sizes the lock and stays the caller's own
    expression: the entry points do not all derive it from ``gpu``/``cpu`` the same way.
    """
    from konfai.utils.clock import restart_startup_clock
    from konfai.utils.runtime import execute_distributed_object

    with _one_workflow_at_a_time(ranks):
        with restart_startup_clock().phase("build"):  # this call's own clock, not the previous workflow's
            workflow = build()
        execute_distributed_object(workflow, gpu=list(gpu or []), cpu=cpu, overwrite=overwrite, quiet=quiet)
        return finish(workflow)


def _yaml_safe(value: object, where: str) -> object:
    """``value`` as the config file could hold it: or a refusal that names the argument."""
    # Before the Python scalars: np.float64 IS a float subclass, and ruamel refuses it.
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
    """A chain (a sequence of stages), as the mapping the config tree holds, in order.

    The tree is a mapping, so two stages of the same class need distinct spellings: the second
    occurrence is written module-qualified (which resolves to the same class); a third has no
    spelling left and is refused: the YAML file has the same limit.
    """
    if isinstance(stages, Mapping):  # a chain already spelled as its tree
        return {str(key): _yaml_safe(value, f"{where}.{key}") for key, value in stages.items()}
    tree: dict[str, object] = {}
    for index, stage in enumerate(_stage_sequence(stages, where)):
        name, kwargs = _stage_entry(stage, default_modules, f"{where}[{index}]")
        if name in tree and ":" not in name:
            name = _qualified_spelling(stage, name, default_modules)
        if name in tree:
            raise ConfigError(
                f"'{where}' holds three stages spelled '{name.split(':')[-1]}'; the tree is a"
                " mapping and has two spellings (bare and module-qualified), not three.",
                "Split the chain around a Save/Write boundary, or subclass the stage under a distinct name.",
            )
        tree[name] = kwargs
    return tree


def _stage_sequence(stages: object, where: str) -> Sequence[object]:
    if isinstance(stages, Sequence) and not isinstance(stages, (str, bytes)):
        return stages
    raise ConfigError(
        f"'{where}' is a {type(stages).__name__}; a chain is a sequence of stages.",
        "Pass the stages in application order: [Clip(min_value=0), Write(dataset='./Out:mha')].",
    )


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

    #: The run directory (``Transforms/<name>``): logs (the plan opens them), the resolved config,
    #: ``outputs.json``: never the deliverable, which each ``Write`` placed in the caller's tree.
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
    mapping tree. Every chain ends in a ``Write``: the same rule, message and plan as the CLI,
    which is this function with a YAML file. GPU is opt-in (``gpu=[0]``), as it is on the CLI.
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
    """:func:`transform`'s dry-run twin: build, plan, print, return the plan: the run never starts.

    Same arguments, same verdicts: the returned ``TransformPlan`` is the run's own routing
    (STREAM/LOAD/WHOLE-VOLUME/SKIP/REDUCE), measured, not estimated.
    """
    tree = _transform_tree(name, datasets, chains, memory_budget, on_fallback, manual_seed, dataset_options)
    from konfai.transformer import plan_transform as _plan_transform

    ranks = len(gpu or []) or cpu
    with _one_workflow_at_a_time(ranks):
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
    one-entry mappings. ``transforms`` optionally names a pre-metric chain per group (a cast, a
    clip): the groups themselves are derived from ``metrics``, declared once, not twice.
    """
    # A composite target ("Seg;Mask") is one metric key over several dataset groups: split it here,
    # as the Evaluator does, so each named group is actually loaded.
    groups = sorted(
        {str(group) for group in metrics}
        | {part for targets in metrics.values() for target in targets for part in str(target).split(";")}
    )
    groups_src: dict[str, object] = {}
    for group in groups:
        # An undeclared chain is spelled None, never left out: the binder materializes its own
        # default (Normalize) for an absent key, and each group rescaled to [-1, 1] by its own
        # extrema erases the very difference the metrics measure (MAE 1.9e-8 instead of 0.025 on a
        # pair related by 0.9x + 0.05). ``transforms`` is the only key GroupTransformMetric reads.
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

    Reading a KonfAI config resolves and REWRITES it: the record the workspace keeps. A tree is
    passed through; a caller's FILE is not this call's to rewrite, so the write-back lands on a
    scratch copy instead (removed at exit, like :func:`_materialized_config`'s).
    """
    if isinstance(config, Mapping):
        return dict(config)
    source = Path(config)
    scratch = Path(tempfile.mkdtemp(prefix="konfai_config_"))
    atexit.register(shutil.rmtree, scratch, ignore_errors=True)
    return Path(shutil.copy2(source, scratch / source.name))


# ------------------------------------------------------------------------- PREDICTION / TRAINING


def predict(
    models: Sequence[Path | str],
    config: Mapping[str, object] | Path | str,
    *,
    gpu: Sequence[int] | None = None,
    cpu: int = 1,
    quiet: bool = False,
    overwrite: bool = False,
    predictions_dir: Path | str = Path("./Predictions"),
) -> Path:
    """Run a PREDICTION workflow; return its workspace (``Predictions/<name>``).

    ``config`` is a ``Prediction.yml`` path or the same tree as a dict: a prediction's substance
    (checkpoints, patching, TTA, ensembling) is wiring, and the tree is its honest spelling.
    """
    from konfai.predictor import build_predict

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

    ``config`` is a ``Config.yml`` path or the same tree as a dict. The Python idiom for a sweep
    is the tree: load the YAML once, change the keys under study, call this: the resolved config
    each run keeps IS the record of what was tried.
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

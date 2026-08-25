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

"""Compare the chains two configs spell for the same model input groups.

Same checkpoint, different preprocessing is silent: the run succeeds and only the values are wrong.
The comparison is on the config trees as written, so it needs neither config to be bound.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from konfai.utils.config import _load_tree, _parse_bool

#: Everything from this stage on runs once per COPY (a TTA draw), not on the case the model reads.
_COPY_MARKER = "Expand"

#: An argument of the OUTPUT path: whether the stage is replayed inverted on the prediction. It says
#: nothing about the values the model reads.
_OUTPUT_ONLY_ARGUMENT = "inverse"

#: What a stage's argument reads as when the other chain declares it and this one does not.
_UNSET = "(unset)"

#: Where TRAIN leaves the resolved config of a run, beside ``<checkpoints_dir>/<train_name>/``.
_STATISTICS = "Statistics"

#: The root key TRAIN's config file is spelled under.
_TRAINER_ROOT = "Trainer"


@dataclass(frozen=True)
class ChainDifference:
    """One position where two chains of the same group disagree."""

    group: str
    chain: str
    index: int
    stage: str
    detail: str

    def __str__(self) -> str:
        return f"'{self.group}' {self.chain}[{self.index}] {self.stage}: {self.detail}"


def dataset_tree(config: Path, root: str) -> Mapping[str, Any]:
    """The ``Dataset`` block a config file spells under ``root``, read without writing it back."""
    return _mapping(_mapping(_load_tree(config).get(root)).get("Dataset"))


def training_dataset_tree(checkpoint: Path) -> Mapping[str, Any] | None:
    """The ``Dataset`` block of the resolved config the run that wrote ``checkpoint`` left behind.

    A workspace holds ``<checkpoints_dir>/<train_name>/*.pt`` beside ``Statistics/<train_name>/*.yml``,
    where TRAIN copies its config; the newest one there is the run's. ``None`` when the run kept none
    within reach: a checkpoint from an app bundle, a hand-copied ``.pt``, a moved statistics directory.
    """
    run = checkpoint.parent
    statistics = run.parent.parent / _STATISTICS / run.name
    configs = sorted(statistics.glob("*.yml"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not configs:
        return None
    # An empty block means the newest file there is no training config: nothing to compare against.
    return dataset_tree(configs[0], _TRAINER_ROOT) or None


def input_chain_differences(trained: Mapping[str, Any], applied: Mapping[str, Any]) -> list[ChainDifference]:
    """Every difference between the chains two ``Dataset`` trees apply to the model's input groups.

    Only what reaches the model is compared: a stage that alters no value, everything from an
    ``Expand`` marker on, and the ``inverse`` argument are not differences. A group ``applied``
    declares as an input and ``trained`` does not have is compared against an empty chain.
    """
    trained_groups = _groups(trained)
    differences: list[ChainDifference] = []
    for group, applied_group in _groups(applied).items():
        if not _parse_bool(applied_group.get("is_input", True)):
            continue
        trained_group = _mapping(trained_groups.get(group))
        for chain in ("transforms", "patch_transforms"):
            differences.extend(
                _stage_differences(
                    ":".join(group), chain, _stages(trained_group.get(chain)), _stages(applied_group.get(chain))
                )
            )
    return differences


def _mapping(value: Any) -> Mapping[str, Any]:
    """``value`` as a mapping; anything else (a missing key, the ``None`` a chain writes) is empty."""
    return value if isinstance(value, Mapping) else {}


def _plain(value: Any) -> Any:
    """A ruamel node as plain Python, so two configs compare and print by value."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _stage_name(classpath: str) -> str:
    """The class a chain key names, without its module and without the ``/n`` that keeps two stages
    of the same class apart (the rule :func:`konfai.utils.utils.get_module` reads a classpath by)."""
    return classpath.split(":")[-1].split(".")[-1].split("/")[0]


def _groups(dataset: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Every ``(group_src, group_dest)`` group a ``Dataset`` tree declares, in declaration order."""
    groups = {}
    for group_src, group in _mapping(dataset.get("groups_src")).items():
        for group_dest, chain in _mapping(_mapping(group).get("groups_dest")).items():
            groups[(str(group_src), str(group_dest))] = _mapping(chain)
    return groups


def _alters_values(classpath: str) -> bool:
    """Whether the stage changes the values it is handed, as :attr:`Transform.alters_values` declares.

    Read off KonfAI's own stage namespace, which the workflow has already imported. A class KonfAI
    does not own is taken to alter values: importing a foreign module to ask would run its code here.
    """
    from konfai.data import transform

    module, separator, _ = classpath.rpartition(":")
    if separator and module != transform.__name__:
        return True
    stage = getattr(transform, _stage_name(classpath), None)
    if not isinstance(stage, type) or not issubclass(stage, transform.Transform):
        return True
    return stage.alters_values


def _stages(chain: Any) -> list[tuple[str, dict[str, Any]]]:
    """The stages of one chain that shape what the model reads, as ``(class, arguments)``."""
    stages = []
    for classpath, arguments in _mapping(chain).items():
        name = _stage_name(str(classpath))
        if name == _COPY_MARKER:
            break
        if not _alters_values(str(classpath)):
            continue
        given = _plain(_mapping(arguments))
        stages.append((name, {key: value for key, value in given.items() if key != _OUTPUT_ONLY_ARGUMENT}))
    return stages


def _stage_differences(
    group: str,
    chain: str,
    trained: list[tuple[str, dict[str, Any]]],
    applied: list[tuple[str, dict[str, Any]]],
) -> Iterator[ChainDifference]:
    """Walk two chains of one group position by position, yielding where they disagree."""
    for index in range(max(len(trained), len(applied))):
        if index >= len(trained):
            yield ChainDifference(
                group, chain, index, applied[index][0], "applied here, absent from the training chain"
            )
            continue
        if index >= len(applied):
            yield ChainDifference(group, chain, index, trained[index][0], "in the training chain, not applied here")
            continue
        (trained_stage, trained_arguments), (applied_stage, applied_arguments) = trained[index], applied[index]
        if trained_stage != applied_stage:
            detail = f"the training chain has '{trained_stage}' at this position"
            yield ChainDifference(group, chain, index, applied_stage, detail)
            continue
        details = []
        for key in sorted(trained_arguments.keys() | applied_arguments.keys()):
            before, after = trained_arguments.get(key, _UNSET), applied_arguments.get(key, _UNSET)
            if before != after:
                details.append(f"{key}: {before!r} in training, {after!r} here")
        if details:
            yield ChainDifference(group, chain, index, applied_stage, "; ".join(details))

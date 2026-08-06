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

"""The Python front door (:mod:`konfai.api`): live objects and the YAML file are two spellings of
one engine. Pins the kwargs recording, the object->tree serialization, the run contract (raise, not
exit; the process env left as found; one workflow at a time), and byte-identity between the two
spellings of the same run."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

sitk = pytest.importorskip("SimpleITK")

from konfai import api  # noqa: E402
from konfai.data.reduction import Std  # noqa: E402
from konfai.data.transform import Clip, Magnitude, Resample, Warp, Write  # noqa: E402
from konfai.metric.measure import Dice  # noqa: E402
from konfai.utils.errors import ConfigError, KonfAIError  # noqa: E402

# --------------------------------------------------------------------------- recording and trees


def test_a_stage_records_the_arguments_as_given() -> None:
    stage = Clip(min_value=-100.0, max_value=300.0)
    assert stage._konfai_given == {"min_value": -100.0, "max_value": 300.0}


def test_a_criterion_records_too() -> None:
    assert "labels" in Dice(labels=[1, 2])._konfai_given


def test_a_subclass_with_no_init_of_its_own_records_the_inherited_one() -> None:
    """Accuracy inherits Criterion's constructor whole -- the recording must come with it."""
    from konfai.metric.measure import Accuracy

    assert Accuracy()._konfai_given == {}


def test_a_repeated_mapping_stage_is_qualified_by_resolution() -> None:
    """The second occurrence of a bare mapping name gets the module the binder would resolve."""
    tree = api._chain_tree(
        [{"Clip": {"min_value": 0.0}}, {"Clip": {"max_value": 1.0}}],
        api._STAGE_MODULES,
        "chains.CT.CT",
    )
    assert list(tree) == ["Clip", "konfai.data.transform:Clip"]


def test_a_subclass_delegating_to_super_keeps_its_own_spelling() -> None:
    """``Warp(field=...)`` expands into ``Resample`` arguments internally; the recorded spelling is
    the caller's, so the tree references ``Warp`` with the caller's kwargs and rebinds identically."""
    stage = Warp(field="./DVF:omezarr", group="DVF")
    assert stage._konfai_given == {"field": "./DVF:omezarr", "group": "DVF"}


def test_the_chain_tree_is_the_yaml_subtree() -> None:
    tree = api._chain_tree([Clip(min_value=0.0), Write(dataset="./Out:mha")], api._STAGE_MODULES, "chains.CT.CT")
    assert tree == {"Clip": {"min_value": 0.0}, "Write": {"dataset": "./Out:mha"}}


def test_an_unrecordable_stage_is_refused_by_name() -> None:
    class VarArgs(Clip):
        def __init__(self, *bounds: float) -> None:
            super().__init__(min_value=min(bounds))

    with pytest.raises(ConfigError, match="VarArgs"):
        api._chain_tree([VarArgs(1.0, 2.0)], api._STAGE_MODULES, "chains.CT.CT")


def test_a_non_spellable_argument_is_refused_by_name() -> None:
    with pytest.raises(ConfigError, match="max_value"):
        api._chain_tree([Clip(max_value=np.ones(2))], api._STAGE_MODULES, "chains.CT.CT")


# ------------------------------------------------------------------------------------ run contract


def _write_case(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(values), str(path))


@pytest.fixture()
def cohort(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    for case in ("P000", "P001"):
        _write_case(tmp_path / "Raw" / case / "CT.mha", rng.normal(0.0, 200.0, (6, 7, 8)).astype(np.float32))
    return tmp_path


def test_objects_and_yaml_are_two_spellings_of_one_run(cohort: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(cohort)
    result = api.transform(
        "BY_OBJECTS",
        "./Raw:mha",
        {"CT": {"CT": [Clip(min_value=-50.0, max_value=100.0), Write(dataset="./OutA:mha")]}},
        transforms_dir=cohort / "Transforms",
        quiet=True,
    )
    api.transform(
        "BY_TREE",
        "./Raw:mha",
        {"CT": {"CT": {"Clip": {"min_value": -50.0, "max_value": 100.0}, "Write": {"dataset": "./OutB:mha"}}}},
        transforms_dir=cohort / "Transforms",
        quiet=True,
    )
    for case in ("P000", "P001"):
        by_objects = (cohort / "OutA" / case / "CT.mha").read_bytes()
        by_tree = (cohort / "OutB" / case / "CT.mha").read_bytes()
        assert by_objects == by_tree
    assert result.workspace == cohort / "Transforms" / "BY_OBJECTS"
    assert result.outputs[0]["dataset"] == str(cohort / "OutA")
    assert result.config.is_file()


def test_a_designed_refusal_raises_instead_of_exiting(cohort: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(cohort)
    with pytest.raises(KonfAIError):
        api.transform(
            "NO_WRITE",
            "./Raw:mha",
            {"CT": {"CT": [Clip(min_value=0.0)]}},
            transforms_dir=cohort / "Transforms",
            quiet=True,
        )


def test_the_environment_is_left_as_found(cohort: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(cohort)
    for key in [key for key in os.environ if key.startswith("KONFAI")]:
        monkeypatch.delenv(key)
    api.transform(
        "ENV",
        "./Raw:mha",
        {"CT": {"CT": [Write(dataset="./OutEnv:mha")]}},
        transforms_dir=cohort / "Transforms",
        quiet=True,
    )
    assert [key for key in os.environ if key.startswith("KONFAI")] == []


def test_one_workflow_at_a_time_per_process(cohort: Path) -> None:
    assert api._ACTIVE.acquire(blocking=False)
    try:
        with pytest.raises(ConfigError, match="already running"):
            api.transform("BUSY", "./Raw:mha", {"CT": {"CT": [Write(dataset="./Out:mha")]}})
    finally:
        api._ACTIVE.release()


def test_the_reference_follows_the_case(cohort: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``reference: '{case}'`` adopts, per case, the grid of that case's own entry: two cases whose
    Ref grids differ land each on their own, not both on a memoized first."""
    monkeypatch.chdir(cohort)
    grids = {"P000": ((1.0, 1.2, 0.8), (5.0, -3.0, 2.0)), "P001": ((2.0, 0.7, 1.1), (-8.0, 4.0, 0.5))}
    for case, (spacing, origin) in grids.items():
        reference = sitk.GetImageFromArray(np.zeros((5, 6, 7), dtype=np.float32))
        reference.SetSpacing(spacing)
        reference.SetOrigin(origin)
        sitk.WriteImage(reference, str(cohort / "Raw" / case / "Ref.mha"))
    api.transform(
        "PER_CASE",
        "./Raw:mha",
        {"CT": {"Moved": [Resample(reference="{case}", reference_group="Ref"), Write(dataset="./Moved:mha")]}},
        transforms_dir=cohort / "Transforms",
        on_fallback="error",
        quiet=True,
    )
    for case, (spacing, origin) in grids.items():
        moved = sitk.ReadImage(str(cohort / "Moved" / case / "Moved.mha"))
        assert moved.GetSpacing() == pytest.approx(spacing)
        assert moved.GetOrigin() == pytest.approx(origin)
        assert moved.GetSize() == (7, 6, 5)


# --------------------------------------------------------------------------- uncertainty vocabulary


def test_std_reduction_matches_torch_incrementally() -> None:
    rng = np.random.default_rng(3)
    members = [torch.from_numpy(rng.normal(size=(1, 4, 5, 6)).astype(np.float32)) for _ in range(5)]
    expected = torch.stack(members).std(0)

    torch.testing.assert_close(Std()(list(members)), expected)

    incremental = Std()
    incremental.start()
    for member in members:
        incremental.accumulate(member)
    torch.testing.assert_close(incremental.finalize(), expected)


def test_std_of_a_single_case_is_zero() -> None:
    member = torch.ones(1, 2, 3)
    assert Std()([member]).abs().max() == 0.0


def test_magnitude_is_the_channel_norm_and_pointwise() -> None:
    from konfai.data.patching import LocalityKind
    from konfai.utils.dataset import Attribute

    field = torch.tensor([[[3.0]], [[4.0]]])
    stage = Magnitude()
    torch.testing.assert_close(stage("case", field, Attribute()), torch.tensor([[[5.0]]]))
    assert stage.patch_locality(Attribute()).kind is LocalityKind.POINTWISE


# ------------------------------------------------------------------------------------- config tree


def test_a_config_tree_must_hold_the_workflow_root() -> None:
    from konfai.utils.runtime import _materialized_config

    with pytest.raises(ConfigError, match="Transformer"):
        _materialized_config({"Trainer": {}}, "Transformer")
    path = _materialized_config({"Transformer": {"name": "X"}}, "Transformer")
    assert path.is_file()

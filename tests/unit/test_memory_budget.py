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

"""Tests for the B1 memory-budget chooser: it derives ``use_cache`` from a declared RAM budget,
estimates the dataset size from headers alone, and -- for ``"auto"`` -- reads the cgroup limit rather
than the host so a container/SLURM job is not OOM-killed."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from konfai.data import data_manager
from konfai.data.augmentation import DataAugmentationsList
from konfai.data.data_manager import (
    _AUTO_MEMORY_SAFETY_FRACTION,
    DataMetric,
    DataPrediction,
    DataTrain,
    _parse_memory_budget_bytes,
)
from konfai.utils import runtime
from konfai.utils.errors import ConfigError

# --------------------------------------------------------------------------------------
# Budget parsing -- a bare number is GiB, a string carries its own unit
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (24, 24 * 2**30),  # bare number => GiB
        (24.0, 24 * 2**30),
        ("24GB", 24 * 10**9),  # decimal
        ("32GiB", 32 * 2**30),  # binary
        ("512mb", 512 * 10**6),  # case-insensitive
        ("32 GiB", 32 * 2**30),  # optional space
        ("4096b", 4096),  # explicit bytes
        ("24", 24 * 2**30),  # unitless string (YAML face of a bare number) is GiB
    ],
)
def test_parse_memory_budget_bytes(value: str | float, expected: int) -> None:
    assert _parse_memory_budget_bytes(value) == expected


@pytest.mark.parametrize("value", ["twelve", "24 gigabytes", "GB", ""])
def test_parse_memory_budget_bytes_rejects_garbage(value: str) -> None:
    with pytest.raises(ConfigError):
        _parse_memory_budget_bytes(value)


# --------------------------------------------------------------------------------------
# THE CGROUP TRAP -- "auto" must see the cgroup ceiling, not the host's RAM
# --------------------------------------------------------------------------------------


def _point_cgroup_at(monkeypatch: pytest.MonkeyPatch, *, v2: Path | None, v1: Path | None) -> None:
    monkeypatch.setattr(runtime, "_CGROUP_V2_MEMORY_MAX", str(v2) if v2 else "/nonexistent/memory.max")
    monkeypatch.setattr(runtime, "_CGROUP_V1_MEMORY_LIMIT", str(v1) if v1 else "/nonexistent/limit_in_bytes")


def _fake_host_available(monkeypatch: pytest.MonkeyPatch, num_bytes: int) -> None:
    monkeypatch.setattr(runtime.psutil, "virtual_memory", lambda: SimpleNamespace(available=num_bytes))


def test_auto_respects_cgroup_limit_not_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The headline: cgroup grants 8 GB while psutil sees a 512 GB host.
    limit = tmp_path / "memory.max"
    limit.write_text("8000000000")
    _point_cgroup_at(monkeypatch, v2=limit, v1=None)
    _fake_host_available(monkeypatch, 512 * 2**30)

    assert runtime.available_memory_bytes() == (8_000_000_000, "cgroup limit")


def test_cgroup_v2_max_falls_back_to_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    unlimited = tmp_path / "memory.max"
    unlimited.write_text("max\n")
    _point_cgroup_at(monkeypatch, v2=unlimited, v1=None)
    _fake_host_available(monkeypatch, 64 * 2**30)

    assert runtime.available_memory_bytes() == (64 * 2**30, "host available RAM")


def test_cgroup_v1_limit_is_read_when_v2_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    v1 = tmp_path / "memory.limit_in_bytes"
    v1.write_text("8000000000\n")
    _point_cgroup_at(monkeypatch, v2=None, v1=v1)
    _fake_host_available(monkeypatch, 512 * 2**30)

    assert runtime.available_memory_bytes() == (8_000_000_000, "cgroup limit")


def test_cgroup_v1_sentinel_reads_as_unlimited(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    v1 = tmp_path / "memory.limit_in_bytes"
    v1.write_text(str(2**63))  # the near-INT64_MAX "no limit" sentinel
    _point_cgroup_at(monkeypatch, v2=None, v1=v1)
    _fake_host_available(monkeypatch, 64 * 2**30)

    assert runtime.available_memory_bytes() == (64 * 2**30, "host available RAM")


def test_no_cgroup_falls_back_to_host(monkeypatch: pytest.MonkeyPatch) -> None:
    _point_cgroup_at(monkeypatch, v2=None, v1=None)
    _fake_host_available(monkeypatch, 42 * 2**30)

    assert runtime.available_memory_bytes() == (42 * 2**30, "host available RAM")


# --------------------------------------------------------------------------------------
# The chooser -- derive use_cache from the budget vs the estimated dataset size
# --------------------------------------------------------------------------------------

# Two source groups, four cases each, a [1, 8, 8, 8] volume per case:
#   8 volumes x 512 elements x 4 bytes = 16384 bytes.
_GROUP_SHAPE = [1, 8, 8, 8]
_CASES = ["case_a", "case_b", "case_c", "case_d"]
_DATASET_BYTES = 2 * len(_CASES) * 512 * data_manager._CACHE_ELEMENT_BYTES


def _make_train(memory_budget: str | float | None) -> DataTrain:
    """A DataTrain with an injected, header-free prepared dataset (no disk, no config file)."""
    data = DataTrain(augmentations=None, memory_budget=memory_budget)
    managers = {group: [SimpleNamespace(base_shape=list(_GROUP_SHAPE)) for _ in _CASES] for group in ("CT", "SEG")}
    data._prepared_data = managers  # type: ignore[assignment]
    data._prepared_validation_data = {}
    data._prepared_train_names = list(_CASES)
    data._prepared_validation_names = []
    return data


def test_estimate_matches_known_fixture() -> None:
    assert _make_train(None)._estimate_cached_bytes() == _DATASET_BYTES


def test_estimate_counts_one_copy_per_augmentation_draw() -> None:
    """A cached case holds its base tensor plus one per draw, so the estimate must multiply by them.

    Counting the base tensor alone under-reports the cache by the augmentation count -- the budget
    then picks CACHE for a dataset several times too big for it and the run is OOM-killed anyway.
    """
    data = DataTrain(
        augmentations={"Aug_0": DataAugmentationsList(nb=4, data_augmentations={})},
        memory_budget=None,
        validation=None,
    )
    managers = {group: [SimpleNamespace(base_shape=list(_GROUP_SHAPE)) for _ in _CASES] for group in ("CT", "SEG")}
    data._prepared_data = managers  # type: ignore[assignment]
    data._prepared_validation_data = {}
    data._prepared_train_names = list(_CASES)
    data._prepared_validation_names = []

    # 1 base copy + 4 draws.
    assert data._estimate_cached_bytes() == 5 * _DATASET_BYTES


def test_budget_larger_than_dataset_caches() -> None:
    data = _make_train(f"{_DATASET_BYTES + 1}b")
    data._resolve_cache_regime(world_size=1)
    assert data.use_cache is True
    assert data.resolved_num_workers == 0  # caching preloads up front, so no loader workers


def test_budget_smaller_than_dataset_does_not_cache() -> None:
    data = _make_train(f"{_DATASET_BYTES - 1}b")
    data._resolve_cache_regime(world_size=1)
    assert data.use_cache is False
    assert data.resolved_num_workers > 0  # the streaming/buffer path spins workers up


def test_none_budget_means_auto_for_training(monkeypatch: pytest.MonkeyPatch) -> None:
    # No key declared -> "auto": the tiny dataset fits the detected memory, so training caches; on
    # a node too small for it, the same absent key streams instead of blowing past the RAM.
    roomy = _make_train(None)
    monkeypatch.setattr(data_manager, "available_memory_bytes", lambda: (_DATASET_BYTES * 10, "host"))
    roomy._resolve_cache_regime(world_size=1)
    assert roomy.use_cache is True

    tight = _make_train(None)
    monkeypatch.setattr(
        data_manager,
        "available_memory_bytes",
        lambda: (int(_DATASET_BYTES / _AUTO_MEMORY_SAFETY_FRACTION) - 1, "cgroup limit"),
    )
    tight._resolve_cache_regime(world_size=1)
    assert tight.use_cache is False


def test_one_pass_workflows_never_cache_whatever_the_budget() -> None:
    # Prediction AND evaluation read each case exactly once: a cache is never re-read, so even a
    # budget the dataset comfortably fits keeps them on the stream/buffer path.
    for data in (
        DataPrediction(augmentations=None, memory_budget=f"{_DATASET_BYTES * 100}b"),
        DataMetric(memory_budget=f"{_DATASET_BYTES * 100}b"),
    ):
        data._prepared_data = {"CT": [SimpleNamespace(base_shape=[1, 2, 2, 2])]}  # type: ignore[assignment]
        data._prepared_validation_data = {}
        data._prepared_train_names = ["case_a"]
        data._prepared_validation_names = []
        data._resolve_cache_regime(world_size=1)
        assert data.use_cache is False, type(data).__name__


def test_budget_is_per_rank_so_world_size_flips_the_decision() -> None:
    # A budget of half the dataset: it never fits on one rank, but does once sharded across four.
    half = f"{_DATASET_BYTES // 2}b"

    single = _make_train(half)
    single._resolve_cache_regime(world_size=1)
    assert single.use_cache is False

    sharded = _make_train(half)
    sharded._resolve_cache_regime(world_size=4)
    assert sharded.use_cache is True


# --------------------------------------------------------------------------------------
# The evaluation auto-patch -- an AUTO budget is a NODE budget, split across the local ranks
# --------------------------------------------------------------------------------------


def _metric_sizing_budget(
    monkeypatch: pytest.MonkeyPatch, memory_budget: str | float | None, local_ranks: str | None
) -> float:
    """Drive DataMetric._maybe_auto_patch over a fake one-case dataset and capture the budget it
    actually hands to resolve_patch."""
    data = DataMetric(memory_budget=memory_budget)
    data.datasets = {
        "f": SimpleNamespace(get_names=lambda group: ["case"], get_infos=lambda group, name: ([1, 64, 64, 64], None))
    }
    monkeypatch.setattr(DataMetric, "_resolve_dataset_sources", lambda self: {"CT": [("f", False)]}, raising=False)
    monkeypatch.setattr(data_manager, "available_memory_bytes", lambda: (100 * 2**30, "host"))
    captured: dict[str, float] = {}

    def capture(template, shape, channels, element_bytes, budget, **kwargs):
        captured["budget"] = budget
        return list(shape)  # "fits whole": the sizing exits without installing a patch

    monkeypatch.setattr(data_manager, "resolve_patch", capture)
    if local_ranks is None:
        monkeypatch.delenv("KONFAI_LOCAL_RANKS", raising=False)
    else:
        monkeypatch.setenv("KONFAI_LOCAL_RANKS", local_ranks)
    data._maybe_auto_patch()
    return captured["budget"]


def test_eval_auto_budget_is_divided_by_the_local_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    # 4 ranks evaluating on one node share its RAM: each sizes its patch from a quarter of the
    # auto budget, or together they over-commit the host 4-fold.
    node_auto = 100 * 2**30 * _AUTO_MEMORY_SAFETY_FRACTION
    assert _metric_sizing_budget(monkeypatch, None, "4") == node_auto // 4
    # Without the launcher's hint (direct API use), today's undivided behaviour is preserved.
    assert _metric_sizing_budget(monkeypatch, None, None) == node_auto


def test_eval_explicit_budget_is_per_rank_and_never_divided(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _metric_sizing_budget(monkeypatch, "1GiB", "4") == float(2**30)


def test_run_distributed_app_exports_and_restores_local_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    # The wrapper leaves the per-node rank count in the environment while the workflow is built
    # (the KeyboardInterrupt escapes the factory before any spawn), and restores it after -- a
    # leak would silently shrink a later in-process run's patches.
    captured: list[str] = []

    @runtime.run_distributed_app
    def factory(config=None, gpu: list[int] = [], cpu: int = 1):
        captured.append(os.environ["KONFAI_LOCAL_RANKS"])
        raise KeyboardInterrupt

    monkeypatch.delenv("KONFAI_LOCAL_RANKS", raising=False)
    factory(gpu=[0, 1])
    factory(gpu=[], cpu=3)
    assert captured == ["2", "3"]
    assert "KONFAI_LOCAL_RANKS" not in os.environ
    monkeypatch.setenv("KONFAI_LOCAL_RANKS", "7")
    factory(gpu=[0])
    assert captured[-1] == "1" and os.environ["KONFAI_LOCAL_RANKS"] == "7"


def test_auto_budget_uses_detected_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_available() -> tuple[int, str]:
        return budget_source

    monkeypatch.setattr(data_manager, "available_memory_bytes", fake_available)

    # A cgroup so small that 80% of it cannot hold the dataset -> do not cache.
    budget_source = (int(_DATASET_BYTES / _AUTO_MEMORY_SAFETY_FRACTION) - 1, "cgroup limit")
    tight = _make_train("auto")
    tight._resolve_cache_regime(world_size=1)
    assert tight.use_cache is False

    # A roomy node -> cache.
    budget_source = (_DATASET_BYTES * 10, "host available RAM")
    roomy = _make_train("auto")
    roomy._resolve_cache_regime(world_size=1)
    assert roomy.use_cache is True

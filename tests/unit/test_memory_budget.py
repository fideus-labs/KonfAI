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
estimates the dataset size from headers alone, and (for ``"auto"``) reads the cgroup limit rather
than the host so a container/SLURM job is not OOM-killed."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from konfai.data import data_manager
from konfai.data.augmentation import DataAugmentationsList
from konfai.data.data_manager import (
    DataMetric,
    DataPrediction,
    DataTrain,
)
from konfai.utils import budget, runtime
from konfai.utils.budget import AUTO_MEMORY_SAFETY_FRACTION, parse_memory_budget_bytes
from konfai.utils.errors import ConfigError

# --------------------------------------------------------------------------------------
# Budget parsing: a bare number is GiB, a string carries its own unit
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
def testparse_memory_budget_bytes(value: str | float, expected: int) -> None:
    assert parse_memory_budget_bytes(value) == expected


@pytest.mark.parametrize("value", ["twelve", "24 gigabytes", "GB", ""])
def test_parse_memory_budget_bytes_rejects_garbage(value: str) -> None:
    with pytest.raises(ConfigError):
        parse_memory_budget_bytes(value)


# --------------------------------------------------------------------------------------
# THE CGROUP TRAP: "auto" must see the cgroup ceiling, not the host's RAM
# --------------------------------------------------------------------------------------


def _fake_cgroup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, own: str, *, v1: bool = False) -> Path:
    """A cgroup tree under tmp_path whose /proc/self/cgroup names ``own``; returns the process's cgroup dir."""
    root = tmp_path / "cgroup"
    proc = tmp_path / "proc_self_cgroup"
    if v1:
        proc.write_text(f"3:memory:{own}\n1:cpu:{own}\n")
        base = root / "memory"
    else:
        proc.write_text(f"0::{own}\n")
        base = root
    leaf = base / own.lstrip("/")
    leaf.mkdir(parents=True)
    monkeypatch.setattr(budget, "_CGROUP_ROOT", str(root))
    monkeypatch.setattr(budget, "_PROC_SELF_CGROUP", str(proc))
    for key in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        monkeypatch.delenv(key, raising=False)
    return leaf


def _fake_host_available(monkeypatch: pytest.MonkeyPatch, num_bytes: int) -> None:
    monkeypatch.setattr(budget.psutil, "virtual_memory", lambda: SimpleNamespace(available=num_bytes))


def test_auto_respects_cgroup_limit_not_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The headline: cgroup grants 8 GB while psutil sees a 512 GB host (docker run: the process
    # sits at the mount root).
    leaf = _fake_cgroup(monkeypatch, tmp_path, "/")
    (leaf / "memory.max").write_text("8000000000")
    _fake_host_available(monkeypatch, 512 * 2**30)

    assert budget.available_memory_bytes() == (8_000_000_000, "cgroup limit")


def test_the_cgroup_is_the_process_s_own_not_the_mount_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Outside `docker run` the process lives deep in the hierarchy (a SLURM step, a user slice)
    and the mount root has no memory.max at all: the bound is the tightest ancestor's, minus what
    the process's own cgroup already holds."""
    leaf = _fake_cgroup(monkeypatch, tmp_path, "/system.slice/slurmstepd.scope/job_42/step_0/user/task_0")
    job = leaf.parents[2]  # job_42
    (job / "memory.max").write_text("32000000000\n")  # --mem=32G
    (leaf.parent / "memory.max").write_text("max\n")
    (leaf / "memory.max").write_text("max\n")
    (leaf / "memory.stat").write_text("anon 1500000000\nfile 0\nkernel 500000000\n")  # already resident
    _fake_host_available(monkeypatch, 512 * 2**30)

    assert budget.available_memory_bytes() == (30_000_000_000, "cgroup limit")


def test_the_page_cache_is_not_held_memory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A step that has streamed a cohort has a memory.current near its ceiling, almost all of it
    page cache the kernel drops under pressure: the budget subtracts anon + kernel, not current."""
    leaf = _fake_cgroup(monkeypatch, tmp_path, "/docker/abc")
    (leaf / "memory.max").write_text("50000000000\n")
    (leaf / "memory.current").write_text("49000000000\n")
    (leaf / "memory.stat").write_text("anon 9000000000\nfile 39000000000\nkernel 1000000000\nshmem 0\n")
    _fake_host_available(monkeypatch, 512 * 2**30)

    assert budget.available_memory_bytes() == (40_000_000_000, "cgroup limit")


def test_cgroup_v1_holds_rss_not_the_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    leaf = _fake_cgroup(monkeypatch, tmp_path, "/docker/abc", v1=True)
    (leaf / "memory.limit_in_bytes").write_text("8000000000\n")
    (leaf / "memory.usage_in_bytes").write_text("7900000000\n")
    (leaf / "memory.stat").write_text("cache 6900000000\nrss 1000000000\nmapped_file 0\n")
    _fake_host_available(monkeypatch, 512 * 2**30)

    assert budget.available_memory_bytes() == (7_000_000_000, "cgroup limit")


def test_cgroup_v2_max_falls_back_to_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    leaf = _fake_cgroup(monkeypatch, tmp_path, "/user.slice/app.scope")
    (leaf / "memory.max").write_text("max\n")
    _fake_host_available(monkeypatch, 64 * 2**30)

    assert budget.available_memory_bytes() == (64 * 2**30, "host available RAM")


def test_cgroup_v1_limit_is_read_when_v2_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    leaf = _fake_cgroup(monkeypatch, tmp_path, "/docker/abc", v1=True)
    (leaf / "memory.limit_in_bytes").write_text("8000000000\n")
    _fake_host_available(monkeypatch, 512 * 2**30)

    assert budget.available_memory_bytes() == (8_000_000_000, "cgroup limit")


def test_cgroup_v1_sentinel_reads_as_unlimited(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    leaf = _fake_cgroup(monkeypatch, tmp_path, "/", v1=True)
    (leaf / "memory.limit_in_bytes").write_text(str(2**63))  # the near-INT64_MAX "no limit" sentinel
    _fake_host_available(monkeypatch, 64 * 2**30)

    assert budget.available_memory_bytes() == (64 * 2**30, "host available RAM")


def test_no_cgroup_falls_back_to_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(budget, "_PROC_SELF_CGROUP", str(tmp_path / "nonexistent"))
    for key in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        monkeypatch.delenv(key, raising=False)
    _fake_host_available(monkeypatch, 42 * 2**30)

    assert budget.available_memory_bytes() == (42 * 2**30, "host available RAM")


def test_a_slurm_grant_bounds_the_budget_when_no_cgroup_enforces_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(budget, "_PROC_SELF_CGROUP", str(tmp_path / "nonexistent"))
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "32000")  # MB
    monkeypatch.delenv("SLURM_MEM_PER_CPU", raising=False)
    _fake_host_available(monkeypatch, 512 * 2**30)

    assert budget.available_memory_bytes() == (32000 * 2**20, "SLURM memory grant")


def test_a_slurm_grant_of_zero_means_the_whole_node(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(budget, "_PROC_SELF_CGROUP", str(tmp_path / "nonexistent"))
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "0")  # --mem=0
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "0")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    _fake_host_available(monkeypatch, 512 * 2**30)

    assert budget.available_memory_bytes() == (512 * 2**30, "host available RAM")


def test_a_slurm_per_cpu_grant_is_multiplied_by_the_task_cpus(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(budget, "_PROC_SELF_CGROUP", str(tmp_path / "nonexistent"))
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "4000")  # MB
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "64")  # the task's cpus win over the node's
    _fake_host_available(monkeypatch, 512 * 2**30)

    assert budget.available_memory_bytes() == (4000 * 8 * 2**20, "SLURM memory grant")


# --------------------------------------------------------------------------------------
# The chooser: derive use_cache from the budget vs the estimated dataset size
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

    Counting the base tensor alone under-reports the cache by the augmentation count: the budget
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
    monkeypatch.setattr(budget, "available_memory_bytes", lambda: (_DATASET_BYTES * 10, "host"))
    roomy._resolve_cache_regime(world_size=1)
    assert roomy.use_cache is True

    tight = _make_train(None)
    monkeypatch.setattr(
        budget,
        "available_memory_bytes",
        lambda: (int(_DATASET_BYTES / AUTO_MEMORY_SAFETY_FRACTION) - 1, "cgroup limit"),
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


def test_an_auto_budget_is_split_across_one_node_not_the_whole_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    """Four nodes of four GPUs each: a rank shares its RAM with three others, not fifteen.

    Dividing the node by the global world size makes every rank believe it has a sixteenth of one
    node, and the chooser then declines a cache that fits four times over.
    """
    # A node that offers half the dataset: a rank's sixteenth fits its quarter of that four times
    # over, while a sixteenth of it would be half of what the rank has to hold.
    node = _DATASET_BYTES / (2 * AUTO_MEMORY_SAFETY_FRACTION)
    monkeypatch.setattr(budget, "available_memory_bytes", lambda: (node, "host"))
    monkeypatch.setenv("KONFAI_LOCAL_RANKS", "4")

    data = _make_train(None)
    data._resolve_cache_regime(world_size=16)

    assert data.use_cache is True


# --------------------------------------------------------------------------------------
# The evaluation auto-patch: an AUTO budget is a NODE budget, split across the local ranks
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
    monkeypatch.setattr(budget, "available_memory_bytes", lambda: (100 * 2**30, "host"))
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
    node_auto = 100 * 2**30 * AUTO_MEMORY_SAFETY_FRACTION
    assert _metric_sizing_budget(monkeypatch, None, "4") == node_auto // 4
    # Without the launcher's hint (direct API use), today's undivided behaviour is preserved.
    assert _metric_sizing_budget(monkeypatch, None, None) == node_auto


def test_eval_explicit_budget_is_per_rank_and_never_divided(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _metric_sizing_budget(monkeypatch, "1GiB", "4") == float(2**30)


def test_a_garbled_local_ranks_variable_keeps_the_undivided_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # A user-exported junk value must not crash the build: any unparsable content falls back to 1.
    node_auto = 100 * 2**30 * AUTO_MEMORY_SAFETY_FRACTION
    assert _metric_sizing_budget(monkeypatch, None, "two") == node_auto


def test_run_distributed_app_exports_and_restores_local_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    # The wrapper leaves the per-node rank count in the environment while the workflow is built
    # (the KeyboardInterrupt escapes the factory before any spawn), and restores it after: a
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

    # A genuine factory failure (not the swallowed KeyboardInterrupt) must restore the variable too: # the restore lives in a finally, not in the interrupt handler.
    monkeypatch.delenv("KONFAI_LOCAL_RANKS", raising=False)

    @runtime.run_distributed_app
    def broken(config=None, gpu: list[int] = [], cpu: int = 1):
        raise ValueError("factory died")

    with pytest.raises(ValueError, match="factory died"):
        broken(gpu=[0, 1])
    assert "KONFAI_LOCAL_RANKS" not in os.environ


def test_auto_budget_uses_detected_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_available() -> tuple[int, str]:
        return budget_source

    monkeypatch.setattr(budget, "available_memory_bytes", fake_available)

    # A cgroup so small that 80% of it cannot hold the dataset -> do not cache.
    budget_source = (int(_DATASET_BYTES / AUTO_MEMORY_SAFETY_FRACTION) - 1, "cgroup limit")
    tight = _make_train("auto")
    tight._resolve_cache_regime(world_size=1)
    assert tight.use_cache is False

    # A roomy node -> cache.
    budget_source = (_DATASET_BYTES * 10, "host available RAM")
    roomy = _make_train("auto")
    roomy._resolve_cache_regime(world_size=1)
    assert roomy.use_cache is True

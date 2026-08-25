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
    data._managers = managers  # type: ignore[assignment]
    data._validation_managers = {}
    data.case_names = list(_CASES)
    data._validation_names = []
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
    data._managers = managers  # type: ignore[assignment]
    data._validation_managers = {}
    data.case_names = list(_CASES)
    data._validation_names = []

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
        data._managers = {"CT": [SimpleNamespace(base_shape=[1, 2, 2, 2])]}  # type: ignore[assignment]
        data._validation_managers = {}
        data.case_names = ["case_a"]
        data._validation_names = []
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
        "f": SimpleNamespace(
            select_names=lambda group, requested: ["case"],
            get_infos=lambda group, name: ([1, 64, 64, 64], None),
        )
    }
    monkeypatch.setattr(
        DataMetric, "_resolve_dataset_sources", lambda self, requested=None: {"CT": [("f", False)]}, raising=False
    )
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


def _metric_sizing(monkeypatch: pytest.MonkeyPatch, shape: list[int], budget_bytes: int, halo: int) -> DataMetric:
    """Drive DataMetric._maybe_auto_patch over a fake two-group case of ``shape`` with a real
    sizing, the metrics' widest halo declared."""
    data = DataMetric(memory_budget=f"{budget_bytes}b")
    data.datasets = {
        "f": SimpleNamespace(
            select_names=lambda group, requested: ["case"], get_infos=lambda group, name: (shape, None)
        )
    }
    monkeypatch.setattr(
        DataMetric,
        "_resolve_dataset_sources",
        lambda self, requested=None: {"CT": [("f", False)], "sCT": [("f", False)]},
        raising=False,
    )
    data.patch_halo = halo
    data._maybe_auto_patch()
    return data


def test_eval_sizing_reserves_the_metrics_halo_on_each_face(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two float32 groups of 64^3 at 24 B/voxel sized (resident + 2x intermediates): the READ is
    # what fits, and the slot is the read less a halo on each face, so every patch a metric is
    # handed, halo included, stays within the sizing.
    budget_bytes = 2 * 2**20
    plain = _metric_sizing(monkeypatch, [1, 64, 64, 64], budget_bytes, 0)
    haloed = _metric_sizing(monkeypatch, [1, 64, 64, 64], budget_bytes, 3)

    assert plain.patch is not None and haloed.patch is not None
    assert plain.patch.halo == 0 and haloed.patch.halo == 3
    assert haloed.patch.patch_size == [size - 6 for size in plain.patch.patch_size]
    read = [size + 6 for size in haloed.patch.patch_size]
    assert 3 * 2 * 4 * read[0] * read[1] * read[2] <= budget_bytes * 0.8


def test_eval_sizing_spans_an_axis_too_thin_for_its_halo(monkeypatch: pytest.MonkeyPatch) -> None:
    # An isotropic cut leaves a 7-deep axis 5 or 6 deep: thinner than two halos of 3, so it is
    # read whole and the other axes pay for it instead.
    data = _metric_sizing(monkeypatch, [1, 7, 256, 256], 512 * 2**10, 3)

    assert data.patch is not None
    assert data.patch.patch_size[0] == 7
    assert all(1 <= size < 256 - 6 for size in data.patch.patch_size[1:])


def test_eval_sizing_refuses_a_budget_no_halo_patch_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    # Its remedy is "raise memory_budget", so the refusal names the budget it broke, not only the
    # case it could not hold: without the figure the reader has to go and find what it was.
    from konfai.utils.errors import DatasetManagerError

    with pytest.raises(DatasetManagerError, match=r"budget \(4\.00 KiB\).+halo"):
        _metric_sizing(monkeypatch, [1, 64, 64, 64], 4096, 3)


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


# --------------------------------------------------------------------------------------
# The decoded-chunk cache is part of what the budget bounds, in every workflow
# --------------------------------------------------------------------------------------


@pytest.fixture
def _chunk_cache_restored():
    from konfai.utils import ome_zarr

    yield
    ome_zarr.set_chunk_cache_budget(None)


@pytest.mark.parametrize(
    ("declared", "capacity"),
    [
        (64 << 20, 64 << 20),  # under the floor: the budget itself, never more than it
        (512 << 20, 256 << 20),  # a third would be under the floor: the floor
        (3 << 30, 1 << 30),  # a third
    ],
)
def test_the_chunk_cache_takes_a_third_of_the_budget_and_never_more_than_it(
    declared: int, capacity: int, _chunk_cache_restored: None
) -> None:
    """The cache is part of what the process holds, so a budget it exceeded would be exceeded
    before a region was read."""
    from konfai.utils import ome_zarr

    assert ome_zarr.set_chunk_cache_budget(declared) == capacity
    assert ome_zarr._chunk_cache().capacity == capacity


def test_an_undeclared_budget_leaves_the_chunk_cache_its_share_of_free_ram(
    monkeypatch: pytest.MonkeyPatch, _chunk_cache_restored: None
) -> None:
    from konfai.utils import ome_zarr

    monkeypatch.setattr(budget, "available_memory_bytes", lambda: (100 * 2**30, "host"))
    assert ome_zarr.set_chunk_cache_budget(None) == 5 * 2**30


_WORKFLOW_ENV = (
    "KONFAI_config_file",
    "KONFAI_ROOT",
    "KONFAI_STATE",
    "KONFAI_CONFIG_MODE",
    "KONFAI_PREDICTIONS_DIRECTORY",
    "KONFAI_EVALUATIONS_DIRECTORY",
)
_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "Workflows"


def _tiny_cohort(root: Path) -> None:
    """Two cases of MR and CT and a prediction of each, the shape the TinySynth asset works on."""
    import numpy as np
    from konfai.utils.dataset import Attribute, Dataset

    attributes = Attribute()
    attributes["Origin"] = np.asarray([0.0, 0.0, 0.0])
    attributes["Spacing"] = np.asarray([1.0, 1.0, 1.0])
    attributes["Direction"] = np.eye(3, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(0)
    for name in ("CASE_000", "CASE_001"):
        for group in ("MR", "CT"):
            Dataset(root / "Dataset", "mha").write(
                group, name, rng.random((1, 3, 16, 16)).astype(np.float32), attributes
            )
        Dataset(root / "Predictions", "mha").write(
            "sCT", name, rng.random((1, 3, 16, 16)).astype(np.float32), attributes
        )


def _workflow_config(root: Path, asset: str, replacements: dict[str, str], batch_line: str) -> Path:
    """The asset's config with the cohort's paths and a declared budget under ``Dataset:``."""
    text = (_ASSETS / asset).read_text(encoding="utf-8")
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    assert batch_line in text
    path = root / asset
    path.write_text(text.replace(batch_line, f"{batch_line}    memory_budget: 64MiB\n"), encoding="utf-8")
    return path


def _build_evaluator(root: Path, monkeypatch: pytest.MonkeyPatch):
    from konfai.evaluator import Evaluator
    from konfai.utils.config import apply_config, strict_config
    from konfai.utils.runtime import State, configure_workflow_environment

    replacements = {
        "__DATASET_DIR__": str(root / "Dataset"),
        "__PREDICTIONS_DATASET_DIR__": str(root / "Predictions"),
        "__TRAIN_NAME__": "CACHE_01",
    }
    config = _workflow_config(root, "Evaluation.yml", replacements, "    batch_size: 4\n")
    configure_workflow_environment(
        config_path=config,
        root="Evaluator",
        state=State.EVALUATION,
        path_env={"KONFAI_EVALUATIONS_DIRECTORY": root / "Evaluations"},
    )
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    with strict_config("Evaluator", refuse=False):
        return apply_config()(Evaluator)()


def _build_predictor(root: Path, monkeypatch: pytest.MonkeyPatch):
    import shutil

    from konfai.predictor import Predictor
    from konfai.utils.config import apply_config, strict_config
    from konfai.utils.runtime import State, configure_workflow_environment

    shutil.copy2(_ASSETS / "TinySynth.py", root / "TinySynth.py")
    monkeypatch.syspath_prepend(str(root))
    replacements = {"__DATASET_DIR__": str(root / "Dataset"), "__TRAIN_NAME__": "CACHE_01"}
    config = _workflow_config(root, "Prediction.yml", replacements, "    batch_size: 16\n")
    configure_workflow_environment(
        config_path=config,
        root="Predictor",
        state=State.PREDICTION,
        path_env={"KONFAI_PREDICTIONS_DIRECTORY": root / "PredictionsOut"},
    )
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    with strict_config("Predictor", refuse=False):
        return apply_config()(Predictor)()


_HALO_EVALUATION = """Evaluator:
  metrics:
    sCT:
      targets_criterions:
        CT:
          criterions_loader:
            MAE:
              reduction: mean
            SSIM:
              dynamic_range: 2.0
  Dataset:
    groups_src:
      CT:
        groups_dest:
          CT:
            transforms: None
            patch_transforms: None
            is_input: true
      sCT:
        groups_dest:
          sCT:
            transforms: None
            patch_transforms: None
            is_input: true
    subset: None
    validation: None
    memory_budget: 40000b
    dataset_filenames:
      - __DATASET_DIR__:a:mha
      - __PREDICTIONS_DATASET_DIR__:i:mha
  train_name: HALO_01
"""


def test_a_halo_metric_no_longer_vetoes_the_patched_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SSIM beside MAE under a budget the case exceeds: the run takes the patched path, reading
    each slot with SSIM's halo, where SSIM once kept the whole run on the whole-volume path."""
    pytest.importorskip("SimpleITK")
    import numpy as np
    from konfai.evaluator import Evaluator
    from konfai.utils.config import apply_config, strict_config
    from konfai.utils.dataset import Attribute, Dataset
    from konfai.utils.runtime import State, configure_workflow_environment

    for key in _WORKFLOW_ENV:
        monkeypatch.setenv(key, "sentinel")
        monkeypatch.delenv(key)
    attributes = Attribute()
    attributes["Origin"] = np.asarray([0.0, 0.0, 0.0])
    attributes["Spacing"] = np.asarray([1.0, 1.0, 1.0])
    attributes["Direction"] = np.eye(3, dtype=np.float64).reshape(-1)
    volume = np.random.default_rng(0).random((1, 12, 16, 16)).astype(np.float32)
    Dataset(tmp_path / "Dataset", "mha").write("CT", "CASE_000", volume, attributes)
    Dataset(tmp_path / "Predictions", "mha").write("sCT", "CASE_000", volume, attributes)
    config = tmp_path / "Evaluation.yml"
    config.write_text(
        _HALO_EVALUATION.replace("__DATASET_DIR__", str(tmp_path / "Dataset")).replace(
            "__PREDICTIONS_DATASET_DIR__", str(tmp_path / "Predictions")
        ),
        encoding="utf-8",
    )
    configure_workflow_environment(
        config_path=config,
        root="Evaluator",
        state=State.EVALUATION,
        path_env={"KONFAI_EVALUATIONS_DIRECTORY": tmp_path / "Evaluations"},
    )
    os.environ["KONFAI_CONFIG_MODE"] = "Done"
    with strict_config("Evaluator", refuse=False):
        evaluator = apply_config()(Evaluator)()

    assert evaluator.dataset.auto_patch_allowed
    assert evaluator._streamed and evaluator._halo == 3
    assert evaluator.dataset.patch is not None and evaluator.dataset.patch.halo == 3
    # 2 groups x 12x16x16 float32 at 24 B/voxel against 32000 B: cut on every axis, the halo reserved.
    assert all(
        1 <= size < extent for size, extent in zip(evaluator.dataset.patch.patch_size, [12, 16, 16], strict=True)
    )


def test_a_declared_budget_bounds_the_chunk_cache_in_prediction_and_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _chunk_cache_restored: None
) -> None:
    """The same per-rank budget every workflow resolves bounds the same cache: a store read under
    PREDICTION or EVALUATION is read through it as under TRANSFORM (whose plan pins it in
    test_transformer_workflow)."""
    pytest.importorskip("SimpleITK")
    from konfai.utils import ome_zarr

    for key in _WORKFLOW_ENV:
        monkeypatch.setenv(key, "sentinel")
        monkeypatch.delenv(key)
    _tiny_cohort(tmp_path)

    ome_zarr.set_chunk_cache_budget(None)
    assert ome_zarr._chunk_cache().capacity >= ome_zarr.CHUNK_CACHE_FLOOR
    _build_evaluator(tmp_path, monkeypatch)
    assert ome_zarr._chunk_cache().capacity == 64 << 20

    ome_zarr.set_chunk_cache_budget(None)
    _build_predictor(tmp_path, monkeypatch)
    assert ome_zarr._chunk_cache().capacity == 64 << 20

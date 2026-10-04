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

"""Tests for ``konfai.utils.runtime``: workflow guards, environment normalisation,
overwrite confirmation, distributed-launch bookkeeping, and progress/DDP
synchronisation."""

import contextlib
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import konfai as konfai_module
import konfai.utils.runtime.distributed as rt_dist
import konfai.utils.runtime.logging as rt_logg
import pytest
from konfai.evaluator import Evaluator
from konfai.predictor import Predictor
from konfai.trainer import Trainer
from konfai.utils.errors import ConfigError, KonfAIError, KonfAIWarning
from konfai.utils.runtime import (
    DistributedObject,
    State,
    configure_workflow_environment,
    confirm_overwrite_or_raise,
    execute_distributed_object,
    is_interactive_session,
)

# ---------------------------------------------------------------------------
# Workflow guards, environment normalisation, overwrite confirmation, and
# distributed-launch bookkeeping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [Trainer, Predictor, Evaluator])
def test_core_workflows_raise_config_error_when_mode_is_not_done(
    monkeypatch: pytest.MonkeyPatch,
    factory: type[Trainer] | type[Predictor] | type[Evaluator],
) -> None:
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "default")

    with pytest.raises(ConfigError, match="KONFAI_CONFIG_MODE='Done'"):
        factory()


def test_configure_workflow_environment_normalizes_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KONFAI_config_file", raising=False)
    monkeypatch.delenv("KONFAI_ROOT", raising=False)
    monkeypatch.delenv("KONFAI_STATE", raising=False)
    monkeypatch.delenv("KONFAI_STATISTICS_DIRECTORY", raising=False)

    configure_workflow_environment(
        config_path=tmp_path / "Config.yml",
        root="Trainer",
        state=State.TRAIN,
        path_env={"KONFAI_STATISTICS_DIRECTORY": tmp_path / "Statistics"},
    )

    assert Path(os.environ["KONFAI_config_file"]).name == "Config.yml"
    assert os.environ["KONFAI_ROOT"] == "Trainer"
    assert os.environ["KONFAI_STATE"] == str(State.TRAIN)
    assert Path(os.environ["KONFAI_STATISTICS_DIRECTORY"]).name == "Statistics"


def test_confirm_overwrite_or_raise_requires_flag_in_non_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KONFAI_OVERWRITE", raising=False)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: False))

    with pytest.raises(ConfigError, match="Pass -y/--overwrite"):
        confirm_overwrite_or_raise(Path("/tmp/output"), "prediction", ConfigError)


def test_confirm_overwrite_or_raise_accepts_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KONFAI_OVERWRITE", raising=False)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")

    confirm_overwrite_or_raise(Path("/tmp/output"), "prediction", ConfigError)


def test_confirm_overwrite_or_raise_rejects_decline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KONFAI_OVERWRITE", raising=False)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt: "no")

    with pytest.raises(ConfigError, match="Overwrite was declined"):
        confirm_overwrite_or_raise(Path("/tmp/output"), "prediction", ConfigError)


def test_execute_distributed_object_sets_shared_master_port_without_forcing_launch_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("KONFAI_MASTER_PORT", raising=False)
    monkeypatch.delenv("CUDA_LAUNCH_BLOCKING", raising=False)

    class DummyContext:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, value, traceback) -> None:
            return None

    class DummyDistributed(DistributedObject):
        uses_collectives = False  # spawns two ranks on Windows too, which opens no process group

        def __init__(self) -> None:
            super().__init__("dummy")

        def setup(self, world_size: int):
            self.dataloader = [[] for _ in range(world_size)]

        def run_process(self, world_size: int, global_rank: int, local_rank: int, dataloaders):
            raise AssertionError("run_process should not be called in this unit test")

    spawn_calls: dict[str, object] = {}

    def fake_spawn(fn, nprocs: int, *args, **kwargs) -> None:
        spawn_calls["fn"] = fn
        spawn_calls["nprocs"] = nprocs
        spawn_calls["master_port"] = os.environ["KONFAI_MASTER_PORT"]
        spawn_calls["cuda_visible_devices"] = os.environ["CUDA_VISIBLE_DEVICES"]

    monkeypatch.setattr("konfai.utils.runtime.distributed.Log", DummyContext)
    monkeypatch.setattr("konfai.utils.runtime.distributed.TensorBoard", DummyContext)
    monkeypatch.setattr("konfai.utils.runtime.distributed.mp.spawn", fake_spawn)

    execute_distributed_object(DummyDistributed(), gpu=[0, 1], cpu=1, quiet=True)

    assert str(spawn_calls["master_port"]).isdigit()
    assert spawn_calls["cuda_visible_devices"] == "0,1"
    assert "KONFAI_MASTER_PORT" not in os.environ
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert "CUDA_LAUNCH_BLOCKING" not in os.environ
    assert spawn_calls["nprocs"] == 2


def test_cluster_kwargs_route_the_run_through_submitit_instead_of_spawning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    submitted = []

    class DummyContext:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, value, traceback) -> None:
            return None

    class DummyExecutor:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def update_parameters(self, *_args, **_kwargs) -> None:
            pass

        def submit(self, *args, **_kwargs) -> None:
            submitted.append(args)

    class DummyDistributed(DistributedObject):
        def __init__(self) -> None:
            super().__init__("dummy")

        def setup(self, world_size: int):
            self.dataloader = [[] for _ in range(world_size)]

        def run_process(self, world_size, global_rank, local_rank, dataloaders):
            raise AssertionError("run_process should not be called on the submitting side")

    monkeypatch.setattr("konfai.utils.runtime.distributed.Log", DummyContext)
    monkeypatch.setattr("konfai.utils.runtime.distributed.TensorBoard", DummyContext)
    monkeypatch.setitem(sys.modules, "submitit", SimpleNamespace(AutoExecutor=DummyExecutor))

    cluster_kwargs = {"name": "job", "memory": 8, "num_nodes": 1, "time_limit": 60}
    execute_distributed_object(DummyDistributed(), gpu=[0], cpu=1, quiet=True, cluster_kwargs=cluster_kwargs)

    assert len(submitted) == 1


def test_a_cluster_submission_without_gpus_is_refused_before_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cluster job runs one rank per GPU of each node: without --gpu it has no rank, and it was
    submitted with zero tasks after setup had already run (and cleared the run's outputs)."""
    parameters: dict[str, object] = {}
    setups: list[int] = []

    class Executor:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def update_parameters(self, **kwargs) -> None:
            parameters.update(kwargs)

        def submit(self, *_args, **_kwargs) -> None:
            pass

    class Workflow(DistributedObject):
        def setup(self, world_size: int):
            setups.append(world_size)
            self.dataloader = []

        def run_process(self, world_size, global_rank, local_rank, dataloaders):
            raise AssertionError("run_process should not be called on the submitting side")

    monkeypatch.setattr(rt_dist, "Log", lambda *_args, **_kwargs: contextlib.nullcontext())
    monkeypatch.setattr(rt_dist, "TensorBoard", lambda *_args, **_kwargs: contextlib.nullcontext())
    monkeypatch.setitem(sys.modules, "submitit", SimpleNamespace(AutoExecutor=Executor))

    cluster_kwargs = {"name": "job", "memory": 8, "num_nodes": 2, "time_limit": 60}
    with pytest.raises(ConfigError, match="--gpu"):
        execute_distributed_object(Workflow("job"), gpu=[], cpu=1, quiet=True, cluster_kwargs=cluster_kwargs)
    assert setups == [] and parameters == {}


def test_get_available_devices_maps_visible_env_ids_to_local_torch_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")

    queried_indices: list[int] = []

    def fake_get_device_name(index: int) -> str:
        queried_indices.append(index)
        return f"GPU{index}"

    # get_available_devices imports get_device_name lazily from torch.cuda, so patch it at the source.
    monkeypatch.setattr("torch.cuda.get_device_name", fake_get_device_name)

    devices_index, devices_name = konfai_module.get_available_devices()

    assert devices_index == [3, 5]
    assert devices_name == ["GPU0", "GPU1"]
    assert queried_indices == [0, 1]


def test_cuda_visible_devices_refuses_a_device_named_by_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    """A UUID entry (a MIG slice, some containers) has no index ``--gpu`` could name, nor one the
    launcher could write back."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-3f2a1b4c")

    with pytest.raises(ConfigError, match="CUDA_VISIBLE_DEVICES='GPU-3f2a1b4c'"):
        konfai_module.cuda_visible_devices()


# ---------------------------------------------------------------------------
# Progress/DDP synchronisation
# ---------------------------------------------------------------------------


def test_synchronize_data_gathers_on_cpu(monkeypatch):
    """gloo/CPU multi-process must still all_gather (not fall back to local rank)."""
    calls = {}

    def fake_all_gather_object(outputs, data):
        calls["called"] = True
        for i in range(len(outputs)):
            outputs[i] = data

    def fail_set_device(*_args, **_kwargs):
        raise AssertionError("set_device must not be called when CUDA is unavailable")

    monkeypatch.setattr(rt_dist.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(rt_dist.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(rt_dist.torch.cuda, "set_device", fail_set_device)
    monkeypatch.setattr(rt_dist.dist, "all_gather_object", fake_all_gather_object)

    result = rt_dist.synchronize_data(3, 0, {"a": 1})

    assert calls.get("called") is True
    assert result == [{"a": 1}, {"a": 1}, {"a": 1}]


def test_synchronize_data_sets_device_on_cuda(monkeypatch):
    """When CUDA is available the target device is selected before gathering."""
    seen = {}

    def fake_all_gather_object(outputs, data):
        for i in range(len(outputs)):
            outputs[i] = data

    monkeypatch.setattr(rt_dist.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(rt_dist.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rt_dist.torch.cuda, "set_device", lambda gpu: seen.setdefault("gpu", gpu))
    monkeypatch.setattr(rt_dist.dist, "all_gather_object", fake_all_gather_object)

    result = rt_dist.synchronize_data(2, 1, {"b": 2})

    assert seen.get("gpu") == 1
    assert result == [{"b": 2}, {"b": 2}]


def test_a_workflow_without_collectives_gets_its_rank_and_no_process_group(monkeypatch):
    """A rank that never talks to the others (TRANSFORM) must not rendezvous: no port, no gloo, no
    scontrol lookup, and none of the flakes those bring on a laptop or a shared login node."""

    def fail_init(*_args, **_kwargs):
        raise AssertionError("no process group must be initialized")

    monkeypatch.setattr(rt_dist.dist, "init_process_group", fail_init)
    monkeypatch.setattr(rt_dist.shutil, "which", lambda _name: pytest.fail("scontrol must not be looked up"))
    assert rt_dist.setup_gpu(2, 1, process_group=False) == (1, 1)
    assert rt_dist.setup_gpu(2, 2, process_group=False) == (None, None)  # a rank past the world is idle

    from konfai.transformer import Transformer

    assert Transformer.uses_collectives is False
    assert Predictor.uses_collectives is False
    assert rt_dist.DistributedObject.uses_collectives is True


@pytest.mark.parametrize(("ranks", "expected"), [(1, False), (2, True)])
def test_only_several_ranks_rendezvous(monkeypatch, ranks: int, expected: bool):
    """A single rank has nobody to talk to: no process group, as on Windows. Its gloo threads also
    outlived the run on macOS, where the process then crashed at exit in SimpleITK's destructors."""
    asked: list[bool] = []

    def setup_gpu(world_size: int, rank: int | None = None, process_group: bool = True):
        asked.append(process_group)
        return None, None  # the rank returns before any work

    class Workflow(rt_dist.DistributedObject):
        def setup(self, world_size: int) -> None:
            pass

        def run_process(self, *args, **kwargs) -> None:  # pragma: no cover - never reached
            pass

    workflow = Workflow("rendezvous")
    workflow.dataloader = [[] for _ in range(ranks)]
    monkeypatch.setattr(rt_dist, "setup_gpu", setup_gpu)
    workflow(0)
    assert asked == [expected]


def _gloo_rendezvous(monkeypatch) -> dict[str, object]:
    """Drive ``setup_gpu`` down its gloo branch and report what it passed to torch, plus the
    interface gloo would have read as it built its device (``interface``)."""
    initialized: dict[str, object] = {}

    def init_process_group(**kwargs) -> None:
        initialized.update(kwargs, interface=os.environ.get("GLOO_SOCKET_IFNAME"))

    monkeypatch.setattr(rt_dist.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(rt_dist.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(rt_dist.dist, "init_process_group", init_process_group)
    monkeypatch.setenv("KONFAI_MASTER_PORT", "29500")
    return initialized


@pytest.mark.skipif(os.name == "nt", reason="setup_gpu builds no process group on Windows")
def test_a_single_node_gloo_world_is_pinned_to_the_loopback_interface(monkeypatch):
    """gloo resolves the host's name to choose an interface, and a macOS runner's ``.local`` name
    resolves to nothing: the single-node world rendezvous over the loopback that carries it."""
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    initialized = _gloo_rendezvous(monkeypatch)

    assert rt_dist.setup_gpu(2, 0) == (0, 0)

    assert initialized["backend"] == "gloo"
    assert initialized["init_method"] == "tcp://localhost:29500"
    assert initialized["interface"] in {name for _, name in rt_dist.socket.if_nameindex()}
    # The pin lasts the rendezvous: left behind, it would follow a later multi-node group, or a
    # child of this process, onto an interface that reaches no other node.
    assert "GLOO_SOCKET_IFNAME" not in os.environ


@pytest.mark.skipif(os.name == "nt", reason="setup_gpu builds no process group on Windows")
def test_an_explicit_gloo_interface_keeps_authority(monkeypatch):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "eth0")
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    initialized = _gloo_rendezvous(monkeypatch)

    rt_dist.setup_gpu(2, 0)

    assert initialized["interface"] == "eth0"
    assert os.environ["GLOO_SOCKET_IFNAME"] == "eth0"


@pytest.mark.skipif(os.name == "nt", reason="setup_gpu builds no process group on Windows")
def test_a_multi_node_gloo_world_is_left_to_its_own_interface(monkeypatch):
    """Off this host the loopback reaches no other rank: only a localhost rendezvous is pinned."""
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[001-002]")
    monkeypatch.setattr(rt_dist.shutil, "which", lambda _name: "/usr/bin/scontrol")
    monkeypatch.setattr(rt_dist.subprocess, "check_output", lambda *_args, **_kwargs: "node001\nnode002\n")
    initialized = _gloo_rendezvous(monkeypatch)

    rt_dist.setup_gpu(2, 0)

    assert initialized["init_method"] == "tcp://node001:29500"
    assert initialized["interface"] is None
    assert "GLOO_SOCKET_IFNAME" not in os.environ


@pytest.mark.skipif(os.name == "nt", reason="setup_gpu builds no process group on Windows")
def test_a_multi_node_job_that_cannot_name_its_master_is_refused(monkeypatch):
    """Without scontrol every node of a cluster job would rendezvous on its own localhost and wait
    there until the timeout. A single-node job still rendezvous on localhost."""
    nodes = {"count": 2}
    monkeypatch.setitem(
        sys.modules,
        "submitit",
        SimpleNamespace(JobEnvironment=lambda: SimpleNamespace(global_rank=2, local_rank=0, num_nodes=nodes["count"])),
    )
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[001-002]")
    monkeypatch.setattr(rt_dist.shutil, "which", lambda _name: None)
    initialized = _gloo_rendezvous(monkeypatch)

    with pytest.raises(ConfigError, match="scontrol not found"):
        rt_dist.setup_gpu(4, None)
    assert initialized == {}

    nodes["count"] = 1
    assert rt_dist.setup_gpu(4, None) == (2, 0)
    assert initialized["init_method"] == "tcp://localhost:29500"


def test_synchronize_data_no_dist(monkeypatch):
    """Without an active process group the local data is returned as-is."""
    monkeypatch.setattr(rt_dist.dist, "is_initialized", lambda: False)
    assert rt_dist.synchronize_data(4, 0, {"a": 1}) == [{"a": 1}]


def _run_execute(monkeypatch, obj):
    monkeypatch.setattr(rt_dist, "Log", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(rt_dist, "TensorBoard", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(rt_dist.mp, "spawn", lambda *a, **k: None)
    # These cover what the PARENT does before execution, and stub spawn to skip the run itself.
    # The single-rank inline path would run it here instead, so it is turned off.
    monkeypatch.setenv("KONFAI_INLINE_SINGLE_RANK", "0")
    rt_dist.execute_distributed_object(obj, gpu=None, cpu=1)


def test_execute_seeds_parent_before_setup(monkeypatch):
    """The parent process is seeded before ``setup``: two runs with one seed draw the same there."""

    recorded = []

    class FakeObject(rt_dist.DistributedObject):
        def __init__(self) -> None:
            super().__init__("fake-seeded")
            self.manual_seed = 123

        def setup(self, world_size: int) -> None:
            recorded.append(random.random())

        def run_process(self, *args, **kwargs) -> None:  # pragma: no cover - not spawned
            pass

    _run_execute(monkeypatch, FakeObject())
    _run_execute(monkeypatch, FakeObject())

    assert recorded[0] == recorded[1]


def test_execute_puts_the_callers_rng_and_cudnn_flags_back(monkeypatch):
    """Inline (the single-rank default) the run seeds the CALLER's process: a notebook or Slicer
    whose own random draws must not become a function of having run a KonfAI workflow."""

    class FakeObject(rt_dist.DistributedObject):
        def __init__(self) -> None:
            super().__init__("fake-seeded")
            self.manual_seed = 123

        def setup(self, world_size: int) -> None:
            pass

        def run_process(self, *args, **kwargs) -> None:  # pragma: no cover - not spawned
            pass

    import numpy as np
    import torch

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    expected = (random.random(), float(np.random.random()), float(torch.rand(1)))
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    _run_execute(monkeypatch, FakeObject())
    assert (random.random(), float(np.random.random()), float(torch.rand(1))) == expected
    assert (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) == (True, False)


def test_preserved_rng_puts_the_three_cpu_generators_back():
    import numpy as np
    import torch

    rt_dist.seed_all(7)
    expected = (random.random(), float(np.random.random()), float(torch.rand(1)))
    rt_dist.seed_all(7)
    with rt_dist.preserved_rng():
        rt_dist.seed_all(123)
        random.random(), np.random.random(), torch.rand(1)
    assert (random.random(), float(np.random.random()), float(torch.rand(1))) == expected


def test_execute_puts_the_callers_cuda_rng_back(monkeypatch):
    """torch.manual_seed reseeds every CUDA generator too; a caller with CUDA up gets its own back."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    class FakeObject(rt_dist.DistributedObject):
        def __init__(self) -> None:
            super().__init__("fake-seeded")
            self.manual_seed = 123

        def setup(self, world_size: int) -> None:
            pass

        def run_process(self, *args, **kwargs) -> None:  # pragma: no cover - not spawned
            pass

    torch.cuda.init()
    torch.cuda.manual_seed_all(7)
    expected = float(torch.rand(1, device="cuda"))
    torch.cuda.manual_seed_all(7)
    _run_execute(monkeypatch, FakeObject())
    assert float(torch.rand(1, device="cuda")) == expected


# ---------------------------------------------------------------------------
# is_interactive_session must not crash when stdout has no isatty
# ---------------------------------------------------------------------------
class _FakeTTY:
    def isatty(self) -> bool:
        return True


class _LogProxy:
    """Mimics Log/MinimalLog: write/flush/fileno only, no isatty."""

    def write(self, msg: str) -> None:
        pass

    def flush(self) -> None:
        pass


def test_is_interactive_session_survives_stdout_without_isatty(monkeypatch) -> None:
    # During a run stdout is swapped for a Log proxy that has no isatty; an unconditional
    # stdout.isatty() call raises AttributeError. It must degrade to non-interactive.
    monkeypatch.setattr(sys, "stdin", _FakeTTY())
    monkeypatch.setattr(sys, "stdout", _LogProxy())

    assert is_interactive_session() is False


def test_is_interactive_session_true_on_real_tty(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", _FakeTTY())
    monkeypatch.setattr(sys, "stdout", _FakeTTY())

    assert is_interactive_session() is True


def test_clear_directory_except_logs_keeps_the_live_log(tmp_path):
    """The overwrite branch clears a run directory AROUND its open log_*.txt: an rmtree unlinks the
    open file (parent-process lines and crash tracebacks lost; PermissionError on Windows)."""
    from konfai.utils.runtime import clear_directory_except_logs

    run_dir = tmp_path / "Statistics" / "RUN"
    (run_dir / "events").mkdir(parents=True)
    (run_dir / "events" / "tb.bin").write_text("stale")
    (run_dir / "Config.yml").write_text("stale")
    log = open(run_dir / "log_0.txt", "a", buffering=1)
    try:
        log.write("parent line\n")
        clear_directory_except_logs(run_dir)
        log.write("after clear\n")
    finally:
        log.close()

    assert not (run_dir / "events").exists()
    assert not (run_dir / "Config.yml").exists()
    assert (run_dir / "log_0.txt").read_text() == "parent line\nafter clear\n"


class _FileLikeMirror:
    """What the mirror target is inside an MCP job or a slurm run: a plain writable, not a TTY."""

    def __init__(self):
        self.written = ""

    def write(self, msg):
        self.written += msg

    def flush(self):
        pass

    def isatty(self):
        return False


def test_the_mirror_folds_a_redrawing_bar_off_a_terminal(monkeypatch):
    """A file appends what a terminal overwrites: mirrored raw, one bar's animation alone reached 1.6 MB
    per short run. Off a terminal the mirror keeps at most one folded frame per throttle window, never
    loses the bar's final state, and leaves normal messages untouched."""
    monkeypatch.setattr(sys, "stdout", _FileLikeMirror())
    monkeypatch.setattr(sys, "stderr", sys.stdout)
    monkeypatch.setenv("KONFAI_VERBOSE", "True")
    log = rt_logg.MinimalLog(rank=0)

    for i in range(500):
        log.write(f"\rProgress: {i}/500")
    log.write("\n")
    log.write("epoch 1 done\n")

    mirrored = log._stdout_bak.written
    frames = [line for line in mirrored.splitlines() if line.startswith("Progress:")]
    # The throttle admits the first frame; every skipped one stays pending, so the final state is the
    # second and last, not 500 lines, and never a lost 499/500.
    assert frames[0] == "Progress: 0/500"
    assert frames[-1] == "Progress: 499/500"
    assert len(frames) < 500 / 10, f"the animation was archived, not folded: {len(frames)} frames"
    assert "\r" not in mirrored
    assert mirrored.endswith("epoch 1 done\n")


def test_the_mirror_stays_raw_on_a_terminal(monkeypatch):
    """On a real console the animation IS the point: the fold must not degrade the interactive view."""

    class _Terminal(_FileLikeMirror):
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdout", _Terminal())
    monkeypatch.setattr(sys, "stderr", sys.stdout)
    monkeypatch.setenv("KONFAI_VERBOSE", "True")
    log = rt_logg.MinimalLog(rank=0)

    log.write("\rProgress: 1/2")
    log.write("\rProgress: 2/2")

    assert log._stdout_bak.written == "\rProgress: 1/2\rProgress: 2/2"


def test_a_crlf_line_is_a_message_not_a_bar_frame(monkeypatch):
    """'warning\\r\\n' folds to the text after its last \\r: nothing. Classified as a redraw it would
    vanish from the mirror and the log file both; a CRLF terminator is not an animation."""
    monkeypatch.setattr(sys, "stdout", _FileLikeMirror())
    monkeypatch.setattr(sys, "stderr", sys.stdout)
    monkeypatch.setenv("KONFAI_VERBOSE", "True")
    log = rt_logg.MinimalLog(rank=0)

    log.write("\rProgress: 1/100")
    log.write("important warning\r\n")

    assert "important warning" in log._stdout_bak.written
    assert log._buffered_line == "important warning"


def test_a_bars_cursor_moves_leave_no_blank_line_off_a_terminal(monkeypatch):
    """Nested tqdm bars position themselves with bare newlines and clear themselves with an empty frame;
    off a terminal both left blank lines, more of them than lines of content."""
    monkeypatch.setattr(sys, "stdout", _FileLikeMirror())
    monkeypatch.setattr(sys, "stderr", sys.stdout)
    monkeypatch.setenv("KONFAI_VERBOSE", "True")
    log = rt_logg.MinimalLog(rank=0)

    log.write("[KonfAI] start")
    log.write("\n")
    log.write("\n")  # a nested bar moving down to its position
    log.write("\rProgress: 1/2")
    log.write("\r          \r")  # the bar clearing itself on close
    log.write("\n")
    log.write("done\n")

    assert log._stdout_bak.written == "[KonfAI] start\nProgress: 1/2\ndone\n"


def test_the_bar_state_held_by_the_throttle_lands_on_exit(monkeypatch):
    """A run's last writes are often throttled frames; dropped at __exit__, the job sink would freeze on
    a stale frame and misreport where the run actually stopped."""
    monkeypatch.setattr(sys, "stdout", _FileLikeMirror())
    monkeypatch.setattr(sys, "stderr", sys.stdout)
    monkeypatch.setenv("KONFAI_VERBOSE", "True")
    mirror = sys.stdout

    with rt_logg.MinimalLog(rank=0) as log:
        log.write("\rProgress: 0/100")
        log.write("\rProgress: 100/100")  # inside the throttle window: withheld

    assert mirror.written.splitlines()[-1] == "Progress: 100/100"


def test_interleaved_bars_each_keep_their_final_state(monkeypatch):
    """Train and validation redraw through the same stream; a single pending slot would let one bar
    overwrite the other's withheld frame, ending the run without its final state ever mirrored."""
    monkeypatch.setattr(sys, "stdout", _FileLikeMirror())
    monkeypatch.setattr(sys, "stderr", sys.stdout)
    monkeypatch.setenv("KONFAI_VERBOSE", "True")
    mirror = sys.stdout

    with rt_logg.MinimalLog(rank=0) as log:
        for i in range(50):
            log.write(f"\rTrain: {i}/50")
            log.write(f"\rVal: {i}/50")

    lines = mirror.written.splitlines()
    assert "Train: 49/50" in lines and "Val: 49/50" in lines


def test_record_keeps_detail_in_the_log_without_printing_it(tmp_path, monkeypatch):
    """The run's log is where a run is read after the fact, so detail too long for a console belongs
    there (the TRANSFORM plan). Without a Log installed there is no run directory to keep it in, and
    recording is a no-op rather than a print that would land on the console it exists to spare."""
    monkeypatch.setattr(sys, "stdout", _FileLikeMirror())
    monkeypatch.setattr(sys, "stderr", sys.stdout)
    monkeypatch.setenv("KONFAI_VERBOSE", "True")
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    monkeypatch.setenv("KONFAI_STATE", "TRAIN")
    monkeypatch.setenv("KONFAI_STATISTICS_DIRECTORY", str(tmp_path))
    mirror = sys.stdout

    rt_logg.record("nothing is installed: this goes nowhere")
    with rt_dist.Log("RUN", 0) as log:
        rt_logg.record("line one\nline two")
        log.write("printed\n")

    assert "goes nowhere" not in mirror.written
    assert "line one" not in mirror.written, "recorded detail must not reach the console"
    assert "printed" in mirror.written
    header, *lines = (tmp_path / "RUN" / "log_0.txt").read_text().splitlines()
    assert header.startswith("[KonfAI] ==== TRAIN 'RUN' rank 0 |")
    assert lines == ["line one", "line two", "printed"]


@pytest.mark.parametrize("raised", [ConfigError("'Trainer.Dataset' is empty."), ZeroDivisionError("division by zero")])
def test_the_run_log_ends_on_why_the_run_failed(tmp_path, monkeypatch, raised: Exception) -> None:
    """The file ended on the last progress line: the reason a run failed reached the console only."""
    monkeypatch.setattr(sys, "stdout", _FileLikeMirror())
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    monkeypatch.setenv("KONFAI_STATE", "TRAIN")
    monkeypatch.setenv("KONFAI_STATISTICS_DIRECTORY", str(tmp_path))

    with pytest.raises(type(raised)), rt_dist.Log("RUN", 0):
        raise raised

    text = (tmp_path / "RUN" / "log_0.txt").read_text()
    if isinstance(raised, ConfigError):
        assert text.rstrip().endswith("[Config] 'Trainer.Dataset' is empty.")
        assert "Traceback" not in text
    else:
        assert "Traceback" in text and text.rstrip().endswith("ZeroDivisionError: division by zero")


def test_an_inline_rank_writes_its_log_once_and_warnings_read_as_konfai(tmp_path, monkeypatch):
    """A single rank runs inside the launcher's Log on the same file: each line lands there once, and
    KonfAI's warnings and logger records carry the console's own prefix."""
    import logging
    import warnings

    monkeypatch.setattr(sys, "stdout", _FileLikeMirror())
    monkeypatch.setattr(sys, "stderr", sys.stdout)
    monkeypatch.setenv("KONFAI_VERBOSE", "True")
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    monkeypatch.setenv("KONFAI_STATE", "TRAIN")
    monkeypatch.setenv("KONFAI_STATISTICS_DIRECTORY", str(tmp_path))

    with warnings.catch_warnings(), rt_dist.Log("RUN", 0) as outer, rt_dist.Log("RUN", 0) as inner:
        assert inner is outer
        warnings.simplefilter("always")
        print("hello")
        rt_logg._show_warning("constant case", UserWarning, rt_logg.__file__, 1)
        # A stacklevel names the caller's frame, outside KonfAI: the category still marks it as KonfAI's.
        rt_logg._show_warning("from a caller", KonfAIWarning, "/elsewhere/script.py", 1)
        logging.getLogger("konfai.test").warning("head resized")

    header, *lines = (tmp_path / "RUN" / "log_0.txt").read_text().splitlines()
    assert header.startswith("[KonfAI] ==== TRAIN 'RUN'")
    assert lines == [
        "hello",
        "[KonfAI] WARNING: constant case",
        "[KonfAI] WARNING: from a caller",
        "[KonfAI] WARNING: head resized",
    ]
    assert rt_logg._CONSOLE_HANDLER not in logging.getLogger("konfai").handlers


# ---------------------------------------------------------------------------
# data_log: the TensorBoard strategies
# ---------------------------------------------------------------------------


def test_data_log_entries_parse_to_a_strategy_and_a_count_per_target() -> None:
    parsed = rt_logg.DataLog.parse(["CT/IMAGES/5", "Generator:Head:Tanh/VIDEO/2"])
    assert parsed == {"CT": (rt_logg.DataLog.IMAGES, 5), "Generator.Head.Tanh": (rt_logg.DataLog.VIDEO, 2)}


@pytest.mark.parametrize(
    "entry", ["CT/VIDEO", "CT/MOVIE/2", "CT/IMAGES/two", "CT/IMAGE/0", "CT/IMAGES/-1", "/IMAGE/1", " /VIDEO/2"]
)
def test_a_malformed_data_log_entry_is_a_config_error_naming_it(entry: str) -> None:
    with pytest.raises(ConfigError, match=f"'{entry}'") as refusal:
        rt_logg.DataLog.parse([entry])
    assert "IMAGES, VIDEO" in str(refusal.value)


@pytest.mark.parametrize("strategy", [rt_logg.DataLog.IMAGE, rt_logg.DataLog.IMAGES])
@pytest.mark.parametrize("shape", [(2, 1, 7, 4, 5), (2, 3, 7, 4, 5), (2, 1, 4, 5), (2, 3, 4, 5), (2, 4, 5)])
@pytest.mark.parametrize("constant", [False, True])
def test_image_logs_keep_numpy_pixels_and_copy_only_displayed_tensor_elements(monkeypatch, strategy, shape, constant):
    import numpy as np
    import torch

    tensor = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape)
    if constant:
        tensor.fill_(3)
    tensor.requires_grad_()
    images = []
    board = SimpleNamespace(add_image=lambda *args: images.append(args), add_images=lambda *args: images.append(args))
    strategy(board, "CT", tensor.detach().numpy(), 9)
    copied = []
    original_cpu = torch.Tensor.cpu

    def counted_cpu(value, *args, **kwargs):
        copied.append(value.numel())
        return original_cpu(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    strategy(board, "CT", tensor, 9)

    assert len(images) == 2
    assert images[0][0] == images[1][0] == "CT"
    assert images[0][2] == images[1][2] == 9
    np.testing.assert_array_equal(images[1][1], images[0][1])
    assert sum(copied) == images[0][1].size  # IMAGE copies only the first sample, IMAGES all selected samples


@pytest.mark.parametrize(
    ("strategy", "method", "shape"),
    [
        (rt_logg.DataLog.VIDEO, "add_video", (2, 3, 4, 5, 6)),
        (rt_logg.DataLog.SIGNAL, "add_scalars", (2, 3, 1, 1)),
        (rt_logg.DataLog.AUDIO, "add_audio", (1, 1, 20)),
    ],
)
def test_other_log_strategies_accept_tensors_with_the_same_values_as_numpy(strategy, method, shape):
    import numpy as np
    import torch

    tensor = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape).requires_grad_()

    def render(layer):
        calls = []
        board = SimpleNamespace(**{method: lambda *args: calls.append(args)})
        strategy(board, "signal", layer, 11)
        return calls

    np.testing.assert_equal(render(tensor), render(tensor.detach().numpy()))


class _Board:
    """Keeps what a VIDEO log hands TensorBoard."""

    def add_video(self, name: str, video, it: int) -> None:
        self.video = video


def _normalized(array):
    return (array - array.min()) / (array.max() - array.min())


def test_a_video_log_shows_each_sample_its_own_frames() -> None:
    """A [B, C, Z, Y, X] layer: one video per sample, a frame per channel, its middle slice in grey."""
    import numpy as np

    layer = np.random.default_rng(0).random((2, 3, 4, 5, 6))
    board = _Board()
    rt_logg.DataLog.VIDEO(board, "CT", layer, 0)
    assert board.video.shape == (2, 3, 3, 5, 6)
    for sample in range(2):
        for frame in range(3):
            for colour in range(3):
                np.testing.assert_allclose(board.video[sample, frame, colour], _normalized(layer[sample, frame, 2]))


def test_a_video_log_of_three_channels_shows_them_as_the_colours_of_each_frame() -> None:
    """A [B, T, C, Z, Y, X] layer of three channels: each channel is one colour of its own sample's frame."""
    import numpy as np

    layer = np.random.default_rng(0).random((2, 2, 3, 4, 5, 6))
    board = _Board()
    rt_logg.DataLog.VIDEO(board, "CT", layer, 0)
    assert board.video.shape == (2, 2, 3, 5, 6)
    for sample in range(2):
        for frame in range(2):
            np.testing.assert_allclose(board.video[sample, frame], _normalized(layer[sample, frame, :, 2]))


# ---------------------------------------------------------------------------
# A single rank runs in this process; more than one still spawns
# ---------------------------------------------------------------------------
def _execute_counting(
    monkeypatch,
    *,
    cpu: int,
    inline: str | None,
    gpu: list[int] | None = None,
    size: int = 1,
    setups: list[int] | None = None,
):
    """Run execute_distributed_object and report who executed: the rank ran here, or spawn was called.

    ``inline`` is the KONFAI_INLINE_SINGLE_RANK value, or None to leave it unset and exercise the default."""
    ran_here: list[int | None] = []
    spawned: list[int] = []

    class FakeObject(rt_dist.DistributedObject):
        uses_collectives = False  # spawns several ranks on Windows too, which opens no process group

        def __init__(self) -> None:
            super().__init__("fake-inline")
            self.size = size

        def setup(self, world_size: int) -> None:
            if setups is not None:
                setups.append(world_size)

        def __call__(self, rank: int | None = None) -> None:
            ran_here.append(rank)

        def run_process(self, *args, **kwargs) -> None:  # pragma: no cover - never spawned here
            pass

    monkeypatch.setattr(rt_dist, "Log", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(rt_dist, "TensorBoard", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(rt_dist.mp, "spawn", lambda obj, nprocs=1, **k: spawned.append(nprocs))
    if inline is None:
        monkeypatch.delenv("KONFAI_INLINE_SINGLE_RANK", raising=False)
    else:
        monkeypatch.setenv("KONFAI_INLINE_SINGLE_RANK", inline)
    rt_dist.execute_distributed_object(FakeObject(), gpu=gpu, cpu=cpu)
    return ran_here, spawned


def test_a_single_rank_runs_in_this_process(monkeypatch) -> None:
    """A spawned child is a fresh interpreter: re-imported torch, re-initialised CUDA, the whole payload
    unpickled. With one rank there is nothing to parallelise, so that start-up buys only isolation."""
    ran_here, spawned = _execute_counting(monkeypatch, cpu=1, inline="1")

    assert ran_here == [0], "the single rank must run here, as rank 0"
    assert spawned == [], "no child may be spawned for one rank"


def test_a_model_split_over_gpus_is_set_up_with_every_gpu(monkeypatch) -> None:
    """``setup`` takes the GPU count and divides it by ``size`` itself (one dataloader list per model
    replica); a rank past the replicas finds no dataloader and returns. Dividing before ``setup`` too
    left two GPUs x size 2 with no replica at all."""
    seen: list[int] = []
    _, spawned = _execute_counting(monkeypatch, cpu=1, inline="1", gpu=[0, 1, 2, 3], size=2, setups=seen)
    assert seen == [4] and spawned == [4]


def test_more_than_one_rank_still_spawns(monkeypatch) -> None:
    """Ranks that must run side by side still need their own processes."""
    ran_here, spawned = _execute_counting(monkeypatch, cpu=3, inline="1")

    assert spawned == [3]
    assert ran_here == []


def test_the_inline_path_can_be_turned_off(monkeypatch) -> None:
    """An embedded caller (Slicer, the apps server) outlives the run and would inherit this process's
    CUDA context; KONFAI_INLINE_SINGLE_RANK=0 gives it the child back."""
    ran_here, spawned = _execute_counting(monkeypatch, cpu=1, inline="0")

    assert spawned == [1]
    assert ran_here == []


def test_the_inline_path_is_the_default(monkeypatch) -> None:
    """Unset is the shape every CLI run takes; a default flipped to False would otherwise go unnoticed."""
    ran_here, spawned = _execute_counting(monkeypatch, cpu=1, inline=None)

    assert ran_here == [0]
    assert spawned == []


def test_the_workflow_wrapper_lets_an_interrupt_and_a_refusal_reach_its_caller(monkeypatch) -> None:
    """The wrapper exits nothing: an in-process caller catches Ctrl+C and a designed refusal as
    exceptions (the CLI turns them into exit statuses)."""
    raised: list[BaseException] = []

    def execute(*args, **kwargs):
        raise raised[-1]

    monkeypatch.setattr(rt_dist, "execute_distributed_object", execute)

    @rt_dist.run_distributed_app
    def workflow(gpu: list[int] = [], cpu: int = 1):
        return object()

    for error in (KeyboardInterrupt(), ConfigError("refused")):
        raised.append(error)
        with pytest.raises(type(error)):
            workflow()


def test_a_warning_the_build_raises_reaches_an_api_caller(monkeypatch) -> None:
    """Only the CLI spells a build warning as KonfAI's console does: a Python caller still records it."""
    import warnings

    monkeypatch.setattr(rt_dist, "execute_distributed_object", lambda *args, **kwargs: None)

    @rt_dist.run_distributed_app
    def workflow(gpu: list[int] = [], cpu: int = 1):
        warnings.warn("[Config] Unknown key(s) in the Trainer configuration.", KonfAIWarning, stacklevel=2)
        return object()

    with pytest.warns(KonfAIWarning, match="Unknown key"):
        workflow()


def _budget_applied(
    monkeypatch,
    cores: int,
    ranks: str | None,
    omp: str | None,
    platform: str = "linux",
    world_size: int | None = None,
) -> list[int]:
    calls: list[int] = []
    monkeypatch.setattr(rt_dist, "_cpu_budget_applied", False)
    monkeypatch.setattr(rt_dist.sys, "platform", platform)
    monkeypatch.setattr(rt_dist, "available_cpus", lambda: cores)
    monkeypatch.setattr(rt_dist.torch, "set_num_threads", calls.append)
    for key, value in (("KONFAI_LOCAL_RANKS", ranks), ("OMP_NUM_THREADS", omp)):
        monkeypatch.delenv(key, raising=False)
        if value is not None:
            monkeypatch.setenv(key, value)
    rt_dist.apply_cpu_thread_budget(world_size)
    return calls


def test_available_cpus_is_the_tighter_of_affinity_and_cgroup_quota(monkeypatch, tmp_path) -> None:
    """A container sees the host's cores in full while being allowed a fraction: os.cpu_count says
    64 where the affinity mask says 8 and cpu.max says 4; every thread past 4 is contention."""
    from konfai.utils import budget as bd

    monkeypatch.setattr(bd.os, "sched_getaffinity", lambda pid: set(range(8)), raising=False)
    root = tmp_path / "cgroup"
    (root / "a" / "b").mkdir(parents=True)
    proc = tmp_path / "proc_self_cgroup"
    proc.write_text("0::/a/b\n")
    monkeypatch.setattr(bd, "_CGROUP_ROOT", str(root))
    monkeypatch.setattr(bd, "_PROC_SELF_CGROUP", str(proc))

    def available_cpus() -> int:
        bd.forget_cgroup_cpu_ceiling()  # the quota is read once per process: re-read it here
        return bd.available_cpus()

    assert available_cpus() == 8  # no quota file: the affinity mask
    (root / "a" / "b" / "cpu.max").write_text("max 100000\n")
    assert available_cpus() == 8  # unbounded quota
    (root / "a" / "cpu.max").write_text("350000 100000\n")  # the quota sits on an ANCESTOR
    assert available_cpus() == 4  # 3.5 CPUs of quota round up
    (root / "a" / "b" / "cpu.max").write_text("garbage\n")  # a malformed file is skipped, not raised
    assert available_cpus() == 4
    (root / "a" / "b" / "cpu.max").write_text("100000 0\n")  # a zero period too
    assert available_cpus() == 4


@pytest.mark.parametrize("mount", ["cpu", "cpu,cpuacct"], ids=["symlinked-cpu", "joint-mount-only"])
def test_available_cpus_reads_a_cgroup_v1_quota(monkeypatch, tmp_path, mount: str) -> None:
    """cgroup v1 spells the same quota as two files under the cpu controller; -1 is unbounded.

    The controller mounts under the name it is mounted with: most distributions mount the joint
    ``cpu,cpuacct`` and drop a ``cpu`` symlink beside it, some only the joint one. Looking under
    ``cpu`` alone found no quota file there and read the host's whole CPU count inside a container.
    """
    from konfai.utils import budget as bd

    monkeypatch.setattr(bd.os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)
    root = tmp_path / "cgroup"
    (root / mount / "docker" / "abc").mkdir(parents=True)
    proc = tmp_path / "proc_self_cgroup"
    proc.write_text("3:cpu,cpuacct:/docker/abc\n1:memory:/docker/abc\n")
    monkeypatch.setattr(bd, "_CGROUP_ROOT", str(root))
    monkeypatch.setattr(bd, "_PROC_SELF_CGROUP", str(proc))
    (root / mount / "docker" / "abc" / "cpu.cfs_quota_us").write_text("-1\n")
    (root / mount / "docker" / "abc" / "cpu.cfs_period_us").write_text("100000\n")
    bd.forget_cgroup_cpu_ceiling()
    assert bd.available_cpus() == 16  # unbounded
    (root / mount / "docker" / "cpu.cfs_quota_us").write_text("250000\n")  # the ancestor's quota
    (root / mount / "docker" / "cpu.cfs_period_us").write_text("100000\n")
    bd.forget_cgroup_cpu_ceiling()
    assert bd.available_cpus() == 3  # 2.5 CPUs round up


def test_cpu_thread_budget_gives_itk_the_rank_share_torch_is_capped_out_of(monkeypatch) -> None:
    """Both pools are bounded by the RANK's share, and they take it differently: torch's cap is
    memory-bus saturation (0.7 s at 12 threads against 67 s at 24 for one gather), which ITK's
    resampler does not hit (10.98 s at 1, 1.11 s at 12, 0.65 s at 24 for one region). Capping ITK
    at torch's 12 left a third of a 24-core node idle; leaving it unbounded would oversubscribe the
    node across ranks. The share, whole, is neither."""
    sitk = pytest.importorskip("SimpleITK")
    before = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    try:
        assert _budget_applied(monkeypatch, cores=24, ranks="4", omp=None) == [6]
        assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == 6  # the share, under the cap
        assert _budget_applied(monkeypatch, cores=24, ranks=None, omp=None) == [12]
        assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == 24  # the whole share, over it
        assert _budget_applied(monkeypatch, cores=24, ranks="4", omp="20") == []
        assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == 20  # OMP_NUM_THREADS rules both
    finally:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(before)


def test_cpu_thread_budget_caps_torchs_every_core_default(monkeypatch) -> None:
    """torch defaults to one intraop thread per core; past bus saturation that only adds barrier
    contention (measured 0.7 s at 12 threads vs 67 s at 24 for the same gather)."""
    assert _budget_applied(monkeypatch, cores=24, ranks=None, omp=None) == [12]


def test_cpu_thread_budget_falls_back_to_the_world_size(monkeypatch) -> None:
    """``KONFAI_LOCAL_RANKS`` is the launcher's, and a direct ``execute_distributed_object`` call has
    no launcher: without the world size standing in, the divisor is 1 and each of four ranks sizes
    itself for the whole node, oversubscribing it fourfold."""
    assert _budget_applied(monkeypatch, cores=24, ranks=None, omp=None, world_size=4) == [6]
    assert _budget_applied(monkeypatch, cores=24, ranks=None, omp=None) == [12], "no count, no divisor"


def test_cpu_thread_budget_splits_the_node_between_ranks(monkeypatch) -> None:
    assert _budget_applied(monkeypatch, cores=24, ranks="4", omp=None) == [6]


def test_cpu_thread_budget_never_rounds_to_zero(monkeypatch) -> None:
    assert _budget_applied(monkeypatch, cores=2, ranks="4", omp=None) == [1]


def test_an_explicit_omp_setting_keeps_authority(monkeypatch) -> None:
    """torch honors OMP_NUM_THREADS at init; the budget must not override the user's choice."""
    assert _budget_applied(monkeypatch, cores=24, ranks="4", omp="20") == []


def test_cpu_thread_budget_is_applied_once_per_process(monkeypatch) -> None:
    """set_num_threads is documented as pre-parallel-work only, and the Python API runs several
    workflows in one process: a second application mid-process can crash the OpenMP runtime."""
    calls = _budget_applied(monkeypatch, cores=24, ranks=None, omp=None)
    rt_dist.apply_cpu_thread_budget()
    assert calls == [12]


def test_cpu_thread_budget_leaves_torch_alone_on_macos(monkeypatch) -> None:
    """On macOS set_num_threads intermittently crashes libomp once any parallel region ran (CI
    SIGSEGV, whichever workflow called it first): torch keeps its default. ITK and zarr, which
    that crash does not concern, still take the rank's share, or N ranks each use every core."""
    sitk = pytest.importorskip("SimpleITK")
    zarr = pytest.importorskip("zarr")
    before = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    concurrency = zarr.config.get("async.concurrency") if hasattr(zarr, "config") else None
    try:
        assert _budget_applied(monkeypatch, cores=24, ranks="4", omp=None, platform="darwin") == []
        assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == 6
        if concurrency is not None:
            assert zarr.config.get("async.concurrency") == 4
    finally:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(before)
        if concurrency is not None:
            zarr.config.set({"async.concurrency": concurrency})


@pytest.mark.parametrize("cores,expected", [(24, 8), (12, 4), (4, 4), (2, 2), (1, 1)])
def test_zarr_keeps_a_small_share_whole(monkeypatch, cores: int, expected: int) -> None:
    """A third of a 24-core share is the measured point; a third of four cores is one chunk in
    flight, which on a remote root is the whole of the read's parallelism."""
    zarr = pytest.importorskip("zarr")
    if not hasattr(zarr, "config"):  # 2.x has no config object, and no async reader to share the cores with
        pytest.skip("zarr 2.x has no async reader to size")
    previous = zarr.config.get("async.concurrency")
    try:
        _budget_applied(monkeypatch, cores=cores, ranks=None, omp=None)
        assert zarr.config.get("async.concurrency") == expected
    finally:
        zarr.config.set({"async.concurrency": previous})


@pytest.mark.parametrize("uses_collectives", [True, False])
def test_windows_refuses_several_ranks_only_where_they_must_talk(monkeypatch, uses_collectives: bool) -> None:
    """Windows gets no process group: TRAIN and EVALUATION ranks would each work alone (the metrics
    counted rank 0's cases only), while ranks that share only the work list still run."""
    spawned: list[int] = []

    class FakeObject(rt_dist.DistributedObject):
        def __init__(self) -> None:
            super().__init__("fake-windows")

        def setup(self, world_size: int) -> None:
            self.dataloader = [[] for _ in range(world_size)]

        def run_process(self, world_size, global_rank, local_rank, dataloaders) -> None:
            pass

    class WindowsOs:
        name = "nt"

        def __getattr__(self, attribute: str):
            return getattr(os, attribute)

    FakeObject.uses_collectives = uses_collectives
    monkeypatch.setattr(rt_dist, "os", WindowsOs())  # only the runtime sees Windows, not pathlib
    monkeypatch.setattr(rt_dist, "Log", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(rt_dist, "TensorBoard", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(rt_dist.mp, "spawn", lambda fn, nprocs, args=(): spawned.append(nprocs))

    if uses_collectives:
        with pytest.raises(ConfigError, match="Windows"):
            rt_dist.execute_distributed_object(FakeObject(), cpu=2, quiet=True)
    else:
        rt_dist.execute_distributed_object(FakeObject(), cpu=2, quiet=True)
    assert spawned == ([] if uses_collectives else [2])


def test_the_startup_line_takes_the_nested_phases_out_and_closes_on_other() -> None:
    """The sweep's format: disjoint phases, ``other`` closing the wall clock exactly. The cohort,
    the grids and the model are inside the build, the checkpoint inside the setup."""
    import time

    clock = rt_dist.StartupClock()
    clock._phases._spent = {"build": 1.0, "cases": 0.2, "grids": 0.1, "model": 0.3, "setup": 0.5, "checkpoint": 0.2}
    now = time.time()
    clock.started, clock.launched = now - 3.0, now - 0.5
    assert clock.report() == (
        "[KonfAI] startup 3.0 s = build 0.4 + cases 0.2 + grids 0.1 + model 0.3 + checkpoint 0.2"
        " + setup 0.3 + launch 0.5 + other 1.0"
    )
    clock.started = now - 0.4
    assert clock.report() is None  # a startup this short has nothing to account for


def test_rank_zero_reports_the_launchers_clock_as_it_starts(monkeypatch, capsys) -> None:
    """The clock built at the launcher's entry crosses to the rank on the workflow object and is
    printed once, by rank 0, before the run: build, setup and launch each charged where they ran."""
    ran: list[tuple[int, rt_dist.StartupClock | None]] = []

    class FakeObject(rt_dist.DistributedObject):
        uses_collectives = False

        def __init__(self) -> None:
            super().__init__("fake-startup")

        def setup(self, world_size: int) -> None:
            self.dataloader = [[]]

        def run_process(self, world_size, global_rank, local_rank, dataloaders) -> None:
            ran.append((global_rank, self.startup_clock))

    monkeypatch.setattr(rt_dist, "Log", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(rt_dist, "TensorBoard", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(rt_dist.torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("KONFAI_INLINE_SINGLE_RANK", "1")
    clock = rt_dist.restart_startup_clock()
    clock.started -= 5.0  # a startup long enough to be reported
    rt_dist.execute_distributed_object(FakeObject(), gpu=None, cpu=1)

    assert ran == [(0, clock)]
    assert clock.spent("setup") > 0 and clock.launched is not None
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith("[KonfAI] startup 5.") and "+ launch 0.0 +" in line and "+ other" in line


def test_the_rank_pool_is_rebuilt_when_the_share_changes(monkeypatch) -> None:
    """A multi-rank build followed by an inline single-rank workflow changes the share within one
    process: the pool follows it instead of keeping the size of its first use."""
    monkeypatch.setattr(rt_dist, "_rank_pool", None)
    monkeypatch.setattr(rt_dist, "_rank_pool_share", 0)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    four = rt_dist.rank_pool()
    assert four is not None and four._max_workers == 4
    assert rt_dist.rank_pool() is four
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    eight = rt_dist.rank_pool()
    assert eight is not four and eight is not None and eight._max_workers == 8
    assert rt_dist.rank_pool() is eight
    # A share of one keeps no pool: the one built for the wider share is let go with its threads,
    # instead of idling for the rest of the process.
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert rt_dist.rank_pool() is None
    assert rt_dist._rank_pool is None
    with pytest.raises(RuntimeError):
        eight.submit(int)


def _map_in_child() -> None:
    seen: list[int] = []
    rt_dist.map_over_rank_pool(seen.append, [1, 2, 3])
    sys.exit(0 if sorted(seen) == [1, 2, 3] else 1)


@pytest.mark.skipif(sys.platform != "linux", reason="forking a threaded process is a Linux contract")
def test_a_forked_child_builds_its_own_rank_pool(monkeypatch) -> None:
    """A child inherits the executor's bookkeeping and none of its threads, so work handed to it
    waits forever; DataLoader workers fork the rank."""
    import multiprocessing

    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setattr(rt_dist, "_rank_pool", None)
    warmed: list[int] = []
    rt_dist.map_over_rank_pool(warmed.append, [1, 2, 3])  # the parent's pool exists and has run

    child = multiprocessing.get_context("fork").Process(target=_map_in_child)
    child.start()
    child.join(30)
    hung = child.is_alive()
    if hung:
        child.kill()
        child.join()
    assert not hung, "the child's read waits on threads it does not have"
    assert child.exitcode == 0


def test_a_rank_bounds_the_chunk_cache_by_its_own_share_of_the_budget(monkeypatch) -> None:
    """A spawned rank is a new process: a bound the launcher set is a module global it never sees,
    so the rank sets it at its own entry, from the budget every workflow's dataset resolves."""
    from konfai.utils import budget as budget_module
    from konfai.utils import ome_zarr

    class FakeBudget:
        def per_rank_bytes(self, world_size: int) -> float:
            return 96 << 20

        def work_bytes(self, world_size: int) -> float:
            # A declared budget is what the work may take, so it is published as it stands.
            return self.per_rank_bytes(world_size)

    class FakeDataset:
        def resolved_budget(self) -> FakeBudget:
            return FakeBudget()

    class FakeObject(rt_dist.DistributedObject):
        uses_collectives = False

        def __init__(self) -> None:
            super().__init__("fake-budget")
            self.dataset = FakeDataset()

        def setup(self, world_size: int) -> None:
            self.dataloader = [[]]

        def run_process(self, world_size, global_rank, local_rank, dataloaders) -> None:
            pass

    monkeypatch.setattr(rt_dist, "Log", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(rt_dist, "TensorBoard", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(rt_dist.torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("KONFAI_INLINE_SINGLE_RANK", "1")
    monkeypatch.setattr(budget_module, "_per_rank_bytes", None)
    rt_dist.execute_distributed_object(FakeObject(), gpu=None, cpu=1)

    assert budget_module.per_rank_budget_bytes() == 96 << 20
    assert ome_zarr._chunk_cache().capacity == ome_zarr.chunk_cache_capacity()


def test_run_distributed_app_refuses_a_kwarg_the_entrypoint_does_not_declare() -> None:
    """A kwarg outside the signature and the cluster set must refuse, not vanish: the silent drop
    is what forced main.py's --plan short-circuit."""

    class Sentinel(Exception):
        pass

    @rt_dist.run_distributed_app
    def build(gpu: list[int] | None = None, cpu: int | None = None) -> None:
        raise Sentinel

    with pytest.raises(ConfigError, match="plan"):
        build(plan=True)

    # The tolerated names pass the gate and reach the build: the cluster set is read from the raw
    # kwargs and 'command' is the CLI dispatch discriminator only TRAIN/RESUME declares.
    with pytest.raises(Sentinel):
        build(command="PREDICTION")


def test_a_seed_makes_cudnn_deterministic_unless_the_run_benchmarks() -> None:
    assert rt_dist.cudnn_flags(None, False) == (True, False)
    assert rt_dist.cudnn_flags(7, False) == (False, True)
    assert rt_dist.cudnn_flags(7, True) == (True, False)
    assert rt_dist.cudnn_flags(None, True) == (True, False)


@pytest.mark.skipif(os.name == "nt", reason="SIGKILL is POSIX")
@pytest.mark.parametrize(
    ("how", "raised", "message"),
    [
        ("refuses", ConfigError, "Rank refuses."),
        ("is_killed", KonfAIError, "Rank 1 was killed by SIGKILL, the signal the kernel's out-of-memory killer"),
    ],
)
def test_a_spawned_rank_that_refuses_or_is_killed_reaches_the_caller_as_a_konfai_error(
    monkeypatch, tmp_path, how: str, raised: type, message: str
) -> None:
    """Two CPU ranks, the last one refuses or dies by SIGKILL (what the out-of-memory killer sends): the
    caller catches the rank's own refusal, or a KonfAIError naming the likely cause, not torch's
    ProcessRaisedException or ProcessExitedException."""
    from rank_failures import FailingLastRank

    monkeypatch.setenv("KONFAI_STATE", "TRAIN")
    monkeypatch.setenv("KONFAI_STATISTICS_DIRECTORY", str(tmp_path))
    with pytest.raises(raised) as refusal:
        execute_distributed_object(FailingLastRank(how), cpu=2, quiet=True)
    assert type(refusal.value) is raised
    assert message in str(refusal.value)


def test_tensorboard_without_its_executable_is_refused_before_the_setup(monkeypatch, tmp_path) -> None:
    """-tb launches the tensorboard executable: without it the run is refused with the extra to install,
    before the workflow's setup loads anything."""
    set_up: list[int] = []

    class Workflow(rt_dist.DistributedObject):
        def setup(self, world_size: int) -> None:
            set_up.append(world_size)
            self.dataloader = [[] for _ in range(world_size)]

        def run_process(self, world_size, global_rank, local_rank, dataloaders) -> None:
            pass

    monkeypatch.setenv("KONFAI_STATE", "TRAIN")
    monkeypatch.setenv("KONFAI_STATISTICS_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(rt_dist.shutil, "which", lambda name: None)
    monkeypatch.setattr("konfai.utils.runtime.logging.shutil.which", lambda name: None)
    with pytest.raises(ConfigError, match=r"pip install konfai\[tensorboard\]"):
        rt_dist.execute_distributed_object(Workflow("no-tensorboard"), cpu=1, quiet=True, tensorboard=True)
    assert set_up == []


@pytest.mark.parametrize(
    ("host", "bound", "shown"),
    [
        (None, "127.0.0.1", "127.0.0.1"),
        ("", "127.0.0.1", "127.0.0.1"),
        ("0.0.0.0", "0.0.0.0", "192.0.2.7"),
        ("10.1.2.3", "10.1.2.3", "10.1.2.3"),
        ("::1", "::1", "[::1]"),
    ],
)
def test_tensorboard_binds_loopback_unless_an_address_is_asked_for(
    monkeypatch, tmp_path, capsys, host: str | None, bound: str, shown: str
) -> None:
    """TensorBoard serves the curves and the DataLog images without authentication: -tb binds 127.0.0.1,
    KONFAI_TENSORBOARD_HOST names another address, and the printed URL is one a browser reaches (the
    network address for a wildcard bind)."""
    commands: list[list[str]] = []

    class Process:
        def __init__(self, command, **_kwargs) -> None:
            commands.append(command)

        def terminate(self) -> None:
            pass

        def wait(self) -> None:
            pass

    class Route:
        """The UDP socket that names this host's address on its default route."""

        def __init__(self, *_args) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> None:
            pass

        def connect(self, _address) -> None:
            pass

        def getsockname(self) -> tuple[str, int]:
            return ("192.0.2.7", 40000)

        def close(self) -> None:
            pass

    monkeypatch.setenv("KONFAI_STATE", "TRAIN")
    monkeypatch.setenv("KONFAI_STATISTICS_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("KONFAI_TENSORBOARD_PORT", "6123")
    if host is None:
        monkeypatch.delenv("KONFAI_TENSORBOARD_HOST", raising=False)
    else:
        monkeypatch.setenv("KONFAI_TENSORBOARD_HOST", host)
    monkeypatch.setattr(rt_logg.shutil, "which", lambda name: "/opt/bin/tensorboard")
    monkeypatch.setattr(rt_logg.subprocess, "Popen", Process)
    monkeypatch.setattr(rt_logg.socket, "socket", Route)
    with rt_logg.TensorBoard("RUN"):
        pass
    assert commands == [["/opt/bin/tensorboard", "--logdir", str(tmp_path / "RUN"), "--port", "6123", "--host", bound]]
    assert capsys.readouterr().out == f"[KonfAI] Tensorboard : http://{shown}:6123/\n"


@pytest.mark.skipif(os.name != "posix", reason="AF_UNIX socket paths are a POSIX limit")
def test_a_temporary_directory_too_long_for_a_socket_is_reached_through_a_short_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A socket path past the AF_UNIX limit made torch's shared-memory manager fail and the DataLoader hang: the run
    sees a short TMPDIR, and what it writes there lands in the long one. The link was removed as the run ended, under
    the temporary directory multiprocessing still had to remove: a FileNotFoundError at exit."""
    import socket
    import tempfile

    from konfai.utils.runtime.distributed import SOCKET_TMPDIR_MAX, short_socket_tmpdir

    long_dir = tmp_path / ("d" * 100)
    long_dir.mkdir()
    monkeypatch.setenv("TMPDIR", str(long_dir))
    monkeypatch.setattr(tempfile, "tempdir", None)
    with short_socket_tmpdir():
        short = os.environ["TMPDIR"]
        assert len(short) <= SOCKET_TMPDIR_MAX and tempfile.gettempdir() == short
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(os.path.join(tempfile.mkdtemp(prefix="pymp-"), "listener-12345678"))
        assert any(long_dir.iterdir())
    # The link outlives the run: multiprocessing names its temporary directory through it for the whole process.
    assert os.environ["TMPDIR"] == str(long_dir) and os.path.exists(short)
    with short_socket_tmpdir():
        assert os.environ["TMPDIR"] == short

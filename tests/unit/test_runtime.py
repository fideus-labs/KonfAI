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
from konfai.utils.errors import ConfigError
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
    assert rt_dist.DistributedObject.uses_collectives is True


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
    """The parent process (which runs the train/val split) must be seeded."""

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
    assert (tmp_path / "RUN" / "log_0.txt").read_text() == "line one\nline two\nprinted\n"


# ---------------------------------------------------------------------------
# A single rank runs in this process; more than one still spawns
# ---------------------------------------------------------------------------
def _execute_counting(monkeypatch, *, cpu: int, inline: str | None):
    """Run execute_distributed_object and report who executed: the rank ran here, or spawn was called.

    ``inline`` is the KONFAI_INLINE_SINGLE_RANK value, or None to leave it unset and exercise the default."""
    ran_here: list[int | None] = []
    spawned: list[int] = []

    class FakeObject(rt_dist.DistributedObject):
        def __init__(self) -> None:
            super().__init__("fake-inline")

        def setup(self, world_size: int) -> None:
            pass

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
    rt_dist.execute_distributed_object(FakeObject(), gpu=None, cpu=cpu)
    return ran_here, spawned


def test_a_single_rank_runs_in_this_process(monkeypatch) -> None:
    """A spawned child is a fresh interpreter: re-imported torch, re-initialised CUDA, the whole payload
    unpickled. With one rank there is nothing to parallelise, so that start-up buys only isolation."""
    ran_here, spawned = _execute_counting(monkeypatch, cpu=1, inline="1")

    assert ran_here == [0], "the single rank must run here, as rank 0"
    assert spawned == [], "no child may be spawned for one rank"


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
    assert bd.available_cpus() == 8  # no quota file: the affinity mask
    (root / "a" / "b" / "cpu.max").write_text("max 100000\n")
    assert bd.available_cpus() == 8  # unbounded quota
    (root / "a" / "cpu.max").write_text("350000 100000\n")  # the quota sits on an ANCESTOR
    assert bd.available_cpus() == 4  # 3.5 CPUs of quota round up
    (root / "a" / "b" / "cpu.max").write_text("garbage\n")  # a malformed file is skipped, not raised
    assert bd.available_cpus() == 4
    (root / "a" / "b" / "cpu.max").write_text("100000 0\n")  # a zero period too
    assert bd.available_cpus() == 4


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
    assert bd.available_cpus() == 16  # unbounded
    (root / mount / "docker" / "cpu.cfs_quota_us").write_text("250000\n")  # the ancestor's quota
    (root / mount / "docker" / "cpu.cfs_period_us").write_text("100000\n")
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


def test_cpu_thread_budget_skips_macos(monkeypatch) -> None:
    """On macOS set_num_threads intermittently crashes libomp once any parallel region ran (CI
    SIGSEGV, whichever workflow called it first); the default stays."""
    assert _budget_applied(monkeypatch, cores=24, ranks=None, omp=None, platform="darwin") == []


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

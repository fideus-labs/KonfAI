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


"""The distributed runtime: one process per device, rank pools, thread budgets, the SLURM path."""

import inspect
import os
import random
import shutil
import socket
import subprocess  # nosec B404
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from functools import wraps
from typing import Any, TypeVar

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

try:
    import pynvml

    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False
from konfai import (
    cuda_visible_devices,
)
from konfai.utils.budget import available_cpus, node_local_ranks, set_per_rank_budget
from konfai.utils.clock import StartupClock, restart_startup_clock, startup_clock
from konfai.utils.errors import ConfigError, KonfAIError
from konfai.utils.runtime.environment import ClusterKwargs
from konfai.utils.runtime.logging import Log, TensorBoard
from konfai.utils.utils import env_flag

_T = TypeVar("_T")

_cpu_budget_applied = False
_rank_pool: ThreadPoolExecutor | None = None
_rank_pool_share = 0  # the share the pool was sized for
_rank_pool_lock = threading.Lock()


@contextmanager
def preserved_rng() -> Iterator[None]:
    """Snapshot random, numpy and torch's CPU generator, plus every CUDA generator when CUDA is already
    initialised, and put them back on exit.

    ``torch.manual_seed`` reseeds the CUDA generators too, so they belong in the snapshot; reading them
    would initialise CUDA in a caller that never asked (a CPU data-loader worker, a notebook), so the
    gate is ``torch.cuda.is_initialized()``, not ``is_available()``.
    """
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def seed_all(seed: int) -> None:
    """Seed random, numpy and torch (``torch.manual_seed`` reaches every CUDA generator too)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DistributedObject(ABC):
    """Base class for trainer, predictor, and evaluator distributed workflows."""

    #: Whether the ranks talk to each other (DDP, gathers). A workflow whose ranks only share the
    #: work list sets it False and runs without a process group: no rendezvous port, no gloo/NCCL.
    uses_collectives: bool = True

    def __init__(self, name: str) -> None:
        self.dataloader: list[list[DataLoader]]
        self.manual_seed: int | None = None
        self.name = name
        self.size = 1
        #: The launcher's clock, handed over before the ranks start; rank 0 reports it.
        self.startup_clock: StartupClock | None = None

    @abstractmethod
    def setup(self, world_size: int):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        cleanup()

    @abstractmethod
    def run_process(
        self,
        world_size: int,
        global_rank: int,
        local_rank: int,
        dataloaders: list[DataLoader],
    ):
        pass

    @staticmethod
    def get_measure(
        world_size: int,
        global_rank: int,
        gpu: int,
        models: dict[str, torch.nn.Module],
        n: int,
        sync: bool = True,
    ) -> dict[str, tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]]:
        data = {}
        for label, model in models.items():
            for name, network in model.get_networks().items():
                if network.measure is not None:
                    data[f"{name}{label}"] = (
                        network.measure.format_loss(True, n),
                        network.measure.format_loss(False, n),
                    )
        # `sync=False` skips the cross-rank all_gather: prediction shards whole cases per rank with unequal
        # batch counts, so a per-batch collective would hang once the shortest shard stops calling it.
        outputs = synchronize_data(world_size, gpu, data) if sync else [data]
        result: dict[str, tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]] = {}
        if global_rank == 0:
            for output in outputs:
                for k, v in output.items():
                    for t in range(len(v)):
                        for u, n in v[t].items():
                            if k not in result:
                                result[k] = ({}, {})
                            if u not in result[k][t]:
                                result[k][t][u] = (n[0], 0)  # type: ignore[index]
                            result[k][t][u] = (
                                result[k][t][u][0],
                                result[k][t][u][1] + n[1] / world_size,  # type: ignore[index]
                            )
        return result

    @property
    def world_size(self) -> int:
        """How many ranks ``setup`` sharded the run over: one dataloader list per rank."""
        return len(self.dataloader)

    def rank_dataloaders(self, global_rank: int) -> "list[DataLoader]":
        return self.dataloader[global_rank]

    def _bound_chunk_cache(self, world_size: int) -> None:
        """Bound the decoded-chunk cache by this rank's share of the memory budget.

        Set here, on the rank: a spawned rank is a new process, and a bound set by the launcher is
        a module global the child never sees.
        """
        from konfai.utils.ome_zarr import bound_chunk_cache

        dataset = getattr(self, "dataset", None)
        if dataset is not None and hasattr(dataset, "resolved_budget"):
            # work_bytes and not per_rank_bytes: the interpreter and the imaging libraries are
            # resident before the rank reads anything, and a budget that ignores them spends those
            # bytes twice. Measured on the rank, which is where they were paid.
            set_per_rank_budget(dataset.resolved_budget().work_bytes(node_local_ranks(world_size)))
            bound_chunk_cache()

    def __call__(self, rank: int | None = None) -> None:
        world_size = self.world_size
        global_rank, local_rank = setup_gpu(world_size, rank, process_group=self.uses_collectives)
        if global_rank is None or local_rank is None:
            return
        apply_cpu_thread_budget(world_size)
        self._bound_chunk_cache(world_size)
        with Log(self.name, global_rank):
            if torch.cuda.is_available() and _PYNVML_AVAILABLE:
                pynvml.nvmlInit()
            if self.manual_seed is not None:
                seed_all(self.manual_seed * world_size + global_rank)
            torch.backends.cudnn.benchmark = self.manual_seed is None
            torch.backends.cudnn.deterministic = self.manual_seed is not None
            dataloaders = self.rank_dataloaders(global_rank)
            # device_count as well: on a process whose CUDA runtime latched BEFORE the launcher
            # narrowed CUDA_VISIBLE_DEVICES to nothing, is_available() stays True while the count
            # honestly reads 0, and set_device would pin a GPU the launch explicitly excluded.
            if torch.cuda.is_available() and 0 <= local_rank < torch.cuda.device_count():
                torch.cuda.set_device(local_rank)
            if global_rank == 0 and self.startup_clock is not None and (startup := self.startup_clock.report()):
                print(startup)
            try:
                self.run_process(world_size, global_rank, local_rank, dataloaders)
            finally:
                cleanup()
                if torch.cuda.is_available() and _PYNVML_AVAILABLE:
                    pynvml.nvmlShutdown()


def run_distributed_app(
    func: Callable[..., DistributedObject],
) -> Callable[..., None]:
    """Wrap a workflow factory so it executes with KonfAI runtime conventions."""

    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        params = sig.parameters
        kwargs_fun = {k: v for k, v in kwargs.items() if k in params}

        bound = sig.bind_partial(*args, **kwargs_fun)
        bound.apply_defaults()
        is_cluster = "resubmit" in kwargs
        # The auto memory budget is a NODE budget, but build-time sizing (the evaluation auto-patch)
        # runs while ``func(...)`` constructs the workflow: before the spawn where world_size exists.
        # The launcher therefore leaves the per-node rank count in the environment, and restores it
        # after: a leak would silently shrink a later in-process run (tests, embedded Python).
        local_ranks = len(list(bound.arguments.get("gpu") or [])) or int(bound.arguments.get("cpu") or 1)
        previous_local_ranks = os.environ.get("KONFAI_LOCAL_RANKS")
        os.environ["KONFAI_LOCAL_RANKS"] = str(max(1, local_ranks))
        try:
            with restart_startup_clock().phase("build"):
                workflow = func(*args, **kwargs_fun)
            execute_distributed_object(
                workflow,
                gpu=bound.arguments.get("gpu", []),
                cpu=bound.arguments.get("cpu", 1),
                overwrite=bool(bound.arguments.get("overwrite", False)),
                quiet=bool(bound.arguments.get("quiet", False)),
                tensorboard=bool(bound.arguments.get("tensorboard", False)),
                cluster_kwargs=(
                    {
                        "name": kwargs["name"],
                        "memory": kwargs["memory"],
                        "num_nodes": kwargs["num_nodes"],
                        "time_limit": kwargs["time_limit"],
                        "resubmit": bool(kwargs.get("resubmit", False)),
                    }
                    if is_cluster
                    else None
                ),
            )
        except KeyboardInterrupt:
            print("\n[KonfAI] Manual interruption (Ctrl+C)")
        except KonfAIError as error:
            # A designed refusal: the message says what is wrong and the remedy what to change.
            # The traceback under it is framework internals: 28 lines burying the 3 that matter --
            # so it is shown only to a reader who asked (KONFAI_DEBUG=1).
            if env_flag("KONFAI_DEBUG", False):
                raise
            print(str(error).strip(), file=sys.stderr)
            sys.exit(1)
        finally:
            if previous_local_ranks is None:
                os.environ.pop("KONFAI_LOCAL_RANKS", None)
            else:
                os.environ["KONFAI_LOCAL_RANKS"] = previous_local_ranks

    return wrapper


def _runs_inline(world_size: int) -> bool:
    """Whether the single rank runs here instead of in a spawned child.

    A spawned child is a fresh interpreter: it re-imports torch, re-initialises CUDA and unpickles the
    whole payload before doing any work: measured at ~3 s, which a short prediction pays in full. With
    one rank there is nothing to parallelise, so that cost buys only isolation.

    Isolation is worth keeping where the caller outlives the run: an embedded interpreter (Slicer, the
    apps server) would inherit this process's CUDA context and its memory. ``KONFAI_INLINE_SINGLE_RANK``
    is the switch: default on for a CLI run, set it to 0 to force the child back.
    """
    if world_size != 1:
        return False
    return env_flag("KONFAI_INLINE_SINGLE_RANK", True)


def execute_distributed_object(
    distributed_object: DistributedObject,
    *,
    gpu: list[int] | None = None,
    cpu: int | None = 1,
    overwrite: bool = False,
    quiet: bool = False,
    tensorboard: bool = False,
    cluster_kwargs: ClusterKwargs | None = None,
) -> None:
    """
    Execute a previously built KonfAI workflow object.

    Parameters
    ----------
    distributed_object : DistributedObject
        Configured workflow returned by a build step.
    gpu : list[int] | None, optional
        GPU ids exposed to the workflow.
    cpu : int | None, optional
        Number of CPU workers when running without GPUs.
    overwrite : bool, optional
        Whether existing outputs may be overwritten.
    quiet : bool, optional
        Whether console output should be reduced.
    tensorboard : bool, optional
        Whether TensorBoard should be started for the workflow.
    cluster_kwargs : dict[str, Any] | None, optional
        Optional cluster submission parameters used by ``submitit``.
    """
    gpu_ids = [] if gpu is None else list(gpu)
    cpu_workers = 1 if cpu is None else int(cpu)
    if cpu_workers < 1:
        raise ConfigError(f"cpu={cpu!r} is not a rank count.", "Pass cpu=1 or more (the CLI refuses it the same way).")

    managed_env = [
        "CUDA_VISIBLE_DEVICES",
        "KONFAI_OVERWRITE",
        "KONFAI_CONFIG_MODE",
        "KONFAI_TENSORBOARD_PORT",
        "KONFAI_MASTER_PORT",
        "KONFAI_VERBOSE",
        "KONFAI_CLUSTER",
    ]
    previous_env = {key: os.environ.get(key) for key in managed_env}
    # The run seeds the process-wide RNGs and sets the cudnn flags; inline (the single-rank default)
    # that process is the caller's (a notebook, Slicer), so what it found is put back.
    previous_cudnn = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)

    with preserved_rng():
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpu_ids if i >= 0])
            os.environ["KONFAI_OVERWRITE"] = str(overwrite)
            os.environ["KONFAI_CONFIG_MODE"] = "Done"
            if tensorboard:
                os.environ["KONFAI_TENSORBOARD_PORT"] = str(find_free_port())
            if "KONFAI_MASTER_PORT" not in os.environ:
                os.environ["KONFAI_MASTER_PORT"] = str(find_free_port())
            os.environ["KONFAI_VERBOSE"] = str(not quiet)

            cluster_config = cluster_kwargs
            if cluster_config is not None:
                os.environ["KONFAI_OVERWRITE"] = "True"
                os.environ["KONFAI_CLUSTER"] = "True"

            clock = startup_clock()
            with distributed_object as configured_object:
                with Log(configured_object.name, 0):
                    if configured_object.manual_seed is not None:
                        seed_all(configured_object.manual_seed)
                    if cluster_config is not None:
                        if cluster_config["resubmit"]:
                            # Auto-requeue is not implemented; warn instead of silently dropping the flag.
                            print(
                                "[KonfAI] WARNING: --resubmit is not implemented yet; this job will NOT "
                                "auto-requeue at the time limit. Relaunch manually with the RESUME command "
                                "pointing at the latest checkpoint to continue training."
                            )
                        with clock.phase("setup"):
                            configured_object.setup(len(gpu_ids) * cluster_config["num_nodes"])
                        clock.launch()
                        configured_object.startup_clock = clock
                        import submitit

                        executor = submitit.AutoExecutor(folder="./Cluster/")
                        executor.update_parameters(
                            name=cluster_config["name"],
                            mem_gb=cluster_config["memory"],
                            gpus_per_node=len(gpu_ids),
                            tasks_per_node=len(gpu_ids) // configured_object.size,
                            cpus_per_task=1,
                            nodes=cluster_config["num_nodes"],
                            timeout_min=cluster_config["time_limit"],
                        )
                        with TensorBoard(configured_object.name):
                            executor.submit(configured_object)
                        return

                    world_size = len(gpu_ids)
                    if world_size == 0:
                        world_size = cpu_workers
                    with clock.phase("setup"):
                        configured_object.setup(world_size)
                    # Share tensors through /dev/shm files instead of one file descriptor per tensor:
                    # spawning a worker that pickles a loaded model can otherwise exhaust the process
                    # open-file limit ("Too many open files"), e.g. under Slicer's embedded Python.
                    mp.set_sharing_strategy("file_system")
                    clock.launch()
                    configured_object.startup_clock = clock
                    with TensorBoard(configured_object.name):
                        if _runs_inline(world_size):
                            configured_object(0)
                        else:
                            mp.spawn(configured_object, nprocs=world_size)
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous_cudnn


def _forget_rank_pool() -> None:
    """A forked child inherits the executor's bookkeeping and none of its threads: work submitted to
    it waits forever. The child builds its own on first use."""
    global _rank_pool, _rank_pool_share, _rank_pool_lock
    _rank_pool, _rank_pool_share, _rank_pool_lock = None, 0, threading.Lock()


if hasattr(os, "register_at_fork"):  # POSIX only: a platform without fork inherits nothing to forget
    os.register_at_fork(after_in_child=_forget_rank_pool)


def rank_cpu_share(world_size: int | None = None) -> int:
    """The cores this rank may use: the node's, divided between its local ranks, or exactly
    ``OMP_NUM_THREADS`` when that is set.

    One number for every consumer: torch's intraop pool, ITK's, zarr's, :func:`rank_pool`. The world
    size stands in for a launcher that published no ``KONFAI_LOCAL_RANKS``.
    """
    explicit = os.environ.get("OMP_NUM_THREADS")
    if explicit:
        return max(1, int(explicit))
    return max(1, available_cpus() // node_local_ranks(world_size))


def rank_pool() -> ThreadPoolExecutor | None:
    """The rank's shared worker pool, for the host-side GIL-releasing work inside ONE case: sharding
    cases over ranks does nothing for a cohort of one.

    ``None`` at a share of one, where the work stays on the calling thread.
    """
    global _rank_pool, _rank_pool_share
    workers = rank_cpu_share()
    with _rank_pool_lock:
        # The share changes within one process when a multi-rank build is followed by an inline
        # single-rank workflow: the pool is rebuilt at the new size, the old one's idle threads let go.
        # A share of one keeps no pool at all, so the threads go with it.
        if _rank_pool is not None and _rank_pool_share != workers:
            _rank_pool.shutdown(wait=False)
            _rank_pool, _rank_pool_share = None, 0
        if workers <= 1:
            return None
        if _rank_pool is None:
            _rank_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="konfai-rank")
            _rank_pool_share = workers
    return _rank_pool


def map_over_rank_pool(work: Callable[[_T], None], items: Sequence[_T]) -> None:
    """Run ``work`` over ``items`` on the rank's pool, in the caller's thread when there is none.
    Every exception is raised, the first one first: a region half written is not a region."""
    pool = rank_pool()
    if pool is None or len(items) < 2:
        for item in items:
            work(item)
        return
    for future in [pool.submit(work, item) for item in items]:
        future.result()


def apply_cpu_thread_budget(world_size: int | None = None) -> None:
    """Give each rank a bounded share of the machine's cores instead of every library's every-core
    default -- torch's intraop pool AND ITK's, which does the host resample.

    Each of the node's local ranks gets its share of :func:`available_cpus`, and the two pools take
    that share differently. Torch's is capped at 12: past memory-bus saturation more intraop threads
    only add barrier contention, and on a hybrid 24-core CPU the 498^3 separable gather measures
    0.7 s at 12 threads and 67 s at 24. ITK's takes the share whole, because its resampler keeps
    scaling with it (the same region: 10.98 s at 1 thread, 1.11 s at 12, 0.65 s at 24). An explicit
    ``OMP_NUM_THREADS`` keeps authority over both (torch already honors it at init).

    Applied once per process, and never on macOS: torch documents ``set_num_threads`` as to be
    called before any parallel work, and the Python API runs several workflows in one process.
    On macOS the call intermittently crashes libomp with SIGSEGV once any parallel region ran,
    whichever call is the first; the saturation this bounds was measured on many-core Linux
    nodes, so macOS keeps torch's default.
    """
    global _cpu_budget_applied
    if sys.platform == "darwin" or _cpu_budget_applied:
        return
    explicit = os.environ.get("OMP_NUM_THREADS")
    cores = rank_cpu_share(world_size)
    # The cap is torch's alone: past memory-bus saturation its intraop pool only adds barrier
    # contention. ITK's pool is not the same animal -- its resampler, which does the host walk,
    # keeps scaling to the whole share (a fold-sized region through a displacement field: 10.98 s
    # at 1 thread, 1.11 s at 12, 0.65 s at 24), so capping it at 12 left a third of a 24-core node
    # idle and cost 15 s on a measured fold.
    share = int(explicit) if explicit else min(cores, 12)
    itk_share = cores
    if not explicit:
        torch.set_num_threads(share)
    try:
        import SimpleITK as sitk

        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(itk_share)
    except ImportError:
        pass
    try:
        import zarr

        # A THIRD of the share: a pipelined sweep runs three of these at once, the decode of the
        # region being read, the assembly of the one before it, the encode of the one being written.
        # Measured with the chain off the reading thread: 24 cores, ExaSPIM 513x1331x1776 through a stored affine, two runs per point,
        # nothing else moved (a wrapper sets this alone, not OMP_NUM_THREADS, which would move ITK
        # with it). Wall clock, host path then device path:
        #   4  -> 12.1 / 13.4 s     4  -> 5.1 / 5.2 s
        #   8  -> 10.7 / 11.1 s     8  -> 4.9 / 5.1 s
        #   12 -> 11.0 / 11.3 s     12 -> 4.7 / 5.1 s
        #   24 -> 11.0 / 12.3 s     24 -> 5.3 / 5.4 s
        # A third wins or ties on both; starving it costs the host path 1.4 s and oversubscribing
        # costs the reader its own throughput (read busy 4.4-4.6 s at 8, 5.1-5.2 s at 24). A share
        # of a few cores keeps them all, up to four: a third of it is one chunk in flight, and on a
        # remote root that is the whole of the read's parallelism.
        if hasattr(zarr, "config"):  # 2.x has no config object, and no async reader to share the cores with
            zarr.config.set({"async.concurrency": max(min(cores, 4), cores // 3)})
    except ImportError:
        pass
    # Marked applied only once it is: a raise above (a bad OMP_NUM_THREADS) leaves the next call free to retry.
    _cpu_budget_applied = True


@contextmanager
def pin_gloo_to_loopback(local: bool) -> Iterator[None]:
    """Bind gloo to the loopback interface for a rendezvous whose ranks all sit on this host.

    gloo picks its interface by resolving the host's name. On a macOS runner that name is an mDNS
    ``.local`` name no resolver answers, and the rendezvous fails there rather than on the loopback
    that carries the whole single-node world (the streamed-prediction integration tests flaked on
    the macos-latest runner for exactly that). ``local`` says whether the world is that one: off
    this host the loopback reaches no other rank, and gloo's own resolution stands.

    gloo reads the variable as it builds the device, so it is taken back out of the environment
    once the group is up: a later multi-node rendezvous in the same process, or a child that
    inherits this environment, would otherwise be pinned to an interface reaching no other node.
    An explicit ``GLOO_SOCKET_IFNAME`` keeps authority and is left untouched, and a host with no
    loopback in ``if_nameindex`` is left to gloo's own resolution.
    """
    loopback = None
    if local and not os.environ.get("GLOO_SOCKET_IFNAME"):
        interfaces = {name for _, name in socket.if_nameindex()}
        loopback = next((name for name in ("lo", "lo0") if name in interfaces), None)
    if loopback is not None:
        os.environ["GLOO_SOCKET_IFNAME"] = loopback
    try:
        yield
    finally:
        if loopback is not None:
            os.environ.pop("GLOO_SOCKET_IFNAME", None)


def setup_gpu(world_size: int, rank: int | None = None, process_group: bool = True) -> tuple[int | None, int | None]:
    """Resolve the rank and, with ``process_group``, initialize torch distributed on it."""
    if rank is None:
        import submitit

        job_env = submitit.JobEnvironment()
        global_rank = job_env.global_rank
        local_rank = job_env.local_rank
    else:
        global_rank = rank
        local_rank = rank
    if global_rank >= world_size:
        return None, None
    if os.name == "nt" or not process_group:
        return global_rank, local_rank
    try:
        nodelist = os.getenv("SLURM_JOB_NODELIST")
        if nodelist is None:
            raise RuntimeError("SLURM_JOB_NODELIST is not set.")
        scontrol_path = shutil.which("scontrol")
        if scontrol_path is None:
            raise FileNotFoundError("scontrol not found in PATH")
        # `scontrol show hostnames` prints one host per line; for a multi-node job take only the FIRST
        # (the rendezvous master). Without this the whole newline-joined list leaks into init_method
        # (tcp://node001\nnode002:port) and init_process_group can never rendezvous.
        host_name = (
            subprocess.check_output(  # nosec B603
                [scontrol_path, "show", "hostnames", nodelist], text=True, stderr=subprocess.DEVNULL
            )
            .strip()
            .splitlines()[0]
        )
    except Exception:
        host_name = "localhost"
    port = os.environ.get("KONFAI_MASTER_PORT")
    if not port:
        port = str(find_free_port())
        os.environ["KONFAI_MASTER_PORT"] = port
    if dist.is_nccl_available() and torch.cuda.is_available() and len(cuda_visible_devices()):
        torch.cuda.empty_cache()
        dist.init_process_group(
            backend="nccl",
            rank=global_rank,
            init_method=f"tcp://{host_name}:{port}",
            world_size=world_size,
        )
    else:
        if not dist.is_initialized():
            with pin_gloo_to_loopback(local=host_name == "localhost"):
                dist.init_process_group(
                    backend="gloo",
                    init_method=f"tcp://{host_name}:{port}",
                    rank=global_rank,
                    world_size=world_size,
                )
    return global_rank, local_rank


def find_free_port():
    """Reserve and return an ephemeral TCP port on the current host."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def cleanup():
    """Destroy the active torch distributed process group when present."""
    if dist.is_initialized():
        dist.destroy_process_group()


def synchronize_data(world_size: int, gpu: int, data: Any) -> list[Any]:
    """Gather arbitrary Python objects across ranks when distributed is active."""
    if dist.is_initialized():
        outputs: list[Any] = [None] * world_size
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu)
        dist.all_gather_object(outputs, data)
    else:
        outputs = [data]
    return outputs

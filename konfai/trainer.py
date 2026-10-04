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

"""Training workflow entrypoints and orchestration for KonfAI."""

import math
import os
import random
import shutil
import signal
import tempfile
import threading
import warnings
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from ruamel.yaml import YAML
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]

from konfai import (
    checkpoints_directory,
    config_file,
    cuda_visible_devices,
    current_date,
    konfai_state,
    statistics_directory,
)
from konfai.data.data_manager import BatchSample, DatasetIter, DataTrain
from konfai.data.data_manager.subset import case_list_encoding
from konfai.network.network import Measure, Model, ModelLoader, NetState, Network, place_graph
from konfai.utils import vram
from konfai.utils.clock import SweepClock, startup_clock
from konfai.utils.config import apply_config, config, strict_config
from konfai.utils.errors import ConfigError, KonfAIWarning, TrainerError
from konfai.utils.live_control import LiveControl
from konfai.utils.pretrained import _evaluating
from konfai.utils.runtime import (
    DataLog,
    DistributedObject,
    NullSummaryWriter,
    ProgressBar,
    State,
    checkpoint_source,
    clear_directory_except_logs,
    configure_workflow_environment,
    confirm_overwrite_or_raise,
    description,
    preserved_rng,
    run_distributed_app,
    safe_torch_load,
    seed_all,
    synchronize_data,
)


def _checkpoint_score(path: Path, default: float) -> float:
    """Read only the score, releasing mapped storages before BEST prunes files."""
    state = safe_torch_load(path, torch.device("cpu"), mmap=True)
    return float(state.get("loss", default))


def _ddp_kwargs(model: Network, local_rank: int, size: int) -> dict[str, Any]:
    """DDP options for the graph's gradient accumulation cadence: an accumulating network needs the
    ordinary reducer, a graph stepping every batch keeps the static-graph fast path."""
    accumulates = any(
        network.optimizer is not None and network.nb_batch_per_step > 1 for network in model.get_networks().values()
    )
    options: dict[str, Any] = {"static_graph": not accumulates}
    if len(cuda_visible_devices()) and size == 1:
        options.update({"device_ids": [local_rank], "output_device": local_rank})
    return options


def _checkpoint_rng() -> dict[str, Any]:
    """Keep generator states in weights-only-loadable primitives and tensors."""
    numpy = cast(tuple[Any, ...], np.random.get_state())
    return {
        "python": random.getstate(),
        "numpy": (numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() and cuda_visible_devices() else None,
    }


def _restore_checkpoint_rng(states: dict[str, Any]) -> None:
    random.setstate(states["python"])
    numpy = states["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), numpy[2], numpy[3], numpy[4]))
    torch.set_rng_state(states["torch"])
    if states["cuda"] is not None:
        if len(states["cuda"]) != torch.cuda.device_count():
            raise TrainerError("RESUME requires the same CUDA device count to restore its saved generators.")
        torch.cuda.set_rng_state_all(states["cuda"])


class EarlyStoppingBase:
    """Minimal protocol for early stopping strategies used by :class:`Trainer`."""

    # Direction of the monitored score, read by early stopping and BEST-checkpoint retention. The
    # default (no EarlyStopping) monitors the summed loss, lower is better.
    mode: str = "min"

    def __init__(self):
        self.early_stop = False

    def is_stopped(self) -> bool:
        return self.early_stop

    def get_score(self, values: dict[str, float]):
        return sum(list(values.values()))

    def is_better(self, score: float, reference: float) -> bool:
        """True if `score` is strictly better than `reference` under this monitor's direction."""
        return score > reference if self.mode == "max" else score < reference

    @property
    def worst_score(self) -> float:
        """The worst possible score for this direction (a starting sentinel for best-tracking)."""
        return float("-inf") if self.mode == "max" else float("inf")

    def __call__(self, current_score: float) -> bool:
        return False

    def stop(self) -> None:
        self.early_stop = True


@config()
class EarlyStopping(EarlyStoppingBase):
    """
    Implements early stopping logic with configurable patience and monitored metrics.

    Attributes:
        monitor (list[str]): Metrics to monitor.
        patience (int): Number of checks with no improvement before stopping.
        min_delta (float): Minimum change to qualify as improvement.
        mode (str): "min" or "max" depending on optimization direction.
    """

    def __init__(
        self,
        monitor: list[str] | None = None,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        super().__init__()
        if mode not in {"min", "max"}:
            raise ConfigError(
                f"EarlyStopping.mode must be 'min' or 'max' (got '{mode}').",
                "It is the direction the monitored score improves in, and both early stopping and"
                " BEST-checkpoint retention read it.",
            )
        self.monitor = [] if monitor is None else monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        #: The run stopped for another reason than the patience (the learning rate at zero, no finite loss).
        self.stopped_otherwise = False

    def stop(self) -> None:
        self.stopped_otherwise = self.stopped_otherwise or not self.early_stop
        super().stop()

    def restore(self, state: dict[str, Any]) -> None:
        """Take back a RESUME cursor's state. A stop the patience made is decided again under this
        patience, so a RESUME with a larger one trains on; a stop for another reason stands."""
        self.counter = state["counter"]
        self.best_score = state["best_score"]
        # A cursor that does not record why it stopped keeps its stop.
        self.stopped_otherwise = state.get("stopped_otherwise", state["early_stop"])
        self.early_stop = self.stopped_otherwise or (state["early_stop"] and self.counter >= self.patience)

    def get_score(self, values: dict[str, float]):
        if len(self.monitor) == 0:
            return super().get_score(values)
        for v in self.monitor:
            if v not in values.keys():
                raise TrainerError(
                    f"Metric '{v}' specified in EarlyStopping.monitor not found in logged values. "
                    f"Available keys: {sorted(values.keys())}. Please check your configuration."
                )
        return sum([i for v, i in values.items() if v in self.monitor])

    def __call__(self, current_score: float) -> bool:
        if math.isfinite(current_score) and (self.best_score is None or not math.isfinite(self.best_score)):
            # The first finite score also repairs an undefined baseline restored from an older run.
            self.best_score = current_score
            self.counter = 0
            return self.early_stop

        if not math.isfinite(current_score) or self.best_score is None:
            improvement = float("-inf")  # undefined evaluations consume patience, never become the baseline
        elif self.mode == "min":
            improvement = self.best_score - current_score
        elif self.mode == "max":
            improvement = current_score - self.best_score
        else:
            raise TrainerError("Mode must be 'min' or 'max'.")

        if improvement > self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


def _on_host(value: Any) -> Any:
    """``value`` with every tensor copied into host memory. Containers keep their type and attributes
    (a state dict is an OrderedDict, possibly carrying ``_metadata``)."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        copy = value.__class__((key, _on_host(item)) for key, item in value.items())
        for name, attribute in getattr(value, "__dict__", {}).items():
            setattr(copy, name, attribute)
        return copy
    if isinstance(value, list | tuple):
        return value.__class__(_on_host(item) for item in value)
    return value


def _dataset(loader: DataLoader) -> DatasetIter:
    """The loader's ``DatasetIter``: ``DataLoader`` types it as a bare ``Dataset``."""
    return cast(DatasetIter, loader.dataset)


class _CheckpointWriter:
    """Serialises one checkpoint at a time on a thread of its own: ``submit`` joins the previous write,
    and a failure on the thread is raised by the next ``join`` on the training thread."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def submit(self, work: Callable[[], None]) -> None:
        self.join()
        self._thread = threading.Thread(target=self._run, args=(work,), name="konfai-checkpoint")
        self._thread.start()

    def _run(self, work: Callable[[], None]) -> None:
        try:
            work()
        except BaseException as error:  # carried to the training thread by join()
            self._error = error

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            error, self._error = self._error, None
            raise error


def _listed(names: set[str], shown: int = 10) -> str:
    """``names`` sorted, the first ``shown`` of them spelled out."""
    ordered = sorted(names)
    more = f" and {len(ordered) - shown} more" if len(ordered) > shown else ""
    return ", ".join(ordered[:shown]) + more


def _ema_network(model_ema: AveragedModel) -> Network:
    """The ``Network`` an EMA averages: ``AveragedModel`` types its copy as a bare ``Module``."""
    return cast(Network, model_ema.module)


def _traced_interventions(snapshot: Path) -> list[Any]:
    """The ``Interventions`` trace of a run's config snapshot, empty when it holds none."""
    data = YAML().load(snapshot.read_text(encoding="utf-8")) if snapshot.is_file() else None
    trace = data.get("Interventions") if isinstance(data, dict) else None
    return list(trace) if isinstance(trace, list) else []


def _record_interventions(snapshot: Path, entries: list[Any], it_validation: int | None = None) -> None:
    """Append ``entries`` to the snapshot's ``Interventions`` trace, once per iteration and key, and
    record ``it_validation`` under ``Trainer`` when given. Atomic."""
    if not snapshot.is_file():
        return
    yaml = YAML()
    with open(snapshot, encoding="utf-8") as file:
        data = yaml.load(file)
    if not isinstance(data, dict):
        return
    existing = data.get("Interventions")
    existing = list(existing) if isinstance(existing, list) else []
    seen = {(e.get("it"), e.get("key")) for e in existing if isinstance(e, dict)}
    data["Interventions"] = existing + [e for e in entries if (e.get("it"), e.get("key")) not in seen]
    if it_validation is not None and isinstance(data.get("Trainer"), dict):
        data["Trainer"]["it_validation"] = it_validation
    tmp = snapshot.with_name(f"{snapshot.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as file:
        yaml.dump(data, file)
    os.replace(tmp, snapshot)


class _Trainer:
    """Training loop for one process, distributed or standalone: epochs with optional validation,
    autocast, EMA, early stopping, TensorBoard logging, checkpoint saving (ALL or BEST).

    Used as a context manager (``with _Trainer(...) as trainer:``) inside :class:`Trainer`.
    """

    # Poll cadence (iterations), rank-aligned, shared by the validation-request and live-tunable
    # pollers and by the progress bar's description refresh.
    _LIVE_POLL_INTERVAL = 20

    def __init__(
        self,
        world_size: int,
        global_rank: int,
        local_rank: int,
        size: int,
        train_name: str,
        early_stopping: EarlyStopping | None,
        data_log: list[str] | None,
        save_checkpoint_mode: str,
        epochs: int,
        epoch: int,
        autocast: bool,
        it_validation: int | None,
        it_lr_update: int | None,
        it: int,
        model: Model,
        model_ema: AveragedModel | None,
        config_snapshot: Path,
        dataloader_training: DataLoader,
        dataloader_validation: DataLoader | None = None,
        auto_patched: bool = False,
        resume_state: dict[str, Any] | None = None,
    ) -> None:
        self.world_size = world_size
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.size = size
        self._validate_now = False  # set by SIGUSR1 to request an on-demand validation (see __enter__)
        self.save_checkpoint_mode = save_checkpoint_mode
        self.train_name = train_name
        self.epochs = epochs
        self.epoch = epoch
        self.model = model
        self.dataloader_training = dataloader_training
        self.dataloader_validation = dataloader_validation
        self.autocast = autocast
        self.model_ema = model_ema
        self.early_stopping = EarlyStoppingBase() if early_stopping is None else early_stopping
        self._resume_state = resume_state
        self._epoch_complete = False
        self._deferred_score: float | None = None
        self._resume_cursor: dict[str, Any] = {
            "version": 1,
            "kind": "unavailable",
            "reason": "The epoch has not completed; its sample cursor and pending gradients were not saved.",
        }
        if resume_state is not None:
            if resume_state["batches_per_epoch"] != len(dataloader_training):
                raise TrainerError("RESUME requires the same number of training batches per epoch.")
            if isinstance(self.early_stopping, EarlyStopping) and resume_state.get("early_stopping"):
                self.early_stopping.restore(resume_state["early_stopping"])

        self.it_validation = len(dataloader_training) if it_validation is None else it_validation
        self.it_lr_update = len(dataloader_training) if it_lr_update is None else it_lr_update
        self.it = it
        self._declare_measure_window()
        # Live steering: control.json in the run dir; each new revision is applied at a DDP poll
        # boundary and recorded in the run's config snapshot.
        self._config_snapshot = config_snapshot
        self._live_control = LiveControl(statistics_directory() / self.train_name / "control.json")
        self._interventions: list[dict[str, Any]] = []
        if SummaryWriter is None:
            # A missing logger never refuses the run: only the curves are lost.
            if self.global_rank == 0:
                print(
                    "[KonfAI] TensorBoard is not installed: no curves or images will be logged"
                    " (pip install konfai[tensorboard] to keep them)."
                )
            self.tb: Any = NullSummaryWriter()
        elif self.global_rank != 0:
            self.tb = NullSummaryWriter()  # rank 0 alone writes the curves
        else:
            self.tb = SummaryWriter(log_dir=statistics_directory() / self.train_name / "tb")
        self._best_checkpoint_path: Path | None = None
        self._best_checkpoint_loss: float | None = None
        self._checkpoint_writer = _CheckpointWriter()
        #: Whether an OOM here restarts the run instead of ending it (a free patch axis declared).
        self._auto_patched = auto_patched
        #: The iteration of the last save: an exit at the same iteration has nothing new to record.
        self._saved_at_it = it
        self._loss_score: dict[str, float] = {}
        if self.global_rank == 0 and self.save_checkpoint_mode == "BEST":
            self._initialize_best_checkpoint_state()
        self.data_log = DataLog.parse(data_log)

    def __enter__(self):
        # SIGUSR1 requests an on-demand validation, consumed at a poll boundary (_poll_live_requests). Inline,
        # this is the caller's process: its handler is back at exit.
        self._previous_sigusr1 = None
        if (sigusr1 := getattr(signal, "SIGUSR1", None)) is not None:  # absent on Windows
            with suppress(ValueError, OSError):  # signals only install on the main thread
                self._previous_sigusr1 = signal.signal(sigusr1, lambda *_: setattr(self, "_validate_now", True))
        return self

    def __exit__(self, exc_type, value, traceback):
        """Close the writer and save an exit checkpoint only when it records anything.

        An exit at the last save's iteration and the auto-patch OOM restart add nothing; a failure
        that advanced keeps its crash-save.
        """
        try:
            if self.tb is not None:
                self.tb.close()
            oom_restart = (
                self._auto_patched and exc_type is not None and issubclass(exc_type, torch.cuda.OutOfMemoryError)
            )
            if not oom_restart and self.it != self._saved_at_it:
                self.checkpoint_save(None, crash=exc_type is not None)
            self._checkpoint_writer.join()
        finally:
            if self._previous_sigusr1 is not None:
                signal.signal(signal.SIGUSR1, self._previous_sigusr1)

    def _measures(self) -> list[Measure]:
        """Every measure the logs read: the model's networks', then the EMA copy's."""
        models = [self.model.module] + ([_ema_network(self.model_ema)] if self.model_ema is not None else [])
        return [
            network.measure
            for model in models
            for network in model.get_networks().values()
            if network.measure is not None
        ]

    def _declare_measure_window(self) -> None:
        """The widest window the logs read from a criterion's history: the training window or the
        validation pass. Everything older is only read as a running mean (ReduceLROnPlateau)."""
        validation = len(self.dataloader_validation) if self.dataloader_validation is not None else 0
        for measure in self._measures():
            measure.set_window(max(self.it_validation, validation))

    def _initialize_best_checkpoint_state(self) -> None:
        """Bootstrap BEST-checkpoint tracking once, including on resume.

        Crash saves (``crash_*.pt``) are never a contender for best and never pruned. When only
        unscored checkpoints remain, the newest is kept and the rest are pruned.
        """
        path = checkpoints_directory() / self.train_name
        if not path.exists():
            return

        all_checkpoints = sorted(
            p for p in path.glob("*.pt") if not p.name.startswith("crash_") and p.name != "resume_latest.pt"
        )
        best_loss = self.early_stopping.worst_score
        best_ckpt: Path | None = None
        for checkpoint_path in all_checkpoints:
            checkpoint_loss = _checkpoint_score(checkpoint_path, self.early_stopping.worst_score)
            if not math.isfinite(checkpoint_loss):
                checkpoint_loss = self.early_stopping.worst_score
            if self.early_stopping.is_better(checkpoint_loss, best_loss):
                best_loss = checkpoint_loss
                best_ckpt = checkpoint_path

        if best_ckpt is None and all_checkpoints:
            best_ckpt = max(all_checkpoints, key=lambda p: p.stat().st_mtime)
            best_loss = self.early_stopping.worst_score
        if best_ckpt is not None:
            self._best_checkpoint_path = best_ckpt
            self._best_checkpoint_loss = best_loss if math.isfinite(best_loss) else None
            for checkpoint_path in all_checkpoints:
                if checkpoint_path != best_ckpt:
                    checkpoint_path.unlink()

    def _update_best_checkpoint(self, checkpoint_path: Path, loss: float) -> None:
        """Keep only the current best checkpoint without rescanning all saves."""
        finite = math.isfinite(loss)
        is_new_best = self._best_checkpoint_loss is None or (
            finite and self.early_stopping.is_better(loss, self._best_checkpoint_loss)
        )
        if is_new_best:
            previous_best = self._best_checkpoint_path
            self._best_checkpoint_loss = loss if finite else None
            self._best_checkpoint_path = checkpoint_path
            if previous_best is not None and previous_best != checkpoint_path and previous_best.exists():
                previous_best.unlink()
            return

        checkpoint_path.unlink()

    def run(self) -> None:
        """Run the training loop one epoch at a time, with early stopping and augmentation resets."""
        _dataset(self.dataloader_training).load("Train")
        if self.dataloader_validation is not None:
            _dataset(self.dataloader_validation).load("Validation")
            if State[konfai_state()] != State.TRAIN and self._resume_state is None:
                self._validate()

        if self._resume_state is not None:
            if self.model_ema is not None:
                for name, network in _ema_network(self.model_ema).get_networks().items():
                    if name in self._resume_state.get("ema_iterations", {}):
                        network._it = self._resume_state["ema_iterations"][name]
            limits = self._resume_state.get("replay_limits", []) + self._replay_limits()
            if limits and self.global_rank == 0:
                warnings.warn(
                    "RESUME continues at next_epoch, but exact stochastic replay is not guaranteed: "
                    + "; ".join(sorted(set(limits))),
                    RuntimeWarning,
                    stacklevel=2,
                )
            measure_states = self._resume_state.get("measure_by_rank")
            if measure_states is not None:
                for kind, networks in measure_states[self.global_rank].items():
                    model = (
                        self.model.module
                        if kind == "Model"
                        else (_ema_network(self.model_ema) if self.model_ema is not None else None)
                    )
                    if model is None:
                        continue
                    for name, network in model.get_networks().items():
                        if network.measure is not None and name in networks:
                            network.measure.load_checkpoint_state(networks[name])
            _restore_checkpoint_rng(self._resume_state["rng_by_rank"][self.global_rank])

        if self.early_stopping.is_stopped():
            return

        with ProgressBar(
            iterable=range(self.epoch, self.epochs),
            leave=False,
            total=self.epochs,
            initial=self.epoch,
            desc="Progress",
        ) as epoch_tqdm:
            for self.epoch in epoch_tqdm:
                self.train()
                if self._epoch_complete:
                    if not self.early_stopping.is_stopped():
                        _dataset(self.dataloader_training).reset_augmentation("Train")
                    self._save_epoch_boundary()
                if self.early_stopping.is_stopped():
                    break

    def _replay_limits(self) -> list[str]:
        limits = []
        for name, loader in (("training", self.dataloader_training), ("validation", self.dataloader_validation)):
            if loader is None:
                continue
            if getattr(loader, "num_workers", 0):
                limits.append(f"{name} DataLoader worker RNG/cache state is not serialized")
            if getattr(getattr(loader, "dataset", None), "has_augmented_samples", False):
                limits.append(f"{name} augmentation draws/cache state is not serialized")
        return limits

    def _save_epoch_boundary(self) -> None:
        """Publish a continuation cursor only after every optimizer finishes its accumulation window.

        Every rank contributes its generator state; an incomplete window stays live for the next batch.
        """
        pending = [
            name
            for name, network in self.model.module.get_networks().items()
            if network.optimizer is not None
            and (
                network._it % network.nb_batch_per_step
                or any(
                    parameter.grad is not None
                    for group in network.optimizer.param_groups
                    for parameter in group["params"]
                )
            )
        ]
        models = {"Model": self.model.module}
        if self.model_ema is not None:
            models["Model_EMA"] = _ema_network(self.model_ema)
        measure_states = {
            kind: {
                name: network.measure.checkpoint_state()
                for name, network in model.get_networks().items()
                if network.measure is not None
            }
            for kind, model in models.items()
        }
        local = {"rng": _checkpoint_rng(), "pending": pending, "measures": measure_states}
        ranks: list[Any] = [local]
        if dist.is_initialized():
            ranks = [None] * self.world_size
            dist.all_gather_object(ranks, local)
        if any(rank["pending"] for rank in ranks):
            self._resume_cursor = {
                "version": 1,
                "kind": "unavailable",
                "reason": "The epoch ended with pending accumulated gradients, which are not serialized. "
                "Use a later epoch checkpoint where all optimizer accumulation windows close.",
            }
            if self.global_rank == 0:
                warnings.warn(
                    "Epoch checkpoint cannot RESUME: accumulation windows are still open. "
                    "No extra optimizer step was taken. Train through a later epoch whose batch count "
                    "closes every nb_batch_per_step window, or use a prior eligible epoch checkpoint.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            self._resume_cursor = {
                "version": 1,
                "kind": "epoch_boundary",
                "next_epoch": self.epoch + 1,
                "world_size": self.world_size,
                "batches_per_epoch": len(self.dataloader_training),
                "rng_by_rank": [rank["rng"] for rank in ranks],
                "measure_by_rank": [rank["measures"] for rank in ranks],
                "ema_iterations": (
                    {name: network._it for name, network in _ema_network(self.model_ema).get_networks().items()}
                    if self.model_ema is not None
                    else {}
                ),
                "replay_limits": self._replay_limits(),
                "early_stopping": (
                    {
                        key: getattr(self.early_stopping, key)
                        for key in ("counter", "best_score", "early_stop", "stopped_otherwise")
                    }
                    if isinstance(self.early_stopping, EarlyStopping)
                    else None
                ),
            }
        self.checkpoint_save(self._deferred_score)

    def train(self) -> None:
        """One training epoch: autocast, DDP or CPU, EMA, logging, checkpoints, validation at ``it_validation``."""
        self._epoch_complete = False
        self._deferred_score = None
        self._resume_cursor = {
            "version": 1,
            "kind": "unavailable",
            "reason": "The epoch has not completed; its sample cursor and pending gradients were not saved.",
        }
        self.model.train()
        self.model.module.set_state(NetState.TRAIN)
        if self.model_ema is not None:
            self.model_ema.eval()
            _ema_network(self.model_ema).set_state(NetState.TRAIN)

        clock = SweepClock()
        with (
            clock.phase("epoch"),
            ProgressBar(
                iterable=clock.waiting("wait(data)", enumerate(self.dataloader_training)),
                desc=f"Training : {description(self.model, self.model_ema)}",
                total=len(self.dataloader_training),
                leave=False,
                ncols=0,
            ) as batch_iter,
        ):
            for batch_index, batch_sample in batch_iter:
                with torch.amp.autocast("cuda", enabled=self.autocast):
                    # Forward and backward under one DDP context: a micro-batch that steps no
                    # optimizer accumulates locally, the one that closes the window reduces.
                    with self.model.module.accumulation_sync(self.model):
                        with clock.phase("forward"):
                            self.model(batch_sample, clock=clock)
                        with clock.phase("backward+step"):
                            self.model.module.backward(self.model)
                    if self.model_ema is not None:
                        with clock.phase("ema"):
                            self.model_ema.update_parameters(self.model.module)
                    self.it += 1

                    validate_now, pending = self._poll_live_requests()
                    if pending:
                        self._apply_live_tunables(pending)  # before update_lr, so a fresh LR anchors the next step

                    if (self.it) % self.it_lr_update == 0:
                        self.model.module.update_lr()

                    if self.dataloader_validation is not None and validate_now:
                        with clock.phase("validation"):
                            self._validate()  # on-demand (SIGUSR1): metrics only, no checkpoint / early-stopping

                    if (self.it) % self.it_validation == 0:
                        with clock.phase("telemetry"):
                            loss = self._train_log(batch_sample)
                            # The networks whose every loss of the window is NaN or infinite: nothing left
                            # to step on. Read here, on the training window, before the validation's log.
                            stalled = [
                                name
                                for name, network in self.model.module.get_networks().items()
                                if network.measure is not None and not network.measure.learns(self.it_validation)
                            ]

                        if self.dataloader_validation is not None:
                            with clock.phase("validation"):
                                loss = self._validate()

                        stop = False
                        if self.global_rank == 0:
                            # Default selection scores the losses only (lower is better); an explicit
                            # monitor reads the full dict with the direction of EarlyStopping.mode.
                            if isinstance(self.early_stopping, EarlyStopping) and self.early_stopping.monitor:
                                score = self.early_stopping.get_score(loss)
                            else:
                                score = self.early_stopping.get_score(self._loss_score)
                            stop = self.early_stopping(score)

                            if batch_index + 1 == len(self.dataloader_training):
                                self._deferred_score = score
                            else:
                                with clock.phase("checkpoint"):
                                    self.checkpoint_save(score)

                            # Stop once the schedulers have decayed the learning rate to zero.
                            optimizer = self.model.module.optimizer
                            if not stop and optimizer is not None and optimizer.param_groups[0]["lr"] <= 0:
                                self.early_stopping.stop()
                                stop = True
                            if not stop and stalled:
                                print(
                                    f"[KonfAI] Training stopped at iteration {self.it}: no loss of"
                                    f" {', '.join(stalled)} was finite over the last {self.it_validation} step(s)."
                                    " Under autocast the forward overflowed float16; add a normalization to the"
                                    " network, lower the learning rate, or set autocast: false.",
                                    flush=True,
                                )
                                self.early_stopping.stop()
                                stop = True

                        if self._broadcast_stop(stop):
                            self.early_stopping.stop()
                            self._epoch_complete = batch_index + 1 == len(self.dataloader_training)
                            break

                    self._epoch_complete = batch_index + 1 == len(self.dataloader_training)

                if self.it % self._LIVE_POLL_INTERVAL == 0:
                    with clock.phase("telemetry"):
                        batch_iter.set_description(
                            f"Training : {description(self.model, self.model_ema)}", refresh=False
                        )
        report = self._epoch_report(clock)
        if self.global_rank == 0 and report is not None:
            tqdm.tqdm.write(report)

    @staticmethod
    def _epoch_report(clock: SweepClock, min_seconds: float = 1.0) -> str | None:
        """One line accounting for the epoch's wall clock, in the sweep report's format, or ``None``
        below ``min_seconds``. ``forward`` is the graph walk alone; what no phase names is ``other``. On
        a device a phase is the time its kernels took to enqueue, so ``criteria`` carries the forward's
        kernels."""
        wall = clock.spent("epoch")
        if wall < min_seconds:
            return None
        phases = ("wait(data)", "forward", "criteria", "backward+step", "ema", "telemetry", "validation", "checkpoint")
        named = {phase: clock.spent(phase) for phase in phases}
        named["forward"] -= named["criteria"]
        parts = " + ".join(f"{phase} {value:.1f}" for phase, value in named.items())
        return f"[KonfAI] epoch {wall:.1f} s = {parts} + other {wall - sum(named.values()):.1f}"

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Validation pass: losses and metrics, model states updated, augmentation reset for validation.

        Returns:
            dict[str, float]: Validation losses and metrics; empty off rank 0 or without a validation set.
        """
        if self.dataloader_validation is None:
            return {}
        self.model.eval()
        self.model.module.set_state(NetState.PREDICTION)
        if self.model_ema is not None:
            _ema_network(self.model_ema).set_state(NetState.PREDICTION)

        batch_sample: BatchSample = {}
        # A rank's padding is its last batch (``Data._split_validation``): run, so every rank runs as
        # many forwards, and recorded weighing nothing.
        scored = len(self.dataloader_validation) - (
            1 if getattr(self.dataloader_validation.sampler, "padding", 0) else 0
        )
        measures = self._measures()
        try:
            with ProgressBar(
                iterable=enumerate(self.dataloader_validation),
                desc=f"Validation : {description(self.model, self.model_ema)}",
                total=len(self.dataloader_validation),
                leave=False,
                ncols=0,
            ) as batch_iter:
                for i, batch_sample in batch_iter:
                    for measure in measures:
                        measure.scored = i < scored
                    self.model(batch_sample)
                    if self.model_ema is not None:
                        self.model_ema.module(batch_sample)

                    if i % self._LIVE_POLL_INTERVAL == 0:
                        batch_iter.set_description(
                            f"Validation : {description(self.model, self.model_ema)}", refresh=False
                        )
        finally:
            for measure in measures:
                measure.scored = True
        _dataset(self.dataloader_validation).reset_augmentation("Validation")
        if dist.is_initialized():
            # Named, or NCCL warns that it guesses the device.
            dist.barrier(device_ids=[self._collective_device] if dist.get_backend() == "nccl" else None)
        self.model.train()
        self.model.module.set_state(NetState.TRAIN)
        if self.model_ema is not None:
            _ema_network(self.model_ema).set_state(NetState.TRAIN)
        return self._validation_log(batch_sample)

    @property
    def _collective_device(self) -> int:
        """The GPU this rank's collectives run on: the last of its model-parallel block."""
        return self.local_rank * self.size + self.size - 1

    def _broadcast_from_master(self, value: Any) -> Any:
        """Rank 0's value on every rank, one broadcast of a pickled object; callers cast as they need."""
        if not dist.is_initialized():
            return value
        payload = [value if self.global_rank == 0 else None]
        if torch.cuda.is_available():
            torch.cuda.set_device(self._collective_device)
        dist.broadcast_object_list(payload, src=0)
        return payload[0]

    def _broadcast_stop(self, stop: bool) -> bool:
        """Rank 0's stop decision on every rank, so the loop is left together."""
        return bool(self._broadcast_from_master(stop))

    def _poll_from_rank0(self, producer: Callable[[], Any]) -> Any:
        """Rank 0 produces the value and every rank receives it; single-process runs skip the collective."""
        value = producer() if self.global_rank == 0 else None
        if self.world_size > 1:
            value = self._broadcast_from_master(value)
        return value

    def _poll_live_requests(self) -> tuple[bool, dict[str, Any] | None]:
        """Rank 0's pending requests at the poll cadence, in one broadcast: an on-demand validation
        (SIGUSR1) and the control file's new tunables."""
        if self.it % self._LIVE_POLL_INTERVAL != 0:
            return False, None
        requested, pending = self._poll_from_rank0(lambda: (self._validate_now, self._live_control.take()))
        if requested:
            self._validate_now = False
        return bool(requested), pending

    def _apply_live_tunables(self, pending: dict[str, Any]) -> None:
        """Apply the tunables rank 0 polled; rank 0 also records them into the run's config snapshot."""
        applied = self._apply_tunables(pending)
        if self.global_rank == 0 and applied:
            self._interventions.extend(applied)
            self._record_interventions()

    def _apply_tunables(self, pending: dict[str, Any]) -> list[dict[str, Any]]:
        """Apply the pending tunables to this rank's live state; return one audit entry per change."""
        applied: list[dict[str, Any]] = []
        if "lr" in pending:
            new_lr = float(pending["lr"])
            applied.append({"it": self.it, "key": "lr", "from": self._current_lr(), "to": new_lr})
            self.model.module.rebase_lr(new_lr)
        if "it_validation" in pending:
            old = self.it_validation
            self.it_validation = max(1, int(pending["it_validation"]))
            self._declare_measure_window()
            applied.append({"it": self.it, "key": "it_validation", "from": old, "to": self.it_validation})
        return applied

    def _current_lr(self) -> float | None:
        """The learning rate of the first network that owns an optimizer, for the audit trail."""
        for network in self.model.module.get_networks().values():
            if network.optimizer is not None:
                return float(network.optimizer.param_groups[0]["lr"])
        return None

    def _record_interventions(self) -> None:
        """Append the intervention audit trail and the current it_validation to the config snapshot. Rank 0."""
        _record_interventions(self._config_snapshot, self._interventions, self.it_validation)

    def checkpoint_save(self, loss: float | None, crash: bool = False) -> None:
        """Save model and optimizer states, keeping all checkpoints or only the best one.

        The training thread copies the states into host memory; serialisation, publish and BEST
        pruning run on the writer's thread, joined before the next save and at exit.

        Args:
            loss (float): Current loss used for best checkpoint selection.
            crash (bool): A save on an exceptional exit, named ``crash_<date>.pt`` and left outside
                BEST retention.
        """
        if self.global_rank != 0:
            return
        self._checkpoint_writer.join()  # the previous file is on disk before this one's name is chosen

        path = checkpoints_directory() / self.train_name
        path.mkdir(parents=True, exist_ok=True)

        date = f"crash_{current_date()}" if crash else current_date()
        save_path = path / f"{date}.pt"
        collision = 1
        while save_path.exists():
            save_path = path / f"{date}_{collision}.pt"
            collision += 1
        self._saved_at_it = self.it

        # An unscored checkpoint carries the worst possible score so BEST mode retires it.
        checkpoint_loss = loss if loss is not None else self.early_stopping.worst_score
        save_dict: dict[str, Any] = {
            "epoch": self.epoch,
            "it": self.it,
            "loss": checkpoint_loss,
            "Model": self.model.module.network_states(),
            "resume": self._resume_cursor,
        }

        if self.model_ema is not None:
            save_dict["Model_EMA"] = _ema_network(self.model_ema).network_states()
            save_dict["Model_EMA_n_averaged"] = int(self.model_ema.n_averaged)

        save_dict.update(
            {
                f"{name}_optimizer_state_dict": network.optimizer.state_dict()
                for name, network in self.model.module.get_networks().items()
                if network.optimizer is not None
            }
        )
        save_dict.update(
            {
                f"{name}_it": network._it
                for name, network in self.model.module.get_networks().items()
                if network.optimizer is not None
            }
        )
        save_dict.update(
            {
                f"{name}_nb_lr_update": network._nb_lr_update
                for name, network in self.model.module.get_networks().items()
                if network.optimizer is not None
            }
        )
        save_dict.update(
            {
                f"{name}_schedulers_state_dict": network.schedule_states()
                for name, network in self.model.module.get_networks().items()
                if network.optimizer is not None
            }
        )

        snapshot = _on_host(save_dict)

        def publish() -> None:
            # Staged and renamed: a kill mid-write must not leave a truncated .pt under a plausible name.
            staging = save_path.with_name(f"{save_path.name}.{os.getpid()}.tmp")
            torch.save(snapshot, staging)
            os.replace(staging, save_path)
            if not crash and snapshot["resume"]["kind"] == "epoch_boundary":
                # An explicit latest continuation, sharing storage with its dated file.
                latest = path / "resume_latest.pt"
                # A reused name may be a stale hard link; truncating it would also truncate its old BEST inode.
                with tempfile.TemporaryDirectory(prefix=".resume-", dir=path) as temporary:
                    resume_staging = Path(temporary) / "checkpoint.pt"
                    try:
                        os.link(save_path, resume_staging)
                    except OSError:
                        shutil.copyfile(save_path, resume_staging)
                    os.replace(resume_staging, latest)
            if self.save_checkpoint_mode == "BEST" and not crash:
                self._update_best_checkpoint(save_path, checkpoint_loss)

        self._checkpoint_writer.submit(publish)

    @torch.no_grad()
    def _log(
        self,
        type_log: str,
        batch_sample: BatchSample,
    ) -> dict[str, float]:
        """Log losses, metrics and optionally images to TensorBoard.

        Args:
            type_log (str): "Training" or "Validation".
            batch_sample (BatchSample): The current batch, one item per destination group.

        Returns:
            dict[str, float]: Aggregated losses and metrics on rank 0; empty on the other ranks.
        """
        models: dict[str, Network] = {"": self.model.module}
        if self.model_ema is not None:
            models["_EMA"] = _ema_network(self.model_ema)

        measures = DistributedObject.get_measure(
            self.world_size,
            self.global_rank,
            self._collective_device,
            models,
            (
                self.it_validation
                if type_log == "Training" or self.dataloader_validation is None
                else len(self.dataloader_validation)
            ),
        )
        # get_measure gathers across ranks, so every rank calls it; only rank 0 writes and reports.
        if self.global_rank != 0:
            return {}

        images_log = []
        if self.data_log and not isinstance(self.tb, NullSummaryWriter):
            for name, data_type in self.data_log.items():
                if name in batch_sample:
                    data_type[0](
                        self.tb,
                        f"{type_log}/{name}",
                        batch_sample[name].tensor[: self.data_log[name][1]],
                        self.it,
                    )
                else:
                    images_log.append(name.replace(":", "."))

        for label, model in models.items():
            for name, network in model.get_networks().items():
                # EMA has no training forward: its first window may be empty, and the collector omits it.
                if network.measure is None or f"{name}{label}" not in measures:
                    continue
                # Losses and metrics take the same pair of boards: the value, and the weight that scaled it.
                for board, table in (("Loss", 0), ("Metric", 1)):
                    entries = measures[f"{name}{label}"][table]
                    self.tb.add_scalars(
                        f"{type_log}/{name}/{board}/{label}",
                        {k.replace(":", "."): v[1] for k, v in entries.items()},
                        self.it,
                    )
                    self.tb.add_scalars(
                        f"{type_log}/{name}/{board}_weight/{label}",
                        {k.replace(":", "."): v[0] for k, v in entries.items()},
                        self.it,
                    )

            if len(images_log):
                # get_layers is model-scoped: run it once per model, not once per network.
                # A visualization must not train BatchNorm or consume the next training draw.
                with preserved_rng(), _evaluating(model):
                    for name, layer, _ in model.get_layers(
                        [v.tensor for v in batch_sample.values() if v.is_input],
                        images_log,
                    ):
                        self.data_log[name][0](
                            self.tb,
                            f"{type_log}/{name}{label}",
                            layer[: self.data_log[name][1]],
                            self.it,
                        )

        if type_log == "Training":
            for name, network in self.model.module.get_networks().items():
                if network.optimizer is not None:
                    self.tb.add_scalar(
                        f"{type_log}/{name}/Learning Rate",
                        network.optimizer.param_groups[0]["lr"],
                        self.it,
                    )

        loss = {}
        minimized: dict[str, float] = {}
        for name, network in self.model.module.get_networks().items():
            if network.measure is not None and name in measures:
                minimized.update({k: v[2] for k, v in measures[name][0].items()})
                loss.update({k: v[1] for k, v in measures[name][0].items()})
                loss.update({k: v[1] for k, v in measures[name][1].items()})
        # The default selection scores the losses by what they minimized, not by what they report.
        self._loss_score = minimized
        return loss

    @torch.no_grad()
    def _train_log(self, batch_sample: BatchSample) -> dict[str, float]:
        """Wrapper for _log during training."""
        return self._log("Training", batch_sample)

    @torch.no_grad()
    def _validation_log(self, batch_sample: BatchSample) -> dict[str, float]:
        """Wrapper for _log during validation."""
        return self._log("Validation", batch_sample)


def _agreed_patch(gathered: list, template: list[int]) -> list[int] | None:
    """The per-axis MIN of the candidates gathered at the OOM shrink rendezvous, ``None`` when no rank
    proposed one. A gathered entry that is not a patch candidate is an asymmetric OOM (another rank's
    collective crossed this rendezvous): unrecoverable, failed as a diagnosis."""
    proposals = [proposal for proposal in gathered if proposal is not None]
    if not proposals:
        return None
    if any(
        not isinstance(proposal, list)
        or len(proposal) != len(template)
        or not all(isinstance(size, int) for size in proposal)
        for proposal in proposals
    ):
        raise TrainerError(
            "The OOM shrink rendezvous gathered data that is not a patch candidate:",
            f"gathered: {gathered}",
            "Another rank was still training, so its collective crossed this rendezvous.",
            "An asymmetric OOM is not recoverable; rerun with a smaller patch or fewer ranks.",
        )
    return [min(sizes) for sizes in zip(*proposals, strict=True)]


@config()
class Trainer(vram.VramAutoPatchMixin, DistributedObject):
    """Public API for training a model: setup, checkpointing, resuming, logging, and the distributed
    ``_Trainer`` launch.

    Args:
        model (ModelLoader): Loader for model architecture.
        dataset (DataTrain): Training/validation dataset.
        train_name (str): Training session name.
        manual_seed (int | None): Random seed.
        epochs (int): Number of epochs to run.
        it_validation (int | None): Validation interval.
        it_lr_update (int | None): Learning rate update interval.
        autocast (bool): Enable AMP training.
        channels_last (bool): Lay the convolution weights and inputs out channels-last; cuDNN then picks other
            kernels, faster or slower depending on the model.
        cudnn_benchmark (bool): Let cuDNN benchmark its kernels under ``manual_seed``: faster, no bit-for-bit replay.
        torch_compile (bool): Compile the graph walk with torch.compile; the first steps pay the compilation.
        gradient_checkpoints (list[str] | None): Modules to use gradient checkpointing on.
        gpu_checkpoints (list[str] | None): Modules to pin on specific GPUs.
        ema_decay (float): EMA decay factor.
        data_log (list[str] | None): Logging instructions.
        early_stopping (EarlyStopping | None): Optional early stopping config.
        save_checkpoint_mode (str): Either "BEST" or "ALL".
    """

    def __init__(
        self,
        model: ModelLoader = ModelLoader(),
        dataset: DataTrain = DataTrain(),
        train_name: str = "default|TRAIN_01",
        manual_seed: int | None = None,
        epochs: int = 100,
        it_validation: int | None = None,
        it_lr_update: int | None = None,
        autocast: bool = False,
        channels_last: bool = False,
        cudnn_benchmark: bool = False,
        torch_compile: bool = False,
        gradient_checkpoints: list[str] | None = None,
        gpu_checkpoints: list[str] | None = None,
        ema_decay: float = 0,
        data_log: list[str] | None = None,
        early_stopping: EarlyStopping | None = None,
        save_checkpoint_mode: str = "BEST",
    ) -> None:
        if os.environ["KONFAI_CONFIG_MODE"] != "Done":
            raise ConfigError("Trainer requires KONFAI_CONFIG_MODE='Done' before initialization.")
        super().__init__(train_name)
        self.manual_seed = manual_seed
        # Without manual_seed, a seed is drawn (read back on RESUME) and every draw of the run comes
        # from it, as from a configured one: manual_seed set to the recorded Seed.txt replays the run.
        self.drawn_seed = self._resolve_seed(State[konfai_state()])
        self.dataset = dataset
        self.dataset.manual_seed = self.run_seed
        self._capture_vram_patch_template(dataset.patch)
        self.autocast = autocast
        self.channels_last = channels_last
        self.cudnn_benchmark = cudnn_benchmark
        self.torch_compile = torch_compile
        self.epochs = epochs
        self.epoch = 0
        self._resume_state: dict[str, Any] | None = None
        self.override_lr: float | None = None
        self.early_stopping = early_stopping
        self.it = 0
        self.it_validation = it_validation
        self.it_lr_update = it_lr_update
        # A weight the load(init=True) of a TRAIN does not redraw (an Embedding) keeps the draw made here.
        seed_all(self.drawn_seed)
        with startup_clock().phase("model"):
            self.model = model.get_model(train=True)
        self.ema_decay = ema_decay
        self.model_ema: AveragedModel | None = None
        self.data_log = data_log

        self.gradient_checkpoints = gradient_checkpoints
        self.gpu_checkpoints = gpu_checkpoints
        self.save_checkpoint_mode = save_checkpoint_mode
        self.config_path_src = config_file()
        self.config_namefile = statistics_directory() / self.name / self.config_path_src.name
        self.size = len(self.gpu_checkpoints) + 1 if self.gpu_checkpoints else 1

        state = State[konfai_state()]
        # The model's downsampling multiple is final before init(); each case's free axis rounds up to it.
        self.dataset.set_free_axis_multiple(self.model.downsampling_factor())
        # The split is drawn on the launcher before spawn, from the run's seed: an unseeded split would be
        # redrawn on RESUME and leak validation cases into training.
        seed_all(self.drawn_seed)
        self.dataset.prepare()
        self.model.bind(
            self.autocast, state, self.dataset.get_groups_dest(), self.gradient_checkpoints, self.gpu_checkpoints
        )
        # The per-axis multiple a free patch axis rounds up to, read off the model's downsampling graph.
        self._downsampling_factor = self.model.downsampling_factor()

    def _resolve_seed(self, state: State) -> int:
        """The seed every draw of the run comes from: the configured ``manual_seed``, else the seed the
        TRAIN run recorded (RESUME rebuilds the split the checkpoint trained on), else a fresh draw
        recorded by ``setup`` for the next RESUME and for a replay."""
        if self.manual_seed is not None:
            return self.manual_seed
        if state == State.RESUME:
            recorded = self._recorded_seed()
            if recorded is not None:
                return recorded
        return int.from_bytes(os.urandom(4), "little")

    def _recorded_seed(self) -> int | None:
        try:
            return int((statistics_directory() / self.name / "Seed.txt").read_text().strip())
        except (OSError, ValueError):
            return None

    def outputs(self) -> list[Path]:
        return [checkpoints_directory() / self.name, statistics_directory() / self.name]

    def setup(self, world_size: int):
        """Initialize the training environment: clear previous outputs unless resuming, build the model
        and EMA, load the checkpoint when resuming, prepare the dataloaders.

        Args:
            world_size (int): Total number of distributed processes.
        """
        state = State[konfai_state()]
        if state != State.RESUME and (checkpoints_directory() / self.name).exists():
            confirm_overwrite_or_raise(checkpoints_directory() / self.name, "model", TrainerError)
            checkpoints_path = checkpoints_directory() / self.name
            if checkpoints_path.is_dir():
                shutil.rmtree(checkpoints_path)
            elif checkpoints_path.exists():
                checkpoints_path.unlink()
            # The statistics directory holds the rank-0 log already open: clear around it, not rmtree.
            statistics_path = statistics_directory() / self.name
            if statistics_path.is_dir():
                clear_directory_except_logs(statistics_path)
            elif statistics_path.exists():
                statistics_path.unlink()

        state_dict = {}
        with startup_clock().phase("checkpoint"):
            if state != State.TRAIN:
                state_dict = self._load()
                # Refused here, on the launcher, before the run writes into its statistics directory.
                if self._resume_state is not None and self._resume_state["world_size"] != world_size // self.size:
                    raise TrainerError(
                        "RESUME requires the same number of training ranks as its epoch checkpoint: it was"
                        f" written by {self._resume_state['world_size']}, this run has {world_size // self.size}.",
                        f"Relaunch RESUME on {self._resume_state['world_size']} training rank(s).",
                    )
            self.model.load(state_dict, init=True, ema=False, override_lr=self.override_lr)
            if self.ema_decay > 0:
                self.model_ema = AveragedModel(self.model, **self._ema_update())
                _ema_network(self.model_ema).load(state_dict, init=False, ema=True)
                if "Model_EMA_n_averaged" in state_dict:
                    self.model_ema.n_averaged.fill_(cast(int, state_dict["Model_EMA_n_averaged"]))

        (statistics_directory() / self.name).mkdir(exist_ok=True)
        # The snapshot traces the run's live changes, which a RESUME continues: the trace outlives the copy.
        traced = _traced_interventions(self.config_namefile) if state == State.RESUME else []
        shutil.copyfile(self.config_path_src, self.config_namefile)
        if traced:
            _record_interventions(self.config_namefile, traced)

        self.dataloader, train_names, validation_names = self.dataset.get_data(world_size // self.size)
        # A checkpoint of weights alone (no cursor, iteration 0) starts a new training: no split to keep.
        if state == State.RESUME and (self._resume_state is not None or self.it > 0):
            self._report_split_drift(train_names, validation_names)
        for split, names in (("Train", train_names), ("Validation", validation_names)):
            # Written as a subset or validation list reads it back.
            path = statistics_directory() / self.name / f"{split}_{self.it}.txt"
            path.write_text("".join(f"{name}\n" for name in names), encoding=case_list_encoding())
        # The run's seed, where _resolve_seed reads it on RESUME; written after the clearing above.
        (statistics_directory() / self.name / "Seed.txt").write_text(f"{self.drawn_seed}\n")

    def _report_split_drift(self, train_names: list[str], validation_names: list[str]) -> None:
        """Warn when the split RESUME redrew is not the one the run recorded last.

        The split is redrawn from the seed on the cases found today: a case added, removed or renamed
        since moves others between training and validation, and validation may then score cases the
        checkpoint trained on. The run goes on with the redrawn split.
        """
        directory = statistics_directory() / self.name
        recorded = [
            int(it)
            for it in (path.stem.removeprefix("Train_") for path in directory.glob("Train_*.txt"))
            if it.isdigit() and (directory / f"Validation_{it}.txt").is_file()
        ]
        if not recorded:
            return
        it = max(recorded)
        encoding = case_list_encoding()
        trained = set((directory / f"Train_{it}.txt").read_text(encoding=encoding).split("\n")) - {""}
        validated = set((directory / f"Validation_{it}.txt").read_text(encoding=encoding).split("\n")) - {""}
        training, validation = set(train_names), set(validation_names)
        changes = {
            "Trained before, validated now": trained & validation,
            "Validated before, trained now": validated & training,
            "New": (training | validation) - trained - validated,
            "Gone": (trained | validated) - training - validation,
        }
        lines = [f"{label} ({len(names)}): {_listed(names)}." for label, names in changes.items() if names]
        if not lines:
            return
        message = TrainerError(
            "RESUME redrew the train/validation split on the cases found today, and it is not the one"
            f" recorded in Train_{it}.txt and Validation_{it}.txt ({directory}):",
            *lines,
            "The run goes on with the redrawn split. To change a cohort on purpose, give 'validation' by"
            " case names (a list or a case-list file): a named case keeps its side.",
        )
        warnings.warn(str(message), KonfAIWarning, stacklevel=2)

    def set_model(self, path_to_model: str | Path) -> None:
        self.path_to_model = str(path_to_model)

    def set_lr(self, lr: float | None) -> None:
        self.override_lr = lr

    def _load(self) -> dict[str, Any]:
        """Load a previously saved checkpoint from local disk or URL.

        Returns:
            dict: State dictionary loaded from checkpoint.
        """
        state_dict = safe_torch_load(checkpoint_source(self.path_to_model, TrainerError), torch.device("cpu"))

        self._resume_state = None
        if "resume" in state_dict:
            cursor = state_dict["resume"]
            if not isinstance(cursor, dict) or cursor.get("version") != 1:
                raise TrainerError("Unsupported checkpoint resume cursor version.")
            if cursor.get("kind") != "epoch_boundary":
                raise TrainerError(
                    "This checkpoint cannot continue training: " + str(cursor.get("reason", "no epoch boundary")),
                    "Select a completed-epoch checkpoint with no pending gradients. "
                    "This checkpoint's model weights remain usable for PREDICTION.",
                )
            next_epoch = cursor.get("next_epoch")
            if type(next_epoch) is not int or next_epoch < 0:
                raise TrainerError("Invalid next_epoch in checkpoint resume cursor.")
            if not isinstance(cursor.get("rng_by_rank"), list) or len(cursor["rng_by_rank"]) != cursor.get(
                "world_size"
            ):
                raise TrainerError("Invalid rank generator states in checkpoint resume cursor.")
            self.epoch = next_epoch
            self._resume_state = cursor
        elif "epoch" in state_dict:
            # An integer "epoch" is the epoch being executed, never next_epoch.
            self.epoch = state_dict["epoch"]
        if "it" in state_dict:
            self.it = state_dict["it"]
        return state_dict

    def _ema_update(self) -> dict[str, Any]:
        """The EMA rule for AveragedModel: torch's fused ``multi_avg_fn``."""
        return {"multi_avg_fn": get_ema_multi_avg_fn(self.ema_decay)}

    def run_process(
        self,
        world_size: int,
        global_rank: int,
        local_rank: int,
        dataloaders: list[DataLoader],
    ):
        """Launch the training via ``_Trainer``: wrap the model with DDP or the CPU fallback, attach EMA.

        Args:
            world_size (int): Number of model replicas sharding the data: the spawned process count
                already divided by the model-parallel size (``gpu_checkpoints``), NOT the GPU count.
            global_rank (int): Global rank of the current process.
            local_rank (int): Local rank within the node.
            dataloaders (list[DataLoader]): Training and validation dataloaders.
        """
        model = place_graph(self.model, local_rank * self.size) if len(cuda_visible_devices()) else self.model
        if self.channels_last:
            Network.set_channels_last(model)
        if self.torch_compile:
            eager = self.model.compile_walk()
            if eager is not None and global_rank == 0:
                print(f"[KonfAI] torch_compile is set, but the graph walk stays eager: {eager}.", flush=True)
        if dist.is_initialized():
            model = DDP(model, **_ddp_kwargs(model, local_rank, self.size))
        else:
            model = Model(model)
        if self.model_ema is not None:
            self.model_ema.module = place_graph(_ema_network(self.model_ema), local_rank * self.size)
            if self.channels_last:
                Network.set_channels_last(_ema_network(self.model_ema))
        device = local_rank * self.size if len(cuda_visible_devices()) else None
        if self._presize_free_axes():
            dataloaders = self.dataset.get_data(world_size)[0][global_rank]
        while True:
            try:
                with _Trainer(
                    world_size,
                    global_rank,
                    local_rank,
                    self.size,
                    self.name,
                    self.early_stopping,
                    self.data_log,
                    self.save_checkpoint_mode,
                    self.epochs,
                    self.epoch,
                    self.autocast,
                    self.it_validation,
                    self.it_lr_update,
                    self.it,
                    cast(Model, model),  # DDP stands in for Model: the same module/train/eval/call surface
                    self.model_ema,
                    self.config_namefile,
                    dataloaders[0],
                    dataloaders[1] if len(dataloaders) > 1 else None,
                    auto_patched=self._vram_patch_template is not None,
                    resume_state=self._resume_state,
                ) as t:
                    t.run()
                return
            except torch.cuda.OutOfMemoryError:
                if self._vram_patch_template is None:
                    raise  # no free axis declared: not auto-patched
                # The step that just OOMed measured its transient. Drop its gradients before reading free
                # VRAM. The OOM fires on the first batch's forward, before any optimizer.step(); a mid-step
                # OOM leaves a one-batch partial update in place, which the restart continues from.
                measured = vram.transient_at_oom(device)
                self.model.zero_grad(set_to_none=True)
                candidate = self._shrunken_patch(measured, vram.usable_after_oom(device))
                # Every rank must train the same grid: each failing rank proposes a candidate and all
                # adopt the per-axis MIN. A rank that did not run out never reaches this all-gather and
                # the job dies at the collective timeout; an offset mismatch pairs foreign payloads.
                if world_size > 1:
                    print(
                        f"[KonfAI] VRAM: rank {global_rank} ran out of memory -> waiting at the shrink"
                        " rendezvous (a rank that did NOT run out aborts the job at the collective timeout)."
                    )
                agreed = _agreed_patch(
                    synchronize_data(world_size, local_rank * self.size, candidate), self._vram_patch_template
                )
                if agreed is None:
                    raise
                print(
                    f"[KonfAI] VRAM: rank {global_rank} ran out of memory -> "
                    f"re-planning the free patch axes to {agreed} and restarting the training run."
                )
                self._adopt_patch_candidate(agreed)
                vram.reset_peak(device)
                dataloaders = self.dataset.get_data(world_size)[0][global_rank]


def build_train(
    command: State = State.TRAIN,
    model: Path | str | None = None,
    config: Path | str | dict = Path("./Config.yml"),
    checkpoints_dir: Path | str = Path("./Checkpoints/"),
    statistics_dir: Path | str = Path("./Statistics/"),
    lr: float | None = None,
) -> DistributedObject:
    """
    Build and return the configured training workflow without executing it.

    Parameters
    ----------
    command : State, optional
        ``State.TRAIN`` or ``State.RESUME``.
    model : Path | str | None, optional
        Checkpoint path used when resuming training.
    config : Path | str | dict, optional
        The training configuration: its file, or the config tree itself (``{"Trainer": {...}}``).
    checkpoints_dir : Path | str, optional
        Output directory for checkpoints.
    statistics_dir : Path | str, optional
        Output directory for statistics and logs.
    lr : float | None, optional
        Learning-rate override when resuming: ``None`` resumes the checkpoint's rate and scheduler,
        a value restarts from it.

    Returns
    -------
    DistributedObject
        Configured trainer, executed by the runtime wrapper.
    """
    if command == State.RESUME and model is None:
        raise TrainerError(
            "RESUME continues from a checkpoint, and none was given.",
            "Pass model= the checkpoint to resume from (the CLI's --model), such as"
            " Checkpoints/<train_name>/resume_latest.pt.",
        )
    configure_workflow_environment(
        config_path=config,
        root="Trainer",
        state=command,
        path_env={
            "KONFAI_CHECKPOINTS_DIRECTORY": checkpoints_dir,
            "KONFAI_STATISTICS_DIRECTORY": statistics_dir,
        },
    )
    os.environ["KONFAI_CONFIG_MODE"] = "Done"
    # A warning, not a refusal: files written back by earlier versions carry keys nothing reads now.
    with strict_config("Trainer", refuse=False):
        trainer = apply_config()(Trainer)()
    if model is not None:
        # Keep https:// checkpoint URLs as raw strings: Path() collapses the '//'.
        trainer.set_model(model if isinstance(model, str) and model.startswith("https://") else Path(model))
    trainer.set_lr(lr)
    return trainer


@run_distributed_app
def train(
    command: State = State.TRAIN,
    overwrite: bool = False,
    model: Path | str | None = None,
    gpu: list[int] | None = None,
    cpu: int | None = None,
    quiet: bool = False,
    tensorboard: bool = False,
    config: Path | str | dict = Path("./Config.yml"),
    checkpoints_dir: Path | str = Path("./Checkpoints/"),
    statistics_dir: Path | str = Path("./Statistics/"),
    lr: float | None = None,
) -> DistributedObject:
    """Build and execute the configured training workflow.

    ``overwrite``/``gpu``/``cpu``/``quiet``/``tensorboard`` are read by :func:`run_distributed_app`
    from the bound signature; the body drops them. The pure build step is :func:`build_train`.
    """
    del overwrite, gpu, cpu, quiet, tensorboard
    return build_train(
        command=command,
        model=model,
        config=config,
        checkpoints_dir=checkpoints_dir,
        statistics_dir=statistics_dir,
        lr=lr,
    )

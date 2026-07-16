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
import shutil
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel
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
from konfai.data.data_manager import BatchSample, DataTrain
from konfai.network.network import Model, ModelLoader, NetState, Network
from konfai.utils.config import apply_config, config
from konfai.utils.errors import ConfigError, TrainerError
from konfai.utils.runtime import (
    DataLog,
    DistributedObject,
    State,
    configure_workflow_environment,
    confirm_overwrite_or_raise,
    description,
    run_distributed_app,
    safe_torch_load,
    synchronize_data,
)


class EarlyStoppingBase:
    """Minimal protocol for early stopping strategies used by :class:`Trainer`."""

    # Single source of truth for the optimisation direction of the monitored score. The default
    # (no configured EarlyStopping) monitors the summed loss, so lower is better; EarlyStopping
    # overrides this from its `mode` config. Both early stopping and BEST-checkpoint retention read it.
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
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "min":
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


class _Trainer:
    """
    Internal class for managing the training loop in a distributed or standalone setting.

    Handles:
    - Epoch iteration with training and optional validation
    - Mixed precision support (autocast)
    - Exponential Moving Average (EMA) model tracking
    - Early stopping
    - Logging to TensorBoard
    - Model checkpoint saving and selection (ALL or BEST)

    This class is intended to be used via a context manager
    (`with _Trainer(...) as trainer:`)  inside the public `Trainer` class.
    """

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
        model_ema: AveragedModel,
        dataloader_training: DataLoader,
        dataloader_validation: DataLoader | None = None,
    ) -> None:
        self.world_size = world_size
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.size = size
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

        self.it_validation = len(dataloader_training) if it_validation is None else it_validation
        self.it_lr_update = len(dataloader_training) if it_lr_update is None else it_lr_update
        self.it = it
        if SummaryWriter is None:
            raise ImportError(
                "TensorBoard is required for training logging. Install it with: pip install konfai[tensorboard]"
            )
        self.tb = SummaryWriter(log_dir=statistics_directory() / self.train_name / "tb")
        self._best_checkpoint_path: Path | None = None
        self._best_checkpoint_loss: float | None = None
        self._loss_keys: set[str] = set()
        if self.global_rank == 0 and self.save_checkpoint_mode == "BEST":
            self._initialize_best_checkpoint_state()
        self.data_log: dict[str, tuple[DataLog, int]] = {}
        if data_log is not None:
            for data in data_log:
                self.data_log[data.split("/")[0].replace(":", ".")] = (
                    DataLog[data.split("/")[1]],
                    int(data.split("/")[2]),
                )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        """Closes the SummaryWriter if used."""
        if self.tb is not None:
            self.tb.close()
        self.checkpoint_save(None)

    def _initialize_best_checkpoint_state(self) -> None:
        """Bootstrap BEST-checkpoint tracking once, including resume scenarios."""
        path = checkpoints_directory() / self.train_name
        if not path.exists():
            return

        all_checkpoints = sorted(path.glob("*.pt"))
        best_loss = self.early_stopping.worst_score
        best_ckpt: Path | None = None
        for checkpoint_path in all_checkpoints:
            state_dict = safe_torch_load(checkpoint_path, torch.device("cpu"))
            checkpoint_loss = float(state_dict.get("loss", self.early_stopping.worst_score))
            if not math.isfinite(checkpoint_loss):
                checkpoint_loss = self.early_stopping.worst_score
            if self.early_stopping.is_better(checkpoint_loss, best_loss):
                best_loss = checkpoint_loss
                best_ckpt = checkpoint_path

        if best_ckpt is not None:
            self._best_checkpoint_path = best_ckpt
            self._best_checkpoint_loss = best_loss
            for checkpoint_path in all_checkpoints:
                if checkpoint_path != best_ckpt:
                    checkpoint_path.unlink()

    def _update_best_checkpoint(self, checkpoint_path: Path, loss: float) -> None:
        """Keep only the current best checkpoint without rescanning all saves."""
        is_new_best = self._best_checkpoint_loss is None or self.early_stopping.is_better(
            loss, self._best_checkpoint_loss
        )
        if is_new_best:
            previous_best = self._best_checkpoint_path
            self._best_checkpoint_loss = loss
            self._best_checkpoint_path = checkpoint_path
            if previous_best is not None and previous_best != checkpoint_path and previous_best.exists():
                previous_best.unlink()
            return

        checkpoint_path.unlink()

    def run(self) -> None:
        """
        Launches the training loop, performing one epoch at a time.
        Triggers early stopping and resets data augmentations between epochs.
        """
        self.dataloader_training.dataset.load("Train")
        if self.dataloader_validation is not None:
            self.dataloader_validation.dataset.load("Validation")
            if State[konfai_state()] != State.TRAIN:
                self._validate()

        with tqdm.tqdm(
            iterable=range(self.epoch, self.epochs),
            leave=False,
            total=self.epochs,
            initial=self.epoch,
            desc="Progress",
        ) as epoch_tqdm:
            for self.epoch in epoch_tqdm:
                self.train()
                if self.early_stopping.is_stopped():
                    break
                self.dataloader_training.dataset.reset_augmentation("Train")

    def train(self) -> None:
        """
        Performs a full training epoch with support for:
        - mixed precision
        - DDP / CPU training
        - EMA updates
        - loss logging and checkpoint saving
        - validation at configurable iteration interval
        """
        self.model.train()
        self.model.module.set_state(NetState.TRAIN)
        if self.model_ema is not None:
            self.model_ema.eval()
            self.model_ema.module.set_state(NetState.TRAIN)

        with tqdm.tqdm(
            iterable=enumerate(self.dataloader_training),
            desc=f"Training : {description(self.model, self.model_ema)}",
            total=len(self.dataloader_training),
            leave=False,
            ncols=0,
        ) as batch_iter:
            for _, batch_sample in batch_iter:
                with torch.amp.autocast("cuda", enabled=self.autocast):
                    self.model(batch_sample)
                    self.model.module.backward(self.model)
                    if self.model_ema is not None:
                        self.model_ema.update_parameters(self.model)
                    self.it += 1

                    if (self.it) % self.it_lr_update == 0:
                        self.model.module.update_lr()

                    if (self.it) % self.it_validation == 0:
                        loss = self._train_log(batch_sample)

                        if self.dataloader_validation is not None:
                            loss = self._validate()

                        stop = False
                        if self.global_rank == 0:
                            # Default selection scores on the losses only (always lower-is-better).
                            # An explicit monitor may reference a metric; then use the full dict, and
                            # its direction comes from EarlyStopping.mode (is_better).
                            if isinstance(self.early_stopping, EarlyStopping) and self.early_stopping.monitor:
                                score = self.early_stopping.get_score(loss)
                            else:
                                score = self.early_stopping.get_score(
                                    {key: loss[key] for key in self._loss_keys if key in loss}
                                )
                            self.checkpoint_save(score)
                            stop = self.early_stopping(score)

                            # Stop once the schedulers have decayed the learning rate to zero:
                            # no further optimisation is possible, so end the run cleanly.
                            optimizer = self.model.module.optimizer
                            if not stop and optimizer is not None and optimizer.param_groups[0]["lr"] <= 0:
                                self.early_stopping.stop()
                                stop = True

                        if self._broadcast_stop(stop):
                            self.early_stopping.stop()
                            break

                batch_iter.set_description(f"Training : {description(self.model, self.model_ema)}")

    @torch.no_grad()
    def _validate(self) -> float:
        """
        Executes the validation phase, evaluates loss and metrics.
        Updates model states and resets augmentation for validation set.

        Returns:
            float: Validation loss.
        """
        if self.dataloader_validation is None:
            return 0
        self.model.eval()
        self.model.module.set_state(NetState.PREDICTION)
        if self.model_ema is not None:
            self.model_ema.module.set_state(NetState.PREDICTION)

        batch_sample: BatchSample = {}
        with tqdm.tqdm(
            iterable=enumerate(self.dataloader_validation),
            desc=f"Validation : {description(self.model, self.model_ema)}",
            total=len(self.dataloader_validation),
            leave=False,
            ncols=0,
        ) as batch_iter:
            for _, batch_sample in batch_iter:
                self.model(batch_sample)
                if self.model_ema is not None:
                    self.model_ema.module(batch_sample)

                batch_iter.set_description(f"Validation : {description(self.model, self.model_ema)}")
        self.dataloader_validation.dataset.reset_augmentation("Validation")
        if dist.is_initialized():
            dist.barrier()
        self.model.train()
        self.model.module.set_state(NetState.TRAIN)
        if self.model_ema is not None:
            self.model_ema.module.set_state(NetState.TRAIN)
        return self._validation_log(batch_sample)

    def _broadcast_stop(self, stop: bool) -> bool:
        """
        Share rank 0's stop decision with every rank so the training loop is left together.

        Only rank 0 owns the aggregated metrics and therefore the early-stopping decision.
        Broadcasting it prevents ranks from diverging (some breaking, some continuing).
        """
        outputs = synchronize_data(
            self.world_size,
            self.local_rank * self.size + self.size - 1,
            stop,
        )
        return bool(outputs[0])

    def checkpoint_save(self, loss: float | None) -> None:
        """
        Saves model and optimizer states. Keeps either all checkpoints or only the best one.

        Args:
            loss (float): Current loss used for best checkpoint selection.
        """
        if self.global_rank != 0:
            return

        path = checkpoints_directory() / self.train_name
        path.mkdir(parents=True, exist_ok=True)

        date = current_date()
        save_path = path / f"{date}.pt"
        collision = 1
        while save_path.exists():
            save_path = path / f"{date}_{collision}.pt"
            collision += 1

        # An unscored checkpoint (the final save at close) carries the worst possible score so
        # `_update_best_checkpoint` retires it in BEST mode instead of leaving it beside the real best.
        checkpoint_loss = loss if loss is not None else self.early_stopping.worst_score
        save_dict = {
            "epoch": self.epoch,
            "it": self.it,
            "loss": checkpoint_loss,
            "Model": self.model.module.state_dict(),
        }

        if self.model_ema is not None:
            save_dict["Model_EMA"] = self.model_ema.module.state_dict()
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

        torch.save(save_dict, save_path)

        if self.save_checkpoint_mode == "BEST":
            self._update_best_checkpoint(save_path, checkpoint_loss)

    @torch.no_grad()
    def _log(
        self,
        type_log: str,
        batch_sample: BatchSample,
    ) -> dict[str, float] | None:
        """
        Logs losses, metrics and optionally images to TensorBoard.

        Args:
            type_log (str): "Training" or "Validation".
            batch_item (dict): Dictionary of BatchItem from current batch.

        Returns:
            dict[str, float] | None: Dictionary of aggregated losses and metrics if rank == 0.
        """
        models: dict[str, Network] = {"": self.model.module}
        if self.model_ema is not None:
            models["_EMA"] = self.model_ema.module

        measures = DistributedObject.get_measure(
            self.world_size,
            self.global_rank,
            self.local_rank * self.size + self.size - 1,
            models,
            (
                self.it_validation
                if type_log == "Training" or self.dataloader_validation is None
                else len(self.dataloader_validation)
            ),
        )

        if self.global_rank == 0:
            images_log = []
            if len(self.data_log):
                for name, data_type in self.data_log.items():
                    if name in batch_sample:
                        data_type[0](
                            self.tb,
                            f"{type_log}/{name}",
                            batch_sample[name].tensor[: self.data_log[name][1]].detach().cpu().numpy(),
                            self.it,
                        )
                    else:
                        images_log.append(name.replace(":", "."))

            for label, model in models.items():
                for name, network in model.get_networks().items():
                    if network.measure is not None:
                        self.tb.add_scalars(
                            f"{type_log}/{name}/Loss/{label}",
                            {k.replace(":", "."): v[1] for k, v in measures[f"{name}{label}"][0].items()},
                            self.it,
                        )
                        self.tb.add_scalars(
                            f"{type_log}/{name}/Loss_weight/{label}",
                            {k.replace(":", "."): v[0] for k, v in measures[f"{name}{label}"][0].items()},
                            self.it,
                        )

                        self.tb.add_scalars(
                            f"{type_log}/{name}/Metric/{label}",
                            {k.replace(":", "."): v[1] for k, v in measures[f"{name}{label}"][1].items()},
                            self.it,
                        )
                        self.tb.add_scalars(
                            f"{type_log}/{name}/Metric_weight/{label}",
                            {k.replace(":", "."): v[0] for k, v in measures[f"{name}{label}"][1].items()},
                            self.it,
                        )

                    if len(images_log):
                        for name, layer, _ in model.get_layers(
                            [v.tensor for v in batch_sample.values() if v.is_input],
                            images_log,
                        ):
                            self.data_log[name][0](
                                self.tb,
                                f"{type_log}/{name}{label}",
                                layer[: self.data_log[name][1]].detach().cpu().numpy(),
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

        if self.global_rank == 0:
            loss = {}
            loss_keys: set[str] = set()
            for name, network in self.model.module.get_networks().items():
                if network.measure is not None:
                    losses = {k: v[1] for k, v in measures[name][0].items()}
                    loss_keys.update(losses)
                    loss.update(losses)
                    loss.update({k: v[1] for k, v in measures[name][1].items()})
            # Remember which keys are losses (always minimise) vs metrics (direction varies), so
            # default checkpoint/early-stop selection scores on the losses only -- summing a
            # maximise-metric (e.g. Dice) into a minimised score would keep the worst model.
            self._loss_keys = loss_keys
            return loss
        return None

    @torch.no_grad()
    def _train_log(self, batch_sample: BatchSample) -> dict[str, float]:
        """Wrapper for _log during training."""
        return self._log("Training", batch_sample)

    @torch.no_grad()
    def _validation_log(self, batch_sample: BatchSample) -> dict[str, float]:
        """Wrapper for _log during validation."""
        return self._log("Validation", batch_sample)


@config()
class Trainer(DistributedObject):
    """
    Public API for training a model using the KonfAI framework.
    Wraps setup, checkpointing, resuming, logging, and launching distributed _Trainer.

    Main responsibilities:
    - Initialization from config (via @config)
    - Model and EMA setup
    - Checkpoint loading and saving
    - Distributed setup and launch

    Args:
        model (ModelLoader): Loader for model architecture.
        dataset (DataTrain): Training/validation dataset.
        train_name (str): Training session name.
        manual_seed (int | None): Random seed.
        epochs (int): Number of epochs to run.
        it_validation (int | None): Validation interval.
        it_lr_update (int | None): Learning rate update interval.
        autocast (bool): Enable AMP training.
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
        self.dataset = dataset
        self.autocast = autocast
        self.epochs = epochs
        self.epoch = 0
        self.override_lr: float | None = None
        self.early_stopping = early_stopping
        self.it = 0
        self.it_validation = it_validation
        self.it_lr_update = it_lr_update
        self.model = model.get_model(train=True)
        self.ema_decay = ema_decay
        self.model_ema: torch.optim.swa_utils.AveragedModel | None = None
        self.data_log = data_log

        self.gradient_checkpoints = gradient_checkpoints
        self.gpu_checkpoints = gpu_checkpoints
        self.save_checkpoint_mode = save_checkpoint_mode
        self.config_path_src = config_file()
        config_namefile = self.config_path_src.name.replace(".yml", "")
        self.config_namefile = statistics_directory() / self.name / f"{config_namefile}_{self.it}.yml"
        self.size = len(self.gpu_checkpoints) + 1 if self.gpu_checkpoints else 1

        state = State[konfai_state()]
        self.dataset.prepare()
        self.model.init(self.autocast, state, self.dataset.get_groups_dest())
        self.model.init_outputs_group()
        self.model._compute_channels_trace(
            self.model,
            self.model.in_channels,
            self.gradient_checkpoints,
            self.gpu_checkpoints,
        )

    def setup(self, world_size: int):
        """
        Initializes the training environment:
        - Clears previous outputs (unless resuming)
        - Initializes model and EMA
        - Loads checkpoint (if resuming)
        - Prepares dataloaders

        Args:
            world_size (int): Total number of distributed processes.
        """
        state = State[konfai_state()]
        if state != State.RESUME and (checkpoints_directory() / self.name).exists():
            confirm_overwrite_or_raise(checkpoints_directory() / self.name, "model", TrainerError)
            for directory_path in [
                statistics_directory(),
                checkpoints_directory(),
            ]:
                path = directory_path / self.name
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()

        state_dict = {}
        if state != State.TRAIN:
            state_dict = self._load()
        self.model.load(state_dict, init=True, ema=False, override_lr=self.override_lr)
        if self.ema_decay > 0:
            self.model_ema = AveragedModel(self.model, avg_fn=self._avg_fn)
            if state_dict is not None:
                self.model_ema.module.load(state_dict, init=False, ema=True)
                if "Model_EMA_n_averaged" in state_dict:
                    self.model_ema.n_averaged.fill_(cast(int, state_dict["Model_EMA_n_averaged"]))

        (statistics_directory() / self.name).mkdir(exist_ok=True)
        shutil.copyfile(self.config_path_src, self.config_namefile)

        self.dataloader, train_names, validation_names = self.dataset.get_data(world_size // self.size)
        with open(statistics_directory() / self.name / f"Train_{self.it}.txt", "w") as f:
            for name in train_names:
                f.write(name + "\n")
        with open(statistics_directory() / self.name / f"Validation_{self.it}.txt", "w") as f:
            for name in validation_names:
                f.write(name + "\n")

    def set_model(self, path_to_model: str | Path) -> None:
        self.path_to_model = str(path_to_model)

    def set_lr(self, lr: float | None) -> None:
        self.override_lr = lr

    def __exit__(self, exc_type, value, traceback):
        """Exit training context and trigger save of model/checkpoints."""
        super().__exit__(exc_type, value, traceback)
        self._save()

    def _load(self) -> dict[str, dict[str, torch.Tensor]]:
        """
        Loads a previously saved checkpoint from local disk or URL.

        Returns:
            dict: State dictionary loaded from checkpoint.
        """
        if self.path_to_model.startswith("https://") or Path(self.path_to_model).exists():
            state_dict = safe_torch_load(self.path_to_model, torch.device("cpu"))
        else:
            raise ValueError(f"Invalid model path entry: {self.path_to_model}")

        if "epoch" in state_dict:
            self.epoch = state_dict["epoch"]
        if "it" in state_dict:
            self.it = state_dict["it"]
        return state_dict

    def _save(self) -> None:
        if self.config_namefile.exists():
            new_name = f"{self.config_namefile.stem}_{self.it}.yml"
            os.rename(
                self.config_namefile,
                self.config_namefile.parent / new_name,
            )

    def _avg_fn(self, averaged_model_parameter: float, model_parameter, num_averaged):
        """EMA update (AveragedModel avg_fn): ``ema_decay * averaged + (1 - ema_decay) * model``."""
        return self.ema_decay * averaged_model_parameter + (1 - self.ema_decay) * model_parameter

    def run_process(
        self,
        world_size: int,
        global_rank: int,
        local_rank: int,
        dataloaders: list[DataLoader],
    ):
        """
        Launches the actual training process via internal `_Trainer` class.
        Wraps model with DDP or CPU fallback, attaches EMA, and starts training.

        Args:
            world_size (int): Total number of distributed processes.
            global_rank (int): Global rank of the current process.
            local_rank (int): Local rank within the node.
            dataloaders (list[DataLoader]): Training and validation dataloaders.
        """
        model = Network.to(self.model, local_rank * self.size) if len(cuda_visible_devices()) else self.model
        if dist.is_initialized():
            ddp_kwargs: dict[str, object] = {"static_graph": True}
            if len(cuda_visible_devices()) and self.size == 1:
                ddp_kwargs.update({"device_ids": [local_rank], "output_device": local_rank})
            model = DDP(model, **ddp_kwargs)
        else:
            model = Model(model)
        if self.model_ema is not None:
            self.model_ema.module = Network.to(self.model_ema.module, local_rank * self.size)
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
            model,
            self.model_ema,
            *dataloaders,
        ) as t:
            t.run()


def build_train(
    command: State = State.TRAIN,
    model: Path | str | None = None,
    config: Path | str = Path("./Config.yml"),
    checkpoints_dir: Path | str = Path("./Checkpoints/"),
    statistics_dir: Path | str = Path("./Statistics/"),
    lr: float | None = None,
) -> DistributedObject:
    """
    Build and return the configured training workflow without executing it.

    Parameters
    ----------
    command : State, optional
        Training command variant, typically ``State.TRAIN`` or ``State.RESUME``.
    model : Path | str | None, optional
        Checkpoint path used when resuming training.
    config : Path | str, optional
        Training configuration file.
    checkpoints_dir : Path | str, optional
        Output directory for checkpoints.
    statistics_dir : Path | str, optional
        Output directory for statistics and logs.
    lr : float | None, optional
        Runtime learning-rate override applied when resuming/fine-tuning. When
        ``None`` the checkpoint learning rate is resumed and the scheduler
        continues; when set, the learning rate restarts from this value.

    Returns
    -------
    DistributedObject
        Configured trainer object ready to be executed by the runtime wrapper.
    """
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
    trainer = apply_config()(Trainer)()
    if model is not None:
        # Keep https:// checkpoint URLs as raw strings: Path() collapses the '//' into
        # 'https:/…', which then fails both the startswith('https://') check and Path.exists().
        trainer.set_model(model if isinstance(model, str) and model.startswith("https://") else Path(model))
    trainer.set_lr(lr)
    return trainer


@run_distributed_app
def train(
    command: State = State.TRAIN,
    overwrite: bool = False,
    model: Path | str | None = None,
    gpu: list[int] | None = cuda_visible_devices(),
    cpu: int | None = None,
    quiet: bool = False,
    tensorboard: bool = False,
    config: Path | str = Path("./Config.yml"),
    checkpoints_dir: Path | str = Path("./Checkpoints/"),
    statistics_dir: Path | str = Path("./Statistics/"),
    lr: float | None = None,
) -> DistributedObject:
    """
    Build and execute the configured training workflow.

    This compatibility wrapper preserves the historical CLI-facing API while
    delegating the pure build step to :func:`build_train`.
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


if __name__ == "__main__":
    train(State.TRAIN, False, None)

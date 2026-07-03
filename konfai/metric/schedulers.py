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

"""Schedulers used to modulate metric and loss weights during training."""

from abc import abstractmethod
from functools import partial

import numpy as np
import torch


class Scheduler:
    """Base class for scalar schedulers used by KonfAI criteria."""

    def __init__(self, start_value: float) -> None:
        self.baseValue = float(start_value)
        self.it = 0

    def step(self, it: int):
        self.it = it

    @abstractmethod
    def get_value(self) -> float:
        pass


class Constant(Scheduler):
    """Scheduler returning a constant value for all iterations."""

    def __init__(self, value: float = 1):
        super().__init__(value)

    def get_value(self) -> float:
        return self.baseValue


class CosineAnnealing(Scheduler):
    """Cosine annealing scheduler for criterion weights."""

    def __init__(self, start_value: float = 1, eta_min: float = 0.00001, t_max: int = 100):
        super().__init__(start_value)
        self.eta_min = eta_min
        self.t_max = t_max

    def get_value(self):
        return self.eta_min + (self.baseValue - self.eta_min) * (1 + np.cos(self.it * torch.pi / self.t_max)) / 2


class Warmup(torch.optim.lr_scheduler.LambdaLR):
    """Learning-rate warmup wrapper compatible with PyTorch optimizers."""

    @staticmethod
    def warmup(warmup_steps: int, step: int) -> float:
        return min(1.0, (step + 1) / (warmup_steps + 1))

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 10,
        last_epoch=-1,
        verbose="deprecated",
    ):
        super().__init__(optimizer, partial(Warmup.warmup, warmup_steps), last_epoch, verbose)


class PolyLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        max_steps: int,
        exponent: float = 0.9,
        current_step: int | None = None,
    ):
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0 if current_step is None else current_step

        for param_group in optimizer.param_groups:
            param_group["lr"] = initial_lr
            param_group.setdefault("initial_lr", initial_lr)

        super().__init__(optimizer, last_epoch=-1)

    def step(self, current_step=None):
        if current_step is None:
            if self.last_epoch != -1:
                current_step = self.last_epoch
                self.last_epoch += 1
            else:
                current_step = self.ctr
                self.ctr += 1

        current_step = min(current_step, self.max_steps)

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr

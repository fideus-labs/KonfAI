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


"""What a step is: the network state, patch-indexed batches, the batched step."""

import inspect
from collections.abc import Callable
from enum import Enum
from typing import Any

import torch

from konfai import cuda_visible_devices
from konfai.data.patching import ModelPatch


class NetState(Enum):
    """Execution state of a network inside KonfAI workflows."""

    TRAIN = (0,)
    PREDICTION = 1


class PatchIndexed:
    """Track progress while consuming the patches produced by a :class:`ModelPatch`."""

    def __init__(self, patch: ModelPatch, index: int) -> None:
        self.patch = patch
        self.index = index

    def is_full(self) -> bool:
        return len(self.patch.get_patch_slices(0)) == self.index


def batched_step(
    optimizer_class: type[torch.optim.Optimizer], parameters: list[torch.nn.parameter.Parameter]
) -> Callable[..., torch.optim.Optimizer]:
    """``optimizer_class``, its step batched over ``parameters`` when the config leaves the choice open.

    ``fused`` and ``foreach`` both unset ask for the widest batched step the optimizer implements: fused
    when it has one, foreach when it only has that. The device is the one the run will place the graph on,
    not the one the parameters sit on here, because an optimizer is built on the launcher and ``Network.to``
    moves the graph afterwards. Off CUDA torch's own default stands (forcing foreach there costs 526 ms a
    step against 2.0 ms, over the 40 tensors of the Segmentation example); an explicit value in the config
    is passed through either way.

    One RTX PRO 5000, those 40 tensors / 1.93 M parameters: an AdamW step costs 0.059 ms of host time fused
    against 0.188 ms foreach, in two kernels against eight, and a fused step is the only one that applies the
    AMP scale itself, so ``GradScaler.step`` stops reading ``found_inf`` back to the host once a step. Fused
    sums in another order: after 100 steps its parameters sit 3.7e-05 from the foreach ones, 8.3e-06 of their
    own scale.
    """
    signature = inspect.signature(optimizer_class)
    on_cuda = len(cuda_visible_devices()) > 0

    def build(params: Any, **kwargs: Any) -> torch.optim.Optimizer:
        batchable = (
            on_cuda
            and not kwargs.get("differentiable", False)
            and all(parameter.is_floating_point() for parameter in parameters)
        )
        if batchable and kwargs.get("fused") is None and kwargs.get("foreach") is None:
            for name in ("fused", "foreach"):
                if name in signature.parameters:
                    kwargs[name] = True
                    break
        return optimizer_class(params, **kwargs)

    # The binder reads this signature: the optimizer's own keys, so the config keeps exactly them.
    build.__signature__ = signature  # type: ignore[attr-defined]
    return build

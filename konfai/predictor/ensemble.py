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


"""The ensemble of checkpoints one prediction runs."""

import copy
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import torch

from konfai.data.reduction import Mean, Reduction
from konfai.network.network import Network
from konfai.utils.errors import PredictorError
from konfai.utils.runtime import (
    safe_torch_load,
)


def _colocate_loaded_modules(model: torch.nn.Module) -> None:
    """Move any still-CPU leaf module onto the model's device.

    A custom :meth:`Network.load` may append modules after the model was already placed on its
    device (e.g. a head sized from the checkpoint's class count), and those default to CPU, which
    then raises a device mismatch on the forward pass. This re-homes any fully-CPU leaf onto the
    device the rest of the model already lives on. Modules already on a device (including
    model-parallel splits across several GPUs) are left untouched.
    """
    target = next((p.device for p in model.parameters() if p.device.type != "cpu"), None)
    if target is None:
        return
    for sub in model.modules():
        own = [*sub.parameters(recurse=False), *sub.buffers(recurse=False)]
        if own and all(t.device.type == "cpu" for t in own):
            sub.to(target)


class ModelComposite(Network):
    """
    A composite model that replicates a given base network multiple times and combines their outputs.

    This class is designed to handle model ensembles or repeated predictions from the same architecture.
    It creates `nb_models` deep copies of the input `model`, each with its own name and output branch,
    and aggregates their outputs using a provided `Reduction` strategy (e.g., mean, median).

    Args:
        model (Network): The base network to replicate.
        nb_models (int): Number of copies of the model to create.
        combine (konfai.data.reduction.Reduction): The reduction method used to combine outputs from
            all model replicas.

    Attributes:
        combine (konfai.data.reduction.Reduction): The reduction used during forward inference.
    """

    def __init__(self, model: Network, combine: Reduction):
        super().__init__(
            model.in_channels,
            model.optimizer,
            model.lr_schedulers_loader,
            model.outputs_criterions_loader,
            model.patch,
            model.nb_batch_per_step,
            model.init_type,
            model.init_gain,
            model.dim,
        )
        self.combine = combine
        self._model_name = "Model_0"
        self._base_model_name = model.get_name()
        self._state_sources: list[dict[str, Any] | Path | str] = []
        self._loaded = False  # load() has run: distinguishes "not loaded yet" from "loaded, weightless"
        self._loaded_state_index: int | None = None
        # Cache the CPU state_dict per index so a local-path ensemble is read from
        # disk once, not re-read + re-unpickled on every batch (the index cycles
        # 0..N-1 each forward, so the next batch would otherwise reload all N).
        self._state_cache: dict[int, dict[str, Any]] = {}
        self.add_module(
            self._model_name,
            copy.deepcopy(model),
            in_branch=[0],
            out_branch=["output_0"],
        )

    def _get_model(self) -> Network:
        return cast(Network, self[self._model_name])

    def _read_state_source(self, source: dict[str, Any] | Path | str) -> dict[str, Any]:
        if isinstance(source, dict):
            return source
        if isinstance(source, str) and source.startswith("https://"):
            return torch.hub.load_state_dict_from_url(url=source, map_location="cpu", check_hash=True)
        return safe_torch_load(source, torch.device("cpu"))

    def _ensure_model_loaded(self, index: int) -> Network:
        model = self._get_model()
        if self._loaded_state_index != index:
            state = self._state_cache.get(index)
            if state is None:
                state = self._read_state_source(self._state_sources[index])
                self._state_cache[index] = state
            # Checkpoints are keyed by the base model name, not by the streamed
            # ensemble suffix added after the previous load.
            model.set_name(self._base_model_name)
            model.load(state, init=False)
            # A custom load() may append checkpoint-sized modules (e.g. the head) on CPU; co-locate
            # them with the already device-placed model so the forward pass doesn't hit a mismatch.
            _colocate_loaded_modules(model)
            model.set_name(f"{self._base_model_name}_{index}")
            self._loaded_state_index = index
        return model

    def _model_for_index(self, index: int) -> Network:
        # With no checkpoint sources the model is weightless (0 parameters, e.g. a classical/optimisation
        # engine): run it as constructed, once. The Predictor guards this, it only reaches here with empty
        # sources when the model has no parameters to load, so there is nothing to stream.
        if not self._state_sources:
            return self._get_model()
        return self._ensure_model_loaded(index)

    def load(self, state_sources: list[dict[str, Any] | Path | str]):
        """
        Load weights for each sub-model in the composite from the corresponding state dictionaries.

        Args:
            state_sources (list): One checkpoint source per model replica. Empty ONLY for a weightless model
                (0 parameters), which is then run once with its constructed weights; empty sources for a model
                that has trainable parameters is refused here, so a caller cannot silently run random weights.
        """
        if not state_sources and any(parameter.numel() for parameter in self._get_model().parameters()):
            raise PredictorError(
                "ModelComposite.load() received no checkpoint sources for a model with trainable parameters.",
                "A weightless model (0 parameters) may run with no checkpoint; a parameterised one may not.",
                "Pass at least one checkpoint source, or wrap a model that has no parameters.",
            )
        self._state_sources = state_sources
        self._loaded = True
        self._loaded_state_index = None
        self._state_cache = {}
        if len(self._state_sources) == 1:
            self._ensure_model_loaded(0)

    @torch.inference_mode()
    def forward(  # type: ignore[override]
        self,
        data_dict: dict[tuple[str, bool], torch.Tensor],
        output_layers: list[str] = [],
    ) -> list[tuple[str, list[int], torch.Tensor]]:
        """
        Perform a forward pass on all model replicas and aggregate their outputs.

        Args:
            data_dict (dict): A dictionary mapping (group_name, requires_grad) to input tensors.
            output_layers (list): List of output layer names to extract from each sub-model.

        Returns:
            list[tuple[str, torch.Tensor]]: Aggregated output for each layer, after applying the reduction.
        """
        final_outputs: list[tuple[str, list[int], torch.Tensor]] = []
        if not self._loaded:
            raise PredictorError(
                "ModelComposite.forward() called before load().",
                "Prediction ran before the composite's checkpoint sources were set.",
                "Call load(...) first (load([]) for a weightless model).",
            )
        # A weightless model (loaded with no checkpoint sources) is a single replica: the model as constructed.
        n_replicas = len(self._state_sources) or 1
        if isinstance(self.combine, Mean):
            sum_acc: dict[str, torch.Tensor] = {}
            count: dict[str, int] = defaultdict(int)
            channels: dict[str, list[int]] = defaultdict(list)
            for model_index in range(n_replicas):
                for key, tensor in self._model_for_index(model_index)(data_dict, output_layers):
                    if tensor.dtype == torch.float32:
                        tensor = tensor.to(torch.float16)
                    channels[key].append(tensor.shape[1])
                    if key not in sum_acc:
                        sum_acc[key] = tensor
                    else:
                        sum_acc[key].add_(tensor)
                    count[key] += 1
            for key, acc in sum_acc.items():
                # The sum was folded in place into the first model's output; a lone model's is the
                # answer as it stands. Dividing by one copied the batch output (56 MiB per
                # [1, 14, 128^3] fp16 patch, 512 MiB at 122 channels), on every single-model run.
                final_outputs.append((key, channels[key], acc if count[key] == 1 else acc.div_(count[key])))
        else:
            aggregated = defaultdict(list)
            for model_index in range(n_replicas):
                for key, tensor in self._model_for_index(model_index)(data_dict, output_layers):
                    if tensor.dtype == torch.float32:
                        tensor = tensor.to(torch.float16)
                    aggregated[key].append(tensor)

            for key, tensors in aggregated.items():
                # Mean, Median -> [N, C, ...] | Concat -> [N, C*M, ...]
                final_outputs.append((key, [t.shape[1] for t in tensors], self.combine(tensors)))

        return final_outputs

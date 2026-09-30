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
import math
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, cast

import torch

from konfai.data.reduction import Mean, Reduction
from konfai.network.network import Network
from konfai.utils import vram
from konfai.utils.errors import PredictorError
from konfai.utils.runtime import (
    safe_torch_load,
)


def _colocate_loaded_modules(model: torch.nn.Module) -> None:
    """Move any still-CPU leaf module onto the model's device.

    A custom :meth:`Network.load` may append modules after the model was placed, and those default to
    CPU. Modules already on a device, model-parallel splits included, are left untouched.
    """
    target = next((p.device for p in model.parameters() if p.device.type != "cpu"), None)
    if target is None:
        return
    for sub in model.modules():
        own = [*sub.parameters(recurse=False), *sub.buffers(recurse=False)]
        if own and all(t.device.type == "cpu" for t in own):
            sub.to(target)


def _stock_loader(model: Network) -> bool:
    """Whether ``model`` loads through ``Network.load``: a class that overrides ``load`` owns its format."""
    from konfai.network.network.network import MinimalModel

    return getattr(type(model), "load", None) in (Network.load, MinimalModel.load)


def _member_weights(model: Network) -> list[torch.Tensor]:
    """The tensors a stock load writes: every parameter and persistent buffer of the graph."""
    return [
        tensor
        for module in model.modules()
        for name, tensor in (*module._parameters.items(), *module._buffers.items())
        if tensor is not None and name not in module._non_persistent_buffers_set
    ]


def _require_weights_entry(model: Network, state: dict[str, Any], source: dict[str, Any] | Path | str) -> None:
    """Refuse a checkpoint the stock loader would take no weights from.

    ``Network.load`` reads weights from a KonfAI checkpoint's ``Model`` entry and is silent without
    one. A model whose class overrides ``load`` owns its format, and a weightless model loads nothing.
    """
    if not _stock_loader(model) or not any(True for _ in model.parameters()):
        return
    if "Model" in state:
        return
    keys = sorted(str(key) for key in state)[:12]
    raise PredictorError(
        f"Checkpoint '{source}' holds no 'Model' entry (top-level keys: {keys}): not a KonfAI checkpoint, "
        f"so '{model.get_name()}' would predict with its constructor weights.",
        "Give a checkpoint a KonfAI TRAIN wrote, or a model class whose 'load' reads this format. "
        "To predict from EMA weights alone, export their network entries under 'Model'.",
    )


def _inference_entries(model: Network, state: dict[str, Any]) -> dict[str, Any]:
    """What the host cache keeps of a checkpoint: the weights inference reads, dropping the optimizer
    state a training checkpoint carries beside them. A model whose class owns its ``load`` keeps the
    file whole."""
    if not _stock_loader(model):
        return state
    # The composite calls load(..., ema=False): even when present, Model_EMA is never read.
    return {key: value for key, value in state.items() if key == "Model"}


def _checkpoint_bytes(value: Any) -> int | None:
    """Conservative retained size, counting shared tensor storage once within an entry and Python
    containers too. Unknown custom objects answer ``None`` and bypass the cache. Separate entries may
    overcount shared allocations, never undercount."""
    seen: set[int] = set()
    storages: set[tuple[str, int, int]] = set()

    def visit(item: Any) -> int:
        if id(item) in seen:
            return 0
        seen.add(id(item))
        # Only the exact types below have a complete accounting contract; a subclass can hide allocations.
        if type(item) not in (
            dict,
            OrderedDict,
            list,
            tuple,
            set,
            frozenset,
            bool,
            int,
            float,
            complex,
            str,
            bytes,
            bytearray,
            type(None),
            torch.dtype,
            torch.device,
            torch.Tensor,
            torch.nn.Parameter,
        ):
            raise TypeError(type(item).__name__)
        size = sys.getsizeof(item)
        if isinstance(item, torch.Tensor):
            if item.layout != torch.strided or item.grad_fn is not None:
                raise TypeError("non-strided storage or retained autograd graph")
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr(), storage.nbytes())
            if key not in storages:
                storages.add(key)
                size += storage.nbytes()
            return size + visit(item.__dict__) + visit(item.grad) + visit(item._base)
        if isinstance(item, dict):
            size += sum(visit(key) + visit(child) for key, child in item.items())
            if hasattr(item, "__dict__"):
                size += visit(item.__dict__)
            return size
        if isinstance(item, (list, tuple, set, frozenset)):
            return size + sum(visit(child) for child in item)
        if item is None or isinstance(
            item, (bool, int, float, complex, str, bytes, bytearray, torch.dtype, torch.device)
        ):
            return size
        raise TypeError(type(item).__name__)

    try:
        return visit(value)
    except (TypeError, RuntimeError):
        return None


class ModelComposite(Network):
    """
    One reusable model streams ensemble checkpoints and combines their outputs.

    Once :meth:`keep_resident` has run, each member keeps weights of its own on the model's devices and a
    forward points the model at them, instead of loading the member's checkpoint into it.

    Args:
        model (Network): The base network to replicate.
        combine (konfai.data.reduction.Reduction): The reduction combining the replicas' outputs.
        checkpoint_cache_gib (float): Per-process retained checkpoint budget. Reloadable members
            that do not fit bypass the cache. Dictionary sources must fit in this budget.
    """

    def __init__(self, model: Network, combine: Reduction, checkpoint_cache_gib: float = 1.0):
        if not math.isfinite(checkpoint_cache_gib * 1024**3) or checkpoint_cache_gib < 0:
            raise PredictorError("checkpoint_cache_gib must be finite and non-negative.")
        super().__init__(
            model.in_channels,
            model.optimizerLoader,
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
        self._cache_limit_bytes = int(checkpoint_cache_gib * 1024**3)
        self._state_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._state_cache_bytes = 0
        self._cache_entry_bytes: dict[int, int] = {}
        self._state_stamps: dict[int, tuple[int, int, int, int, int] | None] = {}
        #: Each resident member's weights, in ``_weights`` order; ``None`` while members load per forward.
        self._resident: dict[int, list[torch.Tensor]] | None = None
        #: The model's tensors a member's weights take the place of.
        self._weights: list[torch.Tensor] = []
        #: What the other members' weights hold on each device while they are resident.
        self._resident_bytes: dict[torch.device, int] = {}
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
            return torch.hub.load_state_dict_from_url(
                url=source, map_location="cpu", check_hash=True, weights_only=True
            )
        return safe_torch_load(source, torch.device("cpu"), mmap=True)

    @staticmethod
    def _source_stamp(source: dict[str, Any] | Path | str) -> tuple[int, int, int, int, int] | None:
        if isinstance(source, dict) or (isinstance(source, str) and source.startswith("https://")):
            return None
        stat = Path(source).stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def _drop_cached_state(self, index: int) -> None:
        self._state_cache.pop(index, None)
        self._state_cache_bytes -= self._cache_entry_bytes.pop(index, 0)

    def _cache_state(self, index: int, state: dict[str, Any], *, replace: bool) -> None:
        if isinstance(self._state_sources[index], dict):
            # Already charged as a resident source; this second reference owns no new payload.
            self._state_cache[index] = state
            return
        size = _checkpoint_bytes(state)
        if size is None or size > self._cache_limit_bytes:
            return
        # New members only take free room, keeping the resident subset hot across a cyclic ensemble.
        # A cached file that changed size may evict the least recently used reloadable entries.
        if replace:
            for victim in list(self._state_cache):
                if self._state_cache_bytes + size <= self._cache_limit_bytes:
                    break
                if victim in self._cache_entry_bytes:
                    self._drop_cached_state(victim)
        if self._state_cache_bytes + size <= self._cache_limit_bytes:
            self._state_cache[index] = state
            self._cache_entry_bytes[index] = size
            self._state_cache_bytes += size

    def _ensure_model_loaded(self, index: int) -> Network:
        model = self._get_model()
        source = self._state_sources[index]
        stamp = self._source_stamp(source)
        changed = index in self._state_stamps and self._state_stamps[index] != stamp
        replace = changed and index in self._state_cache
        if changed:
            self._drop_cached_state(index)
            if self._resident is not None:
                self._resident.pop(index, None)
            if self._loaded_state_index == index:
                self._loaded_state_index = None
        if index in self._state_cache:
            self._state_cache.move_to_end(index)
        resident = self._resident.get(index) if self._resident is not None else None
        if self._loaded_state_index != index and resident is not None:
            for tensor, weights in zip(self._weights, resident, strict=True):
                tensor.data = weights
        elif self._loaded_state_index != index:
            state = self._state_cache.get(index)
            if state is None:
                state = self._read_state_source(source)
                if self._source_stamp(source) != stamp:
                    raise PredictorError(f"Checkpoint '{source}' changed while being read; retry prediction.")
                # Checkpoints are keyed by the base model name, not by the streamed ensemble suffix.
                model.set_name(self._base_model_name)
                _require_weights_entry(model, state, source)
                state = _inference_entries(model, state)
                self._cache_state(index, state, replace=replace)
            self._state_stamps[index] = stamp
            if self._resident:
                # The model holds a resident member's weights: this member loads into a copy of its own.
                with torch.inference_mode(False):
                    for tensor in self._weights:
                        tensor.data = tensor.data.clone()
            model.set_name(self._base_model_name)
            try:
                model.load(state, init=False)
            except RuntimeError as error:
                # The stock loader's strict load: a checkpoint of another architecture.
                if not _stock_loader(model) or isinstance(error, torch.cuda.OutOfMemoryError):
                    raise
                named = "The in-memory checkpoint" if isinstance(source, dict) else f"Checkpoint '{source}'"
                raise PredictorError(
                    f"{named} does not fit the model this configuration builds.",
                    str(error).strip(),
                    f"Predictor.Model.{self._base_model_name} must declare the architecture the checkpoint"
                    " was trained with: the same parameters as in the training Config.yml.",
                ) from error
            # A custom load() may append modules on CPU: co-locate them with the placed model.
            _colocate_loaded_modules(model)
            if self._resident is not None:
                self._resident[index] = [tensor.data for tensor in self._weights]
        if self._loaded_state_index != index:
            model.set_name(f"{self._base_model_name}_{index}")
            self._loaded_state_index = index
        return model

    def keep_resident(self) -> bool:
        """Load every member of an ensemble into weights of its own on the model's devices, when the copies
        fit in what a measured batch leaves of each device's VRAM (:func:`vram.resident_room`): run before the
        batch is measured, which then counts them. ``False``, and members keep loading per forward, for a
        single model, a weightless one, one whose class owns its ``load`` or ``initialized`` (either may set
        more than the checkpoint's tensors), one with a module's ``extra_state`` (restored by the load, not a
        tensor kept here), or copies that do not fit.
        """
        model = self._get_model()
        if (
            len(self._state_sources) < 2
            or not _stock_loader(model)
            or any(
                type(network).initialized is not Network.initialized
                for network in model.modules()
                if isinstance(network, Network)
            )
            or any(type(module).get_extra_state is not torch.nn.Module.get_extra_state for module in model.modules())
        ):
            return False
        weights = _member_weights(model)
        need: dict[torch.device, int] = defaultdict(int)
        for tensor in weights:
            need[tensor.device] += tensor.nbytes * (len(self._state_sources) - 1)
        if any(size > vram.resident_room(device) for device, size in need.items()):
            return False
        self._weights = weights
        self._resident = {}
        self._resident_bytes = dict(need)
        self._loaded_state_index = None
        try:
            for index in range(len(self._state_sources)):
                self._ensure_model_loaded(index)
        except torch.cuda.OutOfMemoryError:
            # The room was measured, not reserved: another process can take it meanwhile.
            self.release_resident()
            self._loaded_state_index = None
            return False
        return True

    def release_resident(self) -> bool:
        """Drop the other members' weights: members load per forward again. Whether any were resident."""
        released = self._resident is not None
        self._resident = None
        self._weights = []
        self._resident_bytes = {}
        return released

    def resident_bytes(self, device: int) -> int:
        """What the other members' weights hold on CUDA ``device``: the room releasing them gives back."""
        return self._resident_bytes.get(torch.device("cuda", device), 0)

    def _model_for_index(self, index: int) -> Network:
        # No checkpoint source means a weightless model (0 parameters): run it as constructed, once.
        if not self._state_sources:
            return self._get_model()
        return self._ensure_model_loaded(index)

    def load(self, state_sources: list[dict[str, Any] | Path | str]):
        """
        Load weights for each sub-model in the composite from the corresponding state dictionaries.

        Args:
            state_sources (list): One checkpoint source per model replica. Empty ONLY for a weightless
                model (0 parameters), run once with its constructed weights; empty sources for a model
                with trainable parameters are refused here.
        """
        if not state_sources and any(parameter.numel() for parameter in self._get_model().parameters()):
            raise PredictorError(
                "ModelComposite.load() received no checkpoint sources for a model with trainable parameters.",
                "A weightless model (0 parameters) may run with no checkpoint; a parameterised one may not.",
                "Pass at least one checkpoint source, or wrap a model that has no parameters.",
            )
        # A dictionary cannot be reloaded after eviction: charge it against the same ceiling.
        sources: list[dict[str, Any] | Path | str] = []
        resident_bytes = 0
        model = self._get_model()
        for source in state_sources:
            if isinstance(source, dict):
                _require_weights_entry(model, source, "in-memory checkpoint")
                source = _inference_entries(model, source)
                size = _checkpoint_bytes(source)
                if size is None or resident_bytes + size > self._cache_limit_bytes:
                    raise PredictorError(
                        "In-memory checkpoint sources exceed checkpoint_cache_gib or contain objects whose "
                        "retained size cannot be measured.",
                        "Pass checkpoint file paths for bounded on-demand loading, or increase "
                        "Predictor.checkpoint_cache_gib for measurable dictionaries. Zero disables caching "
                        "of reloadable paths/URLs; dictionary sources still need room in the budget.",
                    )
                resident_bytes += size
            sources.append(source)
        self._state_sources = sources
        self._loaded = True
        self._loaded_state_index = None
        self._state_cache = OrderedDict()
        self._state_cache_bytes = resident_bytes
        self._cache_entry_bytes = {}
        self._state_stamps = {}
        self.release_resident()
        if len(self._state_sources) == 1:
            self._ensure_model_loaded(0)

    def _accumulated(self, tensor: torch.Tensor) -> torch.Tensor:
        """An output at the precision it is combined, blended and accumulated in.

        Half, from float32: what a class probability or an intensity survives, at half the memory. A model whose
        outputs are world coordinates keeps float32 by declaring ``full_precision_outputs`` (a registration's
        displacement field, 0.06 mm steps at 100 mm in float16). Float64 is never kept: it is read as float32 first,
        as the patch planning prices it.
        """
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        if tensor.dtype == torch.float32 and not getattr(self._get_model(), "full_precision_outputs", False):
            return tensor.to(torch.float16)
        return tensor

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
            list[tuple[str, list[int], torch.Tensor]]: per layer, its key, the channel counts of
            the models that produced it, and the aggregated tensor.
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
                    tensor = self._accumulated(tensor)
                    channels[key].append(tensor.shape[1])
                    if key not in sum_acc:
                        sum_acc[key] = tensor
                    elif not (sum_acc[key].is_floating_point() or sum_acc[key].is_complex()):
                        raise PredictorError(
                            f"combine: Mean cannot average '{key.replace('.', ':')}' over {n_replicas} checkpoints:"
                            f" it is a {sum_acc[key].dtype} label map (an Argmax head), and a mean of class indices"
                            " is another class.",
                            "Several --models run an ensemble; resume_latest.pt is a training continuation,"
                            " not a model to predict with.",
                            "To ensemble a segmentation, write the Softmax output and apply Argmax in"
                            " final_transforms (the members' probabilities are averaged, then the labels drawn),"
                            " or fold the label maps with combine: Vote.",
                        )
                    else:
                        sum_acc[key].add_(tensor)
                    count[key] += 1
            for key, acc in sum_acc.items():
                # The sum was folded in place into the first model's output; a lone model's is the
                # answer as it stands, and dividing it by one would copy the batch output.
                final_outputs.append((key, channels[key], acc if count[key] == 1 else acc.div_(count[key])))
        else:
            aggregated = defaultdict(list)
            for model_index in range(n_replicas):
                for key, tensor in self._model_for_index(model_index)(data_dict, output_layers):
                    aggregated[key].append(self._accumulated(tensor))

            for key, tensors in aggregated.items():
                # Mean, Median -> [N, C, ...] | Concat -> [N, C*M, ...]
                final_outputs.append((key, [t.shape[1] for t in tensors], self.combine(tensors)))

        return final_outputs

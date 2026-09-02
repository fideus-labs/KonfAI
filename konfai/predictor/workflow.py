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


"""The configured prediction workflow and its Python entrypoints."""

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from konfai import config_file, cuda_visible_devices, konfai_root, predictions_directory
from konfai.data.data_manager import (
    DataPrediction,
)
from konfai.data.reduction import Concat
from konfai.network.network import Model, ModelLoader, Network
from konfai.predictor.ensemble import ModelComposite
from konfai.predictor.loop import _Predictor
from konfai.predictor.output import OutputDatasetLoader
from konfai.utils import vram
from konfai.utils.budget import node_local_ranks, set_per_rank_budget
from konfai.utils.chain_diff import dataset_tree, input_chain_differences, training_dataset_tree
from konfai.utils.clock import startup_clock
from konfai.utils.config import apply_config, config, strict_config
from konfai.utils.errors import ConfigError, KonfAIError, PredictorError
from konfai.utils.ome_zarr import bound_chunk_cache
from konfai.utils.runtime import (
    DataLog,
    DistributedObject,
    State,
    configure_workflow_environment,
    run_distributed_app,
)
from konfai.utils.utils import concretize_patch_size, get_module


@config()
class Predictor(vram.VramAutoPatchMixin, DistributedObject):
    """
    KonfAI's main prediction controller.

    This class orchestrates the prediction phase by:
    - Loading model weights from checkpoint(s) or URL(s)
    - Preparing datasets and output configurations
    - Managing distributed inference with optional multi-GPU support
    - Applying transformations and saving predictions
    - Optionally logging results to TensorBoard

    Attributes:
        model (Network): The neural network model to use for prediction.
        dataset (konfai.data.data_manager.DataPrediction): Dataset manager for prediction data.
        combine_classpath (str): Path to the reduction strategy (e.g., "Mean").
        autocast (bool): Whether to enable AMP inference.
        outputs_dataset (dict[str, OutputDataset]): Mapping from layer names to output writers.
        data_log (list[str] | None): List of tensors to log during inference.
    """

    def __init__(
        self,
        model: ModelLoader = ModelLoader(),
        dataset: DataPrediction = DataPrediction(),
        combine: str = "Mean",
        train_name: str = "name",
        manual_seed: int | None = None,
        gpu_checkpoints: list[str] | None = None,
        autocast: bool = False,
        channels_last: bool = False,
        outputs_dataset: dict[str, OutputDatasetLoader] | None = {"default|Default": OutputDatasetLoader()},
        data_log: list[str] | None = None,
        check_training_transforms: bool = True,
    ) -> None:
        if os.environ["KONFAI_CONFIG_MODE"] != "Done":
            raise ConfigError("Predictor requires KONFAI_CONFIG_MODE='Done' before initialization.")
        super().__init__(train_name)
        self.manual_seed = manual_seed
        self.dataset = dataset
        self._capture_vram_patch_template(dataset.patch)
        #: Cases whose every configured output already existed when the run started: frozen at
        #: ``setup`` on the launcher, so every rank (restarts included) shards the same work list.
        self._done_case_indices: set[int] = set()
        module, name = get_module(combine, "konfai.predictor")
        if module.__name__ == "konfai.predictor":
            self.combine = getattr(module, name)()
        else:
            self.combine = apply_config(f"{konfai_root()}.{combine}")(getattr(module, name))()

        self.autocast = autocast
        self.channels_last = channels_last
        self.check_training_transforms = check_training_transforms
        with startup_clock().phase("model"):
            self.model = model.get_model(train=False)
        self.it = 0
        self.outputs_dataset_loader = outputs_dataset if outputs_dataset else {}
        self.outputs_dataset = {
            name.replace(":", "."): value.get_output_dataset(name)
            for name, value in self.outputs_dataset_loader.items()
        }

        self.datasets_filename = []
        self.predict_path = predictions_directory() / self.name
        per_rank_budget = self.dataset.resolved_budget().per_rank_bytes(node_local_ranks())
        set_per_rank_budget(per_rank_budget)
        bound_chunk_cache()
        for output_dataset in self.outputs_dataset.values():
            output_dataset.set_memory_budget(per_rank_budget)
            self.datasets_filename.append(output_dataset.filename)
            # Rebase under the run directory, re-deriving is_directory: a bare string + "/" would flag an
            # h5 output as a directory and write the hidden dotfile Predictions/<run>/Dataset/.h5.
            output_dataset.rebase(self.predict_path)
        self.data_log = data_log
        modules = [name for name, _ in self.model.named_modules()]
        for target in DataLog.parse(self.data_log):
            if target not in self.dataset.get_groups_dest() and target not in modules:
                raise PredictorError(
                    f"Invalid key '{target}' in `data_log`.",
                    f"This key is neither a destination group from the dataset ({self.dataset.get_groups_dest()})",
                    f"nor a valid module name in the model ({modules}).",
                    "Please check your `data_log` configuration,"
                    " it should reference either a model output or a dataset group.",
                )

        self.gpu_checkpoints = gpu_checkpoints
        # Cut the grids with the model's downsampling multiple already known, so each case's free axis
        # rounds up to a valid input size (the graph (hence the factor) is final before init()).
        self.dataset.set_free_axis_multiple(self.model.downsampling_factor())
        self.dataset.prepare()
        self.model.bind(
            self.autocast, State.PREDICTION, self.dataset.get_groups_dest(), gpu_checkpoints=self.gpu_checkpoints
        )
        # The per-axis multiple a free patch axis rounds up to, read off the model's downsampling graph.
        self._downsampling_factor = self.model.downsampling_factor()
        self.output_modules = [name for name, _, _ in self.model.named_module_args_dict()]

        for output_group in self.outputs_dataset.keys():
            if output_group.replace(";accu;", "") not in self.output_modules:
                raise PredictorError(
                    f"The output group '{output_group}' defined in 'outputs_criterions' "
                    "does not correspond to any module in the model.",
                    f"Available modules: {self.output_modules}",
                    "Please check that the name matches exactly a submodule or output of your model architecture.",
                )

        dataset_groups = {
            group_src: list(groups_dest.keys()) for group_src, groups_dest in self.dataset.groups_src.items()
        }

        for name, output_dataset in self.outputs_dataset.items():
            output_dataset.prepare(name.replace(".", ":"))
            output_dataset.setup(
                list(self.dataset.datasets.values()),
                dataset_groups,
            )

        if len(self.outputs_dataset) == 0 and not any(
            network.measure is not None for network in self.model.get_networks().values()
        ):
            raise PredictorError(
                "No prediction outputs or runtime measures are configured.",
                "Define at least one outputs_dataset entry or enable a network measure.",
            )

    def setup(self, world_size: int):
        """
        Set up the predictor for inference.

        This method performs all necessary initialization steps before running predictions:
        - Ensures output directories exist, and optionally prompts the user before overwriting existing predictions.
        - Copies the current configuration file (Prediction.yml) into the output directory for reproducibility.
        - Dynamically loads pretrained weights from local files or remote URLs.
        - Wraps the base model into a `ModelComposite` to support ensemble inference.
        - Initializes the prediction dataloader, with proper distribution across available GPUs.

        Args:
            world_size (int): Total number of processes or GPUs used for distributed prediction.

        """
        for dataset_filename in self.datasets_filename:
            path = self.predict_path / dataset_filename
            if not os.path.exists(path):
                os.makedirs(path)

        shutil.copyfile(config_file(), self.predict_path / "Prediction.yml")

        # Per-case resume, the semantics TRANSFORM documents: a case whose every configured output
        # is already on disk is skipped, so a rerun after a mid-cohort failure pays only the missing
        # cases; --overwrite recomputes everything. The set is frozen here, on the launcher, so
        # every rank (and every OOM-restart re-plan) shards the same reduced work list.
        if os.environ.get("KONFAI_OVERWRITE") != "True" and self.outputs_dataset:
            self._done_case_indices = {
                index
                for index, name in enumerate(self.dataset.case_names)
                if all(output.is_dataset_exist(output.group, name) for output in self.outputs_dataset.values())
            }
            if self._done_case_indices:
                print(
                    f"[KonfAI] prediction: {len(self._done_case_indices)}/{len(self.dataset.case_names)}"
                    " case(s) already written -> skipped (--overwrite recomputes)."
                )

        self.model_composite = ModelComposite(self.model, self.combine)
        if not self.path_to_models and any(parameter.numel() for parameter in self.model.parameters()):
            # A model WITH weights but no checkpoint would run with random weights and silently produce
            # garbage: refuse it. A WEIGHTLESS model (0 parameters, e.g. a classical/optimisation engine
            # such as registration) is legitimate with no checkpoint: it is run once as constructed.
            raise PredictorError(
                "No model checkpoint available for prediction.",
                "This model has trainable weights, so at least one '.pt' checkpoint must be provided (for "
                "KonfAI Apps, declare it via the 'models' field in app.json).",
                "Without a checkpoint its weights are random and prediction would silently produce garbage.",
            )
        with startup_clock().phase("checkpoint"):
            self.model_composite.load(self._load())
        try:
            self._report_chain_drift()
        except (OSError, KonfAIError, ValueError, TypeError) as error:
            # A diagnostic reading someone else's config file never fails the prediction it reports
            # on; what stopped it is said instead of swallowed.
            print(f"[KonfAI] the training-chain check did not run: {type(error).__name__}: {error}")

        self.size = len(self.gpu_checkpoints) + 1 if self.gpu_checkpoints else 1

        self._drop_done_cases()
        self.dataloader, _, _ = self.dataset.get_data(world_size // self.size)

    def _drop_done_cases(self) -> None:
        """Drop the already-written cases' entries from the prepared patch mapping.

        Applied to the mapping rather than the case list so the surviving cases keep their indices
        (the managers and the loader's remapping stay untouched), and re-applied after every
        ``replan_patch``, which rebuilds the mapping from scratch.
        """
        if not self._done_case_indices:
            return
        self.dataset._prepared_mapping = [
            entry for entry in self.dataset._prepared_mapping if entry[0] not in self._done_case_indices
        ]

    def _report_chain_drift(self) -> None:
        """Warn when the chain applied to a model input is not the one its checkpoint trained on.

        Same checkpoint, different preprocessing is silent: the run succeeds and only the values are
        wrong (the Synthesis example shipped ``Standardize(mask: None)`` in training against
        ``Standardize(mask: MASK)`` here, and paid 409 HU of MAE instead of 98). A legitimate
        difference exists, so this warns and never refuses, and ``check_training_transforms: false``
        silences it. Compared against the resolved config the training run left in its ``Statistics``
        directory, read without writing it back.
        """
        if not self.check_training_transforms:
            return
        applied = dataset_tree(config_file(), konfai_root())
        runs: list[tuple[Mapping[str, Any], list[str]]] = []
        unchecked: list[Path] = []
        for path_to_model in self.path_to_models:
            checkpoint = Path(path_to_model)
            trained = training_dataset_tree(checkpoint) if checkpoint.is_file() else None
            if trained is None:
                unchecked.append(checkpoint)
                continue
            # Folds of one experiment spell the same chains: compare each distinct one once.
            for known, names in runs:
                if known == trained:
                    names.append(checkpoint.parent.name)
                    break
            else:
                runs.append((trained, [checkpoint.parent.name]))
        if unchecked:
            print(
                f"[KonfAI] the training-chain check did not run for {len(unchecked)} checkpoint(s):"
                f" no resolved training config beside '{unchecked[0]}'"
                " (a TRAIN run keeps one in Statistics/<train_name>/)."
            )
        for trained, names in runs:
            differences = input_chain_differences(trained, applied)
            if not differences:
                continue
            print(f"[KonfAI] WARNING: this run preprocesses a model input differently from {', '.join(names)}:")
            for difference in differences:
                print(f"[KonfAI]   {difference}")
            print(
                "[KonfAI] The same checkpoint on differently preprocessed inputs predicts wrong values"
                " without failing. Set 'check_training_transforms: false' under Predictor once the"
                " difference is deliberate."
            )

    def set_models(self, path_to_models: list[Path | str]) -> None:
        self.path_to_models = path_to_models

    def _load(self) -> list[dict[str, Any] | Path | str]:
        """
        Resolve checkpoint sources for ensemble prediction.

        This method handles both remote and local model sources:
        - If the model path is a URL (starting with "https://"), it eagerly downloads and loads the state dict
          once because re-fetching it every batch would be prohibitively slow.
        - If the model path is local:
            - it keeps only the checkpoint path and lets `ModelComposite` stream weights into a single model
              instance during prediction to reduce memory pressure.

        Returns:
            list[dict[str, dict[str, torch.Tensor]] | Path | str]: A list of checkpoint sources, one per model.

        Raises:
            Exception: If a model path does not exist or cannot be loaded.
        """
        state_dicts = []
        for path_to_model in self.path_to_models:
            if isinstance(path_to_model, str) and path_to_model.startswith("https://"):
                try:
                    state_dicts.append(
                        torch.hub.load_state_dict_from_url(url=path_to_model, map_location="cpu", check_hash=True)
                    )
                except Exception as exc:
                    raise Exception(f"Model : {path_to_model} does not exist !") from exc
            elif Path(path_to_model).exists():
                state_dicts.append(Path(path_to_model))
            else:
                raise ValueError(f"Invalid model path entry: {path_to_model}")
        return state_dicts

    def run_process(
        self,
        world_size: int,
        global_rank: int,
        local_rank: int,
        dataloaders: list[DataLoader],
    ):
        """
        Launch prediction on the given process rank.

        Args:
            world_size (int): Number of model replicas sharding the data: the spawned process count
                already divided by the model-parallel size (``gpu_checkpoints``), NOT the GPU count.
            global_rank (int): Rank of the current process.
            local_rank (int): Local device rank.
            dataloaders (list[DataLoader]): List of data loaders for prediction.
        """

        model_composite = (
            Network.to(self.model_composite, local_rank * self.size)
            if len(cuda_visible_devices())
            else self.model_composite
        )
        if self.channels_last:
            Network.set_channels_last(model_composite)
        if len(cuda_visible_devices()):
            # Co-locate the output writers with the model so their reduction/transforms know the GPU.
            for output_dataset in self.outputs_dataset.values():
                output_dataset.to(local_rank * self.size)
        model_composite = Model(model_composite)
        device = local_rank * self.size if len(cuda_visible_devices()) else None
        dataloader = dataloaders[0]
        # A whole-axis extent still too large for VRAM OOMs into the shrink loop below, which keeps
        # the size valid too (the border padding fills the round-up, cropped back after the forward).
        if self._vram_patch_candidate is None and self._presize_free_axes():
            dataloader = self._rank_dataloader(world_size, global_rank)
        while True:
            try:
                with _Predictor(
                    world_size,
                    global_rank,
                    local_rank,
                    self.autocast,
                    self.predict_path,
                    self.data_log,
                    self.outputs_dataset,
                    model_composite,
                    dataloader,
                ) as p:
                    p.run()
                return
            except torch.cuda.OutOfMemoryError:
                # The restart loop IS the sizing iteration (no probe phase): the run that just OOMed
                # already measured the step's transient for free. Read it BEFORE the reset (the peak
                # still includes the resident accumulators on both sides of the difference), free the
                # in-flight state: open streamed sinks abort and remove their partial entries, so a
                # reader never sees a half-written volume even when the OOM is fatal, then read the
                # honest free VRAM.
                measured = vram.transient_at_oom(device)
                for output_dataset in self.outputs_dataset.values():
                    output_dataset.reset()
                if self._vram_patch_template is None:
                    raise  # no free axis declared: not auto-patched
                candidate = self._shrunken_patch(measured, vram.usable_after_oom(device))
                if candidate is None:
                    raise
                vram.reset_peak(device)
                print(
                    f"[KonfAI] VRAM: rank {global_rank} ran out of memory -> "
                    f"re-planning the free patch axes to {candidate} and restarting this rank's cases."
                )
                self._adopt_patch_candidate(candidate)
                dataloader = self._rank_dataloader(world_size, global_rank)

    def _rank_dataloader(self, world_size: int, global_rank: int) -> DataLoader:
        """This rank's loader over the re-planned grids, the already-written cases dropped again
        (a re-plan rebuilds the mapping from scratch)."""
        self._drop_done_cases()
        return self.dataset.get_data(world_size)[0][global_rank][0]

    def _shrunken_patch(self, measured: int | None, usable: float) -> list[int] | None:
        """The shared shrink step, with the blend kept on the GPU when it fits: the accumulation
        footprint is RESERVED beside the forward, so the sized patch passes the accumulation gate.
        Only when that reserve fits at no size (or cannot be priced) is the forward sized alone:
        the gate's memory-safe CPU blend absorbs that case.
        """
        if self._vram_patch_template is None:
            return None
        worst = self.dataset.worst_case_shape()
        if worst is None:
            return None
        candidate = self._vram_patch_candidate or concretize_patch_size(
            self._vram_patch_template, worst, self._downsampling_factor
        )
        reserve = self._accumulation_reserve(candidate, worst)
        if reserve is not None:
            shrunk = super()._shrunken_patch(measured, usable - reserve)
            if shrunk is not None:
                return shrunk
        return super()._shrunken_patch(measured, usable)

    def _accumulation_reserve(self, candidate: list[int], worst: list[int]) -> float | None:
        """Bytes each case keeps resident while its patches accumulate, per output writer: the
        streamed window (one patch extent x the cross-section) when the writer will stream --
        single augmentation, voxel-local reduction: the assembled volume otherwise. ``None``
        when a writer's channels cannot be read off the model trace (no reserve, gate decides).
        """
        trace = {name: args.out_channels for name, _, args in self.model.named_module_args_dict()}
        elem = 2  # ModelComposite casts float32 outputs to float16 before accumulation
        reserve = 0.0
        for name, writer in self.outputs_dataset.items():
            out_channels = trace.get(name.replace(";accu;", ""))
            if not out_channels:
                return None
            if isinstance(self.combine, Concat):
                out_channels *= max(1, len(self.path_to_models))
            nb_augmentation = max(1, writer.nb_data_augmentation)
            streams = nb_augmentation == 1 and writer.reduction.voxel_local
            voxels = candidate[0] * np.prod(worst[1:], dtype=np.int64) if streams else np.prod(worst, dtype=np.int64)
            reserve += float((out_channels + 1) * voxels * elem * nb_augmentation)
        return reserve

    def __str__(self) -> str:
        params = {
            "model": self.model,
            "dataset": self.dataset,
            "combine": self.combine,
            "train_name": self.name,
            "manual_seed": self.manual_seed,
            "gpu_checkpoints": self.gpu_checkpoints,
            "autocast": self.autocast,
            "outputs_dataset": self.outputs_dataset,
            "data_log": self.data_log,
        }
        return str(params)

    def __repr__(self) -> str:
        return str(self)


def build_predict(
    models: list[Path],
    prediction_file: Path | str | dict = Path("./Prediction.yml"),
    predictions_dir: Path | str = Path("./Predictions"),
) -> DistributedObject:
    """
    Build and return the configured prediction workflow without executing it.

    Parameters
    ----------
    models : list[Path]
        One or more checkpoint files to load for prediction.
    prediction_file : Path | str, optional
        Prediction configuration file.
    predictions_dir : Path | str, optional
        Directory where prediction outputs are written.

    Returns
    -------
    DistributedObject
        Configured predictor object ready to be executed by the runtime wrapper.
    """
    configure_workflow_environment(
        config_path=prediction_file,
        root="Predictor",
        state=State.PREDICTION,
        path_env={"KONFAI_PREDICTIONS_DIRECTORY": predictions_dir},
    )
    os.environ["KONFAI_CONFIG_MODE"] = "Done"
    with strict_config("Predictor", refuse=False):
        predictor = apply_config()(Predictor)()
    predictor.set_models(models)
    return predictor


@run_distributed_app
def predict(
    models: list[Path],
    overwrite: bool = False,
    gpu: list[int] | None = None,
    cpu: int = 1,
    quiet: bool = False,
    tensorboard: bool = False,
    prediction_file: Path | str | dict = Path("./Prediction.yml"),
    predictions_dir: Path | str = Path("./Predictions"),
) -> DistributedObject:
    """
    Build and execute the configured prediction workflow.

    ``overwrite``/``gpu``/``cpu``/``quiet``/``tensorboard`` are load-bearing even though the body
    drops them: :func:`run_distributed_app` reads them from the bound signature to drive the launch.
    The pure build step is :func:`build_predict`.
    """
    del overwrite, gpu, cpu, quiet, tensorboard
    return build_predict(
        models=models,
        prediction_file=prediction_file,
        predictions_dir=predictions_dir,
    )

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

import hashlib
import json
import multiprocessing
import os
import shutil
import warnings
import zipfile
from collections.abc import Sequence
from multiprocessing.sharedctypes import SynchronizedArray
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from konfai import config_file, cuda_visible_devices, konfai_root, predictions_directory
from konfai.data.data_manager import DataPrediction, GrowingBatchSampler
from konfai.data.reduction import Concat
from konfai.network.network import Model, ModelLoader, Network, place_graph
from konfai.predictor.ensemble import ModelComposite
from konfai.predictor.loop import _Predictor
from konfai.predictor.output import OutputDatasetLoader
from konfai.utils import vram
from konfai.utils.budget import node_local_ranks, set_per_rank_budget
from konfai.utils.clock import startup_clock
from konfai.utils.config import _load_tree, apply_config, config, strict_config
from konfai.utils.dataset import refuse_shared_single_file
from konfai.utils.errors import ConfigError, KonfAIWarning, PredictorError
from konfai.utils.ome_zarr import bound_chunk_cache
from konfai.utils.runtime import (
    DataLog,
    DistributedObject,
    State,
    checkpoint_source,
    configure_workflow_environment,
    run_distributed_app,
)
from konfai.utils.utils import concretize_patch_size, get_module, module_attribute


@config()
class Predictor(vram.VramAutoPatchMixin, DistributedObject):
    """
    KonfAI's main prediction controller.

    It loads the model weights from checkpoint(s) or URL(s), prepares the datasets and the output
    configurations, runs the inference distributed over the available GPUs, applies the transforms and
    writes the predictions, and optionally logs to TensorBoard.

    Attributes:
        model (Network): The neural network model to use for prediction.
        dataset (konfai.data.data_manager.DataPrediction): Dataset manager for prediction data.
        combine (konfai.data.reduction.Reduction): How the ensemble members' outputs are folded (``Mean`` by default).
        autocast (bool): Whether to enable AMP inference.
        cudnn_benchmark (bool): Let cuDNN benchmark its kernels under ``manual_seed``: faster, no bit-for-bit replay.
        torch_compile (bool): Compile the model's graph walk with torch.compile; the first batches pay the compilation.
        outputs_dataset (dict[str, OutputDataset]): Mapping from layer names to output writers.
        data_log (list[str] | None): List of tensors to log during inference.
    """

    # The ranks share the work list and nothing else: each writes its own cases.
    uses_collectives = False

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
        cudnn_benchmark: bool = False,
        torch_compile: bool = False,
        outputs_dataset: dict[str, OutputDatasetLoader] | None = {"default|Default": OutputDatasetLoader()},
        data_log: list[str] | None = None,
        checkpoint_cache_gib: float = 1.0,
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
        #: One flag per case, set by the rank that predicted it, while the launcher waits for the ranks.
        self._predicted_flags: SynchronizedArray[int] | None = None
        #: The cases the last launch predicted; ``None`` when no launcher waited for the ranks.
        self.predicted: list[str] | None = None
        module, name = get_module(combine, "konfai.data.reduction")
        if module.__name__ == "konfai.data.reduction":
            self.combine = module_attribute(module, name)()
        else:
            self.combine = apply_config(f"{konfai_root()}.{combine}")(module_attribute(module, name))()

        self.autocast = autocast
        self.channels_last = channels_last
        self.cudnn_benchmark = cudnn_benchmark
        self.torch_compile = torch_compile
        self.checkpoint_cache_gib = checkpoint_cache_gib
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
        # The outputs hold their cases at once: each is priced against its share of the rank's budget.
        output_budget = per_rank_budget / max(1, len(self.outputs_dataset))
        for output_dataset in self.outputs_dataset.values():
            output_dataset.set_memory_budget(output_budget)
            self.datasets_filename.append(output_dataset.filename)
            # Rebase under the run directory, re-deriving is_directory from the path, not from a trailing "/".
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
        # Cut the grids with the model's downsampling multiple known, so each case's free axis rounds up
        # to a valid input size.
        self.dataset.set_free_axis_multiple(self.model.downsampling_factor())
        # The TTA draws happen in prepare(), before the run seeds anything: keyed by this seed and the
        # case name, a case's copies depend neither on the cases drawn before it nor on the caller's RNG.
        for augmentations in self.dataset.data_augmentations_list.values():
            augmentations.draw_seed = 0 if manual_seed is None else manual_seed
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
                    f"The output group '{output_group}' under 'outputs_dataset' "
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

    def outputs(self) -> list[Path]:
        # A dataset_filename may be absolute, so each output dataset is named, not only the run directory.
        roots = [Path(output_dataset.filename) for output_dataset in self.outputs_dataset.values()]
        return list(dict.fromkeys(roots)) or [self.predict_path]

    def setup(self, world_size: int):
        """Set up the predictor for inference: create the output directories, copy the configuration
        file (Prediction.yml) into the output directory, load the pretrained weights from local files or
        remote URLs, wrap the base model into a ``ModelComposite`` for ensemble inference and build the
        prediction dataloader, distributed over the ``world_size`` processes or GPUs.
        """
        self.size = len(self.gpu_checkpoints) + 1 if self.gpu_checkpoints else 1
        refuse_shared_single_file(world_size // self.size, self.outputs_dataset.values(), PredictorError)
        for dataset_filename in self.datasets_filename:
            path = self.predict_path / dataset_filename
            if not os.path.exists(path):
                os.makedirs(path)

        # Per-case resume, the semantics TRANSFORM documents: a case whose every configured output is
        # already on disk is skipped, and --overwrite recomputes everything. The set is frozen here, on
        # the launcher, so every rank (and every OOM-restart re-plan) shards the same work list.
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

        # The kept cases' recipe, config and weights, is the archived one: a changed recipe is not written
        # over it, and the new cases are not computed with it.
        archived, weights = self.predict_path / "Prediction.yml", self.predict_path / "Weights.json"
        identity = [weights_identity(source) for source in checkpoint_sources(self.path_to_models)]
        if self._done_case_indices and not (
            archived.exists()
            and weights.exists()
            and _load_tree(archived) == _load_tree(config_file())
            and json.loads(weights.read_text()) == identity
        ):
            raise PredictorError(
                f"{len(self._done_case_indices)} case(s) in '{self.predict_path}' were written by another "
                "Prediction.yml, model or checkpoints, which this run would no longer describe.",
                "Run with --overwrite to recompute them with this one, or give this run another train_name.",
            )
        shutil.copyfile(config_file(), archived)
        weights.write_text(json.dumps(identity))

        self.model_composite = ModelComposite(self.model, self.combine, checkpoint_cache_gib=self.checkpoint_cache_gib)
        if not self.path_to_models and any(parameter.numel() for parameter in self.model.parameters()):
            # A model WITH weights but no checkpoint would run with random weights: refuse it. A WEIGHTLESS
            # model (0 parameters, a classical engine such as registration) runs once as constructed.
            raise PredictorError(
                "No model checkpoint available for prediction.",
                "This model has trainable weights, so at least one '.pt' checkpoint must be provided (for "
                "KonfAI Apps, declare it via the 'models' field in app.json).",
                "Without a checkpoint its weights are random and prediction would silently produce garbage.",
            )
        with startup_clock().phase("checkpoint"):
            self.model_composite.load(self._load())

        self._drop_done_cases()
        self.dataloader, _, _ = self.dataset.get_data(world_size // self.size)

    def _drop_done_cases(self) -> None:
        """Drop the already-written cases' entries from the prepared patch mapping.

        Applied to the mapping rather than the case list so the surviving cases keep their indices, and
        re-applied after every ``replan_patch``, which rebuilds the mapping from scratch.
        """
        if not self._done_case_indices:
            return
        self.dataset._prepared_mapping = [
            entry for entry in self.dataset._prepared_mapping if entry[0] not in self._done_case_indices
        ]

    def set_models(self, path_to_models: list[Path | str]) -> None:
        self.path_to_models = path_to_models

    def _load(self) -> list[dict[str, Any] | Path | str]:
        """The checkpoint sources for ensemble prediction, one per model (:func:`checkpoint_sources`)."""
        return [*checkpoint_sources(self.path_to_models)]

    def run_process(
        self,
        world_size: int,
        global_rank: int,
        local_rank: int,
        dataloaders: list[DataLoader],
    ):
        """Launch prediction on the given process rank.

        ``world_size`` is the number of model replicas sharding the data: the spawned process count
        already divided by the model-parallel size (``gpu_checkpoints``), NOT the GPU count.
        """

        model_composite = (
            place_graph(self.model_composite, local_rank * self.size)
            if len(cuda_visible_devices())
            else self.model_composite
        )
        if self.channels_last:
            Network.set_channels_last(model_composite)
        if self.torch_compile:
            # The replicas of an ensemble run in turn through this one model, so one compiled walk
            # serves them all.
            eager = self.model_composite._get_model().compile_walk()
            if eager is not None and global_rank == 0:
                print(f"[KonfAI] torch_compile is set, but the graph walk stays eager: {eager}.", flush=True)
        if len(cuda_visible_devices()):
            # Co-locate the output writers with the model so their reduction/transforms know the GPU.
            for output_dataset in self.outputs_dataset.values():
                output_dataset.to(local_rank * self.size)
        # Before the first forward: a measured batch is measured beside the members kept on the device.
        if self.model_composite.keep_resident() and global_rank == 0:
            print(f"[KonfAI] VRAM: the {len(self.path_to_models)} checkpoints stay on the device.", flush=True)
        model_composite = Model(model_composite)
        device = local_rank * self.size if len(cuda_visible_devices()) else None
        dataloader = dataloaders[0]
        # A whole-axis extent still too large for VRAM OOMs into the shrink loop below, which keeps the
        # size valid too.
        if self._vram_patch_candidate is None and self._presize_free_axes():
            dataloader = self._rank_dataloader(world_size, global_rank)
        # The loader grows where the batch is measured: on a GPU, over patches that stack.
        measure_batch_on = device if isinstance(dataloader.batch_sampler, GrowingBatchSampler) else None
        batch_cap: int | None = None
        while True:
            predictor = _Predictor(
                world_size,
                global_rank,
                local_rank,
                self.autocast,
                self.predict_path,
                self.data_log,
                self.outputs_dataset,
                model_composite,
                dataloader,
                measure_batch_on,
                batch_cap,
                device,
            )
            try:
                with predictor:
                    predictor.run()
                # Each rank names its own; the cases whose header no rank could read, rank 0.
                set_aside = dict(predictor.set_aside.values())
                if global_rank == 0:
                    set_aside.update(self.dataset.unreadable[0])
                if set_aside:
                    who = f"rank {global_rank}: " if world_size > 1 else ""
                    warnings.warn(
                        f"{who}{len(set_aside)} case(s) could not be read and have no prediction:"
                        f" {', '.join(sorted(set_aside))}. A rerun predicts them once their files read.",
                        KonfAIWarning,
                        stacklevel=2,
                    )
                if self._predicted_flags is None:
                    self._refuse_a_cohort_set_aside(world_size, predictor, set_aside)
                else:
                    managers = next(iter(predictor.dataset.data.values()))
                    for x in {x for x, _, _ in predictor.dataset.mapping} - predictor.set_aside.keys():
                        self._predicted_flags[managers[x].index] = 1
                return
            except torch.cuda.OutOfMemoryError:
                # The restart loop IS the sizing iteration: the run that just OOMed already measured the
                # step's transient. Read it BEFORE the reset, free the in-flight state (open streamed sinks
                # abort and remove their partial entries), then read the honest free VRAM.
                measured = vram.transient_at_oom(device)
                for output_dataset in self.outputs_dataset.values():
                    output_dataset.reset()
                if self.model_composite.release_resident():
                    # The resident members give their room back before the batch or the patch shrinks.
                    vram.reset_peak(device)
                    print(f"[KonfAI] VRAM: rank {global_rank} ran out of memory -> checkpoints reload, restarting.")
                    dataloader = self._rank_dataloader(world_size, global_rank)
                    continue
                if measure_batch_on is not None and predictor.batch > 1:
                    # A measured batch over what the device holds halves before any patch shrinks.
                    batch_cap = predictor.batch // 2
                    vram.reset_peak(device)
                    print(f"[KonfAI] VRAM: rank {global_rank} ran out of memory -> batch {batch_cap}, restarting.")
                    dataloader = self._rank_dataloader(world_size, global_rank)
                    continue
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

    def launch_ranks(self, world_size: int) -> None:
        """Run the ranks, then refuse a run that predicted nothing because no case of the cohort could
        be read: that is a wrong tree or an unreadable disk more often than a bad file. The ranks share
        no process group, so each flags the cases it predicted in memory this launcher reads."""
        self._predicted_flags = multiprocessing.get_context("spawn").Array("b", len(self.dataset.case_names))
        try:
            super().launch_ranks(world_size)
            flags = self._predicted_flags[:]
        finally:
            self._predicted_flags = None
        self.predicted = [name for name, flag in zip(self.dataset.case_names, flags, strict=True) if flag]
        cohort = len(self.dataset.case_names) + len(self.dataset.unreadable[0])
        if cohort and not self.predicted and not self._done_case_indices:
            raise _nothing_predicted(cohort)

    def _refuse_a_cohort_set_aside(self, world_size: int, predictor: _Predictor, set_aside: dict[str, str]) -> None:
        """On a rank no launcher waits for (a cluster job), the same refusal, where the rank can tell:
        the only rank, or every rank when no header read."""
        if not set_aside or self._done_case_indices or (world_size > 1 and self.dataset.case_names):
            return
        if {x for x, _, _ in predictor.dataset.mapping} <= predictor.set_aside.keys():
            raise _nothing_predicted(len(set_aside))

    def _rank_dataloader(self, world_size: int, global_rank: int) -> DataLoader:
        """This rank's loader over the re-planned grids, the already-written cases dropped again
        (a re-plan rebuilds the mapping from scratch)."""
        self._drop_done_cases()
        return self.dataset.get_data(world_size)[0][global_rank][0]

    def _presize_free_axes(self) -> bool:
        """The shared pre-sizing, and where the batch is measured, every free axis pinned to the largest extent
        among the cases: the patches then share one shape, a smaller case padded up to it, and the batch grows
        over all of them. Measured on a GPU only; one patch per batch needs no common shape."""
        if super()._presize_free_axes():
            return True
        if (
            self._vram_patch_template is None
            or not self.dataset.measures_batch
            or not cuda_visible_devices()
            or self.dataset.patches_stack()
        ):
            return False
        worst = self.dataset.worst_case_shape()
        if worst is None:
            return False
        self._adopt_patch_candidate(concretize_patch_size(self._vram_patch_template, worst, self._downsampling_factor))
        return True

    def _shrunken_patch(self, measured: int | None, usable: float) -> list[int] | None:
        """The shared shrink step, with the blend kept on the GPU when it fits: the accumulation
        footprint is RESERVED beside the forward, so the sized patch passes the accumulation gate.
        Only when that reserve fits at no size, or cannot be priced, is the forward sized alone.
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
        """Bytes each case keeps resident while its patches accumulate, per output writer: the streamed
        window (one patch extent x the cross-section) when the writer will stream (single augmentation,
        voxel-local reduction), the assembled volume otherwise. ``None`` when a writer's channels cannot
        be read off the model trace.
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


def checkpoint_sources(path_to_models: Sequence[Path | str]) -> list[Path | str]:
    """Each checkpoint as ``ModelComposite`` loads it. A URL remains a reloadable source: torch.hub keeps
    its download on disk, while the composite's bounded host cache decides which deserialized weights
    stay resident. A local path is kept as a path, its weights streamed into a single model instance
    during prediction. A path that does not exist or is a directory is refused by name
    (:func:`~konfai.utils.runtime.checkpoint_source`)."""
    return [checkpoint_source(path_to_model, PredictorError) for path_to_model in path_to_models]


def weights_identity(source: Path | str) -> str:
    """What a checkpoint holds, read cheaply. A URL is its address. A torch checkpoint is a zip whose
    directory lists each tensor's CRC-32 and size, read without the weights; its top folder is named
    after the file, so a copy under another name holds the same weights. Another file is its SHA-256."""
    if isinstance(source, str):
        return source
    try:
        with zipfile.ZipFile(source) as archive:
            members = sorted((info.filename.split("/", 1)[-1], info.CRC, info.file_size) for info in archive.infolist())
        return hashlib.sha256(repr(members).encode()).hexdigest()
    except zipfile.BadZipFile:
        with open(source, "rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()


def build_predict(
    models: Sequence[Path | str],
    prediction_file: Path | str | dict = Path("./Prediction.yml"),
    predictions_dir: Path | str = Path("./Predictions"),
) -> DistributedObject:
    """Build and return the configured prediction workflow without executing it.

    ``models`` are the checkpoint files to load, ``prediction_file`` the prediction configuration and
    ``predictions_dir`` the directory the outputs are written to. The predictor comes back ready to be
    executed by the runtime wrapper.
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


def _nothing_predicted(cases: int) -> PredictorError:
    return PredictorError(
        f"None of the {cases} case(s) could be read, so nothing was predicted.",
        "Check the dataset path and the files' permissions; the warnings name each case and why.",
    )


@run_distributed_app
def predict(
    models: Sequence[Path | str],
    overwrite: bool = False,
    gpu: list[int] | None = None,
    cpu: int = 1,
    quiet: bool = False,
    tensorboard: bool = False,
    prediction_file: Path | str | dict = Path("./Prediction.yml"),
    predictions_dir: Path | str = Path("./Predictions"),
) -> DistributedObject:
    """Build and execute the configured prediction workflow.

    ``overwrite``/``gpu``/``cpu``/``quiet``/``tensorboard`` are load-bearing even though the body drops
    them: :func:`run_distributed_app` reads them from the bound signature to drive the launch. The pure
    build step is :func:`build_predict`.
    """
    del overwrite, gpu, cpu, quiet, tensorboard
    checkpoint_sources(models)  # before the build reads the config and lists the dataset
    return build_predict(
        models=models,
        prediction_file=prediction_file,
        predictions_dir=predictions_dir,
    )

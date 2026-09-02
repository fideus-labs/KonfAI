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


"""The per-rank prediction loop: fetch, forward, blend, finalize."""

from pathlib import Path

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]
from konfai.data.data_manager import (
    BatchSample,
    DatasetIter,
)
from konfai.network.network import Model, NetState
from konfai.predictor.output import PREDICTION_CLOCK, OutputDataset
from konfai.utils.clock import SweepClock
from konfai.utils.runtime import (
    DataLog,
    DistributedObject,
    NullSummaryWriter,
    description,
)

#: Batches between two refreshes of the progress bar's status. The status is an NVML query, two
#: psutil calls and a forced redraw (95 us per batch, measured) that no batch needs to be current.
_DESCRIPTION_EVERY = 10


def _prediction_report(clock: SweepClock, min_seconds: float = 1.0) -> str | None:
    """One line accounting for the prediction loop's wall clock, in the sweep report's shape, or
    ``None`` below ``min_seconds``.

    The sum before the bar is the loop's own thread and closes exactly: what its phases do not
    name (the logging, the progress bar, the bookkeeping between them) is ``other``. ``fetch`` is
    the wait for the loader's next batch, ``blend`` a patch's inverses and its blend into the
    accumulator (the copy home included on the host route), the two ``finalize`` the slabs and the
    cases handed to the writer, ``drain`` the writes still queued when the loop ends. On a GPU the
    loop only enqueues the forward and the blend; the device's time is waited for where a result
    crosses to the host.

    After the bar is the writer: its own thread's time, and how long the loop stood waiting on it
    inside the finalize phases (a full queue, or the write itself when the destination keeps it
    inline). A slow destination shows there, as the wait that turns the background writer
    synchronous, which nothing reported before.
    """
    wall = clock.spent("prediction")
    if wall < min_seconds:
        return None
    phases = ("fetch", "forward", "blend", "finalize(stream)", "finalize(case)", "drain")
    named = {phase: clock.spent(phase) for phase in phases}
    parts = " + ".join(f"{phase} {value:.1f}" for phase, value in named.items())
    return (
        f"[KonfAI] prediction {wall:.1f} s = {parts} + other {wall - sum(named.values()):.1f}"
        f" | writer {clock.spent('write'):.1f} s, waited on {clock.spent('wait(write)'):.1f} s"
    )


class _Predictor:
    """
    Internal class that runs distributed inference over a dataset using a composite model.

    This class handles patch-wise prediction, output accumulation, logging to TensorBoard, and
    writing final predictions to disk. It is designed to be used as a context manager and
    supports model ensembles via `ModelComposite`.

    Args:
        world_size (int): Total number of processes or GPUs used.
        global_rank (int): Rank of the current process across all nodes.
        local_rank (int): Local GPU index within a single node.
        autocast (bool): Whether to use automatic mixed precision (AMP).
        predict_path (str): Output directory path where predictions and metrics are saved.
        data_log (list[str] | None): List of logging targets in the format 'group/DataLogType/N'.
        outputs_dataset (dict[str, OutputDataset]): Dictionary of output datasets to store predictions.
        model_composite (Model): Model container that wraps the prediction model(s).
        dataloader_prediction (DataLoader): DataLoader that provides prediction batches.
    """

    def __init__(
        self,
        world_size: int,
        global_rank: int,
        local_rank: int,
        autocast: bool,
        predict_path: Path,
        data_log: list[str] | None,
        outputs_dataset: dict[str, OutputDataset],
        model_composite: Model,
        dataloader_prediction: DataLoader,
    ) -> None:
        self.world_size = world_size
        self.global_rank = global_rank
        self.local_rank = local_rank

        self.model_composite = model_composite

        self.dataloader_prediction = dataloader_prediction
        self.outputs_dataset = outputs_dataset
        self.autocast = autocast
        self.it = 0

        self.dataset: DatasetIter = self.dataloader_prediction.dataset
        patch_size, overlap = self.dataset.get_patch_config()
        for output_dataset in self.outputs_dataset.values():
            output_dataset.set_patch_config(
                patch_size,
                overlap,
                np.max(
                    [
                        int(
                            np.sum([data_augmentation.nb for data_augmentation in self.dataset.data_augmentations_list])
                            + 1
                        ),
                        1,
                    ]
                ),
            )
        self.data_log = DataLog.parse(data_log)
        self._has_runtime_measures = any(
            network.measure is not None for network in self.model_composite.module.get_networks().values()
        )
        if self._has_runtime_measures or len(self.data_log):
            if SummaryWriter is None:
                # A missing logger must never refuse the run: the predictions are still written,
                # only the curves and images are lost. One line says so; the extra keeps them.
                if self.global_rank == 0:
                    print(
                        "[KonfAI] TensorBoard is not installed: no curves or images will be logged"
                        " (pip install konfai[tensorboard] to keep them)."
                    )
                self.tb = NullSummaryWriter()
            else:
                self.tb = SummaryWriter(log_dir=predict_path / "Metric")
        else:
            self.tb = None

    def __enter__(self):
        """
        Enters the prediction context and returns the predictor instance.
        """
        return self

    def __exit__(self, exc_type, value, traceback):
        """
        Closes the TensorBoard writer upon exit.
        """
        if self.tb:
            self.tb.close()

    def run(self):
        """
        Run the full prediction loop.

        Iterates over the prediction DataLoader, performs inference using the composite model,
        applies reduction (e.g., mean), and writes the final results using each `OutputDataset`.

        Also logs intermediate data and metrics to TensorBoard if enabled.
        """

        self.model_composite.eval()
        self.model_composite.module.set_state(NetState.PREDICTION)
        self.dataloader_prediction.dataset.load("Prediction")
        PREDICTION_CLOCK.reset()
        try:
            with PREDICTION_CLOCK.phase("prediction"):
                self._run_batches()
        finally:
            # Every submitted write must be on disk before the run returns: including on the error
            # path, where the drain also closes the sinks the abort operations enqueued.
            with PREDICTION_CLOCK.phase("prediction"), PREDICTION_CLOCK.phase("drain"):
                for output_dataset in self.outputs_dataset.values():
                    output_dataset.finalize_writes()
        if self.global_rank == 0:
            clock = _prediction_report(PREDICTION_CLOCK)
            if clock is not None:
                print(clock)

    def _run_batches(self) -> None:
        with tqdm.tqdm(
            iterable=enumerate(PREDICTION_CLOCK.waiting("fetch", self.dataloader_prediction)),
            leave=True,
            desc=f"Prediction : {description(self.model_composite)}",
            total=len(self.dataloader_prediction),
            ncols=0,
        ) as batch_iter:
            with torch.inference_mode():
                with torch.amp.autocast("cuda", enabled=self.autocast):
                    for batch_index, batch_sample in batch_iter:
                        with PREDICTION_CLOCK.phase("forward"):
                            outputs = self.model_composite(
                                batch_sample,
                                list(self.outputs_dataset.keys()),
                            )
                        self._predict_log(batch_sample)
                        for name, number_of_channels_per_model, output in outputs:
                            output_dataset = self.outputs_dataset[name]
                            group = getattr(output_dataset, "group_dest", next(iter(batch_sample)))
                            for i, (index, patch_augmentation, patch_index) in enumerate(
                                [
                                    (int(index), int(patch_augmentation), int(patch_index))
                                    for index, patch_augmentation, patch_index in zip(
                                        batch_sample[group].x,
                                        batch_sample[group].a,
                                        batch_sample[group].p,
                                        strict=False,
                                    )
                                ]
                            ):
                                output_dataset.add_layer(
                                    index,
                                    patch_augmentation,
                                    patch_index,
                                    output[i],
                                    self.dataset,
                                    batch_sample[group].attribute[i],
                                    number_of_channels_per_model,
                                )
                                if output_dataset.is_done(index):
                                    with PREDICTION_CLOCK.phase("finalize(case)"):
                                        output_dataset.write_prediction(
                                            index,
                                            batch_sample[group].name[i],
                                            output_dataset.get_output(
                                                index, number_of_channels_per_model, self.dataset
                                            ),
                                        )

                        if batch_index % _DESCRIPTION_EVERY == 0:
                            # The bar redraws on its own clock; the status only has to be there by then.
                            batch_iter.set_description(
                                f"Prediction : {description(self.model_composite)}", refresh=False
                            )
                        self.it += 1

    def _predict_log(
        self,
        batch_sample: BatchSample,
    ):
        """
        Log prediction results to TensorBoard, including images and metrics.

        This method handles:
        - Logging image-like data (e.g., inputs, outputs, masks) using `DataLog` instances,
        based on the `data_log` configuration.
        - Logging scalar loss and metric values (if present in the network) under the `Prediction/` namespace.
        - Dynamically retrieving additional feature maps or intermediate layers if requested via `data_log`.

        Logging is performed only on the global rank 0 process and only if `TensorBoard` is active.

        Args:
            data_dict (dict): Dictionary mapping group names to 6-tuples containing:
                - input tensor,
                - index,
                - patch_augmentation,
                - patch_index,
                - metadata (list of strings),
                - `requires_grad` flag (as a tensor).
        """
        if self.tb is None or self.global_rank != 0:
            # Prediction logging is a rank-0 progress indicator; gate before touching the measures so a
            # non-zero rank never enters a cross-rank collective the unequal shards would deadlock on.
            return

        measures: dict[str, tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]] = {}
        if self._has_runtime_measures:
            measures = DistributedObject.get_measure(
                1,
                0,
                self.local_rank,
                {"": self.model_composite.module},
                1,
                sync=False,
            )

        for name, network in self.model_composite.module.get_networks().items():
            if network.measure is not None:
                self.tb.add_scalars(
                    f"Prediction/{name}/Loss",
                    {k.replace(":", "."): v[1] for k, v in measures[name][0].items()},
                    self.it,
                )
                self.tb.add_scalars(
                    f"Prediction/{name}/Metric",
                    {k.replace(":", "."): v[1] for k, v in measures[name][1].items()},
                    self.it,
                )

        # Images are a progress peek, not a per-batch record, and a module-layer target re-runs a
        # full forward (get_layers): both throttle to the status cadence.
        if not len(self.data_log) or self.it % _DESCRIPTION_EVERY != 0:
            return
        images_log = []
        for name, data_type in self.data_log.items():
            if name in batch_sample:
                data_type[0](
                    self.tb,
                    f"Prediction/{name}",
                    batch_sample[name].tensor[: self.data_log[name][1]].detach().cpu().numpy(),
                    self.it,
                )
            else:
                images_log.append(name.replace(":", "."))
        if len(images_log):
            # get_layers is model-scoped, not per-network: run it once per model, or a multi-network
            # model (a GAN's generator + discriminator) repeats the forward extraction and writes
            # each image event once per network.
            for layer_name, layer, _ in self.model_composite.module.get_layers(
                [v.tensor for v in batch_sample.values() if v.is_input],
                images_log,
            ):
                self.data_log[layer_name][0](
                    self.tb,
                    f"Prediction/{layer_name}",
                    layer[: self.data_log[layer_name][1]].detach().cpu().numpy(),
                    self.it,
                )

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

"""Evaluation workflow classes and helpers for KonfAI."""

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from konfai import config_file, cuda_visible_devices, evaluations_directory, konfai_root
from konfai.data.data_manager import BatchDataItem, BatchSample, DataMetric, DatasetIter
from konfai.network.network import build_configured_criterions
from konfai.utils.config import apply_config, config
from konfai.utils.dataset import Attribute, Dataset, DataStream
from konfai.utils.errors import ConfigError, EvaluatorError
from konfai.utils.runtime import (
    DistributedObject,
    State,
    clear_directory_except_logs,
    configure_workflow_environment,
    confirm_overwrite_or_raise,
    run_distributed_app,
    synchronize_data,
)
from konfai.utils.utils import split_path_spec


class CriterionsLoader:
    """
    Loader for multiple criterion modules to be applied between a model output and one or more targets.

    Each loss module (e.g., Dice, CrossEntropy, NCC) is dynamically loaded using its fully-qualified
    classpath. Evaluation criteria carry no per-criterion attributes, so the config value bound to each
    classpath is an unused placeholder (``None``).

    Args:
        criterions_loader (dict): A mapping from module classpaths (as strings) to placeholder values.
                                  The module path is parsed and instantiated via `get_module`.

    """

    def __init__(
        self,
        criterions_loader: dict[str, Any] = {"default|torch:nn:CrossEntropyLoss|Dice|NCC": None},
    ) -> None:
        self.criterions_loader = criterions_loader

    def get_criterions(self, output_group: str, target_group: str) -> dict[torch.nn.Module, Any]:
        return build_configured_criterions(
            self.criterions_loader,
            f"{konfai_root()}.metrics.{output_group}.targets_criterions.{target_group}",
        )


class TargetCriterionsLoader:
    """
    Loader class for handling multiple target groups with associated criterion configurations.

    This class allows defining a set of criterion loaders (e.g., Dice, BCE, MSE) for each
    target group to be used during evaluation or training. Each target group corresponds
    to one or more loss functions, all linked to a specific model output.

    Args:
        targets_criterions (dict[str, CriterionsLoader]): Dictionary mapping each target group name
            to a `CriterionsLoader` instance that defines its associated loss functions.
    """

    def __init__(
        self,
        targets_criterions: dict[str, CriterionsLoader] = {"default": CriterionsLoader()},
    ) -> None:
        self.targets_criterions = targets_criterions

    def get_targets_criterions(self, output_group: str) -> dict[str, dict[torch.nn.Module, Any]]:
        """
        Retrieve the criterion modules and their attributes for a specific output group.

        This function prepares the loss functions to be applied for a given model output,
        grouped by their target group.

        Args:
            output_group (str): Name of the model output group (e.g., "output_segmentation").

        Returns:
            dict[str, dict[nn.Module, Any]]: A nested dictionary where the first key is the
            target group name, and the value is a dictionary mapping each loss module to its placeholder.
        """
        targets_criterions = {}
        for target_group, criterions_loader in self.targets_criterions.items():
            targets_criterions[target_group] = criterions_loader.get_criterions(output_group, target_group)
        return targets_criterions


class Statistics:
    """
    Utility class to accumulate, structure, and write evaluation metric results.

    This class is used to:
    - Collect metrics for each dataset sample.
    - Compute aggregate statistics (mean, std, percentiles, etc.).
    - Export all results in a structured JSON format, including both per-case and aggregate values.

    Args:
        filename (str): Path to the output JSON file that will store the final results.
    """

    def __init__(self, filename: Path) -> None:
        self.measures: dict[str, dict[str, float]] = {}
        self.filename = filename
        # Per-metric optimisation direction ("max"/"min"), declared by each criterion's `maximize`
        # property, so downstream ranking (the MCP leaderboard) reads it instead of guessing from names.
        self.directions: dict[str, str] = {}

    def add(self, values: dict[str, float], name_dataset: str) -> None:
        """
        Add a set of metric values for a given dataset case.

        Args:
            values (dict): Dictionary of metric names and their values.
            name_dataset (str): Identifier (e.g., case name) for the sample.
        """
        for name, value in values.items():
            if name_dataset not in self.measures:
                self.measures[name_dataset] = {}
            self.measures[name_dataset][name] = value

    @staticmethod
    def get_statistic(values: list[float]) -> dict[str, float]:
        """
        Compute statistical aggregates for a list of metric values.

        Args:
            values (list of float): Values to summarize.

        Returns:
            dict[str, float]: A dictionary containing:
                - max, min, std
                - 25th, 50th, and 75th percentiles
                - mean and count
        """
        return {
            "max": float(np.nanmax(values)) if np.any(~np.isnan(values)) else np.nan,
            "min": float(np.nanmin(values)) if np.any(~np.isnan(values)) else np.nan,
            "std": float(np.nanstd(values)) if np.any(~np.isnan(values)) else np.nan,
            "25pc": float(np.nanpercentile(values, 25)) if np.any(~np.isnan(values)) else np.nan,
            "50pc": float(np.nanpercentile(values, 50)) if np.any(~np.isnan(values)) else np.nan,
            "75pc": float(np.nanpercentile(values, 75)) if np.any(~np.isnan(values)) else np.nan,
            "mean": float(np.nanmean(values)) if np.any(~np.isnan(values)) else np.nan,
            "count": float(np.count_nonzero(~np.isnan(values))),
        }

    @staticmethod
    def _to_serializable(obj: Any) -> Any:
        """
        Recursively replace non-finite floating-point values with ``None``.

        NaN and ±Infinity have no representation in standard JSON. Converting them
        to ``null`` keeps the serialized report parseable by strict JSON readers.

        Args:
            obj: Any structure (dict, list, scalar) to normalize.

        Returns:
            The same structure with every non-finite float replaced by ``None``.
        """
        if isinstance(obj, dict):
            return {key: Statistics._to_serializable(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [Statistics._to_serializable(value) for value in obj]
        if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
            return None
        return obj

    def write(self, outputs: list[dict[str, dict[str, Any]]]) -> None:
        """
        Write the collected and aggregated statistics to the configured output file.

        The output JSON structure contains:
        - `case`: All individual metrics per sample.
        - `aggregates`: Global statistics computed over all cases.

        Args:
            outputs (list): List of metric dictionaries to merge and serialize.
        """
        measures = {}
        for output in outputs:
            measures.update(output)
        # JSON payload with heterogeneous blocks: "case"/"aggregates" are nested dicts, "directions"
        # maps metric-name -> "max"/"min".
        result: dict[str, Any] = {}
        result["case"] = {}
        for name, v in measures.items():
            for metric_name, value in v.items():
                if metric_name not in result["case"]:
                    result["case"][metric_name] = {}
                result["case"][metric_name][name] = value

        result["aggregates"] = {}
        tmp: dict[str, list[float]] = {}
        for _, v in measures.items():
            for metric_name, _ in v.items():
                if metric_name not in tmp:
                    tmp[metric_name] = []
                tmp[metric_name].append(v[metric_name])
        for metric_name, values in tmp.items():
            result["aggregates"][metric_name] = Statistics.get_statistic(values)

        # Declare each metric's optimisation direction so consumers rank without guessing.
        directions = {name: self.directions[name] for name in result["aggregates"] if name in self.directions}
        if directions:
            result["directions"] = directions

        with open(self.filename, "w") as f:
            f.write(json.dumps(Statistics._to_serializable(result), indent=4, allow_nan=False))

    def read(self) -> dict[str, float]:
        with open(self.filename) as f:
            json_data = json.load(f)

        result: dict[str, float] = {}

        aggregates = json_data.get("aggregates", {})

        for key, stats in aggregates.items():
            mean_value = stats.get("mean", None)
            if mean_value is None:
                continue

            # A dict-valued metric emits both an aggregate entry
            # ("output:target:Metric") and one entry per component
            # ("output:target:Metric:component"); the latter share the aggregate
            # key as a prefix, so keep only top-level metrics.
            if key.rsplit(":", 1)[0] in aggregates:
                continue

            result[key] = mean_value

        return result


@config()
class Evaluator(DistributedObject):
    """
    Distributed evaluation engine for computing metrics on model predictions.

    This class handles the evaluation of predicted outputs using predefined metric loaders.
    It supports multi-output and multi-target configurations, computes aggregated statistics
    across training and validation datasets, and synchronizes results across processes.

    Evaluation results are stored in JSON format and optionally displayed during iteration.

    Args:
        train_name (str): Unique name of the evaluation run, used for logging and output folders.
        metrics (dict[str, TargetCriterionsLoader]): Dictionary mapping output groups to loaders of target metrics.
        dataset (DataMetric): Dataset provider configured for evaluation mode.

    Attributes:
        statistics_train (Statistics): Object used to store training evaluation metrics.
        statistics_validation (Statistics): Object used to store validation evaluation metrics.
        dataloader (list[DataLoader]): DataLoaders for training and validation sets.
        metric_path (str): Path to the evaluation output directory.
        metrics (dict): Instantiated metrics organized by output and target groups.
    """

    def __init__(
        self,
        train_name: str = "default|TRAIN_01",
        metrics: dict[str, TargetCriterionsLoader] = {"default": TargetCriterionsLoader()},
        dataset: DataMetric = DataMetric(),
    ) -> None:
        if os.environ["KONFAI_CONFIG_MODE"] != "Done":
            raise ConfigError("Evaluator requires KONFAI_CONFIG_MODE='Done' before initialization.")
        super().__init__(train_name)
        self.metric_path = evaluations_directory() / self.name
        self.metricsLoader = metrics if metrics else {}
        self.dataset = dataset
        self.metrics = {k: v.get_targets_criterions(k) for k, v in self.metricsLoader.items()}
        self.statistics_train = Statistics(self.metric_path / "Metric_TRAIN.json")
        self.statistics_validation = Statistics(self.metric_path / "Metric_VALIDATION.json")
        # A memory budget may patch the evaluation, but only when EVERY metric can rebuild its
        # whole-case value from partial states; one non-reducible metric keeps the whole-volume path
        # for everything (correct beats bounded).
        self.dataset.auto_patch_allowed = all(
            getattr(metric, "reducible", False)
            for targets in self.metrics.values()
            for criterions in targets.values()
            for metric in criterions
        )
        self.dataset.prepare()
        # Set iff the budget actually patched: batches then carry one disjoint patch of a case, and
        # update() accumulates partial states until the case's last patch before recording it.
        self._streamed = self.dataset.patch is not None
        self._pending: dict[tuple[str, str, int], tuple] = {}
        self._pending_name: str | None = None
        self._last_result: dict[str, float] = {}
        # Per-voxel error maps under the patched path: one region-write sink per (metric, case),
        # opened at the case's first patch, closed when the case flushes. Disjoint unpadded patches
        # mean every voxel is written exactly once -- the streamed map equals the whole-volume one.
        self._map_sinks: dict[tuple[str, str, int], DataStream] = {}
        self._iter_dataset: DatasetIter | None = None
        self._validate_metric_groups()

    def _validate_metric_groups(self) -> None:
        groups_dest = self.dataset.get_groups_dest()
        missing_outputs = set(self.metrics.keys()) - set(groups_dest)
        if missing_outputs:
            raise EvaluatorError(
                f"The following metric output groups are missing from 'groups_dest': {sorted(missing_outputs)}. ",
                f"Available groups: {sorted(groups_dest)}",
            )

        target_groups = []
        for targets in self.metrics.values():
            for target_group in targets:
                target_groups.extend(target_group.split(";"))
        missing_targets = set(target_groups) - ({*groups_dest, "None"})
        if missing_targets:
            raise EvaluatorError(
                f"The following metric target groups are missing from 'groups_dest': {sorted(missing_targets)}. ",
                f"Available groups: {sorted(groups_dest)}",
            )

    def setup(self, world_size: int):
        """
        Prepare the evaluator for distributed metric computation.

        This method performs the following steps:
        - Checks whether previous evaluation results exist and optionally overwrites them.
        - Creates the output directory and copies the current configuration file for reproducibility.
        - Loads the evaluation dataset according to the world size.

        Args:
            world_size (int): Number of processes in the distributed evaluation setup.

        """
        if self.metric_path.exists() and len(list(self.metric_path.rglob("*.yml"))):
            confirm_overwrite_or_raise(self.metric_path, "metric", EvaluatorError)
            if self.metric_path.exists():
                # This directory holds the rank-0 evaluation log this process already has open:
                # clear around it instead of rmtree'ing it out from under the live file.
                clear_directory_except_logs(self.metric_path)

        os.makedirs(self.metric_path, exist_ok=True)
        shutil.copyfile(
            config_file(),
            self.metric_path / config_file().name,
        )

        self.dataloader, _, _ = self.dataset.get_data(world_size)

    def update(self, batch_sample: BatchSample, statistics: Statistics) -> dict[str, float]:
        """
        Compute metrics for a batch and update running statistics.

        Args:
            batch_sample (BatchSample): The batch sample object containing tensors and their metadata.
            statistics (Statistics): The statistics object to update (train or validation).

        Returns:
            dict[str, float]: Dictionary of computed metric values with keys in the format
                            'output_group:target_group:MetricName'.
        """
        if self._streamed:
            return self._update_streamed(batch_sample, statistics)
        result: dict[str, float] = {}
        for output_group in self.metrics:
            output_tensor = batch_sample[output_group].tensor
            metric_device = output_tensor.device
            for target_group in self.metrics[output_group]:
                targets = self._targets_on(batch_sample, target_group, metric_device)
                target_attribute = [batch_sample[output_group].attribute] + [
                    batch_sample[group].attribute for group in target_group.split(";") if group in batch_sample
                ]
                name = batch_sample[output_group].name[0]
                for metric in self.metrics[output_group][target_group]:
                    if getattr(metric, "accepts_attributes", False):
                        with torch.no_grad():
                            loss = metric(
                                output_tensor,
                                *targets,
                                attributes=target_attribute,
                            )
                    else:
                        with torch.no_grad():
                            loss = metric(
                                output_tensor,
                                *targets,
                            )
                    if isinstance(loss, tuple):
                        true_loss = loss[1]
                        if len(loss) == 3:
                            if metric.dataset:
                                filename, _, file_format = split_path_spec(metric.dataset)
                                map_dataset = Dataset(filename, file_format)
                                group = metric.group if metric.group else output_group
                                for dataset in self.dataset.datasets.values():
                                    for g in dataset.get_group():
                                        if dataset.is_dataset_exist(g, name):
                                            _, cache_attribute = dataset.get_infos(g, name)
                                            map_dataset.write(
                                                group,
                                                name,
                                                loss[2].squeeze(0).numpy(),
                                                cache_attribute,
                                            )
                                            break
                    else:
                        true_loss = loss.item()

                    direction = "max" if getattr(metric, "maximize", False) else "min"
                    base_key = f"{output_group}:{target_group}:{metric.get_name()}"
                    Evaluator._record_value(result, statistics, base_key, true_loss, direction)
        if len(self.metrics) > 0:
            statistics.add(result, name)
        return result

    @staticmethod
    def _targets_on(batch_sample: BatchSample, target_group: str, metric_device: torch.device) -> list[torch.Tensor]:
        """The target tensors of a ``;``-joined group spec, moved to the metric's device."""
        return [
            (
                batch_sample[group].tensor.to(
                    metric_device, non_blocking=batch_sample[group].tensor.device.type == "cpu"
                )
                if batch_sample[group].tensor.device != metric_device
                else batch_sample[group].tensor
            )
            for group in target_group.split(";")
            if group in batch_sample
        ]

    @staticmethod
    def _record_value(
        result: dict[str, float],
        statistics: Statistics,
        base_key: str,
        true_loss: float | dict,
        direction: str,
    ) -> None:
        """Record one metric value: a dict records each component plus their NaN-skipping mean."""
        if isinstance(true_loss, dict):
            total = 0.0
            count = 0
            for k, v in true_loss.items():
                component_key = f"{base_key}:{k}"
                result[component_key] = v
                statistics.directions[component_key] = direction
                if not np.isnan(v):
                    total += v
                    count += 1
            result[base_key] = total / count if count > 0 else np.nan
            statistics.directions[base_key] = direction
        else:
            result[base_key] = true_loss
            statistics.directions[base_key] = direction

    def _update_streamed(self, batch_sample: BatchSample, statistics: Statistics) -> dict[str, float]:
        """Accumulate one PATCH's partial states; record the case when its next sibling arrives.

        The evaluation loader walks a case's disjoint patches contiguously (cases shard whole per
        rank), so a change of case name marks the previous case complete -- ``_flush_pending`` at the
        end of the split closes the last one.
        """
        name = batch_sample[next(iter(self.metrics))].name[0]
        if self._pending_name is not None and name != self._pending_name:
            self._flush_pending(statistics)
        self._pending_name = name
        for output_group in self.metrics:
            output_tensor = batch_sample[output_group].tensor
            metric_device = output_tensor.device
            for target_group in self.metrics[output_group]:
                targets = self._targets_on(batch_sample, target_group, metric_device)
                for index, metric in enumerate(self.metrics[output_group][target_group]):
                    with torch.no_grad():
                        state = metric.partial_metric(output_tensor, *targets)
                    entry = self._pending.setdefault((output_group, target_group, index), (metric, []))
                    entry[1].append(state)
                    if getattr(metric, "dataset", None) and hasattr(metric, "partial_map"):
                        with torch.no_grad():
                            patch_map = metric.partial_map(output_tensor, *targets).squeeze(0)
                        self._write_map_patch(
                            (output_group, target_group, index),
                            metric,
                            batch_sample[output_group],
                            output_group,
                            patch_map,
                        )
        return self._last_result

    def _write_map_patch(
        self,
        key: tuple[str, str, int],
        metric: Any,
        item: BatchDataItem,
        output_group: str,
        patch_map: torch.Tensor,
    ) -> None:
        """Write one patch's per-voxel map into its case's region-write sink.

        ``partial_map`` is voxel-local, so the patch's map is exactly the region of the whole-case
        map; the disjoint unpadded evaluation grid writes every voxel once, never twice.
        """
        if self._iter_dataset is None:
            raise EvaluatorError("Internal error: the streamed evaluation loop has no dataset iterator.")
        manager = self._iter_dataset.get_dataset_from_index(output_group, int(item.x[0]))
        array = patch_map.numpy()
        sink = self._map_sinks.get(key)
        if sink is None:
            filename, _, file_format = split_path_spec(metric.dataset)
            group = metric.group if metric.group else output_group
            sink = Dataset(filename, file_format).open_data_stream(
                group,
                manager.name,
                [array.shape[0], *manager.shapes[0]],
                array.dtype,
                Attribute(manager.cache_attributes[0]),
            )
            if sink is None:
                raise EvaluatorError(
                    f"The '{file_format}' backend cannot serve region writes for the "
                    f"'{metric.get_name()}' error map under a memory_budget.",
                    "Write the map to an mha, h5 or omezarr dataset, or drop 'memory_budget' to "
                    "evaluate whole volumes.",
                )
            self._map_sinks[key] = sink
        region = manager.patch.get_patch_slices(int(item.a[0]))[int(item.p[0])]
        sink.write_slice((slice(0, array.shape[0]), *region), array)

    def _abort_map_sinks(self, error: BaseException) -> None:
        """Close open map sinks WITH the error so the backends remove their partial entries."""
        for sink in self._map_sinks.values():
            sink.abort(error)
        self._map_sinks = {}

    def _flush_pending(self, statistics: Statistics) -> None:
        """Combine the pending case's partial states into its exact values and record them."""
        if self._pending_name is None:
            return
        result: dict[str, float] = {}
        for (output_group, target_group, _index), (metric, states) in self._pending.items():
            loss = metric.combine_metric(states)
            true_loss = loss[1] if isinstance(loss, tuple) else float(loss.item())
            direction = "max" if getattr(metric, "maximize", False) else "min"
            base_key = f"{output_group}:{target_group}:{metric.get_name()}"
            Evaluator._record_value(result, statistics, base_key, true_loss, direction)
        for sink in self._map_sinks.values():
            sink.close()
        self._map_sinks = {}
        if len(self.metrics) > 0:
            statistics.add(result, self._pending_name)
        self._pending = {}
        self._pending_name = None
        self._last_result = result

    def run_process(self, world_size: int, global_rank: int, gpu: int, dataloaders: list[DataLoader]):
        """
        Execute the distributed evaluation loop over the training and validation datasets.

        This method iterates through the provided DataLoaders (train and optionally validation),
        updates the metric statistics using the configured `metrics` dictionary, and synchronizes
        the results across all processes. On the global rank 0, the metrics are saved as JSON files.

        Metrics are displayed in real-time using `tqdm` progress bars, showing a summary of the
        current batch's computed values.

        Args:
            world_size (int): Total number of distributed processes.
            global_rank (int): Global rank of the current process (used for writing results).
            gpu (int): Local GPU ID used for synchronization.
            dataloaders (list[DataLoader]): A list containing one or two DataLoaders:
                - `dataloaders[0]` is used for training evaluation.
                - `dataloaders[1]` (optional) is used for validation evaluation.

        Notes:
            - Only the main process (`global_rank == 0`) writes final results to disk.
        """

        self._evaluate_split(dataloaders[0], self.statistics_train, "TRAIN", world_size, gpu, global_rank)
        if len(dataloaders) == 2:
            self._evaluate_split(dataloaders[1], self.statistics_validation, "VALIDATION", world_size, gpu, global_rank)

    def _evaluate_split(
        self,
        dataloader: DataLoader,
        statistics: Statistics,
        label: str,
        world_size: int,
        gpu: int,
        global_rank: int,
    ) -> None:
        def description(measure):
            return (
                f"Metric {label} : {' | '.join(f'{k}: {v:.4f}' for k, v in measure.items())}"
                if measure is not None
                else f"Metric {label} : "
            )

        self._iter_dataset = dataloader.dataset
        try:
            with tqdm.tqdm(
                iterable=enumerate(dataloader),
                leave=True,
                desc=description(None),
                total=len(dataloader),
                ncols=0,
            ) as batch_iter:
                for _, batch_sample in batch_iter:
                    batch_iter.set_description(description(self.update(batch_sample, statistics)))
            self._flush_pending(statistics)  # close the split's last case
        except BaseException as error:
            # A half-written error map must not survive as a valid-looking file: abort the open
            # region-write sinks so their backends remove the partial entries, then re-raise.
            self._abort_map_sinks(error)
            raise
        outputs = synchronize_data(world_size, gpu, statistics.measures)
        if global_rank == 0:
            statistics.write(outputs)


def build_evaluate(
    evaluations_file: Path | str = Path("./Evaluation.yml").resolve(),
    evaluations_dir: Path | str = Path("./Evaluations").resolve(),
) -> DistributedObject:
    """
    Build and return the configured evaluation workflow without executing it.

    Parameters
    ----------
    evaluations_file : Path | str, optional
        Evaluation configuration file.
    evaluations_dir : Path | str, optional
        Directory where metrics and JSON reports are written.

    Returns
    -------
    DistributedObject
        Configured evaluator object ready to be executed by the runtime wrapper.
    """
    configure_workflow_environment(
        config_path=evaluations_file,
        root="Evaluator",
        state=State.EVALUATION,
        path_env={"KONFAI_EVALUATIONS_DIRECTORY": evaluations_dir},
    )
    os.environ["KONFAI_CONFIG_MODE"] = "Done"
    return apply_config()(Evaluator)()


@run_distributed_app
def evaluate(
    overwrite: bool = False,
    gpu: list[int] | None = cuda_visible_devices(),
    cpu: int = 1,
    quiet: bool = False,
    tensorboard: bool = False,
    evaluations_file: Path | str = Path("./Evaluation.yml").resolve(),
    evaluations_dir: Path | str = Path("./Evaluations").resolve(),
) -> DistributedObject:
    """
    Build and execute the configured evaluation workflow.

    This compatibility wrapper preserves the historical CLI-facing API while
    delegating the pure build step to :func:`build_evaluate`.
    """
    del overwrite, gpu, cpu, quiet, tensorboard
    return build_evaluate(
        evaluations_file=evaluations_file,
        evaluations_dir=evaluations_dir,
    )

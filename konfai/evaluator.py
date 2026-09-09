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
from typing import Any, cast

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from konfai import config_file, cuda_visible_devices, evaluations_directory, konfai_root
from konfai.data.data_manager import BatchDataItem, BatchSample, DataMetric, DatasetIter
from konfai.metric.measure.base import Criterion
from konfai.network.network import build_configured_criterions
from konfai.network.network.measure import CriterionResult
from konfai.utils.budget import node_local_ranks, set_per_rank_budget
from konfai.utils.clock import SweepClock
from konfai.utils.config import apply_config, config, strict_config
from konfai.utils.dataset import Attribute, Dataset, DataStream
from konfai.utils.errors import ConfigError, EvaluatorError
from konfai.utils.ome_zarr import bound_chunk_cache
from konfai.utils.runtime import (
    DistributedObject,
    State,
    clear_directory_except_logs,
    configure_workflow_environment,
    confirm_overwrite_or_raise,
    get_device,
    run_distributed_app,
    synchronize_data,
)
from konfai.utils.utils import split_path_spec


class CriterionsLoader:
    """Loader for the criterion modules applied between a model output and one or more targets.

    Evaluation criteria carry no per-criterion attributes: the value bound to each classpath is an
    unused placeholder (``None``).

    Args:
        criterions_loader (dict): Mapping from module classpaths to placeholder values.
    """

    def __init__(
        self,
        criterions_loader: dict[str, Any] = {"default|torch:nn:CrossEntropyLoss|Dice|NCC": None},
    ) -> None:
        self.criterions_loader = criterions_loader

    def get_criterions(self, output_group: str, target_group: str) -> dict[Criterion, Any]:
        # Scored through the Criterion contract (get_name, partial_metric); the builder types a bare Module.
        return cast(
            dict[Criterion, Any],
            build_configured_criterions(
                self.criterions_loader,
                f"{konfai_root()}.metrics.{output_group}.targets_criterions.{target_group}",
            ),
        )


class TargetCriterionsLoader:
    """Loader for the criterion configurations of several target groups, all linked to one model output.

    Args:
        targets_criterions (dict[str, CriterionsLoader]): Each target group name to its `CriterionsLoader`.
    """

    def __init__(
        self,
        targets_criterions: dict[str, CriterionsLoader] = {"default": CriterionsLoader()},
    ) -> None:
        self.targets_criterions = targets_criterions

    def get_targets_criterions(self, output_group: str) -> dict[str, dict[Criterion, Any]]:
        """The criterion modules and their placeholders for one output group, keyed by target group.

        Args:
            output_group (str): Name of the model output group (e.g., "output_segmentation").

        Returns:
            dict[str, dict[nn.Module, Any]]: target group name -> {loss module: placeholder}.
        """
        targets_criterions = {}
        for target_group, criterions_loader in self.targets_criterions.items():
            targets_criterions[target_group] = criterions_loader.get_criterions(output_group, target_group)
        return targets_criterions


class Statistics:
    """Accumulate per-case metric values, compute aggregates and write both as one JSON report.

    Args:
        filename (str): Path to the output JSON file.
    """

    def __init__(self, filename: Path) -> None:
        self.measures: dict[str, dict[str, float]] = {}
        self.filename = filename
        # Per-metric optimisation direction ("max"/"min"), from each criterion's `maximize` property.
        self.directions: dict[str, str] = {}
        self._incremental_path: Path | None = None

    def open_incremental(self, path: Path) -> None:
        """Append every case recorded from now on to ``path``, one JSON object per line, as it completes."""
        self._incremental_path = path

    def add(self, values: dict[str, float], name_dataset: str) -> None:
        """Add the metric values of one dataset case.

        Args:
            values (dict): Metric names to values.
            name_dataset (str): Case identifier.
        """
        for name, value in values.items():
            if name_dataset not in self.measures:
                self.measures[name_dataset] = {}
            self.measures[name_dataset][name] = value
        if self._incremental_path is not None and values:
            recorded = {name: float(value) if np.isfinite(value) else None for name, value in values.items()}
            with open(self._incremental_path, "a") as file:
                file.write(json.dumps({"name": name_dataset, "values": recorded}, allow_nan=False) + "\n")

    @staticmethod
    def load_incremental(paths: list[Path]) -> dict[str, dict[str, float]]:
        """The cases the JSONL files hold, last row per name winning; a truncated tail line is dropped,
        never an error. ``null`` reads back as NaN."""
        rows: dict[str, dict[str, float]] = {}
        for path in paths:
            try:
                lines = path.read_text().splitlines()
            except OSError:
                continue
            for line in lines:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or not isinstance(row.get("values"), dict):
                    continue
                rows[str(row["name"])] = {
                    key: (float("nan") if value is None else value) for key, value in row["values"].items()
                }
        return rows

    @staticmethod
    def get_statistic(values: list[float]) -> dict[str, float]:
        """Max, min, std, quartiles, mean and count of the non-NaN ``values``: all NaN when there is none."""
        array = np.asarray(values, dtype=float)
        count = int(np.count_nonzero(~np.isnan(array)))
        if count == 0:
            return dict.fromkeys(("max", "min", "std", "25pc", "50pc", "75pc", "mean"), np.nan) | {"count": 0.0}
        return {
            "max": float(np.nanmax(array)),
            "min": float(np.nanmin(array)),
            "std": float(np.nanstd(array)),
            "25pc": float(np.nanpercentile(array, 25)),
            "50pc": float(np.nanpercentile(array, 50)),
            "75pc": float(np.nanpercentile(array, 75)),
            "mean": float(np.nanmean(array)),
            "count": float(count),
        }

    @staticmethod
    def _to_serializable(obj: Any) -> Any:
        """Recursively replace non-finite floats with ``None``: NaN and infinities have no JSON representation.

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
        """Write the collected and aggregated statistics to the configured output file: ``case`` holds
        the per-sample metrics, ``aggregates`` the global statistics.

        Args:
            outputs (list): List of metric dictionaries to merge and serialize.
        """
        measures = {}
        for output in outputs:
            measures.update(output)
        # "case"/"aggregates" are nested dicts, "directions" maps metric name -> "max"/"min".
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

            # A dict-valued metric emits an aggregate entry ("output:target:Metric") and one per
            # component ("output:target:Metric:component"): keep only the top-level metrics.
            if key.rsplit(":", 1)[0] in aggregates:
                continue

            result[key] = mean_value

        return result


@config()
class Evaluator(DistributedObject):
    """Distributed evaluation engine: metrics on model predictions, multi-output and multi-target,
    aggregated over the training and validation datasets and synchronized across processes. Results
    are written as JSON.

    Args:
        train_name (str): Name of the evaluation run, used for logging and output folders.
        metrics (dict[str, TargetCriterionsLoader]): Output groups to loaders of target metrics.
        dataset (DataMetric): Dataset provider configured for evaluation mode.

    Attributes:
        statistics_train (Statistics): Training evaluation metrics.
        statistics_validation (Statistics): Validation evaluation metrics.
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
        # A memory budget patches the evaluation only when EVERY metric can rebuild its whole-case
        # value from partial states. A windowed metric declares its halo, and the widest one is read for all.
        criterions = [metric for targets in self.metrics.values() for group in targets.values() for metric in group]
        self.dataset.auto_patch_allowed = all(getattr(metric, "reducible", False) for metric in criterions)
        self.dataset.patch_halo = max((int(getattr(metric, "halo", 0)) for metric in criterions), default=0)
        self.dataset.prepare()
        set_per_rank_budget(self.dataset.resolved_budget().per_rank_bytes(node_local_ranks()))
        bound_chunk_cache()
        # Set iff the budget patched: batches carry one disjoint patch, accumulated until the case's last.
        self._streamed = self.dataset.patch is not None
        # The halo each patch is read with; a metric that declared it is told where the slot sits.
        self._halo = self.dataset.patch.halo if self.dataset.patch is not None else 0
        self._pending: dict[tuple[str, str, int], tuple] = {}
        self._pending_name: str | None = None
        self._last_result: dict[str, float] = {}
        #: Cases an interrupted run already scored: their batches are skipped, their rows reach the aggregate.
        self._scored_names: set[str] = set()
        # Per-voxel error maps under the patched path: one region-write sink per (metric, case).
        self._map_sinks: dict[tuple[str, str, int], DataStream] = {}
        self._iter_dataset: DatasetIter | None = None
        # The metrics' device; tensors arrive from the DataLoader on CPU, `run_process` sets the real device.
        self._device: torch.device | int = torch.device("cpu")
        # Where a split's wall clock goes, one clock per split (see _evaluate_split).
        self._clock = SweepClock()
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
        """Prepare the evaluator: check for previous results and overwrite or resume, create the output
        directory with a copy of the configuration, load the dataset for ``world_size`` processes.

        Args:
            world_size (int): Number of processes in the distributed evaluation setup.
        """
        # An interrupted run (case rows on disk, no aggregate) resumes without a prompt; a completed
        # run keeps the overwrite confirmation; --overwrite clears everything, case rows included.
        resumable = os.environ.get("KONFAI_OVERWRITE") != "True" and self._is_resumable()
        if not resumable and self.metric_path.exists() and len(list(self.metric_path.rglob("*.yml"))):
            confirm_overwrite_or_raise(self.metric_path, "metric", EvaluatorError)
            if self.metric_path.exists():
                # The directory holds the rank-0 log already open: clear around it, not rmtree.
                clear_directory_except_logs(self.metric_path)

        os.makedirs(self.metric_path, exist_ok=True)
        shutil.copyfile(
            config_file(),
            self.metric_path / config_file().name,
        )

        self.dataloader, _, _ = self.dataset.get_data(world_size)

    def _incremental_case_files(self, statistics: Statistics) -> list[Path]:
        """Every rank's case file for this split, this run's and an interrupted predecessor's alike."""
        return sorted(self.metric_path.glob(f"{statistics.filename.stem}.cases.*.jsonl"))

    def _is_resumable(self) -> bool:
        """Whether an interrupted run left case rows without their aggregate for some split."""
        return any(
            self._incremental_case_files(statistics) and not statistics.filename.exists()
            for statistics in (self.statistics_train, self.statistics_validation)
        )

    def update(self, batch_sample: BatchSample, statistics: Statistics) -> dict[str, float]:
        """Compute the metrics of one batch and update the running statistics.

        Args:
            batch_sample (BatchSample): Tensors and their metadata.
            statistics (Statistics): The statistics object to update (train or validation).

        Returns:
            dict[str, float]: Metric values keyed 'output_group:target_group:MetricName'.
        """
        if self._streamed:
            return self._update_streamed(batch_sample, statistics)
        if self._scored_names and len(self.metrics):
            name = batch_sample[next(iter(self.metrics))].name[0]
            if name in self._scored_names:
                return statistics.measures.get(name, {})
        result: dict[str, float] = {}
        moved = self._groups_on(batch_sample)
        for output_group in self.metrics:
            output_tensor = moved[output_group]
            for target_group in self.metrics[output_group]:
                targets = [moved[group] for group in target_group.split(";") if group in batch_sample]
                target_attribute = [batch_sample[output_group].attribute] + [
                    batch_sample[group].attribute for group in target_group.split(";") if group in batch_sample
                ]
                name = batch_sample[output_group].name[0]
                for metric in self.metrics[output_group][target_group]:
                    metric_name: str = metric.get_name()
                    with self._clock.phase(metric_name), torch.no_grad():
                        if getattr(metric, "accepts_attributes", False):
                            loss = metric(output_tensor, *targets, attributes=target_attribute)
                        else:
                            loss = metric(output_tensor, *targets)
                    outcome = CriterionResult.of(loss, metric_name)
                    true_loss = outcome.materialized()
                    if outcome.map is not None and getattr(metric, "dataset", None):
                        with self._clock.phase("map"):
                            self._write_map(metric, output_group, name, outcome.map)

                    direction = "max" if getattr(metric, "maximize", False) else "min"
                    base_key = f"{output_group}:{target_group}:{metric_name}"
                    Evaluator._record_value(result, statistics, base_key, true_loss, direction)
        if len(self.metrics) > 0:
            statistics.add(result, name)
        return result

    def _write_map(self, metric: Any, output_group: str, name: str, map_: torch.Tensor) -> None:
        """Write a metric's whole-case map beside the case, with the case's own geometry."""
        filename, _, file_format = split_path_spec(metric.dataset)
        map_dataset = Dataset(filename, file_format)
        group = metric.group if metric.group else output_group
        for dataset in self.dataset.datasets.values():
            for g in dataset.get_group():
                if dataset.is_dataset_exist(g, name):
                    _, cache_attribute = dataset.get_infos(g, name)
                    map_dataset.write(group, name, map_.squeeze(0).numpy(), cache_attribute)
                    return

    @staticmethod
    def _on(tensor: torch.Tensor, metric_device: torch.device | int) -> torch.Tensor:
        """A tensor on the metric's device, moved only when it is not already there."""
        if tensor.device == torch.device(metric_device):
            return tensor
        return tensor.to(metric_device, non_blocking=tensor.device.type == "cpu")

    def _groups_on(self, batch_sample: BatchSample) -> dict[str, torch.Tensor]:
        """Every group a metric names, on the metric device, moved once per update. Sharing one copy is
        safe: a metric never writes into its inputs."""
        groups = {
            group
            for output_group, targets in self.metrics.items()
            for spec in (output_group, *targets)
            for group in spec.split(";")
            if group in batch_sample
        }
        with self._clock.phase("h2d"):
            return {group: self._on(batch_sample[group].tensor, self._device) for group in sorted(groups)}

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
        """Accumulate one PATCH's partial states; record the case when its next sibling arrives. Cases
        shard whole per rank and their patches arrive contiguously, so a change of case name completes
        the previous one; ``_flush_pending`` closes the split's last case."""
        name = batch_sample[next(iter(self.metrics))].name[0]
        if name in self._scored_names:
            return self._last_result
        if self._pending_name is not None and name != self._pending_name:
            with self._clock.phase("flush"):
                self._flush_pending(statistics)
        self._pending_name = name
        moved = self._groups_on(batch_sample)
        for output_group in self.metrics:
            output_tensor = moved[output_group]
            core = self._core_in_read(output_group, batch_sample[output_group])
            for target_group in self.metrics[output_group]:
                targets = [moved[group] for group in target_group.split(";") if group in batch_sample]
                tensors = [output_tensor, *targets]
                cored = tensors if core is None else [t[(slice(None), slice(None), *core)] for t in tensors]
                for index, metric in enumerate(self.metrics[output_group][target_group]):
                    reads_halo = core is not None and int(getattr(metric, "halo", 0)) > 0
                    with self._clock.phase(metric.get_name()), torch.no_grad():
                        # Criterion declares neither the halo's core= nor partial_map: the reducible hooks.
                        state = (
                            cast(Any, metric).partial_metric(*tensors, core=core)
                            if reads_halo
                            else metric.partial_metric(*cored)
                        )
                    entry = self._pending.setdefault((output_group, target_group, index), (metric, []))
                    entry[1].append(state)
                    if getattr(metric, "dataset", None) and hasattr(metric, "partial_map"):
                        with self._clock.phase(metric.get_name()), torch.no_grad():
                            patch_map = cast(Any, metric).partial_map(*cored).squeeze(0)
                        with self._clock.phase("map"):
                            self._write_map_patch(
                                (output_group, target_group, index),
                                metric,
                                batch_sample[output_group],
                                output_group,
                                patch_map,
                            )
        return self._last_result

    def _manager(self, output_group: str, item: BatchDataItem):
        """The manager whose grid cut the patch ``item`` carries."""
        if self._iter_dataset is None:
            raise EvaluatorError("Internal error: the streamed evaluation loop has no dataset iterator.")
        return self._iter_dataset.get_dataset_from_index(output_group, int(item.x[0]))

    def _core_in_read(self, output_group: str, item: BatchDataItem) -> tuple[slice, ...] | None:
        """Where the patch's slot sits within its tensors; ``None`` when the grid reads no halo."""
        if not self._halo:
            return None
        return self._manager(output_group, item).patch.core_in_read(int(item.a[0]), int(item.p[0]))

    def _write_map_patch(
        self,
        key: tuple[str, str, int],
        metric: Any,
        item: BatchDataItem,
        output_group: str,
        patch_map: torch.Tensor,
    ) -> None:
        """Write one patch's per-voxel map into its case's region-write sink: ``partial_map`` is
        voxel-local and the disjoint unpadded grid writes every voxel once."""
        manager = self._manager(output_group, item)
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
            true_loss = CriterionResult.of(metric.combine_metric(states), metric.get_name()).materialized()
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
        """Run the evaluation loop over the training and validation datasets, synchronize the results
        across processes and, on rank 0, write the JSON files.

        Args:
            world_size (int): Total number of distributed processes.
            global_rank (int): Global rank of the current process.
            gpu (int): Local GPU ID used for synchronization.
            dataloaders (list[DataLoader]): ``[train]`` or ``[train, validation]``.
        """

        self._device = get_device(gpu) if len(cuda_visible_devices()) else torch.device("cpu")
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

        self._iter_dataset = cast(DatasetIter, dataloader.dataset)
        self._clock = SweepClock()
        # Cases an interrupted run scored are read back and skipped; each case scored from here on is
        # appended to this rank's own case file. The aggregate is built from the union.
        scored = Statistics.load_incremental(self._incremental_case_files(statistics))
        self._scored_names = set(scored)
        if scored:
            statistics.measures.update(scored)
            if global_rank == 0:
                print(
                    f"[KonfAI] evaluation {label}: {len(scored)} case(s) already scored ->"
                    " skipped (--overwrite recomputes)."
                )
        statistics.open_incremental(self.metric_path / f"{statistics.filename.stem}.cases.rank{global_rank}.jsonl")
        try:
            with (
                self._clock.phase("split"),
                tqdm.tqdm(
                    iterable=enumerate(dataloader),
                    leave=True,
                    desc=description(None),
                    total=len(dataloader),
                    ncols=0,
                ) as batch_iter,
            ):
                for _, batch_sample in self._clock.waiting("wait(load)", batch_iter):
                    batch_iter.set_description(description(self.update(batch_sample, statistics)))
                with self._clock.phase("flush"):
                    self._flush_pending(statistics)  # close the split's last case
        except BaseException as error:
            # Abort the open region-write sinks so their backends remove the partial entries, then re-raise.
            self._abort_map_sinks(error)
            raise
        if global_rank == 0:
            report = self._clock_report(label)
            if report is not None:
                print(report)
        outputs = synchronize_data(world_size, gpu, statistics.measures)
        if global_rank == 0:
            statistics.write(outputs)

    def _clock_report(self, label: str, min_seconds: float = 1.0) -> str | None:
        """One line accounting for a split's wall clock, or ``None`` below ``min_seconds``: the wait for
        the loader, the move to the metric device, one phase per metric name, the map writes and the
        streamed flushes; ``other`` is what none of them names. On a GPU a metric's phase is its enqueue."""
        wall = self._clock.spent("split")
        if wall < min_seconds:
            return None
        names = ["wait(load)", "h2d"]
        names += sorted(
            {
                metric.get_name()
                for targets in self.metrics.values()
                for metrics in targets.values()
                for metric in metrics
            }
        )
        names += ["map", "flush"]
        named = {name: self._clock.spent(name) for name in names if self._clock.spent(name) > 0.0}
        parts = " + ".join(f"{name} {value:.1f}" for name, value in named.items())
        return f"[KonfAI] evaluation {label} {wall:.1f} s = {parts} + other {wall - sum(named.values()):.1f}"


def build_evaluate(
    evaluations_file: Path | str | dict = Path("./Evaluation.yml"),
    evaluations_dir: Path | str = Path("./Evaluations"),
) -> DistributedObject:
    """Build and return the configured evaluation workflow without executing it.

    Parameters
    ----------
    evaluations_file : Path | str, optional
        Evaluation configuration file.
    evaluations_dir : Path | str, optional
        Directory where metrics and JSON reports are written.

    Returns
    -------
    DistributedObject
        Configured evaluator, executed by the runtime wrapper.
    """
    configure_workflow_environment(
        config_path=evaluations_file,
        root="Evaluator",
        state=State.EVALUATION,
        path_env={"KONFAI_EVALUATIONS_DIRECTORY": evaluations_dir},
    )
    os.environ["KONFAI_CONFIG_MODE"] = "Done"
    with strict_config("Evaluator", refuse=False):
        return apply_config()(Evaluator)()


@run_distributed_app
def evaluate(
    overwrite: bool = False,
    gpu: list[int] | None = None,
    cpu: int = 1,
    quiet: bool = False,
    tensorboard: bool = False,
    evaluations_file: Path | str | dict = Path("./Evaluation.yml"),
    evaluations_dir: Path | str = Path("./Evaluations"),
) -> DistributedObject:
    """Build and execute the configured evaluation workflow.

    ``overwrite``/``gpu``/``cpu``/``quiet``/``tensorboard`` are read by :func:`run_distributed_app`
    from the bound signature; the body drops them. The pure build step is :func:`build_evaluate`.
    """
    del overwrite, gpu, cpu, quiet, tensorboard
    return build_evaluate(
        evaluations_file=evaluations_file,
        evaluations_dir=evaluations_dir,
    )

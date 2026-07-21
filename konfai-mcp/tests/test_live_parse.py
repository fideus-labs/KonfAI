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

"""The shared live-log parser — one source of truth for Studio's feed and read_live_metrics."""

from konfai_mcp.live_parse import parse_host_stats, parse_live_metric_line, parse_live_progress

TRAIN_LINE = (
    "Training : Loss (UNetpp5(0.001000) : MAE(1.00) : 0.012225 SAM_Perceptual(1.00) : 0.080258) "
    "GPU([0]) Memory GPU (19.55G (76.22 %)) | Memory (33.96G (27.60 %)):  16% 335/2130 "
    "[19:06<1:03:43,  2.13s/it]"
)
CACHING_LINE = (
    "Caching Train: Memory (18.62G (15.10 %)) | Memory forecast (114.78G (93.34 %)) | "
    "CPU (23.60 %):   5%|5         | 3/60 [00:01<00:31,  1.81it/s]"
)
EVAL_LINE = (
    "Metric TRAIN : PRED:SEG:Dice: 0.9123 | PRED:SEG:HD95: 3.4210:  50%|#####     | 5/10 [00:04<00:04,  1.30it/s]"
)


def test_training_line_parses_metrics_lr_and_both_memories() -> None:
    entry = parse_live_metric_line(TRAIN_LINE)
    assert entry is not None
    assert entry["stage"] == "Training"
    assert entry["flat_metrics"]["UNetpp5:MAE"] == 0.012225
    assert entry["flat_metrics"]["UNetpp5:SAM_Perceptual"] == 0.080258
    assert entry["models"][0]["lr"] == 0.001
    # GPU and process memory are distinct readouts on the same line — both must be picked up.
    assert entry["memory_gpu_gb"] == 19.55
    assert entry["memory_gpu_percent"] == 76.22
    assert entry["memory_gb"] == 33.96
    assert entry["progress"]["step"] == 335
    assert entry["progress"]["total"] == 2130
    assert entry["progress"]["rate_unit"] == "s/it"


def test_evaluation_line_becomes_live_metrics() -> None:
    entry = parse_live_metric_line(EVAL_LINE)
    assert entry is not None
    assert entry["stage"] == "Evaluation"
    assert entry["label"] == "Metric TRAIN"
    # Keys keep their output:target:Name identity even though ':' also separates the pair from its value.
    assert entry["flat_metrics"]["PRED:SEG:Dice"] == 0.9123
    assert entry["flat_metrics"]["PRED:SEG:HD95"] == 3.4210
    assert entry["progress"]["step"] == 5


def test_evaluation_split_case_is_normalised() -> None:
    entry = parse_live_metric_line("Metric VALIDATION : A:B:Dice: 0.5:  10%|#| 1/10 [00:00<00:09,  1.0it/s]")
    assert entry is not None
    assert entry["label"] == "Metric VALIDATION"
    assert entry["flat_metrics"]["A:B:Dice"] == 0.5


def test_caching_line_has_no_metrics_only_progress_and_host() -> None:
    # A metric-less phase: the metric parser declines, but progress + host stats still read.
    assert parse_live_metric_line(CACHING_LINE) is None
    progress = parse_live_progress(CACHING_LINE)
    assert progress is not None and progress["step"] == 3 and progress["total"] == 60
    host = parse_host_stats(CACHING_LINE)
    assert host["memory_gb"] == 18.62
    assert host["cpu_percent"] == 23.60
    assert "memory_gpu_gb" not in host  # no GPU readout on a caching line


def test_process_memory_not_confused_with_gpu_memory() -> None:
    host = parse_host_stats("… Memory GPU (19.55G (76.22 %)) | Memory (33.96G (27.60 %)) …")
    assert host["memory_gpu_gb"] == 19.55
    assert host["memory_gb"] == 33.96


def test_blank_and_unrelated_lines_return_none() -> None:
    assert parse_live_metric_line("") is None
    assert parse_live_metric_line("[konfai-mcp] job started") is None

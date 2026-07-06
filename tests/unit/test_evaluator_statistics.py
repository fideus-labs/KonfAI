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

"""Aggregation, serialization and summary behaviour of :class:`Statistics`."""

import json

import numpy as np
import pytest
from konfai.evaluator import Statistics


class TestGetStatisticCount:
    def test_count_is_zero_when_every_value_is_nan(self):
        """An all-NaN series must report a count of 0, not NaN."""
        stats = Statistics.get_statistic([np.nan, np.nan])

        assert stats["count"] == 0.0
        assert np.isnan(stats["mean"])
        assert np.isnan(stats["max"])
        assert np.isnan(stats["min"])

    def test_count_ignores_nan_entries(self):
        """count aggregates only the finite entries of the series."""
        stats = Statistics.get_statistic([1.0, np.nan, 3.0, 5.0])

        assert stats["count"] == 3.0
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["max"] == pytest.approx(5.0)
        assert stats["min"] == pytest.approx(1.0)
        # Population std (ddof=0) of [1, 3, 5] = sqrt(8/3).
        assert stats["std"] == pytest.approx(np.sqrt(8.0 / 3.0))
        assert stats["25pc"] == pytest.approx(2.0)
        assert stats["50pc"] == pytest.approx(3.0)
        assert stats["75pc"] == pytest.approx(4.0)


class TestWriteJsonValidity:
    def test_write_emits_standard_json_for_non_finite_values(self, tmp_path):
        """NaN/Infinity must serialize as JSON null, never as bare NaN/Infinity."""
        statistics = Statistics(tmp_path / "Metric.json")
        statistics.write(
            [
                {
                    "case1": {
                        "PRED:SEG:Dice": float("nan"),
                        "PRED:SEG:PSNR": float("inf"),
                        "PRED:SEG:MAE": 0.5,
                    }
                }
            ]
        )

        content = (tmp_path / "Metric.json").read_text()
        # Standard JSON has no NaN/Infinity literals.
        assert "NaN" not in content
        assert "Infinity" not in content

        parsed = json.loads(content)
        # Non-finite per-case values become null.
        assert parsed["case"]["PRED:SEG:Dice"]["case1"] is None
        assert parsed["case"]["PRED:SEG:PSNR"]["case1"] is None
        assert parsed["case"]["PRED:SEG:MAE"]["case1"] == 0.5

        # Aggregates: an all-NaN metric keeps a finite count of 0 with a null mean.
        assert parsed["aggregates"]["PRED:SEG:Dice"]["count"] == 0.0
        assert parsed["aggregates"]["PRED:SEG:Dice"]["mean"] is None
        assert parsed["aggregates"]["PRED:SEG:MAE"]["mean"] == 0.5


class TestReadSummary:
    def _write_aggregates(self, path, aggregates):
        path.write_text(json.dumps({"case": {}, "aggregates": aggregates}))

    def test_read_keeps_distinct_output_groups(self, tmp_path):
        """Two output groups sharing a metric name must not overwrite each other."""
        path = tmp_path / "Metric.json"
        self._write_aggregates(
            path,
            {
                "PRED1:SEG:Dice": {"mean": 0.80, "count": 3.0},
                "PRED2:SEG:Dice": {"mean": 0.60, "count": 3.0},
            },
        )

        summary = Statistics(path).read()

        assert summary["PRED1:SEG:Dice"] == pytest.approx(0.80)
        assert summary["PRED2:SEG:Dice"] == pytest.approx(0.60)
        assert len(summary) == 2

    def test_read_drops_dict_metric_components(self, tmp_path):
        """Per-label / per-landmark sub-entries must be dropped for every metric."""
        path = tmp_path / "Metric.json"
        self._write_aggregates(
            path,
            {
                "PRED:SEG:Dice": {"mean": 0.80, "count": 3.0},
                "PRED:SEG:Dice:1": {"mean": 0.75, "count": 3.0},
                "PRED:SEG:Dice:2": {"mean": 0.85, "count": 3.0},
                "OUT:TGT:TRE": {"mean": 2.0, "count": 3.0},
                "OUT:TGT:TRE:Landmarks_0": {"mean": 1.5, "count": 3.0},
            },
        )

        summary = Statistics(path).read()

        assert summary == {
            "PRED:SEG:Dice": pytest.approx(0.80),
            "OUT:TGT:TRE": pytest.approx(2.0),
        }
        # The TRE component sub-key must not leak into the summary.
        assert not any("Landmarks" in key for key in summary)
        assert "PRED:SEG:Dice:1" not in summary

    def test_read_skips_metrics_without_mean(self, tmp_path):
        """A metric whose mean is null (formerly NaN) is excluded from the summary."""
        path = tmp_path / "Metric.json"
        self._write_aggregates(
            path,
            {
                "PRED:SEG:Dice": {"mean": None, "count": 0.0},
                "PRED:SEG:MAE": {"mean": 0.5, "count": 3.0},
            },
        )

        summary = Statistics(path).read()

        assert summary == {"PRED:SEG:MAE": pytest.approx(0.5)}

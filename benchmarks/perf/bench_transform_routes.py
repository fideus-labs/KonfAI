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

"""TRANSFORM's other routes on the shipped Transform example's cohort: a fold, copies, the whole volume.

    python benchmarks/perf/bench_transform_routes.py [--repeats 3] [--force] [--quick]

``examples/Transform/make_dataset.py`` writes six cases that share no grid. Three chains run on a
scratch copy through the CLI, each in a fresh process: ``Transform.yml`` (Clip, Resample onto one
case's grid, a Median ``Reduce`` into one template), ``Transform_expand.yml`` (Clip, ``Expand`` into
four copies, a ``Brightness`` draw), and a chain whose one stage declares no locality, which the plan
routes whole-volume. A time counts only when the run's ``[KonfAI] done`` line reports the route the
chain is built for and its outputs exist. ``bench_transform.py`` keeps the streamed route against a
plain loop.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from harness import copy_example, fingerprint, konfai_executable, machine_gate, run_cli, write_result

_DONE = re.compile(
    r"done in (?P<wall>[0-9.]+) s: (?P<written>\d+) written \((?P<streamed>\d+) streamed, (?P<loaded>\d+) loaded,"
    r" (?P<whole>\d+) whole-volume, (?P<reduced>\d+) reduced\)"
)

_WHOLE_STAGE = '''from konfai.data.transform import Transform


class Offset(Transform):
    """One added to every voxel, and no locality declared: the plan routes it whole-volume."""

    def __call__(self, name, tensor, cache_attribute):
        return tensor + 1

    def transform_shape(self, group_src, name, shape, cache_attribute):
        return shape
'''

_WHOLE_CHAIN = """Transformer:
  name: WHOLE
  on_fallback: allow
  Dataset:
    dataset_filenames:
      - ./Raw:mha
    memory_budget: 2G
    groups_src:
      CT:
        groups_dest:
          CT_whole:
            transforms:
              WholeVolume:Offset: {}
              Write: {dataset: ./Whole:mha}
"""

#: Each route: its config, the output directory, and what its done line must report.
ROUTES = {
    "fold": ("Transform.yml", "Template", lambda done: done["reduced"] >= 1),
    "copies": ("Transform_expand.yml", "Augmented", lambda done: done["written"] == 24 and done["whole"] == 0),
    "whole_volume": ("Transform_whole.yml", "Whole", lambda done: done["whole"] >= 6),
}


def done_line(output: str) -> dict[str, float] | None:
    match = _DONE.search(output)
    return {key: float(value) for key, value in match.groupdict().items()} if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    gate = machine_gate(force=args.force)
    fp = fingerprint()
    repeats = 1 if args.quick else args.repeats
    scratch = Path(tempfile.mkdtemp(prefix="konfai_perf_routes_"))
    copy = copy_example("Transform", scratch)
    subprocess.run([sys.executable, "make_dataset.py"], cwd=copy, check=True, capture_output=True)
    (copy / "WholeVolume.py").write_text(_WHOLE_STAGE)
    (copy / "Transform_whole.yml").write_text(_WHOLE_CHAIN)
    pristine = {config: (copy / config).read_text() for config, _, _ in ROUTES.values()}

    result: dict[str, object] = {"gate_warnings": gate.warnings, "repeats": repeats, "scratch": str(scratch)}
    metrics: dict[str, float] = {}
    for route, (config, out, expected) in ROUTES.items():
        walls, peaks = [], []
        for repeat in range(repeats):
            (copy / config).write_text(pristine[config])  # reading a config rewrites it
            shutil.rmtree(copy / out, ignore_errors=True)  # an output that exists is skipped: a resumed run
            argv = [*konfai_executable(), "TRANSFORM", "-c", config, "-y", "--transforms-dir", str(scratch / "Tr")]
            command = run_cli(argv, cwd=copy, log_path=scratch / f"{route}_{repeat}.log")
            done = done_line(command.output)
            outputs = list((copy / out).rglob("*.mha"))
            if command.returncode != 0 or done is None or not expected(done) or not outputs:
                result["failure"] = f"{route}: rc {command.returncode}, done {done}, {len(outputs)} output(s)"
                write_result("transform_routes", result, fp=fp)
                raise SystemExit(f"[perf] {result['failure']} (log {scratch / f'{route}_{repeat}.log'})")
            walls.append(command.wall_s)
            peaks.append(command.peak_rss_gib)
            result[f"{route}_{repeat}"] = {
                "done": done,
                "sweep": command.clocks.get("sweep", {}),
                "outputs": len(outputs),
            }
        metrics[f"wall_s_{route}"] = round(statistics.median(walls), 3)
        metrics[f"peak_rss_gib_{route}"] = max(peaks)

    result["metrics"] = metrics
    result["headline"] = " | ".join(
        f"{route} {metrics[f'wall_s_{route}']} s, {metrics[f'peak_rss_gib_{route}']} GiB" for route in ROUTES
    )
    path = write_result("transform_routes", result, fp=fp)
    print(json.dumps(metrics, indent=2))
    print(f"[perf] {result['headline']}")
    print(f"[perf] written {path}")


if __name__ == "__main__":
    main()

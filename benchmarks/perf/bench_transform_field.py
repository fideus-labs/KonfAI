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

"""A case resampled through a dense displacement field, streamed, against the budget it was given.

    python benchmarks/perf/bench_transform_field.py [--repeats 3] [--force] [--quick]

One synthetic case and a smooth float32 field on its grid, the registration warp IMPACT-Reg runs.
The plan prices the field as no displacement and the run sizes each region from the field it reads,
so this is the route whose memory the plan does not see. Each budget runs TRANSFORM in a fresh
process: what the run held above the process floor, as KonfAI reports it, must stay under the
budget, and the output must equal SimpleITK's whole-volume resample through the same field.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from facts import fact
from harness import fingerprint, konfai_executable, machine_gate, run_cli, write_result

SHAPE = (192, 256, 256)  # (z, y, x)
BUDGETS = ("256M", "512M", "1G")
_HELD = re.compile(r"held (?P<value>[0-9.]+) (?P<unit>[KMG]iB) above the process floor")

_CHAIN = """Transformer:
  name: FIELD_{tag}
  Dataset:
    dataset_filenames:
      - ./Raw:mha
    memory_budget: {budget}
    groups_src:
      CT:
        groups_dest:
          CT_moved:
            transforms:
              Resample: {{reference: '{{case}}', reference_group: CT, field: ./Fields:mha, field_group: DVF}}
              Write: {{dataset: ./Moved_{tag}:mha}}
"""


def _budget_bytes(budget: str) -> int:
    # KonfAI's units: '512M' is 512e6 bytes.
    return int(float(budget[:-1]) * {"M": 1e6, "G": 1e9}[budget[-1]])


def _held_bytes(output: str) -> int | None:
    match = _HELD.search(output)
    return None if match is None else int(float(match["value"]) * 2 ** {"KiB": 10, "MiB": 20, "GiB": 30}[match["unit"]])


def _write_case(root: Path) -> tuple[sitk.Image, sitk.Image]:
    """A textured volume and a smooth field of up to five voxels, both on one grid."""
    rng = np.random.default_rng(0)
    z, y, x = np.meshgrid(*[np.linspace(0.0, np.pi, n, dtype=np.float32) for n in SHAPE], indexing="ij")
    volume = (np.sin(4 * x) * np.cos(3 * y) * np.sin(2 * z) * 500 + rng.normal(0, 20, SHAPE)).astype(np.float32)
    field = np.stack([5 * np.sin(z) * np.cos(y), 4 * np.sin(x) * np.sin(z), 3 * np.cos(x) * np.sin(y)], axis=-1)
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing((1.0, 1.0, 1.5))
    displacement = sitk.GetImageFromArray(field.astype(np.float32), isVector=True)
    displacement.CopyInformation(image)
    for group in ("Raw", "Fields"):
        (root / group / "CASE_000").mkdir(parents=True)
    sitk.WriteImage(image, str(root / "Raw" / "CASE_000" / "CT.mha"))
    sitk.WriteImage(displacement, str(root / "Fields" / "CASE_000" / "DVF.mha"))
    return image, displacement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    gate = machine_gate(force=args.force)
    fp = fingerprint()
    repeats = 1 if args.quick else args.repeats
    scratch = Path(tempfile.mkdtemp(prefix="konfai_perf_field_"))
    image, displacement = _write_case(scratch)
    transform = sitk.DisplacementFieldTransform(sitk.Cast(displacement, sitk.sitkVectorFloat64))
    reference = sitk.GetArrayFromImage(sitk.Resample(image, image, transform, sitk.sitkLinear, 0.0))

    result: dict[str, object] = {"gate_warnings": gate.warnings, "repeats": repeats, "shape": SHAPE}
    metrics: dict[str, float] = {}
    facts = []
    for budget in BUDGETS:
        tag = budget.lower()
        config = scratch / f"Field_{tag}.yml"
        walls, peaks, helds = [], [], []
        for repeat in range(repeats):
            config.write_text(_CHAIN.format(tag=tag, budget=budget))  # reading a config rewrites it
            shutil.rmtree(scratch / f"Moved_{tag}", ignore_errors=True)
            argv = [*konfai_executable(), "TRANSFORM", "-c", config.name, "-y", "--transforms-dir", str(scratch / "Tr")]
            command = run_cli(argv, cwd=scratch, log_path=scratch / f"{tag}_{repeat}.log")
            written = list((scratch / f"Moved_{tag}").rglob("*.mha"))
            held = _held_bytes(command.output)
            if command.returncode != 0 or not written or held is None:
                result["failure"] = f"{budget}: rc {command.returncode}, {len(written)} output(s)"
                write_result("transform_field", result, fp=fp)
                raise SystemExit(f"[perf] {result['failure']} (log {scratch / f'{tag}_{repeat}.log'})")
            walls.append(command.wall_s)
            peaks.append(command.peak_rss_bytes)
            helds.append(held)
        moved = sitk.GetArrayFromImage(sitk.ReadImage(str(written[0])))
        difference = float(np.abs(moved - reference).max())
        metrics[f"wall_s_{tag}"] = round(statistics.median(walls), 3)
        metrics[f"peak_rss_gib_{tag}"] = round(max(peaks) / 2**30, 3)
        metrics[f"held_gib_{tag}"] = round(max(helds) / 2**30, 3)
        metrics[f"max_abs_diff_{tag}"] = difference
        facts.append(fact(f"field_held_within_budget_{tag}", max(helds), _budget_bytes(budget), "<="))
        facts.append(fact(f"field_equals_whole_volume_{tag}", difference, 1e-3, "<="))

    result["metrics"] = metrics
    result["facts"] = facts
    result["headline"] = " | ".join(
        f"{budget}: {metrics[f'wall_s_{budget.lower()}']} s, held {metrics[f'held_gib_{budget.lower()}']} GiB,"
        f" max diff {metrics[f'max_abs_diff_{budget.lower()}']:.2g}"
        for budget in BUDGETS
    )
    path = write_result("transform_field", result, fp=fp)
    print(json.dumps(metrics, indent=2))
    print(f"[perf] {result['headline']}")
    print(f"[perf] written {path}")


if __name__ == "__main__":
    main()

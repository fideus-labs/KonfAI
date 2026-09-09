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

"""TRAIN on the shipped Segmentation example: where an epoch goes, by the trainer's own clocks.

    python benchmarks/perf/bench_train_epoch.py [--epochs 2] [--force] [--quick] [--cprofile] [--variants fp32,autocast]

The CLI runs on a scratch copy of ``examples/Segmentation`` whose ``Config.yml`` is rewritten to the
requested epoch count and, per variant, ``autocast``. Every ``[KonfAI] epoch`` line is parsed (one per
epoch: the first pays the cache fill, the last is the steady state) and the ``[KonfAI] startup`` line
too; the process tree's peak RSS and the GPU's memory delta are sampled from outside. A run counts
only if it exits 0, writes a checkpoint and prints a finite loss. ``--cprofile`` runs the child under
``cProfile`` once (a diagnostic: it slows the host side) and reports the share of the data path and
of the criteria in host time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

from harness import (
    CliRun,
    copy_example,
    cprofile_summary,
    fingerprint,
    konfai_executable,
    machine_gate,
    run_cli,
    write_result,
)

_EPOCH = re.compile(r"\[KonfAI\] epoch (?P<wall>[0-9.]+) s = (?P<parts>.*)")
_PART = re.compile(r"(?P<phase>[a-z()+]+) (?P<value>[0-9.]+)")
_LOSS = re.compile(r"(?:loss|Loss)[^0-9\-]*(-?[0-9]+\.[0-9]+(?:e-?[0-9]+)?)")


def rewrite_config(path: Path, *, epochs: int, autocast: bool) -> None:
    text = path.read_text()
    text, n_epochs = re.subn(r"^(\s*epochs:\s*)\d+", rf"\g<1>{epochs}", text, count=1, flags=re.M)
    text, n_autocast = re.subn(
        r"^(\s*autocast:\s*)\w+", rf"\g<1>{'true' if autocast else 'false'}", text, count=1, flags=re.M
    )
    if not n_epochs or not n_autocast:
        raise SystemExit(f"[perf] could not rewrite epochs/autocast in {path}")
    path.write_text(text)


def epoch_lines(output: str) -> list[dict[str, float]]:
    epochs = []
    for line in output.splitlines():
        match = _EPOCH.search(line)
        if not match:
            continue
        phases = {"wall": float(match.group("wall"))}
        for part in _PART.finditer(match.group("parts")):
            phases[part.group("phase")] = float(part.group("value"))
        epochs.append(phases)
    return epochs


def last_loss(output: str) -> float | None:
    values = _LOSS.findall(output)
    return float(values[-1]) if values else None


def train_once(copy: Path, scratch: Path, variant: str, *, gpu: bool, cprofile: bool) -> CliRun:
    args = [
        "TRAIN",
        "-c",
        "Config.yml",
        "-y",
        "--checkpoints-dir",
        str(scratch / f"ck_{variant}"),
        "--statistics-dir",
        str(scratch / f"st_{variant}"),
    ]
    if gpu:
        args += ["--gpu", "0"]
    if cprofile:
        argv = [
            sys.executable,
            "-m",
            "cProfile",
            "-o",
            str(scratch / f"train_{variant}.prof"),
            "-m",
            "konfai.main",
            *args,
        ]
    else:
        argv = [*konfai_executable(), *args]
    # The runtime pins its own threads (rank_cpu_share); an inherited OMP_NUM_THREADS would override it.
    env = {"OMP_NUM_THREADS": ""}
    return run_cli(argv, cwd=copy, env=env, log_path=scratch / f"train_{variant}.log")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--variants", default="fp32,autocast")
    parser.add_argument("--repeats", type=int, default=1, help="epochs are the repeats here; keep 1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    gate = machine_gate(force=args.force)
    fp = fingerprint()

    import os

    import torch

    os.environ.pop("OMP_NUM_THREADS", None)
    gpu = torch.cuda.is_available() and not args.cpu
    epochs = 1 if args.quick else args.epochs
    variants = ["fp32"] if args.quick else [v.strip() for v in args.variants.split(",") if v.strip()]
    scratch = Path(tempfile.mkdtemp(prefix="konfai_perf_train_"))

    result: dict[str, object] = {"gate_warnings": gate.warnings, "epochs": epochs, "gpu": gpu, "scratch": str(scratch)}
    metrics: dict[str, float] = {}
    for variant in variants:
        copy = copy_example("Segmentation", scratch / variant)
        rewrite_config(copy / "Config.yml", epochs=epochs, autocast=variant == "autocast")
        run = train_once(copy, scratch, variant, gpu=gpu, cprofile=False)
        lines = epoch_lines(run.output)
        checkpoints = (
            sorted((scratch / f"ck_{variant}" / "SEG_BASELINE").glob("*.pt"))
            if (scratch / f"ck_{variant}").exists()
            else []
        )
        loss = last_loss(run.output)
        ok = run.returncode == 0 and bool(checkpoints) and loss is not None and loss == loss
        result[f"run_{variant}"] = {
            "returncode": run.returncode,
            "wall_s": run.wall_s,
            "epoch_lines": lines,
            "startup": run.clocks.get("startup", {}),
            "checkpoints": [c.name for c in checkpoints],
            "last_loss": loss,
            "valid": ok,
            "log": str(scratch / f"train_{variant}.log"),
        }
        if not ok:
            result["failure"] = f"{variant}: rc {run.returncode}, checkpoints {len(checkpoints)}, loss {loss}"
            write_result("train_epoch", result, fp=fp)
            raise SystemExit(f"[perf] TRAIN ({variant}) did not produce a valid run: {result['failure']}")
        metrics[f"wall_s_{variant}"] = run.wall_s
        metrics[f"startup_s_{variant}"] = round(run.clocks.get("startup", {}).get("wall", 0.0), 3)
        for i, line in enumerate(lines, 1):
            metrics[f"epoch{i}_wall_s_{variant}"] = line["wall"]
        last = lines[-1] if lines else {}
        for phase, key in (
            ("wait(data)", "wait_data"),
            ("forward", "forward"),
            ("criteria", "criteria"),
            ("backward+step", "backward_step"),
            ("validation", "validation"),
            ("telemetry", "telemetry"),
        ):
            metrics[f"{key}_s_{variant}"] = last.get(phase, 0.0)
        metrics[f"peak_rss_gib_{variant}"] = run.peak_rss_gib
        metrics[f"gpu_delta_mib_{variant}"] = run.gpu_peak_mib - run.gpu_baseline_mib
        if args.cprofile:
            prof_run = train_once(copy, scratch, variant + "_prof", gpu=gpu, cprofile=True)
            prof = scratch / f"train_{variant}_prof.prof"
            if prof.exists():
                result[f"cprofile_{variant}"] = cprofile_summary(
                    prof,
                    families={
                        "data_path": r"samples\.py|sources\.py|manager\.py|sizer\.py|sweep\.py|sitk_file\.py",
                        "criteria": r"measure/|segmentation\.py|base\.py",
                        "torch": r"torch/",
                    },
                )
                result[f"cprofile_{variant}"]["wall_s_under_cprofile"] = prof_run.wall_s

    first = variants[0]
    last_epoch = f"epoch{epochs}_wall_s_{first}"
    result["metrics"] = metrics
    result["headline"] = (
        f"epoch {first} {metrics.get(last_epoch, 0.0)} s (criteria {metrics[f'criteria_s_{first}']}, validation "
        f"{metrics[f'validation_s_{first}']}, wait(data) {metrics[f'wait_data_s_{first}']})"
        + (f" | autocast {metrics.get(f'epoch{epochs}_wall_s_autocast', 0.0)} s" if "autocast" in variants else "")
        + f" | startup {metrics[f'startup_s_{first}']} s | RSS {metrics[f'peak_rss_gib_{first}']} GiB"
    )
    path = write_result("train_epoch", result, fp=fp)
    print(json.dumps(metrics, indent=2))
    print(f"[perf] {result['headline']}")
    print(f"[perf] written {path}")


if __name__ == "__main__":
    main()

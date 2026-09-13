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

"""TRAIN, then RESUME from its checkpoint, on the shipped Segmentation example: what a resume costs.

    python benchmarks/perf/bench_resume.py [--epochs 1] [--force] [--quick] [--cpu]

One TRAIN of ``--epochs`` epochs on a scratch copy, then a RESUME of the same run from the checkpoint
it wrote, the epoch count doubled. Both commands' ``[KonfAI] startup`` clock (the checkpoint load is
its own phase) and their ``[KonfAI] epoch`` lines are parsed. A resume counts only when it exits 0,
runs exactly the epochs the checkpoint left, writes a checkpoint and prints a finite loss.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from bench_train_epoch import epoch_lines, last_loss, rewrite_config
from harness import CliRun, copy_example, fingerprint, konfai_executable, machine_gate, run_cli, write_result


def run(command: list[str], copy: Path, scratch: Path, label: str, gpu: bool) -> CliRun:
    argv = [
        *konfai_executable(),
        *command,
        "-c",
        "Config.yml",
        "-y",
        "--checkpoints-dir",
        str(scratch / "ck"),
        "--statistics-dir",
        str(scratch / "st"),
    ]
    if gpu:
        argv += ["--gpu", "0"]
    # The runtime pins its own threads; an inherited OMP_NUM_THREADS would override it.
    return run_cli(argv, cwd=copy, env={"OMP_NUM_THREADS": ""}, log_path=scratch / f"{label}.log")


def checkpoints(scratch: Path) -> list[Path]:
    return sorted((scratch / "ck").rglob("*.pt"), key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=1, help="epochs of the TRAIN, and of the RESUME after it")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    gate = machine_gate(force=args.force)
    fp = fingerprint()

    import torch

    gpu = torch.cuda.is_available() and not args.cpu
    epochs = 1 if args.quick else args.epochs
    variant = "autocast" if gpu else "fp32"
    scratch = Path(tempfile.mkdtemp(prefix="konfai_perf_resume_"))
    copy = copy_example("Segmentation", scratch / "run")

    rewrite_config(copy / "Config.yml", epochs=epochs, variant=variant)
    train = run(["TRAIN"], copy, scratch, "train", gpu)
    written = checkpoints(scratch)
    result: dict[str, object] = {"gate_warnings": gate.warnings, "epochs": epochs, "gpu": gpu, "scratch": str(scratch)}
    if train.returncode != 0 or not written:
        result["failure"] = f"TRAIN: rc {train.returncode}, checkpoints {len(written)}"
        write_result("resume", result, fp=fp)
        raise SystemExit(f"[perf] {result['failure']} (log {scratch / 'train.log'})")

    # Taken now: a BEST run prunes the checkpoint the resume starts from once it writes its own.
    trained_at = written[-1].stat().st_mtime
    rewrite_config(copy / "Config.yml", epochs=2 * epochs, variant=variant)
    resume = run(["RESUME", "--model", str(written[-1])], copy, scratch, "resume", gpu)
    resumed = epoch_lines(resume.output)
    loss = last_loss(resume.output)
    after = checkpoints(scratch)
    valid = (
        resume.returncode == 0
        and len(resumed) == epochs
        and loss is not None
        and loss == loss
        and bool(after)
        and after[-1].stat().st_mtime > trained_at
    )
    for label, command in (("train", train), ("resume", resume)):
        lines = epoch_lines(command.output)
        result[label] = {
            "returncode": command.returncode,
            "wall_s": command.wall_s,
            "startup": command.clocks.get("startup", {}),
            "epoch_lines": lines,
            "peak_rss_gib": command.peak_rss_gib,
            "log": str(scratch / f"{label}.log"),
        }
    result["resume"]["last_loss"] = loss  # type: ignore[index]
    if not valid:
        result["failure"] = f"RESUME: rc {resume.returncode}, epochs {len(resumed)} of {epochs}, loss {loss}"
        write_result("resume", result, fp=fp)
        raise SystemExit(f"[perf] {result['failure']} (log {scratch / 'resume.log'})")

    metrics = {
        "train_wall_s": train.wall_s,
        "resume_wall_s": resume.wall_s,
        "train_startup_s": round(train.clocks.get("startup", {}).get("wall", 0.0), 3),
        "resume_startup_s": round(resume.clocks.get("startup", {}).get("wall", 0.0), 3),
        "resume_checkpoint_s": round(resume.clocks.get("startup", {}).get("checkpoint", 0.0), 3),
        "resume_epoch_wall_s": resumed[-1]["wall"],
        "resume_peak_rss_gib": resume.peak_rss_gib,
    }
    result["metrics"] = metrics
    result["headline"] = (
        f"resume {metrics['resume_wall_s']} s (startup {metrics['resume_startup_s']} s, checkpoint load "
        f"{metrics['resume_checkpoint_s']} s, epoch {metrics['resume_epoch_wall_s']} s) against train "
        f"{metrics['train_wall_s']} s, {epochs} epoch(s) each"
    )
    path = write_result("resume", result, fp=fp)
    print(json.dumps(metrics, indent=2))
    print(f"[perf] {result['headline']}")
    print(f"[perf] written {path}")


if __name__ == "__main__":
    main()

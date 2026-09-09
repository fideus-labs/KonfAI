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

"""One training step of a catalog model, phase by phase, under fp32, autocast and channels_last.

    python benchmarks/perf/bench_train_step.py [--model-yaml examples/Segmentation/UNet.yml] [--shape 8,1,256,256]
                                              [--steps 40] [--force] [--quick]

In-process, on the model the shipped Segmentation example trains (built by the YAML builder, its
logits head selected the way ONNX export selects it), with a synthetic batch of the example's patch
shape. Five phases are timed with CUDA events, synchronized on both ends: host-to-device copy of a
pinned batch, forward, loss (cross-entropy on the logits; the example's second term, a Dice on the
softmax head, is not reproduced here and the audit measured the whole criteria phase at 4 % of the
step), backward, optimizer step. Four variants: fp32, autocast (fp16 with a GradScaler),
channels_last, both. ``torch.profiler`` then splits the CUDA kernel time of fp32 and autocast by
family (``harness.kernel_families`` counts every kernel event once) and lists the ten heaviest kernels.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from harness import REPO, fingerprint, kernel_families, machine_gate, write_result

PHASES = ("h2d", "forward", "loss", "backward", "optimizer")


def build(
    model_yaml: Path, shape: tuple[int, ...], device: object, classpath: str | None = None, sys_path: str | None = None
) -> tuple[object, object, int]:
    """The network to step: a catalog YAML model, or ``module:Class`` from ``sys_path`` (a challenge
    repository's own model, a KonfAI Network or a plain nn.Module), its inference head named."""
    import importlib
    import sys

    import torch
    from konfai.export import _NamedHead, select_inference_head
    from konfai.network.network import Network
    from konfai.utils.model_builder import build_model_from_yaml

    if classpath:
        for entry in (sys_path or "").split(":"):
            if entry and entry not in sys.path:
                sys.path.insert(0, entry)
        module_name, class_name = classpath.split(":", 1)
        model = getattr(importlib.import_module(module_name), class_name)()
    else:
        model = build_model_from_yaml(yaml_path=str(model_yaml))
    example = torch.zeros(*shape)
    if isinstance(model, Network):
        head = select_inference_head(model.eval(), example)
        net = _NamedHead(model, head).to(device).train()
    else:
        head, net = type(model).__name__, model.to(device).train()
    with torch.no_grad():
        classes = int(net(example.to(device)).shape[1])
    return net, head, classes


def timed_step(
    net: object, batch: object, labels: object, optimizer: object, *, device: object, autocast: bool, scaler: object
) -> dict[str, float]:
    """One step; each phase between two synchronized CUDA events (perf_counter on CPU)."""
    import time

    import torch
    import torch.nn.functional as functional

    cuda = device.type == "cuda"  # type: ignore[attr-defined]

    def stamp() -> object:
        if cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    marks = [stamp()]
    x = batch.to(device, non_blocking=True)  # type: ignore[attr-defined]
    y = labels.to(device, non_blocking=True)  # type: ignore[attr-defined]
    marks.append(stamp())
    with torch.autocast("cuda", dtype=torch.float16, enabled=cuda and autocast):
        logits = net(x)  # type: ignore[operator]
        marks.append(stamp())
        # A one-channel output is a synthesis: its loss is the L1 the challenge trains on.
        loss = functional.l1_loss(logits.float(), y) if y.is_floating_point() else functional.cross_entropy(logits, y)
    marks.append(stamp())
    optimizer.zero_grad(set_to_none=True)  # type: ignore[attr-defined]
    if scaler is not None:
        scaler.scale(loss).backward()  # type: ignore[attr-defined]
        marks.append(stamp())
        scaler.step(optimizer)  # type: ignore[attr-defined]
        scaler.update()  # type: ignore[attr-defined]
    else:
        loss.backward()
        marks.append(stamp())
        optimizer.step()  # type: ignore[attr-defined]
    marks.append(stamp())
    if cuda:
        torch.cuda.synchronize(device)
        durations = [float(marks[i].elapsed_time(marks[i + 1])) for i in range(len(PHASES))]  # type: ignore[attr-defined]
    else:
        durations = [(marks[i + 1] - marks[i]) * 1e3 for i in range(len(PHASES))]  # type: ignore[operator]
    out = dict(zip(PHASES, durations, strict=True))
    out["step"] = sum(durations)
    return out


def run_variant(
    model_yaml: Path,
    shape: tuple[int, ...],
    classes_hint: int,
    *,
    device: object,
    autocast: bool,
    channels_last: bool,
    warmup: int,
    steps: int,
    profile_steps: int,
    classpath: str | None = None,
    sys_path: str | None = None,
) -> dict[str, object]:
    import torch

    net, head, classes = build(model_yaml, shape, device, classpath, sys_path)
    memory_format = (
        torch.channels_last
        if (channels_last and len(shape) == 4)
        else (torch.channels_last_3d if channels_last else torch.contiguous_format)
    )
    if channels_last:
        net = net.to(memory_format=memory_format)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda") if (autocast and device.type == "cuda") else None  # type: ignore[attr-defined]
    generator = torch.Generator().manual_seed(0)
    batch = torch.randn(*shape, generator=generator)
    labels = (
        torch.randn(shape[0], 1, *shape[2:], generator=generator)
        if classes == 1
        else torch.randint(0, classes, (shape[0], *shape[2:]), generator=generator)
    )
    if channels_last:
        batch = batch.contiguous(memory_format=memory_format)
    if device.type == "cuda":  # type: ignore[attr-defined]
        batch, labels = batch.pin_memory(), labels.pin_memory()
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        timed_step(net, batch, labels, optimizer, device=device, autocast=autocast, scaler=scaler)
    runs = [
        timed_step(net, batch, labels, optimizer, device=device, autocast=autocast, scaler=scaler) for _ in range(steps)
    ]
    medians = {phase: round(statistics.median(r[phase] for r in runs), 3) for phase in (*PHASES, "step")}
    out: dict[str, object] = {
        "head": head,
        "classes": classes,
        "medians_ms": medians,
        "runs_ms": [{k: round(v, 3) for k, v in r.items()} for r in runs],
    }
    if device.type == "cuda":  # type: ignore[attr-defined]
        out["peak_alloc_mib"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
    if profile_steps and device.type == "cuda":  # type: ignore[attr-defined]
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(profile_steps):
                timed_step(net, batch, labels, optimizer, device=device, autocast=autocast, scaler=scaler)
        families = kernel_families(prof)
        total = families.get("total_ms", 0.0) or 1.0
        out["kernel_families_ms"] = families
        out["kernel_families_share"] = {k: round(v / total, 3) for k, v in families.items() if k != "total_ms"}
        rows = sorted(
            (e for e in prof.key_averages() if e.self_device_time_total > 0),
            key=lambda e: e.self_device_time_total,
            reverse=True,
        )
        out["top_kernels"] = [
            {"name": e.key[:90], "self_device_ms": round(e.self_device_time_total / 1e3, 3), "count": e.count}
            for e in rows[:10]
        ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-yaml", type=Path, default=REPO / "examples" / "Segmentation" / "UNet.yml")
    parser.add_argument("--shape", default="8,1,256,256", help="B,C,H,W or B,C,D,H,W of the synthetic batch")
    parser.add_argument("--model-classpath", default=None, help="module:Class of a Python model instead of the YAML")
    parser.add_argument("--sys-path", default=None, help="colon-separated directories the classpath imports from")
    parser.add_argument("--tag", default=None, help="a name for the result file (default: the model's)")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument("--variants", default="fp32,autocast,channels_last,both")
    parser.add_argument("--repeats", type=int, default=1, help="steps are the repeats here")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    gate = machine_gate(force=args.force)
    fp = fingerprint()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = tuple(int(v) for v in args.shape.split(","))
    if args.quick:
        shape = (2, shape[1], *([64] * (len(shape) - 2)))
    steps, warmup, profile_steps = (5, 2, 3) if args.quick else (args.steps, args.warmup, args.profile_steps)
    if device.type == "cpu":
        steps, warmup, profile_steps = min(steps, 5), min(warmup, 1), 0
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if device.type == "cpu":
        variants = ["fp32"]

    result: dict[str, object] = {
        "gate_warnings": gate.warnings,
        "device": str(device),
        "shape": list(shape),
        "steps": steps,
        "warmup": warmup,
        "model_yaml": None if args.model_classpath else str(args.model_yaml),
        "model_classpath": args.model_classpath,
    }
    metrics: dict[str, float] = {}
    for variant in variants:
        out = run_variant(
            args.model_yaml,
            shape,
            0,
            device=device,
            autocast=variant in ("autocast", "both"),
            channels_last=variant in ("channels_last", "both"),
            warmup=warmup,
            steps=steps,
            profile_steps=profile_steps if variant in ("fp32", "autocast") else 0,
            classpath=args.model_classpath,
            sys_path=args.sys_path,
        )
        result[variant] = out
        medians: dict[str, float] = out["medians_ms"]  # type: ignore[assignment]
        metrics[f"step_ms_{variant}"] = medians["step"]
        for phase in PHASES:
            metrics[f"{phase}_ms_{variant}"] = medians[phase]
        if "peak_alloc_mib" in out:
            metrics[f"peak_alloc_mib_{variant}"] = float(out["peak_alloc_mib"])  # type: ignore[arg-type]
        if device.type == "cuda":
            torch.cuda.empty_cache()

    base = metrics.get("step_ms_fp32", 0.0)
    fp32 = result.get("fp32", {}).get("medians_ms", {})  # type: ignore[union-attr]
    share = {p: round(fp32.get(p, 0.0) / base, 3) for p in PHASES} if base else {}
    result["fp32_phase_share"] = share
    result["metrics"] = metrics
    result["headline"] = (
        f"step fp32 {base} ms (backward {share.get('backward', 0) * 100:.0f} %, forward {share.get('forward', 0) * 100:.0f} %, "
        f"loss {share.get('loss', 0) * 100:.0f} %)"
        + "".join(
            f" | {v} {metrics[f'step_ms_{v}']} ms ({base / metrics[f'step_ms_{v}']:.2f}x)"
            for v in variants
            if v != "fp32" and metrics.get(f"step_ms_{v}")
        )
        + (" | CPU only" if device.type == "cpu" else "")
    )
    path = write_result("train_step", result, fp=fp, tag=args.tag)
    print(json.dumps(metrics, indent=2))
    print(f"[perf] {result['headline']}")
    print(f"[perf] written {path}")


if __name__ == "__main__":
    main()

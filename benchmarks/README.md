# KonfAI benchmarks

Runnable evidence for the performance claims the documentation makes. Every script pins what it
measures, prints the environment it ran in (versions, host, device), and emits the markdown row the
docs carry, so a published number is reproducible with one command.

## Protocol

- Wall time is the median of 3 runs after 1 warmup, on an otherwise idle machine.
- Host memory is the peak resident set of the whole process tree (`psutil`), sampled at 50 ms.
- Device memory is `torch.cuda.max_memory_allocated()` plus the NVML process figure when available.
- Every report line carries: konfai/torch/SimpleITK versions, CPU model, GPU model, and the input's
  shape/dtype/checksum, so two numbers are only ever compared on the same footing.

## Scripts

| Script | Claim it evidences |
|---|---|
| `bench_streaming.py` | A volume larger than the declared memory budget is transformed with peak RAM bounded by the budget, not the volume (`--gib 16 --budget 1`). |
| `bench_hotpaths.py` | Framework-side hot paths hold their measured costs: the residual `Add` fold, the one-pass collate view, deferred criterion readout. |

## App-level comparisons

The published app tables (KonfAI-MRSegmentator / KonfAI-TotalSegmentator against the original
tools) are produced by each app's own benchmark entry, which needs the published weights and a
licensed case; see `apps/mrsegmentator/README.md` and `apps/totalsegmentator/README.md`. The
protocol above applies unchanged: same case, same weights, median of 3, whole-tree RSS.

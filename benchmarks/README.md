# KonfAI benchmarks

Runnable evidence for the performance claims the documentation makes. The harness, its protocol,
the map of what each script measures and how a result is read are in
[`perf/README.md`](perf/README.md). One script sits at this level:

| Script | Claim it evidences |
|---|---|
| `bench_streaming.py` | Transforms a volume larger than the declared memory budget and reports the peak RAM of the process tree (`--gib 16 --budget 1`). `perf/bench_transform.py` reuses its synthetic volume. |

The app tables (KonfAI-MRSegmentator and KonfAI-TotalSegmentator against the original tools) are
produced by `perf/bench_apps.py`, from a manifest that names each app's command, the original tool's
command and the S/M/L cases; the published weights, the cases and the original tools' environment
are not in the repository.

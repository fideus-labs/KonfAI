# benchmarks/perf on valentin-ThinkPad-P16-Gen-3 at 2026-09-09T17:08:33+0200

commit v1.8.3-4-gd1c6961b, NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU, profile performance, load [0.39, 1.01, 11.01], OMP_NUM_THREADS=None

| bench | headline | file |
|---|---|---|
| startup | import konfai 0.03 s / konfai.trainer 0.799 s (607 MB); konfai --help 0.054 s | 20260909-170847-d1c6961b-startup.json |
| train_step | step fp32 37.629 ms (backward 66 %, forward 31 %, loss 1 %) | autocast 20.332 ms (1.85x) | channels_last 36.629 ms (1.03x) | both 16.613 ms (2.27x) | 20260909-170900-d1c6961b-train_step.json |
| train_epoch | epoch fp32 17.9 s (criteria 12.2, validation 3.5, wait(data) 0.4) | autocast 8.9 s | startup 0.0 s | RSS 3.763 GiB | 20260909-171005-d1c6961b-train_epoch.json |
| predict | prediction whole 4.43 s (loop 2.5: fetch 1.1 + forward 1.3) | streamed 4.823 s (loop 2.9) | 0 of 58446992 voxels differ | RSS 1.528 GiB | 20260909-171046-d1c6961b-predict.json |
| evaluate | evaluation 3.977 s wall, startup 0.0 s, loop about 3.977 s, RSS 2.783 GiB; 1 metric files, 393 numbers, 0 non-finite | 20260909-171514-d1c6961b-evaluate.json |
| transform | 2 GiB: KonfAI 3.944 s / 1.289 GiB at 1 GiB budget, 4.031 s at 8 GiB | naive 4.114 s / 0.234 GiB | statistics scan 0 % self, 0.365 s cumulative | max diff 1.1920928955078125e-07 | 20260909-172811-760f23ed-transform.json |
| tests | test-fast pinned 25.21 s (376.8 CPU-s, 0 failed) | unpinned 94.25 s (2110.2 CPU-s, 0 failed) | 3.74x | 20260909-173013-760f23ed-tests.json |

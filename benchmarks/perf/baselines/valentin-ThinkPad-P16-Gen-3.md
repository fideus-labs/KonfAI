# benchmarks/perf on valentin-ThinkPad-P16-Gen-3 at 2026-09-07T17:23:41+0200

commit v1.8.3-36-g81a66be8 (dirty), NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU, profile performance, load [2.5, 11.66, 13.09], OMP_NUM_THREADS=None

| bench | headline | file |
|---|---|---|
| startup | import konfai 0.031 s / konfai.trainer 0.743 s (604 MB); konfai --help 0.055 s | 20260907-172355-81a66be8-dirty-startup.json |
| train_step | step fp32 38.113 ms (backward 65 %, forward 32 %, loss 1 %) | autocast 20.112 ms (1.90x) | channels_last 35.262 ms (1.08x) | both 16.705 ms (2.28x) | 20260907-172411-81a66be8-dirty-train_step.json |
| train_epoch | epoch fp32 17.3 s (criteria 11.4, validation 3.3, wait(data) 0.5) | autocast 9.0 s | startup 0.0 s | RSS 3.76 GiB | 20260907-172515-81a66be8-dirty-train_epoch.json |
| predict | prediction whole 4.433 s (loop 2.5: fetch 1.1 + forward 1.3) | streamed 4.788 s (loop 2.9) | 0 of 58446992 voxels differ | RSS 1.516 GiB | 20260907-172555-81a66be8-dirty-predict.json |
| evaluate | evaluation 3.984 s wall, startup 0.0 s, loop about 3.984 s, RSS 2.775 GiB; 1 metric files, 393 numbers, 0 non-finite | 20260907-172614-81a66be8-dirty-evaluate.json |
| transform | 2 GiB: KonfAI 3.913 s / 1.182 GiB at 1 GiB budget, 3.875 s at 8 GiB | naive 3.972 s / 0.234 GiB | statistics scan 0 % self, 0.222 s cumulative | max diff 1.1920928955078125e-07 | 20260907-172732-81a66be8-dirty-transform.json |
| tests | test-fast pinned 22.54 s (313.9 CPU-s, 0 failed) | unpinned 110.01 s (2265.5 CPU-s, 0 failed) | 4.88x | 20260907-172946-81a66be8-dirty-tests.json |

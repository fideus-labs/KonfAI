# Installation, caches and troubleshooting

## What the first run downloads

`pip install impact-reg-konfai` installs the orchestrator, KonfAI and PyTorch. Each engine brings the rest the first time
a preset needs it:

| What | When | Size | Where |
|---|---|---|---|
| The preset files | first use of a preset | kilobytes | Hugging Face cache (`~/.cache/huggingface`) |
| Feature models | first IMPACT preset using them | 15 kB (MIND) to 125 MB (TotalSegmentator) | Hugging Face cache |
| elastix-IMPACT binary | first elastix preset | 40 MB | `~/.cache/konfai/elastix-impact` |
| LibTorch the binary was built against | first elastix preset, when your `torch` is another version | 179 MB (CPU) or 3.8 GB (CUDA) | same directory |
| `fireants` and its dependencies | first FireANTs preset | a few MB | your Python environment (pip) |
| `itk-impact` | first ConvexAdam preset | about 180 MB with ITK | your Python environment (pip) |

The preset's `requirements.txt` is pip-installed into the running environment, and the packages its `app.json` lists
under `requirements_no_deps` without their own dependencies (FireANTs pins versions the presets do not need). Packages the environment already holds
at a set version (`torch`, `konfai`, `konfai-apps`, `impact-reg-konfai`) are never replaced: a preset that needs another
version stops with a message naming it. Set `KONFAI_APPS_INSTALL_REQUIREMENTS=0` to install nothing automatically.

## Platforms

| | Linux x86_64 | Windows x86_64 | macOS (Apple Silicon) |
|---|---|---|---|
| elastix presets | CPU; IMPACT metric on CUDA GPUs | CPU; IMPACT metric on CUDA GPUs | CPU |
| FireANTs presets | GPU or CPU | not available (FireANTs imports `fcntl`) | CPU |
| ConvexAdam presets | GPU or CPU | GPU or CPU | not available (no `itk-impact` wheel) |

Tested on Linux x86_64. The Windows and macOS columns follow the binaries and wheels that exist for them; they are not
tested.

The ConvexAdam presets need `torch` 2.12 (`itk-impact` is built against it). Install it alongside:
`pip install impact-reg-konfai "torch==2.12.*"`, adding `--index-url https://download.pytorch.org/whl/cu128` (or `cu130`)
for a specific CUDA build.

## Devices

Without `--gpu`, everything runs on the CPU, and `register` says so when a GPU is available. `--gpu 0` names a physical
GPU: the run's `CUDA_VISIBLE_DEVICES` becomes the ids given, replacing any mask already set.

## Environment variables

| Variable | Effect |
|---|---|
| `KONFAI_IMPACTREG_REPO` | where the presets come from: a Hugging Face repository (`VBoussot/ImpactReg`), `repo@revision` to pin one, or a local directory of preset folders |
| `KONFAI_ELASTIX_DIR` | an elastix-IMPACT install to use instead of downloading one |
| `KONFAI_APPS_INSTALL_REQUIREMENTS=0` | never pip-install a preset's requirements |
| `HF_HUB_OFFLINE=1` | work from the local caches only |
| `IMPACT_REG_DEBUG=1` | print the full traceback of a failure |

## Working offline

Run each preset once online (or `register --download`), then set `HF_HUB_OFFLINE=1`: `list`, `show`, `register`, `eval`,
`uncertainty` and `apply` work from the caches. To pin what a cohort was registered with, set
`KONFAI_IMPACTREG_REPO=VBoussot/ImpactReg@<revision>`; `register.json`, written beside every run's outputs, records the
repository, the versions, the overrides and the inputs of each case.

## Common problems

| Symptom | Cause and fix |
|---|---|
| `elastix` fails to start with an undefined symbol | the binary does not match your `torch`: current versions download the LibTorch it was built against; an older install can be removed from `~/.cache/konfai/elastix-impact` to start over |
| a ConvexAdam preset stops asking for `torch==2.12.*` | see *Platforms* |
| a deformable MR/CT result pulls the MRI onto the CT's table or head rest, or bends it past its field of view | the metric is scored where only one image has content: give the CT a body mask, `--fixed-mask body.mha` |
| a rigid or affine stage lands millimetres to centimetres off on an image with a few extreme voxels (light-sheet microscopy, a bright artefact) | mutual information bins each image between its minimum and maximum, so a handful of voxels far above the rest squeeze the tissue into one bin. Every intensity stage clamps each image to its 0.01 and 99.99 percentiles first (FireANTs to 0.5 and 99.5); the rigid stage of an elastix IMPACT preset reads the raw intensities its feature models need, so clamp such images yourself before an IMPACT preset |
| `register` refuses an input "in a case directory this run writes" | the moving image is an output of an earlier run into the same `-o`: write the new run elsewhere |
| a CPU run takes very long | FireANTs and the IMPACT metric are GPU methods; on the CPU prefer `Generic_*` and `ConvexAdam_*` |
| out of GPU memory | runs are sized from the memory free at start and shrink once if a pass still runs out; a process that grabs the card mid-run can exhaust that. Lower `--max-voxels` |
| a CPU run is killed by the system | the run is sized from the RAM available at start; something else took RAM during the run. Lower `--max-voxels` |
| results differ between two runs of a large pair | the plan follows the free memory at start: pin `--max-voxels` |

The failed preset's logs and configuration are kept, and the message says where.

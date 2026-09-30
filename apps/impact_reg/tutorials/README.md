# IMPACT-Reg tutorials

Runnable scripts, from a first registration to large images and ensembles. Each one runs from this directory on a
fresh environment:

```bash
pip install impact-reg-konfai "torch==2.12.*" matplotlib "fsspec[http]"
git clone https://github.com/fideus-labs/KonfAI && cd KonfAI/apps/impact_reg/tutorials
bash 01_first_registration.sh
```

`GPU=0` is the default; set `GPU=` (empty) to run on the CPU where a tutorial allows it. Outputs go to `out/`.

| # | Script | What it shows | Time on one GPU |
|---|---|---|---|
| 1 | `01_first_registration.sh` | installing, the presets, a rigid then a deformable registration, the outputs, before/after | ~2 min (more on the first run: downloads) |
| 2 | `02_feature_maps.py` | what IMPACT compares: CT and MRI through MIND, TotalSegmentator and anatomix, and where their features agree | under a minute, CPU |
| 3 | `03_three_backends.sh` | one MR/CT pair through elastix, FireANTs and ConvexAdam, intensity and IMPACT presets, scored alike | ~5 min |
| 4 | `04_tuning.sh` | `show NAME`, `--set`, feature layers for MR/CT versus CT/CBCT, iterations, regularisation | ~9 min |
| 5 | `05_large_images.sh` | two mouse brains of 190 million voxels from public ExaSPIM data through each engine: native tiling, memory, disk | ~17 min (plus the download), 20 GB of disk |
| 6 | `06_evaluation.sh` | MAE, Dice, TRE and Jacobian before and after, the landmark direction, `apply --labels` | ~2 min |
| 7 | `07_ensemble_uncertainty.sh` | three presets ensembled, each member scored, the spread map | ~3 min |

## The data

`prepare_data.py` downloads a head and neck CT and the MRI of the same patient (SynthRAD2025 case 1HNA001, from the
public [`VBoussot/konfai-demo`](https://huggingface.co/datasets/VBoussot/konfai-demo) dataset), resamples both to 2 mm,
and moves the MRI by a known transform: a rigid one (4, -3, 6 degrees; 7, -5, 9 mm) and a smooth deformation (2 mm on
average in the body, up to 12 mm). The labels (soft tissue, bone, air) and the 16 landmarks written for the MRI go
through that same transform, and the tutorials score each registration against them: a rigid registration leaves the
deformation (about 3 mm at the landmarks), a deformable one can recover it. The truth is exact up to the dataset's own
alignment of the two scans, taken apart, which a deformable preset may correct where the truth does not.

`download_exaspim.py` reads two mouse brains imaged by ExaSPIM light-sheet microscopy (Allen Institute for Neural
Dynamics, public `aind-open-data` bucket) at one level of their pyramid, over HTTP (hence `fsspec[http]`).

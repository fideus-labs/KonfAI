[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![PyPI version](https://img.shields.io/pypi/v/impact_reg_konfai.svg?color=blue)](https://pypi.org/project/impact_reg_konfai/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/fideus-labs/KonfAI/actions/workflows/konfai_ci.yml/badge.svg)](https://github.com/fideus-labs/KonfAI/actions/workflows/konfai_ci.yml)
[![Paper](https://img.shields.io/badge/📌%20Paper-KonfAI-blue)](https://www.arxiv.org/abs/2508.09823)

<p align="center">
  <img src="Logo.png" alt="IMPACT-Reg logo" width="220">
</p>

# IMPACT-Reg-KonfAI

**Fast and lightweight CLI for multimodal medical image registration using IMPACT-Reg presets within the KonfAI framework.**

---

## 🧩 Overview

**IMPACT-Reg-KonfAI** is the **command-line interface (CLI)** for running **IMPACT-Reg** registration presets published
in the [`VBoussot/ImpactReg`](https://huggingface.co/VBoussot/ImpactReg) Hugging Face repository, through the
[KonfAI](https://github.com/fideus-labs/KonfAI) deep learning framework.

**IMPACT-Reg** introduces a **semantic similarity metric** for **multimodal registration**, driven by deep features
extracted from large pretrained segmentation and foundation models (TotalSegmentator, MIND, anatomix). The presets run
it in three registration engines, **elastix**, **ConvexAdam** (through `itk-impact`) and **FireANTs**, beside
intensity-based presets for pairs of the same modality.

A registration run combines:

- **fixed** and **moving** images
- one or more **registration presets** resolved from the published preset database (each preset is a KonfAI app)
- optional **image**, **segmentation**, or **landmark** references (with an optional mask) for evaluation

---

## 🧠 Features

- ⚡ **Fast registration** powered by [KonfAI](https://github.com/fideus-labs/KonfAI)
- 🤗 **Automatic preset, parameter-map, and model download** from Hugging Face
- 🧩 **Multi-preset ensembling** (transforms averaged into a single displacement field)
- 📏 **Large images**: a pair too large for the memory is registered whole on a grid sized to fit, then refined on
  native tiles by the presets that declare a tile pass; the outputs are on the native fixed grid
- 🧠 **Semantic IMPACT metric** on deep features from pretrained segmentation / foundation models
- 📐 **Evaluation workflows** against image, segmentation, and landmark references
- 🧾 **Multi-format compatibility:** every format ITK reads, DICOM series, and **OME-Zarr** stores (the moved image
  comes back in the moving image's own format)

---

## 🗂️ Available presets

Each preset is a KonfAI app on [`VBoussot/ImpactReg`](https://huggingface.co/VBoussot/ImpactReg), downloaded with
its parameters and models on first use, and passed to `register` by name. List them with:

```bash
impact-reg-konfai list
```

| Preset | Pair | Engine | What it does |
|---|---|---|---|
| `Generic_Rigid` | any | elastix | Rigid alignment, mutual information, multi-resolution |
| `Generic_Rigid_BSpline` | any | elastix | Rigid, then B-spline deformable refinement |
| `Elastix_IMPACT_Static` | MRI / CT | elastix + IMPACT | Rigid, then a B-spline driven by TotalSegmentator (decoder, then segmentation head) and MIND, features extracted once |
| `Elastix_IMPACT_Jacobian` | CT / CBCT | elastix + IMPACT | Rigid, then a B-spline driven by early TotalSegmentator layers, differentiated through the network |
| `ConvexAdam_Coarse` | any | itk-impact | Linear pre-alignment, then the global coarse coupled-convex initialisation on MIND features |
| `ConvexAdam_Composite` | any | itk-impact | The same coarse pass followed by the Adam instance-optimisation refinement |
| `ConvexAdam_IMPACT_CBCT` | CT / CBCT | itk-impact | The same pipeline on the second layer of TotalSegmentator MR |
| `ConvexAdam_IMPACT_MRCT` | MRI / CT | itk-impact | The same pipeline on the last decoder layer of TotalSegmentator MR and MIND |
| `FireANTs_SyN` | any | FireANTs | Rigid, affine, then SyN diffeomorphic registration on the GPU |
| `FireANTs_IMPACT` | CT / CBCT | FireANTs + IMPACT | Rigid, affine, then SyN driven by IMPACT on early TotalSegmentator layers |
| `FireANTs_Anatomix` | any | FireANTs + IMPACT | anatomix and MIND features extracted once and registered as feature volumes |
| `FireANTs_IMPACT_MRCT` | MRI / CT | FireANTs + IMPACT | Rigid, affine, then SyN on the last decoder layer of TotalSegmentator MR and MIND, extracted once |

Every deformable preset aligns the pair rigidly or affinely first, so a pair that starts centimetres apart needs no
separate rigid run. `impact-reg-konfai show NAME` says what a preset runs, needs and tunes, and
[the presets page](docs/presets.md) which one suits which pair, with the benchmark behind that advice.

> **Give the CT a body mask** (`--fixed-mask`) whenever it shows what the other image does not: a table, a head rest,
> a field of view past the MRI's. A deformable stage otherwise pulls the moving image onto them.

---

## 🚀 Installation

From PyPI:

```bash
python -m pip install impact-reg-konfai
```

The ConvexAdam presets run on `itk-impact`, whose wheels are built against torch 2.12: install that torch alongside,
`python -m pip install impact-reg-konfai "torch==2.12.*"`. A preset's requirements never replace the torch you have.
The first elastix preset downloads the elastix-IMPACT binary (and, when your torch is another version than the one it
was built with, the matching LibTorch), the first FireANTs preset installs `fireants`; see
[Installation, caches and troubleshooting](docs/troubleshooting.md).

From source:

```bash
git clone https://github.com/fideus-labs/KonfAI.git
cd KonfAI
# konfai and konfai-apps must come from the same checkout: this app pins both to its own
# setuptools_scm version, which only exists on PyPI at a release tag.
python -m pip install -e . -e konfai-apps -e apps/impact_reg
```

---

## ⚙️ Usage

The CLI is organised into sub-commands, matching the registration workflow:

| Sub-command | Purpose |
|---|---|
| `list`, `show` | List the presets; say what one runs, needs, costs and tunes (`show NAME`). |
| `register` | Register a moving image onto a fixed image with one or more presets. Several presets are ensembled (their displacement fields are averaged). Writes the transform, the moved image derived from it, a record of the run (`register.json`) and, with `--keep-fields`, each preset's own field under `Ensemble/`. |
| `eval` | Evaluate a registration on any subset of modalities: image (MAE), segmentation (Dice), landmarks (TRE), and for a displacement field its Jacobian (the fraction of folded voxels). |
| `apply` | Warp more moving-side images through a transform: another sequence, or a label map (`--labels`, nearest neighbour). |
| `uncertainty` | Voxel-wise spread map from an ensemble of displacement fields. |

Register a moving image onto a fixed image (ensemble several presets by listing them):

```bash
impact-reg-konfai register <PRESET> [<PRESET_2> ...] -f fixed.nii.gz -m moving.nii.gz -o ./Output --gpu 0
# an MRI onto a CT, with the CT's body mask:
impact-reg-konfai register Elastix_IMPACT_Static -f ct.nii.gz -m mr.nii.gz --fixed-mask ct_body.nii.gz -o ./Output --gpu 0
```

Each case gets its transform, `Output/P000/Transform.h5` (an ITK displacement field on the fixed grid mapping a fixed
point to its moving partner, in mm), and the moving image resampled through it, `Output/P000/Moved.<ext>`. OME-Zarr
stores go in and come out the same way:

```bash
impact-reg-konfai register FireANTs_SyN -f fixed.ome.zarr -m moving.ome.zarr -o ./Output --gpu 0
```

Evaluate a registration on any subset of modalities, always from the original moving data (the transform warps it);
without `--transform` it scores the pair as it came:

```bash
impact-reg-konfai eval \
  --transform ./Output/P000/Transform.h5 \
  -f fixed.nii.gz -m moving.nii.gz --mask roi.nii.gz \
  --gt-fixed-seg fixed_seg.nii.gz --gt-moving-seg moving_seg.nii.gz \
  --gt-fixed-fid fixed.fcsv --gt-moving-fid moving.fcsv \
  -o ./Output --gpu 0
```

Warp a label map of the moving image onto the fixed one:

```bash
impact-reg-konfai apply --transform ./Output/P000/Transform.h5 -f fixed.nii.gz -i moving_labels.nii.gz --labels -o ./Output/P000
```

Estimate uncertainty from the per-preset displacement fields written by `register`:

```bash
impact-reg-konfai register <PRESET_1> <PRESET_2> -f fixed.nii.gz -m moving.nii.gz -o ./Output --gpu 0 --keep-fields
impact-reg-konfai uncertainty --dvf ./Output/P000/Ensemble/*.h5 -o ./Output/P000
```

### `register` arguments

| Flag | Description | Default |
|------|-------------|---------|
| `PRESET`, `-p`, `--preset` | One or more presets (several are ensembled); an unknown name is reported before anything runs | *required* |
| `-f`, `--fixed-images` | Fixed image(s), or a dataset directory | *required* |
| `-m`, `--moving-images` | Moving image(s), or a dataset directory | *required* |
| `--fixed-mask`, `--moving-mask` | Optional masks restricting the metric region | *unset* |
| `-o`, `--output` | Output directory | `./Output/` |
| `--tta` | Test-time-augmentation draws per preset | `0` |
| `--keep-fields` | Keep each preset's field under `Ensemble/` for a later `uncertainty` run | `False` |
| `--fields-only` | Write the transforms and stop; skip deriving the moved images | `False` |
| `--max-voxels` | Register whole a pair of at most this many voxels; a larger one runs in two passes (see *Large images*) | what the device holds at the preset's declared cost |
| `--set [PRESET:]NAME=VALUE` | Tune a preset parameter (repeatable); `PRESET:` limits it to one preset of an ensemble; checked before anything runs | *unset* |
| `--tmp-dir` | Where the volume-sized intermediates are staged, the engines' temporary files included | hidden beside the output |
| `--gpu` / `--cpu` | GPU id(s) / CPU worker processes | CPU if unset |
| `-q`, `--quiet` | Suppress console output | `False` |

### `eval` arguments (at least one modality required)

| Flag | Description | Default |
|------|-------------|---------|
| `--transform` | Transform(s) from a prior `register` (identity if omitted) | *unset* |
| `-f`, `-m` | Fixed / moving images: image modality (MAE) | *unset* |
| `--gt-fixed-seg`, `--gt-moving-seg` | Fixed / moving segmentations: seg modality (Dice) | *unset* |
| `--gt-fixed-fid`, `--gt-moving-fid` | Fixed / moving landmarks: fid modality (TRE) | *unset* |
| `--mask` | Evaluation mask(s) for the image modality | *unset* |
| `-o`, `--output` | Output directory: `Evaluation_summary.json` holds every metric, per case and over the cohort | `./Output/` |

### `uncertainty` arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--dvf` | Two or more ensemble displacement fields (e.g. the per-preset fields from `register`) | *required* |
| `-o`, `--output` | Output directory | `./Output/` |

See the full help of any sub-command with:

```bash
impact-reg-konfai register --help
```

---

## 📖 Documentation and tutorials

- [How it works](docs/how-it-works.md): the transform, the IMPACT features, the three engines.
- [The presets](docs/presets.md): what each runs, which one for which pair, the benchmark.
- [Parameters](docs/parameters.md): what to tune when a registration falls short, with `--set`.
- [Large images](docs/large-images.md): native tiles, memory, disk.
- [Evaluation](docs/evaluation.md): MAE, Dice, TRE, the Jacobian, ensembles and their spread.
- [Installation, caches and troubleshooting](docs/troubleshooting.md), and [architecture](docs/architecture.md) for
  contributors.
- [Tutorials](tutorials/README.md): seven runnable scripts, from a first registration to large images and ensembles,
  on public data with a known transform.

---

## 📏 Large images

A pair of any size is registered with no option to set. Each preset declares what a voxel costs it on the GPU and in
RAM, and KonfAI sizes the run from the GPU memory free when it starts and from the RAM budget, whichever holds fewer
voxels. A pair that fits is registered in one piece, at native resolution. A larger one runs in two passes:

1. **Global pass:** the whole preset runs on the pair resampled onto a grid coarse enough to fit, and its field comes
   back onto the native fixed grid.
2. **Tile pass:** the preset's deformable stage refines that result on native tiles, and the two fields are composed.
   Presets without a deformable stage worth refining locally stop after the global pass.

```text
[ImpactReg] FireANTs_SyN: 257 x 665 x 887 voxels, more than the 19,636,779 it registers whole on this GPU: the preset
on the whole pair resampled to fit, then its deformable stage on native tiles of at most 17,705,407 voxels.
```

Everything around the presets is streamed, outputs included (`.mha`, `.nii`, `.nii.gz`, `.nrrd`, OME-Zarr). Pin
`--max-voxels` for reproducible runs. Details, disk and RAM: [Large images](docs/large-images.md).

---

## ⚡ Performance

Two mouse brains imaged by ExaSPIM light-sheet microscopy at the Allen Institute for Neural Dynamics (specimens
841260, fixed, and 823508, moving, from the public `aind-open-data` bucket), registered from OME-Zarr to OME-Zarr at
three levels of their pyramid on one **NVIDIA RTX PRO 5000 Blackwell Laptop GPU (24 GB)**, with at most 64 GB of RAM,
in a fresh environment as pip installs it. The time is the whole `register` command, the moved image at native
resolution included; the tiles are those of [Large images](#-large-images), for a pair larger than what the preset
registers whole.

| Preset | **S** · 160 µm<br>128 × 332 × 443 | **M** · 80 µm<br>257 × 665 × 887 | **L** · 40 µm<br>514 × 1331 × 1775 |
|---|---|---|---|
| `Generic_Rigid` | 6 s | 24 s | 2 min 18 s |
| `Generic_Rigid_BSpline` | 8 s | 33 s | 7 min 15 s · 10 tiles |
| `ConvexAdam_Coarse` | 38 s | 1 min 18 s | 2 min 37 s |
| `ConvexAdam_Composite` | 42 s | 2 min 30 s · 3 tiles | 9 min 40 s · 30 tiles |
| `FireANTs_SyN` | 1 min 6 s | 5 min 47 s · 5 tiles | 48 min 7 s · 49 tiles |
| `FireANTs_IMPACT` | 3 min 10 s | 27 min 51 s · 12 tiles | |
| `FireANTs_Anatomix` | 1 min 54 s · 2 tiles | 12 min 40 s · 18 tiles | |

Registration takes the correlation between the two brains, inside the fixed brain, from 0.27 to 0.65–0.70 with
`ConvexAdam_Composite` and the FireANTs presets, and the Dice of the two brains from 0.71 to 0.94–0.95, at every size;
`Generic_Rigid` alone reaches 0.52–0.54 (Dice 0.89), `Generic_Rigid_BSpline` 0.56–0.58 (0.91–0.92). At 40 µm,
`FireANTs_IMPACT` and `FireANTs_Anatomix` would take about a hundred and a hundred and fifty tiles: at the time a tile
takes them at 80 µm, some four and two hours, not run here. The process tree peaks under 17 GB of RAM at 80 µm and
between 25 and 39 GB at 40 µm, where KonfAI streams what it resamples and composes within the memory it finds;
the intermediates of a tiled run at 40 µm take up to 78 GB of disk, which `register` checks before it starts.

The elastix IMPACT presets compare TotalSegmentator features at 2 to 6 mm voxels, a scale made for human MRI/CT and
CT/CBCT pairs, where a mouse brain is a few voxels across: they are not run here. How accurate each preset is on
human MRI/CT, CBCT/CT, CT/CT and MRI/MRI pairs, on nine public datasets: [the presets page](docs/presets.md).

---

## 📦 Notes

- Available presets are resolved dynamically from the published IMPACT-Reg preset database.
- Multiple presets can be provided in one command; their displacement fields are averaged into a single field.
- The wrapper orchestrates the preset KonfAI apps (model inference), then ensembles, evaluates, and estimates uncertainty on their outputs.
- [SlicerImpactReg](https://github.com/vboussot/SlicerImpactReg) runs the same `register` from 3D Slicer, and KonfAI
  Studio from its assistant: konfai-mcp's `run_app_infer` on a preset runs `register`, and
  `run_registration_evaluate` runs `eval`.

---

## 📚 References

If you use **IMPACT-Reg-KonfAI** in your work, please cite KonfAI and the IMPACT-Reg paper.

- Boussot, V., Hémon, C., Nunes, J.-C., Dowling, J., Rouzé, S., Lafond, C., Barateau, A., & Dillenseger, J.-L.
  **IMPACT-Reg: A Generic Semantic Loss for Multimodal Medical Image Registration.**

- Boussot, V., & Dillenseger, J.-L. (2025).
  **KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging.**
  arXiv preprint [arXiv:2508.09823](https://arxiv.org/abs/2508.09823)

---

## 🔗 Links

- 🤗 **Model Hub:** [huggingface.co/VBoussot/ImpactReg](https://huggingface.co/VBoussot/ImpactReg)
- 📦 **PyPI Package:** [pypi.org/project/impact_reg_konfai](https://pypi.org/project/impact_reg_konfai)
- 🧠 **KonfAI Repository:** [github.com/fideus-labs/KonfAI](https://github.com/fideus-labs/KonfAI)
- 🧩 **3D Slicer extension:** [github.com/vboussot/SlicerImpactReg](https://github.com/vboussot/SlicerImpactReg)

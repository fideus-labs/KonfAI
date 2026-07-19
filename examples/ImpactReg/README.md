# IMPACT-Reg Example

This example shows **pairwise image registration** with [IMPACT-Reg](https://github.com/vboussot/ImpactLoss),
run through the KonfAI runtime — including **patch tiling + overlap blending**, the mechanism KonfAI uses to
register a volume too large to process in one pass.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fideus-labs/KonfAI/blob/main/examples/ImpactReg/register_demo.ipynb)

Unlike the `Segmentation` and `Synthesis` examples (raw `konfai TRAIN -> PREDICTION -> EVALUATION`), a
registration *preset* is a self-contained KonfAI **app**: it produces, on the fixed grid, the moving image
resampled onto the fixed image (`Moved`) and the displacement field (`DVF`). The `impact-reg-konfai` CLI
runs one or more presets, ensembles their fields, evaluates, and estimates uncertainty.

## What you will find in this folder

```text
examples/ImpactReg/
├── README.md
└── register_demo.ipynb
```

The notebook is self-contained and designed to work from a **fresh environment**, including **Google
Colab**. Rather than downloading data, it **builds a synthetic fixed/moving pair with a known smooth
deformation**, so it can measure exactly how much of the warp the registration recovered.

It walks through:

1. building the `96³` demo pair (moving = fixed warped by a ~4-voxel low-frequency field),
2. inspecting the initial misalignment (magenta/green overlay, checkerboard, NCC),
3. registering with the **FireANTs SyN** preset under forced `64³` patch tiling + `Cosinus` overlap
   blending,
4. verifying the warp is recovered (`NCC ≈ 0.84 → ≈ 0.99`) with **no patch seam** in the reassembled field.

## Quick start (CLI)

The same command the notebook builds, from any fixed/moving pair:

```bash
pip install ./apps/impact_reg   # the impact-reg-konfai CLI

impact-reg-konfai register FireANTs_SyN \
  -f fixed.mha -m moving.mha \
  -o Output --gpu 0 \
  --set Predictor.Dataset.Patch.patch_size=[64,64,64] \
  --set Predictor.Dataset.Patch.overlap=16 \
  --set Predictor.outputs_dataset.MovedImage.OutputDataset.patch_combine=Cosinus \
  --set Predictor.outputs_dataset.DisplacementField.OutputDataset.patch_combine=Cosinus
```

This writes `Output/P000/{Moved.mha, DVF.mha, Transform.h5}`.

- The presets are external model apps on the Hugging Face repo `VBoussot/ImpactReg`; FireANTs needs a CUDA
  GPU. Set `KONFAI_IMPACTREG_REPO` to a local bundle directory to run offline.
- Drop the four patch `--set` overrides to register the whole volume in one pass; keep them (with any patch
  size) for volumes too large to fit at once — the `Cosinus` blend reassembles the tiles seamlessly.

## What to adapt first

1. **preset** — `register ConvexAdam_Fine ...` (itk-impact, GPU) or `register Generic_Rigid ...` (elastix, CPU);
2. **ensemble** — pass several presets as positionals (`register FireANTs_SyN ConvexAdam_Fine ...`); the fields
   are averaged, and `--uncertainty` retains the per-preset fields for an `impact-reg-konfai uncertainty` map;
3. **patch size / overlap** — size the patch to your GPU budget; larger overlap = smoother blend;
4. **inputs** — the same command reads OME-Zarr or DICOM directly (KonfAI auto-detects the store format).

## Evaluate a registration

```bash
impact-reg-konfai eval --preset FireANTs_SyN \
  -f fixed.mha -m moving.mha \
  --gt-fixed-seg fixed_seg.mha --gt-moving-seg moving_seg.mha \
  -o Output
```

scores image MAE, segmentation Dice, or landmark TRE, depending on which ground-truth modality you pass.

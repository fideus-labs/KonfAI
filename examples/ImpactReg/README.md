# IMPACT-Reg Example

This example registers **two different patients** with [IMPACT-Reg](https://github.com/vboussot/ImpactLoss),
run through the KonfAI runtime — real pelvis CT, a real anatomical difference to recover, and a score
computed on reference segmentations rather than on a deformation we invented.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fideus-labs/KonfAI/blob/main/examples/ImpactReg/register_demo.ipynb)

**Run all the cells of `register_demo.ipynb`.** It downloads the public pelvis CT subset, aligns
`1PC032` onto `1PC006`, then scores the result twice — before and after — and plots both. About one
minute on a GPU.

Unlike the `Segmentation` and `Synthesis` examples (raw `konfai TRAIN -> PREDICTION -> EVALUATION`), a
registration *preset* is a self-contained KonfAI **app**: it produces, on the fixed grid, the moving
image resampled onto the fixed image (`Moved`) and the displacement field (`DVF`). The
`impact-reg-konfai` CLI runs one or more presets, ensembles their fields, evaluates, and estimates
uncertainty.

## What you will find in this folder

```text
examples/ImpactReg/
├── README.md
└── register_demo.ipynb
```

## How the result is scored

Both patients ship a 41-label reference segmentation, and that is the ground truth: propagate the
moving patient's labels through the recovered displacement field and measure their Dice against the
fixed patient's labels. `impact-reg-konfai eval` does exactly that — run it with no `--transform` for
the **before**, and with the `Transform.h5` that `register` wrote for the **after**.

On `1PC032 -> 1PC006` with the `FireANTs_SyN` preset, mean Dice over the 23 labels present goes from
**0.29 to 0.60**, and 20 of the 23 improve. Bone and muscle land around 0.83-0.90; a few small
structures move the wrong way, which is ordinary for inter-patient registration and is why the
notebook plots a per-label chart rather than a single number.

## Quick start (CLI)

```bash
# Both from this checkout: the app pins konfai-apps== its own setuptools_scm version, which only
# exists on PyPI at a release tag.
pip install ./konfai-apps ./apps/impact_reg

impact-reg-konfai register FireANTs_SyN \
  -f fixed.mha -m moving.mha \
  -o Output --gpu 0
```

This writes `Output/P000/{Moved.mha, DVF.mha, Transform.h5}` on the fixed grid. Then score it:

```bash
impact-reg-konfai eval \
  -f fixed.mha -m moving.mha \
  --gt-fixed-seg fixed_seg.mha --gt-moving-seg moving_seg.mha \
  --transform Output/P000/Transform.h5 \
  -o Output/after
```

`eval` also takes `--gt-fixed-fid` / `--gt-moving-fid` for landmark TRE, and `--mask` to restrict the
score to a region. The presets are external model apps on the Hugging Face repo `VBoussot/ImpactReg`;
FireANTs needs a CUDA GPU. Set `KONFAI_IMPACTREG_REPO` to a local bundle directory to run offline.

## A volume too large for one pass

Add patch overrides and KonfAI registers overlapping patches, then reassembles the moved image and the
displacement field with a `Cosinus` partition-of-unity window, which leaves no seam in the field:

```bash
impact-reg-konfai register FireANTs_SyN -f fixed.mha -m moving.mha -o Output --gpu 0 \
  --set Predictor.Dataset.Patch.patch_size=[128,128,128] \
  --set Predictor.Dataset.Patch.overlap=16 \
  --set Predictor.outputs_dataset.MovedImage.OutputDataset.patch_combine=Cosinus \
  --set Predictor.outputs_dataset.DisplacementField.OutputDataset.patch_combine=Cosinus
```

Every override is forwarded verbatim to `konfai-apps infer --set`, so the same command scales to any
preset and any volume.

## What to adapt first

1. **preset** — `register ConvexAdam_Fine ...` (itk-impact, GPU) or `register Generic_Rigid ...` (elastix, CPU);
2. **ensemble** — pass several presets as positionals (`register FireANTs_SyN ConvexAdam_Fine ...`); the fields
   are averaged, and `--uncertainty` retains the per-preset fields for an `impact-reg-konfai uncertainty` map;
3. **patch size / overlap** — size the patch to your GPU budget; larger overlap = smoother blend;
4. **inputs** — the same command reads OME-Zarr or DICOM directly (KonfAI auto-detects the store format).

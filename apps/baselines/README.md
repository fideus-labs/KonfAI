# Baselines

The registration baselines of [ImagingBenchmarks](https://github.com/fideus-labs/ImagingBenchmarks) as KonfAI
apps. Each app runs its authors' code at a pinned version, fetched on first use, with the settings the benchmark
runs it with written in its `Prediction.yml`: a wrapper, never a re-implementation. An app is checked against the
benchmark's own line before it is published.

```bash
pip install baselines-konfai
baselines-konfai register ConvexAdam_MIND -i fixed.nii.gz -i moving.nii.gz -o out --gpu 0
```

The apps live on Hugging Face under `fideus-labs/Baselines`; the folders in `bundles/` are their sources.
`BASELINES_KONFAI_REPO` points the command at a local directory of app folders instead, and
`BASELINES_KONFAI_CACHE` (default `~/.cache/baselines-konfai`) is where the authors' code, binaries, weights and
environments land.

Every app takes a fixed and a moving image, already aligned (the deformable stage alone runs), optional masks, and
writes the displacement field on the fixed grid, in mm along the world axes (ITK, LPS): `x` in the fixed image
matches `x + u(x)` in the moving one.

| app | runs | pinned at |
|---|---|---|
| `ConvexAdam_MIND` | `convex_adam_pt` of multimodallearning/convexAdam, defaults | `b229e52` |
| `UniGradICON`, `MultiGradICON` | `unigradicon-register` with the authors' weights, 50 steps of instance optimisation on LNCC | unigradicon 1.0.5, weights by hash |
| `UniGradICON_lungmask`, `MultiGradICON_lungmask` | the same, the lung masks given to the command | unigradicon 1.0.5 |
| `Demons` | ITK's fast symmetric forces Demons as SimpleITK's Demons notebook runs it | SimpleITK ≥ 2.5 |
| `CorrField` | `corrfield.py` of the Lung250M-4B repository, defaults, on a 1 mm grid | `53598ca` |
| `Anatomix_ConvexAdam` | ConvexAdam on anatomix and MIND-SSC features, the anatomix pipeline's defaults | `4c86aee`, weights `842cfe9` |
| `ANTs_SyN` | `antsRegistration` SyN as the LUMIR organisers' `antsRegistrationSyN[so]` call builds it, from the release binaries | ANTs 2.6.5, archive by hash |
| `NiftyReg` | `reg_f3d` with its defaults, built from the authors' repository (cmake) | `ee6a92d`, v2.0.0 |
| `Deeds` | `deedsBCV` with its defaults, built from the authors' repository (make) | `00b4653` |
| `Greedy` | greedy's Quick Start deformable call, NCC 2x2x2, 100x50x10 iterations | picsl_greedy 1.4.0.3 |
| `SynthMorph` | FreeSurfer's `mri_synthmorph register -m deform`, defaults, in its own TensorFlow environment (uv) | freesurfer `21b31bf`, weights by hash |
| `BrainMorph` | `scripts/register.py` as the LUMIR organisers call it: 512 keypoints, H model, `tps_0`, in its own torch 2.3 environment (uv) | `106b029`, weights by hash |

`SynthMorph` and `BrainMorph` need `uv` on the host (`pip install uv`); `NiftyReg` and `Deeds` need a C++ compiler
with cmake and make. ANTs, NiftyReg, Deeds, Greedy and Demons run on the CPU; the others want a GPU.

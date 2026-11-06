# Presets inside an app, and direct options for shared settings

Status: decided 2026-11-06; konfai-apps, konfai-mcp and impact-reg-konfai implemented on this branch. Still to do:
the app repositories on Hugging Face, SlicerKonfAI and SlicerImpactReg, then fidopt and fidlab.

## The problem

**One app per parameterization.** VBoussot/ImpactReg publishes 12 apps over three engines, most a few lines of
`Prediction.yml` apart: `Generic_Rigid_BSpline` is `Generic_Rigid` plus a B-spline map and a 16 mm grid,
`ConvexAdam_Composite` is `ConvexAdam_Coarse` plus the fine stage. `Evaluation_*.yml` and `Uncertainty.yml` are the
same file in every app. Users hit the same wall: SlicerKonfAI's "Save as local app" kept tuned parameters as a new
app folder, and fidopt names a line `<method>@<variant>`, so a tuned configuration of an IMPACT-Reg app had no name.

**Long `--set` paths.** A dotted name started at the config root, so a nested model parameter took
`--set Predictor.Model.RegistrationNet.models.0.layers_mask=01`, and OME-Zarr outputs took
`--set Predictor.outputs_dataset.DisplacementField.OutputDataset.dataset_filename=Transform:omezarr`.

## The design

Three words for the user: the **app** is what runs (an engine), a **preset** is a named way to run it, a **setting**
is what one run changes.

### Presets

- **A preset is a description and a list of `NAME=VALUE`,** exactly what `--set` takes, and `"tiles": false` for one
  that never runs in tiles (a pair too large for the device is then resampled, as the old tile-less apps did):

  ```json
  "presets": {
    "composite":   {"description": "Coarse search, then Adam refinement, on MIND features", "set": []},
    "impact_cbct": {"description": "With TotalSegmentator features, for CBCT to CT",
                    "set": ["models.1.ref=VBoussot/impact-torchscript-models:TS/M730.pt", "models.1.layers_mask=01",
                            "regularization_weight=5.0"]}
  }
  ```

- **Chosen with `#`:** `VBoussot/ImpactReg:ConvexAdam#composite`, `ConvexAdam#composite` where the repository is
  implied, `host:port:app#preset|token` on a server. No `#` is the base config. A name is letters, digits, `_` and
  `-`; no `.`, which an impact-reg-konfai scope (`PRESET:NAME=VALUE`) would read as the start of a dotted name.
- **Applied first:** `Prediction.yml`, then the preset's `set`, then the run's own `--set`, then the direct options;
  the last value wins. One resolver checks them all (`apply_overrides`), and a preset that no longer resolves
  against its app is refused when it is chosen.
- **Method parameters only:** never the output format, the device, the batch or the patch size, so one preset gives
  one result whatever the run's own choices.
- **Flat:** no preset inherits another. One made from another copies its `set` (`preset save --from`), which keeps a
  tuned preset frozen when its parent changes.
- **Everything else belongs to the app:** the weights its presets may use (one pinned list), the memory per voxel
  (the costliest preset's), the tiling, the sample. The provenance of a tuned preset is written in its description;
  its tier is the benchmark's to record, in fidopt.
- **Where they live:** published in the app's `app.json`; a user's own in `~/.konfai/presets/<repo>/<app>.json`
  (`$KONFAI_PRESETS_DIR`), or in the `app.json` of a local app folder. A user's preset of a published name is
  refused: one's own never hides a published one. Publishing is copying an entry into the app's `app.json`.
- **A preset changes parameters, never weights.** An app that ships other weights is another app: ImpactSynth `MR`
  and `CBCT` have the same config and differ by their trained weights, and stay two apps, each with its own
  `trained_on` in fidopt.
- **No compatibility layer:** the old app names go.

### Settings

- `--set NAME=VALUE` is unchanged for a model parameter. A dotted name now starts at the model
  (`--set models.0.layers_mask=01`, `--set optimizer.lr=0.1`) unless its first key is one of the config's own roots
  (`Predictor.`, `Trainer.`), which keeps every full path working.
- Two direct options join `--batch-size`, `--patch-size`, `--tta`, `--max-voxels`, `--gpu`, `--cpu`, `--out`:
  `--output-format [OUTPUT=]FORMAT` (repeatable) and `--seed N`. Each is translated into the `--set` it stands for,
  read off the app's config, so nothing downstream changes; an unknown format or output is refused with the valid
  ones.

### What ran

Every inference writes `konfai-run.json` beside its outputs: the app, its revision (the Hugging Face snapshot), its
preset, every `--set` it applied in order, and an `identity` that every run of the same configuration shares.

## Implemented on this branch

| Commit | What |
|---|---|
| `feat(apps): run an app with one of its presets (APP#PRESET)` | presets, `show APP#PRESET`, `preset save`, user presets, server and client |
| `feat(apps): dot a nested model parameter from the model` | `--set models.0.layers_mask=01` |
| `feat(apps): --output-format and --seed for every app` | the two direct options, on infer, pipeline, fine-tune |
| `feat(apps): record what an inference ran beside its outputs` | `konfai-run.json` |
| `feat(mcp): take an app's presets (ref#preset)` | every app tool takes `#preset`, `describe_app` lists them |
| `feat(impact-reg): register with an app's preset (APP#PRESET)` | presets split between the global and tile passes, ensembles of full references, the direct options on every pass |
| `feat(apps): list each app's presets` | `list` shows them |

## The app repositories

The ten repositories SlicerKonfAI, SlicerImpactReg and fidopt read, and what presets change in each (the neighbours'
`Prediction.yml` compared):

| Repository | Task | Apps | What tells neighbours apart | After |
|---|---|---|---|---|
| VBoussot/ImpactReg | registration | 12 | parameters, over three engines | 3 |
| VBoussot/ImpactSeg | segmentation | 1 | | 1 |
| VBoussot/ImpactSynth | synthesis | 4 | the weights only (identical configs) | 4 |
| VBoussot/TotalSegmentator-KonfAI | segmentation | 4 | the weights (`total` 5 checkpoints, `total-3mm` 1) | 4 |
| VBoussot/MRSegmentator-KonfAI | segmentation | 1 | | 1 |
| VBoussot/DentalSegmentator-KonfAI | segmentation | 1 | | 1 |
| fideus-labs/fidopt-registration | registration | 13 | UniGradICON and MultiGradICON: one parameter (`model`) | 12 |
| fideus-labs/fidopt-segmentation | segmentation | 26 | TotalSegmentator and its 9 tasks: one parameter (`task`); Lungmask and Lungmask_Lobes: `model` | 16 |
| fideus-labs/fidopt-synthesis | synthesis | 12 | the 6 nnsyn: `source` and `region`; the 2 DFCAN: `structure` | 6 |

The ImpactReg engines and their presets, each a list of `set` over its engine's base (`Prediction.yml` of the app
named as the base):

| Engine, base | Preset | `set` entries |
|---|---|---|
| ConvexAdam, Composite | composite, coarse, impact_cbct, impact_mrct | 0, 1, 6, 7 |
| Elastix, Rigid_BSpline | rigid_bspline, rigid, impact_static, impact_jacobian | 0, 2, about 5 each as structured values (65 and 45 flat) |
| FireANTs, SyN | syn, impact, impact_mrct, anatomix | 0, 8, 18, 23 |

fidopt-registration's `ConvexAdam_MIND`, `Elastix_BSpline` and `FireANTs_Greedy` run impact-reg-konfai's engines
too, but stay fidopt's own baselines: the reference to the authors' code that the benchmark keeps apart.

## Open questions

- **Tiling (decided):** the five presets whose old app had no tile pass (ConvexAdam coarse, impact_mrct; Elastix rigid,
  impact_static, impact_jacobian) say `"tiles": false`.
  tile pass out for a preset without a fine stage, or whether the engine's tiling has to say so.
- **Keys a preset leaves out:** Coarse has no fine-stage keys; under the engine's base they stay at the base's
  values. To confirm they are inert without the fine stage.
- **Where fidlab's tuned presets live:** in the engine app beside the authors', or in a repository of their own.

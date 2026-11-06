# Presets inside an app, and direct options for shared settings

Status: proposal, 2026-11-06. Touches konfai-apps first, then impact-reg-konfai, SlicerKonfAI, SlicerImpactReg,
konfai-mcp, and two users of the apps: fidopt (the benchmark) and fidlab (its tuning lab).

## The problem

**One app per parameterization.** VBoussot/ImpactReg publishes 12 apps. They run three engines, and most differ from
a sibling by a few lines of `Prediction.yml`:

| Engine (`models.<engine>:RegistrationNet`) | Apps today |
|---|---|
| ConvexAdam | ConvexAdam_Coarse, ConvexAdam_Composite, ConvexAdam_IMPACT_CBCT, ConvexAdam_IMPACT_MRCT |
| Elastix | Generic_Rigid, Generic_Rigid_BSpline, Elastix_IMPACT_Static, Elastix_IMPACT_Jacobian |
| FireANTs | FireANTs_SyN, FireANTs_IMPACT, FireANTs_IMPACT_MRCT, FireANTs_Anatomix |

`Generic_Rigid_BSpline` is `Generic_Rigid` plus `Parameters_BSpline.txt` and `final_grid_spacing: 16.0`;
`ConvexAdam_Composite` is `ConvexAdam_Coarse` plus the `fine` stage. `Evaluation_*.yml` and `Uncertainty.yml` are the
same file in every app, and `requirements.txt` has one version per engine.

The same thing happens to users. SlicerKonfAI's "Save as local app" keeps a set of tuned parameters by exporting a
new app folder with them as defaults. fidopt names a line `<method>@<variant>`, and for IMPACT-Reg the variant slot
is taken by the app (`impact_reg@ConvexAdam_Composite`), so a tuned configuration of that app has no name of its own.

**Long `--set` paths for common settings.** konfai-apps reads a bare `NAME` as a model parameter
(`--set iterations=300`) and a dotted `NAME` as a full path from the config root. Everything outside the model's own
arguments takes the full path, the output format among them:

```bash
--set Predictor.outputs_dataset.DisplacementField.OutputDataset.dataset_filename=Transform:omezarr
```

A nested model parameter takes it too, since a dotted name starts at the root:
`--set Predictor.Model.RegistrationNet.models.0.layers_mask=01` (fidopt's `tuning.json`, konfai-apps' tests and CLI
reference).

## Proposal

### 1. Presets: named lists of `--set` inside an app

An app is an engine: its code, its base config (`Prediction.yml`), its shared files. A preset is a name for a list of
`NAME=VALUE` overrides of that base, declared in `app.json`:

```json
"presets": {
  "rigid":         {"description": "Rigid, Mattes mutual information",
                    "set": ["parameter_maps=[Parameters_Rigid.txt]"]},
  "rigid_bspline": {"description": "Rigid, then a B-spline on a 16 mm grid",
                    "set": ["parameter_maps=[Parameters_Rigid.txt,Parameters_BSpline.txt]", "final_grid_spacing=16.0"]},
  "hippocampus":   {"based_on": "composite",
                    "set": ["grid_spacing=3", "grid_shrink=1", "iterations=150"],
                    "tuned_on": "the 10 development pairs of fidopt's L2R_HippocampusMR",
                    "source": "https://github.com/fideus-labs/fidlab/pull/2"}
}
```

- **Reference.** `<repo>:<app>#<preset>`: `VBoussot/ImpactReg:ConvexAdam#composite`, or `ConvexAdam#composite` where
  the repository is implied (`impact-reg-konfai register ConvexAdam#composite`). No `#` means the base config.
- **Resolution.** `Prediction.yml`, then the `based_on` chain, then the preset's `set`, then the user's `--set`, then
  the direct options of section 3. A later value wins. The one resolver that checks every `--set` today
  (`apply_overrides`: the name exists or the model declares it, the value fits the declared type) checks the presets
  too, so a published preset that no longer resolves fails at load, not halfway through a run.
- **A preset holds the method's parameters only.** Never the output format, the device, the batch or patch size:
  those are the user's choices, and keeping them out means one preset gives one result whatever the user picks.
- **Every published configuration keeps a name,** the base one included (`"composite": {"set": []}`), so a reference
  never depends on which configuration the base happens to be.
- **The format is a list, shown as a command line.** A value with spaces, quotes or a JSON block
  (`models.1={"ref": "...", "voxel_size": [3, 3, 3]}`) needs no shell quoting in a list. The pass scopes that
  impact-reg-konfai already reads (`global:`, `tile:`) work inside an entry.
- **Per-preset resources.** The 12 apps also differ in `app.json`, not only in the config: `tiling`,
  `ram_bytes_per_voxel`, `vram_bytes_per_voxel`, `model_files`, `sample`. A preset may override those keys
  (`"app": {"tiling": {...}, "vram_bytes_per_voxel": 210}`). `model_files` could instead be derived from the
  `models.*.ref` the preset sets.
- **`#` stays out of URLs.** In a URL, `#` starts the fragment. A remote server's API takes the preset as a field
  of its own; the `#` form is for the command line, configs and identifiers.

### 2. Dotted names relative to the model

A dotted `NAME` whose first segment is a parameter of the model resolves inside the model block, as a bare name
does. A dotted name that starts at a root key (`Predictor.`, `Trainer.`) stays a full path. A name both could mean
is refused, and the error names the two paths.

```bash
--set models.0.layers_mask=01          # was Predictor.Model.RegistrationNet.models.0.layers_mask=01
--set optimizer.lr=0.1                 # was Predictor.Model.RegistrationNet.optimizer.lr=0.1
```

### 3. Direct options for what every app shares

konfai-apps already has `--batch-size`, `--patch-size`, `--tta`, `--mc`, `--ensemble`, `--max-voxels`, `--gpu`,
`--cpu` and `--out`. Two settings are missing, and both are on every config:

| Option | Writes | Seen in |
|---|---|---|
| `--output-format FORMAT`, or `OUTPUT=FORMAT` for one output | `Predictor.outputs_dataset.<OUTPUT>.OutputDataset.dataset_filename=<group>:<FORMAT>` | `<name>:mha` in the 5 apps of VBoussot/ImpactSeg and VBoussot/ImpactSynth, `Transform:itktransform` in the 12 of VBoussot/ImpactReg |
| `--seed N` | `Predictor.manual_seed=N` (`Trainer.manual_seed` when training) | `manual_seed` in each of the 12 registration apps |

```bash
konfai-apps infer VBoussot/ImpactSynth:MR ... --output-format mha
impact-reg-konfai register ConvexAdam#composite ... --output-format Transform=omezarr
```

Each option is translated into the `--set` it stands for, so nothing downstream changes. The format is checked
against the formats KonfAI already knows (`SUPPORTED_EXTENSIONS`, `SUPPORTED_BACKEND_FORMATS`: `mha`, `nii.gz`,
`omezarr`, `ome-zarr`, `nrrd`, `tif`, ...), and an unknown format or output is refused before anything runs, with
the list of valid ones. The output's precision (`TensorCast`) and inference `autocast` are candidates too, left out
until someone needs them.

## The 12 apps as presets

Each preset written as `set` entries over its engine's base (`Prediction.yml` of the app named as the base below).
Every difference is a model parameter: none touches the dataset, the outputs or the device.

| Engine, base | Preset | `set` entries | Notes |
|---|---|---|---|
| ConvexAdam, Composite | coarse | 1 | `stages=[coarse]`. The fine stage's keys, which Coarse leaves out, stay at the base's values and are inert without that stage: to confirm |
| | impact_cbct | 6 | a feature model (`models.1.*`), `regularization_weight=5.0` |
| | impact_mrct | 7 | the same, `stages=[coarse]` |
| Elastix, Rigid_BSpline | rigid | 2 | the rigid parameter map alone |
| | impact_jacobian | 45 | a feature model for each level (`levels.<i>.models.<j>.*`), its own grid and iterations |
| | impact_static | 65 | the same, two feature models a level |
| FireANTs, SyN | impact | 8 | `deformable_metric=impact`, one feature model |
| | impact_mrct | 18 | two feature models, LNCC, the feature patching |
| | anatomix | 23 | the anatomix features, its own iterations and step |

Nine of the twelve need 8 entries or fewer. The two Elastix IMPACT presets repeat a feature model block per level.
Three ways to write them:

1. **Structured values:** one entry a level, `levels.0.models=[{...}, {...}]`, which brings them to about 5 entries.
2. **An overlay file** for a preset that outgrows a list: `"file": "presets/impact_static.yml"`, the same keys as a
   config fragment, read by the same resolver.
3. **Keep them as apps of their own,** if the IMPACT Elastix pipelines are seen as methods apart.

The first keeps one format everywhere. The second is the escape hatch if a preset ever needs more.

## What changes where

| Where | Change |
|---|---|
| konfai-apps | reads `presets`, resolves `#preset` and `based_on`, model-relative dotted names, `--output-format`, `--seed`; lists the presets in its API and its server |
| impact-reg-konfai | `register ConvexAdam#composite`; an ensemble member is a full reference, so two presets of one engine can be ensembled (`ConvexAdam#composite` and `ConvexAdam#hippocampus`), and `PRESET:NAME=VALUE` scopes by that reference |
| SlicerKonfAI | a Preset list under the app; the parameter widgets start from base plus preset, and "differs from the default" compares against the preset, so its values are not sent twice; "Save as local app" becomes "Save as preset" (a name and a `set` list in a presets file of the user's); an Output format list that writes `--output-format` |
| SlicerImpactReg | the ensemble chips hold engine and preset pairs |
| konfai-mcp | `list_app_parameters` and `run_app` take a preset; `describe_app` lists them |
| fidopt | a line is `impact_reg@ConvexAdam#composite`; its IMPACT-Reg rows, off the boards while the presets are retuned (`"board": false`), are run again under the new names; the authors' presets are tier 1 and a preset with `tuned_on` is tier 2 (a protocol change, for the maintainers to decide); `tuning.json` becomes presets with a `source` |
| fidlab | an experiment searches the parameters the app declares (`get_parameters`: values and constraints) from a preset, and a winning one ends as one preset entry |

## Migration

No compatibility layer: the 12 old app names go, with nothing mapping them to the new references.

1. konfai-apps: presets, references, model-relative names, direct options. An app without `presets` is its base
   config, as every app is today.
2. VBoussot/ImpactReg: three engine apps (`ConvexAdam`, `Elastix`, `FireANTs`) with their presets, in place of the
   12 apps. The shared files are kept once per app.
3. impact-reg-konfai, then SlicerKonfAI and SlicerImpactReg.
4. fidopt, then fidlab.

## Open questions

- **The base of each engine:** the most used configuration (Composite, Rigid_BSpline, SyN, as above) or the simplest.
- **Keys a preset leaves out** (Coarse has no fine-stage keys): confirm they are inert, or let an entry reset a key to
  the model's own default.
- **The two Elastix IMPACT presets:** structured values, an overlay file, or apps of their own.
- **Where tuned presets live:** in the engine app beside the authors' ones, marked by `tuned_on`, or in a repository
  of their own; and how fidopt shows them.

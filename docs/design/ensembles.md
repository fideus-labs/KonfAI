# Ensembles: one notion of member for checkpoints, TTA and presets

Status: decided 2026-11-07 (see Decided), not implemented. Builds on `docs/design/presets.md` (presets, `APP#PRESET`) and on the
registration measures (`TransformPoints`, `Resample` through stored transforms).

## The problem

An ensemble runs a model several times and combines the runs. KonfAI has three kinds of runs and three ways to
combine them:

- **Checkpoints and TTA copies** are one predictor run. `ModelComposite` swaps each checkpoint's weights into one
  network (`konfai/predictor/ensemble.py`); `Predictor.combine` reduces the checkpoints, `OutputDataset.reduction`
  the TTA copies. With `Concat` at both levels and `InferenceStack` first after the reduction, the members land on
  one axis, `[M x T, C, ...]`, written as `InferenceStack.<ext>` beside the output; `konfai-apps uncertainty` reads
  that stack against the app's `Uncertainty.yml`.
- **Presets** cannot share that run: a preset changes how the model is built (ConvexAdam's `grid_spacing`), not its
  weights. Only impact-reg-konfai ensembles them: one `konfai-apps infer` per preset, a `Reduce(Mean)` of the fields,
  each member kept as `Ensemble/<preset>.h5` (`--keep-fields`), and an `uncertainty` command of its own
  (`Reduce(Std)` then `Magnitude`).
- **MC dropout** is refused by konfai-apps and never reaches the member axis; it comes back as a fourth kind of
  member (decided).

So a member is three different things, the stack has no defined layout for a member of several channels (a
displacement field's three components slide into a spatial axis in `.mha`, unchecked), and every uncertainty measure
of a registration ensemble (the spread of its landmarks, the disagreement of the segmentations it propagates) has
nothing to read: `Resample` and `TransformPoints` apply one stored transform per group, not each member of a stack.

## The design

### A member, an ensemble, a stack

- **A member is one run of a model:** an app reference with its preset (`APP#PRESET`), one checkpoint, one TTA
  copy, one MC-dropout draw. Two members may differ by any of the four, or by the app itself when their outputs
  match.
- **An ensemble is a list of members,** written as the product of what the run asks for:
  `konfai-apps infer APP#a APP#b --ensemble-models c1 c2 --tta 2` is 2 presets x 2 checkpoints x the TTA copies
  the run makes.
- **The stack is the one record of an ensemble's members,** and its only format: one volume per output whose
  channels are the members' channels, member-major (`[M x C, *spatial]`), its header carrying `konfai_members: M`.
  Any store that holds channels holds it (MetaImage, NIfTI, OME-Zarr), and a reader reshapes it to `[M, C, ...]`
  from the header alone. One member per channel (`C = 1`) is today's layout unchanged.
- **Which member is which is written once, in the run record** (decided): `konfai-run.json` lists the members in the
  stack's order (`APP#PRESET`, checkpoint, TTA copy, MC-dropout draw); the stack holds their count alone. Slicer names
  the members from the record.
- **The output is the stack reduced** by the app's own `combine` operator (`Mean`, `Median`, `Vote`), the one it
  already declares for its checkpoints. Members of several apps must declare the same one.

### Where each kind of member is run

- **Checkpoints, TTA copies and MC-dropout draws of one preset** stay in one predictor run, reduced there by a
  running sum as today, which writes nothing more for an ensemble of checkpoints without uncertainty (decided: the
  two paths stay, one format and one operator between them). With `-uncertainty`, or as a member of a preset
  ensemble, the run writes its rows as it makes them (`InferenceStack.fold_copies` already streams them).
- **Each preset (each app reference) is a run of its own,** whose rows are appended to the same stack. The output is
  reduced once from the whole stack, so a mean weighs every member alike and a vote counts every member, where a
  reduction of per-preset results would not.
- **The stack is kept with `-uncertainty`,** and is a temporary of the reduction otherwise: nothing is written
  twice, and `Ensemble/*.h5` goes.

### Reading a stack: a member at a time

- **Pointwise stages read the member axis already** (`StandardDeviation`, `Variance`, `SegmentationDisagreement`):
  they take the header's member count instead of guessing it from the shape.
- **`Resample` and `TransformPoints` apply each member of a stacked transform:** given a stack of fields (or of any
  stored transforms) under `transforms:`, they return a stack, one result per member, each member read by region as
  a single field is. This is the one new capability of the core.
- **Spread criteria** read a stack against nothing or against a reference: the summary of a spread map (`Mean`, a
  95th percentile), the spread of points pushed through every member (mm, per landmark), and the calibration of a
  spread against the error it predicts (the correlation between per-landmark spread and TRE).

### Uncertainty is an evaluation of the stack

`konfai-apps uncertainty APP -i STACK [--gt ...]` stages as `eval` does: the stack is `Volume_0`, the references
`Reference_i`. An app ships `Uncertainty*.yml` the way it ships `Evaluation*.yml`. For a registration engine:

| Config | Inputs | Measures |
|---|---|---|
| `Uncertainty.yml` | `-i STACK` | the spread of the displacement vectors (std per component, then the norm), its mean and 95th percentile |
| `Uncertainty_with_fid.yml` | `-i STACK --gt FIXED_FID [--gt MOVING_FID]` | the spread of the fixed landmarks pushed through every member; with the moving ones, its calibration against the TRE |
| `Uncertainty_with_seg.yml` | `-i STACK --gt MOVING_SEG` | the disagreement of the moving segmentation pulled through every member onto the fixed grid |

## What moves out of impact-reg-konfai

| Today in impact-reg-konfai | After |
|---|---|
| `_ensemble_mean`, the `Ensemble/` folder, `--keep-fields` | the stack, written by konfai-apps' ensemble; `-uncertainty` keeps it |
| `uncertainty` (`Reduce(Std)`, `Magnitude`) | `konfai-apps uncertainty` with the engine's `Uncertainty.yml` |
| `evaluate` warping, landmarks, Jacobian | done: `konfai-apps eval` (#311) |
| `plane.py` for 2-D pairs | to check engine by engine, now that KonfAI stores and applies 2-D transforms |

It keeps what is registration's own: the global pass, the native-resolution tiles and their composition, splitting a
preset's `--set` between the two passes, the moved image, and `apply`. Each preset member is one of its multi-pass
registrations; its result goes into the stack as any member's does.

## Slicer

- SlicerKonfAI composes an ensemble as the CLI does, a product (decided): a row of preset chips (`APP#PRESET`) beside
  the checkpoint chips it has, and the TTA count; every combination is a member. For an app without checkpoints (the
  registration engines) the preset chips are the whole choice. SlicerImpactReg's own preset chips go.
- The Uncertainty panel hands konfai-apps the stack it wrote, instead of re-stacking the sequence nodes into
  `Volume.mha`, and offers the app's `Uncertainty*.yml` tabs with their `--gt` entries, as the evaluation tabs do.

## Phases

1. The stack format (member-major channels, `konfai_members`), its readers and the pointwise stages on it.
2. Ensembles of app references in `konfai-apps infer`, one run per reference, one reduction of the whole stack.
3. `Resample` and `TransformPoints` through a stack; the spread and calibration criteria; `uncertainty --gt`.
4. The engines' `Uncertainty*.yml`; impact-reg-konfai writes its members to the stack and loses its own ensemble and
   uncertainty.
5. SlicerKonfAI and SlicerImpactReg.

## Decided

- The in-run reduction (checkpoints, TTA, MC dropout) and the stack's (presets) both stay: one format, one operator.
- MC dropout comes back as a fourth kind of member.
- The members are named in `konfai-run.json`, in the stack's order; the stack holds their count.
- Slicer composes an ensemble as a product, as the CLI does.

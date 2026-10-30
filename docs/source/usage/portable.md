<!-- SPDX-License-Identifier: Apache-2.0 -->
# Run a KonfAI workflow without Python

`konfai.portable` compiles a TRANSFORM or PREDICTION config into a versioned JSON
program. The portable runtime executes that program on native platforms or in a
browser. Python is needed to compile the config and export the models, but not
to execute the compiled program. Training and evaluation remain Python workflows.

This is a source-build feature prepared for 1.9 (state of 2026-09-26). The runtime, KonfAI Edge (crate `konfai-edge`), is
distributed apart from the Python package, with its own README for the build
requirements and the executables. The Python wheel includes `konfai/portable.py`;
it does not contain a native executable, a wasm module, model weights, or the ONNX
export dependencies. Install the `export` extra when compiling a prediction. The
`konfai_edge_transform`, `konfai_edge_prediction` and `konfai_edge_manifest` keys name the
runtime that reads these programs, KonfAI Edge, and carry their version.

| Program | Key | Version | Read by |
| --- | --- | --- | --- |
| TRANSFORM | `konfai_edge_transform` | 4 | KonfAI Edge at 2026-09-26 (unreleased) |
| PREDICTION | `konfai_edge_prediction` | 2 | the same |
| one model's manifest | `konfai_edge_manifest` | 4 | the same |

A bump on one side is a line in this table and in the runtime's README.

## Compile, inspect, execute

Work in the directory where your config's dataset paths resolve. Compilation
builds the KonfAI workflow first and reports its config errors before producing a
program. **It resolves defaults back into the config file.** Use a copy beside
the original if you need to preserve the authored YAML and its relative paths.

```python
from konfai.portable import compile_transform

compile_transform("Transform.yml", output="transform.json")
```

The runtime's TRANSFORM executable takes the program and the root the dataset
paths resolve against. With `--plan` it prints the plan and writes nothing: each
route is named and priced against the memory budget (`STREAM`, `LOAD`,
`WHOLE-VOLUME`, `REDUCE`, `SKIP`, or `REFUSED`). A whole-volume fallback states
which stage requires it. A refused plan writes no output. `--memory-budget BYTES`
overrides the budget the program carries.

A prediction also needs the checkpoints to export. The compiler writes their
ONNX files beside the JSON program. Keep them together when moving the program.

```python
from konfai.portable import compile_prediction

compile_prediction(
    "Prediction.yml",
    models=["Checkpoints/experiment/best.pt"],
    output="portable/prediction.json",
)
```

The runtime's PREDICTION executable takes the same two arguments and a backend:
ONNX Runtime loads the exported models as they are; a compiled-model backend
imports the architectures at build time and loads the corresponding weights at
run time, so a different architecture needs another build. The runtime's README
names the backends and the features each build enables.

Existing outputs are skipped. Run the same command again to resume, or pass
`--overwrite` to recompute. Dataset roots are resolved against `--root`; outputs
follow the compiled destinations. Inspect those destinations before executing.

The program's format is a JSON Schema, `konfai.portable_schema.program_schema()`,
committed beside the runtime's test fixtures. Every op a step may name, the keys
it may carry and the ones it must, are in that schema: a key outside it is one no
compiler writes, and the runtime refuses it by name rather than reading around it.

## What the compiler warns about

A refusal is for a chain the two engines would read differently. Where the
difference is known and an author may want it, the compiler warns instead, by
key, as a `KonfAIWarning`:

| Key | What konfai does | What the program does |
|---|---|---|
| `Predictor.autocast: true` | Runs the forward in float16 | The ONNX graph runs in float32 |
| `Resample.precision: fast` | Computes the sampling coordinates in float32 | Computes them in float64; a voxel at a rounding boundary can read its neighbour |
| `InferenceStack` (`dataset`, `name`) | Writes every stacked member under the group `InferenceStack` | Writes the fold only |
| `Reduce.provenance: true` | Records the operator and the cases folded in the output's header | Writes the geometry alone |
| `OutputDataset.attributes` | Writes the listed entries into the output's header | Writes the geometry alone |
| `channels_last`, `cudnn_benchmark`, `torch_compile`, `gpu_checkpoints`, `data_log`, `check_training_transforms`, `checkpoint_cache_gib`, `Dataset.num_workers`, `pin_memory`, `prefetch_factor`, `persistent_workers` | Change how konfai runs, and nothing it writes | No counterpart; one grouped warning names the keys that left their default |
| A test-time `prob` or `prob_permute` share strictly between 0 and 1 | Draws the copies with torch's generator | Draws them with Python's generator from `manual_seed`: reproducible copies, not konfai's |

Parameters konfai reads that change no value in the compiled context, and so
raise no warning: `inverse` on a TRANSFORM stage (a TRANSFORM never undoes a
stage);
`Rotate.a_min` and `a_max` under `is_quarter: true` (a quarter turn draws from
90, 180 and 270 degrees whatever the bounds); `Write.downsample_method` and
`Save.downsample_method` (they average pyramid levels, and `scale_factors`, which
builds the levels, is refused).

## Coverage

The registry is closed: a stage outside it, a class of your own included, is
refused by name. The tables list every stage konfai ships.

### Transforms

| Transform | Portable form |
|---|---|
| `Argmax` | `argmax`; `dim` must be 0 |
| `Canonical` | `canonical`; undone after the forward when `inverse` |
| `Clip` | `clip`; a data-dependent bound reads the `mask` group beside the case |
| `Crop` | `crop`; refused after a stage that moves the grid or the values |
| `Dilate` | `dilate` in a TRANSFORM chain or on a nested model's group; refused elsewhere in a prediction |
| `Expand` | The chain's structure: the steps each copy runs, the copies' entry names and the seed |
| `FlatLabel` | `relabel` |
| `Flatten` | Refused by name |
| `Flip` | `flip`; undone after the forward when `inverse` |
| `Foreign` | Refused by name |
| `Gradient` | `gradient` |
| `HistogramMatching` | `histogram_match`; reads the reference group beside the case |
| `InferenceStack` | The program's reduction, at the head of `after_reduction_transforms` over concatenated copies; refused elsewhere |
| `Magnitude` | `magnitude` |
| `Mask` | `mask` in a TRANSFORM chain; in a prediction, a step over a nested model's group in `before_reduction_transforms`; refused elsewhere |
| `MergeLabels` | The program's `merge_labels` reduction, at the head of `after_reduction_transforms`; refused elsewhere |
| `Norm` | `norm` |
| `Normalize` | `normalize`; undone after the forward when `inverse` |
| `OneHot` | `one_hot`; refused when `inverse` (its inverse is an argmax) |
| `Padding` | `pad`; constant modes only; undone after the forward when `inverse` |
| `Percentage` | `percentage` |
| `Permute` | `permute`; undone after the forward when `inverse` |
| `Reduce` | The chain's structure: the fold between the steps each case runs and the steps run once |
| `Resample` | `resample` onto a spacing, a shape or, in a TRANSFORM, a reference image; undone after the forward when `inverse` |
| `Save` | `save`, naming its destination; in a prediction, no step (see below) |
| `SegmentationDisagreement` | Refused by name |
| `SelectLabel` | `relabel` |
| `Softmax` | `softmax`; `dim` must be 0 |
| `Squeeze` | Refused by name |
| `StandardDeviation` | Refused by name |
| `Standardize` | `standardize`; a measured statistic reads the `mask` group beside the case; undone after the forward when `inverse` |
| `Statistics` | No step: it records a fact and alters no value |
| `Sum` | Refused by name |
| `TensorCast` | `cast`; undone after the forward when `inverse`, back to the stored dtype (refused past a reduction) |
| `UnNormalize` | `unnormalize` |
| `Variance` | Refused by name |
| `Write` | The chain's destination |

### Draws

A prediction's test-time augmentation carries `Flip`, `Permute` and a quarter-turn
`Rotate`, each undone on the copy's output before the reduction; a TRANSFORM
chain carries `Brightness` after an `Expand` marker. Every other draw is refused
by name, in either place.

| Draw | Portable form |
|---|---|
| `Brightness` | `brightness`, after an `Expand`; refused with `groups` |
| `ColorTransform` | Refused by name |
| `Contrast` | Refused by name |
| `ContrastAroundMean` | Refused by name |
| `CutOUT` | Refused by name |
| `Elastix` | Refused by name |
| `EulerTransform` | Refused by name |
| `Flip` | `flip`, with `vector_field` negating the mirrored components |
| `Foreign` | Refused by name |
| `Gamma` | Refused by name |
| `GaussianBlur` | Refused by name |
| `GaussianNoise` | Refused by name |
| `HUE` | Refused by name |
| `LumaFlip` | Refused by name |
| `Noise` | Refused by name |
| `Permute` | `permute`; with `prob_permute: None`, exactly two copies |
| `PlacedDraw` | Refused by name |
| `PlacedMask` | Refused by name |
| `Rotate` | `permute` and `flip` with `is_quarter: true`; free angles refused |
| `Saturation` | Refused by name |
| `Scale` | Refused by name |
| `SimulateLowResolution` | Refused by name |
| `Translate` | Refused by name |

### Reductions

| Reduction | As a `Reduce` operator | As the copies' `reduction` | As the models' `combine` |
|---|---|---|---|
| `Mean` | `mean` | `mean` | `mean`, when no `before_reduction_transforms` runs |
| `Median` | `median` | `median` | Refused with several models |
| `Std` | `std` | Refused | Refused with several models |
| `Vote` | `vote` | `vote` | Refused with several models |
| `Concat` | `concat` | `concat` | `concat`, then an `InferenceStack` or a `MergeLabels` |

konfai combines several models inside the forward, on each batch of patches,
before the overlap blend; a program folds each model's blended output. Only a
mean commutes with the blend. With a `before_reduction_transforms` chain, the
program averages each copy's models and runs the chain on that average, as
konfai does; the copies then fold by the output's `reduction`.

## Refusals, by name

Each refusal names the stage and says what konfai does instead, so the config can
be changed or run with konfai itself.

In either workflow:

- A stage outside the registry, and the stages of the tables above marked
  refused: `Squeeze` (konfai writes the entry back on the grid it kept),
  `Flatten` (ITK refuses the long axis it leaves), `Variance`,
  `StandardDeviation`, `SegmentationDisagreement` (they spread a stacked group's
  members, which a `Reduce` or a reduction folds instead), `Sum` (it sums the
  models' channels or a plain axis depending on the case), `Foreign` (only
  Python runs it).
- `Standardize` or `Normalize` with `lazy: true`: the statistic is recorded for a
  later stage, which the program does not carry.
- `Normalize` with `channels`: konfai rescales the named channels by their own
  extrema, the runtime every channel by the volume's.
- `Standardize` with several values for `mean` or `std`: the runtime applies one
  number to every channel.
- `Clip` with `save_clip_min` or `save_clip_max`: konfai records the bound as the
  volume's statistic, which a later `Normalize` reads in place of the extrema.
- `Padding` with a mode other than `constant` or `constant:<value>`.
- `Resample` with `field`, `field_group` or `transforms`, with an `align` other than `extent`,
  with both or neither of `spacing` and `shape`, with an interpolation other
  than `nearest`, `linear` or `cubic`, with a `reference` outside a TRANSFORM,
  or with a reference whose group konfai cannot tell. The runtime reads `cubic`
  only between grids whose axes keep their own (no permuted or oblique
  direction) and refuses it otherwise.
- A `Crop` after any stage that moves the grid or the values (`Clip`, a cast to
  an integer, `Mask`, `Flip`, `Permute`, `Resample`, `Padding`, another `Crop`):
  konfai crops to the box of the group as stored, the runtime to the box of the
  volume in hand. A cast to a float, a `Statistics` and a `Save` move nothing.
- `Softmax` or `Argmax` with `dim` other than 0.
- `Mask` with a `path` ending in `.mha`: the runtime reads a mask stored as a
  group of the case.
- A group with a `*` component, as a source group, a mask or a path: konfai
  resolves it by listing the dataset, the runtime reads the entry named.
- A dataset read at a resolution level other than 0 (`store:omezarr@2`).
- `Write` or `Save` with `scale_factors`; a `Write` naming no dataset; a chain
  that does not end with a `Write`; an empty chain.

In a TRANSFORM:

- `Reduce` with a `Save` in its chain, with an operator outside the five
  reductions, or writing a format konfai cannot write region by region (`mha`,
  `nii`, `omezarr` and `h5` can be); a stage after the fold that reads across
  space (konfai's own refusal).
- Any draw other than `Brightness` after an `Expand`; `Brightness` with
  `groups`.

In a PREDICTION:

- A model tiling its input by a `ModelPatch`: the export runs the whole patch
  through the network.
- A `patch_size` with an axis at zero or below.
- No group with `is_input: true`.
- A test-time draw other than `Flip`, `Permute` or `Rotate` with `is_quarter`;
  `Permute` with `prob_permute: None` and a copy count other than 2.
- `OneHot` with `inverse: true`: the runtime undoes `cast`, `resample`, `pad`,
  `canonical`, `flip`, `permute`, `crop`, `standardize` and `normalize`, and the
  inverse of a one-hot is an argmax. The inverse of a `cast` restores the stored
  dtype, which no program op names, so past a reduction (several models or
  test-time copies) it is refused rather than hoisted.
- A program stage nothing builds: `Mask`, `Dilate`, `KonfAIInference`,
  `InferenceStack` or `MergeLabels` anywhere but where the compiler builds a
  step for it (the two reductions at the head of `after_reduction_transforms`,
  a `Mask` over a nested model's group in `before_reduction_transforms`, a
  `Dilate` on that group). A stage bound with no parameters (`Reduce:`) in a
  prediction chain.
- A stage that is neither a runtime op nor a pointwise torch transform (which
  the export folds into the graph); a runtime op after a folded transform.
- `patch_transforms` with a stage that is not pointwise or that is undone.
- A second input group whose chain runs a `resample`, `pad`, `crop`,
  `canonical` or `permute` (every group is patched on the first one's grid), or
  that carries `patch_transforms`.
- Several models combined by anything but `Concat`, or by `Mean` with a
  declared reduction; test-time copies reduced by a different reduction than
  the models are combined by, unless a `before_reduction_transforms` chain runs
  on the models' mean; copies reduced by `Std`.
- An `InferenceStack` over copies not concatenated (the models by `Concat`, and
  with test-time copies the output's `reduction: Concat` too).
- A `MergeLabels` with test-time passes, or, with several models, followed by a
  stage other than undoing an index selection or a nearest resample: the
  program merges each model's finished label map.
- A same-class ensemble whose run-once tail undoes anything but one resample,
  or runs an op other than `softmax`, `argmax` or `cast` past the reduction.
- A nested model's group with a stage other than `KonfAIInference`, `Resample`,
  `Dilate` or `Save`; a `Mask` reading a group no nested model builds; a nested
  `Resample` stating no interpolation when the nested chain does not end on a
  cast; a nested model with several checkpoints; an ensemble export with no
  declared reduction (both through `konfai-apps`).

A `Save` in a prediction's input chain compiles to no step: konfai writes what
reaches it and, on a later run, reads that entry in place of the stages before
it; the program writes no cache and runs them again. In the output chains a
`Save` hands its input back and writes nothing, in both engines.

Formats: NIfTI with more than four axes (three spatial axes and an optional
channel axis are read, not a time series or a vector image), VTK legacy, a
single-file DICOM input, DICOM JPEG-LS and JPEG 2000 codecs, and a non-scanner
NIfTI whose sform and qform agree within ITK's tolerance without being identical,
differ by close to it, or state the same axes in another sign or order (ITK
settles them through its SVD's sign conventions; a scanner sform, identical forms
and forms that clearly disagree are read).

## Routes the plan declares slow

These routes run, and the plan names them (`reads ... whole`) so a case that does
not fit the budget is refused before anything is written:

- A `.npy` volume is read whole.
- An HDF5 store written before konfai's version 2 layout is copied whole.
- The companion a statistic reads (the `mask` of a `Clip` or a `Standardize`,
  the reference of a `HistogramMatching`) is read whole.
- A streamed prediction goes through a working store and a second pass.
- A `vote` reduction costs the square of the number of copies.

Native streaming reads regions of OME-Zarr stores, contiguous HDF5 store
entries, ITK transform fields, uncompressed NIfTI and MetaImage files, and
DICOM series, and writes regions into OME-Zarr stores, HDF5 store entries, ITK
transform fields, uncompressed NIfTI files and inline MetaImage files.
Streaming prediction currently takes a single-model program. Gzipped NIfTI,
compressed MetaImage, and the other formats use a priced whole read or write.
Sample values are held in float32, so float64 and wide integer payloads can lose
precision.

## Precision and reproducibility

Use the runtime's `--precision float32 --host-tail` for the reference comparison
route. The default tail keeps suitable operations on the device; its float32
reductions can round differently from the host. `float16` is an optional
execution mode, not the parity baseline. It requires compatible operators and,
in the browser, an adapter granting `shader-f16`.

Compare image geometry exactly: shape, spacing, origin, and direction. Compare
values separately and record the backend, precision, model digest, and whether
the run was cold or warm. KonfAI's prediction output rounding is not applied by
the portable runtime. Argmax maps can differ at nearly tied classes even when
logits are close; a percentage alone does not explain those differences.

Keep the JSON program, models, executable revision, device/driver description,
and any `<output>.autotune.json` record beside a measured result. That record
describes the tuning cache used by the native GPU backend; it does not embed a
runnable cache in a browser bundle. `--device-budget` limits managed prediction
buffers, not the total GPU allocation including tuning and allocator reserves.

## In the browser

The browser accepts local dataset files and runs the same pipeline with a CPU or
a WebGPU backend. It decodes each case whole. On WebGPU a prediction keeps each
model's blend on the GPU, with the operations the GPU runs after it, and reads
the rest of the output chain back a slab at a time. A test-time copy is flipped
back and averaged on the GPU too, and the operations after the average read
back a slab at a time. The plan prices that route
(`STREAM`) at what the tab holds and names what the GPU holds beside it; a model
whose blend the GPU cannot hold is priced on the CPU route. It refuses store
entries, OME-Zarr directories, DICOM series, and reference-image resampling by
name. The browser prices a case from its header before its voxels where the
format keeps one apart (NIfTI, MetaImage, NRRD, TIFF, npy, GIPL, an ITK
transform), and after decoding it otherwise; it does not reproduce the native
streaming plan or bound all JavaScript copies and GPU allocations. Use a
measured, nonempty small case for a live demo and keep a saved result as a
fallback.

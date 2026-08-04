# Changelog

Drafted from the commit history by [Commitizen](https://commitizen-tools.github.io/commitizen/), then
edited. Each version's section is what the GitHub Release for that tag carries.

```bash
cz changelog --unreleased-version vX.Y.Z --start-rev v1.5.8   # a DRAFT, not the final file
```

Conventional Commits started at `v1.5.9`; rendering further back produces empty version headings.

The draft is a starting point, not the answer. It sees only commit subjects, so a squash merge
collapses to one line, a subject with no conventional prefix is dropped entirely, and a subject
written for a reviewer ("what reviewing the data surface turned up") tells a reader nothing. Take the
draft, then say what a user of the package gets that they did not have -- and re-read the section
against the commits that landed *after* you drafted it. Running the command over a section already
written replaces it.

## Unreleased

### ✨ Features

- **transform**: one `Resample`. `ResampleToResolution`, `ResampleToShape`, `ResampleToReference`,
  `ResampleTransform` and `Warp` were five stages answering two questions between them, each with a
  sampler of its own. They are now five spellings of one stage that asks the questions separately:
  **which grid to write on** (nothing, `spacing`, `shape`, or `reference` — a stored image's grid
  adopted whole) and **what map to write it through** (`field`, `transforms`, or neither). Every
  combination is legal, and asked for together they compose into **one interpolation** instead of
  two. The old names still work and are thin argument translations.
- **transform**: `align` says where a `spacing` or `shape` grid sits — `extent` (the default) keeps
  the field of view, `origin` keeps voxel zero's centre. This was decided silently before, and
  differently by the data and by the header.
- **transform**: a resample streams whatever it is asked for, and on the volume's device. Applying a
  stored registration used to hold a whole volume — not because a warp needs one, but because
  nothing on a `sitk.Transform` said how far it reached. A rigid or affine map now bounds exactly; a
  BSpline and a displacement field bound by the largest of their values, which holds at every point
  rather than at the sampled ones. Everything is evaluated in torch, so nothing marshals a
  GPU-resident case out to numpy and back.
- **transform**: a resample no longer requires the grids to share a direction. A rotated reference,
  or a field stored on turned axes, used to be refused with an instruction to run `Canonical` first
  — a second interpolation of the same voxels. Both now work directly.
- **transform**: a field is read on its own grid. `Warp` required the field and the case to share
  one; a field solved at 120 µm now moves a volume stored at 30 µm without being upsampled first.
- **transform**: `ResampleToShape` needs no geometry at all. A count is a count; only a change of
  density needs the density it starts from.

- **transform**: the plan chooses the route from predicted cost against the memory budget.
  Streaming is a memory strategy — splitting re-reads, loading whole reads once — so a case that
  fits the budget is now `LOAD`ed when streaming would re-read the source (a halo re-reads its
  overlap, a regrid pulls each slab's window through its map, a compressed store decodes the whole
  volume per slab — measured up to 9.7x on an oblique map). The plan prints the predicted factor;
  `on_fallback` has nothing to say about a choice. The predictor's streamed-vs-assembled route now
  prices the config's budget instead of the machine's free memory at that moment, so the same case
  takes the same route on a loaded machine and an idle one.
- **transform**: a console that says what changes a decision. A healthy 6-output run is 7 lines —
  the plan's chain lines now carry the stages and the terminal `Write` destination, and one final
  line states what was written, how, in how long, and where `outputs.json` is. The run itself only
  speaks when it deviates from the printed plan. A designed refusal (`on_fallback: error`, a budget
  overrun) prints its message and remedy and exits 1 — 34 lines of framework traceback down to 6;
  `KONFAI_DEBUG=1` re-attaches the traceback. Logs stop amplifying progress frames (~4x smaller),
  and every byte figure prints at the unit that carries digits instead of `0.00 GiB`.
- **transform**: every configuration-dependent fallback says what to change — a masked `Clip` or
  `Standardize`, a percentile bound, a spatial `Sum`/`Argmax`/`Softmax`, an oblique `Canonical`, a
  free-angle `Rotate` draw, a vector-field `Flip` — and `Statistics` streams: its four numbers are
  the disk scan's own, seeded instead of recomputed per region.

### 🐛 Fixes

- **transform**: a resampled label map no longer comes out shifted against the image beside it.
  `F.interpolate`'s nearest reads `floor(o * scale)` where its linear reads `scale * (o + 0.5) - 0.5`,
  so a mask resampled by the same stage as its CT lagged it by `(scale - 1) / 2` source voxels —
  2.5 voxels, 1.25 mm of anatomy, resampling 0.5 mm to 3 mm. Both volumes were entirely plausible on
  their own. Nearest is now ITK's round-half-up on the same physical index the linear sampler reads.
- **transform**: the header a resample records now describes the grid it actually sampled.
  `ResampleToResolution` wrote the spacing that was *asked for* while sampling at `n_in/n_out` times
  the source's (up to a millimetre of drift across a volume) and left the `Origin` alone while
  sampling half a spacing-change away from it. Nothing downstream could see either: the voxels are
  all real, and the header was the only witness.
- **transform**: a voxel count no longer loses a slice to floating point. 90 voxels of 0.7 mm re-cut
  at 1.5 mm is 42, and the count went through float32 to get there — landing on 41 or 42 depending
  on the numbers.
- **transform**: a warp on an oblique case reads the neighbourhood it needs. The halo was derived
  per array axis from a world displacement, which assumes the direction cosines are the identity; on
  a turned case the window was short on the axes the displacement actually reached, and a short
  window returns the border value rather than raising.
- **transform**: a resample refuses what it used to do quietly. A refusal the whole-volume path can
  serve — an undeclared field bound, a case with no geometry — declares `WHOLE_VOLUME` with the
  sentence saying what to change, and the run proceeds assembled. A map neither route can apply —
  an unsupported spline order, a missing entry, `invert: true` on a spline or a field — refuses as
  the plan is built, before a byte is written; it used to print a fallback the run then contradicted
  by dying per case.
- **transform**: `ResampleTransform`'s `inverse` defaults to `false`. It always raised
  `NotImplementedError`, so a prediction finalize through this stage failed at the end of the run
  rather than at its configuration.

- **data**: a stage is judged on the state the stages before it left, on every landing fold. A
  `Resample` behind a `Canonical` recorded the pre-reorientation grid and resampled the wrong axis
  — silently, every voxel real, on the exact chain the published TotalSegmentator bundle ships —
  and a second `Resample` saw the original spacing and handed its input through as a no-op. Both
  now stream, bit-identical to the whole-volume pass.
- **data**: the end plane of a BSpline's valid region is warped as ITK warps it. A grid
  commensurate with the coefficient mesh — what a fitted transform domain produces — hit that
  plane whole planes at a time, every voxel silently unmoved (2.66 mm of displacement dropped,
  against 1e-14 agreement everywhere else).
- **data**: coverage is judged through the declared map before a case is refused as disjoint. An MR
  and a CT 1000 mm apart in stage coordinates with a stored rigid bridging them — the situation the
  apply step exists to serve — were refused as writing nothing but fill.
- **data**: a half-precision volume on CUDA blends through float32 coordinates. The fused blend
  built its sampling grid in the payload's dtype, quantizing a coordinate at ~2^-11 of the window —
  0.06 voxel on a 512 axis, tens of units at a sharp edge (measured 60.0 on a 1000-range fixture,
  0.022 after).
- **data**: interleaved patch reads of two `Expand` copies each keep their own grids; re-reading a
  copy after another was planned handed it the other copy's sampling.
- **transform**: `Clip('min'/'max')` clips to the case's seeded statistic, not the region's own —
  what `save_clip_min`/`save_clip_max` recorded used to depend on which patch happened to run.

### ⚡ Performance

- **data**: a map that factorises is read one axis at a time on global coordinates — most
  resamples, bit-identical streamed or whole (CT-sized case: CPU 2269 -> 160 ms, GPU 46.5 -> 2.1 ms,
  peak 1.09 -> 0.35 GiB) — and the axis that shrinks most is blended first (9.1x on a thick-slice CT
  brought to isotropic). A map that does not factorise (a warp, a rotation, a stored field) goes
  through one fused `grid_sample` kernel (4x on a warp); on that path a streamed region and the
  whole volume agree to ~1e-5 of the data's range rather than bit for bit, which the plan notes
  when a budget shrinks the slabs.

### 🔧 Internals

- **data**: `LocalityKind.RESCALE` is gone. It was the dispatcher's own resample map — a size ratio,
  which says nothing once a target grid has an origin — and with one resample stage there is one
  regime, `REGRID`, that the stage owns both halves of.
- **data**: the sampler owns its rules (`nearest_index`, `window_index`, `sampling_dtype` live in
  `sampling.py`), `utils/ITK.py` keeps only its live decoders (~340 orphaned pre-unification lines
  deleted), and `KONFAI_STREAM_LINEAR_RESAMPLE` — documented, read by nothing — is out of the docs.

## v1.8.0 (2026-08-04)

### ✨ Features

- **transform**: resample a case onto the grid of a declared reference
- **transform**: compose the grid change and the warp into one pass
- **data**: give a chain a per-component statistic
- **transform**: let Warp read the bound the fields recorded, with max_displacement: auto
- **studio**: show a transform run, and point Browse at the data rather than its log
- **mcp**: let an agent plan and run a transform, and read the table for what it accepts
- **transform**: a fifth workflow that reads a dataset, applies a chain, and writes it
- **transform**: give ResampleToReference an `interpolation`, as Warp already had
- **data**: Vote, the reduction operator that folds segmentations without inventing a label
- **data**: declare an OME-Zarr pyramid from a Write, and let a field carry its own bound
- **impact-reg**: seed the rigid from the centre of mass, not only the frame
- **impact-reg**: `--tmp-dir` on register/eval/uncertainty, the option the other app CLIs already carry —
  a caller whose system temp directory is a tmpfs can now stage volume-sized intermediates on real disk
  instead of overriding `TMPDIR` from outside; the same change also writes the moved image and the
  displacement field once per run instead of twice
- **impact-reg**: a registration preset now owes exactly one output, its displacement field, in whatever
  format it declares — `register` derives the moved image from it instead of expecting a second output.
  A preset can drop `MovedImage` entirely, which for a tiled one also drops blending a full-size moved
  across every patch seam for a caller that has the field. Reading the moving image handles an OME-Zarr
  store as well as an ITK file, which fixes the ensemble path too: averaging several presets over
  OME-Zarr inputs failed there, and nowhere else, on `sitk.ReadImage`
- **studio**: bundle icons through the app interface, and a way to stop Studio (#75)
- **examples**: a Transform example -- a template folded out of a cohort, and drawn copies of a case

### 🐛 Bug Fixes

- **transform**: sample a label map by nearest on the warped path too, instead of blending labels
- **transform**: check Warp's declared bound on the whole-volume path, not only the streamed one
- **transform**: build Warp's grid on the volume's device, so a GPU-resident case does not raise
- **data**: cap an OME-Zarr chunk instead of taking the writer's whole trailing plane
- **data**: keep the store's chunking when a Write appends pyramid levels
- **data**: refuse a statistic after a Reduce that an earlier post stage invalidates
- **data**: budget a reduction for what its operator allocates over the buffer it holds
- **data**: refuse a geometry key `grid: strict` cannot compare, instead of skipping it
- **data**: stop a chained sweep at the first failure, so the recorded reason is the cause
- **data**: record the landed state only for a plan that holds
- **data**: restore the CUDA generators a draw's seeding touched
- **transform**: name the argument the stage actually takes in a field refusal
- **transform**: count a chain's Reduce markers where its Expand markers were already counted
- **konfai-mcp**: require the konfai that has the module it imports at load
- **studio**: refuse /api/quit when a proxy header arrives and the peer cannot be trusted
- **examples**: install scikit-image where SSIM is evaluated, and ask for a GPU that exists
- **transform**: hold the auto bound to the cohort, and plan a resampled case on the grid it lands on
- **transform**: refuse a stage key that names something not a class
- **transform**: print the plan's reduction and dropped lines once
- **transform**: give refusals their true remedy and name
- **transform**: hold on_fallback error at run time too
- **transform**: refuse unknown roots and typo'd stage arguments
- **mcp**: name the launcher from the normalized workflow, and plan a transform first
- **mcp**: offer the workflow a session actually wrote, not the one the stage table names
- **data**: refuse two chains that name the same destination group
- **studio**: actually carry a transform's data directory to the panel
- **transform**: keep the auto bound scan's header reads inside its own guard
- **mcp**: guard plan_transform's config from a child that never returns
- **transform**: report the dtype the plan probed with, not a constant beside it
- **transform**: refuse two chains that share an intermediate Save
- **budget**: size the transform plan against the node's ranks, not the cluster's
- **patching**: drop a recorded sweep failure when the Save boundary flips
- **budget**: split a node-scoped auto budget across the node's ranks, not the cluster's
- **transform**: let a WHOLE_VOLUME declaration say why, and make Warp use it
- **data**: make a read plan's pull maps picklable, so a plan can cross the spawn
- **data**: bind a group's chain once, however often prepare is called
- **metric**: let Dice and FocalLoss take the integer label map a segmentation target is
- **transform**: refuse a Reduce grid policy that names no reference case
- **omezarr**: publish derived levels by rename, so the original is never the gap
- **transform**: ask a field store for its groups by the name Dataset exposes
- **dataset**: record a long array whole instead of NumPy's elided print
- **config**: refuse a nested block where a value is expected, instead of binding its repr
- **dataset**: normalize an attribute's value at construction, not only on assignment
- **data**: sweep a chain's pending Saves before serving a region, not after failing on one

### ⚡ Performance

- **data**: chunk a store on the region shape its writer declares
- **runtime**: run a single rank in this process instead of a child (#76)

### ♻️ Refactoring

- **data**: share the fallback constants and the Welford kernel
- **data**: move the reduction operators out of the predictor, into one shared vocabulary
- **data**: give the resolved memory budget a type that knows its own scope
- **data**: name what patching shares with the transform workflow, instead of reaching into privates
- **transform**: state each sampling rule once, over the two gathers that share the one arithmetic
- **ci**: pin every action to a commit SHA, and take the release notes from the committed changelog

### ⚠️ Behaviour changes

No YAML key, class or default was renamed, but the following answer differently for a config that
was not touched. Several are new refusals: what they refuse was being done before, silently and
wrongly.

- **`reduction: Median` returns a different number on an even count.** `torch.median` hands back the
  lower of the two middle values, which over two tensors is the element-wise minimum; it now averages
  the middle pair as `numpy.median` does. A 2-model ensemble or a 2-draw TTA that reduced to `1.0`
  over `[1.0, 3.0]` now reduces to `2.0`. On an odd count the two agree.
- **`Mean` and `Median` widen an integer input to float32**, including the single-tensor path that
  previously returned the tensor untouched. Rounding an average back onto an integer grid is a wrong
  number, not a narrower one -- but a `uint8` prediction output now lands as float32, four times the
  bytes on disk, and a downstream stage sees the wider dtype.
- Together those two make **`Median` the wrong operator for a label map**: it can answer with a
  label that was in no input (over 1 and 5 it gives 3), and over exactly two cases it *is* `Mean`.
  Fold segmentations with the new **`Vote`**, which picks the label the most cases agree on and
  keeps the dtype.
- **A whole-volume statistic after a `Reduce` is refused when an earlier post stage changes the
  values.** The stat pass measures the fold, so `[Reduce, Clip, Normalize]` normalised by the
  *unclipped* statistic and wrote a volume its own header did not describe. The per-case planner
  already refused this; the reduction now does too. Split the chain at the value-changing stage.
- **Two source groups may no longer declare the same destination group name.** The name is the key
  everything downstream indexes by, so the second chain used to be built and then silently dropped;
  it is now refused when the dataset is prepared, in every workflow. Give the chains distinct names
  and say `group:` on each `Write` to store both under one group.
- **A `uint8` label map resampled through a `field` now takes the nearest voxel.** The warped path
  consulted no interpolation rule, so it blended labels and truncated: over a source holding
  `{0, 100}` it returned 29, 79 and 99. Anything stored as another integer dtype needs
  `interpolation: nearest` spelled out -- a dtype cannot tell a label map from a CT.
- **A chain declaring two `Reduce` markers is refused at parse time**, naming the cardinality
  marker, where the second one used to fall past the split and be reported as an ordinary stage
  reading across space.
- **`grid: strict` refuses a geometry key no header records**, rather than skipping the comparison
  it promised. A missing `Direction` is a flip that shows in neither extent nor spacing. Use
  `grid: shape_only` for a cohort that means to fold on extent alone.
- **New OME-Zarr stores are chunked to a size a reader can open.** A streamed writer declaring the
  whole trailing plane produced a chunk of a gigabyte at 2048², past what zarr holds in one buffer
  at 4096². Existing stores keep their own chunking; zarr is self-describing.

## v1.7.0 (2026-07-29)

### ✨ Features

- **studio**: a truthful, guided chat over konfai-mcp
- **mcp**: restore app execution and harden the job runtime
- **py**: sweep the axis whose reassembly window is smallest
- **omezarr**: let an output declare what it writes, fields included
- **py**: select the most central patch when no combine is declared
- **py**: add Trim, a selection combine that keeps the most central patch
- **impact-reg**: accept a displacement field written as an OME-Zarr store
- **omezarr**: store a displacement field as an RFC-5 vector field
- **apps**: port the median ensemble reduction
- **apps**: compile a masked / TTA synthesis bundle to a program
- **apps**: export a multi-model ensemble to a program.json
- **apps**: export a bundle to onnx + manifest
- **export**: fold pointwise transforms into the onnx graph
- **studio**: React chat, NiiVue viewer, live feed, leaderboard, deploy
- **studio**: FastAPI BFF and pluggable agent brains
- **mcp**: studio-driving tools and run published apps as experiments
- **apps**: tunables, session-root runs, and download_bundle
- **core**: live training control and on-demand validation for the Studio feed
- **apps**: honour tunables on the remote path via a per-op option contract
- **network**: error on a named in_branch no module has produced
- **apps**: IMPACT tuning priors + full FireANTs feature distances
- **core**: run a weightless model with no checkpoint
- **mcp**: scaffold a parameter-refine loop for app reuse
- **apps**: fine-tune --set overrides and per-parameter descriptions
- **predict**: say when a case is not window-bounded
- **data**: materialize an unsatisfied Save by a streamed sweep
- **predict**: stream the linear resample inverse by default
- **predict**: stream a SLAB before-reduction transform per slab
- **data**: stream Mask slab by slab in the finalize chain
- **train**: size free patch axes for training through a shared helper
- **predict**: size a free patch axis to the model's valid input multiple
- **network**: read the model's input divisor from its downsampling graph
- **predict**: stream TTA through a slab-synchronized reduce
- **data**: declare slab-local side writes and stream InferenceStack
- **data**: add SlabAligner to merge per-copy slab streams
- **data**: interleave prediction TTA patches along the slab axis
- **data**: a missing memory_budget means auto
- **eval**: stream the SaveMap error maps under a memory budget
- **predict**: reserve the accumulation footprint in the shrink budget
- **train**: shrink free patch axes and restart training on CUDA OOM
- **predict**: shrink free patch axes and restart on CUDA OOM
- **predict**: VRAM shrink-step kernel for the OOM restart loop
- **eval**: memory-bounded evaluation from reducible metric partials
- **data**: per-axis free patch axes and rich overlap specs
- **predict**: stream geometry-inverse finalize chains to the write
- **data**: compose region stages and schedule slabs through them
- **data**: add the write mirror of the patch-streaming contracts
- **predict**: stream prediction outputs slab by slab
- **data**: add region-write streams to the dataset backends

### 🐛 Bug Fixes

- **mcp**: close a stdin that devnull could not replace
- close the CodeRabbit findings on the review branch
- **runtime**: fold the console mirror's bar frames off a terminal
- **evaluator**: use the run's device, and rank PSNR/SSIM as maxima
- **dataset**: make an entry visible across objects and processes
- **py**: keep an off-axis sweep off the streamed write path
- **omezarr**: let ngff-zarr write the streaming-store metadata
- **py**: no combine when nothing is tiled
- **py**: keep a patch whole when Trim has no central band to keep
- **impact-reg**: clear the Moved stem in the ensemble path too
- **deps**: relock ngff-zarr for the RFC-5 constraint
- **impact-reg**: refuse an image store as a field, and leave one output per stem
- **impact-reg**: re-read a store copied over one already read
- **impact-reg**: round-trip the displacement-field Direction through OME-Zarr
- **omezarr**: guard RFC-5 writes on zarr v3 and cover Direction
- **apps**: type-narrow the export paths and guard nested ensembles
- Windows signal guard and the CodeRabbit review findings
- **mcp**: fresh YAML per call and survive an unreadable subdirectory
- **apps**: reject an empty override name at validation
- **runtime**: harden log cleanup against symlinks and train_name traversal
- **metric**: keep the feature loss finite when no voxel is scored
- **config**: bind Optional[Literal] as a literal, not an object
- **data**: skip a crashed writer's temporary in the full sitk read too
- **data**: keep a heavy KonfAIInference nested run within VRAM
- **data**: keep the free-axis overlap default after an OOM re-plan
- **data**: skip a crashed writer's temporary in the sitk path resolver
- **network**: schedule criteria on the owning network's iteration counter
- **runtime,dataset**: keep run logs across overwrite, header-only get_infos
- **mcp**: register the real device reservation for app jobs
- **metric,config**: follow the input device in FID, skip typing-only union origins
- **pretrained**: refuse a tied buffer, not only a tied parameter
- **dataset**: normalize the rebased path to forward slashes on every OS
- **pretrained**: refuse a weight-tied target instead of mis-loading it
- **dataset**: keep an OME-Zarr entry recoverable while it is replaced
- **dataset**: match Attribute stack keys exactly, not by prefix
- **predict**: write an h5 prediction output as a file, not a hidden dotfile
- **data**: keep the train/val split disjoint on the worker and re-plan paths
- **metric**: apply every PerceptualLoss loss to the target
- **metric**: run LPIPS and IMPACTReg on the input device, not a fixed GPU 0
- **network**: validate nested-network losses against the root graph
- **predict**: log prediction measures rank-locally
- **train**: seed the train/validation split so RESUME keeps it
- **data**: keep augmentations diverse across epochs and split loaders
- **metric**: make FID runnable and Accuracy per-batch
- **transform**: resample through SimpleITK so geometry is honoured
- **network**: resume a nested network's optimizer under its dotted key
- **config**: keep a union value's type over lossy coercion
- **mcp**: make trial labels round-trip and reach every launcher
- **apps**: make the local-ref contract hold end to end
- **mcp**: identify an app trial by its label, not the shared RUN dir
- **apps**: honour the engine contracts the annotations promise
- **apps**: resolve model constraints from a package classpath, not only a bundled .py
- **core**: make the weightless path fail-closed in ModelComposite
- **apps**: address review on the moved IMPACT-Reg engines
- **apps**: survive a broken symlink when resolving an app on Windows (#58)
- **data**: flatten nested composite transforms before serializing
- **train**: extract image layers once per model, not per network
- **predict**: give a forward orientation the volume extent, not the slab
- **network**: count AvgPool and seed nested blocks from all inputs
- **network**: see strides inside opaque leaves; align 2D factors
- **data**: split the auto eval budget across local ranks
- **data**: round each case's free patch axis to the model multiple
- **data**: make stream finalize single-shot; address review findings
- **predict**: harden the sweep and the TTA gate, share their duplicated paths
- **data**: give every stream its own temporary and unlock pooled h5 reads
- **data**: keep a streamed entry invisible until it finalizes
- **data**: resolve free patch axes in the accumulator and blend window
- **predict**: budget the slide and in-flight blocks in the VRAM gate
- **predict**: harden streamed-TTA edge cases
- **predict**: measure each restart attempt against its own CUDA peak
- **data**: widen the resample source window to the nearest-picked voxel

### ⚡ Performance

- **apps**: stop revalidating the whole HF cache before every app job
- **omezarr**: keep a resident array out of the ngff disk cache
- **py**: normalise the patch blend per share, not by an accumulated weight
- **dicom**: memoise the per-series header scan
- **predict**: one worth rule for every case, not only TTA
- **predict**: keep a memory-light TTA case off the streamed reduce
- **data**: stream on a single window, not a double-length buffer
- **data**: cache chunked reads on the h5 and zarr paths
- **data**: slide the streaming window instead of ringing it
- **predict**: overlap disk writes with the prediction loop
- **data**: reuse blend staging and ring the streaming window
- **eval**: stream evaluation data instead of caching the dataset

### ♻️ Refactoring

- **py**: emit the patch grid with the sweep axis outermost
- **py**: make the reassembly window slide along a declared axis
- split the Studio BFF god-module and App state
- remove reviewed redundancies
- **mcp**: carve workspace, config IO, and file IO out of server_support
- **mcp**: split SessionService dataset and metrics halves into mixins
- **mcp**: move tool descriptions and prompt text into guide.py
- **tests**: fold micro-files into per-module homes
- **mcp**: one WorkflowSpec table behind every workflow-kind registry
- **apps**: replace the FromHF isinstance dispatch with a refresh hook
- **config**: decompose the apply_config binder into per-kind helpers
- **dataset**: one sitk transform codec for the three copy sites
- **models**: share the nnU-Net UNetDecoder between the two backbones
- **metric**: one masked feature-loss engine for the IMPACT family
- **apps**: define IMPACT-Reg engines once in impact_reg_konfai
- deduplicate the evaluator, budget and trainer sizing paths
- **utils**: raise ConfigError for overlap spec mistakes
- **data**: retire the use_cache config knob

## v1.6.0 (2026-07-16)

### BREAKING CHANGE

- fully-qualified references 'konfai.models.<task>.<Module>:<Class>' become
'konfai.models.python.<task>.<Module>:<Class>' (no compatibility alias).

### ✨ Features

- **data**: name a class from another framework where a transform goes
- **data**: windowed sampler, memory budget and patch_transform checks
- **data**: stream patch regions through the declared localities
- **data**: declare patch locality on transforms and augmentations
- **models**: block-level YAML catalog for ResEnc and UNet++ models
- **models**: add parametric ResidualEncoderUNet catalog model and ResidualBlockD
- **apps**: impact_reg orchestrator, declared-default optional inputs, and app CLI smoke tests
- **core**: auto-detect OME-Zarr/DICOM store format and add IMPACTReg PCA loss
- **mcp**: add the konfai-mcp agent experimentation server
- **apps**: expose app capabilities for MCP-driven workflows
- **examples**: registration example and model in YAML and Python forms
- **core**: declarative YAML model catalog, registry and pretrained bridge

### 🐛 Bug Fixes

- **data**: four promises the code did not keep, and the docs that repeated them (#46)
- **data**: give the process back the random state a foreign draw borrowed
- **config**: bind a callable whose module writes its annotations as text
- **data**: refuse a region stage that records geometry nowhere
- **data**: keep every rank's epoch the same length and size the cache by its copies
- synthesised input defaults on a multiscale dataset
- **config**: honour a None default for an optional nested object
- **core**: correct silent-correctness bugs and guard optional imports
- unblock Slicer inference (restore current_free_vram + file_system tensor sharing) (#35)

### ⚡ Performance

- **dataset**: stream sitk statistics and warn on unstreamable reads

## v1.5.9 (2026-07-06)

### ✨ Features

- tunable app parameters with type-derived constraints (#33)
- **apps**: derive patch size + VRAM plan from config, install requirements by default
- **apps**: add impact-reg-konfai (IMPACT registration orchestrator)
- **apps**: add impact-seg-konfai (multimodal body segmentation app)
- **apps**: expose patch/batch flags in per-app CLIs
- **core**: registration primitives (Norm, Flip vector-field, Attribute)
- **apps**: add --patch-size/--batch-size inference override
- **apps**: multi-checkpoint fine-tune and core fixes for released apps (#15)
- add app bundle command (#14)
- add ONNX model export (#13)
- add declarative YAML model builder (#8)
- add OME-Zarr (ngff-zarr) and DICOM dataset backends (#7)

### 🐛 Bug Fixes

- **predict**: fall back to CPU when the gate-approved blend OOMs
- **transform**: sample Clip percentile bounds from a host view
- **transform**: device-transparent finalize and precedence label merge
- **patching**: floor blend weights at the dtype's smallest normal
- **predict**: per-case finalize device and free-VRAM accumulation gate
- **network**: clear the patch buffer between model-patch iterations
- **pixi**: dedupe lint tasks so bare pixi run check works again
- **predict**: default OutputDataset device to CPU on CPU-only runs
- **patching**: keep overlap-blend reassembly in the patch dtype
- **predict**: co-locate load-added modules onto the model's device
- **examples**: use SAM_Perceptual loss in the Synthesis example
- **apps**: harden the app server, tighten security and packaging
- **train**: checkpoint, resume and IO safety
- **data**: correctness fixes across the lazy patch pipeline
- **core**: correctness fixes across metric, network and models
- **config**: preserve dict defaults and dotted keys on write-back

### ⚡ Performance

- **predict**: GPU-resident accumulation + device-transparent finalize (#34)
- **predict**: stage per-patch GPU->CPU offload via pinned memory
- **predict**: GPU channel reduction (argmax) when it fits VRAM
- **patching**: separable blend windows; drop the distance-map Cosinus, add Gaussian
- **init**: defer requests and torch imports to point of use
- **runtime**: make CPU-utilization telemetry non-blocking
- **transform**: compute Dilate as separable 1-D max-pools
- **data**: device-side accumulator alloc + float32 clamp fast path
- **dicom**: thread series info through slice reads, kill O(Z^2) stats
- **dataset**: memoize Dataset.get_infos across setup passes
- **predict**: cache ensemble state_dicts in RAM
- **patching**: make Accumulator.is_full O(1) via a filled counter

## Earlier releases (v1.0.0 – v1.5.8)

The history before `v1.5.9` predates Conventional Commits and is not rendered here. The main
releases, as announced by their commit messages:

| Version | Date | Announced as |
| --- | --- | --- |
| `1.4.1` | 2025-11-27 | Fix the missing resource package on Windows; handle `CUDA_VISIBLE_DEVICES` when CUDA is unavailable |
| `1.4.0` | 2025-11-26 | Enable Windows inference and improve konfai-apps |
| `1.0.0` | 2025-06-02 | First functional version of KonfAI |

For anything finer: `git log --oneline v1.5.8`.

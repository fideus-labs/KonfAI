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

## v1.8.1 (2026-08-19)

TRANSFORM, the dataset-preparation workflow, is the bulk of this release: the plan reads no voxel,
more stages stream, the memory budget is right in a container and on a cluster, and one failing case
no longer stops a run. Two things a run already wrote come out differently, see the behaviour changes.

### ✨ Features

- **transform**: the chain runs on the rank's device (`--gpu`), in taller slabs than on a CPU, and
  writes the same bytes; a nested `KonfAIInference` runs there too
- **augmentation**: free rotations, scales, noise and cutouts stream, so an `Expand` copy that draws
  them no longer takes the whole-volume path
- **transform**: `Mask` and `Padding` stream, on the reader's regions and on the writer's slabs alike
- **transform**: `Transform.working_multiple` declares what a stage allocates beyond its input and
  output (`Resample` 3, `Gradient` 8, `Dilate` 3); the plan sizes the whole-volume fallback with it
- **reduction**: the plan prices the reads of a `Reduce` whose members sit on a store that cannot
  serve a bounded region
- **transform**: `outputs.json` names the path on disk beside the dataset root, and Studio shows one
  Browse per chain of a transform run
- **mcp**: transform runs steer GPUs, a finished job carries its `outputs`, and `read_dataset_file`
  summarises an ITK transform HDF5
- **apps**: batch size as a first-class fine-tune knob (`--set`, `install_fine_tune`)
- **transform**: a `Resample` on an intensity chain may declare `precision: fast`, a float32
  coordinate walk over half the bytes; `exact` (float64, the default) stays bit-identical to
  `sitk.Resample`, and any other chain is refused with the reason
- **config**: the binder reads a file strictly. `strict_config(root)` records, per level, what the
  file holds against what the binder read, and reports the difference by path with the keys read
  at that level and the closest one. TRANSFORM refuses; TRAIN, PREDICTION and EVALUATION warn

### 🐛 Bug Fixes

- **transform**: the plan prices the route on what the chain reads: a destination that does not
  exist yet is not a read, so h5 to mha and h5 to omezarr stream instead of loading
- **transform**: `Crop` keeps the last row of foreground on every axis. The scan reports the last
  foreground INDEX and the box carries the margin after it, so converting one into the other left
  the far margin one voxel too large and the crop stopped short of that row
- **transform**: `Crop` finds its box by a bounded quantile scan (exactly `numpy.quantile`) instead
  of holding the volume, and the plan reads no voxel for a global statistic: the rank reads it at
  first access, a LOAD case on the volume it holds
- **transform**: a case's copies do not depend on its position in the run, and `Noise` draws its
  field from the copy's own seed
- **transform**: a failing case does not stop the rank's shard; the rank finishes, lists the failures
  and exits non-zero (`on_fallback: error` goes through the same channel)
- **transform**: the ranks run without a process group: no port, no rendezvous, no `scontrol`
- **transform**: an interrupt aborts the stream and leaves no `.tmp` behind
- **runtime**: the auto budget reads the process's own cgroup (not the mount root), subtracts the
  memory it holds and not the page cache, honours a SLURM grant, and reads `--mem=0` as the whole node
- **runtime**: shards are balanced by bytes, and an inline run puts the caller's RNG (CPU and CUDA)
  and cudnn flags back
- **runtime**: the per-rank thread share bounds ITK's pool as well as torch's, and counts as applied
  only once it is. The two take that share differently: torch's is capped at 12 (past memory-bus
  saturation its intraop pool only adds contention), ITK's takes it whole, because its resampler
  keeps scaling with it (one fold region: 10.98 s at 1 thread, 1.11 s at 12, 0.65 s at 24). Capping
  ITK at torch's number left a third of a 24-core node idle: 15 s on a measured fold
- **dataset**: a dead writer's debris is told apart portably (`psutil.pid_exists`), Windows included
- **dataset**: a streamed `.mha` writes MetaIO's `TransformMatrix`, which is the TRANSPOSE of
  ITK's `Direction`: a non-symmetric orientation used to read back mirrored
- **dataset**: an entry whose writer was killed between moving the old version aside and
  publishing the new one is served again. The previous, complete version survived under the
  `<name>.replaced-<pid>` backup that every listing hides, so the output was preserved and not
  served, which reads as data loss. A single backup from a writer that no longer runs is put back
  (h5, MetaImage/NIfTI and OME-Zarr alike) with a warning saying which write to run again; a live
  writer's backup is still left alone
- **dataset**: an h5 replace keeps the old entry until the new one is in place; a streamed transform
  entry lands under the `.h5` name; an entry whose attributes fail is not left behind; one staging
  marker for every backend's temporary; the reader's own `ITK_*` keys do not travel with the volume
- **cli**: `--plan` takes explicit flags, shards the way the run will, and takes back the entry and
  the store its probe created
- **omezarr**: each scale factor shrinks the level above it, as the docs said (`[2, 2]` gives three
  levels at 1, 1/2, 1/4)
- **omezarr**: a coarser level reads its own geometry. The sidecar describes the level the writer
  was handed, so taken at its word on level 1 it put level 0's spacing and origin on level 1's
  voxels: a volume read at `@1` came back four times too small in world coordinates
- **reduction**: `Median` selects the middle with a network of element-wise min/max up to five
  members instead of sorting the stack, and charges what that route holds. Same values to the
  bit (`torch.quantile` is the reference), a fraction of the time (three members: 7.20 to
  0.45 ms on CUDA, 55 to 26 ms on the host) and a third of the working set, so the planner
  cuts taller slabs for the same budget. A `Reduction` may now answer `working_multiple_for`
  per cohort size; the class attribute stays the contract and the worst case
- **transform**: a bare stage name past the `Expand` marker is the draw (`Flip`, `Mask`, `Permute`,
  `Foreign` exist as both); the qualified spelling still forces either
- **transform**: the working set counts what the widest stage allocates; a `Save` cache the run
  sweeps is priced as a bounded source; the plan says own pass for the last copy of a case; the
  slab-height note names only the chains it applies to
- **data**: a case present in two roots is read from the first, and a warning says so
- **augmentation**: a singleton axis sits at 0 in the affine sampler, as `affine_grid` places it
- **api**: a workflow releases its h5 read handles on return
- **network**: a criterion is moved onto the output's device before it is called
- **studio**: a fresh workspace tree, and a job launch brings its run forward
- **mcp**: `Transform.yml` is exempt from the model lint
- **predictor**: a built-in reduction (`Mean`, `Median`, `Concat`) binds from its own block like a
  custom one; the `Mean:` block a resolved config carries was read by nothing

### ⚡ Performance

- **transform**: taller slabs when the chain runs on a GPU (500³ `Clip` 3.4 to 2.6 s)
- **transform**: a streamed `Resample` walks its coordinates on the rank's device, out of core and
  slabbed under the machine's budget: a three-specimen ExaSPIM template round folds in 127 s where
  the host baseline did not finish in 1 h 47, at 4.5 GiB of VRAM, and the same bytes every run
- **transform**: on a host with no GPU that `Resample` goes through ITK's own resampler, 27 ns per
  voxel against the host walk's 326: the full CPU fold of the same round, 2005 s to 892 s
- **runtime**: ITK's pool takes the rank's whole share instead of torch's cap: on a 24-core node
  the resample of a measured fold goes 61.1 s to 46.6 s, the round 104 s to 97 s
- **omezarr**: a pyramid appended to a streamed store no longer rewrites level 0, the coarser
  levels are derived from it lazily (that round's update pass 103 s to 53 s, the 4.9 GB template's
  level 1 53 s to 4 s), and a store is created empty for zarr to fill (2.3 s where the rechunk took 14.4)
- **reduction**: a fold runs on the rank's device, `Vote` counts along the case axis (2 s per 512³
  where the pass it replaces took 31 s) and `Median` reads the middle off a sort (1.5-2x on CPU,
  3.5x on CUDA); the folds a statistics pass computed are kept for the write pass when they fit
- **transform**: a case is planned without listing the whole output directory
- **reduction**: a fold that is not incremental reads its members up to four at a time. Each read is
  a decode plus a replay of that case's chain, the members are independent, and such a fold holds
  every member anyway, so nothing is spent that the plan did not already charge. The members reach
  the operator in cohort order whatever order they arrive in, so the fold writes the same bytes: a
  five-case fold of a compressed cohort through `Clip` and `Resample`, 2.39 s to 0.97 s
- **dataset**: a volume's statistics come from one block walk, folded in cache-sized pieces, and a
  `.npy` entry answers its shape from the header instead of reading the array
- **cli**: `import konfai` 0.9 to 0.08 s, `konfai --help` 2.9 to 0.3 s: torch, dicom and ome-zarr
  load on first use

### ♻️ Refactoring

- **data**: `CaseMaterializer` (`konfai/data/materialize.py`) is the TRANSFORM engine, out of
  `DatasetManager`; one `RegionWriter` for the three engines; one `WorkItem` for the plan, the
  shards and the run; the report is assembled from per-block helpers
- **utils**: the memory budget lives in `konfai.utils.budget`; `State` in `konfai.utils`, importable
  without torch
- **data**: `DataSources` holds what the four workflow datasets share (the roots, the case names
  common to every group, one manager per destination group and case, the resolved budget); `Data`
  adds the batch-loading mechanics and `DataTransform` builds on `DataSources` alone, so it no
  longer forwards eleven loader parameters it never used. The managers and case names are public
  on the base, which is what `Transformer` now reads

### ⚠️ Behaviour changes

- **An `Expand` copy is a different volume than before.** A copy's draws are keyed by the case name,
  not by its index in the run, and `Noise` derives its field from the voxel position and the copy's
  seed. Same config, same seed: the copies written by this version are not the ones an earlier version
  wrote. **An `Expand` output partly written before this version and resumed after it mixes the two
  rules**: the cases already on disk keep the old draws, the resumed ones get the new. Run those
  outputs once with `--overwrite`.
- **A `Median` reduction may plan differently.** Its working set is four buffers of the fold instead
  of two, so a cohort that streamed within a few percent of the budget may now be refused (a
  reduction has no whole-volume path); the plan says so.
- **A failing case does not stop the run.** The rank finishes its shard and exits non-zero with the
  list; a caller that relied on the first failure raising mid-run reads the list instead.
- **The auto memory budget is larger on a host that has streamed a cohort** and smaller under a
  SLURM grant that no cgroup enforces; the plan's header names which bound won.
- **A `Crop` output is one voxel wider per axis than every earlier version wrote.** The box kept
  the last foreground index as the far margin, so the crop stopped one row short and dropped that
  row on each axis. Same config, same data: the volume this version writes is not the one an
  earlier version wrote, and an output partly written before and resumed after mixes the two.
  Run those outputs once with `--overwrite`, and re-derive anything (a model, a metric) that was
  trained or measured on a cropped dataset if the missing row mattered.
- **A streamed `.mha` written by an earlier version can carry mirrored geometry.** The streamed
  writer wrote the `Direction` where MetaIO expects its transpose, so an entry whose orientation
  is not symmetric (an oblique acquisition, an axis-permuting direction) reads back mirrored.
  Entries written whole are unaffected, as are symmetric orientations (an axis-aligned volume).
  Compare the stored `TransformMatrix` against the source's `Direction`, and rewrite the affected
  entries with this version.
- **The written geometry sidecar no longer carries `ITK_*` keys** the reader had added; a consumer
  that read them from a KonfAI output finds them absent.
- **A `Config.yml`, `Prediction.yml` or `Evaluation.yml` carrying a key nothing reads now warns**
  at build, naming the key by path. Files written back by earlier versions carry a few
  (`Patch.mask`, `Model.<name>.yaml_str`, `schedulers.<name>.verbose`, `is_input` under an
  evaluation group): remove them; the run is otherwise unchanged. A `Transform.yml` refuses them, as
  it already refused a typo'd structural key, and the check now covers a `Reduce` operator's own
  parameters and a wrapped foreign class's.

## v1.8.0 (2026-08-07)

### 💥 Breaking changes

Three, and each one names what to change.

- **One `Resample`, and no other spelling.** `ResampleToResolution`, `ResampleToShape`,
  `ResampleToReference`, `ResampleTransform` and `Warp` are gone: five names for one stage. A config
  now names `Resample` and says which grid it targets (`spacing`, `shape` or `reference`) and which
  map it goes through (`field`, `transforms`). So `ResampleToResolution: {spacing: [...]}` becomes
  `Resample: {spacing: [...]}`, and `Warp: {field: ...}` becomes `Resample: {field: ...}`. Two
  `Resample` stages in one chain spell the second module-qualified,
  `konfai.data.transform:Resample`. The Hugging Face bundle configs move with the `konfai` version
  they pin, so a published app keeps working.
- **A field carries no recorded bound.** `DISPLACEMENT_BOUND_ATTRIBUTE`, `displacement_bound` and
  `MaxDisplacement` are removed from `konfai.utils.ome_zarr`. A store that still carries the
  attribute is read fine; the value is ignored. The run now sizes each region's pull from the field
  values it actually reads, and the plan says it prices those reads as a zero field.
- **A backend token is not an extension, and `h5py` is no longer optional for it.** `:itktransform`
  writes `<group>.h5`, so nothing on disk is ever named `.itktransform`; validating a spec against
  the extension list used to reject the format the write side had just produced, which meant a run
  could write a transform it could not read back. Backend tokens live in
  `SUPPORTED_BACKEND_FORMATS` now, and a `path[:flag]:format` spec is checked against the union.
  The backend also stops degrading in silence when `h5py` is missing: it used to hold the whole
  field in float64, so peak memory turned on whether an optional import had succeeded. Install
  `konfai[hdf5]`.

### ✨ Features

- **transform**: resample a case onto the grid of a declared reference
- **data**: cubic interpolation for `Resample`, beside nearest and linear
- **impact-reg**: a preset declares only its field; the moved image and the ensemble are derived
- **transform**: compose the grid change and the warp into one pass
- **data**: give a chain a per-component statistic
- **studio**: show a transform run, and point Browse at the data rather than its log
- **mcp**: let an agent plan and run a transform, and read the table for what it accepts
- **transform**: a fifth workflow that reads a dataset, applies a chain, and writes it
- **transform**: give a reference resample an `interpolation`, as the warped path already had
- **data**: Vote, the reduction operator that folds segmentations without inventing a label
- **data**: declare an OME-Zarr pyramid from a Write
- **impact-reg**: seed the rigid from the centre of mass, not only the frame
- **studio**: bundle icons through the app interface, and a way to stop Studio (#75)
- **examples**: a Transform example: a template folded out of a cohort, and drawn copies of a case

### 🐛 Bug Fixes

- **transform**: sample a label map by nearest on the warped path too, instead of blending labels
- **api**: spell numpy scalars into the config tree, and copy a caller's config file rather than rewriting it
- **data**: a text transform and a stepped region read reach the backend that serves them
- **data**: the backend rewrite lands on `.h5`, and `Std` prices the buffers it allocates
- **impact-reg**: order cases numerically past P999, and refuse an output name that collides with the derived one
- **cli**: the `--gpu` help shows the space-separated form argparse accepts, not `0,1,2`
- **transform**: build the resample grid on the volume's device, so a GPU-resident case does not raise
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
- **transform**: plan a resampled case on the grid it lands on
- **transform**: refuse a stage key that names something not a class
- **transform**: print the plan's reduction and dropped lines once
- **transform**: give refusals their true remedy and name
- **transform**: hold on_fallback error at run time too
- **transform**: refuse unknown roots and typo'd stage arguments
- **mcp**: name the launcher from the normalized workflow, and plan a transform first
- **mcp**: offer the workflow a session actually wrote, not the one the stage table names
- **data**: refuse two chains that name the same destination group
- **studio**: actually carry a transform's data directory to the panel
- **mcp**: guard plan_transform's config from a child that never returns
- **transform**: report the dtype the plan probed with, not a constant beside it
- **transform**: refuse two chains that share an intermediate Save
- **budget**: size the transform plan against the node's ranks, not the cluster's
- **patching**: drop a recorded sweep failure when the Save boundary flips
- **budget**: split a node-scoped auto budget across the node's ranks, not the cluster's
- **transform**: let a WHOLE_VOLUME declaration say why, and make the resample use it
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

- one home for what was written several times: the region-kind set, the statistics-seed rule, the entry resolver, the plan's verdict cascade, the workflow launcher and the CLI builders

- **data**: share the fallback constants and the Welford kernel
- **data**: move the reduction operators out of the predictor, into one shared vocabulary
- **data**: give the resolved memory budget a type that knows its own scope
- **data**: name what patching shares with the transform workflow, instead of reaching into privates
- **transform**: state each sampling rule once, over the two gathers that share the one arithmetic
- **ci**: pin every action to a commit SHA, and take the release notes from the committed changelog

### ⚠️ Behaviour changes

Beyond the renames above, the following answer differently for a config that was not touched.
Several are new refusals: what they refuse was being done before, silently and wrongly.

- **`reduction: Median` returns a different number on an even count.** `torch.median` hands back the
  lower of the two middle values, which over two tensors is the element-wise minimum; it now averages
  the middle pair as `numpy.median` does. A 2-model ensemble or a 2-draw TTA that reduced to `1.0`
  over `[1.0, 3.0]` now reduces to `2.0`. On an odd count the two agree.
- **`Mean` and `Median` widen an integer input to float32**, including the single-tensor path that
  previously returned the tensor untouched. Rounding an average back onto an integer grid is a wrong
  number, not a narrower one, but a `uint8` prediction output now lands as float32, four times the
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
  `interpolation: nearest` spelled out, since a dtype cannot tell a label map from a CT.
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

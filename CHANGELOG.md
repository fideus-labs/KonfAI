# Changelog

Generated from the commit history by [Commitizen](https://commitizen-tools.github.io/commitizen/) —
every entry is a conventional commit, and each version's section is exactly what the GitHub Release
for that tag carries. Regenerate with:

```bash
cz changelog --start-rev v1.5.8
```

`--start-rev` is not a preference: Conventional Commits only took hold at `v1.5.9`, so rendering
further back produces version headings with nothing under them. What came before is described below.

## v1.8.0 (2026-08-01)

### ✨ Features

- **transform**: let Warp read the bound the fields recorded, with max_displacement: auto
- **studio**: show a transform run, and point Browse at the data rather than its log
- **mcp**: let an agent plan and run a transform, and read the table for what it accepts
- **transform**: a fifth workflow that reads a dataset, applies a chain, and writes it
- **data**: declare an OME-Zarr pyramid from a Write, and let a field carry its own bound
- **impact-reg**: seed the rigid from the centre of mass, not only the frame

### 🐛 Bug Fixes

- **transform**: let a WHOLE_VOLUME declaration say why, and make Warp use it
- **data**: make a read plan's pull maps picklable, so a plan can cross the spawn
- **data**: bind a group's chain once, however often prepare is called
- **config**: refuse a nested block where a value is expected, instead of binding its repr
- **dataset**: normalize an attribute's value at construction, not only on assignment
- **data**: sweep a chain's pending Saves before serving a region, not after failing on one

### ♻️ Refactoring

- **data**: move the reduction operators out of the predictor, into one shared vocabulary
- **data**: give the resolved memory budget a type that knows its own scope

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

Not generated, and not reconstructed. The history before `v1.5.9` predates Conventional Commits, so
there is no structure to render — and inventing one would describe a history that never happened.

What those commit messages announce themselves:

| Version | Date | Announced as |
| --- | --- | --- |
| `1.4.1` | 2025-11-27 | Fix the missing resource package on Windows; handle `CUDA_VISIBLE_DEVICES` when CUDA is unavailable |
| `1.4.0` | 2025-11-26 | Enable Windows inference and improve konfai-apps |
| `1.0.0` | 2025-06-02 | First functional version of KonfAI |

For anything finer, read the log:

```bash
git log --oneline v1.5.8
```

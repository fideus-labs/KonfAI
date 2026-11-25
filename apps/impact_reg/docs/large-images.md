# Large images

IMPACT-Reg registers a pair of any size with nothing to set: the whole pair on a grid sized to the memory, then, for
the presets that declare it, a refinement on native tiles. The outputs are on the native fixed grid. The same command
works for a brain MRI and for a light-sheet volume of a billion voxels:

```bash
impact-reg-konfai register FireANTs#syn -f fixed.ome.zarr -m moving.ome.zarr -o Output --gpu 0
```

## How a run is sized

Each preset declares what one voxel of the pair costs it at its peak, on the GPU and in RAM
(`vram_bytes_per_voxel` and `ram_bytes_per_voxel` in its `app.json`, which `show NAME` prints). Before a preset runs,
KonfAI turns that cost into the number of voxels it can register in one piece:

- **on a GPU**, from the memory free on the card when the run starts, keeping a 20 % margin, and from the RAM as on
  the CPU below, whichever holds fewer voxels;
- **on the CPU**, from the RAM available to the run (the machine's, or its cgroup or SLURM limit), keeping a 20 %
  margin.

Every published preset declares its cost, so every run is sized. A preset that declares none gets the pair whole.

## Whole or in two passes

- **The pair fits:** the preset registers it in one piece.
- **It does not:** the preset registers it in two passes.
  1. **Global pass.** The whole preset (rigid, affine, deformable) runs once on the pair, resampled by KonfAI onto a
     grid just coarse enough to fit. Its field comes back onto the native fixed grid.
  2. **Tile pass.** The moving image is warped through that first result, then the preset's deformable stage refines
     it on native tiles with a 20 % overlap, blended where they meet. A tile the fixed mask does not reach gets a zero
     field without running. The two fields are composed on the native grid.

Only presets with a deformable stage worth refining locally run the tile pass: `Elastix#rigid_bspline`,
`ConvexAdam#composite`, `ConvexAdam#impact_cbct` and the FireANTs presets. The others stop after the global pass:
`Elastix#rigid`, `ConvexAdam#coarse`, the elastix IMPACT presets, whose features are computed at a few millimetres
anyway, and `ConvexAdam#impact_mrct`, whose organ-level features need the whole image around them.

The plan is printed before the run:

```text
[ImpactReg] FireANTs#syn: 257 x 665 x 887 voxels, more than the 19,636,779 it registers whole on this GPU: the preset
on the whole pair resampled to fit, then its deformable stage on native tiles of at most 17,705,407 voxels.
```

If a pass still runs out of GPU memory, because another process took some of it, KonfAI shrinks the pass and runs it
again. There is no such second chance on the CPU: Linux kills a process that runs out of RAM, which is why a CPU run
is sized from the RAM before it starts.

## Sampled, at native resolution

`--out-of-core sampled` is the second strategy: a pair over the budget registers at native resolution in one pass,
never resampled, never held whole. The image never changes; what an iteration looks at is the engine's: every box
of the pair, or with `--samples` a random subset drawn anew at each one (elastix draws its own points either way). The
inputs are first read once, by bands, and
written as uncompressed twins the engines map or read by boxes, measured in the fixed image's finest voxel as a run on
the whole pair is (a preset's lengths are voxels); the fields and transforms come back in millimetres. A mask lying
on another grid than its image goes onto it first, streamed. The strategy registers 3-D pairs, once, with its own
seed, in this process, and writes its field as a MetaImage: `--host`, `--tta`, `--seed` and `--output-format` (but
`Moved=FORMAT`) are refused.

### ConvexAdam

1. **Twins.** Each input is read once, by bands, and written as an HDF5 image in ITK's layout with 32-voxel chunks
   (`impact_reg_konfai.native`): the stages read boxes with their halo from it, which touch the few chunks under
   them wherever they land. A mask lying on another grid than its image goes onto it first, streamed.
2. **Coarse stage by regions.** When the preset has one (itk-impact's `ImpactCoarseRegistration`), one pass over
   the boxes pools the native MIND features to its coarse grid, which stays on the device; its cost volume is then
   built by boxes of that grid, never whole, and the coupled-convex iterations run on the pooled maps. Its field
   starts the fine stage.
3. **Fine stage by regions.** One pass reads the pair by boxes of `--region-size` voxels through streaming readers
   and pools the native MIND features to the control grid of the preset's fine stage (itk-impact's
   `ImpactFineRegistration`): the pair is read once. Where the pooled maps fit the device's budget, the preset's
   `iterations` run on them whole, ConvexAdam's own Static mode. Where they do not, the maps, the control grid
   and the optimizer's state live in files of `--tmp-dir` and every iteration steps every cell, by bricks the
   device holds: the same iterations, at any size. With `--samples`, the maps stay on the device where they fit
   its budget too, and an iteration draws boxes at random, that many voxels of them, anew at each one, and steps their
   cells alone, as elastix steps on the points it draws: what it holds and what it costs are its boxes',
   whatever the image. A box steps as often on average as the preset has `iterations` (`--epochs`). Give
   `--tmp-dir` a directory on a disk for a pair larger than memory.
   The dense field is written by regions; the field on the control grid is written beside it,
   `<preset>_ControlGrid.mha`. A preset with the coarse stage alone (`ConvexAdam#coarse`, `#impact_mrct`) writes
   that stage's field, interpolated from its grid region by region.

### elastix

elastix draws the samples of its metric at every iteration: it is a sampled registration already. What keeps a
large pair out of its reach is that it holds every image whole, with a copy for each pyramid level and each
interpolator. With `-out-of-core`, elastix maps the pixels of the pair from the files themselves and the system pages
them in as the samples of an iteration read them. It keeps no copy of a full-size image: the finest pyramid level is
the image, and an interpolation of order 1 reads it directly. The preset runs as it is. At order 1 the transform is
the one of a run in memory, bit for bit, given the same images and the preset's own samples (no `--samples`); at a
higher order the interpolation coefficients
are kept in single precision. It needs an elastix-IMPACT build that has `-out-of-core` (`KONFAI_ELASTIX_DIR`). The
dense field is the transform sampled on the fixed grid band by band; the transform itself is written beside it as
`<preset>_Transform.h5`.

### FireANTs

FireANTs holds both images, the warp and its Adam moments on the device and steps every voxel at every iteration.
Here the same iteration runs on boxes, as large as the device holds; with `--samples`, an iteration draws that many
voxels in random boxes, anew at each one, as elastix draws its points. The pyramid levels are files of the
workspace; the warps and their Adam moments stay on the GPU when they fit three quarters of its
free memory, else in host memory while the memory budget holds them, and in files beyond, where every step
writes them back to the disk.

1. **Rigid and affine.** An iteration scores small boxes spread over the level, about two million voxels, or
   `--samples` of them: a noisy estimate of the whole level's gradient. A level they hold whole gets FireANTs' own
   iteration. The step falls as elastix's gain does, `a / (A + k + 1)^0.602` with `A = 50`, and a scale
   ends on the mean of the transforms of its second half: the transform goes to the whole level's optimum as the
   iterations grow. A scale runs the preset's iterations.
2. **Greedy or SyN.** An iteration steps its boxes: their smoothed gradients folded into Adam's moments (a box's
   gradient is the whole grid's over its voxels), then their steps, normalised by the largest direction the
   boxes last had, composed and smoothed. Without `--samples` its boxes are all of them, which is FireANTs'
   iteration on the whole pair; with it, the ones it draws, each box stepping as often on average as the preset
   gives its scale iterations (`--epochs` for every scale).
3. **IMPACT.** Jacobian mode runs the networks on each box. Static mode first extracts the feature volumes block
   by block into files (the PCA and the means over every block), then registers them.
4. **Field.** Written block by block; for SyN the forward warp is read through the inverse of the reverse one,
   computed on each block.

A level the samples hold whole is registered as FireANTs registers it: each stage is then FireANTs' on the whole
image. `feature_map_update_interval` is not
available by boxes. A preset on features (every ConvexAdam preset, an IMPACT preset of elastix or FireANTs) takes
the pair with an identity direction, its voxel axes those of LPS, since the features are computed along them: reorient
it first, or register it without `--out-of-core sampled`, which reorients it.

## Options

| Option | Effect |
|---|---|
| `--max-voxels N` | register whole a pair of at most `N` voxels, whatever the device holds; the tiles are sized in proportion |
| `--tmp-dir DIR` | where the intermediates go (default: a hidden directory beside `--output`) |
| `--fields-only` | write the transform only, skip the moved image |
| `--out-of-core sampled` | register a pair over the budget at native resolution in one pass, an iteration on every box of the pair, or on a random draw with `--samples` |
| `--region-size N` | side of a box of the sampled mode, in voxels (default 64; ConvexAdam, FireANTs). Without `--samples`, FireANTs' deformable stage takes larger ones when the GPU has room: the result does not depend on them |
| `--samples N` | samples an iteration draws, anew at each one: the work of an iteration follows them, not the image, while the warps and what the stages keep per box (their steps, their loss) follow the image. A sample is a centre, read with what the loss or the model reads around it (its field of view). elastix draws that many points; ConvexAdam and FireANTs that many voxels in whole random boxes (one box at least, 8 voxels a side at least), whose centres share what they read. Default: what the device holds (ConvexAdam and FireANTs then step every box at every iteration, as on the whole pair), the preset's count for elastix |
| `--epochs E` | steps a box takes on average in the deformable stages (default: the stage's own iterations; ConvexAdam, FireANTs) |

The plan follows the memory free when the run starts, so a busy GPU can change it, and with it the result, slightly.
For runs that must be reproducible, pin `--max-voxels`. `register.json` records the plan every preset started with (`plans`) and, per rank, the final plan it ran on after
any out-of-memory re-plan (`ran`); the cases a rank finished before a restart ran on the plan before.

## Disk and RAM

**Disk.** The intermediates are volume-sized: 24 bytes per fixed voxel for each preset's field, plus their mean when
ensembling, and about 60 bytes per voxel at the peak of a two-pass run. The output transform alone is 24 bytes per
voxel, 29 GB for the 1.2 billion voxels of a 40 µm mouse brain. `register` checks the free disk before it starts.

**RAM.** Resampling, warping, composing and averaging are streamed: they read and write regions, never the whole
volume. Outputs are streamed too, in every format except DICOM: `.mha`, `.nii`, `.nii.gz`, `.nrrd` and OME-Zarr. A
DICOM output is written whole, since its rescale slope and intercept depend on the whole volume, and so are the
all-ones masks a DICOM pair gets when you leave its masks out: within the memory budget only, past it the run is
refused, so give the masks yourself or convert the pair. A moved OME-Zarr store comes with a pyramid, halved down to
about 256 voxels along its longest axis.

**3D Slicer.** SlicerImpactReg holds what it shows in memory: the moved image, and for uncertainty every preset's field
as a dense vector volume. A pair the CLI registers out of core may not fit there; run it from the CLI and load a
coarser level (an OME-Zarr pyramid) to look at it.

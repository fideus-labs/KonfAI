# Large images

IMPACT-Reg registers a pair of any size with nothing to set: the whole pair on a grid sized to the memory, then, for
the presets that declare it, a refinement on native tiles. The outputs are on the native fixed grid. The same command
works for a brain MRI and for a light-sheet volume of a billion voxels:

```bash
impact-reg-konfai register FireANTs_SyN -f fixed.ome.zarr -m moving.ome.zarr -o Output --gpu 0
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

Only presets with a deformable stage worth refining locally run the tile pass: `Generic_Rigid_BSpline`,
`ConvexAdam_Composite`, `ConvexAdam_IMPACT_CBCT` and the FireANTs presets. The others stop after the global pass:
`Generic_Rigid`, `ConvexAdam_Coarse`, the elastix IMPACT presets, whose features are computed at a few millimetres
anyway, and `ConvexAdam_IMPACT_MRCT`, whose organ-level features need the whole image around them.

The plan is printed before the run:

```text
[ImpactReg] FireANTs_SyN: 257 x 665 x 887 voxels, more than the 19,636,779 it registers whole on this GPU: the preset
on the whole pair resampled to fit, then its deformable stage on native tiles of at most 17,705,407 voxels.
```

If a pass still runs out of GPU memory, because another process took some of it, KonfAI shrinks the pass and runs it
again. There is no such second chance on the CPU: Linux kills a process that runs out of RAM, which is why a CPU run
is sized from the RAM before it starts.

## Options

| Option | Effect |
|---|---|
| `--max-voxels N` | register whole a pair of at most `N` voxels, whatever the device holds; the tiles are sized in proportion |
| `--tmp-dir DIR` | where the intermediates go (default: a hidden directory beside `--output`) |
| `--fields-only` | write the transform only, skip the moved image |

The plan follows the memory free when the run starts, so a busy GPU can change it, and with it the result, slightly.
For runs that must be reproducible, pin `--max-voxels`. `register.json` records the plan every preset started with.

## Disk and RAM

**Disk.** The intermediates are volume-sized: 24 bytes per fixed voxel for each preset's field, plus their mean when
ensembling, and about 60 bytes per voxel at the peak of a two-pass run. The output transform alone is 24 bytes per
voxel, 29 GB for the 1.2 billion voxels of a 40 µm mouse brain. `register` checks the free disk before it starts.

**RAM.** Resampling, warping, composing and averaging are streamed: they read and write regions, never the whole
volume. Outputs are streamed too, in every format except DICOM: `.mha`, `.nii`, `.nii.gz`, `.nrrd` and OME-Zarr. A
DICOM output is written whole, since its rescale slope and intercept depend on the whole volume. A moved OME-Zarr store
comes with a pyramid, halved down to about 256 voxels along its longest axis.

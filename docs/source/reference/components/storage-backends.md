# Storage backends & formats

KonfAI reads and writes datasets through pluggable backends in
`konfai/utils/dataset.py` (inner classes of `Dataset`). You rarely name a backend
directly — you pick a **format token** in a `dataset_filenames` spec
(`./Dataset:a:mha`), and the token is dispatched to a backend. See
{doc}`../../concepts/datasets` and {doc}`../../concepts/imaging-formats`.

## Backends

| Backend | Format token(s) | Kind | Optional extra |
| --- | --- | --- | --- |
| `Dataset.SitkFile` | `mha, mhd, nii, nii.gz, nrrd, nrrd.gz, gipl(.gz), hdr, img, dcm, tif(f), png, jpg, jpeg, bmp, itk.txt, fcsv, xml, vtk, npy` | Directory of per-case image files (default) | `konfai[itk]` (`SimpleITK`) |
| `Dataset.H5File` | `h5` | Single monolithic HDF5 file | `konfai[hdf5]` (`h5py`) |
| `Dataset.OmeZarrFile` | `omezarr, ome-zarr, ome_zarr, zarr` (+ `@level`) | OME-Zarr pyramid directory | `konfai[omezarr]` (`zarr` + `ngff-zarr`) |
| `Dataset.DicomFile` (DICOM series; scalar-array writes) | `dicom` | DICOM series directory | `konfai[dicom]` (`pydicom`) |

```{tip}
`pip install "konfai[imaging]"` installs **all four** backends at once
(`SimpleITK, h5py, pydicom, zarr, ngff-zarr`).
```

## The `SitkFile` default backend also handles sidecars

Beyond images, the SITK backend reads/writes several sidecar payloads by
extension: `.itk.txt` (SimpleITK transforms), `.fcsv` (Slicer landmarks), `.xml`
(attribute trees), `.vtk` (VTK PolyData points), `.npy` (raw NumPy, memory-mapped
on the slice path). It supports **true partial reads** (reading only the
requested spatial window) for streaming.

## Honest caveats

```{warning}
- **`dcm` ≠ `dicom`.** The token `dcm` reads a **single file** through SITK; only
  the literal token `dicom` uses the DICOM-**series** backend. Easy to trip over.
- **`vtk` is an ungated import** — `.vtk` I/O raises `ImportError` if `vtk` isn't
  installed, and there is no dedicated extra that declares it (`konfai[vtk]`
  installs it, but the sidecar path doesn't advertise the requirement).
- **`Attribute` only round-trips scalars and 1-D arrays.** The geometry sidecar
  stringifies values and reparses with `np.fromstring` after stripping the outer
  brackets, so anything multi-dimensional will **not** survive a read. Geometry is
  safe only because `Origin`/`Spacing` are 1-D and `Direction` is stored
  flattened. Store only flat scalars / 1-D arrays in an `Attribute`.
```

## Patching, streaming & reassembly

The data layer (`konfai/data/patching.py`, `konfai/data/data_manager.py`) never
loads a whole volume when it can avoid it:

- **`DatasetPatch`** (the `Patch:` config block) — `patch_size` (default
  `[128,128,128]`), `overlap` (`None` → auto-tiling), `pad_value` (`None` → pad
  with `data.min()`), `extend_slice` (2.5-D context, only when `patch_size[0]==1`).
- **`ModelPatch`** — patching applied *inside* a model graph, with a
  `patch_combine` blender (`Mean` or `Cosinus`) for overlap reassembly.
- **Streaming** reads only one patch's window from disk (`read_data_slice`). It is
  opt-in and conservative: only the base (non-augmented) patch, and only when
  every trailing transform is on a stream-safe allow-list (`TensorCast`, and
  `Normalize`/`Standardize`/`Clip` without masks, using precomputed statistics).
  Anything else falls back to a full in-RAM load.
- **`Accumulator`** reassembles patches with overlap blending, correcting border
  voxels covered by fewer patches. Patch **read order must match write order** —
  a load-bearing invariant.

```{important}
For PREDICTION / EVALUATION, **all patches of a case stay on the same DDP rank**
(the whole volume is reassembled per rank). For TRAIN, shards are truncated to the
shortest rank so DDP all-reduces stay balanced.
```

## See also

- {doc}`../../concepts/datasets` — grouped dataset layout, selectors, patching
- {doc}`../../concepts/imaging-formats` — DICOM & OME-Zarr specifics
- {doc}`transforms` — the stream-safe transform allow-list

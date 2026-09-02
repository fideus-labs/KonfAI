# Storage backends & formats

KonfAI reads and writes datasets through pluggable backends in
`konfai/utils/dataset/` (one module per backend, addressed as `Dataset.<Backend>`). You rarely name a backend
directly, you pick a **format token** in a `dataset_filenames` spec
(`./Dataset:a:mha`), and the token is dispatched to a backend. See
{doc}`../../concepts/datasets` for the grouped case/group layout these backends
serve; the DICOM and OME-Zarr reader APIs are detailed below.

## Backends

The two streaming columns are what the planner prices: **region reads** says
whether a region decodes only itself (a backend that answers "no" decodes the
whole volume behind every region, which only ever costs speed, never
correctness), and **streamed writes** says whether `open_data_stream` can build
the entry region by region (otherwise the volume is assembled and written
whole).

| Backend | Format token(s) | Kind | Region reads | Streamed writes | Optional extra |
| --- | --- | --- | --- | --- | --- |
| `Dataset.SitkFile` | `mha, mhd, nii, nii.gz, nrrd, nrrd.gz, gipl(.gz), hdr, img, dcm, tif(f), png, jpg, jpeg, bmp, itk.txt, fcsv, xml, vtk, npy` | Directory of per-case image files (default) | **uncompressed MetaImage and NIfTI only** | **the same set**: uncompressed `.mha` and `.nii` | `konfai[itk]` (`SimpleITK`) |
| `Dataset.H5File` | `h5` | Single monolithic HDF5 file | yes (chunked) | yes | `konfai[hdf5]` (`h5py`) |
| `Dataset.OmeZarrFile` | `omezarr, ome-zarr, ome_zarr, zarr` (+ `@level`) | OME-Zarr pyramid directory | yes (chunked) | yes, `scale_factors` pyramids included | `konfai[omezarr]` (`zarr` + `ngff-zarr`) |
| `Dataset.DicomFile` (DICOM series; scalar-array writes) | `dicom` | DICOM series directory | per slice | no (whole series) | `konfai[dicom]` (`pydicom`) |
| `Dataset.ItkTransformFile` | `itktransform` | ITK transform files (`.h5`, `.tfm`), one per case and group | yes, for a displacement entry | yes, for a 3-component 3-D displacement field with image geometry | `konfai[itk]` + `konfai[hdf5]` |

Reading a region is only cheap when the format allows it. A compressed stream is
not seekable, and NRRD never streams in ITK, so those decode the whole volume per
region: correct, but slow. The region-**writable** set is deliberately the
region-**readable** one, a memmap over the raw pixel block, which needs the image
geometry up front. An `:itktransform` entry writes its parameters region by
region and the file is exactly what `sitk.WriteTransform` would have produced;
any other transform kind is written whole.


`pip install "konfai[imaging]"` installs every backend at once
(`SimpleITK, h5py, pydicom, zarr, ngff-zarr`).

## Reading a root from object storage

A `dataset_filenames` entry may name a URI instead of a path:

```yaml
Dataset:
  dataset_filenames:
    - s3://aind-open-data/exaSPIM_822174_..._processed_...:omezarr
```

KonfAI adds no configuration key for credentials, because fsspec already has one.
It merges `FSSPEC_<PROTO>_<KEY>` from the environment and `~/.config/fsspec/*.json`
into every filesystem it builds, and botocore reads the usual `AWS_*` variables
under it. So a public bucket is

```bash
FSSPEC_S3_ANON=true konfai TRANSFORM --config Transform.yml
```

a private one is `AWS_PROFILE` or `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, and
a MinIO is `FSSPEC_S3_ENDPOINT_URL`. Install the filesystem for the scheme
(`pip install "konfai[s3]"` for `s3://`); `fsspec` itself already comes with the
OME-Zarr extra.

Two limits, both deliberate. Only `:omezarr` reads a remote root: it addresses a
store, where the other backends open a path. And a remote root is **read-only**,
because publishing an entry is a rename and an object store has none: point the
`Write:` at a local path.

Give a remote run a `memory_budget` that holds its working set. The store's
decoded-chunk cache is a third of that budget in every workflow (a transform plan
says when that third is below the 256 MiB the cache is worth), and a cache miss
on a remote root is a download where a local one is a decode. The same pair of
brains, the same chain, the same link at 1.6 MB/s: 1.898 GB downloaded in 19:12
with the cache at a share of free RAM, 2.573 GB in 38:34 with it bounded to a
third of a declared 4 GiB. The output was identical either way; only the number
of times a chunk was fetched changed.

A root that cannot be reached raises and names the reason. That is the whole
point of routing these probes rather than leaving them on `os`: `os.path.exists`
answers False for an `s3://` root and `os.listdir` raises `FileNotFoundError`, so
a mistyped bucket, an expired credential and a listing-denied prefix would each
read as an empty cohort, and the run would finish successfully having written
nothing.

The extras column above is the summary; {doc}`../../getting-started/installation`
is the canonical home for the optional-extras table.

## ITK transform files as a dataset

`:itktransform` stores one ITK transform per entry (`<case>/<group>.h5`). The
write side is the point: a displacement field streams into the exact file
`sitk.WriteTransform` would produce (pinned identical through ITK's own reader), without ever holding the field whole in float64. The read side hands back what
`Dataset.read_transform` decodes: a displacement entry as its field (region
reads included), any other stored transform (affine, composite, `.tfm`) as its
parameters. This is how a registration preset's transform deliverable (under whatever
name it declares) is a plain `Write:` like any other, and how a staged
transform file resolves through the same `Dataset` surface as an image.

`itktransform` is a **backend token, not an extension**: the file on disk is
`<group>.h5` (or `.tfm`), and nothing is ever named `.itktransform`. That is the
one thing `SUPPORTED_BACKEND_FORMATS` exists to say: a format may be declared in
a dataset spec without being a suffix any path can carry.

## The `SitkFile` default backend also handles sidecars

Beyond images, the SITK backend reads/writes several sidecar payloads by
extension: `.itk.txt` (SimpleITK transforms), `.fcsv` (Slicer landmarks), `.xml`
(attribute trees), `.vtk` (VTK PolyData points), `.npy` (raw NumPy, memory-mapped
on the slice path). It supports **true partial reads** (reading only the
requested spatial window) for streaming.

## API details: DICOM series

A DICOM acquisition is not a folder of independent images: a CT or MRI *series*
is a set of `.dcm` files that together define one 3-D volume. The reader
(`konfai/utils/dicom.py`) handles series discovery, slice ordering, geometry
extraction, and CT intensity rescale.

### `read_dicom_series`

```python
from konfai.utils.dicom import read_dicom_series

volume, origin, spacing, direction = read_dicom_series("path/to/series")
```

`konfai.utils.dicom.read_dicom_series(directory, *, series_uid=None, apply_rescale=True)`
returns a four-tuple:

| Returned | Shape | Meaning |
| --- | --- | --- |
| `volume` | `(1, Z, Y, X)` `float32` | channel-first voxel data |
| `origin` | `(3,)` | physical position of the first voxel, mm |
| `spacing` | `(3,)` | voxel size `(x, y, z)`, mm |
| `direction` | `(9,)` | row-major 3×3 direction-cosine matrix, flattened |

The `origin` / `spacing` / `direction` triple maps directly onto an `Attribute`
(`Origin`, `Spacing`, `Direction`; see {doc}`../../concepts/datasets`), so a
DICOM series travels through the pipeline under the same geometry contract as
any other format.

Key behaviors:

- **Slice ordering** uses `ImagePositionPatient` projected onto the slice normal
  derived from `ImageOrientationPatient`, not filename or `InstanceNumber`.
- **`apply_rescale=True`** (the default) applies `RescaleSlope` /
  `RescaleIntercept` to convert stored values to Hounsfield Units for CT. Set it
  to `False` to keep raw integers (for example for label maps).
- Missing geometry tags, inconsistent slice shapes, and unreadable pixel data all
  raise `DatasetManagerError` with an actionable message.

### Multi-series folders: `series_uid`

A single folder can hold more than one series (for example a T1 and a T2 from the
same session). `read_dicom_series` resolves this as follows:

- one series present → it is used automatically;
- multiple series with `series_uid=None` → `DatasetManagerError` listing the
  available `SeriesInstanceUID`s;
- pass `series_uid="1.2.840…"` to select one explicitly.

Use `konfai.utils.dicom.discover_series(directory)` to list the available UIDs
first:

```python
from konfai.utils.dicom import discover_series, read_dicom_series

series = discover_series("path/to/study")          # {uid: [Path, ...]}
uid = next(iter(series))
volume, origin, spacing, direction = read_dicom_series("path/to/study", series_uid=uid)
```

`write_dicom_series` writes one uncompressed scalar DICOM series. Integer data
round-trips exactly; floating-point data is stored as signed 16-bit pixels with
`RescaleSlope` and `RescaleIntercept`.

## API details: OME-Zarr

Unlike a NIfTI file, an OME-Zarr array is **already lazy**: it is stored as
chunked Zarr, so reading a sub-region only fetches the chunks it touches. This
maps naturally onto KonfAI's patch-based loading: the reader
(`konfai/utils/ome_zarr.py`) never materializes the whole volume.

### Multiscale levels

OME-NGFF stores a **multiscale pyramid**: the same image at several resolutions,
level `0` being full resolution and each higher level a downsampled copy. Each
level carries its own physical `scale` (spacing) and `translation` (origin) in the
`.zattrs` metadata, so geometry stays correct at every level.

### Regional reads through `Dataset`

```python
from konfai.utils.dataset import Dataset

dataset = Dataset("Dataset", "omezarr")
shape, attributes = dataset.get_infos("Volume", "CASE_001")  # metadata only
patch, patch_attributes = dataset.read_data_slice(
    "Volume",
    "CASE_001",
    (
        slice(None),       # C
        slice(0, 64),      # Z
        slice(0, 256),     # Y
        slice(0, 256),     # X
    ),
)
```

`get_infos()` reads the pyramid metadata without reading pixels.
`read_data_slice()` returns a channel-first `(C, Z, Y, X)` NumPy patch and an
`Attribute` whose origin and spacing have already been updated for the selected
window.

The lower-level equivalent is
`konfai.utils.ome_zarr.read_ome_zarr_data_slice(store_path, slices, *, level=0, timepoint=0)`:

```python
from konfai.utils.ome_zarr import read_ome_zarr_data_slice

patch, metadata = read_ome_zarr_data_slice(
    "image.ome.zarr",
    (slice(None), slice(0, 64), slice(0, 256), slice(0, 256)),
    level=0,
)
```

It returns:

| Returned | Meaning |
| --- | --- |
| `patch` | channel-first `(C, Z, Y, X)` patch preserving the stored dtype |
| `metadata` | axes, full shape, chunks, dtype, scale, translation, and stored KonfAI attributes |

The reader inspects the stored `axes` to place the spatial slices on the right
dimensions and to index optional `T` (time) and `C` (channel) axes, so the same
call works for `ZYX`, `CZYX`, and `TCZYX` arrays.

### Choosing a resolution

Select the pyramid level in the dataset format token. Level `0` is the default;
`omezarr@1` selects the next coarser image:

```python
from konfai.utils.dataset import Dataset

coarse = Dataset("Dataset", "omezarr@1")
coarse_shape, coarse_attributes = coarse.get_infos("Volume", "CASE_001")
```

Use `konfai.utils.ome_zarr.get_ome_zarr_info(store_path, level=0)` for a metadata
summary (`axes`, `shape`, `chunks`, `dtype`, `scale`, `translation`, `n_levels`)
without reading any pixels.

`write_ome_zarr` writes a single-level OME-NGFF store with channel/spatial axes,
chunking, scale, translation, and the original KonfAI attributes.

A store written region by region (a streamed `TRANSFORM` `Write`, a streamed
prediction) is chunked on the writer's region: a slab sweep declares
`[C, slab_rows, Y, X]`, and an axis the region covers end to end is tiled to at
most 128 once the chunk exceeds 32 MiB, so a large plane lands as
`[C, slab_rows, 128, 128]`. `slab_rows` follows the memory budget, so the chunk
layout of the store depends on the machine that wrote it; the values do not. A
reader cutting `32^3` training patches from that store decompresses those
chunks. A store written whole keeps `ngff-zarr`'s default chunking.

### Why `ngff-zarr` over `ome-zarr`

Two Python libraries read OME-NGFF: the OME consortium's `ome-zarr` and
`ngff-zarr`. KonfAI's `omezarr` extra depends on **`ngff-zarr`** (alongside raw
`zarr`) for one decisive reason:

> `ngff-zarr` exposes **per-scale physical coordinates**: the `scale` and
> `translation` of each pyramid level, as plain numeric arrays that convert
> directly to SimpleITK geometry. That is exactly the `(Origin, Spacing,
> Direction)` triple KonfAI stores in an `Attribute`, so OME-Zarr input lines up
> with every other format without a bespoke geometry adapter.

It also handles multiscale selection transparently and adds helpers tuned for 3-D
medical/bioimage workflows. `ngff-zarr` is required, not optional: without it the
reader raises `DatasetManagerError` pointing at `pip install konfai[omezarr]`. Bare
`zarr` is used only to read KonfAI's own attribute sidecar, never as a reader fallback.

## Use as a KonfAI dataset

Both formats use the normal grouped `Dataset` API:

```python
from konfai.utils.dataset import Dataset

dicom_dataset = Dataset("DatasetDicom", "dicom")
ome_dataset = Dataset("DatasetOme", "omezarr")  # aliases: ome-zarr, ome_zarr, zarr

dicom_dataset.write("CT", "CASE_001", volume, attributes)
patch, attributes = dicom_dataset.read_data_slice(
    "CT", "CASE_001", (slice(None), slice(10, 20), slice(32, 96), slice(32, 96))
)

ome_dataset.write("CT", "CASE_001", volume, attributes)
names = ome_dataset.get_names("CT")
```

The layouts are `<root>/<case>/<group>/*.dcm` and
`<root>/<case>/<group>.ome.zarr`. `get_infos` reads only metadata, OME patch
reads touch only selected chunks, and DICOM patch reads decode only selected
slices. In workflow YAML use `./Dataset:dicom` or `./Dataset:omezarr` in
`dataset_filenames`.

## Patching, streaming & reassembly

The data layer (`konfai/data/patching/`, `konfai/data/data_manager/`) never
loads a whole volume when it can avoid it:

- **`DatasetPatch`** (the `Patch:` config block): `patch_size` (default
  `[128,128,128]`), `overlap` (`None` → auto-tiling), `pad_value` (`None` → pad
  with `data.min()`), `extend_slice` (2.5-D context, only when `patch_size[0]==1`).
- **`ModelPatch`**: patching applied *inside* a model graph, with a
  `patch_combine` blender for overlap reassembly: `Mean`, `Cosinus`, `Trim`, or
  `Gaussian` (nnU-Net-style importance weighting).
- **Streaming** reads a planned source region through `read_data_slice`.
  Transform and sampled-augmentation locality determines whether that region is
  exact, haloed, index-remapped, cropped, rescaled, or unavailable. When the
  chain cannot be represented by one bounded region, KonfAI uses the safe
  whole-volume buffer. See {doc}`../../concepts/streaming` for the planner.
- **`Accumulator`** reassembles patches with overlap blending, correcting border
  voxels covered by fewer patches. Patch **read order must match write order**: a load-bearing invariant.

```{important}
For PREDICTION / EVALUATION, **all patches of a case stay on the same DDP rank**
(the whole volume is reassembled per rank). For TRAIN, shards are padded to the
same length so every rank executes the same number of backward passes.
```

## Three things to know

```{warning}
**`dcm` is not `dicom`.** The token `dcm` reads a **single file** through
SimpleITK; only the literal token `dicom` uses the series backend. The two look
alike and read different things.
```

`.vtk` I/O raises `ImportError` when `vtk` is missing. `konfai[vtk]` installs it,
but no extra declares it for the sidecar path, so the requirement is not
advertised where you meet it.

An `Attribute` round-trips **flat scalars and 1-D arrays only**. The sidecar
stringifies each value and reparses it with `np.fromstring`, so anything
multi-dimensional does not survive a read. Geometry is safe because `Origin` and
`Spacing` are 1-D and `Direction` is stored flattened.

## Next steps

- {doc}`../../concepts/datasets`: grouped dataset layout, selectors, patching
- {doc}`../../concepts/streaming`: locality declarations, planner rules, and fallbacks
- {doc}`transforms`: transform capabilities and streamability
- {doc}`../api/extension-points`: adding your own backend (the `AbstractFile`
  declarations and the `BACKENDS` registry)

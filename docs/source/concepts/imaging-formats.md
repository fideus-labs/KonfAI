# Imaging Format Readers

Beyond the SimpleITK formats handled by `konfai/utils/ITK.py`, KonfAI ships two
optional readers for formats common in clinical and bioimaging workflows: DICOM
series and OME-Zarr (OME-NGFF) stores. Both live in `konfai/utils/`, are wired
into `konfai.utils.dataset.Dataset`, and support channel-first, geometry-aware
reads and writes.

Each reader is an **optional dependency** — install only what a given dataset
needs.

## DICOM

Install the reader:

```bash
pip install "konfai[dicom]"   # pulls in pydicom
```

A DICOM acquisition is not a folder of independent images: a CT or MRI *series*
is a set of `.dcm` files that together define one 3-D volume. The reader handles
series discovery, slice ordering, geometry extraction, and CT intensity rescale.

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
(`Origin`, `Spacing`, `Direction`; see {doc}`datasets`), so a DICOM series travels
through the pipeline under the same geometry contract as any other format.

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

## OME-Zarr

Install the reader:

```bash
pip install "konfai[omezarr]"   # pulls in zarr + ngff-zarr
```

Unlike a NIfTI file, an OME-Zarr array is **already lazy**: it is stored as
chunked Zarr, so reading a sub-region only fetches the chunks it touches. This
maps naturally onto KonfAI's patch-based loading — the reader never materializes
the whole volume.

### Multiscale levels

OME-NGFF stores a **multiscale pyramid**: the same image at several resolutions,
level `0` being full resolution and each higher level a downsampled copy. Each
level carries its own physical `scale` (spacing) and `translation` (origin) in the
`.zattrs` metadata, so geometry stays correct at every level.

### `read_ome_zarr_slice`

```python
from konfai.utils.ome_zarr import read_ome_zarr_slice

patch, axes, scale, translation = read_ome_zarr_slice(
    "image.zarr",
    (slice(0, 64), slice(0, 256), slice(0, 256)),   # (Z, Y, X)
    level=0,
)
```

`konfai.utils.ome_zarr.read_ome_zarr_slice(store_path, slices, *, level=0, channel=None, timepoint=0)`
reads one spatial patch and returns:

| Returned | Meaning |
| --- | --- |
| `patch` | channel-first `(C, Z, Y, X)` patch preserving the stored dtype |
| `axes` | axis names from the OME-NGFF metadata (e.g. `['z', 'y', 'x']`) |
| `scale` | voxel spacing for the selected level |
| `translation` | origin translation for the selected level |

The reader inspects the stored `axes` to place the spatial slices on the right
dimensions and to index optional `T` (time) and `C` (channel) axes, so the same
call works for `ZYX`, `CZYX`, and `TCZYX` arrays. Local `.zarr` directories and
remote stores (`s3://`, `gs://`) are both supported.

### Choosing a resolution: `select_level`

`konfai.utils.ome_zarr.select_level(zattrs, target_spacing_mm=None)` picks the
pyramid level whose voxel spacing is closest to a target:

```python
import zarr
from konfai.utils.ome_zarr import select_level, read_ome_zarr_slice

zattrs = dict(zarr.open("image.zarr", mode="r").attrs)
level = select_level(zattrs, target_spacing_mm=1.0)   # nearest level to 1 mm
patch, *_ = read_ome_zarr_slice("image.zarr", (slice(0, 64),) * 3, level=level)
```

With `target_spacing_mm=None` it returns level `0` (full resolution). Otherwise it
compares the median spatial scale of each level against the target and returns the
closest — letting you trade resolution for field of view at a fixed patch size.

Use `konfai.utils.ome_zarr.get_ome_zarr_info(store_path, level=0)` for a metadata
summary (`axes`, `shape`, `chunks`, `dtype`, `scale`, `translation`, `n_levels`)
without reading any pixels.

`write_ome_zarr` writes a single-level OME-NGFF store with channel/spatial axes,
chunking, scale, translation, and the original KonfAI attributes.

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

## Why `ngff-zarr` over `ome-zarr`

Two Python libraries read OME-NGFF: the OME consortium's `ome-zarr` and
`ngff-zarr`. KonfAI's `omezarr` extra depends on **`ngff-zarr`** (alongside raw
`zarr`) for one decisive reason:

> `ngff-zarr` exposes **per-scale physical coordinates** — the `scale` and
> `translation` of each pyramid level — as plain numeric arrays that convert
> directly to SimpleITK geometry. That is exactly the `(Origin, Spacing,
> Direction)` triple KonfAI stores in an `Attribute`, so OME-Zarr input lines up
> with every other format without a bespoke geometry adapter.

It also handles multiscale selection transparently and adds helpers tuned for 3-D
medical/bioimage workflows. When `ngff-zarr` is unavailable, the reader falls back
to parsing the `.zattrs` JSON with `zarr` alone.

## See also

- {doc}`datasets` — the dataset and `Attribute` model these readers feed
- {doc}`configuration`
- {doc}`../getting-started/installation` — the optional-extras table

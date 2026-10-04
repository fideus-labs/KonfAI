# Storage formats

A dataset is named by a path and a format: `./Dataset:mha` in `dataset_filenames`. The format picks the
backend that reads and writes it (`konfai/utils/dataset/`). {doc}`../../config_guide/index` describes the
case and group layout.

| Backend | Formats | Layout | Region reads | Region writes | Extra |
| --- | --- | --- | --- | --- | --- |
| `SitkFile` | `mha`, `mhd`, `nii`, `nii.gz`, `nrrd`, `nrrd.gz`, `gipl`, `hdr`, `img`, `dcm`, `tif`, `png`, `jpg`, `bmp`, `itk.txt`, `fcsv`, `xml`, `vtk`, `npy` | one file per case and group (the default) | MetaImage, NIfTI and NRRD (compressed ones through a decompressed copy) | `mha`, `nii`, `nii.gz` and `nrrd` | `konfai[itk]` |
| `H5File` | `h5` | one HDF5 file for the whole dataset | yes | yes | `konfai[hdf5]` |
| `OmeZarrFile` | `omezarr` (also `ome-zarr`, `zarr`), `omezarr@<level>` | one store per case and group | yes | yes, pyramids included | `konfai[omezarr]` |
| `DicomFile` | `dicom` | one series folder per case and group | slice by slice | no | `konfai[dicom]` |
| `ItkTransformFile` | `itktransform` | one ITK transform file (`<group>.h5`) per case | for displacement fields | for displacement fields | `konfai[itk]`, `konfai[hdf5]` |

`pip install "konfai[imaging]"` installs every backend. A format without region reads still gives the
right result, only slower: the whole volume is decoded for each region ({doc}`../../usage/large-images`).

```{warning}
`dcm` reads a single file through SimpleITK; `dicom` reads a series. They look alike and read different
things.
```

## Compressed files

A `.nii.gz`, a compressed MetaImage or a compressed NRRD cannot be read by region: the whole stream must be
decoded from the start. So the first time a run reads a region of such a file, KonfAI decompresses it once into
an uncompressed copy and reads every region from the copy. Values and geometry are unchanged.

- **Where:** `~/.cache/konfai/decompressed/`, one folder per run. Set `KONFAI_DECOMPRESSED_DIRECTORY` to use
  another disk (avoid `/tmp` if it is in memory).
- **How much:** each copy takes the uncompressed size of its volume. Prediction, evaluation and transform
  delete a case's copies when they are done with it; training keeps them for the run. The copies never take
  more than half the free space: past that, KonfAI reads the compressed file directly and warns.
- **When they go:** at the end of the run, even after an error. A folder left by a killed run is removed by
  the next run on the same machine.

A file read whole anyway (a training cache, a stage that needs the whole volume) gets no copy.

## Reading from object storage

A `dataset_filenames` entry can be a URI:

```yaml
Dataset:
  dataset_filenames:
    - s3://aind-open-data/exaSPIM_822174_2026-04-28_12-29-55_processed_2026-07-09_03-49-09:omezarr
```

```bash
pip install "konfai[s3]"
FSSPEC_S3_ANON=true konfai TRANSFORM --config Transform.yml
```

That bucket is public, hence `FSSPEC_S3_ANON=true`. Credentials use fsspec's and AWS's usual settings:
`AWS_PROFILE`, `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, `FSSPEC_S3_ENDPOINT_URL` for MinIO.

- Only `:omezarr` reads a remote root.
- A remote root is read-only: `Write` to a local path.
- Give the run a generous `memory_budget`: the chunk cache is a third of it, and every cache miss is a
  download.
- A root that cannot be reached (a wrong bucket, an expired credential) is an error, not an empty dataset.

## HDF5

An `:h5` dataset is one file holding every case. HDF5 does not reclaim the space of a replaced entry, so
rewriting entries grows the file; `h5repack <in> <out>` writes a compact copy. Several processes cannot
write the same `.h5`: use a folder format with `--cpu N`.

## ITK transforms

`:itktransform` stores one ITK transform per case (`<case>/<group>.h5`), the file `sitk.WriteTransform`
would write. A displacement field is written region by region, so a registration can `Write` its transform
like any image. `itktransform` names the backend: files are `.h5` or `.tfm`, never `.itktransform`.

## Images, landmarks and other files

`SitkFile` also reads and writes `.itk.txt` (transforms), `.fcsv` (Slicer landmarks), `.xml` (attributes),
`.vtk` (points, needs `konfai[vtk]`) and `.npy`.

- A pixel type the format cannot hold (a `bool` volume, a float `png`) is refused with the reason. Cast
  booleans to `uint8`, or use `h5` or `omezarr`, which keep them.
- A volume the format would store only part of (several channels as `gipl`, a 3-D volume as `png`) is
  refused too. Write it as `mha`, `nrrd`, `h5` or `omezarr`, which hold any.
- Landmarks are read and written in LPS. A `.fcsv` whose `# CoordinateSystem` is `RAS` is converted; `LPS`,
  `0`, `1` or no line are read as LPS. A file from 3D Slicer older than 4.11 writes `0` over RAS points: set
  its line to `RAS`.

## DICOM series

A DICOM series is a folder of slices forming one volume. Each case holds `<case>/<group>/*.dcm`.

```python
from konfai.utils.dicom import discover_series, read_dicom_series

series = discover_series("path/to/study")          # {uid: [Path, ...]}
volume, origin, spacing, direction = read_dicom_series("path/to/study", series_uid=next(iter(series)))
```

`read_dicom_series` returns the volume as `(1, Z, Y, X)` float32 and its geometry.

- Slices are ordered by their position along the slice normal, not by file name.
- `apply_rescale=True` (default) applies `RescaleSlope`/`RescaleIntercept` (Hounsfield units for CT). Use
  `False` to keep raw integers, for label maps.
- `MONOCHROME1` is not inverted (the single-file `dcm` reader does invert it).
- A folder with several series needs `series_uid`; without it, the error lists the available ones.
- Colour slices, mixed orientations and several slices at one position (cine, phases, echoes) are refused.

`write_dicom_series` writes an uncompressed series: integers exactly, floats as 16-bit with a rescale. A
left-handed volume reads back right-handed (flipped along z), with every world point at the same place.

## OME-Zarr

An OME-Zarr store is chunked, so reading a region fetches only the chunks it touches.

```python
from konfai.utils.dataset import Dataset

dataset = Dataset("Dataset", "omezarr")
shape, attributes = dataset.get_infos("Volume", "CASE_001")      # metadata only
patch, patch_attributes = dataset.read_data_slice(
    "Volume", "CASE_001", (slice(None), slice(0, 64), slice(0, 256), slice(0, 256))
)
```

`read_data_slice` returns a channel-first `(C, Z, Y, X)` array and the geometry of that window.
`Dataset("Dataset", "omezarr@1")` reads pyramid level 1. `konfai.utils.ome_zarr.get_ome_zarr_info` gives a
store's axes, shape, chunks, scale and levels without reading pixels.

KonfAI writes NGFF 0.5 (zarr v3); displacement fields are written as NGFF 0.6. Older KonfAI stores (zarr
v2) are read as they are. A store written region by region is chunked like the regions the writer used,
which depends on the machine's memory budget; the values do not.

### Multiscale levels

`scale_factors` on `Save`/`Write` writes a pyramid. Each level is computed from the one above:

| Level 0 dtype | Coarser levels |
| --- | --- |
| `uint8`, `int64`, `bool` | the most frequent value of each window |
| any other | the mean of each window |

So a label map keeps real labels at every level. An `int16` label map would be averaged: store it as
`uint8`, or set `downsample_method` (an `ngff_zarr.Methods` name such as `ITKWASM_LABEL_IMAGE`).

### Units

KonfAI works in millimetres. A store that declares its unit (ExaSPIM volumes are in micrometres) is
converted when read and written back in its own unit. A store without a unit is taken as millimetres, and
KonfAI writes `millimeter` into the stores it creates.

KonfAI uses `ngff-zarr` because it gives each pyramid level's scale and translation as numbers, which map
directly onto the image geometry.

## Geometry

Each case travels with an `Attribute`, a dictionary holding its geometry in millimetres: `Origin`,
`Spacing` (in `x, y, z` order) and `Direction` (the flattened direction matrix). A prediction is written
back in the geometry of its input through it. Values are stored as strings, so only numbers and 1-D arrays
survive a write and a read; read them with `get_np_array(key)`.

## Next steps

- {doc}`../../usage/large-images`: streaming and what each format costs.
- {doc}`../../usage/custom-models`: adding a format of your own.

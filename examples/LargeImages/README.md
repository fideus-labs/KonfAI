# Large images

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fideus-labs/KonfAI/blob/main/examples/LargeImages/LargeImages_demo.ipynb)

A 2.4 GB public OME-Zarr volume (AIND ExaSPIM specimen 822174, CC BY 4.0) read where it lives on
S3, chunk by chunk: the metadata, a coarse overview, one native window, then a TRANSFORM that
streams a pyramid level to a local HDF5 under a memory budget. Open
`LargeImages_demo.ipynb` (Colab-ready). Needs `konfai[imaging,s3]` and
`FSSPEC_S3_ANON=true`; the store is
`s3://aind-open-data/exaSPIM_822174_2026-04-28_12-29-55_processed_2026-07-09_03-49-09/fusion2halves/SPIM.ome.zarr`.

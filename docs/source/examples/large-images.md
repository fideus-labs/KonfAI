```{include} ../../../examples/LargeImages/README.md
```

## Why this example

The other examples fetch their data first. This one reads a 2.4 GB volume that
stays on a public S3 bucket: the pyramid's metadata, a coarse level as an
overview, one native-resolution window (one chunk), then a `TRANSFORM` that
streams a level to a local HDF5 under a memory budget, region by region along
the chunk grid. It is the {doc}`../usage/large-images` guide run for real, on a
store anyone can reach, with `FSSPEC_S3_ANON=true` standing in for credentials.

## Next steps

- {doc}`../usage/large-images`: chunk shapes, patch sizes, workers, the memory budget
- {doc}`../reference/components/storage-backends`: the `:omezarr` backend, its selectors and its URI support
- {doc}`transform`: dataset preparation on local data, with the plan every run prints

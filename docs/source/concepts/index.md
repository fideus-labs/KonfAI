# Core concepts

Nothing here is needed for a first run — the {doc}`../quickstart` stands on its
own. Read this section once that run worked and you want to know what it did, or
when you hit something the task guides assume. {doc}`configuration` is the one
page that pays off immediately; the rest are on-demand.

KonfAI is easiest to understand when you keep five ideas in mind:

1. **YAML builds Python objects** rather than acting as a loose parameter blob.
2. **Datasets are organized by groups** such as `CT`, `MR`, `SEG`, or `MASK`.
3. **Volumes are read as patches**, and a preprocessing chain that allows it is
   streamed from disk rather than loaded.
4. **Model outputs are addressable by module path**, which is how losses,
   metrics, and exported predictions are attached.
5. **The same low-level workflow can later be packaged as a KonfAI App**.

Two topics have moved out of this section: imaging-format specifics (DICOM,
OME-Zarr) now live at {doc}`../reference/components/storage-backends`, and
packaging a finished workflow as a KonfAI App is covered in
{doc}`../usage/apps`.

```{toctree}
:maxdepth: 1

configuration
datasets
streaming
model-graph
yaml-model-builder
execution-flow
```

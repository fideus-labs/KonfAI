Full module reference
=====================

This section documents the modules that make up KonfAI. It is intentionally broader
than the curated API pages and therefore includes lower-level helpers used by
extension authors. The stubs are **hand-maintained**, not generated, so the coverage
is partial: `konfai.data`, `konfai.metric` and `konfai.network` are complete, and
`konfai.utils` now covers `errors` (the `KonfAIError` taxonomy) and `pretrained`
(the weights bridge behind `Model.pretrained_from`). `konfai.models.**`, a few
`konfai.utils` helpers (`runtime`, `model_builder`, `ome_zarr`, `dicom`, `vram`,
`live_control`, `catalog`) and `konfai.export` have no page yet. Read them from
the source until they do; `konfai list models` prints the model catalog.

.. toctree::
   :maxdepth: 4

   konfai

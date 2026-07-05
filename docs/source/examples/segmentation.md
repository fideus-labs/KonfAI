```{include} ../../../examples/Segmentation/README.md
```

## In the docs

**Docs notes.** Training also writes TensorBoard statistics to
`Statistics/SEG_BASELINE/` alongside the checkpoint, prediction, and
evaluation folders. `UNet.yml` defines the routed KonfAI UNet graph through
`add_module` metadata. The class count is set by the `nb_class` config key:
if your dataset is not a `0..40` label map, update both `nb_class` and the
Dice labels together.

Next steps:

- {doc}`../quickstart` — a minimal first end-to-end run outside the examples
- {doc}`../config_guide/training` — reference for the training-side configuration keys

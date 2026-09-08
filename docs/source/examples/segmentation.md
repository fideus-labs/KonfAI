```{include} ../../../examples/Segmentation/README.md
```

The tutorial above teaches the low-level `SEG_BASELINE` workflow; it ships no
trained checkpoint. To see what a mature segmentation model produces on a real
case, {doc}`../usage/apps` shows a completed TotalSegmentator App run on a
synthetic CT, and {doc}`visual-gallery` the transform, augmentation and
registration evidence.

## In the docs

**Docs notes.** Training also writes TensorBoard statistics to
`Statistics/SEG_BASELINE/` alongside the checkpoint, prediction, and
evaluation folders. `UNet.yml` defines the routed KonfAI UNet graph through
`add_module` metadata. The class count is set by the `nb_class` config key:
if your dataset is not a `0..40` label map, update both `nb_class` and the
Dice labels together.

Next steps:

- {doc}`../quickstart`: this same example, walked step by step, with a success signal after each phase
- {doc}`../config_guide/training`: reference for the training-side configuration keys

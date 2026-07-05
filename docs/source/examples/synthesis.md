```{include} ../../../examples/Synthesis/README.md
```

## In the docs

**Docs notes.** `UnNormalize.py` contains a local transform used during
prediction. The two GAN patch scopes correspond to the full config paths
`Trainer.Dataset.Patch` (the global 3D chunk seen by the GAN) and
`Trainer.Model.Gan.UNetpp5.Patch` (the internal 2D/2.5D slices seen by the
generator). When adapting the example, also change the local model
definitions in `Model.py` if the built-in modules are not enough.

Next steps:

- {doc}`../concepts/model-graph` — how routed module graphs like the one in `Model.py` are composed
- {doc}`../usage/custom-models` — to write and reference your own modules through `classpath`
- {doc}`../config_guide/training` — reference for the training-side configuration keys

```{include} ../../../examples/BringYourModel/README.md
```

## Why this example

The three framework examples are YAML configs run by the `konfai` CLI. This one
is the other spelling: a model that already exists in Python (MONAI's `UNet`,
built as MONAI builds it) goes through `konfai.train_model` and
`konfai.predict_model`, and everything else is the same engine: the patch
sampling, the overlap-blended reassembly, the streamed writes, the checkpoint
format and the run record (`Statistics/MONAI_UNET/Trainer.yml`, the resolved
config the run would have read).

The calls are documented in {doc}`../usage/adopting-konfai` (Bring your model)
and {doc}`../usage/python-workflows`; `tests/unit/test_api.py` runs the same two
calls on a synthetic cohort and on a MONAI UNet, so the notebook's claim is
pinned by a test.

## What it leaves out on purpose

A live model runs on one rank, in the process that built it: several GPUs, and
a RESUME from another process, need the model spelled as a classpath
(`monai.networks.nets:UNet`) in a `Config.yml`, which is the
{doc}`segmentation` route. `Statistics/MONAI_UNET/Trainer.yml` is a starting
point for that file: it is the config the ten lines built.

## Next steps

- {doc}`segmentation`: the same task through a `Config.yml`
- {doc}`../usage/adopting-konfai`: the routes into KonfAI, from an import to a routed `Network`
- {doc}`../usage/python-workflows`: the whole API, and how the two spellings relate

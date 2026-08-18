```{include} ../../../examples/Transform/README.md
```

## Why start here

This is the shortest way to see KonfAI do something real. It is dataset
preparation, so there is nothing to train first and no dataset to download:
`make_dataset.py` writes 3.5 MB of synthetic volumes, and both configs run on
CPU in about a minute. `Transform_demo.ipynb` runs the whole thing cell by cell,
in Colab too.

It also shows the two things only this workflow does. `Transform.yml` folds a
cohort into one volume (N to 1), and `Transform_expand.yml` draws four copies of
every case (1 to N). Both print their plan before writing, which is where you see
each case routed to `STREAM`, `LOAD` or `WHOLE-VOLUME`.

The cohort it generates deliberately does **not** share a grid: six volumes with
their own extent, spacing and origin, which is the ordinary state of data as
acquired. That is what makes `Resample: {reference: …}` before the `Reduce` the
point of the example rather than a formality, and why `grid: strict` accepts the
fold instead of refusing it.

## Next steps

- {doc}`../usage/making-data`: the same capabilities as a guide, including
  writing an ITK transform or an OME-Zarr pyramid.
- {doc}`../config_guide/transform`: every key, every refusal, and the eleven
  worked configs the test suite runs.
- {doc}`../concepts/streaming`: what decides whether a chain streams.

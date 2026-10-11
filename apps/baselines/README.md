# Baselines

The registration baselines of [ImagingBenchmarks](https://github.com/fideus-labs/ImagingBenchmarks) as KonfAI
apps. Each app runs its authors' code at a pinned version, fetched on first use, with the settings the benchmark
runs it with written in its `Prediction.yml`: a wrapper, never a re-implementation. An app is checked against the
benchmark's own line before it is published.

```bash
pip install baselines-konfai
baselines-konfai register ConvexAdam_MIND -i fixed.nii.gz -i moving.nii.gz -o out --gpu 0
```

The apps live on Hugging Face under `fideus-labs/Baselines`; the folders in `bundles/` are their sources.
`BASELINES_KONFAI_REPO` points the command at a local directory of app folders instead, and
`BASELINES_KONFAI_CACHE` (default `~/.cache/baselines-konfai`) is where the authors' code and binaries land.

| app | runs | pinned at |
|---|---|---|
| `ConvexAdam_MIND` | `convex_adam_pt(fixed, moving)` of multimodallearning/convexAdam, defaults | `b229e52` |

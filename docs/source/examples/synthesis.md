```{include} ../../../examples/Synthesis/README.md
```

## See synthesis on real medical images

The low-level configurations above are learning templates. The cards below are
from a separate, completed **ImpactSynth App** run on de-identified SynthRAD
2025 Task 1 abdomen case `1ABB124` (CC BY-NC 4.0). Five checkpoints evaluated
the original MR and two test-time augmentations, producing 15 inference states.
All text and measurements remain in HTML; the PNG files contain medical pixels
only. Full attribution and hashes are in the
<a href="../_static/apps/ASSET_PROVENANCE.md">asset provenance manifest</a>.

<ul class="kf-example-grid kf-example-grid--compact" aria-label="Real ImpactSynth input, output, and paired reference">
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-synth/mr-input.png" aria-label="Open the real abdominal MR input"><img src="../_static/apps/impact-synth/mr-input.png" alt="Real abdominal MR plane used as input to the completed ImpactSynth App execution." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">01 · REAL APP INPUT</span><strong>MR input</strong><span>One extracted plane from the paired abdominal case.</span><span class="kf-example-stats">Z +18 MM · 2 MM GRID</span></figcaption></figure></li>
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-synth/synthetic-ct.png" aria-label="Open the real ImpactSynth synthetic CT"><img src="../_static/apps/impact-synth/synthetic-ct.png" alt="Synthetic CT plane produced by the completed five-checkpoint ImpactSynth App ensemble." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">02 · REAL APP OUTPUT</span><strong>ImpactSynth sCT</strong><span>Five checkpoints over the original MR and two TTA states.</span><span class="kf-example-stats">15 INFERENCE STATES</span></figcaption></figure></li>
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-synth/reference-ct.png" aria-label="Open the paired real CT reference"><img src="../_static/apps/impact-synth/reference-ct.png" alt="Paired real abdominal CT reference plane on the same physical geometry as the synthetic CT." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">03 · REAL REFERENCE</span><strong>Paired CT</strong><span>The real target remains separate from the generated image.</span><span class="kf-example-stats">SAME PHYSICAL PLANE · 2 MM GRID</span></figcaption></figure></li>
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-synth/mae-map.png" aria-label="Open the real ImpactSynth evaluation map"><img src="../_static/apps/impact-synth/mae-map.png" alt="Per-voxel absolute-error heat map from the completed ImpactSynth evaluation over the paired CT anatomy." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">04 · APP EVALUATION</span><strong>Absolute-error map</strong><span>Display range 0–438.20 HU (P99); case scores use the complete metric volume.</span><span class="kf-example-stats">MAE 22.94 HU · PSNR 34.16 DB · SSIM 0.913</span></figcaption></figure></li>
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-synth/uncertainty-map.png" aria-label="Open the real ImpactSynth uncertainty map"><img src="../_static/apps/impact-synth/uncertainty-map.png" alt="Reference-free ensemble-uncertainty heat map from the completed 15-state ImpactSynth App workflow over the MR anatomy." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">05 · APP UNCERTAINTY</span><strong>Ensemble uncertainty</strong><span>Display range 0–4520.81% of baseline (P99) across all 15 states.</span><span class="kf-example-stats">MEAN 109.61% BASELINE · DISAGREEMENT 0.016</span></figcaption></figure></li>
</ul>

<p class="kf-example-caption"><strong>One real case through prediction, evaluation, and uncertainty.</strong><span>MR → 15-state sCT → paired CT comparison · all outputs retain physical geometry</span></p>

See {doc}`../usage/apps` for the reproducibility snapshot and the downstream
TotalSegmentator and Slicer workflow.

## In the docs

**Docs notes.** `UnNormalize.py` contains a local transform used during
prediction. The two GAN patch scopes correspond to the full config paths
`Trainer.Dataset.Patch` (the global 3D chunk seen by the GAN) and
`Trainer.Model.Gan.UNetpp5.ModelPatch` (the internal 2D/2.5D slices seen by the
generator). When adapting the example, also change the local model
definitions in `Model.py` if the built-in modules are not enough.

Next steps:

- {doc}`../concepts/model-graph` — how routed module graphs like the one in `Model.py` are composed
- {doc}`../usage/custom-models` — to write and reference your own modules through `classpath`
- {doc}`../config_guide/training` — reference for the training-side configuration keys

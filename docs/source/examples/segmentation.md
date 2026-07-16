```{include} ../../../examples/Segmentation/README.md
```

## See segmentation on a real App output

The tutorial above teaches the low-level `SEG_BASELINE` workflow; it does not
ship a trained checkpoint or claim the result below. These two cards instead
show a separate, completed **TotalSegmentator KonfAI App** execution on the real
synthetic CT produced by ImpactSynth. They illustrate the path from a mature
segmentation model to a medical label map without pretending it is the expected
output of the small 41-class tutorial UNet.

<ul class="kf-example-grid kf-example-grid--compact" aria-label="Separate real TotalSegmentator App input and output">
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-synth/synthetic-ct.png" aria-label="Open the real synthetic CT input to TotalSegmentator"><img src="../_static/apps/impact-synth/synthetic-ct.png" alt="Real abdominal synthetic CT plane used as input to the completed TotalSegmentator App execution." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">SEPARATE APP INPUT</span><strong>ImpactSynth sCT</strong><span>The downstream segmentation App receives a medical CT volume on the 2 mm reference grid.</span><span class="kf-example-stats">REAL APP ARTIFACT · 2 MM GRID</span></figcaption></figure></li>
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-synth/totalsegmentator.png" aria-label="Open the real TotalSegmentator anatomy output"><img src="../_static/apps/impact-synth/totalsegmentator.png" alt="Real TotalSegmentator five-model anatomy labels overlaid on the abdominal synthetic CT plane." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">SEPARATE APP OUTPUT</span><strong>Total anatomy label map</strong><span>The full five-model ensemble materialises anatomical labels on the input geometry.</span><span class="kf-example-stats">FULL TOTAL OVERLAY · 5 MODELS</span></figcaption></figure></li>
</ul>

<p class="kf-example-caption"><strong>Real App evidence, not a fabricated tutorial result.</strong><span>ImpactSynth sCT → five-model TotalSegmentator → medical label-map dataset</span></p>

A separate one-checkpoint `total-3mm` evaluation branch scored Dice `0.665` on
this case. That number does not score the five-model `total` overlay displayed
above. The source is de-identified SynthRAD 2025 Task 1 abdomen case `1ABB124`
(CC BY-NC 4.0); see the
<a href="../_static/apps/ASSET_PROVENANCE.md">asset provenance manifest</a>.

See {doc}`../usage/apps` for the completed App workflow and {doc}`visual-gallery`
for the real transform, augmentation, and registration evidence.

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

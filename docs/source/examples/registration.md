```{include} ../../../examples/Registration/README.md
```

## See registration on real medical images

The four cards below come from a separate, executed IMPACT-Reg App run on
de-identified SynthRAD 2025 Task 1 abdomen case `1ABB123` (CC BY-NC 4.0).
They are **not outputs from the VoxelMorph tutorial
above**. This section demonstrates the packaged App path on real
medical images; the tutorial remains the small, reproducible learning exercise.
Full attribution and hashes are in the
<a href="../_static/apps/ASSET_PROVENANCE.md">asset provenance manifest</a>.

<ul class="kf-example-grid kf-example-grid--registration" aria-label="Real IMPACT-Reg execution stages, separate from the VoxelMorph tutorial">
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-reg/moving-before.png" aria-label="Open the real moving MR before registration"><img src="../_static/apps/impact-reg/moving-before.png" alt="Coronal view of the real moving abdominal MR before registration, with fixed CT contours showing the controlled spatial offset." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">01 · REAL APP INPUT</span><strong>Moving MR: before</strong><span>Fixed-CT contours expose the controlled metadata-only offset.</span><span class="kf-example-stats">NCC 0.129 · MAE 106.11</span></figcaption></figure></li>
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-reg/fixed-ct.png" aria-label="Open the real fixed CT target"><img src="../_static/apps/impact-reg/fixed-ct.png" alt="Coronal view of the real fixed abdominal CT that defines the registration target and output geometry." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">02 · REAL REFERENCE</span><strong>Fixed CT target</strong><span>The reference image defines the physical output grid.</span><span class="kf-example-stats">222 × 226 × 124 · 2 MM GRID</span></figcaption></figure></li>
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-reg/moved-after.png" aria-label="Open the real moved MR after registration"><img src="../_static/apps/impact-reg/moved-after.png" alt="Coronal view of the real moved abdominal MR after ConvexAdam Composite registration on the fixed CT grid." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">03 · REAL APP OUTPUT</span><strong>Moved MR: after</strong><span><code>ConvexAdam_Composite</code> writes the moved image on the fixed grid.</span><span class="kf-example-stats">NCC 0.937 · MAE 21.09</span></figcaption></figure></li>
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-reg/displacement-field.png" aria-label="Open the physical displacement-field visualization"><img src="../_static/apps/impact-reg/displacement-field.png" alt="Visualization of the real three-component displacement field, with physical magnitude and sampled in-plane vectors." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">04 · PHYSICAL FIELD</span><strong>Displacement field</strong><span>Three physical components in millimetres, with sampled vectors.</span><span class="kf-example-stats">MEAN 23.06 MM · P95 25.55 MM</span></figcaption></figure></li>
</ul>

<p class="kf-example-caption"><strong>One real pair, one completed IMPACT-Reg App execution.</strong><span>Controlled origin offset · <code>ConvexAdam_Composite</code> · NCC 0.129 → 0.937 · moved image + DVF + reusable transform</span></p>

The {ref}`registration gallery <gallery-registration>` presents the same
execution alongside the transform and augmentation evidence. Its generator
validates the fixed-grid geometry and reads the completed medical-image
artifacts; it does not synthesize a decorative before/after result.

See {doc}`../usage/apps` for the preset, measured before/after similarity,
fixed-grid geometry checks, reusable transform, and Slicer delivery path.

# Using KonfAI Apps

One command runs a published medical model on your data:

```bash
konfai-apps infer VBoussot/ImpactSynth:CBCT -i input.mha -o ./Output --gpu 0
```

The app brings its weights, preprocessing and evaluation, and the result comes back on your input's
geometry.

```{warning}
Resolving an app **copies and imports its `.py` files**, so it runs arbitrary
code, and it pip-installs its `requirements.txt` by default
(`KONFAI_APPS_INSTALL_REQUIREMENTS=0` opts out). Only run apps from sources you
trust.
```

## What ships today

Figures for the medium case (249 × 246 × 246) of each bundle's README, on an RTX PRO 5000 24 GB
(`benchmarks/perf/bench_apps.py`, 2026-09-09). The rows measure different tasks and are not comparable.

| App | Workload | Measured |
| --- | --- | --- |
| `TotalSegmentator-KonfAI` | CT → 117 labels (`total`: 5 models, `total-3mm`: 1), MRI → 50 labels (`total_mr`: 2, `total_mr-3mm`: 1) | **17.7 s / 5.2 GB RAM**, 15.4 GB VRAM, against the original's 58.3 s / 25.2 GB RAM |
| `MRSegmentator-KonfAI:MRSegmentator` | MRI → 40 labels, five-fold ensemble | **21 s / 5.3 GB RAM**, 16.3 GB VRAM, against the original's 24 s / 8.5 GB RAM |
| `ImpactSeg:body` | one CT/MR/CBCT model → 11 structures | 3.3 s, 1.7 GB RAM, 3.9 GB VRAM |
| `ImpactSynth` | three MR/CBCT→sCT variants, five models each | `MR`: 24.6 s, 2.7 GB RAM, 12.8 GB VRAM |
| `ImpactReg:FireANTs_SyN` | fixed + moving → moved image and displacement field on the fixed grid | 108 s, 6.3 GB RAM, 16.0 GB VRAM |

Each app has a notebook that downloads a demo case and shows the result ({doc}`../examples/index`).

### The command each ships

| App CLI | Task | Modality | Models |
| --- | --- | --- | --- |
| `impact-synth-konfai synthesize` | Synthetic CT (sCT) | MR → CT, CBCT → CT | `MR`, `CBCT`, `MR_CBCT` |
| `impact-seg-konfai segment` | Multimodal body segmentation (11 labels) | CBCT / MR / CT | `body` |
| `mrsegmentator-konfai segment` | Whole-body MRI segmentation | MRI | folds 1–5 |
| `totalsegmentator-konfai segment` | Whole-body CT/MRI segmentation | CT / MRI | `total`, `total_mr`, 3 mm variants |
| `impact-reg-konfai register` | Multimodal deformable registration | MR/CT, CBCT/CT | presets |

Each also has `eval`, `uncertainty` (except TotalSegmentator) and `pipeline` ({doc}`../reference/cli`).

### One real case, end to end

SynthRAD 2025 case `1ABB124` (CC BY-NC 4.0, <a href="../_static/apps/ASSET_PROVENANCE.md">provenance</a>):
ImpactSynth turns the MR into a synthetic CT, TotalSegmentator segments that CT, and KonfAI evaluates and
estimates uncertainty. Every panel is a real output on the same plane.

<ul class="kf-example-grid kf-example-grid--compact" aria-label="Completed real-data KonfAI App workflow stages">
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-synth/mr-input.png" aria-label="Open the real abdominal MR input"><img src="../_static/apps/impact-synth/mr-input.png" alt="Real abdominal MR plane used as input to the completed ImpactSynth App execution." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">01 · INPUT</span><strong>MR input</strong><span>One extracted plane from the paired abdominal case.</span><span class="kf-example-stats">Z +18 MM · 2 MM GRID</span></figcaption></figure></li>
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-synth/synthetic-ct.png" aria-label="Open the real ImpactSynth synthetic CT"><img src="../_static/apps/impact-synth/synthetic-ct.png" alt="Synthetic CT plane produced by the completed five-checkpoint ImpactSynth App ensemble." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">02 · PREDICTION</span><strong>ImpactSynth sCT</strong><span>Five checkpoints over the original MR and two TTA states.</span><span class="kf-example-stats">15 INFERENCE STATES</span></figcaption></figure></li>
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-synth/reference-ct.png" aria-label="Open the paired real CT reference"><img src="../_static/apps/impact-synth/reference-ct.png" alt="Paired real abdominal CT reference plane on the same physical geometry as the synthetic CT." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">03 · REFERENCE</span><strong>Paired CT</strong><span>The real target stays separate from the generated image.</span><span class="kf-example-stats">SAME PHYSICAL PLANE · 2 MM GRID</span></figcaption></figure></li>
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-synth/totalsegmentator.png" aria-label="Open the real TotalSegmentator anatomy output"><img src="../_static/apps/impact-synth/totalsegmentator.png" alt="Real TotalSegmentator five-model anatomy labels overlaid on the abdominal synthetic CT plane." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">04 · DOWNSTREAM APP</span><strong>Total anatomy</strong><span>The full TotalSegmentator ensemble runs directly on the sCT artifact.</span><span class="kf-example-stats">FULL TOTAL OVERLAY · 5 MODELS</span></figcaption></figure></li>
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-synth/mae-map.png" aria-label="Open the real ImpactSynth evaluation map"><img src="../_static/apps/impact-synth/mae-map.png" alt="Per-voxel absolute-error heat map from the completed ImpactSynth evaluation over the paired CT anatomy." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">05 · EVALUATION</span><strong>Absolute-error map</strong><span>Display range 0–438.20 HU (P99); case scores use the complete metric volume.</span><span class="kf-example-stats">MAE 22.94 HU · PSNR 34.16 DB · SSIM 0.913</span></figcaption></figure></li>
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-synth/uncertainty-map.png" aria-label="Open the real ImpactSynth uncertainty map"><img src="../_static/apps/impact-synth/uncertainty-map.png" alt="Reference-free ensemble-uncertainty heat map from the completed 15-state ImpactSynth App workflow over the MR anatomy." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">06 · UNCERTAINTY</span><strong>Ensemble uncertainty</strong><span>Display range 0–4520.81% of baseline (P99) across the 15 states.</span><span class="kf-example-stats">MEAN 109.61% BASELINE · DISAGREEMENT 0.016</span></figcaption></figure></li>
</ul>

## Running one

An app exposes four operations, all on the same package:

```bash
konfai-apps infer       APP -i input.mha -o ./Output --gpu 0
konfai-apps eval        APP -i prediction.mha --gt ct.mha --mask mask.mha
konfai-apps uncertainty APP -i input.mha -o ./Output
konfai-apps pipeline    APP -i input.mha --gt ct.mha -o ./Output -uncertainty
```

`pipeline` runs inference, evaluation and uncertainty in one call.

| Family | Input | Result | Also available |
| --- | --- | --- | --- |
| Segmentation | CT, MR or CBCT | label map on the input grid | Dice evaluation, ensemble/TTA uncertainty |
| Synthesis | MR or CBCT | synthetic CT with the reference geometry | masked MAE/SSIM evaluation, uncertainty |
| Registration | fixed + moving | moved image, displacement field, transform | image, label and landmark evaluation, field spread |

An app is a local folder or a Hugging Face reference `owner/repository:app`, such as
`VBoussot/ImpactSynth:MR`. `owner/repository@rev:app` pins a revision. Without one, a release of
konfai-apps uses the repository's tag for its own version (`v1.9.0`) when it exists, and `main`
otherwise, so each release runs the configs made for it.

The first listing of a repository (in Studio, the MCP server, Slicer or `konfai-apps`) downloads every
app's files except the checkpoints; later ones only ask the Hub for the file list. Offline
(`HF_HUB_OFFLINE=1`), or when the Hub does not answer, the apps already downloaded are listed.
`--force_update` downloads again. Checkpoints are downloaded when an app first runs; `konfai-apps download`
fetches everything ahead of time.

Repeat `-i` for each input group (an image and a mask, for example). The same operations are available
from Python through `konfai_apps.KonfAIApp`.

### Registration

IMPACT-Reg has thirteen presets (rigid, B-spline, MR/CT and CBCT/CT, ConvexAdam, FireANTs). A preset
writes the moved image and the displacement field on the fixed image's grid. Several presets can be
combined, and the spread between them gives an uncertainty map.

```bash
impact-reg-konfai register MR_CT_MRSeg MR_CT_TS \
  -f fixed_ct.mha -m moving_mr.mha \
  --uncertainty -o ./Registration --gpu 0
```

<ul class="kf-example-grid kf-example-grid--registration" aria-label="Completed real-data IMPACT-Reg stages">
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-reg/moving-before.png" aria-label="Open the real moving MR before registration"><img src="../_static/apps/impact-reg/moving-before.png" alt="Real moving abdominal MR before registration, with fixed CT contours showing the controlled spatial offset." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">01 · MOVING INPUT</span><strong>Moving MR: before</strong><span>Fixed-CT contours expose the controlled metadata-only offset.</span><span class="kf-example-stats">NCC 0.129 · MAE 106.11</span></figcaption></figure></li>
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-reg/fixed-ct.png" aria-label="Open the real fixed CT target"><img src="../_static/apps/impact-reg/fixed-ct.png" alt="Real fixed abdominal CT defining the registration target and output geometry." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">02 · FIXED REFERENCE</span><strong>Fixed CT target</strong><span>The reference image defines the physical output grid.</span><span class="kf-example-stats">222 × 226 × 124 · 2 MM GRID</span></figcaption></figure></li>
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-reg/moved-after.png" aria-label="Open the real moved MR after registration"><img src="../_static/apps/impact-reg/moved-after.png" alt="Real moved abdominal MR after ConvexAdam Composite registration on the fixed CT grid." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">03 · MOVED OUTPUT</span><strong>Moved MR: after</strong><span><code>ConvexAdam_Composite</code> writes the result on the fixed grid.</span><span class="kf-example-stats">NCC 0.937 · MAE 21.09</span></figcaption></figure></li>
  <li><figure class="kf-example-card"><a class="kf-example-media" href="../_static/apps/impact-reg/displacement-field.png" aria-label="Open the real physical displacement field"><img src="../_static/apps/impact-reg/displacement-field.png" alt="Real three-component displacement field visualized with physical magnitude and sampled in-plane vectors." width="422" height="350" loading="lazy" decoding="async"></a><figcaption><span class="kf-example-step">04 · PHYSICAL FIELD</span><strong>Displacement field</strong><span>Three physical components in millimetres, with sampled vectors.</span><span class="kf-example-stats">MEAN 23.06 MM · P95 25.55 MM</span></figcaption></figure></li>
</ul>

### Fine-tuning

```bash
konfai-apps fine-tune APP NAME -d ./Dataset --epochs 10 --gpu 0
konfai-apps fine-tune APP NAME -d ./Dataset --models CV_0 CV_1 --epochs 10 --gpu 0
```

Training restarts from each selected checkpoint (`--models`, the first by default) with a fresh optimizer
and schedule, for `--epochs` epochs. The result is a new app. The run works in `./Output` and links your
dataset there, so it refuses a folder that already has a `Dataset` of its own.

## Running on another machine

Add `--host` to any command to run it on a server: the inputs are uploaded, the logs streamed, and the
result downloaded. The server needs `pip install "konfai-apps[server]"`.

```bash
echo '{"apps": ["VBoussot/ImpactSynth:CBCT"]}' > apps.json   # the apps the server exposes
export KONFAI_API_TOKEN="my-secret-token"
konfai-apps-server --host 0.0.0.0 --port 8000 --apps apps.json

konfai-apps infer VBoussot/ImpactSynth:CBCT -i input.mha -o ./Output \
  --host my.server.org --port 8000 --token "$KONFAI_API_TOKEN"
```

The server requires a token unless started with `--auth off`. The HTTP API is in
{doc}`../reference/app-server-api`.

## From 3D Slicer

[SlicerKonfAI](https://github.com/vboussot/SlicerKonfAI) runs apps from 3D Slicer, locally or on a server,
and loads the results into the scene. Test an app with `konfai-apps infer` first, then use the same name in
Slicer.

<figure class="kf-visual kf-visual--app">
  <a class="kf-visual-frame" href="../_static/slicer/inference.webp" aria-label="Open the SlicerKonfAI inference screenshot at full resolution">
    <img src="../_static/slicer/inference.webp" alt="Official SlicerKonfAI inference interface showing a TotalSegmentator MRI App and the returned multi-organ segmentation." width="1578" height="852" loading="lazy" decoding="async">
  </a>
  <figcaption>
    <span class="kf-visual-copy">
      <strong>Inference stays inside the clinical imaging workspace.</strong>
      <span class="kf-visual-meta">TotalSegmentator MRI · App selection, sampling controls, live logs, and returned segmentation</span>
    </span>
    <a class="kf-visual-inspect" href="../_static/slicer/inference.webp">Inspect 1578 × 852 <span aria-hidden="true">↗</span></a>
  </figcaption>
</figure>

<figure class="kf-visual kf-visual--app">
  <a class="kf-visual-frame" href="../_static/slicer/uncertainty.png" aria-label="Open the SlicerKonfAI uncertainty screenshot at full resolution">
    <img src="../_static/slicer/uncertainty.png" alt="Official SlicerKonfAI reference-free uncertainty interface with uncertainty map and summary metric." width="1676" height="852" loading="lazy" decoding="async">
  </a>
  <figcaption>
    <span class="kf-visual-copy">
      <strong>Reference-free uncertainty is part of the same App.</strong>
      <span class="kf-visual-meta">MRSegmentator · ensemble sampling · uncertainty map and summary metric returned to Slicer</span>
    </span>
    <a class="kf-visual-inspect" href="../_static/slicer/uncertainty.png">Inspect 1676 × 852 <span aria-hidden="true">↗</span></a>
  </figcaption>
</figure>

<figure class="kf-visual kf-visual--app">
  <a class="kf-visual-frame" href="../_static/slicer/evaluation.png" aria-label="Open the SlicerKonfAI evaluation screenshot at full resolution">
    <img src="../_static/slicer/evaluation.png" alt="Official SlicerKonfAI reference-based evaluation interface with MAE, PSNR, SSIM, Dice, and error maps." width="1676" height="852" loading="lazy" decoding="async">
  </a>
  <figcaption>
    <span class="kf-visual-copy">
      <strong>Evaluation returns both numbers and inspectable error maps.</strong>
      <span class="kf-visual-meta">ImpactSynth shown case · MAE, PSNR, SSIM, Dice, image outputs, and per-case logs</span>
    </span>
    <a class="kf-visual-inspect" href="../_static/slicer/evaluation.png">Inspect 1676 × 852 <span aria-hidden="true">↗</span></a>
  </figcaption>
</figure>

Screenshots vendored from
[`vboussot/SlicerKonfAI`](https://github.com/vboussot/SlicerKonfAI) at commit
`4508683`, under that repository's Apache-2.0 license.

## Packaging your own workflow

When a workflow is ready, `bundle` packages it as an app. An app is a folder with an `app.json`:

```json
{
  "display_name": "My segmentation model",
  "description": "Segments the target anatomy from CT.",
  "short_description": "CT segmentation",
  "tta": 0,
  "mc_dropout": 0
}
```

```bash
konfai-apps bundle CT_SEG \
  --out dist \
  --app-json app.json \
  --config Prediction.yml Evaluation.yml \
  --checkpoint Checkpoints/SEG_BASELINE/[0-9]*.pt \
  --model-py Model.py
```

This writes `dist/CT_SEG/` with `app.json`, the configs, the checkpoints and your Python code. The model
list comes from the checkpoint names, and a `requirements.txt` is drafted from `Model.py`'s imports: check
it. Leave out `--model-py` if there is no custom code; add `Uncertainty.yml` to offer uncertainty.

Check the images the app produces, not only the exit code:

```bash
konfai-apps infer ./dist/CT_SEG -i input.mha -o ./Output --gpu 0
konfai-apps eval ./dist/CT_SEG -i ./Output/<prediction>.mha --gt reference.mha
```

Then upload `CT_SEG/` to a Hugging Face model repository and use it as `owner/repository:CT_SEG`.

## The ecosystem around an app

KonfAI is the core; the other packages and tools build on it. The map shows what is released and what is
external or experimental.

```{raw} html
<figure class="kf-ecosystem-map" aria-labelledby="kf-ecosystem-map-caption">
  <div class="kf-ecosystem-map__canvas">
    <div class="kf-ecosystem-map__topline">
      <span>KonfAI system map</span>
      <span class="kf-ecosystem-map__legend"><i></i> released path <i class="is-dashed"></i> experimental edge</span>
    </div>
    <p class="kf-sr-only">Resolved YAML and medical data enter the KonfAI core. KonfAI Apps and KonfAI MCP both build on the core; MCP also operates Apps. Hugging Face artifacts feed Apps and the task-specific command-line tools. Slicer clients operate Apps and ImpactReg, while ONNX to konfai-rs remains experimental.</p>
    <svg class="kf-ecosystem-map__routes" viewBox="0 0 1100 992" preserveAspectRatio="none" aria-hidden="true">
      <path class="route route--core" d="M290 148 C330 148 310 224 365 224" />
      <path class="route route--core" d="M550 418 V432 C550 446 292 434 292 448" />
      <path class="route route--core" d="M550 418 V432 C550 446 832 434 832 448" />
      <path class="route route--mcp" d="M478 515 H644" />
      <path class="route route--artifact" d="M930 193 V406 C930 438 462 420 462 448" />
      <path class="route route--artifact" d="M930 193 V606 C930 632 500 612 500 640" />
      <path class="route route--apps" d="M292 578 V620 C292 634 174 626 174 640" />
      <path class="route route--apps" d="M292 578 V620 C292 634 500 626 500 640" />
      <path class="route route--mcp" d="M832 626 V632 C832 638 925 632 925 640" />
      <path class="route route--external" d="M174 787 V800" />
      <path class="route route--external" d="M500 787 V800" />
      <path class="route route--experimental" d="M657 418 C690 540 925 650 925 800" />
      <g class="stations">
        <circle cx="365" cy="224" r="5" /><circle cx="292" cy="448" r="5" /><circle cx="832" cy="448" r="5" />
        <circle cx="462" cy="448" r="5" /><circle cx="500" cy="640" r="5" /><circle cx="174" cy="640" r="5" />
        <circle cx="925" cy="640" r="5" /><circle cx="174" cy="800" r="5" /><circle cx="500" cy="800" r="5" />
      </g>
    </svg>

    <div class="kf-ecosystem-map__node kf-ecosystem-map__node--input">
      <span class="kf-ecosystem-map__kind">Research contract</span>
      <strong>Resolved YAML + medical data</strong>
      <p>Data, transforms, models, losses, metrics, and workflow intent.</p>
    </div>

    <div class="kf-ecosystem-map__node kf-ecosystem-map__node--artifact">
      <span class="kf-ecosystem-map__kind">Published artifacts</span>
      <strong>Hugging Face Hub</strong>
      <p>App configs, checkpoints, metadata, and demo datasets.</p>
      <span class="kf-ecosystem-map__status">published</span>
    </div>

    <div class="kf-ecosystem-map__node kf-ecosystem-map__node--hub">
      <span class="kf-ecosystem-map__kind">Execution foundation · PyPI</span>
      <strong><code>konfai</code></strong>
      <p>One declarative, patch-native engine for medical-imaging workflows.</p>
      <div class="kf-ecosystem-map__commands" aria-label="Core workflows">
        <span>TRAIN</span><span>PREDICTION</span><span>EVALUATION</span>
      </div>
    </div>

    <div class="kf-ecosystem-map__node kf-ecosystem-map__node--apps">
      <span class="kf-ecosystem-map__kind">Application runtime · PyPI</span>
      <strong><code>konfai-apps</code></strong>
      <p>Resolve local or Hub Apps and run the same workflow through Python, CLI, REST, or SSE logs.</p>
    </div>

    <div class="kf-ecosystem-map__node kf-ecosystem-map__node--mcp">
      <span class="kf-ecosystem-map__kind">Agent runtime · PyPI</span>
      <strong><code>konfai-mcp</code></strong>
      <p>Operates core workflows and Apps through structured scientific tools.</p>
      <div class="kf-ecosystem-map__commands" aria-label="MCP transports">
        <span>STDIO</span><span>SSE</span><span>HTTP</span>
      </div>
    </div>

    <div class="kf-ecosystem-map__node kf-ecosystem-map__node--interfaces">
      <span class="kf-ecosystem-map__kind">App surfaces</span>
      <strong>Python · CLI · REST</strong>
      <p>Local and remote execution share one App contract.</p>
    </div>

    <div class="kf-ecosystem-map__node kf-ecosystem-map__node--tasks">
      <span class="kf-ecosystem-map__kind">Task CLIs · PyPI</span>
      <strong>Five ready-to-run entry points</strong>
      <p>ImpactSynth · ImpactSeg · ImpactReg · MRSegmentator · TotalSegmentator</p>
    </div>

    <div class="kf-ecosystem-map__node kf-ecosystem-map__node--agents">
      <span class="kf-ecosystem-map__kind">Agent clients</span>
      <strong>Scientific automation</strong>
      <p>Inspect, configure, validate, execute, compare, and resume.</p>
    </div>

    <div class="kf-ecosystem-map__node kf-ecosystem-map__node--slicer-apps">
      <span class="kf-ecosystem-map__kind">External clinical client</span>
      <strong>SlicerKonfAI</strong>
      <p>Runs KonfAI Apps from 3D Slicer.</p>
      <span class="kf-ecosystem-map__status">external</span>
    </div>

    <div class="kf-ecosystem-map__node kf-ecosystem-map__node--slicer-reg">
      <span class="kf-ecosystem-map__kind">External registration client</span>
      <strong>SlicerImpactReg</strong>
      <p>Drives the complete ImpactReg orchestrator.</p>
      <span class="kf-ecosystem-map__status">external</span>
    </div>

    <div class="kf-ecosystem-map__node kf-ecosystem-map__node--experimental">
      <span class="kf-ecosystem-map__kind">Portable edge</span>
      <strong>ONNX → <code>konfai-rs</code></strong>
      <p>Native and WebAssembly inference path.</p>
      <span class="kf-ecosystem-map__status">experimental</span>
    </div>
  </div>
  <figcaption id="kf-ecosystem-map-caption"><strong>One execution model, several operating surfaces.</strong> Solid routes show released dependencies and orchestration paths; the dotted route is experimental.</figcaption>
</figure>
```

### Released platform

| Piece | Status | What it is |
| --- | --- | --- |
| **`konfai`** | ✅ Shipped (PyPI) | The core: YAML workflows, streamed data, model graphs. |
| **`konfai-apps`** | ✅ Shipped (PyPI) | Runs and packages apps (this page). |
| **`konfai-mcp`** | ✅ Shipped (PyPI) | Lets an LLM agent run KonfAI ({doc}`mcp`). |
| **App CLIs** (`apps/`) | ✅ Shipped | `impact-synth-konfai`, `impact-seg-konfai`, `mrsegmentator-konfai`, `totalsegmentator-konfai`; the models download on first run. |
| **`impact-reg-konfai`** | 🟡 Shipped | The registration orchestrator (Elastix, ConvexAdam, FireANTs, with the IMPACT metric). |
| **Models and demo data** | ✅ Published | On Hugging Face: `VBoussot/konfai-demo`, one repository per app, and `impact-torchscript-models`. |
| **Challenge repositories** | ✅ External | MICCAI challenge entries built on KonfAI (SynthRAD 2025, TrackRAD 2025, PANTHER, CURVAS), listed in the README. |

### External clients and experimental edge

| Piece | Status | Notes |
| --- | --- | --- |
| **SlicerKonfAI** | ✅ External | The [3D Slicer](https://github.com/vboussot/SlicerKonfAI) client of `konfai-apps`. |
| **SlicerImpactReg** | ✅ External | The 3D Slicer client of `impact-reg-konfai`. |
| **ONNX export → `konfai-rs`** | 🧪 Experimental | ONNX export from Python ({doc}`python-api`) for a portable inference engine, which Studio's deployment pane runs in the browser. |

## Next steps

- {doc}`../reference/cli`: every flag of `konfai-apps` and `konfai-apps-server`.
- {doc}`python-api`: the `KonfAIApp` and `KonfAIAppClient` API.
- {doc}`../reference/app-server-api`: the server's HTTP contract.

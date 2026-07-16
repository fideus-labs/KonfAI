# Ecosystem

KonfAI is the core, but several packages and tools sit around it. This page maps
how they relate and — importantly — **what is shipped versus what is external
or experimental**, so you know what you can rely on today.

## The map

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

## Released platform

| Piece | Status | What it is |
| --- | --- | --- |
| **`konfai`** | ✅ Shipped (PyPI) | The core framework: config-by-reflection, lazy patch-based data, model graphs, and the three YAML workflows. This is what the rest of this documentation is about. |
| **`konfai-apps`** | ✅ Shipped (PyPI, own CI) | Packages a mature workflow as a reusable **app** — see {doc}`../usage/apps`. |
| **`konfai-mcp`** | ✅ Shipped (PyPI, own CI) | Operates KonfAI workflows and Apps through structured tools for dataset inspection, config authoring and validation, job execution, monitoring, metrics, and comparison — see {doc}`../usage/mcp`. |
| **App bundles** (`apps/`) | ✅ Shipped (thin wrappers) | Ready-to-use CLI shims: `impact-synth-konfai`, `impact-seg-konfai`, `mrsegmentator-konfai`, `totalsegmentator-konfai`. Config + weights live on Hugging Face and download on first run. |
| **`impact-reg-konfai`** | 🟡 Shipped, heaviest | A full multi-preset registration orchestrator (Elastix + the IMPACT semantic metric), **not** a thin wrapper. The most moving parts of the five. |
| **Demo data & models (HF)** | ✅ Published | `VBoussot/konfai-demo` (demo dataset), plus per-app model repos (`ImpactSynth`, `ImpactSeg`, `TotalSegmentator-KonfAI`, `MRSegmentator-KonfAI`, `ImpactReg`) and `impact-torchscript-models` (the SAM2.1 backbone behind `IMPACTSynth`). |
| **Challenge repos** | ✅ External | Top-ranking MICCAI-challenge projects built on KonfAI (SynthRAD 2025 T1/T2, TrackRAD 2025, Panther, CURVAS, CURVAS-PDACVI). Referenced from the README; not in this tree. |

## Ready-to-use apps

| App CLI | Task | Modality | Models |
| --- | --- | --- | --- |
| `impact-synth-konfai synthesize` | Synthetic CT (sCT) | MR → CT, CBCT → CT | `MR`, `CBCT` |
| `impact-seg-konfai segment` | Multimodal body segmentation (11 labels) | CBCT / MR / CT | `body` |
| `mrsegmentator-konfai segment` | Whole-body MRI segmentation | MRI | folds 1–5 |
| `totalsegmentator-konfai segment` | Whole-body CT/MRI segmentation | CT / MRI | `total`, `total_mr`, 3 mm variants |
| `impact-reg-konfai register` | Multimodal deformable registration | MR/CT, CBCT/CT | presets |

Each also exposes `eval` and (except TotalSegmentator) `uncertainty`; the thin
wrappers add `pipeline`. See {doc}`../usage/apps` for how to run them and
{doc}`../reference/cli` for the full flag reference.

## External clients and experimental edge

| Piece | Status | Notes |
| --- | --- | --- |
| **SlicerKonfAI** | ✅ External GUI | A [3D Slicer](https://github.com/vboussot/SlicerKonfAI) client of the `konfai-apps` CLI/server, covered by API, CLI, and JSON contract tests in the Apps package. |
| **SlicerImpactReg** | ✅ External GUI | A dedicated 3D Slicer client for the complete `impact-reg-konfai` registration orchestrator. |
| **ONNX export → `konfai-rs`** | 🧪 Experimental | `konfai/export.py` produces ONNX + a manifest for a planned portable (native/WASM) inference engine. Python-API-only, single static-shape head, feed-forward models only. See {doc}`../reference/python-api`. |

```{note}
**Trust model.** Resolving a `konfai-apps` app copies and imports its `.py`
files (arbitrary code) and pip-installs its `requirements.txt` by default
(opt out with `KONFAI_APPS_INSTALL_REQUIREMENTS=0`), so only resolve apps
from sources you trust — see {doc}`../reference/python-api`.
```

## See also

- {doc}`../usage/apps` — what an app is and how to run apps from the CLI
- {doc}`../reference/cli` — the full flag reference for the app CLIs

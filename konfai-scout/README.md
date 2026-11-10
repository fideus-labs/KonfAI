<p align="center">
  <img src="assets/konfai-scout-logo.png" width="520" alt="KonfAI Scout">
</p>

<p align="center"><b>An agent that looks at your segmentation, registration or synthesis the way you would, and tells you where it is wrong.</b></p>

---

A score says how good a result is. Scout says where and why: it inspects one case at a viewer, slice by slice, zooms
on what looks off, switches overlays to tell the image from the result, asks KonfAI Eval to measure the suspect
region, and writes a report of localised defects with their type, severity, uncertainty, the views that show them
and the measurements that support them.

- **It explores.** Coverage first (a montage per plane), then targeted views: slices, zooms, checkerboards, difference
  maps, displacement fields, Jacobians, deformed grids, landmarks. Every view carries its position in millimetres.
- **It measures with KonfAI Eval, not with a scorer of its own.** A region is cropped and scored by `konfai.evaluate`
  with KonfAI's criteria: Dice per label, MAE, PSNR, SSIM, TRE where a reference exists; a field's folding, Jacobian
  and displacement statistics without one. Reading, warping and geometry are KonfAI's too: Scout adds the viewer,
  the agent and the report, nothing that KonfAI already does.
- **It keeps its statements apart, and enforces it.** A *measured property*, a *suspected error* (seen, not proven)
  and a *confirmed error* (a measured value past its criterion's threshold: Dice below 0.8, TRE above 3 mm, a
  folded field...) are distinct, and without a truth nothing is confirmed from appearance. `note` refuses an error
  that no cited view shows, a figure that no cited measurement holds, a confirmation resting on measurements taken
  elsewhere. Scout measures every flagged box itself with KonfAI Eval, records it on the observation, and
  downgrades a confirmation that no measured value supports. "Insufficient evidence" is a legitimate conclusion. On
  request, a fresh session of the model re-reads every error blind, from crops of its box and the claim alone, and
  its answer stands in the report next to the first reader's.
- **It judges blind.** The model is told neither the case's name, nor its notes (the method, the dataset), nor its
  prior scores: it judges the images and its own measurements (`--no-blind` hands them over, for a study).
- **It leaves a record.** A versioned JSON report with versions, configuration, prompts, every tool call, every view,
  every measurement, cost and duration, and a Markdown rendition that opens with the verdict and the findings at a
  glance, then each observation with its images. Two modes, fixed views and active exploration, for a study of what
  exploration buys.
- **It works with Claude, GPT, or a local vision model** through an OpenAI-compatible server, and sends images to a
  cloud API only when told to.

## Install

```bash
pip install konfai-scout[anthropic]        # Claude through the Anthropic API
pip install konfai-scout[openai]           # GPT, or any OpenAI-compatible server (vLLM, Ollama, LM Studio)
pip install konfai-scout[claude-code]      # Claude under your Claude Code login (the Claude Agent SDK), no API key
pip install konfai-scout[codex]            # GPT under your ChatGPT login (the Codex CLI), no API key
```

Scout is a package of the [KonfAI](https://github.com/fideus-labs/KonfAI) family (`konfai-scout/` in its repository),
pinned to the `konfai` release it ships with: every volume is read through KonfAI's dataset backends (image files,
OME-Zarr stores, DICOM series, ITK transform files), a registration is warped by KonfAI's own `Resample`, and every
measurement is KonfAI Eval's. From a checkout: `pip install -e ./konfai-scout[dev]` beside `pip install -e .`.

## Inspect a case

```bash
konfai-scout inspect --task segmentation --name case01 \
    --input image=ct.nii.gz --input prediction=seg.nii.gz --input truth=gt.nii.gz \
    --labels 1=liver,2=spleen --evaluation Evaluations/run/Metric_TRAIN.json \
    --out reports/case01 --provider anthropic --allow-cloud-images

konfai-scout inspect --task registration --name pair07 \
    --input fixed=fixed.nii.gz --input moving=moving.nii.gz --input field=dvf.nii.gz \
    --input fixed_labels=fixed_seg.nii.gz --input moving_labels=moving_seg.nii.gz \
    --out reports/pair07 --provider openai --base-url http://localhost:11434/v1 --model llava:34b

konfai-scout inspect --case case.json --out reports/case --provider claude-code --allow-cloud-images  # Claude Code login
konfai-scout inspect --case case.json --out reports/case --provider codex --allow-cloud-images        # Codex (ChatGPT) login
konfai-scout inspect --case case.json --out reports/case --mode fixed --provider mock   # no API at all
```

Roles: segmentation `image`, `prediction`, `truth`, `mask`; registration `fixed`, `moving`, `field` or `transform`,
`fixed_labels`, `moving_labels`, `warped_labels`, `fixed_landmarks`, `moving_landmarks`, `fixed_mask`; synthesis
`source`, `synthetic`, `target`, `mask`. A field is a displacement image in millimetres on the fixed grid (ITK's
reading: the fixed point `x` matches `x + u(x)` in the moving image); a transform is an ITK transform file.

Then read `reports/case01/report.md`, or in Python:

```python
from konfai_scout import InspectionCase, ScoutConfig, inspect

case = InspectionCase("segmentation", "case01", {"image": "ct.nii.gz", "prediction": "seg.nii.gz", "truth": "gt.nii.gz"},
                      labels={1: "liver", 2: "spleen"})
report = inspect(case, ScoutConfig(provider="anthropic", allow_cloud_images=True), "reports/case01")
for o in report.observations:
    print(o.kind, o.defect, o.severity, o.box_mm, o.views, o.measurements)
```

## What the model can do

| Tool | What it does |
|---|---|
| `view` | one slice: a plane, a position in mm or an index, a mode, a window, a zoom box, one label |
| `montage` | equally spaced slices of a plane in one image |
| `measure` | KonfAI Eval on a box: the case's groups cropped and scored with KonfAI's criteria, named as KonfAI names them or by a result key; SSIM is left out of a box thinner than its 7-voxel window |
| `region_stats` | labels, volumes, pieces, intensities (inside the case's mask when it has one), displacement lengths, landmarks in a box: no reference needed; a measurement of kind `scout`, citable |
| `note` | one observation with its kind, defect, severity, confidence, box and the evidence it cites; refused when the evidence does not show what it claims (`docs/report.md`) |
| `conclude` | the verdict |

Modes: segmentation `image`, `prediction`, `truth`, `both`, `disagreement`; registration `fixed`, `moving`, `warped`,
`before`, `checkerboard`, `overlay`, `difference`, `field_magnitude`, `field_arrows`, `grid`, `jacobian`, `labels`,
`landmarks`; synthesis `synthetic`, `target`, `source`, `difference`, `side_by_side`, `checkerboard`. A view's window
is the image's own 1st to 99.9th percentile unless the model asks for `[low, high]` or a CT preset (in HU, shifted by
1024 on a CT stored as HU + 1024).

## Budget, privacy, cost

`--depth` sets how far an inspection looks in one word; any flag below overrides what it sets:

| Depth | Tool calls | Effort | Second reading | For |
|---|---|---|---|---|
| `quick` | 10 | low | no | a cheap first pass over many cases |
| `standard` (default) | 25 | the model's | no | one case |
| `deep` | 40 | high | yes | a case that matters, or a doubt to settle |

The survey of where to look, the coverage montages, the rules on what a report holds and Scout's own check of each
flagged box are the same at every depth.

`--max-tool-calls`, `--max-images`, `--max-seconds`, `--max-usd` bound an inspection; when a limit is reached the
model gets one last turn to note and conclude, else Scout concludes "insufficient evidence" itself and says why. The
call budget counts the calls that explore (`view`, `montage`, `measure`, `region_stats`): a `note` or `conclude` is
never refused. With `claude-code` and `codex`, whose CLI runs the whole loop in one request, the calls and the time
are counted inside the served tools, and `claude-code`'s cost message by message; `codex` reports its cost only once
its loop has run, so the report says `max_usd` was not enforced (as it does for a model whose price is unknown).
`--second-reading` adds, after the inspection, one short session per error observation for the blind re-read.
Images of the case leave the machine only under `--allow-cloud-images`; a loopback `--base-url` is local. The cost
is computed from the published prices of the Claude models and reported as unknown otherwise (`ScoutConfig.prices`
adds a model).

The `claude-code` provider runs the Claude Agent SDK under your Claude Code login, no API key: the SDK drives the
tool loop, Scout's tools are served to it as an in-process MCP server (images included) and no built-in tool is
enabled. It is a cloud provider (`--allow-cloud-images`), and the CLI keeps the session, images included, under
`~/.claude/projects`. The report records the same calls, views, measurements and budget as with the API providers.

The `codex` provider does the same with the Codex CLI under your ChatGPT login: `codex exec` runs the loop, Scout's
tools are served to it over a loopback MCP server, the case images are attached to the prompt; your own Codex
configuration (MCP servers, plugins, shell) is not loaded. The CLI keeps the session under `~/.codex/sessions`.

## Examples and benchmark

```bash
python examples/inspect_synthetic.py                 # synthetic cases with planted defects, the free heuristic model
konfai-scout compare reports/*                       # one row per report
```

`examples/benchmark.py` measures Scout: it plants one defect at a time in real images you point it to (a manifest of
sources: an image and its truth, a CT, an MR and its CT), keeps a clean control of each, inspects every case at each
depth, and counts what was found, missed and invented, the verdicts, the agreement between repeats, the cost and the
time:

```bash
python examples/benchmark.py build manifest.json --out bench --phantoms
python examples/benchmark.py run bench --depth quick standard deep --repeats 2 --provider claude-code --allow-cloud-images
python examples/benchmark.py score bench             # bench/scores.md and scores.json
```

`--mode fixed` among the run's flags gives the ablation of fixed views against active exploration.

`docs/report.md` is the schema, `docs/integrations.md` how KonfAI Eval, konfai-mcp, KonfAI Studio and FidOpt use
Scout.

## Export at a confidentiality level

```bash
konfai-scout export reports/case01 --level regions --out case01.regions.json
```

`full` is the report as is; `regions` the verdict and the observations with their boxes, defects, severities and
measured values, without the case's name, paths, images, descriptions or transcript; `counts` the verdict and the
observations counted by kind, defect and severity. In Python: `konfai_scout.export(Report.load(dir), "regions")`.
`docs/report.md` says what each level holds.

## Development

From the KonfAI checkout, in its `dev` environment:

```bash
pixi run --environment dev python -m pip install -e "./konfai-scout[dev]"
pixi run --environment dev python -m pytest konfai-scout/tests      # no API key: fake clients and a mock model
pixi run --environment dev ruff check konfai-scout
```

Apache-2.0. Copyright (c) 2026 Fideus Labs LLC.

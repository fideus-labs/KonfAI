# KonfAI Scout: agent guide

Read before changing code. `README.md` is the user's side; `docs/` holds the report schema, the study protocol and
the integrations.

## What Scout is

An agent that inspects one result of a segmentation, a registration or a synthesis the way a person would at a
viewer: it looks at slices, zooms, switches overlays, asks for measurements, and writes a report of localised
defects with evidence. It is not a scorer: KonfAI Eval scores (`konfai.evaluate`), Scout reads those scores and
asks KonfAI Eval for more on a region. FidOpt may call Scout; Scout never calls FidOpt. Scout works without FidOpt,
FidLab, konfai-mcp and Studio.

## Layout

| Module | Role |
|---|---|
| `case.py` | `InspectionCase` (task, files by role, labels, prior evaluation) and `Volume` (lazy reads, geometry) |
| `geometry.py` | `Grid`, `WorldBox`, planes: world millimetres (LPS), nothing in pixels |
| `warp.py` | what a registration needs derived once (the warped moving data, the carried landmarks), by SimpleITK |
| `views.py` | the renderer: modes per task, radiological convention, the pixel-to-world affine of every view |
| `measure.py` | descriptive statistics of a region (no reference, no metric) and the Jacobian plane for display |
| `evaluation.py` | the KonfAI Eval bridge: prior reports, region measurements by `konfai.evaluate` on a crop |
| `tools.py` | the tools the model drives, their schemas, the epistemic checks on `note` and `conclude` |
| `prompts.py` | the system prompt and the case card, versioned (`PROMPT_VERSION`) |
| `agent.py` | the loop: coverage, exploration under a budget, the report |
| `report.py` | the versioned JSON record, its Markdown rendition, `compare` |
| `providers/` | `anthropic`, `openai` (and any OpenAI-compatible local server), `mock` (scripted, heuristic) |
| `synthetic.py` | cases with planted defects, for tests, examples and the integrations' tests |
| `cli.py` | `konfai-scout inspect | render | compare` |

## Invariants

- Every location handed to or from the model is in millimetres, LPS. A view records `px_to_mm`; a test checks the
  affine lands on the drawn voxels, on an oblique grid too.
- Three kinds of statement (`measured_property`, `suspected_error`, `confirmed_error`): `note` downgrades a
  confirmation that cites no measurement with a reference (or an error by definition). Without a truth, no
  reference metric is ever computed or reported.
- Images leave the machine only under `allow_cloud_images`; the check runs before any client exists
  (`providers.check_privacy`). A loopback `base_url` or `local_provider` is local.
- Metrics come from KonfAI Eval, by KonfAI's criterion names, on a crop of the case; Scout computes no Dice, MAE or
  Jacobian statistic of its own. The Jacobian determinant in `measure.py` is for the `jacobian` view only.
- Volumes are read by region (`Volume.region`); a whole volume is read only by a warp (`warp.py`) and by the label
  summary of a small map.
- The report is the record: versions, config, prompts, every tool call, every view (sent or not), every
  measurement with its request, usage and cost, budget. `schema_version` changes when a field changes meaning.
- No API key in CI. Providers are tested against fake clients; the loop with `providers.mock`.

## Running

```bash
uv venv && uv pip install -e ".[dev]"            # or pip install -e ".[dev]"
pytest -q                                         # ~10 s, the KonfAI Eval tests skip without konfai
pip install "konfai[itk] @ git+https://github.com/fideus-labs/KonfAI@<branch>" && pytest -q   # with the bridge
ruff check konfai_scout tests examples && ruff format --check konfai_scout tests examples
python examples/inspect_synthetic.py              # the heuristic model, no API
```

## Conventions

Line length 120 (Ruff), SPDX header on every source file, Conventional Commits, no AI-agent branding and no AI
co-author trailers in commits (the convention of the KonfAI family). English in code, docs and prompts.

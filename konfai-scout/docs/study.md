# Fixed views against active exploration

Scout records enough to compare two ways of inspecting the same case with the same model:

- `fixed`: the model sees a fixed set of views (one montage per plane in the task's default mode, and the central
  slice of each plane in the secondary modes), can measure regions and ask for statistics, cannot ask for a view.
- `active`: the model starts from the same montages and explores: slices in any plane, zooms, modes, measurements.

Everything else is identical: the system prompt (only the procedure paragraph differs, and it is recorded), the
tools `measure`, `region_stats`, `note` and `conclude`, the budget, the report.

## What is recorded for the study

Per inspection (`report.json`): the prompt and its version, the configuration, the provider and model, every tool
call with its arguments and duration, every view with where it sits in the world and whether it was sent, every
measurement with its request, the observations with their evidence, the conclusion, tokens, cost, duration, the
budget and what exhausted it. `transcript.json` keeps the conversation.

## Running it

```bash
python examples/study_fixed_vs_active.py --provider anthropic --allow-cloud-images --repeats 3
konfai-scout compare examples/out/study/*/*                    # one row per inspection
konfai-scout compare examples/out/study/*/* --json             # the same for a notebook
```

`compare` lists per inspection: mode, model, verdict, confidence, tool calls, views, images sent, measurements,
observations by kind, tokens, cost, duration, whether the budget was exhausted. The planted defects of a synthetic
case are in its `defects.json`, in millimetres, so a notebook can match observations to them by box overlap.

## Keeping runs comparable

- Pin the model id and the prompt version; a report names both.
- Give both modes the same budget; the fixed mode spends fewer calls by construction, which is part of the result.
- Keep `allow_cloud_images` explicit per run; the report records the provider as remote or local.
- Keep `blind` (the default) unless knowing the method, the notes and the prior scores is what is studied: a model
  told the method's name or a poor Dice looks for what it was told.
- Repeat: a model is not deterministic. `--repeats` runs each mode several times under distinct directories.

## Negative controls and planted defects

Whether the model invents defects is measured, not assumed. `synthetic.py` builds each case in two states: clean
(`clean=True`: the prediction is the truth, the field is the exact one, the synthesis is the target under fresh
noise) and planted (its `defects.json`, in millimetres). On a clean case every error observation is a false
positive by construction; on a planted case an error observation finds a defect when its box overlaps the planted
box (or names a defect planted without a box, like a blur). The model is told nothing of which is which.

```bash
python examples/study_controls.py --provider claude-code --allow-cloud-images --repeats 3 --second-reading
konfai-scout compare examples/out/controls/*/*          # planted, found, false_positives, not_visible per run
```

`match_defects` in `report.py` does the matching; `compare` reads the `defects.json` beside a case's inputs by
itself. Real data makes a stronger control: the same scan as fixed and moving under a zero field, the truth given
as the prediction, with notes that withhold what the pair is. Report the invented errors per inspection, the planted
defects found, and, with `--second-reading`, how many suspicions the blind reader did not reproduce.

## Calibration on real registrations

`examples/calibrate_real.py` plants defects in real development fields and keeps the clean copies as controls:

```bash
python examples/calibrate_real.py --cases cases.json --out calib --provider claude-code --allow-cloud-images \
    --repeats 2 --defects-per-case 1 --max-tool-calls 25 --jobs 4
```

`cases.json` lists the cases you point it to (`[{"name", "fixed", "moving", "field", "fixed_labels"?,
"moving_labels"?}]`; a field is a displacement image on the fixed grid or an ITK transform file), never a test case.
Each case is inspected clean (a real field, so a non-trivial control: what Scout reports there is what a real
registration shows, or what Scout invents) and with one defect per copy: a local fold, a local shift, a global shift.
It writes `calibration.json` (every inspection, the planted boxes, the per-defect rates) and `calibration.md`:
sensitivity (planted defects found) and specificity (no match on the clean copies) per defect type, and test-retest
agreement between the two inspections of each copy. `docs/calibration/` holds a run on five L2R HippocampusMR
development pairs registered by ConvexAdam.

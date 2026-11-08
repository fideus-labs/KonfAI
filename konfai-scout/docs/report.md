# The report

An inspection leaves one directory:

```
<out>/
  report.json        the record, schema_version "1.1"
  report.md          the same for a person: verdict, observations with their evidence, measurements, views, calls
  case.json          the case as inspected (task, roles, labels, prior evaluation, notes)
  transcript.json    the conversation with the model, images replaced by their byte count
  evidence/vNNN.png  every view rendered, sent to the model or not
  work/              what was derived (a registration's warped data) and every KonfAI Eval workspace (mNNN/)
```

## report.json

| Field | What it holds |
|---|---|
| `schema_version` | `"1.1"`; it changes when a field changes meaning (1.1: a confirmation rests on a value past a threshold, an error's measurements end with Scout's check, `budget.spent.tool_calls` counts the calls that explore, `budget.not_enforced`) |
| `scout_version`, `versions` | Scout, Python, numpy, SimpleITK, Pillow, scipy, konfai when installed, the provider SDK, the prompt version |
| `created_at`, `duration_s` | when the inspection started (UTC) and how long it took |
| `case` | the case with every input's path, SHA-256 and geometry, `truth_available`, the reference grid |
| `config` | the `ScoutConfig` in full: provider, model, mode, budget, privacy flags, `blind`, criteria, thresholds, prices |
| `provider` | name, model, whether it is remote, the SDK version |
| `prompts` | the system prompt as sent, and its version |
| `prior_evaluation` | the whole-case values read from the KonfAI Eval reports given (`{metric key: value}`); shown to the model only with `blind` off |
| `tool_calls` | every call: number, tool, arguments, start time, duration, ok, summary, the views, measurements or observation it produced, the error |
| `views` | every view: id, kind (slice, montage), plane, array axis, index, position (mm), the box it covers (mm), mode, window, size, `px_to_mm` (`world = origin + col * u + row * v`), path, SHA-256, whether it was sent, extras (legend, scales, montage tiles) |
| `measurements` | every measurement: id, kind (`konfai_eval` for `measure`, `scout` for the descriptive statistics of `region_stats`), `reference` (measured against a truth), the box requested and the box measured (the voxels cropped), the values by KonfAI's metric keys (`PRED:TRUTH:Dice:1`) or by dotted statistic names, the request (criteria, labels, dataset; `check_of` names the observation when Scout took it to check a flagged box), the workspace, duration |
| `observations` | every note that passed the rules: id, kind, defect, severity, confidence, description, box (mm), plane, the view ids and measurement ids cited (and the id of Scout's own check of the box, last), notes (a downgrade and why, figures read from a view rather than measured), `second_reading` when one was asked (`answer`: visible, not_visible, unsure or unparsed; the reader's sentence; the crops it saw) |
| `conclusion` | verdict, summary, confidence, `by` (`model`, or `scout` when Scout had to conclude itself), reason |
| `usage` | requests, input/output tokens, cache reads and writes, images sent, cost in USD when the price is known |
| `budget` | the limits, what was spent (`tool_calls` counts the calls that explore, not `note` or `conclude`), what exhausted the budget if anything, `not_enforced`: why `max_usd` could not stop this inspection (a price unknown, a CLI that reports its cost only at the end), or `null` |
| `error` | a provider failure or a refusal, when one ended the inspection |
| `second_reading` | when asked: the reader's model, how many error observations were re-read and the counts by answer, the path of its transcripts |

## Statements

Three kinds, never confused:

- `measured_property`: a number measured on the case, reported as is.
- `suspected_error`: something that looks wrong in a view, not established by a measurement against a reference.
- `confirmed_error`: a measured value of the region past its criterion's threshold, in the bad direction. A note
  that claims a confirmation without one is recorded as `suspected_error` with the reason in `notes`.

Thresholds are keyed by KonfAI's result names (`DEFAULT_THRESHOLDS` in `evaluation.py`, extended or overridden by
`ScoutConfig.thresholds`):

| Criterion | An error when |
|---|---|
| `Dice`, `SurfaceDice` | below 0.8 |
| `HD95` | above 5 mm |
| `ASSD` | above 2 mm |
| `TRE` | above 3 mm |
| `JacobianFolding` | above 0 (any folded voxel) |
| `JacobianMin` | below 0 |

A criterion whose scale is the data's (MAE, PSNR, SSIM, displacement lengths, bending energy) has no default: it
confirms nothing until a site sets one (`{"MAE": {"above": 100}}` for a CT in HU). A non-finite statistic of a
field is an error by definition; an empty value (`None`, a box outside the truth or the mask) proves nothing. A
perfect Dice cited against a "confirmed" error leaves a suspicion.

Without a truth, no reference metric exists: nothing is confirmed from appearance, and the report says so.

### The rules `note` enforces

A model can be wrong in two ways: it can see something that is not there, and it can write something it did not see.
The second is caught mechanically; `note` refuses the call and tells the model why, so it can look again or cite
properly:

- an error (suspected or confirmed) cites at least one view, and when it has a box, one of the cited views shows
  that box: a slice or a zoom whose extent intersects it, or a montage with a tile through it. A defect "seen" where
  no cited image looks is not recorded;
- a `measured_property` cites the measurement (`measure` or `region_stats`) that holds its figures, and every figure
  it states with a unit or a metric name is found in a cited measurement, to the precision written. A number that
  no tool returned does not enter the report as a measurement;
- a confirmation rests on a measurement of the region it describes: a `confirmed_error` whose cited measurements
  were all taken outside its box is refused. A measurement taken elsewhere may be cited beside one of the region,
  as context; a note says which.

Softer rules check, downgrade or mark instead of refusing:

- Scout checks every error (suspected or confirmed) that has a box: it measures the box itself with KonfAI Eval
  (every criterion the case allows: a field's folding and Jacobian, Dice or TRE against the references, MAE against
  a synthesis target), cites that measurement on the observation, and notes which of its values are past a threshold
  (`Scout's check m007 of the box: WARPED_LABELS:FIXED_LABELS:Dice:1 = 0.42 < 0.8`), or that none is, or why nothing
  could be measured;
- a confirmation that neither its cited measurements of the region nor Scout's check supports becomes a
  `suspected_error`;
- a figure in an error's description that no cited measurement holds stays, with the note that it was read from a
  view.

The Markdown rendition shows every note in italics under its observation.

### The blind second reading

With `second_reading` (CLI `--second-reading`), a fresh session of the same model re-reads every error observation
that has a box: it receives crops of that box (twice its size, at least a dozen voxels) in two or three modes, the
claim as written, and nothing else: not the method, not the history, not the measurements. It answers `VISIBLE`,
`NOT_VISIBLE` or `UNSURE` with one sentence, recorded on the observation as `second_reading` and summed up in the
report's `second_reading` field. Nothing is removed or re-judged: a confirmation stands on its measurement, and a
reader can see which suspicions a second pair of eyes did not reproduce. Its transcripts are in
`second_reading.json`, its crops among the views.

Defect types per task:

| Task | Defects |
|---|---|
| segmentation | missing_structure, spurious_structure, under_segmentation, over_segmentation, leak, boundary_error, fragmentation, holes, label_confusion, other |
| registration | residual_misalignment, folding, implausible_deformation, over_smoothing, boundary_artifact, missing_coverage, intensity_artifact, other |
| synthesis | intensity_bias, hallucinated_structure, missing_structure, blur, artifact, geometry_error, other |

Severities: minor, moderate, major, critical. Verdicts: acceptable, defective, insufficient_evidence. A `defective`
verdict with no confirmed error carries a `reason` saying it rests on suspected errors.

## Blind by default

The model is not told the case's name, its notes (the method, the dataset) or its prior evaluation: the case card
says they are withheld, and the model judges the images and what it measures. `ScoutConfig.blind = False` (CLI
`--no-blind`) hands them over, for a study of what that knowledge changes. The report keeps them all for its reader.

## Export at a confidentiality level

`konfai-scout export REPORT_DIR --level full|regions|counts --out FILE` (Python: `konfai_scout.export(report,
level)`) writes what may be shown of an inspection to a reader that must not identify the case, such as an agent
tuning a method on cases it must not see (FidOpt's test cases):

| Level | What it holds |
|---|---|
| `full` | `report.json` as is |
| `regions` | `level`, `task`, `verdict`, and per observation: id, kind, defect, severity, confidence, box (mm), plane, the id, box measured and values of each of its measurements, the second reader's answer. No case name, no path, no image or view, no description or note (free text), no transcript, no prior evaluation, no usage. |
| `counts` | `level`, `verdict`, and the observations counted by kind, defect and severity: `{"suspected_error": {"folding": {"major": 1}}}` |

A box is in the patient's physical frame: at `regions`, a reader learns where a defect is, not whose scan it is.
`counts` is the level for a reader that must not learn even that.

## Coordinates

Everything is in millimetres in the LPS frame of the image headers. A view's `px_to_mm` is the affine of its final
image (header included), so a pixel a reader points at is a physical point; a montage carries one per tile. Boxes
are `{"min": [x, y, z], "max": [x, y, z]}`.

## Reading it from code

```python
from konfai_scout import Report
report = Report.load("scout_report")          # the directory or report.json
report.summary()                              # the short form, no images: what integrations carry
for o in report.observations: print(o.kind, o.defect, o.box_mm, o.views, o.measurements)
```

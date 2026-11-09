# Integrations

Scout works alone: `konfai-scout inspect` with a case given by roles. The rest of the ecosystem calls it; Scout
calls only KonfAI Eval.

```
FidLab ──orchestrates──▶ FidOpt ──scores with──▶ KonfAI Eval ◀──measures regions with── Scout
                           │ optionally                                                   ▲
                           └──────────────── inspects a case with ────────────────────────┘
konfai-mcp ── exposes ── Scout (run_scout_inspection, read_scout_report, list_scout_reports)
           ── exposes ── a FidOpt adapter (fidopt_list_datasets, fidopt_describe_dataset, fidopt_list_methods,
                                            fidopt_table, fidopt_run)
konfai-studio ── drives konfai-mcp ── Scout panel: launch, follow the job, read the report, go to a region
```

## KonfAI Eval

`konfai_scout.evaluation`:

- `load_prior(reports, case)` reads a KonfAI `Metric_*.json` and keeps the case's values: the model starts from them.
- `RegionEvaluator.measure(box)` crops the case's groups to the box (SimpleITK region reads, geometry kept), writes
  them as a one-case dataset and calls `konfai.evaluate` with KonfAI's criteria by name. The criteria per kind are
  in `DEFAULT_CRITERIA` and can be replaced through `ScoutConfig.criteria` (the metric vocabulary of another KonfAI
  branch, a tolerance, a dynamic range).
- `DEFAULT_THRESHOLDS` says when a measured value is an error (Dice below 0.8, TRE above 3 mm, a folded field...);
  `ScoutConfig.thresholds` sets one for a criterion whose scale is the data's (MAE in HU). A confirmed error rests
  on such a value, and Scout measures every flagged box itself the same way.
- A registration is measured through its field: the moving labels and image are warped once by SimpleITK with ITK's
  reading of the field (`x` matches `x + u(x)`), the fixed landmarks are carried the same way, and KonfAI scores
  Dice, MAE and TRE on the crop; the field's own statistics (Jacobian folding, minimum and log-std, bending energy,
  displacement lengths) need no reference.

What Scout reuses of KonfAI beyond the criteria: `konfai.data.geometry` (grids, world boxes, the affine of a
voxel), `konfai.utils.dataset` (the staged layout konfai-apps uses, region reads through every backend, landmarks),
`konfai.transform` with `Resample` and `TransformPoints` for the warped moving data and the carried landmarks, and
`konfai.utils.ITK.field_jacobian_determinant` for the Jacobian view. The Python API accepts the `None` target
KonfAI's configs use for a measure of the output alone.

## konfai-mcp

Three Scout tools and five FidOpt tools, in `konfai_mcp/server_scout.py`:

- `run_scout_inspection(task, inputs, ...)`: a tracked job running `konfai-scout inspect` on a case JSON written
  under the session (`Scout/<name>/`); `allow_cloud_images` must be passed to use a cloud provider.
- `read_scout_report(name)`: the report's summary and observations (no image); `list_scout_reports()`.
- `fidopt_list_datasets`, `fidopt_describe_dataset`, `fidopt_list_methods`, `fidopt_table`: read-only, through
  FidOpt's own CLI and Python (`KONFAI_MCP_FIDOPT`, the `fidopt` executable; its sibling `python` runs the
  descriptor queries). `fidopt_run(method, dataset, cases, scout)`: `fidopt run` as a tracked job, allowed only for
  the operations listed in `KONFAI_MCP_FIDOPT_ALLOWED` (default: `check`); nothing of FidOpt is reimplemented.

## konfai-studio

`/api/scout/reports`, `/api/scout/report` and `/api/scout/evidence` in the BFF; a
*Scout* tab in the right panel lists the session's reports, their observations, and a *Go to* button that loads the
case's image in the viewer and puts the crosshair at the observation's centre (LPS to RAS for NiiVue). Progress is
the job stream Studio already shows.

## FidOpt

`fidopt run METHOD DATASET --scout PROVIDER[:MODEL] [--scout-allow-cloud]` and the same flags on `fidopt tune`. After
the rows are scored, while the method's outputs are still on disk, each case is inspected, and its report lands beside
the rows under `results/<machine>/<method>/<dataset>.scout/<case>/`, with its `counts` export (`counts.json`). The rows
and the published scores do not change; the reports are a complementary evaluation. A development case is handed its
truth and its scores; a test case only its images and the method's output, never its truth. Images leave the machine
only under `--scout-allow-cloud`, and `fidopt publish` never sends a report. What an agent may read of a case it must
not identify is an export, never the report itself:

```bash
konfai-scout export results/<machine>/<method>/<dataset>.scout/<case> --level counts --out <case>.counts.json
```

`--level regions` adds the observations' boxes, defects, severities and measured values; `counts` keeps the verdict
and the counts by kind, defect and severity only (`docs/report.md`).

## FidLab

FidLab runs no inspection itself: the reports are FidOpt's, made when `--scout` was asked. `fidlab.py scout EXPERIMENT`
reads, from the runner, the `report.md` of each label's development cases (only their counts for a blind experiment,
whose reports name the method) and, for the one submission to the hidden test, the `counts` exports of its test cases
alone, without their names. The sandbox reads text, never an image.

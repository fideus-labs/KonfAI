# Integrations

Scout works alone: `konfai-scout inspect` with a case given by roles. The rest of the ecosystem calls it; Scout
calls only KonfAI Eval.

```
FidLab ──orchestrates──▶ FidOpt ──scores with──▶ KonfAI Eval ◀──measures regions with── Scout
                           │ optionally                                                   ▲
                           └──────────────── inspects a case with ────────────────────────┘
konfai-mcp ── exposes ── Scout (run_scout_inspection, read_scout_report)
           ── exposes ── a FidOpt adapter (fidopt_catalog,
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
- `read_scout_report(name)`: the report's summary and observations (no image); `read_scout_report()`.
- `fidopt_catalog`, `fidopt_table`: read-only, through
  FidOpt's own CLI and Python (`KONFAI_MCP_FIDOPT`, the `fidopt` executable; its sibling `python` runs the
  descriptor queries). `fidopt_run(method, dataset, cases, scout)`: `fidopt run` as a tracked job, allowed only for
  the operations listed in `KONFAI_MCP_FIDOPT_ALLOWED` (default: `check`); nothing of FidOpt is reimplemented.

## konfai-studio

`/api/scout/reports`, `/api/scout/report` and `/api/scout/evidence` in the BFF; a
*Scout* tab in the right panel lists the session's reports, their observations, and a *Go to* button that loads the
case's image in the viewer and puts the crosshair at the observation's centre (LPS to RAS for NiiVue). Progress is
the job stream Studio already shows.

## FidOpt

`fidopt run METHOD DATASET --scout PROVIDER[:MODEL] [--scout-mode fixed|active] [--scout-allow-cloud]` and the same
flags on `fidopt tune`. After the rows are scored, while the method's outputs are still on disk, each case is staged
(the images as `.mha`, the field, the truth the dataset gives) and inspected; the reports land beside the rows under
`results/<machine>/<method>/<dataset>.scout/<case>/` with a `summary.json`. The rows and the published scores do
not change; the reports are a complementary evaluation. Images of a test case leave the machine only under
`--scout-allow-cloud`, and `fidopt publish` never sends a report.

## FidLab

A round may ask for Scout: `"scout": true` in `rounds/next.json` (or `plan.json`), which the runner turns into
`fidopt tune ... --scout $FIDLAB_SCOUT` when the host sets `FIDLAB_SCOUT` (a provider and model), with
`--scout-allow-cloud` only when the host also sets `FIDLAB_SCOUT_ALLOW_CLOUD=1`. The result of the round carries,
per label, the summaries of the reports (observations, verdicts, counts), never an image: the sandbox reads text.
Scout inspects development cases only, like everything the runner does.

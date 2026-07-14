# Recovering from failures

The tools emit `next_actions` for exactly this reason — follow them instead of guessing.
This file is the map from a symptom to the tool that diagnoses or fixes it.

## Golden rules

- **Never launch a run to "see if it works".** `run_train` / `run_prediction` /
  `run_evaluation` actually execute and write `Checkpoints/`, `Statistics/`,
  `Predictions/`, `Evaluations/` under the session workspace — they are destructive, not
  dry runs. Catch errors with `review_config_semantics` → `validate_config_semantics`
  first (both are side-effect-free on your authored config).
- **One active job per session.** Launching a second run while one is queued/running
  raises `Session '<name>' already has active job(s)`. Either `wait_for_job` on the
  active one or `cancel_job`, or work in a different session.
- **A server restart orphans running jobs.** The server does **not** kill the child on
  restart — it loses the in-memory process handle and relabels any previously-active job
  `status=error`, `recovered=true` ("original subprocess handle is unavailable"). Because
  jobs run as non-daemon processes, an orphaned training child may **keep running and holding
  the GPU**. Verify/kill it at the OS level if needed, then relaunch.

## Symptom → action

| Symptom | Diagnose / fix with |
|---|---|
| Config written but unsure it's valid | `review_config_semantics` → `validate_config_semantics` |
| `review_config_semantics` returns `blocking_issues` | Fix the YAML, re-write with `write_workflow_config`, review again. Do **not** proceed to validate. |
| `validate_config_semantics` returns `ok=false` | Read `error_type` / `error` / `traceback`; fix the offending object. Use `inspect_object_signature` on the classpath in the traceback to get its real parameters. |
| Unknown/ambiguous object name in YAML | `list_components` for that kind, then `inspect_object_signature` on the chosen classpath. |
| "How is X even configured?" | `describe_config_schema` for the workflow; `inspect_object_signature` for one object. |
| Optional dependency error (`itk`, `h5py`, `lpips`, …) | `check_external_dependency` to confirm what's installed and get the install hint. |
| Prediction can't find a checkpoint | Confirm training finished (`get_job_status` → `done`) and that a checkpoint exists in the session workspace. `run_prediction` does **not** search outside the session for a missing checkpoint. |
| Evaluation reports missing predictions | Run/confirm `run_prediction` first; the evaluator does **not** infer missing predictions. |
| Dataset mapping looks wrong (`IMG` vs `CT`) | `inspect_dataset` → `prepare_dataset_aliases`, then re-`design_config_strategy`. |
| Job stuck / need to stop it | `cancel_job` (SIGTERM, waits ~5s, then SIGKILL). |
| Job failed — what next? | The job payload's `next_actions` say `validate_config_semantics` then `retry:<kind>`. Validate first to catch the cause, fix, relaunch. |
| Long training seems to "hang" | It probably isn't — `wait_for_job` **with no `timeout_s`** waits until the job truly finishes. Use `read_live_metrics` / the `job://<id>/log` resource to watch progress. Only pass `timeout_s` when you deliberately want to bound the wait (it raises on expiry). |

## Job status lifecycle

`queued → running → done | error | killed`. Read it with `get_job_status` (reading a
status can itself transition a finished job). Status-driven next steps, straight from the
job payload:

- **queued / running** → `get_job_status`, `read_live_metrics`, read `job://<id>/log`, `cancel_job`
- **done** → `summarize_session`, then `leaderboard` to compare against other runs
- **error / killed** → `validate_config_semantics`, then retry the same `kind`

## Where the artifacts and state live

- Per-job: `job://<id>/status`, `job://<id>/log`, `job://<id>/manifest` (the manifest
  captures the config snapshot taken at launch — the reproducible record of that run).
- Per-session: `session://current/summary`, `session://current/log`,
  `session://current/metrics`, `session://current/config/<workflow>`.
- Config snapshots are copied into a per-job configs dir at launch, so a later
  config edit never rewrites the history of a completed job.

## Validation quirks worth knowing

- Validation runs CPU-only (`CUDA_VISIBLE_DEVICES=''`) and forces overwrite on a
  throwaway workspace, so it cannot clobber real artifacts.
- Validating a **prediction** config without supplying `models` writes a tiny placeholder
  weight file into the scratch workspace so the build can complete — this is expected and
  is not a real checkpoint.
- `KONFAI_MCP_VALIDATE_ROOT` sets the scratch validation workspace. If validation errors
  with a missing-root/KeyError, that env var (or the server's default) is not set.

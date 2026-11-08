# KonfAI MCP. Tool Reference

> GENERATED from the registry by `konfai-mcp/scripts/generate_tool_reference.py`: do not edit by hand.

50 tools, 4 prompts, 23 resources. The live equivalent is the `guide://tool-index` resource.

## Tools

### `app_sources`

Edit the workspace app catalogue that list_apps merges: action 'add' persists an app path or HuggingFace repo across sessions, 'remove' drops it. Does not validate the reference: describe_app does.

### `cancel_job`

Stop a running job: requests cancellation and waits briefly for a clean shutdown. Artifacts are kept.

### `check_component`

Pre-flight a component before wiring it into a config. Without kind: whether the classpath's top-level package is importable (version, licence, install hint), checked without importing it here. With kind: smoke-test the classpath on dummy tensors in an isolated subprocess (a transform's shape contract, a criterion's return and backward). The smoke test runs the code: only on code you or the user wrote.

### `check_config`

Check the session's workflow config. level 'review': schema and semantic checks only (blocking_issues, warnings; no runtime object built). 'instantiate' builds the KonfAI objects, 'setup' also the dataloader, 'train_step' runs one forward and backward on one batch (train only): each in an isolated subprocess, nothing launched. Next: run_train, run_prediction, run_evaluation, or fix the config.

### `compare_runs`

Two runs metric by metric on aligned cases: means, per-case deltas, a direction-aware winner (loss-like metrics count lower as better). Reads their Metric_<split>.json.

### `control_training`

Steer a RUNNING training without restarting it: lr and it_validation are applied at its next poll boundary and recorded in the run's config snapshot; validate_now=True runs a validation pass at the next iteration. POSIX, local jobs; a no-op when nothing runs.

### `delete_run`

Delete one run's outputs of a kind (train: its Statistics and Checkpoints; prediction, evaluation, uncertainty: its folder; transform: its plan and log only), jailed to the session. Never a config or the dataset.

### `describe_app`

One app's manifest: description, input and output modalities, inference, evaluation and uncertainty capabilities, checkpoints, terminology. parameters=True adds its tunable model parameters with their constraints (imports the app's code in an isolated subprocess: needs allow_untrusted_code=True). Next: run_app, fine_tune_app, fetch_app.

### `describe_config_schema`

Top-level schema of a workflow config (train, prediction, evaluation, transform), generated from KonfAI's own classes: each field's literal yaml_key, type, default, whether required, and the classpath to drill into with inspect_object_signature. Next: write_workflow_config.

### `describe_extension_points`

How to add a component of a kind (loss, metric, model, transform, augmentation, scheduler, pretrained model): the base class, the required methods, the YAML reference syntaxes (builtin, File:Class, package.module:Class) and the gotcha. Next: check_component, write_session_file.

### `describe_konfai_capabilities`

What KonfAI can do and which tool to reach for: the four workflows, the component kinds, the extension model, which actions are safe and which destroy data. Start here when orienting. Next: list_apps to reuse a model, inspect_dataset to train one.

### `describe_model_outputs`

The addressable module paths of the session config's networks: the exact keys outputs_criterions and outputs_dataset accept (terminal=true marks output heads). Built side-effect-free. Next: write_workflow_config, check_config.

### `design_config_strategy`

A config-writing plan from the task, the dataset root(s), the group roles and the modeling intent: dataset summary, config plan, customisation options, unresolved questions, compatible example templates. A group's role (input or target) cannot be read off its name: settle it here with the user. Writes nothing. Next: initialize_session, then write_workflow_config.

### `diff_run_configs`

Unified diff of the exact configs two jobs ran with, from their immutable launch-time snapshots.

### `export_run_record`

The reproducibility record of one run in one payload: the job manifest (command, devices, environment with package versions), the launch-time config snapshots, the resolved config, every split's metrics and a log tail.

### `fetch_app`

Bring a published app into the session. Without path: import it as a normal experiment (configs, code and checkpoints copied, requirements installed) for config surgery, custom losses or run_resume(weights_only=True); TRUST GATE allow_untrusted_code=True. With path: export it as a local editable bundle there, set_parameters baked in, no code import; refuses to replace a file unless overwrite=True. Local and HuggingFace apps only.

### `fidopt_catalog`

FidOpt's benchmark catalogue through its own code: the datasets (optionally one task) and the methods; with dataset=<name>, that dataset's descriptor (cases, truth, labels, licence) and the methods. Downloads nothing. Next: fidopt_table, fidopt_run.

### `fidopt_run`

Launch one FidOpt operation as a tracked job: check, run or tune a method on a dataset's cases (only the operations KONFAI_MCP_FIDOPT_ALLOWED permits); scout='PROVIDER[:MODEL]' adds Scout's inspection of the scored outputs. Next: get_job_status(wait=True), fidopt_table.

### `fidopt_table`

A FidOpt dataset's board (fidopt table --json): one line per method and machine, published scores and local rows.

### `fine_tune_app`

Fine-tune a published app's checkpoint on the user's dataset without authoring a config (epochs, it_validation, lr, batch_size as parameters; set_parameters for the model's tunables), as a tracked job that writes a runnable app bundle to output. TRUST GATE: allow_untrusted_code=True. Next: get_job_status(wait=True), then run_app on the bundle.

### `generate_folds`

Split a dataset into K cross-validation folds: one case-list file per fold in the session, plus the subset stanzas to paste into each config ('folds/fold_0.txt' keeps those cases, '~file' the others). Next: write per-fold configs with distinct train_name, then run_batch.

### `get_job_status`

The state of one job (job_id) or of every job of the session (no job_id). wait=True blocks until the job ends or timeout_s expires (timed_out=true then, not an error). The payload names next_actions and the job's log resource.

### `get_run_metrics`

Full evaluation metrics (per-case values and aggregates) of one run or app trial, from Evaluations/<run>/Metric_<split>.json. Does not rerun evaluation.

### `initialize_session`

Create or reset the current session workspace, optionally seeded from an example template (include_support_files=True copies its local .py). DESTRUCTIVE with overwrite=True: deletes Checkpoints/ and Predictions/ too; use session(action='create') for a fresh name instead. Next: write_workflow_config.

### `inspect_dataset`

Groups, entry hints, ambiguities and sampled statistics of one dataset root (include_stats=False for a fast structure scan, groups=[one] for focused statistics). browse=True returns instead a bounded file tree with candidate roots, for a path with nested cohorts or an unclear layout. Next: design_config_strategy.

### `inspect_object_signature`

Signature, defaults, doc and contract hints of any configurable classpath: built-in, local File:Class, external package.module:Class, or a declarative .yml model (its hyperparameters, loss-attachable leaves and how to adapt it). Parsed statically, never built. Next: write_workflow_config, check_config.

### `leaderboard`

Ranked metrics across the session's completed runs, one metric and split, direction-aware, with the available metrics and splits. Next: get_run_metrics, compare_runs.

### `list_apps`

Published KonfAI apps from the catalogue (shipped, workspace file, KONFAI_MCP_APP_CATALOG), HuggingFace repos expanded. Use first when the user wants a result from an existing model: an app can do any task, judge the fit from its description. Imports nothing. Next: describe_app, else design_config_strategy.

### `list_components`

Built-in KonfAI components of one kind (model, loss, metric, transform, augmentation, scheduler, ...): name, where it goes in the YAML, the classpath to inspect. Next: inspect_object_signature.

### `package_app_from_session`

Package a model trained in this session into a runnable app bundle (app.json, config, checkpoint, optional Model.py, requirements, support_files, ONNX export) that describe_app and run_app consume. Does not upload.

### `prepare_dataset_aliases`

Copy, symlink or move dataset files under new group names when the content is right but the filenames do not match the intended config. Reports the created paths and missing_by_case. Next: inspect_dataset.

### `preview_volume`

One slice of a volume as a PNG (any SimpleITK-readable file), windowed between the 1st and 99th percentile and downsampled, for a visual check of a case or a prediction. Reads only.

### `read_file`

Read a bounded text range of one file. A relative path reads the current session (configs, support code, config snapshots, evaluation JSON). An absolute path reads a dataset's small text file: CSV or TSV (with columns and rows), JSON or YAML sidecar, case list, .mhd/.nhdr header, or a Transform.h5 summarised (type, parameters or displacement statistics). template=<example name> reads a file shipped with that template. Binary files are refused. Never writes.

### `read_job_log`

A job's console or runtime log: the last max_lines, or the lines matching a regex, to find the traceback. Next: check_config then the matching run_* tool, or cancel_job.

### `read_live_metrics`

Parsed runtime metrics of a running or finished training (latest snapshot, recent entries, per-stage summaries) instead of raw log lines. Does not block.

### `read_scout_report`

A finished Scout inspection by name: verdict, observations (kind, defect, severity, confidence, box in mm, cited views and measurements), measurements, views, usage, budget; no image bytes. Without name: the session's inspections listed.

### `read_training_curves`

A run's training curves (scalars over iterations) from its TensorBoard event files, downsampled to max_points. Needs the tensorboard package.

### `run_app`

Run a published app as published on the user's data: action infer, evaluate (the app's own Evaluation.yml against gt), uncertainty, or pipeline (all three). A registration app runs through impact-reg-konfai: inputs [[fixed...], [moving...]] plus optional masks; outputs <output>/P000/Transform.h5 (fixed point to moving point, mm) and P000/Moved on the fixed grid; score it with run_registration_evaluate. set_parameters takes tunables by name or by full dotted config path. TRUST GATE: allow_untrusted_code=True required (imports the app's code, installs its requirements). Local and HuggingFace apps only. Next: get_job_status(wait=True).

### `run_batch`

Run several training configs sequentially server-side (a sweep, fold training), blocking until the batch ends, with per-run outcomes. Each config needs a distinct train_name. Next: leaderboard, compare_runs.

### `run_evaluation`

Launch an evaluation job from the session Evaluation.yml on existing predictions. Returns a job payload or a blocker. Next: get_job_status(wait=True), leaderboard.

### `run_prediction`

Launch a prediction job from the session Prediction.yml with a checkpoint of the session (models picks which). Returns a job payload or a blocker. Next: get_job_status(wait=True).

### `run_registration_evaluate`

Score a registration against what the pair has: labels (Dice), landmarks (TRE in mm, fixed points pushed through the transform), same-modality images (MAE), and a field's folded-voxel fraction. Pass the ORIGINAL moving data: the transform warps it. Run with and without transforms to report the gain. A tracked impact-reg-konfai job, no trust gate; Evaluation_summary.json holds the aggregates.

### `run_resume`

Resume a training from a checkpoint (model, optimizer, scheduler and counters restored), or warm-start a fine-tune with weights_only=True (weights only; prefer fine_tune_app for an app used as published). Resumes the newest checkpoint of the configured run and trains up to the live config's epochs. Returns a job payload or a blocker.

### `run_scout_inspection`

Have KonfAI Scout inspect one result visually (segmentation, registration or synthesis): a vision model looks at slices, zooms and overlays, measures regions with KonfAI Eval and reports localised defects with evidence, as a tracked job under Scout/<name>. Images reach a cloud model only with allow_cloud_images=True. Next: get_job_status(wait=True), read_scout_report.

### `run_train`

Launch a training job from the session Config.yml (after check_config). Omitting gpu trains on CPU. Returns a job payload, or a blocker payload naming what is missing. Next: read_live_metrics, get_job_status(wait=True).

### `run_transform`

Batch-process a dataset with the session Transform.yml chain: read, transform, write. dry_run=True plans it first, always: it measures STREAM, LOAD, WHOLE-VOLUME, SKIP or REFUSED per case (touching the output stores) and over_budget says what the run would refuse. The run replans, skips what exists (overwrite recomputes) and refuses a case that fits neither streaming nor memory_budget. Returns a job payload. Next: get_job_status(wait=True), inspect_dataset on the output.

### `session`

Manage session workspaces: action 'list'; 'create' (name; switches to it); 'switch' (name); 'delete' (the current one; force cancels its jobs); 'import' (adopt an on-disk KonfAI experiment from source_dir: its configs and code, its artifacts linked, copied or left out). create, switch and delete are refused while a job runs or when a host UI pinned the session. Next: initialize_session, summarize_session.

### `summarize_session`

One compact snapshot of the current session: readiness, jobs, metric summaries, and on request the leaderboard, a log tail and a config validation. Next: check_config or a run_* tool.

### `write_session_file`

Write one support file into the current session (Model.py, Loss.py, a transform, a manifest). Not validated. Next: inspect_object_signature, check_config.

### `write_workflow_config`

Write one workflow YAML (train, prediction, evaluation, transform) into the session and check it against KonfAI's schema: schema_errors lists every mistake with the nearest valid key; the file is written either way. Next: fix every error and write again, then check_config.

## Prompts

### `clarify_task_and_groups`

Ask the user the minimum questions needed to map dataset groups to a KonfAI task.

### `debug_config_warning`

Reason about KonfAI semantic warnings and propose the next checks before editing YAML.

### `plan_config_strategy`

Plan a KonfAI config-writing strategy from task, dataset summary, and modeling intent.

### `solve_task`

Route a dataset+goal request: use an existing app, fine-tune one, or train from scratch.

## Resources

- `server://info`: Return a compact summary of the MCP server workspace and in-memory jobs.
- `server://capabilities`: Describe the runtime resources visible to the MCP server for device selection.
- `guide://tool-index`: Read the guide to the MCP tool and prompt surface, generated from the registry so it can never drift.
- `guide://config-design`: Read the compact KonfAI config-design summary first, then go into docs if needed.
- `docs://index`: List the broader reasoning docs available through the MCP server.
- `docs://patching`: Read the detailed KonfAI patching doc.
- `docs://modeling`: Read the detailed KonfAI modeling doc.
- `docs://configuration`: Read the detailed KonfAI configuration doc.
- `docs://prediction`: Read the prediction authoring doc: TTA, multi-model ensembles, outputs_dataset reassembly.
- `docs://compute`: Read the compute doc: device selection, DDP semantics, memory knobs, SLURM submission.
- `docs://dataset-mapping`: Read the dataset-to-task mapping doc for clarifying inputs, targets, and support groups.
- `docs://examples`: Read how example templates should be used by an agent.
- `templates://list`: List example templates that can seed the current session workspace.
- `sessions://list`: List session workspaces available under the workspace root.
- `session://current/summary`: Read the current summary for the current session workspace.
- `session://current/log`: Read the latest available session log tail.
- `session://current/metrics`: Read the latest evaluation metrics for the current session as structured JSON.
- `apps://catalog`: Return the resolved app-source catalogue (shipped default + workspace file + env override).
- `template://{name}/summary`: Read the compact template summary, including config/model hints, before opening broader docs.
- `session://current/config/{workflow}`: Read one current-session config chosen by workflow.
- `job://{job_id}/status`: Read the latest status payload for one job.
- `job://{job_id}/log`: Read the latest job log tail.
- `job://{job_id}/manifest`: Read the immutable manifest captured when the job was launched.

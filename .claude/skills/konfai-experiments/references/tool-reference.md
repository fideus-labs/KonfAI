# KonfAI MCP — Tool Reference

> GENERATED from the registry by `konfai-mcp/scripts/generate_tool_reference.py` — do not edit by hand.

56 tools, 4 prompts, 23 resources. The live equivalent is the `guide://tool-index` resource.

## Tools

### `browse_dataset`

Use first when a dataset path may contain nested roots, cohorts, or ambiguous structure. This returns a bounded file tree and candidate dataset roots. It does not infer the task or write configs. Inputs: dataset_dir, optional depth, optional max_entries. Outputs: tree, candidate_dataset_roots, common_groups, missing_by_case, and next_actions. Next: inspect_dataset on one chosen root.

### `cancel_job`

Use when a running job should stop. This requests cancellation and waits briefly for a clean shutdown. It does not delete any session artifacts. Inputs: job_id and optional wait_s. Outputs: final job payload after cancellation. Next: summarize_session.

### `check_external_dependency`

Use to PRE-FLIGHT an external library before integrating a brick from it (e.g. before referencing `monai.losses:DiceLoss` or wrapping `segmentation_models_pytorch`). This reports whether the library is importable, its version and license, whether it is already a KonfAI dependency, and an install hint -- WITHOUT importing the library into the server process (no import side effects run here). Only the TOP-LEVEL package is checked ('monai.losses' checks 'monai'): it answers 'not installed' vs 'installed', not whether the submodule or class exists -- use inspect_object_signature to verify the full classpath. Inputs: module (e.g. 'monai' or 'monai.losses'), optional object name. Outputs: installed, version, license, distribution, is_konfai_dependency, install_hint, caution, next_actions. Next: inspect_object_signature on the chosen classpath, or describe_extension_points to write a wrapper.

### `compare_runs`

Use to COMPARE two runs metric-by-metric on aligned cases: means, per-case deltas, and a direction-aware winner per metric (loss-like metrics count lower as better). This reads both runs' Metric_<SPLIT>.json; it does not rerun evaluation. Inputs: run_a, run_b, optional split (default TRAIN), optional metric (suffix filter), optional session. Outputs: metrics {direction, cases, mean_a/mean_b, mean_delta_b_minus_a, cases_better_a/b, winner, per_case_delta_b_minus_a}, next_actions. Next: get_run_metrics on the winner, or leaderboard.

### `create_session`

Use to CREATE a named session workspace (and switch to it by default), so different experiments or config families live in isolated workspaces instead of overwriting one another. This creates sessions/<name> under the workspace root and makes it the current session. It does not seed configs (initialize_session does) and refuses to switch while a job is active. Inputs: name, optional switch (default true). Outputs: session, created, switched, sessions, next_actions. Next: initialize_session or import_experiment in the new session.

### `delete_session`

Use when you want to remove the current session workspace. This deletes the workspace and can cancel active jobs when forced. It does not preserve artifacts. Inputs: optional force. Outputs: deleted session name and path. Next: none unless you want to reinitialize the session.

### `describe_app`

Use to read one app's manifest so you can decide whether it matches the user's task -- the app's free-text description is the primary signal, with the input/output modality confirming the fit. This resolves a single app and returns its app.json: display name, description, input and output modality (with volume types), inference/evaluation/uncertainty capabilities, checkpoints, and segmentation terminology. It is metadata-only and SAFE: it does not import the app's model code and does not pip-install its requirements (those happen only later, behind an explicit trust gate). Inputs: ref (an app id 'repo_id:app_name', a local app folder path, or 'host:port:name' for a remote server -- a bare HuggingFace repo_id is NOT accepted here; expand it into app ids with list_apps first), optional force_update. Outputs: display_name, description, inputs, outputs, capabilities, checkpoints, terminology, next_actions. Next: run_app_infer / list_app_parameters / fine_tune_app when it fits (next_actions reflect the app's capabilities), or design_config_strategy if no app fits the task.

### `describe_config_schema`

Use before authoring a config to learn the top-level schema of a workflow. This is GENERATED from the Trainer/Predictor/Evaluator constructor via KonfAI's reflection engine, so it never drifts: it returns each top-level field with its type, default, whether it is required, and -- for nested config objects -- a classpath to drill into with inspect_object_signature. It does not return a full ready-to-run config; combine it with the example templates. Inputs: workflow (train/prediction/evaluation), optional path (drill into nested config levels with their YAML keys, e.g. path='Dataset.Patch' or path='Model'). Outputs: root_key, yaml_path, fields[{name,yaml_key,type,default,default_hidden,required,nested_config_classpath}], next_actions. Each field's yaml_key is the LITERAL key to write in the YAML (no casing guesses); default_hidden=true means the default exists but is not JSON-serializable here. Next: describe_config_schema with a deeper path, inspect_object_signature, then write_workflow_config.

### `describe_extension_points`

Use when you want to ADD or EXTEND a component (a new loss, metric, model/network, augmentation, transform, scheduler, or pretrained model) and need to know exactly where/how to plug it into KonfAI. This returns the extension contract per kind: the base class to subclass, the required methods and return/forward contract, where it is referenced in the YAML, and the THREE classpath syntaxes -- builtin name, local `File:Class`, and external `package.module:Class` (e.g. `monai.losses:DiceLoss`) -- plus the load-bearing gotcha for that kind. It does not write code or fetch anything. Inputs: optional kind (loss/metric/model/augmentation/transform/scheduler/pretrained; omit for all). Outputs: extension_point(s), yaml_reference_syntax, principle, next_actions. Next: check_external_dependency, list_components, inspect_object_signature, then write_session_file.

### `describe_konfai_capabilities`

Use at the start, or whenever you need to orient, to learn what KonfAI can do and which tool to reach for. This returns a capability overview: the three workflows, the component kinds (+ list_components), the extension model (+ describe_extension_points, including external-library classpaths), modeling modes, advanced capabilities, and which actions are safe vs should prefer human confirmation. It is a router to other tools and to AGENTS.md (the canonical reference), not a workflow to execute. Inputs: none. Outputs: a structured capability map + next_actions. Next: inspect_dataset, describe_config_schema, list_components, describe_extension_points.

### `describe_model_outputs`

Use to ENUMERATE a model's addressable module paths — the exact keys outputs_criterions and outputs_dataset accept — instead of guessing dotted paths and reading MeasureError lists from failed runs. This builds the workflow from the session config (side-effect-free, like validation) and walks every Network's module graph; terminal=true marks output heads (deep-supervision losses attach to non-terminal paths). Inputs: workflow (default train), optional config_file (alternate train config). Outputs: networks {attr: [{path, terminal, module}]}, reference_hint, next_actions. Next: write_workflow_config, then validate_config_semantics.

### `design_config_strategy`

Use once the user task is known and the dataset root is understood. This builds a config-writing plan from task, one or more dataset roots, group roles, workflows, and modeling intent. It does not write YAML or launch runs. Inputs: task, dataset_dir or dataset_dirs, optional group_roles, optional workflows, optional modeling_intent, optional example, optional extension. Outputs: dataset_summary, config_plan, customization_options, unresolved_questions, compatible_examples, guidance_resources, next_actions, and optional next_resources. Next: initialize_session, optionally write_session_file, then write_workflow_config.

### `diff_run_configs`

Use to DIFF the exact configs two jobs ran with, from their immutable launch-time snapshots — 'what changed between run A and run B' without trusting memory. It does not diff live session files (they may have been rewritten since). Inputs: job_id_a, job_id_b, optional filename (default Config.yml). Outputs: identical flag, unified diff text, next_actions. Next: compare_runs on the two runs' metrics.

### `export_app`

Use to SAVE a HuggingFace / remote-cached app (optionally with tuned parameters) as a local, editable app bundle -- the reproducibility artifact a challenge submission wants. It copies the app's files and, when set_parameters is given, bakes those values into the copied config. Distinct from package_app_from_session, which packages a model YOU trained this session. It copies files and rewrites config only (no model-code import). Local/HuggingFace apps only. Inputs: ref, path (destination folder), optional display_name, optional set_parameters, force_update. Outputs: exported_to, next_actions. Next: describe_app / run_app_infer / register_app_source on the copy.

### `export_run_record`

Use to EXPORT the full reproducibility record of one run: the job manifest (command, devices, environment snapshot with package versions and GPUs), the launch-time config snapshots' CONTENT, the post-run resolved config, every split's metrics, and a log tail — a Methods-section-grade record in one payload. It does not rerun anything. Caveat: resolved_config is read from the LIVE session config, which may have been rewritten since the run -- the launch-time truth is config_snapshots. Inputs: run_name OR job_id, optional log_lines (default 100). Outputs: job, manifest, config_snapshots (text), resolved_config, metrics per split, log_tail. Next: compare_runs or read_training_curves.

### `fine_tune_app`

Use to TRAIN by starting from a published app instead of a blank slate: fine-tune an existing app's checkpoint(s) on the user's dataset. This is the middle option between run_app_infer (use as-is, no training) and design_config_strategy (author a config and train from scratch). It launches a tracked training job and writes a resolvable app bundle (config + code + fine-tuned checkpoint) to the output directory, which you can then run with run_app_infer. TRUST GATE: a local or HuggingFace app imports its Python code and pip-installs its requirements, so pass allow_untrusted_code=True to confirm you trust the source; a remote app trains on the user's own server (the dataset is uploaded there) and needs no code gate. It does not author a config or adapt the dataset layout for you. Inputs: ref (app id, local app folder path, or 'host:port:name[|token]'); dataset (a KonfAI-style dataset directory); optional output bundle dir; optional name, epochs, it_validation, models (which checkpoints to fine-tune), lr, gpu/cpu; allow_untrusted_code; force_update. Outputs: a job payload (status, resources, next_actions) plus mode and the bundle output path. Next: wait_for_job, then run_app_infer on the produced bundle.

### `generate_folds`

Use to SPLIT a dataset into K cross-validation folds: writes one case-list file per fold into the session workspace and returns the exact subset stanzas to paste into the configs. KonfAI's Dataset.subset accepts a case-list file ('folds/fold_0.txt' keeps those cases) and its '~file' negation (trains on every OTHER fold). Inputs: dataset_dir, optional k (default 5), optional seed. Outputs: folds {fold_i: {cases, file, train_subset, eval_subset}}, how_to_use, next_actions. Next: write per-fold configs (distinct train_name each), then run_batch.

### `get_job_status`

Use when you need the latest state for one job without waiting. This returns the current job payload and suggested next actions. It does not parse runtime metrics. Inputs: job_id. Outputs: status payload with resources and next_actions. Next: wait_for_job or read_live_metrics.

### `get_run_metrics`

Use to read the FULL evaluation metrics (per-case values + aggregates) of ONE named run, instead of the newest-file-only view of session://current/metrics — essential when comparing specific past runs. This reads Evaluations/<run_name>/Metric_<SPLIT>.json in the current session. It does not rerun evaluation. Inputs: run_name (the train_name of the run), optional split (default TRAIN; the error lists available runs and splits on a miss), optional session (read another session's run without switching). Outputs: run_name, split, path, updated_at, metrics (full JSON), summary, next_actions. Next: leaderboard or summarize_session.

### `import_experiment`

Use to ADOPT an existing on-disk KonfAI experiment (its Config/Prediction/Evaluation.yml, custom .py and .yml files, and optionally its Checkpoints/Predictions/Evaluations/Statistics/Dataset artifacts) into the current session workspace, so the server can read, validate, rerun, resume, and compare it. Artifacts are symlinked by default (no copy of large checkpoints); pass include_artifacts='copy' to copy or 'none' to import configs/code only. Existing session files are kept unless overwrite=True. Inputs: source_dir, optional include_artifacts (link|copy|none), optional overwrite. Outputs: source, copied, linked, skipped, next_actions. Next: read_session_file / review_config_semantics, then validate_config_semantics.

### `initialize_session`

Use when moving from strategy to the concrete current session workspace. This creates or resets the current session workspace and can seed selected workflow files from one example template. DESTRUCTIVE with overwrite=True: it DELETES everything in the existing workspace, trained Checkpoints/ and Predictions/ included -- to keep those, switch_session to a new name instead. It does not adapt example YAML to your dataset automatically. Referenced .yml models are always seeded; an example whose model/loss lives in a local .py (e.g. Synthesis) needs include_support_files=True to be runnable -- the result carries a warning otherwise. Inputs: optional from_example, optional workflows, optional include_support_files, optional overwrite. Outputs: created workspace paths, copied files, resources, and next_actions. Next: write_workflow_config or inspect copied template files.

### `inspect_dataset`

Use after the dataset root is chosen (browse_dataset first when the root is ambiguous). This returns group structure, dataset entry hints, ambiguities, and (by default) sampled statistics for one dataset root; pass include_stats=False for a fast structure-only scan, or groups=[one group] for focused statistics. It does not infer the task or choose a final model. Inputs: dataset_dir, optional groups, optional extension, optional max_cases_per_group, optional seed, optional include_stats. Outputs: groups, statistics, dataset_entry hints, warnings, and next_actions. Next: design_config_strategy.

### `inspect_object_signature`

Use when choosing, customizing, or debugging any configurable object classpath such as a model, loss, transform, or helper module. This returns local or imported signature details, defaults, doc summary, and detected contract hints. It does not validate the full workflow config or decide which object to use. Inputs: classpath. Local Module:Object classpaths are resolved inside the current session workspace and parsed statically (never executed); an installed-library classpath is actually imported into the server process, so its import side effects run. Outputs: source type, signature, parameters, defaults, detected_contract, limitations, and next_actions. Next: write_session_file, write_workflow_config, or review_config_semantics.

### `leaderboard`

Use after evaluation when you want ranked metrics across completed runs. This reads Metric_<split>.json files and builds a leaderboard. It does not rerun evaluation. Inputs: optional metric, optional split (default TRAIN; maps to Metric_<SPLIT>.json — a miss lists the available splits), optional limit, optional direction ('min'/'max' override when the ranking direction inferred from the metric name is wrong; applies to every metric in the payload), optional session (rank another session's runs without switching). Outputs: available_metrics, available_splits, selected_metric when resolved, leaderboard rows, best row, warnings, and next_actions. Next: get_run_metrics on a chosen run, summarize_session, or launch another run.

### `list_app_parameters`

Use to DISCOVER an app's tunable model parameters (and their allowed values) before tuning a run with set_parameters. Returns {values, constraints}: current values plus Literal/Range/Choices constraints derived from the model's typed signature. TRUST GATE: deriving constraints imports the app's model code, so pass allow_untrusted_code=True; the import runs in an isolated spawn subprocess, never in the server process. Local/HuggingFace apps only (a remote server does not expose this). Inputs: ref, allow_untrusted_code, optional force_update. Outputs: values, constraints, next_actions. Next: run_app_infer / run_app_pipeline with set_parameters.

### `list_apps`

Use FIRST when the user wants a result from an existing model and has NOT asked to train one: check whether a published KonfAI app already does what they want, before authoring and training a config from scratch. An app can do any task, so judge whether one fits from its own description (read it with describe_app) and its declared inputs/outputs -- not from any preset list of tasks. This enumerates apps from a referenced catalogue (shipped default + the editable workspace file + the KONFAI_MCP_APP_CATALOG env file), expanding bare HuggingFace repo ids into their contained apps. It does not run inference or import any app code; without include_summary it does not even resolve manifests. Inputs: optional repos (ad-hoc override list), optional include_summary (fetch each display_name / short_description / modality, slower), optional force_update (refresh HuggingFace listings). Outputs: apps [{ref, source, repo, app_name}], catalog provenance, errors, next_actions. Next: describe_app on a candidate to read its manifest, else design_config_strategy to train one.

### `list_components`

Use to DISCOVER which KonfAI components exist before authoring a config from scratch, when you do not already know the class name/path to put in the YAML. This enumerates the built-in component zoo for one kind. It does not return full constructor signatures -- chain to inspect_object_signature for that. Inputs: kind, one of ['criterion', 'transform', 'augmentation', 'scheduler', 'model', 'block'] (aliases: loss/metric -> criterion, etc.). Outputs: components [{name, config_reference, inspect_classpath, module, doc}], a reference_hint explaining where the name goes in the config, and next_actions. Next: inspect_object_signature on a chosen component, then design_config_strategy or write_workflow_config.

### `list_jobs`

Use when you need the current job registry state. This lists jobs for the current session workspace. It does not wait for jobs or parse live metrics. Inputs: none. Outputs: job payloads sorted by creation time. Next: get_job_status, wait_for_job, or read_live_metrics.

### `package_app_from_session`

Use to PACKAGE a model trained in the current session (the train-from-scratch branch) into a resolvable KonfAI app bundle -- the same endpoint fine_tune_app produces, so a from-scratch run can also finish as a reusable app. It gathers the session's checkpoints and a config, writes an app.json from the metadata you give, and assembles a bundle (app.json + config + checkpoint + optional Model.py/requirements) that describe_app and run_app_infer can then consume. It does not train, and it does not upload the bundle anywhere. Inputs: name (bundle folder), display_name, description, optional short_description/tta/mc_dropout; optional checkpoints and configs (default: discovered from the session Checkpoints/ and Prediction.yml/Config.yml); optional model_py, requirements, output dir; optional onnx export (onnx, onnx_patch_size, onnx_in_channels). Outputs: bundle_path, the packaged checkpoints/configs, next_actions (and onnx path if requested). Next: describe_app or run_app_infer on the produced bundle.

### `prepare_dataset_aliases`

Use when the dataset has the right content but the group filenames do not match your intended config. This creates copied, symlinked, or moved aliases for dataset files. It does not change YAML configs for you. Inputs: dataset_dir, rename_map, optional mode, optional overwrite, optional allow_destructive. Outputs: created paths, missing_by_case, and next_actions. Next: inspect_dataset or design_config_strategy.

### `preview_volume`

Use to SEE a volume: returns one slice as a PNG image (rendered in image-capable MCP clients) for qualitative QC of a dataset case or a produced prediction -- orientation, field of view, obvious artefacts -- instead of judging from numbers alone. This reads any SimpleITK-readable file (mha/nii.gz/...), windows it between the 1st and 99th percentile, and downsamples to max_size. It does not modify the file. Inputs: path (volume file), optional slice_index (default: middle), optional axis (default 0 = first/depth axis), optional max_size (default 512). Outputs: a PNG image content block. Next: inspect_dataset for numbers, or preview_volume on other slices/axes.

### `read_dataset_file`

Use to READ a dataset's small non-image text file: a labels/metadata CSV or TSV, a JSON/YAML sidecar, a case-list txt, or a text header (.mhd/.nhdr). SAFE: bounded read-only preview (it streams at most max_chars characters); binary files are refused with a pointer to inspect_dataset/preview_volume. It does not parse image volumes and does not modify anything. Inputs: path, optional max_lines, optional max_chars. Outputs: content (bounded), total_bytes, truncated; CSV/TSV additionally get columns + rows. Next: inspect_dataset or design_config_strategy.

### `read_job_log`

Use to READ a job's log as a tool — the crash-triage primitive: tail more than the fixed resource tail, page through it, or filter it with a regex to find the traceback. This reads the job console log (or the KonfAI runtime log when present) and returns the selected lines. It does not parse metrics; use read_live_metrics for parsed metrics. Inputs: job_id, optional max_lines (default 200), optional grep (regex applied per line, over a bounded window of the last max(20*max_lines, 2000) lines, before the tail is taken), optional source ('auto' prefers the runtime log for running/done jobs and the console job log — where a crash traceback lives — for failed ones; or 'job'/'runtime'). Outputs: job_id, status, path, content, lines_returned, next_actions. Next: validate_config_semantics then the matching run_* tool to retry, or cancel_job.

### `read_live_metrics`

Use while a job is running and you want parsed runtime metrics instead of raw logs. This reads the runtime log and returns recent metric snapshots. It does not block until the job completes. Inputs: kind, optional job_id, optional max_entries. Outputs: latest metric snapshot, recent entries, by_stage summaries, and job metadata. Next: wait_for_job or summarize_session.

### `read_session_file`

Use to READ BACK a file from the current session workspace: a config, a support file you wrote (Model.py, Loss.py), a copied template file (UNet.yml, Config_GAN.yml), a job config snapshot (the manifest's config_snapshots paths), or an evaluation JSON. This returns a bounded character range of one workspace file; absolute paths are accepted when they resolve inside the workspace. It does not read files outside the session workspace. Inputs: path (workspace-relative, or absolute inside the workspace), optional max_chars, optional offset. Outputs: path, relative_path, content, offset, returned_chars, total_bytes, truncated, next_actions. Next: write_session_file or write_workflow_config to edit, then review_config_semantics.

### `read_template_file`

Use to READ a file shipped with an example template — a reference implementation such as a local model (Model.py), a custom transform (UnNormalize.py), a declarative model (UNet.yml), or an alternate config (Config_GAN.yml) — so you can understand or adapt it instead of guessing what it contains. This returns a bounded character range of one template file. It does not modify templates. Inputs: name (template), filename (a direct child of the template directory), optional max_chars, optional offset. Outputs: template, filename, content, truncated, next_actions. Next: write_session_file to adapt it into the session, or initialize_session to copy files wholesale.

### `read_training_curves`

Use to read a run's TRAINING CURVES (loss/metric scalars over iterations) from the TensorBoard event files KonfAI writes under Statistics/<run_name>/ — the full history, not just the live log tail. This parses tfevents into downsampled scalar series. It requires the 'tensorboard' package. Inputs: run_name, optional tags (substring filters), optional max_points (default 200), optional session. Outputs: tags, curves {tag: [{step, value}]}, next_actions. Next: compare_runs or leaderboard.

### `register_app_source`

Use when the user points at their own app or HuggingFace repo and wants it to persist across sessions. This appends an app reference to the editable workspace catalogue file (the same one list_apps merges). It does not validate that the reference resolves -- call describe_app to check. Inputs: ref (an app id or a bare HuggingFace repo_id). Outputs: ref, added flag, catalog_path, the updated apps list, next_actions. Next: list_apps or describe_app.

### `review_config_semantics`

Use immediately after writing or editing a workflow config. This performs lightweight semantic checks and returns warnings plus blocking issues. It does not instantiate KonfAI runtime objects. Inputs: workflow. Outputs: summary, warnings, blocking_issues, next_checks, and next_actions. Next: validate_config_semantics if there are no blocking issues.

### `run_app_evaluate`

Use to score an app's predictions against ground truth with the app's OWN evaluation config (its shipped Evaluation.yml and metrics), after describe_app reported capabilities.evaluation. This is distinct from run_evaluation, which needs a hand-authored session Evaluation.yml. It launches a tracked job and writes the metric JSON to the output directory. TRUST GATE: a local/HuggingFace app imports its code and pip-installs (pass allow_untrusted_code=True); a remote app runs on the user's server. Inputs: ref; inputs (predictions, list of groups of paths); gt (ground truth, list of groups); optional output, mask, evaluation_file, gpu/cpu; allow_untrusted_code; force_update. Outputs: a job payload plus mode and output. Next: wait_for_job, then read the metric JSON.

### `run_app_infer`

Use to RUN a published KonfAI app on the user's data (the 'use an existing model instead of training' path), after describe_app confirmed the app fits. This launches a tracked inference job and reassembles the app's outputs into the given output directory. TRUST GATE: for a local or HuggingFace app, resolving it imports the app's Python code and pip-installs its requirements, so you MUST pass allow_untrusted_code=True to confirm you trust the source; a remote app (host:port:name) runs on the user's own server and needs no code gate (its inputs are uploaded there). It does not choose the app or prepare the data for you, and set_parameters is local/HuggingFace only. Inputs: ref (app id 'repo_id:app_name', local app folder path, or 'host:port:name[|token]'); inputs as a list of GROUPS (one inner list per input channel/modality, each a list of file or directory paths, paired by order across groups); optional output dir; optional gpu/cpu; optional tta, ensemble, ensemble_models, patch_size, batch_size, uncertainty; optional set_parameters (NAME->VALUE model tuning, e.g. {'iterations': 300}); allow_untrusted_code; force_update. Outputs: a job payload (status, resources, next_actions) plus mode and output. Next: wait_for_job, then inspect the output directory.

### `run_app_pipeline`

Use to run an app end to end in one shot: inference, then evaluation (when gt is given), then uncertainty. It writes Predictions / Evaluations / Uncertainties under the output directory. Prefer run_app_infer for a plain prediction; use this when you want the app's full scoring loop in a single call. TRUST GATE: local/HuggingFace apps import code and pip-install (pass allow_untrusted_code=True); remote runs on the user's server (set_parameters is local/HuggingFace only). Inputs: ref; inputs (list of groups); optional gt (enables evaluation), mask, output, gpu/cpu, tta, ensemble, ensemble_models, patch_size, batch_size, uncertainty (default true), set_parameters; allow_untrusted_code; force_update. Outputs: a job payload plus mode and output. Next: wait_for_job, then inspect the output subdirectories.

### `run_app_uncertainty`

Use to produce uncertainty maps from an app, after describe_app reported capabilities.uncertainty. This runs the app's Uncertainty.yml on multi-channel inference stacks (typically produced by run_app_infer with uncertainty=True). It is the separate step that consumes those stacks; run_app_infer's uncertainty flag only keeps the stack during inference. TRUST GATE: local/HuggingFace apps import code and pip-install (pass allow_untrusted_code=True); remote runs on the user's server. Inputs: ref; inputs (the multi-channel inference stacks, list of groups); optional output, uncertainty_file, gpu/cpu; allow_untrusted_code; force_update. Outputs: a job payload plus mode and output. Next: wait_for_job, then inspect the uncertainty maps.

### `run_batch`

Use to RUN A SWEEP: launch several training configs SEQUENTIALLY server-side (each waits for the previous to finish), collecting per-run outcomes -- fold training or hyperparameter variants in one call instead of hand-chaining run_train/wait_for_job. This blocks until the batch ends, like wait_for_job; each config needs a distinct train_name. Inputs: config_files (alternate train configs in the workspace, e.g. from generate_folds or write_session_file), optional gpu/cpu, overwrite, quiet (default true), single_process, stop_on_error (default true). Outputs: requested, completed, results [{config_file, job_id, run_name, status, error}], next_actions. Next: leaderboard or compare_runs.

### `run_component_smoke_test`

Use to SMOKE-TEST a component you wrote or referenced BEFORE wiring it into a config: it executes the component's runtime contract on dummy tensors. For a transform it asserts transform_shape(shape) == __call__(tensor).shape — the contract whose silent violation corrupts patch reassembly; for a criterion it reports Tensor-vs-tuple return (loss vs metric convention) and whether backward() works. TRUST: this imports and EXECUTES the component's code — in an isolated spawn subprocess, never in the server process — but still only run it on code you or the user wrote. Inputs: classpath (local File:Class, builtin name, or package.module:Class), kind (transform/criterion/loss/metric), optional shape (default [1,8,8,8]), optional init_kwargs. Outputs: ok, stage, contract details (predicted vs actual shape, returns, backward_ok) or the full traceback. Next: write_workflow_config when ok, or write_session_file to fix the component.

### `run_evaluation`

Use after evaluation config review/validation and when required artifacts exist. This launches an evaluation job and returns structured job resources. It does not infer missing predictions. Inputs: optional gpu as an int or list of ints, optional cpu, overwrite, quiet, single_process. Outputs: job payload with resources and next_actions; or, when a prerequisite is missing (dataset path, checkpoint), a blocker payload {ok, blocked, error, missing_paths, next_actions} with no job_id/status. Next: wait_for_job then summarize_session.

### `run_prediction`

Use after prediction config review/validation and when a checkpoint exists. This launches a prediction job and returns structured job resources. It does not search outside the current session workspace for missing checkpoints. Inputs: optional models as a string or list of strings, optional gpu as an int or list of ints, optional cpu, overwrite, quiet, single_process. Outputs: job payload with resources and next_actions; or, when a prerequisite is missing (dataset path, checkpoint), a blocker payload {ok, blocked, error, missing_paths, next_actions} with no job_id/status. Next: wait_for_job.

### `run_resume`

Use to RESUME an interrupted or crashed training run from a checkpoint: model, optimizer, scheduler, and epoch/iteration counters are restored (KonfAI's RESUME command), unlike fine_tune_app which restarts from weights only. This launches a resumed training job from the current session Config.yml. It does not pick between runs: by default it resumes from the newest checkpoint of the configured run (falling back to the newest in the session), avoiding cross-run contamination. It trains up to the LIVE config's epochs: if the run already completed them, raise epochs in Config.yml first or the resume finishes immediately without adding checkpoints. Inputs: optional model (checkpoint path; default as above), optional lr (override the restored learning rate; omit to continue the schedule), optional gpu/cpu, overwrite, quiet, tensorboard, single_process. Outputs: job payload with resources and next_actions; or, when a prerequisite is missing (dataset path, checkpoint), a blocker payload {ok, blocked, error, missing_paths, next_actions} with no job_id/status. Next: wait_for_job or read_live_metrics.

### `run_train`

Use after train config review and validation succeed. This launches a training job and returns structured job resources. It does not choose the device or repair config issues automatically -- omitting gpu trains on CPU, so pass gpu explicitly for GPU training. Inputs: optional gpu as an int or list of ints, optional cpu, overwrite, quiet, tensorboard, single_process, optional config_file (an alternate train config in the workspace, e.g. Config_GAN.yml), optional cluster ({name, memory, num_nodes, time_limit} submits via SLURM/submitit instead of running locally). Jobs on DISJOINT devices may run concurrently; same-device jobs are refused. Outputs: job payload with resources and next_actions; or, when a prerequisite is missing (dataset path, checkpoint), a blocker payload {ok, blocked, error, missing_paths, next_actions} with no job_id/status. Next: read_live_metrics or wait_for_job.

### `summarize_session`

Use when you want one compact session snapshot for planning the next action. This returns readiness, job state, metric summaries, and an optional leaderboard, log tail, or config validation (include_validation=True; off by default to keep the payload lean). It does not launch or repair workflows. Inputs: optional leaderboard, log, and validation controls. Outputs: readiness, metrics_summary, validation, leaderboard, and next_actions. Next: review_config_semantics, validate_config_semantics, or run a workflow.

### `switch_session`

Use to SWITCH the server onto another existing session workspace (create_session makes new ones). All session-scoped tools and resources then operate on that workspace; its persisted job history is reloaded. It refuses to switch while a job is active in the current session. Inputs: name. Outputs: session, sessions, summary, next_actions. Next: summarize_session, or leaderboard(session=...) to compare without switching.

### `unregister_app_source`

Use to drop an app reference previously added to the workspace catalogue. This removes a reference from the editable workspace catalogue file. It does not touch the shipped default catalogue or the KONFAI_MCP_APP_CATALOG env file. Inputs: ref. Outputs: ref, removed flag, catalog_path, the updated apps list, next_actions. Next: list_apps.

### `validate_config_semantics`

Use after semantic review when the config looks coherent enough to instantiate. This instantiates or sets up KonfAI workflow objects in an ISOLATED spawn subprocess to catch runtime-facing errors: edited workspace code is always re-imported fresh, and nothing executes in the server process. It does not launch jobs. Levels: 'instantiate' builds the objects, 'setup' also builds the dataloader, 'train_step' additionally runs ONE forward+backward on ONE batch (train workflow only, single-process CPU, no checkpoint, config restored) to catch runtime-only errors -- target dtype/shape mismatches, an outputs_criterions key that does not resolve, a detached loss. Inputs: workflow or 'all' (validate every present config), validation level, and optional models for prediction. Outputs: ok flag, runtime details, semantic review, and next_actions. Next: run_train, run_prediction, run_evaluation, or fix the config.

### `wait_for_job`

Use after launching a job when you want to block until it finishes. This polls job state until the job reaches a terminal status. It does not stream logs. Inputs: job_id, optional timeout_s (omit/None to wait until the job finishes -- recommended for real multi-hour training; pass a number only to bound the wait, which raises TimeoutError on expiry), optional poll_interval_s. Outputs: final job payload. Next: summarize_session or leaderboard.

### `write_session_file`

Use for session-side support files such as local model code, custom losses, transforms, helper modules, or manifests. This writes one file inside the current session workspace. It does not validate Python semantics. Inputs: relative_path, content, optional overwrite. Outputs: written path, byte count, and next_actions. Next: inspect_object_signature, review_config_semantics, or validate_config_semantics.

### `write_workflow_config`

Use to author or replace one workflow YAML file. This validates the top-level KonfAI root key and writes the config into the current session workspace. It does not patch YAML structurally for you. Inputs: workflow, content, optional overwrite. Outputs: written path, byte count, and next_actions. Next: review_config_semantics.

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

- `server://info` — Return a compact summary of the MCP server workspace and in-memory jobs.
- `server://capabilities` — Describe the runtime resources visible to the MCP server for device selection.
- `guide://tool-index` — Read the guide to the MCP tool and prompt surface, generated from the registry so it can never drift.
- `guide://config-design` — Read the compact KonfAI config-design summary first, then go into docs if needed.
- `docs://index` — List the broader reasoning docs available through the MCP server.
- `docs://patching` — Read the detailed KonfAI patching doc.
- `docs://modeling` — Read the detailed KonfAI modeling doc.
- `docs://configuration` — Read the detailed KonfAI configuration doc.
- `docs://prediction` — Read the prediction authoring doc: TTA, multi-model ensembles, outputs_dataset reassembly.
- `docs://compute` — Read the compute doc: device selection, DDP semantics, memory knobs, SLURM submission.
- `docs://dataset-mapping` — Read the dataset-to-task mapping doc for clarifying inputs, targets, and support groups.
- `docs://examples` — Read how example templates should be used by an agent.
- `templates://list` — List example templates that can seed the current session workspace.
- `sessions://list` — List session workspaces available under the workspace root.
- `session://current/summary` — Read the current summary for the current session workspace.
- `session://current/log` — Read the latest available session log tail.
- `session://current/metrics` — Read the latest evaluation metrics for the current session as structured JSON.
- `apps://catalog` — Return the resolved app-source catalogue (shipped default + workspace file + env override).
- `template://{name}/summary` — Read the compact template summary, including config/model hints, before opening broader docs.
- `session://current/config/{workflow}` — Read one current-session config chosen by workflow.
- `job://{job_id}/status` — Read the latest status payload for one job.
- `job://{job_id}/log` — Read the latest job log tail.
- `job://{job_id}/manifest` — Read the immutable manifest captured when the job was launched.

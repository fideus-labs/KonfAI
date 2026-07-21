# KonfAI MCP Server

`konfai-mcp` turns KonfAI into a backend an LLM agent can *drive*. Give it a dataset
and a goal in plain language and it inspects the data, picks the cheapest path that
fits — reuse a published model, fine-tune one, or train from scratch — writes and
validates the config, runs training / prediction / evaluation, monitors the jobs, and
returns metrics with a record you can reproduce.

It is the agent-facing counterpart to the CLI: same framework, same YAML, but exposed
as structured, deterministic tools instead of hand-edited files. The package is kept
deliberately separate — `konfai` is the machine-learning framework; `konfai-mcp` is the
MCP server that exposes it to agents, and the core never depends on it.

## Vision

The goal of this package is to make KonfAI usable as a deterministic backend for
real scientific experimentation.

Typical target workflows include:

- reproduce a paper from a structured description
- train a model on a new dataset for a known task such as segmentation or synthesis
- run several experiments, evaluate them, and select the best run
- recover from invalid configs, missing artifacts, or dataset mismatches

The server is built around a few principles:

- agent-friendly tool contracts
- explicit experiment workspaces
- reproducible job manifests and config snapshots
- staged validation before expensive runs
- clear separation between the KonfAI runtime and the MCP orchestration layer

## What This Package Contains

This directory is a standalone Python package:

- package code: `konfai-mcp/konfai_mcp/`
- package metadata: `konfai-mcp/pyproject.toml`
- tests: `konfai-mcp/tests/`

It depends on `konfai`, but it is not part of the `konfai` package itself.

That separation is deliberate:

- `konfai` stays focused on training, prediction, evaluation, and runtime logic
- `konfai-mcp` owns MCP tool registration, workspaces, job management, and
  agent-oriented workflows

## Package Layout

Core runtime files:

- `konfai_mcp/server.py`: FastMCP server, tool registration, resources, and
  top-level orchestration
- `konfai_mcp/server_experiments.py`: experiment services, validation,
  dataset inspection, metrics, readiness, and summaries
- `konfai_mcp/server_jobs.py`: persistent subprocess job registry and manifests
- `konfai_mcp/server_support.py`: workspace paths, config IO, template copying,
  and filesystem helpers
- `konfai_mcp/runner.py`: subprocess runner used to execute or validate KonfAI
  workflows from the MCP server

Tests:

- `tests/test_mcp_server*.py`: MCP tool, reliability, pipeline, and end-to-end
  segmentation tests, plus shared helpers in `tests/mcp_test_helpers.py`

## Installation

### From the KonfAI repository

For local development, install both the KonfAI core package and the MCP package:

```bash
git clone https://github.com/fideus-labs/KonfAI.git
cd KonfAI

python -m pip install -U pip
pip install -e ".[dev]"
pip install -e ./konfai-mcp
```

This gives you the MCP entrypoint:

```bash
konfai-mcp
```

### Directly from the MCP package directory

If the KonfAI core package is already available in your environment:

```bash
cd konfai-mcp
pip install -e .
```

## Running the Server

You can launch the server with the installed script:

```bash
konfai-mcp
```

Or run the module directly:

```bash
cd konfai-mcp
python -m konfai_mcp.server
```

## Workspace Model

The server manages explicit experiment workspaces under a single root.

By default:

```text
~/KonfAI_Workspaces
```

You can override this with:

```bash
export KONFAI_MCP_WORKSPACES_ROOT=/path/to/workspaces
```

Each experiment gets its own workspace with configs, outputs, and internal job
metadata. The MCP server also persists:

- per-job logs
- per-job manifests
- config snapshots captured at launch time
- job state for recovery after server restart

Useful environment variables:

- `KONFAI_MCP_WORKSPACES_ROOT`: root directory for experiment workspaces
- `KONFAI_MCP_LOG_TAIL_LINES`: default log tail size exposed by the server

## How the MCP Server Works

The MCP server does not execute training logic itself.

Instead:

1. the server receives an MCP tool call
2. it prepares an experiment workspace and validates the current state
3. it launches a subprocess through `konfai_mcp.runner`
4. the runner builds the KonfAI workflow with `build_train`,
   `build_predict`, or `build_evaluate`
5. the workflow is executed through KonfAI runtime utilities
6. the job registry tracks status, logs, manifests, and snapshots

This design keeps the MCP layer thin and makes it easier to reuse KonfAI's
build/runtime split for validation and execution.

## Main MCP Resources

Structured resources exposed by the server include:

- `server://info` / `server://capabilities`
- `guide://tool-index` — the full tool + prompt index, **generated from the registry** (never drifts)
- `guide://config-design`, `docs://{index,patching,modeling,configuration,dataset-mapping,examples}`
- `templates://list`, `template://{name}/summary`
- `sessions://list`, `session://current/{summary,config/{workflow},log,metrics}`
- `job://{job_id}/{status,log,manifest}`
- `apps://catalog`

These resources are meant to be machine-readable checkpoints for an agent that
needs to recover state, inspect artifacts, or decide the next step.

## MCP Tools

Tools are grouped by stage below. To avoid drift and wasted tokens, each tool's
**contract (inputs, outputs, `next_actions`) is not duplicated here** — it lives
once, generated from the registry:

- live: the `guide://tool-index` resource, with `describe_konfai_capabilities`
  as the orientation hub
- doc: `.claude/skills/konfai-experiments/references/tool-reference.md`,
  regenerated by `konfai-mcp/scripts/generate_tool_reference.py`

At a glance, by stage:

- **Discovery** — `list_components`, `inspect_object_signature`,
  `describe_konfai_capabilities`, `describe_config_schema`,
  `describe_extension_points`, `check_external_dependency`
- **Dataset onboarding** — `browse_dataset`, `inspect_dataset`, `read_dataset_file`,
  `preview_volume`, `prepare_dataset_aliases`
- **Config authoring** — `design_config_strategy`, `initialize_session`,
  `write_workflow_config`, `write_session_file` / `read_session_file`,
  `read_template_file`
- **Validation & iteration** — `review_config_semantics`,
  `validate_config_semantics`, `summarize_session`, `leaderboard`,
  `get_run_metrics`, `compare_runs`, `diff_run_configs`, `describe_model_outputs`,
  `run_component_smoke_test`
- **Execution & monitoring** — `run_train` / `run_resume` / `run_prediction` /
  `run_evaluation`, `run_batch`, `generate_folds`, `list_jobs` /
  `get_job_status` / `cancel_job`, `wait_for_job`, `read_live_metrics`,
  `read_job_log`
- **Apps** — see [Use a published app](#use-a-published-app-instead-of-training)
- **Session lifecycle** — `create_session` / `switch_session` / `delete_session`,
  `import_experiment`, `export_run_record`

Hardware for device selection is a **resource, not a tool**: `server://capabilities`
reports per-GPU total / used / **free** VRAM and a recommended device, and every
train / predict / evaluate / fine-tune launch payload carries a `vram_preflight`
block for the GPUs it will use — so an agent sizes `batch_size` / `patch_size` to
the free VRAM without an extra call.

## Use a published app instead of training

Training from scratch is only one of three ways to satisfy a request. Many tasks
are already solved by a **published KonfAI app** (a config + code + weights bundle
on a local path, a HuggingFace repo, or a remote server). The MCP exposes the whole
*use / adapt / package* half of the lifecycle, so an agent can pick the cheapest
path that actually fits — **without ever training when a model already exists.**

The `solve_task` prompt frames the entry decision as a three-way fork:

1. **Use an app as-is** — no training. Discover with `list_apps`, read each
   candidate's manifest with `describe_app` (judge fit from the app's own
   description and its declared inputs/outputs), then run it:
   - `run_app_infer` — inference on the user's data
   - `run_app_evaluate` — score predictions with the app's own metrics
   - `run_app_uncertainty` — uncertainty maps
   - `run_app_pipeline` — infer → evaluate → uncertainty in one call
   - `list_app_parameters` / `set_parameters` — read tunable parameters (with
     their constraints) and override them per run
2. **Fine-tune an app** — start training from a published model rather than a
   blank slate: `fine_tune_app` adapts it to the user's dataset and writes a
   resolvable app bundle.
3. **Train from scratch** — author a config (the loop above) and, when done,
   `package_app_from_session` turns the trained model into a bundle too.

Both training paths therefore **end at the same reusable artifact — a bundle** —
which `describe_app` / `run_app_infer` can then consume, and `export_app` can
snapshot (with tuned parameters baked in) as the reproducibility record a
challenge submission wants.

**App catalogue.** `list_apps` reads a layered catalogue of app sources — a
shipped default, an editable per-workspace file, and the `KONFAI_MCP_APP_CATALOG`
env file (same `{"apps": [...]}` shape as the `konfai-apps` server's `--apps`),
plus an ad-hoc `repos=[...]` override. `register_app_source` / `unregister_app_source`
let a user pin their own HuggingFace repo or local app. A bare HuggingFace
`repo_id` or a `host:port` remote server entry is expanded into its apps.

> ⚠️ **Trust.** Resolving a **local or HuggingFace** app imports its Python code and
> pip-installs its requirements, so every execution / fine-tune / parameter-read tool
> is gated behind an explicit `allow_untrusted_code=True`. A **remote** app runs on the
> user's own server (its inputs are uploaded there) and needs no code gate.

## Typical Agent Workflow

For a dataset-driven task, the intended loop is:

0. `list_apps` / `describe_app` — check whether a published app already solves it
1. `inspect_dataset`
2. `design_config_strategy`
3. `initialize_session`
4. `write_workflow_config` (and `write_session_file` for custom components)
5. `review_config_semantics`
6. `validate_config_semantics`
7. `run_train`
8. `wait_for_job` (or `read_live_metrics`; `run_resume` after an interruption)
9. `run_prediction`
10. `wait_for_job`
11. `run_evaluation`
12. `summarize_session`
13. `leaderboard` / `get_run_metrics`

For iterative improvement, the agent can then:

- rewrite one or more configs
- validate again
- launch a new run
- compare metrics across runs

Many tool payloads include `next_actions` to help an LLM chain calls without
guessing the next valid step.

## Validation Strategy

The server supports semantic validation before full execution.

`validate_config_semantics` can check whether a workflow can actually be built,
and optionally set it up, using the current experiment state and available
artifacts.

This is important for agent workflows because it catches errors such as:

- broken YAML roots
- missing model checkpoints for prediction
- invalid dataset mappings
- evaluator or predictor setups that cannot run with current artifacts

Validation is tied to the real KonfAI build path, not to a parallel fake schema.

## Codex / MCP Client Integration

This package is intended to be exposed to an MCP client such as Codex through
the installed `konfai-mcp` command.

Example `config.toml` entry:

```toml
[mcp_servers.konfai]
command = "/path/to/venv/bin/konfai-mcp"
cwd = "/path/to/KonfAI/konfai-mcp"
startup_timeout_sec = 20
tool_timeout_sec = 3600

[mcp_servers.konfai.env]
KONFAI_MCP_WORKSPACES_ROOT = "/path/to/KonfAI/mcp-workspace"
KONFAI_MCP_LOG_TAIL_LINES = "400"
```

The exact client configuration depends on the MCP host, but the intended
entrypoint is the package command, not an ad hoc wrapper script.

## Testing

Run the full MCP test suite:

```bash
pytest -q konfai-mcp/tests
```

Useful subsets:

```bash
pytest -q konfai-mcp/tests/test_mcp_server.py
pytest -q konfai-mcp/tests/test_mcp_server_pipeline.py
pytest -q konfai-mcp/tests/test_mcp_server_segmentation_pipeline.py
pytest -q konfai-mcp/tests/test_mcp_server_reliability.py
```

The `test_mcp_server_segmentation_pipeline.py` end-to-end test drives a full
train -> prediction -> evaluation loop against the `examples/Segmentation`
template, so it requires the KonfAI imaging extras (`pip install -e ".[imaging]"`).

## CI

This package has a dedicated GitHub workflow, `.github/workflows/konfai_mcp_ci.yml`,
which:

- installs the KonfAI core package (with imaging extras) and the MCP package
- lints and format-checks `konfai_mcp`
- builds the standalone MCP package
- smoke-imports `konfai_mcp`
- runs `pytest -q konfai-mcp/tests`

> This package lives on the `konfai-mcp` branch of the repository.

## Current Scope and Limitations

This package is intended for real experimentation loops, but it is still scoped
to KonfAI-centered workflows.

It is strong at:

- KonfAI-style dataset onboarding
- config-template adaptation
- explicit train / prediction / evaluation loops
- run summaries, job monitoring, and experiment comparison

It does not try to solve everything:

- it does not automatically infer arbitrary research architectures from papers
- it still assumes KonfAI-compatible datasets and configs
- some advanced scientific workflows still need custom Python files or manual
  template adaptation

## Relationship to the Main Repository

The main KonfAI repository contains:

- the core ML framework in `konfai/`
- example templates in `examples/`
- the standalone MCP package in `konfai-mcp/`

That split is intentional. The MCP server depends on KonfAI, but it is treated
as its own package, with its own tests, CI, and release path.

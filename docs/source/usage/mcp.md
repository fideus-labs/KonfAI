# Agent workflows (MCP server)

Point an LLM agent at a folder of scans and ask in plain language — *"segment these CT
volumes"*, *"train an MR-to-CT model on my dataset"*, *"register these two scans and
tell me whether the alignment holds"* — and let it carry the request out **end to end**:
read the data, choose the cheapest path that fits, write and validate the config, run
train / predict / evaluate, and hand back metrics with a record you can reproduce.

That is **`konfai-mcp`**: a
[Model Context Protocol](https://modelcontextprotocol.io) server that exposes the
whole framework as structured, deterministic tools. The CLI is driven by a human
editing YAML; the MCP server hands that same power to an agent, over a typed tool
contract whose payloads carry `next_actions` hints that chain the steps. It is a
separate package layered on KonfAI's public API — the core framework never depends
on it.

## What it takes off your plate

- **No config archaeology.** The agent writes the *same* YAML you would, and every run
  leaves a fully-resolved config on disk — the experiment *is* the file, reproducible by
  anyone who has it.
- **The cheapest path, not always training.** Many requests are already solved by a
  published model. The agent checks that first (`list_apps` → `describe_app`), runs or
  fine-tunes it on your data, and only trains from scratch when nothing fits — so a
  request like *"segment this cohort"* is answered in one call, no training at all.
- **Mistakes caught before the compute bill.** `validate_config_semantics` builds the
  workflow on a side-effect-free snapshot, so broken YAML, bad dataset mappings, and
  wiring errors surface *before* a multi-hour job — and a prediction config missing
  its checkpoint is flagged up front (with the static review attached) instead of
  failing at launch.
- **Every run reproducible by construction.** Each job persists its exact command,
  environment and package versions, launch-time config snapshots, logs, and metrics —
  a methods-grade record, captured automatically rather than reconstructed later.
- **When something breaks, you can tell why.** Jobs run as tracked subprocesses whose
  full traceback reaches the log, so a failed run is diagnosed and retried from the same
  tools — not from a shrug.

## One request, three ways

The decisive idea: **training from scratch is only one option.** When a user arrives
with a dataset and a goal, the `solve_task` prompt frames a three-way decision and the
agent takes the cheapest path that genuinely fits:

1. **Use a published app as-is** — no training. `list_apps` → `describe_app` (judge fit
   from the app's own description and its declared inputs/outputs) → `import_app`, which
   copies the app (config + code + checkpoints) into the session so it runs as a **normal
   experiment**: `run_prediction` with the returned checkpoints. Tune per run by reading
   `list_app_parameters`, then baking `set_parameters` into the imported config.
2. **Fine-tune a published app** — start from an existing model instead of a blank
   slate: `import_app` it, then `run_resume` with `weights_only=True` warm-starts training
   from the app's weights on the user's dataset.
3. **Train from scratch** — author a config and run the train loop.

Both training paths finish at the **same reusable artifact — a KonfAI app bundle**
(`package_app_from_session` packages a from-scratch model), which the agent can
immediately re-`import_app`, share, or snapshot with `export_app`. No submission service
is involved: the bundle is the KonfAI-native deliverable.

Apps resolve from a **local path, a HuggingFace repo, or a remote server**, read from a
layered catalogue (a shipped default, an editable per-workspace file, and the
`KONFAI_MCP_APP_CATALOG` env file — the same `{"apps": [...]}` shape the app server
consumes). Users pin their own sources with `register_app_source`.

```{admonition} Trust model
:class: warning
Resolving a **local or HuggingFace** app imports its Python code and pip-installs its
requirements, so every app execution / fine-tune / parameter-read tool is gated behind
an explicit `allow_untrusted_code=True`. A **remote** app runs on the user's own server
(inputs are uploaded there) and needs no code gate. Only resolve apps you trust.
```

## Running the server

Install the core package and the MCP package, then launch the entrypoint:

```bash
pip install -e ".[dev,imaging]"
pip install -e ./konfai-apps
pip install -e ./konfai-mcp

konfai-mcp            # stdio transport by default
```

Point an MCP client at the `konfai-mcp` command. Example client entry:

```toml
[mcp_servers.konfai]
command = "/path/to/venv/bin/konfai-mcp"
cwd = "/path/to/KonfAI/konfai-mcp"
tool_timeout_sec = 3600

[mcp_servers.konfai.env]
KONFAI_MCP_WORKSPACES_ROOT = "/path/to/workspaces"
KONFAI_MCP_APP_CATALOG = "/path/to/my_apps.json"   # optional: your own app sources
```

## Tool surface at a glance

| Stage | Tools |
|---|---|
| Orient | `describe_konfai_capabilities`, `describe_config_schema`, `describe_extension_points` |
| Dataset | `browse_dataset`, `inspect_dataset`, `read_dataset_file`, `preview_volume`, `prepare_dataset_aliases` |
| Author | `design_config_strategy`, `initialize_session`, `write_workflow_config`, `write_session_file` |
| Validate | `review_config_semantics`, `validate_config_semantics` |
| Run & monitor | `run_train`, `run_prediction`, `run_evaluation`, `wait_for_job`, `read_live_metrics`, `leaderboard` |
| Use an app | `list_apps`, `describe_app`, `list_app_parameters`, `import_app` (→ `run_prediction` with the returned checkpoints) |
| Adapt & package | `import_app` → `run_resume` (`weights_only=True`), `package_app_from_session`, `export_app`, `register_app_source` |

Most payloads include a `next_actions` list so the agent can chain calls without
guessing the next valid step. The full per-tool contract is not duplicated here: it is
generated from the registry (`guide://tool-index` live, `tool-reference.md` in the repo).

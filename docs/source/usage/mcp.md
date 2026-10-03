# MCP workflows (MCP server)

Connect an MCP client to KonfAI so it can inspect datasets, write configurations,
launch jobs and read their results. If you want a ready-made chat interface, use
{doc}`studio`.

`konfai-mcp` implements the Model Context Protocol. The client operates the same YAML
workflows as the CLI, and each run keeps its configuration, logs and metrics.

## Running the server

Install the core package and the MCP package, then launch the entrypoint:

```bash
pip install "konfai[imaging]" konfai-apps konfai-mcp

konfai-mcp            # stdio transport by default
```

From a checkout, to give the client edited code:

```bash
pip install -e ".[dev,imaging]" -e ./konfai-apps -e ./konfai-mcp
```

Point an MCP client at the `konfai-mcp` command. Example client entry:

```toml
[mcp_servers.konfai]
command = "/path/to/venv/bin/konfai-mcp"
tool_timeout_sec = 3600

[mcp_servers.konfai.env]
KONFAI_MCP_WORKSPACES_ROOT = "/path/to/workspaces"
KONFAI_MCP_APP_CATALOG = "/path/to/my_apps.json"   # optional: your own app sources
```

## Try a first request

Ask the connected client to inspect a dataset before running a job:

> Inspect /path/to/my/data and list its cases and image groups. Find a published
> segmentation app that fits these inputs, then explain how to run and check it.

The client can run a published app, fine-tune it or train from a configuration.
Check the completed job status and written outputs before interpreting its metrics;
validation at its default level only builds the objects. Request
`validate_config_semantics(level="train_step")` for a forward and backward pass.

Tools that load a local or Hugging Face app's Python code require
`allow_untrusted_code=True`; app resolution can also install its requirements.
Use a source you trust. `list_apps` and `describe_app` only read metadata. The catalogue can
list remote app servers, but `run_app` executes **local and Hugging Face apps only**.
Use the {doc}`Apps CLI <apps>` to execute an app on a remote app server.

## Host the MCP server remotely

Over the network, use the streamable HTTP transport with a token:

```bash
konfai-mcp --transport streamable-http --host 0.0.0.0 --port 8123 --bearer-token "$TOKEN"
```

A network address without a token is refused. `--stateless-http` (and `--json-response` for buffering
proxies) lets several replicas run behind a load balancer. Workspaces live on disk under
`KONFAI_MCP_WORKSPACES_ROOT`, but the list of running jobs is per process: point replicas at one workspace
folder and send a job's monitoring calls to the process that runs it.

## Tool surface at a glance

| Stage | Tools |
|---|---|
| Orient | `describe_konfai_capabilities`, `describe_config_schema`, `describe_extension_points` |
| Dataset | `browse_dataset`, `inspect_dataset`, `read_dataset_file`, `preview_volume`, `prepare_dataset_aliases` |
| Author | `design_config_strategy`, `initialize_session`, `write_workflow_config`, `write_session_file` |
| Validate | `review_config_semantics`, `validate_config_semantics` |
| Run & monitor | `run_train`, `run_prediction`, `run_evaluation`, `plan_transform` → `run_transform`, `wait_for_job`, `read_live_metrics`, `leaderboard` |
| Use an app | `list_apps`, `describe_app`, `list_app_parameters`, `run_app` (`infer`, `evaluate`, `uncertainty`, `pipeline`), `import_app` (modify-then-run) |
| Adapt & package | `fine_tune_app`, `import_app` → `run_resume` (`weights_only=True`), `package_app_from_session`, `export_app`, `register_app_source` |

Most answers include `next_actions`, the valid next calls. The full reference of each tool is generated from
the server (`guide://tool-index`).

`import_experiment` brings in an experiment made outside the server: configs and code are copied, and
`Checkpoints/`, `Predictions/` and the other outputs are linked (copied where links are not allowed).
`include_artifacts='copy'` always copies, `'none'` imports configs and code only.

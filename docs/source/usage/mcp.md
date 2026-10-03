# Agent workflows (MCP server)

Give an LLM agent a folder of scans and a request in plain language (*"segment these CT volumes"*, *"train
an MR-to-CT model"*, *"register these two scans"*), and it carries it out: it reads the data, writes and
validates the config, runs the workflows, and returns the metrics with a record you can reproduce.

`konfai-mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server that gives the agent
KonfAI as tools. The agent writes the same YAML you would, every run keeps its resolved config, command,
versions, logs and metrics, and configs are validated before a long job starts.

## One request, three ways

Training is only one option. The agent takes the cheapest one that fits:

1. **Use a published app**: `list_apps`, `describe_app`, then `run_app` (`infer`, or `pipeline` to also
   score it). `list_app_parameters` and `set_parameters` tune a run; `import_app` copies an app into the
   session when it must be modified.
2. **Fine-tune an app** on the user's data: `fine_tune_app`.
3. **Train from scratch**: write a config and train.

Both training paths end with an app (`package_app_from_session`), which can be run again with `run_app` or
saved with `export_app`. Apps come from a local folder, Hugging Face or a remote server, listed in a
catalogue the user extends with `register_app_source` or `KONFAI_MCP_APP_CATALOG`.

```{admonition} Trust model
:class: warning
Running a local or Hugging Face app runs its Python code and installs its requirements, so these tools need
`allow_untrusted_code=True`. A remote app runs on its own server and needs no such flag. Only use apps you
trust.
```

## Running the server

Install the core package and the MCP package, then launch the entrypoint:

```bash
pip install "konfai[imaging]" konfai-apps konfai-mcp

konfai-mcp            # stdio transport by default
```

From a checkout, to give the agent edited code:

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

### Remote and stateless deployments

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

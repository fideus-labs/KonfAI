# Use KonfAI Studio

KonfAI Studio is a chat interface for inspecting datasets, running KonfAI workflows
and viewing their outputs. It uses the same tools as the {doc}`MCP server <mcp>`.
You supply the language model or connect an account you already use.

## Install and open Studio

```bash
python -m pip install konfai-studio
konfai-studio
```

Open `http://127.0.0.1:8730`. The package includes the web interface.
Choose a model backend before starting a conversation.

## Connect a language model

| Backend | What you need | Configuration |
| --- | --- | --- |
| Claude Code, the default | Claude CLI installed and logged in | No API key needed |
| Claude API | The `anthropic` extra and your API credentials | `KONFAI_STUDIO_LLM=anthropic` |
| Local or other OpenAI-compatible endpoint | The `openai` extra and a running model server | `KONFAI_STUDIO_LLM=openai` |

For the Claude API, install the extra, then set your key in the environment or in
Studio's LLM panel:

```bash
python -m pip install "konfai-studio[anthropic]"
```

For an existing local model server, install the other extra:

```bash
python -m pip install "konfai-studio[openai]"
```

In the LLM panel, select the compatible backend, enter its API base URL (for example
`http://localhost:11434/v1` for a local Ollama server) and the name of a model that
server has installed. For setup through the environment, select the backend with
`KONFAI_STUDIO_LLM=anthropic` or `KONFAI_STUDIO_LLM=openai` and set
`KONFAI_STUDIO_MODEL`. The Claude API uses `ANTHROPIC_API_KEY`. An OpenAI-compatible
endpoint uses `KONFAI_STUDIO_LLM_BASE_URL` and, when required,
`KONFAI_STUDIO_LLM_API_KEY`.

Settings entered in the panel are saved in `.konfai_studio/credentials.json` under
the workspace root (`~/KonfAI_Workspaces` by default). Environment variables take
precedence. On Linux and macOS the file is restricted to its owner; on Windows it
inherits the workspace's permissions.

## Know what is sent to the model

KonfAI jobs run on the machine hosting Studio. Your chat messages and tool results
are sent to the selected language-model endpoint; those results can include dataset
paths, metadata and measurements. A hosted model therefore receives part of the
conversation and workflow context.

Use a local model endpoint when those exchanges must stay on your machine. Model,
app and dependency downloads can still contact external services; a local LLM alone
does not make every workflow offline.

## Start with a dataset

For example, ask:

> Inspect the dataset in /path/to/my/data. Tell me which image groups and cases it
> contains, then propose a segmentation workflow.

Review the proposed inputs and configuration, then ask Studio to run it. During a
job, inspect the logs, curves and output samples. After prediction, open the written
images and ask for evaluation against your reference labels. Keep the resolved
configuration and metrics with the run.

```{figure} ../_static/konfai-studio.png
:alt: Studio showing a chat beside training curves and model-output samples.
:width: 100%

A synthesis session with live training results. Connectivity depends on the selected
model endpoint and tools, regardless of the status text in this screenshot.
```

## Access Studio from another machine

Studio binds to loopback by default. A non-loopback address requires
`KONFAI_STUDIO_TOKEN`; use a TLS reverse proxy for network access. See the
[single-operator deployment guide](https://github.com/fideus-labs/KonfAI/blob/main/konfai-studio/docs/REMOTE.md)
for authentication, proxy configuration and service setup.

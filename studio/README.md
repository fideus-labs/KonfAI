# KonfAI Studio

![KonfAI Studio: the agent authors an MR→CT synthesis run (write_workflow_config → validate → run_train) while the live training feed, multi-run loss curves, and model-output samples stream beside the chat — all local, offline, nothing leaves the machine.](docs/screenshot.png)

A single **chatbot** (à la ChatGPT / Claude, specialized for medical imaging) that drives
`konfai-mcp` end to end. A clinician-researcher points it at their own dataset and — from the
conversation alone — onboards data, authors/reuses a model, trains, infers, visualizes results,
compares, keeps & reproduces experiments, then deploys the frozen model privately (on-prem or
100% in the browser). The compute stays on the user's site; nothing is uploaded to a third party.

**This is a product surface, not a new engine.** Every capability maps 1:1 onto an existing
`konfai-mcp` tool (56 today). The build is the web UI + a thin bridge (BFF), plus the ONNX export.

## Layout

- `konfai_studio/` — the Python package (the BFF)
  - `server.py` — FastAPI: streams the chat over SSE, serves the front, streams volumes to NiiVue
  - `agent.py` — the pluggable brain (`KONFAI_STUDIO_LLM`): `claude-code` (Claude Agent SDK, default),
    `openai` (local vLLM/Ollama or any OpenAI-compatible endpoint), `anthropic` (Claude API)
  - `web/` — the built front (`index.html` + `assets/`, git-ignored; logos are committed)
- `frontend/` — the React + Vite source (chat panel + NiiVue viewer; `npm run build` emits into `web/`)
- `docs/` — the spec, the build plan, and the strategy briefs

## Run

```bash
pip install -e ./studio                 # deps: fastapi, uvicorn, fastmcp, claude-agent-sdk
npm --prefix studio/frontend install    # once
npm --prefix studio/frontend run build  # builds the front into konfai_studio/web/
konfai-studio                           # -> http://127.0.0.1:8730
```

The default brain uses your **Claude Code subscription** (no API key). For a local model:
`KONFAI_STUDIO_LLM=openai KONFAI_STUDIO_LLM_BASE_URL=http://localhost:11434/v1 KONFAI_STUDIO_MODEL=qwen2.5:14b konfai-studio`.
Front hot-reload during development: `npm --prefix studio/frontend run dev` (proxies to the BFF).

See [`docs/STUDIO_SPEC.md`](docs/STUDIO_SPEC.md) and [`docs/PLAN.md`](docs/PLAN.md).

# KonfAI Studio

![KonfAI Studio: the agent authors an MR→CT synthesis run (write_workflow_config → validate → run_train) while the live training feed, multi-run loss curves, and model-output samples stream beside the chat, all local, offline, nothing leaves the machine.](docs/screenshot.png)

A single **chatbot** (à la ChatGPT / Claude, specialized for medical imaging) that drives
`konfai-mcp` end to end. A clinician-researcher points it at their own dataset and, from the
conversation alone, onboards data, authors/reuses a model, trains, infers, visualizes results,
compares, keeps & reproduces experiments, then deploys the frozen model privately (on-prem or
100% in the browser). The compute stays on the user's site; nothing is uploaded to a third party.

**This is a product surface, not a new engine.** Every capability maps 1:1 onto an existing
`konfai-mcp` tool (56 today). The build is the web UI + a thin bridge (BFF), plus the ONNX export.

## See it work

Five real sessions on real data. Each video is one continuous take, sped up but never cut.

### Train from your dataset

Point Studio at your images and ask in plain words. The agent inspects the dataset, writes and
validates the config, and asks before spending GPU time. Training runs with live curves and
validation samples, then the held-out cases are predicted, evaluated, and opened in the viewer
next to the ground truth.

<!-- drop clip1_train_from_your_dataset.mp4 here -->

### Iterate and steer, live

Launch one more run and steer it while it trains. Change the learning rate mid-run, stop the
job when you decide, evaluate the same held-out cases, and compare every run on a leaderboard
that flags optimistic numbers.

<!-- drop clip2_iterate_steer_compare.mp4 here -->

### Use a published model

Pick an app from the zoo and run its published weights on your own cases. GPU inference takes
seconds per case, the result is scored against your reference mask, and the segmentation opens
over its CT in the viewer.

<!-- drop clip3_use_a_published_model.mp4 here -->

### Fit a challenge task

How KonfAI is used in challenge season. The agent designs a loss shaped for the task's score,
writes it as a real PyTorch module, smoke tests it before any GPU hour, and plugs it into the
config by classpath. It trains under a time budget, scores the result the way the challenge
does, and claims nothing without a fair baseline.

<!-- drop clip4_fit_a_challenge_task.mp4 here -->

### Prepare a whole dataset

The TRANSFORM workflow processes 75 volumes under a 512 MiB memory cap. The plan states, case
by case, what will stream in slabs and what simply fits in memory. The whole dataset goes
through in about a minute and comes back verified.

<!-- drop clip5_prepare_a_dataset.mp4 here -->

## Layout

- `konfai_studio/`: the Python package (the BFF)
  - `server.py`: FastAPI that streams the chat over SSE, serves the front, streams volumes to NiiVue
  - `agent.py`: the pluggable brain (`KONFAI_STUDIO_LLM`): `claude-code` (Claude Agent SDK, default),
    `openai` (local vLLM/Ollama or any OpenAI-compatible endpoint), `anthropic` (Claude API)
  - `web/`: the built front (`index.html` + `assets/`, git-ignored; logos are committed)
- `frontend/`: the React + Vite source (chat panel + NiiVue viewer; `npm run build` emits into `web/`)
- `docs/`: the spec and the remote-deployment guide

## Run

```bash
pip install konfai-studio
konfai-studio                           # -> http://127.0.0.1:8730
```

The published wheel already ships the built front. The default brain uses your
**Claude Code subscription** (no API key); `konfai-studio[openai]` and
`konfai-studio[anthropic]` add the alternatives. For a local model:
`KONFAI_STUDIO_LLM=openai KONFAI_STUDIO_LLM_BASE_URL=http://localhost:11434/v1 KONFAI_STUDIO_MODEL=qwen2.5:14b konfai-studio`.

## Develop from a checkout

The front is git-ignored, so a checkout has to build it. `konfai-mcp` comes first:
Studio pins it to its own setuptools_scm version, which only exists on PyPI at a
release tag.

```bash
pip install -e ./konfai-mcp             # must precede studio: the pin is version-exact
pip install -e ./studio                 # deps: fastapi, uvicorn, fastmcp, claude-agent-sdk
npm --prefix studio/frontend install    # once
npm --prefix studio/frontend run build  # builds the front into konfai_studio/web/
```

Front hot-reload during development: `npm --prefix studio/frontend run dev` (proxies to the BFF).

See [`docs/STUDIO_SPEC.md`](docs/STUDIO_SPEC.md) for the design, and
[`docs/REMOTE.md`](docs/REMOTE.md) to serve it beyond loopback.

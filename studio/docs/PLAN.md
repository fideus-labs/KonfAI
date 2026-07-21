# KonfAI Studio — Actionable plan

Companion to [`STUDIO_SPEC.md`](STUDIO_SPEC.md). What we build, in order, with concrete tasks.

## Principles

- **Product surface over `konfai-mcp`** — no new *core* code for M0/M1 (only the ONNX export touches
  the Python core, on its own branch).
- **Beautiful & professional is a requirement, not a finishing touch.** A design system from day one
  (tokens, type scale, component kit); every screen designed in light *and* dark; the live console,
  the curves and the viewer get the same craft as the chat. Target: looks like a professional
  medical-AI product, not a dev tool.
- **Local-first; the LLM brain is pluggable; the imaging data never reaches the LLM.**

## Decisions locked

- **LLM backend**: Claude API as default (best tool-use); local-model + custom endpoint as options. A
  setting, not hardcoded.
- **Positioning**: demonstrator-first (optimize the M1 video), product-ready after.
- **Packaging**: a separate `konfai-studio` package (`bff` + `web`) depending on `konfai-mcp`.

## Workstreams (run in parallel)

**A. BFF** (FastAPI) · **B. Web** (React + Vite) · **C. Design system** (continuous) · **D. ONNX export** (on `feat/konfai-onnx-export`)

---

## M0 — The bridge  ·  ✅ DONE + verified end-to-end (2026-07-19)

- [x] `konfai_studio` FastAPI package (`studio/`, own `pyproject.toml`, depends on konfai-mcp).
- [x] **Pluggable brain** (`KONFAI_STUDIO_LLM`), three backends already in place:
  - `claude-code` *(default)* — Claude Agent SDK, driven by the user's **Claude Code subscription** (no API key, no per-token bill); the SDK spawns konfai-mcp itself, `bypassPermissions` + built-in mutators disallowed.
  - `openai` — any OpenAI-compatible server: **local** (Ollama/vLLM/LM Studio) **or remote**, via `KONFAI_STUDIO_LLM_BASE_URL`/`_API_KEY`.
  - `anthropic` — Claude API (metered key or OAuth token).
  - `StudioAgent` (anthropic/openai) holds one FastMCP `Client` over stdio → **56 tools loaded, verified**.
- [x] Agentic loop `POST /api/chat` streaming (SSE): text + tool_call/tool_result/done/error events.
- [x] Minimal streaming-chat front (`web/index.html`, KonfAI identity, tool-event cards); served by `konfai-studio` on localhost.
- [x] **Real round-trip verified on the Claude Code Pro plan:** "list the available apps" → model calls `list_apps` → real catalogue returned → text answer. tool_call ✅ tool_result ✅ text ✅.
- [ ] Deferred to M1: `WS /session/{id}/live`, `GET /files/volume|slice`, React+Vite front, **the brain becomes a UI setting** (not an env var), and disable Claude Code tool-deferral so the model doesn't waste a turn on `ToolSearch` before reaching the MCP tools.

## M1 — Demo-able core  ·  ~1–2 w

- [ ] **Spike (do first):** how a config enables TB image logging (per-output flag) → the agent auto-enables it when authoring.
- [ ] Generative UI: tool-kind → React component registry.
- [ ] **Dataset**: `inspect_dataset` summary card + NiiVue viewer (`/files/volume`) + thumbnails (`/files/slice`).
- [ ] **Live run card** (bound to `job_id`):
  - [ ] Metrics — watcher tails the log → `read_live_metrics` → WS → uPlot (curve grows live).
  - [ ] **Live console** — direct tail of the job console log → WS → virtualized terminal pane (the "logs en direct").
  - [ ] Val-image strip — read TB image summaries from `Statistics/<name>/tb` → WS → overlay refreshes ("watch it learn").
  - [ ] Status/progress — watch `job.json` + parse tqdm → status pill + progress bar; cancel (`cancel_job`).
- [ ] **Results**: on `run_prediction`/`run_evaluation` done → NiiVue overlay (pred vs GT) + metric table (`get_run_metrics`) + `compare_runs` mini-panel + `leaderboard`.
- [ ] Reconcile-on-load from `summarize_session` + `list_jobs` + `read_training_curves`.
- [ ] Design polish pass on every M1 screen (both themes, spacing, motion, reduced-motion safe).
- ✅ **Done when:** the video — prompt → dataset → train (live curve + live console + val image improving) → predict → overlay → metrics.

## Phase 1 — ONNX export  ·  ~3–5 d  ·  `feat/konfai-onnx-export` (parallel)

- [ ] `konfai/export/onnx.py` — `export_to_onnx` (dynamo, select head, opset ≥ 18, fold pre/post, self-contained onnx).
- [ ] `EXPORT` CLI state in `konfai/main.py` / `konfai export`.
- [ ] `onnx` / `onnxruntime` export extra in `pyproject.toml` (same commit).
- [ ] Parity test (onnxruntime vs torch). → emits the `model.onnx + manifest.json` contract both runtimes consume.

## M3 — In-tab deployment  ·  ~2–3 w

- [ ] Wire `konfai-web` (ORT-Web / WebGPU) into a Deploy view: load bundle → infer on an uploaded volume → NiiVue renders result + overlay.
- [ ] Volume tiling + geometry-aware NIfTI/MHA I/O (deterministic-order port of `konfai/data/patching.py`).
- [ ] Unsupported-browser fallback; full-volume parity gate vs the Python `Predictor`.

---

## Cross-cutting — Design system (workstream C, continuous)

- Tokens (color / type / space / radius / elevation), both themes; a component kit: app shell, chat
  bubbles, generative run card, **live console**, live chart, viewer chrome (overlay opacity + axis
  tabs), status pills, experiment rail; motion guidelines.
- Ship a Storybook-style catalog so screens compose from one system.

## Risks / open items

- **TB image-logging enablement** (M1 spike) — gates the "watch it learn" moment.
- Big-volume transfer to NiiVue → HTTP range + a downsampled proxy for first paint.
- LLM tool-use reliability for from-scratch authoring → mitigate with `validate_config_semantics(level='train_step')` before spending GPU.
- Deploy scope: 3D registration stays Python (Burn `grid_sample` is 2D).

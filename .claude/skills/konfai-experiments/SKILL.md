---
name: konfai-experiments
description: >-
  Drive end-to-end KonfAI deep-learning experiments (medical imaging — segmentation,
  synthesis, registration) through the konfai-mcp MCP server: inspect a dataset, author or
  adapt KonfAI YAML configs, validate them, launch and monitor train / prediction /
  evaluation jobs, then compare runs and iterate. Use when the user wants to train, predict,
  or evaluate a KonfAI model, onboard a dataset for KonfAI, author or debug a KonfAI config
  (invalid root, missing checkpoint, dataset mismatch), read live training metrics, or build
  a leaderboard — and whenever the konfai-mcp / mcp__konfai__* tools are in play. Triggers:
  "train a KonfAI model", "run a segmentation/synthesis experiment", "inspect this dataset
  for KonfAI", "validate my config", "evaluate the run", "leaderboard", "why did the job fail".
---

# Driving KonfAI experiments via konfai-mcp

The `konfai-mcp` server exposes KonfAI as a deterministic backend for the full
experimentation loop (the exact tool list lives in
[references/tool-reference.md](references/tool-reference.md), generated from the registry). Your job is to sequence them correctly, keep
runs reproducible, and recover cleanly from failures. The tools do the work — you supply the
ordering and judgment.

## Precondition

These tools are provided by the `konfai-mcp` MCP server and appear namespaced as
`mcp__konfai__<tool>` (e.g. `mcp__konfai__inspect_dataset`). If they are absent, the server
is not wired into this client — run `scripts/check_setup.py` to distinguish "not installed"
from "installed but not wired", then see [references/resources-and-clients.md](references/resources-and-clients.md).
(The package lives only on the private `konfai-mcp` git branch.)

## The canonical loop

This is the tool order verified by the segmentation and synthesis end-to-end tests. Skip
the discovery steps only when the dataset and task are already understood.

**Route first (cheapest fit wins)**
0. `list_apps` → `describe_app` — when the user wants a RESULT, check whether a published app already solves it (`run_app_infer`), or is a close starting point (`fine_tune_app`), BEFORE authoring and training from scratch. `run_resume` continues an interrupted session training.

**Discover (dataset-driven)**
1. `browse_dataset` → `inspect_dataset` — choose the real dataset root, see groups + sampled stats (`include_stats=False` for a fast structural peek; `groups=[...]` when you need intensity ranges for normalization).
2. `prepare_dataset_aliases` — fix the group→role mapping the config needs (e.g. `{"IMG": "CT"}`). Non-destructive `copy` mode; `move` needs `allow_destructive=True`.
3. `design_config_strategy` — turn **task + group_roles + example** into a config plan. It will **not** guess the task; pass all three.

**Author**
4. `initialize_session(from_example=..., workflows=[...])` — create the sandboxed workspace, seeding configs from a template. Everything else resolves paths against this session.
5. If a config will reference a **local** `Model:X` / `Loss:X`, `write_session_file` that `.py` **before** validating or running — otherwise validation blocks on a missing local component.
6. Resolve every object name before writing it: `list_components` (what exists) → `inspect_object_signature` / `describe_config_schema` (how to configure it).
7. `write_workflow_config(workflow, content)` for each of `train` / `prediction` / `evaluation`. The root key (`Trainer`/`Predictor`/`Evaluator`) is validated on write.

**Validate — always, before every run**
8. `review_config_semantics` — cheap static check; clear all `blocking_issues`.
9. `validate_config_semantics(workflow, level="instantiate")` — instantiates KonfAI objects to catch runtime errors. **Side-effect-free on your authored config** (validates on a snapshot, restores your file). Use `workflow="all"` to check every config at once.

**Run — destructive; commits real compute (get human OK first, see below)**
10. `run_train` → then `wait_for_job` (**omit `timeout_s`** for real multi-hour runs) and/or `read_live_metrics` to watch progress.
11. `run_prediction` → `wait_for_job`. Needs a checkpoint from a finished train; it does not look outside the session.
12. `run_evaluation` → `wait_for_job`. Needs prediction artifacts to exist first.

**Compare & iterate**
13. `summarize_session` — one compact snapshot (readiness, latest job, metrics, next actions).
14. `leaderboard(metric=...)` — rank completed runs from the `Metric_<split>.json` files. Then rewrite a config, re-validate, launch a new run, compare.

Most payloads carry a `next_actions` list — follow it instead of guessing the next call.

## Rules that keep experiments correct

- **Validate before you run.** `run_train` / `run_prediction` / `run_evaluation` execute and
  write `Checkpoints/` `Statistics/` `Predictions/` `Evaluations/` — they are destructive, not
  dry runs. `review_config_semantics` → `validate_config_semantics` are the safe pre-checks.
- **Confirm compute-committing and destructive actions with the user first** unless already
  authorized. The server's own `risky_actions_prefer_human_confirmation` list covers the
  `run_*` tools (GPU/compute), destructive dataset operations (`prepare_dataset_aliases(mode="move")`
  / overwrite), executing agent-authored or downloaded Python, accepting an external
  dependency's license, and selecting the final model. `delete_session` and
  `initialize_session(overwrite=True)` are additionally destructive (they `rmtree` the
  workspace) and warrant the same confirmation.
- **One active job per session.** A second launch while one is queued/running raises. Serialize
  and `wait_for_job` first (or use a separate session).
- **A `wait_for_job` timeout is not a failure** — it raises `TimeoutError`; re-poll or raise the
  timeout. For real training, pass no `timeout_s` at all.
- **On any non-`done` terminal status** (`error` / `killed`), read `job://<id>/log` for the
  cause and `job://<id>/manifest` for the exact config snapshot, then `validate_config_semantics`
  → fix → retry. A server restart reports previously-running jobs as `error`/`recovered` — relaunch them.
- **Reading a config mutates it.** KonfAI writes resolved defaults back to disk on load — so
  after a run the on-disk YAML is the fully-resolved snapshot. `validate_config_semantics` is the
  deliberate exception (snapshot + restore).
- **Trust model.** Validation and runs **import the workspace's `.py` files** and can pip-install
  dependencies. Only validate/run configs and code you trust. Vet external libraries with
  `check_external_dependency` before referencing `package.module:Class` in YAML.

## Reference material (load on demand)

- [references/tool-reference.md](references/tool-reference.md) — every tool/prompt/resource, generated from the registry (`python konfai-mcp/scripts/generate_tool_reference.py`).
- [references/config-authoring.md](references/config-authoring.md) — writing KonfAI YAML: the three files, root keys, classpath resolution, conventions.
- [references/troubleshooting.md](references/troubleshooting.md) — symptom → tool recovery map, job lifecycle, validation quirks.
- [references/resources-and-clients.md](references/resources-and-clients.md) — resources, env vars, and wiring the server into Claude Code / Codex.

## Not in scope

This drives KonfAI-shaped workflows. It does not infer arbitrary paper architectures, and
non-standard research models still need a custom `.py` (added via `write_session_file`) or manual
template adaptation. For KonfAI framework internals and conventions, `AGENTS.md` is the source of truth.

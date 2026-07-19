---
name: konfai-maintainer
description: >-
  Review and release changes to the KonfAI framework itself (core konfai, konfai-apps,
  konfai-mcp) without breaking its load-bearing invariants. Use when modifying or reviewing
  framework code — patching/reconstruction, image geometry, streaming, the config-by-reflection
  engine, the model/criterion runtime, checkpoints/RESUME, DDP, storage backends — or when
  preparing a release, judging a performance change, or checking public-API/ecosystem
  compatibility (Slicer, HF bundles). Triggers: "review this KonfAI change", "is this safe to
  merge", "will this break geometry/patching/streaming", "prepare a KonfAI release", "did this
  regress performance", "check the wheel ships the model catalog", "does this break the Slicer/apps
  contract". This is the *maintainer* companion to konfai-cli (running workflows) and
  konfai-experiments (MCP-driven experiments).
---

# Maintaining KonfAI

`AGENTS.md` is the source of truth for architecture, invariants, and commands — read it first and do
**not** duplicate it here. This skill adds the **review workflow**, the **watchlist of traps that have
actually bitten**, and the **release gate**. Use it to decide whether a framework change is safe.

## The reflex before editing

1. **Read before editing.** The most central files are dense with invariants: `utils/config.py` (the
   reflection binder — reading a config *mutates the file*), `data/patching.py` +
   `data/data_manager.py` (patch order, loading regimes), `network/network.py` (the model/criterion
   runtime), `predictor.py` (the streamed-write dispatcher). A one-line change here is rarely local.
2. **Identify which invariant the diff touches** using [references/review-checklist.md](references/review-checklist.md)
   — it maps each subsystem to the invariant a change most easily breaks, and lists the confirmed traps.
3. **Keep the diff one logical change**, no unrelated reformats, and honour the conventions in AGENTS.md §8
   (120 cols, SPDX header on new files, `utils/errors` classes, import-guarded optional deps, Conventional
   Commits, **no AI-authorship trailers**, no `--no-verify`).

## What a green suite does *not* prove

The test suite covers the **single-network bundle happy-path** thoroughly and little else. A change can be
green and still break: composite/GAN models, RESUME of a model with nested Networks, augmentation
determinism across epochs, multi-GPU (DDP) train/predict, `overlap > 0` end to end, and the shipped
example `Prediction.yml` files. If your change touches any of those, **add the missing test** — a passing
CI is not coverage. See the confirmed-trap watchlist in the checklist reference.

Two things that look like proof but are not: a green `validate_config_semantics` at default level runs **no
train step** (only `level='train_step'` does a forward+backward), and a byte-identity test on a config that
tiles **without overlap** never exercises the blend.

## Reviewing a performance change

Performance is a correctness concern here (memory decides whether a case even runs). Judge a perf change by
[references/perf-and-release.md](references/perf-and-release.md), never by wall-clock alone:

- **Byte-identity first.** A/B the new path against `KONFAI_STREAMED_WRITES=0` (or the relevant kill-switch)
  and confirm 0 mismatched voxels + identical geometry *before* looking at speed. A faster wrong answer is a
  regression.
- **Measure memory, not just time.** Peak host RSS (process tree) and peak VRAM (`nvml`), sampled; the wins
  are configuration-specific (large-Z, accumulator-bound, modest channels) and the code should stay honest
  about when a case is *not* window-bounded.
- **Watch for self-inflicted slowdowns** the audit already found: a device/placement gate keyed on a
  process-lifetime peak, or per-patch work that should be per-case.

## Preparing a release

Versions are **tag-derived** (`setuptools_scm`) for all three packages — never hand-edit a version. Before
tagging, run the gate in [references/perf-and-release.md](references/perf-and-release.md). Its non-negotiable
step: **the wheel is what users get, so test the wheel, not the source tree.** An editable install hides
PEP 420 (`konfai/models/python` has no `__init__.py`) and `package-data` (the 14 catalog `.yml`) breakage.
Run:

```bash
python scripts/check_release_ready.py        # from the skill dir; builds+installs the wheel in a clean venv
```

It asserts a clean-venv `import konfai`, the CLI entry point, that `default|UNet.yml` resolves, that
`konfai.models.python.*` imports, that the wheel ships ≥16 `models/python` files + 14 catalog `.yml`, and
that no hyphenated sibling (`konfai-apps`) leaked in.

## Public-API / ecosystem compatibility

Some symbols are consumed by **external** repos and break silently if renamed:

- **SlicerKonfAI / SlicerImpactReg** import specific `konfai_apps.app_repository` symbols (incl.
  `current_free_vram`, `get_parameters`) and the 5 top-level `konfai/__init__` helpers. The contract tests
  (`konfai-apps/tests/test_slicer_*`, `tests/unit/test_slicer_core_api_contract.py`) lock them — if you must
  change one, grep the Slicer checkout first (a dropped `current_free_vram` broke Slicer once).
- **HF bundles** use bundle-relative classpaths; validate a bundle by loading its config on a **copy**
  (reading mutates).

## Reference material (load on demand)

- [references/review-checklist.md](references/review-checklist.md) — per-subsystem invariant + the confirmed-trap watchlist.
- [references/perf-and-release.md](references/perf-and-release.md) — the A/B perf protocol and the release gate steps.
- [references/bundles-and-publishing.md](references/bundles-and-publishing.md) — assemble → validate → HF upload → keep Slicer working (no external repo cloned or hard-coded).
- `scripts/check_release_ready.py` — deterministic clean-wheel install + catalog-resolution check.
- `scripts/check_env_doc_parity.py` — flags `KONFAI_*` env vars used in code but missing from the docs.

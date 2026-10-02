# Performance review + release gate

## A/B protocol for a performance change

Never judge a KonfAI perf change by wall-clock alone: memory decides whether a case runs at all, and a
faster path that changes a voxel is a regression.

1. **Pin both sides.** Hold the app bundle, the input, and everything except the code under test identical.
   For core changes, pin `konfai` to two worktrees (main vs the branch) so a checkout switch can't perturb
   the run; propagate the pin through `mp.spawn` workers (env-gated `sitecustomize`).
2. **Byte-identity first.** A/B against the relevant kill-switch and require **0 mismatched voxels + identical
   geometry** before reading any timing:
   - `KONFAI_STREAMED_WRITES=0`: streamed vs whole-volume writer (the general control).
   - Confirm the path actually streamed via the writer fingerprint (a streamed MetaImage sink omits
     `CenterOfRotation`): this defeats the "silently disabled → trivially equal" false pass.
3. **Measure memory.** Peak host RSS over the process tree and peak VRAM (`nvml` `memory.used` over baseline,
   ~50 ms sampling). Report with the case shape, patch size, overlap, TTA, ensemble size, and cache/stream
   mode: a memory number without those is not reproducible.
4. **Interpret honestly.** Streaming wins only when the volume is large along Z, the accumulator (not the
   forward) is the VRAM bottleneck, and the channel count is modest. Say when a config lands on a
   non-window-bounded path (whole-volume / buffered / host-accumulate): do not claim a universal win.
5. **Watch for self-inflicted regressions.** A device/placement gate keyed on a *process-lifetime* CUDA peak
   demotes every case after the first big one to CPU; per-patch work that belongs per-case; a resident
   footprint that grows and pushes cases off the GPU-accumulate gate (which works *against* streaming).

Reference bench data and harness notes live in `.audit-local/` (git-ignored) from prior campaigns.

## Release gate

Versions are tag-derived (`setuptools_scm`, `^v(?P<version>.*)$`) for every package: **never hand-edit a
version**. Pushing a `v*` tag runs `.github/workflows/publish.yml`: test (core+apps+mcp+studio) → build the
9-package matrix (konfai, konfai-apps, konfai-mcp, konfai-studio and the 5 `apps/*` bundles) → publish via
OIDC → build Docker once konfai is on PyPI. The `apps/*` bundles pin `konfai==` and `konfai-apps==`, and
konfai-studio pins `konfai-mcp==`, all the same version, so the matrix releases in lockstep (AGENTS.md §6b).

Before tagging:

1. `pixi run check` green (lint + format-check + core + apps; the core tests include the wheel tests of step 4).
2. Both sibling suites green: `pip install -e ./konfai-mcp && pixi run --environment dev python -m pytest konfai-mcp/tests`,
   and the apps suite.
3. `pixi run --environment dev typecheck` clean.
4. **`pytest tests/unit/test_packaging.py -m slow`**: tests the *wheel* (a clean non-editable install),
   which is what users receive, where the rest of the suite tests the source tree. It builds the konfai wheel
   (or takes the one `KONFAI_WHEEL` names), checks it ships every `models/python` file and catalog `.yml` of
   the tree and no sibling package, installs it in a fresh venv, and checks that `konfai` and
   `konfai.models.python` import from there and `konfai --version` answers. This catches the PEP 420 /
   `package-data` breakage an editable install hides; CI's build job runs it on the wheel it built.
5. If you touched a public symbol, grep the SlicerKonfAI / SlicerImpactReg checkouts and confirm the contract
   tests still pass (see SKILL.md → ecosystem compatibility).
6. HF bundle staging: confirm the published bundles carry the checkpoints they reference before the tag.

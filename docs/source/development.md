# Development and contributing

This is the single contributor page: environment setup, the test and lint
gates, commit and PR rules, a map of the repository, and the guidelines for
examples and documentation. Read it before making your first change to KonfAI
or its docs. The recommended path uses [Pixi](https://pixi.sh), which manages
Python packages, system libraries, and task runners in a single reproducible
environment.

**All commands on this page assume your working directory is the root of a
cloned KonfAI checkout.**

## Prerequisites

- **Python 3.11 or later**: the minimum version declared in `pyproject.toml`
- **Pixi**: install once with:

  ```bash
  curl -fsSL https://pixi.sh/install.sh | bash
  ```

  See [pixi.sh](https://pixi.sh) for alternative installers.
- **git**

## Clone and install

```bash
git clone https://github.com/fideus-labs/KonfAI.git
cd KonfAI
pixi install       # resolves and installs all Pixi environments
```

`pixi install` creates isolated environments under `.pixi/` and does **not**
touch your system Python or any other virtual environment.

## Repository map

Where each part of the codebase lives:

| Package | Responsibility |
| --- | --- |
| `konfai.main` | CLI entrypoints for low-level workflows and cluster mode |
| `konfai.trainer` | Training workflow and training loop |
| `konfai.predictor` | Prediction workflow and export logic |
| `konfai.evaluator` | Evaluation workflow and metric export |
| `konfai.data` | Dataset discovery, transforms, augmentations, and patching |
| `konfai.network` | Model graph composition, optimizer/scheduler loaders, criterion routing |
| `konfai.metric` | Metrics, losses, and schedulers |
| `konfai.utils` | Config system, dataset helpers, distributed runtime utilities |
| `konfai_apps` | Standalone package (in `konfai-apps/`) for local/remote app execution and the app server |
| `konfai_mcp` | Standalone package (in `konfai-mcp/`) exposing KonfAI workflows and Apps to LLM agents via a FastMCP server |

```{note}
`konfai_apps` and `konfai_mcp` each live in their own directory with their own
`pyproject.toml`, dependencies, and tests, they are installed and tested
separately from the core package (see below).
```

## Available tasks

Run tasks with `pixi run <task>`:

| Task | Command | Description |
| --- | --- | --- |
| `test` | `pytest -q tests/` | Run the full test suite (about 6 min) |
| `test-fast` | `pytest -q -m "not slow and not integration" tests/` | The iteration loop: skips the slow oracle and integration tests (about 1 min 40) |
| `test-cov` | `pytest --cov=konfai tests/` | Run tests with coverage report |
| `lint` | `ruff check konfai konfai-apps/konfai_apps` | Lint the source tree |
| `format` | `ruff format konfai konfai-apps/konfai_apps` | Auto-format source files |
| `format-check` | `ruff format --check ...` | Check formatting without modifying files |
| `typecheck` | `mypy konfai --ignore-missing-imports` | Static type checking |
| `build` | `python -m build` | Build sdist and wheel |
| `test-apps` | `pytest -q konfai-apps/tests` | Run the konfai-apps test suite |
| `check` | lint + format-check + test + test-apps | Full pre-push gate; run it once before finishing any change (needs konfai-apps installed) |

Always run `pixi run check` before pushing or opening a PR.

## pip fallback

If Pixi is unavailable, use an editable pip install:

```bash
pip install -e ".[dev]"
pytest -q tests/
ruff check konfai
ruff format konfai
```

## Pre-commit hooks

The repository ships a `.pre-commit-config.yaml` with both source-file checks and commit-message validation. Install
both hook types once:

```bash
# with Pixi:
pixi run pre-commit-install

# or with pip:
python -m pip install pre-commit
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

After installation, `git commit` runs file checks plus Conventional Commit and forbidden-branding validation. Run all
file checks manually with:

```bash
pre-commit run --all-files
```

## Branches, commits, and pull requests

Never commit directly to `main`. Create a focused feature branch for every change:

```bash
git switch -c fix/short-description
```

Use a Conventional Commit message such as `fix(config): improve YAML validation errors`. Commit messages must not
contain agent names, generated-by/generated-with branding, or AI co-author trailers. The `commit-msg` hooks validate
both the Conventional Commit structure and forbidden branding.

Before pushing, run `pixi run format`, `pixi run check`, and `pre-commit run --all-files`. Push the feature branch,
open a pull request, and leave it open for a maintainer to review and merge; do not merge your own PR.

## Writing and running tests

Tests live under `tests/unit/`. Follow the conventions already established
there:

- one file per module under test (e.g. `tests/unit/test_config.py`)
- use `pytest` fixtures and `monkeypatch` for environment variables
- never import `SimpleITK` or `h5py` unconditionally: guard with `pytest.importorskip`

Run a single test file:

```bash
pixi run test -- tests/unit/test_config.py -v
```

### What CI runs

The GitHub Actions workflow in `.github/workflows/konfai_ci.yml` runs `pytest`
across Python `3.11` to `3.13` on Linux, macOS, and Windows.

### The konfai-apps test suite

The `konfai-apps` package carries its own tests, including an integration test
for the `konfai-apps pipeline` flow in
`konfai-apps/tests/integration/test_konfai_apps.py`, and they are **not** part
of `pixi run test`. Install the package first, then run its suite:

```bash
pip install -e ./konfai-apps
pytest konfai-apps/tests
```

### The konfai-mcp test suite

`konfai-mcp` is a separate package too, with its own suite (and its own CI). It
is likewise **not** part of `pixi run test`:

```bash
pip install -e ".[imaging]" -e ./konfai-mcp
pytest konfai-mcp/tests
```

The segmentation end-to-end test needs the imaging extra (`konfai[imaging]`),
installed above alongside the package.

### Validate an example manually

Some changes are best validated end-to-end against a shipped example. The most
practical manual validation loop is:

1. run a shipped example
2. inspect `Checkpoints/`, `Predictions/`, `Evaluations/`, and `Statistics/`
3. confirm that the generated config copy matches the intended run

## Working on examples

Examples in `examples/` are part of the user-facing documentation of the
framework. When changing example YAML or notebooks:

- keep commands runnable from the example directory
- keep dataset group names and folder layouts explicit
- prefer adapting an existing example over inventing a new undocumented pattern

## Building the documentation

The documentation uses Sphinx with the MyST parser for Markdown files.

Build the HTML output:

```bash
pixi run -e docs build-docs
```

Or in live-reload mode during authoring:

```bash
pixi run -e docs dev-docs
```

Without Pixi:

```bash
pip install -r docs/requirements.txt
make -C docs html
```

`docs/requirements.txt` is the single source for the docs toolchain: it is what
ReadTheDocs and the CI docs job install, and the `[dev]` extra carries the same
list. The `make` route writes to `docs/build/html/`; the live-reload task
(`dev-docs`) serves from `docs/_build/html/`.

### Documentation style

Documentation should stay aligned with the codebase, examples, and tests. When
updating the docs:

- prefer code-backed statements
- call out behavior inferred from code when needed
- avoid documenting private helpers unless they are essential extension points
- update cross-links when you rename or move pages

## Packaging and release

The repository contains a publish workflow in `.github/workflows/publish.yml`
that builds a **9-package matrix**, all sharing a tag-derived version:

- `konfai` (the core framework)
- `konfai-apps`, `konfai-mcp` and `konfai-studio` (the standalone Apps, MCP and
  Studio packages. Studio is wheel-only, and its build job runs `npm ci &&
  npm run build` first because the React front is not in git)
- the five App bundles: `impact-synth-konfai`, `impact-seg-konfai`,
  `mrsegmentator-konfai`, `totalsegmentator-konfai`, `impact-reg-konfai`

The bundles pin `konfai==` and `konfai-apps==` the same version, so the whole
matrix releases in lockstep. A change to the core package can therefore affect
the framework, the two sibling packages, and every published App.

### Cutting a release

Versions are **tag-derived**: `setuptools_scm` reads the tag, so no *package*
version is committed anywhere that could drift from it. `CHANGELOG.md` is drafted
from the commit history and then edited, and the publish workflow takes the committed
section for the tag it is running on **verbatim** as the GitHub Release body, so the
file and the release page cannot describe a version differently. A tag whose section
is missing (or present but empty) fails the job rather than publishing a release
with nothing in it. A tag carrying a pre-release segment (`v1.8.0rc1`, `v1.8.0.dev1`)
publishes as a pre-release and does not take `latest`; a post-release (`v1.8.0.post1`)
is stable and does.

```{note}
No version string is committed anywhere, the Docker image included: it installs the
wheels found in `dist/`, which the release workflow fills from the tag's own build. That
also means the image never waits on PyPI to serve what the run just uploaded.
```

That order matters: **the changelog is written before the tag**, because the
workflow publishes what the file already says.

The generated draft is a starting point, not the answer. It sees commit subjects
only, so a squash merge collapses to one line, a subject with no conventional
prefix is dropped, and a subject written for a reviewer tells a reader nothing.
Take the draft, then say what a *user* of the package gets that they did not have, and re-read it against anything that landed after you drafted it.

```bash
# 1. Draft the section for the version you are about to cut, then edit it
uvx --from commitizen cz changelog --unreleased-version vX.Y.Z --start-rev v1.5.8

# 2. Commit it
git commit -am "ci: changelog for vX.Y.Z"

# 3. Sign, tag and push: this is what triggers the publish workflow
git tag -s vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

Releases are signed. `tag.gpgsign = true` already makes an annotated tag signed
on a machine that has a key configured, so `-s` is spelled out for the machine
that does not: it refuses rather than publishing an unsigned release in silence.

`cz bump --dry-run` is a useful second opinion on whether the commits imply a
major, minor or patch bump. Do not rely on `cz bump` to tag: with
`version_provider = "scm"` there is no version file for it to write, so once the
changelog is current it has nothing to commit and stops without tagging.

`--start-rev v1.5.8` is not a preference. Conventional Commits only took hold at
`v1.5.9`; rendering further back emits version headings with nothing under them,
and everything older is summarised in the changelog's own closing section.

## AI agent rules

If you are an AI agent contributing to this repository, read `AGENTS.md` at
the repository root before making changes. It is the canonical source for branch and PR rules, Conventional Commits,
forbidden commit branding, coding norms, checks, and project-specific pitfalls.

## Next steps

- {doc}`concepts/index`: how the config engine, data pipeline, and model graph fit together before you change them.
- {doc}`examples/index`: the shipped workflows to run when validating a change end-to-end.
- {doc}`reference/api/index`: the curated API surface your extensions and fixes build against.

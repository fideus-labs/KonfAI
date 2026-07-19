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

- **Python 3.10 or later** — the minimum version declared in `pyproject.toml`
- **Pixi** — install once with:

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
`pyproject.toml`, dependencies, and tests — they are installed and tested
separately from the core package (see below).
```

## Available tasks

Run tasks with `pixi run <task>`:

| Task | Command | Description |
| --- | --- | --- |
| `test` | `pytest -q tests/` | Run the full test suite |
| `test-cov` | `pytest --cov=konfai tests/` | Run tests with coverage report |
| `lint` | `ruff check konfai konfai-apps/konfai_apps` | Lint the source tree |
| `format` | `ruff format konfai konfai-apps/konfai_apps` | Auto-format source files |
| `format-check` | `ruff format --check ...` | Check formatting without modifying files |
| `typecheck` | `mypy konfai --ignore-missing-imports` | Static type checking |
| `build` | `python -m build` | Build sdist and wheel |
| `test-apps` | `pytest -q konfai-apps/tests` | Run the konfai-apps test suite |
| `check` | lint + format-check + test + test-apps | Full pre-push gate — run before finishing any change (needs konfai-apps installed) |

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
- never import `SimpleITK` or `h5py` unconditionally — guard with `pytest.importorskip`

Run a single test file:

```bash
pixi run test -- tests/unit/test_config.py -v
```

### What CI runs

The GitHub Actions workflow in `.github/workflows/konfai_ci.yml` runs `pytest`
across Python `3.10` to `3.13` on Linux, macOS, and Windows.

### The konfai-apps test suite

The `konfai-apps` package carries its own tests — including an integration test
for the `konfai-apps pipeline` flow in
`konfai-apps/tests/integration/test_konfai_apps.py` — and they are **not** part
of `pixi run test`. Install the package first, then run its suite:

```bash
pip install -e ./konfai-apps
pytest konfai-apps/tests
```

### The konfai-mcp test suite

`konfai-mcp` is a separate package too, with its own suite (and its own CI). It
is likewise **not** part of `pixi run test`:

```bash
pip install -e ./konfai-mcp
pytest konfai-mcp/tests
```

The segmentation end-to-end test needs the imaging extra (`konfai[imaging]`).

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

The output lands in `docs/_build/html/`.

### Documentation style

Documentation should stay aligned with the codebase, examples, and tests. When
updating the docs:

- prefer code-backed statements
- call out behavior inferred from code when needed
- avoid documenting private helpers unless they are essential extension points
- update cross-links when you rename or move pages

## Packaging and release

The repository contains a publish workflow in `.github/workflows/publish.yml`
that builds an **8-package matrix**, all sharing a tag-derived version:

- `konfai` (the core framework)
- `konfai-apps` and `konfai-mcp` (the standalone Apps and MCP packages)
- the five App bundles: `impact-synth-konfai`, `impact-seg-konfai`,
  `mrsegmentator-konfai`, `totalsegmentator-konfai`, `impact-reg-konfai`

The bundles pin `konfai==` and `konfai-apps==` the same version, so the whole
matrix releases in lockstep. A change to the core package can therefore affect
the framework, the two sibling packages, and every published App.

## AI agent rules

If you are an AI agent contributing to this repository, read `AGENTS.md` at
the repository root before making changes. It is the canonical source for branch and PR rules, Conventional Commits,
forbidden commit branding, coding norms, checks, and project-specific pitfalls.

## Next steps

- {doc}`concepts/index` — how the config engine, data pipeline, and model graph fit together before you change them.
- {doc}`examples/index` — the shipped workflows to run when validating a change end-to-end.
- {doc}`reference/api/index` — the curated API surface your extensions and fixes build against.

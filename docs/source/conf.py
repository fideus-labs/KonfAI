# Copyright (c) 2025 Valentin Boussot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../../"))  # to access konfai/
sys.path.insert(0, os.path.abspath("../../konfai-apps"))  # standalone konfai_apps package

project = "KonfAI"
author = "Valentin Boussot"
copyright = f"{datetime.now().year}, {author}"  # noqa: A001 - required by Sphinx

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx_tabs.tabs",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
html_static_path = ["_static"]
exclude_patterns = ["_build", "build", "Thumbs.db", ".DS_Store"]

html_theme = "shibuya"
html_title = "KonfAI documentation"
html_favicon = os.path.abspath("../../logo.png")
html_css_files = ["custom.css"]
html_js_files = ["mermaid-fit.js"]

# The KonfAI logo is a full wordmark (mark + "KonfAI"). We use transparent-
# background versions of it derived from logo.png: the original mint art for the
# dark header, and a deep-teal recolor for the light header (mint is too light on
# white). shibuya runs pathto(src, 1), so paths are relative to the output root
# (include the _static/ prefix; depth is adjusted per page). The theme's
# duplicate "KonfAI" text label is hidden in custom.css.
html_theme_options = {
    "light_logo": "_static/konfai-logo-light.png",
    "dark_logo": "_static/konfai-logo-dark.png",
    # Persistent header links: the five pages a reader comes back to from anywhere.
    "nav_links": [
        {"title": "Quickstart", "url": "quickstart"},
        {"title": "Config guide", "url": "config_guide/index"},
        {"title": "Components", "url": "reference/components/index"},
        {"title": "CLI", "url": "reference/cli"},
        {"title": "Apps", "url": "usage/apps"},
    ],
}

# Mermaid draws the shapes; the palette comes from _static/custom.css, so a diagram
# follows the light/dark tokens the rest of the site uses.
mermaid_version = "11.12.1"
mermaid_d3_zoom = False

myst_enable_extensions = [
    "deflist",
    "fieldlist",
    "colon_fence",
    "html_admonition",
    "html_image",
]
myst_heading_anchors = 3
suppress_warnings = [
    "sphinx_autodoc_typehints.local_function",
    # Whether a third-party guarded import resolves depends on the environment (torch's
    # tensorboard writer guards dateutil), so it must not fail a -W build.
    "sphinx_autodoc_typehints.guarded_import",
    "intersphinx.external",
]

autodoc_default_options = {
    "members": True,
    "private-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autosummary_generate = True
# SimpleITK is heavy; requests and huggingface_hub are konfai_apps' own dependencies, which the
# ReadTheDocs build (core + docs/requirements.txt only) does not install, while apps.rst autodocs
# konfai_apps from the source tree on sys.path.
autodoc_mock_imports = ["SimpleITK", "requests", "huggingface_hub"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# ---------------------------------------------------------------------------
# llms.txt / llms-full.txt: agent-ingestible copies of the pages an agent needs
# to author a config (quickstart, the config guides, the component catalog).
# Emitted into the HTML output root, so they publish at /llms.txt beside the
# site. llms.txt is the index; llms-full.txt concatenates the page sources.
# ---------------------------------------------------------------------------

_LLMS_BASE_URL = "https://konfai.readthedocs.io/en/latest"

#: (section, source file, published page) in reading order.
_LLMS_PAGES = [
    ("Getting started", "quickstart.rst", "quickstart.html"),
    ("Config guide", "config_guide/index.md", "config_guide/index.html"),
    ("Config guide", "config_guide/training.md", "config_guide/training.html"),
    ("Config guide", "config_guide/prediction.md", "config_guide/prediction.html"),
    ("Config guide", "config_guide/evaluation.md", "config_guide/evaluation.html"),
    ("Config guide", "config_guide/transform.md", "config_guide/transform.html"),
    ("Components", "reference/components/models.md", "reference/components/models.html"),
    ("Components", "reference/components/losses-metrics.md", "reference/components/losses-metrics.html"),
    ("Components", "reference/components/transforms.md", "reference/components/transforms.html"),
    ("Components", "reference/components/storage-backends.md", "reference/components/storage-backends.html"),
    ("Reference", "reference/cli.md", "reference/cli.html"),
]

_LLMS_HEADER = (
    "# KonfAI\n\n"
    "> KonfAI is a declarative deep-learning framework for medical imaging: a model, its data\n"
    "> pipeline, losses/metrics, and the whole train/predict/evaluate/transform workflow are\n"
    "> described in YAML and run by the `konfai` CLI. Configs are complete, reproducible records\n"
    "> of an experiment; volumes are read as patches and never loaded whole on a streamable route.\n"
)


def _write_llms_txt(app, exception):
    if exception is not None or app.builder.name != "html":
        return
    from pathlib import Path as _Path

    srcdir, outdir = _Path(app.srcdir), _Path(app.outdir)
    index_lines = [_LLMS_HEADER]
    full_parts = [_LLMS_HEADER]
    section = None
    for page_section, source, page in _LLMS_PAGES:
        source_path = srcdir / source
        if not source_path.exists():
            continue
        if page_section != section:
            section = page_section
            index_lines.append(f"\n## {section}\n")
        url = f"{_LLMS_BASE_URL}/{page}"
        index_lines.append(f"- [{source}]({url})")
        full_parts.append(f"\n\n---\nSource: {url}\n---\n\n{source_path.read_text(encoding='utf-8')}")
    index_lines.append(f"\n\nFull content: {_LLMS_BASE_URL}/llms-full.txt\n")
    (outdir / "llms.txt").write_text("\n".join(index_lines), encoding="utf-8")
    (outdir / "llms-full.txt").write_text("".join(full_parts), encoding="utf-8")


# ---------------------------------------------------------------------------
# Redirects for pages that were merged into others. The old URL keeps working:
# a stub at the old path forwards to the page that holds the content now.
# Add an entry here whenever a page moves.
# ---------------------------------------------------------------------------

_REDIRECTS = {
    "concepts/index.html": "config_guide/index.html",
    "concepts/configuration.html": "config_guide/index.html",
    "concepts/datasets.html": "config_guide/index.html#the-dataset-block",
    "concepts/execution-flow.html": "config_guide/index.html#four-files-four-commands",
    "concepts/streaming.html": "usage/large-images.html#patch-streaming",
    "concepts/model-graph.html": "reference/components/models.html#model-graph-and-output-naming",
    "concepts/yaml-model-builder.html": "reference/components/models.html#declarative-yaml-model-graphs",
    "usage/index.html": "index.html",
    "usage/python-workflows.html": "usage/python-api.html",
    "usage/docker.html": "getting-started/installation.html#docker",
    "usage/benchmarks.html": "usage/large-images.html#reproducing-the-numbers",
    "reference/index.html": "reference/cli.html",
    "reference/python-api.html": "usage/python-api.html#apps-the-konfai-apps-package",
    "reference/environment.html": "reference/cli.html#environment-variables",
    "reference/api/workflows.html": "usage/python-api.html#signatures",
    "reference/api/apps.html": "usage/python-api.html#app-signatures",
    "reference/api/configuration.html": "reference/api/index.html#configuration-api",
    "reference/api/data.html": "reference/api/index.html#data-api",
    "reference/api/models.html": "reference/api/index.html#models-api",
    "reference/api/extension-points.html": "usage/custom-models.html",
    "reference/components/index.html": "reference/components/models.html",
    "reference/components/augmentations.html": "reference/components/transforms.html#augmentations",
    "reference/components/schedulers.html": "reference/components/losses-metrics.html#schedulers",
    "ecosystem/index.html": "usage/apps.html#the-ecosystem-around-an-app",
    "examples/bring-your-model.html": "examples/index.html#bring-your-model",
    "examples/large-images.html": "examples/index.html#large-images",
}


def _write_redirects(app, exception):
    if exception is not None or app.builder.name != "html":
        return
    from pathlib import Path as _Path

    outdir = _Path(app.outdir)
    for old, new in _REDIRECTS.items():
        target, _, anchor = new.partition("#")
        href = os.path.relpath(target, os.path.dirname(old) or ".").replace(os.sep, "/")
        if anchor:
            href += "#" + anchor
        stub = outdir / old
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text(
            "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            f'<meta http-equiv="refresh" content="0; url={href}">\n'
            f'<link rel="canonical" href="{_LLMS_BASE_URL}/{new}">\n'
            "<title>This page moved</title>\n</head>\n<body>\n"
            f'<p>This page moved to <a href="{href}">{new}</a>.</p>\n</body>\n</html>\n',
            encoding="utf-8",
        )


def setup(app):
    app.connect("build-finished", _write_llms_txt)
    app.connect("build-finished", _write_redirects)

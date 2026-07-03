import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../../"))  # to access konfai/

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
]

templates_path = ["_templates"]
html_static_path = ["_static"]
exclude_patterns = ["_build", "build", "Thumbs.db", ".DS_Store"]

html_theme = "shibuya"
html_title = "KonfAI documentation"
html_favicon = os.path.abspath("../../logo.png")
html_css_files = ["custom.css"]

# The KonfAI logo is a full wordmark (mark + "KonfAI"). We use transparent-
# background versions of it derived from logo.png: the original mint art for the
# dark header, and a deep-teal recolor for the light header (mint is too light on
# white). shibuya runs pathto(src, 1), so paths are relative to the output root
# (include the _static/ prefix; depth is adjusted per page). The theme's
# duplicate "KonfAI" text label is hidden in custom.css.
html_theme_options = {
    "light_logo": "_static/konfai-logo-light.png",
    "dark_logo": "_static/konfai-logo-dark.png",
}

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
    "intersphinx.external",
]

autodoc_default_options = {
    "members": True,
    "private-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

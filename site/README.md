# The KonfAI site

Plain static pages presenting the KonfAI family: the engine, its apps, the MCP server, Scout, Studio and Edge,
plus a short documentation under `docs/` (ten pages for a newcomer; the exhaustive reference stays on
readthedocs). No framework, no build step, no external script: HTML pages, one stylesheet, the logos and one
screenshot under `assets/`, and `llms.txt` for agents. Every fact comes from the repository's own READMEs and docs; the
MCP page's tool table is the first sentence of each tool in the generated reference
(`.claude/skills/konfai-experiments/references/tool-reference.md`): when the tools change, regenerate that
reference, then update the table.

## Serve it locally

```bash
python -m http.server 8000 --directory site
# then open http://127.0.0.1:8000/
```

## Publishing

The site is hosted on GitHub Pages (`https://fideus-labs.github.io/KonfAI/`). Read the Docs stays, as the exhaustive
reference every docs page links to.

`.github/workflows/site.yml` deploys `site/` to GitHub Pages with `actions/upload-pages-artifact` and
`actions/deploy-pages`. Its only trigger is `workflow_dispatch`: nothing deploys until someone runs it by hand.
To switch it on:

1. In the repository settings, Pages, set the source to "GitHub Actions".
2. Run the "site" workflow from the Actions tab (or `gh workflow run site.yml`).

The pages use relative links only, so they also serve from any sub-path or from another static host.
The Edge demo is meant to land under `edge/` here; `docs/edge.html` says so and will link to it. The visual identity (fonts, tokens,
components, the hero) is `design/` at the repository root, copied here by `design/sync.sh`.

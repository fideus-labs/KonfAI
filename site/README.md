# The KonfAI site

Plain static pages presenting the KonfAI family: the engine, its apps, the MCP server, Scout and Studio.
No framework, no build step, no external script: five HTML pages, one stylesheet, the logos and one screenshot
under `assets/`, and `llms.txt` for agents. Every fact comes from the repository's own READMEs and docs; the
MCP page's tool table is the first sentence of each tool in the generated reference
(`.claude/skills/konfai-experiments/references/tool-reference.md`): when the tools change, regenerate that
reference, then update the table.

## Serve it locally

```bash
python -m http.server 8000 --directory site
# then open http://127.0.0.1:8000/
```

## Publishing

The folder is ready for GitHub Pages as a static artifact: a workflow on `main` would check the repository
out, upload `site/` with `actions/upload-pages-artifact` and deploy it with `actions/deploy-pages` (the Pages
source set to "GitHub Actions" in the repository settings). The pages use relative links only, so they also
serve from any sub-path or from another static host. No such workflow is added here.

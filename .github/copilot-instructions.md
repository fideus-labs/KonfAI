# KonfAI. GitHub Copilot Instructions

Follow the canonical repository instructions in `AGENTS.md`.

- Never commit directly to `main`; always use a focused feature branch.
- Keep each diff small and open a pull request for review.
- Do not merge your own pull request.
- Use Conventional Commits.
- Never include Maestro, Claude, Codex, generated-by/generated-with text, or AI co-author branding in commit messages.
- Run `pixi run format`, `pixi run check`, and relevant pre-commit hooks before finalising.
- Do not introduce dependencies silently or load complete medical imaging datasets into RAM.

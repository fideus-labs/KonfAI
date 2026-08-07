<!--
Thanks for contributing to KonfAI!
PR titles must follow Conventional Commits, e.g.:
  feat(utils): add OME-Zarr resolution selection
  fix(data): correct degree-to-radian rotation
Types: feat | fix | perf | refactor | docs | test | build | ci | chore
-->

## Description

<!-- What does this PR change, and why? Keep it focused on one logical change. -->

## Related issues

<!-- e.g. "Closes #123", "Refs #456". Delete if none. -->

## Type of change

- [ ] `feat`: new feature
- [ ] `fix`: bug fix
- [ ] `perf`: performance improvement
- [ ] `refactor`: no behaviour change
- [ ] `docs`: documentation only
- [ ] `test`: tests only
- [ ] `build` / `ci` / `chore`: tooling, deps, CI
- [ ] Breaking change (describe the migration below)

## How has this been tested?

<!-- Commands run and what you observed. -->

```bash
pixi run check
```

## Checklist

- [ ] PR title and commits follow **Conventional Commits**
- [ ] `pixi run check` passes (ruff lint + format + tests)
- [ ] `pixi run --environment dev python -m pytest konfai-apps/tests` passes *(if `konfai-apps/` changed)*
- [ ] Tests added/updated for the change
- [ ] Docs updated *(if user-facing CLI/config behaviour changed)*
- [ ] No new runtime dependency without a matching `pyproject.toml` extra
- [ ] Lazy/patch data access preserved (no full-volume reads added)

## Breaking changes & migration

<!-- Describe any breaking change and how users should migrate. Delete if none. -->

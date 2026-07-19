# Publishing & updating an app bundle (HF + Slicer)

The recurring maintainer workflow that `konfai-apps` covers only in fragments: turn a trained workflow into
a published app bundle, push it to Hugging Face, and keep the Slicer extensions that consume it working.

**No external repo is cloned by this skill, and no address is hard-coded here** — they rot. The canonical
sources are: the **README "App bundles" / "Ecosystem" tables** (the current HF repo IDs and the Slicer repo
URLs), the bundle's own **`app.json`** (its repo/app identity), and your **local checkouts** of the Slicer
repos (which you must have for the grep step below).

## What a bundle is

A bundle is a directory in **HF layout**: `app.json` (metadata) + one or more configs (`Prediction.yml`,
optional `Evaluation.yml`) + checkpoint `.pt` file(s) + an optional custom `Model.py` + `requirements.txt`,
optionally `model.onnx` + `manifest.json` for the portable runtime. Its configs use **bundle-relative
classpaths** (`ResidualEncoderUNet.yml`, `Model:RegistrationNet`), never KonfAI's internal module paths —
that is why the `models`→`models.python` move did not break bundles. A bundle resolves from a **Local dir**,
an **HF repo** (`repo_id:app_name`), or a **Remote server**.

## Assemble (local)

`konfai-apps bundle` writes the HF layout locally; it does **not** upload.

```bash
konfai-apps bundle <name> --out ./bundles --app-json app.json \
  --config Prediction.yml [Evaluation.yml] --checkpoint CV_0.pt [CV_1.pt …] \
  [--model-py Model.py] [--requirements requirements.txt] [--onnx]
```

- Omit `--requirements` and a draft is derived from `Model.py` imports — review it; the app runtime
  pip-installs it by default (trust boundary, opt out with `KONFAI_APPS_INSTALL_REQUIREMENTS=0`).
- `--onnx` also exports `model.onnx` + `manifest.json` (patch size / channels read from the config unless
  overridden). From an MCP session, `package_app_from_session` / `export_app` do the equivalent.

## Validate before publishing

- **Load the bundle's config on a COPY** — reading a config *mutates it in place*; validating the original
  rewrites the file you are about to ship.
- Run the app end to end on a real input (`konfai-apps infer <local-bundle-dir>:<name> -i … -o …`) and open
  the output — a green resolve is not proof (AGENTS.md §5b). For a multi-model ensemble, confirm the label
  merge is `MergeLabels`, not a cumulative `Sum` (the 1.4.0 regression).

## Publish to HF (manual)

There is no `konfai-apps` upload command. Push the assembled directory with `huggingface_hub`:

```bash
hf upload <repo_id> ./bundles/<name> <name> --repo-type model   # or HfApi().upload_folder(...)
```

The repo IDs are those in the README app table (e.g. the `VBoussot/*` family); a bundle records its own in
`app.json`. After the push, `konfai-apps download <repo_id>:<name>` (or an `infer` that resolves it) fetches
the new files — verify the round trip once.

## Keep the Slicer extensions working

SlicerKonfAI (general apps) and SlicerImpactReg (registration) drive `konfai-apps` by CLI + JSON and import
a **frozen public surface** from `konfai_apps.app_repository` (incl. `current_free_vram`,
`get_app_repository_info`, `is_app_repo`, the `LocalAppRepository*` classes, `AppRepositoryError`, and the
`get_parameters() -> {values, constraints}` primitive) plus the top-level `konfai/__init__` helpers.

- The contract is locked by `konfai-apps/tests/test_slicer_*` and `tests/unit/test_slicer_core_api_contract.py`.
  If you must change one of those symbols, **grep your local Slicer checkouts first** (repos linked from the
  README ecosystem table) — a dropped `current_free_vram` broke Slicer once (PR#33 → PR#35).
- Publishing a new bundle does not require a Slicer code change; a Slicer *extension* change (new params
  dialog, engine) needs a commit in its own repo **and a manual extension re-install** in 3D Slicer — that
  last step is not automatable from here.

## Checklist

1. `bundle` assembled, `requirements.txt` reviewed.
2. Config validated on a **copy**; app ran end to end on a real input; output opened; label-merge correct.
3. `hf upload` pushed; `download`/`infer` round-trip verified against the Hub.
4. If a Slicer symbol changed: grepped both Slicer checkouts, contract tests green, extension re-installed.

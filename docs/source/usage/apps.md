# Using KonfAI Apps

KonfAI Apps are the **deployment and reuse layer** of the framework: packaged
workflows that expose a stable user interface on top of KonfAI's low-level
prediction, evaluation, uncertainty, and fine-tuning logic. This page explains
what an app is and shows how to drive packaged apps with the `konfai-apps`
CLI — locally or against a remote server.

## What an app is

Where low-level KonfAI workflows are designed directly through YAML files and
local Python modules, a KonfAI App bundles those assets into a reusable package
that can run through:

- `konfai-apps` on the command line
- Python via `konfai_apps.KonfAIApp`
- a remote FastAPI server via `konfai-apps-server`
- clients such as 3D Slicer integrations

A KonfAI App repository is recognized by the presence of an `app.json` file.
Typical contents are:

```text
my_app/
├── app.json
├── Prediction.yml
├── Evaluation.yml
├── Uncertainty.yml
└── checkpoint.pt
```

Depending on the app, some files are optional:

- `Prediction.yml` is the core inference entrypoint
- `Evaluation.yml` is needed for `eval`
- `Uncertainty.yml` is needed for `uncertainty`
- fine-tuning relies on training assets and checkpoint files that live next to the app

**When to use apps.** Use low-level YAML workflows when you are still designing
or debugging a model. Package it as a KonfAI App once that workflow is stable
and you want:

- a stable inference interface — a smaller, more repeatable interface for end users
- reusable packaging for a team
- distribution through Hugging Face or a private repository
- local and remote execution with the same user-facing command

## The `konfai-apps` CLI

The app CLI currently exposes these subcommands:

- `infer`
- `eval`
- `uncertainty`
- `pipeline`
- `fine-tune`

The main command pattern is:

```bash
konfai-apps <command> <app> [options]
```

## App identifiers

Apps can be local or remote repository identifiers. The repository examples and
tests show Hugging Face style identifiers such as:

- ``VBoussot/ImpactSynth:MR``
- ``VBoussot/ImpactSynth:CBCT``
- ``VBoussot/TotalSegmentator-KonfAI:total``

## Common app workflows

These commands all use the same app package, but they expose different levels
of workflow orchestration.

Inference:

```bash
konfai-apps infer VBoussot/ImpactSynth:CBCT \
  -i input.mha -o ./Output --gpu 0
```

Evaluation:

```bash
konfai-apps eval VBoussot/ImpactSynth:CBCT \
  -i prediction.mha --gt ct.mha --mask mask.mha --gpu 0
```

Pipeline:

```bash
konfai-apps pipeline VBoussot/ImpactSynth:CBCT \
  -i input.mha --gt ct.mha --mask mask.mha -o ./Output -uncertainty
```

## Grouped inputs

The CLI accepts grouped inputs by repeating `--inputs` / `-i`. This matches the
grouping behavior documented in `konfai_apps.cli.add_common_konfai_apps()`.

Use this when an app expects:

- multiple input groups
- multiple files per group
- paired inputs such as image + mask

## Fine-tuning

Fine-tuning is available through:

```bash
konfai-apps fine-tune <app> <name> -d ./Dataset --epochs 10 --gpu 0
konfai-apps fine-tune <app> <name> -d ./Dataset --models CV_0 CV_1 --epochs 10 --gpu 0
```

Under the hood, the app installs training assets, links the dataset, then, for each selected
checkpoint, restarts training from its pretrained weights (fresh optimizer, learning-rate schedule
and epoch counter) so that `--epochs` fine-tuning epochs actually run. Use `--models` to choose which
checkpoint(s) to fine-tune (default: the first available); each is fine-tuned independently and
written back into the output app, which is left as a ready-to-use app bundle.

## Remote execution

Any `konfai-apps` command becomes remote as soon as `--host` is provided — the
CLI still looks the same; only the execution backend changes. The client then:

1. uploads inputs
2. schedules the job
3. streams logs over SSE
4. downloads the zipped result bundle

On the server side, jobs are queued, optionally assigned GPUs, executed in
isolated temporary workspaces, and cleaned up after a grace period.

### Start the server

The server requires a JSON file listing the available apps:

```bash
konfai-apps-server --host 0.0.0.0 --port 8000 --apps konfai-apps/tests/assets/apps.json
```

Bearer-token authentication is enabled by default:

```bash
export KONFAI_API_TOKEN="my-secret-token"
konfai-apps-server --apps konfai-apps/tests/assets/apps.json
```

See {doc}`../reference/cli` for the full `konfai-apps-server` flag reference.

### Run a remote job

```bash
konfai-apps infer VBoussot/ImpactSynth:CBCT \
  -i input.mha -o ./Output \
  --host my.server.org --port 8000 --token "$KONFAI_API_TOKEN"
```

The complete HTTP contract behind this — health, device, and app metadata
endpoints plus job status, log, result, and kill endpoints — is documented in
{doc}`../reference/app-server-api`.

## Next steps

- {doc}`../reference/cli` — the full flag reference for `konfai-apps` and `konfai-apps-server`
- {doc}`../reference/python-api` — the `KonfAIApp` / `KonfAIAppClient` Python API and the trust model
- {doc}`../reference/app-server-api` — the HTTP endpoint contract of the server

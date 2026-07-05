---
name: konfai-cli
description: >-
  Run KonfAI deep-learning workflows for medical imaging (segmentation, synthesis,
  registration) from the command line: author or adapt a YAML config, then train, resume,
  predict, and evaluate with the `konfai` CLI, or run a packaged model with the `konfai-apps`
  CLI. Use when the user wants to train / fine-tune / run inference / evaluate a KonfAI model
  from the terminal, adapt an example (Segmentation / Synthesis) config, understand the
  workspace outputs (Checkpoints / Predictions / Evaluations), reference a custom model or
  loss by classpath, or run and serve a published app (impact-synth, impact-seg, konfai-apps,
  konfai-apps-server). Triggers: "train a KonfAI model", "konfai TRAIN / PREDICTION /
  EVALUATION", "run konfai on the CLI", "konfai-apps infer", "run the segmentation example",
  "evaluate my predictions", "fine-tune this app".
---

# Running KonfAI from the command line

KonfAI is **config-driven**: a model, its data pipeline, losses/metrics, optimizer, and the
whole train/predict/evaluate workflow are described in **YAML** and mapped onto Python objects
by a reflection engine — no experiment-specific code for standard tasks. **The config is the
experiment**, and every run leaves a fully-resolved config on disk.

There are two command-line surfaces:

- **`konfai`** — the low-level engine: author a config and `TRAIN` / `RESUME` / `PREDICTION` /
  `EVALUATION`. This is the main path for building and training a workflow.
- **`konfai-apps`** — the packaged-app runtime: run inference/evaluation with an already-trained
  model bundled as an *app* (local, HuggingFace, or a remote server). Use this to *run* a
  stable model, not to build one. See [references/apps-layer.md](references/apps-layer.md).

## The canonical loop (`konfai`)

Three workflows map to three files, each with one mandatory root key:

| Command | File | Root key |
|---|---|---|
| `TRAIN` / `RESUME` | `Config.yml` | `Trainer:` |
| `PREDICTION` | `Prediction.yml` | `Predictor:` |
| `EVALUATION` | `Evaluation.yml` | `Evaluator:` |

**Don't write configs from scratch — copy a runnable template from `examples/`** (Segmentation
or Synthesis) and adapt it. Then:

```bash
cd examples/Segmentation                 # always run from the dir holding the configs + Dataset/

konfai TRAIN      -y --gpu 0 --config Config.yml
konfai PREDICTION -y --gpu 0 --config Prediction.yml --models Checkpoints/<train_name>/<checkpoint>.pt
konfai EVALUATION -y          --config Evaluation.yml
```

Outputs are namespaced by the `train_name` in the config: `Checkpoints/<train_name>/`,
`Statistics/<train_name>/`, `Predictions/<train_name>/`, `Evaluations/<train_name>/`. To
iterate: edit the YAML (or bump `train_name`), re-run. See
[references/examples-and-recipes.md](references/examples-and-recipes.md) for the verified
Segmentation and Synthesis recipes, and
[references/cli-reference.md](references/cli-reference.md) for every flag.

## Rules that keep runs correct

- **Run from the directory that holds the configs** (and `Dataset/`). KonfAI resolves relative
  paths — configs, `Dataset/`, output dirs, and local `File:Class` classpaths — against the
  current working directory (it prepends CWD to `sys.path`).
- **Reading a config rewrites it on disk.** A run materialises resolved defaults back into the
  YAML (`None` becomes the literal `"None"`). Expect a post-run git diff; keep configs under
  version control. There is no read-only path. (Details:
  [references/workspace-and-runtime.md](references/workspace-and-runtime.md).)
- **`train_name` is the join key.** `Prediction.yml` and `Evaluation.yml` must use the *same*
  `train_name` as the training run whose checkpoints/predictions they consume — the most common
  failure is "evaluation can't find predictions" from a mismatched `train_name`.
- **`--gpu` and `--cpu` are mutually exclusive**; with neither, it runs on CPU. `--gpu` ids are
  validated against the visible CUDA devices. `--cpu N` needs `N > 0`.
- **Install the imaging extra** to read `.mha` / medical formats: `pip install "konfai[imaging]"`
  (a bare install fails on the first data read).
- **`-y/--overwrite` overwrites existing outputs without prompting** — a destructive flag;
  don't add it blindly when a prior run's outputs matter. `RESUME` needs `--model`; `PREDICTION`
  needs `--models`.
- **Custom models/losses/transforms** live in a `.py` beside the config and are referenced by
  classpath `File:Class` (e.g. `Model:UNetpp5`). Write the `.py` before running.

## Running a packaged app (`konfai-apps`)

When a workflow is stable, it can be shipped as an **app** and run without touching YAML:

```bash
konfai-apps infer VBoussot/ImpactSynth:sCT -i patient/mr.nii.gz -o ./Output --gpu 0
```

`konfai-apps` also does `eval`, `uncertainty`, `pipeline`, `fine-tune`, `bundle`, and
`download`; `konfai-apps-server` serves apps over HTTP (bearer auth by default). The published
bundles under `apps/` (e.g. `impact-synth-konfai synthesize ...`, `impact-seg-konfai segment ...`)
are thin task-named wrappers.

> ⚠️ **Trust model.** Resolving/running an app **copies and imports the app's `.py` files**
> (runs arbitrary code) and can pip-install its `requirements.txt`. **Only run apps from sources
> you trust.** See [references/apps-layer.md](references/apps-layer.md).

## Reference material (load on demand)

- [references/cli-reference.md](references/cli-reference.md) — the `konfai` / `konfai-cluster` commands and every flag.
- [references/config-authoring.md](references/config-authoring.md) — writing KonfAI YAML: files, root keys, classpaths, conventions, the mutation invariant.
- [references/examples-and-recipes.md](references/examples-and-recipes.md) — verified Segmentation + Synthesis train→predict→evaluate recipes.
- [references/workspace-and-runtime.md](references/workspace-and-runtime.md) — outputs keyed by `train_name`, env vars, DDP, SLURM, config modes.
- [references/apps-layer.md](references/apps-layer.md) — `konfai-apps` / `konfai-apps-server`, app resolution, trust model, published bundles.

The authoritative user-facing catalogue lives in `docs/source/config_guide/` (`training.md`,
`prediction.md`, `evaluation.md`) and `docs/source/reference/cli.md`; `AGENTS.md` is the source
of truth for framework internals and conventions.

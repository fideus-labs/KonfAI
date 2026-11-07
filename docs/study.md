# Fixed views against active exploration

Scout records enough to compare two ways of inspecting the same case with the same model:

- `fixed`: the model sees a fixed set of views (one montage per plane in the task's default mode, and the central
  slice of each plane in the secondary modes), can measure regions and ask for statistics, cannot ask for a view.
- `active`: the model starts from the same montages and explores: slices in any plane, zooms, modes, measurements.

Everything else is identical: the system prompt (only the procedure paragraph differs, and it is recorded), the
tools `measure`, `region_stats`, `note` and `conclude`, the budget, the report.

## What is recorded for the study

Per inspection (`report.json`): the prompt and its version, the configuration, the provider and model, every tool
call with its arguments and duration, every view with where it sits in the world and whether it was sent, every
measurement with its request, the observations with their evidence, the conclusion, tokens, cost, duration, the
budget and what exhausted it. `transcript.json` keeps the conversation.

## Running it

```bash
python examples/study_fixed_vs_active.py --provider anthropic --allow-cloud-images --repeats 3
konfai-scout compare examples/out/study/*/*                    # one row per inspection
konfai-scout compare examples/out/study/*/* --json             # the same for a notebook
```

`compare` lists per inspection: mode, model, verdict, confidence, tool calls, views, images sent, measurements,
observations by kind, tokens, cost, duration, whether the budget was exhausted. The planted defects of a synthetic
case are in its `defects.json`, in millimetres, so a notebook can match observations to them by box overlap.

## Keeping runs comparable

- Pin the model id and the prompt version; a report names both.
- Give both modes the same budget; the fixed mode spends fewer calls by construction, which is part of the result.
- Keep `allow_cloud_images` explicit per run; the report records the provider as remote or local.
- Repeat: a model is not deterministic. `--repeats` runs each mode several times under distinct directories.

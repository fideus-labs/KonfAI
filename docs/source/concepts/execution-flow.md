# Execution flow

This page walks through what happens when you launch the four KonfAI
workflows (`TRAIN`, `PREDICTION`, `EVALUATION`, and `TRANSFORM`), from config
parsing to the files each one writes. Read it to know where a run's outputs land
and how the distributed runtime wraps every command.

**The model workflows write into a single workspace keyed by `train_name`: `Checkpoints/<train_name>/`, `Predictions/<train_name>/`,
`Evaluations/<train_name>/`: so the `train_name` in each config file must name
the run you intend to touch. `TRANSFORM` is the exception: it is keyed by `name`,
and only its log, its plan and a copy of its config land in the workspace
(`Transforms/<name>/`): the data goes wherever each `Write:` stage says, which
that run directory records in `outputs.json`.**

KonfAI ships four low-level workflows and one higher-level app layer.

```{mermaid}
flowchart LR
    C[Config.yml<br/>Trainer:]:::cfg --> T([konfai TRAIN]):::cmd
    P[Prediction.yml<br/>Predictor:]:::cfg --> R([konfai PREDICTION]):::cmd
    E[Evaluation.yml<br/>Evaluator:]:::cfg --> V([konfai EVALUATION]):::cmd
    X[Transform.yml<br/>Transformer:]:::cfg --> W([konfai TRANSFORM]):::cmd

    T --> TO[Checkpoints/&lt;train_name&gt;<br/>Statistics/&lt;train_name&gt;]:::out
    R --> RO[Predictions/&lt;train_name&gt;]:::out
    V --> VO[Evaluations/&lt;train_name&gt;<br/>Metric_*.json]:::out
    W --> WO[wherever each Write: points<br/>Transforms/&lt;name&gt;/plan.txt]:::out

```

One root key per file, one command per file. The same reflection engine builds
each one: only the root key changes what it builds.

## Low-level workflows

The `konfai` CLI dispatches to four public functions:

- `konfai.trainer.train`
- `konfai.predictor.predict`
- `konfai.evaluator.evaluate`
- `konfai.transformer.transform`

Each wrapper prepares a small execution context, then instantiates the
corresponding configured object:

- `Trainer`
- `Predictor`
- `Evaluator`
- `Transformer`

The key environment variables are documented in
{doc}`../reference/environment`.

## What happens during training

At a high level, `TRAIN` does the following:

1. parse `Config.yml` into a `Trainer`
2. prepare the dataset and its train/validation split
3. initialize the model graph, losses, and schedulers
4. run the training loop
5. save checkpoints and logs
6. copy the active config into the statistics directory

Outputs are written to:

- `Checkpoints/<train_name>/`
- `Statistics/<train_name>/`

## What happens during prediction

`PREDICTION`:

1. parses `Prediction.yml` into a `Predictor`
2. loads one or more checkpoints
3. prepares the inference dataset
4. runs the model in prediction mode
5. writes output datasets defined in `outputs_dataset`
6. copies `Prediction.yml` into the prediction directory

Outputs are written to:

- `Predictions/<train_name>/`

## What happens during evaluation

`EVALUATION`:

1. parses `Evaluation.yml` into an `Evaluator`
2. loads the dataset pairs needed for metric computation
3. validates that configured output and target groups exist
4. computes per-case and aggregate metrics
5. writes JSON reports
6. copies the evaluation config into the evaluation directory

Outputs are written to:

- `Evaluations/<train_name>/Metric_TRAIN.json`
- optionally `Evaluations/<train_name>/Metric_VALIDATION.json`

## What happens during a transform

`TRANSFORM` runs no model:

1. parses `Transform.yml` into a `Transformer`, refusing any key its grammar
   does not know
2. binds every chain and runs the parse-time refusals: a chain not ending in
   `Write`, two chains writing the same target, a `Write` inside a source
3. computes and prints the per-case plan: `STREAM`, `WHOLE-VOLUME`, `REDUCE`,
   `SKIP` or `REFUSED`: probing each destination with a real region-write open
4. refuses **before writing a byte** when an entry's working set exceeds the
   per-rank `memory_budget`
5. shards the cases across ranks and materializes each chain's `Write` stages,
   skipping any case whose output already exists unless `-y`

Outputs are written to:

- wherever each `Write: {dataset: ...}` points
- `Transforms/<name>/plan.txt`, `outputs.json`, the copied config, and the logs

## Programmatic vs CLI entrypoints

The same workflows can also be built programmatically through:

- `build_train(...)`
- `build_predict(...)`
- `build_evaluate(...)`
- `build_transform(transform_file=..., transforms_dir=...)`

This is useful when you want to validate a config before launching the full
runtime. `konfai.transformer.plan_transform(...)` goes one step further: it
builds the workflow, computes the plan, prints it, and returns the
`TransformPlan` without transforming anything: the `--plan` flag's entrypoint.

## Distributed execution

The execution layer is handled by the distributed runtime utilities in
`konfai.utils.runtime`.

From the code, this layer is responsible for:

- setting `CUDA_VISIBLE_DEVICES`
- handling overwrite and verbosity flags
- launching TensorBoard when requested
- spawning worker processes with `torch.multiprocessing.spawn`
- initializing `torch.distributed` with a local TCP port

This means that even local multi-process execution uses the same distributed
bootstrap logic.

## Apps

`konfai-apps` is the higher-level interface. It packages low-level prediction,
evaluation, uncertainty, and fine-tuning workflows into reusable app bundles.

See {doc}`../usage/apps`.

## Next steps

- {doc}`../config_guide/training`: every `Config.yml` key the training workflow reads.
- {doc}`../config_guide/prediction`: configuring checkpoints, patch inference, and `outputs_dataset`.
- {doc}`../config_guide/evaluation`: turning predictions and ground truth into metric JSON.
- {doc}`../config_guide/transform`: the model-less workflow: chains, `Write`, and the plan.

# Execution flow

This page walks through what happens when you launch the three KonfAI
workflows — `TRAIN`, `PREDICTION`, and `EVALUATION` — from config parsing to
the files each one writes. Read it to know where a run's outputs land and how
the distributed runtime wraps every command.

**All three workflows write into a single workspace keyed by `train_name` —
`Checkpoints/<train_name>/`, `Predictions/<train_name>/`,
`Evaluations/<train_name>/` — so the `train_name` in each config file must name
the run you intend to touch.**

KonfAI ships three low-level workflows and one higher-level app layer.

## Low-level workflows

The `konfai` CLI dispatches to three public functions:

- `konfai.trainer.train`
- `konfai.predictor.predict`
- `konfai.evaluator.evaluate`

Each wrapper prepares a small execution context, then instantiates the
corresponding configured object:

- `Trainer`
- `Predictor`
- `Evaluator`

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

## Programmatic vs CLI entrypoints

The same workflows can also be built programmatically through:

- `build_train(...)`
- `build_predict(...)`
- `build_evaluate(...)`

This is useful when you want to validate a config before launching the full
runtime.

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

- {doc}`../config_guide/training` — every `Config.yml` key the training workflow reads.
- {doc}`../config_guide/prediction` — configuring checkpoints, patch inference, and `outputs_dataset`.
- {doc}`../config_guide/evaluation` — turning predictions and ground truth into metric JSON.

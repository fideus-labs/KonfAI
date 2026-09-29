# Glossary

The words KonfAI uses in its configs, logs and docs. Several carry two meanings
depending on where they appear; each entry says which is which.

**App.** A model packaged for `konfai-apps`: a config, its weights and any
custom `.py`, resolved from a local directory, a Hugging Face repository or a
remote server. See {doc}`../usage/apps`.

**Bundle.** Two things. An *app bundle* is the directory or Hugging Face
repository an app is resolved from (`app.json`, the config, the `.pt` files),
which `konfai-apps bundle` assembles. A *MONAI bundle* is the model-zoo format
`konfai.import_bundle` and `konfai.export_bundle` read and write.

**Case.** One subject of a dataset: a name shared by all its groups, for
example `CASE_001` with its `CT` and its `SEG`. Cases are what a run shards
across ranks and what the per-case metrics are keyed by.

**Checkpoint.** A `.pt` file in `Checkpoints/<train_name>/`. The models are
named after the moment they were written, `<YYYY_MM_DD_HH_MM_SS>.pt`. Beside
them, `resume_latest.pt` is the latest training continuation, for `RESUME`,
and `crash_<date>.pt` a save on an exceptional exit; neither is a model to
predict with.

**Copy.** One augmented variant of a case: each augmentation list declares `nb`
of them, drawn per case. At prediction, an ensemble or a TTA makes copies of one
case's prediction, which a reduction folds into one.

**Fold.** Two things. A *cross-validation fold* is one split of the cases (the
`folds/fold_i.txt` lists `generate_folds` writes) and the model trained on it; an
app that ships one checkpoint per fold runs them as an ensemble. A *fold* in the
reduction sense is what a `Reduction` does: combine N tensors into one, the
copies of one case at prediction time, or the cases of a cohort under the
TRANSFORM `Reduce` stage.

**Group.** Under `groups_src` and in a criterion's targets, a *group* is a named
volume of each case (`CT`, `SEG`, `MASK`), and `OutputDataset`'s `group` is the
name the prediction is written under. Under a criterion
(`criterions_loader.<Criterion>.group`), `group` is an unrelated integer: the
losses that share it are summed and back-propagated together, one backward
pass per group.

**Patch.** A tile of a case, `Patch.patch_size`: what the loader serves the
model and what prediction reassembles, blended over their overlap.
`ModelPatch` is a second tiling inside the network.

**Rank.** One process of a run: `--gpu 0 1` runs two ranks, `--cpu 4` four.
The CLI also calls a CPU rank a *worker*, as in `Running on CPU (4 workers)`.

**Region.** A box of a stored volume that a streamed run reads or writes
without loading the rest. See {doc}`../usage/large-images`.

**TRAIN.** The command that trains a model. In an evaluation, `Metric_TRAIN.json`
reports the evaluated cases a `validation` selector does not set apart (all of
them without one), whether or not a model was trained on them, and
`Metric_VALIDATION.json` the cases it sets apart.

**`train_name`.** The name a model workflow's outputs are keyed by:
`Checkpoints/`, `Statistics/`, `Predictions/` and `Evaluations/` each hold a
`<train_name>/` directory, named by the config that wrote it. TRANSFORM spells
it `name` and writes its run logs to `Transforms/<name>/`.

**Worker.** Two things. A *DataLoader worker* is a process that loads data for
one rank (`num_workers`). In the CLI output and `--cpu` help, a *worker* is a
rank (see **Rank**).

**Workspace.** Two things. A run's *workspace* is its output tree, keyed by
`train_name` (see above) and by default under the current directory. Under
`konfai-mcp`, the *workspace root* (`KONFAI_MCP_WORKSPACES_ROOT`, default
`~/KonfAI_Workspaces`) is the directory holding the MCP sessions and their
datasets.

Quickstart
==========

Train, predict and evaluate a two-class segmentation on four tiny synthetic CT
volumes. This first run uses one CPU, downloads no dataset and checks the files
it produces. The labels are **0 = background, 1 = foreground**.

Choose :doc:`usage/apps` to run an existing trained app,
:doc:`usage/adopting-konfai` to bring your own model, or
:doc:`examples/transform` to prepare a dataset without training.

Install a wheel and copy the example
------------------------------------

Use Python 3.11 or newer. These commands use a POSIX shell; on Windows activate
the virtual environment with ``.venv\Scripts\Activate.ps1`` and copy the
directory with ``Copy-Item -Recurse``.

.. code-block:: bash

   git clone https://github.com/fideus-labs/KonfAI.git
   python -m venv .venv
   . .venv/bin/activate
   python -m pip install "./KonfAI[itk]"
   cp -r KonfAI/examples/Segmentation/TwoClasses konfai-first-run
   cd konfai-first-run
   konfai --version

``pip`` installs a wheel built from that checkout, so the example and the
package are the same revision. ``[itk]`` is the reader for the example's
``.mha`` images.

For a published installation in your own project, use
``python -m pip install "konfai[itk]"``; if you copy examples from Git, select
the release tag matching the installed version. See
:doc:`getting-started/installation` for other readers and GPU setup.

Prepare four cases
------------------

.. code-block:: bash

   python quickstart.py prepare

This creates ``Dataset/CASE_000`` through ``CASE_003``, each containing
``CT.mha`` and ``SEG.mha``. The images are 32 × 32 × 4 voxels with non-unit
spacing and a rotated direction matrix, so the final check also exercises
physical geometry. ``CASE_003`` is held out during training.

The three YAML files are short and complete enough to adapt:

* ``Config.yml``: a 2D UNet from the installed catalog, two output channels,
  cross entropy and foreground Dice losses, twenty epochs (about seven seconds
  on one CPU core).
* ``Prediction.yml``: the same model parameters and CT normalization, with
  the ``Argmax`` head written as the label image ``PRED.mha``.
* ``Evaluation.yml``: ``PRED`` against ``SEG``, foreground label ``[1]``.

Train, predict and evaluate
---------------------------

Keep this directory as your working directory. One OpenMP thread is enough for
volumes this small.

.. code-block:: bash

   export OMP_NUM_THREADS=1
   konfai TRAIN -y --cpu 1 --config Config.yml
   python quickstart.py checkpoint
   konfai PREDICTION -y --cpu 1 --config Prediction.yml --models "$(python quickstart.py checkpoint)"
   konfai EVALUATION -y --cpu 1 --config Evaluation.yml
   python quickstart.py verify

On PowerShell, set ``$env:OMP_NUM_THREADS = "1"`` instead of ``export``; the
checkpoint subexpression also works there.

``save_checkpoint_mode: BEST`` keeps one dated model and ``quickstart.py
checkpoint`` prints its filename. ``resume_latest.pt`` is a training
continuation and ``crash_*.pt`` an exceptional save, not models to predict
with; several ``--models`` paths run an ensemble.

A run writes its resolved defaults back into the YAML files. For another fresh
run, copy the example to a new directory.

Check the result
----------------

The final command fails unless all four predictions exist, their voxel sizes,
spacing, origin and direction match both CT and SEG, their labels are finite
0/1 values, and the four reported Dice values match a recalculation from the
written segmentations. A successful report includes ``"verified_cases": 4``
and ``"geometry": "matches CT and SEG"``, followed by the four actual Dice
values. Inspect ``Predictions/CT_TWO_CLASSES/Dataset/`` and
``Evaluations/CT_TWO_CLASSES/Metric_TRAIN.json`` for the files.

The evaluation includes the three training cases and the one held-out case.
Twenty epochs on procedural shapes reach a Dice of about 0.98, the held-out
case included. The check passes on any Dice that agrees with the written files:
it verifies the workflow, not accuracy on medical images.

Adapt it to your CT and labels
------------------------------

Keep one case directory per CT/SEG pair. For two classes, map your target
structure to ``1`` and background to ``0``. Then change:

* ``dataset_filenames`` in all three configs and the prediction path in
  ``Evaluation.yml`` when changing ``train_name``.
* ``validation`` in ``Config.yml`` to your held-out case names.
* CT preprocessing in **both** training and prediction. The synthetic recipe
  divides intensities by 300; choose normalization appropriate for your data.
* Model parameters in **both** configs. A different class count also requires
  matching training/evaluation ``Dice.labels``.
* ``patch_transforms: None`` on every group and the two
  ``*_reduction_transforms: None`` on the output: an absent key defaults to a
  ``Normalize`` to ``[-1, 1]``.

The verifier knows these four cases; for your dataset, give it your case
names. :doc:`examples/segmentation` describes the larger
41-class pelvis example and its notebook; :doc:`config_guide/index` explains
the configuration engine.

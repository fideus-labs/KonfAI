Quickstart
==========

In about seven minutes you will train a segmentation model on real pelvis CT
scans, predict on them, and get a Dice score back. Everything runs from three
YAML files that ship with the repository. You will not write a line of Python.

If you would rather watch it happen cell by cell, open
``examples/Segmentation/Segmentation_demo.ipynb``, or its Colab badge, and run
everything. Same run, same result.

In a hurry, or without a GPU? :doc:`examples/transform` runs in about a minute on
CPU and downloads nothing: it generates its own data, folds a cohort into one
volume and draws augmented copies. It is dataset preparation, nothing to train
first, so it is the shortest path to seeing KonfAI work.

Install
-------

You need Python 3.10 or newer. A GPU makes it faster, and every command below
works with ``--cpu 1`` instead of ``--gpu 0``.

.. code-block:: bash

   git clone https://github.com/fideus-labs/KonfAI.git
   cd KonfAI
   python -m pip install -e ".[imaging]"

The examples live in the repository, which is why you clone it. For your own
project later, ``pip install "konfai[imaging]"`` is enough. Keep the
``[imaging]`` part: it brings SimpleITK, and without it the first data read
fails.

Check it worked:

.. code-block:: bash

   konfai --help

Get the data
------------

Five pelvis CT cases with their reference segmentation, about 114 MB.

.. code-block:: bash

   cd examples/Segmentation
   python -m pip install -U "huggingface_hub[cli]"
   hf download VBoussot/konfai-demo \
     --repo-type dataset \
     --include "Segmentation/**" \
     --local-dir Dataset
   mv Dataset/Segmentation/* Dataset/
   rmdir Dataset/Segmentation
   rm -rf Dataset/.cache

You should end up with one folder per case, each holding ``CT.mha`` and
``SEG.mha``. ``CT`` is what the model reads, ``SEG`` is what it has to
reproduce.

Stay in this directory for the rest of the page. KonfAI resolves paths from
where you launch it.

Train, predict, evaluate
------------------------

Three commands, one config file each.

.. code-block:: bash

   konfai TRAIN      -y --gpu 0 --config Config.yml
   konfai PREDICTION -y --gpu 0 --config Prediction.yml --models Checkpoints/SEG_BASELINE/*.pt
   konfai EVALUATION -y          --config Evaluation.yml

Training writes ``Checkpoints/SEG_BASELINE/`` and ``Statistics/SEG_BASELINE/``.
Prediction writes ``Predictions/SEG_BASELINE/``. Evaluation writes
``Evaluations/SEG_BASELINE/Metric_TRAIN.json``. If those folders appear, your
install, your data and the whole pipeline agree with each other.

The checkpoint is named after the moment it was written, so there is no fixed
filename to type; the glob picks it up. Pass several checkpoints to ``--models``
and they run as an ensemble.

.. warning::

   A run rewrites the config file you point it at, filling in every default it
   resolved. That is how a run stays reproducible, but it means your YAML will
   show a git diff afterwards. Work on a copy if you want the shipped template
   untouched.

Look at what you got
--------------------

Open ``Metric_TRAIN.json`` for the scores, and
``Predictions/SEG_BASELINE/Dataset/`` for the segmentations themselves. The
config KonfAI copied next to them is the exact recipe that produced them.

The score will be low: five epochs prove the pipeline works, they do not train a
model. Raise ``epochs`` in ``Config.yml`` and run again.

When something goes wrong
-------------------------

Most first runs fail for one of five reasons.

- ``--gpu`` refuses your device id: KonfAI checks it against
  ``CUDA_VISIBLE_DEVICES``. Use ``--cpu 1``.
- It asks before overwriting a previous run: add ``-y``.
- It cannot find a group: every case folder needs ``CT.mha`` and ``SEG.mha``,
  named exactly as ``groups_src`` says.
- Evaluation finds no predictions: ``Prediction.yml`` and ``Evaluation.yml``
  must carry the same ``train_name``.
- A metric or output name is rejected: those names are module paths in the
  model graph. Start from a shipped example before inventing your own.

Where to go next
----------------

- :doc:`examples/index` to adapt one of the shipped workflows to your own data.
- :doc:`config_guide/index` for what every key in the three YAML files does.
- :doc:`concepts/index` to understand the machinery you just ran.

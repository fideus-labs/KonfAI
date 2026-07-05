Quickstart
==========

This quickstart takes you **from zero to a scored prediction** on the shipped
``examples/Segmentation`` baseline: install the package, prepare the demo
dataset, train a model, run prediction, then evaluate the saved outputs. Use it
for your very first contact with KonfAI — each phase below ends with an
explicit success signal so you always know whether to continue.

By the end of this page, you should have:

- one trained checkpoint under ``Checkpoints/SEG_BASELINE/``
- one statistics folder under ``Statistics/SEG_BASELINE/``
- optionally one prediction dataset and one evaluation JSON

If you prefer a notebook-driven first run, especially on a fresh machine or in
Google Colab, use ``examples/Segmentation/Segmentation_demo.ipynb`` instead of
the CLI flow below.

Prerequisites
-------------

- Python 3.10 or newer
- a working KonfAI installation
- a terminal in the repository root

Install KonfAI
--------------

From PyPI:

.. code-block:: bash

   python -m pip install "konfai[imaging]"

From source:

.. code-block:: bash

   git clone https://github.com/vboussot/KonfAI.git
   cd KonfAI
   python -m pip install -e ".[imaging]"

.. note::

   The ``[imaging]`` extra pulls in SimpleITK, which is **required to read the
   ``.mha`` demo data** below. Plain ``pip install konfai`` will train-fail with
   an import error on the first run.

Verify the install:

.. code-block:: bash

   konfai --help

**Success signal:** ``konfai --help`` prints the CLI help. If it fails with an
import error instead, revisit the install commands above.

What to keep in mind before you start:

- run the commands from the directory that contains the YAML files
- `Config.yml` is the training workflow
- `Prediction.yml` writes model outputs to disk
- `Evaluation.yml` compares those saved outputs against references

.. warning::

   **KonfAI rewrites your config.** After any run, ``Config.yml`` will contain
   the resolved default values that KonfAI materialised (this is expected and is
   how a run leaves a fully-reproducible config on disk). ``None`` is written as
   the literal string ``None``. If you see a git diff on your YAML after a run,
   nothing is broken — see :doc:`concepts/configuration`.

Download the demo dataset
-------------------------

Run these commands from the repository root. **From here on, every command
assumes your working directory is the example directory itself
(``examples/Segmentation``)** — local YAML references and Python modules
resolve relative to it.

.. code-block:: bash

   cd examples/Segmentation

.. code-block:: bash

   python -m pip install -U "huggingface_hub[cli]"
   hf download VBoussot/konfai-demo \
     --repo-type dataset \
     --include "Segmentation/**" \
     --local-dir Dataset
   mv Dataset/Segmentation/* Dataset/
   rmdir Dataset/Segmentation
   rm -rf Dataset/.cache

After the download, the example expects this layout:

.. code-block:: text

   examples/Segmentation/
   ├── Dataset/
   │   ├── 1PC006/
   │   │   ├── CT.mha
   │   │   └── SEG.mha
   │   └── ...
   ├── Config.yml
   ├── Prediction.yml
   └── Evaluation.yml

**Success signal:** your ``Dataset/`` tree matches the layout above — one
directory per case, each containing ``CT.mha`` and ``SEG.mha``.

Train a baseline
----------------

At this stage, KonfAI reads ``Config.yml`` and builds a ``Trainer`` object from
it.

.. code-block:: bash

   konfai TRAIN -y --gpu 0 --config Config.yml

If you do not have a GPU available, use ``--cpu 1`` instead of ``--gpu 0``.

**Success signal:** training creates, at minimum:

- ``Checkpoints/SEG_BASELINE/``
- ``Statistics/SEG_BASELINE/``

This is the most important first milestone. If these folders are created, your
installation, dataset layout, and training entrypoint are all working together.
If you only want a first success today, it is reasonable to stop here.

Run prediction
--------------

Use one checkpoint from ``Checkpoints/SEG_BASELINE``. ``Prediction.yml`` defines
which outputs are written and under which group names.

First list what training produced, then substitute a real filename for
``<checkpoint>.pt``:

.. code-block:: bash

   ls Checkpoints/SEG_BASELINE/

.. code-block:: bash

   konfai PREDICTION -y --gpu 0 --config Prediction.yml \
     --models Checkpoints/SEG_BASELINE/<checkpoint>.pt

Pass several checkpoints to ``--models`` to run an ensemble.

**Success signal:** prediction writes:

- ``Predictions/SEG_BASELINE/``

Run evaluation
--------------

``Evaluation.yml`` does not run the model again. It compares saved prediction
groups against reference groups on disk.

.. code-block:: bash

   konfai EVALUATION -y --config Evaluation.yml

**Success signal:** evaluation writes:

- ``Evaluations/SEG_BASELINE/Metric_TRAIN.json``

What to inspect
---------------

- The copied YAML files inside ``Statistics/``, ``Predictions/``, and
  ``Evaluations/`` for reproducibility
- The prediction dataset written under ``Predictions/SEG_BASELINE/Dataset/``
- The aggregated metrics in ``Metric_TRAIN.json``

Success checklist
-----------------

You can consider the onboarding successful if:

- ``konfai --help`` works
- the demo dataset matches the expected folder layout
- ``konfai TRAIN`` creates ``Checkpoints/SEG_BASELINE/`` and ``Statistics/SEG_BASELINE/``
- ``konfai PREDICTION`` creates ``Predictions/SEG_BASELINE/``
- ``konfai EVALUATION`` creates ``Evaluations/SEG_BASELINE/Metric_TRAIN.json``

Common first issues
-------------------

- **``--gpu`` rejects your device id**

  ``konfai`` validates GPU ids against ``CUDA_VISIBLE_DEVICES``. Use ``--cpu``
  if no GPU is available, or check the visible devices with a small PyTorch
  snippet.

- **The command asks whether it should overwrite an existing run**

  Add ``-y`` to skip the interactive confirmation.

- **Dataset groups do not match the YAML**

  KonfAI expects the group names used in ``groups_src`` to exist on disk. In
  this example that means ``CT.mha`` and ``SEG.mha`` for every case directory.

- **The workflow runs, but evaluation cannot find predictions**

  Check that ``Prediction.yml`` and ``Evaluation.yml`` use the same
  ``train_name`` and that evaluation points to the correct prediction dataset.

- **A metric or output group name is rejected**

  Output names in ``outputs_criterions`` and ``outputs_dataset`` must match real
  model module paths. Start from the shipped examples before introducing custom
  names.

Next steps
----------

- :doc:`concepts/index` — understand the machinery you just ran: config
  reflection, lazy patch-based datasets, and the model graph.
- :doc:`config_guide/index` — the full key-by-key guide to ``Config.yml``,
  ``Prediction.yml``, and ``Evaluation.yml``.
- :doc:`examples/index` — the shipped segmentation and synthesis workflows to
  adapt to your own data.

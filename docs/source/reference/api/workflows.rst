Workflows API
=============

Two layers sit behind the ``konfai`` CLI. :mod:`konfai.api` is the one to call
from a script or a notebook: five functions with structured results, described in
:doc:`../../usage/python-workflows`. Below it, each workflow's entrypoint and root
class, which the CLI uses and which :mod:`konfai.api` builds on.

The workflow API
----------------

.. currentmodule:: konfai.api

.. autofunction:: transform
   :no-index:

.. autofunction:: plan_transform
   :no-index:

.. autofunction:: evaluate
   :no-index:

.. autofunction:: predict
   :no-index:

.. autofunction:: train
   :no-index:

.. autoclass:: TransformResult
   :members:
   :no-index:

.. autoclass:: EvaluationResult
   :members:
   :no-index:

Training
--------

.. currentmodule:: konfai.trainer

.. autofunction:: train
   :no-index:

.. autoclass:: Trainer
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: EarlyStopping
   :members:
   :show-inheritance:
   :no-index:

Prediction
----------

.. currentmodule:: konfai.predictor

.. autofunction:: predict
   :no-index:

.. autoclass:: Predictor
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: OutputDataset
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: OutputDatasetLoader
   :members:
   :show-inheritance:
   :no-index:

Evaluation
----------

.. currentmodule:: konfai.evaluator

.. autofunction:: evaluate
   :no-index:

.. autoclass:: Evaluator
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: Statistics
   :members:
   :show-inheritance:
   :no-index:

Transform
---------

.. currentmodule:: konfai.transformer

.. autofunction:: transform
   :no-index:

.. autofunction:: build_transform
   :no-index:

.. autofunction:: plan_transform
   :no-index:

.. autoclass:: Transformer
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: TransformPlan
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: TransformPlanEntry
   :members:
   :show-inheritance:
   :no-index:

See also
--------

- :doc:`configuration`
- :doc:`data`
- :doc:`models`

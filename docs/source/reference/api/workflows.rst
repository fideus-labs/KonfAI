Workflows API
=============

The low-level KonfAI workflows are exposed through four entrypoint functions
and four root classes:

- training
- prediction
- evaluation
- transform (no model)

These are the main Python APIs behind the ``konfai`` CLI.

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

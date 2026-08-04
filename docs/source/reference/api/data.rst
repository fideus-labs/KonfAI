Data API
========

KonfAI datasets are built from group definitions, transforms, augmentations, and
optional patching strategies.

Dataset configuration objects
-----------------------------

.. currentmodule:: konfai.data.data_manager

.. autoclass:: DataTrain
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: DataPrediction
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: DataMetric
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: DataTransform
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: DatasetIter
   :members:
   :show-inheritance:
   :no-index:

Patching
--------

.. currentmodule:: konfai.data.patching

.. autoclass:: DatasetPatch
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: ModelPatch
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: DatasetManager
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: Accumulator
   :members:
   :show-inheritance:
   :no-index:

Transforms and metadata
-----------------------

.. currentmodule:: konfai.data.transform

.. autoclass:: Transform
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: TransformInverse
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: TransformLoader
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: Clip
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: Normalize
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: Standardize
   :members:
   :show-inheritance:
   :no-index:

Dataset utilities
-----------------

.. currentmodule:: konfai.utils.dataset

.. autoclass:: Attribute
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: Dataset
   :members:
   :show-inheritance:
   :no-index:

.. autofunction:: data_to_image
   :no-index:
.. autofunction:: image_to_data
   :no-index:
.. autofunction:: get_infos
   :no-index:

See also
--------

- :doc:`/concepts/datasets`
- :doc:`/examples/segmentation`
- :doc:`/examples/synthesis`

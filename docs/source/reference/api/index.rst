API reference
=============

Two levels: curated pages for the public building blocks most developers touch,
and a full module reference generated from the package modules. The workflow
callables (``konfai.transform``, ``train_model`` and the rest) and the
``konfai_apps`` runners are documented where they are used, on
:doc:`/usage/python-api`; the extension base classes are on
:doc:`/usage/custom-models`.

Configuration API
-----------------

KonfAI builds workflows from YAML by combining configuration decorators,
constructor signatures, and recursive object instantiation.

.. currentmodule:: konfai.utils.config

.. autoclass:: Config
   :members:
   :show-inheritance:
   :no-index:

.. autofunction:: config
   :no-index:

.. autofunction:: apply_config
   :no-index:

Related runtime helpers
~~~~~~~~~~~~~~~~~~~~~~~

These helpers expose the active workflow context after the CLI wrapper has set
up the environment:

.. currentmodule:: konfai

.. autofunction:: config_file
   :no-index:
.. autofunction:: konfai_root
   :no-index:
.. autofunction:: konfai_state
   :no-index:
.. autofunction:: checkpoints_directory
   :no-index:
.. autofunction:: predictions_directory
   :no-index:
.. autofunction:: evaluations_directory
   :no-index:
.. autofunction:: statistics_directory
   :no-index:

.. autofunction:: transforms_directory
   :no-index:

Data API
--------

KonfAI datasets are built from group definitions, transforms, augmentations, and
optional patching strategies.

Dataset configuration objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~

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

.. currentmodule:: konfai.data.materialize

.. autoclass:: CaseMaterializer
   :members:
   :show-inheritance:
   :no-index:

.. currentmodule:: konfai.data.patching

.. autoclass:: Accumulator
   :members:
   :show-inheritance:
   :no-index:

Transforms and metadata
~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~

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

Models API
----------

KonfAI model graphs are configured through loaders, routed network containers,
criteria, and reusable blocks.

Model graph and loaders
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: konfai.network.network

.. autoclass:: ModelLoader
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: Model
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: Network
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: ModuleArgsDict
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: OptimizerLoader
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: LRSchedulersLoader
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: TargetCriterionsLoader
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: CriterionsLoader
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: Measure
   :members:
   :show-inheritance:
   :no-index:

Building blocks
~~~~~~~~~~~~~~~

.. currentmodule:: konfai.network.blocks

.. autoclass:: BlockConfig
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: ConvBlock
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: ResBlock
   :members:
   :show-inheritance:
   :no-index:

.. autofunction:: get_torch_module
   :no-index:
.. autofunction:: get_norm
   :no-index:

Loss-weight schedulers
~~~~~~~~~~~~~~~~~~~~~~

These derive from ``Scheduler`` and expose ``get_value()``; a criterion's
``schedulers`` block resolves against this module only.

.. currentmodule:: konfai.metric.schedulers

.. autoclass:: Scheduler
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: Constant
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: CosineAnnealing
   :members:
   :show-inheritance:
   :no-index:

Learning-rate schedulers
~~~~~~~~~~~~~~~~~~~~~~~~

These take the optimizer as their first argument and are resolved against
``torch.optim.lr_scheduler`` first, then this module. They are not interchangeable
with the loss-weight schedulers above.

.. autoclass:: Warmup
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: PolyLRScheduler
   :members:
   :show-inheritance:
   :no-index:

Full module reference
---------------------

.. toctree::
   :maxdepth: 1

   /modules

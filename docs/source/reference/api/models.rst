Models API
==========

KonfAI model graphs are configured through loaders, routed network containers,
criteria, and reusable blocks.

Model graph and loaders
-----------------------

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
---------------

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
----------------------

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
------------------------

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

See also
--------

- :doc:`/concepts/model-graph`
- :doc:`/usage/custom-models`

konfai.utils package
====================

Submodules
----------

konfai.utils.ITK module
-----------------------

.. automodule:: konfai.utils.ITK
   :members:
   :show-inheritance:
   :undoc-members:

konfai.utils.config module
--------------------------

.. automodule:: konfai.utils.config
   :members:
   :show-inheritance:
   :undoc-members:

konfai.utils.dataset package
----------------------------

.. automodule:: konfai.utils.dataset.attribute
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: konfai.utils.dataset.statistics
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: konfai.utils.dataset.staging
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: konfai.utils.dataset.stream
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: konfai.utils.dataset.raw_block
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: konfai.utils.dataset.abstract
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: konfai.utils.dataset.h5
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: konfai.utils.dataset.sitk_file
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: konfai.utils.dataset.ome_zarr_file
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: konfai.utils.dataset.dicom_file
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: konfai.utils.dataset.itk_transform_file
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: konfai.utils.dataset.backend
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: konfai.utils.dataset.landmarks
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: konfai.utils.dataset.core
   :members:
   :show-inheritance:
   :undoc-members:

konfai.utils.errors module
--------------------------

The ``KonfAIError`` taxonomy: every designed refusal a workflow raises. The
Python workflows (:doc:`usage/python-workflows`) let these propagate; only the
CLI catches them.

.. automodule:: konfai.utils.errors
   :members:
   :show-inheritance:
   :undoc-members:

konfai.utils.pretrained module
------------------------------

The pretrained-weights bridge: pairs weighted leaves in forward-execution order,
so a checkpoint from another framework (MONAI, torchvision, nnU-Net) seeds a
KonfAI graph without a key map. ``PretrainedFrom`` is the ``Model.pretrained_from``
config entry; ``transfer_weights_by_execution_order`` is the underlying transfer.
It fills every target tensor or raises: a partial load is never reported as success.

.. automodule:: konfai.utils.pretrained
   :members:
   :show-inheritance:
   :undoc-members:

konfai.utils.utils module
-------------------------

.. automodule:: konfai.utils.utils
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: konfai.utils
   :members:
   :show-inheritance:
   :undoc-members:

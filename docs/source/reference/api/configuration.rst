Configuration API
=================

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
-----------------------

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

See also
--------

- :doc:`/concepts/configuration`
- :doc:`/config_guide/index`

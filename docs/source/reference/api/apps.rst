Apps API
========

KonfAI Apps package stable workflows behind a simpler interface than the raw
``TRAIN`` / ``PREDICTION`` / ``EVALUATION`` commands.

Local and remote app runners
----------------------------

.. currentmodule:: konfai_apps

.. autoclass:: KonfAIApp
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: KonfAIAppClient
   :members:
   :show-inheritance:
   :no-index:

Remote server helpers
---------------------

.. currentmodule:: konfai

.. autoclass:: RemoteServer
   :members:
   :show-inheritance:
   :no-index:

.. autofunction:: check_server
   :no-index:
.. autofunction:: get_available_devices
   :no-index:
.. autofunction:: get_ram
   :no-index:
.. autofunction:: get_vram
   :no-index:

See also
--------

- :doc:`/usage/apps`
- :doc:`/reference/app-server-api`

How-to guides
=============

These guides show how to extend, package, and deploy KonfAI. Running the
train → predict → evaluate workflows is covered by the
:doc:`configuration reference <../config_guide/index>` — each of its pages
starts with the commands that run its workflow.

Start here
----------

Read :doc:`custom-models` when the built-in models, transforms,
augmentations, or losses are not enough. :doc:`apps` and :doc:`docker` cover
packaging and deployment; skip them until you want to ship or serve a model.
:doc:`mcp` drives the whole framework — and published apps — from an LLM agent
through the Model Context Protocol.

.. toctree::
   :maxdepth: 1

   custom-models
   apps
   mcp
   docker

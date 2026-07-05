Configuration reference
=======================

These pages document the **canonical YAML structure** used by KonfAI's three
main workflows. Each page starts with the commands that run its workflow.

They focus on the fields that are stable and clearly visible in the codebase and
the shipped examples. For built-in models, transforms, and metrics, the exact
available parameters still depend on the selected classpath.

**Start here:** read :doc:`training` first — it introduces the structures
(``Model``, ``Dataset``, ``outputs_criterions``) that the other two files
reuse. Then read :doc:`prediction` and :doc:`evaluation` as you reach those
workflows.

.. toctree::
   :maxdepth: 1

   training
   prediction
   evaluation

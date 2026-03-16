uc\_example
============

``code/uc_example.py`` is a self-contained script (not a library module) that
demonstrates solving a **security-constrained unit commitment (SCUC)** problem
with EGRET.

What it does
------------

1. Loads the 24-bus IEEE RTS test case bundled with EGRET
   (``egret/models/tests/uc_test_instances/test_scuc_masked.json``).
2. Calls ``egret.models.unit_commitment.solve_unit_commitment`` with the Gurobi
   solver (1% MIP gap, 5 minute time limit).
3. Prints the total system cost and a dispatch summary for the first few
   generators.
4. Writes the full solved ``ModelData`` object to
   ``results/uc_test_scuc_masked.json``.

Running the script
------------------

.. code-block:: bash

   uv run python code/uc_example.py

The script is meant to be run as a one-shot demonstration.  It produces no
importable symbols and is therefore documented here rather than via autodoc.

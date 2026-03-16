plot\_sensitivity
=================

Plot LMP sensitivity results saved by :mod:`sensitivity_analysis` or
:mod:`finite_difference_sensitivity`.

Accepts one or two JSON result files.  When two files are given (one KKT, one
finite-difference), they are rendered side-by-side for easy comparison.

Usage
-----

.. code-block:: bash

   # Single file
   uv run python code/plot_sensitivity.py results/sensitivity_kkt_<case>.json

   # Side-by-side comparison
   uv run python code/plot_sensitivity.py \
       results/sensitivity_kkt_<case>.json \
       results/sensitivity_fd_<case>.json

   # Custom output path
   uv run python code/plot_sensitivity.py results/sensitivity_kkt_<case>.json --out fig.pdf

API
---

.. autofunction:: plot_sensitivity.load_results
.. autofunction:: plot_sensitivity.plot_sensitivity

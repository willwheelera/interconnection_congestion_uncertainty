finite\_difference\_sensitivity
================================

LMP sensitivity via finite-difference perturbation of DCOPF re-solves.

This module complements :mod:`sensitivity_analysis` by providing a purely
numerical approach.  It is the only method that reveals cross-basis LMP changes
for LP (linear-cost) problems, since the analytical KKT sensitivity is
identically zero within any single LP basis.

.. automodule:: finite_difference_sensitivity
   :members:
   :undoc-members:
   :show-inheritance:

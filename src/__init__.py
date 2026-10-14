"""AMPS — one subpackage per workflow stage.

`src` is the importable root of the project, so every first-party import is absolute and
fully qualified:

    from src.cases import registry
    from src.sampling.problem import TargetProblem

The `src.` prefix is what makes those imports unambiguous.  Without it, a module run as a
script (`python src/figures/style.py`) puts its own directory first on `sys.path`, and a
sibling file could shadow the stage package of the same name.

Stages: `cases` (define a testbed), `model` (one configuration -> duals), `sweeps` (many
configurations -> caches), `sampling` (estimate the tail), `figures` (draw it).

Nothing is imported here: a stage pulls in solvers and plotting libraries, and importing
one stage should not pay for the others.
"""

__all__ = [
    "cases",
    "ercot",
    "figures",
    "model",
    "paths",
    "sampling",
    "sweeps",
]

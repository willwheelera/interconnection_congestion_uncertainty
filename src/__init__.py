"""AMPS — one subpackage per workflow stage.

`src` is the importable root of the project, so every first-party import is absolute and
fully qualified:

    from src.cases import registry
    from src.descriptors.compute import BASES

The `src.` prefix is what makes those imports unambiguous.  Without it, a module run as a
script (`python src/figures/descriptors.py`) puts its own directory first on `sys.path`,
and a sibling file shadows the stage package of the same name: `src/figures/descriptors.py`
answers `import descriptors`, and `descriptors.compute` then fails to resolve.

Stages: `cases` (define a testbed), `model` (one configuration -> duals), `sweeps` (many
configurations -> caches), `sampling` and `descriptors` (estimate the tail), `figures`
(draw it).  Alongside them: `utils`, `diagnostics`, `sensitivity`, `ercot`, `_archive`.

Nothing is imported here: a stage pulls in solvers and plotting libraries, and importing
one stage should not pay for the others.
"""

__all__ = [
    "cases",
    "descriptors",
    "diagnostics",
    "ercot",
    "figures",
    "model",
    "paths",
    "sampling",
    "sensitivity",
    "sweeps",
    "utils",
]

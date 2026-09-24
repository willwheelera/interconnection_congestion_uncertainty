"""The stored definition of a nominal law: its field and coupling, written once.

A law is fully defined by the pair (h, J) over the ngen availability bits, and for the
structured laws h is *solved* rather than written down -- `mrf.calibrate_field`
enumerates all 2^ngen configurations to find the field that reproduces the queue's build
probabilities under a given coupling.  That is a one-time computation whose result is a
sweep *input* -- every sweep, sampler, and plot reads it to weight configurations -- so it
is stored in the tracked per-case directory `data/<case>/laws/<name>.json`, alongside the
relief set and the screened contingency set, not in the regenerable (git-ignored) `results/`.

The file carries the whole definition together -- h, J, the marginals the law was built
at and the ones it actually realizes, and the group structure and coupling strength it
came from -- so a law can be read and checked without running anything.
"""

import json

import numpy as np

from src.sampling.mrf import MarkovRandomField


def law_file(case, name):
    """Where a case's named law is stored: tracked `data/<case>/laws/<name>.json`.

    A law is a sweep input read to weight configurations, so it is tracked like the relief
    set and the screened contingency set rather than living in the git-ignored `results/`
    tree; each case's laws sit in a `laws/` subdirectory of its own `data/<case>/` directory,
    beside the case's queue and screened set.  The same holds for a `sample_only` case, whose
    cluster job reads the law from here.
    """
    return case.data_dir / "laws" / f"{name}.json"


def save_law(case, name, mrf, record):
    """Write h, J, and the build record as one file; returns the path."""
    path = law_file(case, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"case": case.name, "name": name, "ngen": mrf.ngen,
                                "h": mrf.h.tolist(), "J": mrf.J.tolist(), **record},
                               indent=1))
    return path


def load_law(case, name):
    """The stored law as an MRF; an unbuilt law is an error, not an on-the-fly rebuild."""
    path = law_file(case, name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path}: {case.name} has no stored {name} law -- sampling/build_laws.py "
            f"builds and stores a case's laws")
    d = json.loads(path.read_text())
    return MarkovRandomField(np.array(d["h"]), np.array(d["J"]))

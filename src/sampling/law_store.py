"""The stored definition of a nominal law: its field and coupling, written once.

A law is fully defined by the pair (h, J) over the ngen availability bits, and for the
structured laws h is *solved* rather than written down -- `mrf.calibrate_field`
enumerates all 2^ngen configurations to find the field that reproduces the queue's build
probabilities under a given coupling.  That is a one-time computation, so its result is
stored alongside the coupling it belongs to: `build_laws.py` writes
`results/<case>/laws/<name>.json`, and every plot, sweep, and sampler reads the law from
there instead of re-deriving it.

The file carries the whole definition together -- h, J, the marginals the law was built
at and the ones it actually realizes, and the group structure and coupling strength it
came from -- so a law can be read and checked without running anything.
"""

import json

import numpy as np

from src.sampling.mrf import MarkovRandomField


def law_file(case, name):
    """Where a case's named law is stored."""
    return case.results_dir / "laws" / f"{name}.json"


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

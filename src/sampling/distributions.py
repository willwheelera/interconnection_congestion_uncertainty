"""The nominal laws the runner sweeps, as MarkovRandomField instances.

A law is just an (h, J) pair over the ngen availability bits (mrf.py); the samplers
see it only through Problem's generic weight vector.  This factory names the
parametrizations under comparison:

  - bernoulli : independent per generator (J = 0).
  - mrf_a     : weak positive (ferromagnetic) coupling -- generators co-vary.
  - mrf_b     : weak zero-mean, mixed-sign coupling -- mild frustration.
  - mrf_c     : stronger zero-mean, mixed-sign coupling -- strong frustration / multimodal.
  - mrf_cobuild    : a technology's candidates build together or not at all.
  - mrf_substitute : each technology's near-certain candidate crowds out the rest of its
                     class, which in turn build together.

Every law shares the same per-generator marginals, so they differ only in coupling.
A scalar mean p is spread into ngen marginals by a Beta(mean=p, var=P_STD**2), which
stays in [0, 1] for any mean; a per-generator vector p (the calibrated Texas queue) is
used directly without spreading.  The spread P_STD is a shared experiment knob, not tied
to the law name, so a change in the tail is attributable to coupling alone.

`make_dist` constructs a law from its parameters; only `build_laws.py` calls it.  Every
other entry point takes `case_dist`, which reads the case's stored (h, J) off disk --
the field of a structured law is solved by enumeration and is far too expensive to
rebuild each time a figure is drawn.

mrf_a/b/c couple every pair by an unstructured Gaussian draw, which leaves the mu
distribution essentially unchanged: the coupling is spread thin over all pairs and does
not line up with the handful of candidates a congested element responds to.  The two
structured laws instead couple candidates *within a technology group* strongly and with
a definite sign, which is what moves the tail (see diagnostics/check_law_coupling.py).
Their field is solved for rather than mean-field compensated, so their marginals equal
the Bernoulli marginals to solver precision at any coupling strength.
"""

import numpy as np

from src.sampling.law_store import load_law
from src.sampling.mrf import MarkovRandomField, calibrate_field, default_mrf

DIST_NAMES = ["bernoulli", "mrf_a", "mrf_b", "mrf_c"]
COUPLED_NAMES = ["mrf_cobuild", "mrf_substitute"]
DIST_LABEL = {                      # display name of every law, shared by the figures
    "bernoulli": "Bernoulli", "mrf_a": "MRF-A", "mrf_b": "MRF-B", "mrf_c": "MRF-C",
    "mrf_cobuild": "MRF co-build", "mrf_substitute": "MRF substitution",
}
P_STD = 0.15                        # marginal spread shared by every law
COUPLINGS = {                       # (mean, std) of the Gaussian pair coefficients J
    "mrf_a": (0.04, 0.02),          # positive mean -> ferromagnetic, generators co-vary
    "mrf_b": (0.0, 0.08),           # zero mean -> mixed-sign, mild frustration
    "mrf_c": (0.0, 0.15),           # zero mean -> mixed-sign, strong frustration
}
GROUP_COUPLING = 2.0                # pair coefficient magnitude of the structured laws


def beta_params(mean, var):
    """Beta(a, b) shape parameters matching a target mean and variance (moment match)."""
    c = mean * (1.0 - mean) / var - 1.0
    return mean * c, (1.0 - mean) * c


def pairwise_mrf(ngen, p, pvar, coupling, coupling_std, seed=0):
    """Pairwise MRF: per-generator marginals p, Gaussian pair coefficients J.

    A scalar p is spread into ngen marginals by a Beta(mean=p, var=pvar); a vector p is
    used as given.  The single field is coupling-compensated so the marginals stay near p
    as the mean coupling grows; coupling = coupling_std = 0 is an independent law at p.
    """
    rng = np.random.default_rng(seed)
    if np.ndim(p) > 0:
        p = np.asarray(p, float)
    else:
        p = rng.beta(*beta_params(p, pvar), size=ngen)
    h = np.log(p / (1.0 - p)) - (ngen - 1) * coupling * p.mean()
    J = np.triu(rng.normal(coupling, coupling_std, (ngen, ngen)), 1)
    return MarkovRandomField(h, J + J.T)


def marginal_vector(ngen, p, pvar, seed=0):
    """The per-generator marginals every law shares: a vector p as given, a scalar spread."""
    if np.ndim(p) > 0:
        return np.asarray(p, float)
    return np.random.default_rng(seed).beta(*beta_params(p, pvar), size=ngen)


def cobuild_coupling(groups, ngen, c):
    """+c inside each group: a technology's candidates build together or not at all."""
    J = np.zeros((ngen, ngen))
    for g in groups:
        J[np.ix_(g, g)] = c
    np.fill_diagonal(J, 0.0)
    return J


def substitute_coupling(groups, p, ngen, c):
    """Each group's most probable candidate crowds out (-c) the rest, which co-vary (+c).

    The anchor is the near-certain project of its class; it takes the interconnection
    capacity its class competes for, so the speculative projects clear as a block only
    when it does not.
    """
    J = np.zeros((ngen, ngen))
    for g in groups:
        anchor = max(g, key=lambda i: p[i])
        rest = [i for i in g if i != anchor]
        J[np.ix_(rest, rest)] = c
        J[anchor, rest] = J[rest, anchor] = -c
    np.fill_diagonal(J, 0.0)
    return J


def structured_mrf(name, ngen, p, groups, c=GROUP_COUPLING):
    """A group-structured law at exactly the marginals p; groups are the queue's fuels."""
    if groups is None:
        raise ValueError(f"{name} coupling is defined within technology groups, and this "
                         f"case has no queue to read fuels from")
    J = (cobuild_coupling(groups, ngen, c) if name == "mrf_cobuild"
         else substitute_coupling(groups, p, ngen, c))
    return MarkovRandomField(calibrate_field(J, p), J)


def case_dist(case, name="bernoulli"):
    """The named nominal law of a registered case, read from its stored (h, J).

    Built and written by `build_laws.py`; nothing downstream reconstructs a law.
    """
    return load_law(case, name)


def nominal_law(case):
    """The law a case's configurations carry probability under, as the studies weight them.

    A queue case uses its calibrated per-candidate build probabilities; a case with no
    queue falls back to the field the samplers are scored under, which is what `Problem`
    assumes when no law is passed.  Anything that needs the *probability* of a
    configuration (its weight, or the modal configuration) reads it from here, so a
    descriptor's base configuration and the target it is scored against are the same law.
    """
    if case.candidates_file is None:
        return default_mrf(case.ngen)
    return make_dist("bernoulli", case.ngen, case.build_p())


def make_dist(name, ngen, p, p_std=P_STD, groups=None):
    """One named nominal law over ngen bits at marginal mean p, marginal spread p_std.

    p_std (the marginal standard deviation) is shared across the laws -- they are compared
    at identical marginals and differ only in the coupling keyed by name.  A vector p
    (Texas queue) ignores p_std and is used directly.  `groups` are the candidate index
    blocks the structured laws couple within (one block per technology).
    """
    if name in COUPLED_NAMES:
        return structured_mrf(name, ngen, marginal_vector(ngen, p, p_std ** 2), groups)
    coupling, coupling_std = (0.0, 0.0) if name == "bernoulli" else COUPLINGS[name]
    return pairwise_mrf(ngen, p, p_std ** 2, coupling, coupling_std)

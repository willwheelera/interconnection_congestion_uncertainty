"""The nominal laws the runner sweeps, as MarkovRandomField instances.

A law is just an (h, J) pair over the ngen availability bits (mrf.py); the samplers
see it only through Problem's generic weight vector.  This factory names the
parametrizations under comparison:

  - bernoulli : independent per generator (J = 0).
  - mrf_a     : weak positive (ferromagnetic) coupling -- generators co-vary.
  - mrf_b     : weak zero-mean, mixed-sign coupling -- mild frustration.
  - mrf_c     : stronger zero-mean, mixed-sign coupling -- strong frustration / multimodal.

Every law shares the same per-generator marginals, so the four differ only in coupling.
A scalar mean p is spread into ngen marginals by a Beta(mean=p, var=p_std**2), which
stays in [0, 1] for any mean; a per-generator vector p (the calibrated Texas queue) is
used directly without spreading.  The spread p_std is a shared experiment knob, not tied
to the law name, so a change in the tail is attributable to coupling alone.
"""

import numpy as np

from sampling.mrf import MarkovRandomField

DIST_NAMES = ["bernoulli", "mrf_a", "mrf_b", "mrf_c"]
COUPLINGS = {                       # (mean, std) of the Gaussian pair coefficients J
    "mrf_a": (0.04, 0.02),          # positive mean -> ferromagnetic, generators co-vary
    "mrf_b": (0.0, 0.08),           # zero mean -> mixed-sign, mild frustration
    "mrf_c": (0.0, 0.15),           # zero mean -> mixed-sign, strong frustration
}


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


def case_dist(case, name="bernoulli", p_std=0.15):
    """The named nominal law at a registered case's own build-probability marginals.

    A queue case supplies calibrated per-candidate probabilities, used as given; a case
    without a queue supplies a scalar mean that make_dist spreads across its generators.
    """
    return make_dist(name, case.ngen, case.nominal_p(), p_std)


def make_dist(name, ngen, p, p_std=0.15):
    """One named nominal law over ngen bits at marginal mean p, marginal spread p_std.

    p_std (the marginal standard deviation) is shared across the laws -- they are compared
    at identical marginals and differ only in the coupling keyed by name.  A vector p
    (Texas queue) ignores p_std and is used directly.
    """
    coupling, coupling_std = (0.0, 0.0) if name == "bernoulli" else COUPLINGS[name]
    return pairwise_mrf(ngen, p, p_std ** 2, coupling, coupling_std)

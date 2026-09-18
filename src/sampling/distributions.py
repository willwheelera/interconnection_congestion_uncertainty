"""The nominal laws the runner sweeps, as MarkovRandomField instances.

A law is just an (h, J) pair over the ngen availability bits (mrf.py); the samplers
see it only through Problem's generic weight vector.  This factory names the
parametrizations under comparison:

  - bernoulli : independent per generator (J = 0).
  - mrf_a     : zero-mean, mixed-sign coupling on every pair -- unstructured frustration.
  - mrf_cobuild    : a technology's candidates build together or not at all.
  - mrf_substitute : each technology's near-certain candidate crowds out the rest of its
                     class, which in turn build together.

Every law is built in two steps, marginals first and coupling second.  `marginal_vector`
fixes the per-generator marginals -- a vector p (the calibrated Texas queue) as given, a
scalar mean p spread into ngen of them by a Beta(mean=p, var=P_STD**2) -- and every law
of a case is built on that same vector.  The coupling keyed by the law's name is then
laid on top, and the field is *solved* for the marginals under it
(`mrf.calibrate_field`), so they match to solver precision however strong the coupling
is; a mean-field correction would only be first order and would let them drift.  The
spread P_STD is a shared experiment knob, not tied to the law name, so a change in the
tail is attributable to coupling alone.

`make_dist` constructs a law from its parameters; only `build_laws.py` calls it.  Every
other entry point takes `case_dist`, which reads the case's stored (h, J) off disk --
the field is solved by enumeration and is far too expensive to rebuild each time a
figure is drawn.

mrf_a couples every pair by an unstructured Gaussian draw, which leaves the mu
distribution essentially unchanged: the coupling is spread thin over all pairs and does
not line up with the handful of candidates a congested element responds to.  The two
structured laws in STRUCTURED_NAMES instead couple candidates *within a technology
group* strongly and with a definite sign, which is what moves the tail (see
diagnostics/check_law_coupling.py).  Their groups come from the case's queue, so they
are the only laws a case without a queue cannot build.
"""

import numpy as np

from src.sampling.law_store import load_law
from src.sampling.mrf import MarkovRandomField, calibrate_field

DIST_NAMES = ["bernoulli", "mrf_b", "mrf_cobuild", "mrf_substitute"]
STRUCTURED_NAMES = ["mrf_cobuild", "mrf_substitute"]   # couple within the queue's fuels
DIST_LABEL = {                      # display name of every law, shared by the figures
    "bernoulli": "Bernoulli", "mrf_a": "MRF-A", "mrf_b": "MRF-B",
    "mrf_cobuild": "MRF co", "mrf_substitute": "MRF sub",
}
P_STD = 0.15                        # marginal spread shared by every law
COUPLINGS = {                       # (mean, std) of the Gaussian pair coefficients J
    "bernoulli": (0.0, 0.0),
    "mrf_a": (0.0, 0.15),           # zero mean -> mixed-sign, strong frustration
    "mrf_b": (0.0, 2.00),           # zero mean -> mixed-sign, strong frustration
}
GROUP_COUPLING = 2.0                # pair coefficient magnitude of the structured laws


def beta_params(mean, var):
    """Beta(a, b) shape parameters matching a target mean and variance (moment match)."""
    c = mean * (1.0 - mean) / var - 1.0
    return mean * c, (1.0 - mean) * c


def marginal_vector(ngen, p, pvar, seed=0):
    """The per-generator marginals every law is built on: a vector p as given, a scalar spread.

    The one place marginals are decided.  A case that supplies calibrated per-candidate
    probabilities passes them straight through; a case that supplies only a scalar mean
    gets ngen of them drawn from a Beta(mean=p, var=pvar), which stays in [0, 1] for any
    mean.  Every coupling below is then built on top of the vector this returns.
    """
    if np.ndim(p) > 0:
        return np.asarray(p, float)
    return np.random.default_rng(seed).beta(*beta_params(p, pvar), size=ngen)


def pairwise_mrf(ngen, p, coupling, coupling_std, seed=0):
    """Unstructured pairwise MRF at exactly the marginals p, Gaussian pair coefficients J.

    The field is solved for p under the drawn J, so the law differs from Bernoulli in
    correlation alone; coupling = coupling_std = 0 is an independent law at p.
    """
    J = np.triu(np.random.default_rng(seed).normal(coupling, coupling_std, (ngen, ngen)), 1)
    J = J + J.T
    return MarkovRandomField(calibrate_field(J, p), J)


def block_c(c, m, ref_block=None):
    """Within-block coefficient, optionally scaled so (m-1)*c matches a reference block.

    A flat +c across a large fuel block saturates it to all-on/all-off (the pair term
    grows like c*(k choose 2) in the number on), so a 95-candidate block is nothing like
    a 5-candidate one at the same c.  Holding the per-node coupling (m-1)*c at a small
    reference block's value (Curie-Weiss normalization) keeps the induced correlations in
    the range the enumerated cases were built at.  ref_block=None keeps the flat c, the
    behavior the small cases rely on.
    """
    if ref_block is None or m <= 1:
        return c
    return c * (ref_block - 1) / (m - 1)


def cobuild_coupling(groups, ngen, c, ref_block=None):
    """+c inside each group: a technology's candidates build together or not at all."""
    J = np.zeros((ngen, ngen))
    for g in groups:
        J[np.ix_(g, g)] = block_c(c, len(g), ref_block)
    np.fill_diagonal(J, 0.0)
    return J


def substitute_coupling(groups, p, ngen, c, ref_block=None):
    """Each group's most probable candidate crowds out (-c) the rest, which co-vary (+c).

    The anchor is the near-certain project of its class; it takes the interconnection
    capacity its class competes for, so the speculative projects clear as a block only
    when it does not.
    """
    J = np.zeros((ngen, ngen))
    for g in groups:
        cg = block_c(c, len(g), ref_block)
        anchor = max(g, key=lambda i: p[i])
        rest = [i for i in g if i != anchor]
        J[np.ix_(rest, rest)] = cg
        J[anchor, rest] = J[rest, anchor] = -cg
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


def case_law_names(case, names=DIST_NAMES):
    """The requested laws a case actually has, in the order given.

    The structured laws couple within the queue's technology groups, so a case with no
    queue has none of them -- the same rule `build_laws.py` stores them under.  Every
    entry point that iterates over laws filters through here, so a mixed run over cases
    with and without queues asks each one only for what it has.
    """
    if case.candidate_groups() is not None:
        return list(names)
    return [n for n in names if n not in STRUCTURED_NAMES]


def make_dist(name, ngen, p, p_std=P_STD, groups=None):
    """One named nominal law over ngen bits: fix the marginals, then couple on top of them.

    The marginals come first and are the same for every name -- `marginal_vector` resolves
    p (a vector as given, a scalar spread by p_std) once, and the coupling keyed by name is
    built on that vector.  So the laws are comparable by construction and differ only in
    correlation.  `groups` are the candidate index blocks the structured laws couple within
    (one block per technology).
    """
    p = marginal_vector(ngen, p, p_std ** 2)
    if name in STRUCTURED_NAMES:
        return structured_mrf(name, ngen, p, groups)
    return pairwise_mrf(ngen, p, *COUPLINGS[name])

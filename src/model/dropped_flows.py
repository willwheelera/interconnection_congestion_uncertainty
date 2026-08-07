"""Post-outage flows for the contingencies a screened model left out.

A screened SCDCOPF carries flow variables only for the contingencies it kept, so checking
the ones screening dropped is not a comparison against existing variables -- it means
computing what the flow on every branch *would be* after each of those outages.  EGRET
will do that (`VirtualPTDFMatrix.calculate_PFV_delta`), but it solves a sparse system over
every bus once per contingency: 2723 network solves per configuration on ACTIVSg2000,
345 ms, an order of magnitude more than the solve being checked.

Those solves are redundant.  Taking one branch out shifts flow by a pattern fixed by the
topology, scaled by how much flow that branch was carrying:

    post_flow = base_flow + column_c * beta_c

`column_c` depends only on the network, so it is the same for every configuration and
every hour; `beta_c` is one dot product against the solved voltage angles.  EGRET
recomputes the columns every time because its virtual PTDF exists to avoid storing them.
Building them once at startup turns the check into arithmetic over an array.

Two details carry the speed.  The columns are stored one contingency per row, so the
per-contingency maximum runs along contiguous memory; and they are consumed in blocks of
`BLOCK` rows through one preallocated buffer, so the arithmetic stays in cache instead of
streaming a 71 MB temporary three times.  Measured on activsg2000_original_v12, snapshot
2025-02-19 06:00, 2723 dropped contingencies, against a 26-28 ms solve: 345 ms/config
through EGRET, 66 with the columns stored but read flat, 12-15 as written here.

What is left is per configuration and does not batch: the limit pass touches 8.7 M
(contingency, branch) pairs, each configuration brings its own flows, and checking a whole
chunk at once measures the same (35.9 ms/config against 35.3 interleaved).

Verified against `calculate_PFV_delta` on three snapshots: identical violations, and
flows agreeing to 7e-16 pu over all 8.7 M (branch, contingency) pairs.
"""

import numpy as np
import scipy.sparse as sp

# Post-contingency overflow (per unit) above which a dropped contingency counts as
# violated -- about 0.01 MW on a 100 MVA base.
TOL = 1e-4

# Contingencies whose flows are computed in one pass through the working buffer.  Small
# enough that a block stays in cache; the cost is flat from here to about 128.
BLOCK = 32

# Contingencies whose columns are built in one sparse solve, at startup only.  Larger
# because it runs once and only bounds the transient dense arrays.
BUILD_BLOCK = 512


class DroppedFlowCheck:
    """The flow-shift columns of a set of contingencies, and the limit check they feed.

    `ptdf` must carry the full contingency set (see `scdcopf.build_audit_ptdf`); `dropped`
    names the ones to check, normally those the screened model does not carry.  Holds
    8 bytes per (dropped contingency, branch) -- 71 MB on ACTIVSg2000 -- for the life of
    the model it checks.

    `dropped` is empty whenever the model already carries every contingency: a case that
    has not been screened yet, or one whose screen kept everything.  The check is then
    trivially satisfied rather than inapplicable, so it builds and reports no violations.
    """

    def __init__(self, ptdf, dropped):
        self.ptdf = ptdf
        self.names = [cn for cn in ptdf.contingencies if cn in dropped]
        self.limits = ptdf.contingency_limits_array
        self.branches = ptdf.branches_keys
        if not self.names:
            return
        comps = [ptdf.contingency_compensators[cn] for cn in self.names]
        assert all(c.M.shape[1] == 1 and c.VA_compensator is None for c in comps), \
            "only single-branch outages without phase shifters are handled"
        self.M = sp.hstack([c.M for c in comps], format="csc")
        self.out_idx = np.array([ptdf._branchname_to_index_map[c.branch_out] for c in comps])
        self.columns = self._build_columns(np.array([float(c.c) for c in comps]))
        self.buffer = np.empty((BLOCK, len(self.limits)))

    def _build_columns(self, c_scalars):
        """Per-unit flow shift on every branch, one contingency per row.

        The whole startup cost of this module: one sparse solve per block, 0.2 s for
        ACTIVSg2000's 2723 dropped contingencies, never repeated.
        """
        out = np.empty((len(self.names), len(self.branches)))
        for lo in range(0, self.M.shape[1], BUILD_BLOCK):
            hi = lo + BUILD_BLOCK
            shift = self.ptdf.B_dA @ self.ptdf.MLU.solve(self.M[:, lo:hi].toarray())
            out[lo:hi] = (shift * -c_scalars[None, lo:hi]).T
        return out

    def violations(self, m):
        """(contingency, worst branch, |post-outage flow| pu, limit pu) per violation.

        An empty result proves the model's solution also solves the problem carrying every
        dropped contingency: constraints satisfied at an optimum cannot move it.
        """
        if not self.names:
            return []
        PFV, _, VA = self.ptdf.calculate_PFV(m)
        beta = np.asarray(self.M.T @ -VA[self.ptdf.ref_bus_mask]).ravel()
        found = []
        for lo in range(0, len(self.names), BLOCK):
            rows = self.columns[lo:lo + BLOCK]
            over = self.buffer[:len(rows)]
            np.multiply(rows, beta[lo:lo + len(rows), None], out=over)
            over += PFV[None, :]
            over[np.arange(len(rows)), self.out_idx[lo:lo + len(rows)]] = 0.0
            np.abs(over, out=over)
            over -= self.limits[None, :]
            worst = over.argmax(axis=1)
            for j in np.flatnonzero(over[np.arange(len(rows)), worst] > TOL):
                b = worst[j]
                found.append((self.names[lo + j], self.branches[b],
                              float(over[j, b] + self.limits[b]), float(self.limits[b])))
        return found

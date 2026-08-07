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

Three details carry the speed.  The columns are stored one contingency per row, so the
per-contingency maximum runs along contiguous memory; they are consumed in blocks of
`BLOCK` rows through one preallocated buffer, so the arithmetic stays in cache instead of
streaming a large temporary three times; and they are stored in `DTYPE` rather than double.
Measured on activsg2000_original_v12, snapshot 2025-02-19 06:00, 2723 dropped
contingencies, against a 11-28 ms solve: 345 ms/config through EGRET, 66 with the columns
stored but read flat, 14.5 blocked in double, 6.1 as written here.

The cost is arithmetic, not memory traffic, so `DTYPE` pays through SIMD width rather than
through bytes moved.  The array is read start to end once per configuration and is far
larger than any cache, which invites the opposite conclusion, but the same pass over a
1 MB array that stays resident degrades identically when workers fill a node: 1.52x at 10
workers on 10 physical cores against 1.57x for the 35 MB array, doing the same arithmetic.
A 16 KB kernel that never leaves L1 degrades 1.48x over the same range, so what is
contended is the all-core clock, not any level of the memory system, and it is a property
of the machine rather than of this array.  Sharing one copy of the array between workers,
or blocking it so it stays in cache, therefore buys nothing.

What does follow is that a sweep filling a node is paced by this check rather than by the
solve it checks, because the check is simply more expensive: 8.7 ms against 11 ms for a
solve, in a step that runs once per solve.

On top of that, `_reachable` discards outright the contingencies that provably cannot
overload anything at the dispatch in hand -- 47-69% of the set, worth a further 1.37x.

What is left is per configuration and does not batch: the limit pass touches 8.7 M
(contingency, branch) pairs, each configuration brings its own flows, and checking several
configurations against one pass over the columns measures 1.25x at best and reverses beyond
that (15.0 ms/config at one, 12.0 at two, 11.5 at four, 13.4 at eight), because what is
shared is one read and what is not is all the arithmetic.

Verified against `calculate_PFV_delta` on three snapshots: identical violations, and
flows agreeing to 7e-16 pu over all 8.7 M (branch, contingency) pairs in double.  Single
precision moves those flows by at most 2e-6 pu, against a 1e-4 pu violation tolerance.
"""

import numpy as np
import scipy.sparse as sp

# Post-contingency overflow (per unit) above which a dropped contingency counts as
# violated -- about 0.01 MW on a 100 MVA base.
TOL = 1e-4

# Width the columns are stored and the limit pass is run in.  Flows are per unit and O(1),
# so single precision resolves them to ~1e-6 pu -- two orders below `TOL` -- and the check
# is bandwidth-bound, so the narrower type is worth about a factor of two.
DTYPE = np.float32

# Contingencies whose flows are computed in one pass through the working buffer.  Small
# enough that a block stays in cache; the cost is flat from here to about 128.
BLOCK = 32

# Contingencies whose columns are built in one sparse solve, at startup only.  Larger
# because it runs once and only bounds the transient dense arrays.
BUILD_BLOCK = 512

# Relative slack demanded before a contingency is skipped unchecked, so the skip never
# turns on a rounding difference in `DTYPE`.  The measured margin is ~0.16 in per-unit
# terms, four orders larger, so this costs nothing.
SKIP_GUARD = 1e-4


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
        self.limits = ptdf.contingency_limits_array.astype(DTYPE)
        self.branches = ptdf.branches_keys
        if not self.names:
            return
        comps = [ptdf.contingency_compensators[cn] for cn in self.names]
        assert all(c.M.shape[1] == 1 and c.VA_compensator is None for c in comps), \
            "only single-branch outages without phase shifters are handled"
        self.M = sp.hstack([c.M for c in comps], format="csc")
        self.out_idx = np.array([ptdf._branchname_to_index_map[c.branch_out] for c in comps])
        self.columns = self._build_columns(np.array([float(c.c) for c in comps]))
        self.buffer = np.empty((BLOCK, len(self.limits)), dtype=DTYPE)
        assert (self.limits > 0).all(), "the skip bound divides by every branch's limit"
        self.reach = (np.abs(self.columns) / self.limits[None, :]).max(axis=1)

    def _build_columns(self, c_scalars):
        """Per-unit flow shift on every branch, one contingency per row.

        The whole startup cost of this module: one sparse solve per block, 0.2 s for
        ACTIVSg2000's 2723 dropped contingencies, never repeated.  The solve itself runs
        in double and only the stored result is narrowed.
        """
        out = np.empty((len(self.names), len(self.branches)), dtype=DTYPE)
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
        beta = np.asarray(self.M.T @ -VA[self.ptdf.ref_bus_mask]).ravel().astype(DTYPE)
        PFV = PFV.astype(DTYPE)
        found = []
        for idx in self._reachable(PFV, beta):
            over = self.buffer[:len(idx)]
            np.take(self.columns, idx, axis=0, out=over)
            over *= beta[idx, None]
            over += PFV[None, :]
            over[np.arange(len(idx)), self.out_idx[idx]] = 0.0
            np.abs(over, out=over)
            over -= self.limits[None, :]
            worst = over.argmax(axis=1)
            for j in np.flatnonzero(over[np.arange(len(idx)), worst] > TOL):
                b = worst[j]
                found.append((self.names[idx[j]], self.branches[b],
                              float(over[j, b] + self.limits[b]), float(self.limits[b])))
        return found

    def _reachable(self, PFV, beta):
        """Blocks of contingency indices that could violate a limit at this dispatch.

        A contingency shifts branch `j` to `PFV_j + beta_c * col_cj`, so in units of that
        branch's limit its loading is at most `f + |beta_c| * reach_c`, where `f` is the
        base case's worst loading and `reach_c` the largest column entry relative to a
        limit.  When that bound is under one the contingency cannot overload anything and
        is dropped without arithmetic.  The bound only loosens the true maximum, so this
        never hides a violation; when a base branch is already over a contingency limit it
        keeps everything, which is the safe direction.

        Measured on activsg2000_original_v12 it discards 47-69% of the set for a 1.37x, and
        over six configurations -- all candidates on, all off, and four random -- the
        violations found were identical to checking everything and no discarded contingency
        came within 0.157 pu of a limit.  The headroom is structural rather than lucky:
        base flows are held to the long-term rating and compared here against the emergency
        one, so `f` sits at 1/`emergency_factor` unless a base limit is soft-violated.
        """
        f = float((np.abs(PFV) / self.limits).max())
        keep = np.flatnonzero(np.abs(beta) * self.reach > 1.0 - f - SKIP_GUARD)
        return [keep[lo:lo + BLOCK] for lo in range(0, len(keep), BLOCK)]

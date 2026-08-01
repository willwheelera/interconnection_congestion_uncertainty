"""Barrier-regularized sensitivity of LMPs and congestion duals to injections.

The project's original research direction: mu is piecewise constant in an LP, so a
barrier method's smoothed duals stand in for a derivative that does not exist.  Used by
the samplers only through `proxy`, which ranks which generator flip to evaluate next.
"""

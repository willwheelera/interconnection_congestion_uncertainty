"""The physical model: one generator configuration in, congestion duals out.

`scdcopf` builds and warm re-solves the N-1 security-constrained DCOPF under a case's
recipe, `relief` holds the ModelData edits that recipe is made of, `soft_limits` caps the
shadow price, and `evaluator` runs the whole thing in parallel over many configurations.
"""

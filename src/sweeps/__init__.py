"""Mass evaluation and the caches it produces.

`enumerate_configs` sweeps configurations into a long per-dual `mu_cache`; the
`build_*_cache` modules reduce a raw sweep to the common per-config frames the samplers
and descriptor study read.
"""

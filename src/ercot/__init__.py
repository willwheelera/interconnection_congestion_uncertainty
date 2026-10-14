"""ERCOT public-API data access.

A small client for the ERCOT public API: `ercot_auth` writes a short-lived bearer token,
`fetch_ercot` pulls datasets with it, and `data_paths` locates the downloaded files.  Used by
the `cases.texas` snapshot fetchers to build the load and renewable snapshot collection.
"""

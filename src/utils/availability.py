"""Helpers for generator availability vectors and their integer encoding."""

import numpy as np


def avail_to_id(availability) -> int:
    """Encode a 0/1 availability vector as an integer (MSB = first generator)."""
    avail = np.asarray(availability, dtype=int)
    return int("".join(str(b) for b in avail), 2)


def id_to_avail(avail_id: int, ngen: int) -> np.ndarray:
    """Decode an integer into an ngen-bit availability vector (MSB = first generator)."""
    return np.array(list(np.binary_repr(avail_id, ngen)), dtype=np.int8)


def remove_unavailable_generators(md, availability) -> None:
    """Delete generators whose availability flag is 0."""
    gen_names = list(md.data["elements"]["generator"].keys())
    assert len(availability) == len(gen_names), \
        f"Availability list length ({len(availability)}) != number of generators ({len(gen_names)})"
    for name, avail in zip(gen_names, availability):
        if avail == 0:
            del md.data["elements"]["generator"][name]


def get_all_neighbors(ids, NGEN):
    bit_mask = (1 << (NGEN - 1 - np.arange(NGEN))).astype(np.int64)
    neighbor_ids = ids[:, None] ^ bit_mask[None, :]                 # (n, NGEN)
    return neighbor_ids


def get_neighbor_i(ids, i):
    return ids ^ 1<<i

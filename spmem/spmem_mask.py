#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Training-time attention mask for SpMem.

Over the augmented stream `[C reals][m memory] ...`:

  - A real query sees the reals its mipmap taper retains (over real indices), plus earlier memory.
  - A memory query sees the `w0` reals ending at its chunk, earlier memory, and its chunk's m
    siblings in both directions -- the one non-causal term. Its reals are not tapered: thinning
    them would change what the memory compresses.

Earlier memory is kept whole, bounded by a ring, or tapered over memory ordinals, per
`SpMemGeometry`. The predicate is integer arithmetic on positions, so it is tracer-safe and
restarts per packed document. `SpMemCache` is its decode-time peer.
"""

import functools

from axlearn.common.attention_bias import MaskFn

from spmem.spmem_geometry import (
    SpMemGeometry,
    decompose_augmented_position,
)
from twa.twa_mask import taper_survives


@functools.lru_cache(maxsize=None)
def spmem_mask_fn(geometry: SpMemGeometry, *, coset_index: int) -> MaskFn:
    """`MaskFn(query_position, key_position) -> bool` for one global layer.

    Memoized: splash keys its MaskInfo cache on `id(mask_fn)`, so sharing one closure per
    (geometry, coset) builds MaskInfo once per coset rather than once per layer.
    """
    if coset_index not in (0, 1):
        raise ValueError(f"coset_index must be 0 or 1, got {coset_index}")
    c, m, w0 = geometry.mem_chunk_size, geometry.num_mem, geometry.w0
    real_taper = geometry.real_taper
    mem_taper = geometry.mem_taper
    mem_chunks = geometry.mem_window_chunks

    def mask(query_position, key_position):
        q, k = query_position, key_position
        causal = q >= k
        q_chunk, q_off, q_real = decompose_augmented_position(q, mem_chunk_size=c, num_mem=m)
        k_chunk, k_off, k_real = decompose_augmented_position(k, mem_chunk_size=c, num_mem=m)
        # Compared directly rather than negated with `~`, which is wrong on Python ints.
        q_is_real, q_is_mem = q_off < c, q_off >= c
        k_is_real, k_is_mem = k_off < c, k_off >= c

        # Earlier memory, bounded per query kind: a real query's own chunk has no memory yet,
        # while a memory query's own group already holds a slot.
        earlier_mem = causal & k_is_mem
        if mem_taper is not None:
            k_ord = k_chunk * m + (k_off - c)
            real_clock = q_chunk * m - 1
            real_clock = real_clock * (real_clock > 0)
            mem_clock = q_chunk * m + (q_off - c)
            real_q_mem = earlier_mem & taper_survives(
                real_clock, k_ord, geometry=mem_taper, coset_index=coset_index
            )
            mem_q_mem = earlier_mem & taper_survives(
                mem_clock, k_ord, geometry=mem_taper, coset_index=coset_index
            )
        elif mem_chunks is not None:
            real_q_mem = earlier_mem & (q_chunk - k_chunk <= mem_chunks)
            mem_q_mem = earlier_mem & (q_chunk - k_chunk < mem_chunks)
        else:
            real_q_mem = mem_q_mem = earlier_mem

        tapered = taper_survives(q_real, k_real, geometry=real_taper, coset_index=coset_index)
        real_sees = q_is_real & ((causal & k_is_real & tapered) | real_q_mem)

        chunk_end = q_chunk * c + (c - 1)
        mem_window = causal & k_is_real & (chunk_end - k_real < w0)
        siblings = k_is_mem & (k_chunk == q_chunk)
        mem_sees = q_is_mem & (mem_window | mem_q_mem | siblings)

        # Pad and empty cache slots carry a negative position.
        return (real_sees | mem_sees) & (k >= 0)

    return mask

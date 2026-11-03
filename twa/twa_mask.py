#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Training-time attention mask for Tapered Window Attention (TWA).

Provides a closed-form predicate `mask_fn(q_pos, k_pos) -> bool` that agrees with what the model
attends to at every `(q, k)` pair, uniformly across training, prefill, and decode.

**The OCTAVE rule halves the density once per DOUBLING of age.** A dense window of `w0`, then half the
keys per doubling -- so `density ∝ 1/age`, and every octave of age carries the same detail:

    age        = q - k
    level(age) = 0 if age < w0 else floor(log2(age / w0)) + 1      ages double per level
    resident   iff age < 2^K * w0  AND  v_2(k + coset) >= level(age)

Level 0 imposes no condition, so **the most recent `w0` keys are present verbatim at every step**
and SWA(w0) is a subset of TWA. Level j holds `w0/2` keys, so the buffer is `w0 + K*(w0/2)` --
LINEAR in the level count while the reach `2^K * w0` is EXPONENTIAL. That is the point: at
`w0 = 2048, w1 = 128K` it is 8,192 slots reaching 131,072, and doubling the reach costs 1,024 more
slots rather than twice as many. `twa_geometry` owns those derivations and is the only place
that computes them.

OCTAVE needs no anchor. `v_2(0)` is infinite, so key 0 clears every level and dies
only at `w1` -- resident at `w1 - 1`, gone at `w1`. A fixed-width predecessor bought that
permanence by pinning its coarsest level, which was free when that level held one slot; here it
holds `w0/2` and would not be.

The MIPMAP rule (`TaperRule.MIPMAP`) keeps the spans EQUAL at `w0` and halves the count instead,
plus one permanent anchor key; see `_mipmap_survives`. It spends the same budget on a larger
verbatim window where OCTAVE spends it on reach.

The predicate is closed-form and branch-free, and it never computes the level: since the lowest set
bit is a power of two, `v_2(key) >= level(age)` is exactly `age < w0 * 2^v_2(key)`, so every key has
a LIFETIME of `w0 * min(2^v_2(key), 2^K)` and survives while it is younger. `key == 0` takes the
full `w1`, its low bit reading 0 while its valuation is infinite. Constant in K, and since
the lifetime depends only on the key -- which arrives as a ROW -- only a subtract and a compare
touch the full `(block_q, block_kv)` tile. It matters because SplashAttention evaluates
`mask_function` both on the HOST while building MaskInfo and inside the Pallas kernel, in the
forward, dq and dkv kernels alike.

The mask filters sentinel positions in unfilled cache slots via causal `q >= k`; populated slots
satisfy the predicate by construction. The matching caches are `TWACache` and `SpMemCache`,
whose `taper_slot` is this same rule expressed as a slot assignment.
"""

import functools
from typing import Optional

from axlearn.common.attention_bias import MaskFn, MaskFnAttentionBias
from axlearn.common.config import ClassConfigBase

from twa.twa_geometry import (
    TaperGeometry,
    TaperRule,
    anchor_index,
    taper_geometry,
)


# Memoized so identical geometry returns the SAME closure object. SplashAttention's MaskInfo build
# (`_process_mask`) is `lru_cache`d, but axlearn's `ComputableMask` keys that cache on `id(mask_fn)`
# -- a fresh closure per global layer defeats it, rebuilding MaskInfo per layer (a host-side blowup
# that took hours at 128K). Sharing one closure per (geometry, coset) collapses those rebuilds to
# one per distinct pair.
def twa_mask_fn(
    w0: int, coset_index: int, w1: Optional[int] = None, rule: TaperRule = TaperRule.OCTAVE
) -> MaskFn:
    """Builds a `MaskFn` for one global layer's TWA thinned attention.

    Args:
        w0: dense window -- every key younger than this is kept verbatim. A TRAINING-TIME constant:
            changing it changes which keys are retained at every age, near ages included, so a
            checkpoint cannot be evaluated at a different one. Must be even and >= 2.
        coset_index: this layer's coset ∈ {0, 1}.
        w1: receptive field -- the age the taper must still reach. Under OCTAVE it has no usable
            default, and omitting it raises from `taper_geometry` rather than quietly meaning `w0`,
            i.e. a dense window and no taper under a name that says taper. Safe to RAISE at
            inference, which only appends outer levels. IGNORED under MIPMAP, whose reach is
            `log2(w0) * w0`; see `twa_geometry`.
        rule: which retention rule; see `TaperRule`.

    Returns:
        A `MaskFn(query_position, key_position) -> bool`, True iff the model attends to
        `key_position` at `query_position` (causal AND the taper's survival predicate).
    """
    if coset_index not in (0, 1):
        raise ValueError(f"coset_index must be 0 or 1, got {coset_index}")
    # Key the cache on the ROUNDED `w1`, not on the one asked for. Levels are whole doublings, so
    # every request in `(w0*2^(K-1), w0*2^K]` is the same mask, and keying on the request hands out
    # a distinct closure for each -- exactly the MaskInfo rebuild the memoization exists to prevent.
    # `taper_geometry` also validates `w0` here, before the cache rather than behind it.
    geometry = taper_geometry(w0=w0, w1=w1, rule=rule)
    return _twa_mask_fn(w0, coset_index, geometry.w1, rule)


@functools.lru_cache(maxsize=None)
def _twa_mask_fn(w0: int, coset_index: int, w1: int, rule: TaperRule) -> MaskFn:
    """The memoized body; call `twa_mask_fn`, which rounds `w1` up to a doubling first."""
    geometry = taper_geometry(w0=w0, w1=w1, rule=rule)

    def mask(query_position, key_position):
        q = query_position
        k = key_position
        return (q >= k) & taper_survives(q, k, geometry=geometry, coset_index=coset_index)

    # Duck-typed for axlearn's `mask_band`: a true bound on the keys this mask can attend, so the
    # splash kernel band-shrinks its grid instead of scanning every (q_block, kv_block) pair -- the
    # full scan made a 1M-token TWA mask take hours to build. It must be an UPPER bound: a key
    # outside it sits in a block the kernel never visits and is silently dropped. OCTAVE never keeps
    # a key past `w1`. MIPMAP retires every key by `levels * w0` except its anchor, which every
    # query reads forever, so the anchor goes in as a sink rather than stretching the window.
    if geometry.rule is TaperRule.MIPMAP:
        mask.left_context = geometry.levels * w0
        mask.sink_len = anchor_index(geometry.levels, coset_index) + 1
    else:
        mask.left_context = w1
    return mask


def taper_survives(query_index, key_index, *, geometry: TaperGeometry, coset_index: int):
    """Whether `key_index` is still resident when `query_index` runs. NOT causal.

    Split out of `twa_mask_fn` so `spmem_mask` can apply the same thinning over the REAL index
    space of an augmented stream -- and over MEMORY ORDINALS for the pool -- where causality and the
    memory terms are its own to decide. Two copies of this 2-adic arithmetic would be a mask that
    drifts from the cache silently.

    Returns True for `key_index > query_index`; callers apply causality separately. A negative age
    falls in level 0, which is the same thing the caller's `q >= k` rejects anyway.

    Args:
        query_index: query position in the thinned stream (a real index or a memory ordinal).
        key_index: key position in the same space.
        geometry: the taper's `TaperGeometry`; only `w0` and `levels` are read.
        coset_index: this layer's coset, 0 or 1.
    """
    if geometry.rule is TaperRule.MIPMAP:
        return _mipmap_survives(query_index, key_index, geometry=geometry, coset_index=coset_index)

    w0, k_levels = geometry.w0, geometry.levels
    key = key_index + coset_index

    # Each key survives while `age < w0 * min(2^v_2(key), 2^K)`. No level is computed: `v_2(key) >=
    # level(age)` is exactly `age < w0 * 2^v_2(key)` because the low bit is a power of two, and `w1`
    # caps it. The lifetime depends only on the key, which SplashAttention passes as a ROW, so just
    # the subtract and the compare cost a full tile -- 9x, and the mask is built on the host.
    low_bit = key & (-key)
    # `* 0 +` is a dtype carrier, not arithmetic: a Python int against a bool below promotes to
    # int64 and upcasts the whole age tile at the compare -- identical mask, 2.5x slower.
    w1_units = low_bit * 0 + (1 << k_levels)
    # Key 0's low bit reads 0 though its valuation is infinite, so it lives the full `w1` -- hence
    # no anchor is needed. Keys past the last level clamp there too.
    at_w1 = (low_bit == 0) | (low_bit > w1_units)
    lifetime = (low_bit * ~at_w1 + w1_units * at_w1) * w0
    return (query_index - key_index) < lifetime


def _mipmap_survives(query_index, key_index, *, geometry: TaperGeometry, coset_index: int):
    """MIPMAP residency: spans EQUAL at `w0`, counts halving. Reached via `taper_survives`.

        level(q, k) = (q - k) // w0            the level k has aged into
        J           = log2(w0)                 index of the last level, the anchor's
        k_0         = (w0 - coset) mod w0      the key that level holds permanently
        resident    iff k == k_0  or  level <= min(v_2(k + coset), J - 1)

    Level 0 imposes no condition, so the most recent `w0` keys are verbatim and SWA(w0) is a subset
    of this too. Rotating level j holds `w0 >> j` keys and the anchor holds one, so occupancy is a
    flat `geometry.slots` past warmup, and an ordinary key's reach is FINITE at `levels * w0`.

    Warmup ends at `k_0 + J*w0`, which is COSET-DEPENDENT: the anchor is withheld from the rotating
    levels, so one rotating slot stays empty until the anchor has aged past every band -- `J*w0` on
    an even coset, `w0 - 1` steps later on an odd one. Occupancy never exceeds the budget and never
    drops; it just reaches the ceiling later on coset 1.
    """
    w0 = geometry.w0
    # `levels` is how many rings were allocated, which `max_seq` may cap below `log2(w0)`.
    top = geometry.levels - 1  # the last ROTATING level; the level above it is the anchor's

    key = key_index + coset_index
    low_bit = key & (-key)

    # Same lifetime trick as OCTAVE, and for the same reason -- the obvious form shifts by a
    # TILE-sized `level`, which cost 346us a block against 102us here. `w0 = 2^j_max`, so
    # `level <= m` is `age < (m + 1) * w0`, and with `m = min(v_2(key), top)` that is the whole
    # rotating condition including the reach. `min(v_2, top)` counts the powers of two at or below
    # the low bit -- `top` comparisons, but on the KEY, which arrives as a row, so they are vectors
    # rather than tiles. `* 0 +` is the dtype carrier; see the OCTAVE branch.
    capped_v2 = low_bit * 0
    for level in range(1, top + 1):
        capped_v2 = capped_v2 + (low_bit >= (1 << level))
    lifetime = (capped_v2 + 1) * w0

    # `key == 0` needs no disjunct of its own: on an even coset that key IS the anchor.
    return (key_index == anchor_index(geometry.levels, coset_index)) | (
        (query_index - key_index) < lifetime
    )


def twa_attention_bias_config(
    w0: int,
    coset_index: int,
    w1: Optional[int] = None,
    rule: TaperRule = TaperRule.OCTAVE,
) -> ClassConfigBase[MaskFnAttentionBias]:
    """Returns a `MaskFnAttentionBias` config wired with the TWA mask.

    Drop-in replacement for `CausalAttentionBias.default_config()` on a global-attention layer.
    Pass to `attn_cfg.mask = ...`.
    """
    return MaskFnAttentionBias.default_config(
        mask=twa_mask_fn(w0=w0, coset_index=coset_index, w1=w1, rule=rule),
    )

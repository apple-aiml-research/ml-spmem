#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Decode-time KV cache for Tapered Window Attention (TWA).

The buffer is the taper's pyramid laid out flat, and `twa_geometry` is the only place its size
comes from. See `twa_mask.py` for the predicate; this file holds the slot arithmetic.

  Level 0 is a `w0` ring holding the dense window -- the last `w0` keys, verbatim, never thinned.
  Under OCTAVE each tapered level j (1..K) is a ring of `w0 // 2` covering ages
  `[2^(j-1) w0, 2^j w0)` at rate `2^-j`, laid after level 0 at `w0 + (j-1)*(w0//2)`: total
  `w0 + K*(w0//2)`, 8,192 at `w0 = 2048` reaching 128K. Under MIPMAP level j is a ring of `w0 >> j`
  covering ages `[j w0, (j+1) w0)`, plus one permanent anchor slot: at most `2*w0 - 1`.

  A key's slot MOVES as it ages into the next level. That costs nothing here: `extend_step` already
  recomputes every slot from stored positions and scatters the whole buffer each step, so there is
  no cascade of per-level copies to sequence.

Output K/V from `extend_step` is the post-update buffer (re-slotted + written, transposed to BSNH),
`geometry.slots` long -- the persistent buffer, no extra allocation. Slots the pyramid has not filled
hold a LARGE POSITIVE sentinel (2^30) so the mask's causal `q >= k` rejects them without a custom
term: unlike `SlidingWindowKVCache` this mask has no past bound, so causality is the only filter and
it rejects the future.

The training-time mask (`twa_mask.py::twa_mask_fn`) filters identically across training,
prefill and decode, from the same `(w0, w1)` -- that shared geometry is what makes the cache contents
equal the mask's visible set by construction rather than by agreement.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from axlearn.common.attention import MultiheadAttention
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.kv_cache.base_kv_cache import BaseKVCache
from axlearn.common.module import nowrap
from axlearn.common.utils import Nested, Tensor, maybe_shard

from twa.twa_geometry import (
    TaperGeometry,
    TaperRule,
    anchor_index,
    taper_geometry,
)
from twa.twa_mask import taper_survives, twa_attention_bias_config

# Large positive sentinel for "invalid" key positions. Picked so that any
# realistic query position is less than this, so causal_mask (q >= k)
# naturally returns False for invalid slots.
_INVALID_POSITION = 1 << 30  # 2^30 ≈ 1.07e9; max realistic ctx ~10^7


class TWACache(BaseKVCache):
    """KV cache holding exactly the keys the TWA mask admits, in `geometry.slots` slots.

    Stored slot order does not match sequence order; `key_positions` tracks true positions. See the
    module docstring.
    """

    @config_class
    class Config(BaseKVCache.Config):
        """Configures TWACache.

        Attributes:
            w0: dense window -- every key younger than this is kept verbatim. A TRAINING-TIME
                constant; see `twa_geometry`.
            w1: receptive field, same unit as `w0`. REQUIRED, not defaulted: a default would have
                to mean "no taper", and a caller who forgot it would get a plain `w0` sliding window
                under a name that says taper.
            coset_index: This layer's coset ∈ {0, 1}, which selects which half of each tapered
                level's keys this layer keeps.
            rule: which retention rule; see `TaperRule`. OCTAVE unless stated, so the field's
                presence changes no existing config. Under MIPMAP `w1` is derived from `w0` and
                must be omitted.
        """

        w0: Required[int] = REQUIRED
        coset_index: Required[int] = REQUIRED
        w1: Optional[int] = None
        rule: TaperRule = TaperRule.OCTAVE

    @classmethod
    def default_config(
        cls,
        *,
        w1: Optional[int] = None,
        w0: int = 2048,
        coset_index: int = 1,
        rule: TaperRule = TaperRule.OCTAVE,
    ) -> Config:
        cfg = super().default_config()
        # Fail at build, not at the first decode step. Under OCTAVE this is also what rejects a
        # missing `w1`, which is why the field can be Optional without reintroducing a silent
        # dense-window fallback.
        taper_geometry(w0=w0, w1=w1, rule=rule)
        return cfg.set(w0=w0, coset_index=coset_index, w1=w1, rule=rule)

    def _geometry(self) -> TaperGeometry:
        """The one geometry this layer runs on; the mask is built from the same arguments."""
        cfg = self.config
        return taper_geometry(w0=cfg.w0, w1=cfg.w1, rule=cfg.rule)

    def _max_capacity(self) -> int:
        """Cache buffer size, from the geometry so the mask and the buffer cannot disagree."""
        return self._geometry().slots

    @staticmethod
    def _invalid_position() -> int:
        return _INVALID_POSITION

    @nowrap
    def init_states(self, shape: BaseKVCache.Shape, *, dtype: jnp.dtype) -> Nested[Tensor]:
        cfg = self.config
        cap = self._max_capacity()
        # BNHT layout (matches SlidingWindowKVCache for TPU fusion).
        bnht_shape = (shape.batch_size, shape.num_kv_heads, shape.per_head_dim, cap)
        bnht_spec = None
        batch_spec = None
        if cfg.kv_partition_spec is not None:
            b, t, n, h = cfg.kv_partition_spec
            bnht_spec = (b, n, h, t)
            batch_spec = (b, None)
        invalid = self._invalid_position()
        return dict(
            key=maybe_shard(jnp.zeros(shape=bnht_shape, dtype=self._cache_dtype(dtype)), bnht_spec),
            value=maybe_shard(
                jnp.zeros(shape=bnht_shape, dtype=self._cache_dtype(dtype)), bnht_spec
            ),
            key_positions=maybe_shard(
                jnp.full((shape.batch_size, cap), invalid, dtype=jnp.int32), batch_spec
            ),
        )

    def extend_step(
        self,
        cached_states: Nested[Tensor],
        *,
        k_proj: Tensor,
        v_proj: Tensor,
        key_positions: Tensor,
        segment_ids: Optional[Tensor] = None,
        page_pool: Optional[Nested[Tensor]] = None,
    ) -> tuple[Nested[Tensor], BaseKVCache.Output]:
        """Updates the TWA cache in one vectorized per-row scatter.

        Mirrors the dense `KVCache.extend_step` idiom — a single position-keyed
        scatter of the whole chunk, with no per-token helper and no
        `step_len == 1` branch — so decode (step_len=1), prefill (step_len>1
        from an empty cache), and multistep decode (step_len>1 from a non-empty
        cache) all share one path.

        The only twist over the dense cache: dense stores position k at slot k
        (entries never move), whereas here a slot is position-derived and shifts
        as the taper fires. So we concatenate the existing survivors with the new
        chunk and recompute every element's slot from its absolute position with
        the **same closed form the training mask uses** (`taper_slot`, under the
        layer's `TaperRule`, matched to the mask by `enable_twa_attention`). Since
        residency is exactly the mask's
        predicate, the cache contents equal the mask's visible set by
        construction. Survivors map injectively onto slots [0, live count);
        non-survivors (and sentinel/padding columns) route to a trash slot that
        is sliced off.

        The slot computation + scatter run **per batch row** (`jax.vmap`), so
        ragged multibatch — rows with different valid lengths, e.g. eval prefill
        with left-aligned `segment_ids` — is handled correctly: each row re-slots
        by its own positions rather than sharing one schedule.

        Precondition: each row's positions are contiguous from 0 (dense prefix /
        left-aligned padding), as in training and autoregressive decode. A key's
        level is derived from its absolute position, so interior
        position gaps (e.g. `segment_ids=[1,0,1,0]`) are unsupported — they don't
        arise here (the evaler left-aligns padding).

        Output K/V tensor: **post-update cache state**, transposed BNHT → BSNH
        for the attention layer. Source length is exactly `geometry.slots` (the
        persistent buffer size); slots beyond the live count hold sentinel positions
        (`2^30`) and are filtered by the mask's causal `q >= k` check.

        Diverges from `SlidingWindowKVCache`'s `concat(pre-update cache, new
        K/V)` operand: SlidingWindow's ring stores only W but its window spans
        W+1, so it re-adds the just-evicted slot via concat. This buffer IS the
        max working set, so the post-update buffer is already the complete
        per-query operand -- no redundant +1 slot that would break power-of-2
        kernel-tile alignment.
        """
        assert page_pool is None
        cfg = self.config
        geometry = self._geometry()
        coset = cfg.coset_index
        cap = self._max_capacity()
        sentinel = self._invalid_position()
        batch, step_len = k_proj.shape[:2]

        # New chunk positions: [1|batch, step_len] -> [batch, step_len]; padding
        # (segment_ids == 0) is marked invalid so it never occupies a slot.
        new_positions = jnp.broadcast_to(key_positions, (batch, step_len))
        if segment_ids is not None:
            new_positions = jnp.where(segment_ids != 0, new_positions, sentinel)

        # [B, T, N, H] -> [B, N, H, T] (cache axis last), cast to the cache dtype.
        k_new = jnp.einsum("btnh->bnht", k_proj).astype(cached_states["key"].dtype)
        v_new = jnp.einsum("btnh->bnht", v_proj).astype(cached_states["value"].dtype)

        # Existing survivors ++ new chunk. The single scatter re-slots the
        # survivors (they move as they age into the next level) and places the
        # new tokens at once.
        positions = jnp.concatenate([cached_states["key_positions"], new_positions], axis=1)
        keys = jnp.concatenate([cached_states["key"], k_new], axis=-1)
        values = jnp.concatenate([cached_states["value"], v_new], axis=-1)

        def scatter_row(keys_row, values_row, pos_row):
            """Re-slot one batch row's columns by their own absolute positions.

            keys_row/values_row: [N, H, M]; pos_row: [M]. Uses the closed form
            above; runs independently per row (via `jax.vmap`), so a row's slot
            schedule depends only on its own positions — ragged multibatch safe.
            """
            valid = pos_row != sentinel
            q_max = jnp.max(jnp.where(valid, pos_row, -1))
            # `taper_slot` is sentinel-AGNOSTIC and says so: positions must carry no sentinel above
            # the real range. This buffer stores `+2^30`, which reads as a hugely NEGATIVE age, lands
            # in level 0, and comes back resident on a slot a real key also wants -- measured at
            # w0=4: the sentinel takes slot 0, which real position 4 also takes. The `valid &` below
            # keeps it out of the scatter either way, so this is not a live bug; normalizing here
            # makes the contract true at the call that makes it rather than downstream of an `and`.
            survives, slot = taper_slot(
                jnp.where(valid, pos_row, -1), q_max, geometry=geometry, coset_index=coset
            )
            survives = valid & survives
            target = jnp.where(survives, slot, cap)  # non-survivors -> trash slot `cap`
            key_out = jnp.zeros((*keys_row.shape[:2], cap + 1), dtype=keys_row.dtype)
            value_out = jnp.zeros((*values_row.shape[:2], cap + 1), dtype=values_row.dtype)
            pos_out = jnp.full((cap + 1,), sentinel, dtype=jnp.int32)
            key_out = key_out.at[..., target].set(keys_row)
            value_out = value_out.at[..., target].set(values_row)
            pos_out = pos_out.at[target].set(pos_row)
            return key_out[..., :cap], value_out[..., :cap], pos_out[:cap]

        new_key, new_value, new_key_positions = jax.vmap(scatter_row)(keys, values, positions)
        final_state = dict(key=new_key, value=new_value, key_positions=new_key_positions)

        # Output: post-update buffer transposed BNHT -> BSNH for the attn layer.
        out_key = jnp.einsum("bnht->btnh", final_state["key"])
        out_value = jnp.einsum("bnht->btnh", final_state["value"])
        return final_state, self.Output(
            k_proj=out_key,
            v_proj=out_value,
            key_positions=final_state["key_positions"],
        )


def enable_twa_attention(
    cfg: MultiheadAttention.Config,
    *,
    w0: int,
    coset_index: int,
    w1: Optional[int] = None,
    rule: TaperRule = TaperRule.OCTAVE,
) -> MultiheadAttention.Config:
    """Install the TWA cache + matching training-time mask on an attention layer.

    Drop-in analog of `enable_sliding_window_attention`. Sets
    `cfg.kv_cache = TWACache` (used at inference / `extend_step`)
    AND `cfg.mask = twa_attention_bias_config(...)` (used at
    training / forward). The cache and mask encode the same survival
    rule, so the model trains under the exact attention pattern that
    inference will run under.
    """
    if coset_index not in (0, 1):
        raise ValueError(f"coset_index must be 0 or 1, got {coset_index}")
    cfg.set(
        kv_cache=TWACache.default_config(w0=w0, coset_index=coset_index, w1=w1, rule=rule).set(
            cache_dtype=cfg.kv_cache.cache_dtype,
            kv_partition_spec=cfg.kv_cache.kv_partition_spec,
        ),
        mask=twa_attention_bias_config(w0=w0, coset_index=coset_index, w1=w1, rule=rule),
    )
    return cfg


def taper_slot(positions, q_max, *, geometry: TaperGeometry, coset_index: int):
    """`(resident, slot)` for each position under the taper rule; the cache half of the mask.

    The pyramid laid out as one flat buffer of `geometry.slots`: level 0 first as a `w0` ring, then
    each tapered level as a ring of `w0 // 2`.

        age      = q_max - position
        level    = 0 if age < w0 else floor(log2(age / w0)) + 1
        slot     = key mod w0                                                  at level 0
                 = w0 + (level - 1) * (w0//2) + ((key >> level) mod (w0//2))   above it
        resident = age < 2^K * w0  AND  v_2(key) >= level

    Level j spans ages `[2^(j-1) w0, 2^j w0)` and keeps the `w0/2` keys in it whose
    `v_2(key) >= j`, so `key >> j` walks `w0/2` consecutive integers there and the `mod` is a ring
    over exactly that level. Distinct levels take disjoint ranges, so the map is injective at every
    `q_max`. Every ring above level 0 is the SAME size, which is what makes this simpler than a
    fixed-width taper's -- there is no per-level base to derive, just `(level - 1) * (w0//2)`.

    A key's slot MOVES as it ages into the next level, which costs nothing: the scatter recomputes
    and re-places the whole buffer every step anyway.

    Sentinel-agnostic, and that is the caller's contract: `positions` must carry NO sentinel above
    the real range, and `q_max` must be computed with sentinels excluded. One positive sentinel
    makes `age` enormous for every real key and ages the whole buffer out at once. A negative
    sentinel cannot, which is why -1 is the normal form here even in `twa_cache`, whose buffer
    stores `+2^30`.

    Args:
        positions: positions to slot, any shape; sentinels must already read -1.
        q_max: the latest position after this chunk, sentinels excluded.
        geometry: the taper's `TaperGeometry` -- the same `(w0, w1)` the mask was built from, so
            the two cannot disagree about `K` or about how many slots exist.
        coset_index: this layer's coset, 0 or 1.

    Returns:
        `(resident, slot)`, both shaped like `positions`. `slot` is meaningless where `resident`
        is False -- the caller routes those to a trash slot.
    """
    w0, k_levels = geometry.w0, geometry.levels
    half = w0 // 2
    age = q_max - positions
    key = positions + coset_index

    # Residency is the MASK's predicate for BOTH rules, called rather than restated. Expressing it
    # here too is how a cache and a mask come to disagree about one key at one age.
    resident = taper_survives(q_max, positions, geometry=geometry, coset_index=coset_index)

    if geometry.rule is TaperRule.MIPMAP:
        # Rotating level L spans `w0` of age and keeps the `w0 >> L` keys in it whose
        # `v_2(key) >= L`, so `key >> L` walks consecutive integers there and the `mod` is a ring
        # over that level. `base(L) = 2*w0 - 2*(w0 >> L)` stacks the rings -- 0, w0, w0 + w0/2, ...
        # -- and is exact at L = 0 too. Levels 0..J-1 have `size >= 2`, so their slots stay below
        # `2*w0 - 2`, which leaves that one slot free for the anchor forever. That is what keeps the
        # map injective even while the anchor is young enough to also sit in level 0's ring.
        # `age >> shift` is the level; `k_levels` is how many rings exist, which `max_seq` may have
        # capped below `log2(w0)`. The anchor takes the slot just past the last ring, so it is
        # `geometry.slots - 1` rather than a constant.
        shift = w0.bit_length() - 1
        level = jnp.minimum(jnp.maximum(age >> shift, 0), k_levels - 1)
        size = jnp.right_shift(w0, level)
        rotating = 2 * w0 - 2 * size + jnp.remainder(jnp.right_shift(key, level), size)
        is_anchor = positions == anchor_index(k_levels, coset_index)
        return resident, jnp.where(is_anchor, geometry.slots - 1, rotating)

    # The level, unlike residency, is genuinely needed: it selects the ring. Accumulated comparisons
    # against the doubling boundaries, which is cheap here -- a short vector once per step, not a
    # mask tile three times per block.
    level = jnp.zeros_like(age)
    for t in range(k_levels):
        level = level + (age >= (w0 << t))

    tapered = w0 + (level - 1) * half + jnp.remainder(jnp.right_shift(key, level), half)
    return resident, jnp.where(level == 0, jnp.remainder(key, w0), tapered)

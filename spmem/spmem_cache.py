#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Decode-time KV cache for SpMem: a real taper buffer beside a memory pool.

    stored state       = [ memory pool | real taper buffer ]
    extend_step output = [ prior memory | prior reals | this block ]

  - Real buffer: the mipmap taper over real indices, re-slotted every step by `twa_cache.taper_slot`.
  - Memory pool: append-only, a ring over the last `mem_window_chunks` chunks (one-hot write, as in
    `SlidingWindowKVCache`), or the mipmap taper over memory ordinals.

The output uses the buffers from before this step's update. They were thinned for the previous
step's last query, which is no later than any query in this block, so they hold a superset of what
the mask admits here; the post-update buffers would be thinned too far for the block's early
queries. Positions are stored augmented, since that is what the mask reads; empty and pad slots
hold -1, which the mask rejects.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.kv_cache.base_kv_cache import BaseKVCache
from axlearn.common.module import nowrap
from axlearn.common.utils import Nested, Tensor, maybe_shard

from spmem.spmem_geometry import (
    SpMemGeometry,
    decompose_augmented_position,
)
from twa.twa_cache import taper_slot
from twa.twa_geometry import TaperGeometry

# Small and negative, so fp16-safe (cf. `SlidingWindowKVCache`'s `-(W+1)`).
_INVALID_POSITION = -1


class SpMemCache(BaseKVCache):
    """A mipmap taper buffer over the reals beside a memory pool. See the module docstring."""

    @config_class
    class Config(BaseKVCache.Config):
        """Configures SpMemCache.

        Attributes:
            geometry: the model's `SpMemGeometry`, shared with the mask.
            coset_index: this layer's taper coset, 0 or 1.
        """

        geometry: Required[SpMemGeometry] = REQUIRED
        coset_index: Required[int] = REQUIRED

    def __init__(self, cfg: Config, *, parent):
        super().__init__(cfg, parent=parent)
        # `taper_slot` reads the coset as a plain int, so a bad one would thin silently.
        if cfg.coset_index not in (0, 1):
            raise ValueError(f"coset_index must be 0 or 1, got {cfg.coset_index}")

    def _partition_specs(self):
        """(btnh_spec, bt_spec) from `kv_partition_spec`, or (None, None)."""
        cfg = self.config
        if cfg.kv_partition_spec is None:
            return None, None
        # BTNH, not `KVCache`'s BNHT: no decode kernel takes this cache, so BNHT would transpose the
        # whole output every step (measured 2% slower at 32K and 128K on a v5p, 12B, batch 1).
        return tuple(cfg.kv_partition_spec), (cfg.kv_partition_spec[0], None)

    @nowrap
    def init_states(self, shape: BaseKVCache.Shape, *, dtype: jnp.dtype) -> Nested[Tensor]:
        # `kv_len` is the augmented length; `SpMemDecoderMixin.init_states` inflates it.
        geometry = self.config.geometry
        btnh_spec, bt_spec = self._partition_specs()

        def buffers(size):
            btnh = (shape.batch_size, size, shape.num_kv_heads, shape.per_head_dim)
            zeros = jnp.zeros(shape=btnh, dtype=self._cache_dtype(dtype))
            positions = jnp.full((shape.batch_size, size), _INVALID_POSITION, dtype=jnp.int32)
            return (
                maybe_shard(zeros, btnh_spec),
                maybe_shard(zeros, btnh_spec),
                maybe_shard(positions, bt_spec),
            )

        mem_key, mem_value, mem_positions = buffers(geometry.memory_slots(shape.kv_len))
        win_key, win_value, win_positions = buffers(geometry.real_taper.slots)
        return dict(
            mem_key=mem_key,
            mem_value=mem_value,
            mem_positions=mem_positions,
            win_key=win_key,
            win_value=win_value,
            win_positions=win_positions,
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
        """Updates both buffers and returns `[prior memory | prior reals | block]`."""
        assert page_pool is None
        # The decoder always passes segment ids; 0 marks padding.
        assert segment_ids is not None, "SpMemCache.extend_step requires segment_ids ([B, T])."
        geometry = self.config.geometry
        batch, step_len = k_proj.shape[:2]
        key_pos = jnp.broadcast_to(key_positions, (batch, step_len))
        valid = segment_ids != 0
        _, off, _ = decompose_augmented_position(
            key_pos, mem_chunk_size=geometry.mem_chunk_size, num_mem=geometry.num_mem
        )
        is_mem = off >= geometry.mem_chunk_size
        block = dict(k_proj=k_proj, v_proj=v_proj, key_pos=key_pos)

        win_key, win_value, win_positions = self._update_reals(
            cached_states, valid=valid & ~is_mem, **block
        )
        if geometry.mem_taper is not None:
            update_memory = self._update_tapered_memory
        else:
            update_memory = self._update_ring_memory
        mem_key, mem_value, mem_positions = update_memory(
            cached_states, valid=valid & is_mem, **block
        )
        new_states = dict(
            mem_key=mem_key,
            mem_value=mem_value,
            mem_positions=mem_positions,
            win_key=win_key,
            win_value=win_value,
            win_positions=win_positions,
        )
        block_positions = jnp.where(valid, key_pos, _INVALID_POSITION)
        cat_key = jnp.concatenate([cached_states["mem_key"], cached_states["win_key"], k_proj], 1)
        cat_value = jnp.concatenate(
            [cached_states["mem_value"], cached_states["win_value"], v_proj], 1
        )
        cat_positions = jnp.concatenate(
            [cached_states["mem_positions"], cached_states["win_positions"], block_positions], -1
        )
        return new_states, self.Output(
            k_proj=cat_key, v_proj=cat_value, key_positions=cat_positions
        )

    def _update_reals(self, cached_states, *, k_proj, v_proj, key_pos, valid):
        """Re-slots the real buffer plus this block's valid reals by the real taper."""
        geometry = self.config.geometry
        keys, values, aug = _prior_plus_block(
            cached_states, "win", k_proj=k_proj, v_proj=v_proj, block_aug=_live(key_pos, valid)
        )
        _, _, real = decompose_augmented_position(
            aug, mem_chunk_size=geometry.mem_chunk_size, num_mem=geometry.num_mem
        )
        # The sentinel would otherwise decode as a valid-looking real index.
        index = jnp.where(aug != _INVALID_POSITION, real, -1)
        return self._reslot(keys, values, aug, index, taper=geometry.real_taper)

    def _update_tapered_memory(self, cached_states, *, k_proj, v_proj, key_pos, valid):
        """Re-slots the pool plus this block's memory by the taper over memory ordinals."""
        geometry = self.config.geometry
        c, m = geometry.mem_chunk_size, geometry.num_mem
        keys, values, aug = _prior_plus_block(
            cached_states, "mem", k_proj=k_proj, v_proj=v_proj, block_aug=_live(key_pos, valid)
        )
        aug_chunk, aug_off, _ = decompose_augmented_position(aug, mem_chunk_size=c, num_mem=m)
        ordinal = jnp.where(aug != _INVALID_POSITION, aug_chunk * m + (aug_off - c), -1)
        return self._reslot(keys, values, aug, ordinal, taper=geometry.mem_taper)

    def _update_ring_memory(self, cached_states, *, k_proj, v_proj, key_pos, valid):
        """Writes this block's memory to slot `(chunk mod ring_chunks) * m + j`.

        The ring size is read off the allocated pool (unbounded: the wrap is the identity). Only
        the block's last `ring_chunks` groups are written, since colliding one-hot writes would sum.
        """
        c, m = self.config.geometry.mem_chunk_size, self.config.geometry.num_mem
        chunk, off, _ = decompose_augmented_position(key_pos, mem_chunk_size=c, num_mem=m)
        capacity = cached_states["mem_positions"].shape[-1]
        ring_chunks = capacity // m
        max_chunk = jnp.max(jnp.where(valid, chunk, -1), axis=1, keepdims=True)
        in_ring = valid & (chunk > max_chunk - ring_chunks)
        slot = jnp.where(in_ring, (chunk % ring_chunks) * m + (off - c), capacity)
        # One-hot write: an out-of-range slot is an all-zero row, i.e. skipped.
        one_hot = jax.nn.one_hot(slot, capacity, dtype=jnp.int32)
        one_hot_kv = one_hot.astype(cached_states["mem_key"].dtype)
        keep = ~one_hot.any(axis=1)
        keep_kv = keep[:, :, None, None]
        return (
            cached_states["mem_key"] * keep_kv + jnp.einsum("btnh,bts->bsnh", k_proj, one_hot_kv),
            cached_states["mem_value"] * keep_kv + jnp.einsum("btnh,bts->bsnh", v_proj, one_hot_kv),
            cached_states["mem_positions"] * keep + jnp.einsum("bt,bts->bs", key_pos, one_hot),
        )

    def _reslot(self, keys, values, aug, index, *, taper: TaperGeometry):
        """Keeps the taper's survivors of `(keys, values, aug)`, each row by its own positions."""
        capacity = taper.slots

        def scatter_row(keys_row, values_row, aug_row, index_row):
            survives, slot = taper_slot(
                index_row, jnp.max(index_row), geometry=taper, coset_index=self.config.coset_index
            )
            # Non-survivors and sentinels go to a trash slot past the end.
            target = jnp.where((aug_row != _INVALID_POSITION) & survives, slot, capacity)
            key_out = jnp.zeros((capacity + 1, *keys_row.shape[1:]), dtype=keys_row.dtype)
            value_out = jnp.zeros((capacity + 1, *values_row.shape[1:]), dtype=values_row.dtype)
            aug_out = jnp.full((capacity + 1,), _INVALID_POSITION, dtype=jnp.int32)
            return (
                key_out.at[target].set(keys_row)[:capacity],
                value_out.at[target].set(values_row)[:capacity],
                aug_out.at[target].set(aug_row)[:capacity],
            )

        return jax.vmap(scatter_row)(keys, values, aug, index)


def _live(key_pos: Tensor, valid: Tensor) -> Tensor:
    return jnp.where(valid, key_pos, _INVALID_POSITION)


def _prior_plus_block(cached_states, prefix: str, *, k_proj, v_proj, block_aug):
    """The whole stored buffer followed by this block: a taper survivor can be arbitrarily old."""
    return (
        jnp.concatenate([cached_states[f"{prefix}_key"], k_proj], axis=1),
        jnp.concatenate([cached_states[f"{prefix}_value"], v_proj], axis=1),
        jnp.concatenate([cached_states[f"{prefix}_positions"], block_aug], axis=1),
    )

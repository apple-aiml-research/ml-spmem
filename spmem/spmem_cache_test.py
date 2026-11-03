#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Tests for spmem_cache.py -- SpMem two-buffer KV cache (memory pool + mipmap real taper).

`SpMemCache` keeps two SEPARATE buffers, returned concatenated and keyed by augmented position:

  * a mipmap TAPER buffer over the reals, of the geometry's slot count. It is not a ring: a survivor
    can be arbitrarily old, so the whole prior buffer takes part in every re-slot.
  * the memory pool (`m` per chunk): append-only, a ring under `mem_window_size`, or a second taper
    under `mem_w0`.

The `extend_step` OUTPUT is `concat([prior memory | prior taper buffer | this step's block])`, not
the post-update buffer: the stored buffer is thinned to the PREVIOUS step's last position, which is
a superset of what the mask admits at every query in the block.

These check that the returned buffer is a SUPERSET of what the mask selects (the cache and the mask
are independent implementations of one rule -- slot arithmetic against a closed-form predicate),
that a single block extend is BIT-IDENTICAL to token-by-token decode, that the two cosets retain
different reals, that ragged rows slot PER-ROW, and that padding writes nothing. Two intrinsic
invariants -- K tracks position, and live <= capacity -- are folded into `_run_cache`, so every
snapshot-based case exercises them.
"""

# pylint: disable=missing-function-docstring,protected-access,no-self-use

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from axlearn.common import test_utils
from axlearn.common.attention import GroupedQueryAttention

from spmem.spmem_apply import enable_spmem_attention
from spmem.spmem_cache import SpMemCache
from spmem.spmem_geometry import SpMemGeometry, spmem_taper_geometry
from spmem.spmem_mask import spmem_mask_fn
from twa.twa_cache import taper_slot

# The cache STORES a memory pool + a taper buffer; extend_step's OUTPUT appends this step's block.
_STATE_FIELDS = ("mem_key", "mem_value", "mem_positions", "win_key", "win_value", "win_positions")
# Empty and pad slots.
_SENTINEL = -1


class SpMemCacheTest(test_utils.TestCase):
    """The stored buffers, the returned operand, and their agreement with the mask."""

    # The mask's causal visible set at step q must be physically present in what attention reads --
    # the extend_step OUTPUT, not the retained state. This is the one cross-check that cannot pass
    # by a shared bug: `taper_slot`'s scatter and `spmem_mask_fn`'s predicate derive the same
    # retention rule by different routes, so agreement is evidence rather than tautology.
    @parameterized.parameters(
        # Reals must outrun the reach (`log2(w0) * w0`) for the anchor and every ring to matter.
        (2, 1, 4, 48, 0, None),
        (2, 1, 4, 48, 1, None),
        (4, 2, 8, 72, 0, None),
        (4, 2, 8, 72, 1, None),
        (2, 1, 4, 72, 0, 4),
        (2, 1, 4, 72, 1, 4),
    )
    def test_mask_visible_subset_of_cache(self, c, m, w, num_steps, coset, mem_w0):
        # One geometry on both sides.
        fn = spmem_mask_fn(_geometry(w, c, m, mem_w0=mem_w0), coset_index=coset)
        snapshots, _ = _run_cache(
            w0=w,
            mem_chunk_size=c,
            num_mem=m,
            coset_index=coset,
            num_steps=num_steps,
            mem_w0=mem_w0,
        )
        for q in range(num_steps):
            prior = _live_set(snapshots[q - 1][0]) if q > 0 else set()
            live = prior | {q}
            mask_vis = {k for k in range(q + 1) if bool(fn(np.int32(q), np.int32(k)))}
            self.assertTrue(
                mask_vis.issubset(live),
                f"q={q} (C={c} m={m} w0={w} coset={coset}): output missing "
                f"{sorted(mask_vis - live)}",
            )

    @parameterized.parameters((2, 1, 4, 48, 0, None), (4, 2, 8, 72, 1, None), (2, 1, 4, 72, 1, 4))
    def test_block_equals_token_by_token(self, c, m, w, length, coset, mem_w0):
        """A single multi-token extend must reproduce token-by-token decode EXACTLY.

        The taper's re-slot is vectorized over the concatenation of the prior buffer and the whole
        block, so a block spanning several levels is the case a per-token path never exercises --
        and a prefill is just the largest such block.
        """
        batch, heads, dim, msl = 2, 2, 4, 256
        k_all, v_all, pos = _inputs(batch, length, heads, dim, seed=0)

        decode_layer = _layer("decode", w, c, m, coset, mem_w0=mem_w0)
        decode_state = _empty(decode_layer, batch, msl, heads, dim)
        for t in range(length):
            decode_state, _ = decode_layer.extend_step(
                decode_state,
                k_proj=k_all[:, t : t + 1],
                v_proj=v_all[:, t : t + 1],
                key_positions=pos[:, t : t + 1],
                segment_ids=jnp.ones_like(pos[:, t : t + 1]),
            )
        block_layer = _layer("block", w, c, m, coset, mem_w0=mem_w0)
        block_state, _ = block_layer.extend_step(
            _empty(block_layer, batch, msl, heads, dim),
            k_proj=k_all,
            v_proj=v_all,
            key_positions=pos,
            segment_ids=jnp.ones_like(pos),
        )
        for field in _STATE_FIELDS:
            np.testing.assert_array_equal(
                np.asarray(block_state[field]), np.asarray(decode_state[field]), err_msg=field
            )

    @parameterized.parameters((2, 1, 4, 48, 0), (4, 2, 8, 72, 1))
    def test_multiblock_prefill_is_partition_independent(self, c, m, w, length, coset):
        """Splitting the stream must not change the state: the second block lands in a NON-EMPTY
        cache, so its re-slot has to carry the prior buffer's survivors through levels it did not
        itself reach."""
        batch, heads, dim, msl = 2, 2, 4, 256
        split = (length // 2 // (c + m)) * (c + m)  # a chunk boundary, so no memory group is split
        self.assertGreater(split, 0)
        k_all, v_all, pos = _inputs(batch, length, heads, dim, seed=1)

        one_layer = _layer("one", w, c, m, coset)
        one_state, _ = one_layer.extend_step(
            _empty(one_layer, batch, msl, heads, dim),
            k_proj=k_all,
            v_proj=v_all,
            key_positions=pos,
            segment_ids=jnp.ones_like(pos),
        )
        two_layer = _layer("two", w, c, m, coset)
        two_state = _empty(two_layer, batch, msl, heads, dim)
        for lo, hi in ((0, split), (split, length)):
            two_state, _ = two_layer.extend_step(
                two_state,
                k_proj=k_all[:, lo:hi],
                v_proj=v_all[:, lo:hi],
                key_positions=pos[:, lo:hi],
                segment_ids=jnp.ones_like(pos[:, lo:hi]),
            )
        for field in _STATE_FIELDS:
            np.testing.assert_array_equal(
                np.asarray(two_state[field]), np.asarray(one_state[field]), err_msg=field
            )

    def test_real_buffer_is_the_geometrys_slot_count_and_fills_it(self):
        """The slot count is the taper's asymptote. A wrong capacity would not fail loudly: the
        scatter sends non-survivors to a trash slot, so an undersized buffer silently drops reals
        the mask still admits, and an oversized one just wastes HBM at 128K."""
        c, m, w, steps = 4, 2, 8, 120
        snapshots, layer = _run_cache(
            w0=w, mem_chunk_size=c, num_mem=m, coset_index=0, num_steps=steps
        )
        slots = spmem_taper_geometry(w0=w).slots
        state = _empty(layer, 1, steps, 1, 2)
        self.assertEqual(np.asarray(state["win_positions"]).shape[-1], slots)
        period = c + m
        counts = [
            len({p for p in _live_set(snap_pos) if (p % period) < c}) for snap_pos, _ in snapshots
        ]
        self.assertLessEqual(max(counts), slots)
        self.assertEqual(counts[-1], slots, "the buffer never filled past warmup")

    def test_cosets_retain_different_reals(self):
        """The 2-way split is only worth its complexity if the two halves actually diverge -- a
        coset that changed nothing gives every layer the same survivors and no decorrelation."""
        c, m, w, steps = 2, 1, 4, 48
        live = {}
        for coset in (0, 1):
            snapshots, _ = _run_cache(
                w0=w, mem_chunk_size=c, num_mem=m, coset_index=coset, num_steps=steps
            )
            period = c + m
            live[coset] = {p for p in _live_set(snapshots[-1][0]) if (p % period) < c}
        self.assertNotEqual(live[0], live[1], "both cosets retained the same reals")
        self.assertTrue(live[0] and live[1], "a coset retained no reals at all")

    def test_memory_pool_holds_every_memory_token(self):
        """Untapered and unwindowed, the pool must be append-only.

        The `live <= capacity` invariant in `_run_cache` cannot catch an undersized pool -- an
        overflowing write is a dropped one-hot row, which makes the live set SMALLER and so
        satisfies that bound more easily. This is the converse assertion, over both feed schedules
        because the slot arithmetic is per-token either way.
        """
        c, m, w, l_real = 8, 2, 16, 64
        period = c + m
        n_chunks = l_real // c
        aug_len = n_chunks * period
        expected = [j * period + c + i for j in range(n_chunks) for i in range(m)]
        pos = jnp.arange(aug_len, dtype=jnp.int32)[None]
        kv = jnp.broadcast_to(pos.astype(jnp.float32)[..., None, None], (1, aug_len, 1, 2))
        for label, step in (("one block", aug_len), ("token by token", 1)):
            with self.subTest(label):
                layer = _layer(f"pool_{step}", w, c, m, coset_index=0)
                state = _empty(layer, 1, aug_len, 1, 2)
                for t in range(0, aug_len, step):
                    state, _ = layer.extend_step(
                        state,
                        k_proj=kv[:, t : t + step],
                        v_proj=kv[:, t : t + step],
                        key_positions=pos[:, t : t + step],
                        segment_ids=jnp.ones_like(pos[:, t : t + step]),
                    )
                stored = np.asarray(state["mem_positions"])[0]
                got = sorted(stored[stored != _SENTINEL].tolist())
                self.assertEqual(got, expected, f"{label}: memory tokens missing from the pool")

    def test_multibatch_ragged_positions_are_per_row(self):
        # Left-aligned ragged padding: row 0 has 5 valid tokens (all inside the dense window), row 1
        # all 48 (every ring). `scatter_row` takes `q_max` from its OWN row, so the surviving sets
        # must differ -- a shared schedule would thin row 0 by row 1's age.
        c, m, w, msl = 2, 1, 4, 128
        layer = _layer("rag", w, c, m, coset_index=0)
        state = _empty(layer, 2, msl, 2, 4)
        length = 48
        kv = jnp.broadcast_to(
            jnp.arange(length, dtype=jnp.float32)[None, :, None, None], (2, length, 2, 4)
        )
        pos = jnp.arange(length, dtype=jnp.int32)[None, :].repeat(2, axis=0)
        seg = jnp.stack(
            [
                jnp.array([1] * 5 + [0] * (length - 5), dtype=jnp.int32),
                jnp.array([1] * length, dtype=jnp.int32),
            ]
        )
        state, _ = layer.extend_step(
            state, k_proj=kv, v_proj=kv, key_positions=pos, segment_ids=seg
        )
        row0 = _live_set(_cat(state, "positions")[0])
        row1 = _live_set(_cat(state, "positions")[1])
        # Reals are augmented 0,1,3,4 -> real 0..3, all younger than `w0=4`, so all kept verbatim.
        # Augmented 2 is memory and lives in the pool, which the taper does not touch.
        self.assertEqual(row0, {0, 1, 2, 3, 4}, "row 0's own taper, on its own q_max")
        self.assertTrue(row1 - row0, "row 1 must retain positions row 0 never saw")
        self.assertTrue(row0 - row1, "row 1 must have thinned what row 0 still holds")

    def test_padding_tokens_write_nothing(self):
        # A segment_ids=0 token must not occupy a slot, and -- specific to the taper -- must not
        # advance `q_max` either, since a spurious q_max would age out live reals.
        c, m, w, msl = 4, 2, 4, 128
        layer = _layer("pad", w, c, m, coset_index=0)
        state = _empty(layer, 1, msl, 2, 4)
        for q in range(6 * (c + m)):
            kv = jnp.full((1, 1, 2, 4), float(q), dtype=jnp.float32)
            state, _ = layer.extend_step(
                state,
                k_proj=kv,
                v_proj=kv,
                key_positions=jnp.asarray([[q]], dtype=jnp.int32),
                segment_ids=jnp.asarray([[1]], dtype=jnp.int32),
            )
        before_pos = _cat(state, "positions")[0].copy()
        before_key = _cat(state, "key")[0, :, 0, 0].copy()
        state, _ = layer.extend_step(
            state,
            k_proj=jnp.full((1, 1, 2, 4), 999.0),
            v_proj=jnp.full((1, 1, 2, 4), 999.0),
            key_positions=jnp.asarray([[6 * (c + m) + 100]], dtype=jnp.int32),
            segment_ids=jnp.asarray([[0]], dtype=jnp.int32),
        )
        np.testing.assert_array_equal(_cat(state, "positions")[0], before_pos)
        np.testing.assert_array_equal(_cat(state, "key")[0, :, 0, 0], before_key)

    def test_rejects_a_coset_outside_the_pair(self):
        # `taper_slot` reads it as a plain int, so a bad value thins differently, not fails.
        with self.assertRaisesRegex(ValueError, "coset_index must be 0 or 1"):
            _layer("bad", w0=4, mem_chunk_size=4, num_mem=2, coset_index=2)

    def test_enable_installs_one_geometry_on_both_halves(self):
        """`enable_spmem_attention` must wire cache and mask from ONE call: a cache and mask given
        different geometry or cosets is a train/decode divergence nothing else reports."""
        cfg = GroupedQueryAttention.default_config().set(name="a", num_heads=4)
        geometry = _geometry(8, 4, 2)
        enable_spmem_attention(cfg, geometry=geometry, coset_index=1)
        self.assertIs(cfg.kv_cache.klass, SpMemCache)
        self.assertEqual((cfg.kv_cache.geometry, cfg.kv_cache.coset_index), (geometry, 1))
        self.assertIs(cfg.mask.mask, spmem_mask_fn(geometry, coset_index=1))


class SpMemTaperedMemoryTest(test_utils.TestCase):
    """`mem_w0`: the memory pool becomes a second mipmap taper, in memory-SLOT units.

    Without it the pool grows with the context -- 131,072 slots at 1M reals under C8M1. These pin
    the fixed capacity and the thinning.
    """

    def test_the_pool_size_comes_from_the_geometry_not_the_sequence(self):
        mem_w0 = 8
        want = spmem_taper_geometry(w0=mem_w0).slots
        for kv_len in (64, 4096, 131072):
            cache = _layer("c", w0=4, mem_chunk_size=2, num_mem=1, coset_index=0, mem_w0=mem_w0)
            states = _empty(cache, 1, kv_len, 2, 4)
            self.assertEqual(states["mem_positions"].shape[-1], want, f"kv_len={kv_len}")

    def test_the_pool_thins_instead_of_dropping_the_oldest(self):
        """The reason to prefer a taper over `mem_window_size`: same budget, but what survives spans
        the whole prefix rather than the most recent groups. A ring would hold a contiguous tail."""
        c, m, mem_w0 = 2, 1, 4
        slots = spmem_taper_geometry(w0=mem_w0).slots
        cache = _layer("c", w0=4, mem_chunk_size=c, num_mem=m, coset_index=0, mem_w0=mem_w0)
        states = _empty(cache, 1, 256, 2, 4)
        steps = 20 * (c + m)
        for pos in range(steps):
            k = jnp.ones((1, 1, 2, 4), dtype=jnp.float32) * (pos + 1)
            states, _ = cache.extend_step(
                states,
                k_proj=k,
                v_proj=k,
                key_positions=jnp.asarray([[pos]], dtype=jnp.int32),
                segment_ids=jnp.ones((1, 1), dtype=jnp.int32),
            )
        live = sorted(_live_set(np.asarray(states["mem_positions"])[0]))
        self.assertLessEqual(len(live), slots)
        self.assertTrue(live, "the pool went empty")
        span = (live[-1] - live[0]) // (c + m) + 1
        self.assertGreater(span, len(live), f"live groups {live} span only {span} chunks")

    def test_a_positive_sentinel_would_thin_the_whole_buffer(self):
        """The contract `taper_slot` cannot enforce, and the reason both callers normalize to -1.

        `q_max` is a plain max over the positions handed in, so one sentinel above the real range
        ages every real key past the reach. This pins the failure so a third caller passing its own
        sentinel through fails here rather than by silently returning an empty cache.
        """
        geometry = spmem_taper_geometry(w0=4)
        positions = jnp.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.int32)
        good, _ = taper_slot(positions, jnp.max(positions), geometry=geometry, coset_index=0)
        poisoned = positions.at[7].set(1 << 30)  # a foreign sentinel left in place
        bad, _ = taper_slot(poisoned, jnp.max(poisoned), geometry=geometry, coset_index=0)
        self.assertGreater(int(jnp.sum(good)), int(jnp.sum(bad)) + 1)


class SpMemWindowedMemoryTest(test_utils.TestCase):
    """`mem_window_size` turns the append-only pool into a ring over the chunks it spans.

    The bound is stated in REAL tokens and shared with `spmem_mask_fn`, and these pin the two
    together: the mask must never ask for a slot the ring has already reused.
    """

    _W, _C, _M = 8, 4, 2

    def _feed(self, *, mem_window_size, steps, block_len=1):
        """Feed positions 0..steps-1 in `block_len`-sized blocks (1 = decode, `steps` = a bulk
        prefill); return (final state, per-call available set, final stored memory)."""
        layer = _layer(
            "c", self._W, self._C, self._M, coset_index=0, mem_window_size=mem_window_size
        )
        state = _empty(layer, 1, steps, 1, 2)
        pos = jnp.arange(steps, dtype=jnp.int32)[None]
        kv = jnp.broadcast_to(pos.astype(jnp.float32)[..., None, None], (1, steps, 1, 2))
        available = []
        for q in range(0, steps, block_len):
            block = slice(q, q + block_len)
            state, out = layer.extend_step(
                state,
                k_proj=kv[:, block],
                v_proj=kv[:, block],
                key_positions=pos[:, block],
                segment_ids=jnp.ones_like(pos[:, block]),
            )
            available.append({int(p) for p in np.asarray(out.key_positions)[0] if int(p) >= 0})
        stored = sorted(int(p) for p in np.asarray(state["mem_positions"])[0] if int(p) >= 0)
        return state, available, stored

    @parameterized.parameters(dict(mem_window_size=12, chunks=3), dict(mem_window_size=8, chunks=2))
    def test_ring_keeps_exactly_the_windowed_chunks(self, mem_window_size, chunks):
        steps = 60  # 10 chunks of period 6, far past the window
        state, _avail, stored = self._feed(mem_window_size=mem_window_size, steps=steps)
        cap = chunks * self._M
        self.assertEqual(np.asarray(state["mem_positions"]).shape[-1], cap)
        period = self._C + self._M
        last_chunk = (steps - 1) // period
        want = [
            g * period + self._C + j
            for g in range(last_chunk - chunks + 1, last_chunk + 1)
            for j in range(self._M)
        ]
        self.assertEqual(stored, want)

    def test_mask_never_asks_for_an_evicted_slot(self):
        """Were the mask wider than the ring, attention would read a slot another chunk has already
        overwritten -- silently, since the stale entry is a valid key at a valid position."""
        steps, mem_window_size = 60, 12
        _state, available, _stored = self._feed(mem_window_size=mem_window_size, steps=steps)
        fn = spmem_mask_fn(
            _geometry(self._W, self._C, self._M, mem_window_size=mem_window_size), coset_index=0
        )
        for q in range(steps):
            # Causal keys only; the own-chunk sibling term needs batched insertion at decode.
            want = {k for k in range(q + 1) if bool(fn(np.array(q), np.array(k)))}
            self.assertEqual(want - available[q], set(), f"q={q} reads evicted slots")

    def test_pool_never_outgrows_the_sequence(self):
        """A window wider than the stream must not allocate slots the stream cannot fill: the cap is
        the `min` of the two, so a short run behaves exactly like the unbounded pool."""
        steps = 60
        state, _avail, stored = self._feed(mem_window_size=1000, steps=steps)
        self.assertEqual(
            np.asarray(state["mem_positions"]).shape[-1],
            _geometry(self._W, self._C, self._M).memory_slots(steps),
        )
        self.assertLen(stored, 20)  # nothing evicted -- identical to the unbounded pool

    def test_bulk_prefill_matches_step_by_step(self):
        """A prefill arrives as ONE block spanning every chunk, and `_write`'s one-hot einsum SUMS
        two tokens landing in the same slot -- it has no last-wins. So the ring must drop the
        block's older groups; otherwise a 60-token prefill folds 10 chunks' memory into 3 slots,
        and the summed positions are still positive, so no sentinel guard catches it."""
        kwargs = dict(mem_window_size=12, steps=60)
        per_token, _avail, _stored = self._feed(**kwargs)
        bulk, _avail2, _stored2 = self._feed(**kwargs, block_len=kwargs["steps"])
        for field in _STATE_FIELDS:
            test_utils.assert_allclose(bulk[field], per_token[field], err_msg=field)


def _reach(w0):
    """The mipmap rule's full receptive field at `w0`, `log2(w0) * w0`."""
    return spmem_taper_geometry(w0=w0).w1


def _geometry(w0, mem_chunk_size, num_mem, *, mem_w0=None, mem_window_size=None):
    return SpMemGeometry(
        mem_chunk_size=mem_chunk_size,
        num_mem=num_mem,
        w0=w0,
        mem_window_size=mem_window_size,
        mem_w0=mem_w0,
        mem_w1=None if mem_w0 is None else _reach(mem_w0),
    )


def _layer(name, w0, mem_chunk_size, num_mem, coset_index, mem_w0=None, mem_window_size=None):
    geometry = _geometry(
        w0, mem_chunk_size, num_mem, mem_w0=mem_w0, mem_window_size=mem_window_size
    )
    return (
        SpMemCache.default_config()
        .set(name=name, geometry=geometry, coset_index=coset_index)
        .instantiate(parent=None)
    )


def _empty(layer, batch, kv_len, heads, dim):
    return layer.init_states(
        shape=SpMemCache.Shape(
            batch_size=batch, kv_len=kv_len, num_kv_heads=heads, per_head_dim=dim
        ),
        dtype=jnp.float32,
    )


def _inputs(batch, length, heads, dim, *, seed):
    rng = jax.random.PRNGKey(seed)
    k_all = jax.random.normal(jax.random.fold_in(rng, 1), (batch, length, heads, dim))
    v_all = jax.random.normal(jax.random.fold_in(rng, 2), (batch, length, heads, dim))
    pos = jnp.arange(length, dtype=jnp.int32)[None, :].repeat(batch, axis=0)
    return k_all, v_all, pos


def _cat(state, field: str) -> np.ndarray:
    """Concatenate memory pool ++ taper buffer for `field` in {key, value, positions}.

    Time is axis 1 for the BTNH kv buffers and the last axis for the [B, T] position buffers.
    """
    parts = [np.asarray(state[f"mem_{field}"]), np.asarray(state[f"win_{field}"])]
    return np.concatenate(parts, axis=1 if parts[0].ndim == 4 else -1)


def _live_set(snapshot: np.ndarray) -> set[int]:
    return {int(p) for p in snapshot if int(p) != _SENTINEL}


def _run_cache(
    *,
    w0,
    mem_chunk_size,
    num_mem,
    coset_index,
    num_steps,
    max_seq_len=512,
    num_kv_heads=2,
    per_head_dim=4,
    mem_w0=None,
):
    """Feed positions 0..num_steps-1 one at a time; snapshot the STORED state each step.

    K encodes the augmented position (`K[:, t] == t`), so a slot's stored K reveals which position
    last wrote it. Two intrinsic invariants are asserted at EVERY step:
      - each live slot's stored K equals the augmented position it claims (K/pos stay aligned);
      - the live set never exceeds the physical buffer.
    """
    layer = _layer("c", w0, mem_chunk_size, num_mem, coset_index, mem_w0)
    state = _empty(layer, 1, max_seq_len, num_kv_heads, per_head_dim)
    kv = jnp.broadcast_to(
        jnp.arange(num_steps, dtype=jnp.float32)[None, :, None, None],
        (1, num_steps, num_kv_heads, per_head_dim),
    )
    pos = jnp.arange(num_steps, dtype=jnp.int32)[None, :]
    geometry = layer.config.geometry
    cap = geometry.memory_slots(max_seq_len) + geometry.real_taper.slots
    snapshots = []
    for q in range(num_steps):
        prev = state
        state, out = layer.extend_step(
            prev,
            k_proj=kv[:, q : q + 1],
            v_proj=kv[:, q : q + 1],
            key_positions=pos[:, q : q + 1],
            segment_ids=jnp.ones_like(pos[:, q : q + 1]),
        )
        # Output is concat(PRIOR buffers, this step's block): prior positions ++ the fed one.
        expected_pos = np.concatenate([_cat(prev, "positions"), np.array([[q]], np.int32)], axis=-1)
        np.testing.assert_array_equal(np.asarray(out.key_positions), expected_pos)
        snap_pos = _cat(state, "positions")[0]
        snap_key = _cat(state, "key")[0, :, 0, 0]
        live = snap_pos != _SENTINEL
        np.testing.assert_array_equal(snap_key[live], snap_pos[live])
        assert int(live.sum()) <= cap, f"q={q}: live {int(live.sum())} exceeds buffer {cap}"
        snapshots.append((snap_pos, snap_key))
    return snapshots, layer


if __name__ == "__main__":
    absltest.main()

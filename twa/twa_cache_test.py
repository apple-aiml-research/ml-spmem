#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Tests for twa_cache.py -- the TWA decode cache.

Two oracles, no Python re-implementation of the algorithm:

  1. **Literal expected positions** (`test_cache_positions_W8_*`) — the cache is
     run for 40 decode steps and its post-update `key_positions` are asserted
     against hand-frozen lists at each block boundary. The expected behavior is
     stated outright, obvious at a glance.

  2. **Spec cross-check** (`test_cache_output_matches_mask`) — the cache
     (inference-time implementation) must expose exactly the positions that
     `twa_mask_fn` (the training-time spec) deems visible, at every step
     across a grid of (W, coset). This is the implementation-matches-algorithm
     check.
"""

# pylint: disable=missing-function-docstring,protected-access,no-self-use

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from axlearn.common.attention import GroupedQueryAttention
from axlearn.common.attention_bias import MaskFnAttentionBias

from twa.twa_cache import (
    TWACache,
    enable_twa_attention,
)
from twa.twa_geometry import taper_geometry
from twa.twa_mask import twa_mask_fn


def _run_jax_cache(
    *,
    w0: int,
    coset_index: int,
    num_steps: int,
    batch: int = 1,
    num_kv_heads: int = 2,
    per_head_dim: int = 4,
    levels: int = 3,
):
    """Run JAX TWACache for `num_steps` decode steps; return per-step snapshots."""
    layer = (
        TWACache.default_config(w0=w0, w1=w0 << levels, coset_index=coset_index)
        .set(
            name="thin",
        )
        .instantiate(parent=None)
    )
    cap = layer._max_capacity()
    state = layer.init_states(
        shape=TWACache.Shape(
            batch_size=batch, kv_len=cap, num_kv_heads=num_kv_heads, per_head_dim=per_head_dim
        ),
        dtype=jnp.float32,
    )

    rng = jax.random.PRNGKey(0)
    k_all = jax.random.normal(
        jax.random.fold_in(rng, 1), (batch, num_steps, num_kv_heads, per_head_dim)
    )
    v_all = jax.random.normal(
        jax.random.fold_in(rng, 2), (batch, num_steps, num_kv_heads, per_head_dim)
    )
    snapshots = []
    for q in range(num_steps):
        state, _ = layer.extend_step(
            state,
            k_proj=k_all[:, q : q + 1],
            v_proj=v_all[:, q : q + 1],
            key_positions=jnp.asarray([[q]], dtype=jnp.int32),
        )
        snapshots.append(np.array(state["key_positions"][0]))
    return snapshots, layer


def _live_set(positions_snapshot: np.ndarray, invalid_sentinel: int) -> set[int]:
    return set(int(p) for p in positions_snapshot if int(p) != invalid_sentinel)


def _live_sorted(positions_snapshot: np.ndarray, invalid_sentinel: int) -> list[int]:
    return sorted(_live_set(positions_snapshot, invalid_sentinel))


class TWACacheTest(parameterized.TestCase):
    # ------------------------------------------------------------------------
    # Literal expected positions: the cache's post-update key_positions at each
    # block boundary, stated outright (no stub).
    # ------------------------------------------------------------------------

    def test_cache_positions_w0_8_coset0_literal(self):
        """`w0=8, K=3`: 20 slots, reach 64. Stated outright so the shape of the schedule is visible.

        Read a row right to left. The last 8 are always there -- level 0, the dense window. Before
        them the density halves per octave: ages 8..15 keep the evens, 16..31 the multiples of 4,
        32..63 the multiples of 8. There is NO warmup zone, unlike a halve-on-overflow buffer:
        thinning starts the moment a key is older than `w0`, which is why q=15 already has gaps.
        """
        snapshots, layer = _run_jax_cache(w0=8, coset_index=0, num_steps=40)
        sentinel = layer._invalid_position()
        self.assertEqual(_live_sorted(snapshots[7], sentinel), [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(
            _live_sorted(snapshots[15], sentinel),
            [0, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15],
        )
        self.assertEqual(
            _live_sorted(snapshots[23], sentinel),
            [0, 4, 8, 10, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23],
        )
        self.assertEqual(
            _live_sorted(snapshots[31], sentinel),
            [0, 4, 8, 12, 16, 18, 20, 22, 24, 25, 26, 27, 28, 29, 30, 31],
        )
        self.assertEqual(
            _live_sorted(snapshots[39], sentinel),
            [0, 8, 12, 16, 20, 24, 26, 28, 30, 32, 33, 34, 35, 36, 37, 38, 39],
        )

    def test_cache_positions_w0_8_coset1_literal(self):
        """Same geometry, opposite coset: the valuation is taken on `k + 1`, so where coset 0 keeps
        the evens this keeps the odds. The dense window is identical -- only the tapered levels
        move, which is the whole point of the 2-way split."""
        snapshots, layer = _run_jax_cache(w0=8, coset_index=1, num_steps=40)
        sentinel = layer._invalid_position()
        self.assertEqual(_live_sorted(snapshots[7], sentinel), [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(
            _live_sorted(snapshots[15], sentinel),
            [1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        )
        self.assertEqual(
            _live_sorted(snapshots[23], sentinel),
            [3, 7, 9, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        )
        self.assertEqual(
            _live_sorted(snapshots[31], sentinel),
            [3, 7, 11, 15, 17, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        )
        self.assertEqual(
            _live_sorted(snapshots[39], sentinel),
            [7, 11, 15, 19, 23, 25, 27, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        )

    # ------------------------------------------------------------------------
    # Spec cross-check: the cache (inference impl) exposes exactly the positions
    # mask_fn (training spec) deems visible, at every step. Moved here from
    # twa_mask_test.py — verifying impl-matches-algorithm is naturally a
    # kv_cache test. The cache snapshot at step q is the post-update buffer,
    # which equals the mask's visible set {k <= q : mask_fn(q, k)} exactly.
    # ------------------------------------------------------------------------

    @parameterized.product(
        geometry=[(8, 0, 40), (8, 1, 40), (16, 0, 80), (16, 1, 80), (32, 0, 160)],
        levels=[2, 3],
    )
    def test_cache_output_matches_mask(self, geometry, levels):
        """Cache valid key_positions at step q == {k <= q : mask_fn(q, k)}.

        Run for BOTH retention rules with the same flag on both sides -- this is the only test that
        ties the closed form in the mask to the buffer the cache actually holds, so a rule that has
        it on one side only is a rule with no evidence behind it.
        """
        W, coset, num_steps = geometry
        fn = twa_mask_fn(w0=W, coset_index=coset, w1=W << levels)
        snapshots, layer = _run_jax_cache(
            w0=W, coset_index=coset, num_steps=num_steps, levels=levels
        )
        sentinel = layer._invalid_position()
        for q in range(num_steps):
            cache_visible = _live_set(snapshots[q], sentinel)
            mask_visible = {k for k in range(q + 1) if bool(fn(np.int32(q), np.int32(k)))}
            self.assertEqual(
                cache_visible,
                mask_visible,
                f"w0={W} K={levels} coset={coset} q={q}: "
                f"cache-only={sorted(cache_visible - mask_visible)}, "
                f"mask-only={sorted(mask_visible - cache_visible)}",
            )

    def test_cross_coset_union_at_steady_state(self):
        """At T=39 the coset-0 and coset-1 caches differ; their union covers
        more positions than either alone (the point of assigning sibling
        layers different cosets). Both still hold the current dense block
        [32, 39].
        """
        snapshots_0, layer_0 = _run_jax_cache(w0=8, coset_index=0, num_steps=40)
        snapshots_1, _ = _run_jax_cache(w0=8, coset_index=1, num_steps=40)
        sentinel = layer_0._invalid_position()
        coset0 = _live_set(snapshots_0[39], sentinel)
        coset1 = _live_set(snapshots_1[39], sentinel)
        union = coset0 | coset1
        # Both caches contain [32..39], the dense window at q=39.
        for t in range(32, 40):
            self.assertIn(t, coset0, f"block 4 token {t} missing from coset 0")
            self.assertIn(t, coset1, f"block 4 token {t} missing from coset 1")
        # Union strictly larger than either single coset.
        self.assertGreater(len(union), len(coset0))
        self.assertGreater(len(union), len(coset1))

    # ------------------------------------------------------------------------
    # K/V correctness — slot contents track positions
    # ------------------------------------------------------------------------

    def test_kv_tracks_positions(self):
        """K[slot] inserted with value `q` survives to wherever key_positions[slot] == q."""
        W, coset = 8, 0
        num_steps = 32
        layer = (
            TWACache.default_config(w0=W, w1=W * 8, coset_index=coset)
            .set(
                name="kv-track",
            )
            .instantiate(parent=None)
        )
        cap = layer._max_capacity()
        state = layer.init_states(
            shape=TWACache.Shape(batch_size=1, kv_len=cap, num_kv_heads=1, per_head_dim=1),
            dtype=jnp.float32,
        )
        # Insert tokens with K=V=position (so slot contents are identifiable).
        for q in range(num_steps):
            k_new = jnp.full((1, 1, 1, 1), float(q), dtype=jnp.float32)
            v_new = jnp.full((1, 1, 1, 1), float(q), dtype=jnp.float32)
            state, _ = layer.extend_step(
                state,
                k_proj=k_new,
                v_proj=v_new,
                key_positions=jnp.asarray([[q]], dtype=jnp.int32),
            )

        positions = np.array(state["key_positions"][0])  # [cap]
        keys = np.array(state["key"][0, 0, 0, :])  # [cap]
        sentinel = layer._invalid_position()
        for slot, p in enumerate(positions):
            if p == sentinel:
                continue
            # K[slot] was set when token at position p was inserted; we set K = float(p) then.
            # Halving doesn't modify K/V at survivor slots; it just shuffles them.
            self.assertEqual(
                keys[slot],
                p,
                f"slot={slot}: key_position={p} but K[slot]={keys[slot]} — token mixup",
            )

    # ------------------------------------------------------------------------
    # Prefill: step_len > 1 matches step-by-step decode
    # ------------------------------------------------------------------------

    @parameterized.parameters(
        (8, 0, 8),  # dense window only
        (8, 0, 16),  # first tapered level full
        (8, 0, 17),  # first key thinned
        (8, 0, 24),
        (8, 1, 35),  # coset 1, several levels
        (16, 0, 80),  # every level populated
        (16, 1, 81),
    )
    def test_prefill_bit_identical_to_decode(self, W, coset, T):
        """Multi-token prefill (step_len=T from empty) == per-token decode, bit-for-bit.

        Decode (step_len=1) is the reference; the chunked prefill must reproduce
        the SAME buffer — key, value, key_positions — not just the position set,
        since decode's next re-slot depends on the exact slot layout. Both paths are
        fed identical K/V so the values are comparable too.
        """
        B, N, H = 2, 2, 4
        cap = 2 * W

        def _layer(name):
            return (
                TWACache.default_config(w0=W, w1=W * 8, coset_index=coset)
                .set(name=name)
                .instantiate(parent=None)
            )

        def _empty(layer):
            return layer.init_states(
                shape=TWACache.Shape(batch_size=B, kv_len=cap, num_kv_heads=N, per_head_dim=H),
                dtype=jnp.float32,
            )

        rng = jax.random.PRNGKey(0)
        k_all = jax.random.normal(jax.random.fold_in(rng, 1), (B, T, N, H))
        v_all = jax.random.normal(jax.random.fold_in(rng, 2), (B, T, N, H))
        pos = jnp.arange(T, dtype=jnp.int32)[None, :].repeat(B, axis=0)

        # Reference: feed one token at a time (step_len=1).
        decode_layer = _layer("decode")
        state = _empty(decode_layer)
        for t in range(T):
            state, _ = decode_layer.extend_step(
                state,
                k_proj=k_all[:, t : t + 1],
                v_proj=v_all[:, t : t + 1],
                key_positions=pos[:, t : t + 1],
            )

        # Prefill: one extend_step over the whole chunk (step_len=T).
        prefill_layer = _layer("prefill")
        got, out = prefill_layer.extend_step(
            _empty(prefill_layer), k_proj=k_all, v_proj=v_all, key_positions=pos
        )

        for field in ("key", "value", "key_positions"):
            np.testing.assert_array_equal(
                np.asarray(got[field]), np.asarray(state[field]), err_msg=f"state field={field}"
            )
        # Output K/V are the post-update buffer transposed to BSNH; check them too.
        np.testing.assert_array_equal(
            np.einsum("bnht->btnh", np.asarray(state["key"])), np.asarray(out.k_proj)
        )
        np.testing.assert_array_equal(
            np.asarray(state["key_positions"]), np.asarray(out.key_positions)
        )

    @parameterized.parameters(
        (8, 0, 10, 10),  # chunk B crosses the first tapered level from a non-empty cache
        (8, 1, 16, 8),  # chunk A ends on a level boundary; B's first write thins
        (8, 0, 5, 20),  # small A, large B spanning several levels
        (16, 0, 20, 40),  # several levels split across two chunks
        (16, 1, 33, 31),  # odd splits, coset 1
    )
    def test_multistep_from_nonempty_matches_single_token(self, W, coset, T_a, T_b):
        """A multi-token chunk fed onto a NON-empty cache == per-token decode.

        The proof that one scan-based `extend_step` is correct from any
        starting state: feed chunk A (step_len=T_a), then chunk B
        (step_len=T_b) onto the resulting non-empty cache, and the final
        buffer must be bit-identical to feeding A+B one token at a time.
        """
        B, N, H = 2, 2, 4
        cap = 2 * W
        total = T_a + T_b

        def _layer(name):
            return (
                TWACache.default_config(w0=W, w1=W * 8, coset_index=coset)
                .set(name=name)
                .instantiate(parent=None)
            )

        def _empty(layer):
            return layer.init_states(
                shape=TWACache.Shape(batch_size=B, kv_len=cap, num_kv_heads=N, per_head_dim=H),
                dtype=jnp.float32,
            )

        rng = jax.random.PRNGKey(0)
        k_all = jax.random.normal(jax.random.fold_in(rng, 1), (B, total, N, H))
        v_all = jax.random.normal(jax.random.fold_in(rng, 2), (B, total, N, H))
        pos = jnp.arange(total, dtype=jnp.int32)[None, :].repeat(B, axis=0)

        # Reference: feed one token at a time over the whole sequence.
        decode_layer = _layer("decode")
        state = _empty(decode_layer)
        for t in range(total):
            state, _ = decode_layer.extend_step(
                state,
                k_proj=k_all[:, t : t + 1],
                v_proj=v_all[:, t : t + 1],
                key_positions=pos[:, t : t + 1],
            )

        # Two chunks: A (step_len=T_a) from empty, then B (step_len=T_b) from
        # the non-empty cache A produced.
        chunk_layer = _layer("chunk")
        chunk_state = _empty(chunk_layer)
        chunk_state, _ = chunk_layer.extend_step(
            chunk_state,
            k_proj=k_all[:, :T_a],
            v_proj=v_all[:, :T_a],
            key_positions=pos[:, :T_a],
        )
        chunk_state, out = chunk_layer.extend_step(
            chunk_state,
            k_proj=k_all[:, T_a:],
            v_proj=v_all[:, T_a:],
            key_positions=pos[:, T_a:],
        )

        for field in ("key", "value", "key_positions"):
            np.testing.assert_array_equal(
                np.asarray(chunk_state[field]),
                np.asarray(state[field]),
                err_msg=f"state field={field}",
            )
        # Output is the post-update buffer of the second chunk, transposed BSNH.
        np.testing.assert_array_equal(
            np.einsum("bnht->btnh", np.asarray(state["key"])), np.asarray(out.k_proj)
        )
        np.testing.assert_array_equal(
            np.asarray(state["key_positions"]), np.asarray(out.key_positions)
        )

    # ------------------------------------------------------------------------
    # Ragged multibatch: each row is cached from its OWN positions (per-row scatter)
    # ------------------------------------------------------------------------

    def test_multibatch_ragged_positions_are_per_row(self):
        """Ragged multibatch: each row is cached from its own positions (W=8, coset 0).

        One batched prefill with left-aligned `segment_ids`: row 0 has 2 valid
        tokens (all in the dense window), row 1 has 20 (thinned past w0). Their
        surviving positions differ -- which only holds if each row re-slots by its
        own q_max, not row 0's; a shared schedule would collapse row 1 to [0, 1].
        """
        W, N, H, T = 8, 2, 4, 20
        cap = 2 * W
        layer = (
            TWACache.default_config(w0=W, w1=W << 3, coset_index=0)
            .set(name="ragged")
            .instantiate(parent=None)
        )
        state = layer.init_states(
            shape=TWACache.Shape(batch_size=2, kv_len=cap, num_kv_heads=N, per_head_dim=H),
            dtype=jnp.float32,
        )
        rng = jax.random.PRNGKey(0)
        k = jax.random.normal(jax.random.fold_in(rng, 1), (2, T, N, H))
        v = jax.random.normal(jax.random.fold_in(rng, 2), (2, T, N, H))
        positions = jnp.arange(T, dtype=jnp.int32)[None, :].repeat(2, axis=0)
        # Row 0: first 2 steps valid; row 1: all 20 valid.
        segment_ids = jnp.array([[1, 1] + [0] * (T - 2), [1] * T], dtype=jnp.int32)

        batched, _ = layer.extend_step(
            state, k_proj=k, v_proj=v, key_positions=positions, segment_ids=segment_ids
        )
        sentinel = layer._invalid_position()
        # Row 0 (2 valid): both inside the dense window, so nothing is tapered.
        self.assertEqual(_live_sorted(batched["key_positions"][0], sentinel), [0, 1])
        # Row 1 (20 valid): its OWN q_max is 19, so ages 0..7 are dense (12..19), 8..15 keep the
        # evens (4,6,8,10) and 16.. keep the multiples of 4 (0). A shared q_max would thin row 0 too.
        self.assertEqual(
            _live_sorted(batched["key_positions"][1], sentinel),
            [0, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19],
        )

    # ------------------------------------------------------------------------
    # enable_twa_attention helper
    # ------------------------------------------------------------------------

    def test_enable_twa_attention_wires_both(self):
        """`enable_twa_attention` installs cache + TWA mask."""
        for coset in (0, 1):
            cfg = GroupedQueryAttention.default_config().set(
                num_heads=4,
                query_dim=64,
                key_dim=64,
                value_dim=64,
                hidden_dim=64,
            )
            enable_twa_attention(cfg, w0=1024, coset_index=coset, w1=8192)
            self.assertIs(cfg.kv_cache.klass, TWACache)
            self.assertEqual(cfg.kv_cache.w0, 1024)
            self.assertEqual(cfg.kv_cache.coset_index, coset)
            # Mask is now a closed-form predicate (not plain causal) so that
            # training agrees with what the cache has visible at inference.
            self.assertIs(cfg.mask.klass, MaskFnAttentionBias)

    def test_enable_twa_attention_rejects_bad_coset(self):
        """coset_index must be 0 or 1."""
        cfg = GroupedQueryAttention.default_config().set(
            num_heads=4, query_dim=64, key_dim=64, value_dim=64, hidden_dim=64
        )
        with self.assertRaisesRegex(ValueError, "coset_index"):
            enable_twa_attention(cfg, w0=1024, coset_index=2, w1=8192)

    # ------------------------------------------------------------------------
    # Cache capacity
    # ------------------------------------------------------------------------

    def test_max_capacity_comes_from_the_geometry(self):
        """The buffer must be `twa_geometry`'s slot count and nothing else -- computing it here
        as well is how a mask and a buffer come to disagree about how many keys exist."""
        layer = (
            TWACache.default_config(w0=1024, w1=8192, coset_index=0)
            .set(name="cap")
            .instantiate(parent=None)
        )
        self.assertEqual(layer._max_capacity(), taper_geometry(w0=1024, w1=8192).slots)
        self.assertEqual(layer._max_capacity(), 1024 + 3 * 512)

    # ------------------------------------------------------------------------
    # TWACache.default_config
    # ------------------------------------------------------------------------

    def test_default_config(self):
        # Defaults: w0=2048, coset_index=1, w1=None (no taper past the dense window).
        cfg = TWACache.default_config(w1=131072)
        self.assertIs(cfg.klass, TWACache)
        self.assertEqual(cfg.w0, 2048)
        self.assertEqual(cfg.coset_index, 1)
        self.assertEqual(cfg.w1, 131072, "w1 is required, so there is no default to check")
        # Explicit overrides flow through.
        cfg = TWACache.default_config(w0=1024, coset_index=0, w1=1024 * 8)
        self.assertEqual(cfg.w0, 1024)
        self.assertEqual(cfg.w1, 8192)
        self.assertEqual(cfg.coset_index, 0)


if __name__ == "__main__":
    absltest.main()

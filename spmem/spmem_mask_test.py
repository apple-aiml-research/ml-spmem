#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Tests for spmem_mask.py -- SpMem attention mask: compression memory + mipmap-tapered reals.

The stream is AUGMENTED: `m` memory tokens are inserted after every `C` reals, so the period is
`P = C + m` and a position is either a real or a memory. Three terms decide visibility, and each is
pinned separately below:

  * real -> real is the mipmap TAPER over real indices -- exactly `twa_mask_fn(rule=MIPMAP)`, whose
    own tests pin it to hand-derived goldens. Two cosets pick different survivors.
  * real -> memory keeps every earlier memory, or a ring (`mem_window_size`) or a taper (`mem_w0`).
  * memory -> * reads a plain `w0`-real window, every prior memory, and its own chunk's `m` siblings
    in both directions -- untapered, since thinning it would change what is compressed.
"""

# pylint: disable=missing-function-docstring,protected-access,no-self-use

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from axlearn.common import test_utils
from axlearn.common.attention import MultiheadAttention, TransformerAttentionLayer
from axlearn.common.attention_bias import MaskFnAttentionBias
from axlearn.common.flash_attention.common import build_mask
from axlearn.common.module import functional as F

from spmem.spmem_geometry import SpMemGeometry, spmem_taper_geometry
from spmem.spmem_mask import spmem_mask_fn
from twa.twa_geometry import TaperRule
from twa.twa_mask import twa_mask_fn

# w0=4 under mipmap: 2 levels, 7 slots, reach 8 -- small enough to read, deep enough to thin.
_C, _M, _W = 2, 1, 4
_L = 30


class SpMemMaskTest(test_utils.TestCase):
    """The three visibility terms -- tapered reals, memory reads, and the memory pool bound."""

    @parameterized.parameters(0, 1)
    def test_real_rows_are_exactly_the_twa_mipmap_taper(self, coset):
        """The real->real term is `twa_mask_fn(rule=MIPMAP)` over REAL indices, not augmented ones.

        The taper's slot arithmetic describes the real KV buffer, which the memory tokens do not sit
        in. Getting that axis wrong still yields a plausible-looking mask, just one thinning the
        wrong sequence.
        """
        dense = _dense_mask(_mask(coset=coset), _L)
        aug_of_real = np.flatnonzero(~_memory_columns(_L))
        n = len(aug_of_real)
        want = _dense_mask(twa_mask_fn(_W, coset, rule=TaperRule.MIPMAP), n)
        np.testing.assert_array_equal(dense[np.ix_(aug_of_real, aug_of_real)], want)
        # And it genuinely thins inside this length, or the comparison would be vacuous.
        self.assertLess(int(want[-1].sum()), n)

    @parameterized.parameters(0, 1)
    def test_real_rows_keep_every_earlier_memory(self, coset):
        dense = _dense_mask(_mask(coset=coset), _L)
        is_mem = _memory_columns(_L)
        for q in np.flatnonzero(~is_mem):
            want = {k for k in np.flatnonzero(is_mem) if k < q}
            self.assertEqual(set(np.flatnonzero(dense[q] & is_mem).tolist()), want, f"q={q}")

    def test_cosets_differ_only_on_real_to_real_cells(self):
        """A coset that changed the memory terms would change what is being compressed, so the arms
        would no longer be comparable; a coset that changed nothing would make the split decorative.
        """
        c0 = _dense_mask(_mask(), _L)
        c1 = _dense_mask(_mask(coset=1), _L)
        differing = c0 ^ c1
        self.assertTrue(differing.any(), "the two cosets must thin differently")
        is_real = ~_memory_columns(_L)
        outside = differing & ~(is_real[:, None] & is_real[None, :])
        self.assertFalse(outside.any(), f"{int(outside.sum())} cells differ outside real->real")

    def test_memory_reads_a_window_prior_memory_and_its_siblings(self):
        # Larger geometry (C=8, m=4, w0=16) where the memory window spans 2 chunks. A chunk-2 memory
        # query sees: its own chunk's m siblings BIDIRECTIONALLY, every earlier chunk's memory, and
        # the 16 preceding reals -- but NOT chunk-0 reals, so the window genuinely bounds it.
        c, m, w = 8, 4, 16
        period = c + m
        fn = _mask(c, m, w)
        gq = 2 * period + c  # first memory of chunk 2
        self.assertTrue(bool(fn(np.int32(gq), np.int32(2 * period + c + m - 1))), "own-chunk bidir")
        for k in list(range(c, period)) + list(range(period + c, 2 * period)):
            self.assertTrue(bool(fn(np.int32(gq), np.int32(k))), f"prior memory k={k}")
        for k in list(range(2 * period, 2 * period + c)) + list(range(period, period + c)):
            self.assertTrue(bool(fn(np.int32(gq), np.int32(k))), f"in-window real k={k}")
        for k in range(0, c):
            self.assertFalse(bool(fn(np.int32(gq), np.int32(k))), f"out-of-window real k={k}")

    @parameterized.parameters(
        dict(mem_window_size=None, want_chunks=None),  # unbounded: every earlier memory
        dict(mem_window_size=12, want_chunks=3),
        dict(mem_window_size=8, want_chunks=2),
    )
    def test_mem_window_bounds_the_memory_a_query_sees(self, mem_window_size, want_chunks):
        """`mem_window_size` bounds the memory pool to the chunks the ring holds, and the mask must
        never ask for a slot the ring has reused."""
        c, m, w = 4, 2, 8
        period = c + m
        fn = _mask(c, m, w, mem_window_size=mem_window_size)
        real_q, mem_q = 9 * period + 2, 9 * period + c

        def seen(q):
            return {
                k // period
                for k in range(q + 1)
                if (k % period) >= c and bool(fn(np.int32(q), np.int32(k)))
            }

        if want_chunks is None:
            self.assertEqual(seen(real_q), set(range(9)))
            self.assertEqual(seen(mem_q), set(range(10)))
            return
        # A real query's own chunk has no memory yet; a memory query's own group already holds a
        # slot, so its budget is one chunk narrower.
        self.assertEqual(seen(real_q), set(range(9 - want_chunks, 9)))
        self.assertEqual(seen(mem_q), set(range(9 - want_chunks + 1, 10)))

    def test_broadcasts_and_builds_a_block_mask(self):
        # `MaskFnAttentionBias` calls the predicate with [B,T,1] x [B,1,S], and axlearn's
        # block-sparse path needs it tracer-safe at a production geometry.
        fn = _mask(8, 4, 16, coset=1)
        b, t, s = 2, 16, 24
        q_pos = jnp.arange(t, dtype=jnp.int32)[None, :, None].repeat(b, axis=0)
        k_pos = jnp.arange(s, dtype=jnp.int32)[None, None, :].repeat(b, axis=0)
        out = fn(q_pos, k_pos)
        self.assertEqual(out.shape, (b, t, s))
        self.assertEqual(out.dtype, jnp.bool_)
        bm = build_mask(
            _mask(8, 1, 4096),
            q_seq_len=8192,
            kv_seq_len=8192,
            block_q=512,
            block_k=512,
        )
        self.assertEqual(bm.shape, (16, 16))

    def test_mask_fn_memoized_per_geometry_and_coset(self):
        # Splash keys MaskInfo on `id(mask_fn)`, so equal geometries must share one closure.
        fn = _mask(8, 1, 4096)
        self.assertIs(fn, _mask(8, 1, 4096))
        self.assertIsNot(fn, _mask(8, 1, 4096, coset=1))
        self.assertIsNot(fn, _mask(8, 2, 4096))

    def test_rejects_a_bad_coset(self):
        # The taper reads the coset as a plain int, so a bad one would thin silently.
        with self.assertRaisesRegex(ValueError, "coset"):
            _mask(16, 4, 16, coset=2)

    @parameterized.parameters((4, 2, 8, 0), (8, 4, 16, 1))
    def test_real_attention_layer_forward_backward_finite(self, c, m, w, coset):
        """The guard that no softmax row is fully masked and no gradient is non-finite.

        Two terms here can produce an empty row where a plain causal mask cannot: the taper drops
        arbitrarily old reals, and the own-chunk sibling term is non-causal.
        """
        length = 3 * (c + m)
        model_dim, num_heads, batch = 8, 2, 2
        layer = _build_mha(model_dim, num_heads, _bias(c, m, w, coset=coset))
        params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        query = jax.random.normal(jax.random.PRNGKey(1), (batch, length, model_dim), jnp.float32)
        key = jax.random.PRNGKey(2)

        def loss_of(params, query):
            out, _ = F(
                layer, state=params, is_training=False, prng_key=key, inputs=dict(target=query)
            )
            return jnp.sum(out.data**2), out.data

        (loss, fwd), (grad_params, grad_query) = jax.value_and_grad(
            loss_of, argnums=(0, 1), has_aux=True
        )(params, query)
        self.assertTrue(np.all(np.isfinite(np.asarray(fwd))), "forward output non-finite")
        self.assertTrue(np.isfinite(float(loss)))
        param_grads = jax.tree_util.tree_leaves(grad_params)
        self.assertTrue(param_grads, "no parameter gradients produced")
        for g in param_grads:
            self.assertTrue(np.all(np.isfinite(np.asarray(g))), "non-finite parameter gradient")
        self.assertTrue(np.all(np.isfinite(np.asarray(grad_query))), "non-finite grad wrt query")


class SpMemMemoryTaperTest(test_utils.TestCase):
    """`mem_w0`: the memory pool tapers on its own clock, in memory TOKENS.

    Without it the pool grows linearly in context -- 131,072 memory tokens at 1M reals under C8M1 --
    so decoding is not O(1). These pin the property that makes it O(1), and the two ways the axis
    can be wired wrong: the wrong unit, and the wrong query clock.
    """

    @parameterized.product(coset_index=[0, 1], mem_w0=[2, 4])
    def test_live_memory_is_bounded_by_the_pools_slot_count(self, coset_index, mem_w0):
        """The pool's slot count is its asymptote, and it bounds SLOTS, not reals.

        Checked at every query rather than only at the last: a bound that held at the end and not
        in between would still overflow the pool. Long enough for the pool to FILL, so the bound is
        reached and not merely respected.
        """
        chunks = 40
        length = chunks * (_C + _M)
        fn = _mask(coset=coset_index, mem_w0=mem_w0, mem_w1=_reach(mem_w0))
        vis = _dense_mask(fn, length)
        is_mem = _memory_columns(length)
        live = (vis & is_mem[None, :]).sum(axis=1)
        slots = spmem_taper_geometry(w0=mem_w0).slots
        self.assertEqual(int(live.max()), slots)
        self.assertLess(int(live.max()), int(is_mem.sum()))

    def test_memory_rows_are_the_twa_taper_over_memory_ordinals(self):
        """The pool tapers over memory ORDINALS: a memory query's view of earlier memory is exactly
        `twa_mask_fn` over the memory sub-stream, the same rule at the pool's granularity."""
        chunks = 40
        length = chunks * (_C + _M)
        mem_w0 = 4
        vis = _dense_mask(
            _mask(coset=1, mem_w0=mem_w0, mem_w1=_reach(mem_w0)),
            length,
        )
        mem_cols = np.flatnonzero(_memory_columns(length))
        # At m=1 each memory query's own group is itself, so its row IS the taper at that ordinal.
        want = _dense_mask(twa_mask_fn(mem_w0, 1, rule=TaperRule.MIPMAP), chunks)
        np.testing.assert_array_equal(vis[np.ix_(mem_cols, mem_cols)], want)

    def test_a_real_query_sees_one_fewer_memory_group_than_its_memory_sibling(self):
        """The clock asymmetry. A real query's own chunk is not sealed, so the newest memory it can
        see is the previous group's; the chunk's memory token itself already owns a slot. Getting
        this backwards lets a real attend memory that summarizes itself."""
        chunks = 10
        length = chunks * (_C + _M)
        fn = _mask(mem_w0=4, mem_w1=_reach(4))
        vis = _dense_mask(fn, length)
        is_mem = _memory_columns(length)
        for qc in range(2, chunks):
            last_real = qc * (_C + _M) + _C - 1
            own_mem = qc * (_C + _M) + _C
            real_sees = set(np.flatnonzero(vis[last_real] & is_mem))
            mem_sees = set(np.flatnonzero(vis[own_mem] & is_mem))
            self.assertNotIn(
                own_mem, real_sees, f"chunk {qc}: real attends memory of its own chunk"
            )
            self.assertIn(own_mem, mem_sees, f"chunk {qc}: memory cannot see its own group")

    def test_the_two_cosets_retain_different_memory(self):
        """The memory taper reuses the layer's coset rather than adding an axis, so the pool is
        decorrelated across layers for the same reason the reals are."""
        chunks = 40
        length = chunks * (_C + _M)
        kw = dict(mem_w0=4, mem_w1=_reach(4))
        a = _dense_mask(_mask(coset=0, **kw), length)
        b = _dense_mask(_mask(coset=1, **kw), length)
        is_mem = _memory_columns(length)
        differ = ((a ^ b) & is_mem[None, :]).sum()
        self.assertGreater(int(differ), 0, "both cosets keep the same memory")


class SpMemPackedSegmentsTest(test_utils.TestCase):
    """Packed multi-document masking.

    One sequence packs two augmented docs (A ++ B). With per-segment augmented positions -- reset at
    the boundary, as `insert_memory_tokens` emits -- the mask composes with axlearn's
    `SegmentIdAttentionBias` so doc B behaves exactly as if run standalone from 0. The anchor makes
    this matter: it is kept forever, so a position that failed to reset would let doc B attend
    across the boundary rather than merely near it.
    """

    def test_composed_mask_blocks_cross_document(self):
        fn = _mask()
        la, lb = 6, 9
        pos = np.concatenate([np.arange(la), np.arange(lb)]).astype(np.int32)  # per-segment reset
        seg = np.array([1] * la + [2] * lb, dtype=np.int32)
        composed = np.asarray(fn(pos[:, None], pos[None, :])).astype(bool) & (
            seg[:, None] == seg[None, :]
        )
        self.assertFalse(composed[la:, :la].any(), "B queries attend into A")
        self.assertFalse(composed[:la, la:].any(), "A queries attend into B")
        np.testing.assert_array_equal(
            composed[la:, la:], _dense_mask(fn, lb), "within-B block != standalone-B mask"
        )
        np.testing.assert_array_equal(
            composed[:la, :la], _dense_mask(fn, la), "within-A block != standalone-A mask"
        )

    def test_packed_forward_matches_standalone(self):
        la, lb = 6, 9
        batch, model_dim, num_heads = 2, 8, 2
        layer = _build_mha(model_dim, num_heads, _bias())
        params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        q_a = jax.random.normal(jax.random.PRNGKey(1), (batch, la, model_dim))
        q_b = jax.random.normal(jax.random.PRNGKey(2), (batch, lb, model_dim))

        def seg(n, v):
            return jnp.full((batch, n), v, dtype=jnp.int32)

        out_b = _forward_seg(layer, params, q_b, segment_ids=seg(lb, 1))
        packed_seg = jnp.concatenate([seg(la, 1), seg(lb, 2)], axis=1)
        out_packed = _forward_seg(
            layer, params, jnp.concatenate([q_a, q_b], axis=1), segment_ids=packed_seg
        )
        test_utils.assert_allclose(out_packed[:, la:, :], out_b)
        # Perturbing doc A must not move doc B.
        q_a2 = jax.random.normal(jax.random.PRNGKey(3), (batch, la, model_dim))
        out_packed2 = _forward_seg(
            layer, params, jnp.concatenate([q_a2, q_b], axis=1), segment_ids=packed_seg
        )
        test_utils.assert_allclose(out_packed2[:, la:, :], out_packed[:, la:, :])


def _mask(c=_C, m=_M, w0=_W, *, coset=0, **geometry):
    return spmem_mask_fn(
        SpMemGeometry(mem_chunk_size=c, num_mem=m, w0=w0, **geometry), coset_index=coset
    )


def _bias(c=_C, m=_M, w0=_W, *, coset=0):
    return MaskFnAttentionBias.default_config(mask=_mask(c, m, w0, coset=coset))


def _reach(w0):
    """The mipmap rule's full receptive field at `w0`, `log2(w0) * w0`."""
    return spmem_taper_geometry(w0=w0).w1


def _memory_columns(length):
    """Which augmented columns are memory, derived from the period."""
    return (np.arange(length) % (_C + _M)) >= _C


def _dense_mask(fn, length):
    """Materialize the predicate into an `[L, L]` boolean matrix (q rows, k cols)."""
    q = np.arange(length, dtype=np.int32)[:, None]
    k = np.arange(length, dtype=np.int32)[None, :]
    return np.asarray(fn(q, k)).astype(bool)


def _build_mha(model_dim, num_heads, mask_cfg):
    # Wrapped in TransformerAttentionLayer so the forward takes per-segment `target_positions`.
    cfg = TransformerAttentionLayer.default_config().set(
        name="tal", target_dim=model_dim, source_dim=model_dim
    )
    cfg.attention = MultiheadAttention.default_config().set(num_heads=num_heads)
    cfg.attention.mask = mask_cfg
    return cfg.instantiate(parent=None)


def _forward_seg(layer, params, query, *, segment_ids):
    """Self-attention forward with packed `segment_ids`; positions are derived per segment."""
    out, _ = F(
        layer,
        inputs=dict(target=query, segment_ids=segment_ids, target_positions=None),
        is_training=False,
        state=params,
        prng_key=jax.random.PRNGKey(0),
    )
    return out.data


if __name__ == "__main__":
    absltest.main()

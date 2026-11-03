#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""End-to-end forward-vs-extend_step parity for SpMem attention.

The one test that proves the two halves of SpMem agree: `spmem_mask` decides what a query may
attend to at training time, `SpMemCache` decides what is still in the buffer at decode time, and a
divergence between them is silent -- the model trains on one scope and decodes with another.

Each test wires a real `GroupedQueryAttention` in a `TransformerAttentionLayer` with the SpMem
cache + mask (`enable_spmem_attention`) and compares a forward over the augmented stream to the
incrementally-decoded output. Non-flash reference attention, so no mesh is needed.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from axlearn.common import test_utils
from axlearn.common.attention import GroupedQueryAttention, TransformerAttentionLayer
from axlearn.common.attention_bias import CausalAttentionBias
from axlearn.common.module import functional as F

from spmem.spmem_apply import enable_spmem_attention
from spmem.spmem_augment import segment_relative_positions
from spmem.spmem_geometry import SpMemGeometry, spmem_taper_geometry

_DIM, _HEADS = 16, 4
# w0=4 under mipmap reaches 8 reals; the parity streams carry 16, so every ring and the anchor are
# exercised, not just the dense window.
_W, _C, _M = 4, 2, 2
_PERIOD = _C + _M
# Irregular blocks over 8 chunks (32 augmented). Blocks may cross chunk boundaries, but each chunk's
# m memory tokens (offsets 2, 3 of every period) stay inside one block.
_SCHEDULE = [(0, 6), (6, 2), (8, 1), (9, 1), (10, 2), (12, 4), (16, 1), (17, 3), (20, 5), (25, 7)]


def _layer(*, coset_index, mem_window_size=None, mem_w0=None, num_mem=_M):
    cfg = TransformerAttentionLayer.default_config().set(
        name="spmem", target_dim=_DIM, source_dim=_DIM
    )
    cfg.attention = GroupedQueryAttention.default_config().set(num_heads=_HEADS)
    geometry = SpMemGeometry(
        mem_chunk_size=_C,
        num_mem=num_mem,
        w0=_W,
        mem_window_size=mem_window_size,
        mem_w0=mem_w0,
        mem_w1=None if mem_w0 is None else spmem_taper_geometry(w0=mem_w0).w1,
    )
    enable_spmem_attention(cfg.attention, geometry=geometry, coset_index=coset_index)
    return cfg.instantiate(parent=None)


def _inputs(seq_len, batch=2, seed=123):
    key = jax.random.PRNGKey(seed)
    key, data_key = jax.random.split(key)
    query = jax.random.uniform(data_key, (batch, seq_len, _DIM))
    return key, query, jnp.ones((batch, seq_len), dtype=jnp.int32)


def _forward(layer, params, query, segment_ids, prng_key, positions=None):
    if positions is None:
        positions = segment_relative_positions(segment_ids)
    outputs, _ = F(
        layer,
        inputs=dict(target=query, segment_ids=segment_ids, target_positions=positions),
        is_training=False,
        state=params,
        prng_key=prng_key,
    )
    return outputs.data


def _decode(layer, params, query, segment_ids, schedule, *, cached_states=None):
    """Feed `schedule`'s `(start, length)` blocks through `extend_step`; return (states, output)."""
    if cached_states is None:
        cached_states = layer.init_states(
            batch_size=query.shape[0], max_len=query.shape[1], dtype=query.dtype
        )
    got = []
    for start, length in schedule:
        (cached_states, step_outs), _ = F(
            layer,
            inputs=dict(
                cached_states=cached_states,
                target=query[:, start : start + length],
                segment_ids=segment_ids[:, start : start + length],
            ),
            is_training=False,
            state=params,
            prng_key=jax.random.PRNGKey(0),
            method="extend_step",
        )
        got.append(step_outs.data)
    return cached_states, jnp.concatenate(got, axis=1)


class SpMemForwardVsDecodeTest(test_utils.TestCase):
    """`forward` (mask) must equal `extend_step` (cache) at every valid position."""

    @parameterized.product(
        coset_index=[0, 1],
        mem_bound=[None, ("window", 4), ("taper", 2)],
    )
    def test_forward_matches_decode(self, coset_index, mem_bound):
        """`mem_bound` covers all three memory retentions: unbounded, the `mem_window_size` ring,
        and the memory taper. The last is a distinct code path on BOTH sides -- a taper term in the
        mask and a second `_reslot` in the cache -- so the ring case does not stand in for it."""
        kind, size = mem_bound if mem_bound else (None, None)
        layer = _layer(
            coset_index=coset_index,
            mem_window_size=size if kind == "window" else None,
            mem_w0=size if kind == "taper" else None,
        )
        seq_len = 8 * _PERIOD
        self.assertEqual(sum(length for _, length in _SCHEDULE), seq_len)
        prng_key, query, segment_ids = _inputs(seq_len)
        prng_key, init_key = jax.random.split(prng_key)
        params = layer.initialize_parameters_recursively(init_key)
        want = _forward(layer, params, query, segment_ids, prng_key)
        _, got = _decode(layer, params, query, segment_ids, _SCHEDULE)
        self.assertEqual(got.shape, want.shape)
        test_utils.assert_allclose(got, want)

    @parameterized.parameters(0, 1)
    def test_ragged_full_buffer_prefill_then_decode(self, coset_index):
        """The evaler's path: prefill the FULL buffer with segment_ids zeroed past `prefill_len`,
        then decode the rest. Row 0 ends in padding, row 1 is full, so the per-row re-slot and the
        pad sentinel are both on the line."""
        layer = _layer(coset_index=coset_index)
        seq_len, prefill_len, valid_len = 8 * _PERIOD, 4 * _PERIOD, 7 * _PERIOD
        prng_key, query, _ = _inputs(seq_len)
        prng_key, init_key = jax.random.split(prng_key)
        params = layer.initialize_parameters_recursively(init_key)
        segment_ids = jnp.array(
            [[1] * valid_len + [0] * (seq_len - valid_len), [1] * seq_len], dtype=jnp.int32
        )
        query = jnp.where(segment_ids[:, :, None] != 0, query, 0.0)
        want = _forward(layer, params, query, segment_ids, prng_key)

        cached_states = layer.init_states(batch_size=2, max_len=seq_len, dtype=query.dtype)
        prefill_seg = segment_ids.at[:, prefill_len:].set(0)
        (cached_states, prefill_outs), _ = F(
            layer,
            inputs=dict(
                cached_states=cached_states,
                target=jnp.where(prefill_seg[:, :, None] != 0, query, 0.0),
                segment_ids=prefill_seg,
                is_prefill=True,
            ),
            is_training=False,
            state=params,
            prng_key=prng_key,
            method="extend_step",
        )
        # Each sealed chunk carries its m memory with its last real.
        schedule = [(16, 1), (17, 3), (20, 5), (25, 7)]
        _, rest = _decode(layer, params, query, segment_ids, schedule, cached_states=cached_states)
        got = jnp.concatenate([prefill_outs.data[:, :prefill_len], rest], axis=1)
        valid = np.asarray(segment_ids) != 0  # padding outputs are unconstrained
        test_utils.assert_allclose(got[valid], want[valid])

    def test_the_real_buffer_is_the_geometrys_slot_count(self):
        """The buffer must come from `twa_geometry`, not from arithmetic here -- deriving it in two
        places is how a mask and a buffer come to disagree about how many keys exist."""
        layer = _layer(coset_index=0)
        states = layer.init_states(batch_size=1, max_len=8 * _PERIOD, dtype=jnp.float32)
        want = spmem_taper_geometry(w0=_W).slots
        self.assertEqual(states["attention"]["kv_cache"]["win_positions"].shape[-1], want)


class SpMemMemoryGroupSplitTest(test_utils.TestCase, parameterized.TestCase):
    """`m >= 2` forbids splitting a chunk's memory tokens across blocks; `m == 1` cannot violate it.

    The pool slots memory PER CHUNK, so a block that carries one of a chunk's two memory tokens and
    leaves the other for the next block writes a half-formed group. Nothing raises -- the output is
    simply wrong -- so this is the invariant to state outright rather than discover from a decode
    that silently disagrees with training. `m = 1` is immune by construction, which is why the
    shipped C8M1 arms decode a token at a time safely.
    """

    @parameterized.parameters(1, 2)
    def test_per_token_decode_is_safe_only_at_one_memory_token(self, num_mem):
        layer = _layer(coset_index=0, num_mem=num_mem)
        seq_len = 8 * (_C + num_mem)
        prng_key, query, segment_ids = _inputs(seq_len)
        prng_key, init_key = jax.random.split(prng_key)
        params = layer.initialize_parameters_recursively(init_key)
        want = _forward(layer, params, query, segment_ids, prng_key)
        _, decoded = _decode(layer, params, query, segment_ids, [(i, 1) for i in range(seq_len)])
        if num_mem == 1:
            self.assertNestedAllClose(decoded, want)
        else:
            with self.assertRaises(AssertionError):
                self.assertNestedAllClose(decoded, want)


class SpMemPackedForwardTest(test_utils.TestCase):
    """Packed multi-doc training-forward: per-segment positions + cross-segment isolation."""

    def test_packed_segments_match_standalone(self):
        # Pack two segments (A || B); B must match standalone. Positions are DERIVED from segment_ids
        # (segment_relative_positions, reset per segment) and drive RoPE + the SpMem mask; A is not
        # period-aligned, so global positions would shift B's `pos % period` (negative control (3)).
        layer = _layer(coset_index=0)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        params = layer.initialize_parameters_recursively(init_key)

        batch = 2
        len_a, len_b = _PERIOD + 1, 6 * _PERIOD  # A not period-aligned; B deep enough to thin
        prng_key, key_a, key_b, key_a2 = jax.random.split(prng_key, 4)
        q_a = jax.random.normal(key_a, (batch, len_a, _DIM))
        q_b = jax.random.normal(key_b, (batch, len_b, _DIM))

        def seg(n, v):
            return jnp.full((batch, n), v, jnp.int32)

        out_b = _forward(layer, params, q_b, seg(len_b, 1), prng_key)
        packed_q = jnp.concatenate([q_a, q_b], axis=1)
        packed_seg = jnp.concatenate([seg(len_a, 1), seg(len_b, 2)], axis=1)
        out_seg = _forward(layer, params, packed_q, packed_seg, prng_key)

        # (1) derived per-segment positions: B's slice == standalone B.
        test_utils.assert_allclose(out_seg[:, len_a:, :], out_b)
        # (2) cross-segment isolation: perturbing A must not change B.
        q_a2 = jax.random.normal(key_a2, (batch, len_a, _DIM))
        out_seg2 = _forward(
            layer, params, jnp.concatenate([q_a2, q_b], axis=1), packed_seg, prng_key
        )
        test_utils.assert_allclose(out_seg2[:, len_a:, :], out_seg[:, len_a:, :])
        # (3) negative control: forcing GLOBAL positions -> B diverges from standalone, proving
        #     positions must reset per segment.
        glob = jnp.broadcast_to(jnp.arange(len_a + len_b, dtype=jnp.int32), (batch, len_a + len_b))
        out_glob = _forward(layer, params, packed_q, packed_seg, prng_key, positions=glob)
        self.assertGreater(float(jnp.max(jnp.abs(out_glob[:, len_a:, :] - out_b))), 1e-3)


class KeepGlobalForwardVsDecodeTest(test_utils.TestCase):
    """A HYBRID's keep-global layer: the stock causal mask over a stock cache, forward vs decode.

    The layer keeps the stock `KVCache`, whose `extend_step` has TWO branches -- a one-hot scatter
    below `step_size` 32 (CPU/GPU; 1024 on TPU) and `dynamic_update_slice_in_dim` at or above it,
    which writes the block CONTIGUOUSLY from `key_positions[:, 0]` and returns
    `key_positions = arange(source_len)`, i.e. pad slots unsentineled. A long-prompt prefill takes the
    second branch, so both are driven here, and the ragged case checks the branch where pad reaches
    the buffer. The cache spans the augmented stream because `SpMemDecoderMixin.init_states` sizes it
    that way; here the test passes `max_len=aug_len` itself, standing in for the decoder.
    """

    _C, _M = 4, 2  # chunk, memory per chunk
    _DIM, _HEADS = 16, 4

    def _schedules(self, n_chunks):
        """`(label, [(start, length), ...])` pairs covering both `KVCache.extend_step` branches.

        Each chunk's m memory tokens stay inside ONE span: the keep-global mask makes them mutually
        bidirectional, so a sibling split across spans cannot see its successor and parity is lost --
        a property of the scheme, not of this cache.
        """
        period = self._C + self._M
        aug_len = n_chunks * period
        per_chunk = [
            span
            for chunk in range(n_chunks)
            for span in (
                *((chunk * period + i, 1) for i in range(self._C)),
                (chunk * period + self._C, self._M),
            )
        ]
        return (
            ("one block", [(0, aug_len)]),  # >= the CPU threshold 32 -> dynamic_update_slice branch
            ("real by real, memory batched", per_chunk),  # small steps -> one-hot branch
        )

    def _layer(self):
        cfg = TransformerAttentionLayer.default_config().set(
            name="keep_global", target_dim=self._DIM, source_dim=self._DIM
        )
        cfg.attention = GroupedQueryAttention.default_config().set(num_heads=self._HEADS)
        # A keep-global layer carries the stock causal mask over the stock cache.
        cfg.attention.mask = CausalAttentionBias.default_config()
        return cfg.instantiate(parent=None)

    def test_forward_matches_decode(self):
        n_chunks = 8
        aug_len = n_chunks * (self._C + self._M)  # 48 augmented positions
        layer = self._layer()
        params = layer.initialize_parameters_recursively(jax.random.PRNGKey(0))
        query = jax.random.uniform(jax.random.PRNGKey(1), (2, aug_len, self._DIM))
        seg = jnp.ones((2, aug_len), jnp.int32)
        want = _forward(layer, params, query, seg, jax.random.PRNGKey(2))
        for label, spans in self._schedules(n_chunks):
            with self.subTest(label):
                _, decoded = _decode(layer, params, query, seg, spans)
                self.assertEqual(decoded.shape, want.shape)
                test_utils.assert_allclose(decoded, want)

    def test_ragged_rows_and_padding(self):
        # Row 1 is shorter, so its tail is PAD. In the bulk-block branch stock `KVCache` writes those
        # pad slots and reports them as ordinary positions, so this is the case where an unsentineled
        # pad could leak: it can only be masked by causality.
        period = self._C + self._M
        aug_len, short = 8 * period, 5 * period
        layer = self._layer()
        params = layer.initialize_parameters_recursively(jax.random.PRNGKey(3))
        query = jax.random.uniform(jax.random.PRNGKey(4), (2, aug_len, self._DIM))
        seg = jnp.asarray([[1] * aug_len, [1] * short + [0] * (aug_len - short)], dtype=jnp.int32)
        want = _forward(layer, params, query, seg, jax.random.PRNGKey(5))
        for label, spans in self._schedules(8):
            with self.subTest(label):
                _, decoded = _decode(layer, params, query, seg, spans)
                # Only the VALID region is defined; a pad row's tail output is unconstrained.
                test_utils.assert_allclose(decoded[0], want[0])
                test_utils.assert_allclose(decoded[1, :short], want[1, :short])


if __name__ == "__main__":
    absltest.main()

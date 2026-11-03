#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""End-to-end forward-vs-`extend_step` tests for TWA attention.

`extend_step` (inference) must match `forward` (the closed-form TWA mask). This covers the
gap `twa_cache_test.test_prefill_bit_identical_to_decode` misses: it feeds
`key_positions == arange(T)` with no padding, so the padded-prefill path is never exercised.

Each test wires a real `GroupedQueryAttention` with TWA (`enable_twa_attention`) and
asserts forward == token-by-token / prefilled `extend_step`. All use batch=2 (the re-slotting
scatter is per-row under `jax.vmap`) and sequences long enough for the taper to thin; the ragged case
(row 0 shorter) reproduces a padded-prefill layout, which holds only if each row thins on its own
valid length.

Plain (non-flash) reference attention, so no jax mesh context is needed.
"""

# pylint: disable=missing-function-docstring,protected-access,no-self-use

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from axlearn.common import test_utils
from axlearn.common.attention import GroupedQueryAttention
from axlearn.common.attention_bias import CausalAttentionBias
from axlearn.common.module import functional as F

from twa.twa_cache import enable_twa_attention
from twa.twa_geometry import TaperRule

_DIM = 16
_NUM_HEADS = 4


def _prng():
    # Mint at call time so the key matches the active jax_default_prng_impl (a co-running
    # test_utils.TestCase flips the process default to rbg); a module-scope key would freeze.
    return jax.random.PRNGKey(0)


def _twa_layer(
    w0: int,
    coset_index: int,
    *,
    name: str = "taper",
    w1: int | None = None,
    rule: TaperRule = TaperRule.OCTAVE,
):
    """A `GroupedQueryAttention` layer wired with the TWA cache + mask."""
    cfg = GroupedQueryAttention.default_config().set(
        name=name,
        query_dim=_DIM,
        key_dim=_DIM,
        value_dim=_DIM,
        num_heads=_NUM_HEADS,
        hidden_dim=_DIM,
    )
    enable_twa_attention(cfg, w0=w0, coset_index=coset_index, w1=w1, rule=rule)
    return cfg.instantiate(parent=None)


def _dense_causal_layer(*, name: str = "dense"):
    """A plain causal `GroupedQueryAttention` (no TWA) -- the dense reference path."""
    cfg = GroupedQueryAttention.default_config().set(
        name=name,
        query_dim=_DIM,
        key_dim=_DIM,
        value_dim=_DIM,
        num_heads=_NUM_HEADS,
        hidden_dim=_DIM,
        mask=CausalAttentionBias.default_config(),
    )
    return cfg.instantiate(parent=None)


def _init(w0, coset_index, *, w1=None, rule=TaperRule.OCTAVE):
    """Instantiate a TWA layer and its parameters."""
    layer = _twa_layer(w0, coset_index, w1=w1, rule=rule)
    params = layer.initialize_parameters_recursively(jax.random.fold_in(_prng(), 7))
    return layer, params


def _forward(layer, params, query, *, segment_ids=None, query_positions=None):
    """Full mask-based forward pass; returns `[B, T, D]` output."""
    inputs = dict(query=query)
    if segment_ids is not None:
        inputs["segment_ids"] = segment_ids
    if query_positions is not None:
        inputs["query_positions"] = query_positions
    out, _ = F(layer, inputs=inputs, is_training=False, state=params, prng_key=_prng())
    return out.data


def _extend(layer, params, cached_states, query, *, segment_ids=None, is_prefill=False):
    """One `extend_step` (prefill chunk or single-token decode)."""
    inputs = dict(cached_states=cached_states, query=query)
    if is_prefill:
        inputs["is_prefill"] = True
    if segment_ids is not None:
        inputs["segment_ids"] = segment_ids
    (cached_states, out), _ = F(
        layer,
        inputs=inputs,
        is_training=False,
        state=params,
        prng_key=_prng(),
        method="extend_step",
    )
    return cached_states, out.data


class TWAForwardVsDecodeTest(parameterized.TestCase):
    """`forward` (mask) must equal `extend_step` (cache) at every valid position."""

    # W small + seq_len = 4W so the taper thins; batch=2 exercises the per-row vmap scatter.
    @parameterized.product(
        # (w0, coset, K). seq_len is 4*w0 throughout, so K=2 reaches the last level and K=3 leaves
        # the outermost one empty -- both are states a real run passes through. K=1 is the smallest
        # taper there is, and the nearest thing to a dense control the installer will build: K=0 is
        # rejected there on purpose, since a layer given a taper and no receptive field is a silent
        # sliding window.
        case=[(4, 0, 2), (4, 1, 2), (8, 0, 2), (4, 0, 3), (4, 1, 1)],
    )
    def test_forward_matches_full_autoregressive_decode(self, case):
        w0, coset_index, k_levels = case
        batch, seq_len = 2, 4 * w0
        layer, params = _init(w0, coset_index, w1=w0 << k_levels)
        query = jax.random.uniform(jax.random.fold_in(_prng(), 1), (batch, seq_len, _DIM))

        fwd = _forward(layer, params, query)

        cached = layer.init_states(batch_size=batch, max_len=seq_len, dtype=query.dtype)
        steps = []
        for i in range(seq_len):
            cached, step = _extend(layer, params, cached, query[:, i : i + 1])
            steps.append(step)
        decoded = jnp.concatenate(steps, axis=1)

        self.assertEqual(decoded.shape, fwd.shape)
        test_utils.assert_allclose(decoded, fwd)

    @parameterized.product(
        # (W, coset, prefill_len, row0_valid). prefill_len > 2W so thinning happens
        # *inside* the ragged prefill (the padded-prefill path); row0_valid < 2W so
        # row 0 thins less than row 1 -- the per-row q_max divergence. Decode then continues.
        case=[(4, 0, 12, 6), (4, 1, 10, 6), (8, 0, 20, 12)],
        k_levels=[2, 3],
    )
    def test_prefill_then_decode(self, case, k_levels):
        """Ragged batch=2 prefill + autoregressive decode vs forward.

        Row 0 is a short prompt (valid_len < 2W, fully inside the prefill); row 1
        is full length and thins during the prefill. With prefill_len > 2W the
        thinning happens inside the ragged prefill -- the padded layout an
        evaluator feeds -- so this matches forward only if each row thins on its
        OWN q_max, not the padded chunk length. Decode then extends
        row 1 to the end, checking the decode path shares the same buffer.
        Compares valid positions per row.
        """
        window_size, coset_index, prefill_len, row0_valid = case
        batch, seq_len = 2, 4 * window_size
        layer, params = _init(window_size, coset_index, w1=window_size << k_levels)
        query = jax.random.uniform(jax.random.fold_in(_prng(), 2), (batch, seq_len, _DIM))
        # Row 0: `row0_valid` real tokens then right padding; row 1: full length.
        segment_ids = jnp.array(
            [[1] * row0_valid + [0] * (seq_len - row0_valid), [1] * seq_len],
            dtype=jnp.int32,
        )

        fwd = _forward(layer, params, query, segment_ids=segment_ids)

        cached = layer.init_states(batch_size=batch, max_len=seq_len, dtype=query.dtype)
        cached, chunk = _extend(
            layer,
            params,
            cached,
            query[:, :prefill_len],
            segment_ids=segment_ids[:, :prefill_len],
            is_prefill=True,
        )
        steps = [chunk]
        for i in range(prefill_len, seq_len):
            cached, step = _extend(
                layer,
                params,
                cached,
                query[:, i : i + 1],
                segment_ids=segment_ids[:, i : i + 1],
            )
            steps.append(step)
        decoded = jnp.concatenate(steps, axis=1)

        # forward and extend_step need only agree where the token is real.
        valid = np.asarray(segment_ids) != 0  # [B, T]
        self.assertEqual(decoded.shape, fwd.shape)
        test_utils.assert_allclose(decoded[valid], fwd[valid])


class MipmapForwardVsDecodeTest(parameterized.TestCase):
    """The same parity for the MIPMAP rule, whose slot map is a different layout entirely.

    `w0` is a power of two and `w1` is derived, so the cases vary `w0` instead of `K`. `seq_len` is
    `4*w0`, which is past the `log2(w0)*w0` reach at every size here -- the regime where only the
    anchor survives, and the one a 128K run spends most of its time in.
    """

    @parameterized.product(case=[(4, 0), (4, 1), (8, 0), (8, 1), (16, 0)])
    def test_forward_matches_full_autoregressive_decode(self, case):
        w0, coset_index = case
        batch, seq_len = 2, 4 * w0
        layer, params = _init(w0, coset_index, rule=TaperRule.MIPMAP)
        query = jax.random.uniform(jax.random.fold_in(_prng(), 1), (batch, seq_len, _DIM))

        fwd = _forward(layer, params, query)

        cached = layer.init_states(batch_size=batch, max_len=seq_len, dtype=query.dtype)
        steps = []
        for i in range(seq_len):
            cached, step = _extend(layer, params, cached, query[:, i : i + 1])
            steps.append(step)
        test_utils.assert_allclose(jnp.concatenate(steps, axis=1), fwd)

    @parameterized.product(case=[(4, 0, 12, 6), (8, 1, 20, 12)])
    def test_prefill_then_decode(self, case):
        """Ragged prefill under MIPMAP: each row must slot on its OWN `q_max`. Row 0's short prompt
        keeps it inside level 0 while row 1 has aged into the outer rings, so a shared `q_max` would
        place them in different rings and only this asymmetry catches it."""
        w0, coset_index, prefill_len, row0_valid = case
        batch, seq_len = 2, 4 * w0
        layer, params = _init(w0, coset_index, rule=TaperRule.MIPMAP)
        query = jax.random.uniform(jax.random.fold_in(_prng(), 2), (batch, seq_len, _DIM))
        segment_ids = jnp.array(
            [[1] * row0_valid + [0] * (seq_len - row0_valid), [1] * seq_len], dtype=jnp.int32
        )

        fwd = _forward(layer, params, query, segment_ids=segment_ids)

        cached = layer.init_states(batch_size=batch, max_len=seq_len, dtype=query.dtype)
        cached, chunk = _extend(
            layer,
            params,
            cached,
            query[:, :prefill_len],
            segment_ids=segment_ids[:, :prefill_len],
            is_prefill=True,
        )
        steps = [chunk]
        for i in range(prefill_len, seq_len):
            cached, step = _extend(
                layer, params, cached, query[:, i : i + 1], segment_ids=segment_ids[:, i : i + 1]
            )
            steps.append(step)
        decoded = jnp.concatenate(steps, axis=1)
        valid = np.asarray(segment_ids) != 0
        test_utils.assert_allclose(decoded[valid], fwd[valid])


class TWAPackedSegmentsTest(parameterized.TestCase):
    """Packed multi-segment masking, for the dense and TWA paths.

    Complements `test_prefill_then_decode` (ragged batch = different rows): here a single
    sequence packs two segments (A ++ B) and, with per-segment positions + segment ids,
    segment B must behave exactly as if decoded standalone-from-0 and never attend across
    the boundary. This is the direct guard for the per-segment `positions` change.
    """

    @parameterized.named_parameters(
        ("taper", lambda: _twa_layer(w0=4, coset_index=0, w1=16), True),
        ("dense", _dense_causal_layer, False),
    )
    def test_packed_segments_match_standalone(self, build, is_taper):
        # W=4 (2W=8), lb=10 > 2W so taper thinning fires inside B, and B's global-position
        # offset (starting at la=6) crosses the 2W boundary.
        batch, la, lb, dim = 2, 6, 10, _DIM
        layer = build()
        params = layer.initialize_parameters_recursively(jax.random.fold_in(_prng(), 7))
        q_a = jax.random.uniform(jax.random.fold_in(_prng(), 1), (batch, la, dim))
        q_b = jax.random.uniform(jax.random.fold_in(_prng(), 2), (batch, lb, dim))

        def _pos(n):  # per-segment positions 0..n-1, broadcast over batch
            return jnp.broadcast_to(jnp.arange(n, dtype=jnp.int32), (batch, n))

        def _seg(n, v):
            return jnp.full((batch, n), v, dtype=jnp.int32)

        # Segment B decoded standalone from position 0.
        out_b = _forward(layer, params, q_b, segment_ids=_seg(lb, 1), query_positions=_pos(lb))

        # Packed [A | B]: per-segment positions (reset at the boundary) + per-segment ids.
        packed_q = jnp.concatenate([q_a, q_b], axis=1)
        packed_seg = jnp.concatenate([_seg(la, 1), _seg(lb, 2)], axis=1)
        per_seg_pos = jnp.concatenate([_pos(la), _pos(lb)], axis=1)
        out_seg = _forward(
            layer, params, packed_q, segment_ids=packed_seg, query_positions=per_seg_pos
        )

        # (1) B's slice behaves exactly as standalone B (segment-local mask + no cross-seg leak).
        test_utils.assert_allclose(out_seg[:, la:, :], out_b)

        # (2) cross-segment isolation: perturbing segment A must not change segment B's output.
        q_a2 = jax.random.uniform(jax.random.fold_in(_prng(), 3), (batch, la, dim))
        out_seg2 = _forward(
            layer,
            params,
            jnp.concatenate([q_a2, q_b], axis=1),
            segment_ids=packed_seg,
            query_positions=per_seg_pos,
        )
        test_utils.assert_allclose(out_seg2[:, la:, :], out_seg[:, la:, :])

        # (3) taper only: the per-segment reset is load-bearing -- with GLOBAL positions the
        # taper thinning fires at a different offset, so B diverges from standalone. (Dense is
        # unaffected: RoPE is relative, so global vs per-segment is a no-op there.)
        if is_taper:
            out_glob = _forward(
                layer, params, packed_q, segment_ids=packed_seg, query_positions=_pos(la + lb)
            )
            self.assertGreater(float(jnp.max(jnp.abs(out_glob[:, la:, :] - out_b))), 1e-3)


if __name__ == "__main__":
    absltest.main()

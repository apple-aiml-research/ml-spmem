#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Decoder-level parity for `SpMemDecoderDense`: `forward` == `prefill_states` + `extend_step`.

Exercises the generic concrete decoder `SpMemDecoderDense` (`SpMemDecoderMixin` over a plain axlearn
`Decoder`).
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from axlearn.common import test_utils
from axlearn.common.attention import StackedTransformerLayer
from axlearn.common.attention_bias import SlidingWindowAttentionBias
from axlearn.common.causal_lm import gpt_decoder_config
from axlearn.common.module import functional

from spmem.spmem_apply import enable_spmem_attention
from spmem.spmem_decoder import SpMemDecoderDense
from spmem.spmem_embedding import install_spmem_token_embedding
from spmem.spmem_geometry import SpMemGeometry, augmented_seq_len

_W, _C, _M = 8, 4, 2  # real taper's w0 (reals), chunk size, memory tokens per chunk
_MEM_ID_BASE = 6  # memory ids 6..6+_M; below the real-token range (>=10), so no collision
_HW_BLOCK = 512  # flash q-block alignment for L_aug (mesh seq=1 in tests -> align == this)
_GEOMETRY = SpMemGeometry(mem_chunk_size=_C, num_mem=_M, w0=_W)


def _build_decoder(*, vocab=32, max_pos=256):
    """A tiny 2-layer GPT `SpMemDecoderDense` with SpMem installed on every (global) attention
    layer -- the generic concrete decoder."""
    cfg = gpt_decoder_config(
        stack_cfg=StackedTransformerLayer.default_config(),
        num_layers=2,
        hidden_dim=8,
        num_heads=2,
        vocab_size=vocab,
        max_position_embeddings=max_pos,
        activation_function="nn.relu",
    )
    layers = cfg.transformer.layer
    for i, lc in enumerate(layers if isinstance(layers, (list, tuple)) else [layers]):
        attn = lc.self_attention.attention
        is_global = attn.mask is None or attn.mask.klass is not SlidingWindowAttentionBias
        if is_global:
            enable_spmem_attention(attn, geometry=_GEOMETRY, coset_index=i % 2)
    cfg = SpMemDecoderDense.from_decoder_config(
        cfg,
        mem_chunk_size=_C,
        num_mem=_M,
        memory_id_base=_MEM_ID_BASE,
        hw_block_size=_HW_BLOCK,
    )
    # Memory tokens embed from the dedicated `mem_weight`, as `apply_spmem_to_decoder` wires them:
    # forward / prefill / decode must all substitute it identically for parity to hold.
    install_spmem_token_embedding(cfg, memory_id_base=_MEM_ID_BASE, num_mem=_M)
    cfg.set(name="spmem_dec", pad_token_id=0, eos_token_id=1)
    return cfg.instantiate(parent=None), vocab


class SpMemDecoderRoutineTest(test_utils.TestCase):
    """`SpMemDecoderMixin` forward/extend_step routine on a small plain `Decoder`."""

    def test_forward_matches_prefill_then_decode(self):
        """`forward` over pure reals == `prefill_states` + `extend_step` decode, as per-real logits.

        Prefill feeds the full buffer with time_step = prompt length (as the evaler does); decode
        uses irregular blocks -- (7, 3) crosses the chunk-1 seal so a multi-real block carries mem.
        10 reals > W=8, so the tail must read chunk 0 via memory.
        """
        dec, vocab = _build_decoder()
        params = dec.initialize_parameters_recursively(jax.random.PRNGKey(0))

        total, prefill_len, buf_len = 10, 6, 32
        ids = jax.random.randint(jax.random.PRNGKey(3), (1, total), 10, vocab - 1).astype(jnp.int32)

        # forward over pure reals -> per-real logits [1, total, V].
        fwd, _ = functional(
            dec,
            inputs=dict(input_batch={"input_ids": ids}),
            state=params,
            is_training=False,
            prng_key=jax.random.PRNGKey(1),
            method="forward",
        )
        fwd_logits, _ = functional(
            dec,
            inputs=dict(forward_outputs=dict(hidden_states=fwd["hidden_states"])),
            state=params,
            is_training=False,
            prng_key=jax.random.PRNGKey(1),
            method="compute_logits",
        )

        # Prefill the full buffer (reals[:prefill_len], rest pad); time_step = prompt length.
        buf = jnp.zeros((1, buf_len), jnp.int32).at[:, :prefill_len].set(ids[:, :prefill_len])
        (states, prefill_out), _ = functional(
            dec,
            inputs=dict(time_step=jnp.array([prefill_len]), input_batch={"input_ids": buf}),
            state=params,
            is_training=False,
            prng_key=jax.random.PRNGKey(1),
            method="prefill_states",
        )
        # Decode reals [prefill_len:total] in irregular blocks (r7 seals chunk 1 inside (7, 3)).
        got = [prefill_out["logits"]]  # last prefilled real
        schedule = [(6, 1), (7, 3)]
        for start, length in schedule:
            (states, step_out), _ = functional(
                dec,
                inputs=dict(
                    cached_states=states,
                    input_batch={"input_ids": ids[:, start : start + length]},
                    is_prefill=False,
                ),
                state=params,
                is_training=False,
                prng_key=jax.random.PRNGKey(0),
                method="extend_step",
            )
            got.append(step_out["logits"])
        got = jnp.concatenate(got, axis=1)

        # forward's logits at real positions [prefill_len-1 : total] must match.
        expected = fwd_logits[:, prefill_len - 1 :, :]
        self.assertEqual(got.shape, expected.shape)
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-5, rtol=1e-5)


class SpMemDecoderInitStatesTest(test_utils.TestCase):
    """`init_states` sizes every cache for the AUGMENTED stream, not the caller's real length.

    `extend_step` writes keys at augmented positions, so a cache sized for L_real overflows once the
    stream passes it (`updates bound 133120 > operand 131072` at a 128K prompt). A
    keep-global layer is the exposed case: it keeps the stock full-attention cache, which allocates
    exactly `kv_len`.
    """

    def test_caches_span_l_aug(self):
        dec, _ = _build_decoder()
        l_real = 64
        l_aug = augmented_seq_len(l_real, mem_chunk_size=_C, num_mem=_M, align=_HW_BLOCK)
        self.assertGreater(l_aug, l_real)
        states = dec.init_states(batch_size=2, max_sequence_length=l_real, dtype=jnp.float32)
        # `time_step` counts augmented positions, and the SpMem pool holds one memory group per
        # sealed chunk of the augmented stream -- both derived from the same L_aug.
        mem_positions = [
            leaf.shape[-1]
            for path, leaf in jax.tree_util.tree_leaves_with_path(states["transformer_state"])
            if jax.tree_util.keystr(path, simple=True, separator="/").endswith("mem_positions")
        ]
        self.assertNotEmpty(mem_positions)
        want = _GEOMETRY.memory_slots(l_aug)
        for size in mem_positions:
            self.assertEqual(size, want)

    def test_align_requires_a_live_mesh_when_seq_is_sharded(self):
        """The cache (`init_states`) and the stream written into it (`extend_step`) must round L_aug
        the same way; a mesh-less guess of 1 sized them 8704 vs 12288 and failed decode with
        "updates bound is 12288, operand bound is 8704". Unsharded seq is
        `hw_block_size` with or without a mesh, which is why every other test here needs no mesh.
        """
        dec, _ = _build_decoder()
        # pylint: disable-next=protected-access
        self.assertEqual(dec._flash_align(), _HW_BLOCK)  # unsharded seq: no mesh needed

        seq_axis_size = jax.device_count()
        cfg = dec.config.clone()
        cfg.set(logits_partition_spec=(cfg.logits_partition_spec[0], "seq", None))
        sharded = cfg.instantiate(parent=None)
        with self.assertRaisesRegex(RuntimeError, "No live mesh"):
            sharded._flash_align()  # pylint: disable=protected-access
        mesh = jax.sharding.Mesh(
            np.asarray(jax.devices()).reshape(1, seq_axis_size), ("data", "seq")
        )
        with mesh:
            # pylint: disable-next=protected-access
            self.assertEqual(sharded._flash_align(), _HW_BLOCK * seq_axis_size)


if __name__ == "__main__":
    absltest.main()

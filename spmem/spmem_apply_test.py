#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Tests for spmem_apply.py -- installing SpMem on a built decoder config.

Driven on a tiny GPT decoder whose globals are flash-attention layers (the per-shard augmented
length aligns to their block size) with a sliding layer between them.
"""

# pylint: disable=missing-function-docstring

from absl.testing import absltest
from axlearn.common import test_utils
from axlearn.common.attention import (
    QLinear,
    StackedTransformerLayer,
    TransformerLayer,
)
from axlearn.common.attention_bias import MaskFnAttentionBias, SlidingWindowAttentionBias
from axlearn.common.causal_lm import gpt_decoder_config
from axlearn.common.decoder import Decoder
from axlearn.common.flash_attention.layer import FlashAttention
from axlearn.common.kv_cache.kv_cache import KVCache

from spmem.spmem_apply import apply_spmem_to_decoder
from spmem.spmem_cache import SpMemCache
from spmem.spmem_decoder import SpMemDecoderDense
from spmem.spmem_embedding import SpMemTokenEmbedding
from spmem.spmem_geometry import SpMemGeometry

_W0, _C, _M, _MEM_ID_BASE, _BLOCK = 8, 4, 1, 30, 512
_GEOMETRY = SpMemGeometry(mem_chunk_size=_C, num_mem=_M, w0=_W0)


class ApplySpMemTest(test_utils.TestCase):
    def test_installs_on_globals_and_retypes_the_decoder(self):
        decoder = _apply(_decoder_cfg(), coset_assignments=(1, 0, 1))
        self.assertIs(decoder.klass, SpMemDecoderDense)
        self.assertEqual((decoder.mem_chunk_size, decoder.num_mem), (_C, _M))
        self.assertEqual(decoder.hw_block_size, _BLOCK)
        self.assertIs(decoder.emb.token_emb.klass, SpMemTokenEmbedding)
        sliding, *globals_ = _attention_cfgs(decoder)
        self.assertIs(sliding.mask.klass, SlidingWindowAttentionBias)
        for attn, coset in zip(globals_, (1, 0, 1)):
            self.assertIs(attn.kv_cache.klass, SpMemCache)
            self.assertIs(attn.mask.klass, MaskFnAttentionBias)
            self.assertEqual(
                (attn.kv_cache.geometry, attn.kv_cache.coset_index), (_GEOMETRY, coset)
            )

    def test_a_kept_global_keeps_its_stock_mask_and_cache(self):
        """A hybrid's kept global is untouched; the cosets then cover only the compressed ones."""
        before = _attention_cfgs(_decoder_cfg())[2].mask
        decoder = _apply(_decoder_cfg(), coset_assignments=(0, 1), keep_global_ordinals=(1,))
        _, first, kept, last = _attention_cfgs(decoder)
        self.assertEqual(kept.mask, before)
        self.assertIs(kept.kv_cache.klass, KVCache)
        self.assertEqual((first.kv_cache.coset_index, last.kv_cache.coset_index), (0, 1))

    def test_rejects_an_illegal_hybrid(self):
        for ordinals, message in (
            ((0, 1, 2), "leaving no SpMem layer"),
            ((3,), r"outside \[0, 3\)"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                _apply(_decoder_cfg(), coset_assignments=(0,), keep_global_ordinals=ordinals)
        # A kept receiver would read a compressed donor's cache with a full-attention mask.
        with self.assertRaisesRegex(ValueError, "share KV"):
            _apply(
                _decoder_cfg(kv_shared_last=True), coset_assignments=(0,), keep_global_ordinals=(0,)
            )

    def test_rejects_a_decoder_class_mismatch(self):
        class _OtherDecoder(Decoder):
            pass

        decoder = _decoder_cfg()
        decoder.klass = _OtherDecoder
        with self.assertRaisesRegex(NotImplementedError, "not built on decoder klass"):
            _apply(decoder, coset_assignments=(0, 1, 0))


def _apply(decoder_cfg, **kwargs):
    return apply_spmem_to_decoder(
        decoder_cfg, geometry=_GEOMETRY, memory_id_base=_MEM_ID_BASE, **kwargs
    )


def _decoder_cfg(*, kv_shared_last: bool = False) -> Decoder.Config:
    """`Stacked([sliding, global, global, global])`, the globals flash attention."""
    cfg = gpt_decoder_config(
        stack_cfg=StackedTransformerLayer.default_config(),
        num_layers=1,
        hidden_dim=8,
        num_heads=2,
        vocab_size=32,
        max_position_embeddings=64,
        activation_function="nn.relu",
    )
    base = cfg.transformer.layer
    base = base[0] if isinstance(base, (list, tuple)) else base

    def layer(*, sliding: bool = False, kv_shared: bool = False) -> TransformerLayer.Config:
        out = base.clone()
        attn = FlashAttention.default_config().set(num_heads=2, tpu_block_size=_BLOCK, causal=True)
        if sliding:
            attn.set(
                causal=None, mask=SlidingWindowAttentionBias.default_config(sliding_window_size=4)
            )
        if kv_shared:
            attn.input_linear = QLinear.default_config()
        out.self_attention.attention = attn
        return out

    templates = [layer(sliding=True), layer(), layer(), layer(kv_shared=kv_shared_last)]
    cfg.transformer = StackedTransformerLayer.default_config().set(layer=templates)
    return cfg


def _attention_cfgs(decoder_cfg):
    return [t.self_attention.attention for t in decoder_cfg.transformer.layer]


if __name__ == "__main__":
    absltest.main()

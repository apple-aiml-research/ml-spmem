#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Tests for twa_apply.py -- installing TWA on a built decoder config.

Driven on a tiny GPT decoder whose repeated stack holds all three layer kinds the walk must classify:
a sliding layer (untouched), a global producer (own coset, live cache) and a global KV-shared
receiver (its donor's coset, so its mask thins what the donor's cache keeps).
"""

# pylint: disable=missing-function-docstring

from absl.testing import absltest
from axlearn.common import test_utils
from axlearn.common.attention import (
    QLinear,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    TransformerLayer,
)
from axlearn.common.attention_bias import MaskFnAttentionBias, SlidingWindowAttentionBias
from axlearn.common.causal_lm import gpt_decoder_config
from axlearn.common.decoder import Decoder

from twa.twa_apply import (
    apply_twa_to_decoder,
    global_attention_configs,
    is_kv_shared_attention,
)
from twa.twa_cache import TWACache
from twa.twa_mask import twa_mask_fn


class ApplyTWATest(test_utils.TestCase):
    def test_globals_get_the_mask_and_cache_and_sliding_layers_are_untouched(self):
        decoder = _decoder_cfg(num_globals=2)
        apply_twa_to_decoder(decoder, coset_assignments=(1, 0), w0=8, w1=64)
        sliding, *globals_ = _templates(decoder)
        self.assertIs(sliding.mask.klass, SlidingWindowAttentionBias)
        # Producers take the assignments in order; the receiver inherits the last producer's.
        for attn, coset in zip(globals_, (1, 0, 0)):
            self.assertIs(attn.kv_cache.klass, TWACache)
            self.assertIs(attn.mask.klass, MaskFnAttentionBias)
            self.assertEqual(attn.kv_cache.coset_index, coset)
            self.assertIs(attn.mask.mask, twa_mask_fn(8, coset, w1=64))
            # The builder's `causal=True` would otherwise conflict with the installed mask.
            self.assertIsNone(attn.causal)

    def test_the_walk_is_by_template_not_by_repeat(self):
        """A repeated stack is ONE template per layer, so cosets count templates."""
        decoder = _decoder_cfg(num_globals=1)
        attns = global_attention_configs(decoder)
        self.assertLen(attns, 2)
        self.assertEqual([is_kv_shared_attention(a) for a in attns], [False, True])

    def test_rejects_a_wrong_number_of_cosets(self):
        for cosets in ((0,), (0, 1, 0)):
            with self.assertRaisesRegex(ValueError, "coset_assignments has"):
                apply_twa_to_decoder(_decoder_cfg(num_globals=2), coset_assignments=cosets, w1=64)

    def test_rejects_a_coset_outside_the_pair(self):
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            apply_twa_to_decoder(_decoder_cfg(num_globals=2), coset_assignments=(0, 2), w1=64)

    def test_rejects_a_receiver_before_any_producer(self):
        decoder = _decoder_cfg(num_globals=0)
        with self.assertRaisesRegex(ValueError, "precedes every producer"):
            apply_twa_to_decoder(decoder, coset_assignments=(), w1=64)


def _decoder_cfg(*, num_globals: int) -> Decoder.Config:
    """`Repeated(2) x Stacked([sliding, global producer x num_globals, global KV-shared])`."""
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
        attn = out.self_attention.attention
        attn.set(causal=True)
        if sliding:
            attn.set(
                causal=None, mask=SlidingWindowAttentionBias.default_config(sliding_window_size=4)
            )
        if kv_shared:
            attn.input_linear = QLinear.default_config()
        return out

    templates = [layer(sliding=True)] + [layer() for _ in range(num_globals)]
    templates.append(layer(kv_shared=True))
    stack = StackedTransformerLayer.default_config().set(layer=templates)
    cfg.transformer = RepeatedTransformerLayer.default_config().set(num_layers=2, layer=stack)
    return cfg


def _templates(decoder_cfg: Decoder.Config):
    return [t.self_attention.attention for t in decoder_cfg.transformer.layer.layer]


if __name__ == "__main__":
    absltest.main()

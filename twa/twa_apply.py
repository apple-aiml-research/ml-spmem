#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Installs TWA on the global attention layers of a built decoder config.

A post-processor rather than a builder argument: it rewrites a `Decoder.Config` that a model builder
already produced. One in-order walk finds every global (non-sliding) attention layer and gives it a
`TWACache` plus the matching mask, with a coset per layer:

  - A producer (a global that owns its K/V) takes the next entry of `coset_assignments`.
  - A KV-shared receiver inherits the nearest preceding producer's coset. Its cache is inert
    (`MultiheadAttention.init_states` skips it under KV sharing), but its mask must thin the same
    keys its donor's cache keeps.

Sliding-window layers are left untouched. A decoder built from a repeated layer stack is walked at
the TEMPLATE level, so one coset covers every repeat of a template.

    apply_twa_to_decoder(model_cfg.decoder, coset_assignments=(0, 1) * 4, w0=2048, w1=131072)
"""

from collections.abc import Sequence
from typing import Optional

from axlearn.common.attention import MultiheadAttention
from axlearn.common.attention_bias import SlidingWindowAttentionBias
from axlearn.common.decoder import Decoder

from twa.twa_cache import enable_twa_attention
from twa.twa_geometry import TaperRule


def is_global_attention(attn_cfg: MultiheadAttention.Config) -> bool:
    """Global iff the layer's mask is not the sliding-window bias."""
    mask = attn_cfg.mask
    return mask is None or mask.klass is not SlidingWindowAttentionBias


def is_kv_shared_attention(attn_cfg: MultiheadAttention.Config) -> bool:
    """KV-shared iff its input linear draws K/V from another layer's `KVState`."""
    return attn_cfg.input_linear.klass.is_kv_sharing(attn_cfg.input_linear)


def global_attention_configs(decoder_cfg: Decoder.Config) -> list[MultiheadAttention.Config]:
    """The global attention configs of `decoder_cfg.transformer`, in layer order (templates, not
    repeats). The embedding is not walked, so a multimodal encoder's attention is left alone."""
    found: list[MultiheadAttention.Config] = []

    def visit_fn(_, value):
        # `visit_fn` also fires on scalar leaves, so type-guard first.
        if isinstance(value, MultiheadAttention.Config) and is_global_attention(value):
            found.append(value)

    def enter_fn(_, value, default_kv):
        # Stop at an attention config, so the kv_cache/mask it carries are not walked.
        return None if isinstance(value, MultiheadAttention.Config) else default_kv

    decoder_cfg.transformer.visit(visit_fn=visit_fn, enter_fn=enter_fn)
    return found


def assign_cosets(
    attn_cfgs: Sequence[MultiheadAttention.Config], coset_assignments: Sequence[int]
) -> list[int]:
    """One coset per layer: producers consume `coset_assignments`, receivers inherit their donor's.

    Raises:
        ValueError: a coset outside {0, 1}; fewer or more assignments than producers; or a receiver
            with no preceding producer.
    """
    if any(c not in (0, 1) for c in coset_assignments):
        raise ValueError(f"coset_assignments values must be 0 or 1, got {tuple(coset_assignments)}")
    coset_iter = iter(coset_assignments)
    cosets: list[int] = []
    num_producers = 0
    for attn_cfg in attn_cfgs:
        if is_kv_shared_attention(attn_cfg):
            if not cosets:
                raise ValueError("a KV-shared global layer precedes every producer donor.")
            cosets.append(cosets[-1])
            continue
        coset = next(coset_iter, None)
        if coset is None:
            raise ValueError(
                f"coset_assignments has {len(coset_assignments)} entries, fewer than the number of "
                "non-KV-shared global layers."
            )
        cosets.append(coset)
        num_producers += 1
    if num_producers != len(coset_assignments):
        raise ValueError(
            f"coset_assignments has {len(coset_assignments)} entries but the decoder has "
            f"{num_producers} non-KV-shared global layers."
        )
    return cosets


def apply_twa_to_decoder(
    decoder_cfg: Decoder.Config,
    *,
    coset_assignments: Sequence[int],
    w0: int = 2048,
    w1: Optional[int] = None,
    rule: TaperRule = TaperRule.OCTAVE,
) -> None:
    """Install TWA on every global attention layer of `decoder_cfg`, in place.

    Args:
        decoder_cfg: A built `Decoder.Config`.
        coset_assignments: Coset (0 or 1) for each non-KV-shared global layer, in layer order.
            Alternating cosets decorrelate which keys adjacent layers keep.
        w0: dense window; the cache size is derived from `(w0, w1)` by `twa_geometry`.
        w1: receptive field; see `twa_geometry.taper_geometry`.
        rule: the retention rule; see `TaperRule`.

    Raises:
        ValueError: see `assign_cosets`, and `twa_geometry` for the geometry.
    """
    attn_cfgs = global_attention_configs(decoder_cfg)
    cosets = assign_cosets(attn_cfgs, coset_assignments)
    for attn_cfg, coset in zip(attn_cfgs, cosets):
        # The TWA mask supersedes a builder's `causal=True`; the two cannot both be set.
        attn_cfg.set(causal=None)
        enable_twa_attention(attn_cfg, w0=w0, coset_index=coset, w1=w1, rule=rule)

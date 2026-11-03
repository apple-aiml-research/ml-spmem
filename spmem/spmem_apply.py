#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Installs SpMem on the global attention layers of a built decoder config.

Each global layer gets the SpMem mask and cache; the token embedding gains `mem_weight`; the decoder
is re-typed onto a `SpMemDecoderMixin` subclass. Sliding-window layers are left alone (their window
now spans the augmented stream). A KV-shared receiver takes its donor's coset, as in `twa_apply`.

`keep_global_ordinals` builds a hybrid: those globals keep their stock mask and cache.

    decoder_cfg = apply_spmem_to_decoder(
        decoder_cfg,
        geometry=SpMemGeometry(mem_chunk_size=8, num_mem=1, w0=4096),
        coset_assignments=(0, 1) * 4,
        memory_id_base=vocab_size - 1,
    )
"""

from collections.abc import Collection, Sequence

from axlearn.common.attention import MultiheadAttention
from axlearn.common.attention_bias import MaskFnAttentionBias
from axlearn.common.decoder import Decoder

from spmem.spmem_cache import SpMemCache
from spmem.spmem_decoder import SpMemDecoderDense, SpMemDecoderMixin
from spmem.spmem_embedding import install_spmem_token_embedding
from spmem.spmem_geometry import SpMemGeometry, seq_gating_block_size
from spmem.spmem_mask import spmem_mask_fn
from twa.twa_apply import (
    assign_cosets,
    global_attention_configs,
    is_kv_shared_attention,
)


def apply_spmem_to_decoder(
    decoder_cfg: Decoder.Config,
    *,
    geometry: SpMemGeometry,
    coset_assignments: Sequence[int],
    memory_id_base: int,
    keep_global_ordinals: Collection[int] = (),
    spmem_decoder_cls: type[SpMemDecoderMixin] = SpMemDecoderDense,
) -> Decoder.Config:
    """Installs SpMem on `decoder_cfg`'s globals and returns the re-typed decoder config.

    Mutates the layer configs in place; the caller must use the return value.

    Args:
        decoder_cfg: A built `Decoder.Config`.
        geometry: The stream layout and retention.
        coset_assignments: Coset (0 or 1) per non-KV-shared compressed global, in layer order.
        memory_id_base: First of `num_mem` reserved vocab ids that mark memory tokens.
        keep_global_ordinals: Global ordinals (from 0) left at full attention.
        spmem_decoder_cls: The `SpMemDecoderMixin` subclass built on `decoder_cfg.klass`.

    Raises:
        ValueError: A bad `keep_global_ordinals` or `coset_assignments`, or not exactly one flash
            block size across the globals.
        NotImplementedError: `spmem_decoder_cls` is not built on `decoder_cfg.klass`.
    """
    if not issubclass(spmem_decoder_cls, decoder_cfg.klass):
        raise NotImplementedError(
            f"{spmem_decoder_cls.__name__} is not built on decoder klass "
            f"{decoder_cfg.klass.__name__}; pass the matching spmem_decoder_cls."
        )
    attn_cfgs = global_attention_configs(decoder_cfg)
    keep = _validate_keep_global(attn_cfgs, keep_global_ordinals)
    # The augmented length aligns to the flash block, so the globals must agree on one.
    block_sizes = {b for b in map(seq_gating_block_size, attn_cfgs) if b is not None}
    if len(block_sizes) != 1:
        raise ValueError(
            f"expected exactly one flash block size across the global layers; got "
            f"{sorted(block_sizes)}."
        )
    compressed = [a for ordinal, a in enumerate(attn_cfgs) if ordinal not in keep]
    for attn_cfg, coset in zip(compressed, assign_cosets(compressed, coset_assignments)):
        enable_spmem_attention(attn_cfg, geometry=geometry, coset_index=coset)
    install_spmem_token_embedding(
        decoder_cfg, memory_id_base=memory_id_base, num_mem=geometry.num_mem
    )
    return spmem_decoder_cls.from_decoder_config(
        decoder_cfg,
        mem_chunk_size=geometry.mem_chunk_size,
        num_mem=geometry.num_mem,
        memory_id_base=memory_id_base,
        hw_block_size=block_sizes.pop(),
    )


def enable_spmem_attention(
    cfg: MultiheadAttention.Config, *, geometry: SpMemGeometry, coset_index: int
) -> MultiheadAttention.Config:
    """Installs the SpMem cache and its matching mask on one attention layer.

    One call, so the two cannot be given different geometry or cosets.
    """
    cfg.set(
        causal=None,  # Superseded by the SpMem mask.
        kv_cache=SpMemCache.default_config().set(
            geometry=geometry,
            coset_index=coset_index,
            cache_dtype=cfg.kv_cache.cache_dtype,
            kv_partition_spec=cfg.kv_cache.kv_partition_spec,
        ),
        mask=MaskFnAttentionBias.default_config(
            mask=spmem_mask_fn(geometry, coset_index=coset_index)
        ),
    )
    return cfg


def _validate_keep_global(
    attn_cfgs: Sequence[MultiheadAttention.Config], ordinals: Collection[int]
) -> frozenset[int]:
    keep = frozenset(ordinals)
    if not keep:
        return keep
    num_globals = len(attn_cfgs)
    if any(map(is_kv_shared_attention, attn_cfgs)):
        # A kept receiver on a SpMem donor would read a compressed cache with a full mask.
        raise ValueError("keep_global_ordinals is not supported when global layers share KV.")
    out_of_range = sorted(o for o in keep if not 0 <= o < num_globals)
    if out_of_range:
        raise ValueError(f"keep_global_ordinals {out_of_range} are outside [0, {num_globals}).")
    if len(keep) == num_globals:
        raise ValueError(
            f"keep_global_ordinals keeps all {num_globals} global layers, leaving no SpMem layer."
        )
    return keep

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Dedicated trainable embedding for SpMem memory tokens.

Memory tokens are MARKED with reserved vocab ids the training data never emits (see
`spmem_augment.insert_memory_tokens`), but their vector must not come from those vocab rows: the table
is tied to the LM head, so every real position's softmax pushes the marker rows down as an
un-predictable class. `SpMemTokenEmbedding` keeps the marker ids -- placement, mask and cache geometry
are unchanged -- and swaps only the looked-up vector for a `mem_weight [num_mem, dim]` the LM head
never sees.

`mem_weight` is a token-embedding row by construction, so it inherits the table's conventions:
`Embedding`'s `fan_out` normal init (component std `1/sqrt(dim)`, since `mem_weight` matches
`PARAM_REGEXP_WEIGHT`) and the same `Embedding._scale` on the way out (`Scale.UNIT` gives
component std ~1).
"""

import jax.numpy as jnp
from axlearn.common.base_layer import ParameterSpec
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.decoder import Decoder
from axlearn.common.layers import Embedding
from axlearn.common.utils import Tensor, maybe_shard

# Where `install_spmem_token_embedding` puts `mem_weight`, relative to a causal LM model. No
# pretrained checkpoint holds it, so a warm start must leave this scope out of its restore request.
MEM_WEIGHT_PATH = "decoder/emb/token_emb/mem_weight"


def install_spmem_token_embedding(
    decoder_cfg: Decoder.Config, *, memory_id_base: int, num_mem: int
) -> None:
    """Re-types `decoder_cfg.emb.token_emb` onto `SpMemTokenEmbedding` in place.

    Works for any `emb` holding a `token_emb`, or a multimodal one holding it under its text modality,
    since all of them route forward, prefill and decode through it.

    Args:
        decoder_cfg: A built `Decoder.Config` (mutated).
        memory_id_base: First reserved memory-marker id (the j-th memory is base + j).
        num_mem: m -- memory tokens per chunk.
    """
    owner = decoder_cfg.emb
    if "modalities" in owner.keys():
        owner = owner.modalities["text"]
    src = owner.token_emb
    cfg = SpMemTokenEmbedding.default_config()
    # `Embedding.Config` rejects unknown fields, so copy onto a fresh Config that carries the extra
    # ones rather than setting them in place (cf. `SpMemDecoderMixin.from_decoder_config`).
    for key, value in src.items():
        if key != "klass":  # klass is already this SpMem embedding
            cfg.set(**{key: value})
    cfg.set(memory_id_base=memory_id_base, num_mem=num_mem)
    owner.token_emb = cfg


class SpMemTokenEmbedding(Embedding):
    """Token embedding whose memory-marker ids read `mem_weight` instead of their vocab row."""

    @config_class
    class Config(Embedding.Config):
        """`Embedding.Config` + which ids are memory markers."""

        memory_id_base: Required[int] = REQUIRED  # First marker id (the j-th memory is base + j).
        num_mem: Required[int] = REQUIRED  # m -- number of marker ids / `mem_weight` rows.

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        specs = super()._create_layer_parameter_specs()
        # m (e.g. 4) does not divide the vocab-major axes, so that axis is replicated; model-dim keeps
        # the table's spec, so both lookups produce identically sharded activations.
        specs["mem_weight"] = ParameterSpec(
            shape=[cfg.num_mem, cfg.dim], mesh_axes=(None, cfg.param_partition_spec[-1])
        )
        return specs

    def forward(self, x: Tensor) -> Tensor:
        """`Embedding.forward` for real tokens; marker ids take their vector from `mem_weight`.

        `_scale` is elementwise, so scaling the `mem_weight` rows is identical to scaling after the
        substitution -- and it leaves the real-token path as plain `super().forward`.
        """
        cfg = self.config
        activation = super().forward(x)
        mem_table = self._scale(self.parameters["mem_weight"])
        mem_index = x - cfg.memory_id_base
        is_mem = (mem_index >= 0) & (mem_index < cfg.num_mem)
        # Clip keeps the gather in bounds at non-memory positions; `is_mem` discards those rows.
        mem_activation = mem_table[jnp.clip(mem_index, 0, cfg.num_mem - 1)]
        activation = jnp.where(is_mem[..., None], mem_activation, activation)
        activation = maybe_shard(activation, cfg.output_partition_spec)
        return activation

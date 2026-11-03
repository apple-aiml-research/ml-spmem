#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Tests for `spmem_embedding.py` -- SpMem's dedicated memory-token embedding.

The point of `mem_weight` is that a memory token's vector is INDEPENDENT of the reserved vocab row
whose id marks it (that row is tied to the LM head), while still behaving like a token-embedding
row: `1/sqrt(dim)` init, scaled by the table's own `Scale` on the way out.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from axlearn.common import test_utils
from axlearn.common.attention import StackedTransformerLayer
from axlearn.common.causal_lm import gpt_decoder_config
from axlearn.common.layers import Embedding
from axlearn.common.module import functional

from spmem.spmem_embedding import (
    SpMemTokenEmbedding,
    install_spmem_token_embedding,
)

_DIM, _VOCAB, _MEM_BASE, _NUM_MEM = 8, 32, 6, 2


class SpMemTokenEmbeddingTest(test_utils.TestCase):
    """Forward behavior: markers read `mem_weight`, reals read the table, both share `_scale`."""

    @parameterized.parameters(dict(scale=Embedding.Scale.UNIT), dict(scale=None))
    def test_memory_ids_read_mem_weight_not_the_vocab_row(self, scale):
        """A marker id's vector comes from `mem_weight[j]`, so zeroing its vocab row changes nothing
        -- that row only marks the position and stays available to the tied LM head. Memory follows
        the table's own `Scale` convention: `sqrt(dim)` under `Scale.UNIT`, pass-through when unscaled."""
        layer = _embedding(scale=scale)
        params = layer.initialize_parameters_recursively(jax.random.PRNGKey(0))
        ids = jnp.asarray([[10, _MEM_BASE, _MEM_BASE + 1, 11]], dtype=jnp.int32)

        out = _forward(layer, params, ids)
        factor = np.sqrt(_DIM) if scale is not None else 1.0
        self.assertNestedAllClose(out[0, 0], params["weight"][10] * factor)
        self.assertNestedAllClose(out[0, 1], params["mem_weight"][0] * factor)
        self.assertNestedAllClose(out[0, 2], params["mem_weight"][1] * factor)

        # The marker rows of the table are unused: zero them and the memory vectors are unchanged.
        zeroed = dict(params, weight=params["weight"].at[_MEM_BASE : _MEM_BASE + _NUM_MEM].set(0.0))
        self.assertNestedAllClose(_forward(layer, zeroed, ids), out)

    def test_mem_weight_is_a_token_embedding_row(self):
        """`mem_weight` rows init at component std `1/sqrt(dim)` (unit L2 norm) like `token_emb`
        rows, so `Scale.UNIT`'s `sqrt(dim)` yields component std ~1 and a per-row unit-norm
        renormalizer has nothing to undo at step 0. The m axis is replicated -- m does not divide
        the vocab-major mesh axes the table shards over -- and only model-dim keeps the table's spec.
        """
        layer = _embedding(
            scale=Embedding.Scale.UNIT,
            dim=1024,
            num_mem=8,
            param_partition_spec=(("expert", "fsdp", "seq"), "model"),
        )
        specs = layer.create_parameter_specs_recursively()
        self.assertEqual(tuple(specs["mem_weight"].mesh_axes), (None, "model"))
        self.assertEqual(specs["mem_weight"].shape, [8, 1024])

        params = layer.initialize_parameters_recursively(jax.random.PRNGKey(0))
        row_norms = np.linalg.norm(np.asarray(params["mem_weight"]), axis=-1)
        # chi-distributed around 1 with std ~1/sqrt(2*dim) (~2% at dim=1024); 10% is ample slack.
        self.assertNestedAllClose(row_norms, np.ones(8), rtol=0.1)


class InstallSpMemTokenEmbeddingTest(test_utils.TestCase):
    """`install_spmem_token_embedding` re-types `emb.token_emb` and preserves its every field."""

    def test_re_type_preserves_the_table_config(self):
        decoder_cfg = _decoder_cfg()
        decoder_cfg.emb.token_emb.set(
            scale=Embedding.Scale.UNIT, param_partition_spec=(("expert", "fsdp"), "model")
        )
        install_spmem_token_embedding(decoder_cfg, memory_id_base=_MEM_BASE, num_mem=_NUM_MEM)

        token_emb = decoder_cfg.emb.token_emb
        self.assertIs(token_emb.klass, SpMemTokenEmbedding)
        self.assertEqual(token_emb.memory_id_base, _MEM_BASE)
        self.assertEqual(token_emb.num_mem, _NUM_MEM)
        # Every base field survives the re-type (scale, sharding).
        self.assertEqual(token_emb.scale, Embedding.Scale.UNIT)
        self.assertEqual(token_emb.param_partition_spec, (("expert", "fsdp"), "model"))

        # Warm-start converters and any per-row renormalizer match `mem_weight` by path, so it must
        # land under the token table; the decoder still sizes the table itself (vocab x dim).
        decoder = decoder_cfg.set(name="dec").instantiate(parent=None)
        specs = dict(_flatten(decoder.create_parameter_specs_recursively()))
        self.assertEqual(specs["emb/token_emb/mem_weight"].shape, [_NUM_MEM, _DIM])
        self.assertEqual(specs["emb/token_emb/weight"].shape, [_VOCAB, _DIM])


def _decoder_cfg():
    return gpt_decoder_config(
        stack_cfg=StackedTransformerLayer.default_config(),
        num_layers=2,
        hidden_dim=_DIM,
        num_heads=2,
        vocab_size=_VOCAB,
        max_position_embeddings=64,
    )


def _embedding(*, scale, dim: int = _DIM, num_mem: int = _NUM_MEM, param_partition_spec=None):
    """A standalone `SpMemTokenEmbedding` (the layer under test, no decoder around it)."""
    cfg = SpMemTokenEmbedding.default_config().set(
        name="token_emb",
        num_embeddings=_VOCAB,
        dim=dim,
        scale=scale,
        memory_id_base=_MEM_BASE,
        num_mem=num_mem,
    )
    if param_partition_spec is not None:
        cfg.set(param_partition_spec=param_partition_spec)
    return cfg.instantiate(parent=None)


def _forward(layer, params, ids):
    out, _ = functional(
        layer,
        inputs=dict(x=ids),
        state=params,
        is_training=False,
        prng_key=jax.random.PRNGKey(1),
    )
    return out


def _flatten(specs, prefix: str = ""):
    """`(path, spec)` pairs of a nested parameter-spec dict."""
    for key, value in specs.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            yield from _flatten(value, path)
        else:
            yield path, value


if __name__ == "__main__":
    absltest.main()

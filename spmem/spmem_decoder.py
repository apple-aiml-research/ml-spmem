#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""`SpMemDecoderMixin`: memory insertion inside the decoder, for training and decoding.

Callers feed a pure-real stream. `forward` expands it with `insert_memory_tokens`, runs the base
decoder over the augmented stream, and gathers the real positions back out, so the loss never sees
memory. `extend_step` does the same per prefill or decode block: memory lands at global chunk seals
(via `real_offset`), `time_step` counts augmented positions, and logits are taken only at reals.

`SpMemDecoderDense` mixes it into a plain `Decoder`; a model with its own decoder mixes it the same
way.
"""

import math

import jax.numpy as jnp
from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.decoder import Decoder, ForwardMode
from axlearn.common.ein_ops import repeat
from axlearn.common.module import nowrap
from axlearn.common.utils import (
    Nested,
    NestedTensor,
    Tensor,
    get_current_abstract_or_physical_mesh,
    maybe_shard,
)

from spmem.spmem_augment import insert_memory_tokens, segment_relative_positions
from spmem.spmem_geometry import (
    augmented_seq_len,
    decompose_augmented_position,
)
from twa.twa_apply import global_attention_configs


class SpMemDecoderMixin(BaseLayer):
    """Memory insertion, mixed in ahead of a decoder base.

    Carries no Config: axlearn Configs cannot diamond-merge, so each concrete decoder adds these
    fields to its own base's Config (see `SpMemDecoderDense`):

        mem_chunk_size: C -- reals per chunk.
        num_mem: m -- memory tokens after each chunk.
        memory_id_base: first reserved memory id (the j-th memory is base + j).
        hw_block_size: flash block size the augmented length aligns to.

    The augmented streams are sharded by the base's `logits_partition_spec`.
    """

    def __init__(self, cfg, *, parent):
        super().__init__(cfg, parent=parent)
        self._mem_chunk_size = cfg.mem_chunk_size
        self._num_mem = cfg.num_mem
        self._memory_id_base = cfg.memory_id_base
        self._hw_block_size = cfg.hw_block_size
        self._pad_id = cfg.pad_token_id

    @classmethod
    def from_decoder_config(
        cls,
        decoder_cfg,
        *,
        mem_chunk_size: int,
        num_mem: int,
        memory_id_base: int,
        hw_block_size: int,
    ):
        """Copies a built base `Decoder.Config` onto this class's Config, plus the SpMem fields.

        Sub-configs are shared by reference, so layers already given SpMem keep it.
        """
        cfg = cls.default_config()
        for key, value in decoder_cfg.items():
            if key != "klass":
                cfg.set(**{key: value})
        cfg.set(
            mem_chunk_size=mem_chunk_size,
            num_mem=num_mem,
            memory_id_base=memory_id_base,
            hw_block_size=hw_block_size,
        )
        return cfg

    def _flash_align(self) -> int:
        """Each shard's flash query divides the block, and the stream divides its own sharding.

        Shard counts come from the live mesh. Cache and stream must use the same number, so a
        seq-sharded decoder with no mesh raises rather than guessing.
        """
        query_shards = _mesh_size(self._query_seq_axes())
        stream_shards = _mesh_size(self.config.logits_partition_spec[1])
        return math.lcm(self._hw_block_size * query_shards, stream_shards)

    def _query_seq_axes(self) -> tuple[str, ...]:
        """Mesh axes the global layers shard the query's sequence over, from `q_partition_spec`."""
        axes = set()
        for attn_cfg in global_attention_configs(self.config):
            spec = getattr(attn_cfg, "q_partition_spec", None)
            seq = spec[1] if spec is not None else None
            axes.add((seq,) if isinstance(seq, str) else tuple(seq or ()))
        if len(axes) != 1:
            raise ValueError(f"expected one query sharding across the global layers; got {axes}.")
        return axes.pop()

    def _augmented_len(self, real_len: int) -> int:
        return augmented_seq_len(
            real_len,
            mem_chunk_size=self._mem_chunk_size,
            num_mem=self._num_mem,
            align=self._flash_align(),
        )

    @nowrap
    def init_states(self, *, batch_size: int, max_sequence_length: int, dtype: jnp.dtype):
        """`Decoder.init_states` with every cache sized for the augmented stream."""
        return super().init_states(
            batch_size=batch_size,
            max_sequence_length=self._augmented_len(max_sequence_length),
            dtype=dtype,
        )

    def _insert_memory(
        self, input_batch: Nested[Tensor], **kwargs
    ) -> tuple[Tensor, Tensor, Tensor]:
        """`insert_memory_tokens` on `input_batch`; returns `(aug_ids, aug_segment_ids, real_slot)`."""
        return insert_memory_tokens(
            input_batch["input_ids"],
            mem_chunk_size=self._mem_chunk_size,
            num_mem=self._num_mem,
            memory_id_base=self._memory_id_base,
            # None (a single-sequence prompt) is read as one segment.
            segment_ids=input_batch.get("input_segment_ids"),
            pad_id=self._pad_id,
            **kwargs,
        )

    def _augment_sequence(self, input_batch: Nested[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Augments a whole sequence, sized per input; the streams are sharded like activations."""
        aug_ids, aug_seg, real_slot = self._insert_memory(
            input_batch, max_seq_len=self._augmented_len(input_batch["input_ids"].shape[-1])
        )
        spec = self.config.logits_partition_spec[:2]
        return maybe_shard(aug_ids, spec), maybe_shard(aug_seg, spec), real_slot

    def _gather_reals(self, hidden: Tensor, real_slot: Tensor) -> Tensor:
        """`[B, L_aug, D]` -> `[B, L_real, D]`, re-sharded like the base's pre-head hidden."""
        hidden = jnp.take_along_axis(hidden, real_slot[:, :, None], axis=1)
        return maybe_shard(hidden, self.config.logits_partition_spec)

    def forward(self, input_batch: Nested[Tensor], **kwargs) -> dict[str, Tensor]:
        """Augments, runs the base `forward`, and returns real-position hidden states."""
        aug_ids, aug_seg, real_slot = self._augment_sequence(input_batch)
        # Hoisted: deriving positions inside every layer OOMs a 31B model at batch 64.
        aug_pos = maybe_shard(
            segment_relative_positions(aug_seg), self.config.logits_partition_spec[:2]
        )
        aug_batch = {
            **input_batch,
            "input_ids": aug_ids,
            "input_segment_ids": aug_seg,
            "positions": aug_pos,
        }
        outputs = super().forward(aug_batch, **kwargs)
        return {**outputs, "hidden_states": self._gather_reals(outputs["hidden_states"], real_slot)}

    def _augment_for_extend(
        self, input_batch: Nested[Tensor], cached_states: Nested[Tensor], *, is_prefill: bool
    ) -> tuple[Nested[Tensor], Tensor, Tensor]:
        """Returns `(aug_input_batch, logit_index, real_slot)` for one extend step.

        `logit_index` is the augmented position(s) whose logits are kept: each row's last real in
        prefill, every real in decode.
        """
        if is_prefill:
            aug_ids, aug_seg, real_slot = self._augment_sequence(input_batch)
            # The last real, not the last position, which is memory when the prompt seals a chunk.
            num_reals = jnp.sum(input_batch["input_ids"] != self._pad_id, axis=-1)
            logit_index = jnp.take_along_axis(
                real_slot, jnp.maximum(num_reals - 1, 0)[:, None], axis=1
            )
        else:
            # A decode block of K reals, offset to the global chunk grid, in a static worst-case
            # block of `K + ceil(K/C) * m`; off-seal steps leave the memory slots as pad.
            k, c, m = input_batch["input_ids"].shape[-1], self._mem_chunk_size, self._num_mem
            _, _, real_offset = decompose_augmented_position(
                cached_states["time_step"], mem_chunk_size=c, num_mem=m
            )
            aug_ids, aug_seg, real_slot = self._insert_memory(
                input_batch, max_seq_len=k + math.ceil(k / c) * m, real_offset=real_offset
            )
            logit_index = real_slot
        aug_input_batch = {**input_batch, "input_ids": aug_ids, "input_segment_ids": aug_seg}
        return aug_input_batch, logit_index, real_slot

    def extend_step(
        self,
        *,
        cached_states: Nested[Tensor],
        input_batch: Nested[Tensor],
        is_prefill: bool = False,
        **kwargs,
    ) -> tuple[NestedTensor, NestedTensor]:
        """Prefill or decode over the augmented block.

        Mirrors `Decoder.extend_step`, with three deltas marked (1)-(3). `time_step` is the
        augmented position.
        """
        # (1) SpMem: real -> augmented block.
        aug_input_batch, logit_index, real_slot = self._augment_for_extend(
            input_batch, cached_states, is_prefill=is_prefill
        )

        cfg = self.config
        mode = ForwardMode.PREFILL if is_prefill else ForwardMode.EXTEND_STEP
        time_step: Tensor = cached_states["time_step"]
        assert time_step.ndim == 1

        input_ids: Tensor = aug_input_batch["input_ids"]
        batch, step = input_ids.shape
        # This position is for embedding, not attention.
        positions = repeat(jnp.arange(step), "t -> b t", b=batch) + time_step[:, None]

        # (2) SpMem: no T5-style `attention_mask` branch.
        assert getattr(cfg, "attention_mask", None) is None, (
            "SpMem does not support attention_mask."
        )
        if "positions" in kwargs:
            raise ValueError("positions is supported only in FORWARD.")

        input_segment_ids = aug_input_batch.get("input_segment_ids", None)
        updated_states, outputs = super()._forward_for_mode(
            mode=mode,
            input_batch={**aug_input_batch, "positions": positions},
            self_attention_logit_biases=None,
            cached_states=cached_states,
            **kwargs,
        )
        step_len = (
            jnp.sum(input_segment_ids != 0, axis=-1) if input_segment_ids is not None else step
        )
        # (3) SpMem: logits only at `logit_index`, and real-position hidden states for any head
        # that reads them.
        picked = jnp.take_along_axis(outputs["hidden_states"], logit_index[:, :, None], axis=1)
        outputs["logits"] = self.compute_logits(dict(hidden_states=picked))
        self.add_module_output(
            "hidden_states", self._gather_reals(outputs["hidden_states"], real_slot)
        )
        updated_states.update(time_step=cached_states["time_step"] + step_len)
        return updated_states, outputs


class SpMemDecoderDense(SpMemDecoderMixin, Decoder):
    """SpMem over a plain axlearn `Decoder`."""

    @config_class
    class Config(Decoder.Config):
        """`Decoder.Config` + the fields documented in `SpMemDecoderMixin`."""

        mem_chunk_size: Required[int] = REQUIRED
        num_mem: Required[int] = REQUIRED
        memory_id_base: Required[int] = REQUIRED
        hw_block_size: Required[int] = REQUIRED


def _mesh_size(axes) -> int:
    """Product of the live mesh's sizes over `axes` (a name, a tuple of names, or None)."""
    axes = (axes,) if isinstance(axes, str) else tuple(axes or ())
    if not axes:
        return 1
    mesh = get_current_abstract_or_physical_mesh()
    if mesh.empty:
        raise RuntimeError(
            f"No live mesh, but seq is sharded over {axes}; call under the trainer mesh."
        )
    return math.prod(mesh.shape.get(ax, 1) for ax in axes)

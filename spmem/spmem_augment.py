#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Memory-token placement: a pure-real stream in, the augmented stream out.

`insert_memory_tokens` puts `m` memory ids (`memory_id_base + j`) after every full chunk of `C`
reals, per segment, over a fixed augmented length, and returns `real_slot`, the map that gathers
the reals back out. Positions are derived downstream from the augmented segment ids.

Pure `jax.numpy` with static shapes, so it runs inside the model and vmaps over the batch. Padding
is `segment_ids == 0` when segment ids are given, else `input_ids == pad_id`.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from axlearn.common.utils import Tensor


def _augment_one(
    input_ids: Tensor,  # [L] int
    segment_ids: Optional[Tensor],  # [L] int, or None for a single global segment
    real_offset: Tensor,  # [] int: global real index of position 0 (chunk-phase shift; decode)
    *,
    mem_chunk_size: int,
    num_mem: int,
    max_seq_len: int,
    memory_id_base: int,
    pad_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Augment a single pure-real row; see `insert_memory_tokens`. Returns 3 arrays."""
    c, m = mem_chunk_size, num_mem
    length = input_ids.shape[-1]
    idx = jnp.arange(length, dtype=jnp.int32)

    # Segment-start mask: True at each segment start (position 0 always opens segment 1). With
    # `segment_ids`, a boundary is wherever the id changes; else a single global segment.
    if segment_ids is not None:
        prev = jnp.concatenate([segment_ids[:1], segment_ids[:-1]])
        seg_start = segment_ids != prev
    else:
        seg_start = jnp.zeros(length, dtype=bool)
    seg_start = seg_start.at[0].set(True)

    # Per-segment real index (reset at each segment start).
    seg_start_idx = jax.lax.cummax(jnp.where(seg_start, idx, jnp.int32(-1)))
    within_seg = idx - seg_start_idx

    # Real validity, driving BOTH seals and padding below: segment-based when `segment_ids` is
    # given, else the token value (base `Decoder`'s convention). A segment-marked pad is honored
    # even if its id isn't `pad_id` -- the multistep-decode case (speculative / incremental / etc).
    valid = (segment_ids != 0) if segment_ids is not None else (input_ids != pad_id)

    # mem tokens seal after CONTENT (valid), not padding. `real_offset` shifts the chunk phase so a
    # block starting mid-stream (decode) seals at GLOBAL chunk boundaries (0 for training/prefill,
    # where each segment starts a fresh chunk grid).
    seals = ((((within_seg + real_offset) + 1) % c) == 0) & valid
    groups_before = jnp.cumsum(seals) - seals  # exclusive prefix
    real_slot = idx + m * groups_before  # [L] augmented slot of each real

    # Scatter reals (invalid reals -> pad_id, keeping the augmented ids clean), then the m memory
    # ids after each sealed real. The per-seal slots `real_slot+1 .. real_slot+m` are disjoint
    # across seals (next real at `real_slot+1+m`), so one batched scatter suffices — no collisions.
    aug = jnp.full((max_seq_len,), pad_id, dtype=input_ids.dtype)
    aug = aug.at[real_slot].set(jnp.where(valid, input_ids, pad_id), mode="drop")
    oob = jnp.int32(max_seq_len)  # sentinel index -> dropped
    j_idx = jnp.arange(m, dtype=jnp.int32)  # [m]
    mem_slots = jnp.where(seals[:, None], real_slot[:, None] + 1 + j_idx[None, :], oob)  # [L, m]
    mem_vals = jnp.broadcast_to(jnp.asarray(memory_id_base, aug.dtype) + j_idx, mem_slots.shape)
    aug = aug.at[mem_slots].set(mem_vals, mode="drop")

    # Augmented padding = NOT (valid real ∪ sealed-chunk memory), taken from `valid` scattered to
    # aug slots -- not `aug == pad_id`, so validity stays keyed on segment/value, not the token id.
    aug_valid = jnp.zeros((max_seq_len,), dtype=bool).at[real_slot].set(valid, mode="drop")
    aug_valid = aug_valid.at[mem_slots].set(True, mode="drop")  # sealed memory slots; oob dropped
    is_pad = ~aug_valid

    # Scatter VALID real segment-start flags to their augmented slots (memory inherits its chunk's
    # segment); the segment ids then mirror a BOS-derived segmentation (0 on pad). The model derives
    # per-segment positions from these via `segment_relative_positions` (feeding RoPE + the mask).
    aug_start = jnp.zeros((max_seq_len,), dtype=bool)
    aug_start = aug_start.at[real_slot].set(seg_start & valid, mode="drop")
    seg_ids = jnp.where(is_pad, jnp.int32(0), jnp.cumsum(aug_start.astype(jnp.int32)))

    return aug, seg_ids, real_slot


def insert_memory_tokens(
    input_ids: Tensor,  # [B, L] int
    *,
    mem_chunk_size: int,
    num_mem: int,
    max_seq_len: int,
    memory_id_base: int,
    segment_ids: Optional[Tensor] = None,  # [B, L]; None => single segment (global stride)
    real_offset: Tensor = 0,  # [B] or scalar int: global real index of each row's position 0
    pad_id: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Insert memory tokens into a pure-real `[B, L]` stream; per-row via `jax.vmap`.

    Args:
        input_ids (array): `[B, L]` pure-real token ids.
        mem_chunk_size: C -- reals per chunk (memory-insertion period).
        num_mem: m -- memory tokens inserted after each full chunk.
        max_seq_len: L_aug -- fixed augmented output length (>= L*(C+m)/C to avoid
            dropping reals).
        segment_ids (array): `[B, L]` per-token segment ids; a segment starts where the id
            changes (training packs multiple docs per row, segmented upstream by BOS), and
            `segment_ids == 0` marks padding. `None` => one global segment and padding falls back
            to `input_ids == pad_id` (the eval/prefill single-sequence case). Rank mismatch raises.
        real_offset (array): `[B]` (or scalar) global real index of each row's position 0, so a
            block starting mid-stream (decode of K reals) seals at GLOBAL chunk boundaries.
            Default 0 -- training/prefill start a fresh chunk grid (per segment).
        memory_id_base: first reserved memory id (the j-th memory is base + j).
        pad_id: pad id for the augmented tail.

    Returns:
        `(aug_input_ids, aug_segment_ids, real_slot)`: the first two are `[B, max_seq_len]`;
        `real_slot` is `[B, L]` (augmented index of each input real).
    """
    if segment_ids is not None and segment_ids.ndim != input_ids.ndim:
        # segment_ids is vmapped per-row alongside input_ids, so a rank mismatch (e.g. 1-D
        # segment_ids with [B, L] input) would silently map the wrong axis. Reject it.
        raise ValueError(
            "segment_ids must be [B, L] like input_ids; got segment_ids.ndim="
            f"{segment_ids.ndim}, input_ids.ndim={input_ids.ndim}"
        )

    real_offset = jnp.broadcast_to(jnp.asarray(real_offset, dtype=jnp.int32), (input_ids.shape[0],))

    def one(row_ids, row_seg, row_offset):
        return _augment_one(
            row_ids,
            row_seg,
            row_offset,
            mem_chunk_size=mem_chunk_size,
            num_mem=num_mem,
            max_seq_len=max_seq_len,
            memory_id_base=memory_id_base,
            pad_id=pad_id,
        )

    if segment_ids is None:
        return jax.vmap(lambda r, o: one(r, None, o))(input_ids, real_offset)
    return jax.vmap(one)(input_ids, segment_ids, real_offset)


def segment_relative_positions(segment_ids: Tensor) -> Tensor:
    """Positions restarting at 0 in each segment; padding (`segment_ids == 0`) gets 0.

    Args:
        segment_ids: `[batch, seq_len]` integer segment ids.

    Returns:
        `[batch, seq_len]` segment-relative positions.
    """
    shifted = jnp.pad(segment_ids[:, :-1], ((0, 0), (1, 0)))
    is_reset = (segment_ids != shifted) | (segment_ids == 0)
    indices = jnp.arange(1, segment_ids.shape[1] + 1)
    reset_indices = jnp.where(is_reset, indices, 0)
    return indices - jax.lax.cummax(reset_indices, axis=1)

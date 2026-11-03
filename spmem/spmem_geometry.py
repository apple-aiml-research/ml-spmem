#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""SpMem geometry: the augmented stream's layout and the two pools' retention.

The augmented stream puts `num_mem` (m) memory tokens after every `mem_chunk_size` (C) reals, a
period of P = C + m. `SpMemGeometry` is the one validated description the mask, the cache and the
installer share, so training and decoding cannot disagree about what is retained.
"""

import dataclasses
import math
from typing import Optional

from twa.twa_geometry import TaperGeometry, TaperRule, taper_geometry

# Splash block sizes that gate the sequence length (fwd, dq, dkv); the `*_compute` tiles do not.
_SEQ_GATING_SPLASH_BLOCKS = (
    "splash_block_q",
    "splash_block_kv",
    "splash_block_q_dkv",
    "splash_block_kv_dkv",
    "splash_block_q_dq",
    "splash_block_kv_dq",
)


def spmem_taper_geometry(*, w0: int, w1: Optional[int] = None) -> TaperGeometry:
    """The mipmap taper SpMem uses on either pool; `w1` is a demand, None the full reach."""
    return taper_geometry(w0=w0, w1=w1, rule=TaperRule.MIPMAP)


@dataclasses.dataclass(frozen=True)
class SpMemGeometry:
    """Stream layout and retention for one SpMem model; validated on construction.

    Attributes:
        mem_chunk_size: C -- reals per chunk.
        num_mem: m -- memory tokens after each chunk, at most C.
        w0: dense window of the real taper, in reals; also the window a memory query reads.
        w1: receptive field the real taper must reach, in reals; None takes its full reach.
        mem_window_size: bounds the memory pool to a ring over this many reals' chunks.
        mem_w0: dense window of the memory taper, in memory slots. Exclusive with the ring.
        mem_w1: receptive field of the memory taper, in memory slots. Both or neither with
            `mem_w0`. With no ring and no taper, every earlier memory is kept.
    """

    mem_chunk_size: int
    num_mem: int
    w0: int
    w1: Optional[int] = None
    mem_window_size: Optional[int] = None
    mem_w0: Optional[int] = None
    mem_w1: Optional[int] = None

    def __post_init__(self):
        if self.mem_chunk_size <= 0:
            raise ValueError(f"mem_chunk_size must be positive, got {self.mem_chunk_size}.")
        if not 1 <= self.num_mem <= self.mem_chunk_size:
            # Consumers divide by the compression ratio C // m.
            raise ValueError(
                f"num_mem must be in [1, C={self.mem_chunk_size}], got {self.num_mem}."
            )
        spmem_taper_geometry(w0=self.w0, w1=self.w1)  # Raises on a bad w0 / w1.
        if (self.mem_w0 is None) != (self.mem_w1 is None):
            raise ValueError(
                f"the memory taper needs both mem_w0 and mem_w1 or neither; got "
                f"mem_w0={self.mem_w0}, mem_w1={self.mem_w1}."
            )
        if self.mem_w0 is not None:
            if self.mem_window_size is not None:
                raise ValueError("mem_window_size and mem_w0 both bound the memory pool.")
            spmem_taper_geometry(w0=self.mem_w0, w1=self.mem_w1)
        if self.mem_window_size is not None and self.mem_window_size < self.w0:
            # A pool narrower than the real window summarizes nothing the reals do not hold.
            raise ValueError(
                f"mem_window_size must be >= w0, got {self.mem_window_size} < {self.w0}."
            )

    @property
    def real_taper(self) -> TaperGeometry:
        return spmem_taper_geometry(w0=self.w0, w1=self.w1)

    @property
    def mem_taper(self) -> Optional[TaperGeometry]:
        """The memory pool's taper over memory ordinals, or None."""
        if self.mem_w0 is None:
            return None
        return spmem_taper_geometry(w0=self.mem_w0, w1=self.mem_w1)

    @property
    def mem_window_chunks(self) -> Optional[int]:
        """Chunks the memory ring spans (a partial chunk still owns its group), or None."""
        if self.mem_window_size is None:
            return None
        return math.ceil(self.mem_window_size / self.mem_chunk_size)

    def memory_slots(self, aug_len: int) -> int:
        """Memory-pool slots for an `aug_len` augmented stream: the taper's, else ring-capped."""
        mem_taper = self.mem_taper
        if mem_taper is not None:
            return mem_taper.slots
        chunks = math.ceil(aug_len / (self.mem_chunk_size + self.num_mem))
        if self.mem_window_chunks is not None:
            chunks = min(chunks, self.mem_window_chunks)
        return chunks * self.num_mem


def decompose_augmented_position(pos, *, mem_chunk_size: int, num_mem: int):
    """Split augmented position(s) into `(chunk, offset, real_pos)`.

    `pos` is memory iff `offset >= C`; `real_pos` is meaningful only for reals. Array-generic.
    """
    period = mem_chunk_size + num_mem
    chunk = pos // period
    offset = pos % period
    real_pos = chunk * mem_chunk_size + offset
    return chunk, offset, real_pos


def augmented_seq_len(real_len: int, *, mem_chunk_size: int, num_mem: int, align: int) -> int:
    """Augmented length for `real_len` reals, `ceil(real_len * P / C)` rounded up to `align`.

    An upper bound over any segmentation, so no real is dropped. `align` is the flash q-block times
    the seq-shard count.
    """
    min_aug = math.ceil(real_len * (mem_chunk_size + num_mem) / mem_chunk_size)
    return math.ceil(min_aug / align) * align


def seq_gating_block_size(attn_cfg) -> Optional[int]:
    """Largest splash block the sequence length must divide; None when flash is off.

    `max(tpu_block_size, *seq-gating backend_overrides)`, an unset override falling back to
    `tpu_block_size` as in axlearn's `get_backend_overrides`.
    """
    block_size = getattr(attn_cfg, "tpu_block_size", None)
    if block_size is None:
        return None
    overrides = getattr(attn_cfg, "backend_overrides", None) or {}
    # `or`, not `.get(name, default)`: an override explicitly set to None falls back too.
    return max(
        block_size, *(overrides.get(name) or block_size for name in _SEQ_GATING_SPLASH_BLOCKS)
    )

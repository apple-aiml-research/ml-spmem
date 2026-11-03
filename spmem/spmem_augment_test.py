#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Parity tests for spmem_augment.

Most tests exercise the per-row core `_augment_one` (no vmap) against an independent numpy
oracle: `_ref_augment_ids` for the augmented ids and `_ref_segment_ids` for segment ids. The model
derives positions from the segment ids, so `_ref_positions` checks that axlearn's
`segment_relative_positions(aug_segment_ids)` reconstructs them. A few
integration tests drive the batched `insert_memory_tokens` (multi-row vmap + the [B, L] segment_ids
rank guard). Covers single-segment (global stride) and packed multi-segment (BOS-reset) rows over a
few (C, m) geometries, on both fully-packed (training) and padded (eval/prefill) inputs -- incl.
segment-id-vs-token-value pad detection.
"""

# pylint: disable=missing-function-docstring,protected-access,no-self-use

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from axlearn.common import test_utils

from spmem.spmem_augment import (
    _augment_one,
    insert_memory_tokens,
    segment_relative_positions,
)

# First reserved memory-marker id; any id range the real stream never emits works.
MEMORY_ID_BASE = 258_885


def _ref_augment_ids(
    ids: np.ndarray,
    *,
    mem_chunk_size: int,
    num_mem: int,
    max_seq_len: int,
    bos_id=None,
    memory_id_base: int = MEMORY_ID_BASE,
    pad_id: int = 0,
) -> np.ndarray:
    """Numpy oracle: reals + `num_mem` memory ids after every full `mem_chunk_size`-chunk
    (per segment), truncated/padded to `max_seq_len`.

    Segment starts are position 0 and every `bos_id`; a segment's final partial chunk gets
    no memory; the pad tail is never augmented.
    """
    c, m = mem_chunk_size, num_mem
    ids = np.asarray(ids).reshape(-1)
    nonpad = np.nonzero(ids != pad_id)[0]
    n = int(nonpad[-1]) + 1 if nonpad.size else 0
    ids = ids[:n]
    content_idx = np.arange(n)
    is_bos = np.zeros(n, dtype=bool)
    if n:
        is_bos[0] = True  # position 0 opens the first segment
    if bos_id is not None:
        is_bos |= ids == bos_id
    within_seg = content_idx - np.maximum.accumulate(np.where(is_bos, content_idx, -1))
    seals = (within_seg + 1) % c == 0  # last real of a full C-chunk within its segment
    groups_before = np.concatenate([[0], np.cumsum(seals)])[:n]  # exclusive prefix
    aug_slot = content_idx + m * groups_before
    aug_len = n + m * int(seals.sum())
    out = np.full(aug_len, pad_id, dtype=np.int32)
    out[aug_slot] = ids
    sealed = np.nonzero(seals)[0]
    if sealed.size:
        mem_slots = (aug_slot[sealed][:, None] + 1 + np.arange(m)[None, :]).reshape(-1)
        out[mem_slots] = np.tile((memory_id_base + np.arange(m)).astype(np.int32), sealed.size)
    if aug_len >= max_seq_len:
        return out[:max_seq_len]
    return np.concatenate([out, np.full(max_seq_len - aug_len, pad_id, dtype=np.int32)])


def _ref_positions(ids: np.ndarray, bos_id: int) -> np.ndarray:
    """Inline replica of long_context.positions_from_bos (per-segment positions)."""
    is_bos = ids == bos_id
    is_bos[0] = True
    idx = np.arange(len(ids), dtype=np.int32)
    seg_start = np.maximum.accumulate(np.where(is_bos, idx, -1))
    pos = (idx - seg_start).astype(np.int32)
    pos[ids == 0] = 0
    return pos


def _ref_segment_ids(ids: np.ndarray, bos_id: int) -> np.ndarray:
    """Numpy reference: segment ids derived from BOS positions (1-based, 0 reserved for pad)."""
    is_bos = ids == bos_id
    is_bos[0] = True
    seg = np.cumsum(is_bos).astype(np.int32)
    seg[ids == 0] = 0
    return seg


# A bos value that never appears -> global stride (matches insert_memory_tokens bos_id=None).
_NO_BOS = -1


def _aug1(ids_1d, *, segment_ids=None, real_offset=0, **kwargs):
    """Run the per-row core `_augment_one` on a single `[L]` row (no vmap); return the 3 outputs
    as numpy arrays."""
    kwargs.setdefault("memory_id_base", MEMORY_ID_BASE)
    kwargs.setdefault("pad_id", 0)
    seg = None if segment_ids is None else jnp.asarray(segment_ids)
    outs = _augment_one(jnp.asarray(ids_1d), seg, jnp.asarray(real_offset, jnp.int32), **kwargs)
    return tuple(np.asarray(x) for x in outs)


class SpMemAugmentParityTest(test_utils.TestCase):
    def _check(self, ids_np, *, mem_chunk_size, num_mem, ref_bos_id):
        # `ref_bos_id` builds the numpy REFERENCE segmentation only: None => single global segment
        # (eval/prefill); 2 => split on BOS(2), driving insert_memory_tokens with the matching
        # `segment_ids` (as the pipeline provides). Generous max_seq_len so nothing truncates.
        msl = 2 * len(ids_np)
        ref_ids = _ref_augment_ids(
            ids_np,
            mem_chunk_size=mem_chunk_size,
            num_mem=num_mem,
            max_seq_len=msl,
            bos_id=ref_bos_id,
        )
        ref_bos = ref_bos_id if ref_bos_id is not None else _NO_BOS
        ref_pos = _ref_positions(ref_ids, ref_bos)
        ref_seg = _ref_segment_ids(ref_ids, ref_bos)

        seg = None if ref_bos_id is None else _ref_segment_ids(ids_np, ref_bos_id)
        aug_ids, aug_seg, real_slot = _aug1(
            ids_np,
            mem_chunk_size=mem_chunk_size,
            num_mem=num_mem,
            max_seq_len=msl,
            segment_ids=seg,
        )

        np.testing.assert_array_equal(aug_ids, ref_ids, "augmented ids != reference")
        np.testing.assert_array_equal(aug_seg, ref_seg, "augmented segment ids mismatch")
        # The model reconstructs positions from aug_seg via `segment_relative_positions`; assert
        # that reconstruction matches the per-segment oracle.
        derived_pos = np.asarray(segment_relative_positions(jnp.asarray(aug_seg)[None]))[0]
        np.testing.assert_array_equal(derived_pos, ref_pos, "derived positions mismatch")
        # real_slot must pull every input real back out of the augmented stream.
        np.testing.assert_array_equal(aug_ids[real_slot], ids_np, "real_slot gather mismatch")

    def test_augment_ids_literal(self):
        # Hand-computed C=2, m=2 case asserted against a literal array. 5 reals ->
        # [C real][m mem][C real][m mem][partial, no mem] = [10 11][6 7][12 13][6 7][14].
        ids = np.array([10, 11, 12, 13, 14], dtype=np.int32)
        aug_ids, aug_seg, real_slot = _aug1(ids, mem_chunk_size=2, num_mem=2, max_seq_len=12)
        b = MEMORY_ID_BASE
        np.testing.assert_array_equal(
            aug_ids, np.array([10, 11, b, b + 1, 12, 13, b, b + 1, 14, 0, 0, 0], np.int32)
        )
        np.testing.assert_array_equal(
            aug_seg, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], np.int32)
        )
        np.testing.assert_array_equal(real_slot, np.array([0, 1, 4, 5, 8], np.int32))

    def test_real_offset_shifts_seal_phase(self):
        # `real_offset` shifts the chunk-seal phase for a block starting mid-stream (decode). C=2,
        # m=2: offset=0 seals after the 2nd real (fresh chunk); offset=1 (one real already in the
        # chunk) seals after the 1st real, then every 2 reals.
        b0, b1 = MEMORY_ID_BASE, MEMORY_ID_BASE + 1

        def aug(real_offset):
            a, _, slot = _aug1(
                [10, 11, 12], mem_chunk_size=2, num_mem=2, max_seq_len=10, real_offset=real_offset
            )
            return list(a), list(slot)

        a0, slot0 = aug(0)  # fresh chunk grid: [10 11][mem][12 ...]
        self.assertEqual(a0, [10, 11, b0, b1, 12, 0, 0, 0, 0, 0])
        self.assertEqual(slot0, [0, 1, 4])
        a1, slot1 = aug(1)  # one real already in the chunk -> real 0 seals immediately, then real 2
        self.assertEqual(a1, [10, b0, b1, 11, 12, b0, b1, 0, 0, 0])
        self.assertEqual(slot1, [0, 3, 4])
        # A per-row real_offset ([B]) is honored independently per row (insert_memory_tokens vmap).
        a_row, _, _ = insert_memory_tokens(
            jnp.asarray([[10, 11, 12], [10, 11, 12]], dtype=jnp.int32),
            mem_chunk_size=2,
            num_mem=2,
            max_seq_len=10,
            memory_id_base=MEMORY_ID_BASE,
            real_offset=jnp.asarray([0, 1], dtype=jnp.int32),
        )
        self.assertEqual(list(np.asarray(a_row)[0]), a0)  # row 0 uses offset 0
        self.assertEqual(list(np.asarray(a_row)[1]), a1)  # row 1 uses offset 1

    @parameterized.parameters((4, 2, 24), (8, 4, 48), (4, 2, 30), (16, 4, 64))
    def test_single_segment_global_stride(self, mem_chunk_size, num_mem, length):
        # segment_ids=None (eval/prefill): whole row is one segment. Reals all > memory ids, != pad.
        ids = np.arange(100, 100 + length, dtype=np.int32)
        self._check(ids, mem_chunk_size=mem_chunk_size, num_mem=num_mem, ref_bos_id=None)

    @parameterized.parameters((4, 2), (8, 2), (4, 3))
    def test_packed_multi_segment_bos_reset(self, mem_chunk_size, num_mem):
        # Two BOS-delimited docs in one row (+ pad tail): chunk grid resets per doc, pad unsealed.
        bos = 2
        doc_a = np.array([bos] + list(range(101, 111)), np.int32)  # 11 tokens
        doc_b = np.array([bos] + list(range(201, 208)), np.int32)  # 8 tokens
        ids = np.concatenate([doc_a, doc_b, np.zeros([3], np.int32)])  # + 3-token pad tail
        self._check(ids, mem_chunk_size=mem_chunk_size, num_mem=num_mem, ref_bos_id=bos)

    @parameterized.parameters((4, 2), (8, 4), (16, 4))
    def test_packed_random_multi_segment(self, mem_chunk_size, num_mem):
        # Docs packed via BOS(2) at RANDOM interior positions (varied lengths). Driven by the
        # derived segment_ids, must still match the oracle. Seeded for determinism.
        bos, length = 2, 64
        rng = np.random.default_rng(0)
        ids = rng.integers(100, 500, size=length).astype(np.int32)  # reals clear of pad/memory ids
        cuts = rng.choice(np.arange(1, length), size=5, replace=False)  # 5 interior doc boundaries
        ids[cuts] = bos  # position 0 already opens the first segment
        self._check(ids, mem_chunk_size=mem_chunk_size, num_mem=num_mem, ref_bos_id=bos)

    @parameterized.parameters((4, 2, 24), (8, 4, 48))
    def test_memory_positions_match_mask_slots(self, mem_chunk_size, num_mem, length):
        # Memory ids must land at EXACTLY the slots the mask treats as memory --
        # (pos % (C+m)) >= C -- tying placement to spmem_mask's predicate (independent cross-check).
        ids = np.arange(100, 100 + length, dtype=np.int32)
        aug_ids = _aug1(ids, mem_chunk_size=mem_chunk_size, num_mem=num_mem, max_seq_len=2 * length)
        aug_ids = aug_ids[0]
        period = mem_chunk_size + num_mem
        mem_ids = set(range(MEMORY_ID_BASE, MEMORY_ID_BASE + num_mem))
        n_aug = int(np.nonzero(aug_ids != 0)[0][-1]) + 1  # augmented content length
        actual = [p for p in range(n_aug) if int(aug_ids[p]) in mem_ids]
        expected = [p for p in range(n_aug) if (p % period) >= mem_chunk_size]
        self.assertEqual(actual, expected)

    def test_batched_matches_per_row(self):
        # The [B, L] vmap path equals stacking the per-row results (each row via _aug1).
        ids = np.stack([np.arange(100, 124, dtype=np.int32), np.arange(300, 324, dtype=np.int32)])
        batched = [
            np.asarray(x)
            for x in insert_memory_tokens(
                jnp.asarray(ids),
                mem_chunk_size=4,
                num_mem=2,
                max_seq_len=48,
                memory_id_base=MEMORY_ID_BASE,
            )
        ]
        for r in range(ids.shape[0]):
            row = _aug1(ids[r], mem_chunk_size=4, num_mem=2, max_seq_len=48)
            for b, s in zip(batched, row):
                np.testing.assert_array_equal(b[r], s)

    def test_segment_id_marks_pad_over_token_value(self):
        # Segment-based padding: a position with a NON-pad token value but segment_id == 0 is
        # padding (dropped), not content. The value fallback (segment_ids=None) keeps it. Guards
        # the segment_id!=0 pad gate that multistep decode (speculative / incremental-prefill /
        # multimodal) relies on when segment_ids disagrees with the token value.
        ids = [10, 11, 99]  # 99 is a non-pad token value
        aug_ids, aug_seg, _ = _aug1(
            ids, segment_ids=[1, 1, 0], mem_chunk_size=4, num_mem=2, max_seq_len=12
        )
        self.assertNotIn(99, aug_ids.tolist())  # pos 2 (seg 0) dropped despite its non-pad value
        self.assertEqual(int((aug_seg != 0).sum()), 2)  # only the 2 segment-valid reals
        # Value fallback: with no segment_ids, 99 (!= pad_id) is kept as content.
        aug_ids_val, aug_seg_val, _ = _aug1(
            ids, segment_ids=None, mem_chunk_size=4, num_mem=2, max_seq_len=12
        )
        self.assertIn(99, aug_ids_val.tolist())
        self.assertEqual(int((aug_seg_val != 0).sum()), 3)

    def test_segment_ids_rank_must_match_input_ids(self):
        # A 1-D [L] segment_ids with a [B, L] input would silently mis-map under the per-row
        # vmap -> reject with a clear error. Co-shaped [B, L] is fine.
        batched = jnp.asarray(np.stack([np.arange(100, 108), np.arange(200, 208)]).astype(np.int32))
        with self.assertRaisesRegex(ValueError, r"\[B, L\]"):
            insert_memory_tokens(
                batched,
                mem_chunk_size=4,
                num_mem=2,
                max_seq_len=24,
                memory_id_base=MEMORY_ID_BASE,
                segment_ids=jnp.ones(8, jnp.int32),
            )
        # Matching rank ([B, L] with [B, L]) is accepted.
        insert_memory_tokens(
            batched,
            mem_chunk_size=4,
            num_mem=2,
            max_seq_len=24,
            memory_id_base=MEMORY_ID_BASE,
            segment_ids=jnp.ones_like(batched),
        )

    def test_padded_prompt_augments_content_only(self):
        # Eval/prefill shape: content prefix + pad(0) tail. Only content is augmented (pad
        # never seals); must still match the reference.
        ids = np.zeros(40, np.int32)
        ids[:13] = np.arange(100, 113, dtype=np.int32)  # 13 content reals, rest pad
        self._check(ids, mem_chunk_size=4, num_mem=2, ref_bos_id=None)

    def test_padded_prefill_mem_segment_ids_zero_in_pad(self):
        # Evaler prefill passes segment_ids = [1..1 0..0]; the 1->0 change marks the pad start.
        # Memory tokens must inherit their real chunk's segment (1), and the pad region -- including
        # the phantom segment that 1->0 change would open -- must stay 0. So the augmented segment
        # is exactly 1 on content (reals + memory) and 0 on pad.
        ids = np.array([10, 11, 12, 13, 14, 15, 16, 17, 0, 0, 0, 0], np.int32)  # 8 reals + 4 pad
        seg = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], np.int32)
        aug_ids, aug_seg, _ = _aug1(
            ids, segment_ids=seg, mem_chunk_size=4, num_mem=2, max_seq_len=20
        )
        mem_ids = set(range(MEMORY_ID_BASE, MEMORY_ID_BASE + 2))
        self.assertTrue(any(int(v) in mem_ids for v in aug_ids), "expected memory tokens in stream")
        np.testing.assert_array_equal(aug_seg, (aug_ids != 0).astype(np.int32))


if __name__ == "__main__":
    absltest.main()

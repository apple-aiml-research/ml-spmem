#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Tests for spmem_geometry.py -- the stream layout, pool sizing and geometry validation."""

# pylint: disable=missing-function-docstring

from absl.testing import absltest, parameterized
from axlearn.common import test_utils

from spmem.spmem_geometry import (
    SpMemGeometry,
    augmented_seq_len,
    decompose_augmented_position,
)


class StreamLayoutTest(test_utils.TestCase):
    def test_decompose_augmented_position(self):
        # C=2, m=2 (period 4): reals at offsets 0, 1; memory at 2, 3.
        c, m = 2, 2
        cases = [  # (pos, chunk, offset, real_pos)
            (0, 0, 0, 0),
            (1, 0, 1, 1),
            (2, 0, 2, 2),  # memory; real_pos is meaningless
            (3, 0, 3, 3),
            (4, 1, 0, 2),
            (5, 1, 1, 3),
            (9, 2, 1, 5),
        ]
        for pos, chunk, offset, real_pos in cases:
            self.assertEqual(
                decompose_augmented_position(pos, mem_chunk_size=c, num_mem=m),
                (chunk, offset, real_pos),
                f"pos={pos}",
            )

    def test_augmented_seq_len(self):
        def f(real_len, *, c=256, m=4, align=8192):
            return augmented_seq_len(real_len, mem_chunk_size=c, num_mem=m, align=align)

        self.assertEqual(f(4096), 8192)
        self.assertEqual(f(8192), 16384)  # sized per input: a short eval is ~2x, not 17x
        self.assertEqual(f(131072), 139264)
        self.assertEqual(f(6, c=4, m=2, align=1), 9)
        # Never shorter than the content (reals + m per full chunk), so no real is dropped.
        for c, m in [(256, 4), (64, 1), (100, 4), (7, 3)]:
            for t in (1, 63, 64, 65, 99, 100, 101, 251, 4096, 131072):
                got = augmented_seq_len(t, mem_chunk_size=c, num_mem=m, align=1)
                self.assertGreaterEqual(got, t + m * (t // c), f"c={c} m={m} t={t}")


class SpMemGeometryTest(parameterized.TestCase):
    @parameterized.parameters(dict(c=64, m=4), dict(c=256, m=4), dict(c=4096, m=16))
    def test_unbounded_pool_holds_every_sealed_chunk(self, c, m):
        # An overflowing slot is dropped rather than raised, so this bound is the whole contract.
        geometry = SpMemGeometry(mem_chunk_size=c, num_mem=m, w0=4096)
        for real_len in (c, 8192, 131_072, 131_072 + c // 2):
            aug_len = augmented_seq_len(real_len, mem_chunk_size=c, num_mem=m, align=512)
            self.assertGreaterEqual(geometry.memory_slots(aug_len), m * (real_len // c))

    @parameterized.parameters(
        dict(c=128, m=4, window=131_072, chunks=1024),
        dict(c=64, m=4, window=8192, chunks=128),
        dict(c=4, m=1, window=10, chunks=3),  # a partial chunk still owns its group
    )
    def test_ring_is_whole_chunks(self, c, m, window, chunks):
        geometry = SpMemGeometry(mem_chunk_size=c, num_mem=m, w0=8, mem_window_size=window)
        self.assertEqual(geometry.mem_window_chunks, chunks)
        self.assertEqual(geometry.memory_slots(1 << 30), chunks * m)
        self.assertEqual(geometry.memory_slots(c + m), m)  # capped by what the stream can seal

    def test_tapered_pool_is_a_fixed_size(self):
        geometry = SpMemGeometry(mem_chunk_size=8, num_mem=1, w0=4096, mem_w0=4096, mem_w1=16384)
        self.assertEqual(geometry.memory_slots(1 << 20), geometry.mem_taper.slots)
        self.assertIsNone(geometry.mem_window_chunks)

    @parameterized.parameters(
        (dict(mem_chunk_size=0, num_mem=1, w0=8), "mem_chunk_size must be positive"),
        (dict(mem_chunk_size=4, num_mem=0, w0=8), r"num_mem must be in \[1, C=4\]"),
        # m > C would make the compression ratio C // m zero.
        (dict(mem_chunk_size=4, num_mem=5, w0=8), r"num_mem must be in \[1, C=4\]"),
        (dict(mem_chunk_size=4, num_mem=1, w0=6), "power-of-two"),
        (dict(mem_chunk_size=4, num_mem=1, w0=8, w1=1000), "reaches at most"),
        (dict(mem_chunk_size=4, num_mem=1, w0=8, mem_w0=8), "both mem_w0 and mem_w1"),
        (dict(mem_chunk_size=4, num_mem=1, w0=8, mem_w1=8), "both mem_w0 and mem_w1"),
        (
            dict(mem_chunk_size=4, num_mem=1, w0=8, mem_w0=8, mem_w1=8, mem_window_size=64),
            "both bound the memory pool",
        ),
        # A pool narrower than the real window summarizes nothing the reals do not hold.
        (dict(mem_chunk_size=4, num_mem=1, w0=8, mem_window_size=4), "must be >= w0"),
    )
    def test_rejects(self, kwargs, message):
        with self.assertRaisesRegex(ValueError, message):
            SpMemGeometry(**kwargs)

    def test_equal_geometries_hash_equal(self):
        """The mask memoizes on the geometry, so equal values must be one key."""
        a = SpMemGeometry(mem_chunk_size=8, num_mem=1, w0=4096)
        b = SpMemGeometry(mem_chunk_size=8, num_mem=1, w0=4096)
        self.assertEqual(hash(a), hash(b))


if __name__ == "__main__":
    absltest.main()

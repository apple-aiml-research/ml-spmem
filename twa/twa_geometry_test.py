#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Tests for `twa_geometry.py` — `(w0, w1)` in, cache size out.

This module is small and entirely arithmetic, which is exactly why it needs its own tests: it is the
SINGLE source of the cache size, so an off-by-one here is a buffer that disagrees with the mask about
how many keys exist, and nothing downstream would name that as the cause.

Three groups:
  * `TaperGeometryTest` — the derivations, against the shipped table stated outright, plus the two
    properties that decide what may be changed at inference and what may not.
  * `ValidateW0Test` / `MemoryTaperPairTest` — the two guards, including why the precondition is
    EVEN rather than power-of-two, and why the memory pair is checked while `levels == 0` is not.
  * `SuggestW0Test` / `DescribeTest` — the design-time helpers, which are otherwise unexercised.
"""

# pylint: disable=missing-function-docstring,no-self-use

from absl.testing import absltest, parameterized

from twa.twa_geometry import (
    TaperGeometry,
    TaperRule,
    anchor_index,
    suggest_w0,
    taper_geometry,
    validate_w0,
)


class TaperGeometryTest(parameterized.TestCase):
    @parameterized.named_parameters(
        # The shipped table, stated outright rather than recomputed: this is the arithmetic every
        # allocation reads, so it is worth having one place that says what the answers are.
        ("reals_128k", 2048, 131072, 6, 8192, 131072),
        ("reals_1m", 2048, 1 << 20, 9, 11264, 1 << 20),
        ("memory_128k_c8m1", 2048, 16384, 3, 5120, 16384),
        ("memory_1m_c8m1", 2048, 131072, 6, 8192, 131072),
        ("memory_w0_4k_128k", 4096, 16384, 2, 8192, 16384),
        ("memory_w0_4k_1m", 4096, 131072, 5, 14336, 131072),
    )
    def test_the_shipped_geometries(self, w0, w1, levels, slots, granted_w1):
        got = taper_geometry(w0=w0, w1=w1)
        self.assertEqual((got.levels, got.slots, got.w1), (levels, slots, granted_w1))

    def test_the_budget_is_linear_while_w1_is_exponential(self):
        """The whole reason for the scheme. Doubling the reach costs `w0/2` more slots, not twice as
        many, so a 8x longer receptive field is +3 levels rather than +8x budget."""
        base = taper_geometry(w0=2048, w1=131072)
        wider = taper_geometry(w0=2048, w1=131072 * 8)
        self.assertEqual(wider.w1, base.w1 * 8)
        self.assertEqual(wider.slots - base.slots, 3 * (2048 // 2))

    def test_doubling_w0_is_not_double_the_budget(self):
        """Cheap to get wrong when sizing an arm by hand: a larger dense window needs FEWER levels to
        reach the same age, so the two effects partly cancel. At C8M1's memory geometry, 2,048 ->
        4,096 costs 60% more slots rather than 100%, which is why the rungs are chosen by cost."""
        small = taper_geometry(w0=2048, w1=16384)
        large = taper_geometry(w0=4096, w1=16384)
        self.assertEqual((small.slots, large.slots), (5120, 8192))
        self.assertLess(large.slots, 2 * small.slots)

    @parameterized.parameters(2, 4, 6, 8, 2048)
    def test_slots_matches_the_level_by_level_sum(self, w0):
        """`slots` is a closed form; this is the sum it stands for. If they ever disagree the closed
        form is the one every allocation uses, so the buffer would be the wrong size silently."""
        geometry = taper_geometry(w0=w0, w1=w0 * 32)
        self.assertEqual(geometry.slots, w0 + geometry.levels * (w0 // 2))

    def test_a_receptive_field_inside_the_dense_window_has_no_taper(self):
        """`K = 0` is a legal geometry -- a plain dense window -- and NOTHING rejects it, on purpose.
        The real taper cannot reach it by accident because `w1` is a required argument everywhere,
        and on the MEMORY side it is the correct answer at a high compression ratio: `r = 64` makes a
        2,048-slot dense pool span 131,072 reals, so a guard on the level count would turn away the
        geometry that needs a taper least."""
        for w1 in (None, 512, 2048):
            geometry = taper_geometry(w0=2048, w1=w1 or 2048)
            self.assertEqual((geometry.levels, geometry.slots), (0, 2048))

    def test_w1_rounds_up_to_a_doubling(self):
        """The stored `w1` is the GRANTED receptive field, not the requested one: levels are whole
        doublings, so a request between two of them is covered by the next one up. Feeding the
        result back in is a no-op, which is what lets callers normalize a cache key on it."""
        geometry = taper_geometry(w0=1024, w1=5000)
        self.assertEqual((geometry.levels, geometry.w1), (3, 8192))
        self.assertEqual(taper_geometry(w0=1024, w1=geometry.w1), geometry)

    def test_max_seq_is_the_fallback_for_w1(self):
        """What a training config passes when it has no separate opinion: reach the whole sequence.
        On a MEMORY taper the caller divides by `r` first, since the pool's unit is summaries."""
        self.assertEqual(
            taper_geometry(w0=2048, max_seq=131072), taper_geometry(w0=2048, w1=131072)
        )
        with self.assertRaisesRegex(ValueError, "needs w1 or max_seq"):
            taper_geometry(w0=2048)

    def test_w1_is_ignored_when_max_seq_is_also_given(self):
        """`w1` wins: `max_seq` is only a fallback, so a caller that has both has already decided."""
        self.assertEqual(taper_geometry(w0=2048, w1=8192, max_seq=131072).w1, 8192)

    @parameterized.parameters(0, -1)
    def test_a_non_positive_receptive_field_is_rejected(self, w1):
        with self.assertRaisesRegex(ValueError, "w1 must be positive"):
            taper_geometry(w0=2048, w1=w1)

    def test_the_geometry_is_frozen(self):
        """It is passed into the mask closure and the cache scatter and read by both. A mutable one
        could be edited after the mask was memoized, leaving the two describing different schemes."""
        geometry = taper_geometry(w0=2048, w1=131072)
        with self.assertRaises(Exception):
            geometry.slots = 1  # pylint: disable=attribute-defined-outside-init


class MipmapGeometryTest(parameterized.TestCase):
    """MIPMAP derives all three numbers from `w0` alone; OCTAVE takes `w1` as an input."""

    @parameterized.named_parameters(
        # Stated outright: this is the budget every MIPMAP allocation reads. `2*w0 - 1` is the
        # rotating rings `w0 + w0/2 + ... + 2 = 2*w0 - 2` plus the anchor's one permanent slot.
        ("w0_1k", 1024, 10, 2047, 10240),
        ("w0_2k", 2048, 11, 4095, 22528),
        ("w0_4k", 4096, 12, 8191, 49152),
        ("w0_8k", 8192, 13, 16383, 106496),
    )
    def test_the_shipped_geometries(self, w0, levels, slots, reach):
        got = taper_geometry(w0=w0, rule=TaperRule.MIPMAP)
        self.assertEqual((got.levels, got.slots, got.w1), (levels, slots, reach))
        self.assertEqual(got.slots, 2 * w0 - 1)

    def test_w1_is_a_demand_and_max_seq_is_a_fact(self):
        """They are the same argument under OCTAVE and must not be under MIPMAP. `w1` says "the
        taper must still reach this", and MIPMAP's ceiling is `log2(w0)*w0`, so a demand past it is
        refused rather than quietly under-served. `max_seq` says "the sequence is this long", which
        can only ever remove rings no key reaches -- a sequence longer than the ceiling is not a cap
        at all, and raising there would refuse a caller who merely stated the context length."""
        with self.assertRaisesRegex(ValueError, "reaches at most 49152"):
            taper_geometry(w0=4096, w1=131072, rule=TaperRule.MIPMAP)
        stated = taper_geometry(w0=4096, max_seq=131072, rule=TaperRule.MIPMAP)
        self.assertEqual((stated.levels, stated.slots), (12, 8191))
        # And a SHORTER sequence does cap, which is the whole point on the memory pool.
        self.assertEqual(taper_geometry(w0=4096, max_seq=16384, rule=TaperRule.MIPMAP).slots, 7681)

    def test_the_two_rules_spend_the_same_budget_differently(self):
        """Why both exist. At ~8,192 slots MIPMAP buys a 4,096-key verbatim window reaching 49,152;
        OCTAVE buys 1,024 verbatim reaching the whole 128K. Neither dominates, so the arm picks."""
        mipmap = taper_geometry(w0=4096, rule=TaperRule.MIPMAP)
        octave = taper_geometry(w0=1024, w1=131072)
        self.assertEqual((mipmap.slots, octave.slots), (8191, 4608))
        self.assertLess(mipmap.w1, octave.w1)
        self.assertGreater(mipmap.w0, octave.w0)

    @parameterized.parameters(0, 1)
    def test_the_anchor_is_the_smallest_key_the_last_level_admits(self, coset):
        """`v_2(k + coset) >= log2(w0)`, i.e. `(w0 - coset) mod w0`: position 0 on an even coset,
        `w0 - 1` on an odd one. Shared with the cache so the two cannot anchor different keys."""
        w0 = 16
        k0 = anchor_index(w0.bit_length() - 1, coset)
        self.assertEqual(k0, 0 if coset == 0 else 15)
        # `v_2` the honest way: the low-bit trick reads 0 for key 0, which is exactly the key the
        # anchor exists to carry, so using it here would assert the bug instead of the property.
        v2 = float("inf") if k0 + coset == 0 else ((k0 + coset) & -(k0 + coset)).bit_length() - 1
        self.assertGreaterEqual(v2, w0.bit_length() - 1)
        # And it is the SMALLEST such key: everything below it falls short.
        for k in range(k0):
            self.assertLess(((k + coset) & -(k + coset)).bit_length() - 1, w0.bit_length() - 1)


class ValidateW0Test(parameterized.TestCase):
    @parameterized.parameters(2, 4, 6, 8, 100, 2048, 4096)
    def test_even_is_accepted_including_non_powers_of_two(self, w0):
        """EVEN is the real precondition, not power-of-two: level j holds exactly
        `2^(j-1) w0 / 2^j = w0/2` keys, an integer whenever `w0` is even, and the ring
        `(key >> j) mod (w0//2)` is a bijection only because that count is exact. Rejecting `w0=6`
        would turn a working geometry away, which is why the weaker guard is the correct one -- and
        `twa_mask_test` sweeps `w0=6` against the reference to keep that claim measured."""
        validate_w0(w0)  # must not raise
        self.assertEqual(taper_geometry(w0=w0, w1=w0 * 8).slots, w0 + 3 * (w0 // 2))

    @parameterized.parameters(1, 3, 7, 2047, 0, -2)
    def test_odd_or_below_two_is_rejected(self, w0):
        with self.assertRaisesRegex(ValueError, "w0 must be even"):
            validate_w0(w0)


class SuggestW0Test(parameterized.TestCase):
    @parameterized.named_parameters(
        ("reals_128k", 131072, 8192, 2048),
        ("memory_128k_slots", 16384, 8192, 4096),
    )
    def test_it_returns_the_documented_answer(self, rf, budget, want_w0):
        self.assertEqual(suggest_w0(receptive_field=rf, budget=budget).w0, want_w0)

    @parameterized.named_parameters(("a", 131072, 8192), ("b", 16384, 8192), ("c", 65536, 6000))
    def test_it_fits_reaches_and_is_maximal(self, rf, budget):
        """Three claims, and the third is the one a caller relies on: the NEXT even `w0` must not
        fit, or the helper left dense window on the table."""
        got = suggest_w0(receptive_field=rf, budget=budget)
        self.assertLessEqual(got.slots, budget)
        self.assertGreaterEqual(got.w1, rf)
        self.assertGreater(taper_geometry(w0=got.w0 + 2, w1=rf).slots, budget)

    def test_a_budget_no_geometry_fits_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no even w0 reaches"):
            suggest_w0(receptive_field=1 << 20, budget=4)

    def test_the_helper_is_not_how_w0_should_be_chosen_for_one_length(self):
        """Documented trap, asserted so it stays true: `w0` cannot be changed later, so the question
        is what it costs at the LONGEST length intended. The memory pool takes 2,048 rather than the
        4,096 this returns for 128K, because 4,096 needs 14,336 slots at 1M where 2,048 needs 8,192."""
        at_128k = suggest_w0(receptive_field=16384, budget=8192).w0
        self.assertEqual(at_128k, 4096)
        self.assertEqual(taper_geometry(w0=4096, w1=131072).slots, 14336)
        self.assertEqual(taper_geometry(w0=2048, w1=131072).slots, 8192)


class DescribeTest(absltest.TestCase):
    def test_it_states_the_slot_total_and_every_level(self):
        text = taper_geometry(w0=2048, w1=131072).describe()
        self.assertIn("8,192 slots", text)
        self.assertIn("levels=6", text)
        for level in range(7):
            self.assertIn(f"level {level}", text)
        self.assertIn("dense", text)

    def test_a_memory_taper_prints_both_units(self):
        """The requirement this helper exists for: the same `w0 = 2048` is 2,048 reals on the real
        cache and 2,048 SUMMARIES -- 16,384 reals -- on the pool, and a line that prints one number
        gets read as the other."""
        text = taper_geometry(w0=2048, w1=16384).describe(reals_per_slot=8)
        self.assertIn("2,048 slots (16K reals)", text)
        self.assertIn("128K reals", text)
        self.assertIn("5,120 slots", text)

    def test_the_real_side_prints_one_unit(self):
        """Without `reals_per_slot` there is only one unit, so adding a second would be noise -- and
        a `(... reals)` suffix on the real cache would read as if it were a conversion."""
        self.assertNotIn("reals", taper_geometry(w0=2048, w1=131072).describe())

    def test_describe_is_a_method_not_a_free_function(self):
        """So a caller holding the geometry can print it without importing anything else -- the point
        being that the printed slot count comes from the same object the buffer was sized by."""
        geometry = taper_geometry(w0=4, w1=32)
        self.assertIsInstance(geometry, TaperGeometry)
        self.assertIn("10 slots", geometry.describe())  # 4 + K*2 at K=3


if __name__ == "__main__":
    absltest.main()

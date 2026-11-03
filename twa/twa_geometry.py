#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Cache geometry for Tapered Window Attention (TWA): `(w0, w1)` in, cache size out.

The OCTAVE retention rule keeps a dense window and then halves the density once per DOUBLING of age, so
density is proportional to `1/age` and every octave of age carries the same amount of detail:

    level 0            ages [0, w0)                      w0 keys        dense
    level j            ages [2^(j-1)*w0, 2^j*w0)         w0/2 keys      rate 2^-j     (j = 1..K)
    age >= 2^K * w0    nothing

    K      = ceil(log2(w1_requested / w0))
    slots  = w0 + K * (w0 // 2)          <- the cache size, allocated once
    w1     = 2^K * w0                    <- the request, rounded UP to a whole doubling

The budget is LINEAR in the level count while the reach is EXPONENTIAL, which is the whole point:
`w0 = 2048, w1 = 128K` gives `K = 6`, **8,192 slots**, reach 131,072. Doubling the reach costs
`w0/2` more slots, not twice as many.

**`w0` is a training-time constant; `w1` is not.** Changing `w0` changes which keys are retained at
every age, near ages included, so a checkpoint trained at one `w0` cannot be evaluated at another.
Changing `w1` only appends outer levels -- the retained set inside the smaller reach is *identical*
-- so inference may raise it to widen the receptive field, or leave it and decode forever with the
window sliding. That asymmetry is why this module takes `w0` and `w1` as INPUTS and derives the
budget, rather than taking a budget and solving for `w0`: a budget is an output, and `w0` is a
contract. `suggest_w0` exists for the one-time design choice and is not a runtime path.

This module is separate from the mask and the cache because BOTH need it and the cache already
imports the mask -- deriving `slots` in two places is exactly how a mask and a buffer come to
disagree about how many keys exist.
"""

import dataclasses
import enum
from typing import Optional


class TaperRule(enum.Enum):
    """Which retention rule a taper follows. Both keep the last `w0` keys verbatim.

    OCTAVE halves the density once per DOUBLING of age: every level spans twice the ages of
    the one before and keeps the same `w0/2` keys, so the reach is exponential in the level
    count and `(w0, w1)` are independent -- `w1` is what you ask for.

    MIPMAP keeps the spans EQUAL at `w0` and halves the count instead: level j is a ring of
    `w0 >> j`, so the reach is only `log2(w0) * w0` and the budget at most `2*w0 - 1`. It buys a
    large verbatim window per slot where OCTAVE buys reach: at ~8,192 slots MIPMAP gives
    `w0 = 4096` reaching 48K, OCTAVE gives `w0 = 1024` reaching 128K. `w0` must be a power of two,
    or the rings do not sum as they should. The budget is "at most" because a `w1` or `max_seq`
    shorter than the ceiling DROPS the rings no key of that age can reach -- which is most of the
    pool's budget at a high compression ratio; see `taper_geometry`.
    """

    OCTAVE = "octave"
    MIPMAP = "mipmap"


@dataclasses.dataclass(frozen=True)
class TaperGeometry:
    """One taper's geometry. Build it with `taper_geometry`, never by hand.

    `w1` is the receptive field the taper ACTUALLY spans, which is the request rounded up to a whole
    doubling. It is deliberately the same name as the argument: the rounding is the only thing that
    happens to it, nothing downstream may branch on the pre-rounding value, and a second name for
    "the one that is really honoured" would invite code that reads the other one. Feeding a
    geometry's `w1` back in is a no-op, which is what lets callers normalize on it.

    Attributes:
        w0: dense window -- every key younger than this is kept verbatim. INPUT, and fixed at
            training time: changing it invalidates the checkpoint.
        w1: receptive field -- `2^K * w0`, the age at which a key dies however high its valuation.
            The request rounded UP; see above.
        levels: K = ceil(log2(w1 / w0)). Derived.
        slots: the cache size. Derived, and the ONLY source of it.
        rule: which retention rule the three numbers above were derived under; see `TaperRule`.
    """

    w0: int
    w1: int
    levels: int
    slots: int
    rule: TaperRule = TaperRule.OCTAVE

    def describe(self, *, reals_per_slot: Optional[int] = None) -> str:
        """A level-by-level table ending in the slot count, for a launch log or a review.

        Args:
            reals_per_slot: set it on a MEMORY taper (`r = C / m`) so both units are printed. The
                same `w0 = 2048` is 2,048 reals on the real cache and 2,048 SUMMARIES -- 16,384
                reals -- on the memory pool, and a table that prints one number is read as the
                wrong one.
        """
        unit = "" if reals_per_slot is None else f" slots ({_k(self.w0 * reals_per_slot)} reals)"
        rf = "" if reals_per_slot is None else f" ({_k(self.w1 * reals_per_slot)} reals)"
        head = (
            f"{self.rule.value:<6} w0 {self.w0:,}{unit} · w1 {_k(self.w1)}{rf}"
            f"  ->  {self.slots:,} slots   (levels={self.levels})"
        )
        rows = [f"  level 0   ages {0:>9,} - {self.w0 - 1:>9,}   {self.w0:>6,} keys  dense"]
        if self.rule is TaperRule.MIPMAP:
            # Spans EQUAL, counts halve -- the mirror of OCTAVE below. The last level is the
            # anchor's single permanent slot, not a ring, which is what makes it `2*w0 - 1`.
            for j in range(1, self.levels):
                lo, hi = j * self.w0, (j + 1) * self.w0 - 1
                rows.append(
                    f"  level {j}   ages {lo:>9,} - {hi:>9,}   {self.w0 >> j:>6,} keys  1/{1 << j}"
                )
            rows.append(
                f"  level {self.levels}   ages {self.levels * self.w0:>9,} - {'forever':>9}"
                f"   {1:>6,} keys  anchor"
            )
        else:
            for j in range(1, self.levels + 1):
                lo, hi = (self.w0 << (j - 1)), (self.w0 << j) - 1
                rows.append(
                    f"  level {j}   ages {lo:>9,} - {hi:>9,}   {self.w0 // 2:>6,} keys  1/{1 << j}"
                )
        rule = " " * 38 + "-" * 11
        return "\n".join([head, *rows, rule, " " * 38 + f"{self.slots:,} slots"])


def taper_geometry(
    *,
    w0: int,
    w1: Optional[int] = None,
    max_seq: Optional[int] = None,
    rule: TaperRule = TaperRule.OCTAVE,
) -> TaperGeometry:
    """Derive `(w1, levels, slots)` from the dense window and the requested receptive field.

    Args:
        w0: dense window, in the taper's own unit -- reals for the real cache, memory SLOTS for the
            pool. Must be even and at least 2; see `validate_w0`. Under MIPMAP it must also be a
            power of two.
        w1: receptive field, same unit. `None` falls back to `max_seq`, i.e. "reach the whole
            sequence", which is what a training config wants when it has no separate opinion.
            IGNORED under MIPMAP, whose reach is `log2(w0) * w0` and not a free parameter.
        max_seq: used only when `w1` is None. On a memory taper pass `max_seq // r`, since the
            pool's unit is summaries.
        rule: which retention rule; see `TaperRule`. OCTAVE is the default so that adding MIPMAP
            changes no existing config.

    Returns:
        A `TaperGeometry`. Under OCTAVE `w1` is the request rounded up to `2^K * w0`, so it can
        exceed the argument, and `w1 <= w0` gives `K = 0`: a plain dense window, no taper.

    Raises:
        ValueError: `w0` fails its rule's precondition, or neither `w1` nor `max_seq` is given
            under OCTAVE.
    """
    if rule is TaperRule.MIPMAP:
        # `w1` is not an input here: `levels` is the shift that divides by `w0` AND the index of the
        # last level, which coincide only for a power-of-two `w0`, and the rings then sum to
        # the rings. `w1` and `max_seq` do not widen it -- nothing can -- they only cap it.
        validate_power_of_two_w0(w0)
        # Cap the level count by what the stream can actually reach. Rotating level j covers ages
        # `[j*w0, (j+1)*w0)`, so a sequence of `max_seq` never populates a level above
        # `ceil(max_seq/w0) - 1` -- allocating those rings buys nothing and they are the EXPENSIVE
        # half of the budget to leave out. Dropping them does not change the retained set inside
        # `max_seq`, for the same reason raising OCTAVE's `w1` does not change it inside the smaller
        # reach: the levels removed are ones no key of that age exists to enter.
        #
        # It is worth real slots on the memory pool, whose unit is SUMMARIES: at C16M1 a 128K
        # sequence is only 8,192 memory tokens, so `mem_w0 = 4096` needs 2 levels rather than 12 and
        # the pool goes 8,191 slots -> 6,145. At C32M1 it halves.
        full = w0.bit_length() - 1
        # `w1` and `max_seq` are NOT the same input here, though OCTAVE can treat them alike.
        # `w1` is a DEMAND -- "the taper must still reach this" -- and MIPMAP has a hard ceiling of
        # `log2(w0) * w0`, so a demand past it is refused rather than quietly under-served. That is
        # the degradation this package exists to refuse; unlike OCTAVE there is no wider geometry to
        # grant. `max_seq` is a FACT -- "the sequence is this long" -- so it can only ever remove
        # rings no key reaches, and a sequence longer than the ceiling is simply not a cap at all.
        levels = full
        if w1 is not None:
            if w1 <= 0:
                raise ValueError(f"w1 must be positive, got {w1}")
            needed = -(-w1 // w0)  # ceil; rotating level j covers ages [j*w0, (j+1)*w0)
            if needed > full:
                raise ValueError(
                    f"the mipmap rule at w0={w0} reaches at most {full * w0}, short of the "
                    f"{w1} requested. Raise w0, or use the octave rule."
                )
            levels = max(1, needed)
        elif max_seq is not None:
            if max_seq <= 0:
                raise ValueError(f"max_seq must be positive, got {max_seq}")
            levels = max(1, min(full, -(-max_seq // w0)))
        reach = levels * w0
        # Rings 0..levels-1 sum to `2*w0 - 2*(w0 >> levels)`; the anchor takes one more. Uncapped
        # that is the familiar `2*w0 - 1`.
        slots = 2 * w0 - 2 * (w0 >> levels) + 1
        return TaperGeometry(w0=w0, w1=reach, levels=levels, slots=slots, rule=rule)
    validate_w0(w0)
    if w1 is None:
        if max_seq is None:
            raise ValueError("taper_geometry needs w1 or max_seq; both are None.")
        w1 = max_seq
    if w1 <= 0:
        raise ValueError(f"w1 must be positive, got {w1}")
    # K = ceil(log2(w1 / w0)), by integer doubling rather than a float log: `log2` is not exact for
    # every input, and an off-by-one here is a silently smaller cache that no test would name.
    levels, granted = 0, w0
    while granted < w1:
        granted *= 2
        levels += 1
    return TaperGeometry(w0=w0, w1=granted, levels=levels, slots=w0 + levels * (w0 // 2), rule=rule)


def validate_power_of_two_w0(w0: int) -> None:
    """Raise unless `w0` is a power of two and at least 2 -- the MIPMAP rule's precondition.

    `levels = log2(w0)` is both the index of the last level and the shift that divides by `w0`, and
    rotating level j holds `w0 >> j` slots which with the anchor sum to `2*w0 - 1`. All three break
    for a non-power-of-two: at `w0 = 6` the rings are 6 + 3 + 1 = 10, not 11. `w0 = 1` breaks
    separately -- there is no rotating level at all and the shift is undefined. OCTAVE needs only
    EVEN, which is why the two rules validate differently rather than sharing one guard.
    """
    if w0 < 2 or w0 & (w0 - 1):
        raise ValueError(f"the mipmap rule needs a power-of-two w0 >= 2, got {w0}")


def anchor_index(levels: int, coset_index: int) -> int:
    """MIPMAP's one permanent key: the smallest `k` with `v_2(k + coset) >= levels`.

    The last level is a single slot, and under MIPMAP it is PERMANENT rather than a ring of one --
    what a mipmap's coarsest level is, the 1x1 average that is always present. That key is
    `(w0 - coset) mod w0`: position 0 on an even coset, `w0 - 1` on an odd one. Everything else ages
    out at `levels * w0`.

    It costs nothing: that level held exactly one key either way, so what changes is only that the
    key it holds stops rotating. It is also the attention sink a window-based cache otherwise
    reintroduces by hand (StreamingLLM), free here.

    Keyed on `levels` rather than `w0` because `max_seq` can cap the level count below `log2(w0)`,
    and the anchor is whatever the LAST level admits. The two coincide when nothing is capped.

    Shared by the mask and the cache so the two cannot disagree about which key is anchored. OCTAVE
    needs no anchor at all -- `v_2(0)` is infinite, so key 0 survives to the reach on its own.
    """
    period = 1 << levels
    return (period - coset_index) % period


def validate_w0(w0: int) -> None:
    """Raise unless `w0` is even and at least 2.

    Level j holds the keys of age `[2^(j-1) w0, 2^j w0)` whose `v_2(key + coset) >= j`, and there
    are exactly `2^(j-1) w0 / 2^j = w0/2` of them -- an integer only for even `w0`, and the ring
    `(key >> j) mod (w0 // 2)` is a bijection onto its level only because that count is exact.
    Power-of-two is NOT required; even is the real precondition, and over-constraining would reject
    geometries that work -- verified against the reference at `w0 = 6`. Every swept value is a power
    of two anyway, so in practice this guards a typo.
    """
    if w0 < 2 or w0 % 2:
        raise ValueError(f"w0 must be even and >= 2 (level j holds w0/2 keys), got {w0}")


def suggest_w0(*, receptive_field: int, budget: int) -> TaperGeometry:
    """The LARGEST `w0` whose geometry reaches `receptive_field` within `budget` slots.

    A design-time helper, not a runtime path: `w0` is a training-time contract, so it is chosen once
    and then written into the config. Larger `w0` means a larger dense window and fewer levels, so
    maximizing it spends the budget on verbatim keys rather than on tail.

    Beware of choosing it for one context length. `w0` cannot be changed later, so the question is
    what it costs at the LONGEST length you intend, not at the current one -- the memory pool takes
    `w0 = 2048` rather than the 4,096 this helper returns for 128K, because 4,096 needs 14,336 slots
    at 1M while 2,048 needs 8,192.

    Raises:
        ValueError: no even `w0 >= 2` fits, i.e. the budget is smaller than any usable geometry.
    """
    best = None
    for w0 in range(2, budget + 1, 2):
        geometry = taper_geometry(w0=w0, w1=receptive_field)
        if geometry.slots <= budget:
            best = geometry
    if best is None:
        raise ValueError(f"no even w0 reaches {receptive_field} within {budget} slots")
    return best


def _k(n: int) -> str:
    """`131072` -> `128K`, `1048576` -> `1M`; anything else keeps its digits."""
    for size, suffix in ((1 << 20, "M"), (1 << 10, "K")):
        if n >= size and n % size == 0:
            return f"{n // size}{suffix}"
    return f"{n:,}"

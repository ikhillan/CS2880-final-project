"""Finite-catalogue diagnostics for simple random samples of distinct listings."""

from bisect import bisect_right
from functools import lru_cache
from math import comb, isfinite
from operator import index


@lru_cache(maxsize=256)
def _sample_count(population, draws):
    return comb(population, draws)


def degree_probs(L, D, n):
    """Return P(d=0), P(1<=d<=5), P(d>=6), P(d>=2), in this order.

    L is the number of distinct catalogue listings, D is the number feasible for
    this renter, and n is the number drawn uniformly without replacement. d is
    the count of feasible listings in the draw. The first three events partition
    the sample space. The last event describes the shortest degree at which a
    ceiling-of-40-percent budget can hide an option.

    Event counts, including complementary tails, use exact integers. Conversion
    to ordinary floating point occurs once per returned probability.
    """
    L, D, n = index(L), index(D), index(n)
    if L < 0 or not 0 <= D <= L or not 0 <= n <= L:
        raise ValueError("Require 0 <= D <= L and 0 <= n <= L.")
    low = max(0, n - (L - D))
    high = min(n, D)
    if low == high:
        return tuple(float(x) for x in (low == 0, 1 <= low <= 5, low >= 6, low >= 2))
    if low >= 6:
        return (0.0, 0.0, 1.0, 1.0)

    total = _sample_count(L, n)
    # At most six counts are needed. All other degrees enter through an
    # integer complement, which also retains a tiny but positive upper tail.
    counts = [0] * 6
    last = min(5, high)
    count = comb(D, low) * comb(L - D, n - low)
    for k in range(low, last + 1):
        counts[k] = count
        if k < last:
            count = count * (D - k) * (n - k) // ((k + 1) * (L - D - n + k + 1))

    middle = sum(counts[1:])
    return (counts[0] / total, middle / total, (total - counts[0] - middle) / total,
            (total - counts[0] - counts[1]) / total)


def budget_steps(eligible_rents_sorted, budget):
    """Return affordable_count, first_rent, sixth_rent, gap_to_first, gap_to_sixth.

    Supply monthly asking rents of distinct listings passing every non-price
    screen, sorted in nondecreasing order. Repeated prices on distinct units
    count separately. The caller validates the prices and sort order when
    constructing the catalogue. budget is a finite nonnegative monthly
    amount in the same dollars as the rents. The upper affordability boundary
    is inclusive. Each gap is the extra monthly budget needed to reach that
    threshold; absent first or sixth units give None for its rent and gap.
    """
    if not isfinite(budget) or budget < 0:
        raise ValueError("Monthly budget must be finite and nonnegative.")
    rents = eligible_rents_sorted
    first = rents[0] if len(rents) >= 1 else None
    sixth = rents[5] if len(rents) >= 6 else None
    return {
        "affordable_count": bisect_right(rents, budget),
        "first_rent": first,
        "sixth_rent": sixth,
        "gap_to_first": max(0, first - budget) if first is not None else None,
        "gap_to_sixth": max(0, sixth - budget) if sixth is not None else None,
    }

"""Compare the actual C++ branch-and-bound output with independent enumeration of every menu."""

import argparse
import itertools
import random
import subprocess


def verify(binary):
    rng = random.Random(20261113)
    expected, cases = [], []
    for row in range(2048):
        n = 5 + row % 7
        # Small support forces overlapping events and tied objectives. Other
        # cases exercise bit 63, redundant courses, and an attainable entitlement.
        support = [0, 1, 4, 8, 16, 64][row % 6]
        attainable = [rng.getrandbits(support) for _ in range(n)]
        if row % 5 == 0:
            attainable[-1] = attainable[0]
        mandatory = rng.getrandbits(support) if row % 2 else 0
        hits = [rng.randrange(5) for _ in range(n)]

        def objective(menu):
            covered = mandatory
            for edge in menu:
                covered |= attainable[edge]
            return (
                covered.bit_count(),
                sum(hits[i] for i in menu),
                sum(attainable[i].bit_count() for i in menu),
                tuple(-i for i in menu),
            )

        expected.append(max(itertools.combinations(range(n), 5), key=objective))
        cases.append(
            " ".join(
                map(
                    str,
                    [n, mandatory]
                    + [x for pair in zip(hits, attainable) for x in pair],
                )
            )
        )
    result = subprocess.run(
        [binary, "--check-rows"],
        input="\n".join(cases),
        text=True,
        capture_output=True,
        check=True,
    )
    actual = [tuple(map(int, row.split())) for row in result.stdout.splitlines()]
    if actual != expected:
        raise AssertionError("C++ solver differs from independent exact enumeration")
    print(f"Confirmed all {len(actual)} exact menus, including lexical tie breaking.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("binary")
    verify(p.parse_args().binary)

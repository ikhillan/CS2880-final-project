"""Test the two-grade suffix counterexample against the national Chilean solver."""
import argparse
import struct
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


def write_i32(stream, values):
    stream.write(struct.pack(f"<{len(values)}i", *values))


def run_case(root, executable, siblings, extend, family_priority):
    # Schools x, y, u have program IDs 0, 1, 2 in each grade.
    # Highest-grade H and S have courses 5 (u), 3 (x). Juniors use 0 (x), 1 (y).
    n = 2 + 2 * siblings
    grades = [1, 1] + [0] * (n - 2)
    reports = [[5], [5, 3] if extend else [5]]
    reports += [[0] for _ in range(siblings)]       # L pupils, no backup
    reports += [[0, 1] for _ in range(siblings)]    # B pupils, siblings of S
    capacity = [siblings, siblings, 0, 1, 0, 1]
    edges = [j for row in reports for j in row]
    # Lower ticket wins in the ordinary tier. H outranks S and every L beats every B.
    tickets = [i + 1 for i, row in enumerate(reports) for _ in row]
    offsets = [0]
    for row in reports:
        offsets.append(offsets[-1] + len(row))
    links = [(1, i, 1, 0) for i in range(2 + siblings, n)] if family_priority else []
    with (root / "market.bin").open("wb") as data:
        for values in ([n, 6, len(edges)], offsets, edges, tickets, capacity, [-1] * n,
                       [0] * n, grades, [-1] * n, [-1] * n, list(map(len, reports))):
            write_i32(data, values)
        data.write(bytes(16 * len(edges)))   # Two unused arrays of double predictors
    with (root / "market.side").open("wb") as data:
        write_i32(data, [n, 6, len(edges), len(links)])
        write_i32(data, [0, 1, 2] * 2)
        write_i32(data, [0, 0, 0, 1, 1, 1])
        for count in capacity:
            write_i32(data, [count, 0, 0, 0, 0])
        data.write(bytes(n + len(edges)))   # No special designations or edge flags
        for values in (tickets, [-1] * len(edges), [-1] * len(edges)):
            write_i32(data, values)
        for link in links:
            write_i32(data, link)
    subprocess.run([executable, root / "market.bin", root / "market.side", root / "prediction"], check=True)
    result = list(struct.unpack(f"<{n}i", (root / "prediction").read_bytes()))
    if family_priority and extend:
        expected = [5, 3] + [-1] * siblings + [0] * siblings
    else:
        expected = [5, 3 if extend else -1] + [0] * siblings + [1] * siblings
    assert result == expected, (result, expected)
    return sum(j >= 0 for j in result)


def main(publication):
    inst = publication / "two-studies/institutional"
    school = publication / "research/observed/school"
    with TemporaryDirectory() as name:
        root = Path(name)
        solver = root / "extract"
        subprocess.run(["c++", "-std=c++17", "-O2", f"-I{school}", str(inst / "export_programs.cpp"),
                        "-o", str(solver)], check=True)
        for k in (1, 2, 3, 8, 25):
            for family_priority in (False, True):
                old = run_case(root, solver, k, False, family_priority)
                new = run_case(root, solver, k, True, family_priority)
                assert new - old == (1 - k if family_priority else 1)
        print("Verified 20 institutional allocations with and without dynamic sibling priority.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    main(parser.parse_args().publication.resolve())

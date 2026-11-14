"""Check all supported edges of small capacitated assignment problems."""
import itertools
from collections import deque


def residual_support(rows, capacities, baseline):
    edges = [set() for _ in capacities]
    for row, current in zip(rows, baseline):
        edges[current].update(row)
    # Fill reachability independently by BFS from every course rather than
    # by the SCC and reverse search used in the national implementation.
    reach = []
    for j in range(len(capacities)):
        seen = {j}
        pending = deque([j])
        while pending:
            for k in edges[pending.popleft()]:
                if k not in seen:
                    pending.append(k)
                    seen.add(k)
        reach.append(seen)
    slack = {j for j,c in enumerate(capacities) if baseline.count(j) < c}
    return [{j for j in row if old in reach[j] or reach[j] & slack}
            for row, old in zip(rows, baseline)]


def main():
    perms = itertools.product
    choices = [tuple(j for j in range(3) if bits & 1<<j) for bits in range(1,8)]
    comparisons = supported_edges = 0
    for rows in perms(choices,repeat=3):
        options=list(perms(*rows))
        for caps in perms((1,2),repeat=3):
            feasible = [dest for dest in options if all(dest.count(j)<=c for j,c in enumerate(caps))]
            if not feasible:continue
            possible = [set(x) for x in zip(*feasible)]
            for baseline in feasible:
                assert residual_support(rows,caps,baseline)==possible
                comparisons += 1
                supported_edges+=sum(map(len,possible))
    print(f"Verified {comparisons} baseline-specific residual graphs and {supported_edges} supported edges.")

if __name__=="__main__":
    main()

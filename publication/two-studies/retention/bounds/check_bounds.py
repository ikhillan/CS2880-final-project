"""Check national hindsight bounds and full-support certificates from aggregates."""
import argparse
import csv
from pathlib import Path

def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))

def check(directory):
    combined = []
    for y, expected_candidates, strict in [(2022, 1425, 96), (2023, 1633, 111), (2024, 1765, 107)]:
        name = str(y)
        (row,) = rows(directory / (name + "-oracle.csv"))
        q = {k: int(v) for k,v in row.items()}
        draws = rows(directory / (name + "-oracle-draws.csv"))
        assert q["draws"] == 1024
        assert q["rowwise_upper"] == expected_candidates and q["full_contract_lower"] == strict
        assert strict <= q["retention_lower"] <= q["forced_rejection_upper"] <= expected_candidates
        assert len(draws) == 1024 and [int(r["draw"]) for r in draws] == list(range(1,1025))
        assert all(r["year"] == name and int(r["batch"]) == 513 for r in draws)
        for column in ("exits","entrants","switches"):
            assert q[column] == sum(int(r[column]) for r in draws)
        assert q["exits"] == 0
        pruning=rows(directory/(name+"-oracle-pruning.csv"))
        assert q["forced_rejection_upper"]-sum(int(r["edits_removed"]) for r in pruning)==q["retention_lower"]
        combined.append(q)
    for col in ("full_contract_lower","retention_lower","forced_rejection_upper","rowwise_upper","exits","switches"):
        print(col,sum(x[col] for x in combined))
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    check(parser.parse_args().directory)

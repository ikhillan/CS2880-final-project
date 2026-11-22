"""Reconcile the enrollment bounds, placement comparisons, and displacement counts."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_rows(relative_path):
    with (ROOT / relative_path).open(newline="") as source:
        return list(csv.DictReader(source))


def enrollment_bounds():
    labels = {
        r["year"]: r
        for r in read_rows("field/enrollment-target.csv")
        if (r["group"], r["population"], r["target"], r["resolution"])
        == ("all", "register_linked", "enrollment", "school")
    }
    lower = upper = 0
    bounds = read_rows("field/enrollment-bounds.csv")
    assert len(bounds) == len(labels) == 4
    checks = {r["year"]: r for r in read_rows("field/missing-bounds-check.csv")}
    for r in bounds:
        for key in (
            "linked",
            "lower_gain",
            "upper_gain",
            "unlinked_blind_exclusive",
            "unlinked_rank_exclusive",
        ):
            assert r[key] == checks[r["year"]][key]
        label = labels[r["year"]]
        for a, b in [
            ("full", "eligible_graph_hits"),
            ("rank", "rank_hits"),
            ("blind", "blind_hits"),
            ("linked", "applicants"),
            ("gained", "gained_hits"),
            ("lost", "lost_hits"),
        ]:
            assert int(r[a]) == int(label[b])
        delta = int(label["net_hits"])
        lo, hi = int(r["lower_gain"]), int(r["upper_gain"])
        assert 0 < lo == delta - int(r["unlinked_blind_exclusive"])
        assert hi == delta + int(r["unlinked_rank_exclusive"])
        lower += lo
        upper += hi
    assert (lower, upper) == (878, 8395)


def school_orders():
    summary = read_rows("chile/school-tail-summary.csv")
    paths = {r["year"]: r for r in read_rows("chile/school-order-witnesses.csv")}
    for response in ("binding", "in_order", "last", "all_last", "all_original"):
        years = [r for r in summary if r["year"] != "0" and r["response"] == response]
        (pooled,) = [
            r for r in summary if r["year"] == "0" and r["response"] == response
        ]
        for key in ("n", "history", "placed", "first", "gains", "losses"):
            assert sum(int(r[key]) for r in years) == int(pooled[key])
        for r in years:
            assert int(r["placed"]) == int(r["gains"]) - int(r["losses"])
            if response == "in_order":
                assert int(r["first"]) < 0
            if response == "last":
                assert int(r["first"]) > 0
            if response == "all_last":
                w = paths[r["year"]]
                assert int(r["same_set"]) == int(r["n"]) and int(r["history"]) == 0
                for count, endpoints in (
                    ("losses", "distinct_loss_witnesses"),
                    ("gains", "distinct_gain_witnesses"),
                ):
                    assert (
                        int(r[count])
                        == int(w[count])
                        == int(w[endpoints])
                        <= int(w["reversals"])
                    )
    checks = json.loads((ROOT / "chile/school-recovery-verification.json").read_text())
    assert (
        sum(r["applicant_assignments_checked_via_vectors"] for r in checks) == 32714384
    )
    assert sum(r["policies_verified"] for r in checks) == 64
    rent = read_rows("nyc/nyc-full-tail-witnesses.csv")
    assert len(rent) == 480
    assert sum(int(r["source_gains"]) + int(r["source_losses"]) for r in rent) == 1231


if __name__ == "__main__":
    enrollment_bounds()
    school_orders()
    print(
        "Verified missing-enrollment bounds, school responses and displacement counts."
    )

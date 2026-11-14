"""Verify the published panel, denominators and independent receipts using aggregate files only."""

import argparse
import csv
import json
from pathlib import Path


def read(root, name):
    with (root / name).open() as f:
        return list(csv.DictReader(f))


def require_equal(value, target, label):
    if value != target:
        raise ValueError(f"{label}: got {value}, expected {target}")


def run(root):
    base = {
        (int(r["year"]), r["group"]): r for r in read(root, "round-link-counts.csv")
    }
    register = {
        (int(r["year"]), r["historical_stratum"]): r
        for r in read(root, "round-link-register.csv")
    }
    strata = {
        (int(r["year"]), r["historical_stratum"], r["menu_transition"]): r
        for r in read(root, "round-link-policy.csv")
    }
    labels = {
        (int(r["year"]), r["group"], r["population"], r["target"], r["resolution"]): r
        for r in read(root, "enrollment-target.csv")
    }
    receipts = {
        r["year"]: r for r in json.loads((root / "round-link-checks.json").read_text())
    }
    target_receipts = {
        r["year"]: r
        for r in json.loads((root / "enrollment-target-checks.json").read_text())
    }
    fields = [
        "regular",
        "returns_same_grade",
        "any_new_school",
        "offered_second_explicit_new_school",
    ]
    numbers = [
        [43077, 21371, 20858, 15095, 8192, 1007, 993, -405],
        [37757, 18537, 18116, 13895, 7923, 1198, 1182, -482],
        [30486, 14599, 14532, 12340, 7304, 1385, 1474, -434],
        [30025, 16041, 15730, 13071, 7280, 1048, 1170, -288],
    ]
    for i, yr in enumerate(range(2022, 2026)):
        r = base[yr, "regular_never_offered"]
        reg = register[yr, "never_offered"]
        checked = receipts[yr]
        for k, v in checked["counts"].items():
            require_equal(int(r[k]), v, (yr, k))
        for k, v in checked["verified_register_columns"].items():
            require_equal(int(reg[k]), v, (yr, k))
        for pop in ("all", "never_offered", "other_regular"):
            for section in ("historical_offer", "enrollment"):
                for resolution in ("school", "school_type_grade"):
                    for frame in ("all_originals", "register_linked"):
                        lab = labels[yr, pop, frame, section, resolution]

                        def val(c):
                            return int(lab[c])

                        top = val("eligible_graph_hits")
                        if (
                            not 0
                            <= min(val("blind_hits"), val("rank_hits"))
                            <= max(val("blind_hits"), val("rank_hits"))
                            <= top
                            <= val("target_observed")
                            <= val("applicants")
                        ):
                            raise ValueError(
                                (
                                    yr,
                                    pop,
                                    frame,
                                    section,
                                    resolution,
                                    "ceiling exceeded",
                                )
                            )
                        require_equal(
                            val("rank_hits") - val("blind_hits"),
                            val("net_hits"),
                            (yr, "coverage delta"),
                        )
                        require_equal(
                            val("gained_hits") - val("lost_hits"),
                            val("net_hits"),
                            (yr, "gross delta"),
                        )
        enrolled = labels[yr, "all", "register_linked", "enrollment", "school"]
        official = labels[yr, "all", "register_linked", "historical_offer", "school"]
        require_equal(
            int(enrolled["applicants"]),
            target_receipts[yr]["register_linked_applicants"],
            (yr, "linked enrollment records"),
        )
        require_equal(
            target_receipts[yr]["checked_linked_population_rows"],
            12,
            (yr, "source check"),
        )
        gain, loss = [
            int(strata[yr, "all", s]["register_any_original_applicant"])
            for s in ("gained", "lost")
        ]
        actual = [int(r[c]) for c in fields] + [
            int(reg["registered_in_new_offered_school"]),
            int(enrolled["net_hits"]),
            int(official["net_hits"]),
            gain - loss,
        ]
        require_equal(actual, numbers[i], yr)
    print(
        "Verified the aggregate tables, yearly public figures and two source-level check receipts across the four application years."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path(__file__).resolve().parent)
    run(p.parse_args().results)

"""Cross-check same-grade actual behavior against archived round and register counts."""

from pathlib import Path
import argparse
import csv


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main(a):
    field = a.publication / "two-studies/behavior/field"
    old = rows(field / "round-link-counts.csv")
    observed = rows(field / "round-link-register.csv")
    for year in range(2022, 2026):
        curr = rows(a.results / f"relisted-{year}.csv")
        for raw in curr:
            r = {k: (int(v) if k != "group" else v) for k, v in raw.items()}
            hist = "all" if r["group"] == "all" else "never_offered"
            g = "regular_all" if hist == "all" else "regular_never_offered"
            (partner,) = [
                row for row in old if row["group"] == g and int(row["year"]) == year
            ]
            (reference,) = [
                row
                for row in observed
                if row["historical_stratum"] == hist
                and row["menu_transition"] == "all"
                and int(row["year"]) == year
            ]
            for k, v in [
                ("applicants", "regular"),
                ("returned_same_grade_pupils", "returns_same_grade"),
                ("named_any_pupils", "with_manual"),
                ("outside_original_course_pupils", "any_new_course"),
                ("outside_original_school_pupils", "any_new_school"),
            ]:
                assert r[k] == int(partner[v])
            assert r["awarded_then_enrolled_at_new_school_pupils"] == int(
                reference["registered_in_new_offered_school"]
            )
            assert r["named_any_choices"] == sum(
                r[k + "_choices"]
                for k in (
                    "repeated_eligible",
                    "repeated_below_continuity",
                    "outside_original_course",
                )
            )
            for unit in ("pupils", "choices"):

                def v(k):
                    return r[k + "_" + unit]

                assert v("omitted_by_either") <= v("repeated_eligible")
                assert (
                    max(v("omitted_by_blind"), v("omitted_by_rank"))
                    <= v("omitted_by_either")
                    <= v("omitted_by_blind") + v("omitted_by_rank")
                )
                assert v("awarded_then_enrolled_at_omitted") <= v("omitted_by_either")
                assert (
                    v("repeated_open_omission")
                    <= v("returner_with_open_omission")
                    <= v("returner_with_omission")
                )
                assert (
                    v("repeated_open_omission")
                    <= v("omitted_by_either")
                    <= v("returner_with_omission")
                )
    print(
        "Checked the C2 edge partition, offered-and-enrolled linkage, frozen eligible graph and opportunity denominators in four cohorts."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("publication", "results"):
        p.add_argument("--" + k, type=Path, required=True)
    main(p.parse_args())

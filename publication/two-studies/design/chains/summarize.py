"""Verify published flow results and print per-cohort and per-simulated-market summaries."""

from collections import defaultdict
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "all": {
        "primary/uniform": (215, 641, 1006),
        "primary/haiku": (683, 983, 1353),
        "enriched/uniform": (518, 1646, 2102),
        "enriched/haiku": (1926, 2493, 2844),
        "chile/blind": (0, 53, 133),
        "chile/rank": (409, 475, 649),
    },
    "top5": {
        "primary/uniform": (35, 411, 629),
        "primary/haiku": (204, 610, 843),
        "enriched/uniform": (49, 499, 577),
        "enriched/haiku": (370, 804, 901),
        "chile/blind": (0, 50, 127),
        "chile/rank": (1, 2, 4),
    },
}


def read(study, restriction):
    suffix = "" if restriction == "all" else "-top5"
    with (ROOT / f"{study}{suffix}.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            for key in row.keys() - {"policy", "design", "borough", "month"}:
                row[key] = int(row[key])
            assert 0 <= row["direct"] <= row["chain"] <= row["upper"]
            assert row["applicants"] >= row["placed"] + row["upper"]
            assert row["tight"] == (row["chain"] == row["upper"])
            yield row


def main():
    summaries = {}
    for restriction, expected in EXPECTED.items():
        groups = defaultdict(list)
        pairs = defaultdict(dict)
        for row in read("housing", restriction):
            design = row["design"]
            groups[f"{design}/{row['policy']}"].append(row)
            key = (design, row["borough"], row["month"], row["seed"])
            assert row["policy"] not in pairs[key]
            pairs[key][row["policy"]] = row
        for row in read("chile", restriction):
            groups[f"chile/{row['policy']}"].append(row)
        assert set(groups) == set(expected)
        summary = {}
        for group, rows in groups.items():
            assert len(rows) == (4 if group.startswith("chile") else 240)
            if group.startswith("chile"):
                assert sorted(r["year"] for r in rows) == [2022, 2023, 2024, 2025]
            else:
                assert all(r["applicants"] == 64 for r in rows)
                assert len({(r["month"], r["borough"]) for r in rows}) == 12
            totals = {
                key: sum(r[key] for r in rows)
                for key in (
                    "applicants",
                    "placed",
                    "direct",
                    "chain",
                    "upper",
                    "edited",
                    "paths_with_incumbents",
                    "movers_da",
                    "entrants_at_witness_option",
                    "tight",
                )
            }
            assert (
                tuple(totals[k] for k in ("direct", "chain", "upper"))
                == expected[group]
            )
            assert totals["applicants"] == (
                2044649 if group.startswith("chile") else 15360
            )
            totals["markets_or_cohorts"] = len(rows)
            totals["chain_per_market_or_cohort"] = totals["chain"] / len(rows)
            totals["chain_share_of_applicants"] = totals["chain"] / totals["applicants"]
            totals["longest_path"] = max(r["longest_path"] for r in rows)
            summary[group] = totals
        for design, expected_count in zip(
            ("primary", "enriched"), (226, 240) if restriction == "all" else (172, 117)
        ):
            members = [policies for key, policies in pairs.items() if key[0] == design]
            assert all(set(p) == {"haiku", "uniform"} for p in members)
            catches = sum(
                p["haiku"]["placed"] + p["haiku"]["chain"] >= p["uniform"]["placed"]
                for p in members
            )
            assert catches == expected_count
            summary[f"{design}/haiku"]["markets_reaching_unchanged_uniform"] = catches
        deficit = summary["chile/blind"]["placed"] - summary["chile/rank"]["placed"]
        assert deficit == 1787
        summary["chile/rank"]["shortfall_even_at_ceiling"] = (
            deficit - summary["chile/rank"]["upper"]
        )
        summaries[restriction] = summary
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

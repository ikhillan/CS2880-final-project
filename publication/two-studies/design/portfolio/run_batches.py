"""Run independent, identically budgeted prospective menu comparisons and reproduce the frozen baseline."""

import argparse
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run(args):
    here = Path(__file__).resolve().parent
    private = args.private
    binary = private / f"{args.kernel}-portfolio-replay"
    src = args.publication / "two-studies/audit/chile"
    simulator = args.publication / "research/observed/school"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O3",
            f"-I{src}",
            f"-I{simulator}",
            here / "school_portfolio.cpp",
            "-o",
            binary,
        ],
        check=True,
    )
    archived = {}
    for filename in [
        "school-simulated-menu-batches.csv",
        "school-simulated-menu-2024-batches.csv",
    ]:
        for r in csv.DictReader((src / filename).open()):
            if r["policy"] == "simulated" and r["lottery_draws"] != "64":
                continue
            pol = {"simulated": "mc"}.get(r["policy"], r["policy"])
            archived[int(r["year"]), int(r["batch"]), pol] = r

    with (
        args.publication / "two-studies/behavior/field/offer-concordance.csv"
    ).open() as stream:
        register = {(int(r["year"]), r["stratum"]): r for r in csv.DictReader(stream)}

    def one(job):
        year, batch = job
        filename = private / f"{args.kernel}-portfolio-{year}-{batch}.csv"
        numeric = args.early_numeric if year < 2024 else args.late_numeric
        subprocess.run(
            [binary, numeric, private, str(year), str(batch), filename, args.kernel],
            check=True,
        )
        with filename.open() as stream:
            rows = list(csv.DictReader(stream))
        for r in rows:
            if r["policy"] in ["blind", "rank"] or (
                args.kernel == "independent" and r["policy"] == "mc"
            ):
                original = archived[year, batch, r["policy"]]
                for key in [
                    "applicants",
                    "placed",
                    "historical_shown",
                    "full_shown",
                    "full_named",
                    "submitted_first",
                ]:
                    assert original[key] == r[key], (
                        year,
                        batch,
                        r["policy"],
                        key,
                        original[key],
                        r[key],
                    )
            if r["policy"] in ["blind", "rank"]:
                linked = register[year, "all_linked"]
                no_offer = register[year, "no_initial_offer"]
                assert r["linked"] == linked["people"]
                assert r["april_shown"] == linked[r["policy"] + "_hits"]
                assert r["linked_placed"] == linked[r["policy"] + "_placed"]
                assert r["no_official_april_shown"] == no_offer[r["policy"] + "_hits"]
        mc, ablation, coverage = rows[:3]
        assert ablation["simulation_partner_hits"] == mc["simulation_partner_hits"]
        assert int(coverage["simulation_coverage"]) >= int(
            ablation["simulation_coverage"]
        )
        for r in rows:
            assert int(r["placed"]) - int(mc["placed"]) == int(r["gained_vs_mc"]) - int(
                r["lost_vs_mc"]
            )
        print(
            "replicated",
            year,
            batch,
            "placed delta",
            int(coverage["placed"]) - int(mc["placed"]),
            flush=True,
        )
        return rows

    jobs = [(y, b) for b in range(8) for y in [2022, 2023, 2024]]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, jobs))
    with args.out.open("w", newline="") as stream:
        out = csv.DictWriter(stream, fieldnames=results[0][0].keys(), lineterminator="\n")
        out.writeheader()
        for rows in results:
            out.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["publication", "early-numeric", "late-numeric", "private", "out"]:
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--kernel", choices=["independent", "common"], default="independent")
    run(p.parse_args())

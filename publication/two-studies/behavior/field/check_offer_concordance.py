"""Independently reconstruct the three menus' school hits using the release's C1 rows."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from check_round_link import records
from enrollment_bounds import members, table
from link_register import enrollment_register
from round_link import RELEASES


def check_year(year, archive, numeric, registry, published, first_partners):
    offers = {
        int(r["mrun"]): int(r["rbd_admitido"]) if r["rbd_admitido"].strip() else -1
        for r in records(archive, "D1")
    }
    registers, _ = enrollment_register(registry, pd.Index(offers), year + 1)
    april = registers.RBD.to_dict()
    choices = table(
        archive,
        members(archive),
        "C1",
        ["mrun", "rbd", "preferencia_postulante", "prioridad_matriculado"],
    )
    choices = choices.sort_values(["mrun", "preferencia_postulante"])
    cutoff = (
        choices[choices.prioridad_matriculado.eq(1)]
        .set_index("mrun")
        .preferencia_postulante
    )
    assert cutoff.index.is_unique
    limit = choices.mrun.map(cutoff)
    choices = choices[limit.isna() | choices.preferencia_postulante.le(limit)]
    rank = choices.groupby("mrun", sort=False).cumcount()
    menus = {"first": (rank.lt(5) | choices.prioridad_matriculado.eq(1)).to_numpy()}
    for tag in ("blind", "rank"):
        menus[tag] = np.fromfile(numeric / f"{year}-{tag}.mask", np.uint8).astype(bool)
    hit = choices.rbd.eq(choices.mrun.map(april))
    known = set(april)
    strata = {
        "enrolled_at_initial_offer": {i for i in known if offers[i] == april[i]},
        "enrolled_elsewhere": {
            i for i in known if offers[i] >= 0 and offers[i] != april[i]
        },
        "no_initial_offer": {i for i in known if offers[i] < 0},
        "all_linked": known,
    }
    found = {tag: set(choices.mrun[hit & mask]) for tag, mask in menus.items()}
    full = set(choices.mrun[hit])
    roster = np.array(sorted(offers))
    partners = {}
    for tag in ("first", "rank", "blind"):
        directory = first_partners if tag == "first" else numeric
        vector = np.fromfile(directory / f"{year}-{tag}.partners", dtype="<i4")
        assert len(vector) == len(roster)
        partners[tag] = set(roster[vector >= 0])
    for label, people in strata.items():
        row = published.loc[year, label]
        assert row.people == len(people) and row.eligible_hits == len(people & full)
        for tag in ("first", "rank", "blind"):
            assert row[tag + "_hits"] == len(found[tag] & people)
            assert row[tag + "_placed"] == len(partners[tag] & people)
        for other in ("first", "blind"):
            assert row[f"rank_only_vs_{other}_hits"] == len(
                (found["rank"] - found[other]) & people
            )
            assert row[f"{other}_only_vs_rank_hits"] == len(
                (found[other] - found["rank"]) & people
            )
    print(
        f"{year}: coverage and placement counts agree for all three menus in every stratum.",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "early-archives",
        "late-archives",
        "early-numeric",
        "late-numeric",
        "enrollments",
        "results",
        "first-partners",
    ):
        p.add_argument("--" + name, required=True, type=Path)
    args = p.parse_args()
    values = pd.read_csv(args.results).set_index(["year", "stratum"])
    for year, archive in RELEASES.items():
        root = args.early_archives if year < 2024 else args.late_archives
        numeric = args.early_numeric if year < 2024 else args.late_numeric
        check_year(
            year,
            root / archive,
            numeric,
            args.enrollments / f"matricula-{year + 1}.rar",
            values,
            args.first_partners,
        )

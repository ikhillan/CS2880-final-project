"""Rebuild school-level hit indicators from release rows and the two published-menu masks."""

import argparse
from pathlib import Path
import subprocess
import io

import numpy as np
import pandas as pd


def members(archive):
    return subprocess.check_output(
        ["bsdtar", "-tf", str(archive)], text=True
    ).splitlines()


def table(archive, pool, prefix, fields):
    (name,) = [
        k for k in pool if Path(k).name.startswith(prefix + "_") and k.endswith(".csv")
    ]
    content = subprocess.check_output(["bsdtar", "-xOf", str(archive), name])
    return pd.read_csv(
        io.BytesIO(content), sep=";", encoding="utf-8-sig", usecols=fields
    )


def study(year, source, numeric, registry):
    paths = members(source)
    b = table(source, paths, "B1", ["mrun"]).sort_values("mrun").reset_index(drop=True)
    c = table(
        source,
        paths,
        "C1",
        ["mrun", "rbd", "cod_curso", "preferencia_postulante", "prioridad_matriculado"],
    )
    cut = c[c.prioridad_matriculado == 1].set_index("mrun").preferencia_postulante
    threshold = c.mrun.map(cut)
    c = c[threshold.isna() | (c.preferencia_postulante <= threshold)].sort_values(
        ["mrun", "preferencia_postulante"]
    )
    owner = pd.Index(b.mrun).get_indexer(c.mrun)
    assert (owner >= 0).all() and b.mrun.is_unique
    a = table(source, paths, "A1", ["rbd", "cod_curso"])
    with (numeric / f"{year}.bin").open("rb") as fi:
        n, m, e = np.fromfile(fi, "<i4", count=3)
        offs = np.fromfile(fi, "<i4", count=int(n) + 1)
        prog = np.fromfile(fi, "<i4", count=e)
    assert len(b) == n and len(a) == m and len(c) == e
    np.testing.assert_array_equal(np.diff(offs), np.bincount(owner, minlength=n))
    for col in ["rbd", "cod_curso"]:
        np.testing.assert_array_equal(a[col].to_numpy()[prog], c[col].to_numpy())
    (csvname,) = [f for f in members(registry) if f.lower().endswith(".csv")]
    regparts = []
    with subprocess.Popen(
        ["bsdtar", "-xOf", str(registry), csvname], stdout=subprocess.PIPE
    ) as proc:
        for rows in pd.read_csv(
            proc.stdout,
            sep=";",
            encoding="utf-8-sig",
            usecols=["MRUN", "RBD"],
            chunksize=200000,
        ):
            regparts.append(rows[rows.MRUN.isin(b.mrun)])
        if proc.wait():
            raise RuntimeError("register extraction failed")
    targets = pd.concat(regparts).set_index("MRUN").RBD.reindex(b.mrun)
    ids = targets.fillna(-1).to_numpy()
    pos = c.rbd.to_numpy() == ids[owner]
    hit = {}
    unlinked_edges = ~targets.notna().to_numpy()[owner]
    radix = int(c.rbd.max()) + 1
    school_keys = owner.astype(np.int64) * radix + c.rbd.to_numpy()
    missing_menus = {}
    for k in ["full", "blind", "rank"]:
        mask = np.fromfile(numeric / f"{year}-{k}.mask", np.uint8)
        assert mask.size == e and np.isin(mask, [0, 1]).all()
        missing_menus[k] = np.unique(school_keys[unlinked_edges & mask.astype(bool)])
        # Count a person's school exactly once even when several displayed courses are at that school.
        hit[k] = np.bincount(owner[pos & mask.astype(bool)], minlength=n) > 0

    def exclusive_owners(arm, other):
        exclusive = np.setdiff1d(
            missing_menus[arm], missing_menus[other], assume_unique=True
        )
        return np.unique(exclusive // radix).size

    down = exclusive_owners("blind", "rank")
    up = exclusive_owners("rank", "blind")
    delta = int(hit["rank"].sum()) - int(hit["blind"].sum())
    return dict(
        year=year,
        unlinked_blind_exclusive=down,
        unlinked_rank_exclusive=up,
        lower_gain=delta - down,
        upper_gain=delta + up,
        all_applicants=n,
        linked=targets.notna().sum(),
        full=hit["full"].sum(),
        blind=hit["blind"].sum(),
        rank=hit["rank"].sum(),
        gained=(~hit["blind"] & hit["rank"]).sum(),
        lost=(hit["blind"] & ~hit["rank"]).sum(),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in [
        "early-archives",
        "early-numeric",
        "late-archives",
        "late-numeric",
        "registers",
        "out",
    ]:
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    rows = []
    for y in range(2022, 2026):
        source = a.early_archives if y < 2024 else a.late_archives
        numeric = a.early_numeric if y < 2024 else a.late_numeric
        name = "SAE-2023.rar" if y == 2023 else f"SAE_{y}.rar"
        rows.append(
            study(y, source / name, numeric, a.registers / f"matricula-{y + 1}.rar")
        )
    pd.DataFrame(rows).to_csv(a.out, index=False)

"""Check published advice-replay aggregates from the market-level count arrays."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def check(root):
    checks = json.loads((root / "nyc-advice-checks.json").read_text())
    for name, value in checks["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == value
    meta = pd.read_csv(root / "nyc-markets.csv")
    weights = meta.cell_weight / meta.groupby("design").cell_weight.transform("sum")
    with np.load(root / "nyc-advice-counts.npz", allow_pickle=False) as file:
        data = file["counts"].astype(int)
        columns = file["metrics"].tolist()
        scenarios = file["scenarios"].tolist()
    assert data.shape == (480, 32, 2, 6, 11)
    assert (
        meta.people.eq(64).all()
        and meta[["design", "seed", "month", "borough"]].duplicated().sum() == 0
    )
    summary = pd.read_csv(root / "nyc-advice-results.csv").set_index(
        ["design", "scenario", "metric"]
    )
    checked = 0
    arrays = {col: data[..., c] for c, col in enumerate(columns)}
    for base, label in [
        ("covered", "coverage"),
        ("placed", "placed"),
        ("first", "first"),
    ]:
        arrays["delta_" + label] = arrays[base + "_b"] - arrays[base + "_a"]
    assert np.array_equal(arrays["delta_placed"], arrays["gain"] - arrays["loss"])
    for ds, p in meta.groupby("design"):
        for i, scenario in enumerate(scenarios):
            for name, mat in arrays.items():
                x = mat[p.index, :, :, i].mean(axis=2)
                rep = np.average(x, weights=weights.loc[p.index], axis=0) * (100 / 64)
                row = summary.loc[(ds, scenario, name)]
                np.testing.assert_allclose(
                    [rep.mean(), rep.std(ddof=1) / np.sqrt(32), x.mean(axis=1).sum()],
                    [row.weighted_per100, row.pair_mcse_pp, row.pooled_count],
                    rtol=0,
                    atol=1e-10,
                )
                checked += 1
    t = pd.read_csv(root / "nyc-tail-counts.csv")
    sums = pd.read_csv(root / "nyc-tail-results.csv").set_index(
        ["design", "scenario", "metric"]
    )
    assert len(t) == 960
    for ds, scenario, key in sums.index:
        panel = t[(t.design == ds) & (t.scenario == scenario)]
        c = panel[key]
        np.testing.assert_allclose(
            [np.average(c, weights=panel.cell_weight) * 100 / 64, c.sum()],
            sums.loc[(ds, scenario, key), ["weighted_per100", "pooled_count"]],
            rtol=0,
            atol=1e-10,
        )
    for policy in ["a", "b"]:
        for (ds, scenario), g in t.groupby(["design", "scenario"]):
            diff = g["enters_" + policy] - g["exits_" + policy]
            assert (diff >= 0).all()
            before = summary.loc[(ds, "adhere_100", "placed_" + policy)].pooled_count
            assert g["placed_" + policy].sum() - diff.sum() == before
            if scenario == "all":
                assert (g.same_set == 64).all() and (g.delta_covered == 0).all()
            else:
                ranked = summary.loc[
                    (ds, "retrieve_best_omitted", "covered_" + policy)
                ].pooled_count
                assert g["covered_" + policy].sum() == ranked
    witnesses = pd.read_csv(root / "nyc-full-tail-witnesses.csv")
    joined = witnesses.merge(
        t[t.scenario == "all"],
        on=["design", "seed", "borough", "month"],
        validate="one_to_one",
    )
    assert len(joined) == 480
    assert (
        joined.source_losses.eq(joined.loss).all()
        and joined.source_gains.eq(joined.gain).all()
    )
    for col in ("source_losses", "source_gains"):
        assert (joined[col] <= joined.reversals).all()
    print(
        f"{checked} response estimates and {len(sums)} tail estimates reproduce from aggregate market counts."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "directory", nargs="?", type=Path, default=Path(__file__).resolve().parent
    )
    check(p.parse_args().directory)

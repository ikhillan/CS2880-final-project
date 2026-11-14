"""Validate the archived 2019 ranking models on two later cohorts without refitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def run(args):
    sys.path.insert(0, str(args.school.resolve()))
    import prepare

    inputs = json.loads(args.sources.read_text())
    years = (2024, 2025)
    for year in years:
        rec = next(s for s in inputs if str(year) in s["filename"])
        if hashlib.sha256((args.new / rec["filename"]).read_bytes()).hexdigest() != rec["sha256"]:
            raise ValueError(f"Source digest differs for {year}")
        prepare.FILES[year], prepare.URLS[year] = rec["filename"], rec["url"]
    fit_path = args.school / "fit-and-inputs.json"
    fitted = json.loads(fit_path.read_text())
    assert fitted["train_year"] == 2019
    history, _, labels, _, train, _ = prepare.prepare(args.old, 2019)
    valid = train["tier"] != 0
    programs = train["j"][valid]
    base = labels[valid].mean()
    history["win"] = np.bincount(programs, weights=labels[valid], minlength=len(history))
    history["n"] = np.bincount(programs, minlength=len(history))
    info = []
    args.numeric.mkdir(exist_ok=True, parents=True)
    for year in years:
        catalog, pos, _, features, market, record = prepare.prepare(args.new, year)
        prior_counts = catalog.merge(history, how="left", on=prepare.AKEY, validate="one_to_one").fillna(0)
        j = market["j"]
        prior = ((prior_counts.win + 20 * base) / (prior_counts.n + 20)).to_numpy()[j]
        features["log_odds_prior"] = np.log(prior / (1-prior))
        features["prior_sq"] = features.log_odds_prior.to_numpy() ** 2
        for name in ("blind", "rank"):
            if name == "rank":
                features["log_rank"] = np.log(pos)
                features["rank1"] = (pos == 1).astype(float)
                features["rank_pressure"] = features.log_rank * np.log((1 + np.bincount(j, minlength=market["m"])[j]) / (1 + market["cap"][j]))
            model = fitted["models"][name]
            assert features.columns.tolist() == model["features"]
            market[name] = prepare.predict(features.to_numpy(), model)
        offsets = np.r_[0,np.cumsum(np.bincount(market["i"], minlength=market["n"]))]
        assert offsets[-1] == len(j)
        with (args.numeric / f"{year}.bin").open("wb") as out:
            for arr in ([market["n"], market["m"], len(j)], offsets, j, *(market[k] for k in ("priority","cap","off","sen","g","ret","appended","degree"))):
                np.asarray(arr,dtype="<i4").tofile(out)
            for name in ("blind","rank"):
                np.asarray(market[name],dtype="<f8").tofile(out)
        info.append(record)
        print(f"{year}: {record['applicants']} applicants, {record['included_edges']} eligible applications",flush=True)
    args.record.write_text(json.dumps({"frozen_fit_sha256":hashlib.sha256(fit_path.read_bytes()).hexdigest(),"cohorts":info},indent=2)+"\n")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("school","old","new","numeric","sources","record"):
        p.add_argument("--"+key,required=True,type=Path)
    run(p.parse_args())

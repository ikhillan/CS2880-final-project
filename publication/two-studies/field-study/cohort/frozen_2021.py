"""Score the 2021 regular round with the archived 2019 models and program priors."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def score(school, archives, output):
    sys.path.insert(0, str(school.resolve()))
    import prepare

    fits = json.loads((school / "fit-and-inputs.json").read_text())
    assert fits["train_year"] == 2019
    history, _, labels, _, train, _ = prepare.prepare(archives, 2019)
    discretionary = train["tier"] != 0
    programs = train["j"][discretionary]
    base = labels[discretionary].mean()
    history["win"] = np.bincount(programs, weights=labels[discretionary], minlength=len(history))
    history["n"] = np.bincount(programs, minlength=len(history))
    del train, labels, programs

    catalog, position, _, features, market, _ = prepare.prepare(archives, 2021)
    counts = catalog.merge(history, on=prepare.AKEY, how="left", validate="one_to_one").fillna(0)
    j = market["j"]
    prior = ((counts.win + 20 * base) / (counts.n + 20)).to_numpy()[j]
    features["log_odds_prior"] = np.log(prior / (1 - prior))
    features["prior_sq"] = features.log_odds_prior.to_numpy() ** 2
    for name in ("blind", "rank"):
        if name == "rank":
            features["log_rank"] = np.log(position)
            features["rank1"] = (position == 1).astype(float)
            program_load = np.bincount(j, minlength=market["m"])
            features["rank_pressure"] = features.log_rank * np.log(
                (1 + program_load[j]) / (1 + market["cap"][j])
            )
        model = fits["models"][name]
        assert features.columns.tolist() == model["features"]
        market[name] = prepare.predict(features.to_numpy(), model)

    n = market["n"]
    offsets = np.r_[0, np.cumsum(np.bincount(market["i"], minlength=n))]
    assert offsets[-1] == len(j)
    with output.open("wb") as stream:
        integer_fields = ["priority", "cap", "off", "sen", "g", "ret", "appended", "degree"]
        arrays = [[n, market["m"], len(j)], offsets, j] + [market[key] for key in integer_fields]
        for array in arrays:
            np.asarray(array, dtype="<i4").tofile(stream)
        for name in ("blind", "rank"):
            np.asarray(market[name], dtype="<f8").tofile(stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("school", "archives", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    score(args.school, args.archives, args.output)

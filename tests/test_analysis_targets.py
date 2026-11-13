"""Analytic controls for paired, geography-standardized estimates."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments import analyze
from experiments.config import expand_labeled, load_config

ROOT = Path(__file__).resolve().parents[1]


def result(mid, place, arm, lower, higher, seed=1):
    row = {"market_id": mid, "borough": place, "month": "m", "exposure": arm, "seed": seed}
    for label, (n, eligible, blocked) in [("lower_income", lower), ("higher_income", higher)]:
        row.update({label + "_n": n, label + "_can_hide_count": eligible, label + "_blocker_count": blocked,
                    label + "_blocker_rate_given_can_hide": blocked / eligible if eligible else None})
    return row


class AnalysisTargets(unittest.TestCase):
    def test_conditional_common_support_and_market_pairing(self):
        # At A, the treated effects by group are both +1. Only the high group can hide at B.
        # The original unequal-support conditional estimate was 1 - .5 = +.5.
        data = pd.DataFrame([
            result("a", "A", "C", (2, 2, 2), (2, 2, 2)), result("a", "A", "U", (2, 2, 0), (2, 2, 0)),
            result("b", "B", "C", (2, 0, 0), (2, 2, 0)), result("b", "B", "U", (2, 0, 0), (2, 2, 0)),
            result("unpaired", "A", "U", (2, 2, 2), (2, 2, 0)),
        ])
        weights = {("A", "m"): .5, ("B", "m"): .5}
        conditional = analyze.contrast_T(data, "C", "U", weights, conditional=True)
        self.assertEqual(conditional["T"], 0.)
        self.assertEqual(conditional["coverage"], .5)
        self.assertEqual((conditional["paired_markets"], conditional["unpaired_markets"]), (2, 1))
        primary = analyze.contrast_T(data, "C", "U", weights)
        self.assertEqual((primary["T"], primary["coverage"]), (0., 1.))
        self.assertEqual(primary["B_higher_income_C"], .5)
        # No common support remains if the high group has no observations at A.
        data.loc[data.borough.eq("A"), "higher_income_can_hide_count"] = 0
        empty = analyze.contrast_T(data, "C", "U", weights, conditional=True)
        self.assertEqual(empty["coverage"], 0.)
        self.assertTrue(np.isnan(empty["T"]))

    def test_primary_pools_renters_and_holds_bootstrap_target_fixed(self):
        data = pd.DataFrame([result(str(i), "A", p, (n, e, y if p == "C" else 0), (5, 5, 0), i)
                             for i, n, e, y in [(0, 4, 2, 2), (1, 1, 0, 0)] for p in ["C", "U"]])
        w = {("A", "m"): .8, ("B", "m"): .2}
        point = analyze.estimate_t(data, "C", "U", w, False, 100)
        self.assertAlmostEqual(point["T"], 2 / 5)
        self.assertAlmostEqual(point["coverage"], .8)
        self.assertEqual(point["valid_bootstrap_draws"], 100)
        cond = analyze.estimate_t(data, "C", "U", w, True, 100)
        self.assertEqual(cond["T"], 1.)
        self.assertLess(cond["valid_bootstrap_draws"], 100)
        self.assertTrue(np.isnan(cond["T_lo"]))

    @patch.object(analyze, "load_renters", return_value=None)
    @patch.object(analyze, "borough_weights", return_value={
        "Manhattan": .34, "Brooklyn": .32, "Queens": .18, "Bronx": .139, "Staten Island": .021})
    def test_stage2_periods_have_distinct_planned_targets(self, *_):
        cfg = load_config(ROOT / "configs/stage2_main.json")
        plan = analyze.planned_weights([], pd.DataFrame(), expand_labeled(cfg))
        self.assertEqual(len(plan["primary", 64]), 10)
        self.assertEqual(len(plan["replication_2025", 64]), 15)
        for grid in plan.values():
            self.assertAlmostEqual(sum(grid.values()), 1.)
            self.assertAlmostEqual(sum(w for (b, _), w in grid.items() if b != "Staten Island"), .979)

    def test_protection_uses_cell_weights_and_keeps_raw_counts(self):
        rows = []
        for mid, cell, value in [("a", "A", 1), ("b", "B", 0)]:
            for method, cost in [("exact_prefix", 1), ("exact_prefix_protect_lower", 1 + value)]:
                rows.append(dict(variant="primary", n=64, seed=0, month="m", exposure="C", market_id=mid, borough=cell,
                                 method=method, cost=cost, certified=True, protected_renters=1, cost_scale=1, lower_bound=cost))
        tab = analyze.protection_table(pd.DataFrame(rows), {("primary", 64): {("A", "m"): .9, ("B", "m"): .1}})
        self.assertAlmostEqual(tab.iloc[0].mean_premium_low, .9)
        self.assertAlmostEqual(tab.iloc[0].positive_share_high, .9)
        self.assertEqual(tab.iloc[0].unweighted_mean_premium_low, .5)

    def test_switches_keep_sizes_separate_and_resumed_markets_count_once(self):
        markets = []
        for n, sign in [(24, True), (64, False)]:
            sw = {arm: {"lower_n": n // 2, "higher_n": n // 2, "lower_count": 0, "higher_count": 0}
                  for arm in ["AA", "BA", "AB", "BB"]}
            for arm in ["BA", "BB"]:
                sw[arm]["lower_count" if sign else "higher_count"] = n // 2
            markets.append(dict(market_id=f"m{n}", borough="A", month="m", seed=0, variant="primary", n=n,
                                group_switch={"compare": sw}))
        with TemporaryDirectory() as d, patch.object(analyze, "ROOT", Path(d)):
            folder = Path(d) / "results" / "test"
            folder.mkdir(parents=True)
            (folder / "runs.jsonl").write_text('{}\n')
            records = [markets[0], *markets, {**markets[0], "group_switch": {}}]
            (folder / "markets.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
            _, unique = analyze.load("test")
        self.assertEqual(len(unique), 2)
        w = {("primary", n): {("A", "m"): 1} for n in [24, 64]}
        tab = analyze.switch_table(unique, w, 5).set_index("n")
        self.assertEqual(tab.loc[24, "T"], 1.)
        self.assertEqual(tab.loc[64, "T"], -1.)
        self.assertTrue(tab.S_L_plus_S_H_equals_T.all())

"""Check our solver against the 160 archived core instances of the original audit package. Writes
results/archive_check/report.md.

    python -m experiments.check_archive [--zip PATH] [--no-brute]

The archive is read in place with zipfile and only its JSON is parsed; no archive code is executed. Definitions follow
ijcai-2027-paper-audit/verify.py (core_check) and methodology/audit_solver.py:
  * DA: buyer(renter)-proposing, FIFO queue of free buyers; an edge is exposed if it is initial or either endpoint's firm is audited.
  * mask: bit f set iff firm f is audited.
  * strict (protected) set: initially matched buyers whose full-disclosure rank is <= their initial rank.
  * protected optimum: cheapest mask that is stable and keeps every strict buyer at rank <= initial rank.
  * plain / protected support bound: first support-master optimum without / with one equality row per strict buyer.
Our side: hiddenoptions.audit (exact, enumerate_optimal_plans, protection_sets, ProtectedOracle, SupportMaster).
"""
from __future__ import annotations

import argparse
import os
import json
import time
import zipfile
from collections import Counter, deque
from pathlib import Path

import numpy as np

from hiddenoptions.audit import (Oracle, ProtectedOracle, SupportMaster, enumerate_optimal_plans, exact, protection_sets,
                                 support_cost)
from hiddenoptions.matching import NOT_EDGE, Instance, deferred_acceptance

ROOT = Path(__file__).resolve().parents[1]
# Location of the audit package: set AUDIT_PACKAGE_ZIP, or place the zip next to this repo.
ZIP = Path(os.environ.get("AUDIT_PACKAGE_ZIP", Path(__file__).resolve().parents[2] / "IJCAI_2027_AUDIT_PACKAGE.zip"))
INSTANCE_DIR = "hidden-blocking-pairs/artifact/experiments/core/instances/"
EVIDENCE = "ijcai-2027-paper-audit/evidence.json"
CUTS = ("prefix", "component", "assignment")
EXPECTED_COUNTS = {"markets": 160, "plain_bound_equals_protected_optimum": 146, "protected_bound_equals_protected_optimum": 158,
                   "strict_improvement_in_bound": 12, "protected_master_candidate_blocks": 9, "positive_premium": 6}
COUNT_KEYS = list(EXPECTED_COUNTS)


# ------------------------------------------------------------------------------------------- archive
def load_archive(path: Path = ZIP) -> tuple[list[tuple[str, dict]], dict]:
    """[(name, instance json)] sorted by name, and evidence['core'] rows keyed by market name (plus 'counts')."""
    with zipfile.ZipFile(path) as z:
        names = sorted(n for n in z.namelist() if n.startswith(INSTANCE_DIR) and n.endswith(".json"))
        data = [(n[len(INSTANCE_DIR):-5], json.loads(z.read(n))) for n in names]
        core = json.loads(z.read(EVIDENCE))["core"]
    evidence = {r["market"]: r for r in core["instances"]}
    evidence["counts"] = core["counts"]
    return data, evidence


def to_instance(market: dict) -> tuple[Instance, np.ndarray]:
    bp, sp = market["buyer_prefs"], market["seller_prefs"]
    nb, ns = len(bp), len(sp)
    if set(map(float, market["costs"])) != {1.0}:
        raise ValueError(f"{market['name']}: non-unit costs")
    rb = np.full((nb, ns), NOT_EDGE, dtype=np.int64)
    rs = rb.copy()
    for b, prefs in enumerate(bp):
        rb[b, prefs] = np.arange(len(prefs))
    for s, prefs in enumerate(sp):
        rs[prefs, s] = np.arange(len(prefs))
    inst = Instance(rb, rs, np.array(market["buyer_firms"]), np.array(market["seller_firms"]),
                    np.ones(len(market["costs"]), dtype=np.int64))
    e0 = np.zeros((nb, ns), bool)
    for b, s in market["initial_edges"]:
        e0[b, s] = True
    return inst, e0


def mask_of(x) -> int:
    return sum(1 << f for f, v in enumerate(np.asarray(x, bool)) if v)


# ------------------------------------------------------------------------------------------- reference (his definitions)
class Ref:
    """Independent re-statement of the archived Market/evaluate/deferred_acceptance on plain lists and dicts."""

    def __init__(self, market: dict):
        self.bp = [list(map(int, p)) for p in market["buyer_prefs"]]
        self.sp = [list(map(int, p)) for p in market["seller_prefs"]]
        self.bf, self.sf = list(market["buyer_firms"]), list(market["seller_firms"])
        self.n_firms = len(market["costs"])
        self.initial = {tuple(e) for e in market["initial_edges"]}
        self.brank = [{s: k for k, s in enumerate(p)} for p in self.bp]
        self.srank = [{b: k for k, b in enumerate(p)} for p in self.sp]
        self.edges = sorted((b, s) for b, p in enumerate(self.bp) for s in p)
        assert self.edges == sorted((b, s) for s, p in enumerate(self.sp) for b in p), "acceptability must be mutual"

    def exposed(self, mask: int) -> set:
        return {e for e in self.edges if e in self.initial or mask >> self.bf[e[0]] & 1 or mask >> self.sf[e[1]] & 1}

    def da(self, exposed: set) -> tuple[list, list]:
        bm, sm, nxt = [-1] * len(self.bp), [-1] * len(self.sp), [0] * len(self.bp)
        free = deque(range(len(self.bp)))
        while free:
            b = free.popleft()
            prefs = self.bp[b]
            while nxt[b] < len(prefs):
                s = prefs[nxt[b]]
                nxt[b] += 1
                if (b, s) not in exposed:
                    continue
                inc = sm[s]
                if inc == -1 or self.srank[s][b] < self.srank[s][inc]:
                    if inc != -1:
                        bm[inc] = -1
                        free.append(inc)
                    bm[b], sm[s] = s, b
                    break
        return bm, sm

    def blocks(self, bm, sm) -> list:
        out = []
        for b, s in self.edges:
            os_, ob = bm[b], sm[s]
            if os_ == s:
                continue
            if (os_ == -1 or self.brank[b][s] < self.brank[b][os_]) and (ob == -1 or self.srank[s][b] < self.srank[s][ob]):
                out.append((b, s))
        return out

    def ranks(self, bm) -> list:
        return [self.brank[b].get(s, len(self.brank[b])) for b, s in enumerate(bm)]

    def evaluate(self, mask: int):
        bm, sm = self.da(self.exposed(mask))
        return bm, sm, self.blocks(bm, sm)


def brute_force(ref: Ref, strict: dict, old: list) -> dict:
    """All 2^F masks with the reference DA: ordinary optimum, all cheapest masks, protected optimum."""
    stable, protected = [], []
    for mask in range(1 << ref.n_firms):
        bm, _, bl = ref.evaluate(mask)
        if bl:
            continue
        stable.append(mask)
        rr = ref.ranks(bm)
        if all(rr[b] <= old[b] for b in strict):
            protected.append(mask)
    pc = lambda m: bin(m).count("1")
    opt = min(map(pc, stable))
    return {"optimum": opt, "masks": sorted(m for m in stable if pc(m) == opt), "protected": min(map(pc, protected))}


# ------------------------------------------------------------------------------------------- one instance
def check_instance(name: str, d: dict, ev: dict, brute: bool = True, cuts=CUTS, enumerate_plans: bool = True,
                   time_cap: float = 120.0) -> dict:
    market = d["market"]
    t = {}
    clock = time.perf_counter()

    def lap(key):
        nonlocal clock
        now = time.perf_counter()
        t[key] = t.get(key, 0.0) + now - clock
        clock = now

    inst, e0 = to_instance(market)
    o = Oracle(inst, e0)
    ref = Ref(market)
    F = inst.n_firms
    zero, full = np.zeros(F, bool), np.ones(F, bool)
    r = {"name": name, "issues": []}

    # a. DA on E0 and on E, ours vs the reference
    ref0, reffull = ref.evaluate(0), ref.evaluate((1 << F) - 1)
    ours0 = deferred_acceptance(inst, inst.visible(zero, e0))
    oursfull = deferred_acceptance(inst, inst.visible(full, e0))
    r["da_e0_ok"] = ours0[0].tolist() == ref0[0] and ours0[1].tolist() == ref0[1]
    r["da_full_ok"] = oursfull[0].tolist() == reffull[0] and oursfull[1].tolist() == reffull[1]
    r["exposure_ok"] = all(np.array_equal(inst.visible(np.array([m >> f & 1 for f in range(F)], bool), e0),
                                          _mat(ref.exposed(m), inst.E.shape)) for m in (0, (1 << F) - 1, 0b1010101010 % (1 << F)))
    r["full_stable"] = not reffull[2] and o.stable(full)
    lap("a_da")

    # b. exact for each cut family vs the archived enumeration optimum
    archived = {res["method"]: res for res in d["results"]}
    opt = archived["enumeration"]["cost"]
    if opt != int(opt):
        raise ValueError(f"{name}: non-integer optimum {opt}")
    opt = int(opt)
    r["archive_optimum"] = opt
    r["exact"] = {}
    for c in cuts:
        res = exact(inst, o, cuts=c, time_cap=time_cap)
        r["exact"][c] = {"cost": res.cost, "certified": res.certified, "support_cost": res.support_cost, "seconds": res.seconds}
        lap(f"b_exact_{c}")

    # c. support bound
    r["support_cost"] = support_cost(inst, e0)
    lap("c_support")

    # d. all cheapest plans
    if enumerate_plans:
        plans, complete = enumerate_optimal_plans(inst, o, opt, cap=100_000, time_cap=time_cap)
        r["plans"] = sorted(mask_of(x) for x in plans)
        r["plans_complete"] = complete
        lap("d_enumerate")
    sens = d["optimal_outcome_sensitivity"]
    r["archive_masks"] = sorted(sens["masks"])
    r["archive_count"] = sens["count"]

    # e. protection with every renter in the group
    old = ref.ranks(ref0[0])
    strict = {b: s for b, s in enumerate(ref0[0]) if s != -1 and ref.ranks(reffull[0])[b] <= old[b]}
    protected, unavoidable = protection_sets(inst, o, np.ones(inst.n_b, bool))
    r["strict_ok"] = sorted(np.nonzero(protected)[0].tolist()) == sorted(strict)
    r["n_protected"], r["n_unavoidable"] = int(protected.sum()), int(unavoidable.sum())
    po = ProtectedOracle(o, protected, "all")
    rows_ref = {b: ref.bp[b][: ref.brank[b][s] + 1] for b, s in strict.items()}
    r["protect_rows_ok"] = {b: sorted(v) for b, v in po.protect_rows.items()} == {b: sorted(v) for b, v in rows_ref.items()}
    pres = exact(inst, po, cuts="prefix", time_cap=time_cap)
    r["protected"] = {"cost": pres.cost, "certified": pres.certified}
    lap("e_protected_exact")
    r["plain_support"] = r["support_cost"]  # his plain support_bound is the same model as our support_cost (no extra rows)
    r["protected_support"], cand = _support(inst, e0, po.protect_rows, o, po)
    r["candidate_blocks"], r["candidate_violates"] = cand
    lap("e_support_bounds")

    if brute:
        r["brute"] = brute_force(ref, strict, old)
        lap("brute")

    # comparisons
    def issue(cond, msg):
        if not cond:
            r["issues"].append(msg)

    issue(r["da_e0_ok"], "a: DA(E0) differs from reference")
    issue(r["da_full_ok"], "a: DA(E) differs from reference")
    issue(r["exposure_ok"], "a: visible set differs from reference exposure")
    issue(r["full_stable"], "a: DA(E) not stable")
    for c, v in r["exact"].items():
        issue(v["cost"] == opt and v["certified"], f"b: exact_{c} cost {v['cost']} (certified={v['certified']}) vs archive {opt}")
    issue(r["support_cost"] is not None and r["support_cost"] <= opt, f"c: support_cost {r['support_cost']} > optimum {opt}")
    issue(r["exact"].get("prefix", {}).get("support_cost", r["support_cost"]) == r["support_cost"],
          "c: support_cost differs from exact_prefix first master solve")
    if enumerate_plans:
        issue(r["plans"] == r["archive_masks"], f"d: plan masks {r['plans']} vs archive {r['archive_masks']}")
        issue(len(r["plans"]) == r["archive_count"], f"d: plan count {len(r['plans'])} vs archive count {r['archive_count']}")
        issue(r["plans_complete"], "d: enumeration not complete")
    issue(r["strict_ok"], "e: protection_sets differs from his strict set")
    issue(r["protect_rows_ok"], "e: protect_rows differ from his support rows")
    issue(r["protected"]["certified"], "e: protected exact not certified")
    issue(r["protected"]["cost"] == ev["protected"], f"e: protected {r['protected']['cost']} vs evidence {ev['protected']}")
    issue(opt == ev["ordinary"], f"archive optimum {opt} vs evidence ordinary {ev['ordinary']}")
    issue(r["plain_support"] == ev["plain_support"], f"e: plain_support {r['plain_support']} vs evidence {ev['plain_support']}")
    issue(r["protected_support"] == ev["protected_support"], f"e: protected_support {r['protected_support']} vs evidence {ev['protected_support']}")
    if brute:
        bf = r["brute"]
        issue(bf["optimum"] == opt, f"brute: optimum {bf['optimum']} vs archive {opt}")
        issue(bf["masks"] == r["archive_masks"], f"brute: masks {bf['masks']} vs archive {r['archive_masks']}")
        issue(bf["protected"] == ev["protected"], f"brute: protected {bf['protected']} vs evidence {ev['protected']}")
    r["info"] = []
    if r["candidate_blocks"] != ev["protected_master_candidate_has_blocker"]:
        r["info"].append(f"protected-master candidate has blocker: ours {r['candidate_blocks']}, evidence "
                         f"{ev['protected_master_candidate_has_blocker']}")
    if r["candidate_violates"]:
        r["info"].append("protected-master candidate violates protection (his verify.py asserts this never happens)")
    r["seconds"] = t
    return r


def _mat(edges: set, shape) -> np.ndarray:
    m = np.zeros(shape, bool)
    for b, s in edges:
        m[b, s] = True
    return m


def _support(inst, e0, rows, o, po) -> tuple[int, tuple[bool, bool]]:
    """First support-master optimum (as in his support_bound) and whether that candidate blocks / breaks protection."""
    status, x, _ = SupportMaster(inst, e0, protect_rows=rows).solve(60.0)
    if status != "optimal":
        raise RuntimeError(f"support master status {status}")
    return int(inst.cost[x].sum()), (bool(o.evaluate(x)[2].any()), bool(po.violations(x).any()))


def counts(results: list[dict], evidence: dict) -> dict:
    c = Counter({k: 0 for k in COUNT_KEYS})
    for r in results:
        ordinary, target = r["archive_optimum"], r["protected"]["cost"]
        plain, stronger = r["plain_support"], r["protected_support"]
        c["markets"] += 1
        c["plain_bound_equals_protected_optimum"] += plain == target
        c["protected_bound_equals_protected_optimum"] += stronger == target
        c["strict_improvement_in_bound"] += stronger > plain
        c["protected_master_candidate_blocks"] += r["candidate_blocks"]
        c["positive_premium"] += target > ordinary
    return dict(c)


# ------------------------------------------------------------------------------------------- report
def report(results: list[dict], evidence: dict, ours: dict, wall: float, brute: bool) -> str:
    n = len(results)
    ok = lambda f: sum(bool(f(r)) for r in results)
    rows = [
        ("a. DA(E0) = reference DA", ok(lambda r: r["da_e0_ok"])),
        ("a. DA(E) = reference DA", ok(lambda r: r["da_full_ok"])),
        ("a. visible set = reference exposure (3 masks)", ok(lambda r: r["exposure_ok"])),
        ("a. DA(E) stable in E", ok(lambda r: r["full_stable"])),
    ]
    for c in CUTS:
        rows.append((f"b. exact_{c}: cost = archive enumeration and certified",
                     ok(lambda r, c=c: r["exact"][c]["cost"] == r["archive_optimum"] and r["exact"][c]["certified"])))
    rows += [
        ("c. support_cost <= optimum", ok(lambda r: r["support_cost"] is not None and r["support_cost"] <= r["archive_optimum"])),
        ("d. enumerated optimal masks = archive masks (set)", ok(lambda r: r["plans"] == r["archive_masks"])),
        ("d. count = archive count", ok(lambda r: len(r["plans"]) == r["archive_count"])),
        ("d. enumeration reported complete", ok(lambda r: r["plans_complete"])),
        ("e. protection_sets = his strict set", ok(lambda r: r["strict_ok"])),
        ("e. protect_rows = his support-bound rows", ok(lambda r: r["protect_rows_ok"])),
        ("e. protected exact certified", ok(lambda r: r["protected"]["certified"])),
        ("e. protected exact = evidence protected", ok(lambda r: r["protected"]["cost"] == evidence[r["name"]]["protected"])),
        ("e. plain support bound = evidence", ok(lambda r: r["plain_support"] == evidence[r["name"]]["plain_support"])),
        ("e. protected support bound = evidence", ok(lambda r: r["protected_support"] == evidence[r["name"]]["protected_support"])),
        ("e. protected-master candidate keeps protection", ok(lambda r: not r["candidate_violates"])),
        ("(info) protected-master candidate blocker flag = evidence",
         ok(lambda r: r["candidate_blocks"] == evidence[r["name"]]["protected_master_candidate_has_blocker"])),
    ]
    if brute:
        rows += [
            ("brute force (reference DA, all 1024 masks): optimum = archive", ok(lambda r: r["brute"]["optimum"] == r["archive_optimum"])),
            ("brute force: cheapest masks = archive masks", ok(lambda r: r["brute"]["masks"] == r["archive_masks"])),
            ("brute force: protected optimum = evidence", ok(lambda r: r["brute"]["protected"] == evidence[r["name"]]["protected"])),
        ]
    out = ["# Archive check: original audit package vs hiddenoptions", "",
           f"{n} core instances from `{ZIP.name}`; {wall:.1f} s wall time.", "",
           "## Checks", "", "| check | pass | of |", "|---|---|---|"]
    out += [f"| {k} | {v} | {n} |" for k, v in rows]
    out += ["", "## Counts (evidence.json core.counts)", "", "| count | ours | evidence |", "|---|---|---|"]
    out += [f"| {k} | {ours[k]} | {evidence['counts'][k]} |" for k in COUNT_KEYS]
    out += ["", "`protected_master_candidate_blocks` depends on which tied optimum the support MILP returns, so a difference there is "
            "informational.", "", "## Mismatches", ""]
    bad = [r for r in results if r["issues"]]
    out += [f"- `{r['name']}`: " + "; ".join(r["issues"]) for r in bad] or ["None."]
    out += ["", "## Informational differences", ""]
    info = [r for r in results if r["info"]]
    out += [f"- `{r['name']}`: " + "; ".join(r["info"]) for r in info] or ["None."]
    out += ["", "## Positive-premium and bound-gap instances", "",
            "| instance | ordinary | protected | plain support | protected support | #protected | #unavoidable |", "|---|---|---|---|---|---|---|"]
    for r in results:
        if r["protected"]["cost"] > r["archive_optimum"] or r["protected_support"] != r["protected"]["cost"] or r["protected_support"] > r["plain_support"]:
            out.append(f"| {r['name']} | {r['archive_optimum']} | {r['protected']['cost']} | {r['plain_support']} | "
                       f"{r['protected_support']} | {r['n_protected']} | {r['n_unavoidable']} |")
    tot = Counter()
    for r in results:
        tot.update(r["seconds"])
    slow = sorted(results, key=lambda r: -sum(r["seconds"].values()))[:5]
    out += ["", "## Timings", "", "| stage | total s | mean s/instance |", "|---|---|---|"]
    out += [f"| {k} | {v:.2f} | {v / n:.3f} |" for k, v in sorted(tot.items())]
    out += ["", "Slowest instances: " + ", ".join(f"`{r['name']}` ({sum(r['seconds'].values()):.2f} s)" for r in slow)]
    return "\n".join(out) + "\n"


def run(path: Path = ZIP, brute: bool = True) -> tuple[list[dict], dict, dict]:
    data, evidence = load_archive(path)
    results = [check_instance(name, d, evidence[name], brute=brute) for name, d in data]
    return results, evidence, counts(results, evidence)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zip", type=Path, default=ZIP)
    ap.add_argument("--no-brute", action="store_true", help="skip the 2^F reference-DA enumeration")
    args = ap.parse_args()
    t0 = time.perf_counter()
    results, evidence, ours = run(args.zip, brute=not args.no_brute)
    wall = time.perf_counter() - t0
    out = ROOT / "results" / "archive_check"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report(results, evidence, ours, wall, not args.no_brute))
    n_bad = sum(bool(r["issues"]) for r in results)
    counts_ok = all(ours[k] == evidence["counts"][k] for k in COUNT_KEYS if k != "protected_master_candidate_blocks")
    print(f"{len(results)} instances, {n_bad} with mismatches; counts "
          + ", ".join(f"{k}={ours[k]}/{evidence['counts'][k]}" for k in COUNT_KEYS)
          + f" ({'match' if counts_ok else 'MISMATCH'}); {wall:.1f} s -> {out / 'report.md'}")


if __name__ == "__main__":
    main()

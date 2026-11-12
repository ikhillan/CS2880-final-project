"""Our solver against the original audit package's 160 core instances (see experiments/check_todd_archive.py).

All 160 instances: exact (prefix cuts) = archived enumeration optimum, certified; protected exact = evidence.json; the
plain/protected support bounds and the counts 146/158/12/6. The component and assignment cut families and the
enumeration of every cheapest plan run on a subset (every 10th instance) to keep the module near a minute; the full
sweep is `python -m experiments.check_todd_archive`.
"""
import pytest

from experiments.check_todd_archive import COUNT_KEYS, ZIP, check_instance, counts, load_archive

pytestmark = pytest.mark.skipif(not ZIP.exists(), reason="archive zip not present")
SUBSET_STEP = 10


@pytest.fixture(scope="module")
def checked():
    data, evidence = load_archive(ZIP)
    results = []
    for i, (name, d) in enumerate(data):
        full = i % SUBSET_STEP == 0
        results.append(check_instance(name, d, evidence[name], brute=False, enumerate_plans=full,
                                      cuts=("prefix", "component", "assignment") if full else ("prefix",)))
    return results, evidence


def test_exact_matches_enumeration(checked):
    results, _ = checked
    assert len(results) == 160
    bad = [(r["name"], c, v["cost"], v["certified"], r["archive_optimum"]) for r in results for c, v in r["exact"].items()
           if v["cost"] != r["archive_optimum"] or not v["certified"]]
    assert not bad
    assert sum(len(r["exact"]) == 3 for r in results) == 16


def test_da_and_support(checked):
    results, _ = checked
    assert all(r["da_e0_ok"] and r["da_full_ok"] and r["exposure_ok"] and r["full_stable"] for r in results)
    assert all(r["support_cost"] <= r["archive_optimum"] for r in results)


def test_optimal_plan_sets(checked):
    results, _ = checked
    sub = [r for r in results if "plans" in r]
    assert len(sub) == 16
    bad = [(r["name"], r["plans"], r["archive_masks"]) for r in sub
           if r["plans"] != r["archive_masks"] or len(r["plans"]) != r["archive_count"] or not r["plans_complete"]]
    assert not bad


def test_protection(checked):
    results, evidence = checked
    assert all(r["strict_ok"] and r["protect_rows_ok"] for r in results)
    bad = [(r["name"], r["protected"], evidence[r["name"]]["protected"]) for r in results
           if not r["protected"]["certified"] or r["protected"]["cost"] != evidence[r["name"]]["protected"]]
    assert not bad
    bad = [(r["name"], r["plain_support"], r["protected_support"]) for r in results
           if (r["plain_support"], r["protected_support"]) != (evidence[r["name"]]["plain_support"], evidence[r["name"]]["protected_support"])]
    assert not bad


def test_counts(checked):
    results, evidence = checked
    ours = counts(results, evidence)
    expected = {k: v for k, v in evidence["counts"].items() if k != "protected_master_candidate_blocks"}  # depends on MILP tie-breaking
    assert {k: ours[k] for k in expected} == expected
    assert expected == {"markets": 160, "plain_bound_equals_protected_optimum": 146, "protected_bound_equals_protected_optimum": 158,
                        "strict_improvement_in_bound": 12, "positive_premium": 6}
    assert set(COUNT_KEYS) == set(evidence["counts"])

"""Reconcile cohort-specific checks and pooled, institutional and family ceilings."""
import argparse
import csv
from collections import Counter
from pathlib import Path


def rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def ceilings(folder, prefix):
    data = {}
    for year in (2022, 2023, 2024):
        for row in rows(folder / f"{prefix}-{year}.csv"):
            key = (year, row["priority"], int(row["support"]), row["target"])
            assert key not in data
            data[key] = int(row["additional_reachable"])
    return data


def audit(folder, label, require_all):
    fields = ("missing", "blocking", "exits", "entries", "changed_program", "changed_detailed")
    summary = Counter()
    for year in (2022, 2023, 2024):
        data = rows(folder / f"{label}-{year}.csv.audit.csv")
        for row in data:
            split = row["split"]
            certificate = row.get("certificate", "stable")
            if split == "design_check":
                assert int(row["missing"]) == int(row["blocking"]) == 0
                if certificate == "proposal_prefix":
                    assert int(row["changed_detailed"]) == 0
            key = (year, split, certificate)
            summary[key + ("count",)] += 1
            for field in fields:
                if field in row:
                    summary[key + (field,)] += int(row[field])
            if "changed_detailed" in row and int(row["changed_detailed"]) > 0:
                summary[key + ("draws_with_detailed_changes",)] += 1
        if require_all:
            design = [r for r in data if r["split"] == "design_check" and r.get("certificate","stable") == "stable"]
            assert [int(r["draw"]) for r in design] == list(range(1,1025))
            assert summary[year, "design_check", "stable", "count"] == 1024
        elif label == "paired":
            assert summary[year, "holdout", "stable", "count"] == 64
    return summary


def main(args):
    pooled = ceilings(args.folder, "pooled")
    matched = ceilings(args.folder, "paired")
    assert [pooled[y, "design", 1024, "april_school"] for y in (2022, 2023, 2024)] == [81,95,102]
    assert all(matched[y,"design",0,t] == pooled[y,"design",0,t] for y in (2022,2023,2024) for t in ("initial_offer","april_school"))
    assert [matched[y,"design",1024,"april_school"] for y in (2022,2023,2024)] == [96,111,107]
    for prefix,expected in (("family",[97,113,107]),("course-family",[96,113,105])):
        data = ceilings(args.folder, prefix)
        assert [data[y,"design",1024,"april_school"] for y in (2022,2023,2024)] == expected
    prefixes = ["pooled", "paired", "family", "course-family"]
    with args.summary.open("w",newline="") as out:
        writer = csv.writer(out,lineterminator="\n")
        writer.writerow(["model","criterion","support","target","2022","2023","2024","total"])
        for prefix in prefixes:
            table = ceilings(args.folder, prefix)
            for target in ("initial_offer","april_school"):
                criteria = [("recorded",[1]), ("design",[0,16,64,256,1024])]
                if prefix == "paired":
                    criteria.append(("proposal_prefix",[16,64,256,1024]))
                for priority, supports in criteria:
                    prev = None
                    for support in supports:
                        key = (2022,priority,support,target)
                        if key not in table:
                            assert priority == "proposal_prefix" and prefix != "paired"
                            continue
                        by_year = [table[y,priority,support,target] for y in (2022,2023,2024)]
                        if prev is not None:
                            assert all(n<=old for n,old in zip(by_year,prev))
                        if priority == "proposal_prefix":
                            assert all(n<=table[y,"design",support,target] for y,n in zip((2022,2023,2024),by_year))
                        writer.writerow([prefix,priority,support,target,*by_year,sum(by_year)])
                        prev=by_year
    for prefix in ("paired","family","course-family"):
        audit(args.folder,prefix,False)
    totals={} if args.skip_all_design else audit(args.folder,"exact",True)
    outfields=["count","missing","blocking","exits","entries","changed_program","changed_detailed","draws_with_detailed_changes"]
    all_evaluations={"paired":audit(args.folder,"paired",False), "family":audit(args.folder,"family",False),
                     "course-family":audit(args.folder,"course-family",False), "exact":totals}
    with args.audit.open("w",newline="") as out:
        writer=csv.writer(out,lineterminator="\n")
        writer.writerow(["specification","year","split","certificate",*outfields])
        for model,stats in all_evaluations.items():
            for y,sp,cert in sorted({k[:3] for k in stats}):
                if model in ("family","course-family") and cert != "stable":
                    continue
                detailed_available = model in ("paired","exact")
                values=[stats[y,sp,cert,k] if detailed_available or k not in ("changed_detailed","draws_with_detailed_changes")
                        else "" for k in outfields]
                writer.writerow([model,y,sp,cert,*values])
    if args.private is None:
        return
    import numpy as np
    with args.overlap.open("w",newline="") as out:
        writer=csv.writer(out,lineterminator="\n")
        writer.writerow(["year","model","target","both","pooled_only","reconstruction_only"])
        for year in (2022,2023,2024):
            baseline=np.fromfile(args.private/f"pooled-{year}.bits",dtype="u1").reshape(2,-1)
            for model in ["paired","family","course-family"]:
                alternatives = [args.private/f"{model}-{year}.bits", args.private/f"{model.replace('-', '_')}-{year}.bits"]
                indicator = next(path for path in alternatives if path.exists())
                alternative=np.fromfile(indicator,dtype="u1").reshape(2,-1)
                assert alternative.shape == baseline.shape
                for idx,target in enumerate(("initial_offer","april_school")):
                    old,new=baseline[idx].astype(bool),alternative[idx].astype(bool)
                    assert old.sum()==pooled[year,"design",1024,target]
                    assert new.sum()==ceilings(args.folder,model)[year,"design",1024,target]
                    writer.writerow([year,model,target,(old&new).sum(),(old & ~new).sum(),(new & ~old).sum()])


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder",type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument("--summary",type=Path,default=Path("ceiling-summary.csv"))
    ap.add_argument("--audit",type=Path,default=Path("exact-design-checks.csv"))
    ap.add_argument("--skip-all-design",action="store_true",help="omit full within-design replay diagnostics")
    ap.add_argument("--private",type=Path,help="optional local indicators for paired overlap")
    ap.add_argument("--overlap",type=Path,default=Path("ceiling-overlap.csv"))
    main(ap.parse_args())

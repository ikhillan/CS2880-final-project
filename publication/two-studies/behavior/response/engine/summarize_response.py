"""Verify and summarize the national shortlist-response comparisons."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import statistics

def read(path):
    with path.open() as f:
        return [{k: (v if k in ("scenario","stratum","policy") else int(v)) for k,v in row.items()}
                for row in csv.DictReader(f)]

def write(path, records):
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator="\n")
        w.writeheader()
        w.writerows(records)

METRICS=("placed","linked_placed","april","first","offers","nooffer_linked_placed","nooffer_april")
def measures(row):
    metrics={k:row["rank_"+k]-row["blind_"+k] for k in METRICS}
    metrics.update({k:row[k] for k in ("n","linked","exits","entries","adding_blind","adding_rank","edges_added_blind","edges_added_rank","potential_blind","potential_rank")})
    return metrics

def summarize(folder, institutional, output):
    allrows=[]
    for year in range(2022,2026):
        groups=["endpoints","reapplication","field","multiple"]+(["risk"] if year<2025 else [])
        yearly=[r for group in groups for r in read(folder/f"{group}-{year}.csv")]
        assert all(r["year"]==year for r in yearly)
        keys={(r["scenario"],r["permille"],r["rep"]):r for r in yearly}
        assert len(keys)==len(yearly)==(5+4+116+16+(64 if year<2025 else 0))
        baseline=keys["binding",0,0]
        all_trans={r["stratum"]:r for r in read(institutional/f"transitions-{year}.csv")}
        for k in ("rank_placed","blind_placed"):
            assert baseline[k]==all_trans["all"][k]
            assert baseline[k.replace("_placed","_linked_placed")]==all_trans["all_linked"][k]
        for k in ("rank","blind"):
            assert baseline[k+"_april"]==all_trans["all_linked"][k+"_april_shown"]
        for row in yearly:
            assert row["rank_placed"]-row["blind_placed"]==row["entries"]-row["exits"]
            assert 0<=row["adding_blind"]<=row["potential_blind"]
            assert 0<=row["adding_rank"]<=row["potential_rank"]
            name=row["scenario"]
            if name.endswith("_original"):
                other=keys[name.removesuffix("_original")+"_tail",row["permille"],row["rep"]]
                for policy in ("rank","blind"):
                    assert row[policy+"_april"]==other[policy+"_april"]
                    assert row[policy+"_offers"]==other[policy+"_offers"]
                    assert row["adding_"+policy]==other["adding_"+policy]
            if name=="all_original":
                assert not any(measures(row)[k] for k in METRICS)
            elif name!="all_tail":
                assert row["rank_placed"]<row["blind_placed"] and row["rank_linked_placed"]<row["blind_linked_placed"]
                assert row["rank_april"]>row["blind_april"]
        comparisons=read(folder/f"reference-{year}.csv")
        assert len(comparisons)==16 and all(r["applicants"]==baseline["n"] and r["changed_contract_category"]==r["changed_original_course"]==0 for r in comparisons)
        allrows+=yearly
    # Per-seed four-cohort totals first; seeds are nested across participation intensities.
    buckets=defaultdict(lambda:defaultdict(lambda:defaultdict(int)))
    for row in allrows:
        for y in ("aggregate",str(row["year"])):
            g=(row["scenario"],row["permille"],y)
            dest=buckets[g][row["rep"]]
            for k,v in measures(row).items():dest[k]+=v
    result=[]
    for (scenario,rate,scope),seeds in sorted(buckets.items()):
        assert len(seeds)==(1 if rate in (0,1000) else 8)
        row={"scope":scope,"scenario":scenario,"permille":rate,"repetitions":len(seeds)}
        for key in next(iter(seeds.values())):
            values=[s[key] for s in seeds.values()]
            row[key+"_mean"]=statistics.mean(values)
            if key in METRICS:
                row[key+"_min"]=min(values);row[key+"_max"]=max(values)
                row[key+"_mcse"]=round(statistics.stdev(values)/len(values)**.5,5) if len(values)>1 else ""
        result.append(row)
    write(output/"response-summary.csv",result)
    print(len(allrows),"cohort–scenario–seed rows;",len(result),"summaries; all aggregate and synchronous checks passed.")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("results","institutional","out"):p.add_argument("--"+key,required=True,type=Path)
    a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    summarize(a.results,a.institutional,a.out)

"""Build NYC recent-mover histograms from public Census housing data without exporting household IDs.

Usage: python3 prepare_acs.py [--directory DIRECTORY]
Download csv_hny.zip and tract_to_puma_2020.txt from the two Census URLs below.
"""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

RELEASE = "ACS2020-2024"
PUMS = "https://www2.census.gov/programs-surveys/acs/data/pums/2024/5-Year/csv_hny.zip"
CROSSWALK = "https://www2.census.gov/geo/docs/maps-data/data/rel2020/2020_Census_Tract_to_2020_PUMA.txt"
CPI = {"2024_annual": 313.689, "2026-06": 333.952, "2026-07": 333.918, "2026-08": 334.980}
CPI_TABLE = "https://www.bls.gov/regions/mid-atlantic/data/consumerpriceindexhistorical_us_table.htm"
BOROUGHS = {"005": "Bronx", "047": "Brooklyn", "061": "Manhattan", "081": "Queens", "085": "Staten Island"}
RATIO = Decimal(str(CPI["2026-07"])) / Decimal(str(CPI["2024_annual"]))

def file_hash(path):
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()

def puma_mapping(path):
    counties = defaultdict(set)
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["STATEFP"] == "36":
                counties[row["PUMA5CE"]].add(row["COUNTYFP"])
    result = {}
    for puma, members in counties.items():
        if members.intersection(BOROUGHS):
            if len(members) != 1:
                raise ValueError(f"NYC PUMA {puma} crosses counties: {members}")
            result[puma] = BOROUGHS[next(iter(members))]
    if set(result.values()) != set(BOROUGHS.values()):
        raise ValueError("Missing a borough in 2020 PUMA crosswalk")
    return result, counties

def split_for(serial):
    digest = hashlib.sha256((RELEASE + "|" + serial).encode()).digest()
    return "calibration" if int.from_bytes(digest[:8], "big") % 10 < 7 else "evaluation"

def median(incomes):
    total = sum(incomes.values())
    acc = 0
    for levels, weight in sorted(incomes.items()):
        acc += weight
        if acc * 2 >= total:
            return Decimal(levels) / Decimal(1_000_000)
    raise ValueError("No rental households in analysis")

def run(directory):
    zip_path = directory / "csv_hny.zip"
    cross_path = directory / "tract_to_puma_2020.txt"
    puma, ny_pumas = puma_mapping(cross_path)
    income_weights = Counter()
    renters = Counter()
    recent = Counter()
    weight = Counter()
    weights_recent = Counter()
    donors = defaultdict(lambda: [0, 0])
    excluded = Counter()
    with zipfile.ZipFile(zip_path) as archive, archive.open("psam_h36.csv") as raw:
        data = csv.reader(io.TextIOWrapper(raw))
        header = next(data)
        idx = {field:header.index(field) for field in ["PUMA","TEN","NP","HINCP","ADJINC","WGTP","MV","SERIALNO","TYPEHUGQ"]}
        for row in data:
            p = row[idx["PUMA"]]
            if p not in ny_pumas:
                excluded["unrecognized_ny_puma_all_file_rows"] += 1
            if p not in puma or row[idx["TEN"]] != "3" or row[idx["TYPEHUGQ"]] != "1":
                continue
            if any(row[idx[k]].strip() == "" for k in ["NP","HINCP","ADJINC","WGTP","MV"]):
                excluded["renting_missing_required_field"] += 1
                continue
            w=int(row[idx["WGTP"]]); h=int(row[idx["NP"]]); inc=int(row[idx["HINCP"]]); adj=int(row[idx["ADJINC"]])
            if w <= 0 or h < 1 or adj <= 0:
                excluded["unweighted_empty_or_invalid_factor"] += 1
                continue
            microdollars=inc*adj
            bor=puma[p]
            renters[bor] += 1
            weight[bor] += w
            income_weights[microdollars] += w
            if row[idx["MV"]] != "1":
                continue
            recent[bor] += 1
            weights_recent[bor] += w
            cell = donors[(bor,split_for(row[idx["SERIALNO"]]),h,microdollars)]
            cell[0]+=w
            cell[1]+=1
    cutoff=median(income_weights)
    outfile=directory/"recent_donors.csv"
    with outfile.open("w",newline="") as fh:
        out=csv.writer(fh)
        out.writerow(["borough","donor_split","household_size","income_2024","income_july2026","weight","rows"])
        for (bor,split,h,microdollars),(w,n) in sorted(donors.items()):
            inc=Decimal(microdollars)/Decimal(1_000_000)
            out.writerow([bor,split,h,f"{inc:.6f}",f"{inc*RATIO:.12f}",w,n])
    assert sum(v[0] for v in donors.values()) == sum(weights_recent.values())
    assert sum(v[1] for v in donors.values()) == sum(recent.values())
    summary={"release":RELEASE,"sources":{"housing":PUMS,"geography":CROSSWALK,"cpi":CPI_TABLE,
        "cpi_2024_annual":"https://www.govinfo.gov/content/pkg/ECONI-2026-08/pdf/ECONI-2026-08.pdf",
        "guide":"https://www2.census.gov/programs-surveys/acs/tech_docs/pums/2020_2024ACS_PUMS_User_Guide.pdf"},
        "sha256":{"housing_zip":file_hash(zip_path),"crosswalk":file_hash(cross_path),"histogram":file_hash(outfile)},
        "cpi_u":CPI,"filters":"TYPEHUGQ=1, TEN=3, NP>=1, positive WGTP and ADJINC, nonmissing income/fields. Recent mover additionally MV=1. Full NYC renter distribution defines threshold.",
        "income":"HINCP * ADJINC/1e6 converts to 2024 dollars; multiply once by 333.918/313.689 for July 2026 dollars.",
        "weighted_median_definition":"first ordered income whose cumulative household weight >= half total",
        "median_income_2024":float(cutoff),"median_income_july2026":float(cutoff*RATIO),
        "max_monthly_rent_40pct_at_median_july2026":float(cutoff*RATIO/30),
        "renters_rows":sum(renters.values()),"renters_weight":sum(weight.values()),"recent_rows":sum(recent.values()),"recent_weight":sum(weights_recent.values()),
        "by_borough":{b:{"renters_rows":renters[b],"renters_weight":weight[b],"recent_rows":recent[b],"recent_weight":weights_recent[b]} for b in sorted(BOROUGHS.values())},
        "puma_nyc":dict(sorted(puma.items())), "exclusion_counts":dict(excluded),
        "donor_split":"SHA256('ACS2020-2024|' + SERIALNO), first eight digest bytes as big endian unsigned integer modulo 10. Buckets 0–6 calibration; 7–9 evaluation. Split before aggregation.",
        "income_group":"lower iff adjusted 2024 income <= median_income_2024 (exact equality included)."}
    (directory/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:summary[k] for k in ["median_income_2024","median_income_july2026","max_monthly_rent_40pct_at_median_july2026","renters_rows","renters_weight","recent_rows","recent_weight","by_borough","exclusion_counts"]},indent=2))

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--directory",type=Path,default=Path(__file__).parent/"data/acs")
    run(parser.parse_args().directory)

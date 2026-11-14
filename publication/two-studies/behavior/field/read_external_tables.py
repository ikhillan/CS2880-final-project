"""Extract selected reported estimates from the authors' two public aggregate tables."""

import argparse
import csv
import hashlib
import re
import urllib.request
from pathlib import Path

SOURCES = {
    "popup": (5862803, "0e2651cb3a0ec4086e83432b492fb960"),
    "whats": (5862829, "4f0c5c49c83a00066a2d85e5f8beb03b"),
}
SELECT = {
    "popup": {
        "B": ["Add any", "Add as last", "Drop any"],
        "C": ["Placed to preference", "Enrolled in placed"],
    },
    "whats": {"C": ["Add any", "Schools Added"], "D": ["Add any", "Schools Added"]},
}


def selected_table(file, source):
    code, checksum = SOURCES[source]
    if hashlib.md5(file.read_bytes()).hexdigest() != checksum:
        raise ValueError(f"Original published table checksum changed: {file}")
    lines = file.read_text().splitlines()
    numbers = []
    panel = None
    for i, line in enumerate(lines):
        header = re.search(r"\\textit\{([A-D])\.", line)
        if header:
            panel = header[1]
        columns = line.split("&")
        name = columns[0].strip()
        if name not in SELECT[source].get(panel, []):
            continue
        effect = float(columns[1])
        se = float(lines[i + 1].split("&")[1].strip().strip("()"))
        numbers.append(
            {
                "source": source,
                "panel": panel,
                "column": 1,
                "outcome": name,
                "estimate": effect,
                "reported_se": se,
                "unit": "schools" if name == "Schools Added" else "proportion",
                "table_access": f"https://dataverse.harvard.edu/api/access/datafile/{code}",
            }
        )
    assert len(numbers) == sum(map(len, SELECT[source].values()))
    return numbers


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tables", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--download",
        action="store_true",
        help="Download source tables from the authors’ open release",
    )
    args = p.parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (code, _) in SOURCES.items():
        path = args.tables / (name + ".tex")
        if args.download:
            urllib.request.urlretrieve(
                f"https://dataverse.harvard.edu/api/access/datafile/{code}", path
            )
        rows.extend(selected_table(path, name))
    with (args.out / "published-estimates.csv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

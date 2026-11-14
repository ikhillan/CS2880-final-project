"""Extract the published Chilean report-card and school value-added ITTs."""
import argparse
import csv
import hashlib
import re
import subprocess
from math import erfc, sqrt
from pathlib import Path

SOURCES = {
    "paper": ("https://www.nber.org/system/files/working_papers/w35837/w35837.pdf",
              "da6bed807eed933df94853dc52d5c6bce4925de139f4a8b7d4994d77107fc432"),
    "appendix": ("https://www.nber.org/data-appendix/w35837/SchoolChoiceInfoExp_R3_OnlineAppendix.pdf",
                 "70e206bfcbd1a62004dac23cff5f4c24726329b608b3b14c72f97fe7b14e1a73"),
}
GROUPS = {"A": "all", "B": "already_enrolled", "C": "pre_enrollment"}
NUMBERS = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def document(path, kind):
    url, target_hash = SOURCES[kind]
    if hashlib.sha256(path.read_bytes()).hexdigest() != target_hash:
        raise ValueError(f"Unexpected PDF version for {kind}. Source: {url}")
    return subprocess.check_output(["pdftotext", "-layout", str(path), "-"], text=True)


def panels(text, name):
    # Match the centered heading, not a prose reference or the table of contents.
    heading = re.search(r"(?m)^\s*TABLE " + name + r"\s*$", text)
    if not heading:
        raise ValueError(f"Missing table {name}")
    table = text[heading.end():].split("Notes:", 1)[0]
    fragments = re.split(r"(?m)^\s*Panel ([A-C]):[^\n]*\n", table)
    if fragments[1::2] != list(GROUPS):
        raise ValueError(f"Unexpected panels in table {name}")
    for panel, fragment in zip(fragments[1::2], fragments[2::2]):
        lines = fragment.splitlines()
        index = next(i for i, line in enumerate(lines) if line.strip().startswith("Treatment "))
        count_line = next(line for line in lines if line.strip().startswith("N obs."))
        rows = [[float(x.replace(",", "")) for x in NUMBERS.findall(line)]
                for line in [lines[index], lines[index + 1], count_line]]
        if any(len(row) != 6 for row in rows):
            raise ValueError(f"Unexpected cells in {name}, panel {panel}")
        yield panel, list(zip(*rows))


def extract(paper, appendix, out):
    records = []
    specs = [
        ("appendix", appendix, "O-4", [
            ("enroll_listed", "randomization", 0, "probability"),
            ("enroll_listed", "expanded", 1, "probability"),
            ("choose_top_five_given_listed", "expanded", 5, "probability"),
        ]),
        ("paper", paper, "I", [("attended_school_value_added", "expanded", 5, "normalized_units")]),
    ]
    for kind, path, table, columns in specs:
        for panel, values in panels(document(path, kind), table):
            for outcome, controls, col, unit in columns:
                estimate, se, n = values[col]
                records.append({"table": table, "panel": panel, "column": col + 1,
                                "sample": GROUPS[panel], "outcome":outcome, "unit":unit,
                                "controls":controls, "estimate": f"{estimate:.3f}",
                                "clustered_se":f"{se:.3f}", "n":int(n),
                                "normal_95_lo":f"{estimate - 1.96 * se:.5f}",
                                "normal_95_hi":f"{estimate + 1.96 * se:.5f}"})
    with out.open("w") as stream:
        writer = csv.DictWriter(stream, records[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("paper","appendix","out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    r = extract(args.paper, args.appendix, args.out)
    # A sign check combining two published regression endpoints requires both
    # effects to be in their nominated directions; their covariance need not be known.
    rows = {s["outcome"]:s for s in r if s["sample"] == "pre_enrollment" and s["controls"] == "expanded"}
    def directional_p(outcome, sign):
        row = rows[outcome]
        z = sign * float(row["estimate"]) / float(row["clustered_se"])
        return erfc(z/sqrt(2)) / 2
    p_both = max(directional_p("enroll_listed", -1),
                 directional_p("attended_school_value_added", 1))
    print(f"Extracted {len(r)} published estimates; exploratory two-sign normal-approximation p={p_both:.6f}.")

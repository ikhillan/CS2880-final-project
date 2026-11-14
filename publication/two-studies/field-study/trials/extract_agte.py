"""Read the published pooled coefficients from Agte et al.'s March 2026 working paper."""
import argparse
import csv
import hashlib
import re
import subprocess
from pathlib import Path

SOURCE_URL = "https://www.nber.org/system/files/working_papers/w32670/w32670.pdf"
SOURCE_SHA256 = "c4cd883a5f9bce91caaad931e3a1cbc9fc78f2c556f362d31bf0e61db8417ea5"
SOURCE_PAGES = {"I": 33, "II": 35, "A.XVIII": 86}
FIELDS = ("table", "panel", "pdf_page", "arm", "estimand", "column", "outcome",
          "unit", "estimate", "se", "control_mean", "n")

def row(lines, label):
    matches = [i for i, line in enumerate(lines) if line.strip().startswith(label)]
    if len(matches) != 1:
        raise ValueError(f"Expected one row {label!r}, found {len(matches)}")
    return matches[0]

def numbers(line):
    # All reported means and coefficients have three decimals. For counts there
    # are no decimals. The caller discards the textual stub before parsing.
    return re.findall(r"-?\d+(?:\.\d+)?", line)

def panel(pdf, table):
    page = SOURCE_PAGES[table]
    text = subprocess.check_output([
        "pdftotext", "-layout", "-f", str(page), "-l", str(page), str(pdf), "-"
    ], text=True)
    assert re.search(r"\bTABLE " + re.escape(table) + r"\s*\n", text), table
    return text.split("Panel A: Pooled")[1].split("Panel B:")[0].splitlines()

def extract(pdf, output):
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == SOURCE_SHA256, \
        "Source PDF differs from the March 2026 revision. Verify the tables before updating the hash."
    outcomes = {
        "I": [
            (6, "enrolled_highlight_worthy", "probability"),
            (7, "enrolled_school_value_added", "SD"),
            (8, "enrolled_school_distance", "km")],
        "A.XVIII": [
            (1, "submitted_application", "probability"), (2, "application_length", "programs"),
            (3, "first_ranked_highlight_worthy", "probability"), (7, "second_ranked_highlight_worthy", "probability"),
            (11, "not_assigned", "probability"), (12, "enrolled_at_assigned_school", "probability")],
        "II": [
            (1, "opened_feedback", "probability"), (4, "added_school", "probability"),
            (5, "changed_first_school", "probability"), (6, "deleted_school", "probability"),
            (7, "enrolled_highlight_worthy", "probability"), (8, "enrolled_school_value_added", "SD"),
            (9, "enrolled_school_distance", "km")]
    }
    records = []
    for table, variables in outcomes.items():
        data = panel(pdf, table)
        means = numbers(data[row(data, "Control Group Mean")])
        sizes = numbers(data[row(data, "Observations")])
        if table == "II":
            labels = [("Feedback Treatment", "feedback_invited", "first_stage"),
                      ("Open Feedback", "feedback_opened", "IV")]
        else:
            labels = [("Treatment 1", "aggregate", "assigned_vs_control"),
                      ("Treatment 2", "aggregate_and_highlights", "assigned_vs_control")]
        for label, arm, estimand in labels:
            i = row(data, label)
            estimate = numbers(data[i].strip()[len(label):])
            se = numbers(data[i+1])
            for col, var, unit in variables:
                if table == "II" and ((col == 1) != (label == "Feedback Treatment")):
                    continue
                idx = col-2 if label == "Open Feedback" else col-1
                records.append([table,"A",SOURCE_PAGES[table],arm,estimand,col,var,unit,
                                estimate[idx], se[idx], means[col-1], sizes[col-1]])
    assert len(records) == 25
    with output.open("w",newline="") as f:
        writer = csv.writer(f,lineterminator="\n")
        writer.writerow(FIELDS)
        writer.writerows(records)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    extract(args.pdf,args.out)

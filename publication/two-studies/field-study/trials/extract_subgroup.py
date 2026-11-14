"""Read the maternal-education comparison from Panel B of Agte et al., Table I."""
import argparse
import csv
import hashlib
import subprocess
from pathlib import Path

from extract_agte import SOURCE_SHA256, numbers, row


def extract(pdf, output):
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Expected the March 2026 revision.")
    text = subprocess.check_output(
        ["pdftotext", "-layout", "-f", "33", "-l", "33", pdf, "-"], text=True
    )
    lines = text.split("Panel B: Heterogeneity by Parental Education")[1].splitlines()

    def cell(label, offset=0):
        index = row(lines, label)
        line = lines[index + offset].strip()
        values = numbers(line if offset else line[len(label):])
        if len(values) != 8:
            raise ValueError(f"Expected eight estimates or counts after {label}")
        return values[5]

    equality = cell("p-value: Treat 2 x High SES")
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["mother_completed_college", "highlight_minus_map", "se",
                         "n", "p_high_equals_low"])
        for high, group, number in [(True, "High", 1), (False, "Low", 2)]:
            label = f"Treatment 2 × {group} SES"
            writer.writerow([int(high), cell(label), cell(label, 1),
                             cell(f"Observations {number} ({group} SES)"), equality])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    extract(args.pdf, args.out)

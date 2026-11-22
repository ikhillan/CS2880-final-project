"""Inventory ministry CSV column names without extracting applicant records."""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

SAE_2021_SHA256 = "8bd459562803c6a57e60e51ecf8198beb212100400e8cdd058cbcfbf1aa29261"


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect(archive):
    sha256 = file_digest(archive)
    if sha256 != SAE_2021_SHA256:
        raise ValueError("Expected the audited ministry archive for application year 2021.")

    members = subprocess.check_output(["bsdtar", "-tf", archive], text=True).splitlines()
    result = []
    for member in members:
        if not member.lower().endswith(".csv"):
            continue
        with subprocess.Popen(
            ["bsdtar", "-xOf", archive, member],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ) as child:
            first = child.stdout.readline().decode("utf-8-sig")
            child.stdout.close()
        header = next(csv.reader([first], delimiter=";"))
        if len(header) < 2 or any(name.isdigit() for name in header):
            raise ValueError(f"Could not identify a CSV header in {member}")
        result.append({"file": member, "columns": header})
    return {"source_archive_sha256": sha256, "tables": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = inspect(args.archive)
    args.output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")

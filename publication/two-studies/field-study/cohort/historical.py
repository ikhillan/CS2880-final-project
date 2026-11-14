"""Use the existing offer-blind all-report builder and independent D1 comparator for 2021."""
import argparse
from pathlib import Path
import sys

def main(args):
    sys.path.insert(0,str(args.publication/"two-studies/behavior/field"))
    sys.path.insert(0,str(args.publication/"two-studies/institutional"))
    from round_link import RELEASES
    import prepare_unabridged
    import compare_official
    RELEASES[2021] = "SAE_2021.rar"
    if args.action == "prepare":
        prepare_unabridged.prepare(args)
    else:
        args.report = "all_submitted"
        compare_official.main(args)

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare","compare"))
    for name in ("publication","archives","out"):
        p.add_argument("--"+name, required=True, type=Path)
    p.add_argument("--frozen", type=Path)
    p.add_argument("--programs", type=Path)
    p.set_defaults(year=2021)
    main(p.parse_args())

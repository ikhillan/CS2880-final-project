"""Regenerate the development inputs consumed by the team's feature/protocol pipeline.

Requires the local raw FirstMover June-August 2026 CSVs and the public 2020-24
ACS NY housing ZIP. No recommendations, evaluation outcomes or held-out listings
are accessed. The original loader defines income groups using all cash renters.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pandas as pd


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--repo', type=Path, required=True)
    cli.add_argument('--housing-zip', type=Path, required=True)
    cli.add_argument('--raw-listings', type=Path, required=True)
    cli.add_argument('--report', type=Path, required=True)
    args = cli.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / 'scripts'))
    from hiddenoptions import data
    import firstmover_clean as cleaning
    import common

    housing_sha = sha(args.housing_zip)
    assert housing_sha == '393444456bfaf6bf935920852861979253b0a7a79c9eeb8ec0e2a7053c6d478d'
    report = {'housing_archive_sha256': housing_sha, 'listing_catalogues': {}}
    # Project on exactly the columns selected by build_renter_profiles; its
    # assignments, row ordering and precision match a full-width NYC extract.
    cols = [c for c in data.PUMS_COLUMNS if c != 'NYC_BOROUGH']
    nrows, pumas = 0, set()
    with zipfile.ZipFile(args.housing_zip) as z, z.open('psam_h36.csv') as src:
        chunks = pd.read_csv(src, usecols=cols, dtype=str, keep_default_na=False, chunksize=50000)
        tmp = data.PUMS_HOUSEHOLDS.with_suffix('.tmp')
        with tmp.open('w') as sink:
            for chunk in chunks:
                borough = chunk.PUMA.str[:3].map(common.BOROUGH_BY_PUMA_PREFIX)
                nyc = chunk[borough.notna()].copy()
                nyc['NYC_BOROUGH'] = borough[borough.notna()]
                nyc.to_csv(sink, index=False, header=(nrows == 0), lineterminator='\n')
                nrows += len(nyc)
                pumas.update(nyc.PUMA.unique())
        assert len(pumas) == 55 and nrows > 0
        tmp.replace(data.PUMS_HOUSEHOLDS)
    profiles = data.load_renters(refresh=True)
    movers = profiles[profiles.recent_mover]
    report.update(nyc_households=nrows, cash_renters=len(profiles),
                  recent_movers=len(movers), calibration_recent_movers=int((movers.donors=='calibration').sum()),
                  median_income_july=data.lower_income_cutoff(profiles), pumas=len(pumas))

    cleaning.RAW_DIR = args.raw_listings.resolve()
    cleaning.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    for month in data.DEV_MONTHS:
        raw_hash = sha(cleaning.RAW_DIR / (month + '.csv'))
        manifest = repo / 'data/firstmover/raw' / (month + '.manifest.json')
        assert raw_hash == json.loads(manifest.read_text())['sha256'], month
        frame = cleaning.clean(month)
        # CSV.GZ is the normal entry point; the runner does not use parquet.
        out = cleaning.CLEAN_DIR / f'firstmover_listings_{month}.csv.gz'
        tmp = out.with_suffix('.tmp')
        frame.to_csv(tmp, index=False, compression={'method':'gzip', 'mtime':0, 'compresslevel':3})
        tmp.replace(out)
        report['listing_catalogues'][month] = {
            'raw_rows': len(frame), 'raw_sha256': raw_hash,
            'included': int(frame.in_analysis_catalog.sum()),
            'by_borough': data.load_listings(month).borough.value_counts().to_dict()}
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

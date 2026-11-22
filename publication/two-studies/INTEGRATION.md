# Current-paper integration

This subtree was imported with `git archive` from the tracked
`publication/two-studies` tree at commit
`75830f6c4f5d210def18920f92db3a2100c93ca6` of
`ikhillan/CS2880-final-project`. The import contained 607 files and 7,675,253
bytes. Every imported file matched its original Git blob before editing.
`import-provenance.json` records those blob IDs. Ignored data, reply caches,
local logs and build products were not imported.

The current integration adds a jobs section and methods appendix, using four
already released public aggregate tables in `jobs/sources/`. It derives the
anchored-GPT-minus-anchored-random point differences from those tables.
`jobs/rebuild.py` authenticates the sources and rebuilds the manuscript table.
No new jobs experiment has been run. Jobs post-anchor coverage, individual exits and
entrants, rank changes, and paired uncertainty against coordinated random
remain pending the licensed inputs and private caches.

The jobs appendix also states the anchor's displayed-capacity certificate and
a four-applicant counterexample: one permitted edit raises maximum matching
capacity from three to four but reduces DA placement from three to two. The
example fixes a valid maximum-reuse anchor and drop rule; it makes no claim
about every tie choice, empirical frequency, or novelty relative to prior
literature. It is checked by the repository's coordinated-policy test suite.

An additional proposition bounds gross exits and entries by the number of
changed applicant menus, for fixed strict preferences and priorities and
responsive capacities. The proof uses the classical DA rejection argument
and proposal-order independence; it makes no novelty claim or extension to
Chile's endogenous priorities. Combining the edit counts with net placement
changes gives exit outer intervals of 0–459 and 0–308 for the two jobs panels.
These are neither observed transitions nor sharp identified intervals.
`jobs/rebuild.py` reconstructs the counts in `jobs/access-bounds.csv` from
the same authenticated aggregate sources and the documented market sizes.

The completed public GPT rental follow-up is a separate empirical replay,
not a recovery of the missing jobs outcomes. `rental-access-main.tex` and
`rental-access-appendix.tex` report all three original windows and both supply
designs, including all fixed controls, actual replay exits, grouped counts
and anchor tie sensitivity. Its six aggregate sources are authenticated by
`rental-access/provenance.json`; `rental-access/rebuild.py` verifies the 36
placement checks and rebuilds seven tables. The original Haiku analysis
retains its own unconditional coverage and geographic weights. New GPT
coverage instead averages conditional coverage equally across markets and
seed blocks. The follow-up is post-hoc: original results were already known.
Balancing can edit multiple positions, whereas the anchor edits at most one.
The manuscript includes the favorable score controls and the gains over
random on both outcomes; no inherent prediction–placement impossibility is
claimed. Individual input records and cached replies are not copied here.

The imported main manuscript, introduction, design table, discussion, AI-use
record and README are updated to include jobs. Existing rental and Chilean
results retain their own denominators and mechanisms. The paper's title now
names housing, jobs and schools. The compiled `who-gets-a-place.pdf` is the
integrated version; the imported review documents describe the earlier
snapshot and are retained as historical records, not new referee assessments.

From this directory:

```sh
python3 jobs/rebuild.py
python3 rental-access/rebuild.py
TEXINPUTS=.:..: BSTINPUTS=.:..: latexmk -pdf -halt-on-error -outdir=.build main.tex
cp .build/main.pdf who-gets-a-place.pdf
```

The parent `publication/` directory supplies the ACM class and bibliography
style. New build intermediates belong in `.build/` and are excluded by the
local ignore file. Figure rendering for visual review is likewise local.

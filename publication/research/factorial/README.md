# Crossing catalogue and renter sampling

The local design was written after the 20-block pilot and before computing these fresh-seed outcomes. All 40 fresh blocks use four boroughs, three development months and the same ranking model. The crossed design separately changes how an initial inventory is drawn and how renter profiles are selected. Selected lower-income renters must have a feasible option in that inventory. This is a follow-up simulation under common data and model assumptions. It does not amount to random assignment of apartments to actual renters.

The ordinary/ordinary arm reproduces the native generator's ranks under its stream; the other arms change only the input sampling and use the same scoring routines. List construction follows the same rule across arms. A common seeded permutation applied to different numbers of eligible listings need not couple the same specific apartments. For the hybrid using ordinary listings and selected renters, some cell–seed draws have no eligible lower donor. We keep the 72 missing draws, show the available sample per scenario, and report estimates and intervals on the 408 draws supporting all four comparisons. Every original cell remains represented. All group rates pool counts across seeds within each borough-month before averaging with common renormalized weights, with 5,000 paired block-bootstrap samples. Unavailable draws do not supply a fabricated zero for a selected group.

From the extracted archive root with the calibrated inputs prepared, run:
~~~sh
python -B research/factorial/run.py --repo research/implementation --out replay/factorial
~~~
The JSON includes all panels and missing draw keys. The compressed CSV holds integer counts by market, sampling protocol, and quota.

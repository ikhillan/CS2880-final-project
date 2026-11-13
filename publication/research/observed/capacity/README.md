# Matching capacity on the archived rental masks

The scope and estimand were fixed in the companion university/CAPACITY-FOLLOWUP.md after the original policy comparisons. For each of the 2,304 archived rental simulation markets (48 blocks × four boroughs × three months × two designs × two seasons), display the frozen uniform or contextual top-five menus to every applicant. For each graph calculate ν, its maximum bipartite matching size, and A, the cardinality of the applicant-proposing DA matching. The shortfall G is ν−A. Differences between the two policies obey ΔA=Δν−ΔG. This is an accounting identity. An allocation of size ν can violate the original priorities.

The script imports make_mask and da from the earlier learned study, including exact menu lotteries and visibility checks. Each of 4,608 DA cardinalities must equal its archived keyed value. Independent Hopcroft–Karp and maximum-weight complete assignment solvers agree on all 4,608 values of ν. The compressed counts file has one row per market and policy. The summary pools each measure within borough–month and standardizes with the original housing weights, then reports paired percentile intervals from 5,000 bootstrap samples of the 48 blocks per season. Units are percentages among all applicants per market, or percentage points for the contrast.

For the archive replay, point --data to the extracted research/ai/panels directory, --learned to the extracted research/ai/learned directory, and --out to a new output folder. For example, from this script’s directory, replacing the paths with actual locations:

~~~sh
python -B nyc_capacity.py --data /path/to/research/ai/panels --learned /path/to/research/ai/learned --out replay
~~~

The script also accepts the locally prepared data, scores and market keys under work/ec-ai-learn via --data. If the scores do not reside in --data the script loads them from the predictions subdirectory of --learned. The original masks and trained weights remain in the earlier evidence directory. The population weights and month labels are read from each split's original JSON metadata.

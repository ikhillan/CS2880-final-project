# Results: stage1_pilot

480 markets with results; market status counts: {'ok': 480, 'unestimable': 120}. Every returned plan passed its full-market (and protection) check.

## Audit methods

| variant | n | exposure | method | markets | cost | certified | seconds_median | seconds_max | n_firms |
|---|---|---|---|---|---|---|---|---|---|
| enriched | 64 | claude_k5 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 106.662 |
| enriched | 64 | expensive_first_k5 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 106.662 |
| enriched | 64 | expensive_first_q40 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 106.662 |
| enriched | 64 | full | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 106.662 |
| enriched | 64 | transparent_k5 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 106.662 |
| enriched | 64 | transparent_q40 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 106.662 |
| enriched | 64 | uniform_k5 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 106.662 |
| enriched | 64 | uniform_q40 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 106.662 |
| primary | 64 | claude_k5 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 106.454 |
| primary | 64 | expensive_first_k5 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 106.454 |
| primary | 64 | expensive_first_q40 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 106.454 |
| primary | 64 | full | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 106.454 |
| primary | 64 | transparent_k5 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 106.454 |
| primary | 64 | transparent_q40 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 106.454 |
| primary | 64 | uniform_k5 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 106.454 |
| primary | 64 | uniform_q40 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 106.454 |

## Primary contrast T: claude_k5 vs uniform_k5

The primary endpoint includes all renters. The conditional endpoint restricts to renters whose shortlist can omit an option.

| variant | n | renters | coverage | supported_cells | paired_markets | unpaired_markets | valid_bootstrap_draws | B_lower_income_claude_k5 | B_higher_income_claude_k5 | B_lower_income_uniform_k5 | B_higher_income_uniform_k5 | T_pp | T_lo_pp | T_hi_pp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| enriched | 64 | all | 0.979 | 12 | 240 | 0 | 10000 | 0.413 | 0.78 | 0.37 | 0.852 | 11.461 | 9.985 | 12.93 |
| enriched | 64 | can_hide only | 0.979 | 12 | 240 | 0 | 10000 | 0.655 | 0.795 | 0.586 | 0.869 | 14.161 | 11.967 | 16.304 |
| enriched | 64 | all, 2026-06 | 0.979 | 4 | 80 | 0 | 10000 | 0.408 | 0.795 | 0.363 | 0.857 | 10.709 | 7.077 | 14.249 |
| enriched | 64 | all, 2026-07 | 0.979 | 4 | 80 | 0 | 10000 | 0.394 | 0.763 | 0.359 | 0.849 | 12.036 | 7.843 | 16.41 |
| enriched | 64 | all, 2026-08 | 0.979 | 4 | 80 | 0 | 10000 | 0.438 | 0.782 | 0.389 | 0.849 | 11.637 | 9.14 | 14.217 |
| primary | 64 | all | 0.979 | 12 | 240 | 0 | 10000 | 0.005 | 0.684 | 0.004 | 0.736 | 5.362 | 4.113 | 6.516 |
| primary | 64 | can_hide only | 0.303 | 6 | 240 | 0 | 9584 | 0.169 | 0.722 | 0.191 | 0.813 | 6.86 |  |  |
| primary | 64 | all, 2026-06 | 0.979 | 4 | 80 | 0 | 10000 | 0.005 | 0.693 | 0.005 | 0.737 | 4.404 | 2.844 | 6.011 |
| primary | 64 | all, 2026-07 | 0.979 | 4 | 80 | 0 | 10000 | 0.006 | 0.667 | 0.004 | 0.727 | 6.186 | 4.547 | 7.781 |
| primary | 64 | all, 2026-08 | 0.979 | 4 | 80 | 0 | 10000 | 0.004 | 0.691 | 0.003 | 0.745 | 5.497 | 3.491 | 7.552 |

Coverage below 1 means some cells (e.g. Staten Island) had no observations, so the estimate is renormalized over cells with positive denominators in both groups and both arms. Unpaired markets are excluded; including a cell in the analysis requires paired observations. Intervals use fixed cell support. A blank interval with invalid bootstrap draws indicates a lost denominator.

## Group-switch decomposition (A = control, B = treated)

| switch | variant | n | coverage | supported_cells | valid_bootstrap_draws | Z_UU | Z_CU | Z_UC | Z_CC | B_lower_UU | B_higher_UU | B_lower_CU | B_higher_CU | B_lower_UC | B_higher_UC | B_lower_CC | B_higher_CC | S_L | S_L_lo | S_L_hi | S_H | S_H_lo | S_H_hi | T | S_L_plus_S_H_equals_T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude_vs_uniform | enriched | 64 | 0.979 | 12 | 10000 | -0.482 | -0.475 | -0.386 | -0.367 | 0.37 | 0.852 | 0.374 | 0.85 | 0.387 | 0.773 | 0.413 | 0.78 | 0.013 | 0.002 | 0.023 | 0.102 | 0.092 | 0.111 | 0.115 | True |
| claude_vs_uniform | primary | 64 | 0.979 | 12 | 10000 | -0.733 | -0.731 | -0.679 | -0.679 | 0.004 | 0.736 | 0.005 | 0.737 | 0.005 | 0.684 | 0.005 | 0.684 | 0.0 | -0.001 | 0.001 | 0.053 | 0.041 | 0.065 | 0.054 | True |
| transparent_vs_uniform_k5 | enriched | 64 | 0.979 | 12 | 10000 | -0.482 | -0.406 | -0.419 | -0.296 | 0.37 | 0.852 | 0.445 | 0.851 | 0.389 | 0.807 | 0.525 | 0.821 | 0.099 | 0.087 | 0.111 | 0.087 | 0.072 | 0.101 | 0.186 | True |
| transparent_vs_uniform_k5 | primary | 64 | 0.979 | 12 | 10000 | -0.733 | -0.73 | -0.727 | -0.724 | 0.004 | 0.736 | 0.006 | 0.736 | 0.005 | 0.732 | 0.01 | 0.734 | 0.003 | 0.001 | 0.004 | 0.006 | -0.005 | 0.018 | 0.009 | True |
| transparent_vs_uniform_q40 | enriched | 64 | 0.979 | 12 | 10000 | -0.307 | -0.218 | -0.178 | -0.043 | 0.27 | 0.578 | 0.361 | 0.579 | 0.257 | 0.435 | 0.414 | 0.458 | 0.112 | 0.101 | 0.123 | 0.152 | 0.127 | 0.177 | 0.264 | True |
| transparent_vs_uniform_q40 | primary | 64 | 0.979 | 12 | 10000 | -0.495 | -0.494 | -0.37 | -0.367 | 0.011 | 0.506 | 0.014 | 0.507 | 0.008 | 0.378 | 0.013 | 0.38 | 0.002 | 0.0 | 0.004 | 0.126 | 0.11 | 0.141 | 0.128 | True |

## Lower vs. higher income (raw market averages)

| variant | n | exposure | method | group | no_feasible_listing | blocker_rate | can_hide_share | price_percentile_shown | match_before | match_after | worse_off | unavoidable_loss | severity_positive | severity_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| enriched | 64 | claude_k5 | no_audit | higher_income | 0.006 | 0.773 | 0.976 | 0.481 | 0.868 | 0.868 | 0.0 | 0.204 | 0.624 | 10.134 |
| enriched | 64 | claude_k5 | no_audit | lower_income | 0.0 | 0.413 | 0.637 | 0.495 | 0.352 | 0.352 | 0.0 | 0.239 | 0.211 | 0.333 |
| enriched | 64 | expensive_first_k5 | no_audit | higher_income | 0.006 | 0.954 | 0.976 | 0.935 | 0.759 | 0.759 | 0.0 | 0.056 | 0.893 | 26.356 |
| enriched | 64 | expensive_first_k5 | no_audit | lower_income | 0.0 | 0.529 | 0.637 | 0.72 | 0.546 | 0.546 | 0.0 | 0.424 | 0.239 | 0.659 |
| enriched | 64 | expensive_first_q40 | no_audit | higher_income | 0.006 | 0.887 | 0.991 | 0.799 | 0.95 | 0.95 | 0.0 | 0.064 | 0.749 | 9.253 |
| enriched | 64 | expensive_first_q40 | no_audit | lower_income | 0.0 | 0.68 | 0.836 | 0.766 | 0.544 | 0.544 | 0.0 | 0.393 | 0.231 | 0.368 |
| enriched | 64 | full | no_audit | higher_income | 0.006 | 0.0 | 0.0 | 0.5 | 0.987 | 0.987 | 0.0 | 0.0 | 0.0 | 0.0 |
| enriched | 64 | full | no_audit | lower_income | 0.0 | 0.0 | 0.0 | 0.5 | 0.374 | 0.374 | 0.0 | 0.0 | 0.0 | 0.0 |
| enriched | 64 | transparent_k5 | no_audit | higher_income | 0.006 | 0.816 | 0.976 | 0.347 | 0.544 | 0.544 | 0.0 | 0.111 | 0.716 | 19.785 |
| enriched | 64 | transparent_k5 | no_audit | lower_income | 0.0 | 0.531 | 0.637 | 0.388 | 0.19 | 0.19 | 0.0 | 0.116 | 0.276 | 2.587 |
| enriched | 64 | transparent_q40 | no_audit | higher_income | 0.006 | 0.453 | 0.991 | 0.366 | 0.922 | 0.922 | 0.0 | 0.14 | 0.398 | 3.074 |
| enriched | 64 | transparent_q40 | no_audit | lower_income | 0.0 | 0.421 | 0.836 | 0.379 | 0.205 | 0.205 | 0.0 | 0.068 | 0.235 | 1.647 |
| enriched | 64 | uniform_k5 | no_audit | higher_income | 0.006 | 0.847 | 0.976 | 0.5 | 0.964 | 0.964 | 0.0 | 0.181 | 0.716 | 9.782 |
| enriched | 64 | uniform_k5 | no_audit | lower_income | 0.0 | 0.382 | 0.637 | 0.5 | 0.373 | 0.373 | 0.0 | 0.213 | 0.218 | 0.656 |
| enriched | 64 | uniform_q40 | no_audit | higher_income | 0.006 | 0.575 | 0.991 | 0.501 | 0.98 | 0.98 | 0.0 | 0.229 | 0.5 | 2.204 |
| enriched | 64 | uniform_q40 | no_audit | lower_income | 0.0 | 0.28 | 0.836 | 0.499 | 0.359 | 0.359 | 0.0 | 0.154 | 0.197 | 0.505 |
| primary | 64 | claude_k5 | no_audit | higher_income | 0.01 | 0.69 | 0.927 | 0.445 | 0.83 | 0.83 | 0.0 | 0.193 | 0.563 | 6.882 |
| primary | 64 | claude_k5 | no_audit | lower_income | 0.84 | 0.01 | 0.033 | 0.499 | 0.029 | 0.029 | 0.0 | 0.017 | 0.007 | -0.022 |
| primary | 64 | expensive_first_k5 | no_audit | higher_income | 0.01 | 0.861 | 0.927 | 0.895 | 0.836 | 0.836 | 0.0 | 0.189 | 0.723 | 17.939 |
| primary | 64 | expensive_first_k5 | no_audit | lower_income | 0.84 | 0.018 | 0.033 | 0.538 | 0.078 | 0.078 | 0.0 | 0.068 | 0.005 | -0.193 |
| primary | 64 | expensive_first_q40 | no_audit | higher_income | 0.01 | 0.873 | 0.977 | 0.802 | 0.946 | 0.946 | 0.0 | 0.194 | 0.62 | 5.783 |
| primary | 64 | expensive_first_q40 | no_audit | lower_income | 0.84 | 0.071 | 0.099 | 0.703 | 0.083 | 0.083 | 0.0 | 0.072 | 0.008 | -0.139 |
| primary | 64 | full | no_audit | higher_income | 0.01 | 0.0 | 0.0 | 0.5 | 0.89 | 0.89 | 0.0 | 0.0 | 0.0 | 0.0 |
| primary | 64 | full | no_audit | lower_income | 0.84 | 0.0 | 0.0 | 0.5 | 0.02 | 0.02 | 0.0 | 0.0 | 0.0 | 0.0 |
| primary | 64 | transparent_k5 | no_audit | higher_income | 0.01 | 0.733 | 0.927 | 0.337 | 0.537 | 0.537 | 0.0 | 0.094 | 0.633 | 15.066 |
| primary | 64 | transparent_k5 | no_audit | lower_income | 0.84 | 0.019 | 0.033 | 0.484 | 0.02 | 0.02 | 0.0 | 0.01 | 0.01 | 0.01 |
| primary | 64 | transparent_q40 | no_audit | higher_income | 0.01 | 0.363 | 0.977 | 0.351 | 0.8 | 0.8 | 0.0 | 0.083 | 0.351 | 1.952 |
| primary | 64 | transparent_q40 | no_audit | lower_income | 0.84 | 0.023 | 0.099 | 0.428 | 0.011 | 0.011 | 0.0 | 0.002 | 0.012 | 0.034 |
| primary | 64 | uniform_k5 | no_audit | higher_income | 0.01 | 0.752 | 0.927 | 0.499 | 0.884 | 0.884 | 0.0 | 0.179 | 0.638 | 7.372 |
| primary | 64 | uniform_k5 | no_audit | lower_income | 0.84 | 0.008 | 0.033 | 0.501 | 0.032 | 0.032 | 0.0 | 0.02 | 0.007 | -0.041 |
| primary | 64 | uniform_q40 | no_audit | higher_income | 0.01 | 0.517 | 0.977 | 0.5 | 0.877 | 0.877 | 0.0 | 0.195 | 0.454 | 1.784 |
| primary | 64 | uniform_q40 | no_audit | lower_income | 0.84 | 0.019 | 0.099 | 0.493 | 0.02 | 0.02 | 0.0 | 0.011 | 0.012 | 0.01 |

## Scarcity categories (raw market averages)

| variant | n | exposure | group | scarcity_no_option_in_cohort | scarcity_none_in_draw | scarcity_unmatched_full | scarcity_matched_full |
|---|---|---|---|---|---|---|---|
| enriched | 64 | claude_k5 | higher_income | 0.001 | 0.005 | 0.007 | 0.987 |
| enriched | 64 | claude_k5 | lower_income | 0.0 | 0.0 | 0.626 | 0.374 |
| primary | 64 | claude_k5 | higher_income | 0.001 | 0.009 | 0.1 | 0.89 |
| primary | 64 | claude_k5 | lower_income | 0.579 | 0.262 | 0.139 | 0.02 |

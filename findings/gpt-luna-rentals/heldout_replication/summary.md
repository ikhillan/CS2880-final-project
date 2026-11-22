# Results: heldout_replication

480 markets with results; market status counts: {'ok': 480, 'unestimable': 120}. Every returned plan passed its full-market (and protection) check.

## Audit methods

| variant | n | exposure | method | markets | cost | certified | seconds_median | seconds_max | n_firms |
|---|---|---|---|---|---|---|---|---|---|
| enriched | 64 | claude_k5 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 107.358 |
| enriched | 64 | full | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 107.358 |
| enriched | 64 | gpt_k5 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 107.358 |
| enriched | 64 | transparent_k5 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 107.358 |
| enriched | 64 | uniform_k5 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 107.358 |
| primary | 64 | claude_k5 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 105.842 |
| primary | 64 | full | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 105.842 |
| primary | 64 | gpt_k5 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 105.842 |
| primary | 64 | transparent_k5 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 105.842 |
| primary | 64 | uniform_k5 | no_audit | 240 | 0.0 | 0.0 | 0.0 | 0.0 | 105.842 |

## Primary contrast T: gpt_k5 vs uniform_k5

The primary endpoint includes all renters. The conditional endpoint restricts to renters whose shortlist can omit an option.

| variant | n | renters | coverage | supported_cells | paired_markets | unpaired_markets | valid_bootstrap_draws | B_lower_income_gpt_k5 | B_higher_income_gpt_k5 | B_lower_income_uniform_k5 | B_higher_income_uniform_k5 | T_pp | T_lo_pp | T_hi_pp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| enriched | 64 | all | 0.979 | 12 | 240 | 0 | 10000 | 0.441 | 0.81 | 0.397 | 0.846 | 8.071 | 6.03 | 10.173 |
| enriched | 64 | can_hide only | 0.979 | 12 | 240 | 0 | 10000 | 0.692 | 0.831 | 0.622 | 0.869 | 10.805 | 8.167 | 13.454 |
| enriched | 64 | all, 2025-06 | 0.979 | 4 | 80 | 0 | 10000 | 0.455 | 0.797 | 0.431 | 0.85 | 7.746 | 4.417 | 10.809 |
| enriched | 64 | all, 2025-07 | 0.979 | 4 | 80 | 0 | 10000 | 0.418 | 0.813 | 0.373 | 0.84 | 7.122 | 4.664 | 9.68 |
| enriched | 64 | all, 2025-08 | 0.979 | 4 | 80 | 0 | 10000 | 0.451 | 0.82 | 0.387 | 0.849 | 9.345 | 6.432 | 12.508 |
| primary | 64 | all | 0.979 | 12 | 240 | 0 | 10000 | 0.006 | 0.709 | 0.005 | 0.752 | 4.363 | 3.635 | 5.129 |
| primary | 64 | can_hide only | 0.409 | 7 | 240 | 0 | 6372 | 0.203 | 0.741 | 0.168 | 0.796 | 9.092 |  |  |
| primary | 64 | all, 2025-06 | 0.979 | 4 | 80 | 0 | 10000 | 0.007 | 0.707 | 0.007 | 0.764 | 5.682 | 4.346 | 7.029 |
| primary | 64 | all, 2025-07 | 0.979 | 4 | 80 | 0 | 10000 | 0.006 | 0.704 | 0.006 | 0.75 | 4.686 | 2.777 | 6.731 |
| primary | 64 | all, 2025-08 | 0.979 | 4 | 80 | 0 | 10000 | 0.003 | 0.716 | 0.002 | 0.742 | 2.721 | 0.678 | 4.713 |

Coverage below 1 means some cells (e.g. Staten Island) had no observations, so the estimate is renormalized over cells with positive denominators in both groups and both arms. Unpaired markets are excluded; including a cell in the analysis requires paired observations. Intervals use fixed cell support. A blank interval with invalid bootstrap draws indicates a lost denominator.

## Group-switch decomposition (A = control, B = treated)

| switch | variant | n | coverage | supported_cells | valid_bootstrap_draws | Z_UU | Z_CU | Z_UC | Z_CC | B_lower_UU | B_higher_UU | B_lower_CU | B_higher_CU | B_lower_UC | B_higher_UC | B_lower_CC | B_higher_CC | S_L | S_L_lo | S_L_hi | S_H | S_H_lo | S_H_hi | T | S_L_plus_S_H_equals_T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude_vs_uniform | enriched | 64 | 0.979 | 12 | 10000 | -0.449 | -0.436 | -0.372 | -0.343 | 0.397 | 0.846 | 0.413 | 0.848 | 0.415 | 0.787 | 0.451 | 0.795 | 0.021 | 0.011 | 0.031 | 0.085 | 0.068 | 0.103 | 0.106 | True |
| claude_vs_uniform | primary | 64 | 0.979 | 12 | 10000 | -0.747 | -0.747 | -0.699 | -0.697 | 0.005 | 0.752 | 0.005 | 0.752 | 0.004 | 0.703 | 0.005 | 0.702 | 0.001 | -0.0 | 0.002 | 0.049 | 0.04 | 0.059 | 0.05 | True |
| gpt_vs_uniform | enriched | 64 | 0.979 | 12 | 10000 | -0.449 | -0.439 | -0.385 | -0.369 | 0.397 | 0.846 | 0.409 | 0.848 | 0.418 | 0.804 | 0.441 | 0.81 | 0.014 | 0.002 | 0.024 | 0.067 | 0.052 | 0.082 | 0.081 | True |
| gpt_vs_uniform | primary | 64 | 0.979 | 12 | 10000 | -0.747 | -0.747 | -0.705 | -0.704 | 0.005 | 0.752 | 0.005 | 0.752 | 0.005 | 0.709 | 0.006 | 0.709 | 0.001 | -0.001 | 0.002 | 0.043 | 0.036 | 0.051 | 0.044 | True |
| transparent_vs_uniform_k5 | enriched | 64 | 0.979 | 12 | 10000 | -0.449 | -0.397 | -0.39 | -0.298 | 0.397 | 0.846 | 0.448 | 0.845 | 0.415 | 0.806 | 0.522 | 0.821 | 0.072 | 0.062 | 0.083 | 0.079 | 0.065 | 0.094 | 0.151 | True |
| transparent_vs_uniform_k5 | primary | 64 | 0.979 | 12 | 10000 | -0.747 | -0.746 | -0.741 | -0.737 | 0.005 | 0.752 | 0.006 | 0.752 | 0.004 | 0.746 | 0.011 | 0.747 | 0.003 | 0.001 | 0.005 | 0.008 | -0.001 | 0.017 | 0.011 | True |

## Lower vs. higher income (raw market averages)

| variant | n | exposure | method | group | no_feasible_listing | blocker_rate | can_hide_share | price_percentile_shown | match_before | match_after | worse_off | unavoidable_loss | severity_positive | severity_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| enriched | 64 | claude_k5 | no_audit | higher_income | 0.01 | 0.77 | 0.964 | 0.486 | 0.86 | 0.86 | 0.0 | 0.193 | 0.626 | 10.238 |
| enriched | 64 | claude_k5 | no_audit | lower_income | 0.0 | 0.423 | 0.626 | 0.494 | 0.339 | 0.339 | 0.0 | 0.221 | 0.231 | 0.871 |
| enriched | 64 | full | no_audit | higher_income | 0.01 | 0.0 | 0.0 | 0.5 | 0.973 | 0.973 | 0.0 | 0.0 | 0.0 | 0.0 |
| enriched | 64 | full | no_audit | lower_income | 0.0 | 0.0 | 0.0 | 0.5 | 0.389 | 0.389 | 0.0 | 0.0 | 0.0 | 0.0 |
| enriched | 64 | gpt_k5 | no_audit | higher_income | 0.01 | 0.786 | 0.964 | 0.569 | 0.893 | 0.893 | 0.0 | 0.2 | 0.63 | 9.517 |
| enriched | 64 | gpt_k5 | no_audit | lower_income | 0.0 | 0.416 | 0.626 | 0.498 | 0.377 | 0.377 | 0.0 | 0.254 | 0.21 | 0.25 |
| enriched | 64 | transparent_k5 | no_audit | higher_income | 0.01 | 0.805 | 0.964 | 0.333 | 0.531 | 0.531 | 0.0 | 0.101 | 0.709 | 19.504 |
| enriched | 64 | transparent_k5 | no_audit | lower_income | 0.0 | 0.507 | 0.626 | 0.384 | 0.188 | 0.188 | 0.0 | 0.108 | 0.291 | 2.967 |
| enriched | 64 | uniform_k5 | no_audit | higher_income | 0.01 | 0.833 | 0.964 | 0.499 | 0.953 | 0.953 | 0.0 | 0.173 | 0.709 | 9.589 |
| enriched | 64 | uniform_k5 | no_audit | lower_income | 0.0 | 0.378 | 0.626 | 0.5 | 0.382 | 0.382 | 0.0 | 0.206 | 0.234 | 0.906 |
| primary | 64 | claude_k5 | no_audit | higher_income | 0.012 | 0.694 | 0.928 | 0.441 | 0.814 | 0.814 | 0.0 | 0.202 | 0.565 | 7.265 |
| primary | 64 | claude_k5 | no_audit | lower_income | 0.834 | 0.009 | 0.031 | 0.501 | 0.027 | 0.027 | 0.0 | 0.017 | 0.006 | -0.031 |
| primary | 64 | full | no_audit | higher_income | 0.012 | 0.0 | 0.0 | 0.5 | 0.889 | 0.889 | 0.0 | 0.0 | 0.0 | 0.0 |
| primary | 64 | full | no_audit | lower_income | 0.834 | 0.0 | 0.0 | 0.5 | 0.02 | 0.02 | 0.0 | 0.0 | 0.0 | 0.0 |
| primary | 64 | gpt_k5 | no_audit | higher_income | 0.012 | 0.702 | 0.928 | 0.491 | 0.845 | 0.845 | 0.0 | 0.22 | 0.555 | 6.588 |
| primary | 64 | gpt_k5 | no_audit | lower_income | 0.834 | 0.01 | 0.031 | 0.5 | 0.029 | 0.029 | 0.0 | 0.017 | 0.006 | -0.033 |
| primary | 64 | transparent_k5 | no_audit | higher_income | 0.012 | 0.737 | 0.928 | 0.332 | 0.519 | 0.519 | 0.0 | 0.101 | 0.625 | 15.253 |
| primary | 64 | transparent_k5 | no_audit | lower_income | 0.834 | 0.018 | 0.031 | 0.483 | 0.016 | 0.016 | 0.0 | 0.008 | 0.011 | 0.025 |
| primary | 64 | uniform_k5 | no_audit | higher_income | 0.012 | 0.754 | 0.928 | 0.499 | 0.887 | 0.887 | 0.0 | 0.186 | 0.635 | 7.434 |
| primary | 64 | uniform_k5 | no_audit | lower_income | 0.834 | 0.009 | 0.031 | 0.499 | 0.03 | 0.03 | 0.0 | 0.019 | 0.007 | -0.033 |

## Scarcity categories (raw market averages)

| variant | n | exposure | group | scarcity_no_option_in_cohort | scarcity_none_in_draw | scarcity_unmatched_full | scarcity_matched_full |
|---|---|---|---|---|---|---|---|
| enriched | 64 | claude_k5 | higher_income | 0.002 | 0.008 | 0.017 | 0.973 |
| enriched | 64 | claude_k5 | lower_income | 0.0 | 0.0 | 0.611 | 0.389 |
| primary | 64 | claude_k5 | higher_income | 0.002 | 0.011 | 0.098 | 0.889 |
| primary | 64 | claude_k5 | lower_income | 0.595 | 0.24 | 0.146 | 0.02 |

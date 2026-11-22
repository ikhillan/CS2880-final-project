# Results: heldout_primary

320 markets with results; market status counts: {'ok': 320, 'unestimable': 80}. Every returned plan passed its full-market (and protection) check.

## Audit methods

| variant | n | exposure | method | markets | cost | certified | seconds_median | seconds_max | n_firms |
|---|---|---|---|---|---|---|---|---|---|
| enriched | 64 | claude_k5 | no_audit | 160 | 0.0 | 0.0 | 0.0 | 0.0 | 108.75 |
| enriched | 64 | full | no_audit | 160 | 0.0 | 0.0 | 0.0 | 0.0 | 108.75 |
| enriched | 64 | gpt_k5 | no_audit | 160 | 0.0 | 0.0 | 0.0 | 0.0 | 108.75 |
| enriched | 64 | transparent_k5 | no_audit | 160 | 0.0 | 0.0 | 0.0 | 0.0 | 108.75 |
| enriched | 64 | uniform_k5 | no_audit | 160 | 0.0 | 0.0 | 0.0 | 0.0 | 108.75 |
| primary | 64 | claude_k5 | no_audit | 160 | 0.0 | 0.0 | 0.0 | 0.0 | 105.556 |
| primary | 64 | full | no_audit | 160 | 0.0 | 0.0 | 0.0 | 0.0 | 105.556 |
| primary | 64 | gpt_k5 | no_audit | 160 | 0.0 | 0.0 | 0.0 | 0.0 | 105.556 |
| primary | 64 | transparent_k5 | no_audit | 160 | 0.0 | 0.0 | 0.0 | 0.0 | 105.556 |
| primary | 64 | uniform_k5 | no_audit | 160 | 0.0 | 0.0 | 0.0 | 0.0 | 105.556 |

## Primary contrast T: gpt_k5 vs uniform_k5

The primary endpoint includes all renters. The conditional endpoint restricts to renters whose shortlist can omit an option.

| variant | n | renters | coverage | supported_cells | paired_markets | unpaired_markets | valid_bootstrap_draws | B_lower_income_gpt_k5 | B_higher_income_gpt_k5 | B_lower_income_uniform_k5 | B_higher_income_uniform_k5 | T_pp | T_lo_pp | T_hi_pp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| enriched | 64 | all | 0.979 | 8 | 160 | 0 | 10000 | 0.393 | 0.813 | 0.376 | 0.863 | 6.733 | 4.932 | 8.46 |
| enriched | 64 | can_hide only | 0.979 | 8 | 160 | 0 | 10000 | 0.691 | 0.83 | 0.66 | 0.881 | 8.249 | 5.972 | 10.326 |
| enriched | 64 | all, 2026-09 | 0.979 | 4 | 80 | 0 | 10000 | 0.375 | 0.81 | 0.365 | 0.858 | 5.824 | 3.517 | 8.055 |
| enriched | 64 | all, 2026-10 | 0.979 | 4 | 80 | 0 | 10000 | 0.412 | 0.815 | 0.387 | 0.867 | 7.642 | 4.836 | 10.083 |
| primary | 64 | all | 0.979 | 8 | 160 | 0 | 10000 | 0.007 | 0.725 | 0.005 | 0.759 | 3.574 | 2.075 | 5.031 |
| primary | 64 | can_hide only | 0.462 | 5 | 160 | 0 | 6371 | 0.221 | 0.778 | 0.162 | 0.826 | 10.688 |  |  |
| primary | 64 | all, 2026-09 | 0.979 | 4 | 80 | 0 | 10000 | 0.006 | 0.726 | 0.004 | 0.758 | 3.43 | 1.718 | 5.059 |
| primary | 64 | all, 2026-10 | 0.979 | 4 | 80 | 0 | 10000 | 0.007 | 0.725 | 0.006 | 0.761 | 3.717 | 1.636 | 5.722 |

Coverage below 1 means some cells (e.g. Staten Island) had no observations, so the estimate is renormalized over cells with positive denominators in both groups and both arms. Unpaired markets are excluded; including a cell in the analysis requires paired observations. Intervals use fixed cell support. A blank interval with invalid bootstrap draws indicates a lost denominator.

## Group-switch decomposition (A = control, B = treated)

| switch | variant | n | coverage | supported_cells | valid_bootstrap_draws | Z_UU | Z_CU | Z_UC | Z_CC | B_lower_UU | B_higher_UU | B_lower_CU | B_higher_CU | B_lower_UC | B_higher_UC | B_lower_CC | B_higher_CC | S_L | S_L_lo | S_L_hi | S_H | S_H_lo | S_H_hi | T | S_L_plus_S_H_equals_T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude_vs_uniform | enriched | 64 | 0.979 | 8 | 10000 | -0.487 | -0.497 | -0.403 | -0.406 | 0.376 | 0.863 | 0.369 | 0.866 | 0.385 | 0.789 | 0.398 | 0.804 | -0.007 | -0.015 | 0.001 | 0.087 | 0.069 | 0.106 | 0.08 | True |
| claude_vs_uniform | primary | 64 | 0.979 | 8 | 10000 | -0.755 | -0.755 | -0.714 | -0.713 | 0.005 | 0.759 | 0.005 | 0.759 | 0.006 | 0.72 | 0.007 | 0.72 | 0.001 | -0.001 | 0.003 | 0.041 | 0.028 | 0.054 | 0.042 | True |
| gpt_vs_uniform | enriched | 64 | 0.979 | 8 | 10000 | -0.487 | -0.498 | -0.419 | -0.419 | 0.376 | 0.863 | 0.367 | 0.866 | 0.384 | 0.803 | 0.393 | 0.813 | -0.006 | -0.013 | 0.002 | 0.073 | 0.053 | 0.094 | 0.067 | True |
| gpt_vs_uniform | primary | 64 | 0.979 | 8 | 10000 | -0.755 | -0.755 | -0.72 | -0.719 | 0.005 | 0.759 | 0.004 | 0.759 | 0.005 | 0.725 | 0.007 | 0.725 | 0.0 | -0.001 | 0.002 | 0.035 | 0.02 | 0.05 | 0.036 | True |
| transparent_vs_uniform_k5 | enriched | 64 | 0.979 | 8 | 10000 | -0.487 | -0.448 | -0.427 | -0.35 | 0.376 | 0.863 | 0.414 | 0.862 | 0.384 | 0.812 | 0.478 | 0.829 | 0.058 | 0.049 | 0.066 | 0.079 | 0.065 | 0.092 | 0.136 | True |
| transparent_vs_uniform_k5 | primary | 64 | 0.979 | 8 | 10000 | -0.755 | -0.755 | -0.751 | -0.745 | 0.005 | 0.759 | 0.004 | 0.759 | 0.005 | 0.755 | 0.011 | 0.756 | 0.002 | 0.0 | 0.004 | 0.007 | -0.004 | 0.018 | 0.009 | True |

## Lower vs. higher income (raw market averages)

| variant | n | exposure | method | group | no_feasible_listing | blocker_rate | can_hide_share | price_percentile_shown | match_before | match_after | worse_off | unavoidable_loss | severity_positive | severity_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| enriched | 64 | claude_k5 | no_audit | higher_income | 0.004 | 0.787 | 0.971 | 0.473 | 0.853 | 0.853 | 0.0 | 0.18 | 0.654 | 11.129 |
| enriched | 64 | claude_k5 | no_audit | lower_income | 0.0 | 0.389 | 0.57 | 0.491 | 0.322 | 0.322 | 0.0 | 0.206 | 0.221 | 0.919 |
| enriched | 64 | full | no_audit | higher_income | 0.004 | 0.0 | 0.0 | 0.5 | 0.981 | 0.981 | 0.0 | 0.0 | 0.0 | 0.0 |
| enriched | 64 | full | no_audit | lower_income | 0.0 | 0.0 | 0.0 | 0.5 | 0.381 | 0.381 | 0.0 | 0.0 | 0.0 | 0.0 |
| enriched | 64 | gpt_k5 | no_audit | higher_income | 0.004 | 0.794 | 0.971 | 0.572 | 0.895 | 0.895 | 0.0 | 0.188 | 0.658 | 10.413 |
| enriched | 64 | gpt_k5 | no_audit | lower_income | 0.0 | 0.385 | 0.57 | 0.496 | 0.369 | 0.369 | 0.0 | 0.247 | 0.199 | 0.247 |
| enriched | 64 | transparent_k5 | no_audit | higher_income | 0.004 | 0.816 | 0.971 | 0.321 | 0.521 | 0.521 | 0.0 | 0.107 | 0.719 | 21.207 |
| enriched | 64 | transparent_k5 | no_audit | lower_income | 0.0 | 0.476 | 0.57 | 0.379 | 0.167 | 0.167 | 0.0 | 0.088 | 0.291 | 3.196 |
| enriched | 64 | uniform_k5 | no_audit | higher_income | 0.004 | 0.855 | 0.971 | 0.501 | 0.962 | 0.962 | 0.0 | 0.165 | 0.725 | 10.364 |
| enriched | 64 | uniform_k5 | no_audit | lower_income | 0.0 | 0.365 | 0.57 | 0.5 | 0.376 | 0.376 | 0.0 | 0.203 | 0.226 | 1.044 |
| primary | 64 | claude_k5 | no_audit | higher_income | 0.011 | 0.72 | 0.934 | 0.452 | 0.839 | 0.839 | 0.0 | 0.204 | 0.581 | 7.541 |
| primary | 64 | claude_k5 | no_audit | lower_income | 0.824 | 0.012 | 0.031 | 0.494 | 0.033 | 0.033 | 0.0 | 0.021 | 0.008 | -0.037 |
| primary | 64 | full | no_audit | higher_income | 0.011 | 0.0 | 0.0 | 0.5 | 0.908 | 0.908 | 0.0 | 0.0 | 0.0 | 0.0 |
| primary | 64 | full | no_audit | lower_income | 0.824 | 0.0 | 0.0 | 0.5 | 0.022 | 0.022 | 0.0 | 0.0 | 0.0 | 0.0 |
| primary | 64 | gpt_k5 | no_audit | higher_income | 0.011 | 0.724 | 0.934 | 0.5 | 0.868 | 0.868 | 0.0 | 0.221 | 0.569 | 6.827 |
| primary | 64 | gpt_k5 | no_audit | lower_income | 0.824 | 0.011 | 0.031 | 0.494 | 0.034 | 0.034 | 0.0 | 0.021 | 0.007 | -0.043 |
| primary | 64 | transparent_k5 | no_audit | higher_income | 0.011 | 0.751 | 0.934 | 0.348 | 0.525 | 0.525 | 0.0 | 0.106 | 0.653 | 16.514 |
| primary | 64 | transparent_k5 | no_audit | lower_income | 0.824 | 0.018 | 0.031 | 0.486 | 0.02 | 0.02 | 0.0 | 0.01 | 0.011 | 0.013 |
| primary | 64 | uniform_k5 | no_audit | higher_income | 0.011 | 0.766 | 0.934 | 0.504 | 0.9 | 0.9 | 0.0 | 0.185 | 0.648 | 7.509 |
| primary | 64 | uniform_k5 | no_audit | lower_income | 0.824 | 0.009 | 0.031 | 0.498 | 0.033 | 0.033 | 0.0 | 0.021 | 0.006 | -0.041 |

## Scarcity categories (raw market averages)

| variant | n | exposure | group | scarcity_no_option_in_cohort | scarcity_none_in_draw | scarcity_unmatched_full | scarcity_matched_full |
|---|---|---|---|---|---|---|---|
| enriched | 64 | claude_k5 | higher_income | 0.0 | 0.004 | 0.014 | 0.981 |
| enriched | 64 | claude_k5 | lower_income | 0.0 | 0.0 | 0.619 | 0.381 |
| primary | 64 | claude_k5 | higher_income | 0.0 | 0.011 | 0.081 | 0.908 |
| primary | 64 | claude_k5 | lower_income | 0.572 | 0.253 | 0.153 | 0.022 |

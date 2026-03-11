# Q2 Partial-Identification Report

## Objective
Compare the semester-2 decision `Continue` vs `Opt-out` using partial-identification bounds for BAP and VAI students.

## Data and Setup
- Number of group-outcome rows: **4**
- Treatment definition: `Continue2 = 1[Tut_20162 >= 1]`
- Outcomes: `Prom__20162` and `Tasa_aprob_20162`

## Interval Summary
| Group | Outcome | ATE Lower | ATE Upper | Robust Sign | Optimistic | Conservative |
|---|---|---:|---:|---|---|---|
| BAP_students | Prom__20162 | -3.3955 | 3.7432 | Ambiguous (sign not identified) | Continue | Opt-out |
| BAP_students | Tasa_aprob_20162 | -0.4794 | 0.7104 | Ambiguous (sign not identified) | Continue | Opt-out |
| VAI_students | Prom__20162 | -3.5247 | 3.7873 | Ambiguous (sign not identified) | Continue | Opt-out |
| VAI_students | Tasa_aprob_20162 | -0.5164 | 0.7023 | Ambiguous (sign not identified) | Continue | Opt-out |

## Files Generated
- Tables: `tables/q2_continue_vs_optout_bounds.csv`, `tables/q2_decision_summary.csv`
- Figure: `figures/q2_continue_vs_optout_intervals.png`

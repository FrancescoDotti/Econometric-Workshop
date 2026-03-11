# Q1 Partial-Identification Report

## Objective
Evaluate how the tutoring intensity threshold `k` changes the partial-identification bounds for semester-1 outcomes among BAP students.

## Data and Setup
- Number of evaluated thresholds: **47**
- Treatment definition: `Z_k = 1[Tut_20161 >= k]`
- Outcomes: `Prom__20161` and `Tasa_aprob_20161`

## Main Findings
- Best grade midpoint threshold: **k = 8**
- Best pass-rate midpoint threshold: **k = 2**
- Best joint midpoint threshold: **k = 4**
- Thresholds with robust positive grade effect (`grade_ate_lower > 0`): **0**
- Thresholds with robust positive pass-rate effect (`pass_ate_lower > 0`): **0**

## Files Generated
- Tables: `tables/q1_threshold_bounds.csv`, `tables/q1_recommended_k.csv`
- Figures: `figures/q1_grade_bounds_vs_k.png`, `figures/q1_passrate_bounds_vs_k.png`, `figures/q1_share_and_uncertainty_vs_k.png`, `figures/q1_midpoint_heatmap_vs_k.png`

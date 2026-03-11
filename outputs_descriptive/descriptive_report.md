# Descriptive Analysis Report

## Dataset Overview
- Rows: **3866**
- Columns: **12**
- Numeric columns analyzed: **11**

## Missing Data Highlights
- Tut_20162: 3471 missing (89.78%)
- Tut_20161: 3455 missing (89.37%)
- Tasa_aprob_20162: 495 missing (12.80%)
- Prom__20162: 495 missing (12.80%)
- Tasa_aprob_20161: 162 missing (4.19%)
- Prom__20161: 161 missing (4.16%)

## Tables Generated
- See CSV tables in `Econometric Workshop/outputs_descriptive/tables`

## Core Variable Statistics (Exploratory)
- The table below reports mean, variance, minimum, and maximum for each analysis variable.
- `ID_Number` is excluded because it is an identifier, not an analysis feature.

| Variable | Mean | Variance | Min | Max |
|---|---:|---:|---:|---:|
| PTJE_RANKING | 712.9896 | 7398.9470 | 414.0000 | 850.0000 |
| PTJE_LYC | 594.2352 | 5769.4408 | 212.0000 | 818.0000 |
| PTJE_MAT | 604.7977 | 4949.8426 | 194.0000 | 819.0000 |
| VAI | 0.0970 | 0.0876 | 0.0000 | 1.0000 |
| BAP | 0.1063 | 0.0950 | 0.0000 | 1.0000 |
| Tut_20161 | 14.3869 | 67.7695 | 1.0000 | 48.0000 |
| Tut_20162 | 12.6582 | 56.5250 | 1.0000 | 41.0000 |
| Tasa_aprob_20161 | 0.8048 | 0.0789 | 0.0000 | 1.0000 |
| Prom__20161 | 4.7416 | 1.1535 | 1.0000 | 6.7200 |
| Tasa_aprob_20162 | 0.7704 | 0.0805 | 0.0000 | 1.0000 |
| Prom__20162 | 4.5472 | 1.0374 | 1.0000 | 6.7300 |

## Interpretation Guide
- Use `numeric_summary_statistics.csv` to inspect center, spread, skewness, and tails.
- Use `missingness_summary.csv` to identify potential selection/missing-data concerns.
- Use grouped tables by BAP/VAI to compare subgroup profiles.

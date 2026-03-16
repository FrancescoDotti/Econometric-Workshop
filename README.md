# Econometric Workshop

This project applies partial-identification methods (Manski-style worst-case bounds) to evaluate tutoring effects in two settings:

1- Q1 (Semester 1, BAP students): Finds how treatment intensity thresholds (k) (attendance (\ge k)) change outcomes and identifies useful candidate thresholds.

2- Q2 (Semester 2, BAP and VAI students): Compares Continue tutoring vs Opt-out using ATE intervals for grade and approval outcomes.

It also includes a descriptive-analysis script with summary tables and exploratory plots.

## Recent additions (brief)

- Added high-impact analysis scripts:
	- `partial_id_q1_high_impact.py`
	- `partial_id_q2_high_impact.py`
- High-impact assumption ladder currently uses: **Manski, MTR, MTS** (MIV removed).
- Added strong-ignorability (exogenous switching) point-estimate scripts:
	- `q1_strong_ignorability.py`
	- `q2_strong_ignorability.py`
- Added comparison graph generator:
	- `compare_partial_vs_strong_graphs.py`
	- Saves interval-vs-point figures in `outputs_q1_strong_ignorability/figures` and `outputs_q2_strong_ignorability/figures`.

## Quick run

- `python partial_id_q1_high_impact.py`
- `python partial_id_q2_high_impact.py`
- `python q1_strong_ignorability.py`
- `python q2_strong_ignorability.py`
- `python compare_partial_vs_strong_graphs.py`

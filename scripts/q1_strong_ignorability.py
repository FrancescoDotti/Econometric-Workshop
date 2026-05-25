#!/usr/bin/env python3
"""Q1 analysis under strong ignorability (exogenous switching).

This script answers Q1 using point estimates (difference in means) as if treatment assignment is unconfounded.
"""

import argparse
import os
import numpy as np
import pandas as pd

from exam2026_core import validate_required_columns

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COVARIATES = ["PTJE_RANKING", "PTJE_LYC", "PTJE_MAT", "VAI"]


def _ols_predict(X_fit: np.ndarray, y_fit: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
    """Fit OLS y = X_fit beta and return predictions on X_pred. X already includes intercept."""
    beta, *_ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
    return X_pred @ beta


def conditional_ate(
    sub: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariates: list,
) -> float:
    """Conditional strong-ignorability ATE via g-computation (outcome regression).

    Fits two separate linear models mu_1(x), mu_0(x) on treated/control with observed outcome,
    then averages the imputed difference over the full BAP subgroup with non-missing covariates.
    """
    needed = sub[[treatment_col, outcome_col] + covariates].copy()
    needed = needed.dropna(subset=[treatment_col] + covariates)
    if needed.empty:
        return float("nan")
    X_all = np.column_stack([np.ones(len(needed)), needed[covariates].to_numpy(dtype=float)])
    treated_obs = needed[(needed[treatment_col] == 1) & needed[outcome_col].notna()]
    control_obs = needed[(needed[treatment_col] == 0) & needed[outcome_col].notna()]
    if len(treated_obs) <= len(covariates) + 1 or len(control_obs) <= len(covariates) + 1:
        return float("nan")
    X_t = np.column_stack([np.ones(len(treated_obs)), treated_obs[covariates].to_numpy(dtype=float)])
    X_c = np.column_stack([np.ones(len(control_obs)), control_obs[covariates].to_numpy(dtype=float)])
    y_t = treated_obs[outcome_col].to_numpy(dtype=float)
    y_c = control_obs[outcome_col].to_numpy(dtype=float)
    mu1 = _ols_predict(X_t, y_t, X_all)
    mu0 = _ols_predict(X_c, y_c, X_all)
    return float(np.mean(mu1 - mu0))


def ensure_output_dir(base_output_dir: str) -> str:
    """Create output folder if it doesn't exist."""
    os.makedirs(base_output_dir, exist_ok=True)
    return base_output_dir


def analyze_q1_strong_ignorability(df: pd.DataFrame) -> pd.DataFrame:
    bap = df[df["BAP"] == 1].copy()
    bap = bap.dropna(subset=["Tut_20161"]).copy()
    bap["Tut_20161"] = bap["Tut_20161"].astype(float)
    max_k = int(bap["Tut_20161"].max())
    rows = []
    for k in range(1, max_k + 1):
        bap["Zk"] = (bap["Tut_20161"] >= k).astype(int)
        if bap["Zk"].nunique() < 2:
            continue
        for outcome in ["Prom__20161", "Tasa_aprob_20161"]:
            # Include all rows in group size (matching exam2026_core.py p_treated denominator),
            # but only use non-missing outcomes for mean calculation.
            treated_all = bap[bap["Zk"] == 1]
            control_all = bap[bap["Zk"] == 0]
            treated_obs = treated_all[outcome].dropna()
            control_obs = control_all[outcome].dropna()
            if treated_obs.empty or control_obs.empty:
                continue
            diff = treated_obs.mean() - control_obs.mean()
            ate_cond = conditional_ate(bap, "Zk", outcome, COVARIATES)
            rows.append({
                "k": k,
                "outcome": outcome,
                "treated_mean": treated_obs.mean(),
                "control_mean": control_obs.mean(),
                "point_estimate": diff,
                "point_estimate_conditional": ate_cond,
                "n_treated": len(treated_all),  # Include missing outcomes in count
                "n_control": len(control_all),  # Include missing outcomes in count
            })
    return pd.DataFrame(rows)


def write_report(results: pd.DataFrame, output_dir: str) -> str:
    report_path = os.path.join(output_dir, "q1_strong_ignorability_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Q1 Strong Ignorability (Exogenous Switching) Report\n\n")
        f.write("## Method\n")
        f.write("- Unconditional: difference in observed treated and control means (assumes Y(0),Y(1) ⊥ D).\n")
        f.write(f"- Conditional: g-computation regression adjustment on covariates {COVARIATES} "
                "(assumes Y(0),Y(1) ⊥ D | X).\n\n")
        if results.empty:
            f.write("No valid rows produced.\n")
            return report_path
        f.write("## Results (sample rows)\n")
        f.write("| k | outcome | treated_mean | control_mean | ATE_uncond | ATE_cond | n_treated | n_control |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for _, row in results.head(12).iterrows():
            f.write(
                f"| {int(row['k'])} | {row['outcome']} | {row['treated_mean']:.4f} | "
                f"{row['control_mean']:.4f} | {row['point_estimate']:.4f} | "
                f"{row['point_estimate_conditional']:.4f} | "
                f"{int(row['n_treated'])} | {int(row['n_control'])} |\n"
            )
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Q1 strong ignorability analysis")
    parser.add_argument("--data-path", default="Data2026.xls", help="Input dataset path")
    parser.add_argument("--output-dir", default="outputs_q1_strong_ignorability", help="Output directory")
    return parser


def run(args: argparse.Namespace) -> None:
    data_path = args.data_path if os.path.isabs(args.data_path) else os.path.join(PROJECT_DIR, args.data_path)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(PROJECT_DIR, args.output_dir)
    df = pd.read_excel(data_path)
    validate_required_columns(df)
    df["BAP"] = df["BAP"].astype(int)
    df["VAI"] = df["VAI"].astype(int)
    ensure_output_dir(output_dir)
    results = analyze_q1_strong_ignorability(df)
    results.to_csv(os.path.join(output_dir, "q1_strong_ignorability_results.csv"), index=False)
    report_path = write_report(results, output_dir)
    print("Q1 strong ignorability analysis complete.")
    print(f"Rows in results table: {len(results)}")
    print(f"Output directory: {output_dir}")
    print(f"Report file: {report_path}")

if __name__ == "__main__":
    run(build_parser().parse_args())

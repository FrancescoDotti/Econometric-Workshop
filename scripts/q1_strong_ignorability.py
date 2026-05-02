#!/usr/bin/env python3
"""Q1 analysis under strong ignorability (exogenous switching).

This script answers Q1 using point estimates (difference in means) as if treatment assignment is unconfounded.
"""

import argparse
import os
import pandas as pd

from exam2026_core import validate_required_columns

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
            rows.append({
                "k": k,
                "outcome": outcome,
                "treated_mean": treated_obs.mean(),
                "control_mean": control_obs.mean(),
                "point_estimate": diff,
                "n_treated": len(treated_all),  # Include missing outcomes in count
                "n_control": len(control_all),  # Include missing outcomes in count
            })
    return pd.DataFrame(rows)


def write_report(results: pd.DataFrame, output_dir: str) -> str:
    report_path = os.path.join(output_dir, "q1_strong_ignorability_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Q1 Strong Ignorability (Exogenous Switching) Report\n\n")
        f.write("## Method\n")
        f.write("- Point estimates are computed as difference in means between treated and control, assuming unconfoundedness.\n\n")
        if results.empty:
            f.write("No valid rows produced.\n")
            return report_path
        f.write("## Results (sample rows)\n")
        f.write("| k | outcome | treated_mean | control_mean | point_estimate | n_treated | n_control |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for _, row in results.head(12).iterrows():
            f.write(f"| {int(row['k'])} | {row['outcome']} | {row['treated_mean']:.4f} | {row['control_mean']:.4f} | {row['point_estimate']:.4f} | {int(row['n_treated'])} | {int(row['n_control'])} |\n")
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

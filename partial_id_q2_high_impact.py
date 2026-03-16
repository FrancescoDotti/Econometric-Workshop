#!/usr/bin/env python3
"""Q2 high-impact extensions: advanced Continue vs Opt-out analysis.

This script is additive and leaves existing exam scripts unchanged.
It implements:
1. Assumption ladder (Manski, MTR, MTS).
2. Bootstrap uncertainty for bound endpoints.
3. Missing-support sensitivity.
4. Extra heterogeneity (BAP/VAI + PTJE terciles).
"""

from __future__ import annotations

import argparse
import os
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from advanced_partial_id_core import (
    bootstrap_bounds,
    manski_bounds,
    mtr_bounds_from_manski,
    mts_bounds_from_manski,
    support_from_observed,
)
from exam2026_core import classify_decision, validate_required_columns


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_dirs(base_output_dir: str) -> tuple[str, str]:
    """Create output folders and return table/figure directories."""
    tables_dir = os.path.join(base_output_dir, "tables")
    figures_dir = os.path.join(base_output_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    return tables_dir, figures_dir


def _resolve_miv_col(df: pd.DataFrame) -> str:
    """Choose an available MIV proxy score column."""
    for col in ["PTJE_RANKING", "PTJE_MAT", "PTJE_LYC"]:
        if col in df.columns:
            return col
    raise ValueError("No MIV proxy found. Expected one of PTJE_RANKING/PTJE_MAT/PTJE_LYC.")


def _get_support_scenarios(df: pd.DataFrame, outcome: str) -> Dict[str, tuple[float, float]]:
    """Create full and trimmed support scenarios for sensitivity checks."""
    if outcome.startswith("Prom"):
        hard_min, hard_max = 1.0, 7.0
    else:
        hard_min, hard_max = 0.0, 1.0

    return {
        "full_support": (hard_min, hard_max),
        "trim_01_99": support_from_observed(df[outcome], hard_min, hard_max, 0.01, 0.99),
        "trim_05_95": support_from_observed(df[outcome], hard_min, hard_max, 0.05, 0.95),
    }


def _build_estimators(
    treatment_col: str,
    outcome_col: str,
    y_min: float,
    y_max: float,
) -> Dict[str, Callable[[pd.DataFrame], object]]:
    """Create assumption-specific estimator functions for one outcome."""

    def manski_est(df_in: pd.DataFrame):
        return manski_bounds(df_in, treatment_col, outcome_col, y_min, y_max)

    def mtr_est(df_in: pd.DataFrame):
        return mtr_bounds_from_manski(manski_est(df_in), direction="nondecreasing")

    def mts_est(df_in: pd.DataFrame):
        return mts_bounds_from_manski(df_in, treatment_col, outcome_col, y_min, y_max)

    return {
        "manski": manski_est,
        "mtr": mtr_est,
        "mts": mts_est,
    }


def _build_groups(df: pd.DataFrame, miv_col: str) -> Dict[str, pd.DataFrame]:
    """Build subgroup dictionary with baseline and extended heterogeneity groups."""
    groups = {
        "BAP_students": df[df["BAP"] == 1].copy(),
        "VAI_students": df[df["VAI"] == 1].copy(),
    }

    # Add baseline-score terciles as extra heterogeneity slices.
    score_df = df.dropna(subset=[miv_col]).copy()
    ranks = score_df[miv_col].rank(method="average")
    score_df["_score_tercile"] = pd.qcut(ranks, q=3, labels=["low", "mid", "high"], duplicates="drop")
    for label in sorted(score_df["_score_tercile"].dropna().unique()):
        key = f"score_tercile_{label}"
        groups[key] = score_df[score_df["_score_tercile"] == label].copy()

    return groups


def analyze_q2_high_impact(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Run advanced Q2 analysis and return a long-format results table."""
    work = df.copy()
    work["Continue2"] = (work["Tut_20162"].fillna(0) >= 1).astype(int)

    miv_col = _resolve_miv_col(work)
    groups = _build_groups(work, miv_col=miv_col)

    rows: List[Dict[str, float]] = []

    for group_name, gdf in groups.items():
        # Keep rows with defined treatment for each subgroup computation.
        gdf = gdf.dropna(subset=["Continue2"]).copy()
        if gdf.empty or gdf["Continue2"].nunique() < 2:
            continue

        for outcome in ["Prom__20162", "Tasa_aprob_20162"]:
            support_cases = _get_support_scenarios(gdf, outcome)

            for support_name, (y_min, y_max) in support_cases.items():
                estimators = _build_estimators(
                    treatment_col="Continue2",
                    outcome_col=outcome,
                    y_min=y_min,
                    y_max=y_max,
                )

                for assumption, estimator in estimators.items():
                    try:
                        res = estimator(gdf)
                    except Exception:
                        continue

                    boot = bootstrap_bounds(gdf, estimator, n_boot=n_boot, seed=seed + len(rows) + 1)
                    decision = classify_decision(res.ate_lower, res.ate_upper)

                    rows.append(
                        {
                            "group": group_name,
                            "outcome": outcome,
                            "support_scenario": support_name,
                            "assumption": assumption,
                            "y_min": y_min,
                            "y_max": y_max,
                            "n": res.n,
                            "treated_share": res.p_treated,
                            "ate_lower": res.ate_lower,
                            "ate_upper": res.ate_upper,
                            "ate_midpoint": res.ate_midpoint,
                            "ate_width": res.ate_width,
                            **decision,
                            **boot,
                        }
                    )

    return pd.DataFrame(rows)


def make_q2_high_impact_figures(results: pd.DataFrame, figures_dir: str) -> None:
    """Create concise visual diagnostics for advanced Q2 results."""
    if results.empty:
        return

    sns.set_theme(style="whitegrid", context="talk")

    # Build one interval chart for the baseline support scenario.
    base = results[(results["support_scenario"] == "full_support") & (results["assumption"] == "manski")].copy()
    if not base.empty:
        base["label"] = base["group"] + " | " + base["outcome"]
        base = base.sort_values(["outcome", "group"]).reset_index(drop=True)

        plt.figure(figsize=(14, 7))
        for i, row in base.iterrows():
            plt.hlines(i, row["ate_lower"], row["ate_upper"], color="#5e3c99", linewidth=3.5)
            plt.plot(row["ate_midpoint"], i, "o", color="black", markersize=6)
        plt.axvline(0, color="black", linestyle="--", linewidth=1.3)
        plt.yticks(range(len(base)), base["label"])
        plt.xlabel("ATE interval (Continue - Opt-out)")
        plt.ylabel("Group | Outcome")
        plt.title("Q2 High-Impact: Baseline Manski intervals across subgroups")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "q2_hi_baseline_interval_forest.png"), dpi=180)
        plt.close()

    # Plot interval widths by assumption to show identifying-power gains.
    width_df = results[results["support_scenario"] == "full_support"].copy()
    if not width_df.empty:
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=width_df,
            x="assumption",
            y="ate_width",
            hue="assumption",
            palette="Set2",
            dodge=False,
        )
        leg = plt.gca().get_legend()
        if leg is not None:
            leg.remove()
        plt.title("Q2 High-Impact: Interval width distribution by assumption")
        plt.xlabel("Assumption")
        plt.ylabel("ATE interval width")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "q2_hi_width_by_assumption.png"), dpi=180)
        plt.close()


def write_q2_high_impact_report(results: pd.DataFrame, output_dir: str) -> str:
    """Write markdown report summarizing the added high-impact analyses."""
    report_path = os.path.join(output_dir, "q2_high_impact_report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Q2 High-Impact Partial-ID Report\n\n")
        f.write("## What was added\n")
        f.write("- Assumption ladder (Manski, MTR, MTS).\n")
        f.write("- Bootstrap uncertainty for interval endpoints.\n")
        f.write("- Missing-support sensitivity (full and trimmed supports).\n")
        f.write("- Extra heterogeneity by baseline score terciles.\n\n")

        if results.empty:
            f.write("No valid rows were produced.\n")
            return report_path

        f.write("## Baseline support decision snapshot\n")
        f.write("| Group | Outcome | Assumption | ATE Lower | ATE Upper | Robust Sign |\n")
        f.write("|---|---|---|---:|---:|---|\n")

        base = results[results["support_scenario"] == "full_support"].copy()
        for _, row in base.iterrows():
            f.write(
                f"| {row['group']} | {row['outcome']} | {row['assumption']} | "
                f"{row['ate_lower']:.4f} | {row['ate_upper']:.4f} | {row['robust_sign']} |\n"
            )

        f.write("\n## Tightening summary\n")
        summary = base.groupby("assumption", as_index=False)["ate_width"].mean().sort_values("ate_width")
        for _, row in summary.iterrows():
            f.write(f"- Mean interval width ({row['assumption']}): {row['ate_width']:.4f}\n")

    return report_path


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for advanced Q2 analysis."""
    parser = argparse.ArgumentParser(description="Q2 high-impact partial-ID analysis")
    parser.add_argument("--data-path", default="Data2026.xls", help="Input dataset path")
    parser.add_argument("--output-dir", default="outputs_q2_high_impact", help="Output directory")
    parser.add_argument("--n-boot", type=int, default=120, help="Bootstrap draws per configuration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def run(args: argparse.Namespace) -> None:
    """Execute the complete high-impact Q2 pipeline."""
    data_path = args.data_path if os.path.isabs(args.data_path) else os.path.join(PROJECT_DIR, args.data_path)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(PROJECT_DIR, args.output_dir)

    df = pd.read_excel(data_path)
    validate_required_columns(df)
    df["BAP"] = df["BAP"].astype(int)
    df["VAI"] = df["VAI"].astype(int)

    tables_dir, figures_dir = ensure_dirs(output_dir)

    results = analyze_q2_high_impact(df, n_boot=args.n_boot, seed=args.seed)
    results.to_csv(os.path.join(tables_dir, "q2_high_impact_bounds.csv"), index=False)

    make_q2_high_impact_figures(results, figures_dir)
    report_path = write_q2_high_impact_report(results, output_dir)

    print("Q2 high-impact analysis complete.")
    print(f"Rows in advanced bounds table: {len(results)}")
    print(f"Tables directory: {tables_dir}")
    print(f"Figures directory: {figures_dir}")
    print(f"Report file: {report_path}")


if __name__ == "__main__":
    run(build_parser().parse_args())

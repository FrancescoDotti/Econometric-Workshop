#!/usr/bin/env python3
"""Question 2: Partial-identification comparison of Continue vs Opt-out.

Goal:
- In semester 2, define Continue2 = 1[Tut_20162 >= 1], else 0.
- For BAP and VAI students, compare Continue vs Opt-out using partial-ID ATE bounds.
- Evaluate two outcomes:
  1) Prom__20162 (average grade)
  2) Tasa_aprob_20162 (approval rate)
- Report robust/optimistic/conservative choices and produce an interval plot.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from exam2026_core import analyze_question_2, validate_required_columns


def ensure_dirs(base_output_dir: str) -> tuple[str, str]:
    """Create output subfolders and return table/figure paths."""
    tables_dir = os.path.join(base_output_dir, "tables")
    figures_dir = os.path.join(base_output_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    return tables_dir, figures_dir


def save_q2_tables(q2: pd.DataFrame, tables_dir: str) -> None:
    """Save detailed and summary tables for Question 2 decisions."""
    # Save complete bounds table.
    q2.to_csv(os.path.join(tables_dir, "q2_continue_vs_optout_bounds.csv"), index=False)

    # Save a compact decision-focused summary table.
    decision_cols = [
        "group",
        "outcome",
        "ate_lower",
        "ate_upper",
        "ate_midpoint",
        "robust_sign",
        "optimistic_choice",
        "conservative_choice",
    ]
    q2[decision_cols].to_csv(
        os.path.join(tables_dir, "q2_decision_summary.csv"),
        index=False,
    )


def make_q2_graph(q2: pd.DataFrame, figures_dir: str) -> None:
    """Create interval plot to compare Continue vs Opt-out by group/outcome."""
    if q2.empty:
        return

    # Use clean style and vivid colors for readability.
    sns.set_theme(style="whitegrid", context="talk")

    # Build display label to put group and outcome on one axis.
    plot_df = q2.copy()
    plot_df["label"] = plot_df["group"] + " | " + plot_df["outcome"]
    plot_df = plot_df.sort_values(["group", "outcome"]).reset_index(drop=True)

    plt.figure(figsize=(13, 6))

    # Draw each ATE interval and its midpoint as a point.
    for i, row in plot_df.iterrows():
        color = "#1b9e77" if row["ate_lower"] > 0 else "#d95f02" if row["ate_upper"] < 0 else "#7570b3"
        plt.hlines(i, row["ate_lower"], row["ate_upper"], color=color, linewidth=4)
        plt.plot(row["ate_midpoint"], i, "o", color="black", markersize=7)

    # Vertical zero line helps interpret beneficial vs harmful ranges.
    plt.axvline(0, color="black", linestyle="--", linewidth=1.5)

    plt.yticks(range(len(plot_df)), plot_df["label"])
    plt.xlabel("ATE interval (Continue - Opt-out)")
    plt.ylabel("Group | Outcome")
    plt.title("Q2: Partial-ID ATE Intervals for Continue vs Opt-out")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "q2_continue_vs_optout_intervals.png"), dpi=180)
    plt.close()


def write_q2_report(q2: pd.DataFrame, output_dir: str) -> str:
    """Write a short markdown report that summarizes Question 2 findings."""
    report_path = os.path.join(output_dir, "q2_partial_id_report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Q2 Partial-Identification Report\n\n")
        f.write("## Objective\n")
        f.write(
            "Compare the semester-2 decision `Continue` vs `Opt-out` using partial-"
            "identification bounds for BAP and VAI students.\n\n"
        )

        f.write("## Data and Setup\n")
        f.write(f"- Number of group-outcome rows: **{len(q2)}**\n")
        f.write("- Treatment definition: `Continue2 = 1[Tut_20162 >= 1]`\n")
        f.write("- Outcomes: `Prom__20162` and `Tasa_aprob_20162`\n\n")

        if q2.empty:
            f.write("## Result\n")
            f.write("No valid rows were produced for the selected groups/outcomes.\n")
            return report_path

        f.write("## Interval Summary\n")
        f.write("| Group | Outcome | ATE Lower | ATE Upper | Robust Sign | Optimistic | Conservative |\n")
        f.write("|---|---|---:|---:|---|---|---|\n")
        for _, row in q2.iterrows():
            f.write(
                f"| {row['group']} | {row['outcome']} | {row['ate_lower']:.4f} | "
                f"{row['ate_upper']:.4f} | {row['robust_sign']} | "
                f"{row['optimistic_choice']} | {row['conservative_choice']} |\n"
            )
        f.write("\n")

        f.write("## Files Generated\n")
        f.write("- Tables: `tables/q2_continue_vs_optout_bounds.csv`, `tables/q2_decision_summary.csv`\n")
        f.write("- Figure: `figures/q2_continue_vs_optout_intervals.png`\n")

    return report_path


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for dataset path and output destination."""
    parser = argparse.ArgumentParser(description="Q2 partial-identification decision analysis")
    parser.add_argument(
        "--data-path",
        default="Data2026.xls",
        help="Path to the dataset (default: Data2026.xls)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_q2_partial_id",
        help="Directory where Q2 outputs are saved",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """Execute full Q2 continue-vs-opt-out analysis pipeline."""
    # Load raw dataset.
    df = pd.read_excel(args.data_path)

    # Validate required columns before analysis.
    validate_required_columns(df)

    # Ensure binary indicators are integer typed for subgroup filtering.
    df["BAP"] = df["BAP"].astype(int)
    df["VAI"] = df["VAI"].astype(int)

    # Create output folders.
    tables_dir, figures_dir = ensure_dirs(args.output_dir)

    # Run Question 2 partial-ID comparison using the shared core formulas.
    q2 = analyze_question_2(df, tables_dir)

    # Save final tables and interval visualization.
    save_q2_tables(q2, tables_dir)
    make_q2_graph(q2, figures_dir)

    # Write a plain-language markdown summary for fast interpretation.
    report_path = write_q2_report(q2, args.output_dir)

    # Print execution summary for the user.
    print("Q2 partial-identification analysis complete.")
    print(f"Rows analyzed in Q2 summary: {len(q2)}")
    print(f"Tables directory: {tables_dir}")
    print(f"Figures directory: {figures_dir}")
    print(f"Report file: {report_path}")


if __name__ == "__main__":
    run(build_parser().parse_args())

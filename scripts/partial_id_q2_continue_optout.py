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


# Base directory for this project (Econometric Workshop folder).
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_output_dir(base_output_dir: str) -> str:
    """Create output folder if it doesn't exist."""
    os.makedirs(base_output_dir, exist_ok=True)
    return base_output_dir


def save_q2_tables(q2: pd.DataFrame, output_dir: str) -> None:
    """Save detailed and summary tables for Question 2 decisions."""
    # Save complete bounds table.
    q2.to_csv(os.path.join(output_dir, "q2_continue_vs_optout_bounds.csv"), index=False)

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
        os.path.join(output_dir, "q2_decision_summary.csv"),
        index=False,
    )


def make_q2_graph(q2: pd.DataFrame, output_dir: str) -> None:
    """Create interval plot to compare Continue vs Opt-out by group/outcome (solarized theme)."""
    if q2.empty:
        return

    # Light Solarized color palette
    SOLARIZED = {
        "base03": "#002b36", "base02": "#073642", "base01": "#586e75",
        "base00": "#657b83", "base0": "#839496", "base1": "#93a1a1",
        "base2": "#eee8d5", "base3": "#fdf6e3", "yellow": "#b58900",
        "orange": "#cb4b16", "red": "#dc322f", "magenta": "#d33682",
        "violet": "#6c71c4", "blue": "#268bd2", "cyan": "#2aa198", "green": "#859900",
    }

    # Apply light solarized theme
    plt.rcParams["figure.facecolor"] = SOLARIZED["base3"]
    plt.rcParams["axes.facecolor"] = SOLARIZED["base3"]
    plt.rcParams["axes.edgecolor"] = SOLARIZED["base1"]
    plt.rcParams["axes.labelcolor"] = SOLARIZED["base00"]
    plt.rcParams["text.color"] = SOLARIZED["base00"]
    plt.rcParams["xtick.color"] = SOLARIZED["base00"]
    plt.rcParams["ytick.color"] = SOLARIZED["base00"]
    plt.rcParams["grid.color"] = SOLARIZED["base2"]
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    sns.set_theme(style="whitegrid", context="talk")

    # Build display label to put group and outcome on one axis.
    plot_df = q2.copy()
    plot_df["label"] = plot_df["group"] + " | " + plot_df["outcome"]
    plot_df = plot_df.sort_values(["group", "outcome"]).reset_index(drop=True)

    plt.figure(figsize=(13, 6), facecolor=SOLARIZED["base3"])
    ax = plt.gca()
    ax.set_facecolor(SOLARIZED["base3"])

    # Draw each ATE interval and its midpoint as a point (solarized colors).
    for i, row in plot_df.iterrows():
        if row["ate_lower"] > 0:
            color = SOLARIZED["green"]
        elif row["ate_upper"] < 0:
            color = SOLARIZED["orange"]
        else:
            color = SOLARIZED["violet"]
        plt.hlines(i, row["ate_lower"], row["ate_upper"], color=color, linewidth=4)
        plt.plot(row["ate_midpoint"], i, "o", color=SOLARIZED["base01"], markersize=7)

    # Vertical zero line helps interpret beneficial vs harmful ranges.
    plt.axvline(0, color=SOLARIZED["base01"], linestyle="--", linewidth=1.5)

    plt.yticks(range(len(plot_df)), plot_df["label"])
    plt.xlabel("ATE interval (Continue - Opt-out)")
    plt.ylabel("Group | Outcome")
    plt.title("Q2: Partial-ID ATE Intervals for Continue vs Opt-out")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "q2_continue_vs_optout_intervals.png"), dpi=180)
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
        f.write("- Tables: `q2_continue_vs_optout_bounds.csv`, `q2_decision_summary.csv`\n")
        f.write("- Figure: `q2_continue_vs_optout_intervals.png`\n")

    return report_path


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for dataset path and output destination."""
    parser = argparse.ArgumentParser(description="Q2 partial-identification decision analysis")
    parser.add_argument(
        "--data-path",
        default="Data2026.xls",
        help="Path to the dataset (default: Data2026.xls in Econometric Workshop)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_q2_partial_id",
        help="Directory where Q2 outputs are saved (inside Econometric Workshop by default)",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """Execute full Q2 continue-vs-opt-out analysis pipeline."""
    # Resolve input/output paths from project folder when relative paths are used.
    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(PROJECT_DIR, data_path)

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_DIR, output_dir)

    # Load raw dataset.
    df = pd.read_excel(data_path)

    # Validate required columns before analysis.
    validate_required_columns(df)

    # Ensure binary indicators are integer typed for subgroup filtering.
    df["BAP"] = df["BAP"].astype(int)
    df["VAI"] = df["VAI"].astype(int)

    # Create output directory.
    ensure_output_dir(output_dir)

    # Run Question 2 partial-ID comparison using the shared core formulas.
    q2 = analyze_question_2(df, output_dir)

    # Save final tables and interval visualization.
    save_q2_tables(q2, output_dir)
    make_q2_graph(q2, output_dir)

    # Write a plain-language markdown summary for fast interpretation.
    report_path = write_q2_report(q2, output_dir)

    # Print execution summary for the user.
    print("Q2 partial-identification analysis complete.")
    print(f"Rows analyzed in Q2 summary: {len(q2)}")
    print(f"Output directory: {output_dir}")
    print(f"Report file: {report_path}")


if __name__ == "__main__":
    run(build_parser().parse_args())

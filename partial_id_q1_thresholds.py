#!/usr/bin/env python3
"""Question 1: Partial-identification analysis over tutoring thresholds k.

Goal:
- For BAP students in semester 1, define Z_k = 1[Tut_20161 >= k].
- For each k, estimate worst-case (Manski-style) ATE bounds for:
  1) Prom__20161 (average grade)
  2) Tasa_aprob_20161 (approval rate)
- Identify which threshold k gives the strongest impact under different rules.
- Produce colorful graphs showing how outcomes vary with k.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from exam2026_core import analyze_question_1, validate_required_columns


# Base directory for this project (Econometric Workshop folder).
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_dirs(base_output_dir: str) -> tuple[str, str]:
    """Create output subfolders for tables and figures."""
    tables_dir = os.path.join(base_output_dir, "tables")
    figures_dir = os.path.join(base_output_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    return tables_dir, figures_dir


def save_q1_tables(q1: pd.DataFrame, tables_dir: str) -> None:
    """Save key tables used to interpret threshold effects."""
    # Save the complete threshold-by-threshold bounds table.
    q1.to_csv(os.path.join(tables_dir, "q1_threshold_bounds.csv"), index=False)

    # Build a small recommendation table for quick policy reading.
    if q1.empty:
        return

    selection_rules = [
        ("grade_conservative_k", "grade_ate_lower", "max"),
        ("grade_optimistic_k", "grade_ate_upper", "max"),
        ("grade_midpoint_k", "grade_ate_midpoint", "max"),
        ("pass_conservative_k", "pass_ate_lower", "max"),
        ("pass_optimistic_k", "pass_ate_upper", "max"),
        ("pass_midpoint_k", "pass_ate_midpoint", "max"),
        ("joint_midpoint_k", "joint_midpoint_score", "max"),
    ]

    summary_rows = []
    for rule_name, column, mode in selection_rules:
        # Choose k that optimizes each criterion.
        idx = q1[column].idxmax() if mode == "max" else q1[column].idxmin()
        row = q1.loc[idx].to_dict()
        row["selection_rule"] = rule_name
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(tables_dir, "q1_recommended_k.csv"),
        index=False,
    )


def make_q1_graphs(q1: pd.DataFrame, figures_dir: str) -> None:
    """Create colorful figures to show how k changes partial-ID outputs."""
    if q1.empty:
        return

    # Use a vivid style so curves and intervals are easy to distinguish.
    sns.set_theme(style="whitegrid", context="talk")

    # 1) Grade ATE interval and midpoint across k.
    plt.figure(figsize=(12, 6))
    plt.fill_between(
        q1["k"],
        q1["grade_ate_lower"],
        q1["grade_ate_upper"],
        color="#66c2a5",
        alpha=0.25,
        label="Grade ATE bounds",
    )
    sns.lineplot(data=q1, x="k", y="grade_ate_midpoint", color="#1b9e77", linewidth=2.5)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.2)
    plt.title("Q1: Grade Effect Bounds vs Tutoring Threshold k")
    plt.xlabel("Threshold k")
    plt.ylabel("ATE on Prom__20161")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "q1_grade_bounds_vs_k.png"), dpi=180)
    plt.close()

    # 2) Approval-rate ATE interval and midpoint across k.
    plt.figure(figsize=(12, 6))
    plt.fill_between(
        q1["k"],
        q1["pass_ate_lower"],
        q1["pass_ate_upper"],
        color="#fc8d62",
        alpha=0.25,
        label="Pass-rate ATE bounds",
    )
    sns.lineplot(data=q1, x="k", y="pass_ate_midpoint", color="#d95f02", linewidth=2.5)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.2)
    plt.title("Q1: Approval-Rate Effect Bounds vs Tutoring Threshold k")
    plt.xlabel("Threshold k")
    plt.ylabel("ATE on Tasa_aprob_20161")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "q1_passrate_bounds_vs_k.png"), dpi=180)
    plt.close()

    # 3) Treatment share and uncertainty widths across k.
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=q1, x="k", y="treated_share", color="#7570b3", linewidth=2.5, ax=ax1)
    ax1.set_xlabel("Threshold k")
    ax1.set_ylabel("Share with Z_k = 1", color="#7570b3")
    ax1.tick_params(axis="y", labelcolor="#7570b3")

    # Add a second axis to visualize how interval width changes with k.
    ax2 = ax1.twinx()
    sns.lineplot(data=q1, x="k", y="grade_ate_width", color="#e7298a", linewidth=2, ax=ax2)
    sns.lineplot(data=q1, x="k", y="pass_ate_width", color="#66a61e", linewidth=2, ax=ax2)
    ax2.set_ylabel("ATE interval width", color="#444444")

    ax1.set_title("Q1: Composition (treated share) and Uncertainty (interval width) vs k")
    fig.tight_layout()
    plt.savefig(os.path.join(figures_dir, "q1_share_and_uncertainty_vs_k.png"), dpi=180)
    plt.close(fig)

    # 4) Heatmap to compare midpoint effects jointly across k.
    heat = q1[["k", "grade_ate_midpoint", "pass_ate_midpoint"]].set_index("k").T

    # Use a wider canvas and disable dense annotations to avoid unreadable overlap.
    plt.figure(figsize=(16, 4.8))
    ax = sns.heatmap(
        heat,
        cmap="RdYlGn",
        center=0,
        annot=False,
        cbar_kws={"label": "ATE midpoint"},
        linewidths=0.25,
        linecolor="white",
    )

    # Show fewer x labels so threshold ticks remain legible.
    all_k = heat.columns.tolist()
    show_every = 2
    tick_positions = [i + 0.5 for i in range(len(all_k)) if i % show_every == 0]
    tick_labels = [str(all_k[i]) for i in range(len(all_k)) if i % show_every == 0]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0)

    # Use cleaner row labels for readability.
    ax.set_yticklabels(["Grade midpoint ATE", "Pass-rate midpoint ATE"], rotation=0)

    plt.title("Q1: Midpoint ATE by Threshold k", pad=10)
    plt.xlabel("Threshold k")
    plt.ylabel("Outcome")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "q1_midpoint_heatmap_vs_k.png"), dpi=200)
    plt.close()


def write_q1_report(q1: pd.DataFrame, output_dir: str) -> str:
    """Write a short markdown report that summarizes Question 1 findings."""
    report_path = os.path.join(output_dir, "q1_partial_id_report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Q1 Partial-Identification Report\n\n")
        f.write("## Objective\n")
        f.write(
            "Evaluate how the tutoring intensity threshold `k` changes the partial-"
            "identification bounds for semester-1 outcomes among BAP students.\n\n"
        )

        f.write("## Data and Setup\n")
        f.write(f"- Number of evaluated thresholds: **{len(q1)}**\n")
        f.write("- Treatment definition: `Z_k = 1[Tut_20161 >= k]`\n")
        f.write("- Outcomes: `Prom__20161` and `Tasa_aprob_20161`\n\n")

        if q1.empty:
            f.write("## Result\n")
            f.write("No valid thresholds were found (no treated/control variation by k).\n")
            return report_path

        # Compute key k values under common decision summaries.
        k_grade_mid = int(q1.loc[q1["grade_ate_midpoint"].idxmax(), "k"])
        k_pass_mid = int(q1.loc[q1["pass_ate_midpoint"].idxmax(), "k"])
        k_joint_mid = int(q1.loc[q1["joint_midpoint_score"].idxmax(), "k"])

        robust_grade = int((q1["grade_ate_lower"] > 0).sum())
        robust_pass = int((q1["pass_ate_lower"] > 0).sum())

        f.write("## Main Findings\n")
        f.write(f"- Best grade midpoint threshold: **k = {k_grade_mid}**\n")
        f.write(f"- Best pass-rate midpoint threshold: **k = {k_pass_mid}**\n")
        f.write(f"- Best joint midpoint threshold: **k = {k_joint_mid}**\n")
        f.write(
            f"- Thresholds with robust positive grade effect (`grade_ate_lower > 0`): **{robust_grade}**\n"
        )
        f.write(
            f"- Thresholds with robust positive pass-rate effect (`pass_ate_lower > 0`): **{robust_pass}**\n\n"
        )

        f.write("## Files Generated\n")
        f.write("- Tables: `tables/q1_threshold_bounds.csv`, `tables/q1_recommended_k.csv`\n")
        f.write(
            "- Figures: `figures/q1_grade_bounds_vs_k.png`, `figures/q1_passrate_bounds_vs_k.png`, "
            "`figures/q1_share_and_uncertainty_vs_k.png`, `figures/q1_midpoint_heatmap_vs_k.png`\n"
        )

    return report_path


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for data path and output folder."""
    parser = argparse.ArgumentParser(description="Q1 partial-identification threshold analysis")
    parser.add_argument(
        "--data-path",
        default="Data2026.xls",
        help="Path to the dataset (default: Data2026.xls in Econometric Workshop)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_q1_partial_id",
        help="Directory where Q1 outputs are saved (inside Econometric Workshop by default)",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """Execute full Q1 analysis pipeline and write outputs."""
    # Resolve input/output paths from project folder when relative paths are used.
    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(PROJECT_DIR, data_path)

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_DIR, output_dir)

    # Load raw dataset.
    df = pd.read_excel(data_path)

    # Validate columns required by the partial-ID formulas.
    validate_required_columns(df)

    # Ensure binary indicators are numeric for subgroup filtering.
    df["BAP"] = df["BAP"].astype(int)
    df["VAI"] = df["VAI"].astype(int)

    # Create clean folder structure for outputs.
    tables_dir, figures_dir = ensure_dirs(output_dir)

    # Run the partial-identification threshold analysis for Question 1.
    q1 = analyze_question_1(df, tables_dir)

    # Save standardized tables and graphs for interpretation.
    save_q1_tables(q1, tables_dir)
    make_q1_graphs(q1, figures_dir)

    # Write a plain-language markdown summary for quick interpretation.
    report_path = write_q1_report(q1, output_dir)

    # Print summary so the user can verify successful execution.
    print("Q1 partial-identification analysis complete.")
    print(f"Thresholds evaluated: {len(q1)}")
    print(f"Tables directory: {tables_dir}")
    print(f"Figures directory: {figures_dir}")
    print(f"Report file: {report_path}")


if __name__ == "__main__":
    run(build_parser().parse_args())

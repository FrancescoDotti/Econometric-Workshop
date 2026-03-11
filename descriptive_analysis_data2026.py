#!/usr/bin/env python3
"""Descriptive analysis for Data2026.xls.

This script builds a broad descriptive overview of the dataset by creating:
1. Summary statistics tables.
2. Missing-value diagnostics.
3. Grouped descriptive tables (BAP and VAI).
4. Colorful visualizations for distribution, correlation, and group patterns.

Outputs are saved to a user-selected folder (default: outputs_descriptive).
"""

from __future__ import annotations

import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Base directory for this project (Econometric Workshop folder).
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_output_dirs(base_out_dir: str) -> tuple[str, str]:
    """Create output folders and return (tables_dir, figures_dir)."""
    tables_dir = os.path.join(base_out_dir, "tables")
    figures_dir = os.path.join(base_out_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    return tables_dir, figures_dir


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric columns, excluding pure identifiers when possible."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c != "ID_Number"]


def save_summary_tables(df: pd.DataFrame, tables_dir: str) -> List[str]:
    """Create and save descriptive tables used in the analysis report."""
    created_files: List[str] = []

    # Basic dataset shape and type summary.
    overview = pd.DataFrame(
        {
            "metric": ["n_rows", "n_columns"],
            "value": [df.shape[0], df.shape[1]],
        }
    )
    overview_path = os.path.join(tables_dir, "dataset_overview.csv")
    overview.to_csv(overview_path, index=False)
    created_files.append(overview_path)

    # Missingness counts and percentages by variable.
    missing = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
        }
    )
    missing["missing_pct"] = 100.0 * missing["missing_count"] / len(df)
    missing = missing.sort_values("missing_count", ascending=False)
    missing_path = os.path.join(tables_dir, "missingness_summary.csv")
    missing.to_csv(missing_path, index=False)
    created_files.append(missing_path)

    # Rich summary stats for numeric columns.
    numeric_cols = get_numeric_columns(df)
    desc = df[numeric_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    desc["skew"] = df[numeric_cols].skew(numeric_only=True)
    desc["kurtosis"] = df[numeric_cols].kurtosis(numeric_only=True)
    desc_path = os.path.join(tables_dir, "numeric_summary_statistics.csv")
    desc.to_csv(desc_path)
    created_files.append(desc_path)

    # Correlation matrix for numeric variables.
    corr = df[numeric_cols].corr(numeric_only=True)
    corr_path = os.path.join(tables_dir, "numeric_correlation_matrix.csv")
    corr.to_csv(corr_path)
    created_files.append(corr_path)

    # Grouped means and medians by BAP/VAI flags if available.
    for group_col in ["BAP", "VAI"]:
        if group_col in df.columns:
            grouped_mean = df.groupby(group_col)[numeric_cols].mean(numeric_only=True)
            grouped_median = df.groupby(group_col)[numeric_cols].median(numeric_only=True)

            mean_path = os.path.join(tables_dir, f"grouped_mean_by_{group_col}.csv")
            median_path = os.path.join(tables_dir, f"grouped_median_by_{group_col}.csv")
            grouped_mean.to_csv(mean_path)
            grouped_median.to_csv(median_path)
            created_files.extend([mean_path, median_path])

    return created_files


def save_descriptive_figures(df: pd.DataFrame, figures_dir: str) -> List[str]:
    """Create and save colorful figures for quick visual understanding."""
    created_files: List[str] = []

    # Global plot style for a clean and colorful look.
    sns.set_theme(style="whitegrid", context="talk")

    numeric_cols = get_numeric_columns(df)
    palette = sns.color_palette("Spectral", n_colors=max(3, len(numeric_cols)))

    # Keep binary flags out of histogram panel; they are shown with dedicated count plots.
    hist_cols = [c for c in numeric_cols if c not in {"BAP", "VAI"}]

    # 1) Missingness bar chart.
    missing_counts = df.isna().sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=missing_counts.index,
        y=missing_counts.values,
        palette=sns.color_palette("magma", n_colors=len(missing_counts)),
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("Missing Values by Variable", fontsize=16)
    plt.xlabel("Variable")
    plt.ylabel("Missing Count")
    plt.tight_layout()
    p = os.path.join(figures_dir, "missingness_barplot.png")
    plt.savefig(p, dpi=160)
    plt.close()
    created_files.append(p)

    # 2) Correlation heatmap for numeric columns.
    corr = df[numeric_cols].corr(numeric_only=True)
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
    )
    plt.title("Correlation Heatmap (Numeric Variables)", fontsize=16)
    plt.tight_layout()
    p = os.path.join(figures_dir, "correlation_heatmap.png")
    plt.savefig(p, dpi=180)
    plt.close()
    created_files.append(p)

    # 3) Histograms + KDE for continuous/count variables (excluding BAP/VAI flags).
    n = len(hist_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(hist_cols):
        sns.histplot(
            df[col].dropna(),
            kde=True,
            bins=30,
            color=palette[i % len(palette)],
            ax=axes[i],
            line_kws={"linewidth": 2},
        )

        # Add requested policy-reference lines for selected score variables.
        if col == "PTJE_RANKING":
            axes[i].axvline(750, color="red", linestyle="--", linewidth=2)
        elif col in {"PTJE_LYC", "PTJE_MAT"}:
            axes[i].axvline(550, color="black", linestyle="--", linewidth=2)

        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")

    # Hide unused subplot cells.
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    p = os.path.join(figures_dir, "numeric_distributions_hist_kde.png")
    plt.savefig(p, dpi=160)
    plt.close(fig)
    created_files.append(p)

    # 4) Boxplots for main outcomes by BAP and VAI.
    outcome_cols = [
        c for c in ["Prom__20161", "Prom__20162", "Tasa_aprob_20161", "Tasa_aprob_20162"]
        if c in df.columns
    ]

    for group_col in ["BAP", "VAI"]:
        if group_col not in df.columns or not outcome_cols:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, out_col in enumerate(outcome_cols):
            sns.boxplot(
                data=df,
                x=group_col,
                y=out_col,
                palette="Set2",
                ax=axes[i],
            )
            axes[i].set_title(f"{out_col} by {group_col}")
            axes[i].set_xlabel(group_col)
            axes[i].set_ylabel(out_col)

        for j in range(len(outcome_cols), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        p = os.path.join(figures_dir, f"boxplots_outcomes_by_{group_col}.png")
        plt.savefig(p, dpi=160)
        plt.close(fig)
        created_files.append(p)

    # 5) Count plots for binary policy groups.
    for col in ["BAP", "VAI"]:
        if col in df.columns:
            plt.figure(figsize=(7, 5))
            sns.countplot(
                data=df,
                x=col,
                palette=sns.color_palette("Set1", n_colors=2),
            )
            plt.title(f"Count of Students by {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            p = os.path.join(figures_dir, f"countplot_{col}.png")
            plt.savefig(p, dpi=160)
            plt.close()
            created_files.append(p)

    # 6) Pairplot on a selected subset of key variables.
    pair_cols = [
        c
        for c in [
            "PTJE_RANKING",
            "PTJE_LYC",
            "PTJE_MAT",
            "Prom__20161",
            "Prom__20162",
            "Tasa_aprob_20161",
            "Tasa_aprob_20162",
        ]
        if c in df.columns
    ]

    if len(pair_cols) >= 3:
        pair_df = df[pair_cols].dropna()
        # Sample to keep rendering time and file size manageable.
        if len(pair_df) > 1000:
            pair_df = pair_df.sample(1000, random_state=42)

        pair = sns.pairplot(
            pair_df,
            corner=True,
            diag_kind="kde",
            plot_kws={"alpha": 0.5, "s": 24, "color": "#1f77b4"},
            diag_kws={"fill": True, "color": "#ff7f0e"},
        )
        pair.fig.suptitle("Pairwise Relationships of Key Variables", y=1.02)
        p = os.path.join(figures_dir, "pairplot_key_variables.png")
        pair.savefig(p, dpi=150)
        plt.close(pair.fig)
        created_files.append(p)

    return created_files


def save_text_report(df: pd.DataFrame, tables_dir: str, out_dir: str) -> str:
    """Write a concise markdown report with key descriptive insights."""
    numeric_cols = get_numeric_columns(df)
    missing = df.isna().sum().sort_values(ascending=False)

    # Build a compact exploratory stats table (excluding identifier variables).
    stats_df = pd.DataFrame(index=numeric_cols)
    stats_df["mean"] = df[numeric_cols].mean(numeric_only=True)
    stats_df["variance"] = df[numeric_cols].var(numeric_only=True)
    stats_df["min"] = df[numeric_cols].min(numeric_only=True)
    stats_df["max"] = df[numeric_cols].max(numeric_only=True)

    report_path = os.path.join(out_dir, "descriptive_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Descriptive Analysis Report\n\n")
        f.write("## Dataset Overview\n")
        f.write(f"- Rows: **{df.shape[0]}**\n")
        f.write(f"- Columns: **{df.shape[1]}**\n")
        f.write(f"- Numeric columns analyzed: **{len(numeric_cols)}**\n\n")

        f.write("## Missing Data Highlights\n")
        for col, count in missing.head(6).items():
            pct = 100.0 * count / len(df)
            f.write(f"- {col}: {count} missing ({pct:.2f}%)\n")
        f.write("\n")

        f.write("## Tables Generated\n")
        f.write(f"- See CSV tables in `{tables_dir}`\n\n")

        f.write("## Core Variable Statistics (Exploratory)\n")
        f.write("- The table below reports mean, variance, minimum, and maximum for each analysis variable.\n")
        f.write("- `ID_Number` is excluded because it is an identifier, not an analysis feature.\n\n")
        f.write("| Variable | Mean | Variance | Min | Max |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for col in stats_df.index:
            row = stats_df.loc[col]
            f.write(
                f"| {col} | {row['mean']:.4f} | {row['variance']:.4f} | "
                f"{row['min']:.4f} | {row['max']:.4f} |\n"
            )
        f.write("\n")

        f.write("## Interpretation Guide\n")
        f.write("- Use `numeric_summary_statistics.csv` to inspect center, spread, skewness, and tails.\n")
        f.write("- Use `missingness_summary.csv` to identify potential selection/missing-data concerns.\n")
        f.write("- Use grouped tables by BAP/VAI to compare subgroup profiles.\n")

    return report_path


def build_parser() -> argparse.ArgumentParser:
    """Create command-line arguments for data and output paths."""
    parser = argparse.ArgumentParser(description="Descriptive analysis for Data2026.xls")
    parser.add_argument(
        "--data-path",
        default="Data2026.xls",
        help="Path to input dataset (default: Data2026.xls in Econometric Workshop)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_descriptive",
        help="Directory for tables/figures/report (inside Econometric Workshop by default)",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """Run the complete descriptive-analysis pipeline."""
    # Resolve input/output paths from project folder when relative paths are used.
    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(PROJECT_DIR, data_path)

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_DIR, output_dir)

    # Load dataset.
    df = pd.read_excel(data_path)

    # Create output structure.
    tables_dir, figures_dir = ensure_output_dirs(output_dir)

    # Save summary tables and colorful figures.
    table_files = save_summary_tables(df, tables_dir)
    figure_files = save_descriptive_figures(df, figures_dir)

    # Save plain-language markdown report.
    report_file = save_text_report(df, tables_dir, output_dir)

    # Print summary so user knows exactly where outputs are.
    print("Descriptive analysis complete.")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Tables created: {len(table_files)}")
    print(f"Figures created: {len(figure_files)}")
    print(f"Report: {report_file}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    run(build_parser().parse_args())

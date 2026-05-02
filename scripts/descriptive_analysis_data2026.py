#!/usr/bin/env python3
"""Descriptive analysis for Data2026.xls.

This script builds a broad descriptive overview of the dataset by creating:
1. A markdown narrative report (no CSV/table exports).
2. Colorful visualizations for distribution, correlation, and group patterns.
3. Violin summaries of Prom and Tasa_aprob outcomes by BAP and VAI.

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
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_output_dir(base_out_dir: str) -> str:
    """Create output folder if it doesn't exist."""
    os.makedirs(base_out_dir, exist_ok=True)
    return base_out_dir


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric columns, excluding pure identifiers when possible."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c != "ID_Number"]


def save_descriptive_figures(df: pd.DataFrame, output_dir: str) -> List[str]:
    """Create and save figures with light solarized color scheme."""
    created_files: List[str] = []

    # Light Solarized color palette
    SOLARIZED = {
        "base03": "#002b36",
        "base02": "#073642",
        "base01": "#586e75",
        "base00": "#657b83",
        "base0": "#839496",
        "base1": "#93a1a1",
        "base2": "#eee8d5",
        "base3": "#fdf6e3",
        "yellow": "#b58900",
        "orange": "#cb4b16",
        "red": "#dc322f",
        "magenta": "#d33682",
        "violet": "#6c71c4",
        "blue": "#268bd2",
        "cyan": "#2aa198",
        "green": "#859900",
    }

    # Apply light solarized theme globally
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
    sns.set_palette([
        SOLARIZED["blue"],
        SOLARIZED["green"],
        SOLARIZED["cyan"],
        SOLARIZED["violet"],
        SOLARIZED["orange"],
        SOLARIZED["magenta"],
    ])

    numeric_cols = get_numeric_columns(df)
    palette = [
        SOLARIZED["blue"],
        SOLARIZED["green"],
        SOLARIZED["cyan"],
        SOLARIZED["violet"],
        SOLARIZED["orange"],
        SOLARIZED["magenta"],
        SOLARIZED["yellow"],
        SOLARIZED["red"],
    ]

    # Keep binary flags out of histogram panel; they are shown with dedicated count plots.
    hist_cols = [c for c in numeric_cols if c not in {"BAP", "VAI"}]

    # 1) Missingness bar chart.
    missing_counts = df.isna().sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6), facecolor=SOLARIZED["base3"])
    ax = plt.gca()
    ax.set_facecolor(SOLARIZED["base3"])
    sns.barplot(
        x=missing_counts.index,
        y=missing_counts.values,
        palette=[SOLARIZED["cyan"], SOLARIZED["blue"], SOLARIZED["violet"], SOLARIZED["magenta"], SOLARIZED["orange"], SOLARIZED["red"]],
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("Missing Values by Variable", fontsize=16)
    plt.xlabel("Variable")
    plt.ylabel("Missing Count")
    plt.tight_layout()
    p = os.path.join(output_dir, "missingness_barplot.png")
    plt.savefig(p, dpi=160)
    plt.close()
    created_files.append(p)

    # 2) Correlation heatmap for numeric columns (solarized colormap).
    from matplotlib.colors import LinearSegmentedColormap
    corr = df[numeric_cols].corr(numeric_only=True)
    solarized_cmap = LinearSegmentedColormap.from_list(
        "solarized",
        [SOLARIZED["blue"], SOLARIZED["base3"], SOLARIZED["orange"]],
    )
    plt.figure(figsize=(11, 9), facecolor=SOLARIZED["base3"])
    ax = plt.gca()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=solarized_cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_facecolor(SOLARIZED["base3"])
    plt.title("Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    p = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(p, dpi=180)
    plt.close()
    created_files.append(p)

    # 3) Histograms + KDE for continuous/count variables (excluding BAP/VAI flags).
    n = len(hist_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 5 * nrows), facecolor=SOLARIZED["base3"])
    axes = np.array(axes).reshape(-1)

    # Define axis limits for paired variables to ensure consistency across semesters.
    axis_limits = {
        "Tut_20161": (0, 50, 0, 70),
        "Tut_20162": (0, 50, 0, 70),
        "Prom__20161": (0, 7, None, 420),
        "Prom__20162": (0, 7, None, 420),
        "Tasa_aprob_20161": (0, 1, None, 2000),
        "Tasa_aprob_20162": (0, 1, None, 2000),
    }

    for i, col in enumerate(hist_cols):
        ax = axes[i]
        ax.set_facecolor(SOLARIZED["base3"])
        sns.histplot(
            df[col].dropna(),
            kde=True,
            bins=30,
            color=palette[i % len(palette)],
            ax=ax,
            line_kws={"linewidth": 2, "color": SOLARIZED["base01"]},
        )

        # Add requested policy-reference lines for selected score variables.
        if col in {"PTJE_LYC", "PTJE_MAT"}:
            ax.axvline(550, color=SOLARIZED["base01"], linestyle="--", linewidth=2)

        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")

        # Apply consistent axis limits for paired variables.
        if col in axis_limits:
            x_min, x_max, y_min, y_max = axis_limits[col]
            ax.set_xlim(x_min, x_max)
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
                ax.set_yticks(range(y_min, y_max + 1, 10))

    # Hide unused subplot cells.
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    p = os.path.join(output_dir, "numeric_distributions_hist_kde.png")
    plt.savefig(p, dpi=160)
    plt.close(fig)
    created_files.append(p)

    # 4) Violin panels for main outcomes by BAP and VAI.
    outcome_cols = [
        c for c in ["Prom__20161", "Prom__20162", "Tasa_aprob_20161", "Tasa_aprob_20162"]
        if c in df.columns
    ]

    group_specs = []
    if "BAP" in df.columns:
        group_specs.append(("BAP", "BAP"))
    if "VAI" in df.columns:
        group_specs.append(("VAI", "VAI"))

    for group_col, group_label in group_specs:
        if group_col not in df.columns or not outcome_cols:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, out_col in enumerate(outcome_cols):
            axes[i].set_facecolor(SOLARIZED["base3"])
            sns.violinplot(
                data=df,
                x=group_col,
                y=out_col,
                inner="box",
                cut=0,
                palette=[SOLARIZED["cyan"], SOLARIZED["blue"]],
                ax=axes[i],
            )
            axes[i].set_title(f"{out_col} by {group_label}")
            axes[i].set_xlabel(group_label)
            axes[i].set_ylabel(out_col)

        for j in range(len(outcome_cols), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        p = os.path.join(output_dir, f"violinplots_outcomes_by_{group_label}.png")
        plt.savefig(p, dpi=160)
        plt.close(fig)
        created_files.append(p)

    # 5) Count plots for binary policy groups.
    for col, label in group_specs:
        if col in df.columns:
            plt.figure(figsize=(7, 5), facecolor=SOLARIZED["base3"])
            ax = plt.gca()
            ax.set_facecolor(SOLARIZED["base3"])
            sns.countplot(
                data=df,
                x=col,
                palette=[SOLARIZED["cyan"], SOLARIZED["blue"]],
            )
            plt.title(f"Count of Students by {label}")
            plt.xlabel(label)
            plt.ylabel("Count")
            plt.tight_layout()
            p = os.path.join(output_dir, f"countplot_{label}.png")
            plt.savefig(p, dpi=160)
            plt.close()
            created_files.append(p)

    # 6) BAP continuation from semester 1 to semester 2.
    if "BAP" in df.columns and "Tut_20161" in df.columns and "Tut_20162" in df.columns:
        # Filter to students enrolled in BAP first semester
        bap_sem1 = df[df["BAP"] == 1].copy()

        # Define continuation: attended at least 1 tutorial session in semester 2
        bap_sem1["Continued_Sem2"] = (bap_sem1["Tut_20162"].fillna(0) >= 1).astype(int)

        # Create counts for the visualization
        continued_counts = bap_sem1["Continued_Sem2"].value_counts().sort_index()
        continued_labels = ["Dropped Out", "Continued"]
        continued_values = [continued_counts.get(0, 0), continued_counts.get(1, 0)]

        fig, ax = plt.subplots(figsize=(10, 6), facecolor=SOLARIZED["base3"])
        ax.set_facecolor(SOLARIZED["base3"])

        bars = ax.bar(
            continued_labels,
            continued_values,
            color=[SOLARIZED["red"], SOLARIZED["green"]],
            edgecolor=SOLARIZED["base01"],
            linewidth=1.5,
        )

        # Add value labels on bars
        for bar, val in zip(bars, continued_values):
            height = bar.get_height()
            ax.annotate(
                f"{int(val)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color=SOLARIZED["base00"],
            )

        total_bap_sem1 = len(bap_sem1)
        continued_pct = 100 * continued_values[1] / total_bap_sem1 if total_bap_sem1 > 0 else 0

        ax.set_title("Continue or opt-out?", fontsize=28)
        ax.set_ylabel("Number of Students")
        ax.set_ylim(0, max(continued_values) * 1.15)

        plt.tight_layout()
        p = os.path.join(output_dir, "bap_continuation_sem1_to_sem2.png")
        plt.savefig(p, dpi=160)
        plt.close()
        created_files.append(p)

    # 7) Cross-distribution heatmap for VAI and BAP.
    from matplotlib.colors import LinearSegmentedColormap
    bap_col = "BAP"
    if bap_col in df.columns and "VAI" in df.columns:
        cross = pd.crosstab(df["VAI"], df[bap_col], dropna=False)

        # Keep a stable 2x2 order for binary indicators.
        cross = cross.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

        # Create warm solarized colormap for heatmap.
        solarized_cmap = LinearSegmentedColormap.from_list(
            "solarized_warm",
            [SOLARIZED["base3"], SOLARIZED["base2"], SOLARIZED["yellow"], SOLARIZED["orange"], SOLARIZED["red"]],
        )

        plt.figure(figsize=(10, 8), facecolor=SOLARIZED["base3"])
        ax = plt.gca()
        ax.set_facecolor(SOLARIZED["base3"])

        sns.heatmap(
            cross,
            annot=True,
            fmt="d",
            cmap=solarized_cmap,
            linewidths=1,
            linecolor=SOLARIZED["base2"],
            cbar=False,
            ax=ax,
            annot_kws={"size": 14, "weight": "bold"},
        )

        # Human-readable labels to mirror report style.
        ax.set_yticklabels(["VAI = 0", "VAI = 1"], rotation=0, va="center")
        ax.set_xticklabels(["BAP = 0", "BAP = 1"], rotation=0, ha="center")
        ax.set_ylabel("VAI")
        ax.set_xlabel("BAP")

        # Add row totals on the right (positioned to avoid overlap).
        row_totals = cross.sum(axis=1)
        for i, total in enumerate(row_totals):
            ax.text(
                cross.shape[1] + 0.15,
                i + 0.5,
                f"Total: {int(total)}",
                va="center",
                ha="left",
                fontsize=11,
                color=SOLARIZED["base00"],
                clip_on=False,
            )

        # Add column totals at the bottom (positioned to avoid overlap with labels).
        col_totals = cross.sum(axis=0)
        for j, total in enumerate(col_totals):
            ax.text(
                j + 0.5,
                cross.shape[0] + 0.3,
                f"Total: {int(total)}",
                va="bottom",
                ha="center",
                fontsize=11,
                color=SOLARIZED["base00"],
                clip_on=False,
            )

        plt.title("Cross-distribution of VAI and BAP", fontsize=16, pad=20)
        plt.tight_layout()
        p = os.path.join(output_dir, "vai_bap_crosstab_heatmap.png")
        plt.savefig(p, dpi=180, bbox_inches="tight")
        plt.close()
        created_files.append(p)

    return created_files


def save_text_report(df: pd.DataFrame, output_dir: str) -> str:
    """Write a concise markdown report with key descriptive insights."""
    numeric_cols = get_numeric_columns(df)
    missing = df.isna().sum().sort_values(ascending=False)

    # Build a compact exploratory stats table (excluding identifier variables).
    stats_df = pd.DataFrame(index=numeric_cols)
    stats_df["mean"] = df[numeric_cols].mean(numeric_only=True)
    stats_df["variance"] = df[numeric_cols].var(numeric_only=True)
    stats_df["min"] = df[numeric_cols].min(numeric_only=True)
    stats_df["max"] = df[numeric_cols].max(numeric_only=True)

    report_path = os.path.join(output_dir, "descriptive_report.md")
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

        f.write("## Core Variable Statistics (Exploratory)\n")
        f.write("- The summary below reports mean, variance, minimum, and maximum for each analysis variable.\n")
        f.write("- ID_Number is excluded because it is an identifier, not an analysis feature.\n\n")
        for col in stats_df.index:
            row = stats_df.loc[col]
            f.write(
                f"- {col}: mean={row['mean']:.4f}, variance={row['variance']:.4f}, "
                f"min={row['min']:.4f}, max={row['max']:.4f}\n"
            )
        f.write("\n")

        f.write("## Figures Generated\n")
        f.write("- Missingness bar chart, correlation heatmap, histograms/KDE panels, violin outcome panels, and count plots.\n\n")

        f.write("## Interpretation Guide\n")
        f.write("- Use violin outcome panels to compare conditional distributions (level, spread, tails) by policy group.\n")
        f.write("- Use histograms and violin panels to diagnose asymmetry, outliers, and semester-to-semester shifts.\n")

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

    # Create output directory.
    ensure_output_dir(output_dir)

    # Save colorful figures (CSV/table exports are disabled).
    figure_files = save_descriptive_figures(df, output_dir)

    # Save plain-language markdown report.
    report_file = save_text_report(df, output_dir)

    # Print summary so user knows exactly where outputs are.
    print("Descriptive analysis complete.")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("CSV/table files created: 0 (disabled by design)")
    print(f"Figures created: {len(figure_files)}")
    print(f"Report: {report_file}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    run(build_parser().parse_args())

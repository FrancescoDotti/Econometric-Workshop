#!/usr/bin/env python3
"""Create comparison graphs: partial-ID intervals vs strong-ignorability points.

This script reads existing output tables and saves new figures into:
- outputs_q1_strong_ignorability/
- outputs_q2_strong_ignorability/
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Q1 and Q2 tables for partial-ID and strong-ignorability results."""
    q1_partial = pd.read_csv(
        os.path.join(PROJECT_DIR, "outputs_q1_high_impact", "q1_high_impact_bounds.csv")
    )
    q1_point = pd.read_csv(
        os.path.join(
            PROJECT_DIR,
            "outputs_q1_strong_ignorability",
            "q1_strong_ignorability_results.csv",
        )
    )

    q2_partial = pd.read_csv(
        os.path.join(PROJECT_DIR, "outputs_q2_high_impact", "q2_high_impact_bounds.csv")
    )
    q2_point = pd.read_csv(
        os.path.join(
            PROJECT_DIR,
            "outputs_q2_strong_ignorability",
            "q2_strong_ignorability_results.csv",
        )
    )

    return q1_partial, q1_point, q2_partial, q2_point


def _ensure_output_dirs() -> tuple[str, str]:
    """Ensure output folders exist and return their paths."""
    q1_dir = os.path.join(PROJECT_DIR, "outputs_q1_strong_ignorability")
    q2_dir = os.path.join(PROJECT_DIR, "outputs_q2_strong_ignorability")
    os.makedirs(q1_dir, exist_ok=True)
    os.makedirs(q2_dir, exist_ok=True)
    return q1_dir, q2_dir


def make_q1_comparison_graphs(q1_partial: pd.DataFrame, q1_point: pd.DataFrame, q1_dir: str) -> None:
    """Create Q1 graphs comparing partial intervals against point estimates (solarized theme)."""
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

    # Keep the baseline support only so the graph is easy to read.
    base = q1_partial[q1_partial["support_scenario"] == "full_support"].copy()

    # Build one plot per outcome to avoid crowding.
    for outcome in ["Prom__20161", "Tasa_aprob_20161"]:
        part = base[base["outcome"] == outcome].copy()
        point = q1_point[q1_point["outcome"] == outcome].copy()
        if part.empty or point.empty:
            continue

        # Use only the assumptions currently included in the high-impact script.
        assumptions = ["manski", "mtr", "mts"]
        colors = {"manski": SOLARIZED["violet"], "mtr": SOLARIZED["green"], "mts": SOLARIZED["orange"]}

        plt.figure(figsize=(13, 7), facecolor=SOLARIZED["base3"])
        ax = plt.gca()
        ax.set_facecolor(SOLARIZED["base3"])

        # Draw interval endpoints as lines by assumption.
        for assumption in assumptions:
            a_df = part[part["assumption"] == assumption].sort_values("k")
            if a_df.empty:
                continue
            plt.plot(
                a_df["k"],
                a_df["ate_lower"],
                linestyle="--",
                linewidth=1.8,
                color=colors[assumption],
                alpha=0.9,
                label=f"{assumption} lower",
            )
            plt.plot(
                a_df["k"],
                a_df["ate_upper"],
                linestyle="-",
                linewidth=2.0,
                color=colors[assumption],
                alpha=0.9,
                label=f"{assumption} upper",
            )

        # Overlay strong-ignorability point estimates: unconditional and conditional.
        point = point.sort_values("k")
        plt.plot(
            point["k"],
            point["point_estimate"],
            color=SOLARIZED["base1"],
            linewidth=2.0,
            linestyle=(0, (3, 2)),
            marker="o",
            markersize=3,
            label="strong ignorability (unconditional)",
        )
        if "point_estimate_conditional" in point.columns:
            plt.plot(
                point["k"],
                point["point_estimate_conditional"],
                color=SOLARIZED["base01"],
                linewidth=2.8,
                marker="s",
                markersize=4,
                label="strong ignorability (conditional)",
            )

        plt.axhline(0, color=SOLARIZED["base01"], linestyle=":" , linewidth=1.2)
        plt.xlabel("Threshold k")
        plt.ylabel("Effect size")
        plt.title(f"Q1 Comparison: Partial-ID intervals vs point estimates ({outcome})")
        plt.legend(loc="best", ncol=2, fontsize=10)
        plt.tight_layout()

        file_name = (
            "q1_compare_partial_vs_point_grade.png"
            if outcome == "Prom__20161"
            else "q1_compare_partial_vs_point_passrate.png"
        )
        plt.savefig(os.path.join(q1_dir, file_name), dpi=180)
        plt.close()


def make_q2_comparison_graph(q2_partial: pd.DataFrame, q2_point: pd.DataFrame, q2_dir: str) -> None:
    """Create Q2 forest-style graph with intervals and point estimates (solarized theme)."""
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

    # Keep baseline support and baseline groups for a compact comparison figure.
    base = q2_partial[
        (q2_partial["support_scenario"] == "full_support")
        & (q2_partial["group"].isin(["BAP_students", "VAI_students"]))
    ].copy()
    if base.empty or q2_point.empty:
        return

    # Merge point estimates into the partial-ID table for plotting.
    point = q2_point.copy()
    point["label"] = point["group"] + " | " + point["outcome"]
    point_cols = ["group", "outcome", "point_estimate", "label"]
    if "point_estimate_conditional" in point.columns:
        point_cols.insert(3, "point_estimate_conditional")
    point = point[point_cols]

    base = base.merge(point, on=["group", "outcome"], how="left")
    base["label"] = base["group"] + " | " + base["outcome"]

    # Control row order for readability.
    row_order = [
        "BAP_students | Prom__20162",
        "BAP_students | Tasa_aprob_20162",
        "VAI_students | Prom__20162",
        "VAI_students | Tasa_aprob_20162",
    ]
    y_map = {label: idx for idx, label in enumerate(row_order)}

    colors = {"manski": SOLARIZED["violet"], "mtr": SOLARIZED["green"], "mts": SOLARIZED["orange"]}
    offsets = {"manski": -0.18, "mtr": 0.0, "mts": 0.18}

    plt.figure(figsize=(14, 8), facecolor=SOLARIZED["base3"])
    ax = plt.gca()
    ax.set_facecolor(SOLARIZED["base3"])

    # Draw one horizontal interval per assumption, slightly offset.
    for _, row in base.iterrows():
        if row["label"] not in y_map:
            continue
        y = y_map[row["label"]] + offsets.get(row["assumption"], 0.0)
        plt.hlines(
            y=y,
            xmin=row["ate_lower"],
            xmax=row["ate_upper"],
            color=colors.get(row["assumption"], "gray"),
            linewidth=3,
            alpha=0.95,
        )

    # Draw strong-ignorability estimates per row: unconditional (open circle) and conditional (filled square).
    has_cond = "point_estimate_conditional" in point.columns
    for label in row_order:
        p = point[point["label"] == label]
        if p.empty:
            continue
        plt.plot(
            float(p.iloc[0]["point_estimate"]),
            y_map[label],
            "o",
            markerfacecolor="none",
            markeredgecolor=SOLARIZED["base1"],
            markeredgewidth=1.5,
            markersize=8,
        )
        if has_cond and pd.notna(p.iloc[0]["point_estimate_conditional"]):
            plt.plot(
                float(p.iloc[0]["point_estimate_conditional"]),
                y_map[label],
                "s",
                color=SOLARIZED["base01"],
                markersize=8,
            )

    # Add reference line and labels.
    plt.axvline(0, color=SOLARIZED["base01"], linestyle="--", linewidth=1.2)
    plt.yticks(range(len(row_order)), row_order)
    plt.xlabel("Effect (Continue - Opt-out)")
    plt.ylabel("Group | Outcome")
    plt.title("Q2 Comparison: Partial-ID intervals vs strong-ignorability points")

    # Build a manual legend so it is clear what each element means.
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color=colors["manski"], lw=3, label="Manski interval"),
        Line2D([0], [0], color=colors["mtr"], lw=3, label="MTR interval"),
        Line2D([0], [0], color=colors["mts"], lw=3, label="MTS interval"),
        Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor=SOLARIZED["base1"],
               markeredgewidth=1.5, color="w", lw=0, label="SI unconditional", markersize=8),
        Line2D([0], [0], marker="s", color=SOLARIZED["base01"], lw=0, label="SI conditional", markersize=8),
    ]
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(q2_dir, "q2_compare_partial_vs_point_forest.png"), dpi=180)
    plt.close()


def run() -> None:
    """Run the full plotting workflow and save all comparison graphs."""
    q1_partial, q1_point, q2_partial, q2_point = _load_tables()
    q1_dir, q2_dir = _ensure_output_dirs()

    make_q1_comparison_graphs(q1_partial, q1_point, q1_dir)
    make_q2_comparison_graph(q2_partial, q2_point, q2_dir)

    print("Comparison graphs created successfully.")
    print(f"Q1 output directory: {q1_dir}")
    print(f"Q2 output directory: {q2_dir}")


if __name__ == "__main__":
    run()

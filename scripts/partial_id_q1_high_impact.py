#!/usr/bin/env python3
"""Q1 high-impact extensions: advanced partial-ID threshold analysis.

This script is fully additive and does not modify existing exam scripts.
It implements four upgrades:
1. Assumption tightening (Manski, MTR, MTS).
2. Bootstrap uncertainty for interval endpoints.
3. Missing-support sensitivity analysis.
4. Threshold stability analysis for recommended k.
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
from exam2026_core import validate_required_columns


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_output_dir(base_output_dir: str) -> str:
    """Create output folder if it doesn't exist."""
    os.makedirs(base_output_dir, exist_ok=True)
    return base_output_dir


def _get_support_scenarios(df: pd.DataFrame, outcome: str) -> Dict[str, tuple[float, float]]:
    """Build support scenarios used for missing-outcome sensitivity checks."""
    if outcome.startswith("Prom"):
        hard_min, hard_max = 1.0, 7.0
    else:
        hard_min, hard_max = 0.0, 1.0

    # Build multiple support assumptions from very conservative to trimmed.
    full = (hard_min, hard_max)
    trim_01_99 = support_from_observed(df[outcome], hard_min, hard_max, 0.01, 0.99)
    trim_05_95 = support_from_observed(df[outcome], hard_min, hard_max, 0.05, 0.95)
    return {
        "full_support": full,
        "trim_01_99": trim_01_99,
        "trim_05_95": trim_05_95,
    }


def _build_assumption_estimators(
    treatment_col: str,
    outcome_col: str,
    y_min: float,
    y_max: float,
) -> Dict[str, Callable[[pd.DataFrame], object]]:
    """Create estimator functions for each assumption level."""

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


def analyze_q1_high_impact(df: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full Q1 high-impact analysis and return main/stability tables."""
    # Restrict to the exam subgroup and prepare threshold source column.
    bap = df[df["BAP"] == 1].copy()
    bap = bap.dropna(subset=["Tut_20161"]).copy()
    bap["Tut_20161"] = bap["Tut_20161"].astype(float)

    if bap.empty:
        return pd.DataFrame(), pd.DataFrame()

    max_k = int(np.floor(bap["Tut_20161"].max()))
    rows: List[Dict[str, float]] = []

    # Scan all thresholds k and evaluate all assumptions + support scenarios.
    for k in range(1, max_k + 1):
        # Use a copy to avoid in-place mutation that bootstrap_bounds would see incorrectly.
        bap_k = bap.copy()
        bap_k["Zk"] = (bap_k["Tut_20161"] >= k).astype(int)
        if bap_k["Zk"].nunique() < 2:
            continue

        for outcome in ["Prom__20161", "Tasa_aprob_20161"]:
            support_cases = _get_support_scenarios(bap_k, outcome)

            for support_name, (y_min, y_max) in support_cases.items():
                estimators = _build_assumption_estimators(
                    treatment_col="Zk",
                    outcome_col=outcome,
                    y_min=y_min,
                    y_max=y_max,
                )

                for assumption, estimator in estimators.items():
                    try:
                        res = estimator(bap_k)
                    except Exception:
                        continue

                    boot = bootstrap_bounds(bap_k, estimator, n_boot=n_boot, seed=seed + k)

                    rows.append(
                        {
                            "k": k,
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
                            **boot,
                        }
                    )

    main = pd.DataFrame(rows)
    if main.empty:
        return main, pd.DataFrame()

    # Build threshold stability from bootstrap selections under baseline scenario.
    stab_rows = []
    baseline = main[
        (main["support_scenario"] == "full_support")
        & (main["assumption"] == "manski")
    ].copy()

    for outcome in ["Prom__20161", "Tasa_aprob_20161"]:
        use = baseline[baseline["outcome"] == outcome].copy()
        if use.empty:
            continue

        counts = {int(k): 0 for k in sorted(use["k"].unique())}

        # Recompute best-k midpoint over bootstrap samples to measure stability.
        rng = np.random.default_rng(seed + (11 if outcome.startswith("Prom") else 22))
        for _ in range(n_boot):
            idx = rng.integers(0, len(bap), size=len(bap))
            boot_df = bap.iloc[idx].copy()

            best_k = None
            best_score = -np.inf
            for k in counts.keys():
                boot_df["Zk"] = (boot_df["Tut_20161"] >= k).astype(int)
                if boot_df["Zk"].nunique() < 2:
                    continue

                ymin, ymax = (1.0, 7.0) if outcome.startswith("Prom") else (0.0, 1.0)
                res = manski_bounds(boot_df, "Zk", outcome, ymin, ymax)
                if res.ate_midpoint > best_score:
                    best_score = res.ate_midpoint
                    best_k = k

            # Guard against best_k not being in counts (can happen if a bootstrap
            # sample has no variation for some k values).
            if best_k is not None and int(best_k) in counts:
                counts[int(best_k)] += 1

        for k, count in counts.items():
            stab_rows.append(
                {
                    "outcome": outcome,
                    "k": k,
                    "selection_count": count,
                    "selection_freq": count / max(n_boot, 1),
                }
            )

    stability = pd.DataFrame(stab_rows)
    return main, stability


def make_q1_high_impact_figures(main: pd.DataFrame, stability: pd.DataFrame, output_dir: str) -> None:
    """Create compact visual summaries for the advanced Q1 outputs (solarized theme)."""
    if main.empty:
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
    sns.set_palette([SOLARIZED["blue"], SOLARIZED["green"], SOLARIZED["cyan"], SOLARIZED["violet"]])

    # Plot assumption ladder for grade midpoint under full support.
    grade = main[(main["outcome"] == "Prom__20161") & (main["support_scenario"] == "full_support")]
    plt.figure(figsize=(12, 6), facecolor=SOLARIZED["base3"])
    ax = plt.gca()
    ax.set_facecolor(SOLARIZED["base3"])
    sns.lineplot(data=grade, x="k", y="ate_midpoint", hue="assumption", linewidth=2.2)
    plt.axhline(0, color=SOLARIZED["base01"], linestyle="--", linewidth=1.2)
    plt.title("Q1 High-Impact: Grade midpoint ATE by assumption")
    plt.xlabel("Threshold k")
    plt.ylabel("ATE midpoint")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "q1_hi_grade_midpoint_assumption_ladder.png"), dpi=180)
    plt.close()

    # Plot uncertainty width by assumption for pass-rate outcome.
    pass_df = main[(main["outcome"] == "Tasa_aprob_20161") & (main["support_scenario"] == "full_support")]
    plt.figure(figsize=(12, 6), facecolor=SOLARIZED["base3"])
    ax = plt.gca()
    ax.set_facecolor(SOLARIZED["base3"])
    sns.lineplot(data=pass_df, x="k", y="ate_width", hue="assumption", linewidth=2.2)
    plt.title("Q1 High-Impact: Pass-rate interval width by assumption")
    plt.xlabel("Threshold k")
    plt.ylabel("ATE interval width")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "q1_hi_pass_width_assumption_ladder.png"), dpi=180)
    plt.close()

    # Plot threshold stability frequencies if available.
    if not stability.empty:
        plt.figure(figsize=(12, 6), facecolor=SOLARIZED["base3"])
        ax = plt.gca()
        ax.set_facecolor(SOLARIZED["base3"])
        sns.lineplot(data=stability, x="k", y="selection_freq", hue="outcome", marker="o")
        plt.title("Q1 High-Impact: Bootstrap stability of selected threshold k")
        plt.xlabel("Threshold k")
        plt.ylabel("Selection frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "q1_hi_threshold_stability.png"), dpi=180)
        plt.close()


def write_q1_high_impact_report(main: pd.DataFrame, stability: pd.DataFrame, output_dir: str) -> str:
    """Write a short markdown report summarizing all high-impact additions."""
    report_path = os.path.join(output_dir, "q1_high_impact_report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Q1 High-Impact Partial-ID Report\n\n")
        f.write("## What was added\n")
        f.write("- Assumption ladder: Manski, MTR, MTS.\n")
        f.write("- Bootstrap endpoint uncertainty for every threshold/outcome/assumption.\n")
        f.write("- Missing-support sensitivity via full and trimmed supports.\n")
        f.write("- Threshold-stability frequencies for selected k.\n\n")

        if main.empty:
            f.write("No valid rows were produced.\n")
            return report_path

        f.write("## Assumption ladder snapshot (full support, best midpoint by outcome)\n")
        f.write("| Outcome | Assumption | Best k | Midpoint | Width |\n")
        f.write("|---|---|---:|---:|---:|\n")
        snap = main[main["support_scenario"] == "full_support"].copy()
        for outcome in ["Prom__20161", "Tasa_aprob_20161"]:
            for assumption in ["manski", "mtr", "mts"]:
                subset = snap[(snap["outcome"] == outcome) & (snap["assumption"] == assumption)]
                if subset.empty:
                    continue
                row = subset.loc[subset["ate_midpoint"].idxmax()]
                f.write(
                    f"| {outcome} | {assumption} | {int(row['k'])} | "
                    f"{row['ate_midpoint']:.4f} | {row['ate_width']:.4f} |\n"
                )

        if not stability.empty:
            f.write("\n## Threshold stability highlights\n")
            top = stability.sort_values("selection_freq", ascending=False).head(8)
            for _, row in top.iterrows():
                f.write(
                    f"- {row['outcome']}: k={int(row['k'])} selected "
                    f"{row['selection_freq']:.2%} of bootstrap samples.\n"
                )

    return report_path


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for the high-impact Q1 script."""
    parser = argparse.ArgumentParser(description="Q1 high-impact partial-ID analysis")
    parser.add_argument("--data-path", default="Data2026.xls", help="Input dataset path")
    parser.add_argument("--output-dir", default="outputs_q1_high_impact", help="Output directory")
    parser.add_argument("--n-boot", type=int, default=120, help="Bootstrap draws per configuration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def run(args: argparse.Namespace) -> None:
    """Execute the full high-impact Q1 pipeline."""
    data_path = args.data_path if os.path.isabs(args.data_path) else os.path.join(PROJECT_DIR, args.data_path)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(PROJECT_DIR, args.output_dir)

    df = pd.read_excel(data_path)
    validate_required_columns(df)
    df["BAP"] = df["BAP"].astype(int)
    df["VAI"] = df["VAI"].astype(int)

    ensure_output_dir(output_dir)

    main, stability = analyze_q1_high_impact(df, n_boot=args.n_boot, seed=args.seed)

    main.to_csv(os.path.join(output_dir, "q1_high_impact_bounds.csv"), index=False)
    stability.to_csv(os.path.join(output_dir, "q1_high_impact_threshold_stability.csv"), index=False)

    make_q1_high_impact_figures(main, stability, output_dir)
    report_path = write_q1_high_impact_report(main, stability, output_dir)

    print("Q1 high-impact analysis complete.")
    print(f"Rows in advanced bounds table: {len(main)}")
    print(f"Rows in stability table: {len(stability)}")
    print(f"Output directory: {output_dir}")
    print(f"Report file: {report_path}")


if __name__ == "__main__":
    run(build_parser().parse_args())

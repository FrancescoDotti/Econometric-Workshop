#!/usr/bin/env python3
"""Bivariate (k, y*) sweep of CTE bounds for Q1.

For BAP students in Semester 1, computes Manski bounds on

    CTE_k(y*) = P(Y_1 <= y*) - P(Y_0 <= y*)

over a fine grid of treatment-intensity thresholds k and outcome thresholds y*.
Produces:
  - Heatmap of the Manski upper bound across (k, y*), highlighting the
    sign-identification frontier (cells where Manski_U < 0 give a robustly
    negative CTE, i.e. tutoring stochastically dominates no-tutoring at that
    outcome threshold).
  - Heatmap of bound widths (Manski_U - Manski_L) across (k, y*).
  - CSV with all bounds at every grid point.

Sign convention: NEGATIVE CTE means treated dominates control at threshold y*.

The CTE bounds formulas are identical to those in cte_q1_sensitivity.py;
duplicated here for self-containment.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exam2026_core import validate_required_columns

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOLARIZED = {
    "base01": "#586e75", "base00": "#657b83", "base1": "#93a1a1",
    "base2": "#eee8d5", "base3": "#fdf6e3",
}


def apply_theme() -> None:
    plt.rcParams["figure.facecolor"] = SOLARIZED["base3"]
    plt.rcParams["axes.facecolor"] = SOLARIZED["base3"]
    plt.rcParams["axes.edgecolor"] = SOLARIZED["base1"]
    plt.rcParams["axes.labelcolor"] = SOLARIZED["base00"]
    plt.rcParams["text.color"] = SOLARIZED["base00"]
    plt.rcParams["xtick.color"] = SOLARIZED["base00"]
    plt.rcParams["ytick.color"] = SOLARIZED["base00"]
    plt.rcParams["grid.color"] = SOLARIZED["base2"]
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def cte_manski(y: np.ndarray, z: np.ndarray, y_star: float) -> dict | None:
    """Manski bounds on CTE_k(y*). See cte_q1_sensitivity.py for the derivation."""
    mask = ~np.isnan(y)
    y, z = y[mask], z[mask]
    n = len(y); n1 = int((z == 1).sum()); n0 = n - n1
    if n1 == 0 or n0 == 0:
        return None
    p1 = n1 / n; p0 = 1.0 - p1
    F1 = float((y[z == 1] <= y_star).mean())
    F0 = float((y[z == 0] <= y_star).mean())
    cte_L = F1 * p1 - (F0 * p0 + p1)
    cte_U = F1 * p1 + p0 - F0 * p0
    return {
        "n": n, "p_treated": p1, "F1": F1, "F0": F0,
        "naive": F1 - F0, "manski_L": cte_L, "manski_U": cte_U,
        "midpoint": 0.5 * (cte_L + cte_U), "width": cte_U - cte_L,
    }


def sweep_grid(
    df: pd.DataFrame, outcome: str, y_grid: np.ndarray, k_grid: list[int]
) -> pd.DataFrame:
    """Compute CTE Manski bounds at every (k, y*) cell on the grid."""
    bap = df[df["BAP"] == 1].copy()
    bap = bap.dropna(subset=["Tut_20161"])
    rows = []
    tut = bap["Tut_20161"].to_numpy()
    out_arr = bap[outcome].to_numpy(dtype=float)
    for k in k_grid:
        z = (tut >= k).astype(int)
        for y_star in y_grid:
            b = cte_manski(out_arr, z, float(y_star))
            if b is None:
                continue
            b.update({"k": k, "y_star": float(y_star), "outcome": outcome})
            rows.append(b)
    return pd.DataFrame(rows)


def plot_heatmap(
    results: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: str,
    cmap: str = "RdBu_r",
    center: float | None = 0.0,
) -> None:
    """Render a (k x y*) heatmap of the chosen value column."""
    apply_theme()
    pivot = results.pivot(index="y_star", columns="k", values=value_col)
    # Higher y* at the top for grade-like outcomes (more demanding threshold above).
    pivot = pivot.sort_index(ascending=False)
    arr = pivot.to_numpy(dtype=float)
    if center is not None:
        vmax = float(np.nanmax(np.abs(arr)))
        vmin = -vmax
    else:
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(
        arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        extent=[pivot.columns.min() - 0.5, pivot.columns.max() + 0.5,
                len(pivot.index) - 0.5, -0.5],
        interpolation="nearest",
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
    xt = [int(c) for c in pivot.columns if int(c) % 2 == 0]
    ax.set_xticks(xt)
    ax.set_xticklabels(xt)
    ax.set_xlabel("Threshold k (sessions)")
    ax.set_ylabel(r"Outcome threshold $y^*$")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(value_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_sign_frontier_naive(results: pd.DataFrame, out_path: str, outcome_label: str) -> None:
    """Boolean heatmap: 1 if naive CTE = F1-F0 < 0 (treatment dominates control at y*).

    Under MTS, the naive CTE F1-F0 equals the lower bound of the CTE interval,
    so naive < 0 corresponds to a robustly negative CTE under the MTS assumption
    (or equivalently, under exogenous treatment assignment / strong ignorability).
    """
    apply_theme()
    pivot = results.pivot(index="y_star", columns="k", values="naive")
    pivot = pivot.sort_index(ascending=False)
    arr = (pivot.to_numpy(dtype=float) < 0).astype(int)
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(
        arr, aspect="auto", cmap="Greens", vmin=0, vmax=1,
        extent=[pivot.columns.min() - 0.5, pivot.columns.max() + 0.5,
                len(pivot.index) - 0.5, -0.5],
        interpolation="nearest",
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
    xt = [int(c) for c in pivot.columns if int(c) % 2 == 0]
    ax.set_xticks(xt)
    ax.set_xticklabels(xt)
    ax.set_xlabel("Threshold k (sessions)")
    ax.set_ylabel(r"Outcome threshold $y^*$")
    ax.set_title(
        f"Naive CTE sign frontier ({outcome_label}): green = $F_1-F_0<0$ "
        f"(treated dominates control at threshold $y^*$, under MTS / exogenous switching)"
    )
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=[0, 1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Q1 bivariate (k, y*) CTE sweep")
    parser.add_argument("--data-path", default="Data2026.xls", help="Input dataset path")
    parser.add_argument("--output-dir", default="outputs_cte_q1",
                        help="Output directory")
    parser.add_argument("--k-max", type=int, default=30,
                        help="Maximum threshold k to sweep")
    parser.add_argument("--y-grade-points", type=int, default=25,
                        help="Number of y* values for the grade outcome (in [1,7])")
    parser.add_argument("--y-passrate-points", type=int, default=21,
                        help="Number of y* values for the pass-rate outcome (in [0,1])")
    return parser


def run(args: argparse.Namespace) -> None:
    data_path = (
        args.data_path
        if os.path.isabs(args.data_path)
        else os.path.join(PROJECT_DIR, args.data_path)
    )
    output_dir = (
        args.output_dir
        if os.path.isabs(args.output_dir)
        else os.path.join(PROJECT_DIR, args.output_dir)
    )
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.read_excel(data_path)
    validate_required_columns(df)
    df["BAP"] = df["BAP"].astype(int)

    k_grid = list(range(2, args.k_max + 1))
    y_grid_grade = np.linspace(1.5, 6.5, args.y_grade_points)
    y_grid_pass = np.linspace(0.1, 0.95, args.y_passrate_points)

    results_grade = sweep_grid(df, "Prom__20161", y_grid_grade, k_grid)
    results_pass = sweep_grid(df, "Tasa_aprob_20161", y_grid_pass, k_grid)

    combined = pd.concat([results_grade, results_pass], ignore_index=True)
    csv_path = os.path.join(output_dir, "cte_q1_bivariate_bounds.csv")
    combined.to_csv(csv_path, index=False)

    # Three views per outcome:
    #  - Manski upper bound (always non-negative under no-assumption counterfactual
    #    partial-id; useful for showing how wide the bound stays)
    #  - Manski midpoint (a midpoint summary of the bound interval)
    #  - Naive CTE = F1 - F0 (the Jauder-style under-exogenous-switching estimate,
    #    coincides with the MTS lower bound and with the strong-ignorability point)
    plot_heatmap(
        results_grade, "manski_U",
        r"Manski CTE upper bound vs $(k, y^*)$ -- grade outcome (always $\geq 0$)",
        os.path.join(fig_dir, "cte_q1_heatmap_grade_manski_upper.png"),
        center=None, cmap="Reds",
    )
    plot_heatmap(
        results_grade, "midpoint",
        r"Manski CTE midpoint vs $(k, y^*)$ -- grade outcome",
        os.path.join(fig_dir, "cte_q1_heatmap_grade_midpoint.png"),
    )
    plot_heatmap(
        results_grade, "naive",
        r"Naive CTE $F_1-F_0$ (= MTS lower bound = SI point) vs $(k, y^*)$ -- grade outcome",
        os.path.join(fig_dir, "cte_q1_heatmap_grade_naive.png"),
    )
    plot_sign_frontier_naive(
        results_grade,
        os.path.join(fig_dir, "cte_q1_signfrontier_grade.png"),
        "grade",
    )

    plot_heatmap(
        results_pass, "manski_U",
        r"Manski CTE upper bound vs $(k, y^*)$ -- pass-rate outcome (always $\geq 0$)",
        os.path.join(fig_dir, "cte_q1_heatmap_passrate_manski_upper.png"),
        center=None, cmap="Reds",
    )
    plot_heatmap(
        results_pass, "midpoint",
        r"Manski CTE midpoint vs $(k, y^*)$ -- pass-rate outcome",
        os.path.join(fig_dir, "cte_q1_heatmap_passrate_midpoint.png"),
    )
    plot_heatmap(
        results_pass, "naive",
        r"Naive CTE $F_1-F_0$ (= MTS lower bound = SI point) vs $(k, y^*)$ -- pass-rate outcome",
        os.path.join(fig_dir, "cte_q1_heatmap_passrate_naive.png"),
    )
    plot_sign_frontier_naive(
        results_pass,
        os.path.join(fig_dir, "cte_q1_signfrontier_passrate.png"),
        "pass-rate",
    )

    n_grade_naive_neg = int((results_grade["naive"] < 0).sum())
    n_pass_naive_neg = int((results_pass["naive"] < 0).sum())
    print("CTE Q1 bivariate sweep complete.")
    print(
        f"Grade grid:    {len(results_grade)} cells; "
        f"{n_grade_naive_neg} with naive CTE < 0 (treatment dominates at threshold)."
    )
    print(
        f"Pass-rate grid:{len(results_pass)} cells; "
        f"{n_pass_naive_neg} with naive CTE < 0 (treatment dominates at threshold)."
    )
    print("Note: Manski upper bound is always >= 0 under no-assumption counterfactual")
    print("partial-id, so the 'sign frontier' is shown for the naive (= MTS-lower-")
    print("bound = strong-ignorability) estimate, which is the Jauder-style view.")
    print(f"CSV: {csv_path}")
    print(f"Figures in: {fig_dir}")


if __name__ == "__main__":
    run(build_parser().parse_args())

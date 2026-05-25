#!/usr/bin/env python3
"""Classical Treatment Effect (CTE) sensitivity analysis for Q1.

For BAP students in Semester 1, computes partial-identification bounds on the
Classical Treatment Effect parameter

    CTE_k(y*) = P(Y_1 <= y*) - P(Y_0 <= y*),

where Z_k = 1[Tut_20161 >= k] is the treatment intensity indicator. CTE is a
parameter that respects stochastic dominance (lecture section 3.3) and is
typically easier to identify than the ATE on the [1,7] grade scale, because
the unidentified counterfactual probabilities live in [0,1].

Sign convention: NEGATIVE CTE means the treated outcome distribution
stochastically dominates the control distribution at threshold y* (less mass
below threshold for treated), so "tutoring helps" corresponds to a robustly
NEGATIVE upper bound on CTE.

This script produces line plots of CTE bounds versus k at canonical y* values
(passing grade y*=4, sample-mean grade y*=4.74; pass-rate y*=0.6, y*=0.8).

For the bivariate (k, y*) heatmap see cte_q1_bivariate.py.
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
    "base2": "#eee8d5", "base3": "#fdf6e3", "yellow": "#b58900",
    "orange": "#cb4b16", "red": "#dc322f", "violet": "#6c71c4",
    "blue": "#268bd2", "green": "#859900",
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
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def cte_bounds(y: np.ndarray, z: np.ndarray, y_star: float) -> dict | None:
    """Compute Manski / MTR / MTS bounds on CTE_k(y*).

    Manski:
      P(Y_1<=y*) in [F1*p1, F1*p1 + p0],  P(Y_0<=y*) in [F0*p0, F0*p0 + p1]
      so CTE_L = F1*p1 - (F0*p0 + p1), CTE_U = F1*p1 + p0 - F0*p0.

    MTR (Y_1 >= Y_0 pointwise => CTE <= 0):
      P(Y_1<=y*|Z=0) in [0, F0],  P(Y_0<=y*|Z=1) in [F1, 1]
      Algebraically: CTE_L^MTR = Manski_L; CTE_U^MTR = 0.

    Distribution-MTS (F_{Y_t|Z=1} dominates F_{Y_t|Z=0}):
      P(Y_1<=y*|Z=0) in [F1, 1],  P(Y_0<=y*|Z=1) in [0, F0]
      CTE_L^MTS = F1 - F0 (naive), CTE_U^MTS = Manski_U.
    """
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    mask = ~np.isnan(y)
    y, z = y[mask], z[mask]
    n = len(y)
    n1 = int((z == 1).sum())
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return None
    p1 = n1 / n
    p0 = 1.0 - p1
    F1 = float((y[z == 1] <= y_star).mean())
    F0 = float((y[z == 0] <= y_star).mean())
    cte_L = F1 * p1 - (F0 * p0 + p1)
    cte_U = F1 * p1 + p0 - F0 * p0
    return {
        "n": n, "n_treated": n1, "n_control": n0, "p_treated": p1,
        "F1": F1, "F0": F0, "naive": F1 - F0,
        "manski_L": cte_L, "manski_U": cte_U,
        "mtr_L": cte_L, "mtr_U": 0.0,
        "mts_L": F1 - F0, "mts_U": cte_U,
    }


def sweep_k(
    df: pd.DataFrame, outcome: str, y_star: float, k_grid: list[int]
) -> pd.DataFrame:
    """Sweep CTE bounds across k for a single (outcome, y*) combination."""
    rows = []
    bap = df[df["BAP"] == 1].copy()
    bap = bap.dropna(subset=["Tut_20161"])
    for k in k_grid:
        z = (bap["Tut_20161"] >= k).astype(int).to_numpy()
        y = bap[outcome].to_numpy(dtype=float)
        b = cte_bounds(y, z, y_star)
        if b is None:
            continue
        b.update({"k": k, "outcome": outcome, "y_star": y_star})
        rows.append(b)
    return pd.DataFrame(rows)


def plot_cte_vs_k(results: pd.DataFrame, outcome: str, y_star: float, out_path: str) -> None:
    """Reproduce a Jauder-style CTE-bounds-vs-k figure with Manski + MTR + MTS."""
    apply_theme()
    fig, ax = plt.subplots(figsize=(12, 6))
    r = results.sort_values("k")
    ax.plot(r["k"], r["manski_L"], color=SOLARIZED["violet"], linestyle="--",
            linewidth=2.0, marker="o", markersize=3, label="Manski lower")
    ax.plot(r["k"], r["manski_U"], color=SOLARIZED["violet"], linestyle="-",
            linewidth=2.0, marker="o", markersize=3, label="Manski upper")
    ax.plot(r["k"], r["mtr_U"], color=SOLARIZED["green"], linestyle="-",
            linewidth=2.0, label="MTR upper (= 0)")
    ax.plot(r["k"], r["mts_L"], color=SOLARIZED["orange"], linestyle="--",
            linewidth=2.0, marker="s", markersize=3, label="MTS lower (= naive)")
    ax.plot(r["k"], r["naive"], color=SOLARIZED["base01"], linestyle=":",
            linewidth=1.8, label="Naive CTE = F1 - F0")
    ax.axhline(0.0, color=SOLARIZED["base01"], linestyle=":", linewidth=1.0)
    ax.set_xlabel("Threshold k (sessions)")
    ax.set_ylabel(r"$\mathrm{CTE}_k(y^*)$")
    pretty = "grade" if outcome.startswith("Prom") else "approval rate"
    ax.set_title(
        f"CTE bounds vs k -- BAP students, S1, outcome = {pretty}, "
        rf"$y^*={y_star}$"
    )
    ax.legend(loc="best", ncol=2, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_report(all_results: pd.DataFrame, output_dir: str) -> str:
    report_path = os.path.join(output_dir, "cte_q1_sensitivity_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# CTE Sensitivity Report (Q1, BAP students, Semester 1)\n\n")
        f.write("## Method\n")
        f.write("- Parameter: CTE_k(y*) = P(Y_1 <= y*) - P(Y_0 <= y*).\n")
        f.write("- Sign: negative CTE means treated stochastically dominates control.\n")
        f.write("- Three identification levels: Manski, MTR (CTE <= 0), MTS (stochastic-dominance selection).\n\n")
        f.write("## Sign-identification highlights\n")
        f.write("Rows below report the smallest k at which the Manski upper bound becomes negative ")
        f.write("(robust sign of CTE under the no-assumption model). NA if never identified.\n\n")
        f.write("| Outcome | y* | first k with CTE_U < 0 | Manski_U at that k | n |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for (outcome, y_star), grp in all_results.groupby(["outcome", "y_star"]):
            grp = grp.sort_values("k")
            neg = grp[grp["manski_U"] < 0]
            if neg.empty:
                first_k = "NA"
                first_U = "NA"
                n = int(grp.iloc[0]["n"])
            else:
                row = neg.iloc[0]
                first_k = int(row["k"])
                first_U = f"{row['manski_U']:.4f}"
                n = int(row["n"])
            f.write(f"| {outcome} | {y_star} | {first_k} | {first_U} | {n} |\n")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Q1 CTE sensitivity at canonical y* values")
    parser.add_argument("--data-path", default="Data2026.xls", help="Input dataset path")
    parser.add_argument("--output-dir", default="outputs_cte_q1",
                        help="Output directory")
    parser.add_argument("--k-max", type=int, default=30, help="Maximum threshold k to sweep")
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

    # Canonical y* values: aligned with Jauder. Grade y*=4 is the passing
    # threshold; y*=4.74 is the BAP-sample mean (descriptive Table 1); y*=5
    # is a "good performance" cutoff. Pass-rate y*=0.6 separates failing
    # from majority pass; y*=0.8 marks consistent course success.
    canonical = [
        ("Prom__20161", 4.0),
        ("Prom__20161", 4.74),
        ("Prom__20161", 5.0),
        ("Tasa_aprob_20161", 0.6),
        ("Tasa_aprob_20161", 0.8),
    ]

    all_results = []
    for outcome, y_star in canonical:
        res = sweep_k(df, outcome, y_star, k_grid)
        if res.empty:
            continue
        all_results.append(res)
        tag = "grade" if outcome.startswith("Prom") else "passrate"
        fname = f"cte_q1_{tag}_ystar{y_star}.png".replace(".", "_", 1)
        plot_cte_vs_k(res, outcome, y_star, os.path.join(fig_dir, fname))

    combined = pd.concat(all_results, ignore_index=True)
    csv_path = os.path.join(output_dir, "cte_q1_sensitivity_bounds.csv")
    combined.to_csv(csv_path, index=False)
    report_path = write_report(combined, output_dir)

    print("CTE Q1 sensitivity analysis complete.")
    print(f"Rows in results table: {len(combined)}")
    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Figures in: {fig_dir}")


if __name__ == "__main__":
    run(build_parser().parse_args())

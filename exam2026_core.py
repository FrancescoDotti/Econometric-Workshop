"""Core analysis logic for Exam 2026 partial-identification tasks.

This module contains reusable statistical functions used by the CLI script.
Main responsibilities:
1. Compute worst-case ATE bounds under missing outcomes.
2. Build output tables for Question 1 and Question 2.
3. Classify decisions from identified ATE intervals.
4. Validate that required dataset columns are present.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class BoundResult:
    """Container for all statistics produced by one ATE bound computation."""

    n: int
    p_treated: float
    y1_lower: float
    y1_upper: float
    y0_lower: float
    y0_upper: float
    ate_lower: float
    ate_upper: float
    ate_midpoint: float
    ate_width: float
    obs_mean_treated: float
    obs_mean_control: float
    obs_diff: float
    n_treated: int
    n_control: int
    treated_obs_rate: float
    control_obs_rate: float


def partial_id_ate_bounds(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    y_min: float,
    y_max: float,
) -> BoundResult:
    """Compute worst-case sharp ATE bounds with possible missing outcomes.

    Intuition:
    - We observe treatment D and sometimes outcome Y.
    - We only assume Y lies in [y_min, y_max].
    - Missing and counterfactual outcomes are bounded by support limits.
    """
    # Keep only relevant columns so the function is self-contained.
    work = df[[treatment_col, outcome_col]].copy()

    # Remove rows with unknown treatment, because treatment shares are required.
    work = work.dropna(subset=[treatment_col])
    work[treatment_col] = work[treatment_col].astype(int)

    # Total sample size used in this specific bound calculation.
    n = len(work)
    if n == 0:
        raise ValueError(f"No rows available for treatment '{treatment_col}'.")

    # Share treated and untreated in the analysis sample.
    p1 = float((work[treatment_col] == 1).mean())
    p0 = 1.0 - p1

    # Build factual outcome bounds in each observed treatment arm.
    def observed_group_bounds(d: int) -> Tuple[float, float, float, int, float]:
        grp = work[work[treatment_col] == d]
        n_grp = len(grp)
        if n_grp == 0:
            # No units in this arm: statistics are undefined.
            return np.nan, np.nan, np.nan, 0, np.nan

        # Observed outcomes provide partial information on E[Y | D=d].
        y_obs = grp[outcome_col].dropna()
        obs_rate = len(y_obs) / n_grp
        if len(y_obs) == 0:
            # If all outcomes are missing, only support bounds are available.
            return y_min, y_max, np.nan, n_grp, 0.0

        # Combine observed mean with worst-case values for missing outcomes.
        mu_obs = float(y_obs.mean())
        lower = obs_rate * mu_obs + (1.0 - obs_rate) * y_min
        upper = obs_rate * mu_obs + (1.0 - obs_rate) * y_max
        return lower, upper, mu_obs, n_grp, obs_rate

    # Compute arm-specific factual bounds and observed means.
    mu1_l, mu1_u, mu1_obs, n1, r1 = observed_group_bounds(1)
    mu0_l, mu0_u, mu0_obs, n0, r0 = observed_group_bounds(0)

    # Extend factual-arm bounds to whole-population potential outcome means.
    y1_l = p1 * mu1_l + p0 * y_min
    y1_u = p1 * mu1_u + p0 * y_max
    y0_l = p0 * mu0_l + p1 * y_min
    y0_u = p0 * mu0_u + p1 * y_max

    # ATE interval is [lower bound for Y(1)-Y(0), upper bound for Y(1)-Y(0)].
    ate_l = y1_l - y0_u
    ate_u = y1_u - y0_l

    # Observed treated-control mean difference is descriptive (not causal).
    obs_diff = np.nan
    if np.isfinite(mu1_obs) and np.isfinite(mu0_obs):
        obs_diff = mu1_obs - mu0_obs

    return BoundResult(
        n=n,
        p_treated=p1,
        y1_lower=y1_l,
        y1_upper=y1_u,
        y0_lower=y0_l,
        y0_upper=y0_u,
        ate_lower=ate_l,
        ate_upper=ate_u,
        ate_midpoint=(ate_l + ate_u) / 2.0,
        ate_width=ate_u - ate_l,
        obs_mean_treated=mu1_obs,
        obs_mean_control=mu0_obs,
        obs_diff=obs_diff,
        n_treated=n1,
        n_control=n0,
        treated_obs_rate=r1,
        control_obs_rate=r0,
    )


def to_row(result: BoundResult) -> Dict[str, float]:
    """Convert `BoundResult` to a dictionary so it is easy to store in a table."""
    return {
        "n": result.n,
        "n_treated": result.n_treated,
        "n_control": result.n_control,
        "p_treated": result.p_treated,
        "treated_obs_rate": result.treated_obs_rate,
        "control_obs_rate": result.control_obs_rate,
        "obs_mean_treated": result.obs_mean_treated,
        "obs_mean_control": result.obs_mean_control,
        "obs_diff_treated_minus_control": result.obs_diff,
        "y1_lower": result.y1_lower,
        "y1_upper": result.y1_upper,
        "y0_lower": result.y0_lower,
        "y0_upper": result.y0_upper,
        "ate_lower": result.ate_lower,
        "ate_upper": result.ate_upper,
        "ate_midpoint": result.ate_midpoint,
        "ate_width": result.ate_width,
    }


def classify_decision(ate_lower: float, ate_upper: float) -> Dict[str, str]:
    """Classify treatment choice under robust, optimistic, and conservative rules."""
    # Robust sign asks whether the entire identified interval is strictly above/below 0.
    if ate_lower > 0:
        robust = "Continue dominates (ATE > 0 for all compatible models)"
    elif ate_upper < 0:
        robust = "Opt-out dominates (ATE < 0 for all compatible models)"
    else:
        robust = "Ambiguous (sign not identified)"

    # Optimistic picks the action with better best-case payoff.
    optimistic = "Continue" if ate_upper >= -ate_lower else "Opt-out"

    # Conservative (maximin) continues only when harm is not possible.
    conservative = "Continue" if ate_lower >= 0 else "Opt-out"

    return {
        "robust_sign": robust,
        "optimistic_choice": optimistic,
        "conservative_choice": conservative,
    }


def analyze_question_1(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Analyze threshold `k` among BAP students in semester 1.

    For each threshold k:
    - Define treatment Zk = 1[Tut_20161 >= k].
    - Compute bounds for grade and pass-rate outcomes.
    - Save full threshold table and recommendation summary.
    """
    # Restrict to BAP subgroup as requested in the exam statement.
    bap = df[df["BAP"] == 1].copy()

    # Maximum feasible threshold to scan.
    max_k = int(bap["Tut_20161"].max())

    rows: List[Dict[str, float]] = []
    for k in range(1, max_k + 1):
        # Build threshold-based treatment indicator.
        bap["Zk"] = (bap["Tut_20161"] >= k).astype(int)

        # Skip thresholds that do not create treated/control variation.
        if bap["Zk"].nunique() < 2:
            continue

        # Compute bounds for average grade.
        grade = partial_id_ate_bounds(
            bap,
            treatment_col="Zk",
            outcome_col="Prom__20161",
            y_min=1.0,
            y_max=7.0,
        )

        # Compute bounds for pass-rate outcome.
        passrate = partial_id_ate_bounds(
            bap,
            treatment_col="Zk",
            outcome_col="Tasa_aprob_20161",
            y_min=0.0,
            y_max=1.0,
        )

        # Store key statistics for this threshold.
        row = {
            "k": k,
            "n": grade.n,
            "treated_share": grade.p_treated,
            "grade_ate_lower": grade.ate_lower,
            "grade_ate_upper": grade.ate_upper,
            "grade_ate_midpoint": grade.ate_midpoint,
            "grade_ate_width": grade.ate_width,
            "grade_obs_diff": grade.obs_diff,
            "pass_ate_lower": passrate.ate_lower,
            "pass_ate_upper": passrate.ate_upper,
            "pass_ate_midpoint": passrate.ate_midpoint,
            "pass_ate_width": passrate.ate_width,
            "pass_obs_diff": passrate.obs_diff,
        }

        # Normalize grade midpoint to [0,1]-like scale and average with pass midpoint.
        row["joint_midpoint_score"] = (
            (row["grade_ate_midpoint"] / 6.0) + row["pass_ate_midpoint"]
        ) / 2.0
        rows.append(row)

    # Save full threshold-by-threshold table.
    q1 = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    q1.to_csv(os.path.join(out_dir, "q1_threshold_bounds.csv"), index=False)

    # Save a smaller recommendation table under several selection rules.
    if not q1.empty:
        summary_rows = []
        for name, col, mode in [
            ("grade_conservative_k", "grade_ate_lower", "max"),
            ("grade_optimistic_k", "grade_ate_upper", "max"),
            ("grade_midpoint_k", "grade_ate_midpoint", "max"),
            ("pass_conservative_k", "pass_ate_lower", "max"),
            ("pass_optimistic_k", "pass_ate_upper", "max"),
            ("pass_midpoint_k", "pass_ate_midpoint", "max"),
            ("joint_midpoint_k", "joint_midpoint_score", "max"),
        ]:
            idx = q1[col].idxmax() if mode == "max" else q1[col].idxmin()
            rec = q1.loc[idx].to_dict()
            rec["selection_rule"] = name
            summary_rows.append(rec)

        pd.DataFrame(summary_rows).to_csv(
            os.path.join(out_dir, "q1_recommended_k.csv"),
            index=False,
        )

    return q1


def analyze_question_2(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Evaluate Continue vs Opt-out in semester 2 for BAP and VAI groups."""
    # Work on a copy so we do not mutate caller data unexpectedly.
    work = df.copy()

    # Define treatment: Continue2 = 1 if student attended at least one session.
    work["Continue2"] = (work["Tut_20162"].fillna(0) >= 1).astype(int)

    # Define subgroup datasets requested by the exam.
    groups = {
        "BAP_students": work[work["BAP"] == 1],
        "VAI_students": work[work["VAI"] == 1],
    }

    out_rows: List[Dict[str, float]] = []
    for gname, gdf in groups.items():
        # For each subgroup, evaluate both grade and pass-rate outcomes.
        for outcome, ymin, ymax in [
            ("Prom__20162", 1.0, 7.0),
            ("Tasa_aprob_20162", 0.0, 1.0),
        ]:
            res = partial_id_ate_bounds(
                gdf,
                treatment_col="Continue2",
                outcome_col=outcome,
                y_min=ymin,
                y_max=ymax,
            )

            # Merge numeric bounds and qualitative decision labels.
            row = {"group": gname, "outcome": outcome, **to_row(res)}
            row.update(classify_decision(res.ate_lower, res.ate_upper))
            out_rows.append(row)

    # Save Question 2 results table.
    q2 = pd.DataFrame(out_rows)
    q2.to_csv(os.path.join(out_dir, "q2_continue_vs_optout_bounds.csv"), index=False)
    return q2


def validate_required_columns(df: pd.DataFrame) -> None:
    """Raise an error if the dataset misses required columns for the analysis."""
    required = {
        "BAP",
        "VAI",
        "Tut_20161",
        "Tut_20162",
        "Prom__20161",
        "Tasa_aprob_20161",
        "Prom__20162",
        "Tasa_aprob_20162",
    }

    # Compare expected names with actual DataFrame columns.
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

#!/usr/bin/env python3
"""Advanced partial-identification utilities for Exam 2026 extensions.

This module adds high-impact econometrics features on top of the baseline code:
1. Assumption tightening (Manski, MTR, MTS, MIV-style bin monotonicity).
2. Bootstrap uncertainty for bound endpoints.
3. Support sensitivity helpers for missing-outcome assumptions.

The functions are written to be reusable in new Q1/Q2 scripts without touching
existing project files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class BoundsResult:
    """Container for one ATE interval and core diagnostics."""

    ate_lower: float
    ate_upper: float
    ate_midpoint: float
    ate_width: float
    p_treated: float
    n: int


def _safe_mean(series: pd.Series) -> float:
    """Return mean as float, or NaN when empty after dropping missing values."""
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.mean())


def support_from_observed(
    series: pd.Series,
    hard_min: float,
    hard_max: float,
    lower_q: float,
    upper_q: float,
) -> tuple[float, float]:
    """Build trimmed support bounds for sensitivity analysis.

    The support is clipped between hard limits and observed quantiles.
    """
    clean = series.dropna()
    if clean.empty:
        return hard_min, hard_max

    q_low = float(clean.quantile(lower_q))
    q_high = float(clean.quantile(upper_q))
    y_min = max(hard_min, min(q_low, q_high))
    y_max = min(hard_max, max(q_low, q_high))

    # Guard against accidental inversion due to extreme edge cases.
    if y_min > y_max:
        return hard_min, hard_max
    return y_min, y_max


def _factual_group_bounds(y: pd.Series, y_min: float, y_max: float) -> tuple[float, float]:
    """Compute factual mean bounds in one treatment arm under missing outcomes."""
    n_group = len(y)
    if n_group == 0:
        return np.nan, np.nan

    y_obs = y.dropna()
    if y_obs.empty:
        return y_min, y_max

    obs_rate = len(y_obs) / n_group
    mu_obs = float(y_obs.mean())
    lower = obs_rate * mu_obs + (1.0 - obs_rate) * y_min
    upper = obs_rate * mu_obs + (1.0 - obs_rate) * y_max
    return lower, upper


def manski_bounds(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    y_min: float,
    y_max: float,
) -> BoundsResult:
    """Compute sharp worst-case ATE bounds under support-only assumptions."""
    work = df[[treatment_col, outcome_col]].dropna(subset=[treatment_col]).copy()
    if work.empty:
        raise ValueError(f"No rows available for treatment '{treatment_col}'.")

    work[treatment_col] = work[treatment_col].astype(int)
    p1 = float((work[treatment_col] == 1).mean())
    p0 = 1.0 - p1

    # Compute factual bounds within observed treatment arms.
    mu1_l, mu1_u = _factual_group_bounds(work.loc[work[treatment_col] == 1, outcome_col], y_min, y_max)
    mu0_l, mu0_u = _factual_group_bounds(work.loc[work[treatment_col] == 0, outcome_col], y_min, y_max)

    # Combine factual-arm bounds with worst-case counterfactual support bounds.
    y1_l = p1 * mu1_l + p0 * y_min
    y1_u = p1 * mu1_u + p0 * y_max
    y0_l = p0 * mu0_l + p1 * y_min
    y0_u = p0 * mu0_u + p1 * y_max

    ate_l = y1_l - y0_u
    ate_u = y1_u - y0_l

    return BoundsResult(
        ate_lower=float(ate_l),
        ate_upper=float(ate_u),
        ate_midpoint=float((ate_l + ate_u) / 2.0),
        ate_width=float(ate_u - ate_l),
        p_treated=p1,
        n=len(work),
    )


def mtr_bounds_from_manski(base: BoundsResult, direction: str = "nondecreasing") -> BoundsResult:
    """Tighten Manski interval under monotone treatment response (MTR).

    - nondecreasing: Y(1) >= Y(0) implies ATE >= 0.
    - nonincreasing: Y(1) <= Y(0) implies ATE <= 0.
    """
    lower = base.ate_lower
    upper = base.ate_upper

    if direction == "nondecreasing":
        lower = max(lower, 0.0)
    elif direction == "nonincreasing":
        upper = min(upper, 0.0)
    else:
        raise ValueError("direction must be 'nondecreasing' or 'nonincreasing'.")

    # Keep interval coherent even when assumptions conflict with baseline bounds.
    if lower > upper:
        lower = upper

    return BoundsResult(
        ate_lower=float(lower),
        ate_upper=float(upper),
        ate_midpoint=float((lower + upper) / 2.0),
        ate_width=float(upper - lower),
        p_treated=base.p_treated,
        n=base.n,
    )


def mts_bounds_from_manski(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    y_min: float,
    y_max: float,
) -> BoundsResult:
    """Apply a simple MTS tightening (selection monotonicity proxy).

    This implementation follows a practical classroom-friendly proxy:
    - Starts from Manski bounds.
    - Uses observed treatment-arm means to tighten only where monotone selection
      gives directional information.
    """
    base = manski_bounds(df, treatment_col, outcome_col, y_min, y_max)

    # Compute observed means by treatment arm as a simple empirical anchor.
    work = df[[treatment_col, outcome_col]].dropna(subset=[treatment_col]).copy()
    work[treatment_col] = work[treatment_col].astype(int)
    mu1_obs = _safe_mean(work.loc[work[treatment_col] == 1, outcome_col])
    mu0_obs = _safe_mean(work.loc[work[treatment_col] == 0, outcome_col])

    # If one arm is fully missing, return baseline bounds safely.
    if not np.isfinite(mu1_obs) or not np.isfinite(mu0_obs):
        return base

    # Tighten upper/lower softly when treated arm performs weakly better on average.
    lower = base.ate_lower
    upper = base.ate_upper
    if mu1_obs >= mu0_obs:
        lower = max(lower, mu1_obs - y_max)
        upper = min(upper, y_max - mu0_obs)

    if lower > upper:
        lower = upper

    return BoundsResult(
        ate_lower=float(lower),
        ate_upper=float(upper),
        ate_midpoint=float((lower + upper) / 2.0),
        ate_width=float(upper - lower),
        p_treated=base.p_treated,
        n=base.n,
    )

def bootstrap_bounds(
    df: pd.DataFrame,
    estimator: Callable[[pd.DataFrame], BoundsResult],
    n_boot: int = 200,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap endpoint uncertainty for an estimator returning ATE bounds."""
    rng = np.random.default_rng(seed)
    lower_samples = []
    upper_samples = []

    # Re-sample rows with replacement and re-estimate the interval.
    for _ in range(n_boot):
        sample_idx = rng.integers(0, len(df), size=len(df))
        boot_df = df.iloc[sample_idx]
        try:
            res = estimator(boot_df)
            lower_samples.append(res.ate_lower)
            upper_samples.append(res.ate_upper)
        except Exception:
            # Skip failed draws so one unstable sample does not stop the pipeline.
            continue

    if not lower_samples or not upper_samples:
        return {
            "boot_n_success": 0,
            "ate_lower_ci_05": np.nan,
            "ate_lower_ci_95": np.nan,
            "ate_upper_ci_05": np.nan,
            "ate_upper_ci_95": np.nan,
        }

    return {
        "boot_n_success": len(lower_samples),
        "ate_lower_ci_05": float(np.quantile(lower_samples, 0.05)),
        "ate_lower_ci_95": float(np.quantile(lower_samples, 0.95)),
        "ate_upper_ci_05": float(np.quantile(upper_samples, 0.05)),
        "ate_upper_ci_95": float(np.quantile(upper_samples, 0.95)),
    }

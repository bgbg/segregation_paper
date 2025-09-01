"""Diagnostics for voter transition model fits.

This module provides posterior predictive checks, convergence diagnostics,
and model validation utilities.
"""

import logging
from typing import Any, Dict, Optional

import arviz as az
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_diagnostics(trace: az.InferenceData) -> Dict[str, Any]:
    """Compute convergence diagnostics for MCMC trace.

    Args:
        trace: ArviZ InferenceData object

    Returns:
        Dictionary with diagnostic statistics
    """
    diagnostics = {}

    # R-hat statistics
    rhat = az.rhat(trace)
    diagnostics["rhat_max"] = float(rhat.max().values)
    diagnostics["rhat_mean"] = float(rhat.mean().values)
    diagnostics["rhat_good"] = bool(diagnostics["rhat_max"] < 1.01)

    # Effective sample size
    ess = az.ess(trace)
    diagnostics["ess_min"] = float(ess.min().values)
    diagnostics["ess_mean"] = float(ess.mean().values)
    diagnostics["ess_good"] = bool(diagnostics["ess_min"] > 400)

    # Monte Carlo standard error
    mcse = az.mcse(trace)
    diagnostics["mcse_max"] = float(mcse.max().values)
    diagnostics["mcse_mean"] = float(mcse.mean().values)

    # Energy diagnostics
    if hasattr(trace, "sample_stats"):
        energy = trace.sample_stats.energy
        diagnostics["energy_bfmi"] = float(az.bfmi(trace))
        diagnostics["energy_good"] = bool(diagnostics["energy_bfmi"] > 0.2)

    # Divergences
    if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
        n_divergent = trace.sample_stats.diverging.sum().values
        diagnostics["n_divergent"] = int(n_divergent)
        diagnostics["divergent_good"] = bool(n_divergent == 0)

    # Overall convergence assessment
    convergence_checks = [
        diagnostics.get("rhat_good", False),
        diagnostics.get("ess_good", False),
        diagnostics.get("energy_good", True),  # Default True if not available
        diagnostics.get("divergent_good", True),
    ]
    diagnostics["converged"] = all(convergence_checks)

    return diagnostics


def run_posterior_predictive_checks(
    model,
    trace: az.InferenceData,
    data: Dict[str, Dict[str, np.ndarray]],
    n_samples: int = 100,
) -> Dict[str, Any]:
    """Run posterior predictive checks to validate model fit.

    Args:
        model: PyMC model object
        trace: Posterior samples
        data: Original data tensors
        n_samples: Number of posterior samples to use

    Returns:
        Dictionary with PPC results
    """
    try:
        import pymc as pm

        with model:
            # Generate posterior predictive samples
            ppc = pm.sample_posterior_predictive(
                trace, predictions=True, extend_inferencedata=False, progressbar=False
            )

        ppc_results = {
            "generated": True,
            "n_samples": n_samples,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        # Compare observed vs predicted party totals
        country_data = data["country"]
        observed_totals = country_data["x2"].sum(axis=0)

        # Extract predicted totals from PPC
        if "x2_country_obs" in ppc.predictions:
            predicted = ppc.predictions["x2_country_obs"]
            predicted_totals = predicted.sum(axis=1).mean(
                axis=0
            )  # Mean across samples and stations

            ppc_results["country_totals"] = {
                "observed": observed_totals.tolist(),
                "predicted": predicted_totals.tolist(),
                "residuals": (observed_totals - predicted_totals).tolist(),
            }

        return ppc_results

    except Exception as e:
        logger.warning(f"Could not run posterior predictive checks: {e}")
        return {
            "generated": False,
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat(),
        }


def assess_stability(
    trace: az.InferenceData, param_name: str = "M_country", thin: int = 10
) -> Dict[str, float]:
    """Assess parameter stability across MCMC chains.

    Args:
        trace: Posterior samples
        param_name: Parameter to assess
        thin: Thinning factor for samples

    Returns:
        Dictionary with stability metrics
    """
    if param_name not in trace.posterior:
        return {"error": f"Parameter {param_name} not found"}

    # Extract thinned samples
    samples = trace.posterior[param_name][::thin]

    # Compute running means
    n_samples = samples.sizes["draw"]
    running_means = samples.cumsum("draw") / np.arange(1, n_samples + 1)

    # Stability metric: relative change in final vs middle estimates
    final_mean = running_means.isel(draw=-1)
    middle_mean = running_means.isel(draw=n_samples // 2)

    rel_change = np.abs(final_mean - middle_mean) / (np.abs(middle_mean) + 1e-8)
    max_rel_change = float(rel_change.max().values)

    return {
        "max_relative_change": max_rel_change,
        "stable": bool(max_rel_change < 0.05),  # 5% threshold
        "n_samples_used": int(n_samples),
    }

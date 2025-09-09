"""Utilities for temporal priors between consecutive transitions.

This module extracts compact posterior summaries suitable to be reused as
priors for the next transition, and provides helpers to persist/load them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import arviz as az
import numpy as np
import json


def _to_list(arr: np.ndarray) -> list:
    return np.asarray(arr).tolist()


def build_priors_from_trace(
    trace: az.InferenceData,
    data: Dict[str, Dict[str, np.ndarray]],
    *,
    center: str = "mean",
) -> Dict[str, Any]:
    """Build a compact priors dict from posterior samples.

    Args:
        trace: Posterior samples
        data: Data dict used to build the model (to extract city names/order)
        center: 'mean' or 'median' for posterior centers

    Returns:
        Dict with centers for Z_country, diag_bias, log_phi, and optional Z_city.
    """
    reducer = {
        "mean": lambda xr: xr.mean(dim=["chain", "draw"]),
        "median": lambda xr: xr.median(dim=["chain", "draw"]),
    }[center]

    priors: Dict[str, Any] = {"center": center}

    # Z_country
    if "Z_country" in trace.posterior:
        zc = reducer(trace.posterior["Z_country"]).values  # (K, K)
        priors["Z_country"] = _to_list(zc)

    # diag_bias
    if "diag_bias" in trace.posterior:
        db = reducer(trace.posterior["diag_bias"]).values.item()
        priors["diag_bias"] = float(db)

    # log_phi
    if "log_phi" in trace.posterior:
        lp = reducer(trace.posterior["log_phi"]).values.item()
        priors["log_phi"] = float(lp)

    # City-level Z if present
    cities: List[str] = [k for k in data.keys() if k != "country"]
    if cities and "Z_city" in trace.posterior:
        zc_city = reducer(trace.posterior["Z_city"]).values  # (n_cities, K, K)
        city_priors = {}
        for idx, city in enumerate(cities):
            city_priors[city] = _to_list(zc_city[idx])
        priors["Z_city"] = city_priors
        priors["cities_order"] = cities

    return priors


def save_priors(priors: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(priors, f, indent=2)


def load_priors(input_path: Path) -> Optional[Dict[str, Any]]:
    if not input_path.exists():
        return None
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)

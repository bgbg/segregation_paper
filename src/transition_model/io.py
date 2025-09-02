"""Input/output utilities for voter transition models.

This module handles loading and saving of ArviZ InferenceData objects,
CSV point estimates, and JSON fit summaries.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import arviz as az
import numpy as np
import pandas as pd


def save_inference_data(
    trace: az.InferenceData, output_path: Path, scope: str = "country"
) -> None:
    """Save posterior samples to NetCDF file.

    Args:
        trace: ArviZ InferenceData object
        output_path: Path for .nc output file
        scope: Scope of the data ('country' or city name)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract relevant variables based on scope
    if scope == "country":
        # Save country-level parameters
        posterior_vars = ["M_country", "kappa", "phi"]
    else:
        # For cities, save city-specific matrix and global parameters
        posterior_vars = ["M_cities", "M_country", "kappa", "phi"]

    # Create filtered trace with only relevant variables
    filtered_trace = trace.sel(
        var=[v for v in posterior_vars if v in trace.posterior.data_vars]
    )

    # Save to NetCDF
    filtered_trace.to_netcdf(output_path)


def load_inference_data(input_path: Path) -> az.InferenceData:
    """Load posterior samples from NetCDF file.

    Args:
        input_path: Path to .nc file

    Returns:
        ArviZ InferenceData object
    """
    return az.from_netcdf(input_path)


def save_point_estimates(
    trace: az.InferenceData,
    output_path: Path,
    scope: str = "country",
    estimator: str = "mean",
    credible_interval: float = 0.95,
    city_index: int = None,
) -> None:
    """Save point estimates of transition matrix to CSV.

    Args:
        trace: Posterior samples
        output_path: Path for CSV output
        scope: Scope ('country' or city name)
        estimator: Point estimator ('mean' or 'median')
        credible_interval: Width of credible intervals
        city_index: Index of city in the cities array (required for city scope)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract transition matrix
    if scope == "country":
        # For country-level, look for column variables
        col_vars = [
            var for var in trace.posterior.data_vars
            if var.startswith("M_country_col_")
        ]
        if not col_vars:
            raise ValueError("Country transition matrix variables not found in trace")
        
        # Stack the columns to form the matrix
        import xarray as xr
        col_data = []
        for col_var in sorted(col_vars):
            col_data.append(trace.posterior[col_var])
        matrix_samples = xr.concat(col_data, dim="matrix_col")
        
    else:
        # For cities, we need to extract city-specific matrices
        if city_index is None:
            raise ValueError(f"city_index is required for city scope: {scope}")
            
        # Get column variables for this city
        col_vars = [
            var for var in trace.posterior.data_vars
            if var.startswith(f"M_city_{city_index}_col_")
        ]
        
        if not col_vars:
            raise ValueError(f"City transition matrix variables not found for city index {city_index}")
        
        # Stack the columns to form the city matrix
        import xarray as xr
        col_data = []
        for col_var in sorted(col_vars):
            col_data.append(trace.posterior[col_var])
        matrix_samples = xr.concat(col_data, dim="matrix_col")

    # Compute point estimates
    if estimator == "mean":
        point_est = matrix_samples.mean(dim=["chain", "draw"])
    elif estimator == "median":
        point_est = matrix_samples.median(dim=["chain", "draw"])
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    # Compute credible intervals
    alpha = 1 - credible_interval
    lower = matrix_samples.quantile(alpha / 2, dim=["chain", "draw"])
    upper = matrix_samples.quantile(1 - alpha / 2, dim=["chain", "draw"])

    # Create DataFrame with estimates and intervals
    categories = ["Shas", "Agudat_Israel", "Other", "Abstained"]

    # Convert to DataFrame format
    results = []
    for i, from_cat in enumerate(categories):
        for j, to_cat in enumerate(categories):
            results.append(
                {
                    "from_category": from_cat,
                    "to_category": to_cat,
                    "estimate": float(point_est[i, j].values.flatten()[0]),
                    "lower_ci": float(lower[i, j].values.flatten()[0]),
                    "upper_ci": float(upper[i, j].values.flatten()[0]),
                }
            )

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)


def save_vote_movements(
    trace: az.InferenceData,
    output_path: Path,
    vote_totals: Dict[str, float],
    scope: str = "country",
    estimator: str = "mean",
    credible_interval: float = 0.95,
    min_votes: float = 5000.0,
    city_index: int = None,
) -> None:
    """Save vote movements (actual vote counts) from transition matrix to CSV.

    Args:
        trace: Posterior samples
        output_path: Path for CSV output
        vote_totals: Dictionary with total votes by category from election t
        scope: Scope ('country' or city name)
        estimator: Point estimator ('mean' or 'median')
        credible_interval: Width of credible intervals
        min_votes: Minimum vote count threshold (movements below this are set to 0)
        city_index: Index of city in the cities array (required for city scope)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract transition matrix
    if scope == "country":
        # For country-level, look for column variables
        col_vars = [
            var for var in trace.posterior.data_vars
            if var.startswith("M_country_col_")
        ]
        if not col_vars:
            raise ValueError("Country transition matrix variables not found in trace")
        
        # Stack the columns to form the matrix
        import xarray as xr
        col_data = []
        for col_var in sorted(col_vars):
            col_data.append(trace.posterior[col_var])
        matrix_samples = xr.concat(col_data, dim="matrix_col")
        
    else:
        # For cities, we need to extract city-specific matrices
        if city_index is None:
            raise ValueError(f"city_index is required for city scope: {scope}")
            
        # Get column variables for this city
        col_vars = [
            var for var in trace.posterior.data_vars
            if var.startswith(f"M_city_{city_index}_col_")
        ]
        
        if not col_vars:
            raise ValueError(f"City transition matrix variables not found for city index {city_index}")
        
        # Stack the columns to form the city matrix
        import xarray as xr
        col_data = []
        for col_var in sorted(col_vars):
            col_data.append(trace.posterior[col_var])
        matrix_samples = xr.concat(col_data, dim="matrix_col")

    # Compute point estimates
    if estimator == "mean":
        point_est = matrix_samples.mean(dim=["chain", "draw"])
    elif estimator == "median":
        point_est = matrix_samples.median(dim=["chain", "draw"])
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    # Compute credible intervals
    alpha = 1 - credible_interval
    lower = matrix_samples.quantile(alpha / 2, dim=["chain", "draw"])
    upper = matrix_samples.quantile(1 - alpha / 2, dim=["chain", "draw"])

    # Categories and their vote totals from election t
    categories = ["Shas", "Agudat_Israel", "Other", "Abstained"]
    
    # Convert to DataFrame format with vote movements
    results = []
    for i, from_cat in enumerate(categories):
        from_votes = vote_totals.get(from_cat, 0.0)
        for j, to_cat in enumerate(categories):
            # Calculate vote movements: probability * total_votes_from_category
            prob_point = float(point_est[i, j].values.flatten()[0])
            prob_lower = float(lower[i, j].values.flatten()[0])
            prob_upper = float(upper[i, j].values.flatten()[0])
            
            vote_movement = prob_point * from_votes
            vote_lower = prob_lower * from_votes
            vote_upper = prob_upper * from_votes
            
            # Apply minimum threshold filter
            if vote_movement < min_votes:
                vote_movement = 0.0
                vote_lower = 0.0
                vote_upper = 0.0
            
            results.append(
                {
                    "from_category": from_cat,
                    "to_category": to_cat,
                    "vote_count": vote_movement,
                    "lower_ci_votes": vote_lower,
                    "upper_ci_votes": vote_upper,
                }
            )

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)


def save_fit_summary(summary: Dict[str, Any], output_path: Path) -> None:
    """Save fit summary to JSON file.

    Args:
        summary: Summary dictionary
        output_path: Path for JSON output
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        else:
            return obj

    converted_summary = convert_numpy(summary)

    with open(output_path, "w") as f:
        json.dump(converted_summary, f, indent=2)


def load_fit_summary(input_path: Path) -> Dict[str, Any]:
    """Load fit summary from JSON file.

    Args:
        input_path: Path to JSON file

    Returns:
        Summary dictionary
    """
    with open(input_path, "r") as f:
        return json.load(f)


def load_point_estimates(input_path: Path) -> pd.DataFrame:
    """Load point estimates from CSV file.

    Args:
        input_path: Path to CSV file

    Returns:
        DataFrame with estimates and credible intervals
    """
    return pd.read_csv(input_path)

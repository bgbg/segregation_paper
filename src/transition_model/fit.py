"""Orchestration for fitting voter transition models.

This module provides high-level functions to coordinate the entire
model fitting pipeline for a single election transition.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pymc as pm

from .diagnostics import compute_diagnostics, run_posterior_predictive_checks
from .io import (
    save_fit_summary,
    save_inference_data,
    save_point_estimates,
    save_vote_movements,
)
from .preprocess import compute_categories, prepare_hierarchical_data
from .pymc_model import build_hierarchical_model, sample_model

logger = logging.getLogger(__name__)


def save_model_visualization(model: pm.Model, output_path: Path) -> None:
    """Save PyMC model structure as PNG visualization.

    Args:
        model: PyMC model object
        output_path: Path for PNG output file
    """
    try:
        # Generate model graph using PyMC's graphviz integration
        graph = pm.model_to_graphviz(model)

        # Save as PNG
        graph.render(str(output_path.with_suffix("")), format="png", cleanup=True)

        logger.info(f"Model visualization saved as {output_path}")

    except Exception as e:
        logger.warning(f"Failed to save model visualization: {e}")
        logger.info("Make sure graphviz is installed: pip install graphviz")
        logger.info(
            "And system graphviz: brew install graphviz (macOS) or apt-get install graphviz (Linux)"
        )


def load_election_data(
    election_t_path: Path, election_t1_path: Path, columns_mapping: Dict[str, str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess election data for adjacent elections.

    Args:
        election_t_path: Path to election t CSV file
        election_t1_path: Path to election t+1 CSV file
        columns_mapping: Column name mappings

    Returns:
        Tuple of (df_t, df_t1) preprocessed DataFrames
    """
    df_t = pd.read_parquet(election_t_path)
    df_t1 = pd.read_parquet(election_t1_path)

    # Apply category computations
    df_t = compute_categories(df_t, columns_mapping)
    df_t1 = compute_categories(df_t1, columns_mapping)

    return df_t, df_t1


def fit_transition_pair(
    pair_tag: str,
    election_t_path: Path,
    election_t1_path: Path,
    output_dir: Path,
    target_cities: List[str],
    columns_mapping: Dict[str, str],
    model_params: Optional[Dict] = None,
    sampling_params: Optional[Dict] = None,
    config: Optional[Dict] = None,
    force: bool = False,
) -> Dict:
    """Fit transition model for a single election pair.

    Args:
        pair_tag: Transition identifier (e.g., 'kn20_21')
        election_t_path: Path to election t data
        election_t1_path: Path to election t+1 data
        output_dir: Directory for outputs
        target_cities: Cities to model separately
        columns_mapping: Column mappings for parties
        model_params: Model hyperparameters
        sampling_params: MCMC sampling parameters
        force: Whether to overwrite existing outputs

    Returns:
        Dictionary with fit summary information
    """
    # Set default parameters
    if model_params is None:
        model_params = {
            "diag_bias_mean": 3.0,
            "diag_bias_sigma": 0.5,
            "sigma_country": 1.0,
            "sigma_city": 0.5,
            "nu_scale": 5.0,
        }

    if sampling_params is None:
        sampling_params = {
            "draws": 1000,
            "tune": 1000,
            "chains": 4,
            "target_accept": 0.98,
            "random_seed": 42,
        }

    # Create output directory
    pair_output_dir = output_dir / pair_tag
    pair_output_dir.mkdir(exist_ok=True)

    # Check if outputs already exist
    country_trace_path = pair_output_dir / "country_trace.nc"
    if not force and country_trace_path.exists():
        logger.info(
            f"Outputs for {pair_tag} already exist, skipping (use --force to overwrite)"
        )
        return {"status": "skipped", "reason": "outputs_exist"}

    logger.info(f"Fitting transition model for {pair_tag}")

    # Load and preprocess data
    logger.info("Loading election data...")
    df_t, df_t1 = load_election_data(election_t_path, election_t1_path, columns_mapping)

    # Prepare hierarchical data
    logger.info("Preparing hierarchical data tensors...")
    data = prepare_hierarchical_data(df_t, df_t1, target_cities)

    # Build model
    logger.info("Building PyMC model...")
    model = build_hierarchical_model(data, **model_params)

    # Save model visualization
    logger.info("Saving model visualization...")
    model_viz_path = pair_output_dir / "model.png"
    save_model_visualization(model, model_viz_path)

    # Sample posterior
    logger.info("Sampling posterior...")
    trace = sample_model(model, **sampling_params)

    # Run diagnostics
    logger.info("Computing diagnostics...")
    diagnostics = compute_diagnostics(trace, config)

    # Posterior predictive checks
    logger.info("Running posterior predictive checks...")
    ppc_results = run_posterior_predictive_checks(model, trace, data)

    # Save outputs
    logger.info("Saving outputs...")

    # Save posterior traces
    save_inference_data(trace, pair_output_dir / "country_trace.nc", scope="country")

    for city in target_cities:
        if city in data:
            city_trace_path = (
                pair_output_dir / f'city_{city.lower().replace(" ", "_")}_trace.nc'
            )
            save_inference_data(trace, city_trace_path, scope=city)

    # Save point estimates (transition probabilities)
    save_point_estimates(trace, pair_output_dir / "country_map.csv", scope="country")

    # Save country vote movements (actual vote counts)
    country_vote_totals = data["country"]["vote_totals"]
    save_vote_movements(
        trace,
        pair_output_dir / "country_movements.csv",
        country_vote_totals,
        scope="country",
    )

    # Save city-specific results (both probabilities and movements)
    city_index = 0  # Cities are indexed in order they appear in target_cities
    for city in target_cities:
        if city in data:
            city_slug = city.lower().replace(" ", "_")

            # Save city transition probabilities
            city_map_path = pair_output_dir / f"city_{city_slug}_map.csv"
            save_point_estimates(
                trace, city_map_path, scope=city, city_index=city_index
            )

            # Save city vote movements
            city_movements_path = pair_output_dir / f"city_{city_slug}_movements.csv"
            city_vote_totals = data[city]["vote_totals"]
            save_vote_movements(
                trace,
                city_movements_path,
                city_vote_totals,
                scope=city,
                city_index=city_index,
            )

            city_index += 1  # Increment index for each city with data

    # Save fit summary
    fit_summary = {
        "pair_tag": pair_tag,
        "model_params": model_params,
        "sampling_params": sampling_params,
        "diagnostics": diagnostics,
        "ppc_results": ppc_results,
        "n_stations_country": len(data["country"]["x1"]),
        "cities_modeled": list(data.keys())[1:],  # Exclude 'country'
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    logs_dir = output_dir.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    save_fit_summary(fit_summary, logs_dir / f"fit_summary_{pair_tag}.json")

    logger.info(f"Completed fitting for {pair_tag}")
    return fit_summary

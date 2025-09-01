"""Orchestration for fitting voter transition models.

This module provides high-level functions to coordinate the entire
model fitting pipeline for a single election transition.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .preprocess import compute_categories, prepare_hierarchical_data
from .pymc_model import build_hierarchical_model, sample_model
from .io import save_inference_data, save_point_estimates, save_fit_summary
from .diagnostics import compute_diagnostics, run_posterior_predictive_checks

logger = logging.getLogger(__name__)


def load_election_data(
    election_t_path: Path,
    election_t1_path: Path,
    columns_mapping: Dict[str, str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess election data for adjacent elections.
    
    Args:
        election_t_path: Path to election t CSV file
        election_t1_path: Path to election t+1 CSV file
        columns_mapping: Column name mappings
        
    Returns:
        Tuple of (df_t, df_t1) preprocessed DataFrames
    """
    df_t = pd.read_csv(election_t_path)
    df_t1 = pd.read_csv(election_t1_path)
    
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
    force: bool = False
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
            'alpha_diag': 10.0,
            'kappa_prior_scale': 100.0
        }
    
    if sampling_params is None:
        sampling_params = {
            'draws': 2000,
            'tune': 2000,
            'chains': 4,
            'target_accept': 0.9,
            'random_seed': 42
        }
    
    # Create output directory
    pair_output_dir = output_dir / pair_tag
    pair_output_dir.mkdir(exist_ok=True)
    
    # Check if outputs already exist
    country_trace_path = pair_output_dir / 'country_trace.nc'
    if not force and country_trace_path.exists():
        logger.info(f"Outputs for {pair_tag} already exist, skipping (use --force to overwrite)")
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
    
    # Sample posterior
    logger.info("Sampling posterior...")
    trace = sample_model(model, **sampling_params)
    
    # Run diagnostics
    logger.info("Computing diagnostics...")
    diagnostics = compute_diagnostics(trace)
    
    # Posterior predictive checks
    logger.info("Running posterior predictive checks...")
    ppc_results = run_posterior_predictive_checks(model, trace, data)
    
    # Save outputs
    logger.info("Saving outputs...")
    
    # Save posterior traces
    save_inference_data(trace, pair_output_dir / 'country_trace.nc', scope='country')
    
    for city in target_cities:
        if city in data:
            city_trace_path = pair_output_dir / f'city_{city.lower().replace(" ", "_")}_trace.nc'
            save_inference_data(trace, city_trace_path, scope=city)
    
    # Save point estimates  
    save_point_estimates(trace, pair_output_dir / 'country_map.csv', scope='country')
    
    for city in target_cities:
        if city in data:
            city_map_path = pair_output_dir / f'city_{city.lower().replace(" ", "_")}_map.csv'
            save_point_estimates(trace, city_map_path, scope=city)
    
    # Save fit summary
    fit_summary = {
        'pair_tag': pair_tag,
        'model_params': model_params,
        'sampling_params': sampling_params,
        'diagnostics': diagnostics,
        'ppc_results': ppc_results,
        'n_stations_country': len(data['country']['x1']),
        'cities_modeled': list(data.keys())[1:],  # Exclude 'country'
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    logs_dir = output_dir.parent / 'logs'
    logs_dir.mkdir(exist_ok=True)
    save_fit_summary(fit_summary, logs_dir / f'fit_summary_{pair_tag}.json')
    
    logger.info(f"Completed fitting for {pair_tag}")
    return fit_summary
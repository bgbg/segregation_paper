"""Command-line interface for voter transition analysis.

This module provides CLI commands for the complete voter transition analysis pipeline:
- Data harmonization
- Tensor preparation
- Model fitting
- Results summarization
"""

import logging

# Import our modules
import sys
from pathlib import Path

import defopt

# Add project root to path to enable imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data_harmonizer import harmonize_all_elections, load_config
from src.transition_model.fit import fit_transition_pair
from src.transition_model.preprocess import prepare_transition_data


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def prep_data(
    *, config_path: str = "data/config.yaml", force: bool = False, verbose: bool = False
):
    """Harmonize raw election data and prepare for analysis.

    This command processes all configured elections by:
    - Loading raw CSV files
    - Applying column mappings
    - Computing voter categories with corrected abstained calculation
    - Applying homogenic filtering (if enabled)
    - Filtering for target cities
    - Saving harmonized parquet files

    Args:
        config_path: Path to YAML configuration file
        force: Force reprocessing even if outputs exist
        verbose: Enable verbose logging
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    logger.info("Starting data harmonization pipeline...")

    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    logger.info(f"Target elections: {config['data']['elections']}")
    logger.info(f"Target cities: {config['cities']['target_cities']}")

    homogenic_enabled = config["data"]["homogenic_filtering"]["enabled"]
    homogenic_threshold = config["data"]["homogenic_filtering"]["threshold"]
    logger.info(
        f"Homogenic filtering: {'enabled' if homogenic_enabled else 'disabled'} "
        f"(threshold: {homogenic_threshold})"
    )

    # Run harmonization
    harmonized_data = harmonize_all_elections(config_path, force)

    if harmonized_data:
        logger.info(f"✓ Successfully harmonized {len(harmonized_data)} elections")

        # Print summary statistics
        total_stations = sum(len(df) for df in harmonized_data.values())
        logger.info(f"Total harmonized stations: {total_stations}")

        for election_num, df in harmonized_data.items():
            shas_votes = df["A_shas"].sum()
            agudat_votes = df["B_agudat"].sum()
            other_votes = df["Other"].sum()
            abstained = df["Abstained"].sum()
            total_votes = shas_votes + agudat_votes + other_votes + abstained

            logger.info(
                f"Election {election_num}: {len(df)} stations, "
                f"{total_votes:.0f} total votes "
                f"(Shas: {shas_votes/total_votes*100:.1f}%, "
                f"Agudat: {agudat_votes/total_votes*100:.1f}%, "
                f"Other: {other_votes/total_votes*100:.1f}%, "
                f"Abstained: {abstained/total_votes*100:.1f}%)"
            )
    else:
        logger.warning("⚠ No election data was harmonized")
        logger.warning("Check if raw data files exist in data/raw/ directory")
        return 1

    logger.info("✓ Data harmonization completed successfully")
    return 0


def build_tensors(
    *,
    pair: str,
    config_path: str = "data/config.yaml",
    verbose: bool = False,
):
    """Build station-level tensors for a specific election pair.

    This command prepares hierarchical data tensors for PyMC modeling by:
    - Loading harmonized election data
    - Aligning stations between elections
    - Building country-wide and city-specific tensors
    - Validating data consistency

    Args:
        pair: Election pair identifier (e.g., 'kn20_21')
        config_path: Path to YAML configuration file
        verbose: Enable verbose logging
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    logger.info(f"Building tensors for transition pair: {pair}")

    # Validate pair format
    if not pair.startswith("kn") or "_" not in pair:
        logger.error(f"Invalid pair format: {pair}. Expected format: 'knXX_YY'")
        return 1

    # Load configuration
    config = load_config(config_path)

    # Check if this is a valid transition pair
    valid_pairs = config["data"]["transition_pairs"]
    if pair not in valid_pairs:
        logger.warning(
            f"Pair {pair} not in configured transition pairs: {valid_pairs}"
        )
        logger.info("Proceeding anyway...")

    # Prepare transition data
    logger.info("Loading harmonized data and building tensors...")
    data = prepare_transition_data(pair, config_path)

    # Print data summary
    logger.info(f"✓ Tensors built for {pair}")
    logger.info(f"Country stations: {len(data['country']['x1'])}")

    cities_with_data = [k for k in data.keys() if k != "country"]
    logger.info(f"Cities with data: {len(cities_with_data)}")

    for city in cities_with_data:
        city_stations = len(data[city]["x1"])
        logger.info(f"  {city}: {city_stations} stations")

    # Validate tensor shapes
    categories = config["categories"]
    K = len(categories)

    for scope, scope_data in data.items():
        x1_shape = scope_data["x1"].shape
        x2_shape = scope_data["x2"].shape

        if x1_shape[1] != K or x2_shape[1] != K:
            logger.error(
                f"Invalid tensor shapes for {scope}: "
                f"x1={x1_shape}, x2={x2_shape}, expected K={K}"
            )
            return 1

        logger.debug(f"{scope}: x1={x1_shape}, x2={x2_shape}")

    logger.info("✓ Tensor validation passed")

    logger.info("✓ Tensor building completed successfully")
    return 0


def fit_model(
    *,
    pair: str,
    config_path: str = "data/config.yaml",
    force: bool = False,
    verbose: bool = False,
):
    """Fit PyMC transition model for a specific election pair.

    This command runs the full Bayesian modeling pipeline:
    - Loading prepared tensor data
    - Building hierarchical PyMC model
    - Running MCMC sampling
    - Computing diagnostics
    - Saving results (traces, point estimates, summaries)

    Args:
        pair: Election pair identifier (e.g., 'kn20_21')
        config_path: Path to YAML configuration file
        force: Force refitting even if outputs exist
        verbose: Enable verbose logging
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    logger.info(f"Fitting transition model for pair: {pair}")

    # Load configuration
    config = load_config(config_path)

    # Extract model parameters from config
    model_params = {
        "alpha_diag": config["model"]["alpha_diag"],
        "kappa_prior_scale": config["model"]["kappa_prior_scale"],
    }

    sampling_params = config["model"]["sampling"]
    target_cities = config["cities"]["target_cities"]

    logger.info(f"Model parameters: {model_params}")
    logger.info(f"Sampling parameters: {sampling_params}")

    # Define paths
    output_dir = Path(config["paths"]["processed_data"]) / "transitions"

    # Parse election numbers for file paths
    elections_part = pair[2:]  # Remove 'kn'
    election_t, election_t1 = elections_part.split("_")

    # Create dummy file paths (fit_transition_pair expects actual files)
    # In a real implementation, these would be the harmonized data files
    election_t_path = (
        Path(config["paths"]["interim_data"]) / f"harmonized_{election_t}.parquet"
    )
    election_t1_path = (
        Path(config["paths"]["interim_data"]) / f"harmonized_{election_t1}.parquet"
    )

    # Create column mapping (empty for harmonized data)
    columns_mapping = {}

    # Run fitting pipeline
    logger.info("Starting model fitting...")

    fit_summary = fit_transition_pair(
        pair_tag=pair,
        election_t_path=election_t_path,
        election_t1_path=election_t1_path,
        output_dir=output_dir,
        target_cities=target_cities,
        columns_mapping=columns_mapping,
        model_params=model_params,
        sampling_params=sampling_params,
        force=force,
    )

    if fit_summary.get("status") == "skipped":
        logger.info(f"✓ Model fitting skipped - outputs already exist for {pair}")
        logger.info("Use --force to overwrite existing outputs")
    else:
        logger.info(f"✓ Model fitting completed for {pair}")

        # Print fit summary
        if "diagnostics" in fit_summary:
            diag = fit_summary["diagnostics"]
            logger.info(
                f"Convergence: max_rhat={diag.get('max_rhat', 'N/A'):.3f}, "
                f"min_ess={diag.get('min_ess', 'N/A')}"
            )

        n_stations = fit_summary.get("n_stations_country", 0)
        cities_modeled = fit_summary.get("cities_modeled", [])
        logger.info(
            f"Stations: {n_stations} country-wide, {len(cities_modeled)} cities"
        )

    logger.info("✓ Model fitting completed successfully")
    return 0


def summarize(*, pair: str, verbose: bool = False):
    """Generate summary of results for a specific election pair.

    This command creates summary reports and visualizations:
    - Loading saved model results
    - Computing transition matrix summaries
    - Generating diagnostic plots
    - Creating summary tables

    Args:
        pair: Election pair identifier (e.g., 'kn20_21')
        verbose: Enable verbose logging
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    logger.info(f"Generating summary for transition pair: {pair}")
    logger.warning("Summary generation not yet implemented")
    logger.info("This will include:")
    logger.info("- Transition matrix tables with confidence intervals")
    logger.info("- Diagnostic plots and convergence checks")
    logger.info("- Flow diagrams and visualizations")

    return 0


if __name__ == "__main__":
    defopt.run([prep_data, build_tensors, fit_model, summarize])

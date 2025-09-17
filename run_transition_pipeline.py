#!/usr/bin/env python3
"""Complete voter transition analysis pipeline.

This script orchestrates the entire voter transition model pipeline:
1. Data harmonization (raw CSV → harmonized parquet)
2. Model fitting for all transition pairs
3. Results visualization and summarization

Usage:
    python run_transition_pipeline.py [--config-path CONFIG_PATH] [--force] [--verbose] [--skip-visualization]
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import defopt
import shutil
from colorama import Fore, Back, Style, init

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

# Import pipeline modules
from src.data_harmonizer import harmonize_all_elections, load_config
from src.transition_model.fit import fit_transition_pair
from src.transition_model.priors import load_priors
from src.transition_model.preprocess import (
    prepare_hierarchical_data,
    load_harmonized_data,
)
from src.transition_model.io import save_fit_summary, load_fit_summary
from src.transition_model.diagnostics import compute_diagnostics
from src.transition_model.pymc_model import build_hierarchical_model, sample_model


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    # Color mapping
    COLORS = {
        "DEBUG": Fore.LIGHTBLACK_EX,  # Light gray
        "INFO": Fore.WHITE,  # Gray/white
        "WARNING": Fore.YELLOW,  # Yellow
        "ERROR": Fore.RED,  # Red
        "CRITICAL": Fore.RED + Style.BRIGHT,  # Bright red
    }

    def format(self, record):
        # Get the original formatted message
        log_message = super().format(record)

        # Add color based on log level
        color = self.COLORS.get(record.levelname, "")
        if color:
            log_message = f"{color}{log_message}{Style.RESET_ALL}"

        return log_message


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration with colors."""
    # Initialize colorama for cross-platform color support
    init(autoreset=True)

    level = logging.DEBUG if verbose else logging.INFO

    # Create a custom handler with colored formatter
    handler = logging.StreamHandler(sys.stdout)
    formatter = ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # Remove any existing handlers
    root_logger.addHandler(handler)

    return logging.getLogger(__name__)


def step1_data_harmonization(
    config: Dict, force: bool, logger: logging.Logger
) -> Dict[int, any]:
    """Step 1: Harmonize raw election data.

    Args:
        config: Configuration dictionary
        force: Force reprocessing even if outputs exist
        logger: Logger instance

    Returns:
        Dictionary mapping election numbers to harmonized DataFrames
    """
    logger.info("=" * 60)
    logger.info("STEP 1: DATA HARMONIZATION")
    logger.info("=" * 60)

    # Load column mappings for harmonization
    from src.data_harmonizer import load_column_mappings, load_city_mappings

    elections = config["data"]["elections"]
    target_cities = config["cities"]["target_cities"]

    logger.info(f"Processing elections: {elections}")
    logger.info(f"Target cities: {target_cities}")

    homogenic_enabled = config["data"]["homogenic_filtering"]["enabled"]
    homogenic_threshold = config["data"]["homogenic_filtering"]["threshold"]
    logger.info(
        f"Homogenic filtering: {'enabled' if homogenic_enabled else 'disabled'} (threshold: {homogenic_threshold})"
    )

    # Run harmonization for all elections
    harmonized_data = harmonize_all_elections("data/config.yaml", force)

    logger.info(f"✓ Harmonization complete: processed {len(harmonized_data)} elections")
    for election_num, df in harmonized_data.items():
        logger.info(f"  Election {election_num}: {len(df)} stations")

    return harmonized_data


def step2_model_fitting(
    config: Dict, force: bool, logger: logging.Logger
) -> Dict[str, Dict]:
    """Step 2: Fit transition models for all election pairs.

    Args:
        config: Configuration dictionary
        force: Force refitting even if outputs exist
        logger: Logger instance

    Returns:
        Dictionary mapping pair tags to fit summaries
    """
    logger.info("=" * 60)
    logger.info("STEP 2: MODEL FITTING")
    logger.info("=" * 60)

    transition_pairs = config["data"]["transition_pairs"]
    target_cities = config["cities"]["target_cities"]

    # Load column mappings for model fitting
    from src.data_harmonizer import load_column_mappings

    columns_mapping = load_column_mappings(config["paths"]["columns_mapping"])

    # Extract model parameters from config
    model_params = config["model"]["logistic_normal"]
    sampling_params = config["model"]["sampling"]
    temporal_cfg = config["model"].get("temporal_priors", {"enabled": False})
    innovation = temporal_cfg.get("innovation", {})

    logger.info(f"Fitting models for {len(transition_pairs)} transition pairs")
    logger.info(f"Model parameters: {model_params}")
    logger.info(f"Sampling parameters: {sampling_params}")

    fit_summaries = {}

    for idx, pair_tag in enumerate(transition_pairs):
        logger.info(f"\n--- Fitting model for pair: {pair_tag} ---")

        # Parse election numbers from pair tag
        elections_part = pair_tag[2:]  # Remove 'kn' prefix
        election_t, election_t1 = elections_part.split("_")
        election_t, election_t1 = int(election_t), int(election_t1)

        # Define paths
        election_t_path = (
            Path(config["paths"]["interim_data"]) / f"harmonized_{election_t}.parquet"
        )
        election_t1_path = (
            Path(config["paths"]["interim_data"]) / f"harmonized_{election_t1}.parquet"
        )
        output_dir = Path(config["paths"]["transitions_dir"])

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if outputs already exist
        pair_output_dir = output_dir / pair_tag
        country_trace_path = pair_output_dir / "country_trace.nc"

        if not force and country_trace_path.exists():
            logger.info(
                f"Outputs for {pair_tag} already exist, skipping (use --force to overwrite)"
            )
            continue

        # Load priors from previous pair if enabled and not the first pair
        priors_payload = None
        if temporal_cfg.get("enabled", False) and idx > 0:
            prev_pair = transition_pairs[idx - 1]
            prev_priors_path = output_dir / prev_pair / "priors.json"
            priors_payload = load_priors(prev_priors_path)
            if priors_payload is None:
                logger.warning(
                    f"Temporal priors enabled but missing for {prev_pair} (Knesset {election_t}-{election_t1}); using default priors."
                )

        # Fit the model
        fit_summary = fit_transition_pair(
            pair_tag=pair_tag,
            election_t_path=election_t_path,
            election_t1_path=election_t1_path,
            output_dir=output_dir,
            target_cities=target_cities,
            columns_mapping=columns_mapping,
            model_params=model_params,
            sampling_params=sampling_params,
            config=config,
            force=force,
            priors=priors_payload,
            innovation=innovation,
        )

        fit_summaries[pair_tag] = fit_summary

        logger.info(f"✓ Successfully fitted model for {pair_tag}")

    logger.info(f"\n✓ Model fitting complete: {len(fit_summaries)} pairs processed")

    return fit_summaries


def step3_results_summary(
    config: Dict, fit_summaries: Dict[str, Dict], logger: logging.Logger
) -> None:
    """Step 3: Generate results summary and diagnostics.

    Args:
        config: Configuration dictionary
        fit_summaries: Results from model fitting
        logger: Logger instance
    """
    logger.info("=" * 60)
    logger.info("STEP 3: RESULTS SUMMARY")
    logger.info("=" * 60)

    transition_pairs = config["data"]["transition_pairs"]

    logger.info(f"Generating summary for {len(transition_pairs)} transition pairs")

    # Summary statistics
    total_pairs = len(transition_pairs)
    successful_pairs = len(fit_summaries)

    logger.info(f"Successfully processed: {successful_pairs}/{total_pairs} pairs")

    # Check convergence for each pair
    converged_pairs = 0
    for pair_tag in transition_pairs:
        logs_dir = Path(config["paths"]["logs_dir"])
        summary_path = logs_dir / f"fit_summary_{pair_tag}.json"

        if summary_path.exists():
            try:
                summary = load_fit_summary(summary_path)
                diagnostics = summary.get("diagnostics", {})

                if diagnostics.get("converged", False):
                    converged_pairs += 1
                    logger.info(
                        f"✓ {pair_tag}: Converged (R-hat: {diagnostics.get('rhat_max', 'N/A'):.3f})"
                    )
                else:
                    logger.warning(
                        f"⚠ {pair_tag}: Not converged (R-hat: {diagnostics.get('rhat_max', 'N/A'):.3f})"
                    )

            except Exception as e:
                logger.error(f"✗ {pair_tag}: Error loading diagnostics: {e}")

    logger.info(f"Converged models: {converged_pairs}/{successful_pairs}")

    # Output file summary
    logger.info("\nOutput files generated:")
    logger.info(f"  - Transition matrices: {config['paths']['transitions_dir']}")
    logger.info(f"  - Model diagnostics: {config['paths']['logs_dir']}")
    logger.info(f"  - Visualization: Run 'python visualize_transitions.py' for plots")


def step4_visualization(config: Dict, logger: logging.Logger) -> None:
    """Step 4: Generate visualizations (optional).

    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("=" * 60)
    logger.info("STEP 4: VISUALIZATION")
    logger.info("=" * 60)

    try:
        # Import visualization modules
        from src.visualize_transitions import main as visualize_main

        logger.info("Generating transition matrix visualizations...")

        # Run visualization (this will process all available transition pairs)
        visualize_main()

        logger.info("✓ Visualization complete")

    except ImportError as e:
        logger.warning(f"Visualization module not available: {e}")
    except Exception as e:
        logger.error(f"Visualization failed: {e}")


def run_pipeline(
    *,
    config_path: str = "data/config.yaml",
    force: bool = True,
    verbose: bool = False,
    skip_visualization: bool = False,
    clear_all: bool = False,
) -> int:
    """Run complete voter transition analysis pipeline.

    This script orchestrates the entire voter transition model pipeline:
    1. Data harmonization (raw CSV → harmonized parquet)
    2. Model fitting for all transition pairs
    3. Results visualization and summarization

    Args:
        config_path: Path to configuration YAML file
        force: Force reprocessing even if outputs exist
        verbose: Enable verbose logging
        skip_visualization: Skip visualization step
        clear_all: Clear all interim files before running

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Setup logging
    logger = setup_logging(verbose)

    logger.info("Starting voter transition analysis pipeline...")
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Force reprocessing: {force}")

    # Load configuration
    config = load_config(config_path)

    # Optionally clear interim files before anything else
    if clear_all:
        logger.info("Clearing interim files as requested (--clear-all)...")
        interim_dir = Path(config["paths"]["interim_data"]).resolve()
        if interim_dir.exists():
            try:
                shutil.rmtree(interim_dir)
                logger.info(f"Removed {interim_dir}")
            except Exception as e:
                logger.error(f"Failed clearing interim directory {interim_dir}: {e}")
                raise
        interim_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Interim directory recreated.")

    # Step 1: Data harmonization
    harmonized_data = step1_data_harmonization(config, force, logger)

    if not harmonized_data:
        logger.error("No harmonized data available. Check raw data files.")
        return 1

    # Step 2: Model fitting
    fit_summaries = step2_model_fitting(config, force, logger)

    if not fit_summaries:
        logger.error("No models were successfully fitted.")
        return 1

    # Step 3: Results summary
    step3_results_summary(config, fit_summaries, logger)

    # Step 4: Visualization (optional)
    if not skip_visualization:
        step4_visualization(config, logger)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("✓ All steps completed successfully!")
    logger.info(f"✓ Processed {len(harmonized_data)} elections")
    logger.info(f"✓ Fitted {len(fit_summaries)} transition models")
    logger.info("\nNext steps:")
    logger.info("  - Review diagnostics in data/processed/logs/")
    logger.info("  - Examine transition matrices in data/processed/transitions/")
    logger.info("  - Run 'python visualize_transitions.py' for plots")

    return 0


if __name__ == "__main__":
    defopt.run(run_pipeline)

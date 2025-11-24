#!/usr/bin/env python3
"""Pipeline for fitting independent city voter transition models.

This script fits transition models where each city is estimated independently
using the countrywide posterior as a prior. This allows maximum inter-city
variability compared to hierarchical pooling approaches.
"""

import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import defopt
import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_harmonizer import harmonize_election_data, load_column_mappings
from src.transition_model.fit import fit_transition_pair_independent
from src.transition_model.priors import load_priors
from src.visualize_transitions import (
    plot_all_cities_aggregate_deviation_subplots,
    plot_transition_matrix_over_elections,
)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for better readability."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging with colored output.

    Args:
        verbose: Enable verbose logging

    Returns:
        Logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Colored formatter
    formatter = ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return logging.getLogger(__name__)


def step1_data_harmonization(
    config: Dict, force: bool, logger: logging.Logger
) -> Optional[Dict]:
    """Step 1: Data harmonization.

    Args:
        config: Configuration dictionary
        force: Force reprocessing
        logger: Logger instance

    Returns:
        Dictionary of harmonized election data
    """
    logger.info("=" * 60)
    logger.info("STEP 1: DATA HARMONIZATION")
    logger.info("=" * 60)

    elections = config["data"]["elections"]
    target_cities = config["cities"]["target_cities"]

    logger.info(f"Processing elections: {elections}")
    logger.info(f"Target cities: {target_cities}")

    # Homogenic filtering settings
    homogenic_enabled = config["data"]["homogenic_filtering"]["enabled"]
    homogenic_threshold = config["data"]["homogenic_filtering"]["threshold"]
    logger.info(
        f"Homogenic filtering: {'enabled' if homogenic_enabled else 'disabled'} "
        f"(threshold: {homogenic_threshold})"
    )

    # Harmonize data for all elections
    harmonized_data = {}

    for election_num in elections:
        try:
            df_harmonized = harmonize_election_data(
                election_num,
                config,
                force=force,
            )
            harmonized_data[election_num] = df_harmonized
        except Exception as e:
            logger.error(f"Failed to harmonize election {election_num}: {e}")
            continue

    logger.info(f"✓ Harmonization complete: processed {len(harmonized_data)} elections")
    for election_num, df in harmonized_data.items():
        logger.info(f"  Election {election_num}: {len(df)} stations")

    return harmonized_data


def step2_model_fitting(
    config: Dict, force: bool, logger: logging.Logger
) -> List[Dict]:
    """Step 2: Fit independent city models for all transition pairs.

    Args:
        config: Configuration dictionary
        force: Force reprocessing
        logger: Logger instance

    Returns:
        List of fit summaries
    """
    logger.info("=" * 60)
    logger.info("STEP 2: INDEPENDENT MODEL FITTING")
    logger.info("=" * 60)

    transition_pairs = config["data"]["transition_pairs"]
    target_cities = config["cities"]["target_cities"]

    logger.info(f"Fitting models for {len(transition_pairs)} transition pairs")
    logger.info(f"Model type: INDEPENDENT (no hierarchical pooling)")
    logger.info(f"Model parameters: {config['model']['logistic_normal']}")
    logger.info(f"Sampling parameters: {config['model']['sampling']}")

    # Setup paths
    interim_dir = Path(config["paths"]["interim_data"])
    output_dir = Path(config["paths"]["transitions_dir"])

    # Load columns mapping
    columns_mapping = load_column_mappings(config["paths"]["columns_mapping"])

    # Fit models for each transition pair
    fit_summaries = []
    priors = None  # Will be loaded from previous pair if temporal priors enabled

    for pair_tag in transition_pairs:
        logger.info(f"\n--- Fitting independent models for pair: {pair_tag} ---")

        # Parse election numbers from pair tag
        kn_nums = pair_tag.replace("kn", "").split("_")
        election_t = int(kn_nums[0])
        election_t1 = int(kn_nums[1])

        # Construct file paths
        election_t_path = interim_dir / f"harmonized_{election_t}.parquet"
        election_t1_path = interim_dir / f"harmonized_{election_t1}.parquet"

        # Check if data files exist
        if not election_t_path.exists() or not election_t1_path.exists():
            logger.warning(
                f"Data files missing for {pair_tag}, skipping:\n"
                f"  {election_t_path}\n"
                f"  {election_t1_path}"
            )
            continue

        # Load priors if temporal priors are enabled
        if config["model"]["temporal_priors"]["enabled"] and priors is not None:
            logger.info("Using temporal priors from previous transition")
            innovation = config["model"]["temporal_priors"]["innovation"]
        else:
            innovation = None

        # Fit model
        try:
            fit_summary = fit_transition_pair_independent(
                pair_tag=pair_tag,
                election_t_path=election_t_path,
                election_t1_path=election_t1_path,
                output_dir=output_dir,
                target_cities=target_cities,
                columns_mapping=columns_mapping,
                model_params=config["model"]["logistic_normal"],
                sampling_params=config["model"]["sampling"],
                config=config,
                force=force,
                priors=priors,
                innovation=innovation,
            )

            fit_summaries.append(fit_summary)

            # Load priors for next iteration if temporal priors enabled
            if config["model"]["temporal_priors"]["enabled"]:
                priors_path = output_dir / pair_tag / "priors.json"
                if priors_path.exists():
                    priors = load_priors(priors_path)
                    logger.info(f"Loaded priors for next transition from {priors_path}")

        except Exception as e:
            logger.error(f"Failed to fit model for {pair_tag}: {e}")
            import traceback

            traceback.print_exc()
            continue

    return fit_summaries


def step3_results_summary(
    config: Dict, fit_summaries: List[Dict], logger: logging.Logger
) -> None:
    """Step 3: Summarize fitting results.

    Args:
        config: Configuration dictionary
        fit_summaries: List of fit summaries
        logger: Logger instance
    """
    logger.info("=" * 60)
    logger.info("STEP 3: RESULTS SUMMARY")
    logger.info("=" * 60)

    for summary in fit_summaries:
        pair_tag = summary["pair_tag"]
        logger.info(f"\n{pair_tag}:")

        # Country diagnostics
        if "country_diagnostics" in summary:
            country_diag = summary["country_diagnostics"]
            logger.info(f"  Country model:")
            logger.info(f"    Status: {country_diag.get('status', 'unknown')}")
            if "issues" in country_diag and country_diag["issues"]:
                logger.warning(f"    Issues: {country_diag['issues']}")

        # City diagnostics
        if "city_diagnostics" in summary:
            for city, city_diag in summary["city_diagnostics"].items():
                logger.info(f"  City model ({city}):")
                logger.info(f"    Status: {city_diag.get('status', 'unknown')}")
                if "issues" in city_diag and city_diag["issues"]:
                    logger.warning(f"    Issues: {city_diag['issues']}")


def step4_visualization(config: Dict, force: bool, logger: logging.Logger) -> None:
    """Step 4: Generate visualizations.

    Args:
        config: Configuration dictionary
        force: Force regeneration
        logger: Logger instance
    """
    logger.info("=" * 60)
    logger.info("STEP 4: VISUALIZATION")
    logger.info("=" * 60)

    transitions_dir = Path(config["paths"]["transitions_dir"])
    target_cities = config["cities"]["target_cities"]

    # Generate plots
    logger.info("Generating transition matrix plots...")
    # Note: Visualization code may need adaptation for independent model output
    logger.info(
        "Note: Visualization generation may require custom code for independent models"
    )


def run_pipeline(
    *,
    config_path: str = "data/config_independent.yaml",
    force: bool = False,
    verbose: bool = False,
    skip_visualization: bool = False,
    clear_all: bool = False,
) -> int:
    """Run complete independent city voter transition analysis pipeline.

    This script orchestrates the entire pipeline for independent city models:
    1. Data harmonization (raw CSV → harmonized parquet)
    2. Independent model fitting for all transition pairs
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

    logger.info("Starting independent city voter transition analysis pipeline...")
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Force reprocessing: {force}")

    # Load configuration
    config = load_config(config_path)

    # Optionally clear interim files
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

    # Step 2: Independent model fitting
    fit_summaries = step2_model_fitting(config, force, logger)

    if not fit_summaries:
        logger.error("No models were successfully fitted.")
        return 1

    # Step 3: Results summary
    step3_results_summary(config, fit_summaries, logger)

    # Step 4: Visualization (optional)
    if not skip_visualization:
        step4_visualization(config, force, logger)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("✓ All steps completed successfully!")
    logger.info(f"✓ Processed {len(harmonized_data)} elections")
    logger.info(f"✓ Fitted {len(fit_summaries)} independent transition models")
    logger.info("\nNext steps:")
    logger.info("  - Review diagnostics: data/processed/logs_independent/")
    logger.info("  - Examine transition matrices: data/processed/transitions_independent/")
    logger.info(
        "  - Compare with hierarchical results: data/processed/transitions/ and transitions_relaxed/"
    )

    return 0


if __name__ == "__main__":
    defopt.run(run_pipeline)

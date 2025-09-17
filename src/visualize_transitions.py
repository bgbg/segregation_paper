#!/usr/bin/env python3
"""Visualization module for voter transition matrices over time.

This module provides functions to visualize transition probabilities across
election pairs, supporting both individual time series and 4x4 matrix grids.
Includes support for partial time ranges with fixed x-axis domains for smooth
presentation transitions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import defopt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

try:
    from .plot_utils import heb
except ImportError:
    # Handle case when running as script
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent))
    from plot_utils import heb


# Color scheme for categories
COLOR_SHAS = "#FFD700"  # Gold, as in Shas logo
COLOR_AGUDA = "#1B3161"  # Blue, as in Agudat Israel logo
COLOR_OTHER = "C3"  # Default matplotlib color
COLOR_ABSTAINED = "C4"  # Default matplotlib color
COLOR_OFF_DIAGONAL = "k"  # Black for off-diagonal elements

# Category order for consistent plotting
CATEGORY_ORDER = ["Shas", "Agudat_Israel", "Other", "Abstained"]

# Color mapping for diagonal elements
DIAGONAL_COLORS = {
    "Shas": COLOR_SHAS,
    "Agudat_Israel": COLOR_AGUDA,
    "Other": COLOR_OTHER,
    "Abstained": COLOR_ABSTAINED,
}


def collect_transition_estimates(
    transitions_dir: str = "data/processed/transitions",
    pairs: Optional[List[str]] = None,
    level: str = "country",
    city: Optional[str] = None,
) -> pd.DataFrame:
    """Load and concatenate transition CSV files into a single DataFrame.

    Args:
        transitions_dir: Directory containing transition pair subdirectories
        pairs: List of pair tags to load; if None, load all available
        level: Either "country" or "city" to specify which files to load
        city: City name for city-level data (required if level="city")

    Returns:
        DataFrame with columns: pair_tag, kn_location, from_category, to_category,
                               estimate, lower_ci, upper_ci
    """
    transitions_path = Path(transitions_dir)

    if level == "city" and city is None:
        raise ValueError("City name must be provided when level='city'")

    if pairs is None:
        # Find all available pairs
        pairs = [
            p.name
            for p in transitions_path.iterdir()
            if p.is_dir() and p.name.startswith("kn")
        ]
        pairs.sort()  # Ensure consistent ordering

    dfs = []

    for pair_tag in pairs:
        pair_dir = transitions_path / pair_tag

        # Determine which file to load based on level
        if level == "country":
            map_file = pair_dir / "country_map.csv"
            file_desc = "country_map.csv"
        else:  # level == "city"
            # Convert city name to slug format (lowercase, spaces to underscores, keep apostrophes)
            city_slug = city.lower().replace(" ", "_")
            map_file = pair_dir / f"city_{city_slug}_map.csv"
            file_desc = f"city_{city_slug}_map.csv"

        if not map_file.exists():
            warnings.warn(f"Missing {file_desc} for {pair_tag}, skipping")
            continue

        try:
            df = pd.read_csv(map_file)
            df["pair_tag"] = pair_tag
            df["level"] = level
            if level == "city":
                df["city"] = city

            # Extract kn_location for sorting (e.g., kn19_20 -> 19.5, between 19 and 20)
            kn_start = int(pair_tag.split("_")[0][2:])
            kn_end = int(pair_tag.split("_")[1])
            df["kn_location"] = (kn_start + kn_end) / 2.0

            dfs.append(df)

        except Exception as e:
            warnings.warn(f"Error loading {map_file}: {e}")
            continue

    if not dfs:
        raise ValueError(f"No valid transition data found in {transitions_dir}")

    # Concatenate all data
    df_all = pd.concat(dfs, ignore_index=True)

    # Ensure required columns exist
    required_cols = [
        "pair_tag",
        "kn_location",
        "from_category",
        "to_category",
        "estimate",
        "lower_ci",
        "upper_ci",
    ]
    missing_cols = [col for col in required_cols if col not in df_all.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df_all


def get_transition_string(from_category: str, to_category: str) -> str:
    """Generate Hebrew transition string for plot titles."""
    # Mapping to Hebrew names
    heb_names = {
        "Shas": "ש״ס",
        "Agudat_Israel": "אגודת ישראל",
        "Other": "אחרים",
        "Abstained": "נמנעו",
    }

    from_heb = heb_names.get(from_category, from_category)
    to_heb = heb_names.get(to_category, to_category)

    return heb(f"{from_heb} ← {to_heb}")


def set_xlabel(ax: plt.Axes, label: str = None) -> None:
    """Set Hebrew x-axis label with proper alignment."""
    if label is None:
        label = heb("כנסת מספר")
    ax.set_xlabel(label, rotation=0, ha="right", va="top", x=1)


def plot_transition_time_series(
    df_all: pd.DataFrame,
    from_category: str,
    to_category: str,
    *,
    pairs_to_show: Optional[List[str]] = None,
    color: Optional[str] = None,
    show_ci: bool = False,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a single transition probability time series over elections.

    Args:
        df_all: DataFrame with transition data
        from_category: Source category for transition
        to_category: Target category for transition
        pairs_to_show: Subset of pairs to show; others masked with NaN
        color: Line color; if None, uses defaults
        show_ci: Whether to show credible interval band
        ax: Matplotlib axes; if None, creates new figure

    Returns:
        The matplotlib axes with the plot
    """
    # Filter data for this specific transition
    df_sel = df_all[
        (df_all.from_category == from_category) & (df_all.to_category == to_category)
    ].copy()

    if df_sel.empty:
        raise ValueError(
            f"No data found for transition {from_category} -> {to_category}"
        )

    # Sort by kn_location for proper time ordering
    df_sel = df_sel.sort_values("kn_location").reset_index(drop=True)

    # Convert to percentages
    df_sel["estimate"] *= 100
    df_sel["lower_ci"] *= 100
    df_sel["upper_ci"] *= 100

    # Apply pairs_to_show masking by setting values to NaN
    if pairs_to_show is not None:
        mask = ~df_sel["pair_tag"].isin(pairs_to_show)
        df_sel.loc[mask, ["estimate", "lower_ci", "upper_ci"]] = np.nan

    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))

    # Determine color
    if color is None:
        # Use diagonal colors if this is a diagonal transition
        if from_category == to_category:
            color = DIAGONAL_COLORS.get(from_category, "C0")
        else:
            color = COLOR_OFF_DIAGONAL

    # Plot line and points
    ax.plot(df_sel["kn_location"], df_sel["estimate"], "-o", color=color)

    # Add credible interval if requested
    if show_ci:
        ax.fill_between(
            df_sel["kn_location"],
            df_sel["lower_ci"],
            df_sel["upper_ci"],
            color=color,
            alpha=0.2,
            zorder=-1,
        )

    # Set y-axis formatting
    ax.set_ylim(-2, 102)

    # Smart tick selection
    valid_estimates = df_sel["estimate"].dropna()
    if not valid_estimates.empty:
        tks = [0, 100] + [
            round(valid_estimates.min(), 0),
            round(valid_estimates.max(), 0),
        ]
        tks = list(sorted({int(t) for t in tks}))
        ax.set_yticks(tks)
        ax.set_yticklabels([f"{t}%" for t in tks])

    # Set x-axis to span full domain with Knesset numbers as ticks
    # Data points are at x.5 (between Knessets), but we want integer ticks
    x_min_data = df_sel["kn_location"].min()
    x_max_data = df_sel["kn_location"].max()

    # Convert back to Knesset numbers for ticks
    kn_min = int(np.floor(x_min_data))
    kn_max = int(np.ceil(x_max_data))

    # Set ticks at integer Knesset numbers
    ax.set_xticks(np.arange(kn_min, kn_max + 1))

    # Set x-axis limits to show some padding around the data
    ax.set_xlim(kn_min - 0.5, kn_max + 0.5)

    # Set title
    title = get_transition_string(from_category, to_category)
    ax.set_title(title)

    # Set Hebrew x-label and clean up spines
    set_xlabel(ax)
    sns.despine(ax=ax)

    return ax


def plot_transition_matrix_over_elections(
    df_all: pd.DataFrame,
    *,
    pairs_to_show: Optional[List[str]] = None,
    show_ci: bool = False,
    figsize: Tuple[float, float] = (12, 10),
    suptitle: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot 4x4 grid of transition time series.

    Args:
        df_all: DataFrame with transition data
        pairs_to_show: Subset of pairs to show; others masked
        show_ci: Whether to show credible intervals
        figsize: Figure size tuple
        suptitle: Super title for the figure
        save_path: Path to save figure; if None, doesn't save

    Returns:
        Tuple of (figure, axes_array)
    """
    # Create 4x4 subplot grid
    fig, axes = plt.subplots(4, 4, figsize=figsize, sharex=True)

    # Plot each transition
    for i, to_cat in enumerate(CATEGORY_ORDER):
        for j, from_cat in enumerate(CATEGORY_ORDER):
            ax = axes[i, j]

            # Determine color
            if from_cat == to_cat:
                color = DIAGONAL_COLORS[from_cat]
            else:
                color = COLOR_OFF_DIAGONAL

            try:
                plot_transition_time_series(
                    df_all=df_all,
                    from_category=from_cat,
                    to_category=to_cat,
                    pairs_to_show=pairs_to_show,
                    color=color,
                    show_ci=show_ci,
                    ax=ax,
                )
            except ValueError as e:
                # Handle missing transitions gracefully
                warnings.warn(f"Skipping {from_cat}->{to_cat}: {e}")
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

    # Set super title if provided
    if suptitle:
        fig.suptitle(suptitle)

    # Adjust layout
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    return fig, axes


def parse_pairs_to_show(pairs: List[str], spec: str) -> List[str]:
    """Parse pairs_to_show specification into a validated subset.

    Args:
        pairs: Full list of available pairs in order
        spec: Either comma-separated list or start:end range

    Returns:
        Validated subset of pairs preserving order
    """
    if ":" in spec:
        # Handle start:end range
        start, end = spec.split(":", 1)
        try:
            start_idx = pairs.index(start)
            end_idx = pairs.index(end)
            if start_idx > end_idx:
                raise ValueError(f"Start pair {start} comes after end pair {end}")
            return pairs[start_idx : end_idx + 1]
        except ValueError as e:
            if "not in list" in str(e):
                raise ValueError(f"Pair not found in available pairs: {e}")
            raise
    else:
        # Handle comma-separated list
        requested = [p.strip() for p in spec.split(",")]
        # Validate all pairs exist
        missing = [p for p in requested if p not in pairs]
        if missing:
            raise ValueError(f"Pairs not found: {missing}")
        # Return in original order
        return [p for p in pairs if p in requested]


def plot_all_transitions(
    config_path: str = "data/config.yaml",
    pairs_to_show: Optional[str] = None,
    show_ci: bool = True,
    output_dir: str = "data/processed/transitions/plots",
    figsize: str = "12,10",
) -> None:
    """Generate all transition matrix visualizations (country + cities).

    Args:
        config_path: Path to configuration YAML file
        pairs_to_show: Pairs to show (comma-separated or start:end range)
        show_ci: Show credible intervals
        output_dir: Directory to save plots
        figsize: Figure size as "width,height"
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    transition_pairs = config["data"]["transition_pairs"]
    transitions_dir = config["paths"]["transitions_dir"]
    target_cities = config["cities"]["target_cities"]

    print(f"Loading transition data from {transitions_dir}")
    print(f"Available pairs: {transition_pairs}")
    print(f"Target cities: {target_cities}")

    # Parse pairs_to_show if provided
    pairs_subset = None
    if pairs_to_show:
        pairs_subset = parse_pairs_to_show(transition_pairs, pairs_to_show)
        print(f"Showing subset: {pairs_subset}")

    # Parse figure size
    try:
        width, height = map(float, figsize.split(","))
        figsize_tuple = (width, height)
    except ValueError:
        print(f"Invalid figsize '{figsize}', using default (12, 10)")
        figsize_tuple = (12, 10)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Generate country-wide plot
    print("\n" + "=" * 60)
    print("GENERATING COUNTRY-WIDE TRANSITIONS")
    print("=" * 60)

    df_country = collect_transition_estimates(
        transitions_dir, transition_pairs, level="country"
    )
    print(
        f"Loaded country data: {df_country['pair_tag'].nunique()} pairs, {len(df_country)} transitions"
    )

    country_save_path = output_path / "country_transition_matrix_over_elections.png"

    fig, axes = plot_transition_matrix_over_elections(
        df_all=df_country,
        pairs_to_show=pairs_subset,
        show_ci=show_ci,
        figsize=figsize_tuple,
        suptitle="Country-wide Voter Transition Matrices Over Elections",
        save_path=str(country_save_path),
    )
    plt.close(fig)  # Close to free memory

    # 2. Generate city-level plots
    print("\n" + "=" * 60)
    print("GENERATING CITY-LEVEL TRANSITIONS")
    print("=" * 60)

    for city in target_cities:
        print(f"\nProcessing city: {city}")

        try:
            df_city = collect_transition_estimates(
                transitions_dir, transition_pairs, level="city", city=city
            )
            print(
                f"Loaded {city} data: {df_city['pair_tag'].nunique()} pairs, {len(df_city)} transitions"
            )

            # Create safe filename (keep apostrophes for data loading, remove for file saving)
            city_slug_file = city.lower().replace(" ", "_").replace("'", "")
            city_save_path = (
                output_path
                / f"city_{city_slug_file}_transition_matrix_over_elections.png"
            )

            fig, axes = plot_transition_matrix_over_elections(
                df_all=df_city,
                pairs_to_show=pairs_subset,
                show_ci=show_ci,
                figsize=figsize_tuple,
                suptitle=f"{city.title()} - Voter Transition Matrices Over Elections",
                save_path=str(city_save_path),
            )
            plt.close(fig)  # Close to free memory

        except Exception as e:
            print(f"⚠ Error processing {city}: {e}")
            continue

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print("✓ Generated plots:")
    print(f"  - Country-wide: {country_save_path}")
    for city in target_cities:
        city_slug_file = city.lower().replace(" ", "_").replace("'", "")
        city_path = (
            output_path / f"city_{city_slug_file}_transition_matrix_over_elections.png"
        )
        if city_path.exists():
            print(f"  - {city.title()}: {city_path}")

    if pairs_subset:
        print(f"✓ Displayed {len(pairs_subset)} of {len(transition_pairs)} pairs")
    else:
        print(f"✓ Displayed all {len(transition_pairs)} pairs")


def main(
    *,
    config_path: str = "data/config.yaml",
    pairs_to_show: Optional[str] = None,
    show_ci: bool = True,
    output_dir: str = "data/processed/transitions/plots",
    figsize: str = "12,10",
    plot_all: bool = False,
) -> None:
    """Generate transition matrix visualization plots.

    Args:
        config_path: Path to configuration YAML file
        pairs_to_show: Pairs to show (comma-separated or start:end range)
        show_ci: Show credible intervals
        output_dir: Directory to save plots
        figsize: Figure size as "width,height"
        plot_all: Generate plots for both country and all cities
    """
    # If plot_all is requested, use the comprehensive function
    if plot_all:
        plot_all_transitions(
            config_path=config_path,
            pairs_to_show=pairs_to_show,
            show_ci=show_ci,
            output_dir=output_dir,
            figsize=figsize,
        )
        return

    # Original single country plot behavior
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Get transition pairs from config
    transition_pairs = config["data"]["transition_pairs"]
    transitions_dir = config["paths"]["transitions_dir"]

    print(f"Loading transition data from {transitions_dir}")
    print(f"Available pairs: {transition_pairs}")

    # Load data
    df_all = collect_transition_estimates(
        transitions_dir, transition_pairs, level="country"
    )
    print(
        f"Loaded data for {df_all['pair_tag'].nunique()} pairs, "
        f"{len(df_all)} total transitions"
    )

    # Parse pairs_to_show if provided
    pairs_subset = None
    if pairs_to_show:
        pairs_subset = parse_pairs_to_show(transition_pairs, pairs_to_show)
        print(f"Showing subset: {pairs_subset}")

    # Parse figure size
    try:
        width, height = map(float, figsize.split(","))
        figsize_tuple = (width, height)
    except ValueError:
        print(f"Invalid figsize '{figsize}', using default (12, 10)")
        figsize_tuple = (12, 10)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate plot
    save_path = output_path / "transition_matrix_over_elections.png"

    print("Generating transition matrix plot...")
    fig, axes = plot_transition_matrix_over_elections(
        df_all=df_all,
        pairs_to_show=pairs_subset,
        show_ci=show_ci,
        figsize=figsize_tuple,
        suptitle="Country-wide Voter Transition Matrices Over Elections",
        save_path=str(save_path),
    )

    print("✓ Visualization complete!")
    print(f"✓ Saved plot to {save_path}")

    # Show summary statistics
    if pairs_subset:
        print(f"✓ Displayed {len(pairs_subset)} of {len(transition_pairs)} pairs")
    else:
        print(f"✓ Displayed all {len(transition_pairs)} pairs")


if __name__ == "__main__":
    defopt.run(main)

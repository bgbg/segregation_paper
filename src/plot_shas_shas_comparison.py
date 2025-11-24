"""Generate comparison plot for Shas-to-Shas transitions across all models."""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_shas_shas_transitions(
    transitions_dir: Path, pair_tags: List[str], cities: List[str]
) -> Dict[str, pd.DataFrame]:
    """Load Shas→Shas transition probabilities for all cities.

    Args:
        transitions_dir: Directory containing transition results
        pair_tags: List of transition pair identifiers
        cities: List of city names

    Returns:
        Dictionary mapping city names to DataFrames with Shas→Shas values per pair
    """
    results = {}

    # Add country
    all_locations = ["country"] + cities

    for location in all_locations:
        location_data = []

        for pair_tag in pair_tags:
            pair_dir = transitions_dir / pair_tag

            if location == "country":
                file_path = pair_dir / "country_map.csv"
            else:
                city_slug = location.lower().replace(" ", "_")
                file_path = pair_dir / f"city_{city_slug}_map.csv"

            if not file_path.exists():
                continue

            df = pd.read_csv(file_path)

            # Check format and extract Shas→Shas value
            if "from_category" in df.columns:
                # Long format
                shas_shas = df[
                    (df["from_category"] == "Shas") & (df["to_category"] == "Shas")
                ]["estimate"].values
                if len(shas_shas) > 0:
                    value = shas_shas[0]
                else:
                    continue
            else:
                # Matrix format (from independent model)
                df_matrix = df.set_index(df.columns[0])
                if "shas" in df_matrix.columns and "shas" in df_matrix.index:
                    value = df_matrix.loc["shas", "shas"]
                elif "Shas" in df_matrix.columns and "Shas" in df_matrix.index:
                    value = df_matrix.loc["Shas", "Shas"]
                else:
                    continue

            location_data.append({
                "pair": pair_tag,
                "shas_shas": float(value)
            })

        if location_data:
            results[location] = pd.DataFrame(location_data)

    return results


def plot_shas_shas_comparison(
    original_dir: Path,
    relaxed_dir: Path,
    independent_dir: Path,
    pair_tags: List[str],
    cities: List[str],
    output_path: Path,
) -> None:
    """Create comparison plot for Shas→Shas transitions across models.

    Args:
        original_dir: Directory with original model results
        relaxed_dir: Directory with relaxed model results
        independent_dir: Directory with independent model results
        pair_tags: List of transition pairs
        cities: List of cities
        output_path: Path to save plot
    """
    # Load data from all three models
    print("Loading Shas→Shas transitions from original model...")
    original = load_shas_shas_transitions(original_dir, pair_tags, cities)

    print("Loading Shas→Shas transitions from relaxed model...")
    relaxed = load_shas_shas_transitions(relaxed_dir, pair_tags, cities)

    print("Loading Shas→Shas transitions from independent model...")
    independent = load_shas_shas_transitions(independent_dir, pair_tags, cities)

    # Create plot
    locations = ["country"] + cities
    n_locations = len(locations)

    fig, axes = plt.subplots(n_locations, 1, figsize=(12, 3 * n_locations))

    if n_locations == 1:
        axes = [axes]

    colors = {
        "original": "#1f77b4",
        "relaxed": "#ff7f0e",
        "independent": "#2ca02c"
    }

    for idx, location in enumerate(locations):
        ax = axes[idx]

        # Plot each model if data exists
        if location in original and not original[location].empty:
            df = original[location]
            ax.plot(
                df["pair"], df["shas_shas"],
                marker="o", label="Original", color=colors["original"],
                linewidth=2, markersize=8
            )

        if location in relaxed and not relaxed[location].empty:
            df = relaxed[location]
            ax.plot(
                df["pair"], df["shas_shas"],
                marker="s", label="Relaxed", color=colors["relaxed"],
                linewidth=2, markersize=8
            )

        if location in independent and not independent[location].empty:
            df = independent[location]
            ax.plot(
                df["pair"], df["shas_shas"],
                marker="^", label="Independent", color=colors["independent"],
                linewidth=2, markersize=8
            )

        # Formatting
        ax.set_ylabel("Shas → Shas\nTransition Probability", fontsize=11)
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)

        # Title
        location_title = location.title() if location != "country" else "Country"
        ax.set_title(
            f"{location_title}: Shas Voter Loyalty Across Models",
            fontsize=12, fontweight="bold"
        )

        # Rotate x-axis labels
        ax.tick_params(axis="x", rotation=45)

    # Overall title
    fig.suptitle(
        "Shas → Shas Transition Comparison\nVoter Loyalty Across Three Modeling Approaches",
        fontsize=14, fontweight="bold", y=0.995
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved comparison plot to: {output_path}")
    plt.close()

    # Also print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS: Shas → Shas Transitions")
    print("=" * 60)

    for location in locations:
        print(f"\n{location.upper()}:")

        for model_name, data in [
            ("Original", original),
            ("Relaxed", relaxed),
            ("Independent", independent)
        ]:
            if location in data and not data[location].empty:
                values = data[location]["shas_shas"].values
                print(f"  {model_name:12s}: mean={np.mean(values):.4f}, "
                      f"std={np.std(values):.4f}, "
                      f"min={np.min(values):.4f}, "
                      f"max={np.max(values):.4f}")


def main():
    """Main function."""
    # Configuration
    base_dir = Path("data/processed")
    original_dir = base_dir / "transitions"
    relaxed_dir = base_dir / "transitions_relaxed"
    independent_dir = base_dir / "transitions_independent"

    output_dir = base_dir / "comparison"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "shas_shas_city_comparison.png"

    pair_tags = ["kn18_19", "kn19_20", "kn20_21", "kn21_22", "kn22_23", "kn23_24", "kn24_25"]
    cities = ["ashdod", "beit shemesh", "elad", "bnei brak", "jerusalem", "modi'in illit"]

    print("=" * 60)
    print("GENERATING SHAS→SHAS COMPARISON PLOT")
    print("=" * 60)

    plot_shas_shas_comparison(
        original_dir, relaxed_dir, independent_dir,
        pair_tags, cities, output_path
    )

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

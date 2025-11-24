"""Generate comparison plot for Shas-to-Shas transitions with cities grouped by model."""

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


def plot_shas_shas_by_model(
    original_dir: Path,
    relaxed_dir: Path,
    independent_dir: Path,
    pair_tags: List[str],
    cities: List[str],
    output_path: Path,
) -> None:
    """Create comparison plot with three subplots, one per model showing all cities.

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

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Color palette for cities
    city_colors = sns.color_palette("husl", len(cities))
    city_color_map = dict(zip(cities, city_colors))

    models = [
        ("Original", original, axes[0]),
        ("Relaxed", relaxed, axes[1]),
        ("Independent", independent, axes[2])
    ]

    for model_name, data, ax in models:
        # Plot country as thick gray line
        if "country" in data and not data["country"].empty:
            df = data["country"]
            ax.plot(
                df["pair"], df["shas_shas"],
                color="gray", linewidth=3, alpha=0.6,
                label="Country", linestyle="--", zorder=1
            )

        # Plot each city
        for city in cities:
            if city in data and not data[city].empty:
                df = data[city]
                ax.plot(
                    df["pair"], df["shas_shas"],
                    marker="o", label=city.title(),
                    color=city_color_map[city],
                    linewidth=2, markersize=6, alpha=0.8, zorder=2
                )

        # Formatting
        ax.set_ylabel("Shas → Shas Probability", fontsize=12)
        ax.set_xlabel("Election Pair", fontsize=12)
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{model_name} Model", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)

        # Legend
        ax.legend(loc="lower left", fontsize=9, ncol=1)

    # Overall title
    fig.suptitle(
        "Shas Voter Loyalty: Inter-City Variability by Model\n"
        "Shas → Shas Transition Probability Across Elections",
        fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved comparison plot to: {output_path}")
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 70)
    print("INTER-CITY VARIABILITY SUMMARY (Standard Deviation of Shas→Shas)")
    print("=" * 70)

    for model_name, data in [
        ("Original", original),
        ("Relaxed", relaxed),
        ("Independent", independent)
    ]:
        print(f"\n{model_name.upper()} MODEL:")

        for pair_tag in pair_tags:
            city_values = []
            for city in cities:
                if city in data and not data[city].empty:
                    df = data[city]
                    pair_data = df[df["pair"] == pair_tag]
                    if not pair_data.empty:
                        city_values.append(pair_data["shas_shas"].values[0])

            if len(city_values) >= 2:
                mean_val = np.mean(city_values)
                std_val = np.std(city_values)
                range_val = np.max(city_values) - np.min(city_values)
                print(f"  {pair_tag}: mean={mean_val:.4f}, std={std_val:.4f}, "
                      f"range={range_val:.4f}, n_cities={len(city_values)}")


def main():
    """Main function."""
    # Configuration
    base_dir = Path("data/processed")
    original_dir = base_dir / "transitions"
    relaxed_dir = base_dir / "transitions_relaxed"
    independent_dir = base_dir / "transitions_independent"

    output_dir = base_dir / "comparison"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "shas_shas_by_model.png"

    pair_tags = ["kn18_19", "kn19_20", "kn20_21", "kn21_22", "kn22_23", "kn23_24", "kn24_25"]
    cities = ["ashdod", "beit shemesh", "elad", "bnei brak", "jerusalem", "modi'in illit"]

    print("=" * 70)
    print("GENERATING SHAS→SHAS BY MODEL COMPARISON PLOT")
    print("=" * 70)

    plot_shas_shas_by_model(
        original_dir, relaxed_dir, independent_dir,
        pair_tags, cities, output_path
    )

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

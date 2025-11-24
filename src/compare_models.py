"""Compare different model approaches for voter transition analysis.

This module provides functions to compute metrics and generate visualizations
comparing the original hierarchical model, relaxed hierarchical model, and
independent city models.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_transition_results(
    transitions_dir: Path, pair_tags: List[str], cities: List[str]
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load transition matrix results for all pairs and cities.

    Args:
        transitions_dir: Directory containing transition results
        pair_tags: List of transition pair identifiers
        cities: List of city names

    Returns:
        Dictionary with structure:
        {
            'country': {pair_tag: transition_matrix_df},
            city_name: {pair_tag: transition_matrix_df}
        }
    """
    results = {"country": {}}

    for city in cities:
        results[city] = {}

    categories = ["Shas", "Agudat_Israel", "Other", "Abstained"]

    for pair_tag in pair_tags:
        pair_dir = transitions_dir / pair_tag

        # Load country matrix
        country_file = pair_dir / "country_map.csv"
        if country_file.exists():
            df_long = pd.read_csv(country_file)
            # Convert from long format to matrix format
            # Pivot: from_category as columns, to_category as rows
            df_matrix = df_long.pivot(
                index="to_category", columns="from_category", values="estimate"
            )
            # Reorder to ensure consistent ordering
            df_matrix = df_matrix.reindex(index=categories, columns=categories)
            results["country"][pair_tag] = df_matrix
        else:
            print(f"Warning: Missing {country_file}")

        # Load city matrices
        for city in cities:
            city_slug = city.lower().replace(" ", "_")
            city_file = pair_dir / f"city_{city_slug}_map.csv"
            if city_file.exists():
                df_long = pd.read_csv(city_file)
                # Convert from long format to matrix format
                if "from_category" in df_long.columns:
                    df_matrix = df_long.pivot(
                        index="to_category", columns="from_category", values="estimate"
                    )
                    # Reorder to ensure consistent ordering
                    df_matrix = df_matrix.reindex(index=categories, columns=categories)
                    results[city][pair_tag] = df_matrix
                else:
                    # Already in matrix format (from independent model)
                    df = pd.read_csv(city_file, index_col=0)
                    if "pair" in df.columns:
                        df = df.drop(columns=["pair"])
                    results[city][pair_tag] = df
            # else:
            #     print(f"Warning: Missing {city_file}")

    return results


def compute_mad_from_country(
    results: Dict[str, Dict[str, pd.DataFrame]], cities: List[str], pair_tags: List[str]
) -> pd.DataFrame:
    """Compute Mean Absolute Deviation (MAD) of cities from country average.

    Args:
        results: Results dictionary from load_transition_results
        cities: List of city names
        pair_tags: List of transition pair identifiers

    Returns:
        DataFrame with MAD values for each city and pair
    """
    mad_data = []

    for pair_tag in pair_tags:
        if pair_tag not in results["country"]:
            continue

        country_matrix = results["country"][pair_tag].values

        for city in cities:
            if pair_tag not in results[city]:
                continue

            city_matrix = results[city][pair_tag].values

            # Compute MAD (mean absolute deviation)
            mad = np.mean(np.abs(city_matrix - country_matrix))

            mad_data.append(
                {
                    "pair": pair_tag,
                    "city": city,
                    "mad": mad,
                }
            )

    return pd.DataFrame(mad_data)


def compute_variability_metrics(
    results: Dict[str, Dict[str, pd.DataFrame]], cities: List[str], pair_tags: List[str]
) -> pd.DataFrame:
    """Compute comprehensive variability metrics across cities.

    Args:
        results: Results dictionary from load_transition_results
        cities: List of city names
        pair_tags: List of transition pair identifiers

    Returns:
        DataFrame with metrics for each transition element and pair
    """
    metrics_data = []

    categories = ["Shas", "Agudat_Israel", "Other", "Abstained"]

    for pair_tag in pair_tags:
        # Check if we have data for this pair
        if pair_tag not in results["country"]:
            continue

        # For each transition element (from_cat -> to_cat)
        for from_cat in categories:
            for to_cat in categories:
                # Collect values across cities
                city_values = []

                for city in cities:
                    if pair_tag in results[city]:
                        matrix = results[city][pair_tag]
                        if from_cat in matrix.columns and to_cat in matrix.index:
                            value = matrix.loc[to_cat, from_cat]
                            city_values.append(value)

                if len(city_values) < 2:
                    continue

                # Compute metrics
                mean_val = np.mean(city_values)
                std_val = np.std(city_values)
                cv = std_val / mean_val if mean_val > 0 else 0  # Coefficient of variation
                range_val = np.max(city_values) - np.min(city_values)

                # Get country value
                country_matrix = results["country"][pair_tag]
                country_val = country_matrix.loc[to_cat, from_cat]

                # Mean absolute deviation from country
                mad_from_country = np.mean([abs(v - country_val) for v in city_values])

                metrics_data.append(
                    {
                        "pair": pair_tag,
                        "from": from_cat,
                        "to": to_cat,
                        "mean": mean_val,
                        "std": std_val,
                        "cv": cv,
                        "range": range_val,
                        "country": country_val,
                        "mad_from_country": mad_from_country,
                    }
                )

    return pd.DataFrame(metrics_data)


def compare_models(
    original_dir: Path,
    relaxed_dir: Path,
    independent_dir: Path,
    pair_tags: List[str],
    cities: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare three model approaches.

    Args:
        original_dir: Directory with original hierarchical model results
        relaxed_dir: Directory with relaxed hierarchical model results
        independent_dir: Directory with independent model results
        pair_tags: List of transition pairs
        cities: List of cities

    Returns:
        Tuple of (mad_comparison_df, metrics_comparison_df)
    """
    # Load results from all three models
    print("Loading original model results...")
    original_results = load_transition_results(original_dir, pair_tags, cities)

    print("Loading relaxed model results...")
    relaxed_results = load_transition_results(relaxed_dir, pair_tags, cities)

    print("Loading independent model results...")
    independent_results = load_transition_results(independent_dir, pair_tags, cities)

    # Compute MAD for each model
    print("Computing MAD metrics...")
    mad_original = compute_mad_from_country(original_results, cities, pair_tags)
    mad_original["model"] = "original"

    mad_relaxed = compute_mad_from_country(relaxed_results, cities, pair_tags)
    mad_relaxed["model"] = "relaxed"

    mad_independent = compute_mad_from_country(independent_results, cities, pair_tags)
    mad_independent["model"] = "independent"

    mad_comparison = pd.concat([mad_original, mad_relaxed, mad_independent])

    # Compute comprehensive variability metrics
    print("Computing variability metrics...")
    metrics_original = compute_variability_metrics(original_results, cities, pair_tags)
    metrics_original["model"] = "original"

    metrics_relaxed = compute_variability_metrics(relaxed_results, cities, pair_tags)
    metrics_relaxed["model"] = "relaxed"

    metrics_independent = compute_variability_metrics(independent_results, cities, pair_tags)
    metrics_independent["model"] = "independent"

    metrics_comparison = pd.concat([metrics_original, metrics_relaxed, metrics_independent])

    return mad_comparison, metrics_comparison


def plot_mad_comparison(
    mad_comparison: pd.DataFrame, output_path: Path, title: str = "Inter-City Variability: MAD from Country Average"
) -> None:
    """Plot Mean Absolute Deviation comparison across models.

    Args:
        mad_comparison: DataFrame with MAD values
        output_path: Path to save plot
        title: Plot title
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: MAD by city across models
    ax = axes[0]
    sns.boxplot(data=mad_comparison, x="city", y="mad", hue="model", ax=ax)
    ax.set_title("MAD from Country by City")
    ax.set_xlabel("City")
    ax.set_ylabel("Mean Absolute Deviation")
    ax.legend(title="Model")
    ax.tick_params(axis="x", rotation=45)

    # Plot 2: Overall MAD distribution by model
    ax = axes[1]
    sns.violinplot(data=mad_comparison, x="model", y="mad", ax=ax)
    ax.set_title("Overall MAD Distribution by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Absolute Deviation")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved MAD comparison plot to {output_path}")


def plot_metrics_comparison(
    metrics_comparison: pd.DataFrame, output_path: Path
) -> None:
    """Plot comprehensive variability metrics comparison.

    Args:
        metrics_comparison: DataFrame with variability metrics
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Average metrics by model
    avg_metrics = (
        metrics_comparison.groupby("model")[["std", "cv", "range", "mad_from_country"]]
        .mean()
        .reset_index()
    )

    # Plot 1: Standard deviation
    ax = axes[0, 0]
    sns.barplot(data=avg_metrics, x="model", y="std", ax=ax)
    ax.set_title("Average Standard Deviation Across Cities")
    ax.set_ylabel("Standard Deviation")

    # Plot 2: Coefficient of variation
    ax = axes[0, 1]
    sns.barplot(data=avg_metrics, x="model", y="cv", ax=ax)
    ax.set_title("Average Coefficient of Variation")
    ax.set_ylabel("CV")

    # Plot 3: Range
    ax = axes[1, 0]
    sns.barplot(data=avg_metrics, x="model", y="range", ax=ax)
    ax.set_title("Average Range Across Cities")
    ax.set_ylabel("Range")

    # Plot 4: MAD from country
    ax = axes[1, 1]
    sns.barplot(data=avg_metrics, x="model", y="mad_from_country", ax=ax)
    ax.set_title("Average MAD from Country")
    ax.set_ylabel("MAD from Country")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved metrics comparison plot to {output_path}")


def create_summary_report(
    mad_comparison: pd.DataFrame,
    metrics_comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create a markdown summary report comparing models.

    Args:
        mad_comparison: DataFrame with MAD values
        metrics_comparison: DataFrame with variability metrics
        output_path: Path to save report
    """
    # Compute summary statistics
    mad_summary = mad_comparison.groupby("model")["mad"].agg(
        ["mean", "median", "std", "min", "max"]
    )

    metrics_summary = metrics_comparison.groupby("model").agg(
        {
            "std": ["mean", "median"],
            "cv": ["mean", "median"],
            "range": ["mean", "median"],
            "mad_from_country": ["mean", "median"],
        }
    )

    # Create report
    report = []
    report.append("# Model Comparison Report: Voter Transition Analysis\n")
    report.append(
        "Comparison of three approaches to modeling voter transitions:\n"
    )
    report.append("1. **Original**: Hierarchical model with tight priors (baseline)\n")
    report.append(
        "2. **Relaxed**: Hierarchical model with increased variability parameters\n"
    )
    report.append(
        "3. **Independent**: Each city fitted independently using country posterior as prior\n\n"
    )

    report.append("## Mean Absolute Deviation (MAD) from Country Average\n\n")
    report.append(mad_summary.to_markdown())
    report.append("\n\n")

    report.append("## Variability Metrics Summary\n\n")
    report.append("### Standard Deviation (averaged across transitions)\n\n")
    report.append(metrics_summary["std"].to_markdown())
    report.append("\n\n")

    report.append("### Coefficient of Variation (averaged across transitions)\n\n")
    report.append(metrics_summary["cv"].to_markdown())
    report.append("\n\n")

    report.append("### Range (max - min across cities, averaged)\n\n")
    report.append(metrics_summary["range"].to_markdown())
    report.append("\n\n")

    report.append("### MAD from Country (averaged across transitions)\n\n")
    report.append(metrics_summary["mad_from_country"].to_markdown())
    report.append("\n\n")

    # Interpretation
    report.append("## Interpretation\n\n")

    # Compare models
    mad_means = mad_summary["mean"]
    original_mad = mad_means["original"]
    relaxed_mad = mad_means["relaxed"]
    independent_mad = mad_means["independent"]

    report.append(f"- **Original model**: Average MAD = {original_mad:.4f}\n")
    report.append(f"- **Relaxed model**: Average MAD = {relaxed_mad:.4f}\n")
    report.append(f"- **Independent model**: Average MAD = {independent_mad:.4f}\n\n")

    if relaxed_mad > original_mad:
        increase_pct = ((relaxed_mad - original_mad) / original_mad) * 100
        report.append(
            f"The **relaxed model** shows a {increase_pct:.1f}% increase in inter-city variability "
            f"compared to the original model.\n\n"
        )
    else:
        report.append(
            "The **relaxed model** did not significantly increase inter-city variability.\n\n"
        )

    if independent_mad > original_mad:
        increase_pct = ((independent_mad - original_mad) / original_mad) * 100
        report.append(
            f"The **independent model** shows a {increase_pct:.1f}% increase in inter-city variability "
            f"compared to the original model.\n\n"
        )

    # Recommendations
    report.append("## Recommendations\n\n")
    report.append(
        "Review the generated plots to visually compare the models. Consider:\n\n"
    )
    report.append(
        "1. **If inter-city variability is too low** (cities look too similar): "
        "Use the independent or relaxed model\n"
    )
    report.append(
        "2. **If results look unrealistic** (too much variability): "
        "The original model with hierarchical pooling may be more appropriate\n"
    )
    report.append(
        "3. **Model diagnostics**: Check convergence diagnostics (R-hat, ESS) for all models\n"
    )
    report.append(
        "4. **Substantive interpretation**: Do the city-specific patterns make sense "
        "given local political context?\n\n"
    )

    # Write report
    with open(output_path, "w") as f:
        f.writelines(report)

    print(f"Saved summary report to {output_path}")


def main():
    """Main comparison script."""
    # Configuration
    base_dir = Path("data/processed")
    original_dir = base_dir / "transitions"
    relaxed_dir = base_dir / "transitions_relaxed"
    independent_dir = base_dir / "transitions_independent"

    output_dir = base_dir / "comparison"
    output_dir.mkdir(exist_ok=True)

    pair_tags = ["kn18_19", "kn19_20", "kn20_21", "kn21_22", "kn22_23"]
    cities = ["ashdod", "beit shemesh", "elad", "bnei brak", "jerusalem", "modi'in illit"]

    print("=" * 60)
    print("MODEL COMPARISON ANALYSIS")
    print("=" * 60)

    # Compare models
    mad_comparison, metrics_comparison = compare_models(
        original_dir, relaxed_dir, independent_dir, pair_tags, cities
    )

    # Save raw data
    print("\nSaving comparison data...")
    mad_comparison.to_csv(output_dir / "mad_comparison.csv", index=False)
    metrics_comparison.to_csv(output_dir / "metrics_comparison.csv", index=False)

    # Generate plots
    print("\nGenerating plots...")
    plot_mad_comparison(mad_comparison, output_dir / "mad_comparison.png")
    plot_metrics_comparison(metrics_comparison, output_dir / "metrics_comparison.png")

    # Create summary report
    print("\nCreating summary report...")
    create_summary_report(
        mad_comparison, metrics_comparison, output_dir / "comparison_report.md"
    )

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - mad_comparison.csv")
    print("  - metrics_comparison.csv")
    print("  - mad_comparison.png")
    print("  - metrics_comparison.png")
    print("  - comparison_report.md")


if __name__ == "__main__":
    main()

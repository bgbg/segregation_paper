from __future__ import annotations

"""Unified reporting and visualization entrypoints.

- Builds Markdown report with formatted tables and diagnostic plots
- Reuses existing visualization functions
- Provides a single entrypoint to generate all outputs
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import arviz as az
import numpy as np
import pandas as pd

from src.transition_model.io import (
    load_fit_summary,
    load_point_estimates,
)
from src.visualize_transitions import (
    collect_transition_estimates,
    compute_city_deviations,
    compute_city_aggregate_deviation,
    plot_all_cities_aggregate_deviation_subplots,
    plot_transition_matrix_over_elections,
)


@dataclass
class ReportPaths:
    transitions_dir: Path
    plots_dir: Path
    reports_dir: Path
    logs_dir: Path


def _format_percent(x: float) -> str:
    return f"{x*100:.1f}%"


def _format_pp(x: float) -> str:
    return f"{x:.1f}pp"


def _ensure_dirs(paths: ReportPaths) -> None:
    paths.plots_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)


def _load_config(config_path: str) -> Dict:
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _transition_table_percent(df: pd.DataFrame) -> pd.DataFrame:
    dfp = df.copy()
    dfp["estimate"] = (dfp["estimate"] * 100).round(1)
    dfp["lower_ci"] = (dfp["lower_ci"] * 100).round(1)
    dfp["upper_ci"] = (dfp["upper_ci"] * 100).round(1)
    # Wide 4x4 with from_category as columns and to_category as rows
    table = (
        dfp.pivot_table(
            index=["pair_tag", "to_category"],
            columns=["from_category"],
            values="estimate",
            aggfunc="first",
        )
        .reindex(
            index=pd.MultiIndex.from_product(
                [
                    sorted(dfp["pair_tag"].unique()),
                    ["Shas", "Agudat_Israel", "Other", "Abstained"],
                ]
            )
        )
        .reindex(columns=["Shas", "Agudat_Israel", "Other", "Abstained"])
    )
    return table.reset_index()


def _mad_summary_tables(
    df_country: pd.DataFrame,
    transitions_dir: str,
    transition_pairs: List[str],
    target_cities: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict] = []
    per_pair_rows: List[Dict] = []
    for city in target_cities:
        df_city = collect_transition_estimates(
            transitions_dir=transitions_dir,
            pairs=transition_pairs,
            level="city",
            city=city,
        )
        df_dev = compute_city_deviations(df_country, df_city, city)
        df_agg = compute_city_aggregate_deviation(df_dev)
        df_agg = df_agg.sort_values("kn_location")
        # Per-pair table rows
        for _, r in df_agg.iterrows():
            per_pair_rows.append(
                {
                    "city": city,
                    "pair_tag": r["pair_tag"],
                    "mean_abs_deviation_pp": round(r["mean_abs_deviation"] * 100, 1),
                }
            )
        # Overall summary
        rows.append(
            {
                "city": city,
                "mean_mad_pp": round(df_agg["mean_abs_deviation"].mean() * 100, 1),
                "max_mad_pp": round(df_agg["mean_abs_deviation"].max() * 100, 1),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(per_pair_rows)


def _load_diagnostics_for_pair(logs_dir: Path, pair_tag: str) -> Optional[Dict]:
    summary_path = logs_dir / f"fit_summary_{pair_tag}.json"
    if not summary_path.exists():
        return None
    try:
        return load_fit_summary(summary_path)
    except Exception:
        return None


def _short_diagnostic_text(d: Dict) -> str:
    if not d:
        return "No diagnostics available"
    rhat = d.get("diagnostics", {}).get("rhat_max")
    ess = d.get("diagnostics", {}).get("ess_min")
    div = d.get("diagnostics", {}).get("divergences")
    bfmi = d.get("diagnostics", {}).get("bfmi_min")
    parts = []
    if rhat is not None:
        parts.append(f"R-hat max: {rhat:.3f}")
    if ess is not None:
        parts.append(f"ESS min: {int(ess)}")
    if div is not None:
        parts.append(f"Divergences: {int(div)}")
    if bfmi is not None:
        parts.append(f"BFMI min: {bfmi:.3f}")
    return ", ".join(parts)


def _interpret_diagnostics(df_diag: pd.DataFrame) -> str:
    """Generate interpretation text for sampling diagnostics."""
    lines = []
    lines.append("**Interpretation:**")
    lines.append("")

    # Count convergence issues by checking R-hat values
    good_rhat = 0
    total_pairs = 0
    for _, row in df_diag.iterrows():
        summary = row.get("summary", "")
        if "R-hat max:" in summary:
            total_pairs += 1
            # Extract R-hat value
            import re

            rhat_match = re.search(r"R-hat max: ([\d.]+)", summary)
            if rhat_match:
                rhat_val = float(rhat_match.group(1))
                if rhat_val <= 1.01:
                    good_rhat += 1

    if total_pairs > 0:
        lines.append(
            f"- **Convergence**: {good_rhat}/{total_pairs} pairs have R-hat ≤ 1.01 (good)"
        )

        if good_rhat < total_pairs:
            lines.append(
                "- **R-hat > 1.01**: Chains haven't mixed well; model needs more tuning"
            )
            lines.append(
                "- **Low ESS**: Effective sample size insufficient; increase draws"
            )
            lines.append(
                "- **Divergences > 0**: Problematic posterior geometry; check model"
            )
            lines.append(
                "- **Low BFMI < 0.3**: Poor energy transitions; review parameterization"
            )
        else:
            lines.append("- All models show excellent convergence (R-hat ≤ 1.01)")

    lines.append("")
    lines.append("**Thresholds**: R-hat ≤ 1.01, ESS > 400, Divergences = 0, BFMI > 0.3")

    return "\n".join(lines)


def _save_diagnostic_plots(trace_path: Path, out_dir: Path, prefix: str) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    try:
        idata = az.from_netcdf(trace_path)
    except Exception:
        return paths
    import matplotlib.pyplot as plt
    import warnings

    # Temporarily increase max subplots and suppress warnings
    original_max_subplots = plt.rcParams.get("figure.max_open_warning", 20)
    plt.rcParams["figure.max_open_warning"] = 100

    # Rank plots - limit to key variables to reduce complexity
    p1 = out_dir / f"{prefix}_rank.png"
    # Only plot key variables (country matrix and first few city parameters)
    key_vars = [
        v
        for v in idata.posterior.data_vars
        if v.startswith("M_country")
        or (v.startswith("M_city") and any(f"city_{i}_" in v for i in range(2)))
    ][:15]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if key_vars:
            az.plot_rank(idata, var_names=key_vars)
        else:
            az.plot_rank(idata)
    plt.tight_layout()
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(p1)

    # Energy plot
    p2 = out_dir / f"{prefix}_energy.png"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        az.plot_energy(idata)
    plt.tight_layout()
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(p2)

    # Autocorr plot - limit variables and max_lag
    p3 = out_dir / f"{prefix}_autocorr.png"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if key_vars:
            az.plot_autocorr(idata, var_names=key_vars[:10], max_lag=50)
        else:
            az.plot_autocorr(idata, max_lag=50)
    plt.tight_layout()
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(p3)

    # Restore original rcParams
    plt.rcParams["figure.max_open_warning"] = original_max_subplots
    return paths


def _df_to_markdown(df: pd.DataFrame) -> str:
    # Simple GitHub-flavored table without excessive precision
    buf = []
    headers = [str(c) for c in df.columns]
    buf.append("| " + " | ".join(headers) + " |")
    buf.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.1f}")
            else:
                vals.append(str(v))
        buf.append("| " + " | ".join(vals) + " |")
    return "\n".join(buf)


def _create_mad_table_lens(
    per_pair_mad: pd.DataFrame, transition_pairs: List[str], plots_dir: Path
) -> Path:
    """Create table lens visualization with all MAD bar plots as subplots."""
    import matplotlib.pyplot as plt

    if per_pair_mad.empty:
        return None

    # Get all cities from the data
    all_cities = per_pair_mad["city"].unique().tolist()
    
    # Get the last pair to determine sorting order
    last_pair = transition_pairs[-1]
    last_pair_data = per_pair_mad[per_pair_mad["pair_tag"] == last_pair].copy()

    if last_pair_data.empty:
        # Fallback to overall sorting if last pair has no data
        city_order = (
            per_pair_mad.groupby("city")["mean_abs_deviation_pp"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
    else:
        # Sort cities by MAD in the last pair (largest to smallest)
        last_pair_data = last_pair_data.sort_values(
            "mean_abs_deviation_pp", ascending=False
        )
        city_order = last_pair_data["city"].tolist()
        
        # Add any missing cities to the end
        for city in all_cities:
            if city not in city_order:
                city_order.append(city)

    # Create subplots - one column per transition pair
    n_pairs = len(transition_pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(n_pairs * 3, 8), sharey=True)

    # Handle single subplot case
    if n_pairs == 1:
        axes = [axes]

    for i, pair in enumerate(transition_pairs):
        ax = axes[i]

        # Filter data for this pair
        pair_data = per_pair_mad[per_pair_mad["pair_tag"] == pair].copy()

        if pair_data.empty:
            ax.text(
                0.5,
                0.5,
                f"No data for {pair}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(pair, fontsize=10)
            continue

        # Create a complete dataset with all cities (fill missing with 0 or NaN)
        complete_data = pd.DataFrame({"city": city_order})
        complete_data = complete_data.merge(pair_data, on="city", how="left")
        complete_data["mean_abs_deviation_pp"] = complete_data["mean_abs_deviation_pp"].fillna(0)
        
        cities = complete_data["city"].tolist()
        mad_values = complete_data["mean_abs_deviation_pp"].tolist()
        
        # Reverse the order for horizontal bars (highest at top)
        cities.reverse()
        mad_values.reverse()

        # Create horizontal bars
        bars = ax.barh(cities, mad_values, color="black", height=0.6)

        # Add value labels to the right of each bar
        for bar, value in zip(bars, mad_values):
            ax.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        # Set x-axis limits and ticks
        max_mad = max(mad_values) if mad_values else 10
        ax.set_xlim(0, 11)
        ax.set_xticks([0, max_mad])
        ax.set_xticklabels([0, f"{max_mad:.1f}"], fontsize=9)

        # Set title (keep original Knesset pair notation)
        ax.set_title(pair, fontsize=10, pad=5)

        # Clean up appearance
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=9)

        # Only show x-axis label on bottom and y-axis labels on leftmost subplot
        ax.set_xlabel("MAD (pp)", fontsize=9)
        if i > 0:  # Hide y-axis labels for all but leftmost subplot
            ax.set_yticklabels([])

    plt.tight_layout()

    # Save plot
    plot_path = plots_dir / "mad_table_lens.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path


def generate_all_outputs(
    *,
    config_path: str = "data/config.yaml",
    pairs_to_show: Optional[List[str]] = None,
    save_dir: str = "data/processed",
) -> Path:
    """Generate all plots and a Markdown report.

    Returns path to the generated markdown file.
    """
    import os

    cfg = _load_config(config_path)
    paths = ReportPaths(
        transitions_dir=Path(cfg["paths"]["transitions_dir"]).resolve(),
        plots_dir=(Path(save_dir) / "reports" / "plots").resolve(),
        reports_dir=(Path(save_dir) / "reports").resolve(),
        logs_dir=Path(cfg["paths"]["logs_dir"]).resolve(),
    )
    _ensure_dirs(paths)

    transition_pairs = cfg["data"]["transition_pairs"]
    target_cities = cfg["cities"]["target_cities"]

    # 1) Country matrices table and figure
    df_country = collect_transition_estimates(
        transitions_dir=str(paths.transitions_dir),
        pairs=transition_pairs,
        level="country",
    )
    country_table = _transition_table_percent(df_country)

    country_plot_path = paths.plots_dir / "country_transition_matrix_over_elections.png"
    plot_transition_matrix_over_elections(
        df_all=df_country,
        pairs_to_show=pairs_to_show,
        show_ci=True,
        figsize=(12, 10),
        suptitle="Country-wide Voter Transition Matrices Over Elections",
        save_path=str(country_plot_path),
    )

    # 2) City matrices tables (per city)
    city_tables: Dict[str, pd.DataFrame] = {}
    city_plot_paths: Dict[str, Path] = {}
    for city in target_cities:
        try:
            df_city = collect_transition_estimates(
                transitions_dir=str(paths.transitions_dir),
                pairs=transition_pairs,
                level="city",
                city=city,
            )
        except Exception:
            continue
        city_tables[city] = _transition_table_percent(df_city)
        city_slug_file = city.lower().replace(" ", "_").replace("'", "")
        p = (
            paths.plots_dir
            / f"city_{city_slug_file}_transition_matrix_over_elections.png"
        )
        plot_transition_matrix_over_elections(
            df_all=df_city,
            pairs_to_show=pairs_to_show,
            show_ci=True,
            figsize=(12, 10),
            suptitle=f"{city.title()} - Voter Transition Matrices Over Elections",
            save_path=str(p),
        )
        city_plot_paths[city] = p

    # 3) MAD tables and plots
    overall_mad, per_pair_mad = _mad_summary_tables(
        df_country=df_country,
        transitions_dir=str(paths.transitions_dir),
        transition_pairs=transition_pairs,
        target_cities=target_cities,
    )
    mad_plot_path = paths.plots_dir / "cities_aggregate_deviation_comparison.png"
    plot_all_cities_aggregate_deviation_subplots(
        target_cities=target_cities,
        transitions_dir=str(paths.transitions_dir),
        transition_pairs=transition_pairs,
        metric="mean_abs_deviation",
        pairs_to_show=pairs_to_show,
        save_path=str(mad_plot_path),
    )

    # 4) MAD table lens plot
    mad_table_lens = _create_mad_table_lens(
        per_pair_mad, transition_pairs, paths.plots_dir
    )

    # 5) Diagnostics per pair
    diag_rows = []
    diag_images: Dict[str, List[Path]] = {}
    for pair in transition_pairs:
        d = _load_diagnostics_for_pair(paths.logs_dir, pair)
        diag_rows.append({"pair_tag": pair, "summary": _short_diagnostic_text(d or {})})
        trace_path = paths.transitions_dir / pair / "country_trace.nc"
        if trace_path.exists():
            diag_images[pair] = _save_diagnostic_plots(
                trace_path, paths.plots_dir, prefix=f"diag_{pair}"
            )

    df_diag = pd.DataFrame(diag_rows)

    # 5) Assemble Markdown
    md_path = paths.reports_dir / "summary.md"
    lines: List[str] = []
    lines.append("## Voter Transition Analysis Report")
    lines.append("")
    lines.append("### Country-level Transition Matrices (estimate, 1 decimal %)")
    lines.append("")
    lines.append(_df_to_markdown(country_table))
    lines.append("")
    rel_country = os.path.relpath(country_plot_path, start=paths.reports_dir)
    lines.append(f"![Country transitions]({rel_country})")
    lines.append("")
    lines.append("### City-level Transition Matrices")
    for city, table in city_tables.items():
        lines.append("")
        lines.append(f"#### {city.title()}")
        lines.append(_df_to_markdown(table))
        p = city_plot_paths.get(city)
        if p is not None:
            lines.append("")
            rel_p = os.path.relpath(p, start=paths.reports_dir)
            lines.append(f"![{city} transitions]({rel_p})")
    lines.append("")
    lines.append("### Mean Absolute Deviation (MAD) Summaries")
    lines.append("")
    lines.append("#### Overall by City (pp)")
    lines.append(
        _df_to_markdown(
            overall_mad.rename(
                columns={"mean_mad_pp": "Mean MAD (pp)", "max_mad_pp": "Max MAD (pp)"}
            )
        )
    )
    lines.append("")
    lines.append("#### Per Pair by City (pp)")
    lines.append(
        _df_to_markdown(
            per_pair_mad.rename(columns={"mean_abs_deviation_pp": "MAD (pp)"})
        )
    )
    lines.append("")
    rel_mad = os.path.relpath(mad_plot_path, start=paths.reports_dir)
    lines.append(f"![City MAD subplots]({rel_mad})")
    lines.append("")

    # Add MAD table lens plot
    if mad_table_lens:
        lines.append("#### MAD Rankings by Knesset Pair")
        rel_table_lens = os.path.relpath(mad_table_lens, start=paths.reports_dir)
        lines.append(f"![MAD Table Lens]({rel_table_lens})")
        lines.append("")

    lines.append("### Sampling Diagnostics")
    lines.append(_df_to_markdown(df_diag))
    lines.append("")
    lines.append(_interpret_diagnostics(df_diag))
    lines.append("")

    for pair, imgs in diag_images.items():
        if not imgs:
            continue
        lines.append("")
        lines.append(f"#### {pair} Diagnostic plots")
        for img in imgs:
            rel_img = os.path.relpath(img, start=paths.reports_dir)
            lines.append(f"![{pair} diag]({rel_img})")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path

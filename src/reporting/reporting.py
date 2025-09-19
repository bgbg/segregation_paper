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


def _save_diagnostic_plots(trace_path: Path, out_dir: Path, prefix: str) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    try:
        idata = az.from_netcdf(trace_path)
    except Exception:
        return paths
    # Rank plots
    p1 = out_dir / f"{prefix}_rank.png"
    az.plot_rank(idata)
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(p1)
    # Energy plot
    p2 = out_dir / f"{prefix}_energy.png"
    az.plot_energy(idata)
    plt.tight_layout()
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(p2)
    # Autocorr plot (thin)
    p3 = out_dir / f"{prefix}_autocorr.png"
    az.plot_autocorr(idata, max_lag=100)
    plt.tight_layout()
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(p3)
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
        plots_dir=(Path(save_dir) / "transitions" / "plots").resolve(),
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

    # 4) Diagnostics per pair
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
    lines.append("### Sampling Diagnostics")
    lines.append(_df_to_markdown(df_diag))
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

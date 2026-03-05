"""Generate raw vote-share line chart for Haredi-filtered polling stations.

Produces a figure showing Shas% and UTJ% across elections 21-25, by city,
independent of the ecological inference model. Tufte/Few principles:
high data-ink ratio, no chartjunk.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ELECTION_LABELS = {
    21: "Apr '19\n(21)",
    22: "Sep '19\n(22)",
    23: "Mar '20\n(23)",
    24: "Mar '21\n(24)",
    25: "Nov '22\n(25)",
}

CITY_NAMES_EN = {
    "אשדוד": "Ashdod",
    "בית שמש": "Beit Shemesh",
    "אלעד": "Elad",
    "בני ברק": "Bnei Brak",
    "ירושלים": "Jerusalem",
    "מודיעין עילית": "Modi'in Illit",
}

KNESSETS = [21, 22, 23, 24, 25]


def load_and_aggregate(knessets: list[int], data_dir: Path) -> pd.DataFrame:
    """Load harmonized parquet files and compute city-level vote shares."""
    rows = []
    for kn in knessets:
        df = pd.read_parquet(data_dir / f"harmonized_{kn}.parquet")
        for city_he, city_en in CITY_NAMES_EN.items():
            city_df = df[df["city_name"] == city_he]
            if city_df.empty:
                continue
            total_legal = city_df["legal"].sum()
            shas_pct = city_df["A_shas"].sum() / total_legal * 100
            utj_pct = city_df["B_agudat"].sum() / total_legal * 100
            rows.append({
                "knesset": kn,
                "city": city_en,
                "shas_pct": shas_pct,
                "utj_pct": utj_pct,
            })
        # Country aggregate
        total_legal = df["legal"].sum()
        rows.append({
            "knesset": kn,
            "city": "All cities",
            "shas_pct": df["A_shas"].sum() / total_legal * 100,
            "utj_pct": df["B_agudat"].sum() / total_legal * 100,
        })
    return pd.DataFrame(rows)


def plot_raw_vote_shares(data: pd.DataFrame, output_path: Path) -> None:
    """Create a 6x2 grid: one row per city, columns for Shas% and UTJ%."""
    cities_order = ["Ashdod", "Beit Shemesh", "Elad", "Bnei Brak",
                    "Jerusalem", "Modi'in Illit"]

    x_positions = list(range(len(KNESSETS)))
    x_labels = [ELECTION_LABELS[k] for k in KNESSETS]

    # Normalize: deviation from each city's Kn 23 value (pre-disruption peak)
    baseline_kn = 23
    fig, ax = plt.subplots(figsize=(5, 3.5))

    city_colors = {
        "Ashdod": "#1f77b4",
        "Beit Shemesh": "#ff7f0e",
        "Elad": "#2ca02c",
        "Bnei Brak": "#d62728",
        "Jerusalem": "#9467bd",
        "Modi'in Illit": "#8c564b",
    }

    for city in cities_order:
        city_data = data[data["city"] == city].sort_values("knesset")
        baseline = city_data.loc[city_data["knesset"] == baseline_kn, "shas_pct"].values[0]
        deviation = city_data["shas_pct"].values - baseline
        ax.plot(
            x_positions, deviation,
            color=city_colors[city], linewidth=1.1,
            marker="o", markersize=3,
        )
    # Label at the Kn 24 dip point for each city
    # Label at the end of each line (Kn 25), stacked to avoid overlap
    end_vals = {}
    for city in cities_order:
        city_data = data[data["city"] == city].sort_values("knesset")
        baseline = city_data.loc[city_data["knesset"] == baseline_kn, "shas_pct"].values[0]
        deviation = city_data["shas_pct"].values - baseline
        end_vals[city] = deviation[-1]

    # Sort by end value and space labels apart
    sorted_cities = sorted(end_vals.items(), key=lambda x: x[1])
    min_gap = 0.35
    label_positions = {}
    for i, (city, val) in enumerate(sorted_cities):
        if i > 0:
            prev_city = sorted_cities[i - 1][0]
            prev_pos = label_positions[prev_city]
            label_positions[city] = max(val, prev_pos + min_gap)
        else:
            label_positions[city] = val

    for city in cities_order:
        ax.annotate(
            city, (x_positions[-1], end_vals[city]),
            textcoords="offset points",
            xytext=(5, (label_positions[city] - end_vals[city]) * 15),
            fontsize=6, va="center", color=city_colors[city],
        )

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvspan(2.5, 3.5, color="#e0e0e0", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.4)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.set_ylabel("Change in Shas vote share (pp)", fontsize=8)
    ax.set_title("Deviation from Knesset 23 baseline", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    data_dir = Path("data/interim")
    output_path = Path("transition_paper/plots/raw_vote_shares.png")

    data = load_and_aggregate(KNESSETS, data_dir)

    # Print summary table
    print("\nRaw vote shares (%) in Haredi-filtered polling stations:\n")
    for kn in KNESSETS:
        kn_data = data[data["knesset"] == kn]
        print(f"Knesset {kn} ({ELECTION_LABELS[kn].replace(chr(10), ' ')}):")
        for _, row in kn_data.iterrows():
            print(f"  {row['city']:18s}  Shas={row['shas_pct']:5.1f}%  UTJ={row['utj_pct']:5.1f}%")
        print()

    plot_raw_vote_shares(data, output_path)


if __name__ == "__main__":
    main()

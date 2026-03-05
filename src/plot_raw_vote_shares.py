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

    fig, axes = plt.subplots(
        6, 2, figsize=(6, 8), sharex=True, sharey=False,
    )

    for row_idx, city in enumerate(cities_order):
        city_data = data[data["city"] == city].sort_values("knesset")

        for col_idx, metric in enumerate(["shas_pct", "utj_pct"]):
            ax = axes[row_idx, col_idx]
            ax.plot(
                x_positions, city_data[metric].values,
                color="black", linewidth=1.2,
                marker="o", markersize=3,
            )
            # Shade the 23->24 dip region
            ax.axvspan(2.5, 3.5, color="#e0e0e0", zorder=0)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.2, linewidth=0.4)

            # City label on left column only
            if col_idx == 0:
                ax.set_ylabel(city, fontsize=8, rotation=0, labelpad=60,
                              ha="right", va="center")

            # X-axis labels only on bottom row
            if row_idx == 5:
                ax.set_xticks(x_positions)
                ax.set_xticklabels(x_labels, fontsize=6.5)

            # Column titles on top row only
            if row_idx == 0:
                title = "Shas (%)" if col_idx == 0 else "UTJ (%)"
                ax.set_title(title, fontsize=9)

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

"""Generate raw vote-share line chart for Haredi-filtered polling stations.

Produces a figure showing Shas% and UTJ% across elections 21-25, by city,
independent of the ecological inference model. Tufte/Few principles:
high data-ink ratio, no chartjunk.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ELECTION_LABELS = {
    21: "Apr 2019\n(Kn 21)",
    22: "Sep 2019\n(Kn 22)",
    23: "Mar 2020\n(Kn 23)",
    24: "Mar 2021\n(Kn 24)",
    25: "Nov 2022\n(Kn 25)",
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
    """Create a two-panel line chart: Shas% and UTJ% by city over elections."""
    fig, (ax_shas, ax_utj) = plt.subplots(
        1, 2, figsize=(10, 4.5), sharey=False
    )

    cities_order = ["Ashdod", "Beit Shemesh", "Elad", "Bnei Brak",
                    "Jerusalem", "Modi'in Illit"]

    x_positions = list(range(len(KNESSETS)))
    x_labels = [ELECTION_LABELS[k] for k in KNESSETS]

    # Muted palette
    city_colors = {
        "Ashdod": "#1f77b4",
        "Beit Shemesh": "#ff7f0e",
        "Elad": "#2ca02c",
        "Bnei Brak": "#d62728",
        "Jerusalem": "#9467bd",
        "Modi'in Illit": "#8c564b",
    }

    for ax, metric, title in [
        (ax_shas, "shas_pct", "Shas vote share (%)"),
        (ax_utj, "utj_pct", "UTJ vote share (%)"),
    ]:
        # Plot aggregate as thick gray dashed line
        agg = data[data["city"] == "All cities"].sort_values("knesset")
        ax.plot(
            x_positions, agg[metric].values,
            color="gray", linewidth=2.5, linestyle="--",
            label="All cities", zorder=1,
        )

        # Plot individual cities
        for city in cities_order:
            city_data = data[data["city"] == city].sort_values("knesset")
            ax.plot(
                x_positions, city_data[metric].values,
                color=city_colors[city], linewidth=1.3,
                marker="o", markersize=4,
                label=city, zorder=2,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    ax_shas.set_ylabel("Vote share (%)", fontsize=10)
    ax_shas.legend(
        fontsize=7, loc="lower left", framealpha=0.8,
        ncol=2, handlelength=1.5,
    )

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

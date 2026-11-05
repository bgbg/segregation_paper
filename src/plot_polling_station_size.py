"""Distribution of polling-station size (registered voters) in the latest Knesset election.

Evidence that Israeli polling-station size is administratively mechanical: the
Central Elections Committee splits stations so that registered voters per station
(בז"ב = בעלי זכות בחירה) stay under a hard ceiling of ~800, giving a tight,
country-wide distribution.

Usage:
    python src/plot_polling_station_size.py            # Knesset 25 (default)
    python src/plot_polling_station_size.py --knesset 24
"""
from pathlib import Path

import defopt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_utils import despine_and_remove_grid

RAW = Path("data/raw")
OUT = Path("transition_paper/plots")
EXTERNAL_ENVELOPES = "מעטפות חיצוניות"  # double-envelope votes, not real stations
BZB = "בזב"  # בעלי זכות בחירה -- registered / eligible voters at the station
CITY = "שם ישוב"


def load_station_sizes(knesset: int) -> pd.Series:
    """Registered voters per polling station, external-envelope rows dropped."""
    df = pd.read_csv(RAW / f"results_{knesset}.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()  # K20 header has stray leading spaces
    real = df[df[CITY].str.strip() != EXTERNAL_ENVELOPES]
    return real[BZB].astype(int)


def summarize(sizes: pd.Series, label: str) -> None:
    q = sizes.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    band = ((sizes >= 450) & (sizes <= 750)).mean()
    print(
        f"{label}: n={len(sizes):>6,}  median={sizes.median():.0f}  "
        f"IQR={q[0.25]:.0f}-{q[0.75]:.0f}  p05-p95={q[0.05]:.0f}-{q[0.95]:.0f}  "
        f"max={sizes.max()}  CV={sizes.std() / sizes.mean():.2f}  "
        f"share 450-750={band:.0%}"
    )


def main(*, knesset: int = 25) -> None:
    """Plot the polling-station size distribution for one Knesset election.

    :param knesset: Knesset number (15-25).
    """
    sizes = load_station_sizes(knesset)

    # Time-stability context: same summary for the last few elections.
    print("Registered voters per polling station (external envelopes excluded):")
    for k in range(20, 26):
        try:
            summarize(load_station_sizes(k), f"  Knesset {k}")
        except FileNotFoundError:
            pass

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(sizes, bins=np.arange(0, sizes.max() + 20, 20), color="#4C72B0", ax=ax)

    ymax = ax.get_ylim()[1]
    median = sizes.median()
    ax.axvline(median, color="black", lw=1.5, ls="--")
    ax.text(median - 12, ymax * 0.97, f"median {median:.0f}",
            ha="right", va="top", fontsize=11)

    p99 = sizes.quantile(0.99)
    ax.annotate(
        f"hard upper bound\n99% of stations below {p99:.0f} (max {sizes.max()})",
        xy=(p99 + 8, ymax * 0.08), xytext=(sizes.min() + 40, ymax * 0.78),
        ha="left", fontsize=10, color="#555",
        arrowprops=dict(arrowstyle="->", color="#555",
                        connectionstyle="arc3,rad=-0.2"),
    )

    ax.set_xlabel("Registered voters at the polling station")
    ax.set_ylabel("Polling stations")
    ax.set_title(
        f"Israeli polling-station size is administratively bounded\n"
        f"Knesset {knesset} — {len(sizes):,} stations, "
        f"IQR {sizes.quantile(.25):.0f}–{sizes.quantile(.75):.0f}, "
        f"CV {sizes.std() / sizes.mean():.2f}",
        fontsize=12, loc="left",
    )
    despine_and_remove_grid(ax)
    ax.spines["bottom"].set_visible(True)

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"polling_station_size_k{knesset}.png"
    fig.tight_layout()
    fig.savefig(dest, dpi=200)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    defopt.run(main)

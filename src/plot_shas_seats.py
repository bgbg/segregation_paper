"""Generate a Shas-seats-per-Knesset plot matching the city-comparison aesthetics."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


# Shas Knesset seats (Wikipedia / knesset.gov.il, verified for Knessets 18–25)
SEATS = [
    (18, "Feb 2009", 11),
    (19, "Jan 2013", 11),
    (20, "Mar 2015", 7),
    (21, "Apr 2019", 8),
    (22, "Sep 2019", 9),
    (23, "Mar 2020", 9),
    (24, "Mar 2021", 9),
    (25, "Nov 2022", 11),
]

OUTPUT_PATH = Path("transition_paper/plots/shas_seats_over_knessets.png")


def main() -> None:
    knessets = [k for k, _, _ in SEATS]
    dates = [d for _, d, _ in SEATS]
    seats = [s for _, _, s in SEATS]

    fig, ax = plt.subplots(figsize=(18, 9))

    ax.plot(knessets, seats, "-ok", linewidth=3, markersize=12)

    # Knesset/date labels at the top of the plot
    y_top = max(seats) + 1.5
    for k, d in zip(knessets, dates):
        ax.text(
            k,
            y_top,
            f"#{k}\n{d}",
            ha="center",
            va="bottom",
            fontsize=14,
            color="black",
        )

    # Aesthetics: no top/right spines, no grid
    sns.despine(ax=ax)
    ax.grid(False)

    # Axis labels in the same style as the reference plot
    ax.set_ylabel(
        "Shas Seats\nin Knesset",
        ha="right",
        ma="left",
        rotation=0,
        y=1,
        va="top",
        fontsize=16,
    )
    ax.set_xlabel("Knesset Number", ha="right", va="top", x=1)

    ax.tick_params(axis="both", which="major", labelsize=16)

    # X-axis: one tick per Knesset
    ax.set_xticks(knessets)
    ax.set_xlim(min(knessets) - 0.5, max(knessets) + 0.5)

    # Y-axis: integer seats, headroom for the labels at top
    ax.set_ylim(0, y_top + 1.5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

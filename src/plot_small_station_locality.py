"""Are very small ballot boxes confined to very small localities?

Tests the claim that a polling station with few registered voters (בז"ב) only
occurs in a small town/village. Plots every station's size against its locality's
total electorate, highlights the sub-threshold stations, and traces the lower
envelope of station size by locality size.

Verdict for Knesset 25: confirmed. 99% of stations below 250 registered voters
sit in single-station localities under 500 electorate; the only exception is one
special-purpose station in Dimona.

Usage:
    python src/plot_small_station_locality.py                 # Knesset 25
    python src/plot_small_station_locality.py --knesset 24 --small 300
"""
from pathlib import Path

import defopt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import despine_and_remove_grid

RAW = Path("data/raw")
OUT = Path("transition_paper/plots")
EXTERNAL_ENVELOPES = "מעטפות חיצוניות"
BZB = "בזב"            # registered voters at the station
CITY = "שם ישוב"
CITY_CODE = "סמל ישוב"  # locality identifier (stable across name spellings)


def load(knesset: int) -> pd.DataFrame:
    df = pd.read_csv(RAW / f"results_{knesset}.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df[df[CITY].str.strip() != EXTERNAL_ENVELOPES].copy()
    df[BZB] = df[BZB].astype(int)
    loc = df.groupby(CITY_CODE)[BZB].agg(loc_electorate="sum", n_stations="size")
    return df.join(loc, on=CITY_CODE)


def main(*, knesset: int = 25, small: int = 250) -> None:
    """Plot station size vs locality electorate and test the "small town only" claim.

    :param knesset: Knesset number (15-25).
    :param small: threshold (registered voters) defining a "very small" station.
    """
    df = load(knesset)
    tiny = df[df[BZB] < small]
    in_small_loc = (tiny["loc_electorate"] < 500).mean()
    single = (tiny["n_stations"] == 1).mean()
    exceptions = tiny[tiny["loc_electorate"] > 2000].sort_values("loc_electorate")

    # Threshold sensitivity: how many "big locality" exceptions at other cutoffs.
    sens = {t: (df[df[BZB] < t]["loc_electorate"] > 2000).sum()
            for t in (small - 50, small, small + 50)}
    sens_txt = ", ".join(f"<{t}: {n}" for t, n in sens.items())

    print(f"Knesset {knesset}: {len(df):,} stations, {df[CITY_CODE].nunique():,} localities")
    print(f"stations with <{small} registered voters: {len(tiny)} ({len(tiny)/len(df):.1%})")
    print(f"  ... in localities under 500 electorate: {in_small_loc:.1%}")
    print(f"  ... that are the only station in their locality: {single:.1%}")
    print(f"  exceptions (locality electorate > 2000):")
    print(exceptions[[CITY, "קלפי", BZB, "loc_electorate", "n_stations"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    big = df[df[BZB] >= small]
    ax.scatter(big["loc_electorate"], big[BZB], s=8, c="#B7C4D8",
               alpha=0.5, lw=0, label=f"station ≥ {small}")
    ax.scatter(tiny["loc_electorate"], tiny[BZB], s=22, c="#D1662A",
               alpha=0.9, lw=0, label=f"station < {small} (\"very small\")")

    # Lower envelope: 1st percentile of station size per locality-electorate bin.
    edges = np.logspace(np.log10(df["loc_electorate"].min()),
                        np.log10(df["loc_electorate"].max()), 16)
    mid = np.sqrt(edges[:-1] * edges[1:])
    p01 = [df.loc[(df["loc_electorate"] >= lo) & (df["loc_electorate"] < hi), BZB].quantile(0.01)
           for lo, hi in zip(edges[:-1], edges[1:])]
    ax.plot(mid, p01, "-", color="#333", lw=1.8, label="1st percentile of station size (binned)")

    ax.axhline(small, color="#D1662A", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Locality total registered voters (log scale)")
    ax.set_ylabel("Registered voters at the polling station")

    # City names are Hebrew (matplotlib can't render them); annotate with numbers.
    for _, r in exceptions.iterrows():
        ax.annotate(f"{r[BZB]}-voter station in a\n"
                    f"{int(r['loc_electorate']):,}-voter locality ({int(r['n_stations'])} stations)",
                    xy=(r["loc_electorate"], r[BZB]),
                    xytext=(r["loc_electorate"] * 0.18, r[BZB] + 260),
                    fontsize=9, color="#333",
                    arrowprops=dict(arrowstyle="->", color="#333"))

    verdict = "confirmed" if len(exceptions) <= 2 else "mostly holds"
    txt = (f"Claim: very small stations occur only in very small localities\n"
           f"Knesset {knesset} - verdict: {verdict} ({len(exceptions)} exception"
           f"{'s' if len(exceptions) != 1 else ''})\n\n"
           f"stations < {small} voters: {len(tiny)} of {len(df):,} ({len(tiny)/len(df):.1%})\n"
           f"  {in_small_loc:.0%} are in localities < 500 total voters\n"
           f"  {single:.0%} are the locality's only station\n"
           f"exceptions (locality > 2000 voters) by cutoff: {sens_txt}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9.5, family="monospace",
            bbox=dict(boxstyle="round", fc="#F5F2EC", ec="#CCC"))

    ax.legend(loc="lower right", frameon=False, fontsize=9)
    despine_and_remove_grid(ax)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"small_station_vs_locality_k{knesset}.png"
    fig.tight_layout()
    fig.savefig(dest, dpi=200)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    defopt.run(main)

"""Is Dimona's tiny ballot box (kalpi 91) politically different from the rest of the city?

The user flagged a ~196-voter station in Dimona (Knesset 25, kalpi 91) as an
outlier. This traces, election by election (K20-K25):

  * Gini of the party-vote vector  (1 = one party takes everything)
  * Effective number of parties, ENP = 1 / sum(p_i^2)
  * the leading party / bloc share

for three groups of Dimona boxes:
  * kalpi 91  (kalpi 990 in K21 -- same profile)   -- "the box"
  * kalpi 90  -- the other small box, opposite character
  * every other Dimona box ("regular")

Usage:
    python src/plot_dimona_anomalous_box.py
"""
from pathlib import Path

import defopt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import despine_and_remove_grid

RAW = Path("data/raw")
OUT = Path("transition_paper/plots")
CITY_HE = "דימונה"
KNESSETS = list(range(20, 26))
ELECTION_DATE = {20: "Mar-15", 21: "Apr-19", 22: "Sep-19", 23: "Mar-20",
                 24: "Mar-21", 25: "Nov-22"}
META = {"סמל ועדה", "ברזל", "שם ישוב", "סמל ישוב", "קלפי", "ריכוז", "שופט",
        "בזב", "מצביעים", "פסולים", "כשרים", "בוחרים", "ברק", "נתניהו",
        "Unnamed: 4", "Unnamed: 10", "בזב.1", "מצביעים.1", "פסולים.1"}
# party letter code -> readable name, only the codes that surface as top-2 in Dimona
PARTY = {
    "מחל": "Likud", "שס": "Shas", "ג": "UTJ", "אמת": "Labor",
    "ל": "Yisrael Beiteinu", "כ": "Kulanu", "טב": "Jewish Home/Yamina",
    "ת": "New Hope", "ב": "Yamina", "כן": "National Unity",
}


def party_name(code: str, knesset: int) -> str:
    if code == "פה":
        return "Yesh Atid" if knesset in (20, 24, 25) else "Kahol Lavan"
    return PARTY.get(code, code)


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    return (2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n


def enp(p: np.ndarray) -> float:
    p = np.asarray(p, float)
    s = p.sum()
    return np.nan if s == 0 else 1.0 / np.sum((p / s) ** 2)


def load_dimona(knesset: int) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(RAW / f"results_{knesset}.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    citycol = next(c for c in df.columns if "ישוב" in c and "סמל" not in c)
    d = df[df[citycol].astype(str).str.strip() == CITY_HE].copy()
    parties = [c for c in d.columns if c not in META]
    d[parties] = d[parties].apply(pd.to_numeric, errors="coerce").fillna(0)
    d["kalpi_num"] = pd.to_numeric(d["קלפי"], errors="coerce")
    d["group"] = np.where(d["kalpi_num"].isin([91, 990]), "box91",
                 np.where(d["kalpi_num"].eq(90), "box90", "regular"))
    return d, parties


def per_box_metrics(d: pd.DataFrame, parties: list[str]) -> pd.DataFrame:
    v = d[parties].to_numpy(float)
    return pd.DataFrame({
        "kalpi": d["קלפי"].values, "group": d["group"].values,
        "valid": v.sum(1),
        "gini": [gini(row) for row in v],
        "enp": [enp(row) for row in v],
    })


def group_leader(d: pd.DataFrame, parties: list[str], knesset: int) -> dict:
    tot = d[parties].sum()
    tot = tot[tot > 0].sort_values(ascending=False)
    s = tot.sum()
    reg = pd.to_numeric(d["בזב"], errors="coerce").sum()
    voted = pd.to_numeric(d["מצביעים"], errors="coerce").sum()
    return {
        "winner": party_name(tot.index[0], knesset), "winner_share": tot.iloc[0] / s,
        "likud": d["מחל"].sum() / s if "מחל" in parties else np.nan,
        "center": d["פה"].sum() / s if "פה" in parties else np.nan,
        "turnout": voted / reg if reg else np.nan,
        "valid": s,
    }


def main() -> None:
    """Plot Gini, ENP and leading party for Dimona's box 91 vs box 90 vs the rest."""
    boxrows, leaders = [], []
    for k in KNESSETS:
        d, parties = load_dimona(k)
        m = per_box_metrics(d, parties)
        m["knesset"] = k
        boxrows.append(m)
        for g, sub in d.groupby("group"):
            leaders.append({"knesset": k, "group": g, **group_leader(sub, parties, k)})
    box = pd.concat(boxrows, ignore_index=True)
    lead = pd.DataFrame(leaders)

    print(box.groupby(["knesset", "group"])[["gini", "enp", "valid"]].mean().round(2))
    print()
    print(lead.set_index(["knesset", "group"])[
        ["winner", "winner_share", "likud", "center", "turnout", "valid"]].round(2))

    reg = box[box.group == "regular"]
    reg_stats = reg.groupby("knesset")[["gini", "enp"]].agg(["median", "min", "max"])
    b91 = box[box.group == "box91"].groupby("knesset")[["gini", "enp"]].mean()
    b90 = box[box.group == "box90"].groupby("knesset")[["gini", "enp"]].mean()

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    x = np.array(KNESSETS)
    xt = [f"K{k}\n{ELECTION_DATE[k]}" for k in KNESSETS]

    for ax, metric, label in [
        (axes[0], "gini", "Gini of party-vote vector\n(1 = single party sweeps)"),
        (axes[1], "enp", "Effective number of parties\n1 / Σ pᵢ²"),
    ]:
        lo, hi = reg_stats[metric]["min"], reg_stats[metric]["max"]
        ax.fill_between(x, lo, hi, color="#CBD3DF", alpha=0.6,
                        label="regular Dimona boxes (range)")
        ax.plot(x, reg_stats[metric]["median"], "o-", color="#5B6B85", lw=2,
                label="regular Dimona boxes (median)")
        ax.scatter(reg["knesset"], reg[metric], s=10, color="#8494AC", alpha=0.35, zorder=1)
        ax.plot(b91.index, b91[metric], "s-", color="#D1662A", lw=2.5,
                markersize=8, label="kalpi 91  (\"the box\")")
        ax.plot(b90.index, b90[metric], "^--", color="#3A7D44", lw=2,
                markersize=8, label="kalpi 90  (other small box)")
        ax.set_ylabel(label)
        despine_and_remove_grid(ax)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[0].set_ylim(0.75, 0.97)
    axes[0].text(20, 0.955, "Gini barely flags box 91 — its vote is about as "
                 "concentrated as any Dimona box.\nWhat differs is which party it "
                 "concentrates on (panel 3).", fontsize=8, style="italic", color="#555")
    axes[1].text(20, 6.0, "box 91 fragments in K24-K25: its anti-Likud vote splits "
                 "across several centre/left lists", fontsize=8, style="italic", color="#555")

    # Panel 3: leading-party share, regular vs box 91, with winner labels.
    ax = axes[2]
    for g, color, mk, lbl, dy in [
        ("regular", "#5B6B85", "o", "Dimona, all boxes except 90 & 91", 11),
        ("box91", "#D1662A", "s", 'kalpi 91  ("the box")', -18),
    ]:
        sub = lead[lead.group == g].sort_values("knesset")
        ax.plot(sub.knesset, sub.winner_share, f"{mk}-", color=color, lw=2.5,
                markersize=8, label=f"{lbl} — winner's share")
        for _, r in sub.iterrows():
            ax.annotate(r.winner, (r.knesset, r.winner_share),
                        textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=8, color=color, fontweight="bold")
    ax.set_ylabel("Leading party's vote share\n(within the group)")
    ax.set_ylim(0, 0.8)
    ax.text(20, 0.74, "the rest of Dimona elects Likud every time; kalpi 91 never does",
            fontsize=8.5, style="italic", color="#333")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    despine_and_remove_grid(ax)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(xt)
    fig.suptitle("Dimona's kalpi 91 votes against the rest of the city, every election\n"
                 "~200 registered voters — centrist/anti-Likud, while Dimona is a "
                 "Likud stronghold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "dimona_anomalous_box.png"
    fig.savefig(dest, dpi=200)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    defopt.run(main)

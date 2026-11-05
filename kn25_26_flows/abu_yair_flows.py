"""Arab-sector voter flows around the 2021 "Abu Yair" campaign.

Same model as flows.py, fitted to consecutive election pairs (Kn23->24->25->26)
on polling places that were Arab-sector (>75% of valid votes to Arab lists)
in the earlier election. Fits are cached in data/processed/kn25_26_flows/.

Run (conda env `bayesian`):
    python kn25_26_flows/abu_yair_flows.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arviz as az  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pymc as pm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from flows import OUT, ROOT, build_model, load_groups  # noqa: E402

sys.path.insert(0, str(ROOT))
from src.plot_utils import heb  # noqa: E402

THRESHOLD = 0.75
LIKUD, SHAS = ["מחל"], ["שס"]
GROUPS = {  # "joint" = every Arab list except Ra'am (Ra'am was inside the Joint List in 2020)
    23: {"joint": ["ודעם"], "likud": LIKUD, "shas": SHAS},
    24: {"raam": ["עם"], "joint": ["ודעם"], "likud": LIKUD, "shas": SHAS},
    25: {"raam": ["עם"], "joint": ["ום", "ד"], "likud": LIKUD, "shas": SHAS},
    26: {"raam": ["עם"], "joint": ["ודם", "ד"], "likud": LIKUD, "shas": SHAS},
}
YEAR = {23: 2020, 24: 2021, 25: 2022, 26: 2026}
PAIRS = [(23, 24), (24, 25), (25, 26)]


def cats(k):
    return list(GROUPS[k]) + ["other", "abstained"]


def arab_sector(df, k):
    arab = df[[g for g in ("raam", "joint") if g in GROUPS[k]]].sum(axis=1)
    legal = df[cats(k)].drop(columns="abstained").sum(axis=1)
    return arab > THRESHOLD * legal


def aligned(a, b, max_ratio=4 / 3):
    x1, x2 = load_groups(a, GROUPS[a]), load_groups(b, GROUPS[b])
    x1 = x1[arab_sector(x1, a) & (x1["can_vote"] > 0)]
    common = x1.index.intersection(x2.index)
    x1, x2 = x1.loc[common], x2.loc[common]
    r = x2["can_vote"] / x1["can_vote"]
    keep = (r < max_ratio) & (r > 1 / max_ratio)
    return x1.loc[keep, cats(a)], x2.loc[keep, cats(b)]


def fit(a, b):
    path = OUT / f"abu_yair_kn{a}_{b}.nc"
    x1, x2 = aligned(a, b)
    if not path.exists():
        with build_model(x1.values, x2.values, frm=cats(a), to=cats(b)):
            idata = pm.sample(draws=1000, tune=1000, chains=4, target_accept=0.9, random_seed=a)
        idata.to_netcdf(path)
    idata = az.from_netcdf(path)
    rhat = float(az.rhat(idata, var_names=["M"])["M"].max())
    M = idata.posterior["M"].stack(s=("chain", "draw")).values  # to x from x samples
    flows = M * x1.sum().values[None, :, None]
    print(f"Kn{a}->{b}: {len(x1)} stations, max r_hat {rhat:.3f}")
    return flows, x1, x2


def flow_table(results):
    rows = []
    for (a, b), (flows, *_) in results.items():
        for i, t in enumerate(cats(b)):
            for j, f in enumerate(cats(a)):
                s = flows[i, j]
                rows.append(dict(pair=f"{YEAR[a]}->{YEAR[b]}", src=f, dst=t, mean=s.mean(),
                                 q05=np.quantile(s, 0.05), q95=np.quantile(s, 0.95)))
    return pd.DataFrame(rows)


# --- network over time, focused on Likud ---------------------------------
H = {"raam": "רע״ם", "joint": "שאר הרשימות הערביות", "likud": "ליכוד", "shas": "ש״ס", "other": "מפלגות אחרות",
     "abstained": "לא הצביעו"}
H23 = dict(H, joint="הרשימה המשותפת (כולל רע״ם)")
ROW = {"raam": 5, "joint": 4, "likud": 2.5, "shas": 1.6, "other": 0.8, "abstained": 0}
BLUE, GREY, GREEN = "#1f4e9c", "#999999", "#2e8b57"
NODE_C = {"raam": GREEN, "joint": GREEN, "likud": BLUE, "shas": "#2a2a2a", "other": "#c8a000", "abstained": GREY}


def network(results, sector_totals, min_votes=300):
    xs = {k: x for k, x in zip([23, 24, 25, 26], [0, 4, 8, 12])}
    W, Hh = 2.6, 0.62
    fig, ax = plt.subplots(figsize=(16, 8.5))
    for k, x in xs.items():
        ax.text(x, 5.85, str(YEAR[k]), ha="center", fontsize=16, weight="bold")
        for g in cats(k):
            if g == "shas":
                continue
            y = ROW[g]
            hl = g == "likud"
            ax.add_patch(FancyBboxPatch((x - W / 2, y - Hh / 2), W, Hh, boxstyle="round,pad=0.03,rounding_size=0.1",
                                        ec=NODE_C[g], fc="#eef2fa" if hl else "white", lw=2.6 if hl else 1.4, zorder=3))
            name = (H23 if k == 23 else H)[g]
            ax.text(x, y + 0.1, heb(name), ha="center", va="center", fontsize=11 if not hl else 13,
                    weight="bold", color=NODE_C[g], zorder=4)
            ax.text(x, y - 0.17, heb(f"{sector_totals.loc[k, g]:,.0f} קולות"), ha="center", va="center",
                    fontsize=9, color="#444444", zorder=4)
    for (a, b), (flows, *_) in results.items():
        mean = flows.mean(-1)
        for i, t in enumerate(cats(b)):
            for j, f in enumerate(cats(a)):
                v = mean[i, j]
                if "likud" not in (f, t) or v < min_votes or "shas" in (f, t):
                    continue
                y0, y1 = ROW[f], ROW[t]
                p0, p1 = (xs[a] + W / 2 + 0.05, y0), (xs[b] - W / 2 - 0.05, y1)
                color = BLUE if f == "likud" else NODE_C[f]
                lw = 1.2 + v / 350
                ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=f"-|>,head_length={0.5 + lw / 20},head_width={0.3 + lw / 30}",
                                             mutation_scale=18, lw=lw, color=color, alpha=0.7, zorder=2))
                lt = 0.3 if f == "likud" else 0.45  # keep labels off the crowded Likud ends
                mx, my = p0[0] + lt * (p1[0] - p0[0]), p0[1] + lt * (p1[1] - p0[1])
                ax.text(mx, my, heb(f"כ־{v / 1e3:.1f} אלף") if v >= 1000 else heb(f"כ־{v:,.0f}"),
                        ha="center", va="center", fontsize=10, weight="bold", color=color, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=1))
    ax.axvspan(1.5, 2.5, ymin=0.02, ymax=0.88, color=BLUE, alpha=0.06, lw=0)
    ax.text(2, 5.55, heb("קמפיין אבו יאיר"), ha="center", fontsize=12, color=BLUE, weight="bold")
    ax.set_xlim(-1.6, 13.6)
    ax.set_ylim(-0.6, 6.2)
    ax.axis("off")
    ax.set_title(heb("מאיפה באו הקולות של אבו יאיר, ולאן הלכו"), fontsize=18, weight="bold", loc="right")
    fig.text(0.98, 0.02, heb(f"קלפיות במגזר הערבי (מעל {THRESHOLD:.0%} מהקולות למפלגות ערביות). מוצגים רק מעברים אל הליכוד "
                             f"וממנו, של {min_votes} קולות ומעלה. הערכה לפי מודל, בקלפיות שהוצמדו בין כל זוג בחירות;")
             + "\n" + heb("בגלל שהליכוד קטן מאוד במגזר, אי־הוודאות במעברים האלה גדולה. בוריס גורליק, he.gorelik.net"),
             ha="right", fontsize=9, color="#666666")
    fig.savefig(OUT / "abu_yair_network.png", dpi=150, bbox_inches="tight", facecolor="white")


# --- Likud votes by locality (raw counts, no model) -----------------------
def by_locality(top=12):
    rows = []
    for k in [23, 24, 25, 26]:
        df = pd.read_csv(ROOT / f"data/raw/results_{k}.csv")
        df = df[df["בזב"] > 0]
        arab = df[[l for g in ("raam", "joint") if g in GROUPS[k] for l in GROUPS[k][g]]].sum(axis=1)
        sec = df[arab > THRESHOLD * df["כשרים"]]
        g = sec.groupby("סמל ישוב").agg(likud=("מחל", "sum"), legal=("כשרים", "sum"))
        g["year"], g["name"] = YEAR[k], sec.groupby("סמל ישוב")["שם ישוב"].first()
        rows.append(g.reset_index())
    d = pd.concat(rows)
    piv = d.pivot_table(index="סמל ישוב", columns="year", values="likud", fill_value=0)
    names = d[d["year"] == 2021].set_index("סמל ישוב")["name"]
    piv = piv.loc[piv[2021].nlargest(top).index]
    piv.index = [names[c] for c in piv.index]

    fig, ax = plt.subplots(figsize=(8, 6))
    yc = {2020: "#9db3d9", 2021: BLUE, 2022: "#6f8fc9", 2026: "#cccccc"}
    for i, (town, r) in enumerate(piv.iterrows()):
        ax.plot([min(r), max(r)], [i, i], color="#e5e5e5", lw=5, solid_capstyle="round", zorder=1)
        for y in (2026, 2020, 2022, 2021):
            ax.plot(r[y], i, "o", color=yc[y], ms=11 if y == 2021 else 8, mec="white", zorder=2)
    ax.set_yticks(range(len(piv)), [heb(t) for t in piv.index])
    ax.invert_yaxis()
    for y in (2020, 2021, 2022, 2026):
        ax.plot([], [], "o", color=yc[y], ms=8, label=str(y))
    ax.legend(frameon=False, ncol=4, loc="lower right")
    ax.set_xlabel(heb("קולות לליכוד בקלפיות של המגזר הערבי ביישוב"))
    ax.set_title(heb("איפה אבו יאיר הצליח: 12 היישובים עם הכי הרבה קולות לליכוד ב־2021"), fontsize=13)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    fig.savefig(OUT / "abu_yair_towns.png", dpi=150)
    return piv


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Arial"
    results = {(a, b): fit(a, b) for a, b in PAIRS}
    table = flow_table(results)
    table.to_csv(OUT / "abu_yair_flows.csv", index=False)
    lik = table[(table["src"] == "likud") | (table["dst"] == "likud")]
    print(lik[lik["mean"] >= 100].round(0).to_string(index=False))

    tot = {}
    for k in [23, 24, 25, 26]:
        x = load_groups(k, GROUPS[k])
        tot[k] = x[arab_sector(x, k) & (x["can_vote"] > 0)][cats(k)].sum()
    network(results, pd.DataFrame(tot).T)
    print(by_locality().astype(int).to_string())

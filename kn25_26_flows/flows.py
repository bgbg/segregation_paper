"""Country-wide voter flows 2022 (Knesset 25) -> 2026 (Knesset 26).

Same approach as src/transition_model: polling places aligned by
city_code + integer ballot code, abstained = can_vote - legal, column-softmax
transition matrix with a loyalty bias, Dirichlet-multinomial likelihood.
Generalised to a rectangular K_from x K_to matrix, country level only.

Run (from repo root, `bayesian` conda env):
    python kn25_26_flows/flows.py
"""

import platform
from pathlib import Path

# ponytail: PyTensor adds "-ld64" on macOS>=15, which current Apple clang rejects
# (pytensor/link/c/cmodule.py). Pretend macOS 14; drop once fixed upstream.
platform.mac_ver = lambda *a: ("14.0", ("", "", ""), "")

import arviz as az
import defopt
import numpy as np
import pandas as pd
import pymc as pm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/processed/kn25_26_flows"

# Ballot letters -> groups. Unlisted letters fall into "other".
GROUPS_25 = {
    "likud": ["מחל"],
    "shas": ["שס"],
    "rz_otzma": ["ט"],
    "bayit_yehudi": ["ב"],
    "national_unity": ["כן"],
    "change": ["פה", "ל", "אמת", "מרצ"],
    "utj": ["ג"],
    "arab": ["עם", "ום", "ד"],
}
GROUPS_26 = {
    "likud": ["מחל"],
    "shas": ["שס"],
    "rz_otzma": ["ט", "ב"],  # ט = RZ-Zehut, ב = Otzma Yehudit in 2026
    "winter": ["ך"],
    "hendel": ["די"],
    "change": ["דרך", "רק", "אמת", "ל"],  # Yashar!, Beyachad, Democrats, Liberman
    "utj": ["ג"],
    "arab": ["עם", "ודם", "ד"],
}
FROM = list(GROUPS_25) + ["other", "abstained"]
TO = list(GROUPS_26) + ["other", "abstained"]
META = ["סמל ועדה", "ברזל", "שם ישוב", "סמל ישוב", "קלפי", "ריכוז", "שופט",
        "בזב", "מצביעים", "פסולים", "כשרים", "חיצוניות", "קלפי אחרת"]


def load_groups(knesset: int, groups: dict) -> pd.DataFrame:
    """Votes per group per polling place (sub-boxes N.k summed into N)."""
    df = pd.read_csv(ROOT / f"data/raw/results_{knesset}.csv")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    parties = [c for c in df.columns if c not in META]
    missing = {l for ls in groups.values() for l in ls} - set(parties)
    assert not missing, f"letters not in results_{knesset}: {missing}"
    out = pd.DataFrame({g: df[ls].sum(axis=1) for g, ls in groups.items()})
    out["other"] = df["כשרים"] - out.sum(axis=1)
    out["abstained"] = (df["בזב"] - df["כשרים"]).clip(lower=0)
    out["can_vote"] = df["בזב"]
    out["box_id"] = (df["סמל ישוב"].astype(str) + "_"
                     + df["קלפי"].astype(float).astype(int).astype(str))
    return out.groupby("box_id").sum()


def aligned(max_ratio: float = 4 / 3):
    """Stations present in both elections with a similar electorate size.

    Box numbers are reused across elections; a big change in registered
    voters means a different place, so such pairs are dropped.
    """
    a, b = load_groups(25, GROUPS_25), load_groups(26, GROUPS_26)
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    r = b["can_vote"] / a["can_vote"]
    keep = (r < max_ratio) & (r > 1 / max_ratio) & (a["can_vote"] > 0)
    return a.loc[keep, FROM], b.loc[keep, TO]


def build_model(x1: np.ndarray, x2: np.ndarray, phi: float = 100.0, frm=FROM, to=TO) -> pm.Model:
    K1, K2 = len(frm), len(to)
    loyal = np.zeros((K2, K1))
    for j, g in enumerate(frm):
        if g in to:
            loyal[to.index(g), j] = 1.0
    p1 = x1 / x1.sum(axis=1, keepdims=True)
    with pm.Model() as model:
        Z = pm.Normal("Z", 0.0, 1.0, shape=(K2, K1))
        diag_bias = pm.Normal("diag_bias", 2.0, 0.5)
        # columns = from, rows = to; each column sums to 1
        M = pm.Deterministic("M", pm.math.softmax(Z + diag_bias * loyal, axis=0))
        q = pm.math.dot(p1, M.T)
        pm.DirichletMultinomial("x2", n=x2.sum(axis=1), a=phi * q, observed=x2)
    return model


def main(*, draws: int = 1000, tune: int = 1000, chains: int = 4):
    """Fit the 2022->2026 model and save posterior + aligned data."""
    x1, x2 = aligned()
    print(f"{len(x1)} aligned stations, "
          f"{x1['abstained'].sum() + x1.drop(columns='abstained').sum().sum():,.0f} "
          "registered voters (2022)")
    OUT.mkdir(parents=True, exist_ok=True)
    x1.to_parquet(OUT / "x1.parquet")
    x2.to_parquet(OUT / "x2.parquet")
    with build_model(x1.values, x2.values):
        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=0.9, random_seed=26)
    idata.to_netcdf(OUT / "trace.nc")
    print(az.summary(idata, var_names=["diag_bias"]))
    print("max r_hat", float(az.rhat(idata, var_names=["M"])["M"].max()))


if __name__ == "__main__":
    defopt.run(main)

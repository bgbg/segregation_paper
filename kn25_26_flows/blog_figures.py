"""Extra figures for the he.gorelik.net post comparing with LittleMoiz.

Run after flows.py (conda env `bayesian`):
    python kn25_26_flows/blog_figures.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "kn25_26_flows")]

import arviz as az  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

from flows import FROM, GROUPS_25, GROUPS_26, OUT, TO, load_groups  # noqa: E402
from src.plot_utils import heb  # noqa: E402

plt.rcParams["font.family"] = "Arial"
BLUE, ORANGE, GREY = "#1f4e9c", "#e08a00", "#999999"

H25 = {"likud": "ליכוד", "shas": "ש״ס", "rz_otzma": "צ״ד ועוצמה", "bayit_yehudi": "הבית היהודי",
       "national_unity": "המחנה הממלכתי", "change": "גוש השינוי", "utj": "יהדות התורה",
       "arab": "רשימות ערביות", "other": "רשימות קטנות", "abstained": "לא הצביעו"}
H26 = {"likud": "ליכוד", "shas": "ש״ס", "rz_otzma": "צ״ד ועוצמה", "winter": "וינטר", "hendel": "הנדל וזליכה",
       "change": "גוש השינוי", "utj": "יהדות התורה", "arab": "רשימות ערביות", "other": "רשימות קטנות",
       "abstained": "לא הצביעו"}

M = az.from_netcdf(OUT / "trace.nc").posterior["M"].stack(s=("chain", "draw")).values
n25 = load_groups(25, GROUPS_25)[FROM].sum()
flows = M * n25.values[None, :, None] / 1e3  # thousands, to x from x samples


def flow(to, frm):
    return sum(flows[TO.index(to), FROM.index(f)] for f in frm)


def clean(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)


# 1. LittleMoiz vs this model, per flow
pairs = [
    ("ש״ס ← ליכוד", 34, flow("likud", ["shas"])),
    ("ש״ס ← צ״ד ועוצמה", 49, flow("rz_otzma", ["shas"])),
    ("המחנה הממלכתי ← הנדל", 32, flow("hendel", ["national_unity"])),
    ("לא הצביעו ← ליכוד", 53, flow("likud", ["abstained", "other"])),
    ("לא הצביעו ← צ״ד ועוצמה", 16, flow("rz_otzma", ["abstained", "other"])),
    ("לא הצביעו ← וינטר", 13, flow("winter", ["abstained", "other"])),
    ("לא הצביעו ← הנדל", 67, flow("hendel", ["abstained", "other"])),
    ("לא הצביעו ← גוש השינוי", 140, flow("change", ["abstained", "other"])),
    ("הבית היהודי ← וינטר", 13, flow("winter", ["bayit_yehudi"])),
    ("צ״ד ועוצמה ← וינטר", 37, flow("winter", ["rz_otzma"])),
    ("צ״ד ועוצמה ← הנדל", 22, flow("hendel", ["rz_otzma"])),
    ("צ״ד ועוצמה ← גוש השינוי", 21, flow("change", ["rz_otzma"])),
    ("צ״ד ועוצמה ← ליכוד", 28, flow("likud", ["rz_otzma"])),
    ("ליכוד ← צ״ד ועוצמה", 91, flow("rz_otzma", ["likud"])),
]
fig, ax = plt.subplots(figsize=(8, 7))
for i, (lab, lm, s) in enumerate(pairs):
    lo, mid, hi = np.quantile(s, [0.05, 0.5, 0.95])
    ax.plot([lm, mid], [i, i], color="#dddddd", lw=6, solid_capstyle="round", zorder=1)
    ax.plot([lo, hi], [i, i], color=BLUE, lw=2, zorder=2)
    ax.plot(mid, i, "o", color=BLUE, ms=8, zorder=3)
    ax.plot(lm, i, "D", color=ORANGE, ms=7, zorder=3)
ax.set_yticks(range(len(pairs)), [heb(p[0]) for p in pairs])
ax.axhline(8.5, color=GREY, lw=0.8, ls=":")
ax.text(195, 8.6, heb("למעלה: הימין. כאן אנחנו לא מסכימים"), ha="right", va="bottom", color=GREY, fontsize=10)
ax.set_xlabel(heb("אלפי מצביעים"))
ax.set_xlim(0, 200)
ax.plot([], [], "D", color=ORANGE, label="LittleMoiz")
ax.plot([], [], "o-", color=BLUE, label=heb("המודל שלי (90% אי־ודאות)"))
ax.legend(frameon=False, loc="upper right")
ax.set_title(heb("אותם מעברים, שתי הערכות"), fontsize=14)
clean(ax)
fig.tight_layout()
fig.savefig(OUT / "compare_littlemoiz.png", dpi=150)

# 2. Where did 2026 voters of new/changed lists come from
targets = ["winter", "hendel", "change", "rz_otzma", "likud"]
sources = ["likud", "shas", "rz_otzma", "bayit_yehudi", "national_unity", "change", "abstained"]
colors = {"likud": BLUE, "shas": "#2a2a2a", "rz_otzma": ORANGE, "bayit_yehudi": "#c2a14d",
          "national_unity": "#4a90c2", "change": "#c8102e", "abstained": "#cccccc", "rest": "#eeeeee"}
mean = flows.mean(-1)
fig, ax = plt.subplots(figsize=(8, 4.5))
for i, t in enumerate(targets):
    row = mean[TO.index(t)]
    tot = row.sum()
    parts = [(s, row[FROM.index(s)]) for s in sources]
    parts.append(("rest", tot - sum(v for _, v in parts)))
    left = 0
    for s, v in parts:
        w = v / tot * 100
        ax.barh(i, w, left=left, color=colors[s], edgecolor="white")
        if w >= 8:
            ax.text(left + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=9,
                    color="white" if s in ("likud", "shas", "change") else "black")
        left += w
    ax.text(101, i, heb(f"{tot:,.0f} אלף"), va="center", fontsize=9, color=GREY)
ax.set_yticks(range(len(targets)), [heb(H26[t] + " 2026") for t in targets])
ax.invert_yaxis()
ax.set_xlim(0, 112)
ax.set_xticks([])
handles = [Patch(color=colors[s], label=heb("אחרות" if s == "rest" else H25[s] + " 2022"))
           for s in sources + ["rest"]]
ax.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.02), fontsize=9)
ax.set_title(heb("מאיפה הגיעו המצביעים של 2026 (לפי המודל שלי)"), fontsize=14)
clean(ax)
ax.spines["bottom"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "origins_2026.png", dpi=150, bbox_inches="tight")

# 3. Model-implied 2026 totals vs actual
actual = load_groups(26, GROUPS_26)[TO].sum() / 1e3
show = [t for t in TO if t not in ("abstained", "other")]
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, t in enumerate(show):
    m, a = mean[TO.index(t)].sum(), actual[t]
    ax.plot([m, a], [i, i], color="#dddddd", lw=6, solid_capstyle="round")
    ax.plot(a, i, "s", color="black", ms=7)
    ax.plot(m, i, "o", color=BLUE, ms=8)
    if abs(a - m) > 100:
        ax.text(a + 40, i, heb(f"חסרים {a - m:,.0f} אלף"), va="center", fontsize=9, color="#c8102e")
ax.set_yticks(range(len(show)), [heb(H26[t]) for t in show])
ax.invert_yaxis()
ax.plot([], [], "s", color="black", label=heb("תוצאה בפועל"))
ax.plot([], [], "o", color=BLUE, label=heb("המודל"))
ax.legend(frameon=False, loc="lower right")
ax.set_xlabel(heb("אלפי קולות, 2026"))
ax.set_title(heb("המודל מול התוצאות בפועל של 2026"), fontsize=14)
clean(ax)
fig.tight_layout()
fig.savefig(OUT / "model_vs_actual.png", dpi=150)
print("saved")

# 4. Network view: parties as nodes, flows >= MIN_K thousand as arrows
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

MIN_K = 10
RED = "#c8102e"
# node: (x, y, title, colour); x grows to the right = right-wing bloc
NODES = {
    "shas": (10.0, 6.6, "ש״ס", BLUE),
    "likud": (10.0, 1.0, "הליכוד", BLUE),
    "rz_otzma": (6.4, 3.8, "הציונות הדתית ועוצמה", BLUE),
    "bayit_yehudi": (7.2, 8.1, "הבית היהודי", BLUE),
    "winter": (2.4, 6.6, "וינטר", ORANGE),
    "hendel": (2.4, 1.0, "הנדל וזליכה", ORANGE),
    "change": (-1.6, 3.8, "גוש השינוי", RED),
    "national_unity": (-1.6, -0.2, "המחנה הממלכתי", RED),
}
CURVE = {("likud", "winter"): 0.3, ("national_unity", "winter"): 0.15}  # route around boxes
LABEL_T = {("likud", "winter"): 0.7, ("bayit_yehudi", "winter"): 0.3}  # label position along the arrow
W, H = 2.9, 1.15
mean_m = M.mean(-1)


def subtitle(g):
    if g in FROM and g in TO:
        return f"{mean_m[TO.index(g), FROM.index(g)]:.0%} ממצביעיה נשארו"
    if g in FROM:
        return "רק ב־2022"
    return f"{actual[g] * 1e3:,.0f} קולות, חדשה"


def edge_point(src, dst):
    """Point on the box border of src facing dst."""
    (x0, y0), (x1, y1) = src, dst
    dx, dy = x1 - x0, y1 - y0
    t = min(W / 2 / abs(dx) if dx else np.inf, H / 2 / abs(dy) if dy else np.inf) * 1.08
    return x0 + dx * t, y0 + dy * t


def gray_arrow(p0, p1, v):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=22,
                                 lw=min(1.5 + v / 12, 9), color=GREY, zorder=2))


fig, ax = plt.subplots(figsize=(15, 9.5))
for g, (x, y, title, c) in NODES.items():
    new = g in ("winter", "hendel")
    ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.05,rounding_size=0.15",
                                ec=c, fc="#fff6e8" if new else "white", lw=2.2, ls="--" if new else "-", zorder=3))
    ax.text(x, y + 0.17, heb(title), ha="center", va="center", fontsize=14, weight="bold", color=c, zorder=4)
    ax.text(x, y - 0.25, heb(subtitle(g)), ha="center", va="center", fontsize=10, color="#444444", zorder=4)

edges = [(f, t, mean[TO.index(t), FROM.index(f)]) for f in NODES if f in FROM for t in NODES
         if t in TO and t != f]
for f, t, v in sorted(edges, key=lambda e: -e[2]):
    if v < MIN_K:
        continue
    rad = CURVE.get((f, t), 0.0)
    p0 = np.array(edge_point(NODES[f][:2], NODES[t][:2]))
    p1 = np.array(edge_point(NODES[t][:2], NODES[f][:2]))
    lw = min(1.5 + v / 6, 16)
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle=f"-|>,head_length={0.4 + lw / 25},head_width={0.25 + lw / 40}",
                                 mutation_scale=20, lw=lw, color=NODES[f][3], alpha=0.75, zorder=2))
    d = p1 - p0
    c = (p0 + p1) / 2 + rad * np.array([d[1], -d[0]])  # arc3 control point
    lt = LABEL_T.get((f, t), 0.5)
    mx, my = (1 - lt) ** 2 * p0 + 2 * lt * (1 - lt) * c + lt**2 * p1
    ax.text(mx, my, heb(f"כ־{v:.0f} אלף"), ha="center", va="center", fontsize=11, weight="bold", color=NODES[f][3],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=NODES[f][3], lw=1.2), zorder=5)

# new voters in (from abstained + small lists) and leavers out (to abstained), in grey
for g, (x, y, *_) in NODES.items():
    if g in TO:
        v_in = mean[TO.index(g), FROM.index("abstained")] + mean[TO.index(g), FROM.index("other")]
        if v_in >= MIN_K:
            label = heb(f"+כ־{v_in:.0f} אלף") + "\n" + heb("שלא הצביעו או הצביעו לרשימה קטנה")
            if g == "change":  # below is taken by the Gantz arrow
                gray_arrow((x, y + H / 2 + 0.9), (x, y + H / 2 + 0.08), v_in)
                ax.text(x, y + H / 2 + 1.0, label, ha="center", va="bottom", fontsize=9.5, color="#555555")
            else:
                gray_arrow((x, y - H / 2 - 0.75), (x, y - H / 2 - 0.08), v_in)
                ax.text(x, y - H / 2 - 0.85, label, ha="center", va="top", fontsize=9.5, color="#555555")
    if g in FROM:
        v_out = mean[TO.index("abstained"), FROM.index(g)]
        if v_out >= MIN_K:
            gray_arrow((x, y + H / 2 + 0.08), (x, y + H / 2 + 0.75), v_out)
            ax.text(x + 0.2, y + H / 2 + 0.45, heb(f"כ־{v_out:.0f} אלף") + "\n" + heb("לא הצביעו ב־2026"),
                    ha="left", va="center", fontsize=9.5, color="#555555")

for x, lab, c in [(10.0, "גוש הימין", BLUE), (4.4, "באמצע: קיבלו משני הצדדים", "black"), (-1.6, "גוש השינוי", RED)]:
    ax.text(x, 9.3, heb(lab), ha="center", fontsize=13, weight="bold", color=c)
ax.set_xlim(-3.3, 11.7)
ax.set_ylim(-1.9, 9.6)
ax.axis("off")
ax.set_title(heb("איך זזו הקולות, 2022 עד 2026: המודל שלי"), fontsize=18, weight="bold", loc="right")
fig.text(0.98, 0.02, heb(f"הערכה לפי {4117:,} קלפיות שהוצמדו בין 2022 ל־2026. מוצגים מעברים של {MIN_K} אלף ומעלה. "
                         "יהדות התורה והרשימות הערביות לא מוצגות.") + "\n" + heb("בוריס גורליק, he.gorelik.net"),
         ha="right", fontsize=9, color="#666666")
fig.savefig(OUT / "network_flows.png", dpi=150, bbox_inches="tight", facecolor="white")
print("saved network")

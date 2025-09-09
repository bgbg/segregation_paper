#!/usr/bin/env python3
"""
Script to visualize all transition pairs in a 4x4 grid for country-wide data.
Run this in your notebook or as a standalone script.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bidi.algorithm import get_display
import os
from glob import glob

# Color definitions
COLOR_SHAS = "#FFD700"  # gold, as in Shas logo
COLOR_AGUDA = "#1B3161"  # blue, as in Agudat Israel logo
COLOR_OTHER = "lightgrey"
COLOR_ABSTAINED = "darkgrey"

# Hebrew category mapping
heb_category_from_eng = {
    "Shas": "ש״ס",
    "Agudat_Israel": "אגודת ישראל",
    "Other": "מפלגות אחרות",
    "Abstained": "נמנעו",
}


def get_transition_string(from_category, to_category, heb=True):
    h_from = heb_category_from_eng[from_category]
    h_to = heb_category_from_eng[to_category]
    # use proper Hebrew maqaf (U+05BE: ־)
    ret = f"מ־{h_from} ל־{h_to}"
    if heb:
        ret = get_display(ret)
    return ret


def plot_transition_over_time(
    df,
    from_category,
    to_category,
    col_x="kn_location",
    col_y="estimate",
    col_y_lower="lower_ci",
    col_y_upper="upper_ci",
    color="C0",
    title_prefix="",
    ax=None,
):
    def set_xlabel(label=get_display("כנסת מספר"), ax=None):
        if ax is None:
            ax = plt.gca()
        ax.set_xlabel(label, rotation=0, ha="right", va="top", x=1)

    df_sel = (
        df[(df.from_category == from_category) & (df.to_category == to_category)]
        .sort_values(col_x)
        .reset_index(drop=True)
    )

    if len(df_sel) == 0:
        return ax

    df_sel[col_y] *= 100
    df_sel[col_y_lower] *= 100
    df_sel[col_y_upper] *= 100

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(df_sel[col_x], df_sel[col_y], "-o", color=color)
    ax.fill_between(
        df_sel[col_x],
        df_sel[col_y_lower],
        df_sel[col_y_upper],
        color=color,
        alpha=0.2,
        zorder=-1,
    )

    ax.set_ylim(-2, 102)
    tks = [0, 100] + [round(min(df_sel[col_y]), 0), round(max(df_sel[col_y]), 0)]
    tks = list(sorted({int(t) for t in tks}))
    ax.set_yticks(tks)
    ax.set_yticklabels([f"{t}%" for t in tks])

    x_min = np.floor(df_sel[col_x].min()).astype(int)
    x_max = np.ceil(df_sel[col_x].max()).astype(int)
    ax.set_xticks(np.arange(x_min, x_max + 1))

    ttl = get_transition_string(from_category, to_category)
    if title_prefix:
        ttl = f"{title_prefix}\n{ttl}"
    ax.set_title(ttl)
    set_xlabel()
    sns.despine(ax=ax)

    return ax


def load_transition_data():
    """Load all transition data from the processed directory."""
    dir_transitions = "/Users/boris/devel/jce/segregation/data/processed/transitions"
    transition_directories = glob(dir_transitions + "/kn*_*")
    data = []

    for transition_directory in transition_directories:
        toks = os.path.split(transition_directory)[-1].replace("kn", "").split("_")
        assert len(toks) == 2
        kn_from = int(toks[0])
        kn_to = int(toks[1])
        df_country_transition_curr = pd.read_csv(
            os.path.join(transition_directory, "country_map.csv")
        )
        df_country_transition_curr["kn_from"] = kn_from
        df_country_transition_curr["kn_to"] = kn_to
        df_country_transition_curr["kn_location"] = (kn_to + kn_from) / 2
        data.append(df_country_transition_curr)

    return pd.concat(data)


def create_transition_grid(df_country_transition):
    """Create 4x4 subplot grid for all transition pairs."""
    # Create 4x4 subplot grid for all transition pairs
    fig, axes = plt.subplots(
        nrows=4, ncols=4, figsize=(16, 16), sharex=True, sharey=True
    )

    # Define categories and colors
    categories = ["Shas", "Agudat_Israel", "Other", "Abstained"]
    colors = [COLOR_SHAS, COLOR_AGUDA, COLOR_OTHER, COLOR_ABSTAINED]

    # Plot each transition pair
    for i, from_cat in enumerate(categories):
        for j, to_cat in enumerate(categories):
            ax = axes[i, j]

            # Get color for this transition
            if from_cat == to_cat:
                # Diagonal elements (loyalty) use the party's own color
                color = colors[i] if i < 3 else COLOR_ABSTAINED
            else:
                # Off-diagonal elements use a neutral color
                color = "gray"

            # Plot the transition
            plot_transition_over_time(
                df=df_country_transition,
                from_category=from_cat,
                to_category=to_cat,
                ax=ax,
                color=color,
                title_prefix="",
            )

            # Remove title for cleaner look in grid
            ax.set_title("")

            # Add category labels on axes
            if j == 0:  # Leftmost column
                ax.set_ylabel(
                    get_display(heb_category_from_eng[from_cat]),
                    rotation=0,
                    ha="right",
                    va="center",
                )
            if i == 3:  # Bottom row
                ax.set_xlabel(
                    get_display(heb_category_from_eng[to_cat]),
                    rotation=45,
                    ha="right",
                    va="top",
                )

    # Add overall title
    fig.suptitle(get_display("מעברים בין בחירות - רמה ארצית"), fontsize=16, y=0.95)

    # Adjust layout
    plt.tight_layout()
    plt.show()

    return fig


if __name__ == "__main__":
    # Load data
    df_country_transition = load_transition_data()

    # Create visualization
    fig = create_transition_grid(df_country_transition)

import os
import sys
import numpy as np
import pandas as pd
from cachier import cachier
from tqdm.auto import tqdm
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from bidi.algorithm import get_display

sys.path.append("..")
from src import vote_utils as vu

# Directory and file paths
dir_data = "../data/"
dir_external = os.path.abspath(os.path.join(dir_data, "external"))


# Utility functions
def load_data(knesset_number: int):
    fn_in = os.path.join(dir_external, f"results_{knesset_number}.csv")
    fn_columns = os.path.join(dir_external, "columns.csv")

    sr_columns = pd.read_csv(fn_columns).set_index("heb")["eng"].to_dict()
    dtypes = {
        c: "str" if "code" in c or "name" in c else "int" for c in sr_columns.values()
    }
    df = pd.read_csv(fn_in, dtype=dtypes)
    df.columns = [c.strip() for c in df.columns]
    for heb, eng in sr_columns.items():
        if heb in df.columns:
            df.rename(columns={heb: eng}, inplace=True)
        elif eng.startswith("party"):
            df[eng] = 0
        else:
            raise ValueError(
                f"Column '{heb}' not found in data for knesset number {knesset_number}"
            )

    # normalize city codes
    if df.city_code.isna().any():
        codes = []
        for _, row in df.iterrows():
            if pd.isnull(row.city_code):
                city_name = row.city_name
                hsh = abs(hash(city_name)) % 10_000 + 99_000_000
                codes.append(hsh)
            else:
                codes.append(int(row.city_code))
        df["city_code"] = codes
    df.city_code = df.city_code.astype(str)
    df = df[sr_columns.values()]
    df["knesset_number"] = knesset_number

    return df


@cachier(stale_after=timedelta(days=60))
def simulate_dissimilarity_exceedance(
    n_party_a,
    n_party_b,
    n_party_rest,
    ballot_sizes,
    actual_dissimilarity,
    n_simulations=100,
    return_val="fraction",
):
    total_votes = n_party_a + n_party_b + n_party_rest
    assert (
        sum(ballot_sizes) == total_votes
    ), "Ballot sizes must sum to total number of votes"
    p_party_a, p_party_b, p_party_rest = (
        np.array([n_party_a, n_party_b, n_party_rest]) / total_votes
    )
    p_ballot_box = np.array(ballot_sizes) / total_votes

    exceed_count = 0
    simulation_results = []

    for _ in range(n_simulations):
        simulated_votes = np.random.choice(
            [0, 1, 2], size=total_votes, p=[p_party_a, p_party_b, p_party_rest]
        )
        simulated_boxes = np.random.choice(
            range(len(ballot_sizes)), size=total_votes, p=p_ballot_box
        )

        df_sim = pd.DataFrame({"votes": simulated_votes, "ballot_box": simulated_boxes})
        df_sim_wide = df_sim.pivot_table(
            index="ballot_box", columns="votes", aggfunc="size", fill_value=0
        )
        df_sim_wide.columns = [f"party_{c}" for c in df_sim_wide.columns]
        df_sim_wide["legal"] = df_sim_wide.sum(axis=1)

        sim_dissimilarity = vu.dissimilarity_index(
            df_sim_wide, "party_0", "party_1", "legal"
        )
        simulation_results.append(sim_dissimilarity)
        if sim_dissimilarity > actual_dissimilarity:
            exceed_count += 1

    if return_val == "fraction":
        return exceed_count / n_simulations
    return {
        "fraction": exceed_count / n_simulations,
        "simulation_results": simulation_results,
    }


def analyze_city_data(df, party_a, party_b, control_name=""):
    data = []
    pbar = tqdm(df.groupby(["knesset_number", "city_code"]))
    for (knesset_number, city_code), df_city in pbar:
        city_name = df_city["city_name"].iloc[0]
        pbar.set_description(f"Processing {city_name}, (knesset {knesset_number})")
        if control_name:
            city_name = f"{city_name} ({control_name})"
        n_boxes, n_voters = len(df_city), df_city["can_vote"].sum()
        n_legal = df_city["legal"].sum()
        if control_name:
            # in controls, we already have the filtered out irrelevant boxes
            df_city_relevant = df_city.copy()
        else:
            sel_relevant = vu.is_homogenic(df_city, [party_a, party_b])
            df_city_relevant = df_city[sel_relevant].copy()

        if df_city_relevant["legal"].sum() < 500 or len(df_city_relevant) < 5:
            continue

        n_relevant_boxes = len(df_city_relevant)
        n_relevant_voters = df_city_relevant["can_vote"].sum()
        n_relevant_legal = df_city_relevant["legal"].sum()

        # Summing votes for each party
        party_a_votes = df_city_relevant[party_a].sum()
        party_b_votes = df_city_relevant[party_b].sum()

        dissimilarity = vu.dissimilarity_index(
            df_city_relevant, party_a, party_b, "legal"
        )
        n_party_rest = df_city_relevant["legal"].sum() - party_a_votes - party_b_votes
        simulation = simulate_dissimilarity_exceedance(
            n_party_a=party_a_votes,
            n_party_b=party_b_votes,
            n_party_rest=n_party_rest,
            ballot_sizes=df_city_relevant["legal"].values,
            actual_dissimilarity=dissimilarity,
            return_val="detailed",
        )

        data.append(
            {
                "knesset_number": knesset_number,
                "city_code": (
                    f"{city_code} ({control_name})" if control_name else city_code
                ),
                "city_name": city_name,
                # counts
                "n_boxes": n_boxes,
                "n_voters": n_voters,
                "n_legal": n_legal,
                # counts of relevant boxes
                "n_relevant_boxes": n_relevant_boxes,
                "n_relevant_voters": n_relevant_voters,
                "n_relevant_legal": n_relevant_legal,
                # more data
                "city_ref_ratio": party_a_votes / party_b_votes,
                "city_ratios": (
                    df_city_relevant[party_a] / df_city_relevant[party_b]
                ).values,
                "dissimilarity": dissimilarity,
                "dissimilarity_simulation": simulation["simulation_results"],
                "simulation_fraction": simulation["fraction"],
            }
        )

    return (
        pd.DataFrame(data)
        .sort_values("dissimilarity", ascending=False)
        .reset_index(drop=True)
    )


def plot_dissimilarity_latest_point(
    df_data: pd.DataFrame,
    multiyear: bool = None,
    ax=None,
    main_color="C0",
    control_color="C3",
):
    # Determine if analysis is multiyear based on unique knesset numbers
    multiyear = (
        multiyear if multiyear is not None else len(df_data.knesset_number.unique()) > 1
    )
    if multiyear:
        df_knesset_ranges = df_data.groupby("city_name")["knesset_number"].agg(
            ["min", "max"]
        )
    else:
        df_knesset_ranges = None

    # Set up the plotting axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 9), dpi=120)

    # Initialize y-axis ticks and labels
    yticks, y_ticklabels = [], []

    first_time = True
    for _, row in df_data.iterrows():
        yticks.append(row.y)
        curr_city = row.city_name
        simulations = row["dissimilarity_simulation"]

        # Determine color based on city_name content
        color = control_color if "control" in curr_city.lower() else main_color

        # Plot main dissimilarity point with color
        ax.plot(
            row.dissimilarity,
            row.y,
            "o",
            color=color,
            label="Actual dissimilarity" if first_time else None,
        )

        # Calculate percentile range and plot confidence intervals
        p5, p25, p50, p75, p95 = np.percentile(simulations, [5, 25, 50, 75, 95])
        ax.plot(
            [p5, p95],
            [row.y, row.y],
            color="gray",
            lw=1,
            label="Control (95% CI)" if first_time else None,
        )
        ax.plot([p25, p75], [row.y, row.y], color="gray", lw=4)

        # Configure y-axis label for multiyear data
        if multiyear:
            min_knesset = df_knesset_ranges.loc[curr_city, "min"]
            max_knesset = df_knesset_ranges.loc[curr_city, "max"]

            if row.knesset_number == min_knesset:
                lbl = f"{curr_city} ({row.knesset_number})"
            elif row.knesset_number == max_knesset:
                lbl = f"({row.knesset_number})"
            else:
                lbl = ""

            # Draw horizontal lines for first and last knesset appearances
            dy_city = 0.1
            if row.knesset_number == min_knesset:
                ax.axhline(row.y - dy_city, color="gray", lw=0.1, zorder=-1)
            if row.knesset_number == max_knesset:
                ax.axhline(row.y + dy_city, color="gray", lw=0.1, zorder=-1)

            y_ticklabels.append(lbl)
        else:
            y_ticklabels.append(get_display(curr_city))
        first_time = False
    ax.legend()

    # Set y-axis ticks and labels
    ax.set_yticks(yticks)
    ax.set_yticklabels(y_ticklabels)

    # Customize x-axis ticks and formatting
    xticks = np.round([0, df_data.dissimilarity.min(), df_data.dissimilarity.max()], 2)
    ax.set_xticks(xticks)
    ax.set_xlabel("Dissimilarity index")

    # Styling and return
    ax.grid(False)
    sns.despine(ax=ax, left=True)

    return ax


def get_positive_control(df_boxes):
    positive_control_cities = {"ירושלים"}
    positive_control_knesset_numbers = {25}
    df_control = df_boxes[
        df_boxes.city_name.isin(positive_control_cities)
        & df_boxes.knesset_number.isin(positive_control_knesset_numbers)
    ].copy()

    secular_parties = [
        "party_avoda",
        "party_meretz",
        "party_yesh_atid",
        # "party_israel_beitenu",
    ]

    control_secular = df_control[secular_parties].sum(axis=1)
    control_haredi = df_control[vu.HAREDI_PARTIES].sum(axis=1)
    df_control["control_secular"] = control_secular
    df_control["control_haredi"] = control_haredi
    # only take boxes with at least 1 vote in either group
    sel_take = (control_secular > 0) | (control_haredi > 0)
    ret = df_control.loc[sel_take].copy()
    return ret


def get_df_translations():
    dir_external = os.path.abspath(os.path.join(dir_data, "external"))
    fn_translations = os.path.join(dir_external, "israeli_cities_hebrew_english.csv")
    df_translations = pd.read_csv(fn_translations).set_index("city_name")
    return df_translations


def translate_city_names(df_data):
    df_data = df_data.copy()
    df_translations = get_df_translations()
    df_data["city_name"] = df_translations.loc[
        df_data["city_name"].values, "city_name_english"
    ].values
    return df_data


def load_haredi_demographics():
    fn_demographics = os.path.join(dir_external, "haredi_population_by_city.xlsx")
    df_demographics = (
        pd.read_excel(fn_demographics, skiprows=4)
        .rename(columns={"Unnamed: 1": "city_name"})
        .drop(columns="Unnamed: 0")
        .dropna()
        .rename(
            columns={
                "אוכלוסייה חרדית ביישוב": "n_haredi_population",
                "שיעור החרדים ביישוב": "fraction_haredim",
                "שיעור החרדים מהאוכלוסייה היהודית ביישוב": "fraction_haredim_of_jewish_population",
                "שיעור מכלל החרדים בישראל": "fraction_of_total_haredim_in_israel",
            }
        )
    )  #
    # .drop(
    #     columns=[
    #         "fraction_of_total_haredim_in_israel",
    #         "fraction_haredim_of_jewish_population",
    #     ]
    # )
    df_translations = get_df_translations()
    df_demographics.city_name = df_translations.loc[
        df_demographics.city_name, "city_name_english"
    ].values
    for c in df_demographics.columns:
        if c.startswith("n_"):
            df_demographics[c] = df_demographics[c].astype(int)

    return df_demographics


def prepare_data_for_plot(df_data, knesset_numbers=None):
    if knesset_numbers is None:
        knesset_numbers = df_data.knesset_number.unique()
    df_order = (
        df_data.groupby("city_name")["dissimilarity"]
        .mean()
        .sort_values(ascending=True)
        .to_frame()
        .reset_index()
    )
    df_order["y_city"] = df_order.index
    df_plot = df_data.merge(df_order[["city_name", "y_city"]], on="city_name")
    if len(knesset_numbers) > 1:
        max_knesset = df_data.knesset_number.max()
        dy_knesset = -0.15
        df_plot["y_knesset"] = (df_plot.knesset_number - max_knesset) * dy_knesset
        df_plot["y"] = df_plot["y_city"] + df_plot["y_knesset"]
        df_plot = df_plot.sort_values("y")
    else:
        df_plot["y"] = df_plot["y_city"]
    return df_plot


def main(knesset_numbers=None, add_controls=False):
    if knesset_numbers is None:
        knesset_numbers = [25, 23, 21]
    elif isinstance(knesset_numbers, int):
        knesset_numbers = [knesset_numbers]

    df_ballot_boxes = pd.concat(
        [load_data(knesset_number) for knesset_number in knesset_numbers],
    )
    df_data = analyze_city_data(
        df_ballot_boxes,
        "party_shas",
        "party_agudat_israel",
    )
    if add_controls:
        df_data = pd.concat(
            [
                df_data,
                analyze_city_data(
                    get_positive_control(df_ballot_boxes),
                    "control_secular",
                    "control_haredi",
                    "positive control",
                ),
            ],
            ignore_index=True,
        )
    df_data = translate_city_names(df_data)
    df_data = prepare_data_for_plot(df_data)
    df_demographics = load_haredi_demographics()
    df_data = df_data.merge(df_demographics, on="city_name")

    ax = plot_dissimilarity_latest_point(
        df_data.loc[df_data.knesset_number == df_data.knesset_number.max()]
    )
    fig_latest_point = ax.get_figure()

    return {
        "df_data_boxes": df_ballot_boxes,
        "df_data": df_data,
        "fig_latest_point": fig_latest_point,
    }


if __name__ == "__main__":
    res = main(
        knesset_numbers=list(range(15, 26)),
        add_controls=False,
    )
    df = res["df_data"]
    print(df.knesset_number.value_counts(dropna=False))
    plt.show()

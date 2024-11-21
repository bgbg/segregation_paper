from typing import Literal, Union

import pandas as pd
import scipy.stats as stats
from scipy.stats import mannwhitneyu, chi2_contingency, wilcoxon

# Constants
HAREDI_PARTIES = ["party_shas", "party_agudat_israel"]
HOMOGENIC_FRACTION_CUTOFF = 0.75


# Functions
def party_fraction(df: pd.DataFrame, parties, col_ref="legal") -> pd.Series:
    return df[parties].sum(axis=1) / df[col_ref]


def is_homogenic(
    df: pd.DataFrame,
    parties=HAREDI_PARTIES,
    col_ref="legal",
    cutoff=HOMOGENIC_FRACTION_CUTOFF,
) -> pd.Series:
    """
    Check if a given row (ballot or city) is homogenic according to the given parties.
    """
    return party_fraction(df, parties, col_ref) > cutoff


def dissimilarity_index(
    df: pd.DataFrame, party_a_col: str, party_b_col: str, total_col: str
) -> float:
    """
    Calculates the Dissimilarity Index between two parties, indicating the extent of segregation between them
    across neighborhoods. The index represents the proportion of one party's votes that would need to be
    relocated to achieve an even distribution matching the overall city composition.

    Args:
        df: DataFrame containing vote counts for neighborhoods.
        party_a_col: Column name for party A's vote count.
        party_b_col: Column name for party B's vote count.
        total_col: Column name for the total vote count in each neighborhood.

    Returns:
        Dissimilarity Index as a float between 0 and 1, where 0 indicates complete integration
        and 1 indicates complete segregation.
    """
    total_votes = df[total_col].sum()
    fraction_a_total = df[party_a_col].sum() / total_votes
    fraction_b_total = df[party_b_col].sum() / total_votes

    dissimilarity_sum = (
        0.5 * abs((df[party_a_col] / df[total_col]) - fraction_a_total)
    ).sum() + (0.5 * abs((df[party_b_col] / df[total_col]) - fraction_b_total)).sum()

    return dissimilarity_sum / df.shape[0]


def duncan_2_dissimilarity_index(
    df: pd.DataFrame, party_a_col: str, party_b_col: str, ignored
) -> float:
    """
    Calculates the Duncan & Duncan Index of Dissimilarity between two groups.

    The Index of Dissimilarity (D) is defined as:
        D = 0.5 * Σ |(A_i / A) - (B_i / B)|

    Where:
        - A_i and B_i are the populations of groups A and B in subarea i
        - A and B are the total populations of groups A and B in the larger area
        - Σ denotes the summation over all subareas

    Reference:
        Duncan, O. D., & Duncan, B. (1955). A Methodological Analysis of Segregation Indexes.
        American Sociological Review, 20(2), 210–217. doi:10.2307/2088328

    Args:
        df: DataFrame containing population counts for subareas.
        party_a_col: Column name for group A's population count.
        party_b_col: Column name for group B's population count.

    Returns:
        Dissimilarity Index as a float between 0 and 1.
    """
    # Total population for each group
    total_a = df[party_a_col].sum()
    total_b = df[party_b_col].sum()

    # Calculate the dissimilarity index
    dissimilarity_sum = 0.5 * sum(
        abs(
            (df[party_a_col].values[i] / total_a)
            - (df[party_b_col].values[i] / total_b)
        )
        for i in range(len(df))
    )

    return dissimilarity_sum


def gini_coefficient(
    df: pd.DataFrame, party_a_col: str, party_b_col: str, total_col: str
) -> float:
    """
    Calculates the Gini Coefficient for two parties, measuring the inequality in their distribution
    across neighborhoods. Higher values indicate greater segregation.

    Args:
        df: DataFrame containing vote counts for neighborhoods.
        party_a_col: Column name for party A's vote count.
        party_b_col: Column name for party B's vote count.
        total_col: Column name for the total vote count in each neighborhood.

    Returns:
        Gini Coefficient as a float between 0 and 1, where 0 indicates no inequality in distribution
        and 1 indicates maximum inequality.
    """
    total_votes = df[total_col].sum()
    fraction_a_total = df[party_a_col].sum() / total_votes
    fraction_b_total = df[party_b_col].sum() / total_votes

    gini_sum = sum(
        abs(
            (df[party_a_col] / df[total_col]).iloc[i] * df[total_col].iloc[i]
            - (df[party_b_col] / df[total_col]).iloc[j] * df[total_col].iloc[j]
        )
        for i in range(df.shape[0])
        for j in range(df.shape[0])
    )

    return gini_sum / (2 * total_votes**2 * fraction_a_total * fraction_b_total)


def isolation_index(
    df: pd.DataFrame, party_a_col: str, party_b_col: str, total_col: str
) -> float:
    """
    Calculates the Isolation Index for a given party, measuring the likelihood that a member of
    the party shares a neighborhood with other members of the same party.

    Args:
        df: DataFrame containing vote counts for neighborhoods.
        party_a_col: Column name for party A's vote count.
        party_b_col: Column name for party B's vote count.
        total_col: Column name for the total vote count in each neighborhood.

    Returns:
        Isolation Index as a float between 0 and 1, where 0 indicates no isolation and 1 indicates
        complete isolation.
    """
    total_votes = df[total_col].sum()
    fraction_a_total = df[party_a_col].sum() / total_votes

    isolation_sum = (
        (df[party_a_col] / df[total_col]) * (df[party_a_col] / total_votes)
    ).sum()
    return isolation_sum / fraction_a_total


def interaction_index(
    df: pd.DataFrame, party_a_col: str, party_b_col: str, total_col: str
) -> float:
    """
    Calculates the Interaction Index for a given party, measuring the likelihood that a member of
    the party interacts with a member of the other party within neighborhoods.

    Args:
        df: DataFrame containing vote counts for neighborhoods.
        party_a_col: Column name for party A's vote count.
        party_b_col: Column name for party B's vote count.
        total_col: Column name for the total vote count in each neighborhood.

    Returns:
        Interaction Index as a float between 0 and 1, where 0 indicates no interaction
        and 1 indicates complete interaction.
    """
    total_votes = df[total_col].sum()
    fraction_a_total = df[party_a_col].sum() / total_votes

    interaction_sum = (
        (df[party_a_col] / df[total_col]) * (df[party_b_col] / total_votes)
    ).sum()
    return interaction_sum / fraction_a_total


def coleman_index(
    df: pd.DataFrame, party_a_col: str, party_b_col: str, total_col: str
) -> float:
    """
    Calculates Coleman's Segregation Index, comparing actual group concentration to a random distribution.

    Args:
        df: DataFrame containing vote counts for neighborhoods.
        party_a_col: Column name for party A's vote count.
        party_b_col: Column name for party B's vote count.
        total_col: Column name for the total vote count in each neighborhood.

    Returns:
        Coleman's Segregation Index as a float, where positive values indicate segregation and
        negative values suggest inverse segregation.
    """
    total_votes = df[total_col].sum()
    fraction_a_total = df[party_a_col].sum() / total_votes

    coleman_sum = (
        (df[party_a_col] / df[total_col] - fraction_a_total) * df[total_col]
    ).sum()
    return coleman_sum / (total_votes * (1 - fraction_a_total))


def mann_whitney_party_comparison(
    df: pd.DataFrame,
    party_a_col: str,
    party_b_col: str,
    total_col: str,
    ret_value: Literal["u", "p", "both"] = "p",
) -> Union[float, dict]:
    """
    Performs a Mann-Whitney U test to compare the distributions of vote proportions for two
    parties across neighborhoods, without assuming normality.

    Args:
        df: DataFrame containing vote counts for neighborhoods.
        party_a_col: Column name for party A's vote count.
        party_b_col: Column name for party B's vote count.
        total_col: Column name for the total vote count in each neighborhood.
        ret_value: Specifies which result to return: 'u' for U statistic, 'p' for p-value, or
                   'both' for both values as a dictionary.

    Returns:
        A float or dictionary based on `ret_value`. If 'u', returns U statistic. If 'p',
        returns p-value. If 'both', returns {'U_statistic': U statistic, 'p_value': p-value}.
    """
    # Calculate vote proportions for each party in each neighborhood
    party_a_fraction = df[party_a_col] / df[total_col]
    party_b_fraction = df[party_b_col] / df[total_col]

    # Perform Mann-Whitney U test
    u_statistic, p_value = mannwhitneyu(
        party_a_fraction, party_b_fraction, alternative="two-sided"
    )

    if ret_value == "u":
        return u_statistic
    elif ret_value == "p":
        return p_value
    else:
        return {"U_statistic": u_statistic, "p_value": p_value}


def chi_squared_test_votes(
    df: pd.DataFrame,
    ballot_col: str,
    party_a_col: str,
    party_b_col: str,
    ret_val: Literal["p", "chi2", "both"] = "p",
) -> Union[float, dict]:
    """
    Conducts a chi-squared test of independence to determine whether the proportion of votes
    for Party A versus Party B is significantly associated with the ballot box.

    Args:
        df: DataFrame containing vote counts for neighborhoods.
        ballot_col: Column name for the ballot box identifiers.
        party_a_col: Column name for Party A's vote count.
        party_b_col: Column name for Party B's vote count.
        ret_val: Specifies which result to return: 'p' for p-value, 'chi2' for chi-squared statistic,
                 or 'both' for both values in a dictionary.

    Returns:
        A float or dictionary based on `ret_val`. If 'p', returns p-value. If 'chi2', returns
        chi-squared statistic. If 'both', returns {'chi2_statistic': chi2_statistic, 'p_value': p_value}.
    """
    # Create a contingency table with vote counts for each party by ballot box
    contingency_table = df.pivot_table(
        index=ballot_col, values=[party_a_col, party_b_col], aggfunc="sum"
    ).fillna(0) + 1 / len(df)
    # Conduct chi-squared test of independence
    chi2_statistic, p_value, degrees_of_freedom, expected_freq = chi2_contingency(
        contingency_table
    )

    # Return based on ret_val
    if ret_val == "p":
        return p_value
    elif ret_val == "chi2":
        return chi2_statistic
    else:
        return {"chi2_statistic": chi2_statistic, "p_value": p_value}


def test_party_ratio_deviation(
    df: pd.DataFrame,
    party_a_col: str,
    party_b_col: str,
    ret_val="p",
):
    raise NotImplementedError(
        "This computation is invalid in the context of this project"
    )
    total_a = df[party_a_col].sum()
    total_b = df[party_b_col].sum()
    city_ratio = total_a / total_b

    # Calculate the individual ratios for each ballot box
    individual_ratios = df[party_a_col] / df[party_b_col]

    # Perform the Wilcoxon Signed-Rank Test
    test_statistic, p_value = wilcoxon(individual_ratios - city_ratio)

    # Return either p-value or test statistic as specified
    if ret_val == "p":
        return p_value
    elif ret_val == "stat":
        return test_statistic
    else:
        return {"p_value": p_value, "test_statistic": test_statistic}


DISIMILARITY_FUNCTIONS = {
    "Dissimilarity Index": dissimilarity_index,
    "Gini Coefficient": gini_coefficient,
    "Isolation Index": isolation_index,
    "Interaction Index": interaction_index,
    "Coleman Index": coleman_index,
    "Mann-Whitney U Test": mann_whitney_party_comparison,
}

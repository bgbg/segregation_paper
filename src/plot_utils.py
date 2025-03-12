"""
This module provides utility functions for plotting, including handling bidirectional text,
removing spines, and disabling grids on matplotlib axes.
"""

from bidi.algorithm import get_display


def hep(s):
    """
    Convert a string to its bidirectional display form.

    Parameters:
    s (str): The string to convert.

    Returns:
    str: The bidirectional display form of the string.
    """
    # Example usage:
    # hep("שלום") -> "םולש"


def despine_and_remove_grid(ax):
    """
    Remove the top, right, left, and bottom spines and disable the grid on both axes.

    Parameters:
    ax (matplotlib.axes.Axes): The axes object to modify.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.yaxis.grid(False)
    # Example usage:
    # fig, ax = plt.subplots()
    # ax.plot([0, 1], [0, 1])
    # despine_and_remove_grid(ax)
    # plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    despine_and_remove_grid(ax)
    plt.show()

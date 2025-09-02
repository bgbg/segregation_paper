# -*- coding: utf-8 -*-
"""
Sankey diagrams for voter transition analysis using matplotlib.
Accepts pandas DataFrames with vote movements data (actual vote counts).
"""

from collections import defaultdict
from pathlib import Path
from typing import Optional

import defopt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def sankey(
    df,
    col_left="from_category",
    col_right="to_category",
    col_vote_count="vote_count",
    col_alpha=None,
    category_colors=None,
    left_labels=None,
    right_labels=None,
    aspect=4,
    right_color=False,
    font_size=14,
    ax=None,
):
    """
    Make Sankey Diagram from vote movements DataFrame.

    Args:
        df: DataFrame with vote movements data (actual vote counts, not probabilities)
        col_left: Column name for source categories (default: 'from_category')
        col_right: Column name for target categories (default: 'to_category')
        col_vote_count: Column name for vote movements (default: 'vote_count')
        col_alpha: Column name for transparency values (optional)
        category_colors: Dict mapping categories to colors (optional)
        left_labels: Order of left labels in diagram (optional)
        right_labels: Order of right labels in diagram (optional)
        aspect: Vertical extent of diagram in units of horizontal extent
        right_color: If True, color strips by right label instead of left
        font_size: Font size for labels

    Returns:
        Matplotlib axes object with the Sankey diagram
    """
    # Validate required columns
    required_cols = [col_left, col_right, col_vote_count]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for null values
    if df[[col_left, col_right]].isnull().any().any():
        raise ValueError(
            "Sankey graph does not support null values in category columns"
        )

    # Extract data from DataFrame
    left = df[col_left].values
    right = df[col_right].values
    vote_weights = df[col_vote_count].values
    
    # Filter out zero or very small vote movements for cleaner visualization
    non_zero_mask = vote_weights > 0
    left = left[non_zero_mask]
    right = right[non_zero_mask]
    vote_weights = vote_weights[non_zero_mask]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Configure font settings
    plt.rc("text", usetex=False)
    plt.rc("font", family="serif")

    # Get alpha values if specified
    alpha_values = None
    if col_alpha and col_alpha in df.columns:
        alpha_values = df[col_alpha].values[non_zero_mask]

    # Create working DataFrame
    data_frame = pd.DataFrame(
        {
            "left": left,
            "right": right,
            "vote_weight": vote_weights,
        }
    )

    # Add alpha column if provided
    if alpha_values is not None:
        data_frame["alpha"] = alpha_values

    # Identify all unique labels
    all_labels = pd.Series(
        np.r_[data_frame.left.unique(), data_frame.right.unique()]
    ).unique()

    # Set left and right labels
    if left_labels is None:
        left_labels = sorted(data_frame.left.unique())
    if right_labels is None:
        right_labels = sorted(data_frame.right.unique())

    # Set up colors
    if category_colors is None:
        palette = "hls"
        color_palette = sns.color_palette(palette, len(all_labels))
        category_colors = {
            label: color_palette[i] for i, label in enumerate(all_labels)
        }
    else:
        # Fill in missing colors with defaults
        missing_labels = [label for label in all_labels if label not in category_colors]
        if missing_labels:
            default_palette = sns.color_palette("hls", len(missing_labels))
            for i, label in enumerate(missing_labels):
                category_colors[label] = default_palette[i]

    # Calculate strip widths for each label combination
    ns_l = defaultdict(dict)
    ns_r = defaultdict(dict)

    for left_label in left_labels:
        for right_label in right_labels:
            vote_weight_sum = data_frame[
                (data_frame.left == left_label) & (data_frame.right == right_label)
            ].vote_weight.sum()

            # For vote movements, the flow is the same from left to right
            ns_l[left_label][right_label] = vote_weight_sum
            ns_r[left_label][right_label] = vote_weight_sum

    # Calculate positions for left label patches
    left_widths = {}
    for i, left_label in enumerate(left_labels):
        left_total = data_frame[data_frame.left == left_label].vote_weight.sum()

        if i == 0:
            bottom = 0
            top = left_total
        else:
            bottom = (
                left_widths[left_labels[i - 1]]["top"]
                + 0.02 * data_frame.vote_weight.sum()
            )
            top = bottom + left_total

        left_widths[left_label] = {"left": left_total, "bottom": bottom, "top": top}

    # Calculate positions for right label patches
    right_widths = {}
    for i, right_label in enumerate(right_labels):
        right_total = data_frame[data_frame.right == right_label].vote_weight.sum()

        if i == 0:
            bottom = 0
            top = right_total
        else:
            bottom = (
                right_widths[right_labels[i - 1]]["top"]
                + 0.02 * data_frame.vote_weight.sum()
            )
            top = bottom + right_total

        right_widths[right_label] = {"right": right_total, "bottom": bottom, "top": top}

    # Total vertical extent
    top_edge = max(
        [left_widths[label]["top"] for label in left_labels]
        + [right_widths[label]["top"] for label in right_labels]
    )
    x_max = top_edge / aspect

    # Draw left label bars
    for left_label in left_labels:
        ax.fill_between(
            [-0.02 * x_max, 0],
            [left_widths[left_label]["bottom"]] * 2,
            [left_widths[left_label]["top"]] * 2,
            color=category_colors[left_label],
            alpha=0.99,  # Category bars always opaque
        )
        ax.text(
            -0.05 * x_max,
            left_widths[left_label]["bottom"] + 0.5 * left_widths[left_label]["left"],
            left_label,
            ha="right",
            va="center",
            fontsize=font_size,
        )

    # Draw right label bars
    for right_label in right_labels:
        ax.fill_between(
            [x_max, 1.02 * x_max],
            [right_widths[right_label]["bottom"]] * 2,
            [right_widths[right_label]["top"]] * 2,
            color=category_colors[right_label],
            alpha=0.99,  # Category bars always opaque
        )
        ax.text(
            1.05 * x_max,
            right_widths[right_label]["bottom"]
            + 0.5 * right_widths[right_label]["right"],
            right_label,
            ha="left",
            va="center",
            fontsize=font_size,
        )

    # Draw transition strips/chords
    for left_label in left_labels:
        for right_label in right_labels:
            strip_data = data_frame[
                (data_frame.left == left_label) & (data_frame.right == right_label)
            ]

            if len(strip_data) > 0 and ns_l[left_label][right_label] > 0:
                # Determine strip color
                label_color = left_label if not right_color else right_label

                # Get alpha value for this transition
                strip_alpha = 0.65  # default
                if alpha_values is not None and len(strip_data) > 0:
                    strip_alpha = strip_data["alpha"].iloc[0]

                # Create smooth transition curves
                y_bottom = np.array(
                    50 * [left_widths[left_label]["bottom"]]
                    + 50 * [right_widths[right_label]["bottom"]]
                )
                y_top = np.array(
                    50
                    * [
                        left_widths[left_label]["bottom"]
                        + ns_l[left_label][right_label]
                    ]
                    + 50
                    * [
                        right_widths[right_label]["bottom"]
                        + ns_r[left_label][right_label]
                    ]
                )

                # Smooth the curves
                y_bottom = np.convolve(y_bottom, 0.05 * np.ones(20), mode="valid")
                y_bottom = np.convolve(y_bottom, 0.05 * np.ones(20), mode="valid")
                y_top = np.convolve(y_top, 0.05 * np.ones(20), mode="valid")
                y_top = np.convolve(y_top, 0.05 * np.ones(20), mode="valid")

                # Update positions for next strips
                left_widths[left_label]["bottom"] += ns_l[left_label][right_label]
                right_widths[right_label]["bottom"] += ns_r[left_label][right_label]

                # Draw the strip
                ax.fill_between(
                    np.linspace(0, x_max, len(y_bottom)),
                    y_bottom,
                    y_top,
                    alpha=strip_alpha,  # Use variable alpha for chords
                    color=category_colors[label_color],
                )

    ax.axis("off")

    return ax


def main(
    *,
    file_path: str = "/Users/boris/devel/jce/segregation/data/processed/transitions/kn20_21/country_movements.csv",
    output_path: Optional[str] = None,
    title: str = "Voter Movement Flow (Knesset 20→21)",
) -> None:
    """
    Generate and save a Sankey diagram from vote movements data.

    Args:
        file_path: Path to the CSV file containing vote movements data
        output_path: Path where to save the file (PNG). If None, a default path is used.
        title: Title for the diagram
    """
    # Validate input file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Vote movements file not found: {file_path}")
    
    print(f"Loading vote movements data from {file_path}")
    
    # Load CSV data
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        raise ValueError(f"Failed to load CSV file: {e}")
    
    # Validate required columns
    required_columns = ["from_category", "to_category", "vote_count"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"Found categories: {sorted(df['from_category'].unique())}")
    
    # Show total vote movements
    total_votes = df['vote_count'].sum()
    non_zero_movements = (df['vote_count'] > 0).sum()
    print(f"Total vote movements: {total_votes:,.0f} votes across {non_zero_movements} flows")
    
    # Set default output path if not provided
    if output_path is None:
        output_path = "vote_movements_sankey.png"
    
    print(f"Creating Sankey diagram: '{title}'")
    
    # Create sankey diagram
    ax = sankey(df)
    
    # Add title to the diagram
    ax.set_title(title, fontsize=16, pad=20)
    
    # Save the diagram
    print(f"Saving diagram to {output_path}")
    ax.figure.savefig(output_path, bbox_inches="tight", dpi=150)
    
    print("✓ Sankey diagram generated successfully!")


if __name__ == "__main__":
    defopt.run(main)

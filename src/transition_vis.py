#!/usr/bin/env python3
"""
Transition visualization tool for voter transition matrices.

Generates Sankey diagrams from transition matrix CSV files showing voter flow
between categories (Shas, Agudat_Israel, Other, Abstained) across elections.
"""

import logging
from pathlib import Path
from typing import Optional

import defopt
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set choreographer and kaleido logging to warning level
logging.getLogger("choreographer").setLevel(logging.WARNING)
logging.getLogger("kaleido").setLevel(logging.WARNING)


def load_transition_data(file_path: str) -> pd.DataFrame:
    """
    Load transition matrix data from CSV file.

    Args:
        file_path: Path to the CSV file containing transition data

    Returns:
        DataFrame with transition matrix data
    """
    df = pd.read_csv(file_path)
    logger.info(f"Loaded transition data from {file_path}")
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Categories: {df['from_category'].unique()}")

    return df


def create_sankey_diagram(
    df: pd.DataFrame, title: str = "Voter Transition Flow"
) -> go.Figure:
    """
    Create a Sankey diagram from transition matrix data.

    Args:
        df: DataFrame with transition data (from_category, to_category, estimate)
        title: Title for the diagram

    Returns:
        Plotly figure with Sankey diagram
    """
    # Get unique categories
    categories = sorted(df["from_category"].unique())

    # Create node labels (source + target categories)
    node_labels = categories + categories  # Duplicate for source and target

    # Create source and target indices
    source_indices = []
    target_indices = []
    values = []

    # Map categories to indices
    category_to_index = {cat: i for i, cat in enumerate(categories)}

    for _, row in df.iterrows():
        from_cat = row["from_category"]
        to_cat = row["to_category"]
        value = row["estimate"]

        # Source index (first set of nodes)
        source_idx = category_to_index[from_cat]

        # Target index (second set of nodes, offset by number of categories)
        target_idx = category_to_index[to_cat] + len(categories)

        source_indices.append(source_idx)
        target_indices.append(target_idx)
        values.append(value)

    # Create Sankey diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                    color=[
                        "#1f77b4",
                        "#ff7f0e",
                        "#2ca02c",
                        "#d62728",  # Source nodes
                        "#1f77b4",
                        "#ff7f0e",
                        "#2ca02c",
                        "#d62728",  # Target nodes
                    ],
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=values,
                    color=["rgba(0,0,0,0.2)"] * len(values),  # Semi-transparent links
                ),
            )
        ]
    )

    # Update layout
    fig.update_layout(
        title_text=title, font_size=12, font_family="Arial", height=600, width=800
    )

    return fig


def save_diagram(fig: go.Figure, output_path: str) -> None:
    """
    Save the diagram to a file (PNG or HTML).

    Args:
        fig: Plotly figure to save
        output_path: Path where to save the file
    """
    if output_path.lower().endswith((".html", ".htm")):
        fig.write_html(output_path)
        logger.info(f"Diagram saved as HTML: {output_path}")
    else:
        fig.write_image(output_path)
        logger.info(f"Diagram saved as PNG: {output_path}")


def main(
    *,
    file_path: str = "/Users/boris/devel/jce/segregation/data/processed/transitions/kn20_21/country_map.csv",
    output_path: Optional[str] = None,
    title: str = "Voter Transition Flow (Knesset 20→21)",
) -> None:
    """
    Generate and save a Sankey diagram from transition matrix data.

    Args:
        file_path: Path to the CSV file containing transition data
        output_path: Path where to save the file (PNG or HTML)
        title: Title for the diagram
    """
    # Validate input file
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Transition file not found: {file_path}")

    # Load data
    df = load_transition_data(file_path)

    # Validate data structure
    required_columns = ["from_category", "to_category", "estimate"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Create Sankey diagram
    logger.info("Creating Sankey diagram...")
    fig = create_sankey_diagram(df, title)

    # Save diagram based on output path extension
    if output_path:
        save_diagram(fig, output_path)
    else:
        # Default to PNG if no output path specified
        default_path = "transition_sankey.png"
        logger.info(f"No output path specified, saving to {default_path}")
        save_diagram(fig, default_path)

    logger.info("Transition visualization completed successfully!")


if __name__ == "__main__":
    defopt.run(main)

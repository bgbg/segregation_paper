"""Reporting and visualization entrypoints for voter transition analysis.

This package provides a single high-level entrypoint that generates all
visualizations and a Markdown report with well-formatted tables and key
diagnostic plots. It reuses existing visualization functions and I/O utilities.
"""

from .reporting import generate_all_outputs  # noqa: F401

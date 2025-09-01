"""Tests for data harmonization pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_harmonizer import (
    apply_column_mappings,
    apply_homogenic_filtering,
    compute_voter_categories,
    filter_target_cities,
    load_city_mappings,
    load_column_mappings,
)


def test_apply_column_mappings(sample_raw_election_data, column_mappings):
    """Test Hebrew to English column mapping."""
    df_mapped = apply_column_mappings(sample_raw_election_data, column_mappings)

    # Should have English column names
    assert "city_name" in df_mapped.columns
    assert "can_vote" in df_mapped.columns
    assert "party_shas" in df_mapped.columns

    # Hebrew columns should be renamed
    assert "שם ישוב" not in df_mapped.columns
    assert "בזב" not in df_mapped.columns
    assert "שס" not in df_mapped.columns

    # Data should be preserved
    assert len(df_mapped) == len(sample_raw_election_data)


def test_compute_voter_categories(sample_raw_election_data, column_mappings):
    """Test computation of voter categories with corrected abstained calculation."""
    df_mapped = apply_column_mappings(sample_raw_election_data, column_mappings)
    df_categories = compute_voter_categories(df_mapped)

    # Should have all four categories
    expected_categories = ["A_shas", "B_agudat", "Other", "Abstained"]
    for cat in expected_categories:
        assert cat in df_categories.columns, f"Missing category: {cat}"

    # Test corrected abstained calculation: can_vote - legal
    expected_abstained = df_categories["can_vote"] - df_categories["legal"]
    actual_abstained = df_categories["Abstained"]

    # Should match (allowing for non-negative constraint)
    pd.testing.assert_series_equal(
        actual_abstained, np.maximum(0, expected_abstained), check_names=False
    )

    # All categories should be non-negative
    for cat in expected_categories:
        assert (df_categories[cat] >= 0).all(), f"Negative values in {cat}"

    # Shas and Agudat should match original party votes
    assert (df_categories["A_shas"] >= 0).all()
    assert (df_categories["B_agudat"] >= 0).all()


def test_abstained_computation_edge_cases():
    """Test edge cases in abstained vote computation."""
    # Case 1: legal > can_vote (should not happen but handle gracefully)
    df_edge = pd.DataFrame(
        {
            "can_vote": [100, 200, 300],
            "legal": [120, 180, 250],  # legal > can_vote in first case
            "party_shas": [10, 20, 30],
            "party_agudat_israel": [5, 15, 25],
        }
    )

    df_result = compute_voter_categories(df_edge)

    # Abstained should never be negative
    assert (df_result["Abstained"] >= 0).all()

    # Should be zero when legal >= can_vote
    assert df_result.loc[0, "Abstained"] == 0


def test_apply_homogenic_filtering(sample_harmonized_data):
    """Test homogenic filtering based on Haredi vote percentage."""
    # Apply filtering with 75% threshold
    df_filtered = apply_homogenic_filtering(sample_harmonized_data, threshold=0.75)

    # Should have fewer or equal stations
    assert len(df_filtered) <= len(sample_harmonized_data)

    # All remaining stations should have >75% Haredi votes (if any pass the filter)
    if len(df_filtered) > 0:
        haredi_fraction = (
            df_filtered["A_shas"] + df_filtered["B_agudat"]
        ) / df_filtered["legal"]
        assert (
            haredi_fraction > 0.75
        ).all(), "Some stations don't meet homogenic threshold"

    # Test with lower threshold that should include more stations
    df_filtered_low = apply_homogenic_filtering(sample_harmonized_data, threshold=0.4)
    assert len(df_filtered_low) >= len(
        df_filtered
    ), "Lower threshold should include more stations"

    # With very low threshold, should have >40% Haredi votes
    if len(df_filtered_low) > 0:
        haredi_fraction_low = (
            df_filtered_low["A_shas"] + df_filtered_low["B_agudat"]
        ) / df_filtered_low["legal"]
        assert (
            haredi_fraction_low > 0.4
        ).all(), "Some stations don't meet low threshold"


def test_filter_target_cities(sample_harmonized_data, city_mappings):
    """Test filtering for target cities."""
    target_cities = ["jerusalem", "bnei brak"]
    df_filtered = filter_target_cities(
        sample_harmonized_data, target_cities, city_mappings
    )

    # Should have fewer or equal stations
    assert len(df_filtered) <= len(sample_harmonized_data)

    # All remaining stations should be from target cities
    city_names = df_filtered["city_name"].unique()
    expected_hebrew_cities = ["ירושלים", "בני ברק"]  # Hebrew equivalents

    # Check that only target cities remain (allowing for variations in city mapping)
    for city in city_names:
        assert any(
            target in city_mappings.get(city, "").lower()
            or city.lower() in target_cities
            for target in target_cities
        ), f"Unexpected city in filtered data: {city}"


def test_compute_voter_categories_missing_columns():
    """Test graceful handling of missing party columns."""
    df_minimal = pd.DataFrame({"can_vote": [100, 200], "legal": [80, 150]})

    df_result = compute_voter_categories(df_minimal)

    # Should create categories with defaults
    assert "A_shas" in df_result.columns
    assert "B_agudat" in df_result.columns
    assert "Other" in df_result.columns
    assert "Abstained" in df_result.columns

    # Should handle missing party columns gracefully
    assert (df_result["A_shas"] == 0).all()
    assert (df_result["B_agudat"] == 0).all()


def test_homogenic_filtering_missing_columns():
    """Test homogenic filtering with missing required columns."""
    df_incomplete = pd.DataFrame({"can_vote": [100, 200], "legal": [80, 150]})

    # Should return original data when required columns missing
    df_result = apply_homogenic_filtering(df_incomplete, threshold=0.75)
    pd.testing.assert_frame_equal(df_result, df_incomplete)


def test_category_consistency():
    """Test that vote categories sum correctly."""
    df_test = pd.DataFrame(
        {
            "can_vote": [1000, 800, 1200],
            "legal": [900, 750, 1100],
            "party_shas": [200, 150, 300],
            "party_agudat_israel": [150, 100, 250],
        }
    )

    df_result = compute_voter_categories(df_test)

    # Total votes should equal can_vote
    total_computed = (
        df_result["A_shas"]
        + df_result["B_agudat"]
        + df_result["Other"]
        + df_result["Abstained"]
    )

    pd.testing.assert_series_equal(
        total_computed, df_result["can_vote"], check_names=False
    )

    # Legal votes should equal A + B + Other
    legal_computed = df_result["A_shas"] + df_result["B_agudat"] + df_result["Other"]
    pd.testing.assert_series_equal(
        legal_computed, df_result["legal"], check_names=False
    )


def test_load_mappings_file_format():
    """Test loading column and city mappings from CSV format."""
    # Create temporary mapping files
    column_data = pd.DataFrame(
        {
            "heb": ["שס", "ג", "בזב"],
            "eng": ["party_shas", "party_agudat_israel", "can_vote"],
        }
    )

    city_data = pd.DataFrame(
        {
            "city_name": ["ירושלים", "בני ברק"],
            "city_name_english": ["Jerusalem", "Bnei Brak"],
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        column_data.to_csv(f.name, index=False)
        column_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        city_data.to_csv(f.name, index=False)
        city_path = f.name

    try:
        # Test loading
        column_mapping = load_column_mappings(column_path)
        city_mapping = load_city_mappings(city_path)

        # Verify mappings
        assert column_mapping["שס"] == "party_shas"
        assert column_mapping["ג"] == "party_agudat_israel"
        assert (
            "jerusalem" in city_mapping.values() or "Jerusalem" in city_mapping.values()
        )

    finally:
        Path(column_path).unlink(missing_ok=True)
        Path(city_path).unlink(missing_ok=True)

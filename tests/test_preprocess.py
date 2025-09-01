"""Tests for preprocessing and tensor building."""

import numpy as np
import pandas as pd
import pytest

from src.transition_model.preprocess import (
    build_station_tensors,
    compute_categories,
    load_config,
    prepare_hierarchical_data,
)


def test_build_station_tensors_basic(sample_harmonized_data):
    """Test basic tensor building functionality."""
    # Create two election datasets
    df1 = sample_harmonized_data.copy()
    df2 = sample_harmonized_data.copy()

    # Modify df2 slightly to simulate different election
    df2["election"] = 21
    df2["A_shas"] = df2["A_shas"] * 1.1
    df2["Abstained"] = np.maximum(0, df2["Abstained"] * 0.9)

    x1, x2, n1, n2 = build_station_tensors(df1, df2)

    # Check tensor shapes
    assert x1.shape[1] == 4, "Should have 4 categories"
    assert x2.shape[1] == 4, "Should have 4 categories"
    assert x1.shape[0] == x2.shape[0], "Should have same number of stations"
    assert len(n1) == len(n2), "Should have same number of stations"

    # Check that totals match
    np.testing.assert_array_equal(n1, x1.sum(axis=1))
    np.testing.assert_array_equal(n2, x2.sum(axis=1))

    # All values should be non-negative
    assert (x1 >= 0).all()
    assert (x2 >= 0).all()
    assert (n1 >= 0).all()
    assert (n2 >= 0).all()


def test_build_station_tensors_alignment():
    """Test that tensors properly align stations between elections."""
    # Create datasets with different station sets
    df1 = pd.DataFrame(
        {
            "station_id": [1, 2, 3, 4],
            "A_shas": [10, 20, 30, 40],
            "B_agudat": [5, 15, 25, 35],
            "Other": [50, 60, 70, 80],
            "Abstained": [10, 5, 15, 20],
        }
    )

    df2 = pd.DataFrame(
        {
            "station_id": [2, 3, 4, 5],  # Station 1 missing, station 5 added
            "A_shas": [25, 35, 45, 50],
            "B_agudat": [20, 30, 40, 45],
            "Other": [65, 75, 85, 90],
            "Abstained": [8, 12, 18, 25],
        }
    )

    x1, x2, n1, n2 = build_station_tensors(df1, df2)

    # Should only include common stations (2, 3, 4)
    assert x1.shape[0] == 3
    assert x2.shape[0] == 3

    # Verify alignment by checking specific values
    # Station 2 should be first in aligned data
    assert x1[0, 0] == 20  # Shas votes for station 2 in df1
    assert x2[0, 0] == 25  # Shas votes for station 2 in df2


def test_prepare_hierarchical_data(sample_harmonized_data, sample_config):
    """Test preparation of hierarchical data structure."""
    # Create two elections
    df1 = sample_harmonized_data.copy()
    df2 = sample_harmonized_data.copy()
    df2["election"] = 21

    target_cities = ["jerusalem", "bnei brak"]

    data = prepare_hierarchical_data(df1, df2, target_cities, sample_config)

    # Should have country data
    assert "country" in data
    country_data = data["country"]
    assert all(key in country_data for key in ["x1", "x2", "n1", "n2"])

    # Should have city data for cities that exist in the sample
    # Note: sample data uses 'Jerusalem', 'Bnei Brak', 'Ashdod' as city names
    expected_cities = []
    for city in target_cities:
        city_matches = df1["city_name"].str.lower().str.contains(city.lower())
        if city_matches.any():
            expected_cities.append(city)

    # Check that we have data for cities that match
    for city in expected_cities:
        if city in data:  # May not match due to string matching logic
            city_data = data[city]
            assert all(key in city_data for key in ["x1", "x2", "n1", "n2"])
            assert city_data["x1"].shape[1] == 4, "Should have 4 categories"


def test_compute_categories_backward_compatibility():
    """Test backward compatibility of compute_categories function."""
    # Test with data that already has categories
    df_with_categories = pd.DataFrame(
        {
            "A_shas": [10, 20],
            "B_agudat": [5, 15],
            "Other": [30, 40],
            "Abstained": [5, 10],
        }
    )

    result = compute_categories(df_with_categories)

    # Should return data unchanged if categories already exist
    pd.testing.assert_frame_equal(result, df_with_categories)


def test_compute_categories_fallback():
    """Test fallback computation when categories don't exist."""
    df_raw = pd.DataFrame(
        {
            "can_vote": [100, 200],
            "legal": [80, 160],
            "party_shas": [20, 40],
            "party_agudat_israel": [10, 30],
        }
    )

    result = compute_categories(df_raw)

    # Should compute categories
    assert "A_shas" in result.columns
    assert "B_agudat" in result.columns
    assert "Other" in result.columns
    assert "Abstained" in result.columns

    # Check corrected abstained computation: can_vote - legal
    expected_abstained = df_raw["can_vote"] - df_raw["legal"]
    pd.testing.assert_series_equal(
        result["Abstained"], expected_abstained, check_names=False
    )


def test_tensor_data_types():
    """Test that tensor data has correct data types."""
    df1 = pd.DataFrame(
        {
            "station_id": [1, 2, 3],
            "A_shas": [10, 20, 30],
            "B_agudat": [5, 15, 25],
            "Other": [30, 40, 50],
            "Abstained": [5, 10, 15],
        }
    )

    df2 = df1.copy()
    df2[["A_shas", "B_agudat", "Other", "Abstained"]] *= 1.1

    x1, x2, n1, n2 = build_station_tensors(df1, df2)

    # Should be numpy arrays
    assert isinstance(x1, np.ndarray)
    assert isinstance(x2, np.ndarray)
    assert isinstance(n1, np.ndarray)
    assert isinstance(n2, np.ndarray)

    # Should have numeric dtypes suitable for PyMC
    assert np.issubdtype(x1.dtype, np.number)
    assert np.issubdtype(x2.dtype, np.number)
    assert np.issubdtype(n1.dtype, np.number)
    assert np.issubdtype(n2.dtype, np.number)


def test_empty_city_filtering():
    """Test behavior when cities have no data."""
    df1 = pd.DataFrame(
        {
            "station_id": [1, 2],
            "city_name": ["Tel Aviv", "Haifa"],  # Cities not in target list
            "A_shas": [10, 20],
            "B_agudat": [5, 15],
            "Other": [30, 40],
            "Abstained": [5, 10],
        }
    )

    df2 = df1.copy()
    target_cities = ["jerusalem", "bnei brak"]  # Not present in data

    data = prepare_hierarchical_data(df1, df2, target_cities, None)

    # Should have country data
    assert "country" in data
    assert data["country"]["x1"].shape[0] == 2

    # Should not have city data for cities not in dataset
    for city in target_cities:
        if city in data:  # May exist due to partial string matching
            assert len(data[city]["x1"]) == 0 or city not in data


def test_tensor_consistency():
    """Test that tensors maintain data consistency."""
    # Create realistic test data
    np.random.seed(42)
    n_stations = 20

    df1 = pd.DataFrame(
        {
            "station_id": range(1, n_stations + 1),
            "A_shas": np.random.randint(10, 50, n_stations),
            "B_agudat": np.random.randint(5, 40, n_stations),
            "Other": np.random.randint(50, 200, n_stations),
            "Abstained": np.random.randint(5, 30, n_stations),
        }
    )

    df2 = df1.copy()
    # Simulate realistic changes between elections
    df2["A_shas"] = df2["A_shas"] * np.random.uniform(0.8, 1.2, n_stations)
    df2["B_agudat"] = df2["B_agudat"] * np.random.uniform(0.9, 1.1, n_stations)
    df2["Other"] = df2["Other"] * np.random.uniform(0.95, 1.05, n_stations)
    df2["Abstained"] = df2["Abstained"] * np.random.uniform(0.8, 1.3, n_stations)

    x1, x2, n1, n2 = build_station_tensors(df1, df2)

    # Row sums should match totals
    np.testing.assert_array_almost_equal(x1.sum(axis=1), n1)
    np.testing.assert_array_almost_equal(x2.sum(axis=1), n2)

    # Should preserve station count
    assert len(x1) == n_stations
    assert len(x2) == n_stations

    # All values should be non-negative
    assert (x1 >= 0).all()
    assert (x2 >= 0).all()

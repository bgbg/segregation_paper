"""Tests for PyMC model building and validation."""

import arviz as az
import numpy as np
import pymc as pm
import pytest

from src.transition_model.pymc_model import build_hierarchical_model, sample_model


def test_build_hierarchical_model_structure(sample_tensor_data):
    """Test that PyMC model builds with correct structure."""
    model = build_hierarchical_model(sample_tensor_data)

    # Should be a PyMC model
    assert isinstance(model, pm.Model)

    # Check that model has expected variables
    var_names = [var.name for var in model.unobserved_RVs]

    # Should have country-level matrix columns
    country_cols = [name for name in var_names if "M_country_col_" in name]
    assert len(country_cols) == 4, "Should have 4 country matrix columns"

    # Should have logistic-normal parameters
    assert any("Z_country" in name for name in var_names), "Should have Z_country parameter"
    assert any("diag_bias" in name for name in var_names), "Should have diag_bias parameter"
    assert any("sigma_country" in name for name in var_names), "Should have sigma_country parameter"

    # Should have overdispersion parameter (now log-parameterized)
    assert any("phi" in name for name in var_names), "Should have phi parameter"
    assert any("log_phi" in name for name in var_names), "Should have log_phi parameter"

    # Should have city-level matrices if cities are present
    city_vars = [name for name in var_names if "M_city_" in name]
    cities = [k for k in sample_tensor_data.keys() if k != "country"]
    if cities:
        assert (
            len(city_vars) > 0
        ), "Should have city-level variables when cities present"


def test_build_model_parameter_validation(sample_tensor_data):
    """Test model building with different parameter values."""
    # Test with custom parameters
    model = build_hierarchical_model(
        sample_tensor_data,
        diag_bias_mean=3.0,
        diag_bias_sigma=0.5,
        sigma_country=1.0,
        sigma_city=0.5,
        nu_scale=5.0,
    )

    # Model should build successfully
    assert isinstance(model, pm.Model)

    # Test with different valid parameter values
    model2 = build_hierarchical_model(sample_tensor_data, diag_bias_mean=1.0)
    assert isinstance(model2, pm.Model)

    model3 = build_hierarchical_model(sample_tensor_data, nu_scale=10.0)
    assert isinstance(model3, pm.Model)


def test_model_with_country_only():
    """Test model building with country data only."""
    country_only_data = {
        "country": {
            "x1": np.random.multinomial(500, [0.2, 0.15, 0.55, 0.1], 50).astype(float),
            "x2": np.random.multinomial(520, [0.22, 0.13, 0.57, 0.08], 50).astype(
                float
            ),
            "n1": None,  # Will be computed
            "n2": None,  # Will be computed
        }
    }

    # Compute totals
    country_only_data["country"]["n1"] = country_only_data["country"]["x1"].sum(axis=1)
    country_only_data["country"]["n2"] = country_only_data["country"]["x2"].sum(axis=1)

    model = build_hierarchical_model(country_only_data)

    # Should build successfully even without cities
    assert isinstance(model, pm.Model)

    # Should have country variables but no city variables
    var_names = [var.name for var in model.unobserved_RVs]
    country_vars = [name for name in var_names if "M_country_" in name]
    city_vars = [name for name in var_names if "M_city_" in name]

    assert len(country_vars) > 0, "Should have country variables"
    assert len(city_vars) == 0, "Should not have city variables"


def test_sample_model_basic(sample_tensor_data):
    """Test basic model sampling with small parameters."""
    model = build_hierarchical_model(sample_tensor_data)

    # Test sampling with minimal parameters for speed
    trace = sample_model(model, draws=10, tune=10, chains=1, random_seed=42)

    # Should return ArviZ InferenceData
    assert isinstance(trace, az.InferenceData)

    # Should have posterior samples
    assert hasattr(trace, "posterior")
    assert trace.posterior.dims["draw"] == 10
    assert trace.posterior.dims["chain"] == 1


def test_model_data_shapes(sample_tensor_data):
    """Test that model handles different data shapes correctly."""
    # Test with different numbers of stations
    data_small = {
        "country": {
            "x1": np.random.multinomial(300, [0.2, 0.15, 0.55, 0.1], 10).astype(float),
            "x2": np.random.multinomial(320, [0.22, 0.13, 0.57, 0.08], 10).astype(
                float
            ),
            "n1": None,
            "n2": None,
        }
    }
    data_small["country"]["n1"] = data_small["country"]["x1"].sum(axis=1)
    data_small["country"]["n2"] = data_small["country"]["x2"].sum(axis=1)

    model_small = build_hierarchical_model(data_small)
    assert isinstance(model_small, pm.Model)

    # Test with larger dataset
    data_large = {
        "country": {
            "x1": np.random.multinomial(800, [0.2, 0.15, 0.55, 0.1], 200).astype(float),
            "x2": np.random.multinomial(820, [0.22, 0.13, 0.57, 0.08], 200).astype(
                float
            ),
            "n1": None,
            "n2": None,
        }
    }
    data_large["country"]["n1"] = data_large["country"]["x1"].sum(axis=1)
    data_large["country"]["n2"] = data_large["country"]["x2"].sum(axis=1)

    model_large = build_hierarchical_model(data_large)
    assert isinstance(model_large, pm.Model)


def test_model_likelihood_computation(sample_tensor_data):
    """Test that model likelihood computation works."""
    model = build_hierarchical_model(sample_tensor_data)

    with model:
        # Should be able to compute log probability
        try:
            # Sample a single point to test likelihood
            point = pm.sample_prior_predictive(samples=1, random_seed=42)
            assert point is not None
        except Exception as e:
            pytest.fail(f"Model likelihood computation failed: {e}")


def test_model_parameter_interpretation():
    """Test that model parameters have sensible interpretations."""
    np.random.seed(42)

    # Create data with known transition pattern (high inertia)
    n_stations = 50
    # Election 1: mostly Shas voters
    x1 = np.zeros((n_stations, 4))
    x1[:, 0] = 150  # Shas
    x1[:, 1] = 30  # Agudat
    x1[:, 2] = 200  # Other
    x1[:, 3] = 20  # Abstained

    # Election 2: similar pattern (high inertia)
    x2 = np.zeros((n_stations, 4))
    x2[:, 0] = 140  # Shas (slight decrease)
    x2[:, 1] = 35  # Agudat (slight increase)
    x2[:, 2] = 210  # Other (slight increase)
    x2[:, 3] = 15  # Abstained (decrease)

    test_data = {
        "country": {
            "x1": x1.astype(float),
            "x2": x2.astype(float),
            "n1": x1.sum(axis=1).astype(float),
            "n2": x2.sum(axis=1).astype(float),
        }
    }

    model = build_hierarchical_model(
        test_data, diag_bias_mean=5.0, diag_bias_sigma=0.3
    )  # High inertia prior

    # Model should build successfully with realistic data
    assert isinstance(model, pm.Model)

    # Test that we can sample from prior
    with model:
        prior_samples = pm.sample_prior_predictive(samples=5, random_seed=42)
        assert prior_samples is not None


def test_model_edge_cases():
    """Test model behavior with edge cases."""
    # Test with very small vote counts
    tiny_data = {
        "country": {
            "x1": np.array([[1, 1, 2, 1], [2, 1, 1, 1]], dtype=float),
            "x2": np.array([[1, 2, 1, 1], [1, 1, 2, 1]], dtype=float),
            "n1": np.array([5, 5], dtype=float),
            "n2": np.array([5, 5], dtype=float),
        }
    }

    model = build_hierarchical_model(tiny_data)
    assert isinstance(model, pm.Model)

    # Test with zero votes in some categories
    zero_category_data = {
        "country": {
            "x1": np.array([[0, 10, 20, 5], [5, 0, 25, 10]], dtype=float),
            "x2": np.array([[2, 8, 18, 7], [3, 2, 22, 13]], dtype=float),
            "n1": np.array([35, 40], dtype=float),
            "n2": np.array([35, 40], dtype=float),
        }
    }

    model_zero = build_hierarchical_model(zero_category_data)
    assert isinstance(model_zero, pm.Model)


def test_sampling_parameter_validation():
    """Test sampling with different parameter values."""
    model = build_hierarchical_model(
        {
            "country": {
                "x1": np.array([[10, 5, 20, 5]], dtype=float),
                "x2": np.array([[12, 4, 18, 6]], dtype=float),
                "n1": np.array([40], dtype=float),
                "n2": np.array([40], dtype=float),
            }
        }
    )

    # Test sampling with minimal parameters
    trace = sample_model(model, draws=5, tune=5, chains=1)
    assert isinstance(trace, az.InferenceData)

    # Test different valid parameter combinations
    trace2 = sample_model(model, draws=10, tune=10, chains=2, target_accept=0.8)
    assert isinstance(trace2, az.InferenceData)

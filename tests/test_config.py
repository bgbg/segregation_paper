"""Tests for configuration loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.data_harmonizer import load_config


def test_load_config_success(temp_config_file, sample_config):
    """Test successful configuration loading."""
    config = load_config(temp_config_file)

    assert config is not None
    assert "cities" in config
    assert "data" in config
    assert "model" in config

    # Check specific values
    assert config["cities"]["target_cities"] == sample_config["cities"]["target_cities"]
    assert config["data"]["homogenic_filtering"]["threshold"] == 0.75
    assert config["model"]["logistic_normal"]["diag_bias_mean"] == 3.0


def test_load_config_missing_file():
    """Test loading non-existent config file."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")


def test_config_structure_validation(sample_config):
    """Test that config has required structure."""
    # Required top-level keys
    required_keys = ["cities", "data", "model", "categories", "paths"]
    for key in required_keys:
        assert key in sample_config, f"Missing required config key: {key}"

    # Required cities config
    assert "target_cities" in sample_config["cities"]
    assert isinstance(sample_config["cities"]["target_cities"], list)
    assert len(sample_config["cities"]["target_cities"]) > 0

    # Required data config
    data_config = sample_config["data"]
    assert "homogenic_filtering" in data_config
    assert "enabled" in data_config["homogenic_filtering"]
    assert "threshold" in data_config["homogenic_filtering"]
    assert "elections" in data_config
    assert "transition_pairs" in data_config

    # Required model config
    model_config = sample_config["model"]
    assert "logistic_normal" in model_config
    assert "sampling" in model_config

    # Check logistic-normal parameters
    logistic_normal_config = model_config["logistic_normal"]
    required_params = [
        "diag_bias_mean",
        "diag_bias_sigma",
        "sigma_country",
        "sigma_city",
        "nu_scale",
    ]
    for param in required_params:
        assert (
            param in logistic_normal_config
        ), f"Missing logistic_normal parameter: {param}"

    sampling_config = model_config["sampling"]
    sampling_keys = [
        "draws",
        "tune",
        "chains",
        "target_accept",
        "max_treedepth",
        "init",
        "random_seed",
    ]
    for key in sampling_keys:
        assert key in sampling_config, f"Missing sampling parameter: {key}"


def test_config_parameter_types(sample_config):
    """Test that config parameters have correct types."""
    # Cities should be list of strings
    cities = sample_config["cities"]["target_cities"]
    assert isinstance(cities, list)
    assert all(isinstance(city, str) for city in cities)

    # Homogenic threshold should be float between 0 and 1
    threshold = sample_config["data"]["homogenic_filtering"]["threshold"]
    assert isinstance(threshold, float)
    assert 0.0 <= threshold <= 1.0

    # Elections should be list of integers
    elections = sample_config["data"]["elections"]
    assert isinstance(elections, list)
    assert all(isinstance(election, int) for election in elections)

    # Logistic-normal parameters should be numeric
    logistic_normal = sample_config["model"]["logistic_normal"]
    assert isinstance(logistic_normal["diag_bias_mean"], (int, float))
    assert isinstance(logistic_normal["diag_bias_sigma"], (int, float))
    assert isinstance(logistic_normal["sigma_country"], (int, float))
    assert isinstance(logistic_normal["sigma_city"], (int, float))
    assert isinstance(logistic_normal["nu_scale"], (int, float))

    # Sampling parameters
    sampling = sample_config["model"]["sampling"]
    assert isinstance(sampling["draws"], int)
    assert isinstance(sampling["tune"], int)
    assert isinstance(sampling["chains"], int)
    assert isinstance(sampling["target_accept"], float)
    assert isinstance(sampling["max_treedepth"], int)
    assert isinstance(sampling["init"], str)
    assert isinstance(sampling["random_seed"], int)


def test_config_parameter_ranges(sample_config):
    """Test that config parameters are in valid ranges."""
    # Homogenic threshold
    threshold = sample_config["data"]["homogenic_filtering"]["threshold"]
    assert 0.0 < threshold <= 1.0, "Homogenic threshold should be between 0 and 1"

    # Logistic-normal parameters should be positive
    logistic_normal = sample_config["model"]["logistic_normal"]
    assert logistic_normal["diag_bias_mean"] > 0, "diag_bias_mean should be positive"
    assert logistic_normal["diag_bias_sigma"] > 0, "diag_bias_sigma should be positive"
    assert logistic_normal["sigma_country"] > 0, "sigma_country should be positive"
    assert logistic_normal["sigma_city"] > 0, "sigma_city should be positive"
    assert logistic_normal["nu_scale"] > 0, "nu_scale should be positive"

    # Sampling parameters
    sampling = sample_config["model"]["sampling"]
    assert sampling["draws"] > 0, "draws should be positive"
    assert sampling["tune"] > 0, "tune should be positive"
    assert sampling["chains"] > 0, "chains should be positive"
    assert (
        0.0 < sampling["target_accept"] < 1.0
    ), "target_accept should be between 0 and 1"
    assert sampling["max_treedepth"] > 0, "max_treedepth should be positive"
    assert sampling["init"] in [
        "jitter+adapt_diag",
        "adapt_diag",
        "jitter",
    ], "init should be valid initialization method"


def test_config_yaml_encoding():
    """Test that config can handle Unicode characters."""
    config_with_hebrew = {
        "test_cities": ["ירושלים", "בני ברק"],
        "test_mapping": {"שס": "shas", "ג": "agudat_israel"},
    }

    # Write and read back with Unicode
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(config_with_hebrew, f, default_flow_style=False, allow_unicode=True)
        temp_path = f.name

    try:
        # Should load without errors
        with open(temp_path, "r", encoding="utf-8") as f:
            loaded_config = yaml.safe_load(f)

        assert loaded_config["test_cities"] == ["ירושלים", "בני ברק"]
        assert loaded_config["test_mapping"]["שס"] == "shas"

    finally:
        Path(temp_path).unlink(missing_ok=True)

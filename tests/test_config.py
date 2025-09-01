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
    assert config["model"]["alpha_diag"] == 10.0


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
    assert "alpha_diag" in model_config
    assert "kappa_prior_scale" in model_config
    assert "sampling" in model_config

    sampling_config = model_config["sampling"]
    sampling_keys = ["draws", "tune", "chains", "target_accept", "random_seed"]
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

    # Model parameters should be numeric
    assert isinstance(sample_config["model"]["alpha_diag"], (int, float))
    assert isinstance(sample_config["model"]["kappa_prior_scale"], (int, float))

    # Sampling parameters
    sampling = sample_config["model"]["sampling"]
    assert isinstance(sampling["draws"], int)
    assert isinstance(sampling["tune"], int)
    assert isinstance(sampling["chains"], int)
    assert isinstance(sampling["target_accept"], float)
    assert isinstance(sampling["random_seed"], int)


def test_config_parameter_ranges(sample_config):
    """Test that config parameters are in valid ranges."""
    # Homogenic threshold
    threshold = sample_config["data"]["homogenic_filtering"]["threshold"]
    assert 0.0 < threshold <= 1.0, "Homogenic threshold should be between 0 and 1"

    # Model parameters should be positive
    assert sample_config["model"]["alpha_diag"] > 0, "alpha_diag should be positive"
    assert (
        sample_config["model"]["kappa_prior_scale"] > 0
    ), "kappa_prior_scale should be positive"

    # Sampling parameters
    sampling = sample_config["model"]["sampling"]
    assert sampling["draws"] > 0, "draws should be positive"
    assert sampling["tune"] > 0, "tune should be positive"
    assert sampling["chains"] > 0, "chains should be positive"
    assert (
        0.0 < sampling["target_accept"] < 1.0
    ), "target_accept should be between 0 and 1"


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

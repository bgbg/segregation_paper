"""Test fixtures for voter transition analysis tests."""

import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "cities": {"target_cities": ["jerusalem", "bnei brak", "ashdod"]},
        "data": {
            "homogenic_filtering": {"enabled": True, "threshold": 0.75},
            "elections": [20, 21, 22],
            "transition_pairs": ["kn20_21", "kn21_22"],
        },
        "model": {
            "logistic_normal": {
                "diag_bias_mean": 3.0,
                "diag_bias_sigma": 0.5,
                "sigma_country": 1.0,
                "sigma_city": 0.5,
                "nu_scale": 5.0,
            },
            "sampling": {
                "draws": 100,  # Small for testing
                "tune": 100,
                "chains": 2,
                "target_accept": 0.98,
                "random_seed": 42,
            },
            "diagnostics": {
                "max_rhat": 1.01,
                "min_ess_bulk": 400,
                "min_ess_tail": 400,
            },
        },
        "categories": ["shas", "agudat_israel", "other", "abstained"],
        "paths": {
            "raw_data": "data/raw",
            "interim_data": "data/interim",
            "processed_data": "data/processed",
            "columns_mapping": "data/external/columns.csv",
            "cities_mapping": "data/external/israeli_cities_hebrew_english.csv",
        },
    }


@pytest.fixture
def sample_raw_election_data():
    """Sample raw election data with Hebrew columns."""
    np.random.seed(42)
    n_stations = 50

    data = {
        "שם ישוב": ["ירושלים"] * 20 + ["בני ברק"] * 15 + ["אשדוד"] * 15,
        "קלפי": range(1, n_stations + 1),
        "בזב": np.random.randint(200, 800, n_stations),  # can_vote
        "כשרים": np.random.randint(150, 600, n_stations),  # legal votes
        "שס": np.random.randint(20, 200, n_stations),  # Shas votes
        "ג": np.random.randint(15, 180, n_stations),  # Agudat Israel votes
    }

    df = pd.DataFrame(data)

    # Ensure legal <= can_vote
    df["כשרים"] = np.minimum(df["כשרים"], df["בזב"] - 10)

    # Ensure party votes <= legal votes
    total_haredi = df["שס"] + df["ג"]
    df.loc[total_haredi > df["כשרים"], "שס"] = df["כשרים"] * 0.4
    df.loc[total_haredi > df["כשרים"], "ג"] = df["כשרים"] * 0.3

    return df


@pytest.fixture
def sample_harmonized_data():
    """Sample harmonized election data with English columns and computed categories."""
    np.random.seed(42)
    n_stations = 50

    city_names = ["Jerusalem"] * 20 + ["Bnei Brak"] * 15 + ["Ashdod"] * 15

    data = {
        "city_name": city_names,
        "ballot_code": range(1, n_stations + 1),
        "can_vote": np.random.randint(200, 800, n_stations),
        "legal": np.random.randint(150, 600, n_stations),
        "party_shas": np.random.randint(20, 200, n_stations),
        "party_agudat_israel": np.random.randint(15, 180, n_stations),
        "election": [20] * n_stations,
    }

    df = pd.DataFrame(data)

    # Ensure realistic constraints
    df["legal"] = np.minimum(df["legal"], df["can_vote"] - 10)
    total_haredi = df["party_shas"] + df["party_agudat_israel"]
    mask = total_haredi > df["legal"]
    df.loc[mask, "party_shas"] = (df.loc[mask, "legal"] * 0.4).astype(int)
    df.loc[mask, "party_agudat_israel"] = (df.loc[mask, "legal"] * 0.3).astype(int)

    # Compute categories
    df["A_shas"] = df["party_shas"]
    df["B_agudat"] = df["party_agudat_israel"]
    df["Other"] = df["legal"] - df["A_shas"] - df["B_agudat"]
    df["Abstained"] = df["can_vote"] - df["legal"]  # Corrected calculation

    # Ensure non-negative
    for col in ["A_shas", "B_agudat", "Other", "Abstained"]:
        df[col] = np.maximum(0, df[col])

    return df


@pytest.fixture
def column_mappings():
    """Sample Hebrew to English column mappings."""
    return {
        "שם ישוב": "city_name",
        "קלפי": "ballot_code",
        "בזב": "can_vote",
        "כשרים": "legal",
        "שס": "party_shas",
        "ג": "party_agudat_israel",
    }


@pytest.fixture
def city_mappings():
    """Sample Hebrew to English city name mappings."""
    return {"ירושלים": "jerusalem", "בני ברק": "bnei brak", "אשדוד": "ashdod"}


@pytest.fixture
def temp_config_file(sample_config):
    """Create temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_config, f, default_flow_style=False, allow_unicode=True)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def sample_tensor_data():
    """Sample tensor data for PyMC model testing."""
    np.random.seed(42)

    # Country data
    n_stations_country = 100
    x1_country = np.random.multinomial(
        500, [0.15, 0.12, 0.63, 0.10], n_stations_country
    )
    x2_country = np.random.multinomial(
        520, [0.16, 0.11, 0.65, 0.08], n_stations_country
    )

    # City data
    n_stations_city = 30
    x1_city = np.random.multinomial(300, [0.25, 0.20, 0.45, 0.10], n_stations_city)
    x2_city = np.random.multinomial(310, [0.26, 0.19, 0.47, 0.08], n_stations_city)

    return {
        "country": {
            "x1": x1_country.astype(float),
            "x2": x2_country.astype(float),
            "n1": x1_country.sum(axis=1).astype(float),
            "n2": x2_country.sum(axis=1).astype(float),
        },
        "jerusalem": {
            "x1": x1_city.astype(float),
            "x2": x2_city.astype(float),
            "n1": x1_city.sum(axis=1).astype(float),
            "n2": x2_city.sum(axis=1).astype(float),
        },
    }

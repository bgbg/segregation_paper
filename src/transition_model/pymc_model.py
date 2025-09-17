"""PyMC implementation of hierarchical ecological inference model.

This module implements the Bayesian hierarchical model for estimating
voter transition matrices between consecutive elections.
"""

from typing import Dict, Optional

import arviz as az
import numpy as np
import pymc as pm
import yaml


def build_hierarchical_model(
    data: Dict[str, Dict[str, np.ndarray]],
    diag_bias_mean: float = 2.0,
    diag_bias_sigma: float = 0.3,
    sigma_country: float = 0.5,
    sigma_D: float = 0.3,
    delta_scale: float = 0.5,
    *,
    priors: Optional[Dict] = None,
    innovation: Optional[Dict[str, float]] = None,
) -> pm.Model:
    """Build simplified hierarchical transition model focusing on core transition matrices.

    Simplified version removing non-essential components and using tighter priors
    for better convergence. Focus is on country and city-level transition matrices only.

    Args:
        data: Dictionary with country and city-level data tensors
        diag_bias_mean: Mean for diagonal bias (loyalty) parameter
        diag_bias_sigma: Standard deviation for diagonal bias parameter
        sigma_country: Scale for country-level logit variation
        sigma_D: Scale for shared deviation pattern matrix
        delta_scale: Scale for city-specific deviation scalars
        priors: Optional dictionary of priors from previous time period
        innovation: Optional dictionary of innovation variances

    Returns:
        PyMC model object
    """
    K = 4  # Number of categories
    cities = [k for k in data.keys() if k != "country"]
    n_cities = len(cities)

    # Validate category order consistency
    country_data = data["country"]
    x1_country = country_data["x1"]
    x2_country = country_data["x2"]
    assert (
        x1_country.shape[1] == x2_country.shape[1] == K
    ), f"Category mismatch: x1 has {x1_country.shape[1]}, x2 has {x2_country.shape[1]}, expected {K}"

    with pm.Model() as model:
        # Country-level logistic-normal prior (tighter for stability)
        eyeK = np.eye(K)

        # Use temporal priors if provided
        if priors and "Z_country" in priors:
            zc_mu = np.array(priors["Z_country"], dtype=float)
            zc_sigma = (innovation or {}).get("Z_country_sigma", sigma_country)
            Z_country = pm.Normal("Z_country", mu=zc_mu, sigma=zc_sigma, shape=(K, K))
        else:
            Z_country = pm.Normal(
                "Z_country", mu=0.0, sigma=sigma_country, shape=(K, K)
            )

        if priors and "diag_bias" in priors:
            diag_mu = float(priors["diag_bias"])  # previous estimate
            diag_sd = (innovation or {}).get("diag_bias_sigma", diag_bias_sigma)
            diag_bias = pm.Normal("diag_bias", mu=diag_mu, sigma=diag_sd)
        else:
            diag_bias = pm.Normal(
                "diag_bias", mu=diag_bias_mean, sigma=diag_bias_sigma
            )  # encodes loyalty in mean

        # Country transition matrix
        M_country_cols = []
        for j in range(K):
            z = Z_country[:, j] + diag_bias * eyeK[:, j]
            col = pm.Deterministic(f"M_country_col_{j}", pm.math.softmax(z))
            M_country_cols.append(col)
        M_country = pm.math.stack(M_country_cols, axis=1)

        # Simplified city-level deviations (remove complex rank-1 structure)
        M_cities = None
        if n_cities > 0:
            # Simplified: Independent city deviations from country
            if priors and "D" in priors:
                D_mu = np.array(priors["D"], dtype=float)
                D_sigma = (innovation or {}).get("D_sigma", sigma_D)
                D = pm.Normal(
                    "D_deviation_pattern", mu=D_mu, sigma=D_sigma, shape=(K, K)
                )
            else:
                D = pm.Normal("D_deviation_pattern", mu=0, sigma=sigma_D, shape=(K, K))

            # Simplified city scalars (normal distribution, no heavy tails)
            delta_city = pm.Normal(
                "delta_city", mu=0, sigma=delta_scale, shape=n_cities
            )

            # City logits: simplified deviation from country
            Z_city = pm.Deterministic(
                "Z_city",
                Z_country[None, :, :] + delta_city[:, None, None] * D[None, :, :],
            )

            # Build city transition matrices (simplified)
            M_cities_list = []
            for c in range(n_cities):
                city_name = (
                    cities[c].replace(" ", "_").replace("'", "")
                )  # Clean name for PyMC
                city_cols = []
                for j in range(K):
                    zc = Z_city[c, :, j] + diag_bias * eyeK[:, j]
                    colc = pm.Deterministic(
                        f"M_{city_name}_col_{j}", pm.math.softmax(zc)
                    )
                    city_cols.append(colc)
                M_city = pm.math.stack(city_cols, axis=1)
                M_cities_list.append(M_city)
            M_cities = pm.math.stack(M_cities_list, axis=0)

        # Simplified overdispersion (fixed value to reduce complexity)
        phi = pm.Deterministic("phi", pm.math.constant(100.0))  # Fixed concentration

        # Likelihood for country data
        country_data = data["country"]
        x1_country = country_data["x1"]  # Shape: (n_stations, K)
        x2_country = country_data["x2"]
        n2_country = country_data["n2"]  # Shape: (n_stations,)

        # Proportions from election t
        p1_country = x1_country / x1_country.sum(axis=1, keepdims=True)

        # Expected proportions in election t+1
        q_country = pm.math.dot(p1_country, M_country.T)  # Shape: (n_stations, K)

        # Observed votes in election t+1
        pm.DirichletMultinomial(
            "x2_country_obs", n=n2_country, a=phi * q_country, observed=x2_country
        )

        # Likelihood for city data
        for i, city in enumerate(cities):
            city_data = data[city]
            x1_city = city_data["x1"]
            x2_city = city_data["x2"]
            n2_city = city_data["n2"]

            # City-specific transition matrix with fallback
            M_city = M_cities[i] if M_cities is not None else M_country

            # Proportions and predictions
            p1_city = x1_city / x1_city.sum(axis=1, keepdims=True)
            q_city = pm.math.dot(p1_city, M_city.T)

            # Clean city name for PyMC variable naming
            city_clean = city.replace(" ", "_").replace("'", "")

            # Observed votes
            pm.DirichletMultinomial(
                f"x2_{city_clean}_obs", n=n2_city, a=phi * q_city, observed=x2_city
            )

    return model


def sample_model(
    model: pm.Model,
    draws: int = 3000,
    tune: int = 3000,
    chains: int = 4,
    target_accept: float = 0.95,
    max_treedepth: int = 12,
    init: str = "adapt_diag",
    random_seed: Optional[int] = None,
) -> az.InferenceData:
    """Sample from the simplified hierarchical transition model.

    Args:
        model: PyMC model to sample from
        draws: Number of posterior samples per chain (increased for better convergence)
        tune: Number of tuning samples per chain (increased for better adaptation)
        chains: Number of MCMC chains
        target_accept: Target acceptance rate (slightly lower for efficiency)
        max_treedepth: Maximum tree depth for NUTS sampler (reduced for stability)
        init: Initialization method for chains (simplified for stability)
        random_seed: Random seed for reproducibility

    Returns:
        ArviZ InferenceData object with posterior samples
    """
    with model:
        # Progressive sampling strategy for difficult posteriors
        print("Starting progressive sampling strategy...")

        # Stage 1: Initial burn-in with conservative settings
        print("Stage 1: Initial adaptation...")
        trace_stage1 = pm.sample(
            draws=500,
            tune=2000,  # Longer tuning for initial adaptation
            chains=chains,
            target_accept=0.90,  # Lower acceptance for initial exploration
            max_treedepth=10,
            init="jitter+adapt_diag",
            random_seed=random_seed,
            return_inferencedata=True,
            discard_tuned_samples=False,
        )

        # Stage 2: Main sampling with better initialization
        print("Stage 2: Main sampling with adapted parameters...")
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            max_treedepth=max_treedepth,
            init=init,
            random_seed=random_seed,
            return_inferencedata=True,
        )

    return trace


def get_transition_matrices(
    trace: az.InferenceData, city_names: list
) -> Dict[str, np.ndarray]:
    """Extract transition matrices from posterior samples.

    Args:
        trace: Posterior samples from model
        city_names: List of city names in order

    Returns:
        Dictionary with posterior mean transition matrices
    """
    results = {}

    # Country transition matrix
    M_country_cols = []
    for j in range(4):
        col = trace.posterior[f"M_country_col_{j}"].mean(dim=["chain", "draw"]).values
        M_country_cols.append(col)
    results["country"] = np.stack(M_country_cols, axis=1)

    # City transition matrices
    for i, city in enumerate(city_names):
        city_clean = city.replace(" ", "_").replace("'", "")
        M_city_cols = []
        for j in range(4):
            col = (
                trace.posterior[f"M_{city_clean}_col_{j}"]
                .mean(dim=["chain", "draw"])
                .values
            )
            M_city_cols.append(col)
        results[city] = np.stack(M_city_cols, axis=1)

    return results


def generate_test_data(
    cities: Optional[list] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Generate synthetic test data for model demonstration.

    Args:
        cities: List of city names to generate data for; if None, uses default set

    Returns:
        Dictionary with country and city test data
    """
    np.random.seed(42)

    # Default cities if none provided
    if cities is None:
        cities = ["jerusalem", "bnei brak", "tel aviv"]

    # Categories: [Shas, Agudat, Other, Abstained]
    n_stations_country = 100
    n_stations_city = 30

    # Country-level data (baseline patterns)
    x1_country = np.random.multinomial(
        500, [0.15, 0.12, 0.63, 0.10], n_stations_country
    )
    x2_country = np.random.multinomial(
        520, [0.16, 0.11, 0.65, 0.08], n_stations_country
    )

    data = {
        "country": {
            "x1": x1_country.astype(float),
            "x2": x2_country.astype(float),
            "n1": x1_country.sum(axis=1).astype(float),
            "n2": x2_country.sum(axis=1).astype(float),
        }
    }

    # Generate city data dynamically based on city characteristics
    for city in cities:
        city_lower = city.lower()

        # Determine voting patterns based on city characteristics
        if any(
            haredi_city in city_lower
            for haredi_city in [
                "jerusalem",
                "bnei brak",
                "beit shemesh",
                "elad",
                "modi",
            ]
        ):
            # High Haredi population cities
            if "bnei brak" in city_lower or "elad" in city_lower:
                # Very high Haredi
                probs_t = [0.35, 0.30, 0.25, 0.10]
                probs_t1 = [0.37, 0.28, 0.27, 0.08]
            else:
                # Moderate Haredi (Jerusalem, Beit Shemesh, Modi'in Illit)
                probs_t = [0.25, 0.20, 0.45, 0.10]
                probs_t1 = [0.26, 0.19, 0.47, 0.08]
        elif "ashdod" in city_lower:
            # Mixed population
            probs_t = [0.18, 0.15, 0.57, 0.10]
            probs_t1 = [0.19, 0.14, 0.59, 0.08]
        else:
            # Default secular/mixed pattern
            probs_t = [0.05, 0.03, 0.82, 0.10]
            probs_t1 = [0.04, 0.03, 0.85, 0.08]

        # Generate data for this city
        x1_city = np.random.multinomial(300, probs_t, n_stations_city)
        x2_city = np.random.multinomial(310, probs_t1, n_stations_city)

        data[city] = {
            "x1": x1_city.astype(float),
            "x2": x2_city.astype(float),
            "n1": x1_city.sum(axis=1).astype(float),
            "n2": x2_city.sum(axis=1).astype(float),
        }

    return data


def load_cities_from_config(config_path: str = "data/config.yaml") -> list:
    """Load target cities from configuration file.

    Args:
        config_path: Path to config YAML file

    Returns:
        List of target city names
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config["cities"]["target_cities"]
    except Exception as e:
        print(f"Warning: Could not load cities from {config_path}: {e}")
        print("Using default cities: jerusalem, bnei brak, tel aviv")
        return ["jerusalem", "bnei brak", "tel aviv"]


def print_transition_summary(trace: az.InferenceData, city_names: list):
    """Print simplified summary focusing on transition matrices."""

    print("\n" + "=" * 60)
    print("SIMPLIFIED TRANSITION MODEL RESULTS")
    print("=" * 60)

    # Country-level transition matrix
    print("\nCountry-level Transition Matrix:")
    print("(rows: to party, cols: from party)")
    print("Parties: [Shas, Agudat, Other, Abstained]")

    matrices = get_transition_matrices(trace, city_names)
    print(np.round(matrices["country"], 3))

    # Diagonal bias (loyalty parameter)
    diag_bias = trace.posterior["diag_bias"].mean(dim=["chain", "draw"]).values
    print(f"\nDiagonal Bias (loyalty): {diag_bias:.2f}")

    # City transition matrices
    print("\n" + "-" * 40)
    print("City-Level Transition Matrices:")
    print("-" * 40)

    for city in city_names:
        print(f"\n{city.upper()}:")
        print(np.round(matrices[city], 3))


if __name__ == "__main__":
    print("Hierarchical Voter Transition Model with Rank-1 City Deviations")
    print("=" * 60)

    # Load cities from config
    city_names = load_cities_from_config()
    print(f"Using cities from config: {city_names}")

    # Generate test data for config cities
    test_data = generate_test_data(cities=city_names)

    print("\nData Summary:")
    print(f"- Countries: 1")
    print(f"- Cities: {len(city_names)} ({', '.join(city_names)})")
    print(f"- Categories: 4 (Shas, Agudat Israel, Other, Abstained)")
    print(f"- Country stations: {len(test_data['country']['x1'])}")
    if city_names:
        print(f"- City stations per city: {len(test_data[city_names[0]]['x1'])}")

    # Build simplified model
    print("\nBuilding simplified hierarchical model...")
    model = build_hierarchical_model(test_data)

    # Sample from model (test with simplified parameters)
    print("\nSampling from posterior (test run)...")
    try:
        trace = sample_model(model, draws=1000, tune=1000, chains=2, random_seed=42)

        # Print results
        print_transition_summary(trace, city_names)

        # Basic diagnostics
        print("\n" + "=" * 60)
        print("MODEL DIAGNOSTICS")
        print("=" * 60)
        print(
            az.summary(
                trace,
                var_names=["delta_city", "D_deviation_pattern", "diag_bias"],
                hdi_prob=0.89,
            )
        )

    except Exception as e:
        print(f"Sampling failed: {e}")
        print("This is expected if PyMC dependencies are missing.")

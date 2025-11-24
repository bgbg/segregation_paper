"""PyMC implementation of independent city models with countrywide prior.

This module implements a two-stage approach:
1. Fit countrywide transition matrix
2. Fit each city independently using country posterior as prior

This approach allows more inter-city variability compared to hierarchical pooling.
"""

from typing import Dict, Optional, Tuple

import arviz as az
import numpy as np
import pymc as pm


def build_country_model(
    country_data: Dict[str, np.ndarray],
    diag_bias_mean: float = 2.0,
    diag_bias_sigma: float = 0.3,
    sigma_country: float = 0.5,
    *,
    priors: Optional[Dict] = None,
    innovation: Optional[Dict[str, float]] = None,
) -> pm.Model:
    """Build model for countrywide transition matrix only.

    Args:
        country_data: Dictionary with country-level data tensors
        diag_bias_mean: Mean for diagonal bias (loyalty) parameter
        diag_bias_sigma: Standard deviation for diagonal bias parameter
        sigma_country: Scale for country-level logit variation
        priors: Optional dictionary of priors from previous time period
        innovation: Optional dictionary of innovation variances

    Returns:
        PyMC model object for country-level estimation
    """
    K = 4  # Number of categories

    # Validate data
    x1_country = country_data["x1"]
    x2_country = country_data["x2"]
    n2_country = country_data["n2"]
    assert (
        x1_country.shape[1] == x2_country.shape[1] == K
    ), f"Category mismatch: x1 has {x1_country.shape[1]}, x2 has {x2_country.shape[1]}, expected {K}"

    with pm.Model() as model:
        eyeK = np.eye(K)

        # Country-level logistic-normal prior
        if priors and "Z_country" in priors:
            zc_mu = np.array(priors["Z_country"], dtype=float)
            zc_sigma = (innovation or {}).get("Z_country_sigma", sigma_country)
            Z_country = pm.Normal("Z_country", mu=zc_mu, sigma=zc_sigma, shape=(K, K))
        else:
            Z_country = pm.Normal(
                "Z_country", mu=0.0, sigma=sigma_country, shape=(K, K)
            )

        # Diagonal bias for voter loyalty
        if priors and "diag_bias" in priors:
            diag_mu = float(priors["diag_bias"])
            diag_sd = (innovation or {}).get("diag_bias_sigma", diag_bias_sigma)
            diag_bias = pm.Normal("diag_bias", mu=diag_mu, sigma=diag_sd)
        else:
            diag_bias = pm.Normal(
                "diag_bias", mu=diag_bias_mean, sigma=diag_bias_sigma
            )

        # Country transition matrix
        M_country_cols = []
        for j in range(K):
            z = Z_country[:, j] + diag_bias * eyeK[:, j]
            col = pm.Deterministic(f"M_country_col_{j}", pm.math.softmax(z))
            M_country_cols.append(col)
        M_country = pm.math.stack(M_country_cols, axis=1)

        # Fixed concentration parameter
        phi = pm.Deterministic("phi", pm.math.constant(100.0))

        # Proportions from election t
        p1_country = x1_country / x1_country.sum(axis=1, keepdims=True)

        # Expected proportions in election t+1
        q_country = pm.math.dot(p1_country, M_country.T)

        # Observed votes in election t+1
        pm.DirichletMultinomial(
            "x2_country_obs", n=n2_country, a=phi * q_country, observed=x2_country
        )

    return model


def build_city_model(
    city_data: Dict[str, np.ndarray],
    city_name: str,
    country_posterior: Dict[str, np.ndarray],
    sigma_city: float = 0.8,
    use_country_diag_bias: bool = True,
) -> pm.Model:
    """Build independent model for a single city using country posterior as prior.

    Args:
        city_data: Dictionary with city-level data tensors
        city_name: Name of the city (for variable naming)
        country_posterior: Dictionary with posterior means from country model
            Keys: 'Z_country', 'diag_bias'
        sigma_city: Scale for city-level deviation from country prior
        use_country_diag_bias: If True, use country diag_bias; if False, estimate separately

    Returns:
        PyMC model object for city-level estimation
    """
    K = 4  # Number of categories

    # Validate data
    x1_city = city_data["x1"]
    x2_city = city_data["x2"]
    n2_city = city_data["n2"]
    assert (
        x1_city.shape[1] == x2_city.shape[1] == K
    ), f"Category mismatch: x1 has {x1_city.shape[1]}, x2 has {x2_city.shape[1]}, expected {K}"

    # Clean city name for PyMC
    city_clean = city_name.replace(" ", "_").replace("'", "")

    with pm.Model() as model:
        eyeK = np.eye(K)

        # Use country posterior as prior mean for city logits
        Z_country_prior = country_posterior["Z_country"]
        Z_city = pm.Normal(
            f"Z_{city_clean}",
            mu=Z_country_prior,
            sigma=sigma_city,
            shape=(K, K)
        )

        # Diagonal bias: either use country estimate or estimate separately
        if use_country_diag_bias:
            diag_bias = pm.Deterministic(
                "diag_bias",
                pm.math.constant(float(country_posterior["diag_bias"]))
            )
        else:
            # Estimate city-specific diagonal bias with country posterior as prior
            diag_bias_prior = float(country_posterior["diag_bias"])
            diag_bias = pm.Normal(
                "diag_bias",
                mu=diag_bias_prior,
                sigma=0.3
            )

        # City transition matrix
        M_city_cols = []
        for j in range(K):
            z = Z_city[:, j] + diag_bias * eyeK[:, j]
            col = pm.Deterministic(f"M_{city_clean}_col_{j}", pm.math.softmax(z))
            M_city_cols.append(col)
        M_city = pm.math.stack(M_city_cols, axis=1)

        # Fixed concentration parameter
        phi = pm.Deterministic("phi", pm.math.constant(100.0))

        # Proportions from election t
        p1_city = x1_city / x1_city.sum(axis=1, keepdims=True)

        # Expected proportions in election t+1
        q_city = pm.math.dot(p1_city, M_city.T)

        # Observed votes in election t+1
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
    progressive: bool = True,
) -> az.InferenceData:
    """Sample from a PyMC model.

    Args:
        model: PyMC model to sample from
        draws: Number of posterior samples per chain
        tune: Number of tuning samples per chain
        chains: Number of MCMC chains
        target_accept: Target acceptance rate
        max_treedepth: Maximum tree depth for NUTS sampler
        init: Initialization method for chains
        random_seed: Random seed for reproducibility
        progressive: Whether to use progressive sampling strategy

    Returns:
        ArviZ InferenceData object with posterior samples
    """
    with model:
        if progressive:
            # Progressive sampling strategy for difficult posteriors
            print("Starting progressive sampling strategy...")

            # Stage 1: Initial burn-in with conservative settings
            print("Stage 1: Initial adaptation...")
            trace_stage1 = pm.sample(
                draws=500,
                tune=2000,
                chains=chains,
                target_accept=0.90,
                max_treedepth=10,
                init="jitter+adapt_diag",
                random_seed=random_seed,
                return_inferencedata=True,
                discard_tuned_samples=False,
            )

            # Stage 2: Main sampling
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


def extract_country_posterior_summary(trace: az.InferenceData) -> Dict[str, np.ndarray]:
    """Extract posterior means from country model to use as priors for cities.

    Args:
        trace: Posterior samples from country model

    Returns:
        Dictionary with posterior means for Z_country and diag_bias
    """
    posterior = trace.posterior

    # Extract posterior means
    Z_country_mean = posterior["Z_country"].mean(dim=["chain", "draw"]).values
    diag_bias_mean = posterior["diag_bias"].mean(dim=["chain", "draw"]).values

    return {
        "Z_country": Z_country_mean,
        "diag_bias": diag_bias_mean,
    }


def fit_independent_models(
    data: Dict[str, Dict[str, np.ndarray]],
    diag_bias_mean: float = 2.0,
    diag_bias_sigma: float = 0.3,
    sigma_country: float = 0.5,
    sigma_city: float = 0.8,
    draws: int = 3000,
    tune: int = 3000,
    chains: int = 4,
    target_accept: float = 0.95,
    max_treedepth: int = 12,
    init: str = "adapt_diag",
    random_seed: Optional[int] = None,
    priors: Optional[Dict] = None,
    innovation: Optional[Dict[str, float]] = None,
) -> Tuple[az.InferenceData, Dict[str, az.InferenceData]]:
    """Fit country model followed by independent city models.

    Args:
        data: Dictionary with country and city-level data tensors
        diag_bias_mean: Mean for diagonal bias parameter
        diag_bias_sigma: Standard deviation for diagonal bias parameter
        sigma_country: Scale for country-level logit variation
        sigma_city: Scale for city-level deviation from country prior
        draws: Number of posterior samples per chain
        tune: Number of tuning samples per chain
        chains: Number of MCMC chains
        target_accept: Target acceptance rate
        max_treedepth: Maximum tree depth for NUTS
        init: Initialization method
        random_seed: Random seed for reproducibility
        priors: Optional temporal priors from previous election
        innovation: Optional innovation variances for temporal priors

    Returns:
        Tuple of (country_trace, dict of city traces)
    """
    print("=" * 60)
    print("STAGE 1: Fitting country-level model")
    print("=" * 60)

    # Stage 1: Fit country model
    country_model = build_country_model(
        data["country"],
        diag_bias_mean=diag_bias_mean,
        diag_bias_sigma=diag_bias_sigma,
        sigma_country=sigma_country,
        priors=priors,
        innovation=innovation,
    )

    country_trace = sample_model(
        country_model,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        max_treedepth=max_treedepth,
        init=init,
        random_seed=random_seed,
        progressive=True,
    )

    # Extract country posterior for use as city priors
    country_posterior = extract_country_posterior_summary(country_trace)

    print("\n" + "=" * 60)
    print("STAGE 2: Fitting city-level models independently")
    print("=" * 60)

    # Stage 2: Fit each city independently
    city_traces = {}
    cities = [k for k in data.keys() if k != "country"]

    for i, city in enumerate(cities):
        print(f"\nFitting city {i+1}/{len(cities)}: {city}")
        print("-" * 60)

        city_model = build_city_model(
            data[city],
            city_name=city,
            country_posterior=country_posterior,
            sigma_city=sigma_city,
            use_country_diag_bias=True,
        )

        city_trace = sample_model(
            city_model,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            max_treedepth=max_treedepth,
            init=init,
            random_seed=random_seed if random_seed is None else random_seed + i + 1,
            progressive=False,  # Country already converged, cities should be easier
        )

        city_traces[city] = city_trace

    print("\n" + "=" * 60)
    print("Independent model fitting complete!")
    print("=" * 60)

    return country_trace, city_traces


def get_transition_matrices(
    country_trace: az.InferenceData,
    city_traces: Dict[str, az.InferenceData],
) -> Dict[str, np.ndarray]:
    """Extract transition matrices from independent model posteriors.

    Args:
        country_trace: Posterior samples from country model
        city_traces: Dictionary of posterior samples from city models

    Returns:
        Dictionary with posterior mean transition matrices
    """
    results = {}

    # Country transition matrix
    M_country_cols = []
    for j in range(4):
        col_samples = country_trace.posterior[f"M_country_col_{j}"]
        col_mean = col_samples.mean(dim=["chain", "draw"]).values
        M_country_cols.append(col_mean)
    results["country"] = np.stack(M_country_cols, axis=1)

    # City transition matrices
    for city_name, city_trace in city_traces.items():
        city_clean = city_name.replace(" ", "_").replace("'", "")
        M_city_cols = []
        for j in range(4):
            col_samples = city_trace.posterior[f"M_{city_clean}_col_{j}"]
            col_mean = col_samples.mean(dim=["chain", "draw"]).values
            M_city_cols.append(col_mean)
        results[city_name] = np.stack(M_city_cols, axis=1)

    return results

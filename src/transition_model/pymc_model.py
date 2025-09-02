"""PyMC implementation of hierarchical ecological inference model.

This module implements the Bayesian hierarchical model for estimating
voter transition matrices between consecutive elections.
"""

from typing import Dict, Optional

import arviz as az
import numpy as np
import pymc as pm


def build_hierarchical_model(
    data: Dict[str, Dict[str, np.ndarray]],
    diag_bias_mean: float = 3.0,
    diag_bias_sigma: float = 0.5,
    sigma_country: float = 1.0,
    sigma_city: float = 0.5,
    nu_scale: float = 5.0,
) -> pm.Model:
    """Build hierarchical logistic-normal transition model.

    Uses logistic-normal parameterization with Student-t deviations for cities.
    Country matrices use Normal priors on logits with diagonal bias for loyalty.
    Cities deviate via heavy-tailed Student-t around country logits (sparse/outlier-friendly).

    Args:
        data: Dictionary with country and city-level data tensors
        diag_bias_mean: Mean for diagonal bias (loyalty) parameter
        diag_bias_sigma: Standard deviation for diagonal bias parameter
        sigma_country: Scale for country-level logit variation
        sigma_city: Scale for city-level logit deviations
        nu_scale: Scale parameter for Student-t degrees of freedom

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
    assert x1_country.shape[1] == x2_country.shape[1] == K, f"Category mismatch: x1 has {x1_country.shape[1]}, x2 has {x2_country.shape[1]}, expected {K}"

    with pm.Model() as model:
        # Country-level logistic-normal prior (robust)
        eyeK = np.eye(K)
        
        sigma_country_param = pm.HalfNormal("sigma_country", sigma=sigma_country)
        Z_country = pm.Normal("Z_country", mu=0.0, sigma=sigma_country_param, shape=(K, K))
        
        diag_bias = pm.Normal("diag_bias", mu=diag_bias_mean, sigma=diag_bias_sigma)  # encodes loyalty in mean
        
        M_country_cols = []
        for j in range(K):
            z = Z_country[:, j] + diag_bias * eyeK[:, j]
            col = pm.Deterministic(f"M_country_col_{j}", pm.math.softmax(z))
            M_country_cols.append(col)
        M_country = pm.math.stack(M_country_cols, axis=1)

        # City-level sparse (heavy-tailed) deviations on logits
        sigma_city_param = pm.HalfNormal("sigma_city", sigma=sigma_city)
        nu_raw = pm.Exponential("nu_raw", lam=1/nu_scale)
        nu = pm.Deterministic("nu", nu_raw + 2.0)

        M_cities = None
        if n_cities > 0:
            Z_city = pm.StudentT("Z_city", nu=nu, mu=Z_country, sigma=sigma_city_param, shape=(n_cities, K, K))
            M_cities_list = []
            for c in range(n_cities):
                city_cols = []
                for j in range(K):
                    zc = Z_city[c, :, j] + diag_bias * eyeK[:, j]
                    colc = pm.Deterministic(f"M_city_{c}_col_{j}", pm.math.softmax(zc))
                    city_cols.append(colc)
                M_city = pm.math.stack(city_cols, axis=1)
                M_cities_list.append(M_city)
            M_cities = pm.math.stack(M_cities_list, axis=0)

        # Overdispersion (log-parameterized)
        log_phi = pm.Normal("log_phi", mu=0.0, sigma=1.0)
        phi = pm.Deterministic("phi", pm.math.exp(log_phi))

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

            # Observed votes
            pm.DirichletMultinomial(
                f"x2_{city}_obs", n=n2_city, a=phi * q_city, observed=x2_city
            )

    return model


def sample_model(
    model: pm.Model,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.98,
    random_seed: Optional[int] = None,
) -> az.InferenceData:
    """Sample from the hierarchical transition model.

    Args:
        model: PyMC model to sample from
        draws: Number of posterior samples per chain
        tune: Number of tuning samples per chain
        chains: Number of MCMC chains
        target_accept: Target acceptance rate
        random_seed: Random seed for reproducibility

    Returns:
        ArviZ InferenceData object with posterior samples
    """
    with model:
        # Use NUTS sampler with higher target acceptance for better geometry
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
        )

    return trace


def generate_test_data() -> Dict[str, Dict[str, np.ndarray]]:
    """Generate synthetic test data for model visualization.

    Returns:
        Dictionary with country and city test data
    """
    np.random.seed(42)

    # Categories: [Shas, Agudat, Other, Abstained]
    n_stations_country = 100
    n_stations_city = 30

    # Country data
    x1_country = np.random.multinomial(
        500, [0.15, 0.12, 0.63, 0.10], n_stations_country
    )
    x2_country = np.random.multinomial(
        520, [0.16, 0.11, 0.65, 0.08], n_stations_country
    )
    n1_country = x1_country.sum(axis=1)
    n2_country = x2_country.sum(axis=1)

    # City data (Jerusalem example)
    x1_city = np.random.multinomial(300, [0.25, 0.20, 0.45, 0.10], n_stations_city)
    x2_city = np.random.multinomial(310, [0.26, 0.19, 0.47, 0.08], n_stations_city)
    n1_city = x1_city.sum(axis=1)
    n2_city = x2_city.sum(axis=1)

    # Generate data for 2 cities: Jerusalem and Bnei Brak
    cities = ["jerusalem", "bnei brak"]

    data = {
        "country": {
            "x1": x1_country.astype(float),
            "x2": x2_country.astype(float),
            "n1": n1_country.astype(float),
            "n2": n2_country.astype(float),
        }
    }

    # Jerusalem data (higher Haredi population)
    x1_jerusalem = np.random.multinomial(300, [0.25, 0.20, 0.45, 0.10], n_stations_city)
    x2_jerusalem = np.random.multinomial(310, [0.26, 0.19, 0.47, 0.08], n_stations_city)

    # Bnei Brak data (very high Haredi population)
    x1_bnei_brak = np.random.multinomial(280, [0.35, 0.30, 0.25, 0.10], n_stations_city)
    x2_bnei_brak = np.random.multinomial(290, [0.37, 0.28, 0.27, 0.08], n_stations_city)

    data["jerusalem"] = {
        "x1": x1_jerusalem.astype(float),
        "x2": x2_jerusalem.astype(float),
        "n1": x1_jerusalem.sum(axis=1).astype(float),
        "n2": x2_jerusalem.sum(axis=1).astype(float),
    }

    data["bnei brak"] = {
        "x1": x1_bnei_brak.astype(float),
        "x2": x2_bnei_brak.astype(float),
        "n1": x1_bnei_brak.sum(axis=1).astype(float),
        "n2": x2_bnei_brak.sum(axis=1).astype(float),
    }

    return data


if __name__ == "__main__":
    print("Generating voter transition model visualization...")

    # Generate test data
    test_data = generate_test_data()

    # Build model
    model = build_hierarchical_model(test_data)

    try:
        # Generate model graph
        graph = pm.model_to_graphviz(model)

        # Save as PNG
        output_file = "voter_transition_model"
        graph.render(output_file, format="png", cleanup=True)

        print(f"Model visualization saved as {output_file}.png")

        # Print model summary
        print("\nModel structure:")
        print(f"- Countries: 1")
        cities = [k for k in test_data.keys() if k != "country"]
        print(f"- Cities: {len(cities)} ({', '.join(cities)})")
        print(f"- Categories: 4 (Shas, Agudat Israel, Other, Abstained)")
        print(f"- Country stations: {len(test_data['country']['x1'])}")
        print(f"- City stations per city: {len(test_data['jerusalem']['x1'])}")

    except Exception as e:
        print(f"Error generating visualization: {e}")
        print("Make sure graphviz is installed: pip install graphviz")
        print(
            "And system graphviz: brew install graphviz (macOS) or apt-get install graphviz (Linux)"
        )

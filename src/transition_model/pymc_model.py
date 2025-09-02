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
    alpha_diag: float = 5.0,
    alpha_offdiag_floor: float = 1.0,
    kappa_prior_scale: float = 10.0,
) -> pm.Model:
    """Build hierarchical Dirichlet-Multinomial transition model.

    Args:
        data: Dictionary with country and city-level data tensors
        alpha_diag: Prior strength for diagonal (inertia) elements
        alpha_offdiag_floor: Minimum prior strength for off-diagonal elements
        kappa_prior_scale: Scale parameter for pooling strength prior

    Returns:
        PyMC model object
    """
    K = 4  # Number of categories
    cities = [k for k in data.keys() if k != "country"]
    n_cities = len(cities)

    with pm.Model() as model:
        # Country-level transition matrix (K x K, column-stochastic)
        # Prior encourages inertia (higher diagonal elements)
        alpha_base = np.ones((K, K)) * alpha_offdiag_floor
        np.fill_diagonal(alpha_base, alpha_diag)

        # Country transition matrix columns (each column sums to 1)
        M_country_cols = []
        for j in range(K):
            col = pm.Dirichlet(f"M_country_col_{j}", a=alpha_base[:, j])
            M_country_cols.append(col)

        M_country = pm.math.stack(M_country_cols, axis=1)  # Shape: (K, K)

        # Pooling strength parameter
        kappa = pm.Exponential("kappa", lam=1 / kappa_prior_scale)

        # City-level matrices with partial pooling
        if n_cities > 0:
            M_cities_list = []
            for c in range(n_cities):
                city_cols = []
                for j in range(K):
                    city_col = pm.Dirichlet(
                        f"M_city_{c}_col_{j}", a=kappa * M_country_cols[j]
                    )
                    city_cols.append(city_col)
                M_city = pm.math.stack(city_cols, axis=1)  # Shape: (K, K)
                M_cities_list.append(M_city)

            M_cities = pm.math.stack(M_cities_list, axis=0)  # Shape: (n_cities, K, K)

        # Overdispersion parameter
        phi = pm.Exponential("phi", lam=0.1)

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

            # City-specific transition matrix
            M_city = M_cities[i]  # Shape: (K, K)

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

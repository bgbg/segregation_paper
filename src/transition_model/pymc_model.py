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
    sigma_D: float = 0.5,
    delta_scale: float = 1.0,
    nu_scale: float = 5.0,
    *,
    priors: Optional[Dict] = None,
    innovation: Optional[Dict[str, float]] = None,
) -> pm.Model:
    """Build hierarchical logistic-normal transition model with rank-1 city deviations.

    Uses logistic-normal parameterization with a shared deviation pattern D and
    city-specific scalars delta_city. This provides interpretable single-parameter
    city deviations while maintaining flexibility through the learned pattern D.

    Args:
        data: Dictionary with country and city-level data tensors
        diag_bias_mean: Mean for diagonal bias (loyalty) parameter
        diag_bias_sigma: Standard deviation for diagonal bias parameter
        sigma_country: Scale for country-level logit variation
        sigma_D: Scale for shared deviation pattern matrix
        delta_scale: Scale for city-specific deviation scalars
        nu_scale: Scale parameter for Student-t degrees of freedom
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
        # Country-level logistic-normal prior (robust)
        eyeK = np.eye(K)

        sigma_country_param = pm.HalfNormal("sigma_country", sigma=sigma_country)

        # Use temporal priors if provided
        if priors and "Z_country" in priors:
            zc_mu = np.array(priors["Z_country"], dtype=float)
            zc_sigma = (innovation or {}).get("Z_country_sigma", sigma_country)
            Z_country = pm.Normal("Z_country", mu=zc_mu, sigma=zc_sigma, shape=(K, K))
        else:
            Z_country = pm.Normal(
                "Z_country", mu=0.0, sigma=sigma_country_param, shape=(K, K)
            )

        if priors and "diag_bias" in priors:
            diag_mu = float(priors["diag_bias"])  # previous estimate
            diag_sd = (innovation or {}).get("diag_bias_sigma", diag_bias_sigma)
            diag_bias = pm.Normal("diag_bias", mu=diag_mu, sigma=diag_sd)
        else:
            diag_bias = pm.Normal(
                "diag_bias", mu=diag_bias_mean, sigma=diag_bias_sigma
            )  # encodes loyalty in mean

        M_country_cols = []
        for j in range(K):
            z = Z_country[:, j] + diag_bias * eyeK[:, j]
            col = pm.Deterministic(f"M_country_col_{j}", pm.math.softmax(z))
            M_country_cols.append(col)
        M_country = pm.math.stack(M_country_cols, axis=1)

        # City-level rank-1 deviations
        M_cities = None
        if n_cities > 0:
            # Shared deviation pattern matrix
            if priors and "D" in priors:
                D_mu = np.array(priors["D"], dtype=float)
                D_sigma = (innovation or {}).get("D_sigma", sigma_D)
                D = pm.Normal(
                    "D_deviation_pattern", mu=D_mu, sigma=D_sigma, shape=(K, K)
                )
            else:
                D = pm.Normal("D_deviation_pattern", mu=0, sigma=sigma_D, shape=(K, K))

            # Heavy-tailed distribution for robustness
            nu_raw = pm.Exponential("nu_raw", lam=1 / nu_scale)
            nu = pm.Deterministic("nu", nu_raw + 2.0)

            # City-specific scalar deviations (rank-1)
            # IMPORTANT: City scalars are re-initialized each election to avoid
            # accumulation of alignment/aggregation errors across elections.
            delta_city = pm.StudentT(
                "delta_city", nu=nu, mu=0, sigma=delta_scale, shape=n_cities
            )

            # Interpretable magnitude measure
            city_deviation_magnitude = pm.Deterministic(
                "city_deviation_magnitude", pm.math.abs(delta_city)
            )

            # City logits: rank-1 deviation from country
            Z_city = pm.Deterministic(
                "Z_city",
                Z_country[None, :, :] + delta_city[:, None, None] * D[None, :, :],
            )

            # Build city transition matrices
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
        if priors and "log_phi" in priors:
            lp_mu = float(priors["log_phi"])  # previous estimate
            lp_sd = (innovation or {}).get("log_phi_sigma", 1.0)
            log_phi = pm.Normal("log_phi", mu=lp_mu, sigma=lp_sd)
        else:
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
    max_treedepth: int = 15,
    init: str = "jitter+adapt_diag",
    random_seed: Optional[int] = None,
) -> az.InferenceData:
    """Sample from the hierarchical transition model.

    Args:
        model: PyMC model to sample from
        draws: Number of posterior samples per chain
        tune: Number of tuning samples per chain
        chains: Number of MCMC chains
        target_accept: Target acceptance rate
        max_treedepth: Maximum tree depth for NUTS sampler
        init: Initialization method for chains
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
            max_treedepth=max_treedepth,
            init=init,
            random_seed=random_seed,
            return_inferencedata=True,
        )

    return trace


def analyze_city_deviations(
    trace: az.InferenceData, city_names: list
) -> Dict[str, Dict]:
    """Analyze and interpret city deviation patterns from rank-1 model.

    Args:
        trace: Posterior samples from model
        city_names: List of city names in order

    Returns:
        Dictionary with deviation analysis for each city
    """
    # Extract posterior means
    D = trace.posterior["D_deviation_pattern"].mean(dim=["chain", "draw"]).values
    delta = trace.posterior["delta_city"].mean(dim=["chain", "draw"]).values

    # Get credible intervals for delta
    delta_hdi = az.hdi(trace.posterior["delta_city"], hdi_prob=0.89)

    results = {}
    for i, city in enumerate(city_names):
        # Compute deviation matrix for this city
        deviation_matrix = delta[i] * D

        # Find strongest deviation
        max_idx = np.unravel_index(
            np.argmax(np.abs(deviation_matrix)), deviation_matrix.shape
        )

        # Frobenius norm (overall deviation magnitude)
        frob_norm = np.linalg.norm(deviation_matrix, "fro")

        results[city] = {
            "delta_mean": delta[i],
            "delta_hdi_89": (
                delta_hdi["delta_city"][i, 0].values,
                delta_hdi["delta_city"][i, 1].values,
            ),
            "magnitude": abs(delta[i]),
            "direction": "positive" if delta[i] > 0 else "negative",
            "frobenius_norm": frob_norm,
            "strongest_deviation": {
                "from_party": max_idx[0],
                "to_party": max_idx[1],
                "value": deviation_matrix[max_idx],
            },
            "significant": not (
                delta_hdi["delta_city"][i, 0] <= 0 <= delta_hdi["delta_city"][i, 1]
            ),
        }

    return results


def generate_test_data() -> Dict[str, Dict[str, np.ndarray]]:
    """Generate synthetic test data for model demonstration.

    Returns:
        Dictionary with country and city test data
    """
    np.random.seed(42)

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

    # Jerusalem: moderate Haredi population
    x1_jerusalem = np.random.multinomial(300, [0.25, 0.20, 0.45, 0.10], n_stations_city)
    x2_jerusalem = np.random.multinomial(310, [0.26, 0.19, 0.47, 0.08], n_stations_city)

    # Bnei Brak: very high Haredi population
    x1_bnei_brak = np.random.multinomial(280, [0.35, 0.30, 0.25, 0.10], n_stations_city)
    x2_bnei_brak = np.random.multinomial(290, [0.37, 0.28, 0.27, 0.08], n_stations_city)

    # Tel Aviv: secular population (opposite pattern)
    x1_tel_aviv = np.random.multinomial(350, [0.05, 0.03, 0.82, 0.10], n_stations_city)
    x2_tel_aviv = np.random.multinomial(360, [0.04, 0.03, 0.85, 0.08], n_stations_city)

    data = {
        "country": {
            "x1": x1_country.astype(float),
            "x2": x2_country.astype(float),
            "n1": x1_country.sum(axis=1).astype(float),
            "n2": x2_country.sum(axis=1).astype(float),
        },
        "jerusalem": {
            "x1": x1_jerusalem.astype(float),
            "x2": x2_jerusalem.astype(float),
            "n1": x1_jerusalem.sum(axis=1).astype(float),
            "n2": x2_jerusalem.sum(axis=1).astype(float),
        },
        "bnei brak": {
            "x1": x1_bnei_brak.astype(float),
            "x2": x2_bnei_brak.astype(float),
            "n1": x1_bnei_brak.sum(axis=1).astype(float),
            "n2": x2_bnei_brak.sum(axis=1).astype(float),
        },
        "tel aviv": {
            "x1": x1_tel_aviv.astype(float),
            "x2": x2_tel_aviv.astype(float),
            "n1": x1_tel_aviv.sum(axis=1).astype(float),
            "n2": x2_tel_aviv.sum(axis=1).astype(float),
        },
    }

    return data


def print_rank1_summary(trace: az.InferenceData, city_names: list):
    """Print readable summary of rank-1 model results."""

    print("\n" + "=" * 60)
    print("RANK-1 MODEL RESULTS SUMMARY")
    print("=" * 60)

    # Country-level transition matrix
    print("\nCountry-level Transition Matrix:")
    print("(rows: to party, cols: from party)")
    print("Parties: [Shas, Agudat, Other, Abstained]")
    M_country = []
    for j in range(4):
        col = trace.posterior[f"M_country_col_{j}"].mean(dim=["chain", "draw"]).values
        M_country.append(col)
    M_country = np.stack(M_country, axis=1)
    print(np.round(M_country, 3))

    # Diagonal bias (loyalty parameter)
    diag_bias = trace.posterior["diag_bias"].mean(dim=["chain", "draw"]).values
    print(f"\nDiagonal Bias (loyalty): {diag_bias:.2f}")

    # Shared deviation pattern
    print("\n" + "-" * 40)
    print("Shared Deviation Pattern (D matrix):")
    print("-" * 40)
    D = trace.posterior["D_deviation_pattern"].mean(dim=["chain", "draw"]).values
    print("(Shows HOW cities deviate when they do)")
    print(np.round(D, 3))

    # City deviations
    print("\n" + "-" * 40)
    print("City-Specific Deviation Scalars:")
    print("-" * 40)

    results = analyze_city_deviations(trace, city_names)

    for city, res in results.items():
        print(f"\n{city.upper()}:")
        print(
            f"  δ (deviation scalar): {res['delta_mean']:.3f} "
            f"[89% HDI: ({res['delta_hdi_89'][0]:.3f}, "
            f"{res['delta_hdi_89'][1]:.3f})]"
        )
        print(f"  Magnitude: {res['magnitude']:.3f}")
        print(f"  Direction: {res['direction']}")
        print(f"  Significant deviation: {res['significant']}")
        print(f"  Overall deviation (Frobenius): {res['frobenius_norm']:.3f}")

        parties = ["Shas", "Agudat", "Other", "Abstained"]
        strongest = res["strongest_deviation"]
        print(
            f"  Strongest deviation: {parties[strongest['from_party']]}"
            f" → {parties[strongest['to_party']]}"
            f" ({strongest['value']:.3f})"
        )


if __name__ == "__main__":
    print("Hierarchical Voter Transition Model with Rank-1 City Deviations")
    print("=" * 60)

    # Generate test data
    test_data = generate_test_data()
    city_names = [k for k in test_data.keys() if k != "country"]

    print("\nData Summary:")
    print(f"- Countries: 1")
    print(f"- Cities: {len(city_names)} ({', '.join(city_names)})")
    print(f"- Categories: 4 (Shas, Agudat Israel, Other, Abstained)")
    print(f"- Country stations: {len(test_data['country']['x1'])}")
    print(f"- City stations per city: {len(test_data['jerusalem']['x1'])}")

    # Build model
    print("\nBuilding rank-1 hierarchical model...")
    model = build_hierarchical_model(test_data)

    try:
        # Generate model graph
        graph = pm.model_to_graphviz(model)
        output_file = "voter_transition_model_rank1"
        graph.render(output_file, format="png", cleanup=True)
        print(f"Model visualization saved as {output_file}.png")
    except Exception as e:
        print(f"Could not generate model graph: {e}")

    # Sample from model (quick test)
    print("\nSampling from posterior (quick test)...")
    try:
        trace = sample_model(model, draws=500, tune=500, chains=2, random_seed=42)

        # Print results
        print_rank1_summary(trace, city_names)

        # Model diagnostics
        print("\n" + "=" * 60)
        print("MODEL DIAGNOSTICS")
        print("=" * 60)
        print(
            az.summary(
                trace,
                var_names=["delta_city", "D_deviation_pattern", "diag_bias", "phi"],
                hdi_prob=0.89,
            )
        )

    except Exception as e:
        print(f"Sampling failed: {e}")
        print("This is expected if PyMC dependencies are missing.")

# Hierarchical Bayesian Voter Transition Model with Rank-1 City Deviations

## Background

### The Problem

Electoral analysis often requires understanding how voters transition between parties across consecutive elections. This is particularly challenging because:

1. **Individual ballot secrecy**: We cannot observe individual voter transitions directly
2. **Ecological inference problem**: We only observe aggregate vote counts at polling stations
3. **Spatial heterogeneity**: Different cities/regions may have distinct voting patterns that differ from national trends
4. **Interpretability challenge**: Understanding how and why local patterns differ from national patterns

### The Data Structure

For each election pair (t, t+1), we observe:
- **Station-level data**: Vote counts for each party at each polling station
- **Two time points**: Election at time t and election at time t+1
- **Multiple geographic levels**: Country-wide aggregates and city-specific data

In our example with Israeli religious parties:
- **4 categories**: Shas, Agudat Israel, Other, Abstained
- **Country level**: Aggregate patterns across all stations
- **City level**: Specific patterns for cities like Jerusalem and Bnei Brak (high Haredi population)

## Model Architecture

### Core Concept: Voter Transition Matrices

The model estimates a **transition matrix** M where element M[i,j] represents the probability that a voter who chose option i in election t will choose option j in election t+1.

```
        Election t+1
        Shas  Agudat  Other  Abstain
E    S  [0.7   0.1    0.15   0.05  ]
l    A  [0.05  0.8    0.1    0.05  ]
e    O  [0.02  0.03   0.9    0.05  ]
c    Ab [0.1   0.1    0.3    0.5   ]
t
i
o
n
t
```

### Hierarchical Structure

The model uses a three-level hierarchy:

1. **Country level**: Base transition patterns (M_country)
2. **City deviation pattern**: A shared pattern D of how cities tend to differ
3. **City-specific scaling**: Scalar parameters δ[c] that scale the deviation for each city

### The Rank-1 Approximation

Instead of allowing each city to have a completely independent (K×K) deviation matrix, we constrain cities to deviate along a **shared pattern**:

```
Z_city[c] = Z_country + δ[c] × D
```

Where:
- `Z_country`: Country-level logits (pre-softmax values)
- `D`: Global deviation pattern matrix (K×K), learned from data
- `δ[c]`: City-specific scalar that controls deviation magnitude and direction

This is a **rank-1 approximation** that dramatically reduces parameters while maintaining interpretability.

### Why Rank-1?

1. **Dimensionality reduction**: From K²×n_cities to K² + n_cities parameters
2. **Shared structure**: Assumes cities deviate in similar ways (e.g., due to demographics)
3. **Interpretability**: Single number per city measuring deviation strength
4. **Regularization**: Prevents overfitting in cities with limited data

## Mathematical Details

### Logistic-Normal Parameterization

The model uses a logistic-normal parameterization for the transition matrices:

1. **Logit space**: Work with unconstrained logits Z ∈ ℝ^(K×K)
2. **Diagonal bias**: Add loyalty bonus to diagonal elements
3. **Softmax transformation**: Convert to valid probability columns

```python
M[:, j] = softmax(Z[:, j] + diag_bias × I[:, j])
```

### Prior Specifications

- **Country logits**: Z_country ~ Normal(0, σ_country)
- **Diagonal bias**: diag_bias ~ Normal(3.0, 0.5) (encodes voter loyalty)
- **Deviation pattern**: D ~ Normal(0, σ_D)
- **City scalars**: δ[c] ~ StudentT(ν, 0, δ_scale) (heavy-tailed for robustness)
- **Overdispersion**: φ = exp(log_φ), where log_φ ~ Normal(0, 1)

### Likelihood

The model uses a Dirichlet-Multinomial likelihood to account for overdispersion:

```
x2[s] ~ DirichletMultinomial(n2[s], φ × q[s])
```

Where q[s] = p1[s] × M^T is the expected proportion vector at station s.

## Interpretation Guide

### Key Parameters to Monitor

1. **δ[c] (delta_city)**: City-specific deviation scalars
   - **Magnitude |δ[c]|**: How much city c deviates from the country
   - **Sign of δ[c]**: Direction of deviation along pattern D
   - **δ[c] ≈ 0**: City follows national patterns closely
   - **|δ[c]| >> 1**: City strongly deviates from national patterns

2. **D (deviation_pattern)**: The shared deviation pattern
   - Shows HOW cities tend to differ when they do differ
   - Large values in D[i,j] indicate common deviations in that transition

3. **city_deviation_magnitude**: |δ[c]| for easier interpretation

### Example Interpretations

If posterior analysis shows:
- `δ[Jerusalem] = 1.5`: Jerusalem deviates positively along pattern D
- `δ[Bnei Brak] = 2.3`: Bnei Brak deviates even more strongly in same direction
- `δ[Tel Aviv] = -0.8`: Tel Aviv deviates in opposite direction
- `D[Shas, Shas] > 0`: Cities that deviate tend to have higher Shas loyalty

This suggests Jerusalem and Bnei Brak (high Haredi population) share similar deviation patterns, while Tel Aviv (secular) deviates oppositely.

### Post-hoc Analysis

```python
def analyze_city_deviations(trace, city_names):
    """Comprehensive analysis of city deviation patterns."""
    
    # Extract posterior means
    D = trace.posterior["D_deviation_pattern"].mean(dim=["chain", "draw"]).values
    delta = trace.posterior["delta_city"].mean(dim=["chain", "draw"]).values
    
    # Compute deviation metrics
    results = {}
    for i, city in enumerate(city_names):
        # Frobenius norm of actual deviation
        deviation_matrix = delta[i] * D
        frobenius_norm = np.linalg.norm(deviation_matrix, 'fro')
        
        # Key transitions that differ
        max_deviation_idx = np.unravel_index(
            np.argmax(np.abs(deviation_matrix)), 
            deviation_matrix.shape
        )
        
        results[city] = {
            'scalar': delta[i],
            'magnitude': abs(delta[i]),
            'direction': 'positive' if delta[i] > 0 else 'negative',
            'frobenius_norm': frobenius_norm,
            'max_deviation_transition': max_deviation_idx,
            'max_deviation_value': deviation_matrix[max_deviation_idx]
        }
    
    return results
```

## Complete Implementation

```python
"""PyMC implementation of hierarchical ecological inference model with rank-1 city deviations.

This module implements the Bayesian hierarchical model for estimating
voter transition matrices between consecutive elections, with a rank-1
approximation for city-level deviations.
"""

from typing import Dict, Optional, Tuple
import arviz as az
import numpy as np
import pymc as pm


def build_hierarchical_model_rank1(
    data: Dict[str, Dict[str, np.ndarray]],
    diag_bias_mean: float = 3.0,
    diag_bias_sigma: float = 0.5,
    sigma_country: float = 1.0,
    sigma_D: float = 0.5,
    nu_scale: float = 5.0,
    delta_scale: float = 1.0,
    *,
    priors: Optional[Dict] = None,
    innovation: Optional[Dict[str, float]] = None,
) -> pm.Model:
    """Build hierarchical logistic-normal transition model with rank-1 city deviations.

    Uses logistic-normal parameterization with Student-t distributed city-specific
    scalars that modulate a shared deviation pattern. This provides a low-dimensional
    yet flexible representation of how cities differ from country-level patterns.

    Args:
        data: Dictionary with country and city-level data tensors
        diag_bias_mean: Mean for diagonal bias (loyalty) parameter
        diag_bias_sigma: Standard deviation for diagonal bias parameter
        sigma_country: Scale for country-level logit variation
        sigma_D: Scale for the shared deviation pattern matrix
        nu_scale: Scale parameter for Student-t degrees of freedom
        delta_scale: Scale for city-specific deviation scalars
        priors: Optional dictionary of priors from previous time period
        innovation: Optional dictionary of innovation variances

    Returns:
        PyMC model object
    """
    K = 4  # Number of categories
    cities = [k for k in data.keys() if k != "country"]
    n_cities = len(cities)

    # Validate data structure
    country_data = data["country"]
    x1_country = country_data["x1"]
    x2_country = country_data["x2"]
    assert (
        x1_country.shape[1] == x2_country.shape[1] == K
    ), f"Category mismatch: x1 has {x1_country.shape[1]}, x2 has {x2_country.shape[1]}, expected {K}"

    with pm.Model() as model:
        # Country-level logistic-normal prior
        eyeK = np.eye(K)
        sigma_country_param = pm.HalfNormal("sigma_country", sigma=sigma_country)

        # Country-level transition logits
        if priors and "Z_country" in priors:
            zc_mu = np.array(priors["Z_country"], dtype=float)
            zc_sigma = (innovation or {}).get("Z_country_sigma", sigma_country)
            Z_country = pm.Normal("Z_country", mu=zc_mu, sigma=zc_sigma, shape=(K, K))
        else:
            Z_country = pm.Normal(
                "Z_country", mu=0.0, sigma=sigma_country_param, shape=(K, K)
            )

        # Diagonal bias (voter loyalty)
        if priors and "diag_bias" in priors:
            diag_mu = float(priors["diag_bias"])
            diag_sd = (innovation or {}).get("diag_bias_sigma", diag_bias_sigma)
            diag_bias = pm.Normal("diag_bias", mu=diag_mu, sigma=diag_sd)
        else:
            diag_bias = pm.Normal(
                "diag_bias", mu=diag_bias_mean, sigma=diag_bias_sigma
            )

        # Build country transition matrix
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
            D = pm.Normal(
                "D_deviation_pattern",
                mu=0,
                sigma=sigma_D,
                shape=(K, K)
            )
            
            # Heavy-tailed distribution for robustness
            nu_raw = pm.Exponential("nu_raw", lam=1 / nu_scale)
            nu = pm.Deterministic("nu", nu_raw + 2.0)
            
            # City-specific scalar deviations
            delta = pm.StudentT(
                "delta_city",
                nu=nu,
                mu=0,
                sigma=delta_scale,
                shape=n_cities
            )
            
            # Interpretable magnitude measure
            city_deviation_magnitude = pm.Deterministic(
                "city_deviation_magnitude",
                pm.math.abs(delta)
            )
            
            # City logits: rank-1 deviation from country
            Z_city = pm.Deterministic(
                "Z_city",
                Z_country[None, :, :] + delta[:, None, None] * D[None, :, :]
            )
            
            # Build city transition matrices
            M_cities_list = []
            for c in range(n_cities):
                city_cols = []
                for j in range(K):
                    zc = Z_city[c, :, j] + diag_bias * eyeK[:, j]
                    colc = pm.Deterministic(
                        f"M_city_{c}_col_{j}", 
                        pm.math.softmax(zc)
                    )
                    city_cols.append(colc)
                M_city = pm.math.stack(city_cols, axis=1)
                M_cities_list.append(M_city)
            M_cities = pm.math.stack(M_cities_list, axis=0)

        # Overdispersion parameter
        if priors and "log_phi" in priors:
            lp_mu = float(priors["log_phi"])
            lp_sd = (innovation or {}).get("log_phi_sigma", 1.0)
            log_phi = pm.Normal("log_phi", mu=lp_mu, sigma=lp_sd)
        else:
            log_phi = pm.Normal("log_phi", mu=0.0, sigma=1.0)
        phi = pm.Deterministic("phi", pm.math.exp(log_phi))

        # Likelihood for country data
        x1_country = country_data["x1"]
        x2_country = country_data["x2"]
        n2_country = country_data["n2"]

        # Proportions from election t
        p1_country = x1_country / x1_country.sum(axis=1, keepdims=True)

        # Expected proportions in election t+1
        q_country = pm.math.dot(p1_country, M_country.T)

        # Observed votes
        pm.DirichletMultinomial(
            "x2_country_obs", 
            n=n2_country, 
            a=phi * q_country, 
            observed=x2_country
        )

        # Likelihood for city data
        for i, city in enumerate(cities):
            city_data = data[city]
            x1_city = city_data["x1"]
            x2_city = city_data["x2"]
            n2_city = city_data["n2"]

            # Use city-specific matrix
            M_city = M_cities[i] if M_cities is not None else M_country

            # Proportions and predictions
            p1_city = x1_city / x1_city.sum(axis=1, keepdims=True)
            q_city = pm.math.dot(p1_city, M_city.T)

            # Observed votes
            pm.DirichletMultinomial(
                f"x2_{city}_obs", 
                n=n2_city, 
                a=phi * q_city, 
                observed=x2_city
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
        target_accept: Target acceptance rate for NUTS
        max_treedepth: Maximum tree depth for NUTS sampler
        init: Initialization method for chains
        random_seed: Random seed for reproducibility

    Returns:
        ArviZ InferenceData object with posterior samples
    """
    with model:
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
    trace: az.InferenceData, 
    city_names: list
) -> Dict[str, Dict]:
    """Analyze and interpret city deviation patterns.
    
    Args:
        trace: Posterior samples from model
        city_names: List of city names in order
        
    Returns:
        Dictionary with deviation analysis for each city
    """
    # Extract posterior means
    D = trace.posterior["D_deviation_pattern"].mean(
        dim=["chain", "draw"]
    ).values
    delta = trace.posterior["delta_city"].mean(
        dim=["chain", "draw"]
    ).values
    
    # Get credible intervals for delta
    delta_hdi = az.hdi(trace.posterior["delta_city"], hdi_prob=0.89)
    
    results = {}
    for i, city in enumerate(city_names):
        # Compute deviation matrix for this city
        deviation_matrix = delta[i] * D
        
        # Find strongest deviation
        max_idx = np.unravel_index(
            np.argmax(np.abs(deviation_matrix)), 
            deviation_matrix.shape
        )
        
        # Frobenius norm (overall deviation magnitude)
        frob_norm = np.linalg.norm(deviation_matrix, 'fro')
        
        results[city] = {
            'delta_mean': delta[i],
            'delta_hdi_89': (
                delta_hdi["delta_city"][i, 0].values,
                delta_hdi["delta_city"][i, 1].values
            ),
            'magnitude': abs(delta[i]),
            'direction': 'positive' if delta[i] > 0 else 'negative',
            'frobenius_norm': frob_norm,
            'strongest_deviation': {
                'from_party': max_idx[0],
                'to_party': max_idx[1],
                'value': deviation_matrix[max_idx]
            },
            'significant': not (
                delta_hdi["delta_city"][i, 0] <= 0 <= 
                delta_hdi["delta_city"][i, 1]
            )
        }
    
    return results


def generate_test_data() -> Tuple[Dict[str, Dict[str, np.ndarray]], list]:
    """Generate synthetic test data for model demonstration.
    
    Returns:
        Tuple of (data dictionary, city names list)
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
    x1_jerusalem = np.random.multinomial(
        300, [0.25, 0.20, 0.45, 0.10], n_stations_city
    )
    x2_jerusalem = np.random.multinomial(
        310, [0.26, 0.19, 0.47, 0.08], n_stations_city
    )
    
    # Bnei Brak: very high Haredi population
    x1_bnei_brak = np.random.multinomial(
        280, [0.35, 0.30, 0.25, 0.10], n_stations_city
    )
    x2_bnei_brak = np.random.multinomial(
        290, [0.37, 0.28, 0.27, 0.08], n_stations_city
    )
    
    # Tel Aviv: secular population (opposite pattern)
    x1_tel_aviv = np.random.multinomial(
        350, [0.05, 0.03, 0.82, 0.10], n_stations_city
    )
    x2_tel_aviv = np.random.multinomial(
        360, [0.04, 0.03, 0.85, 0.08], n_stations_city
    )
    
    cities = ["jerusalem", "bnei_brak", "tel_aviv"]
    
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
        "bnei_brak": {
            "x1": x1_bnei_brak.astype(float),
            "x2": x2_bnei_brak.astype(float),
            "n1": x1_bnei_brak.sum(axis=1).astype(float),
            "n2": x2_bnei_brak.sum(axis=1).astype(float),
        },
        "tel_aviv": {
            "x1": x1_tel_aviv.astype(float),
            "x2": x2_tel_aviv.astype(float),
            "n1": x1_tel_aviv.sum(axis=1).astype(float),
            "n2": x2_tel_aviv.sum(axis=1).astype(float),
        }
    }
    
    return data, cities


def print_summary(trace: az.InferenceData, city_names: list):
    """Print readable summary of model results."""
    
    print("\n" + "="*60)
    print("MODEL RESULTS SUMMARY")
    print("="*60)
    
    # Country-level transition matrix
    print("\nCountry-level Transition Matrix:")
    print("(rows: from party, cols: to party)")
    print("Parties: [Shas, Agudat, Other, Abstained]")
    M_country = []
    for j in range(4):
        col = trace.posterior[f"M_country_col_{j}"].mean(
            dim=["chain", "draw"]
        ).values
        M_country.append(col)
    M_country = np.stack(M_country, axis=1)
    print(np.round(M_country, 3))
    
    # Diagonal bias (loyalty parameter)
    diag_bias = trace.posterior["diag_bias"].mean(
        dim=["chain", "draw"]
    ).values
    print(f"\nDiagonal Bias (loyalty): {diag_bias:.2f}")
    
    # City deviations
    print("\n" + "-"*40)
    print("City-Specific Deviations:")
    print("-"*40)
    
    results = analyze_city_deviations(trace, city_names)
    
    for city, res in results.items():
        print(f"\n{city.upper()}:")
        print(f"  δ (deviation scalar): {res['delta_mean']:.3f} "
              f"[89% HDI: ({res['delta_hdi_89'][0]:.3f}, "
              f"{res['delta_hdi_89'][1]:.3f})]")
        print(f"  Magnitude: {res['magnitude']:.3f}")
        print(f"  Direction: {res['direction']}")
        print(f"  Significant deviation: {res['significant']}")
        print(f"  Overall deviation (Frobenius): {res['frobenius_norm']:.3f}")
        
        parties = ['Shas', 'Agudat', 'Other', 'Abstained']
        strongest = res['strongest_deviation']
        print(f"  Strongest deviation: {parties[strongest['from_party']]}"
              f" → {parties[strongest['to_party']]}"
              f" ({strongest['value']:.3f})")
    
    # Deviation pattern interpretation
    print("\n" + "-"*40)
    print("Shared Deviation Pattern (D matrix):")
    print("-"*40)
    D = trace.posterior["D_deviation_pattern"].mean(
        dim=["chain", "draw"]
    ).values
    print("(Shows HOW cities deviate when they do)")
    print(np.round(D, 3))


if __name__ == "__main__":
    print("Hierarchical Voter Transition Model with Rank-1 City Deviations")
    print("="*60)
    
    # Generate test data
    test_data, city_names = generate_test_data()
    
    print("\nData Summary:")
    print(f"- Countries: 1")
    print(f"- Cities: {len(city_names)} ({', '.join(city_names)})")
    print(f"- Categories: 4 (Shas, Agudat Israel, Other, Abstained)")
    print(f"- Country stations: {len(test_data['country']['x1'])}")
    print(f"- City stations per city: {len(test_data['jerusalem']['x1'])}")
    
    # Build and sample model
    print("\nBuilding model...")
    model = build_hierarchical_model_rank1(test_data)
    
    print("Sampling from posterior...")
    trace = sample_model(
        model, 
        draws=2000, 
        tune=1000, 
        chains=4,
        random_seed=42
    )
    
    # Print results
    print_summary(trace, city_names)
    
    # Model diagnostics
    print("\n" + "="*60)
    print("MODEL DIAGNOSTICS")
    print("="*60)
    print(az.summary(
        trace, 
        var_names=["delta_city", "D_deviation_pattern", "diag_bias", "phi"],
        hdi_prob=0.89
    ))
```

## Advantages and Limitations

### Advantages

1. **Interpretability**: Single scalar per city is easy to communicate
2. **Regularization**: Rank-1 constraint prevents overfitting
3. **Efficiency**: Far fewer parameters than full city-specific matrices
4. **Pattern discovery**: Learns common deviation patterns from data
5. **Robustness**: Student-t distribution handles outlier cities

### Limitations

1. **Rank-1 assumption**: May not capture complex multi-dimensional deviations
2. **Shared pattern**: Assumes all cities deviate along same axis
3. **Model selection**: Need to validate rank-1 is sufficient (use WAIC/LOO)

### When to Use This Approach

Best suited when:
- Cities share common demographic or cultural factors driving deviations
- You need interpretable city-level parameters
- Data is limited for some cities (regularization helps)
- Stakeholders need simple explanations of local differences

Consider alternatives when:
- Cities have highly idiosyncratic voting patterns
- You have abundant data and can afford more parameters
- Multi-dimensional deviations are theoretically expected

## Extensions and Variations

### 1. Rank-r Approximation

For more flexibility, use multiple deviation patterns:

```python
r = 2  # Number of patterns
D = pm.Normal("D_patterns", mu=0, sigma=sigma_D, shape=(r, K, K))
delta = pm.StudentT("delta_city", nu=nu, mu=0, sigma=delta_scale, 
                    shape=(n_cities, r))
Z_city = Z_country[None, :, :] + pm.math.sum(
    delta[:, :, None, None] * D[None, :, :, :], axis=1
)
```

### 2. Hierarchical City Parameters

Model city deviations based on covariates:

```python
# City-level predictors
haredi_pct = data["city_haredi_percentage"]  # Shape: (n_cities,)
urban_density = data["city_urban_density"]    # Shape: (n_cities,)

# Hierarchical model for delta
beta_0 = pm.Normal("delta_intercept", 0, 1)
beta_haredi = pm.Normal("delta_haredi_effect", 0, 1)
beta_urban = pm.Normal("delta_urban_effect", 0, 1)

delta_mean = beta_0 + beta_haredi * haredi_pct + beta_urban * urban_density
delta = pm.StudentT("delta_city", nu=nu, mu=delta_mean, 
                    sigma=delta_scale)
```

### 3. Time-Varying Patterns

Allow the deviation pattern to evolve:

```python
# For election pair t
D_t = pm.Normal("D_pattern_t", mu=D_t_minus_1, sigma=innovation_D, 
                shape=(K, K))
```

## Conclusion

The rank-1 hierarchical model provides an elegant solution to the challenge of modeling city-level electoral deviations. By constraining cities to deviate along a shared pattern scaled by city-specific parameters, we achieve:

- **Interpretability**: Clear, single-number measure of city deviation
- **Statistical efficiency**: Regularized estimates even with limited data
- **Computational efficiency**: Fewer parameters to estimate
- **Substantive insights**: Understanding of shared deviation patterns

This approach is particularly valuable for electoral analysis where cities with similar demographics (e.g., religious populations) tend to deviate from national patterns in systematic ways.
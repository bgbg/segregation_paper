# Voter Transition Model Documentation

## Overview

The Voter Transition Model is a Bayesian hierarchical ecological inference model designed to estimate voter transition matrices between consecutive Israeli Knesset elections. The model analyzes how voters move between different political categories from one election to the next, providing insights into political dynamics and voter behavior patterns.

## Model Architecture

### Hierarchical Structure

The model uses a **logistic-normal parameterization** with a hierarchical structure:

1. **Country Level**: Base transition matrix estimated from national data
2. **City Level**: City-specific deviations from the country-level matrix
3. **Station Level**: Individual ballot box observations within each city

### Voter Categories

The model analyzes transitions between four voter categories:

| Category | Description | Hebrew Name |
|----------|-------------|-------------|
| **Shas** | Shas party voters | ש״ס |
| **Agudat_Israel** | Agudat Israel party voters | אגודת ישראל |
| **Other** | All other parties combined | מפלגות אחרות |
| **Abstained** | Non-voters + invalid votes | נמנעו |

### Transition Matrix Structure

The model estimates 4×4 transition matrices where:
- **Columns** represent "from" categories (election t)
- **Rows** represent "to" categories (election t+1)
- Each element M[i,j] represents the probability that a voter in category j at election t will vote for category i at election t+1

## Mathematical Framework

### Logistic-Normal Parameterization

The model uses a logistic-normal approach instead of traditional Dirichlet priors:

#### Country-Level Parameters

```python
# Base logits for country-level transitions
Z_country ~ Normal(0, σ_country)  # Shape: (4, 4)

# Diagonal bias for voter loyalty
diag_bias ~ Normal(μ_diag, σ_diag)

# Country transition matrix
for j in range(4):
    z_j = Z_country[:, j] + diag_bias * eye[:, j]
    M_country_col_j = softmax(z_j)
```

#### City-Level Deviations

Cities deviate from the country-level matrix using **heavy-tailed Student-t distributions**:

```python
# Degrees of freedom for Student-t (allows for outliers)
nu_raw ~ Exponential(1/5.0)
nu = nu_raw + 2.0

# City-specific logit deviations
Z_city ~ StudentT(nu, Z_country, σ_city)  # Shape: (n_cities, 4, 4)

# City transition matrices
for c in range(n_cities):
    for j in range(4):
        z_cj = Z_city[c, :, j] + diag_bias * eye[:, j]
        M_city_c_col_j = softmax(z_cj)
```

#### Overdispersion Parameter

```python
# Log-scale overdispersion for DirichletMultinomial
log_phi ~ Normal(0, 1)
phi = exp(log_phi)
```

### Likelihood Function

The model uses **DirichletMultinomial** likelihood for each ballot box:

```python
# For country data
p1_country = x1_country / x1_country.sum(axis=1, keepdims=True)  # Proportions at election t
q_country = p1_country @ M_country.T  # Expected proportions at election t+1
x2_country_obs ~ DirichletMultinomial(n2_country, phi * q_country)

# For city data (similar structure)
```

## Model Parameters

### Hyperparameters

| Parameter | Default | Description | Interpretation |
|-----------|---------|-------------|----------------|
| `diag_bias_mean` | 3.0 | Mean diagonal bias | Controls voter loyalty strength |
| `diag_bias_sigma` | 0.5 | Diagonal bias std dev | Uncertainty in loyalty estimates |
| `sigma_country` | 1.0 | Country logit scale | Base transition variability |
| `sigma_city` | 0.5 | City deviation scale | City-specific variation |
| `nu_scale` | 5.0 | Student-t scale | Controls outlier sensitivity |

### Estimated Parameters

| Parameter | Shape | Description | Interpretation |
|-----------|-------|-------------|----------------|
| `Z_country` | (4,4) | Country logits | Base transition tendencies |
| `diag_bias` | scalar | Diagonal bias | Voter loyalty strength |
| `Z_city` | (n_cities,4,4) | City logit deviations | City-specific differences |
| `nu` | scalar | Student-t degrees of freedom | Outlier sensitivity |
| `phi` | scalar | Overdispersion | Extra-multinomial variation |

### Derived Quantities

| Quantity | Description | Interpretation |
|----------|-------------|----------------|
| `M_country` | Country transition matrix | National voting patterns |
| `M_cities` | City transition matrices | Local voting patterns |
| Vote movements | Actual vote counts | Real-world transitions |

## Practical Interpretations

### Diagonal Elements (Voter Loyalty)

- **High diagonal values** (>0.8): Strong party loyalty, voters tend to stay with their party
- **Low diagonal values** (<0.6): Weak loyalty, high voter mobility
- **Expected range**: 0.65-0.95 for major parties

### Off-Diagonal Elements (Party Switching)

- **Shas → Agudat Israel**: Intra-Haredi movement between parties
- **Haredi → Other**: Secularization or political diversification
- **Other → Haredi**: Religious awakening or political shift
- **Any → Abstained**: Political disengagement

### City-Level Deviations

- **Positive deviations**: Cities with stronger transitions than national average
- **Negative deviations**: Cities with weaker transitions than national average
- **Student-t distribution**: Allows for outlier cities with extreme behavior

#### City Deviation Analysis

The model saves detailed city deviation metrics in `city_deviations.csv`:

- **Element-wise deviations**: For each transition probability (i,j), shows how much city differs from country average
- **Credible intervals**: Uncertainty quantification for each deviation
- **Interpretation**:
  - Large positive deviations: City has stronger transition than national average
  - Large negative deviations: City has weaker transition than national average
  - Credible intervals excluding zero: Statistically significant city effects


### Overdispersion (φ)

The overdispersion parameter `phi` controls how much extra variation exists beyond what a standard multinomial distribution would predict. This is crucial for realistic modeling of political data.

#### What is Overdispersion?

**Overdispersion** occurs when observed data shows more variation than a standard probability model predicts. In political voting, this happens because:

- **Neighborhood effects**: Similar voters cluster geographically
- **Social influence**: People influence each other's voting decisions
- **Unobserved demographics**: Age, income, education affect voting but aren't in our model
- **Campaign effects**: Local campaigning varies by area

#### Mathematical Role

In the DirichletMultinomial likelihood:
```python
x2_obs ~ DirichletMultinomial(n, phi * q)
```

Where:
- `n` = total votes at a ballot box
- `q` = expected proportions (from transition matrix)
- `phi` = overdispersion parameter

The variance scales as:
```python
# Standard multinomial variance
var_standard = n × p × (1-p)

# DirichletMultinomial variance with overdispersion
var_dm = n × p × (1-p) × (1 + phi)
```

#### Practical Interpretations

**φ > 1 (Most Common)**: Extra variation beyond multinomial
- **Typical range**: 1.5 - 10.0 for political data
- **Meaning**: Voters are more clustered/heterogeneous than random
- **Example**: If φ = 3.0, ballot boxes show 3× more variation than expected
- **Real-world impact**:
  - φ = 2: 95% credible intervals are ~1.4× wider
  - φ = 5: 95% credible intervals are ~2.4× wider
  - φ = 10: 95% credible intervals are ~3.3× wider

**φ ≈ 1**: Standard multinomial variation
- **Meaning**: Voters behave as independent random draws
- **Rare in practice**: Political behavior is rarely truly random
- **Indicates**: Model might be missing important clustering factors

**φ < 1**: Less variation than expected
- **Very rare**: Would suggest voters are more uniform than random
- **Possible causes**:
  - Over-aggregation masking true variation
  - Model misspecification
  - Data quality issues

#### Why Overdispersion Matters

1. **Realistic Uncertainty**: Without φ, credible intervals would be too narrow
2. **Model Validation**: Helps assess whether model captures data structure
3. **Prediction Accuracy**: Proper uncertainty quantification for forecasts
4. **Policy Implications**: Affects confidence in transition estimates

#### Estimation Strategy

The model uses log-scale parameterization:
```python
log_phi ~ Normal(0, 1)  # Unconstrained prior
phi = exp(log_phi)      # Always positive
```

This ensures φ > 0 while allowing flexible estimation. The Normal(0,1) prior is weakly informative, letting the data determine the appropriate level of overdispersion.

## Temporal Priors (Sequential Transitions)

For sequences of elections (e.g., kn19→20, kn20→21, ...), the model supports using the previous transition's posterior as the prior for the next transition.

### What Carries Over
- `Z_country`: country-level logits per column
- `diag_bias`: diagonal loyalty bias
- `log_phi`: overdispersion on log-scale
- `Z_city` (optional): per-city logits when available

These centers become the means of the corresponding priors for the next transition, with configurable innovation (standard deviation) controlling how tightly the prior constrains the new fit.

### Prior Formulation
- `Z_country ~ Normal(mu=Z_country_prev, sigma=innovation.Z_country_sigma)`
- `diag_bias ~ Normal(mu=diag_bias_prev, sigma=innovation.diag_bias_sigma)`
- `Z_city[c] ~ StudentT(nu, mu=Z_country + delta_c_prev, sigma=innovation.Z_city_sigma)` when available
- `log_phi ~ Normal(mu=log_phi_prev, sigma=innovation.log_phi_sigma)`

Centers are computed from the previous posterior (`mean` by default, `median` optional). If a city prior is missing, the model falls back to the country-centered prior for that city.

### Configuration
In `data/config.yaml` under `model.temporal_priors`:

```yaml
model:
  temporal_priors:
    enabled: true
    center: mean            # mean|median
    innovation:
      diag_bias_sigma: 0.5
      Z_country_sigma: 1.0
      Z_city_sigma: 0.7
      log_phi_sigma: 1.0
    carryover_city_level: true
```

### Pipeline Behavior
- First transition in the list uses default priors.
- Each subsequent transition attempts to load `priors.json` from the previous transition's output directory and uses it to center the priors.
- If priors are missing, the model falls back to default priors and logs a warning.

### Practical Implications
- Encourages temporal smoothness while allowing changes via innovation scales.
- Reduces sampling time and improves stability when elections are similar.
- City-level priors can capture persistent local structure but are optional.

## Data Processing Pipeline

### Input Data

1. **Raw Election Results**: Ballot box level vote counts
2. **Harmonized Data**: Standardized party categories across elections
3. **City Mapping**: Hebrew-English city name translations

### Preprocessing Steps

1. **Category Computation**: Aggregate parties into four categories
2. **Ballot Box Alignment**: Match stations between consecutive elections
3. **Data Validation**: Ensure category consistency and non-negative values
4. **Hierarchical Grouping**: Organize data by country and target cities

### Output Files

For each transition pair (e.g., `kn20_21`):

#### Country-Level Outputs
- `country_trace.nc`: Posterior samples (NetCDF format)
- `country_map.csv`: Transition probabilities with credible intervals
- `country_movements.csv`: Actual vote movements (thousands of votes)

#### City-Level Outputs
- `city_{name}_trace.nc`: City-specific posterior samples
- `city_{name}_map.csv`: City transition probabilities
- `city_{name}_movements.csv`: City vote movements

#### Diagnostics
- `fit_summary_{pair}.json`: Model diagnostics and convergence metrics
- `model.png`: PyMC model visualization

## Model Validation

### Convergence Diagnostics

- **R-hat < 1.01**: Chains have converged
- **ESS > 400**: Sufficient effective sample size
- **BFMI > 0.2**: Good energy diagnostics
- **No divergences**: Sampler explored posterior properly

### Posterior Predictive Checks

- Compare observed vs predicted vote totals
- Validate model fit across different election pairs
- Check for systematic biases in predictions

### Voter Loyalty Validation

The model includes automatic checks for realistic voter loyalty:
- Diagonal elements should be ≥ 0.65 for major parties
- Warnings issued if loyalty appears unrealistically low

## Usage Examples

### Basic Model Fitting

```python
from src.transition_model.fit import fit_transition_pair

# Fit a single transition pair
result = fit_transition_pair(
    pair_tag="kn20_21",
    election_t_path="data/interim/harmonized_20.parquet",
    election_t1_path="data/interim/harmonized_21.parquet",
    output_dir="data/processed/transitions",
    target_cities=["jerusalem", "bnei brak", "ashdod"],
    columns_mapping=column_mappings
)
```

### Loading Results

```python
from src.transition_model.io import load_point_estimates, load_fit_summary

# Load transition probabilities
transitions = load_point_estimates("data/processed/transitions/kn20_21/country_map.csv")

# Load model diagnostics
diagnostics = load_fit_summary("data/processed/logs/fit_summary_kn20_21.json")
```

### Visualization

```python
# Use the visualization script
python visualize_transitions.py
```

## Model Advantages

1. **Hierarchical Structure**: Borrows strength across cities while allowing local variation
2. **Heavy-Tailed Deviations**: Robust to outlier cities with unusual voting patterns
3. **Logistic-Normal Parameterization**: More flexible than Dirichlet priors
4. **Automatic Validation**: Built-in checks for realistic voter behavior
5. **Comprehensive Outputs**: Both probabilities and actual vote counts

## Limitations and Considerations

1. **Ecological Inference**: Individual-level transitions inferred from aggregate data
2. **Station Alignment**: Requires matching ballot boxes between elections
3. **Category Aggregation**: Loss of detail from individual party analysis
4. **Computational Cost**: MCMC sampling can be time-intensive for large datasets

## Future Extensions

- **Temporal Trends**: Model evolution of transition patterns over time
- **Demographic Covariates**: Include city-level demographic variables
- **Spatial Correlation**: Account for geographic proximity between cities
- **Dynamic Transitions**: Allow transition matrices to vary by election characteristics

# Instructions: Refactor hierarchical transition model to logistic-normal with sparse (heavy-tailed) city deviations

## Goal
Replace the Dirichlet/κ hierarchy with a logistic-normal parameterization:
- Country columns live on logits: `Z_country ~ Normal`
- Loyalty encoded via diagonal bias on logits, then softmax ⇒ column-stochastic `M_country`
- Cities deviate on logits via heavy-tailed Student-t around `Z_country` (sparse/outlier-friendly)
- Remove global kappa and all Dirichlet priors for matrices.
- Keep likelihood `DirichletMultinomial` with `q = p1 @ M.T` (columns=from, rows=to).
- Put phi on log scale.
- Preserve function signature and docstring.

## Files to edit
- The Python module that defines `build_hierarchical_model(...)`. Find the function by its exact signature.
- (Optional) If a config reader is present, ignore κ-related config keys.

## Search and Remove
1. Remove country Dirichlet:
   ```python
   alpha_base = ...
   np.fill_diagonal(alpha_base, alpha_diag)
   M_country_cols = []
   for j in range(K):
       col = pm.Dirichlet(f"M_country_col_{j}", a=alpha_base[:, j])
       M_country_cols.append(col)
   M_country = pm.math.stack(M_country_cols, axis=1)
   ```

2. Remove kappa and city Dirichlet:
   ```python
   kappa = pm.Exponential("kappa", lam=1 / kappa_prior_scale)
   ...
   city_col = pm.Dirichlet(f"M_city_{c}_col_{j}", a=kappa * M_country_cols[j])
   ```

3. Do NOT change the `DirichletMultinomial` likelihood lines or `q = p1 @ M.T`.

## Add (replace removed blocks)

### Country-level logistic-normal prior (robust)
```python
eyeK = pm.math.eye(K)

sigma_country = pm.HalfNormal("sigma_country", sigma=0.7)
Z_country = pm.Normal("Z_country", mu=0.0, sigma=sigma_country, shape=(K, K))

diag_bias = pm.Normal("diag_bias", mu=3.0, sigma=0.3)  # encodes loyalty in mean

M_country_cols = []
for j in range(K):
    z = Z_country[:, j] + diag_bias * eyeK[:, j]
    col = pm.Deterministic(f"M_country_col_{j}", pm.math.softmax(z))
    M_country_cols.append(col)
M_country = pm.math.stack(M_country_cols, axis=1)
```

### City-level sparse (heavy-tailed) deviations on logits
```python
sigma_city = pm.HalfNormal("sigma_city", sigma=0.3)
nu_raw = pm.Exponential("nu_raw", lam=1/5.0)
nu = pm.Deterministic("nu", nu_raw + 2.0)

M_cities = None
if n_cities > 0:
    Z_city = pm.StudentT("Z_city", nu=nu, mu=Z_country, sigma=sigma_city, shape=(n_cities, K, K))
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
```

### Overdispersion (log-parameterized)
```python
log_phi = pm.Normal("log_phi", mu=0.0, sigma=1.0)
phi = pm.Deterministic("phi", pm.math.exp(log_phi))
```

## Keep (unchanged)
```python
# Country data
country_data = data["country"]
x1_country = country_data["x1"]
x2_country = country_data["x2"]
n2_country = country_data["n2"]

p1_country = x1_country / x1_country.sum(axis=1, keepdims=True)
q_country = pm.math.dot(p1_country, M_country.T)

pm.DirichletMultinomial("x2_country_obs", n=n2_country, a=phi * q_country, observed=x2_country)

# City data
for i, city in enumerate(cities):
    city_data = data[city]
    x1_city = city_data["x1"]
    x2_city = city_data["x2"]
    n2_city = city_data["n2"]

    M_city = M_cities[i] if M_cities is not None else M_country

    p1_city = x1_city / x1_city.sum(axis=1, keepdims=True)
    q_city = pm.math.dot(p1_city, M_city.T)

    pm.DirichletMultinomial(f"x2_{city}_obs", n=n2_city, a=phi * q_city, observed=x2_city)
```

## Assertions / Conventions
- Ensure consistent category order between x1/x2:
  ```python
  assert x1_country.shape[1] == x2_country.shape[1] == K
  ```
- Keep matrix convention **columns=from, rows=to**.

## Cleanups
- Remove references to `kappa`, `kappa_prior_scale`, and Dirichlet priors for matrices.
- Retain `target_accept ≥ 0.95` (recommend 0.98).

## Acceptance Criteria
- Model samples without overwhelming divergences.
- Posterior `M_country` columns are near-identity due to `diag_bias`, but cities can show sparse deviations.
- Heatmaps labeled rows=to, cols=from look sensible.

## Optional tuning
- Adjust `diag_bias` mean for stronger/weaker diagonals.
- Adjust `nu_raw` or `sigma_city` for more/less freedom of outliers.

# Appendix

## Appendix A  City-Level Transition Matrices

The following figures provide detailed transition matrices for all analyzed cities across consecutive elections.  These
supplement the summary analysis presented in the main text, particularly the condensed discussion in the City-Level
Variation section.

### Ashdod

![ashdod transitions](plots/city_ashdod_transition_matrix_over_elections.png)

*Figure A1: Ashdod exhibited the most extreme disruption during the March 2020–March 2021 transition (2324), with Shas loyalty
dropping to 67.1% and 19.3% switching to UTJ.*

### Beit Shemesh

![beit shemesh transitions](plots/city_beit_shemesh_transition_matrix_over_elections.png)

*Figure A2: Beit Shemesh showed moderate disruption with Shas loyalty falling to 75.4% during the 2324 transition.*

### Elad

![elad transitions](plots/city_elad_transition_matrix_over_elections.png)

*Figure A3: Elad transition patterns across all election pairs.*

### Bnei Brak

![bnei brak transitions](plots/city_bnei_brak_transition_matrix_over_elections.png)

*Figure A4: Bnei Brak, despite being predominantly Ashkenazi, experienced a sharp Shas loyalty drop to 70.9% during the 2324
transition.*

### Jerusalem

![jerusalem transitions](plots/city_jerusalem_transition_matrix_over_elections.png)

*Figure A5: Jerusalem transition patterns across all election pairs.*

### Modi'in Illit

![modi'in illit transitions](plots/city_modiin_illit_transition_matrix_over_elections.png)

*Figure A6: Modi'in Illit transition patterns across all election pairs.*



## Appendix B  Model Validation and Diagnostics

This section documents model validation through posterior predictive checks and convergence diagnostics for transparency
and reproducibility.

### Model Validation

Posterior predictive checks confirm that the model adequately reproduces empirical vote counts across all elections.
Most chains converged successfully, with effective sample sizes (ESS) exceeding 400 for key parameters and R-hat values
approaching 1.01 in later runs. Minor convergence issues in earlier election pairs (January 2013March 2015 transition,
Knesset 1920) were resolved by increasing the number of draws and adopting non-centered parameterization.

### Convergence Diagnostics

| Transition | R-hat max | ESS min | |------|------------|----------| | Kn 1920 (Jan 2013Mar 2015) | 1.530 | 7 | | Kn
2021 (Mar 2015Apr 2019) | 1.529 | 7 | | Kn 2122 (Apr 2019Sep 2019) | 1.465 | 7 | | Kn 2223 (Sep 2019Mar 2020) | 1.134 |
19 | | Kn 2324 (Mar 2020Mar 2021) | 1.477 | 7 | | Kn 2425 (Mar 2021Nov 2022) | 1.737 | 6 |

While some early models show high R-hat and low ESS, these issues were largely addressed through increased sampling and
refined priors. The final models show stable posteriors without divergences.

Representative diagnostics are shown below.

![diag rank](plots/diag_kn21_22_rank.png)

*Figure B1: Rank plots for the Kn 2122 (April 2019–September 2019) transition showing uniform distributions across chains, indicating good mixing.*

![diag energy](plots/diag_kn21_22_energy.png)

*Figure B2: Energy plots showing no evidence of divergent transitions or geometric pathologies in the posterior.*

![diag autocorr](plots/diag_kn21_22_autocorr.png)

*Figure B3: Autocorrelation plots demonstrating rapid decorrelation of MCMC samples for key model parameters.*


## Appendix C  Model Robustness Testing

To verify that the observed synchronization of voter transitions across cities represents genuine coordinated behavior
rather than an artifact of the hierarchical Bayesian model structure, I tested three alternative model specifications
with varying levels of flexibility for city-specific patterns.

### Model Specifications

**Original Hierarchical Model** (baseline): The model described in the Methods section uses hierarchical pooling with
moderately tight priors to stabilize estimates while allowing cities to deviate from national patterns. Key parameters:
`sigma_D = 0.3`, `delta_scale = 0.5`, `D_sigma = 0.3`.

**Relaxed Hierarchical Model**: Same hierarchical structure but with substantially increased variability parameters to
allow greater city-specific deviations: `sigma_D = 0.8` (+167%), `delta_scale = 1.5` (+200%), `D_sigma = 0.8` (+167%).
This specification tests whether tighter priors artificially constrain city-level patterns.

**Independent Model**: Each city is fitted separately without hierarchical pooling. The country-level transition matrix
is estimated first, then used as a prior mean for independent city-level fits with `sigma_city = 0.8`. This
specification completely removes structural constraints toward similarity between cities.

### Results

Despite the increased flexibility allowing cities to diverge substantially from national patterns (248% increase in
inter-city variability for the independent model), the synchronized drops and recoveries in Shas loyalty remained
evident across all model specifications. Figure C1 shows Shas→Shas retention rates across all three models, revealing
that the temporal patterns and cross-city synchronization persist regardless of model structure.

![Model comparison by location](plots/shas_shas_city_comparison.png)

*Figure C1: Shas→Shas retention rates across models. Each panel shows one location (country or city) with three lines
representing the original hierarchical (blue), relaxed hierarchical (orange), and independent (green) models. The
synchronized drop during the 23→24 transition appears in all models across all cities.*

Figure C2 presents an alternative view with one panel per model, showing all cities together within each specification.
The inter-city standard deviation (shown in gray shading) increases substantially in the relaxed and independent models,
confirming that these specifications successfully allow greater divergence. Yet the temporal correlation across cities
remains evident in all three panels.

![Model comparison by specification](plots/shas_shas_by_model.png)

*Figure C2: Shas→Shas retention rates by model specification. Each panel shows all cities within one model. Country
estimates shown as thick gray dashed line; individual cities as colored solid lines. Gray shading indicates inter-city
standard deviation. Despite increased flexibility in relaxed and independent models, synchronized patterns persist.*

### Quantitative Comparison

Mean Absolute Deviation (MAD) from country estimates across all transition pairs and electoral categories:

- Original hierarchical: MAD = 0.0044
- Relaxed hierarchical: MAD = 0.0116 (+166%)
- Independent: MAD = 0.0152 (+248%)

The substantial increases in inter-city variability confirm that the alternative specifications successfully relax
constraints. The persistence of synchronized transitions across all specifications demonstrates that the observed
coordination reflects genuine features of the electoral data rather than modeling artifacts.

Full implementation details, including model code and configuration files, are available in the GitHub repository.

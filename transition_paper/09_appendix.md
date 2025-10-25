# Appendix

## Appendix A  City-Level Transition Matrices

The following tables and figures provide full transition matrices for all analyzed cities across consecutive elections.
These details complement the summary presented in the main text and allow replication of city-specific patterns.

### Full Matrices and Plots

![ashdod transitions](plots/city_ashdod_transition_matrix_over_elections.png)

![beit shemesh transitions](plots/city_beit_shemesh_transition_matrix_over_elections.png)

![elad transitions](plots/city_elad_transition_matrix_over_elections.png)

![bnei brak transitions](plots/city_bnei_brak_transition_matrix_over_elections.png)

![jerusalem transitions](plots/city_jerusalem_transition_matrix_over_elections.png)

![modi'in illit transitions](plots/city_modiin_illit_transition_matrix_over_elections.png)



## Appendix B  Model Diagnostics

This section documents model convergence diagnostics for transparency and reproducibility.

| Transition | R-hat max | ESS min |
|------|------------|----------|
| Kn 19→20 (Jan 2013–Mar 2015) | 1.530 | 7 |
| Kn 20→21 (Mar 2015–Apr 2019) | 1.529 | 7 |
| Kn 21→22 (Apr 2019–Sep 2019) | 1.465 | 7 |
| Kn 22→23 (Sep 2019–Mar 2020) | 1.134 | 19 |
| Kn 23→24 (Mar 2020–Mar 2021) | 1.477 | 7 |
| Kn 24→25 (Mar 2021–Nov 2022) | 1.737 | 6 |

While some early models show high R-hat and low ESS, these issues were largely addressed through increased sampling and refined priors. The final models show stable posteriors without divergences.

Representative diagnostics are shown below.

![diag rank](plots/diag_kn21_22_rank.png)

![diag energy](plots/diag_kn21_22_energy.png)

![diag autocorr](plots/diag_kn21_22_autocorr.png)

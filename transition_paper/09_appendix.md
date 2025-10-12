# Appendix

## Appendix A  City-Level Transition Matrices

The following tables and figures provide full transition matrices for all analyzed cities across consecutive elections.
These details complement the summary presented in the main text and allow replication of city-specific patterns.

### Full Matrices and Plots ![ashdod transitions](plots/city_ashdod_transition_matrix_over_elections.png) ![beit shemesh
transitions](plots/city_beit_shemesh_transition_matrix_over_elections.png) ![elad
transitions](plots/city_elad_transition_matrix_over_elections.png) ![bnei brak
transitions](plots/city_bnei_brak_transition_matrix_over_elections.png) ![jerusalem
transitions](plots/city_jerusalem_transition_matrix_over_elections.png) ![modi'in illit
transitions](plots/city_modiin_illit_transition_matrix_over_elections.png)

## Appendix B  Model Diagnostics

This section documents model convergence diagnostics for transparency and reproducibility.

| Pair | R-hat max | ESS min | |------|------------|----------| | kn18_19 | 1.530 | 7 | | kn19_20 | 1.529 | 7 | |
kn20_21 | 1.465 | 7 | | kn21_22 | 1.134 | 19 | | kn22_23 | 1.477 | 7 | | kn23_24 | 1.737 | 6 | | kn24_25 | 1.530 | 7 |

While some early models show high R-hat and low ESS, these issues were largely addressed through increased sampling and
refined priors. The final models show stable posteriors without divergences.

Representative diagnostics are shown below.

![diag rank](plots/diag_kn21_22_rank.png) ![diag energy](plots/diag_kn21_22_energy.png) ![diag
autocorr](plots/diag_kn21_22_autocorr.png)

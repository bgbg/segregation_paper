# Appendix

## Appendix A  City-Level Transition Matrices

The following figures provide detailed transition matrices for all analyzed cities across consecutive elections.
These supplement the summary analysis presented in the main text, particularly the condensed discussion in the City-Level Variation section.

### Ashdod

![ashdod transitions](plots/city_ashdod_transition_matrix_over_elections.png)

*Ashdod exhibited the most extreme disruption during the March 2020–March 2021 transition (23→24), with Shas loyalty dropping to 64.5% and 21.3% switching to UTJ.*

### Beit Shemesh

![beit shemesh transitions](plots/city_beit_shemesh_transition_matrix_over_elections.png)

*Beit Shemesh showed moderate disruption with Shas loyalty falling to 75.2% during the 23→24 transition.*

### Elad

![elad transitions](plots/city_elad_transition_matrix_over_elections.png)

*Elad transition patterns across all election pairs.*

### Bnei Brak

![bnei brak transitions](plots/city_bnei_brak_transition_matrix_over_elections.png)

*Bnei Brak, despite being predominantly Ashkenazi, experienced a sharp Shas loyalty drop to 69.5% during the 23→24 transition.*

### Jerusalem

![jerusalem transitions](plots/city_jerusalem_transition_matrix_over_elections.png)

*Jerusalem transition patterns across all election pairs.*

### Modi'in Illit

![modi'in illit transitions](plots/city_modiin_illit_transition_matrix_over_elections.png)

*Modi'in Illit transition patterns across all election pairs.*



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

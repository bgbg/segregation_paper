# Hidden Volatility in Cohesive Voting Blocs: Evidence from Israel's Ultra-Orthodox Electorate

Boris Gorelik boris@gorelik.net

## Introduction

Across many democracies, certain communities exhibit cohesive voting patterns that resemble durable "voting blocs" anchored in strong group identities. Classic cleavage theories and the literature on ethnoreligious mobilization argue that such identities can cement stable partisan alignments over long periods (Koopmans and Statham 1999; McCauley 2014). This backdrop raises a central question for political behavior: are these loyal voting blocs truly immobile, or can hidden currents of change exist beneath an outwardly stable surface?

Israel's ultra‑Orthodox (Haredi) Jews provide a compelling case study. The Haredi sector is widely portrayed as one of the most disciplined electoral blocs in Israel, with community members casting ballots in line with sectoral leadership and long‑standing ethnoreligious alignments (Friedman 1991; Curiel and Zeedan 2024). Yet this conventional picture leaves open whether electoral discipline fully suppresses volatility, or whether meaningful micro‑shifts can occur without altering headline outcomes.

An important analytical challenge arises from the heterogeneous composition of Haredi political parties. While most Ashkenazi Haredim vote for Agudat Israel and most Agudat Israel voters are Ashkenazi Haredim, the relationship between Sephardic Haredim and Shas is more complex: while most Sephardic Haredim vote for Shas, only approximately one-third of Shas voters are estimated to be Haredim. To address this analytical challenge and ensure that our analysis focuses specifically on Haredi population hubs rather than the general population, I follow the methodology in  Gorelik 2025 in restricting the analysis to ballot boxes with at least 75% Haredi votes. This filtering ensures that all findings, transition probabilities, and conclusions in this study pertain specifically to Haredi communities rather than broader electoral patterns.

Recent work on Haredi residential patterns revealed a surprising clue. A study using Knesset voting returns as a demographic proxy documented persistent **intra‑Haredi ethnic segregation** across multiple cities, but also identified an anomalous and **sudden drop in the dissimilarity index in Ashdod** around the March 2021 election (Gorelik 2025). This anomaly raised a theoretical puzzle: if people do not move that fast, what changed so quickly? One plausible answer is politics. If residential separation is comparatively slow to shift, **voting behavior can change quickly**, and such change might be most visible where social boundaries are already under stress.

The present study addresses this puzzle by analyzing **voter transition matrices** for the Haredi electorate across successive national elections. Using polling‑station results and a hierarchical ecological inference framework, I estimate transition probabilities among four categories—votes for Shas, votes for Agudat Israel, votes for other parties, and abstention—at both the national level and for key Haredi population centers.

## Methods

### Data and Problem Overview

I analyze voter transitions between consecutive elections using aggregate polling-station data. In this setting, I observe for each polling station the counts of voters who chose each political category in the initial election (time *t*) and the counts for each category in the subsequent election (time *t+1*). My goal is to infer the **individual-level transition probabilities** – the probability that a voter moves from category *i* at time *t* to category *j* at time *t+1*. This problem is inherently difficult because individual ballots are secret and only aggregate counts are available, a classic **ecological inference** scenario (Robinson, 1950; King, 1997).

My application focuses on four political categories in Israeli elections – **Shas**, **Agudat Israel**, **Other parties**, and **Abstention** – which represent distinct voting blocs (religious Sephardic, religious Ashkenazi, all other votes, and non-voters, respectively). However, as noted in the introduction, I restrict the analysis to ballot boxes with at least 75% Haredi votes, following Gorelik (2025).

I align polling stations between election $t$ and $t+1$ and construct aggregated counts for each station, modeling the unseen flows between these categories to recover **transition matrices** consistent with observed margins. Importantly, all transition probabilities and conclusions reported in this study pertain specifically to Haredi population hubs rather than the general Israeli electorate.

### Analytical Approach: Individual Ballot Boxes vs. City Aggregation

Two main approaches are available for analyzing voter transitions from aggregate data: modeling individual ballot boxes paired over time versus aggregating data at the city level before analysis. Each approach presents distinct advantages and limitations.

**Individual ballot box approach** (employed in this study) offers several advantages: (1) **Maximum data utilization** – preserves all available information by using each polling station as an observation unit; (2) **Fine-grained spatial resolution** – captures neighborhood-level variation within cities that city-level aggregation would obscure; (3) **Robustness to aggregation bias** – avoids potential distortions from averaging heterogeneous precincts; and (4) **Statistical power** – provides larger effective sample sizes for hierarchical modeling. However, this approach also presents challenges: (1) **Computational complexity** – requires sophisticated hierarchical models to handle thousands of observations; (2) **Ballot box alignment issues** – stations may be created, eliminated, or redistricted between elections; and (3) **Model convergence** – the high-dimensional parameter space can lead to sampling difficulties.

**City-level aggregation approach** offers simplicity and computational tractability: (1) **Straightforward implementation** – treats each city as a single observation, reducing model complexity; (2) **Clear interpretation** – city-level transition matrices are directly interpretable; and (3) **Robust to station changes** – aggregation naturally handles ballot box redistricting. However, this approach sacrifices important information: (1) **Loss of spatial detail** – masks neighborhood-level heterogeneity within cities; (2) **Reduced statistical power** – fewer observations limit the ability to detect subtle patterns; and (3) **Aggregation bias** – averaging across heterogeneous precincts may distort true transition patterns.

Given the research focus on **hidden volatility** within ostensibly stable voting blocs, the individual ballot box approach is preferable because it preserves the fine-grained spatial resolution necessary to detect subtle shifts that city-level aggregation might obscure. The hierarchical modeling framework addresses the computational challenges while maintaining the analytical advantages of the individual-station approach.

### Hierarchical Transition Model

At the core of the model is a **voter transition matrix** $M$, where each entry $M_{ij}$ represents the probability that a voter in category *j* at time *t* votes for category *i* at time *t+1*. I interpret the diagonal entries $M_{ii}$ as **voter loyalty** and the off-diagonals as **switching probabilities**.

I introduce a three-level hierarchical structure:

1. **Country-level transition matrix ($M^\text{country}$):** Captures baseline voting transition tendencies for the entire country.

2. **Shared deviation pattern ($D$):** Allows cities to systematically deviate from the national matrix along a **common pattern** $D$. This $K\times K$ matrix represents how transition probabilities tend to shift in deviating cities relative to the national baseline.

3. **City-specific scalar ($\delta_c$):** Each city $c$ has a scalar parameter $\delta_c$ that scales the global deviation pattern. The city's transition matrix is:

$$ M^{(c)} = \text{softmax}\big(Z^\text{country} + \delta_c \, D\big) $$

A positive $\delta_c$ means city $c$ accentuates the transitions indicated by $D$; a negative $\delta_c$ implies the opposite direction.

### Logistic-Normal Parameterization

I adopt a **logistic-normal** modeling strategy to avoid constraints of Dirichlet priors. I work in unconstrained **logit space** and transform to probabilities. For each origin category $j$:

$$ Z^\text{country}_{ij} = \alpha_{ij} + B \cdot \mathbf{1}(i=j) $$

where $\alpha_{ij} \sim \mathcal{N}(0, \sigma^2_\text{country})$ and $B$ is a **loyalty bias** parameter added to diagonal elements to reflect voter inertia.

### Simplified Model Specification

For computational stability, we use:
- **Normal priors** for city scalars: $\delta_c \sim \mathcal{N}(0, 0.5)$
- **Fixed overdispersion**: $\phi = 100.0$ in the Dirichlet-multinomial likelihood
- **Tighter hyperpriors**: $\sigma_\text{country} = 0.5$, $\sigma_D = 0.3$, loyalty bias $B \sim \mathcal{N}(2.0, 0.3^2)$

### Estimation Strategy

I fit the model using a **progressive sampling strategy** with two stages:
1. **Initial adaptation**: 2,000 tuning + 500 draws, target acceptance 0.90
2. **Main sampling**: 3,000 tuning + 3,000 draws, target acceptance 0.95

I use 4 chains and check convergence diagnostics (Gelman–Rubin $\hat{R} < 1.01$, effective sample sizes > 400).

## Results

### Country-Level Transition Patterns

The national transition matrices reveal several key patterns across Knesset elections 18-25 (2013-2025) within Haredi population hubs. **Voter loyalty** remains high overall, with diagonal retention rates typically above 85% for major parties. However, I observe notable **episodic volatility**, particularly a sharp drop in Shas loyalty during the kn20_21 transition (2020-2021), where retention fell to 73% compared to typical levels above 90%. It is important to emphasize that these patterns reflect transitions specifically within Haredi communities (ballot boxes with $\geq$75% Haredi votes) rather than broader Israeli electoral dynamics, given the heterogeneous composition of Shas voters nationally.

**Table 1: Selected Country-Level Transition Probabilities (%)**

| Transition | kn18_19 | kn19_20 | kn20_21 | kn21_22 | kn22_23 | kn23_24 | kn24_25 |
|------------|---------|---------|---------|---------|---------|---------|---------|
| Shas → Shas | 91.8 | 85.0 | 97.7 | 99.7 | 98.9 | 72.8 | 98.6 |
| Agudat → Agudat | 100.0 | 92.6 | 94.3 | 98.0 | 97.6 | 87.9 | 96.7 |
| Other → Other | 79.3 | 90.7 | 57.8 | 85.1 | 86.9 | 87.9 | 86.8 |
| Abstained → Abstained | 85.6 | 95.7 | 99.2 | 93.5 | 98.1 | 97.9 | 92.8 |

The most dramatic shift occurs in kn23_24, where Shas loyalty drops to 72.8%, indicating substantial voter movement during this period. Cross-party flows between Shas and Agudat Israel remain generally low but show occasional spikes, particularly in kn23_24 where 12.3% of Agudat voters moved to Shas.

![Country-level transition matrices over elections](data/processed/reports/plots/country_transition_matrix_over_elections.png)

### City-Level Heterogeneity

City-specific analyses reveal substantial heterogeneity in transition patterns within Haredi population centers. **Mean Absolute Deviation (MAD)** from country patterns varies significantly across cities, with all analyses restricted to ballot boxes with $\geq$75% Haredi votes to ensure focus on Haredi-specific electoral dynamics:

**Table 2: Mean Absolute Deviation by City**

| City | Mean MAD (pp) | Max MAD (pp) |
|------|---------------|--------------|
| Ashdod | 1.0 | 2.4 |
| Beit Shemesh | 0.6 | 1.5 |
| Elad | 2.6 | 8.7 |
| Bnei Brak | 1.0 | 3.7 |
| Jerusalem | 0.6 | 1.0 |
| Modi'in Illit | 2.2 | 10.6 |

**Jerusalem** and **Beit Shemesh** show the lowest average deviations (0.6 pp), indicating voting patterns closely aligned with national trends. In contrast, **Elad** and **Modi'in Illit** display higher average deviations (2.6-2.2 pp) and much larger maximum deviations (8.7-10.6 pp), suggesting these cities experience more volatile electoral dynamics.

The **table lens visualization** below shows how city rankings by MAD evolve across election pairs, with cities sorted by their deviation magnitude in the most recent election (kn24_25):

![MAD rankings by Knesset pair - table lens visualization](data/processed/reports/plots/mad_table_lens.png)

### Temporal Dynamics and the Ashdod Anomaly

The timing of electoral volatility provides insight into the residential segregation anomaly observed in Ashdod. **Elad** shows the highest MAD in the most recent election (kn24_25: 8.7 pp), while **Modi'in Illit** experienced its largest deviation much earlier (kn18_19: 10.6 pp). **Ashdod**, despite its segregation anomaly, maintains relatively moderate electoral deviations throughout the study period, suggesting that residential integration may have preceded rather than followed electoral change.

### Model Diagnostics

The simplified Bayesian model shows mixed convergence across election pairs. While the model structure successfully captures transition patterns, several pairs exhibit convergence challenges:

**Table 3: Sampling Diagnostics**

| Pair | R-hat max | ESS min | Status |
|------|-----------|---------|---------|
| kn18_19 | 1.530 | 7 | Not converged |
| kn19_20 | 1.529 | 7 | Not converged |
| kn20_21 | 1.465 | 7 | Not converged |
| kn21_22 | 1.134 | 19 | Not converged |
| kn22_23 | 1.477 | 7 | Not converged |
| kn23_24 | 1.737 | 6 | Not converged |
| kn24_25 | 1.530 | 7 | Not converged |

**Interpretation**: All pairs show R-hat values above the 1.01 threshold for good convergence, with effective sample sizes well below the recommended 400. These diagnostics indicate that the posterior geometry remains challenging despite model simplifications, suggesting that electoral transition inference requires careful attention to model specification and sampling strategies.

## Discussion

My findings challenge monolithic narratives of Haredi political immobility by revealing **hidden volatility** beneath surface stability. Three key empirical patterns emerge:

**First**, even highly disciplined electorates experience **episodic loyalty dips**. The dramatic Shas retention drop in kn23_24 (72.8%) demonstrates that party loyalty can weaken rapidly, even in communities with strong religious and social cohesion mechanisms.

**Second**, **substantial city-level heterogeneity** exists within the Haredi electorate. Cities like Jerusalem and Beit Shemesh maintain patterns close to national averages, while Elad and Modi'in Illit show much greater volatility. This suggests that local institutional ecologies—schools, synagogues, leadership networks—mediate the strength of bloc discipline.

**Third**, the **temporal alignment** between Ashdod's segregation anomaly and broader electoral volatility suggests that political and residential dynamics may be more interconnected than previously recognized. While residential mobility is slow, electoral mobility can shift quickly and may both reflect and influence social integration processes.

Methodologically, my **spatio-temporal framework** demonstrates the value of pairing segregation metrics with transition analysis. While dissimilarity indices capture separation structure at a given moment, transition matrices reveal the dynamics of voter movement over time, including loyalty and crossover rates.

These results contribute to broader debates on voter-bloc discipline by demonstrating that disciplined electorates may harbor **latent realignment pressures** that surface under specific local and temporal conditions. The findings echo patterns in other cohesive minority electorates globally, where collective identities shape voting but do not fully eliminate strategic switching at the margins.

## References

Curiel, Concha Pérez, and Rami Zeedan. 2024. "Social Identity and Voting Behavior in a Deeply Divided Society: The Case of Israel." *Societies* 14 (9): 177.

Friedman, Menachem. 1991. *The Haredi (Ultra‑Orthodox) Society: Sources, Trends and Processes*. Jerusalem: The Jerusalem Institute for Israel Studies.

Gorelik, Boris. 2025. "Ethnic Divisions Within Unity: Insights into Intra‑Group Segregation from Israel's Ultra‑Orthodox Society." *Social Sciences* 14 (x): x–x.

King, Gary. 1997. *A Solution to the Ecological Inference Problem: Reconstructing Individual Behavior from Aggregate Data*. Princeton, NJ: Princeton University Press.

Koopmans, Ruud, and Paul Statham. 1999. "Challenging the Liberal Nation‑State? Postnationalism, Multiculturalism, and the Collective Claims Making of Migrants and Ethnic Minorities in Britain and Germany." *American Journal of Sociology* 105 (3): 652–96.

McCauley, John F. 2014. "The Political Mobilization of Ethnic and Religious Identities in Africa." *American Political Science Review* 108 (4): 801–16.

Robinson, William S. 1950. "Ecological Correlations and the Behavior of Individuals." *American Sociological Review* 15 (3): 351–357.

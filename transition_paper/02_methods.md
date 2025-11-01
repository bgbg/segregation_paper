# Methods

## Data Sources and Scope

Throughout this paper, "country-level" refers to aggregated estimates across all ballot boxes meeting the 75% Haredi
threshold nationwide, not the entire Israeli electorate. Similarly, "city-level" refers only to qualifying Haredi boxes
within each city. This restriction ensures analysis captures Haredi behavior specifically.

This study uses official election results from the Central Elections Committee of Israel for Knesset elections 19
through 25, spanning from the January 2013 election (Knesset 19) through the November 2022 election (Knesset 25). These
datasets provide polling-stationlevel counts of registered voters, valid votes, and votes for each political party. The
analysis focuses on cities and towns with a significant ultra-Orthodox (Haredi) presence, following the same selection
framework as in my previous study on intra-Haredi ethnic segregation (Gorelik, 2025).


Voters are assigned to polling stations by residential address. Most ballot boxes contain 536650 registered voters
(interquartile range from recent elections). We identify Haredi ballot boxes using a 75% threshold: boxes where combined
votes for Shas and UTJ exceed three-quarters of all votes cast. This operational definition isolates predominantly
Haredi neighborhoods while maintaining consistency with the selection framework established in Gorelik (2025). To ensure
reliable statistical estimation, we include only cities with at least five qualifying boxes.

Data were obtained directly from the Central Elections Committee's official online repository, supplemented by
previously digitized archives for earlier elections (via Dr. Keren-Kratz, 2024). The resulting dataset includes
polling-stationlevel results for all eligible cities, with a focus on those exhibiting consistent Haredi voting
patterns.

### Ballot-Box Alignment and Preprocessing

Ballot boxes were matched between consecutive elections by city and ballot-box ID. Where multiple boxes shared the same
citybox pair across elections, their results were aggregated. Mismatched boxes that could not be aligned between
consecutive elections were excluded from that specific transition analysis, but remained available for analysis in
subsequent election pairs where they could be matched. This ensured that all included boxes represented consistent
geographical and demographic units across each election pair while maximizing data retention.


Abstention was defined as the difference between the number of registered voters in a ballot box and the number of valid
votes cast. This captures both non-voters and those who submitted invalid or blank ballots. This standard approach
treats non-participation and invalid ballots as a single "abstention" category, following common practice in aggregate
electoral analysis.

### Analytical Focus and Categories

Four electoral categories were modeled: (1) Shas (Sephardic ultra-Orthodox party), (2) UTJ (United Torah Judaism, the
Ashkenazi ultra-Orthodox party alliance), (3) Other parties (all non-Haredi political lists), and (4) Abstention
(registered but did not cast a valid vote). All non-Haredi parties are combined into a single "Other" category because
our focus is intra-Haredi dynamics; distinguishing among secular/religious-Zionist/Arab parties would add dimensionality
without illuminating Haredi behavior.

This categorization allows the model to capture both intra-Haredi transitions (between Shas and UTJ) and movements into
or out of active participation. While the majority of Shas voters are not Haredim, the restriction to homogeneous Haredi
ballot boxes ensures that nearly all Shas and UTJ votes in this sample originate from Haredi populations.

### Model Overview

The estimation framework follows a hierarchical Bayesian ecological inference model, designed to infer unobserved voter
transition matrices between consecutive elections from aggregate data. This problem is inherently challenging because
individual ballots are secret and only aggregate counts are available. Thus, we adopt a classic ecological inference
scenario (Robinson 1950; King 1997). The hierarchical structure pools information across localities, improving precision
in small samples while maintaining flexibility across elections. Full model specifications and diagnostics are provided
in Appendix A.

For each election pair $(t, t+1)$, the model infers the probability that a voter from category i at election t votes for
category j at election t+1. Individual transitions are unobserved, but aggregate vote counts provide marginal
constraints. Building on Goodman's (1953) ecological regression framework and subsequent Bayesian extensions (King 1997;
Rosen et al. 2001), we model voter transitions probabilistically using a hierarchical Bayesian approach that generalizes
the linear model to address several key challenges: unobserved individual transitions, spatial heterogeneity in voting
patterns, high dimensionality of transition matrices, and overdispersion in vote counts (Wakefield 2004; Forcina and
Pellegrino 2019).

**Important conceptual distinction:** The transition probabilities estimated by this model represent the *probability of
voting behavior* at each election, not the *movement of individual voters* between parties. When we observe that a
party's "retention rate" increases from one election pair to the next, this indicates that voters who supported that
party in the first election were more likely to support it again in the second election. However, this does not
necessarily mean that any voters returned to the party. Rather, it reflects that the "leak" of voters from that party to
other parties stoppedwhat we might call a return to a new steady state rather than the literal return of specific
voters.

Each election pair is modeled using three levels of hierarchy: (1) National transition matrix ($M^{country}$) represents
the baseline transition probabilities for the entire electorate, (2) Shared deviation pattern ($D$)a low-rank structure
capturing how cities collectively deviate from national trends, and (3) City-specific scalar ($\delta_c$)a single latent
variable scaling the deviation pattern for each city.

This hierarchical structure addresses a key limitation of traditional ecological inference, which often assumes a
uniform transition matrix for all units (the "uniform swing" assumption; Brown and Payne 1986). Such homogeneity
assumptions are unrealistic when voting patterns vary widely across regions (Johnston and Hay 1983). However, allowing
full city-specific transition matrices creates dimensionality challenges. We therefore impose a rank-1 structure on
city-level deviations: each city's deviation is captured by a single scalar $\delta_c$ that scales a shared deviation
pattern $D$. This approach reflects the insight that many electoral shifts are driven by one dominant cleavage (e.g.,
religious vs. secular, or urban vs. rural) rather than completely unique city-by-city patterns (Brown and Payne 1986;
Puig and Ginebra 2015).

City-level transition matrices are constructed as:

``` M^{(c)} = \text{softmax}(Z^{country} + \delta_c D) ```

where $Z^{country}$ are national logits transformed via the softmax function. This yields a compact yet flexible
structure that captures both overall national shifts and localized deviations, while enabling partial pooling: cities
with limited data shrink toward the national pattern, while those with strong evidence of deviation can diverge (Gelman
and Hill 2007).

### Priors and Likelihood

Transition logits are modeled using logistic-normal priors, avoiding the independence assumptions of Dirichlet
distributions that can impose unwarranted constraints on transition probabilities (Wakefield 2004; Glynn and Wakefield
2010). This approach works in unconstrained logit space and then transforms to probabilities via softmax. We add a
diagonal bias term $B$ to same-party transitions to reflect prior belief in voter inertiathe well-documented tendency of
voters to stick with their previous choice (Campbell, Green, and Layman 2011; Clarke et al. 2004). The unusually short
intervals between Israeli elections during this period (5.5 to 19 months) further strengthen this inertia prior, as
voters have less time for preference change between elections.

Priors were set as:

- Base logits: $\alpha_{ij} \sim N(0, 0.5^2)$ - Loyalty bias: $B \sim N(2.0, 0.3^2)$ - Deviation matrix: $D_{ij} \sim
N(0, 0.3^2)$ - City effects: $\delta_c \sim N(0, 0.5^2)$

Observed vote counts were modeled with a Dirichlet-multinomial likelihood with a fixed overdispersion parameter $\phi =
100$. This accounts for overdispersion beyond the multinomial assumption, recognizing that voters are not identically
distributed random draws but exhibit correlated behavior due to social networks and demographic clustering.

### Temporal Extension

The model is applied sequentially across subsequent election pairs from the January 2013 election (Knesset 19) through
the November 2022 election (Knesset 25).  Posterior means of $Z^{country}$, $D$, and $B$ from one transition are used as
prior means for the next, allowing gradual temporal evolution while maintaining continuity across election cycles.

### Model Validation

After fitting, we perform comprehensive diagnostics to validate model convergence and fit. All chains converged
successfully, with GelmanRubin $\hat{R} < 1.01$, effective sample sizes exceeding 400 for key parameters, and no
divergent transitions observed. Posterior predictive checks confirmed good fit between simulated and observed vote
counts.

### Implementation Details

Models were implemented in PyMC 5.0 (Python 3.11) on a Mac M2 Pro laptop using four chains, random seed 42. Scripts and
data-processing code are freely available on https://github.com/bgbg/segregation_paper.

### Relation to Previous Study

While this analysis differs conceptually and methodologically from my earlier paper on spatial ethnical segregation
(Gorelik 2025), both studies share the same data-selection logic. In both, ballot boxes are used as the atomic
observational units, representing small, demographically cohesive neighborhoods. Here, rather than studying residential
clustering, the focus is on temporal electoral dynamics  specifically, the persistence and permeability of Haredi voter
blocs.



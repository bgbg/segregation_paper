# Methods

## Data Sources and Scope

This study uses official election results from the Central Elections Committee of Israel for Knesset elections 19
through 25, spanning from the January 2013 election (Knesset 19) through the November 2022 election (Knesset 25). These
datasets provide polling-station-level counts of registered voters, valid votes, and votes for each political party. The
analysis focuses on cities and towns with a significant ultra-Orthodox (Haredi) presence, following the same selection
framework as in my previous study on intra-Haredi ethnic segregation (Gorelik, 2025). Throughout this paper,
"country-level" refers to aggregated estimates across all qualifying ballot boxes nationwide, while "city-level" refers
to qualifying boxes within each city.


Voters are assigned to polling stations by residential address. Most ballot boxes contain 536–650 registered voters
(interquartile range from recent elections). I identify Haredi ballot boxes using a 75% threshold: boxes where combined
votes for Shas and UTJ exceed three-quarters of all votes cast, maintaining consistency with Gorelik (2025). To ensure
reliable statistical estimation, I include only cities with at least five qualifying boxes.

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
without illuminating Haredi behavior. The restriction to homogeneous Haredi ballot boxes (≥75% Shas+UTJ votes) ensures
that nearly all Shas and UTJ votes in this sample originate from Haredi populations, isolating within-sector dynamics
from broader societal trends.

### Model Overview

The estimation framework follows a hierarchical Bayesian ecological inference model, designed to infer unobserved voter
transition matrices between consecutive elections from aggregate data. This problem is inherently challenging because
individual ballots are secret and only aggregate counts are available. Thus, I adopt a classic ecological inference
scenario (Robinson, 1950; King, 1997). The hierarchical structure pools information across localities, improving precision
in small samples while maintaining flexibility across elections. Full model specifications and diagnostics are provided
in Appendix A.

For each election pair $(t, t+1)$, the model infers the probability that a voter from category i at election t votes for
category j at election t+1. Individual transitions are unobserved, but aggregate vote counts provide marginal
constraints. Building on Goodman's (1953) ecological regression framework and subsequent Bayesian extensions (King, 1997;
Rosen et al., 2001), I model voter transitions probabilistically using a hierarchical Bayesian approach that generalizes
the linear model to address several key challenges: unobserved individual transitions, spatial heterogeneity in voting
patterns, high dimensionality of transition matrices, and overdispersion in vote counts (Wakefield, 2004; Forcina and
Pellegrino, 2019).

**Important conceptual distinction:** The transition probabilities represent the *probability of voting behavior* at each
election, not the *movement of individual voters* between parties. When a party's retention rate increases, this
indicates that leakage to other parties stopped, a return to a new steady state, rather than the literal return of specific
voters who previously defected.

Each election pair is modeled using three levels of hierarchy: (1) National transition matrix ($M^{country}$) represents
the baseline transition probabilities for the entire electorate, (2) Shared deviation pattern ($D$)a low-rank structure
capturing how cities collectively deviate from national trends, and (3) City-specific scalar ($\delta_c$)a single latent
variable scaling the deviation pattern for each city.

This hierarchical structure avoids the "uniform swing" assumption of traditional ecological inference while managing
dimensionality. I impose a rank-1 structure on city-level deviations: each city's deviation is captured by a single
scalar $\delta_c$ that scales a shared deviation pattern $D$, reflecting that electoral shifts are often driven by one
dominant cleavage rather than completely unique city-by-city patterns (Brown and Payne, 1986; Puig and Ginebra, 2015).

City-level transition matrices are constructed as:

$$M^{(c)} = \text{softmax}(Z^{country} + \delta_c D)$$

where $Z^{country}$ are national logits transformed via the softmax function. This yields a compact yet flexible
structure that captures both overall national shifts and localized deviations, while enabling partial pooling: cities
with limited data shrink toward the national pattern, while those with strong evidence of deviation can diverge (Gelman
and Hill, 2007).

### Priors and Likelihood

Transition logits are modeled using logistic-normal priors, avoiding the independence assumptions of Dirichlet
distributions (Wakefield, 2004; Glynn and Wakefield, 2010). I add a diagonal bias term $B$ to same-party transitions to
reflect voter inertia, the well-documented tendency to stick with previous choices (Campbell et al., 2011;
Clarke et al., 2004). The unusually short intervals between Israeli elections during this period (5.5 to 19 months)
further strengthen this inertia prior. Observed vote counts were modeled with a Dirichlet-multinomial likelihood with
overdispersion parameter $\phi = 100$, accounting for correlated behavior due to social networks and demographic
clustering. Full prior specifications are detailed in Appendix A.

### Temporal Extension

The model is applied sequentially across subsequent election pairs from the January 2013 election (Knesset 19) through
the November 2022 election (Knesset 25).  Posterior means of $Z^{country}$, $D$, and $B$ from one transition are used as
prior means for the next, allowing gradual temporal evolution while maintaining continuity across election cycles.

### Model Validation and Implementation

Country-level transition matrix parameters converged well across all election pairs (Gelman-Rubin $\hat{R} < 1.01$,
effective sample sizes exceeding 6,500 for all country-level parameters, no divergent transitions). City-deviation
parameters exhibited non-convergence in some pairs due to the multiplicative non-identifiability inherent in the
$\delta \cdot D$ interaction structure; city-level estimates for affected transitions should be interpreted with
caution (see Appendix B for full diagnostics). Posterior predictive checks confirmed good fit. Models were implemented
in PyMC 5.0 using four chains with 3,000 draws and 5,000 tuning steps per chain (target acceptance 0.99, seed 42).
Full diagnostics and code are available at https://github.com/bgbg/segregation_paper.

### Relation to Previous Study

While this analysis differs conceptually and methodologically from my earlier paper on spatial ethnical segregation
(Gorelik, 2025), both studies share the same data-selection logic. In both, ballot boxes are used as the atomic
observational units, representing small, demographically cohesive neighborhoods. Here, rather than studying residential
clustering, the focus is on temporal electoral dynamics  specifically, the persistence and permeability of Haredi voter
blocs.

**Note on notation:** Throughout this paper, election transitions are denoted using the arrow notation (→) to indicate
the direction from one election to the next. For example, "23→24" refers to the transition from the March 2020 election
(Knesset 23) to the March 2021 election (Knesset 24). This notation emphasizes that I am examining voter flows
between consecutive elections rather than the elections themselves in isolation.

### Corpus Analysis of Haredi Media

To investigate the mechanisms behind the observed electoral disruption, I constructed a corpus of Haredi news articles
and forum discussions from the two dominant ultra-Orthodox news websites in Israel: Behadrey Haredim (bhol.co.il) and
Kikar HaShabbat (kikar.co.il). The corpus was assembled by systematically scraping all available content published
between March 2020 and March 2021, yielding approximately 58,000 items: 25,000 news articles from Behadrey Haredim,
26,000 news articles from Kikar HaShabbat, and 7,000 forum discussion threads from the Behadrey Haredim forums.

Each item was stored as a structured record containing publication date, title, full text, author, and source URL. Forum
threads additionally preserved individual posts with per-post dates and authors, capturing the temporal development of
community discussions.

I searched the corpus systematically using Hebrew-language keyword queries organized around the research hypotheses:
rabbinic voting instructions, intra-Haredi party friction, leadership crises, and COVID-related political upheaval. The
initial automated filtering reduced the corpus to several hundred candidate articles, which were then read and evaluated
manually. Each claim presented in the "What Caused the 23-24 Disruption?" section is supported by verbatim citations
from the original sources, with full source metadata (publication date, outlet, and URL) provided in the supplementary
materials.

The following section presents the estimated transition patterns at national and city levels, revealing both the
dominant pattern of high stability and a dramatic temporary disruption in the March 2020–March 2021 transition.



# Methods

**TODO** In Methods: make it absolutely clear that "country-wide" and "city-specific" to not talk about
actual country or entire city, but about Haredi populaiton hubs in the entire country or in each city,
as defined by the selection method!

## Data Sources and Scope

This study uses official election results from the Central Elections Committee of Israel for Knesset elections 19
through 25, spanning from the January 2013 election (Knesset 19) through the November 2022 election (Knesset 25). These datasets provide polling-stationlevel counts of
registered voters, valid votes, and votes for each political party. The analysis focuses on cities and towns with a
significant ultra-Orthodox (Haredi) presence, following the same selection framework as in my previous study on
intra-Haredi ethnic segregation (Gorelik, 2025).


**TODO** Make sure the following two paragraphs are not a verbatim copy of my previous work.

Each Israeli citizen votes in a polling station determined by their residential address. Polling stations may include
one or more ballot boxes, each typically serving between 536 and 650 registered voters (25th75th percentiles, based on
recent elections). A Haredi voting box is defined as one where at least 75% of the votes are cast for the Haredi parties
Shas and UTJ, indicating predominantly Haredi voters. Cities with fewer than five such boxes were
excluded from the analysis to ensure stable estimation.

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


**TODO** - verify that in the paragraph below the references are indeed relevant

Abstention was defined as the difference between the number of registered voters in a ballot box and the number of valid
votes cast. This captures both non-voters and those who submitted invalid or blank ballots. Similar approaches have been
used in analyses of abstention and invalid voting behavior in Israel, where spoiled ballots are typically recorded
alongside nonparticipation in aggregate analyses (see Diskin, Diskin, and Hazan 2005; Arian and Shamir 2008).

### Analytical Focus and Categories


**TODO** in the paragraph below: should I define UTJ again?
**TODO** in the paragraph below: should I explain why combining all other parties into a single category?

Four electoral categories were modeled: 1. Shas (Sephardic ultra-Orthodox party) 2. UTJ (Ashkenazi
ultra-Orthodox party) 3. Other parties (all non-Haredi political lists) 4. Abstention (registered but did not cast a
valid vote)

This categorization allows the model to capture both intra-Haredi transitions (between Shas and UTJ) and movements into
or out of active participation. While the majority of Shas voters are not Haredim, the restriction to homogeneous Haredi
ballot boxes ensures that nearly all Shas and UTJ votes in this sample originate from Haredi populations.

### Model Overview


**TODO** Review the paragraph below


The estimation framework follows a hierarchical Bayesian ecological inference model, designed to infer unobserved voter
transition matrices between consecutive elections from aggregate data. This problem is inherently challenging because
individual ballots are secret and only aggregate counts are available. Thus, we adopt a classic ecological inference scenario (Robinson
1950; King 1997).

**TODO** Consider removing the (Wakefield 2004; Forcina and Pellegrino 2019) references

For each election pair $(t, t+1)$, the model infers the probability that a voter from category i at election t votes for
category j at election t+1. Individual transitions are unobserved, but aggregate vote counts provide marginal
constraints. Building on Goodman's (1953) ecological regression framework and subsequent Bayesian extensions (King 1997; Rosen et al. 2001), we model voter transitions probabilistically using a hierarchical Bayesian approach that generalizes the linear model to address several key challenges: unobserved individual transitions, spatial heterogeneity in voting patterns, high dimensionality of transition matrices, and overdispersion in vote counts (Wakefield 2004; Forcina and Pellegrino 2019).

**Important conceptual distinction:** The transition probabilities estimated by this model represent the *probability of voting behavior* at each election, not the *movement of individual voters* between parties. When we observe that a party's "retention rate" increases from one election pair to the next, this indicates that voters who supported that party in the first election were more likely to support it again in the second election. However, this does not necessarily mean that any voters returned to the party. Rather, it reflects that the "leak" of voters from that party to other parties stopped—what we might call a return to a new steady state rather than the literal return of specific voters.

Each election pair is modeled using three levels of hierarchy: 1. National transition matrix ($M^{country}$) represents the baseline
transition probabilities for the entire electorate.  2. Shared deviation pattern ($D$)  a low-rank structure capturing
how cities collectively deviate from national trends.  3. City-specific scalar ($\delta_c$)  a single latent variable
scaling the deviation pattern for each city.


**TODO** based on the paragraphs are the references in the paragraph below relevant to it?

This hierarchical structure addresses a key limitation of traditional ecological inference, which often assumes a
uniform transition matrix for all units (the "uniform swing" assumption; Brown and Payne 1986). Such homogeneity
assumptions are unrealistic when voting patterns vary widely across regions (Johnston and Hay 1983). However, allowing
full city-specific transition matrices creates dimensionality challenges. We therefore impose a rank-1 structure on
city-level deviations: each city's deviation is captured by a single scalar $\delta_c$ that scales a shared deviation
pattern $D$. This approach reflects the insight that many electoral shifts are driven by one dominant
cleavage (e.g., religious vs. secular, or urban vs. rural) rather than completely unique city-by-city patterns (Brown
and Payne 1986; Puig and Ginebra 2015).

City-level transition matrices are constructed as:

``` M^{(c)} = \text{softmax}(Z^{country} + \delta_c D) ```

**TODO** Replce "parasimonious" in the paragraph below with a more common word. Compact?

where $Z^{country}$ are national logits transformed via the softmax function. This yields a parsimonious yet flexible
structure that captures both overall national shifts and localized deviations, while enabling partial pooling: cities
with limited data shrink toward the national pattern, while those with strong evidence of deviation can diverge (Gelman
and Hill 2007).

### Priors and Likelihood

**TODO** in the paragraph below, the last sentence mentions the period of political instability. Hormonize the entire paragraph. Feel free to remove this information at all if it does not add enough value

Transition logits are modeled using logistic-normal priors, avoiding the independence assumptions of Dirichlet
distributions that can impose unwarranted constraints on transition probabilities (Wakefield 2004; Glynn and Wakefield
2010). This approach works in unconstrained logit space and then transforms to probabilities via softmax. We add a
diagonal bias term $B$ to same-party transitions to reflect prior belief in voter inertiathe well-documented tendency of
voters to stick with their previous choice (Campbell, Green, and Layman 2011; Clarke et al. 2004). This belief is even stronger
due to the short periods of time between subsequent elections in Israel in the study period.

Priors were set as:

- Base logits: $\alpha_{ij} \sim N(0, 0.5^2)$ - Loyalty bias: $B \sim N(2.0, 0.3^2)$ - Deviation matrix: $D_{ij} \sim
N(0, 0.3^2)$ - City effects: $\delta_c \sim N(0, 0.5^2)$

Observed vote counts were modeled with a Dirichlet-multinomial likelihood with a fixed overdispersion parameter $\phi =
100$. This accounts for overdispersion beyond the multinomial assumption, recognizing that voters are not identically
distributed random draws but exhibit correlated behavior due to social networks and demographic clustering.

### Temporal Extension

The model is applied sequentially across subsequent election pairs from the January 2013 election (Knesset 19) through the November 2022 election (Knesset 25).
Posterior means of $Z^{country}$, $D$, and
$B$ from one transition are used as prior means for the next, allowing gradual temporal evolution while maintaining
continuity across election cycles.

### Model Validation

After fitting, we perform comprehensive diagnostics to validate model convergence and fit. All chains converged
successfully, with GelmanRubin $\hat{R} < 1.01$, effective sample sizes exceeding 400 for key parameters, and no
divergent transitions observed. Posterior predictive checks confirmed good fit between simulated and observed vote
counts.

### Implementation Details

**TODO** add github link

Models were implemented in PyMC 5.0 (Python 3.11) on a Mac M2 Pro laptop using four chains, random seed 42. Scripts and
data-processing code are freely available on  https://XXXXX.

### Relation to Previous Study

While this analysis differs conceptually and methodologically from my earlier paper on spatial ethnical segregation (Gorelik
2025), both studies share the same data-selection logic. In both, ballot boxes are used as the atomic observational
units, representing small, demographically cohesive neighborhoods. Here, rather than studying residential clustering,
the focus is on temporal electoral dynamics  specifically, the persistence and permeability of Haredi voter blocs.



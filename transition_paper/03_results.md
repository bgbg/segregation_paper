# Results

## Overview

This section presents the results of the Bayesian ecological model estimating voter transitions among the two main
Haredi parties  *Shas* and *United Torah Judaism (UTJ)*  and between these parties, other political groups, and
abstention. The results cover transitions between consecutive Knesset elections from the 19th (2013) to the 25th (2022).

## Country-Level Transitions

At the national level, the transition matrices reveal strong voter loyalty within both Haredi parties. Shas retained on
average more than 90% of its voters across elections, while UTJ consistently preserved above 95%. However, the magnitude
of "within-bloc permeability"  voters shifting between Shas and UTJ  varied notably over time.

The 20202021 election pair (Knesset 23 to 24) showed an unusual and dramatic decline in intra-Haredi loyalty,
particularly among Shas voters. As shown in Figure 2, Shas-to-Shas loyalty plummeted from 98.9% (in the 2223 transition)
to just 72.8% in the 2324 transition. Simultaneously, the probability of Shas voters switching to UTJ jumped from near
zero to 12.3%. UTJ voters also experienced reduced loyaltydropping from 97.6% to 87.9%with 12.3% of UTJ voters defecting
to Shas. This cross-flow pattern represents an unprecedented disruption in the typically stable Haredi voting bloc.
Notably, this disruption was temporary: in the subsequent 2425 transition, loyalty rates recovered to pre-crisis levels
(Shas: 98.6%, UTJ: 96.7%).

The timing of these elections is crucial for interpretation. As shown in Table 1, this period was marked by political
instability with elections occurring in rapid succession: only 5.5 months separated Knesset 22 (September 2019) and 23
(March 2020), followed by a 12.7-month interval to Knesset 24 (March 2021), and a longer 19.3-month gap to Knesset 25
(November 2022). These short intervalsparticularly the 5.5 and 12.7-month gapsstrongly suggest that the observed
transitions reflect genuine voter switching rather than demographic change through migration or generational
replacement. The close temporal proximity makes it unlikely that residential mobility or population turnover could
account for the dramatic shifts in voting patterns.

The aggregated country-level transition matrix (Figure 2) demonstrates that abstention rates remained low and stable
within the Haredi population, contrasting with fluctuating participation trends among non-Haredi groups. Cross-over
voting between Haredi and non-Haredi parties remained marginal, indicating persistent political segmentation despite
broader electoral turbulence.

**Table 1: Election Dates and Intervals**

| Knesset | Election Date | Days Since Previous | Months Since Previous |
|---------|--------------|---------------------|----------------------| | 22 | September 17, 2019 |  |  | | 23 | March
2, 2020 | 167 | 5.5 | | 24 | March 23, 2021 | 386 | 12.7 | | 25 | November 1, 2022 | 588 | 19.3 |

![Country transitions](plots/country_transition_matrix_over_elections.png) *Figure 2: Country-level transition matrices
across election pairs*

## City-Level Variation

While the national patterns indicate high stability, city-level analysis reveals substantial variation in the magnitude
of the 2324 loyalty disruption across Haredi strongholds. The temporary collapse in Shas loyalty was not uniform but
varied significantly by city, reflecting distinct local political dynamics.

Figure 3 provides a detailed view of Shas-to-Shas loyalty trajectories across all major Haredi cities, highlighting the
dramatic 2324 dip and subsequent recovery. This visualization reveals several key patterns. First, the disruption was
system-wide: every city experienced reduced Shas loyalty in the 2324 transition, with no exceptions. Second, despite the
universal pattern, the magnitude varied substantiallyAshdod (blue line) shows the steepest drop to approximately 65%,
while other cities cluster between 70-78%. Third, the recovery in 2425 was equally universal and nearly complete, with
all cities returning to loyalty rates above 95%. This synchronized pattern of disruption and recovery across
geographically dispersed cities points to a centralized shock affecting the entire Haredi political system rather than
independent local factors.

Notably, this temporal pattern in Shas voter loyalty mirrors the segregation index dynamics observed in my previous
study (Gorelik 2025, shown in Figure 1). Just as Ashdod's segregation index exhibited a sharp drop between Knesset 23
and 24 followed by rapid recovery, the Shas loyalty rates show an identical V-shaped pattern in the same time window.
This parallel provides strong evidence that the segregation anomaly was indeed driven by voting behavior changes rather
than residential mobility: when Sephardic Haredi voters temporarily defected from Shas to UTJ, ballot boxes that had
been homogeneously Shas (marking Sephardic neighborhoods) suddenly showed mixed voting patterns, artificially reducing
the measured residential segregation between ethnic groups. The fact that both the loyalty rates and segregation indices
recovered in tandem confirms that the underlying residential structure remained stable throughoutonly political behavior
fluctuated.

![Shas to Shas City Comparison](plots/shas_shas_city_comparison.png) *Figure 3: Shas-to-Shas transition probabilities
across cities and election pairs. All cities show the characteristic 2324 loyalty drop followed by 2425 recovery, with
Ashdod (blue) exhibiting the most extreme deviation. The temporal pattern closely matches the segregation index dynamics
shown in Figure 1.*

**Ashdod** exhibited the most dramatic deviation from national patterns. As shown in Figure 4, Shas-to-Shas loyalty in
the 2324 transition dropped to just 64.5%even lower than the national 72.8%while the Shas-to-UTJ switching rate surged
to 21.3%, nearly double the national rate of 12.3%. This exceptionally high cross-party flow in Ashdod suggests
particularly fluid political boundaries between Sephardic and Ashkenazi Haredi communities in this mixed city. In the
subsequent 2425 transition, Ashdod's Shas loyalty recovered to 99.2%, closely tracking the national recovery pattern.

![ashdod transitions](plots/city_ashdod_transition_matrix_over_elections.png) *Figure 4: Ashdod transition matrices
showing pronounced Shas-to-UTJ switching in 2324*

**Beit Shemesh** and **Bnei Brak** showed similar but somewhat more moderate disruptions. In Beit Shemesh (Figure 5),
Shas-to-Shas loyalty fell to 75.2% in the 2324 transition, with 9.5% switching to UTJ. Bnei Brak (Figure 6), despite
being a predominantly Ashkenazi ultra-Orthodox stronghold, experienced an even sharper Shas loyalty drop to 69.5%, with
15.8% of Shas voters moving to UTJ. Both cities fully recovered by the 2425 transition, with Shas loyalty returning to
approximately 99% and Shas-to-UTJ flows dropping back to near zero.

![beit shemesh transitions](plots/city_beit_shemesh_transition_matrix_over_elections.png) *Figure 5: Beit Shemesh
transition matrices*

![bnei brak transitions](plots/city_bnei_brak_transition_matrix_over_elections.png) *Figure 6: Bnei Brak transition
matrices*

The consistency of this temporal pattern across citiessharp disruption in 2324 followed by full recovery in 2425strongly
suggests a coordinated or system-wide shock rather than city-specific factors. The variation in magnitude, however,
indicates that local socio-political contexts modulated the intensity of the disruption. Cities with more integrated
Sephardic-Ashkenazi populations (Ashdod, Bnei Brak) showed larger cross-flows than more homogeneous communities.

![City MAD subplots](plots/cities_aggregate_deviation_comparison.png) *Figure 7: Mean absolute deviation comparison
across cities*

## Connection to Residential Segregation Patterns

The dramatic Shas-to-UTJ switching in Ashdod during the 2324 transition provides a compelling explanation for an anomaly
observed in my previous study (Gorelik 2025). That study documented a sudden, unexplained drop in the residential
segregation index (dissimilarity) between Ashkenazi and Sephardic Haredim in Ashdod around the 2021 election period.
Since no major demographic relocation or institutional change occurred at that time, the segregation drop appeared
puzzling.

The present analysis reveals the mechanism: the 21.3% of Shas voters who switched to UTJ in Ashdod during this period
created a temporary geographic mixing of voting patterns. Ballot boxes that had been predominantly Shas (Sephardic)
suddenly showed substantial UTJ (Ashkenazi) support, reducing the spatial correlation between ethnicity and party
voteprecisely what the segregation index measures. This occurred without physical residential movement; rather, voters
crossed ethnic-political boundaries at the ballot box. The subsequent recovery of both Shas loyalty (to 99.2% in 2425)
and the segregation index to previous levels confirms that this was a temporary electoral realignment rather than a
permanent demographic shift. This finding demonstrates that electoral transitions can manifest as apparent "integration"
in segregation indices even when residential patterns remain stable.

## Other Notable Disruptions Beyond 2324

While the 2324 transition represents the most dramatic disruption, several other election pairs exhibited noteworthy
volatility at both national and city levels, indicating that hidden voter switching occurs periodically even in this
highly cohesive electorate.

**Country-level disruptions:** - **Kn1920 (20132015):** Shas-to-Shas loyalty dropped from 91.8% to 85.0%, with 13.8%
defecting to "Other" parties. This represents an early instance of Shas voters exploring non-Haredi options, possibly
reflecting dissatisfaction with sectoral leadership during that period.  - **Kn2021 (20152019):** A dramatic collapse in
"Other-to-Other" loyalty from 90.7% to 57.8%, with substantial flows toward both Haredi parties (13.2% to Shas, 28.7% to
UTJ). This suggests non-Haredi voters in predominantly Haredi ballot boxes increasingly aligned with sectoral parties,
possibly reflecting neighborhood demographic changes or political realignment.  - **Kn2324 (20202021):** Beyond the
Shas-UTJ cross-flows, this transition also showed unusual mobilization patterns, with 10.0% of previous abstainers
voting for Shassuggesting either turnout efforts or new voter registration in Shas strongholds during this turbulent
period.

**Ashdod-specific disruptions:** - **Kn1819 and Kn1920:** Shas loyalty in Ashdod showed early volatility, dropping from
91.8% (kn1819) to 83.6% (kn1920), foreshadowing the more dramatic 2324 collapse. This suggests Ashdod's Sephardic Haredi
population was already experiencing political flux before the system-wide shock.  - **Kn2021:** A striking collapse in
"Other-to-Other" retention to just 48.7% (compared to 57.8% nationally), with 41.7% of "Other" voters switching to UTJan
unprecedented cross-sectoral flow. This spike from near-zero to over 40% indicates either non-Haredi residents in mixed
neighborhoods adopting Haredi political allegiances, or measurement artifacts from changing ballot box compositions.

These patterns reveal that voter transitions in the Haredi sector are not confined to singular crisis moments but occur
episodically, with certain cities (particularly Ashdod) serving as bellwethers of broader instability. The combination
of national and local disruptions suggests both system-wide political shocks and city-specific social dynamics shape
voting volatility.

## Quantifying Permeability

To quantify variation in voter stability, I computed the Mean Absolute Deviation (MAD) between modeled and observed vote
shares across all city-level election pairs. MAD serves as a synthetic indicator of political permeability and temporal
volatility. Lower values indicate consistent voting patterns; higher values denote flux between parties or toward
abstention.

| City | Mean MAD (pp) | Max MAD (pp) | |------|----------------|--------------| | Ashdod | 1.0 | 2.4 | | Beit Shemesh |
0.6 | 1.5 | | Elad | 2.6 | 8.7 | | Bnei Brak | 1.0 | 3.7 | | Jerusalem | 0.6 | 1.0 | | Modi'in Illit | 2.2 | 10.6 |

Two clusters emerge: highly stable cities (Jerusalem, Beit Shemesh, Ashdod) and relatively fluid ones (Elad, Modi'in
Illit), suggesting differing degrees of internal cohesion. These differences likely reflect demographic composition,
institutional affiliations, and rabbinic leadership structures rather than socioeconomic variation. Figure 8 shows the
temporal pattern of MAD values across cities and election pairs, highlighting the 2324 disruption as a clear outlier
across all locations.

## Model Validation

Posterior predictive checks confirm that the model adequately reproduces empirical vote counts across all elections.
Most chains converged successfully, with effective sample sizes (ESS) exceeding 400 for key parameters and R-hat values
approaching 1.01 in later runs. Minor convergence issues in earlier election pairs (20132015) were resolved by
increasing the number of draws and adopting non-centered parameterization.

![MAD Table Lens](plots/mad_table_lens.png) *Figure 8: MAD rankings by election pair and city, visualizing temporal
patterns of deviation*

## Summary of Findings

1. **Dominant pattern of high stability:** Across most election pairs, Shas and UTJ maintained loyalty rates above 95%,
confirming the general portrait of Haredi voting as highly disciplined and predictable.

2. **Sharp temporary disruption in 2324:** The March 2020 to March 2021 transition showed unprecedented cross-party
switching. Shas loyalty dropped to 72.8% nationally (64.5% in Ashdod), with 12.3% of Shas voters switching to UTJ (21.3%
in Ashdod). UTJ also experienced reduced loyalty (87.9%), with substantial flows to Shas.

3. **Full recovery in 2425:** Loyalty patterns returned to pre-crisis levels by the November 2022 election, indicating
the disruption was temporary rather than a permanent realignment.

4. **Geographic variation in disruption magnitude:** The 2324 shock affected all cities but varied in intensity. Ashdod
showed the largest deviations, followed by Bnei Brak and Beit Shemesh, suggesting that local demographic integration
modulates electoral volatility.

5. **Temporal proximity rules out demographic explanations:** The short intervals between elections (12-19 months) make
it unlikely that migration or generational turnover caused the observed transitions. The patterns reflect genuine voter
switching.

6. **Stable abstention rates:** Haredi participation remained resilient throughout, even during the 2324 disruption.
Abstention flows remained minimal, contrasting with the significant inter-party switching.

These findings reveal that even highly cohesive voting blocs can experience rapid, system-wide disruptions in loyalty
patternsbut that such disruptions may be reversible when triggering conditions resolve. The consistency across cities
points to centralized political dynamics, while the variation in magnitude highlights how local social integration
shapes the boundaries of permissible switching.

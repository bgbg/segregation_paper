# Results

The analysis confirms high baseline loyalty within both Shas and UTJ, yet reveals significant temporal and geographic
variation in voter transitions. Most strikingly, a dramatic but temporary disruption in the March 2020–March 2021
transition (Knesset 23→24) affected all major Haredi cities, followed by complete recovery. These findings demonstrate
that even highly disciplined voting blocs can experience rapid, system-wide loyalty disruptions that fully reverse when
triggering conditions resolve. Model diagnostics confirming adequate fit are provided in Appendix B.

**Note on notation:** Throughout this paper, election transitions are denoted using the arrow notation (→) to indicate
the direction from one election to the next. For example, "23→24" refers to the transition from the March 2020 election
(Knesset 23) to the March 2021 election (Knesset 24). This notation emphasizes that we are examining voter flows
between consecutive elections rather than the elections themselves in isolation.

## Country-Level Transitions

At the national level, the transition matrices reveal strong voter loyalty within both Haredi parties. Shas retained on
average more than 90% of its voters across elections, while UTJ consistently preserved above 95%. However, the magnitude
of "within-bloc permeability"  voters shifting between Shas and UTJ  varied notably over time.


The March 2020–March 2021 transition (Knesset 23→24) showed an unusual and dramatic decline in intra-Haredi loyalty,
particularly among Shas voters. At the country level (Figure 2), Shas-to-Shas loyalty plummeted from 98.9% (in the
September 2019–March 2020 transition, Knesset 22→23) to just 73.5% in the 23→24 transition. Simultaneously, the
probability of Shas voters switching to UTJ jumped from near zero to 12.3%. UTJ voters also experienced reduced
loyalty—dropping from 96.6% to 87.9%, with 4.6% of UTJ voters defecting to Shas. The estimated decline in party
retention corresponds to roughly one parliamentary seat per party, illustrating the political significance of even modest
swings in Haredi voting patterns. The temporary drop thus had a tangible potential to alter coalition outcomes. This
cross-flow pattern represents an unprecedented disruption in the typically stable Haredi voting bloc. Critically, this
disruption was observed across multiple geographic scales: both at the national level (Figure 2) and across individual
cities (Figure 3), indicating a system-wide rather than localized phenomenon. Notably, this disruption was temporary: in
the subsequent March 2021–November 2022 transition (Knesset 24→25), loyalty rates recovered substantially (Shas: 96.9%,
UTJ: 95.5%).

**Important clarification:** This recovery in retention probabilities does not indicate that individual voters who had
"strayed" from Shas returned to the party. Rather, it reflects that the "leak" of voters from Shas to other parties
stopped. The actual voter movements are captured by the off-diagonal elements of the transition matrix—the flows between
parties—not by the diagonal retention rates themselves. Critically, none of the off-diagonal transitions into Shas
(UTJ→Shas, Other→Shas, Abstain→Shas) showed unusual spikes in the March 2021–November 2022 transition (Knesset 24→25),
confirming that voters who left Shas in the 23→24 disruption did not return.

Paradoxically, despite losing core Haredi voters in 23→24 without recovering them in 24→25, Shas's national vote share
increased from 7.17% to 8.25% (9 to 11 seats). Since Haredi population hubs show no corresponding Shas influx, this
growth originated from voters outside major Haredi centers. This demonstrates how aggregate vote-share growth can mask
internal dynamics: Shas simultaneously lost its core ultra-Orthodox base (to UTJ in 23→24) while gaining peripheral
traditional Sephardic supporters.

The timing of these elections is crucial for interpretation. As shown in Table 1, this period was marked by political
instability with elections occurring in rapid succession: only 5.5 months separated the September 2019 (Knesset 22) and
March 2020 (Knesset 23) elections, followed by a 13-month interval to the March 2021 election (Knesset 24), and a
longer 19-month gap to the November 2022 election (Knesset 25). These short intervals, particularly the 5.5 and
13-month gaps, strongly suggest that the observed
transitions reflect genuine voter switching rather than demographic change through migration or generational
replacement. The close temporal proximity makes it unlikely that residential mobility or population turnover could
account for the dramatic shifts in voting patterns.

Figure 2 shows that Haredi abstention rates remained low and stable, contrasting with fluctuating non-Haredi
participation. Cross-over voting between Haredi and non-Haredi parties remained marginal, indicating persistent political
segmentation despite broader electoral turbulence.

**Table 1: Election Dates and Intervals**

| Knesset | Election Date | Days Since Previous | Months Since Previous |
|---------|---------------|---------------------|----------------------|
| 22 | September 17, 2019 | | |
| 23 | March 2, 2020 | 167 | 5.5 |
| 24 | March 23, 2021 | 386 | 12.7 |
| 25 | November 1, 2022 | 588 | 19.3 |


![Country transitions](plots/country_transition_matrix_over_elections.png)

*Figure 2: Country-level transition matrices across election pairs*


## City-Level Variation

While the national patterns indicate high stability, city-level analysis reveals substantial variation in the magnitude
of the March 2020–March 2021 (23→24) loyalty disruption across Haredi strongholds. Figure 3 demonstrates that the
disruption was universal—every city experienced reduced Shas loyalty in the 23→24 transition—but the magnitude varied
substantially. Ashdod (blue line) shows the steepest drop to approximately 65%, while other cities cluster between
70–78%. The recovery in the March 2021–November 2022 transition (24→25) was equally universal and nearly complete, with
all cities returning to loyalty rates above 95%.

![Shas to Shas City Comparison](plots/shas_shas_city_comparison.png)

*Figure 3: Shas-to-Shas transition probabilities across cities and election pairs. All cities show the characteristic
March 2020–March 2021 (23→24) loyalty drop followed by March 2021–November 2022 (24→25) recovery, with Ashdod (blue)
exhibiting the most extreme deviation.*

**Ashdod** exhibited the most dramatic deviation from national patterns. As shown in Figure 4, Shas-to-Shas loyalty
dropped to just 67.1% (compared to 73.5% nationally), while the Shas-to-UTJ switching rate surged to 19.3%—more than
50% higher than the national rate of 12.3%. Ashdod's experience can be contextualized through anecdotal accounts of
community leadership disputes reported during that period, suggesting local religious authority fragmentation may have
temporarily influenced voting behavior. This exceptionally high cross-party flow suggests particularly fluid political
boundaries between Sephardic and Ashkenazi Haredi communities in this mixed city. In the subsequent 24→25 transition,
Ashdod's Shas loyalty recovered to 96.9%, closely tracking the national pattern.

![ashdod transitions](plots/city_ashdod_transition_matrix_over_elections.png)

*Figure 4: Ashdod transition matrices showing pronounced Shas-to-UTJ switching in the March 2020–March 2021 transition
(23→24)*

Other major Haredi cities showed similar but more moderate disruptions. Beit Shemesh experienced a Shas loyalty drop to
75.2% (with 9.5% switching to UTJ), while Bnei Brak—despite being a predominantly Ashkenazi stronghold—saw Shas loyalty
fall to 70.9% (with 15.8% switching to UTJ). Both cities fully recovered by the 24→25 transition. Detailed transition
matrices for these cities are provided in Appendix A.

The consistency of this temporal pattern across cities—sharp disruption followed by full recovery—strongly suggests a
coordinated or system-wide shock rather than city-specific factors. However, the variation in magnitude indicates that
local socio-political contexts modulated the intensity: cities with more integrated Sephardic-Ashkenazi populations
(Ashdod, Bnei Brak) showed larger cross-flows than more homogeneous communities. Notably, while this disruption is
dramatic in the quantitative data, it appears to have received limited attention in contemporary political discourse,
suggesting that even substantial voter transitions within the Haredi sector can remain largely hidden beneath the
surface of stable electoral outcomes (both parties maintained their coalition positions despite the internal
reshuffling).

Importantly, while the voter transition disruption was universal across all major Haredi cities, only Ashdod exhibited
a dramatic drop in residential segregation indices in my previous study (Gorelik, 2025). This city-specific segregation
response, despite the system-wide voting disruption, proved methodologically revealing. Ashdod's exceptionally high
switching rate (19.3% compared to 12.3% nationally) apparently crossed a threshold that made the confounding between
voting patterns and residential segregation visible in the dissimilarity index. This fortunate coincidence—that the city
with the most extreme switching happened to be the same city where segregation indices were carefully tracked—allowed
identification of a fundamental measurement issue: using party votes as proxies for ethnic residential patterns can be
compromised when voter loyalty fluctuates. Had the switching been more uniform across cities, or had Ashdod's rate been
closer to the national average, this methodological confound might have remained undetected.

## Connection to Residential Segregation Patterns

The 19.3% Shas-to-UTJ switching in Ashdod created apparent "integration" in segregation indices without actual
residential movement. Ballot boxes previously dominated by Shas (Sephardic) suddenly showed substantial UTJ (Ashkenazi)
support, reducing the spatial correlation between ethnicity and party vote—precisely what the dissimilarity index
measures. The synchronized recovery of both voter loyalty (to 96.9% in 24→25) and segregation levels confirms this
electoral mechanism.

This reveals a critical methodological limitation: using party votes as proxies for ethnic residential patterns assumes
stable political loyalty. The proxy works well when ethnic-political boundaries are rigid but fails during loyalty
disruptions. This temporally contingent validity has implications for any study conflating political and demographic
boundaries—a pattern potentially observable in other contexts where voting behavior proxies for group membership
(religious denominations, immigrant origin groups, linguistic communities).

## Other Notable Disruptions Beyond The March 2020–March 2021 Transition

While the March 2020–March 2021 transition (23→24) represents the most dramatic disruption, the data reveal episodic
deviations from typical loyalty patterns. At the national level, the January 2013–March 2015 transition (19→20) showed
Shas-to-Shas loyalty dropping to 85.0%, with 13.8% defecting to the "Other" category (comprising numerous non-Haredi
parties)—an early instance of Shas voters exploring options outside the Haredi sectoral bloc, possibly reflecting
dissatisfaction with party leadership during that period.

More significantly, Ashdod exhibited early signs of instability in the transitions immediately preceding the 23→24
collapse. Shas loyalty in Ashdod declined from 91.8% in the April 2019–September 2019 transition (21→22) to 83.6% in the
September 2019–March 2020 transition (22→23), well before the system-wide shock. This suggests that Ashdod's Sephardic
Haredi population was already experiencing political flux, making the city particularly vulnerable to the subsequent
disruption.

These patterns indicate that voter transitions in the Haredi sector are not confined to singular crisis moments but
occur episodically, with certain cities—particularly Ashdod—serving as bellwethers of broader instability.

## Quantifying Permeability

To quantify variation in voter stability, I computed the Mean Absolute Deviation (MAD) between modeled and observed vote
shares across city-level election pairs. MAD indicates political permeability: lower values signal consistency, higher
values signal flux between parties or toward abstention.

| City | Mean MAD (pp) | Max MAD (pp) |
|------|----------------|--------------|
| Ashdod | 1.0 | 2.4 | | Beit Shemesh |
0.6 | 1.5 | | Elad | 2.6 | 8.7 |
| Bnei Brak | 1.0 | 3.7 |
| Jerusalem | 0.6 | 1.0 |
| Modi'in Illit | 2.2 | 10.6 |

Two clusters emerge: highly stable cities (Jerusalem, Beit Shemesh, Ashdod) and relatively fluid ones (Elad, Modi'in
Illit), likely reflecting differences in demographic composition, institutional affiliations, and rabbinic leadership
rather than socioeconomic variation. Figure 8 shows MAD patterns across cities and elections, with the 23→24 disruption
as a clear outlier across all locations.

![MAD Table Lens](plots/mad_table_lens.png)

*Figure 8: MAD rankings by election pair and city, visualizing temporal patterns of deviation*

## Summary of Findings

The analysis yields six key findings about Haredi voter transition patterns:

1. **Dominant pattern of high stability:** Across most election pairs, Shas and UTJ maintained loyalty rates above 95%,
confirming the general portrait of Haredi voting as highly disciplined and predictable.

2. **Sharp temporary disruption in the March 2020–March 2021 transition (23→24):** Shas loyalty dropped to 73.5%
nationally (67.1% in Ashdod), with 12.3% switching to UTJ (19.3% in Ashdod). UTJ loyalty also fell to 87.9%, with
4.6% of UTJ voters switching to Shas.

3. **Full recovery in the March 2021–November 2022 transition (24→25):** Loyalty patterns returned to pre-crisis levels,
indicating the disruption was temporary rather than a permanent realignment.

4. **Geographic variation in disruption magnitude:** The 23→24 shock affected all cities but varied in intensity. Ashdod
showed the largest deviations, followed by Bnei Brak and Beit Shemesh, suggesting that local demographic integration
modulates electoral volatility.

5. **Temporal proximity rules out demographic explanations:** The short intervals between elections (5.5–19 months)
indicate genuine voter switching rather than migration or generational turnover.

6. **Stable abstention rates:** Haredi participation remained resilient throughout, even during the 23→24 disruption,
with minimal abstention flows contrasting with significant inter-party switching.

These findings reveal that even highly cohesive voting blocs can experience rapid, system-wide disruptions in loyalty
patterns that fully reverse when triggering conditions resolve—a pattern of rigidity with stress fractures potentially
observable across identity-based electorates from U.S. evangelicals to European regional parties. The consistency across
cities points to centralized political dynamics, while the variation in magnitude highlights how local social integration
modulates the boundaries of permissible switching within otherwise rigid structures.

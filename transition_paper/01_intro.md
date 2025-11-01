# Introduction

Across many democracies, communities exhibit cohesive voting patterns that resemble durable "voting blocs" anchored in
strong group identities. Classic cleavage theories and the literature on ethnoreligious mobilization argue that such
identities can cement stable partisan alignments over long periods (Lipset & Rokkan, 1967; Curiel & Zeedan, 2024). From U.S. evangelicals to
European regional party loyalists to African ethnic voting coalitions, these blocs appear remarkably stable over time.
Yet a central question for political behavior remains: are these loyal voting blocs truly immobile, or can their rigid
boundaries temporarily yield under stress? This question functions as a stress test of bloc discipline theories,
exploiting a rare quasi-natural experiment—the 2020–21 electoral upheaval in Israel—that allows us to observe how deeply
internalized political loyalties respond to sudden systemic shocks.

Israel's ultra-Orthodox (*Haredi*, plural *Haredim*) Jews provide an extreme-case test of this question. The Haredi sector is widely
portrayed as one of the most disciplined electoral blocs in Israel, with high turnout rates and reliably pro-clerical
voting reinforced by dense institutional networks of yeshivas and synagogues (Freedman 2020; Leon 2014; Malach 2025).
Haredi political culture traditionally frames voting not as an individual right but as a collective religious duty
dictated by spiritual authorities. Yet recent research challenges this monolithic portrayal. Zalcberg (2023) notes that
the community is factionalized and not all members obey rabbinic voting directives; roughly 11% of self-identified Haredi
Jews voted for non-Haredi lists in the 2021 election despite communal pressures. Similar patterns of partial defection
within cohesive electorates have been documented elsewhere: Simas and Lothamer (2025) find that even strong party
identifiers may temporarily defect following intraparty conflict, revealing that perceived unity can mask underlying
volatility. This pattern is particularly striking given Israel's electoral history: Andersen and Yaish (2003) found that
the 1993–1999 electoral reform failed to shift social cleavage voting patterns, suggesting that institutional change
alone rarely disrupts voting blocs. Yet the 2020–21 shock disrupted even the most disciplined Haredi communities, making
this episode distinctive. This raises a theoretical puzzle applicable beyond Israel: can highly disciplined voting blocs
harbor latent realignment pressures that surface under specific temporal and local conditions?

Crucially, the Haredi sector is internally divided along ethnic lines between Ashkenazi (European-origin) and
Sephardi/Mizrahi (Middle Eastern/North African-origin) streams, each represented by separate political parties: United
Torah Judaism (UTJ) and Shas, respectively. Ashkenazi Haredim are represented by UTJ, whose electorate is almost entirely
ultra-Orthodox Ashkenazi. In contrast, Sephardi/Mizrahi Haredim predominantly vote for Shas, though Shas's voter base
extends well beyond the strictly ultra-Orthodox community. As Cincotta (2013) observed, "the vast majority of Shas's
electoral support has come from non-ultra-Orthodox Sephardim and Mizrahim," reflecting the party's broad appeal (see also
Keren-Kratz 2025). Leon (2014) characterizes Sephardi ultra-Orthodoxy as "strict ideology, liquid identity," suggesting
more permeable boundaries than Ashkenazi streams. Malach (2025) finds that UTJ behaves as a "sectarian party with dynamic
fringes," drawing ~95% of its potential core support but subject to modest flows at the edges. This ethnic cleavage
within a religiously unified population provides a unique opportunity to study how internal boundaries within cohesive
blocs respond to electoral shocks. Analyzing transitions between these ethnically-aligned parties reveals whether
ethnic-political boundaries can temporarily weaken during crises without triggering permanent realignment—a pattern with
broad implications for understanding identity-based voting blocs worldwide.

This study was sparked by an anomaly discovered in my previous research on Haredi residential segregation (Gorelik 2025).
That study documented persistent spatial separation between Ashkenazi and Sephardi Haredim across Israel's cities.
However, Ashdod—a southern coastal city—exhibited a sudden, unexplained drop in residential segregation between the March
2020 and March 2021 elections, followed by rapid recovery (Figure 1). With no major demographic relocation or
institutional change occurring during this brief period, the anomaly was puzzling. If people did not physically move,
what changed? One plausible answer: politics. Residential segregation changes slowly, but voting behavior can shift
rapidly. If Sephardi voters temporarily switched from Shas to UTJ (or vice versa), this would create apparent
"integration" in segregation indices—which rely on party votes as proxies for ethnicity—without any residential movement.
This possibility motivated the present study: Was Ashdod's anomaly evidence of a broader, under-the-radar pattern of
voter transitions within this ostensibly rigid bloc? The period's exceptional political volatility—with four national
elections between April 2019 and March 2021—provides a natural experiment for testing how electoral shocks affect even
highly disciplined voting communities.

![Dissimilarity Index Dynamics](plots/dissimilarity_dynamics_kn.png)

*Figure 1: Ashdod exhibits a distinctive sharp drop in dissimilarity index between Knesset 2324 (March 2020March 2021),
followed by rapid recovery, while other major Haredi cities show stable segregation patterns. This anomaly coincides
with the voter transition disruption analyzed in this study. The index measures spatial segregation between Ashkenazi
and Sephardi Haredi populations within each city, with higher values indicating greater residential separation.
Reconstructed from data in Gorelik, 2025*

I address this puzzle by analyzing voter transition matrices across successive elections from January 2013 through
November 2022 (Knessets 19-25), using hierarchical Bayesian ecological inference applied to polling-station results
(detailed in the Methods section; King 1997; Greiner & Quinn 2010). The analysis estimates transition probabilities
among four categories: Shas, UTJ, other parties, and abstention. To focus specifically on Haredi behavior, I restrict the sample to polling stations at least 75%
Haredi in composition, ensuring the transitions reflect dynamics within the ultra-Orthodox sector rather than broader
trends. The framework provides both national-level estimates and city-specific patterns for major Haredi population
centers.


Three key findings emerge. First, Ashdod's 2021 segregation anomaly coincides precisely with a dramatic but temporary
disruption in voter loyalty during the March 2020–March 2021 transition. Shas-to-Shas loyalty plummeted from 99% to 74%
nationally (to 67% in Ashdod), with unprecedented cross-flows to UTJ. Critically, loyalty fully recovered by the next
election (March 2021–November 2022), confirming that electoral switching—not residential movement—explained Ashdod's
segregation drop. Second, this pattern of temporary disruption followed by complete recovery characterizes a rigid
system experiencing stress fractures, not ongoing fluidity. The underlying ethnic-political boundaries remained intact
despite the shock, challenging narratives of gradual erosion in identity-based voting. Third, geographic variation in
disruption magnitude reveals how local contexts modulate boundary permeability: ethnically mixed cities like Ashdod
showed larger temporary swings, while homogeneous strongholds like Bnei Brak remained more stable, yet all recovered
fully.



These findings have implications extending far beyond Israel. The pattern of temporary disruption followed by recovery—
what I term "rigidity with stress fractures"—offers a new framework for understanding cohesive voting blocs globally.
Rather than assuming either permanent stability or gradual erosion, this framework recognizes that strong institutional
boundaries can temporarily yield under extraordinary pressure, then rapidly reconstitute. This insight applies to any
democracy featuring identity-based voting blocs: from U.S. evangelicals to European regional parties to African ethnic
coalitions. Israel's Haredi sector, as an extreme case of institutional discipline, provides a conservative test: if even
this highly bounded community experiences measurable temporary volatility, similar dynamics likely operate—perhaps more
strongly—in less institutionally rigid contexts.

Methodologically, integrating spatial segregation metrics with temporal transition matrices offers a replicable framework
for studying segmented populations. This pairing applies to immigrant communities divided by origin region, religious
denominations within shared faith traditions, or linguistic minorities maintaining distinct political allegiances. By
combining static measures (segregation at a point in time) with dynamic measures (transitions over time), researchers can
quantify both the degree of separation and the rate of exchange between subgroups. The Ashdod anomaly also reveals a
critical limitation: using voting patterns as proxies for residential segregation fails when political loyalty becomes
unstable, with implications for any study conflating political and demographic boundaries.


Specifically, this study contributes by: (1) documenting temporary boundary permeability within an extreme-case
disciplined bloc, providing a conservative estimate for volatility in identity-based voting; (2) introducing "rigidity
with stress fractures" as an alternative to permanent-stability or gradual-erosion models; (3) revealing methodological
confounding between political and residential measures when loyalty fluctuates; and (4) demonstrating hierarchical
Bayesian ecological inference for analyzing voter transitions in cohesive subpopulations. Understanding the conditions
under which rigid boundaries temporarily yield, and how quickly they reconstitute, is crucial for anticipating coalition
dynamics and democratic responsiveness in any polity with identity-based voting. The following section details the data
sources, model specification, and analytical approach.

# Introduction

Across many democracies, communities exhibit cohesive voting patterns that resemble durable "voting blocs" anchored in
strong group identities. Classic cleavage theories and the literature on ethnoreligious mobilization argue that such
identities can cement stable partisan alignments over long periods (Lipset & Rokkan, 1967; Curiel & Zeedan, 2024). From U.S. evangelicals to
European regional party loyalists to African ethnic voting coalitions, these blocs appear remarkably stable over time.
Yet a central question for political behavior remains: are these loyal voting blocs truly immobile, or can their rigid
boundaries temporarily yield under stress? This question functions as a stress test of bloc discipline theories,
exploiting a rare quasi-natural experiment, the 2020–21 electoral upheaval in Israel, that allows us to observe how deeply
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
ethnic-political boundaries can temporarily weaken during crises without triggering permanent realignment, a pattern with
broad implications for understanding identity-based voting blocs worldwide.

This study was sparked by an anomaly discovered in my previous research on Haredi residential segregation (Gorelik 2025).
That study documented persistent spatial separation between Ashkenazi and Sephardi Haredim across Israel's cities.
However, Ashdod, a southern coastal city, exhibited a sudden, unexplained drop in residential segregation between the March
2020 and March 2021 elections, followed by rapid recovery (Figure 1). With no major demographic relocation or
institutional change occurring during this brief period, the anomaly was puzzling. If people did not physically move,
what changed? One plausible answer: politics. Residential segregation changes slowly, but voting behavior can shift
rapidly. If Sephardi voters temporarily switched from Shas to UTJ (or vice versa), this would create apparent
"integration" in segregation indices, which rely on party votes as proxies for ethnicity, without any residential movement.
While this proxy is theoretically grounded and performs well under stable conditions, the Ashdod episode suggests it
is particularly fragile in highly disciplined populations where centralized authority can rapidly redirect voting behavior.
This possibility motivated the present study: Was Ashdod's anomaly evidence of a broader, under-the-radar pattern of
voter transitions within this ostensibly rigid bloc? The period's exceptional political volatility, with four national
elections between April 2019 and March 2021, provides a natural experiment for testing how electoral shocks affect even
highly disciplined voting communities.

![Dissimilarity Index Dynamics](plots/dissimilarity_dynamics_kn.png)

*Figure 1: Ashdod exhibits a distinctive sharp drop in dissimilarity index between Knesset 2324 (March 2020–March 2021),
followed by rapid recovery, while other major Haredi cities show stable segregation patterns. This anomaly coincides
with the voter transition disruption analyzed in this study. The index measures spatial segregation between Ashkenazi
and Sephardi Haredi populations within each city, with higher values indicating greater residential separation.
Reconstructed from data in Gorelik, 2025*

### Theoretical Framework

Three established literatures each capture part of the pattern observed in this study but none accounts for the
full trajectory of stability, sudden disruption, and rapid recovery. Classic cleavage theory (Lipset and Rokkan 1967;
Bartolini and Mair 1990) predicts that identity-based voting blocs remain durable so long as the social networks and
organizational structures that encapsulate them persist. This explains the Haredi sector's high baseline loyalty but
cannot account for the sudden disruption: if encapsulation is intact, why do loyalty rates plummet? Electoral
volatility typologies (Pedersen 1979; Mainwaring and Zoco 2007) distinguish within-bloc from between-bloc volatility
and recognize that elite-driven supply-side changes can produce temporary swings, but they do not specify the
conditions under which such swings will reverse rather than crystallize into permanent realignment. Punctuated
equilibrium models (Baumgartner and Jones 1993) capture the rhythm of long stasis interrupted by sudden change, yet
they originated in policy studies and do not address the institutional mechanisms, specifically centralized spiritual
authority, that can both trigger and terminate electoral disruptions in identity-based blocs.

This paper proposes a synthesizing framework, "rigidity with stress fractures," that integrates these perspectives.
The framework predicts that identity-based voting blocs will exhibit high stability punctuated by sudden, reversible
disruptions when three scope conditions are met: (i) a centralized institutional authority, such as rabbinic
leadership, exercises strong influence over electoral behavior; (ii) that authority operates through organizational
networks, such as yeshiva systems, with sufficient reach to produce geographically synchronized shifts; and (iii) the
triggering crisis is exogenous and temporary rather than structural, so that withdrawal of the disrupting directive
restores the prior equilibrium. When these conditions hold, observed volatility reflects coordinated elite action
within an intact institutional framework rather than genuine erosion of group boundaries. The framework thus
distinguishes temporary stress fractures, reversible disruptions driven by elite coordination, from both the permanent
stability predicted by frozen-cleavage models and the gradual erosion assumed by dealignment theories.

I address this puzzle by analyzing voter transition matrices across successive elections from January 2013 through
November 2022 (Knessets 19 - 25), using hierarchical Bayesian ecological inference applied to polling-station results
(detailed in the Methods section; King 1997; Greiner & Quinn 2010). The analysis estimates transition probabilities
among four categories: Shas, UTJ, other parties, and abstention. To focus specifically on Haredi behavior, I restrict the sample to polling stations at least 75%
Haredi in composition, ensuring the transitions reflect dynamics within the ultra-Orthodox sector rather than broader
trends. The framework provides both national-level estimates and city-specific patterns for major Haredi population
centers.


Three findings emerge: a nationwide collapse in voter loyalty during the March 2020–March 2021 transition, with
substantial city-level variation in disruption magnitude, followed by rapid recovery that suggests temporary stress
fractures rather than permanent realignment. This pattern, and its resolution of Ashdod's segregation anomaly through
electoral switching rather than residential movement, carries implications for any democracy featuring identity-based
voting blocs.

Specifically, this study contributes by: (1) documenting temporary boundary permeability within an extreme-case
disciplined bloc, providing a conservative estimate for volatility in identity-based voting; (2) grounding "rigidity
with stress fractures" as an analytical framework that synthesizes cleavage theory, volatility typologies, and
punctuated equilibrium models, specifying the scope conditions under which identity-based blocs experience reversible
disruption rather than permanent realignment; and (3) applying hierarchical Bayesian ecological inference for
analyzing voter transitions in cohesive subpopulations. Understanding the conditions
under which rigid boundaries temporarily yield, and how quickly they reconstitute, is crucial for anticipating coalition
dynamics and democratic responsiveness in any polity with identity-based voting. The following section details the data
sources, model specification, and analytical approach.

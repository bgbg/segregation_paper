# Issue 9: Scale back comparative claims — frame as extreme-case study

## Strategy

Frame the paper as an **extreme-case study** (Gerring 2007) that generates testable hypotheses.
The Haredi bloc is uniquely suited to reveal a mechanism — elite-coordinated reversible volatility —
that existing theories predict piecemeal but none fully accounts for. The case is the *site*, not
the *scope*. The scope conditions (centralized authority, network reach, exogenous temporary crisis)
are abstract enough to travel.

Use **one** brief comparative illustration — Lanzara et al. (2024) on Italian bishops and Christian
Democracy vote share — where the analogy is strongest, with explicit acknowledgment of where it
breaks down. Move all other scattered references to a future-work gesture.

## Assumptions

- The worktree is on `main`, which lacks `transition_paper/`. Edits will be made on the current
  working branch (`develop/6-demote-segregation-to-appendix`) and cherry-picked or merged later.
- Existing references (Bullock et al. 2019, Smith 2019, Aha et al. 2024, McCauley 2014) stay in
  the references list but are removed from the main text (or moved to future-work sentence).
- Italy comparative: cite Lanzara et al. 2024 (*Journal of Public Economics*).

## Open Questions — RESOLVED

- Italy source: **Lanzara, Gianandrea, Sara Lazzaroni, Paolo Masella, and Mara P. Squicciarini.
  2024. "Do Bishops Matter for Politics? Evidence from Italy." *Journal of Public Economics*
  238: 105177.** Shows bishop identity explains significant variation in Christian Democracy vote
  share across Italian dioceses 1948-1992; bishop replacement produces ~3pp swing. Closest
  published parallel: centralized religious authority → organizational networks → electoral effects.
  Analogy breaks down on temporal dimension (cross-sectional, not sudden synchronized disruption
  and recovery).
- Gerring 2007: Use the article (user has access to article, not book).

## Steps

### 1. Abstract (`00_abstract.md`)
- **Line 3:** Replace "evangelical churches, immigrant enclaves, ethnic minorities, religious
  denominations" with a single conditional phrase: "wherever centralized authority structures
  shape bloc voting"
- Keep the rest of the abstract unchanged — it already uses appropriately hedged language

### 2. Introduction (`01_intro.md`)
- **Lines 1-10 (opening paragraph):** Replace the laundry-list comparative gesture ("From U.S.
  evangelicals to European regional party loyalists to African ethnic voting coalitions") with
  a general statement about identity-based voting blocs, without naming specific cases
- **After line 12 ("extreme-case test"):** Add one sentence citing Gerring 2007 to justify
  extreme-case methodology: the Haredi bloc's combination of centralized authority, dense
  networks, and ballot-box data availability makes it uniquely suited to reveal mechanisms
  that are theorized but rarely directly observable
- **Lines 104-112 (contributions paragraph):** No change needed — already uses appropriate
  framing ("extreme-case disciplined bloc")

### 3. Conclusions (`04_conclusions.md`)

#### 3a. Theoretical Implications — Mapping Findings onto the Framework (lines 113-142)
- **Lines 138-142:** Remove the sentence listing Aha et al., McCauley, Bullock, Smith as
  "comparable reversible volatility episodes." Replace with the Lanzara et al. (2024)
  illustration: 3-4 sentences covering (a) bishop identity explains significant variation in
  Christian Democracy vote share across Italian dioceses 1948-1992, (b) bishop replacement
  produces ~3pp swing — centralized religious authority operating through diocesan networks,
  (c) where the analogy breaks down: Italian evidence is cross-sectional variation in bishop
  influence, not sudden synchronized temporal disruption and recovery — underscoring why the
  Haredi case, with repeated elections over a compressed period, provides a uniquely informative
  window into the dynamics of elite-coordinated volatility.

#### 3b. Synchronized Disruption as Evidence of Discipline (lines 144-162)
- **Lines 154-156:** Remove "from U.S. evangelical communities (Bullock et al. 2019) to Catholic
  voters in Latin America (Smith 2019)." Replace with a general statement: "This parallels
  theoretical expectations for other identity-based voting blocs with centralized institutional
  authority, though comparable micro-level evidence of synchronized reversible switching has
  not been documented elsewhere."

#### 3c. Future work (lines 192-200)
- **After the existing future-work sentence about voter surveys:** Add: "Comparative studies
  could test whether the scope conditions identified here — centralized authority, organizational
  network reach, and exogenous temporary crisis — produce similar patterns of elite-coordinated
  reversible volatility in other settings, such as evangelical communities, ethnic party systems
  in sub-Saharan Africa, or Catholic electorates in southern Europe."

### 4. References (`10_references.md`)
- Add Gerring 2007
- Add Lanzara et al. 2024 (Journal of Public Economics 238: 105177)
- Keep existing references (Bullock, Smith, Aha, McCauley) even if removed from main text —
  they may still be cited in appendices or other sections. Verify before removing.

### 5. Verify no orphaned citations
- Grep all .md files for each reference that was removed from the text to confirm it's either
  still cited somewhere or can be safely removed from the references list

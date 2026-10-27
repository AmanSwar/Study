# Product Management for Engineers - outline

As of 2026-10-26. Kind: deep dive, flat, 6 pages (the cap). Sources: `sources.json` (31 sources, S1-S31). Coverage: 7 subtopics well covered, 6 thin, 1 none. Total estimate about 56,500 words (each page inside the 6,000-12,000 band).

Aman's request asks for "a course". This run is a deep dive capped at six pages, so the pages carry the engineer-relevant core that the sources can actually support. The field is larger than six pages: see "Does the field need more than six pages" below.

## Reading order and rationale

Pages are readable in any order (no prerequisites are required; a page that leans on the four risks re-states them in one paragraph). The recommended order runs from the uncertainty a product team faces to the business that has to survive it.

1. **`01-discovery-delivery-four-risks`** - first because every later page assumes the vocabulary: product work is reducing four risks cheaply before paying for delivery. Feasibility, the engineer's native risk, is one of four.
2. **`02-problem-framing-user-jobs`** - what discovery should be aimed at. A request becomes a problem with a boundary; personas vs jobs; the opportunity tree. Holds the thin jobs-to-be-done material as a bounded section.
3. **`03-prioritization-roadmaps`** - choosing and sequencing: RICE, appetite, cost of delay, backlog attrition, cycles, outcome roadmaps and date commitments. The ranking methods disagree most sharply here.
4. **`04-metrics-experiments-goals`** - how to know a bet worked and how to set goals: OEC, guardrails, HEART, OKRs, trust in data, and ML features that change over time. After page 3 because it grades the decisions page 3 makes.
5. **`05-roles-decision-rights-specs`** - who decides, and which document carries the decision. Placed after the mechanisms so the role models can be judged by what each does with risk, ranking and goals. An engineer who wants the role map first can start here.
6. **`06-viability-strategy-pricing`** - the fourth risk in depth, for a developer or infrastructure product: stakeholders, strategy, pricing, positioning, and an explicit map of where evidence ends. Last because it is the most applied and the thinnest.

## Page list

| # | id | shortTitle | est. words | sources | disagreements (log order) |
|---|----|-----------|-----------|---------|----------------|
| 1 | 01-discovery-delivery-four-risks | Discovery & Risk | 9,000 | S1,S2,S3,S5,S6,S9,S16,S20 | D2 discovery method |
| 2 | 02-problem-framing-user-jobs | Problem Framing & Jobs | 7,500 | S1,S7,S8,S9,S10,S23,S29 | D8 personas vs jobs |
| 3 | 03-prioritization-roadmaps | Priorities & Roadmaps | 11,000 | S1,S2,S4,S7,S11,S17,S18,S28 | D4 roadmaps/backlogs, D5 prioritization |
| 4 | 04-metrics-experiments-goals | Metrics & Experiments | 10,500 | S1,S2,S6,S12,S13,S14,S15,S17,S18 | D6 evidence vs judgment, D7 goal alignment, D10 speed vs trust |
| 5 | 05-roles-decision-rights-specs | Roles & Specs | 11,500 | S1,S3,S6,S8,S9,S16,S17,S18,S19,S20,S21,S22,S23,S24 | D1 who decides, D3 spec artifact |
| 6 | 06-viability-strategy-pricing | Viability & Pricing | 7,000 | S2,S3,S4,S7,S15,S20,S22,S23,S25,S26,S27,S30,S31 | D9 pricing, D11 strategy |

All 11 logged disagreements are assigned, each to exactly one page. Cross-page overlaps are resolved by assigning a mechanism to one page and pointing to it from the other:

- Idea-success base rates: the attrition chart is on page 1; page 4 uses the same figures only for the portfolio argument. The four bases (1/3, 10-20%, ~30%, "90%") are never merged.
- PR/FAQ: mechanics on page 5; page 1 names it only as one position in the discovery disagreement.
- OKRs: S4's outcome-roadmap idea is on page 3; OKR grading and counts are on page 4.
- High-integrity commitments: page 3; page 6 refers back.
- Strategy as the choice of OEC (S2): page 4 explains the OEC; page 6 treats it as one of four definitions of strategy.

## Where thin and none areas land (decision)

| Subtopic | Coverage | Decision | What is marked unverified or stated as a gap |
|---|---|---|---|
| users-jtbd | thin | A bounded section on page 2 (milkshake case from S10; S29 as claim only) | Rival JTBD school (search snippet only); any application to a developer-facing runtime |
| strategy | thin | A section on page 6 (D11), with S2's strategy-to-OEC link on page 4 | Rumelt's kernel; complements mapped onto an inference runtime |
| stakeholders | thin | A section on page 6 (S7, S15, S22), decision rights on page 5 | Conflict resolution and executive demands (S7 gives no method) |
| pricing-devtools | thin | The core of page 6 (S20 official policy, S26 vendor guidance, S25) | Anything on on-device pricing; no price, margin or licensing figure exists in the source map |
| positioning | thin | A half-section on page 6 (S31 book page) | Dunford's five component names; publication year (edition unresolved) |
| ai-ml-product | thin | Three sections: page 1 (AI in discovery, S6), page 4 (S14 guidelines, speed vs trust), page 5 (role shifts, S24) | Offline-vs-online evaluation of ML models (the Booking.com paper was not fetched); Ng's ratios (single unsourced source) |
| infra-platform-product | none | No page. A short gap section on page 6 and a stated omission on page 5 (S23 itself leaves it out) | Which mechanisms transfer to internal platform teams; S27 is a first-edition table-of-contents pointer only |

The closest analog the sources offer for an inference-runtime company is a developer-product company (PostHog, Linear, Basecamp), not an internal platform team. The pages say so and do not pretend the analogy is evidence.

## What is deliberately excluded and why

- Exercises, quizzes, deliverables, schedules, assessments: standing rule.
- The course-candidate areas listed under "more than six pages" below: each needs sources this map does not have, or would turn a page into a survey.
- Book-level claims whose books were not fetched (S27, S29, S30, S31): used only for what the catalog page says the book claims, never for mechanism.
- Any number about prices, margins, COGS or licensing of an on-device runtime: no source; the page derives a framework symbolically and says where it ends.
- Data comparing PM-led and engineer-led teams: none exists in S18-S20, all self-reported; page 5 says so.

## Claim-grounding rules (for the writers)

No page's key number or factual claim may rest on a single blog entry. Where the sources cannot support that, the claim is attributed as one organization's estimate or marked unverified, and is not used as a finding.

- Page 1: 10-20% base rate has two bases (S2 and S5) shown separately; "order of magnitude cheaper" and "3-5 iterations" are SVPG's own estimates, attributed (S5 and S6 are one organization).
- Page 2: milkshake facts are one secondhand HBS case (S10); Basecamp's six-week-to-one-day example is S1's own anecdote; both attributed.
- Page 3: cost-of-delay figures use Reinertsen's primary wording only (15% of product developers, p. 4; 50-to-1, footnote p. 5); the "85% of PMs" rewording is not used; WSJF only as SAFe states it; RICE scores are derived arithmetic from S11's inputs and labeled so.
- Page 4: four success-rate bases kept apart; OEC "one to two weeks" to be confirmed in S2 before it is stated; no "Type 1 / Type 2" labels.
- Page 5: Engineer:PM ratios are Ng's unsourced assertion (S24 only) and marked unverified; PostHog's "split at six" and "8+ products" are S19 self-report.
- Page 6: PostHog's pricing rules are one company's policy (S20), not a consensus; seven planned unverified marks are listed in the plan's scope.

## Does the field need more than six pages? Yes

For Aman's stated aim of a course, six pages are not enough. The researcher judged these areas too large for a deep dive: stakeholder influence; user-research craft; positioning and go-to-market for developer and infrastructure products; pricing research methods and inference unit economics; product analytics and instrumentation; experimentation statistics beyond Kohavi Chapter 1; business cases and opportunity sizing; strategy frameworks; platform and infrastructure product management; AI product management in depth; organizational transformation; PM careers.

A course (3-8 parts, 10-40 modules) is the right container for those. It would not be only a larger plan: six subtopics are thin and one has no sources, so a course needs a second research pass first (the books behind S27-S31 plus Torres, Fitzpatrick, Ramanujam and Tacke, and Kohavi Parts II-V were not fetched). Recommended: `/study-course` for those areas after this deep dive is read, with a fresh research pass.

## Source-map summary

31 sources: 16 blog, 7 textbook (4 not fetched: S27, S29, S30, S31), 5 official, 2 paper, 1 analysis. 27 fetched.

Concentration risk: SVPG/Cagan accounts for six entries (S3-S8, spanning 2005-2026) plus the S27 pointer, and the same author's positions change across them (PRD in 2005, prototype as spec in 2026). Pages present these as one author's evolving position, not as independent confirmation.

Top 8 by authority:
1. **S2 Kohavi, Tang and Xu, *Trustworthy Online Controlled Experiments*** (2020): the standard text, by the builders of the Microsoft, Google and LinkedIn platforms. Only the front matter, table of contents, Preface and Chapter 1 fetched.
2. **S1 Shape Up** (37signals, web edition): the builder's own published method; practitioner assertion, no measured data.
3. **S20 PostHog Handbook** (2026): a developer-product company's published policy on engineer-led product teams and pricing; living document.
4. **S13 Rodden, Hutchinson and Fu, HEART** (CHI 2010): Google UX research applied to more than 20 products; no outcome data.
5. **S14 Amershi et al., Guidelines for Human-AI Interaction** (CHI 2019): Microsoft Research, 18 guidelines validated with 49 practitioners across 20 AI products.
6. **S28 Reinertsen, *Principles of Product Development Flow*** (2009): origin of cost of delay; Chapter 1 and table of contents only, figures are his own course surveys.
7. **S12 Google re:Work OKR guide** (2016 record): official; one case (Sears), no outcome data.
8. **S15 Bezos, 2016 Letter to Shareholders**: primary statement of Amazon's decision principles (70% information, two-way doors, disagree and commit).

Currency: S6 (2026-04), S7 (2025-12), S26 (2026-04) and the PostHog handbook are within the last year and likely to change; S21 (written about 15 years before 2012), S25 (2002), S8 (2005) are dated and presented as such.

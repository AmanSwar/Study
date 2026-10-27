# Sources — Product management and product thinking for software engineers
As of 2026-10-26 · 31 sources · deep-dive
Updated at assembly 2026-10-27: 29 further sources cited by the module pages (section "Added at assembly"); 3 corrections (section "Corrections at assembly").

Count by type: official 5, textbook 7, blog 16, analysis 1, paper 2. 27 entries have fetched text; 4 are `fetched: false` (S27, S29, S30, S31: books where only a catalog, ToC or author page was fetched — see each entry's "Fetched parts"). Other books are pointers in Notes.

## Subtopics
1. `discovery-delivery` — Discovery vs delivery: the four product risks, build-to-learn vs build-to-earn, shaping vs building
2. `problem-framing` — Problem framing: from 'what could we build' to 'what is going wrong', goals, appetite, scope
3. `users-jtbd` — Users and jobs-to-be-done: personas vs jobs, opportunities, customer interviews
4. `prioritization` — Prioritization and estimation: scoring (RICE), appetite and betting, backlog attrition
5. `metrics-experimentation` — Metrics and experimentation: OEC, guardrails, HEART, A/B tests, base rates of idea success
6. `roadmaps-okrs` — Roadmaps, backlogs, cycles and OKRs: outcome roadmaps, date commitments, goal-setting
7. `strategy` — Strategy: vision, product direction, strategy as a choice of OEC, commoditizing complements
8. `specs-artifacts` — Specs and artifacts: PRD, prototype-as-spec, pitch, PR/FAQ, short project spec
9. `roles-engineer-pm` — Roles: PM vs engineer vs designer; the product-minded or product engineer; AI-era role shifts
10. `stakeholders` — Working with stakeholders: constraints, commitments, disagree-and-commit
11. `pricing-devtools` — Pricing a developer or infrastructure product: usage vs seat vs hybrid, free tiers, competitor anchoring
12. `positioning` — Positioning a developer or infrastructure product (competitive alternatives, category, differentiated value)
13. `ai-ml-product` — Product thinking for ML/AI features: probabilistic UX, offline vs online evaluation, AI-era PM work
14. `infra-platform-product` — Product thinking for infrastructure/platform teams whose customers are developers (non-user-facing)

## Coverage
- **well**: discovery-delivery, problem-framing, prioritization, metrics-experimentation, roadmaps-okrs, specs-artifacts, roles-engineer-pm
- **thin**: users-jtbd, strategy, stakeholders, pricing-devtools, ai-ml-product, positioning
- **none**: infra-platform-product

Sources per subtopic:
- `discovery-delivery`: S1, S2, S3, S5, S6, S7, S9, S16, S17, S19, S20, S27
- `problem-framing`: S1, S3, S8, S9, S10, S13, S16, S23
- `users-jtbd`: S8, S9, S10, S29
- `prioritization`: S1, S2, S8, S11, S17, S18, S20, S28
- `metrics-experimentation`: S2, S13, S14, S15
- `roadmaps-okrs`: S1, S4, S5, S7, S12, S17, S18, S19
- `strategy`: S2, S4, S12, S15, S25, S30
- `specs-artifacts`: S1, S6, S8, S16, S17, S18, S24, S27, S28
- `roles-engineer-pm`: S1, S3, S5, S6, S9, S15, S17, S18, S19, S20, S21, S22, S23, S24, S27
- `stakeholders`: S7, S15, S22, S23
- `pricing-devtools`: S20, S25, S26
- `positioning`: S31
- `ai-ml-product`: S6, S14, S24, S26
- `infra-platform-product`: (no fetched source)

Why thin or none:
- `users-jtbd` — thin: S10 (HBS analysis, with the milkshake facts) and S29 (Competing Against Luck, publisher blurb and catalog entries only; no text fetched). The HBR article is paywalled. Two entries now exist, but only S10 carries mechanism-level facts; schedule at most a section and cite S10 for the case.
- `strategy` — thin: S30 (Rumelt) is a table of contents and publisher blurb only, so the kernel is still unverified; S2 (pp. 20-23), S4, S12, S25 give partial, adjacent material. Still no source that states a strategy method with text. Schedule only as a section tied to metrics and roadmaps.
- `stakeholders` — thin: S7 is the only source whose subject is stakeholders (product-team side, no method for conflict); S15 gives disagree-and-commit; S22/S23 touch influence. Candid stakeholder mechanics (negotiation, saying no to executives) are unsourced.
- `pricing-devtools` — thin: S20 is a real policy (official) with no numbers; S26 is qualitative vendor guidance with no prices or margins; S25 is a 2002 essay on complements. No source on willingness-to-pay methods (Ramanujam and Tacke not fetched) and none on on-device or per-device licensing.
- `ai-ml-product` — thin: S14 is a rigorous paper but about UI design for probabilistic behavior, not business or product strategy; S24 gives role-shift claims with no methodology; the Booking.com paper (offline model metrics vs business impact) could not be fetched.
- `positioning` — thin (was none): S31 is the author's book page, with verbatim statements that positioning breaks into five component pieces and three positioning styles, but the component names and the method are not on any fetched page. No source applies positioning to developer or infrastructure products. Schedule as a half-section at most, and mark the five components unverified.
- `infra-platform-product` — none: S23 explicitly leaves platform and infrastructure teams uncovered. S27 (1st-edition table of contents only) shows a chapter titled "Keys to Platform Products"; that is a pointer, not a source, and it may not survive into the 2nd edition. The largest gap and the one closest to Aman's own work.
- `prioritization` — well (strengthened): S28 (Reinertsen, ToC and Ch. 1 text) now gives a primary cost-of-delay source alongside S1, S11, S17. Chapters 2-9 and the WSJF derivation are not fetched.

## Disagreements found
1. **Who decides what gets built: the PM as CEO of the product, the engineer, or a cross-functional trio?**
   - The PM is 'the CEO of the product': defines the what, manages the product team, takes full responsibility for outcomes; the PM owns value and viability risk. [S21, S3]
   - The PM is not the CEO: no authority over design, engineering, marketing, sales; authority is earned and product management is a team sport. [S22]
   - Engineers run the product and manage the team; PMs are a compass (information and coaching) and do not own the roadmap. At Linear engineers lead projects (rotating) with one PM in the company. [S19, S18, S20]
   - A trio/cross-functional team with split risk ownership (PM: value+viability, designer: usability, lead engineer: feasibility); in 2026 the PM is 'a builder, not a manager' who builds and tests prototypes. [S3, S9, S6]
   - AI makes building cheaper, so demand rises for people who write clear specs of what to build; typical Engineer:PM ratio is about 6:1 today (4:1 to 10:1 range) and strong PMs are expected to prototype themselves. [S24, S6]
   - Shaping needs a generalist or one to two collaborators; the Shape Up pages name no PM (betting table: CEO, CTO, senior programmer, product strategist). [S1]
2. **How should discovery be done: weekly interviews, prototypes, experiments, or shape-and-bet?**
   - Continuous discovery: interview customers every week, map opportunities in an opportunity solution tree, compare more than one solution per opportunity, test assumptions. PostHog's PM rule of thumb is 1 interview per week per PM. [S9, S20]
   - Discovery is mostly prototypes by the product team, at least an order of magnitude cheaper than building, iterated 3-5 times before delivery. [S5, S6]
   - Randomized controlled experiments are the arbiter, but most ideas fail (1/3, 10-20%, ~30%, '90% wrong' depending on domain), so test everything and use complementary techniques (logs, surveys, UER) where experiments cannot run; metrics are not a substitute for formative user research. [S2, S13]
   - Shape then bet: no customer-research cadence prescribed; ideas that matter 'come back'; shaping is rough, solved and bounded work done before betting. [S1]
   - Write the customer announcement and FAQ first; vet the idea in writing before assigning engineers. [S16]
3. **What is the right spec artifact: PRD, prototype, pitch, PR/FAQ, or a short project spec?**
   - A ten-step PRD is the highest-leverage document and drives the whole company (2005). [S8]
   - The prototype is the spec; the PRD only carries use cases and non-functional requirements a prototype cannot show (2026). [S6]
   - A pitch (problem, appetite, solution sketch, rabbit holes, no-gos) at deliberately low fidelity: do not over-specify with wireframes or high-fidelity mocks. [S1]
   - A PR/FAQ written backwards from the customer announcement; first draft in a few hours. [S16]
   - A concise 1-2 page project spec for situational awareness, plus engineers deciding readiness without a checklist document. [S17, S18]
   - As building gets cheaper, clear written specs gain value and PMs are expected to prototype as well. [S24]
4. **Roadmaps and backlogs: reshape, abolish or keep?**
   - Replace feature roadmaps with vision, prioritized business objectives and OKRs; give dates only as high-integrity commitments that cost discovery work and are used sparingly. [S4, S5, S7]
   - No backlog and no central list; bet six weeks at a time with a two-week cool-down; circuit breaker kills unfinished work by default; two-week cycles are too short. [S1]
   - Keep a manageable backlog and let low-priority items die; two-week cycles are most common; keep a roadmap of themes for about six months and project end dates. [S17, S18]
5. **Prioritization: estimate and rank, set an appetite and bet, or let attrition filter?**
   - Score by Reach x Impact x Confidence / Effort (person-months), with the author's own caveat that it is not a hard rule. [S11]
   - Appetites start with a number and end with a design; estimates start with a design and end with a number; over-specified work is harder to estimate. [S1]
   - Do not keep every request: important ones resurface, low-priority ones never get fixed. [S1, S17]
   - Let measured idea success rates (1/3 or lower) decide: the Bing change that sat six months in the backlog was worth over $100M/year, so a prior ranking can be badly wrong. [S2]
   - Prioritize by economics: quantify cost of delay in money and sequence by it; FIFO ordering is almost never optimal in product development; where cost of delay is not calculated, answers for the same project vary by 50 to 1 (course-survey statistics: only 15% of developers know their cost of delay). [S28]
6. **Evidence vs judgment: how much data before acting?**
   - Be data-driven: formalize an OEC, run controlled experiments, let data inform the HiPPO; the organization is poor at assessing the value of ideas. [S2]
   - Decide at about 70% of the information you wish you had; disagree and commit; beware proxies such as process, surveys and beta-test averages standing in for customers. [S15]
   - Apple under Steve Jobs is cited as the notable organization that did not test ads at all. [S2]
   - Metrics must be tied to goals first and triangulated with qualitative research; they evaluate launched products and do not replace formative research. [S13]
7. **Aligning teams to goals: OKRs, a single OEC with guardrails, initiatives, or bets?**
   - OKRs: three to five objectives, about three key results each, graded 0.0-1.0 with a 0.6-0.7 target, public, not an evaluation tool; Cagan's 'current favorite'. [S12, S4]
   - A single composite OEC, measurable in one to two weeks and predictive of long-term goals, plus guardrails for what must not degrade; 'percent of plan delivered' is a poor key metric. [S2]
   - Product initiatives plus a product timeline (Linear); no OKR mention on the fetched pages (silence, not opposition). [S17]
   - Appetite-bounded bets with no goal-tracking apparatus on the fetched chapters. [S1]
8. **Personas or jobs-to-be-done for understanding users?**
   - Build demographic personas (e.g., 'Leon the Power Seller is a 46 year old male ...') as design archetypes. [S8]
   - Segments by product category or demographics correlate with but do not cause purchase; interview for the job the customer hires the product to do. [S10]
   - Map opportunities as customer needs, pain points and desires under a product outcome, interviewing weekly. [S9]
   - Understanding customer jobs, not understanding customers, drives innovation success (the publisher's statement of the book's thesis; mechanism not fetched). [S29]
9. **Pricing a developer or infrastructure product**
   - Usage-based as the default, no per-extra-user fee, match the cheapest big competitor that makes real revenue at scale, generous but bounded free tier, free tier is a one-way door, yearly pricing review. [S20]
   - Flat per-seat is simple but does not reflect compute cost; usage-based costs predictability and may slow adoption; hybrid (seat + usage) keeps both; flat seat plus heavy agentic usage 'will eventually create a margin problem'. [S26]
   - Push down the price of your product's complements. Pricing power comes from controlling the stack around you. [S25]
10. **Speed of change vs user trust (relevant to ML features that adapt over time)**
   - Run a very large volume of experiments and iterate quickly; most big redesigns fail but cheap incremental tests pay off. [S2]
   - Change aversion is real: satisfaction dropped after a redesign and recovered. [S13]
   - 'Update and adapt cautiously'; limit disruptive changes when the AI system adapts. [S14]
   - 'If you release something and iterate too much, it can be difficult for your customers'; Cagan warns of 'constant, erratic change'. [S18, S6]
11. **What a strategy is and where product strategy comes from**
   - Strategy should not be equated with ambition, leadership, vision or planning (publisher's statement of the book's thesis; the kernel itself was not fetched). [S30]
   - Product vision plus prioritized business objectives per team, expressed as OKRs, replace the feature roadmap as the statement of direction. [S4, S12]
   - Strategy is encoded in the OEC: the strategy drives the choice of OEC, guardrails state what the organization will not trade away, and experiments hill-climb within it. [S2]
   - Strategy is positional economics: commoditize the complements of your product so demand for it rises. [S25]

## Sources

### S1 · official · 37signals / Basecamp — Shape Up: Stop Running in Circles and Ship Work that Matters (web edition; Ch. 2, 3, 6, 7, 8 and glossary fetched) (web edition (page states only 'Copyright ©1999-2026 37signals LLC'; no edition year given) / web edition as accessed 2026-10-26)
Authors: Ryan Singer
https://basecamp.com/shapeup · accessed 2026-10-26 · fetched: true
Authority: The builder organization's own published product-development method, free online; the primary source for appetite, shaping, pitches, the betting table and the no-backlog stance. Claims rest on Basecamp's experience, not measured data.
Covers: discovery-delivery, problem-framing, prioritization, roadmaps-okrs, specs-artifacts, roles-engineer-pm

Facts:
- Cycle is six weeks; cool-down is two weeks (Ch. 7 'Bets, Not Backlogs' page, glossary (/shapeup/2.1-chapter-07)) — “A six week period of time where teams work uninterruptedly on shaped projects.”
- Cool-down definition includes the betting table (Ch. 7 glossary) — “A two-week break between cycles to do ad-hoc tasks, fix bugs, and hold a betting table.”
- Why six weeks (and why not two) (Ch. 8 'The Betting Table' (/shapeup/2.2-chapter-08), 'Cycle length') — “We learned that two weeks is too short to get anything meaningful done.”
- Backlogs rejected explicitly (Ch. 7, 'No backlogs') — “Backlogs are a big weight we don’t need to carry.”
- Cost of backlog grooming (Ch. 7, 'No backlogs') — “The time spent constantly reviewing, grooming and organizing old ideas prevents everyone from moving forward”
- No central list feeds the betting process; lists are decentralized by function (Ch. 7, 'Decentralized lists') — “There’s no one backlog or central list and none of these lists are direct inputs to the betting process.”
- Important ideas resurface on their own (the filter that replaces a backlog) (Ch. 7, 'Important ideas come back') — “Really important ideas will come back to you.”
- Appetite is a time budget set before the design; opposite of an estimate (Ch. 3 'Set Boundaries' (/shapeup/1.2-chapter-03), 'Fixed time, variable scope') — “Estimates start with a design and end with a number. Appetites start with a number and end with a design.”
- Appetite defined; two standard sizes (Ch. 3, 'Setting the appetite') — “We call this the appetite. You can think of the appetite as a time budget for a standard team size.”
- Small batch size (Ch. 3, 'Setting the appetite') — “This is a project that a team of one designer and one or two programmers can build in one or two weeks.”
- Grab-bag projects (redesigns, refactors) are a red flag; '2.0' label is the tell (Ch. 3, 'Watch out for grab-bags') — “A tell-tale sign of a grab-bag is the ‘2.0’ label.”
- Problem framing: flip from solution-space to problem-space (Ch. 3, 'Narrow down the problem') — “we flip from asking ‘What could we build?’ to ‘What’s really going wrong?’”
- Narrowing example: a one-day fix replaces a six-week project (Ch. 3, permissions example) — “That’s a one-day change instead of a six-week project.”
- 'Good' is relative to constraints (no absolute best solution) (Ch. 3, '‘Good’ is relative') — “The best is relative to your constraints.”
- Shaping is design plus strategy; technical literacy required, programming not (Ch. 2 'Principles of Shaping' (/shapeup/1.1-chapter-02)) — “You don't need to be a programmer to shape, but you need to be technically literate.”
- Shaping staffing: a generalist or 1-2 collaborators (Ch. 2) — “you'll need to either embody these skills as a generalist or collaborate with one or two other people.”
- Over-specifying hurts estimation (Ch. 2) — “Counterintuitive as it may seem, the more specific the work is, the harder it can be to estimate.”
- Shaped work has three properties: rough, solved, bounded (Ch. 2) — “Work in the shaping stage is rough.”
- Calendar case study: appetite of six weeks vs a six-month build (Ch. 2 case study) — “It can easily take six months or more to build a proper calendar.”
- Pitch has five ingredients: problem, appetite, solution, rabbit holes, no-gos (Ch. 6 'Write the Pitch' (/shapeup/1.5-chapter-06)) — “There are five ingredients that we always want to include in a pitch”
- Best problem definition is one story (Ch. 6, Ingredient 1) — “The best problem definition consists of a single specific story that shows why the status quo doesn’t work.”
- Pitch deliberately not a spec (Ch. 6) — “We need more concreteness, but we don’t want to over-specify the design with wireframes or high-fidelity mocks.”
- Betting table membership at Basecamp (Ch. 8, 'Who attends') — “the CEO (who in our case is the last word on product), CTO, a senior programmer, and a product strategist”
- Betting call is short (Ch. 8) — “The call rarely goes longer than an hour or two.”
- Team composition for a bet (Ch. 8, 'What is bet on') — “Our project teams consist of either one designer and two programmers or one designer and one programmer.”
- Circuit breaker: no default extension when the bet is not shipped (Ch. 8, 'Circuit breaker') — “If they don’t finish, by default the project doesn’t get an extension.”
- Uninterrupted time is the commitment that makes the bet credible (Ch. 8, 'Uninterrupted time') — “We do not allow the team to be interrupted or pulled away to do other things.”

Conflicts:
- S17 (Linear Method) lists 'Keep a manageable backlog' as a practice and says '2-week cycles are the most common'; S1 rejects backlogs outright and says 'two weeks is too short'.
- S11 (RICE) scores 'Effort' in person-months, an estimate-first approach; S1 says appetites start from a number and end with a design (ch. 3).
- S1 contains no measured evidence for its claims (anecdote and one case study); treat every claim as practitioner assertion.
- S1 does not state how bets are ranked, who writes a pitch, or what 'success' of a bet is (ch. 6, 7, 8 all silent).

### S2 · textbook · Cambridge University Press (authors at Microsoft, Google, LinkedIn) — Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing (front matter, full table of contents, Preface and Chapter 1 'Introduction and Motivation', book pp. i-xviii and 3-24, fetched from the authors' companion-site PDF) (2020 / First published 2020; DOI 10.1017/9781108653985; ISBN 9781108724265 (paperback))
Authors: Ron Kohavi, Diane Tang, Ya Xu
https://experimentguide.com/wp-content/uploads/TrustworthyOnlineControlledExperiments_PracticalGuideToABTesting_Chapter1.pdf · accessed 2026-10-26 · fetched: true
Authority: The standard practitioner text on online controlled experimentation, written by the people who built the experimentation platforms at Microsoft, Google and LinkedIn. Only the sample (front matter + Ch. 1) was fetched; chapter numbers beyond 1 are from its table of contents.
Covers: metrics-experimentation, prioritization, strategy, discovery-delivery

Facts:
- Bing ad-headline change: +12% revenue, over $100M/year US, after sitting six months in the backlog (Ch. 1, p. 3) — “Bing’s revenue increased by a whopping 12%, which at the time translated to over $100M annually in the US alone, without significantly hurting key user-experience metrics.”
- The idea was prioritised low and languished before an engineer tried it (Ch. 1, p. 3) — “The feature was prioritized low and languished in the backlog for more than six months until a software developer decided to try the change, given how easy it was to code.”
- Big wins are rare; volume of experiments (Ch. 1, p. 5) — “Bing runs over 10,000 experiments a year, but simple features resulting in such a big improvement happen only once every few years.”
- Scale claim from the cover text (front-cover blurb, p. i) — “companies that each runs more than 20,000 controlled experiments a year”
- Definition of OEC (overall evaluation criterion) (Ch. 1, p. 6) — “The OEC must be measurable in the short term (the duration of an experiment) yet believed to causally drive long-term strategic objectives”
- Three organizational tenets for experimentation (Ch. 1, p. 11) — “1. The organization wants to make data-driven decisions and has formalized an OEC.”
- Profit is a poor OEC; customer lifetime value is a strategically powerful one (Ch. 1, Tenet 1, p. 12) — ““Profit” is not a good OEC, as short-term theatrics (e.g., raising prices) can increase short-term profit, but may hurt it in the long run. Customer lifetime value is a strategically powerful OEC”
- 'Percent of plan delivered' as an anti-metric (Ch. 1, Tenet 1, p. 12) — “It is often easier to generate a plan, execute against it, and declare success, with the key metric being: “percent of plan delivered,” ignoring whether the feature has any positive impact to key metrics.”
- Data-driven means informing the HiPPO, not replacing judgment (Ch. 1, p. 12) — “A data-driven or a data-informed organization gathers relevant data to drive a decision and inform the HiPPO (Highest Paid Person’s Opinion) rather than relying on intuition”
- Apple under Jobs as the notable counterexample to being data-driven (Ch. 1, Tenet 1, pp. 11-12) — ““we didn’t test a single ad. Not for print, TV, billboards, the web, retail, or anything””
- Idea success rates: one third at Microsoft; 10-20% in optimized domains (Ch. 1, Tenet 3, p. 13) — “Only one third of the ideas tested at Microsoft improved the metric(s) they were designed to improve (Kohavi, Crook and Longbotham 2009). Success is even harder to find in well-optimized domains like Bing and Google, whereby some measures’ success rate is about 10–20% (Manzi 2012).”
- Slack monetization experiments: ~30% positive (Ch. 1, Tenet 3, p. 13) — “only about 30% of monetization experiments show positive results; “if you are on an experiment-driven team, get used to, at best, 70% of your work being thrown away. Build your processes accordingly””
- Netflix: 90% of what they try considered wrong (secondhand, Moran 2007) (Ch. 1, Tenet 3, p. 13) — “Mike Moran (Moran 2007, 240) wrote that Netflix considers 90% of what they try to be wrong.”
- Twyman's law as the trust heuristic (Preface, p. xv) — “Any figure that looks interesting or different is usually wrong”
- Strategy drives the OEC; experiments hill-climb within a strategy (Ch. 1, 'Strategy, Tactics, and Their Relationship to Experiments', p. 21) — “Having a strategy is critical for running experiments: the strategy is what drives the choice of OEC.”
- Redesigns and big jumps usually fail; keep a portfolio (Ch. 1, Scenario 2, p. 23) — “Our experience is that most big jumps fail (e.g., big site redesigns), yet there is a risk/reward tradeoff: the rare successes may lead to large rewards that compensate for many failures.”
- Portfolio recommendation: mostly near-optimisation, a few radical bets (Ch. 1, Scenario 2, p. 23) — “we recommend always having a portfolio of ideas: most should be investments in attempting to optimize “near” the current location, but a few radical ideas should be tried”
- Guardrail metrics state what the org will not trade away (Ch. 1, p. 22) — “Defining guardrail metrics for experiments is important for identifying what the organization is not willing to change”
- Part IV (platform) is aimed at engineers; Part II for leaders (Preface, p. xvi) — “Part IV focuses on building an experimentation platform and is aimed toward engineers.”
- Table of contents anchors for later chapters (not fetched): Ch. 6 Organizational Metrics p. 90; Ch. 7 OEC p. 102 incl. 'Goodhart’s Law, Campbell’s Law, and the Lucas Critique' p. 109; Ch. 10 Complementary Techniques p. 127 (logs, human evaluation, UER, focus groups, surveys, external data); Ch. 15 ramping p. 171; Ch. 21 Sample Ratio Mismatch p. 219; Ch. 23 long-term effects p. 235 (Contents, pp. ix-xiii) — “7 Metrics for Experimentation and the Overall Evaluation Criterion 102 ... Goodhart’s Law, Campbell’s Law, and the Lucas Critique 109”

Conflicts:
- Idea-success figures in Ch. 1 are not one number: 1/3 (Microsoft), 10-20% (Bing/Google, Manzi 2012), ~30% (Slack monetization), '90% wrong' (Netflix via Moran 2007) measure different things in different domains. S5 reports '10-20%' for roadmap items with positive ROI from unnamed 'industry analysts' - same range, different base.
- S15 (Bezos 2016) argues for deciding at ~70% information and warns that survey/beta averages can substitute for customers; S2 argues experiments should inform the HiPPO, and itself notes Apple under Jobs ran no ad tests.

### S3 · blog · Silicon Valley Product Group (SVPG) — The Four Big Risks (2017 / dated December 4, 2017 (byline))
Authors: Marty Cagan
https://www.svpg.com/four-big-risks/ · accessed 2026-10-26 · fetched: true
Authority: Practitioner essay by the originator of the four-risks framing; widely used in product-discovery practice. Assertion, not data.
Covers: discovery-delivery, roles-engineer-pm, problem-framing

Facts:
- Value risk definition (¶4) — “value risk (whether customers will buy it or users will choose to use it)”
- Usability risk definition (¶4) — “usability risk (whether users can figure out how to use it)”
- Feasibility risk definition (¶4) — “feasibility risk (whether our engineers can build what we need with the time, skills and technology we have)”
- Business viability risk definition (¶4) — “business viability risk (whether this solution also works for the various aspects of our business)”
- Viability includes go-to-market and sales channel fit (¶5) — “Business viability risk includes whether the product fits with the go-to-market or sales channel”
- PM owns value and viability (¶6) — “responsible for the value and viability risks”
- Designer owns usability (¶7) — “responsible for the usability risk”
- Lead engineer owns feasibility (¶8) — “responsible for the feasibility risk”
- The neglected risk is viability (¶3) — “too often under-appreciated and under-estimated (or simply avoided) by the product manager”
- Which risk is hardest (¶9) — “the toughest risk is usually value risk”
- Advice on sequencing (¶10) — “Tackle the big risks early – especially value risk and business risk”
- Original (first-edition) framing had three risks, which let PMs overlook business value (¶2) — “too easy for product managers to focus on customer value and overlook business value”

Conflicts:
- The page names no technique for testing any risk; testing methods are in S5, S6 and S9.
- S1 (Shape Up) names no PM and assigns shaping to a generalist/small group; S3 gives the PM the value and viability risks.

### S4 · blog · Silicon Valley Product Group (SVPG) — The Alternative to Roadmaps (2015 / dated September 7, 2015 (byline))
Authors: Marty Cagan
https://www.svpg.com/the-alternative-to-roadmaps/ · accessed 2026-10-26 · fetched: true
Authority: Originating statement of the outcome-roadmap position by the SVPG founder. Dated; the hypothetical numbers are not data.
Covers: roadmaps-okrs, strategy

Facts:
- Critique of the standard roadmap (¶2) — “typical roadmaps do just what the General warned against – they tell the team what to do.”
- What a typical roadmap is (¶2) — “a prioritized list of features or projects”
- Roadmaps are a command-and-control artefact (¶11) — “Roadmaps derive from the old centralized command-and-control model.”
- Alternative: vision + prioritized business objectives per team + OKRs (¶17-19) — “my current favorite is the OKR system (Objectives and Key Results).”
- Hypothetical objective: onboarding time 30 days to under 3 hours (illustrative, not data) (¶21) — “average on-boarding time less than 3 hours.”
- Date commitments are legitimate when needed (¶9) — “there are cases where they actually need to make date-based commitments”
- The mechanism for dates: high-integrity commitments (¶26) — “We address this with the concept of high-integrity commitments.”
- Teams are told the problem, not the solution (¶19) — “tell the team what you need them to accomplish”

Conflicts:
- S1 abandons both roadmap and backlog and bets one cycle at a time; S4 keeps objectives and OKRs but removes the feature list.
- S17/S18 (Linear) keep a roadmap with themes for about six months and a manageable backlog.
- The page does not critique dates; its target is feature/project lists (see ¶9, ¶32).

### S5 · blog · Silicon Valley Product Group (SVPG) — Changing How You Solve Problems (2022 / dated September 20, 2022; part 2 of a three-part sequence)
Authors: Jon Moore and Marty Cagan
https://www.svpg.com/changing-how-you-solve-problems/ · accessed 2026-10-26 · fetched: true
Authority: SVPG's description of discovery vs delivery and feature teams vs empowered teams; figures are the authors' estimates and the 10-20% is attributed to unnamed analysts.
Covers: discovery-delivery, roadmaps-okrs, roles-engineer-pm

Facts:
- Share of roadmap items that pay off (¶3) — “Most industry analysts place it in the 10-20% range.”
- What a feature team is for (¶10) — “They are only there to produce output – the features.”
- Accountability differs by team type (¶10) — “you can’t hold the feature team accountable to business results”
- Inputs differ: roadmap vs problems (¶15) — “a set of problems to solve and desired outcomes to achieve”
- A feature is a hypothesis (¶8) — “is a potential solution to some underlying problem”
- Discovery defined by its incentive (¶28) — “the incentive moves to being able to quickly determine if a particular product idea or approach will work.”
- Discovery is done with prototypes (¶30) — “Mostly the product team creates prototypes”
- Cost rule for discovery (¶31) — “should be at least one order of magnitude less expensive and faster”
- Iteration count and timing (authors' estimate) (¶32-33) — “If, on average, it takes 3-5 iterations for a typical feature”
- Prototypes can test the four risks (¶17) — “valuable,” “usable,” “feasible,” and “viable.”
- Engineers as innovation source (one attributed quote only) (article body) — “the single most important source of innovation”

Conflicts:
- The 10-20% figure is unsourced here; S2 gives a sourced but differently-defined 10-20% (Bing/Google metric-improvement rate, Manzi 2012).
- S2 treats randomized experiments as the arbiter; S5 says discovery is 'mostly prototypes', which are not randomized trials.

### S6 · blog · Silicon Valley Product Group (SVPG) — Build To Learn FAQ (2026 / dated April 27, 2026)
Authors: Marty Cagan
https://www.svpg.com/build-to-learn-faq/ · accessed 2026-10-26 · fetched: true
Authority: Most recent SVPG position (2026) on discovery in the age of generative AI and on prototype-as-spec; the originator updating his own earlier stance.
Covers: discovery-delivery, specs-artifacts, roles-engineer-pm, ai-ml-product

Facts:
- Discovery = build to learn; delivery = build to earn (introduction) — “building to learn (known as product discovery) versus building to earn (known as product delivery)”
- Prototype as spec ('What is the role of the PRD?') — “This is referred to as ‘prototype as spec.’”
- What the PRD still holds ('What is the role of the PRD?') — “This is usually an enumeration of specific use cases, and non-functional requirements”
- PRD is not a substitute for discovery ('What is the role of the PRD?') — “instead of product discovery”
- PMs build prototypes themselves ('So then what is my job?') — “The product manager is building and testing prototypes”
- PM is a builder, not a manager ('Isn't my job as PM to be ‘the manager’?') — “You are a builder, not a manager.”
- AI role differs in discovery vs delivery ('Can AI accelerate discovery like it has accelerated delivery?') — “In discovery, AI plays more of a prototyping and decision support role.”
- AI in delivery is code generation (same section) — “The main difference is that in delivery, generative AI plays very much an automation / code generation role.”
- Rate-of-change caution ('What about the logic of faster output -> faster outcomes?') — “constant, erratic change”
- Prototypes test the four risks ('What exactly are we trying to learn?') — “testing these prototypes against these four risks”
- PM decision load ('Isn't my job as PM to be ‘the decider’?') — “dozens or even hundreds of decisions every day”

Conflicts:
- S8 (same author, 2005): the PRD is 'a more important, higher leverage piece of work' and the primary spec; S6 (2026): prototype is the primary spec and the PRD only covers what a prototype cannot show.
- S24 (Ng, 2025): AI makes clear written specs more valuable; S6 (2026): prototype, not document, is the spec.
- S6 names no risks of AI-assisted prototyping and does not use the term 'vibe coding'.

### S7 · blog · Silicon Valley Product Group (SVPG) — Stakeholders and the Product Model (2025 / dated December 1, 2025 (page metadata); defers detail to the book TRANSFORMED)
Authors: Chris Jones and Marty Cagan
https://www.svpg.com/stakeholders-and-the-product-model/ · accessed 2026-10-26 · fetched: true
Authority: SVPG's current statement of what stakeholders should expect from product teams; the only fetched source whose subject is stakeholder management, and it is written from the product-team side.
Covers: stakeholders, roadmaps-okrs, discovery-delivery

Facts:
- Definition of stakeholder (Intro ¶3) — “You are a stakeholder if you are responsible for some key aspect of your company’s business.”
- Stakeholders supply business constraints that solutions must satisfy (Foundation > 1. Sharing Business Context, ¶3) — “solutions must solve for both customers and also for the various dimensions of the business.”
- Feature roadmaps rarely deliver business results (Foundation > 2. Framing Work, ¶1) — “roadmaps of features so seldom generate the necessary business results”
- Reframing requests into problems (Foundation > 2. Framing Work, ¶3) — “reframe the many requests for features and projects (no matter where they originate)”
- Team access to customers and data is a precondition (Foundation > 3. Providing Access, ¶1) — “product teams require direct, unencumbered access to customers, users and data”
- High-integrity commitment: dates cost discovery work and are used sparingly (Practice > High-Integrity Commitments, ¶2) — “This is a date you can count on, but it requires some product discovery work”
- Cost of date commitments (Practice > High-Integrity Commitments, ¶2) — “These come at a cost, so they should be used sparingly”
- Product team must show it understands constraints and invite feedback (Product Development > Discovering Effective Solutions, ¶2) — “demonstrate they understand your business constraints”

Conflicts:
- The page gives no method for resolving conflicting stakeholder constraints or for handling executive demands; it defers to the book TRANSFORMED (not fetched).
- Written from the product-team side with no dissenting view; Amazon's 'disagree and commit' (S15) is a different mechanism for resolving stakeholder conflict.

### S8 · blog · Silicon Valley Product Group (SVPG) — How To Write a Good PRD (SVPG white paper, 26 pp.) (2005 / © 2005 SVPG; PDF re-hosted on svpg.com 2024/07; footnote 2: 'based in part on the notes and insights of Ben Horowitz')
Authors: Martin (Marty) Cagan
https://www.svpg.com/wp-content/uploads/2024/07/How-To-Write-a-Good-PRD.pdf · accessed 2026-10-26 · fetched: true
Authority: The traditional-PRD position, stated by the same author who later argued for prototype-as-spec (S6). Useful as the 'before' in the artifact debate; the author's current position differs.
Covers: specs-artifacts, users-jtbd, problem-framing, prioritization

Facts:
- PRD purpose (Overview, p. 1) — “The purpose of the product requirements document (PRD) or product spec is to clearly and unambiguously articulate the product’s purpose, features, functionality, and behavior.”
- PRD called highest-leverage work (Overview, p. 1) — “It’s hard to come up with a more important, higher leverage piece of work for a company.”
- Relation to strategy and roadmap (Overview, p. 1) — “The product strategy describes a vision, typically between two and five years out, of where you want the product to go, and the product roadmap describes the various steps to get there. The PRD describes a particular product release along that path.”
- Ten steps: 1 Do Your Homework; 2 Define the Product’s Purpose; 3 Define the User Profiles, Goals and Tasks; 4 Define your Product Principles; 5 Prototype and Test the Product Concept; 6 Identify and Question Your Assumptions; 7 Write It Down; 8 Prioritize; 9 Test Completeness; 10 Managing the Product (step headings throughout the paper) — “Step 5: Prototype and Test the Product Concept”
- Over-confidence in specs is the commonest mistake (Step 5) — “One of the biggest and most common mistakes product teams make is to have far more confidence in their product specifications than they should”
- Elevator-pitch test for the value proposition (Step 2, p. 2) — “If you had a chance to ride the elevator with the CEO of your company, and she asked what the point of your product was, could you answer that question before the ride is up?”
- Persona example (demographic persona; contrast with S10 on JTBD) (Step 3, pp. 3-4) — “Leon the Power Seller is a 46 year old male that lives in Fresno and runs a small motorcycle parts business.”
- Objectives must be prioritized and measurable (Step 2, p. 3) — “The product requirements need to specify exactly what the objectives of this specific product release is, and how they will be measured. The objectives should also be prioritized.”
- Provenance (footnote 2, p. 1) — “This paper is based in part on the notes and insights of Ben Horowitz, President and CEO of Opsware.”

Conflicts:
- S6 (same author, 2026) reverses the emphasis: prototype as spec, PRD only for what prototypes cannot carry.
- S10 (HBS on JTBD) argues demographic segments correlate with but do not cause purchase; S8 Step 3 builds demographic personas.
- Step 5 here (prototype and test) already anticipates S6; the change is one of primacy, not of kind.

### S9 · blog · Product Talk — Opportunity Solution Trees (Product Talk) (undated / page undated; author of the book Continuous Discovery Habits (2021 per secondary sources; the book was not fetched))
Authors: Teresa Torres
https://www.producttalk.org/opportunity-solution-tree/ · accessed 2026-10-26 · fetched: true
Authority: The originator's own description of the opportunity solution tree and continuous discovery. Author discloses commercial ties ('I also license AI services to Vistaly'). Prescriptive; no outcome data.
Covers: discovery-delivery, problem-framing, users-jtbd, roles-engineer-pm

Facts:
- Definition of an OST (page passage: opening) — “An opportunity solution tree is a simple way of visually representing the paths you might take to reach a desired outcome.”
- Opportunities are customer needs, pain points, desires (page passage: Opportunities) — “These are the customer needs, pain points, and desires that, if addressed, will drive your desired outcome.”
- Prefer a product outcome at the root (page passage: Start with a product outcome) — “As a general rule, I like to see a product outcome at the top of an opportunity solution tree.”
- Weekly interviews (page passage: Continuous discovery) — “You want to interview customers continuously—week over week.”
- Revision trigger (page passage: Continuous discovery) — “For every three to four interviews that you conduct”
- Compare-and-contrast claim (uncited on the page) (page passage: Compare and contrast) — “Decision-making research tells us when we compare and contrast our options, we make better decisions.”
- Consider more than one solution per opportunity (page passage: Compare and contrast) — “As a general rule, I recommend that teams consider more than one solution for their target opportunity.”
- Product trio defined (image caption) (page passage: image caption) — “The product trio consists of a product manager, a designer, and a software engineer.”
- Garbage-in warning (page passage: prerequisites) — “Opportunity solution trees are nearly impossible to create when you don’t have good inputs.”
- Do not build one company-wide tree (page passage: limits) — “I don’t recommend this. I’ve seen many companies try and it never turns out well.”
- Sales/support inputs lack context (page passage: limits) — “most of the time when we hear about an opportunity through these sources, it is missing context.”
- Discovery has no black-and-white answers (page passage: limits) — “There are no black and white answers in discovery.”

Conflicts:
- The page recommends a product outcome at the root yet also says 'An opportunity solution tree starts with business value' (internal tension).
- S1 (Shape Up) gives no customer-interview cadence and expects ideas that matter to 'come back'; S9 prescribes weekly interviews.
- S19/S20 (PostHog) give a 'rule of thumb' of 1 interview per week per PM, with engineers not required to interview weekly; S9 assigns interviewing to the whole trio including the engineer.

### S10 · analysis · Harvard Business School Working Knowledge — Clay Christensen's Milkshake Marketing (HBS Working Knowledge) (2011 / dated February 14, 2011 (byline))
Authors: Carmen Nobel (reporting on Clayton Christensen's teaching note)
https://www.library.hbs.edu/working-knowledge/clay-christensens-milkshake-marketing · accessed 2026-10-26 · fetched: true
Authority: HBS's own account of the jobs-to-be-done framing and the milkshake case. Secondhand report of Christensen's teaching note; the primary texts (Christensen et al., HBR Sept 2016, 94(9):54-62; Competing Against Luck, 2016) are paywalled/not fetched.
Covers: users-jtbd, problem-framing

Facts:
- Base rate of new-product failure used as motivation (opening) — “each year 30,000 new consumer products are launched—and many of them fail.”
- Jobs-to-be-done framing (body) — “they essentially "hire" a product to do that job.”
- Milkshake share bought in the morning (milkshake case) — “40 percent of the milkshakes were purchased first thing in the morning, by commuters who ordered them to go.”
- The job: a long boring commute (milkshake case) — “They faced a long, boring commute”
- Constraints that define the job (milkshake case) — “They were in a hurry, they were wearing work clothes, and they had (at most) one free hand.”
- Why product/demographic segmentation fails (argument) — “The problem is that consumers usually don't go about their shopping by conforming to particular segments.”
- Demographics correlate, they do not cause (argument) — “It may be correlated with the decision, but it doesn't cause it.”

Conflicts:
- A competing JTBD school (Strategyn/Ulwick, via a search snippet only, not fetched) argues a long boring commute is a context, not the job - the framework itself has no single definition of 'job'.
- S8 builds demographic personas ('a 46 year old male that lives in Fresno'); S10 says that kind of segment does not cause the purchase.

### S11 · blog · Intercom — RICE: Simple prioritization for product managers (2018 / dated 5 January 2018 (page); a search result gave 2016 - the page date is used)
Authors: Sean McBride
https://www.intercom.com/blog/rice-simple-prioritization-for-product-managers/ · accessed 2026-10-26 · fetched: true
Authority: The originating builder-org post for RICE scoring. The formula appears only as an image on the page; the scales are stated in text. The page itself warns the scores are not a rule.
Covers: prioritization

Facts:
- RICE measures total impact per time worked (page passage: opening) — “total impact per time worked”
- Reach unit (page passage: Reach) — “Reach is measured in number of people/events per time period.”
- Impact scale: 3 massive, 2 high, 1 medium, 0.5 low, 0.25 minimal (page passage: Impact) — “3 for “massive impact”, 2 for “high”, 1 for “medium”, 0.5 for “low”, and finally 0.25 for “minimal”.”
- Confidence scale (page passage: Confidence) — “100% is “high confidence”, 80% is “medium”, 50% is “low”.”
- Below 50% is a moonshot (page passage: Confidence) — “Anything below that is “total moonshot”.”
- Effort in person-months across product, design, engineering (page passage: Effort) — “person-months”
- Worked example inputs: Project 1 reach 450/quarter, impact 3, confidence 100%, effort 2 person-months; Project 2 2,000/quarter, 1, 80%, 4; Project 3 800/quarter, 2, 50%, 1 (page passage: worked example table) — “500 × 30% × 3 = 450 customers per quarter.”
- The method is not a rule (page passage: limitations) — “RICE scores shouldn’t be used as a hard and fast rule.”
- Impact is a guess (page passage: limitations) — “Choosing an impact number may seem unscientific.”
- Dependencies are not modelled (page passage: limitations) — “One project may be a dependency for another project, so it needs to happen first”

Conflicts:
- Estimate-first (S11 effort in person-months) vs appetite-first (S1 ch. 3).
- S2 reports idea success rates of 10-33%; S11's confidence scale (100/80/50%) is a subjective multiplier, not calibrated against such base rates (researcher observation, not stated in either source).
- The page does not show the computed scores for its own worked example; the formula (Reach x Impact x Confidence / Effort) is described in prose only.

### S12 · official · Google — Guide: Set goals with OKRs (2016 (guide); retained as a record / page notes 'some references may reflect their original era')
Authors: Google re:Work
https://rework.withgoogle.com/intl/en/guides/set-goals-with-okrs · accessed 2026-10-26 · fetched: true
Authority: Google's own published description of how it sets OKRs; official, but the page cites no outcome data beyond a Sears example.
Covers: roadmaps-okrs, strategy

Facts:
- Objective definition (page passage: What are OKRs?) — “Objectives are ambitious and may feel somewhat uncomfortable”
- Key result definition (page passage: What are OKRs?) — “Key results are measurable and should be easy to grade with a number”
- Grading scale (page passage: Grading) — “OKRs are usually graded on a scale of 0.0 to 1.0, 1.0 meaning the objective was fully achieved”
- Target score is below 1.0 (page passage: Grading) — “The "sweet spot" for an OKR grade is 60% – 70%”
- Always hitting 1.0 means under-ambition (page passage: Grading) — “if someone consistently fully attains their objectives, their OKRs aren't ambitious enough”
- Cadence (page passage: How Google uses OKRs) — “Today, Google sets annual and quarterly OKRs and holds company-wide meetings quarterly to share and grade OKRs”
- Recommended count (page passage: Tips) — “Pick just three to five objectives”
- Key results per objective (page passage: Tips) — “Determine around three key results per objective”
- OKRs are public (page passage: Transparency) — “OKRs are public so that everyone in the organization can see what others are working on”
- OKRs are not performance reviews (page passage: OKRs vs evaluations) — “OKRs are not synonymous with employee evaluations”
- Caveat on stretch goals (page passage: caveats) — “Creating unachievable goals is tricky as it could be seen as setting a team up for failure”
- Honest limit (page passage: caveats) — “This isn't a science but it's important to be honest”

Conflicts:
- S4 endorses OKRs as the alternative to feature roadmaps; S1 and S17 (Linear) describe their planning without OKRs (absence of mention, not an expressed position against).
- S2 recommends an OEC as the alignment mechanism for experiment-driven teams and warns 'percent of plan delivered' ignores impact; S12's 0.6-0.7 target grades attainment of key results, which can themselves be output measures.
- The page does not address committed vs aspirational OKRs (some secondary sources do; not asserted here).

### S13 · paper · Google (CHI 2010, Atlanta, April 10-15, 2010; ACM author's version) — Measuring the User Experience on a Large Scale: User-Centered Metrics for Web Applications (CHI 2010 note, 4 pp.) (2010 / ACM author's version hosted by Google Research)
Authors: Kerry Rodden, Hilary Hutchinson, Xin Fu
https://research.google.com/pubs/archive/36299.pdf · accessed 2026-10-26 · fetched: true
Authority: The originating paper for the HEART framework and Goals-Signals-Metrics process, from the Google UX research team that used them on 20+ products.
Covers: metrics-experimentation, problem-framing

Facts:
- HEART categories ('HEART METRICS') — “HEART: Happiness, Engagement, Adoption, Retention, and Task success.”
- PULSE metrics (the thing HEART complements) ('PULSE METRICS') — “Page views, Uptime, Latency, Seven-day active users (i.e. the number of unique users who used the product at least once in the last week), and Earnings.”
- Metrics do not replace formative research ('RELATED WORK') — “they are primarily useful for evaluation of launched products, and are not a substitute for early or formative user research.”
- Why seven-day actives can mislead ('PULSE METRICS') — “In a worst-case retention scenario of 100% turnover in the user base from week to week, the count of seven-day active users could still increase, in theory.”
- Engagement reported per user, not as a total ('Engagement') — “It is generally more useful to report Engagement metrics as an average per user, rather than as a total count”
- Gmail engagement metric and its predictive value ('Engagement') — “the percentage of active users who visited the product on five or more days during the last week”
- ...strongly predictive of retention ('Engagement') — “this was strongly predictive of longer-term retention”
- Change aversion example (iGoogle redesign; satisfaction dipped and recovered) ('Happiness') — “indicating that change aversion was probably the cause”
- Task Success example: Google Maps single vs dual search box A/B test measured by error rates ('Task Success') — “They compared error rates in the two versions”
- Goals-Signals-Metrics process ('GOALS – SIGNALS – METRICS') — “identifying the goals of a product or feature, then identifying signals that indicate success, and finally building specific metrics to track on a dashboard.”
- Teams often do not agree on goals; the process surfaces this ('Goals') — “Different team members may disagree about what the project goals are.”
- Normalize raw counts ('Metrics') — “Raw counts will go up as your user base grows, and need to be normalized; ratios, percentages, or averages per user are often more useful.”
- Scope of validation ('CONCLUSIONS') — “applied to more than 20 different products and projects from a wide variety of areas within Google”

Conflicts:
- The paper reports no data on whether HEART improved product outcomes; evidence is Google-internal examples (Gmail, iGoogle, Maps, Finance).
- S2 puts a single composite OEC (and guardrails) at the center of decisions; S13 offers a category menu from which a team chooses metrics per goal and says not every category applies.

### S14 · paper · Microsoft Research (CHI 2019, Glasgow, May 4-9, 2019; Honorable Mention Award) — Guidelines for Human-AI Interaction (CHI 2019) (2019 / camera-ready PDF; DOI 10.1145/3290605.3300233)
Authors: Saleema Amershi, Dan Weld, Mihaela Vorvoreanu, Adam Fourney, Besmira Nushi, Penny Collisson, Jina Suh, Shamsi Iqbal, Paul Bennett, Kori Inkpen, Jaime Teevan, Ruth Kikin-Gil, Eric Horvitz
https://www.microsoft.com/en-us/research/wp-content/uploads/2019/01/Guidelines-for-Human-AI-Interaction-camera-ready.pdf · accessed 2026-10-26 · fetched: true
Authority: Peer-reviewed CHI paper from Microsoft Research that is the basis of the HAX Toolkit; validated with 49 practitioners across 20 AI products. The only fetched source on product design for probabilistic (ML) behavior.
Covers: ai-ml-product, metrics-experimentation

Facts:
- Scope and validation (Abstract) — “We propose 18 generally applicable design guidelines for human-AI interaction. These guidelines are validated through multiple rounds of evaluation including a user study with 49 design practitioners who tested the guidelines against 20 popular AI-infused products.”
- G1 (Table 1, 'Initially') — “Make clear what the system can do.”
- G2 (Table 1, 'Initially') — “Make clear how well the system can do what it can do. Help the user understand how often the AI system may make mistakes.”
- G3-G6 (During interaction): time services based on context; show contextually relevant information; match relevant social norms; mitigate social biases (Table 1) — “G3 Time services based on context.”
- G7-G11 (When wrong): support efficient invocation; dismissal; correction; scope services when in doubt; make clear why the system did what it did (Table 1) — “G10 Scope services when in doubt.”
- G11 (Table 1) — “Make clear why the system did what it did.”
- G12-G18 (Over time): remember recent interactions; learn from user behavior; update and adapt cautiously; encourage granular feedback; convey the consequences of user actions; provide global controls; notify users about changes (Table 1) — “G14 Update and adapt cautiously.”
- Why AI breaks classic UI heuristics (§1) — “AI-infused systems can violate established usability guidelines of traditional user interface design”
- Probabilistic behavior as the root issue (§1) — “automated inferences are typically performed under uncertainty, often producing false positives and false negatives”
- Method: 150+ recommendations consolidated to 20 guidelines, internal heuristic evaluation to 18, user study with 49 participants, expert evaluation (§3) — “In Phase 1 we consolidated more than 150 design recommendations from multiple sources into a set of 20 guidelines.”
- Overlap with earlier mixed-initiative principles (related-work paragraph (section number not captured)) — “8 of our 18 guidelines map to principles outlined in that work”

Conflicts:
- Phase labels: §3 text groups G1-2 / G3-6 / G7-11 / G12-18; Table 1 prints the labels beside the rows - the grouping used here is the §3 text.
- G14 'Update and adapt cautiously' vs S2's experimentation volume (10,000+ experiments/year at Bing): a tension between continuous change and user trust (S13 'change aversion', S18 'iterate too much' say the same).
- The paper evaluates guideline applicability, not business or retention outcomes.

### S15 · official · Amazon — 2016 Letter to Shareholders (2016 (page is titled '2016 Letter to Shareholders'; the page itself carries no date) / aboutamazon.com copy)
Authors: Jeff Bezos
https://www.aboutamazon.com/news/company-news/2016-letter-to-shareholders · accessed 2026-10-26 · fetched: true
Authority: Primary, official statement of Amazon's decision-making principles by its CEO. Excerpts are capped in this extraction; the 'Type 1 / Type 2' terminology is NOT on this page (see notes).
Covers: stakeholders, metrics-experimentation, strategy, roles-engineer-pm

Facts:
- No single decision process (page passage: High-Velocity Decision Making) — “never use a one-size-fits-all decision-making process.”
- Reversible decisions (page passage: High-Velocity Decision Making) — “Many decisions are reversible, two-way doors.”
- 70% information rule (page passage: High-Velocity Decision Making) — “most decisions should probably be made with somewhere around 70% of the information you wish you had.”
- Disagree and commit as the conflict mechanism (page passage: High-Velocity Decision Making) — “use the phrase “disagree and commit.””
- Process can become a proxy for outcomes (page passage: Resist Proxies) — “A common example is process as proxy.”
- Process serves customers (page passage: Resist Proxies) — “Good process serves you so you can serve customers.”
- Who owns whom (page passage: Resist Proxies) — “do we own the process or does the process own us?”
- Customers are never satisfied; invent on their behalf (page passage: True Customer Obsession) — “customers are always beautifully, wonderfully dissatisfied”

Conflicts:
- Search results disagree on whether the Type 1/Type 2 labels first appear in the 2015 or 2016 letter; the fetched 2016 page uses 'two-way doors' and not 'Type 1/Type 2'. Locate the 2015 letter before using the label.
- S2 argues for experiment-based evidence; S15 argues survey and beta-test averages can stand in for real customers and that 70% information is enough - opposite emphasis on how much evidence before acting.

### S16 · textbook · workingbackwards.com (authors' site); the book is titled Working Backwards (Bryar and Carr, per a Commoncog description in search results; edition/year and subtitle not verified - book not fetched) — Working Backwards: the PR/FAQ process (authors' companion site for the book Working Backwards) (undated page / web page as accessed; the book text was not fetched)
Authors: Colin Bryar and Bill Carr
https://www.workingbackwards.com/concepts/working-backwards-pr-faq-process/ · accessed 2026-10-26 · fetched: true
Authority: The former Amazon executives' own description of the PR/FAQ; the page also sells a course, so treat as authors' practitioner account, not independent evidence.
Covers: specs-artifacts, problem-framing, discovery-delivery

Facts:
- What Working Backwards is (page passage: opening) — “Working Backwards is a systematic way to vet ideas and create new products.”
- Principal tool is a written narrative (page passage: opening) — “Its principal tool is a second form of written narrative called the PR/FAQ”
- Low cost of a first draft (page passage: process) — “The first draft of a PR/FAQ should take only a few hours, not a few days.”
- External then internal FAQ (page passage: FAQ) — “The first section is devoted to External FAQs.”
- Internal FAQ holds the hard problems (page passage: FAQ) — “the internal FAQ should cover all the challenging problems that need to be solved to build the product”
- Customer-first rule (page passage: principle) — “Start with the customer and work backwards”
- Meeting format (page passage: review meeting) — “15-20 minutes of silent reading and note-taking.”
- Review group size (page passage: review) — “A small group of about 10 contributors makes sense for the first few drafts.”
- AWS timeline example (page passage: example) — “spent more than a year writing, revising, debating, and discussing PR/FAQs”

Conflicts:
- The page gives no length limit for the press release and does not mention a six-page memo; secondary sources disagree on length (one-page press release vs up to six pages).
- S1's pitch and S16's PR/FAQ are both written-narrative alternatives to a PRD but start from different places: S1 from the problem plus an appetite; S16 from the finished customer announcement.
- S6 argues the prototype is the spec; S16's artifact is a document, written before any build.

### S17 · official · Linear — Linear Method: Principles & Practices (and 'Set the product direction') (undated / linear.app/method/introduction and /method/product-direction, accessed 2026-10-26)
Authors: Linear
https://linear.app/method/introduction · accessed 2026-10-26 · fetched: true
Authority: Linear's own published practices (a builder-org's operating method for a developer-adjacent product). Claims are stated as practice; no data. Page does not mention product managers.
Covers: roadmaps-okrs, specs-artifacts, prioritization, roles-engineer-pm, discovery-delivery

Facts:
- Eight named principles (Build for the creators; Purpose-built; Create momentum – don't sprint; Meaningful direction; Aim for clarity; Say no to busy work; Simple first, then powerful; Decide and move on) (page passage: introduction) — “Create momentum – don't sprint”
- Twelve named practices incl. 'Work in n-week cycles', 'Keep a manageable backlog', 'Write project specs', 'Mix feature and quality work', 'Measure progress with actual work', 'Run cross-functional teams' (page passage: introduction) — “Keep a manageable backlog”
- Cycle length (page passage: Work in n-week cycles) — “2-week cycles are the most common in software building.”
- Backlog policy: let low-priority items die (page passage: Keep a manageable backlog) — “Important ones will resurface, low priority ones will never get fixed.”
- Cycle trade-off (page passage: Work in n-week cycles) — “They’re short enough to not lose sight of other priorities, but long enough to build significant features.”
- Named project owner (page passage: Specify project and issue owners) — “Every project should have a named owner who’s responsible for writing the project brief and delivery.”
- Designer/engineer split (page passage: Run cross-functional teams) — “Designers and engineers should work together on projects, creating a natural push and pull.”
- Momentum over deadline-rushing (page passage: Create momentum – don’t sprint) — “The goal is to maintain a healthy momentum with your teams, not to rush towards the end.”
- Reserve space for unplanned work (page passage: Mix feature and quality work) — “Reserve some space on your product timelines for unplanned work that comes up unexpectedly”
- Direction page: initiatives articulate a vision (page passage: /method/product-direction) — “Initiatives are a critical tool for shaping this direction.”
- Decide-and-move-on (page passage: principles) — “Sometimes the most important thing is to make a decision, and move on.”

Conflicts:
- S1 (Shape Up) has no backlog and a six-week cycle; S17 keeps a 'manageable backlog' and calls 2-week cycles the most common.
- The introduction page mentions designers and engineers but never product managers; S18 (Linear's own post) states it has one PM.
- The product-direction page does not mention OKRs and links to a separate 'Set useful goals' page that was not fetched.

### S18 · blog · Linear — How we run projects at Linear (2023 / dated October 05, 2023)
Authors: Nan Yu (Product), Sabin Roman (Engineering), Raissa Largman (Engineering)
https://linear.app/now/how-we-run-projects-at-linear · accessed 2026-10-26 · fetched: true
Authority: First-person account from the builder org: a small, senior, engineer-led team with one PM. Candid about the limits of the model, but self-reported.
Covers: roles-engineer-pm, specs-artifacts, roadmaps-okrs, prioritization

Facts:
- Engineers lead projects; one PM ('Who leads') — “we have always had engineers leading projects”
- Only one PM ('Who leads') — “only one PM (Nan, our Head of Product)”
- Rotating leads ('Who leads') — “The project leads are rotational. Nobody is the de facto project lead.”
- What counts as a project ('What counts as a project') — “If multiple people work on it, and it takes over two weeks, that’s a project.”
- Roadmap horizon (planning) — “a high-level roadmap with themes that we want to address in the next six months or so.”
- Spec length ('From exploration to specs') — “concise specs (1-2 pages)”
- Purpose of a spec ('From exploration to specs') — “Giving people situational awareness is the main goal of project specs.”
- Engineers decide readiness ('Deciding when to ship') — “There’s no strict process or a document with checkmarks.”
- Debates are about product, not implementation ('Who leads') — “Most of our discussions and debates are around product decisions rather than technical implementation.”
- Over-iteration cost to customers (trade-offs) — “If you release something and iterate too much, it can be difficult for your customers.”
- Weekly updates by the lead ('Project updates') — “Each project lead writes a weekly update”

Conflicts:
- The page ties 'fewer dedicated PMs' to being small with a senior team (trade-offs); S19 (PostHog) reached the opposite conclusion in practice and hired PMs.
- The page does not say who has final authority over what gets built or how projects are prioritized.

### S19 · blog · PostHog — Product management is broken. Engineers can fix it (2024 / dated Dec 03, 2024)
Authors: James Hawkins and Lior Neu-ner
https://posthog.com/newsletter/product-management-is-broken · accessed 2026-10-26 · fetched: true
Authority: Co-founder/CEO of a developer-product company explaining why PostHog gave engineers product ownership and then hired PMs anyway; self-reported, no data beyond a few counts.
Covers: roles-engineer-pm, roadmaps-okrs, discovery-delivery

Facts:
- The failure mode of PM (page passage: opening) — “Too often product managers exist to control engineers, or shield them from organizational dysfunction.”
- Decision rights moved to engineers (page passage: What we changed) — “At PostHog, PMs no longer own the roadmap, make product decisions”
- Engineers run product teams (page passage: What we changed) — “Instead, our engineers run the show. They manage product teams.”
- PM as compass (page passage: What PMs do) — “We think of PMs as the team’s compass. They don’t decide the destination”
- Founders reversed their no-PM stance (page passage: What we learned) — “I was (partially) wrong. We need product managers.”
- Reason PMs were needed (page passage: numbers) — “we couldn’t have shipped 8+ products, or hit our revenue goals without them.”
- Team size rule (page passage: Team structure) — “Split the team when it hits 6 people. No exceptions.”
- Trust is the precondition (page passage: limits) — “Of course, this requires an extremely high-level of trust in your engineer’s decision-making”
- Autonomy needs accountability (page passage: limits) — “Without this accountability, autonomy can become directionless.”
- PM duties listed (page passage: What PMs do) — “Analyze product analytics,” “Investigate opportunities,” “Do competitor research,” “Conduct user research,” “Share industry news,” and “Track the results of the team's work.”

Conflicts:
- S21 (Horowitz) 'PM as CEO of the product' who manages the product team; S19 has engineers manage the team and PMs act as a compass.
- S18 (Linear) says fewer dedicated PMs is workable when small and senior; S19 says PostHog needed PMs to ship 8+ products.
- No outcome comparison between PM-led and engineer-led teams is given anywhere in S18-S20.

### S20 · official · PostHog — PostHog Handbook: 'Product management at PostHog' (product-team) and 'Pricing principles' (feature-pricing) (2026 (page references Q2 2026 themes) / handbook pages as accessed 2026-10-26: posthog.com/handbook/product/product-team.md and posthog.com/handbook/engineering/feature-pricing.md)
Authors: PostHog
https://posthog.com/handbook/engineering/feature-pricing.md · accessed 2026-10-26 · fetched: true
Authority: A developer-product company's own internal policy, published; the only fetched primary source on pricing a developer product and on running engineer-led product teams with PMs in support. Handbook is living and time-bound.
Covers: pricing-devtools, roles-engineer-pm, discovery-delivery, prioritization

Facts:
- Engineers own sprint planning and spec'ing; PM sets cross-product context (page passage: product-team page, overview) — “Engineers own sprint planning and spec'ing out solutions.”
- PM role (page passage: product-team page) — “Product managers primarily support their teams in reaching their goals.”
- Interview rule of thumb (page passage: product-team page) — “Our rule of thumb is 1 interview per week per PM.”
- Monthly growth review for products with PMF (page passage: product-team page) — “run a growth review at the beginning of every month for each of their products.”
- Some product teams have no PM (page passage: product-team page) — “Product teams with no PM currently”
- Usage-based pricing is the default (page passage: pricing page, Usage-based pricing) — “Usage-based pricing as the default”
- Price tracks usage (page passage: pricing page) — “The more a customer uses a product, generally the higher their bill should be.”
- Flat fees only where cost does not scale (page passage: pricing page) — “Flat-fee add-ons should be reserved for functionality where cost doesn't scale with usage”
- Competitor rule (page passage: pricing page) — “For products in an established market, we should roughly match the cheapest competitor”
- Only count competitors with real revenue (page passage: pricing page) — “a competitor must be *making actual revenue* at significant scale”
- Free-tier default for new features (page passage: pricing page) — “Our default assumption for new features is that full usage is only available on the paid plans.”
- Free tier is a one-way door (page passage: pricing page) — “It's easy to increase the free tier for existing customers, but it's very painful to decrease it”
- No per-seat charge (page passage: pricing page) — “we shouldn't (and don't) charge for extra users in an organization”
- Enterprise gating (page passage: pricing page) — “Features that are focused around extra security, permissioning, compliance”
- Cadence (page passage: pricing page) — “Each product should run a yearly pricing evaluation.”
- Admitted uncertainty (page passage: pricing page) — “we accept that we might not fully understand our cost base and make money on every product on day one”

Conflicts:
- S26 (Stripe) says a flat seat fee 'will eventually create a margin problem' for compute-heavy AI tools; S20 avoids seats altogether (no per-extra-user charge) - consistent, but only S20 shows it in a policy.
- S25 (Spolsky) says commoditize complements and that the demand curve cannot be known; S20 anchors price to the cheapest big competitor 'so long as the unit economics make sense'.
- S20 names no owner for pricing decisions; the closest process is a yearly RFC.

### S21 · blog · Andreessen Horowitz (a16z) — Good Product Manager/Bad Product Manager (2012 (posted); written about 15 years earlier per the page's own warning (Netscape era) / Posted June 15, 2012; editor warning: 'This document was written 15 years ago and is probably not relevant for today’s product managers.')
Authors: Ben Horowitz
https://a16z.com/good-product-manager-bad-product-manager/ · accessed 2026-10-26 · fetched: true
Authority: The canonical 'PM as CEO of the product' statement; an influential opinion memo with no data. Its own host warns it may be outdated.
Covers: roles-engineer-pm

Facts:
- The CEO analogy (¶1) — “A good product manager is the CEO of the product.”
- Full responsibility (¶1) — “A good product manager takes full responsibility and measures themselves in terms of the success of the product.”
- PM defines the 'what', not the 'how' (¶3) — “Good product managers crisply define the target, the “what” (as opposed to the how) and manage the delivery of the “what.””
- PM manages the product team (¶3) — “They are not part of the product team; they manage the product team.”
- Engineering counterpart (¶3) — “Good product managers are the marketing counterpart of the engineering manager.”
- Not a gopher (¶3) — “they are not gophers for engineering.”
- Bad PMs like the 'how' (¶3) — “Bad product managers feel best about themselves when they figure out “how”.”
- Written communication (¶3) — “Good product managers communicate crisply to engineering in writing as well as verbally.”
- Focus (¶5) — “Good product managers focus the team on revenue and customers.”
- Excuses (¶2) — “Bad product managers have lots of excuses.”

Conflicts:
- S22 (Eriksson): 'you are not the CEO of anything'; S19 (PostHog): engineers manage product teams; S6 (Cagan 2026): 'You are a builder, not a manager.'
- S21 says PMs 'manage the product team' and define 'what', engineering 'how'; S1 and S18 have designers/engineers shaping and deciding the product; S3 gives value/viability risk to the PM and feasibility to the lead engineer.
- The page never mentions discovery, user research, prioritization, specs by name, or metrics beyond 'revenue' and 'market share'.

### S22 · blog · Mind the Product — Product managers — you are not the CEO of anything (2017 / dated March 15, 2017)
Authors: Martin Eriksson
https://www.mindtheproduct.com/product-managers-not-ceo-anything/ · accessed 2026-10-26 · fetched: true
Authority: Co-founder of Mind the Product stating the counter-position to the CEO analogy; opinion with a quote from Ken Norton (GV) via the book Product Leadership. No data.
Covers: roles-engineer-pm, stakeholders

Facts:
- Thesis (page passage: opening) — “Unless you’re the founder and the product manager at the same time, you are not the CEO of anything.”
- PM lacks authority over the functions that decide success (page passage: opening) — “Product managers simply don’t have any direct authority over most of the things needed to make their products successful”
- Senior PMs only hire/fire direct reports (page passage: opening) — “Even today’s most senior product leaders only have hiring and firing control over their direct reports”
- Failure mode: the mini-CEO (page passage: Not just semantics) — “They’re mini-CEOs after all!”
- Authority is earned, not conferred (Ken Norton, quoted) (page passage: Lead, don’t command) — “When you’re a product manager, you’re generally not the boss.”
- Second Norton sentence (page passage: Lead, don’t command) — “You need to gain authority through your actions and your leadership skills, not your role.”
- Alternative: team sport (page passage: closing) — “Product management is a team sport.”
- Not a lone genius (page passage: You are not the CEO) — “you are not a lone genius designing a product from your ivory tower”

Conflicts:
- Direct counter to S21; the page engages Horowitz's memo in one line only.
- The 'disastrous results' of mini-CEO PMs are anecdotal.

### S23 · blog · The Pragmatic Engineer — The Product-Minded Software Engineer (2019 (originally published 05 Sep 2019; last updated 01 Feb 2022) / as stated on the page)
Authors: Gergely Orosz
https://blog.pragmaticengineer.com/the-product-minded-engineer/ · accessed 2026-10-26 · fetched: true
Authority: Known author (former Uber engineering manager; author of The Software Engineer's Guidebook) naming the traits of product-minded engineers from observation; no data, and the page says nothing about infrastructure/platform teams.
Covers: roles-engineer-pm, problem-framing, stakeholders

Facts:
- Definition (page passage: opening) — “Product-minded engineers are developers with lots of interest in the product itself.”
- Trait 1: do not just implement the spec (page passage: trait 1) — “Product-minded engineers don’t settle for getting a specification and jumping to implement it.”
- Trait 2: business and data interest (page passage: trait 2) — “They take the time to understand how the business works, how the product fits in, and what its goals are.”
- Trait 5: offer tradeoffs upfront (page passage: trait 5) — “They can quickly go back-and-forth between the two sides of the same coin”
- Trait 6: edge cases (page passage: trait 6) — “Product-minded engineers quickly map out edge cases and think of ways to reduce work on them”
- Trait 7: validate (page passage: trait 7) — “how can we validate that people will use this feature, the way we think they will?”
- Trait 8: end-to-end ownership including results (page passage: trait 8) — “They consider their work done only after getting results on user behavior and business metrics.”
- Advice: relationship with the PM (page passage: How to grow) — “Build a strong relationship with your product manager”
- Advice: user research and support (page passage: How to grow) — “Engage in user research, customer support”
- Career claim (observation, no data) (page passage: closing) — “They often become key contributors, the goto people for product managers and frequently advance to being team leads.”

Conflicts:
- Page cites no data for any trait or career outcome and does not address infrastructure or platform teams (the page's own gap; relevant to an inference-runtime engineer).
- S19/S20 (PostHog) formalise this as 'product engineers' with accountability for outcomes; S23 frames it as a trait of engineers who work alongside a PM.

### S24 · blog · DeepLearning.AI — The Batch, Issue 284 - Andrew Ng's letter on AI Product Management (2025 / dated January 15, 2025; issue heading 'Tumbling Training Costs, Desktop AI Supercomputer, Tighter AI Export Restrictions, and more...')
Authors: Andrew Ng
https://www.deeplearning.ai/the-batch/issue-284 · accessed 2026-10-26 · fetched: true
Authority: A well-known AI practitioner and educator's first-hand estimate (no source or methodology given for the ratios). Later claims about 1:1 or 2 PMs per engineer are secondary and unverified here.
Covers: roles-engineer-pm, ai-ml-product, specs-artifacts

Facts:
- Typical Engineer:PM ratio (page passage: letter) — “Many companies have an Engineer:PM ratio of, say, 6:1.”
- Range (page passage: letter) — “The ratio varies widely by company and industry, and anywhere from 4:1 to 10:1 is typical.”
- Prototypes are getting cheaper (page passage: letter) — “Writing software, especially prototypes, is becoming cheaper.”
- Value shifts to specifying (page passage: letter) — “This will significantly increase demand for people who can come up with clear specs for valuable things to build.”
- PMs should prototype (page passage: letter) — “Increasingly, I also expect strong product managers to be able to build prototypes for themselves.”
- Engineers adopt AI faster than PMs (page passage: letter) — “Software Engineers, being technical, are understanding and embracing AI much faster than Product Managers.”
- Shortage claim (page passage: letter) — “I expect this shortage to grow.”

Conflicts:
- The 6:1 and 4:1-10:1 figures are unsourced assertions. Search snippets (not fetched) attribute later, lower ratios (down to 1:1; and '1 PM per 0.5 engineers' at one AI Fund team) and a stated expectation that PM work will grow as a fraction of staffing to Ng; those are secondary and not verified here.
- S6 (Cagan 2026) says the prototype is the spec; S24 says clear specs gain value as building gets cheaper and PMs prototype themselves - overlapping but differently weighted.
- S18 (Linear) runs a company with one PM and engineer-led projects; S24 expects growing demand for people who specify what to build, which pulls the other way.

### S25 · blog · Joel on Software — Strategy Letter V (Commoditize your complements) (2002 / posted June 12, 2002 (URL path date; the page also shows 'December 5, 2016' unexplained))
Authors: Joel Spolsky
https://www.joelonsoftware.com/2002/06/12/strategy-letter-v/ · accessed 2026-10-26 · fetched: true
Authority: Widely cited essay by a software founder (Fog Creek, Stack Overflow) on why companies fund open source; economics argument by example, no data. Relevant to positioning an inference runtime relative to hardware vendors and model providers.
Covers: strategy, pricing-devtools

Facts:
- Definition: substitute (page passage: opening) — “A substitute is another product you might buy if the first product is too expensive.”
- Definition: complement (page passage: opening) — “A complement is a product that you usually buy together with another product.”
- Core economic claim (page passage: body) — “All else being equal, demand for a product increases when the prices of its complements decrease.”
- Consequence for pricing power (page passage: body) — “If you can do this, demand for your product will increase and you will be able to charge more and make more.”
- Strategy (page passage: body) — “Smart companies try to commoditize their products’ complements.”
- Commodity price (page passage: body) — “the price that arises when you have a bunch of competitors offering indistinguishable goods.”
- IBM open-source example (page passage: examples) — “They’re doing this because IBM is becoming an IT consulting company.”
- Netscape example (page passage: examples) — “They’re doing this to commoditize the web browser.”
- Sun/HP example (page passage: examples) — “Sun and HP are hardware companies.”

Conflicts:
- Page offers no data and concedes parts of the Netscape argument are speculative; the claim that complements' prices drive demand is verbal, with no elasticity figures.
- S20 (PostHog) prices by matching the cheapest large competitor; S25 argues for pushing prices down on complements - different levers on different sides of the stack.

### S26 · blog · Stripe — Pricing models for AI coding companies: What works at each stage of tool adoption (2026 / Last updated 19 April 2026)
Authors: Stripe (no author shown)
https://stripe.com/resources/more/pricing-models-for-ai-coding-companies · accessed 2026-10-26 · fetched: true
Authority: Billing-vendor guide to pricing a developer tool with variable compute cost. Qualitative; gives no prices, margins, conversion or churn data, and Stripe sells billing infrastructure.
Covers: pricing-devtools, ai-ml-product

Facts:
- Five models named: per-seat, tiered subscriptions, capability-based, usage-based, hybrid (page passage: What are the core pricing models used in AI coding tools?) — “AI coding tools use five basic pricing models”
- Per-seat trade-off (page passage: per-seat) — “Per-seat pricing is simple to procure and easy to budget for, but it doesn't reflect compute costs or value delivery.”
- Cross-subsidy in flat pricing (page passage: per-seat) — “The lighter users are subsidised by the heavier ones”
- Usage-based trade-off is predictability (page passage: usage-based) — “The trade-off is predictability”
- Unpredictability slows adoption (page passage: usage-based) — “That discomfort often slows adoption, even when average costs are lower than a flat seat fee would be.”
- Usage-based matches cost to use (page passage: usage-based) — “Charging by consumption (e.g., requests, tokens, credits) matches costs with actual use”
- Hybrid: predictable seat plus usage layer (page passage: hybrid) — “The seat price stays predictable for the customer, while the usage component keeps the vendor's economics intact”
- Margin risk of flat seats with agentic usage (page passage: margin) — “a flat seat fee will eventually create a margin problem”
- Tier differentiation must be obvious (page passage: capability-based) — “Vendors that use this pricing model must provide obvious differentiation between tiers,”
- Pricing mistakes are slow to show (page passage: closing) — “Many pricing mistakes aren't apparent until 12 months in”

Conflicts:
- Gives no numbers on prices, COGS or margins; the '88% of businesses use AI' figure links to a survey with no method given.
- S20 uses usage-based pricing as its default and charges no per-seat fee; S26 treats hybrid as the stable pattern when compute cost varies, and names no downside for hybrid.
- Scope is AI coding tools; applying it to an on-device inference runtime (where marginal compute runs on the customer's hardware) is the planner's inference, not stated by the source.

### S27 · textbook · Wiley — INSPIRED: How to Create Tech Products Customers Love, 2nd edition (the fetched table of contents is the 2008 FIRST edition, 'Inspired: How To Create Products Customers Love', not this edition) (2017 / 2nd ed.; ISBN 9781119387503 (taken from the O'Reilly and Booktopia listing URLs returned by search); O'Reilly listing dated December 2017 per a search-result summary (page not fetched, HTTP 403). 1st ed.: SVPG Press, June 18, 2008, ISBN-13 9780981690407 (Open Library page, fetched).)
Authors: Marty Cagan
https://www.oreilly.com/library/view/inspired-2nd-edition/9781119387503/ · accessed 2026-10-26 · fetched: false
Fetched parts: Open Library catalog page for the 1st edition (https://openlibrary.org/books/OL25426571M/Inspired) with its table of contents. No ToC or text of the 2nd edition could be fetched (O'Reilly 403).
Authority: The standard practitioner book on product discovery and product teams, by the SVPG founder; the 2nd edition is the one SVPG's own articles (S3) say revised the first edition's three-risk framing to four. All facts below are from the 1st-edition table of contents and must not be cited as 2nd-edition chapters.
Covers: discovery-delivery, specs-artifacts, roles-engineer-pm

Facts:
- 1st edition is organized as Part 1 People, Part 2 Process, Part 3 Product plus a Summary (Open Library ToC, 1st ed. (2008), part headings) — “Part 1. People ... Part 2. Process ... Part 3. Product”
- 1st ed. has a dedicated discovery chapter (Part 2) (Open Library ToC, 1st ed.) — “Chapter 12. Product Discovery”
- 1st ed. chapter on the spec (compare S8 vs S6) (Open Library ToC, 1st ed.) — “Chapter 18. Reinventing the Product Spec”
- 1st ed. chapter on personas (compare S10 on jobs vs segments) (Open Library ToC, 1st ed.) — “Chapter 17. Personas for Product Management”
- 1st ed. chapter on the PM-engineering boundary (Open Library ToC, 1st ed.) — “Chapter 5. Product Management vs. Engineering”
- 1st ed. chapters on validation and prototype testing (Open Library ToC, 1st ed.) — “Chapter 21. Product Validation”
- 1st ed. has a chapter on platform products (relevant to infra/platform teams; content not fetched, and may not exist in the 2nd ed.) (Open Library ToC, 1st ed., Part 3) — “Chapter 39. Keys to Platform Products”
- The first-edition framing named three risks; the later four-risks article says that framing let PMs overlook business value (evidence the 2nd edition differs) (S3 ¶2 (fetched)) — “too easy for product managers to focus on customer value and overlook business value”

Conflicts:
- Search-result summaries of Perlego and Pubmatch listings (not fetched) describe the 2nd edition as five parts: lessons from top tech companies, then the right people, the right product, the right process, the right culture. That is unverified and does not match the 1st-edition ToC fetched here; do not reuse chapter numbers or titles above for the 2nd edition.
- SVPG's later writing (S5, S6) is the better-evidenced source for current Cagan positions; S8 (2005) is closer to the first edition's era.

### S28 · textbook · Celeritas Publishing — The Principles of Product Development Flow: Second Generation Lean Product Development (publisher's sample: front matter, full table of contents, Chapter 1 'The Principles of Flow', pp. 1-24; chapters 2-9 not fetched) (2009 / 1st printing 2009; ISBN-13 978-1-935401-00-1, ISBN-10 1-935401-00-9; sample PDF distributed from lpd2.com (author's site))
Authors: Donald G. Reinertsen
http://lpd2.com/wp-content/uploads/2013/06/ReinertsenFLOWChap1.pdf · accessed 2026-10-26 · fetched: true
Fetched parts: Table of Contents and Chapter 1 only (32-page PDF read via pdftotext). The cost-of-delay quotation and WSJF formula come from SAFe's WSJF page (https://framework.scaledagile.com/wsjf/, fetched, intro only), not from this PDF.
Authority: Written from an engineering-economics framing (queues, batch size, WIP, cost of delay) by a product-development consultant; the origin of cost of delay as the prioritization variable. Statistics are from the author's own course surveys, not a representative sample.
Covers: prioritization, specs-artifacts

Facts:
- Thesis: dominant paradigm is wrong (Ch. 1, p. 1) — “I believe that the dominant paradigm for managing product development is fundamentally wrong.”
- Requirements are never fully known before design starts (Ch. 1, p. 2) — “When I survey managers in my product development courses, 95 percent will admit that they begin design before they know all requirements. In fact, the average product developer begins design when 50 percent of requirements are known.”
- FIFO is rarely the right prioritization order in product development (Ch. 1, p. 2) — “FIFO prioritization is almost never economically optimal in product development, because product development deals with high variability, nonrepetitive, nonhomogeneous flows.”
- Different projects have different delay costs (Ch. 1, p. 2) — “Different projects almost always have different delay costs, and they present different loads on our resources.”
- Twelve critical problems with orthodoxy; #1 is failure to quantify economics, #11 noneconomic flow control (Figure 1-1, p. 4) — “1. Failure to Correctly Quantify Economics”
- Only 15% of developers know their cost of delay (survey) (Ch. 1, p. 4) — “only 15 percent of product developers know the cost of delay associated with their projects.”
- Cost-of-delay estimates vary wildly if not calculated (Ch. 1, footnote 1, p. 5) — “people working on the same project will give answers that vary by 50 to 1.”
- Queues are the main cause of poor performance (Ch. 1, 'Blindness to Queues', p. 5) — “Few developers realize that queues are the single most important cause of poor product development performance.”
- Almost nobody measures queues (Ch. 1, p. 6) — “Today, only 2 percent of product developers measure queues.”
- Cost of delay is what makes queues visible (Ch. 1, p. 6) — “Without a cost of delay, queues appear to be free and therefore, unworthy of attention.”
- Executives report very high utilization (Ch. 1, 'Worship of Efficiency', p. 7) — “Executives coming to my product development classes report operating at 98.5 percent utilization in the precourse surveys.”
- Variability is treated as the enemy; the goal should be its economic impact (Ch. 1, 'Hostility to Variability', p. 8) — “Minimizing the economic impact of variability is a profoundly different goal than minimizing variability.”
- Cost-of-delay maxim, attributed to the book on SAFe's page (SAFe WSJF page (framework.scaledagile.com/wsjf), intro) — “If you only quantify one thing, quantify the Cost of Delay.”
- WSJF formula as SAFe states it (prose only on the visible page) (SAFe WSJF page, intro) — “In SAFe, WSJF is estimated as the relative cost of delay divided by the relative job duration.”
- Table of contents: Ch. 2 The Economic View p. 27; Ch. 3 Managing Queues p. 53; Ch. 4 Exploiting Variability p. 85; Ch. 5 Reducing Batch Size p. 111; Ch. 6 Applying WIP Constraints p. 143; Ch. 7 Controlling Flow Under Uncertainty p. 169 (incl. 'Sequencing Work' p. 191); Ch. 8 Using Fast Feedback p. 211; Ch. 9 Achieving Decentralized Control p. 243 (Contents, pp. vii-ix) — “2. The Economic View 27 ... 3. Managing Queues 53 ... 7. Controlling Flow Under Uncertainty 169”

Conflicts:
- The '85% of product managers cannot quantify cost of delay' claim in secondary sources is not what the primary says: p. 4 says 'only 15 percent of product developers know the cost of delay associated with their projects' (course-survey respondents, developers not PMs).
- S11 (RICE) ranks by subjective Impact x Confidence / person-months; Reinertsen argues for dollar-denominated cost of delay and sequencing by it (researcher's comparison; neither source discusses the other).
- S1 (Shape Up) fixes time and varies scope, and drops lists; Reinertsen's queue and WIP framing would call a large backlog a queue with a cost, which is consistent with S1's attrition stance but argues for measuring it rather than ignoring it (researcher's inference, unverified).
- Chapters 2-9 (the economic framework, WSJF derivation) were not fetched; the 2009 edition is old but the cost-of-delay material is the part still cited in practice.

### S29 · textbook · HarperBusiness / HarperCollins — Competing Against Luck: The Story of Innovation and Customer Choice (2016 / published 4 Oct 2016; Google Books page shows 'HarperCollins', 352 pages, ISBN 0062435639 / 9780062435637 (a library catalog record seen in search results lists xix + 262 pages: edition or counting difference, unresolved); HarperBusiness imprint per search results, not shown on the fetched page)
Authors: Clayton M. Christensen, Taddy Hall, Karen Dillon, David S. Duncan
https://books.google.com/books/about/Competing_Against_Luck.html?id=zGd_CwAAQBAJ · accessed 2026-10-26 · fetched: false
Fetched parts: Google Books catalog page (publisher description and a partial contents list only). No preview text appeared. The related HBR article (Sept 2016, 94(9):54-62) is paywalled; see S10 for the secondary account.
Authority: The authors' full statement of jobs-to-be-done (the causal-mechanism claim). Facts below are publisher blurb and catalog entries, so they establish what the book claims, not the mechanism's detail.
Covers: users-jtbd

Facts:
- Publisher's central claim: jobs, not customers (Google Books, publisher description) — “Understanding customers does not drive innovation success, he argues. Understanding customer jobs does.”
- Publisher's framing of the old maxim as wrong (Google Books, publisher description) — “our long held maxim—that understanding the customer is the crux of innovation—is wrong.”
- Framing question (Google Books, publisher description) — “Why do most companies fail at innovation, and what is the causal mechanism that will allow you to compete against luck?”
- Description feature headings (Google Books, publisher description) — “A New Perspective on the Customer”
- Contents entry visible on the page (spacing collapsed on the page): application chapter (Google Books, contents list) — “The Hard Workand Payoffof Applying Jobs”
- Contents entry visible on the page: organization chapter (Google Books, contents list) — “The Jobs to Be Done Organization”

Conflicts:
- No chapter numbers or text were fetched. A search-result summary of a library record (not fetched) lists 'Chapter 7: Integrating around a job' through 'Chapter 10: Final observations about the theory of jobs'; unverified.
- S10 (HBS) says demographic segments correlate with but do not cause purchase; this book's blurb makes the same move at larger scale. A rival JTBD school (Strategyn/Ulwick, seen only as a search snippet) disputes what counts as a job.
- The blurb is promotional copy; do not cite it as evidence that the framework works.

### S30 · textbook · Crown Business (US); the fetched catalog page is the Profile Books (UK) edition — Good Strategy/Bad Strategy: The Difference and Why It Matters (2011 / Crown, 2011, 336 pp., ISBN 0307886239 per a Google Books search listing (id 0FWPDQAAQBAJ; not fetched). Fetched page: Profile Books, 9 Jun 2011, 336 pp., ISBN 184765746X / 9781847657466 (Google Books id nTU65JgxgOAC). Page numbers below are the Profile edition's.)
Authors: Richard P. Rumelt
https://books.google.com/books/about/Good_Strategy_Bad_Strategy.html?id=nTU65JgxgOAC · accessed 2026-10-26 · fetched: false
Fetched parts: Google Books catalog page for the Profile Books edition: publisher description and table of contents with page numbers. No book text. The McKinsey Quarterly article 'The perils of bad strategy' timed out three times.
Authority: The standard text on what a strategy is (diagnosis, guiding policy, coherent action, per secondary summaries, unverified here). Only headings and publisher copy were fetched.
Covers: strategy

Facts:
- Publisher: strategy is not vision or planning (Google Books, publisher description) — “Strategy should not be equated with ambition, leadership, vision or planning”
- ToC: the kernel chapter (Google Books ToC (Profile ed.), p. 77) — “THE KERNEL OF GOOD STRATEGY”
- ToC: bad strategy chapter (Google Books ToC (Profile ed.), p. 32) — “BAD STRATEGY”
- ToC: sources of power (Google Books ToC (Profile ed.), p. 95) — “SOURCES OF POWER”
- ToC: proximate objectives (Google Books ToC (Profile ed.), p. 106) — “PROXIMATE OBJECTIVES”
- ToC: using design; focus; growth; using dynamics; inertia and entropy (Google Books ToC (Profile ed.), pp. 124-202) — “USING DESIGN”
- ToC: putting it together; thinking like a strategist (Google Books ToC (Profile ed.), pp. 223, 239) — “PUTTING IT TOGETHER”
- Publisher description heading (Google Books, publisher description) — “HOW TO IMPLEMENT ACTION-ORIENTATED STRATEGY THAT WORKS”

Conflicts:
- The kernel's three parts (diagnosis, guiding policy, coherent action) are attested only by search-result summaries of the McKinsey article and of secondary reviews, not by anything fetched; treat as unverified.
- The blurb's 'Strategy should not be equated with ... vision or planning' contradicts how S4 and S8 place a product vision at the head of strategy, and sits apart from S2's use of strategy as the choice of OEC.
- The page's mapping of titles to page numbers around pp. 43 and 58 is ambiguous ('WHY SO MUCH BAD STRATEGY?' vs an untitled entry); two editions (Crown and Profile) differ in pagination.

### S31 · textbook · Ideapress (per the coordinator and search results; not shown on the fetched page) — Obviously Awesome: How to Nail Product Positioning so Customers Get it, Buy it, Love it (author's book page) (2019 (per the coordinator; the fetched page shows no year and says 'newly updated and expanded edition', so at least two editions exist) / edition unresolved; publisher and year not stated on the author's page; the page's store links carry different identifiers (1999023056, 1999023005) that could not be matched to editions)
Authors: April Dunford
https://www.aprildunford.com/books · accessed 2026-10-26 · fetched: false
Fetched parts: Author's book page only (marketing copy and a five-item 'what you will learn' list). No table of contents, no chapter text, no preview; the author's home page shows no framework text either.
Authority: The practitioner reference for B2B/product positioning (consultant and former startup executive). Only the author's marketing description was fetched.
Covers: positioning

Facts:
- What the book promises (author's book page) — “a step by step process that any startup can follow to position their product, service or company.”
- Positioning decomposes into five components (names not given on the page) (author's book page, 'what you will learn') — “How to break positioning down into five component pieces.”
- Components are interdependent (author's book page) — “How the component pieces relate to each other and why that matters.”
- Market choice and three positioning styles (author's book page) — “How to choose the best market for your product, including three styles of positioning.”
- Market trends as a lever (author's book page) — “How to leverage market trends to help close more business right now.”
- Edition statement (author's book page) — “The newly updated and expanded edition”
- Sales claim (undated, unsourced) (author's book page) — “Over 100,000 copies sold!”

Conflicts:
- The names of the five components (competitive alternatives, unique attributes, value, target customers, market category, plus trends as a 'plus one') appear only in secondary summaries returned by search, not on any fetched page; secondary sources also disagree on the step count (10 vs 12). Treat as unverified.
- Aimed at B2B/startup products with a sales motion; no source fetched applies it to an infrastructure runtime sold to developers.

## Notes
(1) Source count is 31: the original 26 (all fetched, page or PDF) plus five books added at the coordinator's request as textbook entries (S27-S31). Only S28 (Reinertsen) has fetched text (table of contents and Chapter 1, from the publisher's sample PDF); S27, S29, S30, S31 are fetched:false with a catalog or author page fetched (see each entry's fetchedParts). Do not cite book specifics beyond what each entry's facts show. The count is 5-6 above the 25-26 target; see (9) for why nothing was retired and the drop order if the master wants 26. (2) Quotes for web pages passed through an extraction model that was told to quote verbatim and to cap quotes at about 125 characters; where-labels for web pages are the extraction's section labels or descriptive labels (not always the page's own headings) and paragraph numbers (¶) are the extraction's count. Writers must re-open the URL and confirm exact wording and location before putting a block quote in a page. Kohavi (S2), HEART (S13), Amershi (S14) and SVPG PRD (S8) were read from PDFs directly. (3) Several quotes carry curly quotation marks, as on the page. (4) TOO LARGE FOR A DEEP DIVE (log for the checkpoint; would need a course): stakeholder management and influence (S7 is one article); user-research craft (interviews, surveys, usability) - Torres and Fitzpatrick are books, not fetched; positioning and go-to-market for dev/infra (Dunford, PLG, sales motion, hardware-vendor and OS distribution); pricing research methods (willingness to pay, unit economics of inference COGS); product analytics and instrumentation (north star, funnels, cohorts, retention); experimentation statistics beyond Ch. 1 (variance reduction, SRM, interference, long-term effects; Kohavi Parts IV-V); business cases and opportunity sizing; strategy frameworks (Rumelt, Helmer, Porter); platform/infrastructure product management; AI product management in depth (evals as spec, probabilistic UX, cost per query); product operating model and organizational transformation (Cagan's TRANSFORMED); PM career, hiring and levels. (5) CLAIMS I COULD NOT SOURCE (mark <mark class="unverified"> or cut): (a) Amazon 'Type 1 / Type 2' labels - not on the fetched 2016 letter; search results disagree between the 2015 and 2016 letters. (b) Andrew Ng's later ratios (1:1, '1 PM per 0.5 engineers') - secondary only; the fetched primary has 6:1 and 4:1-10:1 for the present. (c) Dunford's five positioning components - the author's book page (S31) says positioning breaks into 'five component pieces' but names none; the names (competitive alternatives, unique attributes, value, target customers, market category, plus trends) are from secondary summaries only; publisher and year are not on the page. (d) The Mom Test rules - secondary summaries only; the official site lists none. (e) HBR 'Know Your Customers Jobs to Be Done' (Sept 2016, 94(9):54-62) - paywalled; a teaser showed only 'McKinsey poll: 84% of executives said innovation is highly important to growth; 94% dissatisfied with their organization's innovation performance'; a third-party hosted copy was blocked by the environment and not downloaded. (f) Rumelt's kernel (diagnosis, guiding policy, coherent action) and the Nvidia example - the McKinsey page timed out three times and the book (S30) was fetched as a table of contents only; the kernel is attested only by search snippets. (g) Booking.com KDD 2019 paper (Bernardi, Mavridis, Estevez, '150 Successful Machine Learning Models: 6 Lessons Learned at Booking.com', DOI 10.1145/3292500.3330744, pp. 1743-1751) - ACM returned 403 and Semantic Scholar 403; only the search-snippet abstract ('about 150 successful customer facing applications ... validated through rigorous Randomized Controlled Trials'; 'an iterative, hypothesis driven process was fundamental') exists; the 'six lessons' list is Adrian Colyer's summary, not the paper's. Writers must fetch the paper before using it. (h) RESOLVED in part by S28: the primary says 'only 15 percent of product developers know the cost of delay associated with their projects' (Ch. 1 p. 4; his course surveys). The secondary claim that '85% of PMs cannot quantify cost of delay' is a rewording of that and should not be repeated. Still unfetched: Reinertsen Ch. 2-9 (the economic framework) and the WSJF derivation; SAFe's WSJF page shows only an intro with the formula in prose and the attributed maxim 'If you only quantify one thing, quantify the Cost of Delay.' (i) Any number on prices, margins or licensing of an on-device inference runtime or SDK - no source. (j) Any data comparing PM-led vs engineer-led teams - none anywhere in S18-S20; all are self-reported. (k) Perri's 'build trap' definition and the value-exchange system - secondary summaries only. (6) BOOK POINTERS (not in the sources list; not fetched; editions as reported in search results, unverified): Marty Cagan, Empowered (with Jones, 2020) and Transformed (with Jones, 2024) - SVPG pages S3/S6/S7 refer to them; Teresa Torres, Continuous Discovery Habits (2021); Melissa Perri, Escaping the Build Trap (2018); Rob Fitzpatrick, The Mom Test (2013 - official site momtestbook.com gives only the subtitle); Madhavan Ramanujam and Georg Tacke, Monetizing Innovation (Wiley, 2016); John Doerr, Measure What Matters (2018); Eric Ries, The Lean Startup (2011); Hamilton Helmer, 7 Powers (2016); Colin Bryar and Bill Carr, Working Backwards (2021; companion site is S16). Cagan's Inspired (2nd ed.), Reinertsen, Christensen et al., Rumelt and Dunford are now entries S27-S31. (7) FETCH LOG: about 60 fetch/curl calls, the whole 60 budget (about 52 for the first 26 sources, about 8 for the book entries); about 16 were failures, redirects or truncations (O'Reilly 403 for Inspired 2nd ed.; Google Books redirected twice before returning ToC/description pages; Open Library returned the 1st edition ToC; PostHog product-engineer pages returned only navigation; HBR paywall; ACM, Semantic Scholar 403; McKinsey x3 timeout; HBS once 503; producthabits.com 404). Also fetched but not used as sources: Spolsky 'Camels and Rubber Duckies' (2004; price discrimination and segmentation: 'charge $0.05 ... unless it does bug tracking' joke, consumer surplus example $399 vs $220) - available at https://www.joelonsoftware.com/2004/12/15/camels-and-rubber-duckies/ if the planner wants pricing theory; SVPG 'Product vs Feature Teams' (2019-08-29; https://www.svpg.com/product-vs-feature-teams/); Dunford site; momtestbook.com. (8) CURRENCY: the SVPG 2026 page and Stripe 2026 page are within the last 6 months; PostHog handbook is a living document (Q2 2026 themes); Linear and Shape Up pages are undated. Flag as likely to change: AI-era role claims (S6, S24), PostHog policies, Stripe's pricing guidance.(9) RETIREMENTS: none. The coordinator asked to retire blog entries that a book or official source now covers, to hold the total at 25-26, and to keep the total if a retirement would lose a claim. Checked each candidate: the four new fetched:false books carry only catalog and ToC facts and S28 only Ch. 1, so none of them reproduces any verbatim claim held by a blog entry. Every blog entry is the only fetched source for at least one disagreement position or quotable fact (S26 seat/hybrid trade-off; S22 'not the CEO'; S19 'I was (partially) wrong' and '8+ products'; S18 one-PM and rotating-lead facts; S9 weekly interviews and the trio; S5 order-of-magnitude discovery cost and 3-5 iterations; S10 milkshake facts; S3 four-risk definitions and ownership; S4 OKR endorsement and high-integrity commitments). DROP ORDER if the master must reach 26 (lowest loss first, 5 cuts): S26 (loses the seat vs hybrid position in the pricing disagreement), S19 (D1 engineer-run position stays via S18 and S20), S18 (D1/D3/D4 each lose a Linear data point), S5 (S2 keeps the sourced 10-20% base rate; loses discovery-cost and iteration figures), S10 (only if S29's text becomes fetchable). Removing an entry requires deleting its id from sources.json, coverageSources, every disagreement, and every conflicts line that names it. 


## Added at assembly (2026-10-27)

Cited by the module pages but absent from the research map. The page writers fetched these; each page's reference list gives the exact use.

### S32 · official · PostHog. How to do product, as an engineer (PostHog Handbook, engineering). https://posthog.com/handbook/engineering/product-engineering.md. Accessed 2026-10-26
https://posthog.com/handbook/engineering/product-engineering.md · accessed 2026-10-26 · fetched true · cited by 01#4, 05#15, 06#6
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: PostHog. How to do product, as an engineer (PostHog Handbook, engineering). https://posthog.com/handbook/engineering/product-engineering.md. Accessed 2026-10-26.
Used for: good and bad product engineers; "Just shipping is often the best way to validate an idea" and its mitigations; the three PMF phases; how to talk to users; interview response rates. Living document; added by the page writer, not in the source map.

### S33 · paper · Kohavi, R., Crook, T. and Longbotham, R. Online Experimentation at Microsoft. Third Workshop on Data Mining Case Studies and Practice Prize, 2009. https://exp-p
https://exp-platform.com/Documents/ExP_DMCaseStudies.pdf · accessed 2026-10-26 · fetched true · cited by 01#6, 04#2
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: Kohavi, R., Crook, T. and Longbotham, R. Online Experimentation at Microsoft. Third Workshop on Data Mining Case Studies and Practice Prize, 2009. https://exp-platform.com/Documents/ExP_DMCaseStudies.pdf. Accessed 2026-10-26.
Used for: the originating statement that about one third of well-designed experiments improved the key metric (Section 5, p. 5); the population "ideas that pass all the internal evaluations"; "a failure of an experiment is not a mistake"; the McConnell 10-100x claim as reported. Added by the page writer, not in the source map.

### S34 · official · PostHog. User feedback (PostHog Handbook, product). https://posthog.com/handbook/product/user-feedback.md. Accessed 2026-10-26.
https://posthog.com/handbook/product/user-feedback.md · accessed 2026-10-26 · fetched true · cited by 01#10
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: PostHog. User feedback (PostHog Handbook, product). https://posthog.com/handbook/product/user-feedback.md. Accessed 2026-10-26.
Used for: usability tests for big features "run by the engineer building that feature"; PM-led general feedback. Living document; added by the page writer, not in the source map.

### S35 · official · PostHog. Product management at PostHog (PostHog Handbook, product). https://posthog.com/handbook/product/product-team.md. Accessed 2026-10-26.
https://posthog.com/handbook/product/product-team.md · accessed 2026-10-26 · fetched true · cited by 01#11, 04#15, 05#13, 06#28
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: PostHog. Product management at PostHog (PostHog Handbook, product). https://posthog.com/handbook/product/product-team.md. Accessed 2026-10-26.
Used for: engineers own sprint planning and spec'ing; PMs' "rule of thumb is 1 interview per week per PM"; Q2 2026 themes. Living document.

### S36 · official · Christensen Institute. Jobs to Be Done Theory. Undated organization page (footer: © 2016-2026 Clayton Christensen Institute). https://www.christenseninstitute.o
https://www.christenseninstitute.org/theory/jobs-to-be-done/ · accessed 2026-10-26 · fetched true · cited by 02#8
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: Christensen Institute. Jobs to Be Done Theory. Undated organization page (footer: © 2016-2026 Clayton Christensen Institute). https://www.christenseninstitute.org/theory/jobs-to-be-done/. Accessed 2026-10-26.
Used for: the definition (circumstances or forces; progress in particular circumstances), the morning-milkshake summary, the condominium case (Moesta) and its self-reported figures, the college example (200+ stories, five jobs). Added by the writer beyond the source map.

### S37 · blog · Ulwick, A. Jobs-to-be-Done (A Comprehensive Guide). Strategyn (consultancy vendor page; undated; © 2025). https://strategyn.com/jobs-to-be-done/. Accessed 2026-
https://strategyn.com/jobs-to-be-done/ · accessed 2026-10-26 · fetched true · cited by 02#10
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use. VENDOR PAGE: its success-rate claim is an unverified vendor claim.
Citation: Ulwick, A. Jobs-to-be-Done (A Comprehensive Guide). Strategyn (consultancy vendor page; undated; © 2025). https://strategyn.com/jobs-to-be-done/. Accessed 2026-10-26.
Used for: Ulwick's definition, solution-free job statement, desired outcomes, eight-step job map, survey stage, three customer roles; vendor-stated success figures cited as claims only. Added by the writer beyond the source map.

### S38 · official · Scaled Agile, Inc. Weighted Shortest Job First (WSJF). Scaled Agile Framework. https://framework.scaledagile.com/wsjf. Accessed 2026-10-26.
https://framework.scaledagile.com/wsjf · accessed 2026-10-26 · fetched true · cited by 03#4
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: Scaled Agile, Inc. Weighted Shortest Job First (WSJF). Scaled Agile Framework. https://framework.scaledagile.com/wsjf. Accessed 2026-10-26.
Used for: SAFe's definition of WSJF as relative cost of delay divided by relative job duration, its listed inputs, and the epigraph attributed to Reinertsen. Only the introduction was visible; the "Details" section is behind a login.

### S39 · blog · Cagan, M. (SVPG). High-Integrity Commitments. Silicon Valley Product Group, 15 December 2012. https://svpg.com/managing-commitments-in-an-agile-team/. Accessed 
https://svpg.com/managing-commitments-in-an-agile-team/ · accessed 2026-10-26 · fetched true · cited by 03#9
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: Cagan, M. (SVPG). High-Integrity Commitments. Silicon Valley Product Group, 15 December 2012. https://svpg.com/managing-commitments-in-an-agile-team/. Accessed 2026-10-26.
Used for: why commitments fail (made too early), the give-and-take with discovery, and the role of delivery managers in dating. Linked from the 2015 article above.

### S40 · textbook · Basic Scheduling Algorithms for Single Machine Problems (lecture handout, "Single machine problems"). Hosted under the NYU Stern faculty web space of M. Pinedo,
https://web-static.stern.nyu.edu/om/faculty/pinedo/scheduling/shakhlevich/handout02.pdf · accessed 2026-10-26 · fetched true · cited by 03#11
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use. WEAK: lecture handout with no author or date shown. Used only for Smith's rule (WSPT); Smith's 1956 paper was not fetched.
Citation: Basic Scheduling Algorithms for Single Machine Problems (lecture handout, "Single machine problems"). Hosted under the NYU Stern faculty web space of M. Pinedo, path "scheduling/shakhlevich"; the handout itself shows no author or date. https://web-static.stern.nyu.edu/om/faculty/pinedo/scheduling/shakhlevich/handout02.pdf. Accessed 2026-10-26.
Used for: Section 2, Theorem 1 (the weighted shortest processing time rule is optimal for total weighted completion time on one machine; proof by adjacent pairwise interchange) and its attribution to W. E. Smith, 1956. Smith's original paper was not fetched.

### S41 · textbook · Adan, I. and Resing, J. Queueing Systems. Lecture notes, Eindhoven University of Technology, 26 March 2015. https://www.win.tue.nl/~iadan/queueing.pdf. Accessed
https://www.win.tue.nl/~iadan/queueing.pdf · accessed 2026-10-26 · fetched true · cited by 03#12
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: Adan, I. and Resing, J. Queueing Systems. Lecture notes, Eindhoven University of Technology, 26 March 2015. https://www.win.tue.nl/~iadan/queueing.pdf. Accessed 2026-10-26.
Used for: §3.4 (Little's law \( E(L) = \lambda E(S) \) and its stability assumption); §4.2-4.3 (M/M/1 equilibrium distribution, mean number in system, eq. 4.9 for mean time in system, mean waiting time).

### S42 · textbook · Tran-Gia, P. and Hossfeld, T. Performance Modeling and Analysis of Communication Networks: A Lecture Note. Würzburg University Press, 2021, DOI 10.25972/WUP-978
https://hossfeld.github.io/performance-modeling/chapter5_nonmarkovianSystems/ch5-6-KingmanApproximation.html · accessed 2026-10-26 · fetched true · cited by 03#13
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: Tran-Gia, P. and Hossfeld, T. Performance Modeling and Analysis of Communication Networks: A Lecture Note. Würzburg University Press, 2021, DOI 10.25972/WUP-978-3-95826-153-2; online version, §5.6.3 "Kingman's Approximation of Mean Waiting Times". https://hossfeld.github.io/performance-modeling/chapter5_nonmarkovianSystems/ch5-6-KingmanApproximation.html. Accessed 2026-10-26.
Used for: Kingman's approximation of the mean waiting time of a GI/GI/1 queue and Daley's upper bound.

### S43 · official · PostHog. Setting quarterly goals, Handbook. Undated living page. https://posthog.com/handbook/company/goal-setting.md. Accessed 2026-10-26.
https://posthog.com/handbook/company/goal-setting.md · accessed 2026-10-26 · fetched true · cited by 04#8
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: PostHog. Setting quarterly goals, Handbook. Undated living page. https://posthog.com/handbook/company/goal-setting.md. Accessed 2026-10-26.
Used for: output-based goals, "use metrics only if they help you," the goal template, named owners, the pay-review and missed-objectives wording.

### S44 · official · Singer, R. Shape Up: Stop Running in Circles and Ship Work that Matters. 37signals / Basecamp, web edition (page states only "Copyright 1999-2026 37signals LLC"
https://basecamp.com/shapeup/2.1-chapter-07 · accessed 2026-10-26 · fetched true · cited by 04#10
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: Singer, R. Shape Up: Stop Running in Circles and Ship Work that Matters. 37signals / Basecamp, web edition (page states only "Copyright 1999-2026 37signals LLC"). Chapters 7 and 8: https://basecamp.com/shapeup/2.1-chapter-07, https://basecamp.com/shapeup/2.2-chapter-08. Accessed 2026-10-26.
Used for: the betting table's membership, "a smart bet has a cap on the downside," the pitch's "specific payout," failure read as a shaping error, and the absence of the words metric, OKR, goal and measure from both chapters.

### S45 · paper · Bernardi, L., Mavridis, T. and Estevez, P. 150 Successful Machine Learning Models: 6 Lessons Learned at Booking.com. KDD 2019, Applied Data Science Track (DOI 1
https://www.kdd.org/kdd2019/accepted-papers/view/150-successful-machine-learning-models-6-lessons-learned-at-booking.com · accessed 2026-10-26 · fetched true · cited by 04#13
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use. WEAK ACCESS: read from a PDF copy on a personal blog because the ACM page returned 403; the kdd.org listing is the canonical URL.
Citation: Bernardi, L., Mavridis, T. and Estevez, P. 150 Successful Machine Learning Models: 6 Lessons Learned at Booking.com. KDD 2019, Applied Data Science Track (DOI 10.1145/3292500.3330744). Listing: https://www.kdd.org/kdd2019/accepted-papers/view/150-successful-machine-learning-models-6-lessons-learned-at-booking.com; text read from a PDF copy of the paper hosted at https://blog.kevinhu.me/2021/04/25/25-Paper-Reading-Booking.com-Experiences/bernardi2019.pdf (the ACM page returned HTTP 403). Accessed 2026-10-26.
Used for: §3 and Figure 4 (23 comparisons, correlations and intervals), the four mechanisms, §5 latency experiment, §7.2-7.3 three-group design.

### S46 · official · Schwaber, K. and Sutherland, J. The 2020 Scrum Guide. scrumguides.org, HTML port of the November 2020 version. https://scrumguides.org/scrum-guide.html. Accesse
https://scrumguides.org/scrum-guide.html · accessed 2026-10-26 · fetched true · cited by 05#9
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: Schwaber, K. and Sutherland, J. The 2020 Scrum Guide. scrumguides.org, HTML port of the November 2020 version. https://scrumguides.org/scrum-guide.html. Accessed 2026-10-26.
Used for: Scrum Team, Developers and Product Owner accountabilities, Sprint Planning topic three, sizing, team of 10 or fewer, shared Product Owner (fetched beyond the source map).

### S47 · official · PostHog. What product managers do at PostHog. PostHog Handbook, living page. https://posthog.com/handbook/product/product-manager-role. Accessed 2026-10-26.
https://posthog.com/handbook/product/product-manager-role · accessed 2026-10-26 · fetched true · cited by 05#14, 06#2
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: PostHog. What product managers do at PostHog. PostHog Handbook, living page. https://posthog.com/handbook/product/product-manager-role. Accessed 2026-10-26.
Used for: own, share and support tiers (Table 1), team lead as closest to product owner, informed captain, stage table, when to add a PM, the infrastructure-heavy exception (added beyond the source map).

### S48 · blog · Spolsky, J. Camels and Rubber Duckies. Joel on Software, 15 December 2004. https://www.joelonsoftware.com/2004/12/15/camels-and-rubber-duckies/. Accessed 2026-1
https://www.joelonsoftware.com/2004/12/15/camels-and-rubber-duckies/ · accessed 2026-10-27 · fetched true · cited by 06#8
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: Spolsky, J. Camels and Rubber Duckies. Joel on Software, 15 December 2004. https://www.joelonsoftware.com/2004/12/15/camels-and-rubber-duckies/. Accessed 2026-10-27.
Used for: the claim that the demand curve cannot be observed, the free, cheap and dear price bands with the sales-cost estimate, and the segmentation example with its stated totals. Added beyond the source map; the researcher fetched it but did not list it.

### S49 · official · PostHog. Pricing (Markdown view of posthog.com/pricing). Living page as accessed. https://posthog.com/pricing.md. Accessed 2026-10-26.
https://posthog.com/pricing.md · accessed 2026-10-26 · fetched true · cited by 06#9
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: PostHog. Pricing (Markdown view of posthog.com/pricing). Living page as accessed. https://posthog.com/pricing.md. Accessed 2026-10-26.
Used for: bracket prices and free tiers (Table 7, Figure 7), platform packages, the annual-commitment discount table, and "no per-seat charges". Added beyond the source map.

### S50 · analysis · Rumelt, R. The perils of bad strategy. McKinsey Quarterly, 1 June 2011. https://www.mckinsey.com/capabilities/strategy-and-corporate-finance/our-insights/the-pe
https://www.mckinsey.com/capabilities/strategy-and-corporate-finance/our-insights/the-perils-of-bad-strategy · accessed 2026-10-27 · fetched true · cited by 06#12
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use. ARCHIVE COPY: read through an Internet Archive copy because the live page timed out.
Citation: Rumelt, R. The perils of bad strategy. McKinsey Quarterly, 1 June 2011. https://www.mckinsey.com/capabilities/strategy-and-corporate-finance/our-insights/the-perils-of-bad-strategy, read through the Internet Archive copy https://web.archive.org/web/2023/https://www.mckinsey.com/capabilities/strategy-and-corporate-finance/our-insights/the-perils-of-bad-strategy because the live page timed out. Accessed 2026-10-27.
Used for: the kernel of good strategy, the Nvidia, DEC and International Harvester cases, and the four hallmarks of bad strategy. The author's own article, adapted from the book; added beyond the source map.

### S51 · official · PostHog. Estimating usage and costs (Docs). Living page as accessed. https://posthog.com/docs/billing/estimating-usage-costs. Accessed 2026-10-26.
https://posthog.com/docs/billing/estimating-usage-costs · accessed 2026-10-26 · fetched true · cited by 06#17
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: PostHog. Estimating usage and costs (Docs). Living page as accessed. https://posthog.com/docs/billing/estimating-usage-costs. Accessed 2026-10-26.
Used for: why PostHog prices on usage, the argument against monthly active users, the local-evaluation request arithmetic, billing limits, and cost-reduction guidance. Added beyond the source map.

### S52 · official · PostHog. Product Analytics pricing (Docs). Living page as accessed. https://posthog.com/docs/product-analytics/pricing. Accessed 2026-10-26.
https://posthog.com/docs/product-analytics/pricing · accessed 2026-10-26 · fetched true · cited by 06#18
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: PostHog. Product Analytics pricing (Docs). Living page as accessed. https://posthog.com/docs/product-analytics/pricing. Accessed 2026-10-26.
Used for: the per-bracket allocation and subtotal in the calculator (graduated reading), and the company's claim that more than 90% of companies use PostHog for free. Added beyond the source map.

### S53 · official · PostHog. Contract rules (Handbook, growth/sales). Living page as accessed. https://posthog.com/handbook/growth/sales/contract-rules. Accessed 2026-10-27.
https://posthog.com/handbook/growth/sales/contract-rules · accessed 2026-10-27 · fetched true · cited by 06#19
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: PostHog. Contract rules (Handbook, growth/sales). Living page as accessed. https://posthog.com/handbook/growth/sales/contract-rules. Accessed 2026-10-27.
Used for: the cost-to-serve rationale for volume discounts and "discounts are earned, not given". Added beyond the source map.

### S54 · official · PostHog. Common questions about billing (Docs). Living page as accessed. https://posthog.com/docs/billing/common-questions. Accessed 2026-10-26.
https://posthog.com/docs/billing/common-questions · accessed 2026-10-26 · fetched true · cited by 06#20
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: PostHog. Common questions about billing (Docs). Living page as accessed. https://posthog.com/docs/billing/common-questions. Accessed 2026-10-26.
Used for: organization-wide usage with one free allowance, and data being dropped after a billing limit. Added beyond the source map.

### S55 · official · Stripe. Recurring pricing models (Docs). Living page as accessed. https://docs.stripe.com/products-prices/pricing-models. Accessed 2026-10-26.
https://docs.stripe.com/products-prices/pricing-models · accessed 2026-10-26 · fetched true · cited by 06#21
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: Stripe. Recurring pricing models (Docs). Living page as accessed. https://docs.stripe.com/products-prices/pricing-models. Accessed 2026-10-26.
Used for: Stripe's four pricing-model families (flat rate, per-seat, tiered, usage-based) and the forms of usage-based pricing. Added beyond the source map.

### S56 · official · Stripe. Set up tiered pricing (Docs). Living page as accessed. https://docs.stripe.com/subscriptions/pricing-models/tiered-pricing. Accessed 2026-10-27.
https://docs.stripe.com/subscriptions/pricing-models/tiered-pricing · accessed 2026-10-27 · fetched true · cited by 06#22
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: Stripe. Set up tiered pricing (Docs). Living page as accessed. https://docs.stripe.com/subscriptions/pricing-models/tiered-pricing. Accessed 2026-10-27.
Used for: volume-based versus graduated pricing and the 7 / 6.5 / 6 USD fonts example (Figure 8). Added beyond the source map.

### S57 · official · Linear. Pricing. Living page as accessed. https://linear.app/pricing. Accessed 2026-10-26.
https://linear.app/pricing · accessed 2026-10-26 · fetched true · cited by 06#23
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: Linear. Pricing. Living page as accessed. https://linear.app/pricing. Accessed 2026-10-26.
Used for: plan prices per user per month, the free-tier limits, the Enterprise governance features, and the "Requires AI credits" footnote (Table 9). Added beyond the source map.

### S58 · blog · Dunford, A. A Quickstart Guide to Positioning. aprildunford.com, 15 March 2021. https://www.aprildunford.com/post/a-quickstart-guide-to-positioning. Accessed 20
https://www.aprildunford.com/post/a-quickstart-guide-to-positioning · accessed 2026-10-27 · fetched true · cited by 06#25
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: Dunford, A. A Quickstart Guide to Positioning. aprildunford.com, 15 March 2021. https://www.aprildunford.com/post/a-quickstart-guide-to-positioning. Accessed 2026-10-27.
Used for: the definition, the five components in order, the phantom-competitor warning, the price-assumption sentence, and the embeddable-database anecdote. The author's own summary of the book; added beyond the source map.

### S59 · textbook · Cagan, M. Inspired: How To Create Products Customers Love, first edition. SVPG Press, 18 June 2008 (Open Library catalog record with table of contents). https:/
https://openlibrary.org/books/OL25426571M/Inspired · accessed 2026-10-27 · fetched true · cited by 06#26
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use. FIRST EDITION (2008) catalog record only; do not cite as second-edition content.
Citation: Cagan, M. Inspired: How To Create Products Customers Love, first edition. SVPG Press, 18 June 2008 (Open Library catalog record with table of contents). https://openlibrary.org/books/OL25426571M/Inspired. Accessed 2026-10-27.
Used for: the heading "Chapter 39. Keys to Platform Products" only. The second edition (Wiley, 2017) was not fetched, so the chapter's content and its survival are unknown.

### S60 · official · Stripe. Basic usage-based billing (Docs). Living page as accessed. https://docs.stripe.com/billing/subscriptions/usage-based. Accessed 2026-10-27.
https://docs.stripe.com/billing/subscriptions/usage-based · accessed 2026-10-27 · fetched true · cited by 06#27
Added at assembly (2026-10-27): cited by the module pages listed in citedBy, absent from the research map. Fetched by the writer of those pages; the page's reference entry gives the use.
Citation: Stripe. Basic usage-based billing (Docs). Living page as accessed. https://docs.stripe.com/billing/subscriptions/usage-based. Accessed 2026-10-27.
Used for: the statement that Metronome is now part of Stripe and is the recommended starting point for new usage-based integrations. Added beyond the source map.

## Corrections at assembly (2026-10-27)

- S20: pricing-owner claim corrected (PostHog role page names the PM; S20 names none).
- S20: demand-curve sentence re-attributed (it is in Spolsky 2004, not Strategy Letter V / S25).
- S4: never uses the word 'strategy'; the strategy coverage tag is a writer's mapping (noted, tag kept).
- Pricing disagreement, position 3: the demand-curve clause was attributed to S25. Strategy Letter V has no occurrence of demand curve. The clause was removed from S25 position and recorded in S25 conflicts; the sentence is in S48 (Spolsky 2004).

## Map entries cited by the module pages (cross-check 2026-10-27)

- Cited by at least one module page: 30 of 31 mapped sources.
- Mapped but not cited by any page: S27.
- URL variants merged onto a mapped source (same page, different URL form): S24 <- https://www.deeplearning.ai/the-batch/issue-284/; S12 <- https://rework.withgoogle.com/intl/en/guides/set-goals-with-okrs/; S20 <- https://posthog.com/handbook/engineering/feature-pricing.


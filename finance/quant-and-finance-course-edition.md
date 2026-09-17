# The Course Edition: University Courses for Every Box

Companion to the main curriculum. Same 26 boxes, same phase order — but each mapped to a specific university course (MIT OCW, Stanford, Harvard, Yale, NPTEL/IIT, and a few audit-free Coursera courses) instead of a book. Use this as the *spine*; keep the books as reference for when a lecture leaves you wanting depth.

---

## How to use this, and one honest warning

**One course per box, on purpose.** Same discipline as the book list — a menu of five courses per topic is the breadth trap wearing a graduation gown.

**The uncomfortable truth about course coverage:** the technical spine and the classical finance-theory boxes have *world-class* free courses. But the *modern, edge-bearing* boxes — market microstructure, backtest overfitting, statistical arbitrage, position sizing — have **no good free university course anywhere.** That's not an oversight in my search; it's the actual state of the world. Universities teach the theory that's settled and publishable; they don't teach the stuff that's either too practitioner-specific (microstructure execution) or too new and adversarial (overfitting defense). So for those boxes I'm telling you plainly: *the book stays, there's no course to swap in.* Anyone who hands you a tidy course for "how not to overfit a backtest" is selling something.

**NPTEL note:** IIT courses re-run every semester with new codes (noc25\_, noc26\_…). If a link is dead, search the course *title* on nptel.ac.in — the course still exists, just under a new instance. NPTEL's paid certificate is optional; all lectures are free on YouTube.

**Damodaran note:** Aswath Damodaran (NYU Stern) posts his *entire* semester courses free on YouTube — Valuation, Corporate Finance, and Investment Philosophies. For the Buffett-craft side of your goal, these three are the single best free resource on earth, and they directly serve your stated aim of *understanding* rather than line-drawing. Lean on them heavily.

---

# PHASE 0 — Foundations

### Track A — Math

**Linear algebra** → **MIT 18.06 — Linear Algebra** (Gilbert Strang), MIT OCW.
The canonical linear algebra course, full video lectures + problem sets. Nothing beats it.
`ocw.mit.edu/courses/18-06-linear-algebra-spring-2010`

**Calculus (multivariable)** → **MIT 18.02 — Multivariable Calculus**, MIT OCW.
Gradients, multiple integrals — the calculus you actually need downstream. (18.01 first if single-variable is rusty.)
`ocw.mit.edu/courses/18-02-multivariable-calculus-fall-2007`

**Probability & statistics** → **Harvard STAT 110 — Introduction to Probability** (Joe Blitzstein), free on YouTube + edX.
The best probability course available anywhere, and it's free. Pair with **MIT 18.05 — Introduction to Probability and Statistics** for the stats/inference half.
`projects.iq.harvard.edu/stat110` (lectures linked there)

### Track B — Literacy

**Financial accounting** → **Wharton / Coursera — Introduction to Financial Accounting** (Brian Bushee).
Legendary teacher; builds the three statements from scratch. Audit-free.
Find on Coursera under the title above.

**Corporate finance** → **Damodaran (NYU) — Corporate Finance**, free full semester on YouTube.
The definitive treatment, free. Covers capital structure, cost of capital, capital allocation.
Search "Aswath Damodaran Corporate Finance" on YouTube (he posts the full class).

**The financial system / plumbing** → **Columbia / Coursera — Economics of Money and Banking** (Perry Mehrling).
The single best explanation of the plumbing — repo, collateral, dealers, central banks. Audit-free.
Find on Coursera under the title above.

---

# PHASE 1 — Core

### Technical

**Optimization (convex)** → **Stanford EE364A — Convex Optimization** (Stephen Boyd), Stanford Online / YouTube.
Boyd's course *is* the standard. Full lectures + the free Boyd–Vandenberghe book as notes.
`see.stanford.edu` / search "Stanford EE364A Boyd".

**Econometrics + financial time series** → **NPTEL — Applied Econometrics** (IIT Madras, Sabuj Kumar Mandal) → **NPTEL — Applied Time-Series Analysis** (IIT Madras, Arun Tangirala).
The econometrics course builds regression → IV → the modeling toolkit. Tangirala's time-series course is genuinely excellent — stationarity, ARIMA/SARIMA, spectral analysis, estimation, with R throughout. For the finance-specific angle (volatility/GARCH, ML-in-time-series), **NPTEL — Time Series Modelling and Forecasting with Applications in R** (IIT Bombay, Sudeep Bapat) explicitly covers volatility modelling and stock-price forecasting.
`nptel.ac.in/courses/110106165` (Applied Econometrics) · `nptel.ac.in/courses/103106123` (Applied Time-Series)

### Literacy

**Valuation** → **Damodaran (NYU) — Valuation**, free full semester on YouTube.
The definitive valuation course on the planet, free. DCF, multiples, the works.
Search "Aswath Damodaran Valuation" on YouTube.

**Value / fundamental investing** → **Damodaran (NYU) — Investment Philosophies**, free on YouTube.
The honest pick. There is *no* rigorous free course that "teaches you to be Buffett" — that's book-and-letters territory (Graham + the Buffett shareholder letters). But Damodaran's Investment Philosophies course is the closest structured thing: it dissects value investing, *and* dismantles the technical-analysis line-drawing you explicitly want to avoid, with data. Watch it, then read the letters.

**Behavioral finance** → **Duke / Coursera — A Beginner's Guide to Irrational Behavior** (Dan Ariely).
Accessible, rigorous-enough intro to the biases. Audit-free. (Shiller's Yale course below also covers this.)

**Fixed income & rates** → **MIT 15.401 — Finance Theory I** (Andrew Lo), MIT OCW — *fixed-income lectures.*
Lo's course covers PV relations, fixed income, yield curves and bootstrapping directly. This same course anchors several Phase-1/3 boxes (see below) — one course, high coverage.
`ocw.mit.edu/courses/15-401-finance-theory-i-fall-2008`
*(Depth follow-up: no great free dedicated fixed-income course exists — Tuckman's book stays for duration/convexity depth.)*

---

# PHASE 2 — The pricing spine

**The anchor course for this whole phase:** **MIT 18.S096 / 18.642 — Topics in Mathematics with Applications in Finance**, MIT OCW. MIT mathematicians teach the math, industry professionals teach the application, and it walks straight through linear algebra → probability → stochastic processes → **Itô calculus** → regression → **deriving Black-Scholes** → VaR → yield curves. It is almost purpose-built for you.
`ocw.mit.edu/courses/18-642-topics-in-mathematics-with-applications-in-finance-fall-2024` (Fall 2024, with lecture videos) — the older Fall 2013 18.S096 version is also fully online.

**Measure-theoretic probability** → *(compress, as flagged in the main doc)*. No finance course teaches this; it's pure-math (MIT 18.100 Real Analysis is the prerequisite path). Absorb the finance-relevant concepts — filtrations, martingales, conditional expectation — directly from the stochastic-calculus lectures below rather than doing a full measure-theory course. This is the one box where "course instead of book" doesn't really apply.

**Stochastic calculus** → **MIT 18.642** (Itô calculus lectures, above) → **NPTEL — Mathematical Finance** (IIT Guwahati, N. Selvaraju).
Selvaraju's course explicitly covers stochastic calculus and portfolio theory for the Mathematics-and-Computing track — a rigorous, dedicated treatment. Between these two you get the engine without buying Shreve (though Shreve stays as the reference when a proof bites).
`nptel.ac.in/courses/111105041` area — search "NPTEL Mathematical Finance Selvaraju".

**Derivative pricing** → **Columbia / Coursera — Financial Engineering and Risk Management** (Martin Haugh & Garud Iyengar) + **NPTEL — Financial Mathematics** (IIT Roorkee, P.K. Jha).
The Columbia specialization is the strongest structured pricing course-set: binomial → Black-Scholes → risk-neutral pricing → term structure → and into portfolio/risk. NPTEL's Financial Mathematics covers the pricing math with an Indian-institute rigor. And 18.642 has you *derive* Black-Scholes yourself.
Columbia: find "Financial Engineering and Risk Management" on Coursera (audit-free). NPTEL: `nptel.ac.in/courses/112107260`.

**Derivatives (usage side)** → covered within the Columbia course above for strategy/hedging. *(Natenberg's book stays for the trader's vol-surface intuition — no course replaces it.)*

**Alternatives / VC** → **Techstars / Kauffman — Venture Deals** (free online course by Brad Feld & Jason Mendelson, the book's authors).
Directly useful to you as a founder — term sheets, dilution, economics vs. control. Runs periodically free online; the book is the fallback. *(PE/HF/real-estate: no strong free course; treat as light reading.)*

---

# PHASE 3 — Modern quant + reality + environment

### Technical

**Machine learning** → **Stanford CS229 — Machine Learning** (Andrew Ng), free full course on YouTube (Autumn 2018).
The course that defined how ML is taught — mathematically demanding, derivation-first, exactly your taste. 27+ hours, full lecture notes free.
`see.stanford.edu/Course/CS229` · YouTube: "Stanford CS229 Andrew Ng full course".
Indian alternative: **NPTEL — Introduction to Machine Learning** (IIT Madras, Balaraman Ravindran) — one of NPTEL's flagship courses, 60k+ enrolled.

> **The finance-specific ML rigor** — purged/embargoed cross-validation, meta-labeling — has **no course.** López de Prado's *Advances in Financial Machine Learning* stays. This is the single most important "book, not course" swap in the whole document. There is no MIT lecture on how financial ML leaks the future into the past; you learn it from that book and by getting burned.

**Market microstructure** → **⚠ No good free course exists.** This is the starkest gap. Universities barely teach it and no quality MOOC covers order books, market impact, and adverse selection properly. **Larry Harris's *Trading and Exchanges* stays as the primary** — treat it as the "course." Supplement with scattered exchange/quant-firm lecture talks on YouTube, but there's no structured substitute. Don't skip the box just because there's no video to press play on; this is mandatory before any strategy work.

**Data integrity** → no course (it's a discipline). Absorbed through the López de Prado book + doing the capstone.

### Literacy

**Macroeconomics & monetary policy** → **MIT 14.02 — Principles of Macroeconomics**, MIT OCW, + Mehrling's course (Phase 0) for the monetary/central-bank side.
For the markets-facing view of macro, **Yale ECON 252** (below) is excellent.

**Financial history & crises** → **Yale ECON 252 — Financial Markets** (Robert Shiller), Open Yale Courses (free).
A Nobel laureate walking through bubbles, crises, institutions, and behavioral finance. Covers the history/crisis box *and* doubles as behavioral finance. *(When Genius Failed — the LTCM/quant-hubris book — still stays; read it regardless. No course tells that story as well.)*
`oyc.yale.edu/economics/econ-252-11`

---

# PHASE 4 — Synthesis + the rigor frontier

**Here's where course coverage collapses — and that's the point.** The convergence zone is where practitioner knowledge lives, and academia mostly doesn't teach it. Read this section as "mostly books, a few courses."

**Backtest overfitting & multiple testing** → **⚠ No course. Anywhere.** López de Prado's overfitting chapters + his deflated-Sharpe papers are the only real source. I said it in the main doc and I'll say it again: this is *the* most important box, and the fact that no university packages it is exactly why most people never learn it and blow up. The absence of a course here is a feature of your edge, not a gap in your plan.

**Factor models & cross-sectional prediction** → **EDHEC / Coursera — Investment Management with Python and Machine Learning** specialization.
The best structured free-to-audit course on factor investing and portfolio construction, taught quantitatively with code. Pairs with **MIT 15.401**'s CAPM/APT lectures for the theory foundation.
Find "EDHEC Investment Management with Python and Machine Learning" on Coursera.

**Statistical arbitrage** → **⚠ No good free course.** Ernest Chan's *Algorithmic Trading* stays as the practical guide. Some paid quant bootcamps exist; nothing free and rigorous.

**Portfolio optimization** → **Columbia FE&RM** (Phase 2, part 2 covers portfolio) + **EDHEC specialization** (above).
Mean-variance, Black-Litterman, risk budgeting — both cover it with code.

**Risk management** → **Columbia FE&RM** covers VaR/CVaR and risk directly; sufficient as the course. *(Hull's risk book stays for institutional depth.)*

**Position sizing & money management** → **⚠ No course.** Kelly criterion, risk parity — practitioner territory. Thorp's writing + the Chan/Grinold-Kahn books. Small box, but no video for it.

**Portfolio management & asset allocation (literacy capstone)** → **EDHEC specialization** ties it together.

---

# PHASE 5 — Capstone

No course. It never was one. Pick one strategy, get data, build the backtester with real costs, fail, diagnose through the theory. This is the box that turns every course above into skill — and the one your instinct will most want to replace with "one more course." Don't.

---

## Master table — the course spine

| Box | Course | Provider | Free? |
|-----|--------|----------|-------|
| Linear algebra | 18.06 Linear Algebra (Strang) | MIT OCW | ✅ |
| Calculus | 18.02 Multivariable Calculus | MIT OCW | ✅ |
| Probability & stats | STAT 110 (Blitzstein) + 18.05 | Harvard / MIT | ✅ |
| Accounting | Introduction to Financial Accounting (Bushee) | Wharton/Coursera | ✅ audit |
| Corporate finance | Corporate Finance (Damodaran) | NYU / YouTube | ✅ |
| Financial system | Economics of Money & Banking (Mehrling) | Columbia/Coursera | ✅ audit |
| Optimization | EE364A Convex Optimization (Boyd) | Stanford | ✅ |
| Econometrics / time series | Applied Econometrics + Applied Time-Series | NPTEL (IIT-M) | ✅ |
| Valuation | Valuation (Damodaran) | NYU / YouTube | ✅ |
| Value investing | Investment Philosophies (Damodaran) | NYU / YouTube | ✅ |
| Behavioral finance | Irrational Behavior (Ariely) / ECON 252 | Duke / Yale | ✅ |
| Fixed income & rates | 15.401 Finance Theory I (Lo) | MIT OCW | ✅ |
| Math+finance anchor | 18.S096 / 18.642 Applications in Finance | MIT OCW | ✅ |
| Stochastic calculus | 18.642 + Mathematical Finance (Selvaraju) | MIT / NPTEL | ✅ |
| Derivative pricing | Financial Eng. & Risk Mgmt (Haugh/Iyengar) + Financial Mathematics | Columbia / NPTEL | ✅ audit |
| Alternatives / VC | Venture Deals (Feld/Mendelson) | Techstars | ✅ |
| Machine learning | CS229 (Ng) / Intro to ML (Ravindran) | Stanford / NPTEL | ✅ |
| **Financial ML rigor** | **— (López de Prado book)** | **none** | 📕 |
| **Market microstructure** | **— (Harris book)** | **none** | 📕 |
| Macro & monetary | 14.02 Macro + Money & Banking | MIT / Columbia | ✅ |
| Financial history | ECON 252 Financial Markets (Shiller) | Yale (OYC) | ✅ |
| **Backtest overfitting** | **— (López de Prado)** | **none** | 📕 |
| Factor models | Investment Mgmt w/ Python & ML | EDHEC/Coursera | ✅ audit |
| **Statistical arbitrage** | **— (Chan book)** | **none** | 📕 |
| Portfolio optimization | FE&RM pt.2 + EDHEC | Columbia / EDHEC | ✅ audit |
| Risk management | FE&RM (VaR/CVaR) | Columbia | ✅ audit |
| **Position sizing** | **— (Thorp / Chan)** | **none** | 📕 |

**✅ = free full lectures. audit = free to audit on Coursera (pay only for certificate). 📕 = no quality course exists; book is primary.**

---

## The one-sentence read on all this

The classical spine — math, pricing, econometrics, valuation, finance theory — is *fully covered* by free world-class courses, and you should absolutely learn it that way. But notice the pattern in the 📕 rows: **every box with no course is a box on the edge-bearing frontier.** Microstructure, overfitting, stat arb, sizing. That's not a coincidence — it's the map telling you where the real, non-commoditized knowledge lives. The courses make you literate. The four books in the 📕 rows, plus the capstone, are what make you dangerous.

Start with 18.06 and Damodaran's Valuation this week. Press play.

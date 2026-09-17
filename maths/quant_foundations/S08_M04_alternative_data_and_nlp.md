# Subject 8, Module 4: Alternative Data and NLP for Finance

*Mathematical Foundations for Quantitative Research: From JEE to Jane Street*

> *"Data is the new oil — but crude. You have to refine it, ship it, and mind the leaks."* — an unnamed CTO

---

## 8.4.0 Where We Are

Alternative data has exploded since 2010: satellite images, credit card panels, shipping manifests, app downloads, social media, geolocation. The challenge is not acquiring data but extracting a *replicable*, *causal*, and *timely* signal from it. This module surveys the quantitative techniques.

### Prerequisites

- **Module 7.1–7.3** (regression, classification, boosting).
- **Module 7.7** (causal inference).
- **Module 8.3** (backtesting discipline).
- Familiarity with basic NLP ideas (token, vocabulary) is helpful.

### Plan

1. Alternative data types and life cycles (§8.4.1).
2. Data quality, survivorship, and selection bias (§8.4.2).
3. Text representation: from BoW to transformers (§8.4.3).
4. Financial NLP tasks: sentiment, topic, event extraction (§8.4.4).
5. Large language models in quant research (§8.4.5).
6. Satellite and geospatial imagery (§8.4.6).
7. Credit-card, web-traffic, and consumer panels (§8.4.7).
8. Alt-data evaluation: predictive power, orthogonality, capacity (§8.4.8).
9. Causal identification with alt data (§8.4.9).
10. Python: text-based signals (§8.4.10).
11. Applications (§8.4.11).
12. Exercises (§8.4.12).

---

## 8.4.1 Alternative Data Types and Life Cycles

### Major categories

- **Text**: earnings transcripts, news, social media, filings (10-K/10-Q/8-K), research reports.
- **Satellite imagery**: parking-lot counts, oil-tank levels, crop health, construction, shipping.
- **Web-scraped**: app store rankings, pricing data, job postings, hiring.
- **Transaction data**: credit card panels, online checkout, B2B transactions.
- **Geolocation**: phone-based foot-traffic to specific venues.
- **IoT / sensors**: utility consumption, traffic cameras.
- **Scientific/biomedical**: clinical trial reports, FDA filings.

### Alt-data life cycle

1. **Novelty (year 0–1)**: dataset new, few funds using it. Signal high; capacity small; noise very high.
2. **Adoption (year 1–3)**: word spreads; data vendor builds productized feed; handful of funds extract alpha.
3. **Maturity (year 3–5)**: many funds using it; signal erodes as flow competes; need domain-specific edges.
4. **Commoditization**: widely available in standardized form; alpha margin thin; requires idiosyncratic enrichment or causal identification to remain usable.

This cycle shapes the research investment decision: a fund must enter early to capture alpha, but early data is the messiest and requires the most capital to clean.

---

## 8.4.2 Data Quality Pitfalls

### Point-in-time (PIT) accuracy

Data must reflect what was knowable at the historical date — not the vendor's current snapshot. Critical for:
- Fundamental data (EPS, sales): initial reports vs. restatements.
- Corporate actions: splits, dividends, delistings applied at the correct moment.
- Index constituents: historical S&P 500 membership with effective dates.
- Vendor quirks: backfill, restated, or retroactively extended series.

### Survivorship

Datasets derived from current entities exclude:
- Bankrupt firms.
- Acquired subsidiaries.
- Delisted tickers.
- Failed apps/services.

Always use *full historical roster including failures*.

### Selection bias

Vendors sampling the data may have non-random coverage:
- Credit-card panels over-represent higher-income households.
- App-download panels may exclude international.
- Satellite may prefer cloudless regions.

Characterize the sample vs. population; adjust if possible.

### Leakage

Timing mistakes — e.g., satellite imagery reported with a lag that's smaller in the backtest than in reality. Often the backtest assumes "same-day availability" while the vendor actually provides the signal 48 hours later.

### Signal lifespan

Alt-data signals decay fast once more funds use them. Monitor IC monthly; refresh the data contract when marginal alpha drops below cost.

---

## 8.4.3 Text Representation

### Bag of Words and TF-IDF

$$\mathrm{tfidf}(w, d) = \mathrm{tf}(w, d) \cdot \log\frac{N}{df(w)}.$$
A document becomes a sparse $|V|$-vector. Good baseline; fast; interpretable.

### N-grams

Capture local context: "earnings per share" as a single token. Quickly explodes vocabulary; pruned by document frequency.

### Word embeddings

Dense vectors capturing semantic similarity.
- **Word2Vec** (Mikolov et al. 2013): skip-gram or CBOW objectives; softmax over context words.
- **GloVe** (Pennington et al. 2014): factorize co-occurrence matrix.
- **FastText** (Bojanowski et al. 2017): subword embeddings help with OOV.

Training objective (skip-gram, negative sampling):
$$\mathcal{L} = \sum_{(w,c) \in D_+} \log\sigma(v_c^\top v_w) + \sum_{(w,c) \in D_-}\log\sigma(-v_c^\top v_w).$$

### Transformers and contextual embeddings

**BERT** (Devlin et al. 2018): bidirectional encoder; pretrained with masked language modeling. Produces contextual embeddings: same word has different vectors in different sentences.

Attention mechanism:
$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.$$
Multi-head attention stacks $h$ parallel heads with different projections. Transformer encoder: self-attention + feed-forward + residuals + layer-norm.

### Domain-specific models

- **FinBERT** (Yang–Huang 2020): BERT fine-tuned on financial text.
- **BloombergGPT** (Wu et al. 2023): 50B-parameter LLM trained on 700B tokens of financial text.
- **FLANG** (Shah et al. 2022): family of domain-adapted BERT variants.
- **Llama-based models** fine-tuned on SEC filings, earnings transcripts, etc.

---

## 8.4.4 Financial NLP Tasks

### Sentiment analysis

Determine whether text is positive/negative/neutral about an entity:
- **Dictionary approach** (Loughran–McDonald 2011): finance-specific lexicons with positive/negative/litigious/uncertainty/constraining wordlists.
- **Supervised classifier**: labeled training set + logistic / LSTM / transformer.
- **Zero-shot via LLM**: prompt a model to classify.

Research has linked negative-word frequency in 10-K filings to forward abnormal returns (Loughran–McDonald 2011, Feldman–Govindaraj–Livnat 2010). Twitter/StockTwits sentiment generates stat-arb signals; decay fast.

### Topic modeling

Latent Dirichlet Allocation (Blei–Ng–Jordan 2003): documents mixtures of latent topics, topics mixtures of words. EM or variational Bayes. Applications:
- Industry classification from 10-K business descriptions.
- Earnings-call topic shifts across quarters.
- FOMC minutes topic evolution.

### Event extraction

Structured information from unstructured text: "Company X is acquiring Company Y for $Z billion." Named-entity recognition + relation extraction. Increasingly solved with LLM prompting ("extract all M&A events and return as JSON").

### Readability and complexity

Longer, more complex 10-Ks associated with worse future performance (Li 2008). Gunning-Fog index, Flesch-Kincaid, average sentence length.

### Similarity metrics

Document-to-document similarity for:
- Detecting boilerplate ("copy-paste" 10-K language).
- Peer-company identification beyond GICS.
- Tracking narrative convergence across analysts (Cohen–Malloy–Nguyen 2020: "Lazy Prices").

---

## 8.4.5 Large Language Models in Quant

### Use cases

1. **Research assistance**: literature review, idea summarization, code generation.
2. **Extraction**: parse structured info from filings, contracts, transcripts.
3. **Classification**: zero-shot sentiment, topic, event classification.
4. **Summarization**: of analyst reports, earnings calls, news.
5. **Agents**: chain-of-thought reasoning for multi-step queries (e.g., "find all biotech firms with Phase-3 trial readouts next quarter").
6. **Signal generation**: embeddings + classifier on LLM-processed text.

### Practical constraints

- **Latency**: transformer inference can take seconds per document; batching essential.
- **Cost**: API fees scale with tokens; large filings (100k+ tokens) need chunking.
- **Reproducibility**: proprietary APIs may change silently; for production signals prefer open-weight or self-hosted models.
- **Hallucination**: LLMs invent facts. Never use LLM-generated "facts" as signal inputs without verification.
- **Look-ahead leakage**: pretrained models have seen much of the internet post-date; use only models whose training cutoff predates the signal period (or use strictly extractive pipelines).

### Signal validation for LLM features

An LLM-derived feature (e.g., sentiment score) must pass the same backtesting discipline: cross-sectional IC, orthogonality to known factors, turnover analysis, capacity estimate. "LLM said so" is not validation.

---

## 8.4.6 Satellite and Geospatial Imagery

### Typical pipeline

1. Acquire raw imagery from providers (Planet, Maxar, ESA Sentinel).
2. Geo-reference and cloud-mask.
3. Detect objects (cars, ships, storage tanks) via CNN classifiers (YOLO, Mask R-CNN).
4. Aggregate counts over time series at specific sites (Walmart parking lots, oil tanks, container terminals).
5. Normalize (seasonality, weather, cloud cover) and align to company-level fundamentals.

### Example signals

- **Parking lot counts** predicting retail quarterly sales.
- **Oil storage levels** informing WTI/Brent positioning.
- **Shipping container throughput** for global trade indicators.
- **Crop health (NDVI)** for grain prices.
- **Construction progress** for emerging-market infrastructure.

### Challenges

- **Seasonality**: parking lots vary by holiday, weather.
- **Site selection bias**: which stores are sampled vs. the full chain.
- **Panel consistency**: satellite revisit rate and cloud cover.
- **Survivorship**: closed stores and new openings must be tracked manually.
- **Regulatory**: data licensing, personal-information concerns.

---

## 8.4.7 Credit Card, Web Traffic, Consumer Panels

### Credit card

Aggregated card-transaction data (Yodlee, Earnest, Envestnet) offer near-real-time view of corporate revenue. Key metrics: year-over-year growth, weekly velocity, penetration (card vs. total revenue).

### Web traffic

App store rankings, daily-active-users proxies (Sensor Tower, SimilarWeb). Predictive for internet stocks in early years; heavily competed now.

### Physical foot traffic

Phone geolocation data (e.g., SafeGraph) with differential privacy processing. Predicts retail same-store sales, restaurant revenue.

### Hiring data

Job postings (LinkedIn, Indeed). Posting intensity correlates with planned capex and growth. Analyzed by Chen–Kogan–Papanikolaou (2020s), showing links between hiring and forward earnings.

### Methodology

- **Triangulate**: never rely on one dataset; combine two or three that agree for confirmation.
- **Nowcast**: regress reported company sales on alt-data panel with quarterly cadence, then use the regression to predict the current quarter.
- **Residual alpha**: subtract consensus expectations before using alt-data-derived estimates as signals.

---

## 8.4.8 Alt-Data Evaluation

### Predictive power

- Cross-sectional IC vs. forward returns.
- Coefficient in panel regression with factor controls.
- Earnings-surprise prediction R².

### Orthogonality

Regress the new signal on existing signals; residual $R^2 > 0.8$ is a good sign. Signals that are mostly orthogonal are more valuable than redundant ones.

### Capacity

Given a new signal, compute the universe's dollar cross-sectional spread and the implied capacity at target participation rates.

### Breakeven cost threshold

Signal must generate alpha exceeding the marginal cost of data + refresh research.

### Decay analysis

Hold out post-purchase data; measure IC trajectory; retire signal when 6-month rolling IC crosses zero.

---

## 8.4.9 Causal Identification with Alt Data

Alt data often *reveals* structural shocks, enabling causal identification.

### Shift-share instruments

A local shock (e.g., change in regional foot traffic) combined with a firm's geographic exposure shares creates an instrument for firm-level outcomes.

### Natural experiments

COVID lockdowns, weather events, regional policy shifts — identified from alt data — support DiD designs (Module 7.7).

### Panel FE with alt-data shocks

Firm × time fixed effects absorb macro and firm-specific trends; alt-data residual variation identifies firm-level responses.

### Pitfalls

- **Reverse causality**: alt data may reflect anticipation of the same events it purports to predict.
- **Correlated measurement error**: vendors have similar biases; triangulation with independent sources essential.
- **Confounders**: the variable driving alt-data changes may also directly drive returns (e.g., commodity prices driving both oil storage and energy-stock returns).

---

## 8.4.10 Python: Text-Based Signal

```python
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

# ----- Loughran-McDonald sentiment -----
# Simplified dictionaries (real LM list is several thousand words each)
LM_NEG = set(["loss", "losses", "declined", "impairment", "adverse", "litigation",
              "bankruptcy", "default", "downturn", "restatement", "weakness"])
LM_POS = set(["growth", "gains", "strong", "achievement", "profit", "profitable",
              "favorable", "improved", "success"])

def lm_sentiment(text):
    tokens = re.findall(r"[a-z]+", text.lower())
    n_pos = sum(w in LM_POS for w in tokens)
    n_neg = sum(w in LM_NEG for w in tokens)
    n = len(tokens)
    if n == 0:
        return 0.0
    return (n_pos - n_neg) / n  # normalized tone

# ----- TF-IDF + logistic sentiment -----
def train_tfidf_sentiment(texts, labels, max_features=10000):
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, min_df=3)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, labels)
    return vec, clf

def tfidf_score(vec, clf, texts):
    X = vec.transform(texts)
    return clf.predict_proba(X)[:, 1] - 0.5  # centered score

# ----- Event study around earnings -----
def event_study(returns, events, window=(-5, 5)):
    """
    returns: date x ticker DataFrame of returns
    events: DataFrame with columns ['ticker', 'date', 'surprise']
    """
    aligned = []
    for _, row in events.iterrows():
        t = row['ticker']; d = row['date']
        idx = returns.index.get_indexer([d], method='nearest')[0]
        lo, hi = max(0, idx + window[0]), min(len(returns), idx + window[1] + 1)
        seg = returns[t].iloc[lo:hi].reset_index(drop=True)
        # Pad if window hits boundary
        out = np.full(window[1]-window[0]+1, np.nan)
        offset = (window[0] if lo == idx + window[0] else lo - idx)
        out[-len(seg):] = seg.values
        aligned.append((row['surprise'], out))
    high = np.nanmean([o[1] for o in aligned if o[0] > 0], axis=0)
    low = np.nanmean([o[1] for o in aligned if o[0] < 0], axis=0)
    return high, low

# ----- Simple earnings-call nowcast -----
def nowcast_earnings(alt_feature, realized_earnings, n_windows=5):
    """
    Rolling regression of realized earnings on alt-data feature.
    Returns predicted earnings for each period.
    """
    tscv = TimeSeriesSplit(n_splits=n_windows)
    preds = np.full_like(realized_earnings, np.nan, dtype=float)
    for train, test in tscv.split(alt_feature):
        A = alt_feature[train].reshape(-1, 1)
        y = realized_earnings[train]
        beta = np.linalg.lstsq(np.hstack([np.ones((len(A),1)), A]), y, rcond=None)[0]
        preds[test] = beta[0] + beta[1] * alt_feature[test]
    return preds

# ----- Example: LM sentiment on synthetic texts -----
texts = [
    "The company reported strong growth and favorable profit margins.",
    "Ongoing litigation and severe losses from the adverse quarter.",
    "Results were mixed with both gains and impairment charges.",
]
for t in texts:
    print(f"{t[:50]}... : sentiment={lm_sentiment(t):+.4f}")
```

---

## 8.4.11 Applications

1. **Earnings-call sentiment**: real-time scoring of live transcripts; aggressive-delivery tone associated with post-call returns.
2. **10-K complexity**: long, vague filings predict negative forward returns.
3. **Job-posting signal**: firms ramping hiring outperform in subsequent 6-12 months.
4. **Satellite retail**: mall-parking counts predict quarterly retailer beats/misses.
5. **Oil-tank monitoring**: feeds into WTI/Brent positioning and energy-stock PCA residuals.
6. **App-download signal**: early indicator for mobile-game and consumer-app revenue.
7. **Credit card consumer-panel**: near-real-time same-store-sales estimate for restaurants, retail.
8. **Twitter/Reddit sentiment**: short-horizon retail-flow indicator; capacity very small.
9. **ESG / controversy scraping**: reputation-risk signals from news.
10. **FDA-filing text**: drug-approval probability estimation from trial language.
11. **SEC 8-K scraping**: event-driven strategies (8-K item 1.01 = entry into definitive agreement).
12. **Analyst-report similarity**: detecting copy-paste in sell-side research as a contrarian signal.

---

## 8.4.12 Exercises

### ★

1. For a bag-of-words model with 10,000 tokens and 1,000 documents, compute the TF-IDF matrix dimension and sparsity.
2. Define the Loughran-McDonald positive and negative wordlists; why are they domain-specific rather than generic?
3. For a word embedding with 300 dimensions and 1M vocabulary, compute the number of parameters.
4. Derive the skip-gram with negative-sampling loss.
5. Identify three common selection biases in credit-card panel data.
6. Suppose a new alt-data signal has IC of 0.06 and existing signals have IC of 0.04. Under what conditions is the new signal worth acquiring?

### ★★

7. Train a TF-IDF + logistic regression classifier on a labeled corpus of earnings-call transcripts; evaluate with a 5-fold walk-forward split.
8. Implement an LDA topic model on 10-K filings and interpret the top topics.
9. Use a BERT embedding to compute document similarity and identify earnings calls with unusually high similarity to prior quarter's call (suggesting script reuse).
10. Design an event study around 8-K 1.01 filings; report CAR over $(-1, +5)$ days.
11. Given satellite car-count data for a retailer and quarterly same-store-sales, build a nowcast regression; evaluate R².
12. Simulate a two-period DiD design leveraging a geolocation shock and show the implied treatment effect is identified under parallel trends.

### ★★★

13. Prove that TF-IDF with cosine similarity is equivalent to a particular inner-product in a reweighted vector space, and identify the implicit kernel.
14. Derive the variational ELBO for LDA and show the fixed-point update equations.
15. Prove convergence of the skip-gram with negative sampling to a factorization of the shifted PMI matrix (Levy-Goldberg 2014).
16. Analyze the identification of an alt-data-driven shift-share instrument; derive consistency and inference properties.
17. Establish regret bounds for an adaptively updated alt-data signal under signal decay with unknown half-life.
18. Prove that a Bayesian combination of LLM sentiment and quantitative signal dominates each individually under explicit specification of prior uncertainties.

---

*— End of Module 8.4. Next: Module 8.5, Bayesian Methods and MCMC for Finance.*

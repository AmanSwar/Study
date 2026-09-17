# Subject 8, Module 7: Systems and MLOps for Quant Research

*Mathematical Foundations for Quantitative Research: From JEE to Jane Street*

> *"Research is asymptotic in its depth; production is asymptotic in its breadth. Great quants operate in both limits."* — desk aphorism

---

## 8.7.0 Where We Are — Subject 8 Capstone

Modules 8.1–8.6 built the analytical substrate of a modern quant desk; this capstone assembles the engineering infrastructure that turns research into production. A theoretically brilliant strategy that cannot be reliably deployed and monitored is academic. The difference between a paper PnL chart and a live PnL chart is almost entirely systems.

### Prerequisites

- **Module 8.3** (Algorithmic trading design): the pipeline view.
- **Module 8.1, 8.2** (Portfolio + risk): what flows through the system.
- General software engineering: version control, CI, containerization.

### Plan

1. The quant research stack (§8.7.1).
2. Data systems: raw → feature store → model features (§8.7.2).
3. Experiment tracking and reproducibility (§8.7.3).
4. Research-to-production hand-off (§8.7.4).
5. Backtesting infrastructure (§8.7.5).
6. Feature drift and model monitoring (§8.7.6).
7. Compute infrastructure: CPU, GPU, distributed (§8.7.7).
8. Scaling ML: training, inference, serving (§8.7.8).
9. Security, audit, and compliance logging (§8.7.9).
10. Case study: an end-to-end quant MLOps stack (§8.7.10).
11. Closing Subject 8 and Track A capstone summary (§8.7.11).
12. Exercises (§8.7.12).

---

## 8.7.1 The Quant Research Stack

A practical research-to-production environment has the following layers:

1. **Storage**: raw market data, reference data, alt data; tick-level to daily.
2. **Compute**: CPU/GPU clusters; interactive notebooks + batch pipelines.
3. **Feature computation**: transformations of raw data to signal inputs.
4. **Modeling**: library of model specifications (regressions, trees, NNs, factor models).
5. **Backtesting**: simulation engine with realistic costs and constraints.
6. **Production engine**: live execution of models against live data.
7. **Monitoring**: data-quality, model-drift, PnL, risk.

Modern quant firms either build proprietary stacks (Two Sigma's Beacon, D.E. Shaw's Maestro) or blend open-source (MLflow, Airflow, Dask) with commercial (Databricks, Snowflake) tooling.

---

## 8.7.2 Data Systems

### Raw data ingestion

Market data vendors (ICE, Refinitiv, S&P, Nasdaq TotalView, exchange direct feeds) publish feeds over multicast or TCP. Ingestion pipelines must:
- Tolerate feed gaps, sequence resets.
- Archive raw in addition to normalized format for replay.
- Normalize to a canonical schema: (venue, symbol, timestamp, event_type, price, size, sequence).
- Persist with immutable timestamps for audit.

### Point-in-time storage

Every mutation to any value (e.g., an EPS restatement, a corporate-action adjustment) must be stored with its *knowledge date* — when the fact became known to the market. Databases like Kx, Arctic (on MongoDB), or cloud data-lake formats (Delta Lake, Iceberg) support bitemporal data natively.

### Feature stores

Platforms where features are (a) defined once, (b) computed consistently for training and inference, (c) discoverable and reusable. Key properties:
- **Train/serve skew prevention**: the same code produces features in both modes.
- **Time-travel**: get feature values as of any historical timestamp (for backtesting).
- **Materialization**: precompute for latency-sensitive inference.
- **Lineage**: track which raw data and transformations produced each feature.

Open-source (Feast, Tecton's OSS variant) and commercial (Tecton, Databricks Feature Store) options exist. Many quant desks build in-house.

### Data versioning

DVC, LakeFS, or snapshot-based systems (Delta time travel) version datasets so a backtest can be reproduced bit-exactly months later. Critical for audit and for diagnosing live-vs-backtest discrepancies.

### Data quality SLAs

Each feed has a quality SLA: latency, completeness, schema conformance, value distributions (Kullback-Leibler between today and historical). Automated checks fail loudly on violations; production models roll to previous-day data rather than trade on bad feeds.

---

## 8.7.3 Experiment Tracking and Reproducibility

### Experiment metadata

Every backtest should log:
- Code commit SHA.
- Data snapshot version (or timestamp).
- Model hyperparameters.
- Random seeds.
- Compute environment (Docker image, Python version, package versions).
- Output artifacts (model weights, predictions, PnL series).

Tools: MLflow, Weights & Biases, Sacred, in-house systems.

### Reproducibility discipline

- **Deterministic seeds** on all RNG sources.
- **Pinned dependency versions** (poetry lock, requirements.txt with hashes).
- **Immutable data snapshots** (no mutable "latest" pointers).
- **Containerization** so the environment is bit-reproducible.

A backtest six months ago should reproduce to the last decimal digit today.

### Research hygiene

- Commit every experiment's config and results, even failures.
- Separate "exploratory" notebooks from production code.
- Code review for anything crossing the research-production boundary.
- Document the "why" behind parameter choices in structured metadata.

### PBO and data snooping tracking

Maintain a registry of every strategy variant evaluated on a dataset — the denominator of multiple-testing correction. Without this, the org cannot honestly assess the probability of backtest overfitting.

---

## 8.7.4 Research-to-Production Hand-Off

### Strategy lifecycle

1. **Ideation**: hypothesis, background, intuition.
2. **Prototype backtest**: quick/dirty validation.
3. **Rigorous backtest**: walk-forward, cost models, risk limits.
4. **Paper trade**: run live without real money.
5. **Limited real trading**: small size, monitored.
6. **Full production**: scaled up under normal risk controls.
7. **Maintenance / eventual retirement**.

Each transition requires a review (research lead, risk officer, infra lead). Reviews compare backtest expectations vs. paper-trade vs. small-money reality and sign off on proceeding.

### Research → production specification

A production deployment requires a machine-readable specification:
- Inputs: feature list with lineage.
- Parameters: fixed weights/coefficients/model binary.
- Operational constraints: trading hours, size limits, instrument universe.
- Risk controls: explicit position and exposure bounds.
- Monitoring: metrics to alarm on.

The specification is the contract between research and production.

### Shadow mode

Before any strategy takes real positions, run it in "shadow" — receiving live data and generating proposed trades — without actually executing. Compare shadow trades to live-production trades to catch simulation-reality discrepancies (slippage surprises, latency issues).

### Canary deployment

Phased rollout: 1% of intended size for a day, 10% for a week, 100%. Abort automatically if PnL or risk deviates from projections.

---

## 8.7.5 Backtesting Infrastructure

### Vectorized vs. event-driven

- **Vectorized backtesting** (pandas, NumPy): fast, good for daily signal research, poor for intraday fills and liquidity.
- **Event-driven backtesting**: simulate each tick or order-book event; slow but necessary for HFT and execution research.

Modern stacks maintain both engines and run the same strategy through each — vectorized for rapid iteration, event-driven for pre-production validation.

### Cost simulation engines

A realistic simulator integrates:
- **Fill modeling**: probabilistic fills given quote aggressiveness and market state.
- **Market impact**: Almgren-Chriss or proprietary models.
- **Queue position**: track FIFO queue at each price level.
- **Rejections / cancellations**: modeled with historical rates.

### Distributed backtesting

A walk-forward backtest over many parameter combinations is embarrassingly parallel. Tools: Ray, Dask, Spark. At scale, $N = 10^5$ configurations × $T = 20$ years × 1000 assets demands careful orchestration.

### Result caching

Incremental backtesting caches intermediate results (computed features, rolling windows) to avoid recomputing unchanged pieces. Saves hours to days on iteration cycles.

### Deterministic mode

A "deterministic" mode where all stochastic elements (order fills, random tie-breaks) use reproducible seeds ensures backtests are bit-reproducible. A "stochastic" mode runs many Monte Carlo realizations to characterize variance.

---

## 8.7.6 Feature Drift and Model Monitoring

### Data drift

Feature distribution today differs from training — detectable by:
- **Population Stability Index (PSI)**: $\sum_i (p_i - q_i)\log(p_i/q_i)$.
- **KL divergence** on binned features.
- **Earth Mover's Distance** for continuous features.

Alerts on PSI > 0.2 or KL > threshold trigger investigation.

### Concept drift

Input-output relationship changes: a feature that predicted returns no longer does. Detected by:
- Rolling IC plots with confidence bands.
- Rolling model R² or accuracy.
- Residual autocorrelation (should be zero under a good model).

### PnL monitoring

- **Live vs. expected PnL**: compare realized to backtest projection; large deviations trigger investigation.
- **PnL attribution** decomposes realized returns to isolate the misbehaving component.
- **Peer comparison**: if similar strategies across the desk underperform simultaneously, probably a market regime shift rather than model issue.

### Slippage monitoring

- **Expected vs. realized slippage** per order, by size bucket.
- **Implementation shortfall** tracking vs. benchmark.
- **Cost-model recalibration** weekly based on realized cost.

### Incident response

When monitoring fires an alarm:
1. **Triage**: severity + strategy isolation.
2. **Pause** affected strategies; switch to safe-mode execution.
3. **Diagnose**: data feed, model state, market regime.
4. **Remediate**: fix data, retrain model, adjust parameters.
5. **Resume** cautiously with canary deployment.
6. **Post-mortem**: document root cause and prevention.

---

## 8.7.7 Compute Infrastructure

### CPU workloads

- Feature computation, backtesting, statistical inference.
- Tools: NumPy, Pandas, Polars, Rust/C++ for hot loops.
- Scheduling: Slurm, Kubernetes, or cloud-native (AWS Batch, GCP Batch).

### GPU workloads

- Deep learning training and inference.
- Fourier pricing on large-batch option grids.
- Monte Carlo on GPUs: embarrassingly parallel path simulation.
- Tools: CUDA, PyTorch, JAX. JAX particularly well-suited to quant because its `vmap`/`jit`/`grad` composition aligns with scientific computing idioms.

### Distributed computing

- **Dask** for parallel Pandas-like workflows.
- **Ray** for task-parallel Python with actor model; widely used for distributed RL and ML experiments.
- **Spark** for large-batch ETL, declining in quant as specialized tools arrive.

### Cloud vs. on-prem

Trade-offs:
- Cloud: elastic, fast provisioning, vendor lock-in, recurring cost.
- On-prem: fixed cost, control, data residency, ops burden.
- Hybrid: on-prem for low-latency trading; cloud for research.

### Low-latency paths

Production trading engines are typically written in C++ with lock-free data structures, kernel bypass networking, and dedicated cores pinned to NUMA regions. This is a different discipline from research code; most firms maintain separate teams for strategy research and execution engine development.

---

## 8.7.8 ML in Production

### Model serving

Options:
1. **Embed model weights in the trading engine** (TorchScript, ONNX Runtime): minimum latency, tight coupling.
2. **RPC to a model server** (TorchServe, Triton): loose coupling, higher latency.
3. **Custom serialized representation** for simple models: manually encode coefficients as floats for maximum speed.

Latency budget dictates the choice. HFT models must often be simple enough to hand-code (linear combinations, tree ensembles) for <10μs inference.

### Model training

- **Reproducible training runs**: same seed + data + code = same model.
- **Distributed training**: data-parallel via PyTorch DDP or Horovod; model-parallel for very large nets.
- **Hyperparameter search**: Optuna, Ray Tune; budgeted via successive halving / Hyperband.
- **Gradient accumulation** for large effective batch sizes on limited memory.

### Online learning

Some models update intra-day (e.g., adverse-selection classifiers). Considerations:
- **Forgetting**: exponential weighting or fixed-window retraining.
- **Catastrophic drift**: anchor updates to avoid drift on temporary regime changes.
- **Safety**: online updates gated by validation on held-out data; automatic rollback on divergence.

### Inference monitoring

Track:
- Prediction histograms (distribution shift).
- Latency percentiles (50th, 99th, 99.9th).
- Error rates / exceptions.
- Resource utilization.

---

## 8.7.9 Security, Audit, and Compliance

### Access controls

- **Authentication**: SSO, MFA.
- **Authorization**: principle of least privilege; role-based access to data and systems.
- **Audit**: every research and trading action logged with user, timestamp, resource.

### Data classification

- **Public** (market data): freely usable.
- **Confidential** (research IP, strategies): restricted sharing.
- **MNPI** (material non-public information): strictly compartmentalized; information barriers between research and trading desks for positions of interest.

### Regulatory recordkeeping

SEC Rule 17a-4 (broker-dealers), MiFID II (EU): retain trading records for 5+ years in write-once format. Applies to orders, communications, models, supporting data.

### Audit trail for models

Every production model must have a reproducible audit trail:
- Source code version.
- Training data version and timestamps.
- Hyperparameters and validation results.
- Approver sign-off records.

### Incident reporting

Serious incidents (large losses, compliance breaches, system failures) require reporting internally to risk/compliance and externally to regulators. Most desks maintain pre-written playbooks for common incident types.

---

## 8.7.10 Case Study: An End-to-End Quant MLOps Stack

Consider a mid-sized systematic equity desk trading a 3000-name universe with daily rebalancing:

### Data
- **Raw market data**: Nasdaq TotalView, NYSE OpenBook, archived in KDB.
- **Fundamentals**: Compustat + Bloomberg, PIT-adjusted.
- **Alternative data**: 5 vendors, ingested to S3 and cataloged in a data lake.
- **Reference data**: CRSP linking historical tickers; delisting information.

### Research infrastructure
- **Shared GPU cluster** with 200 nodes, Slurm scheduled.
- **JupyterHub** for interactive research.
- **Airflow** for scheduled pipelines (daily feature computation, nightly backtests).
- **MLflow** for experiment tracking.
- **Git** for code; monorepo with CI via GitHub Actions / Jenkins.

### Feature store
- In-house: features defined as Python functions with declared lineage; precomputed daily; time-travel queries supported.

### Backtester
- Vectorized engine in Pandas/Polars for daily signal research.
- Event-driven engine in C++/Python for execution research.
- Standardized cost model calibrated weekly.

### Production
- **Execution engine** in C++; connects to multiple venues via FIX; risk gateways before any order.
- **Model serving**: serialized linear and tree models loaded directly; JAX for any NN inference on small hot batch.
- **Monitoring**: Grafana dashboards on PnL, exposure, slippage, feed-health; PagerDuty for alarms.

### Risk
- Pre-trade checks (position, exposure, notional caps).
- Real-time stress VaR computed every minute.
- End-of-day VaR/ES reported to risk committee.

### People workflow
- Researchers produce strategy proposals → review panel → limited paper trade → small-size live → scale up.
- Production strategies are versioned; changes go through code review + risk sign-off.

The stack is deliberately uneventful; the goal is for trading to be a boring background process, leaving human attention for research and exceptional events.

---

## 8.7.11 Closing Subject 8 — and Track A

### Subject 8 in review

Across seven modules, Subject 8 translated the mathematics of Subjects 0–7 into the operational reality of modern quant desks:

- **8.1 Portfolio Optimization Under Estimation Error**: robust, Bayesian, and hierarchical alternatives to brittle MVO.
- **8.2 Risk Management**: coherent risk measures, VaR/ES estimation and backtesting, stress and copula aggregation.
- **8.3 Algorithmic Trading Design**: the pipeline from data to live trades, overlaid with the backtesting discipline required to avoid self-deception.
- **8.4 Alternative Data and NLP**: modern sources of alpha, from text and transformers to satellite and consumer panels.
- **8.5 Bayesian Methods and MCMC**: coherent uncertainty quantification and decision theory for finance.
- **8.6 HFT and Market Making**: microsecond economics, optimal quoting, latency infrastructure.
- **8.7 Systems and MLOps**: the engineering substrate making all of the above routinely reliable.

### Track A capstone summary

The curriculum (Subjects 0–8) now spans:

- **S0 Prerequisites**: logic, linear algebra, real and complex analysis, optimization.
- **S1 Measure Theory**: σ-algebras, Lebesgue integration, product measures, $L^p$, Radon-Nikodym.
- **S2 Probability**: spaces, random variables, conditional expectation, convergence, characteristic functions, martingales.
- **S3 Stochastic Processes**: Brownian motion, Itô calculus, SDEs, jump processes, continuous-time martingales.
- **S4 Optimal Stopping and Control**: Snell envelope, dynamic programming, HJB, Merton, LQG, RL/ADP.
- **S5 Asset Pricing**: no-arbitrage fundamentals, BS, exotics, interest-rate models, credit, stochastic vol, microstructure.
- **S6 Numerical Methods**: Monte Carlo, QMC, finite differences, trees, Fourier, American-option numerics, ML pricing and hedging.
- **S7 Statistical Learning & Econometrics**: regression, classification, trees/boosting, time series, volatility, factor models, causal.
- **S8 Advanced Applications**: portfolio construction, risk, trading systems, alt data, Bayesian, HFT, MLOps.

This is the modern quant research curriculum — from measure theory to MLOps, with every layer theoretically grounded and practically actionable.

### Where to go next

Track A has covered the foundations. Graduate-level research extensions include:

- **Rough volatility and path-dependent calculus** (Bayer-Friz-Gatheral, Fukasawa).
- **Mean-field games in finance** (Lasry-Lions, Guéant-Lasry-Lions).
- **Path-dependent PDEs and functional Itô calculus** (Dupire, Cont-Fournié).
- **Robust and distributionally-robust methods** under model uncertainty.
- **Reinforcement learning in continuous time** (Jia-Zhou, Wang-Zariphopoulou-Zhou).
- **Quantum algorithms for quant finance** (Monte Carlo speed-up, QAE).
- **Transformer-scale models for finance** (time-series foundation models).
- **Decentralized finance mathematics**: AMMs, auction dynamics, on-chain microstructure.

A researcher who has absorbed Track A is well-placed to read any modern quant paper, critique any methodology, and design novel strategies from first principles. The rest is practice, domain knowledge, and the market.

---

## 8.7.12 Exercises

### ★

1. Describe three components of a feature store and why each is important for MLOps discipline.
2. Give two examples of lookahead bias that a PIT database prevents.
3. Define concept drift and data drift; give an example of each.
4. List three items that every reproducible experiment record should contain.
5. Describe the role of a canary deployment in a trading system rollout.
6. Explain the difference between vectorized and event-driven backtesting.

### ★★

7. Design a feature store API supporting (a) offline time-travel queries, (b) online serving with <10ms latency. Specify the interfaces.
8. Build a data-drift detection pipeline using PSI with daily alarms; choose reasonable thresholds and justify.
9. Implement a walk-forward backtesting harness with experiment tracking (MLflow or equivalent) for 100 parameter combinations.
10. Describe the post-mortem for a deployment regression: the live strategy's Sharpe drops from 2.0 to 1.0 within a week. Layout the steps of investigation.
11. Design a canary deployment policy: at what tranches of notional do you expand, and what automatic rollback triggers apply?
12. Compare three strategies for model serving in HFT (embedded weights, RPC, custom serialized) and recommend the right choice for a strategy with <100μs latency budget.

### ★★★

13. Propose and justify a formal SLA framework for a critical market-data feed, including latency, completeness, and value-distribution checks.
14. Design a governance framework for new alpha deployment that integrates research, risk, and compliance reviews. Map each review to an artifact (doc, approval, audit record).
15. Formalize the "probability of backtest overfitting" argument (Bailey-López de Prado) and describe how a firm's experiment registry can be used to estimate it.
16. Architect a system for online learning of a market-making toxicity classifier that (a) updates every minute, (b) has automatic rollback on divergence, (c) is reproducible.
17. Specify the security posture for a research environment: data classification, access controls, audit logging, incident response. Identify at least three failure modes and their mitigations.
18. Draft the research-to-production contract for a statistical arbitrage strategy: inputs, parameters, constraints, monitoring metrics, escalation criteria. Aim for a complete and unambiguous specification.

---

*— End of Module 8.7, Subject 8, and Track A. The final chapter of a mathematical foundations course does not end in theorems, but in the recognition that the theorems you have mastered are the *starting point*.*

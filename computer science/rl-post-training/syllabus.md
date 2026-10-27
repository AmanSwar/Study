# Syllabus: Reinforcement Learning for LLM Post-Training

As of 2026-10-26. 8 parts, 34 modules, about 229,700 estimated words (about 17 hours at 220 words per minute). Source map: 60 sources, 376 facts, 15 disagreements (`sources.json`, `sources.md`). Written for one reader: an LLM-inference engineer who knows pretraining, SFT and transformers, and has never run the RL phase.

## How the course is built

The course follows the dependency order of the mechanisms, not the order in which the field announced them.

1. Module 1 is the quantitative frame. In RL post-training the frame is a cost model: one RL iteration is mostly inference-shaped work (rollouts) plus a smaller training-shaped update. Every later module refers back to its vocabulary (prompts per step, samples per prompt, response length, minibatches per rollout, off-policy lag) and its numbers (R1-Zero 101,376 GPU-hours, Olmo 3's 1,000-second step with 125 seconds of training).
2. Part 1 gives the policy-gradient and trust-region mechanics that every algorithm modifies. Part 2 is classical RLHF (reward model, PPO, KL, DPO, AI feedback). Part 3 drops the critic (GRPO), defines RLVR, and works through the loss, advantage and ratio choices of the reasoning-RL recipes. Part 4 covers what breaks in long runs and what compute buys. Part 5 covers where rewards come from beyond a unit test, what an RL environment is, agentic multi-turn RL, reward hacking and how to evaluate gains. Part 6 covers the paradigms that change the training signal itself (on-policy distillation, multi-teacher distillation, length and effort control, calibration, and the RLCD claims). Part 7 reads seven documented recipes stage by stage. Part 8 closes with when-to-use-which and the frontier.
3. Disagreements are distributed where they arise (table below), not collected into one controversies module. Every module ends with a "Where experts disagree" section; where the sources record no disagreement for a module (Module 17, environments), the section says so and lists the open questions instead of inventing a dispute.
4. Case-study modules compose mechanisms already taught. They reference the mechanism modules by number and spend their words on ordering, numbers, hyperparameters, compute and what each report says went wrong.
5. No exercises, quizzes, deliverables, projects, schedules or assessments anywhere.

## Corrections to the request, carried into the material

- **TRPO vs GRPO.** DeepSeek-R1 uses GRPO, introduced in DeepSeekMath (Shao et al., Feb 2024, arXiv 2402.03300). TRPO (Schulman et al., 2015) is PPO's ancestor. Module 3 teaches TRPO -> PPO; Module 9 teaches GRPO; Module 27 (R1) states the correct lineage.
- **R1 was not the first** to introduce GRPO (DeepSeekMath), nor the term RLVR (Tulu 3, 22 Nov 2024), nor reasoning RL (OpenAI o1-preview, Sep 2024; the OpenAI page returned HTTP 403, so its existence is attested only through Lambert's bibliography). R1 popularized pure rule-based RLVR with GRPO at frontier scale. R1 (22 Jan 2025) and Kimi k1.5 were posted the same day.
- **RLCD** is not one thing. Four methods share or collide with the acronym (Yang et al. 2023 contrastive distillation; TypeSafe AI 2026 calibrated decisions; RLCR; RLHV). Module 8 teaches the 2023 one; Module 24 teaches the public calibration machinery; Module 25 is the claims ledger for TypeSafe's. See "RLCD framing" below.

## Reading order

Prerequisites are module numbers within this course. Cross-references to the System Design course are by its module numbers and are never re-covered here.

### Part 1. The RL Frame for a Language-Model Policy

| # | Module | Words | Needs |
|---|---|---|---|
| 1 | The RL Post-Training Loop as a Cost Model: Rollout, Reward, Update, and Where the GPU-Hours Go | 6,500 | none |
| 2 | Policy Gradients for Token Sequences: the Log-Derivative Trick, Baselines, and Advantages for an LLM Policy | 6,500 | 1 |
| 3 | From TRPO to PPO: Trust Regions, Importance Ratios, the Clipped Surrogate, and GAE | 6,500 | 2 |

1: the post-training map, one iteration decomposed, reported run arithmetic, model counts per method. 2: token-MDP vs bandit views, REINFORCE as reward-weighted SFT, baselines, advantage. 3: TRPO's constraint, PPO's clip and min, GAE, what clipping does not guarantee, the lineage table.

### Part 2. Preference Learning and Classical RLHF

| # | Module | Words | Needs |
|---|---|---|---|
| 4 | Reward Models from Preferences: the Bradley-Terry Loss, RM Architecture and Scale, and Outcome versus Process Supervision (thin) | 5,500 | 2 |
| 5 | PPO-RLHF as a System: Policy, Reference, Reward and Value Models, Per-Token KL Reward, GAE, and the Pretraining Mix | 7,000 | 3, 4 |
| 6 | KL Regularization and Reward-Model Over-Optimization: Reference Policies, the k1/k2/k3 Estimators, and Gold-versus-Proxy Scaling Laws | 6,000 | 3, 4, 5 |
| 7 | Direct Alignment: Deriving DPO, the Implicit Reward, Its Failure Modes, and the KTO, Length-Normalized and NLL-Regularized Variants | 7,500 | 4, 6 |
| 8 | AI Feedback as the Labeler: RLAIF, Constitutional AI, and Contrastive-Prompt Preference Pairs (Yang et al.'s RLCD) (thin) | 5,200 | 4, 5, 7 |

4: Bradley-Terry loss, RM head, 6B-vs-175B stability, ORM vs PRM. 5: the four-model PPO system, per-token KL reward, RM-initialised value, PPO-ptx, Tulu 3's PPO details. 6: KL objective and optimum, estimator tables, biased k3 gradient, Gao et al.'s forms. 7: DPO derivation, failure modes, variants, the DPO-vs-PPO evidence ledger. 8: CAI, RLAIF, Yang et al.'s contrastive prompts, the acronym map.

### Part 3. Critic-Free Policy Optimization and Verifiable Rewards

| # | Module | Words | Needs |
|---|---|---|---|
| 9 | Critic-Free Policy Gradient: REINFORCE with Group Baselines, RLOO, GRPO, and Kimi's Mirror-Descent Objective | 7,500 | 3, 5, 6 |
| 10 | RL with Verifiable Rewards: Rule-Based Reward Design, Prompt Selection, and What R1-Zero Showed | 7,000 | 5, 9 |
| 11 | The GRPO Loss and Advantage Under the Microscope: Length Bias, Difficulty Bias, Token-Level Averaging, and Zero-Variance Groups | 7,000 | 9 |
| 12 | Importance Ratios and Clipping: Token-Level, Sequence-Level (GSPO), IS-Weight Clipping (CISPO), and Log-Ratio Masks | 7,500 | 3, 9, 11 |

9: group baseline, RLOO, DeepSeekMath settings, unified gradient view, Kimi's objective, the critic debate. 10: verifier taxonomy, reward composition, prompt selection, base-model dependence. 11: per-token weight derivations, Dr. GRPO, DAPO ladder, zero-variance groups. 12: when the ratio is non-trivial, GSPO, Clip-Higher, CISPO, K2.5, GLM-5, V3.2 masks.

### Part 4. Making RL Run: Dynamics, Numerical Mismatch, and Compute Scaling

| # | Module | Words | Needs |
|---|---|---|---|
| 13 | Training Dynamics: Entropy Collapse, KL Drift, Length Growth, and the Controls That Work | 7,000 | 6, 11, 12 |
| 14 | Training-Inference Mismatch: Why the Rollout Engine and the Trainer Disagree on Probabilities, and the Four Families of Fixes | 7,000 | 1, 12 |
| 15 | RL Compute Scaling: Sigmoidal Fits, the ScaleRL Recipe, and What Longer Runs Actually Buy | 6,000 | 1, 11, 13 |

13: entropy law and covariance mechanism, Clip-Cov, ProRL, the KL keep-or-drop debate, run-health signals. 14: sources of divergence between vLLM-style engines and trainers, four fix families; this is where the reader's inference background is the lever. 15: ScaleRL's sigmoid, efficiency vs asymptote, compute budgets across reports.

### Part 5. Rewards, Environments, Agents, and Their Failure Modes

| # | Module | Words | Needs |
|---|---|---|---|
| 16 | Rewards Beyond Ground Truth: LLM Judges, Rubric Rewards, Generative Reward Models, and Judges Trained Inside the Loop | 7,500 | 4, 8, 10 |
| 17 | RL Environments for LLMs: Task Datasets, Harnesses, Verifiers, State, and Reward Shaping | 7,500 | 10, 16 |
| 18 | Agentic and Multi-Turn RL: Trajectories as Episodes, Credit Assignment, Observation Tokens, and Staleness | 7,500 | 12, 14, 17 |
| 19 | Reward Hacking in Practice: Proxy Exploits, Emergent Misalignment from Hacked Environments, and the Defenses Reports Describe | 6,500 | 6, 10, 16, 17 |
| 20 | Evaluating What RL Did: pass@1 versus pass@k, Spurious Rewards, Contamination, and the Sharpen-or-Expand Debate | 7,000 | 10, 13, 19 |

16: verifier-to-judge ladder, K2/K3/V3.2/V4/Qwen3 judge designs. 17: Gymnasium vs LLM environments, NeMo Gym/OpenEnv/Reasoning Gym, synthesis at scale, harness diversity. 18: loss masks, GiGPO, PARL, partial rollouts, async agent RL. 19: documented exploits and defenses, emergent misalignment. 20: pass@k evidence for and against capability expansion.

### Part 6. Beyond One Scalar Reward: Distillation, Effort Control, and Calibration

| # | Module | Words | Needs |
|---|---|---|---|
| 21 | On-Policy Distillation as Dense-Reward RL: Per-Token Reverse KL, Cost Arithmetic, and When It Beats RL | 6,500 | 9, 10, 20 |
| 22 | Capability Integration: Mixed RL, Cascade RL, Parameter Merging, and Multi-Teacher On-Policy Distillation (MOPD) | 7,000 | 16, 18, 21 |
| 23 | Length and Reasoning-Effort Control: Length Rewards, Token Budgets, Toggle, and Effort-Conditioned Experts | 6,000 | 11, 13, 22 |
| 24 | Calibration-Aware RL: Proper Scoring Rules, the RLCR Reward, and Why Binary Rewards Teach Guessing | 6,000 | 10, 20 |
| 25 | The Name RLCD: What TypeSafe AI Claims, What Is Public, and the OpenJev Reconstruction (thin, claims-only) | 5,000 | 8, 10, 24 |

21: taxonomy, objective, Thinking Machines and Qwen3 numbers. 22: integration strategies, V4/K3/Nemotron/GLM-5 designs. 23: k1.5 length reward through K3 effort experts. 24: Brier-term reward, Theorem 1, Kalai et al.'s grading argument. 25: ledger of TypeSafe's claims, what is not disclosed, the OpenJev reconstruction labelled as such.

### Part 7. Case Studies: Documented Post-Training Recipes, 2024-2026

| # | Module | Words | Needs |
|---|---|---|---|
| 26 | Case Study: Llama 3 and Tulu 3 (2024) - DPO-First Post-Training, the PPO Comparison, and the Birth of RLVR | 6,500 | 5, 7, 10 |
| 27 | Case Study: DeepSeek-R1 - R1-Zero, the Four-Stage R1 Pipeline, Its Compute, and What the Nature Version Changed | 9,000 | 9, 10, 13, 20 |
| 28 | Case Study: Kimi k1.5 to K3 - Mirror-Descent Policy Optimization, Self-Critic Rewards, Token-Budget RL, and Nine-Expert MOPD | 8,500 | 9, 12, 16, 18, 22, 23 |
| 29 | Case Study: Qwen3 - Four-Stage Reasoning Pipeline, 3,995 Query-Verifier Pairs, Thinking-Mode Fusion, and Strong-to-Weak Distillation | 5,500 | 10, 16, 21 |
| 30 | Case Study: DeepSeek-V3.2 to V4 - Scaling GRPO Past 10% of Pre-Training Compute, Then Replacing Mixed RL with On-Policy Distillation | 8,000 | 11, 14, 16, 22, 23, 27 |
| 31 | Case Study: Olmo 3 - OlmoRL, RL-Zero from a Fully Open Base, and Measured Rollout Economics | 5,500 | 1, 11, 14, 20 |
| 32 | Case Study: GLM-5 and Nemotron 3 Ultra - Asynchronous Agentic RL, Deterministic Indexer Top-k, Open Environments, and the Limits of MOPD | 6,500 | 14, 17, 18, 22 |

Ordered roughly chronologically by report date so that the convergence (no KL, token-level aggregation, mismatch correction, MOPD) is visible as it happens. Every one of these has a primary technical report in the source map. Excluded as case studies: OpenAI, Anthropic and Google frontier models (no primary recipe found), Qwen3.5 (no official report located), gpt-oss (model card says only "similar CoT RL techniques as OpenAI o3"; pointer in Module 34), MiMo-V2-Flash and Magistral (pointers, covered through Modules 13 and 22).

### Part 8. Choosing Methods and Reading the Frontier

| # | Module | Words | Needs |
|---|---|---|---|
| 33 | Choosing a Method: PPO, DPO, GRPO-Family, On-Policy Distillation, and Hybrids Under Reward, Data, Compute, and Stability Constraints | 7,500 | 7, 9, 11, 15, 21, 22, 24 |
| 34 | The Frontier as of October 2026: Open Problems, Moving Parts, and What Closed Labs Do and Do Not Disclose | 6,500 | 18, 22, 25, 33 |

33: decision framework, each cell sourced and tagged with evidence strength, "no source" cells stated. 34: dated settled/contested/open map, leading indicators, bounded closed-lab section, date and version hygiene.

## Where each disagreement is treated

`*` marks the module that carries the primary treatment; the others apply or summarise it.

| Disagreement (sources.json) | Modules |
|---|---|
| DPO enough, or does online RL (PPO) win? | 7*, 5, 26, 33 (cost framing in 1) |
| Critic or no critic | 9*, 5, 26, 28, 33 (framing in 2) |
| Does RLVR create capability or only sharpen? | 20*, 27, 29, 31, 34 |
| GRPO loss and advantage normalization | 11*, 15, 30, 31, 33 |
| Keep or remove KL | 13*, 6, 30, 31, 32 |
| Importance ratio and clipping | 12*, 3, 18, 28 |
| Preventing entropy collapse | 13* |
| Training-inference mismatch fixes | 14*, 31, 32 |
| Learned RM vs verifier vs rubric judge | 16*, 4, 8, 10, 17, 19, 29, 30 |
| Process vs outcome supervision | 4*, 16, 18, 27 |
| Mixed RL vs sequential vs specialists + OPD | 22*, 18, 28, 30, 32, 33, 34 |
| RL or distillation for small models | 21*, 27, 29, 33, 34 |
| Length and reasoning-effort control | 23*, 11, 28 |
| Calibration: train it in or fix grading | 24*, 25, 34 |
| TypeSafe RLCD: new regime or rebranding | 25*, 34 |

Source-level conflicts that are not in the 15 (flagged in the module `disagreements` fields): 3.75% vs >10% of pretraining compute for RL (Module 1, 15); R1 vs Dr. GRPO on where "reflection" comes from (Module 10); the contrastive-prompt advantage shrinking at 30B simulation (Module 8); an unread third-party replication of the Anthropic reward-hacking paper (Module 19).

## RLCD framing

TypeSafe AI's "Reinforcement Learning for Calibrated Decisions" exists only as company claims: a launch post (15 Sep 2026), the docs AI primer, a founder interview and one TechCrunch article. There is no paper, repository, reward function, algorithm or dataset, and the founder says it is unpublished. The mechanism subtopic has coverage `none` and is not scheduled. The course therefore treats it in exactly three ways:

1. What is claimed and by whom, separating vendor statements from independent ones (Module 25).
2. The public machinery the claims most plausibly resemble, namely proper-scoring-rule rewards for calibration (RLCR, Behaviorally Calibrated RL, Kalai et al.), taught on its own terms in Module 24 and stated in Module 25 as a resemblance, never as TypeSafe's method.
3. The independent OpenJev-RLCD preprint (Qwen3-1.7B), presented as a reconstruction that says itself that TypeSafe discloses no algorithm.

The older RLCD (Reinforcement Learning from Contrastive Distillation, Yang et al., ICLR 2024) is taught as a method in its own right in Module 8. Module 25 is the module most likely to come in below its word estimate; the writer is instructed to report that rather than pad.

## What is deliberately excluded, and why

| Excluded | Reason |
|---|---|
| Pretraining, SFT mechanics, transformer and KV-cache internals, quantization, classical knowledge distillation | He has them. SFT appears only as a stage in the pipeline map (Module 1) and as the special case of the policy gradient (Module 2). Distillation appears only where it competes with RL (Modules 21, 22). |
| RL training infrastructure, async trainer/rollout design, weight sync, framework comparison (verl, OpenRLHF, SkyRL, AReaL, slime, NeMo-RL), post-training pipeline/registry plumbing, sandboxes | Covered in the System Design course (Modules 36, 37, 38; related 27, 32, 34, 39) and cross-referenced by number. This course takes only the algorithm-side consequences (why importance-sampling correction exists, how staleness enters the loss). |
| Classical deep RL beyond the policy-gradient lineage (Q-learning/DQN, SAC, model-based RL, AlphaZero-style search) | Not used by any documented LLM recipe. MCTS appears once, as R1's failed attempt (Module 27). |
| Inference-time scaling and search at test time | Not RL training, and his home field. Appears only where it defines metrics (pass@k, majority vote, Module 20). |
| Multimodal and visual RL | No source in the map treats it beyond a Kimi K2.5 mention; PARL and zero-vision SFT are covered only for their RL content. |
| Safety alignment, red-teaming, model character and product behaviour | Not an RL-mechanism topic. Reward hacking and emergent misalignment are included because they are RL failure modes (Module 19). |
| Standalone modules on IPO, ORPO, SimPO, ReMax, VAPO, SAPO, Magistral, MiMo-V2-Flash | Verified to exist (abstract level) but not read in the source map. They appear as pointers inside Modules 7, 9, 12, 13, 22 with a "fetch before citing" instruction. |
| A standalone module on what closed labs disclose | Decision: not scheduled. The source map holds no primary recipe for OpenAI, Anthropic or Google models; what exists is research papers (S3, S6, S28, S33, S7, S29) and one pointer (the gpt-oss model card). That supports a bounded section, not a module. It lives in Module 34, uses only primary statements, lists what is not disclosed, and contains no rumoured recipes. |
| TypeSafe's reward function or algorithm | No source discloses it (coverage none). Not invented, not scheduled. |
| `LLM/llm-training-syllabus.md` | Used for scope only; not citable. |
| Exercises, quizzes, deliverables, projects, week-by-week schedules, assessments | Excluded by rule, in every module. |

## Where the sources are thin, and what the writers must do

- **Thin by the coverage map:** reward modeling (Module 4: the Bradley-Terry loss was not extracted; read InstructGPT Section 3.5, DPO Sections 3-4, Llama 3 Section 4.1.2, Lambert chapter 5), RLAIF and Constitutional AI (Module 8: CAI read to abstract and Section 1; RLAIF itself is not in the map), the TypeSafe claims (Module 25).
- **Abstract-only sources that carry whole arguments:** Xu et al. and Ivison et al. (S11, S12) in Modules 5 and 7; Yue et al. coverage curves and the two-phase reconciliation (S24, S26) in Module 20; Let's Verify Step by Step (S30) in Module 4; Rubrics as Rewards (S31) in Module 16; Behaviorally Calibrated RL (S10) in Module 24; KTO (S9) in Module 7; GiGPO (S58) in Module 18. Writers fetch the full text before leaning on these.
- **Reports read only in part:** Kimi K2, K2.5 and K3, DeepSeek-V4, GLM-5, Qwen3, Nemotron 3 Ultra, Llama 3, Olmo 3 (RL sections only). Case-study writers fetch the remaining sections and mark anything they cannot verify.
- **Second-hand or extraction-only facts, to carry as unverified:** Sutton & Barto chapter and page numbers; the Nature publication details of R1; Yao et al.'s TIS and Liu et al.'s sequence-ratio result (Notion pages, known via S27 and S48); the ~25% cost of importance-sampling correction; the Thinking Machines, TRL, OpenEnv, NeMo Gym and Gymnasium numbers (WebFetch extraction); TypeSafe's ECE 0.246 and "non-autoregressive" attributions (appear on no TypeSafe page read); the founder-interview reading of "not yet published".
- **Version and date traps:** R1 v1 (AIME 71.0%) vs v2/Nature (77.9%); R1's printed clip epsilon of 10; R1's 80 hours vs "about 4 days"; DeepSeek-V4's arXiv id (2606.19348) vs API date (2026-04-26); Lambert's book (arXiv v12, web edition 14 Oct 2026, print July 2026); the Thinking Machines blog's June 2026 update; unversioned docs snapshots; Kalai et al.'s printed penalty for t = 0.75 (2) vs t/(1-t) = 3 (Module 24 flags it).
- **Not mined:** conference talks (GTC, NeurIPS, ICML, Hot Chips) and the OpenAI/Anthropic engineering-blog tier.

## Source-map summary

60 sources: 48 papers, 3 official docs, 3 blogs, 2 repos, 2 textbooks, 1 talk, 1 analysis. Most heavily grounding (by number of modules citing): Lambert's RLHF book (19 modules), DeepSeek-V3.2 (16), DeepSeek-R1 (15), Olmo 3 (10), Kimi K3 (10), Tulu 3 (9), DeepSeekMath (9), TRL GRPO docs (9), Qwen3 (9), DAPO (8). Every one of the 60 sources is cited by at least one module.

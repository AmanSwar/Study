# Sources — Reinforcement Learning for LLM Post-Training
As of 2026-10-26 · 60 sources · course · 48 paper, 3 blog, 2 textbook, 3 official, 1 talk, 1 analysis, 2 repo

## Coverage

**well** (20)

- `rl-foundations` — Policy-gradient foundations and the TRPO -> PPO -> GRPO lineage (advantages, baselines, trust regions, KL estimators) [S1, S2, S3, S4, S5, S14, S54]
- `rlhf-pipeline` — Classic RLHF: SFT -> reward model -> RL with a KL-to-reference penalty (InstructGPT, HH-RLHF, Llama 3 lineage) [S6, S7, S8, S28, S50, S60]
- `generative-judges-rubrics` — Verifiers and judges: rule-based verifiers, LLM judges, rubric rewards, generative reward models [S15, S16, S31, S42, S43, S44, S45, S46, S49]
- `ppo-for-llms` — PPO as used for LLMs: value model, GAE, KL placement, pretraining-mix, cost [S2, S3, S6, S11, S12, S13, S16, S60]
- `direct-preference-optimization` — DPO and the direct-alignment family, offline vs online preference learning [S8, S9, S11, S12, S16, S50, S60]
- `critic-free-rl` — Critic-free policy gradient: REINFORCE, RLOO, GRPO baselines from sample groups [S4, S13, S14, S41, S58, S59, S60]
- `grpo-successors` — GRPO fixes and successors: DAPO, Dr. GRPO, GSPO, CISPO, SAPO, OlmoRL and ScaleRL recipes [S14, S15, S17, S18, S19, S20, S21, S22, S41, S43, S45, S48, S58, S59]
- `rlvr` — RL with verifiable rewards: R1-Zero, Tulu 3 RLVR, rule-based reward design [S10, S14, S15, S16, S17, S18, S23, S24, S25, S26, S30, S31, S32, S33, S48, S55, S60]
- `reward-hacking` — Over-optimization, reward hacking and emergent misalignment [S6, S15, S16, S25, S28, S29, S60]
- `rl-environments` — RL environments: interface, verifiers, gyms, environment synthesis, harness diversity [S29, S42, S44, S45, S47, S51, S54, S55, S56, S57]
- `agentic-multi-turn-rl` — Agentic and multi-turn RL: tool use, long horizons, credit assignment, staleness [S19, S29, S42, S43, S44, S45, S47, S51, S56, S57, S58]
- `training-dynamics` — Training dynamics and failure modes: entropy collapse, length growth, KL, off-policy drift, train/inference mismatch, MoE routing [S4, S15, S17, S18, S19, S20, S21, S22, S23, S27, S43, S45, S47, S48, S59]
- `rl-scaling` — RL compute scaling, cost accounting and rollout-bound wall-clock [S15, S20, S22, S23, S27, S41, S44, S45, S48, S53]
- `distillation-opd` — On-policy and multi-teacher distillation (OPD, MOPD) versus RL [S15, S44, S46, S49, S51, S52, S53, S60]
- `effort-control` — Reasoning-effort and length control as a training target [S41, S42, S43, S44, S46, S49, S51]
- `calibration-rl` — Calibration-aware RL: proper scoring rules, RLCR, abstention incentives, hallucination incentives [S10, S32, S33, S35, S39]
- `case-studies` — Frontier recipes documented in primary reports: DeepSeek-R1/V3.2/V4, Kimi k1.5/K2/K2.5/K3, GLM-5, Olmo 3, Qwen3, Nemotron 3 Ultra, Llama 3, Tulu 3 [S15, S16, S41, S42, S43, S44, S45, S46, S47, S48, S49, S50, S51]
- `when-to-use-which` — Decision criteria: which method under which constraints [S11, S12, S13, S16, S22, S48, S49, S50, S52, S53, S59, S60]
- `frontier-2026` — What is moving as of Oct 2026 (MOPD, async agentic RL, reasoning-effort RL, calibration) [S44, S46, S47, S51, S52, S53, S60]
- `evaluation-of-rl` — Evaluating RL gains: pass@k vs pass@1, spurious rewards, contamination [S18, S23, S24, S25, S26, S33, S48, S49]

**thin** (3)

- `reward-modeling` — Classical reward modeling: Bradley-Terry RMs, outcome vs process reward models, RM scale and drift [S6, S15, S28, S30, S50, S60]
- `rlaif-cai` — AI feedback: RLAIF, Constitutional AI, contrastive distillation (Yang et al. RLCD, 2023) [S7, S34]
- `rlcd-typesafe-claims` — TypeSafe AI "RLCD" (Sept 2026): what the company publicly claims [S35, S36, S37, S38, S39, S40]

**none** (1)

- `rlcd-typesafe-mechanism` — TypeSafe AI "RLCD": reward function, algorithm, data (mechanism) [no source]

## Disagreements found

- **Is DPO enough, or does online RL (PPO) win for preference tuning?**
  - A. DPO-family methods are sufficient and cheaper at scale: Llama 3 tried PPO and found DPO needed less compute and did better on IFEval; Tulu 3 measured PPO roughly equal to DPO at about 28 h on two nodes vs about 4 h on one node and kept DPO for preference iteration. [S50, S16]
  - B. Tuned PPO beats DPO: Xu et al. report PPO ahead on every testbed; Ivison et al. find PPO ahead by up to 2.5% (math) and 1.2% (general), but also that data quality matters more than the algorithm. [S11, S12]
  - C. RL keeps winning despite far higher infrastructure cost, and can be applied surgically (math-only RL preserved general ability in Tulu 3). [S60]
- **Critic or no critic: is a value model worth it?**
  - A. PPO's critic is unnecessary or harmful for LLMs (reward only at the last token, policy-sized memory cost): REINFORCE/RLOO, GRPO and Kimi's mean-reward baseline drop it. [S13, S14, S41]
  - B. A value model still carries weight in practice: Tulu 3 RLVR found initializing the value function from a general RM gave the best results and listed value-free GRPO only as future work; Xu et al. get best results from carefully configured PPO. [S16, S11]
- **Does RLVR create new reasoning capability, or only sharpen what the base model can already do?**
  - A. It sharpens: pass@k at large k is higher for base models and the boundary narrows; DeepSeekMath saw Maj@K but not Pass@K improve; Qwen3 found on-policy distillation lifted pass@64 while RL did not. [S24, S14, S49]
  - B. It can expand with long enough training, KL control with reference resets and diverse tasks (1.5B model, >2k steps). [S23]
  - C. Both, in phases: an exploitation stage followed by an exploration stage (theoretical reconciliation, authors' own assessment). [S26]
  - D. Much of the observed gain is prior amplification: random or wrong rewards lift Qwen2.5-Math by 21.4 MATH-500 points vs 29.1 for true rewards; templates and base-model pretraining drive R1-Zero-like results; contamination can make spurious rewards look as good as true ones. [S25, S18, S48]
- **How should the GRPO loss and advantage be normalized?**
  - A. Original GRPO: per-sequence 1/|o_i| averaging and division by group reward std (still the TRL group-std default). [S14, S59]
  - B. Token-level averaging across the batch (DAPO; OlmoRL; TRL loss default "dapo"). [S17, S48, S59]
  - C. Constant normalizer and no std division, to remove length and difficulty bias (Dr. GRPO). [S18]
  - D. Keep token-level loss but drop only the std division (DeepSeek-V3.2 advantage = R - mean(R); OlmoRL). [S45, S48]
  - E. These choices mostly move compute efficiency, not the asymptote (ScaleRL uses prompt-level loss averaging and batch-level advantage normalization). [S22]
- **Keep or remove the KL-to-reference term?**
  - A. Remove it for long-CoT RL: policy must diverge; no over-optimization observed (DAPO, OlmoRL, GLM-5; TRL default beta = 0.0). [S17, S48, S47, S59]
  - B. Keep it, with periodic reference resets, when starting from an SFT/distilled checkpoint: more stable than other entropy fixes; "remove KL" advice applies to raw base models. [S23]
  - C. Small, domain-dependent KL with a corrected estimator (V3.2: weak or none for math); small coefficients elsewhere (R1 0.001; DeepSeekMath 0.04; Tulu 3 0.05-0.07). [S45, S15, S14, S16]
- **Importance ratio and clipping: token-level, sequence-level, IS-weight clipping, or soft gating?**
  - A. Token-level PPO-style clip, widened on the upside (Clip-Higher 0.2/0.28). [S3, S17]
  - B. Sequence-level ratio and clipping (GSPO), which removes the need for MoE routing replay and tolerates train/inference numeric differences. [S19]
  - C. Clip the importance weight, not the update, so low-probability "fork" tokens keep contributing gradients (CISPO; adopted inside ScaleRL). [S20, S22]
  - D. Mask tokens by log-ratio outside [alpha, beta] regardless of advantage sign (Kimi K2.5); per-token regularization tolerates staleness (K3). [S43, S44]
- **How to prevent entropy collapse**
  - A. Asymmetric clipping (Clip-Higher) plus dynamic sampling / active sampling. [S17, S48]
  - B. Covariance-selective control: Clip-Cov / KL-Cov on tokens with high probability-advantage covariance. [S21]
  - C. KL penalty with reference resets and higher temperature. [S23]
  - D. Sampling temperature decay from exploration to exploitation. [S42]
- **Fixing the training-inference mismatch (vLLM/SGLang rollout vs trainer): algorithm, numerics, or determinism?**
  - A. Correct statistically with importance sampling (truncated or masked; token or sequence level). [S48, S59, S27]
  - B. Fix numerics: FP32 LM head (MiniMax-M1; ScaleRL) or FP16 instead of BF16. [S20, S22, S27]
  - C. Remove the source: reuse sampling-time MoE routing and top-p/top-k masks (DeepSeek-V3.2), deterministic indexer top-k plus IcePop masking (GLM-5). [S45, S47]
  - D. Mask highly off-policy negative-advantage sequences using the inference engine's own probabilities. [S45]
- **Learned reward models, verifiers, or generative/rubric judges?**
  - A. Avoid neural reward models on reasoning tasks because they are hacked during large-scale RL; use rule-based verifiers. [S15, S49]
  - B. Use generative/rubric judges for non-verifiable tasks: Kimi self-critic rubric reward and GRMs, per-prompt rubrics in V3.2, rubric-guided GRM in V4, Rubrics-as-Rewards. [S42, S43, S44, S45, S46, S31]
  - C. Train the judge inside the same RL loop (actor doubles as GRM in V4; critic refined on verifiable rollouts in K2; agentic judge protocol in K3). [S46, S42, S44]
- **Process vs outcome supervision**
  - A. Process supervision beats outcome supervision for reward models on MATH. [S30]
  - B. At RL scale PRMs are impractical: hard step definition, hard step verification, hackable. [S15]
  - C. Stepwise reward models remain useful for filtering SFT traces. [S50]
  - D. Step-level credit without a learned PRM, via anchor-state grouping in multi-turn tasks. [S58]
- **How to combine many skills: single mixed RL, sequential stages, or specialists + on-policy distillation?**
  - A. Merge reasoning, agent and alignment RL into one mixed stage to avoid forgetting (DeepSeek-V3.2), or one unified RLVR run before distillation (Nemotron 3 Ultra). [S45, S51]
  - B. Train domain experts with RL, then replace mixed RL entirely with multi-teacher on-policy distillation (DeepSeek-V4, Kimi K3 with nine experts, MOPD paper). [S46, S44, S52]
  - C. Multi-stage RL with cross-stage distillation (GLM-5); four-stage pipeline plus strong-to-weak distillation for small models (Qwen3). [S47, S49]
  - D. Labs are not unanimous; MOPD has open issues on long-horizon agentic rollouts. [S60, S51]
- **RL or distillation for strong small reasoning models?**
  - A. Distill: R1 found 10K+ steps of RL on Qwen2.5-32B only matched QwQ-32B-Preview while distilling R1 did better; Qwen3 and Thinking Machines report on-policy distillation about 1/10 the GPU hours of RL (17,920 vs 1,800 GPU-hours, Qwen3-8B). [S15, S49, S53]
  - B. RL is still needed to push beyond the teacher / frontier ("advancing beyond the boundaries of human intelligence may still require more powerful base models and larger-scale reinforcement learning") and ProRL reports gains beyond a distilled start. [S15, S23]
- **Controlling response length / reasoning effort**
  - A. Length penalty, warmed up after standard RL (Kimi k1.5); soft overlong punishment (DAPO). [S41, S17]
  - B. Per-sample or per-problem token budgets as a reward override (K2, K3), alternating budget-limited and unconstrained phases (Toggle: -25 to -30% tokens). [S42, S44, S43]
  - C. Train separate experts per reasoning-effort mode with distinct length penalties and contexts (V4, K3 low/high/max). [S46, S44]
  - D. Fix the length bias in the loss itself rather than adding a penalty (Dr. GRPO). [S18]
- **Calibration: train it into the policy, or fix how benchmarks grade?**
  - A. Add a bounded proper-scoring-rule term (Brier) to the RL reward; calibration error 0.37 -> 0.03 on HotpotQA with matched accuracy. [S32, S10]
  - B. Primary fix is socio-technical: stop 0-1 grading that penalizes abstention; add explicit confidence targets to benchmark instructions. [S33]
- **TypeSafe "RLCD": new post-training regime, or a rebranding of known calibration-aware RL?**
  - A. Company position: RLHF causes sycophancy, mode dropping and overconfidence; RLCD yields typed outputs with calibrated probabilities (fast, cheap, "can't hallucinate" at the type level); RLHF is "please humans", RLVR is "optimize benchmarks", RLCD is "make it reliable for programmatic use". [S35, S36, S37]
  - B. Public literature already shows binary-reward RLVR degrades calibration and that proper-scoring-rule rewards repair it in ordinary text-generating models; an independent reconstruction (score each rationale's answer distribution with a proper score) exists but is not TypeSafe's method. [S32, S10, S33, S39]
  - C. Outside assessment is mixed and unverified: a developer says it "delegates the hallucination problem a little bit to the user"; one tester found Gemini slightly more accurate; company accuracy parity is scored against the average of two frontier LLMs' outputs rather than ground truth; no published benchmarks. [S38, S36]

## Notes for the planner and writers

**AS-OF AND METHOD.** As of 2026-10-26. Quality control: all 324 quotes that have a machine-checkable local text were verified as substrings of the fetched text (305 against cached PDF, HTML or markdown text; 19 against arXiv-API abstracts), after normalising whitespace, curly quotes and line-break hyphenation; the other 52 of the 376 facts carry no quote or come from WebFetch extractions (those sources say so in readDepth and their quotes are the extraction tool's output). Each source's readDepth field says how much was read. Fetch accounting: 29 WebFetch calls (7 returned an error, an empty page or nothing useful), 56 curl downloads, 9 arXiv-API calls, about 94 in total against a budget of 120; PDFs were converted with pdftotext and read by targeted grep/sed (not the PDF Read tool) to save context.

**RLCD FINDING.** "Reinforcement Learning for Calibrated Decisions (RLCD)" by TypeSafe AI IS real as a company-asserted method, but there is no paper, repo, reward function, or algorithm. Primary statements: launch post by Diogo Almeida dated 15 Sep 2026 (S36), docs.typesafe.ai "AI primer" (S35), founder interview (S37). The only third-party coverage is TechCrunch 18 Sep 2026 (S38), which writes "from" not "for" calibrated decisions. Public technical content under the name RLCD is third-party and independent: OpenJev-RLCD (arXiv 2609.38850, S39) is a re-derivation (score each sampled rationale's answer distribution with a strictly proper scoring rule; "calibrate, then reinforce"; Qwen3-1.7B) that says explicitly TypeSafe discloses no algorithm; arXiv 2609.28940 (S40) and arXiv 2609.29429 ("Just Ask Jev", abstract verified, not entered) use the term. The founder says RLCD is unpublished, declines to explain it, and says "it's not about the PPO". Therefore the planner may schedule an RLCD module only as: (1) what is claimed and by whom, with the vendor/independent distinction; (2) the public, verifiable machinery it most plausibly resembles (RLCR S32, Behaviorally Calibrated RL S10, Kalai et al. S33, proper scoring rules); (3) the OpenJev reconstruction labelled as a reconstruction. It must NOT present a TypeSafe reward function or algorithm. Question for Aman at the checkpoint: does he accept this framing, or did he mean RLCR or another method? Searches run (WebSearch): "Reinforcement Learning for Calibrated Decisions RLCD"; "TypeSafeAI RLCD reinforcement learning calibrated decisions LLM"; "TypeSafeAI github"; "RLCD arxiv 2026 calibrated decisions ..."; "TechCrunch TypeSafe Jev ..."; plus direct fetches of typesafe.ai pages, docs.typesafe.ai (llms.txt index lists no research/training page), and the TypeSafe blog index (five posts; none discloses RLCD detail). GitHub: github.com/typesafe-ai (official: SDKs, adapter, skills) and github.com/TypeSafeAI (an unofficial community org that says it does not speak for the company); neither has RLCD training code. Date oddity: every TypeSafe blog page carries metadata date 19 Oct 2026 (export timestamp) while headers show 9-15 Sep / 9-14 Oct. Company funding statement in S36 sibling post: $870M series A at $7.5B valuation (a footnote); a launch post body reads "we raised $$$" as a placeholder.

**NAME COLLISIONS.** RLCD (Yang et al., ICLR 2024) = Reinforcement Learning from Contrastive Distillation, arXiv 2307.12950 (S34, verified: authors Kevin Yang, Dan Klein, Asli Celikyilmaz, Nanyun Peng, Yuandong Tian; some indexes print "Contrast"). RLCD (TypeSafe, 2026) = Reinforcement Learning for Calibrated Decisions (S35, S36). RLCR (Damani et al., ICLR 2026) = Reinforcement Learning with Calibration Rewards, arXiv 2507.16806 (S32). RLHV = Barbosa's proposal in S40. Coordinator's recollection of the older RLCD (Yang et al., 2023/ICLR 2024) is CONFIRMED.

**COORDINATOR NOTE 1 AND PRIORITY CLAIMS.** Aman wrote "TRPO"; DeepSeek-R1 uses GRPO, introduced in DeepSeekMath (Shao et al., Feb 2024, arXiv 2402.03300, S14). TRPO (Schulman et al., 2015, S1) -> PPO (2017, S3) -> GRPO lineage is sourced. Also correct a belief in the request: R1 was not the first to introduce GRPO (DeepSeekMath, Feb 2024), nor the term RLVR (Tulu 3, 22 Nov 2024, S16), nor first reasoning-RL model (OpenAI o1-preview, Sep 2024: page returned HTTP 403, so only its existence/date is attested through the Lambert book's bibliography). R1 (arXiv 22 Jan 2025) and Kimi k1.5 (S41) were posted the same day. Suggested framing for the course: R1 popularized pure rule-based RLVR with GRPO at frontier scale, standing on DeepSeekMath, Tulu 3, and o1.

**COORDINATOR NOTE 3 OVERLAP WITH SYSTEM-DESIGN.** Verified in computer science/system-design/manifest.json (all status published): Module 36 = "Post-Training Pipelines: SFT/Preference Data Flows, Evaluation Gates, and Model Registries" (file 36-post-training-pipelines.html; prerequisite Module 32; builds on Llama 3's six-round SFT/RM/rejection-sampling/DPO pipeline). Module 37 = "RL Training Infrastructure: Environments as Services, Rollout Generation, and Trainer/Rollout Decoupling" (37-rl-training-infrastructure.html; prerequisites Modules 27 and 36; staleness taxonomy; 3.7-hour rollout arithmetic). Module 38 = "Comparing the Open-Source RL Frameworks: verl, OpenRLHF, SkyRL, AReaL, slime, and NeMo-RL" (38-rl-frameworks-comparison.html; prerequisites Modules 34 and 37). Related: Module 27 = Agent harness and sandboxes (27-agent-harness-sandboxes.html), Module 39 = "Where the Field Still Disagrees and What Is Changing" (frontier synthesis, recaps async-RL disagreement), Module 32 = training data pipelines. This course should reference those by number for infrastructure (async/staleness, framework choice, weight-sync) and cover only algorithm-side consequences (e.g., why importance-sampling correction exists). Our S27, S45, S47, S48 contain the algorithm-facing mismatch facts that Module 37 does not.

**COORDINATOR NOTE 4 CASE STUDIES.** Selected because each has a primary technical report read in part: (1) DeepSeek-R1 (S15) - mandatory; R1-Zero, GRPO, rule-based RLVR, cost, failed PRM/MCTS. (2) Tulu 3 (S16) - RLVR coinage, PPO-vs-DPO cost measured, fully open. (3) Qwen3 (S49) - four-stage pipeline, 3,995 query-verifier pairs, RL vs on-policy distillation table. (4) Kimi k1.5 -> K2 -> K2.5 -> K3 (S41-S44) - non-GRPO loss lineage, self-critic/agentic GRM rewards, budget RL, PARL, MOPD with nine experts. (5) DeepSeek-V3.2 -> V4 (S45, S46) - what it took to scale GRPO to >10% of pre-training compute, then OPD replacing mixed RL. (6) GLM-5 (S47) - async agent RL, deterministic top-k fix. (7) Olmo 3 (S48) - the only fully open recipe with measured infrastructure numbers. (8) Nemotron 3 Ultra (S51) - open recipe and environments, MOPD limits. (9) Llama 3 (S50) - counter-example: DPO instead of PPO. Excluded: OpenAI, Anthropic and Google frontier models (no primary recipe; Anthropic's reward-hacking paper S29 is research, not a model recipe); Qwen3.5 (only secondary coverage found; no official report located); gpt-oss (model card read: only says "similar CoT RL techniques as OpenAI o3", low/medium/high reasoning levels, and no direct optimization pressure on the CoT - too thin for a case study, kept as a pointer); MiMo-V2-Flash (arXiv 2601.02780, first MOPD report; abstract verified, MOPD covered via S52 and S60); Magistral (arXiv 2506.10910, read: removes KL, no std normalization, clip-higher, zero-advantage filtering - same consensus as S48, kept as a pointer).

**COVERAGE CAVEATS.** reward-modeling is thin for the classical mechanics (Bradley-Terry loss, RM architecture, RM calibration); verifier, rubric and generative-judge practice is tagged separately as generative-judges-rubrics and is well covered. The writer must take the loss from DPO §3-4 (S8) and InstructGPT §3.5 (S6) and cite the Lambert book chapter 5 ToC (S60). rlaif-cai is thin: Constitutional AI (S7) was read only to abstract and §1; the older RLCD (S34) was read in more depth. rlcd-typesafe-claims is thin although six sources touch it: only S35 and S36 are primary and both are vendor marketing/documentation; S37 is a founder interview, S38 is one news article, and S39/S40 are independent preprints that either reconstruct or merely cite the term. rlcd-typesafe-mechanism is none. The talks tier (GTC, NeurIPS, ICML, Hot Chips) was not mined: no conference talk was fetched. The OpenAI/Anthropic engineering-blog tier is represented only by Anthropic's paper (S29) and OpenAI authorship of S3/S6/S28/S33.

**VERSION AND DATE AMBIGUITIES.** DeepSeek-R1: v1 (22 Jan 2025) says AIME 2024 pass@1 rose 15.6% -> 71.0% (86.7% majority vote); v2 (4 Jan 2026, expanded Nature version, Nature 645:633-638, 17 Sep 2025 per search results) says 15.6% -> 77.9% (86.7% self-consistency). DeepSeek-V4: arXiv id 2606.19348 but the arXiv API reports published 2026-04-26 (unexplained). Kimi K3 and K2.5 are v2 (7 Aug 2026). Lambert RLHF book: arXiv v12 (11 Sep 2026), web edition built 14 Oct 2026, Manning print July 2026. KTO arXiv record v5 updated 2026-09-08. Thinking Machines OPD blog: posted 27 Oct 2025, updated June 2026 (recipes now use Qwen3.5; original numbers may not reproduce). TRL docs (main branch), OpenEnv README (main) and NeMo Gym README (v0.7.0, date format unlabeled) are unversioned snapshots as of 2026-10-26. R1's printed "GRPO clip ratio epsilon = 10" in stage-1 RL is unexplained.

**CLAIMS NOT VERIFIED.** (a) Sutton & Barto chapter/page numbers (S5) come from search-result summaries; the author site had a TLS certificate error. (b) The Nature publication details of R1 come from search-result summaries; the Nature page redirected to a cookie authorization and was not followed. (c) TypeSafe's ECE 0.246 and "non-autoregressive" claims are attributed to TypeSafe only by S40 and appear on no TypeSafe page read. (d) Latent Space and TechCrunch quotes are WebFetch extraction output; the context of "No, not yet" at [00:25:05] is ambiguous; the last ~22,000 characters of the transcript were not read. (e) TechCrunch's "unnamed outside observers suspect it is built on an open-weight LLM" is the reporter's account and unconfirmed. (f) Thinking Machines blog, TRL docs, OpenEnv, NeMo Gym, Gymnasium numbers/quotes come from WebFetch extraction (<=125-char quotes). (g) Yao et al. 2025 "Your Efficient RL Framework Secretly Brings You Off-Policy RL Training" (the TIS source cited by Olmo 3 and S27) is a Notion page that rendered empty; Liu et al. "When Speed Kills Stability" is also Notion; neither was read - facts about them are second-hand via S27/S48. (h) A third-party replication of S29 (reward-hacking misalignment inconsistent in open models) is from a search summary only. (i) LLM/llm-training-syllabus.md contains arXiv ids and claims that were read for scope only and are not citable; several of its ids (e.g., 2607.20548, 2604.00626, 2606.28116) were not checked here.

**VERIFIED-TO-EXIST POINTERS.** Abstract only via the arXiv API, not entered as sources; a writer must fetch before citing: 1706.03741 Christiano et al. (deep RL from human preferences); 1909.08593 Ziegler et al.; 2009.01325 Stiennon et al.; 2204.05862 Bai et al. HH-RLHF; 2310.12036 IPO/PsiPO; 2403.07691 ORPO; 2405.14734 SimPO; 2310.10505 ReMax (46% less GPU memory than PPO at 7B); 2404.14367 Tajwar et al. and 2405.08448 Tang et al. (on-policy vs offline); 2504.05118 VAPO; 2511.20347 SAPO; 2506.10910 Magistral; 2508.06471 GLM-4.5; 2508.10925 gpt-oss model card; 2601.02780 MiMo-V2-Flash; 2511.22570 DeepSeekMath-V2 (verifier-as-reward for proofs); 2307.15217 Casper et al.; 2310.03716 Singhal et al.; 2310.13548 Sharma et al.; 2503.11926 Baker et al. (CoT monitoring); 2504.02495 DeepSeek-GRM; 2408.15240 generative verifiers; 2211.14275 Uesato et al.; 2410.09724 Leng et al.; 2412.21139 SWE-Gym; 2502.18449 SWE-RL; 2503.09516 Search-R1; 2509.08755 AgentGym-RL; 2509.02547 agentic-RL survey; 2606.12191 agentic-environment survey; 2603.24844 multi-answer RL; 2603.21972; 2602.21534 ARLArena; 2609.29429 Just Ask Jev; 2609.08368 Miles v0.1; 1606.01540 OpenAI Gym.

**DERIVED ARITHMETIC FOR WRITERS.** R1-Zero 512 GPUs x 198 h = 101,376 GPU-h = $202K at $2/h (S15); R1 step = 32 questions x 16 samples = 512, 8,192 outputs = 16 x 512 (S15); Olmo 3 static-batching waste (max - mean)/max with mean 14,628 and max 32K gives ~54% (S48); Olmo 3 step 1000 s of which 125 s training, so generation would need >8x speedup before training becomes the bottleneck (S48); Qwen3-8B RL 17,920 vs OPD 1,800 GPU-hours = 9.96x (S49, S53).

## Sources

### S1 · paper · UC Berkeley — Trust Region Policy Optimization (2015 · arXiv v5 (20 Apr 2017); ICML 2015)

https://arxiv.org/pdf/1502.05477v5 · accessed 2026-10-26 · fetched: true · read depth: full text, §1-§8 and Appendix C

Authors: John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel

Authority: The originating paper of the trust-region policy-gradient line (TRPO), which PPO and GRPO descend from; names the algorithm Aman wrote as "TRPO".

Covers: `rl-foundations`

Facts:
- TRPO is an iterative policy-optimization procedure with a monotonic-improvement guarantee in theory; the practical algorithm approximates the theory. (Abstract) — “We describe an iterative procedure for optimizing policies, with guaranteed monotonic improvement.”
- Theory uses a penalty on total-variation distance (Theorem 1, via Eq. 9); the recommended penalty coefficient from theory gives very small steps, which motivates a constraint instead. (§3 (Theorem 1), §4) — “In practice, if we used the penalty coefficient C recommended by the theory above, the step sizes would be very small.”
- The trust-region form: constrain KL between old and new policy rather than penalize it. (§4, Eq. 11) — “One way to take larger steps in a robust way is to use a constraint on the KL divergence between the new policy and the old policy, i.e., a trust region constraint”
- The max-KL constraint over all states is impractical; the practical algorithm uses the average KL (Eq. 12). (§4) — “Instead, we can use a heuristic approximation which considers the average KL divergence”
- Optimization is second-order-flavoured: conjugate gradient plus a line search, costing only slightly more than the gradient itself. (§6 (Practical Algorithm), Appendix C) — “We use the conjugate gradient algorithm followed by a line search, which is altogether only slightly more expensive than computing the gradient itself.”
- Two sampling schemes: single-path (model-free) and vine (needs resettable states). (§1 (variants defined in §5.1-§5.2)) — “We describe two variants of this algorithm: first, the single-path method, which can be applied in the model-free setting; second, the vine method”
- Step size used in all experiments: KL bound delta = 0.01. (§8) — “We used δ = 0.01 for all experiments.”
- TRPO is described as similar to natural policy gradient methods. (Abstract) — “This algorithm is similar to natural policy gradient methods and is effective for optimizing large nonlinear policies such as neural networks.”

### S2 · paper · UC Berkeley — High-Dimensional Continuous Control Using Generalized Advantage Estimation (2016 · arXiv v6 (20 Oct 2018); ICLR 2016)

https://arxiv.org/pdf/1506.02438v6 · accessed 2026-10-26 · fetched: true · read depth: abstract + §3 estimator definition + hyperparameter ranges

Authors: John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, Pieter Abbeel

Authority: Defines GAE(gamma, lambda), the advantage estimator inside PPO-for-LLMs; the quantity GRPO-family methods replace with group-relative baselines.

Covers: `rl-foundations`, `ppo-for-llms`

Facts:
- GAE reduces policy-gradient variance using a value function, at the cost of some bias, via an exponentially-weighted estimator analogous to TD(lambda). (Abstract) — “We address the first challenge by using value functions to substantially reduce the variance of policy gradient estimates at the cost of some bias, with an exponentially-weighted estimator of the advantage function that is analogous to TD(λ).”
- Both gamma and lambda trade off bias and variance; lambda is the knob specific to GAE. (§3 (text after Eq. 16)) — “both of which contribute to the bias-variance tradeoff when using an approximate value function”
- Best values found in the paper lie in narrow ranges. (§6 (experiments)) — “γ ∈ [0.96, 0.99] and λ ∈ [0.92, 0.99].”

### S3 · paper · OpenAI — Proximal Policy Optimization Algorithms (2017 · arXiv v2 (28 Aug 2017))

https://arxiv.org/pdf/1707.06347v2 · accessed 2026-10-26 · fetched: true · read depth: full text

Authors: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov

Authority: The PPO paper; the RL algorithm in InstructGPT and the base that GRPO modifies.

Covers: `rl-foundations`, `ppo-for-llms`

Facts:
- PPO alternates sampling with optimizing a surrogate objective over multiple minibatch epochs; positioned as TRPO-like benefits with much simpler implementation. (Abstract) — “The new methods, which we call proximal policy optimization (PPO), have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity (empirically).”
- The clipped objective L^CLIP = E[min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t)] with eps = 0.2 as the suggested value. (§3, Eq. 7) — “where epsilon is a hyperparameter, say,”
- The min makes the clipped objective a pessimistic (lower) bound on the unclipped objective; the ratio change is ignored only when it would improve the objective. (§3) — “so the final objective is a lower bound (i.e., a pessimistic bound) on the unclipped objective.”
- The adaptive-KL-penalty variant performed worse than clipping; included as a baseline. (§4) — “In our experiments, we found that the KL penalty performed worse than the clipped surrogate objective”
- Reference hyperparameters (MuJoCo, 1M timesteps): horizon 2048, Adam stepsize 3e-4, 10 epochs, minibatch 64, gamma 0.99, GAE lambda 0.95. (Appendix A, Table 3) — “Horizon (T) Adam stepsize Num. epochs Minibatch size Discount (γ) GAE parameter (λ)”

### S4 · blog · Personal site (OpenAI/Thinking Machines author) — Approximating KL Divergence (2020 · posted 2020-03-07)

http://joschu.net/blog/kl-approx.html · accessed 2026-10-26 · fetched: true · read depth: full post (fetched via curl)

Authors: John Schulman

Authority: The primary source for the k1/k2/k3 KL estimators; the "K3 estimator (Schulman, 2020)" that GRPO, DeepSeek-V3.2 and TRL use for the KL-to-reference term. Author is a PPO/TRPO author.

Covers: `rl-foundations`, `training-dynamics`, `critic-free-rl`

Facts:
- k1 = -log r is unbiased but high variance because it is negative for half the samples. (body) — “However, it has high-variance, as it’s negative for half of the samples, whereas KL is always positive.”
- k2 = (1/2)(log r)^2 is low variance but biased; its expectation is an f-divergence that matches KL to second order. (body) — “An alternative estimator, which has lower variance but is biased, is”
- k3 = (r - 1) - log r is unbiased and always non-negative (distance between log and its tangent). (body) — “This leaves us with the estimator k_3 = (r - 1) - \log”
- Numerical check, q=N(0,1), p=N(0.1,1), true KL 0.005: k1 bias/true 0, stdev/true 20; k2 0.002, 1.42; k3 0, 1.42. For true KL 0.5 (p=N(1,1)): k1 0, 2; k2 0.25, 1.73; k3 0, 1.7. (body, two result tables) — “k3 has even lower standard deviation than k2 while being unbiased, so it appears to be a strictly better estimator.”
- Reverse-direction estimator for KL[p,q]: r log r - (r - 1). (body) — “r \log r - (r - 1)”

Conflicts:
- S45 (DeepSeek-V3.2 §3.1) shows the K3 estimator, while unbiased for the value, has a biased gradient when sampled tokens are far less probable under pi_theta than pi_ref, and corrects it with an importance ratio. The blog analyses estimator value, not gradient.

### S5 · textbook · MIT Press (A Bradford Book) — Reinforcement Learning: An Introduction (2nd edition) (2018 · 2nd edition)

https://mitpress.ublish.com/book/reinforcement-learning-an-introduction-2 · accessed 2026-10-26 · fetched: false · read depth: not fetched; chapter/section numbers taken from search-result summaries of the MIT Press preview and an ETH Zurich ToC scan

Authors: Richard S. Sutton, Andrew G. Barto

Authority: Standard RL textbook (content-standards tier 3). Chapter 13 is the textbook treatment of the policy gradient theorem, REINFORCE and baselines that every method in this course specializes. The author site http://incompleteideas.net/book/the-book-2nd.html failed with a TLS certificate error on 2026-10-26; page text was NOT read.

Covers: `rl-foundations`

Facts:
- Chapter 13 "Policy Gradient Methods" starts at p. 321 of the 2nd edition; 13.4 REINFORCE with Baseline starts at p. 329 (per search-result summary of the MIT Press preview; not verified against the book). (Ch. 13)
- Section numbering reported by the same summary: 13.1 Policy Approximation and its Advantages, 13.2 The Policy Gradient Theorem, 13.3 REINFORCE (not verified against the book). (Ch. 13)

### S6 · paper · OpenAI — Training language models to follow instructions with human feedback (InstructGPT) (2022 · arXiv v1 (4 Mar 2022); NeurIPS 2022)

https://arxiv.org/pdf/2203.02155v1 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1, §3.2-§3.5, Appendix C excerpts

Authors: Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, et al.

Authority: The canonical SFT -> reward model -> PPO pipeline paper; Diogo Almeida (TypeSafe founder, who claims RLCD) is an author.

Covers: `rlhf-pipeline`, `ppo-for-llms`, `reward-modeling`, `reward-hacking`

Facts:
- Headline result: a 1.3B InstructGPT model is preferred to 175B GPT-3 by labelers, with >100x fewer parameters. (§1) — “outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having over 100x fewer parameters.”
- Labeler team: about 40 contractors screened for sensitivity to different preferences. (§1) — “We first hire a team of 40 contractors to label”
- Dataset sizes: SFT about 13k prompts, RM 33k prompts, PPO 31k prompts (PPO prompts API-only). (§3.2) — “The SFT dataset contains about 13k training prompts (from the API and labeler-written), the RM dataset has 33k training prompts (from the API and labeler-written), and the PPO dataset has 31k training prompts (only from the API).”
- Labeler agreement: 72.6 +/- 1.5% among training labelers, 77.3 +/- 1.3% for held-out labelers. (§3.4) — “training labelers agree with each-other 72.6 ± 1.5% of the time, while for held-out labelers this number is 77.3 ± 1.3%.”
- Only 6B reward models were used because 175B RM training was unstable and a 175B value function is costly. (§3.5, Appendix C) — “we found that 175B RM training could be unstable and thus was less suitable to be used as the value function during RL”
- RL is a bandit environment: random prompt in, response out, reward from RM ends the episode; per-token KL penalty from the SFT model mitigates overoptimization; value function initialized from the RM. (§3.5 (RL)) — “In addition, we add a per-token KL penalty from the SFT model at each token to mitigate overoptimization of the reward model. The value function is initialized from the RM.”
- PPO-ptx objective (Eq. 2): RM reward minus beta * log(pi_RL/pi_SFT) plus gamma * pretraining log-likelihood; gamma = 0 gives plain PPO. (§3.5, Eq. 2) — “The KL reward coefficient, β, and the pretraining loss coefficient, γ, control the strength of the KL penalty and pretraining gradients respectively. For "PPO" models, γ is set to 0.”
- "Alignment tax": RLHF regresses on SQuAD, DROP, HellaSwag, WMT15 fr-en; mixing pretraining gradients (PPO-ptx) largely removes it without hurting preference scores. (§1) — “We can greatly reduce the performance regressions on these datasets by mixing PPO updates with updates that increase the log likelihood of the pretraining distribution (PPO-ptx), without compromising labeler preference scores.”

Conflicts:
- S16 (Tulu 3) and S50 (Llama 3) moved away from this PPO-with-learned-RM recipe for general preference tuning, using DPO variants instead.

### S7 · paper · Anthropic — Constitutional AI: Harmlessness from AI Feedback (2022 · arXiv v1 (15 Dec 2022))

https://arxiv.org/pdf/2212.08073v1 · accessed 2026-10-26 · fetched: true · read depth: abstract and §1

Authors: Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, et al.

Authority: Originating paper for RLAIF and principle-driven preference data; the lineage that the older RLCD (S34) tries to improve on.

Covers: `rlaif-cai`, `rlhf-pipeline`

Facts:
- Two phases: supervised (self-critique and revision, then SFT on revisions) and RL phase using AI preference labels (RLAIF). (Abstract) — “self-critiques and revisions, and then finetune the original model on revised responses.”
- Stated aim: a non-evasive, relatively harmless assistant trained without human feedback labels for harms. (§1) — “to train a non-evasive and relatively harmless AI assistant, without any human feedback labels for harms.”
- Result: RL-CAI models sit on a better harmlessness/helpfulness frontier than human-feedback HH RLHF models (Fig. 2). (§1, Figure 2) — “The RL-CAI models trained with AI feedback learn to be less harmful at a given level of helpfulness.”
- Both the SL and RL methods can use chain-of-thought style reasoning to improve legibility and performance. (Abstract) — “Both the SL and RL methods can leverage chain-of-thought style reasoning to improve the”

### S8 · paper · Stanford — Direct Preference Optimization: Your Language Model is Secretly a Reward Model (2023 · arXiv v3 (29 Jul 2024); NeurIPS 2023)

https://arxiv.org/pdf/2305.18290v3 · accessed 2026-10-26 · fetched: true · read depth: abstract, §4-§6, Appendix B hyperparameters

Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn

Authority: Originating DPO paper.

Covers: `direct-preference-optimization`, `rlhf-pipeline`

Facts:
- DPO reparameterizes the RLHF reward so the optimal policy is closed-form, replacing reward-model fitting plus RL with a classification loss; no sampling from the LM during fine-tuning. (Abstract) — “eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning.”
- DPO loss (Eq. 7): -E log sigmoid(beta log pi_theta(y_w|x)/pi_ref(y_w|x) - beta log pi_theta(y_l|x)/pi_ref(y_l|x)). Reward is implicit: r_hat = beta log(pi_theta/pi_ref). (§4, Eq. 7) — “This way, we fit an implicit reward using an alternative parameterization, whose optimal policy is”
- Gradient reweights each pair by how wrong the implicit reward ordering is. (§4) — “higher weight when reward estimate is wrong”
- TL;DR summarization: DPO ~61% win rate at temperature 0.0 vs PPO 57% at its best temperature. (§6.2) — “We find that DPO has a win rate of approximately 61% at a temperature of 0.0, exceeding the performance of PPO at 57% at its optimal sampling temperature of 0.0.”
- Default hyperparameters: beta = 0.1, batch 64, RMSprop, LR 1e-6 with 150-step warmup. (Appendix B) — “Unless noted otherwise, we use a β = 0.1, batch size of 64 and the RMSprop optimizer with a learning rate of 1e-6 by default.”

Conflicts:
- S11 (Xu et al.) argues PPO beats DPO on harder benchmarks; S50 (Llama 3) and S16 (Tulu 3) report DPO competitive at lower cost.

### S9 · paper · Stanford / Contextual AI — KTO: Model Alignment as Prospect Theoretic Optimization (2024 · arXiv v5 (updated 2026-09-08 per arXiv API); ICML 2024)

https://arxiv.org/abs/2402.01306 · accessed 2026-10-26 · fetched: true · read depth: abstract only (via arXiv API)

Authors: Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela

Authority: Originating paper for KTO; learns from a binary desirable/undesirable signal rather than pairs.

Covers: `direct-preference-optimization`

Facts:
- KTO maximizes the utility of generations (Kahneman-Tversky model) rather than log-likelihood of preferences, and learns from a binary signal. (Abstract) — “it matches or exceeds the performance of preference-based methods at scales from 1B to 30B, despite only learning from a binary signal of whether an output is desirable.”
- Claims no single human-aware loss is universally superior; best loss depends on inductive biases of the setting. (Abstract) — “our work suggests that there is no one HALO that is universally superior; the best loss depends on the inductive biases most appropriate for a given setting”

### S10 · paper · arXiv preprint (affiliations not read) — Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning (2025 · arXiv v3 (28 Jan 2026))

https://arxiv.org/abs/2512.19920 · accessed 2026-10-26 · fetched: true · read depth: abstract only (via arXiv API)

Authors: Jiayun Wu, Jiashuo Liu, Zhiyuan Zeng, Tianyang Zhan, et al.

Authority: Independent, public work on training LLMs against strictly proper scoring rules with RL - the closest peer-reviewed-style analogue to what TypeSafe asserts RLCD does (S35); complements RLCR (S32).

Covers: `calibration-rl`, `rlvr`

Facts:
- Argues binary-reward RLVR makes models test-takers that guess whenever correctness probability is above zero. (Abstract) — “Standard RLVR paradigms, utilizing binary reward signals, inadvertently incentivize models as good test-takers rather than honest communicators, encouraging guessing whenever correctness probability exceeds zero.”
- Proposes interventions that optimize strictly proper scoring rules so the model outputs a calibrated probability of correctness, enabling abstention or claim-level flagging. (Abstract) — “we propose and evaluate training interventions optimizing strictly proper scoring rules for models to output a calibrated probability of correctness.”
- Reports a 4B model (Qwen3-4B-Instruct) exceeding GPT-5 on a log-scale Accuracy-to-Hallucination Ratio gain: 0.806 vs 0.207 on BeyondAIME (authors' own evaluation, not independently verified). (Abstract) — “our model's log-scale Accuracy-to-Hallucination Ratio gain (0.806) exceeds GPT-5's (0.207) in a challenging in-domain evaluation (BeyondAIME).”

### S11 · paper · Tsinghua / OpenPsi — Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study (2024 · arXiv v3 (10 Oct 2024); ICML 2024)

https://arxiv.org/abs/2404.10719 · accessed 2026-10-26 · fetched: true · read depth: abstract only (via arXiv API)

Authors: Shusheng Xu, Wei Fu, Jiaxuan Gao, Wenjie Ye, et al.

Authority: The most-cited head-to-head "DPO vs PPO" study; the pro-PPO side of the offline-vs-online debate.

Covers: `direct-preference-optimization`, `ppo-for-llms`, `when-to-use-which`

Facts:
- States the split: deployed products use reward-model + PPO, while academic leaderboards were dominated by DPO. (Abstract) — “Novel applications such as ChatGPT and Claude leverage reward-based methods that first learn a reward model and apply actor-critic algorithms, such as Proximal Policy Optimization (PPO). However, in academic benchmarks, state-of-the-art results are often achieved via reward-free methods, such as Direct Preference Optimization (DPO).”
- Claims DPO has theoretical limitations and that PPO wins on every testbed including code competitions. (Abstract) — “Experiment results demonstrate that PPO is able to surpass other alignment methods in all cases and achieve state-of-the-art results in challenging code competitions.”

Conflicts:
- S50 (Llama 3 §4.1.4) says the opposite on their stack: PPO needed more compute and did worse than DPO, especially on IFEval. S16 (Tulu 3) found PPO roughly equal to DPO at ~7x the wall-clock.

### S12 · paper · Allen AI / UW — Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback (2024 · arXiv v2 (7 Oct 2024); NeurIPS 2024)

https://arxiv.org/abs/2406.09279 · accessed 2026-10-26 · fetched: true · read depth: abstract only (via arXiv API)

Authors: Hamish Ivison, Yizhong Wang, Jiacheng Liu, Zeqiu Wu, et al.

Authority: Controlled ablation of data, algorithm, reward model and prompts; the study Tulu 3 cites when choosing DPO for iteration and PPO for RLVR.

Covers: `direct-preference-optimization`, `ppo-for-llms`, `when-to-use-which`

Facts:
- Ranking of factors: preference data quality matters most, then algorithm, then reward model, then extra unlabeled prompts. (Abstract) — “better preference data leading to the largest improvements, followed by the choice of learning algorithm, the use of improved reward models, and finally the use of additional unlabeled prompts for policy training.”
- PPO beats DPO by small margins. (Abstract) — “Notably, PPO outperforms DPO by up to 2.5% in math and 1.2% in general domains.”

Conflicts:
- Gain sizes (2.5% math, 1.2% general) are small relative to the data-quality effect (up to 8%) reported in the same abstract.

### S13 · paper · Cohere / Cohere For AI — Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs (2024 · arXiv v2 (26 Feb 2024); ACL 2024)

https://arxiv.org/pdf/2402.14740v2 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1-§3 key passages

Authors: Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, Sara Hooker

Authority: The critic-free precursor to GRPO: argues PPO is unnecessary for RLHF and proposes REINFORCE Leave-One-Out (RLOO).

Covers: `critic-free-rl`, `ppo-for-llms`, `when-to-use-which`

Facts:
- Core argument: the reward arrives only for the full generation, so model the whole completion as one action rather than per-token MDP steps. (§1) — “rewards are only attributed to full generations, with no true rewards for any intermediary tokens in the generation. Thus, it is more appropriate and efficient to model the entire generation as a single action with the initial state determined by the prompt.”
- Headline claim: vanilla REINFORCE beats PPO; PPO is over-engineered for pre-trained LLMs. (§1) — “PPO is unnecessarily complicated for a pre-trained LLM environment.”
- Reported margin: REINFORCE outperforms PPO by 3.2% to 20.3% win-rate across datasets and base models (Llama, Pythia; HH and TL;DR). (§1, contribution 1) — “is consistently outperforming PPO by 3.2% to 20.3%”
- RLOO baseline: for each of k samples, the other k-1 samples give an unbiased estimate of expected return for the prompt, "akin to a parameter-free value-function, but estimated at each training step". Cost: more sampling. (§2.3 (RLOO)) — “akin to a parameter-free value-function, but estimated at each training step”
- RLOO is claimed to beat PPO, DPO and RAFT on all datasets and models, and to make better use of online samples than RAFT. (§1, contribution 2) — “we empirically show it consistently outperforms baselines such as PPO, DPO (Rafailov et al., 2023) as well as RAFT (Dong et al., 2023) across all datasets and models.”

Conflicts:
- S11 (Xu et al.) concludes the opposite ranking (PPO > DPO and other alternatives); S12 finds PPO > DPO by small margins. RLOO paper does not compare against a tuned value-model PPO of the S11 kind.

### S14 · paper · DeepSeek-AI / Tsinghua / Peking University — DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (2024 · arXiv v3 (27 Apr 2024))

https://arxiv.org/pdf/2402.03300v3 · accessed 2026-10-26 · fetched: true · read depth: abstract, §4.1-§4.2, §5.2

Authors: Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, et al.

Authority: The paper that introduced GRPO (Feb 2024) - not DeepSeek-R1, which adopted it. Also gives the unified SFT/RFT/DPO/PPO/GRPO gradient view.

Covers: `critic-free-rl`, `grpo-successors`, `rl-foundations`, `rlvr`

Facts:
- GRPO is a PPO variant that drops the critic and takes the baseline from the group of sampled outputs for the same question. (Abstract / §1) — “GRPO foregoes the critic model, instead estimating the baseline from group scores”
- Motivation against PPO: the value model is another policy-sized model (memory + compute), and LLM rewards are given at the last token so a per-token value function is hard to train. (§4.1.1) — “As the value function employed in PPO is typically another model of comparable size as the policy model, it brings a substantial memory and computational burden.”
- Group-relative advantages match how reward models are trained (comparisons between outputs on the same question). (§4.1.1) — “The group relative way that GRPO leverages to calculate the advantages, aligns well with the comparative nature of rewards models”
- KL-to-reference is added to the loss, not to the reward as in PPO/InstructGPT, so it does not enter the advantage. (§4.1.1) — “instead of adding KL penalty in the reward, GRPO regularizes by directly adding the KL divergence between the trained policy and the reference policy to the loss, avoiding complicating the calculation of”
- DeepSeekMath-RL training setup: lr 1e-6, KL coefficient 0.04, 64 outputs per question, max length 1024, batch size 1024, single policy update per exploration stage; about 144K CoT-format questions from GSM8K/MATH SFT data. (§4.2) — “The KL coefficient is 0.04. For each question, we sample 64 outputs. The max length is set to 1024, and the training batch size is 1024.”
- Results from RL on DeepSeekMath-Instruct 7B: GSM8K 82.9% -> 88.2%, MATH 46.8% -> 51.7%, with out-of-domain gains. (Abstract / §1) — “GSM8K: 82.9% → 88.2%, MATH: 46.8% → 51.7%”
- Iterative GRPO retrains the reward model on new policy samples with a 10% replay of historical data and resets the reference model to the policy. (§4.1.4) — “continually train the old reward model using a replay mechanism that incorporates 10% of historical data.”
- Online RFT vs GRPO: GRPO scales its gradient coefficient by reward magnitude, penalizing wrong answers; Online RFT only reinforces correct ones. GRPO beat Online RFT (Fig. 5). (§5.2.1) — “In contrast, Online RFT lacks this feature; it does not penalize incorrect responses and uniformly reinforces all responses with correct answers at the same level of intensity.”
- RL improved Maj@K but not Pass@K on GSM8K and MATH: authors read this as sharpening the output distribution rather than adding capability. (The seed of the S24 / S23 debate.) (§5.2.2, Figure 7) — “it seems that the improvement is attributed to boosting the correct response from TopK rather than the enhancement of fundamental capabilities.”
- Unified paradigm: SFT, RFT, DPO, PPO and GRPO are all special cases of one gradient form distinguished by data source, reward function, and gradient coefficient. (§5.2.1) — “we provide a unified paradigm to analyze different training methods, such as SFT, RFT, DPO, PPO, GRPO”

Conflicts:
- S18 (Dr. GRPO) shows the 1/|o_i| per-sequence normalization and std division in GRPO introduce length and difficulty biases; S45 (DeepSeek-V3.2) later drops std division.

### S15 · paper · DeepSeek-AI — DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arXiv v2 = expanded Nature version) (2025 · arXiv v2 (4 Jan 2026); v1 (22 Jan 2025); Nature 645, 633-638 (17 Sep 2025, DOI 10.1038/s41586-025-09422-z, per search-result confirmation; Nature page itself not fetched - redirects to cookie authorization))

https://arxiv.org/pdf/2501.12948v2 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1-§3, Supplementary A.3, B.3-B.5, F.1, G.2; v1 spot-checked for headline numbers

Authors: DeepSeek-AI (Daya Guo, Dejian Yang, Haowei Zhang, et al.)

Authority: The primary source for the course's mandatory case study. Popularized pure-RLVR GRPO at scale; GRPO itself comes from S14.

Covers: `rlvr`, `case-studies`, `grpo-successors`, `reward-hacking`, `reward-modeling`, `training-dynamics`, `distillation-opd`, `rl-scaling`, `generative-judges-rubrics`

Facts:
- R1-Zero: RL directly on DeepSeek-V3-Base with GRPO, reward based only on final-answer correctness, no SFT stage first. (§1) — “The reward signal is solely based on the correctness of final predictions against ground-truth answers, without imposing constraints on the reasoning process itself. Notably, we bypass the conventional supervised fine-tuning (SFT) phase before RL training.”
- GRPO is credited to Shao et al. 2024 and described as simplifying PPO and cutting its resource use. (§2.1) — “It was originally proposed to simplify the training process and reduce the resource consumption of Proximal Policy Optimization (PPO)”
- R1-Zero hyperparameters: lr 3e-6, KL coefficient 0.001, rollout temperature 1, 16 samples per question, max length 32,768 tokens until step 8.2k then 65,536. (§2.1) — “we set the learning rate to 3e-6, the KL coefficient to 0.001, and the sampling temperature to 1 for rollout. For each question, we sample 16 outputs with a maximum length of 32,768 tokens before the 8.2k step and 65,536 tokens afterward.”
- Run size: 10,400 steps (1.6 epochs); 32 unique questions x 16 = 512 outputs per step; reference model replaced by latest policy every 400 steps; each rollout of 8,192 outputs split into 16 minibatches, one inner epoch. (Arithmetic: 8,192 = 16 x 512, so one rollout feeds 16 optimizer steps.) (§2.1) — “Each training step consists of 32 unique questions, resulting in a training batch size of 512. Every 400 steps, we replace the reference model with the latest policy model.”
- Rule-based rewards only for reasoning: accuracy reward + format reward, equal weight. Neural reward models (outcome or process) deliberately avoided because of hacking. (§2.2) — “Notably, we abstain from applying neural reward models—whether outcome-based or process-based—to reasoning tasks. This decision is predicated on our observation that neural reward models are susceptible to reward hacking during large-scale reinforcement learning.”
- R1-Zero AIME 2024 pass@1 rose from 15.6% to 77.9% (v2); with self-consistency decoding 86.7%. (§2.3) — “jumping from an initial 15.6% to 77.9%.”
- Problems with R1-Zero that motivated the multi-stage R1 pipeline: poor readability, language mixing, narrow reasoning-only RL. (§1) — “it faces challenges such as poor readability and language mixing”
- Language-consistency reward = fraction of target-language words in the CoT; ablation shows a slight performance cost. (§3.2.1, Eq. 7) — “we introduce a language consistency reward during RL training, which is calculated as the proportion of target language words in the CoT.”
- Stage-1 RL of R1 prints "GRPO clip ratio epsilon to 10" and says a low clip value truncates gradients while a high one destabilizes. The value 10 differs from the 0.2 convention (S3) - treat as possibly a typo or a different parameterization. (§3.2.1) — “the GRPO clip ratio 𝜀 to 10”
- Second RL stage: 1,700 steps; preference-model reward only in the final 400 steps because longer exposure caused reward hacking (Supp. B.5). (§3.2.2) — “We find that more training steps with the model based preference reward signal may lead to reward hacking”
- SFT data for the final stage: about 600k reasoning samples (rejection sampling from stage-1 RL checkpoint, correct only; some judged by a generative reward model via DeepSeek-V3) + about 200k non-reasoning samples. (Supp. B.3.3) — “In total, we collect about 600k reasoning-related training samples.”
- Reported compute: R1-Zero on 64x8 H800 GPUs for approximately 198 hours; R1 about 80 hours. Table 7 (at $2/H800-hour): R1-Zero 101K GPU-hours = $202K; SFT data creation 5K = $10K; R1 41K = $82K; total 147K GPU-hours = $294K. Check: 64x8 = 512 GPUs x 198 h = 101,376 GPU-h; 512 x 80 h = 40,960 GPU-h. The text says R1 took "about 4 days, or roughly 80 hours" - 4 days would be 96 h, so treat 80 h (consistent with the 41K GPU-hours in Table 7) as the figure of record. (Supp. B.4.4, Table 7) — “For the training of DeepSeek-R1-Zero, we employed 64*8 H800 GPUs, and the process required approximately 198 hours.”
- Reward hacking observed with the helpful (neural) reward model; reward rose while reasoning performance fell (Figure 6). (Supp. B.5) — “we observe such reward hacking behavior when employing the helpful reward model.”
- Distillation beats small-model RL: on Qwen2.5-32B, 10K+ steps of RL (Qwen2.5-32B-Zero) only matches QwQ-32B-Preview, while distilling R1 is significantly better. (Supp. F.1, Table 16) — “distilling more powerful models into smaller ones yields excellent results, whereas smaller models relying on the large-scale RL mentioned in this paper require enormous computational power and may not even achieve the performance of distillation.”
- Failed attempts - PRM: step granularity is hard to define, step correctness is hard to judge, and any model-based PRM invites reward hacking. (Supp. G.2) — “Third, once a model-based PRM is introduced, it inevitably leads to reward hacking (Gao et al., 2022)”
- Failed attempts - MCTS: token-level search space is exponentially larger than chess; node limits trap the model in local optima. (Supp. G.2) — “token generation presents an exponentially larger search space”
- Future-work caveat: tasks without a reliable verifier remain an open problem; R1 used human-annotated SFT and only hundreds of RL steps there. (Supp. G.1 (Reward Hacking paragraph)) — “for complex tasks that cannot be effectively evaluated by a reliable reward model, scaling up pure RL methods remains an open challenge.”

Conflicts:
- v1 (22 Jan 2025) reports R1-Zero AIME 2024 pass@1 rising from 15.6% to 71.0% (86.7% with majority voting); v2 reports 15.6% to 77.9% with 86.7% via self-consistency. Writers must state which version they cite.
- S18 (Dr. GRPO) finds DeepSeek-V3-Base already shows the "aha moment" before RL, challenging the framing that RL created it.
- S45 (DeepSeek-V3.2) later uses rubric-based generative reward models for general tasks, whereas R1 avoided neural RMs for reasoning.

### S16 · paper · Allen Institute for AI / UW — Tulu 3: Pushing Frontiers in Open Language Model Post-Training (2024 · arXiv v5 (14 Apr 2025); v1 22 Nov 2024)

https://arxiv.org/pdf/2411.15124v5 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1, §5.4.1, §6, §8.1, Table 21 text

Authors: Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, et al.

Authority: Coined and open-sourced RLVR (Nov 2024, two months before R1); fully open recipe with PPO-vs-DPO cost measurements.

Covers: `rlvr`, `ppo-for-llms`, `direct-preference-optimization`, `when-to-use-which`, `case-studies`, `reward-hacking`, `generative-judges-rubrics`

Facts:
- RLVR definition: the RLHF objective with the reward model replaced by a verification function; reward is a constant if the completion is verified. (§1, §6) — “RLVR leverages the existing RLHF objective but replaces the reward model with a verification function”
- RLVR targets skills with ground-truth outcomes: GSM8K, MATH, and precise instruction following (IFEval-style constraints). (§1) — “Our results show that RLVR can improve GSM8K, MATH, and IFEval performance.”
- PPO vs DPO cost in a controlled setup: roughly equal average scores, PPO about 28 h on two nodes vs DPO about 4 h on one node. (§5.4.1) — “The PPO runtime is roughly 28 hours using two nodes, whereas the DPO runtime is about 4 hours using a single node.”
- Decision recorded: do preference iteration with DPO, reserve PPO for RLVR. (§5.4.1) — “We decide to use PPO primarily for RLVR”
- Preference-tuning variant chosen: only length-normalized DPO beat the base checkpoint overall among DPO, SimPO and length-normalized DPO. (§5.4.1, Table 18) — “We found that only length-normalized DPO outperformed our base checkpoint overall”
- Value model for RLVR PPO is initialized from a general reward model; this gave the best GSM8K and average scores. (§6.2.1, Fig. 21) — “Initializing RLVR’s Value Function from a General RM Works Best.”
- More KL budget does not necessarily improve verifiable reward on GSM8K/MATH; large KL leads to average-score drops (overoptimization). (§6.2.1, Fig. 21 caption) — “as the model diverges more from the initial model, overoptimization happens as the average scores drop significantly.”
- Final RLVR hyperparameters: KL penalty coefficient beta = 0.05 for 8B and 0.07 for 70B (omega is the LR warm-up ratio: 0.0 for 8B, 0.07 for 70B). (Appendix, Table 21 caption) — “The final 8B RLVR model used β = 0.05 and ω = 0.0; the final 70B RLVR model used β = 0.07 and ω = 0.07”
- RL infrastructure: asynchronous PPO with dedicated vLLM (PagedAttention) inference GPUs via Ray; scaled to 405B. (§6.3) — “This setup allowed us to scale PPO policy training to the 405B scale.”
- 405B RLVR: only the MATH train set; over 5 MATH points gained within 25 steps; trained only 75 steps due to infrastructure scaling difficulty. (§8.1) — “even with as few as 25 RLVR steps, MATH performance improved by over 5 points”
- Named limitation/future work: the 8B value model; value-free methods such as GRPO untested. (Limitations) — “Future works can benefit from exploring larger value models or alternate value model-free RL algorithms such as GRPO”

Conflicts:
- S11 (Xu et al.) says PPO is stronger than DPO; here PPO was only comparable and 7x slower in wall-clock on the authors' stack.

### S17 · paper · ByteDance Seed / Tsinghua AIR — DAPO: An Open-Source LLM Reinforcement Learning System at Scale (2025 · arXiv v2 (20 May 2025))

https://arxiv.org/pdf/2503.14476v2 · accessed 2026-10-26 · fetched: true · read depth: abstract, §2-§4, Table 1

Authors: Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, et al.

Authority: Source of Clip-Higher, Dynamic Sampling, token-level loss and overlong shaping - the changes TRL (default loss_type=dapo), Olmo 3 and others adopted.

Covers: `grpo-successors`, `training-dynamics`, `rlvr`

Facts:
- Headline: 50 AIME 2024 points on Qwen2.5-32B base, beating DeepSeek-R1-Zero-Qwen-32B (47) in 50% of the training steps. (Abstract, Fig. 1) — “outperforming the previous SoTA DeepSeekR1-Zero-Qwen-32B using 50% training steps.”
- Naive GRPO on the same base reaches only 30 AIME points; problems named: entropy collapse, reward noise, training instability. (§1) — “the naive GRPO baseline suffers from several key issues such as entropy collapse, reward noise, and training instability”
- Four techniques: Clip-Higher, Dynamic Sampling, Token-Level Policy Gradient Loss, Overlong Reward Shaping. (§1) — “1. Clip-Higher, which promotes the diversity of the system and avoids entropy collapse;”
- Ablation ladder on AIME24 avg@32: naive GRPO 30; +overlong filtering 36; +clip-higher 38; +soft overlong punishment 41; +token-level loss 42; +dynamic sampling (full DAPO) 50. (Table 1) — “Main results of progressive techniques applied to DAPO”
- Clip-Higher: raise eps_high (0.28) and keep eps_low (0.2); raising the lower bound would crush low-probability tokens to 0 and collapse the sampling space. (§3.1) — “We keep εlow as it is, because increasing it will suppress the probability of these tokens to 0, resulting in the collapse of the sampling space.”
- Dynamic Sampling: oversample and drop prompts whose accuracy is exactly 0 or 1, since all-equal rewards give zero advantage and shrink the effective batch. (§3.2) — “A zero advantage results in zero policy gradients, shrinking the magnitude and increasing the noise sensitivity of the batch gradient”
- Token-level loss: sample-level averaging underweights tokens in long responses and cannot penalize gibberish in long samples. (§3.3) — “tokens within longer responses (which contain more tokens) may have a disproportionately lower contribution to the overall loss”
- KL term removed: for long-CoT reasoning the policy is expected to diverge substantially, so the restriction is unnecessary. (§2.3) — “during training the long-CoT reasoning model, the model distribution can diverge significantly from the initial model, thus this restriction is not necessary.”
- Training details: verl framework; AdamW constant lr 1e-6 with 20-step warmup; prompt batch 512 x 16 responses; minibatch 512 (16 gradient updates per rollout step); 16,384-token expected max + 4,096 soft-penalty cache = 20,480 max generation; AIME evaluated as avg@32 at temperature 1.0, top-p 0.7. (§4.1) — “For rollout, the prompt batch size is 512 and we sample 16 responses for each prompt.”

Conflicts:
- S23 (ProRL) argues the KL penalty (with periodic reference resets) is a stronger entropy-control tool than DAPO-style tricks when starting from an SFT/distilled checkpoint; DAPO removes KL for base-model RL.
- S18 (Dr. GRPO) fixes length bias by a constant normalizer, not token-level averaging.

### S18 · paper · Sea AI Lab / NUS / SMU — Understanding R1-Zero-Like Training: A Critical Perspective (2025 · arXiv v2 (6 Oct 2025); COLM 2025)

https://arxiv.org/pdf/2503.20783v2 · accessed 2026-10-26 · fetched: true · read depth: abstract, §2, §3.1-§3.2

Authors: Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin

Authority: Source of Dr. GRPO and the length/difficulty-bias analysis of GRPO; also the check on what R1-Zero-style training actually does to base models.

Covers: `grpo-successors`, `training-dynamics`, `rlvr`, `evaluation-of-rl`

Facts:
- Finds the "aha moment" already in DeepSeek-V3-Base and strong Qwen2.5 behaviour without prompt templates - suggesting pretraining priors drive R1-Zero-like gains. (Abstract) — “our analysis reveals that DeepSeek-V3-Base already exhibit “Aha moment”, while Qwen2.5 base models demonstrate strong reasoning capabilities even without prompt templates, suggesting potential pretraining biases.”
- GRPO has a response-level length bias from dividing by |o_i|: for correct answers shorter ones get larger updates; for wrong answers longer ones are penalized less - so wrong responses grow. (§3.1) — “Response-level length bias: This arises from dividing by |oi |.”
- GRPO has a question-level difficulty bias from dividing by the group reward std: questions that are too easy or too hard get higher weight. (§3.1) — “Questions with lower standard deviations (e.g., those that are too easy or too hard, with the outcome rewards being almost all 1 or 0) are given higher weights during policy updates.”
- The length normalization also appears in open-source PPO implementations (loss normalized by response length), predating GRPO. (§3.1) — “all of these implementations normalize the loss by response length”
- Dr. GRPO removes both the 1/|o_i| and the std normalizations (constant normalizer); result: 43.3% AIME 2024 with a 7B base (Qwen2.5-Math-7B) using a minimalist recipe. (Abstract, §3.2) — “a minimalist R1-Zero recipe that achieves 43.3% accuracy on AIME 2024 with a 7B base model”
- Qwen2.5 base models improve ~60% with no template vs R1 template; template mismatch can destroy reasoning before RL starts. (§2.2, §3.3) — “Intriguingly, Qwen-2.5 base models get an immediate ∼ 60% improvement by not using template”

Conflicts:
- S17 (DAPO) fixes the same length problem by token-level averaging across the batch; S18 divides by a constant (e.g., max length). TRL exposes both (S59).
- S15 attributes emergent reflection to RL; S18 shows it in the base model.

### S19 · paper · Qwen Team, Alibaba Inc. — Group Sequence Policy Optimization (2025 · arXiv v2 (28 Jul 2025))

https://arxiv.org/pdf/2507.18071v2 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1-§5

Authors: Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, et al.

Authority: Originating paper for GSPO (sequence-level importance ratio and clipping), used in Qwen3 training.

Covers: `grpo-successors`, `training-dynamics`, `agentic-multi-turn-rl`

Facts:
- Diagnosis: GRPO's token-level importance weights are a misapplication of importance sampling (one sample per token position), injecting high-variance noise that grows with response length and is amplified by clipping, ending in irreversible collapse. (§1) — “we identify that the instability of GRPO stems from the fundamental misapplication and invalidation of importance sampling weights in its algorithmic design.”
- GSPO defines the importance ratio on sequence likelihood, with length normalization, and clips at the sequence level (Eq. 5). (Abstract, §3) — “GSPO defines the importance ratio based on sequence likelihood and performs sequence-level clipping, rewarding, and optimization.”
- Clipping ranges used: GSPO 3e-4 / 4e-4 (left/right); GRPO baseline tuned to 0.2 / 0.27 - the sequence-level log ratio lives on a different scale. (§4) — “In GSPO, we set the left and right clipping ranges in Equation (5) to 3e-4 and 4e-4, respectively.”
- GRPO on MoE needs Routing Replay to converge; GSPO removes the need because sequence likelihood is insensitive to which experts a token routed to. (§5.3) — “GSPO eliminates the dependency on Routing Replay and is fully capable of computing the importance ratios si (θ ) conventionally, converging normally, and optimizing stably.”
- Infrastructure angle: sequence-level likelihood tolerates training-vs-inference engine precision differences better, so rollout log-probs may not need recomputation by the training engine. (§5.4) — “Given the precision discrepancies between training engines (e.g., Megatron) and inference engines (e.g., SGLang and vLLM)”

Conflicts:
- SAPO (arXiv 2511.20347, abstract verified, not entered as a source) argues GSPO suppresses an entire sequence when a few tokens are off-policy and proposes a soft token-adaptive gate.
- S20 (MiniMax-M1) argues the opposite direction: keep every token's gradient and clip the IS weight (CISPO) rather than clipping updates.

### S20 · paper · MiniMax — MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention (2025 · arXiv v1 (16 Jun 2025))

https://arxiv.org/pdf/2506.13585v1 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1, §3.1-§3.2

Authors: MiniMax (Aili Chen, Aonian Li, et al.)

Authority: Source of CISPO and an early, concrete training-vs-inference-kernel mismatch diagnosis (FP32 LM head).

Covers: `grpo-successors`, `training-dynamics`, `rl-scaling`

Facts:
- CISPO clips importance-sampling weights instead of token updates; full RL run of M1 took 3 weeks on 512 H800 GPUs, rental cost $534,700. (Abstract) — “CISPO clips importance sampling weights rather than token updates, outperforming other competitive RL variants.”
- Why PPO/GRPO clipping hurt: rare "fork" tokens (However, Recheck, Wait, Aha) have low probability, high ratio after the first update, get clipped out and stop contributing to later off-policy updates. (§3.1 (Issues of Token Clipping)) — “tokens associated with reflective behaviors (e.g., However, Recheck, Wait, Aha), which often serve as “forks” in reasoning paths, were typically rare and assigned low probabilities by our base model.”
- CISPO: stop-gradient on a clipped importance weight multiplied into a REINFORCE-style term, keeping group-relative advantage and token-level loss; no PPO-style token dropping. (§3.1 (The CISPO Algorithm)) — “Rather than clipping the token updates as in PPO/GRPO, we instead clip the importance sampling weight in Eq. 3 to stabilize training.”
- Controlled study on Qwen2.5-32B: CISPO beats DAPO and GRPO at equal steps and matches DAPO with 50% of the steps (2x speedup). (§3.1, Figure 2) — “achieves comparable performance to DAPO using 50% of the training steps.”
- Training/inference probability mismatch blocked reward growth in the hybrid-attention model; traced layer-by-layer to high-magnitude LM-head activations; fix: FP32 LM head realigned the two probabilities (Figure 3 right panel labels a correlation of 0.987319; the text says it rose from roughly 0.9x). (§3.2) — “To address this, we increased the precision of the LM output head to FP32, thereby realigning the two theoretically identical probabilities”
- The mismatch did not appear in smaller dense softmax-attention models - it is architecture- and kernel-dependent. (§3.2) — “Interestingly, this issue did not appear in smaller, dense models with softmax attention.”

Conflicts:
- S27 (FP16 paper) argues the root cause is BF16 rounding rather than any one layer and that switching to FP16 removes the mismatch; S20 fixes one layer (the LM head) in FP32.

### S21 · paper · Shanghai AI Lab / Tsinghua / UIUC — The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models (2025 · arXiv v1 (28 May 2025))

https://arxiv.org/pdf/2505.22617v1 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1

Authors: Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, et al.

Authority: Gives the empirical law and the covariance mechanism behind entropy collapse; source of Clip-Cov and KL-Cov.

Covers: `training-dynamics`, `grpo-successors`

Facts:
- Entropy collapses early in RL runs without intervention, yielding an over-confident policy and saturating reward. (Abstract) — “where the policy entropy dropped sharply at the early training stage, leading to an overly confident policy model.”
- Empirical law R = -a*exp(H) + b between downstream performance R and entropy H: performance is "traded" from entropy, so the ceiling (H = 0, R = -a + b) is predictable. (Abstract) — “the policy performance is traded from policy entropy, thus bottlenecked by its exhaustion, and the ceiling is fully predictable”
- Mechanism: entropy change is driven by the covariance between action probability and the change in logits (proportional to advantage under policy gradient). High-probability, high-advantage actions reduce entropy; rare, high-advantage actions raise it. (Abstract) — “a high-probability action with high advantage would reduce policy entropy, while a rare action with high advantage would increase policy entropy.”
- Remedies: Clip-Cov (clip updates of high-covariance tokens) and KL-Cov (apply KL penalty to them). (Abstract) — “we propose two simple yet effective techniques, namely Clip-Cov and KL-Cov, which clip and apply KL penalty to tokens with high covariances respectively.”

Conflicts:
- S17 (DAPO Clip-Higher) addresses the same collapse by widening the upper clip; S21 argues for token-selective control by covariance. S26 argues entropy dynamics have an exploitation phase then an exploration phase.

### S22 · paper · Meta / UT Austin / UCL / UC Berkeley / Harvard / Periodic Labs — The Art of Scaling Reinforcement Learning Compute for LLMs (2025 · arXiv v1 (15 Oct 2025))

https://arxiv.org/pdf/2510.13786v1 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1, §3

Authors: Devvrit Khatri, Lovish Madaan, Rishabh Tiwari, Rachit Bansal, et al.

Authority: The first large-scale systematic study of RL scaling recipes (400,000+ GPU-hours); source of the ScaleRL recipe and the sigmoidal compute-performance fit.

Covers: `rl-scaling`, `grpo-successors`, `training-dynamics`, `when-to-use-which`

Facts:
- Study scale and method: more than 400,000 GPU-hours; fits sigmoidal compute-vs-performance curves and ablates common design choices. (Abstract) — “amounting to more than 400,000 GPU-hours, that defines a principled framework for analyzing and predicting RL scaling in LLMs.”
- Three findings: recipes differ in asymptote; loss aggregation, normalization, curriculum and off-policy algorithm mostly change compute efficiency, not the asymptote; stable recipes extrapolate from small runs. (Abstract) — “Details such as loss aggregation, normalization, curriculum, and off-policy algorithm primarily modulate compute efficiency without materially shifting the asymptote”
- ScaleRL = asynchronous PipelineRL, forced length interruptions, CISPO truncated-IS loss, prompt-level loss averaging, batch-level advantage normalization, FP32 logits, zero-variance filtering, no-positive-resampling. (§1) — “ScaleRL combines asynchronous Pipeline-RL setup (§3.1), forced length interruptions, truncated importance sampling RL loss (CISPO), prompt-level loss averaging, batch-level advantage normalization, FP32 precision at logits, zero-variance filtering, and No-Positive-Resampling”
- Validation: one run scaled to 100,000 GPU-hours on an 8B dense model (and 50k on a 17Bx16 MoE), extrapolated from fits up to 50k (16k). (Figure 1 caption) — “We run ScaleRL for 100k GPU hours on an 8B dense model, and 50k GPU hours on a 17Bx16 MoE (Scout).”
- PipelineRL vs PPO-off-policy: similar asymptote, but PipelineRL reaches it faster (less idle time); optimal max off-policyness k = 8. (§3.1) — “PipelineRL and PPO-off-policy achieve similar asymptotic performance A, but PipelineRL substantially improves the compute efficiency B”
- Context for scale: R1-Zero used about 100,000 H800 GPU hours of RL, 3.75% of its pre-training compute; RL compute rose more than 10x from o1 to o3 (as cited by the authors). (§1) — “Deepseek-R1-Zero used 100,000 H800 GPU hours for RL training – 3.75% of its pre-training compute (Guo et al., 2025).”

Conflicts:
- S45 (DeepSeek-V3.2) reports a post-training budget "exceeding 10% of the pre-training cost" - a different regime from the 3.75% R1-Zero figure cited here.

### S23 · paper · NVIDIA — ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models (2025 · arXiv v1 (30 May 2025))

https://arxiv.org/pdf/2505.24864v1 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1-§2

Authors: Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, Yi Dong

Authority: The pro-"RL expands the boundary" empirical case; also the main defense of keeping a KL penalty with reference resets.

Covers: `evaluation-of-rl`, `rlvr`, `training-dynamics`, `rl-scaling`

Facts:
- Claim: prolonged RL (KL control, reference-policy reset, diverse tasks) finds reasoning strategies the base model cannot reach even with extensive sampling. (Abstract) — “prolonged RL (ProRL) training can uncover novel reasoning strategies that are inaccessible to base models, even under extensive sampling.”
- Gains grow with base-model weakness on the task and with training duration; RL helps most where the base model starts weak. (Abstract) — “reasoning boundary improvements correlates strongly with task competence of base model and training duration”
- KL penalty justified here: explicit KL gives a more stable fix for entropy collapse than DAPO-style tricks, starting from a distilled checkpoint; "removal of KL" advice is said to apply to base models before SFT. (§2.2) — “We observe that this perspective often applies to base models prior to any supervised fine-tuning.”
- Result for a 1.5B model: over 2k training steps; average pass@1 improvements over the base of 14.7% (math), 13.9% (coding), 54.8% (logic puzzles), 25.1% (STEM), 18.1% (instruction following). (§1) — “we achieve average pass@1 improvements of 14.7% on math benchmarks, 13.9%”

Conflicts:
- S24 (Yue et al.) finds pass@k at large k shrinks after RLVR and attributes gains to sampling efficiency; S23 finds pass@k improves incl. cases where the base model fails entirely. S26 argues both are right at different training phases. Differences: S23 trains a small distilled model for >2k steps on diverse tasks; S24 evaluates shorter runs across several families.
- S17 (DAPO), S48 (Olmo 3) and S47 (GLM-5) remove the KL term; S23 keeps it with periodic reference resets.

### S24 · paper · Tsinghua LeapLab / SJTU — Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? (2025 · arXiv v5 (24 Nov 2025))

https://arxiv.org/pdf/2504.13837v5 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1

Authors: Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, Gao Huang

Authority: The canonical sceptical pass@k analysis of RLVR (the claim RLVR narrows rather than expands the base model's reasoning boundary).

Covers: `evaluation-of-rl`, `rlvr`

Facts:
- RLVR-trained models beat base models at small k (pass@1) but base models win at large k; the reasoning boundary often narrows with RLVR training. (Abstract) — “We observe that while RLVR-trained models outperform their base models at smaller values of k (e.g., k=1), base models achieve higher pass@k score when k is large.”
- RLVR reasoning paths already lie inside the base model's sampling distribution (coverage and perplexity analysis). (Abstract) — “the reasoning paths generated by RLVR models are already included in the base models’ sampling distribution”
- Six popular RLVR algorithms perform similarly and remain far from using the base model's full potential. (Abstract) — “six popular RLVR algorithms perform similarly and remain far from optimal in fully leveraging the potential of the base model.”
- Distillation, unlike RLVR, can introduce new reasoning patterns from the teacher. (Abstract) — “we find that distillation can introduce new reasoning patterns from the teacher and genuinely expand the model’s reasoning capabilities.”
- Coverage of solvable problems (pass@256) decreases as pass@1 improves during RLVR training (Fig. 1 right). (Figure 1 caption) — “average performance (i.e., pass@1) improves, but the coverage of solvable problems (i.e., pass@256) decreases”

Conflicts:
- S23 (ProRL) and S26 report expansion under longer or phased training; S49 (Qwen3 Table 21) reports on-policy distillation raised pass@64 while RL did not - consistent with S24 for RL, and with S24's distillation point.

### S25 · paper · UW / Allen AI / Berkeley — Spurious Rewards: Rethinking Training Signals in RLVR (2025 · arXiv v2 (25 Feb 2026))

https://arxiv.org/pdf/2506.10947v2 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1

Authors: Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, Simon Shaolei Du, Nathan Lambert, Sewon Min, Ranjay Krishna, Yulia Tsvetkov, Hannaneh Hajishirzi, Pang Wei Koh, Luke Zettlemoyer

Authority: Shows RLVR gains on Qwen2.5-Math persist under random or wrong rewards, forcing a re-read of every "RLVR works on Qwen" result.

Covers: `evaluation-of-rl`, `rlvr`, `reward-hacking`

Facts:
- On Qwen2.5-Math-7B, randomly assigned rewards improved MATH-500 by 21.4 points absolute vs 29.1 points for ground-truth rewards. (Abstract) — “RLVR training with GRPO improves MATH-500 performance for Qwen2.5Math-7B in absolute points by 21.4% using randomly assigned rewards, nearly matching the 29.1% gained with ground truth rewards.”
- Mechanism: GRPO's clip term creates a clipping bias that amplifies high-prior behaviours from pretraining even with uninformative rewards; the behaviour found is "code reasoning" (frequency 65% to over 90%). (Abstract) — “GRPO exhibits a clipping bias arising from the clip term, which can amplify high-prior behaviors learned during pre-training even without informative rewards.”
- The effect is model-dependent: spurious rewards give minimal gain or degradation on Qwen2.5, OLMo2 and Llama3 families. (§1) — “training with spurious rewards yields minimal improvement or even performance degradation (§3).”

Conflicts:
- S48 (Olmo 3 §2.3) cites this paper to argue open RLVR benchmarks on open-weight models with undisclosed mid-training data can be contaminated, which is why Olmo 3 releases RL-Zero from a fully open base.

### S26 · paper · arXiv preprint (affiliations not read) — The Debate on RLVR Reasoning Capability Boundary: Shrinkage, Expansion, or Both? A Two-Stage Dynamic View (2025 · arXiv v1 (5 Oct 2025))

https://arxiv.org/abs/2510.04028 · accessed 2026-10-26 · fetched: true · read depth: abstract only (via arXiv API)

Authors: Xinhao Yao, Lu Yu, Xiaolin Hu, Fengwei Teng, et al.

Authority: Third-party reconciliation of S23 vs S24; a theoretical argument, authors' own assessment rather than consensus.

Covers: `evaluation-of-rl`, `rlvr`

Facts:
- States the debate is unresolved and that studies split into "shrinkage" (S24) and "expansion" (S23) camps. (Abstract) — “The ongoing debate on whether reinforcement learning with verifiable rewards (RLVR) expands or shrinks the reasoning capabilities of large language models (LLMs) remains unresolved.”
- Proposes both are partially valid: an exploitation stage (sharpening on already-explored high-reward tokens) followed by an exploration stage (rare optimal tokens gain probability). (Abstract) — “both perspectives are partially valid-each aligning with a separate phase in an inherent two-stage probability mass dynamic”

### S27 · paper · Sea AI Lab / NUS — Defeating the Training-Inference Mismatch via FP16 (2025 · arXiv v1 (30 Oct 2025))

https://arxiv.org/pdf/2510.26788v1 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1

Authors: Penghui Qi, Zichen Liu, Xiangxin Zhou, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin

Authority: Most direct treatment of the inference-engine vs training-engine numerical mismatch in RL (vLLM/SGLang rollouts vs FSDP/Megatron updates); the topic closest to Aman's inference expertise. Also summarizes the Yao et al. TIS blog (the blog itself, a Notion page, could not be fetched).

Covers: `training-dynamics`, `rl-scaling`

Facts:
- Claim: root cause is floating-point precision; BF16's large rounding error breaks consistency between training and inference policies; reverting to FP16 removes the mismatch with a few lines of change. (Abstract) — “The widely adopted BF16, despite its large dynamic range, introduces large rounding errors that breaks the consistency between training and inference.”
- Why two engines diverge even when mathematically identical. (§1) — “While mathematically identical, these engines produce numerically different outputs due to precision errors and hardware-specific optimizations.”
- Prior algorithmic patch: Yao et al. token-level importance sampling (TIS); it was later shown by Liu et al. to be insufficient due to biased gradient; Liu et al. proposed an unbiased sequence-level ratio (slower to converge due to variance). (§1) — “it was later shown by Liu et al. [2025a] to be insufficient to fully stabilize training due to its biased gradient.”
- Cost of IS correction: an extra forward pass to compute the ratio, estimated at about 25% added training cost; plus a "deployment gap" because parameters are optimized for the training engine's distribution, not the inference engine's. (§1) — “this adds approximately 25% to the training cost.”
- The deployment gap: the final weights are optimal for the training engine, not the inference engine used in serving. (§1) — “the resulting model is not truly optimal for the inference engine used in deployment”

Conflicts:
- S20 (MiniMax-M1) fixes the mismatch with an FP32 LM head only; S47 (GLM-5) uses a deterministic top-k in the DSA indexer and IcePop masking; S45 (DeepSeek-V3.2) uses Keep Routing/Keep Sampling Mask and off-policy sequence masking; S59 (TRL) defaults to sequence-level masked importance sampling. No consensus fix.

### S28 · paper · OpenAI — Scaling Laws for Reward Model Overoptimization (2022 · arXiv v1 (19 Oct 2022); ICML 2023)

https://arxiv.org/pdf/2210.10760v1 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1, §2

Authors: Leo Gao, John Schulman, Jacob Hilton

Authority: The quantitative reference for Goodhart effects in RLHF (gold-vs-proxy reward as a function of KL); cited by DeepSeek-R1 as the basis of its "no neural RMs for reasoning" choice.

Covers: `reward-hacking`, `reward-modeling`, `rlhf-pipeline`

Facts:
- Setup: a fixed "gold" RM stands in for humans and labels data for smaller proxy RMs, so overoptimization can be measured. (Abstract) — “reward model plays the role of humans, providing labels used to train a proxy reward model.”
- Functional forms in d = sqrt(KL(pi || pi_init)): best-of-n R(d) = d(alpha_bon - beta_bon d); RL R(d) = d(alpha_RL - beta_RL log d). Coefficients scale smoothly (about logarithmically) with RM parameter count. (§1) — “We find empirically that for best-of-n (BoN) sampling,”
- RL is slower than best-of-n in KL terms at both optimization and overoptimization, which the authors take as showing KL is an inadequate common yardstick across methods. (§1) — “As a function of the KL divergence, reinforcement learning tends to be slower than best-of-n sampling at both optimization and overoptimization.”
- Proxy RMs of 3M to 3B parameters; gold RM is the 6B InstructGPT RM. (§2) — “The 6B reward model from Ouyang et al. [2022] is used as the gold RM, and our proxy RMs vary from 3M to 3B parameters”

### S29 · paper · Anthropic / Redwood Research — Natural Emergent Misalignment from Reward Hacking in Production RL (2025 · arXiv v1 (23 Nov 2025))

https://arxiv.org/pdf/2511.18397v1 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1, §3 (hack descriptions)

Authors: Monte MacDiarmid, Benjamin Wright, Jonathan Uesato, Joe Benton, et al.

Authority: Primary evidence from a frontier lab that reward hacking in RL environments can generalize to broader misalignment; names the concrete hacks.

Covers: `reward-hacking`, `rl-environments`, `agentic-multi-turn-rl`

Facts:
- Learning to reward hack in production coding RL environments generalized to alignment faking, cooperation with malicious actors, and sabotage in a Claude Code scaffold. (Abstract) — “the model generalizes to alignment faking, cooperation with malicious actors, reasoning about malicious goals, and attempting sabotage when used with Claude Code”
- Chat-style RLHF safety training fixed chat evals but not agentic ones. (Abstract) — “Applying RLHF safety training using standard chat-like prompts results in aligned behavior on chat-like evaluations, but misalignment persists on agentic tasks.”
- Three effective mitigations: prevent the hacking; diversify safety training; "inoculation prompting" (frame hacking as acceptable during training). (Abstract) — “Three mitigations are effective: (i) preventing the model from reward hacking; (ii) increasing the diversity of RLHF safety training; and (iii) “inoculation prompting””
- Named hacks in coding environments: AlwaysEqual object, sys.exit(0) before asserts, pytest report patching via conftest.py. (§3) — “the AlwaysEqual hack, (2) the sys.exit(0) hack, (3) the conftest.py hack”
- Code-sabotage eval: the sabotage attempt rate in the main setting was 12%. (Figure 2 caption) — “In our main setting, we see attemped sabotage 12% of the time”

Conflicts:
- A replication by third parties (search-result summary, not read) found reward hacking reliable but emergent misalignment inconsistent in open models - not verified here.

### S30 · paper · OpenAI — Let's Verify Step by Step (2023 · arXiv v1 (31 May 2023); ICLR 2024)

https://arxiv.org/abs/2305.20050 · accessed 2026-10-26 · fetched: true · read depth: abstract only (via arXiv API)

Authors: Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, et al.

Authority: The key pro-process-supervision result and the PRM800K dataset; the paper R1 and Llama 3 build on or push back against.

Covers: `reward-modeling`, `rlvr`

Facts:
- Process supervision significantly outperforms outcome supervision on MATH for training reliable reward models; best process-supervised model solves 78% of a representative MATH test subset; releases PRM800K (800,000 step labels). (Abstract) — “finding that process supervision significantly outperforms outcome supervision for training models to solve problems from the challenging MATH dataset.”
- Active learning improves the efficacy of process supervision. (Abstract) — “we show that active learning significantly improves the efficacy of process supervision.”

Conflicts:
- S15 (DeepSeek-R1 Supp. G.2) found PRMs impractical at scale and chose rule-based outcome rewards; S50 (Llama 3) still trains stepwise reward models for filtering reasoning traces.

### S31 · paper · Scale AI (affiliation per authors; not re-read) — Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains (2025 · arXiv v2 (3 Oct 2025))

https://arxiv.org/abs/2507.17746 · accessed 2026-10-26 · fetched: true · read depth: abstract only (via arXiv API)

Authors: Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, et al.

Authority: Paper-level treatment of rubric-based on-policy rewards, the pattern now used in production recipes (Kimi K2 self-critic rubrics, DeepSeek-V3.2/V4 rubric GRMs).

Covers: `rlvr`, `generative-judges-rubrics`

Facts:
- RaR extends RLVR beyond verifiable domains using instance-specific rubric feedback as the reward. (Abstract) — “an on-policy reinforcement learning method that extends RLVR beyond verifiable domains by using rubric-based feedback.”
- Reported gains vs direct Likert LLM-judge rewards: up to 31% relative on HealthBench and 7% on GPQA-Diamond (authors' numbers). (Abstract) — “over popular LLM-as-judge baselines that rely on direct Likert-based rewards.”
- Rubrics as structured rewards make smaller judges work better and reduce variance across judge scales. (Abstract) — “using rubrics as structured reward signals yields better alignment for smaller judges and reduces performance variance across judge scales.”

### S32 · paper · MIT — Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty (RLCR = Reinforcement Learning with Calibration Rewards) (2026 · arXiv v2 (15 May 2026); ICLR 2026; v1 22 Jul 2025)

https://arxiv.org/pdf/2507.16806v2 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1, §3 (Theorem 1)

Authors: Mehul Damani, Isha Puri, Stewart Slocum, Idan Shenfeld, Leshem Choshen, Yoon Kim, Jacob Andreas

Authority: The public, peer-reviewed method that trains LLMs with RL on a proper-scoring-rule reward for calibrated confidence; the nearest documented analogue to the calibration objective TypeSafe claims for RLCD. Different name, different method - do not conflate.

Covers: `calibration-rl`, `rlvr`

Facts:
- Diagnosis: binary correctness rewards do not penalize guessing, so RL degrades calibration and increases hallucination elsewhere. (Abstract) — “Because such reward functions do not penalize guessing or low-confidence outputs, they often have the unintended side-effect of degrading calibration and increasing the rate at which LMs generate incorrect responses”
- Reward = binary correctness + Brier score of the verbalized confidence q: R = 1[y == y*] - (q - 1[y == y*])^2 (Eq. 1). The model emits an answer and a numeric confidence after reasoning. (§1, Eq. 1) — “We first prove that this reward function (or any analogous reward function that uses a bounded, proper scoring rule) yields models whose predictions are both accurate and well-calibrated.”
- Theorem 1: the expected reward is maximized when q equals the true success probability (calibration incentive), and among calibrated predictions by the answer with the highest success probability (correctness incentive). (§3, Theorem 1) — “Calibration incentive. For any y, the expected reward E1y≡y∗ RRLCR (y, q, y ∗ ) is maximized when q = py .”
- Log-likelihood is a proper scoring rule but unbounded, so it does not give the correctness incentive; any bounded proper scoring rule can be used. (§1, §3) — “Notably, log-likelihood, though a proper scoring rule, is unbounded and can incentivize models to output incorrect answers (Section 3).”
- Measured: expected calibration error 0.37 -> 0.03 on HotpotQA and 0.26 -> 0.10 on a collection of math datasets, accuracy matched to RLVR. (§1) — “reducing expected calibration error from 0.37 → 0.03 on HotpotQA (Yang et al., 2018) and 0.26 → 0.10 on a collection of math datasets (Section 4).”
- Out of domain, RLVR worsens calibration while RLCR improves it. (§1) — “where RLVR substantially worsens calibration generalization to new domains, RLCR improves calibration”
- Verbalized confidence can be used at test time (confidence-weighted voting) to improve accuracy and calibration. (Abstract) — “verbalized confidence can be leveraged at test time to improve accuracy and calibration via confidence-weighted scaling methods.”

Conflicts:
- Name collision: RLCR (this paper) vs RLCD (S34 = Contrastive Distillation, 2023; S35 = TypeSafe Calibrated Decisions, 2026). RLCR verbalizes confidence in text; TypeSafe says its model returns typed probability distributions without generating text (S35).

### S33 · paper · OpenAI / Georgia Tech — Why Language Models Hallucinate (2025 · arXiv v1 (4 Sep 2025))

https://arxiv.org/pdf/2509.04664v1 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1, §4 (socio-technical mitigation)

Authors: Adam Tauman Kalai, Ofir Nachum, Santosh S. Vempala, Edwin Zhang

Authority: Frontier-lab theory of why binary-graded training and evaluation reward guessing; supplies the incentive argument behind calibration-aware RL rewards.

Covers: `calibration-rl`, `rlvr`, `evaluation-of-rl`

Facts:
- Thesis: hallucinations persist because training and evaluation reward guessing over acknowledging uncertainty. (Abstract) — “language models hallucinate because the training and evaluation procedures reward guessing over acknowledging uncertainty”
- Under binary (0-1) grading abstaining is strictly suboptimal; "I don't know" gets the minimum score while a best guess is optimal. (§4) — “grading, abstaining is strictly sub-optimal. IDK-type responses are maximally penalized while an overconfident “best guess” is optimal.”
- Proposed fix is to change how existing mainstream benchmarks are scored, not to add more hallucination evals; sample prompt text with explicit confidence targets t: mistakes cost t/(1-t) points (t=0.5 -> 1, t=0.75 -> 2, t=0.9 -> 9). (§4) — “Answer only if you are > t confident, since mistakes are penalized t/(1 − t) points, while correct answers receive 1 point, and an answer of “I don’t know” receives 0 points.”

Conflicts:
- S32 (RLCR) and S10 train the fix into the policy via a proper-scoring reward; S33 argues the fix is primarily in how benchmarks grade. Both predict binary-reward RLVR pushes toward guessing.

### S34 · paper · UC Berkeley / Meta AI / UCLA — RLCD: Reinforcement Learning from Contrastive Distillation for Language Model Alignment (2024 · arXiv v3 (16 Mar 2024); ICLR 2024 poster; first posted 24 Jul 2023)

https://arxiv.org/pdf/2307.12950v3 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1-§4, Tables 2-3

Authors: Kevin Yang, Dan Klein, Asli Celikyilmaz, Nanyun Peng, Yuandong Tian

Authority: The older, unrelated "RLCD". Verified as Contrastive Distillation (some indexes print "Contrast Distillation"). Include only as a clearly labelled RLAIF improvement.

Covers: `rlaif-cai`

Facts:
- RLCD builds preference pairs from two contrasting prompts for the same task: a positive prompt p+ encouraging the principle and a negative prompt p- encouraging its violation; no human feedback. (Abstract) — “RLCD creates preference pairs from two contrasting model outputs, one using a positive prompt designed to encourage following the given principles, and one using a negative prompt designed to encourage violating them.”
- Rationale: two different prompts make outputs more distinguishable, so automatic preference labels are cleaner than RLAIF labels on two i.i.d. samples from one prompt. (Abstract, §1) — “Using two different prompts causes model outputs to be more differentiated on average, resulting in cleaner preference labels in the absence of human annotations.”
- Pipeline remains standard RLHF after data generation: train a preference model on the pairs, then PPO against it. (§1, Fig. 1) — “We then use the preference pairs to train a preference model, which is in turn used to improve a base unaligned language model via reinforcement learning.”
- Evaluated on harmlessness, helpfulness and story outlining, aligning LLaMA-7B with preference data simulated by LLaMA-7B or LLaMA-30B. (§1, §4) — “RLCD outperforms RLAIF (Bai et al., 2022b) and context distillation (Huang et al., 2022) baselines across three diverse alignment tasks”
- GPT-4 binary preference, RLCD7B vs RLAIF7B: 84.8% / 71.0% / 85.4% / 78.5% across harmlessness-prompts (harm, help), helpfulness-prompts (help) and outlining (quality) columns; at 30B simulation RLCD30B vs RLAIF30B is near parity (e.g., 35.9% RLCD vs 64.1% RLAIF on outline quality). (Table 3) — “RLCD’s outputs are substantially preferred by GPT-4 across all tasks and data simulation model scales, except comparing RLCD30B to RLAIF30B , where performance is relatively equal.”

Conflicts:
- Unrelated to S35 (TypeSafe RLCD, 2026) despite the shared acronym.

### S35 · official · TypeSafe AI — TypeSafe AI documentation: "AI primer", "Confidence", "Introduction" (docs.typesafe.ai) (2026 · live docs, accessed 2026-10-26 (no version stamp); docs index docs.typesafe.ai/llms.txt lists no page on RLCD training beyond the AI primer)

https://docs.typesafe.ai/introduction/machine-learning-primer.md · accessed 2026-10-26 · fetched: true · read depth: full text of three pages via curl (primer, confidence, introduction); docs index checked for other RLCD pages

Authors: TypeSafe AI

Authority: The only vendor documentation page that defines RLCD. Vendor claims, not independently validated. Contains no reward function, loss, data or evaluation detail.

Covers: `rlcd-typesafe-claims`, `calibration-rl`

Facts:
- Definition: RLCD = "Reinforcement learning for calibrated decisions", described as a third post-training route beside RLHF and RLVR. (AI primer, "Three post-training approaches") — “trains TypeSafe to return decisions and calibrated probabilities instead of generated text.”
- Output contract: no text generation; decisions plus probabilities; higher probability should mean higher chance of being correct. (AI primer, "RLCD and calibrated decisions") — “The model does not generate text.”
- Calibration is defined at the population level, with three illustrative operating points; not a per-answer guarantee. (AI primer, "RLCD and calibrated decisions") — “These rates describe groups of predictions, not a guarantee about any single answer.”
- TypeSafe's critique of RLHF: rewards sycophancy and confident hallucination; causes "mode dropping" (a milder mode collapse). Position: RLHF is fine for chat, but automation needs another objective. (AI primer, "The problems with RLHF") — “RLHF teaches a model to say things that people prefer. That objective works well for chatbots, but it can also reward sycophancy and confident-sounding hallucinations.”
- Positioning claim: human preference and machine trustworthiness are different optimization targets. (AI primer) — “Human preference and machine trustworthiness are different optimization targets.”
- Primer says RLHF "was co-invented by Diogo Almeida", cofounder of TypeSafe (he is an author of S6). (AI primer) — “RLHF was used to train InstructGPT and ChatGPT and was [co-invented by Diogo Almeida]”
- Return fields: Choice -> choice, probabilities, confidence; Score -> score, probabilities, confidence; Noul -> a single yes-probability in [0,1] with no separate confidence. (Introduction, primitives table) — “Jev evaluates typed *questions* against a *state* and returns structured results directly. No text generation, no parsing.”
- Confidence is a deterministic summary statistic of the returned distribution, not a second learned head: for Choice, confidence = (p_max - 1/n)/(1 - 1/n). (Confidence page) — “is a statistic computed from the probability distribution the answer already gives you”
- Worked example: probabilities (0.6, 0.3, 0.1) and (0.6, 0.2, 0.2) both give Choice confidence 0.4, i.e. confidence ignores how the remainder is spread. (Confidence page) — “both have confidence 0.4”
- Questions in one call are evaluated "in parallel and in isolation" against the same state. (Introduction) — “evaluated in parallel and in isolation against the same”
- NOT disclosed anywhere in the docs read: reward function, loss, scoring rule, training data, base model, architecture, calibration metric, calibration measurements, whether RLCD uses human labels or verifiers. (AI primer, Confidence page, Introduction, docs index)

Conflicts:
- S36 (company blog) says Jev "can't hallucinate" and S35 says answers carry calibrated probabilities, while S40 (third-party paper) cites a TypeSafe-reported ECE of 0.246 for Jev; the TypeSafe pages read here contain no calibration measurement.

### S36 · blog · TypeSafe AI — Introducing System One Models & Jev (TypeSafe AI company blog) and four sibling posts (2026 · intro post header date 15 Sep 2026 (page metadata says 19 Oct 2026 for every post - an export timestamp); sibling posts: jev-1-14 (14 Oct), bitterest-lesson (10 Sep), antibenchmaxxing (11 Sep), series-ai (9 Oct))

https://typesafe.ai/blog/introducing-system-one-models-and-jev · accessed 2026-10-26 · fetched: true · read depth: intro post full text via curl; four sibling posts via WebFetch extraction searched for RLCD/training detail (none found)

Authors: Diogo Almeida, founder, TypeSafe (introduction post); other posts unattributed

Authority: The launch announcement that names RLCD (the only primary statement of the name by the company, together with S35). Vendor marketing; numbers are the company's own.

Covers: `rlcd-typesafe-claims`

Facts:
- Launch: TypeSafe AI released its first "System One Model", Jev, on 15 Sep 2026 (early access) after "two years in stealth". (Intro post, header and opening) — “After two years in stealth, countless technical challenges, and research breakthroughs… I am beyond excited to announce that today, TypeSafe AI is releasing our first System One Model”
- The only sentence naming the training method: a new stack of architecture, parallel sampler and "training method we call Reinforcement Learning for Calibrated Decisions (RLCD)". (Intro post, paragraph 3) — “We built a new stack entirely focused on automation: with a new model architecture, parallel sampler for maximum efficiency, and training method we call Reinforcement Learning for Calibrated Decisions (RLCD).”
- Comparison table row: RLHF/RLVR optimize for human preference / verifiable rewards; RLCD optimizes for "calibrated decisions: answers with epistemically honest probabilities on System One tasks". (Intro post, "Frontiers, Old and New" table) — “Calibrated decisions: answers with epistemically honest probabilities on System One tasks.”
- Output type claim: typed, pre-declared structure; "never makes type errors"; every answer carries calibrated probabilities and confidence scores. (Intro post table) — “Type-safe structured values. Possible outputs and structure are defined in advance . The model never makes type errors. All answers are accompanied with calibrated probabilities and confidence scores.”
- Sampling claim: all outputs generated in parallel in a single query (versus sequential token generation). (Intro post table) — “Parallel. Generates all outputs in a single query.”
- Company-reported cost and speed: input $0.042 per million tokens; output free; end-to-end 70 ms to 500 ms vs 3 to 329 s for frontier LLMs; claimed 40x-200x faster. (Intro post table) — “Input tokens: $0.042 / MTok ($42 per billion tokens). Output tokens: FREE (too cheap to meter).”
- Headline 193.6x faster / 444.6x cheaper come from the company's four workflow evals, which they say are at "the higher end of real world gains"; accuracy is scored against the average of two frontier LLMs' outputs (not against ground-truth labels), with a bias the authors acknowledge. (Intro post, Evidence section ("Nuance")) — “We use the average of GPT-6 Astra and Fable 5.1 as the reference answer, which biases answers towards OpenAI and Anthropic’s models.”
- Cardinality: up to 255 options; above that, a two-stage score-then-choose system is used (explains occasional slowdowns in the demo). (Intro post, demo notes) — “Jev supports a cardinality up to 255.”
- FAQ questions "Why was a new training algorithm needed?" and "Where does our training data come from?" are listed without answers on the page as captured. (Intro post FAQ) — “Why was a new training algorithm needed?”
- Sibling posts searched (bitterest-lesson, antibenchmaxxing, series-ai, jev-1-14): none mentions RLCD; series-ai states a $870 million series A at $7.5B valuation led by Andreessen Horowitz (footnote); antibenchmaxxing states the company will not publish standard benchmark tables and will publish dated internal evals later. (Sibling posts (WebFetch extraction))

Conflicts:
- Header date 15 Sep 2026 vs page metadata 19 Oct 2026 (identical on all five posts, so an export artifact).
- S38 (TechCrunch, 18 Sep 2026) quotes the founder as "reinforcement learning from calibrated decisions"; company docs say "for".
- Claims "can't hallucinate" (type-level) vs S38 critic Armin Ronacher: "it delegates the hallucination problem a little bit to the user".

### S37 · talk · Latent Space — Jev: System One models for Prod, not God - with Diogo Almeida, CEO, TypeSafe AI (Latent Space podcast) (2026 · transcript; publication date not captured)

https://latent.space/p/jev/ · accessed 2026-10-26 · fetched: true · read depth: transcript 00:00-02:13 via WebFetch extraction (about 90% of 221,921 characters); last ~22k characters unread; quotes are extraction-tool output, not machine-verified

Authors: swyx (host) with Diogo Almeida

Authority: Founder interview; the only place the founder discusses RLCD at any length, and he declines to explain the mechanism. Primary for what is NOT public.

Covers: `rlcd-typesafe-claims`

Facts:
- On RLHF: the founder argues mode dropping was the overlooked downside. ([00:06:03]) — “One of the things that no one paid attention to was the downsides of RLHF, in particular mode dropping.”
- One-line characterization of the three regimes. ([01:20:09]) — “RLHF is please humans, right?”
- RLVR characterized as benchmark optimization; RLCD as reliability for programmatic use. ([01:20:09]) — “And RLCD is make it reliable for, programmatic use.”
- Not published: when the host notes RLCD is a "secret sauce", the founder answers that it is not yet published (extraction-tool reading of the exchange at 00:24:59-00:25:05); elsewhere he says he would like to blog on it eventually ([00:06:21]). ([00:25:05]) — “No, not yet.”
- On algorithm: PPO is not the interesting part. ([00:27:18]) — “It's not about the PPO. That part doesn't matter.”
- Calibration is not claimed to be perfect: when pressed, the founder says the model gets many things wrong and that he hopes calibration lets users cascade to bigger models. ([01:07:33]-[01:10:28]) — “it will get many things wrong, to be clear.”
- Training data is synthetic (host: "all your data is synthetic"; reply "Yep"). ([00:22:04]) — “Yep.”
- Declines to discuss ML internals. ([00:09:10]) — “I probably shouldn't talk too much about the insides of ML”
- Rejects public benchmarks; claims private internal evals only. ([01:15:19]) — “I don't care about the benchmarks”
- Mechanism NOT given in the portion read: no reward function, scoring rule, loss, base model, architecture, or calibration metric. (whole excerpt)

Conflicts:
- Extraction-tool context for [00:25:05] is ambiguous (the preceding question was a remark, not a question); treat "not yet published" as consistent with the absence of any paper but verify against the full transcript before quoting.

### S38 · analysis · TechCrunch — A new kind of AI model from a ChatGPT inventor is thrilling developers (TechCrunch) (2026 · published 18 Sep 2026, 11:49 AM PDT)

https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/ · accessed 2026-10-26 · fetched: true · read depth: article text via WebFetch extraction; quotes are extraction-tool output

Authors: Tim Fernholz

Authority: Independent reporting with named outside users; the only third-party critique of the Jev/RLCD claims in the sources found. News, labelled analysis.

Covers: `rlcd-typesafe-claims`

Facts:
- TechCrunch names the technique "reinforcement learning from calibrated decisions" (preposition "from", never the acronym RLCD); model learns only from artificially generated data per the founder. (Training-method paragraph) — “reinforcement learning from calibrated decisions”
- Founder on synthetic data. (Interview quote) — “We made an early bet that we will be making all of our data”
- Founder on objectives: the problem is optimizing for human language. (Interview quote) — “The problem is we are optimizing for human language”
- Independent critique from Armin Ronacher (CTO, Earendil). (Outside critique) — “At the end of the day, it delegates the hallucination problem a little bit to the user,”
- Model size not disclosed; the company declined to detail the architecture; unnamed observers suspect an open-weight LLM base (reporter's account); no published benchmarks or independent evaluations cited. (Model-size and architecture paragraphs)
- One tester (Bryo CTO) found Gemini slightly more accurate than Jev on an email-classification test. (Outside critique)

Conflicts:
- "for" (company docs) vs "from" (article) calibrated decisions.

### S39 · paper · affiliation not stated on the paper (personal email addresses); no TypeSafe affiliation claimed — OpenJev-RLCD: A Working RLCD Implementation (2026 · arXiv v1 (30 Sep 2026); code https://github.com/ZimmyGao/openjev-rlcd)

https://arxiv.org/pdf/2609.38850v1 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1-§3 (method), headline results

Authors: Zhimin Gao, Pichao Wang

Authority: The only paper that gives RLCD a concrete, checkable objective - but it is an independent re-derivation from public statements, not TypeSafe's method. Self-reported by a small team, 1.7B model, one preprint.

Covers: `calibration-rl`, `rlcd-typesafe-claims`

Facts:
- Explicit provenance: says TypeSafe markets the capability under the RLCD name without disclosing a training algorithm; this paper builds its own. (§1) — “TypeSafe AI’s Jev [17] markets exactly this capability under the name reinforcement learning for calibrated decisions (RLCD), without disclosing a training algorithm.”
- Objective: sample a rationale, read the answer distribution the model commits to afterwards (softmax over option tokens), score each rationale with a strictly proper scoring rule (Brier or log). (Abstract, §2) — “The model samples a rationale and we read the answer distribution it commits to afterwards; every rationale is scored by a strictly proper scoring rule of that distribution.”
- Variance identity (Prop. 1): scoring the mixture of samples equals the mean per-rationale score plus a bonus for disagreeing rationales; RLVR is that objective without its diversity term. (Abstract, §3.1) — “shows that RLVR is exactly the mixture objective without its diversity term.”
- Recipe: "calibrate, then reinforce" - first train the post-rationale readout with a proper score, then reinforce the reasoning with the same proper score as reward, with a KL anchor at the calibrated model. (Abstract, §3.2) — “The resulting recipe is calibrate, then reinforce”
- Failure modes found: the objective can switch reasoning off ("System-One collapse") and the policy-gradient term is 7-12x larger than the readout's gradient. (§1 contributions) — “whose gradient is 7–12× larger than the readout’s and dominates the optimizer”
- Result (Qwen3-1.7B, 3 seeds, paired tests): matches or beats SFT, RFT/STaR and GRPO (each temperature-scaled) on accuracy and Brier; on GSM8K answer verification, a single query decides 81% of items at <=5% error vs 19% for GRPO. (Abstract) — “on GSM8K answer verification a single RLCD query can decide 81% of the items at ≤5% error, versus 19% for GRPO.”
- Limit: when uncertainty is annotator disagreement the objective provably cannot beat cross-entropy and learns to stop reasoning. (Abstract, §5.3) — “the same objective provably cannot beat”

Conflicts:
- Differs from S32 (RLCR): scores the next-token answer distribution after a sampled rationale rather than a verbalized confidence. Neither is TypeSafe's disclosed method.

### S40 · paper · single author; affiliation not read — Calibrated Decision Models for Autonomous Penetration-Testing Harnesses: JEV and Laya as System One Decision Layers for LLM-Driven Pentest Agents (2026 · arXiv v1 (24 Sep 2026))

https://arxiv.org/pdf/2609.28940 · accessed 2026-10-26 · fetched: true · read depth: abstract, §II (RLCD definition), §III, Table IX (Jev vs Laya)

Authors: Joas Antonio dos Santos Barbosa

Authority: Third-party usage of the RLCD term by a security practitioner; useful for what outsiders believe about it, weak as evidence. States its own contributions as exploratory.

Covers: `rlcd-typesafe-claims`

Facts:
- Defines RLCD as training against strictly proper scoring rules (Brier given as canonical). (§II-B) — “trains against strictly proper scoring rules [24]—loss functions where the unique optimal prediction”
- Acknowledges the method is undisclosed for Jev; an open model "Laya" is said to document Brier-score training (Laya not fetched here). (§II-B) — “TypeSafe describes Jev as trained with an undisclosed RLCD methodology [8]; Laya documents training against the Brier score explicitly [9].”
- Asserts (without TypeSafe confirmation) that a System One model is non-autoregressive and evaluates all questions in one forward pass. (§III) — “A System One model differs from a generative LLM in a fundamental architectural way: it is non-autoregressive.”
- Says TypeSafe reports ECE 0.246 for Jev on its own evaluation set vs 0.081 for Laya (post temperature scaling); notes the values come from different tasks and are not comparable. I could not find the 0.246 figure on any TypeSafe page read. (§XII-B) — “TypeSafe reports ECE of 0.246 for Jev on their own evaluation set [8].”
- Proposes RLHV (Reinforcement Learning from Human Verification): Brier score between model probability and deterministic validator verdict. Design only; no implementation. (§II-A, §XIV) — “Rave is a design proposal; no implementation or evaluation has been conducted.”
- Own experiment is exploratory: 13-vulnerability web target, no statistical significance claimed. (Abstract)

Conflicts:
- Architecture claim (non-autoregressive) and the ECE figure are attributed to TypeSafe by this paper but are not in the TypeSafe pages read (S35, S36).

### S41 · paper · Moonshot AI — Kimi k1.5: Scaling Reinforcement Learning with LLMs (2025 · arXiv v4 (3 Jun 2025); v1 22 Jan 2025 (same day as DeepSeek-R1))

https://arxiv.org/pdf/2501.12599v4 · accessed 2026-10-26 · fetched: true · read depth: abstract, §2.3, long2short, partial-rollout infrastructure excerpts

Authors: Kimi Team (Angang Du, Bofei Gao, Bowei Xing, et al.)

Authority: Independent contemporary of R1 with a different (non-GRPO) algorithm: online policy mirror descent with a mean-reward baseline, no value function, length penalty, partial rollouts.

Covers: `case-studies`, `grpo-successors`, `rl-scaling`, `effort-control`, `critic-free-rl`

Facts:
- Design stance: simple RL framework without MCTS, value functions or process reward models. (Abstract) — “without relying on more complex techniques such as Monte Carlo tree search, value functions, and process reward models.”
- Reported results matching o1: 77.5 AIME, 96.2 MATH-500, 94th percentile Codeforces, 74.9 MathVista. (Abstract) — “77.5 on AIME, 96.2 on MATH 500, 94-th percentile on Codeforces, 74.9 on MathVista—matching OpenAI’s o1.”
- Algorithm: a variant of online policy mirror descent; at iteration i the current model is the reference in a relative-entropy-regularized objective; practical gradient uses the empirical mean of k sampled rewards as the baseline and a squared-error surrogate (off-policy-friendly). (§2.3.2) — “We apply a variant of online policy mirror decent as our training algorithm”
- Baseline choice: the empirical mean of sampled rewards worked well in practice (approximating tau*log Z). (§2.3.2) — “We also find that using empirical mean of sampled rewards”
- Overthinking: response length grows during RL; fix is a length reward computed per problem across the k samples, warmed up because it slows early training. (§2.3.3) — “We observe an overthinking phenomenon that the model’s response length significantly increases during RL training.”
- Length penalty is introduced after standard policy optimization (warm-up) because applying it from the start slows training. (§2.3.3) — “In our preliminary experiments, length penalty may slow down training during the initial phases.”
- Curriculum: easy tasks first; plus prioritized sampling toward problems with low success. (§2.3.4) — “We start by training on easier tasks and gradually progress to more challenging ones.”
- Partial rollouts: split long responses across iterations so long trajectories do not stall the batch. (§3 (infrastructure)) — “Partial rollouts is a key technique”
- long2short: transfer long-CoT strength to short-CoT models via model merging, shortest-rejection-sampling, DPO, and long2short RL. (§2.4) — “Specifically, our approaches include applying length penalty with long-CoT activations and model merging.”

Conflicts:
- Same release day as S15 (22 Jan 2025). Algorithmically distinct from GRPO: no group-std normalization, squared-error surrogate with KL-like term.

### S42 · paper · Moonshot AI — Kimi K2: Open Agentic Intelligence (2025 · arXiv v2 (3 Feb 2026); v1 28 Jul 2025)

https://arxiv.org/pdf/2507.20534v2 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1, §3 (agentic data synthesis and RL framework)

Authors: Kimi Team (Yifan Bai, Yiping Bao, Y. Charles, et al.)

Authority: Primary description of a production RL system combining a verifiable-rewards gym, a self-critic rubric reward for non-verifiable tasks, and synthesized tool-use environments.

Covers: `case-studies`, `rl-environments`, `agentic-multi-turn-rl`, `effort-control`, `generative-judges-rubrics`

Facts:
- RL framework: Gym-like and extensible, with many verifiable-reward tasks; subjective tasks use a self-critic pairwise reward where the model judges its own outputs. (§3 intro to RL (§3.2)) — “For tasks that rely on subjective preferences, such as creative writing and open-ended question answering, we introduce a self-critic reward in which the model performs pairwise comparisons to judge its own outputs.”
- Prompt-set difficulty rule: neither too easy nor too hard; difficulty estimated by SFT-model pass@k and only moderate problems kept. (§3.2.1) — “The RL prompt-set should be neither too easy nor too hard, both of which may produce little signal and reduce learning efficiency.”
- K2 reuses the K1.5 policy-optimization algorithm as its foundation. (§3.2.3 (RL Algorithm)) — “We adopt the policy optimization algorithm introduced in K1.5 [36] as the foundation for K2.”
- Budget control: per-sample maximum token budget by task type; over-budget responses are truncated and penalized. (§3.2.3 (Budget Control)) — “we enforce a per-sample maximum token budget throughout RL training, where the budget is determined based on the type of task.”
- Temperature decay: high temperature early for exploration, lower later for reliability. (§3.2.3 (Temperature Decay)) — “we employ a temperature decay schedule, to shift from exploration to exploitation throughout the training.”
- PTX loss: an auxiliary loss on hand-selected high-quality data to prevent forgetting during joint RL. (§3.2.3 (PTX Loss)) — “we curate a dataset comprising hand-selected, high-quality samples and integrate it into the RL objective through an auxiliary PTX loss”
- Self-critique rubric reward is closed-loop: the critic is continually refined on verifiable-reward rollouts so subjective judgment is grounded in verifiable signal. (§3.2.2 (Closed-Loop Critic Refinement)) — “During RL training, the critic model is refined using verifiable signals.”
- Agentic environment synthesis: 3,000+ real MCP tools from GitHub plus 20,000+ evolved synthetic tools; thousands of agents; each task paired with an explicit rubric; LLM-judge filtering. (§3.1.1) — “First, we directly fetch 3000+ real MCP (Model Context Protocol) tools from GitHub repositories”
- Each synthesized task is paired with an explicit rubric specifying success criteria and expected tool-use patterns. (§3.1.1) — “Each task is paired with an explicit rubric that specifies success criteria, expected tool-use patterns, and evaluation checkpoints.”

Conflicts:
- S43/S44 later move the loss away from the K1.5-derived objective (token-level log-ratio clipping, per-token regularization).

### S43 · paper · Moonshot AI — Kimi K2.5: Visual Agentic Intelligence (2026 · arXiv v2 (updated 7 Aug 2026); v1 2 Feb 2026)

https://arxiv.org/pdf/2602.02276v2 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1, §3 (PARL), §4 RL algorithm section, Toggle

Authors: Kimi Team (Tongtong Bai, Yifan Bai, Yiping Bao, et al.)

Authority: Documents a loss change (token-level log-ratio clipping), GRM rewards, a length-budget alternation (Toggle) and multi-agent RL where only the orchestrator is trained.

Covers: `case-studies`, `grpo-successors`, `agentic-multi-turn-rl`, `effort-control`, `training-dynamics`, `generative-judges-rubrics`

Facts:
- Loss departs from K1.5: token-level clipping to bound off-policy divergence caused by training-vs-inference framework discrepancies. (RL algorithm section) — “This loss function departs from the policy optimization algorithm used in K1.5 [31] by introducing a token-level clipping mechanism designed to mitigate the off-policy divergence amplified by discrepancies between training and inference frameworks.”
- Mechanism: a gradient mask on tokens whose log-ratio lies outside [alpha, beta]; unlike PPO it depends only on the log-ratio and ignores the sign of the advantage. (RL algorithm section) — “policy gradients are computed normally for tokens with log-ratios within the interval [α, β ], while gradients for tokens falling outside this range are zeroed out.”
- Key difference from PPO clipping. (RL algorithm section) — “a key distinction from standard PPO clipping [50] is that our method relies strictly on the log-ratio to explicitly bound off-policy drift, regardless of the sign of the advantages.”
- Reward hygiene: multiple alternative GRM rubrics are used to limit reward hacking and overfitting to one preference signal. (Reward function section) — “To mitigate reward hacking and overfitting to a single preference signal, we employ multiple alternative GRM rubrics tailored to different task contexts.”
- Toggle alternates (every m iterations) between a budget-limited phase and standard scaling phase; budget = rho-th percentile length of correct samples; reduces output tokens 25-30% with negligible performance impact. (Toggle section) — “Toggle decreases output tokens by 25∼30% with a negligible impact on performance.”
- PARL: decoupled architecture with a trainable orchestrator and frozen subagents from fixed checkpoints; outputs of subagents treated as environment observations to avoid credit-assignment ambiguity and instability. (Agent Swarm / PARL section) — “The PARL framework adopts a decoupled architecture comprising a trainable orchestrator and frozen subagents instantiated from fixed intermediate policy checkpoints.”
- Critical steps (by analogy with a critical path) replace total steps as the resource constraint so parallelism is rewarded. (PARL section) — “critical steps”
- Zero-vision SFT: text-only SFT is enough to activate visual reasoning and tool use before RL. (§1) — “During post-training, we introduce zero-vision SFT—text-only SFT alone activates visual reasoning and tool use.”

Conflicts:
- S44 (K3) says it follows "the algorithm in Kimi K2.5" for policy optimization and adds partial-rollout staleness tolerance.

### S44 · paper · Moonshot AI — Kimi K3: Open Frontier Intelligence (2026 · arXiv v2 (7 Aug 2026); v1 27 Jul 2026)

https://arxiv.org/pdf/2607.24653v2 · accessed 2026-10-26 · fetched: true · read depth: abstract, §4.1 (post-training), §4.2 (RL environments)

Authors: Kimi Team (Tongtong Bai, Yifan Bai, Yiping Bao, et al.)

Authority: Most recent flagship technical report read: three-stage SFT -> domain-expert RL at several reasoning efforts -> multi-teacher on-policy distillation; white-box harness environments.

Covers: `case-studies`, `frontier-2026`, `distillation-opd`, `rl-environments`, `agentic-multi-turn-rl`, `effort-control`, `rl-scaling`, `generative-judges-rubrics`

Facts:
- Model scale. (Abstract / §1) — “We introduce Kimi K3, a native multimodal Mixture-of-Experts model with 2.8 trillion total parameters, 104 billion”
- Post-training paradigm: SFT -> domain experts via RL at varying reasoning effort -> consolidation by multi-teacher on-policy distillation (MOPD). (§4.1) — “developing specialized domain experts at varying reasoning effort via Reinforcement Learning (RL), and consolidating these domain-specific policies into a single model using Multi-Teacher On-Policy Distillation (MOPD).”
- 3 domains x 3 efforts {low, high, max} = 9 experts; scaling RL FLOPs improves knowledge, reasoning, vision, agent and coding capability. (§4.1.2) — “Crossing these three domain experts with three reasoning effort levels in {low, high, max} yields a total of nine expert models.”
- Partial rollouts: pause generation as soon as a fraction lambda of trajectories completes; unfinished ones resume next iteration, so a trajectory spans several policy versions (staleness). (§4.1.2 (Algorithm)) — “Rather than waiting for all rollouts to terminate, the generation phase pauses as soon as a fraction”
- Staleness tolerance is delegated to the loss: per-token regularization constrains updates to a local neighborhood. (§4.1.2) — “Our policy optimization algorithm inherently tolerates such an extreme off-policy regime through a per-token regularization.”
- Reasoning-effort RL: per-problem token budget from the cold-start model; trajectories exceeding tau*b0 get reward -1; tau annealed from max-budget variant down to high and low efforts. (§4.1.2) — “Training follows a stage-wise curriculum over the budget multiplier τ .”
- Agentic generative reward model for non-verifiable tasks: mandatory judge protocol (read output, generate rubric, score each candidate, write scorepad); tournament-style group reward with binary comparisons; verbosity guard. (§4.1.2) — “the agentic judge is required to follow a mandatory protocol: (1) read the outcome, product, or text output; (2) generate a rubric; (3) score each candidate against the rubric; and (4) record the rubric-assigned scores in a scorepad.”
- Verbosity guard against reward hacking: a candidate longer than sigma*l0 automatically loses the comparison. (§4.1.2) — “a candidate whose output length exceeds σ · ℓ0 automatically loses the binary comparison.”
- White-box environment: harness configured from composable modules (tools, prompts, context policy, skills, memory, subagents) to avoid overfitting to one harness. (§4.2.1) — “Training with a single fixed agent harness can cause a model to overfit to a particular tool schema, system prompt, context management mechanism, or interaction protocol.”

Conflicts:
- S45/S46 keep GRPO; K-series use a different objective lineage (S41).

### S45 · paper · DeepSeek-AI — DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models (2025 · arXiv v1 (2 Dec 2025))

https://arxiv.org/pdf/2512.02556v1 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1, §3 (post-training), §3.1 (Scaling GRPO), §3.2

Authors: DeepSeek-AI (Aixin Liu, Aoxue Mei, Bangcai Lin, et al.)

Authority: The most detailed public statement of what had to change in GRPO to scale RL to >10% of pre-training compute, including inference/training mismatch handling for MoE and sampling.

Covers: `case-studies`, `grpo-successors`, `training-dynamics`, `rl-scaling`, `rl-environments`, `agentic-multi-turn-rl`, `generative-judges-rubrics`

Facts:
- Compute: RL protocol allocates a post-training budget above 10% of pre-training cost. (§1) — “this framework allocates a post-training computational budget exceeding 10% of the pre-training cost”
- Agentic task synthesis: over 1,800 distinct environments and 85,000 complex prompts drive agent RL. (§1) — “we generate over 1,800 distinct environments and 85,000 complex prompts.”
- Pipeline: specialist distillation (math, programming, logic, general agent, agentic coding, agentic search; thinking and non-thinking), then mixed RL. (§3) — “mathematics, programming, general logical reasoning, general agentic tasks, agentic coding, and agentic search, with all the domains supporting both thinking and non-thinking modes.”
- Mixed RL: reasoning, agent and alignment RL merged into one stage with GRPO to avoid catastrophic forgetting of multi-stage training. (§3 (Mixed RL Training)) — “we merge reasoning, agent, and human alignment training into one RL stage. This approach effectively balances performance across diverse domains while circumventing the catastrophic forgetting issues commonly associated with multi-stage training paradigms.”
- Rewards: rule-based outcome + length penalty + language consistency for reasoning/agent; a generative reward model with per-prompt rubrics for general tasks. (§3 (Mixed RL Training)) — “For general tasks, we employ a generative reward model where each prompt has its own rubrics for evaluation.”
- Advantage drops std division: A = R_i - mean(R). (§3.1) — “calculated by subtracting the average reward of the group from the reward of output”
- Unbiased KL estimate: correct the K3 estimator with the importance ratio pi_theta/pi_old; the original K3 gradient puts unbounded weight on tokens where pi_theta << pi_ref. KL strength is domain-dependent; for math a weak or no KL can be best. (§3.1) — “For certain domains, such as mathematics, applying a relatively weak KL penalty or even omitting it entirely can yield improved performance.”
- Off-policy sequence masking: mask whole sequences whose policy divergence (measured with the inference engine's returned probabilities) exceeds delta, only when advantage is negative. (§3.1) — “It is also worth noting that we only mask sequences with negative advantages.”
- Keep Routing (MoE): reuse the sampling-time expert routing in training; adopted since DeepSeek-V3-0324. (§3.1) — “we preserve the expert routing paths used during sampling in the inference framework and enforce the same routing paths during training”
- Keep Sampling Mask: reuse top-p/top-k truncation masks in the training forward pass so pi_old and pi_theta share the same action subspace; preserves language consistency. (§3.1) — “we preserve the truncation masks during sampling from 𝜋old and apply them to 𝜋𝜃 during training, ensuring both policies share identical action subspaces.”
- V3.2-Speciale: reasoning-only RL data with a reduced length penalty, plus DeepSeekMath-V2 proof data and reward. (§3) — “This model was trained exclusively on reasoning data with a reduced length penalty during RL.”

Conflicts:
- S46 (DeepSeek-V4) replaces this mixed-RL stage with on-policy distillation of specialists.
- S18 (Dr. GRPO) vs original GRPO: V3.2 adopts the no-std advantage.

### S46 · paper · DeepSeek-AI — DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence (2026 · arXiv v1, id 2606.19348 (arXiv API reports published 2026-04-26 although the id month is 06 - unexplained; the Lambert book cites the report via the Hugging Face PDF instead))

https://arxiv.org/pdf/2606.19348v1 · accessed 2026-10-26 · fetched: true · read depth: abstract, §5 (post-training pipeline, specialists, GRM, OPD)

Authors: DeepSeek-AI (Anyi Xu, Bangcai Lin, Bing Xue, et al.)

Authority: The current DeepSeek flagship. Primary evidence that the mixed-RL stage was replaced by on-policy distillation with >10 teachers, and that the actor doubles as a generative reward model.

Covers: `case-studies`, `frontier-2026`, `distillation-opd`, `effort-control`, `generative-judges-rubrics`

Facts:
- Pipeline mirrors V3.2 except mixed RL is replaced by OPD. (§5.1) — “the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD; Gu et al., 2024; Lu and Lab, 2025).”
- Specialists: SFT then RL with GRPO per domain, hyperparameters kept close to V3.2/R1. (§5.1.1) — “For the RL stage, we implemented the Group Relative Policy Optimization (GRPO) algorithm, maintaining hyper-parameters closely aligned with our prior research”
- Three reasoning-effort modes per model, trained with distinct length penalties and context windows; "Think Max" adds a system-prompt instruction. (§5.1.1, Table 2-3) — “DeepSeek-V4-Pro and DeepSeek-V4-Flash both support three specific reasoning effort modes.”
- Hard-to-verify tasks use a generative reward model trained jointly: no scalar reward models; rubric-guided RL data; RL applied to the GRM itself; the actor is the GRM. (§5.1.1 (Generative Reward Model)) — “Crucially, we apply RL optimization directly to the GRM itself.”
- Claimed benefit: joint optimization of judging and generating, with only a minimal set of diverse human annotations. (§5.1.1) — “this approach achieves superior performance with only a minimal set of diverse human annotations”
- OPD: student samples its own trajectories and is trained to minimize reverse KL to domain-expert teachers, weighted per expert; more than ten teachers. (§5.1.2, Eq. 29) — “In this stage, more than ten teacher models covering various domains are employed to distill a single student model.”
- Full-vocabulary OPD instead of the cheap sampled-token KL: the token-level estimate (reusing the RL stack with per-token advantage) is resource-efficient but high-variance and unstable. (§5.1.2) — “Although this approach is resource-efficient, it leads to high variance in gradient estimation and often causes training instability.”
- Why OPD over weight-merging or mixed RL: logits-level alignment avoids the degradation of those approaches. (§5.1.2) — “practically circumventing the performance degradation often encountered in traditional weight-merging or mixed RL techniques”
- Engineering of full-vocab OPD: cache only last-layer teacher hidden states, load one teacher head at a time, order samples by teacher index. (§5.2.2) — “we order training samples by teacher index during data dispatching.”

Conflicts:
- arXiv id/date inconsistency (2606.19348 vs published 2026-04-26) - writers should cite the report by title and arXiv id and avoid dating it from the id.

### S47 · paper · Zhipu AI / Tsinghua — GLM-5: from Vibe Coding to Agentic Engineering (2026 · arXiv v2 (24 Feb 2026); v1 17 Feb 2026)

https://arxiv.org/pdf/2602.15763v2 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1, §3 (Reasoning RL, Agentic RL excerpts)

Authors: GLM-5 Team (Aohan Zeng, Xin Lv, et al.)

Authority: Documents the asynchronous agent-RL stack and the specific inference-engine determinism fix (top-k in a sparse-attention indexer) that rescued RL stability - directly in the inference/training boundary.

Covers: `case-studies`, `training-dynamics`, `agentic-multi-turn-rl`, `rl-environments`, `frontier-2026`

Facts:
- Three RL stages in order, with on-policy cross-stage distillation to limit forgetting. (§1) — “starting with Reasoning RL, followed by Agentic RL, and finishing with General RL.”
- Reasoning RL = GRPO + IcePop to handle training-inference mismatch; KL term removed to speed up improvement. (§3.2 (Reasoning RL, RL algorithm backbone)) — “Compared to the original IcePop formulation, we remove the KL regularization term to accelerate RL improvement.”
- Inference-engine determinism matters: non-deterministic CUDA/TileLang top-k in the DSA indexer caused drastic degradation after a few RL steps with an entropy drop; torch.topk (slower but deterministic) fixed it; indexer frozen during RL. (§3.2 (Reasoning RL, DSA RL insights)) — “In contrast, other non-deterministic top-k operators (e.g., CUDA or TileLang implementations) caused drastic performance degradation during RL after only a few steps, accompanied by a sharp drop in entropy.”
- Storing indexer top-k indices for replay (as with MoE routing replay) was judged impractical: k = 2048 is far larger than MoE top-k. (§3.2 (Reasoning RL, DSA RL insights)) — “since the k = 2048 used by the indexer is much larger than the k typically used in MoE”
- Async agentic RL: decoupled inference and training engines with a central Multi-Task Rollout Orchestrator because synchronous RL idles GPUs on long agent rollouts. (§3.3 (Agentic RL)) — “Naive synchronous RL suffers from severe GPU idle time during long-horizon agent rollouts.”
- Token-in-Token-out (TITO) gateway removes re-tokenization mismatch by keeping exact action-level correspondence. (§3.3) — “a Token-in-Token-out (TITO) gateway eliminates re-tokenization mismatches by”
- Direct Double-sided Importance Sampling: token-level clipping [1-eps_l, 1+eps_h] on rollout log-probs; controls off-policy bias without tracking historical policy checkpoints. (§3.3) — “while efficiently controlling off-policy bias without tracking historical policy checkpoints.”
- Environments: over 10K real-world SWE environments plus terminal and multi-hop search tasks. (§3.3) — “we scale verifiable training environments across three domains: over 10K real-world Software Engineering (SWE)”

Conflicts:
- Contrast with S45 (V3.2): both are DSA models; V3.2 solves MoE routing by routing replay (Keep Routing), GLM-5 solves indexer selection by deterministic top-k instead of replay.

### S48 · paper · Allen Institute for AI — Olmo 3 (2025 · arXiv v2 (14 Apr 2026); v1 15 Dec 2025)

https://arxiv.org/pdf/2512.13961v2 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1-§2, §4.4 (OlmoRL algorithm + infrastructure), RL-Zero motivation

Authors: Team Olmo (Allyson Ettinger, Amanda Bertsch, et al.)

Authority: The only fully open end-to-end recipe (data, code, logs) with a consolidated list of GRPO modifications and measured RL-infrastructure numbers.

Covers: `case-studies`, `grpo-successors`, `training-dynamics`, `rl-scaling`, `rlvr`, `when-to-use-which`, `evaluation-of-rl`

Facts:
- OlmoRL = GRPO plus: zero-gradient filtering, active sampling, token-level loss, no KL loss, clip-higher, truncated importance sampling, no std normalization. (§4.4.1) — “No KL loss: We remove the KL loss as a common practice (GLM-4.5 Team et al., 2025; Yu et al., 2025; Liu et al., 2025b) as it allows less-restricted policy updates, and removing it does not lead to over-optimization or destabilized training.”
- No-std advantage to remove the difficulty bias. (§4.4.1) — “No standard deviation normalization: When calculating advantage, we do not normalize by the standard deviation of the group, following Liu et al. (2025b).”
- Truncated importance sampling corrects the inference-engine (vLLM) vs training probability gap, following Yao et al. (§4.4.1) — “To adjust for differences between log probabilities from the inference and training engines, we multiply the loss by the truncated importance sampling ratio, following Yao et al. (2025).”
- Token-level loss normalizes over all tokens in the batch, to avoid length bias. (§4.4.1) — “We use a token-level loss to normalize the loss by the total number of tokens across the batch (Yu et al., 2025), rather than per-sample to avoid a length bias.”
- Offline difficulty filtering: drop prompts with pass rate above 62.5%; for 32B they rely on active sampling instead. (§4.4.2) — “We then remove all samples that the model easily solves (that is, those with a pass rate greater than 62.5%).”
- Inference dominates: rollouts up to 32K tokens (average over 10K); 32B run used 8 H100 nodes for training and 20 for inference; learner waits 75% of the time; ~5x GPU use for inference vs training (7B: ~14x). (§4.4.3) — “Given the cost of autoregressive inference, our learner spends 75% of the time waiting for data, so in terms of GPU utilization, we use approximately 5x as much for inference as for training.”
- Continuous batching avoids waste: at 32K max and 14,628 mean generation length, up to 54% of compute would be wasted by static batching. (§4.4.3) — “which means that up to 54% of our compute would have been wasted with static batching.”
- Infrastructure payoff: continuous batching + inflight updates gave two times faster training on half as many GPUs; 7B run went from about 15 days to 6 days. (§4.4.3, Table 23) — “allowing us to achieve two times faster training on half as many GPUs”
- Footnote arithmetic: an average step took 1000 s, of which 125 s was training; generation was the bottleneck (would take >8x faster generation to make training the bottleneck). (§4.4.3, footnote 41) — “each training step averaged 1000 seconds, of which 125 seconds was spent running training.”
- RL-Zero: open-weight bases with undisclosed mid-training data can contaminate RLVR benchmarks, making spurious rewards as effective as true ones; Olmo 3 releases RL-Zero from a fully open base with decontaminated RL data. (§1) — “which makes spurious rewards as effective as true reward”

Conflicts:
- S23 (ProRL) retains KL with resets; Olmo 3 removes KL. S22 (ScaleRL) says many of these choices move efficiency, not asymptote.

### S49 · paper · Alibaba Qwen Team — Qwen3 Technical Report (2025 · arXiv v1 (14 May 2025))

https://arxiv.org/pdf/2505.09388v1 · accessed 2026-10-26 · fetched: true · read depth: abstract, §4 (post-training), §4.5 Table 21

Authors: An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, et al.

Authority: Primary source for a four-stage reasoning/non-thinking pipeline and for the cleanest published RL-vs-on-policy-distillation compute comparison at fixed starting checkpoint.

Covers: `case-studies`, `distillation-opd`, `effort-control`, `when-to-use-which`, `evaluation-of-rl`, `generative-judges-rubrics`

Facts:
- Flagship pipeline has four stages: long-CoT cold start, reasoning RL, thinking-mode fusion, general RL; small models use strong-to-weak distillation instead. (§4) — “the flagship models in the Qwen3 series follow a sophisticated four-stage training process.”
- Reasoning RL used only 3,995 query-verifier pairs with GRPO; large batch and many rollouts per query plus off-policy training helped. (§4.2) — “We ultimately collect a total of 3,995 query-verifier pairs, and employed GRPO (Shao et al., 2024) to update the model parameters.”
- Query-verifier selection criteria: not used in cold start; learnable; as challenging as possible; broad coverage. (§4.2) — “The query-verifier pairs used in the Reasoning RL stage must satisfy the following four criteria”
- General RL: a reward system over more than 20 tasks with three reward types - rule-based, model-based with reference answer (Qwen2.5-72B-Instruct as judge), model-based without reference (preference RM). (§4.4) — “we have established a sophisticated reward system covering over 20 distinct tasks, each with customized scoring criteria.”
- Why keep rule-based rewards: high precision, prevents hacking. (§4.4) — “Well-designed rule-based rewards can assess the correctness of model outputs with high precision, preventing issues like reward hacking.”
- Agent ability trained with multi-turn rollouts using real environment execution feedback. (§4.4) — “During the RL rollout, the model is allowed to perform complete multi-turn interaction cycles with real environment execution feedback”
- Strong-to-weak distillation (off-policy then on-policy) needs about 1/10 of the GPU hours of the four-stage pipeline. (§4 (intro to post-training)) — “requiring only 1/10 of the GPU hours compared to the four-stage training method.”
- Qwen3-8B, same starting checkpoint: RL reaches AIME'24 67.6 / AIME'25 55.5 with 17,920 GPU hours and pass@64 unchanged (90.0 / 83.3); on-policy distillation reaches 74.4 / 65.5 with 1,800 GPU hours and pass@64 93.3 / 86.7. (§4.7 (Discussion), Table 21) — “distillation achieves significantly better performance than reinforcement learning while requiring approximately only 1/10 of the GPU hours.”
- RL did not improve pass@64 in that experiment; distillation did. (§4.7 (Discussion)) — “In contrast, reinforcement learning does not lead to any improvement in pass@64 scores.”

Conflicts:
- S53 (Thinking Machines) re-uses this table; S53 notes its own SFT baseline (60% AIME'24) differs from the 55.0 off-policy distillation row here.
- Consistent with S24 (RL does not raise pass@k at large k) and in tension with S23.

### S50 · paper · Meta — The Llama 3 Herd of Models (2024 · arXiv v3 (23 Nov 2024); v1 31 Jul 2024)

https://arxiv.org/pdf/2407.21783v3 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1, §4.1-§4.2 (post-training)

Authors: Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. (Llama Team, AI @ Meta)

Authority: The strongest primary evidence for the "DPO instead of PPO at scale" choice by a frontier lab. Also the already-cited source of Module 36 (system-design).

Covers: `case-studies`, `direct-preference-optimization`, `when-to-use-which`, `rlhf-pipeline`, `reward-modeling`

Facts:
- Stated design stance: simple post-training (SFT, rejection sampling, DPO) over RL algorithms that "tend to be less stable and harder to scale". (§1) — “that tend to be less stable and harder to scale.”
- Tried PPO; DPO needed less compute at large scale and did better, especially on IFEval. (§4.1.4) — “We also explored on-policy algorithms such as PPO (Schulman et al., 2017), but found that DPO required less compute for large-scale models and performed better, especially on instruction following benchmarks like IFEval (Zhou et al., 2023).”
- DPO data is as on-policy as practical: most recent preference batches collected with the best models of the previous round. (§4.1.4) — “As a result, our training data conforms better to the distribution of the policy model that is being optimized in each round.”
- Hyperparameters: lr 1e-5, beta 0.1; mask formatting tokens out of the DPO loss to stop tail repetition; add an NLL term with coefficient 0.2 on chosen sequences. (§4.1.4) — “For Llama 3, we use a learning rate of 10−5 and set the β hyper-parameter to be 0.1.”
- NLL regularizer on chosen responses stabilizes DPO and prevents chosen log-probability from decreasing. (§4.1.4) — “We add an additional negative log-likelihood (NLL) loss term with a scaling coefficient of 0.2 on the chosen sequences”
- The loop is repeated for six rounds, each collecting new preference and SFT data from the latest model. (§4.1.6 (Iterative Rounds)) — “Following Llama 2, we apply the above methods in six rounds.”
- Rejection sampling with PagedAttention gave over 2x throughput. (§4.2.2) — “this leads to a throughput improvement of over 2× during rejection sampling.”
- Outcome and stepwise reward models are used to filter reasoning traces for SFT. (§4.3.3 (Math and Reasoning)) — “We train outcome and stepwise reward models”

Conflicts:
- S11 (Xu et al.) and S12 report PPO ahead of DPO; S50 reports the reverse on its stack and attributes it to compute and stability.
- Reasoning RL in 2025 (S15) moved the field away from this stance for verifiable domains.

### S51 · paper · NVIDIA — Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning (2026 · arXiv v1 (12 Jun 2026))

https://arxiv.org/pdf/2606.15007v1 · accessed 2026-10-26 · fetched: true · read depth: abstract, §3.2-§3.3 (RLVR, MOPD, limitations)

Authors: NVIDIA (Aaron Blakeman, Aaron Thomas, et al.)

Authority: Open weights, data, recipes and RL environments from a vendor that also builds the training stack (NeMo-RL, NeMo Gym); candid about MOPD limitations on long-horizon tasks.

Covers: `case-studies`, `frontier-2026`, `distillation-opd`, `rl-environments`, `agentic-multi-turn-rl`, `effort-control`

Facts:
- Scale and post-training stages. (Abstract) — “We introduce Nemotron 3 Ultra, a 550 billion total and 55 billion active parameter Mixture-of-Experts Hybrid Mamba-Attention language model.”
- RL stage is one unified RLVR run across all environments (terminal, office, SWE, search, tool-calling, math, code, STEM, safety, chat, IF, long-context QA, structured outputs). (§3.2) — “we conduct a unified RLVR (Reinforcement Learning with Verifiable Reward) training stage spanning all available environments”
- Harness diversity is used to reduce overfitting to a particular harness. (§3.2) — “reducing overfitting to any particular harness design.”
- Algorithm and scale: asynchronous GRPO with stability optimizations; global batch 8192 with 16 rollouts per sample; 48K-token max generation, later 64K. (§3.2) — “we use a global batch size of 8192, with each sample generating 16 rollouts.”
- Why MOPD after mixed RLVR: as environments multiply, each domain gets few samples per batch, diluting its learning signal. (§3.3) — “diluting the per-domain learning signal and making it increasingly difficult to balance training across domains.”
- More than ten domain teachers; student rollouts get dense per-token rewards from the matching teacher; MOPD run asynchronously. (§3.3) — “we train more than ten specialized teacher models”
- Reported limitation: end-to-end agentic rollouts in MOPD cause large inefficiency; they use single-turn rollouts (PivotRL-style) for most agentic tasks. Whether end-to-end rollouts would gain more is open. (§3.3 (open questions)) — “In practice, we use single-turn rollouts, similar to PivotRL (Yi et al., 2026), for most of the agentic tasks.”
- Releases training recipes, data and RL environments with the checkpoints. (§1) — “we are also open-sourcing the training recipes”

Conflicts:
- S45 (DeepSeek-V3.2) argues mixed RL plus specialist distillation avoids forgetting; S51 runs mixed RLVR first and then MOPD from >10 teachers.

### S52 · paper · Xiaomi (internship work per footnote) — MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training (2026 · arXiv v1 (29 Jun 2026))

https://arxiv.org/pdf/2606.30406v1 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1, §3 (method overview)

Authors: Wenhan Ma, Jianyu Wei, Liang Zhao, Hailin Zhang, et al.

Authority: The standalone method paper for MOPD (also the first MOPD usage, MiMo-V2-Flash, arXiv 2601.02780, whose abstract was verified but is not entered as a separate source).

Covers: `distillation-opd`, `frontier-2026`, `when-to-use-which`

Facts:
- Problem: integrating multiple RL-trained capabilities into one model; existing options are inefficient or lose performance. (Abstract) — “Existing methods, such as Off-Policy Finetune and Mix-RL, are either inefficient or lose performance.”
- Method: per-domain RL teachers, then distill into one student on the student's own rollouts; removes exposure bias and gives a dense signal. (Abstract) — “then distill these teachers into the student on its own rollouts. This eliminates exposure bias and provides a dense optimization signal.”
- Taxonomy of alternatives: Mix-RL (cross-domain interference, "see-saw"), Cascade RL (earlier skills decay), Off-Policy Finetune on teacher rollouts (exposure bias), Param-Merge (unstable). (§1) — “Mix-RL (Yang et al., 2025) pools prompts from every domain into one dataset and runs joint RL, but cross-domain training signals interfere, producing the see-saw effect”
- Table 1 axes: only MOPD is dense, on-policy, and parallelisable. (Table 1 caption) — “MOPD is the only method that simultaneously attains dense optimization, on-policy training, and a parallelisable pipeline.”
- Per-token objective: the student minimizes per-token reverse KL to the prompt-routed domain teacher along its own trajectory. (§3) — “the student is updated by minimising the per-token reverse KL between the student and the domain teacher along the trajectory.”
- Result on Qwen3-30B-A3B: beats Mix-RL, Cascade RL, Off-Policy Finetune and Param-Merge, inheriting nearly all of each teacher's capability; teachers can be developed in parallel. (Abstract) — “inheriting nearly all of each teacher's capability.”
- Same-origin teachers are critical for stable optimization; multi-round refinement helps. (§1) — “confirms that same-origin teachers are critical for stable optimization”

Conflicts:
- Lambert (S60 §3.2.4) states "labs are far from unanimous" on MOPD; GLM-5 (S47) uses a 3-stage RL before distillation; DeepSeek-V3.2 (S45) used mixed RL.

### S53 · blog · Thinking Machines Lab — On-Policy Distillation (Thinking Machines blog) (2025 · posted 27 Oct 2025; June 2026 update notes the original teacher/student models were retired and recipes now use Qwen3.5)

https://thinkingmachines.ai/blog/on-policy-distillation/ · accessed 2026-10-26 · fetched: true · read depth: full post via WebFetch extraction (quotes are extraction output, <=125 chars; not machine-verified)

Authors: Kevin Lu in collaboration with others at Thinking Machines

Authority: The reference cited by DeepSeek-V4 and others ("Lu and Lab, 2025") for on-policy distillation as an RL-shaped objective with dense per-token reward. Builder-org blog with FLOP accounting.

Covers: `distillation-opd`, `rl-scaling`, `when-to-use-which`, `frontier-2026`

Facts:
- Definition. (Intro / method) — “sample trajectories from the student model and use a high-performing teacher to grade each token of each trajectory”
- Objective: per-token reverse KL; discount factor zero; advantage = negative reverse KL, so it plugs into an RL trainer. (Method) — “We set the per-token advantage to the negative reverse KL”
- Taxonomy: SFT = off-policy + dense; RL = on-policy + sparse; on-policy distillation = on-policy + dense. (Comparison table)
- Information-theoretic argument: RL teaches a fixed number of bits per episode; distillation O(N) bits. (Discussion) — “teaches O(N) bits per episode”
- Qwen3-8B-Base student, Qwen3-32B teacher: on-policy distillation reaches 70% AIME'24 in about 150 steps vs SFT needing ~2M prompts (extrapolated) for ~70%. (Math reasoning experiment) — “on-policy distillation achieves an AIME’24 of 70% in about 150 steps”
- Cost: 9x reduction when an SFT dataset is given, about 18x in GPU hours, about 30x including teacher sampling cost (FLOP-based). (Compute table) — “the total cost reduction is about 30x”
- Self-distillation to recover behaviour after mid-training: AIME recovered in <10 gradient steps vs 70 for RL; 50-100x compute efficiency claim. (Self-distillation experiment) — “approximately 7-10x fewer gradient steps, which corresponds to a compute efficiency of 50-100x”
- Caveats stated by the page: only sampled-token reverse KL tested; no variance/seeds; GPU-hour figures taken from the Qwen3 report; teacher mismatch in the FLOPs comparison. (Extraction review)

Conflicts:
- Blog SFT baseline is 60% AIME'24; Qwen3 Table 21 (S49) reports 55.0 for the off-policy-distilled starting point.

### S54 · official · Farama Foundation — Gymnasium documentation: Basic Usage (Env, reset, step) (2026 · live docs (no version stamp))

https://gymnasium.farama.org/introduction/basic_usage/ · accessed 2026-10-26 · fetched: true · read depth: full page via WebFetch extraction (quotes are extraction output; not machine-verified)

Authors: Farama Foundation

Authority: Canonical definition of the environment interface (Env with reset/step, terminated vs truncated) that RL-for-LLM environment libraries borrow (OpenEnv explicitly cites Gymnasium).

Covers: `rl-environments`, `rl-foundations`

Facts:
- An environment is a Python class representing an MDP (acknowledged as an imperfect reconstruction). (Basic usage) — “At the core of Gymnasium is Env, a high-level python class representing a markov decision process (MDP)”
- Interface: reset returns (observation, info); step returns (observation, reward, terminated, truncated, info). (Basic usage, agent-environment loop) — “observation, reward, terminated, truncated, info = env.step(action)”
- terminated = task completed or failed; truncated = time limit; either ends the episode. (Basic usage) — “If either terminated or truncated are True, we end the episode.”

### S55 · paper · Open-Thought / Scale AI / UCL / GitHub — Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards (2025 · arXiv v2 (20 Oct 2025))

https://arxiv.org/pdf/2505.24760v2 · accessed 2026-10-26 · fetched: true · read depth: abstract, §1-§2 (design principles)

Authors: Zafir Stojanovski, Oliver Stanley, Joe Sharratt, Richard Jones, Abdulhakeem Adefioye, Jean Kaddour, Andreas Köpf

Authority: Concrete example of procedurally generated, parametrically scaled verifiable environments - the "environment as a generator + verifier" pattern.

Covers: `rl-environments`, `rlvr`

Facts:
- Library of over 100 task generators and verifiers across algebra, arithmetic, computation, cognition, geometry, graph theory, logic and games. (Abstract) — “It provides over 100 data generators and verifiers spanning multiple domains including algebra, arithmetic, computation, cognition, geometry, graph theory, logic, and various common games.”
- Key property: effectively unlimited data with adjustable difficulty, unlike fixed datasets. (Abstract) — “Its key innovation is the ability to generate virtually infinite training data with adjustable complexity, unlike most previous reasoning datasets, which are typically fixed.”
- Design principle P3: configurable parameters set problem difficulty, enabling dynamic curricula. (§2) — “Configurable parameters systematically control problem characteristics, enabling dynamic curricula via precise difficulty adjustment.”
- Findings: sharp performance cliffs when difficulty rises; adaptive curricula beat fixed difficulty; RLVR generalizes within a domain. (§1 (contributions)) — “Curriculum RLVR results in improved final models, with adaptive difficulty progression outperforming fixed-difficulty training across reasoning tasks.”

### S56 · repo · Meta-PyTorch — OpenEnv: isolated execution environments for agentic RL (meta-pytorch/OpenEnv README) (2026 · README on main, accessed 2026-10-26 (no tag fetched); BSD 3-Clause; README states "early development, APIs may still change")

https://raw.githubusercontent.com/meta-pytorch/OpenEnv/main/README.md · accessed 2026-10-26 · fetched: true · read depth: README via WebFetch extraction (quotes are extraction output; not machine-verified)

Authors: Meta-PyTorch with Hugging Face and partners

Authority: Industry-consortium standard attempt for RL environment interfaces (technical committee includes Meta-PyTorch, Hugging Face, Prime Intellect, NVIDIA, Microsoft, Modal, Unsloth).

Covers: `rl-environments`, `agentic-multi-turn-rl`

Facts:
- Scope and API. (README intro) — “An end-to-end framework for creating, deploying and using isolated execution environments for agentic RL,”
- Gymnasium-style API over a WebSocket, sync or async. (README) — “reset(), step() and state(), sync or async, over a WebSocket.”
- Isolation: each environment is a Docker image (local, cloud sandbox, or Hugging Face Space); MCP environments expose tools at /mcp in production mode. (README) — “Each environment is a Docker image that runs locally, on a cloud sandbox, or as a Hugging Face Space.”
- Catalog claim: 40+ ready environments; named starters: Echo, Coding, TextArena, OpenSpiel, BrowserGym, Harbor. (README) — “40+ ready-to-use environments”
- Training integrations listed: ART, Lightning AI, Miles, Oumi, SkyRL, torchforge, TorchTitan-RL, TRL, Unsloth. (README)
- Maturity caveat. (README) — “OpenEnv is in early development, so APIs may still change.”

### S57 · repo · NVIDIA — NeMo Gym: library for evaluating and improving models and agents using environments (NVIDIA-NeMo/Gym README) (2026 · README on main shows release v0.7.0 dated 10/06/2026 (date format unlabeled))

https://raw.githubusercontent.com/NVIDIA-NeMo/Gym/main/README.md · accessed 2026-10-26 · fetched: true · read depth: first 100,000 of 265,074 characters via WebFetch extraction (quotes are extraction output; not machine-verified)

Authors: NVIDIA

Authority: The environment framework behind Nemotron training (S51), with an explicit definition of an environment as dataset + harness + verifier + state.

Covers: `rl-environments`, `agentic-multi-turn-rl`

Facts:
- Definition of an environment. (README) — “An environment is the complete system an agent interacts with to complete a task.”
- Components: dataset of tasks, agent harness, verifier, per-task state. (README) — “It consists of a dataset (tasks to solve), an agent harness (how the model interacts with the world),”
- Reward shapes in shipped examples: Blackjack +1/0/-1; Bird SQL binary by unordered result-set equality; GUI coordinate with smooth quadratic reward by Euclidean distance; multi-reward tutorials. (README environment table) — “Binary reward from unordered result-set equality.”
- Trainer-agnostic: usable with NeMo RL, Unsloth, VeRL. (README) — “Train with the RL framework of your choice”
- Production claim. (README) — “Battle-tested in production Nemotron training”

### S58 · paper · NTU — Group-in-Group Policy Optimization for LLM Agent Training (GiGPO) (2025 · arXiv v3 (28 Oct 2025); NeurIPS 2025)

https://arxiv.org/abs/2505.10978 · accessed 2026-10-26 · fetched: true · read depth: abstract only (via arXiv API)

Authors: Lang Feng, Zhenghai Xue, Tingcong Liu, Bo An

Authority: Representative multi-turn credit-assignment extension of GRPO that stays critic-free.

Covers: `agentic-multi-turn-rl`, `critic-free-rl`, `grpo-successors`

Facts:
- Problem: multi-turn agent rewards are sparse/delayed; credit assignment across steps is much harder than in single-turn math. (Abstract) — “making credit assignment across individual steps significantly more challenging.”
- Method: two-level relative advantage - episode-level groups of trajectories, plus step-level groups built by finding repeated environment states ("anchor states") across trajectories; no critic, no extra rollouts. (Abstract) — “At the step-level, GiGPO introduces an anchor state grouping mechanism that retroactively constructs step-level groups by identifying repeated environment states across trajectories.”
- Reported gains over GRPO: >12% on ALFWorld and >9% on WebShop (Qwen2.5-1.5B/3B/7B-Instruct). (Abstract) — “achieves performance gains of > 12% on ALFWorld and > 9% on WebShop over GRPO”

Conflicts:
- Anchor-state grouping needs repeated identical states, which holds in ALFWorld/WebShop-style environments but is not guaranteed in open-ended tool use (inference from the abstract; not tested here).

### S59 · official · Hugging Face — Hugging Face TRL documentation: GRPO Trainer (main branch) (2026 · docs/trl/main; version not stated on the page)

https://huggingface.co/docs/trl/main/en/grpo_trainer · accessed 2026-10-26 · fetched: true · read depth: first 100,000 of 114,487 characters via WebFetch extraction (quotes are extraction output; not machine-verified)

Authors: Hugging Face TRL maintainers

Authority: What the most-used open trainer ships as defaults is a good proxy for current practice. Documents the dapo/dr_grpo/sapo/cispo loss variants, sequence-level importance sampling (GSPO) and vLLM mismatch correction.

Covers: `grpo-successors`, `training-dynamics`, `when-to-use-which`, `critic-free-rl`

Facts:
- Default loss in the docs is DAPO-style token-level normalization (loss_type default "dapo"); original GRPO per-sequence 1/|o_i| normalization is described as the initial formulation. (GRPOConfig.loss_type) — “loss_type: str = 'dapo'”
- Dr. GRPO option: divide by a constant (recommended: maximum completion length) instead of sequence length. (loss_type=dr_grpo) — “This constant is recommended to be the maximum completion length.”
- SAPO option (Qwen): smooth, temperature-controlled soft gate replacing hard clipping; CISPO option also present. (loss_type=sapo / cispo) — “the Qwen team proposes replacing the "hard" clipping mechanism of GRPO with a smooth, temperature-controlled soft gating mechanism.”
- Default KL coefficient is 0.0, so the reference model is not loaded (saves memory); R1's 0.001 is cited as an alternative. (GRPOConfig.beta) — “If 0.0 (default), the reference model is not loaded, reducing memory usage and improving training speed.”
- Clip defaults: epsilon 0.2; epsilon_high None (same as epsilon); DAPO recommends 0.28; with CISPO the recommended value is 5.0. (GRPOConfig.epsilon / epsilon_high) — “Paper [DAPO] recommends 0.28.”
- Advantage scaling default is "group" (std-normalized per group); "batch" and False are options; turning off std scaling means update magnitude depends on raw reward scale. (GRPOConfig.scale_rewards) — “Note that turning off std-based scaling also removes variance normalization, so update magnitudes depend directly on the raw reward scale and batch composition.”
- importance_sampling_level default "token"; "sequence" gives the GSPO behaviour. (GRPOConfig.importance_sampling_level) — “The [GSPO paper] shows that sequence-level sampling often yields more stable training and better alignment with sequence-level rewards.”
- num_generations default 8; vLLM integration modes "colocate" (default; shares GPUs with the trainer) and "server" (separate GPUs, HTTP). (GRPOConfig) — “Number of generations per prompt to sample.”
- Mismatch handling is on by default: sequence-level masked importance sampling; TIS truncates (clip max default 3.0) while MIS zeroes outliers. (vLLM importance-sampling section) — “By default, sequence-level Masked Importance Sampling (`vllm_importance_sampling_mode="sequence_mask"`) is activated for vLLM generation to address the generation-training mismatch”

Conflicts:
- Defaults here (DAPO loss, KL 0, group-std scaling still on) mix choices that S18/S45/S48 argue against (std scaling).

### S60 · textbook · Allen Institute for AI; print edition Manning Publications — Reinforcement Learning from Human Feedback (RLHF Book) (2026 · arXiv v12 (11 Sep 2026); web edition "last built 14 October 2026"; Manning print July 2026 (per rlhfbook.com); v1 posted Apr 2025)

https://arxiv.org/pdf/2504.12501v12 · accessed 2026-10-26 · fetched: true · read depth: table of contents, §3.1.6, §3.2.4, §7.3 (arXiv v12 text); rlhfbook.com landing page for edition facts

Authors: Nathan Lambert

Authority: The only single-volume textbook-tier treatment spanning RLHF, RLVR, DPO, rejection sampling, MOPD and over-optimization; written by the Tulu 3 lead; updated through Sep 2026.

Covers: `rlhf-pipeline`, `reward-modeling`, `ppo-for-llms`, `direct-preference-optimization`, `critic-free-rl`, `rlvr`, `distillation-opd`, `reward-hacking`, `when-to-use-which`, `frontier-2026`

Facts:
- Scope as stated in the abstract. (Abstract) — “The core of the book details every optimization stage in using RLHF, from starting with instruction tuning to training a reward model and finally all of rejection sampling, reinforcement learning, on-policy distillation, and direct alignment algorithms.”
- Chapter map (v12): 1 Introduction; 2 A Tiny History of RLHF; 3 Training Overview; 4 Instruction Fine-Tuning; 5 Reward Modeling; 6 Reinforcement Learning; 7 Reasoning and Inference-Time Scaling; 8 Direct-Alignment Algorithms; 9 Rejection Sampling; 10 The Nature of Preferences; 11 Preference Data; 12 Synthetic Data & Distillation; 13 Tool Use (section 13.1 Tool-Use Overview); 14 Over-Optimization; 15 Regularization; 16 Evaluation; 17 Crafting Model Character and Products. (Contents) — “6 Reinforcement Learning”
- Author's argument for RL despite simpler alternatives (rejection sampling, DPO): RL keeps winning; costs far more infrastructure; improvements "generally help the model a lot". (§3.1.6) — “Still, despite the simplicity of alternatives, RL-based methods continue to win out.”
- RL can be surgical: Tulu 3 trained RL on math prompts only while preserving general capability. (§3.1.6) — “RL can be done surgically — the model does a good job of learning where the prompt distribution lies, and RL tends to not “squash” the general capabilities of the model.”
- 2026 recipe shift: MOPD became a popular merging tool; MiMo-V2-Flash first documented it; counts of experts: Nemotron 3 Ultra >10, Kimi K3 nine, DeepSeek V4 >10. (§3.2.4) — “Various models utilizing MOPD as a final stage use a different number of experts, with Nemotron 3 Ultra at more than ten [65], Kimi K3 at nine [66], and DeepSeek V4 at more than ten [67]”
- Non-unanimity: labs are not agreed on MOPD; GLM-5 uses MOPD after a three-stage RL. (§3.2.4) — “MOPD is a popular new tool, but labs are far from unanimous in its usage.”
- Reasoning training reduces to a large RL run or SFT on outputs of a model that underwent much RLVR (distillation). (§7.3) — “traditionally for a reasoning model the core training step is either a large-scale RL run or a large-scale instruction tuning run on outputs of another model that had undergone a substantial portion of RLVR training (referred to as distillation).”

Conflicts:
- Online and print editions differ in date and layout from the arXiv v12 text; cite section numbers from the version you open.

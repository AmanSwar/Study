# System Design: From Classical Internet Systems to AI Inference and Training Platforms — Syllabus

As-of date: 2026-09-17. 39 modules, 5 parts, continuous numbering. ~293,500 estimated words.

Starting point: Aman is fluent in GPUs, inference engines, single-node serving, and multi-node LLM
inference (interconnects, PD disaggregation, wide EP, Dynamo/llm-d/vLLM/SGLang, K8s ops, capacity
economics — the `multi-node-llm-inference` track). He does **not** have classical system-design
knowledge. This course builds that from principles — never from programming basics — and then
applies the same discipline to the two systems he actually builds: AI inference platforms and AI
training infrastructure. GPU-side mechanics already covered in the multi-node course are referenced,
not repeated; this course designs the *interface* to that stack (Module 25), not the stack itself.

## Reading order and dependency rationale

**Part A — Foundations of System Design (Modules 1–11).** Ordered so that every later part can cite
back instead of re-deriving. Module 1 (request path) and Module 2 (caching) come first because they
are the layer every case study in Part B and every gateway module in Part C sits behind. Modules 3–5
(replication/partitioning → transactions/indexes → consistency/CAP/PACELC) are ordered so that each
assumes the one before: you cannot reason about isolation levels before you know what "a replica" is,
and you cannot classify a system as PA/EL or PC/EC before you've seen both a quorum write (Module 3)
and a two-phase commit (Module 4). Module 6 (consensus) depends on Module 3 because Paxos/Raft are
themselves replication protocols with an added agreement guarantee. Module 7 (file/object storage)
depends on Module 3 because GFS *is* a replication-and-partitioning scheme specialized to files —
kept as its own module because Aman's brief treats it as a distinct bullet with distinct systems
(GFS/HDFS/S3/Tectonic) that Part D's storage module (33) later cites directly. Modules 8–9
(messaging/streams → batch/stream processing) depend on Module 6 and each other in sequence, and
Module 9 is a direct prerequisite for Part D's training-data-pipeline module (32). Module 10
(services/APIs/mesh/backpressure/security) and Module 11 (SLOs/observability/resilience) close the
part because every mechanism in them — rate limiting, circuit breakers, zero trust — reappears
specialized in Part C (Modules 21–31) rather than re-explained there.

**Part B — Classical Case Studies, A to Z (Modules 12–20).** Ordered by data-lineage proximity to
Part A, then by source richness. Module 12 (Google Search + Gmail) leads because it is the most
direct application of Modules 3–7's exact lineage (GFS → Bigtable → Megastore → Spanner). Module 13
(YouTube/Vitess) follows as a second sharding-layer case study. Modules 14–15 split Netflix into video
infrastructure and resilience/personalization because both halves are individually rich enough to
need their own module without diluting either. Module 16 (Twitch) is adjacent (live video, chat at
scale). Module 17 deliberately bundles WhatsApp, Twitter, and Instagram — three case studies whose
*combined* source depth roughly equals one of the other case studies' individual depth — rather than
stretching each into a thin standalone module. Modules 18–20 (Stripe, Uber, Airbnb) close the part;
Module 18 is placed immediately before Part C because Module 23 (AI billing) cites it directly as the
pattern being reused.

**Part C — AI Inference Platform Design, Client to GPU (Modules 21–31).** The most important part,
given full treatment per Aman's instruction, ordered along the request's own path: edge/gateway (21)
→ identity/rate-limits (22, depends on 10 and 21) → billing (23, depends on 18 and 22) → routing/
admission (24, depends on 21–22) → the GPU backend handoff (25, depends on 24) → streaming (26,
depends on 21) → agent harnesses (27, depends on 26 — a harness is built on a streaming, cancellable
session) → prompt caching (28, depends on 2 and 22 — it specializes classical caching into a
billing-bearing mechanism) → safety/moderation (29, depends on 26 — streaming is what makes
moderation timing hard) → data/observability (30, depends on 11 and 23) → multi-tenancy/capacity (31,
depends on 24–25, closing the part on the same isolation-vs-utilization question that recurs at the
frontier). Module 25 is the one place this course explicitly designs an interface to the
multi-node-llm-inference course's territory rather than repeating it.

**Part D — AI Training Infrastructure, Systems Level (Modules 32–38).** Distributed-training
mechanics (TP/PP/FSDP/ZeRO internals, collective algorithms) are excluded throughout — Aman has these
already — referenced only where infrastructure design depends on them. Module 32 (data pipelines)
depends on Module 9 (it is that batch-processing lineage applied to text). Module 33
(storage/checkpointing/topology) depends on Module 7 (Tectonic is GFS's idea one layer up). Module 34
(scheduling/orchestration) depends on Module 6 (consensus underlies every scheduler's coordination
layer) and Module 33. Module 35 (fault tolerance, with incident-patterns folded in) depends on 33–34.
Module 36 (post-training pipelines) depends on 32. Module 37 (RL training infrastructure) depends on
Module 27 (environments-as-services is structurally the same idea as agent-harness sandboxes) and
Module 36. Module 38 (RL framework comparison) depends on 34 and 37, applying the scheduling and
staleness vocabulary already established to six concrete systems.

**Part E — Frontier and Synthesis (Module 39).** Deliberately last and deliberately a single module:
it recaps the disagreements this course could not resolve — because the field has not — rather than
introducing a new one, per the course guide's requirement that a course end at the frontier. Depends
on Modules 5, 31, and 38, the three modules whose disagreements it recaps.

## Part → Module map

**Part A — Foundations of System Design**
1. The Request Path: DNS, Load Balancers, Reverse Proxies, CDNs, and Edge Networks (~7,500w)
2. Caching at Every Layer: Placement, Coherence, and Invalidation (~7,000w)
3. Replication and Partitioning: Making One Dataset Look Like Many Machines (~8,000w)
4. Transactions, Indexes, and the Relational/NoSQL Split (~8,000w)
5. Consistency Models: Linearizability, CAP, and PACELC (~7,500w)
6. Consensus and Coordination: Paxos, Raft, Leases, and ZooKeeper/etcd (~8,000w)
7. Distributed File and Object Storage: GFS, HDFS, S3, and Tectonic (~7,500w)
8. Messaging and Streams: Logs, Queues, Exactly-Once, and Idempotency (~7,500w)
9. Batch and Stream Processing: The MapReduce to Spark/Flink Lineage (~7,500w)
10. Services, APIs, and the Trust Boundary (~8,500w)
11. Reliability Engineering: SLOs, Observability, and Resilience Patterns (~7,500w)

**Part B — Classical Case Studies, A to Z**
12. Google Search and Gmail: Crawl to Index to Serve (~8,000w)
13. YouTube and Vitess: Sharded MySQL at Video Scale, and What Isn't Public (~6,500w)
14. Netflix Video Infrastructure: Open Connect CDN and Per-Title Encoding (~7,000w)
15. Netflix Resilience and Personalization (~7,000w)
16. Twitch: Live Ingest, Low-Latency HLS, and Chat at Scale (~7,000w)
17. WhatsApp, Twitter, and Instagram: Messaging and Feed Architectures at Scale (~8,500w)
18. Stripe: Ledgers, Idempotency Keys, and API Design for Payments (~7,000w)
19. Uber: Dispatch, Geo-Indexing with H3, and Real-Time ETA (~7,000w)
20. Airbnb: From Monolith to SOA (~6,500w)

**Part C — AI Inference Platform Design, Client to GPU**
21. Edge and API Gateway for AI Inference Platforms (~8,000w)
22. Identity, Rate Limits, and Quotas (~8,500w)
23. Billing and Metering: Ledger Design, Idempotency, Holds, and Reconciliation (~8,000w)
24. Request Routing, Admission Control, and SLO Tiers (~8,000w)
25. The GPU Backend Handoff: Designing the Interface to Multi-Node Serving (~6,500w)
26. Streaming, Cancellation, and Resumption (~7,000w)
27. Tool Use, Agent Harnesses, and Sandboxes for Code Execution (~7,500w)
28. Platform Prompt/Prefix Caching: Tiers, TTLs, and Billing Implications (~7,500w)
29. Safety and Moderation Pipelines (~7,000w)
30. Data and Observability: Storage, Retention, Privacy, and Per-Token Tracing (~8,000w)
31. Multi-Tenancy, Isolation, and Capacity/Cost Management (~7,500w)

**Part D — AI Training Infrastructure, Systems Level**
32. Training Data Pipelines: Crawl/Ingest, Filtering/Dedup, Tokenization (~7,000w)
33. Storage and Checkpointing at Scale: Cluster Network Topology (~7,500w)
34. Cluster Scheduling and Orchestration: Slurm, Borg/Kubernetes, Ray, MAST (~8,000w)
35. Fault Tolerance and Elasticity: Checkpoint/Restart, Stragglers, Incidents (~7,500w)
36. Post-Training Pipelines: SFT/Preference Flows, Evaluation Gates (~7,500w)
37. RL Training Infrastructure: Environments, Rollouts, Trainer/Rollout Decoupling (~8,000w)
38. Comparing the Open-Source RL Frameworks (~9,000w)

**Part E — Frontier and Synthesis**
39. Where the Field Still Disagrees and What Is Changing (~6,500w)

## Disagreements and where they land

- **Strong synchronous consistency (Spanner) vs. eventual consistency (Dynamo)** — Module 5
  (primary), Module 12 (Google's own Bigtable→Spanner evolution), Module 17 (Gizzard's Dynamo-like
  parallel), Module 39 (recapped as still-open).
- **Is Raft's understandability advantage over Paxos decisive in practice?** — Module 6 (primary),
  Module 12 (Chubby/Spanner's continued production use of Paxos).
- **Fixed-window vs. sliding-window rate limiting** — Module 10 (general mechanism), Module 22
  (applied to LLM API rate limits).
- **How should token-based rate limits be enforced when true cost is only known after generation?**
  — Module 22 (primary: pessimistic-estimate-and-adjust vs. optimistic-admit-and-reconcile).
- **Should cached tokens count against a rate limit?** — Module 28 (primary).
- **Pre-generation blocking gate vs. post-generation signal for safety moderation?** — Module 29
  (primary).
- **Dedicated per-tenant GPU pools vs. shared pools with per-key limits** — Module 31 (primary),
  Module 39 (recapped as still-open).
- **Synchronous vs. asynchronous trainer/rollout coupling in RL infrastructure** — Module 37
  (primary), Module 38 (applied across six frameworks), Module 39 (recapped as still-open).
- **HPC batch scheduler vs. cloud-native orchestrator for GPU training clusters** — Module 34
  (primary), Module 38 (referenced re: the Ray-based-vs-native-async orchestration fork).

Per the learner profile, disagreements are distributed where they arise rather than collected into
one controversies module; Module 39 exists to recap the three still-genuinely-open ones (consistency,
multi-tenancy, async RL) as unresolved as of the as-of date, not to introduce new ones.

## Excluded, and why

- **GPU-side multi-node serving mechanics** (interconnects, PD disaggregation, wide EP, Dynamo/
  llm-d/vLLM/SGLang internals, K8s serving ops, capacity economics). Aman has just completed the
  `multi-node-llm-inference` course covering exactly this. Module 25 designs the *interface* to this
  stack; it does not re-derive the stack itself.
- **Distributed-training mechanics** (tensor/pipeline/FSDP/ZeRO parallelism internals, collective
  algorithms). Explicitly excluded from Part D per the request; referenced only where an
  infrastructure decision depends on them (e.g. Module 33's checkpoint sizing).
- **Experiment tracking and evaluation-pipeline tooling for pre-training** (mentioned in the
  request's Part D bullet). `sources.json` has no subtopic, coverage rating, or source for this at
  all — not even `thin`. No module is scheduled on it; Module 32 and Module 36 note the gap rather
  than inventing a generic MLOps treatment of it.
- **Model-registry tooling beyond what Llama 3's own release framing documents.** Same reasoning:
  Module 36 states this gap explicitly instead of describing a registry architecture no source
  supports.
- **Ranking-algorithm internals for Google Search**, and **Airbnb's search-ranking and
  booking-consistency internals**, and **YouTube's transcoding pipeline, recommendation system, and
  CDN specifics**. None of these are publicly disclosed by the companies that own them at any level
  of technical detail; Modules 12, 13, and 20 state each gap plainly rather than reconstructing it
  from general assumptions about how such a system "must" work.
- **Instagram's use of Cassandra.** Aman's brief names it, but the only Instagram source in this map
  (S27) documents Postgres-schema-based logical sharding, not Cassandra. Module 17 covers what S27
  actually supports and does not extend the claim to Cassandra.
- **A dedicated incident-patterns module.** `sources.json` rates this subtopic coverage `none` — zero
  sources anywhere in the map. Per the course guide, no module is scheduled on it standalone; the
  closest real material (graceful degradation, admission control, and the Llama 3/MegaScale
  reliability data) is folded into Modules 24 and 35 instead, with the fold stated explicitly in both
  modules' scope.
- **Exercises, quizzes, deliverables, projects, capstones, schedules, assessments.** Never included,
  per the learner profile and study-material style rules — the application happens in Aman's job.

## Where sources are thin

`sources.json` rates 20 of 47 subtopics `thin` and 1 (`incident-patterns`) `none`; 26 are `well`. Every
`thin` subtopic still has a module (scoped narrower, per the course guide), and the `none` subtopic is
folded rather than given a standalone module. The specific gaps, all stated in-module rather than
silently patched:

- **Gmail** has no dedicated Google-published architecture account at all — Module 12 grounds it only
  through its documented appearance as a Megastore consumer and a Borg prod workload.
- **YouTube's** own transcoding/CDN/recommendation architecture is undocumented beyond Vitess —
  Module 13 covers the sharding layer in real depth and states the rest is not public.
- **WhatsApp, Twitter, and Instagram** each rest on a single 2012–2013-vintage primary source with no
  comparably detailed public update since — Module 17 bundles all three and states this explicitly
  rather than implying current architecture.
- **Airbnb's** search and booking internals are entirely undocumented in this map — Module 20 covers
  only the SOA-migration story S31 actually supports.
- **Auth-identity** for AI platforms has no dedicated API-key/OAuth architecture disclosure from any
  platform — Module 22 draws the org/workspace hierarchy from what OpenAI's and Anthropic's rate-limit
  docs reveal incidentally.
- **The GPU backend handoff** (Module 25) is designed from the Kubernetes Gateway API Inference
  Extension's routing signals as first principles, stated as such, not surveyed from multiple
  platforms' disclosed interfaces (none exist at this level of detail).
- **Multi-tenancy's dedicated-cluster claim (Module 31)** rests on Fireworks figures relayed by a
  third party, not Fireworks' own engineering blog — flagged as unverified in-module.
- **Philly (S55)** could not be fetched as a full PDF; Module 34 names it and states it carries zero
  verified facts in this map, rather than asserting anything from it.
- **DDIA (S12)** is cited by chapter numbers inferred from the well-known 1st-edition structure; its
  Feb-2026 2nd-edition table of contents returned 403 to fetch. A writer should confirm chapter
  numbers against a preview copy before publishing a precise citation.

## Source-map summary

66 sources, all cited by at least one of the 39 modules. By type: 20 official (vendor/project
documentation), 19 blog (company engineering blogs), 16 paper (peer-reviewed or arXiv), 7 repo
(READMEs), 2 textbook, 1 talk, 1 analysis.

Top 10 by breadth of module coverage: S12 (Designing Data-Intensive Applications — 13 modules,
spanning nearly all of Part A plus case-study and training-pipeline grounding), S9 (Borg — 9 modules,
the connective tissue between classical scheduling, service admission control, and training-cluster
orchestration), S17 (Cloudflare CDN reference architecture — 6), S3 (Bigtable — 6), S4 (Spanner — 6),
S53 (Llama 3 Herd of Models — 6, the single most detailed public account of training infrastructure),
S39 (LiteLLM proxy architecture — 6), S5 (Dynamo — 5), S2 (The Google File System — 4), S20 (Netflix
Open Connect at 100Gbps — 3).

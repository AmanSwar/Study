# Multi-Node LLM Inference: From One Node to N — Syllabus

As-of date: 2026-09-17. 18 modules, 7 parts, continuous numbering. ~132,500 estimated words.

Starting point: Aman is TP8-on-8×B200 fluent — paged KV cache, continuous batching, CUDA graphs,
quantization, single-node MoE all assumed. This course starts exactly where the second node
appears and does not re-teach intra-node mechanics except where multi-node changes them.

## Reading order and dependency rationale

**Part 1 — The Multi-Node Frame (Modules 1–2).** Everything downstream is either a statement about
bandwidth/latency (does a transfer cross NVLink, NVL72, or IB/RoCE, and what does that cost) or a
statement about routing (which of N replicas gets this request). Module 1 builds the bandwidth
ladder used as the arithmetic reference for every later cost claim — KV transfer costs (Modules 4,
7, 8), all-to-all costs (Module 5), and the AMD/TPU contrasts (Modules 17–18) all cite back to it.
Module 2 establishes that replication breaks stateless load balancing the moment KV cache becomes
per-replica state, which is the precondition for every routing mechanism discussed in the
project-specific stack modules (9–13).

**Part 2 — Prefill/Decode Disaggregation (Modules 3–4).** Module 3 makes the goodput/interference
argument before any mechanism is shown, because whether to disaggregate is a decision made before
how — and the "how" (Module 4's connectors, transfer paths, failure modes) only matters once the
decision is made. Both modules are generic-pattern modules; the five stack modules in Part 5 each
cite back to Module 4 rather than re-deriving PD mechanics.

**Part 3 — Cross-Node Mixture-of-Experts (Modules 5–6).** Wide EP is introduced as a distinct traffic
pattern (all-to-all, not all-reduce) before any specific deployment is discussed. Module 6 then
grounds the whole part in DeepSeek's own two-disclosure history — the only publicly available
production wide-EP deployment with real cost numbers, and a genuine, dated disagreement (EP320 vs.
EP144) that later modules keep referring back to.

**Part 4 — Multi-Node KV Cache Management (Modules 7–8).** Requires Module 4 (PD scheduling, since
Mooncake's four-step workflow assumes it) and, for Module 8, Module 7 (LMCache and NIXL are
described relative to Mooncake as a KV-centric architecture they can plug into as a backend).

**Part 5 — The Open-Source Serving Stack (Modules 9–13).** Ordered Dynamo → llm-d → vLLM → SGLang →
TensorRT-LLM/Triton: Dynamo first because it is the orchestration layer that sits above the other
four as a backend-agnostic router/autoscaler, making it the natural point of comparison for
everything after it. llm-d follows immediately because Module 10 explicitly needs Module 9 as a
contrast object (orchestration layer vs. Kubernetes-native scheduler pattern). The three inference
engines (vLLM, SGLang, TensorRT-LLM) follow in the order most load-bearing for Aman's stack
familiarity, each depending only on Modules 4 and 5 (the generic PD and EP patterns), not on each
other — they can be read in any order among themselves without loss.

**Part 6 — Operations, Economics, and the Frontier (Modules 14–16).** Module 14 (Kubernetes topology
and ops) depends on Modules 9–10 because LWS, InferencePool, and the orchestration-layer landscape
are meaningless without the Dynamo/llm-d architectures they sit under. Module 15 (capacity
planning) depends on Module 3 (the disaggregation decision) and Module 6 (DeepSeek's real cost
numbers) as its worked examples. Module 16 (frontier) is deliberately last in the part and depends
on Modules 4, 6, and 15 because it is explicitly a synthesis module — it revisits this course's
open disagreements rather than introducing a new mechanism category, per the course-guide's
requirement that a course end at the frontier.

**Part 7 — Contrasting Hardware (Modules 17–18).** Placed last because both modules are structured
as "what ports from Modules 1–16, what doesn't" — they are unreadable as a starting point and are
the natural close: one NVIDIA-alternative module (AMD), one non-GPU-architecture module (TPU), per
Aman's explicit constraint of exactly one module each.

## Part → Module map

**Part 1 — The Multi-Node Frame**
1. Interconnect and Transport: The Physical Layer of Multi-Node Serving (~7,000w)
2. Replication and Routing: KV-Cache-Aware, Prefix-Aware, and SLO-Aware Request Placement (~7,500w)

**Part 2 — Prefill/Decode Disaggregation**
3. Why Disaggregation Pays: Goodput, Interference, and the Ratio Decision (~8,000w)
4. KV Transfer and Scheduling Across the Prefill/Decode Boundary (~7,500w)

**Part 3 — Cross-Node Mixture-of-Experts**
5. Wide Expert Parallelism: DeepEP, Placement, and Load Balancing (~8,500w)
6. DeepSeek's Production Inference System: EP Configurations, Economics, and Reproduction (~7,000w)

**Part 4 — Multi-Node KV Cache Management**
7. Prefix Caching and Tiered KV Offload Across Nodes: Mooncake's Architecture (~7,500w)
8. LMCache and NIXL: The Reusable KV Layer and the Transfer-Library Primitive (~7,000w)

**Part 5 — The Open-Source Serving Stack**
9. NVIDIA Dynamo: Router, Planner, KV Block Manager, and Disaggregation Orchestration (~8,000w)
10. llm-d: The Inference Scheduler, Endpoint Picker, and Gateway API Inference Extension (~7,500w)
11. vLLM Multi-Node: Distributed Serving, Disaggregated Prefill, and the Production Stack (~7,500w)
12. SGLang: Router, PD Disaggregation, and Wide-EP Deployment Patterns (~7,000w)
13. TensorRT-LLM and Triton: Disaggregated Serving on GB200 (~6,500w)

**Part 6 — Operations, Economics, and the Frontier**
14. Kubernetes Deployment Topologies and Operations at Fleet Scale (~7,000w)
15. Capacity Planning and Economics: Goodput, Tokens/s/$, and Little's-Law Reasoning at Rack Scale (~7,500w)
16. The Research Frontier: Live Migration, Fused Megakernels, and What Is Still Unsettled (~6,500w)

**Part 7 — Contrasting Hardware**
17. AMD MI300X/MI355X Multi-Node Serving: RCCL, InfiniBand/RoCE, and vLLM-ROCm (~7,500w)
18. TPU Multi-Host Serving: ICI, Pathways, and JetStream (~7,500w)

## Disagreements and where they land

- **When does prefill/decode disaggregation actually pay off?** — Module 3 (primary), Module 15
  (applied as a sizing tool), Module 16 (recapped as unresolved).
- **Is a second (decode) pool worth the operational cost, or does chunked prefill capture most of
  the benefit?** — Module 3 (primary), Module 7 (Mooncake's own reconsideration), Module 11
  (vLLM's default-colocated posture as implicit evidence).
- **How large should the cross-node decode EP degree be for a DeepSeek-class model?** — Module 6
  (primary: EP320 vs. EP144), Module 16 (recapped as unresolved).
- **Approximate vs. precise KV-cache-aware routing** — Module 2 (primary).

Per the learner profile, disagreements are distributed where they arise rather than collected into
one controversies module; Module 16 exists to recap the two still-open ones (disaggregation
conditionality, EP degree) as explicitly unresolved as of the as-of date, not to introduce new ones.

## Excluded, and why

- **Intra-node TP/PP mechanics, single-GPU kernel work, paged KV cache internals, continuous
  batching, CUDA graphs, quantization formats, single-node MoE routing.** Aman has already worked
  through these at depth (learner profile, Blackwell serving syllabus). They appear only as
  reference points inside multi-node modules (e.g., Module 5's "TP=1 attention replicated across DP
  ranks" assumes single-node TP fluency) and are never re-derived.
- **A dedicated "Kubernetes for beginners" or Helm/YAML tutorial module.** Module 14 covers
  deployment *topology* (what shape the deployment takes and why) at the mechanism level, not
  cluster-admin syntax, which is operational rather than architectural and would not survive the
  "no exercises/deliverables" constraint in spirit.
- **A dedicated module for each of Ray Serve, KServe, and AIBrix.** Aman's brief lists these as
  "where relevant," lower-weight than the primary stack (Dynamo/llm-d/vLLM/SGLang/TensorRT-LLM).
  They are folded into Module 14 as the orchestration-layer landscape rather than each getting a
  full module, to stay within the 18-module cap without diluting the primary stack's depth.
- **Assessment, exercises, quizzes, deliverables, capstone projects, schedules.** Never included,
  per the learner profile and study-material style rules — the application happens in Aman's job,
  not in the material.
- **A named project called "Thunder Agent."** It does not exist. Verified by the researcher via
  direct search and fetch (sources.json notes field). The real Hazy Research/Stanford-adjacent
  multi-GPU serving work — Tokasaurus, ThunderMLA, ThunderKittens — is covered in Module 16, with
  the naming correction stated explicitly in-module, not silently substituted.
- **An unsourced Little's-Law blog post** (tianpan.co, claiming 30–70% throughput gains from
  work-conserving schedulers) found during research but excluded as a source: unattributed
  individual blog with an unsourced headline statistic, fails the source-hierarchy bar. Module 15
  grounds equivalent queueing-theoretic reasoning in DistServe's M/D/1 analysis instead, and states
  the exclusion explicitly so Aman knows the claim was seen and rejected, not missed.
- **Cross-vendor MLPerf/InferenceMAX result tables beyond the specific figures cited** (the ~90
  tok/s/user disaggregation crossover, the MI300X-vs-MI355X per-MW comparison). A full MLPerf
  methodology treatment is out of scope for a course about multi-node *mechanism*, not benchmark
  methodology.

## Where sources are thin

sources.json reports all 20 subtopics at coverage rating **well** (≥2 primary/official sources with
extracted facts each) — none `thin`, none `none`. No subtopic required narrowing its module scope
or flagging a writer to fetch more. Two caveats carried forward from the source map, both handled
in-module rather than silently:

- **AMD's own product marketing pages timed out on every fetch attempt** during research. MI355X
  numbers in Module 17 are sourced from a SemiAnalysis/InferenceX aggregation page (analysis-tier),
  cross-checked against AMD's own Hot Chips 2024 talk for the prior-generation MI300X (official
  talk-tier). Module 17 states this provenance explicitly rather than presenting MI355X numbers as
  AMD-primary.
- **The DeepSeek-V3 technical report (Dec 2024) and the Day-6 Open Source Week disclosure (Feb
  2025) genuinely disagree** on decode EP degree (EP320/40 nodes vs. EP144/18 nodes). Module 6
  presents both numbers as sequential production states, not as an error in either source, and
  Module 16 recaps the disagreement as still-open framing rather than resolved.
- **GPUDirect RDMA bandwidth figures in Module 1 come from 2014-era hardware** (Ivy Bridge Xeon,
  K40 GPUs, PCIe3, FDR InfiniBand) — roughly two InfiniBand generations out of date for raw
  bandwidth. Module 1 uses this source only for the RDMA mechanism and its ~1.7µs latency floor,
  never as a current bandwidth reference; current bandwidth numbers come from DeepEP (Module 1/5)
  and the ConnectX-7 datasheet.
- **NIXL and TensorRT-LLM both ship pre-release version tags on a roughly monthly cadence.** Every
  version number cited in Modules 8, 9, and 13 is paired with the 2026-09-17 access date, since
  this layer of the stack is the fastest-moving part of the course.

## Source-map summary

54 sources total, all `fetched: true`. By type: 15 official (vendor/project documentation), 13
repo (READMEs/release notes), 8 paper (peer-reviewed or arXiv), 7 blog (vendor or lab engineering
blogs), 2 analysis (third-party, labelled non-primary), 1 talk (conference presentation).

Top 10 by breadth of module coverage: S12 (vLLM disaggregated prefill docs — Modules 4, 8, 11),
S16 (SGLang PD disaggregation docs — Modules 3, 4, 5, 7, 12), S22 (Mooncake paper, FAST 2025 Best
Paper — Modules 3, 7), S25 (NIXL architecture — Modules 1, 8), S27 (DeepEP README — Modules 1, 5),
S28 (DeepSeek-V3 technical report — Modules 5, 6, 16), S35 (DistServe, OSDI 2024 — Modules 3, 15),
S40 (GB200 NVL72 product page — Modules 1, 5), S43 (InferenceMAX/SemiAnalysis — Modules 3, 15, 16,
17), S2 (Dynamo disaggregated-serving docs — Modules 4, 9, 13).

Every one of the 54 sources is cited by at least one module.

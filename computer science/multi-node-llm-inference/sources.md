# Sources — Multi-node LLM inference: scaling serving from one 8-GPU node to N nodes (replication, routing, prefill/decode disaggregation, cross-node MoE, and the open-source stack)

As of 2026-09-17 · 54 sources · course

## Coverage

**Well covered (20):** replication-routing, pd-disaggregation, wide-ep-moe, multi-node-kv-management, interconnect-transport, stack-dynamo, stack-llm-d, stack-vllm-multinode, stack-sglang, stack-trtllm-triton, stack-mooncake, stack-lmcache, stack-nixl, stack-deepep-deepseek, stack-orchestration, k8s-ops, capacity-economics, frontier-research, amd-multinode, tpu-multihost

**Thin (0):** none

**Not covered (0):** none

## Subtopics

- `replication-routing` — Replication + routing: KV-cache-aware, prefix-aware, load/SLO-aware routers, session affinity
- `pd-disaggregation` — Prefill/decode disaggregation: mechanism, ratios, KV transfer paths, scheduling, failure modes
- `wide-ep-moe` — Cross-node model parallelism for MoE: wide EP, DeepEP all-to-all, expert placement/load balancing, attention-DP+EP layouts, NVL72 vs 8-GPU
- `multi-node-kv-management` — Multi-node KV management: prefix caching across nodes, offload tiers (HBM/host/SSD/remote), Mooncake, LMCache, NIXL
- `interconnect-transport` — Interconnect & transport: NVLink/NVSwitch vs InfiniBand/RoCE vs NVL72 fabric, GPUDirect RDMA, UCX/NIXL, real bandwidth/latency
- `stack-dynamo` — NVIDIA Dynamo: router, planner, KVBM, disaggregation
- `stack-llm-d` — llm-d: inference scheduler, EPP, Kubernetes Gateway API Inference Extension
- `stack-vllm-multinode` — vLLM multi-node + disaggregated prefill + production stack
- `stack-sglang` — SGLang router + PD disaggregation + wide EP
- `stack-trtllm-triton` — TensorRT-LLM / Triton disaggregated serving
- `stack-mooncake` — Mooncake: KVCache-centric disaggregated architecture, Transfer Engine, Conductor
- `stack-lmcache` — LMCache: tiered KV cache offload and reuse layer
- `stack-nixl` — NIXL: NVIDIA Inference Xfer Library
- `stack-deepep-deepseek` — DeepEP and DeepSeek's published inference-system design (EP32/EP320, EPLB)
- `stack-orchestration` — Ray Serve, KServe, AIBrix, LeaderWorkerSet — orchestration layer where relevant
- `k8s-ops` — Kubernetes/operations: deployment topologies, SLO autoscaling, multi-tenant serving, observability, upgrades/failure handling
- `capacity-economics` — Capacity planning & economics: goodput under TTFT/ITL SLOs, tokens/s/$ vs N, when disaggregation pays, Little's-law reasoning
- `frontier-research` — Frontier/research: DistServe, Splitwise, Llumnix, Mooncake paper, Hazy Research multi-GPU serving work
- `amd-multinode` — AMD MI300X/MI355X multi-node serving: RCCL, IB/RoCE, vLLM-ROCm
- `tpu-multihost` — TPU multi-host serving: ICI, Pathways, JetStream

## Disagreements found

### When does prefill/decode disaggregation actually pay off?

- **Position A**: Disaggregation is close to a strict win for goodput under fixed TTFT/TPOT SLOs: DistServe shows 7.4x more requests or 12.6x tighter SLO than colocated systems on A100s; Mooncake reports up to 525% throughput and 75% more real-world requests; TensorRT-LLM measures 1.4x-6.11x speedups on GB200 depending on ISL:OSL ratio. [S35, S22, S20]
- **Position B**: In production TCO terms on current hardware, disaggregation's advantage is conditional on the interactivity (tok/s/user) target: GB200 NVL72 with disaggregated prefill beats single 8-GPU B200 nodes only below about 90 tok/s/user; above that crossover, the simpler aggregated 8-GPU B200 server wins on TCO per million tokens. [S43]

### Is a second (decode) pool worth the operational cost, or does chunked-prefill on a single colocated engine capture most of the benefit?

- **Position A**: Disaggregation is necessary: colocating prefill and decode causes measurable prefill-decode interference (DistServe measures decoding slowdown with each added prefill job) and chunked-prefill only mitigates, does not eliminate, this interference while adding O(N^2) redundant KV-cache HBM reads for N prefill chunks. Mooncake explicitly reconsidered the question after chunked-prefill became common and chose to keep disaggregation for long-context multi-node prefill and VRAM-saving reasons. [S35, S22]
- **Position B**: vLLM's default production path is a single colocated engine with continuous batching and chunked-prefill (with disaggregation offered only as an explicitly 'experimental' KV-connector feature) — implying that for many deployments the added complexity of a second pool, a KV transport layer, and cross-pool scheduling is not (yet) considered worth it outside of large-scale or SLO-critical deployments. [S12]

### How large should the cross-node decode expert-parallel degree be for a DeepSeek-class MoE model?

- **Position A**: DeepSeek-V3's own December 2024 technical report specifies a decode minimum deployment unit of EP320 across 40 nodes (320 GPUs), each GPU hosting exactly one expert, with 64 GPUs dedicated to redundant/shared experts. [S28]
- **Position B**: DeepSeek's own February 2025 Open Source Week disclosure describes production decode running at EP144 across only 18 nodes (144 GPUs), each GPU hosting 2 routed experts plus 1 shared expert — a much smaller unit reached after roughly two months of operating the service. [S29]

### Approximate vs. precise KV-cache-aware routing

- **Position A**: Approximate prefix-cache scoring (fixed-size block hashing plus an in-memory LRU index of recently-routed prefixes) is cheap, scales with fleet size without a synchronous global state, and is llm-d's default scheduling mode. [S8]
- **Position B**: Precise KV-cache indexing subscribes to real KV cache add/evict events emitted by vLLM (and SGLang, TensorRT-LLM) over ZeroMQ and maintains an exact, globally consistent view of which token blocks live on which pod — correct at the cost of an event-streaming subsystem and tokenization round-trips to the engine. [S8]

## Sources

### S1 · repo · NVIDIA — ai-dynamo/dynamo (README) (2026, v1.4.2 (container tags); v1.5.0-dev builds referenced)

https://github.com/ai-dynamo/dynamo · accessed 2026-09-17

*Primary repository and README for NVIDIA Dynamo, the datacenter-scale distributed inference orchestration layer above vLLM/SGLang/TensorRT-LLM.*

_Covers: stack-dynamo, pd-disaggregation, multi-node-kv-management_

Facts:
- Dynamo is described as the orchestration layer above inference engines, not a replacement for them (README header)
- Named components: Router (KV-aware routing), Planner (SLA-driven autoscaler), KV Block Manager (KVBM, offloads KV cache GPU->CPU->SSD->remote), ModelExpress (GPU-to-GPU weight streaming via NIXL/NVLink), Grove (Kubernetes operator for topology-aware gang scheduling) (Core Capabilities / architecture section)
- Dynamo reports 7x higher throughput per GPU (DeepSeek R1 on GB200), 2x faster TTFT with KV-aware routing, and 750x higher throughput (DeepSeek-R1 on GB300) in headline claims (README highlights)
- Supported backends: vLLM, SGLang, TensorRT-LLM (README backend list)

### S2 · official · NVIDIA — Disaggregated Serving (NVIDIA Dynamo Documentation) (2026, docs.nvidia.com/dynamo knowledge-base (live))

https://docs.nvidia.com/dynamo/knowledge-base/concepts/system-architecture/disaggregated-serving · accessed 2026-09-17

*NVIDIA's own architectural description of how Dynamo implements disaggregated prefill/decode, including backend-specific transfer behavior.*

_Covers: pd-disaggregation, stack-dynamo, multi-node-kv-management_

Facts:
- Dynamo uses NIXL to transfer KV cache directly from prefill-engine VRAM to decode-engine VRAM, non-blocking (Efficient KV Transfer section)
- Backend-specific behavior differs: SGLang uses bootstrap coordination so decode can begin while KV transfer proceeds in parallel; vLLM and TensorRT-LLM run prefill synchronously and decode waits for prefill completion (Backend-Specific Transfer Metadata section)
- Dynamo supports runtime-reconfigurable xPyD (x prefill workers, y decode workers); workers register with discovery service and publish RuntimeConfig including KV capacity on add, and drain active requests before deregistering on remove (Runtime-Reconfigurable xPyD section)
- The doc explicitly declines to encode prefill/decode boundary as a universal token threshold (Performance Characteristics section)

### S3 · repo · NVIDIA — ai-dynamo/dynamo Releases (2026, v1.4.2 (2026-08-29); v1.5.0-dev builds in progress)

https://github.com/ai-dynamo/dynamo/releases · accessed 2026-09-17

*Release log establishing Dynamo's current version and release cadence (monthly-scale churn).*

_Covers: stack-dynamo_

Facts:
- Latest tagged release is v1.4.2, dated August 29, 2026, a patch fixing NIXL loader-path resolution in the Frontend and SGLang Runtime images (Releases page)
- v1.4.1 added OpenAI-compatible classify and pooling endpoints to the Dynamo Frontend with vLLM worker support (Releases page)
- A dev build v1.5.0-gemma-4-31b-dev.1 ships an official TensorRT-LLM runtime container built on TensorRT-LLM 1.3.0rc25 (Releases page)

### S4 · official · NVIDIA — Planner Guide (NVIDIA Dynamo Documentation, v1.2.1) (2026, v1.2.1)

https://docs.nvidia.com/dynamo/v1.2.1/components/planner/planner-guide · accessed 2026-09-17

*Official documentation of the Dynamo Planner, the SLO-driven autoscaler for prefill/decode worker counts.*

_Covers: stack-dynamo, k8s-ops, capacity-economics_

Facts:
- The Planner is an autoscaling controller that adjusts prefill and decode engine replica counts at runtime to meet latency SLAs, using Prometheus metrics or load-predictor output plus engine performance models (Planner overview)
- Default SLA targets: TTFT 500.0 ms, ITL 50.0 ms (Planner configuration defaults)
- Four scaling modes exist: throughput, latency, load, sla; when load-based and throughput-based scaling are both active, load-based scaling runs after throughput-based scaling and adjusts above the throughput-derived floor (Scaling Logic section)
- SLA-mode uses a Rust engine performance shim with native AIC estimates plus online FPM tuning or FPM regression fallback, bootstrapped from self-benchmark or profiler-generated (npz/JSON) data (Performance Model Mechanism section)

### S5 · blog · NVIDIA Developer Blog — NVIDIA Dynamo 0.4 Delivers 4x Faster Performance, SLO-Based Autoscaling, and Real-Time Observability (2025, Dynamo 0.4 (published 2025-08-13))

https://developer.nvidia.com/blog/dynamo-0-4-delivers-4x-faster-performance-slo-based-autoscaling-and-real-time-observability · accessed 2026-09-17

*NVIDIA engineering blog with dated, model-specific performance numbers and the observability metric list Dynamo emits.*

_Covers: stack-dynamo, capacity-economics, k8s-ops_

Facts:
- gpt-oss-120b on Dynamo + TensorRT-LLM on B200 achieved up to 4x faster interactivity (tokens/second/user) for very long input sequence lengths (Performance section)
- DeepSeek-R1 671B on GB200 NVL72 with TensorRT-LLM and Dynamo achieved 2.5x higher throughput (tokens/second/GPU) (Performance section)
- Planner predicts future traffic with ARIMA or Prophet time-series models, then calculates minimum PD workers to meet SLA targets under predicted demand (SLO-based autoscaling section)
- Metrics emitted via Prometheus: requests/sec, request duration, TTFT, ITL, input/output sequence length, GPU utilization and power usage (Observability section)

### S6 · repo · llm-d (CNCF Sandbox; Red Hat, Google Cloud, IBM Research, CoreWeave, NVIDIA) — llm-d-inference-scheduler architecture.md (2026, main branch, doc undated)

https://github.com/llm-d/llm-d-inference-scheduler/blob/main/docs/architecture.md · accessed 2026-09-17

*Primary architecture document for llm-d's inference scheduler, the component that implements KV-cache-aware and load-aware scheduling behind the Endpoint Picker.*

_Covers: replication-routing, stack-llm-d_

Facts:
- Routing pipeline stages: Request Control (flow-control admission, global Screener does preliminary endpoint filtering) -> Filtering -> Scoring (weighted set of scorers) -> Selection (highest-scored pod) (Scheduling Components)
- Named plugins: prefix-cache-scorer, decode-filter, max-score-picker, session-id-producer, precise-prefix-cache-producer, inflight-load-producer (Named Plugins & Scorers)
- Data layer follows a Source -> Extract -> Attribute lifecycle populating per-endpoint attributes in a shared datastore for scorers (Data Layer Architecture)
- Design principle: no core changes needed to add new scorers or filters (pluggability) (Design Principles)

### S7 · repo · llm-d (CNCF Sandbox since March 2026; Red Hat, Google Cloud, IBM Research, CoreWeave, NVIDIA) — llm-d/llm-d (README) (2026, v0.7 (May 2026))

https://github.com/llm-d/llm-d · accessed 2026-09-17

*Primary project README stating llm-d's five 'well-lit path' themes, founding orgs, and its own headline benchmark numbers.*

_Covers: stack-llm-d, replication-routing, pd-disaggregation, wide-ep-moe, k8s-ops_

Facts:
- llm-d organizes its offering into five themes: Intelligent Routing (prefix-cache and load-aware balancing), Advanced KV-Cache Management (tiered offloading to CPU/disk), Serving Large Models (prefill/decode disaggregation and wide expert-parallelism), Operational Excellence (flow control and SLO-aware autoscaling), Batch Processing (OpenAI-compatible APIs) (Well-Lit Paths summary)
- Supported inference engines: vLLM and SGLang (Supported Inference Engines)
- Founding organizations: Red Hat, Google Cloud, IBM Research, CoreWeave, NVIDIA; became CNCF Sandbox project March 2026 (Project background)
- Headline claims: 3x higher output throughput and 2x faster TTFT with prefix-cache routing; 40% latency reduction with predicted scheduling; up to 70% higher tokens/sec with disaggregation; 50k tokens/sec cluster throughput on 16x16 B200s; 13.9x throughput improvement with hierarchical KV offloading (README benchmark claims)

### S8 · official · llm-d — Prefix-Cache Aware Routing (llm-d docs) (2026, docs/dev (live))

https://llm-d.ai/docs/dev/architecture/advanced/kv-management/prefix-cache-aware-routing · accessed 2026-09-17

*Primary mechanism description for how llm-d's router learns and scores KV-cache state, in both approximate and precise modes.*

_Covers: replication-routing, stack-llm-d, multi-node-kv-management_

Facts:
- Approximate mode: approx-prefix-cache-producer splits prompts into fixed-size blocks (example: 16 tokens approximated as characters) and builds a rolling hash chain; EPP keeps an in-memory LRU index of which prefix hashes were recently sent to which pods; prefix-cache-scorer scores on ratio of matched blocks to total prompt blocks (Approximate Implementation)
- Precise mode: model servers (vLLM) emit KVEvents over ZeroMQ whenever internal KV cache changes (blocks added/evicted); the KV-Cache Indexer subscribes and maintains a globally consistent view of exactly which token blocks reside on which pods (Precise Implementation)
- Precise mode uses vLLM's HTTP render endpoint (/v1/completions/render) to obtain exact token IDs for tokenization (Precise Implementation, tokenization approach)

### S9 · repo · Kubernetes SIGs — kubernetes-sigs/gateway-api-inference-extension (README) (2026, GA since v1.0.0)

https://github.com/kubernetes-sigs/gateway-api-inference-extension · accessed 2026-09-17

*Official Kubernetes SIG repository defining the InferencePool API and Endpoint Picker Protocol that llm-d, AIBrix, and others build on.*

_Covers: replication-routing, stack-llm-d, k8s-ops_

Facts:
- Project optimizes self-hosting generative models on Kubernetes by leveraging Envoy's External Processing to transform gateways into inference gateways (Project description)
- The Endpoint Picker (EPP) is a data-plane component communicating via the Envoy external processing protocol, acting as the Router, intercepting requests and routing to optimal model server replicas (EPP description)
- Goals: request scheduling algorithm that is KV-cache and request-cost aware; route client model names to use-case-specific LoRA adapters; prefix-cache aware load balancing on roadmap (Named features)

### S10 · official · Kubernetes SIGs — InferencePool (Kubernetes Gateway API Inference Extension docs) (2026, InferencePool GA since v1.0.0; endpointPickerRef optional since v1.5.0)

https://gateway-api-inference-extension.sigs.k8s.io/api-types/inferencepool/ · accessed 2026-09-17

*The formal API spec for InferencePool, the CRD every EPP-based router (llm-d, AIBrix) is built against.*

_Covers: replication-routing, k8s-ops_

Facts:
- InferencePoolSpec has three core fields: selector (must exactly match model server pod labels), targetPorts (ports the Inference Gateway routes to), endpointPickerRef (references the EPP service) (InferencePoolSpec fields)
- An HTTPRoute may reference multiple InferencePools as backendRefs (HTTPRoute relationship)
- Until release v1.5.0, endpointPickerRef was required; now optional, to allow InferencePool usage without a user-managed EPP deployment (endpointPickerRef note)

### S11 · repo · Kubernetes SIGs — EPP Architecture Proposal (0683-epp-architecture-proposal) (2026, main branch proposal doc)

https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/docs/proposals/0683-epp-architecture-proposal/README.md · accessed 2026-09-17

*Design proposal for the layered EPP architecture (Routing / Flow Controller / Scheduling layers) that llm-d's scheduler implements.*

_Covers: replication-routing, stack-llm-d_

Facts:
- Requests flow through three layers: Routing Layer (per-InferenceModel routing rules and request enrichment) -> Flow Controller (priority, fairness, queueing) -> Scheduling Layer (load-balancing algorithm routing based on current InferencePool state) (Routing decision process)

### S12 · official · vLLM Project — Disaggregated Prefilling (vLLM Documentation) (2026, docs.vllm.ai/en/latest (experimental feature); vLLM package version 0.29.0 released 2026-09-09 per PyPI)

https://docs.vllm.ai/en/latest/features/disagg_prefill/ · accessed 2026-09-17

*vLLM's own documentation of its KV-connector abstraction for disaggregated prefill/decode, the mechanism that every connector (LMCache, NIXL, Mooncake) plugs into.*

_Covers: pd-disaggregation, stack-vllm-multinode, multi-node-kv-management_

Facts:
- vLLM disaggregation separates prefill instance and decode instance, using a connector to transfer prefill KV caches and results (Mechanism overview)
- Three core abstractions: Connector (kv consumer retrieves KV caches of a batch from kv producer), LookupBuffer (insert/drop_select APIs), Pipe (single-direction FIFO for tensor transmission) (Architecture abstractions)
- Named connectors: ExampleConnector, LMCacheConnectorV1, NixlConnector, MooncakeConnector, MoRIIOConnector (ROCm only), MultiConnector, OffloadingConnector, FlexKVConnectorV1, PyNcclConnector (Connector list)
- The feature is labelled experimental and subject to change (Document banner)

### S13 · repo · vLLM Project — vllm-project/production-stack (README) (2026, released 2026-01-22)

https://github.com/vllm-project/production-stack · accessed 2026-09-17

*Reference Kubernetes deployment stack for vLLM: router + serving-engine pods + observability, maintained by the vLLM project itself.*

_Covers: stack-vllm-multinode, replication-routing, k8s-ops_

Facts:
- Production Stack scales from a single vLLM instance to a distributed deployment without application code changes; three components: serving engine, request router, observability stack (Prometheus + Grafana) (README architecture)
- Router supports round-robin, session-ID based routing, and automatic Kubernetes service discovery (Router description)
- Integrates KV cache offloading via LMCache (tutorial 6: 'How to Enable KV Cache Offloading with LMCache') (Tutorials list)

### S14 · official · vLLM Project — Expert Parallel Deployment (vLLM Documentation) (2026, docs.vllm.ai/en/latest (live))

https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/ · accessed 2026-09-17

*vLLM's own guide to DP+EP configuration for MoE models across multiple nodes, with concrete CLI flags.*

_Covers: wide-ep-moe, stack-vllm-multinode_

Facts:
- vLLM computes EP_SIZE = TP_SIZE x DP_SIZE; EP is more efficient combined with DP than alone (DP+EP configuration)
- With TP=1, attention weights are replicated across all DP ranks; MoE layers are sharded across all EP ranks using the full EP group size (Attention-DP + Expert-Parallel layout)
- 2-node deployment example: --data-parallel-size 16 (total), --data-parallel-size-local 8 (per node), --data-parallel-start-rank 8 for the second node (Multi-node example)
- EPLB integration parameters: window_size 1000 steps for tracking, step_interval 3000 for rebalancing frequency; memory overhead approximately 2.4 GB per redundant expert per EP rank on DeepSeek-V3 (EPLB integration)
- Single-node H200 example command: vllm serve deepseek-ai/DeepSeek-V3-0324 --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel (Example commands)

### S15 · official · vLLM Project — Metrics (vLLM Design Documentation) (2026, docs.vllm.ai/en/latest/design/metrics (v1 engine))

https://docs.vllm.ai/en/latest/design/metrics/ · accessed 2026-09-17

*Canonical list of Prometheus metrics vLLM exposes, the substrate for goodput/SLO dashboards and autoscaling in any multi-node deployment.*

_Covers: k8s-ops, capacity-economics_

Facts:
- Named gauges: vllm:num_requests_running, vllm:kv_cache_usage_perc (fraction of used KV cache blocks, 0-1), vllm:cache_config_info, vllm:lora_requests_info (Gauges list)
- Named histograms: vllm:time_to_first_token_seconds, vllm:inter_token_latency_seconds, vllm:request_time_per_output_token_seconds, vllm:e2e_request_latency_seconds, vllm:request_prefill_time_seconds, vllm:request_decode_time_seconds, vllm:kv_block_lifetime_seconds, vllm:kv_block_idle_before_evict_seconds, vllm:kv_block_reuse_gap_seconds (Histograms list)
- Deprecated in v1: vllm:num_requests_swapped, vllm:cpu_cache_usage_perc, vllm:time_in_queue_requests (superseded by request_queue_time_seconds) (Deprecated metrics)

### S16 · official · SGLang / RadixArk — PD Disaggregation (SGLang Documentation) (2026, docs.sglang.io (live))

https://docs.sglang.io/advanced_features/pd_disaggregation.html · accessed 2026-09-17

*SGLang's own documentation of its prefill/decode disaggregation implementation, transfer backends, and heterogeneous-TP staging buffer optimization.*

_Covers: pd-disaggregation, stack-sglang, wide-ep-moe_

Facts:
- Disaggregation targets two unified-engine problems: Prefill Interruption (incoming batches delay token generation) and DP Attention Imbalance (parallel workers process mixed workloads) (Core Architecture)
- Two transfer backends: Mooncake (installed via mooncake-transfer-engine, configurable per-GPU InfiniBand device mapping) and NIXL (installed via pip install nixl, UCX backend by default, LIBFABRIC via SGLANG_DISAGGREGATION_NIXL_BACKEND) (Transfer Backends)
- Multi-node DeepSeek example uses TP=16, DP=8, --enable-dp-attention, --moe-a2a-backend deepep across separate prefill/decode master nodes (Deployment Patterns)
- Heterogeneous TP with GPU staging buffer enables 2-5x throughput improvement when prefill/decode use different TP sizes; incompatible with MLA models (DeepSeek-V2/V3) (Advanced Features)
- Decode workers use heartbeat monitoring (default SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL=5.0s) to detect prefill failures; bootstrap handshake has default 300s timeout (Bootstrap Handshake & KV Transfer)

### S17 · repo · SGLang / RadixArk — sglang-router (PyPI package page) (2026, 0.3.2 (2026-01-15))

https://pypi.org/project/sglang-router/ · accessed 2026-09-17

*Version-of-record for SGLang's standalone Rust router, which implements PD-aware load balancing.*

_Covers: stack-sglang, replication-routing_

Facts:
- sglang-router is described as a high-performance Rust-based load balancer for SGLang with multiple routing algorithms and prefill-decode disaggregation support, current version 0.3.2 released 2026-01-15 (Package metadata)

### S18 · blog · LMSYS / SGLang team — Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs (2025, published 2025-05-05)

https://www.lmsys.org/blog/2025-05-05-large-scale-ep/ · accessed 2026-09-17

*First open-source reproduction of DeepSeek-scale wide-EP + PD disaggregation, with concrete GPU counts and per-node throughput matched against DeepSeek's own published numbers.*

_Covers: wide-ep-moe, stack-sglang, stack-deepep-deepseek, capacity-economics_

Facts:
- Deployment: 12 nodes on Atlas Cloud, 8 H100 GPUs each (96 GPUs total); prefill tested at EP32 (4 nodes), decode tested at EP72 (9 nodes) (Hardware Architecture)
- Prefill throughput per node: 57,674 / 54,543 / 50,302 tokens/s for prompt lengths 1K/2K/4K respectively; decode: 22,282 tokens/s per node for 2K inputs; combined headline: 52.3k input tok/s and 22.3k output tok/s per node (Throughput Performance)
- Resulting cost: $0.20/1M output tokens, about one-fifth the official DeepSeek Chat API price; up to 5x higher output throughput than vanilla TP on the same resources (Cost Comparison)
- SGLang prefill with simulated optimal load balancing was within 5.6% of DeepSeek's official profile; decode was 6.6% below DeepSeek's official profile (Comparison to Official DeepSeek)
- Integrates DeepEP, DeepGEMM, and EPLB (Software Components)

Conflicts:
- S29 (DeepSeek's own Day-6 blog) reports 73.7k tok/s/node prefill and 14.8k tok/s/node decode on H800 with its production EP32/EP144 config; SGLang's reproduction reports 52.3k/22.3k on H100 at EP32/EP72 — different hardware (H800 vs H100) and different EP degree for decode, so figures are not directly comparable, only both cited as 'near DeepSeek-scale'.

### S19 · blog · LMSYS / RadixArk / Google Cloud — RadixArk Joins Forces with Google to Bring Full SGLang Features to TPUs (2026, published 2026-07-30)

https://www.lmsys.org/blog/2026-07-30-sglang-google-tpu/ · accessed 2026-09-17

*Announces the SGL-JAX backend bringing SGLang's router/PD/wide-EP stack to Google TPUs — the cross-cutting bridge between the GPU-centric open-source stack and TPU serving.*

_Covers: stack-sglang, tpu-multihost_

Facts:
- SGLang runs on the latest TPU generations today via SGL-JAX, supporting model families including Gemma, Qwen, DeepSeek, GLM, Mimo, Kimi, Ling, MiniMax, and Grok, plus diffusion models Wan and Flux (Current and Upcoming Support)
- RadixArk will add SGL-torchtpu, a PyTorch-native TPU backend with eager execution and MPMD support; features include data/tensor/expert/context/pipeline parallelism, Radix Cache, HiCache, quantization, and speculative decoding on TPU Pallas kernels (Future Support / Key Features)

### S20 · official · NVIDIA — Disaggregated Serving in TensorRT LLM (Tech Blog) (2026, nvidia.github.io/TensorRT-LLM (blog5))

https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html · accessed 2026-09-17

*NVIDIA's own dated benchmark numbers for TensorRT-LLM disaggregated serving across multiple models, ISL/OSL ratios, and GPU counts on GB200.*

_Covers: pd-disaggregation, stack-trtllm-triton, capacity-economics_

Facts:
- DeepSeek R1 (ISL 4400, OSL 1200): disaggregation speedup 1.4x-1.8x without MTP, 1.6x-2.5x with MTP (20-30% higher than MTP-off) (DeepSeek R1 results)
- DeepSeek R1 (ISL 8192, OSL 256): up to 1.73x speedup with GEN4, up to 2x with GEN8 (ISL 8192 OSL 256 results)
- Qwen3 (ISL 8192, OSL 1024): disaggregation speedups over aggregation range 1.7x to 6.11x (Qwen3 results)
- Terminology: context servers (prefill) and generation servers (decode); communication protocols MPI, UCX, and NIXL all supported, using RDMA/NVLink (Architecture description)
- Testing conducted on GB200 GPUs across configurations from 4 to 32 GPUs per instance (Test setup)

Conflicts:
- Speedup magnitude varies 1.4x-6.11x purely by ISL:OSL ratio and model within the same source; contradicts any single-number 'disaggregation gives Nx' claim from other sources — the number is workload-shape-dependent, not a hardware constant.

### S21 · official · NVIDIA — Disaggregated Serving (TensorRT-LLM docs, disagg-serving.md) (2026, main branch docs)

https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/disagg-serving.md · accessed 2026-09-17

*Implementation-level doc for TensorRT-LLM's disaggregated_params/cache-transceiver mechanism, complementing the tech-blog benchmark numbers.*

_Covers: pd-disaggregation, stack-trtllm-triton_

Facts:
- Context and generation phases of one request must share a single request ID (DisaggregatedParams.disagg_request_id) because KV-cache transfer is keyed by it (Disaggregated Params mechanism)
- The KV cache exchange module is responsible for efficient transmission/reception of the cache, implemented via KvCacheTransceiverV2; default backend is NIXL, transferring over RDMA/NVLink, with TRTLLM_NIXL_KVCACHE_BACKEND selecting UCX (default) or LIBFABRIC (Cache Transceiver section)
- Dynamic node joining/leaving for disaggregated deployments is being built on top of the existing mechanism (in progress, not yet GA) (Status note)

### S22 · paper · Moonshot AI / Tsinghua University (MADSys) — Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving (2024, arXiv:2407.00079v4 (2025-09-03); Best Paper Award, USENIX FAST 2025)

https://arxiv.org/pdf/2407.00079 · accessed 2026-09-17

*The originating peer-reviewed paper (FAST 2025 Best Paper) for the KVCache-centric disaggregated architecture that underlies Mooncake, cited/implemented by vLLM, SGLang, and LMDeploy.*

_Covers: multi-node-kv-management, stack-mooncake, pd-disaggregation, frontier-research_

Facts:
- Mooncake is the serving platform for Kimi (Moonshot AI); architecture separates prefill and decoding clusters and leverages underutilized CPU/DRAM/SSD/RDMA of the GPU cluster to implement a disaggregated KVCache (Abstract / §1)
- Compared to baseline, Mooncake achieves up to 525% throughput increase in simulated scenarios while adhering to SLOs; under real workloads, enables handling 75% more requests (Abstract)
- Global scheduler is named Conductor, dispatching requests based on current KVCache distribution and workload; per-request workflow has 4 steps: KVCache Reuse, Incremental Prefill, KVCache Transfer, Decoding (§3, Figure 4)
- SLO definitions used in experiments: TTFT_P90 = 10x baseline single-request TTFT, TBT_P90 = 5x baseline (§2, Preliminary)
- KVCache stored as paged blocks in CPU memory with hash values (block size 512 tokens) determined by own hash and prefix, for deduplication; transfer handled by a GPUDirect-RDMA-based component called Messenger (§3, Figure 3 caption, §4)
- Prefill chunking threshold (prefill_chunk) is typically larger than 1000 tokens (§3, step 2 Incremental Prefill)
- Open-sourced 1-hour real-world request trace: 23,608 entries, average input length 7,590 tokens, average output length 182 tokens, average input:output ratio ~720; cache hit ratio rises from 30% to 50% as block capacity grows from 1,000 to 50,000 (LRU/LFU/LengthAwareCache table) (§4.2, Table 1)
- Paper explicitly debates whether prefill/decode separation is still needed given chunked prefill, and decides to keep it disaggregated for two reasons: cross-node parallelism needs for long context, and VRAM savings (§5)

### S23 · repo · Moonshot AI / Tsinghua MADSys — kvcache-ai/Mooncake (README) (2026, main branch, updates through August 2026)

https://github.com/kvcache-ai/Mooncake/blob/main/README.md · accessed 2026-09-17

*The open-source implementation repository: Transfer Engine, Store, and EP/Process-Group components, plus measured RoCE bandwidth numbers not in the paper.*

_Covers: stack-mooncake, multi-node-kv-management, interconnect-transport_

Facts:
- Repo provides Transfer Engine (batched data movement across storage/network/accelerator), Mooncake Store (distributed KV-cache storage engine), and Mooncake EP & Process Group (fault-tolerant distributed execution for large-scale MoE inference) (README components)
- Transfer Engine supports TCP, RDMA, AWS EFA, NVMe-oF, NVLink, HIP, Barex, CXL, and Ascend-family transports (README supported transports)
- With 40 GB of data, Transfer Engine delivers up to 87 GB/s on 4x200 Gbps RoCE and 190 GB/s on 8x400 Gbps RoCE, about 2.4x and 4.6x faster than TCP respectively (README benchmark)
- vLLM and SGLang both officially support Mooncake Transfer Engine for disaggregated prefilling; Mooncake became a PD-disaggregation backend for LMDeploy in June 2025 (Integration timeline)

### S24 · official · LMCache — Example: Offload KV cache to CPU (LMCache Documentation) (2026, docs.lmcache.ai (live))

https://docs.lmcache.ai/getting_started/quickstart/offload_kv_cache.html · accessed 2026-09-17

*Primary quickstart doc for LMCache's tiered KV-cache offload mechanism and its vLLM connector configuration.*

_Covers: stack-lmcache, multi-node-kv-management_

Facts:
- LMCache automatically stores and retrieves KV cache from secondary storage when GPU memory is insufficient, using chunked transfer (default chunk size 256 tokens) (Mechanism description)
- Supported offload backends: CPU memory, local file system, Mooncake Storage, InfiniStore, Redis, ValKey (Backend list)
- vLLM integration via KVTransferConfig: connector name 'LMCacheConnectorV1', role 'kv_both' (vLLM integration example)
- Example config: CPU memory limit 5.0 GB, ~1.5 GB per 10,000 tokens; observed ~7.43x speedup on second-run inference with CPU offloading enabled (Configuration example / results)

### S24b · repo · LMCache — LMCache/LMCache (README) (2026, main branch)

https://github.com/LMCache/LMCache · accessed 2026-09-17

*Project-level description of LMCache's storage-backend plugin surface and its integration timeline with vLLM and Dynamo.*

_Covers: stack-lmcache, multi-node-kv-management_

Facts:
- LMCache is described as a KV cache management layer for LLM inference that turns KV cache from temporary state into reusable 'AI-native knowledge' (README description)
- Pluggable storage backends: CPU RAM, local disk (SSD), Redis/Valkey, Mooncake, InfiniStore, S3-compatible object storage, NIXL, and GDS (Storage backends list)
- LMCache extended multimodal support in vLLM V1 (July 2025 update); NVIDIA Dynamo integrated LMCache in September 2025 (Integration timeline)

### S25 · official · NVIDIA (ai-dynamo) — NIXL Architecture (nixl.md) (2026, main branch docs)

https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md · accessed 2026-09-17

*The primary architecture document for NIXL, the point-to-point transfer library underlying Dynamo, TensorRT-LLM, and SGLang's KV-transfer paths.*

_Covers: stack-nixl, interconnect-transport, multi-node-kv-management_

Facts:
- NIXL is targeted for accelerating point-to-point communications in AI inference frameworks such as Dynamo, providing abstraction over memory (CPU/GPU) and storage (file/block/object) types via a modular plug-in architecture (Overview)
- Three core abstractions: Memory Sections (unify memory/storage types behind a buffer-list primitive), Transfer Backend Interface (selects optimal backend), Metadata Handler (establishes agent-to-agent communication, caches metadata) (Key Abstractions)
- Actively supported backends: UCX and NVIDIA Magnum IO GPUDirect Storage (GDS); other filesystem/block/object backends in development; transfer protocols include RoCE, InfiniBand, GPUDirect RDMA, NVMe-oF, TCP, and NVLink (Supported Backends)

### S26 · repo · NVIDIA — ai-dynamo/nixl Releases (2026, v1.4.1 (2026-09-01))

https://github.com/ai-dynamo/nixl/releases · accessed 2026-09-17

*Version-of-record for NIXL, showing the pace of change (bug fixes and format optimizations shipping weekly).*

_Covers: stack-nixl_

Facts:
- Latest release v1.4.1 (2026-09-01) closes a use-after-free window in the transfer-request path and restores num_experts parameter for CUDA graph reuse across rank changes (v1.4.1 release notes)
- v1.4.0 introduced stride (compressed) descriptors reducing memory from ~44 MB to ~7 KB, and a new nixl::trace tracing framework with NVTX backend for Nsight Systems integration (v1.4.0 release notes)
- v1.3.2 fixed a cross-node EFA transfer hang introduced in 1.3.1 via a per-rail flush strategy (v1.3.2 release notes)

### S27 · repo · DeepSeek-AI — deepseek-ai/DeepEP (README) (2026, main branch (v2 kernels))

https://github.com/deepseek-ai/DeepEP/blob/main/README.md · accessed 2026-09-17

*The primary repository for DeepEP, the expert-parallel all-to-all communication library that every wide-EP deployment (vLLM, SGLang, Dynamo) integrates for MoE dispatch/combine.*

_Covers: wide-ep-moe, stack-deepep-deepseek, interconnect-transport_

Facts:
- DeepEP provides high-throughput, low-latency all-to-all GPU kernels (MoE dispatch and combine) with low-precision (FP8) support; normal kernels optimize NVLink-to-RDMA-domain forwarding for training/prefill, low-latency kernels use pure RDMA for decode (README overview)
- Benchmark (8K tokens/batch, 7168 hidden dim, top-8 experts, FP8 dispatch/BF16 combine): NVLink SM100 EP8 dispatch 726 GB/s / combine 740 GB/s at 64 SMs (643/675 GB/s at 24 SMs, min config); RDMA SM90 CX7 EP8x2 dispatch/combine 90/81 GB/s at 12 SMs; RDMA SM90 EP8x4 61/61 GB/s at 6 SMs; RDMA SM100 EP8x2 90/91 GB/s at 12 SMs (Performance table)
- Requires Hopper (SM90) GPUs or newer, CUDA 12.3+ for SM90; kernels JIT-compiled at runtime, no CUDA compilation needed at install (Requirements)

### S28 · paper · DeepSeek-AI — DeepSeek-V3 Technical Report (2024, arXiv:2412.19437)

https://arxiv.org/pdf/2412.19437 · accessed 2026-09-17

*The originating technical report for DeepSeek-V3's inference deployment strategy (§3.4), the primary published numbers for cross-node EP degree in production.*

_Covers: wide-ep-moe, stack-deepep-deepseek, frontier-research, interconnect-transport_

Facts:
- DeepSeek-V3 is deployed on an H800 cluster; GPUs within a node connect via NVLink, all GPUs across the cluster connect via IB; prefill and decode stages are deployed separately (§3.4, Inference and Deployment)
- Prefilling minimum deployment unit: 4 nodes, 32 GPUs. Attention: TP4 with Sequence Parallelism, combined with DP8. MoE: EP32. Dense MLPs in shallow layers use 1-way TP. 32 redundant experts deployed (each GPU hosts its original 8 experts plus 1 redundant expert), rearranged every ~10 minutes; two micro-batches overlap attention+MoE of one with dispatch+combine of the other (§3.4.1, Prefilling)
- Decoding minimum deployment unit: 40 nodes, 320 GPUs. Attention: TP4 with SP, combined with DP80. MoE: EP320, each GPU hosts only one expert; 64 GPUs are responsible for hosting redundant and shared experts; all-to-all dispatch/combine performed via direct point-to-point transfers over IB using IBGDA technology (§3.4.2, Decoding)
- Decode batch size per expert is typically within 256 tokens; decode is memory-access bound, not compute bound, so only a small portion of SMs is allocated to dispatch+MoE+combine (§3.4.2, closing paragraph)
- Hardware-design suggestion: current all-to-all communication implementation allocates 20 of the 132 SMs on H800 for communication, an inefficiency the paper asks vendors to address with dedicated co-processor hardware (§3.5.1, Communication Hardware)

Conflicts:
- S29 (DeepSeek's Feb 2025 Day-6 Open Source Week blog, a later production snapshot) reports decoding uses EP144 across 18 nodes (144 GPUs) with each GPU hosting 2 routed experts + 1 shared expert — a much smaller decode deployment unit than this report's original EP320/40-node/320-GPU configuration from December 2024. The company appears to have significantly shrunk the minimum decode deployment unit between the two publications.

### S29 · repo · DeepSeek-AI (open-infra-index, #OpenSourceWeek) — Day 6 (One More Thing): DeepSeek-V3/R1 Inference System Overview (2025, published 2025-02 (Open Source Week Day 6))

https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md · accessed 2026-09-17

*DeepSeek's own later production-deployment disclosure with per-node throughput and real cost/profit figures for its online service — the closest thing to ground truth for wide-EP economics at hyperscale.*

_Covers: wide-ep-moe, stack-deepep-deepseek, capacity-economics, frontier-research_

Facts:
- Prefilling: Routed Expert EP32, MLA/Shared Expert DP32, across 4 nodes; each GPU handles 9 routed experts and 1 shared expert (Parallelism Configurations)
- Decoding: Routed Expert EP144, MLA/Shared Expert DP144, across 18 nodes; each GPU manages 2 routed experts and 1 shared expert (Parallelism Configurations)
- Each H800 node delivers average throughput of ~73.7k tokens/s input (including cache hits) during prefilling, and ~14.8k tokens/s output during decoding (Throughput Metrics)
- Daily operational cost $87,072; theoretical daily revenue at R1 pricing $562,027, yielding 545% cost profit margin (actual revenue lower due to V3's cheaper pricing and discounts) (Cost and Profitability)
- Three load balancers used: prefill, decode, and expert-parallel, addressing computational and dispatch imbalance; two microbatches executed alternately so communication cost of one is hidden behind computation of the other (Load Balancing section)

Conflicts:
- See S28 conflict note: this blog's EP144/18-node decode configuration is much smaller than the technical report's EP320/40-node configuration published ~2 months earlier.

### S30 · repo · DeepSeek-AI — deepseek-ai/EPLB (README) (2026, main branch)

https://github.com/deepseek-ai/EPLB/blob/main/README.md · accessed 2026-09-17

*The open-sourced reference implementation of DeepSeek's expert-parallel load balancing algorithm (redundant-expert replication + placement), used across the wide-EP ecosystem (vLLM, SGLang).*

_Covers: wide-ep-moe, stack-deepep-deepseek_

Facts:
- Core strategy: redundant-experts approach duplicates heavy-loaded experts, then heuristically packs duplicated experts to GPUs for load balance; group-limited expert routing tries to place experts of the same group on the same node to reduce inter-node traffic (Overview)
- Hierarchical load balancing policy: pack expert groups to nodes evenly first, then replicate experts within each node; used in prefilling stage with smaller expert-parallel size (Hierarchical Load Balancing)
- Global load balancing policy: replicates experts globally regardless of expert groups, packs to individual GPUs; used in decoding stage with larger expert-parallel size (Global Load Balancing)
- Main function eplb.rebalance_experts computes a balanced expert replication/placement plan from estimated expert loads (Implementation)

### S31 · repo · ByteDance (now under vllm-project) — vllm-project/aibrix (README) (2026, v0.7.0 (2026-06-16))

https://github.com/vllm-project/aibrix · accessed 2026-09-17

*ByteDance's open-source Kubernetes control plane for vLLM, now hosted under the vllm-project org, contributing to Gateway API Inference Extension standardization.*

_Covers: stack-orchestration, k8s-ops, replication-routing_

Facts:
- AIBrix components: LLM Gateway and Routing, LLM App-Tailored Autoscaler, Distributed KV Cache (cross-engine reuse), Unified AI Runtime sidecar, High-Density LoRA Management, GPU Hardware Failure Detection, cost-efficient heterogeneous serving with SLO guarantees (README architecture)
- Latest release v0.7.0, dated 2026-06-16 (Release info)
- ByteDance collaborated with Google to drive standardization of LLM serving on Kubernetes via Working Group Serving, contributing to the Gateway API Inference Extension (Collaboration note)

### S32 · official · Anyscale (Ray project) — Serving LLMs (Ray Serve Documentation) (2026, Ray 2.58.0)

https://docs.ray.io/en/latest/serve/llm/index.html · accessed 2026-09-17

*Official Ray Serve documentation for its LLM serving primitives, covering multi-node deployment configuration.*

_Covers: stack-orchestration, k8s-ops_

Facts:
- Ray Serve LLM builds on Ray Serve primitives for distributed, multi-node LLM serving and exposes an OpenAI-compatible API; supports multi-node, multi-model deployment with autoscaling and load balancing (Overview)
- Named interfaces: LLMConfig (deployment configuration object), build_openai_app (constructs OpenAI-compatible app), serve.run() (execution) (Configuration example)

### S33 · official · KServe (CNCF) — Multi-node/Multi-GPU Inference (KServe docs) (2026, kserve.github.io/website (live))

https://kserve.github.io/website/docs/model-serving/generative-inference/multi-node · accessed 2026-09-17

*Official KServe documentation for its multi-node ServingRuntime built on vLLM, with concrete parallelism-to-GPU-count arithmetic.*

_Covers: stack-orchestration, k8s-ops_

Facts:
- tensorParallelSize configures how model weights are sharded across GPUs; pipelineParallelSize determines how model layers are split across GPUs; Total GPUs = tensorParallelSize x pipelineParallelSize (Parameter definitions)
- 16-GPU/2-node example: pipelineParallelSize=2, tensorParallelSize=8, workerSpec.resources.requests.gpu=8, resulting in vllm serve --tensor-parallel-size 8 --pipeline-parallel-size 2 (Case 1 example)
- Supported ServingRuntime is kserve-huggingfaceserver-multinode, built on vLLM's multi-node/multi-GPU feature (Supported runtime)

### S34 · repo · Kubernetes SIGs — kubernetes-sigs/lws (LeaderWorkerSet README) (2026, main branch)

https://github.com/kubernetes-sigs/lws/blob/main/README.md · accessed 2026-09-17

*The Kubernetes SIG API for deploying a group of pods as a unit of replication, the substrate underneath most multi-node inference deployment topologies (vLLM, SGLang, JetStream on GKE).*

_Covers: k8s-ops, stack-orchestration_

Facts:
- LeaderWorkerSet is an API for deploying a group of pods as a unit of replication, addressing multi-host inference workloads where the LLM is sharded and run across multiple devices on multiple nodes (README description)
- LWS and llm-d APIs are being co-designed together (llm-d is a CNCF Sandbox project) (Collaboration note)

### S35 · paper · Peking University / UC San Diego / StepFun; USENIX OSDI 2024 — DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving (2024, arXiv:2401.09670v3 (2024-06-06))

https://arxiv.org/pdf/2401.09670 · accessed 2026-09-17

*The foundational OSDI 2024 paper defining goodput-optimized prefill/decode disaggregation and the placement algorithms every later system (Mooncake, Dynamo, TensorRT-LLM, SGLang) cites.*

_Covers: pd-disaggregation, frontier-research, capacity-economics_

Facts:
- DistServe serves up to 7.4x more requests or 12.6x tighter SLO than state-of-the-art systems while staying within latency constraints for >90% of requests (Abstract)
- Per-GPU goodput defined as the maximum request rate servable adhering to SLO attainment goal (e.g. 90%) for each provisioned GPU; on a single A100 serving a 13B LLM, colocated per-GPU goodput is only ~1.6 rps versus 5.6 rps prefill-only / 10 rps decode-only when phases are separated (§1, Figure 1 discussion)
- Cluster testbed: 4 nodes, 32 GPUs; each node 8x NVIDIA 80GB A100 SXM connected with NVLink; cross-node bandwidth 25 Gbps (§6.1)
- KV cache transmission overhead is insubstantial: even for OPT-175B, KV cache transmission accounts for less than 0.1% of total latency; over 95% of requests experience transfer delay under 30ms despite limited cross-node bandwidth, because the placement algorithm keeps same-stage prefill/decode segments on one node using intra-node NVLink (§6.3, Figure 10)
- Two placement algorithms: high node-affinity (Infiniband clusters, negligible cross-node transfer overhead, any two nodes can host prefill/decode) vs low node-affinity (colocate prefill/decode instance segments of the same pipeline stage on one node to use NVLink, required when cross-node bandwidth is limited) (§4.1-4.2)
- Fault tolerance and preemption are explicitly out of scope / future work; a fault in a decode instance mapped to multiple prefill instances could cripple the entire cluster due to prefill-decode dependency (§4.3, Preemption and fault tolerance)

Conflicts:
- S44 (SemiAnalysis InferenceMAX) finds disaggregation's TCO advantage is conditional on interactivity target (crosses over around 90 tok/s/user on GB200 NVL72 vs 8-GPU B200), whereas this paper's framing treats disaggregation as a near-unconditional win for goodput under fixed SLOs on A100-era hardware — the two are reconciled by noting DistServe measures request-rate goodput at fixed SLOs while SemiAnalysis measures TCO per token across a range of interactivity targets on newer hardware where single-node aggregated throughput is much higher.

### S36 · paper · University of Washington / Microsoft; ISCA 2024 — Splitwise: Efficient Generative LLM Inference Using Phase Splitting (2023, arXiv:2311.18677v2)

https://arxiv.org/pdf/2311.18677 · accessed 2026-09-17

*The Microsoft/UW ISCA 2024 paper that independently proposed phase-splitting (prefill/decode) around the same time as DistServe, with a production-trace-driven cluster design methodology.*

_Covers: pd-disaggregation, frontier-research, interconnect-transport_

Facts:
- A100 vs H100: TFLOPs 19.5 vs 66.9 (3.43x), HBM capacity 80GB vs 80GB (1.00x), HBM bandwidth 2039 vs 3352 GBps (1.64x), power 400W vs 700W (1.75x), NVLink 50 vs 100 Gbps [per-lane], InfiniBand 200 vs 400 GBps, cost per machine $17.6/hr vs $38/hr (2.16x) (Table I)
- Splitwise-based clusters achieve 1.4x higher throughput at 20% lower cost than current designs, or alternatively 2.35x more throughput under the same cost and power budgets (Abstract)
- Traces from two production Azure LLM inference services (coding, conversation), 20 minutes long, captured November 11, 2023, publicly released (§III, Production traces)
- Coding service median prompt size 1500 tokens, median 13 generated tokens; conversation service median prompt 1020 tokens, near-bimodal generated distribution with median 129 tokens (§III.A)
- State transfer between prompt-computation and token-generation machines is implemented over back-end InfiniBand interconnects available in today's datacenters, achieving efficiency without perceived performance loss (§I)

### S37 · paper · Alibaba Group; USENIX OSDI 2024 — Llumnix: Dynamic Scheduling for Large Language Model Serving (2024, arXiv:2406.03243v1)

https://arxiv.org/pdf/2406.03243 · accessed 2026-09-17

*The OSDI 2024 paper introducing cross-instance live request migration for LLM serving, the mechanism underlying rebalancing/de-fragmentation/priority-scheduling/failure-recovery across a multi-instance fleet.*

_Covers: k8s-ops, frontier-research, pd-disaggregation_

Facts:
- Llumnix improves P99 first-token latency by up to 15x and P99 per-token generation latency by up to 2x compared to state-of-the-art scheduler INFaaS, on a 16-GPU cluster with realistic workloads; accelerates high-priority requests by 1.5x and achieves 36% cost saving while delivering similar tail latencies (§1, Introduction summary)
- Four rescheduling scenarios unified via 'virtual usage': load balancing, de-fragmentation, prioritization, auto-scaling; live migration is near-zero downtime, constant to sequence length (§1, Figure 1)
- Implemented as a scheduling layer on top of vLLM; distributed architecture enabling continuous rescheduling with high scalability (§1, closing)
- Motivating experiment: LLaMA-7B on A10 GPU, 2,000-request Poisson trace with power-law input/output lengths, shows request preemptions rise sharply with memory pressure, significantly inflating P95/P99 latency (§3, Figure 3)

### S38 · blog · Stanford University, Scaling Intelligence Lab (SAIL) — Tokasaurus: An LLM Inference Engine for High-Throughput Workloads (2025, published 2025)

https://scalingintelligence.stanford.edu/blogs/tokasaurus/ · accessed 2026-09-17

*The real Stanford project in this space — verifies that 'Thunder Agent' is not a real named project; the actual Hazy Research/Stanford-adjacent multi-GPU serving work is Tokasaurus, ThunderMLA, and ThunderKittens.*

_Covers: frontier-research_

Facts:
- Tokasaurus is an LLM inference engine optimized for throughput-intensive workloads, built by Stanford's Scaling Intelligence Lab (SAIL) (Introduction)
- Mechanisms: dynamic Hydragen grouping to exploit shared prefixes, asynchronous/adaptive CPU management to minimize overhead, pipeline parallelism for GPUs without NVLink, async tensor parallelism for NVLink-equipped GPUs (Key mechanisms)
- Outperforms vLLM and SGLang by up to 3x+ on throughput benchmarks; improves throughput by over 3x versus vLLM and SGLang on Llama-3.1-70B with eight L40S GPUs (Performance comparisons)

### S39 · blog · Stanford University, Hazy Research — ThunderMLA: FlashMLA, Faster and Fused-er! (2025, published 2025-03-04)

https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla · accessed 2026-09-17

*Hazy Research's own blog on ThunderMLA, its fused decode megakernel built on ThunderKittens — the actual Hazy Research multi-GPU-serving-adjacent kernel work Aman is referring to.*

_Covers: frontier-research, wide-ep-moe_

Facts:
- ThunderMLA is a completely fused 'megakernel' for decode that is 20-35% faster than DeepSeek's FlashMLA on diverse workloads (Introduction)
- Megakernel design eliminates kernel-launch overhead by creating a virtual instruction set on the GPU, fusing kernels via an interpreter template within ThunderKittens (Design description)
- Representative workload (4 prompts, 4 tokens): ThunderMLA runs in 41 us achieving 183 TFLOPs / 1520 GB/s vs FlashMLA's 52 us achieving 144 TFLOPS / 1199 GB/s on an SXM H100 (Benchmark example)
- Published March 4, 2025 (Post metadata)

### S40 · official · NVIDIA — GB200 NVL72 (product page) (2026, n/a (live product page))

https://www.nvidia.com/en-us/data-center/gb200-nvl72/ · accessed 2026-09-17

*Vendor spec sheet for the GB200 NVL72 rack, the reference rack-scale unit for what changes when the NVLink domain grows past one 8-GPU node.*

_Covers: interconnect-transport, wide-ep-moe_

Facts:
- GB200 NVL72 configuration: 36 Grace CPUs, 72 Blackwell GPUs (Specs table)
- NVLink bandwidth across the rack: 130 TB/s (Specs table)
- GPU memory: 13.4 TB HBM3E at 576 TB/s aggregate bandwidth (Specs table)
- NVFP4 Tensor Core 1,440/720 PFLOPS (sparse/dense); FP8/FP6 720 PFLOPS; FP32 5,760 TFLOPS; FP64 2,880 TFLOPS (Specs table)
- Networking beyond the rack uses NVIDIA Quantum-X800 InfiniBand, Spectrum-X800 Ethernet, and BlueField-3 DPUs for scaling across hundreds/thousands of Blackwell GPUs (Networking section)

### S41 · official · NVIDIA — NVIDIA ConnectX-7 NDR 400G InfiniBand Adapter Card (datasheet) (2021, Datasheet Apr21)

https://www.nvidia.com/content/dam/en-zz/Solutions/networking/infiniband-adapters/infiniband-connectx7-data-sheet.pdf · accessed 2026-09-17

*Vendor datasheet for ConnectX-7 NDR, the standard scale-out NIC deployed with H100/H200-generation clusters (predecessor to the Quantum-X800/ConnectX-8 generation used with GB200 NVL72).*

_Covers: interconnect-transport_

Facts:
- Max total bandwidth 400 Gb/s, IBTA Spec 1.5 compliant, 1/2/4 network ports, PCIe Gen5 up to x32 lanes, RDMA message rate 330-370 million messages per second (Product specifications table)
- Supports GPUDirect RDMA, GPUDirect Storage, hardware-based congestion control, out-of-order RDMA with adaptive routing, and NVIDIA In-Network Computing (collective operations, MPI Tag Matching, MPI All-to-All offloads) (Features list)
- HPC software libraries supported: NVIDIA HPC-X and UCX, UCC, NCCL, OpenMPI, MVAPICH, MPICH, OpenSHMEM, PGAS (HPC Software Libraries)

### S42 · blog · NVIDIA Developer Blog — Benchmarking GPUDirect RDMA on Modern Server Platforms (2014, n/a (historical, Ivy Bridge Xeon / K40m GPUs))

https://developer.nvidia.com/blog/benchmarking-gpudirect-rdma-on-modern-server-platforms/ · accessed 2026-09-17

*NVIDIA's own microbenchmark establishing the GPUDirect RDMA mechanism and latency floor; numbers are from 2014-era hardware (Ivy Bridge/K40, PCIe3, single/dual-rail FDR InfiniBand) and are two Infiniband generations out of date for bandwidth, but the latency-floor mechanism and 'faster than staging below ~400-500KB' crossover logic still hold architecturally.*

_Covers: interconnect-transport_

Facts:
- Host-to-Host latency measured with ibv_ud_pingpong is 1.3us; Host-to-GPU latency 1.7us via GPUDirect RDMA (Latency section)
- On a PCIe-switch platform, GPUDirect RDMA achieves dual-rail bandwidth up to 11.6 GB/s host-to-GPU and 7.4 GB/s peak GPU-to-GPU; on Ivy Bridge without a PCIe switch, inter-socket QPI-traversal GPU-to-GPU write bandwidth collapses to 250 MB/s (Bandwidth figures)
- GPUDirect RDMA is faster than the staging (bounce-buffer) approach for message sizes up to 400-500KB on Ivy Bridge Xeon (Trade-off statement)

Conflicts:
- Bandwidth figures here (single-digit GB/s, 2014 PCIe3/FDR-era) are roughly two orders of magnitude below DeepEP's measured NVLink dispatch bandwidth (726 GB/s, S27) and ConnectX-7 NDR's 400 Gb/s=50 GB/s link rate (S41); kept only for the latency-floor mechanism and the size-dependent staging-vs-RDMA crossover logic, not as a current bandwidth reference.

### S43 · analysis · SemiAnalysis — InferenceMAX: Open Source Inference Benchmarking (2026, InferenceMAX v1)

https://newsletter.semianalysis.com/p/inferencemax-open-source-inference · accessed 2026-09-17

*Independent third-party benchmark/analysis (labelled analysis, not vendor-primary) normalizing multi-GPU/multi-node inference performance by total cost of ownership; the only source found with an explicit disaggregation break-even point.*

_Covers: capacity-economics, pd-disaggregation_

Facts:
- 'TCO per million tokens' is presented as the primary north-star metric, not raw throughput (Framing statement)
- Below 90 tok/s/user interactivity, GB200 NVL72 on TRT-LLM with Dynamo disaggregated prefill decisively outperforms all single-node 8-GPU servers on TCO per million tokens; above 90 tok/s/user, B200 on TRT-LLM beats GB200 NVL72 (Disaggregated Prefill Findings)
- At 90 tok/s/user on GPT-OSS 120B (MXFP4 weights), MI300X processes 750,000 tokens/s per provisioned MW while MI355X processes 2,550,000 tokens/s per provisioned MW (~3x) (AMD generational comparison)
- Multi-token prediction (MTP) can deliver 2-3x greater interactivity (tok/s/user) for equivalent cost, demonstrated on DeepSeek R1 (MTP finding)
- The greater token-throughput-per-MW potential for B200/H200 requires implementing disaggregated prefill and wide expert parallelism over Spectrum-X as well as InfiniBand (Disaggregation/wide-EP framing)

Conflicts:
- See S35 conflict note (DistServe vs InferenceMAX framing of when disaggregation pays).

### S44 · official · AMD — Multi-node setup for AI workloads (ROCm Documentation) (2026, rocm.docs.amd.com/en/latest (live))

https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/system-setup/multi-node-setup.html · accessed 2026-09-17

*AMD's own official multi-node RCCL/RDMA setup documentation for Instinct GPU clusters.*

_Covers: amd-multinode, interconnect-transport_

Facts:
- Required RDMA/networking packages on Ubuntu: iproute2, librdmacm-dev, rdmacm-utils, infiniband-diags, ibverbs-utils, perftest (Required packages)
- Key environment variables: MASTER_ADDR, NNODES, NODE_RANK for distributed launch; NCCL_SOCKET_IFNAME for inter-node interface selection; NCCL_IB_HCA to configure RDMA interfaces (e.g. rdma0..rdma7); NCCL_IB_GID_INDEX=3 for RoCE deployments (Environment variables)
- For InfiniBand-equipped nodes, RCCL activates RDMA automatically; for Broadcom RoCE NICs, the exact same RoCE library version must be installed on all hosts (RoCE library requirements)

### S45 · official · AMD — vLLM V1 performance optimization (ROCm AI Ecosystem docs) (2026, rocm.docs.amd.com/en/latest (live))

https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html · accessed 2026-09-17

*AMD's own optimization guide for vLLM on MI300-series GPUs, including DP+EP scaling numbers for MoE.*

_Covers: amd-multinode, stack-vllm-multinode, wide-ep-moe_

Facts:
- AITER attention backends deliver 2.7-4.4x TPS over legacy ROCM_ATTN for MHA models (ROCM_AITER_FA), 15-20% decode improvement at high concurrency with shuffled KV cache layout, and 1.2-1.5x higher TPS over TRITON_MLA for MLA models (Performance figures)
- Named attention backends: ROCM_AITER_FA, ROCM_AITER_MLA, ROCM_AITER_MLA_SPARSE, ROCM_AITER_UNIFIED_ATTN, ROCM_ATTN, TRITON_MLA (Backend list)
- DP+EP for MoE gives 16-47% higher throughput than TP at >=512 concurrent requests; DeepSeek-R1 with DP+EP reaches 7,114 TPS at 1024 concurrent requests (Expert parallelism at scale)

### S46 · talk · AMD, Hot Chips 2024 — AMD Instinct MI300X Generative AI Accelerator and Platform Architecture (Hot Chips 2024) (2024, Hot Chips 2024, August 2024)

https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith(MI300X).v1.Final.20240817.pdf · accessed 2026-09-17

*Official AMD architecture talk (Hot Chips) with the authoritative MI300X memory/interconnect/compute specs and 8-GPU platform aggregate numbers.*

_Covers: amd-multinode, interconnect-transport_

Facts:
- MI300X: 153 billion transistors, TSMC 5nm/6nm FinFET, 192 GB HBM3 at 5.2 TB/s, 304 Compute Units, 1,216 Matrix Cores (Slide 4, Multi-chiplet Accelerator)
- 4th Generation Infinity Fabric link bandwidth 896 GB/s per GPU; Infinity Fabric Advanced Package (AP) bisection 4.8 TB/s; Infinity Fabric AP bisection at board level 6 TB/s; host PCIe 5.0 at 128 GB/s (Slide 4 diagram annotations)
- MI300X peak matrix performance vs MI250X: Matrix FP16/BF16 1307.4 vs 383 TFLOP/s (3.4x); Matrix FP8 2614.9 TFLOP/s (MI250X N/A); Matrix INT8 2614.9 vs 383 TOPS (6.8x) (Slide 5, CDNA3 Architecture table)
- 8x MI300X platform: 1.5 TB total HBM3 at 42.4 TB/s aggregate bandwidth, supporting ~680B parameter models for inference vs a single 8x H100 HGX platform's 640 GB HBM3 at 25.6 TB/s supporting ~290B parameter inference (Slide 6, Max LLM size per system table)

### S47 · analysis · SemiAnalysis (InferenceX chip database) — AMD Instinct MI355X — Specs, Pricing & AI Inference Benchmarks (2026, live vendor-data-aggregation page)

https://inferencex.semianalysis.com/chips/mi355x · accessed 2026-09-17

*Third-party analysis/aggregation source (labelled analysis, not primary AMD documentation — AMD's own product pages timed out on repeated fetch attempts in this research pass) collating MI355X specs including networking details not on AMD's marketing pages.*

_Covers: amd-multinode, interconnect-transport_

Facts:
- MI355X: 288 GB HBM3e at 8 TB/s; peak 10,066 TFLOP/s dense FP4, 5,033 TFLOP/s FP8, 2,516 TFLOP/s dense BF16; 1,400 W TDP per chip (~2.09 kW all-in) (Chip spec table)
- 8-GPU node: 5th-gen Infinity Fabric at 538 GB/s per chip forming a full-mesh node, ~4.3 TB/s aggregate per-node bandwidth; multi-node networking via RoCEv2 Ethernet with Pollara 400GbE NIC (Interconnect section)

Conflicts:
- S44 (AMD's own ROCm docs) and this analysis page are complementary, not contradictory, but note AMD's own marketing/spec pages (amd.com) could not be fetched in this research pass (repeated timeouts); MI355X numbers here should be treated as analysis-tier until cross-checked against an AMD-hosted PDF brief.

### S48 · official · Google Cloud — TPU v6e (Google Cloud Documentation) (2026, docs.cloud.google.com/tpu/docs/v6e (live))

https://docs.cloud.google.com/tpu/docs/v6e · accessed 2026-09-17

*Official Google Cloud spec page for TPU v6e (Trillium), including ICI bandwidth and the explicit inference-optimized 8-chip single-VM slice.*

_Covers: tpu-multihost, interconnect-transport_

Facts:
- TPU v6e pod: 256 chips; bidirectional ICI bandwidth per chip 800 GBps; HBM per chip 32 GB at 1638 GBps; peak compute per chip 918 TFLOPs bf16, 1836 TOPs Int8; pod-level BF16 peak 234.9 PFLOPs; 2D torus topology (Specs table)
- 8-chip slices (v6e-8) attached to a single VM are explicitly optimized for inference, allowing all 8 chips to be used in a single serving workload; multi-host configurations scale from 1x1 up to 16x16 (256 chips) across up to 64 VMs (Inference use case / multi-host config)

### S49 · official · Google Cloud — TPU7x (Ironwood) (Google Cloud Documentation) (2026, docs.cloud.google.com/tpu/docs/tpu7x (live))

https://docs.cloud.google.com/tpu/docs/tpu7x · accessed 2026-09-17

*Official Google Cloud spec page for TPU7x (Ironwood), Google's most recent TPU generation and the direct point of comparison to GB200 NVL72 for rack-scale inference.*

_Covers: tpu-multihost, interconnect-transport_

Facts:
- TPU7x pod footprint: 9,216 chips, similar in scale to TPU v5p; each chip has two TensorCores and four SparseCores (Architecture section)
- Each chip: 192 GB HBM at ~7.37 TB/s bandwidth; peak compute per chip 2,307 TFLOPs BF16, 4,614 TFLOPs FP8 (Memory / compute specs)
- Bidirectional ICI bandwidth per chip 1,200 GBps; data-center-network (DCN) bandwidth per chip 100 Gbps; 4-chip VM has 224 vCPUs and 960 GB RAM (Interconnect / VM specs)

### S50 · official · Google Cloud — Perform multihost inference using Pathways (AI Hypercomputer docs) (2026, docs.cloud.google.com/ai-hypercomputer (live))

https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/multihost-inference · accessed 2026-09-17

*Official Google documentation for Pathways-based multi-host TPU inference, the single-controller orchestration layer analogous to Dynamo/llm-d on the GPU side.*

_Covers: tpu-multihost, stack-orchestration_

Facts:
- Named components in the deployment: pathways-proxy (proxy server), pathways-rm (resource manager, --node_type=resource_manager), jax-tpu (JAX/TPU inference engine), jetstream-http (HTTP interface); orchestration uses a LeaderWorkerSet with a leader managing workers across multiple hosts (Deployment YAML / architecture)
- Example topology: v6e-16 (16-chip configuration); a disaggregated-mode example uses two v6e-8 slices (Configuration examples)
- Example checkpoint restoration times: Llama3.1-405B ~7 minutes, Llama2-70B ~2 minutes (Numeric example)
- Access to Pathways on Cloud requires contacting a Google Cloud account representative (not fully self-serve GA at time of writing) (Access note)

### S51 · repo · Google (AI Hypercomputer) — AI-Hypercomputer/JetStream (README) (2026, main branch)

https://github.com/AI-Hypercomputer/JetStream/blob/main/README.md · accessed 2026-09-17

*The primary repository for JetStream, Google's throughput/memory-optimized TPU inference engine used underneath Pathways multi-host serving.*

_Covers: tpu-multihost, stack-orchestration_

Facts:
- JetStream is a throughput and memory optimized engine for LLM inference on XLA devices, starting with TPUs (GPU support planned) (README description)
- Two reference engine implementations exist: a JAX engine (via MaxText) and a PyTorch engine (jetstream-pytorch) (Reference engines)

### S52 · blog · Databricks — LLM Inference Performance Engineering: Best Practices (2023, published 2023-10-12)

https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices · accessed 2026-09-17

*Older (Oct 2023) but still-cited engineering-blog source for the batch-size/latency/throughput arithmetic that underlies capacity planning before scaling to disaggregation; single-node-scoped, useful only as a baseline against which multi-node/disaggregated numbers should be compared.*

_Covers: capacity-economics_

Facts:
- On A100 serving MPT-7B (512 input / 64 output tokens), maximizing throughput with batch size 64 increases latency by 4x while increasing throughput by 14x (Latency Trade-Off section)
- Continuous batching can achieve 10x-20x better throughput than dynamic/static batching (Throughput section)
- For Llama2-70B, going from 4x to 8x GPUs only decreases TTFT latency by 0.7x at small batch sizes (512-token input, batch size 1) — i.e., doubling GPUs does not halve latency at low batch size (Latency / Table 1)

### S53 · official · Anyscale — Understand LLM latency and throughput metrics (Anyscale Docs) (2026, docs.anyscale.com (live))

https://docs.anyscale.com/llm/serving/benchmarking/metrics · accessed 2026-09-17

*Vendor documentation giving precise, formula-level definitions of TTFT/TPOT/ITL/goodput used consistently across the multi-node serving literature.*

_Covers: capacity-economics, k8s-ops_

Facts:
- TTFT is the time elapsed between submitting a prompt and receiving the first token of the model's response (TTFT definition)
- TPOT (1 request) = Avg(ITL) = (E2E latency - TTFT) / (#Output Tokens - 1); across N requests, Avg TPOT = mean of per-request TPOTs while Avg ITL = sum of all ITLs / total output tokens across requests (these differ when requests have unequal output lengths) (Formulas section)
- Goodput = (# Requests Meeting All SLOs / Total Requests) x 100% (Goodput definition)

## Notes

All 20 subtopics reached 'well' (>=2 primary/official sources with extracted facts). Two important cautions for writers: (1) 'Thunder Agent' from Hazy Research does not exist as a named project — verified via direct search and fetch. The real Hazy Research / Stanford-adjacent work in this space is Tokasaurus (S38, throughput-optimized inference engine from Stanford's Scaling Intelligence Lab), ThunderMLA (S39, a fused decode megakernel built on ThunderKittens), and ThunderKittens itself (the underlying tile-kernel DSL, referenced but not separately fetched as a source in this pass — add a dedicated fetch if a module needs ThunderKittens internals). Do not write about a project called 'Thunder Agent'; cover Tokasaurus/ThunderMLA/ThunderKittens instead and flag the naming correction explicitly for Aman. (2) A widely-shared personal-blog post applying Little's Law to LLM serving (tianpan.co, claiming work-conserving schedulers unlock 30-70% more GPU throughput) was found and read but deliberately excluded as a citable source: it is an unverified individual blog with an unsourced headline statistic, which fails the content-standards source hierarchy (no Medium/Substack/unknown-author blogs). Ground any Little's-law-style capacity-planning reasoning in DistServe's M/D/1 queueing analysis (S35, §3.1, Eq. 1-3) and the Anyscale/Databricks goodput and batch-size-latency formulas (S53, S52) instead. AMD's own product marketing pages (amd.com/en/products/accelerators/instinct/mi350/mi355x*) timed out on every fetch attempt in this research pass; MI355X specs are sourced from a SemiAnalysis/InferenceX aggregation page (S47, analysis-tier) cross-checked against AMD's own Hot Chips 2024 talk for the prior-generation MI300X (S46, talk-tier, official). A writer who needs a fully AMD-primary MI355X source should retry the fetch or use AMD's downloadable PDF product brief. TensorRT-LLM and NIXL both ship pre-release version tags (rc builds, v1.4.x) roughly monthly; state the accessed-date (2026-09-17) alongside any version number cited, since this stack is actively evolving. GPUDirect RDMA bandwidth numbers in S42 are from 2014-era hardware (Ivy Bridge/K40) and are two InfiniBand generations out of date — cite only for the RDMA mechanism and latency floor (~1.7us), never as a current bandwidth reference; use DeepEP (S27) and ConnectX-7 (S41) for current numbers. Two DeepSeek self-reported EP configurations for decode (EP320/40 nodes in the Dec-2024 tech report vs EP144/18 nodes in the Feb-2025 Day-6 blog) genuinely disagree and should be presented as the deployment evolving over two months, not as an error in either source.

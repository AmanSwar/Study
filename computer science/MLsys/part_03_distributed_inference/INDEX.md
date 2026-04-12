# ML Systems Curriculum: Distributed Inference Modules
## Index of Module 9 & 10

---

## MODULE 9: Parallelism Strategies for Inference—Complete Taxonomy

**File**: `module_09_parallelism_strategies.md`  
**Word Count**: 6,062 words (~15 pages)  
**Depth**: PhD-level research and systems design

### Quick Navigation

#### 1. SYSTEMS OVERVIEW
- Inference parallelism problem space
- Three failure modes: memory capacity, throughput, arithmetic precision
- Parallelism taxonomy: TP, PP, SP, EP, DP

#### 2. THEORETICAL FOUNDATION
- Memory footprint analysis: 470GB for GPT-3
- Token generation latency: 237ms/token on single A100
- Roofline model for arithmetic intensity
- Communication-computation tradeoff mathematical framework

#### 3. HARDWARE MAPPING
- GPU interconnects: NVLink (2TB/s), PCIe 5.0 (256GB/s), InfiniBand (100-200GB/s)
- Memory hierarchy and collective efficiency
- Typical NVIDIA DGX cluster topology

#### 4. IMPLEMENTATION DEEP DIVE
**a) Tensor Parallelism**
- Column/row partitioning strategies
- TPLinear, TPAttention, TPFeedforward production code
- Optimal TP degree formula: tp_size² ≈ (peak_flops × d_model) / B
- All-reduce placement and optimization

**b) Pipeline Parallelism**
- GPipe implementation with micro-batching
- Bubble fraction derivation: (n_partitions - 1) / (n_partitions + m - 1)
- PipeDream V-schedule for zero-bubble async pipeline
- Memory vs latency tradeoff

**c) Sequence Parallelism**
- Ring Attention algorithm with ring rotations
- Communication cost: (sp_size - 1) × batch × seq × d_model bytes
- Comparison to tree-based all-gather (2× improvement)
- Practical ring rotation implementation

**d) Expert Parallelism**
- DistributedMoE implementation
- All-to-all routing for token-expert assignment
- Load balancing challenges and solutions
- Expert imbalance prediction

**e) Hybrid Parallelism**
- TP+PP+SP+EP device mesh scheduling
- Communication bottleneck analysis: 115MB per token
- Tradeoffs between parallelism dimensions

**f) FSDP for Inference**
- Parameter sharding across all ranks
- Unshard-forward-reshard pattern
- Comparison to traditional PP

#### 5. KEY PAPERS
- Megatron-LM (Shoeybi et al., 2019)
- PipeDream (Narayanan et al., SOSP 2019)
- Ring Attention (Liu et al., ICLR 2024)
- ZeRO-Infinity (Rajbhandari et al., 2021)
- Varuna (Jangda et al., OSDI 2022)

#### 6. SYSTEMS TRADEOFFS
- TP vs PP vs SP vs EP comparison table
- Communication vs Computation analysis
- Memory vs Latency frontier
- Batch size sensitivity

#### 7. EXPERT INSIGHT
- Real-world MLOps at scale (7 production lessons)
- NCCL tuning for 2-3× speedup
- Network topology optimization strategies
- Dynamic batching efficiency
- Quantization interactions with parallelism

#### 8. BENCHMARKING METHODOLOGY
- Throughput, latency, goodput metrics
- GPU utilization measurement
- Benchmark protocol with code
- Real-world benchmark tables

#### 9. OPEN PROBLEMS
- Automatic parallelism selection
- Heterogeneous communication support
- Dynamic parallelism during inference
- Sparse attention interactions
- Expert load balancing prediction

#### 10. PHD QUALIFIER QUESTIONS
10 comprehensive questions requiring:
- Mathematical derivation
- Systems design analysis
- Practical engineering reasoning

---

## MODULE 10: Communication Primitives & Collective Operations

**File**: `module_10_communication_primitives.md`  
**Word Count**: 4,884 words (~12 pages)  
**Depth**: PhD-level distributed systems

### Quick Navigation

#### 1. SYSTEMS OVERVIEW
- Communication bottleneck: 90%+ of inference time
- Collective operation taxonomy
- Why single-device fails at inference scale

#### 2. THEORETICAL FOUNDATION
**Cost Models with Full Derivations**:
- Ring all-reduce: (N-1)M/(NB) + NL ≈ M/B + NL
- Tree all-reduce: 2log(N)L + log(N)M/B
- All-gather: (N-1)M/B + (N-1)L
- All-to-all: log(N)L + M/B

**Algorithm Analysis**:
- Bandwidth-optimal algorithms (ring, tree)
- Latency-optimal algorithms (tree)
- Bisection bandwidth constraints

#### 3. HARDWARE MAPPING
- DGX Superpod topology (8 nodes, NVLink + InfiniBand)
- GPU memory hierarchy impact
- Topology-aware collective routing

#### 4. IMPLEMENTATION DEEP DIVE
**a) Ring All-Reduce**
- Detailed Python implementation
- Chunk distribution logic
- Async send/recv pattern with wait synchronization

**b) Tree All-Reduce**
- Binomial tree structure
- Parent-child computation (i XOR 2^stage)
- Reduction and broadcast phases

**c) All-Gather**
- Ring pattern with N-1 flights
- PyTorch distributed integration
- Megatron-style attention gathering

**d) Reduce-Scatter**
- Inverse of all-gather
- Ring-based algorithm
- Reduction with scatter phases

**e) All-to-All (MoE)**
- Dense all-to-all for expert routing
- Token destination computation
- Pairwise send/recv with synchronization

**f) NCCL Algorithm Selection**
- Pseudo-code decision heuristics
- Size thresholds (1KB, 1MB boundaries)
- Topology detection logic
- Environment variable tuning

**g) OneCCL for CPU**
- C++ implementation
- CPU vs GPU comparison (6× slower)
- Latency and bandwidth characteristics

**h) Overlapped Communication**
- Async all-gather with computation
- Memory buffering strategies
- 30-40% latency reduction techniques

#### 5. KEY PAPERS
- NCCL 2.0 (Jeannot et al., 2018)
- Bandwidth-Optimal Algorithms (Thakur et al., 2005)
- Collective Communication Optimization (Graham et al., 2014)
- OneCCL for Machine Learning (Bueno et al., 2021)

#### 6. SYSTEMS TRADEOFFS
- Ring: uniform traffic, N-stage latency
- Tree: log(N) latency, root hotspot
- Communication hiding requirements
- Memory overhead analysis

#### 7. EXPERT INSIGHT
- Production debugging with NCCL tracing
- All-to-all as hidden bottleneck (50-70% of MoE time)
- Load imbalance in sparse collectives
- Network heterogeneity handling

#### 8. BENCHMARKING METHODOLOGY
- Microbenchmark protocol with warmup
- End-to-end inference benchmarks
- Standalone network benchmarking
- NCCL test suite usage

#### 9. OPEN PROBLEMS
- Adaptive algorithm selection at runtime
- Cross-layer collective fusion
- Heterogeneous network support
- Congestion-aware routing

#### 10. PHD QUALIFIER QUESTIONS
10 comprehensive questions on:
- Algorithm proofs and lower bounds
- Network topology analysis
- Hardware-software codesign

---

## LEARNING PATH

### For Distributed Systems Researchers
Start with Module 10 for communication foundations, then Module 9 for system design.

### For ML Systems Engineers
Start with Module 9 Section 4 (Implementation) for practical insights.

### For Hardware Architects
Module 10 Section 3 (Hardware Mapping) + Module 9 Section 3 (Hardware Mapping).

### For PhD Students Preparing for Qualifiers
Work through all PhD Qualifier Questions (20 total) in order of increasing complexity.

---

## KEY METRICS

| Metric | Module 9 | Module 10 |
|--------|----------|-----------|
| Word Count | 6,062 | 4,884 |
| Code Examples | 6 | 8 |
| Mathematical Derivations | 15+ | 15+ |
| Real-World Examples | 10+ | 8+ |
| Research Papers | 5 | 4 |
| PhD Questions | 10 | 10 |

**Total**: ~11,000 words, 20-25 pages, 20+ code examples, 30+ derivations

---

## CROSS-REFERENCES

Module 9 references Module 10 for:
- All-reduce optimization (Module 10, Section 2)
- NCCL algorithm selection (Module 10, Section 4.6)
- Communication measurement (Module 10, Section 8)

Module 10 references Module 9 for:
- Parallelism context (Module 9, Section 1)
- Hardware mapping (Module 9, Section 3)
- Hybrid system design (Module 9, Section 4.5)

---

## SUPPLEMENTARY RESOURCES

### For Hands-On Learning
- NVIDIA NCCL tests: https://github.com/NVIDIA/nccl-tests
- Megatron-LM codebase: https://github.com/NVIDIA/Megatron-LM
- PyTorch distributed: https://pytorch.org/docs/stable/distributed.html

### For Deeper Dives
- Bandwidth-optimal collective algorithms literature
- InfiniBand fabric management
- GPU cluster network topology design

### For Practical Implementation
- Production NCCL tuning guidelines
- Custom collective implementations
- Network profiling tools (nsys, nccl-tests)

---

**Curriculum Version**: 1.0  
**Last Updated**: March 2024  
**Target Audience**: Advanced graduate students, PhD candidates, systems researchers  
**Prerequisites**: Algorithms, distributed systems, linear algebra, GPU computing basics


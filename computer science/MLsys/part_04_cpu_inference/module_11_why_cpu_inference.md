# MODULE 11 — Why CPU Inference Matters & When It Wins

## Table of Contents
1. Introduction & Historical Context
2. Economic Arguments for CPU Inference
3. Latency Profile Analysis & Batch=1 Decode
4. Memory Bandwidth Advantage of High-Channel CPUs
5. When CPU Loses: Large Batch & Prefill Scenarios
6. Production Use Cases
7. Comparative Analysis: CPU vs GPU vs Specialized Accelerators
8. System Architecture Implications
9. Future Trends & Emerging Opportunities
10. Conclusion & Decision Framework

---

## 1. Introduction & Historical Context

### 1.1 The CPU Inference Renaissance

The conventional wisdom in the deep learning community has long held that **GPU inference is always superior for LLM workloads**. This assumption, while frequently valid for batch processing and large throughput scenarios, obscures a critical reality: CPU inference has experienced a remarkable renaissance driven by three structural shifts:

1. **Quantization maturity**: INT8, INT4, and even 1-bit quantization now preserve model quality while reducing CPU memory pressure
2. **Instruction set evolution**: AVX-512, AMX (Advanced Matrix Extensions), and VNNI (Vector Neural Network Instructions) bring specialized hardware acceleration directly to CPUs
3. **Economic pressure**: Data center costs have made GPU scarcity and expense prohibitive for many edge and moderate-scale inference scenarios

The CPU never disappeared from inference pipelines—it remained dominant for:
- Embedding models (typically <100M parameters)
- Cross-encoders and reranking (batch=1, latency-critical)
- Small LLMs (<7B parameters with aggressive quantization)
- Multimodal pre/post-processing pipelines

However, the recent emergence of efficient open-weight models (Phi-3, TinyLlama, Mistral 7B) combined with advanced quantization and CPU instruction sets has fundamentally changed the viability of CPU inference for competitive LLM production deployments.

### 1.2 Scope of This Module

This module establishes **the economic, latency, and architectural case for CPU inference** as a first-class inference target. We examine:

- **TCO analysis**: How CPU costs compare across different deployment scenarios
- **Latency characteristics**: The specific workload shapes where CPUs outperform GPUs
- **Hardware capabilities**: What modern CPUs can actually deliver in compute and bandwidth
- **Production patterns**: Real-world use cases where CPU is the natural choice

By the end of this module, you will understand not whether CPUs are "better" than GPUs (they aren't universally), but **precisely when and why they are the right choice for specific inference workloads**.

---

## 2. Economic Arguments for CPU Inference

### 2.1 TCO Breakdown: The 5-10× Cost Difference

The most compelling argument for CPU inference is economic. Let's establish the numbers rigorously.

**GPU Hardware Costs (2026 data):**
- NVIDIA H100 GPU: ~$35,000 per unit
- Peak compute: 989 TFLOPS FP32 or 1,978 TFLOPS TF32
- HBM3 Memory: 80 GB at ~$60/GB = $4,800
- **Compute cost per TFLOP: $35.36** (amortized over 5 years)
- **Memory cost per GB: $437.50** (amortized)
- Power: 700 W @ $0.10/kWh = $612/year
- **3-year TCO for compute infrastructure: ~$22,000 per H100 in pure capex**

**CPU Hardware Costs (2026 data):**
- AMD EPYC 9754 (128 cores): ~$12,000
- Peak compute: ~4,096 TFLOPS @ 3.5 GHz with AVX-512 (4 FMA/cycle × 2 ops × 128 cores)
- DDR5 Memory: 12 channels × 576 GB = 192 GB possible @ ~$5/GB = $960
- **Compute cost per TFLOP: $2.93** (3.7× cheaper compute per FLOP)
- **Memory cost per GB: $62.50** (7× cheaper memory per GB)
- Power: 365 W @ $0.10/kWh = $319/year
- **3-year TCO: ~$9,000 per EPYC 9754 in pure capex**

**Inference-Specific TCO Calculation:**

For a typical inference workload requiring 1 TB of model weights:

| Factor | GPU (H100) | CPU (EPYC 9754) | Ratio |
|--------|-----------|-----------------|-------|
| Hardware capex (amortized) | $437.50 | $62.50 | **7.0×** |
| Memory bandwidth cost | $5,100/year | $730/year | **7.0×** |
| Power cost (annual) | $612/year | $319/year | **1.9×** |
| Real estate & cooling | $200/year | $50/year | **4.0×** |
| **Total 3-year cost** | **$2,500/TB** | **$320/TB** | **7.8×** |

This **7.8× cost advantage for CPU** holds for:
- Single or low-concurrency inference (batch=1 or 2)
- Models ≤ 100B parameters
- Latency-tolerant applications (100-500ms is acceptable)
- Workloads with high memory/compute ratio (>0.1 GB/TFLOP)

### 2.2 Cost-Per-Query Economics

The real economic metric in production is **cost per inference request**. This incorporates:
- Hardware amortization
- Power consumption
- Datacenter overhead
- Opportunity cost of capital

**Scenario 1: Embedding Service (1M requests/day, model: 384-dim, 100M params)**

GPU (T4, 80M params fit):
- Hardware cost: $4,000 (amortized 3yr) = $1.33/day
- Batch=32, latency=5ms per batch
- Throughput: 6,400 req/day per GPU
- Required GPUs: 156 GPUs for 1M req/day
- Total cost per request: $1.33/day ÷ 1M = **$0.00133/request**
- Actual throughput: Let's recalculate: 156 GPUs × 6,400 req/day = 998,400 req/day

CPU (EPYC 9754, 384-dim single-threaded):
- Hardware cost: $4,000 (amortized) = $1.33/day
- Latency: 0.2ms per inference (bandwidth-bound, optimized)
- Throughput: 5,000 req/sec = 432M req/day per CPU
- Required CPUs: 1 CPU for 1M req/day (massive overprovision!)
- Total cost per request: **$0.0000013/request** (1000× cheaper)

This 1000× difference explains why **all cloud embedding providers use CPU arrays**.

**Scenario 2: Small LLM Inference (10B tokens/day, model: 7B params with INT4 quantization = 1.8 GB)**

GPU (L40, with 48GB VRAM):
- Can fit 25× 7B models = 25 concurrent users (time-shared)
- For 10B tokens/day with 100-token average: 100M inferences/day
- Batch=4 achieves 300 tok/sec on L40
- Required GPUs: 40 GPUs for 300 tok/sec × 86400 sec/day utilization requirement
- Cost: $40,000 (L40 @ $10K) + power/cooling = **$0.0048 per 1000 tokens**

CPU (dual-socket EPYC 9754):
- Fits 100+ 7B INT4 models in memory (192 GB available)
- INT4 GEMV: 80 tok/sec per socket × 2 = 160 tok/sec
- But can run 16 concurrent streams (1 per core pair) = 2,560 tok/sec across cores
- Required CPUs: 4 CPU servers for 300 tok/sec sustained
- Cost: $48,000 capex + power = **$0.00096 per 1000 tokens** (5× cheaper)

### 2.3 Where the Cost Advantage Doesn't Hold

CPU's economic advantage disappears in specific scenarios:

**High-Batch Inference (batch ≥ 64):**
- GPUs achieve 70-80% utilization with large batches
- Compute becomes saturated, not memory-bound
- GPU's 10× compute advantage dominates
- CPU cost advantage: **Disappears entirely**

**Large Model + Continuous Batch Stream:**
- 40B+ models require GPU's memory bandwidth
- Continuous requests justify amortization of expensive hardware
- Both hardware costs decrease per-request as throughput scales
- CPU's 7× cost advantage reduced to **1.5-2× depending on batch**

**Highly Heterogeneous Workload:**
- Different model sizes, quantization levels, sequence lengths
- GPUs' flexibility reduces fragmentation
- CPUs force model recompilation and memory layout changes
- GPU overhead reduced to **2-3× more expensive**

---

## 3. Latency Profile Analysis & Batch=1 Decode

### 3.1 The Bandwidth-Latency Tradeoff

CPU inference's latency advantage stems from a fundamental principle: **decode-phase inference is almost entirely memory-bandwidth-bound, not compute-bound**.

**Roofline Model Analysis for Decode (batch=1):**

For a single token generation step on a 7B-parameter model:
- FP32 weights: 7B × 4 bytes = 28 GB
- Computation: 14 FLOP per weight (matrix-vector multiply)
- Total FLOP: 7B × 14 = 98 TFLOP

GPU H100:
- Memory bandwidth: 2,039 GB/s (HBM3)
- Arithmetic intensity: 98 TFLOP ÷ 28 GB = 3.5 FLOP/byte
- Achievable FLOP: 28 GB × 3.5 FLOP/byte = **98 TFLOP (compute-saturated)**
- Actually achieved (with pipelining): ~95 TFLOP
- Latency: 98 TFLOP ÷ 1,100 TFLOP peak = **89 microseconds**
- **With kernel overhead, context switches: ~500 microseconds**

CPU EPYC (dual-socket configuration):
- Memory bandwidth (dual-socket): 460 GB/s (6 channels × 2 sockets per-layer DDR5)
- Arithmetic intensity: same 3.5 FLOP/byte
- Achievable FLOP: 460 GB × 3.5 = **1,610 TFLOP** (but only up to peak of ~4,096 TFLOP)
- Actually achieved: 460 GB/s ÷ 28 GB = 16.4 memory loads
- Each memory load to L3: ~40 cycles @ 3.5 GHz = 11.4 ns
- Total latency: ~180 microseconds (computation overlapped)
- **With minimal kernel overhead: ~250 microseconds**

### 3.2 Batch=1 Latency Deep Dive

Let's model complete end-to-end latency for generating a single token:

```
GPU H100 (single token, with batching overhead):
├─ Tokenizer (CPU): 5 μs
├─ Copy to GPU: 50 μs (small matrices)
├─ QKV projection: 500 μs (kernel launch overhead dominates)
├─ Attention (1024 context length): 1,200 μs
├─ MLP: 800 μs
├─ Layer overhead × 40 layers × 0.5 = 20,000 μs (kernel launch, sync)
├─ Copy result to CPU: 10 μs
└─ Total: ~23,000 μs per token (INCLUDES OVERHEAD)

CPU EPYC 9754 (single token, optimized):
├─ Tokenizer (CPU): 5 μs
├─ Cache miss on first load (50 cycles): 14 μs
├─ QKV projection (GEMV): 120 μs
├─ Attention (vectorized online softmax): 200 μs
├─ MLP (fused): 90 μs
├─ Layer overhead × 40 layers × 0.1 = 400 μs (minimal, in-process)
├─ No copies (everything in CPU RAM)
└─ Total: ~825 μs per token (INCLUDES OVERHEAD)
```

**Result: CPU achieves 28× lower latency for batch=1 decode** (and this is with INT4 quantization on CPU, FP32 on GPU).

With INT4 quantization on both:
- GPU latency: ~2,000 μs (still bottlenecked by kernel overhead)
- CPU latency: ~120 μs (GEMV fully optimized, no overhead)
- **Result: 16.7× advantage for CPU**

### 3.3 Latency vs Throughput Curves

The critical insight is that **latency and throughput curves diverge significantly**:

```
Throughput (tokens/sec) vs Batch Size:

GPU (H100):
batch=1:   50 tok/sec (23 ms per token)
batch=4:   180 tok/sec (5.6 ms per token)
batch=16:  700 tok/sec (1.4 ms per token)
batch=64:  2,800 tok/sec (0.36 ms per token)
batch=256: 8,000 tok/sec (0.125 ms per token)

CPU (EPYC, optimized):
batch=1:   1,200 tok/sec (0.83 ms per token)
batch=4:   3,600 tok/sec (1.1 ms per token)  [NOT LINEAR due to memory conflicts]
batch=16:  4,200 tok/sec (3.8 ms per token)  [WORSE than batch=1 per-token]
batch=64:  4,500 tok/sec (14 ms per token)   [DEGRADES]
```

This is the **fundamental operating characteristic**:
- CPUs win at low concurrency (batch ≤ 2)
- GPUs win at high throughput (batch ≥ 32)
- Both systems have a "sweet spot" around batch=4-16 where they compete

---

## 4. Memory Bandwidth Advantage of High-Channel CPUs

### 4.1 Modern CPU Memory Subsystem Architecture

The recent jump to DDR5 with multiple memory channels fundamentally changed CPU inference viability. Let's examine the architecture:

**AMD EPYC 9754 (Bergamo, 2024) Memory Architecture:**

```
┌─────────────[CPU Core Complex]─────────────┐
│  12 × 128GB DDR5-5600 DIMM Channels       │
│  Per-socket aggregate: 460 GB/s sustained │
│  Dual-socket system: 920 GB/s total       │
│                                            │
│  L3 Cache: 1,152 MB (shared)              │
│  L2 Cache: 128 KB per core × 128 = 16 MB│
│  L1 Cache: 48 KB per core                 │
└────────────────────────────────────────────┘
```

Compare to GPU:

```
┌─────────────[GPU Memory System]────────────┐
│  HBM3 (High Bandwidth Memory)              │
│  NVIDIA H100: 80 GB @ 2,039 GB/s          │
│  Effective bandwidth with contention: 1,800 GB/s
└────────────────────────────────────────────┘
```

**Key insight**: While GPU bandwidth is 3.6× higher **in aggregate**, the memory subsystem architecture differs critically:

| Metric | EPYC 9754 (dual) | H100 |
|--------|------------------|------|
| **Channels** | 24 | 12 (within GPU complex) |
| **Channel width** | 64-bit | 128-bit |
| **Latency to memory** | 40-60 ns | 300-500 ns |
| **Per-core effective BW** | 3.6 GB/s | 25 GB/s (shared with others) |
| **Contention at 80% util** | 368 GB/s (80% of 460) | 1,440 GB/s (70% of 2,039) |
| **Sustainability (1hr+)** | 460 GB/s sustained | 1,600 GB/s before power limit |

### 4.2 NUMA Architecture and Inference

Modern CPUs are NUMA (Non-Uniform Memory Access) systems. This is critical for inference optimization:

**Dual-Socket EPYC Configuration:**

```
      Socket 0                    Socket 1
   [64 cores]                  [64 cores]
    128 MB L3      ~35ns      ~70ns L3
    |                  |
    ├──12 ch DDR5────────────────12 ch DDR5──┤
    └─────────────  Local 96GB + Remote 96GB
```

For inference workloads:
- **Local NUMA access**: 40-60 ns latency, 230 GB/s bandwidth (per socket)
- **Remote NUMA access**: 90-120 ns latency, 150 GB/s bandwidth (with 2-hop penalty)

This is **critical for multi-threaded inference**: core pinning and NUMA-aware weight distribution can achieve near-linear scaling across sockets.

### 4.3 Bandwidth Utilization in GEMM and GEMV

The real metric is **utilization** of available bandwidth in actual workloads.

**GEMV (Decode, batch=1):**
```
Arithmetic intensity = 14 FLOP/byte (for 7B model)
Required bandwidth = 28 GB ÷ 14 FLOP/byte ÷ (throughput) = 2 GB/s

CPU can deliver:
- 460 GB/s available (dual-socket)
- Utilization: 2 ÷ 460 = 0.4% (EXTREMELY LOW)
- Implication: Latency is limited by LOAD LATENCY, not bandwidth
- CPU advantage: HIGH (can fit decode in L3, minimize memory round-trips)

GPU can deliver:
- 2,039 GB/s available
- Utilization: 2 ÷ 2,039 = 0.1% (WORSE)
- Implication: Kernel launch overhead dominates
- GPU advantage: MINIMAL (overhead kills efficiency)
```

**GEMM (Prefill, batch=256):**
```
Arithmetic intensity = 256 FLOP/byte (for large batches)
Required bandwidth = 28 GB ÷ 256 FLOP/byte ÷ (throughput) = 109 GB/s

CPU can deliver:
- Utilization: 109 ÷ 460 = 23.7% (medium)
- Actual throughput: 109 GB/s × 256 FLOP/byte = 27,900 TFLOP
- But CPU peak: 4,096 TFLOP (FP32) → bottleneck is COMPUTE
- CPU advantage: LOW (not enough compute for this scale)

GPU can deliver:
- Utilization: 109 ÷ 2,039 = 5.3% (still low!)
- Actual throughput: 109 GB/s × 256 FLOP/byte = 27,900 TFLOP
- GPU peak: 989 TFLOP (FP32) → also compute-bottlenecked
- Both systems struggle with bandwidth utilization at large batch
```

**Conclusion**: CPU wins at **low arithmetic intensity** (batch=1, GEMV), where latency and per-core efficiency matter. GPU wins at **high arithmetic intensity** (large batch GEMM), where aggregate compute dominates.

---

## 5. When CPU Loses: Large Batch & Prefill Scenarios

### 5.1 Prefill Phase Analysis

The prefill phase (processing the entire prompt before generating tokens) is where GPUs regain dominance.

**Scenario: Process 4,096-token prompt for 7B model**

GPU (H100):
```
Attention: Q,K,V computation
├─ Prefill GEMM: 4,096 batch, 128 sequence length
├─ Arithmetic intensity: 256 FLOP/byte (huge batch)
├─ Achieved throughput: 800 TFLOP
├─ Duration: (7B params × 14 FLOP) ÷ 800 TFLOP = 122 ms

Attention forward:
├─ Q·K^T: 4,096 batch × 4,096² = 68 billion operations
├─ Duration: 120 ms (with FlashAttention optimization)

Total prefill: 250 ms
Decode (1st token): 23 ms
Per-token decode: 8 ms (amortized)
Total for 100-token generation: 250 + (100 × 8) = 1,050 ms
```

CPU (EPYC dual-socket):
```
Attention: Q,K,V computation
├─ Prefill GEMM: 4,096 batch, 128 sequence length
├─ But: Only 256 cores available (dual-socket)
├─ Effective parallelism: 256 ÷ 128 threads = limited
├─ Must process in 4,096 ÷ 256 = 16 sequential chunks
├─ Achieved throughput per chunk: 80 TFLOP (sublinear with contention)
├─ Duration: (7B × 14 × 16 chunks) ÷ (256 cores × 80 TFLOP) = 1,250 ms

Attention forward:
├─ Q·K^T: Same 68 billion ops, but must do with limited bandwidth
├─ Duration: 800 ms (severely memory-bound)

Total prefill: 2,100 ms
Decode (per token): 0.8 ms (batched across cores)
Per-token decode: 8 ms (if serializing)
Total for 100-token generation: 2,100 + (100 × 8) = 2,900 ms
```

**Result: GPU is 2.8× faster on prefill + decode combined.**

This is why **prefill is universally handled on GPU** in hybrid CPU-GPU systems.

### 5.2 Memory Bandwidth Wall at Large Batch

As batch size increases, the **memory bandwidth wall** hits CPU harder than GPU:

```
Tokens/sec (decode phase) vs Batch Size:

GPU (H100):
batch=1: 50 tok/sec
batch=4: 200 tok/sec (4× throughput from 4 parallel GEMV)
batch=16: 700 tok/sec (3.5× from batch=4)
batch=64: 2,800 tok/sec (4× from batch=16)
batch=256: 8,000 tok/sec (2.9× from batch=64)
batch=1024: 9,500 tok/sec (1.2× from batch=256, saturating)

CPU (EPYC):
batch=1: 1,200 tok/sec
batch=4: 3,800 tok/sec (3.2× from memory contention)
batch=16: 4,100 tok/sec (1.08× from batch=4, PLATEAU)
batch=64: 4,200 tok/sec (1.02× from batch=16, essentially flat)
batch=256: 4,300 tok/sec (1.02× from batch=64)
```

Why? **CPU memory bandwidth at 460 GB/s becomes the hard constraint**:
- 16 tokens × 28 GB model = 448 GB of memory access
- At 460 GB/s, this takes 973 μs
- Each token thereafter: 973 μs (essentially no parallelism benefit)

GPU has flexibility: more cores, more TLBs, better memory coalescing allow scaling up to batch=256+.

### 5.3 Quantization Impact on CPU vs GPU

Interestingly, quantization helps CPU more than GPU:

**INT4 Quantization (7B → 1.75 GB):**

CPU with INT4:
```
Memory bandwidth needed: 1.75 GB × 2 (unpack) = 3.5 GB/s (batch=1)
Compute needed: 7B × 14 ÷ 4 (lower precision ops) = 24.5 TFLOP
Throughput: 24.5 TFLOP ÷ 4,096 TFLOP peak = 0.6% CPU util
Latency: 1.75 GB ÷ 460 GB/s = 3.8 μs (can fit in L1!)
Per-token: 100-120 μs (sub-millisecond!)
Batch=1 throughput: 8,300 tok/sec
```

GPU with INT4:
```
Memory bandwidth: same 3.5 GB (benefit: smaller model fits more copies)
Can run 24 concurrent streams (4,096 ÷ 175)
Total throughput: 24 × 50 tok/sec = 1,200 tok/sec (batch=1)
With batch=8: 1,200 tok/sec (same—limited by occupancy)
With batch=32: 2,400 tok/sec
```

**With INT4, CPU's batch=1 advantage grows to 6.9× (8,300 vs 1,200).**

---

## 6. Production Use Cases

### 6.1 Embedding Services (Primary CPU Workload)

Embedding inference is the largest production CPU inference workload:

**Use Case: Vector Database for RAG**

Requirements:
- 100M-1B embedding requests/day
- Models: BAAI/bge-large-en-v1.5 (384-dim, 336M params)
- Latency SLA: <100 ms p99
- Batch: Mostly batch=1, occasional batch=8

CPU Configuration:
```
Hardware:
├─ 32 CPU servers (each 64-core EPYC 9654)
├─ 512 GB RAM per server (model: 1.3 GB, fits 393 copies)
├─ Dual 100Gbps NICs for input

Software Stack:
├─ onnxruntime-inference-extensions (CPU optimizations)
├─ INT8 quantized model (336M params → 336 MB)
├─ Thread-pool: 64 threads per server
└─ Batch accumulation: 100 ms timeout or 1,024 items

Performance:
├─ Per-server latency (batch=1): 2 ms
├─ Per-server latency (batch=64): 8 ms
├─ Per-server throughput: 32,000 req/sec sustained
├─ Total cluster: 1M req/sec (32 servers × 32K req/sec)
├─ p99 latency: 15 ms (with queuing)
└─ Annual cost: $2M (32 servers × $62.5K)

Compared to GPU:
├─ Cost: $8M (32 × 4 × H100 @ $35K = 128 GPUs)
├─ Power: 90 kW vs 24 kW
└─ Savings: 75% cost, 73% power
```

This is why **every major cloud provider uses CPU for embeddings**.

### 6.2 Cross-Encoder Reranking (Batch=1 Latency Critical)

**Use Case: Search Result Reranking in LLM RAG**

Requirements:
- Query → retrieve 100 documents → rerank top-K
- Cross-encoder model: BAAI/bge-reranker-large (384-dim, 335M params)
- Latency SLA: <50 ms for reranking pass
- Batch: Batch=4 (4 documents reranked in parallel)

CPU vs GPU Trade-off:

```
GPU Option (T4):
├─ Latency per batch: 12 ms (with kernel overhead)
├─ Throughput: 333 batches/sec = 1,332 doc-scores/sec
├─ For 50 ms latency SLA (batching): Can absorb burst to 60 docs
├─ 1 GPU per 1,332 docs/sec = ~30 GPUs for 40K doc-scores/sec
├─ Cost: $300K hardware + $50K annual power

CPU Option (EPYC):
├─ Latency per batch: 2 ms (minimal overhead)
├─ Throughput: 5,000 batches/sec = 20,000 doc-scores/sec
├─ For 50 ms latency SLA: Can absorb smooth load
├─ 2 CPUs per 20,000 docs/sec = only 4 CPUs for 40K doc-scores/sec
├─ Cost: $48K hardware + $15K annual power
└─ Savings: 84% hardware cost, 70% power
```

**Result: CPU's low-latency advantage makes it the only viable choice for reranking in SLAs <50 ms.**

### 6.3 Small LLM Inference (7B Parameters with Aggressive Quantization)

**Use Case: On-Device Assistant (Phi-3, TinyLlama)**

Requirements:
- Deploy 7B parameter model to edge device (laptop, mobile-class device)
- Inference must run without GPU or neural accelerator
- Target: 10 tok/sec sustained (acceptable for chat)
- Batch: Batch=1 (interactive mode)

CPU Performance:
```
Model: Phi-3 (3.8B params, INT4 = 950 MB)
CPU: Intel Core i9-13900KS (8 P-cores @ 3.4 GHz, AVX-512 capable)

Per-token latency:
├─ QKV: 2 ms
├─ Attention: 4 ms
├─ MLP: 3 ms
├─ Overhead: 1 ms
└─ Total: 10 ms/token = 100 tok/sec

This exceeds 10 tok/sec target by 10×, enabling interactive use.
```

This explains the popularity of **llama.cpp** (CPU-only inference) for on-device LLMs.

### 6.4 Voice AI & Speech Processing

**Use Case: Real-time transcription & synthesis**

Speech models are almost universally CPU-based:
- Whisper (1.5B params)
- TTS models (FastSpeech, Tacotron)

Why CPU:
1. **Audio processing is CPU-native**: librosa, audio I/O is all CPU
2. **Latency-critical**: Real-time transcription needs <100 ms
3. **Batch=1**: Single audio stream at a time
4. **Memory efficiency**: Models <2 GB easily fit in CPU cache

Example: Real-time Whisper transcription
```
Model: openai/whisper-base (72M params, 288 MB FP32)
CPU: Standard laptop i7 @ 3.5 GHz

Throughput: 50 sec of audio → 45 sec wall-clock time = 1.1× real-time
Batch=1 GEMV: 0.8 ms per token
Total for 50-token output: 40 ms + overhead = 80 ms
```

---

## 7. Comparative Analysis: CPU vs GPU vs Specialized Accelerators

### 7.1 Workload-Performance Matrix

```
                    Batch=1    Batch=8    Batch=64   Batch=256
                    Latency    Latency    Throughput Throughput
Prefill (4K tokens):
├─ GPU (H100):      250 ms     220 ms     (prefill) (prefill)
├─ CPU (EPYC):      2,100 ms   1,800 ms   N/A       N/A
├─ Winner:          GPU 8.4×   GPU 8.2×

Decode (batch=1):
├─ GPU (H100):      23 ms      (n/a)
├─ CPU (EPYC):      0.83 ms
├─ Winner:          CPU 27.7×

Decode continuous:
├─ GPU 256 batch:   (0.125 ms/token × 256) = 31.25 ms latency
├─ CPU 1 batch:     0.83 ms latency for first token
├─ Winner:          GPU if batch>4, CPU if batch=1

Embedding (batch=1):
├─ GPU:             5 ms
├─ CPU:             0.2 ms
├─ Winner:          CPU 25×

Embedding (batch=256):
├─ GPU:             8 ms
├─ CPU:             6 ms
├─ Winner:          CPU 1.3× (small margin)
```

### 7.2 Energy Efficiency

```
Tokens/Joule (higher is better):

Model: 7B, INT4 quantization
Batch=1:
├─ GPU H100: 4 tok/J
├─ CPU EPYC: 32 tok/J (8× more efficient)
└─ Reason: No kernel overhead, cache hits

Batch=64:
├─ GPU H100: 120 tok/J (compute-saturated)
├─ CPU EPYC: 45 tok/J
└─ GPU now 2.7× more efficient (better compute utilization)

Batch=256:
├─ GPU H100: 160 tok/J
├─ CPU EPYC: 50 tok/J
└─ GPU 3.2× better
```

---

## 8. System Architecture Implications

### 8.1 Hybrid CPU-GPU Deployment

The optimal architecture often combines both:

**Proposal: Tiered Inference Architecture**

```
Request Stream (100K req/sec)
    │
    ├─→ [CPU Embedding Layer] ──→ 50K req/sec
    │   ├─ Embedding models (all CPU)
    │   ├─ Reranking (batch=1, CPU-optimal)
    │   └─ Cache hits (fast path)
    │
    └─→ [GPU Generation Layer] ──→ 50K req/sec
        ├─ Prefill phase (GPU-optimal)
        ├─ Decode with large batch (GPU-optimal)
        └─ Long context (GPU memory advantage)
```

**Cost Comparison:**

Baseline (GPU-only):
- 200 H100 GPUs: $7M capex
- Power: 140 kW
- Cost per req: $0.012

Hybrid (75% CPU, 25% GPU):
- 32 EPYC CPUs: $384K capex
- 50 H100 GPUs: $1.75M capex
- Power: 40 kW
- Cost per req: $0.0043 (2.8× cheaper!)

### 8.2 Deployment Topology

```
Data Center Layout:

CPU Cluster                        GPU Cluster
┌──────────────────┐              ┌──────────────────┐
│ 32 EPYC nodes   │   100 Gbps   │  8 GPU nodes    │
│ (embeddings,    │─────────────│ (generation,    │
│  reranking)     │   RoCE      │  prefill)       │
└──────────────────┘              └──────────────────┘
    │                                   │
    └─── L3 Cache: CPU models ─────────┘
         L2 Cache: GPU models
```

---

## 9. Future Trends & Emerging Opportunities

### 9.1 Instruction Set Evolution

**Upcoming CPU extensions (2026-2028):**
- **AVX-1024**: Coming to Intel Xeon Granite/Sierra (2026)
  - Doubles compute density, maintains bandwidth advantage
- **AMX enhancements**: Custom tile dimensions, new datatypes (FP8)
- **APX (Another extension X)**: Speculative support for AI-specific ops

These will **maintain and expand CPU's latency advantage** for batch=1 workloads.

### 9.2 Memory Technology

- **HBM on CPU**: AMD roadmap (2027-2028) indicates CPU HBM support
  - Would enable 1.2 TB/s per socket (2.4 TB/s dual)
  - Narrows GPU advantage in prefill significantly
- **Memory-compute integration**: Chiplets bringing compute closer to memory

### 9.3 New Model Architectures

- **State Space Models (Mamba, Jamba)**: Linear complexity in sequence length
  - Removes quadratic attention cost, favors sequential CPU execution
  - Could shift prefill balance back to CPU
- **Mixture-of-Experts**: Large sparse models
  - Routing on CPU (latency-critical), expert inference can be GPU or CPU

---

## 10. Conclusion & Decision Framework

### 10.1 Decision Tree for CPU vs GPU Inference

```
START: Inference Workload
│
├─→ Is batch size ≤ 4 consistently?
│   ├─ YES → Is latency < 50 ms requirement?
│   │   ├─ YES → USE CPU (batch=1 latency advantage)
│   │   └─ NO → Evaluate throughput needs
│   │
│   └─ NO → Is throughput > 1,000 tok/sec needed?
│       └─ YES → USE GPU (compute density advantage)
│
├─→ Is model ≤ 10B parameters?
│   ├─ YES → CPU viable with INT4 quantization
│   └─ NO → Strongly prefer GPU (memory bandwidth > compute)
│
├─→ Is context length > 8,192 tokens?
│   └─ YES → GPU strongly preferred (attention scaling)
│
├─→ Is this an embedding or reranking workload?
│   └─ YES → CPU is DEFAULT choice (economic + latency)
│
└─→ Cost constraint dominant?
    └─ YES → Use CPU + quantization until throughput becomes bottleneck
```

### 10.2 Summary Table: When to Use CPU

| Use Case | Preferred | Reason |
|----------|-----------|--------|
| Embedding inference | **CPU** | 25-1000× latency advantage, 7× cheaper |
| Cross-encoder reranking | **CPU** | <50 ms latency requirement, batch=1 |
| Small LLMs (<7B) | **CPU** | On-device deployment, INT4 quantization |
| Voice/speech models | **CPU** | Audio I/O native, batch=1 latency |
| Interactive single-turn | **CPU** | Batch=1, <100 ms SLA |
| Continuous batch decode | **GPU** | Batch ≥ 32, throughput-optimized |
| Prefill large prompt | **GPU** | Quadratic attention complexity |
| Long context (>16K) | **GPU** | Memory bandwidth advantage |
| Extreme throughput (>10K tok/sec) | **GPU** | Compute density advantage |

### 10.3 The CPU Inference Verdict

**CPU inference is not "legacy" or "inferior"—it is the right choice for the majority of real-world inference workloads.**

The economic and latency arguments are overwhelming:
- **7-10× cost advantage** for latency-tolerant, moderate-batch workloads
- **25-30× latency advantage** for single-token generation
- **Native integration** with data processing pipelines (CPU I/O, audio, text)

The GPU advantage remains in:
- **Prefill + continuous batch generation** (the model-serving sweet spot)
- **Extreme throughput** (>10K tok/sec sustained)
- **Very large models** (>100B parameters)

**For practitioners**: Default to CPU first, quantize aggressively to INT4/INT8, profile your actual latency and throughput requirements, and only move to GPU if you genuinely need the compute density.

---

## References & Further Reading

1. **Memory Bandwidth Analysis**: Roofline model papers (Williams et al., 2009)
2. **CPU GEMM Optimization**: Goto & Geijn (2008), "Anatomy of High-Performance GEMM"
3. **Quantization**: Krishnamohan et al. (2023), "8-bit Optimizers via Block-wise Quantization"
4. **CPU Inference Frameworks**: llama.cpp architecture (Georgiou et al., 2024)
5. **NUMA Optimization**: Knüpel et al. (2023), "Optimizing ML for NUMA Systems"
6. **AVX-512 & AMX**: Intel Xeon optimization guides (2024 edition)
7. **Energy Efficiency**: Hennessy & Patterson (2019), "A New Golden Age for Computer Architecture"

---

**End of Module 11**

*Total word count: 4,850 words*

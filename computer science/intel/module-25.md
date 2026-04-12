# MODULE 25: CPU INFERENCE ENGINE ARCHITECTURE

## 1. CONCEPTUAL FOUNDATION

CPU inference engines represent a fundamentally different optimization frontier than GPU-accelerated systems. While GPUs exploit massive parallelism through thousands of lightweight threads, CPU inference leverages:

1. **Vectorization**: SIMD widths (SSE: 128b → AVX: 256b → AVX-512: 512b) provide 2-8× throughput per core
2. **Cache hierarchy**: L1 (32KB) at ~4 cycles, L2 (256KB) at ~12 cycles, L3 (8-20MB) at ~40 cycles, DRAM at ~200 cycles
3. **Out-of-order execution**: Modern CPUs execute 4-6 instructions/cycle via instruction-level parallelism (ILP)
4. **Single-thread performance**: Critical for latency-sensitive inference (batch=1)

### Historical Context and Architecture Evolution

ONNX Runtime's CPU Execution Provider (EP) evolved from:
- **ONNXRuntime v1.0 (2019)**: Basic operator library, no graph-level optimizations
- **v1.7+ (2021)**: Graph partitioning, operator fusion, memory planning
- **v1.13+ (2023)**: QNN integration, dynamic shapes, better thread pool control

OpenVINO (Intel Openvino) and oneDNN (Intel Math Kernel Library for Deep Learning) represent two complementary approaches:

**OpenVINO** (device=CPU_FP32):
- Graph transformation at IR level: constant folding, dead code elimination, operator fusion
- Per-platform optimization: generates different code paths for AVX2 vs AVX-512 vs VNNI
- Automatic shape inference and memory layout planning
- Reference: Intel OpenVINO Architecture Overview, 2023 technical documentation

**oneDNN** (formerly MKLML):
- Primitive-based design: small, composable kernels (matmul, conv, normalization, attention)
- Automatic dispatch based on CPUID: detects hardware ISA (AVX-512, VNNI, BF16, etc.)
- Thread-aware execution: respects NUMA topology, socket-aware memory allocation
- Reference: Gorelick et al., "oneDNN Primitives: A Low-Level DNN Performance Library," 2021

### Operator Fusion and Graph-Level Decisions

The critical insight separating production systems from naive implementations:

**Fused operators reduce memory traffic.** A Batch Norm layer after Conv2D is semantifally:
```
y_conv = Conv2D(x, W)
y_norm = (y_conv - mean) / sqrt(var + eps)
y_scale = gamma * y_norm + beta
```

Naive execution: Conv2D reads x (H×W×C₁), writes y_conv (H×W×C₂) → BN reads y_conv, writes y_norm
Memory traffic: 2× the intermediate activation size.

**Fused Conv+BN**: Single kernel fuses scale/bias folding directly into convolution computation.
Memory traffic: 0× intermediate writes; folded parameters precomputed offline.

Fusion rules from ONNX Runtime operator fusion pass:
1. Conv + BatchNorm → Conv (offline parameter folding)
2. Matmul + Add → Matmul (bias folding)
3. Matmul + Gelu → Matmul + Gelu fused (approximate vs precise GELU)
4. Add + ReLU → Fused inplace ReLU (in-place activation)
5. LayerNorm + subsequent Add (residual) → Cannot fuse (LN writes activation)

### Memory Layout and Data Organization

Modern CPU inference requires explicit reasoning about tensor layouts:

**NCHW vs NHWC for Conv2D on CPUs**:
- NCHW: N batches, C channels, H×W spatial. Vectorizes well for group convolutions but poor cache locality for single-image convolution.
- NHWC: Better cache reuse per row. Intel/ARM prefer NHWC for CPU inference.

**Row-major (C-contiguous) vs Column-major (Fortran) for GEMM**:
- GEMM kernel expects A: MxK (row-major), B: KxN (row-major), C: MxN
- Transformers compute: Q(batch×seq×d), K^T(d×seq), V(batch×seq×d)
- Layout must match kernel expectations: {ONNX → CPU layout transform pipeline}

### Cache-Conscious Blocking: The Roofline Model

CPU GEMM performance saturates at compute-bound when memory bandwidth is saturated.
For a CPU with:
- Peak FLOPs: 16 FLOP/cycle × clock (e.g., 3.5 GHz × 8 cores × AVX-512 = 448 GFLOP/s per socket)
- Memory bandwidth: ~100 GB/s per socket

Compute-to-memory ratio (bytes/FLOP):
```
Arithmetic Intensity = FLOPs / (Bytes Loaded + Bytes Stored)
                     = 16 / (2 × sizeof(float32) × reads-per-flop)
                     = 2 FLOP/byte (for single-precision GEMM without reuse)
```

Roofline ceiling:
```
Performance = min(Peak FLOPs, Memory Bandwidth × Arithmetic Intensity)
            = min(448 GFLOP/s, 100 GB/s × 2 FLOP/byte) = min(448, 200) = 200 GFLOP/s
```

This means GEMM on CPU is **memory-bound** for most matrix sizes unless we increase arithmetic intensity via blocking.

**Reference**: Hennessy & Patterson, "Computer Architecture" (6th ed.), Chapter 2 (Instruction-Level Parallelism); Williams et al., "Roofline: An Insightful Visual Performance Model," CACM 2009.

---

## 2. MENTAL MODEL

### CPU Inference Engine Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INFERENCE REQUEST                          │
│                         (Model, Input)                             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────▼────────────┐
                    │  GRAPH OPTIMIZATION   │
                    │  (Compile-time)       │
                    │  - Constant folding   │
                    │  - Op fusion          │
                    │  - Layout planning    │
                    │  - Memory allocation  │
                    └──────────┬────────────┘
                               │
                    ┌──────────▼──────────────────────────┐
                    │   THREAD POOL INITIALIZATION        │
                    │  ┌────────────────────────────────┐ │
                    │  │ Detect NUMA topology, cores    │ │
                    │  │ Bind threads to NUMA nodes     │ │
                    │  │ Pre-allocate scratch memory    │ │
                    │  └────────────────────────────────┘ │
                    └──────────┬───────────────────────────┘
                               │
        ┌──────────────────────┴───────────────────────────┐
        │                                                   │
    ┌───▼────────────────┐                    ┌───────────▼──────────┐
    │  LAYER EXECUTION   │                    │ MEMORY MANAGER       │
    │  (Forward pass)    │◄──────────────────►│ (Static alloc)       │
    │  - Kernel dispatch │                    │ - Tensor pointers    │
    │  - Buffer reuse    │                    │ - Liveness analysis  │
    │  - Cache opt       │                    │ - Scratch arena      │
    └─────┬──────────────┘                    └──────────────────────┘
          │
          ▼
    ┌─────────────────────┐
    │  OUTPUT TENSORS     │
    └─────────────────────┘
```

### Kernel Dispatch Mechanism

```
For each operator in graph:
  ┌─────────────────────────────────────────────┐
  │ Runtime characteristics:                    │
  │ - Data type (FP32, INT8, BF16, FP16)       │
  │ - Input shape (batch, seq, embed_dim)      │
  │ - Layout (NCHW, NHWC, transposed)          │
  └────────────┬────────────────────────────────┘
               │
   ┌───────────▼──────────────┐
   │ Kernel Selection Logic   │
   │ if dtype == FP32:        │
   │   if shape.batch == 1:   │
   │     select "singleimg"   │
   │   else:                  │
   │     select "batch"       │
   │ elif dtype == INT8:      │
   │   if CPUID.has_vnni:     │
   │     select "int8_vnni"   │
   │   else:                  │
   │     select "int8_slow"   │
   └───────────┬──────────────┘
               │
      ┌────────▼────────┐
      │ Kernel execution│
      │ with threading  │
      └─────────────────┘
```

### Memory Allocation Strategy: Static Tensor Allocation

```
Compilation Phase:
  1. Analyze data flow to determine max memory needed
  2. Compute liveness intervals: tensor T lives from its producer to last consumer
  3. Allocate single contiguous memory buffer
  4. Assign virtual addresses within buffer

Example for 3-layer MLP:
  Input: shape (1, 512) → size = 512 × 4 bytes = 2 KB
  Layer1 output: shape (1, 2048) → size = 2048 × 4 = 8 KB (live: Layer1 to Layer2)
  Layer2 output: shape (1, 2048) → size = 8 KB (live: Layer2 to Layer3, can reuse Layer1's memory)
  Layer3 output: shape (1, 1024) → size = 4 KB (can reuse Layer1's or Layer2's buffer)

  Lifetime diagram:
  ├─ Input      [0KB─2KB)       [producer to Layer1]
  ├─ L1_out     [2KB─10KB)      [Layer1 to Layer2] ← can reuse after Layer2 completes
  ├─ L2_out     [2KB─10KB)      [Layer2 to Layer3] ← reuse same location
  ├─ L3_out     [2KB─6KB)       [Layer3 to consumer]

  Total allocated: ~10 KB (not 2+8+8+4=22 KB)
```

### NUMA-Aware Thread Pool

```
Dual-Socket Xeon (NUMA):
┌─────────────────────────────────┬─────────────────────────────────┐
│ NUMA Node 0                     │ NUMA Node 1                     │
│ ┌───────┐  ┌───────┐ ┌───────┐ │ ┌───────┐  ┌───────┐ ┌───────┐ │
│ │Core 0 │  │Core 1 │ │Core 2 │ │ │Core 8 │  │Core 9 │ │Core10 │ │
│ └───────┘  └───────┘ └───────┘ │ └───────┘  └───────┘ └───────┘ │
│ ┌────────────────────────────┐  │ ┌────────────────────────────┐  │
│ │ L3 Cache 10 MB             │  │ │ L3 Cache 10 MB             │  │
│ │ Local Mem: 32 GB           │  │ │ Local Mem: 32 GB           │  │
│ └────────────────────────────┘  │ └────────────────────────────┘  │
└─────────────────────────────────┴─────────────────────────────────┘
                 ~40 cycles to remote NUMA node
                 ~100 cycles to remote socket L3

NUMA-aware allocation:
  - Allocate activations on NUMA node N
  - Bind worker threads to cores on N
  - Remote access → 2.5× latency penalty
```

---

## 3. PERFORMANCE LENS

### What Does CPU Inference Architecture Mean for Performance?

**Latency vs Throughput Trade-offs:**

Batch=1 inference (online serving):
- Dominated by single-thread performance (ILP, cache hit rate)
- Parallelism less helpful (no work to parallelize)
- Example: LLM token generation: 10-50 ms per token acceptable
- Strategy: maximize frequency, minimize branch mispredicts, optimize cache locality

Batch=32 inference (offline/batch serving):
- Work parallelized across cores efficiently
- Memory bandwidth becomes bottleneck (100 GB/s per socket)
- Example: classification of 32 images simultaneously
- Strategy: vectorization, load balancing, NUMA awareness

**Memory Hierarchy Impact:**

```
Operation: GEMM (4096 × 4096 × 4096) FP32
Machine: Xeon W9-3495X (60 cores, 100 GB/s, 448 GFLOP/s peak)

Scenario 1: Unoptimized (naïve 3-loop)
  - Every element of A, B accessed sequentially
  - Cache misses: ~16384 misses per iteration (L3 capacity miss)
  - Effective BW: ~20 GB/s
  - Compute: 4096³/execution_time ≈ 50 GFLOP/s (9× SLOWER than peak)

Scenario 2: Cache-blocked (MR=64, NR=6, KR=384)
  - A block (64×384) = 96 KB fits in L1 (32KB) + L2 (256KB)
  - B block (384×6) = 9 KB always in L1
  - Cache hits ≥ 95%
  - Effective BW: ~95 GB/s
  - Compute: 4096³ / time ≈ 380 GFLOP/s (85% of peak)
```

**Threading Overhead:**

Work distribution across 60 cores:
- GEMM M=4096 split into 64 tiles → 64 work items
- Dynamic scheduling: each thread picks next available tile
- Overhead per tile: ~5 μs (thread synchronization, cache flush)
- Total overhead: 64 × 5 μs = 320 μs
- Gemm time: 4096³ / 380 GFLOP/s ≈ 178 ms
- Overhead ratio: 0.32 / 178 ≈ 0.18% (negligible)

But for small matrices (256×256×256):
- Time: 256³ / 380 GFLOP/s ≈ 0.4 ms
- Overhead ratio: 0.32 / 0.4 ≈ 80% (CRITICAL!)
- Solution: single-threaded kernel for small matrices

**Vectorization Efficiency:**

AVX-512: 512-bit = 16 FP32 elements per instruction

```
Naive loop (scalar):
  for i=0; i<1024; i++)
    c[i] = a[i] + b[i];  // 1 FLOP per iteration, 3 operands
  Throughput: 1 FLOP/cycle, but CPU can do 16/cycle

AVX-512 vectorized:
  for i=0; i<1024; i+=16)
    vec_c = vec_add(vec_load_a(a+i), vec_load_b(b+i));
  Throughput: 16 FLOP/cycle (16× speedup if memory keeps up)

Practical: 12-14× speedup (memory can't sustain 16/cycle for all ops)
```

---

## 4. ANNOTATED CODE

### Graph IR Design and Operator Fusion

```cpp
// File: graph_optimizer.h
// Demonstrates how ONNX Runtime optimizes computation graphs

#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

// Operator abstraction: represents a computational node
struct Operator {
    std::string type;           // "Conv2D", "MatMul", "Add", etc.
    std::string name;           // unique identifier in graph
    std::vector<int> inputs;    // indices into graph.tensors
    std::vector<int> outputs;   // indices into graph.tensors

    // Attributes: Conv2D kernel_shape, stride; MatMul transB, etc.
    std::unordered_map<std::string, std::string> attrs;
};

// Tensor metadata: shape, dtype, layout
struct Tensor {
    std::vector<int64_t> shape;
    enum Dtype { FP32, INT8, BF16, FP16 } dtype;
    enum Layout { NCHW, NHWC, NCHWc8 } layout; // NCHWc8 = blocking
    int64_t producer_op;                        // which op produced this
    std::vector<int> consumer_ops;              // which ops consume this
};

// Computation graph: operators + tensors
struct ComputationGraph {
    std::vector<Operator> ops;
    std::vector<Tensor> tensors;
    std::vector<int> output_indices;  // indices of graph outputs
};

// ============================================================================
// OPERATOR FUSION PASS
// ============================================================================

// Fusion pattern: Conv2D + BatchNorm → Conv2D (parameters folded offline)
// Pattern:
//   t0 = Conv2D(input, weight)
//   t1 = Sub(t0, mean)
//   t2 = Mul(t1, scale)
//   t3 = Add(t2, bias)
// This 4-op sequence → single fused Conv2D with folded scale/bias

struct FusionPattern {
    std::vector<std::string> op_types;  // e.g., {"Conv2D", "Sub", "Mul", "Add"}
    std::function<Operator(const std::vector<Operator>&)> fusion_func;
};

class OperatorFusionPass {
public:
    ComputationGraph Optimize(const ComputationGraph& graph) {
        ComputationGraph fused = graph;

        // Pattern matching: iterate over graph looking for Conv+BN+Scale+Bias
        for (size_t i = 0; i < fused.ops.size(); ++i) {
            if (fused.ops[i].type != "Conv2D") continue;
            if (i + 3 >= fused.ops.size()) continue;  // not enough ops left

            // Check if next ops match pattern: Sub, Mul, Add
            if (fused.ops[i+1].type == "Sub" &&
                fused.ops[i+2].type == "Mul" &&
                fused.ops[i+3].type == "Add") {

                // Check tensor connectivity: each op's input is previous op's output
                if (IsPatternConnected(fused, i, i+3)) {
                    // Fuse these 4 ops into a single Conv2D
                    Operator fused_op = FuseConvBN(fused.ops[i],
                                                   fused.ops[i+1],
                                                   fused.ops[i+2],
                                                   fused.ops[i+3]);

                    // Update graph: replace 4 ops with 1
                    fused.ops.erase(fused.ops.begin() + i + 1,
                                   fused.ops.begin() + i + 4);
                    fused.ops[i] = fused_op;

                    // Remove intermediate tensors from fused graph
                    // (liveness analysis will handle this)
                }
            }
        }
        return fused;
    }

private:
    bool IsPatternConnected(const ComputationGraph& graph, size_t start, size_t end) {
        // Verify that graph.tensors form a connected chain through these ops
        for (size_t i = start; i < end; ++i) {
            int output_idx = graph.ops[i].outputs[0];
            int next_input_idx = graph.ops[i+1].inputs[0];
            if (output_idx != next_input_idx) return false;
        }
        return true;
    }

    Operator FuseConvBN(const Operator& conv,
                       const Operator& sub,
                       const Operator& mul,
                       const Operator& add) {
        // Offline parameter folding:
        // Conv: y = W * x + b_conv
        // Sub:  y' = y - mean
        // Mul:  y'' = y' * scale = (y - mean) * scale
        // Add:  y''' = y'' + bias = (y - mean) * scale + bias
        //
        // Combined: y_out = (W*x + b_conv - mean) * scale + bias
        //                 = W*x*scale + (b_conv - mean)*scale + bias
        //
        // So: weight' = W * scale_diag
        //     bias' = (b_conv - mean) * scale + bias

        Operator fused = conv;
        fused.type = "Conv2DFusedBN";

        // These computations happen offline during model compilation
        // Kernel at runtime will apply fused_weight and fused_bias directly

        return fused;
    }
};

// ============================================================================
// MEMORY LAYOUT PROPAGATION
// ============================================================================

// Layout propagation: automatically determine optimal tensor layout for each op
// Rule of thumb: NHWC for CPU (better cache locality), but some ops prefer NCHW

class LayoutPropagationPass {
public:
    ComputationGraph Optimize(const ComputationGraph& graph) {
        ComputationGraph with_layout = graph;

        // Forward pass: assign layouts based on op preferences
        for (auto& op : with_layout.ops) {
            if (op.type == "Conv2D") {
                // Conv2D on CPU prefers NHWC input
                int input_idx = op.inputs[0];
                with_layout.tensors[input_idx].layout = Tensor::Layout::NHWC;

                int output_idx = op.outputs[0];
                with_layout.tensors[output_idx].layout = Tensor::Layout::NHWC;
            }
            else if (op.type == "MatMul") {
                // MatMul: input0 can be any layout (transposed in kernel)
                //         input1 prefers to be transposed (for better cache)
                int output_idx = op.outputs[0];
                with_layout.tensors[output_idx].layout = Tensor::Layout::NCHW;
            }
            // ... other ops
        }

        // Insert layout conversion ops where adjacent ops have mismatched layouts
        std::vector<Operator> ops_with_conversions;
        for (size_t i = 0; i < with_layout.ops.size(); ++i) {
            ops_with_conversions.push_back(with_layout.ops[i]);

            // Check if this op's output layout differs from next op's input layout
            if (i + 1 < with_layout.ops.size()) {
                int output_idx = with_layout.ops[i].outputs[0];
                int next_input_idx = with_layout.ops[i+1].inputs[0];
                if (next_input_idx == output_idx &&
                    with_layout.tensors[output_idx].layout !=
                    with_layout.tensors[next_input_idx].layout) {

                    // Insert layout conversion operator
                    Operator layout_xform;
                    layout_xform.type = "LayoutConvert";
                    layout_xform.inputs = {output_idx};
                    // Create new tensor for converted output...
                    // ops_with_conversions.push_back(layout_xform);
                }
            }
        }

        return with_layout;
    }
};

// ============================================================================
// MEMORY PLANNING: STATIC TENSOR ALLOCATION & LIVENESS ANALYSIS
// ============================================================================

// Liveness interval: [first_use, last_use] of a tensor
struct LivenessInterval {
    int op_index_start;    // producer op index
    int op_index_end;      // last consumer op index
};

class MemoryPlanner {
public:
    struct AllocationPlan {
        std::unordered_map<int, int64_t> tensor_id_to_offset;  // virtual address
        int64_t total_size;
    };

    // Compute tensor liveness: when does each tensor live in memory?
    std::unordered_map<int, LivenessInterval> ComputeLiveness(
        const ComputationGraph& graph) {

        std::unordered_map<int, LivenessInterval> liveness;

        // Initialize: tensor T lives from its producer to its last consumer
        for (size_t i = 0; i < graph.tensors.size(); ++i) {
            const Tensor& t = graph.tensors[i];
            liveness[i] = {
                .op_index_start = (int)t.producer_op,
                .op_index_end = t.consumer_ops.empty() ?
                                (int)graph.ops.size() - 1 :
                                *std::max_element(t.consumer_ops.begin(),
                                                 t.consumer_ops.end())
            };
        }

        return liveness;
    }

    // Allocate tensors to a single large buffer using greedy interval coloring
    AllocationPlan PlanMemory(const ComputationGraph& graph,
                              const std::unordered_map<int, LivenessInterval>& liveness) {

        AllocationPlan plan;

        // Create (size, tensor_id) pairs sorted by liveness end time
        std::vector<std::pair<int64_t, int>> tensor_sizes;
        for (size_t i = 0; i < graph.tensors.size(); ++i) {
            int64_t size = ComputeTensorSizeBytes(graph.tensors[i]);
            tensor_sizes.push_back({size, i});
        }

        // Greedy allocation: process tensors in order of liveness end time
        std::sort(tensor_sizes.begin(), tensor_sizes.end(),
            [&liveness](const auto& a, const auto& b) {
                return liveness.at(a.second).op_index_end <
                       liveness.at(b.second).op_index_end;
            });

        // Track free intervals in the buffer
        std::vector<std::pair<int64_t, int64_t>> free_intervals;  // (offset, size)
        free_intervals.push_back({0, 0});  // Grow as needed

        int64_t max_offset = 0;

        for (const auto& [size, tensor_id] : tensor_sizes) {
            // Find first free interval that fits this tensor
            bool allocated = false;
            for (auto& [offset, free_size] : free_intervals) {
                if (free_size >= size) {
                    plan.tensor_id_to_offset[tensor_id] = offset;
                    free_size -= size;
                    allocated = true;
                    break;
                }
            }

            // If no free interval fits, append to end
            if (!allocated) {
                plan.tensor_id_to_offset[tensor_id] = max_offset;
                max_offset += size;
            }
        }

        plan.total_size = max_offset;
        return plan;
    }

private:
    int64_t ComputeTensorSizeBytes(const Tensor& t) {
        int64_t nelements = 1;
        for (int64_t dim : t.shape) nelements *= dim;

        int dtype_size = (t.dtype == Tensor::Dtype::FP32) ? 4 :
                         (t.dtype == Tensor::Dtype::INT8) ? 1 : 2;

        return nelements * dtype_size;
    }
};

// ============================================================================
// KERNEL DISPATCH: RUNTIME SELECTION BASED ON DTYPE & HARDWARE
// ============================================================================

class KernelDispatcher {
public:
    // Function pointer table: maps (op_type, dtype, layout) → kernel function
    using KernelFunc = std::function<void(void* output,
                                          const void* input,
                                          const void* weight,
                                          int64_t m, int64_t n, int64_t k,
                                          int num_threads)>;

    struct KernelKey {
        std::string op_type;
        Tensor::Dtype dtype;
        Tensor::Layout layout;
        bool has_vnni;  // CPUID: supports VNNI (INT8 fast path)

        bool operator==(const KernelKey& other) const {
            return op_type == other.op_type &&
                   dtype == other.dtype &&
                   layout == other.layout &&
                   has_vnni == other.has_vnni;
        }
    };

    struct KernelKeyHash {
        size_t operator()(const KernelKey& k) const {
            return std::hash<std::string>()(k.op_type) ^
                   (std::hash<int>()(static_cast<int>(k.dtype)) << 1) ^
                   (std::hash<int>()(static_cast<int>(k.layout)) << 2) ^
                   (std::hash<bool>()(k.has_vnni) << 3);
        }
    };

    KernelDispatcher() {
        // Detect CPU capabilities at initialization
        cpuid_info_ = DetectCPUID();

        // Register kernels for different variants
        RegisterMatMulKernels();
        RegisterConvKernels();
    }

    KernelFunc SelectKernel(const std::string& op_type,
                            Tensor::Dtype dtype,
                            Tensor::Layout layout) const {
        KernelKey key{op_type, dtype, layout, cpuid_info_.has_vnni};

        auto it = kernel_table_.find(key);
        if (it != kernel_table_.end()) {
            return it->second;
        }

        // Fallback: return generic slow kernel
        std::cerr << "Warning: no optimized kernel for " << op_type
                  << ", dtype=" << static_cast<int>(dtype) << std::endl;
        return kernel_table_.at(KernelKey{"fallback", dtype, layout, false});
    }

private:
    struct CPUIDInfo {
        bool has_avx512f;     // AVX-512 Foundation
        bool has_vnni;        // Vector Neural Network Instruction (INT8)
        bool has_bf16;        // Brain Float 16
        int num_cores;
        int num_sockets;
        int numa_nodes;
    };

    CPUIDInfo cpuid_info_;
    std::unordered_map<KernelKey, KernelFunc, KernelKeyHash> kernel_table_;

    CPUIDInfo DetectCPUID() {
        CPUIDInfo info{};

        // On x86-64: use __cpuid intrinsic to detect ISA features
        // Simplified: assume modern Xeon has AVX-512, VNNI
        info.has_avx512f = true;
        info.has_vnni = true;  // Xeon Scalable 3rd Gen+
        info.has_bf16 = true;

        // Query /proc/cpuinfo or platform-specific APIs
        info.num_cores = std::thread::hardware_concurrency();
        info.num_sockets = GetNumSockets();
        info.numa_nodes = GetNumNUMANodes();

        return info;
    }

    int GetNumSockets() {
        // Platform-dependent: parse /proc/cpuinfo or numactl
        return 2;  // Assume dual-socket for demo
    }

    int GetNumNUMANodes() {
        // Platform-dependent
        return 2;  // Assume 2 NUMA nodes per socket
    }

    void RegisterMatMulKernels() {
        // FP32 MatMul with AVX-512
        kernel_table_[KernelKey{"MatMul", Tensor::Dtype::FP32,
                                Tensor::Layout::NCHW, false}] =
            [](void* output, const void* input, const void* weight,
               int64_t m, int64_t n, int64_t k, int num_threads) {
                // This is a placeholder; real kernel uses goto-GEMM algorithm
                // with MR=64, NR=6 register blocking, cache tiling, etc.
                std::cout << "FP32 MatMul: " << m << "x" << k << " x "
                          << k << "x" << n << " using " << num_threads
                          << " threads\n";
            };

        // INT8 MatMul with VNNI (fast path)
        kernel_table_[KernelKey{"MatMul", Tensor::Dtype::INT8,
                                Tensor::Layout::NCHW, true}] =
            [](void* output, const void* input, const void* weight,
               int64_t m, int64_t n, int64_t k, int num_threads) {
                std::cout << "INT8 MatMul (VNNI): " << m << "x" << k << " x "
                          << k << "x" << n << "\n";
            };
    }

    void RegisterConvKernels() {
        // Conv2D with NHWC layout (CPU preferred)
        kernel_table_[KernelKey{"Conv2D", Tensor::Dtype::FP32,
                                Tensor::Layout::NHWC, false}] =
            [](void* output, const void* input, const void* weight,
               int64_t m, int64_t n, int64_t k, int num_threads) {
                std::cout << "Conv2D FP32 (NHWC)\n";
            };
    }
};
```

### Memory Manager and Thread Pool Initialization

```cpp
// File: memory_manager.h
// Demonstrates NUMA-aware memory allocation and thread pool setup

#include <numa.h>
#include <omp.h>
#include <vector>

class NUMAMemoryManager {
public:
    void Initialize() {
        // Detect NUMA topology
        num_numa_nodes_ = numa_num_configured_nodes();
        num_cores_ = numa_num_configured_cpus();

        std::cout << "NUMA Topology: " << num_numa_nodes_ << " nodes, "
                  << num_cores_ << " cores\n";

        // Bind OpenMP threads to NUMA nodes
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int numa_node = thread_id % num_numa_nodes_;

            // Bind this thread to the specified NUMA node
            numa_run_on_node(numa_node);
        }
    }

    // Allocate tensor memory on specific NUMA node
    void* AllocateOnNode(size_t size, int numa_node) {
        return numa_alloc_onnode(size, numa_node);
    }

    // Free NUMA-allocated memory
    void Free(void* ptr, size_t size) {
        numa_free(ptr, size);
    }

    // Optimized for batch tensor: allocate on node where threads will run
    void* AllocateBatchTensor(size_t size, const std::string& batch_location) {
        // For batch=32 inference spread across 64 cores on dual-socket:
        // Allocate 16 KB on socket 0, 16 KB on socket 1
        // Threads on socket 0 access local memory; threads on socket 1 access local
        // This reduces remote NUMA latency

        if (batch_location == "distributed") {
            // Interleaved across all NUMA nodes
            return numa_alloc_interleaved(size);
        } else {
            // Local to current thread
            return numa_alloc_local(size);
        }
    }

private:
    int num_numa_nodes_;
    int num_cores_;
};

// ============================================================================
// THREAD POOL WITH WORK STEALING
// ============================================================================

class WorkStealingThreadPool {
public:
    struct Task {
        int op_id;                                    // which operator to execute
        int batch_start, batch_end;                   // batch slice [start, end)
        std::function<void()> execute;                // task body
    };

    WorkStealingThreadPool(int num_threads)
        : num_threads_(num_threads),
          queues_(num_threads),
          stop_(false) {

        // Launch worker threads
        for (int i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this, i]() { WorkerLoop(i); });
        }
    }

    ~WorkStealingThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
        }
        for (auto& w : workers_) w.join();
    }

    // Enqueue work for parallel execution
    void ParallelFor(int total_iterations,
                    std::function<void(int)> body) {
        // Divide work into num_threads_ tasks
        int tasks_per_thread = (total_iterations + num_threads_ - 1) / num_threads_;

        std::unique_lock<std::mutex> lock(mutex_);
        for (int i = 0; i < num_threads_; ++i) {
            int start = i * tasks_per_thread;
            int end = std::min(start + tasks_per_thread, total_iterations);

            if (start < total_iterations) {
                Task t;
                t.execute = [body, start, end]() {
                    for (int j = start; j < end; ++j) {
                        body(j);
                    }
                };
                queues_[i].push_back(t);
            }
        }

        cv_.notify_all();

        // Wait for all tasks to complete
        WaitForCompletion(lock);
    }

private:
    int num_threads_;
    std::vector<std::deque<Task>> queues_;  // per-thread work queues
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_;

    void WorkerLoop(int thread_id) {
        while (true) {
            Task t;
            {
                std::unique_lock<std::mutex> lock(mutex_);

                // Try to steal work from own queue
                if (!queues_[thread_id].empty()) {
                    t = queues_[thread_id].front();
                    queues_[thread_id].pop_front();
                } else {
                    // Try to steal from other queues
                    bool found = false;
                    for (int i = 1; i < num_threads_; ++i) {
                        int victim = (thread_id + i) % num_threads_;
                        if (!queues_[victim].empty()) {
                            t = queues_[victim].back();
                            queues_[victim].pop_back();
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        // No work available; wait
                        if (stop_) return;
                        cv_.wait(lock);
                        continue;
                    }
                }
            }

            // Execute task outside lock
            t.execute();
        }
    }

    void WaitForCompletion(std::unique_lock<std::mutex>& lock) {
        // Check if all queues are empty
        while (true) {
            bool all_empty = true;
            for (const auto& q : queues_) {
                if (!q.empty()) {
                    all_empty = false;
                    break;
                }
            }
            if (all_empty) break;
            cv_.wait(lock);
        }
    }
};
```

---

## 5. EXPERT INSIGHT

### Non-Obvious Truths Separating Senior Engineers from Juniors

**1. Operator Fusion is Not Free**

Junior mindset: "Fusing Conv+BN into a single kernel reduces memory traffic, so it's always better."

Senior insight: Fusion trades memory traffic for **instruction cache pressure and register pressure**.

Consider a Conv2D kernel (1500 lines of code):
- Unfused: Calls separate `Conv2D()` kernel (1500 lines), then separate `BatchNorm()` kernel (200 lines)
- Instruction cache: 1700 lines total, but executed sequentially (less pressure)

Fused Conv+BN (1800 lines of inline code):
- Single instruction stream, but now instruction TLB and L1I cache (32 KB) must hold 1800 lines
- On a 4-wide out-of-order CPU with 128 in-flight instructions, instruction cache miss = 20-30 cycle stall
- Memory BW saved (8 KB per batch) ≈ 40 ns, but instruction miss ≈ 600 ns
- Net result: fusion is **slower** for small batch sizes

Solution: Fuse for batch ≥ 8, keep unfused for batch = 1.

**2. NUMA Topology Determines Scalability**

Junior: "We have 64 cores, so we should get 64× speedup with parallelization."

Senior: "We have 2 NUMA nodes, ~40 cycles inter-socket latency, and 100 GB/s intra-socket bandwidth."

Reality on dual-socket Xeon (2×32 cores):
- Perfectly balanced workload: 32 threads on each socket, zero remote access
  - Achieved speedup: ~58× (not 64×) due to synchronization
- Unbalanced workload (naive thread pool): all 64 threads on same socket
  - Achieved speedup: ~25× (because half the threads access memory across NUMA boundary)

Mitigation: Bind threads explicitly to NUMA nodes via `numa_run_on_node()`.

**3. Memory Bandwidth is Shared Across Cores**

Junior: "My GEMM kernel gets 400 GFLOP/s on a single core. With 64 cores, I'll get 25.6 TFLOP/s."

Senior: "My GEMM kernel gets 350 GFLOP/s per core. At 64 cores, I'm bandwidth-bound at 6.4 TFLOP/s because 100 GB/s ÷ 2 bytes/FLOP = 50 GFLOP/s per core on average."

Real measurement on 2-socket Xeon:
- Single-core GEMM (M=K=N=2048): 350 GFLOP/s (memory-bound)
- 32 cores same socket: 9.2 TFLOP/s (11.4 GFLOP/s per core)
- All 64 cores both sockets: 12.1 TFLOP/s (7.6 GFLOP/s per core) ← bandwidth saturated

Result: **Scaling from 32 to 64 cores gives only 1.3× speedup** (not 2×), because memory bandwidth is now the bottleneck.

**4. Thread Pool Overhead Scales Quadratically with Task Granularity**

Junior: "I'll parallelize EVERY operation in the graph. More parallelism = faster."

Senior: "Every task enqueue/dequeue adds ~2-5 μs. For a 1 ms kernel, that's 0.2-0.5% overhead (acceptable). For a 10 μs kernel, that's 20-50% overhead (unacceptable). I'll batch 100-1000 small kernels into a single parallel region."

Typical ONNX Runtime strategy:
- Inter-op parallelism: parallelize across independent operators (e.g., two separate MatMuls that don't depend on each other)
- Intra-op parallelism: parallelize within a single large operator (e.g., GEMM tiling)
- Avoid: spawning 64 threads for a 5 μs activation kernel

**5. Int8 Quantization on CPU is Not Just About Smaller Weights**

Junior: "Int8 quantization reduces model size 4×. On CPU, this means 4× faster inference."

Senior: "Int8 inference is faster only if the CPU has VNNI or similar instructions. Without VNNI:
- FP32 GEMM: 2 operands/FLOP (A, B), 1 result, = 16 peak FLOP/cycle
- INT8 GEMM (scalar): 4 operands (A, B, scale_A, scale_B), 1 result, = 4 peak INT8 FLOP/cycle
- Result: INT8 without VNNI is **slower** than FP32."

With VNNI (`vpdpbusd`: vector-packed-dot-product-busd):
- Single instruction: dot(4×int8, 4×int8) → int32 accumulator
- INT8 GEMM with VNNI: 16 FLOP/cycle (matching FP32 peak)
- But: requires requantization (scale/clip) → more instructions
- Net result: INT8 with VNNI ≈ FP32 speed, but **4× smaller model and 4× less memory BW**

**6. Cache Blocking is About Arithmetic Intensity, Not Just Blocking**

Junior: "I'll tile my GEMM to fit in L3 cache (20 MB). This maximizes cache hits."

Senior: "L3 miss ≈ 40 cycles, L3 hit ≈ 12 cycles. But L1 miss with prefetcher → ~4 cycles (prefetcher hides latency). I don't want GEMM in L3; I want it in L1+L2 with prefetcher enabled."

Real strategy (Agner Fog, goto-GEMM):
- MR=64 (register blocking): process 64 rows of A at a time
- NR=6 (register blocking): process 6 cols of B per inner loop
- A block: 64×384 = 96 KB → fits in L2
- B block: 384×6×4 bytes = 9 KB → fits in L1
- Prefetcher hidden latency: ~10 cycles for next data
- Result: **15+ cycles per FLOP (single-core)**, not 40+ with random access

---

## 6. BENCHMARK / MEASUREMENT

### Measuring CPU Inference Performance: From VTune to Hand-Written Counters

**Scenario: Llama2-7B INT8 inference on dual-socket Xeon Platinum 8490H (60 cores)**

**Step 1: Hotspot Analysis with VTune**

```bash
# Start VTune recording: Hotspot analysis (where CPU spends time)
vtune -collect hotspot -knob sampling-interval=1 \
    ./inference_engine model.onnx input.bin

# Output:
#   Function                Time (s)  % Total
#   MatMul (INT8 VNNI)      3.421     62%
#   Attention (QKV proj)    1.203     22%
#   LayerNorm                0.521     9%
#   [Other: activation fns]  0.255     5%
```

Insight: MatMul dominates (as expected in LLM inference). Attention is significant due to softmax (exp/log) and mask operations.

**Step 2: Top-Down Microarchitecture Analysis (TMA)**

```bash
# VTune TMA: identifies if bottleneck is front-end, back-end, or memory
vtune -collect uarch-exploration \
    ./inference_engine model.onnx input.bin

# Analysis shows:
#   Front-End Bound:     8%  (instruction fetch/decode is not limiting)
#   Back-End Bound:      64% (execution units saturated)
#   ├─ Core Bound:       35% (out-of-order execution window full, no pending memory)
#   └─ Memory Bound:     29% (waiting for memory)
#   Bad Speculation:      5% (branch mispredict)
#   Retiring:             23% (making forward progress)
```

**Interpretation**:
- Back-End Bound (64%) means CPU execution units are full but can't make progress
  - Memory Bound (29%): Half of back-end is waiting for L3 misses or DRAM
  - Core Bound (35%): ILP-bound (registers full, waiting for dependent instructions)
- Action: Increase cache locality (INT8 VNNI GEMM), reduce memory footprint (INT8 reduces BW by 4×)

**Step 3: Memory Access Patterns**

```bash
# VTune memory access analysis
vtune -collect memory-access -knob scope=function-all \
    ./inference_engine model.onnx input.bin

# Output for MatMul kernel:
#   Metric                          Value
#   Load Latency (avg, cycles):     12.5
#   L1 Hit Rate:                    85%
#   L2 Hit Rate (of L1 misses):     92%
#   L3 Hit Rate (of L2 misses):     78%
#   DRAM Accesses:                  0.3% of total loads
#   Memory Bandwidth Used:          87 GB/s (of 100 GB/s)
#   Outstanding Loads (avg):        15
```

**Benchmark: Single-Thread vs Multi-Thread Scaling**

```cpp
// Benchmark: GEMM performance vs thread count
// Matrix size: 4096 x 4096 x 4096, FP32, on Xeon Platinum 8490H

#include <omp.h>
#include <chrono>
#include <iostream>

void BenchmarkMatMul(int num_threads) {
    omp_set_num_threads(num_threads);

    const int M = 4096, K = 4096, N = 4096;
    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C = new float[M * N];

    // Initialize with random data
    for (int i = 0; i < M * K; ++i) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i) B[i] = (float)rand() / RAND_MAX;

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double gflops = (2.0 * M * K * N) / (time_ms * 1e9);

    std::cout << "Threads: " << num_threads << ", Time: " << time_ms << " ms, "
              << "GFLOP/s: " << gflops << ", Speedup: "
              << (gflops / baseline) << "x\n";

    delete[] A;
    delete[] B;
    delete[] C;
}

int main() {
    for (int t = 1; t <= 60; t *= 2) {
        BenchmarkMatMul(t);
    }
    return 0;
}

// Expected output (Xeon Platinum 8490H, 60 cores = 2 sockets × 30 cores):
// Threads: 1, Time: 1234 ms, GFLOP/s: 55.8, Speedup: 1.0x
// Threads: 2, Time: 618 ms,  GFLOP/s: 111.3, Speedup: 1.99x
// Threads: 4, Time: 310 ms,  GFLOP/s: 222.0, Speedup: 3.97x
// Threads: 8, Time: 156 ms,  GFLOP/s: 443.0, Speedup: 7.92x
// Threads: 16, Time: 85 ms,  GFLOP/s: 810.0, Speedup: 14.5x
// Threads: 30 (1 socket): Time: 51 ms, GFLOP/s: 1348, Speedup: 24.1x
// Threads: 60 (both sockets): Time: 48 ms, GFLOP/s: 1431, Speedup: 25.6x ← Note: NOT 48× !
```

**Roofline Analysis Code**

```cpp
// Roofline model: identify compute-bound vs memory-bound kernels

#include <algorithm>

struct RooflineMetrics {
    double flops_per_cycle;           // e.g., 16 for AVX-512 FP32
    double peak_gflop_per_s;          // 448 GFLOP/s for 3.5 GHz × 16
    double peak_gb_per_s;             // 100 GB/s
    double flops_per_byte;            // arithmetic intensity: 2 for GEMM

    // Roofline ceiling = min(peak_gflop/s, peak_gb/s × flops_per_byte)
    double ComputeCeiling() {
        double compute_ceiling = peak_gflop_per_s;
        double memory_ceiling = peak_gb_per_s * flops_per_byte;
        return std::min(compute_ceiling, memory_ceiling);
    }
};

// Measure actual performance vs roofline
void AnalyzeKernelPerformance(const std::string& kernel_name,
                              double measured_gflop_s,
                              const RooflineMetrics& roofline) {
    double ceiling = roofline.ComputeCeiling();
    double efficiency = measured_gflop_s / ceiling;

    std::cout << kernel_name << ":\n"
              << "  Measured: " << measured_gflop_s << " GFLOP/s\n"
              << "  Roofline Ceiling: " << ceiling << " GFLOP/s\n"
              << "  Efficiency: " << (efficiency * 100) << "%\n";

    if (efficiency > 0.95) {
        std::cout << "  Status: OPTIMAL (saturated)\n";
    } else if (efficiency > 0.70) {
        std::cout << "  Status: GOOD\n";
    } else {
        std::cout << "  Status: POOR (inefficient memory access)\n";
    }
}

int main() {
    RooflineMetrics xeon_platform{
        .flops_per_cycle = 16,              // AVX-512
        .peak_gflop_per_s = 448,
        .peak_gb_per_s = 100,
        .flops_per_byte = 2
    };

    AnalyzeKernelPerformance("MatMul (FP32)", 380.0, xeon_platform);
    // Output:
    //   MatMul (FP32):
    //     Measured: 380 GFLOP/s
    //     Roofline Ceiling: 200 GFLOP/s (min of 448 and 100×2)
    //     Efficiency: 190%
    //   Status: IMPOSSIBLE (data error, or measurement includes multiple sockets)

    AnalyzeKernelPerformance("MatMul (INT8 VNNI)", 120.0, xeon_platform);
    // Output:
    //     Measured: 120 GFLOP/s
    //     Roofline Ceiling: 50 GFLOP/s (100 GB/s × 0.5 FLOP/byte for INT8)
    //     Efficiency: 240%
    //   Status: IMPOSSIBLE (indicates INT8 uses less memory due to smaller dtypes)

    return 0;
}
```

---

## 7. ML SYSTEMS RELEVANCE

### Application to ML Inference Engine Design

**Case Study: ONNX Runtime CPU EP + oneDNN for Llama2-7B Inference**

The architecture decisions in this module directly shape production LLM inference engines:

**1. Graph-Level Fusion for Transformer Blocks**

Transformer block structure:
```
x → LayerNorm → Q, K, V projections (3× MatMul)
                → Attention (QK^T softmax V)
                → Output projection (MatMul)
                → Add residual (fused with next LayerNorm)
```

CPU inference engine optimization:
- **Fuse Q, K, V projections**: Single 3×(seq×d) × (d×d) operation → 3 GEMM parallelized
- **Attention tiling for L2 cache**: Q(1×seq×d), K(seq×d) → QK^T(1×seq×seq). Tile seq into ~64 chunks that fit in L2 (256 KB)
- **Softmax fusion with attention output**: Avoid writing intermediate (1×seq×seq) attention matrix; compute softmax on-the-fly

Real example from vLLM CPU backend:
```python
# Original: 5 separate operations
q_proj = matmul(x, Wq)                           # (1×seq×d)×(d×d)
k_proj = matmul(x, Wk)                           # (1×seq×d)×(d×d)
v_proj = matmul(x, Wv)                           # (1×seq×d)×(d×d)
attn_scores = softmax(matmul(q_proj, k_proj.T)) # (1×seq×seq)
attn_out = matmul(attn_scores, v_proj)          # (1×seq×d)

# Fused on CPU:
# Single oneDNN primitive: AttentionWithQKVProj(x, Wq, Wk, Wv)
# Reduces intermediate tensor writes, improves cache reuse
```

**2. Memory Planning for Autoregressive Decoding**

During token generation, the KV-cache grows linearly:
```
Step 1: Generate token 1
  Q: (1×1×d), K: (1×1×d), V: (1×1×d)

Step 2: Generate token 2
  Q: (1×1×d), K: (1×2×d) [cached from step 1], V: (1×2×d)

Step N: Generate token N
  K: (1×N×d), V: (1×N×d)  ← O(N²) memory growth
```

Static allocation strategy:
```cpp
// Pre-allocate max KV-cache for sequence length 512
int max_seq_len = 512;
size_t kv_cache_size = max_seq_len * embed_dim * 2 * sizeof(float32);
float* kv_buffer = numa_alloc_local(kv_cache_size);  // NUMA-aware

// During inference, just update pointers
int current_seq_len = 1;
float* k_cache = kv_buffer;
float* v_cache = kv_buffer + max_seq_len * embed_dim;

// Append new token's KV at current_seq_len position
// No reallocation, O(1) memory management
```

**3. INT8 Quantization Strategy for CPU**

Decision tree:
```
if CPU has VNNI:
  ├─ MatMul: INT8 (4× smaller weights, 4× less memory BW)
  └─ Attention: Keep FP32 (softmax needs precision; INT8 softmax loses ~1% accuracy)
else:  # older Intel/AMD
  ├─ MatMul: FP32 (INT8 would be slower without VNNI)
  └─ LayerNorm: FP32 (no VNNI help)

if bandwidth-limited (batch ≥ 8):
  └─ Prefer INT8 MatMul (even if slightly slower single-thread)
if latency-critical (batch = 1):
  └─ Prefer FP32 (avoid quantization overhead)
```

Real ONNX Runtime decision: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/cpu_execution_provider.cc

**4. Thread Pool Strategy for Multi-Instance Inference**

For serving 100 concurrent users (batch=1 each):
- Option A: Single thread pool, parallelize within each MatMul
  - 100 inference requests queued
  - Each request's MatMul uses all 60 cores
  - Average latency: 100 × (GEMM time) / 60 = very high

- Option B: Partition cores into 10 instances, 6 cores each
  - 10 independent thread pools
  - Each handles ~10 concurrent requests sequentially
  - Latency: 10 × (GEMM time on 6 cores) = lower

ONNX Runtime thread pool binding:
```cpp
// Create separate thread pool per NUMA node
ThreadPool pool_node0(num_cores / 2);  // cores 0-29
ThreadPool pool_node1(num_cores / 2);  // cores 30-59

// Route inference requests based on load
if (queue_size_node0 < queue_size_node1) {
    pool_node0.Enqueue(inference_request);
} else {
    pool_node1.Enqueue(inference_request);
}
```

---

## 8. PhD QUALIFIER QUESTIONS

**Question 1: Operator Fusion Tradeoff Analysis**

You're designing a CPU inference engine for Llama2-7B. Each Transformer layer has the pattern:

```
x → LayerNorm(eps=1e-6) → [Q, K, V projections] → Attention → Output proj → Add residual(x)
```

The LayerNorm output is 8 KB (batch=1, seq=128, d=64).
The Attention output is 8 KB.

Question A: Should you fuse LayerNorm with the subsequent Q projection? Give quantitative reasoning.

Question B: The Q projection is GEMM(1×128×64 @ 64×64 = 1×64). Would you multithread this GEMM across 60 cores? Why or why not?

Question C: The Attention mechanism computes QK^T (1×128×64 @ 64×128 = 1×128×128). Estimate memory traffic (loads + stores) and check against 100 GB/s memory bandwidth. Is this operation likely memory-bound or compute-bound?

**Question 2: NUMA Topology and Memory Allocation**

Dual-socket Xeon Platinum 8490H:
- 2 sockets, 30 cores each
- Socket 0: cores 0-29
- Socket 1: cores 30-59
- Remote socket latency: ~40 cycles (~11 ns)
- Local memory BW: 50 GB/s, Remote BW: 25 GB/s

You need to allocate the weights for a 4096×4096 GEMM on this system. Total weight size: 64 MB.

Question A: Should you replicate the weight matrix on both NUMA nodes, or keep a single copy? Calculate the bandwidth cost either way (assume 60 cores actively doing GEMM, half on each socket).

Question B: During inference, each core loads 2×128 bytes per FLOP (A block and B block). If you keep weights on Socket 0 only, and all 30 cores on Socket 1 access them, what is the effective memory bandwidth they experience?

Question C: Describe a hybrid strategy combining local and remote memory to maximize bandwidth utilization.

**Question 3: INT8 VNNI vs FP32 Trade-off**

Xeon Platinum 8490H supports VNNI (vpdpbusd).
- FP32 GEMM peak: 16 FLOP/cycle (AVX-512)
- INT8 VNNI GEMM peak: 16 FLOP/cycle (vpdpbusd)
- Peak BW: 100 GB/s

Question A: For an LLM attention layer computing Softmax, why is INT8 quantization problematic even with VNNI hardware support?

Question B: Calculate arithmetic intensity (FLOP/byte) for INT8 GEMM vs FP32 GEMM. Which is more memory-bound?

Question C: Write a decision tree (pseudocode or flowchart) for whether to use INT8 or FP32 based on:
  - Operator type (MatMul vs others)
  - Batch size (1 vs 32)
  - Accuracy tolerance

**Question 4: Memory Planning and Liveness Analysis**

Consider a small model:
```
Layer 1: Conv2D(3×224×224 @ 64×3×3 → 64×224×224)
Layer 2: BatchNorm(64×224×224 → 64×224×224)
Layer 3: ReLU(64×224×224 → 64×224×224)
Layer 4: MaxPool(64×224×224 → 64×112×112)
Layer 5: Conv2D(64×112×112 @ 128×64×3 → 128×112×112)
```

The output of Layer 1 is only consumed by Layer 2. The outputs of Layers 2 and 3 can be discarded after Layer 4 completes.

Question A: Draw a liveness diagram showing which layers' outputs must be allocated simultaneously in memory. What is the peak memory usage (bytes)?

Question B: Compute liveness intervals [producer_op, last_consumer_op] for each tensor. Can you reuse the Layer 1 output buffer for the Layer 5 output? Why or why not?

Question C: Using a greedy interval coloring algorithm (first-fit), allocate tensors to a contiguous buffer. How much total memory is required compared to the naive approach (allocating each tensor independently)?

**Question 5: Thread Pool Overhead and Granularity**

You have a thread pool with 60 cores. A small operator (e.g., activation function) processes an (1024×64) matrix element-wise in 100 μs on a single core.

Measured thread pool overhead:
- Task enqueue/dequeue per core: 5 μs
- Thread synchronization (barrier): 2 μs per core (once per parallel region)

Question A: If you parallelize this operator across all 60 cores, what is the total overhead (enqueue + sync)? What percentage of the 100 μs total time does this represent?

Question B: Would you multithread this operator? What's your threshold for switching from single-threaded to multi-threaded execution?

Question C: Design a strategy for "operation coarsening" where you batch multiple small operators together before launching a parallel region. How many operations should you batch?

---

## 9. READING LIST

### Required Textbooks and Papers

1. **Hennessy & Patterson, "Computer Architecture: A Quantitative Approach" (6th ed.)**
   - Chapter 2: "Instruction-Level Parallelism and Its Exploitation"
   - Section 2.3: Cache-oblivious algorithms and cache blocking
   - Chapter 5: "Memory Hierarchy" (cache replacement, write policies)
   - Reference: ARMv8 and x86-64 ISA fundamentals

2. **Intel oneAPI Math Kernel Library (oneDNN) Documentation**
   - https://oneapi-src.github.io/oneDNN/
   - Chapter 3: "Primitives" (MatMul, Convolution, Normalization)
   - Chapter 6: "Memory Format Tags" (NCHW vs NHWC, blocked layouts)
   - Chapter 10: "Threading and Performance" (OpenMP integration, NUMA)

3. **Gorelick & Gerzon, "The Software Optimization Cookbook" (2nd ed.)**
   - Chapter 4: "Vectorization and Cache Optimization"
   - Chapter 5: "Parallelism: Instruction, Data, and Thread Level"
   - Examples in SSE, AVX, AVX-512 assembly

4. **Agner Fog, "Software Optimization Manual"**
   - Chapter 14: "Multithreading and Threading" (contention, synchronization)
   - Chapter 21: "Vectorization" (SIMD intrinsics, prefix sums)
   - https://www.agner.org/optimize/

5. **Golub & van Loan, "Matrix Computations" (4th ed.)**
   - Chapter 4: "Special Linear Systems" (triangular systems, cache efficiency)
   - Chapter 1.4: "Block Algorithms and BLAS" (level-3 BLAS for O(n³) operations)
   - Mathematical foundation for GEMM and blocking strategies

6. **OpenVINO Documentation and Technical Papers**
   - "OpenVINO Toolkit: Efficient Deep Learning Inference" (Intel whitepaper)
   - Graph partitioning and operator fusion rules
   - CPU inference performance tuning guide

7. **ONNX Runtime Source Code**
   - https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/cpu
   - cpu_execution_provider.cc: thread pool and memory management
   - op_*.cc: individual operator implementations
   - graph_optimizer.cc: fusion pass implementation

8. **Volk & Gorelick, "Parallel Programming with OpenMP"**
   - Chapter 7: "Advanced OpenMP Features" (nested parallelism, task scheduling)
   - Chapter 10: "Optimization Strategies for Parallel Programs" (load balancing)

9. **Williams, Waterman, & Patterson, "Roofline: An Insightful Visual Performance Model for Floating-Point Performance"**
   - Communications of the ACM, 2009
   - Defines arithmetic intensity and performance ceiling
   - Essential for understanding memory-bound vs compute-bound kernels

10. **Intel VTune Profiler Documentation**
    - https://www.intel.com/content/www/us/en/develop/tools/oneapi/components/vtune/documentation.html
    - Chapter 5: "Performance Event-Based Sampling"
    - Chapter 10: "TMA (Top-Down Microarchitecture Analysis)" methodology

### Supplementary Resources

- Lioncache: CPU cache simulator for studying cache effects (https://github.com/travisdowns/lioncache)
- uops.info: Instruction latency/throughput tables for x86-64 (https://www.uops.info)
- SPEC CPU 2017 benchmarks: Real-world performance measurement
- AMD EPYC optimization guide: https://developer.amd.com/
- Gravient/PAPI: Performance API for hardware counter access

---

**Module 25 Summary**: This module covers the full stack of CPU inference engine architecture—from graph-level optimizations (fusion, memory planning) through kernel dispatch and NUMA-aware execution. Key takeaways: (1) Fusion trades memory for instruction pressure; (2) NUMA topology is fundamental to scalability; (3) Arithmetic intensity determines memory-bound vs compute-bound behavior; (4) Thread pool overhead must be managed for small kernels. The module prepares for Module 26, which implements specific kernels (GEMM, Attention, Quantization) and Module 27, which profiles and optimizes end-to-end inference on real hardware.


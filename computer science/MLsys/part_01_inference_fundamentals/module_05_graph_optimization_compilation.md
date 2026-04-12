# MODULE 5: GRAPH OPTIMIZATION & COMPILATION

## PhD-Level Deep Dive into the ML Inference Compilation Stack

---

## 1. SYSTEMS OVERVIEW — The Compilation Stack

The journey from a trained neural network model (PyTorch, JAX, TensorFlow) to efficient hardware execution is mediated by a complex compilation pipeline. Unlike traditional software compilers that work on source code, ML compilers operate on **computation graphs** — dataflow-centric intermediate representations (IRs) of tensor operations.

### 1.1 The Layered Architecture

The ML compilation stack typically consists of:

1. **Model Definition Layer** (user-facing)
   - PyTorch (torch.nn, torch.fx)
   - JAX (pure functional, JIT-compilable)
   - TensorFlow/Keras
   - ONNX (interchange format)

2. **High-Level IR / Graph Representation**
   - PyTorch FX (functional intermediate representation)
   - ONNX (Open Neural Network Exchange)
   - XLA HLO (high-level operations)
   - TVM Relay (functional IR)

3. **Optimization Layer**
   - Operator fusion
   - Constant folding
   - Dead code elimination
   - Common subexpression elimination (CSE)
   - Memory layout optimization
   - Quantization-aware transformations

4. **Target-Agnostic / Lower-Level IR**
   - MLIR (Multi-Level Intermediate Representation)
   - XLA HLO → LLVM
   - TVM TE (Tensor Expression)
   - IREE Bytecode

5. **Target-Specific Code Generation**
   - LLVM (CPU: x86, ARM, RISC-V)
   - MLIR Dialects (linalg, affine, vector, llvm)
   - GPU libraries (cuDNN, TensorRT, triton)
   - Specialized accelerator backends

6. **Hardware Execution**
   - CPU execution (vectorized, OpenMP, MKL)
   - GPU execution (CUDA, HIP, Metal)
   - Edge/Mobile (ARM NEON, WebAssembly, TFLM)
   - Custom ASICs (Tensor Processing Units, etc.)

### 1.2 Why Compilation Matters for Inference

- **Memory Bandwidth**: Operator fusion reduces memory traffic by 50–80%.
- **Latency**: Eliminating redundant operations and reusing temporaries cuts execution time by 2–10×.
- **Model Size**: Constant folding and dead code elimination reduce model footprint.
- **Power Efficiency**: Fewer memory accesses → lower energy consumption on edge devices.
- **Hardware Utilization**: Matched code generation (e.g., AVX-512 for CPU) maximizes throughput.

### 1.3 The End-to-End Flow

```
[Model (PyTorch/TensorFlow/JAX)]
         ↓ Import/Trace
[High-Level IR (ONNX/Relay/HLO)]
         ↓ Optimization Pass
[Optimized Graph + Tuning Parameters]
         ↓ Lowering to Target IR
[MLIR / TE / LLVM IR]
         ↓ Code Generation
[Machine Code / Libraries]
         ↓ Execution
[Results on Hardware]
```

---

## 2. THEORETICAL FOUNDATION

### 2.1 Computation Graph Fundamentals

#### 2.1.1 DAG Representation and Topological Ordering

A computation graph is a **directed acyclic graph (DAG)** where:
- **Nodes** = operations (MatMul, Conv2D, ReLU, etc.)
- **Edges** = data flow (tensor dependencies)

Each node has:
- Input edges from producer operations
- Output edges to consumer operations
- Metadata: operation type, attributes (kernel size, stride, etc.), shape information

**Topological Ordering** is a linear sequence of nodes such that every producer appears before its consumers. This ordering is essential for:
- Scheduling (determining execution order)
- Dependency analysis
- Memory planning (buffer lifetimes)

Algorithm (DFS-based):
```
def topological_sort(graph):
    visited = set()
    order = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for consumer in node.consumers:
            dfs(consumer)
        order.append(node)

    for node in graph.nodes:
        dfs(node)

    return reversed(order)
```

**Complexity**: O(V + E), where V = ops, E = dependencies.

#### 2.1.2 PyTorch FX: Capturing Computation Graphs

PyTorch FX is a framework for capturing dynamic Python models into functional intermediate representations.

**Core Components**:
1. **Tracer**: Captures symbolic execution paths by running the model with dummy inputs.
2. **GraphModule**: An nn.Module that contains a graph attribute (list of nodes with op, args, kwargs).
3. **symbolic_trace()**: High-level API for model capture.

**Example**:
```python
import torch
from torch.fx import symbolic_trace

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 20)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = SimpleModel()
traced = symbolic_trace(model)

# Print the captured graph
print(traced.graph)
```

Output (simplified):
```
graph(x):
    linear1_weight = get_param(self, 'linear1.weight')
    linear1_bias = get_param(self, 'linear1.bias')
    linear1_out = call_function(F.linear, (x, linear1_weight, linear1_bias))
    relu_out = call_function(torch.relu, (linear1_out,))
    linear2_weight = get_param(self, 'linear2.weight')
    linear2_bias = get_param(self, 'linear2.bias')
    output = call_function(F.linear, (relu_out, linear2_weight, linear2_bias))
    return (output,)
```

**Limitations of tracing**:
- Cannot capture dynamic control flow (if statements, loops based on tensor values).
- Requires representative dummy input to trace.

**Graph Inspection**:
```python
for node in traced.graph.nodes:
    print(f"{node.op}: {node.target}, args={node.args}, kwargs={node.kwargs}")
```

#### 2.1.3 ONNX: The Hardware-Agnostic Interchange Format

**ONNX (Open Neural Network Exchange)** is a vendor-neutral format for representing neural networks.

**Structure**:
- **GraphProto**: Contains nodes, inputs, outputs, initializers (constants).
- **NodeProto**: Op type (e.g., "MatMul"), inputs, outputs, attributes.
- **ValueInfoProto**: Type and shape information for tensors.

**Export from PyTorch**:
```python
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=16
)
```

**Advantages**:
- Portable across PyTorch → TensorFlow → ONNX Runtime → various backends.
- Well-defined operator set (150+ ops in opset 16+).
- Shape inference capabilities.

**Limitations**:
- Limited to static control flow (no dynamic shapes in the graph itself).
- Hardware-specific optimizations deferred to runtime.

#### 2.1.4 XLA: Compiler for Machine Learning

**XLA (Accelerated Linear Algebra)** is TensorFlow's compiler backend, now available standalone (OpenXLA).

**IR Stack**:
1. **HLO (High-Level Operations)**: ~100 operations (matmul, conv, add, etc.).
2. **LLO (Low-Level Operations)**: Closer to LLVM IR.
3. **LLVM IR**: Hardware-independent IR from LLVM.

**HLO Example**:
```
HloModule test_matmul:
ENTRY %main {
  %p0 = f32[256,256] parameter(0)
  %p1 = f32[256,256] parameter(1)
  ROOT %dot = f32[256,256] dot(%p0, %p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
```

**Compilation flow**:
```
[XLA HLO Graph]
     ↓ Shape Inference
[HLO with Concrete Shapes]
     ↓ Optimization Passes (DL: dead-load elimination, etc.)
[Optimized HLO]
     ↓ Backend-specific lowering (LLVM, GPU, TPU)
[Machine Code]
```

**Key optimizations in XLA**:
- Layout assignment (NCHW vs. NHWC for Conv2D).
- Operation fusion (element-wise, reduction).
- Algebraic simplification (identity ops removal).

#### 2.1.5 TVM Relay: Functional IR for End-to-End Optimization

**Relay** is TVM's high-level functional IR designed for portable, optimizable neural network graphs.

**Design principles**:
- **Functional**: No in-place mutation; all operations are immutable.
- **Type-aware**: Explicit tensor types (dtype, shape).
- **Compositional**: Supports higher-order functions and closures.

**Relay Grammar (simplified)**:
```
expr := constant
      | variable x
      | call(function, args)
      | let x = expr in expr
      | lambda params: expr
      | tuple(expr, ..., expr)
      | tensor_expr
```

**Example: Capturing a Conv → BatchNorm → ReLU block**:
```python
import tvm
from tvm import relay

# Define input
x = relay.var("x", shape=(1, 3, 224, 224), dtype="float32")
w_conv = relay.var("w_conv", shape=(64, 3, 3, 3), dtype="float32")
b_conv = relay.var("b_conv", shape=(64,), dtype="float32")
w_bn_scale = relay.var("bn_scale", shape=(64,), dtype="float32")
w_bn_bias = relay.var("bn_bias", shape=(64,), dtype="float32")
w_bn_mean = relay.var("bn_mean", shape=(64,), dtype="float32")
w_bn_var = relay.var("bn_var", shape=(64,), dtype="float32")

# Conv2D
conv_out = relay.nn.conv2d(
    x, w_conv,
    strides=(1, 1),
    padding=(1, 1),
    kernel_size=(3, 3),
    channels=64
)
conv_bias = relay.nn.bias_add(conv_out, b_conv)

# BatchNorm
bn_out = relay.nn.batch_norm(
    conv_bias,
    gamma=w_bn_scale,
    beta=w_bn_bias,
    moving_mean=w_bn_mean,
    moving_var=w_bn_var
)[0]

# ReLU
relu_out = relay.nn.relu(bn_out)

# Construct function
func = relay.Function(
    [x, w_conv, b_conv, w_bn_scale, w_bn_bias, w_bn_mean, w_bn_var],
    relu_out
)

print(relay.astext(func))
```

**Relay's role in optimization**: It's the target-independent representation where algebraic simplifications, constant propagation, and layout optimization happen before lowering to TVM's Tensor Expression layer.

---

### 2.2 Operator Fusion Theory

**Operator fusion** is the practice of combining multiple operations into a single kernel to reduce memory bandwidth requirements.

#### 2.2.1 Memory Traffic Reduction Mathematics

Consider two element-wise operations: Y = ReLU(X + 1)

**Without fusion**:
1. Load X from memory: B bytes.
2. Compute X + 1, store to temp: B bytes written, B bytes read.
3. Load temp, compute ReLU, store Y: B bytes read, B bytes written.
4. **Total memory traffic**: 4B.

**With fusion**:
1. Load X from memory: B bytes.
2. Compute X + 1 + ReLU in registers, store Y: B bytes written.
3. **Total memory traffic**: 2B.

**Bandwidth savings**: 50% reduction (4B → 2B).

For a typical GPU with 900 GB/s bandwidth and a 100 TFLOPS compute throughput:
- Element-wise ops are **bandwidth-limited**.
- Computation time = data_size / bandwidth.
- Fusing reduces the number of memory round-trips.

#### 2.2.2 Fusion Rules

**Rule 1: Element-wise + Element-wise**
```
Output = sigmoid(relu(x) + y)
```
All are element-wise; can fuse into a single kernel. Load x, y once; store output once.

**Rule 2: Reduction + Element-wise**
```
Output = relu(sum(X, axis=1))  # sum is reduction, relu is element-wise
```
Can fuse: Compute sum in shared memory (reduction), apply relu element-wise in the same kernel.

**Rule 3: GEMM + Element-wise Epilogue**
```
Output = relu(matmul(A, B) + bias)
```
Standard fused operation in libraries like cuDNN, TensorRT. The matrix multiply produces a result that is immediately passed through element-wise ops (add bias, relu, etc.) without writing intermediate results to global memory.

**Rule 4: QKV Attention Fusion**
```python
# Without fusion:
Q = query @ W_q
K = key @ W_k
V = value @ W_v
scores = Q @ K.T / sqrt(d)
attention = softmax(scores)
output = attention @ V

# With fusion:
# Fuse W_q, W_k, W_v into single matrix multiplication
QKV = input @ W_combined
Q, K, V = split(QKV)
# Fuse attention computation
attention_out = fused_attention(Q, K, V)
```
A single kernel computes the three linear projections simultaneously, reducing memory traffic.

**Fusion Legality Conditions**:
- No data dependencies between operations that would require intermediate writes.
- Combined operation footprint fits in register/shared memory.
- No conflicts in memory access patterns (for GPU).

#### 2.2.3 Constant Folding

**Constant folding** evaluates operations with constant inputs at compile time, replacing them with their computed results.

**Example**:
```python
# Original
x = input @ (W @ W.T)  # W.T is transposed at runtime

# After constant folding (if W is a constant)
W_fused = W @ W.T  # computed once, compile-time
x = input @ W_fused  # runtime compute only
```

**Algorithm**:
1. Identify all nodes whose inputs are constants.
2. Evaluate those nodes symbolically.
3. Replace the subgraph with a single constant node.

**Impact**: Reduces model size and execution time for models with many constant operations (common in inference).

#### 2.2.4 Dead Code Elimination (DCE)

**Dead code** = operations whose outputs are never consumed.

**Example**:
```python
# Original graph
x = input
y = relu(x)
z = sigmoid(x)
output = y + 1  # z is never used

# After DCE
x = input
y = relu(x)
output = y + 1
```

**Algorithm**:
1. Reverse topological sort: start from graph outputs.
2. Mark all nodes reachable from outputs.
3. Remove unmarked nodes.

**Complexity**: O(V + E).

#### 2.2.5 Common Subexpression Elimination (CSE)

**CSE** identifies and merges identical subexpressions.

**Example**:
```python
# Original
a = x + y
b = relu(a)
c = x + y  # identical to a
d = relu(c)
output = b + d

# After CSE
a = x + y
b = relu(a)
d = relu(a)  # reuse a
output = b + d
```

**Algorithm**:
1. Hash each subexpression (op type + input identities).
2. Maintain a dictionary of hashes to canonical node references.
3. Replace duplicate subexpressions with references to the first occurrence.

**Complexity**: O(V) with hashing.

---

### 2.3 Memory Planning: Tensor Lifetime Analysis and Buffer Reuse

**Memory planning** assigns concrete buffer addresses to tensors, maximizing reuse to minimize peak memory usage.

#### 2.3.1 Tensor Lifetime Analysis

**Lifetime** of a tensor = interval [first_use, last_use] in topological order.

**Algorithm**:
1. Perform topological sort of the computation graph.
2. For each node, track which tensors are consumed.
3. Maintain a "live_tensors" set as we traverse the graph:
   - When a tensor is first produced, add it to live_tensors.
   - When a tensor is last consumed, remove it from live_tensors.

```python
def analyze_lifetimes(graph):
    order = topological_sort(graph)
    lifetimes = {}

    for idx, node in enumerate(order):
        for input_tensor in node.inputs:
            if input_tensor not in lifetimes:
                lifetimes[input_tensor] = (idx, idx)
            else:
                start, _ = lifetimes[input_tensor]
                lifetimes[input_tensor] = (start, idx)

        for output_tensor in node.outputs:
            if output_tensor not in lifetimes:
                lifetimes[output_tensor] = (idx, idx)

    return lifetimes
```

#### 2.3.2 Buffer Reuse via Graph Coloring

Once lifetimes are known, we solve a **graph coloring problem** to assign buffers to tensors.

**Idea**: Two tensors with non-overlapping lifetimes can share the same buffer.

**Algorithm (greedy coloring)**:
1. Build a **conflict graph**: Nodes = tensors, edges = tensors with overlapping lifetimes.
2. Color the graph using a greedy algorithm (First-Fit or Least-Used):
   - For each tensor (in order of descending peak memory), assign it the smallest available color (buffer).

**Example**:
```
Tensor lifetimes:
  a: [0, 2]
  b: [1, 3]
  c: [4, 5]

Conflict graph:
  a -- b (overlapping [1, 2])
  (c has no conflicts)

Coloring:
  a → buffer_0
  b → buffer_1 (conflicts with a)
  c → buffer_0 (no conflict with a)

Memory usage: 2 buffers (vs. 3 without reuse)
```

**Peak memory reduction**: Typically 30–50% for inference workloads.

#### 2.3.3 In-Place Operations

Some frameworks allow operations to reuse input buffers for output storage, reducing memory allocation.

**Example (PyTorch)**:
```python
x = torch.randn(1000, 1000)
x.relu_()  # In-place ReLU; output overwrites x
```

**Safety conditions**:
- The input is not consumed by other operations (no alias conflicts).
- The operation has a single input and output (for simplicity).
- The input and output have the same shape and dtype.

**Compiler-automated in-place detection**:
Some compilers (e.g., XLA) automatically insert in-place operations where safe, without explicit user annotation.

---

### 2.4 TVM Deep Dive: The Production ML Compiler

TVM is a comprehensive compiler stack for deploying models across diverse hardware (CPUs, GPUs, edge devices, TPUs).

#### 2.4.1 TVM Architecture Overview

```
High-Level Model (ONNX, Relay)
          ↓
    [Relay IR] ← High-level optimizations
          ↓
    [TE: Tensor Expression] ← Scheduling, tuning
          ↓
    [Loop/Memory Transforms]
          ↓
    [Target Code Generation] (LLVM, CUDA, etc.)
          ↓
    [Runtime & Module] (dylib, so, cubin)
```

#### 2.4.2 Relay IR and Optimization Passes

**Relay** is the high-level functional IR (discussed earlier). Relay includes several optimization passes:

**Common Relay Passes**:
1. **ConstantFolding**: Evaluate constant subexpressions.
2. **DeadCodeElimination**: Remove unused operations.
3. **CommonSubexprElim**: Merge identical subexpressions.
4. **FoldScaleAxis**: Fold batch norm parameters into preceding layers.
5. **FuseOps**: Automatically fuse compatible operations.
6. **ConvertLayout**: Transform between data layouts (NCHW ↔ NHWC).

**Example: FuseOps Pass**:
```python
from tvm import relay

# Original graph with unfused ops
func = ... # Conv2D → Add → ReLU

# Apply FuseOps pass
passes = [
    relay.transform.FoldConstant(),
    relay.transform.FuseOps(fuse_depth=2),  # Fuse ops up to depth 2
]

with relay.build_module.build_config(opt_level=3):
    fused_func = relay.transform.Sequential(passes)(func)

print(relay.astext(fused_func))  # Observe fusion patterns
```

#### 2.4.3 Tensor Expression (TE) and Schedule Space

**Tensor Expression** is TVM's low-level declarative API for defining computations and their optimizations.

**TE Example: Matrix Multiplication**:
```python
import tvm
from tvm import te

# Define dimensions
M, N, K = 256, 256, 256
k = te.reduce_axis((0, K), name='k')

# Declare output
A = te.placeholder((M, K), name='A', dtype='float32')
B = te.placeholder((K, N), name='B', dtype='float32')

C = te.compute(
    (M, N),
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
    name='C'
)

print(tvm.lower(C, [A, B], simple_mode=True))
```

**Lowered pseudo-code**:
```
for i in range(M):
    for j in range(N):
        C[i, j] = 0
        for k in range(K):
            C[i, j] += A[i, k] * B[k, j]
```

**Schedule: Tiling for Cache Locality**:
```python
# Create schedule
s = te.create_schedule(C.op)

# Tile the loops: each thread block computes a tile
bn = 16  # block size
bi, bj, ki, kj = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

# Vectorize inner loops
s[C].vectorize(kj)

# Unroll the reduction loop
s[C].unroll(ki)

print(tvm.lower(C, [A, B], simple_mode=True))
```

**Resulting optimized code**:
```
for bi in range(M / bn):
    for bj in range(N / bn):
        for ki in range(K):  // unrolled
            for i_block in range(bn):
                for j_block in range(bn):  // vectorized
                    C[bi*bn + i_block, bj*bn + j_block] +=
                        A[bi*bn + i_block, ki] * B[ki, bj*bn + j_block]
```

**Schedule primitives**:
- `tile(axis, factor)`: Split axis into (coarse, fine).
- `fuse(axis1, axis2)`: Merge two axes.
- `reorder(axes)`: Change loop order.
- `bind(axis, thread)`: Map axis to GPU thread/block.
- `unroll(axis)`: Unroll loop body.
- `vectorize(axis)`: SIMD vectorization.
- `cache_read/cache_write`: Introduce intermediate buffers (shared memory, L1).

#### 2.4.4 AutoTVM: Automated Tuning via Bayesian Optimization

**Problem**: The schedule space is exponentially large. Manual tuning is infeasible.

**AutoTVM Solution**: Use Bayesian optimization to explore promising schedules.

**Workflow**:
1. **Search Space Definition**: A template (schedule + tunable parameters).
2. **Sampling**: Generate candidate schedules (random, prior from previous runs).
3. **Measurement**: Compile and execute on target hardware, measure runtime.
4. **Model Update**: Train a cost model (XGBoost, neural network) on measurements.
5. **Acquisition**: Select next promising schedule to evaluate.
6. **Iteration**: Repeat until convergence or budget exhausted.

**AutoTVM Code**:
```python
import tvm
from tvm import te, auto_scheduler
from tvm.contrib import utils

# Define TE compute
def matmul_te(M, N, K):
    A = te.placeholder((M, K), name='A', dtype='float32')
    B = te.placeholder((K, N), name='B', dtype='float32')
    k = te.reduce_axis((0, K))
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name='C'
    )
    return [A, B, C]

# Tune for CPU target
target = tvm.target.Target("llvm")
task = auto_scheduler.create_task(matmul_te, args=(256, 256, 256), target=target)

tuner = auto_scheduler.TaskScheduler(
    [task],
    load_log_file=None,
    log_file="matmul_autotvm.json"
)
tuner.tune(n_trials=1000)  # Explore 1000 schedules

# Inspect best schedule
with auto_scheduler.ApplyHistoryBest("matmul_autotvm.json"):
    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.build(task.compute_dag, target=target)
```

#### 2.4.5 Ansor: Neural-Guided Scheduling (OSDI 2020)

**Ansor** is a successor to AutoTVM that uses a learned cost model to guide exploration more efficiently.

**Key innovations**:
1. **Learned Cost Model**: A Graph Neural Network (GNN) + XGBoost predicts schedule latency without expensive hardware execution.
2. **Evolutionary Search**: Mutates schedules in the schedule space, guided by the cost model.
3. **Hardware-Aware Tuning**: Captures device-specific characteristics (memory hierarchy, compute units).

**Cost Model Training**:
```
[Schedule] → [TE IR] → [Extracted Features] → [GNN/XGBoost] → [Predicted Latency]
```

**Speedup over AutoTVM**:
- Ansor finds better schedules with 5–10× fewer hardware measurements.
- Typical speedup: 2–5× on diverse workloads (ResNet, MobileNet, BERT).

**Ansor Workflow**:
```python
from tvm import auto_scheduler

# Build task
task = auto_scheduler.create_task(matmul_te, args=(1024, 1024, 1024), target=target)

# Ansor tuning
tuner = auto_scheduler.TaskScheduler(
    [task],
    load_log_file=None,
    log_file="ansor_results.json",
    tune_option=auto_scheduler.TuneOption(
        num_measure_trials=500,
        early_stopping=50,
        num_measures_per_round=10,
    ),
)

tuner.tune()
```

#### 2.4.6 Target-Specific Codegen

TVM generates target-specific code for each platform:

**CPU (x86_64 with AVX-512)**:
```python
target = tvm.target.Target("llvm -mcpu=skylake-avx512")
# TVM emits AVX-512 intrinsics (vector width = 512 bits / 32 bits = 16 floats)
```

**ARM NEON (Mobile/Edge)**:
```python
target = tvm.target.Target("llvm -mtriple=armv7-linux-gnueabihf -mattr=+neon")
# TVM uses NEON 128-bit vectors
```

**CUDA (NVIDIA GPU)**:
```python
target = tvm.target.Target("cuda -model=sm_80")  # RTX 3090
# TVM generates CUDA C++ / PTX assembly
```

#### 2.4.7 Apache TVM MetaSchedule

**MetaSchedule** is the newest TVM auto-tuning system (as of 2023–2024), integrating lessons from AutoTVM and Ansor.

**Features**:
1. **SpaceGenerator**: Expands the schedule space automatically (evolutionary mutation of existing good schedules).
2. **CostModel**: Trains a lightweight model on measured data.
3. **Adaptive Sampling**: Balances exploration vs. exploitation.

**Code**:
```python
from tvm import meta_scheduler

# Create tuning context
with meta_scheduler.TuneContext(
    mod=relay_module,
    target=target,
    work_dir="./results",
):
    cost_model = meta_scheduler.cost_model.XGBModel()
    tuner = meta_scheduler.TaskScheduler(
        tasks=...,
        cost_model=cost_model,
        log_dir="./results",
    )
    tuner.tune(num_trials=2000)
```

---

### 2.5 MLIR: Multi-Level Intermediate Representation

**MLIR** is a compiler infrastructure supporting multiple levels of abstraction, from high-level operations down to low-level hardware instructions.

#### 2.5.1 Dialect System

MLIR uses **dialects** to represent operations at different levels:

**High-Level Dialects**:
- **linalg**: Linear algebra (matmul, conv2d, etc.)
- **onnx**: ONNX operations

**Mid-Level Dialects**:
- **affine**: Affine loop structures and optimizations
- **scf**: Structured control flow (loops, conditionals)

**Low-Level Dialects**:
- **memref**: Memory references and transformations
- **vector**: Vector operations (SIMD)
- **gpu**: GPU-specific operations (blocks, threads)
- **llvm**: LLVM IR operations
- **spirv**: SPIR-V (Vulkan, OpenCL)

**Example: Linalg MatMul**:
```mlir
func.func @matmul(%A: memref<256x256xf32>,
                   %B: memref<256x256xf32>,
                   %C: memref<256x256xf32>) {
  linalg.matmul ins(%A, %B : memref<256x256xf32>, memref<256x256xf32>)
                 outs(%C : memref<256x256xf32>)
  return
}
```

#### 2.5.2 Lowering and Transformation Passes

MLIR compilation is a series of lowering passes, each converting high-level operations to lower-level ones.

**Example Lowering Chain**:
```
[linalg.matmul]
       ↓ Linalg → Affine
[affine.for loops with affine.load/store]
       ↓ Affine → SCF
[scf.for loops]
       ↓ SCF → LLVM
[llvm.call intrinsics]
       ↓ LLVM Codegen
[Assembly / Object Code]
```

**Transformation: Loop Tiling**:
```mlir
// Before:
affine.for %i = 0 to 256 {
  affine.for %j = 0 to 256 {
    // matmul body
  }
}

// After (tile size 16):
affine.for %ii = 0 to 256 step 16 {
  affine.for %jj = 0 to 256 step 16 {
    affine.for %i = %ii to min(%ii + 16, 256) {
      affine.for %j = %jj to min(%jj + 16, 256) {
        // matmul body
      }
    }
  }
}
```

#### 2.5.3 MLIR in IREE and OpenXLA

**IREE (Intermediate Representation Execution Engine)** uses MLIR as its core IR.

**IREE Compilation**:
```
[TOSA/Linalg IR]
       ↓ Vectorization
[Vector Dialect]
       ↓ Lowering to memref + loops
[Memref + SCF Dialect]
       ↓ LLVM Lowering
[LLVM Dialect]
       ↓ Target Codegen
[Machine Code / HAL Bytecode]
```

**OpenXLA** uses MLIR to lower HLO to target-specific code (LLVM, GPU).

---

### 2.6 torch.compile and TorchInductor

**torch.compile** (PyTorch 2.0+) enables automatic compilation of PyTorch models using a new compiler backend, TorchInductor.

#### 2.6.1 Dynamo Capture (PEP 523)

**Dynamo** is a graph capture mechanism that intercepts Python bytecode execution.

**Mechanism (via sys.settrace + bytecode introspection)**:
1. Python interpreter calls a registered trace function for each line executed.
2. Dynamo's trace function detects tensor operations and builds a graph.
3. When a guard condition is violated (e.g., shape changes), Dynamo falls back to eager execution.

**Example**:
```python
import torch

@torch.compile
def forward(x, y):
    z = x + y
    return z.relu()

x = torch.randn(10, requires_grad=True)
y = torch.randn(10, requires_grad=True)
output = forward(x, y)
```

**What Dynamo captures**:
```
FX Graph:
  add(x, y) → temp
  relu(temp) → output
```

**Guard conditions**:
- Input shape: x.shape == (10,)
- Input dtype: x.dtype == float32
- Input device: x.device == cuda:0

If any guard fails (e.g., different batch size), Dynamo recompiles.

#### 2.6.2 TorchInductor: Lowering to Triton / C++

**TorchInductor** is the default compiler backend for torch.compile.

**Lowering pipeline**:
```
[PyTorch FX Graph]
       ↓ Decompose high-level ops
[Lower-level ops (aten-level)]
       ↓ Loop & memory scheduling
[Grouped loops with buffers assigned]
       ↓ Triton / C++ codegen
[Triton Kernel / C++ Code]
       ↓ Compilation
[CUDA Kernel / Compiled C++]
```

**Inductor optimizations**:
1. **Vertical fusion**: Fuse consecutive element-wise operations.
2. **Horizontal fusion**: Combine independent operations for data reuse.
3. **Reduction fusion**: Fuse reductions with preceding operations.
4. **Layout optimization**: Choose NCHW vs. NHWC based on consumer ops.

**Example: Inductor on CPU**:
```python
import torch

# Enable Inductor backend
torch._dynamo.config.suppress_errors = True

@torch.compile(backend="inductor")
def model(x):
    return torch.relu(x + 1.0)

x = torch.randn(1000, 1000)
output = model(x)

# Inspect generated code (experimental)
import torch._inductor.config
torch._inductor.config.debug = True
model(x)  # Prints codegen'd C++ / Triton
```

**Generated C++ pseudocode**:
```cpp
void fused_kernel(float *x, float *output, int64_t numel) {
    #pragma omp parallel for
    for (int64_t i = 0; i < numel; ++i) {
        output[i] = std::max(0.f, x[i] + 1.f);
    }
}
```

#### 2.6.3 Inspection and Debugging

**torch.compile Debugging**:
```python
@torch.compile(backend="inductor", options={"trace.enabled": True})
def f(x):
    return x + 1

x = torch.randn(10)
f(x)

# Inspect FX graph
import torch._dynamo as dynamo
gm, _ = dynamo.export(f, x)
print(gm.graph)

# View IR lowering
torch._inductor.config.print_graph = True
f(x)
```

---

### 2.7 IREE: Intermediate Representation Execution Engine

**IREE** is a compiler and runtime designed for efficient inference on diverse hardware (CPU, GPU, edge devices).

#### 2.7.1 HAL Abstraction

**HAL (Hardware Abstraction Layer)** in IREE provides a unified interface to different hardware backends.

**HAL Components**:
- **Device**: Represents a physical hardware accelerator (CPU, GPU).
- **Buffer**: GPU/CPU memory allocation.
- **CommandBuffer**: Sequence of commands (kernel launch, memory copy).
- **Executable**: Compiled code module.

**HAL Targets**:
- **CPU**: x86, ARM, RISC-V
- **GPU**: Vulkan, Metal, CUDA, HIP
- **WebGPU**: Browser-based GPU
- **Hexagon**: Qualcomm DSP

#### 2.7.2 Compilation Pipeline

```
[TOSA / Linalg IR]
       ↓ Preprocessing (shape propagation, type conversion)
[Standardized IR]
       ↓ Sequence of MLIR lowering passes
[Bufferization → Memref → Loops → LLVM]
       ↓ Per-target backend
[LLVM IR / VM Bytecode / GPU Code]
       ↓ Linking
[HAL Module]
```

**Example: Compiling for Vulkan GPU**:
```python
from iree.compiler import compile_str

mlir_text = """
func.func @matmul(%A: tensor<256x256xf32>,
                   %B: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %C = linalg.matmul ins(%A, %B : tensor<256x256xf32>, tensor<256x256xf32>)
                     outs(... : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %C : tensor<256x256xf32>
}
"""

config = iree.compiler.Config()
config.target_backends = ["vulkan"]

vm_module = iree.compiler.compile_str(mlir_text, config=config)

# Execute on Vulkan device
device = iree.runtime.get_device("vulkan")
hal_module = iree.runtime.load_module(vm_module, device)
f = hal_module["matmul"]
result = f(A, B)
```

---

### 2.8 Hardware Mapping

Compilation adapts to specific hardware: CPU (AVX-512, AMX), Apple Silicon (Metal, NEON), Edge (ARM, TFLite).

#### 2.8.1 CPU: AVX-512, AMX, OpenMP

**AVX-512 (Intel Skylake+)**:
- Vector width: 512 bits = 16 float32s or 8 float64s per instruction.
- Operations: FMA (fused multiply-add), gather, scatter, mask operations.

**Inductor → AVX-512**:
```python
@torch.compile(backend="inductor")
def matmul(A, B):
    return torch.matmul(A, B)

# Target CPU with AVX-512
A = torch.randn(256, 256)
B = torch.randn(256, 256)
C = matmul(A, B)  # Uses AVX-512 instructions internally
```

**AMX (Advanced Matrix Extensions, Intel 4th Gen Xeon)**:
- Specialized tile matrix multiplication unit.
- Tiles: 16 × 8 = 128 bytes per tile (for fp32).
- Operations: TMUL (tile matrix multiply), TADD (tile add).

**TVM targeting AMX**:
```python
target = tvm.target.Target("llvm -mcpu=sapphire-rapids")  # 4th Gen Xeon
# TVM's scheduler detects AMX availability and generates tile operations
```

**OpenMP Parallelization**:
```python
# TVM-generated code (pseudo)
#pragma omp parallel for collapse(2)
for (int i = 0; i < M; i += TILE_M) {
    for (int j = 0; j < N; j += TILE_N) {
        // Thread-local matmul of tile
        for (int k = 0; k < K; ++k) {
            // Vectorized FMA
        }
    }
}
```

#### 2.8.2 Apple Silicon: Metal, NEON

**Metal (Apple's GPU API)**:
- Used on M1, M2, M3 (unified memory architecture).
- TensorFlow and PyTorch support Metal via native backends.

**Metal Shader**:
```metal
kernel void gemm(device const float *A [[buffer(0)]],
                 device const float *B [[buffer(1)]],
                 device float *C [[buffer(2)]],
                 uint2 gid [[thread_position_in_grid]]) {
    int i = gid.x, j = gid.y;
    float sum = 0.0;
    for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
    }
    C[i * N + j] = sum;
}
```

**NEON (ARM SIMD)**:
- Available on all modern ARM cores.
- 128-bit vectors: 4 float32s or 2 float64s per instruction.

**TVM NEON codegen**:
```python
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+neon")
```

#### 2.8.3 Edge: ARM NEON, TFLite Quantization

**TFLite (TensorFlow Lite) Compilation Pipeline**:
```
[PyTorch / TensorFlow Model]
       ↓ Convert to TFLite FlatBuffer
[FlatBuffer (ARM/TFLite ops)]
       ↓ Quantization-aware transforms
[Quantized Model]
       ↓ Delegate to ARM NEON / Hexagon / GPU]
       ↓ Mobile Runtime
[Results on Phone / IoT]
```

**Quantization for Edge**:
```python
import tensorflow as tf

def representative_dataset():
    for _ in range(100):
        yield [tf.random.normal((1, 224, 224, 3))]

converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

quantized_tflite_model = converter.convert()
```

---

## 3. HARDWARE MAPPING — Detailed Treatment

### 3.1 CPU Architectures: x86-64, AVX-512, AMX

**x86-64 Feature Levels**:

1. **Baseline (SSE2)**:
   - 128-bit vectors: 4 float32s per instruction.
   - Operations: multiply, add, shuffle, horizontal reductions.

2. **AVX (Advanced Vector Extensions)**:
   - 256-bit vectors: 8 float32s per instruction.
   - Introduced in Intel Sandy Bridge (2011).

3. **AVX-512**:
   - 512-bit vectors: 16 float32s or 8 float64s.
   - Intel Skylake, Cascade Lake, Cooper Lake, etc.
   - Includes mask operations (conditional execution).

4. **AMX (Advanced Matrix Extensions)**:
   - Tile matrix operations (Intel 4th Gen Xeon).
   - Tiles: 16 × 8 = 128 bytes (for fp32 tiles, 4 elements × 8 bytes each = 256 bytes in practice).
   - Operations: TMULPS (tile multiply), TADDPS (tile add), TSTORE (write to memory).

**Compiler targeting AVX-512**:

```python
import tvm
target = tvm.target.Target("llvm -mcpu=skylake-avx512")

# TVM generates intrinsics like:
# _mm512_fmadd_ps (fused multiply-add on 16 float32s)
# _mm512_loadu_ps (load 16 aligned float32s)
```

**Compiler targeting AMX**:

```python
target = tvm.target.Target("llvm -mcpu=sapphire-rapids")
# TVM inserts _tile_mulf (tile matrix multiply floating point)
```

### 3.2 ARM: NEON, SVE, SVE2

**NEON (All ARMv7+, ARMv8)**:
- 128-bit vectors: 4 float32s or 2 float64s.
- Available on mobile phones, embedded systems, AWS Graviton.

**Scalable Vector Extension (SVE)**:
- Vector length dynamically determined at runtime.
- Support for vector lengths 128, 256, 512 bits (and beyond).
- Available on AWS Graviton3, Fujitsu A64FX.

**Compiler targeting NEON**:

```python
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+neon")
# Generates NEON intrinsics: vmulq_f32, vaddq_f32
```

**Compiler targeting SVE**:

```python
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+sve")
# Generates SVE intrinsics: svmul_f32_x, svadd_f32_x
```

### 3.3 Apple Silicon: M1, M2, M3 (Metal, NEON)

**Apple's Unified Memory Architecture**:
- GPU and CPU share the same memory pool (no PCIe transfer latency).
- Asymmetric core design: performance cores + efficiency cores.

**Compilation targets**:
1. **CPU (performance cores)**: NEON + ARM SVE (limited).
2. **GPU (Metal)**: Custom GPU with tiling shaders.

**Metal Shader Example** (PyTorch Metal backend):

```metal
kernel void matmul(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x, j = gid.y;
    if (i < M && j < N) {
        float sum = 0.0;
        for (uint k = 0; k < K; ++k) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}
```

**Deployment via PyTorch**:
```python
import torch

model = torch.nn.Linear(1024, 1024)
model = model.to("mps")  # Move to Metal Performance Shaders

input_data = torch.randn(1, 1024, device="mps")
output = model(input_data)
```

### 3.4 GPU: NVIDIA CUDA, AMD HIP, Intel GPU

**NVIDIA CUDA Programming Model**:
- Grids of thread blocks.
- Shared memory (L1 cache) per thread block.
- Global memory (DRAM) accessible by all threads.

**Compilation from TVM → CUDA**:

```python
target = tvm.target.Target("cuda -model=sm_80")  # RTX 3090

# TVM generates:
# - Thread block grid: blockDim.x, blockDim.y, gridDim.x, gridDim.y
# - Shared memory allocation: __shared__ float smem[...]
# - Kernel launch: <<<gridDim, blockDim>>> kernel(args)
```

**AMD HIP** (compatible with CUDA):
```python
target = tvm.target.Target("rocm")
# Semantically similar to CUDA, different runtime API
```

**Intel Arc GPU**:
```python
target = tvm.target.Target("opencl")
# Uses OpenCL for portability
```

### 3.5 Edge & Embedded: TensorFlow Lite, NNAPI, MCU

**TensorFlow Lite (TFLite)**:
- Quantized inference (int8, fp16).
- Delegates to optimized backends: ARM NEON, Hexagon (Qualcomm DSP), NNAPI.

**Deployment on mobile**:
```python
import tensorflow as tf

# Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
quantized_tflite = converter.convert()

# Save
with open("model.tflite", "wb") as f:
    f.write(quantized_tflite)
```

**NNAPI** (Android Neural Networks API):
- Hardware-accelerated inference on Android devices.
- Delegates to CPU, GPU, or specialized accelerators (Snapdragon Hexagon, Tensor Processing Unit).

**MCU Deployment** (TensorFlow Lite Micro):
- For microcontrollers (ARM Cortex-M4, RISC-V).
- Typical model size: <1 MB, latency <100 ms.

```python
# C++ generated code (via xxd or embedding tools)
const unsigned char model_data[] = {...};  // Quantized model binary
tflite::Interpreter interpreter(model_data);
interpreter.Invoke();
```

---

## 4. IMPLEMENTATION DEEP DIVE

### 4.1 TVM End-to-End Compilation Pipeline for CPU

**Complete workflow**: ONNX import → Relay optimization → Ansor tuning → codegen → execution.

```python
import onnx
import tvm
from tvm import relay, auto_scheduler
from tvm.relay import transform
import tvm.contrib.utils as utils

# ========== STEP 1: Load ONNX Model ==========
onnx_model_path = "resnet50.onnx"
onnx_model = onnx.load(onnx_model_path)

# Verify and prepare
onnx.checker.check_model(onnx_model)

# ========== STEP 2: Convert ONNX to Relay ==========
input_name = "data"
shape_dict = {input_name: (1, 3, 224, 224)}  # (batch, channels, height, width)
dtype_dict = {input_name: "float32"}

# Import ONNX into Relay
relay_module, relay_params = relay.frontend.from_onnx(
    onnx_model,
    shape_dict=shape_dict,
    dtype_dict=dtype_dict
)

print("=== Original Relay Module ===")
print(relay_module)

# ========== STEP 3: Optimize Relay ==========
# Apply high-level optimizations
passes = [
    transform.FoldConstant(),
    transform.FoldScaleAxis(),
    transform.SimplifyExpr(),
    transform.DeadCodeElimination(),
    transform.CommonSubexprElimElim(),
    transform.ConvertLayout({
        "nn.conv2d": ["NCHW", "default"],
    }),
    transform.FuseOps(fuse_depth=10),  # Fuse operations
]

# Execute passes sequentially
with tvm.transform.PassContext(opt_level=3):
    relay_module = transform.Sequential(passes)(relay_module)

print("\n=== Optimized Relay Module ===")
print(relay_module)

# ========== STEP 4: Prepare for Compilation ==========
target = tvm.target.Target("llvm -mcpu=skylake-avx512")
target_host = tvm.target.Target("llvm -mcpu=skylake-avx512")

# Build task for auto-scheduling
print("\n=== Creating Auto-Tuning Task ===")

# For production, use Ansor (auto_scheduler)
# For demo, we'll use a simple schedule

# ========== STEP 5: Compile without Auto-tuning (Direct Compilation) ==========
print("\n=== Compiling with Direct Strategy ===")

with tvm.transform.PassContext(opt_level=3):
    # Create an executable module
    module = relay.build(relay_module, target=target, params=relay_params)

# ========== STEP 6: Generate C/C++ Code Inspection ==========
print("\n=== Generated LLVM IR (first 1000 lines) ===")

# Get the LLVM code (by re-lowering)
from tvm import lowered_build

# For inspection, we can build and extract IR
try:
    # The module contains compiled code; extraction depends on TVM version
    print("Module compiled successfully.")
    print(f"Exported function: {module.exported_symbols}")
except Exception as e:
    print(f"Code inspection: {e}")

# ========== STEP 7: Prepare Runtime Inputs ==========
import numpy as np

# Create dummy input
input_data = np.random.randn(1, 3, 224, 224).astype("float32")

# Create runtime environment
dev = tvm.device(str(target), 0)
runtime_module = tvm.contrib.graph_executor.GraphModule(module["default"](dev))

# ========== STEP 8: Bind Inputs and Execute ==========
print("\n=== Running Inference ===")
runtime_module.set_input(input_name, input_data)

import time

# Warm up
for _ in range(5):
    runtime_module.run()

# Benchmark
num_runs = 100
start = time.time()
for _ in range(num_runs):
    runtime_module.run()
end = time.time()

avg_latency_ms = (end - start) / num_runs * 1000
print(f"Average latency: {avg_latency_ms:.3f} ms over {num_runs} runs")

# ========== STEP 9: Extract Output ==========
output = runtime_module.get_output(0)
output_np = output.numpy()
print(f"\nOutput shape: {output_np.shape}")
print(f"Output dtype: {output_np.dtype}")
print(f"First 10 values: {output_np.flatten()[:10]}")

# ========== STEP 10: Optional - Ansor Auto-tuning for Production ==========
print("\n=== Auto-tuning with Ansor (Optional for Production) ===")

# Define TE-based task for auto-scheduler
def conv2d_task(M, N, K, C_in, C_out, kernel_h, kernel_w):
    """Define a Conv2D computation in TE for auto-tuning."""
    from tvm import te

    input_h, input_w = 224, 224
    pad_h, pad_w = 1, 1
    stride_h, stride_w = 1, 1

    # Output spatial dimensions
    output_h = (input_h + 2*pad_h - kernel_h) // stride_h + 1
    output_w = (input_w + 2*pad_w - kernel_w) // stride_w + 1

    # Input
    A = te.placeholder((1, C_in, input_h, input_w), name="A", dtype="float32")
    # Weight
    W = te.placeholder((C_out, C_in, kernel_h, kernel_w), name="W", dtype="float32")
    # Bias
    B = te.placeholder((C_out,), name="B", dtype="float32")

    # Reduction axes
    rc = te.reduce_axis((0, C_in), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")

    # Compute Conv2D
    Output = te.compute(
        (1, C_out, output_h, output_w),
        lambda n, c, h, w: te.sum(
            A[n, rc, h*stride_h + ry, w*stride_w + rx] * W[c, rc, ry, rx],
            axis=[rc, ry, rx]
        ),
        name="Output"
    )

    return [A, W, Output]

# Create tuning task
from tvm import auto_scheduler

print("Ansor tuning requires hardware and is time-consuming.")
print("Typical workflow:")
print("""
task = auto_scheduler.create_task(conv2d_task, args=(...), target=target)
tuner = auto_scheduler.TaskScheduler([task], load_log_file=None, log_file='conv2d.json')
tuner.tune(n_trials=2000)

# Then load best schedule and recompile
with auto_scheduler.ApplyHistoryBest('conv2d.json'):
    with tvm.transform.PassContext(opt_level=3):
        module_tuned = relay.build(relay_module, target=target, params=relay_params)
""")

print("\n=== TVM End-to-End Compilation Complete ===")
```

**Key observations**:

1. **Relay optimization passes** significantly reduce graph size and reduce memory pressure.
2. **Target specification** (AVX-512) enables automatic SIMD code generation.
3. **Auto-tuning** (Ansor) is essential for production: it explores millions of schedules to find the fastest.
4. **Graph execution** via GraphExecutor provides efficient batched inference.

### 4.2 torch.compile Internals: Inspecting Generated Code

```python
import torch
import torch._dynamo as dynamo
import torch._inductor.config as inductorconfig

# ========== STEP 1: Enable Debug Mode ==========
# Captures graph and prints IR lowering
inductorconfig.debug = True
inductorconfig.trace.enabled = True

# ========== STEP 2: Define Model ==========
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(256, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.bn1 = torch.nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        return x

model = SimpleModel().eval()

# ========== STEP 3: Export to FX via Dynamo ==========
# Dynamo captures the control flow and operations
example_input = torch.randn(32, 256)

# Manual export using _dynamo.export (experimental)
try:
    gm, spec = dynamo.export(model, example_input, aten_graph=False)
    print("=== PyTorch FX Graph ===")
    print(gm.graph)
    print(gm.code)
except Exception as e:
    print(f"Export failed: {e}")
    print("torch.compile might use internal APIs.")

# ========== STEP 4: Compile with torch.compile ==========
compiled_model = torch.compile(model, backend="inductor", options={
    "triton.cudagraphs": False,
})

# ========== STEP 5: Run Compiled Model ==========
print("\n=== Running Compiled Model ===")
with torch.no_grad():
    output = compiled_model(example_input)
    print(f"Output shape: {output.shape}")

# ========== STEP 6: Inspect Generated Code (Experimental) ==========
print("\n=== Inductor Generated Code ===")
# TorchInductor generates C++ or Triton code
# Debug mode may print to stderr or to files

# Alternative: Check generated artifacts
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    # Configure Inductor to save generated code
    inductorconfig.save_path = tmpdir

    compiled_model_debug = torch.compile(model, backend="inductor", options={})
    output = compiled_model_debug(example_input)

    # List generated files
    generated_files = os.listdir(tmpdir) if os.path.exists(tmpdir) else []
    print(f"Generated files: {generated_files}")

# ========== STEP 7: Benchmark ==========
print("\n=== Benchmarking ===")
import time

# Warm up
for _ in range(10):
    with torch.no_grad():
        _ = compiled_model(example_input)

# Time eager execution
torch._dynamo.reset()  # Clear compiled cache
model_eager = SimpleModel().eval()
num_runs = 100
start = time.time()
for _ in range(num_runs):
    with torch.no_grad():
        _ = model_eager(example_input)
eager_time = time.time() - start

# Time compiled execution
start = time.time()
for _ in range(num_runs):
    with torch.no_grad():
        _ = compiled_model(example_input)
compiled_time = time.time() - start

print(f"Eager execution: {eager_time / num_runs * 1000:.3f} ms/iter")
print(f"Compiled execution: {compiled_time / num_runs * 1000:.3f} ms/iter")
print(f"Speedup: {eager_time / compiled_time:.2f}x")

# ========== STEP 8: Fusion Analysis ==========
print("\n=== Observing Fusion ===")
print("""
Without torch.compile:
  Linear(input) → (output1)  [memory bandwidth: load input, store output1]
  ReLU(output1) → (output2)  [memory bandwidth: load output1, store output2]
  BatchNorm1d(output2) → (output3)  [memory bandwidth: load output2, store output3]
  Total: 6 memory rounds

With torch.compile (fusion):
  Linear(input) → (intermediate1, no write to memory)
  ReLU(intermediate1) → (intermediate2, no write to memory)
  BatchNorm1d(intermediate2) → (output, single write)
  Total: 2 memory rounds (50% reduction)
""")
```

### 4.3 MLIR Lowering Example: From linalg to LLVM

```python
import mlir
from mlir import ir, passes
from mlir.dialects import linalg, memref, scf, vector, arith, func

# ========== STEP 1: Construct Linalg MatMul in MLIR Text ==========
mlir_text = """
#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @matmul(%A: memref<256x256xf32>,
                   %B: memref<256x256xf32>,
                   %C: memref<256x256xf32>) {
  linalg.matmul ins(%A, %B : memref<256x256xf32>, memref<256x256xf32>)
                 outs(%C : memref<256x256xf32>)
  return
}
"""

# ========== STEP 2: Parse MLIR ==========
context = ir.Context()
module_str = mlir_text
module = ir.Module.parse(module_str, context=context)

print("=== Original linalg.matmul ===")
print(module)

# ========== STEP 3: Apply Lowering Passes ==========
# Sequence of MLIR passes to lower from linalg → affine → scf → llvm

# Linalg Lowering: Convert linalg ops to affine loops + memory ops
pm = passes.PassManager(context)
pm.add_pass(passes.ConvertLinalgToLoopsPass())
pm.run(module)

print("\n=== After Linalg → Affine Conversion ===")
print(module)

# ========== STEP 4: Vectorization (Optional) ==========
pm = passes.PassManager(context)
pm.add_pass(passes.VectorizeMemrefPass())
pm.run(module)

print("\n=== After Vectorization ===")
print(module)

# ========== STEP 5: Affine Loop Optimization ==========
pm = passes.PassManager(context)
# Tile loops: 16x16 tiles for better cache locality
pm.add_pass(passes.AffineLoopTilePass(tileSize=16))
pm.run(module)

print("\n=== After Loop Tiling ===")
print(module)

# ========== STEP 6: Lower to SCF (Structured Control Flow) ==========
pm = passes.PassManager(context)
pm.add_pass(passes.ConvertAffineToStdPass())
pm.run(module)

print("\n=== After Affine → SCF Conversion ===")
print(module)

# ========== STEP 7: Lower to LLVM Dialect ==========
pm = passes.PassManager(context)
pm.add_pass(passes.ConvertSCFToStdPass())
pm.add_pass(passes.ConvertStdToLLVMPass())
pm.run(module)

print("\n=== After SCF → LLVM Conversion ===")
print(module)

# ========== STEP 8: Inspect Generated LLVM IR ==========
print("\n=== Final LLVM IR Module ===")
print(module)

# ========== STEP 9: Serialize to LLVM IR ==========
# (MLIR LLVM dialect can be lowered to actual LLVM IR)
# This would typically be done via mlir-translate tool:
# $ mlir-translate -mlir-to-llvmir module.mlir -o module.ll

print("\n=== Typical Loop Structure (After All Lowerings) ===")
print("""
for i in 0 to 256 step 16:        // Tile I
  for j in 0 to 256 step 16:      // Tile J
    C[i:i+16, j:j+16] = 0
    for k in 0 to 256:            // K (full)
      C[i:i+16, j:j+16] += A[i:i+16, k] @ B[k, j:j+16]
""")
```

---

## 5. KEY PAPERS AND REFERENCES

### Primary References

1. **TVM: An Automated End-to-End Optimizing Compiler for Deep Learning** (Chen et al., OSDI 2018)
   - **URL**: https://arxiv.org/abs/1802.04799
   - **Contribution**: Introduced the TVM stack (Relay + TE + AutoTVM).
   - **Key insight**: Automated scheduling via Bayesian optimization outperforms manual tuning.
   - **Impact**: Now widely deployed in production (PyTorch, TensorFlow, MXNet backends).

2. **Ansor: Generating High-Performance Tensor Programs for Deep Learning** (Zheng et al., OSDI 2020)
   - **URL**: https://arxiv.org/abs/2006.06762
   - **Contribution**: Introduced learned cost models + evolutionary search for schedule discovery.
   - **Key insight**: GNN-based cost prediction is vastly faster than hardware measurements.
   - **Results**: 2–5× speedup over AutoTVM on diverse workloads.

3. **MLIR: Compiler Infrastructure for the End of Moore's Law** (Lattner et al., CGO 2021)
   - **URL**: https://arxiv.org/abs/2002.11054
   - **Contribution**: Multi-level IR design, dialect system, lowering framework.
   - **Key insight**: Retargetable IR enables compiler reuse across AI and traditional domains.
   - **Impact**: Underpins OpenXLA, IREE, PyTorch Triton, Modular Mojo.

4. **PyTorch 2.0: Beyond Compiler Limitations with _Higher-Level_ Compilation** (Ansel et al., ASPLOS 2024)
   - **URL**: https://arxiv.org/abs/2311.08782
   - **Contribution**: torch.compile, Dynamo capture, TorchInductor.
   - **Key insight**: Python bytecode introspection + functional graph extraction enables aggressive JIT compilation.
   - **Results**: 1.1–1.8× speedup on real workloads without user intervention.

5. **IREE: Intermediate Representation Execution Engine** (Google IREE Team, 2021–2024)
   - **URL**: https://github.com/openxla/iree
   - **Contribution**: HAL abstraction, modular compilation pipeline, edge deployment framework.
   - **Key insight**: Bytecode-based VM + lazy compilation enables portable deployment.

6. **Triton: An Intermediate Language and Compiler for Tiled Neural Network Computation** (Tillet, Zheng, Jia, MAPL 2019)
   - **URL**: https://arxiv.org/abs/1905.13021
   - **Contribution**: Higher-level abstraction for GPU kernel writing, auto-tuning for kernels.
   - **Key insight**: Enables domain experts to write performant GPU code without CUDA expertise.
   - **Impact**: Powers PyTorch Inductor's GPU codegen.

7. **Attention Is All You Need** (Vaswani et al., NeurIPS 2017)
   - **URL**: https://arxiv.org/abs/1706.03762
   - **Context**: Defines transformer architecture; critical for understanding attention fusion in modern compilers.

8. **Quantization and Training of Neural Networks for Efficient Integer Arithmetic** (Jacob et al., CVPR 2018)
   - **URL**: https://arxiv.org/abs/1712.05877
   - **Contribution**: QAT (quantization-aware training) framework.
   - **Impact**: Enables edge deployment with 8-bit or 4-bit operations.

---

## 6. ADVANCED TOPICS IN GRAPH OPTIMIZATION

### 6.1 Quantization-Aware Compilation

**Quantization** reduces precision from fp32 to int8, fp16, or int4, cutting memory bandwidth and latency.

**Integration into Compilation Stack**:

```python
import torch
from torch.quantization import quantize_dynamic, convert

# Step 1: Insert quantization stubs
model = torch.nn.Sequential(
    torch.nn.Linear(256, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
)

# Step 2: Quantization-aware training (QAT) or post-training quantization (PTQ)
# For simplicity, post-training:
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Step 3: Compile quantized model
# torch.compile handles quantization ops natively
compiled_quantized = torch.compile(quantized_model, backend="inductor")

# Step 4: Execute with reduced precision
input_data = torch.randn(1, 256)
output = compiled_quantized(input_data)

print(f"Quantized model output: {output.shape}")
print(f"Model size: {sum(p.numel() for p in quantized_model.parameters() if p.dtype == torch.qint8) * 1 // 1024 // 1024} MB (vs. 4 MB fp32)")
```

**Compiler considerations**:
- **Quantization type mapping**: int8 matmul → vpmaddubsw (AVX2) or sdot (NEON).
- **Dequant/Requant elimination**: Fuse quantization ops with preceding/following operations.
- **Mixed precision**: Some ops (e.g., reductions) stay in fp32 for numerical stability.

### 6.2 Dynamic Shapes and Graph Recompilation

Many models have dynamic batch sizes or sequence lengths (e.g., BERT with variable input length).

**Compilation challenge**: Schedules optimized for shape (1, 512) may not work for (32, 512).

**Approaches**:

1. **Explicit recompilation**:
```python
@torch.compile
def forward(x):
    return model(x)

# First call: batch_size=1
output1 = forward(torch.randn(1, 512))

# Second call: batch_size=32 → recompilation triggered
output2 = forward(torch.randn(32, 512))

# Cost: 2 compilations
```

2. **Shape-polymorphic scheduling** (advanced TVM):
```python
# Define schedules parameterized by shape
M, N, K = tvm.var("M"), tvm.var("N"), tvm.var("K")

# Schedule adjusts tile size based on dynamic M, N, K
if M.value >= 256:
    tile_m = 64
else:
    tile_m = 32
```

3. **Bytecode + lazy JIT** (IREE approach):
   - Compile once to bytecode (shape-agnostic).
   - At runtime, micro-JIT the hot kernels for the specific shape.

### 6.3 Memory Efficient Inference: KV-Cache Optimization

For autoregressive transformers (GPT, LLaMA), the KV-cache grows linearly with sequence length.

**Compilation strategy**:

```python
# Without optimization: naive implementation
def transformer_inference(tokens, kv_cache):
    for pos in range(len(tokens)):
        hidden = embed(tokens[pos])
        for layer in layers:
            q = hidden @ W_q
            # Read entire KV-cache (large!)
            k = kv_cache[layer][:pos]
            v = kv_cache[layer][layer, :pos]
            attn = softmax(q @ k.T / sqrt(d))
            output = attn @ v
            hidden = output
        kv_cache[layer][pos] = (hidden @ W_k, hidden @ W_v)
    return hidden

# With compiler optimization:
# 1. Fuse attention computation: Q @ KV^T in single kernel
# 2. Keep KV-cache in fast memory (GPU HBM, CPU cache)
# 3. Auto-tune memory access patterns
# 4. Quantize KV-cache to fp8 (if acceptable)
```

**Compiler support** (via hand-written kernels or auto-tuning):
- **FlashAttention** integration: IREE, PyTorch can use optimized attention kernels.
- **KV-cache layouts**: Compiler chooses between flat layout, striped layout, or sparse formats.

### 6.4 Model Parallelism and Distributed Compilation

For models larger than single-device memory, compilation strategies differ:

**Tensor Parallelism** (split weight across devices):
```python
# Original: C = A @ B (A: 1x1024, B: 1024x1024)
# Partition B across 4 GPUs: B = [B0, B1, B2, B3] (1024x256 each)

# Compiled kernel:
def parallel_matmul(A, B_partition):
    # Local computation
    partial_C = A @ B_partition
    # AllGather to combine
    C = allgather(partial_C)
    return C
```

**Pipeline Parallelism** (split layers across devices):
```python
# Device 0: layers[0:10]
# Device 1: layers[10:20]
# Compiler inserts Send/Receive primitives at boundaries
```

---

## 7. PERFORMANCE ANALYSIS AND PROFILING

### 7.1 Roofline Model for Compiled Code

The **roofline model** predicts achievable performance given hardware and code arithmetic intensity.

**Peak throughput** (roof) is determined by:
- Peak compute capacity: TFLOPS
- Peak memory bandwidth: GB/s

**Arithmetic intensity** (I) = FLOPs / bytes accessed

**Attainable GFLOPs** = min(peak_flops, peak_bw × I)

**Example**:
```
CPU: 100 GFLOPS peak, 50 GB/s bandwidth
Matrix multiply (I = 2): min(100, 50*2) = 100 GFLOPS (compute-bound)
Element-wise add (I = 1/8): min(100, 50*1/8) = 6.25 GFLOPS (bandwidth-bound)

After fusion, element-wise add's I improves via kernel fusion.
```

### 7.2 Compilation Profiling and Autotuning Metrics

**Metrics to track**:

1. **Compilation time**: Wall-clock time from model import to executable.
2. **Tuning time**: Hours spent in AutoTVM/Ansor.
3. **Code size**: Compiled binary size.
4. **Latency**: Inference time (ms).
5. **Throughput**: Samples/sec (for batched inference).
6. **Memory usage**: Peak GPU/CPU memory during inference.

```python
import time
import torch
from torch.utils.benchmark import Timer

model = ...

# Measure compilation time
compile_start = time.time()
compiled_model = torch.compile(model, backend="inductor")
compile_time = time.time() - compile_start

# Warm-up
for _ in range(10):
    with torch.no_grad():
        _ = compiled_model(torch.randn(1, 3, 224, 224))

# Measure inference latency
timer = Timer(
    "compiled_model(x)",
    setup="from __main__ import compiled_model, x",
)
result = timer.timeit(number=100)

print(f"Compilation time: {compile_time:.2f}s")
print(f"Inference latency: {result.mean * 1000:.3f} ms")
```

---

## 8. LIMITATIONS AND OPEN CHALLENGES

### 8.1 Dynamic Control Flow

**Problem**: Compilers assume static graphs. Models with data-dependent loops (e.g., beam search) are difficult to compile.

```python
# Incompatible with traditional compilers:
def beam_search(logits):
    for step in range(max_length):
        # Number of iterations depends on data
        if early_stopping_condition(state):
            break
        predictions = sample_topk(logits)
    return predictions
```

**Solutions**:
- **Fallback to eager**: Use fallback to eager execution for dynamic regions.
- **Bytecode + VM**: IREE's approach handles dynamic control via bytecode interpreter.
- **Ahead-of-time compilation** of all possible paths (expensive).

### 8.2 Model-Specific Optimization Opportunities

**Generic compilers miss domain-specific optimizations**:
- Attention patterns (token budget, sparse attention).
- Activation functions (custom GELU variants).
- Quantization schemes (non-uniform, group-wise).

**Research direction**: Compiler plug-ins that let model authors specify custom optimizations.

### 8.3 Hardware Evolution

**New hardware requires new compiler backends**:
- Intel Gaudi, Google TPU 4, ARM Scalable Processor.
- Vendor-specific libraries evolve faster than compiler support.

**Mitigation**: Abstract HAL + dispatch to vendor libraries for unknown ops.

---

## 9. FUTURE DIRECTIONS

### 9.1 Machine Learning for Compiler Optimization

**Trend**: Use ML to improve compiler decisions.

- **Cost models**: Graph neural networks predict op latency better than heuristics.
- **Scheduling**: Reinforcement learning explores schedule space more efficiently than Bayesian optimization.
- **Layout selection**: Graph autoencoders learn which layout (NCHW vs. NHWC) minimizes total latency.

**Challenge**: Compiler + ML = complex systems; hard to debug.

### 9.2 Unified Compilation Framework

**Goal**: Single compiler backend for all AI frameworks.

**Status**: MLIR + OpenXLA + IREE moving toward this, but fragmentation remains.

**Open question**: Will PyTorch, TensorFlow, JAX converge on a single IR?

### 9.3 Speculative Compilation

**Idea**: Compile multiple versions of a function, speculatively execute the expected one, roll back if guard fails.

**Benefit**: Avoid recompilation overhead for shapes that are likely but not guaranteed.

---

## 10. APPENDIX: COMPILATION STACK COMPARISON TABLE

| Component | PyTorch FX | ONNX | XLA/HLO | TVM Relay | MLIR | torch.compile |
|-----------|-----------|------|---------|-----------|------|---------------|
| **Frontend** | Python code | Model format | TF/JAX | ONNX/TVM AST | Text/API | Python code |
| **IR Level** | Functional | Graph | High-level ops | High-level | Multi-level | Functional |
| **Auto-tuning** | No | No | Limited | Yes (Ansor) | Via plugins | Limited |
| **Hardware Support** | Limited | Runtime-dependent | XLA targets | CPU/GPU/Mobile | Extensible | CUDA/CPU |
| **Production Use** | Research | Widespread | Google TPU | Production (AMD, Intel) | Emerging | PyTorch 2.0+ |
| **Ease of Use** | Moderate | High | High | Moderate | Hard | Very High |

---

## SUMMARY

This module covered the complete ML compilation stack:

1. **Graphs** (DAG, PyTorch FX, ONNX, XLA HLO, TVM Relay) capture models as dataflow.
2. **Optimizations** (fusion, constant folding, DCE, CSE, memory planning) reduce computation and bandwidth.
3. **Code generation** (MLIR, TVM TE, TorchInductor) lowers to hardware-specific machine code.
4. **Auto-tuning** (AutoTVM, Ansor, MetaSchedule) explores schedules efficiently.
5. **Deployment** (IREE, TorchInductor, XLA) enables efficient inference across devices.

**Key takeaway**: Modern ML compilers are as complex as traditional optimizing compilers (GCC, LLVM), with added challenges of tensor operations, heterogeneous hardware, and dynamic shapes. Research is ongoing in ML-guided optimization, unified frameworks, and speculative compilation.

---

## RECOMMENDED READING ORDER

1. Chen et al. (TVM) for end-to-end overview.
2. Zheng et al. (Ansor) for auto-tuning insights.
3. Lattner et al. (MLIR) for compiler infrastructure.
4. Ansel et al. (torch.compile) for practical PyTorch integration.
5. IREE docs for edge deployment.
6. Triton for GPU kernel authoring.

---

**Last updated**: 2025-03 | **Target audience**: PhD students, ML systems researchers, compiler engineers.


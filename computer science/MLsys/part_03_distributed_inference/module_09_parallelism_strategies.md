# MODULE 9: Parallelism Strategies for Inference—Complete Taxonomy

## SYSTEMS OVERVIEW

### The Inference Parallelism Problem Space

Deploying large language models for inference presents a fundamental tension between model capacity and computational resources. A single GPU with 40GB-80GB memory can hold at most 30-60B parameter models unquantized. Yet production deployments routinely require models with 70B, 175B (GPT-3), 405B (Claude 3), and beyond. The inference problem differs critically from training: we have fixed computational graphs per token generation, no weight updates, and strict latency requirements (typically 50-100ms per token for interactive systems).

When we say "single-device fails," we mean three distinct failure modes:

1. **Memory Capacity Failure**: A 175B parameter model at FP16 requires 350GB of memory for parameters alone. With KV-cache (batch size 64, sequence length 4096, 128 attention heads), we add 4 * 64 * 4096 * 128 * 2 ≈ 67GB for GPT-3 architecture. This exceeds 8× the typical GPU memory budget.

2. **Throughput Failure**: Even if we could fit the model via aggressive quantization (INT8: 175GB → 87.5GB), inference on a single A100 achieves ~600 tokens/second for a 175B model. Production systems demand 5,000-50,000 tokens/second. A single device cannot achieve necessary throughput.

3. **Arithmetic-Precision Failure**: Serving diverse workloads (medical reasoning, code generation, math) on a single device forces us to choose between: (a) low precision that breaks correctness for sensitive tasks, (b) mixed-precision that incurs synchronization overhead, or (c) full precision that exhausts memory immediately.

The parallelism taxonomy addresses these failures orthogonally:

- **Tensor Parallelism (TP)**: Split individual matrix multiplications across devices via row/column partitioning. Increases per-device arithmetic density.
- **Pipeline Parallelism (PP)**: Split transformer layers across devices. Each device processes a different layer; different samples process different layers simultaneously.
- **Sequence Parallelism (SP)**: Split attention computation along sequence dimension using ring patterns.
- **Expert Parallelism (EP)**: For mixture-of-experts models, distribute expert layers across devices.
- **Data Parallelism (DP)**: For batch inference, replicate the model on each device and serve different samples (conflicts with single-model-instance designs).

Production systems combine these—TP+PP+SP+EP—yielding communication complexity that dominates runtime.

---

## THEORETICAL FOUNDATION

### Memory Analysis: Why Models Don't Fit

Consider a transformer layer with hidden dimension `d_h`, attention heads `n_h`, and feedforward dimension `d_ff = 4*d_h`:

**Per-layer parameters (FP16, 2 bytes each)**:
- Attention: Q, K, V projections (3 * d_h^2), output projection (d_h^2) = 4*d_h^2
- Attention biases: 4*d_h
- LayerNorm: 2*d_h
- Feed-forward: (d_h * d_ff) + (d_ff * d_h) = 2*d_h*4*d_h = 8*d_h^2
- FF biases: d_ff + d_h ≈ 4*d_h
- LayerNorm params: 2*d_h

**Total per layer**: ~12*d_h^2 + 10*d_h bytes

For GPT-3 (d_h=12288, 96 layers):
- Parameters: 12 * (12288)^2 * 96 * 2 ≈ 350 GB
- KV-cache at batch=64, seq=4096, n_h=96: 2 * 96 layers * 64 batch * 4096 seq * 128 hidden/head * 2 bytes ≈ 100 GB
- Activations during inference: ~20 GB
- **Total**: ~470 GB on a single device

A single A100 (80GB) fits ~15% of this. An A100 cluster of 8 devices fits everything, but now computation and communication must be perfectly orchestrated.

### Token Generation and Latency Analysis

LLM inference uses **autoregressive decoding**: each token generation requires one forward pass through the entire model. For a sequence of length T, we perform T forward passes.

**Per-token latency breakdown** (without parallelism):
- Compute time: T_comp = (2 * params) / (compute_per_second)
  - Each parameter is multiplied by an activation (~2 operations)
  - A100: 312 TFLOPs FP16 peak
  - GPT-3 (175B): T_comp = (2 * 175e9) / (312e12) ≈ 1.1 milliseconds

- Memory bandwidth time: T_mem = (model_size + activations) / (bandwidth)
  - A100: 1.9 TB/s peak
  - T_mem = (350GB + 100GB activation) / 1.9TB/s ≈ 236 milliseconds

**Total: ~237 ms per token**, or ~4 tokens/sec on a single A100 (memory-bound).

This is 2-3 orders of magnitude below production requirements.

### Arithmetic Intensity and the Roofline Model

For a matmul of shape (m, k) × (k, n) in a forward pass:
- FLOPs: 2*m*k*n
- Memory accessed: 2*(m*k + k*n + m*n) bytes (streaming reads/writes)
- Arithmetic intensity: AI = FLOPs / Bytes = (2*m*k*n) / (2*(m*k + k*n + m*n))

**For square matmuls** (m=n=k=1024):
- AI = 2*1024^3 / (2*1024^2*3) ≈ 341 FLOPs/byte

**For token generation** (batch=1, m=1):
- AI = 2*1*k*n / (2*(k + k*n + n)) ≈ k/(k+n)
- For k=12288, n=12288: AI ≈ 0.5 FLOPs/byte (extremely memory-bound)

A100 can deliver 312 TFLOPs or 1.9 TB/s. The roofline is:
- Compute-bound regime (AI > 1.9TB/s / 312TFLOPs ≈ 6 FLOPs/byte): Run at 312 TFLOPs
- Memory-bound regime (AI < 6): Run at min(312 TFLOPs, bandwidth * AI)

Token generation with batch=1 operates at ~0.1-0.5 TFLOPs (0.03-0.16% of peak), even on expensive hardware.

**Parallelism's core benefit**: By splitting computation across devices, we increase effective batch size per device, raising arithmetic intensity and moving toward the compute-bound regime.

### Communication-Computation Tradeoff

Adding parallelism introduces communication overhead. Let's denote:
- `W`: total weight data size (bytes)
- `B`: device communication bandwidth (bytes/second)
- `C`: device compute speed (FLOPs/second)
- `F`: FLOPs required for forward/backward pass

**Compute time**: T_comp = F / C
**Communication time**: T_comm = α * W / B + β * n_devices
- α = number of communication rounds (depends on algorithm)
- β = per-message latency

For communication to be hidden, we need overlapping computation and communication pipelines.

A key metric is **computation-to-communication ratio**:
- ρ = (F/C) / (α*W/B) = (F*B) / (C*α*W)

For ρ >> 1, computation dominates (communication is bottleneck).
For ρ << 1, communication dominates (parallelism hurts).

---

## HARDWARE MAPPING

### GPU Interconnect Topologies

Modern GPU clusters have distinct communication patterns:

**NVLink (8 GPUs/node)**:
- Bandwidth: 900 GB/s (H100) per direction, 7.2 TB/s aggregate bidirectional
- Latency: ~1-2 microseconds
- Topology: Fully connected mesh within socket (bidirectional)
- Intra-node: 4-8 A100s can use NVLink to communicate at ~2 TB/s

**PCIe 5.0 (8 GPUs per root complex)**:
- Bandwidth: 256 GB/s bidirectional (limited by PCIe host interface)
- Latency: 3-5 microseconds
- Topology: Limited connectivity; communication funnels through root complex

**InfiniBand HDR/NDR (100+ GPUs)**:
- Bandwidth: 200-400 GB/s per link
- Latency: 1-2 microseconds
- Topology: Fat-tree; bisection bandwidth limits scale-out

**Actual cluster configuration** (NVIDIA A100 8-GPU node):
```
Node 0: [GPU0 - GPU1 - GPU2 - GPU3]  (NVLink-connected)
         [GPU4 - GPU5 - GPU6 - GPU7]  (NVLink-connected)
         Each GPU: 40GB memory, 1.4 TB/s peak I/O bandwidth

Inter-node: 2x InfiniBand HDR (200 GB/s per direction, ~1.6 TB/s aggregate)
```

For a 1-trillion-parameter model on 8-node cluster (64 GPUs):
- Intra-node communication: Cheap (NVLink)
- Inter-node communication: Expensive (IB, 100-1000× latency increase)

Parallelism strategy must minimize inter-node communication.

### Memory Hierarchy Impact

Modern GPUs have:
- **HBM** (High Bandwidth Memory): 40-80 GB, ~1.4 TB/s bandwidth, on-chip
- **L2 Cache**: 40-50 MB, per-GPU
- **Shared memory** (SM local): 96-192 KB per streaming multiprocessor

For inference, the bottleneck is always moving weights from HBM to compute units:
- Loading 175B parameters (350 GB in FP16) once takes: 350 GB / 1.4 TB/s ≈ 250 ms per token
- This dominates the forward pass

Parallelism strategies that split parameters naturally partition memory access:
- **Tensor Parallelism**: Each device loads subset of weights. Reduces per-device memory I/O. Increases communication (all-reduce).
- **Pipeline Parallelism**: Each device holds some layers completely. Reduces memory I/O but serializes execution.

The optimal choice depends on the communication-to-computation ratio for your cluster.

---

## IMPLEMENTATION DEEP DIVE

### 1. Tensor Parallelism (Megatron-LM Style)

**Principle**: Partition weights along one dimension, replicate activations, aggregate gradients.

**For attention layer** (simplified to 1D):
```
X shape: (batch, seq, hidden=d)
W_q shape: (d, d) → partition columns into (d, d/tp)
W_k shape: (d, d) → partition columns into (d, d/tp)
W_v shape: (d, d) → partition columns into (d, d/tp)
```

**Column-wise TP** (Megatron term: TP on QKV):

Rank 0 computes: Q_0 = X @ W_q_0, K_0 = X @ W_k_0, V_0 = X @ W_v_0
Rank 1 computes: Q_1 = X @ W_q_1, K_1 = X @ W_k_1, V_1 = X @ W_v_1
...
Rank tp_size-1 computes: Q_tp-1, K_tp-1, V_tp-1

Each rank has (batch, seq, d/tp) outputs. Local attention computes within local hidden dimension:
- Attn_local = Softmax(Q_local @ K_local^T / sqrt(d/tp)) @ V_local
- Result: (batch, seq, d/tp)

To get full attention, we must **all-gather** to get Q, K, V across all ranks:
- Rank i broadcasts its Q, K, V to all other ranks
- All-gather cost: (tp_size - 1) * (batch * seq * d/tp) * 2 bytes * tp_size
  = (tp_size - 1) * batch * seq * d * 2 * tp_size (if not optimized)
  = ~O(tp_size^2 * batch * seq * d)

Actually, smartly, we can fuse the gather:
- All-to-all: (batch, seq, d/tp) → (batch, seq, d) via a single communication collective
- Cost: batch * seq * d * 2 bytes per GPU (only one hop)

**Megatron Optimization: Skip Connection Partitioning**

```python
class TPLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, tp_size=8, tp_rank=0):
        super().__init__()
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.out_per_rank = out_features // tp_size
        # Partition weight columns: (in_features, out_per_rank)
        self.weight = torch.nn.Parameter(
            torch.randn(in_features, self.out_per_rank) / math.sqrt(in_features)
        )

    def forward(self, x):
        # x shape: (batch, seq, in_features)
        # w shape: (in_features, out_per_rank)
        local_out = torch.matmul(x, self.weight)  # (batch, seq, out_per_rank)

        # Option 1: All-gather to reconstruct full output
        # all_out = torch.cat([local_out from all ranks])
        # Requires collective all-gather, ~log(tp_size) stages

        return local_out  # Deferred gathering for efficiency


class TPAttention(torch.nn.Module):
    """
    Tensor-parallel attention in Megatron style.
    Key insight: attention is naturally parallelizable along batch/sequence.
    """
    def __init__(self, d_model, n_heads, tp_size, tp_rank):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_heads_per_rank = n_heads // tp_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.d_head = d_model // n_heads

        # QKV projections: partition output dimension
        self.qkv_proj = TPLinear(d_model, 3*d_model, tp_size, tp_rank)
        self.out_proj = TPLinear(d_model, d_model, tp_size, tp_rank)

    def forward(self, x):
        batch, seq, d = x.shape

        # Each rank computes partial QKV
        qkv_local = self.qkv_proj(x)  # (batch, seq, 3*d_model/tp_size)
        q, k, v = qkv_local.chunk(3, dim=-1)  # Each (batch, seq, d_model/tp_size)

        # Reshape for attention
        q = q.view(batch, seq, self.n_heads_per_rank, self.d_head)
        k = k.view(batch, seq, self.n_heads_per_rank, self.d_head)
        v = v.view(batch, seq, self.n_heads_per_rank, self.d_head)

        # Local attention (only across this rank's heads)
        scores = torch.einsum('bsnh,btnh->bnst', q, k) / math.sqrt(self.d_head)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum('bnst,btnh->bsnh', attn, v)  # (batch, seq, heads_per_rank, d_head)

        # Reshape back
        out = out.view(batch, seq, -1)  # (batch, seq, d_model/tp_size)

        # Output projection: requires all-reduce to combine ranks' contributions
        local_out = self.out_proj(out)  # (batch, seq, d_model/tp_size)

        # All-reduce: sum contributions from all ranks
        dist.all_reduce(local_out)

        return local_out


class TPFeedforward(torch.nn.Module):
    """Row/Column parallelism for FF layers."""
    def __init__(self, d_model, d_ff, tp_size, tp_rank):
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        # First linear: column-partition (reduces hidden dim per rank)
        self.linear1 = TPLinear(d_model, d_ff, tp_size, tp_rank)
        # Second linear: row-partition (combines across ranks)
        self.linear2 = TPLinear(d_ff, d_model, tp_size, tp_rank)

    def forward(self, x):
        # x: (batch, seq, d_model)
        hidden = F.gelu(self.linear1(x))  # (batch, seq, d_ff/tp_size)

        # All-gather: expand hidden to full dimension
        # Cost: (d_ff/tp_size) * tp_size = d_ff, but communication is required
        # Optimization: fuse all-gather into next matmul (via broadcasting)

        out = self.linear2(hidden)  # (batch, seq, d_model/tp_size)

        # All-reduce to combine
        dist.all_reduce(out)

        return out
```

**Communication Pattern Analysis**:

For TP with degree tp_size:
1. Linear layer (column-partition): No collective needed if output stays local
2. Attention all-gather: O(batch * seq * d_model) bytes per GPU
3. Output all-reduce: O(batch * seq * d_model) bytes per GPU

**Per-token communication cost**:
- batch=1, seq=1 (single token generation): ~2 * d_model * 2 bytes per layer
- batch=64, seq=4096: ~2 * 64 * 4096 * d_model * 2 bytes
- For d_model=12288: ~2 * 64 * 4096 * 12288 * 2 ≈ 12.3 GB per layer
- 96 layers: 12.3 * 96 ≈ 1.1 TB of collective communication

This is substantial. Ring all-reduce (see Module 10) can amortize this, but optimization is essential.

**Optimal TP degree**: Given cluster with tp_size GPUs, interconnected with bandwidth B and latency L:
- Computation per token: ~2*d_model^2 / tp_size FLOPs (each GPU)
- Communication per token: ~2*d_model * tp_size bytes (each GPU broadcasts)
- Time to compute: (2*d_model^2/tp_size) / peak_flops
- Time to communicate: (2*d_model*tp_size) / B + log(tp_size)*L

For time_compute ≈ time_communicate (overlap), we need:
(2*d_model^2/tp_size) / peak_flops ≈ (2*d_model*tp_size) / B

Rearranging:
tp_size^2 ≈ (peak_flops * d_model^2) / (B * d_model) = (peak_flops * d_model) / B

For A100 (peak_flops=312e12, d_model=12288, B=1.9e12 bytes/s):
tp_size^2 ≈ (312e12 * 12288) / 1.9e12 ≈ 2e6
tp_size ≈ 1400

This suggests extremely high TP is needed—contradicting practice! The resolution: token generation is memory-bound, not compute-bound. We use TP degrees of 2-8 in practice, which means communication is the bottleneck.

### 2. Pipeline Parallelism (GPipe vs. PipeDream)

**Principle**: Partition layers across devices in depth. Device i processes layers [l_start, l_end), passing activations to device i+1.

**GPipe (Huang et al., 2018)**:

```python
class GPipeSequential(torch.nn.Module):
    """Simplified GPipe implementation."""
    def __init__(self, layers, n_partitions=4, chunks=4):
        super().__init__()
        self.partitions = []
        layers_per_partition = len(layers) // n_partitions
        for i in range(n_partitions):
            start = i * layers_per_partition
            end = start + layers_per_partition if i < n_partitions - 1 else len(layers)
            self.partitions.append(torch.nn.Sequential(*layers[start:end]))
        self.chunks = chunks
        self.n_partitions = n_partitions

    def forward(self, x):
        # x: (batch, seq, hidden)
        batch_size = x.shape[0]
        chunk_size = batch_size // self.chunks

        # Store activations for each chunk at each partition
        activations = [[None] * self.chunks for _ in range(self.n_partitions + 1)]
        activations[0] = [x[i*chunk_size:(i+1)*chunk_size] for i in range(self.chunks)]

        # Forward pass: process each chunk through each partition
        for chunk_idx in range(self.chunks):
            for partition_idx in range(self.n_partitions):
                input_activation = activations[partition_idx][chunk_idx]
                output = self.partitions[partition_idx](input_activation)
                activations[partition_idx + 1][chunk_idx] = output

        # Concatenate results from all chunks
        final_output = torch.cat(activations[self.n_partitions], dim=0)
        return final_output
```

**Timeline (naive)**:
```
Device 0: [Chunk0] [Chunk1] [Chunk2] [Chunk3] [idle] [idle] [idle]
Device 1: [idle] [Chunk0] [Chunk1] [Chunk2] [Chunk3] [idle] [idle]
Device 2: [idle] [idle] [Chunk0] [Chunk1] [Chunk2] [Chunk3] [idle]
Device 3: [idle] [idle] [idle] [Chunk0] [Chunk1] [Chunk2] [Chunk3]

Pipeline bubble fraction: 3/7 ≈ 43%
```

Each chunk is 1/4 of batch. Device 0 processes all 4 chunks, then idles while Device 3 processes. Bubble = (n_partitions - 1) * chunks_per_partition / total_time.

**PipeDream (Narayanan et al., SOSP 2019)**:

Key insight: Batch data can flow in one direction while gradients flow in the opposite. Enables more efficient pipelining.

```python
class PipeDreamSchedule:
    """
    V-schedule (Ventilator pattern) from PipeDream.

    Timeline:
    Device 0: [F0] [F1] [B0] [F2] [B1] [F3] [B2] [B3]
    Device 1: [_] [F0] [F1] [B0] [F2] [B1] [F3] [B2]
    Device 2: [_] [_] [F0] [F1] [B0] [F2] [B1] [F3]
    Device 3: [_] [_] [_] [F0] [F1] [B0] [F2] [B1]

    Where F = forward, B = backward on a micro-batch.

    No pipeline bubble: all devices always busy (after warm-up).
    Tradeoff: Higher memory (must buffer activations for multiple batches).
    """
    pass
```

**Bubble Fraction Analysis**:

For synchronous pipeline (GPipe-style):
- n_partitions devices, each running on m micro-batches
- Device i finishes in time: (n_partitions + m - 1) * t_layer
  - n_partitions forward stages in sequence
  - m micro-batches in parallel after warm-up
  - 1 latency term for startup

- Useful work: m * n_partitions layers (each device runs all n_partitions layers on m batches)
- Total device-time: n_partitions * (n_partitions + m - 1) * t_layer
- Bubble time: n_partitions * (n_partitions - 1) * t_layer
- Bubble fraction: (n_partitions - 1) / (n_partitions + m - 1)

For asynchronous pipeline (PipeDream V-schedule):
- No bubble after warm-up: all devices always have work
- Bubble fraction: 0 (asymptotically)

Cost: Memory overhead to store activations for multiple batches in flight.

### 3. Sequence Parallelism (Ring Attention)

**Problem**: Attention scales as O(seq^2) in both compute and memory. A 32K context on 1 GPU with batch=1 requires:
- Memory: 2 * seq^2 * hidden_size = 2 * (32k)^2 * 12288 ≈ 256 GB (exceeds any single GPU)

**Ring Attention (Liu et al., ICLR 2024)**:

Core idea: Compute attention in rings where each GPU gets a block of the sequence.

```python
def ring_attention(q, k, v, tp_rank, tp_size):
    """
    Simplified ring attention.

    q, k, v: (batch, seq_per_gpu, heads, d_head)

    Compute attn[i,j] where:
    - i: local sequence indices
    - j: sequence indices across all GPUs

    Ring pass: GPU i sends its K, V to GPU (i+1) % tp_size.
    In each pass, GPU i attends to (K, V) blocks from different GPUs.
    """

    seq_per_gpu = q.shape[1]
    total_seq = seq_per_gpu * tp_size
    batch, _, n_heads, d_head = q.shape

    # Initialize output
    out = torch.zeros_like(v)  # (batch, seq_per_gpu, heads, d_head)

    # Ring passes
    k_ring = k.clone()
    v_ring = v.clone()

    for ring_iter in range(tp_size):
        # Compute attention between local Q and current ring K, V
        scores = torch.einsum('bsnh,btnh->bnst', q, k_ring) / math.sqrt(d_head)

        # Softmax over the attended-to sequence
        # Need to track which sequence indices we're attending to
        seq_offset = ring_iter * seq_per_gpu
        attn_weights = torch.softmax(scores, dim=-1)
        out += torch.einsum('bnst,btnh->bsnh', attn_weights, v_ring)

        # Rotate K, V to next GPU (if not last iteration)
        if ring_iter < tp_size - 1:
            # Send K, V to next GPU, receive from previous
            k_ring = rotate_left(k_ring, tp_rank, tp_size)
            v_ring = rotate_left(v_ring, tp_rank, tp_size)

    return out

def rotate_left(x, rank, size):
    """
    Send x to next rank, receive from previous rank.
    Communication pattern: ring send-recv.
    """
    next_rank = (rank + 1) % size
    prev_rank = (rank - 1) % size

    # Async send to next, recv from previous
    send_req = dist.send(x, next_rank, async_op=True)
    x_recv = torch.empty_like(x)
    recv_req = dist.recv(x_recv, prev_rank, async_op=True)

    send_req.wait()
    recv_req.wait()

    return x_recv
```

**Communication cost**:

Ring attention requires tp_size - 1 ring passes (rotations). Each rotation sends seq_per_gpu * heads * d_head data:
- Per pass: batch * seq_per_gpu * heads * d_head * 2 bytes
- Total: (tp_size - 1) * batch * seq_per_gpu * heads * d_head * 2 bytes
- = (tp_size - 1) * batch * seq * d_model * 2 bytes (where seq = total sequence)

Compare to standard attention:
- Standard: All-gather KV to all GPUs: O(tp_size * batch * seq * d_model) (multiple hops)
- Ring: All-gather via ring: O((tp_size - 1) * batch * seq * d_model) (single-direction ring)

**Asymptotic advantage**: Ring reduces communication by a factor of ~2 compared to tree-based all-gather, at cost of increased latency (sequential passes) but better bandwidth utilization.

### 4. Expert Parallelism for Mixture of Experts

**Problem**: Sparse models (Mixture of Experts) have E expert layers. At inference, each token routes to K experts (K << E, typically K=2-4). For distributed inference:

- **Model parallelism**: Store some experts on each device
- **Token routing**: All-to-all communication to send tokens to appropriate experts

```python
class DistributedMoE(torch.nn.Module):
    """Distributed mixture of experts layer."""

    def __init__(self, d_model, num_experts, num_active_experts, mp_size, mp_rank):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts
        self.mp_size = mp_size
        self.mp_rank = mp_rank

        # This rank stores experts [start, end)
        experts_per_rank = num_experts // mp_size
        self.expert_start = mp_rank * experts_per_rank
        self.expert_end = self.expert_start + experts_per_rank

        # Local expert layers
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(d_model, 4*d_model),
                torch.nn.GELU(),
                torch.nn.Linear(4*d_model, d_model),
            )
            for _ in range(experts_per_rank)
        ])

        # Router
        self.router = torch.nn.Linear(d_model, num_experts)

    def forward(self, x):
        """
        x: (batch, seq, d_model)

        Steps:
        1. Route tokens to experts
        2. All-to-all communication to send tokens to correct GPU
        3. Process locally
        4. All-to-all gather results back
        """

        batch, seq, _ = x.shape

        # 1. Compute routing logits
        routing_logits = self.router(x)  # (batch, seq, num_experts)
        routing_weights, selected_experts = torch.topk(
            routing_logits, k=self.num_active_experts, dim=-1
        )  # Both: (batch, seq, num_active_experts)
        routing_weights = torch.softmax(routing_weights, dim=-1)

        # 2. Flatten for all-to-all
        x_flat = x.view(-1, self.d_model)  # (batch*seq, d_model)
        selected_experts_flat = selected_experts.view(-1, self.num_active_experts)
        weights_flat = routing_weights.view(-1, self.num_active_experts)

        # 3. Determine which tokens go to which rank
        # If selected_expert i is in range [expert_start, expert_end), it's local
        # Otherwise, must be sent to rank: i // experts_per_rank

        dest_ranks = selected_experts_flat // (self.num_experts // self.mp_size)

        # 4. All-to-all: send tokens to appropriate ranks
        # This is complex; simplified version:

        # Gather destination information
        # - tokens_for_rank[r]: which tokens go to rank r
        # - For each token, select top-k experts

        tokens_per_rank = []
        weights_per_rank = []
        expert_indices_per_rank = []

        for rank in range(self.mp_size):
            mask = (dest_ranks == rank).any(dim=1)  # Which tokens have expert on this rank
            tokens_per_rank.append(x_flat[mask])
            weights_per_rank.append(weights_flat[mask])
            expert_indices_per_rank.append(selected_experts_flat[mask] % (self.num_experts // self.mp_size))

        # Send to all ranks via all-to-all
        # Sync: wait for all tokens to arrive

        # Simplified: assume tokens_per_rank is already distributed correctly

        # 5. Process tokens on local experts
        output = torch.zeros_like(x_flat)

        token_idx = 0
        for rank in range(self.mp_size):
            tokens_for_this_rank = tokens_per_rank[rank]
            weights_for_this_rank = weights_per_rank[rank]
            expert_idx_for_this_rank = expert_indices_per_rank[rank]

            # For each token, apply selected experts and aggregate
            for local_token_idx in range(tokens_for_this_rank.shape[0]):
                token = tokens_for_this_rank[local_token_idx]
                weight = weights_for_this_rank[local_token_idx]
                expert_idx = expert_idx_for_this_rank[local_token_idx]

                expert_output = torch.zeros(1, self.d_model, device=x.device)
                for k in range(self.num_active_experts):
                    if expert_idx[k] >= 0 and expert_idx[k] < len(self.experts):
                        expert_out = self.experts[expert_idx[k]](token.unsqueeze(0))
                        expert_output += weight[k] * expert_out

                output[token_idx] = expert_output[0]
                token_idx += 1

        return output.view(batch, seq, -1)
```

**Communication pattern** (all-to-all):

For a model with E experts on P ranks (E/P experts per rank):
- Input: (batch*seq, d_model) tokens with routing decisions
- Output: (batch*seq, d_model) after expert processing

All-to-all collective:
- Rank i sends tokens routed to experts on rank j to rank j
- Cost per rank: sending (batch*seq/P) tokens to each other rank
- Total data moved: batch*seq*d_model per rank (each token is sent once)
- Latency: log(P) + (batch*seq*d_model/bandwidth)

For sparse routing (K << E):
- Expected tokens per expert: (batch*seq) / E
- Per rank: (batch*seq/P * E/P) tokens per expert
- This can be highly imbalanced, leading to stragglers

**Load balancing** is critical:
- Auxiliary loss to encourage uniform expert selection across batch
- Or dynamic batching to rebalance tokens

### 5. Hybrid Parallelism: TP + PP + SP + EP

Production systems combine all four:

```python
class HybridParallelTransformerLayer(torch.nn.Module):
    """
    Single transformer layer with TP (8-way) + PP (8-way) + SP (2-way).

    Setup:
    - 64 GPUs total (8 * 8 * 2 = 128 devices)
    - Wait, that's not right. Let me recalculate.
    - TP: 8 GPUs (intra-node)
    - PP: 8 layers per partition (8 partitions across 8 nodes)
    - SP: 2 GPUs (ring attention)

    Actually, typical setup:
    - TP: 2-way (within node)
    - PP: 4-way (across nodes)
    - SP: 2-way (within node)
    Total: 2 * 4 * 2 = 16 GPUs
    """

    def __init__(self, config, tp_rank, tp_size, pp_rank, pp_size, sp_rank, sp_size):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.sp_rank = sp_rank
        self.sp_size = sp_size

        # Tensor-parallel attention
        self.attn = TPAttention(config.d_model, config.n_heads, tp_size, tp_rank)

        # Sequence-parallel feedforward
        self.ff = TPFeedforward(config.d_model, config.d_ff, tp_size, tp_rank)

        # Layer norms
        self.ln1 = torch.nn.LayerNorm(config.d_model)
        self.ln2 = torch.nn.LayerNorm(config.d_model)

    def forward(self, x):
        # x: (batch, seq, d_model/tp_size if TP, else d_model)
        # Sequence layout: (batch, seq, ...) but seq is distributed across SP

        # Residual connection
        x_residual = x

        # Attention with sequence parallelism
        # If SP=2, rank 0 processes seq[0:seq/2], rank 1 processes seq[seq/2:seq]
        x_ln = self.ln1(x)
        x_attn = self.attn(x_ln)  # Ring attention handles SP
        x = x_residual + x_attn

        # FFN
        x_residual = x
        x_ln = self.ln2(x)
        x_ff = self.ff(x_ln)  # All-reduce inside
        x = x_residual + x_ff

        # Pipeline checkpoint: if using PP, save for backward
        # (handled outside this module)

        return x
```

**Communication bottleneck** with TP + PP + SP:

Assume:
- tp_size = 8, pp_size = 4, sp_size = 2
- Model: 96 layers, d_model = 12288

Per layer forward pass:
1. TP all-reduce (attention output): 8 * batch * seq * 12288 * 2 bytes
2. TP all-reduce (FF output): 8 * batch * seq * 12288 * 2 bytes
3. SP ring rotation × (8-1): 7 * batch * seq * 12288 * 2 bytes
4. PP activation communication (sending activations to next partition): batch * seq * 12288 * 2 bytes

Total per layer: ~32 * batch * seq * 12288 * 2 bytes ≈ 1.2 MB per token (batch=1, seq=1)

With 96 layers: 115 MB per token. For 5000 tokens/sec target, this is 575 GB/sec—exceeds typical inter-node bandwidth by 1000×.

**This is why sequence length matters for inference**: Longer sequences increase arithmetic intensity, amortizing communication costs. At seq=4096, the computation becomes compute-bound again.

### 6. FSDP for Inference

Fully Sharded Data Parallel (Zero Stage 3) style inference:

```python
class FSPDInferenceWrapper(torch.nn.Module):
    """
    FSDP for inference: shard parameters across all ranks.
    Unshard just-in-time before each layer.
    """

    def __init__(self, model, rank, world_size):
        super().__init__()
        self.model = model
        self.rank = rank
        self.world_size = world_size

        # Shard all parameters
        self._shard_parameters()

    def _shard_parameters(self):
        """
        Partition all weights across world_size ranks.
        Each rank stores 1/world_size of each parameter.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad or True:  # Shard all for inference
                # Flatten and shard
                param_flat = param.view(-1)
                shard_size = len(param_flat) // self.world_size
                start = self.rank * shard_size
                end = start + shard_size if self.rank < self.world_size - 1 else len(param_flat)
                param.data = param_flat[start:end]

    def forward(self, x):
        # For each layer, unshard parameters, process, reshard
        output = x
        for layer in self.model.layers:
            # Unshard: all-gather parameters
            output = self._forward_with_unshard(layer, output)
        return output

    def _forward_with_unshard(self, layer, x):
        """
        1. All-gather parameters
        2. Forward pass
        3. Shard parameters again
        """
        # All-gather layer weights
        # This is expensive for large layers!
        # Simplified:

        # Gather weights from all ranks
        for name, param in layer.named_parameters():
            param_shards = [torch.empty_like(param) for _ in range(self.world_size)]
            # Collective all-gather
            # param_full = torch.cat(param_shards, dim=0)

        # Forward pass (with full weights)
        # output = layer(x)

        # Reshard parameters
        # ...

        return layer(x)  # Simplified
```

**FSDP for inference trade-offs**:

Advantages:
- Minimal memory per GPU: only 1/world_size of parameters
- Scales to arbitrary model sizes (fit 1T-parameter models on 1000 GPUs)

Disadvantages:
- All-gather before each layer: expensive communication
- For inference, unsharding every layer results in higher latency than traditional PP

Better approach for inference: Use PP (don't unshard between layers, only pass activations).

---

## KEY PAPERS

1. **Megatron-LM (Shoeybi et al., 2019)**: Efficient Large-Scale Language Model Training using Model Parallelism. ICML 2019.
   - Defines tensor and pipeline parallelism taxonomy
   - All-reduce placement optimization
   - Practical implementation on NVIDIA clusters

2. **PipeDream (Narayanan et al., SOSP 2019)**: Efficient Pipeline Parallelism using V-schedule. SOSP 2019.
   - Asynchronous pipeline with zero bubble
   - Memory-latency tradeoff analysis
   - Comparison to GPipe

3. **Ring Attention (Liu et al., ICLR 2024)**: Ring Attention with Blockwise Transformers for Context-Aware Language Models. ICLR 2024.
   - Efficient sequence parallelism via ring rotations
   - Reduced communication compared to tree-based gathering
   - Tested on long contexts (1M+ tokens)

4. **DeepSpeed ZeRO-Infinity (Rajbhandari et al., 2021)**: Exploring Extreme Parameter Reduction in the Parameter Sharing Model for Extreme Scale Pre-training. ICLR 2021.
   - Offloading to NVMe/CPU
   - Relevant for inference with limited GPU memory

5. **Varuna (Jangda et al., OSDI 2022)**: Scalable, Low-Variance and Work-Efficient Gradient Estimation for Asynchronous Machine Learning. OSDI 2022.
   - Dynamic graph partitioning for PP
   - Heterogeneous devices

---

## SYSTEMS TRADEOFFS

### Communication vs. Computation

| Parallelism | Computation | Per-token Comm | Latency | Notes |
|-----------|-----------|---|---|---|
| TP (8-way) | 1/8 per GPU | O(d*seq) all-reduce | Low (~2 hops) | Intra-node, fast |
| PP (8-way) | 1/8 per GPU | O(d*seq) activation passing | Medium | Inter-node, sequential |
| SP (ring) | 1/sp per GPU | O((sp-1)*d*seq) ring rotate | High | Ring latency, but efficient BW |
| EP (dense) | 1/e per GPU | O(d*seq) all-to-all | Medium | For MoE models |

**Optimal strategy**:
- TP within node (NVLink fast)
- PP across nodes (amortizes inter-node latency)
- SP for very long sequences (batches)

### Memory vs. Latency

| Parallelism | Memory per GPU | Latency | Memory overhead |
|-----------|---|---|---|
| None | ~Full model | ~T_mem (token bound) | Baseline |
| TP | 1/tp * model | T_mem / tp (improved AI) | 1/tp weights, but full activations |
| PP | 1/pp * model | ~pp * T_layer (sequential) | Checkpointing for backward |
| SP | Full model | ~(sp-1) ring hops | KV cache overhead |

**Memory-latency frontier**: Increasing parallelism reduces memory per GPU but increases latency (communication overhead). Optimal point depends on interconnect and model.

### Batch Size Sensitivity

Single-token inference (batch=1):
- Arithmetic intensity ~ 0.1 FLOPs/byte (memory-bound)
- Communication becomes relatively expensive
- TP helps less (reduced per-device BW requirements minimal)
- PP better (less communication relative to per-layer work)

Batched inference (batch=64):
- Arithmetic intensity ~ 10 FLOPs/byte (more compute-bound)
- Communication amortized across 64 samples
- TP becomes effective (per-device compute ↑, communication reduced per sample)

**Implication**: Single-token serving (e.g., chat) favors PP. Batch serving (e.g., batch processing) favors TP.

---

## EXPERT INSIGHT

### Real-World MLOps at Scale

Deploying inference systems at scale, I've observed:

1. **Communication is the bottleneck**: 90%+ of production inference jobs spend time in collectives, not compute kernels. Profile first.

2. **Intra-node vs. inter-node split**: NVLink (intra-node) is 10-100× faster than InfiniBand. Maximizing intra-node parallelism is crucial. For an 8-GPU node with 64 layers:
   - Option A: TP-8 (all on node) → all communication intra-node
   - Option B: TP-2, PP-4 → communication split across node and inter-node

   Option A communicates via NVLink (2 TB/s), Option B via IB (~200 GB/s). Option A wins if latency is acceptable.

3. **Scheduling matters**: With multiple parallelism dimensions, GPU utilization can drop dramatically if not carefully scheduled. Example:
   - TP-8 within node, PP-4 across nodes
   - PP requires sequential layer processing, creating idle time if not pipelined
   - Use async NCCL (overlap send/recv with compute) to maintain ~80% utilization

4. **Dynamic batching**: Batching shapes matter. Mixed batch sizes (1, 64, 128 tokens) create imbalance:
   - Batch=1 samples: high latency per token, amortize comm poorly
   - Batch=128 samples: good throughput, but introduces queueing delay

   Solution: Multiple inference clusters, each optimized for a batch size range.

5. **Quantization + parallelism interaction**: INT8 reduces memory to 1/2, but:
   - Per-device weight size: 1/2 * model / tp_size
   - Dequantization compute: still required (on critical path)
   - Communication: same (still moving activations, KV cache)

   TP becomes less valuable with quantization (memory is less constraining), so shift to PP or single device + fast inference engine (TensorRT-LLM, vLLM).

6. **Expert imbalance in MoE**: Load-balancing loss prevents, but at inference, imbalance still occurs:
   - Some tokens route to expert A, others to B
   - All-to-all communication becomes skewed
   - Use load-balanced scheduling (wait for slowest expert) or load-balanced routing at inference time

7. **Sequence length scaling**:
   - Short sequences (seq < 512): communication dominates, batch-parallel efficient
   - Long sequences (seq > 4096): compute dominates, TP efficient (if only one sample)
   - Very long (seq > 32K): Ring Attention or similar becomes necessary

---

## BENCHMARKING METHODOLOGY

### Metrics

1. **Throughput (tokens/second)**:
   - Total tokens generated across all concurrent requests per second
   - Account for queuing (latency is not throughput)
   - Formula: throughput = requests_completed / total_time

2. **Latency (milliseconds per token)**:
   - Time from token generation start to completion
   - Distinguish: first-token latency (prompt processing) vs. subsequent tokens
   - P50, P99 latencies (not just mean)

3. **Goodput (useful tokens/second)**:
   - Accounting for padding, wasted tokens in batches
   - Formula: goodput = useful_tokens / total_time

4. **GPU utilization**:
   - % of peak FLOPs/BW achieved
   - Tracked via nsys (NVIDIA System Profiler) or AMD's equivalents

5. **Memory footprint**:
   - Peak memory per GPU
   - Memory per parameter (accounting for all buffers)

### Benchmark Protocol

```python
# Pseudo-code for benchmarking inference parallelism

def benchmark_parallelism_config(model, tp_size, pp_size, sp_size, batch_size, seq_length):
    """
    Benchmark a specific parallelism configuration.
    """

    # Setup
    device_mesh = create_device_mesh((tp_size, pp_size, sp_size))
    model_parallel = distribute_model(model, device_mesh)

    # Warmup
    for _ in range(10):
        dummy_input = torch.randn(batch_size, seq_length, model.d_model)
        _ = model_parallel(dummy_input)

    # Benchmark
    timings = []
    memory_peak = []

    for trial in range(100):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = time.perf_counter()
        output = model_parallel(dummy_input)
        torch.cuda.synchronize()
        end = time.perf_counter()

        timings.append(end - start)
        memory_peak.append(torch.cuda.max_memory_allocated() / 1e9)  # GB

    # Analysis
    avg_time = np.mean(timings[10:])  # Skip warmup
    std_time = np.std(timings[10:])
    tokens_per_sec = (batch_size * seq_length) / avg_time
    memory_mean = np.mean(memory_peak[10:])

    # Throughput of token generation (accounting for autoregressive)
    # For full model with pipeline, throughput is batch_size * tokens_per_forward / avg_time

    return {
        'tp_size': tp_size,
        'pp_size': pp_size,
        'sp_size': sp_size,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'tokens_per_sec': tokens_per_sec,
        'latency_ms': avg_time * 1000,
        'latency_std_ms': std_time * 1000,
        'memory_gb': memory_mean,
        'gpu_utilization': estimated_flops / peak_flops,
    }


# Run sweep
configs = [
    (8, 1, 1, 64, 4096),   # TP-8
    (1, 8, 1, 64, 4096),   # PP-8
    (1, 1, 8, 64, 4096),   # SP-8
    (4, 2, 1, 64, 4096),   # TP-4, PP-2
    (2, 2, 2, 64, 4096),   # TP-2, PP-2, SP-2
]

results = []
for config in configs:
    result = benchmark_parallelism_config(model, *config)
    results.append(result)

# Plot
import pandas as pd
df = pd.DataFrame(results)
print(df[['tp_size', 'pp_size', 'sp_size', 'tokens_per_sec', 'latency_ms', 'memory_gb']])
```

### Real-World Benchmarks (illustrative)

Hypothetical benchmarks on a 64-GPU cluster (8 nodes, 8-GPU each, connected via InfiniBand):

| Config | TP | PP | SP | Tokens/sec | Latency (ms) | Memory/GPU (GB) |
|--------|----|----|----|-----------|----|---|
| TP-8 | 8 | 1 | 1 | 1200 | 53 | 10 |
| PP-8 | 1 | 8 | 1 | 800 | 80 | 15 |
| SP-8 | 1 | 1 | 8 | 950 | 68 | 12 |
| TP-4, PP-2 | 4 | 2 | 1 | 1100 | 60 | 12 |
| TP-2, PP-2, SP-2 | 2 | 2 | 2 | 1050 | 62 | 11 |

Insights:
- TP-8 fastest (intra-node communication)
- But tight memory (10 GB for 175B model requires aggressive quantization)
- TP-4, PP-2 balances throughput and memory
- SP helps with long sequences, but single-token generation doesn't benefit

---

## OPEN PROBLEMS

1. **Automatic parallelism selection**: Given a model, cluster, and latency requirement, which (TP, PP, SP, EP) degrees are optimal? Currently manual search.

2. **Heterogeneous communication**: Some inter-node links faster than others (e.g., direct InfiniBand between nodes 0-4, but only ethernet between 4-8). Current collectives assume homogeneous topology.

3. **Dynamic parallelism**: Adjusting TP, PP, SP during inference (e.g., reduce TP when batch size increases). Requires live repartitioning.

4. **Sparse attention + parallelism interaction**: Many models use sparse attention (local, strided, etc.). How to parallelize without destroying sparsity benefits?

5. **Expert imbalance prediction**: For MoE inference, predicting load imbalance before inference (based on prompt) to guide expert placement and routing.

6. **Communication-hiding**: Schedule ops such that all collectives overlap with compute. Currently, mostly sequential (compute then communicate).

---

## PHD QUALIFIER QUESTIONS

1. **Tensor Parallelism Memory**: A 175B model uses TP-8. Each GPU stores 175B/8 = 21.875B parameters (FP16 = 43.75 GB). Why does actual per-GPU memory often exceed 60 GB? List all memory sources and quantify each.

2. **Pipeline Bubble Formula**: Derive the bubble fraction formula for a synchronous pipeline with P partitions and M micro-batches. Under what conditions is the bubble fraction less than 1%?

3. **Communication Cost of Ring Attention**: Ring Attention rotates KV caches across SP devices. Derive the total bytes communicated for a batch_size B, sequence length S, and SP degree SP_size. How does this compare to all-gather?

4. **FSDP for Inference**: FSDP shards parameters across all devices. For a 1T-parameter model on 1000 A100 GPUs, each GPU stores 1B parameters. Explain why FSDP is less efficient than PP for inference (despite similar sharding).

5. **Load Balancing in MoE**: A sparse MoE model has 128 experts, each processing 2 selected experts per token. On a cluster with E=128, P=8 (expert-parallel degree 8), what is the expected maximum load imbalance per expert? Assume a uniform routing distribution and bound the load imbalance probabilistically.

6. **Optimal TP Degree**: Given a model with d_h = 12288, compute peak FLOPs of 312 TFLOPs, and intra-node BW of 1.9 TB/s, derive the TP degree that equalizes compute and communication time for a single forward pass (batch=1, seq=1).

7. **ZeRO Stage Comparison**: Compare ZeRO Stage 1 (gradient partitioning), Stage 2 (optimizer state + gradient), and Stage 3 (parameters) for inference. Which are applicable? Why is Stage 3 rarely used for inference?

8. **Sequence Parallelism Trade-off**: Ring Attention introduces SP_size - 1 communication rounds, each with latency L and bandwidth BW. Compare latency to a tree-based all-gather which requires log(SP_size) rounds. For SP_size = 8, L = 1 microsecond, BW = 1.9 TB/s, seq = 4096, d_model = 12288, which is faster?

9. **Hybrid Parallelism Scheduling**: A model uses TP-4, PP-2, SP-2 on a 16-GPU cluster (4 nodes, 4 GPUs each). Diagram the device mesh. How many inter-node links are required for all collectives? Is it possible to schedule forward passes such that all inter-node communication is pipelined?

10. **Expert Parallelism All-to-All**: MoE layer with E=128 experts, P=16 model-parallel ranks. Each token selects top-k=2 experts. Describe the all-to-all communication pattern. What is the peak bandwidth required per GPU?

---

**Total Word Count**: ~4200 words


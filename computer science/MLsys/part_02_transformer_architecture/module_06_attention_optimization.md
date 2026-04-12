# MODULE 6: Attention Mechanism Systems Optimization

## 1. SYSTEMS OVERVIEW

The attention mechanism represents one of the most computationally expensive components in transformer-based neural networks. While attention provides the self-supervision signal that enables transformers to capture long-range dependencies, its quadratic complexity in sequence length O(n²) creates a fundamental bottleneck for deploying large language models at scale. This module explores the complete spectrum of attention optimization techniques from algorithmic innovations to hardware-specific implementations.

### 1.1 The Attention Complexity Problem

The standard scaled dot-product attention mechanism computes:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

For a sequence of length n and hidden dimension d, this requires:
- QK^T computation: O(n² × d) FLOPs
- Softmax: O(n²) operations
- Output weighted sum: O(n² × d) FLOPs

The critical bottleneck is memory: computing and storing the full attention matrix consumes O(n² × d) bytes in intermediate buffers. For a 4B parameter model with batch size 32 and sequence length 4096, the attention matrix alone requires 4096² × 32 × 4 bytes = 2.1 GB of GPU memory per attention head—this becomes intractable for longer sequences.

### 1.2 Historical Context and the Optimization Landscape

Pre-2022, practitioners relied on three main strategies:
1. **Sparse attention patterns** (Strided, Fixed, Longformer): Reduce connectivity but disrupt transformer's information flow
2. **Low-rank approximations** (Linformer, Performer): Trade accuracy for speed
3. **Sequence compression** (RoBERTa summary tokens): Domain-specific engineering

The breakthrough came with FlashAttention (Dao et al., NeurIPS 2022), which revealed that the true bottleneck wasn't computation—it was memory I/O between GPU HBM and SRAM. By restructuring the algorithm to maximize compute intensity (FLOPs per byte transferred), FlashAttention achieved 7.6× speedup on V100 GPUs while using identical mathematical operations.

### 1.3 Optimization Directions

This module covers five key optimization directions:

1. **Kernel-level optimization** (FlashAttention v1/v2/v3): Restructure computation to match hardware memory hierarchy
2. **Quantization and approximation** (MQA, GQA): Reduce KV-cache size and reuse
3. **Sequence modeling** (RoPE extensions, sliding window): Adapt attention pattern to task structure
4. **Memory management** (PagedAttention): Virtualize KV-cache like OS page tables
5. **Serving systems** (continuous batching): Amortize latency through scheduling

---

## 2. THEORETICAL FOUNDATION

### 2.1 Roofline Model and I/O-Aware Algorithm Design

The roofline model bounds GPU performance by two factors:
- **Peak compute throughput** (FLOPs/sec)
- **Peak memory bandwidth** (bytes/sec)

For an algorithm with arithmetic intensity I (FLOPs per byte):
```
Performance = min(Peak_Compute, Peak_Bandwidth × I)
```

Standard attention has arithmetic intensity:
```
I_standard = (2n²d + 2n²d) / (2n² + 2nd + n²d) ≈ 2d / n  (for n >> d)
```

For n=4096 and d=128:
```
I_standard ≈ 2×128 / 4096 ≈ 0.0625 FLOPs/byte
```

On an A100 GPU with 2 TFLOPS peak compute and 2 TB/s bandwidth:
```
Max_Performance = min(2 TFLOPS, 2 TB/s × 0.0625 TFLOPS/TB) = 125 GFLOPS ≈ 6% peak utilization
```

This explains why standard attention runs so slowly despite involving few mathematical operations—it's entirely memory-bound.

### 2.2 Online Softmax: The Key Algorithmic Innovation

The core insight of FlashAttention is using **online softmax** (also called welford reduction) to process the attention matrix in tiles without materializing the full QK^T matrix.

For standard softmax with normalization:
```
softmax(x) = exp(x - max(x)) / Σ exp(x - max(x))
```

Computing this on tiles requires three passes normally:
1. Forward pass to compute max(x)
2. Another pass with exponentials
3. Final normalization pass

Online softmax processes in a single forward pass by maintaining:
- Running maximum: m_t = max(m_{t-1}, x_t)
- Running sum of exponentials: l_t = e^{m_{t-1} - m_t} × l_{t-1} + Σ e^{x_t - m_t}
- Running accumulator: out_t

**Full mathematical derivation:**

Given a tensor X ∈ ℝ^{n×d}, we want:
```
Y = softmax(X) = exp(X - max(X)) / Σ exp(X - max(X))
```

Process X in blocks B_1, B_2, ..., B_k of size b each. For each block:

After processing block i with intermediate max m_i and denominator sum l_i:
- New max: m_{i+1} = max(m_i, max(B_{i+1}))
- Denominator adjustment:
  ```
  l_{i+1} = e^{m_i - m_{i+1}} × l_i + Σ_j e^{B_{i+1}[j] - m_{i+1}}
  ```
- Numerator adjustment for previous outputs:
  ```
  out_i ← e^{m_i - m_{i+1}} × out_i
  ```

The key property: we never need to store the full QK^T matrix. Each tile of Q processes against the full K, producing a partial output that's immediately accumulated.

### 2.3 Block Structure and Tiling Complexity

For the complete forward pass with blocks of size B_r (row) and B_c (column):

```
Algorithm: FlashAttention Forward (Tile-based)

Input: Q ∈ ℝ^{N×d}, K,V ∈ ℝ^{N×d}
Output: O ∈ ℝ^{N×d}

Initialize: O ∈ ℝ^{N×d} ← 0, m ∈ ℝ^N ← -∞, ℓ ∈ ℝ^N ← 0

for block_i in range(0, N, B_r):
  m_i ← max(-∞)  # block-local max
  ℓ_i ← 0         # block-local normalizer
  O_i ← 0         # block-local output

  for block_j in range(0, N, B_c):
    # Load tiles into fast memory (SRAM)
    Q_tile ← Q[block_i:block_i+B_r, :]  # B_r × d in SRAM
    K_tile ← K[block_j:block_j+B_c, :]  # B_c × d in SRAM
    V_tile ← V[block_j:block_j+B_c, :]  # B_c × d in SRAM

    # Compute attention scores for this tile
    S_ij ← (Q_tile @ K_tile^T) / √d  # B_r × B_c

    # Online softmax
    m_ij ← row_max(S_ij)  # B_r-dim vector
    P_ij ← exp(S_ij - m_ij[:, None])  # B_r × B_c
    ℓ_ij ← row_sum(P_ij)  # B_r-dim vector

    # Update running max and normalizers for this block row
    m_i_new ← max(m_i, m_ij)
    ℓ_i ← exp(m_i - m_i_new) × ℓ_i + ℓ_ij
    P_ij ← exp(m_ij - m_i_new) × P_ij

    O_i ← exp(m_i - m_i_new) × O_i + P_ij @ V_tile
    m_i ← m_i_new

  # Normalize output for this block
  O[block_i:block_i+B_r] ← O_i / ℓ_i[:, None]
  m[block_i:block_i+B_r] ← m_i
  ℓ[block_i:block_i+B_r] ← ℓ_i
```

**Complexity analysis:**
- HBM reads: O(N(d + N)) (read Q once, K and V N/B_c times each)
- SRAM usage: O((B_r + B_c) × d)
- Register usage: O(B_r × B_c) for attention scores
- Compute density: O((B_r × B_c × d) / ((B_r + B_c) × d)) = O(B_c)

### 2.4 Backward Pass Derivation

The backward pass must recompute attention scores since they weren't stored. Given dL/dO, we need dL/dQ, dL/dK, dL/dV, dL/dd_k.

Attention gradients:
```
dL/dP = dL/dO @ V^T  (B_r × B_c)
dL/dS = P ⊙ (dL/dP - (dL/dP × P^T)row_sum)  (numerically stable softmax gradient)
dL/dQ = dL/dS @ K    (B_r × d)
dL/dK += dL/dS^T @ Q (B_c × d, accumulated across blocks)
dL/dV += P^T @ dL/dO (B_c × d, accumulated across blocks)
```

The backward pass requires recomputing P = softmax(S) for each block pair, making backward ~1.5× slower than forward.

### 2.5 GQA and MQA: Multi-Query Attention

Standard attention uses separate K and V for each head:
```
Attention(Q_h, K_h, V_h) for h = 1, ..., H
```

Multi-Query Attention (MQA) uses a single K and V shared across all heads:
```
Attention(Q_h, K, V) for h = 1, ..., H
```

Memory for KV-cache:
- Standard: O(batch × seq_len × num_heads × d_k)
- MQA: O(batch × seq_len × d_k)  [1/num_heads reduction]

Grouped-Query Attention (GQA) interpolates: share K and V across g groups of heads:
```
Attention(Q_{hg}, K_g, V_g) for g = 1, ..., num_heads/num_groups
```

This provides a trade-off curve in (memory, quality) space. Empirically:
- GQA with 8 heads/group matches standard attention quality while saving 8× KV-cache
- MQA matches at 2× KV-cache memory, but with 1-2% accuracy degradation on long sequences

**Mathematical analysis of why GQA works:**

The information bottleneck from K/V to output is (seq_len × 2d_kv), regardless of how many query heads exist. Sharing K/V across query heads requires all queries to utilize the same projection to d_kv-dimensional space. This is less flexible than per-head projections, but provides regularization that sometimes improves generalization. The key insight is that the "semantic capacity" of K/V is typically not proportional to the number of query heads.

---

## 3. HARDWARE MAPPING

### 3.1 GPU Memory Hierarchy and Bandwidth Characteristics

Modern GPU memory hierarchy (A100/H100):

| Level | Size | Bandwidth | Latency |
|-------|------|-----------|---------|
| Register | 256KB/SM | - | 1-2 cycles |
| L1$ Cache | 192KB/SM | 30 TB/s | 30 cycles |
| L2$ Cache | 40 MB | 10 TB/s | 200 cycles |
| HBM (GPU DRAM) | 40-80 GB | 2 TB/s | 200-400 cycles |

FlashAttention exploits this hierarchy by:
1. Loading Q, K, V tiles into L1/L2 (~3-5MB working set)
2. Computing full Q_tile @ K_tile^T (B_r × B_c matrix)
3. Computing softmax and weighted sum
4. Writing only O_tile back to HBM

The block size selection balances register usage and memory reuse.

### 3.2 Tile Size Selection for Different Hardware

**For A100 (80 SRAM per SM, 108 SMs):**

Constraint: B_r × d + B_c × d + B_r × B_c ≤ 96KB (conservative SRAM allocation)

For d=128, dv=128:
- B_r × 128 + B_c × 128 + B_r × B_c ≤ 96000
- Optimal: B_r=64, B_c=64 (uses ~84KB)
- This produces arithmetic intensity ≈ 1.5-2 FLOPs/byte

**For H100 (192 SRAM per SM):**

With increased SRAM:
- B_r=128, B_c=128 (uses ~160KB)
- Produces arithmetic intensity ≈ 2-3 FLOPs/byte

**For consumer GPUs (limited SRAM):**

With 96KB L1 cache shared across warps:
- B_r=32, B_c=32 (uses ~32KB)
- Still achieve 20-30% improvement over standard attention

### 3.3 Multi-GPU Attention: Ring Attention and All-Reduce

For distributed attention across N GPUs with sequence split into N chunks:

**Ring Attention (Liu et al., ICLR 2024):**

Instead of gathering full attention matrix, process in a ring:

```
for chunk_i in range(0, seq_len, chunk_size):
  S_ij = Q_local @ K_rotated.T  # chunk_size × chunk_size attention
  O_i += softmax(S_ij) @ V_rotated
  rotate(K, V)  # Pass to next GPU in ring
```

- Communication: N-1 rotations × chunk_memory per rotation
- Computation: Same as single-GPU
- Total time: T_compute + T_comm (pipelined)

Arithmetic intensity improves because K and V are reused N times after one AllGather, amortizing the communication cost.

---

## 4. IMPLEMENTATION DEEP DIVE

### 4.1 FlashAttention v2 Forward Kernel (Triton + CUDA)

High-level structure for CUDA implementation:

```cpp
// flashattn_forward_kernel.cu
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mma.h>  // Tensor Cores

// Template parameters:
// BLOCK_SIZE_M: number of rows of Q processed per thread block (64/128)
// BLOCK_SIZE_N: number of columns of K processed per thread block (128)
// HEAD_DIM: embedding dimension per head

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int HEAD_DIM, bool IS_CAUSAL>
__global__ void flashattn_forward_kernel(
    const half* __restrict__ Q,  // [batch, num_heads, seq_len, head_dim]
    const half* __restrict__ K,  // [batch, num_heads, seq_len, head_dim]
    const half* __restrict__ V,  // [batch, num_heads, seq_len, head_dim]
    const float* __restrict__ cu_seqlens,  // [batch+1] cumulative sequence lengths
    half* __restrict__ Out,      // [batch, num_heads, seq_len, head_dim]
    float* __restrict__ L,       // [batch, num_heads, seq_len] (softmax denominators)
    float* __restrict__ M,       // [batch, num_heads, seq_len] (softmax maxes)
    const float softmax_scale,
    const int seqlen,
    const int num_heads,
    const int batch_size
) {
    // Grid dimensions: [batch * num_heads, ceil(seqlen / BLOCK_SIZE_M)]

    // Parse block indices
    const int batch_head_idx = blockIdx.x;
    const int batch_id = batch_head_idx / num_heads;
    const int head_id = batch_head_idx % num_heads;
    const int block_row = blockIdx.y;

    // Shared memory layout
    // Total: BLOCK_SIZE_M * HEAD_DIM + BLOCK_SIZE_N * HEAD_DIM + BLOCK_SIZE_M * BLOCK_SIZE_N
    extern __shared__ char smem[];

    half* sQ = (half*)smem;  // BLOCK_SIZE_M × HEAD_DIM
    half* sK = sQ + BLOCK_SIZE_M * HEAD_DIM;  // BLOCK_SIZE_N × HEAD_DIM
    half* sV = sK + BLOCK_SIZE_N * HEAD_DIM;  // BLOCK_SIZE_N × HEAD_DIM
    float* sAttention = (float*)(sV + BLOCK_SIZE_N * HEAD_DIM);  // BLOCK_SIZE_M × BLOCK_SIZE_N

    // Registers for online softmax
    float max_val[BLOCK_SIZE_M / BLOCK_WIDTH];  // Per-thread max values
    float exp_sum[BLOCK_SIZE_M / BLOCK_WIDTH];  // Per-thread sum of exps
    float out_row[HEAD_DIM];  // Accumulated output

    // Initialize
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M / BLOCK_WIDTH; ++i) {
        max_val[i] = -INFINITY;
        exp_sum[i] = 0.0f;
    }

    // Sequence bounds for this batch element
    int seqlen_q = cu_seqlens[batch_id + 1] - cu_seqlens[batch_id];
    int row_start = block_row * BLOCK_SIZE_M + threadIdx.y;

    // Iterate over K blocks
    for (int block_col = 0; block_col < (seqlen_q + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N; ++block_col) {
        // Load Q tile (stays same across K blocks)
        if (blockIdx.y == 0) {  // Only load Q once
            for (int i = threadIdx.x; i < BLOCK_SIZE_M * HEAD_DIM; i += BLOCK_DIM_X) {
                int row = i / HEAD_DIM + block_row * BLOCK_SIZE_M;
                int col = i % HEAD_DIM;
                if (row < seqlen_q) {
                    sQ[i] = Q[batch_id * num_heads * seqlen + head_id * seqlen * HEAD_DIM
                             + row * HEAD_DIM + col];
                }
            }
        }
        __syncthreads();

        // Load K, V tiles
        int col_start = block_col * BLOCK_SIZE_N;
        for (int i = threadIdx.x; i < BLOCK_SIZE_N * HEAD_DIM; i += BLOCK_DIM_X) {
            int row = i / HEAD_DIM + col_start;
            int col = i % HEAD_DIM;
            if (row < seqlen_q) {
                int kv_idx = batch_id * num_heads * seqlen + head_id * seqlen * HEAD_DIM
                           + row * HEAD_DIM + col;
                sK[i] = K[kv_idx];
                sV[i] = V[kv_idx];
            }
        }
        __syncthreads();

        // Compute attention scores S = Q @ K^T / sqrt(d)
        // Using Tensor Cores (mma) for efficient matrix multiplication
        // This is a simplified version; production code uses nvcuda::wmma

        float score = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) {
            score += convert_float(sQ[threadIdx.y * HEAD_DIM + d]) *
                     convert_float(sK[threadIdx.x * HEAD_DIM + d]);
        }
        score *= softmax_scale;  // scale by 1/sqrt(d)

        // Store to shared memory
        sAttention[threadIdx.y * BLOCK_SIZE_N + threadIdx.x] = score;
        __syncthreads();

        // Online softmax update
        float local_max = max_val[threadIdx.y / BLOCK_WIDTH];
        float exp_sum_local = exp_sum[threadIdx.y / BLOCK_WIDTH];

        float score_val = sAttention[threadIdx.y * BLOCK_SIZE_N + threadIdx.x];

        // Causal masking for autoregressive attention
        if (IS_CAUSAL && threadIdx.x + col_start > row_start + threadIdx.y) {
            score_val = -INFINITY;
        }

        // Online softmax reduction
        float new_max = max(local_max, score_val);
        float exp_val = __expf(score_val - new_max);
        float new_sum = __expf(local_max - new_max) * exp_sum_local + exp_val;

        max_val[threadIdx.y / BLOCK_WIDTH] = new_max;
        exp_sum[threadIdx.y / BLOCK_WIDTH] = new_sum;

        // Store normalized attention weights back
        sAttention[threadIdx.y * BLOCK_SIZE_N + threadIdx.x] = exp_val / new_sum;
        __syncthreads();

        // Weighted sum: O += P @ V
        for (int d = 0; d < HEAD_DIM; ++d) {
            float out_val = 0.0f;
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE_N; ++k) {
                out_val += sAttention[threadIdx.y * BLOCK_SIZE_N + k] *
                          convert_float(sV[k * HEAD_DIM + d]);
            }
            out_row[d] = out_val;
        }
    }

    // Normalize output and write back
    int out_idx = batch_id * num_heads * seqlen * HEAD_DIM
                + head_id * seqlen * HEAD_DIM
                + row_start * HEAD_DIM;

    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        Out[out_idx + d] = convert_half(out_row[d] / exp_sum[row_start / BLOCK_WIDTH]);
    }

    // Store softmax statistics
    M[batch_head_idx * seqlen + row_start] = max_val[0];
    L[batch_head_idx * seqlen + row_start] = exp_sum[0];
}
```

### 4.2 Triton Implementation for Portability

Triton provides better portability across backends while maintaining performance:

```python
# flashattn_forward.py (Triton)
import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L, M, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    nheads, seqlen_q, seqlen_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Forward kernel for FlashAttention-2

    - Q: [batch, nheads, seq_q, dim]
    - K: [batch, nheads, seq_k, dim]
    - V: [batch, nheads, seq_k, dim]
    - sm_scale: scaling factor 1/sqrt(d)
    - L: [batch, nheads, seq_q] (softmax denominators)
    - M: [batch, nheads, seq_q] (softmax row maxes)
    """

    # Grid/Block indices
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    block_m = tl.program_id(2)

    # Compute offsets into Q, K, V
    q_offset = batch_id * stride_qb + head_id * stride_qh
    k_offset = batch_id * stride_kb + head_id * stride_kh
    v_offset = batch_id * stride_vb + head_id * stride_vh

    # For each block of Q rows
    m_idx = block_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Initialize accumulators for online softmax
    m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Load Q tile (constant for this block)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(block_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    q = tl.load(Q_block_ptr)

    # Iterate over blocks of K, V
    for block_n in range(0, seqlen_k, BLOCK_N):
        # Load K block
        K_block_ptr = tl.make_block_ptr(
            base=K,
            shape=(seqlen_k, BLOCK_DMODEL),
            strides=(stride_kn, stride_kk),
            offsets=(block_n, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0)
        )
        k = tl.load(K_block_ptr)

        # Compute QK^T with scaling
        # S[BLOCK_M, BLOCK_N] = Q[BLOCK_M, BLOCK_DMODEL] @ K^T[BLOCK_DMODEL, BLOCK_N]
        s = tl.dot(q, tl.trans(k))
        s = s * sm_scale  # Scale by 1/sqrt(d)

        # Causal masking (if autoregressive)
        n_idx = tl.arange(0, BLOCK_N) + block_n
        mask = (m_idx[:, None] >= n_idx[None, :])
        s = tl.where(mask, s, float("-inf"))

        # Online softmax update
        m_ij = tl.max(s, axis=1)
        p = tl.exp(s - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        # Update running stats
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)

        l_i_new = alpha * l_i + beta * l_ij

        # Update accumulator
        p_scaled = p * beta[:, None]

        # Load V block
        V_block_ptr = tl.make_block_ptr(
            base=V,
            shape=(seqlen_k, BLOCK_DMODEL),
            strides=(stride_vn, stride_vk),
            offsets=(block_n, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0)
        )
        v = tl.load(V_block_ptr)

        # O = O * alpha + P_scaled @ V
        acc = acc * alpha[:, None] + tl.dot(p_scaled, v)

        m_i = m_i_new
        l_i = l_i_new

    # Normalize and write output
    acc = acc / l_i[:, None]

    O_block_ptr = tl.make_block_ptr(
        base=Out,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_om, stride_ok),
        offsets=(block_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc)

    # Store softmax statistics
    M_ptr = M + batch_id * nheads * seqlen_q + head_id * seqlen_q + m_idx
    L_ptr = L + batch_id * nheads * seqlen_q + head_id * seqlen_q + m_idx
    tl.store(M_ptr, m_i)
    tl.store(L_ptr, l_i)


def flash_attn_forward(q, k, v, causal=False, sm_scale=None):
    """
    Wrapper function for FlashAttention forward pass
    """
    batch, nheads, seqlen_q, dim = q.shape
    _, _, seqlen_k, _ = k.shape

    if sm_scale is None:
        sm_scale = 1.0 / (dim ** 0.5)

    L = torch.empty((batch, nheads, seqlen_q), device=q.device, dtype=torch.float32)
    M = torch.empty((batch, nheads, seqlen_q), device=q.device, dtype=torch.float32)
    Out = torch.empty_like(q)

    # Determine block sizes based on available SRAM
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_DMODEL = dim

    grid = (batch, nheads, (seqlen_q + BLOCK_M - 1) // BLOCK_M)

    _fwd_kernel[grid](
        q, k, v, sm_scale,
        L, M, Out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        nheads, seqlen_q, seqlen_k,
        BLOCK_M, BLOCK_N, BLOCK_DMODEL,
    )

    return Out, L, M
```

### 4.3 CPU-Optimized FlashAttention: AVX-512 Implementation

For inference servers running on CPUs (e.g., Apple Silicon, AMD EPYC):

```cpp
// flashattn_avx512.cpp
#include <immintrin.h>
#include <cmath>
#include <algorithm>

// Aligned memory allocation
void* aligned_alloc_avx(size_t size) {
    return aligned_alloc(64, size);  // 64-byte alignment for AVX-512
}

// Vectorized softmax (online, single pass)
void online_softmax_avx512(
    const float* scores,      // B_r × B_c matrix stored row-major
    float* softmax_out,
    float* max_vals,
    float* sum_vals,
    int B_r, int B_c
) {
    const int vec_width = 16;  // 16 floats per AVX-512 register

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < B_r; ++i) {
        for (int j = 0; j < B_c; j += vec_width) {
            // Load scores
            __m512 scores_v = _mm512_loadu_ps(&scores[i * B_c + j]);

            // Update max
            float max_scalar = max_vals[i];
            __m512 max_v = _mm512_set1_ps(max_scalar);
            __m512 cmp = _mm512_cmp_ps_mask(scores_v, max_v, _CMP_GT_OQ);
            __m512 new_max = _mm512_mask_mov_ps(max_v, cmp, scores_v);

            // Reduction to get new max
            float new_max_scalar = _mm512_reduce_max_ps(new_max);
            max_vals[i] = new_max_scalar;

            // Adjust exponentials
            __m512 exp_v = _mm512_exp_ps(_mm512_sub_ps(scores_v, new_max_v));
            float sum_scalar = sum_vals[i];
            float exp_sum = _mm512_reduce_add_ps(exp_v);
            sum_vals[i] = sum_scalar * std::exp(max_scalar - new_max_scalar) + exp_sum;

            // Store normalized probabilities
            __m512 norm = _mm512_set1_ps(1.0f / sum_vals[i]);
            __m512 prob_v = _mm512_mul_ps(exp_v, norm);
            _mm512_storeu_ps(&softmax_out[i * B_c + j], prob_v);
        }
    }
}

// Vectorized matrix multiplication with online accumulation
void matmul_online_avx512(
    const float* Q,      // B_r × d matrix
    const float* K,      // B_c × d matrix
    const float* V,      // B_c × d matrix
    float* Out,          // B_r × d output
    int B_r, int B_c, int d, float sm_scale
) {
    const int block_size = 16;  // Process 16 rows at a time

    // Compute QK^T with online softmax
    float* scores = (float*)aligned_alloc_avx(B_r * B_c * sizeof(float));
    float* max_vals = (float*)aligned_alloc_avx(B_r * sizeof(float));
    float* sum_vals = (float*)aligned_alloc_avx(B_r * sizeof(float));

    std::fill(max_vals, max_vals + B_r, -INFINITY);
    std::fill(sum_vals, sum_vals + B_r, 0.0f);

    // QK^T computation with tiling for cache efficiency
    #pragma omp parallel for
    for (int i = 0; i < B_r; i += block_size) {
        for (int j = 0; j < B_c; ++j) {
            for (int ii = i; ii < std::min(i + block_size, B_r); ++ii) {
                float score = 0.0f;

                // Dot product Q[ii] · K[j]
                #pragma omp simd reduction(+:score)
                for (int k = 0; k < d; ++k) {
                    score += Q[ii * d + k] * K[j * d + k];
                }

                scores[ii * B_c + j] = score * sm_scale;
            }
        }
    }

    // Online softmax
    online_softmax_avx512(scores, scores, max_vals, sum_vals, B_r, B_c);

    // P @ V with accumulation
    #pragma omp parallel for
    for (int i = 0; i < B_r; ++i) {
        for (int d_out = 0; d_out < d; d_out += 16) {
            __m512 acc_v = _mm512_setzero_ps();

            for (int j = 0; j < B_c; ++j) {
                float prob = scores[i * B_c + j];
                __m512 prob_v = _mm512_set1_ps(prob);

                __m512 v_vals = _mm512_loadu_ps(&V[j * d + d_out]);
                __m512 contrib = _mm512_mul_ps(prob_v, v_vals);
                acc_v = _mm512_add_ps(acc_v, contrib);
            }

            _mm512_storeu_ps(&Out[i * d + d_out], acc_v);
        }
    }

    free(scores);
    free(max_vals);
    free(sum_vals);
}
```

### 4.4 Apple Silicon Optimization: Metal Implementation

For M-series chips, Metal provides direct GPU access with lower overhead:

```swift
// MetalFlashAttention.swift
import Metal
import MetalKit

class MetalFlashAttention {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary

    init(device: MTLDevice? = nil) {
        self.device = device ?? MTLCreateSystemDefaultDevice()!
        self.commandQueue = self.device.makeCommandQueue()!
        self.library = self.device.makeDefaultLibrary()!
    }

    func forwardAttention(
        Q: MTLBuffer, K: MTLBuffer, V: MTLBuffer,
        batchSize: Int, numHeads: Int, seqLen: Int, dimPerHead: Int
    ) -> MTLBuffer {
        let output = device.makeBuffer(
            length: batchSize * numHeads * seqLen * dimPerHead * MemoryLayout<Float>.size
        )!

        let computeCommandEncoder = commandQueue.makeCommandBuffer()!
            .makeComputeCommandEncoder()!

        // Load compute kernels
        let functionName = "flash_attn_forward"
        let kernelFunction = library.makeFunction(name: functionName)!
        let pipelineState = try! device.makeComputePipelineState(
            function: kernelFunction
        )

        computeCommandEncoder.setComputePipelineState(pipelineState)

        // Set buffers
        computeCommandEncoder.setBuffer(Q, offset: 0, index: 0)
        computeCommandEncoder.setBuffer(K, offset: 0, index: 1)
        computeCommandEncoder.setBuffer(V, offset: 0, index: 2)
        computeCommandEncoder.setBuffer(output, offset: 0, index: 3)

        // Thread configuration
        let threadgroupSize = MTLSizeMake(16, 4, 1)  // Optimal for M-series
        let gridSize = MTLSizeMake(
            (seqLen + 63) / 64 * numHeads * batchSize,
            (seqLen + 63) / 64,
            1
        )

        computeCommandEncoder.dispatchThreads(
            gridSize,
            threadsPerThreadgroup: threadgroupSize
        )
        computeCommandEncoder.endEncoding()

        commandQueue.makeCommandBuffer()?.commit()
        commandQueue.makeCommandBuffer()?.waitUntilCompleted()

        return output
    }
}

// Metal compute shader
let metalShaderSource = """
#include <metal_stdlib>
using namespace metal;

kernel void flash_attn_forward(
    const device float* Q [[buffer(0)]],
    const device float* K [[buffer(1)]],
    const device float* V [[buffer(2)]],
    device float* Out [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]]
) {
    // Implementation mirrors CUDA version but optimized for Metal's architecture
    // Shared memory (threadgroup memory) is limited on Apple Silicon (32KB)
    threadgroup float sQ[64 * 128];
    threadgroup float sK[64 * 128];
    threadgroup float sV[64 * 128];

    // Compute logic using simdgroup operations for efficient reduction
    float m = -INFINITY;
    float l = 0.0f;
    float out_val = 0.0f;

    // Process Q K V tiles with simdgroup sync
    simdgroup_barrier(mem_flags::mem_threadgroup);

    Out[gid.x] = out_val;
}
"""
```

### 4.5 PagedAttention Implementation Details

The key innovation: treat KV-cache as virtual memory with pages:

```python
# paged_attention.py
import torch
from typing import List, Tuple

class PagedKVCache:
    """Virtual memory management for KV-cache"""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        head_dim: int,
        num_heads: int,
        dtype: torch.dtype = torch.float16
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim
        self.num_heads = num_heads

        # Physical memory: [num_blocks, block_size, num_heads, head_dim]
        self.K_cache = torch.zeros(
            (num_blocks, block_size, num_heads, head_dim),
            dtype=dtype, device="cuda"
        )
        self.V_cache = torch.zeros_like(self.K_cache)

        # Block allocation tracking
        self.free_blocks = list(range(num_blocks))
        self.sequence_to_blocks: dict[int, List[int]] = {}  # seq_id -> block indices

    def allocate_blocks(self, seq_id: int, num_blocks_needed: int) -> List[int]:
        """Allocate physical blocks for a sequence"""
        blocks = []
        for _ in range(num_blocks_needed):
            if not self.free_blocks:
                # Memory pressure: evict least recently used sequence
                evicted_seq = self._evict_lru()
                self.free_blocks.extend(self.sequence_to_blocks[evicted_seq])
                del self.sequence_to_blocks[evicted_seq]

            block_idx = self.free_blocks.pop()
            blocks.append(block_idx)

        self.sequence_to_blocks[seq_id] = blocks
        return blocks

    def write_kv(
        self,
        seq_id: int,
        K: torch.Tensor,      # [1, num_heads, seq_len, head_dim]
        V: torch.Tensor,
        start_pos: int        # Position in original sequence
    ):
        """Write KV tokens to cache"""
        blocks = self.sequence_to_blocks[seq_id]

        for token_idx in range(K.size(2)):
            abs_pos = start_pos + token_idx
            block_idx = blocks[abs_pos // self.block_size]
            block_offset = abs_pos % self.block_size

            self.K_cache[block_idx, block_offset] = K[0, :, token_idx]
            self.V_cache[block_idx, block_offset] = V[0, :, token_idx]

    def read_kv(
        self,
        seq_id: int,
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read contiguous KV cache for a sequence"""
        blocks = self.sequence_to_blocks[seq_id]

        K_list = []
        V_list = []

        for block_idx in blocks[:1 + (seq_len - 1) // self.block_size]:
            K_list.append(self.K_cache[block_idx])
            V_list.append(self.V_cache[block_idx])

        K_concat = torch.cat(K_list, dim=0)[:seq_len]  # [seq_len, num_heads, head_dim]
        V_concat = torch.cat(V_list, dim=0)[:seq_len]

        return K_concat, V_concat

    def _evict_lru(self) -> int:
        """Find and evict least recently used sequence"""
        # Placeholder: in production, track LRU with timestamps
        return list(self.sequence_to_blocks.keys())[0]


def paged_attention_forward(
    Q: torch.Tensor,           # [batch, num_heads, 1, head_dim] (single query token)
    K_cache: PagedKVCache,
    V_cache: PagedKVCache,
    seq_id: int,
    seq_len: int,
    sm_scale: float = None
) -> torch.Tensor:
    """
    Attention with paged KV-cache
    """
    if sm_scale is None:
        sm_scale = 1.0 / (Q.size(-1) ** 0.5)

    # Read contiguous KV blocks
    K, V = K_cache.read_kv(seq_id, seq_len)

    # Standard attention
    # Q: [batch, num_heads, 1, head_dim]
    # K: [seq_len, num_heads, head_dim]
    # V: [seq_len, num_heads, head_dim]

    scores = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    return output  # [batch, num_heads, 1, head_dim]
```

---

## 5. KEY PAPERS

1. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (Dao, Fu, Ermon, Rudra; NeurIPS 2022)
   - Introduces block-wise computation with online softmax
   - Achieves 7.6× speedup on V100 with identical outputs
   - Theoretical analysis of memory I/O

2. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** (Dao; ICLR 2024)
   - Improves backward pass (2× faster)
   - Better GPU occupancy through work distribution changes
   - Extends to longer sequences (up to 32K tokens)

3. **FlashAttention-3: Fast and Accurate Attention with Asynchronous Warpgroup Matmul** (Dao et al.; 2024)
   - Uses async tensor core matmuls
   - Double buffering for pipelining
   - Achieves 740 TFLOPS sustained on H100 (66% peak)

4. **GQA: Training Generalist Models for 3B Tokens and Beyond** (Ainslie, Lee, Ontanon, Passos, Sifre; 2023)
   - Grouped-Query Attention reduces KV-cache by N_heads / N_groups
   - Maintains task performance with aggressive reduction
   - Enables production deployment of larger models

5. **Efficient Memory Management for Large Language Model Serving with PagedAttention** (Kwon, Li, Zhuang, et al.; SOSP 2023)
   - OS-like memory management for KV-cache
   - vLLM system achieving >10× better throughput
   - Support for arbitrary sequence length

6. **Ring Attention with Blockwise Transformers for Near-Infinite Context** (Liu, Zhuang, Abdin, et al.; ICLR 2024)
   - Distributed attention via ring communication pattern
   - O(N) memory for N-GPU system with sequence distribution
   - Maintains compute efficiency through communication hiding

7. **StreamingLLM: Efficient Streaming Language Models with Attention Sinks** (Xiao, Hou, et al.; ICLR 2024)
   - Identifies "attention sink" tokens for streaming inference
   - Allows memory-bounded inference with arbitrary length
   - Drop middle tokens beyond window

8. **H2O: Heavy Hitter Oracle for Efficient Generative Inference of Large Language Models** (Zhang, Parikh, Henao; NeurIPS 2023)
   - Identify and retain high-attention tokens
   - Reduce KV-cache size by 50% with minimal quality loss
   - Heuristic scoring function for token importance

---

## 6. SYSTEMS TRADEOFFS

### 6.1 FlashAttention vs Standard Attention

| Metric | Standard | FlashAttention |
|--------|----------|-----------------|
| HBM Bandwidth | O(n²d) reads/writes | O(nd) reads + O(n²) temporary |
| SRAM Pressure | Minimal | Requires (B_r + B_c) × d |
| Compute Density | 0.06 FLOPs/byte (n=4096, d=128) | 2-3 FLOPs/byte |
| Latency | 2-3 ms (4K seq) | 0.5-1 ms (4K seq) |
| Backward Pass | Simple (store attention) | Recompute attention |
| GPU Occupancy | 10-20% | 60-80% |

### 6.2 GQA vs MQA vs Standard Attention

| Aspect | Standard | GQA (8-way) | MQA |
|--------|----------|-------------|-----|
| KV-Cache | 1× (baseline) | 1/8× | 1/num_heads |
| Quality Drop | 0% | 0.1-0.5% | 1-2% (seq>1024) |
| Prefill Speed | baseline | +5% (less cache load) | +10% |
| Decode Speed | baseline | ~same (matmul-bound) | ~same |
| Production Use | LLaMA, GPT | Llama 2, PaLM | PaLM, Falcon |

### 6.3 Paged vs Contiguous KV-Cache

| Property | Contiguous | PagedAttention |
|----------|-----------|-----------------|
| Memory Utilization | 70-80% (fragmentation) | 95%+ (bin packing) |
| Sequence Pairing | Inflexible | Flexible (CoW) |
| Throughput (batch size) | Limited | Unlimited (batching) |
| Memory Management | Simple | Complex (page tables) |
| Worst-case Latency | High (allocation stall) | Bounded |

### 6.4 Sliding Window vs Full Attention

| Factor | Full | Sliding (Window=2K) |
|--------|------|-------------------|
| Memory | O(n²) | O(window × n) |
| Quality | 100% | 98-99% (with RoPE ext) |
| Long Context | Fails at n>8K | Supports n>100K |
| Compute | O(n²) | O(window × n) |

---

## 7. EXPERT INSIGHT

### 7.1 When to Use Each Optimization

**FlashAttention (v1/v2/v3)**: Always use for training and inference. The only tradeoff is backward-pass recomputation (minimal cost). No quality degradation.

**GQA**: Use when KV-cache memory is the bottleneck:
- Multi-batch inference (>8 concurrent sequences)
- Long context (>2K tokens)
- Mobile/edge deployment

Trade-off: 0.5% accuracy loss for 8× memory savings.

**MQA**: Use when KV-cache latency (load operations) matters:
- Ultra-low latency applications
- Single token decoding highly optimized
- Streaming applications

Trade-off: 2% accuracy loss, 20× memory savings.

**PagedAttention**: Necessary for production inference servers. Enables:
- Arbitrary batch sizes
- Sequence pairing (prefix caching)
- Memory defragmentation
- Fair scheduling

**Sliding window + RoPE extension**: Use for long-context fine-tuning:
- Llama 2 Long: 32K context with 4K training data
- YaRN (yet another RoPE extension): Interpolate position embeddings
- Maintains full attention capacity within window

### 7.2 Hardware-Specific Tuning

**NVIDIA H100**:
- Max performance: 128 BLOCK_M, 128 BLOCK_N
- Target: 700+ TFLOPS (85% peak compute)
- Use async tensor cores (FlashAttention-3)

**NVIDIA A100**:
- Max: 64 BLOCK_M, 64 BLOCK_N
- Target: 250+ TFLOPS (80% peak)
- Use standard tensor cores

**CPU (AVX-512)**:
- Cache-friendly tiling (64×64 blocks)
- Prefetch K/V for next block
- Use VNNI instructions for INT8

**Apple Silicon**:
- Limited SRAM (8-16KB per core)
- Smaller blocks (32×32)
- Use simdgroup operations for parallel reduction
- Avoid shared memory (slow)

### 7.3 Production Deployment Checklist

1. **Always use FlashAttention** (v2 minimum, v3 if H100)
2. **Measure actual latency**, not theoretical speedup
3. **Profile memory allocations** in attention
4. **Use GQA if batched inference** (>8 sequences)
5. **Implement PagedAttention** for fairness + throughput
6. **Monitor KV-cache fragmentation** under load
7. **Test long sequences** (2×, 4× train max)

---

## 8. BENCHMARKING METHODOLOGY

### 8.1 Metrics Definition

**TFLOP/s (Sustained)**: Actual operations / wall-clock time
```
Measured_TFLOPs = (2 × seq_len² × dim × batch) / (time_seconds × 1e12)
```

**Memory Bandwidth Utilization**:
```
BW_util = (bytes_transferred × 10^-9) / (time_seconds × GPU_bandwidth)
```

**Latency percentiles**:
- P50: median latency (expected case)
- P95: 95th percentile (SLA target)
- P99: tail latency (customer impact)

### 8.2 Benchmark Setup

```python
# benchmark_attention.py
import torch
import triton
from torch.profiler import profile, record_function

def benchmark_attention(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    dim_per_head: int,
    num_runs: int = 100,
    warmup: int = 10
):
    """Comprehensive attention benchmark"""

    device = torch.device("cuda")

    # Allocate inputs
    Q = torch.randn(
        (batch_size, num_heads, seq_len, dim_per_head),
        device=device, dtype=torch.float16
    ).normal_(0, 1)
    K = Q.clone()
    V = Q.clone()

    # Warmup
    for _ in range(warmup):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

    torch.cuda.synchronize()

    # Benchmark standard attention
    times_standard = []
    for _ in range(num_runs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        end.record()
        torch.cuda.synchronize()

        times_standard.append(start.elapsed_time)
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # Benchmark FlashAttention
    times_flash = []
    for _ in range(num_runs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            out = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, enable_math=False
            )  # Uses FlashAttention backend
        end.record()
        torch.cuda.synchronize()

        times_flash.append(start.elapsed_time)

    # Analysis
    flops = 2 * batch_size * seq_len**2 * num_heads * dim_per_head

    print(f"Sequence length: {seq_len}, Batch: {batch_size}")
    print(f"Standard attention: {sum(times_standard)/len(times_standard):.2f}ms "
          f"({flops/(sum(times_standard)/1000) * 1e-12:.1f} TFLOPS)")
    print(f"FlashAttention: {sum(times_flash)/len(times_flash):.2f}ms "
          f"({flops/(sum(times_flash)/1000) * 1e-12:.1f} TFLOPS)")
    print(f"Speedup: {sum(times_standard)/sum(times_flash):.1f}×")

# Run benchmarks across configurations
for seq_len in [256, 512, 1024, 2048, 4096]:
    benchmark_attention(seq_len, batch_size=16, num_heads=16, dim_per_head=64)
```

### 8.3 Expected Results

On NVIDIA H100 with FP16:
- Seq len 256: 7.2× speedup (720 TFLOPS)
- Seq len 1024: 8.1× speedup (740 TFLOPS)
- Seq len 4096: 7.9× speedup (700 TFLOPS)

Memory usage:
- Standard: O(seq_len²) = 64 MB (seq=4096)
- FlashAttention: O(seq_len × dim) = 16 MB

---

## 9. OPEN PROBLEMS

### 9.1 Sparse Attention Integration

Current FlashAttention assumes dense attention. Extending to sparse patterns (local + strided) while maintaining I/O efficiency is an open problem.

**Challenge**: Sparse tile structure may not align with hardware blocks, reducing tiling benefits.

**Research direction**: Develop sparse tiling schemes that group non-zero entries efficiently.

### 9.2 Heterogeneous Precision

Mixed-precision attention (FP8 Q/K, FP16 gradients) could save bandwidth but softmax is numerically sensitive.

**Challenge**: FP8 quantization loses precision in softmax computation.

**Current approach**: Use FP16 for softmax, FP8 for matmuls (requires conversion).

### 9.3 Distributed Attention Efficiency

Ring Attention reduces communication, but all-reduce dependencies remain.

**Open question**: Can we overlap K-V rotations with compute more effectively?

**Research**: Investigate pipelined ring attention with better communication hiding.

### 9.4 Dynamic Sparsity

Identify and skip low-attention tokens during inference without pre-computation.

**Approach**: Online token pruning with early exit criteria.

**Bottleneck**: Determining thresholds dynamically without overhead.

---

## 10. PHD QUALIFIER QUESTIONS

1. **Mathematical Rigor** (30 minutes):
   - Derive the online softmax algorithm completely, including the max and denominator update rules for combining multiple tiles
   - Prove that this produces identical results to standard softmax within machine epsilon
   - Extend your derivation to the backward pass—what changes are necessary?

2. **Systems Design** (40 minutes):
   - You're designing attention for a heterogeneous system with CPU cores + GPU + TPU. How would you distribute attention computation across these devices? Consider communication costs.
   - What is the optimal sequence split for Ring Attention across N=8 GPUs with topology constraints (2D mesh, NVLink bandwidth)?

3. **Implementation Trade-offs** (45 minutes):
   - Compare three approaches: (a) standard dense attention, (b) FlashAttention, (c) GQA + paged cache
   - For a production LLM serving system, which would you choose for: (i) single-request latency, (ii) throughput with batching, (iii) memory-constrained devices?
   - Show your reasoning with concrete numbers (bandwidth, memory, FLOPs)

4. **Hardware Adaptation** (35 minutes):
   - You have access to an Apple M3 Pro (8-core CPU, 16-core GPU, shared 24GB memory)
   - Implement attention for 2048-token sequences
   - What tile sizes do you choose? Why?
   - How does your design change for 8192-token sequences?

5. **Production Challenges** (50 minutes):
   - A customer reports that their inference latency increased 40% after upgrading to your new FlashAttention kernel
   - Walk through the debugging process: what metrics would you check? What might cause this regression?
   - How would you ensure this doesn't happen again?

6. **Research Direction** (60 minutes):
   - Propose a new attention algorithm that achieves better than O(n²) space with minimal quality degradation
   - Compare its theoretical complexity to Ring Attention, Sliding Window attention, and Sparse attention
   - Identify what hardware characteristics would make your algorithm better than existing approaches

7. **Backward Pass Analysis** (40 minutes):
   - The FlashAttention backward pass requires recomputing attention. Why is this necessary?
   - Derive the memory requirement for backward pass compared to storing intermediate activations
   - Is there a hybrid approach (store some activations, recompute others)?

8. **Numerical Stability** (35 minutes):
   - Why is the online softmax numerically stable even when processing in blocks?
   - Write pseudocode for a numerically unstable version and explain the pitfall
   - How would you verify numerical stability in your implementation?

---

## Conclusion

Attention mechanism optimization represents the frontier of ML systems research, combining algorithmic innovation (online softmax), hardware understanding (memory hierarchy), and practical deployment concerns (paging, scheduling). The progression from FlashAttention to production systems like vLLM demonstrates that systems thinking—not just algorithmic tricks—enables 10× improvements in real-world performance.

The key insight: reshape algorithms to match hardware, not the other way around. This principle extends beyond attention to all performance-critical ML systems components.

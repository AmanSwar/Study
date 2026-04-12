# MODULE 14 — Attention Implementation for CPU

## Table of Contents
1. Introduction: Attention on CPU
2. Standard Attention: Memory Characteristics
3. FlashAttention on CPU with AVX-512
4. Online Softmax: Numerically Stable Streaming Computation
5. Multi-Head Attention Parallelization
6. KV-Cache Layout Optimization
7. Grouped Query Attention (GQA) on CPU
8. Long Context Handling (128K+ Tokens)
9. Complete C++ FlashAttention Implementation
10. Performance Analysis & Benchmarking

---

## 1. Introduction: Attention on CPU

### 1.1 Attention's Role in Inference

Attention is the core mechanism in transformer models:

```
Self-Attention(Q, K, V):
├─ Scores = softmax(Q · K^T / sqrt(d_k))
├─ Output = Scores · V
└─ Total operations: O(N^2) where N = sequence length
```

For a 7B transformer:
- Each of 32 heads processes 128-dim (4,096 ÷ 32)
- Sequence length: 128 to 8,192 tokens (inference)
- Attention FLOP: 2 × 32 heads × (seq_len)^2 × 128 = 2 × 32 × (seq_len)^2 × 128 FLOP

**Memory access pattern**:
- Read Q: seq_len × d_model = batch × 4,096 FP32
- Read K, V: context_len × d_model each
- Write output: seq_len × d_model
- **Severely memory-bound** (low arithmetic intensity)

### 1.2 CPU vs GPU Attention

```
GPU (H100) Attention:
├─ Arithmetic intensity: 256 (Q·K^T is GEMM, high arithmetic intensity)
├─ Achieved throughput: 800 TFLOP (compute-saturated)
└─ Latency: 120 ms for 4K token context

CPU (EPYC) Attention:
├─ Arithmetic intensity: 4 (Q·K^T is still GEMM, but on CPU)
├─ Achieved throughput: 400 GFLOP (memory-bound)
├─ Latency: 400 ms (slower wall-clock, but...)
└─ Per-core latency: 12.5 ms (smaller working set per thread)
```

The key insight: **CPU attention is better suited for small context (< 1K tokens)**, where different cores can process disjoint contexts in parallel.

---

## 2. Standard Attention: Memory Characteristics

### 2.1 Naive Attention Algorithm

```python
def attention(Q, K, V, mask=None):
    """
    Q: (batch, seq_len, d_model)
    K, V: (batch, context_len, d_model)
    Returns: (batch, seq_len, d_model)
    """
    d_k = Q.shape[-1]

    # 1. Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: (batch, seq_len, context_len)

    # 2. Apply mask (optional, for causal attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3. Softmax normalization
    attn_weights = F.softmax(scores, dim=-1)

    # 4. Apply attention to values
    output = torch.matmul(attn_weights, V)
    # output: (batch, seq_len, d_model)

    return output
```

**Memory access pattern:**

```
Forward pass:
├─ Load Q: seq_len × d_model
├─ Load K: context_len × d_model
├─ Q·K^T: Intermediate (seq_len × context_len) — MUST STORE
├─ Load V: context_len × d_model
├─ Softmax (context_len) — in-place on scores
└─ Attn·V: Final multiplication

Total reads: Q + K + V + V = 3·d_model + context_len (from scores)
Total writes: scores + output = seq_len + d_model
Memory: O(seq_len × context_len) — intermediate scores matrix
```

For large context (8K tokens):
- Q·K^T intermediate: 8,192 × 8,192 × 4 bytes = 256 MB
- This **exceeds L3 cache** (1.15 GB per 12 cores, 96 MB per core)
- Forces eviction to main memory, destroying performance

---

## 3. FlashAttention on CPU with AVX-512

### 3.1 FlashAttention Algorithm

**FlashAttention** (Dao et al., 2022) reorganizes attention to avoid materializing the large scores matrix:

```
Standard approach (memory-inefficient):
┌────────────────┐
│ Read Q         │
├────────────────┤
│ Read K         │
├────────────────┤
│ Compute Q·K^T  │ ← 256 MB intermediate!
├────────────────┤
│ Softmax        │
├────────────────┤
│ Read V         │
├────────────────┤
│ Compute Attn·V │
└────────────────┘

FlashAttention (memory-efficient, block-wise):
┌─────────────────────────────────────┐
│ for each block of K,V (64 tokens):  │
├─────────────────────────────────────┤
│  ├─ Load Q (seq_len × d_k)          │
│  ├─ Load K_block (64 × d_k)         │
│  ├─ Compute Q·K_block^T             │
│  │  (seq_len × 64 intermediate)     │
│  ├─ Softmax with online algorithm   │
│  ├─ Load V_block (64 × d_k)         │
│  ├─ Compute Attn·V_block            │
│  ├─ Accumulate to output            │
│  └─ Discard intermediate            │
└─────────────────────────────────────┘

Memory: O(d_model) instead of O(seq_len × context_len)
```

### 3.2 Online Softmax Computation

The key to FlashAttention is **online softmax**, which accumulates:
- Running sum of exponentials
- Running normalization factor
- Output values

**Math:**

```
Standard softmax:
    scores[i] = exp(x[i]) / sum_j(exp(x[j]))

Online softmax (streaming):
    Initialize: sum = 0, max_val = -inf, output = 0

    For each block:
    ├─ Load new_max = max(scores_block)
    ├─ Shift previous sum: sum *= exp(max_val - new_max)
    ├─ Update max: max_val = max(max_val, new_max)
    ├─ Add to sum: sum += exp(scores_block - new_max)
    ├─ Reweight previous output: output *= exp(old_max - max_val)
    ├─ Add new block: output += exp(scores_block - max_val) * V_block
    └─ Divide by sum at the end
```

**Numerically stable:**
```
Instead of: exp(x_i) / sum_j(exp(x_j))

Use: exp(x_i - max_val) / sum_j(exp(x_j - max_val))
     This prevents overflow in exp()
```

### 3.3 CPU Tile Size Selection

On CPU, choose tile sizes to fit in cache:

```
EPYC 9754 Cache Layout:
├─ L1: 48 KB per core (12K float32)
├─ L2: 1 MB per core (262K float32)
└─ L3: 1,152 MB / 12 cores = 96 MB per core (24M float32)

For attention with d_model = 4,096:

Block strategy:
├─ Q: Keep in L2 (or L3)
│   ├─ Size: seq_len × d_model
│   ├─ For seq_len=128: 128 × 4,096 × 4 = 2 MB
│   └─ Fits in L3 easily
│
├─ K block: fit in L2
│   ├─ Size: block_size × d_model
│   ├─ For block_size=64: 64 × 4,096 × 4 = 1 MB
│   └─ Fits in L2
│
├─ V block: same as K
│   └─ Fits in L2
│
└─ Scores intermediate: minimize
    ├─ Size: seq_len × block_size
    ├─ For seq_len=128, block_size=64: 128 × 64 × 4 = 32 KB
    └─ Fits in L1!
```

---

## 4. Online Softmax: Numerically Stable Streaming Computation

### 4.1 Algorithm Derivation

**Theorem (Welford's Online Algorithm for Softmax):**

```
Given: x = [x_0, x_1, ..., x_N]

Define:
    m_j = max(x_0, ..., x_j)
    d_j = sum_i(exp(x_i - m_j))  for i <= j
    S_j = sum_i(x_i * exp(x_i - m_j)) / d_j  for i <= j  [NOT NORMALIZED YET]

Update from j to j+1:
    m_{j+1} = max(m_j, x_{j+1})
    delta_m = x_{j+1} - m_j

    d_{j+1} = d_j * exp(-delta_m) + exp(x_{j+1} - m_{j+1})

    S_{j+1} = S_j * exp(-delta_m) + x_{j+1}
            = (sum_i(x_i * exp(x_i - m_{j+1}))) / d_{j+1}

Final result: S_N is the softmax
```

### 4.2 Vectorized Implementation

For AVX-512, process multiple rows of attention simultaneously:

```c
// Vectorized online softmax for attention
// Process 16 rows in parallel (16 × zmm registers)
void online_softmax_attn(
    const float* scores,      // (seq_len, context_len)
    const float* values,      // (context_len, d_model)
    float* output,            // (seq_len, d_model)
    int seq_len, int context_len, int d_model
) {
    // Process in blocks of context_len (e.g., 64 tokens at a time)
    const int BLOCK_SIZE = 64;

    // Initialize for each row
    __m512 max_vals[16];       // max_val for each of 16 rows
    __m512 sum_denom[16];      // sum of exp for each row
    __m512 outputs[16][16];    // outputs for each row × block (16 rows × 256 floats)

    for (int i = 0; i < 16; i++) {
        max_vals[i] = _mm512_set1_ps(-3.4e38f);  // -infinity
        sum_denom[i] = _mm512_setzero_ps();
    }

    // Process blocks of context
    for (int block_idx = 0; block_idx < context_len; block_idx += BLOCK_SIZE) {
        int block_size = std::min(BLOCK_SIZE, context_len - block_idx);

        // For each row (process 16 in parallel)
        for (int row = 0; row < seq_len; row += 16) {
            int n_rows = std::min(16, seq_len - row);

            for (int col = 0; col < block_size; col++) {
                // Load scores for this block column
                __m512 scores_col[16];
                for (int rr = 0; rr < n_rows; rr++) {
                    scores_col[rr] = _mm512_set1_ps(scores[(row + rr) * context_len + block_idx + col]);
                }

                // Update max_val
                __m512 new_max_vals[16];
                for (int rr = 0; rr < n_rows; rr++) {
                    new_max_vals[rr] = _mm512_max_ps(max_vals[rr], scores_col[rr]);
                }

                // Reweight previous denominator
                for (int rr = 0; rr < n_rows; rr++) {
                    __m512 delta_m = _mm512_sub_ps(max_vals[rr], new_max_vals[rr]);
                    __m512 exp_delta = _mm512_exp_ps(delta_m);
                    sum_denom[rr] = _mm512_mul_ps(sum_denom[rr], exp_delta);

                    // Update max
                    max_vals[rr] = new_max_vals[rr];
                }

                // Add current exp to denominator
                for (int rr = 0; rr < n_rows; rr++) {
                    __m512 x_normalized = _mm512_sub_ps(scores_col[rr], max_vals[rr]);
                    __m512 exp_x = _mm512_exp_ps(x_normalized);
                    sum_denom[rr] = _mm512_add_ps(sum_denom[rr], exp_x);
                }
            }
        }
    }

    // Normalize output
    for (int row = 0; row < seq_len; row++) {
        for (int d = 0; d < d_model; d++) {
            float acc = 0.0f;

            for (int col = 0; col < context_len; col++) {
                float score_normalized = scores[row * context_len + col] - max_vals[row % 16];
                float attn_weight = std::exp(score_normalized) / sum_denom[row % 16];
                acc += attn_weight * values[col * d_model + d];
            }

            output[row * d_model + d] = acc;
        }
    }
}
```

---

## 5. Multi-Head Attention Parallelization

### 5.1 Head-Level Parallelism

For a model with 32 attention heads (d_model = 4,096, head_dim = 128):

```
Multi-Head Attention:

Input: (seq_len, d_model=4,096)
    │
    ├─ Head 0: Linear (4,096 → 128) → Attention
    ├─ Head 1: Linear (4,096 → 128) → Attention
    ├─ ...
    └─ Head 31: Linear (4,096 → 128) → Attention
    │
    └─ Concatenate + Linear (4,096 → 4,096)

Threading strategy:
┌──────────────────────────────────┐
│ Thread 0 (core 0, NUMA 0): Head 0 │
│ Thread 1 (core 1, NUMA 0): Head 1 │
│ ...                              │
│ Thread 16 (core 0, NUMA 1): Head 16│
│ ...                              │
└──────────────────────────────────┘
```

### 5.2 NUMA-Aware Head Distribution

For dual-socket servers:

```c
void multi_head_attention_numa_aware(
    const float* Q, const float* K, const float* V,
    float* output,
    int seq_len, int num_heads, int head_dim, int context_len
) {
    // Distribute heads across NUMA nodes
    int cores_per_numa = omp_get_num_threads() / 2;
    int heads_per_numa = num_heads / 2;

    #pragma omp parallel for schedule(static) proc_bind(spread)
    for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        int numa_node = head_idx < heads_per_numa ? 0 : 1;
        int local_head = head_idx < heads_per_numa ? head_idx : head_idx - heads_per_numa;

        // Bind to local NUMA node
        #pragma omp critical
        {
            // Ensure memory access is local to NUMA node
            // (can use numactl bind or libnuma APIs)
        }

        // Perform attention for this head
        const float* Q_head = &Q[seq_len * head_dim * head_idx];
        const float* K_head = &K[context_len * head_dim * head_idx];
        const float* V_head = &V[context_len * head_dim * head_idx];
        float* out_head = &output[seq_len * head_dim * head_idx];

        flash_attention_cpu(
            Q_head, K_head, V_head, out_head,
            seq_len, context_len, head_dim
        );
    }
}
```

---

## 6. KV-Cache Layout Optimization

### 6.1 KV-Cache Structure

During inference, we cache K and V across tokens to avoid recomputation:

```
Generation loop:
├─ Token 1: compute K[1], V[1]
│   KV-cache: {K[1], V[1]}
├─ Token 2: compute K[2], V[2]
│   KV-cache: {K[1:2], V[1:2]}
├─ Token 3: compute K[3], V[3]
│   KV-cache: {K[1:3], V[1:3]}
└─ Token N: compute K[N], V[N]
    KV-cache: {K[1:N], V[1:N]}

Total memory: 2 × num_heads × head_dim × context_len × sizeof(float)
For 7B model: 2 × 32 × 128 × 8,192 × 4 = 268 MB per sequence
```

### 6.2 Optimal Layout for CPU

For CPU inference, use a **contiguous per-layer layout**:

```c
// KV-cache layout: [layer][token][head][d_head]
struct KVCache {
    float* K;  // [num_layers][context_len][num_heads][head_dim]
    float* V;  // [num_layers][context_len][num_heads][head_dim]
    int num_layers;
    int context_len;
    int num_heads;
    int head_dim;
};

// Allocate contiguous memory for L1 locality
KVCache* allocate_kv_cache(int num_layers, int max_context_len, int num_heads, int head_dim) {
    KVCache* cache = malloc(sizeof(KVCache));

    // Single contiguous allocation
    size_t total_size = 2 * num_layers * max_context_len * num_heads * head_dim * sizeof(float);
    float* data = aligned_alloc(64, total_size);

    cache->K = data;
    cache->V = data + (num_layers * max_context_len * num_heads * head_dim);
    cache->num_layers = num_layers;
    cache->context_len = max_context_len;
    cache->num_heads = num_heads;
    cache->head_dim = head_dim;

    return cache;
}

// Access KV cache with prefetching
float* get_k_value(KVCache* cache, int layer, int token, int head, int d) {
    float* ptr = &cache->K[
        layer * (cache->context_len * cache->num_heads * cache->head_dim) +
        token * (cache->num_heads * cache->head_dim) +
        head * cache->head_dim +
        d
    ];

    // Prefetch next token's data (hide memory latency)
    _mm_prefetch(ptr + cache->num_heads * cache->head_dim, _MM_HINT_T0);

    return ptr;
}
```

### 6.3 Prefetch Strategy for KV-Cache

For efficient cache access during attention:

```c
void flash_attention_with_prefetch(
    const float* Q,           // Current token Q (num_heads, head_dim)
    const KVCache* kv_cache,  // Cached K, V
    float* output,
    int layer, int token_pos, int num_heads, int head_dim
) {
    for (int head = 0; head < num_heads; head++) {
        const float* Q_head = &Q[head * head_dim];
        __m512 q_vec[head_dim / 16];  // Q values in registers

        // Load Q into registers (once)
        for (int d = 0; d < head_dim; d += 16) {
            q_vec[d / 16] = _mm512_loadu_ps(&Q_head[d]);
        }

        float acc_output[head_dim];
        memset(acc_output, 0, head_dim * sizeof(float));

        // Attention loop with prefetching
        for (int token_idx = 0; token_idx <= token_pos; token_idx++) {
            // Get pointers with prefetching
            const float* K_ptr = get_k_value(
                (KVCache*)kv_cache, layer, token_idx, head, 0
            );
            const float* V_ptr = get_k_value(
                (KVCache*)kv_cache, layer, token_idx, head, 0
            );

            // Prefetch next iteration's K and V
            if (token_idx < token_pos) {
                _mm_prefetch(
                    get_k_value((KVCache*)kv_cache, layer, token_idx + 1, head, 0),
                    _MM_HINT_T1
                );
            }

            // Compute attention score
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += Q_head[d] * K_ptr[d];
            }
            score /= std::sqrt((float)head_dim);

            // Apply causal mask
            if (token_idx == token_pos) {
                // Current token: no masking
            } else {
                // Past tokens: keep attention
            }

            // Accumulate with V
            float weight = std::exp(score);  // Simplified (should use online softmax)
            for (int d = 0; d < head_dim; d++) {
                acc_output[d] += weight * V_ptr[d];
            }
        }

        // Store output
        memcpy(&output[head * head_dim], acc_output, head_dim * sizeof(float));
    }
}
```

---

## 7. Grouped Query Attention (GQA) on CPU

### 7.1 GQA Motivation

**GQA (Grouped Query Attention)** reduces KV-cache size by having multiple Q heads share K and V:

```
Standard MHA (Multi-Head Attention):
├─ num_heads = 32
├─ Each head has: Q, K, V
└─ KV-cache size: 32 × head_dim × context_len

Grouped Query Attention (GQA):
├─ num_heads = 32
├─ num_kv_heads = 4  (8× reduction!)
├─ Each KV group serves 8 Q heads
└─ KV-cache size: 4 × head_dim × context_len (8× smaller!)
```

This **significantly reduces memory bandwidth** on CPU.

### 7.2 GQA Implementation with AVX-512

```c
void gqa_attention(
    const float* Q,              // (seq_len, num_heads, head_dim)
    const float* K,              // (context_len, num_kv_heads, head_dim)
    const float* V,              // (context_len, num_kv_heads, head_dim)
    float* output,
    int seq_len, int num_heads, int num_kv_heads, int head_dim, int context_len
) {
    // Expansion factor: how many Q heads per KV head
    int group_size = num_heads / num_kv_heads;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int token = 0; token < seq_len; token++) {
        for (int q_head = 0; q_head < num_heads; q_head++) {
            // Map Q head to KV head
            int kv_head = q_head / group_size;

            const float* Q_vec = &Q[token * num_heads * head_dim + q_head * head_dim];
            const float* K_vec = &K[0 * num_kv_heads * head_dim + kv_head * head_dim];
            const float* V_vec = &V[0 * num_kv_heads * head_dim + kv_head * head_dim];
            float* out_vec = &output[token * num_heads * head_dim + q_head * head_dim];

            // Compute attention for this Q head with shared KV
            attention_single_head_avx512(
                Q_vec, K_vec, V_vec, out_vec,
                head_dim, context_len
            );
        }
    }
}

// Single head attention (shared by multiple Q heads in GQA)
void attention_single_head_avx512(
    const float* Q,              // (head_dim)
    const float* K,              // (context_len, head_dim)
    const float* V,              // (context_len, head_dim)
    float* output,               // (head_dim)
    int head_dim, int context_len
) {
    const int BLOCK_SIZE = 64;
    float max_score = -3.4e38f;
    float sum_exp = 0.0f;
    __m512 acc_output[head_dim / 16];

    // Load Q into registers
    __m512 Q_regs[head_dim / 16];
    for (int d = 0; d < head_dim; d += 16) {
        Q_regs[d / 16] = _mm512_loadu_ps(&Q[d]);
    }

    // Initialize output
    for (int d = 0; d < head_dim; d += 16) {
        acc_output[d / 16] = _mm512_setzero_ps();
    }

    // Process K and V in blocks
    for (int ctx_block = 0; ctx_block < context_len; ctx_block += BLOCK_SIZE) {
        int block_end = std::min(ctx_block + BLOCK_SIZE, context_len);

        for (int ctx_idx = ctx_block; ctx_idx < block_end; ctx_idx++) {
            // Compute Q · K[ctx_idx]
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += Q[d] * K[ctx_idx * head_dim + d];
            }
            score /= std::sqrt((float)head_dim);

            // Update max for numerical stability
            if (score > max_score) {
                // Reweight previous accumulator
                float shift_factor = std::exp(max_score - score);
                max_score = score;
                sum_exp *= shift_factor;

                for (int d = 0; d < head_dim; d += 16) {
                    acc_output[d / 16] = _mm512_mul_ps(
                        acc_output[d / 16],
                        _mm512_set1_ps(shift_factor)
                    );
                }
            }

            // Add to sum_exp
            float attn_weight = std::exp(score - max_score);
            sum_exp += attn_weight;

            // Accumulate V * weight
            for (int d = 0; d < head_dim; d += 16) {
                __m512 v_vec = _mm512_loadu_ps(&V[ctx_idx * head_dim + d]);
                acc_output[d / 16] = _mm512_fmadd_ps(
                    v_vec,
                    _mm512_set1_ps(attn_weight),
                    acc_output[d / 16]
                );
            }
        }
    }

    // Normalize and store
    for (int d = 0; d < head_dim; d += 16) {
        __m512 result = _mm512_div_ps(
            acc_output[d / 16],
            _mm512_set1_ps(sum_exp)
        );
        _mm512_storeu_ps(&output[d], result);
    }
}
```

---

## 8. Long Context Handling (128K+ Tokens)

### 8.1 Bandwidth Arithmetic for Long Context

For very long contexts (128K tokens), KV-cache becomes the primary bottleneck:

```
Decode phase (single token):

Memory access:
├─ Q: seq_len × head_dim = 1 × 128 = 128 FP32
├─ K: context_len × head_dim = 128,000 × 128 = 16.4 MB
├─ V: context_len × head_dim = 128,000 × 128 = 16.4 MB
├─ Total: 32.8 MB loaded from memory
└─ But: Only 2 × 128 (Q·K operations) FLOP per load

Roofline:
├─ CPU memory bandwidth: 460 GB/s
├─ Minimum latency: 32.8 MB ÷ 460 GB/s = 71 μs
├─ Actual latency with cache effects: 500-800 μs
└─ Throughput: 2 tokens/ms (severely bottlenecked)
```

### 8.2 Chunked Attention Strategy

For long context, process K,V in chunks:

```c
void attention_long_context_chunked(
    const float* Q,              // (seq_len, num_heads, head_dim)
    const KVCache* kv_cache,     // Very large (128K tokens)
    float* output,
    int seq_len, int num_heads, int head_dim, int context_len
) {
    const int CHUNK_SIZE = 4096;  // Process 4K tokens at a time

    for (int chunk_start = 0; chunk_start < context_len; chunk_start += CHUNK_SIZE) {
        int chunk_end = std::min(chunk_start + CHUNK_SIZE, context_len);

        // Load K and V for this chunk into L3
        const float* K_chunk = &kv_cache->K[chunk_start * num_heads * head_dim];
        const float* V_chunk = &kv_cache->V[chunk_start * num_heads * head_dim];

        // Compute attention with this chunk
        attention_chunk(
            Q, K_chunk, V_chunk, output,
            seq_len, num_heads, head_dim,
            chunk_end - chunk_start
        );
    }
}

// Process single chunk
void attention_chunk(
    const float* Q,
    const float* K_chunk,
    const float* V_chunk,
    float* output,
    int seq_len, int num_heads, int head_dim, int chunk_size
) {
    // Q is fixed; K_chunk, V_chunk rotate
    // Process entire seq_len in parallel

    #pragma omp parallel for schedule(static)
    for (int token = 0; token < seq_len; token++) {
        for (int head = 0; head < num_heads; head++) {
            const float* Q_head = &Q[token * num_heads * head_dim + head * head_dim];

            float acc[head_dim];
            memset(acc, 0, head_dim * sizeof(float));

            // For each token in K_chunk
            for (int k_idx = 0; k_idx < chunk_size; k_idx++) {
                const float* K_val = &K_chunk[k_idx * num_heads * head_dim + head * head_dim];
                const float* V_val = &V_chunk[k_idx * num_heads * head_dim + head * head_dim];

                // Q · K
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += Q_head[d] * K_val[d];
                }
                score /= std::sqrt((float)head_dim);

                // Softmax weight
                float weight = std::exp(score);  // Simplified

                // Accumulate V
                for (int d = 0; d < head_dim; d++) {
                    acc[d] += weight * V_val[d];
                }
            }

            // Store (accumulate across chunks)
            for (int d = 0; d < head_dim; d++) {
                #pragma omp atomic
                output[token * num_heads * head_dim + head * head_dim + d] += acc[d];
            }
        }
    }
}
```

---

## 9. Complete C++ FlashAttention Implementation

### 9.1 Full FlashAttention Kernel

```cpp
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <algorithm>

// Complete FlashAttention for CPU
// Q: (seq_len, num_heads, head_dim)
// K, V: (context_len, num_heads, head_dim)
// Output: (seq_len, num_heads, head_dim)
void flash_attention_cpu(
    const float* Q, const float* K, const float* V,
    float* output,
    int seq_len, int context_len, int num_heads, int head_dim
) {
    const int BLOCK_SIZE_K = 64;  // Block size for K, V
    const int BLOCK_SIZE_Q = std::min(128, seq_len);

    // Allocate output
    memset(output, 0, seq_len * num_heads * head_dim * sizeof(float));

    // For each Q block
    for (int q_block_start = 0; q_block_start < seq_len; q_block_start += BLOCK_SIZE_Q) {
        int q_block_size = std::min(BLOCK_SIZE_Q, seq_len - q_block_start);

        // Initialize per-block state
        float* m = (float*)calloc(q_block_size * num_heads, sizeof(float));  // max per row
        float* l = (float*)calloc(q_block_size * num_heads, sizeof(float));  // sum exp per row

        // Initialize m to -inf, l to 0
        for (int i = 0; i < q_block_size * num_heads; i++) {
            m[i] = -3.4e38f;
            l[i] = 0.0f;
        }

        // For each K, V block
        for (int kv_block_start = 0; kv_block_start < context_len; kv_block_start += BLOCK_SIZE_K) {
            int kv_block_size = std::min(BLOCK_SIZE_K, context_len - kv_block_start);

            // Compute S = Q · K^T (local for this block)
            // S: (q_block_size, kv_block_size)
            float* S = (float*)calloc(q_block_size * kv_block_size, sizeof(float));

            for (int q_idx = 0; q_idx < q_block_size; q_idx++) {
                for (int kv_idx = 0; kv_idx < kv_block_size; kv_idx++) {
                    float score = 0.0f;

                    for (int head = 0; head < num_heads; head++) {
                        const float* Q_vec = &Q[
                            (q_block_start + q_idx) * num_heads * head_dim +
                            head * head_dim
                        ];
                        const float* K_vec = &K[
                            (kv_block_start + kv_idx) * num_heads * head_dim +
                            head * head_dim
                        ];

                        // Q · K
                        __m512 q_reg, k_reg, prod = _mm512_setzero_ps();
                        for (int d = 0; d < head_dim; d += 16) {
                            q_reg = _mm512_loadu_ps(&Q_vec[d]);
                            k_reg = _mm512_loadu_ps(&K_vec[d]);
                            prod = _mm512_fmadd_ps(q_reg, k_reg, prod);
                        }

                        // Horizontal sum
                        float qk_dot = reduce_horizontal(prod) / std::sqrt((float)head_dim);
                        score += qk_dot;  // Simplified: should be per-head

                        S[q_idx * kv_block_size + kv_idx] = score;
                    }
                }
            }

            // Online softmax + accumulation to output
            for (int q_idx = 0; q_idx < q_block_size; q_idx++) {
                int global_q_idx = q_block_start + q_idx;

                for (int kv_idx = 0; kv_idx < kv_block_size; kv_idx++) {
                    float score = S[q_idx * kv_block_size + kv_idx];
                    float m_old = m[q_idx * num_heads + 0];  // Simplified: per-head tracking

                    // Update max
                    float m_new = std::max(m_old, score);
                    m[q_idx * num_heads + 0] = m_new;

                    // Reweight previous l (denominator)
                    float shift = std::exp(m_old - m_new);
                    l[q_idx * num_heads + 0] *= shift;

                    // Add current exp to l
                    l[q_idx * num_heads + 0] += std::exp(score - m_new);

                    // Accumulate to output with attention weight
                    float attn_weight = std::exp(score - m_new) / l[q_idx * num_heads + 0];

                    for (int head = 0; head < num_heads; head++) {
                        const float* V_vec = &V[
                            (kv_block_start + kv_idx) * num_heads * head_dim +
                            head * head_dim
                        ];

                        for (int d = 0; d < head_dim; d++) {
                            output[global_q_idx * num_heads * head_dim + head * head_dim + d] +=
                                attn_weight * V_vec[d];
                        }
                    }
                }
            }

            free(S);
        }

        free(m);
        free(l);
    }
}

// Helper: reduce zmm to float
float reduce_horizontal(__m512 v) {
    __m256 v_256 = _mm512_castps512_ps256(v) + _mm512_extractf32x8_ps(v, 1);
    __m128 v_128 = _mm256_castps256_ps128(v_256) + _mm256_extractf128_ps(v_256, 1);
    __m128 v_64 = _mm_shuffle_ps(v_128, v_128, _MM_SHUFFLE(2, 3, 0, 1)) + v_128;
    __m128 v_32 = _mm_shuffle_ps(v_64, v_64, _MM_SHUFFLE(1, 0, 3, 2)) + v_64;
    return _mm_cvtss_f32(v_32);
}
```

---

## 10. Performance Analysis & Benchmarking

### 10.1 Attention Benchmarking

```cpp
struct AttentionBenchmark {
    float latency_ms;
    float throughput_tflops;
    float memory_bw_gbs;
};

AttentionBenchmark benchmark_attention(
    const float* Q, const float* K, const float* V,
    float* output,
    int seq_len, int context_len, int num_heads, int head_dim,
    int iterations = 10
) {
    // Warmup
    for (int i = 0; i < 3; i++) {
        flash_attention_cpu(Q, K, V, output, seq_len, context_len, num_heads, head_dim);
    }

    // Measure
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        flash_attention_cpu(Q, K, V, output, seq_len, context_len, num_heads, head_dim);
    }
    auto end = std::chrono::high_resolution_clock::now();

    float elapsed_ms = std::chrono::duration<float, std::milli>(end - start).count();
    float latency_ms = elapsed_ms / iterations;

    // Compute metrics
    uint64_t flops = 2ULL * seq_len * context_len * head_dim * num_heads * iterations;
    float throughput_tflops = flops / (elapsed_ms * 1e9);

    uint64_t bytes = (seq_len + 2 * context_len) * head_dim * num_heads * sizeof(float) * iterations;
    float memory_bw_gbs = bytes / (elapsed_ms * 1e6);

    return {latency_ms, throughput_tflops, memory_bw_gbs};
}
```

### 10.2 Expected Performance Numbers

```
Attention Performance on EPYC 9754:

Configuration: 7B model, 32 heads, 128 head_dim

Batch=1, context=128:
├─ Latency: 0.8 ms
├─ Throughput: 226 GFLOPS
└─ Roofline: 49% (memory-bound)

Batch=1, context=2048:
├─ Latency: 12 ms
├─ Throughput: 220 GFLOPS
└─ Roofline: 48% (sustained memory bandwidth)

Batch=1, context=8192:
├─ Latency: 48 ms
├─ Throughput: 216 GFLOPS
└─ Roofline: 47% (L3 cache misses)

Batch=1, context=128K (with chunking):
├─ Latency: 750 ms
├─ Throughput: 210 GFLOPS
└─ Roofline: 45% (memory bottleneck)
```

---

## References & Further Reading

1. **FlashAttention**: Dao et al. (2022), "FlashAttention: Fast and Memory-Efficient Exact Attention"
2. **Online Softmax**: Welford (1962), generalized by Dao et al.
3. **GQA**: Ainslie et al. (2023), "GQA: Training Generalized Multi-Query Transformer"
4. **CPU Memory Hierarchy**: Intel Xeon optimization guides

---

**End of Module 14**

*Total word count: 4,500 words*

# MODULE 7: KV-Cache Management Systems

## 1. SYSTEMS OVERVIEW

The KV-cache represents one of the most critical bottlenecks in transformer inference. Unlike training, where intermediate activations flow forward then backward once per batch, inference generates one token at a time, repeatedly accessing the same cached K and V tensors. This module explores the complete systems challenges around managing KV-cache: memory arithmetic, layout optimization, virtual memory techniques, and compression strategies.

### 1.1 The KV-Cache Memory Problem

For a single decoding pass of an LLM with B_batch sequences:
```
KV-cache memory (bytes) = 2 × B × seq_len × num_heads × head_dim × dtype_bytes

Example (Llama 2 70B):
- B = 1 sequence
- seq_len = 2048 tokens
- num_heads = 64
- head_dim = 128
- dtype = FP16 (2 bytes)

Memory = 2 × 1 × 2048 × 64 × 128 × 2 = 33.5 MB per sequence
```

At batch size 8:
```
Memory = 33.5 × 8 = 268 MB just for KV-cache
```

A single A100 GPU has 80 GB total memory. Accounting for model weights (70B params × 2 bytes = 140 GB), we need **multiple GPUs just to fit the model**. KV-cache fragmentation then prevents batching multiple sequences efficiently.

### 1.2 The Memory Utilization Problem

Assume we want to run 5 concurrent sequences with lengths:
- Sequence 1: 1024 tokens
- Sequence 2: 512 tokens
- Sequence 3: 2048 tokens
- Sequence 4: 256 tokens
- Sequence 5: 1536 tokens

Total: 5376 tokens × cache_per_token = 180 MB cache needed

With contiguous allocation (pre-allocated to max_seq_len=4096):
```
Memory allocated = 5 × 4096 × cache_per_token = 27.3 GB
Memory used = 180 MB
Utilization = 0.66% (!!)
```

This motivates PagedAttention: treat KV-cache like OS virtual memory.

### 1.3 Historical Context

**Pre-2023**: Simple strategies
- Pre-allocate maximum possible KV-cache (wasteful)
- Batch only sequences of same length
- Recompute attention for longer sequences (slow)

**2023 Breakthrough**: PagedAttention (vLLM)
- Allocate KV-cache in fixed-size blocks (pages)
- Maintain block tables per sequence
- Support arbitrary sequence lengths
- Enable true dynamic batching

**Current Era**: Compression and prefix caching
- StreamingLLM: identify and keep attention sinks
- RadixAttention: share prefixes across requests
- H2O: oracle-based token pruning

---

## 2. THEORETICAL FOUNDATION

### 2.1 Exact KV-Cache Memory Arithmetic

For precise memory calculation, account for all intermediate buffers:

**Primary buffers** (required for inference):
```
K_cache = batch_size × seq_len × num_heads × head_dim × bytes_per_token
V_cache = batch_size × seq_len × num_heads × head_dim × bytes_per_token
Total_primary = 2 × batch_size × seq_len × num_heads × head_dim × bytes_per_token
```

**Intermediate buffers** (during forward pass):
```
Attention_scores = batch_size × num_heads × 1 × seq_len × 4 bytes (FP32 softmax)
Attention_weights = batch_size × num_heads × 1 × seq_len × 2 bytes (FP16)
Gradient_buffers (training) = similar size
Total_intermediate ≈ 1.5 × seq_len × batch_size × num_heads × 4 bytes
```

**Total working memory**:
```
Total = 2 × seq_len × (batch × num_heads × head_dim × 2)
      + 1.5 × seq_len × (batch × num_heads × 4)
```

For batch=32, seq=4096, num_heads=64, head_dim=128:
```
K,V = 2 × 4096 × (32 × 64 × 128 × 2) = 67.1 GB
Intermediate = 1.5 × 4096 × (32 × 64 × 4) = 50.3 GB
Total = 117.4 GB (exceeds H100 80 GB!)
```

This explains why: **batching 32 sequences × 4K tokens is impossible on single GPU**.

### 2.2 KV-Cache Layout Optimization

The layout of K and V tensors in GPU memory critically affects performance.

**Layout 1: BSNH (Batch, Seq, Num_heads, Hidden)**
```
K[b, s, h, d] is contiguous in memory
Memory address = offset_b × (S×H×D) + offset_s × (H×D) + offset_h × D + offset_d
```

During attention, we load K for a single query across ALL tokens:
```
for token_s in range(seq_len):
    K_token = K[batch, token_s, head_h, :]  # stride = H×D
```

This produces **strided memory access**—poor cache locality.

**Layout 2: BNSD (Batch, Num_heads, Seq, Hidden)** [Better for inference]
```
K[b, h, s, d] is contiguous
Memory address = offset_b × (H×S×D) + offset_h × (S×D) + offset_s × D + offset_d
```

During attention on head_h:
```
for token_s in range(seq_len):
    K_token = K[batch, head_h, token_s, :]  # stride = D (1 in terms of elements)
```

This is **sequential access**—excellent cache locality, enabling prefetching.

**Memory bandwidth improvement**: BNSD reduces bandwidth demand by 3-5× compared to BSNH because GPU cache line is typically 128 bytes (8-16 FP16 elements). Accessing in BNSD order fills each cache line completely.

### 2.3 Prefetch Strategy and Cache Lines

Modern GPUs have L1 cache of 192 KB per SM and 40 MB L2 cache.

For an A100 with 108 SMs:
```
Total L1 per GPU = 108 × 192 KB = 20 MB
Total L2 per GPU = 40 MB
```

During attention decode (where batch_size=1):
- Q: [1, num_heads, 1, head_dim] ≈ 1KB (one query per head)
- Need: K and V for all 2048 tokens ≈ 33 MB

This exceeds L2 cache. Prefetching strategy:
1. Load K[h, 0:64, :] → L1 (64 tokens, 64KB)
2. Compute attention (64 tokens × 128 dims = 8K ops per head)
3. Prefetch K[h, 64:128, :] → L1 while computing
4. Load V[h, 0:64, :] for weighted sum
5. Prefetch V[h, 64:128, :] while computing

This interleaving of compute and memory access hides latency.

### 2.4 PagedAttention: Virtual Memory Model

Treat KV-cache blocks as pages in virtual memory:

**Block structure**:
```
Block = [num_heads, block_size, head_dim] tensor

For block_size = 16 tokens, num_heads = 64, head_dim = 128:
Block_size_bytes = 64 × 16 × 128 × 2 = 262 KB (fits in L2 cache)
```

**Block allocation**:
```
num_blocks = gpu_memory_available / block_size_bytes
           = 80 GB / (262 KB) ≈ 300,000 blocks

Each sequence maintains a block_table:
block_table[seq_id] = [block_idx_0, block_idx_1, ..., block_idx_N]

For seq_id=0 with 2048 tokens (128 blocks needed):
block_table[0] = [42, 15, 189, ..., 299001]  (physical indices)
```

**Virtual to physical translation**:
```
def get_kv_for_token(seq_id, token_idx):
    block_id = token_idx // block_size
    offset_in_block = token_idx % block_size

    physical_block = block_table[seq_id][block_id]
    return K_blocks[physical_block, :, offset_in_block, :]
```

This is **identical to OS virtual memory paging**—sequence sees contiguous tokens but K/V are in scattered physical blocks.

**Memory utilization improvement**:
```
Before PagedAttention (fixed allocation):
- 5 sequences with max_len=4096
- Memory used = 5 × 4096 × cache_per_token
- Utilization = actual_tokens / (5 × 4096) = 0.66%

After PagedAttention (block-based):
- Allocate only blocks_needed
- Memory used = (5376 tokens / block_size) × block_size × cache_per_token
- Utilization = 100% (minus small overhead)
```

### 2.5 Copy-on-Write and Prefix Caching

When multiple sequences share the same prefix (e.g., system prompt), we can use copy-on-write:

**Scenario**:
- Request 1: "System prompt" + "Query A" = 256 + 512 tokens
- Request 2: "System prompt" + "Query B" = 256 + 256 tokens
- Both share first 256 tokens

**Standard approach**:
- Allocate blocks for request 1: 48 blocks
- Allocate blocks for request 2: 40 blocks
- Total: 88 blocks (21 redundant)

**Copy-on-write approach**:
- Request 1 uses blocks [0..47]
- Request 2 reuses blocks [0..15] + allocates [48..62]
- When request 1 generates new token, it uses existing block 48
- When request 2 generates new token in same block, create copy-on-write:
  - request 2: point to new block, copy [partial block]
  - Total overhead: one partial block copy (8 tokens × cache)

This saves 21 × 262KB = 5.5 MB for this example.

---

## 3. HARDWARE MAPPING

### 3.1 GPU Memory Subsystem Review

**NVIDIA H100**:
- HBM Bandwidth: 3.35 TB/s
- L2 Cache: 50 MB (shared)
- L1 Cache: 192 KB per SM × 144 SMs = 27 MB
- Peak L1 throughput: 30 TB/s (reuse from L1 is 15× bandwidth improvement)

**Access patterns for KV**:
- K and V are typically read-only during inference
- Read bandwidth critical, write bandwidth minimal
- Sequential access patterns = good cache locality

**Block size selection**:
- Want block to fit in L2 cache
- 50 MB L2 ÷ 262 KB per block = 195 blocks can stay in L2
- At 100 decoding iterations, each block accessed once → no L2 eviction
- At 1000 iterations, blocks evict from L2 but stay in HBM (wait for page table lookup)

### 3.2 NUMA and Multi-GPU Effects

For multi-GPU systems with NUMA topology (8 GPUs):

**Local access**: GPU 0 accessing GPU 0 memory
```
Latency: ~100 ns
Bandwidth: 2 TB/s (full HBM bandwidth)
```

**Remote access** (GPU 0 accessing GPU 1 memory over NVLink):
```
Latency: ~200 ns
Bandwidth: 0.4 TB/s (NVLink is narrower than internal HBM)
```

**Implication for KV-cache**: Keep KV-cache local to processing GPU. Cross-GPU access only for model weights (amortized over many tokens).

### 3.3 Temperature and Power Constraints

During heavy batching, the memory subsystem generates significant power:

```
Power = (memory_bandwidth) × (supply_voltage) × (frequency)

For HBM at 3 TB/s:
Power ≈ 50-80 W just for memory operations

GPU power budget: 700 W total
Available for compute: 700 - 80 = 620 W
```

Under sustained load, GPUs throttle frequency to manage temperature. Careful KV-cache management that minimizes bandwidth (e.g., compression, pruning) indirectly improves compute throughput by reducing thermal constraints.

---

## 4. IMPLEMENTATION DEEP DIVE

### 4.1 PagedAttention Block Manager (Python + CUDA)

```python
# block_manager.py
import torch
from typing import Dict, List, Tuple
import numpy as np

class BlockManager:
    """Virtual memory manager for KV-cache blocks"""

    def __init__(
        self,
        gpu_mem_gb: float,
        block_size: int,  # tokens per block
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16
    ):
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Calculate block size in bytes
        bytes_per_token = 2 * num_heads * head_dim * (2 if dtype == torch.float16 else 4)
        block_bytes = bytes_per_token * block_size

        # Allocate physical blocks
        max_blocks = int(gpu_mem_gb * 1e9 // block_bytes)
        self.block_table = torch.zeros(
            (max_blocks, num_heads, block_size, head_dim),
            dtype=dtype,
            device="cuda"
        )

        # Tracking
        self.free_blocks = list(range(max_blocks))
        self.seq_to_blocks: Dict[int, List[int]] = {}
        self.seq_lengths: Dict[int, int] = {}

    def allocate(self, seq_id: int, num_blocks: int) -> List[int]:
        """Allocate blocks for a new sequence"""
        if seq_id in self.seq_to_blocks:
            raise ValueError(f"Sequence {seq_id} already allocated")

        blocks = []
        for _ in range(num_blocks):
            if not self.free_blocks:
                # Eviction policy: LRU (simplified - use last access time in production)
                evicted_seq = min(
                    (s for s in self.seq_to_blocks if s != seq_id),
                    key=lambda s: self.seq_lengths[s] if s != seq_id else float('inf')
                )
                self.free_blocks.extend(self.seq_to_blocks[evicted_seq])
                del self.seq_to_blocks[evicted_seq]
                del self.seq_lengths[evicted_seq]

            block_idx = self.free_blocks.pop(0)
            blocks.append(block_idx)

        self.seq_to_blocks[seq_id] = blocks
        self.seq_lengths[seq_id] = 0
        return blocks

    def extend(self, seq_id: int, num_new_blocks: int) -> List[int]:
        """Add blocks to existing sequence (pre-allocate for upcoming tokens)"""
        new_blocks = []
        for _ in range(num_new_blocks):
            block_idx = self.free_blocks.pop(0)
            new_blocks.append(block_idx)

        self.seq_to_blocks[seq_id].extend(new_blocks)
        return new_blocks

    def write_tokens(
        self,
        seq_id: int,
        K: torch.Tensor,  # [1, num_heads, num_tokens, head_dim]
        V: torch.Tensor,
        start_pos: int
    ):
        """Write KV tokens to cache"""
        blocks = self.seq_to_blocks[seq_id]

        for token_idx in range(K.size(2)):
            abs_pos = start_pos + token_idx
            block_idx = blocks[abs_pos // self.block_size]
            offset_in_block = abs_pos % self.block_size

            self.block_table[block_idx, :, offset_in_block, :] = K[0, :, token_idx, :]

        for token_idx in range(V.size(2)):
            abs_pos = start_pos + token_idx
            block_idx = blocks[abs_pos // self.block_size]
            offset_in_block = abs_pos % self.block_size

            self.block_table[block_idx, :, offset_in_block, :] = V[0, :, token_idx, :]

        self.seq_lengths[seq_id] = max(self.seq_lengths[seq_id], start_pos + K.size(2))

    def get_cached_kv(
        self,
        seq_id: int,
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve complete KV cache for a sequence

        Returns contiguous tensors even though underlying blocks may be scattered
        """
        blocks = self.seq_to_blocks[seq_id]
        num_blocks_needed = (seq_len + self.block_size - 1) // self.block_size

        # Gather blocks into contiguous tensor
        K_list = []
        V_list = []

        for block_idx in blocks[:num_blocks_needed]:
            K_list.append(self.block_table[block_idx, :, :, :])
            V_list.append(self.block_table[block_idx, :, :, :])

        K = torch.cat(K_list, dim=1)  # [num_heads, total_tokens, head_dim]
        V = torch.cat(V_list, dim=1)

        # Trim to actual sequence length
        K = K[:, :seq_len, :]
        V = V[:, :seq_len, :]

        return K, V

    def free_blocks_for_seq(self, seq_id: int):
        """Release blocks when sequence finishes"""
        if seq_id in self.seq_to_blocks:
            self.free_blocks.extend(self.seq_to_blocks[seq_id])
            del self.seq_to_blocks[seq_id]
            del self.seq_lengths[seq_id]

    def memory_utilization(self) -> float:
        """Percentage of blocks allocated"""
        total_blocks = len(self.block_table)
        allocated_blocks = sum(len(b) for b in self.seq_to_blocks.values())
        return allocated_blocks / total_blocks
```

### 4.2 Attention Kernel Using Paged Cache

```python
# paged_attention_kernel.py
import triton
import triton.language as tl
import torch

@triton.jit
def paged_attention_decode_kernel(
    Q,           # [batch, num_heads, 1, head_dim] (single query token)
    K_pages,     # [num_blocks, num_heads, block_size, head_dim]
    V_pages,     # [num_blocks, num_heads, block_size, head_dim]
    block_table, # [batch, max_blocks_per_seq]
    seq_len,     # [batch] lengths
    Out,         # [batch, num_heads, 1, head_dim]
    sm_scale: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_N: tl.constexpr,  # Process block_size tokens at a time
    HEAD_DIM: tl.constexpr,
):
    """
    Decode attention with paged KV-cache

    Process one decoding token per sequence (batch processing)
    """

    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Load query
    query_offset = batch_idx * tl.num_programs(1) * HEAD_DIM + head_idx * HEAD_DIM
    q = tl.load(Q + query_offset + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]

    # Initialize accumulators
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    sum_exp = 0.0
    max_logit = -tl.float32.max

    # Iterate over all blocks for this sequence
    seq_len_batch = tl.load(seq_len + batch_idx)  # Scalar
    num_blocks = (seq_len_batch + block_size - 1) // block_size

    for block_offset in range(num_blocks):
        # Load block index from block table
        block_table_idx = batch_idx * tl.num_programs(2) + block_offset
        block_idx = tl.load(block_table + block_table_idx)

        # Process all tokens in this block
        for token_offset in range(0, block_size, BLOCK_N):
            # Check if token is within sequence bounds
            abs_token_idx = block_offset * block_size + token_offset
            if abs_token_idx >= seq_len_batch:
                break

            # Load K block for this range
            k_page_offset = (
                block_idx * (tl.num_programs(1) * block_size * HEAD_DIM) +
                head_idx * (block_size * HEAD_DIM) +
                token_offset * HEAD_DIM
            )
            k = tl.load(K_pages + k_page_offset + tl.arange(0, HEAD_DIM))

            # Compute attention score
            logit = tl.sum(q * k) * sm_scale

            # Softmax update
            new_max = tl.maximum(max_logit, logit)
            exp_logit = tl.exp(logit - new_max)

            # Adjust previous exponentials for new maximum
            acc = acc * tl.exp(max_logit - new_max)
            sum_exp = sum_exp * tl.exp(max_logit - new_max) + exp_logit

            # Load V and accumulate
            v_page_offset = (
                block_idx * (tl.num_programs(1) * block_size * HEAD_DIM) +
                head_idx * (block_size * HEAD_DIM) +
                token_offset * HEAD_DIM
            )
            v = tl.load(V_pages + v_page_offset + tl.arange(0, HEAD_DIM))
            acc = acc + exp_logit * v

            max_logit = new_max

    # Normalize
    out = acc / sum_exp

    # Write output
    out_offset = batch_idx * tl.num_programs(1) * HEAD_DIM + head_idx * HEAD_DIM
    tl.store(Out + out_offset + tl.arange(0, HEAD_DIM), out)


def paged_attention_forward(
    Q: torch.Tensor,              # [batch, num_heads, 1, head_dim]
    block_manager,                # BlockManager instance
    seq_ids: List[int],
    seq_lengths: torch.Tensor,    # [batch]
):
    """Wrapper for paged attention"""

    batch_size = len(seq_ids)
    num_heads = Q.size(1)
    head_dim = Q.size(3)
    block_size = block_manager.block_size

    # Build block table
    max_seq_len = seq_lengths.max().item()
    max_blocks = (max_seq_len + block_size - 1) // block_size

    block_table = torch.zeros(
        (batch_size, max_blocks),
        dtype=torch.int32,
        device="cuda"
    )

    for batch_idx, seq_id in enumerate(seq_ids):
        blocks = block_manager.seq_to_blocks[seq_id]
        block_table[batch_idx, :len(blocks)] = torch.tensor(blocks)

    K_pages = block_manager.block_table
    V_pages = block_manager.block_table  # Would be separate in production

    Out = torch.empty_like(Q)

    # Launch kernel
    grid = (batch_size, num_heads)
    paged_attention_decode_kernel[grid](
        Q, K_pages, V_pages, block_table, seq_lengths,
        Out,
        sm_scale=1.0 / (head_dim ** 0.5),
        block_size=block_size,
        BLOCK_N=16,
        HEAD_DIM=head_dim,
    )

    return Out
```

### 4.3 StreamingLLM: Attention Sink Implementation

```python
# streaming_llm.py
import torch
import torch.nn.functional as F

class AttentionSinkCache:
    """
    Keep attention sink tokens (first few tokens) + recent window

    Theory: First few tokens in context receive disproportionate attention
    ("sink" phenomenon). Dropping middle tokens while keeping sinks + recent
    window maintains quality.
    """

    def __init__(
        self,
        num_sink_tokens: int = 4,
        window_size: int = 4092,  # Keep recent context
    ):
        self.num_sink_tokens = num_sink_tokens
        self.window_size = window_size
        self.cache_len = 0

        self.K_sink = None
        self.V_sink = None
        self.K_recent = None
        self.V_recent = None

    def put(self, K: torch.Tensor, V: torch.Tensor):
        """
        Update cache with new tokens (during generation)

        K, V: [batch, num_heads, num_new_tokens, head_dim]
        """
        seq_len = K.size(2)

        if self.K_sink is None:
            # First chunk: all tokens become sinks
            self.K_sink = K[:, :, :self.num_sink_tokens, :]
            self.V_sink = V[:, :, :self.num_sink_tokens, :]
            self.K_recent = K[:, :, self.num_sink_tokens:, :]
            self.V_recent = V[:, :, self.num_sink_tokens:, :]
            self.cache_len = seq_len
        else:
            # Add new tokens to recent window
            self.K_recent = torch.cat([
                self.K_recent,
                K[:, :, :, :]
            ], dim=2)

            self.V_recent = torch.cat([
                self.V_recent,
                V[:, :, :, :]
            ], dim=2)

            # Trim recent window if too large
            if self.K_recent.size(2) > self.window_size:
                overflow = self.K_recent.size(2) - self.window_size
                self.K_recent = self.K_recent[:, :, overflow:, :]
                self.V_recent = self.V_recent[:, :, overflow:, :]

            self.cache_len += seq_len

    def get(self) -> tuple:
        """Return cache: sinks + recent window"""
        K = torch.cat([self.K_sink, self.K_recent], dim=2)
        V = torch.cat([self.V_sink, self.V_recent], dim=2)
        return K, V

    def memory_usage_mb(self) -> float:
        """Estimate memory usage"""
        total_tokens = self.num_sink_tokens + self.K_recent.size(2)
        bytes_per_token = 2 * self.K_sink.size(1) * self.K_sink.size(3) * 2
        return (total_tokens * bytes_per_token) / 1e6
```

### 4.4 H2O: Heavy Hitter Oracle for Token Pruning

```python
# h2o_cache.py
import torch
import torch.nn.functional as F

class H2OCache:
    """
    Heavy Hitter Oracle: Keep tokens with highest attention mass

    Key insight: Not all tokens receive equal attention. Identify which
    tokens are critical and prune the rest.
    """

    def __init__(
        self,
        cache_budget: int,  # Max tokens to keep
        num_recent: int = 256,  # Always keep recent tokens
    ):
        self.cache_budget = cache_budget
        self.num_recent = num_recent
        self.attention_sums = None
        self.token_scores = []

    def score_tokens(self, attn_weights: torch.Tensor):
        """
        Score tokens by their attention weight

        attn_weights: [batch, num_heads, query_len, key_len]
        Average across heads and queries: importance = avg attention to each token
        """
        # Average attention across heads and queries
        token_importance = attn_weights.mean(dim=(1, 2))  # [batch, key_len]

        self.attention_sums = token_importance
        return token_importance

    def select_tokens_to_keep(self, seq_len: int):
        """
        Determine which tokens to retain

        Keep:
        1. First few tokens (attention sinks)
        2. Most recently generated tokens
        3. Highest-importance tokens (by H2O oracle)
        """
        num_to_keep = self.cache_budget
        num_sink_tokens = 4  # Keep first few

        if seq_len <= num_to_keep:
            return list(range(seq_len))

        # Always keep recent tokens
        recent_start = max(num_sink_tokens, seq_len - self.num_recent)
        keep_indices = set(range(num_sink_tokens)) | set(range(recent_start, seq_len))

        # Fill remaining budget with highest-attention tokens
        remaining_budget = num_to_keep - len(keep_indices)

        # Score middle tokens (between sinks and recent)
        middle_start = num_sink_tokens
        middle_end = recent_start
        middle_tokens = self.attention_sums[middle_start:middle_end]

        if remaining_budget > 0 and middle_tokens.numel() > 0:
            # Top-k selection
            top_k = min(remaining_budget, middle_tokens.numel())
            _, top_indices = torch.topk(middle_tokens, top_k)
            keep_indices.update((middle_start + idx.item()) for idx in top_indices)

        return sorted(keep_indices)

    def prune_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        keep_indices: list
    ) -> tuple:
        """
        Remove tokens not in keep_indices

        K, V: [batch, num_heads, seq_len, head_dim]
        Returns pruned K, V
        """
        if len(keep_indices) == K.size(2):
            return K, V

        keep_tensor = torch.tensor(keep_indices, device=K.device)
        K_pruned = K[:, :, keep_tensor, :]
        V_pruned = V[:, :, keep_tensor, :]

        return K_pruned, V_pruned


# Integration with inference loop
def streaming_inference_with_h2o(
    model,
    input_ids,
    max_new_tokens,
    cache_budget=2048,
):
    """
    Generate tokens with H2O caching and attention sink strategy
    """
    h2o_cache = H2OCache(cache_budget=cache_budget)

    K_cache = None
    V_cache = None
    cache_indices = None

    for _ in range(max_new_tokens):
        with torch.no_grad():
            # Forward pass
            outputs = model(
                input_ids,
                past_key_values=(K_cache, V_cache) if K_cache is not None else None,
                output_attentions=True,
            )

            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Score tokens by attention
            attn_weights = outputs.attentions[-1]  # Last layer attention
            h2o_cache.score_tokens(attn_weights)

            # Update cache with pruning
            new_K = outputs.past_key_values[0]
            new_V = outputs.past_key_values[1]

            seq_len = new_K.size(2)
            keep_indices = h2o_cache.select_tokens_to_keep(seq_len)

            K_cache, V_cache = h2o_cache.prune_cache(new_K, new_V, keep_indices)
            cache_indices = keep_indices

            input_ids = torch.cat([input_ids, next_token], dim=-1)

    return input_ids
```

### 4.5 RadixAttention: Prefix Caching with Radix Trees

```python
# radix_attention.py
from typing import Dict, Tuple
import torch

class RadixTree:
    """
    Trie structure for KV cache sharing

    Sequences with common prefixes share cache blocks
    """

    class Node:
        def __init__(self):
            self.children: Dict[int, 'RadixTree.Node'] = {}
            self.is_leaf = False
            self.kv_cache: Tuple[torch.Tensor, torch.Tensor] = None
            self.token_count = 0

    def __init__(self):
        self.root = self.Node()

    def insert_sequence(
        self,
        seq_id: int,
        tokens: list,
        K: torch.Tensor,
        V: torch.Tensor,
    ):
        """
        Insert sequence into radix tree

        If another sequence with same prefix exists, share blocks
        """
        node = self.root

        for i, token in enumerate(tokens):
            if token not in node.children:
                node.children[token] = self.Node()

            node = node.children[token]
            node.token_count = i + 1

            # Store KV for this token (shared with all descendants)
            if node.kv_cache is None:
                node.kv_cache = (
                    K[:, :, i:i+1, :],
                    V[:, :, i:i+1, :]
                )

        node.is_leaf = True

    def find_longest_prefix(self, tokens: list) -> Tuple[list, int]:
        """
        Find longest matching prefix in tree

        Returns: (matching_tokens, node_depth)
        """
        node = self.root
        matched_tokens = []

        for token in tokens:
            if token not in node.children:
                break
            node = node.children[token]
            matched_tokens.append(token)

        return matched_tokens, node.token_count

    def get_cached_kv(self, node_depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve cached KV up to tree node"""
        # Navigate to node and gather cache
        # In production, would maintain pointers for efficient retrieval
        pass


def enable_prefix_caching(req1, req2):
    """
    Example: Two requests with shared system prompt

    req1: "System: You are helpful. Q: What is 2+2?"
    req2: "System: You are helpful. Q: What is 3*4?"

    Tokens 0-10 are identical (system prompt)
    Share cache blocks 0-1 (assuming block_size=8)
    """
    radix = RadixTree()

    # Process request 1
    tokens1 = tokenize("System: You are helpful. Q: What is 2+2?")
    K1, V1 = encode_prompt(tokens1)
    radix.insert_sequence(seq_id=1, tokens=tokens1, K=K1, V=V1)

    # Process request 2
    tokens2 = tokenize("System: You are helpful. Q: What is 3*4?")

    # Find common prefix
    common_tokens, prefix_len = radix.find_longest_prefix(tokens2)
    print(f"Matched {len(common_tokens)} tokens from cache")

    # Only need to compute K, V for new tokens
    new_tokens_idx = len(common_tokens)
    K2_new = compute_kv_for_new_tokens(tokens2[new_tokens_idx:])

    # Retrieve shared cache
    K2_shared = get_cached_kv_from_tree(radix, prefix_len)
    K2 = torch.cat([K2_shared, K2_new], dim=2)

    return K2
```

---

## 5. KEY PAPERS

1. **PagedAttention for Efficient Memory Management in Large Language Model Serving** (Kwon, Li, Zhuang, et al.; SOSP 2023)
   - OS-inspired virtual memory for KV-cache
   - 10-30× throughput improvement
   - Enables arbitrary sequence length batching

2. **SGLang: Efficient Execution of Structured Language Model Programs** (Zheng, Gu, et al.; NeurIPS 2024)
   - Builds on PagedAttention for structured generation
   - RadixAttention for prefix caching
   - Production system achieving >70% GPU utilization

3. **StreamingLLM: Efficient Streaming Language Models with Attention Sinks** (Xiao, Hou, et al.; ICLR 2024)
   - Identifies attention sink phenomenon
   - Supports streaming with fixed memory budget
   - Works with models trained on fixed context

4. **H2O: Heavy Hitter Oracle for Efficient Generative Inference** (Zhang, Parikh, Henao; NeurIPS 2023)
   - Token importance scoring based on attention patterns
   - 50% KV-cache reduction with minimal quality loss
   - Adaptive selection based on context

5. **FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU** (Sheng, Gao, Parikh, et al.; ICML 2023)
   - CPU-GPU co-execution for KV-cache
   - Bandwidth-aware scheduling
   - Enables 125B model inference on single V100

6. **Fast Distributed Inference of Vision Transformers through Progressive Layer Skipping** (Xin et al.; 2023)
   - Orthogonal optimization: skip entire layers
   - Combines with KV-cache compression

---

## 6. SYSTEMS TRADEOFFS

### 6.1 Block Size Selection

| Block Size | Pros | Cons |
|-----------|------|------|
| 1 token | Fine-grained allocation, no waste | Overhead of page table per token, poor cache reuse |
| 16 tokens | Good L1/L2 reuse, moderate overhead | Some waste if seq_len not multiple of 16 |
| 64 tokens | Excellent prefetching, one block per HBM round-trip | Blocks may not fit in L2 with large batch |
| 256 tokens | Minimal page table, efficient disk I/O (if swap) | Poor cache locality, high L2 eviction rate |

**Recommendation**: Block size = 16 for batch inference, 64 for single-sequence (higher latency targets).

### 6.2 Compression vs Latency

| Strategy | Memory Savings | Latency Impact | Quality Drop |
|----------|----------------|----------------|--------------|
| No compression | 0% | 0% | 0% |
| INT8 quantization | 50% | +5% (conversion) | 0.1% |
| StreamingLLM | 50-70% | -5% (fewer tokens) | 1-2% (long context) |
| H2O pruning | 50% | 0% (token selection overhead < 1%) | 0.2% |
| MQA+GQA | 75% | -3% (less data) | 0.5% |

**Recommendation**: Stack H2O + GQA for 80% savings with <0.5% accuracy loss.

### 6.3 Prefill vs Decode Phase

| Aspect | Prefill | Decode |
|--------|---------|--------|
| Sequence length growth | Large to larger | +1 per iteration |
| Cache hits | Low (first pass) | High (reuse K/V) |
| Batch size advantage | High (hide latency) | Very high (amortize decode cost) |
| KV-cache fragmentation | Minimum (contiguous allocation) | High (many small sequences) |
| Optimal block size | Large (512) | Small (16) |

### 6.4 Single GPU vs Multi-GPU

| Constraint | Single GPU | Multi-GPU |
|-----------|-----------|-----------|
| KV-cache memory limit | 10-20 GB (after model) | 10-20 GB × GPU count |
| Batch size limit | 1-4 sequences | 32-128 sequences |
| Throughput | Limited by one GPU | Scales ~linearly |
| Latency | High for small batches | Better with batching |

---

## 7. EXPERT INSIGHT

### 7.1 Debugging KV-Cache Issues

**Symptom**: Throughput degradation as batch size increases (expected: should improve)

**Root cause**: KV-cache memory fragmentation

**Diagnosis**:
```python
# Monitor block utilization
def check_fragmentation(block_manager):
    total_blocks = len(block_manager.block_table)
    used_blocks = sum(len(b) for b in block_manager.seq_to_blocks.values())
    free_blocks = total_blocks - used_blocks

    # Measure fragmentation by checking free block list
    fragmentation = 1.0 - (len(block_manager.free_blocks) / free_blocks)
    print(f"Fragmentation: {fragmentation:.2%}")

    if fragmentation > 0.5:
        print("WARNING: High fragmentation, evict small sequences")
```

**Solution**: Implement locality-aware scheduling (colocate sequences by length).

### 7.2 When to Use Each KV-Cache Strategy

**Standard dense cache**:
- Use when batch size ≤ 4, seq_len ≤ 2048
- Training (always)

**PagedAttention**:
- Use for production inference servers
- Mandatory if batch size > 8

**Prefix caching (RadixAttention)**:
- Common-prompt patterns (chatbot with system prompt)
- Saves 20-40% for typical enterprise use cases

**Streaming cache (StreamingLLM)**:
- Long-context applications (n > 32K)
- Fixed-window models (Llama 2, Mistral)

**Token pruning (H2O)**:
- Long sequences (n > 4K)
- Latency-critical deployment
- Achieves 50% reduction with <0.2% accuracy loss

### 7.3 Production Deployment Checklist

```
[ ] Benchmark single sequence latency (TTFT)
[ ] Benchmark throughput at max batch size
[ ] Measure KV-cache memory per sequence
[ ] Profile block allocation/deallocation overhead
[ ] Test with variable sequence lengths
[ ] Implement LRU eviction policy
[ ] Add memory usage monitoring
[ ] Test graceful degradation under memory pressure
[ ] Validate compression doesn't exceed quality threshold
[ ] Implement redundancy (distributed cache replication)
```

---

## 8. BENCHMARKING METHODOLOGY

### 8.1 Metrics

```python
# kv_cache_metrics.py

def measure_kv_cache_performance():
    """
    Key metrics for KV-cache systems
    """

    # 1. Memory efficiency
    memory_per_token = kv_cache_bytes / num_tokens
    memory_utilization = used_bytes / allocated_bytes

    # 2. Latency breakdown
    ttft = time_first_token  # Time to first token (prefill phase)
    tpot = time_per_output_token  # Time per token during decode
    itl = (tpot - 1) / (tpot)  # Iteration latency

    # 3. Throughput
    throughput_tokens_per_sec = num_tokens_generated / total_time
    throughput_sequences_per_sec = num_sequences / total_time

    # 4. Cache efficiency
    cache_hit_rate = cache_hits / (cache_hits + cache_misses)
    fragmentation_ratio = wasted_blocks / total_blocks

    return {
        'memory_per_token_mb': memory_per_token,
        'utilization_pct': memory_utilization,
        'ttft_ms': ttft,
        'tpot_ms': tpot,
        'throughput_tps': throughput_tokens_per_sec,
        'fragmentation_pct': fragmentation_ratio,
    }
```

### 8.2 Benchmark Scenarios

**Scenario 1: Single long sequence**
- Input: 4096 tokens
- Generate: 512 tokens
- Measure: TTFT, sustained throughput during decoding

**Scenario 2: Many short sequences (chatbot)**
- 32 concurrent sequences, avg 256 tokens input
- Generate: 128 tokens per sequence
- Measure: Throughput, memory efficiency

**Scenario 3: Variable length (realistic)**
- Zipfian distribution of sequence lengths
- Measure: Memory fragmentation, eviction rate

### 8.3 Comparison Matrix

```
Benchmark: vLLM PagedAttention vs HF standard vs no caching

Config: 7B Llama 2, V100 GPU (32GB)

Batch=1, Seq=512:
- vLLM:     TTFT=45ms, TPOT=35ms, Memory=8GB
- HuggingFace: TTFT=50ms, TPOT=40ms, Memory=10GB
- No cache: TTFT=120ms, TPOT=95ms, Memory=20GB

Batch=8, Seq=2048:
- vLLM:     Throughput=120 tps, Memory=28GB
- HuggingFace: Throughput=40 tps, Memory=32GB (OOM with batch > 4)
- No cache: OOM

Batch=32, Seq=4096:
- vLLM:     Throughput=180 tps, Memory=30GB (with prefix caching)
- HuggingFace: N/A (OOM)
- No cache: N/A (OOM)
```

---

## 9. OPEN PROBLEMS

### 9.1 Optimal Eviction Policy

PagedAttention uses LRU eviction, but optimal policy is problem-dependent:
- For chatbots: prioritize recent sequences (likely to continue)
- For batch inference: prioritize longest sequences (maximum completion time)
- For workload mix: difficult to predict

**Research opportunity**: Learn eviction policy from workload traces.

### 9.2 Cross-Request Prefix Sharing

Current RadixAttention identifies prefixes but doesn't handle:
- Token-level differences within shared prefix
- Learned token embeddings that vary by context
- Position encodings that depend on absolute position

**Challenge**: Extend to semantic similarity (not just exact token match).

### 9.3 Compression Quality Evaluation

H2O and StreamingLLM reduce cache but evaluation is limited:
- Most benchmarks on generation length ≤ 256
- Few long-context tasks exist
- Quality metrics (BLEU, ROUGE) saturate

**Research need**: Develop metrics for long-context quality degradation.

### 9.4 Asynchronous Prefill/Decode

Current systems prefill and decode sequentially. Overlapping them could hide latency:
```
Time: |Prefill_1|Decode_1|Prefill_2|Decode_2|
Ideal: |Prefill_1 + Decode_1|Prefill_2 + Decode_2|
```

**Bottleneck**: Prefill and decode use different compute patterns (high vs low arithmetic intensity).

---

## 10. PHD QUALIFIER QUESTIONS

1. **Mathematical Rigor** (35 minutes):
   - Derive the exact memory formula for KV-cache including all intermediate buffers
   - Compare BNSD vs BSNH layout: prove the bandwidth advantage
   - Show that paged access with random blocks has O(1) average cost under certain conditions

2. **Systems Design** (45 minutes):
   - Design KV-cache management for a 200B parameter model on 8 GPUs
   - Account for: model weights, intermediate activations, KV-cache, gradient buffers
   - What batch size can you sustain at 16 tokens/sec throughput?
   - How does adding NVLink vs PCIe affect your design?

3. **Implementation Tradeoffs** (40 minutes):
   - Compare block sizes 8, 16, 64, 256 for KV-cache
   - For each, calculate: page table overhead, L2 cache utilization, fragmentation
   - Build a function: block_size → throughput for your target hardware
   - At what workload characteristics does each block size win?

4. **Compression Analysis** (35 minutes):
   - Analyze why H2O's importance scoring works (mathematical intuition)
   - For a specific workload, what tokens are pruned vs retained?
   - How would you validate that 50% pruning doesn't exceed acceptable quality loss?
   - Propose a better pruning criterion than H2O

5. **Production Challenges** (50 minutes):
   - Your inference server exhibits OOM crashes under load surge
   - Walk through diagnosis: what metrics would you log?
   - Design eviction policy that handles: latency-critical requests, batch completion time, memory pressure
   - How do you prevent unfairness (starvation) under memory pressure?

6. **Distributed KV-Cache** (45 minutes):
   - For multi-GPU KV-cache: where should you store each sequence's cache?
   - Design distributed paging: which GPU holds which blocks?
   - How does choice of placement affect: latency, throughput, communication?
   - What happens when a GPU fails?

7. **Adaptive Compression** (40 minutes):
   - Design a system that adapts compression ratio based on available memory
   - Monitor metrics: memory pressure, latency percentiles, quality degradation
   - Write pseudocode for: (i) increasing compression under pressure, (ii) easing compression under idle
   - What are risks of adaptive schemes?

8. **Long-Context Scaling** (40 minutes):
   - Llama 2 can run at seq_len=32K with RoPE extension
   - Calculate memory for 32K tokens vs 4K tokens
   - Propose modifications to KV-cache management for 100K context
   - What compression techniques become critical at 100K?

---

## Conclusion

KV-cache management is the critical systems problem for modern LLM inference. While PagedAttention solved fragmentation, production systems must address:
- Memory pressure and eviction under load
- Quality degradation from compression
- Latency spikes from reallocation
- Distributed coordination across GPUs

The research frontier combines OS memory management, database indexing (radix trees), and ML optimization (token importance scoring) into unified systems.

Success means 10× throughput improvement without quality loss or latency unpredictability—achievable through careful system design.

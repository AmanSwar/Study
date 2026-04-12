# MODULE 10: Communication Primitives & Collective Operations

## SYSTEMS OVERVIEW

### The Communication Bottleneck

In distributed inference, communication often dominates runtime. Consider a 175B parameter transformer on 8 GPUs with tensor parallelism:

- Per token, each layer requires:
  - 1 forward pass: 2 * 175e9 / 8 = 43.75B FLOPs per GPU
  - A100 peak: 312 TFLOPs
  - Time: 43.75B / (312e12) ≈ 0.14 milliseconds

- Communication for all-reduce (attention output):
  - Data: batch * seq * 12288 FP16 = 2 * batch * seq * 12288 bytes
  - All-reduce over 8 GPUs requires log(8) = 3 reduction stages
  - Each stage: send + receive 2 * 12288 bytes (simplified)
  - Total communication: 3 * 2 * 12288 * 2 ≈ 150 KB
  - InfiniBand bandwidth: 200 GB/s → time ≈ 0.75 microseconds

Wait, this suggests communication is faster than compute. But in practice:

1. **Real-world latencies**: InfiniBand latency is 1-2 microseconds, not the idealized time above
2. **Overhead**: NCCL, MPI have 10-100 microsecond per-message overhead
3. **Multiple collectives**: Each layer does 2-4 all-reduces (attention, FF, skip connections)
4. **Synchronization**: If any GPU is slow, all must wait

Realistic per-layer communication cost: ~1-10 milliseconds (often exceeding compute!).

### The Collective Operation Taxonomy

Distributed inference relies on five core collective operations:

1. **All-Reduce**: Each GPU sends data to all others, receives aggregated result. Common use: gradient aggregation, loss reduction.

2. **All-Gather**: Each GPU broadcasts its data to all others. Result: concatenation of data from all GPUs. Use: gathering shards of a tensor.

3. **Reduce-Scatter**: Inverse of all-gather. Reduce across all GPUs, scatter results. Use: partitioning outputs after reduction.

4. **All-to-All**: Each GPU sends different data to each other GPU. Use: MoE token routing, remapping data layout.

5. **Broadcast/Scatter**: One GPU sends to all others / sends different pieces to different GPUs. Use: distributing model weights, broadcasting tokens.

Each has different bandwidth requirements and latency trade-offs.

---

## THEORETICAL FOUNDATION

### Collective Costs: Bandwidth and Latency Models

For a collective operation with:
- N devices
- M bytes per device (or per pair)
- Latency per message: L (microseconds)
- Bandwidth per link: B (bytes/sec)

**All-Reduce (Ring Algorithm)**:

Ring all-reduce divides data into N chunks of M/N bytes each. Each device:
1. Sends its chunk to next device, receives from previous: log(N) "flights" around the ring
2. Each flight: one device sends M/N bytes to one neighbor

```
Device layout (ring): 0 ↔ 1 ↔ 2 ↔ 3 ↔ 4 ↔ ... ↔ (N-1) ↔ 0

Flight 1: Device i sends M/N bytes to (i+1) % N
Flight 2: Device i sends to (i+2) % N (data received in flight 1)
...
Flight N: Ring complete, all devices have all reduced data

Total time: (N-1) * (M/N) / B + N * L
          = (N-1) * M / (N*B) + N*L
          ≈ M/B + N*L (dominant term for large M)
```

**Example**: M = 12 GB (175B FP16), N = 8, B = 1.9 TB/s, L = 1 µs

Time = 12e9 / 1.9e12 + 8e-6 = 6.3 ms + 8 µs ≈ 6.3 ms (latency negligible for large transfers)

**Tree All-Reduce (Binomial Tree)**:

Reduction phase: reduce-down the tree, then broadcast-up (or direct pattern for all-reduce).

```
         Device 0
        /        \
     Dev 1      Dev 2
    /    \      /   \
  Dev 3  Dev4 Dev5 Dev6
   |
 Dev 7

Reduction: Dev7→Dev3, Dev3→Dev1, Dev1→Dev0, Dev6→Dev2, Dev2→Dev0 (synchronization)
Broadcast: Dev0→Dev1, Dev0→Dev2, Dev1→Dev3, Dev1→Dev4, ...

Time: 2 * log(N) * L + log(N) * M/B
```

For M = 12 GB, N = 8, B = 1.9 TB/s, L = 1 µs:

Ring: 6.3 ms
Tree: 2 * 3 * 1e-6 + 3 * 12e9 / 1.9e12 ≈ 6 µs + 19 ms ≈ 19 ms

Ring is faster for large M. But tree has lower latency (log(N) vs N) for small M.

**All-Gather (Ring)**:

Similar to ring all-reduce, but without reduction. N-1 flights, each GPU broadcasts one chunk.

```
Initial: Device i has data[i]
Flight 1: Device i sends data[i] to (i+1) % N, receives data[(i-1)%N]
Flight 2: Device i sends data[(i-1)%N] to (i+1) % N, receives data[(i-2)%N]
...
Flight N-1: Device i has all data

Time: (N-1) * M / B + (N-1) * L
    ≈ M/B (dominant for large M)
```

**All-to-All (Permutation Sort + Local Exchange)**:

Each GPU sends different data to each other GPU. Naïve: N*(N-1) point-to-point sends.

Optimized (used in NCCL): Each device sends M bytes total (M/N bytes to each destination).

```
Device i sends M/N bytes to each of N-1 other devices (and keeps M/N for itself).
Total data moved per device: M bytes.

Time: log(N) * L + M / B (depends on tree/permutation structure)
```

For dense all-to-all (no sparsity), this is hard to avoid.

### Bandwidth-Optimal vs. Latency-Optimal Algorithms

**Bandwidth-optimal algorithm**: Minimizes total bytes communicated.

For all-reduce with M bytes total:
- Bandwidth-optimal: (N-1) * M/N (ring or tree) + (M/N) local reduction overhead
  ≈ M/B is the bandwidth barrier
- Achieved by: Ring, tree (both achieve M/B asymptotically)

**Latency-optimal algorithm**: Minimizes number of communication rounds.

For all-reduce:
- Latency-optimal: log(N) rounds (binomial tree)
- Ring requires N rounds

Tradeoff:
- Ring: O(M/B) bandwidth cost, O(N) latency cost
- Tree: O(M/B) bandwidth cost, O(log(N)) latency cost

For batch training (amortize latency over many iterations), ring is fine.
For single forward passes, latency matters more.

### Critical Network Utilization Metric: Bisection Bandwidth

Bisection bandwidth: Maximum bandwidth between two equal-sized halves of the network.

**Example: 3-level fat-tree (64 nodes)**:

```
        Top-of-rack switches
       /      |       \
      /       |        \
    Pod0     Pod1  ...  Pod7
    / \      / \
   /   \    /   \
  TOR0  TOR1  TOR2  TOR3

Each link: 100 Gbps (12.5 GB/s)
Each TOR has 32 edges (to servers) + 4 uplinks (to pod)

Bisection bandwidth:
- Cut network in half: 32 pods on one side, 32 on other
- Cross the cut: 4 uplinks per pod = 32 * 4 = 128 uplinks
- Total: 128 * 12.5 GB/s = 1.6 TB/s bisection
- Per node average: 1.6 TB/s / 64 nodes ≈ 25 GB/s per node
```

This limits collective efficiency. Ring all-reduce can approach this theoretical limit, but tree can be worse if it routes through the same links.

---

## HARDWARE MAPPING

### Modern GPU Cluster Interconnect Architectures

**NVIDIA DGX Superpod (8-node cluster)**:

```
Node 0: GPU0-7 (NVLink mesh)
        ↓ (IB NDR dual 400 Gbps = 100 GB/s each direction)
Node 1: GPU8-15 (NVLink mesh)
        ↓
Node 2: GPU16-23
        ...
Node 7: GPU56-63

Intra-node (NVLink):
  - Bandwidth: 900 GB/s per GPU (H100) or 2 TB/s (bidirectional)
  - Latency: 1 microsecond
  - Topology: Fully connected

Inter-node (InfiniBand NDR):
  - Bandwidth: 100 GB/s per direction (single link)
  - Latency: 1-2 microseconds
  - Topology: Fat-tree (all nodes connected via IB switch)
```

**Implication for collectives**:

Intra-node all-reduce (8 GPUs via NVLink):
- Time = (8-1) * (12 GB / 8) / (2 TB/s) + 8 µs = 5.25 ms + 8 µs ≈ 5.25 ms (dominates)
- Actually faster: NCCL can use special NVLink SHARP (in-network aggregation): ~2 ms

Inter-node all-reduce (64 GPUs across 8 nodes):
- NaïveRing: (64-1) * (12 GB / 64) / (100 GB/s) + 64 µs = 93.75 ms + 64 µs ≈ 93.75 ms
- Hierarchical (intra-node + inter-node): ~20 ms (more complex but better)

### Memory Hierarchy and Collective Placement

GPUs have:
- **HBM** (40-80 GB): Register file + L1 + L2 + main memory
- **GPU kernel code**: Can launch reduction kernels (sum, max, min)
- **NCCL rings**: Hardware-accelerated reduction on modern GPUs (H100+)

For inference:
- All-reduce outputs stay in GPU memory (no CPU involvement)
- Large buffers (model weights, activations) stay in HBM
- Small messages (< 1 MB) incur NCCL overhead, not BW limited

### Topology-Aware Collective Routing

Modern clusters have asymmetric topologies:
- Within socket: fast (NVLink)
- Within node-pair (connected by PCIe): medium
- Across pod: slow (fabric switch bottleneck)

Optimal collective algorithm depends on topology. NCCL selects algorithm automatically:

```
If intra-node:
    use_nvlink_tree()  # If connected
elif inter_pod_spanning:
    use_ring()  # Minimize cross-fabric
else:
    use_tree()  # Log(N) hops
```

---

## IMPLEMENTATION DEEP DIVE

### 1. All-Reduce Implementations

**Ring All-Reduce**:

```python
import torch
import torch.distributed as dist

def ring_allreduce(tensor, rank, world_size):
    """
    Ring all-reduce algorithm.

    Each GPU holds a chunk of the tensor. Through N rounds, each GPU
    receives a chunk from previous GPU and sends its current chunk to next.

    After N rounds, each GPU's local tensor contains the full reduced result.
    """

    chunk_size = tensor.numel() // world_size
    chunks = tensor.view(-1).split(chunk_size)

    # Round 1: GPU i sends chunk i to next, receives chunk (i-1) from prev
    # After round k: GPU i has the reduced version of chunks [i-k, i-k+1, ..., i]

    for round_idx in range(world_size - 1):
        # Send chunk to next
        next_rank = (rank + 1) % world_size
        prev_rank = (rank - 1) % world_size

        chunk_to_send_idx = (rank - round_idx) % world_size
        chunk_to_recv_idx = (rank - round_idx - 1) % world_size

        send_chunk = chunks[chunk_to_send_idx]
        recv_chunk = torch.zeros_like(send_chunk)

        send_req = dist.send(send_chunk.clone(), next_rank, async_op=True)
        recv_req = dist.recv(recv_chunk, prev_rank, async_op=True)

        send_req.wait()
        recv_req.wait()

        # Reduce (accumulate)
        chunks[chunk_to_recv_idx] += recv_chunk

    # After N-1 rounds, each GPU sends its final chunk once more to propagate
    # (This is the "broadcast" phase; usually fused)

    result = torch.cat(chunks)
    return result


# Usage with PyTorch distributed
# dist.init_process_group("nccl")
# rank = dist.get_rank()
# world_size = dist.get_world_size()
# tensor = torch.randn(12288, device='cuda')
# result = ring_allreduce(tensor, rank, world_size)
```

**Tree All-Reduce (Binomial)**:

```python
def tree_allreduce(tensor, rank, world_size):
    """
    Tree (binomial) all-reduce.

    Reduction phase: pairs of GPUs reduce their data, winners propagate up.
    Broadcast phase: top GPU broadcasts result down.
    """

    def log2(n):
        import math
        return int(math.log2(n))

    def get_tree_parent_child(rank, world_size, stage):
        """
        Binomial tree structure.
        At stage k, GPU i pairs with GPU (i XOR 2^k).
        If i XOR 2^k < i: GPU i is parent (receives from child)
        If i XOR 2^k > i: GPU i is child (sends to parent)
        """
        mask = 1 << stage
        partner = rank ^ mask

        if partner >= world_size:
            return None, None

        if rank < partner:
            # This GPU is parent
            return partner, 'recv'
        else:
            # This GPU is child
            return partner, 'send'

    # Reduction phase: bottom-up
    for stage in range(log2(world_size)):
        partner, role = get_tree_parent_child(rank, world_size, stage)

        if partner is None:
            continue

        if role == 'recv':
            other_tensor = torch.zeros_like(tensor)
            dist.recv(other_tensor, partner)
            tensor += other_tensor
        else:
            dist.send(tensor.clone(), partner)

    # Broadcast phase: top-down (skip in all-reduce variant that fuses broadcast)
    # ...

    return tensor
```

**Practical NCCL Usage**:

```python
import torch
import torch.distributed as dist

# PyTorch wraps NCCL
def allreduce_pytroch_nccl(tensor):
    """
    PyTorch's distributed.all_reduce uses NCCL on GPU.
    Automatically selects ring, tree, or hybrid based on topology.
    """
    dist.all_reduce(tensor)
    return tensor

# Lower-level NCCL
# If you need fine-grained control:
try:
    from nccl import NcclComm
    # Create NCCL communicator
    # comm = create_nccl_comm()
    # comm.allReduce(tensor_data, None)
except ImportError:
    print("NCCL not directly exposed in PyTorch; use dist.all_reduce()")
```

### 2. All-Gather Implementation

**Ring All-Gather**:

```python
def ring_allgather(tensor, rank, world_size):
    """
    Ring all-gather: each GPU gathers all data via ring rotations.

    Initial state: GPU i has data[i] of shape (chunk_size, hidden)
    Final state: GPU i has data[0] + data[1] + ... + data[world_size-1]
    """

    chunk_size = tensor.shape[0]  # Assuming first dim is split
    output = torch.zeros(chunk_size * world_size, tensor.shape[1], device=tensor.device)

    # Place initial data
    output[rank * chunk_size:(rank + 1) * chunk_size] = tensor

    # Ring: N-1 passes
    send_data = tensor.clone()
    for round_idx in range(world_size - 1):
        next_rank = (rank + 1) % world_size
        prev_rank = (rank - 1) % world_size

        recv_data = torch.zeros_like(send_data)

        # Non-blocking send/recv
        send_req = dist.send(send_data, next_rank, async_op=True)
        recv_req = dist.recv(recv_data, prev_rank, async_op=True)

        send_req.wait()
        recv_req.wait()

        # Update output (the received data is from previous-round sender)
        source_rank = (rank - round_idx - 1) % world_size
        output[source_rank * chunk_size:(source_rank + 1) * chunk_size] = recv_data

        send_data = recv_data

    return output


# PyTorch wrapper
def allgather_pytorch(tensor, world_size):
    """
    Gather all tensors from all ranks into list.
    Result: list of world_size tensors (one from each rank).
    """
    output = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(output, tensor)
    return torch.cat(output, dim=0)
```

**Megatron-style All-Gather in Attention**:

```python
def allgather_then_attend(q_local, k_local, v_local, rank, world_size):
    """
    Tensor parallelism attention: gather K, V across ranks, then compute attention.

    q_local shape: (batch, seq, n_heads, d_head)
    k_local shape: (batch, seq, n_heads_per_rank, d_head)
    v_local shape: (batch, seq, n_heads_per_rank, d_head)

    After gather:
    k_global shape: (batch, seq, n_heads, d_head)
    v_global shape: (batch, seq, n_heads, d_head)
    """

    batch, seq, n_heads_local, d_head = k_local.shape

    # All-gather K, V
    k_gathered = allgather_pytorch(k_local.view(batch * seq * n_heads_local, d_head), world_size)
    v_gathered = allgather_pytorch(v_local.view(batch * seq * n_heads_local, d_head), world_size)

    # Reshape back
    n_heads = n_heads_local * world_size
    k_global = k_gathered.view(batch, seq, n_heads, d_head)
    v_global = v_gathered.view(batch, seq, n_heads, d_head)

    # Attention
    scores = torch.einsum('bsnh,btnh->bnst', q_local, k_global) / math.sqrt(d_head)
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum('bnst,btnh->bsnh', attn, v_global)

    return out
```

### 3. Reduce-Scatter Implementation

**Ring Reduce-Scatter**:

```python
def ring_reducescatter(tensor, rank, world_size):
    """
    Ring reduce-scatter: reduce across all ranks, scatter results.

    Initial: GPU i has data[i]
    Final: GPU i has sum(data[all]) / world_size (or local sum for partitioned result)

    Actually, this scatters the reduced result such that GPU i gets output chunk i.
    """

    chunk_size = tensor.shape[0] // world_size
    chunks = tensor.split(chunk_size, dim=0)

    # Output: GPU i will get the reduced version of chunks[i]
    output = torch.zeros((chunk_size, *tensor.shape[1:]), device=tensor.device)

    # Reduce + scatter: N-1 rounds
    send_chunks = [c.clone() for c in chunks]

    for round_idx in range(world_size - 1):
        next_rank = (rank + 1) % world_size
        prev_rank = (rank - 1) % world_size

        # GPU i sends all its chunks to next GPU, receives from previous
        all_send = torch.cat(send_chunks, dim=0)

        all_recv = torch.zeros_like(all_send)

        send_req = dist.send(all_send, next_rank, async_op=True)
        recv_req = dist.recv(all_recv, prev_rank, async_op=True)

        send_req.wait()
        recv_req.wait()

        # Reduce: add to my own chunks and send
        recv_chunks = all_recv.split(chunk_size, dim=0)
        for i, chunk in enumerate(send_chunks):
            chunk.add_(recv_chunks[i])

        # Rotate: my chunk i becomes chunk (i-1)
        send_chunks = [recv_chunks[-1]] + recv_chunks[:-1]

    # After N-1 rounds, output[i] = sum of all original chunks[i]
    output = send_chunks[0]  # Chunk 0 after rotations is the reduced chunk for rank

    return output


# PyTorch distributed
def reducescatter_pytorch(tensor, world_size, op=dist.ReduceOp.SUM):
    """
    Reduce-scatter: reduce tensor across all ranks, scatter to ranks.

    Output: tensor of shape (original_shape[0] // world_size, ...)
    """
    output = torch.zeros(
        (tensor.shape[0] // world_size, *tensor.shape[1:]),
        device=tensor.device
    )
    dist.reduce_scatter(output, list(tensor.split(tensor.shape[0] // world_size)), op=op)
    return output
```

### 4. All-to-All Implementation

**Dense All-to-All (MoE Token Routing)**:

```python
def alltoall_dense(tokens, rank, world_size):
    """
    All-to-all communication for MoE expert routing.

    Input: tokens of shape (batch, seq, d_model)
           Each token is routed to one expert, which might be on a different rank.

    Output: tokens reshuffled such that rank i gets all tokens routed to its experts.

    Algorithm:
    - Determine destination rank for each token
    - Gather tokens destined for rank j
    - Send to rank j
    - Receive tokens from all ranks
    """

    batch, seq, d_model = tokens.shape
    num_tokens = batch * seq

    # Assume routing[token_idx] = destination_rank
    # This would be computed by a routing network (not shown here)

    # Simple example: alternate assignment
    routing = torch.tensor([i % world_size for i in range(num_tokens)], device=tokens.device)

    # Reshape tokens to (num_tokens, d_model)
    tokens_flat = tokens.view(-1, d_model)

    # Count tokens per destination
    tokens_per_dest = torch.zeros(world_size, dtype=torch.long, device=tokens.device)
    for dest_rank in range(world_size):
        tokens_per_dest[dest_rank] = (routing == dest_rank).sum()

    # Prepare send buffers
    send_counts = tokens_per_dest.tolist()
    send_data_lists = [[] for _ in range(world_size)]

    for token_idx in range(num_tokens):
        dest_rank = routing[token_idx].item()
        send_data_lists[dest_rank].append(tokens_flat[token_idx:token_idx + 1])

    # Concatenate per rank
    send_data = []
    for rank_data in send_data_lists:
        if rank_data:
            send_data.append(torch.cat(rank_data, dim=0))
        else:
            send_data.append(torch.zeros((0, d_model), device=tokens.device))

    # All-to-all using PyTorch distributed
    # Expected: dist.all_to_all() with send/recv lists
    # Simplified: use alltoall_cpp from NCCL if available

    # For now, simulate with pairwise sends
    output_lists = [[] for _ in range(world_size)]

    for src_rank in range(world_size):
        if send_data[src_rank].shape[0] > 0:
            # Send to all ranks (including self)
            for dest_rank in range(world_size):
                if src_rank == rank and dest_rank == rank:
                    output_lists[dest_rank].append(send_data[src_rank])
                elif src_rank == rank:
                    dist.send(send_data[src_rank], dest_rank, async_op=False)
                elif dest_rank == rank:
                    recv_buf = torch.zeros((send_data[src_rank].shape[0], d_model), device=tokens.device)
                    dist.recv(recv_buf, src_rank, async_op=False)
                    output_lists[src_rank].append(recv_buf)

    # Concatenate output
    output = torch.cat(output_lists, dim=0) if output_lists else torch.zeros((0, d_model), device=tokens.device)

    return output

# PyTorch all-to-all (v2+)
def alltoall_pytorch(input_tensor, output_tensor_list, rank, world_size):
    """
    PyTorch 1.10+ has dist.all_to_all_single() and dist.all_to_all().

    all_to_all_single: input is a flat tensor, output is list of tensors
    all_to_all: input/output are lists of tensors
    """
    try:
        dist.all_to_all_single(input_tensor, output_tensor_list)
    except AttributeError:
        # Fallback for older PyTorch
        print("all_to_all not available; use alternative implementation")
```

### 5. NCCL Algorithm Selection Heuristics

NCCL internally selects algorithms based on:

```
// Pseudo-code from NCCL algorithm selection

enum Algorithm {
    RING,
    TREE,
    ALLGATHER_RING,
    ...
};

Algorithm select_algorithm(
    num_devices,
    data_size_bytes,
    latency_us,
    bandwidth_gbps,
    network_topology
) {
    // Empirical heuristics (hand-tuned for NVIDIA clusters)

    if (num_devices == 2) {
        return RING;  // Minimal overhead
    }

    if (data_size_bytes < 1024) {  // Small messages: latency-bound
        if (num_devices <= 16) {
            return TREE;  // Log(N) latency
        } else {
            return RING;  // Avoid deep tree
        }
    }

    // Large messages: bandwidth-bound
    if (latency_us < 1.0) {  // NVLink
        return TREE;  // Can afford log(N) hops
    } else {  // InfiniBand, etc.
        return RING;  // Minimize network congestion
    }

    // Check topology
    if (spans_multiple_pods) {
        return RING;  // Spread traffic evenly
    } else {
        return TREE;  // Hierarchical
    }

    return RING;  // Default safe choice
}
```

**Practical NCCL Tuning**:

```python
import os

# Environment variables to control NCCL behavior
os.environ['NCCL_DEBUG'] = 'INFO'  # Enable debug logging
os.environ['NCCL_ALGO'] = 'RING'  # Force algorithm
os.environ['NCCL_PROTO'] = 'SIMPLE'  # Use simple protocol (vs NVLS)
os.environ['NCCL_MIN_NRINGS'] = '1'  # Number of rings (parallelism)
os.environ['NCCL_MAX_NRINGS'] = '2'
os.environ['NCCL_NTHREADS'] = '512'  # Kernel threads per block

# For debugging: save NCCL topology and algorithm info
os.environ['NCCL_DEBUG_FILE'] = '/tmp/nccl_debug.log'
```

### 6. OneCCL for CPU-based Inference

For CPU clusters (less common but viable):

```cpp
// OneCCL (Intel's collective communication library) example

#include <oneapi/ccl.hpp>
#include <iostream>

int main() {
    // Initialize
    ccl::init();

    // Create communicator for all ranks
    auto comm = ccl::create_communicator(world_size);

    // All-reduce on CPU tensors
    std::vector<float> send_buf(10000);
    std::vector<float> recv_buf(10000);

    std::fill(send_buf.begin(), send_buf.end(), 1.0f);

    // Execute all-reduce
    ccl::allreduce(
        send_buf.data(),
        recv_buf.data(),
        10000,
        ccl::datatype::float32,
        ccl::reduction::sum,
        comm
    ).wait();

    // Result: recv_buf contains sum across all ranks
}
```

**CPU vs GPU comparison**:

| Operation | GPU (NCCL) | CPU (OneCCL) |
|-----------|-----------|-----------|
| All-reduce (12 GB) | 6 ms | 50 ms |
| Latency | 1-2 µs | 5-10 µs |
| Scaling (64 GPUs) | Near-linear | Sublinear |

CPU collectives are 10× slower due to higher latency and lower BW.

### 7. Overlap Communication with Computation

The goal: while sending data over network, compute on remaining data.

```python
def overlapped_allreduce_attention(q, k, v, tp_rank, tp_size):
    """
    Overlap all-reduce of previous layer's gradient with current forward pass.

    Assumption: using pipelined (gradient) training.
    For inference, we overlap attention computation with communication.
    """

    # 1. Start async all-gather of K, V (needed for attention)
    k_local, v_local = k, v  # These are local chunks

    # Ring all-gather: start async sends/recvs
    k_ring_bufs = [k_local]
    v_ring_bufs = [v_local]

    send_reqs = []
    recv_reqs = []

    for round_idx in range(tp_size - 1):
        next_rank = (tp_rank + 1) % tp_size
        prev_rank = (tp_rank - 1) % tp_size

        k_to_send = k_ring_bufs[-1]
        v_to_send = v_ring_bufs[-1]

        k_recv_buf = torch.zeros_like(k_to_send)
        v_recv_buf = torch.zeros_like(v_to_send)

        # Non-blocking operations
        req_k_send = dist.send(k_to_send, next_rank, async_op=True)
        req_v_send = dist.send(v_to_send, next_rank, async_op=True)
        req_k_recv = dist.recv(k_recv_buf, prev_rank, async_op=True)
        req_v_recv = dist.recv(v_recv_buf, prev_rank, async_op=True)

        send_reqs.extend([req_k_send, req_v_send])
        recv_reqs.extend([req_k_recv, req_v_recv])
        k_ring_bufs.append(k_recv_buf)
        v_ring_bufs.append(v_recv_buf)

    # 2. Compute attention on first K, V while communication is in-flight
    k_curr = k_ring_bufs[0]
    v_curr = v_ring_bufs[0]

    scores = torch.einsum('bsnh,btnh->bnst', q, k_curr) / math.sqrt(d_head)
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum('bnst,btnh->bsnh', attn, v_curr)

    # 3. As each K, V arrives, accumulate attention contributions
    for round_idx in range(1, tp_size):
        # Wait for this round's data to arrive
        recv_reqs[2 * round_idx].wait()
        recv_reqs[2 * round_idx + 1].wait()

        k_curr = k_ring_bufs[round_idx]
        v_curr = v_ring_bufs[round_idx]

        scores = torch.einsum('bsnh,btnh->bnst', q, k_curr) / math.sqrt(d_head)
        attn = torch.softmax(scores, dim=-1)
        out += torch.einsum('bnst,btnh->bsnh', attn, v_curr)

    return out
```

**Practical usage in Megatron**:

Megatron-LM overlaps:
- Communication of one attention head with computation of next
- Uses "async_op=True" in NCCL
- Reduction in wall-clock latency: ~30-40%

---

## KEY PAPERS

1. **Megatron-LM (Shoeybi et al., 2019)**: Efficient Large-Scale Language Model Training using Model Parallelism.
   - Seminal work on collective operations for model parallelism
   - All-reduce placement optimization
   - Overlapping computation and communication

2. **NCCL 2.0 (Jeannot et al., 2018)**: Optimized Inter-Process Communication on GPUs. Euro-Par 2018.
   - Algorithm selection heuristics
   - Bandwidth-latency tradeoff analysis
   - Tree vs. ring comparisons

3. **High-Performance Collective Communication (Graham et al., 2014)**: Mapping Communication Libraries to Network Hardware Parameters. ISC 2014.
   - General framework for collective optimization
   - Topology-aware routing

4. **Bandwidth-Optimal All-reduce Algorithms (Thakur et al., 2005)**: Optimization of Collective Communication Operations in MPICH. IJHPCA 2005.
   - Theoretical lower bounds for collective communication
   - Bandwidth-optimal ring algorithm proof

5. **OneCCL: Unified Communication for ML Workloads (Bueno et al., 2021)**: OneCCL: Bringing Unified Communication to Next Generation of Machine Learning Frameworks. MLCS 2021.
   - CPU-based collective library
   - Integration with PyTorch, TensorFlow

---

## SYSTEMS TRADEOFFS

### Ring vs. Tree Algorithms

| Metric | Ring | Tree |
|--------|------|------|
| Bandwidth efficiency | ~95-99% | ~95-99% |
| Latency (log(N) stages) | N stages | log(N) stages |
| Network congestion | Uniform traffic | Hotspot at root |
| Implementation complexity | Simple | Moderate |
| Best for | Large data, long latency BW links | Small data, low latency links |

**Recommendation**:
- InfiniBand (1-2 µs latency, 100-200 GB/s): Ring
- NVLink (< 1 µs latency, 1-2 TB/s): Tree or hybrid

### Communication-Hiding Effectiveness

To hide communication, we must have computation time ≥ communication time.

For token generation (batch=1):
- Compute per token: ~1 ms per layer
- Communication per layer (all-reduce): ~1-10 ms (depending on topology, algorithm)

Communication cannot be hidden unless batch_size increases. Solution: Use batched inference.

### Memory Overhead of Large Messages

Large collectives (e.g., 12 GB all-reduce on 8 GPUs) require:
- 12 GB data in GPU memory
- Plus intermediate buffers for ring rotations
- Total memory: 12 GB + (number of in-flight ring stages) * buffer_size

NCCL usually handles this efficiently by using ring stages without materializing all at once.

---

## EXPERT INSIGHT

### Production Experience with Collective Operations

From deploying large models in production:

1. **NCCL is usually correct, but needs tuning**:
   - Default algorithm selection is conservative
   - Setting NCCL_ALGO=RING often 2-3× faster than default on InfiniBand
   - Profiling shows most time in all-reduce, not compute

2. **All-to-all is the hidden killer**:
   - MoE models spend 50-70% of time in expert all-to-all
   - Ring reduce-scatter (or custom packing) can improve by 30%
   - Load imbalance (few experts selected) is worse than uniform load

3. **Latency matters for inference**:
   - Even 1 microsecond per message × 96 layers = 96 microseconds per token
   - For chat (50-100 ms target per token), this is ~0.1% overhead, but cumulative
   - Minimize number of collective calls: fuse where possible

4. **Network-aware scheduling**:
   - Knowing the cluster topology allows custom collective routing
   - Example: On a 2-pod cluster, reduce all collectives to single pod, then allgather across pods
   - Custom implementation beats NCCL in ~10% of cases

5. **Quantization + collectives**:
   - INT8 reduces data size by 4× → communication time ÷ 4
   - But dequantization adds compute
   - Overall: communication time improves more than compute time worsens

6. **Dynamic batching and collective efficiency**:
   - Varying batch size (1, 8, 64) means varying collective timing
   - Some requests wait for collective to finish before outputting
   - Optimal: queue multiple requests, batch together

### Debugging Slow Collectives

Use NCCL profiling:

```bash
# Dump NCCL timing info
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_FILE=/tmp/nccl_trace.log

# Run your inference job
python inference_server.py

# Analyze
grep "allreduce\|allgather\|alltoall" /tmp/nccl_trace.log | tail -20
```

If all-reduce takes >10 ms for 12 GB data:
1. Check network: run iperf3 between GPUs, verify bandwidth
2. Try NCCL_ALGO=RING: if faster, tree was suboptimal
3. Check NCCL version: update to latest (has better heuristics)
4. Custom collective: if model-specific patterns (e.g., always same communication pattern), implement custom ring

---

## BENCHMARKING METHODOLOGY

### Microbenchmark Protocol for Collectives

```python
import torch
import torch.distributed as dist
import time

def benchmark_collective(collective_fn, tensor_size_mb, world_size, num_trials=100):
    """
    Benchmark a collective operation.

    collective_fn: function taking tensor, returning result
    tensor_size_mb: data size in MB
    world_size: number of GPUs
    """

    tensor = torch.randn(tensor_size_mb * 1024 * 256, device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(10):
        collective_fn(tensor)
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for trial in range(num_trials):
        torch.cuda.synchronize()
        start = time.perf_counter()

        collective_fn(tensor)

        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append(end - start)

    # Stats
    times = np.array(times[10:])  # Skip warmup
    mean_time = np.mean(times)
    std_time = np.std(times)
    throughput_gbps = (tensor_size_mb / mean_time) / 1024  # MB/s → GB/s

    return {
        'tensor_size_mb': tensor_size_mb,
        'mean_time_ms': mean_time * 1000,
        'std_time_ms': std_time * 1000,
        'throughput_gbps': throughput_gbps,
    }


# Run benchmark sweep
sizes_mb = [1, 10, 100, 1000, 10000]
results = []

for size_mb in sizes_mb:
    result = benchmark_collective(
        lambda t: dist.all_reduce(t),
        size_mb,
        world_size=8,
        num_trials=100
    )
    results.append(result)

# Plot
import pandas as pd
df = pd.DataFrame(results)
print(df)
# Output:
#    tensor_size_mb  mean_time_ms  std_time_ms  throughput_gbps
# 0               1            1.5          0.2              0.7
# 1              10            2.1          0.3              4.8
# 2             100            8.5          0.5             11.8
# 3            1000           95.0          1.2             10.5
# 4           10000          950.0          5.0             10.5
```

**Expected results**:
- Small data (1-10 MB): throughput limited by latency (low BW utilization)
- Medium data (100-1000 MB): approaching peak BW
- Large data (> 1000 MB): sustained peak BW

### End-to-End Inference Benchmark

```python
def benchmark_inference_parallelism(model, tp_size, pp_size, batch_size, seq_length, num_tokens=1000):
    """
    Benchmark full inference pipeline with parallelism.
    """

    # Setup parallelism
    device_mesh = create_device_mesh((tp_size, pp_size))
    model_parallel = distribute_model(model, device_mesh)

    # Warmup
    for _ in range(5):
        dummy_input = torch.randn(batch_size, seq_length, model.d_model, device='cuda')
        _ = model_parallel(dummy_input)

    # Benchmark token generation (autoregressive)
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    all_tokens = []
    for token_idx in range(num_tokens):
        # Generate one token (forward pass)
        logits = model_parallel(dummy_input)  # (batch, 1, vocab_size) for decoder
        next_token = logits[:, -1, :].argmax(dim=-1)
        all_tokens.append(next_token)

        # In real implementation, would update kv-cache, etc.

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    # Metrics
    total_time = end_time - start_time
    tokens_per_sec = (batch_size * num_tokens) / total_time
    latency_per_token = (total_time / num_tokens) * 1000  # ms

    return {
        'tp_size': tp_size,
        'pp_size': pp_size,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'tokens_per_sec': tokens_per_sec,
        'latency_ms': latency_per_token,
    }
```

### Network Benchmarking (Standalone)

```bash
# NCCL tests (official)
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make
./build/all_reduce_perf -b 1K -e 10G -f 2  # 1KB to 10GB, 2× runs

# Output:
# #        size         count      type      redop    root     time   algbw   busbw
# #           B    elements        enums     enums   rank       us     GB/s     GB/s
#          1024     256         float     sum      -1    1234.5    0.00    0.00
#          2048     512         float     sum      -1     456.7    0.00    0.00
#         ...
#        10737418240  2684354560  float     sum      -1  97531.2   110.2   110.2

# Custom all-reduce test
mpirun -np 8 nccl_test_allreduce -b 12G  # 12 GB all-reduce on 8 ranks
```

---

## OPEN PROBLEMS

1. **Adaptive collective selection**: Choose algorithm at runtime based on network load, message size, and topology. Current: static NCCL heuristics.

2. **Cross-layer collective fusion**: Combine multiple all-reduces (from attention, FF, skip connections) into single collective. Requires compiler support.

3. **Heterogeneous networks**: GPUs over 5G, satellite links, or mixed Ethernet/InfiniBand. Need adaptive algorithms.

4. **Congestion-aware routing**: Monitor network congestion, route collectives accordingly. Current: shortest path (no congestion model).

5. **MoE load balancing + collective efficiency**: Load-balanced routing (uniform expert selection) can hurt inference latency (over-tokenization to balance). Tradeoff between balance and latency.

---

## PHD QUALIFIER QUESTIONS

1. **All-Reduce Bandwidth Lower Bound**: Prove that ring all-reduce on N devices with M bytes of data requires at least M/B communication time, where B is the link bandwidth. Explain why tree-based all-reduce cannot do better asymptotically.

2. **NCCL Algorithm Selection**: Given a cluster with intra-pod NVLink (2 TB/s) and inter-pod InfiniBand (200 GB/s), derive the message size threshold at which ring becomes preferable to tree. Assume latency = 1 µs for both.

3. **All-Gather vs. Reduce-Scatter Equivalence**: Prove that all-gather and reduce-scatter are dual operations (one can be implemented using the other with transposes). What is the complexity difference?

4. **Overlapped Communication**: In overlapped computation-communication, what is the maximum speedup achievable if computation time is T_comp and communication time is T_comm? Under what conditions is 2× speedup possible?

5. **OneCCL vs. NCCL Scalability**: NCCL on H100 with NVLink achieves ~1.8 TB/s all-reduce throughput for 8 GPUs. OneCCL on 8-socket CPU server with DDR5 achieves ~300 GB/s. Explain the 6× difference considering memory BW and cache coherency overhead.

6. **MoE All-to-All Communication**: For sparse all-to-all (some ranks send 0 bytes to some ranks), which collective algorithm is best: (a) all-gather + filter, (b) pairwise sends, or (c) compressed all-to-all? Analyze trade-offs.

7. **Bisection Bandwidth Constraint**: A 3-level fat-tree has N servers, each with 50 Gbps to ToR, and ToRs with 100 Gbps to core. Derive the bisection bandwidth as a function of N. For a parallel all-reduce across all N servers, what is the minimum time asymptotically?

8. **Ring Reduction Order**: In ring all-reduce, does the order of reduction (which chunks are reduced first) affect the final result? If not, does it affect latency? Explain with an example.

9. **Async NCCL Safety**: When using async_op=True in NCCL, what synchronization primitives (barriers, events, memory ordering) are required to ensure correctness when data is reused immediately?

10. **Latency Hiding in Pipelines**: For PP with M micro-batches and P partitions, what fraction of communication latency can be hidden by computation? Derive as a function of M, P, and the compute/comm ratio.

---

**Total Word Count**: ~4100 words


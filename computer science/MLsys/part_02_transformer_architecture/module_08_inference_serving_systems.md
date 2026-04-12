# MODULE 8: Inference Serving Systems Architecture

## 1. SYSTEMS OVERVIEW

Large language model serving represents a qualitatively different systems challenge than traditional ML inference. Where standard inference systems optimize latency or throughput independently, LLM serving must optimize both simultaneously while maintaining predictable SLAs under dynamic workloads. This module explores the complete architecture of production LLM serving systems, from scheduling algorithms to disaggregated prefill/decode infrastructure.

### 1.1 The LLM Serving Challenge

Traditional inference (batch processing):
- Input arrives → GPU processes → output
- Latency = batch setup + forward pass + output gathering
- Throughput = batches per second

LLM inference (token generation):
- Input arrives → prefill phase (process entire prompt) → decode phase (generate tokens one at a time)
- For a 2048-token prompt generating 256 tokens:
  - Prefill: 2048 tokens into GPU (high parallelism)
  - Decode: 256 sequential iterations (low parallelism per iteration)
- New requests arrive while old requests are still decoding
- Some sequences finish decoding while others are in prefill phase

**The fundamental problem**: Prefill and decode have opposite resource requirements:
- **Prefill**: High arithmetic intensity, benefits from batching, compute-bound
- **Decode**: Low arithmetic intensity, limited batching benefit, memory-bound

Scheduling all prefill, then all decode (strict phases) leaves GPU idle.

### 1.2 Orca's Key Insight: Continuous Batching

Pre-Orca (static batching):
```
Time: |Batch 1 (reqs 1-4 prefill + 10 decode steps)|  empty  |Batch 2|
GPU:  GPU is idle after requests finish decoding (different finish times)
```

Orca (continuous batching):
```
Time: |Req1 prefill|Req1 decode1|Req2,3 prefill|Req1,2 decode|Req3 decode|Req4 prefill|Req2,3,4 decode|
GPU:  Continuous work—interleave requests at different phases
```

**Result**: Increase GPU utilization from 40% to 80%+ by never letting GPU sit idle.

### 1.3 System Components

A production LLM serving system requires:

1. **Request scheduler** (FCFS, preemptive, SJF)
2. **Batching logic** (static vs continuous)
3. **Memory manager** (KV-cache allocation—PagedAttention)
4. **Kernel optimization** (FlashAttention)
5. **Prefill/decode disaggregation** (if needed)
6. **SLA enforcement** (latency percentiles)
7. **Autoscaling** (add/remove GPUs dynamically)

This module covers each component with production implementation details.

---

## 2. THEORETICAL FOUNDATION

### 2.1 Queueing Theory and Latency Analysis

Model inference as an M/G/1 queue (Poisson arrivals, General service time, 1 server):

**Arrival process**: Requests follow Poisson with rate λ (requests/sec)

**Service time distribution**: G depends on:
- Prefill time: t_prefill = input_len / throughput_prefill
- Decode time: t_decode = output_len / throughput_decode
- Total: T = t_prefill + t_decode (random variables)

**Little's Law**:
```
L = λ × W
```
where L = average number in system, W = average time in system

For an LLM serving system:
```
W = t_prefill + t_decode  (end-to-end latency)
L = λ × (t_prefill + t_decode)

Average GPU queue length at 10 req/sec, 100ms average latency:
L = 10 × 0.1 = 1 request
```

**SLA constraint**: P(latency > SLA) < ε

For exponential service time: P(W > t) = e^(-t/W_avg)

To achieve P(latency > 1s) < 5% with W_avg = 200ms:
```
e^(-1.0/0.2) = e^(-5) ≈ 0.0067 (0.67% < 5%) ✓
```

### 2.2 Batch Composition and Arithmetic Intensity

For a single GPU batch with heterogeneous sequence lengths:

```
Batch composition:
- 4 sequences in prefill phase (lengths: 512, 1024, 256, 768)
- 8 sequences in decode phase (uniform cost per sequence)

Prefill cost (FLOPs):
For each token in each sequence: 2 × seq_len × d × num_heads
Total_prefill = Σ(2 × input_len_i × d × num_heads) for i=1..4
             = 2 × (512 + 1024 + 256 + 768) × 128 × 64
             = 2 × 2560 × 8192 ≈ 42M FLOPs

Decode cost (FLOPs):
For each sequence: 2 × 1 × seq_len_in_cache × d × num_heads
Total_decode = 8 sequences × 2 × (avg_seq_len) × 128 × 64
```

**Arithmetic intensity** (FLOPs per byte of memory access):

Prefill:
```
I_prefill = (2 × seq_len × d) / (seq_len × d × 2)
          = 1 FLOP per byte (low intensity)
```

Decode (per token):
```
I_decode = (2 × seq_len × d) / (2 × d)
         = seq_len FLOPs per byte (high intensity!)
```

For seq_len=2048, decode is 2048× more arithmetic-intensive than prefill!

### 2.3 Scheduling Algorithms: FCFS vs SJF vs Preemptive

**FCFS (First-Come, First-Served)**:
```
Arrival order: A(100ms), B(200ms), C(50ms)
Service time: A→500ms, B→300ms, C→400ms

Timeline:
0-500ms: A
500-800ms: B
800-1200ms: C

Average latency = (500 + (800-200) + (1200-50)) / 3 = 766ms
Completion time for C = 1200ms (arrived at 50ms!)
```

**SJF (Shortest Job First)**:
```
Reorder by service time: C(400ms), B(300ms), A(500ms)

Timeline:
0-400ms: C
400-700ms: B
700-1200ms: A

Average latency = (400-50 + 700-200 + 1200) / 3 = 650ms
Completion time for A = 1200ms (same as FCFS)
```

**Insight**: SJF minimizes average latency but:
- Can starve long requests (tail latency suffers)
- Requires knowing job length in advance (not possible for variable-output generation)

**Preemptive scheduling**:
```
Introduce preemption points (after each token in decode)

Token 1: A.prefill=100ms → run
Token 1: B.prefill=200ms → run
Token 1: C.prefill=50ms → run
Token 1: A.decode=20ms → run
Token 1: B.decode=20ms → run
...

Interleaving ensures no single request monopolizes GPU
```

### 2.4 Continuous Batching Algorithm

```
Algorithm: Continuous Batching Scheduler

Maintain:
- prefill_queue: requests waiting for prefill
- running_prefill: sequences currently in prefill
- running_decode: sequences currently decoding
- scheduled_tokens: tokens to generate this iteration

Iteration logic:
1. Try to schedule prefill sequences
   - Add sequences to running_prefill until:
     - Prefill queue empty, OR
     - Total prefill tokens > max_prefill_tokens, OR
     - Batch memory exceeds limit

2. Schedule decode tokens
   - Each sequence in running_decode gets 1 token
   - Generate tokens in parallel

3. Update state
   - Remove completed decode sequences
   - Move completed prefill sequences to running_decode

4. Compute
   - Run forward pass on combined prefill + decode batch

5. Check completeness
   - If sequence reaches max_new_tokens, move to output_queue
```

**Key insight**: By running prefill and decode in the same batch, we achieve high GPU utilization even with variable sequence lengths.

### 2.5 Memory Bandwidth as the Bottleneck

During decode:
```
FLOPs per request = 2 × seq_len × d_model × num_layers

For Llama 2 7B (d_model=4096, num_layers=32):
FLOPs_per_token = 2 × seq_len × 4096 × 32

Memory access:
- Read model weights: 7B × 2 bytes = 14 GB (once per batch)
- Read KV-cache: seq_len × 2 × 64 × 128 × 2 bytes (per token)
- For seq_len=2048: 33 MB per sequence

At 1000 sequences being decoded:
Memory bandwidth required = 1000 × 33 MB / 10ms = 3.3 TB/s
GPU bandwidth available: 2 TB/s → BOTTLENECKED
```

This motivates **disaggregated prefill/decode**: use different resources for each phase.

### 2.6 Disaggregated Architecture Analysis

**Co-located prefill/decode** (vLLM default):
```
GPU utilization (prefill): 80%
GPU utilization (decode): 20%
Mixed utilization: 50% (average)
```

**Disaggregated** (Splitwise, DistServe):
```
Prefill GPUs: 100% utilization (only do prefill)
Decode GPUs: 100% utilization (only do decode)
Average: 100% utilization (but requires 2× GPUs)
```

**When disaggregation wins**:
- Workload is 50% prefill, 50% decode (balanced)
- Large batch sizes (N >> 10 sequences)
- Multi-GPU cluster available

**When co-location wins**:
- Single GPU systems
- Batch size < 4 (decoding already underutilizes GPU)

---

## 3. HARDWARE MAPPING

### 3.1 GPU Memory Hierarchy Impact on Batching

During a decode iteration with B sequences:

```
GPU memory access pattern:

1. Load model weights (7B params): ~14 GB (once, amortized)
2. Load KV-cache for B sequences:
   B × avg_seq_len × 2 × num_heads × head_dim × 2 bytes
   = B × 2000 × 2 × 64 × 128 × 2 = B × 67 MB

Example with B=32:
2. Load KV-cache: 32 × 67 MB = 2.1 GB

Total memory movement: 14 + 2.1 = 16.1 GB

Time on 2 TB/s GPU: 16.1 GB / (2 TB/s) = 8 ms

Compute: 2 × 1 (token) × 4096 (dim) × 32 (layers) × 32 (sequences)
       = 8.4M FLOPs
Time at 1.5 TFLOPS: 8.4M / 1.5T = 5.6 us (!)

Memory time (8ms) >> Compute time (5.6us): bandwidth-bound
```

### 3.2 Batching Impact on Arithmetic Intensity

Single sequence decode (seq_len=2048):
```
I = (2 × 1 × 2048 × 4096) / (2 × 4096 × 2) = 1024 FLOPs/byte
```

32 sequence decode (seq_len=2048 each):
```
I = (32 × 2 × 1 × 2048 × 4096) / (32 × 2 × 4096 × 2) = 1024 FLOPs/byte
```

**Key insight**: Arithmetic intensity for decode doesn't change with batch size! The bottleneck is always memory.

However, **kernel launch overhead** amortizes with batch size:
```
Kernel launch overhead: ~1 ms
Batch 1: 1 ms overhead / 1 sequence = 1 ms per sequence
Batch 32: 1 ms overhead / 32 sequences = 0.03 ms per sequence
```

### 3.3 Multi-GPU Prefill/Decode Placement

For a 4-GPU system (2 prefill, 2 decode GPUs):

**Prefill GPU**:
```
Parallelism: 4096 tokens × 32 sequences = high parallelism
Arithmetic intensity: ~1 FLOP/byte
GPU utilization: 60-70% (compute-bound with prefill tokens)
```

**Decode GPU**:
```
Parallelism: 1 token × B sequences
Arithmetic intensity: seq_len FLOPs/byte (2000+)
GPU utilization: 100% (memory-bound)
```

**Communication overhead**:
```
After prefill, KV-cache (67 MB) transferred to decode GPU
At 100 Gbps (12.5 GB/s): 67 MB / 12.5 GB/s = 5.4 ms overhead per sequence
```

Disaggregation only wins if decode savings > 5.4 ms per sequence.

---

## 4. IMPLEMENTATION DEEP DIVE

### 4.1 Continuous Batching Controller

```python
# batch_controller.py
import heapq
from dataclasses import dataclass
from typing import List, Dict
import torch

@dataclass
class RequestState:
    request_id: int
    input_ids: List[int]
    output_ids: List[int]
    kv_cache: Dict  # BlockManager reference
    phase: str  # 'prefill' or 'decode'
    num_compute_tokens: int  # Tokens to compute this iteration
    completion_time: float
    arrival_time: float

class ContinuousBatchingScheduler:
    """
    Orchestrates prefill and decode phases for efficient GPU utilization
    """

    def __init__(
        self,
        max_batch_prefill_tokens: int = 4096,  # Total prefill tokens per iteration
        max_batch_decode_sequences: int = 64,  # Decode sequences per iteration
        max_seq_len: int = 4096,
    ):
        self.max_batch_prefill_tokens = max_batch_prefill_tokens
        self.max_batch_decode_sequences = max_batch_decode_sequences
        self.max_seq_len = max_seq_len

        # State
        self.prefill_queue: List[RequestState] = []  # Waiting for prefill
        self.running_requests: Dict[int, RequestState] = {}  # Currently running
        self.completed_requests: List[RequestState] = []
        self.request_counter = 0

    def add_request(
        self,
        input_ids: List[int],
        max_new_tokens: int,
    ) -> int:
        """Add new request to the system"""
        req_id = self.request_counter
        self.request_counter += 1

        req = RequestState(
            request_id=req_id,
            input_ids=input_ids,
            output_ids=[],
            kv_cache=None,
            phase='prefill',
            num_compute_tokens=len(input_ids),
            completion_time=None,
            arrival_time=time.time(),
        )

        self.prefill_queue.append(req)
        return req_id

    def step(self) -> Dict[str, List[RequestState]]:
        """
        Execute one scheduling iteration

        Returns:
        {
            'prefill_requests': [reqs to prefill],
            'decode_requests': [reqs to decode],
        }
        """
        # Remove completed requests from running set
        self.running_requests = {
            rid: req for rid, req in self.running_requests.items()
            if len(req.output_ids) < req.max_new_tokens  # Still generating
        }

        scheduled_prefill = []
        total_prefill_tokens = 0

        # Schedule prefill requests
        while self.prefill_queue:
            req = self.prefill_queue[0]
            prefill_tokens = len(req.input_ids)

            # Check constraints
            if (total_prefill_tokens + prefill_tokens > self.max_batch_prefill_tokens and
                scheduled_prefill):  # At least one request
                break

            # Move to running
            req.phase = 'prefill'
            req.num_compute_tokens = prefill_tokens
            self.prefill_queue.pop(0)
            self.running_requests[req.request_id] = req
            scheduled_prefill.append(req)
            total_prefill_tokens += prefill_tokens

        # Schedule decode requests
        scheduled_decode = [
            req for rid, req in self.running_requests.items()
            if req.phase == 'decode'  # Requests already past prefill
        ]

        # Limit decode batch size
        scheduled_decode = scheduled_decode[:self.max_batch_decode_sequences]

        # Mark requests for their phase
        for req in scheduled_prefill:
            req.phase = 'prefill'
            req.num_compute_tokens = len(req.input_ids)

        for req in scheduled_decode:
            req.phase = 'decode'
            req.num_compute_tokens = 1  # One token per decode step

        return {
            'prefill_requests': scheduled_prefill,
            'decode_requests': scheduled_decode,
        }

    def update_after_forward(self, scheduled, output_tokens):
        """Update request states after forward pass"""
        prefill_reqs = scheduled['prefill_requests']
        decode_reqs = scheduled['decode_requests']

        # Mark prefill requests as done with prefill
        for i, req in enumerate(prefill_reqs):
            req.phase = 'decode'
            req.num_compute_tokens = 0

        # Update decode requests with new tokens
        for i, req in enumerate(decode_reqs):
            new_token = output_tokens[len(prefill_reqs) + i]
            req.output_ids.append(new_token)

        # Check for completion
        for req in list(self.running_requests.values()):
            if len(req.output_ids) >= self.max_new_tokens:
                req.completion_time = time.time()
                self.completed_requests.append(req)
                del self.running_requests[req.request_id]

    def get_metrics(self):
        """Return latency and throughput metrics"""
        if not self.completed_requests:
            return {}

        latencies = [
            req.completion_time - req.arrival_time
            for req in self.completed_requests
        ]

        return {
            'avg_latency_sec': sum(latencies) / len(latencies),
            'p99_latency_sec': sorted(latencies)[int(0.99 * len(latencies))],
            'throughput_tps': len(self.completed_requests) / (time.time() - self.completed_requests[0].arrival_time),
        }
```

### 4.2 Batched Forward Pass with Continuous Batching

```python
# forward_pass.py
import torch
import torch.nn.functional as F

class ContinuousBatchForward:
    """
    Execute forward pass with mixed prefill/decode batch
    """

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def forward(
        self,
        prefill_reqs: List[RequestState],
        decode_reqs: List[RequestState],
        kv_cache_manager,
    ):
        """
        Combined forward pass for prefill + decode

        Strategy:
        1. Process all prefill requests
        2. Process all decode requests
        3. Combine outputs
        """

        outputs = []

        if prefill_reqs:
            # Prefill: process entire prompt
            prefill_input_ids = []
            prefill_positions = []
            prefill_start_pos = 0

            for req in prefill_reqs:
                prompt_len = len(req.input_ids)
                prefill_input_ids.extend(req.input_ids)
                prefill_positions.extend(range(prefill_start_pos, prefill_start_pos + prompt_len))
                prefill_start_pos += prompt_len

            # Stack into batch
            prefill_input = torch.tensor(prefill_input_ids, device=self.device).unsqueeze(0)
            prefill_positions = torch.tensor(prefill_positions, device=self.device).unsqueeze(0)

            # Forward pass for prefill
            with torch.no_grad():
                prefill_logits, prefill_cache = self.model(
                    prefill_input,
                    positions=prefill_positions,
                    kv_cache=None,  # No KV cache input for prefill
                    return_kv=True,
                )

            # Store KV cache for each request
            # (In production, use paged cache)
            for i, req in enumerate(prefill_reqs):
                prompt_len = len(req.input_ids)
                # Cache handling logic
                req.kv_cache = prefill_cache  # Simplified

            # Get logits for last token of each request
            prefill_outputs = []
            pos = 0
            for req in prefill_reqs:
                prompt_len = len(req.input_ids)
                last_logits = prefill_logits[0, pos + prompt_len - 1, :]
                prefill_outputs.append(last_logits)
                pos += prompt_len

        if decode_reqs:
            # Decode: one token per request, parallel
            decode_input_ids = [
                req.output_ids[-1] if req.output_ids else req.input_ids[-1]
                for req in decode_reqs
            ]
            decode_input = torch.tensor(decode_input_ids, device=self.device).unsqueeze(1)

            # Positions are lengths of generated sequences
            decode_positions = torch.tensor([
                len(req.input_ids) + len(req.output_ids) - 1
                for req in decode_reqs
            ], device=self.device).unsqueeze(1)

            # Forward pass for decode (with KV cache)
            with torch.no_grad():
                decode_logits, _ = self.model(
                    decode_input,
                    positions=decode_positions,
                    kv_cache=[req.kv_cache for req in decode_reqs],
                    return_kv=True,
                )

            decode_outputs = decode_logits[:, 0, :]  # [batch, vocab_size]

        # Sample next tokens
        all_outputs = []
        if prefill_reqs:
            for logits in prefill_outputs:
                next_token = torch.argmax(logits, dim=-1).item()
                all_outputs.append(next_token)

        if decode_reqs:
            for logits in decode_outputs:
                next_token = torch.argmax(logits, dim=-1).item()
                all_outputs.append(next_token)

        return all_outputs  # [len(prefill_reqs) + len(decode_reqs)]

    def forward_with_paged_cache(
        self,
        prefill_reqs,
        decode_reqs,
        kv_cache_manager,
    ):
        """
        Use paged KV-cache (PagedAttention) for memory efficiency
        """
        # Similar to above but:
        # 1. Allocate/manage block pages via kv_cache_manager
        # 2. Pass block tables to attention kernel
        # 3. Handle block allocation/deallocation

        # Prefill phase
        for req in prefill_reqs:
            prompt_len = len(req.input_ids)
            num_blocks = (prompt_len + kv_cache_manager.block_size - 1) // kv_cache_manager.block_size
            blocks = kv_cache_manager.allocate(req.request_id, num_blocks)
            # ... (run forward pass) ...
            kv_cache_manager.write_tokens(req.request_id, K, V, start_pos=0)

        # Decode phase
        for req in decode_reqs:
            # Request allocated blocks already
            # Allocate one more block for next token
            kv_cache_manager.extend(req.request_id, num_blocks=1)
            # ... (run forward pass) ...
```

### 4.3 SLA Enforcement and Preemptive Scheduling

```python
# sla_controller.py
import time
from typing import Dict, List
import numpy as np

class SLAController:
    """
    Monitor and enforce SLA constraints (latency percentiles)
    """

    def __init__(
        self,
        sla_p99_ms: float = 1000,  # 99th percentile latency target
        sla_p95_ms: float = 500,
    ):
        self.sla_p99_ms = sla_p99_ms / 1000  # Convert to seconds
        self.sla_p95_ms = sla_p95_ms / 1000
        self.latencies: List[float] = []

    def check_sla_violation(self, request: RequestState) -> bool:
        """Check if request violates SLA if continued"""
        elapsed = time.time() - request.arrival_time
        remaining_budget = self.sla_p99_ms - elapsed

        # Estimate remaining time
        avg_decode_time_per_token = 0.01  # 10ms per token
        estimated_remaining = avg_decode_time_per_token * (
            request.max_new_tokens - len(request.output_ids)
        )

        return estimated_remaining > remaining_budget

    def get_priority_order(self, requests: List[RequestState]) -> List[RequestState]:
        """
        Prioritize requests that are about to violate SLA
        """
        now = time.time()

        def priority_score(req):
            elapsed = now - req.arrival_time
            remaining_budget = self.sla_p99_ms - elapsed

            # Higher priority = lower score (to be processed first)
            if remaining_budget < 0:
                return -1  # Already violated, process first
            elif remaining_budget < 50e-3:  # Less than 50ms remaining
                return remaining_budget
            else:
                return float('inf')  # Not at risk

        # Sort by priority (earliest deadline first)
        return sorted(requests, key=priority_score)

    def record_latency(self, request: RequestState):
        """Track latency for SLA monitoring"""
        latency = request.completion_time - request.arrival_time
        self.latencies.append(latency)

    def get_sla_metrics(self) -> Dict:
        """Report SLA compliance"""
        if not self.latencies:
            return {}

        p95_actual = np.percentile(self.latencies, 95)
        p99_actual = np.percentile(self.latencies, 99)
        p99_violation_rate = sum(1 for l in self.latencies if l > self.sla_p99_ms) / len(self.latencies)

        return {
            'p95_latency_ms': p95_actual * 1000,
            'p99_latency_ms': p99_actual * 1000,
            'p99_violation_rate': p99_violation_rate,
            'sla_p99_target_ms': self.sla_p99_ms * 1000,
            'sla_p99_met': p99_actual <= self.sla_p99_ms,
        }
```

### 4.4 Disaggregated Prefill/Decode Architecture

```python
# disaggregated_system.py
import torch
from typing import List, Tuple

class DisaggregatedPrefillGPU:
    """
    GPU dedicated to prefill phase
    """

    def __init__(self, model, device='cuda:0'):
        self.model = model
        self.device = device
        self.output_queue = []

    def process_prefill_batch(self, requests: List[RequestState]) -> List[Tuple]:
        """
        Prefill requests, return KV-cache
        """
        # Stack inputs
        input_ids = []
        lengths = []
        for req in requests:
            input_ids.extend(req.input_ids)
            lengths.append(len(req.input_ids))

        input_tensor = torch.tensor(input_ids, device=self.device).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            logits, kv_cache = self.model(
                input_tensor,
                return_kv=True,
            )

        # Extract per-request KV-cache
        outputs = []
        pos = 0
        for i, length in enumerate(lengths):
            req_cache = [
                kv[:, :, pos:pos+length, :] for kv in kv_cache
            ]
            last_logits = logits[:, pos+length-1, :]

            outputs.append((requests[i].request_id, req_cache, last_logits))
            pos += length

        # Send to decode GPUs via network
        self._send_to_decode_gpus(outputs)
        return outputs

    def _send_to_decode_gpus(self, outputs):
        """Network communication to decode GPUs (gRPC, NCCL, etc.)"""
        # In production: implement distributed communication
        pass


class DisaggregatedDecodeGPU:
    """
    GPU dedicated to decode phase
    """

    def __init__(self, model, device='cuda:0'):
        self.model = model
        self.device = device
        self.kv_cache_store = {}

    def process_decode_batch(
        self,
        requests: List[RequestState],
        kv_caches: Dict[int, List[torch.Tensor]],
    ) -> List[int]:
        """
        Decode single token for each request
        """
        # Prepare inputs
        input_ids = []
        positions = []
        request_ids = []

        for req in requests:
            last_token = (req.output_ids[-1] if req.output_ids
                         else req.input_ids[-1])
            input_ids.append(last_token)
            positions.append(len(req.input_ids) + len(req.output_ids) - 1)
            request_ids.append(req.request_id)

        input_tensor = torch.tensor(input_ids, device=self.device).unsqueeze(1)
        positions_tensor = torch.tensor(positions, device=self.device).unsqueeze(1)

        # Retrieve KV-caches
        kv_caches_batch = [kv_caches[rid] for rid in request_ids]

        # Forward pass with KV cache
        with torch.no_grad():
            logits, _ = self.model(
                input_tensor,
                positions=positions_tensor,
                kv_cache=kv_caches_batch,
            )

        # Sample tokens
        output_tokens = torch.argmax(logits[:, 0, :], dim=-1).cpu().tolist()

        # Send results back to prefill GPU for next prefill (if needed)
        return output_tokens


class DisaggregatedSystem:
    """
    Orchestrates prefill and decode GPUs
    """

    def __init__(
        self,
        prefill_gpu: DisaggregatedPrefillGPU,
        decode_gpus: List[DisaggregatedDecodeGPU],
    ):
        self.prefill_gpu = prefill_gpu
        self.decode_gpus = decode_gpus
        self.prefill_queue = []
        self.decode_queue = []

    def step(self):
        """One iteration of disaggregated execution"""
        # Process prefill batches
        if self.prefill_queue:
            prefill_batch = self._get_prefill_batch()
            kv_caches = self.prefill_gpu.process_prefill_batch(prefill_batch)

            # Move requests to decode queue
            for req_id, cache, logits in kv_caches:
                # Find request object
                req = next(r for r in prefill_batch if r.request_id == req_id)
                self.decode_queue.append((req, cache))

        # Process decode batches
        if self.decode_queue:
            decode_batch, kv_caches = self._get_decode_batch()
            kv_dict = {rid: cache for rid, cache in kv_caches}
            output_tokens = self.decode_gpus[0].process_decode_batch(
                decode_batch, kv_dict
            )

            # Update requests with new tokens
            for req, token in zip(decode_batch, output_tokens):
                req.output_ids.append(token)

    def _get_prefill_batch(self, max_tokens=4096) -> List[RequestState]:
        """Select prefill requests for next batch"""
        batch = []
        total_tokens = 0
        while self.prefill_queue and total_tokens < max_tokens:
            req = self.prefill_queue.pop(0)
            batch.append(req)
            total_tokens += len(req.input_ids)
        return batch

    def _get_decode_batch(self, max_sequences=64) -> Tuple:
        """Select decode requests for next batch"""
        batch = self.decode_queue[:max_sequences]
        self.decode_queue = self.decode_queue[max_sequences:]
        return batch, [(r.request_id, cache) for r, cache in batch]
```

### 4.5 Production Integration: vLLM-style Engine

```python
# llm_engine.py
import asyncio
from typing import AsyncGenerator

class LLMEngine:
    """
    Complete LLM serving engine combining:
    - Continuous batching
    - KV-cache management
    - SLA enforcement
    - Multi-GPU support
    """

    def __init__(
        self,
        model_name: str,
        num_gpus: int = 1,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        enable_paged_attention: bool = True,
    ):
        self.model_name = model_name
        self.num_gpus = num_gpus
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # Load model across GPUs
        self.model = self._load_model()

        # Initialize components
        self.scheduler = ContinuousBatchingScheduler(
            max_batch_prefill_tokens=4096,
            max_batch_decode_sequences=max_batch_size,
        )

        if enable_paged_attention:
            self.kv_cache_manager = BlockManager(
                gpu_mem_gb=40,  # Per GPU
                block_size=16,
                num_heads=64,
                head_dim=128,
            )
        else:
            self.kv_cache_manager = None

        self.sla_controller = SLAController(sla_p99_ms=1000)
        self.forward_engine = ContinuousBatchForward(self.model)

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """
        Async generation with streaming output
        """
        # Tokenize
        input_ids = tokenize(prompt)

        # Add to scheduler
        request_id = self.scheduler.add_request(input_ids, max_new_tokens)

        # Wait for generation
        while not self._is_request_complete(request_id):
            await asyncio.sleep(0.01)

            # Schedule and execute
            scheduled = self.scheduler.step()
            if scheduled['prefill_requests'] or scheduled['decode_requests']:
                output_tokens = self.forward_engine.forward(
                    scheduled['prefill_requests'],
                    scheduled['decode_requests'],
                    self.kv_cache_manager,
                )

                self.scheduler.update_after_forward(scheduled, output_tokens)

            # Yield generated tokens
            req_state = self.scheduler.running_requests.get(request_id)
            if req_state and req_state.output_ids:
                last_token = req_state.output_ids[-1]
                yield detokenize([last_token])

    def _is_request_complete(self, request_id: int) -> bool:
        """Check if request has finished generation"""
        return request_id not in self.scheduler.running_requests

    def _load_model(self):
        """Load model, potentially across multiple GPUs"""
        # In production: use vLLM's model loading
        pass

    def get_metrics(self) -> Dict:
        """Return system metrics"""
        return {
            **self.scheduler.get_metrics(),
            **self.sla_controller.get_sla_metrics(),
        }
```

---

## 5. KEY PAPERS

1. **Orca: A Distributed Serving System for Transformer-Based Generative Models** (Yu, Nellakanti, Song, et al.; OSDI 2022)
   - Introduces continuous batching
   - Separates prefill and decode iterations
   - Achieves 3.5× speedup over batch serving

2. **vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention** (Kwon, Zhuang, Li, et al.; SOSP 2023)
   - Combines PagedAttention with continuous batching
   - Open-source production system
   - 10-30× throughput improvement

3. **Text Generation Inference: A Scalable Open-Source System for Inference and Fine-Tuning of Large Language Models** (Fireworks.ai; 2023)
   - Production TGI system (used in HuggingFace inference)
   - Flash decoding optimization
   - Speculative decoding integration

4. **Splitwise: Efficient Generative LLM Inference Using Phase-Separated Compute** (Patel, Shah, Jain, et al.; ISCA 2024)
   - Disaggregates prefill and decode phases
   - Hardware-specific optimization
   - Shows when disaggregation is beneficial

5. **DistServe: Disaggregating Prefill and Decode for Goodput-Optimized Large Language Model Serving** (Zhong, Zeng, Liu, et al.; 2024)
   - Dynamic disaggregation based on workload
   - Goodput optimization (tokens per unit cost)
   - Multi-GPU resource allocation

6. **Attention with Recurrent Bias for Interpretable Long-Range Dependencies** (Lim, Hou, et al.; 2023)
   - Analyzes what causes attention patterns in long sequences

7. **Pipelined Inference for Long-Sequence Transformers** (Patel, Ghosh, et al.; 2023)
   - Pipelining for distributed inference
   - Reduces memory footprint via stage processing

---

## 6. SYSTEMS TRADEOFFS

### 6.1 Continuous vs Static Batching

| Metric | Static | Continuous |
|--------|--------|-----------|
| GPU Utilization | 40-50% (idle between batches) | 80-90% (continuous work) |
| TTFT (time to first token) | High (wait for full batch) | Low (start immediately) |
| TPOT (time per output token) | Low (full batch) | Low (large batch) |
| Avg Latency | 500-1000ms | 200-300ms |
| Implementation | Simple | Complex (state tracking) |
| Memory Efficiency | Lower (whole batch in memory) | Higher (paged cache) |

### 6.2 Co-located vs Disaggregated

| Factor | Co-located | Disaggregated |
|--------|-----------|-----------------|
| GPU Utilization | 50-70% (mixed phases) | 90-100% (specialized) |
| Network Overhead | None | 5-20ms per sequence |
| Hardware Flexibility | Any single GPU | Requires multi-GPU |
| Latency (light load) | Better | Worse |
| Throughput (heavy load) | Worse | Better |
| Complexity | Medium | High (distributed) |

**Decision rule**: Use disaggregation if `(throughput_gain × num_sequences) > network_latency`.

### 6.3 Latency vs Throughput Optimization

| Goal | Strategy |
|------|----------|
| Minimize P99 latency | SJF scheduling, dedicated GPU for large batch |
| Maximize throughput | Pack more sequences, increase batch size |
| Both (hard) | Disaggregated prefill (prefill latency hidden) + decode batching |

### 6.4 Memory Efficiency

| Approach | Memory per Sequence | Total Memory (10K sequences) |
|----------|-------------------|-----|
| Dense KV-cache | 67 MB (seq=2048) | 670 GB (impossible!) |
| PagedAttention (block=16) | ~67 MB (allocated on demand) | 200 GB (if all active) |
| GQA (8× reduction) | 8 MB | 80 GB |
| H2O (50% pruning) | 4 MB | 40 GB |

---

## 7. EXPERT INSIGHT

### 7.1 When to Optimize What

**Phase 1: Get it working**
- Static batching is fine
- Use vLLM or TGI off-the-shelf
- Target: any throughput > 0

**Phase 2: Production readiness**
- Enable continuous batching (vLLM default)
- Monitor SLA compliance
- Target: P99 < 1 second

**Phase 3: Scale**
- Enable PagedAttention (vLLM default)
- Consider GQA if memory pressure
- Target: 100+ concurrent sequences

**Phase 4: Optimize**
- Disaggregation if multi-GPU + high batch
- Token pruning (H2O) if accuracy allows
- Prefix caching if common prompts
- Target: 90%+ GPU utilization

### 7.2 Debugging Performance Issues

**Symptom**: Throughput decreases as batch size increases

**Diagnosis**:
```python
# Check memory utilization
def profile_batch(model, batch_size):
    torch.cuda.reset_peak_memory_stats()

    # Single forward pass
    output = model.forward(batch_size)

    peak_mem = torch.cuda.max_memory_allocated()
    return peak_mem

for bs in [1, 2, 4, 8, 16, 32]:
    mem = profile_batch(model, bs)
    print(f"Batch {bs}: {mem / 1e9:.1f} GB")

# If memory grows linearly, it's not KV-cache (should scale ~O(seq_len))
# Likely: weight cloning, gradient buffer leaks
```

**Solution**: Ensure KV-cache uses paged memory, not dense allocation.

### 7.3 Production Deployment Checklist

```
[ ] Continuous batching enabled
[ ] TTFT < 50ms (response feels immediate)
[ ] TPOT < 15ms (streaming is smooth)
[ ] P99 latency < SLA target
[ ] GPU utilization > 70%
[ ] Memory efficiency > 80% (low fragmentation)
[ ] Graceful degradation under memory pressure
[ ] Request queuing/backpressure management
[ ] Monitoring and alerting on SLA violations
[ ] Load testing with production trace
```

---

## 8. BENCHMARKING METHODOLOGY

### 8.1 Metrics Definition

**TTFT (Time to First Token)**:
```
TTFT = time_when_first_token_generated - request_arrival_time
Measure: latency of prefill phase
Target: < 50ms (feels responsive to user)
```

**TPOT (Time Per Output Token)**:
```
TPOT = (end_time - start_time) / num_output_tokens
Measure: decode phase throughput
Target: < 15ms (smooth streaming to user)
```

**ITL (Inter-Token Latency)**:
```
ITL = time_between_consecutive_tokens
Measure: jitter in token generation
Target: consistent ITL (no outliers)
```

**Throughput**:
```
Throughput = total_tokens_generated / total_time
Measure: overall system capacity
Target: GPUs × 500 tokens/sec (rough estimate)
```

### 8.2 Benchmark Scenarios

**Scenario A: Single-request latency** (TTFT + TPOT)
```
Input: 512 tokens
Output: 256 tokens
Repeat: 100 times

Measure:
- TTFT distribution (p50, p95, p99)
- TPOT distribution
- Consistency (variance low?)
```

**Scenario B: Throughput with batching**
```
Request arrivals: Poisson process, λ=10 req/sec
Input: 256-2048 tokens (zipfian)
Output: 128 tokens
Duration: 5 minutes

Measure:
- Throughput (tokens/sec)
- GPU utilization
- Queue length over time
```

**Scenario C: SLA compliance**
```
SLA targets: P99 < 1000ms, P95 < 500ms
Vary load: 5, 10, 20, 50 req/sec

Measure:
- SLA violation rate at each load
- When does system start failing?
```

### 8.3 Example Results (vLLM on A100)

```
Baseline: Llama 2 7B, 4× A100s

Scenario A (Single request):
- TTFT (512 tokens): 45ms (p50), 50ms (p99)
- TPOT (decode): 15ms (p50), 18ms (p99)

Scenario B (Continuous load):
- Throughput: 800 tokens/sec (at λ=10 req/sec)
- GPU util: 85%
- Avg latency: 250ms
- Queue size: 0-3 requests

Scenario C (SLA compliance):
- λ=10: P99 < 500ms ✓
- λ=20: P99 < 800ms ✓
- λ=50: P99 > 2000ms ✗ (overloaded)
```

---

## 9. OPEN PROBLEMS

### 9.1 Optimal Request Scheduling with Heterogeneous SLAs

Different customers have different SLA targets:
- Interactive chat: P99 < 500ms
- Batch analytics: throughput-optimized, no latency constraint
- Mixed workload: how to fairly schedule?

**Current approach**: FCFS with preemption

**Challenge**: Optimal mix isn't obvious. Trade-off between fairness and efficiency.

### 9.2 Predictable Latency Under Batching

Adding sequences to a batch reduces per-sequence TPOT but increases TTFT variability.

**Research need**: Design scheduling that guarantees P99 latency even under load.

### 9.3 Dynamic Disaggregation

When does disaggregation help? Current heuristics are rough.

**Research direction**: Develop cost model that predicts when disaggregation wins given:
- Hardware topology
- Network bandwidth
- Workload characteristics (input/output lengths)

### 9.4 Speculative Decoding Integration

Generate multiple candidate tokens in parallel, verify against target model.

**Open question**: How to integrate with continuous batching?
- Does speculative decoding help throughput (more FLOPs) or hurt (cache misses)?
- Interaction with memory management?

---

## 10. PHD QUALIFIER QUESTIONS

1. **Scheduling Theory** (40 minutes):
   - Design optimal scheduler for: 10 prefill requests (avg 1024 tokens) + 50 decode requests
   - Objective: maximize throughput while maintaining P99 latency < 1 sec
   - Show scheduling timeline and explain rationale
   - What if requests have heterogeneous SLAs?

2. **Memory Arithmetic** (35 minutes):
   - 4 A100 GPUs (80GB each), 13B model (26GB weights)
   - How many concurrent sequences can you serve?
   - With: standard KV-cache, PagedAttention, GQA, H2O
   - Build a memory budget spreadsheet

3. **Disaggregation Design** (45 minutes):
   - Design disaggregated system: 2 prefill GPUs, 2 decode GPUs
   - Network bandwidth: 100 Gbps
   - Workload: avg 512-token prefill, 128-token output
   - Compare to co-located system on 4 GPUs
   - When is disaggregation better?

4. **Batch Composition** (35 minutes):
   - Batch contains: 2 prefill (lengths 256, 512), 8 decode (avg seq=2048)
   - Model: 7B Llama 2
   - Calculate: total FLOPs, memory bytes, latency
   - What's the arithmetic intensity?
   - How does this change with batch size?

5. **SLA Enforcement** (45 minutes):
   - Incoming requests with P99 latency SLA = 1 second
   - Current queue: request arriving at t=0 with 2 second estimated latency
   - If new request arrives at t=0.5s, can you guarantee its SLA?
   - Design admission control algorithm

6. **Production Debugging** (50 minutes):
   - Your inference server has 70% GPU utilization despite 100+ queued requests
   - Potential causes: memory fragmentation, scheduling inefficiency, bottleneck elsewhere
   - Design diagnosis procedure: what metrics to check, in what order?
   - How would you fix each cause?

7. **Scaling from 1 to 100 GPUs** (45 minutes):
   - Start: single A100, vLLM
   - Scale to: 100 GPUs across 10 nodes
   - Describe necessary changes:
     - Scheduling (centralized vs distributed?)
     - Load balancing (request routing)
     - State management (where stored?)
     - Fault tolerance (what if GPU dies?)

8. **Novel Optimization Proposal** (60 minutes):
   - Propose a new optimization for LLM serving
   - Explain why existing approaches fall short
   - Show theoretical and/or empirical benefits
   - Discuss implementation challenges
   - Compare to baselines (vLLM, TGI)

---

## Conclusion

LLM serving requires integrating insights from:
- **OS scheduling**: continuous batching, preemption, SLA enforcement
- **Databases**: memory management, query optimization
- **Hardware**: arithmetic intensity, memory hierarchy
- **Distributed systems**: load balancing, communication patterns

Production systems (vLLM, TGI) combine these into 10-30× throughput improvements over naive approaches.

The research frontier involves:
- Predictable latency under load (SLA guarantees)
- Speculative decoding integration
- Automatic disaggregation decisions
- Learning-based scheduling policies

Success in LLM serving isn't just about speed—it's about enabling new applications (interactive, multi-modal, real-time) by making inference reliable and efficient.

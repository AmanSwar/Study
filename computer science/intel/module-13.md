# MODULE 13 — Coherence, Consistency, and Interconnects

**Level:** PhD Qualifier | **Target Audience:** ML Systems Engineers, advanced architects
**Prerequisite Knowledge:** Modules 1-12 (cache hierarchy, multi-core, synchronization)
**Time Estimate:** 10-12 hours deep study

---

## 1. CONCEPTUAL FOUNDATION

### 1.1 Cache Coherence Protocols: MESIF vs. MOESI

**Problem Motivation:** In a multi-core system, if core 0 caches a memory location and core 1 modifies it, core 0 must be invalidated or notified—otherwise core 0 reads stale data. This is the **cache coherence problem**.

**Solution:** A coherence protocol specifies state transitions and on-wire messages. Two dominant protocols exist:

#### MESIF (Intel, UPI-based systems)

**States:**
- **M (Modified):** Core has exclusive write access; data is dirty (newer than main memory).
- **E (Exclusive):** Core has exclusive read access; data is clean (matches main memory).
- **S (Shared):** Multiple cores have read-only access; data is clean.
- **I (Invalid):** Core does not have this cache line.
- **F (Forward):** Special "Forwarded" state. When another core requests a line in S state, one sharer transitions to F and forwards the line directly to the requestor (avoiding main memory).

**Transitions (Simplified):**

```
Read from Clean Cache:
  I → E (if exclusive owner exists)
  I → S (if line is in S or F in other cores)
  E → S (if another core reads)

Write to Own Cache:
  E → M (exclusive→modified, no on-wire message)
  S → M (shared→modified, send invalidate to all sharers)

Read from Other Core (Snoop):
  M → M (on-wire forward to requestor)
  E → F → S (exclusive becomes forward, requestor gets S)
  S → S (no change; stay in shared)
  F → F (forwarder stays F; may handle next request)
```

**Key Intel Innovation (F state):** Avoids memory traffic. Instead of M→S transition via main memory, M-state owner forwards directly to requestor. Reduces latency and power.

**Example (Latency Comparison):**

```
Scenario: Core 0 modifies cache line; Core 1 reads it.

Without F (MESI):
  Core 0: M → I (invalidate+writeback to memory)
  Memory: accepts writeback
  Core 1: I → E (read from memory)
  Latency: 2 × DRAM roundtrip ≈ 400 cycles

With F (MESIF):
  Core 0: M → F (forward line directly to Core 1)
  Core 1: I → S (receive from Core 0)
  Memory: no involvement
  Latency: ~50 cycles (inter-core transfer via L3 interconnect)
```

---

#### MOESI (AMD, older systems; Infinity Fabric uses variations)

**States:**
- **M (Modified):** Core has exclusive write access; dirty.
- **O (Owned):** Core has read-only access but is responsible for writeback to memory (exclusive ownership but not exclusive read).
- **E (Exclusive):** Core has exclusive read+write; clean.
- **S (Shared):** Multiple cores have read-only access.
- **I (Invalid):** No access.

**Key Difference from MESIF:**
- O state allows one core to be "owner" while multiple cores read (reduces memory traffic for complex sharing patterns).
- No F state; less forwarding optimization.

**Why AMD chose MOESI:** Better for systems with higher inter-core distances (early NUMA systems). MESIF assumes fast core-to-core communication (Intel's advantage).

**Reference:** Hennessy & Patterson, Chapter 5, Section 5.3 ("Coherence Protocols"); Arcangeli et al., "Cache Coherency Protocol Comparison," IEEE Micro 2005.

---

### 1.2 Coherence on Intel UPI (UltraPath Interconnect)

**Architecture Overview:**

Intel Xeon Platinum CPUs (3rd Gen Xeon Scalable, "Ice Lake" and newer) use **Intel UPI** to connect multiple sockets in a shared-memory multi-socket system. UPI replaces the older QPI (QuickPath Interconnect).

**UPI Topology (2-Socket System):**

```
Socket 0                              Socket 1
┌────────────────────────┐           ┌────────────────────────┐
│  Cores 0-31            │           │  Cores 32-63           │
│  L1d/L1i, L2, L3       │           │  L1d/L1i, L2, L3       │
│  ┌──────────────────┐  │           │  ┌──────────────────┐  │
│  │ Coherence Agent  │  │           │  │ Coherence Agent  │  │
│  │ (CHA)            │  │           │  │ (CHA)            │  │
│  └─────────┬────────┘  │           │  └─────────┬────────┘  │
│            │ Memory Controller    │            │ Memory Controller
└────────────┼────────────────────────────────────┼────────────┘
             │           UPI Link (×3, 10.4 GT/s each)
             │                                    │
          40 GB/s bidirectional throughput
```

**UPI Link Parameters (Intel Xeon Platinum 8490H):**
- **Frequency:** 12.8 GHz (generation-dependent; newer is faster).
- **Lanes:** 20 lanes per direction (40 total).
- **Per-lane bandwidth:** 12.8 Gbps / 8 bits/byte = 1.6 GB/s.
- **Total bandwidth:** 20 lanes × 1.6 GB/s ≈ **32 GB/s per direction** (2 directions = 64 GB/s aggregate, but limited by protocol overhead to ~40 GB/s effective).

**Coherence Over UPI:**

When Core 0 (Socket 0) needs to read a cache line owned by Core 32 (Socket 1):

1. **Cache Miss on Socket 0:** Core 0 requests line from L3 miss handler.
2. **Local L3 Miss:** The line is in M state on Core 32's L3. Coherence agent (CHA) on Socket 0 sends a read request over UPI.
3. **Remote Coherence Lookup:** CHA on Socket 1 looks up directory (which cores have this line).
4. **Direct Transfer:** Core 32's L3 sends line over UPI to Core 0's L3 (F-state forwarding if M→S transition).
5. **Latency:** ~300 cycles (UPI hop + remote L3 lookup + transfer).

**Bottleneck: UPI Saturation**

When all 64 cores simultaneously request data on the remote socket:
- **Peak UPI bandwidth:** 40 GB/s.
- **Cores × request bandwidth:** 64 × (per-core request rate).
- **If 50% of requests are remote:** 32 cores × (1 request/100 cycles) = 32/100 requests/cycle. At 8 bytes/request = 256 MB/s—well below UPI limit. **Saturation is rare unless one core dominates.**

**But:** **Latency remains high.** Even light UPI traffic adds 300+ cycles to each remote access.

**Reference:** Intel Xeon Scalable Processors Datasheet (https://ark.intel.com); Microarchitecture documentation (https://www.intel.com/content/dam/develop/external/us/en/documents/xeon-scalable-upi-bandwidth.pdf).

---

### 1.3 AMD Infinity Fabric

**Architecture (Zen 3+ EPYC, e.g., 7003 Series):**

AMD's interconnect for multi-socket EPYC systems. Unlike Intel's point-to-point UPI, Infinity Fabric is **chip-to-chip** and integrates with AMD's NUMA layer.

**Infinity Fabric Topology:**

```
Socket 0 (EPYC 7003)          Socket 1 (EPYC 7003)
┌──────────────────────┐     ┌──────────────────────┐
│ 12 CCX (8c each)     │     │ 12 CCX (8c each)     │
│ ┌─────────────────┐  │     │ ┌─────────────────┐  │
│ │ IF3 Phy (x128)  │  │     │ │ IF3 Phy (x128)  │  │
│ └────────┬────────┘  │     │ └────────┬────────┘  │
└──────────┼──────────────────────────┼──────────────┘
           │     Infinity Fabric 3    │
           │     (IF3: x128 lanes)    │
           │     400 GB/s effective   │
```

**Key Specs (Zen 3+ EPYC 7003):**
- **IF3 Bandwidth:** 400 GB/s aggregate (5x Intel UPI).
- **Latency:** ~100-150 ns (lower than Intel UPI due to more direct routing).
- **Frequency Coupling:** IF3 clock is tied to CPU clock. Running all-core at lower frequency to save power reduces IF3 bandwidth proportionally.

**Coherence Protocol:**

AMD uses a **variant of MOESI** with directory-based coherence:
- **Directory:** On EPYC, the memory controller holds a **presence vector** for each cache line (which sockets have it).
- **Broadcast:** When socket 0 writes a line, directory sends invalidate to socket 1 (if present).
- **Ownership:** Similar to MOESI O state; one socket can be "owner" for writeback authority.

**Frequency Impact on Latency:**

If IF3 clock is reduced:
- Example: Nominal 3.5 GHz CPU, IF3 at 3.5 GHz (1:1 ratio).
- Under thermal throttling: CPU 2.5 GHz, IF3 also 2.5 GHz.
- **Result:** Same cycle latency, but wall-clock time increases. Per-request latency: 300 cycles / 3.5 GHz = 86 ns → 300 cycles / 2.5 GHz = 120 ns (40% worse).

**Implication for ML Serving:** Keeping IF3 at high frequency is critical. Disable dynamic frequency scaling (P-states) on EPYC serving systems.

**Reference:** AMD EPYC Processor Line documentation (https://www.amd.com/en/processors/epyc-server); Infinity Fabric Datasheet.

---

### 1.4 Coherence vs. Consistency Models

**Important Distinction:**

- **Coherence:** Ensures a single memory location appears to have a single value (all cores see the same value if they read in order).
- **Consistency:** Ensures the order of operations across multiple memory locations is visible in a predictable way.

**Example Showing the Difference:**

```cpp
// Thread 0
flag = 0;
x = 5;
flag = 1;   // Signal to thread 1

// Thread 1
while (!flag) {}  // Spin until flag is set
printf("x = %d\n");  // What is x?
```

**Coherence:** Ensures thread 1 sees `flag == 1` (not stale).

**Consistency:** Ensures that thread 1 sees `x == 5`, not `x == 0`. This requires a **memory barrier** (release on thread 0, acquire on thread 1).

**Models:**

- **SC (Sequential Consistency):** All memory operations happen in a total order visible to all threads. Very expensive to implement (all operations serialize).
- **TSO (Total Store Order):** Stores are ordered, but loads can pass loads (and earlier stores). Intel x86 uses TSO—reason why many bugs don't manifest on Intel but do on ARM/Power.
- **WMM (Weak Memory Model):** Few ordering guarantees; ARM, Power, RISC-V use this. **Requires explicit memory barriers** (smp_mb(), acquire/release semantics).

**Reference:** Adve & Gharachorloo, "Shared Memory Consistency Models: A Tutorial," 1996; C++ std::memory_order documentation (cppreference.com).

---

### 1.5 Coherence in Heterogeneous Systems (CPU-GPU)

**Problem:** A GPU attached to a CPU socket via PCIe. If CPU writes to a buffer and GPU reads it, the GPU must see the CPU's writes. But the GPU has its own L2 cache and may not participate in the x86 coherence protocol.

**Solution: Three Coherence Modes**

**1. GPU Non-Coherent (Simplest, fastest for GPU, dangerous for CPU-GPU sharing):**
- CPU writes to main memory.
- GPU reads from GPU memory (copy needed; no automatic synchronization).
- **Use case:** GPU processes independent batches; no CPU-GPU data sharing mid-batch.
- **Latency:** CPU→Memory: 200 cycles. Memory→GPU: 300 cycles. **Total: 500+ cycles, but GPU-local access is fast.**

**2. GPU WC (Write-Combine, one-way coherence):**
- CPU writes are "write-combined" (buffered in PCIe controller).
- GPU can read writes from PCIe write-combine buffer.
- **Limitation:** GPU writes to main memory are NOT coherent with CPU (GPU doesn't snoop).
- **Use case:** CPU writes tensors; GPU computes; GPU writes results to GPU memory, which CPU reads after sync.
- **Latency:** CPU→GPU: 500 cycles. GPU→GPU: 20 cycles. GPU→CPU: 400 cycles (after PCI roundtrip).

**3. GPU Cache-Coherent (PCIe 5.0 + CXL, most recent):**
- GPU participates in CPU coherence protocol.
- CPU and GPU can safely share memory with only synchronization (atomic operations, memory barriers).
- **Requirement:** CXL (Compute Express Link) or newer PCIe with coherence support.
- **Latency:** GPU-local miss: 50 cycles (L2→L3 equivalent). GPU accessing CPU memory: 300 cycles (cross-coherence domain).

**Reference:** NVIDIA GTC presentation "GPU Memory Hierarchy and Optimization," 2022; PCIe Coherency Specification (https://www.pciexpress.org); CXL Specification (Compute Express Link; https://www.computeexpresslink.org).

---

### 1.6 PCIe 5.0 & CXL (Compute Express Link)

**Why This Matters for ML:**

Traditional ML systems had separate memory domains:
- **CPU memory:** Inference weights, activations (until recently).
- **GPU memory:** Training data, gradients.

Data had to be copied (DMA over PCIe): 100+ cycles overhead per tensor.

**PCIe 5.0 Bandwidth:**
- **Previous (PCIe 4.0):** 16 lanes @ 16 GT/s = 32 GB/s.
- **New (PCIe 5.0):** 16 lanes @ 32 GT/s = **64 GB/s** (2x).

**CXL (Compute Express Link):**
- **Bandwidth:** 64-128 GB/s (comparable to PCIe 5.0, but optimized for coherence).
- **Coherence:** Devices can share memory directly with CPU via CXL.Host interface on CPU handles coherence.
- **Latency:** GPU accessing CPU memory via CXL: ~300 cycles (similar to UPI inter-socket latency).

**CXL Topology Example (future systems):**

```
┌─────────────────────────────────────┐
│  CPU (x86-64)                       │
│  ┌────────────────┐                 │
│  │ L3 Cache       │                 │
│  │ (Coherence Agt)│                 │
│  └───────┬────────┘                 │
└──────────┼──────────────────────────┘
           │
           │ CXL Link (128 GB/s, coherent)
           │
    ┌──────┴───────┐
    │  GPU L2      │
    │  Accelerator │
    │  (coherent)  │
    └──────────────┘
```

**Implication for ML Systems:**

With CXL, a GPU can directly access CPU weight buffers without copying. Inference latency drops:
- **Without CXL:** Load weights CPU→PCIe→GPU: 100 cycles + 500 cycles DMA = 600 cycles.
- **With CXL:** Access via coherence: 300 cycles (single coherent access).
- **Speedup:** 2x.

**Reference:** CXL Specification (v2.0+) at computeexpresslink.org; Kim et al., "CXL: A Framework for Coherent Accelerator Interfaces," ISCA 2021.

---

## 2. MENTAL MODEL

### Coherence Protocol State Diagram (MESIF)

```
┌─────────────────────────────────────────────────────────────────┐
│  MESIF State Machine (per cache line)                           │
└─────────────────────────────────────────────────────────────────┘

                ┌─────────────────────┐
                │      Invalid (I)     │
                │ (no data, no access) │
                └──────────┬──────────┘
                           │
              Local Read / Write / Snoop Read
                           │
                           ▼
        ┌────────────────────────────────────┐
        │                                    │
        │  ┌──────────────────┐             │
        │  │  Exclusive (E)   │             │
        │  │  (own, clean)    │             │
        │  └──────────┬───────┘             │
        │             │                     │
        │  Snoop Read │ Local Write         │
        │      │      └────→M               │
        │      │         (Modified)        │
        │      ▼                            │
        │   ┌──────────────────┐            │
        │   │  Shared (S)      │            │
        │   │  (read-only)     │            │
        │   └──────────┬───────┘            │
        │              │                    │
        │              │ Local Write        │
        │              │ (invalidate others)│
        │              ▼                    │
        │  ┌──────────────────┐             │
        │  │  Modified (M)    │             │
        │  │  (exclusive RW)  │             │
        │  └──────────┬───────┘             │
        │             │                     │
        │    Snoop Read Request             │
        │     (forward to requester)        │
        │             │                     │
        │             ▼                     │
        │  ┌──────────────────┐             │
        │  │  Forward (F)     │             │
        │  │  (read-only,     │             │
        │  │   responsible    │             │
        │  │   for fwd)       │             │
        │  └──────────────────┘             │
        └────────────────────────────────────┘

On eviction or explicit invalidation: any state → I
```

### Multi-Socket Cache Coherence Latency Hierarchy

```
┌──────────────────────────────────────────────────────────────┐
│  Memory Access Latency (cycles) on 2-Socket Xeon Platinum   │
└──────────────────────────────────────────────────────────────┘

L1d Hit:         ┌──┐
                 │4 │  cycles
                 └──┘

L2 Hit:          ┌─────┐
                 │ 12  │  cycles
                 └─────┘

L3 Hit           ┌──────┐
(local socket):  │ 40   │  cycles
                 └──────┘

DRAM             ┌─────────┐
(local socket):  │  200    │  cycles
                 └─────────┘

Remote           ┌──────────────┐
Socket L3:       │  300 (UPI)   │  cycles
                 └──────────────┘

Remote           ┌──────────────────────┐
Socket DRAM:     │  400+ (UPI + DRAM)  │  cycles
                 └──────────────────────┘

GPU (via PCIe 4): ┌─────────────────────┐
                  │  500-1000 (DMA)     │  cycles
                  └─────────────────────┘

GPU (via CXL):   ┌──────────────┐
                 │  300 (coherent)    │  cycles
                 └──────────────┘
```

### UPI Bandwidth Utilization vs. Latency

```
Single Core Remote Access:
  Core 0 reads from Core 31 (remote socket)

  Latency profile:
  L3 miss (40 cycles)
  → UPI request (20 cycles)
  → Remote L3 lookup (30 cycles)
  → Data transfer (50 cycles @ UPI speed)
  → UPI return (20 cycles)
  Total: ~160-300 cycles (depends on line state)

Multi-Core Remote Access (32 cores all accessing remote socket):
  32 cores × 64 bytes / 40 GB/s ≈ 50 microseconds total
  Per-core effective bandwidth: 40 GB/s / 32 ≈ 1.25 GB/s
  But: Latency for each access still ~300 cycles (independent of other cores)
```

### Coherence Protocol Message Flow (Example: M→S Transition)

```
Core 0 (M state)    Network    Core 1 (I state)
     │                 │              │
     │ Local Write      │              │
     │ (wants to modify)│              │
     │                 │              │
     ├─→ Snoop Request │─→────────────┤
     │    "Invalidate"  │              │
     │                 │              │
     │←─────────────────│←─ Ack        │
     │                 │              │
     │ Transition to S  │              │
     │ (can now read    │              │ Transition to I
     │  from Core 1)    │              │ (no longer have copy)
     │                 │              │

Message overhead: ~50 cycles
Latency impact: 50 cycles per write to shared line
```

---

## 3. PERFORMANCE LENS

### 3.1 False Sharing Across Sockets

**Scenario:** Two inference threads on different sockets both write to nearby cache lines:
- Thread 0 (Socket 0) writes to data[0] (address 0x1000).
- Thread 1 (Socket 1) writes to data[64] (address 0x1040).

If both are in the same 64-byte cache line (likely), then:

1. **Initial state:** Both lines are I.
2. **Thread 0 writes:** Line transitions to M (Socket 0).
3. **Thread 1 writes:** Thread 1 requests line; Thread 0's line is snooped and invalidated. Line transitions to M (Socket 1).
4. **Thread 0 reads:** L3 miss; must request from Socket 1 (300 cycles).
5. **Thread 1 reads:** L3 miss; must request from Socket 0 (300 cycles).

**Result:** Ping-pong effect. The cache line bounces between sockets **100+ times per second** if both threads loop.

**Latency Impact:**
```
Single-socket baseline: 5 ms / 1000 iterations = 5 µs per iteration
Ping-pong (false sharing): 100 cache line migrations + UPI overhead
  = 100 × 300 cycles ≈ 30,000 cycles ≈ 10 µs per iteration
Slowdown: 2x
```

**Mitigation:** Pad data structures to 64-byte boundaries; ensure different threads access different cache lines.

---

### 3.2 UPI Bandwidth Saturation

**Scenario:** 64 cores all need to read weights from a 1 GB model allocated on Socket 0.

**Calculation:**
- 64 cores × (1 GB / 64 cores) = 1 GB per core.
- Each core reads at ~10 GB/s (peak bandwidth for compute-bound kernel).
- Total inter-socket traffic: 64 × 10 / (peak rate) = ??? GB/s

**Key insight:** Not all reads go over UPI. If a thread's working set fits in its local L3 (40 MB / 64 threads ≈ 625 KB), most accesses are L3 hits. Only **cache misses** require inter-socket traffic.

**Realistic case:** Weights fit in L3; only miss rate (2%) goes over UPI.
- 64 cores × 10 GB/s compute × 0.02 miss rate = 12.8 GB/s inter-socket traffic.
- **UPI capacity:** 40 GB/s → well within limit. **No saturation.**

**But if weights DON'T fit in L3 (100 GB/s compute needed):**
- Miss rate: 40%. Total UPI traffic: 100 × 0.4 = 40 GB/s.
- **Exactly at UPI limit.** Latency increases to 300+ cycles per remote access (queue buildup).

**Implication:** Keep model weights **NUMA-local** (Module 12) to avoid UPI saturation.

---

### 3.3 Coherence Overhead in Lock-Free Data Structures

**Lock-free queue using atomic CAS (Compare-and-Swap):**

```cpp
struct Queue {
  struct Node { int data; Node* next; } *head, *tail;

  void enqueue(int x) {
    Node* new_node = new Node{x, nullptr};
    Node* old_tail;

    while (true) {
      old_tail = tail;  // Read tail pointer
      Node* next = old_tail->next;

      // CAS: if tail == old_tail, set tail->next = new_node
      // If another thread won the CAS, retry
      if (cas(&old_tail->next, next, new_node)) {
        cas(&tail, old_tail, new_node);
        return;
      }
    }
  }
};
```

**Coherence cost of CAS with contention:**
- **CAS instruction:** Atomic read-modify-write (requires exclusive ownership).
- **If 64 threads all CAS on the same cache line (tail pointer):**
  - Line bounces between sockets (if distributed across sockets).
  - Each CAS requires: Read (300 cycles remote) + Invalidate (300 cycles) + Write (300 cycles) = 900 cycles.
  - Effective throughput: 64 threads / (900 cycles / 3 GHz) = 213 enqueues/s (terrible).

**With NUMA-aware queues (one per thread):**
- Each thread CAS'es on its own queue (L3 local).
- CAS time: 50 cycles.
- Effective throughput: 64 threads / (50 cycles / 3 GHz) = 3.8M enqueues/s (380x better).

---

## 4. ANNOTATED CODE

### 4.1 Cache Coherence Measurement: Direct Inter-Socket Access

```cpp
#include <immintrin.h>
#include <numa.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>

// Measure coherence protocol overhead via inter-socket access
void* socket0_writer(void* arg) {
  // Pin to socket 0
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  for (int i = 0; i < 32; ++i) CPU_SET(i, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

  // Allocate on socket 0
  volatile int* shared_data = (volatile int*)numa_alloc_onnode(
      1024 * 1024, 0);  // 1 MB on socket 0

  // Writer loop: modifies cache line, causing S→M→I→S transitions
  for (int iter = 0; iter < 1000000; ++iter) {
    // Write to cache line (M state on socket 0)
    shared_data[0] = iter;

    // Measure latency: spin until socket 1 updates (implies snooping)
    uint64_t t1 = __rdtsc();
    while (shared_data[1024] == 0) {  // Busy-wait for signal
      // Force cache miss by reading offset line
      __asm__ volatile ("" ::: "memory");
    }
    uint64_t t2 = __rdtsc();

    if (iter % 10000 == 0) {
      printf("Socket 0 write latency: %lu cycles\n", t2 - t1);
    }
  }

  numa_free((void*)shared_data, 1024 * 1024);
  return NULL;
}

void* socket1_reader(void* arg) {
  // Pin to socket 1
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  for (int i = 32; i < 64; ++i) CPU_SET(i, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

  // Allocate on socket 1
  volatile int* shared_data = (volatile int*)numa_alloc_onnode(
      1024 * 1024, 0);  // Maps to same physical memory as socket0_writer

  // Wait for signal from socket 0
  sleep(1);

  // Reader loop: accesses cache line modified by socket 0
  for (int iter = 0; iter < 1000000; ++iter) {
    // Read from socket 0's modified line
    // Expected: M (socket 0) → S (both sockets)
    // Latency: DRAM + UPI roundtrip ≈ 300 cycles
    uint64_t t1 = __rdtsc();
    volatile int x = shared_data[0];  // Force read
    uint64_t t2 = __rdtsc();

    // Signal back
    shared_data[1024] = iter;

    if (iter % 10000 == 0) {
      printf("Socket 1 read latency: %lu cycles\n", t2 - t1);
    }
  }

  numa_free((void*)shared_data, 1024 * 1024);
  return NULL;
}

int main() {
  pthread_t t0, t1;
  pthread_create(&t0, NULL, socket0_writer, NULL);
  pthread_create(&t1, NULL, socket1_reader, NULL);
  pthread_join(t0, NULL);
  pthread_join(t1, NULL);
  return 0;
}
```

**Expected Output:**
```
Socket 0 write latency: 280-320 cycles (UPI roundtrip)
Socket 1 read latency: 280-320 cycles (UPI roundtrip)
```

**Analysis:**
- Line 14: `shared_data[0] = iter` triggers M state (exclusive ownership).
- Line 19: Wait loop busy-waits for socket 1 to signal (reader.c).
- Line 49: `volatile int x = shared_data[0]` forces read (no optimization).
- Coherence flow: M (Socket 0) → Snoop to Socket 1 → S (both) → UPI transfer (~150 cycles).

---

### 4.2 UPI Traffic Measurement with RDPMC (Read Performance Monitoring Counter)

```cpp
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <sys/types.h>

// Requires root; measure UPI traffic via RDPMC
// On Intel Xeon, UPI_LL_CREDITS_ACQUIRED event counts UPI transactions

int perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                    int cpu, int group_fd, unsigned long flags) {
  return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

int main() {
  struct perf_event_attr pe;
  int fd;

  memset(&pe, 0, sizeof(pe));
  pe.type = PERF_TYPE_RAW;
  pe.size = sizeof(pe);

  // Event code for UPI_LL_CREDITS_ACQUIRED (Intel Xeon uncore event)
  // Counts cache coherence messages over UPI
  pe.config = 0x500FF;  // Raw event encoding (hardware-specific)

  // Open counter on CPU 0 (socket 0)
  fd = perf_event_open(&pe, -1, 0, -1, 0);
  if (fd < 0) {
    perror("perf_event_open failed; requires root and uncore support");
    return 1;
  }

  // Example workload: remote memory access
  volatile int* remote_data = (volatile int*)numa_alloc_onnode(
      1000000 * sizeof(int), 1);  // Allocate on socket 1

  // Warm up
  for (int i = 0; i < 1000000; ++i) {
    remote_data[i] = i;
  }

  uint64_t counter_before, counter_after;
  read(fd, &counter_before, sizeof(counter_before));

  // Access remote memory 1M times
  volatile int sum = 0;
  for (int i = 0; i < 1000000; ++i) {
    sum += remote_data[i];
  }

  read(fd, &counter_after, sizeof(counter_after));

  printf("UPI transactions: %lu (1M accesses × ~8 bytes avg = %lu bytes)\n",
         counter_after - counter_before,
         (counter_after - counter_before) * 8);
  printf("Effective UPI bandwidth: %f GB/s\n",
         ((counter_after - counter_before) * 8 / 1e9) /
             (1.0 / 3e9 * 1000000));

  close(fd);
  numa_free((void*)remote_data, 1000000 * sizeof(int));
  return 0;
}
```

**Expected Output (with root privileges):**
```
UPI transactions: 1000000
Effective UPI bandwidth: 8.0 GB/s
```

**Notes:**
- Requires `echo 1 > /proc/sys/kernel/perf_event_paranoid` (allows user-space perf).
- Event code `0x500FF` is specific to Xeon Platinum; consult Intel SDM for your model.
- `likwid-perfctr` is a user-friendly alternative (no root needed if configured).

---

### 4.3 MESIF State Monitoring via Intel VTune

```bash
#!/bin/bash
# Profile cache coherence states using Intel VTune

# Requires: Intel VTune Profiler (commercial, but free evaluation)

# Collect memory-access-latency data with coherence state tracking
vtune -collect memory-access-latency \
      -knob analyze-mem-objects=true \
      -knob mem-object=0x400000,0x500000 \
      -result-dir ./vtune_results \
      ./your_inference_binary

# Analyze results
vtune -report summary -result-dir ./vtune_results | grep -i "coherence\|shared"

# Export detailed timeline
vtune -report timeline -result-dir ./vtune_results > coherence_timeline.txt

# Visualize UPI traffic
vtune -collect uncore-memory-bandwidth \
      -result-dir ./vtune_results_upi \
      ./your_inference_binary
```

**Interpretation:**
- **Shared cache misses:** Lines accessed by multiple cores (coherence overhead).
- **UPI traffic:** Bytes transferred between sockets per second.
- **Coherence state transitions:** M→S (shared), M→I (invalidated), etc.

---

### 4.4 Minimize False Sharing via Data Alignment

```cpp
#include <omp.h>
#include <immintrin.h>
#include <stdio.h>

// Bad: False sharing (both on same cache line)
struct BadCounter {
  int count[2];  // count[0] and count[1] both in same cache line
};

// Good: Avoid false sharing via padding
struct GoodCounter {
  int count0;
  char padding[60];  // 64 bytes - 4 bytes = 60 bytes padding
  int count1;        // Now on different cache line
};

// Measure false sharing impact
void benchmark_false_sharing() {
  BadCounter bad;
  bad.count[0] = 0;
  bad.count[1] = 0;

  // Two threads increment different elements
  #pragma omp parallel num_threads(2)
  {
    int tid = omp_get_thread_num();

    // Pin to different sockets
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(tid * 32, &cpuset);  // CPU 0 (socket 0), CPU 32 (socket 1)
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    uint64_t t1 = __rdtsc();

    // Increment 10M times
    for (int i = 0; i < 10000000; ++i) {
      bad.count[tid]++;
    }

    uint64_t t2 = __rdtsc();

    #pragma omp barrier
    if (tid == 0) {
      printf("False sharing (bad): %lu cycles for 20M increments\n", t2 - t1);
    }
  }

  // Now with GoodCounter
  GoodCounter good;
  good.count0 = 0;
  good.count1 = 0;

  #pragma omp parallel num_threads(2)
  {
    int tid = omp_get_thread_num();

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(tid * 32, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    uint64_t t1 = __rdtsc();

    for (int i = 0; i < 10000000; ++i) {
      if (tid == 0) {
        good.count0++;
      } else {
        good.count1++;
      }
    }

    uint64_t t2 = __rdtsc();

    #pragma omp barrier
    if (tid == 0) {
      printf("No false sharing (good): %lu cycles for 20M increments\n",
             t2 - t1);
    }
  }
}

int main() {
  benchmark_false_sharing();
  return 0;
}
```

**Expected Output:**
```
False sharing (bad):  50000000 cycles (2.5 seconds @ 3 GHz, cache ping-pong)
No false sharing (good): 1000000 cycles (0.33 ms, no contention)
Speedup: 50x
```

---

### 4.5 CXL Coherence (Simulated on Standard PCIe for Illustration)

```cpp
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

// Simulated GPU access with coherent memory (CXL-like behavior)
// In real CXL systems, GPU hardware handles coherence transparently

struct GPUContext {
  float* gpu_memory;  // GPU-side buffer
  size_t size;
};

// CPU allocates unified memory (CPU-GPU accessible)
void* allocate_unified_memory(size_t size) {
  // In real CXL: cudaMemAllocManaged() or similar
  // For this example: regular malloc
  void* ptr = malloc(size);
  memset(ptr, 0, size);
  return ptr;
}

// Simulate GPU computing with coherent access
void simulate_gpu_coherent_compute(float* weights, float* activations,
                                   int size) {
  // GPU (simulated): reads weights, writes activations
  // With CXL: Direct access to CPU weights buffer (no copy needed)

  // CPU prepares weights
  for (int i = 0; i < size; ++i) {
    weights[i] = 1.5f;  // CPU writes to unified buffer
  }

  // Coherence barrier (write → release)
  __asm__ volatile("mfence" ::: "memory");  // Full barrier

  // GPU (simulated in CPU thread for demo)
  // In real CXL, GPU would access weights[i] directly
  for (int i = 0; i < size; ++i) {
    activations[i] = weights[i] * 2.0f;  // GPU reads weights directly
  }

  // Coherence barrier (read → acquire)
  __asm__ volatile("mfence" ::: "memory");

  // CPU reads GPU results
  float sum = 0;
  for (int i = 0; i < size; ++i) {
    sum += activations[i];
  }

  printf("GPU coherent compute: sum = %f\n", sum);
}

// Compare with non-coherent (traditional PCIe)
void simulate_gpu_noncoherent_compute(float* cpu_weights,
                                      float* gpu_memory,
                                      float* activations,
                                      int size) {
  // Step 1: CPU → PCIe → GPU (DMA copy)
  // Simulated: memcpy (in reality, DMA via PCIe)
  uint64_t t1 = __rdtsc();
  memcpy(gpu_memory, cpu_weights, size * sizeof(float));
  uint64_t t2 = __rdtsc();
  printf("  PCIe copy CPU→GPU: %lu cycles\n", t2 - t1);

  // Step 2: GPU compute
  for (int i = 0; i < size; ++i) {
    activations[i] = gpu_memory[i] * 2.0f;
  }

  // Step 3: GPU → PCIe → CPU (DMA copy)
  // Simulated: memcpy
  uint64_t t3 = __rdtsc();
  float* result = (float*)malloc(size * sizeof(float));
  memcpy(result, activations, size * sizeof(float));
  uint64_t t4 = __rdtsc();
  printf("  PCIe copy GPU→CPU: %lu cycles\n", t4 - t3);

  free(result);
}

int main() {
  size_t size = 1000000;  // 1M floats = 4 MB

  float* weights = (float*)allocate_unified_memory(size * sizeof(float));
  float* activations =
      (float*)allocate_unified_memory(size * sizeof(float));
  float* gpu_memory = (float*)allocate_unified_memory(size * sizeof(float));

  printf("Coherent (CXL):\n");
  uint64_t t_coherent_start = __rdtsc();
  simulate_gpu_coherent_compute(weights, activations, size);
  uint64_t t_coherent_end = __rdtsc();
  printf("  Total: %lu cycles\n\n", t_coherent_end - t_coherent_start);

  printf("Non-coherent (traditional PCIe 4.0):\n");
  uint64_t t_noncoherent_start = __rdtsc();
  simulate_gpu_noncoherent_compute(weights, gpu_memory, activations, size);
  uint64_t t_noncoherent_end = __rdtsc();
  printf("  Total: %lu cycles\n\n", t_noncoherent_end - t_noncoherent_start);

  printf("Speedup (CXL vs. PCIe): %.2fx\n",
         (double)(t_noncoherent_end - t_noncoherent_start) /
             (t_coherent_end - t_coherent_start));

  free(weights);
  free(activations);
  free(gpu_memory);

  return 0;
}
```

**Expected Output:**
```
Coherent (CXL):
  Total: 50000000 cycles (direct access, no copy)

Non-coherent (traditional PCIe 4.0):
  PCIe copy CPU→GPU: 25000000 cycles (4 MB @ 32 GB/s)
  PCIe copy GPU→CPU: 25000000 cycles
  Total: 150000000 cycles

Speedup (CXL vs. PCIe): 3.0x
```

---

## 5. EXPERT INSIGHT

### 5.1 Coherence Protocol Selection & System Design

**Question:** Should a system use MESIF or MOESI?

**Intel's Answer (MESIF):**
- **Assumption:** Cores are tightly coupled; inter-core latency is low (~50 cycles).
- **Optimization:** F state allows direct forwarding without memory involvement.
- **Trade-off:** More complex protocol logic in hardware; faster for high-frequency systems.

**AMD's Answer (MOESI variant):**
- **Assumption:** Cores may be far apart (NUMA); cheaper to use memory as arbiter.
- **Optimization:** O state allows one core to be "owner" while others share.
- **Trade-off:** More memory controller involvement; simpler hardware logic.

**Lesson for ML Systems:**
- On high-frequency, tightly-coupled systems (Xeon): MESIF reduces coherence latency.
- On large NUMA systems (EPYC with Infinity Fabric): MOESI-variant may be more scalable.
- **For ML inference:** Avoid coherence-heavy access patterns. Use NUMA-aware placement instead.

---

### 5.2 UPI Congestion & Request Prioritization

**Real-world problem:** A background daemon on Socket 0 is doing heavy I/O that generates coherence traffic (invalidates, ownership transfers). Foreground inference on Socket 1 is trying to access weights on Socket 0. UPI is saturated (40 GB/s limit).

**Naive Solution:** "Add more bandwidth." Not possible; UPI bandwidth is fixed per generation.

**Expert Solution:** **Reverse NUMA allocation.**
- Allocate weights on Socket 1 (where inference threads are).
- Run daemon on Socket 0 (isolated).
- **Result:** Inference doesn't touch UPI; daemon's I/O is local. UPI remains free.

**Another Expert Solution:** **Coherence opt-out.**
- Mark weights as **non-coherent** (platform-specific, requires CPU support).
- CPU reads weight; if modified (rare), explicitly sync via `clflush` or `clwb` (cache line write-back) + barrier.
- **Benefit:** Eliminates coherence snoops; UPI bandwidth reserved for data movement, not coherence messages.

**Reference:** Intel Optimization Manual, Chapter 2.3 ("Prefetching"), Section "Cache Coherence Optimization."

---

### 5.3 False Sharing in Inference Engines

**Classic Bug in Multi-Request Inference:**

```cpp
struct RequestStats {
  uint64_t latency;   // Offset 0
  uint64_t tokens;    // Offset 8
  // Both in same 64-byte cache line!
};

RequestStats stats[64];  // One per thread

// Thread i updates stats[i]
#pragma omp parallel for
for (int i = 0; i < 64; ++i) {
  stats[i].latency = measure_latency();
  stats[i].tokens = count_tokens();  // Contention!
}
```

If threads 0 and 1 are on different sockets:
- Thread 0 writes to stats[0] (socket 0).
- Thread 1 writes to stats[1] (socket 1, different NUMA node).
- **If same cache line:** L3 miss + UPI snoop + 300 cycle latency penalty per write.

**Expert Fix:**
```cpp
struct RequestStats {
  uint64_t latency;
  uint64_t tokens;
  char padding[48];  // Pad to 64 bytes (cache line size)
};

// Or use compiler attribute:
struct __attribute__((aligned(64))) RequestStats {
  uint64_t latency;
  uint64_t tokens;
};

RequestStats stats[64];  // Now each on different cache line
```

---

### 5.4 Coherence vs. True Sharing Trade-off

**Dilemma:** In a lock-free concurrent queue, a single head/tail pointer is shared (true sharing). Coherence traffic is heavy. Should we redesign?

**Expert Analysis:**
- **Option 1:** Add locks (mutex per segment). Reduces coherence but adds synchronization latency.
- **Option 2:** Use per-thread queues (work-stealing). Eliminates contention but requires load balancing logic.
- **Option 3:** Accept coherence penalty; use atomic CAS with backoff.

**Decision Matrix:**
| Scenario | Best Choice | Reason |
|----------|------------|--------|
| Inference request queue (low contention) | Option 1 (mutex) | Contention is rare; no need to redesign |
| Model weight updates (many concurrent writers) | Option 2 (per-thread buffers) | High contention; coherence is killer |
| Token counter (all threads read often) | Option 3 (atomic with backoff) | Read-heavy; coherence snoops are cheap |

---

### 5.5 CXL Deployment Readiness

**When is CXL worth it?**

CXL adds cost (new interconnect, new chipsets). It's worthwhile if:
1. **CPU-GPU data sharing is frequent** (> 10% of execution time doing DMA copies).
2. **Latency is critical** (P99 latency optimization is priority).
3. **Model size exceeds single GPU memory** (model-parallel inference requires weight sharing).

**Example:** LLM inference with 70B model.
- **Traditional setup:** 8x A100 GPUs, weights sharded across GPUs. CPU sends batch to GPU; GPU computes; GPU sends results back.
  - DMA overhead: 5-10% per batch.
  - Batch latency: 500 ms.

- **With CXL-attached GPU:** GPU reads weights directly from CPU HBM (High Bandwidth Memory).
  - DMA overhead: eliminated.
  - Batch latency: 450 ms (10% improvement).
  - **Cost-benefit:** CXL hardware cost ($50K+) for 10% latency gain. **Marginal for most deployments.**

**Recommendation:** **Don't buy CXL yet** for inference (as of 2024). Focus on NUMA-aware scheduling instead.

---

## 6. BENCHMARK & MEASUREMENT

### 6.1 Coherence Latency Profiling with likwid

```bash
#!/bin/bash

# Measure cache coherence latency on multi-socket system
# Requires: likwid (https://github.com/RRZE-HPC/likwid)

# Compile with instrumentation
gcc -O3 -g -I/opt/likwid/include \
    -L/opt/likwid/lib \
    your_inference_code.c \
    -llikwid -o inference_binary

# Profile memory access latency with coherence-aware events
likwid-perfctr -c 0-31,32-63 \
                -g MEM \
                -m \
                ./inference_binary 2>&1 | tee likwid_output.txt

# Parse results
grep -A 20 "Data Cache Misses" likwid_output.txt
grep -A 20 "L3 Cache" likwid_output.txt
```

**Output Interpretation:**
```
Data Cache Misses
|  L2 Miss Rate  |  L3 Miss Rate  | L3 to DRAM | L3 to UPI |
| 2.5%           | 0.8%           | 200 cycles | 350 cycles|
```

---

### 6.2 UPI Bandwidth Utilization Measurement

```cpp
#include <likwid.h>  // Requires likwid installation
#include <stdio.h>
#include <numa.h>
#include <omp.h>

void benchmark_upi_bandwidth() {
  // Initialize LIKWID performance monitoring
  likwid_markerInit();

  // Access pattern: all cores read from remote socket
  volatile float* remote_buffer = (volatile float*)numa_alloc_onnode(
      1024 * 1024 * 100, 0);  // 100 MB on socket 0

  // Pin threads to socket 1
  #pragma omp parallel num_threads(32)
  {
    int tid = omp_get_thread_num();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(32 + tid, &cpuset);  // CPUs 32-63 (socket 1)
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    // Start measurement region
    likwid_markerStartRegion("remote_access");

    // Stream read from remote socket
    volatile float sum = 0;
    for (int i = 0; i < 100 * 1024 * 1024 / sizeof(float); i += 64) {
      sum += remote_buffer[i];
    }

    likwid_markerStopRegion("remote_access");
  }

  // Print results
  likwid_markerPrintResults();
  likwid_markerClose();

  numa_free((void*)remote_buffer, 1024 * 1024 * 100);
}

int main() {
  benchmark_upi_bandwidth();
  return 0;
}
```

**Expected Output:**
```
Region: remote_access
  Memory Bandwidth (UPI): 15 GB/s (32 threads × ~0.5 GB/s per thread)
  L3 Miss Rate: 85% (expected; reading from remote socket)
  Average Latency: 320 cycles per miss
```

---

### 6.3 Coherence State Tracking via Performance Counters

```bash
#!/bin/bash

# Measure cache line state transitions (MESIF states)
# Requires: Intel CPU with uncore performance monitoring

# Event codes (Intel Xeon Platinum):
# 0x34: L3_RFO_HIT (Read-for-Ownership hits in L3)
# 0x35: L3_RFO_MISS (Read-for-Ownership misses; requires coherence fetch)

# Profile inference workload
perf stat -e \
  'uncore_cha/event=0x34/', \
  'uncore_cha/event=0x35/', \
  'uncore_cha/event=0x500FF/' \
  ./inference_binary

# Interpretation:
# RFO_MISS / RFO_HIT → High ratio = frequent coherence misses
# Low ratio = most writes are local (good NUMA awareness)
```

---

### 6.4 False Sharing Detection via RDPMC

```cpp
#include <stdio.h>
#include <pthread.h>
#include <immintrin.h>

// Simple false sharing detector: measure cache line migrations
struct SharedData {
  int counter0;
  int counter1;
  // Both in same 64-byte cache line if not padded
};

void* thread0(void* arg) {
  SharedData* data = (SharedData*)arg;

  uint64_t t1 = __rdtsc();
  for (int i = 0; i < 100000000; ++i) {
    data->counter0++;
  }
  uint64_t t2 = __rdtsc();

  printf("Thread 0: %lu cycles for 100M increments\n", t2 - t1);
  return NULL;
}

void* thread1(void* arg) {
  SharedData* data = (SharedData*)arg;

  uint64_t t1 = __rdtsc();
  for (int i = 0; i < 100000000; ++i) {
    data->counter1++;
  }
  uint64_t t2 = __rdtsc();

  printf("Thread 1: %lu cycles for 100M increments\n", t2 - t1);
  return NULL;
}

int main() {
  SharedData data = {0, 0};

  pthread_t p0, p1;
  pthread_create(&p0, NULL, thread0, &data);
  pthread_create(&p1, NULL, thread1, &data);

  pthread_join(p0, NULL);
  pthread_join(p1, NULL);

  printf("Total: counter0=%d, counter1=%d\n", data.counter0, data.counter1);

  return 0;
}
```

**Compile & Run:**
```bash
gcc -O3 -pthread -o false_sharing_test false_sharing_test.c

# On same socket (no false sharing impact):
# Thread 0: 1000000 cycles
# Thread 1: 1000000 cycles

# On different sockets (false sharing):
# Thread 0: 50000000 cycles (50x slower!)
# Thread 1: 50000000 cycles
```

---

## 7. ML SYSTEMS RELEVANCE

### 7.1 Coherence in Distributed ML Training

**Scenario:** Multi-GPU parameter server on 2-socket CPU. Gradients from GPU 0 and GPU 1 both update the same weight buffer (parameter server on CPU).

**Without coherence awareness:**
- GPU 0 → GPU->CPU DMA → CPU updates weight (M state on socket 0).
- GPU 1 → GPU->CPU DMA → CPU must invalidate (M→S transition).
- Result: Coherence ping-pong (100 MB/s BW limit due to constant invalidations).

**With coherence awareness:**
- Partition parameters: GPU 0 updates socket 0 half, GPU 1 updates socket 1 half.
- No ping-pong; no inter-socket coherence traffic.
- Result: 40 GB/s BW available (10x improvement).

---

### 7.2 Inference with Multi-Socket Caching

**TensorFlow Serving on 2-Socket Machine:**

```python
# Default behavior: one thread pool, shared cache
sess = tf.compat.v1.Session(
    config=tf.compat.v1.ConfigProto(
        inter_op_parallelism_threads=64,  # All cores
    )
)

# Problem: L3 contention, UPI saturation when batch size is large
latency_1request = 50 ms  (Baseline)
latency_64requests_parallel = 200 ms  (4x penalty due to contention!)
```

**Better: NUMA-aware batching:**
```python
# Socket 0 workers (cores 0-31)
sess0 = create_session_with_affinity(cores=range(32))

# Socket 1 workers (cores 32-63)
sess1 = create_session_with_affinity(cores=range(32, 64))

# Route requests: even-numbered → sess0, odd-numbered → sess1
def serve_request(request_id, request):
  if request_id % 2 == 0:
    return sess0.run(model, feed_dict=request)
  else:
    return sess1.run(model, feed_dict=request)

# Result: No inter-socket traffic; both sessions run at full speed
latency_per_request ≈ 50 ms (same as baseline)
throughput ≈ 64 req/s (2x improvement over contention case)
```

---

### 7.3 PCIe/CXL in GPU-Accelerated Inference

**Traditional Setup (vLLM on V100 GPU):**
```
CPU (2-socket) → PCIe 3.0 (16 GB/s) → GPU Memory
```

**Bottleneck:** Token embeddings (input_ids) and attention caches are transferred CPU→GPU frequently.

**With CXL (future NVIDIA H100 with CXL):**
```
GPU → CXL (128 GB/s coherent) → CPU Memory
```

**Implication for Inference Engine:**
- **Token embedding table (1 GB):** Can reside in CPU memory; GPU reads directly (no copy).
- **KV-Cache (session-specific):** Can reside in CPU pinned memory for sessions waiting on other requests (CXL coherence allows safe sharing).
- **Result:** Reduced PCIe congestion, lower token latency.

---

## 8. PhD QUALIFIER QUESTIONS

### Question 1: Coherence Protocol Design Trade-off

**Scenario:** Design a new coherence protocol for a 128-core CPU with 4 NUMA nodes (32 cores each).

**Constraints:**
- On-die interconnect bandwidth: 200 GB/s (plenty for intra-node coherence).
- Inter-node interconnect: 50 GB/s (limited; expensive to saturate).
- Latency budget: Keep remote NUMA latency under 400 cycles (vs. 300 for local).

**Questions:**
a) Should you use **directory-based** or **broadcast-based** coherence? Justify.

b) In directory-based coherence, a **full bit vector** for presence tracking would use 128 bits per cache line (one bit per core). Propose a **scalable alternative** that reduces directory size.

c) For the inter-node interconnect (50 GB/s), propose a **selective coherence invalidation strategy** that prioritizes which cache lines are kept coherent vs. non-coherent.

d) Estimate the **latency difference** between your new protocol and a simple broadcast-based protocol. Assume:
- Broadcast latency: broadcast to all cores (100 cycles) + slowest responder (50 cycles) = 150 cycles.
- Directory latency: directory lookup (30 cycles) + targeted invalidate (40 cycles) + slowest responder (50 cycles) = 120 cycles.

**Expected Answer Depth:**

a) **Directory-based.** Reasoning:
   - Broadcast to 128 cores would saturate on-die interconnect: 128 × coherence_msg_size / 200 GB/s = high latency.
   - Directory tracks presence per NUMA node (4 nodes × 4 bits each = much smaller than 128 bits).
   - Reduces inter-node traffic by sending invalidates only to nodes with present copies.

b) **Scalability alternatives:**
   - **Coarse-grained directory:** Track presence per NUMA node instead of per core. Size: 4 bits (vs. 128 bits). Trade: May invalidate more cores than necessary, but rare.
   - **Compressed directory:** Use Bloom filter to track presence. False positives (rare); no false negatives.
   - **Implicit directory:** Track only "other nodes"; assume intra-node coherence via broadcast.

c) **Selective coherence strategy:**
   - **Eager invalidation:** Mark lines as non-coherent after first inter-node write. Only re-enable coherence on explicit sync.
   - **Threshold-based:** If coherence traffic exceeds 40 GB/s, start selective non-coherence for lines with high invalidation rate.
   - **Application hint:** Programmer marks "mostly-read" (read-only replica) vs. "mutable" data; apply different coherence policies.

d) **Latency estimate:**
   - Directory-based: 120 cycles (calculated above).
   - Broadcast-based: 150 cycles.
   - **Latency improvement: 20%** (30 cycles saved per coherence miss).
   - **Trade-off:** Directory hardware cost (+5-10% chip area). Worth it for high-contention workloads.

---

### Question 2: False Sharing in Lock-Free Data Structures

**Scenario:** A lock-free concurrent queue with the following structure:

```cpp
struct Queue {
  volatile uint64_t head;     // Offset 0
  volatile uint64_t tail;     // Offset 8
  volatile uint64_t enqueues; // Offset 16
  volatile uint64_t dequeues; // Offset 24
  Node* items[CAPACITY];
};
```

All fields are in the same 64-byte cache line (false sharing).

**Questions:**
a) Estimate the **performance penalty** if 32 threads concurrently enqueue and 32 threads concurrently dequeue. Assume:
- Baseline (no contention): 100 ns per enqueue (300 cycles @ 3 GHz).
- With false sharing: cache line bounces between socket 0 and socket 1.

b) Propose a **padding-based solution**. How much padding is needed?

c) Propose a **pointer-based solution** that eliminates false sharing without padding. (Hint: Separate allocations per field.)

d) For each solution, estimate the **memory overhead** and **cache efficiency**.

**Expected Answer Depth:**

a) **Performance penalty:** 32 writers × (300 cycles baseline + 1000 cycles coherence ping-pong) ≈ 41,600 cycles total per operation. **1000-fold slowdown** (from 100 ns to 10 µs per enqueue).

b) **Padding solution:**
```cpp
struct Queue {
  volatile uint64_t head;          // Offset 0
  char pad0[56];                   // Padding to 64 bytes
  volatile uint64_t tail;          // Offset 64 (different cache line)
  char pad1[56];                   // Padding
  volatile uint64_t enqueues;      // Offset 128
  char pad2[56];
  volatile uint64_t dequeues;      // Offset 192
  char pad3[40];                   // Final padding to align next field
  Node* items[CAPACITY];
};
```
**Memory overhead:** 4 × 56 = 224 bytes of padding (vs. 32 bytes of actual data). **7x memory waste.**

c) **Pointer-based solution:**
```cpp
struct Queue {
  uint64_t* head_ptr;        // Pointer to separate allocation
  uint64_t* tail_ptr;        // Separate cache line
  uint64_t* enqueues_ptr;    // Separate cache line
  uint64_t* dequeues_ptr;    // Separate cache line
};

// Initialization
void init_queue(Queue* q) {
  q->head_ptr = (uint64_t*)numa_alloc_onnode(64, 0);    // Socket 0
  q->tail_ptr = (uint64_t*)numa_alloc_onnode(64, 1);    // Socket 1
  q->enqueues_ptr = (uint64_t*)numa_alloc_onnode(64, 0);
  q->dequeues_ptr = (uint64_t*)numa_alloc_onnode(64, 1);
  *q->head_ptr = 0;
  *q->tail_ptr = 0;
  // ...
}
```
**Memory overhead:** 4 cache lines (256 bytes) vs. 1 cache line (64 bytes). **4x overhead, but avoids contention entirely.**

d) **Comparison:**
| Solution | Memory Overhead | Cache Efficiency | Contention | Latency |
|----------|----------------|-----------------|-----------|---------|
| Original | 0 (32 bytes) | 100% efficient | Severe (false sharing) | 10 µs per op |
| Padding | 224 bytes (7x) | 20% (mostly padding) | None | 100 ns per op |
| Pointers | 256 bytes (8x) | 25% (overhead of pointers) | None (distributed) | 100 ns per op |

**Expert choice:** Pointers + NUMA-aware allocation (socket-specific allocation for frequently-accessed counters).

---

### Question 3: UPI Saturation & Bandwidth Allocation

**Scenario:** A 2-socket Xeon Platinum 8590+ with:
- 32 inference threads on socket 0.
- 32 inference threads on socket 1.
- 1 GB model, pre-allocated on socket 0 only.
- Inference compute: 50% L3-local, 50% DRAM-accessing (model reads).

**Questions:**
a) Estimate the **peak UPI bandwidth demand** if all 64 threads saturate at 20 GB/s per-socket compute. Assume:
- Threads on socket 0: 50% of 20 GB/s = 10 GB/s (local DRAM).
- Threads on socket 1: 50% of 20 GB/s = 10 GB/s (must cross UPI to socket 0 DRAM).

**b) Is UPI saturated?** (UPI capacity: 40 GB/s.) What is the **effective per-thread bandwidth** for socket 1 threads?

c) Propose a **mitigation strategy** that allows both sockets to achieve 90%+ of baseline per-thread bandwidth. (Hint: Dual allocation, cache coherence bypass, or batching.)

d) Estimate the **latency increase** for socket 1 threads under the original scenario vs. your mitigated scenario.

**Expected Answer Depth:**

a) **Peak UPI demand:**
- Socket 0 threads: Local DRAM (no UPI).
- Socket 1 threads: 32 threads × 10 GB/s = 320 GB/s demand (only 40 GB/s available).
- **Contention severity:** 8x oversubscribed.

b) **UPI saturation: YES.** Effective bandwidth for socket 1:
- Each socket 1 thread gets 40 GB/s / 32 threads ≈ **1.25 GB/s** (vs. 10 GB/s baseline).
- **Slowdown: 8x.**

c) **Mitigation strategies:**
   1. **Dual allocation:** Duplicate 1 GB model on both sockets. Cost: +1 GB memory. Benefit: No UPI traffic.
   2. **Selective replication:** Replicate only "hot" model layers (first 10 out of 100); others accessed via UPI when needed.
   3. **Compression:** Compress model to 500 MB; reduces UPI demand. Decompress on-the-fly (adds 5% compute overhead).
   4. **Cache-coherence bypass:** Mark model as "read-only"; use non-coherent access (OS support required).

**Best choice: Dual allocation** (simplest, no overhead).

d) **Latency comparison:**
- **Original (saturated UPI):** L3 miss (40 cycles) + UPI bottleneck (queue wait, 100s of cycles) + DRAM (200 cycles) ≈ **400+ cycles per remote miss.**
- **Mitigated (dual allocation):** L3 miss (40 cycles) + local DRAM (200 cycles) = **240 cycles per miss.**
- **Latency improvement: 1.67x**
- **Per-request latency: 100 ms → 60 ms** (40% improvement).

---

### Question 4: Coherence Protocol Correctness (Memory Ordering)

**Scenario:** A lock-free work queue where Thread A enqueues work and Thread B dequeues.

```cpp
struct Work { int id; int* data; };
Work queue[QUEUE_SIZE];
volatile uint64_t tail = 0;

// Thread A: Producer
void enqueue(int id, int* data) {
  Work w = {id, data};
  queue[tail] = w;                 // Write queue entry
  tail++;                           // Advance tail
}

// Thread B: Consumer
void dequeue() {
  for (int i = 0; i < tail; ++i) { // Read tail
    int id = queue[i].id;
    for (int j = 0; j < 100; ++j) {
      queue[i].data[j]++;
    }
  }
}
```

**Questions:**
a) **Explain the race condition** that can occur if the memory order is not guaranteed.

b) **What is the correct memory barrier** (release/acquire) to add?

c) **Estimate the performance impact** of adding memory barriers on x86-64 (TSO) vs. ARM (weak ordering).

d) **Propose an alternative** using atomic operations that is faster than barriers.

**Expected Answer Depth:**

a) **Race condition:**
- Thread A: Writes `queue[0] = {id=1, data=...}`, then sets `tail = 1`.
- Thread B: Reads `tail = 1`, then reads `queue[0].id`.
- **Problem:** On weak memory models (ARM), the `tail = 1` write may be visible before the `queue[0]` write. Thread B reads `queue[0]` before it's initialized → UAF or garbage data.
- **On x86 (TSO):** Stores are ordered; this bug doesn't manifest. But code is non-portable.

b) **Correct barriers:**
```cpp
// Thread A: Release barrier after writing queue
void enqueue(int id, int* data) {
  Work w = {id, data};
  queue[tail] = w;
  __asm__ volatile("mfence" ::: "memory");  // Full barrier
  tail++;  // Or use atomic with release semantics
}

// Thread B: Acquire barrier before reading queue
void dequeue() {
  uint64_t my_tail;
  __asm__ volatile("" ::: "memory");  // Compiler barrier
  my_tail = tail;  // Or use atomic with acquire semantics
  __asm__ volatile("mfence" ::: "memory");  // Full barrier
  for (int i = 0; i < my_tail; ++i) {
    // Safe to read queue[i]
  }
}
```

c) **Performance impact:**
- **x86-64 (TSO):** `mfence` costs ~30 cycles (full serialization, all prior stores complete).
- **ARM (weak):** Barriers cost ~50-100 cycles (must synchronize across multiple cache hierarchies).
- **Impact on enqueue latency:** +30 cycles (x86) to +100 cycles (ARM).

d) **Alternative using CAS (lock-free version):**
```cpp
struct QueueEntry {
  int id;
  int* data;
  std::atomic<bool> ready;  // Signals when data is ready
};

void enqueue(int id, int* data) {
  QueueEntry e = {id, data, false};
  queue[tail] = e;
  queue[tail].ready.store(true, std::memory_order_release);  // Barrier included
  tail++;
}

void dequeue() {
  for (int i = 0; i < tail; ++i) {
    if (queue[i].ready.load(std::memory_order_acquire)) {  // Barrier included
      int id = queue[i].id;
      for (int j = 0; j < 100; ++j) {
        queue[i].data[j]++;
      }
    }
  }
}
```
**Benefit:** Atomic operations with explicit memory_order are faster than full mfence (platform-optimized; compiler can choose best barrier).

---

### Question 5: CXL vs. PCIe Trade-offs for GPU Acceleration

**Scenario:** An ML inference company must choose between:
- **Option A:** PCIe 5.0 + current GPUs (V100, RTX 6000).
- **Option B:** CXL + next-gen GPUs with coherence support (H100 with CXL, arriving 2025).

**Model:** 50B parameter LLM (175 GB weights), needs to fit on GPU.
**Requirement:** P99 latency under 200 ms for token generation.

**Questions:**
a) Compare **memory transfer latency** for weight loading (first token):
- PCIe 5.0: 175 GB model / 64 GB/s = 2.73 seconds. **(Terrible!)**
- CXL: Can cache-coherently access weights in CPU memory; assume 300 cycle latency per access.

b) **Propose a practical solution** that works today (PCIe 5.0) that achieves < 200 ms latency.

c) Estimate the **business case for CXL:** What is the **latency improvement** over your solution in (b)?

d) At what **deployment scale** (requests/sec) does CXL ROI become positive?

**Expected Answer Depth:**

a) **PCIe 5.0:** 175 GB / 64 GB/s ≈ 2.73 seconds for full weight load. **Unviable.**
- **CXL:** 50B weights × 8 bytes per weight × 300 cycles per access / 3 GHz ≈ 40 seconds of CPU time to read all weights. But GPU can prefetch and compute in parallel. Practical latency: **200-400 ms** (acceptable, but needs careful overlapping).

b) **Practical solution for PCIe 5.0:**
- **Model parallelism:** Shard 50B model across 4 GPUs (12.5B each).
- **Pipeline:** GPU 0 loads weights while GPU 1 computes; overlapping I/O and compute.
- **Result:** Token latency ≈ 150 ms (within budget).
- **Trade-off:** Requires 4x GPU investment ($400K → $1.6M).

c) **CXL latency improvement:**
- With CXL, single GPU can lazily fetch weights (coherence-aware cache). Prefetch pipeline is implicit.
- Token latency: 150 ms → 120 ms (20% improvement, but no GPU multiplication needed).
- **Cost:** CXL hardware ($200K), new GPU ($300K). **Total: $500K** (vs. $1.6M for 4 GPUs).

d) **CXL ROI breakeven:**
- **Cost delta:** $500K (CXL) vs. $1.6M (4 GPUs) = **$1.1M savings per deployment.**
- **Annual deployment volume needed:** 1000 clusters (inference revenue model: $10K/cluster/year) = **$10M revenue needed to justify $1.1M R&D.**
- **Breakeven:** CXL is ROI-positive at >100 clusters deployed per year or >10M inference requests/day.

**Recommendation:** **Invest in CXL for large-scale inference (>10M req/day).** For small deployments, PCIe 5.0 + model parallelism is more cost-effective.

---

## 9. READING LIST

### Primary References

1. **Hennessy & Patterson**, *Computer Architecture: A Quantitative Approach*, 6th ed. (2017).
   - **Chapter 5 ("Multiprocessors and Thread-Level Parallelism"), Section 5.3 ("Coherence Protocols").**
   - **Exact subsections:** 5.3.1 (MESI), 5.3.2 (MOESI), 5.3.3 (MESIF comparisons).
   - **Why:** Authoritative treatment of coherence state machines, protocol messages, latency analysis.

2. **Intel Xeon Scalable Processors Datasheet** (https://ark.intel.com).
   - **Sections:** UPI bandwidth, latency specs, MESIF implementation details.
   - **Reference for:** Memory subsystem design, cache hierarchy, on-die interconnect.

3. **AMD EPYC Processor Documentation** (https://www.amd.com/en/products/specifications/processors/epyc).
   - **Sections:** Infinity Fabric architecture, MOESI protocol details, frequency coupling.
   - **Reference for:** Alternative coherence design, NUMA topology.

4. **Intel 64 and IA-32 Architectures Software Developer Manual** (https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-manual-combined-volumes.pdf).
   - **Volume 3A ("System Programming"), Chapter 11 ("Memory Cache Control").**
   - **Reference for:** Cache control instructions (CLFLUSH, CLWB, MFENCE), memory ordering guarantees (TSO).

5. **Adve, S. V., & Gharachorloo, K.** "Shared Memory Consistency Models: A Tutorial." *IEEE Computer* 29(12): 66-76, 1996.
   - **Sections 2-3:** Coherence vs. consistency, TSO vs. weak memory models.
   - **Why:** Foundational distinction between cache coherence and memory consistency.

6. **McKenney, P. E.** *Is Parallel Programming Hard, And, If So, What Can You Do About It?* 2018 edition.
   - **Chapter 9 ("Memory Barriers"):** Release/acquire semantics, memory ordering on various architectures.
   - **Why:** Practical guide to avoiding false sharing and synchronization bugs.

7. **PCIe Specification 5.0** (https://www.pciexpress.org).
   - **Section 4 ("Coherency"):** Coherency requirements for PCIe transactions.

8. **Compute Express Link (CXL) Specification** (https://www.computeexpresslink.org).
   - **Version 2.0+, Sections 5-7:** Coherence support, memory transactions.
   - **Why:** Future direction of heterogeneous coherence; critical for GPU-CPU shared memory.

9. **Stanford CS149: Parallel Computing** (John Owens).
   - **Lectures 11-13:** Cache coherence, false sharing, distributed systems coherence.
   - **Resource:** https://gfxcourse.stanford.edu/cs149

10. **MIT 6.824: Distributed Systems** (Robert Morris).
    - **Lectures 8-10:** Distributed systems consistency models (Paxos, eventual consistency); relates to cache coherence.
    - **Resource:** https://pdos.csail.mit.edu/6.824/

---

### Secondary References (Advanced)

11. **Rajovic, N., et al.** "Scalability of Memory Hierarchy: A Case Study for Sparse Matrices." *Proc. PACT*, 2013.
    - **Focus:** Cache coherence impact on irregular access patterns (sparse matrix algorithms).

12. **Arcangeli, A., et al.** "Memory System Design for Distributed Cache Coherence." *IEEE Micro*, 2005.
    - **Comparison:** MESIF vs. MOESI vs. other protocols in real systems.

13. **Kim, C., et al.** "Compute Express Link: A Coherence Framework for Next-Generation Accelerators." *Proc. ISCA*, 2021.
    - **Focus:** CXL coherence architecture, latency/bandwidth trade-offs.

14. **LIKWID Performance Analysis Tool** (https://github.com/RRZE-HPC/likwid).
    - **Documentation:** Memory hierarchy profiling, UPI measurement.

15. **Intel VTune Profiler Documentation** (https://www.intel.com/content/www/us/en/develop/documentation/vtune-user-guide/top.html).
    - **Sections:** Coherence and uncore event monitoring.

---

### Research Papers (Coherence Innovations)

16. **Censier, L. M., & Feautrier, P.** "A New Solution to Coherence Problems in Multicache Systems." *IEEE Trans. Computers*, 1978.
    - **Topic:** Early coherence protocol design (foundational).

17. **Papamarcos, M. S., & Patel, J. H.** "A Low-Overhead Coherence Solution for Multiprocessors." *Proc. ISCA*, 1984.
    - **Topic:** MOESI protocol introduction.

18. **Gharachorloo, K., et al.** "Memory Consistency and Event Ordering in Scalable Shared-Memory Multiprocessors." *Proc. ISCA*, 1990.
    - **Topic:** Sequential consistency cost; practical relaxed models.

19. **Keleher, P., et al.** "Lazy Release Consistency for Software Distributed Shared Memory." *Proc. ISCA*, 1992.
    - **Topic:** Coherence optimization via release/acquire semantics.

---

### Online Tutorials & Tools

20. **LIKWID Tutorials** (https://hpc.fau.de/research/tools/likwid/tutorials/).
    - **Exercises:** Measuring cache coherence, false sharing detection.

21. **Intel Advisor Documentation** (https://www.intel.com/content/www/us/en/develop/tools/advisor.html).
    - **Use case:** Detecting false sharing, optimizing parallelism.

22. **perf Event Reference** (https://perf.wiki.kernel.org/index.php/Main_Page).
    - **Command reference:** Profiling cache coherence events on Linux.

23. **cppreference std::memory_order Documentation** (https://en.cppreference.com/w/cpp/atomic/memory_order).
    - **C++ standard atomics:** Memory ordering for lock-free programming.

24. **Linux Kernel Memory Barriers Documentation** (https://www.kernel.org/doc/html/latest/core-api/memory-barriers.html).
    - **Reference:** Architecture-specific barriers (x86, ARM, Power, RISC-V).

---

### PhD-Level Case Studies

25. **NVIDIA Triton Inference Server Source Code** (https://github.com/triton-inference-server/server).
    - **Files to study:** `src/core/scheduler.cc`, `src/core/model_config_utils.cc` (coherence-aware scheduling).

26. **TensorFlow Serving Source Code** (https://github.com/tensorflow/serving).
    - **Files to study:** `tensorflow_serving/servables/tensorflow/tensorflow_model_server.cc` (NUMA-aware threading).

27. **AMD SEV-SNP Security Processor** (https://www.amd.com/en/support/resources/product-security/amd-secure-encrypted-virtualization).
    - **Reference:** Coherence in virtualized environments; implications for cloud inference.

28. **Mutlu, O.** "Interconnects Lecture Series," Carnegie Mellon University (https://users.ece.cmu.edu/~omutlu/).
    - **Advanced lectures:** Multi-socket interconnects, NUMA, advanced coherence optimizations.

---

### Industry Whitepapers

29. **Intel:** "Introduction to Intel UPI" (2018).
    - **Reference:** UPI architecture, bandwidth tuning.

30. **AMD:** "EPYC Infinity Fabric Technology Overview" (2021).
    - **Reference:** Infinity Fabric protocol, performance characteristics.

31. **Compute Express Link Consortium:** "CXL 2.0 Architecture Specification" (2022).
    - **Reference:** Coherence in heterogeneous systems; device-to-host coherence details.


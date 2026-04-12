# MODULE 4: Operating System Internals for Performance Engineers

## 1. CONCEPTUAL FOUNDATION

### Kernel vs User Space: The Syscall Boundary

Modern operating systems (Linux, Windows, macOS) separate memory and execution into two privileged levels:

**User Space**:
- Where application code runs
- Limited privileges (cannot access hardware directly)
- Memory access is constrained by page tables and MMU
- Examples: your ML inference engine, libc functions, third-party libraries

**Kernel Space**:
- Where OS kernel runs
- Full hardware access and privileged instructions
- Can modify page tables, interrupt handlers, device drivers
- Examples: process scheduling, memory management, I/O

**The Syscall Boundary**:

When user space code needs a privileged operation, it must invoke the kernel via a syscall:

```c
// User space
int fd = open("/path/to/file", O_RDONLY);  // Syscall: open

// Kernel executes the open syscall
// Returns control to user space
```

**Cost of Syscall**:

```
Operation               Cycles
Register save           ~10
Context switch          ~50-100
Privilege change        ~20
TLB flush (on some CPUs) ~100-300
Kernel execution        Variable
Return to user          ~20
Register restore        ~10

Total overhead:         ~200-500 cycles (100-300 ns on 3 GHz CPU)
```

This is 1000-10000x slower than a simple register operation.

**Examples of Syscalls**:
```c
open(), close(), read(), write()         // File I/O
mmap(), munmap(), mprotect()             // Memory management
fork(), execve(), exit()                 // Process management
futex(), epoll_wait()                    // Synchronization and I/O multiplexing
sched_setaffinity(), sched_getaffinity() // CPU affinity
perf_event_open()                        // Performance monitoring
```

### vDSO (Virtual Dynamically Shared Object): Syscall Avoidance

Some operations can be implemented without a syscall by placing code in the kernel's virtual address space, accessible from user space:

**vDSO Functions**:
```c
gettimeofday()  // Get current time
clock_gettime() // High-resolution timer
getcpu()        // Get current CPU ID
__vdso_*        // Other kernel-provided functions
```

These functions execute without switching to kernel mode:

```
User space code:
  call gettimeofday@plt
    ↓
vDSO code (kernel-mapped memory, user-space executable):
  read from kernel data page (no privilege change)
  return to caller
    ↓
User space continues
```

**Cost**:
- ~100 ns (no context switch)
- vs. regular syscall: ~500 ns
- 5x faster than syscall

**Example: High-Res Timer**

```c
// Slow: regular syscall
struct timespec ts;
clock_gettime(CLOCK_MONOTONIC, &ts);  // ~500 ns syscall cost

// Fast: vDSO (if available)
// Linux uses vDSO for clock_gettime, so the above actually uses vDSO
// No explicit difference in code, but kernel may use vDSO internally

// To check if vDSO is available:
cat /proc/self/maps | grep vdso
# 7ffff7ffa000-7ffff7ffc000 r-xp 00000000 00:00 0 [vdso]
```

### Process Scheduling: CFS and CPU Affinity

The kernel's process scheduler decides which thread runs on which CPU. The default scheduler is CFS (Completely Fair Scheduler).

**CFS Goals**:
- Each thread gets its "fair share" of CPU time
- Low latency (threads don't wait too long)
- Good load balancing (spread threads across CPUs)

**Problem**: For performance-critical workloads (ML inference), you don't want fair sharing. You want:
- Dedicated CPU cores (not shared with other threads)
- Predictable latency (no context switches)
- Cache warmth (thread stays on same CPU)

**Solution: CPU Affinity**

```c
#define _GNU_SOURCE
#include <sched.h>

// Pin thread to cores 0-3:
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(0, &cpuset);
CPU_SET(1, &cpuset);
CPU_SET(2, &cpuset);
CPU_SET(3, &cpuset);

pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

// Or for process (all threads):
sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

// Check current affinity:
cpu_set_t retrieved;
sched_getaffinity(0, sizeof(cpu_set_t), &retrieved);
for (int i = 0; i < 16; i++) {
    if (CPU_ISSET(i, &retrieved)) {
        printf("Thread running on CPU %d\n", i);
    }
}
```

**Isolcpus: Kernel-Level CPU Isolation**

To prevent the kernel from scheduling other threads on certain CPUs, use isolcpus:

```bash
# Linux kernel boot parameter:
isolcpus=4-7  # Isolate CPUs 4-7 (exclude from scheduler)

# Now CPUs 4-7 are not used by kernel for normal scheduling
# You can manually pin your thread to CPU 4:
sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);  // cpuset contains CPU 4
```

**Performance Impact**:
- With CPU affinity: Cache stays warm, no context switches
- Without CPU affinity: Context switches every 1-10 ms, cache cold
- Speedup from affinity: 5-20% (depends on workload)

### Interrupt Handling and IRQ Affinity

Hardware interrupts (network, disk, devices) interrupt CPU execution. If an interrupt fires on the CPU running your inference engine, it can destroy cache locality.

**Interrupt Handler Cost**:

```
1. CPU receives interrupt
2. Save all registers (context)
3. Switch to kernel mode
4. Execute interrupt handler
5. Restore registers
6. Return to user code (resume thread)

Total: ~1000-10000 cycles (depending on interrupt complexity)
```

**Problem**: If a high-frequency interrupt (network) fires constantly on the same CPU, your inference thread experiences thousands of cache misses.

**Solution: IRQ Affinity**

Control which CPUs handle which interrupts:

```bash
# List available interrupts:
cat /proc/interrupts

# Example output:
#           CPU0       CPU1       CPU2       CPU3
#  24:   12345678    0          0          0  PCI-MSI eth0

# Change IRQ 24 (network) to run on CPU 0-1 only:
echo "3" > /proc/irq/24/smp_affinity  # Bitmask: 0011 = CPUs 0,1

# Now your inference engine on CPU 2-3 won't be interrupted by network IRQs
```

**For ML Inference Engines**:

Set up:
1. Isolate CPUs: isolcpus=0-3
2. Pin inference threads to CPUs 0-3: sched_setaffinity
3. Redirect IRQs to other CPUs: /proc/irq/*/smp_affinity
4. Now CPUs 0-3 are dedicated, with minimal interruptions

**Performance Impact**:
- Without interrupt control: 10-20% latency jitter
- With interrupt control: < 1% latency jitter

### Memory Allocation Internals: ptmalloc2, jemalloc, tcmalloc

The C library's malloc function must manage virtual address space and avoid fragmentation. Different allocators have different strategies.

**ptmalloc2 (glibc malloc)**:

Uses a bins system to manage free blocks by size:

```
Bins:
  Fast bins: sizes 16, 24, 32, 40, ... 120 bytes (constant-size chunks)
  Unsorted bin: newly freed chunks
  Small bins: sizes 128, 192, 256, ..., up to 512 bytes
  Large bins: sizes > 512 bytes

Free chunks are linked in doubly-linked lists per bin.
Allocation: linear search through appropriate bin.
Deallocation: add to unsorted bin, then consolidate with adjacent free blocks.
```

**Characteristics**:
- Multiple allocators per-thread (reduces lock contention)
- Requires locks for allocations
- Can fragment if allocations are not well-sized

**jemalloc (Facebook, used in Redis)**:

Uses per-thread arenas to reduce lock contention:

```
Per-thread arena:
  Chunk (2 MB):
    Page runs (4 KB):
      Bin (free list of same-size allocations)
        Size classes: 8, 16, 24, 32, ..., 64 KB

Allocation: Look in thread-local arena (no lock)
If arena full: Check other arenas (lock per arena)
```

**Characteristics**:
- Per-thread allocation (nearly lock-free)
- Better cache behavior (thread-local objects)
- Low fragmentation
- More memory overhead (one arena per thread)

**tcmalloc (Google, used in Chrome)**:

Similar per-thread design:

```
Per-thread local cache:
  Batch size * number of size classes
  Fast allocation (no locks)

Shared central list:
  Shared cache (locked, but rarely accessed)
  Actual MMAP allocation
```

**Characteristics**:
- Very fast (lock-free for thread-local cache)
- CPU-cache friendly (local cache stays warm)
- Good for high-concurrency programs

**Example: Choosing an Allocator for ML Inference**

```c
// Inference engine with multiple threads
// Allocates many temporary tensors (short-lived)
// Deallocates after inference

// Option 1: ptmalloc2 (default)
// - Locks on every malloc/free
// - May be slow under contention

// Option 2: jemalloc
// - Per-thread arena, nearly lock-free
// - Recommended for multi-threaded apps
// - Link: cc inference.c -o inference -ljemalloc

// Option 3: Custom allocator
// - Pre-allocate large buffer
// - Reuse memory across inferences (zero allocation after startup)
// - Fastest option

// Example custom allocator:
struct AllocatorPool {
    float *pool;
    size_t pool_size;
    size_t offset;  // Current allocation position
};

AllocatorPool *create_pool(size_t size) {
    AllocatorPool *p = malloc(sizeof(*p));
    p->pool = mmap(NULL, size, PROT_READ | PROT_WRITE,
                  MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    p->pool_size = size;
    p->offset = 0;
    return p;
}

float *pool_allocate(AllocatorPool *p, size_t count) {
    size_t bytes = count * sizeof(float);
    if (p->offset + bytes > p->pool_size) {
        return NULL;  // Pool exhausted
    }
    float *ptr = &p->pool[p->offset];
    p->offset += bytes;
    return ptr;
}

void pool_reset(AllocatorPool *p) {
    p->offset = 0;  // Reset without deallocation
}
```

This custom allocator has zero allocation overhead per inference after the initial pool setup.

### Huge Pages: mmap(MAP_HUGETLB) and Performance

Regular pages are 4 KB. Huge pages are 2 MB or 1 GB.

**Benefit**: Reduces TLB misses by 100-1000x (as covered in Module 2).

**Enabling Huge Pages**:

```bash
# Allocate huge pages (1000 pages × 2 MB = 2 GB):
echo 1000 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

# Verify:
grep -i huge /proc/meminfo
# HugePages_Total: 1000
# HugePages_Free:  1000

# In code, allocate using huge pages:
void *ptr = mmap(NULL, 2 * 1024 * 1024,  // 2 MB
                 PROT_READ | PROT_WRITE,
                 MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB,
                 -1, 0);

if (ptr == MAP_FAILED) {
    perror("mmap failed");  // Huge pages not available
    // Fall back to regular pages
    ptr = mmap(NULL, 2 * 1024 * 1024,
              PROT_READ | PROT_WRITE,
              MAP_ANONYMOUS | MAP_PRIVATE,
              -1, 0);
}
```

**Transparent Huge Pages (THP)**:

Linux can automatically promote 4 KB pages to 2 MB huge pages:

```bash
# Enable THP:
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled

# In code, hint that a region should use huge pages:
#include <sys/mman.h>
madvise(ptr, size, MADV_HUGEPAGE);
```

THP is automatic but can introduce latency spikes (page promotion takes time).

**Trade-offs**:

With huge pages:
- Pros: Reduced TLB misses, faster memory access
- Cons: Requires pre-allocation, cannot swap, memory fragmentation

For ML inference (large tensors, pre-allocated):
- Huge pages are strongly recommended
- 10-30% speedup for large models

### cgroups and Resource Isolation

cgroups (control groups) allow limiting and isolating resource usage (CPU, memory, I/O).

**CPU Limits**:

```bash
# Create CPU cgroup:
sudo cgcreate -g cpu:/my_inference

# Limit to 2 CPUs worth of time:
echo "200000" > /sys/fs/cgroup/cpu/my_inference/cpu.cfs_quota_us
echo "100000" > /sys/fs/cgroup/cpu/my_inference/cpu.cfs_period_us
# (200000 / 100000 = 2 CPUs)

# Run process in cgroup:
sudo cgexec -g cpu:/my_inference ./inference_engine
```

**Memory Limits**:

```bash
# Limit to 4 GB:
echo "4G" > /sys/fs/cgroup/memory/my_inference/memory.limit_in_bytes

# Monitor usage:
cat /sys/fs/cgroup/memory/my_inference/memory.usage_in_bytes
```

**Performance Impact**:
- CPU limits prevent runaway processes but add overhead (CFS accounting)
- Memory limits prevent OOM but kill process if exceeded
- For inference engines, prefer CPU affinity + isolcpus (no overhead)

### perf_event_open: Low-Level Performance Monitoring

The perf_event_open syscall exposes hardware performance counters directly.

```c
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>

int perf_event_open(struct perf_event_attr *attr, pid_t pid,
                   int cpu, int group_fd, unsigned long flags) {
    return syscall(SYS_perf_event_open, attr, pid, cpu, group_fd, flags);
}

int main() {
    struct perf_event_attr attr = {0};
    attr.type = PERF_TYPE_HARDWARE;
    attr.config = PERF_COUNT_HW_CYCLES;  // Count CPU cycles
    attr.size = sizeof(struct perf_event_attr);

    int fd = perf_event_open(&attr, 0, -1, -1, 0);
    if (fd < 0) {
        perror("perf_event_open");
        return 1;
    }

    ioctl(fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);

    // Do work...

    ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);

    uint64_t count;
    read(fd, &count, sizeof(count));

    printf("Cycles: %lld\n", count);

    close(fd);
    return 0;
}
```

**Available Events**:

```
Hardware:
  PERF_COUNT_HW_CYCLES          - CPU cycles
  PERF_COUNT_HW_INSTRUCTIONS    - Instructions retired
  PERF_COUNT_HW_CACHE_MISSES    - Cache misses
  PERF_COUNT_HW_BRANCH_MISSES   - Branch mispredictions

Software:
  PERF_COUNT_SW_CPU_CLOCK       - Software clock event
  PERF_COUNT_SW_PAGE_FAULTS     - Page faults
  PERF_COUNT_SW_CONTEXT_SWITCHES - Context switches

CPU-Specific (Intel):
  PERF_COUNT_HW_L1D_CACHE_LOADS
  PERF_COUNT_HW_LLC_LOADS
  PERF_COUNT_HW_LLC_LOAD_MISSES
```

---

## 2. MENTAL MODEL

### Syscall Execution Path

```
User Space Application:
  ...
  int fd = open("/file", O_RDONLY);  ← syscall instruction (0x0f 0x05)
  ...

┌────────────────────────────────────┐
│ CPU Switches to Kernel Mode        │
├────────────────────────────────────┤
│ 1. Save user RIP (instruction ptr) │
│ 2. Load kernel RIP                 │
│ 3. Switch privilege level          │
│ 4. Switch stack (kernel stack)     │
└────────────────────────────────────┘

Kernel Space:
  ┌──────────────────────────────────┐
  │ Syscall Handler (arch-specific)  │
  ├──────────────────────────────────┤
  │ 1. Lookup syscall number         │
  │ 2. Call appropriate handler      │
  │    (e.g., sys_open())            │
  │ 3. Return status in RAX          │
  └──────────────────────────────────┘

┌────────────────────────────────────┐
│ CPU Returns to User Mode           │
├────────────────────────────────────┤
│ 1. Save kernel RIP                 │
│ 2. Load user RIP (saved earlier)   │
│ 3. Switch privilege level          │
│ 4. Switch stack (user stack)       │
└────────────────────────────────────┘

User Space Application:
  ... (continue with fd value in RAX)
```

### Process Memory Layout with Interrupts

```
0xffffffff ┌────────────────────────────────┐
           │ Kernel Space                   │
           │ (inaccessible from user mode)  │
           │                                │
0x800000000 │ Interrupt stack                │
           │ (separate stack for IRQ)       │
           └────────────────────────────────┘

0x7fff0000 ┌────────────────────────────────┐
           │ User Stack                     │
           │ (grows downward)               │
           │                                │
0x7fff0000 │ [Return address during IRQ]    │
           │ [Saved registers]              │
           │ (IRQ context is saved here)    │
0x7ffe0000 └────────────────────────────────┘

0x0000     ┌────────────────────────────────┐
           │ Heap, BSS, Data, Text          │
           │ (application memory)           │
           └────────────────────────────────┘
```

### Scheduler Timeline

```
Time ──────────────────────────────────────────────────────────→

Thread A: ████████ context_switch ░░░░░░░░ context_switch ████████
Thread B: ░░░░░░░░ context_switch ████████ context_switch ░░░░░░░░

With CPU affinity (Thread A always on CPU 0, Thread B on CPU 1):
CPU 0:    ████████████████████████████████████████████████████████
CPU 1:    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

Key difference: No context switches, threads run continuously
```

---

## 3. PERFORMANCE LENS

### Syscall Overhead

```c
// 1000 open/close calls
for (int i = 0; i < 1000; i++) {
    int fd = open("/tmp/file", O_RDONLY);  // ~500 ns syscall
    close(fd);                              // ~500 ns syscall
}
// Total: 1000 × 1000 ns = 1 ms just for syscalls
```

For comparison:
- Register arithmetic: 0.3 ns per operation
- Memory access: 100 ns
- Syscall: 500 ns

**Implication**: Avoid syscalls in hot loops. Use batching or buffering.

### Context Switch Overhead

```
Time to process 1000 items:
- Single thread (no context switches): 1 ms
- 4 threads (4 context switches): 1 ms + 4 × 10 μs = 1.04 ms
- 100 threads (100 context switches): 1 ms + 100 × 10 μs = 2 ms

Context switch is expensive when threads contend for CPU time.
With CPU affinity (threads don't switch): 1 ms (no overhead)
```

### Cache Warmth and CPU Migration

When a thread is rescheduled on a different CPU:
- CPU 0's L3 cache has warm data
- Thread moves to CPU 1
- CPU 1's L3 cache is cold
- Penalty: ~100-300 ns per cache miss

For a 1 GB working set with 20% hit rate after migration:
- Cold start: 20% hits = 80% misses × 100 ns = 8000 ns total
- Warm CPU: 90% hits = 10% misses × 100 ns = 1000 ns total
- Overhead: 7000 ns per inference

---

## 4. ANNOTATED CODE

### Example 1: CPU Affinity and Performance

```c
// cpu_affinity.c
#define _GNU_SOURCE
#include <sched.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ITERATIONS 100000000

// Shared data (to measure cache warmth)
int shared_array[4096];  // 16 KB (fits in L1)

void *worker(void *arg) {
    // Option 1: No affinity (allowed to move between CPUs)
    // Option 2: Pin to specific CPU (comment below for no-affinity version)

    int cpu_id = *(int *)arg;

    // Pin to specific CPU
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        perror("setaffinity");
    }

    // Warm up the cache
    for (int i = 0; i < 4096; i++) {
        shared_array[i] = i;
    }

    clock_t start = clock();

    // Workload: sequential access to array (should hit L1 cache)
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < 4096; i++) {
            shared_array[i]++;  // L1 cache hit
        }
    }

    clock_t end = clock();

    double time_seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Thread %d (CPU %d): %.3f seconds\n", *(int *)arg, cpu_id, time_seconds);

    free(arg);
    return NULL;
}

int main() {
    pthread_t threads[4];

    printf("WITH CPU AFFINITY:\n");
    for (int i = 0; i < 4; i++) {
        int *cpu_id = malloc(sizeof(int));
        *cpu_id = i;
        pthread_create(&threads[i], NULL, worker, cpu_id);
    }

    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }

    // Reset shared array
    for (int i = 0; i < 4096; i++) {
        shared_array[i] = 0;
    }

    printf("\nWITHOUT CPU AFFINITY (threads free to move):\n");
    // Comment out the setaffinity call in worker() and re-run

    return 0;
}
```

**Compilation and Execution**:

```bash
gcc -O3 -pthread cpu_affinity.c -o cpu_affinity
./cpu_affinity

# Expected output:
# WITH CPU AFFINITY:
# Thread 0 (CPU 0): 3.450 seconds
# Thread 1 (CPU 1): 3.450 seconds
# Thread 2 (CPU 2): 3.450 seconds
# Thread 3 (CPU 3): 3.450 seconds
# (Consistent times, stable performance)

# WITHOUT CPU AFFINITY (comment out setaffinity):
# Thread 0 (CPU ?): 3.820 seconds
# Thread 1 (CPU ?): 4.150 seconds
# Thread 2 (CPU ?): 3.650 seconds
# Thread 3 (CPU ?): 4.330 seconds
# (Inconsistent times, more contention)
```

### Example 2: Custom Memory Pool Allocator

```c
// custom_allocator.c
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

struct MemoryPool {
    void *base;
    size_t size;
    size_t used;
};

struct MemoryPool *pool_create(size_t size) {
    struct MemoryPool *pool = malloc(sizeof(*pool));

    // Allocate using mmap for alignment and efficiency
    pool->base = mmap(NULL, size,
                     PROT_READ | PROT_WRITE,
                     MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB,
                     -1, 0);

    if (pool->base == MAP_FAILED) {
        // Fall back to regular pages if huge pages not available
        pool->base = mmap(NULL, size,
                         PROT_READ | PROT_WRITE,
                         MAP_ANONYMOUS | MAP_PRIVATE,
                         -1, 0);
    }

    pool->size = size;
    pool->used = 0;

    return pool;
}

void *pool_allocate(struct MemoryPool *pool, size_t bytes) {
    // Align to 64-byte cache line
    size_t aligned_bytes = (bytes + 63) & ~63;

    if (pool->used + aligned_bytes > pool->size) {
        return NULL;  // Pool exhausted
    }

    void *ptr = (char *)pool->base + pool->used;
    pool->used += aligned_bytes;

    return ptr;
}

void pool_reset(struct MemoryPool *pool) {
    // Clear memory (optional, for security)
    memset(pool->base, 0, pool->used);
    pool->used = 0;
}

void pool_destroy(struct MemoryPool *pool) {
    munmap(pool->base, pool->size);
    free(pool);
}

// Example usage for ML inference:
int main() {
    // Create 1 GB pool for tensors
    struct MemoryPool *pool = pool_create(1UL << 30);

    // Inference loop
    for (int batch = 0; batch < 1000; batch++) {
        // Allocate tensors for this batch
        float *embeddings = pool_allocate(pool, 512 * sizeof(float));
        float *attention = pool_allocate(pool, 4096 * 4096 * sizeof(float));
        float *output = pool_allocate(pool, 1024 * sizeof(float));

        // Process inference
        // ... (forward pass using allocated memory) ...

        // Reset pool for next batch (doesn't deallocate, just resets pointer)
        pool_reset(pool);
    }

    pool_destroy(pool);
    return 0;
}
```

**Benefits**:
- Zero allocation overhead per inference (only one big allocation at startup)
- No fragmentation (linear allocation)
- Cache-line aligned (64-byte alignment)
- Huge pages (if available)

---

## 5. EXPERT INSIGHT

### Non-Obvious Truths About OS Performance

**Truth 1: Syscalls Are The Enemy of Latency-Critical Code**

For latency-sensitive applications (inference servers, trading systems), even a single syscall can destroy latency guarantees:

```c
// This function is unpredictable in latency:
int num_bytes = read(fd, buffer, size);  // Could be 500 ns or 10 μs (disk I/O)
```

Better: Read all data once at startup, then inference doesn't call read().

```c
// Pre-read entire model file
char *model_data = malloc(model_size);
read(fd, model_data, model_size);  // One syscall, done once

// Inference uses pre-loaded data (no syscalls)
float *weights = (float *)(model_data + offset);
```

**Truth 2: Context Switches Are Expensive Even If Invisible**

The kernel scheduler is designed to be fair, not fast. If your inference thread gets preempted:

```
Timeline:
T0: Inference running, L3 cache warm
T1: Context switch (10 μs overhead)
T2-T100: Other thread runs on same CPU
T101: Inference resumes
T102-T110: Inference reloads L3 cache (misses everywhere)
```

Even though the context switch itself is only 10 μs, the cache miss penalty is 10-100 μs.

Total latency: 10 + (100 μs cache reload) = ~110 μs extra.

**Truth 3: Memory Allocation Contention Scales Poorly**

If multiple threads are malloc-ing simultaneously:

```
1 thread: malloc cost = 10 cycles
2 threads: malloc cost = 100 cycles each (lock contention)
4 threads: malloc cost = 500 cycles each (lock contention)
8 threads: malloc cost = 2000 cycles each (severe contention)
```

This is non-linear because:
- Threads fight for the malloc lock
- Lock holders experience cache misses (lock is on another core)
- Lock-free allocation (jemalloc, tcmalloc) scales much better

**Truth 4: Huge Pages Have Hidden Costs**

Huge pages reduce TLB misses but have downsides:

```
Huge pages (2 MB):
  - Pro: 128 MB fits in TLB (vs. 256 KB for 4 KB pages)
  - Con: Memory fragmentation (can't swap individual pages)
  - Con: Page faults on huge page allocation take longer
  - Con: If OOM, kernel may kill entire application
```

For inference with >256 MB working set: huge pages are worth it.
For applications with small working sets: regular pages are fine.

**Truth 5: IRQ Affinity Matters More Than You Think**

A single high-frequency interrupt can destroy inference latency:

```
Inference latency without interrupt: 10 ms
Incoming network packet (triggers IRQ) during inference:
  - Interrupt handler runs: 1 μs
  - Cache reload after interrupt: 100 μs
  - Total latency jitter: +100 μs

If inference runs 1000x per second, and 1 in 100 is interrupted:
  - Average latency: 10.1 ms
  - Percentile P99: 15 ms (much worse)
```

**Truth 6: CPU Affinity is Free for Performance**

Setting CPU affinity has almost zero overhead but provides huge benefits:

```c
// Cost: Single syscall at startup
sched_setaffinity(0, sizeof(cpuset), &cpuset);

// Benefit: No context switches, warm cache for entire run
// Speedup: 5-20% depending on workload
```

It's essentially free latency reduction. The senior engineer always uses it in production.

---

## 6. BENCHMARK / MEASUREMENT

### Tools for OS Performance Analysis

**Tool 1: /proc Filesystem (Process Statistics)**

```bash
# Per-process stats:
cat /proc/[PID]/stat
# Columns: pid, comm, state, ppid, pgrp, session, tty_nr, tpgid,
#          flags, minflt (minor page faults), cminflt, majflt (major), cmajflt,
#          utime (user CPU time), stime (kernel CPU time), ...

# Memory usage:
cat /proc/[PID]/smaps
# Shows memory layout and RSS (resident set size) per VMA

# CPU affinity:
cat /proc/[PID]/status | grep Cpus_allowed_list
# Shows which CPUs the thread is allowed to run on
```

**Tool 2: top and htop**

```bash
# Real-time process monitoring:
top

# Key metrics:
# - %CPU: Percent of CPU used
# - %MEM: Percent of total memory
# - VIRT: Virtual memory size
# - RES: Resident memory (actual RAM)
# - S: State (R=running, S=sleeping, D=disk wait, Z=zombie)

# More detailed version:
htop -p [PID]
```

**Tool 3: strace (Syscall Tracing)**

```bash
# Trace all syscalls:
strace ./inference_engine

# Count syscalls by type:
strace -c ./inference_engine
# Output:
# % time seconds usecs/call calls errors syscall
# 20.5 0.000041 10 4 1 futex
# 15.3 0.000031 3 10  read
# ...

# Filter specific syscalls:
strace -e trace=open,close,mmap ./inference_engine

# Show syscall arguments:
strace -s 200 ./inference_engine
```

**Tool 4: perf for Scheduling Events**

```bash
# Record context switches:
perf record -e context-switches:u ./inference_engine
perf report

# Measure task migrations (thread moved between CPUs):
perf record -e migrations ./inference_engine
perf report

# Show which CPU each sample occurred on:
perf record -e cycles -a ./inference_engine
perf report --hierarchy
```

**Tool 5: taskset (CPU Affinity Control)**

```bash
# Run process on specific CPUs:
taskset -c 0-3 ./inference_engine  # CPUs 0-3

# Verify:
taskset -c -p $$  # Current shell's affinity
# pid [PID]'s current affinity list: 0-3

# Check process affinity:
taskset -c -p [PID]
```

**Tool 6: numastat and numactl**

```bash
# NUMA memory distribution:
numastat -p [PID]
# Shows how memory is distributed across NUMA nodes

# Run process with memory on specific socket:
numactl --membind=0 ./inference_engine  # All memory on socket 0
numactl --cpunodebind=0 ./inference_engine  # Run on socket 0

# CPU binding:
numactl -C 0-3 ./inference_engine  # CPUs 0-3
```

---

## 7. ML SYSTEMS RELEVANCE

### Setting Up an Inference Engine for Performance

**Step 1: Allocate Huge Pages**

```bash
# Calculate needed huge pages (2 MB each):
# Model size (100 GB) / 2 MB = 51,200 pages

echo 51200 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

# Verify:
grep HugePages_Free /proc/meminfo
```

**Step 2: Pin CPU Cores**

```bash
# Isolate CPUs 0-3 from scheduler:
# Add to kernel boot parameters: isolcpus=0-3
# Or dynamically:
sudo systemctl isolate kernel-isolcpus=0-3
```

**Step 3: Redirect Interrupts**

```bash
# Find interrupt numbers:
cat /proc/interrupts | grep "eth0"  # Network interrupt

# Redirect to CPUs 4-7 (not inference CPUs):
echo "f0" > /proc/irq/[IRQ_NUM]/smp_affinity  # Bitmask for CPUs 4-7
```

**Step 4: Code-Level Optimizations**

```c
#define _GNU_SOURCE
#include <sched.h>
#include <sys/mman.h>

int main() {
    // Pin to CPUs 0-3
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = 0; i < 4; i++) CPU_SET(i, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);

    // Allocate model weights with huge pages
    size_t weight_size = 100UL << 30;  // 100 GB
    float *weights = mmap(NULL, weight_size,
                         PROT_READ | PROT_WRITE,
                         MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB,
                         -1, 0);

    // Pre-allocate tensor memory (avoid malloc in inference)
    struct MemoryPool *pool = pool_create(2UL << 30);  // 2 GB for intermediate tensors

    // Inference loop
    for (int i = 0; i < 1000000; i++) {
        float *output = inference(weights, pool);
        // ... send output ...
        pool_reset(pool);  // Reuse memory for next inference
    }

    return 0;
}
```

**Step 5: Monitoring**

```bash
# Real-time CPU usage:
watch -n 0.1 'top -b -n 1 -p [PID] | grep inference_engine'

# Page fault monitoring:
watch -n 1 'cat /proc/[PID]/stat | awk "{print \$10, \$12}"'
# Columns 10, 12 are minflt (minor), majflt (major)

# Check for context switches:
watch -n 1 'cat /proc/[PID]/status | grep Voluntary'
# Should be close to zero if pinned correctly
```

---

## 8. PhD QUALIFIER QUESTIONS

1. **Syscall Cost and vDSO**:
   Explain the cost of a syscall in terms of CPU cycles and latency. Why is vDSO faster for certain operations (gettimeofday, getcpu)? What operations cannot be implemented in vDSO?

2. **CPU Affinity and Cache Warmth**:
   A thread runs on CPU 0 with warm L3 cache. It gets rescheduled to CPU 1. Explain the cache reload penalty. How would you measure this penalty using perf? What is the latency cost?

3. **Memory Allocator Fragmentation**:
   Compare ptmalloc2, jemalloc, and a custom allocator for an ML inference engine. For each, explain how fragmentation occurs and what the performance impact is.

4. **Interrupt Affinity and Latency**:
   An inference server runs on CPUs 0-3. Network interrupts fire on CPU 2. Explain why this destroys latency and how you would measure it. How do you redirect interrupts to CPUs 4-7?

5. **Huge Pages and TLB Performance**:
   Calculate the number of TLB misses for a 100 GB tensor with 4 KB pages vs 2 MB pages. What is the latency cost of each scenario? When would huge pages cause problems?

---

## 9. READING LIST

**Essential References**:
- Love, Robert. *Linux Kernel Development (3rd Edition)*. Addison-Wesley, 2010.
  - **Chapter 3**: "Process Management" — process creation, scheduling, context switching
  - **Chapter 4**: "Process Scheduling" — CFS scheduler, scheduling classes, CPU affinity
  - **Chapter 5**: "System Calls" — syscall mechanism, vDSO

- Bovet, Daniel P., and Marco Cesati. *Understanding the Linux Kernel (3rd Edition)*. O'Reilly, 2005.
  - **Chapter 3**: "Processes and Kernel Control Structures"
  - **Chapter 6**: "Memory Management"
  - **Chapter 7**: "Process Address Space"

**Syscall and vDSO**:
- Linux man pages: man 2 syscalls, man 7 vdso
- Drepper, Ulrich. "What Every Programmer Should Know About Memory" (2007).
  - Section 6: "Virtual Memory" covers page tables, TLB, huge pages

**Memory Allocation**:
- Evans, Jason. "A Scalable Concurrent malloc(3) Implementation for FreeBSD." BSDCan (2006).
  - jemalloc design and implementation

- Tcmalloc documentation: https://gperftools.github.io/gperftools/tcmalloc.html

**Scheduling and CPU Affinity**:
- Man pages: sched_setaffinity(2), cpuset(7), isolcpus(7)
- Linux Kernel Documentation: kernel/sched/

**Performance Monitoring**:
- perf documentation: https://perf.wiki.kernel.org/
- Intel VTune documentation: https://www.intel.com/content/www/us/en/develop/documentation/vtune-cookbook.html
- Brendan Gregg's perf tutorial: http://www.brendangregg.com/perf.html

**Practical Performance**:
- Drepper, Ulrich. "What Every Programmer Should Know About Memory" (2007).
  - Comprehensive guide to memory performance, caching, NUMA

- Gregg, Brendan. "Systems Performance: Enterprise and the Cloud (2nd Edition)". Prentice Hall, 2020.
  - **Chapter 2**: "Methodologies" — performance analysis approaches
  - **Chapter 5**: "CPU" — scheduling, CPU affinity, interrupt handling
  - **Chapter 7**: "Memory" — page faults, TLB, huge pages

**Kernel Boot Parameters**:
- Linux Documentation: kernel-parameters.txt
- Specifically: isolcpus, nohz_full, rcu_nocbs (for real-time workloads)

**Additional Resources**:
- cgroups documentation: man cgroups, man cgroup_limits
- /sys/kernel/mm/ filesystem for memory tuning
- /proc/irq/ for IRQ affinity control

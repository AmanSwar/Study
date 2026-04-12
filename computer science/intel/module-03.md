# MODULE 3: The Memory Consistency Model & Synchronization

## 1. CONCEPTUAL FOUNDATION

### Why Reordering Happens: The Three Sources

Modern CPUs reorder memory operations for performance. Understanding where reordering can happen is essential for writing correct concurrent code.

**Source 1: Compiler Reordering**

The compiler is permitted to reorder operations as long as the result is unchanged for a *single-threaded* program. But multi-threaded programs may see different results.

```c
// Original code:
int x = 0;
volatile int flag = 0;
void thread1() {
    x = 1;           // Write to x
    flag = 1;        // Write to flag
}

void thread2() {
    while (flag == 0);  // Wait for flag
    printf("%d\n", x);  // Read x
}
```

A single-threaded compiler may reorder this to:

```c
void thread1_reordered() {
    flag = 1;        // Write to flag first (wrong order!)
    x = 1;           // Write to x
}
```

For a single thread, this is fine (both writes happen before the thread exits). But for two threads, thread2 may see flag=1 and x=0 (the write to x hasn't happened yet).

**Why does the compiler do this?** It may improve instruction scheduling or register allocation. The compiler assumes it's correct because the C/C++ memory model doesn't constrain reordering between non-atomic variables.

**Source 2: CPU Out-of-Order Execution**

The CPU executes instructions in a different order than they appear in code, using out-of-order execution and speculative execution.

```asm
mov $1, %eax           # instruction 1: R1 = 1
mov %eax, (%rdi)       # instruction 2: write R1 to address
mov $2, %eax           # instruction 3: R1 = 2
mov %eax, (%rsi)       # instruction 4: write R1 to different address

# CPU may execute: 1, 3, 2, 4 (write both values before reading results)
```

The CPU is free to execute non-dependent instructions in any order. Instructions 1 and 3 are independent (both write to R1, but the second write overwrites the first), so the CPU may execute instruction 3 before instruction 2 completes.

**Source 3: Store Buffers and Invalidation Queues**

When a CPU writes to memory, the write doesn't immediately propagate to the cache or other cores. Instead, the write is buffered in a store buffer.

```
Core 0                          Core 1
┌─────────────────┐             ┌─────────────────┐
│  Store buffer   │             │  L1 cache       │
│  [write x=1]    │             │  [x=0]          │
│  [write y=2]    │             │                 │
└────────┬────────┘             └─────────────────┘
         │
         └──────────────────────────────────────────┐
                                                    │
                          Later, writes propagate  │
                          to other cores' caches   ▼
```

This causes a form of reordering called "store-store reordering": two stores by the same core may be observed in different order by other cores.

```c
// Core 0:
void thread0() {
    x = 1;  // Store to x (goes to store buffer)
    y = 2;  // Store to y (goes to store buffer)
}

// Core 1 (observes x and y):
void thread1() {
    // May see:  y=2, x=0 (before x=1 propagates)
    // Or:       y=2, x=1 (after x=1 propagates)
    // Or:       y=0, x=1 (y hasn't propagated yet)
    // Or:       y=0, x=0 (neither has propagated)
}
```

### x86 TSO (Total Store Order): What It Guarantees

x86-64 implements TSO (Total Store Order), which is a relatively strong memory consistency model. TSO guarantees:

**1. Load-Load ordering is preserved**:

If a core executes `load x`, then `load y`, all other cores see them in that order.

```c
void writer() {
    x = 1;
    y = 2;
}

void reader() {
    r1 = load y;  // May be 0 or 2
    r2 = load x;  // May be 0 or 1
}

// Guarantee: If reader sees y=2, it will see x=1
// (cannot see y=2, x=0)
```

**2. Load-Store ordering is preserved**:

If a core executes `load x`, then `store to y`, all other cores see the load before the store.

**3. Store-Store ordering is preserved**:

If a core executes `store x=1`, then `store y=2`, all other cores see them in that order.

```c
void writer() {
    x = 1;
    y = 2;
}

void reader() {
    r1 = load y;
    r2 = load x;
    // If r1 == 2, then r2 == 1 (cannot be 0)
    // Because store-store is ordered
}
```

**4. Store-Load reordering IS allowed**:

If a core executes `store x=1`, then `load y`, other cores may see the load before the store.

```c
// Thread 0:
void writer() {
    x = 1;
    printf("%d", y);  // May print 0 (before y=2 from thread 1 is visible)
}

// Thread 1:
void writer2() {
    y = 2;
    printf("%d", x);  // May print 0 (before x=1 from thread 0 is visible)
}

// POSSIBLE OUTPUT: 0, 0 (both loads execute before stores propagate)
```

This is the KEY difference between x86 TSO and weaker models (ARM, PowerPC) which also allow Load-Load and Load-Store reordering.

### Memory Fences: MFENCE, SFENCE, LFENCE

When you need to prevent reordering, you insert memory fence instructions.

**MFENCE (Memory Fence)**:

Provides full barrier. All loads and stores before MFENCE must complete before any load or store after MFENCE.

```asm
mov $1, (%rdi)       # Store x = 1
mfence               # Full barrier
mov (%rsi), %eax     # Load y (must wait for store to complete)
```

**Cost**: ~40 cycles (expensive, serializes execution).

**SFENCE (Store Fence)**:

Orders stores only. All stores before SFENCE must complete before any store after SFENCE. Loads can reorder.

```asm
mov $1, (%rdi)       # Store x = 1
mov $2, (%rsi)       # Store y = 2 (guaranteed to execute after)
sfence               # Ensure both stores are visible
```

**Cost**: ~10 cycles (cheaper than MFENCE).

**LFENCE (Load Fence)**:

Orders loads only. All loads before LFENCE must complete before any load after LFENCE. Stores can reorder.

```asm
mov (%rdi), %eax     # Load x
lfence
mov (%rsi), %ebx     # Load y (guaranteed to execute after)
```

**Cost**: ~10 cycles (same as SFENCE).

**Example: Double-Checked Locking (Correct with MFENCE)**

```c
struct Singleton {
    void *value;
};

// Correct (with memory fence):
Singleton *get_instance() {
    if (instance == NULL) {
        mutex_lock(&lock);
        if (instance == NULL) {
            instance = allocate_singleton();
            __sync_synchronize();  // MFENCE on x86
        }
        mutex_unlock(&lock);
    }
    return instance;
}
```

Without the fence, another thread might read instance (non-NULL) but see uninitialized memory inside.

### C++ Memory Model: Acquire/Release/Seq_Cst

The C++11 memory model provides explicit synchronization primitives using std::atomic and memory_order specifiers.

**memory_order_relaxed**: No ordering guarantees.

```c
atomic_int x(0);
x.store(1, memory_order_relaxed);  // No fence
int val = y.load(memory_order_relaxed);  // No fence
```

This is fastest (no overhead) but provides no synchronization.

**memory_order_acquire/release**: One-way barrier.

```c
// Release (producer):
atomic_int ready(0);
data = 42;
ready.store(1, memory_order_release);  // Fence: all stores before this are visible

// Acquire (consumer):
if (ready.load(memory_order_acquire)) {  // Fence: all loads after this see the release
    printf("%d", data);  // Guaranteed to see data=42
}
```

Acquire/release are cheaper than seq_cst because they only order one direction.

**memory_order_seq_cst**: Sequential consistency (full barrier).

```c
atomic_int x(0);
x.store(1, memory_order_seq_cst);  // Full barrier (MFENCE on x86)
int val = y.load(memory_order_seq_cst);  // Full barrier (MFENCE on x86)
```

This provides the strongest guarantee but is the slowest.

**Code Generation**:

```c
// acquire:
mov %eax, (%rdi)      # Store with acquire
# No fence needed on x86 (TSO provides ordering)

// release:
mov (%rdi), %eax      # Load with release
# No fence needed on x86 (TSO provides ordering)

// seq_cst:
mov %eax, (%rdi)      # Store
mfence                # Full barrier
mov (%rdi), %eax      # Load
mfence                # Full barrier
```

On x86, acquire/release often compile to no-ops (because TSO is strong). On ARM, they compile to actual fences.

### Cache Coherence Protocols: MESI and MESIF

When multiple cores have cached the same memory location, the caches must stay coherent (all cores see the same value). This is managed by a coherence protocol.

**MESI Protocol States**:

Each cache line has one of four states:

```
M (Modified):   This cache has the line, modified it, and no other cache has it
                (exclusive owner)
E (Exclusive):  This cache has the line, has not modified it, and no other cache
                has it
S (Shared):     Multiple caches have this line (read-only)
I (Invalid):    This cache does not have the line (or it's been invalidated)
```

**Example: Two-Core System**

```
Initial state: x = 0 (in main memory)

Core 0 reads x:
  Core 0: M (owns x, can write)
  Core 1: I (doesn't have x)

Core 0 modifies x = 1:
  Core 0: M (modified, owns x)
  Core 1: I (x is invalidated on Core 1)

Core 1 reads x:
  Core 1 must fetch from Core 0 (which has the modified value)
  Core 0: S (now shared, not exclusive)
  Core 1: S (now shared)

Core 0 writes x = 2:
  Core 0 sends invalidation to Core 1
  Core 0: M (exclusive again)
  Core 1: I (invalidated)

Core 1 reads x:
  Core 1 must fetch from Core 0 again
```

**MESIF Protocol** (Intel's variant):

Intel uses MESIF (added F = Forward). The "Forward" state allows one core to forward the line to other cores without going to main memory.

```
F (Forward):    This cache has the line, has not modified it, and can forward
                it to other caches (designated forwarder)
                Other caches still have S (shared)
```

This reduces memory traffic by forwarding between caches instead of fetching from main memory.

**Cost of Coherence Misses**:

```
Type                    Latency    Cycles
L1 hit (on same core)   4 ns       13
L3 hit                  40 ns      130
Coherence transfer      80 ns      250  (another core's cache)
Main memory             100 ns     300
```

A coherence miss (reading from another core's cache) costs ~250 cycles. This is why false sharing is so expensive.

### False Sharing: What It Is and How to Fix It

False sharing occurs when two threads write to different variables on the same cache line.

```c
struct {
    int counter0;  // 4 bytes
    int counter1;  // 4 bytes
    // Both fit in one 64-byte cache line
} counters;

// Thread 0:
void thread0() {
    for (int i = 0; i < 1000000; i++) {
        counters.counter0++;
    }
}

// Thread 1:
void thread1() {
    for (int i = 0; i < 1000000; i++) {
        counters.counter1++;
    }
}
```

Even though the threads write to different variables, they fight over the same cache line:

1. Thread 0 reads cache line (state = Exclusive)
2. Thread 0 writes counter0 (state = Modified)
3. Thread 1 reads cache line (invalidates Thread 0's cache)
4. Thread 1 writes counter1 (state = Modified)
5. Thread 0 reads cache line again (invalidates Thread 1's cache)
... (repeat 1000000 times)

Each increment causes:
- An invalidation from the other thread
- A fetch from the other thread's L1 or L3 cache
- A coherence miss penalty (~250 cycles)

Total time: 1000000 × 250 cycles = 250 million cycles ≈ 75 seconds (on a 3 GHz CPU).

**Solution: Padding**

```c
struct {
    int counter0;
    int padding0[15];  // Pad to 64-byte boundary
    int counter1;
    int padding1[15];
} counters __attribute__((aligned(64)));
```

Now counter0 and counter1 are on different cache lines. No false sharing.

Total time: 1000000 cycles ≈ 0.3 seconds (faster by 250x).

### Lock-Free Data Structures: Atomic Operations and CAS

When synchronization must be extremely fast, you use lock-free algorithms that rely on atomic operations.

**Atomic Load/Store**:

```c
atomic_int x;
x.store(42, memory_order_relaxed);  // Atomic store
int val = x.load(memory_order_relaxed);  // Atomic load
```

On x86, these compile to regular mov instructions (atomicity is guaranteed by hardware).

**Compare-And-Swap (CAS)**:

CAS atomically:
1. Reads the current value
2. Compares it to an expected value
3. If they match, stores a new value
4. Returns whether the operation succeeded

```c
atomic_int version(0);

// Atomic increment:
int old_val = version.load();
while (!version.compare_exchange_strong(old_val, old_val + 1)) {
    // CAS failed, retry with new old_val
}
```

On x86:
```asm
mov $0, %eax           # old_val
lock cmpxchg %ecx, (%rdi)  # Atomic compare-and-swap
                            # If [rdi] == eax, then [rdi] = ecx
                            # Sets flags if successful
jne retry              # Jump if not equal (CAS failed)
```

The `lock` prefix ensures the operation is atomic (acquires the cache line lock).

**Cost**:
- Successful CAS: ~10-20 cycles
- Failed CAS (with retry): ~20-50 cycles (depends on cache state)

**ABA Problem**:

CAS has a subtle bug: if another thread changes the value A → B → A, your CAS will succeed even though the data structure was modified.

```c
// ABA problem example:
struct Node {
    int value;
    struct Node *next;
};

atomic<Node*> head;

// Thread 0: Try to pop from stack
void thread0() {
    Node *old_head = head.load();                    // Read A
    Node *new_head = old_head->next;
    head.compare_exchange_strong(old_head, new_head); // Expect A, set to B
}

// Thread 1: Pop A, then push it back
void thread1() {
    Node *temp = head.load();  // Read A
    // ... do something ...
    head.store(temp);  // Push A back (now it's A again)
}

// Problem: head is now A again (same address)
// Even though the node was removed and re-added
// Thread 0's CAS succeeds but the node is stale
```

**Solution**: Use versioning or hazard pointers to detect ABA.

### Lock-Free Queue for Inference Engine

A practical lock-free queue for ML inference engine job scheduling:

```c
#include <stdatomic.h>

struct Job {
    void *data;
    int id;
};

struct LockFreeQueue {
    struct {
        struct Job job;
        long sequence;  // To handle ABA
    } *items;
    long capacity;

    atomic_long head;
    atomic_long tail;
};

bool enqueue(struct LockFreeQueue *q, struct Job job) {
    long tail = atomic_load_explicit(&q->tail, memory_order_acquire);
    long head = atomic_load_explicit(&q->head, memory_order_acquire);

    if ((tail + 1) % q->capacity == head) {
        return false;  // Queue full
    }

    q->items[tail % q->capacity].job = job;
    q->items[tail % q->capacity].sequence = tail;

    long old_tail = tail;
    while (!atomic_compare_exchange_weak_explicit(&q->tail,
                                                   &old_tail,
                                                   tail + 1,
                                                   memory_order_release,
                                                   memory_order_relaxed)) {
        tail = old_tail;
        if ((tail + 1) % q->capacity == head) {
            return false;
        }
    }

    return true;
}

bool dequeue(struct LockFreeQueue *q, struct Job *out) {
    long head = atomic_load_explicit(&q->head, memory_order_acquire);
    long tail = atomic_load_explicit(&q->tail, memory_order_acquire);

    if (head == tail) {
        return false;  // Queue empty
    }

    *out = q->items[head % q->capacity].job;

    long old_head = head;
    while (!atomic_compare_exchange_weak_explicit(&q->head,
                                                   &old_head,
                                                   head + 1,
                                                   memory_order_release,
                                                   memory_order_relaxed)) {
        head = old_head;
        tail = atomic_load_explicit(&q->tail, memory_order_acquire);
        if (head == tail) {
            return false;
        }
    }

    return true;
}
```

This queue is lock-free: enqueue and dequeue never block (they may retry on CAS failure, but don't wait for locks).

**Reference**: McKenney, Paul E. *Is Parallel Programming Hard, And If So, What Can You Do About It?* (2021).
  - Section 4: "Synchronization"
  - Section 5: "Applications"

---

## 2. MENTAL MODEL

### Memory Consistency Model as a Contract

```
┌─────────────────────────────────────────────────────────────┐
│ PROGRAMMER'S VIEW (What's guaranteed)                      │
│                                                             │
│ Sequential Consistency (SC):                               │
│ - All operations globally ordered                          │
│ - Easiest to reason about, but slow                       │
│                                                             │
│ x86 Total Store Order (TSO):                               │
│ - Load-Load, Load-Store, Store-Store ordered              │
│ - Store-Load reordering allowed                           │
│ - Faster than SC (no SFENCE/LFENCE needed)               │
│                                                             │
│ ARM/PowerPC Weak Consistency:                             │
│ - All orderings allowed by default                         │
│ - Explicit fences required for ordering                   │
│ - Fastest, hardest to reason about                        │
└─────────────────────────────────────────────────────────────┘
          ▲
          │ Memory model
          │ specifies which
          │ reorderings are
          │ allowed
          ▼
┌─────────────────────────────────────────────────────────────┐
│ CPU HARDWARE IMPLEMENTATION                               │
│                                                             │
│ Store Buffer:  Delays stores until cache is available     │
│                Allows store-load reordering               │
│                                                             │
│ Out-of-Order Engine: Reorders non-dependent instructions  │
│                                                             │
│ Cache Coherence Protocol (MESI/MESIF):                   │
│                Invalidations when other core writes        │
│                                                             │
│ Memory Fences:  Force ordering, synchronize caches        │
│                MFENCE: full barrier (40 cycles)           │
│                SFENCE: store barrier (10 cycles)          │
│                LFENCE: load barrier (10 cycles)           │
└─────────────────────────────────────────────────────────────┘
```

### Store Buffer and Reordering

```
Core 0                           Memory
┌──────────────┐
│ Store Buffer │
├──────────────┤
│ write x=1    │────────┐
│ write y=2    │        │
└──────────────┘        │
     ▲                  │
     │                  ▼
  Core 0 executes    Stores propagate
  x = 1; y = 2;      in order or out of
                     order to other cores

                    Time

t0: x = 1 added to store buffer
    (Core 1 may not see x=1 yet)

t1: y = 2 added to store buffer
    (Core 1 may see y=2 before x=1)

t2: x = 1 propagates to Core 1's cache
    (now Core 1 sees x=1)

t3: y = 2 propagates to Core 1's cache

POSSIBLE OBSERVATION BY CORE 1:
- t0-t1: neither value visible
- t1-t2: sees y=2, x=0 (store-load reordering!)
- t2-t3: sees y=2, x=1
- t3+:   sees both
```

### Cache Coherence with MESI Protocol

```
Initial state: x resides in main memory (all caches invalid)

┌─────────────────────────────────────────────────────┐
│ Core 0 loads x:                                     │
├─────────────────────────────────────────────────────┤
│ Before:  Cache0[x] = I   Cache1[x] = I            │
│ After:   Cache0[x] = E   Cache1[x] = I            │
│          (Exclusive: only this cache has it)      │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Core 0 writes x = 1:                                │
├─────────────────────────────────────────────────────┤
│ Before:  Cache0[x] = E   Cache1[x] = I            │
│ After:   Cache0[x] = M   Cache1[x] = I            │
│          (Modified: this cache owns the line)      │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Core 1 reads x:                                     │
├─────────────────────────────────────────────────────┤
│ Before:  Cache0[x] = M   Cache1[x] = I            │
│ After:   Cache0[x] = S   Cache1[x] = S            │
│          (Shared: both caches have it, read-only) │
│          (Core 0 must write back to Core 1's cache)│
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Core 0 writes x = 2:                                │
├─────────────────────────────────────────────────────┤
│ Before:  Cache0[x] = S   Cache1[x] = S            │
│ Invalidate Core 1:       Cache1[x] = I            │
│ After:   Cache0[x] = M   Cache1[x] = I            │
│          (Core 0 is sole owner again)              │
└─────────────────────────────────────────────────────┘
```

---

## 3. PERFORMANCE LENS

### Coherence Misses vs Cache Misses

**Cache Miss** (data not in any cache):
- Fetch from L3 or main memory
- Cost: 40 ns (L3 hit) to 100 ns (memory)
- Latency: 130-300 cycles

**Coherence Miss** (data in another core's cache):
- Fetch from other core's L1 or L3
- Cost: ~80 ns (another core's cache)
- Latency: ~250 cycles (worse than L3 miss due to coordination overhead)

**False Sharing**:
Each write causes invalidation + fetch from other core = coherence miss penalty on every operation.

Example: Two threads incrementing counters at the same address (false sharing):
```
Without false sharing: 1 cycle per increment (register arithmetic)
With false sharing:    250 cycles per increment (coherence miss)
Slowdown factor:       250x
```

### Lock Contention

When multiple threads contend for a lock:

```c
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void critical_section() {
    pthread_mutex_lock(&lock);      // Busy-wait or context switch
    // ... critical section (10 cycles) ...
    pthread_mutex_unlock(&lock);
}
```

**Lock-free**: CAS-based increment
```
Throughput: 1 billion increments/sec per core
Total (4 cores): 4 billion increments/sec
```

**Spin Lock**: Busy-wait
```
Throughput: 500 million increments/sec per core (worse due to coherence traffic)
Total (4 cores): 2 billion increments/sec (only 2x better due to contention)
```

**Mutex**: Context switch
```
Throughput: 100 million increments/sec per core (context switch overhead)
Total (4 cores): 400 million increments/sec
```

Lock-free is 10x faster than mutex when contention is high.

---

## 4. ANNOTATED CODE

### Example 1: Demonstrating TSO Reordering (Store-Load)

```c
// tso_reordering.c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>

#define ITERATIONS 1000000

atomic_int x = 0;
atomic_int y = 0;
int outcomes[4] = {0};  // [00, 01, 10, 11]

void *thread_write(void *arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        atomic_store_explicit(&x, 1, memory_order_relaxed);
        atomic_store_explicit(&y, 1, memory_order_relaxed);
    }
    return NULL;
}

void *thread_read(void *arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        int r1 = atomic_load_explicit(&y, memory_order_relaxed);
        int r2 = atomic_load_explicit(&x, memory_order_relaxed);

        // On x86, cannot observe r1=1, r2=0 (store-load reordering only)
        // Can observe: (r1=0, r2=0), (r1=0, r2=1), (r1=1, r2=1)
        // Cannot observe: (r1=1, r2=0) due to TSO

        int outcome = (r1 << 1) | r2;
        outcomes[outcome]++;
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;

    for (int trial = 0; trial < 100; trial++) {
        memset(outcomes, 0, sizeof(outcomes));

        pthread_create(&t1, NULL, thread_write, NULL);
        pthread_create(&t2, NULL, thread_read, NULL);

        pthread_join(t1, NULL);
        pthread_join(t2, NULL);

        printf("Trial %d: 00=%d, 01=%d, 10=%d, 11=%d\n",
               trial, outcomes[0], outcomes[1], outcomes[2], outcomes[3]);
    }

    return 0;
}
```

**Expected output on x86 (TSO)**:
```
00=xxxxx, 01=xxxxx, 10=xxxxx, 11=xxxxx
11 is the most common (both values seen)
01 is common (y not yet propagated to x's writer)
10 is common (x not yet propagated to y's writer)
00 is least common (both not propagated yet)

Notice: 10 (r1=1, r2=0) NEVER occurs on x86 (would violate store-store ordering)
```

On ARM (weaker model), outcome 10 would occur frequently.

---

### Example 2: False Sharing Benchmark

```c
// false_sharing.c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <time.h>

#define ITERATIONS 10000000

// Scenario 1: False sharing (BAD)
struct {
    atomic_long counter0;
    atomic_long counter1;
    // Both on same 64-byte cache line
} shared_bad;

// Scenario 2: Padding to avoid false sharing (GOOD)
struct {
    atomic_long counter0;
    char padding0[56];  // Pad to separate cache lines
    atomic_long counter1;
    char padding1[56];
} shared_good __attribute__((aligned(64)));

void *thread0_bad(void *arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        atomic_fetch_add_explicit(&shared_bad.counter0, 1, memory_order_relaxed);
    }
    return NULL;
}

void *thread1_bad(void *arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        atomic_fetch_add_explicit(&shared_bad.counter1, 1, memory_order_relaxed);
    }
    return NULL;
}

void *thread0_good(void *arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        atomic_fetch_add_explicit(&shared_good.counter0, 1, memory_order_relaxed);
    }
    return NULL;
}

void *thread1_good(void *arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        atomic_fetch_add_explicit(&shared_good.counter1, 1, memory_order_relaxed);
    }
    return NULL;
}

int main() {
    pthread_t t0, t1;

    printf("FALSE SHARING (BAD):\n");
    clock_t start = clock();

    pthread_create(&t0, NULL, thread0_bad, NULL);
    pthread_create(&t1, NULL, thread1_bad, NULL);
    pthread_join(t0, NULL);
    pthread_join(t1, NULL);

    clock_t end = clock();
    double time_bad = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.3f seconds\n", time_bad);
    printf("Final counters: %ld, %ld\n",
           atomic_load(&shared_bad.counter0),
           atomic_load(&shared_bad.counter1));

    // Reset
    atomic_store(&shared_bad.counter0, 0);
    atomic_store(&shared_bad.counter1, 0);

    printf("\nNO FALSE SHARING (GOOD):\n");
    start = clock();

    pthread_create(&t0, NULL, thread0_good, NULL);
    pthread_create(&t1, NULL, thread1_good, NULL);
    pthread_join(t0, NULL);
    pthread_join(t1, NULL);

    end = clock();
    double time_good = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.3f seconds\n", time_good);
    printf("Final counters: %ld, %ld\n",
           atomic_load(&shared_good.counter0),
           atomic_load(&shared_good.counter1));

    printf("\nSpeedup: %.1fx\n", time_bad / time_good);

    return 0;
}
```

**Compilation and execution**:
```bash
gcc -O3 -pthread false_sharing.c -o false_sharing
./false_sharing

# Expected output:
# FALSE SHARING (BAD):
# Time: 5.234 seconds
# Final counters: 10000000, 10000000
#
# NO FALSE SHARING (GOOD):
# Time: 0.342 seconds
# Final counters: 10000000, 10000000
#
# Speedup: 15.3x
```

Padding provides 10-20x speedup by eliminating false sharing coherence misses.

---

### Example 3: Lock-Free Counter

```c
// lock_free_counter.c
#include <stdatomic.h>
#include <stdio.h>

struct LockFreeCounter {
    atomic_long value;
};

void increment(struct LockFreeCounter *counter) {
    // Atomic increment without explicit loop
    atomic_fetch_add_explicit(&counter->value, 1, memory_order_relaxed);
}

// Alternative with CAS (shows the pattern):
void increment_cas(struct LockFreeCounter *counter) {
    long old = atomic_load_explicit(&counter->value, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(&counter->value,
                                                   &old,
                                                   old + 1,
                                                   memory_order_relaxed,
                                                   memory_order_relaxed)) {
        // CAS failed (another thread incremented), retry
        // old is updated by compare_exchange_weak to the actual value
    }
}

int main() {
    struct LockFreeCounter counter = {0};

    for (int i = 0; i < 1000000; i++) {
        increment(&counter);
    }

    printf("Counter: %ld\n", atomic_load(&counter.value));

    return 0;
}
```

**Code generation for increment on x86**:

```asm
increment:
    mov $1, %eax
    lock addq %eax, (%rdi)   # Atomic add
    ret
```

The `lock` prefix makes the add atomic. Cost: ~10 cycles without contention, ~50-100 cycles with contention.

---

## 5. EXPERT INSIGHT

### Non-Obvious Truths About Synchronization

**Truth 1: Acquire/Release is Often Enough, seq_cst is Often Overkill**

Programmers often use seq_cst (sequential consistency, full barriers) out of caution. But careful use of acquire/release is much faster:

```c
// seq_cst (full barriers, 40+ cycles each):
atomic_int ready(0);
data = 42;
ready.store(1, memory_order_seq_cst);  // MFENCE

// Acquire/release (10 cycles each):
atomic_int ready(0);
data = 42;
ready.store(1, memory_order_release);  // Fence for stores before
```

The acquire/release version is 4x faster and equally correct.

The senior engineer carefully reasons about which operations need which memory order. The junior engineer defaults to seq_cst everywhere.

**Truth 2: Lock-Free is Not Always Faster**

Lock-free data structures avoid blocking, but they have downsides:
- More complex (higher chance of bugs)
- CAS may retry under contention (busy-looping)
- Cache coherence traffic from CAS operations
- May have worse performance than a simple spinlock or mutex

Example:
```
Low contention (1 thread):  Lock-free = fastest
High contention (many threads): Spinlock may be faster (less coherence traffic)
Extreme contention:  Mutex may be faster (threads block instead of busy-wait)
```

**Truth 3: Memory Barriers Are About More Than Ordering**

A barrier does two things:
1. Prevents reordering (ordering guarantee)
2. Flushes store buffers (makes stores visible to other cores)

Without a barrier, a store might remain in the store buffer indefinitely, invisible to other cores. The barrier forces it out.

```c
int x = 42;
flag.store(1, memory_order_release);  // Barrier
// At this point, all stores before this line are visible to other cores
```

**Truth 4: NUMA and Synchronization Interact**

On a NUMA system, a lock may be biased towards one socket:

```c
struct {
    char padding0[64];
    pthread_mutex_t lock;  // Lives on socket 0
    char padding1[64];
} shared;

// Thread on socket 0: Fast lock access (local DRAM)
// Thread on socket 1: Slow lock access (remote DRAM, 2x latency)
```

For high-performance code on NUMA systems, use NUMA-aware locks or organize locks per-socket.

**Truth 5: Store Buffer and Out-of-Order Execution Compound**

Store buffer reordering + out-of-order execution can produce surprising results:

```c
// Thread 0:
void write_then_read() {
    x = 1;           // Goes to store buffer, not yet visible
    int y_val = y;   // May read y before x propagates
}

// Thread 1:
void write_y() {
    y = 2;
}
```

Thread 0 may see y=2 but x=0 on its own side (its store hasn't propagated yet). This is the store-load reordering allowed by TSO.

---

## 6. BENCHMARK / MEASUREMENT

### Tools for Synchronization Analysis

**Tool 1: perf for Lock Contention**

```bash
# Measure context switches (indicator of lock contention):
perf stat -e context-switches:u ./my_program

# Output:
#        1,234,567 context-switches:u
#              # High value = threads fighting over locks

# Measure futex syscalls (lock-based synchronization):
perf stat -e syscalls:sys_enter_futex ./my_program
```

High futex counts indicate:
- Threads blocking on mutexes
- Poor scalability (threads cannot run in parallel)

**Tool 2: perf for Cache Coherence**

```bash
# Measure LLC coherency misses:
perf record -e LLC-loads,LLC-stores,LLC-prefetches,LLC-prefetch-misses ./my_program
perf report

# Or specifically for cache coherence:
perf record -e cache-misses ./my_program
```

High cache miss rates on a multi-core program often indicate false sharing.

**Tool 3: Threadpoolc (custom tool)**

Write custom benchmarks to measure synchronization overhead:

```c
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <stdatomic.h>

atomic_int lock_free_counter = 0;

void *worker(void *arg) {
    for (int i = 0; i < 10000000; i++) {
        atomic_fetch_add(&lock_free_counter, 1);
    }
    return NULL;
}

int main() {
    pthread_t threads[4];

    clock_t start = clock();

    for (int i = 0; i < 4; i++) {
        pthread_create(&threads[i], NULL, worker, NULL);
    }

    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_t end = clock();

    printf("Time for 40M increments (4 threads): %.3f seconds\n",
           (double)(end - start) / CLOCKS_PER_SEC);
    printf("Final counter: %d\n", atomic_load(&lock_free_counter));

    return 0;
}
```

---

## 7. ML SYSTEMS RELEVANCE

### Synchronization in Inference Engines

**Principle 1: Token Generation Queue (Lock-Free Recommended)**

In multi-threaded inference, tokens are generated in a queue and consumed by workers:

```c
struct TokenQueue {
    atomic_long head;     // Readers increment this
    atomic_long tail;     // Writers increment this
    struct Token *items;
};

bool enqueue_token(struct TokenQueue *q, struct Token token) {
    // Lock-free: no mutex, uses CAS
    long old_tail = atomic_load_explicit(&q->tail, memory_order_acquire);
    if ((old_tail + 1) % MAX_TOKENS == atomic_load_explicit(&q->head, memory_order_acquire)) {
        return false;  // Queue full
    }
    q->items[old_tail % MAX_TOKENS] = token;

    // Atomically increment tail
    while (!atomic_compare_exchange_weak_explicit(&q->tail, &old_tail, old_tail + 1,
                                                  memory_order_release, memory_order_relaxed)) {
        if ((old_tail + 1) % MAX_TOKENS == atomic_load_explicit(&q->head, memory_order_acquire)) {
            return false;
        }
    }
    return true;
}
```

This is better than mutex because:
- No blocking (enqueue never waits)
- No context switches
- 10-100x faster for high throughput

**Principle 2: KV Cache Sharing Across Threads**

When multiple threads access the same KV cache (for different batches), false sharing becomes critical:

```c
struct KVCacheEntry {
    float *keys;    // May be on same cache line as values from adjacent batch
    float *values;
    atomic_int refcount;  // Reference counting
};
```

Each thread writing to refcount causes invalidations on other threads' caches. Solution: pad the structure or use per-thread reference counts.

**Principle 3: Memory Ordering for Model Weights**

When loading model weights in one thread and reading in others:

```c
struct Model {
    float *weights;
    atomic_int ready;  // Flag indicating weights are loaded
};

// Thread 1 (loader):
model.weights = load_weights();  // Initialize weights
atomic_store_explicit(&model.ready, 1, memory_order_release);

// Thread 2 (reader):
if (atomic_load_explicit(&model.ready, memory_order_acquire)) {
    float w = model.weights[0];  // Safe to read, guaranteed to see initialized weights
}
```

The release/acquire pair ensures that the weights load is visible before the ready flag.

**Principle 4: Avoiding Lock Contention in Batching**

When a single lock protects the batch queue, contention can destroy performance:

```c
// BAD: Single lock for all threads
pthread_mutex_t batch_lock = PTHREAD_MUTEX_INITIALIZER;

void *worker(void *arg) {
    for (int i = 0; i < 1000; i++) {
        pthread_mutex_lock(&batch_lock);
        Batch b = get_next_batch();  // Critical section: 10 cycles
        pthread_mutex_unlock(&batch_lock);

        process_batch(b);  // Non-critical: 1000 cycles
    }
}
```

The lock is held for only 10 cycles, but with 8 threads, the lock is heavily contended. Solution: use lock-free queue or reduce lock granularity.

```c
// GOOD: Lock-free queue
bool dequeue_batch(struct BatchQueue *q, Batch *b) {
    // No lock, uses CAS
    // Can handle 1000x higher throughput
}
```

---

## 8. PhD QUALIFIER QUESTIONS

1. **Memory Model and Reordering**:
   On x86 (TSO), explain why store-load reordering is allowed but store-store reordering is not. Write a scenario where you can observe store-load reordering, and show how adding an MFENCE prevents it. What is the performance cost?

2. **Cache Coherence Protocol**:
   Describe the MESI state transitions that occur when two cores write to the same cache line repeatedly. Calculate the total latency of each transition. How does MESIF improve over MESI?

3. **False Sharing and Memory Layout**:
   You have an array of 1000 lock variables. How many cache lines does this occupy? If 4 threads each acquire different locks concurrently, calculate the false sharing penalty and propose a solution.

4. **Lock-Free Data Structures and ABA**:
   Explain the ABA problem in lock-free algorithms. Show a specific scenario where a lock-free stack would fail due to ABA. How would you solve it using versioning or hazard pointers?

5. **Acquire/Release vs Seq_Cst**:
   Write a synchronization pattern (producer-consumer) using acquire/release semantics. Show which memory operations can be reordered even with acquire/release. Compare performance to seq_cst.

---

## 9. READING LIST

**Essential Textbooks**:
- McKenney, Paul E. *Is Parallel Programming Hard, And If So, What Can You Do About It?* (2021).
  - **Chapter 4**: "Synchronization" — memory fences, atomic operations, synchronization primitives
  - **Chapter 5**: "Applications" — practical concurrent programming patterns
  - **Chapter 6**: "Data Structures" — lock-free queues, stacks, and trees

- Hennessy, John L., and David A. Patterson. *Computer Architecture: A Quantitative Approach (6th Edition)*. Morgan Kaufmann, 2017.
  - **Chapter 5**: "Memory Hierarchies" — cache coherence protocols (MESI, MOESI, MESIF)
  - **Chapter 9**: "Interconnects" — on-chip interconnects, cache coherence traffic

**Formal Specifications**:
- AMD Inc. *System V Application Binary Interface AMD64 Architecture Processor Supplement (Version 1.0)* (2018).
  - **Section 3.3**: "Memory Model and Synchronization"
  - Defines x86 TSO guarantees

- Boehm, Hans-J., and Sarita V. Adve. "Foundations of the C++ Concurrency Memory Model." PLDI (2008).
  - Formal semantics of C++11 memory model
  - Memory ordering specifiers and their semantics

**Lock-Free Algorithms**:
- Harris, Timothy L. "A Pragmatic Implementation of Non-Blocking Linked-Lists." DISC (2001).
  - Detailed lock-free linked list implementation

- Treiber, R. Kent. "Systems Programming: Coping with Parallelism." IBM Almaden Research Center (1986).
  - Classic lock-free stack algorithm

**Tools and Debugging**:
- Linux Kernel Documentation: "linux/Documentation/atomic_t.txt"
  - Kernel atomics and memory barriers

- GCC/Clang Documentation: "Memory Model"
  - Compiler support for C++11 memory model

**Additional Resources**:
- Intel Corporation. *Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 3A*.
  - **Section 8.2**: "Memory Ordering"
  - **Section 8.3**: "Serializing Instructions" (including MFENCE, SFENCE, LFENCE)

- perf tool documentation: https://perf.wiki.kernel.org/ (lock contention analysis)

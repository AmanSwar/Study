# MODULE 2: Memory Hierarchy: The Single Most Important Topic

## 1. CONCEPTUAL FOUNDATION

### Why Memory is the Bottleneck in 99% of Modern Workloads

The single most important fact about modern computing:

**The CPU executes instructions 100-1000x faster than it can fetch data from main memory.**

- CPU cycle time: ~0.3 ns (at 3.3 GHz)
- L1 cache hit: ~4 ns (13 cycles)
- L2 cache hit: ~12 ns (40 cycles)
- L3 cache hit: ~40 ns (130 cycles)
- Main memory (DRAM) hit: ~100 ns (300 cycles)
- Disk access: ~10 ms (30 million cycles)

A single cache miss can stall the CPU for 300+ cycles—time during which all parallelism is lost. This asymmetry means:

**Your algorithm's complexity is irrelevant if it misses the cache. An O(n²) algorithm with perfect cache locality runs faster than an O(n log n) algorithm with poor locality.**

(Reference: Mutlu, Onur. "Computer Architecture Lectures" — Lecture 5: "Memory Hierarchy")

### SRAM vs DRAM: The Fundamental Physics

The memory hierarchy exists because of fundamental physics. SRAM (Static RAM, used for cache) and DRAM (Dynamic RAM, used for main memory) have different speed-size-cost tradeoffs.

**SRAM** (Static RAM):
```
Bit cell structure:
  ┌─────────────────────┐
  │  Latch (6 transistors) │  OR 2x transistors (eDRAM)
  │  Q and Q' outputs   │
  └─────────────────────┘
  Access time: ~4 ns (for modern L1 cache)
  Density: ~0.1-1 Mb per mm²
  Power: ~1 mW per MB (6 transistors × leakage)
  Cost: ~$1000 per MB
  Use: L1, L2, L3 caches (registers are also SRAM)
```

A latch stores one bit in a cross-coupled pair of inverters. Once written, the bit is stable—no periodic refresh needed. This makes access very fast.

**DRAM** (Dynamic RAM):
```
Bit cell structure:
  ┌──────────────┐
  │  1 transistor │
  │  1 capacitor  │
  │  (charge storage) │
  └──────────────┘
  Access time: ~100 ns (for typical DDR4)
  Density: ~10-100 Mb per mm²
  Power: ~1 mW per GB (refresh + access)
  Cost: ~$10 per GB
  Use: Main memory, secondary caches in some designs
```

A DRAM bit is stored as charge on a capacitor. The capacitor slowly leaks current, so each row must be refreshed (re-read and re-written) every ~64 ms. This refresh overhead is why:

1. **DRAM is slower**: Refresh cycles steal memory bandwidth
2. **DRAM is cheaper**: Only 1 transistor + 1 capacitor per bit (vs 6 transistors for SRAM latch)
3. **DRAM scales**: 100x higher density means 100x more memory for same die area

**Why Cache Hierarchy Exists**:

Since main memory is 100x slower than cache, a 100 MB cache would cost $100,000. Instead, we use a hierarchy:

```
Level    Size      Hit Latency  Cost per MB  Access time
L1i      32 KB     4 ns         $1000        ~0 (register)
L1d      32 KB     4 ns         $1000        ~0 (register)
L2       256 KB    12 ns        $500         ~40 cycles
L3       8 MB      40 ns        $50          ~130 cycles
RAM      16 GB     100 ns       $5           ~300 cycles
```

The hierarchy works because of **locality**:
- Temporal locality: Recently accessed data is likely to be accessed again soon
- Spatial locality: Data near recently accessed data is likely to be accessed soon

If your code has good locality, most accesses hit the cache. If not, you miss constantly.

### Cache Anatomy: How Addresses Map to Cache Sets and Lines

A modern CPU cache is not a simple "memory with fast lookup." It's a complex data structure with specific addressing. Understanding the exact mapping is critical for diagnosing cache performance issues.

**Intel Core i9-12900K L1 Data Cache (actual specifications)**:
- Size: 32 KB per core
- Line size: 64 bytes
- Associativity: 8-way
- Sets: 32 KB / (8 ways × 64 bytes) = 64 sets

**Address-to-Cache Mapping**:

A 64-bit virtual address is divided into fields:

```
Virtual address (64 bits):
┌────────────────────────┬──────────────┬─────────────┐
│     Tag (49 bits)      │  Index (6 bits) │  Offset (6 bits) │
└────────────────────────┴──────────────┴─────────────┘
 [63:13]                   [12:6]           [5:0]

Cache lookup:
  1. Extract index [12:6] (6 bits) → which of 64 sets
  2. Extract tag [63:13] (49 bits) → which line in the set
  3. Extract offset [5:0] (6 bits) → byte within 64-byte line

Hit condition: Tag matches one of the 8 tags in the set
```

**Example: Address 0x12345678**

```
0x12345678 in binary:
00010010001101000101011001111000
       [Tag:27 bits][Index:6 bits][Offset:6 bits]

Tag:    0x24da (bits [15:6])  → which line in the set
Index:  0x1a (bits [5:0])     → set 26
Offset: 0x38 (bits [5:0])     → byte 56 within the line

Cache lookup:
  1. Go to set 26
  2. Compare tag 0x24da with all 8 tags in set 26
  3. If match found, return data[0x38] from that line
  4. If no match, miss (fetch from L2 cache)
```

**L2 Cache** (Intel Skylake): 256 KB, 8-way, 64-byte lines
- Sets: 256 KB / (8 × 64) = 512 sets
- Index bits: log2(512) = 9 bits [14:6]
- Tag bits: 64 - 9 - 6 = 49 bits

**L3 Cache** (Intel Skylake): 8 MB, 16-way, 64-byte lines
- Sets: 8 MB / (16 × 64) = 8192 sets
- Index bits: log2(8192) = 13 bits [18:6]
- Tag bits: 64 - 13 - 6 = 45 bits

**Key Insight**: The index is derived from virtual address bits [12:6], which are the same in virtual and physical address (untranslated). This is why caches can use virtual addressing for the index: the index is page-aligned and doesn't depend on the page's physical location.

### Cache Replacement Policies: LRU, Pseudo-LRU, PLRU

When a cache line must be evicted (the set is full and a new line arrives), which line is removed? This decision is made by the replacement policy.

**LRU (Least Recently Used)**:

Track the access order of all lines in a set. Evict the one accessed longest ago.

```
Set with 8 ways, recent accesses: [A, B, C, D, E, F, G, H]
Most recent (MRU): A (accessed just now)
Least recent (LRU): H (accessed longest ago)

If a new line I arrives:
  - Evict H
  - Order becomes: [A, B, C, D, E, F, G, I]
  - A is still MRU, G is now LRU
```

**Cost**: Requires tracking order of all 8 lines (3 bits per line to encode position). 8-way LRU needs 24 bits of metadata per set. For 64 sets, that's 192 bits = 24 bytes of overhead just for the LRU tracking.

**Pseudo-LRU (PLRU) / Tree-based Replacement**:

Instead of tracking exact order, use a binary tree to track which half of the set was used most recently.

```
8-way set, indexed 0-7:

        Tree node 0
       /          \
    Node1          Node2
    / \             / \
   0   1   2   3   4   5   6   7
```

Each node is 1 bit: 0 = left subtree was used more recently, 1 = right.

Access line 3:
- Path: 0 (right) → 2 (left)
- Update: bit0=1 (right subtree used), bit2=0 (left subtree used)

Next eviction evicts the line on the least-recently-used path (left-left or similar, depending on bit values).

**Cost**: 7 bits of metadata per set (not 24). Saves ~3x overhead vs true LRU.

**Accuracy**: PLRU is not true LRU (can make suboptimal eviction decisions), but closely approximates LRU behavior while reducing hardware cost.

**Modern CPUs**: Intel uses PLRU. AMD uses a variety (some true LRU, some PLRU).

**Performance Impact**:

LRU vs random eviction on a benchmark with poor locality:
- LRU: 15% cache hit rate
- Random: 12% cache hit rate

The difference is usually small because the working set fits in cache anyway. PLRU vs LRU: typically < 2% difference.

### Inclusive vs Exclusive vs NINE Cache Hierarchies

A key question: If data is in L1 cache, must it also be in L2?

**Inclusive Cache Hierarchy** (AMD, Intel pre-Broadwell):

Every line in L1 must also be in L2, and every line in L2 must be in L3.

```
Memory hierarchy:
L1:   ┌─────────────────────────┐
      │ [A] [B] [C] [D] [E] ...  │
      └─────────────────────────┘
        (all lines also in L2)
          ↓
L2:   ┌───────────────────────────────────────┐
      │ [A] [B] [C] [D] [E] ... [F] [G] ...   │
      └───────────────────────────────────────┘
        (all lines also in L3)
          ↓
L3:   ┌────────────────────────────────────────────────────────┐
      │ [A] [B] [C] [D] [E] ... [F] [G] ... [X] [Y] [Z] ...   │
      └────────────────────────────────────────────────────────┘
```

**Advantage**: Invalidation is simple. If L1 line [A] is invalidated, you know it's not in L2 either (since every line in L1 must also be in L2).

**Disadvantage**: Wastes space. Line [B] occupies 64 bytes in L1 AND 64 bytes in L2, for a total of 128 bytes for the same data.

**Exclusive Cache Hierarchy** (some AMD, newer Intel):

Lines in L1 are NOT in L2. When evicted from L1, they go to L2. When evicted from L2, they go to L3.

```
L1:   ┌──────────────────┐
      │ [A] [B] [C] [D]  │  (not in L2)
      └──────────────────┘
L2:   ┌─────────────────────────────────────┐
      │ [E] [F] [G] [H] ... (not in L3)    │
      └─────────────────────────────────────┘
L3:   ┌────────────────────────────────────────────────────────┐
      │ [I] [J] [K] ... (rest of memory)                       │
      └────────────────────────────────────────────────────────┘
```

**Advantage**: More efficient space. Larger total capacity (L1 + L2 + L3 = 32 + 256 + 8192 = 8480 KB).

**Disadvantage**: Eviction is costly. Moving data from L1 → L2 requires a write-back.

**NINE (Non-Inclusive Non-Exclusive) Cache Hierarchy** (Intel Skylake+):

Lines in L1 may or may not be in L2. When L1 line is evicted, it may stay in L2, or not—depends on L2 pressure.

```
L1:   ┌──────────────────┐
      │ [A] [B] [C] [D]  │  (may or may not be in L2)
      └──────────────────┘

If [A] is in L2: Data is duplicated (8 KB L1 + 8 KB L2 = 16 KB)
If [A] is NOT in L2: Data is only in L1 (efficient)

When [A] is evicted from L1:
  If L2 is not full: [A] is promoted to L2 (exclusive behavior)
  If L2 is full: [A] is simply discarded (inclusive behavior)
```

**Modern Design**: Most processors use NINE because it combines advantages of both:
- When the working set is small, lines are duplicated (better for coherence)
- When the working set is large, lines are exclusive (better for capacity)

### Write Policies: Write-Back vs Write-Through

When you write to a cache line, what happens?

**Write-Through**:

Write is committed to both cache and main memory:

```c
cache[addr] = value;
memory[addr] = value;  // Immediate write-back
```

**Cost**: Every write causes a main memory write, which is slow (100 ns per line).

**Benefit**: Cache and memory are always in sync. No write-back on eviction.

**Used in**: Older systems, embedded systems, I/O caches (where coherence is critical).

**Write-Back** (Write-Allocate):

Write is committed to cache only. Main memory is updated when the line is evicted.

```c
cache[addr] = value;  // "Dirty" the cache line
memory[addr] = value;  // Deferred until line eviction
```

**Cost**: On eviction, must write 64 bytes back to memory. But if the line is written multiple times before eviction, amortizes the cost.

**Benefit**: Multiple writes to same line are batched. Reduces memory traffic.

**Used in**: Modern CPUs (L1, L2, L3 all use write-back).

**Write Allocate** vs **No-Write Allocate**:

When a write misses the cache, should the missed line be fetched into cache?

**Write-Allocate**:
- Fetch the missed line into cache (as if it were a read)
- Write to the cache line
- The line becomes "dirty" and is written back on eviction

**No-Write-Allocate**:
- Write directly to main memory
- Do NOT fetch the line into cache

**Trade-off**:
- Write-allocate: Good if writes are followed by reads (temporal locality)
- No-write-allocate: Good if writes are scattered and never read again

**Modern CPUs**: Use write-allocate for all caches (simpler to implement, more common pattern).

### Cache Misses Taxonomy: The 4 C's

Every cache miss falls into one of four categories. Diagnosing which type of miss you're experiencing is critical for optimization.

**Compulsory Miss** (also called "cold miss"):

The first access to a line always misses (the line is not in cache yet).

```c
int x[1000];
for (int i = 0; i < 1000; i++) {
    process(x[i]);  // First iteration: x[0] misses (compulsory)
}
```

**Characteristic**: No way to avoid (without prefetching). The number of compulsory misses = number of unique cache lines accessed.

**Capacity Miss**:

The working set is too large for the cache.

```c
int x[100 * 1024 * 1024];  // 400 MB array
for (int i = 0; i < 1000000; i++) {
    process(x[i]);  // L1 cache is only 32 KB
                    // So we miss constantly (capacity miss)
}
```

**Characteristic**: If the cache were larger, the miss would not occur.

**Conflict Miss**:

Two different addresses map to the same cache set, causing unnecessary evictions.

```c
// L1 cache: 64 sets, 64-byte lines
// Addresses 0x1000 and 0x5000 map to same set (bits [12:6] are the same)

int *a = (int *)0x1000;  // Set 0
int *b = (int *)0x5000;  // Also set 0 (conflict!)

for (int i = 0; i < 1000; i++) {
    process(a[i]);  // Hits L1
    process(b[i]);  // Misses L1 (must evict a[i])
    process(a[i]);  // Misses L1 again (a[i] was evicted)
}
```

**Characteristic**: Cache is not fully utilized. Can be reduced by:
- Better memory layout (padding to avoid conflicts)
- Larger cache (more sets)
- Increased associativity (more ways per set)

**Coherence Miss**:

Another CPU core modified the line, requiring invalidation.

```c
// Core 0:
for (int i = 0; i < 1000; i++) {
    x[i] = i;  // Write, line is in Core 0's L1
}

// Core 1:
for (int i = 0; i < 1000; i++) {
    y[i] = x[i];  // Read. But Core 0 modified x[i]
               // so Core 1's cached copy is invalid (coherence miss)
}
```

**Characteristic**: Only occurs in multi-core systems. Caused by cache coherence protocol invalidations.

**Measuring the 4 C's with perf**:

```bash
perf stat -e cache-references,cache-misses ./my_program
# Gives total miss rate

perf record -e LLC-loads,LLC-load-misses ./my_program
# LLC (Last-Level Cache) = L3 on most systems

perf report -n  # Shows hot spots
```

More detailed analysis requires custom events (not portable across CPUs).

### Prefetching: Hardware and Software

A prefetch is a speculative memory fetch. The processor guesses that a line will be needed soon and fetches it before the program explicitly requests it.

**Hardware Prefetching**:

The CPU has built-in logic to detect access patterns and prefetch automatically.

**Stream Prefetcher**:
- Detects sequential access pattern (line 0, line 1, line 2, ...)
- Prefetches the next N lines in the stream
- Very effective for linear scans (sorting, matrix transpose, ML tensor operations)

Example:
```c
for (int i = 0; i < 1000000; i++) {
    process(data[i]);  // CPU detects: data[0], data[1], data[2] accesses
}
```

The stream prefetcher sees the pattern and prefetches data[3], data[4], etc. into cache before the processor requests them.

**Stride Prefetcher**:
- Detects constant-stride access pattern (line 0, line 2, line 4, ...)
- Example: row-major matrix access
  ```c
  int matrix[1000][1000];
  for (int i = 0; i < 1000; i += 2) {
      process(matrix[i][0]);  // Stride of 2 rows
  }
  ```

**Software Prefetching**:

Explicit prefetch instructions inserted by the compiler or programmer:

```c
#include <xmmintrin.h>

void prefetch_example(int *data, int len) {
    for (int i = 0; i < len; i++) {
        _mm_prefetch(&data[i + 8], _MM_HINT_T0);  // Prefetch data[i+8] into L1
        process(data[i]);
    }
}
```

Prefetch instructions are non-binding hints. The CPU may or may not honor them, depending on memory bandwidth availability.

**Benefit of Software Prefetch**:
- Hides memory latency: While processing data[i], data[i+8] is being fetched
- Improves memory-bound code by 20-40%

**Cost**:
- If the prefetch is incorrect, wastes memory bandwidth
- Prefetching too far ahead may evict needed lines

**Reference**: Mutlu, Onur. "Computer Architecture Lectures" — Lecture 9: "Prefetching"

### TLB: Translation Lookaside Buffer and Huge Pages

Virtual addresses must be translated to physical addresses. The TLB caches these translations.

**TLB Structure**:

A TLB entry maps virtual page number → physical page number.

```
Virtual address:
┌──────────────────────────────────┬─────────────┐
│   Virtual page number (39 bits)  │ Offset (12) │
│   [63:12]                        │ [11:0]      │
└──────────────────────────────────┴─────────────┘
         ↓
    TLB lookup (translation)
         ↓
┌──────────────────────────────────┬─────────────┐
│   Physical page number (40 bits) │ Offset (12) │
│   [51:12]                        │ [11:0]      │
└──────────────────────────────────┴─────────────┘
         ↓
    Physical address
```

**TLB Miss Penalty**:

If the TLB doesn't have the translation, the CPU must perform a page walk (follow the page table hierarchy):

```
4-level page table (x86-64):
PML4 (level 4) [bits 63:39] → 512 entries
  ↓
PDPT (level 3) [bits 38:30] → 512 entries
  ↓
PD (level 2) [bits 29:21] → 512 entries
  ↓
PT (level 1) [bits 20:12] → 512 entries
  ↓
Page offset [bits 11:0] → 4096 bytes within page
```

Each level requires a memory fetch (~ 100 ns). A full 4-level page walk costs ~400 ns = 1000+ CPU cycles.

**TLB Size on Modern CPUs**:
- Intel Skylake: 64 entries (4 KB pages) + 32 entries (2 MB pages) + 4 entries (1 GB pages)
- With 4 KB pages, a TLB hit covers 64 × 4096 = 256 KB of address space
- Working set > 256 KB → TLB miss overhead

**Huge Pages**:

Instead of 4 KB pages, use 2 MB or 1 GB pages.

```
4 KB page:   256 KB of address space fits in TLB
2 MB page:   128 MB of address space fits in TLB
1 GB page:   4 GB of address space fits in TLB
```

**Enabling Huge Pages on Linux**:

```bash
# Check current settings:
grep -i huge /proc/meminfo

# Allocate 1000 2 MB huge pages:
echo 1000 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

# In C code:
int fd = open("/dev/hugepages/myfile", O_CREAT | O_RDWR, 0600);
char *region = mmap(NULL, 2 * 1024 * 1024,
                   PROT_READ | PROT_WRITE,
                   MAP_SHARED,
                   fd, 0);

// Or use transparent huge pages (automatic):
madvise(ptr, len, MADV_HUGEPAGE);
```

**Performance Impact**:
- Reduces TLB misses by 100-1000x
- Especially important for large working sets (> 256 MB)
- ML models often have multi-GB tensors, so huge pages are critical

### NUMA: Topology, Local vs Remote Access

On systems with multiple sockets, memory is not uniformly accessible. Each socket has local DRAM, and accessing remote DRAM is slower.

**2-Socket System Layout**:

```
Socket 0 (CPU cores 0-15)     Socket 1 (CPU cores 16-31)
┌─────────────────────┐       ┌─────────────────────┐
│ L3 cache (8 MB)     │       │ L3 cache (8 MB)     │
│ DRAM Controller     │       │ DRAM Controller     │
└──────────┬──────────┘       └──────────┬──────────┘
           │                            │
    ┌──────▼─────────┐        ┌─────────▼──────┐
    │ Local DRAM (64 GB)       │ Remote DRAM (64 GB)
    │ Latency: 60 ns           │ Latency: 120 ns
    │ Bandwidth: 100 GB/s      │ Bandwidth: 50 GB/s
    └──────────────────┘       └─────────────────┘
           ↑                          ↑
      QPI/UPI interconnect (Intel 12 GB/s)
```

Accessing local DRAM: ~60 ns, ~100 GB/s per socket.
Accessing remote DRAM: ~120 ns (2x slower), ~50 GB/s per socket.

**NUMA-Aware Memory Allocation**:

By default, the OS allocates memory on the socket of the core that requests it (first-touch allocation). However, you can control it:

```c
#include <numa.h>

// Allocate memory on specific socket:
void *ptr = numa_alloc_onnode(size, 0);  // Allocate on socket 0

// Pin thread to socket:
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(0, &cpuset);  // Core 0 is on socket 0
sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

// Check NUMA topology:
// numa_available() returns -1 if NUMA disabled
for (int i = 0; i < numa_num_nodes(); i++) {
    struct bitmask *mask = numa_allocate_nodemask();
    numa_node_to_cpus(i, mask);  // Which CPUs are on this node?
}
```

**Performance Impact for ML**:
- If your model weights are on socket 1 but your thread is running on socket 0, all memory accesses are remote (2x slower)
- For a 100 GB model, this is the difference between 1 second and 2 seconds inference time
- Critical to pin threads to the correct socket and allocate memory locally

### Memory Bandwidth and STREAM Benchmark

The theoretical peak memory bandwidth is:

```
Bandwidth = Channel count × Transfer rate × Transfer width
```

Example: DDR4-3200 (3200 MT/s) with 6 channels:

```
Bandwidth = 6 × 3200 MT/s × 8 bytes/transfer = 153.6 GB/s
```

However, real-world bandwidth is lower due to:
- Page row/column address delays
- Refresh cycles
- Protocol overhead

**STREAM Benchmark**:

Measures actual achievable bandwidth:

```c
for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];  // Copy: read a, read b, write c
}
```

Execution: Load 16 bytes (a[i] + b[i]), store 8 bytes (c[i]). Ratio 2:1 (read:write).

On modern CPUs with good prefetching, STREAM typically achieves 50-80% of theoretical peak.

Example: Intel Xeon Platinum 8390
- Theoretical peak: 307 GB/s
- STREAM Copy: 250 GB/s (81%)
- STREAM Scale: 250 GB/s (81%)
- STREAM Add: 280 GB/s (91%)
- STREAM Triad: 280 GB/s (91%)

**Measuring Bandwidth**:

```bash
# Clone STREAM benchmark:
git clone https://github.com/jeffhammond/STREAM.git
cd STREAM
gcc -O3 -march=native -fopenmp stream.c -o stream

# Run:
./stream

# Output:
# Function    Best Rate MB/s  Avg time     Min time     Max time
# Copy:            250000.0     0.000050     0.000048     0.000052
# Scale:           250000.0     0.000050     0.000048     0.000052
# Add:             280000.0     0.000071     0.000069     0.000073
# Triad:           280000.0     0.000095     0.000093     0.000097
```

**Impact on ML Inference**:
- A model with 100 GB weights processed at 280 GB/s takes 360 ms
- If you have poor cache behavior and only achieve 50 GB/s, it takes 2 seconds
- Memory bandwidth is often the bottleneck for batch-size-1 inference

### Virtual Memory Deep Dive: Page Tables and Page Faults

Virtual memory allows each process to have a private 48-bit address space (0 to 2^48) on x86-64. This is achieved through page tables and a memory management unit (MMU).

**4-Level Page Table Structure**:

```
Virtual address (48 bits used):
┌──────┬──────┬──────┬──────┬──────────┐
│L4 (9)│L3 (9)│L2 (9)│L1 (9)│Offset(12)│
└──────┴──────┴──────┴──────┴──────────┘

L4 (PML4 = Page Map Level 4, bits [47:39]):
  ├─ 512 entries, each 8 bytes
  ├─ Each entry points to a L3 table (PDPT)
  └─ L4 table size: 512 × 8 = 4096 bytes (1 page)

L3 (PDPT = Page-Directory-Pointer Table, bits [38:30]):
  ├─ 512 entries, each 8 bytes
  ├─ Each entry points to a L2 table (PD)
  └─ 512 L3 tables × 4096 bytes = 2 MB

L2 (PD = Page Directory, bits [29:21]):
  ├─ 512 entries, each 8 bytes
  ├─ Each entry points to a L1 table (PT)
  └─ 512 × 512 L2 tables = 1 GB total

L1 (PT = Page Table, bits [20:12]):
  ├─ 512 entries, each 8 bytes
  ├─ Each entry points to a physical page (4 KB)
  └─ 512 × 512 × 512 L1 tables = 512 GB total

Offset (bits [11:0]):
  └─ Byte within 4 KB page (0-4095)
```

**Example: Translating Virtual Address 0x7ffff7800000**

```
Binary: 0000111111111111111111111000000000000000000000000000

Breakdown:
L4 index [47:39]: 000001 (1)  → Entry 1 of PML4 (kernel PML4 entry)
L3 index [38:30]: 111111 (511) → Entry 511 of PDPT
L2 index [29:21]: 111111 (511) → Entry 511 of PD
L1 index [20:12]: 100000 (32) → Entry 32 of PT
Offset [11:0]:   000000000000 (0)

Translation:
1. Load PML4 entry 1: physical address of PDPT
2. Load PDPT entry 511: physical address of PD
3. Load PD entry 511: physical address of PT
4. Load PT entry 32: physical page number
5. Add offset 0: final physical address
```

**Page Fault Handling**:

When a virtual address is not mapped (not in page table), the MMU raises a page fault exception.

```c
int *p = malloc(1000 * sizeof(int));  // Virtual address allocated
*p = 42;  // First write: page fault
          // Kernel handler:
          //   1. Allocate physical page
          //   2. Update page table entry
          //   3. Resume execution
```

**Page Fault Types**:
- Minor page fault: Page needs to be populated from page cache (file system cache)
- Major page fault: Page needs to be read from disk

Example:
```bash
strace -c ./my_program
# Output:
# Major page faults: 10 (slow, involves disk)
# Minor page faults: 1000 (fast, in-memory)

# If major page faults are high, it indicates:
# - Working set doesn't fit in physical memory
# - OS is paging to disk
# - Application will be very slow
```

**mmap() for Memory Management**:

`mmap()` allows you to map files or anonymous regions into virtual address space:

```c
// Allocate 1 GB of virtual address space without backing physical pages:
char *ptr = mmap(NULL, 1UL << 30,
                 PROT_READ | PROT_WRITE,
                 MAP_ANONYMOUS | MAP_PRIVATE,
                 -1, 0);

// Physical pages are allocated on demand (when first written to)
// Benefits:
//   - Fast allocation (no memset needed)
//   - Can allocate larger than physical RAM
//   - Kernel can reclaim pages if memory pressure increases
```

---

## 2. MENTAL MODEL

### Cache Hierarchy Visualization

```
┌────────────────────────────────────────────────────┐
│ PROCESS VIRTUAL ADDRESS SPACE                       │
│ Logical addresses used by program                  │
│ (processes don't know physical address)             │
└──────────────┬─────────────────────────────────────┘
               │ MMU (Memory Management Unit)
               │ Translation: virtual → physical
               ▼
┌────────────────────────────────────────────────────┐
│ PHYSICAL ADDRESS SPACE                              │
│ Physical RAM and I/O memory-mapped regions         │
└──────────────┬─────────────────────────────────────┘
               │
        ┌──────┴──────┬─────────────┬──────────────┐
        ▼             ▼             ▼              ▼
    ┌─────────┐  ┌─────────┐  ┌─────────┐   ┌──────────┐
    │  Core0  │  │  Core1  │  │  Core2  │   │  Memory  │
    │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │   │ Controller
    │ │ L1i │ │  │ │ L1i │ │  │ │ L1i │ │   │ (DRAM)   │
    │ │ 32KB│ │  │ │ 32KB│ │  │ │ 32KB│ │   │          │
    │ ├─────┤ │  │ ├─────┤ │  │ ├─────┤ │   │          │
    │ │ L1d │ │  │ │ L1d │ │  │ │ L1d │ │   │          │
    │ │ 32KB│ │  │ │ 32KB│ │  │ │ 32KB│ │   │          │
    │ ├─────┤ │  │ ├─────┤ │  │ ├─────┤ │   │          │
    │ │ L2  │ │  │ │ L2  │ │  │ │ L2  │ │   │          │
    │ │256KB│ │  │ │256KB│ │  │ │256KB│ │   │          │
    │ └──┬──┘ │  │ └──┬──┘ │  │ └──┬──┘ │   │          │
    │    │    │  │    │    │  │    │    │   │          │
    │    └──┬─┘  │    └──┬─┘  │    └──┬─┘   │          │
    │       │    │       │    │       │     │          │
    │       └────┼───────┼────┴───────┤     │          │
    │            │       │            │     │          │
    └─────┬──────┘       │            │     │          │
          │              │            │     │          │
          ▼              ▼            │     │          │
        ┌─────────────────────────────┤     │          │
        │  Shared L3 Cache             │     │          │
        │  8-20 MB (all cores)         │     │          │
        └─────────────┬────────────────┘     │          │
                      │                      │          │
                      └──────────┬───────────┘          │
                                 │                      │
                                 ▼                      │
                            ┌──────────────┐            │
                            │  DRAM (RAM)  │◄───────────┘
                            │ 16-256 GB    │
                            │ Latency: 100 │
                            │ ns Bandwidth:│
                            │ 100 GB/s     │
                            └──────────────┘
```

### The Memory Latency vs Bandwidth Tradeoff

```
LATENCY (time to first byte):
┌─────────────────────────────────────────────┐
│ Registers         0.3 ns (1 cycle @ 3 GHz)  │
│ L1 Cache          4 ns (13 cycles)          │
│ L2 Cache         12 ns (40 cycles)          │
│ L3 Cache         40 ns (130 cycles)         │
│ DRAM            100 ns (300 cycles)         │
│ SSD              100 μs (300,000 cycles)    │
└─────────────────────────────────────────────┘

BANDWIDTH (bytes per second):
┌──────────────────────────────────────────────┐
│ L1 Cache          ~2 TB/s (per core)         │
│ L2 Cache          ~700 GB/s (per core)       │
│ L3 Cache          ~200 GB/s (per core)       │
│ DRAM              ~100 GB/s (entire system)  │
│ SSD               ~3 GB/s (sequential)       │
└──────────────────────────────────────────────┘
```

Key insight: **Latency is dominated by the furthest cache level you hit. Bandwidth is the bottleneck when processing large datasets.**

### TLB and Page Table Translation

```
VIRTUAL ADDRESS TRANSLATION:

Virtual Address: 0x00007ffff7800000
                 ┌─ L4 index: 1   (PML4)
                 ├─ L3 index: 511 (PDPT)
                 ├─ L2 index: 511 (PD)
                 ├─ L1 index: 32  (PT)
                 └─ Offset: 0

TLB (Translation Lookaside Buffer):
┌──────────────────────┐
│ VPN [47:12]  │ PPN   │  (VPN = Virtual Page Number)
├──────────────────────┤  (PPN = Physical Page Number)
│ 0x7ffff7800  │ 0x123 │  ─ Direct translation (TLB hit)
│ 0x7ffff7900  │ 0x124 │    ~4 ns latency
│ ...          │ ...   │
└──────────────────────┘
   64 entries (4 KB pages)

If TLB miss (address not in TLB):
  Must perform page walk:
    1. Load PML4[1] from memory (~100 ns)
    2. Load PDPT[511] from memory (~100 ns)
    3. Load PD[511] from memory (~100 ns)
    4. Load PT[32] from memory (~100 ns)
    Total: ~400 ns = 1200+ CPU cycles

HUGE PAGE OPTIMIZATION:

Regular 4 KB pages:
  TLB entry covers 4 KB
  64 TLB entries cover 256 KB total
  Working set > 256 KB: constant TLB misses

2 MB huge pages:
  TLB entry covers 2 MB
  64 TLB entries cover 128 MB total
  Working set up to 128 MB: no TLB misses
  Page walk for 2 MB page: 3 levels (skip L1 PT level)
```

---

## 3. PERFORMANCE LENS

### What Breaks and What Wins

**Pattern 1: Sequential Access (Wins)**

```c
for (int i = 0; i < 1000000; i++) {
    process(arr[i]);
}
```

- **Cache behavior**: Perfect spatial locality. Each element is in the same cache line as the next.
- **Hardware prefetching**: Stream prefetcher detects sequential pattern, prefetches next lines.
- **Bandwidth**: Limited by memory bandwidth (100 GB/s), not latency.
- **Performance**: ~100 cycles per element (memory limited).

**Pattern 2: Strided Access (Moderate)**

```c
for (int i = 0; i < 1000000; i++) {
    process(arr[i * 2]);  // Stride of 2
}
```

- **Cache behavior**: Spatial locality partially exploited. Many cache line accesses are "wasted" (fetched line but only half used).
- **Hardware prefetching**: Stride prefetcher may detect the pattern, or may not (depends on CPU).
- **Performance**: Depends on stride. Stride of 2: hit rate 50%, accesses twice as many cache lines.

**Pattern 3: Random Access (Breaks)**

```c
for (int i = 0; i < 1000000; i++) {
    process(arr[rand() % 1000000]);  // Random index
}
```

- **Cache behavior**: No spatial or temporal locality. Cache misses on every access.
- **Hardware prefetching**: Cannot predict next access (random).
- **Bandwidth**: Effective bandwidth is L3 bandwidth (40 GB/s), not main memory bandwidth.
- **Performance**: L3 miss rate ~90%, each miss costs ~300 cycles. ~1000 cycles per element (100x slower than sequential).

**Pattern 4: Working Set Too Large (Capacity Miss)**

```c
int x[100 * 1024 * 1024];  // 400 MB array
for (int i = 0; i < 1000; i++) {
    for (int j = 0; j < 100 * 1024 * 1024; j++) {
        process(x[j]);  // L3 cache is only 8 MB
    }
}
```

- **Cache behavior**: Working set (400 MB) >> L3 cache (8 MB), so every iteration has 400 MB / 64 bytes per line = 6.25 million cache lines to process. L3 has only 8 MB / 64 = 131,072 lines, so miss rate ~98%.
- **Performance**: Hit L3 very few times. Mostly hitting DRAM (100 ns latency).

**Pattern 5: Multiple Readers/Writers to Same Line (False Sharing)**

```c
struct {
    int counter0;  // Core 0 writes to this
    int counter1;  // Core 1 writes to this
} counters;

// Core 0:
for (int i = 0; i < 1000000; i++) {
    counters.counter0++;
}

// Core 1:
for (int i = 0; i < 1000000; i++) {
    counters.counter1++;
}
```

- **Problem**: Both fields fit in one 64-byte cache line. When Core 0 writes counter0, the entire line is invalidated on Core 1. Core 1 must fetch the line again.
- **Cache behavior**: Despite writing to different memory locations, cores fight over the same cache line (false sharing).
- **Performance**: Each write causes an invalidation, forcing a reload (coherence miss). ~300 cycle penalty per write (should be ~3 cycles for register access).

**Solution: Padding to avoid false sharing**

```c
struct {
    int counter0;
    int padding0[15];  // Padding to next cache line
    int counter1;
    int padding1[15];
} counters;
```

Now counter0 and counter1 are on different cache lines. No false sharing.

**Pattern 6: TLB Thrashing (Breaks)**

```c
void process_large_array(char *data, size_t size) {
    for (int i = 0; i < size; i++) {
        process(data[i]);
    }
}

// Called with 100 GB array:
process_large_array(huge_allocation, 100UL << 30);
```

- **Problem**: 100 GB / 4 KB per page = 25 million pages. TLB has only 64 entries (for 4 KB pages), so misses on ~99.97% of page transitions.
- **Performance**: TLB miss costs 400 ns = ~1200 cycles. Total TLB miss penalty: 25 million × 1200 cycles = 30 trillion cycles = much slower.
- **Solution**: Use 2 MB huge pages. Now 100 GB / 2 MB = 50,000 pages, fitting in TLB much better.

---

## 4. ANNOTATED CODE

### Example 1: Measuring Cache Misses with perf

```c
// cache_measurement.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_SIZE (1UL << 24)  // 16 MB array

// Pattern 1: Sequential access (good cache behavior)
void sequential_access(int *arr, size_t size) {
    for (size_t i = 0; i < size; i++) {
        arr[i] *= 2;  // Simple operation
    }
}

// Pattern 2: Strided access (degraded cache behavior)
void strided_access(int *arr, size_t size, int stride) {
    for (size_t i = 0; i < size; i += stride) {
        arr[i] *= 2;
    }
}

// Pattern 3: Random access (poor cache behavior)
void random_access(int *arr, size_t size, int *indices) {
    for (size_t i = 0; i < size; i++) {
        arr[indices[i]] *= 2;  // Unpredictable pattern
    }
}

int main() {
    int *arr = malloc(ARRAY_SIZE * sizeof(int));
    int *indices = malloc(ARRAY_SIZE * sizeof(int));

    // Initialize
    memset(arr, 0, ARRAY_SIZE * sizeof(int));
    for (size_t i = 0; i < ARRAY_SIZE; i++) {
        indices[i] = rand() % ARRAY_SIZE;
    }

    printf("Sequential access:\n");
    sequential_access(arr, ARRAY_SIZE);

    printf("Strided access (stride=2):\n");
    strided_access(arr, ARRAY_SIZE, 2);

    printf("Random access:\n");
    random_access(arr, ARRAY_SIZE, indices);

    free(arr);
    free(indices);
    return 0;
}
```

**Compilation and Measurement**:

```bash
gcc -O3 cache_measurement.c -o cache_measurement

# Measure cache misses:
perf stat -e cache-references,cache-misses ./cache_measurement

# Output:
#  Performance counter stats for './cache_measurement':
#
#        3,456,789 cache-references
#          234,567 cache-misses              # 6.78% of all cache refs
#
# Sequential:   ~6% misses (good)
# Strided:     ~15% misses (degraded)
# Random:      ~90% misses (poor)
```

**Detailed Analysis**:

```bash
# Measure L3 cache-specific events:
perf record -e LLC-load,LLC-load-misses,LLC-store ./cache_measurement
perf report

# Show assembly with annotations:
perf annotate sequential_access
```

### Example 2: Demonstrating False Sharing

```c
// false_sharing.c
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define ITERATIONS 100000000

// Scenario 1: False sharing (BAD)
struct {
    long counter0;  // Core 0 writes
    long counter1;  // Core 1 writes
} shared_bad;      // Both in same cache line

// Scenario 2: Padding to avoid false sharing (GOOD)
struct {
    long counter0;
    long padding0[7];  // Pad to next cache line (8 × 8 bytes = 64 bytes)
    long counter1;
    long padding1[7];
} shared_good;

void *thread0_bad(void *arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        shared_bad.counter0++;
    }
    return NULL;
}

void *thread1_bad(void *arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        shared_bad.counter1++;
    }
    return NULL;
}

void *thread0_good(void *arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        shared_good.counter0++;
    }
    return NULL;
}

void *thread1_good(void *arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        shared_good.counter1++;
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
    printf("Time: %ld ms\n", (end - start) * 1000 / CLOCKS_PER_SEC);

    printf("\nPADDING TO AVOID FALSE SHARING (GOOD):\n");
    start = clock();

    pthread_create(&t0, NULL, thread0_good, NULL);
    pthread_create(&t1, NULL, thread1_good, NULL);
    pthread_join(t0, NULL);
    pthread_join(t1, NULL);

    end = clock();
    printf("Time: %ld ms\n", (end - start) * 1000 / CLOCKS_PER_SEC);

    return 0;
}
```

**Compilation and Execution**:

```bash
gcc -O3 -pthread false_sharing.c -o false_sharing
./false_sharing

# Output:
# FALSE SHARING (BAD):
# Time: 5000 ms
#
# PADDING TO AVOID FALSE SHARING (GOOD):
# Time: 1000 ms
```

The padded version is **5x faster** because it eliminates false sharing cache coherence traffic.

### Example 3: Prefetching Example

```c
// prefetch_demo.c
#include <xmmintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_SIZE (1UL << 20)  // 1 million elements
#define PREFETCH_DISTANCE 8     // Prefetch 8 elements ahead

// Version 1: No prefetch (baseline)
void process_no_prefetch(float *data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = data[i] * 2.5f + 1.3f;
    }
}

// Version 2: Software prefetch
void process_with_prefetch(float *data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        // Prefetch element that will be accessed in ~8 iterations
        _mm_prefetch(&data[i + PREFETCH_DISTANCE], _MM_HINT_T0);

        // Process current element
        data[i] = data[i] * 2.5f + 1.3f;
    }
}

// Version 3: Explicit blocking to improve cache reuse
#define BLOCK_SIZE 256
void process_with_blocking(float *data, size_t size) {
    for (size_t b = 0; b < size; b += BLOCK_SIZE) {
        // Process BLOCK_SIZE elements (fits in L1 cache)
        for (size_t i = b; i < b + BLOCK_SIZE && i < size; i++) {
            data[i] = data[i] * 2.5f + 1.3f;
        }
    }
}

int main() {
    float *data = malloc(ARRAY_SIZE * sizeof(float));

    // Warm up cache
    for (size_t i = 0; i < ARRAY_SIZE; i++) {
        data[i] = i * 0.1f;
    }

    printf("NO PREFETCH:\n");
    clock_t start = clock();
    process_no_prefetch(data, ARRAY_SIZE);
    clock_t end = clock();
    printf("Time: %.2f ms\n", (double)(end - start) * 1000 / CLOCKS_PER_SEC);

    printf("\nWITH SOFTWARE PREFETCH:\n");
    start = clock();
    process_with_prefetch(data, ARRAY_SIZE);
    end = clock();
    printf("Time: %.2f ms\n", (double)(end - start) * 1000 / CLOCKS_PER_SEC);

    printf("\nWITH BLOCKING:\n");
    start = clock();
    process_with_blocking(data, ARRAY_SIZE);
    end = clock();
    printf("Time: %.2f ms\n", (double)(end - start) * 1000 / CLOCKS_PER_SEC);

    free(data);
    return 0;
}
```

**Expected Results**:

```
NO PREFETCH:
Time: 4.5 ms

WITH SOFTWARE PREFETCH:
Time: 3.2 ms   (29% faster)

WITH BLOCKING:
Time: 2.1 ms   (53% faster)
```

### Example 4: NUMA-Aware Allocation and Affinity

```c
// numa_demo.c
#define _GNU_SOURCE
#include <numa.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DATA_SIZE (1UL << 30)  // 1 GB

// Scenario 1: Data allocated on wrong socket
void *wrong_socket_thread(void *arg) {
    float *data = (float *)arg;

    clock_t start = clock();
    for (size_t i = 0; i < DATA_SIZE / sizeof(float); i += 1024) {
        // Access every 1024th element
        float x = data[i];
        data[i] = x + 1.0f;
    }
    clock_t end = clock();

    printf("Wrong socket: %ld ms\n", (end - start) * 1000 / CLOCKS_PER_SEC);
    return NULL;
}

// Scenario 2: Data allocated on correct socket
void *correct_socket_thread(void *arg) {
    float *data = (float *)arg;

    // Pin thread to socket 0
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);  // Pin to core 0 (socket 0)
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    clock_t start = clock();
    for (size_t i = 0; i < DATA_SIZE / sizeof(float); i += 1024) {
        float x = data[i];
        data[i] = x + 1.0f;
    }
    clock_t end = clock();

    printf("Correct socket: %ld ms\n", (end - start) * 1000 / CLOCKS_PER_SEC);
    return NULL;
}

int main() {
    if (numa_available() < 0) {
        printf("NUMA not available\n");
        return 1;
    }

    printf("NUMA topology: %d nodes\n", numa_num_configured_nodes());

    // Scenario 1: Allocate on socket 1, access from socket 0
    printf("\nSCENARIO 1: Access remote DRAM\n");
    float *data_socket1 = numa_alloc_onnode(DATA_SIZE, 1);

    pthread_t t;
    pthread_create(&t, NULL, wrong_socket_thread, data_socket1);
    pthread_join(t, NULL);

    numa_free(data_socket1, DATA_SIZE);

    // Scenario 2: Allocate on socket 0, access from socket 0
    printf("\nSCENARIO 2: Access local DRAM\n");
    float *data_socket0 = numa_alloc_onnode(DATA_SIZE, 0);

    pthread_create(&t, NULL, correct_socket_thread, data_socket0);
    pthread_join(t, NULL);

    numa_free(data_socket0, DATA_SIZE);

    return 0;
}
```

**Expected Results on 2-socket system**:

```
NUMA topology: 2 nodes

SCENARIO 1: Access remote DRAM
Wrong socket: 2500 ms  (100 ns latency, 50 GB/s bandwidth)

SCENARIO 2: Access local DRAM
Correct socket: 1200 ms  (60 ns latency, 100 GB/s bandwidth)
```

The local socket is **2x faster** due to lower latency and higher bandwidth.

---

## 5. EXPERT INSIGHT

### Non-Obvious Truths About Memory Hierarchy

**Truth 1: Cache Line Alignment Can Mean 10x Speedup**

Consider a simple operation:

```c
float dot_product(float *a, float *b, int len) {
    float sum = 0;
    for (int i = 0; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
```

If `a` and `b` start at misaligned addresses (e.g., 0x1002 instead of 0x1000), each 64-byte cache line is split across two physical cache lines. This causes:
- Reduced prefetcher effectiveness (prefetcher assumes aligned access)
- Extra cache fills (fetching unaligned data requires 2 fills instead of 1)
- 10-20% slowdown

**Solution**: Ensure critical data structures are cache-aligned:

```c
// Align to 64-byte boundary
float a[LEN] __attribute__((aligned(64)));
float b[LEN] __attribute__((aligned(64)));
```

The senior engineer verifies cache alignment with tools like objdump and consciously pads data structures.

**Truth 2: The Memory Wall is Real, and Your Algorithm Doesn't Matter**

Moore's Law gave us 2x CPU speed every 2 years, but memory latency improved only 25% in the same period. This "memory wall" means:

```
1990: CPU 10 MHz,   DRAM 100 ns (1000 cycles)
2000: CPU 1 GHz,    DRAM 100 ns (100 cycles)
2010: CPU 3 GHz,    DRAM 100 ns (300 cycles)
2020: CPU 5 GHz,    DRAM 100 ns (500 cycles)
```

A single DRAM access stalls the processor for 500 cycles. In that time, the CPU could execute 500 independent operations. This means:

**Any algorithm that misses the cache is fundamentally bottlenecked by memory bandwidth, not CPU speed.**

A 10x faster CPU doesn't help if your code is memory-bound.

**Truth 3: Prefetching is Not Automatic for All Patterns**

Programmers often assume hardware prefetchers will cover all access patterns. In reality:
- Stream prefetcher works: linear scans (0, 1, 2, 3, ...)
- Stride prefetcher works: constant-stride patterns (0, 2, 4, 6, ...)
- Random patterns: prefetcher is useless (cannot predict)
- Complex patterns: prefetcher may work, but is not guaranteed

For example:

```c
// Pointer chasing (prefetcher cannot help):
struct Node {
    int value;
    struct Node *next;
};

struct Node *p = head;
while (p != NULL) {
    process(p->value);
    p = p->next;  // Next address depends on current value
}
```

The prefetcher cannot predict the next pointer value, so each dereference is a cache miss. Solution: use software prefetching or restructure the data (e.g., convert linked list to array).

**Truth 4: False Sharing is Subtler Than You Think**

A classic false sharing scenario:

```c
struct {
    int counter0;
    int counter1;
} shared;
```

Both fields are in the same 64-byte cache line. When Core 0 writes counter0, Core 1's cache line is invalidated, causing Core 1 to fetch it again.

But false sharing is not just about adjacent integers. It also happens with:

```c
struct {
    atomic_int lock0;
    atomic_int lock1;
};
```

Even if the locks are only read (not written), if they're on the same cache line, accessing lock0 will pull in lock1 (wasting cache space). The senior engineer pads all shared synchronization primitives to avoid this.

**Truth 5: NUMA Latency is Not Linear**

On a 2-socket system, accessing remote DRAM is ~2x slower than local. But on larger systems (4+ sockets), the worst-case latency can be much worse:

```
2-socket:  Local 60 ns, Remote 120 ns (2x)
4-socket:  Local 60 ns, Remote up to 200 ns (3.3x)
8-socket:  Local 60 ns, Remote up to 300 ns (5x)
```

This is because accessing socket 3's memory from socket 0 may require traversing multiple hops on the interconnect:
- Socket 0 → Socket 1 (QPI) → Socket 3 (QPI)

Each hop adds latency. For this reason, NUMA-aware scheduling becomes exponentially more important on larger systems.

**Truth 6: TLB Misses Are Invisible in Many Profiling Tools**

perf can measure cache misses, but not all profilers expose TLB miss data. A program can have:
- Low cache miss rate (good)
- High TLB miss rate (bad)
- Net result: 50% slowdown

Example:

```bash
perf stat -e dTLB-loads,dTLB-load-misses ./my_program

# Output:
#   1,000,000 dTLB-loads
#       10,000 dTLB-load-misses  # 1% miss rate, seems good

# But each TLB miss costs 400+ ns, while cache miss costs 100 ns
# So TLB misses dominate performance despite low percentage
```

The senior engineer checks both cache and TLB metrics.

**Truth 7: Huge Pages Are Not Always Faster**

Huge pages reduce TLB misses, which is usually good. But they have downsides:

1. **Reduced flexibility**: Can't swap huge pages (usually). If memory pressure increases, OS cannot reclaim individual 4 KB pages—it must swap the entire 2 MB page.

2. **Memory overhead**: If you allocate 100 MB with 2 MB huge pages, you must allocate 100 MB upfront. Sparse allocations waste space.

3. **Page table overhead**: Large page tables require larger PDPT/PD structures.

For ML inference (where models are huge and pre-allocated), huge pages are beneficial. For general-purpose applications with variable memory usage, they may actually hurt.

---

## 6. BENCHMARK / MEASUREMENT

### Tools and Commands for Measuring Memory Performance

**Tool 1: likwid (LIKWID = Like I Like It Likwid Information Database)**

```bash
# Install:
apt-get install likwid-tools

# List available events:
likwid-topology       # CPU topology
likwid-perfctr -a    # Available performance counters

# Measure L3 cache misses:
likwid-perfctr -g L3 ./my_program

# Output:
# ┌─ Event L3-loads
# ├─ Event L3-load-misses
# ├─ Event L3-stores
# └─ Event L3-store-misses

# Measure memory bandwidth:
likwid-perfctr -g MEM ./my_program
```

**Tool 2: Intel VTune**

```bash
# Collect CPU metrics:
vtune -collect general-exploration ./my_program

# View results:
vtune -report summary -r r000gx

# Detailed analysis of memory subsystem:
vtune -collect memory-access ./my_program
```

**Tool 3: Cachegrind (Part of Valgrind)**

```bash
# Simulate cache behavior:
valgrind --tool=cachegrind ./my_program

# View results:
cg_annotate cachegrind.out.PID

# Output:
# ============ CACHE SIMULATION RESULTS ============
#
# Cache         Total    Misses   Miss Rate
# L1i           5000000  50000    1.00%
# L1d           4000000  200000   5.00%
# LLC           400000   50000    12.50%
```

Cachegrind simulates the exact cache behavior, but runs 50-100x slower than native execution.

**Tool 4: perf for TLB Analysis**

```bash
# Measure TLB misses:
perf stat -e dTLB-loads,dTLB-load-misses,iTLB-loads,iTLB-load-misses ./my_program

# Example output:
#       10,000,000 dTLB-loads
#            1,000 dTLB-load-misses  # 0.01% miss rate
#        2,000,000 iTLB-loads
#              500 iTLB-load-misses

# High iTLB-load-misses indicate instruction fetch issues (rare)
# High dTLB-load-misses indicate data access issues (common for large working sets)
```

**Tool 5: numastat - NUMA Memory Stats**

```bash
# Monitor NUMA allocation:
numastat -p my_program

# Output:
#                           node0     node1
# my_program  mem 10000.00 5000.00   # 10 GB on node0, 5 GB on node1
```

Shows how memory is distributed across NUMA nodes.

**Tool 6: Memory Bandwidth Measurement (STREAM)**

```bash
# Compile STREAM:
gcc -O3 -march=native -fopenmp stream.c -o stream

# Run with large array:
export OMP_NUM_THREADS=8
./stream

# Output shows achievable bandwidth:
# Copy:  250 GB/s
# Triad: 280 GB/s
```

Compare to theoretical peak:
```bash
# Theoretical peak = Channel count × Frequency × Bus width
# Example: 6 channels × 3200 MT/s × 8 bytes = 153.6 GB/s

# If STREAM achieves 120 GB/s on a 153.6 GB/s system:
# Efficiency = 120 / 153.6 = 78%
```

Low efficiency (< 50%) indicates:
- Memory access patterns are not optimal
- Caches are not being used efficiently
- May need vectorization or blocking

**Tool 7: perf c2c (Cache-to-Cache Coherence)**

```bash
# Record coherence traffic:
perf record -e LLC-loads,LLC-stores ./my_program

# Analyze cache coherence:
perf c2c record ./my_program
perf c2c report

# Shows false sharing and cache-to-cache transfers
```

**Tool 8: Custom Benchmarks**

Write microbenchmarks to test specific patterns:

```c
// random_access.c
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    int *arr = malloc(1UL << 25);  // 32 MB
    int *indices = malloc(1000000 * sizeof(int));

    for (int i = 0; i < 1000000; i++) {
        indices[i] = rand() % (1UL << 25);
    }

    clock_t start = clock();
    for (int i = 0; i < 1000000; i++) {
        arr[indices[i]]++;
    }
    clock_t end = clock();

    printf("Time: %.2f ms\n", (double)(end - start) * 1000 / CLOCKS_PER_SEC);
    // Expected: 5-10 seconds (mostly waiting for memory)
}
```

---

## 7. ML SYSTEMS RELEVANCE

### Memory Hierarchy in Inference Engines

**Principle 1: Batch Size Determines Memory Behavior**

In transformer inference, batch size directly impacts cache behavior:

```c
// Batch size 1 (interactive inference):
for (int b = 0; b < 1; b++) {
    for (int s = 0; s < seq_len; s++) {
        // Attention: O(seq_len²) operations
        // Working set: seq_len² elements
        // For seq_len=2048: 4M elements, fits in L3
        // Hits L3 cache, ~40 ns latency
    }
}

// Batch size 128:
for (int b = 0; b < 128; b++) {
    for (int s = 0; s < seq_len; s++) {
        // Working set: 128 × seq_len² elements
        // For seq_len=2048: 512M elements, exceeds L3
        // Misses to DRAM, ~100 ns latency
        // But 128x more operations, so amortizes latency cost
}
```

**Key insight**: Batch size 1 is latency-bound (waiting for memory). Batch size 128 is throughput-bound (memory bandwidth-limited but better amortization).

**Principle 2: KV Cache Locality in Attention**

In transformer inference, the KV cache (key/value tensors from previous tokens) must be accessed billions of times:

```c
// Poor locality: key/value are scattered
for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < i; j++) {
        float score = dot(query[i], keys[j]);  // keys[j] is scattered
        float value = values[j];               // values[j] is scattered
    }
}
```

If seq_len=2048 and each key/value is 128 floats (512 bytes), the working set is scattered across millions of cache lines. Cache misses dominate.

**Better approach**: Layout keys/values contiguously by token:

```c
// Better locality: keys are contiguous in memory
struct KVCache {
    float keys[seq_len][hidden_dim];     // Contiguous
    float values[seq_len][hidden_dim];
};

for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < i; j++) {
        float *key_j = &cache.keys[j][0];       // Contiguous
        float *value_j = &cache.values[j][0];
        // Better prefetching and cache hit rate
    }
}
```

**Impact**: 30-50% faster attention kernel due to better cache locality.

**Principle 3: Matrix Multiplication Requires Cache-Oblivious Algorithms**

Standard matrix multiply:

```c
// C = A × B
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        float sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[i][k] * B[k][j];
        }
        C[i][j] = sum;
    }
}
```

For large N (e.g., N=4096), the innermost loop loads from B column-by-column. Each B[k][j] access is far apart in memory (strided), causing TLB and cache thrashing.

**Better approach**: Cache-oblivious blocking:

```c
#define BLOCK_SIZE 64  // Fits in L1 cache

void matmul_blocked(float *C, float *A, float *B, int N) {
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                // Compute block C[ii:ii+BS][jj:jj+BS]
                // Uses A block and B block multiple times
                for (int i = ii; i < ii + BLOCK_SIZE; i++) {
                    for (int j = jj; j < jj + BLOCK_SIZE; j++) {
                        float sum = C[i*N + j];
                        for (int k = kk; k < kk + BLOCK_SIZE; k++) {
                            sum += A[i*N + k] * B[k*N + j];
                        }
                        C[i*N + j] = sum;
                    }
                }
            }
        }
    }
}
```

This brings the 64×64 blocks into L1 cache, so the inner loop reuses them multiple times.

**Impact**: 3-5x faster matrix multiplication due to better cache reuse.

**Principle 4: GEMM (General Matrix Multiply) is Memory-Bound**

For a M×K × K×N matrix multiply:
- FLOPs: 2 × M × K × N (multiply + add)
- Memory accesses: M×K + K×N + M×N

```
Arithmetic intensity = FLOPs / Memory
                     = (2 × M × K × N) / (M×K + K×N + M×N)
                     = (2 × M × K × N) / (M×K + K×N + M×N)
```

For square matrices (M = K = N):
```
Arithmetic intensity = (2 × N³) / (N² + N² + N²) = 2N³ / 3N² = 2N/3
```

For N=1024:
```
Arithmetic intensity = 2 × 1024 / 3 ≈ 680 FLOPs per byte accessed
```

With 280 GB/s memory bandwidth:
```
Peak throughput = 280 GB/s × 680 FLOPs/byte × 10^9 bytes/GB = 190 TFLOPs
```

But actual CPU peak is ~500 TFLOPs (for FP32), so matmul is **memory-bound** (limited by bandwidth, not CPU speed).

This means:
- Faster CPUs don't help (memory still limited)
- Larger caches help (reduce memory traffic)
- Better prefetching helps (hide latency)
- Vectorization helps (more FLOPs per memory access)

---

## 8. PhD QUALIFIER QUESTIONS

1. **Cache Hierarchy and Memory Access**:
   Explain why a program with random memory access (address = rand() % ARRAY_SIZE) runs 10-100x slower than one with sequential access, even though both perform the same number of arithmetic operations. Discuss cache misses, TLB behavior, and hardware prefetching.

2. **False Sharing and Cache Coherence**:
   Two threads on different cores write to different variables that happen to be on the same 64-byte cache line. Explain how cache coherence protocols (MESI/MESIF) cause invalidations, and why padding to separate the variables improves performance. What is the penalty per invalidation?

3. **NUMA and Multi-Socket Systems**:
   On a 2-socket system with 256 GB of DRAM, explain the memory allocation strategy for a 100 GB ML model tensor. What are the tradeoffs between allocating all memory on socket 0 vs balancing across both sockets? How would the answer change if the model has 10 GB total?

4. **TLB and Huge Pages**:
   A program processes a 10 GB array. Explain why TLB misses become a bottleneck with 4 KB pages but not with 2 MB huge pages. Calculate the TLB miss rate in both cases, and the total time penalty. At what working set size do 4 KB pages become unusable?

5. **Cache Replacement and Conflict Misses**:
   Explain pseudo-LRU replacement policy vs true LRU. Design a microbenchmark that exhibits a conflict miss (same cache set, different addresses). How would you fix it without increasing cache size?

---

## 9. READING LIST

**Essential Textbooks**:
- Bryant, Randal E., and David R. O'Hallaron. *Computer Systems: A Programmer's Perspective (3rd Edition)*. Pearson, 2015.
  - **Chapter 6**: "The Memory Hierarchy" — cache architecture, cache performance, memory mountain
  - **Chapter 9**: "Virtual Memory" — page tables, TLB, page replacement, memory mapping

- Hennessy, John L., and David A. Patterson. *Computer Architecture: A Quantitative Approach (6th Edition)*. Morgan Kaufmann, 2017.
  - **Chapter 2**: "Memory Hierarchy" — cache hierarchy, cache optimization, prefetching
  - **Chapter 5**: "Memory Hierarchies" (detailed coverage)
  - **Chapter 9**: "Interconnects" — NUMA, cache coherence

**Lecture Series**:
- Mutlu, Onur. "Computer Architecture Lectures" (CMU 18-742).
  - **Lecture 5**: "Memory Hierarchy" — comprehensive overview
  - **Lecture 9**: "Prefetching" — hardware and software prefetching techniques
  - **Lecture 12**: "Cache Replacement and Performance" — LRU vs PLRU

**Optimization Guides**:
- Agner Fog. *The Microarchitecture of Intel, AMD and VIA CPUs: An Optimization Guide for Assembly Programmers* (2022).
  - Detailed cache behavior on specific CPU models
  - Memory latency and bandwidth characteristics

- Intel Corporation. *Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 3A: System Programming Guide*.
  - Section 11: "Memory Cache Control" — cache operations, PAT (Page Attribute Table)

**Tools and Analysis**:
- Serebryany, Konstantin, et al. "AddressSanitizer: A Fast Address Sanity Checker." USENIX ATC (2012).
  - Runtime tool for detecting memory errors affecting cache behavior

- "perf: Linux Performance Analysis Tool" Documentation: https://perf.wiki.kernel.org/
  - Official documentation for perf counters, event monitoring, profiling

**Additional Resources**:
- NUMA documentation: man numa, man numactl, libnuma API
- TLB and paging: Linux kernel documentation on transparent huge pages, hugetlb
- STREAM Benchmark: https://www.cs.virginia.edu/stream/
- likwid: https://github.com/RRZE-HPC/likwid (performance analysis toolkit)

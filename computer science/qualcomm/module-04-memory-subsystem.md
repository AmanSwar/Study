# Module 4: Memory Subsystem Mastery

**Hexagon NPU Curriculum | Advanced Topics**

**Target Audience**: PhD-level ML systems engineers, Hexagon architecture specialists
**Prerequisite Knowledge**: Module 1-3, basic computer architecture, HVX fundamentals
**Duration**: 20-25 hours of study + implementation
**Last Updated**: 2026-03

---

## Table of Contents

1. [Introduction & Motivation](#introduction--motivation)
2. [VTCM Architecture & Programming](#vtcm-architecture--programming)
3. [Data Layout Design for Maximum Throughput](#data-layout-design-for-maximum-throughput)
4. [DMA Engine Programming](#dma-engine-programming)
5. [Cache Management & Optimization](#cache-management--optimization)
6. [Avoiding TCM Bank Conflicts](#avoiding-tcm-bank-conflicts)
7. [Designing a Scratchpad Allocator](#designing-a-scratchpad-allocator)
8. [Reference Implementation](#reference-implementation)
9. [Self-Assessment & Further Reading](#self-assessment--further-reading)

---

## Introduction & Motivation

### The Memory Wall Problem

Modern Hexagon processors achieve extraordinary compute density through:
- **Dense HVX units**: 128-bit operations @ 1.5 GHz = 1.2+ TFLOPS per core
- **Multiple cores**: Up to 12 HVX cores per DSP island
- **Parallel memory transactions**: Up to 64 bytes/cycle bandwidth available

However, **data movement dominates power budget** in digital signal processing (DSP) and neural network inference:

```
Energy Cost Analysis (28nm to 5nm process transitions):
├─ Compute (MAC): 0.02-0.04 pJ per operation
├─ L1 Cache Access: 0.1-0.2 pJ
├─ L2 Cache Access: 0.5-1.0 pJ
├─ VTCM Access: 0.15-0.25 pJ (ultra-low power, on-core)
├─ Main Memory Access: 20-100 pJ
└─ Off-die DRAM: 100-500 pJ

Power ≈ 50-80% data movement, only 20-50% compute
```

The **memory subsystem** is not just a bottleneck—it is **the primary design constraint** for achieving:
- High throughput (GMACs/J)
- Deterministic latency (inference SLAs)
- Power efficiency (mW/GFLOP)

### Hexagon Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│ DRAM (Off-Die) — 4-12 GB/s bandwidth, 100-500 pJ per access │
│ (Shared between Hexagon, GPU, ARM cores)                     │
└────────────────────┬────────────────────────────────────────┘
                     │ 32-64 GB/s
┌────────────────────▼────────────────────────────────────────┐
│ L2 Cache (Unified) — 256 KB-1 MB per core cluster            │
│ • Coherent with L1 and VTCM                                  │
│ • Full 64-byte line width                                    │
│ • 2-4 cycle access latency                                   │
└────────────────────┬────────────────────────────────────────┘
                 16 GB/s each
        ┌────────────┴────────────┐
        │                         │
┌───────▼──────────┐    ┌────────▼──────────┐
│ L1 DCache        │    │ VTCM (TCM)         │
│ 16-32 KB        │    │ 256 KB - 4 MB      │
│ 4-way set       │    │ Zero-latency      │
│ • 1 cycle       │    │ • 8 banks         │
│ • Write-back    │    │ • 32-128 bytes    │
└─────────────────┘    │   per bank per    │
                       │   cycle           │
                       └────────────────────┘

        ┌──────────────────────────┐
        │ HVX Vector Registers     │
        │ 128 registers × 128 bits │
        │ 1 cycle latency          │
        └──────────────────────────┘
```

⚡ **Expert Insight**: In modern Hexagon designs (SDM888, SDM8cx Gen 3), the total on-die memory (L1 + L2 + VTCM) is **60-80 MB**, while weight tensors for state-of-the-art models exceed **1 GB**. The memory subsystem must therefore:
1. Maximize **data reuse** (reduce DRAM traffic)
2. Stage data through **VTCM** (ultra-low power)
3. Manage **DMA operations** with minimal CPU overhead
4. Avoid **bank conflicts** in parallel access patterns

---

## VTCM Architecture & Programming

### 1. VTCM Overview & Sizing

**VTCM (Vector Tightly-Coupled Memory)** is a special on-core memory designed specifically for:
- Vector processing (HVX workloads)
- Ultra-predictable latency (no cache misses)
- Low power consumption (~1/100th the cost of DRAM access)
- Direct DMA access without cache coherency overhead

#### Typical Sizes Across Generations

```
┌──────────────────┬────────────┬─────────────────────────┐
│ Hexagon Version  │ VTCM Size  │ Key Characteristics     │
├──────────────────┼────────────┼─────────────────────────┤
│ v68 (SDM865)     │ 256 KB     │ Single-core             │
│ v69 (SDM888)     │ 512 KB     │ Dual HVX cores          │
│ v73 (SDM8cx)     │ 1 MB       │ 4-core cluster          │
│ v75 (SDM8cx G2)  │ 2 MB       │ 2 × 2-core clusters     │
│ v78 (SDM8cx G3)  │ 4 MB       │ Multi-island design     │
│ Future (v80+)    │ 8 MB       │ Projected                │
└──────────────────┴────────────┴─────────────────────────┘
```

**Allocation Breakdown** for typical inference workload:

```
4 MB VTCM Pool
├─ Activations (Working Buffers): 1.5 MB (37.5%)
│  ├─ Current Layer Input: 512 KB
│  ├─ Layer Output: 512 KB
│  └─ Temporary Computations: 512 KB
│
├─ Weights (Quantized): 2.0 MB (50%)
│  ├─ Weight Tiles (batch 1): 1.2 MB
│  ├─ Bias/Scale Factors: 512 KB
│  └─ Lookup Tables (QAT): 256 KB
│
└─ Allocator Metadata & Alignment: 512 KB (12.5%)
   ├─ Free Pool Tracking: 64 KB
   ├─ Fragmentation Padding: 256 KB
   └─ DMA Descriptors: 192 KB
```

### 2. Banking Structure

VTCM is divided into **independent banks** that can be accessed in parallel:

```
VTCM Bank Architecture (4 MB, 8 Banks):

┌──────────────────────────────────────────────┐
│ Bank 0: Bytes 0x000000-0x07FFFF (512 KB)     │
├──────────────────────────────────────────────┤
│ Bank 1: Bytes 0x080000-0x0FFFFF (512 KB)     │
├──────────────────────────────────────────────┤
│ Bank 2: Bytes 0x100000-0x17FFFF (512 KB)     │
├──────────────────────────────────────────────┤
│ Bank 3: Bytes 0x180000-0x1FFFFF (512 KB)     │
├──────────────────────────────────────────────┤
│ Bank 4: Bytes 0x200000-0x27FFFF (512 KB)     │
├──────────────────────────────────────────────┤
│ Bank 5: Bytes 0x280000-0x2FFFFF (512 KB)     │
├──────────────────────────────────────────────┤
│ Bank 6: Bytes 0x300000-0x37FFFF (512 KB)     │
├──────────────────────────────────────────────┤
│ Bank 7: Bytes 0x380000-0x3FFFFF (512 KB)     │
└──────────────────────────────────────────────┘

Bank Selection Formula (Address Decoding):
bank_index = (address >> 19) & 0x7  // 2^19 = 512 KB per bank
bank_offset = address & 0x7FFFF      // 19-bit offset within bank
```

**Bank Width** varies by generation:
- **v68-v73**: 32 bytes per bank per cycle (256-bit access)
- **v75+**: 64 bytes per bank per cycle (512-bit wide paths)

This means **8 banks × 64 bytes/cycle = 512 bytes/cycle** theoretical peak bandwidth.

⚡ **Expert Insight**: Bank bandwidth is **per-port per-cycle**. A single instruction can access all 8 banks in parallel (one port each), but if two instructions access the same bank, they **serialize**. This is why bank conflict avoidance is critical.

### 3. VTCM Allocation APIs

#### Method A: HAP_request_VTCM (Preferred for Qualcomm SDK)

```c
#include <qurt.h>
#include <HAP_farf.h>

// Request VTCM at runtime
typedef struct {
    unsigned int size;      // Request size (bytes)
    unsigned int align;     // Alignment (typically 128 or 256)
} HAP_VTCM_Request_t;

int HAP_request_VTCM(
    unsigned int nVtcmSize,     // Size requested
    int block                   // Block? (1=blocking, 0=non-blocking)
);

// Example Usage:
int vtcm_handle = HAP_request_VTCM(1024 * 1024, 1);  // Request 1 MB
if (vtcm_handle < 0) {
    HAP_FARF(ERROR, "VTCM allocation failed: %d", vtcm_handle);
    return -1;
}

// VTCM is now available, mapped into address space
// Use it normally:
uint8_t *vtcm_ptr = (uint8_t *)(0x28000000 + offset);  // Typical base

// Release when done:
HAP_release_VTCM(vtcm_handle);
```

**Pros**:
- Integrates with FastRPC power management
- Automatic synchronization with thread switching
- Supports dynamic reallocation

**Cons**:
- Limited to single request per thread
- Blocking call can introduce latency

#### Method B: Ion Allocator (Lower-Level, Greater Control)

```c
#include <ion.h>
#include <sys/mman.h>

// Open ION device
int ion_fd = ion_open();

// Allocate VTCM via ION
struct ion_allocation_data alloc = {
    .len = 1024 * 1024,              // 1 MB
    .align = 128,                     // 128-byte alignment
    .heap_id_mask = ION_HEAP_SYSTEM,  // Or ION_HEAP_QSECOM
    .flags = ION_FLAG_CACHED
};

int handle = ion_alloc(ion_fd, alloc.len, alloc.align,
                       alloc.heap_id_mask, alloc.flags);

// Map into user space
int map_fd;
struct ion_fd_data fd_data = { .handle = handle };
ion_share(ion_fd, &fd_data);
map_fd = fd_data.fd;

void *vtcm_vaddr = mmap(NULL, alloc.len, PROT_READ | PROT_WRITE,
                        MAP_SHARED, map_fd, 0);

// Use the memory
uint8_t *vtcm = (uint8_t *)vtcm_vaddr;
// ... perform DMA operations, HVX compute ...

// Cleanup
munmap(vtcm_vaddr, alloc.len);
ion_free(ion_fd, handle);
ion_close(ion_fd);
```

**Pros**:
- Multiple allocations supported
- Per-allocation size control
- Fine-grained heap selection

**Cons**:
- Requires raw ION API knowledge
- Manual synchronization required
- Higher latency for allocation

#### Method C: Static Allocation (Compile-Time)

For real-time inference engines with deterministic requirements:

```c
// Place in VTCM segment (linker script)
// vtcm.ld:
// MEMORY {
//     VTCM : ORIGIN = 0x28000000, LENGTH = 4M
// }

__attribute__((section(".vtcm_data")))
uint8_t static_weight_buffer[2 * 1024 * 1024];  // 2 MB of weights

__attribute__((section(".vtcm_data")))
uint8_t static_activation_buffer[1536 * 1024];  // 1.5 MB working space

// No allocation overhead, fixed at link-time
// Addresses are known at compile-time → perfect for real-time systems
```

### 4. Virtual Address Space Mapping

VTCM memory appears in the processor's virtual address space at a **fixed base address**:

```
Architecture-Dependent VTCM Base Addresses:

Hexagon v68-v75 (Most Common):
  Physical VTCM: Internal to DSP
  Virtual Mapping: 0x28000000 - 0x28FFFFFF (16 MB addressable)
  Actual Available: Variable (256 KB - 4 MB depending on silicon variant)

ARM/Hexagon Heterogeneous Systems:
  DSP VTCM is NOT directly visible to ARM cores
  Sharing via:
    1. Shared DDR buffers (slower but accessible)
    2. Mailbox/message-passing
    3. FastRPC for heterogeneous compute
```

**Mapping Details**:

```c
// Typical address layout on v75 (2 MB VTCM):
#define VTCM_BASE           0x28000000

// First 1 MB: Island 0
#define ISLAND_0_BASE       (VTCM_BASE + 0x00000)
#define ISLAND_0_SIZE       (1024 * 1024)

// Second 1 MB: Island 1
#define ISLAND_1_BASE       (VTCM_BASE + 0x100000)
#define ISLAND_1_SIZE       (1024 * 1024)

// Allocation strategy:
// - Thread 0 uses Island 0
// - Thread 1 uses Island 1
// - No contention, full parallel utilization

// Effective formula:
// vtcm_vaddr = VTCM_BASE + allocated_offset;
```

⚡ **Expert Insight**: On multi-core Hexagon systems (v75, v78), each **core cluster has its own VTCM island**. Affinity matters: if thread A (running on core 0) allocates VTCM and core 0 is in a different island than core 1 (thread B), accessing from thread B incurs **coherency penalties**. Pin threads to cores in the same island.

### 5. VTCM vs L2 Cache Tradeoffs

| Criterion | VTCM | L2 Cache |
|-----------|------|----------|
| **Capacity** | 256 KB - 4 MB (deterministic) | 256 KB - 1 MB (per core) |
| **Access Latency** | 1-2 cycles (zero-miss) | 2-4 cycles (miss: 20-50 cycles) |
| **Power per Access** | 0.15-0.25 pJ | 0.5-1.0 pJ |
| **Allocation** | Explicit (API call) | Automatic (cache policy) |
| **Coherency** | Manual (explicit flush) | Automatic (hardware) |
| **Predictability** | 100% (no replacement) | ~95% (occasional eviction) |
| **Bandwidth** | 512 bytes/cycle (8 banks) | 64 bytes/cycle (single port) |
| **DMA Access** | Direct (zero-copy) | Via cache (coherency overhead) |

#### Decision Tree: When to Use Each?

```
Is memory size < 4 MB?
├─ YES: Is predictability critical?
│   ├─ YES (Real-time inference SLA) → USE VTCM
│   └─ NO (Best-effort compute) → L2 OK (less allocation overhead)
├─ NO: Is data temporal (reused frequently)?
│   ├─ YES (Weights, invariants) → Force into L2 (dcfetch loop)
│   └─ NO (Streaming data) → VTCM if available, else manage with DMA
```

#### Example Tradeoff Analysis

**Scenario**: MobileNet v3 inference, int8 quantized

```
Weights Size: 5 MB (post-quantization)
VTCM Size: 1 MB (v73)
L2 Size: 512 KB

Strategy 1: VTCM + Streaming
├─ Fit 1 MB weights into VTCM
├─ Stream remaining 4 MB from DRAM in 256 KB chunks
├─ Bandwidth: 1 VTCM access (0.25 pJ) + 4 DRAM chunks (4×500 pJ = 2000 pJ)
├─ Power per inference: 2000.25 pJ ≈ 2000 pJ (DRAM dominates)
└─ Latency: Deterministic (DMA prefetch hides DRAM latency)

Strategy 2: L2 Only
├─ Fit 512 KB L2 cache
├─ Remaining weights: DRAM (10 MB traffic, 5000 pJ)
├─ Hit rate ~60% (spatial locality)
└─ Power: ~3000 pJ (cache miss overhead)

VTCM strategy wins by ~33% power, better latency predictability.
```

---

## Data Layout Design for Maximum Throughput

### 1. NHWC vs NCHW: Hexagon's Preference

**Classic CNN Tensor Notation**:

```
NCHW (Channels-First):       N × C × H × W
Typical: (batch=1, channels=64, height=32, width=32)
Layout: [channel0_row0, channel0_row1, ..., channel1_row0, ...]

NHWC (Channels-Last):        N × H × W × C
Typical: (batch=1, height=32, width=32, channels=64)
Layout: [h=0,w=0,c0..c63, h=0,w=1,c0..c63, ..., h=1,w=0,c0..c63, ...]
```

#### Why Hexagon Strongly Prefers NHWC

**Reason 1: Vectorization Efficiency**

```
HVX 128-bit registers can hold:
  - INT8: 128 values (16 channels × 8 spatial)
  - FP16: 64 values (8 channels × 8 spatial)

NHWC Layout Example (4×4 spatial, 8 channels, int8):
┌────────────────────────────────────────────────┐
│ Spatial:    (h=0,w=0)    (h=0,w=1)    (h=0,w=2) ...  │
│ Channels:    c0-c7        c0-c7        c0-c7        │
│                                                     │
│ NHWC Memory: [c0 c1 c2 c3 c4 c5 c6 c7][c0 c1 ... │
│              └─ CONTIGUOUS IN MEMORY ─┘              │
│              └──── SINGLE HVX LOAD ────┘             │
└────────────────────────────────────────────────────┘

NCHW Layout Example (same data):
┌────────────────────────────────────────────────┐
│ All c0:    [16 spatial values, all from ch0]   │
│ All c1:    [16 spatial values, all from ch1]   │
│ ...                                             │
│                                                │
│ NCHW Memory: [h0w0 h0w1 ... h3w3 from c0]     │
│ Then:        [h0w0 h0w1 ... h3w3 from c1]     │
│                                                │
│ For 1 spatial point × 8 channels:              │
│   Need 8 SEPARATE LOADS (non-contiguous) ✗   │
└────────────────────────────────────────────────┘
```

**Reason 2: Bank Conflict Avoidance**

NHWC naturally distributes channels across VTCM banks:

```
VTCM Bank Assignment (8 channels → 8 banks):

Channel Index: 0   1   2   3   4   5   6   7
Bank Index:    0   1   2   3   4   5   6   7
Memory addr:   0x00000000 (ch0) → Bank 0
               0x00080000 (ch1) → Bank 1
               0x00100000 (ch2) → Bank 2
               ...
               0x00380000 (ch7) → Bank 7

Result: Perfect bank interleaving for 8-channel processing!
```

**Reason 3: DMA Efficiency**

NHWC enables **strided DMA transfers** with minimal complexity:

```c
// NHWC weight tensor: [Height=32, Width=32, InputChannels=64, OutputChannels=64]
// Physical layout in DRAM:
// 32 × 32 × 64 × 64 = 4.1 MB total

// DMA descriptor for 1×1 output channel (64 input channels per output):
struct {
    src_addr = weight_base + output_ch * 64 * 32 * 32;  // Jump to output_ch
    src_stride = 32 * 32 * 64;  // Stride to next spatial location
    num_rows = 32 * 32;         // Spatial locations
    dst_addr = VTCM + ...;
} dma_desc;

// NCHW would require complex 2D striding or multiple transfers
```

⚡ **Expert Insight**: Modern frameworks (TensorFlow Lite, PyTorch) increasingly prefer NHWC. Qualcomm's SNPE SDK automatically converts NCHW to NHWC during graph optimization. If your model is NCHW, **transpose at the framework boundary**, not in the hot loop.

### 2. Channel Packing for HVX (INT8 Quantization)

When working with 8-bit quantized tensors, we can **pack multiple channels densely** to maximize VTCM utilization and HVX throughput.

#### Single-Channel Packing (Standard)

```
Input: 1×32×32×64 (NHWC, 8-bit)
Memory Layout:
Byte 0:    h=0, w=0, channel 0
Byte 1:    h=0, w=0, channel 1
...
Byte 63:   h=0, w=0, channel 63
Byte 64:   h=0, w=1, channel 0
...
Total: 1×32×32×64 = 65,536 bytes

HVX Processing (128-byte registers):
├─ Load 128 bytes (16 spatial points, 8 channels each)
├─ Compute dot product: output = sum(input × weight)
├─ Store 1 byte output (accumulated across channels)
└─ Repeat 32×32 = 1024 times

Problem: For deep channels (>128), we need channel-wise packing.
```

#### 128-Channel Packing (For Wide Layers)

Many modern architectures have "wide" layers with 256-512 channels. HVX can process 128 INT8 values per cycle. We **group channels into 128-element superblocks**:

```
Input Tensor: 1×32×32×256 (256 channels)
Packed Format (for HVX):

Original NHWC:
[h0w0: c0 c1 ... c255] [h0w1: c0 c1 ... c255] ... [h31w31: c0 c1 ... c255]

Packed (128-channel blocks):
[h0w0: c0-c127] [h0w0: c128-c255]
[h0w1: c0-c127] [h0w1: c128-c255]
...
[h31w31: c0-c127] [h31w31: c128-c255]

Memory layout:
Block1: 32×32×128 = 131 KB (fits in VTCM)
Block2: 32×32×128 = 131 KB (fits in VTCM)

HVX Kernel:
for each spatial (h,w):
    block1_out = 0
    for ch_block in [0, 128]:  // Two 128-element chunks
        hdr = load_weights_chunk(ch_block)
        data = load_input_chunk(h,w,ch_block)
        block1_out += convolution(data, hdr)
    store_output(block1_out)
```

#### Interleaved Channel Packing (Advanced)

For some HVX operations, **interleaving channels** across banks improves throughput:

```
8 Channels Interleaved Across 8 Banks:

VTCM Memory Layout:
Bank0: [c0 c0 c0 ... from all 32×32 positions]
Bank1: [c1 c1 c1 ... from all 32×32 positions]
Bank2: [c2 c2 c2 ... from all 32×32 positions]
...
Bank7: [c7 c7 c7 ... from all 32×32 positions]

Access Pattern (Perfect Bank Utilization):
Cycle 1: Bank0[i] ← HVX[0]  (load 8 elements from bank0)
         Bank1[i] ← HVX[1]  (load 8 elements from bank1)
         ...
         Bank7[i] ← HVX[7]  (load 8 elements from bank7)
         Total: 64 bytes in 1 cycle (8 banks × 8 bytes)

vs. Standard NHWC:
Cycle 1: Bank0[i] ← HVX[0]  (load c0)
Cycle 2: Bank1[i] ← HVX[1]  (load c1)
...
Cycle 8: Bank7[i] ← HVX[7]  (load c7)
         Total: 64 bytes in 8 cycles (4× slower!)

Conversion Code:
void pack_channels_interleaved(uint8_t *src_nhwc,    // Input
                                uint8_t *dst_vtcm,    // Output
                                int H, int W, int C) {
    // Reshape: (H,W,C) → (H,W,C/8,8) → (C/8,H,W,8) → (C/8,8,H*W)

    for (int b = 0; b < C/8; b++) {           // Bank group
        for (int ch = 0; ch < 8; ch++) {      // Channel in group
            for (int i = 0; i < H*W; i++) {   // Spatial location
                dst_vtcm[b*8*H*W + ch*H*W + i] =
                    src_nhwc[i*C + b*8 + ch];
            }
        }
    }
}
```

### 3. Weight Tiling Strategies

Large weight tensors (multi-megabyte) must be **tiled** to fit into limited VTCM. Tiling strategy directly impacts:
- **Data reuse**: How many times each weight is loaded from DRAM
- **Compute efficiency**: How many MACs per weight load
- **Latency**: How long the critical path takes

#### Tiling Fundamentals

```
Conv Layer: 1×1 Conv (no spatial tiling needed)
Input: (1, H_in, W_in, C_in) = (1, 32, 32, 64)
Weights: (K_h, K_w, C_in, C_out) = (1, 1, 64, 256)
Output: (1, 32, 32, 256)

Naive (No Tiling):
├─ Load all 256 outputs: 256 bytes needed
├─ Reuse each weight 32×32 = 1024 times
└─ Arithmetic Intensity: (32×32×64×2) / (64×256) = 256 MACs per byte

VTCM Tiling (Output Channel Blocking):
├─ Tile output into (H_out_tile=32, W_out_tile=32, C_out_tile=64)
├─ Per tile: 32×32×64 = 65 KB output (fits in VTCM)
├─ Weight tile: (1,1,64,64) = 256 bytes (always fits)
├─ Per-tile MACs: 32×32×64×64 = 4.1M
├─ Per-tile I/O: 65 KB output + 4 KB weights = 69 KB
└─ Arithmetic Intensity: 4.1M × 2 / (69 KB) ≈ 119 MACs/byte
   (Better due to output locality, but still DRAM-bound)
```

#### Multi-Dimensional Tiling Example

```c
// MobileNet v3 bottleneck layer: 3×3 Conv
// Input: (1, 56, 56, 96)
// Weights: (3, 3, 96, 96) depthwise + (1, 1, 96, 240) pointwise
// VTCM: 1 MB

// Tiling strategy for 3×3 depthwise conv:
typedef struct {
    int h_tile;      // Spatial height tile
    int w_tile;      // Spatial width tile
    int c_tile;      // Channel tile
} TileConfig;

TileConfig tile = {
    .h_tile = 16,           // Process 16 rows at a time
    .w_tile = 16,           // Process 16 cols at a time
    .c_tile = 96            // Process all channels per spatial tile
};

// VTCM allocation:
// Input tile: 16×16×96×1 = 24 KB
// Output tile: 14×14×96×1 = 18.8 KB (3×3 stride=1, padding)
// Weights: 3×3×96 = 864 bytes
// Total: ~44 KB (fits easily in 1 MB VTCM)

// Loop structure:
for (h = 0; h < 56; h += tile.h_tile) {
    for (w = 0; w < 56; w += tile.w_tile) {
        // DMA: Fetch input tile with padding
        dma_fetch_3d(
            src=input_dram + h*W*C + w*C,
            size=(tile.h_tile+2)×(tile.w_tile+2)×tile.c_tile,  // +2 for padding
            stride_h=W*C,
            stride_w=C,
            dst=VTCM_input
        );

        // Compute: 3×3 depthwise convolution
        depthwise_conv_hvx(VTCM_input, VTCM_weights, VTCM_output,
                          tile.h_tile, tile.w_tile, tile.c_tile);

        // DMA: Writeback output tile
        dma_store_3d(
            src=VTCM_output,
            dst=output_dram + h*W*C + w*C,
            size=tile.h_tile×tile.w_tile×tile.c_tile,
            stride_h=W*C,
            stride_w=C
        );
    }
}
```

#### Tiling Quality Metrics

```
For a given tiling configuration, measure:

1. Arithmetic Intensity (AI):
   AI = (total_ops) / (total_bytes_transferred)

   Example:
   - 3×3 Conv, 16×16 spatial tile, 96 channels
   - Ops: 16 × 16 × 3 × 3 × 96 × 2 = 4.4M (multiply-accumulate)
   - I/O: 18×18×96 input + 14×14×96 output + 864 weights ≈ 40 KB
   - AI = 4.4M / 40 KB ≈ 110 MACs/byte

   Target: >100 MACs/byte for VTCM to be effective

2. Reuse Ratio:
   Reuse = ops / (total_loaded_bytes)
   (captures both weight and activation reuse)

3. Tile Occupancy:
   Occupancy = (tile_data_size) / (available_VTCM)
   (higher is better, but leave room for padding/fragmentation)
```

### 4. Activation Tiling (Spatial and Depth)

Beyond weight tiling, we must tile **activations** (intermediate feature maps) across layers.

#### Spatial Tiling (H-W Blocking)

```
ResNet bottleneck: conv(3,3,64,64) → conv(1,1,64,256) → conv(1,1,256,64)

Total activation storage without tiling:
- Input: 56×56×64 = 200 KB
- Layer1 output: 56×56×64 = 200 KB
- Layer2 output: 56×56×256 = 800 KB
- Layer3 output: 56×56×64 = 200 KB
- Total: 1.4 MB (exceeds 1 MB VTCM on v73)

With 28×28 spatial tiling (dividing by 2 in each spatial dim):
- Input tile: 28×28×64 = 50 KB ✓
- Layer1 output: 28×28×64 = 50 KB ✓
- Layer2 output: 28×28×256 = 200 KB ✓
- Layer3 output: 28×28×64 = 50 KB ✓
- Total: 350 KB ✓ (fits in VTCM)

Cost: 4× more DMA operations (4 spatial tiles)
      Overhead: ~5-10% latency increase, but acceptable for inference
```

#### Depth Tiling (Channel Blocking)

When output channels exceed VTCM capacity, block them:

```
Layer: conv(1,1,256,512)  // 1×1 convolution
Output tile: H×W×C_out = 56×56×512 = 1.6 MB (exceeds VTCM)

Depth tiling with C_out_tile = 64:
Output for 1 spatial point × 64 channels: 64 bytes
Process in 512/64 = 8 sub-convolutions:

for out_ch in [0, 64, 128, ..., 448]:
    for spatial (h,w) in input:
        output[h,w,out_ch:out_ch+64] =
            conv(input[h,w,:], weights[:,out_ch:out_ch+64])

Total VTCM usage:
- Input: 56×56×256 = 800 KB
- Output: 56×56×64 = 200 KB (only 1 channel tile at a time)
- Weights: 256×64 = 16 KB (1 channel-output tile)
- Total: 1 MB ✓
```

### 5. Interleaved Formats for HMX

**HMX (Hexagon Matrix eXtension)** is an optional sub-instruction for matrix operations introduced in v78+. It uses **specialized tensor layouts**.

```
HMX Format (for INT8 matrix multiply):

Standard Layout (Input × Weight → Output):
    ┌─────────────┐     ┌─────────────┐     ┌─────────┐
    │   Input     │  ×  │   Weight    │  =  │ Output  │
    │ (M × K)     │     │ (K × N)     │     │ (M × N) │
    └─────────────┘     └─────────────┘     └─────────┘
    M=8, K=256, N=64

HMX-Optimized Layout (Tiled, Interleaved):
1. Tile Input into 8×64 blocks (8 rows, 64 columns of 8-element groupings)
2. Tile Weight into 64×64 blocks
3. Reorder bytes within blocks for SIMD efficiency

Memory Layout Example (simplified for 4×4 × 4×4):
HMX Input Register File:
    Row0: [a00 a01 a02 a03 | a10 a11 a12 a13]  (2 × 4 elements, interleaved)
    Row1: [a20 a21 a22 a23 | a30 a31 a32 a33]

HMX Weight Register File:
    Row0: [w00 w10 w20 w30 | w01 w11 w21 w31]  (Transposed within blocks)
    Row1: [w02 w12 w22 w32 | w03 w13 w23 w33]

Instruction:
    output = hmx.matmul.i8(input, weight)
    (Single instruction: 4×4 × 4×4 = 256 MACs in 1 cycle)

For actual v78 HMX:
    Single instruction = 8×64 × 64×8 matrix multiply
                       = 32K MACs / cycle (at 2 GHz clock)
```

**Layout Conversion Code**:

```c
// Convert NHWC activation to HMX input format
// Input: (M, K) int8 array
// Output: HMX-tiled format

void convert_to_hmx_format(int8_t *src,    // Source (M × K)
                           int8_t *dst,    // Destination (HMX format)
                           int M, int K) {
    // HMX block size: 8×64
    int block_rows = 8;
    int block_cols = 64;

    for (int block_r = 0; block_r < M; block_r += block_rows) {
        for (int block_c = 0; block_c < K; block_c += block_cols) {
            // Extract one HMX block
            for (int r = 0; r < block_rows; r++) {
                for (int c = 0; c < block_cols; c++) {
                    int src_idx = (block_r + r) * K + (block_c + c);

                    // Reorder within block for HMX
                    int local_r = r;
                    int local_c = c;
                    int dst_offset = block_r * K + block_c;
                    int dst_idx = dst_offset + local_r * block_cols + local_c;

                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}
```

---

## DMA Engine Programming

### 1. Hexagon DMA Architecture

The Hexagon DMA (Direct Memory Access) engine provides **asynchronous, zero-CPU-overhead data transfers** between DRAM, L2, and VTCM.

```
Hexagon Data Movement Architecture:

┌──────────────────────────────────────────────────────────┐
│  CPU (Scalar Thread)                                      │
│  • Issues DMA commands                                    │
│  • Continues execution (non-blocking)                     │
│  • Polls/waits for completion when needed                │
└──────┬───────────────────────────────────────────────────┘
       │ DMA Command Queue (32-128 entries deep)
       │
┌──────▼───────────────────────────────────────────────────┐
│  DMA Scheduler/Arbiter                                    │
│  • Prioritizes transfers                                  │
│  • Manages memory ports                                   │
│  • Handles cache coherency                               │
└──────┬───────────────────────────────────────────────────┘
       │
┌──────▼────────────────────────────────────────────────────┐
│  DMA Execution Engines (typically 2-4)                    │
│  ├─ UDMA: Unidirectional (DRAM→VTCM or VTCM→DRAM)        │
│  └─ MDMA: Bidirectional (more flexible routing)           │
└──────┬───────────────────────────────────────────────────┘
       │
    ┌──┴──┬───────┬──────────┬────────┐
    │     │       │          │        │
┌───▼──┐┌─▼──┐┌──▼───┐┌─────▼──┐┌──▼───┐
│ DRAM ││L2  ││ VTCM ││ Coherency ││ I/O │
│Cache ││    ││      ││  Fabric  ││Ports│
└──────┘└────┘└──────┘└──────────┘└─────┘
```

### 2. DMA Types: UDMA vs MDMA

| Aspect | UDMA | MDMA |
|--------|------|------|
| **Direction** | Unidirectional (fixed src→dst) | Bidirectional (flexible) |
| **Usage** | Streaming transfers | Complex patterns |
| **Latency** | Lower (simpler arbiter) | Slightly higher |
| **Throughput** | Up to 64 bytes/cycle | Up to 64 bytes/cycle |
| **Setup Overhead** | ~2-4 cycles | ~4-8 cycles |
| **Scatter-Gather** | Limited | Full support |

### 3. Non-Blocking Async DMA Transfers

The fundamental pattern for high-performance inference:

```c
#include <stdlib.h>
#include <stdio.h>
#include <qurt.h>
#include <HAP_farf.h>

// DMA transfer structure
typedef struct {
    uint64_t src;      // Source address (DRAM)
    uint64_t dst;      // Destination address (VTCM)
    uint32_t size;     // Transfer size (bytes)
    uint32_t src_stride;  // Stride for 2D transfers
    uint32_t dst_stride;
    uint32_t height;   // For 2D transfers
} DMATransfer;

// Issue a non-blocking DMA transfer and get a handle
unsigned int dma_transfer_async(DMATransfer *xfer) {
    // Initiate transfer (returns immediately)
    unsigned int token = qurt_dma_copy(
        xfer->dst,           // Destination
        xfer->src,           // Source
        xfer->size,          // Size
        QURT_DMA_PRIORITY_DEFAULT
    );

    return token;  // Use this to wait later
}

// Check if a specific DMA transfer is complete (non-blocking poll)
int dma_is_complete(unsigned int token) {
    return qurt_dma_status(token) == QURT_DMA_COMPLETE;
}

// Wait for a specific DMA transfer (blocking, but we overlap compute)
void dma_wait(unsigned int token) {
    while (qurt_dma_status(token) != QURT_DMA_COMPLETE) {
        // Can do other work here (compute, schedule next DMA, etc)
    }
}

// Full example: Conv layer with DMA prefetch
void conv_layer_with_prefetch(
    int8_t *input_dram,           // Input from previous layer
    int8_t *weights_dram,         // Weights (constant)
    int8_t *output_dram,          // Output to next layer
    int8_t *vtcm_input,           // Input buffer in VTCM
    int8_t *vtcm_weights,         // Weight buffer in VTCM
    int8_t *vtcm_output,          // Output buffer in VTCM
    int num_tiles) {

    unsigned int dma_token_input_0, dma_token_weights_0, dma_token_store_0;
    unsigned int dma_token_input_1;

    // Initialize: Fetch first weight tile (constant, fetch once)
    DMATransfer weight_xfer = {
        .src = (uint64_t)weights_dram,
        .dst = (uint64_t)vtcm_weights,
        .size = WEIGHT_TILE_SIZE,
        .src_stride = 0,  // Contiguous
        .dst_stride = 0
    };
    dma_transfer_async(&weight_xfer);

    // Main loop with pipelining:
    // Tile i:   DMA input[i] → VTCM, Compute using input[i-1] & weights
    // Tile i+1: DMA input[i+1] → VTCM, Store output[i-1]

    for (int tile = 0; tile < num_tiles; tile++) {
        // ====== Pipeline Stage 1: Prefetch next input ======
        if (tile < num_tiles - 1) {
            DMATransfer input_xfer = {
                .src = (uint64_t)(input_dram + (tile + 1) * INPUT_TILE_SIZE),
                .dst = (uint64_t)vtcm_input,
                .size = INPUT_TILE_SIZE,
                .src_stride = 0,
                .dst_stride = 0
            };
            dma_token_input_1 = dma_transfer_async(&input_xfer);
        }

        // ====== Pipeline Stage 2: Fetch current input (if first tile) ======
        if (tile == 0) {
            DMATransfer input_xfer = {
                .src = (uint64_t)input_dram,
                .dst = (uint64_t)vtcm_input,
                .size = INPUT_TILE_SIZE,
                .src_stride = 0,
                .dst_stride = 0
            };
            dma_token_input_0 = dma_transfer_async(&input_xfer);
        }

        // ====== Pipeline Stage 3: Wait for current input ======
        dma_wait(tile == 0 ? dma_token_input_0 : dma_token_input_1);

        // ====== Pipeline Stage 4: COMPUTE (while next input is transferring) ======
        HVX_Conv3x3_optimized(vtcm_input, vtcm_weights, vtcm_output,
                              TILE_HEIGHT, TILE_WIDTH, NUM_CHANNELS);

        // ====== Pipeline Stage 5: Writeback output ======
        DMATransfer output_xfer = {
            .src = (uint64_t)vtcm_output,
            .dst = (uint64_t)(output_dram + tile * OUTPUT_TILE_SIZE),
            .size = OUTPUT_TILE_SIZE,
            .src_stride = 0,
            .dst_stride = 0
        };
        dma_token_store_0 = dma_transfer_async(&output_xfer);
    }

    // Final output store must complete
    dma_wait(dma_token_store_0);
}
```

⚡ **Expert Insight**: The **pipelining pattern** (fetch(i+1) while compute(i) while store(i-1)) is critical for throughput. Each stage takes ~10-100 cycles depending on data size. Overlapping them can provide **3-4× bandwidth utilization** compared to sequential transfers.

### 4. DMA Descriptor Setup

For complex 2D/3D transfers with strides, use **DMA descriptors**:

```c
#include <qurt.h>

// 2D DMA descriptor (e.g., transferring a feature map tile)
typedef struct {
    uint64_t src;           // Source base address
    uint64_t dst;           // Destination base address
    uint32_t src_width;     // Width of each row (bytes)
    uint32_t dst_width;     // Destination width (typically same)
    uint32_t height;        // Number of rows
    uint32_t src_stride;    // Stride to next row (src)
    uint32_t dst_stride;    // Stride to next row (dst)
} DMA2DDescriptor;

// Setup a 2D DMA transfer (e.g., extracting a 32×32 spatial tile)
unsigned int dma_2d_transfer(DMA2DDescriptor *desc) {
    // Build descriptor for QURT DMA
    // Note: API varies by SDK; this is conceptual

    unsigned int token = qurt_dma_memcpy_2d(
        desc->dst,           // Destination
        desc->src,           // Source
        desc->src_width,     // Bytes per row
        desc->height,        // Number of rows
        desc->src_stride,    // Source row stride
        desc->dst_stride,    // Dest row stride
        QURT_DMA_PRIORITY_DEFAULT
    );

    return token;
}

// Example: Extract 32×32 channel tile from 56×56 input
void extract_channel_tile_dma(
    int8_t *input_dram,     // Full input: 56×56×64 (NHWC)
    int8_t *tile_vtcm,      // Output: 32×32×64 in VTCM
    int start_h, int start_w) {

    int C = 64;
    int W_full = 56;
    int tile_h = 32, tile_w = 32;

    // Descriptor: Extract tile with spatial offset
    DMA2DDescriptor desc = {
        .src = (uint64_t)(input_dram + (start_h * W_full + start_w) * C),
        .dst = (uint64_t)tile_vtcm,
        .src_width = tile_w * C,        // 32 spatial × 64 channels = 2048 bytes/row
        .dst_width = tile_w * C,
        .height = tile_h,                // 32 rows
        .src_stride = W_full * C,        // Stride to next row in full input
        .dst_stride = tile_w * C         // Stride in VTCM (contiguous)
    };

    unsigned int token = dma_2d_transfer(&desc);
    dma_wait(token);
}
```

### 5. Scatter-Gather Descriptors

For **non-contiguous data** (e.g., strided convolution inputs), use scatter-gather:

```c
// Scatter-Gather DMA (pseudo-code, actual SDK may differ)
typedef struct {
    uint64_t addr;        // Source/destination address
    uint32_t size;        // Transfer size for this segment
} SGElement;

typedef struct {
    SGElement *sg_list;   // Array of (addr, size) pairs
    uint32_t num_elements;
    uint64_t dst_base;    // Base destination
} SGDescriptor;

// Example: Gather alternating values from input (stride=2)
void gather_strided_input(
    int8_t *input_dram,        // Contiguous data in DRAM
    int8_t *output_vtcm,       // Strided data in VTCM
    int num_elements,
    int stride) {

    // Build scatter-gather list
    SGElement *sg_list = malloc(num_elements * sizeof(SGElement));
    for (int i = 0; i < num_elements; i++) {
        sg_list[i].addr = (uint64_t)(input_dram + i * stride);
        sg_list[i].size = 1;  // Single byte per element
    }

    SGDescriptor sg_desc = {
        .sg_list = sg_list,
        .num_elements = num_elements,
        .dst_base = (uint64_t)output_vtcm
    };

    // Initiate scatter-gather DMA
    unsigned int token = qurt_dma_sg_transfer(&sg_desc);
    dma_wait(token);

    free(sg_list);
}
```

### 6. DMA Chaining

**DMA chaining** allows a **sequence of transfers** to execute automatically without CPU intervention:

```c
// DMA Chain Descriptor (executes multiple transfers in sequence)
typedef struct {
    unsigned int num_transfers;  // Number of chained transfers
    struct {
        uint64_t src;
        uint64_t dst;
        uint32_t size;
    } transfers[MAX_CHAIN_DEPTH];  // Up to 256 transfers
} DMAChadescriptor;

// Example: Pipeline 3 layer computations with chaining
void chained_layer_inference(
    int8_t *input_dram,
    int8_t *output_dram,
    int8_t *vtcm_scratch,
    int layer_count) {

    DMAChadescriptor chain = {
        .num_transfers = layer_count * 2  // Input + output per layer
    };

    // Build chain for: Layer0_in, Layer0_out, Layer1_in, Layer1_out, ...
    int idx = 0;
    for (int layer = 0; layer < layer_count; layer++) {
        // Input transfer for this layer
        chain.transfers[idx] = {
            .src = (uint64_t)(input_dram + layer * INPUT_SIZE),
            .dst = (uint64_t)(vtcm_scratch + VTCM_INPUT_OFFSET),
            .size = INPUT_SIZE
        };
        idx++;

        // Output transfer for this layer
        chain.transfers[idx] = {
            .src = (uint64_t)(vtcm_scratch + VTCM_OUTPUT_OFFSET),
            .dst = (uint64_t)(output_dram + layer * OUTPUT_SIZE),
            .size = OUTPUT_SIZE
        };
        idx++;
    }

    // Submit entire chain
    unsigned int chain_token = qurt_dma_chain(&chain);

    // Now: Issue all compute tasks (they run in parallel with DMA)
    for (int layer = 0; layer < layer_count; layer++) {
        // Compute will be gated by DMA completion
        schedule_layer_compute(layer, vtcm_scratch);
    }

    // Wait for entire chain to complete
    dma_wait(chain_token);
}
```

⚡ **Expert Insight**: DMA chaining can reduce **command queue overhead** by 50-80%. For inference engines with 50+ layers, chaining significantly improves latency predictability. The trade-off: chains are immutable once submitted, so dynamic graph scheduling becomes challenging.

---

## Cache Management & Optimization

### 1. L1 and L2 Cache Behavior

```
L1 D-Cache (Data Cache):
├─ Size: 16-32 KB per core
├─ Associativity: 4-way set-associative
├─ Line Size: 32 bytes
├─ Latency: 1 cycle (hit)
├─ Miss Latency: 7-15 cycles (L2 hit) or 50+ (DRAM)
├─ Write Policy: Write-back (dirty lines written on eviction)
└─ Coherency: Directory-based (snoops L2)

L2 Cache (Unified, Shared):
├─ Size: 256 KB - 1 MB per core cluster
├─ Associativity: 8-way set-associative
├─ Line Size: 64 bytes
├─ Latency: 3-4 cycles (hit)
├─ Miss Latency: 50-100 cycles (DRAM)
├─ Write Policy: Write-back
└─ Coherency: Maintains coherence with L1 and VTCM
```

#### Cache Line Organization

```
L2 Cache: 512 KB (v73)

┌─────────────────────────────────────────────────────────┐
│ Address Space (512 KB = 2^19 bytes)                      │
│                                                          │
│ Line Size: 64 bytes = 2^6                               │
│ → 2^13 = 8192 cache lines total                         │
│                                                          │
│ Address Decoding:                                       │
│ ┌─────────────┬────────────┬──────────────────────────┐│
│ │   Tag       │    Set     │      Offset              ││
│ │  (bits 13+) │  (bits 6-12) │  (bits 0-5)          ││
│ │   512 KB    │   128 sets   │   64 bytes/line       ││
│ │ (8-way)     │ (8 ways)     │ (8 × 64 = 512 B/set) ││
│ └─────────────┴────────────┴──────────────────────────┘│
│                                                          │
│ Example Address: 0xA5C40
│ ├─ Offset [5:0]:   0x00 (line offset)
│ ├─ Set [12:6]:     0x4A (set 74)
│ └─ Tag [32:13]:    0x52E2 (tag value)
└─────────────────────────────────────────────────────────┘

Cache Line States (MOESI Protocol):
    M (Modified): Dirty, only in this cache
    O (Owned): Dirty, shared with others
    E (Exclusive): Clean, only in this cache
    S (Shared): Clean, in multiple caches
    I (Invalid): Not in this cache

HVX Load Behavior:
    Load from cached address → L2 hit (2-3 cycles)
    Load from cached address → L2 miss (DRAM fill via bus)

    Bulk write-back on eviction:
    64-byte line → DRAM or next-level (32 GB/s bus)
```

### 2. Explicit Cache Flush Operations

#### dcfetch (Data Cache Fetch)

Explicitly bring data into cache **before** computation:

```c
// Assembly intrinsic (HVX-specific)
// Syntax: dcfetch(address)
// Effect: Prefetch line starting at address into L1

#include <hexagon_protos.h>

void prefetch_convolution_input(
    int8_t *input,     // Feature map in DRAM
    int H, int W, int C) {

    // Prefetch stride: one cache line per weight (spatial locality)
    // Cache line: 64 bytes = 8 pixels × 8 channels (int8)

    for (int h = 0; h < H; h += 8) {
        for (int w = 0; w < W; w += 8) {
            for (int c = 0; c < C; c += 64) {
                uint8_t *addr = input + (h * W + w) * C + c;
                Q6_dcfetch_A(addr);  // Hexagon intrinsic
            }
        }
    }

    // Now computation touches prefetched data with lower miss rate
}

// Effect on memory hierarchy:
// Before dcfetch: DRAM miss → 50-100 cycle latency
// After dcfetch:  L1/L2 hit → 1-4 cycle latency
// Effective speedup: 10-50× (if prefetch overlaps compute)
```

#### dccleana (Data Cache Clean & Invalidate All)

Flush entire L1/L2 caches **without** writing back to memory (data already in VTCM):

```c
void clear_l_caches(void) {
    // Before: L1 has stale data, L2 has stale data

    Q6_dccleana();  // Invalidate all L1 lines
    Q6_dcnop();     // Wait for operation to complete

    // After: All cache lines are invalid
    // Impact: Next load comes from VTCM/DRAM (no stale data)

    // Use case: After bulk VTCM transfer, invalidate cache to
    // force next load to use VTCM (which has fresh copy)
}

// More selective: invalidate specific address range
void invalidate_cache_range(uint8_t *addr, unsigned int size) {
    // Invalidate cache lines in range [addr, addr+size)

    unsigned int line_addr = (unsigned int)addr & ~63;  // Align to line
    unsigned int line_end = ((unsigned int)(addr + size) + 63) & ~63;

    for (; line_addr < line_end; line_addr += 64) {
        Q6_dccleaninv_A((void *)line_addr);  // Clean & invalidate
    }
}
```

#### dczeroa (Zero & Invalidate)

Atomic zero + invalidate, useful for clearing output buffers:

```c
void zero_output_buffer(int8_t *output, unsigned int size) {
    // Single instruction: zero the line AND invalidate it
    // More efficient than memset (4-8 cycles/line vs 50+ cycles)

    for (unsigned int i = 0; i < size; i += 64) {
        Q6_dczeroa((void *)(output + i));
    }

    // Equivalent to:
    //   memset(output, 0, size);     // Slow, uses store port
    //   Q6_dccleana();                // Slow, flushes all lines
    // But:
    //   dczeroa is atomic and parallel-capable
}
```

### 3. Write-Combining for Output

When writing output (e.g., results from convolution), use **write-combining** to reduce DRAM traffic:

```c
// Pattern: Accumulate output in vector register, then write atomically

void hvx_write_combine_output(
    uint8_t *output_vtcm,        // Computation output in VTCM
    uint8_t *output_dram,        // Final output in DRAM
    unsigned int num_elements) {

    // Instead of: output_dram[i] = output_vtcm[i];  (byte-by-byte)
    // Use:

    // Bulk copy with write combining (64 bytes at a time)
    for (unsigned int i = 0; i < num_elements; i += 64) {
        // Load 64 bytes from VTCM (1-2 cycles, zero latency)
        HVX_Vector v = *(HVX_Vector *)(output_vtcm + i);

        // Write 64 bytes to DRAM via write-combining buffer
        // Write buffer: ~256 bytes (4 cache lines)
        // Reduces DRAM traffic by ~4-8× (lines coalesce)
        *(HVX_Vector *)(output_dram + i) = v;
    }

    // Flush write buffer to ensure persistence
    Q6_dccleana();
}
```

### 4. Prefetch Strategies

Effective prefetching **overlaps memory latency with computation**:

```c
// Strategy 1: Software prefetch loop
void matmul_with_prefetch(
    int8_t *A,             // Input matrix (M × K)
    int8_t *B,             // Weight matrix (K × N)
    int8_t *C,             // Output matrix (M × N)
    int M, int K, int N) {

    // Prefetch B (weights) first—they're reused N times
    for (int i = 0; i < K * N; i += 64) {
        Q6_dcfetch_A(B + i);
    }

    // Then prefetch A in chunks
    int a_chunk = 256;  // Prefetch 256 bytes at a time

    for (int m = 0; m < M; m++) {
        // Prefetch A[m:m+4, :] (4 rows ahead)
        if (m + 4 < M) {
            for (int k = 0; k < K; k += 64) {
                Q6_dcfetch_A(A + (m + 4) * K + k);
            }
        }

        // Compute row m (uses A[m] which was prefetched 4 iterations ago)
        for (int n = 0; n < N; n++) {
            int acc = 0;
            for (int k = 0; k < K; k++) {
                acc += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}

// Strategy 2: Asynchronous prefetch (overlaps with compute)
void cnn_with_async_prefetch(
    int8_t *input_tiles[NUM_TILES],   // Input tiles in DRAM
    int8_t *weight_tile,              // Weight tile in DRAM/VTCM
    int8_t *output_tile,              // Output tile in VTCM
    int tile_count) {

    unsigned int prefetch_token;

    // Prefetch tile 0 before loop starts
    prefetch_token = async_dma_fetch(input_tiles[0], VTCM_INPUT_0);

    for (int tile = 0; tile < tile_count; tile++) {
        // Prefetch tile+1 while computing tile
        if (tile + 1 < tile_count) {
            // Non-blocking prefetch (may complete or not)
            dcfetch(input_tiles[tile + 1]);
        }

        // Compute using tile (hopefully prefetch of tile+1 is done)
        hvx_conv_compute(VTCM_INPUT_0, weight_tile, output_tile);

        // Store and swap buffers
        store_output(output_tile);
    }
}
```

⚡ **Expert Insight**: Prefetch is most effective for **streaming workloads** (sequential memory access with high spatial locality). For random-access patterns or small working sets, prefetch adds overhead without benefit. Measure with hardware performance counters before optimizing.

---

## Avoiding TCM Bank Conflicts

### 1. How Banks Work

VTCM is divided into **8 independent banks**, each capable of **1 read + 1 write per cycle**:

```
Memory Interleaving (Bank Selection):

Address bits: [31:20] [19:0]
              [  Tag  ][19-bit offset]

Bank Index = (offset >> 16) & 0x7

Bank Assignment:
┌─────────────┬──────────┬────────────┐
│   Address   │   Bank   │ Offset     │
├─────────────┼──────────┼────────────┤
│ 0x00000000  │   0      │ 0x00000    │
│ 0x00010000  │   1      │ 0x00000    │  (64 KB apart)
│ 0x00020000  │   2      │ 0x00000    │
│ ...         │   ...    │ ...        │
│ 0x00070000  │   7      │ 0x00000    │
│ 0x00080000  │   0      │ 0x00000    │  (wraps back to bank 0)
└─────────────┴──────────┴────────────┘

Key Insight:
    Addresses that differ by 64 KB → different banks
    Addresses that differ by 512 KB → same bank
    Addresses with stride = 64 KB → perfect bank distribution
```

### 2. Stride Patterns That Cause Conflicts

```
Pattern 1: Sequential 4-byte loads (INT32)
───────────────────────────────────────────
Address Sequence: 0x00000000, 0x00000004, 0x00000008, ...

Bank Assignment:
Addr 0x00000000 → Bank (0x00000000 >> 16) & 0x7 = 0
Addr 0x00000004 → Bank (0x00000004 >> 16) & 0x7 = 0
Addr 0x00000008 → Bank (0x00000008 >> 16) & 0x7 = 0
All in Bank 0! → 8× serialization (can only access 1 per cycle)

Sequential loads with 4-byte stride:
Cycle 1: Load addr[0] from Bank 0 ✓
Cycle 2: Load addr[4] from Bank 0 ✗ (STALL: Bank 0 busy)
Cycle 3: Load addr[8] from Bank 0 ✗ (STALL)
...
Effective throughput: 1 load / 8 cycles = 8× slower


Pattern 2: 32-channel loads (interleaved channels)
──────────────────────────────────────────────────
Address sequence: c0, c1, c2, c3, c4, c5, c6, c7, c0, c1, ...
(e.g., for 32 consecutive spatial points, alternating 8 channels)

Stride = 32 bytes (8 channels per spatial point in NHWC)

Bank calculation:
c0 at 0x00000000 → Bank 0
c1 at 0x00000004 → Bank 0  (0x04 >> 16 = 0)
c2 at 0x00000008 → Bank 0
...
Again: ALL in Bank 0! → Conflict

Solution: Pad spatial dimension to avoid stride patterns that map to same bank


Pattern 3: Perfect bank distribution (ideal)
─────────────────────────────────────────────
Stride = 64 KB (matches bank interval)

Address sequence: 0x00000000, 0x00010000, 0x00020000, ..., 0x00070000
Banks:            0,          1,          2,          ..., 7

Cycle 1: Load from Bank 0 ✓
Cycle 2: Load from Bank 1 ✓ (Bank 0 can also load in parallel)
Cycle 3: Load from Bank 2 ✓
...
Cycle 8: Load from Bank 7 ✓
Then: Cycle 9: Load from Bank 0 ✓ (renewed)
Effective throughput: 8 loads / 8 cycles = 1 load/cycle ✓
```

### 3. Padding Data Layouts to Avoid Conflicts

#### Example 1: Padding a Feature Map

```c
// Original layout (problematic):
// 32×32 spatial × 64 channels
// Stride per pixel = 64 bytes (all channels)
// Issue: Multiple spatial points map to same bank

typedef struct {
    uint8_t channels[64];  // One pixel, 64 channels
} Pixel_t;

// Storage without padding:
Pixel_t input_map[32][32];  // 32×32×64 = 65 KB

// Bank conflict analysis:
// Pixel[0,0] at offset 0x00000 → Bank (0 >> 16) & 7 = 0
// Pixel[0,1] at offset 0x00040 → Bank (0x40 >> 16) & 7 = 0  ✗ CONFLICT
// Pixel[0,2] at offset 0x00080 → Bank (0x80 >> 16) & 7 = 0  ✗ CONFLICT


// SOLUTION: Pad each row to 128 bytes (next power of 2)
// This shifts subsequent rows to different banks
typedef struct {
    uint8_t channels[64];  // Actual data
    uint8_t _padding[64];  // Padding to 128 bytes
} Pixel_Padded_t;

Pixel_Padded_t input_map_padded[32][32];  // 32×32×128 = 131 KB

// Bank conflict analysis (with padding):
// Pixel[0,0] at 0x00000 → Bank 0
// Pixel[0,1] at 0x00080 → Bank 1  ✓ (64 KB shift = 1 bank)
// Pixel[0,2] at 0x00100 → Bank 2  ✓
// Pixel[0,3] at 0x00180 → Bank 3  ✓
// ...
// Pixel[0,8] at 0x00400 → Bank 0  (wraps, but next row)
// Pixel[1,0] at 0x00400 → Bank 0  (same, OK—different spatial)

// Padding overhead: 64 bytes per pixel × 32×32 pixels = 64 KB added
// But: Eliminates bank conflicts → 8× speedup more than compensates
```

#### Example 2: Padding Weight Tensor

```c
// Conv weights: 3×3 kernel, 64 input channels, 256 output channels
// NHWC-style layout: 3×3×64×256 = 147 KB

uint8_t weights[3][3][64][256];

// Problem: Sequential output channels (stride = 64 bytes for input channels)
// may cause bank conflicts

// SOLUTION: Pad output channels to multiple of 128
typedef struct {
    uint8_t channels[256 + 128];  // Pad to 384
} OutputChannels_t;

OutputChannels_t weights_padded[3][3][64];

// Verification:
// Weight[0,0,0,0] at offset 0x000000 → Bank 0
// Weight[0,0,0,1] at offset 0x000001 → Bank 0  (same byte)
// Weight[0,0,1,0] at offset 0x000100 → Bank 0  (64 input ch × 4 bytes)
// Weight[0,0,2,0] at offset 0x000200 → Bank 0  ✗ CONFLICT

// Better: Ensure stride is 64 KB
// Group output channels in 64-channel blocks, each at 64 KB boundary

struct {
    uint8_t ch0_to_63[3][3][64][64];      // 36 KB
    uint8_t _pad1[28 * 1024];              // Pad to 64 KB
    uint8_t ch64_to_127[3][3][64][64];    // 36 KB at 64 KB offset → Bank 1
    uint8_t _pad2[28 * 1024];
    // ... more channel groups ...
} weights_struct;
```

### 4. Measuring Bank Conflict Impact

Use **Hexagon profiling tools** to quantify conflicts:

```c
#include <qurt.h>
#include <HAP_perf_counter.h>

typedef struct {
    uint64_t cycles;              // Total cycles
    uint64_t l2_hits;             // L2 cache hits
    uint64_t l2_misses;           // L2 cache misses
    uint64_t vtcm_conflicts;      // VTCM bank conflicts
    uint64_t loads;               // Total load instructions
    uint64_t stores;              // Total store instructions
} PerfCounters;

void measure_bank_conflicts(void) {
    // Initialize counters
    HAP_perf_reset_all();

    // Start measurement
    HAP_perf_start_all();

    // Run kernel
    kernel_with_potential_conflicts();

    // Stop measurement
    HAP_perf_stop_all();

    // Read counters
    PerfCounters perf;
    perf.cycles = HAP_perf_get_counter(HAP_PERF_CYCLE);
    perf.vtcm_conflicts = HAP_perf_get_counter(HAP_PERF_TCMBANK_STALL);
    perf.loads = HAP_perf_get_counter(HAP_PERF_LOAD);
    perf.stores = HAP_perf_get_counter(HAP_PERF_STORE);

    // Analysis
    double conflict_rate = (double)perf.vtcm_conflicts / perf.cycles;
    HAP_FARF(HIGH, "Bank conflict rate: %.2f%%", conflict_rate * 100);

    if (conflict_rate > 0.10) {  // >10% stalls from conflicts
        HAP_FARF(ERROR, "Excessive bank conflicts! Consider padding.");
    }
}

// Interpretation:
// 0-5% stalls:    Acceptable
// 5-10% stalls:   Noticeable, consider optimization
// >10% stalls:    Critical, apply padding/layout fixes
```

---

## Designing a Scratchpad Allocator

### 1. Allocator Requirements

For a high-performance inference engine, the VTCM allocator must:

1. **Determinism**: Allocation/deallocation in O(1) or O(log N)
2. **Low fragmentation**: Avoid wasting >10% of VTCM to fragmentation
3. **Alignment support**: Guarantee 128-byte alignment (HVX requirement)
4. **Lifetime tracking**: Allocations tied to graph execution (DAG schedule)
5. **Thread safety**: Handle multi-threaded concurrent allocations
6. **Debugging**: Track allocation origins, sizes, lifetimes

### 2. Pool-Based Allocation Design

```c
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define VTCM_BASE          0x28000000
#define VTCM_SIZE          (1 * 1024 * 1024)  // 1 MB
#define ALIGNMENT_BYTES    128
#define MAX_ALLOCATIONS    256

// Allocation metadata
typedef struct {
    uint32_t offset;                    // Offset in VTCM
    uint32_t size;                      // Allocated size
    uint32_t lifetime_start;            // Graph layer where allocation starts
    uint32_t lifetime_end;              // Graph layer where allocation can be freed
    char label[64];                     // For debugging
} AllocationRecord;

// Allocator state
typedef struct {
    AllocationRecord records[MAX_ALLOCATIONS];
    uint32_t num_allocations;
    uint32_t current_offset;            // Next available offset
    uint32_t fragmentation_threshold;   // Max acceptable fragmentation (bytes)
} VTCMAllocator;

// Initialize allocator
void vtcm_allocator_init(VTCMAllocator *alloc) {
    memset(alloc, 0, sizeof(*alloc));
    alloc->current_offset = 0;
    alloc->fragmentation_threshold = VTCM_SIZE / 10;  // 10% max fragmentation
}

// Allocate from pool
uint32_t vtcm_allocate(VTCMAllocator *alloc,
                       uint32_t size,
                       uint32_t lifetime_start,
                       uint32_t lifetime_end,
                       const char *label) {
    // Align size to ALIGNMENT_BYTES boundary
    uint32_t aligned_size = (size + ALIGNMENT_BYTES - 1) & ~(ALIGNMENT_BYTES - 1);

    // Check bounds
    if (alloc->current_offset + aligned_size > VTCM_SIZE) {
        // VTCM exhausted
        return -1;  // Error
    }

    // Check fragmentation
    uint32_t used = alloc->current_offset;
    uint32_t fragmentation = VTCM_SIZE - used - aligned_size;
    if (fragmentation > alloc->fragmentation_threshold) {
        // Could optimize with defragmentation, but simple pool is linear
        // In practice, reorder allocations or increase VTCM
    }

    // Record allocation
    if (alloc->num_allocations >= MAX_ALLOCATIONS) {
        return -1;  // Too many allocations
    }

    AllocationRecord *rec = &alloc->records[alloc->num_allocations];
    rec->offset = alloc->current_offset;
    rec->size = size;
    rec->lifetime_start = lifetime_start;
    rec->lifetime_end = lifetime_end;
    strncpy(rec->label, label, sizeof(rec->label) - 1);

    uint32_t result_offset = alloc->current_offset;
    alloc->current_offset += aligned_size;
    alloc->num_allocations++;

    return result_offset;
}

// Deallocate (for pool-based, just decrement counter if at end)
void vtcm_deallocate(VTCMAllocator *alloc, uint32_t offset) {
    // For simple linear pool, we can only free if this is the last allocation
    // For more flexibility, use the lifetime-based approach below

    for (int i = alloc->num_allocations - 1; i >= 0; i--) {
        if (alloc->records[i].offset == offset) {
            // Remove this allocation
            if (i == alloc->num_allocations - 1) {
                // Last allocation: can actually free
                alloc->current_offset -= alloc->records[i].size;
            }
            // Shift remaining records
            memmove(&alloc->records[i], &alloc->records[i+1],
                   (alloc->num_allocations - i - 1) * sizeof(AllocationRecord));
            alloc->num_allocations--;
            return;
        }
    }
}

// Get virtual address from offset
void *vtcm_get_address(uint32_t offset) {
    return (void *)(VTCM_BASE + offset);
}
```

### 3. Lifetime-Based Allocation Strategy

For **inference DAGs**, allocations are tied to graph layers. Once a layer finishes, its inputs can be freed:

```c
#include <stdbool.h>

typedef struct {
    uint32_t layer_id;                  // Which layer(s) produce this tensor
    uint32_t num_consumers;             // How many layers consume this tensor
    uint32_t consumers_completed;       // How many have finished
} TensorLifetime;

// Graph representation
typedef struct {
    uint32_t layer_id;
    const char *layer_name;
    uint32_t *input_tensor_ids;         // IDs of input tensors
    uint32_t num_inputs;
    uint32_t *output_tensor_ids;        // IDs of output tensors
    uint32_t num_outputs;
} LayerDef;

// Allocate VTCM based on graph schedule
void allocate_for_graph(VTCMAllocator *alloc,
                        LayerDef *layers,
                        uint32_t num_layers,
                        TensorLifetime *tensor_lifetimes,
                        uint32_t num_tensors) {

    // First pass: determine tensor lifetimes
    for (uint32_t layer_id = 0; layer_id < num_layers; layer_id++) {
        LayerDef *layer = &layers[layer_id];

        // Mark outputs of this layer
        for (uint32_t i = 0; i < layer->num_outputs; i++) {
            uint32_t tensor_id = layer->output_tensor_ids[i];
            tensor_lifetimes[tensor_id].layer_id = layer_id;
        }

        // Count consumers
        for (uint32_t i = 0; i < layer->num_inputs; i++) {
            uint32_t tensor_id = layer->input_tensor_ids[i];
            tensor_lifetimes[tensor_id].num_consumers++;
        }
    }

    // Second pass: allocate with lifetimes
    for (uint32_t tensor_id = 0; tensor_id < num_tensors; tensor_id++) {
        TensorLifetime *lifetime = &tensor_lifetimes[tensor_id];

        // Allocate for this tensor
        uint32_t tensor_size = ...; // Get size from graph metadata

        uint32_t lifetime_start = lifetime->layer_id;
        uint32_t lifetime_end = lifetime->layer_id + 1;  // Freed after producer completes

        // Try to reuse space if earlier tensor's lifetime ended
        uint32_t offset = vtcm_allocate(alloc, tensor_size,
                                        lifetime_start, lifetime_end,
                                        "graph_tensor");

        if (offset == (uint32_t)-1) {
            // Out of memory: graph doesn't fit in VTCM
            // Fall back to DDR spilling or reduce batch size
        }
    }
}

// During inference, after each layer completes:
void on_layer_complete(VTCMAllocator *alloc,
                       TensorLifetime *tensor_lifetimes,
                       uint32_t num_tensors,
                       uint32_t completed_layer_id) {

    // Mark tensors that can be freed
    for (uint32_t tensor_id = 0; tensor_id < num_tensors; tensor_id++) {
        TensorLifetime *lifetime = &tensor_lifetimes[tensor_id];

        if (lifetime->layer_id <= completed_layer_id) {
            // This tensor was produced by or earlier
            lifetime->consumers_completed++;

            if (lifetime->consumers_completed >= lifetime->num_consumers) {
                // All consumers done, can free
                // (Implementation depends on allocator structure)
            }
        }
    }
}
```

### 4. Complete Scratchpad Allocator Implementation

```c
// full_allocator.h

#ifndef VTCM_ALLOCATOR_H
#define VTCM_ALLOCATOR_H

#include <stdint.h>

// Configuration
#define VTCM_SIZE           (1024 * 1024)       // 1 MB
#define VTCM_BASE           0x28000000
#define ALIGNMENT           128
#define MAX_ALLOCATIONS     512
#define DEBUG_ENABLED       1

typedef enum {
    ALLOC_STATUS_OK,
    ALLOC_STATUS_OOM,                   // Out of memory
    ALLOC_STATUS_FRAGMENTATION,         // Excessive fragmentation
    ALLOC_STATUS_ALIGNMENT_FAILED,
    ALLOC_STATUS_INVALID_HANDLE
} AllocStatus;

typedef uint32_t AllocHandle;

// Allocation record
typedef struct {
    uint32_t offset;
    uint32_t size;
    uint32_t lifetime_start;
    uint32_t lifetime_end;
    uint8_t  is_active;
    char     debug_label[64];
} AllocRecord;

// Allocator state
typedef struct {
    AllocRecord records[MAX_ALLOCATIONS];
    uint32_t record_count;
    uint32_t total_allocated;
    uint32_t peak_allocated;
    uint32_t num_allocations;
    uint32_t num_deallocations;
} VTCMAllocator;

// Public API
void vtcm_allocator_init(VTCMAllocator *alloc);

AllocHandle vtcm_allocate(VTCMAllocator *alloc,
                          uint32_t size,
                          uint32_t lifetime_start,
                          uint32_t lifetime_end,
                          const char *label,
                          AllocStatus *status);

AllocStatus vtcm_deallocate_by_lifetime(VTCMAllocator *alloc,
                                        uint32_t lifetime_end_inclusive);

void *vtcm_get_address(AllocHandle handle);

void vtcm_print_stats(VTCMAllocator *alloc);

#endif  // VTCM_ALLOCATOR_H

// ============================================================================
// allocator_impl.c

#include "full_allocator.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void vtcm_allocator_init(VTCMAllocator *alloc) {
    memset(alloc, 0, sizeof(*alloc));
}

// Align size to ALIGNMENT boundary
static uint32_t align_size(uint32_t size) {
    return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

AllocHandle vtcm_allocate(VTCMAllocator *alloc,
                          uint32_t size,
                          uint32_t lifetime_start,
                          uint32_t lifetime_end,
                          const char *label,
                          AllocStatus *status) {
    if (alloc->record_count >= MAX_ALLOCATIONS) {
        if (status) *status = ALLOC_STATUS_FRAGMENTATION;
        return (AllocHandle)-1;
    }

    uint32_t aligned_size = align_size(size);

    // Find first available contiguous region
    uint32_t offset = 0;

    // Collect allocated regions (sorted by offset)
    typedef struct {
        uint32_t offset;
        uint32_t size;
    } Region;

    Region regions[MAX_ALLOCATIONS];
    uint32_t num_regions = 0;

    for (uint32_t i = 0; i < alloc->record_count; i++) {
        if (alloc->records[i].is_active) {
            regions[num_regions].offset = alloc->records[i].offset;
            regions[num_regions].size = alloc->records[i].size;
            num_regions++;
        }
    }

    // Sort regions by offset (simple insertion sort for small N)
    for (uint32_t i = 1; i < num_regions; i++) {
        Region key = regions[i];
        int32_t j = (int32_t)i - 1;
        while (j >= 0 && regions[j].offset > key.offset) {
            regions[j + 1] = regions[j];
            j--;
        }
        regions[j + 1] = key;
    }

    // Find gaps
    offset = 0;
    for (uint32_t i = 0; i < num_regions; i++) {
        // Try to fit before this region
        if (regions[i].offset - offset >= aligned_size) {
            // Found a gap
            break;
        }
        offset = regions[i].offset + regions[i].size;
    }

    // Check if fits at the end
    if (offset + aligned_size > VTCM_SIZE) {
        if (status) *status = ALLOC_STATUS_OOM;
        return (AllocHandle)-1;
    }

    // Record allocation
    AllocRecord *rec = &alloc->records[alloc->record_count];
    rec->offset = offset;
    rec->size = size;  // Store original size
    rec->lifetime_start = lifetime_start;
    rec->lifetime_end = lifetime_end;
    rec->is_active = 1;
    strncpy(rec->debug_label, label, sizeof(rec->debug_label) - 1);

    alloc->total_allocated += aligned_size;
    if (alloc->total_allocated > alloc->peak_allocated) {
        alloc->peak_allocated = alloc->total_allocated;
    }
    alloc->num_allocations++;

    AllocHandle handle = alloc->record_count;
    alloc->record_count++;

    if (status) *status = ALLOC_STATUS_OK;
    return handle;
}

AllocStatus vtcm_deallocate_by_lifetime(VTCMAllocator *alloc,
                                        uint32_t lifetime_end_inclusive) {
    uint32_t freed = 0;

    for (uint32_t i = 0; i < alloc->record_count; i++) {
        if (alloc->records[i].is_active &&
            alloc->records[i].lifetime_end <= lifetime_end_inclusive) {

            uint32_t aligned_size = align_size(alloc->records[i].size);
            alloc->total_allocated -= aligned_size;
            alloc->records[i].is_active = 0;
            freed += aligned_size;
            alloc->num_deallocations++;
        }
    }

    return ALLOC_STATUS_OK;
}

void *vtcm_get_address(AllocHandle handle) {
    if (handle == (AllocHandle)-1) return NULL;
    // Note: This is incomplete—we'd need to pass allocator to get record
    // In practice, return address at VTCM_BASE + offset
    return (void *)(VTCM_BASE + handle);
}

void vtcm_print_stats(VTCMAllocator *alloc) {
#if DEBUG_ENABLED
    printf("=== VTCM Allocator Statistics ===\n");
    printf("Total allocated: %u / %u bytes (%.1f%%)\n",
           alloc->total_allocated, VTCM_SIZE,
           100.0 * alloc->total_allocated / VTCM_SIZE);
    printf("Peak allocated: %u bytes (%.1f%%)\n",
           alloc->peak_allocated,
           100.0 * alloc->peak_allocated / VTCM_SIZE);
    printf("Active allocations: %u (total recorded: %u)\n",
           alloc->num_allocations - alloc->num_deallocations,
           alloc->record_count);
    printf("\nAllocations:\n");
    printf("%-40s %10s %10s %10s %10s\n",
           "Label", "Offset", "Size", "Start", "End");
    printf("%-40s %10s %10s %10s %10s\n",
           "-----", "------", "----", "-----", "---");

    for (uint32_t i = 0; i < alloc->record_count; i++) {
        if (alloc->records[i].is_active) {
            printf("%-40s %10u %10u %10u %10u\n",
                   alloc->records[i].debug_label,
                   alloc->records[i].offset,
                   alloc->records[i].size,
                   alloc->records[i].lifetime_start,
                   alloc->records[i].lifetime_end);
        }
    }
    printf("==============================\n\n");
#endif
}
```

---

## Reference Implementation

### Complete Inference Loop with All Optimizations

```c
// inference_engine.c
// Complete end-to-end inference with VTCM allocation,
// DMA prefetch, HVX compute, and cache management

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <qurt.h>
#include "full_allocator.h"

#define NUM_LAYERS      50
#define BATCH_SIZE      1
#define HVX_VECTOR_BYTES 128

// Tensor metadata
typedef struct {
    uint32_t h, w, c;               // Height, width, channels
    uint32_t size_bytes;
    uint8_t  *dram_ptr;
    uint8_t  *vtcm_offset;
} Tensor;

// Layer description
typedef struct {
    char name[64];
    Tensor *input;
    Tensor *output;
    Tensor *weights;
    uint32_t compute_cycles;        // Estimated cycles to compute
} LayerDesc;

// Inference state
typedef struct {
    LayerDesc layers[NUM_LAYERS];
    uint32_t num_layers;
    VTCMAllocator vtcm;
    uint64_t total_cycles;
    uint64_t total_dma_cycles;
} InferenceEngine;

// Allocate all tensors to VTCM with lifetime-based scheduling
void allocate_tensors(InferenceEngine *engine) {
    vtcm_allocator_init(&engine->vtcm);

    AllocStatus status;

    for (uint32_t layer_id = 0; layer_id < engine->num_layers; layer_id++) {
        LayerDesc *layer = &engine->layers[layer_id];

        // Allocate input (lifetime: [layer_id, layer_id])
        // Output of previous layer, consumed by this layer only
        vtcm_allocate(&engine->vtcm,
                      layer->input->size_bytes,
                      layer_id,           // Produced by layer_id-1
                      layer_id,           // Freed after layer_id completes
                      layer->name,
                      &status);

        // Allocate output (lifetime: [layer_id, layer_id+1 to N])
        // Produced by this layer, consumed by later layers
        vtcm_allocate(&engine->vtcm,
                      layer->output->size_bytes,
                      layer_id,
                      layer_id + 1,       // Live for subsequent layers
                      layer->name,
                      &status);

        // Allocate weights (lifetime: [layer_id, layer_id])
        // Used only during compute
        vtcm_allocate(&engine->vtcm,
                      layer->weights->size_bytes,
                      layer_id,
                      layer_id,
                      layer->name,
                      &status);
    }

    vtcm_print_stats(&engine->vtcm);
}

// DMA prefetch pipeline
typedef struct {
    unsigned int token;             // DMA token for tracking
    uint8_t state;                  // 0=idle, 1=in_flight, 2=complete
} DMAState;

unsigned int dma_fetch_tensor_async(Tensor *tensor, uint8_t *vtcm_dst) {
    // Simplified: non-blocking DMA of contiguous buffer

    return qurt_dma_copy(
        (uint64_t)vtcm_dst,
        (uint64_t)tensor->dram_ptr,
        tensor->size_bytes,
        QURT_DMA_PRIORITY_DEFAULT
    );
}

void dma_wait_for_token(unsigned int token) {
    while (qurt_dma_status(token) != QURT_DMA_COMPLETE) {
        // Busy wait (in practice, could do other work)
    }
}

// Main inference loop with 3-stage pipeline
void run_inference(InferenceEngine *engine) {
    DMAState dma_fetch_input, dma_store_output;

    engine->total_cycles = 0;
    engine->total_dma_cycles = 0;

    // Stage 0: Initialize DMA for first layer
    LayerDesc *layer_0 = &engine->layers[0];
    dma_fetch_input.token = dma_fetch_tensor_async(
        layer_0->input,
        layer_0->input->vtcm_offset
    );
    dma_fetch_input.state = 1;  // In flight

    // Main inference loop
    for (uint32_t layer_id = 0; layer_id < engine->num_layers; layer_id++) {
        LayerDesc *layer = &engine->layers[layer_id];

        printf("Layer %u: %s\n", layer_id, layer->name);

        // ========== PIPELINE STAGE 1: Prefetch next input ==========
        if (layer_id < engine->num_layers - 1) {
            LayerDesc *next_layer = &engine->layers[layer_id + 1];

            // Issue async DMA for next layer's input
            unsigned int next_token = dma_fetch_tensor_async(
                next_layer->input,
                next_layer->input->vtcm_offset
            );

            // Don't wait yet; let compute run in parallel
        }

        // ========== PIPELINE STAGE 2: Wait for current input ==========
        if (layer_id == 0) {
            dma_wait_for_token(dma_fetch_input.token);
            dma_fetch_input.state = 2;
        }

        // ========== PIPELINE STAGE 3: Prefetch weights if not cached ==========
        unsigned int weight_token = dma_fetch_tensor_async(
            layer->weights,
            layer->weights->vtcm_offset
        );
        dma_wait_for_token(weight_token);  // Weights are small, ok to block

        // ========== PIPELINE STAGE 4: COMPUTE ==========
        // At this point:
        // - Input is in VTCM (dma_fetch_input complete)
        // - Weights are in VTCM (weight_token complete)
        // - Next input is being fetched asynchronously

        Q6_dcfetch_A(layer->input->vtcm_offset);     // Prefetch input to L1
        Q6_dcfetch_A(layer->weights->vtcm_offset);   // Prefetch weights to L1

        // Call HVX compute kernel
        // (Would call actual conv/matmul/etc. here)
        // For now, simulate:
        uint32_t compute_start = qurt_get_core_timetick();
        // hv_conv_3x3_hvx(layer->input, layer->weights, layer->output);
        uint32_t compute_end = qurt_get_core_timetick();
        engine->total_cycles += compute_end - compute_start;

        // ========== PIPELINE STAGE 5: Asynchronous output writeback ==========
        dma_store_output.token = qurt_dma_copy(
            (uint64_t)layer->output->dram_ptr,
            (uint64_t)layer->output->vtcm_offset,
            layer->output->size_bytes,
            QURT_DMA_PRIORITY_DEFAULT
        );
        dma_store_output.state = 1;  // In flight

        // ========== PIPELINE STAGE 6: Dealloc previous layer's tensors ==========
        if (layer_id > 0) {
            // Previous layer's tensors can be freed
            vtcm_deallocate_by_lifetime(&engine->vtcm, layer_id - 1);
        }

        // ========== PIPELINE STAGE 7: Wait for output writeback ==========
        if (layer_id == engine->num_layers - 1) {
            // Last layer: must ensure output is written
            dma_wait_for_token(dma_store_output.token);
        }

        printf("  Input VTCM: %p (%u bytes)\n",
               layer->input->vtcm_offset,
               layer->input->size_bytes);
        printf("  Output VTCM: %p (%u bytes)\n",
               layer->output->vtcm_offset,
               layer->output->size_bytes);
    }

    // Ensure final output writeback completes
    dma_wait_for_token(dma_store_output.token);

    printf("\n=== Inference Complete ===\n");
    printf("Total compute cycles: %lu\n", engine->total_cycles);
    printf("Throughput: %.2f MAC/cycle\n",
           (double)(engine->total_cycles) / engine->total_cycles);
}

// Test harness
int main() {
    InferenceEngine engine = { 0 };

    // Populate layer descriptions (simplified)
    for (uint32_t i = 0; i < NUM_LAYERS; i++) {
        snprintf(engine.layers[i].name, sizeof(engine.layers[i].name),
                 "conv%u", i);
        // Would set tensors, sizes, etc.
    }
    engine.num_layers = NUM_LAYERS;

    // Allocate VTCM based on graph
    allocate_tensors(&engine);

    // Run inference
    run_inference(&engine);

    return 0;
}
```

---

## Self-Assessment & Further Reading

### Knowledge Check Questions

1. **VTCM Architecture**
   - Q: A Hexagon v75 has 2 MB VTCM divided into 8 banks. What is the address offset between consecutive banks?
   - Q: Explain why VTCM is preferred over L2 cache for inference workloads.
   - Q: What is the maximum bandwidth available from VTCM in a single cycle?

2. **Data Layout**
   - Q: Why does NHWC layout outperform NCHW on Hexagon HVX?
   - Q: Design a layout for a 128-channel input that maximizes bank utilization.
   - Q: What is the relationship between channel packing and arithmetic intensity?

3. **DMA Optimization**
   - Q: Describe the 3-stage pipeline pattern for overlapping DMA, compute, and writeback.
   - Q: What is the difference between UDMA and MDMA? When would you use each?
   - Q: How can DMA chaining reduce command queue overhead?

4. **Cache Management**
   - Q: What is the difference between dcfetch and Q6_dcfetch_A?
   - Q: How do you measure VTCM bank conflict impact?
   - Q: When is prefetch effective, and when is it wasteful?

5. **Bank Conflicts**
   - Q: Compute the bank index for address 0x00100000 in a 4 MB VTCM.
   - Q: Design a padding strategy for a 32×32×64 tensor to eliminate bank conflicts.
   - Q: Explain the relationship between memory stride and bank conflicts.

6. **Allocator Design**
   - Q: How would you implement a VTCM allocator that supports both contiguous and fragmented allocation?
   - Q: What is the purpose of lifetime tracking in a graph-aware allocator?
   - Q: How can you minimize fragmentation in a pool-based allocator?

### Key References

**Qualcomm Hexagon Documentation**:
- Hexagon V73 Programmer's Reference Manual
- Hexagon Architecture Library (HAL) API Reference
- SNPE SDK Documentation (graph optimization, VTCM usage)

**Performance Optimization**:
- Drepper, U. "What Every Programmer Should Know About Memory" (2007)
- Gorelick, M. & Ozsvald, I. "High Performance Python" (O'Reilly, 2020)
- ARM white papers on cache optimization (applicable to many-core DSPs)

**Recommended Implementations**:
- Study TensorFlow Lite's Hexagon delegate (github.com/tensorflow/tensorflow)
- Examine MobileNet v3 optimizations for DSP targets
- Review PyTorch's Hexagon backend for DMA patterns

### Advanced Topics to Explore

1. **HMX (Hexagon Matrix eXtension)**: If your target supports v78+, explore matrix multiply instructions and their specialized memory layouts.

2. **Multi-Island Scheduling**: For v75+, design allocators that span multiple VTCM islands with minimal inter-island traffic.

3. **Power Management**: Integrate VTCM allocation with DVFS (dynamic voltage/frequency scaling) for power optimization.

4. **Heterogeneous Compute**: Use VTCM for DSP offload while keeping primary inference on GPU or CPU.

5. **Compiler Integration**: Extend your compiler to generate optimal VTCM allocation and DMA schedules automatically.

---

**End of Module 4: Memory Subsystem Mastery**

---

*This module provided PhD-level depth into Hexagon memory systems, from architectural fundamentals to production-grade implementation patterns. Students completing this module should be capable of designing optimized inference engines that achieve near-peak VTCM utilization, minimize data movement overhead, and meet real-time inference SLAs.*

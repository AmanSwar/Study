# APPENDIX B — Hardware Performance Counter Reference

## Overview

This appendix provides exact commands, event names, and interpretation guidelines for performance counter collection on Intel Sapphire Rapids (SPR) and AMD Zen 4. Covers Top-Down Microarchitecture Analysis (TMA) metric computation, cache/branch/memory metrics, and tool-specific workflows.

**Key Tools:**
- **Linux perf:** PMU event collection on Intel/AMD
- **Intel VTune:** Graphical profiling, tuning analysis
- **AMD uProf:** AMD-specific PMU collection and TMA
- **likwid:** Portable counter configuration and calculation

---

## 1. Top-Down Microarchitecture Analysis (TMA) — Intel SPR

TMA is a hierarchical methodology for identifying bottlenecks: Retiring (useful work) vs. Bad Speculation (mispredictions) vs. Frontend Stalls vs. Backend Stalls.

### TMA Level-1 Metrics

**Formula (per-cycle breakdown):**
```
Retiring = UOPS_RETIRED.RETIRE_SLOTS / 4 / CPU_CLK_UNHALTED.THREAD
Frontend_Stalls = (IDQ_UOPS_NOT_DELIVERED.CORE - IDQ_UOPS_NOT_DELIVERED.CORE * FRONTEND_RETIRED.LATENCY_GE_8 / 256) / 4 / CPU_CLK_UNHALTED.THREAD
Bad_Speculation = (UOPS_ISSUED.ANY - UOPS_RETIRED.RETIRE_SLOTS + 5 * INT_MISC.RECOVERY_CYCLES) / 4 / CPU_CLK_UNHALTED.THREAD
Backend_Stalls = 1.0 - Retiring - Frontend_Stalls - Bad_Speculation
```

### Collection Command (SPR, Linux perf)

```bash
# Collect TMA L1 events (single run)
perf stat -e \
  'cpu/umask=0x0,period=1000,event=0xc2/pp' \
  'cpu/umask=0xf,period=1000,event=0xce/pp' \
  'cpu/umask=0x1,period=1000,event=0xae/pp' \
  'cpu/umask=0x0,period=1000,event=0x3c/pp' \
  'cpu/umask=0x81,period=1000,event=0x9c/pp' \
  'cpu/umask=0x1,period=1000,event=0xc3/pp' \
  'cpu/umask=0x0,period=1000,event=0x79/pp' \
  ./myapp
```

**Event Mappings for SPR:**

| Event Code | umask | Name | Long Name |
|---|---|---|---|
| 0xC2 | 0x00 | UOPS_RETIRED.RETIRE_SLOTS | Retiring uops executed |
| 0xCE | 0x0F | IDQ_UOPS_NOT_DELIVERED.CORE | Frontend lost cycles (uop starvation) |
| 0xAE | 0x01 | INT_MISC.RECOVERY_CYCLES | Cycles lost to exception recovery |
| 0x3C | 0x00 | CPU_CLK_UNHALTED.THREAD | Reference cycles (unhalted) |
| 0x9C | 0x81 | UOPS_ISSUED.ANY | Uops dispatched to execution |
| 0xC3 | 0x01 | MACHINE_CLEARS.SMC | Self-modifying code flushes |
| 0x79 | 0x00 | IDQ_UOPS_NOT_DELIVERED.CORE | (Alias for frontend delivery) |

### Calculation Script (Python)

```python
#!/usr/bin/env python3
"""
TMA L1 calculator for SPR from perf stat JSON output.
Usage: perf stat -o perf_data.json ./app && python tma_calc.py perf_data.json
"""

import json
import sys

def calc_tma_l1(data):
    # Extract counter values (thousands: perf reports in absolute counts)
    uops_retired = data['UOPS_RETIRED.RETIRE_SLOTS']['value']
    idq_not_delivered = data['IDQ_UOPS_NOT_DELIVERED.CORE']['value']
    recovery_cycles = data['INT_MISC.RECOVERY_CYCLES']['value']
    ref_cycles = data['CPU_CLK_UNHALTED.THREAD']['value']
    uops_issued = data['UOPS_ISSUED.ANY']['value']

    # Normalize to cycles (SPR has 4 uops/cycle)
    retiring = (uops_retired / 4.0) / ref_cycles

    # Frontend stalls (advanced formula with latency penalty)
    frontend = (idq_not_delivered / 4.0) / ref_cycles

    # Bad speculation
    bad_spec = ((uops_issued - uops_retired + 5 * recovery_cycles) / 4.0) / ref_cycles

    # Backend stalls
    backend = 1.0 - retiring - frontend - bad_spec

    return {
        'retiring': retiring,
        'frontend_stalls': frontend,
        'bad_speculation': bad_spec,
        'backend_stalls': backend
    }

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        data = json.load(f)

    tma = calc_tma_l1(data)
    print(f"Retiring:           {tma['retiring']:.1%}")
    print(f"Frontend Stalls:    {tma['frontend_stalls']:.1%}")
    print(f"Bad Speculation:    {tma['bad_speculation']:.1%}")
    print(f"Backend Stalls:     {tma['backend_stalls']:.1%}")
```

### TMA Level-2 Metrics (Intel SPR)

When **Backend Stalls** > 25%, drill down:

```bash
# Collect L2 backend events
perf stat -e \
  'cpu/event=0xa4,umask=0x10/pp' \  # CYCLES_NO_EXECUTE
  'cpu/event=0xa4,umask=0x0c/pp' \  # PORT_UTILIZATION
  'cpu/event=0x9d,umask=0x01/pp' \  # UOPS_EXECUTED.CORE
  ./myapp
```

| Metric | Event | Formula |
|---|---|---|
| Memory Bound | (CYCLE_ACTIVITY.STALL_MEM_ANY - CYCLE_ACTIVITY.STALL_L3_MISS) / CPU_CLK | Stalled on L1/L2/L3 |
| L3 Bound | CYCLE_ACTIVITY.STALL_L3_MISS / CPU_CLK | Stalled waiting for main memory |
| Core Bound | (UOPS_EXECUTED.CORE - UOPS_RETIRED) / 4 / CPU_CLK | Execution resources exhausted |
| Store Bound | (CYCLE_ACTIVITY.STALL_L1D_PENDING - CYCLE_ACTIVITY.STALL_MEM_ANY) / CPU_CLK | Awaiting store completion |

### When Frontend Stalls > 25%:

```bash
perf stat -e \
  'cpu/event=0x9c,umask=0x01/pp' \  # DECODE_RESTRICTION
  'cpu/event=0x0e,umask=0x1/pp' \   # UOPS_ISSUED.STALL_CYCLES
  'cpu/event=0xc6,umask=0x0/pp' \   # FRONTEND_RETIRED.LATENCY_GE_8
  ./myapp
```

| Condition | Event | Action |
|---|---|---|
| Fetch stalls | FRONTEND_RETIRED.LATENCY_GE_8 | Increase code locality, reduce BTB misses |
| Decode stalls | IDQ_UOPS_NOT_DELIVERED.CORE | Reduce instruction footprint |
| LCP stalls | DECODE_RESTRICTION | Align branches/instructions to cache lines |

---

## 2. Cache Hierarchy Events (SPR and Zen 4)

### L1 Data Cache (L1D)

**SPR:**
```bash
perf stat -e \
  'cpu/event=0x51,umask=0x01/pp' \  # L1D.REPLACEMENT (line allocated)
  'cpu/event=0x51,umask=0x08/pp' \  # L1D.EVICTION (line evicted)
  'cpu/event=0x60/pp' \             # DTLB_LOAD_MISSES.MISS_CAUSES_A_WALK
  ./myapp
```

**Zen 4 (AMD):**
```bash
perf stat -e \
  'amd_l3/l3_cache_accesses/pp' \
  'amd_l3/l3_cache_misses/pp' \
  'amd_memory/page_table_walk/pp' \
  ./myapp
```

### L2 Cache (Unified)

| Platform | Event | Purpose |
|---|---|---|
| **SPR** | `L2_LINES_IN.ALL` | L2 line fills from L3 |
| **SPR** | `L2_LINES_OUT.USELESS_HWPF` | L2 prefetch that doesn't hit |
| **SPR** | `L2_HW_PREFETCH.MTVEC` | Hardware prefetcher activity |
| **Zen 4** | `amd_l3/l2_lines_in` | L2 fills |
| **Zen 4** | `amd_l3/l2_cache_miss` | L2→L3 misses |

### L3 Cache (Shared, per-socket)

```bash
# SPR: L3 miss rate
perf stat -e \
  'cpu/event=0xb7,umask=0x02/pp' \  # MEM_LOAD_RETIRED.L3_MISS
  'cpu/event=0xb7,umask=0x1f/pp' \  # MEM_LOAD_RETIRED.ALL_LOADS
  ./myapp
```

**Expected L3 Miss Ratio:**
- SPR 32 MB L3 (per-socket): 5–15% for typical workloads, 30%+ for streaming
- Zen 4 96 MB L3 (per-socket): 2–10% for same workloads (larger cache advantage)

### Full Cache Command (SPR)

```bash
perf stat \
  -e 'cache-references,cache-misses' \
  -e 'LLC-loads,LLC-load-misses,LLC-stores' \
  -e 'L1-dcache-loads,L1-dcache-load-misses' \
  -e 'L1-icache-loads,L1-icache-load-misses' \
  ./myapp
```

---

## 3. Branch Prediction & Misprediction Events

### SPR Branch Events

```bash
perf stat -e \
  'cpu/event=0xc4,umask=0x00/pp' \  # BR_INST_RETIRED.ALL_BRANCHES
  'cpu/event=0xc5,umask=0x00/pp' \  # BR_MISP_RETIRED.ALL_BRANCHES
  'cpu/event=0x88,umask=0x41/pp' \  # BR_MISP_PRED.ALL_MISPREDICTS
  ./myapp
```

| Event | umask | Meaning |
|---|---|---|
| BR_INST_RETIRED | 0x00 | Total branch instructions that retired |
| BR_INST_RETIRED | 0x01 | Conditional branches only |
| BR_INST_RETIRED | 0x02 | Unconditional branches (JMP, RET) |
| BR_MISP_RETIRED | 0x00 | Total mispredicted branches |
| BR_MISP_RETIRED | 0x01 | Conditional misses |

### Zen 4 Branch Events

```bash
perf stat -e \
  'amd_bpu/retired_branch_instructions/pp' \
  'amd_bpu/retired_mispredicted_branch_instructions/pp' \
  ./myapp
```

### Interpretation

```python
# Calculate branch misprediction ratio
branches = 1_000_000
mispredicts = 50_000
mpki = (mispredicts / branches) * 1000  # Per thousand instructions
miss_ratio = mispredicts / branches

print(f"Misprediction ratio: {miss_ratio:.2%}")
print(f"MPKI: {mpki:.1f}")

# Penalty estimate: ~15 cycles per misprediction on SPR
penalty_cycles = mispredicts * 15
total_cycles = 2_000_000
overhead = penalty_cycles / total_cycles
print(f"Performance overhead: {overhead:.1%}")
```

**Threshold:** MPKI < 1.0 is healthy; > 5.0 warrants investigation.

---

## 4. TLB (Translation Lookaside Buffer) Events

### SPR DTLB (Data)

```bash
perf stat -e \
  'cpu/event=0x49,umask=0x01/pp' \  # DTLB_LOAD_MISSES.MISS_CAUSES_A_WALK
  'cpu/event=0x49,umask=0x04/pp' \  # DTLB_LOAD_MISSES.WALK_COMPLETED_4K
  'cpu/event=0x49,umask=0x08/pp' \  # DTLB_LOAD_MISSES.WALK_COMPLETED_2M
  'cpu/event=0x49,umask=0x20/pp' \  # DTLB_LOAD_MISSES.WALK_PENDING
  ./myapp
```

| Event | Interpretation |
|---|---|
| MISS_CAUSES_A_WALK | DTLB entry missing; page table walk initiated |
| WALK_COMPLETED_4K | Walk completed for 4 KB page |
| WALK_COMPLETED_2M | Walk completed for 2 MB page (huge page) |
| WALK_PENDING | Cycles stalled waiting for walk to complete |

### Zen 4 DTLB

```bash
perf stat -e \
  'amd_memory/page_table_walk/pp' \
  'amd_iommu/dtlb_misses/pp' \
  ./myapp
```

### Huge Page Impact

```bash
# Enable huge pages
echo 128 > /proc/sys/vm/nr_hugepages

# Check enabled
grep HugePages /proc/meminfo
# Output example: HugePages_Total:      128
#                 HugePages_Free:       110

# Measure TLB misses before/after
perf stat -e 'dtlb-loads,dtlb-load-misses' ./myapp_no_hugepages
perf stat -e 'dtlb-loads,dtlb-load-misses' ./myapp_with_hugepages
```

**Expected improvement:** 30–50% fewer DTLB misses with 2 MB pages on SPR.

---

## 5. Memory Bandwidth & Latency Events

### SPR Memory Load Latency

```bash
perf mem record -t load -e cpu/mem-loads/pp,period=100 ./myapp
perf mem report --sort=mem,symbol,source --stdio
```

**Output Example:**
```
Symbol           Memory type       Percent
__main__         L1 hit            45.2%
process_data     L2 hit            28.1%
dgemm_kernel     L3 hit            18.3%
matrix_multiply  Local RAM         6.8%
                 Remote NUMA       1.6%
```

### Memory Bandwidth (Hardware)

```bash
# Intel MLC (Memory Latency Checker) — see Appendix E
/opt/mlc/mlc --peak_injection_bandwidth --latency -T

# Output (example, SPR dual-socket):
Measuring peak bandwidth for sequential access with stride 576 (repeat 50)...
    Results: 921.6 GB/sec

Measuring latency at different strides...
Stride:4   Lat:    4.9
Stride:64  Lat:   68.5
Stride:4096 Lat:  192.3
```

### perf + memory BW command (SPR)

```bash
perf stat -e \
  'unc_imc/dclk/pp' \
  'unc_imc/cas_wr/pp' \
  'unc_imc/cas_rd/pp' \
  ./myapp
```

**Calculation:**
```
BW = (CAS_RD + CAS_WR) * 64 bytes * (DCLK_FREQ_GHz) / runtime_seconds
```

---

## 6. Frontend Stall Metrics (Intel VTune for SPR)

### VTune Collection Command

```bash
# Install VTune (if needed)
sudo apt install intel-vtune

# Profile with TMA
vtune -collect performance -knob enable_tma=true -app-working-dir /path ./myapp

# Generate HTML report
vtune -report hotspots -r r000hs -format html -o report.html
```

### VTune Output Interpretation

**Example TMA report:**
```
Function: matrix_multiply
  Retiring:         32%    ✓ Good utilization
  Bad Speculation:   8%    ✓ Normal branch prediction
  Frontend Stalls:  18%    ⚠ Investigate code layout
  Backend Stalls:   42%    ✗ Memory or execution resource bound
```

### Drill into Backend Stalls

```bash
# Re-run with L2/L3 event collection
vtune -collect memory-access -app-working-dir /path ./myapp
```

---

## 7. AMD uProf (Zen 4 Equivalent to VTune)

### Installation & Basic Usage

```bash
# Download from amd.com/uprof
./AmdUprofInstaller.sh

# Collect TMA metrics
./AMDuProfCLI collect --config TMA --duration 60 ./myapp

# Generate report
./AMDuProfCLI report --db . --format html -o uprof_report.html
```

### uProf TMA Metrics (Zen 4)

| Metric | Good Range | Warning |
|---|---|---|
| Retiring | > 50% | Good IPC |
| Frontend Bound | < 20% | Instruction fetch bottleneck |
| Bad Speculation | < 10% | Misprediction rate okay |
| Backend Bound | < 30% | Resource/memory bottleneck |

---

## 8. Full Example: Complete Profiling Workflow (SPR)

### Step 1: Initial Characterization

```bash
#!/bin/bash

echo "=== Baseline Perf Stat ==="
perf stat -d -d -d ./myapp > /tmp/baseline.txt 2>&1
cat /tmp/baseline.txt

echo "=== TMA L1 ==="
perf record -k 1 -e \
  'cpu/event=0xc2,umask=0x0,period=10000/pp' \
  'cpu/event=0xce,umask=0xf,period=10000/pp' \
  'cpu/event=0xae,umask=0x1,period=10000/pp' \
  -a -- timeout 10 ./myapp

perf report -g --stdio | head -50
```

### Step 2: Isolate Bottleneck

```bash
# If Backend > 30%:
perf stat -e \
  'cpu/event=0x04,umask=0x10/pp' \  # CYCLE_ACTIVITY.STALL_L3_MISS
  'cpu/event=0x04,umask=0x08/pp' \  # CYCLE_ACTIVITY.STALL_MEM_ANY
  'cpu/event=0x04,umask=0x04/pp' \  # CYCLE_ACTIVITY.STALL_L1D_PENDING
  ./myapp
```

### Step 3: Cache-Specific Analysis

```bash
perf stat --detailed --detailed ./myapp | grep -E "cache|Cache"
```

### Step 4: Confirm with Hardware Counters

```bash
# Validate with VTune
vtune -collect hotspots -app-working-dir . ./myapp
vtune -report hotspots -r r000hs -group "Retiring,Bad Speculation,Frontend,Backend" -format html
```

---

## 9. Event Aliases (perf shorthand)

```bash
# Instead of event codes, use friendly names
perf stat -e cache-references,cache-misses,instructions,cycles,stalled-cycles-frontend,stalled-cycles-backend ./myapp
```

| Alias | Equivalent | Platform |
|---|---|---|
| cache-references | LLC lookups | Both |
| cache-misses | LLC misses | Both |
| cycles | CPU_CLK_UNHALTED.THREAD | SPR |
| instructions | INST_RETIRED.ANY | SPR |
| branch-misses | BR_MISP_RETIRED.ALL | SPR |
| dtlb-loads | DTLB_LOAD_MISSES | SPR |

---

## 10. Unprivileged Profiling (Non-Root perf)

```bash
# Enable user-space profiling without sudo
sudo sysctl kernel.perf_event_paranoid=1  # (from 2)

# Collect unprivileged
perf record -e cycles,instructions -u $USER ./myapp
perf report --stdio
```

---

## 11. Real-Time Monitoring (vtop, sar)

### System Activity Reporter (sar)

```bash
# Monitor memory bandwidth in real-time
sar -B 1 10  # Report page buffer activity every 1 second, 10 times

# Output:
# pgpgin/s  pgpgout/s   fault/s  majflt/s
#   1234.5    5678.9   12340.1      0.0
```

### Intel Uncore PMU (NUMA-aware)

```bash
# Monitor per-socket memory traffic
perf stat -e \
  'unc_imc0/cas_rd/pp' \
  'unc_imc1/cas_rd/pp' \
  'unc_imc0/cas_wr/pp' \
  'unc_imc1/cas_wr/pp' \
  ./myapp
```

**Interpretation:** If unc_imc1 (socket 1) >> unc_imc0, consider NUMA-aware optimization (Appendix D).

---

## 12. Scripted Counter Collection

```bash
#!/usr/bin/env python3
"""
Automated perf event collection and TMA calculation.
"""

import subprocess
import json
import re

def collect_events(exe, event_list):
    cmd = ['perf', 'stat', '-j', '-e', ','.join(event_list)] + exe.split()
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stderr)

def main():
    exe = './myapp arg1 arg2'

    # TMA L1 events
    events = [
        'cpu/event=0xc2,umask=0x0/pp',  # UOPS_RETIRED.RETIRE_SLOTS
        'cpu/event=0xce,umask=0xf/pp',  # IDQ_UOPS_NOT_DELIVERED.CORE
        'cpu/event=0xae,umask=0x1/pp',  # INT_MISC.RECOVERY_CYCLES
        'cpu/event=0x3c,umask=0x0/pp',  # CPU_CLK_UNHALTED.THREAD
        'cpu/event=0x9c,umask=0x81/pp', # UOPS_ISSUED.ANY
    ]

    data = collect_events(exe, events)

    # Parse and calculate TMA
    values = {item['event']: item['value'] for item in data}

    retiring = (values.get('0xc2', 0) / 4.0) / values.get('0x3c', 1)
    print(f"Retiring: {retiring:.1%}")

if __name__ == '__main__':
    main()
```

---

## References

1. **Intel 64 and IA-32 Architectures Performance Monitoring Reference Manual**
2. **Top-Down Microarchitecture Analysis** (Yasin, S. "A Top-Down Method for Performance Analysis and Counters Architectures")
3. **Linux perf documentation:** `man perf-list`
4. **Intel VTune:** `software.intel.com/content/www/en/en/develop/tools/oneapi/components/vtune-profiler.html`
5. **AMD uProf:** `amd.com/en/developer/uprof`


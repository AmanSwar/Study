# APPENDIX D — NUMA Topology Interrogation

## Overview

This appendix shows how to fully understand a server's NUMA topology before writing performance-critical code. Covers `numactl`, `lstopo` (hwloc), `/sys/devices/system/node/`, and includes complete topology dumps for real systems.

---

## 1. Quick Start Commands

```bash
# Install tools
sudo apt install numactl hwloc hwloc-gui

# Fast NUMA overview
numactl --hardware

# Detailed visual topology
lstopo --of console

# JSON topology (for parsing)
lstopo --of json > topology.json
```

---

## 2. numactl --hardware Output & Interpretation

### Example: 2-Socket Intel Sapphire Rapids (SPR)

```
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
node 0 memory: 512000 MB
node 0 distances: 10 20
node 1 cpus: 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
node 1 memory: 512000 MB
node 1 distances: 20 10
```

**Interpretation:**

| Line | Meaning |
|---|---|
| `available: 2 nodes (0-1)` | Two NUMA nodes accessible (node 0, node 1) |
| `node 0 cpus: 0-23` | Cores 0–23 are local to node 0 |
| `node 0 memory: 512000 MB` | 512 GB RAM local to node 0 |
| `node 0 distances: 10 20` | Distance 10 to self (node 0), 20 to remote (node 1) |
| `node 1 distances: 20 10` | Distance 20 to remote (node 0), 10 to self (node 1) |

**Distance scale (latency ratio):**
- **10:** Local memory access baseline (~ 60–100 ns on SPR)
- **20:** Remote NUMA access (~ 150–200 ns on SPR)
- **Ratio:** Remote is 1.5–2× slower

### Example: AMD EPYC 9004 (Genoa) 2-Socket

```
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
node 0 memory: 768000 MB
node 0 distances: 10 16
node 1 cpus: 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
node 1 memory: 768000 MB
node 1 distances: 16 10
```

**Differences from SPR:**
- **More cores per socket:** 32 vs 24 (Genoa has up to 12 cores per CXL unit)
- **More memory:** 768 GB per socket (larger channel count)
- **Different distance ratio:** 16 (vs 20), slightly less NUMA penalty due to improved interconnect

---

## 3. lstopo Visual Topology

### Basic lstopo Output (SPR)

```
Machine (1023GB)
  Package L#0 (503GB)
    L3 L#0 (32MB)
      L2 L#0 (2MB)
        L1d L#0 (48KB) + L1i L#0 (32KB)
          Core L#0
            PU L#0 (P#0)
            PU L#1 (P#1)
      ... (3 more cores sharing L3)
    NUMANode L#0 (503GB) weight=503000
  Package L#1 (503GB)
    L3 L#1 (32MB)
      ... (24 cores)
    NUMANode L#1 (503GB)
```

### lstopo with Detailed Output

```bash
# Full terminal visualization
lstopo --of console

# Output with device numbering
lstopo --logical --of console
```

### lstopo JSON for Programmatic Access

```bash
lstopo --of json > topology.json
cat topology.json | python3 -m json.tool | head -100
```

**JSON excerpt (SPR):**

```json
{
  "type": "Machine",
  "depth": 0,
  "name": "Machine",
  "total_memory": 1048576000,
  "children": [
    {
      "type": "Package",
      "depth": 1,
      "os_index": 0,
      "name": "Package L#0",
      "children": [
        {
          "type": "NUMANode",
          "depth": 2,
          "os_index": 0,
          "memory": 536870912,
          "name": "NUMANode L#0"
        },
        {
          "type": "L3Cache",
          "depth": 2,
          "size": 33554432,
          "children": [
            {
              "type": "L2Cache",
              "size": 2097152,
              "children": [
                {
                  "type": "L1dCache",
                  "size": 49152
                },
                {
                  "type": "L1iCache",
                  "size": 32768
                },
                {
                  "type": "Core",
                  "os_index": 0,
                  "children": [
                    {
                      "type": "PU",
                      "os_index": 0,
                      "logical_index": 0
                    },
                    {
                      "type": "PU",
                      "os_index": 1,
                      "logical_index": 1
                    }
                  ]
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

---

## 4. /sys/devices/system/node/ Filesystem

### Directory Structure

```bash
ls -la /sys/devices/system/node/
total 0
drwxr-xr-x  3 root root   0 Mar 18 10:52 .
drwxr-xr-x 14 root root   0 Mar 18 10:52 ..
-rw-r--r--  1 root root 4.0K Mar 18 10:52 has_cpu
-rw-r--r--  1 root root 4.0K Mar 18 10:52 has_memory
-rw-r--r--  1 root root 4.0K Mar 18 10:52 has_normal_memory
drwxr-xr-x  2 root root   0 Mar 18 10:52 node0
drwxr-xr-x  2 root root   0 Mar 18 10:52 node1
-rw-r--r--  1 root root 4.0K Mar 18 10:52 possible
```

### Node-Specific Files (node0)

```bash
ls -la /sys/devices/system/node/node0/
-rw-r--r-- ... cpulist
-rw-r--r-- ... cpumap
-r--r--r-- ... distance
-rw-r--r-- ... hugepages
-rw-r--r-- ... meminfo
-rw-r--r-- ... uevent
```

### Reading CPU Affinity

```bash
# CPUs local to node 0 (list format)
cat /sys/devices/system/node/node0/cpulist
# Output: 0-23

# CPUs as bitmap (hexadecimal)
cat /sys/devices/system/node/node0/cpumap
# Output: 0xffffff,0xffffffff  (48 bits for 48 cores across two nodes)

# NUMA distances (matrix)
cat /sys/devices/system/node/node0/distance
# Output (SPR): 10 20
#        (Zen 4): 10 16
```

### Reading Memory Info

```bash
cat /sys/devices/system/node/node0/meminfo
MemTotal:       524288000 kB
MemFree:        512000000 kB
MemUsed:         12288000 kB
HugePages_Total:      256
HugePages_Free:       256
HugePages_Surp:         0
```

### Parsing Script

```bash
#!/bin/bash
# Dump complete NUMA topology from /sys

echo "=== NUMA Topology from /sys/devices/system/node/ ==="

for node in /sys/devices/system/node/node[0-9]*; do
    node_id=$(basename $node)
    echo ""
    echo "Node: $node_id"
    echo "  CPUs: $(cat $node/cpulist)"
    echo "  Memory: $(grep MemTotal $node/meminfo | awk '{print $2/1024/1024 " GB"}')"
    echo "  Distances: $(cat $node/distance)"
done
```

---

## 5. Complete Topology Dump: Dual-Socket SPR

```bash
#!/bin/bash
# complete_topology.sh - Dump all topology info

echo "=== SYSTEM TOPOLOGY DUMP ==="
echo "Date: $(date)"
echo ""

echo "1. CPUs and Cores:"
lscpu

echo ""
echo "2. NUMA Layout (numactl):"
numactl --hardware

echo ""
echo "3. Cache Topology:"
lstopo --of console | head -50

echo ""
echo "4. CPU Affinity (/sys):"
for node in /sys/devices/system/node/node*; do
    nid=$(basename $node)
    echo "Node $nid CPUs: $(cat $node/cpulist)"
    echo "Node $nid Memory: $(grep MemTotal $node/meminfo | awk '{print $2/1024/1024 " GB"}')"
done

echo ""
echo "5. Core Distribution:"
echo "Cores per socket:"
for node in /sys/devices/system/node/node*; do
    cpus=$(cat $node/cpulist)
    count=$(seq ${cpus%-*} ${cpus#*-} | wc -l)
    echo "  $(basename $node): $count cores"
done

echo ""
echo "6. L3 Cache Size:"
lstopo --of console | grep "L3"

echo ""
echo "7. Memory Channels:"
dmidecode -t memory | grep "Locator\|Size"

echo ""
echo "8. Hugepages:"
grep HugePages /proc/meminfo
```

**Sample Output (SPR):**

```
=== SYSTEM TOPOLOGY DUMP ===
Date: Fri Mar 22 14:35:22 UTC 2026

1. CPUs and Cores:
Architecture:            x86_64
CPU op-mode(s):          32-bit, 64-bit
CPU(s):                  48
On-line CPU(s) list:     0-47
Thread(s) per core:      2
Core(s) per socket:      12
Socket(s):               2
NUMA node(s):            2
Model name:              Intel(R) Xeon(R) Platinum 8480

2. NUMA Layout (numactl):
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 22 23
node 0 memory: 512000 MB
node 0 distances: 10 20
node 1 cpus: 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
node 1 memory: 512000 MB
node 1 distances: 20 10

3. Cache Topology:
Machine (1023GB)
  Package L#0 (503GB)
    L3 L#0 (32MB)
      L2 L#0 (2MB)
        L1d L#0 (48KB)
          Core L#0
            PU L#0
            PU L#1
    ... (12 cores per package)
  Package L#1 (503GB)

6. L3 Cache Size:
  L3 L#0 (32MB)
  L3 L#1 (32MB)
```

---

## 6. Practical: Binding Code to NUMA Nodes

### Using numactl Command-Line

```bash
# Run on node 0 only
numactl --cpunodebind=0 --membind=0 ./myapp

# Run on both nodes, prefer local memory
numactl --interleave=all ./myapp

# Bind to specific cores (0-11 from node 0)
numactl --physcpubind=0-11 ./myapp

# Bind to socket 1 (cores 24-35)
numactl --cpunodebind=1 --membind=1 ./myapp
```

### Programmatic Binding (C)

```c
#define _GNU_SOURCE
#include <numa.h>
#include <numaif.h>
#include <sched.h>

int main(void) {
    /* Initialize NUMA library */
    if (numa_available() < 0) {
        fprintf(stderr, "NUMA not available\n");
        return 1;
    }

    /* Get node mask (all available) */
    struct bitmask *mask = numa_allocate_nodemask();
    numa_parse_nodestring("0", mask);  /* Bind to node 0 */

    /* Set memory policy: strict local allocation */
    numa_set_membind(mask);

    /* Allocate memory (will be on node 0) */
    int *data = numa_alloc_onnode(1024 * 1024, 0);  /* 1 MB on node 0 */

    /* Bind CPU affinity to node 0 */
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = 0; i < 12; i++) {  /* SPR: 12 cores per socket */
        CPU_SET(i, &cpuset);
    }
    sched_setaffinity(0, sizeof(cpuset), &cpuset);

    /* Run compute-intensive work */
    for (int i = 0; i < 1024 * 1024; i++) {
        data[i] += i;
    }

    numa_free(data, 1024 * 1024);
    numa_free_nodemask(mask);
    return 0;
}
```

**Compile:**

```bash
gcc -O2 -o numa_bind numa_bind.c -lnuma
```

---

## 7. NUMA Awareness in OpenMP

```c
#include <omp.h>
#include <numa.h>

int main(void) {
    int num_nodes = numa_num_configured_nodes();
    printf("NUMA nodes: %d\n", num_nodes);

    #pragma omp parallel num_threads(24)
    {
        int thread_id = omp_get_thread_num();
        int node = thread_id / 12;  /* Assuming 12 cores per node */

        /* Bind this thread to its NUMA node */
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        for (int i = 0; i < 12; i++) {
            CPU_SET(node * 12 + i, &cpuset);
        }
        sched_setaffinity(0, sizeof(cpuset), &cpuset);

        printf("Thread %d -> Node %d\n", thread_id, node);

        /* Allocate and compute on local node */
        int *local_data = numa_alloc_onnode(1024, node);
        for (int i = 0; i < 1024; i++) {
            local_data[i] *= 2;
        }
        numa_free(local_data, 1024);
    }

    return 0;
}
```

---

## 8. Detecting Cross-NUMA Accesses (Performance Analysis)

### perf-based Detection

```bash
# Measure NUMA misses (cross-socket memory access)
perf stat -e \
  'cpu/event=0xd1,umask=0x10/pp' \  # MEM_LOAD_UOPS_RETIRED.ALL_LOADS
  'cpu/event=0xcd,umask=0x01/pp' \  # MEM_TRANS_RETIRED.LOAD_LATENCY
  ./myapp

# High latency + high count = NUMA misses
```

### Intel VTune for NUMA Analysis

```bash
vtune -collect memory-access -knob analyze_mem_objects=true ./myapp
vtune -report memory-access-latency -r r000ma -group "remote_numa" -format html
```

### Linux perf numa events

```bash
perf c2c record --all -- ./myapp  # Cache-coherency (NUMA) tracking
perf c2c report --stdio
```

**Output shows:**
- Which load addresses cause remote memory accesses
- Which threads are accessing remote memory
- Latency breakdown

---

## 9. Zen 4 (AMD EPYC 9004) Specific Considerations

### Key Differences from SPR

1. **More cores:** 32 per socket (vs 24 on SPR)
2. **Larger L3:** 96 MB per socket (vs 32 MB on SPR)
3. **CXL units:** Cores grouped in CXL units; intra-unit access faster than inter-unit
4. **Distance ratio:** ~16 (vs 20 on SPR)

### Topology Differences

```bash
# Zen 4 numactl output shows CXL-aware grouping
available: 12 nodes (0-11)  # More NUMA nodes due to CXL subdivision
node 0 cpus: 0-3
node 0 memory: 64000 MB
```

### Optimized NUMA Binding for Zen 4

```bash
# Bind to a full CXL unit (e.g., cores 0-3 on node 0)
numactl --cpunodebind=0 --membind=0 ./myapp

# For large workload, interleave across CXL units
numactl --interleave=0,1,2,3 ./myapp  # 4 CXL units on socket 0
```

---

## 10. Topology-Aware Malloc Strategies

### Preferred NUMA Configuration for Different Workloads

| Workload | Best Binding | Rationale |
|---|---|---|
| **Single-threaded, compute-heavy** | Bind to one NUMA node | Maximize L3 cache, avoid remote access |
| **Multi-threaded, shared data** | Interleave or node-local allocate | Balance memory bandwidth across nodes |
| **Streaming workload** | Bind to source and sink nodes | Minimize NUMA hops in pipeline |
| **Graph traversal** | Thread-local + first-touch | Distribute by access pattern |

### Example: Interleaved Memory Allocation

```c
#include <numa.h>

float *allocate_matrix_numa_interleaved(size_t rows, size_t cols) {
    struct bitmask *mask = numa_allocate_nodemask();

    /* Interleave across all nodes */
    for (int i = 0; i < numa_num_configured_nodes(); i++) {
        numa_bitmask_setbit(mask, i);
    }
    numa_set_membind(mask);

    float *matrix = malloc(rows * cols * sizeof(float));

    /* First-touch (parallel) forces NUMA distribution */
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            matrix[i * cols + j] = 0.0f;
        }
    }

    numa_free_nodemask(mask);
    return matrix;
}
```

---

## 11. Reference: Full Topology Dump Template

```bash
#!/bin/bash
# topology_report.sh - Generate comprehensive topology report

cat > /tmp/topology_report.txt << 'EOF'
=== COMPREHENSIVE NUMA TOPOLOGY REPORT ===
Generated: $(date)

1. NUMA Overview:
EOF

numactl --hardware >> /tmp/topology_report.txt

cat >> /tmp/topology_report.txt << 'EOF'

2. CPU to NUMA Binding:
EOF

for node in /sys/devices/system/node/node*; do
    echo "$(basename $node): $(cat $node/cpulist)" >> /tmp/topology_report.txt
done

cat >> /tmp/topology_report.txt << 'EOF'

3. Memory Distribution:
EOF

free -h >> /tmp/topology_report.txt
grep MemTotal /sys/devices/system/node/*/meminfo >> /tmp/topology_report.txt

cat >> /tmp/topology_report.txt << 'EOF'

4. L3 Cache (lstopo):
EOF

lstopo --of console | grep -E "L3|Package" >> /tmp/topology_report.txt

cat /tmp/topology_report.txt
```


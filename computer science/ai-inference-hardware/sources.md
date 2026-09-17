# Sources — AI inference hardware (NVIDIA, AMD, Google TPU, AWS Trainium/Inferentia)

As of 2026-09-17 · 50 sources · deep-dive

Supplemental research pass (same day) added S30-S50 to close three gaps: NVIDIA/AMD next-gen roadmap, official cloud on-demand pricing, and MLPerf Training benchmarks. See the "SUPPLEMENTAL PASS" note at the end of `sources.json`'s `notes` field for the full account.

## Coverage

**Well covered** (≥2 primary sources with facts): nvidia-hopper, nvidia-blackwell, nvidia-next-gen, amd-instinct, google-tpu-v5-v6, google-tpu-v7, aws-trainium, aws-inferentia, mlperf-inference, mlperf-training, serving-throughput-sizing, interconnect-scaling, decision-framework, cloud-pricing

**Thin**: —

**None**: —

**Honest residual gap** (not a "thin" subtopic, but worth flagging to writers): AWS Trainium/Trainium2 has never submitted to MLPerf Training (confirmed absent from the v5.0 and v5.1 participant lists, S45/S46). Any Trainium training-performance claim in the study material must come from AWS's own peak-FLOPS marketing (S17/S21/S22) with that provenance stated explicitly — there is no third-party/standardized Trainium training benchmark to cite.

## Disagreements found

- **Is TPU cheaper than GPU for inference?** — Introl claims TPU v6e is ~4x better price-performance than H100 at $0.39/chip-hour [S29] vs. Junyi Hou's re-derivation from Artificial Analysis' own dataset showing TPU v6e is only roughly competitive with H100 in offline/batch serving ($0.62 vs $0.67-0.69/M tokens) and ~5x *more expensive* in interactive serving ($5.13 vs $1.06/M tokens) [S28, S27].
- **Does AMD MI300X/MI325X beat NVIDIA H100/H200?** — Workload-dependent: AMD wins on perf/$ and absolute throughput for large dense models and high-concurrency batch serving [S26, S27, S25]; NVIDIA wins decisively on latency-sensitive low-concurrency serving and on measured GEMM throughput (14-22% higher achieved FLOP/s) [S26, S27].
- **Do vendor MLPerf numbers reflect real deployments?** — NVIDIA's own MLPerf v5.0 submissions (TensorRT-LLM-tuned) show up to 30x GB200-vs-H100 system throughput [S5, S23]; independent open-source-framework benchmarking (InferenceMAX) shows materially smaller, workload-dependent gaps and notes vLLM-on-Hopper sometimes beats "mostly open" TRT-LLM [S25].
- **What does the "30x" GB200 NVL72 headline actually measure?** — It is a system-level composite (per-GPU gain × 9x more GPUs per rack), not a per-chip multiplier; NVIDIA's own per-GPU figures range from ~3.1-3.4x (MLPerf Llama workloads) to ~44x (GPT-MoE-1.8T tokens/sec/GPU), so the "30x" varies by workload and is not restated consistently across NVIDIA's own materials [S3, S5, S6].
- **Whose next-gen flagship wins on paper: NVIDIA Rubin or AMD MI455X?** — AMD's own comparison table shows MI455X ahead on every listed metric (40 vs 35 PFLOPS FP4, 432GB vs 288GB memory, 23.3 vs 22.0 TB/s bandwidth) [S37]; NVIDIA's own materials state a different, non-comparable headline (50 PFLOPS NVFP4 "for inference") and never state an HBM4 bandwidth figure at all, so AMD's 35 PFLOPS/22.0TB/s "Rubin" numbers are AMD's own characterization of a competitor's part, not confirmed by NVIDIA [S30, S31, S37].
- **Is Google TPU actually $0.39/chip-hour as Introl claimed?** — Google's own live TPU pricing page shows Trillium's cheapest listed rate (3-year commitment) at $1.22/chip-hour and on-demand at $2.70/chip-hour — more than 3x Introl's figure and nowhere close to $0.39 at any commitment tier [S38], directly corroborating S28's independent conclusion that the Introl claim is unsourced.

## Sources

### S1 · official · NVIDIA — H100 GPU (product page) (2026)
https://www.nvidia.com/en-us/data-center/h100/ · accessed 2026-09-17 · Vendor primary specification page for H100 SXM/NVL.
Facts:
- H100 SXM FP8 Tensor Core 3,958 teraFLOPS, FP16/BF16 Tensor Core 1,979 teraFLOPS, TF32 989 teraFLOPS (Product Specifications table)
- H100 SXM 80GB memory at 3.35TB/s bandwidth, TDP up to 700W, NVLink 900GB/s (Product Specifications table)
- H100 NVL variant: 94GB at 3.9TB/s, NVLink 600GB/s, TDP 350-400W (Product Specifications table)

### S2 · official · NVIDIA — H200 GPU (product page) (2026)
https://www.nvidia.com/en-us/data-center/h200/ · accessed 2026-09-17 · Vendor primary spec page; only Hopper SKU with HBM3e/141GB.
Facts:
- 141GB HBM3e at 4.8TB/s, nearly double H100 capacity, 1.4x more bandwidth (Memory section)
- Up to 1.9x faster Llama2 70B inference, 1.6x faster GPT-3 175B vs H100 (Highlights)
- Up to 7 MIG instances, Confidential Computing (Named features)

### S3 · official · NVIDIA — GB200 NVL72 (product page) (2026)
https://www.nvidia.com/en-us/data-center/gb200-nvl72/ · accessed 2026-09-17 · Vendor primary spec sheet for the rack-scale Blackwell unit.
Facts:
- Rack: 36 Grace CPU, 72 Blackwell GPUs; NVFP4 1,440/720 PFLOPS sparse/dense (Specs table)
- 13.4TB HBM3e at 576TB/s aggregate; NVLink 130TB/s across rack (Specs table)
- 30x LLM inference / 4x training / 25x energy efficiency vs H100, measured at TTL=50ms/FTL=5s/32,768 in/1,024 out tokens (Highlights + footnote)
Conflicts: S6 gives per-GPU throughput (150 vs 3.4 tok/s/GPU, ~44x) that doesn't arithmetically match this page's 30x system-level headline.

### S4 · official · NVIDIA — Blackwell Architecture (technology page) (2026)
https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/ · accessed 2026-09-17 · Vendor architecture overview (PDF technical brief was gated; this HTML page is the fallback primary source).
Facts:
- 208 billion transistors, TSMC 4NP, dual-die, 10TB/s die-to-die interconnect (Architecture overview)
- NVLink Switch: 130TB/s in NVL72, scales to 576 GPUs at 1.8TB/s (NVLink section)
- Native FP4; second-gen Transformer Engine, decompression engine, RAS engine, confidential computing (TEE-I/O)

### S5 · blog · NVIDIA Developer Blog — Blackwell Delivers Massive Performance Leaps in MLPerf Inference v5.0 (2025)
https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/ · accessed 2026-09-17 · Vendor's own MLPerf v5.0 submission breakdown.
Facts:
- GB200 NVL72: up to 3.4x per-GPU, 30x system-level vs 8x H200 on Llama 3.1 405B (Llama 3.1 405B section)
- Llama 2 70B Server: B200 98,443 tok/s vs H200 33,072 tok/s (3x); Offline 98,858 vs 34,988 (2.8x) (Results table)
- Mixtral 8x7B Server: B200 126,845 tok/s vs H200 61,802 tok/s (2.1x) (Results table)
Conflicts: NVIDIA's optimized TRT-LLM submissions vs. S25's independent open-framework measurements showing smaller gaps.

### S6 · blog · NVIDIA Developer Blog — GB200 NVL72 Delivers Trillion-Parameter LLM Training and Real-Time Inference (2024)
https://developer.nvidia.com/blog/nvidia-gb200-nvl72-delivers-trillion-parameter-llm-training-and-real-time-inference/ · accessed 2026-09-17 · Vendor blog with GPT-MoE-1.8T benchmark configuration.
Facts:
- GB200 150 tok/s/GPU vs H100 3.4 tok/s/GPU on GPT-MoE-1.8T (AI Inference Performance)
- 4x faster training with 32k GB200 NVL72 GPUs vs same H100 count (AI Training Performance)
- Per-GPU NVLink 1.8TB/s bidirectional, 14x PCIe Gen5; rack >1PB/s total bandwidth, 30TB unified memory

### S7 · official · AMD — MI300X Accelerator Data Sheet (2025)
https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf · accessed 2026-09-17 · Vendor data sheet, primary spec source for MI300X (CDNA3).
Facts:
- 304 CUs, 1216 matrix cores, 19,456 stream processors, 2100MHz (Specifications table, p.1)
- Up to 192GB HBM3 at 5.3TB/s max peak theoretical, 256MB Infinity Cache (Specifications table, p.1)
- FP16/BF16 1307.4/2614.9 TFLOPs dense/sparse; FP8 2614.9/5229.8 TFLOPs (AI Peak Theoretical table)
- 7 Infinity Fabric links at 128GB/s, 1 PCIe Gen5 x16, max TBP 750W

### S8 · official · AMD — MI325X Accelerator Data Sheet (2025)
https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x-datasheet.pdf · accessed 2026-09-17 · Vendor data sheet for the HBM3E memory-upgraded CDNA3 refresh.
Facts:
- Up to 256GB HBM3E at 6TB/s max peak theoretical (Specifications table, p.1)
- Same compute die config as MI300X: 304 CUs, 1216 matrix cores, 19,456 stream processors
- 256GB enables single accelerator to hold a one-trillion-parameter model; 8x platform = 2TB coherent shared memory, drop-in MI300X replacement
- Max TBP 1000W

### S9 · official · AMD — MI350X Platform Data Sheet (2025)
https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi350x-platform-brochure.pdf · accessed 2026-09-17 · Vendor data sheet for the air-cooled 8-GPU MI350X platform (CDNA4).
Facts:
- 8x platform total: 2048 CUs, 8192 matrix cores, 131,072 stream processors, 2200MHz
- Platform total 2.3TB HBM3E, 8TB/s per-GPU bandwidth; 7 links at 153.6GB/s bidirectional per GPU
- Max TBP 1000W per module (air-cooled); UBB 2.0 allows seamless upgrade across MI300X/MI325X/MI350X
Conflicts: S10 (MI355X) is the same generation at 1400W (liquid-cooled) vs this sheet's 1000W (air-cooled).

### S10 · official · AMD — MI355X GPU Data Sheet (2025)
https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf · accessed 2026-09-17 · Vendor data sheet for AMD's liquid-cooled CDNA4 flagship with native MXFP4/MXFP6.
Facts:
- 256 CUs, 1024 matrix cores, 16,384 stream processors, 2.4GHz (Specifications table, p.1)
- 288GB HBM3E at 8TB/s, 256MB Infinity Cache (Specifications table, p.1)
- MXFP4/MXFP6 peak 10.0663 PFLOPS dense; INT8 5.0332/10.0664 POPS dense/sparse (AI Peak Theoretical table)
- Max TBP 1400W, liquid-cooled, 2U-capable; ROCm supports PyTorch, TensorFlow, JAX, SGLang, Triton, vLLM

### S11 · analysis · Chips and Cheese — AMD's CDNA 4 Architecture Announcement (2025)
https://chipsandcheese.com/p/amds-cdna-4-architecture-announcement · accessed 2026-09-17 · Independent microarchitecture teardown site.
Facts:
- CDNA4 roughly doubles per-CU matrix throughput vs CDNA3, matches B200 SMs in FP6; B200 SMs still 2x per-clock throughput of a CDNA4 CU in 16-/8-bit types
- LDS grows from 64KB (CDNA3) to 160KB (CDNA4), read bandwidth doubles to 256B/clock
- MI300X ~0.03 bytes DRAM bandwidth/FP32 FLOP; MI355X improves to ~0.05

### S12 · official · Google Cloud — TPU v5e documentation (2026)
https://docs.cloud.google.com/tpu/docs/v5e · accessed 2026-09-17 · Vendor system-architecture docs for the cost-optimized inference TPU.
Facts:
- 197 TFLOPs bf16, 393 TOPs Int8 per chip; 16GB HBM at 800GiBps (System architecture table)
- 400GBps bidirectional ICI, 2D torus, pod size 256 chips; pod peak 50.63 PFLOPs bf16 (System architecture table)
- Serving slices limited to 1x1/2x2/2x4; multi-host inference needs Sax; training scales to 256 chips

### S13 · official · Google Cloud — TPU v5p documentation (2026)
https://docs.cloud.google.com/tpu/docs/v5p · accessed 2026-09-17 · Vendor docs for the training/performance-oriented sibling of v5e.
Facts:
- 8960 chips/pod, 459 TFLOPs bf16 and FP8 per chip; 95GiB HBM at 2765GBps (System architecture table)
- 1200GBps bidirectional ICI, 3D torus; single-slice training up to 6144 chips, Multislice to 18,432
- Full pod: 17,920 TensorCores, 8,960 chips, 2,240 hosts, 140 cubes

### S14 · official · Google Cloud — TPU v6e (Trillium) documentation (2026)
https://docs.cloud.google.com/tpu/docs/v6e · accessed 2026-09-17 · Vendor docs for the sixth-gen cost-efficient TPU.
Facts:
- 918 TFLOPs bf16, 1836 TOPs Int8 per chip; 32GB HBM at 1638GBps (System architecture table)
- 800GBps bidirectional ICI, 2D torus, pod size 256; pod: 102.4TB/s all-reduce, 25.6Tbps DCN
- Adds SparseCore for embedding-heavy workloads

### S15 · official · Google Cloud — TPU7x (Ironwood) documentation (2026)
https://docs.cloud.google.com/tpu/docs/tpu7x · accessed 2026-09-17 · Vendor docs for TPU v7 (Ironwood).
Facts:
- 2307 TFLOPs bf16, 4614 TFLOPs FP8 per chip; 192GiB HBM at 7380GBps (~7.37TB/s)
- 9216 chips/pod, 1200GBps bidirectional ICI; dual-chiplet package, each chiplet self-contained with 1 TensorCore, 2 SparseCores, 96GB HBM
- Max documented VM topology 8x16x16 (2048 chips, 512 hosts, 32 cubes)

### S16 · blog · Google Cloud Blog — Inside the Ironwood TPU codesigned AI stack (2025)
https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack · accessed 2026-09-17 · Vendor blog with full-superpod figures and inference-era positioning.
Facts:
- Full superpod: 42.5 FP8 ExaFLOPS, 1.77PB directly addressable HBM across 9,216 chips
- 3D torus, 6 neighbors/chip, 64-chip "cube" per rack, 144 cubes = superpod
- Perf/watt 2x Trillium (v6e), ~30x vs 2018 first Cloud TPU
- Explicitly engineered for two-phase inference: large-batch prefill + memory-bandwidth-bound decode

### S17 · official · AWS Neuron Docs — Trainium2 Architecture (2026)
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html · accessed 2026-09-17 · Vendor per-chip architecture doc.
Facts:
- 8x NeuronCore-v3/chip: 1,299 FP8 TFLOPS, 667 BF16/FP16/TF32 TFLOPS, 2,563 sparse TFLOPS, 181 FP32 TFLOPS
- 96GiB device memory at 2.9TB/s, 3.5TB/s DMA bandwidth; NeuronLink 1.28TB/s/chip
- NeuronCore-v3 SBUF 224MiB (4.7x improvement over v2); supports stochastic rounding, custom ops via GPSIMD

### S18 · official · AWS Neuron Docs — Trn2 Architecture (2026)
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn2-arch.html · accessed 2026-09-17 · Vendor instance-level architecture doc.
Facts:
- trn2.48xlarge: 16 chips in 4x4 2D torus, 192 vCPUs, 1,536GiB device memory, 46.4TB/s bandwidth
- Trn2 UltraServer = 4x trn2u.48xlarge = 64 chips, 768 vCPUs, 6,144GiB device memory, 185.6TB/s bandwidth
- Intra-instance NeuronLink-v3 1,024GB/s/chip; inter-instance (UltraServer only) +256GB/s/chip; EFAv3 3,200Gbps

### S19 · official · AWS Neuron Docs — Inf2 Architecture (2026)
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/inf2-arch.html · accessed 2026-09-17 · Vendor instance-level doc for full Inf2 family.
Facts:
- Scales 1 chip (inf2.xlarge, 16GiB) to 12 chips (inf2.48xlarge, 768GiB)
- Aggregate 190-2,280 FP8/FP16/BF16/TF32 TFLOPS across the range; device memory bandwidth 820-9,840 GiB/s
- Multi-chip instances use NeuronLink-v2 at 192GiB/s/chip

### S20 · blog · AWS ML Blog — Inferentia2 builds on Inferentia1 (2023)
https://aws.amazon.com/blogs/machine-learning/aws-inferentia2-builds-on-aws-inferentia1-by-delivering-4x-higher-throughput-and-10x-lower-latency/ · accessed 2026-09-17 · Only readily available head-to-head Inf1 vs Inf2 chip spec comparison.
Facts:
- Inf1: 4x NeuronCore-v1, 128 INT8 TOPS, 64 FP16/BF16 TFLOPS, 8GB DDR4 at 50GB/s
- Inf2: 2x NeuronCore-v2, 380 INT8 TOPS, 190 FP16/BF16/cFP8/TF32 TFLOPS, 32GB HBM at 820GB/s
- Inf2 up to 4x throughput, 10x lower latency, 16.4x memory bandwidth, 4x memory capacity vs Inf1
- Inf2 supports 175B-parameter models split across accelerators via NeuronLink

### S21 · official · AWS — Amazon EC2 Trn2 Instances (product page) (2026)
https://aws.amazon.com/ec2/instance-types/trn2/ · accessed 2026-09-17 · Vendor product page with system-level FP8-petaflop and price-performance claims.
Facts:
- Trn2: 16 chips, 1.5TB HBM3, up to 20.8 FP8 petaflops, 46TB/s bandwidth, 3.2Tbps EFAv3
- UltraServer: 64 chips, 6TB shared memory, up to 83.2 FP8 petaflops, 185TBps bandwidth, 12.8Tbps EFAv3
- Claimed 30-40% better price-performance than P5e/P5en; 3x more energy efficient than Trn1

### S22 · official · AWS — Trn3 UltraServers announcement (2025)
https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-ec2-trn3-ultraservers/ · accessed 2026-09-17 · Roadmap context only (generation after Trainium2).
Facts:
- Trainium3 chip: 2.52 PFLOPs FP8, 144GB HBM3e at 4.9TB/s, first 3nm AWS AI chip
- Trn3 UltraServer: up to 144 chips, 362 FP8 PFLOPs, 20.7TB HBM3e, 706TB/s aggregate bandwidth
- vs Trainium2: 1.5x memory, 1.7x bandwidth per chip; system-level up to 4.4x performance, 4x perf/watt
- Announced Dec 2, 2025; new NeuronSwitch-v1 all-to-all fabric

### S23 · standard · MLCommons — MLPerf Inference v5.0 Results (2025)
https://mlcommons.org/2025/04/mlperf-inference-v5-0-results/ · accessed 2026-09-17 · Standards body's own announcement.
Facts:
- 17,457 results from 23 organizations; Llama 2 70B submissions grew 2.5x YoY, median score doubled
- New benchmarks: Llama 3.1 405B (128K context), Llama 2 70B Interactive, RGAT, Automotive PointPainting
- New accelerators: AMD MI325X, Google TPU Trillium (v6e), NVIDIA B200/GB200
- Published April 2, 2025

### S24 · standard · MLCommons — MLPerf Inference v5.1 Results (2025)
https://mlcommons.org/2025/09/mlperf-inference-v5-1-results/ · accessed 2026-09-17 · Most recent round as of research date; first with DeepSeek-R1 and MI355X.
Facts:
- 27 submitting organizations (record), 24 on Llama 2 70B; best systems up to 50% faster than best v5.0 system
- New benchmarks: DeepSeek-R1, Llama 3.1 8B (128K context), Whisper Large V3
- New accelerators: AMD MI355X, Intel Arc Pro B60, NVIDIA GB300, RTX 4000 Ada-PCIe, RTX Pro 6000 Blackwell Server Edition
- Released September 9, 2025
Conflicts: v5.0 (S23) has no DeepSeek-R1/MI355X entries — the two rounds aren't directly comparable for those workloads.

### S25 · analysis · SemiAnalysis — InferenceMAX: Open Source Inference Benchmarking (2025)
https://newsletter.semianalysis.com/p/inferencemax-open-source-inference · accessed 2026-09-17 · Independent, continuously-run cross-hardware benchmark (GB200 NVL72, B200, MI355X, H200, MI325X, H100, MI300X) on vLLM/SGLang/TRT-LLM.
Facts:
- MI300X wins below ~20-30 tok/s/user on Llama 3.3 70B FP8 (memory-bandwidth advantage at TP1)
- MI355X on vLLM has lower TCO/M-tokens than B200 on vLLM (GPT-OSS 120B FP4)
- GB200 NVL72 decisively wins below 90 tok/s/user on DeepSeek R1 FP4; single B200 wins above 90
- Power efficiency: MI355X ~2.55M tok/s/MW (~3x MI300X); B200 ~2.8M tok/s/MW (~3x H100); B200 ~20% more efficient than MI355X same-gen
- Electricity <20% of TCO; most TCO is GPU vendor gross margin
- Methodology: 1024/1024, 1024/8192, 8192/1024 in/out patterns; random sequences (no prefix caching); infinite request rate, swept concurrency
- TPU and Trainium not yet included as of this benchmark (planned)
Conflicts: Shows smaller, more workload-dependent Blackwell-vs-Hopper gaps than NVIDIA's own MLPerf submissions (S5); notes vLLM-on-Hopper can beat "mostly open" TRT-LLM.

### S26 · analysis · SemiAnalysis — AMD vs NVIDIA Inference Benchmark (2025)
https://newsletter.semianalysis.com/p/amd-vs-nvidia-inference-benchmark-who-wins-performance-cost-per-million-tokens · accessed 2026-09-17 · Independent GEMM + end-to-end serving benchmark predating InferenceMAX.
Facts:
- Measured GEMM: H100/H200 ~720 TFLOP/s BF16 vs MI300X ~620 (14% slower); FP8 ~1,280 vs ~990 (22% slower)
- Llama3 70B FP16 chat: NVIDIA wins low-latency; MI325X wins at higher batch/concurrency
- Llama3 405B FP8: MI325X consistently outperforms H200/MI300X/H100
- Conclusion: no universal perf/$ winner — depends on workload and latency target
Conflicts: Vendor peak-theoretical FLOPS (S7/S8) diverge 14-22% from these measured GEMM numbers.

### S27 · analysis · Artificial Analysis — Independent Performance Analysis of Leading GPUs (2025)
https://artificialanalysis.ai/articles/independent-analysis-of-leading-gpus-amd-nvidia · accessed 2026-09-17 · Independent lab measuring real rented cloud instances.
Facts:
- DeepSeek R1 @16 concurrency: H200 ~45 tok/s/query & ~600 tok/s system vs MI300X ~35 tok/s/query & ~500 tok/s system
- Llama 4 Maverick @16 concurrency: H100 ~90 tok/s/query & ~2,150 tok/s system vs MI300X ~70 tok/s/query & ~2,100 tok/s system
- At peak concurrency, MI300X reaches higher absolute system throughput on both models tested
- Llama 3.3 70B cost/M-tokens @30 tok/s/user: H100 $1.06, MI300X $2.24, TPU v6e $5.13
- Published June 8, 2025; no B200/MI325X data in this article
Conflicts: Live AA dashboard could not be fetched directly; the $ figures are corroborated independently by S28's re-derivation of the same dataset.

### S28 · analysis · Junyi Hou — Debunking introl/ainewshub "TPU is 4x Cheaper" claim (2026)
https://www.junyi.dev/en/posts/tpu-tco/ · accessed 2026-09-17 · Fact-check cross-referencing MLCommons database and Artificial Analysis' primary data.
Facts:
- The "4.7x better perf/$" claim attributed to "MLPerf v4.1" is unfounded — v4.1's only TPU submission was stable-diffusion-xl, no LLM
- Introl vs ainewshub give inconsistent multipliers (4x vs 4.7x) and inconsistent MLPerf versions (v3.1 vs v4.1)
- Re-derived from Artificial Analysis: peak-throughput TPU v6e $0.62/M-tokens vs H100 $0.67-0.69 (competitive); interactive serving TPU v6e $5.13 vs H100 $1.06 (~5x gap favoring NVIDIA)
- Conclusion: TPU economics favor offline/batch, underperform in online/interactive serving

### S29 · blog · Introl (Blake Crosley) — Google TPU vs NVIDIA GPU Decision Framework 2025 (2026)
https://introl.com/blog/google-tpu-vs-nvidia-gpu-infrastructure-decision-framework-2025 · accessed 2026-09-17 · Practitioner decision-criteria framework. **Use with caution** — specific cost figures disputed by S28 as unsourced/unverifiable; not independently verified here.
Facts:
- Decision axes: utilization threshold (>70% favors TPU), workload type (training→v5p, inference→v6e), framework (JAX/TF native = TPU fit), constraints (multi-cloud mandate = GPU only)
- Claims TPU v6e 4x better price-performance than H100 at $0.39/chip-hour [UNVERIFIED, see S28]
- Same article's own numbers show TPU v6e ~120 tok/s vs H100/H200 ~150 tok/s at low concurrency (internally inconsistent with its "4x better" headline)
- "TPUs optimize for throughput per dollar rather than raw speed"; hybrid multi-vendor strategies recommended over pure single-vendor
Conflicts: Directly contradicted by S28's fact-check of the same "4x cheaper" claim.

### S30 · official · NVIDIA Newsroom — Rubin Platform Announcement (2026)
https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer · accessed 2026-09-17 · Official press release, published 2026-01-05.
Facts:
- Platform: Vera CPU, Rubin GPU, NVLink 6 Switch, ConnectX-9 SuperNIC, BlueField-4 DPU, Spectrum-6 switch, Vera Rubin NVL72, HGX Rubin NVL8, Groq 3 LPU
- NVLink 6: 3.6TB/s per GPU, 260TB/s per NVL72 rack; Rubin GPU 50 petaflops NVFP4 (inference)
- NVL72 = 72 Rubin GPUs + 36 Vera CPUs; Vera CPU has 88 custom Olympus cores
- Claimed 10x lower inference token cost, 4x fewer GPUs to train MoE, vs Blackwell; availability 2H 2026
Conflicts: AMD's own comparison chart (S37) states Rubin at 35 PFLOPS FP4 / 22.0TB/s — neither figure appears on this or NVIDIA's own pages.

### S31 · official · NVIDIA — Vera Rubin Platform (technology page) (2026)
https://www.nvidia.com/en-us/data-center/technologies/rubin/ · accessed 2026-09-17 · Vendor product/technology page.
Facts:
- NVL72 = 72 Rubin GPUs, 36 Vera CPUs, ConnectX-9 SuperNICs, BlueField-4 DPUs
- Vera CPU rack: 256 CPUs, supports 22,500+ concurrent sandbox environments
- NVL4 claimed 4x scientific sim / 6x AI-for-Science training / 8x inference vs Hopper; "now in full production"

### S32 · blog · NVIDIA Developer Blog — Vera Rubin POD: Seven Chips, Five Rack-Scale Systems (2026)
https://developer.nvidia.com/blog/nvidia-vera-rubin-pod-seven-chips-five-rack-scale-systems-one-ai-supercomputer/ · accessed 2026-09-17 · Vendor technical blog with per-component detail.
Facts:
- Rubin GPU 3.6TB/s per-GPU, 260TB/s per-rack NVLink 6; Vera CPU rack 256 CPUs, liquid-cooled
- 8 ConnectX-9 SuperNICs per compute tray; BlueField-4 = Vera CPU + ConnectX-9 combined
- Spectrum-6 SPX switch: 102.4Tb/s, 512 lanes at 200Gb/s

### S33 · official · NVIDIA Newsroom — Rubin CPX Announcement (2025)
https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference · accessed 2026-09-17 · Official press release, published 2025-09-09.
Facts:
- Rubin CPX: 30 petaflops NVFP4, 128GB GDDR7 memory, 3x faster attention vs GB300 NVL72
- Vera Rubin NVL144 CPX rack: 8 exaflops AI performance, 100TB memory, 1.7PB/s memory bandwidth, 7.5x vs GB300 NVL72

### S34 · official · AMD Investor Relations — Advancing AI 2025 Press Release (2025)
https://ir.amd.com/news-events/press-releases/detail/1255/amd-unveils-vision-for-an-open-ai-ecosystem-detailing-new-silicon-software-and-systems-at-advancing-ai-2025 · accessed 2026-09-17 · Official IR press release, published 2025-06-12.
Facts:
- Helios rack: MI400 Series GPUs + Zen 6 EPYC "Venice" CPUs + Pensando "Vulcano" NICs
- MI400 claimed up to 10x more MoE inference performance (vs prior gen)
- MI355X claimed up to 40% more tokens-per-dollar vs competing solutions (vs B200); MI350 broad availability 2H 2025

### S35 · analysis · SemiAnalysis — AMD Advancing AI: MI350X and MI400 UALoE72, MI500 UAL256 (2025)
https://newsletter.semianalysis.com/p/amd-advancing-ai-mi350x-and-mi400-ualoe72-mi500-ual256 · accessed 2026-09-17 · Independent roadmap analysis.
Facts:
- MI400/Helios (UALoE72): 72 logical GPUs, UALink over Ethernet, potentially competitive with NVIDIA VR200 NVL144 in H2 2026
- MI500/UAL256: 256 physical/logical chips vs VR300 NVL576's 144; targeted late 2027

### S36 · official · AMD — Helios Rackscale Solution (product page) (2026)
https://www.amd.com/en/products/rackscale-solutions/helios.html · accessed 2026-09-17 · Vendor product page, fetched via curl (static HTML).
Facts:
- Helios rack: 72x MI455X, 2.9 EF FP4, 1.4 EF FP8, 31TB HBM4, 23.3TB/s per-GPU bandwidth, 260TB/s scale-up, 43TB/s scale-out
- Volume deployments expected 2H 2026; figures footnoted as AMD Performance Labs peak-theoretical projections (June 2026), not measured

### S37 · official · AMD — Instinct MI400 Series GPUs (product page) (2026)
https://www.amd.com/en/products/accelerators/instinct/mi400.html · accessed 2026-09-17 · Vendor product page with direct MI455X-vs-Vera-Rubin comparison table, fetched via curl.
Facts:
- MI455X: 5th-gen CDNA, 256 WGP, 3.6TB/s scale-up/GPU, 432GB HBM4 at 23.3TB/s
- AMD's comparison table: MI455X vs NVIDIA Vera Rubin — FP4 40 vs 35 PFLOPS, FP8/FP6 20 vs 18, FP16/BF16 5 vs 4, memory 432GB vs 288GB, bandwidth 23.3 vs 22.0TB/s
- MI430X (HPC/sovereign AI, FP64) available 2027; MI455X/Helios volume deployments 2H 2026
Conflicts: NVIDIA's own pages (S30) state 50 PFLOPS NVFP4 for Rubin, not the 35 PFLOPS this chart attributes to it — bases not stated on either side.

### S38 · official · Google Cloud — Cloud TPU Pricing (2026)
https://cloud.google.com/tpu/pricing · accessed 2026-09-17 · Vendor pricing page, fetched via curl (static HTML table — NOT JS-only, contrary to the original pass's assumption).
Facts:
- Ironwood: on-demand $12.00/chip-hr (us-central1), 1-yr $8.40, 3-yr $5.40; $13.20 in europe-west2
- Trillium (v6e): on-demand $2.70/chip-hr (us-east1/us-east5), 1-yr $1.89, 3-yr $1.22; up to $3.24 (asia-northeast1)
- TPU v5p: on-demand $4.20/chip-hr, 1-yr $2.94, 3-yr $1.89
- TPU v5e: on-demand $1.20/chip-hr, 1-yr $0.84, 3-yr $0.54 (region-dependent, up to $1.56)
Conflicts: directly contradicts S29's claimed $0.39/chip-hour TPU v6e figure — cheapest real rate (3-yr commit) is $1.22.

### S39 · official · Google Cloud — GPU / Accelerator-optimized VM Pricing (2026)
https://cloud.google.com/products/compute/gpus-pricing · accessed 2026-09-17 · Vendor pricing page, fetched via curl (static HTML table).
Facts:
- a3-ultragpu-8g (8x H200): on-demand $84.806908493/hr
- a3-megagpu-8g (8x H100): on-demand $93.400712807/hr; a3-highgpu-8g (8x H100): on-demand $88.490000119/hr
- a4-highgpu-8g (8x B200): standard on-demand listed N/A; DWS Flex-start $64.44/hr, Calendar Mode $90.22/hr, Spot $39.6336/hr
- a2-ultragpu-8g (8x A100 80GB): on-demand $40.550383123/hr

### S40 · official · AWS — Amazon EC2 Inf2 Instances (pricing table) (2026)
https://aws.amazon.com/ec2/instance-types/inf2/ · accessed 2026-09-17 · Vendor page with server-rendered pricing table, fetched via curl (unique among P5/Trn/Inf2 pages — the others load pricing via JS).
Facts:
- inf2.xlarge: On-Demand $0.76/hr, 1-yr $0.45, 3-yr $0.30
- inf2.8xlarge: On-Demand $1.97/hr; inf2.24xlarge: On-Demand $6.49/hr; inf2.48xlarge: On-Demand $12.98/hr, 3-yr $5.19

### S41 · official · Microsoft Azure — Retail Prices API, ND96isr_MI300X_v5 (2026)
https://prices.azure.com/api/retail/prices?$filter=armSkuName%20eq%20%27Standard_ND96isr_MI300X_v5%27 · accessed 2026-09-17 · Official public, unauthenticated Azure pricing API, queried directly.
Facts:
- ND96isr_MI300X_v5 (8x MI300X): on-demand Linux Consumption price $48.00/hr in eastus2/westus3 (lowest), up to $96.00/hr in brazilsouth, effective 2025-10-01
- Only official cloud on-demand AMD Instinct price found in this pass (AWS/GCP do not list MI300X/MI325X pricing)

### S42 · analysis · Vantage (instances.vantage.sh) — EC2 pricing pages (2026)
https://instances.vantage.sh/aws/ec2/p5.48xlarge · accessed 2026-09-17 · Third-party pricing aggregator, used where AWS's own pages don't embed pricing.
Facts:
- p5.48xlarge (8x H100): On-Demand $55.040/hr, Spot $20.787/hr, 1-yr Reserved $23.777/hr
- p5en.48xlarge (8x H200): On-Demand $63.296/hr; trn1.32xlarge: On-Demand $21.500/hr; inf1.24xlarge: On-Demand $4.721/hr
Conflicts: this site's p5e.48xlarge ($1.843/hr) and trn2.48xlarge ($0.125/hr, spot priced HIGHER than on-demand) entries are internally inconsistent and NOT used — see S43/S44 for the figures actually used instead.

### S43 · analysis · Spare Cores — trn2.48xlarge live pricing snapshot (2026)
https://sparecores.com/server/aws/trn2.48xlarge · accessed 2026-09-17 · Open-source cloud pricing/benchmark tracker, live snapshot 2026-09-17.
Facts:
- trn2.48xlarge (16x Trainium2) on-demand reference price: $13.971-$17.268/hr depending on region/zone
- Spot prices sampled $1.1947-$3.2828/hr

### S44 · blog · Techzine (citing DataCenterDynamics) — AWS Increases EC2 Capacity Block Prices by 15% (2026)
https://www.techzine.eu/news/infrastructure/137670/aws-increases-ec2-capacity-block-prices-by-15-percent/ · accessed 2026-09-17 · Secondary press, published 2026-01-06.
Facts:
- p5e.48xlarge (8x H200): on-demand rose from $34.61 to $39.80/hr; p5en.48xlarge: $36.18 to $41.61/hr
- ~15% increase across GPU-based instances, effective early January 2026

### S45 · standard · MLCommons — MLPerf Training v5.0 Results (2025)
https://mlcommons.org/2025/06/mlperf-training-v5-0-results/ · accessed 2026-09-17 · Standards body's own announcement, published 2025-06-04.
Facts:
- 201 results from 20 organizations (AMD, NVIDIA, Google Cloud, CoreWeave, Dell, Oracle, and 14 others)
- New benchmark: Llama 3.1 405B pretraining (replaces GPT-3); new processors: MI300X, MI325X, GB200, B200-SXM-180GB, TPU-trillium

### S46 · standard · MLCommons — MLPerf Training v5.1 Results (2025)
https://mlcommons.org/2025/11/training-v5-1-results/ · accessed 2026-09-17 · Standards body's own announcement, published 2025-11-12.
Facts:
- 65 unique systems, 12 accelerator types, 20 organizations; nearly half multi-node (+86% vs v4.1)
- New benchmarks: Llama 3.1 8B (replaces BERT), Flux.1 (replaces Stable Diffusion v2)

### S47 · blog · NVIDIA Developer Blog — Blackwell Delivers up to 2.6x Higher MLPerf Training v5.0 Performance (2025)
https://developer.nvidia.com/blog/nvidia-blackwell-delivers-up-to-2-6x-higher-performance-in-mlperf-training-v5-0/ · accessed 2026-09-17 · Vendor blog with per-benchmark time-to-train.
Facts:
- Llama 3.1 405B pretraining, 512 GPUs: Hopper 269.12min vs Blackwell 121.09min (2.2x)
- Llama 2 70B LoRA, 8 GPUs: Hopper 27.93min vs Blackwell 11.14min (2.51x); up to 2.6x across all seven benchmarks

### S48 · blog · NVIDIA Developer Blog — Blackwell Architecture Sweeps MLPerf Training v5.1 (2025)
https://developer.nvidia.com/blog/nvidia-blackwell-architecture-sweeps-mlperf-training-v5-1-benchmarks/ · accessed 2026-09-17 · Vendor blog, first Blackwell Ultra training results.
Facts:
- Llama 3.1 405B: 10min at 5,120 Blackwell GPUs (2.7x vs last round, 3x vs same-count Hopper), 85% scaling efficiency 512→5,120 GPUs
- Llama 3.1 8B: 5.2min at 512 Blackwell Ultra GPUs; Llama 2 70B LoRA: 0.40min at 512 Blackwell Ultra GPUs

### S49 · blog · AMD — AMD Expands AI Momentum with First MLPerf Training Submission (2025)
https://www.amd.com/en/blogs/2025/amd-drives-ai-gains-with-mlperf-training-results.html · accessed 2026-09-17 · Official AMD blog, published 2025-06-04.
Facts:
- AMD's first-ever MLPerf Training submission; first-ever multi-node submission on Instinct hardware
- MI325X beats NVIDIA H200 by up to 8% on Llama 2-70B-LoRA fine-tuning; MI300X "competitive" vs H100 on same workload
- Partner results (Llama2-70B-LoRA): Supermicro liquid-cooled MI325X 21.75min; MangoBoost 4-node/32-GPU MI300X 10.92min; Dell MI300X 8-GPU 28.99min; Gigabyte MI325X 8-GPU 22.1min

### S50 · blog · Google Cloud Blog — Trillium MLPerf 4.1 Training Benchmarks (2024)
https://cloud.google.com/blog/products/compute/trillium-mlperf-41-training-benchmarks/ · accessed 2026-09-17 · Vendor blog, published 2024-11-14 (MLPerf Training v4.1 round — most recent official Google training comparison found; no v5.0/v5.1 equivalent published).
Facts:
- Trillium: up to 1.8x perf/$ vs TPU v5p, 99% weak-scaling efficiency (vs v5p's 94%), 1.8x (45% lower) cost-to-train on GPT3-175b
- Configs: 4x/8x Trillium-256 vs v5p-4096/v5p-8192

## Not fetched / could not verify

- NVIDIA Blackwell Architecture Technical Brief PDF (nvdam.widen.net) — returned an HTML viewer shell, not PDF bytes. Substituted with nvidia.com HTML pages (S3, S4).
- Official NVIDIA H100 datasheet PDF (resources.nvidia.com and a mirror) — both failed to download as valid PDFs. Substituted with nvidia.com/h100 HTML page (S1).
- Artificial Analysis live hardware-inference-stack dashboard — JS-rendered; used the static article (S27) instead, cross-checked against S28's independent quote of the same dataset.
- AWS P5/P5e/P5en/Trn1/Trn2 product pages do not embed server-rendered pricing (unlike Inf2's, S40); the AWS bulk Price List API (~480MB per region/service) was confirmed to exist but is impractical to fetch/parse at this budget. Used Vantage.sh (S42), Spare Cores (S43), and Techzine/DataCenterDynamics (S44) instead, with two specific Vantage data points (p5e.48xlarge, trn2.48xlarge) excluded as internally inconsistent.
- DataCenterDynamics' own article on the AWS H200 price increase returned HTTP 403 on direct fetch; used Techzine's citation of it instead (S44).
- AWS Trainium/Trainium2 has never submitted to MLPerf Training (v5.0 or v5.1) — genuinely absent, not a fetch failure. No third-party Trainium training benchmark exists to cite.
- Google has not published an official Trillium-vs-v5p (or vs. NVIDIA) MLPerf Training blog for the v5.0 or v5.1 rounds; S50 (the most recent one found) is from the v4.1 round, November 2024.

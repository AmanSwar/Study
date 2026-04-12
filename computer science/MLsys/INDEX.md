# ML Systems PhD Curriculum — Master Index

## Comprehensive Reference for Production Inference Systems Engineering
**Targeting:** Server-Grade CPU, Apple Silicon, On-Device/Edge AI

**Total:** 39 Modules + 5 Appendices | ~1.7 MB | ~150,000+ words

---

## PART I — Inference Optimization Fundamentals
*Hardware-agnostic principles that apply everywhere*

| Module | Topic | File |
|--------|-------|------|
| 1 | The Inference Systems Mental Model | `part_01_inference_fundamentals/module_01_inference_systems_mental_model.md` |
| 2 | Quantization: Complete Systems Perspective | `part_01_inference_fundamentals/module_02_quantization.md` |
| 3 | Sparsity: Structured & Unstructured | `part_01_inference_fundamentals/module_03_sparsity.md` |
| 4 | Knowledge Distillation: Systems Perspective | `part_01_inference_fundamentals/module_04_knowledge_distillation.md` |
| 5 | Graph Optimization & Compilation | `part_01_inference_fundamentals/module_05_graph_optimization_compilation.md` |

## PART II — Transformer Inference Architecture
*The algorithms that make LLM inference fast*

| Module | Topic | File |
|--------|-------|------|
| 6 | Attention Mechanism: Systems Optimization | `part_02_transformer_architecture/module_06_attention_optimization.md` |
| 7 | KV-Cache Management Systems | `part_02_transformer_architecture/module_07_kv_cache_management.md` |
| 8 | Inference Serving Systems Architecture | `part_02_transformer_architecture/module_08_inference_serving_systems.md` |

## PART III — Distributed Inference
*Multi-device, multi-node inference at scale*

| Module | Topic | File |
|--------|-------|------|
| 9 | Parallelism Strategies for Inference | `part_03_distributed_inference/module_09_parallelism_strategies.md` |
| 10 | Communication Primitives & Collective Operations | `part_03_distributed_inference/module_10_communication_primitives.md` |

## PART IV — Server-Grade CPU Inference (MASTER LEVEL)
*Intel Xeon & AMD EPYC*

| Module | Topic | File |
|--------|-------|------|
| 11 | Why CPU Inference Matters & When It Wins | `part_04_cpu_inference/module_11_why_cpu_inference.md` |
| 12 | CPU Inference Engine Internals | `part_04_cpu_inference/module_12_cpu_inference_engines.md` |
| 13 | GEMM Optimization for CPU Inference | `part_04_cpu_inference/module_13_gemm_optimization_cpu.md` |
| 14 | Attention Implementation for CPU | `part_04_cpu_inference/module_14_attention_cpu.md` |
| 15 | Full LLM Inference Pipeline on CPU | `part_04_cpu_inference/module_15_full_pipeline_cpu.md` |

## PART V — Apple Silicon Inference (MASTER LEVEL)
*M-series unified memory architecture*

| Module | Topic | File |
|--------|-------|------|
| 16 | Apple Silicon Architecture for ML Engineers | `part_05_apple_silicon/module_16_apple_silicon_architecture.md` |
| 17 | CoreML & ANE Programming | `part_05_apple_silicon/module_17_coreml_ane.md` |
| 18 | Metal & GPU Inference on Apple Silicon | `part_05_apple_silicon/module_18_metal_gpu_inference.md` |
| 19 | llama.cpp on Apple Silicon | `part_05_apple_silicon/module_19_llamacpp_apple.md` |
| 20 | Full Stack Inference on Apple Silicon | `part_05_apple_silicon/module_20_full_stack_apple.md` |

## PART VI — On-Device & Edge AI Inference (MASTER LEVEL)
*Mobile, embedded, microcontroller*

| Module | Topic | File |
|--------|-------|------|
| 21 | Edge AI Hardware Landscape | `part_06_edge_ai/module_21_edge_hardware_landscape.md` |
| 22 | TinyML: Inference on Microcontrollers | `part_06_edge_ai/module_22_tinyml.md` |
| 23 | Mobile Inference: Android & iOS | `part_06_edge_ai/module_23_mobile_inference.md` |
| 24 | Edge LLM Inference | `part_06_edge_ai/module_24_edge_llm.md` |

## PART VII — NVIDIA GPU Inference
*Working knowledge — focus on what's new beyond Ampere*

| Module | Topic | File |
|--------|-------|------|
| 25 | Modern GPU Inference Stack | `part_07_gpu_inference/module_25_modern_gpu_stack.md` |

## PART VIII — Fine-Tuning Systems
*The infrastructure for adapting trained models*

| Module | Topic | File |
|--------|-------|------|
| 26 | Fine-Tuning Infrastructure | `part_08_fine_tuning/module_26_finetuning_infrastructure.md` |
| 27 | LoRA & Parameter-Efficient Fine-Tuning | `part_08_fine_tuning/module_27_lora_peft.md` |
| 28 | RLHF & Alignment Systems | `part_08_fine_tuning/module_28_rlhf_alignment.md` |

## PART IX — Fast Voice AI Systems (MASTER LEVEL)
*Streaming architecture, sub-100ms TTFA*

| Module | Topic | File |
|--------|-------|------|
| 29 | Voice AI Pipeline Architecture | `part_09_voice_ai/module_29_voice_pipeline_architecture.md` |
| 30 | VAD (Voice Activity Detection) Systems | `part_09_voice_ai/module_30_vad.md` |
| 31 | Streaming ASR | `part_09_voice_ai/module_31_streaming_asr.md` |
| 32 | LLM Integration in Voice Pipeline | `part_09_voice_ai/module_32_llm_voice_integration.md` |
| 33 | TTS (Text-to-Speech) Systems Engineering | `part_09_voice_ai/module_33_tts.md` |
| 34 | Audio Processing & Codec Systems | `part_09_voice_ai/module_34_audio_processing.md` |
| 35 | End-to-End Voice AI Optimization | `part_09_voice_ai/module_35_e2e_voice_optimization.md` |

## PART X — RAG & AI Systems Infrastructure

| Module | Topic | File |
|--------|-------|------|
| 36 | RAG Systems Engineering | `part_10_rag_infrastructure/module_36_rag_systems.md` |
| 37 | Inference Compilation & Deployment Toolchain | `part_10_rag_infrastructure/module_37_deployment_toolchain.md` |
| 38 | Multi-Modal Inference Systems | `part_10_rag_infrastructure/module_38_multimodal_inference.md` |
| 39 | Inference Observability & Production Systems | `part_10_rag_infrastructure/module_39_observability.md` |

## Appendices

| Appendix | Topic | File |
|----------|-------|------|
| A | Inference Benchmark Reference (MLPerf) | `appendices/appendix_a_mlperf_benchmarks.md` |
| B | Model Memory Sizing Calculator | `appendices/appendix_b_memory_calculator.md` |
| C | Latency Budget Templates | `appendices/appendix_c_latency_budgets.md` |
| D | Key Paper Reading List (50 Papers) | `appendices/appendix_d_paper_reading_list.md` |
| E | Hardware Comparison Matrix | `appendices/appendix_e_hardware_comparison.md` |

---

## Recommended Study Order

**Week 1-2: Foundations** — Modules 1, 2, 3, 5 (roofline model, quantization, sparsity, compilation)

**Week 3: Transformer Internals** — Modules 6, 7, 8 (attention, KV-cache, serving)

**Week 4: Your Primary Hardware** — Pick your track:
  - CPU Track: Modules 11-15
  - Apple Silicon Track: Modules 16-20
  - Edge Track: Modules 21-24

**Week 5: Speculative Decoding & Distillation** — Module 4

**Week 6: Distributed Systems** — Modules 9, 10

**Week 7: Fine-Tuning** — Modules 26, 27, 28

**Week 8: Voice AI** — Modules 29-35

**Week 9: RAG & Production** — Modules 36-39

**Week 10: GPU & Integration** — Module 25, Appendices A-E

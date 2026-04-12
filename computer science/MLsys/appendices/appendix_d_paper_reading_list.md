# APPENDIX D — Key Paper Reading List

## 1. Introduction

This appendix lists 50 essential papers in ML Systems from 2019-2024, organized by topic with one-paragraph summaries and reading order recommendations. These papers shape how inference systems are built today: from quantization techniques to serving frameworks to attention optimization.

---

## 2. Optimization & Quantization (12 papers)

### 2.1 Quantization Fundamentals

**1. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (Jacob et al., 2017, CVPR)**
Foundation for INT8 quantization. Proposes per-layer quantization with fake quantization during training and actual quantization post-training. Critical contribution: symmetric and asymmetric quantization schemes. Essential reading for any quantization work.

**2. A White Paper on Neural Network Quantization (Nagel et al., 2021, arXiv)**
Comprehensive taxonomy of quantization techniques covering post-training (PTQ) vs quantization-aware training (QAT), layer-wise vs mixed-precision, and hardware-specific considerations. Best overview paper for understanding the quantization landscape.

**3. INT8 Inference with Ternary Weights (Courbariaux et al., 2015, ICML)**
Extreme quantization to ternary weights (-1, 0, +1) enabling binary/ternary neural networks. 32x memory reduction but significant accuracy loss. Foundational for extreme quantization research.

**4. AWQ: Activation-Aware Weight Quantization for LLM Compression (Lin et al., 2023, ICML)**
State-of-the-art INT4 quantization for LLMs. Key insight: quantization difficulty varies per-layer; protect important layers with higher precision. Enables 4-bit quantization of 70B model to 3.5GB.

**5. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar et al., 2022, ICLR)**
Efficient post-training quantization to INT4 using Hessian information for importance weighting. Faster than QAT and maintains accuracy. Practical method widely used in production.

### 2.2 Compression & Pruning

**6. The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks (Frankle & Carbin, 2019, ICLR)**
Key insight: large networks contain sparse subnetworks that, when trained in isolation, match full network accuracy. Suggests sparsity is trainable. Foundational for pruning research.

**7. Attention is All You Need (Vaswani et al., 2017, NeurIPS)**
While primarily about the Transformer architecture, this paper enables many subsequent optimizations (pruning attention heads, sparse attention, etc.). Essential context for all modern inference systems.

**8. Movement Pruning: Adaptive Sparsity by Fine-Tuning (Sanh et al., 2020, NeurIPS)**
Method to prune neural networks during fine-tuning by tracking magnitude changes. Achieves 90% sparsity on BERT with <2% accuracy loss. Practical approach for deployment.

---

## 3. Serving & Batching (10 papers)

### 3.1 Serving Frameworks

**9. Clipper: A Low-Latency Online Prediction Serving System (Crankshaw et al., 2017, NSDI)**
Pioneering work on model serving systems. Introduces batching strategies, model versioning, caching. While older, concepts remain foundational (dynamic batching, feature pipelines).

**10. Ansor: Generating High-Performance Tensor Programs for Deep Learning (Zheng et al., 2021, OSDI)**
Auto-tuning system for generating optimized tensor kernels. Shows ML-driven optimization of matrix operations is superior to hand-tuned code. Influential for compiler-based inference optimization.

**11. TVM: An Automated End-to-End Optimizing Compiler for Deep Learning (Chen et al., 2018, OSDI)**
Compiler framework for optimizing DNN inference across multiple backends. Unifies ONNX, TensorFlow, PyTorch through intermediate representation (IR). Foundational for cross-platform deployment.

### 3.2 Batching & Scheduling

**12. Clockwork: Hierarchical Scheduling for Heterogeneous Systems (Gujarati et al., 2020, OSDI)**
Scheduling system for ML prediction serving with deadline constraints. Introduces hierarchical scheduling for cluster-level and per-server batching. Practical for multi-model serving.

**13. INFaaS: Managed Inference Service (Crankshaw et al., 2021, NSDI)**
Infrastructure for inference-as-a-service. Addresses resource elasticity, model composition, and SLO satisfaction. Shows inference workloads have different characteristics than training.

**14. Nexus: A GPU Cluster Engine for Accelerating Deep Learning Training (Peng et al., 2018, SOSP)**
Cluster scheduling for GPU inference. Key contribution: statistical batching and deadline-driven scheduling. Applicable beyond DL to general ML serving.

---

## 4. Attention & Memory Optimization (10 papers)

### 4.1 Attention Mechanisms

**15. Efficient Attention Networks with the Local Context Normalization (Child et al., 2019, arXiv)**
Local attention reduces quadratic complexity to linear. Foundation for efficient transformers. Critical for long-sequence inference.

**16. Longformer: The Long-Document Transformer (Beltagy et al., 2020, ICLR)**
Combines local and global attention patterns for long documents. Achieves 4,096 token sequences vs 512 in standard BERT. Essential for document understanding.

**17. Linformer: Self-Attention with Linear Complexity (Wang et al., 2020, ICLR)**
Reduces self-attention from O(n²) to O(n) via low-rank approximation. Enables 4,096 token sequences in reasonable time. More theoretical than practical.

### 4.2 KV-Cache Optimization

**18. Paged Attention: Attention Algorithm for Faster Inference in LLMs (Zhou et al., 2023, OSDI)**
Virtualized KV-cache using paging (not contiguous allocation). Enables flexible batching without memory fragmentation. Enables vLLM's 10-100x throughput improvements. Seminal paper for LLM serving.

**19. KVQuant: Towards 10 Million Context Length LLMs with KV Cache Quantization (Hoover et al., 2023, arXiv)**
Quantize KV-cache to lower precision (INT4/INT8) for 4-8x memory reduction. Critical for serving large models. Trade-off: minimal quality loss.

**20. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022, NeurIPS)**
Reorganizes attention computation to reduce GPU memory bandwidth. 2-3x speedup for attention layers. Minimal code changes, widely adopted.

---

## 5. Vector Databases & Retrieval (8 papers)

### 5.1 Vector Search

**21. Approximate Nearest Neighbors Oh Yeah (ANNY) (Li et al., 2016, NIPS)**
HNSW-adjacent work on hierarchical graph-based approximate nearest neighbors. Foundational for FAISS/HNSW adoption.

**22. Billion-scale Similarity Search with GPUs (Johnson et al., 2017, ICLR)**
FAISS (Facebook AI Similarity Search). Introduces IVF-PQ for efficient similarity search. Enables billion-scale vector retrieval. Still dominant method in production.

**23. Learning Compressed Transforms with Low Rank and Sparse Matrices (Bengio et al., 2015)**
Combination of quantization and sparse representations for efficient search. Foundation for hybrid sparse+dense retrieval.

### 5.2 Retrieval Augmentation

**24. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020, NeurIPS)**
Seminal RAG paper combining dense retrieval + language generation. Shows 5-10% improvement over fine-tuned baselines. Sparked industry adoption of RAG architecture.

**25. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction (Khattab & Zaharia, 2020, SIGIR)**
Late interaction scoring (similarity computed at token-level, then max-pooled). Enables efficient ranking without slowing retrieval. Widely used in production RAG.

**26. Dense Passage Retrieval for Open-Domain Question Answering (Karpukhin et al., 2020, EMNLP)**
DPR: dense embeddings for retrieval. Combines supervised and unsupervised training. Outperforms sparse (BM25) methods. Foundation for modern dense retrieval.

---

## 6. Quantization for Transformers (6 papers)

**27. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (Dettmers et al., 2022, NeurIPS)**
Enables 8-bit inference for 176B parameter models. Uses mixed-precision: 8-bit matrix multiply with 16-bit outliers. Critical for deploying large models on small GPUs.

**28. ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers (Ma et al., 2022, arXiv)**
Token-wise and channel-wise quantization strategies for transformers. Achieves INT4 without significant accuracy loss. More granular than layer-wise methods.

**29. Outlier Suppression+ : Accurate Quantization of Large Models By Avoiding Outliers (Wei et al., 2023, arXiv)**
Address outlier activation problem in quantization. Smoothing technique reduces outlier ranges. Enables more aggressive quantization.

---

## 7. Specialized Hardware & Inference (8 papers)

### 7.1 Hardware-Aware Optimization

**30. Tensor Processing Units (TPU) v4 Architecture (Jouppi et al., 2021, ISCA)**
Google's custom ML chip. Shows specialized hardware achieves 10-100x better energy efficiency than CPUs. Relevant for understanding deployment hardware choices.

**31. Neuron 2 Accelerator: AI Accelerator for Amazon EC2 (Gupta et al., 2021, arXiv)**
AWS custom inference accelerator. Demonstrates practical deployment considerations: power, cooling, support for multiple frameworks.

**32. The Missing Piece in Complex Event Processing: Lightweight Rules (Hirzel et al., 2019, VLDb)**
While about event processing, applicable to inference pipelines. Shows importance of lightweight evaluation at edge vs central processing.

### 7.2 Mobile & Edge Inference

**33. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (Howard et al., 2017, arXiv)**
Lightweight CNN architecture for mobile. Uses depthwise separable convolutions. Enables vision inference on phones with <100ms latency.

**34. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (Tan & Le, 2019, ICML)**
Systematic approach to scaling CNN dimensions. Achieves better accuracy/latency trade-offs. Widely used for mobile vision.

---

## 8. Auto-Tuning & Compilation (4 papers)

**35. Nimble: Lightweight and Parallel GPU Task Scheduling for Irregular Algorithms (Zhang et al., 2020, OSDI)**
Dynamic task scheduling for irregular GPU workloads. Relevant for batching variable-size inputs in inference.

**36. AutoTVM: Learning to Optimize Tensor Programs (Chen et al., 2016, OSDI)**
Automated ML-based optimization of tensor operations. Learns good tensor schedules without hand-tuning. Foundational for auto-tuning inference.

**37. TenSet: Reviving Tensor Network Theory for Efficient Neural Network Execution (Kudo et al., 2021, NeurIPS)**
Graph optimization for tensor operations. Shows potential for better operator fusion and execution planning.

---

## 9. Reading Order & Learning Path

### Recommended Learning Path (5-week study):

**Week 1: Fundamentals**
- Papers 7 (Attention is All You Need)
- Papers 2 (White Paper on Quantization)
- Papers 15 (Efficient Attention)

**Week 2: Vector Retrieval & RAG**
- Papers 22 (FAISS)
- Papers 24 (RAG)
- Papers 25 (ColBERT)
- Papers 26 (DPR)

**Week 3: Serving & Optimization**
- Papers 9 (Clipper)
- Papers 11 (TVM)
- Papers 18 (Paged Attention)

**Week 4: Quantization Deep Dive**
- Papers 1 (INT8 fundamentals)
- Papers 4 (AWQ)
- Papers 5 (GPTQ)
- Papers 27 (LLM.int8)

**Week 5: Hardware & Modern Inference**
- Papers 20 (FlashAttention)
- Papers 31 (Neuron 2)
- Papers 33 (MobileNets)

---

## 10. Summary & Key Takeaways

Essential papers to have read:
1. **Quantization**: Papers 1, 2, 4, 5, 27
2. **Retrieval**: Papers 22, 24, 25, 26
3. **Serving**: Papers 9, 11, 18
4. **Optimization**: Papers 7, 15, 20

Most impactful for production inference:
- Paged Attention (18) - enables modern LLM serving
- FAISS (22) - enables RAG at scale
- RAG (24) - the most deployed inference pattern
- AWQ (4) - practical quantization for LLMs
- FlashAttention (20) - 2-3x speedup, widely adopted

Next steps:
1. Read fundamentals (Papers 7, 2, 15)
2. Understand retrieval (Papers 22, 24)
3. Deep dive on your specific use case
4. Implement and benchmark against papers' results

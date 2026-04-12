# MODULE 3: SPARSITY — Structured & Unstructured

## A Comprehensive Treatise on Sparse ML Systems

*For experienced ML systems engineers with CUDA, GPU, PyTorch, and computer architecture background.*

---

## Table of Contents

1. [Systems Overview](#1-systems-overview)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [Hardware Mapping](#3-hardware-mapping)
4. [Implementation Deep Dive](#4-implementation-deep-dive)
5. [Key Papers](#5-key-papers)
6. [Systems Tradeoffs](#6-systems-tradeoffs)
7. [Expert Insight](#7-expert-insight)
8. [Benchmarking Methodology](#8-benchmarking-methodology)
9. [Open Problems](#9-open-problems)
10. [PhD Qualifier Questions](#10-phd-qualifier-questions)

---

## 1. SYSTEMS OVERVIEW

### 1.1 The Sparsity Paradox

Sparsity in neural networks represents one of the most deceptive optimization opportunities in deep learning systems. Theoretically, the appeal is irresistible: if a neural network has 70B parameters and you can maintain accuracy while zeroing out 50% of weights (or activations), you should achieve 2x speedup. The mathematics is straightforward—fewer nonzero elements mean fewer floating-point operations.

However, the production reality is fundamentally different. After more than a decade of sparse neural network research, the overwhelming consensus from practitioners is that **most unstructured sparsity does not provide meaningful speedups on modern GPUs or CPUs without highly specialized hardware support**. This distinction between theoretical compression and practical speedup is the central theme of sparsity in systems.

### 1.2 The Sparsity Landscape

Modern sparsity manifests in three primary forms:

**Weight Sparsity**: Directly pruning parameters from the model. The weight matrix W ∈ ℝ^(m×n) becomes sparse with s% of entries equal to zero. This is the most studied form in literature but paradoxically provides the least practical benefit on commodity hardware.

**Activation Sparsity**: Exploiting the natural sparsity that emerges in hidden layers during inference. For ReLU networks, activations h ∈ ℝ^(batch_size × hidden_dim) contain many zeros. Modern transformers with softmax attention also exhibit high activation sparsity. This form is increasingly critical for efficient inference.

**Structured Sparsity**: Sparsity that respects hardware boundaries. NVIDIA's 2:4 structured sparsity (every 4 consecutive weights, at least 2 must be zero), head pruning in transformers (entire attention heads), neuron pruning in MLPs (entire neurons), and layer skipping represent structured forms that map cleanly to GPU/CPU primitives.

### 1.3 When Sparsity Actually Works

The empirical evidence from large-scale systems (NVIDIA inference engines, Google TPU clusters, Apple Neural Engine) reveals that sparsity provides real speedups in precisely these scenarios:

1. **NVIDIA 2:4 Structured Sparsity on Ampere/Hopper**: With proper kernel implementation, achieves consistent 1.3-1.8x speedup on LLM inference. This is production-deployed at scale.

2. **Activation Sparsity with Specialized Routing**: PowerInfer-style systems that separate hot (active) and cold (inactive) neurons, enabling selective computation, achieve 2-4x speedup depending on layer characteristics.

3. **Expert Selection in MoE**: When only k of n experts are activated per token, and expert parallelism is properly balanced, achieves superlinear scaling with expert count. This scales to thousands of experts.

4. **Head and Layer Pruning in Transformers**: Removing entire attention heads or transformer layers is structured—compiler-friendly and cache-efficient. Achieves 20-40% speedup with <2% accuracy loss on many tasks.

5. **Sparse Attention Patterns**: When self-attention is the bottleneck (long sequences), structured sparsity in the attention matrix (local windows, strided patterns) achieves 2-8x memory reduction and proportional speedup.

### 1.4 When Sparsity Fails

Conversely, the following approaches consistently underdeliver in production:

- **Random/Unstructured Magnitude Pruning**: Requires custom sparse GEMM kernels that don't exist on most hardware. Even with sparse formats (CSR, COO), achieving speedup requires >75% sparsity to overcome format overhead.

- **Fine-grained Activation Sparsity Without Prediction**: Computing and storing sparse masks for every forward pass adds overhead that exceeds computation savings unless >80% of activations are actually zero.

- **Learned Pruning Patterns**: The cost of storing and interpreting per-sample pruning patterns or indices often exceeds savings, especially in batched inference.

---

## 2. THEORETICAL FOUNDATION

### 2.1 Weight Sparsity and Magnitude Pruning

#### 2.1.1 Mathematical Framework

Let W ∈ ℝ^(d_in × d_out) be a weight matrix. Magnitude pruning sets the weight with smallest |w_ij| to zero in a manner that respects some sparsity constraint. The most common formulation:

$$\hat{W} = W \odot \mathbb{1}[|W| > \theta]$$

where θ is a threshold determined by target sparsity level s:

$$\theta = \text{kth-largest}(|W|) \text{ where } k = (1-s) \cdot |W|$$

The resulting sparse matrix has exactly sparsity level s. This approach has several variants:

**Unstructured Magnitude Pruning**: Each element independently subject to thresholding. Provides maximum flexibility but worst hardware utilization.

**Structured Magnitude Pruning**: Applies thresholding at the group level (channels, heads, or layers). For example, channel pruning sets all weights of entire output channels to zero if group magnitude falls below threshold.

#### 2.1.2 Lottery Ticket Hypothesis

The lottery ticket hypothesis (Frankle & Carbin, 2019) posits that dense neural networks contain subnetworks that, when trained in isolation, achieve comparable accuracy to the full network:

$$\text{Hypothesis: } \exists \text{ sparse } W_s \text{ with mask } m: |m| = s \cdot |W|$$

such that when W_s ⊙ m is trained in isolation from random initialization with the same learning rate schedule, it achieves accuracy comparable to W trained from scratch.

This framing reorients the question: rather than "which weights are unimportant?", ask "which weights form the learnable subnetwork?". The critical finding is that lottery tickets exist but are difficult to find without iterative pruning.

The implications for systems are significant:

- One-shot pruning (prune once, done) rarely finds good tickets
- Iterative pruning with retraining converges to better tickets but is computationally expensive
- The discovered tickets are often non-obvious and task-dependent
- Transferring lottery tickets across tasks has limited success (tickets are task-specific)

#### 2.1.3 Optimal Brain Damage (OBD)

Optimal Brain Damage (LeCun, Denker, & Solla, 1989) provides a principled framework for pruning via second-order analysis. The Hessian of the loss with respect to parameters tells us how sensitive loss is to parameter perturbations:

$$H_{ij} = \frac{\partial^2 L}{\partial w_i \partial w_j}$$

For a parameter w_i, the diagonal Hessian entry H_ii approximates the loss increase from removing that parameter:

$$\Delta L \approx \frac{1}{2} H_{ii} w_i^2$$

Parameters with low H_ii × w_i² are "unimportant"—their removal minimally affects loss. This provides principled pruning priority:

$$\text{priority}_i = H_{ii} \cdot w_i^2$$

The connection to magnitude pruning is instructive: if Hessian is uniform, OBD reduces to magnitude pruning. But when Hessian varies across the network, OBD can find much better pruning patterns.

**Computational cost**: Computing full Hessian is O(d²) memory and O(d³) computation, prohibitive for large models. Approximations using only diagonal or block-diagonal Hessian bring cost to O(d).

#### 2.1.4 Movement Pruning

Movement pruning (Victor Sanh et al., 2020) observes that during fine-tuning of pretrained models, weights move from their initialized values. Rather than looking at weight magnitudes, movement pruning considers:

$$m_i = |\Delta w_i| = |w_i^{\text{fine-tuned}} - w_i^{\text{pretrained}}|$$

The hypothesis: if a weight barely moves during task-specific fine-tuning, it was already well-suited to the task by pretraining—keep it. If it moves significantly, it's adapting to the task—keep it. The weights in the middle (small magnitude but small movement, or large magnitude but small movement) are candidates for pruning.

This leads to a multivariate pruning criterion combining magnitude and movement. Movement pruning is particularly effective for transfer learning scenarios and has become standard in practice for fine-tuned models.

### 2.2 NVIDIA 2:4 Structured Sparsity

#### 2.2.1 The 2:4 Pattern

NVIDIA's 2:4 structured sparsity, introduced in the Ampere architecture and continuing through Hopper, enforces a specific constraint on weight matrices: **every 4 consecutive weights in a row must have at least 2 zeros**.

Formally, partition each row of weight matrix W into consecutive groups of 4:

$$W[i, 4k:4k+4] \text{ for } k = 0, 1, 2, ...$$

For each group, exactly 2 (or at least 2, in some formulations) of the 4 values must be zero. This ensures:

- **Deterministic Structure**: The zero pattern is known at compile time, enabling dense kernel generation.
- **Minimal Index Overhead**: Only a 2-bit mask per 4 elements (0.25 bits per weight) encodes where nonzero values are.
- **Hardware Support**: Sparse tensor cores in Ampere (SM 8.0) and Hopper (SM 9.0) natively support this pattern.

The pattern can be visualized for a single row:

```
Weight row: [w0  w1  w2  w3 | w4  w5  w6  w7 | ...]
Pattern:    [0   w1  0   w3 | w4  0   w6  0  | ...]  (one valid 2:4 configuration)
Mask:       [1   0   1   0 | 0   1   0   1  | ...]  (where 1 indicates zero, 0 indicates nonzero)
```

#### 2.2.2 Hardware Mechanism in Ampere/Hopper

NVIDIA's sparse tensor cores process structured sparsity as follows:

**Memory Layout**: Weights are stored in a specialized compressed format:
- Values: densely packed nonzero values (2 values per 4 in original representation)
- Metadata: 2-bit mask indicating which of the 4 positions are nonzero

For a weight matrix W ∈ ℝ^(m × n) with 2:4 sparsity:
- Storage: (m × n × 0.5) + (m × n × 0.0625) bits ≈ 56.25% of original storage
- Each row is divided into 4-element groups, each group has exactly 2 nonzero values

**Execution Model**: When a sparse tensor core processes A ∈ ℝ^(m × k) (dense) × W ∈ ℝ^(k × n) (2:4 sparse):

1. Load a warp of activations A[:, j:j+32] (or appropriate block)
2. For each 4-element group in W rows, load the 2 nonzero values and metadata
3. Execute specialized GEMM instruction that:
   - Multiplies dense A with the 2 nonzero weights
   - Accumulates results according to metadata positions
   - Processes multiple groups per warp instruction

The speedup factor depends on:
- **Compute Density**: 2:4 sparsity means theoretical 2x compute, but actual speedup ~1.5x due to control logic overhead
- **Memory Bandwidth**: With proper format, 1.5-2x memory bandwidth reduction
- **Occupancy**: Better registers/shared memory usage due to sparser computation

#### 2.2.3 Achieving 2:4 via SparseGPT

The challenge: converting a dense, trained weight matrix to 2:4 sparsity while preserving accuracy. Simple magnitude pruning fails—the 2:4 constraint is stricter than general pruning.

**SparseGPT Algorithm** (see Implementation section) solves this via:

1. **Hessian-Aware Pruning**: Use diagonal Hessian H to determine importance
2. **Layerwise Processing**: Process one layer at a time, computing layer Hessian via
   $$H = 2 \cdot X^T X$$
   where X is the layer's activation matrix
3. **Greedy Pruning with Compensation**: For each parameter to prune:
   - Remove parameter w_ij
   - Compute error: Δ L = (w_ij² × H_ij) / 2
   - Compensate by adjusting remaining weights in the layer to absorb this error
4. **2:4 Pattern Enforcement**: During pruning, respect the 2:4 constraint

The key insight: compensation distributes error across remaining weights, allowing recovery of accuracy without retraining.

#### 2.2.4 Practical 2:4 Considerations

**Quantization Compatibility**: 2:4 sparsity composes well with quantization. The sparsity pattern is determined by INT8/FP8 structured pruning, then quantized values are stored in sparse format. This achieves both sparsity and quantization benefits.

**Per-Channel vs Per-Token Sparsity**: The 2:4 pattern is fixed per weight matrix (per-channel). Per-token dynamic sparsity (varying which weights are active per input) is incompatible with NVIDIA sparse tensor cores—it requires dense GEMM or special hardware.

**Attention Sparsity Incompatibility**: Attention matrices are naturally sparse (softmax produces sparse attention weights), but the dynamic nature means 2:4 pattern cannot be precomputed. Sparse attention requires different approaches.

### 2.3 Structured Pruning for CPU and Apple Silicon

#### 2.3.1 Head Pruning in Transformers

Modern transformer models decompose multi-head self-attention into h independent attention heads:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

where each head computes:

$$\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

Head pruning removes entire heads. A head is low-importance if:
- Its attention weights are nearly uniform (not focusing on specific positions)
- Its output contributes minimally to downstream layers
- Its presence is redundant with other heads

**Importance Metrics**:

1. **Attention Entropy**: High entropy (uniform distribution) indicates low information content
   $$H_i = -\sum_{j} \alpha_{ij} \log \alpha_{ij}$$
   Heads with entropy > threshold are candidates for pruning.

2. **Taylor Expansion Importance**: How much loss increases when head output is zeroed
   $$I_i = \left| \frac{\partial L}{\partial h_i} \cdot h_i \right|$$
   Heads with low importance are pruned.

**Hardware Benefits**:
- Removing a head removes ~1/h of computation in that layer (8-12% per head for 8-12 head models)
- Modern CPUs and Apple Neural Engine have no sparse tensor core equivalent, but entire-head removal is compiler-friendly
- Removes h/total_heads attention patterns, reducing memory bandwidth proportionally

**Practical Results**: Removing 20-30% of heads typically causes <1% accuracy loss. Removing 50% causes 2-5% loss on most tasks.

#### 2.3.2 FFN Neuron Pruning

In transformer models, the feed-forward network (FFN) layer contains the majority of parameters:

$$\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2$$

where W_1 ∈ ℝ^(d_model × d_ff) with d_ff typically 4× d_model (e.g., 4096 for 1024-dim models).

FFN neuron pruning removes entire neurons (columns of W_1, corresponding rows of W_2):

$$\hat{W}_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}'}$$

where d_ff' < d_ff and columns corresponding to unimportant neurons are removed.

**Importance Metrics**:

1. **Activation-Based**: Average absolute activation magnitude across training data
   $$I_j = \mathbb{E}_{x \sim D}[|\text{ReLU}(x W_1 + b_1)_j|]$$
   Neurons with low activation are less influential.

2. **Gradient-Based**: Magnitude of gradient with respect to neuron output
   $$I_j = \mathbb{E}_{x \sim D}\left[\left|\frac{\partial L}{\partial \text{hidden}_j}\right|\right]$$

3. **Hessian-Based**: Diagonal Hessian entries (as in OBD)

**Computational Efficiency**: Removing m neurons from d_ff total means removing:
- m × d_model parameters from W_1
- m × d_model parameters from W_2
- Proportional reduction in FFN computation (both inference and training)

For a 70B model with 32 layers, each with 27K neurons, pruning 30% of neurons removes ~18B parameters (~25% total). FFN computation reduces proportionally.

**Hardware Mapping**: CPUs and Apple Neural Engine benefit significantly because:
- No sparse tensor cores, but dense GEMM with smaller matrices is much faster
- Cache efficiency improves (smaller weights fit in L1/L2)
- Vector units operate on smaller matrices with better utilization

#### 2.3.3 Layer Pruning

The most structured form: remove entire transformer layers. A 70B model with 80 layers can have some layers be nearly redundant. Layer pruning removes them:

$$\text{Model}_{\text{pruned}} = \text{Embed} \to \text{Layer}_1 \to ... \to \text{Layer}_{80} \to \text{Head}$$

becomes:

$$\text{Model}_{\text{pruned}} = \text{Embed} \to \text{Layer}_1 \to ... \to \text{Layer}_{\text{selected}} \to \text{Head}$$

Selecting which layers to keep:

1. **Layer Similarity**: If layer_i output is very similar to layer_i-1, the layer adds little
   $$\text{sim}(x_i, x_{i-1}) = \frac{\mathbb{E}[||x_i - x_{i-1}||^2]}{||x_i||^2}$$
   Small similarity indicates redundancy.

2. **Attention Head Consensus**: Layers where attention heads produce very similar patterns contribute redundantly

3. **Empirical Removal**: Iteratively remove layers from bottom and measure accuracy loss

**Impact**: Removing k of n layers means k/n speedup (e.g., removing 20 of 80 layers = 25% speedup). Accuracy loss depends heavily on task and layer position (typically, later layers are more task-specific and harder to remove).

### 2.4 Activation Sparsity

#### 2.4.1 ReLU Sparsity

ReLU networks exhibit natural activation sparsity: approximately 50% of activations are zero at initialization, increasing to 70-90% in deeper layers after training on typical tasks.

**Sparsity Pattern**: After a ReLU layer, h = ReLU(z) where z = Wx + b, the sparsity level is:

$$s = \frac{|\{i : h_i = 0\}|}{d}$$

This sparsity is:
- **Dynamic**: Varies per input and layer
- **Data-Dependent**: Different inputs produce different sparsity patterns
- **Non-Trivial to Exploit**: Requires sparse matrix operations or special hardware

#### 2.4.2 Transformer Activation Sparsity

Transformers exhibit sparsity in multiple ways:

**Softmax Sparsity**: After softmax in attention, α_ij = softmax(Q_i K_j^T / √d_k), only top-k entries are meaningful (others are near-zero). Directly setting sub-threshold entries to zero changes the computation but approximates the original.

**MLP Output Sparsity**: Similar to ReLU sparsity, transformer MLP outputs (GELU(W_1 x) W_2) are naturally sparse. GELU acts as a probabilistic gate—roughly 30-50% of neuron outputs are near-zero.

**Attention Pattern Sparsity**: Some positions receive near-zero attention weight across all heads, indicating the model attends minimally to that position. These positions can be skipped.

#### 2.4.3 Exploiting Activation Sparsity: Deja Vu

The **Deja Vu** paper (Liu et al., NeurIPS 2023) exploits the observation that activation sparsity patterns are predictable across nearby inputs in a sequence:

**Hypothesis**: For a transformer processing a sequence, the activation sparsity pattern at layer l for token i is similar to the pattern at layer l for nearby tokens (i-1, i+1, etc.).

**Mechanism**:
1. For the first token in a batch, compute all layers normally
2. For subsequent tokens, use the sparsity pattern from token i-1:
   - Determine which neurons are "active" (produce nonzero outputs) based on previous token
   - Only compute those neurons for current token
   - Use cached computation for others

**Prediction Accuracy**: Across transformer layers, sparsity pattern agreement between consecutive tokens is 75-95%, enabling ~10-30% computation reduction with minimal accuracy loss.

**Hardware Considerations**:
- Requires dynamic masking during computation
- Benefits from GPU with efficient masked computation primitives
- CPU implementation possible but requires careful cache management

#### 2.4.4 PowerInfer: Activation-Aware Inference

**PowerInfer** (Song et al., SOSP 2024) takes activation sparsity further through hot/cold neuron separation:

**Key Insight**: In large models, some neurons consistently activate (hot neurons) across different inputs, while others are input-dependent (cold neurons). By analyzing many inputs during profiling:

1. **Profiling Phase**: Run model on diverse inputs, track neuron activation frequency
2. **Classification**: Neurons above threshold (e.g., >80% activation frequency) are "hot"; others are "cold"
3. **Execution Strategy**:
   - Hot neurons: Keep in GPU VRAM or CPU cache, compute always
   - Cold neurons: Store in CPU RAM, compute selectively based on input

The separation enables:
- Hot neurons (GPU): Fast dense computation
- Cold neurons (CPU): Sparse computation only when activated
- Heterogeneous execution: Overlap cold CPU computation with hot GPU computation

**Speedup Mechanism**: For a 13B model, profiling reveals ~30-40% of neurons are hot. Computing full model on GPU takes time T. PowerInfer:
- GPU computation on hot neurons: ~0.4T (40% of computation)
- CPU computation on cold neurons: ~0.3T (30% computation load, but CPU is slower, so takes longer)
- Overlap via pipelining: Total ~0.5T, achieving 2x speedup

**Practical Requirements**:
- Significant profiling overhead (requires representative input distribution)
- GPU-CPU PCIe bandwidth must be sufficient (requires >50 GB/s)
- Temperature and power constraints must be managed

---

## 3. HARDWARE MAPPING

### 3.1 Why Unstructured Sparsity Fails

Unstructured sparsity—random zero patterns in weight matrices—is the most researched form in the literature but provides minimal practical speedup on modern hardware. Understanding why requires examining hardware primitives.

#### 3.1.1 Memory Layout and Access Patterns

Modern CPUs and GPUs are optimized for **dense, regular memory access**. A dense weight matrix W ∈ ℝ^(m × n) stored in row-major format presents:

- **Spatial Locality**: Consecutive memory locations are accessed sequentially
- **Regular Strides**: SIMD units load aligned chunks (256-bit AVX-512, 128-bit NEON)
- **Cache Efficiency**: Prefetchers anticipate sequential access

Unstructured sparse matrices break these assumptions:

```
Dense GEMM:     Load consecutive weights → predictable memory pattern
                Cache-line hit rate: ~90%
                SIMD utilization: >80%

Unstructured    Load index → compute address → load weight
Sparse GEMM:    Scattered memory access → unpredictable pattern
                Cache-line hit rate: ~20-40%
                SIMD utilization: <20% (many warp lanes do nothing)
```

The overhead of computing sparse indices and scattered loads dominates any computation savings until sparsity is extreme (>75%).

#### 3.1.2 Sparse Matrix Formats and Their Costs

Common sparse formats (CSR, COO, ELL) trade off compression and computational efficiency:

**Compressed Sparse Row (CSR)**:
- Storage: Three arrays (values, column_indices, row_pointers)
- Example: For 1M×1M matrix with 10M nonzeros (0.001% sparsity):
  - Values: 10M × 4B = 40 MB
  - Column indices: 10M × 4B = 40 MB
  - Row pointers: 1M × 4B = 4 MB
  - **Total overhead**: 84 MB for 40 MB of data (2.1x)

- Access pattern: For each row, load row_pointers[i] to row_pointers[i+1] to determine nonzero range
- GEMM cost: O(nnz) multiplication + O(nnz) index computation + O(nnz × log(m)) cache misses

**Coordinate Format (COO)**:
- Storage: Two arrays (row_indices, column_indices) + values
- Better for iteration but requires sorting by row, additional overhead

**Ellpack (ELL)**:
- Fixed K nonzeros per row, padded if necessary
- Good if sparsity pattern is uniform, terrible if skewed

**Hardware Reality**:
- None of these formats map efficiently to GPU warp operations or CPU SIMD
- Custom sparse kernels required for each format-hardware combination
- Sparse libraries (cuSPARSE, MKL) exist but have high constant factors

#### 3.1.3 Sparse GEMM Bottleneck

Sparse matrix multiplication C = A × B (both sparse) is fundamentally hard:

1. **Load Imbalance**: Different threads encounter different nnz density, some threads idle
2. **Work Distribution**: No simple block decomposition when sparsity pattern is arbitrary
3. **Synchronization**: Aggregating results from scattered computations requires atomics (expensive)

Benchmarks on modern hardware (V100, A100, RTX 3090):

```
Matrix: 8192 × 8192, various sparsity levels
Sparsity    cuSPARSE  Dense GEMM  Speedup
10%         13 TFLOPs  71 TFLOPs   0.18x (sparse is slower!)
50%         18 TFLOPs  71 TFLOPs   0.25x
75%         25 TFLOPs  71 TFLOPs   0.35x
90%         45 TFLOPs  71 TFLOPs   0.63x
95%         60 TFLOPs  71 TFLOPs   0.84x (finally breaks even)
99%         68 TFLOPs  71 TFLOPs   0.96x (nearly 1x)
```

**Breakeven Sparsity**: For typical hardware, achieving 1x speedup requires >90% sparsity. For 2x speedup, require >95% sparsity. These thresholds are model-dependent (wider models benefit slightly more).

### 3.2 Structured Sparsity on Modern Hardware

#### 3.2.1 NVIDIA Sparse Tensor Cores (Ampere/Hopper)

NVIDIA's 2:4 structured sparsity overcomes the unstructured sparsity problem through hardware support:

**Ampere Architecture (RTX 30 series, A100)**:
- Sparse tensor cores introduced alongside dense tensor cores
- Each SM has 2 sparse tensor cores (vs. many dense cores)
- Instruction: MMATRIX_SPARSE (mixed precision)
- Throughput: 1312 TFLOP/s (sparse) vs. 1312 TFLOP/s (dense) per SM
- The 1:1 ratio means 2:4 sparsity achieves ~1.5x speedup (2x theoretical, minus metadata overhead)

**Hopper Architecture (H100)**:
- Sparse tensor cores more tightly integrated
- Better load balancing and metadata handling
- Improved from Ampere: can achieve closer to 1.8x speedup in practice

**Kernel Implementation**:

```cuda
// Simplified sparse GEMM with 2:4 sparsity
// Input: A (dense), W (2:4 sparse in specialized format)
// Output: C
__global__ void sparse_gemm_2_4(
    half *A, half *W, half *C,
    uint8_t *metadata,  // 2-bit masks per 4-element group
    int M, int N, int K) {

    // Warp processes 8 outputs (M dimension) × 128 (N dimension)
    int warp_row = blockIdx.x * 32 + threadIdx.x / 4;  // 32 warps in block
    int warp_col = blockIdx.y * 128;

    half acc[8] = {0};  // 8 accumulators per thread

    // Process K dimension in groups of 8
    for (int k = 0; k < K; k += 8) {
        // Load dense A activation (8 values per thread)
        half a_vals[8] = load_A_activation(...);

        // Load sparse W (2 values per 4-group due to 2:4 pattern)
        half w_vals[2] = load_W_sparse(...);
        uint8_t mask = metadata[...];  // Which 2 of 4 positions are nonzero

        // Execute multiplication with mask-guided accumulation
        // Hardware sparse tensor core does this efficiently
        sparse_mma(acc, a_vals, w_vals, mask);
    }

    store_C_output(C, acc, ...);
}
```

The hardware handles the complexity of masked accumulation efficiently.

#### 3.2.2 CPU Efficiency from Structured Sparsity

For CPUs (x86, ARM, Apple Silicon), structured sparsity benefits through compiler optimization:

**Head Pruning Impact**:
- Attention for h heads: compute attention for all h, concat, multiply by W^O
- With head pruning (h' < h heads):
  - Compute h' softmax operations (h'/h speedup)
  - Reduce memory bandwidth by h'/h
  - Cache efficiency improves: h' smaller attention matrices fit in L2

**Neuron Pruning Impact**:
- FFN: [batch, d] × [d, 4d] × [4d, d] matrix chain
- Pruning 4d neurons to 4d' means:
  - First GEMM: same cost (full input dimension)
  - Second GEMM: 4d' × d instead of 4d × d (25% reduction for 25% neuron pruning)

**Memory Bandwidth Advantage**:
- Dense GEMM is bandwidth-limited (arithmetic intensity ~2-4 ops/byte)
- Smaller matrices from pruning fit in caches
- L1 cache hit rate improves 30-50% with aggressive pruning

#### 3.2.3 Breakeven Sparsity for Sparse vs Dense

The critical question for practitioners: at what sparsity level is structured pruning worth implementing?

For CPUs/Apple Silicon:

```
Operation: FFN neuron pruning in transformer inference
Model: 7B parameters, 32 layers, 4096→10240→4096 FFN

Pruning %   FFN Size    Compute Time    Memory BW    Overall Speedup
0%          4096        1.0x            1.0x         1.0x (baseline)
10%         3686        0.9x            0.92x        0.93x
20%         3277        0.8x            0.85x        0.87x
30%         2867        0.7x            0.79x        0.81x
40%         2457        0.6x            0.73x        0.72x
50%         2048        0.5x            0.67x        0.63x
```

The relationship is sublinear due to bandwidth reduction benefits. At 30% pruning, achieve ~20% speedup. At 50% pruning, achieve ~37% speedup (much better than naive 50%).

For GPUs with 2:4 sparse tensor cores:

```
Sparsity    Compute    Memory BW    Metadata    Overall
0% (baseline) 1.0x    1.0x         0x          1.0x
2:4 (50%)     1.5x    1.5x         0.04x       1.48x
```

The metadata cost is negligible, and both compute and bandwidth improve, yielding true ~1.5x speedup.

### 3.3 DeepSparse: CPU Sparse Inference

DeepSparse is a specialized inference engine optimized for sparse models on CPUs. It demonstrates the effectiveness of structured sparsity optimizations:

**Architecture**:
1. **Sparse-first design**: All operations assume structured sparsity
2. **Custom kernels**: Specialized for common sparsity patterns (head pruning, neuron pruning)
3. **Graph optimization**: Prunes entire layers, fuses operations

**Performance**:
- Unstructured sparsity: 0.8-1.2x speedup (negligible)
- Structured sparsity (head/neuron pruning): 2-4x speedup

DeepSparse validates the principle: structure the sparsity to match hardware, and significant speedups are achievable on CPUs.

---

## 4. IMPLEMENTATION DEEP DIVE

### 4.1 SparseGPT Algorithm

SparseGPT (Frantar & Alistarh, ICML 2023) is the definitive algorithm for post-training pruning of large language models. It combines magnitude pruning with Hessian-based error compensation to achieve high sparsity with minimal accuracy loss.

#### 4.1.1 Mathematical Formulation

Goal: Prune weight matrix W ∈ ℝ^(m × n) to sparsity target s while minimizing output error.

**Naive Approach**: For each parameter w_ij, compute importance score and remove bottom s × m × n parameters. Problem: pruning affects downstream computations; naive approach ignores interaction.

**SparseGPT Approach**: Use Hessian information to prune and compensate iteratively.

**Hessian Computation**:

For a linear layer output y = Wx where W ∈ ℝ^(m × n), the Hessian of MSE loss between original and pruned outputs is:

$$H = \nabla^2 L = 2X^T X$$

where X ∈ ℝ^(b × n) is the batch of activation inputs to this layer. For large language models:

- Compute Hessian H ∈ ℝ^(n × n) by passing a representative dataset through the layer
- H is symmetric positive semi-definite
- Diagonal H_ii tells importance of each weight: higher diagonal = more important

**Pruning with Compensation**:

For each weight w_ij to be pruned (set to zero):

1. **Error Quantification**: Setting w_ij = 0 causes output change:
   $$\Delta y \approx x_j \cdot e_i$$

   where e_i is the i-th standard basis vector. The squared error is:
   $$\Delta L \approx w_{ij}^2 H_{jj}$$

2. **Compensation**: Rather than absorb this error, distribute it among remaining weights in the same row:
   $$w_{i,\text{remaining}} \leftarrow w_{i,\text{remaining}} + \frac{w_{ij}}{w_{i,\text{remaining}}^T H^{-1} w_{i,\text{remaining}}} H^{-1} w_{i,\text{remaining}}$$

   This compensates by updating remaining weights to minimize output change.

3. **Update Hessian**: After pruning a weight, update Hessian to account for the compensation
   $$H \leftarrow H - \frac{H_{:,j} H_{j,:}}{H_{jj}}$$

#### 4.1.2 Pseudocode Implementation

```python
import torch
import numpy as np

class SparseGPT:
    def __init__(self, layer_module, target_sparsity=0.5):
        """
        Initialize SparseGPT for a single layer.

        Args:
            layer_module: nn.Linear or equivalent layer with weight matrix
            target_sparsity: fraction of weights to set to zero (0.0 to 1.0)
        """
        self.layer = layer_module
        self.target_sparsity = target_sparsity
        self.W = layer_module.weight.data.clone()
        self.m, self.n = self.W.shape
        self.H = None  # Hessian matrix
        self.nprune = int(target_sparsity * self.m * self.n)

    def compute_hessian(self, X):
        """
        Compute Hessian H = 2 * X^T X for layer inputs X.

        Args:
            X: Input activation tensor of shape [batch, in_features]

        Returns:
            H: Hessian matrix [in_features, in_features]
        """
        # X shape: [batch, n]
        # H = X^T X gives Hessian of loss w.r.t. parameters
        X = X.float()
        H = X.t() @ X
        H /= float(X.shape[0])  # Normalize by batch size
        H += 1e-6 * torch.eye(H.shape[0], device=H.device)  # Regularization
        return H

    def prune(self, X):
        """
        Prune weights using SparseGPT algorithm.

        Args:
            X: Activation inputs to this layer [batch, in_features]
        """
        # Step 1: Compute Hessian
        self.H = self.compute_hessian(X)
        print(f"[SparseGPT] Hessian computed: {self.H.shape}")

        # Step 2: Compute initial importance scores
        # Importance of weight w_ij is approximately w_ij^2 * H_jj
        scores = self.W ** 2 * torch.diag(self.H).unsqueeze(0)

        # Step 3: Prune weights iteratively
        pruned_count = 0
        for step in range(self.nprune):
            # Find weight with minimum importance
            idx_row, idx_col = torch.where(self.W != 0)  # Non-zero weights
            scores_nz = scores[idx_row, idx_col]
            min_idx = torch.argmin(scores_nz)
            row, col = idx_row[min_idx], idx_col[min_idx]

            # Prune this weight
            w_val = self.W[row, col].item()
            self.W[row, col] = 0

            # Compensate: adjust remaining weights in the row
            # Update only non-zero weights in same row
            mask = self.W[row] != 0
            if mask.sum() > 0:
                # Hessian inverse term (approximation)
                H_inv_col = torch.inverse(self.H)[col]
                h_inv_sum = (H_inv_col ** 2).sum()

                if h_inv_sum > 1e-8:
                    # Distribute error among remaining weights
                    update = (w_val / h_inv_sum) * H_inv_col
                    self.W[row, mask] += update[mask] * 0.1  # Conservative update

            # Update Hessian (fast rank-1 update)
            H_col = self.H[:, col]
            H_diag = self.H[col, col]
            if H_diag.abs() > 1e-8:
                self.H = self.H - (H_col.unsqueeze(1) @ H_col.unsqueeze(0)) / H_diag

            pruned_count += 1
            if (pruned_count + 1) % max(1, self.nprune // 10) == 0:
                print(f"[SparseGPT] Pruned {pruned_count}/{self.nprune} weights")

        # Step 4: Update layer with pruned weights
        self.layer.weight.data.copy_(self.W)
        sparsity_achieved = (self.W == 0).float().mean().item()
        print(f"[SparseGPT] Pruning complete. Final sparsity: {sparsity_achieved:.4f}")

        return self.W

    def enforce_2_4_sparsity(self):
        """
        Convert pruned weights to 2:4 structured sparsity pattern.
        Groups 4 consecutive weights per row; ensures exactly 2 zeros per group.
        """
        W = self.W.clone()
        for row in range(self.m):
            for group_start in range(0, self.n, 4):
                group_end = min(group_start + 4, self.n)
                group = W[row, group_start:group_end]

                # Find k smallest magnitude weights in group
                k_zero = 2  # 2:4 pattern
                if group.numel() == 4:
                    _, indices = torch.topk(group.abs(), k=2, largest=False)
                    group[indices] = 0
                    W[row, group_start:group_end] = group

        self.W = W
        self.layer.weight.data.copy_(W)
        return W

# Usage example
def prune_model_sparsegpt(model, calib_data, sparsity=0.5):
    """
    Apply SparseGPT to all linear layers in a model.

    Args:
        model: Neural network model
        calib_data: Calibration data for Hessian computation
        sparsity: Target sparsity level
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Run calibration data through this layer to get activations
                X_calib = []
                for batch in calib_data:
                    # Forward pass to this layer
                    x = model(batch, return_layer_input=name)
                    X_calib.append(x)
                X_calib = torch.cat(X_calib, dim=0)

                # Apply SparseGPT
                sg = SparseGPT(module, target_sparsity=sparsity)
                sg.prune(X_calib)

                print(f"Pruned {name}: sparsity achieved")
```

#### 4.1.3 Practical Considerations

**Computational Cost**: For a layer with d_in features:
- Hessian computation: O(d_in²) time (X^T X matrix multiplication)
- Hessian inversion/update: O(d_in²) memory and O(d_in) time per pruning step
- Total: O(d_in² + n_prune × d_in) ≈ O(d_in² + m × n) for m × n weight matrix

For a 70B model (token embedding 4096-dim), this is tractable.

**Calibration Data**: Requires a few thousand examples of typical inputs to compute accurate Hessian. Using random data or bad calibration data degrades performance significantly.

**Hyperparameter Tuning**:
- **Sparsity target**: Usually 0.3-0.6 for language models without accuracy loss
- **Compensation strength**: Factor by which to apply Hessian-based updates (0.01-1.0)
- **Hessian damping**: Regularization term (1e-6 to 1e-4)

### 4.2 PowerInfer: Activation-Aware Inference

PowerInfer (Song et al., SOSP 2024) exploits activation sparsity through GPU-CPU heterogeneous execution. Unlike SparseGPT which modifies the model, PowerInfer works with any model.

#### 4.2.1 Profiling Phase

First, identify hot and cold neurons through profiling:

```python
class PowerInferProfiler:
    def __init__(self, model, profile_threshold=0.8):
        """
        Profile a model to identify hot/cold neurons.

        Args:
            model: Transformer model
            profile_threshold: Neurons active in >threshold fraction of samples are hot
        """
        self.model = model
        self.threshold = profile_threshold
        self.neuron_stats = {}  # Track activation stats per neuron

    def profile(self, dataloader, num_samples=1000):
        """
        Run model on diverse inputs and track neuron activations.

        Args:
            dataloader: DataLoader with representative inputs
            num_samples: Number of samples to profile
        """
        neuron_activation_counts = {}
        total_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx * batch['input_ids'].shape[0] >= num_samples:
                    break

                input_ids = batch['input_ids'].cuda()

                # Forward pass with activation hooking
                activations = {}

                def create_hook(layer_name):
                    def hook(module, input, output):
                        # output shape: [batch, seq_len, hidden_dim]
                        if len(output.shape) == 3:
                            # For each neuron, check if it's activated (nonzero)
                            active = (output > 0).float()  # [batch, seq_len, hidden_dim]
                            if layer_name not in neuron_activation_counts:
                                neuron_activation_counts[layer_name] = torch.zeros(output.shape[-1])
                            # Aggregate across batch and sequence dimensions
                            neuron_activation_counts[layer_name] += active.sum(dim=[0, 1]).cpu()
                    return hook

                # Register hooks on all MLP layers
                handles = []
                for name, module in self.model.named_modules():
                    if 'mlp' in name.lower() or 'intermediate' in name.lower():
                        h = module.register_forward_hook(create_hook(name))
                        handles.append(h)

                self.model(input_ids)
                total_samples += input_ids.shape[0] * input_ids.shape[1]

                # Clean up hooks
                for h in handles:
                    h.remove()

                if (batch_idx + 1) % 10 == 0:
                    print(f"Profiled {total_samples} tokens")

        # Compute neuron hotness scores
        for layer_name in neuron_activation_counts:
            counts = neuron_activation_counts[layer_name]
            activation_fraction = counts / total_samples

            hot_mask = activation_fraction > self.threshold
            cold_mask = ~hot_mask

            self.neuron_stats[layer_name] = {
                'hot_neurons': hot_mask,
                'cold_neurons': cold_mask,
                'activation_fraction': activation_fraction,
                'num_hot': hot_mask.sum().item(),
                'num_cold': cold_mask.sum().item(),
            }

            print(f"Layer {layer_name}: {self.neuron_stats[layer_name]['num_hot']} hot neurons, "
                  f"{self.neuron_stats[layer_name]['num_cold']} cold neurons")

        return self.neuron_stats

class PowerInferExecutor:
    def __init__(self, model, neuron_stats, gpu_device='cuda:0', cpu_threads=16):
        """
        Execute model with GPU-CPU heterogeneous computation.

        Args:
            model: Model with identified hot/cold neurons
            neuron_stats: Output from profiler
            gpu_device: GPU device for hot neuron computation
            cpu_threads: Number of CPU threads for cold computation
        """
        self.model = model
        self.neuron_stats = neuron_stats
        self.gpu_device = gpu_device
        self.cpu_threads = cpu_threads

    def forward(self, input_ids, **kwargs):
        """
        Execute forward pass with heterogeneous computation.

        Hot neurons: GPU
        Cold neurons: CPU (or skip if not activated)
        """
        # This is a simplified illustration; real implementation requires
        # carefully managing memory transfers and computation scheduling

        hidden_states = self.model.embedding(input_ids).to(self.gpu_device)

        for layer_idx, layer in enumerate(self.model.layers):
            # Attention part: execute on GPU (relatively small)
            attn_output = layer.attention(hidden_states)
            hidden_states = attn_output + hidden_states

            # MLP part: heterogeneous execution
            mlp_input = layer.norm(hidden_states)

            # Get hot/cold mask for this layer
            layer_name = f"layer.{layer_idx}.mlp"
            if layer_name in self.neuron_stats:
                hot_mask = self.neuron_stats[layer_name]['hot_neurons'].to(self.gpu_device)

                # GPU computation: all neurons (for simplicity)
                # In optimized version, only compute hot neurons on GPU
                mlp_hidden = layer.mlp.w1(mlp_input)  # GPU computation
                mlp_hidden = torch.nn.functional.gelu(mlp_hidden)
                mlp_output = layer.mlp.w2(mlp_hidden)
            else:
                mlp_output = layer.mlp(mlp_input)

            hidden_states = mlp_output + hidden_states

        return hidden_states
```

#### 4.2.2 Execution Strategy

The execution model in PowerInfer:

1. **Hot neurons**: Computed on GPU every forward pass
2. **Cold neurons**: Computed on CPU only if predicted to activate
3. **Prediction**: Use simple predictor (e.g., feed-forward network) to predict which cold neurons activate

The overall time is:

$$T_{\text{total}} = T_{\text{GPU-hot}} + T_{\text{CPU-cold}} + T_{\text{PCIe-transfer}}$$

where:
- T_GPU-hot: Time for hot neurons on GPU (~0.4T baseline)
- T_CPU-cold: Time for cold neurons on CPU (~0.3T baseline, but slower CPU)
- T_PCIe-transfer: Time to transfer activations between GPU/CPU

With careful implementation and overlapping, achieves 2-4x overall speedup.

### 4.3 Wanda Pruning

Wanda (Sun et al., ICML 2024) simplifies magnitude pruning by combining weight magnitudes with activation norms, achieving high-quality one-shot pruning without retraining.

#### 4.3.1 Algorithm

**Observation**: Weight importance depends not just on magnitude but on how much it's "exercised" by activations.

**Wanda Importance Score**:

$$I_{ij} = |W_{ij}| \cdot ||A_j||$$

where:
- W_ij: weight matrix element
- A_j: j-th column of activation matrix (activations seen by this weight)

**Intuition**:
- Small weights that multiply large activations can still be important
- Large weights that multiply inactive neurons are less important

**Algorithm**:

```python
def wanda_prune(W, A, sparsity_target=0.5):
    """
    WANDA: pruning weights based on magnitude and activation norms.

    Args:
        W: Weight matrix [out_features, in_features]
        A: Activation matrix [batch_size, in_features]
        sparsity_target: Target sparsity level

    Returns:
        Pruned weight matrix
    """
    # Step 1: Compute activation norms (importance of each input feature)
    activation_norms = torch.norm(A, p=2, dim=0)  # [in_features]
    activation_norms = activation_norms.unsqueeze(0)  # [1, in_features]

    # Step 2: Compute importance score
    importance = torch.abs(W) * activation_norms  # [out_features, in_features]

    # Step 3: Determine threshold for target sparsity
    num_params = W.shape[0] * W.shape[1]
    num_to_prune = int(num_params * sparsity_target)
    threshold = torch.kthvalue(importance.flatten(), num_to_prune).values

    # Step 4: Prune
    mask = importance > threshold
    W_pruned = W * mask.float()

    return W_pruned

# Usage
def prune_transformer_with_wanda(model, calib_loader, sparsity=0.5):
    """
    Apply WANDA pruning to transformer model layers.
    """
    with torch.no_grad():
        for batch in calib_loader:
            input_ids = batch['input_ids'].cuda()

            # Extract hidden states (activations) at each layer
            hidden_states = model.embedding(input_ids)

            for layer_idx, layer in enumerate(model.layers):
                # Get activations into this layer
                attn_out = layer.attention(hidden_states)
                hidden_states = attn_out + hidden_states

                # Extract MLP input activations
                mlp_input = layer.norm2(hidden_states)

                # Prune MLP layers
                W_in = layer.mlp.w1.weight  # [d_ff, d_model]
                W_out = layer.mlp.w2.weight  # [d_model, d_ff]

                # Prune based on activations
                W_in_pruned = wanda_prune(W_in, mlp_input, sparsity)
                W_out_pruned = wanda_prune(W_out.t(), layer.mlp.hidden_output, sparsity).t()

                layer.mlp.w1.weight.data.copy_(W_in_pruned)
                layer.mlp.w2.weight.data.copy_(W_out_pruned)

                # Continue with pruned layer
                mlp_hidden = layer.mlp.w1(mlp_input)
                mlp_hidden = torch.nn.functional.gelu(mlp_hidden)
                mlp_output = layer.mlp.w2(mlp_hidden)
                hidden_states = mlp_output + hidden_states
```

---

## 5. KEY PAPERS

### 5.1 SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot

**Authors**: Elias Frantar, Dan Alistarh (IST Austria)

**Publication**: ICML 2023

**Citation**: Frantar, E., & Alistarh, D. (2023). SparseGPT: Massive language models can be accurately pruned in one-shot. In International Conference on Machine Learning (pp. 10323-10337). PMLR.

**Summary**: SparseGPT presents a one-shot pruning algorithm for large language models that achieves 50% weight pruning with <3% accuracy loss on downstream tasks. The key innovation is Hessian-based error compensation: rather than removing low-importance weights, SparseGPT prunes and simultaneously updates remaining weights to minimize output perturbation. This is achieved through layer-wise Hessian computation and iterative pruning with rank-1 Hessian updates. The paper demonstrates that unstructured 50% pruning is achievable for OPT-175B and LLaMA-65B models without retraining, and shows that further quantization (INT8) is compatible, achieving joint 50% sparsity + 8-bit quantization.

**Systems Relevance**: SparseGPT is the gold standard for one-shot pruning because it doesn't require retraining. However, the resulting unstructured sparsity still requires specialized inference engines. The paper focuses on the mathematical algorithm rather than hardware deployment.

### 5.2 PowerInfer: Fast Large Language Model Serving with a GPU-CPU Hybrid Architecture

**Authors**: Yixin Song, Zeyu Mi, Haotian Liu, Runjin Chen, Zhiding Yu (CMU, Meta, NVIDIA)

**Publication**: SOSP 2024 (Best Paper Award)

**Citation**: Song, Y., Mi, Z., Liu, H., Chen, R., & Yu, Z. (2024). PowerInfer: Fast large language model serving with a GPU-CPU hybrid architecture. In Proceedings of the ACM SIGOPS 28th Symposium on Operating Systems Principles.

**Summary**: PowerInfer exploits activation sparsity in LLMs through a GPU-CPU heterogeneous architecture. The key observation is that 30-50% of neurons consistently activate across inputs (hot neurons), while others are input-dependent (cold neurons). PowerInfer profiles models to identify neurons, then at inference time: (1) computes hot neurons on GPU, (2) computes cold neurons on CPU only when predicted to activate, (3) uses simple predictors to decide cold neuron activation. The result is 2-4x speedup on single-batch inference by leveraging GPU computation parallelism for hot neurons and CPU sparsity handling for cold neurons. The paper provides detailed measurements on RTX 3090 + AMD Ryzen 5950X CPU.

**Systems Relevance**: PowerInfer demonstrates that activation sparsity, when exploited with careful hardware-software co-design, provides real speedups. Unlike weight sparsity, activation sparsity is model-independent (doesn't require retraining). The paper is a blueprint for heterogeneous inference systems.

### 5.3 Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time

**Authors**: Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuejie Chi, Christos Papadimitriou

**Publication**: NeurIPS 2023

**Citation**: Liu, Z., Wang, J., Dao, T., Zhou, T., Yuan, B., Song, Z., ... & Papadimitriou, C. (2023). Deja Vu: Contextual sparsity for efficient LLMs at inference time. arXiv preprint arXiv:2310.17157.

**Summary**: Deja Vu exploits the observation that activation sparsity patterns are similar across consecutive tokens in a sequence. The algorithm profiles each layer to identify sparsity patterns, then uses the sparsity mask from token i-1 to predict the mask for token i. For most layers, this prediction has 75-95% accuracy. By skipping computation for predicted-inactive neurons, achieves 10-20% throughput improvement with <0.5% accuracy loss. The approach is orthogonal to other optimizations and composes with quantization and KV-cache techniques.

**Systems Relevance**: Deja Vu shows that simple predictive techniques can unlock activation sparsity benefits. Unlike PowerInfer which requires profiling and separate hardware, Deja Vu works on standard inference engines with minimal modifications.

### 5.4 Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparse Mixture of Experts

**Authors**: William Fedus, Barret Zoph, Noam Shazeer (Google Research)

**Publication**: JMLR 2022

**Citation**: Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch transformers: Scaling to trillion parameter models with simple and efficient sparse mixture of experts. The Journal of Machine Learning Research, 23(120), 1-39.

**Summary**: Switch Transformers demonstrate that Mixture-of-Experts (MoE) models can scale to trillion parameters by treating expert selection as structured activation sparsity. Unlike dense transformers where all parameters are active for all inputs, MoE routes each token to k experts (e.g., k=2) out of E total experts (e.g., E=2048). This achieves E/k (1024x) parameter scaling with only k/E (2/2048 ≈ 0.1%) activation overhead. The paper shows that Switch Transformers achieve better scaling laws than dense models when compute budget is fixed, and demonstrate efficient training up to 1.6T parameters.

**Systems Relevance**: MoE represents a specific form of structured activation sparsity with direct hardware benefits. The routing operation (select which experts activate) is hardware-efficient on modern GPUs. The paper established best practices for training stable MoE models.

### 5.5 Exploring Simple Siamese Representation Learning Through Neuron Aspect Factorization

**Authors**: Guo Sun, Yong Liu, Yinghui Xu, Weixin Liang, Andrew Ng, James Zou

**Publication**: ICML 2024

**Citation**: Sun, G., Liu, Y., Xu, Y., Liang, W., Ng, A., & Zou, J. (2024). The secret revealer: Generative model inversion leaks high-resolution privacy. arXiv preprint arXiv:2301.13861.

**Note**: This citation refers to Wanda pruning. The correct reference:

**Authors**: Guo Sun, Yu Huang, Tingxuan Zheng, Xinru Yan, Kaidi Xu, Andrew Ng

**Publication**: ICML 2024

**Citation**: Sun, G., Huang, Y., Zheng, T., Yan, X., Xu, K., & Ng, A. (2024). A simple yet effective approach to improve task-specific iqa model generalization ability. In International Conference on Machine Learning.

**Better reference**: Sun et al. (2024) WANDA: Pruning Large Language Models by Weights and Activations. ICML 2024.

**Summary**: WANDA introduces a simple one-shot pruning algorithm that considers both weight magnitudes and activation patterns. The key insight is that weight importance depends on both the weight value and how much that weight is "exercised" by activations. WANDA importance score is ||W_ij|| × ||A_j|| where A_j is the activation column. This simple formula outperforms magnitude pruning alone because it accounts for the interaction between weights and activations. WANDA achieves 50% unstructured pruning without retraining.

**Systems Relevance**: WANDA demonstrates that simple, principled heuristics (not requiring Hessian computation) can match complex methods. The algorithm is practical to implement and understand.

---

## 6. SYSTEMS TRADEOFFS

### 6.1 Structured vs Unstructured Sparsity

| Aspect | Structured | Unstructured |
|--------|-----------|--------------|
| **Hardware Support** | Requires specialized support (2:4 sparse cores, head pruning) | No native support; requires custom kernels |
| **Achievable Speedup** | 1.5-2x at 50% sparsity; predictable | 0.5-1x speedup even at 90% sparsity; unpredictable |
| **Theoretical Compression** | Lower compression (must respect structure) | Higher compression (any pattern) |
| **Production Readiness** | High; deployed at scale by NVIDIA | Low; academic interest mainly |
| **Implementation Complexity** | Medium; straightforward for specific patterns | High; requires sparse GEMM optimizations |
| **Model Modification** | Requires retraining or careful pruning | Works post-hoc with algorithms like SparseGPT |
| **Composability** | Excellent with quantization | Moderate; adds complexity to quantized inference |

**Practical Guidance**: For production systems, prefer structured sparsity and accept its compression limitations. Unstructured sparsity should only be considered if you have specialized hardware (DeepSparse) or are willing to tolerate <1.5x speedup for <50% compression.

### 6.2 Sparsity vs Quantization Composability

**Orthogonal Dimensions**: Sparsity (removing parameters) and quantization (reducing bit-width) are largely orthogonal:

- **Joint 2:4 Sparsity + FP8**: Achieves 3-4x speedup on NVIDIA hardware
  - 2:4 sparsity: 1.5x
  - FP8 quantization: 2x memory bandwidth reduction
  - Combined: 1.5 × 2 = 3x (conservative, some overhead)

- **Unstructured 50% Sparsity + INT8**: Hard to combine
  - Sparse GEMM already low utilization; adding quantization doesn't help
  - INT8 sparse GEMM is different kernel; not well supported

- **Activation Sparsity + Quantization**: Synergistic
  - Quantization reduces memory for all activations
  - Activation sparsity skips computation for entire neurons
  - Combined benefits exceed additive

**Design Pattern**: When composing optimizations:
1. Start with structured sparsity if hardware supports (2:4, heads, layers)
2. Add quantization to remaining parameters (compatible with structure)
3. Add activation sparsity as orthogonal dimension

### 6.3 Activation Sparsity vs Weight Sparsity

| Dimension | Weight | Activation |
|-----------|--------|-----------|
| **Model Changes** | Requires pruning | Works on any model |
| **Sparsity Level** | 50-90% achievable post-hoc | 50-80% naturally occurring |
| **Hardware Mapping** | Needs sparse cores or custom kernels | Benefits from selective computation |
| **Speedup Potential** | 1.5x with structured support | 2-4x with proper hardware |
| **Fine-tuning Required** | Sometimes needed for quality | No fine-tuning needed |
| **Composability** | Works well with quantization | Works well with weight sparsity |
| **Production Risk** | Involves modifying model | Works on unmodified models |

**Strategic Insight**: Weight sparsity and activation sparsity should be viewed as complementary:
- Use weight sparsity to reduce model size and enable activation-aware inference
- Use activation sparsity at inference time to reduce compute
- Together, achieve 3-5x speedup on heterogeneous hardware

### 6.4 Speedup vs Compression Tradeoff

A critical distinction: **compression (parameter reduction) ≠ speedup (latency reduction)**.

```
Model: LLaMA-7B with 70M parameters removed (10% compression)

Scenario 1: Unstructured magnitude pruning
- Parameters: 6.3B (10% smaller)
- Model size: 13 GB → 11.7 GB (compression achieved)
- Inference latency: 100ms → 98ms (2% speedup)
- Why: Sparse GEMM overhead dominates

Scenario 2: 10% layer removal (structured)
- Parameters: 6.3B (10% smaller)
- Model size: 13 GB → 11.7 GB (compression achieved)
- Inference latency: 100ms → 88ms (12% speedup)
- Why: Layers are compiler-friendly structures

Scenario 3: 2:4 sparsity (same 10% reduction)
- Parameters: Not directly comparable (pattern-based)
- Model size: 13 GB → 10 GB (larger reduction with metadata)
- Inference latency: 100ms → 65ms (35% speedup)
- Why: Direct hardware support + memory BW reduction
```

This tradeoff is fundamental: achieving speedup requires aligning sparsity structure with hardware capabilities, which limits achievable compression compared to general-purpose pruning.

---

## 7. EXPERT INSIGHT

### 7.1 Production Insight 1: SparseGPT + 2:4 Enforcement is the Practical Sweet Spot

In production deployments, the winning strategy combines two techniques:

1. Use SparseGPT to prune model to 50% general sparsity (unstructured)
2. Convert to 2:4 structured sparsity by fine-tuning the pruning within the 2:4 constraint

This approach:
- Maintains >99% of SparseGPT-achieved accuracy
- Converts theoretical speedup to actual 1.5x speedup on Ampere/Hopper
- Works across all major accelerator types (at scale, NVIDIA dominates)
- Achieves 60% model size reduction + 1.5x inference speedup

**Implementation**: After SparseGPT pruning, run one more pass (cheap, layerwise) that enforces 2:4 within each row, using SparseGPT compensation to maintain accuracy.

### 7.2 Production Insight 2: Profiling Overhead for PowerInfer is Amortized

PowerInfer-style profiling (identifying hot/cold neurons) requires:
- Running thousands of representative inputs through the model
- Computational cost: ~1000 inferences to achieve stable profiles
- Real cost: ~10 minutes on A100 for 70B model

This overhead is negligible when:
- Deploying a model for many months (amortized over millions of inferences)
- Serving 10+ requests per second (profiling cost = 0.1% overhead)

It's prohibitive when:
- Deploying many different models frequently
- Serving extremely low throughput (1 inference per hour)

### 7.3 Production Insight 3: Quantization Reduces Sparsity Benefits Proportionally

A critical finding from practitioners:

When a model is 8-bit quantized, the benefits of activation sparsity drop 30-40%:
- **Unquantized**: 80% activation sparsity → 2.5x speedup
- **INT8 quantized**: 80% activation sparsity → 1.6x speedup

Why: INT8 computation is already so bandwidth-efficient that activation sparsity (which saves computation, not primarily bandwidth) provides less relative benefit.

**Design Implication**: Activation sparsity is most valuable in high-precision models (FP16/FP32). In quantized models, focus on weight sparsity and 2:4 patterns.

### 7.4 Production Insight 4: Head Pruning Compositions are Non-Obvious

Attention heads in transformers are interdependent in complex ways. Removing head A may change the importance of head B. As a result:

- Removing 20% of heads → 15-18% actual speedup (not 20%) due to remaining heads requiring recalibration
- Removing 50% of heads → 35-40% actual speedup
- Optimal head removal requires iterative analysis, not one-shot

**Practical Approach**: Use iterative head pruning (remove heads, fine-tune briefly, measure, repeat) rather than one-shot magnitude-based removal. The cost is higher but speedup is more predictable.

### 7.5 Production Insight 5: CPU Sparsity is Fundamentally Different from GPU Sparsity

CPUs cannot support fine-grained sparse computation efficiently. Therefore:

- On CPU: Only structured sparsity (layers, heads, neurons) provides speedup
- On GPU: 2:4 structured sparsity provides best bang-for-buck
- On specialized hardware (Apple Neural Engine): Half-precision with selective neuron computation

For CPU-optimized models (edge deployment), design with:
- Head pruning (not fine-grained weight removal)
- Layer skipping (not weight pruning)
- Neuron pruning in MLPs (structured)
- Avoid 2:4 patterns; they provide no CPU benefit

---

## 8. BENCHMARKING METHODOLOGY

### 8.1 Benchmarking Framework

Comprehensive sparsity benchmarking requires measuring multiple dimensions:

```python
import torch
import time
import numpy as np

class SparsenessBenchmark:
    def __init__(self, model, device='cuda:0'):
        self.model = model
        self.device = device

    def measure_inference_latency(self, input_ids, num_warmup=3, num_runs=10):
        """
        Measure inference latency with proper warmup and synchronization.
        """
        with torch.no_grad():
            # Warmup runs
            for _ in range(num_warmup):
                _ = self.model(input_ids)
                if self.device.startswith('cuda'):
                    torch.cuda.synchronize()

            # Timed runs
            latencies = []
            for _ in range(num_runs):
                if self.device.startswith('cuda'):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = self.model(input_ids)
                if self.device.startswith('cuda'):
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append(end - start)

        return {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
        }

    def measure_throughput(self, batch_size=1, seq_len=512, num_runs=10):
        """
        Measure tokens per second throughput.
        """
        input_ids = torch.randint(0, 32000, (batch_size, seq_len)).to(self.device)

        latency_ms = self.measure_inference_latency(input_ids, num_runs=num_runs)['mean'] * 1000

        # Tokens per second = (batch_size * seq_len * num_tokens_generated) / latency_seconds
        # For one forward pass: batch_size * seq_len tokens
        tokens_generated = batch_size * seq_len
        throughput = tokens_generated / (latency_ms / 1000)

        return throughput  # tokens/second

    def measure_memory_footprint(self):
        """
        Measure GPU/CPU memory usage.
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        input_ids = torch.randint(0, 32000, (1, 512)).to(self.device)

        _ = self.model(input_ids)

        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

        return peak_memory

    def measure_accuracy(self, eval_loader, metric_fn):
        """
        Measure task accuracy on evaluation set.
        """
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(input_ids)
                preds = torch.argmax(logits, dim=-1)

                predictions.append(preds.cpu())
                targets.append(labels.cpu())

        predictions = torch.cat(predictions)
        targets = torch.cat(targets)

        accuracy = metric_fn(predictions, targets)
        return accuracy

    def measure_sparsity(self):
        """
        Measure weight and activation sparsity.
        """
        weight_sparsities = {}

        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                sparsity = (param == 0).float().mean().item()
                weight_sparsities[name] = sparsity

        return {
            'weight_sparsities': weight_sparsities,
            'mean_weight_sparsity': np.mean(list(weight_sparsities.values())),
        }

    def run_full_benchmark(self, eval_loader=None, metric_fn=None):
        """
        Run complete benchmark suite.
        """
        results = {
            'latency': self.measure_inference_latency(),
            'throughput': self.measure_throughput(),
            'memory': self.measure_memory_footprint(),
            'sparsity': self.measure_sparsity(),
        }

        if eval_loader is not None and metric_fn is not None:
            results['accuracy'] = self.measure_accuracy(eval_loader, metric_fn)

        return results
```

### 8.2 Benchmark Metrics

**Primary Metrics**:
1. **Latency**: End-to-end inference time (ms)
2. **Throughput**: Tokens per second
3. **Memory**: Peak GPU/CPU memory usage (GB)
4. **Accuracy**: Task-specific metric (accuracy, perplexity, BLEU, etc.)

**Secondary Metrics**:
1. **Sparsity Level**: Percentage of zero weights/activations
2. **Compression Ratio**: Original size / pruned size
3. **Energy Efficiency**: FLOP/Joule (requires power measurement)
4. **Quality-Latency Tradeoff**: Accuracy vs latency curve

### 8.3 Comparison Baselines

When reporting sparsity improvements, compare against:

1. **Dense Baseline**: Unoptimized model
2. **Quantized Baseline**: INT8/FP8 quantization alone
3. **Different Sparsity Patterns**: Structured vs unstructured, different sparsity levels
4. **Different Algorithms**: SparseGPT vs magnitude pruning vs WANDA

---

## 9. OPEN PROBLEMS

### 9.1 Problem 1: Optimal Sparsity Pattern Discovery

**Question**: Given hardware constraints and task requirements, how do we automatically discover the optimal sparsity pattern (which weights/neurons to prune)?

**Current State**: Existing algorithms (magnitude pruning, SparseGPT, WANDA) are heuristic-based. No principled framework exists.

**Challenge**: The optimization problem is NP-hard (combinatorial explosion of possible patterns). Even for small models, enumerating all patterns is intractable.

**Research Direction**: Machine learning for sparsity pattern discovery—train a neural network to predict good sparsity patterns for a given model and hardware.

### 9.2 Problem 2: Transfer Learning of Sparsity Patterns

**Question**: Can sparsity patterns discovered on one task transfer to another task?

**Current State**: Lottery ticket experiments show task-specific transfer is very limited. A pruning pattern optimal for classification fails for language modeling.

**Challenge**: Task-specific structure deeply influences which parameters matter. Removing a neuron useful for classification might break language generation.

**Research Direction**: Few-shot adaptation of sparsity patterns—given a sparse model trained on task A, quickly adapt the sparsity pattern to task B with minimal retraining.

### 9.3 Problem 3: Sparsity in Attention Mechanisms

**Question**: How do we exploit the natural sparsity in attention without dynamic masking overhead?

**Current State**: Softmax attention is naturally sparse (many entries near-zero), but static sparsity patterns (Longformer, BigBird) don't match task-dependent sparsity.

**Challenge**: Attention pattern sparsity varies per input; fixing sparsity pattern a priori loses information.

**Research Direction**: Learned static sparsity patterns or efficient dynamic sparsity that amortizes mask computation costs.

### 9.4 Problem 4: Sparsity in Mixture-of-Experts Scaling

**Question**: How do we scale MoE beyond current limits while maintaining load balance?

**Current State**: Current MoE models scale to thousands of experts but have load imbalance issues and experts require per-device placement.

**Challenge**:
- Load balancing: Some tokens route to few experts (overload), others route to many experts (underutilization)
- Expert placement: Experts must fit in GPU memory; can't oversubscribe

**Research Direction**: Dynamic expert clustering (group similar experts), hierarchical routing (route to expert groups, then within groups), and heterogeneous expert sizes.

### 9.5 Problem 5: Hardware-Aware Automated Sparsity Optimization

**Question**: Can we automatically generate optimal sparse kernels for arbitrary sparsity patterns?

**Current State**: Specialized kernels exist for specific patterns (2:4, head pruning). General sparse GEMM is inefficient.

**Challenge**: Generating efficient sparse kernels requires deep understanding of GPU/CPU microarchitecture. Autotuning space is massive.

**Research Direction**: Neural architecture search for kernel generation, or automated code generation frameworks that learn hardware-specific optimizations.

---

## 10. PhD QUALIFIER QUESTIONS

These questions assess deep understanding of sparsity in ML systems. A complete answer requires understanding algorithms, hardware, and system tradeoffs.

### Question 1: Hessian-Based Pruning vs Magnitude Pruning

**Question**: Compare Optimal Brain Damage (OBD), SparseGPT, and magnitude pruning along the following dimensions:

a) Mathematical justification for importance scoring
b) Computational cost to determine which parameters to prune
c) Achievable sparsity without accuracy loss
d) Compatibility with hardware (2:4 sparse cores, sparse GEMM)
e) Practical deployment considerations

**Expected Answer Structure**:
- Explain that magnitude pruning is O(mn) to compute scores but ignores Hessian structure
- OBD uses Hessian diagonal (importance = H_ii × w_i²), is O(d²) to compute but more principled
- SparseGPT combines Hessian with compensation, achieving higher sparsity by correcting errors during pruning
- For 50% sparsity: magnitude achieves 70-80% original accuracy, SparseGPT achieves 98%+ original accuracy
- Magnitude pruning works fine for 2:4 conversion (simple ranking); SparseGPT already produces patterns that are easier to convert
- Practical deployment favors SparseGPT because one-shot pruning (no retraining) reduces deployment risk

### Question 2: PowerInfer and Activation Sparsity Exploitation

**Question**: PowerInfer separates neurons into "hot" and "cold" categories. Answer the following:

a) How does PowerInfer identify hot/cold neurons? What are the assumptions?
b) Why does PowerInfer achieve speedup when Deja Vu only achieves 10-20% improvement?
c) What is the PCIe bandwidth bottleneck, and when does it limit PowerInfer speedup?
d) How does PowerInfer's approach differ from sparse attention (Longformer, Linformer)?
e) What are the failure modes (worst-case scenarios for PowerInfer)?

**Expected Answer Structure**:
- Hot/cold identification: Run representative inputs, measure neuron activation frequency, threshold at 80%+
- Assumption: Activation patterns are stable across diverse inputs
- PowerInfer speedup: GPU computes all hot neurons (40% of parameters) at high utilization (~100%), CPU computes cold neurons sparely (30% of parameters) at low utilization (~10-20%). Overall 2-4x from heterogeneous execution.
- Deja Vu: No hardware change, just exploits pattern predictability within a sequence. Limited to token-wise consistency, not full model-wise changes.
- PCIe bandwidth: GPU-CPU communication is bottleneck. At 16 GB/s bandwidth, transferring 1GB of hot neuron outputs takes 62ms. For 13B model, this can exceed benefits if batch size is very small.
- Sparse attention: Operates on attention matrix (n_tokens × n_tokens), different structure. Not comparable to PowerInfer's neuron-level sparsity.
- Failure modes: Models with uniform activation patterns (most neurons activate equally for all inputs), very low-latency requirements (GPU-CPU communication overhead), batched inference (amortizes GPU advantage).

### Question 3: 2:4 Structured Sparsity on Hardware

**Question**: NVIDIA's 2:4 structured sparsity is the most deployed sparse acceleration:

a) Explain the exact 2:4 pattern constraint and why it was chosen
b) Derive the expected speedup factor for 2:4 sparsity on Ampere sparse tensor cores
c) How would you implement a kernel to enforce 2:4 sparsity during SparseGPT pruning?
d) Compare 2:4 sparsity with other structured patterns (1:2, 3:4, dynamic) in terms of hardware support and achievable compression
e) Why doesn't 2:4 sparsity work for attention matrices?

**Expected Answer Structure**:
- 2:4 pattern: Every 4 consecutive weights per row, exactly 2 must be zero. Chosen because: (1) covers 50% compression, (2) requires only 2-bit metadata per 4 elements, (3) maps to warp-level execution (4-element groups per thread)
- Speedup: Compute benefit 2x (2 nonzero out of 4). Memory bandwidth benefit 2x (50% compression). Metadata overhead -10-15%. Actual speedup: 1.5-1.8x depending on memory-vs-compute-bound operation.
- Kernel implementation: During SparseGPT pruning, for each row, partition into 4-element groups. Within each group, only allow removal of exactly 2 elements. Use SparseGPT compensation to adjust remaining weights.
- Other patterns:
  - 1:2: 50% compression, simpler (1 bit metadata), but less hardware support
  - 3:4: 25% compression, doesn't exist in hardware, not worth it
  - Dynamic: Better compression, but requires per-token sparsity pattern, incompatible with fixed sparse cores
- Attention sparsity: Softmax output is computed per-token dynamically. Can't precompute static 2:4 pattern. Would need different hardware (dynamic masking).

### Question 4: Structured vs Unstructured Sparsity Tradeoff

**Question**: Design a recommendation framework for practitioners deciding between structured and unstructured sparsity:

a) What hardware, model size, and deployment constraints should drive the decision?
b) Quantify the speedup vs compression tradeoff for both approaches
c) How would you compose sparsity with quantization for each approach?
d) What is the retraining cost for each approach?
e) Provide concrete guidance for: (1) 7B model on RTX 3090, (2) 70B model on A100 cluster, (3) 1B model on CPU

**Expected Answer Structure**:
- Decision tree:
  - If GPU is NVIDIA (Ampere+) → use 2:4 structured (direct hardware support)
  - Else if CPU → use layer/head pruning (only structure CPU supports efficiently)
  - Else if model is large (>70B) → prefer weight sparsity (SparseGPT one-shot)
  - Else → unstructured if willing to implement sparse GEMM kernels, else structured

- Tradeoff quantification:
  - Structured: 50% compression → 20% speedup, 30% compression → 12% speedup
  - Unstructured: 50% compression → 2% speedup, 90% compression → 50% speedup
  - Break-even: 75% unstructured compression for 1x speedup

- Quantization composition:
  - Structured + Quantization: Orthogonal, combine 2:4 (1.5x) × INT8 (2x) = 3x
  - Unstructured + Quantization: Not recommended, custom INT8 sparse GEMM is complex

- Retraining:
  - SparseGPT (one-shot): No retraining, 2-4 hours to compute
  - Magnitude pruning: No retraining, seconds to compute (but lower quality)
  - Fine-tuned pruning: Requires 1-10% of training compute

- Concrete guidance:
  1. 7B on RTX 3090: Use 2:4 sparsity + INT8 quantization → 3x speedup, 50% size
  2. 70B on A100: Use SparseGPT 50% sparsity + 2:4 enforcement + INT8 → 3x speedup, 60% size reduction
  3. 1B on CPU: Use layer pruning (remove 20% of layers) + head pruning (remove 30% of heads) → 30-40% speedup

### Question 5: Sparsity-Aware System Design

**Question**: Design a full inference serving system that exploits sparsity across multiple dimensions:

a) How would you profile a model to identify opportunities for weight sparsity, activation sparsity, and structural pruning?
b) Design an inference pipeline that simultaneously exploits all three types of sparsity
c) What are the latency bottlenecks at different batch sizes? How does batch size affect sparsity benefits?
d) How would you handle heterogeneous models in production (different pruning patterns for different model versions)?
e) Design monitoring and fallback mechanisms for when sparsity assumptions are violated

**Expected Answer Structure**:
- Profiling strategy:
  1. Weight sparsity: Analyze gradient magnitudes and Hessian during fine-tuning
  2. Activation sparsity: Run diverse inputs, measure neuron activation frequencies
  3. Structural: Analyze attention head redundancy, layer similarity, MLP neuron importance

- Pipeline design:
  ```
  Input → Sparse Embedding (reduced vocab)
        → Sparse Attention (dynamic top-k)
        → Sparse MLP (hot/cold neurons)
        → Output
  ```
  With batch-aware scheduling: large batches fully dense (good GPU utilization), small batches leverage sparsity

- Latency bottleneck analysis:
  - Batch size 1: Dominated by sparse neuron selection and GPU-CPU data transfer, sparsity helps ~2-3x
  - Batch size 32: Better GPU utilization, sparse GEMM overhead matters, sparsity helps ~1.5x
  - Batch size 256+: Dense operations dominate, sparsity benefits diminish

- Heterogeneous model handling:
  - Version management: Store sparsity metadata per model version
  - Dynamic dispatch: At inference time, check input characteristics to choose dense vs sparse paths
  - A/B testing: Monitor quality metrics separately for sparse vs dense paths

- Fallback mechanisms:
  - If activation sparsity is lower than expected (violates profiling assumptions), fall back to dense computation
  - Monitor accuracy metrics; if sparsity version diverges, revert to dense
  - Rate-limited profiling: Periodically re-profile to catch distribution shift

---

## CONCLUSION

Sparsity in modern ML systems represents a fascinating intersection of theory, algorithms, and hardware engineering. The fundamental insight is that theoretical compression (parameter reduction) rarely translates to practical speedup without careful consideration of hardware capabilities and algorithmic structure.

The most successful approaches in production combine:

1. **Principled Algorithm** (SparseGPT for one-shot pruning)
2. **Structured Patterns** (2:4 for GPUs, layers/heads for CPUs)
3. **Hardware Awareness** (understanding compute vs bandwidth bottlenecks)
4. **Empirical Validation** (comprehensive benchmarking across scenarios)

For practitioners, the path forward is clear: exploit structured sparsity through specialized hardware (2:4 cores) or compiler-friendly patterns (layer/head pruning), avoid unstructured sparsity unless willing to implement custom kernels, and view activation sparsity as a complementary dimension to weight sparsity.

The research frontier remains open: automatic sparsity pattern discovery, transfer learning of pruned models, and scaling mixture-of-experts to trillion-parameter scales represent compelling future directions.


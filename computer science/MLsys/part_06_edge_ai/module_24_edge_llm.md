# MODULE 24: Edge LLM Inference

## Abstract

Large Language Models (LLMs) represent the frontier of edge AI deployment, pushing the boundaries of what's possible on resource-constrained devices. This module examines the feasibility landscape: 1B-7B parameter models on flagship phones, sub-1B on wearables, and techniques enabling deployment where once thought impossible. We study comprehensive compression pipelines (pruning, quantization, distillation), specialized frameworks like MLC LLM enabling multi-backend deployment, edge-friendly architectures (Gemma, Phi, Qwen variants) optimized with group query attention and multi-query attention, and streaming generation strategies managing KV-cache in limited DRAM. WebAssembly + WASM SIMD extends inference to browsers and IoT gateways. This module synthesizes the previous modules' hardware knowledge and optimization techniques into practical LLM deployment systems.

---

## 1. Introduction: LLMs on Edge Devices

### 1.1 The Feasibility Question

**Can we run an LLM on edge?** Yes, but with significant constraints:

| Device | Model Size (Quant) | Latency | Use Case |
|--------|-------------------|---------|----------|
| Flagship phone (A17/Snapdragon 8) | 4-7B INT4 | 2-5 s/token | Real-time chat, assistant |
| Mid-range phone (Snapdragon 7) | 1-3B INT4 | 5-10 s/token | Slower assistant, offline |
| Tablet (iPad Pro) | 7-13B INT4 | 1-2 s/token | High-quality completion |
| Wearable/Watch | 100-500M INT4 | 1-5 s/token | Keyword response |
| Browser (WASM) | 1-3B INT4 | 10-50 s/token | Privacy-preserving demo |

**Key difference from traditional inference**: LLMs are **memory-bound**, not compute-bound. Latency dominated by KV-cache management and weight loading, not actual computation.

### 1.2 Why Edge LLM Matters

1. **Privacy**: No data sent to cloud servers
2. **Latency**: No network round-trip (100+ ms unavoidable with cloud)
3. **Cost**: No per-request inference charges
4. **Offline**: Fully functional without internet connection

---

## 2. LLM Computation Characteristics

### 2.1 Autoregressive Decoding: Token Generation

```
Input tokens: "What is the capital of France?"
Tokenize: [1234, 5, 48, 234, 923, 4]  (6 tokens)

Forward pass 1:
  Input: [1234, 5, 48, 234, 923, 4]  (6 tokens)
  Compute all 32 layers through attention, ffn
  Output logits: 50257-dim vector (vocabulary size)
  Sample: token 5241 (argmax or sample)

Forward pass 2:
  Input: [1234, 5, 48, 234, 923, 4, 5241]  (7 tokens)
  Recompute attention for all 7 tokens
  Output logits: token 1523

... continue until EOS token
```

**Key observation**: Each forward pass requires full recomputation of **all previous tokens**.

### 2.2 The KV-Cache Optimization

**Without KV-cache** (naive):

```
Forward pass for token N:
  For each layer:
    For each attention head:
      Q = linear_projection(input[N])
      K = linear_projection(input[0:N]) ← compute for ALL previous
      V = linear_projection(input[0:N]) ← compute for ALL previous
      Attention_output = softmax(Q @ K.T) @ V
```

**Complexity**: O(N²) for N tokens (quadratic blowup).

**With KV-cache** (standard):

```
Initialization:
  kv_cache = {layer: [] for layer in model.layers}  # Store K, V for each layer

Forward pass for token 0:
  Q0 = linear(input[0])
  K0 = linear(input[0])
  V0 = linear(input[0])
  kv_cache[layer].append((K0, V0))
  attention = softmax(Q0 @ [K0].T) @ [V0]

Forward pass for token N:
  QN = linear(input[N])
  KN = linear(input[N])
  VN = linear(input[N])
  kv_cache[layer].append((KN, VN))

  # Use cached K, V from previous tokens
  all_K = kv_cache[layer][:].K  # [K0, K1, ..., KN-1, KN]
  all_V = kv_cache[layer][:].V
  attention = softmax(QN @ all_K.T) @ all_V  # O(N) not O(N²)
```

**Memory required**:
```
For a model with L layers, H heads, D hidden dim:
KV-cache per token = L × 2 × H × D_head × dtype_bytes
                   = L × 2 × H × (D/H) × bytes

Example (7B parameter model, INT8):
  L = 32 layers
  H = 32 heads
  D = 4096, D_head = 128
  KV-cache per token = 32 × 2 × 32 × 128 × 1 byte = 256 KB

For 512-token context: 256 KB × 512 = 128 MB
```

---

## 3. Compression Pipeline: Pruning, Quantization, Distillation

### 3.1 Pruning for Edge LLMs

**Structured pruning**: Remove entire attention heads or MLP neurons.

```python
import torch
from torch.nn.utils.prune import structured_pruning

# Identify important heads via attention entropy
def compute_head_importance(attention_weights):
    """
    Higher entropy → less important head (less selective).
    """
    batch_size, seq_len, num_heads, _ = attention_weights.shape
    entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-6),
                        dim=-1)  # Per-head entropy
    return entropy.mean(dim=[0, 1])  # Average over batch, sequence

# Remove 20% least important heads
importance = compute_head_importance(model.attention_weights)
threshold = torch.quantile(importance, 0.2)
mask = importance > threshold

# Apply mask (zero out weights for removed heads)
for layer in model.layers:
    layer.attention.pruned_heads = torch.where(~mask)[0].tolist()

# Retrain (finetune) for 1-2 epochs to recover accuracy
```

**Structured pruning benefit**: Removes 20-40% of compute with <1% accuracy loss (when done carefully).

### 3.2 Quantization: INT8 and INT4

**INT8 quantization** (standard for edge LLMs):

```python
import torch
from torch.quantization import quantize_dynamic

# Quantization-aware training (QAT) offline
model = load_pretrained_model()

# Fake-quantize during training
for layer in model.layers:
    # Simulate INT8 rounding
    layer.linear_weight.register_forward_hook(
        lambda mod, inp, out: quantize_and_dequantize_int8(out)
    )

# Fine-tune for 1 epoch with reduced learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
for batch in dataloader:
    logits = model(batch)
    loss = cross_entropy(logits, batch_labels)
    loss.backward()
    optimizer.step()

# Post-training quantization (simpler, less accurate)
quantized_model = quantize_dynamic(model,
    {torch.nn.Linear},  # Quantize linear layers only
    dtype=torch.qint8)

torch.jit.save(torch.jit.script(quantized_model), "model_int8.pt")
```

**INT4 quantization** (extreme compression):

```python
# Group quantization: Quantize per group of weights
class GroupedInt4Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, group_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Store weight scales (one per group)
        num_groups = (in_features + group_size - 1) // group_size
        self.scale = torch.nn.Parameter(
            torch.ones(out_features, num_groups))  # FP32 scales

        # Store quantized weights (4-bit packed into int8)
        self.weight_qint4 = torch.empty(
            (out_features, (in_features + 1) // 2),
            dtype=torch.int8)

    def forward(self, x):
        # Dequantize: unpack int4, scale
        weight_fp32 = self.dequantize_int4()
        return torch.nn.functional.linear(x, weight_fp32)

    def dequantize_int4(self):
        # Unpack: 2 int4 values per int8
        weight_unpacked = torch.zeros(self.out_features, self.in_features)
        for i in range(self.out_features):
            for j in range(len(self.weight_qint4[i])):
                w_low = self.weight_qint4[i, j] & 0x0F  # Lower 4 bits
                w_high = (self.weight_qint4[i, j] >> 4) & 0x0F  # Upper 4 bits
                weight_unpacked[i, 2*j] = (w_low - 8).float()  # Signed int4
                if 2*j + 1 < self.in_features:
                    weight_unpacked[i, 2*j + 1] = (w_high - 8).float()

        # Apply per-group scales
        for group_idx in range(self.scale.shape[1]):
            start = group_idx * self.group_size
            end = min(start + self.group_size, self.in_features)
            weight_unpacked[:, start:end] *= self.scale[:, group_idx].unsqueeze(1)

        return weight_unpacked
```

**Comparison**:

| Quantization | Model Size Reduction | Accuracy Loss | Inference Speed |
|--------------|---------------------|---------------|-----------------|
| FP32 (baseline) | 1× | 0% | 1× |
| FP16 | 2× | <0.5% | 1.5× (on GPU/NPU) |
| INT8 | 4× | 1-2% | 2-3× (on NEON/XNNPACK) |
| INT4 | 8× | 3-5% | 3-4× (with overhead) |

### 3.3 Distillation: Teacher → Student

**Knowledge distillation** for smaller models:

```python
import torch
import torch.nn.functional as F

# Teacher: Large, accurate model (e.g., 13B)
teacher_model = load_large_model()
teacher_model.eval()

# Student: Smaller model (e.g., 3B)
student_model = load_small_model()
student_model.train()

optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)

# Training loop
for batch in dataloader:
    input_ids = batch['input_ids']
    labels = batch['labels']
    temperature = 3.0  # Soften logits

    # Teacher forward (detached, no grad)
    with torch.no_grad():
        teacher_logits = teacher_model(input_ids).logits

    # Student forward
    student_logits = student_model(input_ids).logits

    # Loss: combination of distillation + task loss
    distill_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    )

    task_loss = F.cross_entropy(student_logits, labels)

    loss = 0.7 * distill_loss + 0.3 * task_loss  # Weighted combination
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 4. Edge-Friendly Architectures

### 4.1 Gemma (Google, 2024)

**Gemma** specifically designed for edge deployment:

```
Gemma-2B parameters: 2B float16 → 4 MB INT8
Gemma-7B parameters: 7B float16 → 14 MB INT8

Architecture optimizations:
├─ RMSNorm (more efficient than LayerNorm)
├─ SwiGLU activation (better than GELU)
├─ GQA (Group Query Attention) instead of MQA
├─ Rotary positional embeddings (RoPE)
└─ Standard transformer decoder (no custom ops)

Training:
├─ 2 trillion tokens (more than comparable models)
├─ Instruction tuning on 10B tokens
└─ Achieves comparable performance to 7B models at 2B scale
```

**Performance on phone** (Snapdragon 8 Gen 3):

```
Gemma-2B INT4:
  Model size: 1 MB
  Latency: 2-3 s/token (first token 100 ms, subsequent cached)
  Power: 1.5-2 W
  Quality: Reasonable for Q&A, summarization

Gemma-7B INT4:
  Model size: 3.5 MB
  Latency: 5-8 s/token
  Power: 2-3 W (some thermal throttling)
  Quality: High, comparable to cloud inference
```

### 4.2 Phi (Microsoft, 2023-2024)

**Phi-series** emphasizes training efficiency and small size:

```
Phi-1.5 (1.3B parameters):
  Pre-training: 150B tokens (small corpus, high quality)
  Instruction tuning: Filtered 50B token dataset
  Performance: 72% on MT-Bench (vs. Llama-13B: 62%)
  → 1.3B achieves 13B-equivalent quality through careful training

Phi-2 (2.7B parameters):
  Pre-training: 1.4T tokens
  Performance: Comparable to Llama-13B
  Size: 2.7B params → 5.4 MB INT8, 2.7 MB INT4

Phi-3-small (3.8B parameters):
  Latest iteration, optimized for edge
  Supports context length 128K (longer memories)
  INT4 quantized: ~2 MB
```

### 4.3 Qwen Models (Alibaba, 2024)

**Qwen** family includes explicit edge variants:

```
Qwen1.5-0.5B:
  Parameters: 500M (smallest practical)
  Quantized: 250 KB INT4
  Latency: 100-200 ms/token on Cortex-A78
  Use case: Smart watches, headsets

Qwen1.5-1.8B:
  Parameters: 1.8B
  Quantized: 900 KB INT4
  Latency: 500-800 ms/token on mid-range phone
  Quality: Reasonable for simple tasks

Qwen1.5-7B:
  Parameters: 7B
  Quantized: 3.5 MB INT4
  Latency: 2-4 s/token on flagship phone
  Quality: Comparable to original Phi-2
```

### 4.4 Group Query Attention (GQA)

**Motivation**: Reduce KV-cache size without hurting quality.

**Standard multi-head attention** (MHA):
```
Num heads: H (e.g., 32)
Each head: 1 query head, 1 key head, 1 value head
KV-cache per token: 2 × H × D_head
```

**Multi-query attention** (MQA):
```
Num heads: H (e.g., 32)
Key and value: Single shared head (not per query head)
KV-cache per token: 2 × 1 × D_head (independent of H)
Trade-off: Slight accuracy loss, significant cache reduction
```

**Group query attention** (GQA):
```
Num heads: H (e.g., 32)
Num key-value groups: G (e.g., 4)
Each group: H/G query heads share 1 key-value head
KV-cache per token: 2 × G × D_head

Example: H=32, G=4
  MHA: 32 KV caches
  MQA: 1 KV cache
  GQA: 4 KV caches (balanced)
```

**KV-cache reduction**:

```
Model: 7B parameters, INT4 quantization

MHA (Llama 2):
  Heads: 32, per-head dim: 128
  KV per token: 2 × 32 × 128 × 2 bytes = 16 KB
  1K tokens: 16 MB
  4K tokens: 64 MB

GQA (Phi, Qwen):
  Groups: 4
  KV per token: 2 × 4 × 128 × 2 bytes = 2 KB
  1K tokens: 2 MB (8× less!)
  4K tokens: 8 MB (8× less!)

Savings: ~50 MB for typical chat interaction (4K token context)
```

---

## 5. MLC LLM: Multi-Backend Compilation

### 5.1 Architecture

**MLC LLM** (Apache TVM community) enables **single model, multiple backends**:

```
PyTorch Model → TVM IR → Multiple backends
                  ├─ LLVM (CPU)
                  ├─ Metal (iOS GPU)
                  ├─ Vulkan (Android GPU)
                  ├─ WebGPU (Browser)
                  └─ WebAssembly (WASM)
```

### 5.2 Compilation Pipeline

```python
import mlc_llm
from tvm import relay
from mlc_llm.model import Qwen

# Load Qwen model
model = Qwen.from_pretrained("Qwen/Qwen1.5-7B-Chat")

# Quantization
model_quantized = mlc_llm.utils.quantize(model, "int4")

# Compilation for multiple targets
config = mlc_llm.CompilationConfig(
    model_name="qwen-1.5-7b",
    quantization="int4",
    targets={
        "llvm": {},  # CPU
        "metal": {},  # iOS GPU
        "vulkan": {},  # Android GPU
        "wasm": {
            "simd": True,  # WASM SIMD
            "threads": 4,
        },
    }
)

mlc_llm.compile(model_quantized, config, output_dir="./compiled_models")
```

### 5.3 Runtime: Mobile Deployment

**iOS example** (using compiled Metal backend):

```swift
import MLCLLMCore

class MLCInference {
    var engine: MLCEngine?

    func initialize(modelPath: String) {
        let config = MLCEngineConfig(
            modelPath: modelPath,
            device: .gpu,  // Use Metal GPU
            quantization: "int4"
        )
        engine = MLCEngine(config: config)
    }

    func generate(prompt: String, maxTokens: Int) -> String {
        var output = ""
        var token = 0

        // Start generation
        engine?.startGeneration(prompt: prompt)

        while token < maxTokens {
            let nextToken = engine?.step()
            if nextToken == nil { break }  // EOS token

            let text = engine?.tokenToString(nextToken!)
            output += text ?? ""
            token += 1

            // Real-time streaming output
            print("\(text ?? "")", terminator: "")
        }

        return output
    }
}
```

**Android example** (using Vulkan backend):

```kotlin
import mlc.llm.MLCEngine
import mlc.llm.MLCEngineConfig

class MLCInference(context: Context) {
    private val engine: MLCEngine

    init {
        val config = MLCEngineConfig().apply {
            modelPath = "file:///data/models/qwen-7b-int4"
            device = "gpu"  // Use Vulkan
            quantization = "int4"
        }
        engine = MLCEngine(config)
    }

    fun generate(prompt: String): Sequence<String> = sequence {
        engine.startGeneration(prompt)

        while (true) {
            val token = engine.step() ?: break  // null = EOS
            val text = engine.tokenToString(token)
            yield(text)
        }
    }
}
```

---

## 6. Streaming Generation and KV-Cache Management

### 6.1 Rolling Buffer for Limited DRAM

**Problem**: 4K-token context requires ~32 MB KV-cache (8× smaller than MHA, but still large for phones).

**Solution**: Rolling buffer (only keep recent N tokens in cache).

```c++
class RollingKVCache {
    vector<Tensor> k_cache, v_cache;
    int capacity;  // Max tokens to keep in cache
    int write_pos = 0;

public:
    RollingKVCache(int capacity) : capacity(capacity) {
        // Pre-allocate for max capacity
        for (int layer = 0; layer < num_layers; layer++) {
            k_cache.push_back(Tensor({capacity, num_heads, head_dim}));
            v_cache.push_back(Tensor({capacity, num_heads, head_dim}));
        }
    }

    void add_token(vector<Tensor>& new_k, vector<Tensor>& new_v) {
        for (int layer = 0; layer < num_layers; layer++) {
            // Overwrite oldest token if full
            int pos = write_pos % capacity;
            k_cache[layer].write_slice(pos, new_k[layer]);
            v_cache[layer].write_slice(pos, new_v[layer]);
        }
        write_pos++;
    }

    pair<vector<Tensor>, vector<Tensor>> get_cache() {
        // Return circular view of cache (most recent capacity tokens)
        int start = max(0, write_pos - capacity);
        return {
            k_cache.slice(start % capacity, write_pos),
            v_cache.slice(start % capacity, write_pos)
        };
    }
};

// Usage during generation
RollingKVCache kv_cache(/*capacity=*/ 512);  // Keep 512 most recent tokens

for (int step = 0; step < max_gen_steps; step++) {
    // Compute attention only on recent tokens
    auto [k_cached, v_cached] = kv_cache.get_cache();

    // Attention: Q @ K_cached.T → only 512 comparisons, not 4K
    output_logits = attention(q, k_cached, v_cached);

    // Sample next token
    next_token = sample(output_logits);

    // Add new KV to cache (overwrites oldest if necessary)
    auto [new_k, new_v] = compute_kv(next_token);
    kv_cache.add_token(new_k, new_v);
}
```

**Memory savings**:
```
Without rolling buffer (4K context):
  KV-cache: 2 × 512 tokens × 4 heads × 128 dim × 2 bytes = 1 MB per layer
  32 layers × 1 MB = 32 MB total

With rolling buffer (512 token window):
  KV-cache: 2 × 512 tokens × 4 heads × 128 dim × 2 bytes = 1 MB per layer
  32 layers × 1 MB = 32 MB (same, but only recent tokens)
  → Attention window: O(512) comparisons, not O(4K)
```

**Trade-off**: Loses long-range context beyond 512 tokens (acceptable for real-time chat).

### 6.2 Streaming Output (Token-by-Token)

**Key pattern**: Return tokens as they're generated (streaming) rather than waiting for full sequence.

```python
def generate_streaming(model, prompt: str, max_tokens: int):
    """
    Yield tokens one at a time (for real-time display).
    """
    input_ids = tokenize(prompt)
    kv_cache = {}

    for step in range(max_tokens):
        with torch.no_grad():
            logits, kv_cache = model.forward(
                input_ids[-1:],  # Feed only last token (use KV-cache)
                kv_cache=kv_cache
            )

        # Sample next token
        next_token_id = torch.argmax(logits[0, -1, :])
        next_token_str = tokenizer.decode([next_token_id])

        yield next_token_str  # Stream immediately

        # Append to input for next step
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)])

        # Rolling buffer (keep only recent tokens)
        if len(input_ids) > 512:
            input_ids = input_ids[-512:]

        if next_token_id == EOS_TOKEN_ID:
            break
```

**UI usage**:

```swift
// Real-time streaming in iOS
for token in mlc_engine.generate(prompt: userInput) {
    DispatchQueue.main.async {
        self.outputText.append(token)
        self.scrollToBottom()
    }
}
```

---

## 7. WebAssembly and Browser Deployment

### 7.1 WASM SIMD for LLM Inference

**WebAssembly** enables LLM inference in browser (Chrome, Firefox, Safari 17+).

**WASM compilation pipeline**:

```python
import mlc_llm
from tvm.target import WebAssemblyTarget

config = mlc_llm.CompilationConfig(
    model_name="phi-2",
    quantization="int4",
    target=WebAssemblyTarget(
        features=["simd"],  # WASM SIMD 128
        threads=4,  # WebWorker threads
    ),
)

mlc_llm.compile(model_quantized, config, output_dir="./wasm_build")
```

**WASM SIMD characteristics**:

```
WASM SIMD (128-bit):
├─ Int8x16 operations: 16 lanes × int8
├─ Int16x8: 8 lanes × int16
├─ Int32x4: 4 lanes × int32
└─ Float32x4: 4 lanes × float32

Example: INT8 MAC (multiply-accumulate)
  i8x16.dot_i16x8_s(a: int8x16, b: int8x16) → int16x8
  Performs 8 × (2 int8 multiplies + add) in single instruction
  Throughput: 8 MACs per cycle (on supporting CPU)
```

### 7.2 Browser Runtime

**JavaScript host code**:

```javascript
// Load compiled WASM module
async function initializeLLM() {
    const response = await fetch('./phi-2-int4.wasm');
    const wasmModule = await WebAssembly.instantiate(
        await response.arrayBuffer(),
        {
            env: {
                memory: new WebAssembly.Memory({ initial: 256, maximum: 512 }),
            }
        }
    );

    return new LLMEngine(wasmModule.instance);
}

class LLMEngine {
    constructor(wasmInstance) {
        this.wasm = wasmInstance;
    }

    async generate(prompt, maxTokens = 100) {
        const inputIds = this.tokenize(prompt);

        let output = "";
        for (let step = 0; step < maxTokens; step++) {
            // Call WASM forward pass
            const logits = this.wasm.exports.forward(
                new BigInt64Array(inputIds),
                inputIds.length
            );

            // Sample next token
            const nextTokenId = this.argmax(logits);
            output += this.decode([nextTokenId]);

            // Update input for next step
            inputIds.push(nextTokenId);

            // Streaming UI update
            document.getElementById('output').innerHTML = output;

            // Small delay to avoid blocking UI
            await new Promise(r => setTimeout(r, 10));

            if (nextTokenId === this.EOS_TOKEN_ID) break;
        }

        return output;
    }

    tokenize(text) {
        // Call WASM tokenizer
        const tokens = this.wasm.exports.tokenize(text);
        return tokens;
    }

    argmax(logits) {
        let maxIdx = 0, maxVal = logits[0];
        for (let i = 1; i < logits.length; i++) {
            if (logits[i] > maxVal) {
                maxVal = logits[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}

// Usage
(async () => {
    const engine = await initializeLLM();
    const result = await engine.generate("Tell me a joke");
    console.log(result);
})();
```

### 7.3 Performance on WASM

**Realistic benchmarks** (Chrome on M1 MacBook):

```
Phi-2 (2.7B) INT4:

First token (prefill, 100 input tokens):
  Latency: 2-3 seconds
  Reason: Full attention over 100 tokens, no KV-cache yet

Subsequent tokens (decode, 1 token input):
  Latency: 500-800 ms per token
  Reason: Only 1 query, use KV-cache for 100+ keys

Memory:
  Model weights: 1.4 MB (INT4)
  KV-cache (100 tokens): 3 MB
  Total in WASM memory: ~5 MB (fits in 64 MB limit)
```

---

## 8. Practical LLM Deployment: End-to-End Example

### 8.1 Model Selection and Optimization Pipeline

```python
import torch
import transformers
import mlc_llm
from torch.quantization import quantize_dynamic

# Step 1: Choose edge-friendly model
model_name = "Phi-2"  # 2.7B, designed for efficiency
model = transformers.AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16,
)

# Step 2: Quantization (INT4 for extreme compression)
quantization_config = {
    "quant_method": "gptq",
    "group_size": 128,
    "bits": 4,
    "dataset": "wikitext2",  # Calibration dataset
}

# Use GPTQ quantization library
from auto_gptq import AutoGPTQForCausalLM
model_quantized = AutoGPTQForCausalLM.from_quantized(
    "Phi-2-gptq",
    use_safetensors=True,
    device_map="auto",
    quantization_config=quantization_config,
)

# Step 3: Compile for different targets
targets = {
    "iphone": "metal",
    "android": "vulkan",
    "browser": "wasm",
    "server": "llvm",
}

for device_name, target in targets.items():
    config = mlc_llm.CompilationConfig(
        model_name="phi-2",
        target=target,
        quantization="int4",
        optimization_level=3,  # Aggressive optimization
    )
    mlc_llm.compile(model_quantized, config,
                   output_dir=f"./models/{device_name}")

# Step 4: Benchmark on target devices
results = {}
for device_name in targets:
    latency = benchmark_device(
        model_path=f"./models/{device_name}",
        prompt="The future of AI is",
        num_runs=10,
    )
    results[device_name] = latency

print("Latency per token (ms):")
for device, latency_ms in results.items():
    print(f"  {device}: {latency_ms:.1f} ms")
```

### 8.2 Mobile Deployment (iOS)

```swift
import MLCLLMCore
import UIKit

class ChatViewController: UIViewController, UITextFieldDelegate {
    @IBOutlet weak var inputField: UITextField!
    @IBOutlet weak var outputView: UITextView!

    var mlcEngine: MLCEngine?
    var isGenerating = false

    override func viewDidLoad() {
        super.viewDidLoad()
        setupMLCEngine()
    }

    func setupMLCEngine() {
        // Load compiled model (bundled in app)
        let modelURL = Bundle.main.url(forResource: "phi-2-int4",
                                      withExtension: "mlmodel")!

        let config = MLCEngineConfig(
            modelPath: modelURL.path,
            device: .gpu,  // Use Neural Engine (A17 Pro)
            quantization: "int4"
        )

        mlcEngine = MLCEngine(config: config)

        // Warm-up (pre-load model into memory)
        _ = mlcEngine?.generate(prompt: "Hello", maxTokens: 1)
    }

    @IBAction func sendButtonTapped(_ sender: UIButton) {
        guard !isGenerating, let prompt = inputField.text, !prompt.isEmpty else {
            return
        }

        isGenerating = true
        inputField.isEnabled = false

        // Generate in background thread
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self, let engine = self.mlcEngine else { return }

            var fullOutput = ""

            // Stream tokens as they're generated
            for token in engine.generate(
                prompt: prompt,
                maxTokens: 256,
                temperature: 0.7
            ) {
                fullOutput += token

                // Update UI on main thread
                DispatchQueue.main.async {
                    self.outputView.text = fullOutput
                    self.outputView.scrollToBottom()
                }
            }

            DispatchQueue.main.async {
                self.isGenerating = false
                self.inputField.isEnabled = true
            }
        }
    }

    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        sendButtonTapped(UIButton())
        return true
    }
}
```

### 8.3 Android Deployment

```kotlin
import mlc.llm.MLCEngine
import mlc.llm.MLCEngineConfig
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.*
import androidx.compose.runtime.*
import androidx.compose.ui.graphics.Color

@Composable
fun ChatScreen(viewModel: ChatViewModel) {
    val outputText by viewModel.outputText
    val inputText by viewModel.inputText
    val isGenerating by viewModel.isGenerating

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .background(Color.White)
    ) {
        // Output display
        Text(
            text = outputText,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .background(Color.LightGray)
                .padding(8.dp)
        )

        // Input and send
        Row(modifier = Modifier.fillMaxWidth()) {
            TextField(
                value = inputText,
                onValueChange = { viewModel.inputText.value = it },
                modifier = Modifier.weight(1f),
                enabled = !isGenerating,
            )
            Button(
                onClick = { viewModel.generateResponse() },
                enabled = !isGenerating && inputText.isNotEmpty(),
            ) {
                Text(if (isGenerating) "Generating..." else "Send")
            }
        }
    }
}

class ChatViewModel : ViewModel() {
    private val mlcEngine = MLCEngine(
        modelPath = "file:///data/models/phi-2-int4",
        device = "gpu",
    )

    val outputText = mutableStateOf("")
    val inputText = mutableStateOf("")
    val isGenerating = mutableStateOf(false)

    fun generateResponse() {
        isGenerating.value = true

        viewModelScope.launch(Dispatchers.Default) {
            var fullOutput = ""

            try {
                mlcEngine.startGeneration(inputText.value)

                while (true) {
                    val token = mlcEngine.step() ?: break

                    fullOutput += token

                    // Update UI
                    withContext(Dispatchers.Main) {
                        outputText.value = fullOutput
                    }
                }
            } finally {
                withContext(Dispatchers.Main) {
                    isGenerating.value = false
                }
            }
        }
    }
}
```

---

## 9. Performance Analysis and Benchmarking

### 9.1 Throughput vs. Latency

**Key metrics for LLM inference**:

```
Time to first token (TTFT):
  Time until first output token appears
  Measured in milliseconds
  Critical for user perception of "responsiveness"

Tokens per second (TPS):
  Throughput after initial token
  Inverse of latency per token
  Critical for "feel" of real-time conversation

Example: Phi-2 on A17 Pro
  TTFT: 100-150 ms (for 100-token input)
  TPS: 2-4 tokens/sec (4-5 ms per token)
  User experience: "Responsive enough for chat"
```

### 9.2 Memory Profiling

```python
import torch
import psutil

def profile_memory(model, batch_size=1, seq_length=1024):
    """
    Measure peak memory during inference.
    """
    # Forward pass
    input_ids = torch.randint(0, 32000, (batch_size, seq_length))

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1e6  # MB

    with torch.no_grad():
        outputs = model(input_ids)

    mem_after = process.memory_info().rss / 1e6
    peak_memory = mem_after - mem_before

    print(f"Peak memory for seq_len={seq_length}: {peak_memory:.1f} MB")

    # Breakdown: model weights + activations + KV-cache
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    kv_cache_size = (seq_length * model.config.hidden_size * 2 / 1e6)

    print(f"  Model: {model_size:.1f} MB")
    print(f"  KV-cache (est): {kv_cache_size:.1f} MB")
    print(f"  Other (activations, buffers): {peak_memory - model_size - kv_cache_size:.1f} MB")

profile_memory(model, seq_length=2048)
```

---

## 10. Conclusion and Future Directions

### 10.1 Feasibility Summary (2024)

| Device | Recommended Model | Expected Quality |
|--------|-------------------|-----------------|
| **Flagship Phone** (A17, Snapdragon 8) | 7B-13B INT4 | Good (70-80% of cloud) |
| **Mid-range Phone** (Snapdragon 7) | 3B-7B INT4 | Acceptable (60-70%) |
| **Tablet** (iPad Pro) | 13B-20B INT4 | Very Good (80-90%) |
| **Watch/Wearable** | 500M-1B INT4 | Basic (50-60%) |
| **Browser (WASM)** | 1B-3B INT4 | Acceptable (60-70%) |

### 10.2 Emerging Techniques

1. **Speculative Decoding**: Use small model to predict next tokens, verify with large model
2. **Continued Pre-training**: Further train models on domain-specific data (edge deployment improves over time)
3. **LoRA Fine-tuning**: Low-rank adaptation for personalization without retraining
4. **MoE (Mixture of Experts)**: Conditional computation (activate 2-4 experts per token, not all)

### 10.3 Key Takeaways

- **KV-cache is the real bottleneck** (not compute), optimize memory footprint first
- **Quantization (INT4) mandatory** for any model >1B on phones
- **GQA/MQA essential** for reducing KV-cache, 8× reduction achievable
- **Architecture matters**: Phi, Gemma, Qwen designed for edge; Llama less so
- **Rolling buffer/windowing** enables 4K+ context without memory blow-up
- **Streaming output** critical for UX (token-by-token vs. batch)
- **Benchmark on real hardware**: Simulation ≠ reality (thermal, memory pressure)

### 10.4 Research Frontiers

1. **Sub-100ms first-token latency**: Currently 100-300 ms, need better prefill optimization
2. **Long-context efficient inference**: Beyond 4K tokens without quadratic memory
3. **Federated fine-tuning**: Training on-device for personalization
4. **Hardware-software co-design**: Next-gen NPUs optimized for transformer ops

---

## Further Reading

- Lin et al.: "Scalable Transformer based Models for Language Understanding" (MLSys 2024)
- Xiao et al.: "Efficient Streaming Language Models with Attention Sinks" (ICLR 2024)
- Apple: "Introducing On-Device LLM Inference on iPhones" (WWDC 2024)
- Alibaba: "Qwen Technical Report" (2023)
- Microsoft: "Phi-2: The Surprising Power of Small Language Models" (2023)
- Apache TVM: "MLC LLM: A Compiled, Language Model Inference Engine" (2023)
- WebAssembly: "SIMD Support in WebAssembly" (W3C Specification)

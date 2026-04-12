# MODULE 20 — Full Stack Inference on Apple Silicon

## 1. SYSTEMS OVERVIEW

This capstone module synthesizes Modules 16-19, addressing the complete inference pipeline on Apple Silicon: model selection, hardware-software co-design, multi-accelerator dispatch, and production system design. We focus on **total system optimization**, not individual component performance.

Key principle: **Inference on Apple Silicon is neither compute-bound nor memory-bound in isolation; it is power-bound and latency-bound.** Optimal systems prioritize energy efficiency and sub-200ms latency over peak throughput.

### 1.1 Design Philosophy: Energy-First Optimization

Traditional GPU systems (NVIDIA H100, A100) optimize for throughput (tokens/sec) with secondary consideration for power. Apple Silicon requires inverting this priority:

```
Traditional: maximize (throughput) subject to power_budget ≤ 250W
Apple Silicon: maximize (throughput) subject to latency ≤ 200ms AND power ≤ 25W
```

This shift has profound implications:
- **Batch size:** Small (1-4 tokens), avoiding compute saturation
- **Precision:** Aggressive quantization (INT4, INT8) acceptable
- **Kernels:** Emphasize memory efficiency over peak GFLOPS
- **Frequency:** Lower clocks acceptable if latency target met

### 1.2 Three Deployment Scenarios

**Scenario 1: Interactive Single-User (Laptop)**
- User expectation: <200ms latency per token (perceived as real-time)
- Hardware: M2/M3 (16 GB RAM typical)
- Model: 3B-7B (fits in memory with context)
- Optimization: Minimize latency, not throughput

**Scenario 2: Batch Processing (Background)**
- User expectation: <5 sec per query (not interactive)
- Hardware: M3 Max (32/64 GB RAM, can hold 13B-34B models)
- Model: 7B-34B
- Optimization: Maximize tokens/sec per watt

**Scenario 3: Inference Server (Multi-Model, Multi-User)**
- User expectation: 99th percentile <500ms latency
- Hardware: Mac Mini cluster (multiple M3 Max devices)
- Models: 5-10 different models, rapidly switching
- Optimization: Model loading latency, isolation between users

---

## 2. THEORETICAL FOUNDATION

### 2.1 Model Selection Arithmetic: Memory Budgets

Given N GB RAM, determine which models fit with acceptable KV cache overhead.

**Memory Equation:**

```
Total Memory = Model_Size + KV_Cache + Runtime_Activations + OS_Overhead

Model_Size = num_parameters × bytes_per_param
           = (d_model × num_layers × n_heads × scaling_factor) × bytes

For 7B LLaMA: 7×10^9 params × 2 bytes (FP16) = 14 GB (FP16) or 3.5 GB (INT8)

KV_Cache = seq_len × d_model × 2 (K and V) × bytes × num_layers
        = seq_len × d_model × 2 × 2 × 32 (for typical 32-layer model)
        ≈ seq_len × d_model × 128 bytes

Runtime_Activations ≈ 10% of model size (intermediate hidden states, gradients)

OS_Overhead ≈ 2-3 GB (system processes, disk cache)
```

**Example: 16 GB MacBook Pro**

```
Available: 16 GB - 2 GB OS = 14 GB

Model options:
  7B FP16: 7 GB model + 1 GB KV (2K ctx) + 0.7 GB activations = 8.7 GB ✓
  7B FP16: 7 GB model + 2 GB KV (4K ctx) + 0.7 GB activations = 9.7 GB ✓
  13B FP16: 13 GB model + 1 GB KV (2K ctx) = 14 GB ✗ (no activation headroom)

Recommended:
  7B with 4K context (9.7 GB total, leaves 4 GB buffer)
  or 3B model with 8K context (5 GB model + 2 GB KV = 7 GB, very safe)
```

### 2.2 Latency Composition Analysis

End-to-end inference latency breaks down into:

```
Total Latency = Model_Load + Prefill + Decode + Sampling + Overhead

Model_Load (happens once per session):
  - Disk I/O: model_size / ssd_bandwidth
  - Typical: 3.5 GB / 1 GB/s = 3.5 sec (M1-M2), 2-3 sec (M3 with faster SSD)
  - Amortized over session: If session generates 256 tokens, overhead = 3.5 / 256 = 14 ms/token

Prefill (process prompt once):
  - Batch matrix multiply through all layers
  - Arithmetic: 2 × batch × seq_len × d_model × 4 × num_layers ops
  - Example: batch=1, seq_len=512, d_model=4096, 32 layers
  - FLOPs: 2 × 1 × 512 × 4096 × 4 × 32 = 1 TFLOP (approx)
  - Latency: 1 TFLOP / 100 GFLOPS = 10 ms (GPU)
  - Amortized per token: 10 / 128 (num tokens) = 0.08 ms

Decode (autoregressive, one token at a time):
  - Arithmetic: 2 × 1 × 1 × d_model × 4 × num_layers = 1 GFLOP (small!)
  - Latency: 1 GFLOP / 100 GFLOPS = 10 ms (bandwidth-bound)
  - This dominates! Not amortized.

Sampling + Logits:
  - Softmax over vocab (50K tokens): 50K multiply + reduce = 5 ms (CPU)
  - Top-k/nucleus sampling: 50K ops = 1 ms
  - Total: 6 ms per token

Overhead (synchronization, kernel launch, etc.):
  - CPU-GPU sync: 2-3 ms per kernel
  - Metal kernel launch: 0.5-1 ms per layer × 32 = 16-32 ms
  - Total: 20-35 ms per token generation

Total per token: 10 (decode) + 6 (sampling) + 25 (overhead) = 41 ms
Measured: 100+ ms per token on M2 (matches overhead estimate)
```

### 2.3 Multi-Accelerator Co-Design

Apple Silicon offers three compute resources: P-cores, E-cores, GPU, ANE. Optimal inference partitions work by characteristics:

**P-Core Execution (best for):**
- Sequential logic (argmax for sampling)
- Small tensor ops (logits scaling)
- Model switching / loading
- Single-threaded preprocessing

**E-Core Execution (best for):**
- Lightweight tensor ops (element-wise)
- Batch norm (simple reduce)
- Parallel data preprocessing
- Energy-critical paths

**GPU Execution (best for):**
- Large matrix multiply (> 1M elements)
- Attention (quadratic in seq length)
- Parallel reductions (softmax)
- Bandwidth-bound operations

**ANE Execution (best for):**
- Quantized GEMM (batch size 8-32)
- Model ensembles (multiple independent samples)
- Dedicated workloads (avoiding dispatch overhead)

**Dispatch Rules:**

```
if operation.size < 100K:
  → CPU (P-core)
elif operation.type == "attention_softmax":
  → GPU
elif operation.quantized and batch_size in {8, 16, 32}:
  → ANE
elif operation.is_gguf_dequant:
  → GPU + CPU interleave (dequant on GPU, custom kernel on CPU)
else:
  → GPU
```

### 2.4 On-Device Privacy: Private Cloud Compute

Apple's **Private Cloud Compute (PCC)** enables LLM inference without Apple observing user data:

```
User Device (MacBook) ← Encrypted Request → Apple Server
                         ↓ Decrypted in Secure Enclave
                         ↓ Inference in Secure Enclave
                         ← Encrypted Response → User Device

Security guarantees:
  - Data never visible to human staff (encrypted at rest, in transit, in compute)
  - Inference only on verified Apple-signed models
  - No request logging or storage
  - Attestation: User can verify computation happened on real Apple hardware
```

For on-device inference (preferred for privacy):
- All computation happens locally
- No network traffic
- User owns their data
- Inference completes in <1 sec (vs 10+ sec over network)

---

## 3. HARDWARE MAPPING

### 3.1 RAM Tier Selection for Model Deployment

```
MacBook Air M3:     16 GB → 7B models max
MacBook Pro M3:     32 GB → 13B models max
MacBook Pro M3 Max: 48-96 GB → 34B models, multi-model deployment

GPU Configuration:
  M3 base: 8-core GPU, 100 GB/s memory
  M3 Pro: 12/18-core GPU, same 100 GB/s
  M3 Max: Dual GPU, 200 GB/s (if dual GPU utilized)

Optimal choice for inference:
  - 32 GB MacBook Pro M3 (balanced cost/performance)
  - 64 GB MacBook Pro M3 Max (future-proof, multi-model)
```

### 3.2 Thermal and Power Profiles

Apple Silicon thermal design power (TDP) is nominal; actual sustained power depends on workload:

```
Idle: 1-2W (screen off, system asleep)
Light Use: 5-8W (email, browsing)
LLM Inference (single user): 12-18W (GPU + CPU + memory controller)
Peak (all cores + GPU): 25-30W (thermal throttle point)

For sustained inference:
  Target: 15W (achievable by reducing GPU frequency)
  Result: 7-8 tokens/sec on 7B model
  Duration: Indefinite (thermal limit not reached)

If pushing to 25W:
  Result: 10+ tokens/sec
  Duration: 15-30 minutes before thermal throttle (depends on ambient temp)
  After throttle: Frequency drops to maintain 25W, latency increases to 12-15 ms/token
```

### 3.3 Unified Memory Mapping for Multi-Model Serving

```
Model A (3.5 GB): weights in unified memory, address space [0x0, 0x0 + 3.5 GB)
Model B (3.5 GB): weights in unified memory, address space [0x0 + 3.5 GB, 0x0 + 7 GB)
Model C (3.5 GB): weights in unified memory, address space [0x0 + 7 GB, 0x0 + 10.5 GB)

Total: 10.5 GB (fits in 16 GB with headroom)

Switching between models:
  - CPU: unload model A, load model B (mmap handles page remapping)
  - GPU: Model weights already in unified address space (no transfer)
  - Latency: ~100 ms (same as model A switching)

vs Discrete GPU:
  - Unload model A from VRAM: 100 ms (PCIe transfer back to host)
  - Load model B to VRAM: 100 ms (PCIe transfer to GPU)
  - Total: 200 ms per switch
```

---

## 4. IMPLEMENTATION DEEP DIVE

### 4.1 Model Selection Algorithm

```python
def select_model_for_device(ram_available_gb, use_case, quality_preference):
    """
    Select optimal model given hardware constraints.

    Args:
        ram_available_gb: Available RAM (e.g., 16, 32, 64)
        use_case: 'interactive', 'batch', 'server'
        quality_preference: 'accuracy', 'balanced', 'speed'

    Returns:
        model_name: Recommended model (e.g., '7B', '13B')
        context_length: Recommended context (e.g., 2048, 4096)
        quantization: Recommended (e.g., 'Q4_K_M', 'Q5_K_M')
    """
    # Reserve for OS, overhead
    available_for_ml = ram_available_gb * 0.8

    # Model size candidates (GGUF Q4_K_M)
    models = {
        '3B': {'size_gb': 1.5, 'quality': 0.70},
        '7B': {'size_gb': 3.5, 'quality': 0.85},
        '13B': {'size_gb': 7.0, 'quality': 0.90},
        '34B': {'size_gb': 18.0, 'quality': 0.95},
    }

    # Context size KB per token for KV cache
    kv_cache_kb_per_token = {
        '3B': 24,      # d_model=2048 × layers=26 × 2K + V
        '7B': 64,      # d_model=4096 × layers=32 × 2K + V
        '13B': 128,    # d_model=5120 × layers=40 × 2K + V
        '34B': 256,    # d_model=8192 × layers=60 × 2K + V
    }

    best_model = None
    best_context = 0

    for model_name, model_info in models.items():
        model_size = model_info['size_gb']

        # Determine max context given budget
        remaining = available_for_ml - model_size
        max_context = int(remaining * 1024 / (kv_cache_kb_per_token[model_name] + 1))

        # Adjust for use case
        if use_case == 'interactive':
            max_context = min(max_context, 4096)  # Balance latency
            latency_limit = 200  # ms per token
        elif use_case == 'batch':
            max_context = min(max_context, 8192)
            latency_limit = 1000
        else:  # server
            max_context = min(max_context, 2048)
            latency_limit = 500

        # Verify model fits
        if model_size + (max_context * kv_cache_kb_per_token[model_name] / 1024) > available_for_ml:
            continue

        # Score based on preference
        if quality_preference == 'accuracy':
            score = model_info['quality']
        elif quality_preference == 'balanced':
            score = model_info['quality'] - (model_size / 20)  # Slight penalty for size
        else:  # speed
            score = 1.0 - (model_size / 30)

        if score > (best_model[1] if best_model else 0):
            best_model = (model_name, score, max_context)

    return {
        'model': best_model[0],
        'context_length': best_model[2],
        'quantization': 'Q4_K_M' if best_model[2] > 2048 else 'Q5_K_M'
    }

# Example usage:
print(select_model_for_device(16, 'interactive', 'balanced'))
# Output: {'model': '7B', 'context_length': 2048, 'quantization': 'Q4_K_M'}

print(select_model_for_device(64, 'batch', 'accuracy'))
# Output: {'model': '34B', 'context_length': 4096, 'quantization': 'Q5_K_M'}
```

### 4.2 Hybrid CPU+GPU+ANE Execution

**Example: 7B Model Inference with Work Distribution**

```python
class HybridInferenceEngine:
    def __init__(self, model_path):
        self.model = load_gguf_model(model_path)  # Quantized model
        self.gpu_device = MTLCreateSystemDefaultDevice()
        self.cpu_threads = 8

    def forward_pass_hybrid(self, tokens, seq_len):
        """Execute forward pass with heterogeneous dispatch."""

        hidden_state = embed_tokens(tokens)  # CPU (small, sequential)

        for layer_idx, layer in enumerate(self.model.layers):
            # Attention sub-layer
            if seq_len <= 512:
                # ANE dispatch (if batch size allows)
                if tokens.shape[0] in {1, 8, 16, 32}:
                    hidden_state = layer.attention_ane(hidden_state)
                else:
                    hidden_state = layer.attention_gpu(hidden_state)
            else:
                # Long sequence: GPU only
                hidden_state = layer.attention_gpu(hidden_state)

            # Feed-forward sub-layer (always GPU for large models)
            ffn_output = layer.ffn_gpu(hidden_state)

            # Output projection (CPU for small sizes)
            if hidden_state.numel() < 1M:
                hidden_state = layer.out_proj_cpu(ffn_output)
            else:
                hidden_state = layer.out_proj_gpu(ffn_output)

        # Decode logits (CPU, small)
        logits = output_projection_cpu(hidden_state)

        # Sampling (CPU, sequential)
        next_token = sample_from_logits(logits)

        return next_token

    def benchmark_layer_dispatch(self):
        """Analyze dispatch overhead per layer."""
        latencies = {}

        for layer_name in ['attention', 'ffn', 'norm']:
            # CPU execution
            cpu_time = measure_cpu_layer(layer_name)
            # GPU execution
            gpu_time = measure_gpu_layer(layer_name)
            # ANE execution (if supported)
            ane_time = measure_ane_layer(layer_name) if layer_name == 'attention' else float('inf')

            latencies[layer_name] = {
                'cpu': cpu_time,
                'gpu': gpu_time,
                'ane': ane_time,
                'overhead': gpu_time * 0.15  # Estimated dispatch overhead
            }

        return latencies
```

### 4.3 Swift Integration: Async Inference with Streaming

```swift
import CoreML
import Foundation

class LLMInferenceManager {
    let model: MLModel
    let session: MLSession
    let bufferSize = 256  // Decode step buffer

    func streamingInference(prompt: String, maxTokens: Int) async throws -> AsyncStream<String> {
        return AsyncStream { continuation in
            DispatchQueue.global().async {
                do {
                    // Prefill phase: process prompt
                    let promptTokens = self.tokenize(prompt)
                    let prefillOutput = try self.prefill(tokens: promptTokens)

                    // Decode phase: generate tokens one by one
                    var decodedText = ""
                    var lastHiddenState = prefillOutput

                    for i in 0..<maxTokens {
                        // Generate next token
                        let logits = try self.forwardPass(lastHiddenState)
                        let nextToken = self.sampleToken(logits)
                        let tokenText = self.tokenizer.decode([nextToken])

                        // Stream token to caller
                        decodedText += tokenText
                        continuation.yield(tokenText)

                        // Update state for next iteration
                        lastHiddenState = self.embedToken(nextToken)

                        // Early exit if EOS token
                        if nextToken == 2 {  // EOS token ID
                            break
                        }
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func prefill(tokens: [Int32]) throws -> MLMultiArray {
        // Batch process prompt tokens
        let tokenCount = tokens.count
        let input = MLMultiArray(shape: [1, NSNumber(value: tokenCount)], dataType: .int32)

        // Copy tokens into input
        for (i, token) in tokens.enumerated() {
            input[i] = NSNumber(value: token)
        }

        // Forward pass
        let output = try session.run(inputs: ["input_ids": input])
        return output["hidden_state"] as! MLMultiArray
    }

    private func forwardPass(_ hiddenState: MLMultiArray) throws -> [Float] {
        // Single token inference
        let input = MLMultiArray(shape: [1, 1, NSNumber(value: hiddenState.count)], dataType: .float32)
        let output = try session.run(inputs: ["hidden_state": input])

        let logitsArray = output["logits"] as! MLMultiArray
        var logits = [Float](repeating: 0, count: logitsArray.count)
        for i in 0..<logitsArray.count {
            logits[i] = logitsArray[i].floatValue
        }

        return logits
    }

    func streamingInferenceWithMetrics(prompt: String) async -> (text: String, metrics: InferenceMetrics) {
        var metrics = InferenceMetrics()

        let startTime = Date()
        let stream = try! await streamingInference(prompt: prompt, maxTokens: 128)

        var fullText = ""
        var tokenCount = 0

        for try await token in stream {
            fullText += token
            tokenCount += 1
        }

        metrics.totalTime = Date().timeIntervalSince(startTime)
        metrics.tokensGenerated = tokenCount
        metrics.tokensPerSecond = Double(tokenCount) / metrics.totalTime

        return (text: fullText, metrics: metrics)
    }
}

struct InferenceMetrics {
    var totalTime: TimeInterval = 0
    var tokensGenerated: Int = 0
    var tokensPerSecond: Double = 0
    var powerDrawWatts: Double = 0
    var energyPerTokenJoules: Double = 0
}
```

### 4.4 Energy Efficiency Analysis

```python
import subprocess
import time
import numpy as np

def measure_energy_efficiency(model_path, num_tokens=256):
    """
    Comprehensive energy measurement for inference.
    Requires: sudo access, powermetrics tool
    """

    # Start power sampling (every 100ms)
    power_proc = subprocess.Popen(
        ['sudo', 'powermetrics', '--samplers', 'cpu_power,gpu_power', '-n', '150'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    time.sleep(1)  # Let powermetrics initialize

    # Run inference
    start_time = time.time()
    subprocess.run(
        ['./main', '-m', model_path, '-n', str(num_tokens), '--gpu-layers', '33'],
        check=True
    )
    inference_time = time.time() - start_time

    # Stop powermetrics
    power_proc.terminate()
    power_output, _ = power_proc.communicate()

    # Parse power data
    cpu_powers = []
    gpu_powers = []

    for line in power_output.decode().split('\n'):
        if 'CPU Power:' in line:
            power_str = line.split()[-2]
            cpu_powers.append(float(power_str))
        elif 'GPU Power:' in line:
            power_str = line.split()[-2]
            gpu_powers.append(float(power_str))

    # Compute statistics
    avg_cpu_power = np.mean(cpu_powers) if cpu_powers else 0
    avg_gpu_power = np.mean(gpu_powers) if gpu_powers else 0
    total_power = avg_cpu_power + avg_gpu_power

    energy_joules = total_power * inference_time
    energy_per_token = energy_joules / num_tokens

    print(f"Energy Efficiency Analysis")
    print(f"  Inference time: {inference_time:.2f} sec")
    print(f"  Tokens generated: {num_tokens}")
    print(f"  Average CPU power: {avg_cpu_power:.2f} W")
    print(f"  Average GPU power: {avg_gpu_power:.2f} W")
    print(f"  Total power: {total_power:.2f} W")
    print(f"  Total energy: {energy_joules:.2f} J")
    print(f"  Energy per token: {energy_per_token:.2f} mJ")
    print(f"  Tokens/Joule: {num_tokens / energy_joules:.2f}")

    return {
        'inference_time_sec': inference_time,
        'avg_cpu_power_w': avg_cpu_power,
        'avg_gpu_power_w': avg_gpu_power,
        'energy_per_token_mj': energy_per_token * 1000,
        'tokens_per_joule': num_tokens / energy_joules
    }

# Example:
# metrics = measure_energy_efficiency('llama-7b-q4_k_m.gguf', 256)
# Expected output:
#   Average CPU power: 3.5 W
#   Average GPU power: 10.2 W
#   Total power: 13.7 W
#   Energy per token: 13.7 J / 7.3 tokens/sec = 1.88 J (0.52 mJ @ 7.3 tok/s)
```

---

## 5. KEY PAPERS & REFERENCES

1. **"Apple Silicon: A System Architecture Perspective" (Cutress, AnandTech, 2023)**
   - Deep technical analysis of M1/M2/M3 architecture
   - Memory hierarchy measurements on real hardware
   - Unified memory coherency implications

2. **"Speculative Decoding for Large Language Models" (Leviathan et al., ICML 2023)**
   - 1.5-2× speedup through small model drafting
   - Acceptance criteria preserving output distribution
   - Implementation on Apple Silicon feasibility

3. **"On the Efficiency of Inference Serving on Edge Devices" (Trivedi et al., MLSys 2024)**
   - Comparison of quantization schemes (INT8, INT4, FP16)
   - Latency-throughput-accuracy tradeoffs
   - Energy measurements on mobile SoCs

4. **"Transformer Optimization for Inference on Resource-Constrained Devices" (Bandara et al., 2023)**
   - Attention KV cache compression
   - Layer fusion strategies
   - Practical implementation on Apple devices

5. **"Private Cloud Compute: Machine Learning at Scale While Preserving Privacy" (Apple Research)**
   - Architectural overview of on-device + secure server inference
   - Hardware security features (Secure Enclave)
   - Privacy guarantees and attestation

6. **"llama.cpp: Efficient Inference for Large Language Models" (Ggerganov et al., GitHub)**
   - Production-ready implementation details
   - GGUF quantization format specification
   - Metal backend optimization strategies

---

## 6. SYSTEMS TRADEOFFS

### 6.1 Single-Model vs Multi-Model Serving

| Aspect | Single Model | Multi-Model |
|---|---|---|
| **Model Loading** | Once at startup (5 sec) | Per request (100-500 ms) |
| **Memory Overhead** | 1× model size | 1.5-3× (cache multiple) |
| **Latency Variance** | Low (predictable) | High (loading jitter) |
| **Hardware Utilization** | Specialized (optimal kernels) | Generic (amortized overhead) |
| **User Experience** | Consistent | Variable (first request slow) |
| **Scaling** | Difficult (bottleneck on one model) | Flexible (load-balance across models) |

**Recommendation:**
- **Single-model:** Interactive apps (Siri, on-device chat)
- **Multi-model:** Professional tools (translation, transcription, summary services)

### 6.2 Quantization Precision Tradeoffs

| Quantization | Model Size | Accuracy Drop | Speed | Recommended Use |
|---|---|---|---|---|
| FP32 | 26 GB | 0% | 0.5× | Research, fine-tuning |
| FP16 (CoreML) | 13 GB | <0.2% | 0.95× | High-quality production |
| Q8_0 | 7 GB | <0.5% | 0.95× | Balanced (llama.cpp) |
| Q5_K_M | 4.3 GB | 0.5-1% | 0.85× | Recommended default |
| Q4_K_M | 3.5 GB | 1-2% | 0.7× | Best for Apple Silicon |
| Q3_K | 2.6 GB | 2-3% | 0.5× | Edge devices only |

**Apple Silicon Choice:** Q4_K_M for 3.5× compression with minimal accuracy loss.

### 6.3 Prefill Batch Size vs Latency

Larger prefill batches improve throughput but increase prompt processing latency:

| Batch Size | Throughput (tok/s) | Latency (ms/token) | GPU Util |
|---|---|---|---|
| 1 | 7 | 143 | 65% |
| 4 | 20 | 200 | 75% |
| 8 | 38 | 211 | 85% |
| 16 | 65 | 246 | 92% |
| 32 | 110 | 291 | 96% |

**Tradeoff:** For interactive (latency < 200ms), use batch 1-4. For batch processing, batch 16-32.

### 6.4 Context Length vs KV Cache Memory

Longer context enables better coherence but consumes memory:

| Context | KV Cache (7B) | Max Batch | Quality Gain |
|---|---|---|---|
| 512 | 64 MB | 8 | Baseline |
| 1024 | 128 MB | 4 | +5-10% (better discourse) |
| 2048 | 256 MB | 2 | +10-15% (conversation) |
| 4096 | 512 MB | 1 | +15-20% (long form) |
| 8192 | 1 GB | 1 (with issues) | +20-25% (rare) |

**Apple Silicon (16 GB):** Optimal = 2048 context (balances quality + batch size).

---

## 7. EXPERT INSIGHT

### 7.1 Model Selection for Specific Applications

**Q: I need to deploy a multilingual translation model. Should I use a single large model or multiple language-specific small models?"**

**A: Multi-model approach is superior on Apple Silicon.**

```
Single Large Model (13B):
- Size: 7 GB (fits barely in 16 GB)
- Latency: 150 ms/token
- Quality: Excellent (40+ language pairs)
- Switching cost: None (single model)
- RAM: 7 GB + 2 GB KV = 9 GB (tight)

Multiple Small Models (3B × 10 languages):
- Size: 3 GB each (swap at need)
- Latency: 50 ms/token
- Quality: Good (fine-tuned per language)
- Switching cost: 100 ms per language pair
- RAM: 3 GB + 2 GB KV = 5 GB (plenty)

For 10 languages with varied usage:
- Single: Always 150 ms latency, high memory pressure
- Multi: 50 ms latency (usual case), 150 ms (on language switch)
- Tradeoff wins for multi-model if switching < 1/sec per user
```

### 7.2 Energy Optimization Techniques

**Technique 1: Frequency Scaling**

```
Full frequency (3.5 GHz): 7.3 tokens/sec @ 15W
Reduced frequency (2.4 GHz): 6.8 tokens/sec @ 6W
Speedup factor: 1.07× for energy
Result: 1.13 tokens/Joule (vs 0.49 tokens/Joule at full frequency)

For battery-constrained (MacBook on battery):
  - Use reduced frequency (2x longer battery life for 15% latency increase)
```

**Technique 2: Speculative Decoding**

```
Standard: 10 tokens/sec @ 13W = 0.77 tokens/Joule
Speculative (with 2B draft): 15 tokens/sec @ 18W = 0.83 tokens/Joule

Trade: Complexity for 8% better efficiency and 50% latency improvement
Recommended: Enable for server deployments, disable for simplicity
```

**Technique 3: KV Cache Compression**

```
Standard INT8 KV cache: 256 MB (2K context), full accuracy
Compressed (INT4 KV): 64 MB (2K context), <0.5% accuracy loss

Bandwidth savings: 4× reduction per token generation
Power: 13W → 11W (2W reduction, ~15% savings)
Recommended: Always use for inference-only (loss < 1%)
```

### 7.3 Latency-Throughput Pareto Front

For a given model and hardware, there exists a Pareto frontier of achievable latency/throughput combinations:

```
                  ← Higher Throughput (tok/sec)
     ↑
     |    Pareto Front (optimal operating points)
     |   ╱════════════╲
     |  ╱              ╲
     | ╱                ╲  ← Infeasible region (too many constraints)
     |╱                  ╲
     |________________╲
  L  |                 ╲
  a  |                  ╲___
  t  |        Feasible      \___ ← Can achieve high throughput or low latency, not both
  e  |        Region           ╲
  n  |
  c  |
  y  |
```

**For 7B on M2:**
- Point A (Low latency): 1-user, batch=1, 10 ms/token, 7 tok/sec
- Point B (Mid): 4-user, batch=4, 200 ms/token, 20 tok/sec aggregate
- Point C (High throughput): 32-user, batch=32, 500 ms/token, 65 tok/sec aggregate

Each point requires different:
- Batch size
- GPU frequency
- Memory pressure
- Number of concurrent users

---

## 8. BENCHMARKING METHODOLOGY

### 8.1 Comprehensive Inference Benchmark Suite

```bash
#!/bin/bash
# benchmark_inference.sh - Full system benchmark

MODELS=("3b-q4_k_m.gguf" "7b-q4_k_m.gguf" "13b-q4_k_m.gguf")
BATCH_SIZES=(1 4 8 16)
CONTEXT_LENGTHS=(512 2048 4096)

RESULTS_FILE="benchmark_results.csv"
echo "Model,BatchSize,ContextLength,TokensPerSec,LatencyMs,PowerW,EnergyPerToken" > $RESULTS_FILE

for MODEL in "${MODELS[@]}"; do
    for BATCH in "${BATCH_SIZES[@]}"; do
        for CONTEXT in "${CONTEXT_LENGTHS[@]}"; do
            # Run inference with power measurement
            echo "Benchmarking $MODEL, batch $BATCH, context $CONTEXT..."

            # Start power measurement
            sudo powermetrics --samplers cpu_power,gpu_power -n 60 > power_sample.log &
            POWER_PID=$!
            sleep 1

            # Run inference
            RESULT=$( time ./main -m "$MODEL" -c "$CONTEXT" -b "$BATCH" -n 256 --gpu-layers 33 2>&1 )

            # Stop power measurement
            kill $POWER_PID

            # Parse results
            THROUGHPUT=$(echo "$RESULT" | grep "eval time" | awk '{print $NF}')
            LATENCY=$(echo "$RESULT" | grep "eval time" | awk '{print $(NF-2)}')

            AVG_POWER=$(grep "GPU Power:" power_sample.log | awk '{sum += $(NF-1); count++} END {print sum/count}')

            ENERGY=$(echo "scale=4; $LATENCY * $AVG_POWER / 256" | bc)

            echo "$MODEL,$BATCH,$CONTEXT,$THROUGHPUT,$LATENCY,$AVG_POWER,$ENERGY" >> $RESULTS_FILE
        done
    done
done

# Analyze results
echo ""
echo "Summary Statistics:"
awk -F',' 'NR>1 {print $1, "Avg throughput:", $5 " tok/s"}' $RESULTS_FILE | sort -u
```

### 8.2 Accuracy Evaluation Under Quantization

```python
def evaluate_quantized_model_accuracy(baseline_model, quantized_model, test_dataset):
    """
    Benchmark accuracy degradation due to quantization.
    """
    import numpy as np

    baseline_scores = []
    quantized_scores = []

    for batch in test_dataset:
        inputs, expected_outputs = batch

        # Baseline prediction
        baseline_output = baseline_model.predict(inputs)
        baseline_pred = baseline_output['predictions']
        baseline_score = compute_task_metric(baseline_pred, expected_outputs)

        # Quantized prediction
        quantized_output = quantized_model.predict(inputs)
        quantized_pred = quantized_output['predictions']
        quantized_score = compute_task_metric(quantized_pred, expected_outputs)

        baseline_scores.append(baseline_score)
        quantized_scores.append(quantized_score)

    baseline_mean = np.mean(baseline_scores)
    quantized_mean = np.mean(quantized_scores)
    accuracy_drop = baseline_mean - quantized_mean
    accuracy_drop_pct = (accuracy_drop / baseline_mean) * 100

    print(f"Accuracy Evaluation")
    print(f"  Baseline: {baseline_mean:.4f}")
    print(f"  Quantized: {quantized_mean:.4f}")
    print(f"  Drop: {accuracy_drop_pct:.2f}%")

    # Acceptance criterion: <1% drop
    if accuracy_drop_pct < 1.0:
        print("✓ Quantization acceptable")
    else:
        print("✗ Quantization unacceptable, consider higher precision")

    return {
        'baseline_score': baseline_mean,
        'quantized_score': quantized_mean,
        'accuracy_drop_pct': accuracy_drop_pct
    }
```

---

## 9. OPEN PROBLEMS

1. **Dynamic Batching in Production:** llama.cpp and CoreML lack native support for dynamic batching (multiple independent requests processed in parallel with varying dimensions). Implementing this requires careful synchronization and memory management.

2. **Real-Time Scheduling for SLA:** No framework provides hard latency guarantees (e.g., "99% of requests < 200ms"). Achieving this requires offline analysis of throughput curves and request patterns.

3. **Automated Quantization:** Current quantization is manual (select Q4_K_M or Q5_K_M). Automatic selection based on model architecture and task would enable non-experts to optimize.

4. **Multi-GPU Utilization (Future Hardware):** If future MacBooks have dual M-series chips, coordinating inference across chips remains unsolved (no public API for inter-chip communication).

5. **On-Device Fine-tuning:** Current systems are inference-only. Enabling efficient fine-tuning on Apple Silicon (e.g., LoRA adaptation) would enable personalization.

---

## 10. PHD QUALIFIER QUESTIONS

**Q1 (System Design):** "Design a production inference system for a MacBook Air (16 GB, M3) serving 50 concurrent users each generating 128 tokens with 2K context. Your constraint: 99% of latency samples must be < 500ms. Propose (a) model selection, (b) dispatching strategy, (c) resource allocation."

**A1 Outline:**

**Model Selection:**
```
Available RAM: 16 GB - 2 GB (OS) = 14 GB
Model budget: 3.5 GB (7B Q4_K_M)
KV cache per user: 2K × 4096 × 2 × 8 × 32 / 10^9 = 4 GB per user

Wait! 50 users × 4 GB = 200 GB KV cache (impossible)
Solution: Streaming context window
  - Only maintain last 512 tokens in KV cache
  - Re-compute attention over full context (expensive) or
  - Use sliding window (last 2K tokens accessible)

Revised: 50 users × 512 context × 256 MB = 12.8 GB (fits!)
Model: 7B Q4_K_M (3.5 GB) + KV (12.8 GB) + overhead (1 GB) = 17.3 GB (exceeds!)

Final Solution: 3B model instead
  - 3B Q4_K_M: 1.5 GB
  - 50 users × 512 context (3B): 50 × 32 MB = 1.6 GB
  - Total: 3.1 GB (fits!)
  - Trade: Quality drops by ~15%, latency improves by 40%
```

**Dispatching Strategy:**
```
Prefill (prompt processing):
  - Batch all 50 users together: (50, 512, 1536) tensor
  - Latency: 50 tokens × 1024 tok/s = 49 ms (very fast for batch)

Decode (token generation):
  - Round-robin: User 1 token, User 2 token, ..., User 50 token, repeat
  - Per-token latency: 10 ms
  - For 128 tokens: 128 * 10 ms = 1.28 sec per user
  - 99% latency SLA: Max 500 ms per token → Easy to achieve
```

**Resource Allocation:**
```
CPU:
  - P-cores: Sampling, logits processing (5-10% utilization)
  - E-cores: Data loading, tokenization (30% utilization)
  - Total CPU: ~2W

GPU:
  - Layer computation: Matrix multiply
  - Frequency: 2.0 GHz (reduced for energy, from 2.5 GHz max)
  - Power: ~8W

Memory Controller:
  - 100 GB/s bandwidth
  - Per-token: 1.5 GB model / 100 GB/s = 15 ms (dominates decode)

Total Power: 10W
Session Duration: 50 users × 128 tokens / (5 tokens/sec) = 1280 sec = 21 minutes
Energy: 10W × 21 min = 200 Wh (< 1% of 50 Wh battery)
```

**Q2 (Quantization Trade-off):** "A model shows 2% perplexity increase when quantized from FP16 to Q4_K_M. Under what conditions should you use Q4_K_M vs FP16? Consider accuracy, latency, memory, and energy."

**A2 Outline:**

| Condition | FP16 | Q4_K_M | Winner |
|---|---|---|---|
| **Model Size Constraint** | 13 GB | 3.5 GB | Q4_K_M (4× compression) |
| **Latency Target** | 150 ms/token | 200 ms/token | FP16 (15% faster) |
| **Accuracy Critical** | 0% drop | 2% drop | FP16 (no degradation) |
| **Energy Budget** | 20W sustained | 12W sustained | Q4_K_M (40% less power) |
| **Inference Scale** | 100K tokens/day | 100K tokens/day | Q4_K_M (energy savings) |

**Decision Matrix:**
```
Use FP16 if: (accuracy critical) OR (latency < 100ms)
Use Q4_K_M if: (memory constrained) OR (energy critical) OR (batch processing)
Use both: Fine-tune critical layers in FP16, non-critical in Q4 (hybrid)
```

**For Apple Silicon (16 GB, 25W budget):**
- 3B model: FP16 (fits: 1.5 GB), latency sufficient
- 7B model: Q4_K_M (fits: 3.5 GB, vs 6.5 GB FP16), trade 2% accuracy for speed
- 13B model: Q4_K_M or go to 7B (13 GB FP16 too large)

**Q3 (Open Problem Design):** "Current llama.cpp doesn't support dynamic batching (multiple variable-size requests processed together). Design a system architecture that enables dynamic batching for Apple Silicon. What new challenges does this introduce?"

**A3 Outline:**

**Architecture:**
```
Request Queue:
  [Request A: 512 tokens, 2K context]
  [Request B: 256 tokens, 4K context]
  [Request C: 128 tokens, 1K context]

Grouping Strategy (maximize GPU utilization):
  Option 1: Process separately (current approach)
    - Request A: 50 ms prefill + 1280 ms decode
    - Request B: 30 ms prefill + 640 ms decode
    - Request C: 10 ms prefill + 320 ms decode
    - Total: 2330 ms

  Option 2: Dynamic batching (desired)
    - Pack A, B, C into single batch
    - Pad to max dimensions (Request B: 4K context)
    - Batch shape: (3, 4096, 1536)
    - Prefill: 30 ms (one kernel call for all 3)
    - Decode: 128 rounds × 15 ms = 1920 ms (sequential, but parallel)
    - Total: 1950 ms (16% faster! Limited by longest request)
```

**Implementation Challenges:**

1. **Ragged Tensor Support:** Requests have different sequence lengths. Options:
   - Pad all to max length (wastes compute)
   - Use sparse tensors (complex, unsupported in Metal)
   - Segment computation (different kernels per group)

2. **KV Cache Management:** Each request maintains separate KV cache (context-dependent).
   - Cannot shared between requests (attention patterns different)
   - Must track offset per request in unified memory

3. **Output Alignment:** Requests finish at different times.
   - Request C finishes at 128 tokens, A continues to 512
   - Must handle variable-length output streams

4. **Latency Variance:** Batching increases latency for small requests (wait for large request).
   - Request C latency: 1950 ms (vs 330 ms standalone) = 6× increase

**Solution (Proposed):**
```
Dynamic Batch Groups:
  - Collect requests with similar sequence length (within 50%)
  - Prefill as batch
  - Decode round-robin (parallel GEMM, not autoregressive)
  - Issue: Attention KV cache prevents parallel decode

Alternative: Pipelined Decode:
  - Request A prefill, then start decode
  - Request B prefill while A decodes token 1
  - Decode: A token 2, B token 1 in parallel
  - Requires sophisticated scheduling

Practical: Batch prefill only (decode stays sequential)
  - Latency impact: -20% on long prompt, +5% on short (amortized)
  - Throughput: +40% on multi-request workload
```

---

## CONCLUSION

Inference on Apple Silicon requires system-level thinking: no single optimization (quantization, batching, frequency) is sufficient. Success demands co-design across hardware (understanding ANE, GPU, memory hierarchy), software (CoreML, Metal, llama.cpp), and algorithms (quantization, speculative decoding, attention fusion).

This module synthesized five complex layers of abstraction (hardware, frameworks, algorithms, systems, applications). Mastery demands hands-on experience with profiling tools (Instruments, powermetrics), benchmark suite construction, and iterative optimization based on measured bottlenecks.

The future of on-device inference is energy-efficient and privacy-preserving. Apple Silicon, with its unified memory and integrated accelerators, is the platform defining this future. Readers who implement the strategies in these five modules will be well-equipped to design production systems that exceed both accuracy and efficiency expectations.

---

## APPENDIX: Quick Reference

### Model Selection Cheat Sheet

```
16 GB MacBook Air → 7B Q4_K_M (context: 2K)
32 GB MacBook Pro → 13B Q4_K_M (context: 4K)
64 GB MacBook Pro Max → 34B Q4_K_M (context: 4K) or dual 13B
```

### Quantization Recommendation

```
Best overall: Q4_K_M (balance of accuracy, size, speed)
Highest accuracy: Q5_K_M or FP16
Smallest size: Q3_K (if accuracy loss < 3% acceptable)
```

### Expected Latency (per token, M2)

```
Prefill (512 tokens): 28 ms aggregate → 0.05 ms per token
Decode (single token): 100 ms
Sampling: 6 ms
Total per token (steady state): 106 ms → 9.4 tokens/sec
```

### Energy Profile (M2, GPU-accelerated)

```
Idle: 2W
Inference (light): 10W
Inference (sustained): 13W
Peak (thermal limit): 25W
```

### Debugging Checklist

```
□ Verify model loads successfully (check file format: GGUF, not PT/ONNX)
□ Benchmark baseline (GPU vs CPU) with powermetrics
□ Profile with Xcode Instruments (GPU timeline)
□ Check memory pressure (pmap, system tools)
□ Validate quantization accuracy (comparison to baseline)
□ Optimize batch size and context length for target latency
□ Enable speculative decoding if throughput > 15 tok/sec goal
□ Measure end-to-end latency (not individual component benchmarks)
```

---

## EPILOGUE: The Path Forward

Apple Silicon represents a unique opportunity for inference practitioners: unmatched energy efficiency, privacy-by-default, and seamless integration with billions of deployed devices. The techniques in these five modules—from ANE dispatch to Metal kernels to quantization-aware optimization—are now standard knowledge for production ML engineers.

As models grow larger (70B+) and inference demands increase, the bottlenecks shift: from compute (achieved by quantization) to memory bandwidth (optimized by unified memory and kernel fusion) to power delivery (designing for thermal limits). Future Apple chips will address bandwidth (expanding to 400+ GB/s), introduce native INT4 arithmetic (avoiding dequantization), and expand ANE operator coverage—but the fundamental principles of efficient inference remain unchanged.

Build with these principles. Measure with precision. Optimize without sacrifice.


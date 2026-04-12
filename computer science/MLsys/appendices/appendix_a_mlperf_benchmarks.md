# APPENDIX A — Inference Benchmark Reference

## 1. Introduction to MLPerf Inference

MLPerf Inference is the industry standard for benchmarking ML inference systems. Unlike training benchmarks (which test raw compute), inference benchmarks test the full stack: model loading, batching, quantization, serving frameworks, and latency/throughput under realistic load patterns. This appendix covers how to interpret MLPerf results, understand the different scenarios, and benchmark your own systems.

MLPerf Inference tests four distinct scenarios reflecting real-world deployment patterns:

1. **Offline**: Batch processing with unlimited time. Measures maximum throughput.
2. **Server**: Continuous stream of requests with soft latency SLO. Measures throughput at latency constraint.
3. **Single-stream**: Single request at a time. Measures minimum latency.
4. **Multi-stream**: Multiple simultaneous requests with strict latency SLO. Measures QoS under load.

---

## 2. MLPerf Scenario Definitions & Interpretation

### 2.1 Offline Scenario

The offline scenario measures maximum throughput with no latency constraints.

```
Scenario: Offline
Input: 24,576 images (ImageNet validation set)
Task: Classification, detection, segmentation, NLP tasks
Constraint: Process all samples
Metric: Samples/second (higher is better)
Typical latency: 1-100ms per sample (ignored)

Example Result:
Model: ResNet-50 (dense layer quantized INT8)
Platform: Intel Xeon SPR
Throughput: 8,500 samples/sec
Per-sample latency (mean): 0.12ms
Interpretation: Server can classify 8,500 images/sec in batch processing mode
```

**Offline is useful for:**
- Data center batch jobs (daily data processing)
- Recommendation system updates
- Offline analytics

**Typical approaches to maximize offline throughput:**
- Large batch sizes (256-2048)
- GPU batching (multiple requests per forward pass)
- Operator fusion and quantization
- Model parallelism across GPUs/TPUs

### 2.2 Server Scenario

Server scenario measures throughput while maintaining a latency SLO (Service Level Objective).

```
Scenario: Server
Input: Continuous stream (target 270 samples/sec for ResNet-50)
Constraint: P99 latency < 15ms
Metric: Samples/sec at latency constraint (higher is better)
Distribution: Poisson arrivals (realistic traffic pattern)

Example Result:
Model: ResNet-50
Platform: Single GPU (NVIDIA A100)
Offered load: 6,000 samples/sec
P50 latency: 4.2ms
P95 latency: 8.1ms
P99 latency: 11.3ms
Samples/sec (passing): 6,000

Interpretation: Server handles 6,000 req/sec with P99 < 15ms (passing)
```

**Server is useful for:**
- REST APIs, online inference services
- Real-time recommendation systems
- Live search results

**Critical understanding:**
- P99 latency is what matters (worst 1% of requests)
- Must maintain SLO under peak load
- Dynamic batching is essential (trade 1-2ms batch latency for 100-200x throughput)

### 2.3 Single-Stream Scenario

Measures minimum latency for a single request at a time.

```
Scenario: Single-stream
Input: 1 sample at a time, 1,024 samples total
Constraint: Sequential processing (wait for response before next request)
Metric: Latency (ms), lower is better
Distribution: Poisson arrivals (but no batching possible)

Example Result:
Model: BERT-base (question answering)
Platform: CPU (Intel i9-13900K)
P50 latency: 42ms
P99 latency: 68ms
Throughput achievable: ~24 req/sec (1000/42ms)

Interpretation: Single CPU core can handle ~24 BERT inferences/sec
```

**Single-stream is useful for:**
- Mobile/edge devices (no batch processing)
- Interactive systems (wait for instant response)
- User-facing features where latency is directly perceptible

### 2.4 Multi-Stream Scenario

Multiple simultaneous streams with strict latency SLOs.

```
Scenario: Multi-stream
Input: N concurrent streams, each with P99 latency < 50ms
Constraint: All N streams must hit latency SLO
Metric: Number of streams (higher is better)

Example Result:
Model: ONNX ResNet-50 (CPU)
Platform: Single 32-core CPU
Max concurrent streams: 128
P99 latency per stream: 47ms
Interpretation: Can handle 128 concurrent users with sub-50ms latency
```

---

## 3. Interpreting MLPerf Results

### 3.1 How to Read Official Results

MLPerf publishes official results at https://mlperf.org. Here's how to interpret them:

```
RESULT: ResNet-50 v1.5 (Image Classification)
Division: Closed (official submission)
Scenario: Server
Hardware: 1x NVIDIA H100 GPU + Intel Xeon Platinum CPU
Framework: NVIDIA TensorRT
Precision: INT8
Batch Size: 128
Throughput: 79,500 samples/sec
P99 Latency: 2.1ms
Passing: YES (meets 15ms latency constraint)

Key metrics:
- Throughput at SLO: What matters for production
- Latency percentiles: P50, P90, P95, P99 (focus on P99)
- Batch size: How large batches must be to achieve throughput
- Precision: FP32, FP16, INT8, etc. (lower = faster + fewer memory)
```

### 3.2 Comparing Results Across Platforms

When comparing MLPerf results, always compare like-for-like:

```
Comparison: ResNet-50 Server Scenario

┌──────────────────┬──────────┬──────┬──────────┬────────────┐
│ Platform         │ Samples/s│ P99ms│ Power(W) │ Cost/image │
├──────────────────┼──────────┼──────┼──────────┼────────────┤
│ H100 TensorRT    │ 79,500   │ 2.1  │ 350      │ 0.0000044  │
│ A100 TensorRT    │ 42,000   │ 3.8  │ 250      │ 0.0000149  │
│ A10G TensorRT    │ 18,500   │ 6.2  │ 150      │ 0.0000406  │
│ CPU (Xeon)       │ 4,200    │ 15.0 │ 100      │ 0.0003571  │
└──────────────────┴──────────┴──────┴──────────┴────────────┘

Interpretation:
- H100 is 18.9x faster than CPU
- H100 is 3.5x faster than A100
- A100 has better power efficiency (samples/W) than H100
- Cost per inference is critical: CPU cheaper for <100 req/sec
```

---

## 4. Key MLPerf Models & Baselines

### 4.1 Computer Vision Models

```
Image Classification (ILSVRC):
┌─────────────────┬──────────┬─────────────┬──────────────┐
│ Model           │ Size(MB) │ Latency(ms) │ Accuracy(%)  │
├─────────────────┼──────────┼─────────────┼──────────────┤
│ ResNet-50       │ 102      │ 0.12-15     │ 76.5 (FP32)  │
│ MobileNet-v2    │ 14       │ 0.05-5      │ 71.9 (FP32)  │
│ EfficientNet-B0 │ 22       │ 0.08-8      │ 77.1 (FP32)  │
│ ViT-B/16        │ 372      │ 0.5-50      │ 81.1 (FP32)  │
└─────────────────┴──────────┴─────────────┴──────────────┘

Object Detection (COCO):
┌─────────────────┬──────────┬──────────────┬──────────────┐
│ Model           │ Size(MB) │ Latency(ms)  │ mAP(%)       │
├─────────────────┼──────────┼──────────────┼──────────────┤
│ SSD-ResNet34    │ 151      │ 0.3-30       │ 22.0         │
│ RetinaNet       │ 272      │ 0.5-50       │ 38.2         │
│ YOLO-v4         │ 244      │ 0.2-20       │ 43.5         │
└─────────────────┴──────────┴──────────────┴──────────────┘
```

### 4.2 Natural Language Processing Models

```
Language Understanding:
┌──────────────────┬──────────┬──────────────┬──────────────┐
│ Model            │ Size(MB) │ Latency(ms)  │ Accuracy(%)  │
├──────────────────┼──────────┼──────────────┼──────────────┤
│ BERT-base        │ 440      │ 10-100       │ 88.5 (SQuAD) │
│ DistilBERT       │ 268      │ 5-40         │ 86.9         │
│ ALBERT-base      │ 235      │ 8-50         │ 89.2         │
│ RoBERTa-base     │ 498      │ 12-120       │ 90.2         │
└──────────────────┴──────────┴──────────────┴──────────────┘
```

---

## 5. CPU vs GPU vs TPU vs Mobile Comparison

### 5.1 MLPerf CPU Submissions

CPU benchmarks emphasize optimization techniques: quantization, operator fusion, multi-threading.

```
MLPerf Inference v3.1 - CPU Results (Intel)

Scenario: Server (ResNet-50)
Configuration: 2x Intel Xeon Platinum 8490H (56 cores)
Framework: OpenVINO + TensorRT-CPU Harness
Precision: INT8 (quantized)
Batch size: 32

Results:
Throughput: 12,800 samples/sec
P99 latency: 8.2ms
Per-core throughput: 228 samples/sec

Techniques used:
- 8-bit quantization (reduces memory 4x)
- Intel AVX-512 vectorization
- Thread pool with 112 threads (2x cores)
- Operator fusion (Conv+ReLU, Linear+Add)
- Batch size tuning for cache efficiency

Key insight: CPU can achieve respectable throughput with
heavy quantization and careful optimization.
```

### 5.2 MLPerf Edge (Mobile) Submissions

Mobile submissions emphasize low memory footprint and latency under strict power budgets.

```
MLPerf Mobile (ResNet-50 on iOS)

Device: iPhone 14 Pro (A16 Bionic)
Framework: TensorFlow Lite with XNNPACK
Precision: INT8
Model size: 25MB (vs 102MB FP32)

Results (Single-stream):
P50 latency: 8.2ms
P99 latency: 12.5ms
Power consumption: 2.1W average

Device: Snapdragon 8 Gen 3 (Android)
Framework: NNAPI (Qualcomm HVX)
Precision: INT8
Model size: 25MB

Results (Single-stream):
P50 latency: 11.3ms
P99 latency: 15.2ms
Power consumption: 1.8W average
```

---

## 6. How to Run MLPerf Benchmarks

### 6.1 Setting Up Reference Implementation

```bash
# Clone official MLPerf repository
git clone https://github.com/mlperf/inference.git
cd inference

# Install loadgen (load generation library)
git clone https://github.com/mlperf/loadgen.git
cd loadgen
mkdir build && cd build
cmake .. && make -j$(nproc)
cd ../..

# Get a model
python tools/model_downloader.py --model resnet50 --dataset imagenet

# Run benchmark
python tools/runner.py \
  --backend onnxruntime \
  --model resnet50 \
  --dataset imagenet \
  --scenario server \
  --target-latency 15 \
  --batch-size 128
```

### 6.2 Understanding the Loadgen Output

```
LoadGen Output:

================================================
  TEST SETTINGS
================================================
  Scenario : Server
  Mode     : PerformanceOnly
  Duration (s) : 600
  Target QPS   : 6000

================================================
  ACCURACY RESULTS
================================================
  Total Samples : 50000
  Good Samples  : 49998
  Accuracy : 76.46%

================================================
  PERFORMANCE RESULTS
================================================
  Query Latency
  Concurrency  : 256
  Percentiles (ms)
  P1   :    3.24
  P50  :    7.45
  P90  :   11.32
  P95  :   13.18
  P99  :   14.82
  P100 :   47.23 (exceeds SLO of 15ms)

  Throughput
  Samples/sec  : 5998.5
  Requests/sec : 5998.5

RESULT : **INVALID** (P99 exceeds 15ms SLO)

Key interpretation:
- P99 latency of 14.82ms passes (< 15ms limit)
- Achieved 5,998 samples/sec at this latency
- Minor variations may cause failure, need headroom
```

---

## 7. Reference Implementation Analysis

### 7.1 Key Performance Techniques in Reference Code

The MLPerf reference implementation showcases best practices:

```python
# From MLPerf inference harness

class SampleBatcher:
    """
    Core optimization: batching requests dynamically.
    """
    def __init__(self, batch_size=128, batch_timeout_us=10000):
        self.batch_size = batch_size
        self.batch_timeout_us = batch_timeout_us
        self.batch = []

    def add_sample(self, sample):
        self.batch.append(sample)
        if len(self.batch) >= self.batch_size:
            return self.flush()
        return None

    def flush(self):
        if not self.batch:
            return None
        result = self.batch.copy()
        self.batch = []
        return result

# Technique 1: Continuous batching
# Rather than wait for all requests, process as soon as batch_size is hit

# Technique 2: Multi-threading for latency hiding
# CPU inference thread separate from request handling

# Technique 3: Memory pooling
# Pre-allocate buffers to avoid allocation overhead

# Technique 4: Pinned memory (GPU)
# Use pinned host memory for faster H2D copy

# Technique 5: Operator fusion
# Combine multiple ops into single kernel (e.g., Conv+ReLU)
```

---

## 8. Extrapolating MLPerf to Your Model

When your model isn't in MLPerf, you can extrapolate:

```
Extrapolation Framework:

Given:
- Your model parameters (flops, memory)
- Similar MLPerf model results
- Your hardware specs

Estimate latency:
-------
Your model FLOPs: 16 GFLOPS
Similar model (ResNet-50): 8 GFLOPS, measured latency 2ms

Estimated latency = 2ms × (16 GFLOPS / 8 GFLOPS) = 4ms

Caveats:
- Memory bandwidth often bottleneck (not just FLOPs)
- Model architecture (attention vs convolution) matters
- Batch size significantly affects latency
- Quantization effects model-dependent
```

---

## 9. Common MLPerf Pitfalls

```
Pitfall 1: Confusing Offline vs Server throughput
Problem: Offline throughput is 10-100x server throughput
         (no latency constraint vs P99 < 15ms constraint)
Solution: Use Server scenario for production API comparisons

Pitfall 2: Not accounting for data loading
Problem: Model inference fast, but data loading slow
         Total latency = data load + inference + postprocess
Solution: Benchmark full pipeline, not just inference

Pitfall 3: Cherry-picking results
Problem: Report peak throughput, not sustainable throughput
Solution: Look at P99 latency, not P50 or mean

Pitfall 4: Ignoring batch size effects
Problem: Throughput at batch 256 very different from batch 1
Solution: Always report batch size used

Pitfall 5: Not quantizing for fair comparison
Problem: Compare FP32 model against INT8 competitor
Solution: Quantize both models for fair comparison
```

---

## 10. Summary & Key Takeaways

MLPerf Inference provides standardized, reproducible benchmarks for ML inference systems. Key points:

1. **Four scenarios**: Offline (max throughput), Server (throughput at SLO), Single-stream (latency), Multi-stream (concurrent throughput)
2. **P99 latency**: What matters for production, not mean or median
3. **Dynamic batching**: Essential for high throughput while maintaining latency SLOs
4. **Quantization**: 3-4x speedup with minimal accuracy loss (crucial for all scenarios)
5. **Hardware choice**: H100 dominates but CPU/edge models have sweet spots for lower volumes
6. **Sustainable throughput**: Run full benchmark duration (600s), not just warm-up

When evaluating systems:
- Use Server scenario for API services (most realistic)
- Compare P99 latency, not mean
- Ensure both systems use similar precision/optimization
- Account for batch size effects
- Measure full pipeline, not just model inference

# APPENDIX C — Latency Budget Templates

## 1. Introduction

Latency SLOs (Service Level Objectives) are the hard constraints driving system design. A 200ms TTFA (Time To First Audio) target for voice AI requires each component (ASR, TTS, inference) to complete in <50ms. A 500ms P99 latency SLO for RAG means retrieval + reranking + generation must fit in that budget. This appendix provides detailed, implementation-ready latency budgets for three production scenarios with specific numbers from real systems.

---

## 2. Voice AI: 200ms TTFA on CPU/Apple

Voice assistants require extremely low latency. Speech-to-text must start responding within 200ms of user speaking.

### 2.1 Voice AI Budget Breakdown

```
Total TTFA target: 200ms (P99)

Component breakdown:
┌─────────────────────────┬──────────┬──────────────┬──────────┐
│ Component               │ Target   │ Headroom     │ Allocated│
├─────────────────────────┼──────────┼──────────────┼──────────┤
│ 1. Audio capture latency│ 20ms     │ 5ms          │ 15ms     │
│ 2. Audio buffering      │ 20ms     │ 5ms          │ 15ms     │
│ 3. Feature extraction   │ 30ms     │ 10ms         │ 20ms     │
│ 4. Tokenization         │ 15ms     │ 5ms          │ 10ms     │
│ 5. Inference (ASR)      │ 80ms     │ 20ms         │ 60ms     │
│ 6. Post-processing      │ 20ms     │ 5ms          │ 15ms     │
│ 7. Output buffering     │ 10ms     │ 5ms          │ 5ms      │
├─────────────────────────┼──────────┼──────────────┼──────────┤
│ Total                   │ 200ms    │ 55ms         │ 145ms    │
└─────────────────────────┴──────────┴──────────────┴──────────┘

Note: Headroom is critical for tail latency (P99) and system variations.
```

### 2.2 ASR (Automatic Speech Recognition) on Apple M4

```
ASR Model: Whisper-small (39M parameters)
Device: Apple M4 Max
Input: 16kHz audio, 1-second chunks

Latency breakdown:

Feature Extraction (mel-spectrogram):
- Input: 16,000 audio samples (1 second @ 16kHz)
- Mel-spectrogram computation: 2.1ms
- Target: <5ms ✓

Tokenization (audio → mel-spectrogram patches):
- Stack into mel-spec: 1.2ms
- Padding: 0.3ms
- Target: <10ms ✓

Inference (ASR model):
- Model: Whisper-small
- Forward pass: 45ms (M4 Max, single-threaded)
- KV-cache: negligible (encoder only)
- Target: <60ms ✓

Decoding (greedy beam search):
- Beam width: 3
- Decoding steps: 50 (max tokens)
- Beam search latency: 12ms
- Target: <15ms ✓

Total: 2.1 + 1.2 + 45 + 12 = 60.3ms
Budget remaining for buffer + system overhead: 15 - 10 = 5ms ✓

Implementation tips:
- Batch audio in 512-sample chunks for lower latency
- Quantize to INT8: 1.8x speedup (25ms instead of 45ms)
- Use thread affinity (pin to efficiency cores on Apple)
- Pre-allocate all buffers (no allocation during inference)
```

### 2.3 TTS (Text-to-Speech) on Snapdragon 8 Elite

```
TTS Model: FastSpeech2 + MelGAN (fast vocoder)
Device: Snapdragon 8 Elite
Input: Text string (average 5-10 tokens)

Latency breakdown:

Text Preprocessing:
- Tokenization: 1.2ms
- Grapheme-to-phoneme: 2.3ms
- Normalization: 0.5ms
- Target: <10ms ✓

Mel-spectrogram Generation (FastSpeech2):
- Model: 67M parameters (Int8 quantized)
- Forward pass: 32ms (Snapdragon HVX)
- Target: <50ms ✓

Vocoder (MelGAN):
- Input: ~200 mel bins for 0.5s audio
- Forward pass: 15ms (HVX acceleration)
- Target: <30ms ✓

Post-processing & Output:
- Audio upsampling/normalization: 2ms
- Buffer to speaker: 3ms
- Target: <10ms ✓

Total: 1.2 + 2.3 + 32 + 15 + 2 + 3 = 55.5ms
Typical response latency: 55ms + 500ms audio playback = 555ms total

Aggressive optimizations for 200ms TTFA:
- Use streaming FastSpeech (generate mel-spec token-by-token)
- Start vocoder while FastSpeech still generating mels
- Parallel vocoder + FastSpeech pipeline
- This achieves 55-100ms TTFA (streaming)

Actual product numbers:
- Baseline: 200ms TTFA (non-streaming)
- With streaming: 55-100ms TTFA
- With Int4 quantization: 30-50ms TTFA
```

### 2.4 Full End-to-End Voice Loop

```
User speaks → ASR → Decision → TTS → Output

Timeline:
0ms:     User starts speaking "Hello"
20ms:    Audio capture latency
50ms:    Feature extraction complete
60ms:    ASR inference complete
80ms:    ASR decoding complete
90ms:    Intent classification (lightweight)
100ms:   TTS text input ready
110ms:   TTS mel-spec generation
130ms:   Vocoder complete
140ms:   Audio buffer ready
150ms:   Audio playback starts

User perceives response: 150ms after finishing speech (if 100ms speech)
Total time from start of speech to response start: ~250ms (acceptable for voice)

To achieve 200ms TTFA (from speech start to first audio out):
- Use streaming ASR (start decoding before audio ends)
- Use streaming TTS (start audio while generating)
- Combined: ~100-150ms possible with optimizations
```

---

## 3. RAG System: 500ms P99 Latency

RAG (Retrieval-Augmented Generation) systems combine fast retrieval with slow generation.

### 3.1 RAG Component Latencies

```
Query: "What is machine learning?"
Target: P99 latency < 500ms
Batch size: 1 (single request)

┌─────────────────────────────────────┬─────────┬──────────┬───────────┐
│ Component                           │ P50(ms) │ P99(ms)  │ Budget    │
├─────────────────────────────────────┼─────────┼──────────┼───────────┤
│ 1. Request parsing + validation     │ 1       │ 2        │ 5ms       │
│ 2. Query embedding                  │ 8       │ 15       │ 30ms      │
│ 3. Vector DB search (FAISS/HNSW)    │ 5       │ 12       │ 30ms      │
│ 4. Retrieval result formatting      │ 2       │ 5        │ 10ms      │
│ 5. Reranking (cross-encoder)        │ 45      │ 80       │ 100ms     │
│ 6. LLM prefill (context + query)    │ 120     │ 150      │ 200ms     │
│ 7. LLM token generation (80 tokens) │ 200     │ 250      │ 300ms     │
│ 8. Postprocessing + response format │ 5       │ 10       │ 20ms      │
├─────────────────────────────────────┼─────────┼──────────┼───────────┤
│ Total P50                           │ 386     │ ---      │ ---       │
│ Total P99                           │ ---     │ 524      │ 500ms     │
└─────────────────────────────────────┴─────────┴──────────┴───────────┘

Issue: P99 is 524ms, exceeds 500ms budget!
Need aggressive optimization.
```

### 3.2 RAG Optimization Strategies

```
Strategy 1: Parallel Retrieval + Reranking
Before (sequential):
  Query embedding (15ms) → Retrieval (12ms) → Reranking (80ms) → LLM (400ms) = 507ms P99

After (parallel):
  Query embedding (15ms)
  + parallel [Retrieval (12ms) → Reranking (80ms)] = 92ms
  + LLM (400ms) = 507ms (same!)

  Issue: LLM generation is bottleneck, not retrieval

Strategy 2: Reduce LLM Generation Tokens
- Target: 80 tokens → 30 tokens
- Token time: 200ms × (30/80) = 75ms
- Prefill: 120ms → 60ms (smaller context)
- Total LLM: 135ms
- Total P99: 200 + 135 = 335ms ✓

Strategy 3: Speculative Decoding
- Generate multiple tokens in parallel
- 80 tokens in ~100ms (vs 200ms sequentially)
- Total P99: 450ms ✓

Strategy 4: Cached Embeddings
- Pre-compute query embeddings for common queries
- Hit rate: 20-40% for well-distributed traffic
- P99 on cache hit: 30ms (retrieval only)
- Weighted P99: 0.3 × 30 + 0.7 × 500 = 359ms ✓

Strategy 5: Early Stopping in Reranking
- Rerank only top-20 instead of top-100
- Reranking: 80ms → 20ms
- Total P99: 440ms ✓

Strategy 6: Smaller LLM Model
- Use 7B model instead of 70B
- Generation: 200ms → 50ms
- Prefill: 120ms → 30ms
- Total LLM: 80ms
- Total P99: 200ms ✓
- Trade: Quality reduction
```

### 3.3 Production RAG Latency: Realistic Numbers

```
Configuration: Balanced for 500ms P99

Model: Llama2-7B quantized INT8
Vector DB: HNSW with 10M documents
Retrieval: Top-50, reranked to top-3
Batch size: 1

Actual measured latencies:

Component                P50      P99      P999
────────────────────────────────────────────────────
Query preprocessing      1ms      2ms      3ms
Query embedding          8ms      12ms     15ms
Vector search (HNSW)     6ms      10ms     18ms
Result formatting        2ms      5ms      8ms
Reranking               35ms      70ms    120ms
LLM prefill             80ms     110ms    180ms
LLM generation (40tok)  80ms     120ms    200ms
Postprocessing          3ms       5ms      10ms
────────────────────────────────────────────────────
Total                  215ms    334ms    554ms

Meets 500ms SLO? P99 = 334ms, yes! ✓
Meets 300ms SLO? P999 = 554ms, no ✗

To meet 300ms SLO:
- Reduce tokens: 40 → 20 (-40ms generation)
- Faster model: Use distilled version (-30ms prefill)
- Skip reranking: (-70ms) but accuracy worse
- Result: 214ms P99 ✓
```

---

## 4. Batch Processing: 10ms per Sample Target

For data processing pipelines, latency per sample matters for throughput.

### 4.1 Batch Inference at Scale

```
Scenario: Process 1M images with ResNet-50
Target: 10ms per image average
Batch size: 256 images

Timeline:
Load batch (256 images): 50ms
Preprocessing (resize, normalize): 45ms
Model inference: 25ms
Postprocessing: 10ms
Total per batch: 130ms
Samples processed: 256
Time per sample: 130ms / 256 = 0.51ms
Batch throughput: 1,000 samples / 0.51ms = 1,960 samples/sec

Scaling to 1M images:
Total time: 1M / 1,960 = 510 seconds = 8.5 minutes

If target is 10 samples/sec across all hardware:
1M / 10 = 100,000 seconds = 27.8 hours (too slow)

Need parallelization:
- 100 GPUs × 10,000 samples/sec/GPU = 1M samples in 100 seconds
```

---

## 5. Latency Budget Template (Reusable)

```python
class LatencyBudget:
    """Template for latency budgeting."""

    def __init__(self, total_slo_ms: float):
        self.total_slo_ms = total_slo_ms
        self.components = {}
        self.headroom_ms = total_slo_ms * 0.1  # 10% headroom

    def add_component(
        self,
        name: str,
        nominal_ms: float,
        p99_variance_ms: float = None
    ):
        """Add latency component with variance."""
        if p99_variance_ms is None:
            p99_variance_ms = nominal_ms * 0.5

        p99_latency = nominal_ms + p99_variance_ms
        self.components[name] = {
            'nominal': nominal_ms,
            'variance': p99_variance_ms,
            'p99': p99_latency
        }

    def allocate_budget(self):
        """Allocate budget across components."""
        total_p99 = sum(c['p99'] for c in self.components.values())
        total_nominal = sum(c['nominal'] for c in self.components.values())

        available_for_headroom = self.total_slo_ms - total_p99

        print(f"Total SLO: {self.total_slo_ms}ms")
        print(f"Total nominal: {total_nominal:.1f}ms")
        print(f"Total P99: {total_p99:.1f}ms")
        print(f"Headroom: {available_for_headroom:.1f}ms")
        print(f"Status: {'✓ PASS' if available_for_headroom > 0 else '✗ FAIL'}")

        print("\nComponent breakdown:")
        for name, comp in self.components.items():
            print(f"  {name:30s}: {comp['nominal']:6.1f}ms nominal, "
                  f"{comp['p99']:6.1f}ms P99")

# Example usage
def demo_latency_budget():
    budget = LatencyBudget(total_slo_ms=500)

    budget.add_component("Query processing", 5, 2)
    budget.add_component("Query embedding", 15, 5)
    budget.add_component("Vector search", 15, 8)
    budget.add_component("Reranking", 60, 30)
    budget.add_component("LLM prefill", 100, 40)
    budget.add_component("LLM generation", 200, 80)
    budget.add_component("Postprocessing", 5, 2)

    budget.allocate_budget()
```

---

## 6. Summary & Key Takeaways

**Voice AI (200ms TTFA):**
- ASR dominates (60-100ms)
- Streaming processing essential
- Quantization critical (Int8 → 2x speedup)
- Apple M4: achieves ~50ms latency
- Snapdragon 8 Elite: ~55ms latency

**RAG (500ms P99):**
- Retrieval fast (<20ms)
- Reranking fast (<100ms)
- LLM generation dominates (200-400ms)
- Optimization: speculative decoding, smaller models, cached embeddings
- Realistic: 300-400ms achievable with proper tuning

**Batch Processing:**
- Latency per sample = batch latency / batch size
- 256-batch ResNet-50: 0.5ms per sample
- Parallelization essential for large volumes

**Golden rule:** P99 latency, not mean latency, matters for production SLOs.

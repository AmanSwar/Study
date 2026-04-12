# MODULE 29 — Voice AI Pipeline Architecture

## 1. INTRODUCTION & MOTIVATION

Voice AI systems represent one of the most stringent real-time ML challenges in modern computing. Unlike offline batch processing or even interactive chatbots where users tolerate 1-2 second latencies, voice applications demand sub-300ms Time-To-First-Audio (TTFA) for natural conversation flow. The human brain's capacity to detect latency in speech is extraordinarily sensitive: studies show that latencies exceeding 250ms create noticeable disruption in turn-taking behavior, while competitive voice experiences target 150-200ms TTFA. This requirement cascades through every component in the pipeline, from audio capture and Voice Activity Detection (VAD) through automatic speech recognition (ASR), large language model (LLM) inference, text-to-speech (TTS) synthesis, and finally audio playback.

The architectural challenge is profound: we must orchestrate five to seven independent neural network components, each with their own latency characteristics, through a real-time streaming pipeline where the output of one stage becomes the input to the next. Unlike batch processing where we optimize total throughput, voice systems must optimize for latency percentiles—specifically the 95th and 99th percentile TTFA, not the mean. A single blocking operation or poorly designed buffer can destroy the user experience.

This module establishes the foundational architectural patterns used in production voice systems: producer-consumer pipelines, streaming-first design where every component processes audio incrementally rather than in complete utterances, pipeline parallelism that exploits the multi-stage nature of the system, and interrupt/barge-in mechanisms that allow users to cut off the system mid-response. We cover the hardware and software abstractions that enable 200ms TTFA on commodity CPUs and mobile processors.

## 2. FULL STACK VOICE PIPELINE

The complete voice AI pipeline consists of the following sequential stages:

```
┌──────────────────────────────────────────────────────────────────┐
│                    Voice AI Full Stack Pipeline                   │
├──────────────────────────────────────────────────────────────────┤
│  1. Audio Capture        (Microphone → PCM buffer, ~10ms chunks)  │
│  2. Audio Processing     (Resampling, preprocessing)              │
│  3. VAD (Voice Activity)  (Detect speech start/end)               │
│  4. ASR (Speech→Text)     (Streaming speech recognition)          │
│  5. LLM (Text→Response)   (Large language model inference)        │
│  6. TTS (Text→Speech)     (Text-to-speech synthesis)              │
│  7. Audio Playback        (Speaker output, simultaneous capture)  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.1 Audio Capture & Input Buffer

Audio capture begins with device selection and microphone configuration. Most systems use 16kHz or 48kHz sampling rates depending on hardware capabilities. The microphone delivers audio in small frames (typically 10ms chunks at 16kHz = 160 samples per frame) through platform-specific APIs:

- **Linux/ALSA**: Direct device access with ring buffers, minimal latency
- **macOS/CoreAudio**: Hardware abstraction layer with callback-based I/O
- **iOS/Android**: High-level audio APIs with inherent platform buffering
- **Windows/WASAPI**: Modern exclusive mode for minimum latency

The critical design principle is **zero-copy buffering**: audio data flows through shared ring buffers where each consumer reads the producer's written frames without copying. A 10ms capture latency is typical; modern systems achieve 5ms with careful tuning.

### 2.2 Audio Preprocessing

Raw microphone audio requires preprocessing before entering VAD and ASR:

1. **Resampling** (if needed): Most ASR models expect 16kHz; resampling from 48kHz adds ~5-10ms latency
2. **Normalization**: Peak normalization to [-1.0, 1.0] range
3. **Windowing**: Apply Hann window for spectral analysis
4. **Mel filterbank**: STFT → log-mel spectrogram for VAD/ASR (5-10ms latency)

Mel filterbank computation is one of the highest-frequency operations in the pipeline (running 100x/second at 16kHz with 10ms frames) and benefits enormously from SIMD vectorization.

### 2.3 Voice Activity Detection (VAD)

VAD runs continuously on the preprocessed audio stream and solves two problems:
1. **Speech detection**: Identify when the user starts speaking
2. **Endpointing**: Detect when the user finishes speaking (end-of-utterance)

VAD must run with sub-100ms latency and sub-1% false positive rate (erroneously detecting silence as speech). Most systems use probabilistic outputs (0.0-1.0 confidence) rather than hard binary decisions, allowing downstream stages to weight uncertainty.

### 2.4 Automatic Speech Recognition (ASR)

The ASR module converts streaming audio to text tokens. Modern streaming ASR architectures (RNN-T, Whisper) produce output incrementally, emitting partial hypotheses every 500ms-2s that progressively improve as more audio arrives. Key metrics:

- **Latency to first token**: 500-2000ms (typically 1s)
- **Confidence scores**: Most systems output top-1 hypothesis + alternatives
- **Streaming stability**: Incremental results shouldn't regress (monotonicity)

ASR represents 30-50% of overall TTFA budget in voice systems.

### 2.5 Large Language Model (LLM)

The LLM converts recognized text to a response. In voice systems, the LLM must:
1. Accept incremental ASR results (handle incomplete sentences)
2. Generate responses token-by-token (streaming)
3. Meet strict latency SLA (TTFT < 100ms for voice, much tighter than text)

The LLM stage typically dominates latency for long responses. Serving optimizations include prefix caching, KV cache warmup, and model quantization.

### 2.6 Text-to-Speech (TTS)

TTS synthesizes LLM tokens into audio, ideally producing audio as tokens arrive (streaming TTS). Key requirements:

- **Latency to first audio**: 100-200ms after first token
- **Voice quality**: Natural prosody and speaking rate
- **Streaming stability**: Audio shouldn't backtrack or produce clicks when new tokens arrive

TTS latency is highly variable: the first token produces 200-400ms of audio synthesis latency, but subsequent tokens may arrive faster due to caching effects.

### 2.7 Audio Playback & Echo Management

Audio output must be synchronized precisely with microphone input (within 20-50ms) to enable echo cancellation. Systems typically use the same audio device abstraction API for both I/O, maintaining a common timebase. Ring buffers for playback must be sized for smooth streaming without underruns.

## 3. LATENCY TARGETS & BUDGETS

### 3.1 TTFA (Time-To-First-Audio) Requirements

Natural conversation requires TTFA < 300ms; competitive experiences target 150-200ms. The tightest designs achieve 150ms on local CPU with optimized models. We break down a 200ms budget:

| Stage | Budget | Typical |
|-------|--------|---------|
| Audio capture + VAD | 50ms | 40-60ms |
| ASR | 30-50ms | 40-100ms |
| LLM (TTFT) | 80ms | 100-300ms |
| TTS (first audio) | 40ms | 100-200ms |
| **Total TTFA** | **200ms** | **180-660ms** |

**Aggressive optimization for <200ms TTFA**:
- VAD: Use <20ms RNN (Silero v5 on CPU)
- ASR: Low-latency streaming Whisper with local agreement; emit first token at 100ms audio
- LLM: Small 3B-7B model with prefill <50ms on modern CPU
- TTS: Lightweight FastSpeech-based vocoder with 50ms synthesis startup
- Architecture: Streaming all components; LLM starts once first ASR token available (don't wait for complete sentence)

### 3.2 Latency Percentiles

In practice, TTFA follows a distribution. The user experience is driven by 95th and 99th percentile latency, not mean:

```
Example distribution (200ms budget):
P50 (median):     145ms
P75:              165ms
P90:              190ms
P95:              220ms (UNACCEPTABLE - exceeds budget)
P99:              450ms (user detects failure)
```

Sources of latency variance:
- **CPU contention**: Background tasks, GC pauses (Java/Python)
- **Model inference variance**: Autoregressive models have data-dependent latency
- **System noise**: Scheduler, interrupt handling, page faults
- **I/O jitter**: Audio buffer refill timing, network variance (for cloud-based components)

Production systems maintain detailed latency histograms (e.g., using prometheus quantile buckets) to detect degradation.

## 4. STREAMING ARCHITECTURE & PRODUCER-CONSUMER

The core architectural pattern is **multi-stage producer-consumer pipelines** where each stage runs in its own thread (or async context) and communicates via lock-free queues.

### 4.1 Lock-Free Ring Buffers

Audio data flows through lock-free ring buffers (circular FIFO) with single producer and single consumer:

```cpp
template<typename T, size_t Capacity>
class RingBuffer {
  T buffer[Capacity];
  std::atomic<size_t> write_pos{0};
  std::atomic<size_t> read_pos{0};

public:
  bool push(const T& value) {
    size_t next_write = (write_pos + 1) % Capacity;
    if (next_write == read_pos.load(std::memory_order_acquire)) {
      return false; // Buffer full
    }
    buffer[write_pos] = value;
    write_pos.store(next_write, std::memory_order_release);
    return true;
  }

  bool pop(T& value) {
    if (read_pos.load(std::memory_order_acquire) == write_pos) {
      return false; // Buffer empty
    }
    value = buffer[read_pos];
    read_pos.store((read_pos + 1) % Capacity, std::memory_order_release);
    return true;
  }
};
```

These buffers handle timing mismatches between stages (e.g., VAD processes slowly while audio arrives at constant rate) without memory allocation or mutex contention.

### 4.2 Pipeline Threading Model

A typical voice system might use:

```
Thread 1: Audio I/O Thread
  ├─ Audio capture callback (high priority)
  ├─ Push to VAD ring buffer
  ├─ Audio playback callback (high priority)
  └─ Consume from playback ring buffer

Thread 2: VAD Processing
  ├─ Consume from audio ring buffer
  ├─ Mel filterbank computation (SIMD)
  ├─ VAD inference (Silero ONNX)
  └─ Push to ASR ring buffer (when speech detected)

Thread 3: ASR Processing
  ├─ Consume from VAD ring buffer
  ├─ Streaming Whisper or RNN-T inference
  └─ Push to LLM ring buffer (emit tokens as available)

Thread 4: LLM Processing
  ├─ Consume from ASR ring buffer
  ├─ Context management, prefix caching
  ├─ LLM inference (batched if multiple sessions)
  └─ Push to TTS ring buffer (stream tokens)

Thread 5: TTS Processing
  ├─ Consume from LLM ring buffer
  ├─ TTS inference (streaming)
  ├─ Vocoder execution
  └─ Push to playback ring buffer
```

Each thread processes independently, pulling data when available. The ring buffers naturally decouple component latencies: if ASR is slow, VAD output buffers; if LLM is fast, it waits for ASR tokens.

### 4.3 Flow Control & Backpressure

When a ring buffer fills, the producer must slow down or drop frames. Voice systems typically:

1. **For audio capture**: Use fixed-size capture buffers from the OS; pushing to full ring buffer is an error condition
2. **For processing stages**: Implement timeout-based pop() operations; if no data within 100ms, assume pipeline stalled and emit diagnostic
3. **For output stages**: Drop audio frames rather than block (prefer skipping one frame of audio to maintaining audio sync)

## 5. PIPELINE PARALLELISM & OVERLAP TIMING

Voice systems exploit significant parallelism across stages. While one utterance is in ASR, previous utterances can be in TTS output.

### 5.1 Timing Diagram: Three Utterances in Flight

```
Time →

User 1:
Utterance:    ░░░░░░░░░                 (speech duration: 2s)
             [VAD]
                [ASR                    ]  (1s latency)
                   [LLM           ]         (0.5s latency)
                        [TTS           ]    (1s latency)
                             [Playback    ]

User 2:
Utterance:                     ░░░░░░░░░
                              [VAD]
                                 [ASR    ]
                                    [LLM  ]
                                        [TTS]

User 3:
Utterance:                                ░░░░░
                                         [VAD]
                                            [ASR]
```

In a multi-user system (e.g., voice AI agent handling multiple sessions), parallelism is automatic: each session has independent pipeline instances. On single-user device (e.g., smart speaker), parallelism comes from:

1. **Overlap within utterance**: While ASR processes, LLM can start early with partial results
2. **Early response generation**: LLM generates tokens before ASR completes (if using incremental/prefix caching)
3. **TTS pipelining**: Synthesize and buffer output while user still speaking

### 5.2 Incremental Processing for Latency

Modern systems process incrementally rather than waiting for complete utterances. Example ASR flow:

```
T=0ms:     Audio arrives 160 samples (10ms @ 16kHz)
T=10ms:    VAD processes, confidence <0.5 (silence)
T=20ms:    Audio continues, VAD updates
...
T=500ms:   Audio accumulated, VAD confidence > 0.8
           ASR starts consuming 1s of audio
T=1500ms:  ASR emits first token "What" (confidence 0.95)
           LLM can now start processing!

T=2000ms:  ASR emits second token "time" (confidence 0.93)
           LLM refines response context

T=2500ms:  ASR emits complete hypothesis: "What time is it?"
           LLM finishes first token of response
           TTS starts consuming first token

T=2800ms:  TTS outputs first audio (TTFA achieved!)
```

## 6. INTERRUPT & BARGE-IN ARCHITECTURE

Real-time voice systems must handle interruption: when the user speaks while the system is already speaking, the system should stop and respond to the user's input.

### 6.1 Barge-In Detection

Barge-in (user interrupting system output) requires:

1. **Simultaneous capture & playback**: Microphone active while speaker plays system response
2. **Echo cancellation**: Remove system's own audio from microphone input before VAD
3. **Barge-in detection**: Distinguish user speech (with system audio playing) from system audio alone

Architecture:

```
Microphone input:
  ├─ System audio (echo of what we're playing)
  └─ User speech (what we want to detect)
             ↓
      Echo Cancellation (AEC)
             ↓
      Residual (user speech only)
             ↓
      VAD with high sensitivity (barge-in mode)
             ↓
      If speech detected → Cancel playback, flush TTS queue
```

AEC (Acoustic Echo Cancellation) is critical for barge-in; poor AEC causes false positives (system cancels itself) or false negatives (misses user interruption). Modern neural AEC (DCCRN) achieves >95% echo suppression.

### 6.2 Cancellation Propagation

When user barges in, the system must cancel in-flight processing:

```cpp
class VoiceSession {
private:
  std::atomic<bool> should_interrupt{false};

public:
  void on_barge_in_detected() {
    should_interrupt.store(true, std::memory_order_release);

    // Stop TTS synthesis immediately
    tts_context->cancel();

    // Drain playback buffer
    playback_queue.clear();

    // Clear LLM context (or keep for context preservation)
    // Depends on product behavior

    // ASR continues processing user's new speech
    asr_context->reset_buffer();
  }

  void on_tts_frame() {
    if (should_interrupt.load(std::memory_order_acquire)) {
      return; // Don't produce more audio
    }
    // Normal TTS processing
  }
};
```

Propagation latency (time from VAD detecting barge-in to audio actually stopping) must be <200ms for natural feel. This requires:
- Real-time VAD (< 50ms latency)
- Lock-free queue drain (<10ms)
- Efficient audio buffer flush (<50ms to silence)

### 6.3 State Management During Interruption

Interruptions create complex state management problems:

**Problem 1: Partial LLM response in flight**
- If LLM generated 10 tokens but only 3 made it to audio, user hears incomplete sentence
- Solution: Never stream LLM output to TTS; buffer at least 1 sentence before synthesizing, or accept incomplete audio

**Problem 2: Context contamination**
- User interrupts system response; should LLM remember the interrupted context?
- Option A: Clear everything (stateless interruption) - simpler, user refocus
- Option B: Keep context (stateful interruption) - complex, but feels more natural
- Most products choose Option A for simplicity

**Problem 3: ASR confidence during user+system audio**
- When user speaks over system audio, ASR reliability degrades
- Either: (a) suppress ASR input until system finishes, (b) use higher confidence threshold, or (c) retrain ASR on degraded audio

## 7. LATENCY MEASUREMENT & INSTRUMENTATION

Production voice systems instrument every stage to measure latency distributions:

```cpp
struct PipelineMetrics {
  int64_t audio_capture_ts;      // µs since epoch
  int64_t vad_start_ts, vad_end_ts;
  int64_t asr_start_ts, asr_first_token_ts, asr_end_ts;
  int64_t llm_start_ts, llm_first_token_ts, llm_end_ts;
  int64_t tts_start_ts, tts_first_audio_ts, tts_end_ts;
  int64_t playback_start_ts;

  int64_t ttfa_ms() {
    return (tts_first_audio_ts - audio_capture_ts) / 1000;
  }
};
```

Metrics are logged to time-series database (Prometheus, InfluxDB) and queried for percentiles:
- `histogram_quantile(0.95, voice_ttfa_ms)` → 95th percentile TTFA
- `histogram_quantile(0.99, voice_ttfa_ms)` → 99th percentile (SLA threshold)

Alerts fire when P99 TTFA exceeds budget (e.g., >250ms).

## 8. ARCHITECTURAL VARIATIONS

### 8.1 Cloud-Based ASR/LLM

Some systems run VAD/TTS locally but stream audio to cloud ASR and send LLM inference to cloud servers:

```
Device:        VAD ──→ [Stream audio] ──→ Cloud ASR → LLM → [Stream tokens] → TTS
               ↑                                                                 ↓
               └─────────────────────────────────────────────────────────────────┘
```

Trade-off: Cloud ASR/LLM offer better accuracy but add network latency (50-100ms round-trip). Total TTFA increases to 300-500ms unless optimized (partial audio streaming, early token generation).

### 8.2 Hybrid Local-Cloud

Many commercial systems split:
- **Local**: VAD, lightweight ASR (on-device model), TTS
- **Cloud**: High-accuracy ASR, powerful LLM

This balances latency (VAD/TTS local for <200ms TTFA on local audio processing) and capability (cloud LLM for state-of-the-art responses).

### 8.3 Serverless/Batched Architecture

Backend voice AI (e.g., voice assistant transcription) can batch requests:

```
Device → Cloud endpoint ──batches 50 requests──→ GPU server (batch inference)
         (collect 10ms-100ms)                  → Returns 50 transcriptions
```

This prioritizes throughput over latency; acceptable for async transcription but not for real-time voice interaction.

## 9. IMPLEMENTATION CONSIDERATIONS

### 9.1 Thread Affinity & CPU Isolation

To meet latency targets, pin threads to dedicated CPU cores:

```cpp
void pin_thread_to_core(std::thread& t, int core_id) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core_id, &set);
  pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &set);
}

// Usage:
std::thread vad_thread(vad_process);
pin_thread_to_core(vad_thread, 1); // Core 1

std::thread asr_thread(asr_process);
pin_thread_to_core(asr_thread, 2); // Core 2
```

CPU isolation prevents scheduler preemption and cache thrashing.

### 9.2 Memory Layout & Cache Optimization

Audio buffers should be NUMA-local (on systems with multiple memory controllers):

```cpp
// Allocate ring buffer on NUMA node 0
int node = 0;
void* buffer = numa_alloc_onnode(buffer_size, node);
```

Ring buffer capacity should fit in L3 cache (typical 8-16MB per core) to prevent memory stalls during latency-critical operations.

### 9.3 GC Pauses & Language Selection

Java/Python GC pauses (100-500ms) are incompatible with <300ms TTFA targets. Production voice systems typically use:
- **C++**: Latency-critical components (VAD, ASR, TTS execution)
- **Python/Go/Rust**: Orchestration, non-latency-sensitive logic

Or: Use Java with continuous GC (ZGC, Shenandoah) and low pause targets (<10ms).

## 10. SUMMARY & DESIGN PRINCIPLES

**Key architectural principles for voice AI systems**:

1. **Streaming-first**: Every component must process incrementally, not wait for complete inputs
2. **Producer-consumer decoupling**: Use lock-free queues to decouple component latencies
3. **Parallelism exploitation**: Overlap stages aggressively (early LLM on partial ASR)
4. **Latency instrumentation**: Measure every stage; optimize P95/P99, not mean
5. **Interrupt handling**: Design state management for barge-in from ground up, not retrofit
6. **Low-latency runtime**: Use C++, lock-free data structures, CPU pinning, avoid GC pauses
7. **Model selection for latency**: Choose models optimized for streaming (RNN-T, FastSpeech), not accuracy-only (Whisper)
8. **Latency budgets**: Allocate 40-60ms VAD, 30-100ms ASR, 80-150ms LLM, 40-100ms TTS for <300ms TTFA

The voice pipeline is a coordinated optimization across hardware (CPU affinity, memory layout), software (lock-free concurrent data structures), and ML (streaming-optimized model architectures). Success requires deep integration across all layers.

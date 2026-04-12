# MODULE 35 — End-to-End Voice AI Optimization

## 1. INTRODUCTION & MOTIVATION

Optimizing an individual pipeline component to sub-50ms latency is insufficient if the total system latency exceeds 300ms. Voice AI systems require end-to-end optimization: every millisecond counts, and a single poorly-architected component destroys the user experience. This module addresses the system-level challenges: latency breakdown, pipeline warm-up patterns, stateful session management, and platform-specific optimizations for CPU servers, Apple Silicon, and Snapdragon processors.

The core engineering challenge is managing latency variance. A system achieving 150ms median TTFA but 400ms P99 TTFA creates frustrating user experience (one in hundred utterances feels broken). Production systems must optimize for 95th and 99th percentile latency, not mean.

This module is the capstone: we bring together VAD, ASR, LLM, TTS, and audio processing into complete systems targeting specific platforms and hardware constraints. We show step-by-step how to allocate the 200ms TTFA budget across components, measure end-to-end latency reliably, and debug latency bottlenecks in production.

## 2. 200MS TTFA DETAILED BREAKDOWN

### 2.1 Component Latency Allocation

For 200ms Time-To-First-Audio budget:

```
TTFA = VAD latency + ASR-first-token + LLM-TTFT + TTS-first-audio + overhead

Target allocation:
├─ Audio capture + VAD:          50ms (capture 10ms + VAD inference 40ms)
├─ ASR first token:              30ms (streaming buffer release threshold)
├─ LLM TTFT:                      80ms (prefill on small model)
├─ TTS first audio:              40ms (mel→vocoder)
└─ Overhead (switches, queues):  10ms
───────────────────────────────────────
Total:                           210ms (slightly over, acceptable with optimization)
```

### 2.2 Latency Waterfall Diagram

```
T=0ms:    User starts speaking
          ├─ Microphone capture starts

T=10ms:   First audio frame (160 samples @ 16kHz) captured
          ├─ Push to VAD ring buffer

T=20ms:   VAD accumulates 512ms context window
          ├─ VAD RNN inference starts

T=50ms:   VAD outputs confidence > 0.8
          ├─ Signal ASR to start processing

T=60ms:   ASR consumes VAD buffer, acoustic encoder processes
          ├─ 500ms audio accumulated

T=1050ms: ASR acoustic encoder completes
          ├─ RNN-T joint network processes first frames

T=1080ms: ASR outputs first token "What" (confidence 0.92)
          ├─ LLM context becomes: "<user> What"

T=1090ms: LLM prefill on first token + "What"
          ├─ Load model weights from memory (memory-bound)

T=1160ms: LLM first output token "The" (TTFT = 70ms from ASR token)
          ├─ Push to TTS ring queue

T=1170ms: TTS receives token "The"
          ├─ Accumulate tokens for 5-token window

T=1200ms: TTS has 5 tokens: "The current time is 3"
          ├─ FastSpeech2 text→mel (15ms)

T=1215ms: TTS mel-spectrogram ready
          ├─ Vocoder inference (10ms)

T=1225ms: TTS waveform complete (~0.4s audio)
          ├─ Push to playback buffer

T=1245ms: Playback buffer ready, audio sent to speaker
          ├─ TTFA = 1245 - 0 = 1245ms (FAILED target!)
```

**Problem diagnosis**: ASR latency is the bottleneck! The 500ms acoustic encoder dominates.

### 2.3 Aggressive Optimization for <200ms TTFA

To achieve 200ms TTFA, make aggressive trade-offs:

**VAD optimization**:
- Use Silero v5 lightweight variant (10-20ms inference)
- Start ASR at VAD confidence > 0.5 (more false positives, but faster)
- **New latency: 30-40ms**

**ASR optimization**:
- Use streaming Whisper with local agreement (emit first token at 100ms audio)
- Or use RNN-T with 100ms acoustic context (smaller encoder)
- **New latency: 100-150ms to first token** (key: emit early on partial audio)

**LLM optimization**:
- Use 3B model (Phi-2) with INT8 quantization
- Pre-warm KV cache with common prefixes ("What", "How", "Tell me")
- **New latency: 40-60ms TTFT**

**TTS optimization**:
- Use Vocos vocoder (ultra-fast iSTFT-based)
- Single-token TTS (synthesize "The" immediately, don't wait for 5-token window)
- **New latency: 20-30ms mel→audio**

**Revised budget**:
```
VAD:              40ms
ASR (first token): 100ms
  └─ (emit at 100ms audio, not complete sentence)
LLM (TTFT):       50ms
TTS (first audio): 25ms
───────────────────────────
Total:            215ms

But! We can overlap:
├─ While ASR processes 100ms audio → LLM can start with <100ms partial
├─ LLM and TTS can overlap (LLM outputs "The" → TTS starts)

Actual TTFA with overlap:
├─ ASR first token:  100ms
├─ LLM prefill:      40ms (overlaps with ASR, starts at 100ms)
├─ First LLM token:  100 + 40 = 140ms
├─ TTS vocoder:      20ms (overlaps with LLM)
├─ First audio:      140 + 20 = 160ms ✓
```

### 2.4 Latency Measurement Infrastructure

```cpp
struct PipelineMetrics {
  int64_t capture_start_ts;      // User speaks
  int64_t vad_detection_ts;      // VAD detects speech
  int64_t asr_start_ts;          // ASR begins
  int64_t asr_first_token_ts;    // ASR outputs first token
  int64_t llm_start_ts;          // LLM receives first token
  int64_t llm_first_token_ts;    // LLM outputs first token
  int64_t tts_start_ts;          // TTS receives first token
  int64_t tts_first_audio_ts;    // TTS outputs first audio
  int64_t playback_start_ts;     // Audio sent to speaker

  int64_t ttfa_ms() {
    return (tts_first_audio_ts - capture_start_ts) / 1000;
  }

  void log_breakdown() {
    printf("VAD latency:        %ldms\n", (vad_detection_ts - capture_start_ts) / 1000);
    printf("ASR latency:        %ldms\n", (asr_first_token_ts - asr_start_ts) / 1000);
    printf("LLM latency:        %ldms\n", (llm_first_token_ts - llm_start_ts) / 1000);
    printf("TTS latency:        %ldms\n", (tts_first_audio_ts - tts_start_ts) / 1000);
    printf("Total TTFA:         %ldms\n", ttfa_ms());
  }
};

class PipelineInstrumentation {
  PipelineMetrics metrics;

public:
  void record(const std::string& stage, const std::string& event) {
    int64_t now = get_time_microseconds();

    if (stage == "vad" && event == "detection") {
      metrics.vad_detection_ts = now;
    }
    if (stage == "asr" && event == "first_token") {
      metrics.asr_first_token_ts = now;
    }
    // ... etc ...

    // Log to metrics database (Prometheus, etc.)
    emit_metric("voice_pipeline_" + stage + "_" + event, now);
  }
};
```

## 3. PIPELINE WARM-UP & SESSION MANAGEMENT

### 3.1 Model Loading and JIT Compilation

Cold-start latency (first utterance) can be 2-3x slower than warm-start due to:
1. Model weight loading from disk
2. JIT compilation (if using compiled backends)
3. Kernel launch overhead

```cpp
class WarmupOptimization {
private:
  Silero vad_model;
  Sherpa asr_model;
  PhiLLM llm_model;
  FastSpeech2 tts_model;

public:
  void warmup() {
    // Preload all models into memory
    vad_model.load("silero_vad.onnx");
    asr_model.load("conformer.onnx");
    llm_model.load("phi_2.onnx");
    tts_model.load("fastspeech2.onnx");

    // JIT warm-up: run inference on dummy data
    std::vector<float> dummy_mel(128 * 512);  // 512 frames
    vad_model.forward(dummy_mel);

    std::vector<int> dummy_tokens = {1, 2, 3};  // Dummy text
    asr_model.forward(dummy_tokens);
    llm_model.forward(dummy_tokens);
    tts_model.forward(dummy_tokens);

    // CPU cache warmup
    std::vector<float> dummy_audio(16000);  // 1s of audio
    mel_computer.compute(dummy_audio.data(), dummy_audio.size());

    // Vocos warmup
    std::vector<float> dummy_mel2(128 * 100);
    vocoder.forward(dummy_mel2);

    printf("Pipeline warmup complete\n");
  }

  int64_t measure_cold_start_latency() {
    auto t0 = get_time_microseconds();
    // User speaks: trigger full pipeline
    // ...
    auto t1 = get_time_microseconds();
    return (t1 - t0) / 1000;  // ms
  }

  int64_t measure_warm_start_latency() {
    // After warmup
    auto t0 = get_time_microseconds();
    // User speaks
    // ...
    auto t1 = get_time_microseconds();
    return (t1 - t0) / 1000;
  }
};
```

Typical results:
- Cold start: 300-500ms TTFA
- Warm start: 150-200ms TTFA

### 3.2 Stateful Session Management

Voice systems maintain per-user session state:

```cpp
class VoiceSession {
private:
  std::string session_id;
  std::vector<Turn> conversation_history;
  LLMState llm_state;         // KV cache, context window
  VADState vad_state;         // Current VAD state
  bool in_interruption = false;

public:
  struct Turn {
    std::string user_text;
    std::string system_response;
    int64_t timestamp;
  };

  VoiceSession(const std::string& id) : session_id(id) {
    // Initialize state
    vad_state = VAD_SILENCE;
  }

  void on_user_speech_start() {
    if (!in_interruption) {
      vad_state = VAD_SPEECH;
      asr_context.reset();  // Start fresh ASR
      llm_context.reset_to_last_turn();  // Keep conversation history
    }
  }

  void on_asr_final(const std::string& user_text) {
    // Store turn
    conversation_history.push_back({
      user_text,
      "",  // System response (filled later)
      get_current_timestamp()
    });

    // Trigger LLM with full context
    generate_response(user_text);
  }

  void on_user_interrupt() {
    in_interruption = true;
    // Cancel in-flight TTS, flush playback buffer
    tts_context.cancel();
    // LLM will re-process on new input
  }

  void on_response_complete() {
    // Update last turn with full response
    if (!conversation_history.empty()) {
      conversation_history.back().system_response = llm_output;
    }
  }

private:
  void generate_response(const std::string& user_text) {
    // Build prompt with conversation history
    std::string prompt = build_prompt_with_context(user_text);

    // LLM inference with cached KV state
    llm_context.generate_streaming(prompt);
  }

  std::string build_prompt_with_context(const std::string& new_input) {
    std::string prompt = "<system> You are a helpful voice assistant.\n";

    // Include last 3 turns (tuning parameter)
    for (int i = std::max(0, (int)conversation_history.size() - 3);
         i < conversation_history.size(); i++) {
      const auto& turn = conversation_history[i];
      prompt += "<user> " + turn.user_text + "\n";
      prompt += "<assistant> " + turn.system_response + "\n";
    }

    prompt += "<user> " + new_input + "\n<assistant> ";
    return prompt;
  }
};

class SessionManager {
private:
  std::unordered_map<std::string, std::unique_ptr<VoiceSession>> sessions;

public:
  VoiceSession* get_or_create_session(const std::string& user_id) {
    if (sessions.find(user_id) == sessions.end()) {
      sessions[user_id] = std::make_unique<VoiceSession>(user_id);
    }
    return sessions[user_id].get();
  }

  void close_session(const std::string& user_id) {
    sessions.erase(user_id);
  }
};
```

## 4. TELEPHONY INTEGRATION

### 4.1 SIP & RTP for Voice Calls

Integrating voice AI with telephony (SIP/RTP):

```cpp
#include <pjlib.h>
#include <pjsua-lib/pjsua.h>

class TelephonyVoiceAI {
private:
  pjsua_call_id call_id;
  RingBuffer<int16_t> rx_buffer;  // From caller
  RingBuffer<int16_t> tx_buffer;  // To caller
  VoiceSession session;

public:
  void on_incoming_call(pjsua_call_id call) {
    call_id = call;
    session = VoiceSession::create();

    // Answer call
    pj_status_t status = pjsua_call_answer(call, 200, NULL, NULL);

    // Start processing
    processing_thread = std::thread([this] { process_call(); });
  }

private:
  void process_call() {
    while (pjsua_call_is_active(call_id, PJ_TRUE)) {
      // 1. Receive audio from caller (every 20ms)
      int16_t rx_frame[320];  // 20ms @ 16kHz
      int received = receive_rtp_frame(rx_frame, 320);

      // 2. Echo cancellation
      int16_t speaker_frame[320];
      int sent = get_recent_sent_frame(speaker_frame, 320);
      auto cleaned = aec.process(rx_frame, speaker_frame);

      // 3. VAD + ASR
      auto mel = mel_computer.compute(cleaned.data(), cleaned.size());
      session.on_audio_frame(mel);

      // 4. Generate response (streaming)
      auto response_token = session.get_next_token();
      if (response_token.size() > 0) {
        // TTS → audio
        auto speech = tts.synthesize(response_token);

        // 5. Send audio to caller (RTP)
        send_rtp_frame(speech.data(), speech.size());
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  int receive_rtp_frame(int16_t* buffer, size_t size) {
    // RTP packet received from network
    // Extract audio payload, handle SSRC, timestamp, sequence numbers
    // Manage jitter buffer

    return 0;  // frames received
  }

  int send_rtp_frame(int16_t* buffer, size_t size) {
    // Create RTP packet, send to caller
    // Handle timestamp, sequence number, payload type
    return 0;
  }
};
```

### 4.2 Jitter Buffer & RTP Codec Negotiation

```cpp
class RTPJitterBuffer {
private:
  std::deque<RTPPacket> buffer;
  uint32_t expected_seq = 0;
  int jitter_buffer_ms = 100;  // 100ms buffer
  int missing_packets = 0;

public:
  void on_rtp_packet(const RTPPacket& packet) {
    // Check for packet loss
    if (packet.seq != expected_seq) {
      missing_packets += (packet.seq - expected_seq) & 0xFFFF;
      // PLC (Packet Loss Concealment)
      for (int i = 0; i < missing_packets; i++) {
        buffer.push_back(generate_plc_packet());
      }
    }

    buffer.push_back(packet);
    expected_seq = packet.seq + 1;
  }

  std::vector<int16_t> pop_frame() {
    // Wait for jitter buffer to fill
    while (buffer.size() < jitter_buffer_ms * 16) {  // 16 samples per ms @ 16kHz
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // Dequeue frame
    std::vector<int16_t> frame;
    for (int i = 0; i < 320; i++) {
      if (!buffer.empty()) {
        auto& packet = buffer.front();
        frame.push_back(packet.audio[i]);
        if (packet.audio.size() <= 1) {
          buffer.pop_front();
        }
      }
    }

    return frame;
  }

private:
  RTPPacket generate_plc_packet() {
    // Comfort noise or last-frame extrapolation
    // Opus has built-in PLC support
    RTPPacket plc;
    plc.audio = opus_decoder.decode_plc();
    return plc;
  }
};
```

## 5. CPU SERVER DEPLOYMENT

### 5.1 100 Concurrent Sessions on CPU Server

Deploying 100 concurrent voice AI sessions on a single CPU server:

```cpp
class CPUServerVoiceAI {
private:
  std::vector<std::unique_ptr<VoiceSession>> sessions;
  std::vector<std::thread> processing_threads;
  static constexpr int NUM_SESSIONS = 100;
  static constexpr int NUM_WORKER_THREADS = 16;  // Physical cores / hyperthreading

public:
  void initialize() {
    for (int i = 0; i < NUM_SESSIONS; i++) {
      sessions.push_back(std::make_unique<VoiceSession>(std::to_string(i)));
    }

    // Distribute sessions across worker threads
    for (int t = 0; t < NUM_WORKER_THREADS; t++) {
      processing_threads.push_back(std::thread([this, t] {
        process_sessions(t);
      }));
    }
  }

private:
  void process_sessions(int thread_id) {
    // Pin thread to core
    cpu_set_t set;
    CPU_ZERO(&set);
    int core = thread_id;
    CPU_SET(core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);

    // Set NUMA local allocation
    numa_set_preferred(numa_node_of_cpu(core));

    // Process sessions assigned to this thread
    int start_idx = (thread_id * NUM_SESSIONS) / NUM_WORKER_THREADS;
    int end_idx = ((thread_id + 1) * NUM_SESSIONS) / NUM_WORKER_THREADS;

    while (true) {
      for (int i = start_idx; i < end_idx; i++) {
        if (sessions[i]->has_audio()) {
          sessions[i]->process_next_frame();
        }
      }
    }
  }
};
```

**Capacity planning**:
- CPU: 16 physical cores (Intel Xeon, AMD EPYC)
- Memory: 64GB DRAM
- Per-session resources:
  - Memory: 500MB-1GB (KV cache for LLM, audio buffers, model weights cached)
  - CPU: 5-10% per session (100ms processing window with 20ms frame)
- **Capacity: 100 concurrent sessions with 50-80% CPU utilization**

### 5.2 NUMA-Aware Memory Layout

```cpp
class NUMAAwareSessionManager {
private:
  std::vector<std::vector<std::unique_ptr<VoiceSession>>> sessions_per_node;

public:
  void initialize() {
    int num_nodes = numa_num_configured_nodes();

    sessions_per_node.resize(num_nodes);

    // Allocate sessions on local NUMA nodes
    for (int node = 0; node < num_nodes; node++) {
      int sessions_per_node_count = NUM_SESSIONS / num_nodes;
      for (int i = 0; i < sessions_per_node_count; i++) {
        // Allocate on specific NUMA node
        void* ptr = numa_alloc_onnode(sizeof(VoiceSession), node);
        sessions_per_node[node].push_back(
          std::unique_ptr<VoiceSession>(new(ptr) VoiceSession()));
      }
    }
  }

  void process_sessions_numa_local(int node_id) {
    // Process all sessions on this NUMA node
    // Memory access stays local (no inter-node traffic)
    for (auto& session : sessions_per_node[node_id]) {
      session->process_next_frame();
    }
  }
};
```

Benefit: NUMA-local processing prevents remote memory access (50-100ns latency vs. 1ns local), critical for 16-socket systems.

## 6. APPLE SILICON OPTIMIZATION (M4 PRO)

### 6.1 Component Allocation on M4 Pro

M4 Pro has:
- 12 CPU cores (8 P-cores, 4 E-cores)
- 16-core Neural Engine
- 20-core GPU
- Unified memory architecture

**Allocation strategy**:

```
Neural Engine (16 cores):
├─ VAD: Silero ONNX (always-on)
├─ ASR: Conformer encoder (1 utterance)
└─ TTS: VITS vocoder

GPU:
├─ LLM prefill (batched if multiple sessions)
├─ iSTFT for Vocos vocoder

CPU cores:
├─ Audio I/O (1-2 cores)
├─ LLM decode (1-2 cores per session, shared)
├─ Text encoding/decoding
├─ Session management
```

### 6.2 CoreML Deployment

```swift
import CoreML
import Metal

class M4VoiceAI {
    let vad_model: MLModel
    let asr_encoder: MLModel
    let tts_vocoder: MLModel

    init() {
        // Models compiled for Neural Engine
        vad_model = try! MLModel(contentsOf:
            Bundle.main.url(forResource: "vad_ne", withExtension: "mlmodelc")!)

        asr_encoder = try! MLModel(contentsOf:
            Bundle.main.url(forResource: "asr_ne", withExtension: "mlmodelc")!)

        tts_vocoder = try! MLModel(contentsOf:
            Bundle.main.url(forResource: "tts_ne", withExtension: "mlmodelc")!)
    }

    func process_audio(pcm: [Float]) -> String {
        // 1. VAD on Neural Engine
        let vad_input = MLMultiArray(shape: [1, 128, 512], dataType: .float32)
        let vad_pred = try! vad_model.prediction(input: vad_input)
        let vad_confidence = vad_pred.featureValue(for: "output")!.floatValue

        if vad_confidence < 0.5 {
            return ""  // No speech
        }

        // 2. ASR on Neural Engine (Conformer)
        let asr_input = MLMultiArray(shape: [1, 128, 100], dataType: .float32)
        let asr_pred = try! asr_encoder.prediction(input: asr_input)

        // 3. LLM on GPU (via MLProgram or custom Metal kernel)
        // ... LLM inference ...

        // 4. TTS vocoder on Neural Engine
        let tts_input = MLMultiArray(shape: [1, 128, 100], dataType: .float32)
        let tts_pred = try! tts_vocoder.prediction(input: tts_input)

        return "Response"
    }
}
```

**Performance on M4 Pro**:
- VAD: 5-10ms (Neural Engine)
- ASR: 30-50ms (Neural Engine)
- LLM: 50-80ms (GPU prefill, depends on model size)
- TTS: 15-25ms (Neural Engine vocoder)
- **Total: 100-165ms TTFA** ✓

## 7. SNAPDRAGON 8 ELITE OPTIMIZATION

### 7.1 Hexagon + Adreno Architecture

Snapdragon 8 Elite has:
- Kryo CPU (2.96 GHz)
- Adreno GPU (840 MHz)
- Hexagon DSP (with always-on processor)
- Sensor Processing Engine

**Allocation**:

```
Always-On Processor:
└─ Keyword spotting ("Hey Assistant")
   └─ <1mW power consumption

Hexagon DSP:
├─ VAD (Silero on NNAPI)
├─ Audio preprocessing (mel-spectrogram)
└─ Real-time audio effects

Adreno GPU:
├─ ASR (RNN-T)
├─ LLM (using NNAPI/QNN)
└─ TTS vocoder

Kryo CPU:
├─ LLM text generation (fallback if GPU busy)
├─ Session management
└─ Orchestration
```

### 7.2 Qualcomm Neural Processing SDK

```cpp
#include <QnnCpp.h>

class SnapdragonVoiceAI {
private:
  qnn::HTP htp_backend;    // Hexagon
  qnn::GPU gpu_backend;    // Adreno
  qnn::Model vad_model, asr_model, tts_model;

public:
  SnapdragonVoiceAI() {
    // Load models for different backends
    vad_model = qnn::compile("silero_vad.qnn", htp_backend);
    asr_model = qnn::compile("asr.qnn", gpu_backend);
    tts_model = qnn::compile("tts.qnn", htp_backend);
  }

  void process_audio(const std::vector<float>& pcm) {
    // 1. VAD on Hexagon
    auto vad_input = qnn::prepare_input(pcm);
    auto vad_output = vad_model.execute(vad_input);
    float confidence = vad_output.get_scalar();

    if (confidence < 0.5f) return;

    // 2. ASR on Adreno
    auto asr_input = qnn::prepare_input(pcm);
    auto asr_output = asr_model.execute(asr_input);
    std::string text = vad_output.get_string();

    // 3. LLM inference (CPU or GPU)
    std::string response = llm.generate(text);

    // 4. TTS on Hexagon
    auto tts_input = qnn::prepare_input(response);
    auto tts_output = tts_model.execute(tts_input);
    playback_audio(tts_output.get_audio());
  }
};
```

**Performance on Snapdragon 8 Elite**:
- VAD: 10-20ms (Hexagon DSP)
- ASR: 40-80ms (Adreno GPU)
- LLM: 60-100ms (Adreno GPU)
- TTS: 20-30ms (Hexagon DSP)
- **Total: 130-230ms TTFA** (within budget with optimization)

## 8. LATENCY DEBUGGING & ROOT CAUSE ANALYSIS

### 8.1 Instrumentation & Tracing

```cpp
class LatencyTracer {
private:
  struct Span {
    std::string name;
    int64_t start_time;
    int64_t end_time;
  };
  std::vector<Span> spans;

public:
  class ScopedSpan {
  private:
    LatencyTracer* tracer;
    std::string name;
    int64_t start_time;

  public:
    ScopedSpan(LatencyTracer* t, const std::string& n)
        : tracer(t), name(n) {
      start_time = get_time_microseconds();
    }

    ~ScopedSpan() {
      int64_t end_time = get_time_microseconds();
      tracer->record_span(name, start_time, end_time);
    }
  };

  void record_span(const std::string& name, int64_t start, int64_t end) {
    spans.push_back({name, start, end});
  }

  void print_trace() {
    for (const auto& span : spans) {
      int64_t duration = (span.end_time - span.start_time) / 1000;
      printf("%s: %ld ms\n", span.name.c_str(), duration);
    }
  }
};
```

**Usage**:
```cpp
{
  LatencyTracer::ScopedSpan span(&tracer, "vad_inference");
  vad_output = vad_model.forward(input);
}

{
  LatencyTracer::ScopedSpan span(&tracer, "asr_inference");
  asr_output = asr_model.forward(input);
}
```

Output:
```
vad_inference: 15 ms
asr_inference: 45 ms
llm_prefill: 78 ms
llm_decode: 8 ms
tts_vocoder: 22 ms
───────────
total: 168 ms ✓
```

### 8.2 Identifying Latency Regressions

```cpp
class LatencyMonitor {
private:
  std::deque<int64_t> latency_history;  // Last 1000 measurements
  int64_t latency_budget_ms = 200;
  float alert_threshold = 0.95;  // P95

public:
  void on_measurement(int64_t latency_ms) {
    latency_history.push_back(latency_ms);
    if (latency_history.size() > 1000) {
      latency_history.pop_front();
    }

    // Compute percentiles
    auto sorted = latency_history;
    std::sort(sorted.begin(), sorted.end());

    int64_t p50 = sorted[sorted.size() / 2];
    int64_t p95 = sorted[(int)(sorted.size() * 0.95)];
    int64_t p99 = sorted[(int)(sorted.size() * 0.99)];

    if (p95 > latency_budget_ms * alert_threshold) {
      alert("Latency regression: P95=" + std::to_string(p95) + "ms");
    }
  }
};
```

## 9. PRODUCTION MONITORING & OBSERVABILITY

### 9.1 Key Metrics for Voice AI

```cpp
class VoiceAIMetrics {
public:
  // Latency metrics
  prometheus::Histogram ttfa_histogram;      // TTFA distribution
  prometheus::Histogram ttft_histogram;      // LLM Time-to-First-Token
  prometheus::Histogram component_latency;  // Per-component breakdown

  // Quality metrics
  prometheus::Gauge asr_wer;                // Word Error Rate
  prometheus::Gauge tts_mos;                // Mean Opinion Score
  prometheus::Gauge llm_relevance;          // Response relevance

  // System metrics
  prometheus::Gauge sessions_active;        // Concurrent sessions
  prometheus::Counter audio_bytes_processed;
  prometheus::Gauge aec_erle;               // Echo Return Loss Enhancement

  void on_utterance_complete(const UtteranceMetrics& metrics) {
    ttfa_histogram.observe(metrics.ttfa_ms);
    component_latency.labels({"vad"}).observe(metrics.vad_latency_ms);
    component_latency.labels({"asr"}).observe(metrics.asr_latency_ms);
    component_latency.labels({"llm"}).observe(metrics.llm_latency_ms);
    component_latency.labels({"tts"}).observe(metrics.tts_latency_ms);

    asr_wer.set(metrics.word_error_rate);
    llm_relevance.set(metrics.relevance_score);
  }
};
```

### 9.2 Dashboarding

Typical Grafana dashboard for voice AI:

```
┌─────────────────────────────────────────────────┐
│  Voice AI System Dashboard                       │
├─────────────────────────────────────────────────┤
│                                                  │
│  TTFA (P50/P95/P99):  165ms / 215ms / 380ms   │
│  [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  │
│                                                  │
│  Component Latency Breakdown:                   │
│  ├─ VAD:       40ms (20%)                      │
│  ├─ ASR:       50ms (25%)                      │
│  ├─ LLM:       60ms (30%)                      │
│  ├─ TTS:       30ms (15%)                      │
│  └─ Overhead:  20ms (10%)                      │
│                                                  │
│  System Health:                                  │
│  ├─ Active Sessions: 47/100 (47%)              │
│  ├─ CPU Utilization: 65%                       │
│  ├─ Memory Usage: 32GB/64GB (50%)              │
│  ├─ ASR WER: 8.3%                              │
│  └─ TTS MOS: 4.2/5.0                           │
│                                                  │
│  Latency SLA:                                   │
│  ├─ P99 TTFA < 250ms: ✓ (380ms > 250ms) ⚠    │
│  ├─ P95 TTFA < 200ms: ✗ (215ms > 200ms) ⚠    │
│  └─ Audio Quality: ✓ (AEC ERLE = 32dB)        │
│                                                  │
└─────────────────────────────────────────────────┘
```

## 10. SUMMARY & DESIGN PRINCIPLES

**End-to-end voice AI optimization**:

1. **Latency budget**: Allocate 50ms VAD, 30-100ms ASR, 80ms LLM, 40ms TTS for 200ms TTFA
2. **Component overlap**: Start LLM before ASR finishes; start TTS before LLM completes
3. **Warm-up critical**: Cold-start latency 2-3x worse; pre-warm all models on startup
4. **Session state**: Maintain conversation history, KV cache state; enable multi-turn interaction
5. **Platform-specific**: Leverage Neural Engine (Apple), Hexagon (Snapdragon), AVX-512 (CPU)
6. **Monitoring obsession**: Track P95/P99 latency, not mean; measure every stage continuously
7. **Scaling model**: 100 concurrent sessions on 16-core CPU; NUMA-local processing essential
8. **Telephony**: SIP/RTP integration requires jitter buffer, echo cancellation, codec negotiation
9. **Debugging tools**: Instrumentation spans, latency histograms, regression alerts
10. **Trade-offs ruthless**: Accept 85% accuracy for 30ms latency; 95% accuracy with 200ms latency ruins UX

The difference between good voice AI (150-200ms TTFA, natural) and poor voice AI (400-500ms TTFA, frustrating) is system-level optimization: careful component tuning, intelligent resource allocation, continuous monitoring, and ruthless latency discipline. End-to-end optimization is the final frontier—where ML meets systems.

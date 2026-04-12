# MODULE 30 — VAD (Voice Activity Detection) Systems

## 1. INTRODUCTION & MOTIVATION

Voice Activity Detection stands as the critical gatekeeper for voice AI pipelines. A VAD system must answer a deceptively simple question: "Is this audio frame speech or silence?" But the challenge conceals itself in stringent real-world constraints:

1. **Latency**: Must detect speech onset within 100-200ms (human sensitivity to delay in voice response)
2. **Robustness**: Operate reliably across diverse acoustic environments (quiet home, noisy street, car, office)
3. **Compute efficiency**: Often runs on battery-powered devices, always-on use cases
4. **False positive rate**: <1% false positives (system shouldn't wake on coughs, keyboard typing, music)
5. **False negative rate**: <5% false negatives (shouldn't miss user speech)
6. **Streaming**: Process audio incrementally (100-200ms chunks) rather than waiting for complete utterances

VAD directly impacts user experience: too aggressive and the system wastes battery detecting false activations; too conservative and users must repeat themselves. Modern VAD achieves this balance using neural networks trained on diverse acoustic data, but the engineering challenge is deploying these models with <50ms latency on energy-constrained devices.

This module covers VAD algorithms from classical (WebRTC Gaussian Mixture Models) to modern neural approaches (Silero), streaming architecture that maintains accuracy with 20-40ms processing latency, CPU optimization techniques for ultra-low-power deployment, and edge architectures for always-on wake-word detection on specialized processors (Snapdragon Always-On, Apple Neural Engine).

## 2. VAD ALGORITHMS

### 2.1 Classical Approach: WebRTC GMM-Based VAD

The WebRTC Audio Processing Module (APM) implements a classical GMM-based VAD that remains competitive for many applications. The algorithm models the audio feature space as a mixture of speech and noise Gaussian distributions.

**Feature extraction**:
1. Compute MFCC (Mel-Frequency Cepstral Coefficients) from 10-30ms frames
2. Extract 13-dimensional feature vectors (energy + 12 MFCC coefficients)
3. Compute dynamic features (delta, delta-delta)

**Gaussian Mixture Model**:
- Two Gaussians: one for speech, one for noise
- Compute likelihood ratio: P(speech|features) / P(noise|features)
- Decision threshold: if likelihood ratio > τ, frame is speech

**Properties**:
- Latency: 10-30ms per frame
- CPU usage: ~0.5-1% on modern CPU core
- Accuracy: 85-90% on clean audio, degrades with noise
- Advantage: Interpretable, requires no DNN training
- Disadvantage: Noise robustness limited; poor on music/speech distinction

**Implementation sketch**:
```cpp
class WebRTCVAD {
private:
  std::vector<float> mfcc_buffer;  // 13 MFCCs
  GaussianModel speech_model, noise_model;

public:
  bool process_frame(const float* audio_frame, size_t frame_size) {
    // Extract MFCCs (10ms @ 16kHz = 160 samples)
    std::vector<float> features = extract_mfcc(audio_frame, frame_size);

    // Compute GMM likelihoods
    float p_speech = speech_model.likelihood(features);
    float p_noise = noise_model.likelihood(features);

    // Decision
    return (p_speech > p_noise) ? SPEECH : SILENCE;
  }
};
```

### 2.2 Modern Neural VAD: Silero

Silero VAD (developed by Silero AI) represents state-of-the-art neural VAD, achieving 95%+ accuracy across diverse environments. The model is a lightweight RNN trained on multilingual audio spanning 40+ languages.

**Architecture**:
- Input: Log-mel spectrogram (64 mel bins, 512ms context window)
- RNN layers: 2-3 stacked LSTM cells (128-256 hidden units)
- Output: Sigmoid → probability [0.0, 1.0] (speech likelihood)
- Model size: ~200KB (fp32), ~100KB (INT8 quantized)

**Key properties**:
- Latency: ~5-20ms per frame (depending on quantization, hardware)
- False positive rate: 0.1-0.5% on diverse data
- False negative rate: 1-3%
- Multilingual: Works across 40+ languages without retraining
- Streaming capable: Maintains internal state (RNN hidden states) across frames

**Silero v5 specifics** (as of 2024):
- Input frame duration: 512ms (optimal) or streaming at 20-32ms per frame
- Minimum context: 4 frames (80ms for real-time operation)
- Output: Confidence score + speaker embedding (for speaker verification)
- Checkpoint size: ~200KB (ONNX format)

### 2.3 Comparison: GMM vs Neural VAD

| Metric | WebRTC GMM | Silero Neural |
|--------|------------|---------------|
| Latency | 10-30ms | 5-20ms |
| CPU (single frame) | 0.5-1% | 2-5% |
| Accuracy (clean) | 85% | 97% |
| Accuracy (noisy) | 60-70% | 92-95% |
| Robustness to music | Poor | Good |
| Model size | <10KB | ~200KB |
| Language coverage | N/A (acoustic) | 40+ languages |
| Always-on capable | Yes | Yes (INT8) |
| Streaming support | Yes | Yes |

For general-purpose voice AI, Silero has become the industry standard. GMM-based VAD is preferred only in ultra-constrained scenarios (embedded systems with <1MB memory).

## 3. STREAMING VAD

### 3.1 Chunk-Based Streaming

Streaming VAD processes audio in small, overlapping chunks and maintains internal state (e.g., RNN hidden states) across chunks.

**Typical configuration**:
- Chunk size: 20-40ms (320-640 samples @ 16kHz)
- Overlap: 50% (sliding window)
- Mel spectrogram window: 512ms (non-overlapping frames used by RNN)

**Processing loop**:

```cpp
class StreamingVAD {
private:
  SileroModel model;
  std::vector<float> mel_spectrogram_buffer;  // 512ms accumulator
  RNNHiddenState rnn_state;  // Maintained across chunks
  static constexpr size_t CHUNK_SAMPLES = 320;  // 20ms @ 16kHz
  static constexpr size_t WINDOW_SAMPLES = 8192;  // 512ms @ 16kHz

public:
  float process_chunk(const float* audio_chunk, size_t chunk_size) {
    // Add to buffer
    mel_spectrogram_buffer.insert(
      mel_spectrogram_buffer.end(),
      audio_chunk, audio_chunk + chunk_size);

    // Once we have 512ms, run inference
    if (mel_spectrogram_buffer.size() >= WINDOW_SAMPLES) {
      // Extract mel spectrogram
      auto mel_spec = extract_mel_spectrogram(
        mel_spectrogram_buffer.data(), WINDOW_SAMPLES);

      // RNN inference (stateful)
      float confidence = model.forward(mel_spec, rnn_state);
      // rnn_state updated in-place

      // Slide buffer by 320ms (one chunk)
      mel_spectrogram_buffer.erase(
        mel_spectrogram_buffer.begin(),
        mel_spectrogram_buffer.begin() + CHUNK_SAMPLES);

      return confidence;
    }
    return -1.0f;  // Not enough data yet
  }
};
```

### 3.2 Latency vs Accuracy Tradeoff

Streaming VAD introduces a classical tradeoff:

**Larger window size** (512ms):
- Pros: More context, higher accuracy (95%+)
- Cons: Higher latency (speech onset delayed by ~256ms in worst case)

**Smaller window size** (40-80ms):
- Pros: Lower latency, faster speech detection (<100ms)
- Cons: Lower accuracy, more false positives

Production systems typically use 512ms effective window (accumulated into RNN context) but process every 20-40ms chunk, achieving low latency with high accuracy by relying on RNN's ability to detect state changes quickly.

### 3.3 Endpointing: End-of-Utterance Detection

VAD must not only detect speech onset but also signal when the user finishes speaking (end-of-utterance, EOU). This is critical for ASR pipelines: ASR can't output final results until it knows where the utterance ends.

**Simple approach**: Silence duration threshold
- If confidence < 0.5 for >1.0s, declare end-of-utterance
- Problems: Works poorly with natural speech pauses (breathing, thinking)

**Better approach**: Neural EOU prediction
- Train separate model to predict speech boundary from context
- Input: posterior speech probabilities over 2-3 second window
- Output: probability of sentence boundary in next 500ms
- Achieves 85-90% F1-score on utterance boundaries

**Heuristic hybrid approach** (most robust):
```cpp
class VADWithEndpointing {
private:
  float confidence_buffer[100];  // Last 2s @ 20ms frames
  int buffer_idx = 0;
  int silence_frames = 0;

public:
  VoiceActivityEvent process_frame(float confidence) {
    // Shift buffer
    confidence_buffer[buffer_idx] = confidence;
    buffer_idx = (buffer_idx + 1) % 100;

    // Check for silence
    if (confidence < 0.3f) {
      silence_frames++;
    } else {
      silence_frames = 0;
    }

    // Endpointing heuristics:
    // 1. Silence >1.2s → EOU (after speech detected)
    if (silence_frames > 60 && has_recent_speech()) {
      return {VOICE_ACTIVITY_EVENT_EOU};
    }

    // 2. Energy drops sharply (user stopped mid-word)
    float avg_recent = average_confidence(0, 10);  // Last 200ms
    float avg_before = average_confidence(40, 60);  // 800-1200ms ago
    if (avg_before > 0.7f && avg_recent < 0.3f) {
      return {VOICE_ACTIVITY_EVENT_EOU};
    }

    // 3. No update → Speech continues
    return {VOICE_ACTIVITY_EVENT_SPEECH, confidence};
  }

private:
  bool has_recent_speech() {
    for (int i = 0; i < 20; i++) {  // Last 400ms
      if (confidence_buffer[(buffer_idx - i + 100) % 100] > 0.7f) {
        return true;
      }
    }
    return false;
  }

  float average_confidence(int start_offset, int end_offset) {
    float sum = 0.0f;
    for (int i = start_offset; i <= end_offset; i++) {
      sum += confidence_buffer[(buffer_idx - i + 100) % 100];
    }
    return sum / (end_offset - start_offset + 1);
  }
};
```

## 4. VAD ON CPU: SILERO ONNX IMPLEMENTATION

### 4.1 ONNX Runtime Setup

Deploying Silero VAD on CPU efficiently requires ONNX Runtime with specific optimizations:

```cpp
#include <onnxruntime_cxx_api.h>

class SileroVAD_ONNX {
private:
  Ort::Env env;
  Ort::Session session;
  std::vector<float> mel_spec_buffer;

public:
  SileroVAD_ONNX(const std::string& model_path)
      : env(ORT_LOGGING_LEVEL_WARNING, "SileroVAD") {

    // Session options for CPU optimization
    Ort::SessionOptions session_options;

    // Thread settings
    session_options.SetIntraOpNumThreads(1);  // Single-threaded inference
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // CPU-specific optimizations
    OrtCUDAProviderOptions cuda_options;
    // For CPU: no CUDA needed

    session = Ort::Session(env, model_path.c_str(), session_options);
  }

  float process_frame(const float* audio, size_t audio_size) {
    // Extract mel spectrogram (512ms)
    std::vector<float> mel_spec = compute_mel_spectrogram(audio, audio_size);

    // Prepare inputs
    std::vector<float> input_data = mel_spec;
    std::vector<int64_t> input_shape = {1, 512, 64};  // batch, frames, mels

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
      input_data.data(), input_data.size(),
      input_shape.data(), input_shape.size());

    // Run inference
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    auto output_tensors = session.Run(
      Ort::RunOptions{nullptr},
      input_names, &input_tensor, 1,
      output_names, 1);

    // Extract confidence
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    return output_data[0];  // Confidence in [0, 1]
  }

private:
  std::vector<float> compute_mel_spectrogram(const float* audio, size_t size) {
    // Compute STFT → log-mel spectrogram
    // Assuming 512ms audio at 16kHz = 8192 samples
    // Output: 512 frames × 64 mel bins (one frame per 10ms roughly)
    // Implementation: use librosa or custom C++ code

    std::vector<float> mel_spec(512 * 64);
    // ... computation details ...
    return mel_spec;
  }
};
```

### 4.2 SIMD Optimization for Mel Filterbank

The mel spectrogram computation is the most CPU-intensive part of VAD. Optimizing it with SIMD (AVX-512, AVX2) can reduce latency by 3-5x.

**Standard implementation**:
```cpp
void mel_filterbank_scalar(const float* power_spectrum,  // 1025 bins
                          float* mel_spectrum) {        // 64 bins
  for (int mel_bin = 0; mel_bin < 64; mel_bin++) {
    float mel_value = 0.0f;
    for (int freq_bin = 0; freq_bin < 1025; freq_bin++) {
      float coeff = mel_filter_weights[mel_bin][freq_bin];
      mel_value += coeff * power_spectrum[freq_bin];
    }
    mel_spectrum[mel_bin] = std::log(mel_value + 1e-10f);
  }
}
```

**AVX-512 optimized version**:
```cpp
void mel_filterbank_avx512(const float* power_spectrum,
                           float* mel_spectrum) {
  const __m512 epsilon = _mm512_set1_ps(1e-10f);

  for (int mel_bin = 0; mel_bin < 64; mel_bin++) {
    __m512 mel_value = _mm512_setzero_ps();

    // Process 16 frequency bins per iteration (512-bit register / 32-bit float)
    for (int freq_bin = 0; freq_bin < 1025; freq_bin += 16) {
      if (freq_bin + 16 <= 1025) {
        __m512 power_vec = _mm512_loadu_ps(&power_spectrum[freq_bin]);
        __m512 coeff_vec = _mm512_loadu_ps(&mel_filter_weights[mel_bin][freq_bin]);
        mel_value = _mm512_fmadd_ps(coeff_vec, power_vec, mel_value);
      }
    }

    // Horizontal sum
    float mel_sum = _mm512_reduce_add_ps(mel_value);

    // Log
    mel_spectrum[mel_bin] = std::log(mel_sum + 1e-10f);
  }
}
```

**Performance**:
- Scalar: ~5-10µs per frame (10ms audio)
- AVX-512: ~1-2µs per frame
- Speedup: 5-10x

For a voice AI device processing 100 concurrent streams (e.g., call center), mel filterbank optimization is critical.

### 4.3 CPU Affinity & Cache Warmup

To achieve consistent <10ms VAD latency:

```cpp
void setup_vad_thread() {
  // Pin to core 0
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(0, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);

  // Set thread priority
  struct sched_param param;
  param.sched_priority = sched_get_priority_max(SCHED_FIFO) - 10;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

  // Warm up cache with model weights
  model.warmup();
}
```

## 5. VAD ON EDGE: ALWAYS-ON WAKE-WORD DETECTION

### 5.1 Ultra-Low-Power Always-On Systems

Wake-word detection (e.g., "Hey Siri", "Alexa") represents an extreme VAD variant: must listen continuously with <1mW power consumption, detecting a specific phrase while ignoring similar audio.

**Snapdragon Always-On** (Qualcomm):
- Dedicated low-power processor (not main CPU)
- Runs continuously with <1mW power draw
- Typical latency: 100-500ms to detect wake-word
- Model: Keyword spotting (KWS) using small neural network (<100KB)

**Architecture**:
```
Microphone → ADC (16-bit @ 16kHz)
           → Dedicated low-power processor
           → Mel spectrogram (on-processor)
           → Tiny neural network (KWS)
           → If "Alexa" detected → Wake main processor
           → Main processor runs full ASR/LLM pipeline
```

### 5.2 Keyword Spotting (KWS) Networks

KWS models are optimized for two constraints:
1. **Model size**: <100KB (must fit in always-on processor L3 cache)
2. **Latency**: <50ms per frame (ultra-tight power budget)

**Typical KWS architecture**:
- Input: 20-40ms mel spectrogram chunk
- Layers:
  - 2x Conv2D (16 filters, 3×3 kernel)
  - Batch norm
  - ReLU + 50% dropout
  - Flatten → 128-dim FC layer → softmax
- Output: 10-100 classes (one for each keyword + "silence")
- Model size: 50-100KB (quantized INT8)
- Latency: 2-5ms on low-power processor
- Accuracy: 95%+ on target keyword, <1% false positive rate on background speech

**Implementation sketch**:
```python
import tensorflow as tf

def build_kws_model(num_classes=10):
    """Keyword spotting model for always-on detection"""
    model = tf.keras.Sequential([
        # Input: (batch, time_steps, n_mels)
        tf.keras.layers.Input(shape=(40, 64)),  # 512ms @ 64 mels

        # Conv block 1
        tf.keras.layers.Conv1D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        # Conv block 2
        tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        # Global average pooling
        tf.keras.layers.GlobalAveragePooling1D(),

        # Dense
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # Output
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
```

### 5.3 Apple Neural Engine & "Hey Siri"

Apple's "Hey Siri" detection runs on the Neural Engine (a fixed-function ML accelerator in Apple SoCs).

**Architecture**:
- Always-on neural engine in A/M-series chips
- Runs wake-word detector in <1mW
- Latency: 200-500ms (acceptable for always-on)
- Model: ~1MB (larger than Snapdragon KWS, higher accuracy)

**Key differences from Snapdragon**:
- Apple processes audio at 16kHz continuous streaming
- Uses spectral+temporal context (500ms+ audio)
- Achieves <0.01% false positive rate on negative dataset
- Can recognize user-specific aspects of speech (personalization)

For voice AI developers, "always-on wake-word detection" is usually licensed technology (Apple, Amazon, Google); custom KWS models require significant effort to match commercial accuracy.

## 6. VAD STREAMING IMPLEMENTATION

### 6.1 Complete Streaming VAD Pipeline

```cpp
class StreamingVADPipeline {
private:
  SileroVAD_ONNX vad_model;
  std::vector<float> audio_buffer;
  std::vector<float> mel_buffer;
  VADWithEndpointing voice_detector;
  std::atomic<VADState> state{VAD_SILENCE};

  static constexpr size_t CHUNK_SAMPLES = 320;      // 20ms @ 16kHz
  static constexpr size_t MEL_WINDOW_SAMPLES = 8192; // 512ms @ 16kHz
  static constexpr float SPEECH_THRESHOLD = 0.5f;
  static constexpr float SILENCE_THRESHOLD = 0.3f;

public:
  StreamingVADPipeline() : vad_model("silero_vad.onnx") {}

  void on_audio_chunk(const float* chunk, size_t chunk_size) {
    // Add to buffer
    audio_buffer.insert(audio_buffer.end(), chunk, chunk + chunk_size);

    // Process when we have a full chunk
    if (audio_buffer.size() >= CHUNK_SAMPLES) {
      process_one_chunk();
      audio_buffer.erase(audio_buffer.begin(),
                        audio_buffer.begin() + CHUNK_SAMPLES);
    }
  }

private:
  void process_one_chunk() {
    // Accumulate mel spectrogram
    if (mel_buffer.size() < MEL_WINDOW_SAMPLES) {
      mel_buffer.insert(mel_buffer.end(),
                       audio_buffer.begin(),
                       audio_buffer.end());
      return;
    }

    // We have 512ms accumulated
    float confidence = vad_model.process_frame(mel_buffer.data(),
                                             mel_buffer.size());

    // Update state machine
    VADEvent event = voice_detector.process_frame(confidence);

    switch (event.type) {
      case VOICE_ACTIVITY_EVENT_SPEECH_START:
        state = VAD_SPEECH;
        notify_listeners(VAD_SPEECH_START);
        break;

      case VOICE_ACTIVITY_EVENT_SPEECH:
        if (state == VAD_SILENCE) {
          state = VAD_SPEECH;
          notify_listeners(VAD_SPEECH_START);
        }
        break;

      case VOICE_ACTIVITY_EVENT_EOU:
        if (state == VAD_SPEECH) {
          state = VAD_SILENCE;
          notify_listeners(VAD_SPEECH_END);
        }
        break;

      case VOICE_ACTIVITY_EVENT_SILENCE:
        break;
    }

    // Slide mel buffer for next frame
    mel_buffer.erase(mel_buffer.begin(),
                    mel_buffer.begin() + CHUNK_SAMPLES);
  }

  void notify_listeners(VADStateChange change) {
    // Trigger downstream processing (ASR, etc.)
  }
};
```

### 6.2 Multiple VAD Detectors in Parallel

Some systems run multiple VAD algorithms in parallel for robustness:

```cpp
class EnsembleVAD {
private:
  SileroVAD_ONNX silero;
  WebRTCVAD webrtc;

public:
  float get_confidence(const float* audio, size_t size) {
    float silero_conf = silero.process_frame(audio, size);
    bool webrtc_speech = webrtc.process_frame(audio, size);

    // Weight: Silero 80%, WebRTC 20%
    // WebRTC provides robustness; Silero primary accuracy
    float webrtc_conf = webrtc_speech ? 0.7f : 0.2f;
    return 0.8f * silero_conf + 0.2f * webrtc_conf;
  }
};
```

Benefits:
- Silero catches speech well but can hallucinate on background noise
- WebRTC is conservative (fewer false positives)
- Ensemble: fewer false positives + higher speech coverage

Overhead: 10-15% additional CPU (modest for VAD).

## 7. NOISE ROBUSTNESS & PREPROCESSING

### 7.1 Spectral Subtraction

To improve VAD accuracy in noisy environments, apply spectral subtraction before VAD:

```cpp
class SpectralSubtractionPreprocessor {
private:
  std::vector<float> noise_profile;  // Estimated noise spectrum
  float subtraction_factor = 2.0f;

public:
  void learn_noise_profile(const float* quiet_audio, size_t duration_samples) {
    // Estimate noise spectrum from quiet audio
    auto spectrum = compute_power_spectrum(quiet_audio, duration_samples);
    noise_profile = spectrum;
  }

  std::vector<float> process(const float* audio, size_t size) {
    auto spectrum = compute_power_spectrum(audio, size);

    // Spectral subtraction
    for (size_t i = 0; i < spectrum.size(); i++) {
      spectrum[i] = std::max(0.0f,
        spectrum[i] - subtraction_factor * noise_profile[i]);
    }

    // Back to time domain
    return istft(spectrum);
  }
};
```

Effectiveness: Improves VAD accuracy by 5-15% in noisy conditions (SNR 5-10dB).

### 7.2 Wiener Filtering

More sophisticated noise reduction:

```cpp
std::vector<float> wiener_filter(const float* noisy_audio,
                                 const std::vector<float>& noise_spectrum) {
  auto spectrum = compute_power_spectrum(noisy_audio, noisy_audio_size);

  std::vector<float> wiener_gain(spectrum.size());
  for (size_t i = 0; i < spectrum.size(); i++) {
    float signal_power = spectrum[i] - noise_spectrum[i];
    signal_power = std::max(0.0f, signal_power);
    wiener_gain[i] = signal_power / (spectrum[i] + 1e-10f);
  }

  // Apply gain to spectrum
  for (size_t i = 0; i < spectrum.size(); i++) {
    spectrum[i] *= wiener_gain[i];
  }

  return istft(spectrum);
}
```

Effectiveness: Improves VAD accuracy by 10-20% with 5-10dB SNR improvement; adds ~20-30ms latency.

## 8. ADVANCED TOPICS

### 8.1 Language-Specific VAD

Silero VAD works across 40+ languages without fine-tuning, but language-specific VAD models can improve accuracy:

- **Mandarin**: Chinese speech has different prosody; specialized models improve accuracy by 3-5%
- **Indian English**: Code-switching (mixing English and Hindi) confuses general VAD
- **Arabic**: Dialectal variations require regional models

For multilingual systems, run language identification first, then use language-specific VAD.

### 8.2 Speaker Embedding Integration

Some advanced VAD systems output speaker embeddings (fixed-dimensional vectors representing speaker identity). This enables:

1. **Speaker tracking**: Detect when different speakers talk (multiperson scenarios)
2. **Personalization**: Adjust VAD thresholds per speaker
3. **Spoofing detection**: Distinguish live speaker from playback (limited effectiveness)

Silero v5+ outputs 192-dimensional speaker embeddings alongside VAD confidence.

### 8.3 VAD for Non-Speech Detection

Variants of VAD detect:
- **Music detection**: Distinguish speech from background music
- **Crying/laughing**: Emotional vocalizations
- **Coughing/sneezing**: Environmental sounds

Implementation: Train additional classifiers on top of mel spectrograms.

## 9. PRODUCTION DEPLOYMENT PATTERNS

### 9.1 A/B Testing VAD Changes

VAD changes (new model version, threshold adjustment) must be validated before rollout:

```cpp
enum VADVariant {
  VAD_CONTROL,     // Current production model
  VAD_CANDIDATE,   // New model under test
};

class VADExperiment {
private:
  std::unordered_map<std::string, VADVariant> user_variants;
  SileroVAD_ONNX vad_control, vad_candidate;

public:
  float get_confidence(const std::string& user_id,
                      const float* audio, size_t size) {
    VADVariant variant = get_user_variant(user_id);

    float confidence;
    if (variant == VAD_CONTROL) {
      confidence = vad_control.process_frame(audio, size);
    } else {
      confidence = vad_candidate.process_frame(audio, size);
    }

    // Log confidence + ground truth (user tapped speech button?)
    log_vad_event(user_id, confidence, variant);

    return confidence;
  }
};
```

Metrics to track:
- False positive rate (false detections per hour)
- False negative rate (missed speech)
- Latency (P50, P95, P99)
- User satisfaction (A/B test)

### 9.2 Monitoring & Alerting

Production voice systems monitor VAD health continuously:

```cpp
void on_vad_result(float confidence) {
  vad_confidence_histogram.observe(confidence);

  if (vad_false_positive_rate > 0.02) {  // Alert if >2% FPR
    alert("VAD false positive rate exceeded: " +
          std::to_string(vad_false_positive_rate));
  }

  if (vad_latency_p99 > 50) {  // Alert if P99 latency >50ms
    alert("VAD latency degraded: P99=" +
          std::to_string(vad_latency_p99) + "ms");
  }
}
```

## 10. SUMMARY & DESIGN GUIDELINES

**Key takeaways**:

1. **Algorithm choice**: Use Silero neural VAD for general-purpose voice AI (95%+ accuracy, streaming-capable)
2. **Streaming architecture**: Process 20-40ms chunks; maintain RNN state across chunks
3. **Latency targets**: <50ms per frame for real-time VAD
4. **CPU optimization**: Use AVX-512 for mel filterbank (5-10x speedup); pin threads to cores
5. **Always-on design**: Keyword spotting on low-power processor + full VAD on main processor
6. **Robustness**: Combine Silero + WebRTC ensemble; add spectral subtraction for noisy environments
7. **Endpointing**: Critical for ASR; use hybrid heuristic (silence duration + neural EOU)
8. **Deployment**: Monitor FPR/FNR continuously; A/B test model updates before rollout

VAD is deceptively simple (binary decision: speech or silence) but operationally critical. Production quality requires careful algorithm selection, streaming architecture, CPU optimization, and continuous monitoring. The difference between excellent VAD (95%+ accuracy, <50ms latency) and poor VAD (85% accuracy, 100ms+ latency) is 10-20ms and a few percentage points of accuracy—but this transforms user experience from natural to frustrating.

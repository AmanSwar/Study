# MODULE 31 — Streaming ASR

## 1. INTRODUCTION & MOTIVATION

Automatic Speech Recognition (ASR) converts audio to text, serving as the interface between voice input and language understanding. Unlike classical speech recognition systems that waited for complete utterances before outputting results, modern voice AI demands streaming ASR: incremental recognition that emits partial hypotheses as audio arrives, progressively refining them.

This requirement cascades from voice interaction patterns. A user begins speaking; within 500-1000ms, ASR should output preliminary text (even if incomplete) so the LLM can begin processing context. As the user continues speaking, ASR emits updated hypotheses with increasing confidence. This streaming paradigm enables the 150-200ms TTFA targets that define natural voice experiences.

The technical challenge is profound: streaming ASR architectures (RNN-T, Conformer, Whisper) trade accuracy for latency compared to batch (offline) ASR. Batch systems like Whisper can process complete audio and emit final transcriptions with >95% accuracy; streaming systems must emit partial results with 10-20ms latency, accepting lower confidence and potential corrections.

This module covers the evolution from CTC (Connectionist Temporal Classification)—the foundational streaming ASR framework—through RNN-T (Recurrent Neural Network Transducer) that enables true streaming, to Whisper which offers competitive accuracy with lower latency through local agreement windowing. We detail the core algorithms, CPU implementation optimizing for <0.1x real-time factor (RTF), and integration patterns for voice AI pipelines.

## 2. ASR ARCHITECTURE EVOLUTION

### 2.1 CTC (Connectionist Temporal Classification)

CTC, introduced by Graves (2006), was the first practical streaming ASR architecture. It solves the alignment problem without requiring character-level labels in training: the model predicts output characters at each time step and a special blank token.

**CTC Architecture**:
```
Audio → Acoustic encoder (RNN/CNN) → Output sequence
        (processes variable-length input)

    ┌──────────────────────────┐
    │ Frame 1 Frame 2 ... Frame T │
    │   ↓       ↓       ↓        │
    │  [RNN states]             │
    │   ↓       ↓       ↓        │
    │   p(a)   p(b)   p(a)  ... │  (character probabilities)
    └──────────────────────────┘

Loss: CTC loss (dynamic programming over alignments)
Output: Sequence of characters with blanks (e.g., "aabbbbaa" → "aba")
```

**Frame-Synchronous Decoding**:
CTC processes frames independently; streaming decoding uses prefix beam search:

```cpp
class CTCStreaming {
private:
  std::vector<std::string> beam;  // Hypotheses
  int beam_width = 20;

public:
  std::string on_frame(const std::vector<float>& frame_logits) {
    // frame_logits: probabilities over alphabet + blank token
    // Update beam with new frame

    std::vector<std::string> new_beam;
    for (const auto& hyp : beam) {
      // For each hypothesis, extend with highest probability character
      char next_char = argmax(frame_logits);
      if (next_char != BLANK_TOKEN) {
        new_beam.push_back(hyp + next_char);
      } else {
        new_beam.push_back(hyp);  // Skip blank
      }
    }

    // Keep top beam_width hypotheses
    std::sort(new_beam.begin(), new_beam.end(),
              [](const auto& a, const auto& b) {
                return score(a) > score(b);
              });
    beam.assign(new_beam.begin(),
               new_beam.begin() + std::min((int)new_beam.size(),
                                          beam_width));

    return beam[0];  // Best hypothesis
  }
};
```

**CTC Properties**:
- Latency: 10-30ms per frame
- Accuracy: 80-85% on clean audio (significantly lower than offline systems)
- Streaming: Natural, frame-synchronous
- Drawback: Independent frame predictions → poor long-range dependencies

### 2.2 RNN-T (Recurrent Neural Network Transducer)

RNN-T, introduced by Graves (2012) and popularized by Google in Streaming RNN-T, extends CTC with explicit prediction of output tokens. It maintains internal state that captures long-range context.

**RNN-T Architecture**:
```
Audio encoder (RNN):           Text decoder (RNN):
  Frame 1 ──→ RNN state        Prev output ──→ RNN state
  Frame 2 ──→ (accumulate)     + Joint network → Output character
  ...

Joint Network (FC layers):
  [encoder_state, decoder_state] → softmax over alphabet + EOS
```

**RNN-T Inference**:
```cpp
class RNNTStreaming {
private:
  struct RNNTState {
    std::vector<float> encoder_state;  // From audio encoder
    std::vector<float> decoder_state;  // From text decoder
  } state;

  RNNTModel model;

public:
  std::vector<std::string> process_frame(const std::vector<float>& audio_frame) {
    // Encoder: process new audio frame
    state.encoder_state = model.encoder(audio_frame, state.encoder_state);

    // Decoding loop: generate tokens until EOS or max_tokens
    std::string current_hypothesis;
    std::vector<std::string> hypotheses;

    while (true) {
      // Joint network: combine encoder + decoder states
      auto [logits, new_decoder_state] = model.joint_network(
        state.encoder_state, state.decoder_state);

      // Greedy decoding (or beam search)
      int token_id = argmax(logits);
      if (token_id == EOS_TOKEN) {
        hypotheses.push_back(current_hypothesis);
        state.decoder_state = new_decoder_state;
        break;
      }

      // Extend hypothesis
      char token = id_to_char(token_id);
      if (token != BLANK_TOKEN) {
        current_hypothesis += token;
      }
      state.decoder_state = new_decoder_state;
    }

    return hypotheses;
  }
};
```

**RNN-T Properties**:
- Latency: 50-200ms (longer context window than CTC)
- Accuracy: 85-90% streaming (approaching offline systems)
- Streaming: True streaming; internal state maintains context
- Advantage: Superior to CTC; handles long-range dependencies
- Disadvantage: More complex; slower inference than CTC

### 2.3 Whisper with Streaming

OpenAI's Whisper is primarily an offline model but can be adapted for streaming using local agreement windowing: confidence becomes high when multiple frames agree on the same token.

**Streaming Whisper Strategy**:
```
1. Accumulate audio in 2-3s windows (overlap: 1s)
2. Run Whisper on each window
3. Use local agreement: emit token only when confidence stabilizes
4. Trade: 2-3s startup latency for 95%+ accuracy

Example:
T=0-3s:   Whisper decodes window[0:3s]
          Output: "what time is it now please"

T=1-4s:   Whisper decodes window[1:4s]
          Output: "what time is it now please help me"

Local agreement:
          [0:3s] outputs "what time is it"
          [1:4s] outputs "what time is it"
          → Emit "what time is it" (stable first 4 tokens)

          [1:4s] outputs "now please help"
          [2:5s] outputs "now please help me"
          → Emit "now please help me" (stable)
```

**Implementation**:
```cpp
class StreamingWhisper {
private:
  WhisperModel model;
  std::vector<float> audio_buffer;
  std::vector<std::string> previous_output;
  std::vector<float> previous_confidence;
  int stable_tokens = 0;  // How many tokens have stabilized

public:
  std::vector<std::string> process_audio_chunk(
      const float* chunk, size_t chunk_size) {

    audio_buffer.insert(audio_buffer.end(), chunk, chunk + chunk_size);

    // Process every 1s of new audio
    static constexpr size_t PROCESS_INTERVAL = 16000;  // 1s @ 16kHz

    std::vector<std::string> new_outputs;
    while (audio_buffer.size() >= PROCESS_INTERVAL) {
      // Get 3s window (with 1s overlap)
      size_t window_size = 48000;  // 3s @ 16kHz
      if (audio_buffer.size() < window_size) {
        window_size = audio_buffer.size();
      }

      // Whisper inference on window
      auto [output, confidence] = model.decode(
        audio_buffer.data(), window_size);

      // Local agreement: find stable prefix
      int agreement_length = find_agreement_length(
        previous_output, output,
        previous_confidence, confidence);

      // Emit only the newly stable tokens
      for (int i = stable_tokens; i < agreement_length; i++) {
        new_outputs.push_back(output[i]);
      }

      stable_tokens = agreement_length;
      previous_output = output;
      previous_confidence = confidence;

      // Slide buffer
      audio_buffer.erase(audio_buffer.begin(),
                        audio_buffer.begin() + PROCESS_INTERVAL);
    }

    return new_outputs;
  }

private:
  int find_agreement_length(const std::vector<std::string>& prev_output,
                           const std::vector<std::string>& curr_output,
                           const std::vector<float>& prev_conf,
                           const std::vector<float>& curr_conf) {
    int match_length = 0;
    for (int i = 0; i < prev_output.size() && i < curr_output.size(); i++) {
      if (prev_output[i] == curr_output[i] &&
          prev_conf[i] > 0.8f && curr_conf[i] > 0.8f) {
        match_length++;
      } else {
        break;
      }
    }
    return match_length;
  }
};
```

**Whisper Properties**:
- Latency: 2-4s (large, due to windowing)
- Accuracy: 95%+ (offline-level quality)
- Streaming: Pseudo-streaming via local agreement
- Advantage: Simplicity; offline model adapted
- Disadvantage: High latency; unsuitable for real-time dialogue

## 3. STREAMING ASR ON CPU

### 3.1 Real-Time Factor (RTF)

ASR performance is measured by Real-Time Factor (RTF): the ratio of inference time to audio duration.

```
RTF = inference_time / audio_duration

Examples:
- Streaming ASR RTF=0.05: 1s audio processes in 50ms (20x faster than real-time)
- Batch Whisper RTF=0.2: 60s audio processes in 12s

For voice AI:
- <0.1 RTF required (process 1s audio in <100ms)
- <0.05 RTF preferred (<50ms for responsive UI)
- >0.1 RTF (streaming falls behind) causes latency accumulation
```

### 3.2 Sherpa-ONNX: CPU Streaming ASR

Sherpa-ONNX (from Next-Gen Kaldi, k2 project) provides optimized streaming ASR on CPU using pre-trained models.

**Setup & Usage**:
```cpp
#include "sherpa-onnx/c-api/c-api.h"

class SherpaSpeechRecognizer {
private:
  SherpaOnnxRecognizer* recognizer;

public:
  SherpaSpeechRecognizer(const std::string& model_dir) {
    SherpaOnnxRecognizerConfig config;
    config.model_config.transducer.encoder_model_filename =
      model_dir + "/encoder.onnx";
    config.model_config.transducer.decoder_model_filename =
      model_dir + "/decoder.onnx";
    config.model_config.transducer.joiner_model_filename =
      model_dir + "/joiner.onnx";
    config.model_config.tokens_filename = model_dir + "/tokens.txt";
    config.num_threads = 1;
    config.provider = "cpu";

    recognizer = SherpaOnnxCreateRecognizer(&config);
  }

  std::string accept_audio(const float* samples, size_t num_samples) {
    SherpaOnnxAcceptWaveform(recognizer, 16000, samples, num_samples);

    // Check if we have partial results
    const char* partial = SherpaOnnxGetPartialResult(recognizer);
    return std::string(partial);
  }

  std::string finalize() {
    SherpaOnnxInputFinished(recognizer);
    const char* final = SherpaOnnxGetResult(recognizer);
    return std::string(final);
  }

  ~SherpaSpeechRecognizer() {
    SherpaOnnxDestroyRecognizer(recognizer);
  }
};
```

**Performance Characteristics**:
- Model size: 50-200MB (Conformer, RNN-T)
- Encoder RTF: 0.05-0.1 (10-20x faster than real-time)
- Accuracy: 85-92% on various languages

### 3.3 AVX-512 & SIMD Optimization

Streaming ASR models (RNN-T encoders, attention layers) benefit from SIMD vectorization. Key operations:

**Matrix Multiplication in FC Layers**:
```cpp
// Standard implementation
void fc_layer(const float* input, float* output,
              const float* weights, const float* bias,
              int input_size, int output_size) {
  for (int o = 0; o < output_size; o++) {
    float sum = bias[o];
    for (int i = 0; i < input_size; i++) {
      sum += weights[o * input_size + i] * input[i];
    }
    output[o] = sum;
  }
}

// AVX-512 optimized
void fc_layer_avx512(const float* input, float* output,
                     const float* weights, const float* bias,
                     int input_size, int output_size) {
  for (int o = 0; o < output_size; o++) {
    __m512 sum = _mm512_set1_ps(bias[o]);

    for (int i = 0; i < input_size; i += 16) {
      __m512 inp = _mm512_loadu_ps(&input[i]);
      __m512 w = _mm512_loadu_ps(&weights[o * input_size + i]);
      sum = _mm512_fmadd_ps(inp, w, sum);
    }

    // Horizontal sum
    output[o] = _mm512_reduce_add_ps(sum);
  }
}
```

**Softmax Computation**:
```cpp
// AVX-512 optimized softmax
void softmax_avx512(float* logits, int size) {
  // Find max (for numerical stability)
  __m512 max_val = _mm512_set1_ps(-std::numeric_limits<float>::max());
  for (int i = 0; i < size; i += 16) {
    __m512 v = _mm512_loadu_ps(&logits[i]);
    max_val = _mm512_max_ps(max_val, v);
  }
  float max_scalar = _mm512_reduce_max_ps(max_val);

  // Compute exp(x - max)
  float sum = 0.0f;
  for (int i = 0; i < size; i += 16) {
    __m512 v = _mm512_loadu_ps(&logits[i]);
    __m512 shifted = _mm512_sub_ps(v, _mm512_set1_ps(max_scalar));
    __m512 exp_val = _mm512_exp_ps(shifted);  // AVX-512 has exp
    sum += _mm512_reduce_add_ps(exp_val);
    _mm512_storeu_ps(&logits[i], exp_val);  // Store exp values
  }

  // Divide by sum
  __m512 inv_sum = _mm512_set1_ps(1.0f / sum);
  for (int i = 0; i < size; i += 16) {
    __m512 v = _mm512_loadu_ps(&logits[i]);
    __m512 normalized = _mm512_mul_ps(v, inv_sum);
    _mm512_storeu_ps(&logits[i], normalized);
  }
}
```

**Speedups**:
- FC layers: 3-5x (vectorization of dot product)
- Softmax: 5-8x (horizontal reductions)
- Overall encoder: 2-3x

## 4. WORD-BY-WORD STREAMING

### 4.1 Word Emission Heuristics

Instead of emitting characters/tokens, emit complete words when confidence stabilizes:

```cpp
class WordStreamingASR {
private:
  RNNTStreaming transducer;
  std::vector<std::string> token_buffer;
  std::vector<float> token_confidence;
  float confidence_threshold = 0.85f;

public:
  std::vector<std::string> process_frame(const std::vector<float>& audio_frame) {
    auto tokens = transducer.process_frame(audio_frame);

    // Add to buffer
    for (const auto& token : tokens) {
      token_buffer.push_back(token);
      token_confidence.push_back(transducer.get_confidence(token));
    }

    // Emit complete words when confident
    std::vector<std::string> emitted_words;
    while (!token_buffer.empty()) {
      // Check if we have a word boundary (space token or punctuation)
      if (is_word_boundary(token_buffer.back())) {
        // Check confidence of this word
        float word_confidence = compute_word_confidence(
          token_buffer.begin() + get_word_start(),
          token_buffer.end());

        if (word_confidence > confidence_threshold) {
          // Emit the word
          std::string word = tokens_to_word(
            token_buffer.begin() + get_word_start(),
            token_buffer.end());
          emitted_words.push_back(word);

          // Remove from buffer
          token_buffer.erase(token_buffer.begin() + get_word_start(),
                           token_buffer.end());
        } else {
          break;  // Not confident yet
        }
      } else {
        break;  // No word boundary
      }
    }

    return emitted_words;
  }

private:
  bool is_word_boundary(const std::string& token) {
    return token == " " || token == "." || token == "," ||
           token == "!" || token == "?";
  }

  float compute_word_confidence(
      std::vector<std::string>::iterator start,
      std::vector<std::string>::iterator end) {
    float avg_conf = 0.0f;
    int count = 0;
    for (auto it = start; it != end; ++it) {
      // Find confidence for this token
      int idx = std::distance(token_buffer.begin(), it);
      avg_conf += token_confidence[idx];
      count++;
    }
    return count > 0 ? avg_conf / count : 0.0f;
  }

  int get_word_start() {
    // Find last word boundary
    for (int i = token_buffer.size() - 1; i >= 0; i--) {
      if (is_word_boundary(token_buffer[i])) {
        return i + 1;
      }
    }
    return 0;
  }
};
```

Benefits:
- User sees complete words instead of partial tokens
- Better context for LLM
- Reduced need for correction

Drawback:
- Higher latency (wait for word completion)

## 5. CONFIDENCE SCORES & UNCERTAINTY

### 5.1 Token-Level Confidence

RNN-T and CTC models output logits (unnormalized scores) that become probabilities via softmax:

```cpp
float get_token_confidence(const std::vector<float>& logits) {
  // Softmax
  std::vector<float> probs = softmax(logits);

  // Top-1 confidence
  int top_idx = argmax(probs);
  float top_conf = probs[top_idx];

  // Margin: difference between top-1 and top-2
  std::sort(probs.begin(), probs.end(), std::greater<float>());
  float margin = probs[0] - probs[1];

  // Combined metric: confidence * margin
  return top_conf * (1.0f + margin);
}
```

### 5.2 Sequence-Level Confidence

For complete hypotheses, compute average token confidence:

```cpp
float get_hypothesis_confidence(const std::vector<float>& token_confidences) {
  float sum = 0.0f;
  for (float conf : token_confidences) {
    sum += conf;
  }
  return sum / token_confidences.size();
}

// Or: use entropy
float get_hypothesis_entropy(const std::vector<std::vector<float>>& logits) {
  float total_entropy = 0.0f;
  for (const auto& token_logits : logits) {
    auto probs = softmax(token_logits);
    float entropy = 0.0f;
    for (float p : probs) {
      if (p > 1e-10f) {
        entropy -= p * std::log(p);
      }
    }
    total_entropy += entropy;
  }
  return total_entropy / logits.size();
}
```

## 6. STREAMING ASR INTEGRATION IN VOICE PIPELINE

### 6.1 Latency Budget Allocation

For 200ms TTFA target, ASR consumes 30-50ms:

```
T=0ms:     User speaks "What time is it"
T=500ms:   VAD confidence > 0.8, ASR starts consuming audio
T=1000ms:  ASR processes 500ms of audio
           Emits token "What" (confidence 0.92)
           LLM starts with context: "<user> What"

T=1100ms:  ASR emits "time" (confidence 0.95)
           LLM refines response

T=1150ms:  ASR emits "is" (confidence 0.93)

T=1200ms:  ASR emits "it" (confidence 0.91)
           Hypothesis complete: "What time is it"
           LLM has full context, can generate final response

T=1250ms:  LLM outputs first token of response ("The")
           TTS starts synthesis (40-50ms startup latency)

T=1300ms:  TTS outputs first audio
           TTFA = 1300ms (MISSED target!)
```

**Problem**: ASR latency is the bottleneck. Solutions:

1. **Smaller encoder**: Lightweight Conformer (30M params → 100ms latency)
2. **Early emission**: Output tokens before complete sentence (aggressive streaming)
3. **Prefix caching**: Pre-cache common queries in LLM

### 6.2 Streaming ASR Error Handling

Streaming ASR outputs are provisional (may be corrected):

```cpp
enum ASREventType {
  ASR_PARTIAL,      // Intermediate hypothesis (may change)
  ASR_FINAL,        // Final result (unlikely to change)
  ASR_CORRECTION,   // Previous result corrected
};

struct ASREvent {
  ASREventType type;
  std::string text;
  float confidence;
  int start_token_idx;  // For tracking corrections
  int end_token_idx;
};

class ASRErrorHandler {
public:
  void on_asr_event(const ASREvent& event) {
    if (event.type == ASR_PARTIAL) {
      // Update LLM with provisional text
      // Don't commit to response yet
      update_llm_context_provisional(event.text);

    } else if (event.type == ASR_FINAL) {
      // Confirmed: update response
      update_llm_context_final(event.text);

    } else if (event.type == ASR_CORRECTION) {
      // Previous tokens corrected: rollback LLM state
      rollback_llm_to_token(event.start_token_idx);
      update_llm_with_corrected_text(event.text);
    }
  }
};
```

## 7. MULTILINGUAL & CODE-SWITCHING ASR

### 7.1 Multilingual Models

Whisper and Conformer-based models support 40-100 languages with a single model:

```python
# Fine-tuning for multilingual ASR
def build_multilingual_asr(num_languages=40):
    model = ConformerModel(
        input_dim=64,  # Mel bins
        encoder_dim=256,
        num_encoder_layers=12,
        output_vocab_size=5000 + num_languages,  # Include language tokens
    )
    return model

# Decoding with language annotation
hypothesis = model.decode(audio, language_id=LANGUAGE_EN)
# Or auto-detect: language_id = model.detect_language(audio)
```

### 7.2 Code-Switching

Some users switch languages mid-utterance ("Hello, 你好, how are you?"). Specialized ASR handles this:

```cpp
class CodeSwitchingASR {
private:
  std::unordered_map<std::string, Sherpa> language_models;
  LanguageDetector detector;

public:
  std::string decode_with_code_switching(const float* audio, size_t size) {
    // Segment audio by language
    auto segments = segment_by_language(audio, size);

    std::string result;
    for (const auto& [lang, seg_audio] : segments) {
      auto asr = language_models[lang];
      result += asr.decode(seg_audio.data(), seg_audio.size());
      result += " ";
    }
    return result;
  }

private:
  std::vector<std::pair<std::string, std::vector<float>>>
  segment_by_language(const float* audio, size_t size) {
    // Compute language embedding/confidence over time
    auto [lang_embeddings, confidence] = detector.run_sliding_window(
      audio, size, /*window=*/500 /*ms*/);

    // Cluster by language
    std::vector<std::pair<std::string, std::vector<float>>> result;
    // ... clustering logic ...
    return result;
  }
};
```

## 8. END-TO-END STREAMING ASR SYSTEM

### 8.1 Complete Inference Pipeline

```cpp
class VoiceASRPipeline {
private:
  Sherpa asr_model;
  std::vector<float> audio_buffer;
  std::vector<std::string> hypotheses;
  std::vector<float> confidences;

public:
  struct ASRResult {
    std::string partial_text;
    std::string final_text;
    std::vector<std::string> word_segments;
    std::vector<float> word_confidences;
  };

  ASRResult process_audio_chunk(const float* chunk, size_t chunk_size) {
    // Buffer audio
    audio_buffer.insert(audio_buffer.end(), chunk, chunk + chunk_size);

    // Run ASR every 100ms of new audio
    static constexpr size_t PROCESS_INTERVAL = 1600;  // 100ms @ 16kHz

    ASRResult result;
    while (audio_buffer.size() >= PROCESS_INTERVAL) {
      // Streaming ASR inference
      asr_model.accept_waveform(audio_buffer.data(), PROCESS_INTERVAL);

      // Get partial result
      result.partial_text = asr_model.get_partial_result();

      // Emit word-by-word
      auto words = extract_new_words(result.partial_text);
      for (const auto& word : words) {
        result.word_segments.push_back(word.text);
        result.word_confidences.push_back(word.confidence);
      }

      // Slide buffer
      audio_buffer.erase(audio_buffer.begin(),
                        audio_buffer.begin() + PROCESS_INTERVAL);
    }

    return result;
  }

  ASRResult finalize() {
    asr_model.input_finished();
    ASRResult result;
    result.final_text = asr_model.get_result();
    return result;
  }

private:
  std::vector<WordSegment> extract_new_words(const std::string& text) {
    // Extract words from text, skip already emitted
    // Implementation: tokenize, compute confidence per word
    std::vector<WordSegment> words;
    // ...
    return words;
  }
};
```

## 9. MONITORING & PRODUCTION DEPLOYMENT

### 9.1 ASR Quality Metrics

Track:
- **WER** (Word Error Rate): % of words incorrect
- **CER** (Character Error Rate): % of characters incorrect
- **Latency**: P50, P95, P99 TTFA per token
- **Confidence**: Average confidence score per session
- **Correction rate**: % of partial results later corrected

```cpp
void log_asr_metrics(const std::string& transcript,
                    const std::string& ground_truth,
                    float latency_ms,
                    float confidence) {
  int wer = compute_wer(transcript, ground_truth);
  metrics.asr_wer.observe(wer);
  metrics.asr_latency.observe(latency_ms);
  metrics.asr_confidence.observe(confidence);
}
```

### 9.2 Model Versioning & A/B Testing

```cpp
enum ASRModelVersion {
  ASR_V1_CONTROL,     // Production
  ASR_V2_CANDIDATE,   // Smaller encoder, lower latency
  ASR_V3_ACCURATE,    // Larger model, higher accuracy
};

class ASRModelSelector {
private:
  std::unordered_map<std::string, ASRModelVersion> user_variants;

public:
  Sherpa* get_model(const std::string& user_id) {
    auto variant = user_variants[user_id];
    switch (variant) {
      case ASR_V1_CONTROL:
        return &model_v1;
      case ASR_V2_CANDIDATE:
        return &model_v2;
      case ASR_V3_ACCURATE:
        return &model_v3;
    }
  }
};
```

## 10. SUMMARY & DESIGN PRINCIPLES

**Key architectural insights**:

1. **Algorithm choice**: Use RNN-T or Conformer for streaming ASR (superior to CTC); Whisper for offline accuracy
2. **Streaming design**: Emit tokens incrementally; use word-level segmentation for better UX
3. **Latency target**: 30-50ms per frame; aim for RTF < 0.1 (process 1s audio in <100ms)
4. **CPU optimization**: Use SIMD (AVX-512) for matrix ops (3-5x speedup); lightweight models for edge
5. **Confidence scores**: Use both token-level confidence and margin-based metrics
6. **Integration pattern**: Start LLM as soon as first ASR token available (don't wait for complete utterance)
7. **Error handling**: Treat partial results as provisional; handle corrections gracefully
8. **Monitoring**: Track WER, latency percentiles, confidence distribution; A/B test model updates

Streaming ASR is the bridge between raw audio and language understanding. The engineering challenge is balancing latency (emit tokens fast) with accuracy (emit correct tokens) while maintaining reasonable model size for edge deployment. Modern streaming systems achieve 85-92% accuracy with <100ms latency, enabling natural voice interaction.

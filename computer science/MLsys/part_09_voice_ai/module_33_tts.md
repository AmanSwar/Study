# MODULE 33 — TTS Systems Engineering

## 1. INTRODUCTION & MOTIVATION

Text-to-Speech (TTS) synthesis transforms LLM output into natural-sounding audio, completing the voice AI pipeline. Unlike classical TTS (concatenative, unit-selection) which simply chains pre-recorded audio segments, modern neural TTS uses deep learning to generate spectrograms or waveforms end-to-end, achieving human-level naturalness.

The engineering challenge is formidable: TTS must simultaneously optimize for latency (100-200ms startup for first audio), quality (natural prosody, speaker consistency), and efficiency (real-time streaming on CPU/mobile). A poorly implemented TTS can destroy an otherwise excellent voice system: 300ms TTS latency cascades into 500-600ms total TTFA, breaking natural conversation flow.

This module covers the evolution from sequence-to-sequence models (Tacotron2) through feed-forward non-autoregressive architectures (FastSpeech2, VITS) that enable low-latency synthesis, streaming TTS patterns that output audio incrementally, and vocoder designs (HiFi-GAN, Vocos) that convert spectrograms to waveforms in real-time. We detail CPU and mobile deployments achieving Real-Time Factor < 0.05 (synthesize 1 second of audio in <50ms).

## 2. TTS ARCHITECTURE EVOLUTION

### 2.1 Tacotron2: Sequence-to-Sequence Baseline

Tacotron2 (Shen et al., 2017) was the first end-to-end neural TTS producing natural speech:

```
Text input → Embedding layer → Encoder (bidirectional LSTM)
                               ↓
                        Attention mechanism
                               ↓
                    Decoder (LSTM) → Mel-spectrogram
                               ↓
                           Vocoder (WaveGlow)
                               ↓
                          Waveform (audio)
```

**Properties**:
- Latency: 50-200ms for encoding + decoding text to mel-spectrogram
- Vocoder latency: 500-2000ms (autoregressive, generates sample-by-sample)
- Total latency: 1-3 seconds (too slow for voice AI)
- Quality: 95%+ naturalness (state-of-the-art at 2017)
- Autoregressive: Decoder can't parallelize; output depends on all previous tokens

**Why too slow**: Autoregressive attention + autoregressive vocoder = exponential latency scaling with output length.

### 2.2 FastSpeech: Non-Autoregressive Synthesis

FastSpeech (Ren et al., 2019) eliminated autoregressive bottleneck by replacing attention with feed-forward Transformer blocks:

```
Text → Embedding → Encoder (stacked self-attention) → Length predictor
                                                              ↓
       Length predictor estimates output duration    (e.g., text "hello"
                                                       → 30 frames mel spec)
                      ↓
       Decoder (stacked self-attention) → Mel-spectrogram
                      ↓
                   Vocoder → Waveform
```

**Key innovation**: Length predictor eliminates need for attention alignment; every input token maps to fixed number of output frames.

**Properties**:
- Latency: 20-50ms for text-to-mel (10-100x faster than Tacotron2)
- Vocoder latency: 100-500ms (depends on vocoder choice)
- Total latency: 200-800ms
- Quality: 90-95% naturalness (slightly degraded from Tacotron2)
- Fully parallelizable: Can process all input tokens in parallel

### 2.3 VITS: Variational Inference-Based TTS

VITS (Kim et al., 2020) is a modern non-autoregressive TTS that combines:

1. **Encoder**: Text → latent variables (posterior distribution)
2. **VAE (Variational Autoencoder)**: Samples from distribution
3. **Decoder**: Latent variables → mel-spectrogram
4. **Flow**: Invertible transformation for better latent distributions
5. **Vocoder**: Integrated mel-to-waveform (within model)

```
Text → Encoder → VAE → Decoder → Flow → Vocoder → Waveform
       (LSTM)   (sample)  (Conv) (invertible)  (HiFi-GAN)
```

**Properties**:
- Latency: 30-100ms (text to waveform; includes vocoder)
- Quality: 97%+ naturalness (state-of-the-art 2020-2021)
- End-to-end: Vocoder integrated; no separate vocoder inference
- Efficient: 10-50MB model size (vs. 100MB+ for Tacotron2)
- Streaming: Can incrementally generate waveform chunks

**Key advantage for voice AI**: Extremely fast (50-100ms) with high quality.

### 2.4 VITS Architecture Deep Dive

```cpp
class VITS {
private:
  Encoder encoder;      // Text → posterior distribution
  Decoder decoder;      // Latent → mel-spectrogram
  FlowNetwork flow;     // Improve latent distributions
  VocoderNN vocoder;    // Mel → waveform

public:
  std::vector<float> synthesize(const std::string& text) {
    // 1. Encode text
    auto text_ids = tokenize(text);
    auto [mu, log_sigma] = encoder.forward(text_ids);

    // 2. Sample from posterior (VAE)
    auto z = sample_gaussian(mu, log_sigma);

    // 3. Flow transformation (improve sample quality)
    z = flow.forward(z);

    // 4. Decode to mel-spectrogram
    auto mel_spectrogram = decoder.forward(z);

    // 5. Vocoder: mel → waveform
    auto waveform = vocoder.forward(mel_spectrogram);

    return waveform;
  }
};
```

## 3. NON-AUTOREGRESSIVE TTS FOR LOW LATENCY

### 3.1 FastSpeech2 with Character-Level Control

FastSpeech2 adds explicit control over speech rate and pitch, enabling personalization:

```cpp
class FastSpeech2 {
private:
  TextEncoder encoder;
  DurationPredictor duration_predictor;
  PitchPredictor pitch_predictor;
  EnergyPredictor energy_predictor;
  SpectrogramDecoder decoder;

public:
  std::vector<float> synthesize(
      const std::string& text,
      float duration_scale = 1.0f,
      float pitch_scale = 1.0f,
      float energy_scale = 1.0f) {

    // 1. Encode text
    auto encoder_output = encoder.forward(text);

    // 2. Predict duration (number of frames per character)
    auto durations = duration_predictor.forward(encoder_output);
    // Scale duration for speed control
    for (auto& dur : durations) {
      dur = (int)(dur * duration_scale);
    }

    // 3. Predict pitch contour
    auto pitch = pitch_predictor.forward(encoder_output);
    for (auto& p : pitch) {
      p *= pitch_scale;  // Scale pitch up/down
    }

    // 4. Predict energy contour
    auto energy = energy_predictor.forward(encoder_output);
    for (auto& e : energy) {
      e *= energy_scale;
    }

    // 5. Length regulation: repeat each encoder output by duration
    auto expanded = expand_by_duration(encoder_output, durations);

    // 6. Add pitch and energy (conditioning)
    for (int i = 0; i < expanded.size(); i++) {
      expanded[i].pitch = pitch[i];
      expanded[i].energy = energy[i];
    }

    // 7. Decode to mel-spectrogram
    auto mel_spec = decoder.forward(expanded);

    return mel_spec;
  }

private:
  std::vector<FrameFeatures> expand_by_duration(
      const std::vector<FrameFeatures>& encoder_output,
      const std::vector<int>& durations) {
    std::vector<FrameFeatures> expanded;
    for (int i = 0; i < encoder_output.size(); i++) {
      for (int j = 0; j < durations[i]; j++) {
        expanded.push_back(encoder_output[i]);
      }
    }
    return expanded;
  }
};
```

**Advantages for voice AI**:
1. **Duration control**: Slow down/speed up speech (adaptive to user patience)
2. **Pitch control**: Vary intonation (add prosody)
3. **Energy control**: Vary volume (emotional expressiveness)
4. **Parallelizable**: All predictions independent of output frames

## 4. STREAMING TTS ARCHITECTURE

### 4.1 Chunk-Based Streaming Synthesis

Streaming TTS processes input tokens incrementally and outputs audio chunks as soon as enough context is available:

```cpp
class StreamingTTS {
private:
  FastSpeech2 synthesizer;
  std::deque<std::string> token_buffer;
  std::vector<float> pending_audio;
  std::vector<float> output_buffer;

public:
  void on_llm_token(const std::string& token) {
    token_buffer.push_back(token);

    // Synthesis every N tokens or at sentence boundary
    if (token_buffer.size() >= SYNTHESIS_WINDOW || is_sentence_end(token)) {
      synthesize_and_emit();
    }
  }

  std::vector<float> get_audio_chunk() {
    // TTS output thread calls this to get synthesized audio
    std::lock_guard<std::mutex> lock(output_lock);
    auto result = output_buffer;
    output_buffer.clear();
    return result;
  }

private:
  static constexpr size_t SYNTHESIS_WINDOW = 5;  // Synthesize every 5 tokens

  void synthesize_and_emit() {
    // Build text from buffered tokens
    std::string text;
    for (const auto& token : token_buffer) {
      text += token;
    }

    // Synthesize
    auto mel_spec = synthesizer.synthesize(text);

    // Vocoder: mel → waveform
    auto waveform = vocoder.forward(mel_spec);

    // Add to output buffer
    {
      std::lock_guard<std::mutex> lock(output_lock);
      output_buffer.insert(output_buffer.end(),
                          waveform.begin(), waveform.end());
    }

    // Clear buffer (or keep for next synthesis)
    // Option: Keep last 1-2 tokens for context
    if (token_buffer.size() > 2) {
      token_buffer.erase(token_buffer.begin(),
                        token_buffer.begin() + token_buffer.size() - 2);
    }
  }

  bool is_sentence_end(const std::string& token) {
    return token == "." || token == "!" || token == "?" ||
           token == "\n";
  }

  std::mutex output_lock;
};
```

### 4.2 Minimum Buffer Strategy

To minimize latency, buffer only the minimum needed for coherent synthesis:

```
Strategy 1: Phrase-level streaming
├─ Wait for ~5-10 words
├─ Synthesize phrase independently
├─ Output audio
├─ Risk: Prosody breaks between phrases (clicks, unnatural pauses)

Strategy 2: Sentence-level streaming (recommended)
├─ Wait for complete sentence (. ! ?)
├─ Synthesize full sentence
├─ Output audio
├─ Latency: ~500-1000ms (wait for user to finish sentence)
├─ Quality: Natural prosody within sentence

Strategy 3: Hybrid (aggressive)
├─ Buffer 3-5 words
├─ Synthesize with "sentence continuation" prediction
├─ Assume likely next words (common continuations)
├─ Output audio optimistically
├─ If user says something else → backtrack and resynthesise
├─ Risk: High if prediction wrong
├─ Latency: ~200-300ms
```

For voice AI, Strategy 2 (sentence-level) is most robust.

## 5. VOCODER DEEP DIVE

### 5.1 HiFi-GAN: Fast Neural Vocoder

HiFi-GAN (Kong et al., 2020) is the dominant neural vocoder for TTS, converting mel-spectrograms to high-quality waveforms in real-time.

**Architecture**:
```
Mel-spectrogram (128 bins × 100 frames)
         ↓
    Upsampling (4× 4× 4× 4) → 12.8k samples (0.8s @ 16kHz)
         ↓
  Multi-scale discriminator refinement
         ↓
  Waveform (16-bit PCM)
```

**Key properties**:
- Latency: 5-30ms per utterance (forward pass only)
- Quality: Indistinguishable from ground truth (95%+ in blind tests)
- Model size: 10-50MB
- Speed: 10-50x faster than previous vocoders (WaveGlow, WaveRNN)

**Implementation**:
```cpp
class HiFiGAN {
private:
  std::vector<Conv1dLayer> upsampling_layers;  // 4 layers
  std::vector<ResidualBlock> residual_blocks;

public:
  std::vector<float> forward(const std::vector<float>& mel_spec) {
    // Input: (batch=1, channels=128, frames=100)
    // Output: (batch=1, samples=16000)

    auto x = mel_spec;

    // Upsampling: 100 frames → 1600 samples (4x each)
    for (auto& layer : upsampling_layers) {
      x = layer.forward(x);  // Transpose conv
    }

    // Residual blocks: refine waveform quality
    for (auto& block : residual_blocks) {
      auto residual = block.forward(x);
      x = x + residual;  // Skip connection
    }

    // Tanh activation → [-1, 1]
    for (auto& sample : x) {
      sample = std::tanh(sample);
    }

    return x;
  }
};
```

### 5.2 Vocos: ISTFT-Based Fast Vocoder

Vocos (Siuzdak, 2023) represents a newer, even faster vocoding approach using invertible STFT:

**Key idea**: STFT is invertible; use fast neural network to predict magnitude/phase, then ISTFT to waveform.

```
Mel-spectrogram
        ↓
  Neural network (feature extraction)
        ↓
  Predict magnitude (from mel)
  Predict phase (learnable)
        ↓
  Complex spectrogram (magnitude + phase)
        ↓
  iSTFT (fast, non-neural)
        ↓
  Waveform
```

**Properties**:
- Latency: 2-5ms (significantly faster than HiFi-GAN due to simple NN)
- Quality: 93-97% (slightly below HiFi-GAN but excellent)
- Model size: 1-5MB (10x smaller than HiFi-GAN)
- Speed: Hardware-accelerated iSTFT available on CPU/GPU

**Advantage for voice AI**: Extreme speed; enables ultra-low latency TTS.

### 5.3 On-Device Vocoder Optimization

```cpp
class FastVocoder {
private:
  ONNXModel vocoder;  // Quantized (INT8) HiFi-GAN or Vocos
  FFTLibrary fft;     // For iSTFT if using Vocos

public:
  std::vector<float> synthesize(const std::vector<float>& mel_spec) {
    // Mel-spec: (128, frames)

    // 1. Neural network forward (quantized)
    auto neural_output = vocoder.forward(mel_spec);  // 5-15ms on CPU

    // 2. For HiFi-GAN: already waveform
    // For Vocos: need iSTFT (2-5ms)
    auto waveform = process_vocoder_output(neural_output);

    return waveform;
  }

private:
  std::vector<float> process_vocoder_output(
      const std::vector<float>& output) {
    // If using HiFi-GAN: direct output
    // If using Vocos: ISTFT of magnitude + phase

    // For Vocos (ISTFT-based):
    auto [magnitude, phase] = parse_vocoder_output(output);

    // Create complex spectrogram
    std::vector<std::complex<float>> spec;
    for (int i = 0; i < magnitude.size(); i++) {
      spec.push_back(magnitude[i] * std::exp(std::complex<float>(0, phase[i])));
    }

    // iSTFT (fast, vectorized)
    auto waveform = fft.istft(spec);

    return waveform;
  }
};
```

## 6. TTS ON CPU

### 6.1 ONNX Export & INT8 Quantization

```cpp
class CPUTTSPipeline {
public:
  CPUTTSPipeline() {
    // Load quantized models
    synthesizer = ONNXModel::load("fastspeech2_int8.onnx");
    vocoder = ONNXModel::load("vocos_int8.onnx");
  }

  std::vector<float> synthesize(const std::string& text) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. Text encoding (CPU, <1ms)
    auto text_ids = tokenize(text);

    // 2. FastSpeech2 inference (INT8, 10-30ms)
    auto mel_spec = synthesizer.forward(text_ids);

    // 3. Vocoder inference (Vocos INT8, 5-15ms)
    auto waveform = vocoder.forward(mel_spec);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    std::cout << "TTS latency: " << duration.count() << "ms\n";

    return waveform;
  }

private:
  ONNXModel synthesizer, vocoder;
};
```

**Performance on CPU** (single core, Intel i7-12700K):
- FastSpeech2 INT8 (text → mel): 15-30ms
- Vocos INT8 (mel → waveform): 5-10ms
- **Total**: 20-40ms per utterance
- **RTF**: ~0.05-0.1 (process 1s audio in 50-100ms)

### 6.2 AVX-512 SIMD for Convolution

TTS convolutions benefit from SIMD vectorization:

```cpp
void conv1d_avx512(const float* input, float* output,
                   const float* kernel, const float* bias,
                   int in_size, int kernel_size, int out_size) {
  for (int o = 0; o < out_size; o++) {
    __m512 sum = _mm512_set1_ps(bias[o]);

    for (int k = 0; k < kernel_size; k += 16) {
      __m512 in_vec = _mm512_loadu_ps(&input[o + k]);
      __m512 kernel_vec = _mm512_loadu_ps(&kernel[o * kernel_size + k]);

      sum = _mm512_fmadd_ps(in_vec, kernel_vec, sum);
    }

    output[o] = _mm512_reduce_add_ps(sum);
  }
}
```

Speedup: 3-5x for convolution-heavy TTS models.

### 6.3 oneDNN for HiFi-GAN

Intel's oneDNN library provides optimized convolution kernels:

```cpp
#include "dnnl.hpp"

class HiFiGANWithoneDNN {
private:
  dnnl::engine engine{dnnl::engine::kind::cpu, 0};
  std::vector<dnnl::primitive> primitives;

public:
  std::vector<float> forward(const std::vector<float>& mel_spec) {
    // oneDNN automatically optimizes convolutions
    // Uses AVX-512, cache optimization, etc.

    // Forward pass with oneDNN primitives
    for (auto& prim : primitives) {
      prim.execute(stream, args_map);
    }

    return output;
  }
};
```

Speedup: 2-3x on AVX-512 hardware.

## 7. TTS ON APPLE SILICON

### 7.1 CoreML + Metal for M4 Pro

```swift
import CoreML
import Metal

class AppleSiliconTTS {
    let synthesizer: MLModel
    let vocoder: MLModel
    let metalCommandQueue: MTLCommandQueue

    init() {
        // Load quantized models
        let synthesizer_url = Bundle.main.url(forResource: "fastspeech2",
                                             withExtension: "mlmodelc")!
        synthesizer = try! MLModel(contentsOf: synthesizer_url)

        let vocoder_url = Bundle.main.url(forResource: "vocos",
                                         withExtension: "mlmodelc")!
        vocoder = try! MLModel(contentsOf: vocoder_url)

        let device = MTLCreateSystemDefaultDevice()!
        metalCommandQueue = device.makeCommandQueue()!
    }

    func synthesize(text: String) -> [Float] {
        let t0 = Date()

        // 1. Tokenize (CPU)
        let tokens = tokenize(text)

        // 2. FastSpeech2 on Neural Engine or GPU
        // CoreML automatically routes to optimal hardware
        let melInput = MLMultiArray(shape: [1, 128, 100] as [NSNumber],
                                    dataType: .float32)
        // ... populate melInput ...

        let synthesizer_output = try! synthesizer.prediction(input: melInput)

        // 3. Vocos on GPU via Metal
        let vocoder_output = synthesize_with_metal(melSpec: melInput)

        let t1 = Date()
        print("TTS latency: \(t1.timeIntervalSince(t0) * 1000)ms")

        return vocoder_output
    }

    private func synthesize_with_metal(melSpec: MLMultiArray) -> [Float] {
        // Metal kernel for iSTFT (Vocos)
        // Runs on GPU compute units

        let commandBuffer = metalCommandQueue.makeCommandBuffer()!
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!

        // Dispatch compute kernel for iSTFT
        // ...

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return output_buffer
    }
}
```

**Apple Neural Engine**: Optimized for on-device models; 10-20x faster than CPU for neural inference.

### 7.2 ANE Performance Characteristics

```
M4 Pro Neural Engine:
├─ 16 cores (8 ML cores + 8 vector cores)
├─ Peak: 38 TFLOPS (FP16)
├─ Power: 1-2W
├─ Latency: 5-15ms for 1B-5B param models
├─ Bottleneck: I/O (moving data on/off ANE)

For FastSpeech2 (120M params):
├─ Compute: 2-5ms
├─ I/O overhead: 5-10ms
├─ Total: 7-15ms
```

## 8. TTS ON EDGE: PIPER

Piper is a lightweight, on-device TTS specifically designed for edge (RPi, Android, embedded):

```python
# Piper inference
import piper

# Load model (~20MB quantized)
synthesizer = piper.VoiceModel.load("en_US-amy-medium.onnx")

# Synthesize
audio = synthesizer.synthesize("Hello, how are you?",
                              speaker=0,  # Speaker ID
                              length_scale=1.0)  # Speed control

# Output: PCM audio, 22050Hz
```

**Properties**:
- Model size: 10-50MB (ONNX, quantized)
- Latency: 50-200ms (CPU inference)
- Quality: 85-90% naturalness (good, not perfect)
- Languages: 20+
- Voices: 50+ per language
- Runtime: Python (piper), C++ (piper-tts)

**Deployment on RPi4**:
```bash
# Install
pip install piper-tts

# Command-line synthesis
echo "Hello world" | piper --model en_US-amy-medium --output_file hello.wav
# Latency: ~3-5 seconds for "Hello world" (slow, but functional)
```

## 9. STREAMING TTS INTEGRATION IN VOICE PIPELINE

### 9.1 End-to-End Streaming TTS

```cpp
class VoicePipeline_TTS {
private:
  StreamingTTS tts;
  AudioPlayback playback;
  std::thread tts_thread, playback_thread;

public:
  void on_llm_token(const std::string& token) {
    // LLM emits token → TTS processes
    tts.on_llm_token(token);
  }

  void run_tts_thread() {
    while (is_running) {
      std::vector<float> audio = tts.get_audio_chunk();
      if (!audio.empty()) {
        playback.enqueue(audio);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  void run_playback_thread() {
    while (is_running) {
      auto audio_frame = playback.dequeue();
      if (!audio_frame.empty()) {
        playback.write_to_device(audio_frame);
      }
    }
  }
};
```

### 9.2 Latency Measurement

```cpp
void measure_tts_latency() {
  // TTFA budget:
  // VAD: 50ms
  // ASR: 50ms (first token)
  // LLM: 80ms (first token)
  // TTS: 100ms (first audio)
  // Total: 280ms (over 200ms budget!)

  // Optimize:
  // 1. Use lighter ASR (Whisper local, RTF=0.1)
  // 2. Use lighter LLM (3B model, TTFT=50ms)
  // 3. Use Vocos vocoder (5ms) instead of HiFi-GAN (20ms)
  // 4. Use sentence-level streaming (output first tokens early)

  // Revised:
  // VAD: 50ms
  // ASR: 30ms
  // LLM: 50ms
  // TTS: 30ms
  // Total: 160ms ✓
}
```

## 10. SUMMARY & DESIGN PRINCIPLES

**Key TTS engineering insights**:

1. **Architecture choice**: Use VITS or FastSpeech2 (non-autoregressive); avoid Tacotron2 for voice AI
2. **Vocoder selection**: Vocos for speed (<5ms); HiFi-GAN for quality (15-30ms) and streaming capability
3. **Latency target**: 30-100ms text-to-waveform (includes vocoder)
4. **Streaming pattern**: Emit tokens every 5-10 words; synthesize phrase-by-phrase
5. **CPU optimization**: INT8 quantization + SIMD (3-5x speedup); RTF < 0.1 achievable
6. **Apple Silicon**: Use CoreML + Metal; Neural Engine automatically routes 1B-5B models efficiently
7. **Edge deployment**: Piper for RPi/embedded; Vocos for ultra-constrained environments
8. **Quality vs latency**: For voice AI, latency dominates; 85% quality at 30ms beats 95% quality at 200ms
9. **Integration**: Streaming TTS feeds directly to audio playback; no buffering between components
10. **Monitoring**: Track TTFA, audio quality (MOS), naturalness; measure RTF on target hardware

TTS is the final component completing voice interaction. Unlike ASR which processes incremental input, TTS must balance latency (output first audio in 30-100ms) with quality (natural prosody across sentence boundaries). Modern non-autoregressive TTS with fast vocoders achieves this balance, enabling 150-250ms TTFA on commodity hardware.

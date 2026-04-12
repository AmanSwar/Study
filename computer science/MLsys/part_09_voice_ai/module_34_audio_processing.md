# MODULE 34 — Audio Processing & Codec Systems

## 1. INTRODUCTION & MOTIVATION

Audio processing is the foundation of voice AI: it converts raw microphone data into signal representation suitable for ML models (VAD, ASR, etc.) and manages output streams for playback with real-time echo cancellation. The engineering challenge lies in simultaneity and latency: voice systems must capture audio, process it for speech detection/recognition, and play back synthesized audio—all with sub-10ms latency variation.

Core requirements drive the design:

1. **Low latency**: <10ms audio capture-to-processing latency
2. **Real-time processing**: Mel filterbank, echo cancellation, automatic gain control must complete in <5ms per frame
3. **Synchronization**: Microphone and speaker must be synchronized within ±20ms for echo cancellation to work
4. **Sample rate conversion**: Convert between device rates (48kHz, 44.1kHz) and model rates (16kHz) with minimal quality loss
5. **Echo cancellation**: Remove system audio from microphone input (enabling barge-in detection)
6. **Codec support**: Opus, Speex for bandwidth-limited scenarios (telephony)

This module covers audio I/O abstractions (ALSA, CoreAudio, WASAPI), real-time mel filterbank computation optimized with SIMD (AVX-512, ARM NEON), sample rate conversion (polyphase filters), codec systems (Opus with packet loss concealment), WebRTC audio processing (AEC, NS, AGC), and neural acoustic echo cancellation (DCCRN) achieving >95% echo suppression.

## 2. AUDIO I/O FUNDAMENTALS

### 2.1 Cross-Platform Audio APIs

**Linux/ALSA (Advanced Linux Sound Architecture)**:
```cpp
#include <alsa/asoundlib.h>

class ALSAAudioCapture {
private:
  snd_pcm_t* capture_handle;
  snd_pcm_hw_params_t* hw_params;

public:
  ALSAAudioCapture(const std::string& device = "default") {
    snd_pcm_open(&capture_handle, device.c_str(),
                 SND_PCM_STREAM_CAPTURE, 0);

    snd_pcm_hw_params_malloc(&hw_params);
    snd_pcm_hw_params_any(capture_handle, hw_params);

    // Configure: 16-bit, mono, 16kHz
    snd_pcm_hw_params_set_access(capture_handle, hw_params,
                                 SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(capture_handle, hw_params,
                                 SND_PCM_FORMAT_S16_LE);
    snd_pcm_hw_params_set_channels(capture_handle, hw_params, 1);

    unsigned int rate = 16000;
    snd_pcm_hw_params_set_rate_near(capture_handle, hw_params, &rate, 0);

    // Period size: 10ms @ 16kHz = 160 samples
    snd_pcm_uframes_t period_size = 160;
    snd_pcm_hw_params_set_period_size_near(capture_handle, hw_params,
                                           &period_size, 0);

    snd_pcm_hw_params(capture_handle, hw_params);
  }

  std::vector<int16_t> read_frame() {
    int16_t buffer[160];
    int frames = snd_pcm_readi(capture_handle, buffer, 160);

    if (frames < 0) {
      snd_pcm_recover(capture_handle, frames, 0);
      frames = 0;
    }

    return std::vector<int16_t>(buffer, buffer + frames);
  }

  ~ALSAAudioCapture() {
    snd_pcm_close(capture_handle);
  }
};
```

**macOS/CoreAudio**:
```cpp
#include <AudioToolbox/AudioToolbox.h>

class CoreAudioCapture {
private:
  AudioUnit input_unit;
  std::function<void(const AudioBufferList*)> callback;

  // AudioUnit callback (C-style)
  static OSStatus AudioInputCallback(
      void* user_data,
      AudioUnitRenderActionFlags* flags,
      const AudioTimeStamp* timestamp,
      UInt32 bus,
      UInt32 frame_count,
      AudioBufferList* buffer_list) {

    auto* self = static_cast<CoreAudioCapture*>(user_data);

    // Render audio data
    AudioBuffer audio_buffer;
    audio_buffer.mNumberChannels = 1;
    audio_buffer.mDataByteSize = frame_count * sizeof(float);
    audio_buffer.mData = malloc(audio_buffer.mDataByteSize);

    AudioBufferList render_list;
    render_list.mNumberBuffers = 1;
    render_list.mBuffers[0] = audio_buffer;

    AudioUnitRender(self->input_unit, flags, timestamp, bus,
                   frame_count, &render_list);

    self->callback(&render_list);

    free(audio_buffer.mData);
    return noErr;
  }

public:
  CoreAudioCapture(std::function<void(const AudioBufferList*)> cb)
      : callback(cb) {

    // Setup audio session for iOS
    AVAudioSession* session = [AVAudioSession sharedInstance];
    [session setCategory:AVAudioSessionCategoryRecord error:nil];
    [session setActive:YES error:nil];

    // Create audio component
    AudioComponentDescription desc = {};
    desc.componentType = kAudioUnitType_Output;
    desc.componentSubType = kAudioUnitSubType_VoiceProcessingIO;
    desc.componentManufacturer = kAudioUnitManufacturer_Apple;

    AudioComponent audio_comp = AudioComponentFindNext(nullptr, &desc);
    AudioComponentInstanceNew(audio_comp, &input_unit);

    // Configure format: 16kHz, mono
    AudioStreamBasicDescription format = {};
    format.mSampleRate = 16000;
    format.mFormatID = kAudioFormatLinearPCM;
    format.mFormatFlags = kAudioFormatFlagIsFloat |
                          kAudioFormatFlagIsPacked;
    format.mBytesPerPacket = 4;
    format.mFramesPerPacket = 1;
    format.mBytesPerFrame = 4;
    format.mChannelsPerFrame = 1;
    format.mBitsPerChannel = 32;

    AudioUnitSetProperty(input_unit,
                        kAudioUnitProperty_StreamFormat,
                        kAudioUnitScope_Output, 1, &format,
                        sizeof(format));

    // Set callback
    AURenderCallbackStruct callback_struct = {};
    callback_struct.inputProc = AudioInputCallback;
    callback_struct.inputProcRefCon = this;

    AudioUnitSetProperty(input_unit,
                        kAudioOutputUnitProperty_SetInputCallback,
                        kAudioUnitScope_Output, 0,
                        &callback_struct, sizeof(callback_struct));

    AudioUnitInitialize(input_unit);
  }

  void start() {
    AudioOutputUnitStart(input_unit);
  }

  ~CoreAudioCapture() {
    AudioOutputUnitStop(input_unit);
    AudioComponentInstanceDispose(input_unit);
  }
};
```

**Windows/WASAPI**:
```cpp
#include <mmdeviceapi.h>
#include <audioclient.h>

class WASAPIAudioCapture {
private:
  IMMDeviceEnumerator* device_enum;
  IMMDevice* device;
  IAudioClient* audio_client;
  IAudioCaptureClient* capture_client;

public:
  WASAPIAudioCapture() {
    // COM initialization
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

    // Get device enumerator
    CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr,
                    CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
                    (void**)&device_enum);

    // Get default audio input device
    device_enum->GetDefaultAudioEndpoint(eCapture, eCommunications, &device);

    // Activate audio client
    device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr,
                    (void**)&audio_client);

    // Set format: 16-bit, mono, 16kHz
    WAVEFORMATEX format = {};
    format.wFormatTag = WAVE_FORMAT_PCM;
    format.nChannels = 1;
    format.nSamplesPerSec = 16000;
    format.nAvgBytesPerSec = 32000;
    format.nBlockAlign = 2;
    format.wBitsPerSample = 16;
    format.cbSize = 0;

    // Initialize client (exclusive mode for low latency)
    audio_client->Initialize(AUDCLNT_SHAREMODE_EXCLUSIVE,
                            AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                            10000,  // 1ms buffer
                            10000, &format, nullptr);

    // Get capture client
    audio_client->GetService(__uuidof(IAudioCaptureClient),
                            (void**)&capture_client);

    // Start capture
    audio_client->Start();
  }

  std::vector<int16_t> read_frame() {
    UINT32 packet_length = 0;
    capture_client->GetNextPacketSize(&packet_length);

    std::vector<int16_t> samples;
    BYTE* data;
    DWORD flags;

    if (SUCCEEDED(capture_client->GetBuffer(&data, &packet_length, &flags, nullptr, nullptr))) {
      int16_t* audio_data = reinterpret_cast<int16_t*>(data);
      samples.assign(audio_data, audio_data + packet_length);
      capture_client->ReleaseBuffer(packet_length);
    }

    return samples;
  }

  ~WASAPIAudioCapture() {
    audio_client->Stop();
    audio_client->Release();
    capture_client->Release();
    device->Release();
    device_enum->Release();
  }
};
```

### 2.2 Ring Buffer for Real-Time Audio

Ring buffers decouple producer (audio I/O device) from consumer (processing):

```cpp
template<typename T, size_t Capacity>
class AudioRingBuffer {
private:
  std::array<T, Capacity> buffer;
  std::atomic<size_t> write_index{0};
  std::atomic<size_t> read_index{0};

public:
  // Push samples from audio device
  bool push(const T* samples, size_t count) {
    size_t available = (Capacity - write_index + read_index) % Capacity;
    if (available < count) {
      return false;  // Buffer full; discard (or apply backpressure)
    }

    size_t write_end = write_index + count;
    if (write_end <= Capacity) {
      std::memcpy(&buffer[write_index], samples, count * sizeof(T));
    } else {
      // Wrap around
      size_t first_part = Capacity - write_index;
      std::memcpy(&buffer[write_index], samples, first_part * sizeof(T));
      std::memcpy(&buffer[0], samples + first_part,
                 (count - first_part) * sizeof(T));
    }

    write_index.store((write_index + count) % Capacity,
                     std::memory_order_release);
    return true;
  }

  // Pop samples for processing
  bool pop(T* samples, size_t count) {
    size_t available = (write_index - read_index + Capacity) % Capacity;
    if (available < count) {
      return false;  // Not enough data
    }

    size_t read_end = read_index + count;
    if (read_end <= Capacity) {
      std::memcpy(samples, &buffer[read_index], count * sizeof(T));
    } else {
      // Wrap around
      size_t first_part = Capacity - read_index;
      std::memcpy(samples, &buffer[read_index], first_part * sizeof(T));
      std::memcpy(samples + first_part, &buffer[0],
                 (count - first_part) * sizeof(T));
    }

    read_index.store((read_index + count) % Capacity,
                    std::memory_order_release);
    return true;
  }

  size_t available() const {
    return (write_index.load(std::memory_order_acquire) -
            read_index.load(std::memory_order_acquire) + Capacity) % Capacity;
  }
};
```

## 3. MEL FILTERBANK COMPUTATION

### 3.1 Mel Spectrogram Pipeline

```
Raw audio (PCM)
    ↓
Windowing (Hann window, 25ms frames)
    ↓
FFT (1024-point) → Power spectrum
    ↓
Mel filterbank (128 triangular filters)
    ↓
Log compression (20 * log10(power))
    ↓
Mel spectrogram (feature for ML models)
```

### 3.2 Scalar Implementation

```cpp
class MelSpectrogramCompute {
private:
  static constexpr size_t FFT_SIZE = 1024;
  static constexpr size_t N_MELS = 128;
  static constexpr float SAMPLE_RATE = 16000.0f;

  std::vector<float> mel_filterbank;  // 128 × 513 (freq bins)
  std::vector<float> hann_window;

public:
  MelSpectrogramCompute() {
    // Precompute Hann window
    hann_window.resize(FFT_SIZE);
    for (int i = 0; i < FFT_SIZE; i++) {
      hann_window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (FFT_SIZE - 1)));
    }

    // Precompute mel filterbank
    create_mel_filterbank();
  }

  std::vector<float> compute(const int16_t* audio, size_t frame_count) {
    // 1. Convert to float and normalize
    std::vector<float> float_audio(frame_count);
    for (size_t i = 0; i < frame_count; i++) {
      float_audio[i] = static_cast<float>(audio[i]) / 32768.0f;
    }

    // 2. Apply Hann window
    for (size_t i = 0; i < std::min(frame_count, FFT_SIZE); i++) {
      float_audio[i] *= hann_window[i];
    }

    // 3. FFT (use librosa or custom)
    auto power_spectrum = compute_power_spectrum(float_audio);

    // 4. Mel filterbank
    std::vector<float> mel_spec(N_MELS);
    for (int mel_idx = 0; mel_idx < N_MELS; mel_idx++) {
      float mel_value = 0.0f;
      for (int freq_idx = 0; freq_idx < power_spectrum.size(); freq_idx++) {
        float coeff = mel_filterbank[mel_idx * power_spectrum.size() + freq_idx];
        mel_value += coeff * power_spectrum[freq_idx];
      }
      // Log compression
      mel_spec[mel_idx] = 10.0f * std::log10(mel_value + 1e-10f);
    }

    return mel_spec;
  }

private:
  void create_mel_filterbank() {
    // Create triangular mel filters (standard procedure)
    mel_filterbank.resize(N_MELS * (FFT_SIZE / 2 + 1));

    // ... implementation (based on librosa or kaldi) ...
  }

  std::vector<float> compute_power_spectrum(const std::vector<float>& audio) {
    // Use FFTW or MKL for FFT
    // ... implementation ...
    return spectrum;
  }
};
```

Scalar latency: 5-10ms per frame @ 16kHz.

### 3.3 AVX-512 Optimized Mel Filterbank

```cpp
class MelSpectrogramAVX512 {
private:
  // Precomputed mel filterbank
  std::vector<__m512> mel_filterbank_avx;  // vectorized

public:
  std::vector<float> compute(const int16_t* audio, size_t frame_count) {
    // 1-3. Same as scalar (window, FFT)
    auto power_spectrum = compute_power_spectrum(audio, frame_count);

    // 4. Mel filterbank with AVX-512
    std::vector<float> mel_spec(128);

    for (int mel_idx = 0; mel_idx < 128; mel_idx++) {
      __m512 mel_value = _mm512_setzero_ps();

      // Process 16 frequency bins per iteration (512 / 32-bit)
      for (int freq_idx = 0; freq_idx < power_spectrum.size(); freq_idx += 16) {
        __m512 power_vec = _mm512_loadu_ps(&power_spectrum[freq_idx]);
        __m512 coeff_vec = mel_filterbank_avx[mel_idx * 32 + freq_idx / 16];

        mel_value = _mm512_fmadd_ps(coeff_vec, power_vec, mel_value);
      }

      // Horizontal reduction: sum across vector
      float sum = _mm512_reduce_add_ps(mel_value);
      mel_spec[mel_idx] = 10.0f * std::log10(sum + 1e-10f);
    }

    return mel_spec;
  }
};
```

AVX-512 latency: 1-2ms per frame (5-10x speedup).

### 3.4 ARM NEON for Mobile

```cpp
class MelSpectrogramNEON {
public:
  std::vector<float> compute(const int16_t* audio, size_t frame_count) {
    auto power_spectrum = compute_power_spectrum(audio, frame_count);

    std::vector<float> mel_spec(128);

    for (int mel_idx = 0; mel_idx < 128; mel_idx++) {
      float32x4_t mel_value = vdupq_n_f32(0.0f);

      // Process 4 frequency bins per iteration
      for (int freq_idx = 0; freq_idx < power_spectrum.size(); freq_idx += 4) {
        float32x4_t power_vec = vld1q_f32(&power_spectrum[freq_idx]);
        float32x4_t coeff_vec = vld1q_f32(&mel_filterbank[mel_idx * 513 + freq_idx]);

        // FMA: mel_value += coeff * power
        mel_value = vmlaq_f32(mel_value, coeff_vec, power_vec);
      }

      // Horizontal sum
      float sum = vaddvq_f32(mel_value);
      mel_spec[mel_idx] = 10.0f * logf(sum + 1e-10f);
    }

    return mel_spec;
  }
};
```

ARM NEON latency: 2-4ms per frame on ARM Cortex-A7x (mid-range mobile).

## 4. SAMPLE RATE CONVERSION

### 4.1 Polyphase Filter Banks

Sample rate conversion (resampling) uses polyphase filters for high quality:

```cpp
class SampleRateConverter {
private:
  std::vector<float> filter_coeffs;  // Polyphase filter
  std::deque<float> history;  // Previous samples

public:
  // Resample from 48kHz to 16kHz (1:3 ratio)
  std::vector<float> resample_48k_to_16k(const std::vector<float>& input) {
    std::vector<float> output;

    for (float sample : input) {
      history.push_back(sample);
      if (history.size() > 512) {
        history.pop_front();
      }

      // Every 3rd sample of input = 1 sample of output
      if (history.size() % 3 == 0) {
        float out_sample = 0.0f;
        for (int i = 0; i < filter_coeffs.size(); i++) {
          int idx = history.size() - 1 - i;
          if (idx >= 0 && idx < history.size()) {
            out_sample += filter_coeffs[i] * history[idx];
          }
        }
        output.push_back(out_sample);
      }
    }

    return output;
  }

private:
  void init_polyphase_filter() {
    // Design low-pass filter with Hamming window
    // Then create polyphase decomposition
    // Standard approach: use scipy.signal.remez for filter design
  }
};
```

Latency: 2-5ms (depends on filter order).

### 4.2 SAMPLERATE Library (libsamplerate)

For production, use optimized libraries:

```cpp
#include "samplerate.h"

class HighQualityResampler {
private:
  SRC_STATE* src_state;

public:
  HighQualityResampler(float input_rate, float output_rate) {
    int error;
    src_state = src_new(SRC_SINC_BEST, 1, &error);  // Best quality

    // Compute resampling ratio
    src_ratio = output_rate / input_rate;
  }

  std::vector<float> resample(const std::vector<float>& input) {
    SRC_DATA data;
    data.data_in = const_cast<float*>(input.data());
    data.input_frames = input.size();
    data.data_out = new float[input.size() * 2];  // Allocate output buffer
    data.output_frames = input.size() * 2;
    data.src_ratio = src_ratio;

    src_process(src_state, &data);

    std::vector<float> output(data.data_out, data.data_out + data.output_frames_gen);
    delete[] data.data_out;

    return output;
  }

private:
  double src_ratio;
};
```

Latency: 1-3ms; quality: indistinguishable from original.

## 5. OPUS CODEC FOR VOICE

### 5.1 Opus Basics

Opus is the modern standard for voice transmission (used in WebRTC, Discord, Telegram):

```cpp
#include <opus/opus.h>

class OpusEncoder {
private:
  OpusEncoder* opus_enc;
  OpusDecoder* opus_dec;
  static constexpr int SAMPLE_RATE = 16000;
  static constexpr int BITRATE = 24000;  // 24kbps

public:
  OpusEncoder() {
    int error;
    opus_enc = opus_encoder_create(SAMPLE_RATE, 1, OPUS_APPLICATION_VOIP, &error);
    opus_decoder_create(SAMPLE_RATE, 1, &error);

    opus_encoder_ctl(opus_enc, OPUS_SET_BITRATE(BITRATE));
    opus_encoder_ctl(opus_enc, OPUS_SET_COMPLEXITY(10));  // Max quality
  }

  std::vector<uint8_t> encode(const float* pcm, size_t frame_count) {
    // Convert float to int16
    std::vector<int16_t> int_pcm(frame_count);
    for (size_t i = 0; i < frame_count; i++) {
      int_pcm[i] = static_cast<int16_t>(pcm[i] * 32767.0f);
    }

    // Encode
    std::vector<uint8_t> encoded(4000);  // Max frame size
    int bytes = opus_encode(opus_enc, int_pcm.data(), frame_count,
                           encoded.data(), encoded.size());

    encoded.resize(bytes);
    return encoded;
  }

  std::vector<float> decode(const uint8_t* encoded, size_t encoded_size) {
    std::vector<int16_t> decoded(2880);  // Max frame
    int frame_count = opus_decode(opus_dec, encoded, encoded_size,
                                 decoded.data(), 2880, 0);

    // Convert int16 to float
    std::vector<float> float_pcm(frame_count);
    for (int i = 0; i < frame_count; i++) {
      float_pcm[i] = static_cast<float>(decoded[i]) / 32768.0f;
    }

    return float_pcm;
  }

  ~OpusEncoder() {
    opus_encoder_destroy(opus_enc);
    opus_decoder_destroy(opus_dec);
  }
};
```

**Opus properties**:
- Bitrate: 6-510 kbps (configurable)
- Latency: 5-20ms (depends on frame size)
- Quality: 24kbps achieves 90%+ speech quality
- Efficiency: 3-5x smaller than raw PCM

### 5.2 Packet Loss Concealment (PLC)

Opus has built-in PLC for handling lost packets:

```cpp
void decode_with_plc(const std::vector<uint8_t>& encoded_packets) {
  OpusDecoder* dec = opus_decoder_create(16000, 1, nullptr);

  for (const auto& packet : encoded_packets) {
    if (packet_is_lost) {
      // PLC: decode with NULL input (generates comfort noise)
      std::vector<int16_t> output(960);  // 60ms frame
      opus_decode(dec, nullptr, 0, output.data(), 960, 1);
      // output contains comfort noise, not garbage
    } else {
      // Normal decoding
      std::vector<int16_t> output(960);
      opus_decode(dec, packet.data(), packet.size(), output.data(), 960, 0);
    }
  }
}
```

Result: Loss of 10-20% packets barely noticeable to users (vs. 30-50% degradation with basic silence substitution).

## 6. WEBRTC AUDIO PROCESSING

### 6.1 Echo Cancellation (AEC)

WebRTC's AEC is a mature, production-grade echo canceller:

```cpp
#include "api/audio/audio_processing.h"

using namespace webrtc;

class WebRTCAudioProcessor {
private:
  rtc::scoped_refptr<AudioProcessing> apm;

public:
  WebRTCAudioProcessor() {
    AudioProcessing::Config config;
    config.echo_canceller.enabled = true;
    config.noise_suppression.enabled = true;
    config.automatic_gain_control.enabled = true;

    apm = AudioProcessingBuilder().Create();
    apm->ApplyConfig(config);

    // Set sample rate
    apm->SetInputAnalogLevel(100);
  }

  void process_capture(const float* microphone_data,
                      const float* playback_data,
                      size_t frame_count) {
    // Set up streams
    StreamConfig config(16000, 1, false);

    // Process
    apm->ProcessStream(const_cast<float*>(microphone_data),
                      config, config, nullptr);

    // Echo reference: signal from speaker
    apm->ProcessReverseStream(const_cast<float*>(playback_data),
                             config, config, nullptr);

    // Output: microphone_data now has echo removed
  }
};
```

**Echo cancellation performance**:
- ERLE (Echo Return Loss Enhancement): 20-30dB (removes 99.9% of echo)
- Latency: <5ms
- Robustness: Works across diverse speaker+microphone pairs

### 6.2 Neural Acoustic Echo Cancellation (DCCRN)

DCCRN (Dual-Channel Conformer Recurrent Network) uses neural networks for state-of-the-art AEC:

```cpp
class DCCRNEchoCanceller {
private:
  ONNXModel dccrn_model;  // DCCRN v3, quantized
  std::vector<float> speaker_features;
  std::vector<float> microphone_features;

public:
  std::vector<float> process(const std::vector<float>& microphone_audio,
                            const std::vector<float>& speaker_audio) {
    // 1. Extract features from both channels
    auto speaker_mel = extract_mel_spectrogram(speaker_audio);
    auto micro_mel = extract_mel_spectrogram(microphone_audio);

    // 2. Stack features [speaker_mel, microphone_mel]
    std::vector<float> input;
    input.insert(input.end(), speaker_mel.begin(), speaker_mel.end());
    input.insert(input.end(), micro_mel.begin(), micro_mel.end());

    // 3. DCCRN inference
    auto output = dccrn_model.forward(input);  // [0, 1] mask

    // 4. Apply mask to microphone spectrogram
    auto enhanced_mel = apply_mask(micro_mel, output);

    // 5. iSTFT back to waveform
    auto enhanced_audio = istft(enhanced_mel);

    return enhanced_audio;
  }

private:
  std::vector<float> apply_mask(const std::vector<float>& mel,
                               const std::vector<float>& mask) {
    std::vector<float> masked(mel.size());
    for (size_t i = 0; i < mel.size(); i++) {
      masked[i] = mel[i] * mask[i];  // Element-wise multiplication
    }
    return masked;
  }
};
```

**DCCRN performance**:
- ERLE: 30-50dB (matches or exceeds WebRTC)
- Latency: 10-20ms (acceptable for voice AI)
- Model size: 1-5MB
- Advantage: Handles non-linear echo, far-field scenarios

## 7. AUTOMATIC GAIN CONTROL (AGC)

### 7.1 Simple Peaking Detector

```cpp
class SimpleAGC {
private:
  float target_level = -20.0f;  // dBFS target
  float gain = 0.0f;

public:
  std::vector<float> process(const std::vector<float>& audio) {
    // 1. Measure loudness (RMS)
    float rms = 0.0f;
    for (float sample : audio) {
      rms += sample * sample;
    }
    rms = std::sqrt(rms / audio.size());

    float loudness_db = 20.0f * std::log10(rms + 1e-10f);

    // 2. Compute gain adjustment
    float error = target_level - loudness_db;
    gain += 0.1f * error;  // Simple low-pass filter

    // 3. Apply gain with clipping
    std::vector<float> output;
    for (float sample : audio) {
      float amplified = sample * std::pow(10.0f, gain / 20.0f);
      output.push_back(std::clamp(amplified, -1.0f, 1.0f));
    }

    return output;
  }
};
```

### 7.2 WebRTC AGC

WebRTC's AGC is more sophisticated (handles compression knee, attack/release):

```cpp
// Use through WebRTC AudioProcessing
apm->ProcessStream(...);
apm->GetStatistics(/* outputs loudness metrics */);
```

## 8. COMPLETE AUDIO PIPELINE

```cpp
class VoiceAudioPipeline {
private:
  ALSAAudioCapture capture;
  AudioRingBuffer<float, 48000> capture_buffer;
  AudioRingBuffer<float, 48000> playback_buffer;

  SampleRateConverter resampler_48_to_16;
  SampleRateConverter resampler_16_to_48;

  MelSpectrogramAVX512 mel_computer;
  DCCRNEchoCanceller echo_canceller;
  SimpleAGC agc;

  std::thread capture_thread, processing_thread, playback_thread;

public:
  void run() {
    // Start threads
    capture_thread = std::thread([this] { capture_loop(); });
    processing_thread = std::thread([this] { processing_loop(); });
    playback_thread = std::thread([this] { playback_loop(); });
  }

private:
  void capture_loop() {
    while (true) {
      auto audio = capture.read_frame();
      capture_buffer.push(audio.data(), audio.size());
    }
  }

  void processing_loop() {
    while (true) {
      std::vector<float> micro_audio(160);
      std::vector<float> speaker_audio(160);

      // Get synchronized frames
      capture_buffer.pop(micro_audio.data(), 160);
      playback_buffer.pop(speaker_audio.data(), 160);

      // Echo cancellation
      auto echo_cancelled = echo_canceller.process(micro_audio, speaker_audio);

      // AGC
      auto agc_output = agc.process(echo_cancelled);

      // Mel spectrogram for ML
      auto mel_spec = mel_computer.compute(
        reinterpret_cast<int16_t*>(agc_output.data()), agc_output.size());

      // Send mel_spec to VAD, ASR, etc.
      on_audio_frame(mel_spec);
    }
  }

  void playback_loop() {
    std::vector<float> speaker_audio;
    while (true) {
      if (playback_buffer.pop(speaker_audio.data(), 160)) {
        playback.write_to_device(speaker_audio);
      }
    }
  }

  void on_audio_frame(const std::vector<float>& mel_spec) {
    // Feed to VAD, ASR pipeline
  }
};
```

## 9. PERFORMANCE CHARACTERISTICS

### 9.1 Latency Budget per Component

```
Capture (ALSA):           5ms
Resampling (48→16kHz):    2ms
Echo cancellation:        15ms
AGC:                      2ms
Mel spectrogram:          1ms (with AVX-512)
─────────────────────────────
Total processing:         25ms
```

### 9.2 CPU Usage

Single stream (VAD + ASR):
- Capture/playback: <1% (driver overhead)
- Resampling: <1%
- Echo cancellation: 2-3%
- Mel spectrogram: <1%
- **Total**: 3-5% of single core

100 concurrent streams:
- **30-50 cores** on modern server CPU (assuming 10-12% per stream with thread contention)

## 10. SUMMARY & DESIGN PRINCIPLES

**Key audio processing insights**:

1. **Ring buffers**: Decouple real-time I/O from processing; essential for latency control
2. **Mel spectrogram**: Use SIMD (AVX-512/NEON); achieve <2ms latency
3. **Echo cancellation**: Critical for barge-in detection; DCCRN outperforms WebRTC AEC
4. **Opus codec**: Use for bandwidth-limited scenarios (18kHz telephony); 24kbps achieves 90%+ quality
5. **Sample rate conversion**: Polyphase filters preferred; <5ms latency achievable
6. **AGC**: Essential for consistent ASR input; simple peaking detector sufficient
7. **Synchronization**: Microphone and speaker must be in sync (±20ms) for effective AEC
8. **SIMD optimization**: 3-5x speedup achievable for audio processing on modern CPUs
9. **Real-time scheduling**: Audio processing threads need RT priority (SCHED_FIFO)
10. **Monitoring**: Track audio levels, echo suppression ERLE, loudness consistency

Audio processing is the unglamorous foundation of voice AI, but poor audio processing cascades through the entire pipeline. Clean, synchronized, properly normalized audio inputs are prerequisite for accurate VAD and ASR.

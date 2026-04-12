# MODULE 32 — LLM Integration in Voice Pipeline

## 1. INTRODUCTION & MOTIVATION

The Large Language Model (LLM) component transforms speech-to-text into intelligent, context-aware responses. Unlike text-based chatbots where users tolerate 1-2 second latency, voice systems demand sub-100ms Time-To-First-Token (TTFT) for natural turn-taking. This is extraordinarily demanding: while typical cloud LLM deployments achieve 100-300ms TTFT, voice systems require local or optimized cloud execution with 50-100ms TTFT.

The challenge manifests across multiple dimensions:

1. **Latency SLA**: TTFT < 100ms (vs. 100-300ms for text applications)
2. **Streaming integration**: LLM must accept incomplete ASR results (partial hypotheses) and begin generating while user still speaking
3. **Streaming output**: Token generation must feed directly to TTS (not wait for complete response)
4. **Interruption handling**: User may interrupt mid-response; LLM state must roll back cleanly
5. **Model selection**: Large models (70B+ parameters) can't meet latency targets; require 3B-7B models with custom optimization

This module covers the unique challenges of integrating LLMs into voice pipelines: prefix caching for incremental ASR input, streaming token generation patterns, voice-specific serving optimizations, and the surprising finding that smaller models (3B-7B) often outperform large models for voice tasks due to tighter latency bounds and reduced hallucination.

## 2. LLM LATENCY PROBLEM IN VOICE

### 2.1 Typical LLM Latency Breakdown

For a standard 7B-parameter model generating 10 tokens:

```
TTFT (Time-To-First-Token):  80-150ms
  ├─ Prefill (encode user input):     40-80ms
  ├─ Compute first token:              40-70ms
  └─ Total:                            80-150ms

Token generation latency:        10-20ms per token
  (once key-value cache "warmed")

Example: "What time is it?"
├─ Input encoding (4 tokens):     50ms
├─ First token generation:         60ms (total TTFT = 110ms)
├─ Token 2 ("The"):               12ms
├─ Token 3 ("current"):           13ms
├─ Token 4 ("time"):              11ms
├─ Token 5 ("is"):                12ms
└─ Total generation:              ~10 tokens in ~120ms

Voice AI requirement:
├─ TTFA target: 200ms
├─ Allocated to LLM: 80ms
├─ AVAILABLE for LLM: Requires 80-100ms TTFT + 50-70ms first tokens
└─ Typical cloud latency: 150-200ms FAILS target
```

### 2.2 Latency Sources

**Prefill phase** (processing input tokens):
- Memory bandwidth limited: loading model weights from memory
- Compute-to-memory ratio: prefill is memory-bound, not compute-bound
- Typical: 40-100µs per token with batching

**Decode phase** (generating new tokens):
- Compute-bound: only one token generated per forward pass
- Requires attention over growing KV cache
- Typical: 10-20µs per token once cache is warm

**Optimization insight**: Prefill and decode are fundamentally different workloads:
- Prefill dominates TTFT latency
- Decode dominates total throughput latency
- Voice requires latency optimization (fast prefill), not throughput optimization

## 3. STREAMING TOKEN GENERATION FOR VOICE

### 3.1 Streaming LLM Integration Pattern

Unlike batch text generation, voice systems must stream tokens to TTS as they're generated:

```
T=0ms:    ASR emits first token "What"
          LLM begins inference with context: "<user> What"

T=50ms:   LLM processes input, computes first output token
          Output: "The" → immediately send to TTS

T=60ms:   TTS begins synthesis with "The"
          LLM generates token 2: "current"

T=75ms:   TTS outputs first audio (TTFA = 75ms + VAD/ASR latency)
          LLM generates token 3: "time"

T=90ms:   LLM generates token 4: "is"
T=105ms:  LLM generates token 5: "15:30"
          ...
T=200ms:  TTS finishes synthesis, user hears "The current time is..."
```

**Implementation**:

```cpp
class StreamingLLMforVoice {
private:
  LLMModel model;
  std::queue<std::string> token_queue;
  std::mutex token_queue_lock;

public:
  void on_asr_partial(const std::string& partial_text) {
    // Start new LLM inference with partial ASR hypothesis
    std::thread inference_thread([this, partial_text]() {
      this->generate_tokens(partial_text);
    });
    inference_thread.detach();  // Non-blocking
  }

private:
  void generate_tokens(const std::string& prompt) {
    // Prepare input
    auto tokens = model.tokenize(prompt);
    auto input_ids = tokens.ids;

    // KV cache for this sequence
    KVCache cache;

    // Prefill: process all input tokens
    auto [logits, cache] = model.forward(input_ids, cache);

    // Decode: generate tokens one-by-one
    for (int i = 0; i < MAX_GENERATION_LENGTH; i++) {
      // Greedy decoding (or beam search)
      int token_id = argmax(logits);
      std::string token = model.decode_token(token_id);

      // Streaming: send immediately to TTS
      {
        std::lock_guard<std::mutex> lock(token_queue_lock);
        token_queue.push(token);
      }

      // Signal TTS that new token available
      tts_condition_var.notify_one();

      // Stop condition
      if (token == model.eos_token()) {
        break;
      }

      // Next forward pass: just the new token
      auto [new_logits, new_cache] = model.forward({token_id}, cache);
      logits = new_logits;
      cache = new_cache;
    }
  }

public:
  bool get_next_token(std::string& token, int timeout_ms) {
    std::unique_lock<std::mutex> lock(token_queue_lock);

    if (tts_condition_var.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                   [this] { return !token_queue.empty(); })) {
      token = token_queue.front();
      token_queue.pop();
      return true;
    }
    return false;  // Timeout
  }
};
```

### 3.2 Sentence Boundary Detection

Voice systems shouldn't stream every token; instead, batch tokens into sentences before sending to TTS. Why?

1. **TTS buffering**: TTS synthesizes more naturally with sentence context (prosody decisions)
2. **Interruption handling**: Easier to cancel mid-sentence than mid-word
3. **Natural grouping**: Users perceive sentence-level pauses, not token-level

**Heuristics for sentence detection**:

```cpp
class SentenceBoundaryDetector {
private:
  std::string token_buffer;

public:
  std::vector<std::string> detect_sentence_boundaries(
      const std::string& token) {
    token_buffer += token;

    std::vector<std::string> complete_sentences;

    // Rule 1: Explicit punctuation
    if (token == "." || token == "!" || token == "?") {
      complete_sentences.push_back(token_buffer);
      token_buffer.clear();
    }

    // Rule 2: Comma + next token is capitalized (new clause)
    else if (token == "," && next_token_is_capitalized()) {
      // Tentatively send; may join if next token is lowercase
      // (e.g., "However, we decided..." should not split)
    }

    // Rule 3: Newline marker
    else if (token == "\n") {
      complete_sentences.push_back(token_buffer);
      token_buffer.clear();
    }

    // Rule 4: Long buffer without punctuation (fallback)
    else if (token_buffer.length() > 200) {  // ~30-40 words
      complete_sentences.push_back(token_buffer);
      token_buffer.clear();
    }

    return complete_sentences;
  }

private:
  bool next_token_is_capitalized() {
    // Requires lookahead to next token
    // Practical: use heuristic on buffer state
    return false;  // Simplified
  }
};
```

**Better approach: Neural sentence boundary detection**

```cpp
class NeuralSentenceBoundaryDetector {
private:
  SmallNN boundary_model;  // Lightweight classifier

public:
  bool is_sentence_boundary(const std::vector<int>& token_ids,
                           const std::string& current_token,
                           const std::string& next_token) {
    // Input: last 10 tokens + current + next
    // Output: probability of sentence boundary

    auto input = prepare_features(token_ids, current_token, next_token);
    float boundary_prob = boundary_model.forward(input)[0];

    return boundary_prob > 0.7f;
  }

private:
  std::vector<float> prepare_features(const std::vector<int>& token_ids,
                                     const std::string& current,
                                     const std::string& next) {
    std::vector<float> features;
    // Token embeddings: token_ids → embeddings
    // Lexical features: is "." in current? is next capitalized?
    // Linguistic features: is next a conjunction?
    return features;
  }
};
```

Model size: ~1-5MB (can run on phone); training: 1-2 days on modern GPU with labeled boundary data.

## 4. VOICE-SPECIFIC LLM OPTIMIZATION

### 4.1 Model Selection: Smaller is Better

Surprising finding in voice AI: **smaller models (3B-7B) outperform larger models (13B-70B)** due to latency constraints.

**Latency vs. Accuracy Tradeoff**:

| Model | Params | TTFT | Accuracy* | Best For |
|-------|--------|------|-----------|----------|
| Llama 2 (3B) | 3B | 40-60ms | 65% | Voice AI ✓ |
| Phi-2 | 2.7B | 35-50ms | 68% | Voice AI ✓ |
| Llama 2 (7B) | 7B | 60-100ms | 75% | Voice AI |
| Mistral 7B | 7B | 70-110ms | 78% | Voice AI |
| Llama 2 (13B) | 13B | 120-180ms | 82% | Text only |
| GPT-3.5 equiv | 175B+ | 300-500ms | 90% | Cloud only |

*Accuracy: Typical performance on QA benchmarks

**Why smaller models win for voice**:
1. **Latency**: 3B model (50ms) + TTS (100ms) = 150ms total acceptable; 70B (300ms) + TTS = 400ms total ❌
2. **Hallucination**: Smaller models hallucinate less on factual queries (time, weather), important for voice
3. **Fine-tuning**: Easier to fine-tune 3B model for voice domain (chitchat, facts, short answers)
4. **Cost**: 10-100x cheaper to serve 3B vs 70B

**Example**: Google Assistant uses primarily 1-3B models locally on-device, supplemented by larger cloud models for complex queries.

### 4.2 Prefix Caching for Incremental ASR

As ASR emits partial hypotheses, we want to avoid re-processing prior tokens. Prefix caching maintains KV cache from previous LLM runs.

**Pattern: Prefix caching with ASR updates**

```cpp
class PrefixCachedLLM {
private:
  LLMModel model;
  KVCache cached_kv;
  int cached_token_count = 0;
  std::string previous_input;

public:
  void on_asr_update(const std::string& new_hypothesis) {
    // Check if new hypothesis is prefix of old
    if (is_prefix(previous_input, new_hypothesis)) {
      // Extend from cache
      extend_from_cache(new_hypothesis);
    } else {
      // Reset cache and start fresh
      reset_cache();
      process_new_hypothesis(new_hypothesis);
    }

    previous_input = new_hypothesis;
  }

private:
  bool is_prefix(const std::string& old_text, const std::string& new_text) {
    return new_text.substr(0, old_text.length()) == old_text;
  }

  void extend_from_cache(const std::string& new_hypothesis) {
    // Get new tokens
    auto tokens = model.tokenize(new_hypothesis);

    // Find where cache ends
    int cache_end = cached_token_count;
    int new_tokens_start = find_divergence_point(tokens.ids);

    // Forward pass only on new tokens
    auto new_token_ids = std::vector<int>(
      tokens.ids.begin() + new_tokens_start,
      tokens.ids.end());

    auto [logits, new_cache] = model.forward_with_kv(
      new_token_ids, cached_kv);

    // Extend cache
    cached_kv = new_cache;
    cached_token_count = tokens.ids.size();

    // Generate first token
    int first_token_id = argmax(logits);
    generate_from_cache(first_token_id);
  }

  int find_divergence_point(const std::vector<int>& new_tokens) {
    // Compare with cached tokens
    // Return index where they diverge
    auto old_tokens = model.tokenize(previous_input).ids;
    for (int i = 0; i < std::min(old_tokens.size(), new_tokens.size()); i++) {
      if (old_tokens[i] != new_tokens[i]) {
        return i;
      }
    }
    return old_tokens.size();
  }

  void reset_cache() {
    cached_kv.clear();
    cached_token_count = 0;
  }

  void process_new_hypothesis(const std::string& hypothesis) {
    auto tokens = model.tokenize(hypothesis);

    // Prefill: process all input tokens
    auto [logits, kv] = model.forward(tokens.ids, KVCache());

    cached_kv = kv;
    cached_token_count = tokens.ids.size();

    // Emit first token
    int token_id = argmax(logits);
    generate_from_cache(token_id);
  }

  void generate_from_cache(int token_id) {
    std::string token = model.decode_token(token_id);
    tts_queue.push(token);
  }
};
```

**Performance benefit**: Prefix caching reduces redundant computation by 50-70% when ASR updates incrementally.

### 4.3 KV Cache Warmup for Low Latency

Before user speaks, pre-warm the KV cache with common prefixes:

```cpp
class KVCacheWarmup {
private:
  struct WarmupEntry {
    std::string prefix;
    KVCache cached_kv;
  };
  std::vector<WarmupEntry> warmup_cache;

public:
  void setup_warmup() {
    // Pre-compute KV cache for common conversation starters
    std::vector<std::string> common_prefixes = {
      "<user> Hello",
      "<user> Hi",
      "<user> What",
      "<user> How",
      "<user> Tell me",
      "<user> Can you",
      "<user> I need",
    };

    for (const auto& prefix : common_prefixes) {
      auto tokens = model.tokenize(prefix);
      auto [logits, kv] = model.forward(tokens.ids, KVCache());

      warmup_cache.push_back({prefix, kv});
    }
  }

  KVCache get_warmup_cache(const std::string& user_input) {
    // Find longest matching prefix
    for (const auto& entry : warmup_cache) {
      if (user_input.find(entry.prefix) == 0) {
        return entry.cached_kv;  // Return warmup cache
      }
    }
    return KVCache();  // Empty cache if no match
  }

  void generate_with_warmup(const std::string& user_input) {
    auto warmup_kv = get_warmup_cache(user_input);

    if (!warmup_kv.empty()) {
      // Use warmup cache: skip prefill for common prefix
      // Continue with new tokens only
      auto tokens = model.tokenize(user_input);
      auto [logits, kv] = model.forward_with_kv(tokens.ids, warmup_kv);

      // TTFT is now 30-50ms (skipped most of prefill)
      emit_first_token(argmax(logits));
    } else {
      // No warmup: full prefill
      auto tokens = model.tokenize(user_input);
      auto [logits, kv] = model.forward(tokens.ids, KVCache());

      emit_first_token(argmax(logits));
    }
  }
};
```

**Benefit**: Reduces TTFT from 60-100ms to 30-50ms for common queries (10-20% of voice interactions).

## 5. INTERRUPTION & CONTEXT MANAGEMENT

### 5.1 State Rollback on Interruption

When user interrupts (barges in), we must:
1. Stop current response generation
2. Discard partial output
3. Clear LLM state
4. Process new user input

```cpp
class InterruptableVoiceLLM {
private:
  enum LLMState {
    IDLE,
    PROCESSING_USER_INPUT,
    GENERATING_RESPONSE,
    INTERRUPTED,
  } state{IDLE};

  std::thread llm_thread;
  std::atomic<bool> should_interrupt{false};

public:
  void on_barge_in_detected() {
    should_interrupt.store(true, std::memory_order_release);
    state = INTERRUPTED;

    // Wait for LLM thread to finish
    llm_thread.join();

    // Clear TTS queue (discard partial response)
    tts_queue.clear();

    // Reset state
    reset_context();
  }

  void on_new_asr_input(const std::string& new_input) {
    should_interrupt.store(false, std::memory_order_release);
    state = PROCESSING_USER_INPUT;

    llm_thread = std::thread([this, new_input]() {
      this->generate_with_interrupt_check(new_input);
    });
  }

private:
  void generate_with_interrupt_check(const std::string& prompt) {
    auto tokens = model.tokenize(prompt);
    auto [logits, kv] = model.forward(tokens.ids, KVCache());

    state = GENERATING_RESPONSE;

    // Token generation loop
    for (int i = 0; i < MAX_LENGTH; i++) {
      // Check for interruption
      if (should_interrupt.load(std::memory_order_acquire)) {
        break;  // Stop generation
      }

      int token_id = argmax(logits);
      std::string token = model.decode_token(token_id);

      tts_queue.push(token);

      if (token == model.eos_token()) {
        break;
      }

      auto [new_logits, new_kv] = model.forward_with_kv({token_id}, kv);
      logits = new_logits;
      kv = new_kv;
    }

    state = IDLE;
  }

  void reset_context() {
    // Option 1: Stateless (simplest)
    // Clear everything; next query starts fresh

    // Option 2: Keep conversation history
    // Remove last turn only (user's input + system response)
    conversation_history.pop_back();
    conversation_history.pop_back();
  }
};
```

### 5.2 Conversation Context Management

Voice systems often maintain multi-turn context (remember previous queries):

```cpp
class VoiceConversationManager {
private:
  struct Turn {
    std::string user_text;
    std::string system_response;
    int64_t timestamp;
  };

  std::vector<Turn> conversation_history;
  static constexpr size_t MAX_CONTEXT_TOKENS = 1024;

public:
  std::string build_prompt_with_context(const std::string& new_user_input) {
    std::string prompt = "<system> You are a helpful voice assistant.\n";

    // Include recent context (up to token limit)
    int context_tokens = 0;
    for (int i = conversation_history.size() - 1; i >= 0; i--) {
      const auto& turn = conversation_history[i];

      std::string turn_text = "<user> " + turn.user_text + "\n" +
                              "<assistant> " + turn.system_response + "\n";

      int tokens = model.count_tokens(turn_text);
      if (context_tokens + tokens > MAX_CONTEXT_TOKENS) {
        break;  // Stop here; remaining context too old
      }

      prompt = turn_text + prompt;
      context_tokens += tokens;
    }

    // Add new input
    prompt += "<user> " + new_user_input + "\n<assistant> ";

    return prompt;
  }

  void add_turn(const std::string& user_text,
               const std::string& system_response) {
    conversation_history.push_back({
      user_text,
      system_response,
      get_current_timestamp(),
    });

    // Limit context window
    while (conversation_history.size() > 10) {
      conversation_history.erase(conversation_history.begin());
    }
  }
};
```

## 6. VOICE-SPECIFIC LLM DESIGN

### 6.1 Prompt Engineering for Voice

Prompts for voice differ from text:

```
Text prompt (longer, more detailed):
"Given the user's question about the current time, respond concisely with
the time in 12-hour format. If the exact time is unknown, estimate based
on context. Only provide the time; do not include additional information."

Voice prompt (shorter, natural):
"You are a helpful voice assistant. Answer briefly and naturally."
```

Voice prompts should:
1. **Be concise**: Fewer instructions, simpler language
2. **Encourage short answers**: "Answer in 1-2 sentences"
3. **Specify output format**: For facts (time, weather), request structured output

```python
class VoicePromptTemplate:
    @staticmethod
    def build_prompt(user_input, task="general"):
        templates = {
            "general": """You are a helpful voice assistant.
Answer the user's question briefly and naturally.
Keep response under 30 words.
User: {user_input}
Assistant:""",

            "factual": """You are a voice assistant providing facts.
Answer the user's question with the specific fact requested.
Be concise and accurate.
User: {user_input}
Assistant:""",

            "conversational": """You are a friendly voice assistant.
Respond naturally to the user's message.
Be warm and conversational, but concise.
User: {user_input}
Assistant:""",
        }

        return templates[task].format(user_input=user_input)
```

### 6.2 Response Length Control

Voice systems must limit response length (too long → poor UX):

```cpp
class VoiceResponseLimiter {
private:
  struct GenerationConfig {
    int max_tokens = 30;        // ~20-30 words for voice
    int max_sentences = 2;      // Typically 1-2 sentences
    float length_penalty = -2.0f; // Encourage shorter responses
  } config;

public:
  std::string generate_response(const std::string& prompt) {
    // Add length penalty to model configuration
    auto response = model.generate(
      prompt,
      max_length=config.max_tokens,
      length_penalty=config.length_penalty,
      early_stopping=true);

    // Post-processing: truncate at sentence boundary
    return truncate_at_sentence(response);
  }

private:
  std::string truncate_at_sentence(const std::string& text) {
    // Find first 2 sentences
    std::vector<std::string> sentences = split_sentences(text);

    std::string result;
    for (int i = 0; i < std::min(2, (int)sentences.size()); i++) {
      result += sentences[i];
    }

    return result;
  }

  std::vector<std::string> split_sentences(const std::string& text) {
    // Simple: split on ".", "!", "?"
    // Better: use NLTK or similar
    std::vector<std::string> sentences;
    // ... implementation ...
    return sentences;
  }
};
```

## 7. PRODUCTION VOICE LLM SERVING

### 7.1 vLLM with Voice Optimizations

vLLM is a production serving framework. For voice, optimize:

```python
from vllm import LLM, SamplingParams

class VoiceOptimizedLLM:
    def __init__(self, model_id="Phi-2"):
        self.llm = LLM(
            model=model_id,
            dtype="float16",  # Reduce latency
            gpu_memory_utilization=0.8,
            max_model_len=1024,
            enable_prefix_caching=True,  # For incremental ASR
            max_num_batched_tokens=512,
            enforce_eager=False,  # Use paged attention for low latency
        )

        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=30,  # Voice response length limit
            use_beam_search=False,  # Beam search too slow for voice
        )

    def generate(self, prompt):
        """Generate with voice-specific optimizations"""
        outputs = self.llm.generate(
            prompt,
            sampling_params=self.sampling_params,
            use_tqdm=False)

        return outputs[0].outputs[0].text

# Benchmark: Phi-2 on single A100
# TTFT: 45ms (prefill)
# Token latency: 12ms
# 30 tokens total: ~400ms (excluding other pipeline stages)
```

### 7.2 Multi-GPU Batching for Multiple Sessions

For voice call centers (100+ concurrent sessions):

```python
class BatchedVoiceLLM:
    def __init__(self, num_gpus=4):
        self.llms = [
            LLM(model="Phi-2", gpu_memory_utilization=0.9)
            for _ in range(num_gpus)
        ]
        self.llm_load = [0] * num_gpus

    def generate_async(self, session_id, prompt):
        """Route request to least-loaded GPU"""
        min_load_idx = self.llm_load.index(min(self.llm_load))
        self.llm_load[min_load_idx] += 1

        async def _generate():
            llm = self.llms[min_load_idx]
            result = llm.generate(prompt, ...)
            self.llm_load[min_load_idx] -= 1
            return result

        return _generate()

# Capacity: 4 GPUs × 250 concurrent sessions/GPU = 1000 concurrent voice sessions
```

## 8. EDGE LLM FOR VOICE

### 8.1 On-Device LLM with Quantization

For privacy and latency, run LLM locally:

```cpp
class EdgeVoiceLLM {
private:
  OnnxModel model;  // Quantized 3B model (INT8)

public:
  EdgeVoiceLLM(const std::string& model_path) {
    // Load INT8 quantized model
    // Size: ~1-2GB (vs. 6GB for FP16)
    // Latency: 50-100ms TTFT (vs. 40ms cloud)
    model.load(model_path);
  }

  std::string generate(const std::string& prompt) {
    auto tokens = model.tokenize(prompt);

    // Single-threaded inference on CPU
    auto [logits, kv] = model.forward(tokens, KVCache());

    std::string response;
    for (int i = 0; i < 30; i++) {
      int token_id = argmax(logits);
      std::string token = model.decode_token(token_id);

      response += token;

      if (token == model.eos_token()) break;

      auto [new_logits, new_kv] = model.forward_with_kv({token_id}, kv);
      logits = new_logits;
      kv = new_kv;
    }

    return response;
  }
};
```

**Trade-offs**:
- Latency: Slightly higher (50-100ms vs. 40ms cloud)
- Privacy: 100% local (no data leaves device)
- Bandwidth: Zero (no network latency)
- Cost: Zero inference cost
- Accuracy: Slightly lower (due to quantization)

## 9. MONITORING & OPTIMIZATION

### 9.1 LLM Metrics for Voice

Track:
- **TTFT**: Time-to-first-token (target: <100ms)
- **Token latency**: Per-token latency (target: 10-20ms)
- **Response length**: Words per response (target: 10-30 words)
- **Relevance**: BLEU/ROUGE against reference (target: >0.5)
- **Hallucination rate**: % of responses with false facts (target: <5%)

```cpp
void log_llm_metrics(const std::string& prompt,
                    const std::string& response,
                    int64_t ttft_ms,
                    int num_tokens,
                    float relevance_score) {
  metrics.llm_ttft.observe(ttft_ms);
  metrics.llm_token_latency.observe(ttft_ms / num_tokens);
  metrics.llm_response_length.observe(response.length());
  metrics.llm_relevance.observe(relevance_score);
}
```

## 10. SUMMARY & DESIGN PRINCIPLES

**Key insights for voice LLM integration**:

1. **Model size**: Smaller models (3B-7B) win for voice due to latency constraints; 70B+ models too slow
2. **TTFT target**: <100ms critical; achieved via low-latency serving, quantization, KV cache warmup
3. **Streaming**: Emit tokens to TTS immediately; batch into sentences for natural prosody
4. **Prefix caching**: Maintain KV cache across ASR updates; reduce redundant computation 50-70%
5. **Interruption**: Design rollback mechanism from ground up; users will interrupt
6. **Sentence boundary**: Critical for TTS integration; use neural detector for accuracy
7. **Response length**: Limit to 20-40 words; longer responses hurt UX
8. **Edge deployment**: Quantized 3B model on-device achieves 50-100ms TTFT with zero privacy loss
9. **Monitoring**: Track TTFT, token latency, hallucination rate continuously

The LLM stage in voice pipelines is fundamentally constrained by latency in ways text-only systems are not. Success requires ruthless optimization: model selection for latency over accuracy, prefix caching to eliminate redundant computation, and architectural patterns (streaming tokens, sentence batching, rollback on interruption) designed specifically for real-time voice interaction.

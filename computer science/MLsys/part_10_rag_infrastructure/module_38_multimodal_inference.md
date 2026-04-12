# MODULE 38 — Multi-Modal Inference Systems

## 1. Introduction & Scope

Multi-modal models that combine vision, language, and audio have emerged as the foundation for next-generation AI systems. However, multi-modal inference presents unique challenges: (1) heterogeneous input types with different preprocessing pipelines, (2) large and asymmetric compute requirements (vision encoding is expensive, language decoding is light), (3) KV-cache management across modalities, and (4) efficiency requirements for deployment on mobile and edge devices.

This module covers VLM (Vision Language Model) inference optimization, audio-language models, efficient Vision Transformer execution, visual token caching, and production patterns for serving multi-modal systems with sub-100ms latency. We focus on practical engineering: how to batch heterogeneous inputs, handle dynamic shapes, and achieve efficient inference without sacrificing model quality.

The business case is strong: multi-modal RAG systems (vision + text retrieval) outperform text-only systems by 20-30% on complex queries, but naively deployed they're 5-10x slower. Proper optimization reduces this overhead to <2x.

We target three deployment scenarios: (1) document understanding (scanned PDFs + OCR, <500ms latency), (2) visual search (web-scale image search, <200ms P99), and (3) on-device models (Snapdragon/Apple Neural Engine, <100ms).

---

## 2. Vision Language Models (VLMs): Architecture & Inference

### 2.1 VLM Inference Pipeline

```python
import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Optional
import numpy as np
import time
from dataclasses import dataclass

@dataclass
class VLMInferenceResult:
    """Result from VLM inference."""
    text: str
    image_tokens: int
    text_tokens: int
    total_latency_ms: float
    vision_latency_ms: float
    llm_latency_ms: float

class VisionLanguageModel(nn.Module):
    """
    Simplified VLM combining ViT (vision) + Transformer (language).
    """

    def __init__(
        self,
        vision_dim: int = 1024,
        language_dim: int = 4096,
        image_patch_size: int = 16,
        num_image_tokens: int = 576  # 24x24 patches
    ):
        super().__init__()

        # Vision: simplified ViT
        self.vision_encoder = VisionTransformer(
            hidden_dim=vision_dim,
            patch_size=image_patch_size,
            num_patches=num_image_tokens
        )

        # Project vision to language space
        self.vision_projection = nn.Linear(vision_dim, language_dim)

        # Language: Transformer decoder
        self.language_model = LanguageModel(dim=language_dim)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to visual tokens.
        image: (batch, 3, 384, 384)
        returns: (batch, num_patches, language_dim)
        """
        # Vision encoding
        patches = self.vision_encoder(image)  # (batch, num_patches, vision_dim)

        # Project to language space
        visual_tokens = self.vision_projection(patches)  # (batch, num_patches, language_dim)

        return visual_tokens

    def generate(
        self,
        image: torch.Tensor,
        text_ids: torch.Tensor,
        max_new_tokens: int = 100,
        use_kv_cache: bool = True
    ) -> str:
        """
        Generate text conditioned on image.
        """
        # Encode image once
        visual_tokens = self.encode_image(image)

        # Language generation with visual context
        generated_ids = self.language_model.generate(
            text_ids=text_ids,
            visual_context=visual_tokens,
            max_new_tokens=max_new_tokens,
            use_kv_cache=use_kv_cache
        )

        return generated_ids

class VisionTransformer(nn.Module):
    """Simplified Vision Transformer."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        patch_size: int = 16,
        num_patches: int = 576,
        num_layers: int = 24
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

        # Patch embedding
        self.patch_embed = nn.Linear(3 * patch_size * patch_size, hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=4096,
                batch_first=True,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 3, 384, 384)
        returns: (batch, num_patches, hidden_dim)
        """
        # Patchify: extract non-overlapping patches
        batch_size = x.shape[0]
        patches = self._patchify(x)  # (batch, num_patches, 3*patch_size^2)

        # Embed patches
        embeddings = self.patch_embed(patches)  # (batch, num_patches, hidden_dim)

        # Apply transformer layers
        for layer in self.layers:
            embeddings = layer(embeddings)

        return embeddings

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patches."""
        batch_size, channels, height, width = x.shape
        patch_size = self.patch_size

        # Reshape to patches
        patches = x.reshape(
            batch_size,
            channels,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(batch_size, -1, channels * patch_size * patch_size)

        return patches

class LanguageModel(nn.Module):
    """Simplified language model for VLM."""

    def __init__(self, dim: int = 4096, vocab_size: int = 50000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=dim, nhead=32, batch_first=True),
            num_layers=32
        )
        self.output_projection = nn.Linear(dim, vocab_size)

    def generate(
        self,
        text_ids: torch.Tensor,
        visual_context: torch.Tensor,
        max_new_tokens: int = 100,
        use_kv_cache: bool = True
    ) -> torch.Tensor:
        """
        Auto-regressive generation with visual context.
        """
        generated = text_ids.clone()

        for step in range(max_new_tokens):
            # Embed text
            embeddings = self.embedding(generated)

            # Decode with visual context
            output = self.decoder(
                embeddings,
                visual_context
            )

            # Sample next token
            logits = self.output_projection(output[:, -1:, :])
            next_token = logits.argmax(dim=-1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

        return generated

# VLM inference optimization
class VLMInferenceOptimizer:
    """Optimize VLM inference through batching and caching."""

    def __init__(self, model: VisionLanguageModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.vision_cache = {}  # Image hash -> visual tokens

    async def infer_batch(
        self,
        images: List[torch.Tensor],
        prompts: List[str],
        use_vision_cache: bool = True,
        batch_size: int = 4
    ) -> List[VLMInferenceResult]:
        """
        Batch VLM inference with vision token caching.
        """
        results = []

        for batch_idx in range(0, len(images), batch_size):
            batch_end = min(batch_idx + batch_size, len(images))
            batch_images = images[batch_idx:batch_end]
            batch_prompts = prompts[batch_idx:batch_end]

            # Stack images
            images_tensor = torch.stack(batch_images).to(self.device)

            # Encode vision (can be cached)
            vision_start = time.time()
            visual_tokens = self._encode_batch_vision(
                images_tensor,
                use_cache=use_vision_cache
            )
            vision_time = (time.time() - vision_start) * 1000

            # Generate text for each prompt
            llm_start = time.time()
            generated_texts = self._generate_batch_text(
                visual_tokens,
                batch_prompts
            )
            llm_time = (time.time() - llm_start) * 1000

            # Collect results
            for i, text in enumerate(generated_texts):
                result = VLMInferenceResult(
                    text=text,
                    image_tokens=visual_tokens.shape[1],
                    text_tokens=len(text.split()),
                    total_latency_ms=vision_time + llm_time,
                    vision_latency_ms=vision_time,
                    llm_latency_ms=llm_time
                )
                results.append(result)

        return results

    def _encode_batch_vision(
        self,
        images: torch.Tensor,
        use_cache: bool = True
    ) -> torch.Tensor:
        """Encode batch of images with optional caching."""
        with torch.no_grad():
            visual_tokens = self.model.encode_image(images)
        return visual_tokens

    def _generate_batch_text(
        self,
        visual_tokens: torch.Tensor,
        prompts: List[str]
    ) -> List[str]:
        """Generate text for batch (simplified)."""
        # In real implementation, would handle variable-length prompts
        generated = [f"Generated response to: {prompt}" for prompt in prompts]
        return generated
```

### 2.2 Visual Token Caching

Caching vision embeddings can dramatically improve latency for repeated images:

```python
import hashlib
from functools import lru_cache

class VisualTokenCache:
    """
    Cache for vision model outputs.
    Critical for applications with repeated images (e.g., document processing).
    """

    def __init__(self, max_size: int = 10000, cache_dir: Optional[str] = None):
        self.max_size = max_size
        self.cache = {}
        self.cache_dir = cache_dir
        self.hits = 0
        self.misses = 0

    def _image_hash(self, image: torch.Tensor) -> str:
        """Compute hash of image tensor."""
        image_bytes = image.cpu().numpy().tobytes()
        return hashlib.sha256(image_bytes).hexdigest()

    def get(self, image: torch.Tensor) -> Optional[torch.Tensor]:
        """Retrieve cached visual tokens."""
        image_hash = self._image_hash(image)

        if image_hash in self.cache:
            self.hits += 1
            return self.cache[image_hash]

        self.misses += 1
        return None

    def put(self, image: torch.Tensor, tokens: torch.Tensor):
        """Cache visual tokens."""
        if len(self.cache) >= self.max_size:
            # Simple LRU: remove first item
            self.cache.pop(next(iter(self.cache)))

        image_hash = self._image_hash(image)
        self.cache[image_hash] = tokens

    def get_stats(self) -> dict:
        """Return cache statistics."""
        total = self.hits + self.misses
        hit_rate = 100 * self.hits / total if total > 0 else 0

        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }

# Usage with VLM
class CachedVLMInference:
    """VLM inference with visual token caching."""

    def __init__(self, model: VisionLanguageModel, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.vision_cache = VisualTokenCache(max_size=10000)

    def infer(
        self,
        image: torch.Tensor,
        prompt: str,
        use_cache: bool = True
    ) -> Tuple[str, float]:
        """Infer with optional caching."""
        start_time = time.time()

        # Check cache
        if use_cache:
            visual_tokens = self.vision_cache.get(image)

            if visual_tokens is None:
                # Cache miss: encode vision
                with torch.no_grad():
                    visual_tokens = self.model.encode_image(image.unsqueeze(0).to(self.device))
                self.vision_cache.put(image, visual_tokens)
        else:
            # No cache
            with torch.no_grad():
                visual_tokens = self.model.encode_image(image.unsqueeze(0).to(self.device))

        # Generate text
        text_ids = torch.tensor([[1]])  # BOS token
        with torch.no_grad():
            generated = self.model.generate(image, text_ids, max_new_tokens=50)

        elapsed = (time.time() - start_time) * 1000

        # Log cache performance
        stats = self.vision_cache.get_stats()
        if stats['hits'] + stats['misses'] > 100:
            print(f"[Cache] Hit rate: {stats['hit_rate']:.1f}%")

        return "Generated response", elapsed
```

---

## 3. Audio-Language Models

### 3.1 Whisper/WhisperX: Speech-to-Text

```python
import librosa
import numpy as np
from typing import Tuple, List
import time

class AudioEncoder:
    """
    Encode audio to tokens efficiently.
    Focus: WhisperX optimizations.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 400
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft

    def preprocess_audio(
        self,
        audio: np.ndarray,
        target_length_s: float = 30.0
    ) -> np.ndarray:
        """
        Preprocess audio to mel-spectrogram.
        """
        # Resample if needed
        if audio.shape[0] != self.sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=audio.shape[0],
                target_sr=self.sample_rate
            )

        # Pad/truncate to target length
        target_samples = int(target_length_s * self.sample_rate)
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        else:
            audio = audio[:target_samples]

        # Convert to mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft
        )

        # Log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        mel_spec_db = (mel_spec_db + 80) / 80

        return mel_spec_db

    def batch_preprocess(
        self,
        audio_list: List[np.ndarray],
        max_length_s: float = 30.0
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Preprocess batch of audio with variable lengths.
        Returns: (padded_batch, lengths)
        """
        processed = []
        lengths = []

        for audio in audio_list:
            mel_spec = self.preprocess_audio(audio, max_length_s)
            processed.append(mel_spec)
            lengths.append(mel_spec.shape[1])

        # Pad to max length in batch
        max_len = max(lengths)
        padded = []

        for mel_spec in processed:
            if mel_spec.shape[1] < max_len:
                mel_spec = np.pad(
                    mel_spec,
                    ((0, 0), (0, max_len - mel_spec.shape[1])),
                    mode='constant'
                )
            padded.append(mel_spec)

        batch = np.stack(padded)  # (batch, n_mels, time_steps)
        return batch, lengths

class AudioLanguageModel(nn.Module):
    """
    Audio + Language model (Qwen-Audio style).
    """

    def __init__(
        self,
        audio_embedding_dim: int = 512,
        language_dim: int = 4096,
        n_mels: int = 128
    ):
        super().__init__()

        # Audio encoder (simplified)
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(n_mels, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(256, audio_embedding_dim, kernel_size=3, stride=2)
        )

        # Project to language space
        self.audio_projection = nn.Linear(audio_embedding_dim, language_dim)

        # Language model
        self.language_model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=language_dim, nhead=32),
            num_layers=24
        )

    def encode_audio(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Encode mel-spectrogram to audio tokens.
        mel_spec: (batch, n_mels, time_steps)
        returns: (batch, audio_tokens, language_dim)
        """
        # Conv encoding
        encoded = self.audio_encoder(mel_spec)  # (batch, audio_dim, reduced_time)

        # Transpose for language model
        encoded = encoded.transpose(1, 2)  # (batch, reduced_time, audio_dim)

        # Project to language space
        audio_tokens = self.audio_projection(encoded)

        return audio_tokens

    def transcribe(
        self,
        mel_spec: torch.Tensor,
        max_tokens: int = 500
    ) -> str:
        """
        Transcribe audio to text.
        """
        audio_tokens = self.encode_audio(mel_spec)

        # Language model decoding (simplified)
        text_ids = torch.ones((mel_spec.shape[0], 1), dtype=torch.long)

        for _ in range(max_tokens):
            # Decode
            output = self.language_model(text_ids, audio_tokens)

            # Sample next token
            logits = output[:, -1, :100]  # Project to vocab
            next_token = logits.argmax(dim=-1, keepdim=True)

            text_ids = torch.cat([text_ids, next_token], dim=1)

            # Check for EOS
            if next_token[0, 0] == 50257:  # EOS token
                break

        return "transcribed text"  # Placeholder
```

### 3.2 Streaming Audio Processing

```python
import queue
from collections import deque

class StreamingAudioProcessor:
    """
    Process audio stream in real-time chunks.
    Used for real-time transcription and voice assistants.
    """

    def __init__(
        self,
        model: AudioLanguageModel,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,
        context_window_ms: int = 1000
    ):
        self.model = model
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        self.context_window_samples = int(sample_rate * context_window_ms / 1000)

        # Buffer for streaming
        self.audio_buffer = deque(maxlen=self.context_window_samples)
        self.output_queue = queue.Queue()

    def process_chunk(self, chunk: np.ndarray):
        """Process incoming audio chunk."""
        # Add to buffer
        self.audio_buffer.extend(chunk)

        # If buffer is full, process
        if len(self.audio_buffer) >= self.context_window_samples:
            audio_array = np.array(list(self.audio_buffer))

            # Convert to mel-spectrogram
            encoder = AudioEncoder()
            mel_spec = encoder.preprocess_audio(audio_array)

            # Encode and transcribe
            mel_tensor = torch.from_numpy(mel_spec[np.newaxis, :, :]).float()
            transcription = self.model.transcribe(mel_tensor)

            # Queue output
            self.output_queue.put({
                'transcription': transcription,
                'timestamp': time.time()
            })

    def get_transcription(self, timeout_s: float = 1.0) -> Optional[str]:
        """Get next transcription."""
        try:
            result = self.output_queue.get(timeout=timeout_s)
            return result['transcription']
        except queue.Empty:
            return None
```

---

## 4. Multi-Modal Batching & Scheduling

### 4.1 Heterogeneous Batch Management

```python
from dataclasses import dataclass
from enum import Enum
from typing import Union

class ModalityType(Enum):
    IMAGE = "image"
    AUDIO = "audio"
    TEXT = "text"

@dataclass
class MultiModalRequest:
    """Single multi-modal request."""
    request_id: str
    modalities: dict  # modality -> data
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class MultiModalBatcher:
    """
    Batch requests with heterogeneous modalities.
    Key challenge: different shapes for image (384x384) vs audio (3000 samples).
    """

    def __init__(
        self,
        batch_timeout_ms: float = 50,
        max_batch_size: int = 32
    ):
        self.batch_timeout_ms = batch_timeout_ms
        self.max_batch_size = max_batch_size
        self.request_queue = asyncio.Queue()

    async def add_request(self, request: MultiModalRequest):
        """Add request to queue."""
        await self.request_queue.put(request)

    async def get_batch(self) -> List[MultiModalRequest]:
        """
        Collect batch with timeout.
        Returns when batch is full or timeout expires.
        """
        batch = []
        start_time = time.time()

        while len(batch) < self.max_batch_size:
            elapsed_ms = (time.time() - start_time) * 1000

            if elapsed_ms > self.batch_timeout_ms and batch:
                break

            wait_time_ms = self.batch_timeout_ms - elapsed_ms
            wait_time_s = max(0, wait_time_ms / 1000)

            try:
                request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=wait_time_s
                )
                batch.append(request)
            except asyncio.TimeoutError:
                break

        return batch

class DynamicMultiModalRouter:
    """
    Route requests to appropriate inference engine based on modality.
    """

    def __init__(
        self,
        vlm_model,
        audio_model,
        text_model,
        device: str = 'cpu'
    ):
        self.vlm_model = vlm_model
        self.audio_model = audio_model
        self.text_model = text_model
        self.device = device

    async def infer(self, request: MultiModalRequest) -> dict:
        """Route and infer based on modalities."""
        result = {}
        timing = {}

        # Route by modality
        if 'image' in request.modalities:
            start = time.time()
            image_result = await self._infer_vlm(request)
            timing['image'] = (time.time() - start) * 1000
            result.update(image_result)

        if 'audio' in request.modalities:
            start = time.time()
            audio_result = await self._infer_audio(request)
            timing['audio'] = (time.time() - start) * 1000
            result.update(audio_result)

        if 'text' in request.modalities:
            start = time.time()
            text_result = await self._infer_text(request)
            timing['text'] = (time.time() - start) * 1000
            result.update(text_result)

        result['timing'] = timing
        return result

    async def _infer_vlm(self, request: MultiModalRequest) -> dict:
        """Infer with VLM."""
        image = request.modalities['image']
        # Run VLM
        return {'vision_understanding': 'analyzed image'}

    async def _infer_audio(self, request: MultiModalRequest) -> dict:
        """Infer with audio model."""
        audio = request.modalities['audio']
        # Run audio model
        return {'transcription': 'transcribed text'}

    async def _infer_text(self, request: MultiModalRequest) -> dict:
        """Infer with text model."""
        text = request.modalities['text']
        # Run language model
        return {'response': 'generated response'}
```

---

## 5. Efficient Vision Transformer on CPU

For mobile and edge deployment, ViT must be CPU-optimized:

```python
class EfficientViT(nn.Module):
    """
    Vision Transformer optimized for CPU inference.
    Techniques: depthwise separable convolutions, int8 quantization, AVX-512 kernels.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        patch_size: int = 16,
        num_patches: int = 196,
        num_layers: int = 12
    ):
        super().__init__()

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # Efficient patch embedding using depthwise separable conv
        self.patch_embed = DepthwiseSeparableConv(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Efficient attention layers
        self.layers = nn.ModuleList([
            EfficientAttentionLayer(hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 3, 224, 224)
        returns: (batch, num_patches, hidden_dim)
        """
        # Efficient patch embedding
        x = self.patch_embed(x)  # (batch, hidden_dim, num_patches, 1)
        x = x.squeeze(-1).transpose(1, 2)  # (batch, num_patches, hidden_dim)

        # Efficient attention
        for layer in self.layers:
            x = layer(x)

        return x

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: 8-9x fewer parameters than standard."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0
    ):
        super().__init__()

        # Depthwise: per-channel convolution
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )

        # Pointwise: 1x1 convolution
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class EfficientAttentionLayer(nn.Module):
    """
    Efficient attention for CPU:
    - Local attention (only attend to nearby patches)
    - Linear attention approximation
    """

    def __init__(
        self,
        dim: int,
        window_size: int = 7,
        use_linear_attention: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.use_linear_attention = use_linear_attention

        if use_linear_attention:
            # Linear attention: O(n) instead of O(n^2)
            self.attention = LinearAttention(dim)
        else:
            # Local window attention
            self.attention = LocalWindowAttention(dim, window_size)

    def forward(self, x):
        return self.attention(x)

class LinearAttention(nn.Module):
    """Linear attention approximation using kernel methods."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: (batch, seq_len, dim)
        Linear attention: avoid computing full similarity matrix.
        """
        Q = self.q_proj(x)  # (batch, seq_len, dim)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Linear kernel (instead of softmax)
        # More stable for long sequences
        Q = torch.nn.functional.elu(Q) + 1
        K = torch.nn.functional.elu(K) + 1

        # Compute attention: (Q @ K^T) @ V, but without materializing Q @ K^T
        # Use associativity: Q @ (K^T @ V)
        K_sum = K.sum(dim=1, keepdim=True)  # (batch, 1, dim)
        KV = torch.einsum('bnd,bne->bde', K, V)  # (batch, dim, dim)
        output = torch.einsum('bnd,bde->bne', Q, KV)  # (batch, seq_len, dim)

        # Normalize
        Q_sum = Q.sum(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        output = output / (Q_sum @ K_sum.transpose(1, 2))

        return output

class LocalWindowAttention(nn.Module):
    """Attention only within local windows."""

    def __init__(self, dim: int, window_size: int = 7):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)

    def forward(self, x):
        """
        x: (batch, seq_len, dim)
        Compute attention within sliding windows.
        """
        batch_size, seq_len, dim = x.shape
        output = torch.zeros_like(x)

        for start_idx in range(0, seq_len, self.window_size):
            end_idx = min(start_idx + self.window_size, seq_len)
            window = x[:, start_idx:end_idx, :]

            attn_output, _ = self.attention(window, window, window)
            output[:, start_idx:end_idx, :] = attn_output

        return output

# Benchmark CPU-optimized ViT
def benchmark_vit_cpu():
    """Benchmark ViT on CPU."""
    model = EfficientViT(
        hidden_dim=256,
        num_layers=12
    )
    model.eval()

    # Dummy input
    x = torch.randn(1, 3, 224, 224)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(x)

    # Timing
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            model(x)
        elapsed = time.time() - start

    latency_ms = (elapsed / 100) * 1000
    print(f"[Benchmark] CPU ViT: {latency_ms:.1f}ms per image")

    return latency_ms
```

---

## 6. Production Multi-Modal Patterns

### 6.1 Orchestration & Pipelines

```python
class MultiModalPipeline:
    """
    End-to-end multi-modal inference pipeline.
    Orchestrates heterogeneous models with latency budgets.
    """

    def __init__(
        self,
        vlm_model,
        audio_model,
        text_model,
        target_latency_ms: float = 500
    ):
        self.vlm_model = vlm_model
        self.audio_model = audio_model
        self.text_model = text_model
        self.target_latency_ms = target_latency_ms

    async def process(
        self,
        request: MultiModalRequest
    ) -> dict:
        """Process multi-modal request with latency optimization."""
        pipeline_start = time.time()
        results = {}

        # Allocate latency budget proportionally
        modalities = list(request.modalities.keys())
        latency_per_modality = self.target_latency_ms / len(modalities)

        # Process in parallel where possible
        tasks = []

        if 'image' in request.modalities:
            tasks.append(
                self._infer_with_budget(
                    'vlm',
                    request.modalities['image'],
                    latency_per_modality
                )
            )

        if 'audio' in request.modalities:
            tasks.append(
                self._infer_with_budget(
                    'audio',
                    request.modalities['audio'],
                    latency_per_modality
                )
            )

        # Run in parallel
        if tasks:
            parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
            for modality, result in zip(['vlm', 'audio'], parallel_results):
                if not isinstance(result, Exception):
                    results[modality] = result

        # Fusion: combine multi-modal results
        if len(results) > 1:
            fused_result = self._fuse_modalities(results)
            results['fused'] = fused_result

        total_time = (time.time() - pipeline_start) * 1000

        return {
            'results': results,
            'latency_ms': total_time,
            'budget_ms': self.target_latency_ms
        }

    async def _infer_with_budget(
        self,
        modality: str,
        data,
        budget_ms: float
    ) -> dict:
        """Infer with latency budget."""
        start = time.time()

        if modality == 'vlm':
            result = await asyncio.to_thread(self.vlm_model.infer, data)
        elif modality == 'audio':
            result = await asyncio.to_thread(self.audio_model.infer, data)
        else:
            result = {}

        elapsed = (time.time() - start) * 1000

        if elapsed > budget_ms * 1.1:  # 10% over budget
            print(f"[Pipeline] {modality} overbudget: {elapsed:.0f}ms > {budget_ms:.0f}ms")

        return result

    def _fuse_modalities(self, results: dict) -> dict:
        """Combine results from multiple modalities."""
        # Example: combine vision understanding + audio transcription
        return {
            'unified_understanding': 'combined multi-modal analysis',
            'confidence': 0.95
        }
```

---

## 7. Latency Analysis & Optimization

```python
class MultiModalLatencyAnalysis:
    """
    Analyze and optimize multi-modal latency.
    """

    @staticmethod
    def profile_vlm_components():
        """Profile VLM inference stages."""
        print("\n=== VLM Latency Breakdown ===")
        print("┌─────────────────────┬────────┬──────────┐")
        print("│ Component           │ Time   │ Percent  │")
        print("├─────────────────────┼────────┼──────────┤")
        print("│ Image preprocessing │ 10ms   │  5%      │")
        print("│ Vision encoding     │ 150ms  │ 75%      │")
        print("│ LLM prefill         │ 30ms   │ 15%      │")
        print("│ Total               │ 190ms  │ 100%     │")
        print("└─────────────────────┴────────┴──────────┘")

        print("\nOptimization opportunities:")
        print("1. Vision encoding dominates (75%)")
        print("   - Reduce image resolution: 384x384 -> 224x224")
        print("   - Use efficient ViT or CNNs")
        print("   - Quantize vision encoder (INT8)")
        print("\n2. LLM prefill is <20%")
        print("   - Adequate, not bottleneck")

    @staticmethod
    def profile_audio_components():
        """Profile audio model latency."""
        print("\n=== Audio Model Latency Breakdown ===")
        print("┌─────────────────────┬────────┬──────────┐")
        print("│ Component           │ Time   │ Percent  │")
        print("├─────────────────────┼────────┼──────────┤")
        print("│ Audio preprocessing │ 20ms   │ 10%      │")
        print("│ Audio encoding      │ 120ms  │ 60%      │")
        print("│ ASR decoding        │ 60ms   │ 30%      │")
        print("│ Total               │ 200ms  │ 100%     │")
        print("└─────────────────────┴────────┴──────────┘")

        print("\nOptimization opportunities:")
        print("1. Parallel preprocessing with streaming")
        print("2. Shorter context window for faster encoding")
```

---

## 8. Summary & Key Takeaways

This module covered multi-modal inference systems:

1. **VLM Inference**: Vision encoding (expensive) + language generation (light)
2. **Visual Token Caching**: 80-90% cache hit rate for repeated images
3. **Audio Models**: Streaming processing for real-time transcription
4. **Multi-modal Batching**: Handle heterogeneous shapes and sizes
5. **Efficient Vision Transformers**: Linear attention, local windows, depthwise convolution
6. **Orchestration**: Parallel inference with latency budgets

**Key metrics:**
- Vision encoding: 100-200ms (bottleneck)
- LLM generation: 50-100ms
- Audio encoding: 80-150ms
- Vision cache hit rate: 60-90% for repeated data
- Total latency target: <500ms for document understanding

**Next steps**: Start with baseline VLM, profile to identify bottleneck (usually vision), apply vision token caching, optimize slowest component, and measure end-to-end improvement.

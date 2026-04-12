# MODULE 37 — Inference Compilation & Deployment Toolchain

## 1. Introduction & Scope

The path from a trained PyTorch or TensorFlow model to a production inference service involves five critical stages: (1) checkpoint export to a standard format, (2) optimization passes (quantization, pruning, operator fusion), (3) compilation to hardware-specific targets, (4) containerization with proper resource isolation, and (5) serving frameworks that manage batching, versioning, and multi-tenancy.

This module covers the complete inference toolchain: model export formats (ONNX, TorchScript, torch.export), the ONNX ecosystem and execution providers, comparison of serving frameworks (Triton, TorchServe, OpenVINO Model Server, Ollama, vLLM), containerization details (NUMA locality, CPU affinity, huge page setup), and production deployment patterns.

The business case is compelling: inference serving is 3-10x more cost-sensitive than training (higher volume, lower per-unit margin). A well-tuned inference pipeline can reduce per-token cost by 50-70% compared to naive deployment. Conversely, poor choices (wrong quantization, suboptimal batching, inefficient serialization) can increase costs by 200%.

We target three deployment scenarios: (1) high-throughput batch serving (LLMs, 1000+ req/sec), (2) low-latency interactive serving (RAG, chat, <100ms P99), and (3) edge/mobile inference (quantized models, <5W power).

---

## 2. Model Export & Format Choices

### 2.1 Export Formats: ONNX vs. TorchScript vs. torch.export

```python
import torch
import torch.nn as nn
import torch.onnx
from typing import Dict, Any
import numpy as np
import time

class SimpleTransformer(nn.Module):
    """Minimal Transformer block for demonstration."""

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, 3072)
        self.linear2 = nn.Linear(3072, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = x + residual
        return x

class ModelExporter:
    """
    Export PyTorch models to multiple formats with validation.
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)

    def export_onnx(
        self,
        output_path: str,
        input_shape: tuple = (1, 128, 768),
        opset_version: int = 14,
        dynamic_axes: Dict[str, Dict[int, str]] = None
    ):
        """
        Export to ONNX (Open Neural Network Exchange).
        ONNX is hardware-agnostic but requires careful opset selection.
        """
        print(f"[ONNX Export] Exporting to {output_path}")

        # Create dummy input
        dummy_input = torch.randn(input_shape, device=self.device)

        # Default dynamic axes for batch and sequence length
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size', 1: 'sequence_length'}
            }

        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=False
            )
            print(f"[ONNX Export] Success. Size: {self._get_file_size(output_path)}")

            # Validate
            self._validate_onnx(output_path, input_shape)

        except Exception as e:
            print(f"[ONNX Export] Failed: {e}")
            raise

    def export_torchscript(
        self,
        output_path: str,
        input_shape: tuple = (1, 128, 768),
        method: str = 'trace'  # 'trace' or 'script'
    ):
        """
        Export to TorchScript (PyTorch native format).
        Trace is faster but requires representative input.
        Script is slower but handles conditionals.
        """
        print(f"[TorchScript Export] Using {method} method")

        try:
            if method == 'trace':
                # Tracing: run model once to capture operations
                dummy_input = torch.randn(input_shape, device=self.device)
                scripted = torch.jit.trace(self.model, dummy_input)
            else:
                # Scripting: compile source code
                scripted = torch.jit.script(self.model)

            scripted.save(output_path)
            print(f"[TorchScript Export] Success. Size: {self._get_file_size(output_path)}")

        except Exception as e:
            print(f"[TorchScript Export] Failed: {e}")
            raise

    def export_torch_export(
        self,
        output_path: str,
        input_shape: tuple = (1, 128, 768)
    ):
        """
        Export using torch.export (PyTorch 2.0+).
        Most advanced: captures full computation graph with guards.
        """
        print("[torch.export] Exporting with torch.export")

        try:
            dummy_input = torch.randn(input_shape, device=self.device)

            # Capture with torch.export
            exported = torch.export.export(
                self.model,
                (dummy_input,)
            )

            # Serialize
            with open(output_path, 'wb') as f:
                torch.save(exported, f)

            print(f"[torch.export] Success. Size: {self._get_file_size(output_path)}")

        except Exception as e:
            print(f"[torch.export] Failed: {e}")
            raise

    def benchmark_inference(
        self,
        input_shape: tuple = (1, 128, 768),
        num_iterations: int = 100
    ):
        """Benchmark inference latency."""
        print("\n[Benchmark] Running inference latency test...")

        dummy_input = torch.randn(input_shape, device=self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                self.model(dummy_input)

        # Timing
        torch.cuda.synchronize() if self.device == 'cuda' else None
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_iterations):
                self.model(dummy_input)

        torch.cuda.synchronize() if self.device == 'cuda' else None
        elapsed = time.time() - start_time

        latency_ms = (elapsed / num_iterations) * 1000
        throughput = num_iterations / elapsed

        print(f"[Benchmark] Latency: {latency_ms:.2f}ms")
        print(f"[Benchmark] Throughput: {throughput:.0f} iter/sec")

        return {'latency_ms': latency_ms, 'throughput': throughput}

    def _validate_onnx(self, onnx_path: str, input_shape: tuple):
        """Validate ONNX model by comparing outputs."""
        try:
            import onnx
            import onnxruntime as ort

            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print(f"[ONNX Validate] Model structure valid")

            # Compare outputs
            sess = ort.InferenceSession(onnx_path)
            dummy_input = torch.randn(input_shape).numpy().astype(np.float32)

            with torch.no_grad():
                torch_output = self.model(torch.from_numpy(dummy_input)).cpu().numpy()

            onnx_output = sess.run(None, {'input': dummy_input})[0]

            diff = np.abs(torch_output - onnx_output).max()
            print(f"[ONNX Validate] Output difference: {diff:.2e}")

            if diff > 1e-3:
                print(f"[ONNX Validate] WARNING: Large difference detected")

        except ImportError:
            print("[ONNX Validate] onnx/onnxruntime not installed, skipping")

    @staticmethod
    def _get_file_size(path: str) -> str:
        """Get human-readable file size."""
        import os
        size_bytes = os.path.getsize(path)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}TB"

# Example: Compare export formats
def demo_export_formats():
    """Demonstrate exporting to different formats."""
    model = SimpleTransformer(hidden_dim=768)
    exporter = ModelExporter(model, device='cpu')

    input_shape = (2, 128, 768)

    # Benchmark baseline
    print("=== Baseline (PyTorch) ===")
    baseline = exporter.benchmark_inference(input_shape)

    # Export to ONNX
    print("\n=== ONNX Export ===")
    exporter.export_onnx('model.onnx', input_shape)

    # Export to TorchScript
    print("\n=== TorchScript Export ===")
    exporter.export_torchscript('model.pt', input_shape)

    # Export with torch.export
    print("\n=== torch.export ===")
    try:
        exporter.export_torch_export('model_exported.pt', input_shape)
    except Exception as e:
        print(f"torch.export not available: {e}")
```

### 2.2 Model Optimization Before Export

```python
import torch.quantization as quantization
from torch.utils.mobile_optimizer import optimize_for_mobile

class ModelOptimizer:
    """
    Apply optimization passes before export.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.original_state = model.state_dict().copy()

    def quantize_dynamic(self) -> nn.Module:
        """
        Dynamic quantization: only weights quantized.
        Fastest to apply, good for CPU inference.
        """
        print("[Quantize] Applying dynamic quantization...")

        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )

        return quantized_model

    def quantize_static(
        self,
        train_dataloader,
        backend: str = 'fbgemm'  # 'fbgemm' (Intel), 'qnnpack' (ARM)
    ) -> nn.Module:
        """
        Static quantization: weights and activations quantized.
        Requires calibration data, better accuracy than dynamic.
        """
        print("[Quantize] Applying static quantization...")

        # Set backend
        torch.backends.quantized.engine = backend

        # Prepare model
        self.model.qconfig = torch.quantization.get_default_qconfig(backend)
        quantization.prepare(self.model, inplace=True)

        # Calibration: run on sample data
        print("[Quantize] Running calibration...")
        self.model.eval()
        with torch.no_grad():
            for batch in train_dataloader:
                self.model(batch)

        # Convert to quantized
        quantization.convert(self.model, inplace=True)

        return self.model

    def prune_weights(self, amount: float = 0.3) -> nn.Module:
        """
        Structured pruning: remove entire channels/filters.
        amount: fraction of weights to prune [0, 1]
        """
        print(f"[Prune] Pruning {amount*100:.0f}% of weights...")

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                torch.nn.utils.prune.l1_unstructured(
                    module, name='weight', amount=amount
                )
                # Remove pruning reparameterization
                torch.nn.utils.prune.remove(module, 'weight')

        return self.model

    def fuse_modules(self) -> nn.Module:
        """
        Operator fusion: combine consecutive operations.
        E.g., Linear + ReLU -> single fused operation.
        """
        print("[Fuse] Fusing modules...")

        # Identify fusible patterns
        torch.quantization.fuse_modules(
            self.model,
            [['linear1', 'norm'], ['linear2', 'gelu']],
            inplace=True
        )

        return self.model

    def benchmark_optimizations(
        self,
        input_shape: tuple = (4, 128, 768)
    ) -> Dict[str, Any]:
        """Benchmark different optimizations."""
        dummy_input = torch.randn(input_shape)

        results = {}

        # Baseline
        self.model.eval()
        with torch.no_grad():
            start = time.time()
            for _ in range(100):
                self.model(dummy_input)
            baseline_time = (time.time() - start) / 100

        results['baseline'] = baseline_time
        print(f"Baseline: {baseline_time*1000:.2f}ms")

        # Dynamic quantization
        try:
            quant_model = self.quantize_dynamic()
            quant_model.eval()
            with torch.no_grad():
                start = time.time()
                for _ in range(100):
                    quant_model(dummy_input)
                quant_time = (time.time() - start) / 100
            results['quantized'] = quant_time
            speedup = baseline_time / quant_time
            print(f"Quantized: {quant_time*1000:.2f}ms ({speedup:.1f}x speedup)")
        except Exception as e:
            print(f"Quantization failed: {e}")

        return results
```

---

## 3. ONNX Ecosystem Deep Dive

### 3.1 ONNX Execution Providers

ONNX Runtime supports multiple execution providers, each optimized for different hardware:

```python
import onnxruntime as ort
import numpy as np
from typing import List, Dict
import time

class ONNXRuntimeBenchmark:
    """
    Benchmark ONNX models across different execution providers.
    """

    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path
        self.available_providers = ort.get_available_providers()
        print(f"[ONNX] Available providers: {self.available_providers}")

    def benchmark_all_providers(
        self,
        input_name: str,
        input_shape: tuple,
        num_iterations: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark inference across all available providers.
        """
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        input_dict = {input_name: dummy_input}

        results = {}

        for provider in self.available_providers:
            print(f"\n[Benchmark] Testing {provider}...")

            try:
                sess = ort.InferenceSession(
                    self.onnx_path,
                    providers=[provider]
                )

                # Warmup
                for _ in range(10):
                    sess.run(None, input_dict)

                # Timing
                start_time = time.time()
                for _ in range(num_iterations):
                    sess.run(None, input_dict)
                elapsed = time.time() - start_time

                latency_ms = (elapsed / num_iterations) * 1000
                throughput = num_iterations / elapsed

                results[provider] = {
                    'latency_ms': latency_ms,
                    'throughput': throughput
                }

                print(f"  Latency: {latency_ms:.2f}ms")
                print(f"  Throughput: {throughput:.0f} iter/sec")

            except Exception as e:
                print(f"  Failed: {e}")
                results[provider] = {'error': str(e)}

        return results

    @staticmethod
    def list_opset_support():
        """
        List which operators are supported in different opsets.
        Important for compatibility.
        """
        print("\n[ONNX] Opset Support:")
        print("  opset_version=13: No dynamic shapes for LSTM/GRU")
        print("  opset_version=14: Improved dynamic shapes, Add quantization operators")
        print("  opset_version=15: Inline graphs for control flow")
        print("  opset_version=16+: Latest operators, best compatibility")

# Execution provider details
EXECUTION_PROVIDERS = {
    'CPUExecutionProvider': {
        'hardware': 'CPU',
        'latency': 'Medium',
        'power': 'Medium',
        'setup': 'Always available',
        'ideal_for': 'Fallback, edge devices'
    },
    'CUDAExecutionProvider': {
        'hardware': 'NVIDIA GPU',
        'latency': 'Low',
        'power': 'High',
        'setup': 'Requires CUDA 11.6+',
        'ideal_for': 'High throughput, RTX/H100'
    },
    'TensorrtExecutionProvider': {
        'hardware': 'NVIDIA GPU',
        'latency': 'Very Low',
        'power': 'High',
        'setup': 'Requires TensorRT 8.0+',
        'ideal_for': 'Production batch inference'
    },
    'CoreMLExecutionProvider': {
        'hardware': 'Apple Silicon',
        'latency': 'Low',
        'power': 'Low',
        'setup': 'macOS/iOS only',
        'ideal_for': 'Mobile, M1/M2/M3/M4'
    },
    'QNNExecutionProvider': {
        'hardware': 'Snapdragon (ARM)',
        'latency': 'Low',
        'power': 'Low',
        'setup': 'Snapdragon 8/Gen3',
        'ideal_for': 'Mobile Android'
    }
}

def print_provider_matrix():
    """Print comparison of execution providers."""
    print("\n=== ONNX Execution Providers ===\n")
    for provider, details in EXECUTION_PROVIDERS.items():
        print(f"{provider}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
        print()
```

---

## 4. Serving Frameworks Comparison

### 4.1 Triton Inference Server

Triton is the industry standard for high-performance, multi-model serving.

```python
import requests
import json
import numpy as np
import time
from typing import List, Dict

class TritonClient:
    """
    Client for NVIDIA Triton Inference Server.
    Supports batching, dynamic shapes, multiple models.
    """

    def __init__(self, url: str = "localhost:8000"):
        self.url = f"http://{url}"
        self.models_url = f"{self.url}/v2/models"

    def list_models(self) -> List[str]:
        """List available models."""
        response = requests.get(f"{self.models_url}")
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [m['name'] for m in models]
        return []

    def infer(
        self,
        model_name: str,
        input_data: Dict[str, np.ndarray],
        output_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on Triton.
        Supports dynamic batching and model ensemble.
        """
        # Prepare request
        request = {
            "inputs": [
                {
                    "name": k,
                    "shape": v.shape,
                    "datatype": self._numpy_to_triton_dtype(v.dtype),
                    "data": v.tolist()
                }
                for k, v in input_data.items()
            ],
            "outputs": [
                {"name": output_name}
                for output_name in output_names
            ]
        }

        # Send request
        infer_url = f"{self.models_url}/{model_name}/infer"
        response = requests.post(infer_url, json=request)

        if response.status_code != 200:
            raise RuntimeError(f"Inference failed: {response.text}")

        # Parse response
        result = response.json()
        outputs = {}
        for output in result['outputs']:
            output_name = output['name']
            output_shape = output['shape']
            output_data = np.array(output['data'])
            outputs[output_name] = output_data.reshape(output_shape)

        return outputs

    @staticmethod
    def _numpy_to_triton_dtype(dtype) -> str:
        """Convert numpy dtype to Triton dtype string."""
        dtype_map = {
            np.float32: 'FP32',
            np.float64: 'FP64',
            np.int32: 'INT32',
            np.int64: 'INT64',
            np.uint8: 'UINT8',
            np.uint32: 'UINT32'
        }
        return dtype_map.get(dtype, 'FP32')

# Triton configuration example
TRITON_CONFIG = """
# model_repository/transformer/config.pbtxt
name: "transformer"
platform: "onnxruntime_onnx"
max_batch_size: 256
default_model_filename: "model.onnx"

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [-1, 128, 768]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, 128, 768]
  }
]

# Dynamic batching configuration
dynamic_batching {
  max_queue_delay_microseconds: 1000
  preferred_batch_size: [32, 64, 128]
  max_enqueued_requests: 10000
}

# Model ensemble: combine multiple models
# ensemble_scheduling {
#   step [
#     {
#       model_name: "preprocessing"
#       model_version: -1
#       input_map { key: "input" value: "input" }
#       output_map { key: "output" value: "prep_output" }
#     },
#     {
#       model_name: "transformer"
#       model_version: -1
#       input_map { key: "input" value: "prep_output" }
#       output_map { key: "output" value: "output" }
#     }
#   ]
# }
"""
```

### 4.2 vLLM: LLM-Optimized Serving

vLLM is purpose-built for LLM inference with PagedAttention:

```python
from typing import List, Optional, Dict
import asyncio

class vLLMClient:
    """
    Client for vLLM inference engine.
    Optimized for autoregressive LLM inference with KV-cache.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    async def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate response using vLLM.
        Supports streaming, top-k, top-p sampling.
        """
        import httpx

        async with httpx.AsyncClient() as client:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop_sequences or []
            }

            response = await client.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=30.0
            )

            if response.status_code != 200:
                raise RuntimeError(f"vLLM failed: {response.text}")

            result = response.json()
            return result['choices'][0]['text']

    async def batch_generate(
        self,
        model: str,
        prompts: List[str],
        max_tokens: int = 256,
        **kwargs
    ) -> List[str]:
        """
        Batch generation: vLLM automatically batches and schedules.
        Key advantage: Paged Attention allows flexible batching.
        """
        # vLLM queues all requests and batches them internally
        tasks = [
            self.generate(model, prompt, max_tokens, **kwargs)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

# vLLM launch command
VLLM_LAUNCH = """
# vLLM is extremely efficient for batch serving
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs 256 \
  --dtype float16 \
  --port 8000
"""

VLLM_ADVANTAGES = {
    'PagedAttention': 'Flexible KV cache, no memory fragmentation',
    'Prefix caching': 'Reuse KV for repeated prefixes',
    'Chunked prefill': 'Overlap prefill with decode',
    'Throughput': '5-10x better than baseline',
    'Latency': '20-40ms for 7B decode token, 100-200ms for prefill'
}
```

### 4.3 TorchServe: PyTorch Native

```python
class TorchServeConfig:
    """
    TorchServe configuration for production deployment.
    """

    @staticmethod
    def handler_template() -> str:
        return '''
import torch
from ts.torch_handler.base_handler import BaseHandler

class TransformerHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.model = None

    def initialize(self, context):
        """Called once at startup."""
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = torch.load(f"{model_dir}/model.pt")
        self.model.eval()

    def preprocess(self, data):
        """Preprocess batch of requests."""
        return torch.tensor(data)

    def inference(self, model_input):
        """Run inference."""
        with torch.no_grad():
            return self.model(model_input)

    def postprocess(self, inference_output):
        """Format output."""
        return inference_output.tolist()
'''

    @staticmethod
    def config_properties() -> str:
        return '''
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082

enable_metrics_api=true

# Model configuration
default_workers_per_gpu=1
default_response_timeout=30
default_request_timeout=30

# Batching
ts_inference_address=http://0.0.0.0:8080

# Logging
log_location=/var/log/torchserve
metrics_location=/var/log/torchserve
'''

# Launch TorchServe
TORCHSERVE_LAUNCH = """
# Package model
torch-model-archiver \
  --model-name transformer \
  --version 1.0 \
  --model-file model.py \
  --serialized-file model.pt \
  --handler handler.py \
  --extra-files index_to_name.json

# Start server
torchserve --start --model-store model_store --models all

# Scale workers
curl -X PUT "http://localhost:8081/models/transformer?min_worker=4&max_worker=8"
"""
```

---

## 5. Containerization & Resource Management

### 5.1 NUMA Locality in Docker

Modern CPUs are NUMA (Non-Uniform Memory Access). Proper NUMA binding can improve latency 20-30%.

```python
import subprocess
import os
from typing import List

class NUMAOptimizer:
    """
    NUMA-aware containerization and process binding.
    """

    @staticmethod
    def detect_numa_topology() -> dict:
        """Detect NUMA topology."""
        try:
            result = subprocess.run(['numactl', '--hardware'], capture_output=True, text=True)
            print("[NUMA] Topology:")
            print(result.stdout)
            return {'raw': result.stdout}
        except FileNotFoundError:
            print("[NUMA] numactl not found, NUMA optimization unavailable")
            return {}

    @staticmethod
    def dockerfile_with_numa() -> str:
        """Dockerfile with NUMA-aware resource allocation."""
        return '''
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install numactl
RUN apt-get update && apt-get install -y numactl

# Copy application
COPY . /app
WORKDIR /app

# Install dependencies
RUN pip install -r requirements.txt

# ENTRYPOINT: bind to NUMA node 0
ENTRYPOINT ["numactl", "--cpunodebind=0", "--membind=0", "python", "serve.py"]
'''

    @staticmethod
    def kubernetes_numa_affinity() -> str:
        """Kubernetes Pod manifest with NUMA awareness."""
        return '''
apiVersion: v1
kind: Pod
metadata:
  name: inference-server
spec:
  affinity:
    # Pin to specific CPU nodes
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: kubernetes.io/hostname
            operator: In
            values:
            - worker-1

  containers:
  - name: server
    image: inference-server:latest
    resources:
      requests:
        memory: "32Gi"
        cpu: "16"
      limits:
        memory: "32Gi"
        cpu: "16"

    # CPU affinity: cores 0-15 on NUMA node 0
    env:
    - name: PYTHONUNBUFFERED
      value: "1"
    - name: OMP_NUM_THREADS
      value: "16"
    - name: GOMP_CPU_AFFINITY
      value: "0-15"

  # Ensure all pods on same node
  affinity:
    podAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - model-server
        topologyKey: kubernetes.io/hostname
'''
```

### 5.2 Huge Pages for Memory Performance

Huge pages reduce TLB misses and improve memory throughput:

```bash
# Setup (host)
echo 1024 > /proc/sys/vm/nr_hugepages  # 2GB of 2MB pages
# or
echo 512 > /proc/sys/vm/nr_hugepages_1GB  # 512GB of 1GB pages

# Docker: mount huge pages
docker run \
  -v /dev/hugepages:/dev/hugepages \
  -v /sys/kernel/mm/hugepages:/sys/kernel/mm/hugepages \
  --ulimit memlock=-1 \
  inference-server

# Kubernetes: request huge pages
resources:
  requests:
    hugepages-2Mi: 2Gi
  limits:
    hugepages-2Mi: 2Gi
```

---

## 6. Optimization Patterns

### 6.1 Quantization in Depth

```python
import torch
import torch.quantization as Q

class QuantizationStrategy:
    """
    Three quantization approaches and when to use each.
    """

    @staticmethod
    def dynamic_quantization_demo():
        """
        Dynamic quantization: only weights are quantized (INT8).
        Activations are floating-point.
        Best for: CPU inference, ~3-4x speedup, slight accuracy loss (0-2%).
        """
        model = SimpleTransformer()
        quantized = Q.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        return quantized

    @staticmethod
    def int8_static_quantization_demo():
        """
        INT8 static quantization: weights and activations quantized.
        Requires calibration data.
        Best for: CPU + mobile, ~4x speedup, 1-3% accuracy loss.
        """
        model = SimpleTransformer()
        model.qconfig = Q.get_default_qconfig('fbgemm')
        Q.prepare(model, inplace=True)

        # Calibration (needs representative data)
        calibration_data = torch.randn(100, 128, 768)
        with torch.no_grad():
            for batch in calibration_data.split(10):
                model(batch)

        Q.convert(model, inplace=True)
        return model

    @staticmethod
    def fp8_quantization_demo():
        """
        FP8 quantization: 8-bit floating point (scales + mantissa).
        NVIDIA H100+ only.
        Best for: GPU serving, 2x speedup, minimal accuracy loss (<0.5%).
        """
        # PyTorch 2.0+
        import torch._dynamo
        import torch._inductor

        model = SimpleTransformer()

        # Compile with FP8 quantization
        compiled = torch.compile(
            model,
            mode="reduce-overhead",
            options={"triton.cudagraphs": True}
        )

        # Apply FP8 scaling
        model = model.to(torch.float8_e4m3fn)
        return model

    @staticmethod
    def mixed_precision_strategy():
        """
        Mixed precision: different precision for different layers.
        Weights: INT8, critical activations: FP32, non-critical: FP16
        Best for: Balances speed and accuracy.
        """
        class MixedPrecisionTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dense1 = torch.nn.Linear(768, 3072)
                self.dense2 = torch.nn.Linear(3072, 768)

            def forward(self, x):
                # Key computation: FP32
                with torch.cuda.amp.autocast(dtype=torch.float32):
                    x = self.dense1(x)

                # Less critical: FP16
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    x = torch.nn.functional.gelu(x)
                    x = self.dense2(x)

                return x

        return MixedPrecisionTransformer()

# Quantization comparison
QUANTIZATION_COMPARISON = """
┌────────────────┬─────────┬──────────┬──────────┬────────────┐
│ Method         │ Speedup │ Accuracy │ Hardware │ Use Case   │
├────────────────┼─────────┼──────────┼──────────┼────────────┤
│ FP32 (baseline)│ 1.0x    │ 100%     │ All      │ Baseline   │
│ FP16           │ 2.0x    │ 99.5%    │ GPU      │ GPU batch  │
│ Dynamic INT8   │ 3.5x    │ 98.0%    │ CPU      │ CPU mobile │
│ Static INT8    │ 4.0x    │ 97.5%    │ CPU+GPU  │ Optimized  │
│ FP8            │ 2.5x    │ 99.0%    │ H100+    │ New GPU    │
│ INT4 (AWQ)     │ 6.0x    │ 95.0%    │ GPU/CPU  │ LLM edge   │
└────────────────┴─────────┴──────────┴──────────┴────────────┘
"""
```

---

## 7. Deployment Patterns & Case Studies

### 7.1 Staging: From Development to Production

```python
class DeploymentStaging:
    """
    Safe rollout strategy for inference models.
    """

    stages = {
        'Development': {
            'environment': 'localhost',
            'batching': 'dynamic_batch_size=1',
            'monitoring': 'basic logging',
            'users': 'internal'
        },
        'Staging': {
            'environment': 'staging cluster',
            'batching': 'dynamic_batch_size=32',
            'monitoring': 'full metrics (latency, throughput, errors)',
            'users': 'beta users',
            'slo': 'P99 <200ms'
        },
        'Canary': {
            'environment': 'production',
            'batching': 'dynamic_batch_size=64',
            'monitoring': 'production metrics + alerts',
            'users': '1-5% of traffic',
            'slo': 'P99 <150ms'
        },
        'Production': {
            'environment': 'production',
            'batching': 'dynamic_batch_size=256',
            'monitoring': 'real-time dashboards, incident response',
            'users': '100% traffic',
            'slo': 'P99 <150ms, P50 <50ms'
        }
    }

    @staticmethod
    def canary_rollout_config() -> str:
        """Kubernetes Flagger config for canary rollout."""
        return '''
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: inference-server
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-server

  progressDeadlineSeconds: 300

  service:
    port: 8000
    targetPort: 8000

  analysis:
    interval: 1m
    threshold: 5
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
    - name: request-duration
      thresholdRange:
        max: 150  # max 150ms P99

  stages:
  - weight: 5    # 5% traffic
    duration: 1h
  - weight: 25   # 25% traffic
    duration: 1h
  - weight: 50   # 50% traffic
    duration: 1h
  - weight: 100  # 100% traffic
    duration: 1h
'''
```

---

## 8. Troubleshooting & Common Issues

```python
class InferenceDebugging:
    """
    Common inference issues and diagnostics.
    """

    issues = {
        'OOM (Out of Memory)': {
            'cause': 'Batch size too large or KV cache growth',
            'diagnosis': 'Monitor nvidia-smi, check peak memory in logging',
            'solution': 'Reduce batch_size, enable gradient checkpointing'
        },
        'High latency (>2s)': {
            'cause': 'Model load time, slow tokenizer, large batches',
            'diagnosis': 'Profile with torch.profiler, check queue depth',
            'solution': 'Warm up model, reduce batch size, optimize tokenizer'
        },
        'Quantization quality loss': {
            'cause': 'Aggressive quantization with poor calibration',
            'diagnosis': 'Compare outputs on test set, measure accuracy drop',
            'solution': 'Use better calibration data, adjust bits/precision'
        },
        'Numerical instability': {
            'cause': 'Large accumulation errors in long sequences',
            'diagnosis': 'Check for NaN/inf in outputs, compare FP32 baseline',
            'solution': 'Use mixed precision, stable algorithms'
        }
    }

    @staticmethod
    def memory_profiler():
        """Profile memory usage during inference."""
        import tracemalloc

        tracemalloc.start()

        # Run inference
        model = SimpleTransformer()
        x = torch.randn(32, 128, 768)
        _ = model(x)

        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory: {current / 1024 / 1024:.1f}MB")
        print(f"Peak memory: {peak / 1024 / 1024:.1f}MB")

        tracemalloc.stop()
```

---

## 9. Production Deployment Checklist

```python
class ProductionChecklistInference:
    """Production readiness for inference systems."""

    checklist = {
        'Model Export': [
            'Export to ONNX with dynamic shapes',
            'Validate ONNX outputs match PyTorch',
            'Test on target hardware (CPU/GPU)',
            'Measure export + load time'
        ],
        'Optimization': [
            'Quantize (dynamic INT8 minimum)',
            'Profile to identify bottlenecks',
            'Fuse operators where possible',
            'Test accuracy on validation set'
        ],
        'Serving': [
            'Choose serving framework (Triton/vLLM)',
            'Configure batching parameters',
            'Set up model versioning',
            'Test A/B switchover'
        ],
        'Deployment': [
            'Containerize with health checks',
            'Configure resource limits (CPU affinity, memory)',
            'Set up NUMA binding on multi-socket',
            'Test with production traffic patterns'
        ],
        'Monitoring': [
            'Track per-request latency (P50/P95/P99)',
            'Monitor GPU/CPU utilization',
            'Alert on queue depth growth',
            'Track accuracy metrics on live data'
        ]
    }
```

---

## 10. Summary & Key Takeaways

This module covered the complete inference deployment pipeline:

1. **Export**: ONNX for portability, TorchScript for speed, torch.export for advanced compilation
2. **Optimization**: Dynamic quantization (3-4x), static INT8 (4x), mixed precision for balance
3. **ONNX Runtime**: Execution providers vary by hardware; choose CPU/CUDA/TRT/CoreML/QNN
4. **Serving frameworks**: Triton for multi-model, vLLM for LLMs, TorchServe for PyTorch-native
5. **Containerization**: NUMA binding, huge pages, CPU affinity improve performance 20-30%
6. **Batching**: Dynamic batching is critical; tune queue delay and preferred batch sizes
7. **Monitoring**: Track latency P50/P95/P99, GPU utilization, queue depth, accuracy

**Key metrics:**
- Model export size (should be <20% of original after quantization)
- Inference latency (target: <100ms for interactive, <50ms for real-time)
- Throughput (tokens/sec for LLMs, req/sec for others)
- Memory footprint (should fit with 20-30% headroom)

**Next steps**: Start with a baseline PyTorch model, export to ONNX, apply dynamic INT8 quantization, benchmark on target hardware, and deploy with Triton or vLLM. Measure actual latencies and optimize the slowest component first.

# MODULE 23: Mobile Inference — Android & iOS

## Abstract

Mobile phones represent the sweet spot for AI deployment: sufficient compute (2-5 TFLOPS), ample memory (8-12 GB), and global reach (billions of devices). Yet mobile inference introduces unique constraints: thermal throttling, background execution limits, OS-level sandboxing, and deeply heterogeneous hardware (CPU/GPU/NPU coordination). This module examines production mobile inference stacks on Android (NNAPI, TFLite delegates, Hexagon delegation) and iOS (CoreML, Neural Engine integration). We study XNNPACK as the reference CPU backend (ARM NEON assembly, microkernel architecture), ExecuTorch for export-time optimization, and thermal management strategies that prevent throttling during real-time inference. Model architecture patterns (MobileNet, EfficientNet, MobileViT) designed for mobile constraints form the foundation. Both theoretical understanding and production-grade code examples are provided.

---

## 1. Introduction: The Mobile ML Landscape

Mobile devices are fundamentally different from both datacenter servers and microcontroller units:

**Defining characteristics**:
- **Heterogeneous compute**: CPU (big/little cores), GPU, dedicated NPU operating in parallel
- **Thermal constraint**: Peak performance (5W) sustainable only for seconds; thermal throttling reduces to 1-2W
- **Background execution**: Battery drain must remain <50 mA for acceptable user experience
- **OS overhead**: 1-2 GB RAM reserved for system, apps share remaining memory
- **Cost per unit**: Inference must not overly drain battery (milliseconds of compute acceptable, seconds unacceptable)

**Deployment reality** (2024):
- 2+ billion Android devices
- 1+ billion iOS devices
- ~50% have capable hardware for real-time 2D vision (GPU or NPU)
- ~20% have 5+ TFLOPS INT8 capable (flagship)

### 1.1 The Mobile ML Stack (Android)

```
┌──────────────────────────────────────┐
│       Application (Kotlin/Java)      │
│  (Camera app, Assistant, Social app) │
└──────────────────────────────────────┘
              ↓ ML Framework
┌──────────────────────────────────────┐
│   TensorFlow Lite / PyTorch Mobile   │
│   ML Kit (Google proprietary)        │
└──────────────────────────────────────┘
              ↓ NNAPI / Hardware Delegation
┌──────────────────────────────────────┐
│  NNAPI (Android Neural Networks API) │
│  ├─ GPU delegate (Vulkan)           │
│  ├─ NPU delegate (Hexagon, etc.)    │
│  └─ CPU delegate (XNNPACK)          │
└──────────────────────────────────────┘
              ↓ Hardware Drivers
┌──────────────────────────────────────┐
│  GPU Driver (Mali, Adreno)          │
│  NPU Driver (Qualcomm Hexagon)      │
│  CPU (ARM NEON assembly)            │
└──────────────────────────────────────┘
```

### 1.2 Performance Targets

| Use Case | Latency | Throughput | Example |
|----------|---------|-----------|---------|
| Real-time video | 16-33 ms | 30-60 FPS | Face detection |
| User-responsive | 100-500 ms | On-demand | Image classification |
| Background | 1-5 s | Sparse | Wake word detection |

---

## 2. Android ML Stack: NNAPI and Delegation

### 2.1 NNAPI (Android Neural Networks API)

**NNAPI**, introduced in Android 8.1 (2018), provides **hardware-agnostic abstraction** for neural networks.

**Design goals**:
- Single API supports diverse hardware (CPU, GPU, NPU)
- Hardware-specific optimization delegated to vendor drivers
- Battery efficiency through OS-level power management

**NNAPI model representation**:

```c
// Simplified NNAPI model graph (C API)
ANeuralNetworksModel* model;
ANeuralNetworksModel_create(&model);

// Define inputs
ANeuralNetworksModel_addOperand(model, &(ANeuralNetworksOperandType){
    .type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
    .dimensionCount = 4,
    .dimensions = {1, 224, 224, 3},
    .scale = 1.0 / 255.0,
    .zeroPoint = 0
}, &input_idx);

// Define operations
ANeuralNetworksModel_addOperation(
    model,
    ANEURALNETWORKS_CONV_2D,
    3, (uint32_t[]){input_idx, weight_idx, bias_idx},
    1, (uint32_t[]){output_idx}
);

// Mark inputs/outputs
ANeuralNetworksModel_identifyInputsAndOutputs(model, 1, &input_idx, 1, &output_idx);
ANeuralNetworksModel_finish(model);

// Compile with device selection
ANeuralNetworksCompilation* compilation;
ANeuralNetworksCompilation_create(model, &compilation);

// Optionally force accelerator
// ANeuralNetworksCompilation_setPreference(
//     compilation, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);

ANeuralNetworksCompilation_finish(compilation);

// Execute
ANeuralNetworksExecution* execution;
ANeuralNetworksExecution_create(compilation, &execution);

// Set input tensor
ANeuralNetworksExecution_setInput(execution, 0, &(ANeuralNetworksOperandType){...},
                                 input_buffer, 224*224*3);

// Run
ANeuralNetworksExecution_compute(execution);

// Get output
ANeuralNetworksExecution_getOutput(execution, 0, output_buffer, output_size);
```

### 2.2 Delegate Pattern

**NNAPI delegates** enable vendor-specific optimization:

```
Model graph (generic) → Delegate (vendor code) → Hardware-specific execution
```

**Delegate workflow**:

```c
typedef struct {
    // Called when NNAPI needs to execute this operation
    int (*invoke)(struct ANeuralNetworksDevice* device,
                 ANeuralNetworksExecution* execution);
    // Optional: prepare (allocate buffers)
    int (*prepare)(struct ANeuralNetworksDevice* device, ...);
} NNAPIDelegate;
```

**Qualcomm Hexagon delegate** (example):

```c
// Register Hexagon delegate with NNAPI
ANeuralNetworksDevice* hexagon_device;
ANeuralNetworksDevice_getDeviceByName("com.qualcomm.hexagon", &hexagon_device);

ANeuralNetworksCompilation_setDevice(compilation, hexagon_device);

// NNAPI routes all operations to Hexagon driver
// Hexagon driver converts graph to HVX/HTA instructions
// Executes on Hexagon DSP (5-10 TOPS INT8)
```

### 2.3 GPU Delegate (Vulkan Compute)

Modern Android (11+) supports GPU acceleration via **Vulkan compute shaders**:

```c
// TFLite GPU delegate (uses Vulkan under hood)
TfLiteGpuDelegateOptions gpu_options = {
    .wait_type = kPassive,  // Don't busy-wait
    .is_precision_loss_allowed = true,  // Allow FP16
};

TfLiteDelegate* gpu_delegate = TfLiteGpuDelegateCreate(&gpu_options);
interpreter->ModifyGraphWithDelegate(gpu_delegate);
```

**Performance characteristics** (Snapdragon 8 Gen 3):
- Peak compute: 3-4 TFLOPS FP32, 8-16 TFLOPS FP16
- Memory bandwidth: 102 GB/s (LPDDR5X)
- Thermal: Shares power budget with CPU (~5W total)

---

## 3. TensorFlow Lite: Core Framework

### 3.1 Model Format and Flatbuffer

TFLite uses **FlatBuffer** format (covered in Module 22) enabling zero-copy model loading:

```flatbuffer
// Simplified TFLite schema
table Tensor {
  shape: [ubyte];
  type: TensorType;
  buffer: uint;
  name: string;
  quantization: QuantizationParameters;
}

table Operator {
  opcode_index: uint;
  inputs: [uint];
  outputs: [uint];
  builtin_options: table;
}

table SubGraph {
  tensors: [Tensor];
  inputs: [uint];
  outputs: [uint];
  operators: [Operator];
}

table Model {
  subgraphs: [SubGraph];
  buffers: [Buffer];
}
```

**Model file layout**:

```
┌────────────────────────────────────┐
│ Header (4 bytes: magic "TFL\3")   │
├────────────────────────────────────┤
│ Root offset (4 bytes)             │
├────────────────────────────────────┤
│ Model metadata (variable)         │
│ ├─ Subgraph 0                     │
│ │  ├─ Tensors                     │
│ │  ├─ Operators                   │
│ │  └─ I/O indices                 │
│ └─ Buffers (weights)              │
├────────────────────────────────────┤
│ Operator implementations (code)   │
│ (linked at runtime or AOT)        │
└────────────────────────────────────┘
```

### 3.2 TFLite Interpreter

**Interpreter execution model** (similar to Module 22 MCU version, but with heterogeneous delegates):

```cpp
class Interpreter {
public:
    Interpreter(const Model* model) : model_(model) {}

    // Load model from flatbuffer
    Status AllocateTensors() {
        // Parse subgraph
        subgraph_ = model_->subgraphs(0);

        // Allocate tensors (allocate in order)
        for (int i = 0; i < subgraph_->tensors()->size(); i++) {
            auto tensor_desc = subgraph_->tensors(i);
            auto* tensor = &tensors_[i];
            tensor->bytes = CalculateTensorBytes(tensor_desc);
            tensor->data = arena_.allocate(tensor->bytes);
        }
        return kOk;
    }

    Status Invoke() {
        // Find which delegate can execute each op
        for (int op_idx = 0; op_idx < subgraph_->operators()->size(); op_idx++) {
            auto op = subgraph_->operators(op_idx);

            // Check GPU delegate capability
            if (gpu_delegate_.CanExecute(op)) {
                gpu_delegate_.Execute(op, tensors_);
            }
            // Check NPU delegate capability
            else if (npu_delegate_.CanExecute(op)) {
                npu_delegate_.Execute(op, tensors_);
            }
            // Fall back to CPU
            else {
                cpu_kernels_.Execute(op, tensors_);
            }
        }
        return kOk;
    }

private:
    std::vector<Tensor> tensors_;
    MemoryArena arena_;
    GpuDelegate gpu_delegate_;
    NpuDelegate npu_delegate_;
    CpuKernels cpu_kernels_;
};
```

### 3.3 Quantization for TFLite Mobile

**Mobile quantization strategy**:

```python
import tensorflow as tf

# Quantization-aware training
model = build_mobilenet_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train with simulated quantization
quantize_model = tf.quantization.quantize_model(model)
quantize_model.fit(train_data, train_labels, epochs=10)

# Post-training quantization (if QAT not done)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

# Representative dataset for calibration
def representative_dataset():
    for image in test_data[:100]:
        # Image already preprocessed to model input shape
        yield [image[np.newaxis, ...].astype(np.float32)]

converter.representative_dataset = representative_dataset

# Convert to quantized TFLite
tflite_quant_model = converter.convert()

with open('model_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)
```

---

## 4. XNNPACK: The CPU Inference Engine

### 4.1 Architecture Overview

**XNNPACK** is the reference CPU backend for TFLite (and also PyTorch Mobile). Designed specifically for **mobile ARM processors** with hand-optimized NEON kernels.

**Philosophy**:
- **Microkernel abstraction**: Small, reusable tile-processing functions
- **Vectorization**: ARM NEON assembly for maximum throughput
- **Memory hierarchy awareness**: Tile sizes chosen to fit L1 cache

### 4.2 Microkernel Architecture

**Concept**: Every operation decomposed into small, repeatable microkernels.

**Convolution example** (simplified):

```
Convolution (MxNx(K×K) filter) decomposed as:
├─ Im2Col: Expand input image (5D tensor) into columns (2D matrix)
├─ GEMM: Matrix multiply (Weights × Columns = Output features)
└─ Bias/ReLU: Activation function
```

**Microkernel for packed integer matrix multiply** (XNNPACK's HGEMM):

```c
// Signature: 1×4 INT8 × 4×8 INT8 weights → 1×8 INT32 output
void xnn_s8_gemm_minmax_1x8c4__neon(
    size_t mr,  // m rows (usually 1 for mobile inference)
    size_t nr,  // n columns (usually 8, fitting NEON width)
    size_t kc,  // k columns (inner dimension, tiled)
    const int8_t* a,
    size_t a_stride,
    const void* w,  // Weights (already packed in k_stride × nr layout)
    int8_t* c,
    size_t cm_stride,
    size_t cn_stride,
    const xnn_int8_minmax_params* params
) {
    // Initialize accumulator to 0 (or bias)
    int32x4_t acc0 = vmovq_n_s32(0);
    int32x4_t acc1 = vmovq_n_s32(0);

    const int8_t* a_ptr = a;
    const int8_t* w_ptr = (const int8_t*)w;

    // Process in tiles of 4 weights at a time (exploits vector width)
    size_t k = kc;
    while (k >= 4) {
        // Load 1 input element (broadcast to 4 lanes)
        int8x16_t va = vmovq_n_s8(a_ptr[0]);
        a_ptr += 4 * a_stride;  // Load 4 activations (GEMM packing)

        // Load 4×8 weight block (packed layout)
        int8x16_t vw0 = vld1q_s8(w_ptr);      // First 4 weights for 8 outputs
        int8x16_t vw1 = vld1q_s8(w_ptr + 16);  // Next 4 weights
        w_ptr += 32;

        // Multiply-accumulate: va (1×4) × vw0 (4×8) → acc (1×8)
        // Expand int8 to int16, multiply, add to int32 accumulator
        int16x8_t prod0 = vmull_s8(vget_low_s8(va), vget_low_s8(vw0));
        int16x8_t prod1 = vmull_s8(vget_high_s8(va), vget_high_s8(vw0));
        acc0 = vpadalq_s16(acc0, prod0);
        acc1 = vpadalq_s16(acc1, prod1);

        k -= 4;
    }

    // Post-processing: quantization (INT32 → INT8)
    int32x4_t scaled0 = vqdmulhq_s32(acc0, params->multiplier);
    int32x4_t scaled1 = vqdmulhq_s32(acc1, params->multiplier);

    // Clamping
    int16x8_t clamped = vqaddq_s16(
        vcombine_s16(vqmovn_s32(scaled0), vqmovn_s32(scaled1)),
        vmovq_n_s16(params->zero_point)
    );
    int8x16_t output = vqmovn_s16(clamped);

    // Store result
    vst1_s8(c, vget_low_s8(output));
    vst1_s8(c + 8, vget_high_s8(output));
}
```

### 4.3 Operator Fusion

XNNPACK fuses multiple operators into single kernel to **reduce memory traffic**:

**Conv2D + ReLU fusion**:

```c
// Without fusion:
// conv2d output → SRAM → relu input → relu output
// Memory traffic: 2× activation tensor size

// With fusion:
// conv2d output stays in registers → relu operates in-place
// Memory traffic: 1× output only

void xnn_conv2d_relu_s8_to_u8(
    size_t batch_size,
    size_t output_height,
    size_t output_width,
    const int8_t* input_data,
    const int8_t* kernel_data,
    const int32_t* bias_data,
    uint8_t* output_data,
    size_t output_stride,
    const xnn_quantization_params* params
) {
    for (int b = 0; b < batch_size; b++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                // Conv2D
                int32_t sum = bias_data[oc];
                for (int kh = 0; kh < kernel_height; kh++) {
                    for (int kw = 0; kw < kernel_width; kw++) {
                        sum += (int32_t)input_data[...] * kernel_data[...];
                    }
                }

                // Quantize (INT32 → INT8/UINT8)
                int32_t scaled = (sum >> params->output_shift) + params->zp;

                // ReLU (clamp to [0, 255] for UINT8)
                output_data[oh * output_width + ow] =
                    max(0, min(255, scaled));  // Fused ReLU
            }
        }
    }
}
```

### 4.4 XNNPACK Performance

**Benchmarks** (Cortex-A78, 3 GHz, NEON):

| Op | Model Size | Latency (ms) | Throughput (MMACS) |
|----|-----------|-------------|--------------------|
| Conv 3×3, 32 channels | 4.6 KB | 0.5 | 900 |
| Conv 3×3, 256 channels | 73 KB | 6.2 | 1200 |
| FC 1024→1024 INT8 | 1 MB | 2.8 | 700 |

---

## 5. ExecuTorch: Export-Time Optimization

### 5.1 Motivation

**ExecuTorch** (Facebook/Meta) provides **export-time compilation** for mobile inference:

```
PyTorch model (training) → ExecuTorch export → Optimized mobile binary
                          (quantization, fusion, memory planning)
```

**Key difference from TFLite**: Compiler runs once at export time, not runtime.

### 5.2 ExecuTorch Pipeline

```python
import torch
import executorch as et
from executorch.exir import to_edge

# Load trained model
model = torchvision.models.mobilenet_v2(pretrained=True)

# Convert to ExecuTorch format
example_input = torch.randn(1, 3, 224, 224)

# Step 1: Trace to EXIR (Executable IR)
traced = torch.fx.symbolic_trace(model)
edge_program = to_edge(traced)

# Step 2: Quantization (optional)
from torch.quantization import quantize_dynamic
quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Step 3: Memory planning and optimization
optimized = edge_program.to_executorch(
    config=et.ExecutorchConfig(
        memory_planning_pass=True,
        constant_propagation=True,
        decompose_complex_ops=True,
    )
)

# Step 4: Export to binary
optimized.save_to_path("model_optimized.pte")
```

### 5.3 Mobile Runtime

**ExecuTorch runtime** (C++) is extremely lightweight:

```cpp
#include "executorch/runtime/core/evalue.h"
#include "executorch/runtime/executor/method.h"

// Load model binary
FILE* fp = fopen("model_optimized.pte", "rb");
char* model_bytes = malloc(file_size);
fread(model_bytes, 1, file_size, fp);

// Create program
const Result<Program> program = Program::load(
    model_bytes, file_size);

// Create executor
const Result<Method> method = program.get().load_method(
    DEFAULT_CHAIN, new MemoryAllocator());

// Set input
EValue input{torch::empty({1, 3, 224, 224})};
method.get().set_input(0, input);

// Execute
ET_LOG_AND_RETURN_IF_ERROR(method.get().execute());

// Get output
const EValue& output = method.get().get_output(0);
at::Tensor result = output.to_tensor();
```

---

## 6. iOS CoreML Integration

### 6.1 CoreML Framework

Apple's **CoreML** is the native iOS ML framework, tightly integrated with Neural Engine.

**Architecture**:

```
┌──────────────────────────────┐
│   Swift / Objective-C App    │
│  (Vision, Sound, Text APIs)  │
└──────────────────────────────┘
           ↓ CoreML
┌──────────────────────────────┐
│   CoreML Models (.mlmodel)   │
│   ├─ Neural Network          │
│   ├─ Tree Ensemble           │
│   └─ Flexible Input/Output   │
└──────────────────────────────┘
           ↓ Neural Engine Drivers
┌──────────────────────────────┐
│   Neural Engine (Apple)      │
│   GPU (Metal) / CPU (Accelerate)
└──────────────────────────────┘
```

### 6.2 Converting Models to CoreML

```python
import coremltools as ct
import torch

# Convert PyTorch to CoreML
model = torch.hub.load('pytorch/vision:v0.9', 'mobilenet_v2')
model.eval()

# Create dummy input
example_input = torch.randn(1, 3, 224, 224)

# Trace with torch.jit
traced_model = torch.jit.trace(model, example_input)

# Convert to CoreML
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name='input', shape=(1, 3, 224, 224))],
    outputs=[ct.ImageType(name='output')],
    minimum_deployment_target=ct.target.iOS15,
    compute_units=ct.ComputeUnit.CPU_AND_NE,  # Use Neural Engine if available
)

# Save
coreml_model.save('MobileNetV2.mlmodel')
```

### 6.3 Inference on iOS

```swift
import CoreML
import Vision

// Load model
let config = MLModelConfiguration()
config.computeUnits = .cpuAndNeuralEngine

guard let modelURL = Bundle.main.url(forResource: "MobileNetV2",
                                    withExtension: "mlmodel") else {
    fatalError("Model not found")
}

let mlModel = try! MLModel(contentsOf: modelURL, configuration: config)
let model = try! MobileNetV2(model: mlModel)

// Prepare input image
let inputImage = UIImage(named: "input.jpg")!
let pixelBuffer = inputImage.pixelBuffer()

// Run inference
let input = MobileNetV2Input(inputWith: pixelBuffer)
let output = try! model.prediction(input: input)

// Get top-1 prediction
if let classLabel = output.classLabel {
    print("Predicted class: \(classLabel)")
}
```

### 6.4 Neural Engine Architecture

**Apple Neural Engine** (A14+):

- 16-core matrix multiplication unit
- Specialized for quantized (INT8) and low-precision (FP16) inference
- Peak throughput: 11 TOPS INT8, 5.5 TOPS FP16
- Power efficiency: ~1W for 2-3 second video frame processing

---

## 7. Thermal Management

### 7.1 Thermal Throttling Mechanism

**Mobile thermal control** (Snapdragon 8 Gen 3 example):

```
Device temperature monitoring:
- Junction temperature Tj measured by thermal sensors
- Baseline throttling threshold: 80°C
- Critical shutdown: 90°C

Throttling policy:
├─ 75-80°C: Reduce GPU frequency 10-20%
├─ 80-85°C: Reduce all frequencies 30-50%
├─ 85-90°C: Power off non-essential cores
└─ >90°C: Emergency shutdown
```

**Implication for inference**:
- Peak performance (8 TOPS) available for ~5-10 seconds
- After thermal throttling kicks in, sustainable performance drops to 2-3 TOPS
- Continuous inference must be designed for throttled performance

### 7.2 Thermal-Aware Scheduling

**Strategy**: Detect thermal state and adjust inference strategy:

```python
def run_inference_thermally_aware(model, input_batch, max_batches=10):
    """
    Adaptive batching based on device temperature.
    """
    import time

    results = []
    total_latency = 0.0

    for batch_idx in range(max_batches):
        # Check device thermal state (via TFLite API)
        # Note: No direct API in TFLite; would be OS-specific

        # Approach 1: Measure latency, adapt batch size
        start_time = time.time()
        output = model.invoke(input_batch[batch_idx:batch_idx+1])
        latency = time.time() - start_time

        results.append(output)
        total_latency += latency

        # Throttling detection: latency increase > 2x
        if batch_idx > 1 and latency > 2 * (total_latency / batch_idx):
            print(f"Thermal throttling detected at batch {batch_idx}")
            print(f"Latency increased from {(total_latency / batch_idx):.2f}ms to {latency:.2f}ms")

            # Strategy: Sleep to cool device, then resume
            time.sleep(0.5)

    return results, total_latency
```

### 7.3 Power Consumption Profiling

**Example**: Video processing on phone (30 FPS):

```
Frame rate: 30 FPS → 33.3 ms per frame
Model latency: 25 ms (peak) / 50 ms (throttled)

Scenario 1: Peak performance
├─ Power draw: 3-4W for inference + display (~6W total)
├─ Battery drain: 6W × (33.3 ms / 1000 ms) = 200 mJ/frame
├─ Sustainable duration: 3500 mAh × 3.7V / 6W = 2 hours (approximate)
└─ User experience: Unacceptable (drains battery very quickly)

Scenario 2: Throttled performance (50 ms latency)
├─ Model runs for 50 ms, results in time for 33 ms frames (overlap okay with buffering)
├─ Power: 1-2W for inference + display (~4W total)
├─ Sustainable duration: 3500 mAh × 3.7V / 4W = 3+ hours
└─ User experience: Acceptable (video plays smoothly, battery drain modest)
```

---

## 8. Model Architecture Patterns for Mobile

### 8.1 MobileNet Family

**MobileNetV1** (Howard et al., 2017): Depthwise-separable convolution.

```
Standard convolution:
  Input: H×W×C_in → Conv 3×3 → Output: H×W×C_out
  FLOPs: H × W × 3 × 3 × C_in × C_out

Depthwise-separable:
  Input: H×W×C_in
    ├─ Depthwise conv (3×3 per channel): H × W × 3 × 3 × C_in FLOPs
    └─ Pointwise conv (1×1): H × W × C_in × C_out FLOPs
  Total: H × W × (9 × C_in + C_in × C_out) << standard
```

**MobileNetV1 structure**:

```
Input (224×224×3)
  ↓
Conv 3×3 32 (stride 2)     [13×13×32]   ~37 MB FLOPs
  ↓
DepthwiseConv / 1×1Conv    [13×13×64]   ~20 MB FLOPs  (5×)
  ↓ (repeat for 13 blocks, reducing spatial dims)
...
GlobalAvgPool              [1×1×1024]
  ↓
FC 1000 classes           [1×1×1000]    ~1 MB FLOPs

Total: ~550 MB FLOPs (vs. 7 B for standard ResNet)
```

### 8.2 EfficientNet

**EfficientNet** (Tan & Le, 2019): Balanced scaling of width, depth, resolution.

```
EfficientNet-B0 (baseline):
  Resolution: 224×224
  Depth: 1.0× (all layers × 1.0)
  Width: 1.0× (all channels × 1.0)
  FLOPs: ~390 MB
  ImageNet Top-1: 77.1%

EfficientNet-B1:
  Resolution: 240×240 (1.1×)
  Depth: 1.1× (add 10% more layers)
  Width: 1.05× (increase channels 5%)
  FLOPs: ~700 MB
  ImageNet Top-1: 79.8%

...

EfficientNet-B7:
  Resolution: 600×600 (2.7×)
  Depth: 2.0×
  Width: 2.0×
  FLOPs: ~37 B (large, not mobile)
  ImageNet Top-1: 84.4%
```

### 8.3 MobileViT: Vision Transformer for Mobile

**MobileViT** (Mehta & Rastegari, 2021): Hybrid CNN + Vision Transformer.

**Design**: Replace global operations with local vision transformer blocks.

```
MobileViT-S (small):
  Input: 256×256×3

  Stage 1 (CNN): Conv 3×3 stride 2 → 128×128×64
  Stage 2 (CNN-ViT):
    ├─ Conv 3×3 stride 2 → 64×64×96
    ├─ ViT block (local 2×2 patches): models local spatial relationships
    └─ Conv 1×1 → 64×64×192
  ...

  Output: 8×8×320
  Global avg pool → FC 1000

  FLOPs: ~1.2 B
  ImageNet Top-1: 78.5% (comparable to MobileNetV2 with fewer parameters)
```

---

## 9. Complete Mobile Inference Example

### 9.1 Android Implementation

```java
import android.content.Context;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import java.nio.ByteBuffer;

public class MobileInference {
    private Interpreter tfliteInterpreter;
    private GpuDelegate gpuDelegate;
    private final int IMAGE_SIZE = 224;

    public MobileInference(Context context, String modelPath) throws Exception {
        // Load model options
        Interpreter.Options options = new Interpreter.Options();

        // Add GPU delegate if available
        CompatibilityList compatList = new CompatibilityList();
        if (compatList.isDelegateSupportedOnThisDevice()) {
            gpuDelegate = new GpuDelegate(compatList.getBestOptionsForThisDevice());
            options.addDelegate(gpuDelegate);
        } else {
            // Fall back to CPU with XNNPACK (default)
        }

        // Load model
        java.nio.MappedByteBuffer modelBuffer = loadModelFile(context, modelPath);
        tfliteInterpreter = new Interpreter(modelBuffer, options);

        // Print model info
        int[] inputShape = tfliteInterpreter.getInputTensor(0).shape();
        DataType inputType = tfliteInterpreter.getInputTensor(0).dataType();
        System.out.println("Input shape: " + java.util.Arrays.toString(inputShape));
        System.out.println("Input type: " + inputType);
    }

    public float[] runInference(android.graphics.Bitmap bitmap) {
        // Prepare input: Bitmap → float tensor [1, 224, 224, 3]
        TensorImage image = new TensorImage(DataType.FLOAT32);
        image.load(bitmap);
        // (Would normally include preprocessing: normalization, etc.)

        // Run inference
        long startTime = System.nanoTime();
        float[][] output = new float[1][1000];
        tfliteInterpreter.run(image.getBuffer(), output);
        long inferenceTime = (System.nanoTime() - startTime) / 1e6f;  // ms

        System.out.println("Inference time: " + inferenceTime + " ms");

        // Return output
        return output[0];
    }

    private java.nio.MappedByteBuffer loadModelFile(Context context, String filename)
            throws Exception {
        java.nio.channels.FileChannel fileChannel =
                new java.io.RandomAccessFile(filename, "r").getChannel();
        long startOffset = 0;
        long declaredLength = fileChannel.size();
        return fileChannel.map(
                java.nio.channels.FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public void cleanup() {
        tfliteInterpreter.close();
        if (gpuDelegate != null) {
            gpuDelegate.close();
        }
    }
}
```

### 9.2 iOS Implementation (Swift)

```swift
import CoreML
import Vision
import UIKit

class MobileInferenceVC: UIViewController {
    var model: VNCoreMLModel?
    var request: VNCoreMLRequest?

    override func viewDidLoad() {
        super.viewDidLoad()

        // Load CoreML model
        guard let mlModelURL = Bundle.main.url(forResource: "MobileNetV2",
                                               withExtension: "mlmodel") else {
            print("Model not found")
            return
        }

        do {
            let modelConfig = MLModelConfiguration()
            modelConfig.computeUnits = .cpuAndNeuralEngine

            let mlModel = try MLModel(contentsOf: mlModelURL,
                                     configuration: modelConfig)
            model = try VNCoreMLModel(for: mlModel)

            // Create Vision request
            request = VNCoreMLRequest(model: model!) { request, error in
                self.processResults(request, error)
            }
            request?.imageCropAndScaleOption = .centerCrop
        } catch {
            print("Failed to load model: \(error)")
        }
    }

    func runInference(image: UIImage) {
        guard let request = request else { return }

        let handler = VNImageRequestHandler(cgImage: image.cgImage!,
                                           options: [:])

        let startTime = Date()
        do {
            try handler.perform([request])
            let inferenceTime = Date().timeIntervalSince(startTime) * 1000
            print("Inference time: \(inferenceTime) ms")
        } catch {
            print("Failed to run inference: \(error)")
        }
    }

    func processResults(_ request: VNRequest, _ error: Error?) {
        guard let results = request.results as? [VNClassificationObservation] else {
            print("No results")
            return
        }

        for (index, classification) in results.prefix(5).enumerated() {
            print("\(index + 1). \(classification.identifier): \(classification.confidence)")
        }
    }
}
```

---

## 10. Conclusion and Best Practices

### 10.1 Performance Optimization Checklist

- [ ] **Profile first**: Measure latency, memory, power on target device
- [ ] **Quantize aggressively**: INT8 baseline, INT4 for extreme cases
- [ ] **Choose right architecture**: MobileNet/EfficientNet/MobileViT as starting point
- [ ] **Use delegates**: GPU/NPU if available and beneficial (measure!)
- [ ] **Fuse operations**: Conv+ReLU, etc., reduce memory bandwidth
- [ ] **Thermal aware**: Expect 30-50% performance reduction after 5-10 seconds
- [ ] **Battery-first design**: 50 mA average acceptable; 100+ mA unacceptable
- [ ] **Test on real hardware**: Simulator ≠ actual device (thermal, memory pressure different)

### 10.2 Architecture Selection

| Model | FLOPs | Memory | Accuracy | Latency (Flagship) | Best For |
|-------|-------|--------|----------|-------------------|----------|
| MobileNetV2 | 300 MB | 9 MB | 74.7% | 25 ms | Baseline, proven |
| EfficientNet-Lite0 | 400 MB | 12 MB | 75.1% | 35 ms | Better accuracy |
| MobileViT-S | 1.2 B | 18 MB | 78.5% | 80 ms | Cutting edge |

### 10.3 Key Insights

1. **Thermal is the real bottleneck**, not compute (peak latency short-lived)
2. **GPU/NPU helpful** for dense ops (large conv) but overhead for small ops
3. **INT8 mandatory** for sub-100ms latency (vs. FP32)
4. **Memory bandwidth critical**: Model architecture should minimize activation reuse distance
5. **OS integration matters**: Background execution limits, battery optimization APIs

---

## Further Reading

- XNNPACK: https://github.com/google/XNNPACK
- TensorFlow Lite Delegates: https://www.tensorflow.org/lite/guide/delegates
- ARM NNAPI Architecture: https://source.android.com/devices/neural-networks
- ExecuTorch: https://github.com/pytorch/executorch
- Apple CoreML: https://developer.apple.com/coreml/
- Howard et al.: "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (2017)
- Tan & Le: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (ICML 2019)

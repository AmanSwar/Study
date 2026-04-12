'use client'

import { DiagramCanvas } from '../core/DiagramCanvas'
import { DiagramThemeProvider, useDiagramTheme } from '../core/DiagramTheme'
import { DiagramTooltipPortal, useDiagramTooltip } from '../core/DiagramTooltip'
import { AnimationController, AnimationStep } from '../core/AnimationController'
import { usePipelineAnimation } from '../animations/usePipelineAnimation'
import { Block } from '../primitives/Block'
import { DataFlowArrow } from '../primitives/DataFlowArrow'

const steps: AnimationStep[] = [
  {
    id: 'api',
    title: 'User API Layer',
    description: 'Request scheduling, batching, and SLA management (vLLM, TGI)',
    duration: 3000,
    highlightElements: ['api-layer'],
    animateArrows: [],
  },
  {
    id: 'graph',
    title: 'Model Graph Layer',
    description: 'Computational graph optimization, kernel fusion, quantization (TransformerEngine, DeepSpeed)',
    duration: 3000,
    highlightElements: ['graph-layer'],
    animateArrows: ['arrow-api-graph'],
  },
  {
    id: 'kernel',
    title: 'Kernel Layer',
    description: 'CUDA/HIP kernels for specific ops — roofline-aware kernel selection (CUTLASS, TVM, Triton)',
    duration: 3000,
    highlightElements: ['kernel-layer'],
    animateArrows: ['arrow-graph-kernel'],
  },
  {
    id: 'runtime',
    title: 'Runtime Layer',
    description: 'Memory management, scheduling, device orchestration (CUDA, HIP, oneAPI)',
    duration: 3000,
    highlightElements: ['runtime-layer'],
    animateArrows: ['arrow-kernel-runtime'],
  },
  {
    id: 'hardware',
    title: 'Hardware',
    description: 'GPU, CPU, NPU silicon — where the actual compute happens',
    duration: 3000,
    highlightElements: ['hardware-layer'],
    animateArrows: ['arrow-runtime-hw'],
  },
]

function InferenceStackInner() {
  const colors = useDiagramTheme()
  const { tooltip, showTooltip, moveTooltip, hideTooltip } = useDiagramTooltip()
  const { currentStep, isPlaying, activeElements, activeArrows, togglePlayPause, goToStep } =
    usePipelineAnimation(steps)

  const W = 700
  const H = 520
  const layerW = 500
  const layerH = 65
  const gap = 20
  const startX = (W - layerW) / 2
  const startY = 30

  const layers = [
    { id: 'api-layer', label: 'User Inference API', sublabel: 'vLLM, TGI, TensorRT-LLM', variant: 'primary' as const, overhead: '~5% overhead', tooltip: 'Request scheduling, batching, continuous batching, SLA enforcement' },
    { id: 'graph-layer', label: 'Model Graph Layer', sublabel: 'TransformerEngine, DeepSpeed', variant: 'secondary' as const, overhead: '5-15% overhead', tooltip: 'Graph optimization, kernel fusion, quantization, layer reordering' },
    { id: 'kernel-layer', label: 'Kernel Layer', sublabel: 'CUTLASS, TVM, Triton', variant: 'accent' as const, overhead: '10-20% gap from roofline', tooltip: 'Optimized CUDA/HIP kernels, roofline-aware kernel selection, tiling strategies' },
    { id: 'runtime-layer', label: 'Runtime Layer', sublabel: 'CUDA, HIP, oneAPI, Metal', variant: 'muted' as const, overhead: '5-10% overhead', tooltip: 'Memory allocation, stream scheduling, device synchronization' },
    { id: 'hardware-layer', label: 'Hardware', sublabel: 'GPU · CPU · NPU', variant: 'compute' as const, overhead: 'Silicon', tooltip: 'GPU (H100), CPU (Xeon/EPYC), NPU (Hexagon/ANE), Apple Silicon' },
  ]

  const arrowIds = ['arrow-api-graph', 'arrow-graph-kernel', 'arrow-kernel-runtime', 'arrow-runtime-hw']

  return (
    <>
      <div className="my-8 rounded-xl border border-border-primary overflow-hidden">
        <div className="px-4 py-2 border-b border-border-primary bg-bg-surface/50">
          <span className="text-xs font-semibold text-text-tertiary uppercase tracking-wider">
            The Inference Stack — Layers of Abstraction
          </span>
        </div>

        <DiagramCanvas width={W} height={H} zoomable={false} className="!my-0 !rounded-none !border-0">
          <defs>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Layers */}
          {layers.map((layer, i) => {
            const y = startY + i * (layerH + gap)
            return (
              <g key={layer.id}>
                <Block
                  id={layer.id}
                  x={startX}
                  y={y}
                  width={layerW}
                  height={layerH}
                  label={layer.label}
                  sublabel={layer.sublabel}
                  variant={layer.variant}
                  highlighted={activeElements.has(layer.id)}
                  dimmed={activeElements.size > 0 && !activeElements.has(layer.id)}
                  onMouseEnter={(e) => showTooltip(e, (
                    <div>
                      <div className="font-semibold mb-1">{layer.label}</div>
                      <div>{layer.tooltip}</div>
                      <div className="mt-1 text-accent-orange">{layer.overhead}</div>
                    </div>
                  ))}
                  onMouseMove={moveTooltip}
                  onMouseLeave={hideTooltip}
                />

                {/* Overhead annotation */}
                <text
                  x={startX + layerW + 15}
                  y={y + layerH / 2}
                  dominantBaseline="central"
                  fill={colors.textLight}
                  fontSize={9}
                  fontFamily="JetBrains Mono, monospace"
                  opacity={activeElements.has(layer.id) ? 1 : 0.5}
                >
                  {layer.overhead}
                </text>
              </g>
            )
          })}

          {/* Arrows between layers */}
          {arrowIds.map((arrowId, i) => {
            const fromY = startY + i * (layerH + gap) + layerH
            const toY = startY + (i + 1) * (layerH + gap)
            return (
              <DataFlowArrow
                key={arrowId}
                id={arrowId}
                from={{ x: W / 2, y: fromY + 2 }}
                to={{ x: W / 2, y: toY - 2 }}
                active={activeArrows.has(arrowId)}
                dimmed={activeArrows.size > 0 && !activeArrows.has(arrowId)}
                packetCount={2}
                speed={0.8}
              />
            )
          })}

          {/* Efficiency annotation */}
          <text
            x={W / 2}
            y={H - 10}
            textAnchor="middle"
            fill={colors.textLight}
            fontSize={10}
            fontFamily="Inter, system-ui, sans-serif"
          >
            Real-world H100 efficiency: ~35-40% of roofline in production decode
          </text>
        </DiagramCanvas>

        <AnimationController
          steps={steps}
          currentStep={currentStep}
          isPlaying={isPlaying}
          onStepChange={goToStep}
          onPlayPause={togglePlayPause}
        />
      </div>
      <DiagramTooltipPortal tooltip={tooltip} />
    </>
  )
}

export function InferenceStackDiagram() {
  return (
    <DiagramThemeProvider>
      <InferenceStackInner />
    </DiagramThemeProvider>
  )
}

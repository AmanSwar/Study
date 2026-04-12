'use client'

import { DiagramCanvas } from '../core/DiagramCanvas'
import { DiagramThemeProvider, useDiagramTheme } from '../core/DiagramTheme'
import { DiagramTooltipPortal, useDiagramTooltip } from '../core/DiagramTooltip'
import { AnimationController, AnimationStep } from '../core/AnimationController'
import { usePipelineAnimation } from '../animations/usePipelineAnimation'
import { Block } from '../primitives/Block'
import { Arrow } from '../primitives/Arrow'
import { motion } from 'framer-motion'

const steps: AnimationStep[] = [
  { id: 'input', title: 'Input Embeddings', description: 'Token embeddings + positional encoding enter the block', duration: 2500, highlightElements: ['input'], animateArrows: [] },
  { id: 'ln1', title: 'Layer Norm 1', description: 'Pre-norm: normalize activations before attention', duration: 2000, highlightElements: ['layernorm1'], animateArrows: ['arrow-input-ln1'] },
  { id: 'qkv', title: 'QKV Projections', description: 'Three parallel GEMMs: Q, K, V projections. Compute-bound in prefill, memory-bound in decode.', duration: 3000, highlightElements: ['q-proj', 'k-proj', 'v-proj'], animateArrows: ['arrow-ln1-qkv'] },
  { id: 'attn', title: 'Scaled Dot-Product Attention', description: 'Q·K^T / √d → softmax → ×V. O(n²) in prefill, O(n) in decode. FlashAttention optimizes memory access.', duration: 3500, highlightElements: ['attention'], animateArrows: ['arrow-qkv-attn'] },
  { id: 'out-proj', title: 'Output Projection', description: 'Linear projection of concatenated attention heads back to d_model dimensions', duration: 2000, highlightElements: ['out-proj'], animateArrows: ['arrow-attn-out'] },
  { id: 'residual1', title: 'Residual Connection 1', description: 'Add input directly to attention output — enables gradient flow in deep networks', duration: 2500, highlightElements: ['residual1'], animateArrows: ['arrow-residual1'] },
  { id: 'ln2', title: 'Layer Norm 2', description: 'Pre-norm before FFN', duration: 2000, highlightElements: ['layernorm2'], animateArrows: ['arrow-res1-ln2'] },
  { id: 'ffn', title: 'Feed-Forward Network', description: 'Two GEMMs: up-projection (d→4d) + SiLU + down-projection (4d→d). Dominates decode latency due to parameter loading.', duration: 3500, highlightElements: ['ffn-up', 'ffn-act', 'ffn-down'], animateArrows: ['arrow-ln2-ffn'] },
  { id: 'residual2', title: 'Residual Connection 2 → Output', description: 'Final residual add produces the block output. Repeat for N layers (80 for Llama-2 70B).', duration: 2500, highlightElements: ['residual2', 'output'], animateArrows: ['arrow-residual2'] },
]

function TransformerBlockInner() {
  const colors = useDiagramTheme()
  const { tooltip, showTooltip, moveTooltip, hideTooltip } = useDiagramTooltip()
  const { currentStep, isPlaying, activeElements, activeArrows, togglePlayPause, goToStep } =
    usePipelineAnimation(steps)

  const W = 700
  const H = 700
  const centerX = W / 2
  const blockW = 160
  const blockH = 42
  const smallBlockW = 100
  const smallBlockH = 36

  // Vertical positions
  let y = 25
  const spacing = 62

  const isActive = (id: string) => activeElements.has(id)
  const isDimmed = (id: string) => activeElements.size > 0 && !isActive(id)

  return (
    <>
      <div className="my-8 rounded-xl border border-border-primary overflow-hidden">
        <div className="px-4 py-2 border-b border-border-primary bg-bg-surface/50">
          <span className="text-xs font-semibold text-text-tertiary uppercase tracking-wider">
            Transformer Block — Forward Pass Walkthrough
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

          {/* Input */}
          <Block id="input" x={centerX - blockW / 2} y={y} width={blockW} height={blockH}
            label="Input" sublabel="(B, L, d_model)" variant="muted"
            highlighted={isActive('input')} dimmed={isDimmed('input')}
            onMouseEnter={(e) => showTooltip(e, <div>Token embeddings + positional encoding<br/>Shape: (batch, seq_len, d_model)</div>)}
            onMouseMove={moveTooltip} onMouseLeave={hideTooltip} />

          {/* Arrow: Input → LN1 */}
          <Arrow id="arrow-input-ln1" from={{ x: centerX, y: y + blockH }} to={{ x: centerX, y: y + spacing }}
            animated={activeArrows.has('arrow-input-ln1')} dimmed={isDimmed('layernorm1')} />

          {/* Residual path 1 (left side) */}
          {(() => {
            const resStartY = y + blockH / 2
            const resEndY = y + spacing * 5 + blockH / 2
            const resX = centerX - blockW / 2 - 40
            return (
              <g id="residual1" style={{ opacity: isDimmed('residual1') ? 0.2 : 1 }}>
                <motion.path
                  d={`M${centerX - blockW / 2},${resStartY} L${resX},${resStartY} L${resX},${resEndY} L${centerX - blockW / 2},${resEndY}`}
                  fill="none" stroke={isActive('residual1') ? colors.success : colors.textLight}
                  strokeWidth={isActive('residual1') ? 2 : 1} strokeDasharray="6 3" strokeOpacity={0.6}
                />
                <text x={resX - 5} y={(resStartY + resEndY) / 2} textAnchor="end" dominantBaseline="central"
                  fill={colors.success} fontSize={9} fontWeight={500} fontFamily="Inter, system-ui, sans-serif"
                  transform={`rotate(-90, ${resX - 5}, ${(resStartY + resEndY) / 2})`}>
                  Residual
                </text>
                {isActive('residual1') && (
                  <motion.circle r={4} fill={colors.success} filter="url(#glow)"
                    animate={{
                      cx: [centerX - blockW / 2, resX, resX, centerX - blockW / 2],
                      cy: [resStartY, resStartY, resEndY, resEndY],
                    }}
                    transition={{ duration: 2, repeat: Infinity, ease: 'linear' }} />
                )}
              </g>
            )
          })()}

          {/* LayerNorm 1 */}
          {(y += spacing) && null}
          <Block id="layernorm1" x={centerX - blockW / 2} y={y} width={blockW} height={blockH}
            label="Layer Norm" variant="muted"
            highlighted={isActive('layernorm1')} dimmed={isDimmed('layernorm1')}
            onMouseEnter={(e) => showTooltip(e, <div>RMSNorm / LayerNorm<br/>Normalizes across d_model dimension</div>)}
            onMouseMove={moveTooltip} onMouseLeave={hideTooltip} />

          {/* Arrow: LN1 → QKV */}
          <Arrow id="arrow-ln1-qkv" from={{ x: centerX, y: y + blockH }} to={{ x: centerX, y: y + spacing }}
            animated={activeArrows.has('arrow-ln1-qkv')} dimmed={isDimmed('q-proj')} />

          {/* QKV Projections */}
          {(y += spacing) && null}
          <Block id="q-proj" x={centerX - smallBlockW * 1.7} y={y} width={smallBlockW} height={smallBlockH}
            label="Q Projection" variant="compute" fontSize={10}
            highlighted={isActive('q-proj')} dimmed={isDimmed('q-proj')}
            onMouseEnter={(e) => showTooltip(e, <div>Query projection: W_Q × input<br/>GEMM: (B×L, d) × (d, d) → (B×L, d)<br/><strong>Compute-bound in prefill</strong></div>)}
            onMouseMove={moveTooltip} onMouseLeave={hideTooltip} />
          <Block id="k-proj" x={centerX - smallBlockW / 2} y={y} width={smallBlockW} height={smallBlockH}
            label="K Projection" variant="compute" fontSize={10}
            highlighted={isActive('k-proj')} dimmed={isDimmed('k-proj')} />
          <Block id="v-proj" x={centerX + smallBlockW * 0.7} y={y} width={smallBlockW} height={smallBlockH}
            label="V Projection" variant="compute" fontSize={10}
            highlighted={isActive('v-proj')} dimmed={isDimmed('v-proj')} />

          {/* Arrow: QKV → Attention */}
          <Arrow id="arrow-qkv-attn" from={{ x: centerX, y: y + smallBlockH }} to={{ x: centerX, y: y + spacing }}
            animated={activeArrows.has('arrow-qkv-attn')} dimmed={isDimmed('attention')} />

          {/* Attention */}
          {(y += spacing) && null}
          <Block id="attention" x={centerX - blockW / 2} y={y} width={blockW} height={blockH + 8}
            label="Scaled Dot-Product" sublabel="Attention" variant="accent"
            highlighted={isActive('attention')} dimmed={isDimmed('attention')}
            onMouseEnter={(e) => showTooltip(e, (
              <div>
                <div className="font-semibold">Q·K<sup>T</sup> / √d → softmax → ×V</div>
                <div className="mt-1">Prefill: O(n²) — compute-bound</div>
                <div>Decode: O(n) — memory-bound (KV-cache)</div>
                <div className="mt-1 text-accent-cyan">FlashAttention optimizes this</div>
              </div>
            ))}
            onMouseMove={moveTooltip} onMouseLeave={hideTooltip} />

          {/* Arrow: Attn → Out Proj */}
          <Arrow id="arrow-attn-out" from={{ x: centerX, y: y + blockH + 8 }} to={{ x: centerX, y: y + spacing + 8 }}
            animated={activeArrows.has('arrow-attn-out')} dimmed={isDimmed('out-proj')} />

          {/* Output Projection */}
          {(y += spacing + 8) && null}
          <Block id="out-proj" x={centerX - blockW / 2} y={y} width={blockW} height={blockH}
            label="Output Projection" variant="primary"
            highlighted={isActive('out-proj')} dimmed={isDimmed('out-proj')} />

          {/* Add + Residual 1 → result */}
          <Arrow id="arrow-residual1" from={{ x: centerX, y: y + blockH }} to={{ x: centerX, y: y + spacing - 4 }}
            label="+ residual" animated={activeArrows.has('arrow-residual1')} dimmed={isDimmed('layernorm2')} />

          {/* LayerNorm 2 */}
          {(y += spacing) && null}
          <Block id="layernorm2" x={centerX - blockW / 2} y={y} width={blockW} height={blockH}
            label="Layer Norm" variant="muted"
            highlighted={isActive('layernorm2')} dimmed={isDimmed('layernorm2')} />

          {/* Arrow: LN2 → FFN */}
          <Arrow id="arrow-ln2-ffn" from={{ x: centerX, y: y + blockH }} to={{ x: centerX, y: y + spacing }}
            animated={activeArrows.has('arrow-ln2-ffn')} dimmed={isDimmed('ffn-up')} />

          {/* Residual path 2 (right side) */}
          {(() => {
            const resStartY = y + blockH / 2
            const resEndY = y + spacing * 3 + blockH / 2 - 8
            const resX = centerX + blockW / 2 + 40
            return (
              <g id="residual2" style={{ opacity: isDimmed('residual2') ? 0.2 : 1 }}>
                <motion.path
                  d={`M${centerX + blockW / 2},${resStartY} L${resX},${resStartY} L${resX},${resEndY} L${centerX + blockW / 2},${resEndY}`}
                  fill="none" stroke={isActive('residual2') ? colors.success : colors.textLight}
                  strokeWidth={isActive('residual2') ? 2 : 1} strokeDasharray="6 3" strokeOpacity={0.6}
                />
                <text x={resX + 5} y={(resStartY + resEndY) / 2} dominantBaseline="central"
                  fill={colors.success} fontSize={9} fontWeight={500} fontFamily="Inter, system-ui, sans-serif"
                  transform={`rotate(90, ${resX + 5}, ${(resStartY + resEndY) / 2})`}>
                  Residual
                </text>
              </g>
            )
          })()}

          {/* FFN */}
          {(y += spacing) && null}
          <Block id="ffn-up" x={centerX - smallBlockW * 1.5 - 5} y={y} width={smallBlockW} height={smallBlockH}
            label="Up Proj" sublabel="d → 4d" variant="memory" fontSize={10} sublabelSize={8}
            highlighted={isActive('ffn-up')} dimmed={isDimmed('ffn-up')}
            onMouseEnter={(e) => showTooltip(e, (
              <div>
                <div className="font-semibold">FFN Up-Projection</div>
                <div>GEMM: (B×L, d) × (d, 4d)</div>
                <div className="mt-1 text-accent-orange">Dominates decode latency —</div>
                <div className="text-accent-orange">Must load all 4d² params from DRAM</div>
              </div>
            ))}
            onMouseMove={moveTooltip} onMouseLeave={hideTooltip} />
          <Block id="ffn-act" x={centerX - smallBlockW / 2} y={y} width={smallBlockW} height={smallBlockH}
            label="SiLU / GeLU" variant="warning" fontSize={10}
            highlighted={isActive('ffn-act')} dimmed={isDimmed('ffn-act')} />
          <Block id="ffn-down" x={centerX + smallBlockW * 0.5 + 5} y={y} width={smallBlockW} height={smallBlockH}
            label="Down Proj" sublabel="4d → d" variant="memory" fontSize={10} sublabelSize={8}
            highlighted={isActive('ffn-down')} dimmed={isDimmed('ffn-down')} />

          {/* Arrow: FFN → Output */}
          <Arrow id="arrow-residual2" from={{ x: centerX, y: y + smallBlockH }} to={{ x: centerX, y: y + spacing }}
            label="+ residual" animated={activeArrows.has('arrow-residual2')} dimmed={isDimmed('output')} />

          {/* Output */}
          {(y += spacing) && null}
          <Block id="output" x={centerX - blockW / 2} y={y} width={blockW} height={blockH}
            label="Block Output" sublabel="→ Next layer" variant="muted"
            highlighted={isActive('output')} dimmed={isDimmed('output')} />

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

export function TransformerBlockDiagram() {
  return (
    <DiagramThemeProvider>
      <TransformerBlockInner />
    </DiagramThemeProvider>
  )
}

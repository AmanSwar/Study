'use client'

import { useState, useEffect, useRef, useCallback } from 'react'

interface DataFlowPacket {
  id: number
  progress: number // 0 to 1 along path
}

interface UseDataFlowAnimationOptions {
  active: boolean
  speed?: number         // progress per frame (0.005 = moderate)
  packetCount?: number
  packetSpacing?: number // spacing between packets (0-1)
}

interface UseDataFlowAnimationReturn {
  packets: DataFlowPacket[]
}

export function useDataFlowAnimation({
  active,
  speed = 0.008,
  packetCount = 3,
  packetSpacing = 0.33,
}: UseDataFlowAnimationOptions): UseDataFlowAnimationReturn {
  const [packets, setPackets] = useState<DataFlowPacket[]>(() =>
    Array.from({ length: packetCount }, (_, i) => ({
      id: i,
      progress: i * packetSpacing,
    }))
  )
  const rafRef = useRef<number>(0)

  const animate = useCallback(() => {
    setPackets((prev) =>
      prev.map((p) => ({
        ...p,
        progress: (p.progress + speed) % (1 + packetSpacing),
      }))
    )
    rafRef.current = requestAnimationFrame(animate)
  }, [speed, packetSpacing])

  useEffect(() => {
    if (active) {
      rafRef.current = requestAnimationFrame(animate)
    }
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [active, animate])

  return {
    packets: packets.filter((p) => p.progress >= 0 && p.progress <= 1),
  }
}

/**
 * Get a point along a straight line between two points at progress t (0-1)
 */
export function interpolateLine(
  x1: number, y1: number,
  x2: number, y2: number,
  t: number
): { x: number; y: number } {
  return {
    x: x1 + (x2 - x1) * t,
    y: y1 + (y2 - y1) * t,
  }
}

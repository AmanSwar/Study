import { Track } from '@/lib/types'
import { mlsysTrack } from './mlsys'
import { intelTrack } from './intel'
import { qualcommTrack } from './qualcomm'
import { quantTrack } from './quant'

export { mlsysTrack, intelTrack, qualcommTrack, quantTrack }

export const allTracks: Track[] = [mlsysTrack, intelTrack, qualcommTrack, quantTrack]

export function getTrack(id: string): Track | undefined {
  return allTracks.find(t => t.id === id)
}

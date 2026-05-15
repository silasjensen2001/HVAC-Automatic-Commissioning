import axios from 'axios'
import type { SimResults } from './types'

const BASE = 'http://localhost:8000'

export async function runSimulation(
  nodes: unknown[],
  edges: unknown[],
  simParams: unknown,
  signal?: AbortSignal,
): Promise<SimResults> {
  const { data } = await axios.post<SimResults>(`${BASE}/simulate`, {
    nodes,
    edges,
    sim_params: simParams,
  }, { signal })
  return data
}

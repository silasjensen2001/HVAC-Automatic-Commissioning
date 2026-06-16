import axios from 'axios'
import type { SimResults, SystemInfo } from './types'

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

export async function fetchSystemInfo(
  nodes: unknown[],
  edges: unknown[],
  simParams: unknown,
): Promise<SystemInfo> {
  const { data } = await axios.post<SystemInfo>(`${BASE}/system_info`, {
    nodes,
    edges,
    sim_params: simParams,
  })
  return data
}

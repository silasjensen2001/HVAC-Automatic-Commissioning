export interface HxData {
  label: string
  T_out_target: number       // °C
  volume_flow_rate: number   // m³/s
  water_supply_T: number     // °C
  num_segments: number
  num_pipes: number
  gamma: number
  cross_area_water: number
  heat_exchanger_depth: number
  heat_exchanger_width: number
  heat_exchanger_height: number
  Kvs: number
  flipHandles?: boolean
}

export interface DuctData {
  label: string
  volume_flow_rate: number
  cross_section_area: number
  duct_length: number
  num_segments: number
  flipHandles?: boolean
}

export interface JunctionData {
  label: string
  num_inputs:   number
  num_outputs:  number
  inputFlows?:  (number | null)[]
  outputFlow?:  number | null
  exhaustFlow?: number | null
}

export interface OutdoorAirData {
  label: string
  T_fresh: number
  disturbance_type: 'constant' | 'sinusoidal'
  disturbance_amplitude: number
  disturbance_period: number
  flipHandles?: boolean
}

export interface FanData {
  label: string
  volume_flow_rate: number  // m³/s
  flipHandles?: boolean
}

export interface ExhaustData {
  label: string
  volume_flow_rate: number
  flipHandles?: boolean
}

export interface EdgeData {
  flow_rate: number
}

export interface SimParams {
  t_end: number
  Q_scale: number
  R_scale: number
  model_mode: 'linear' | 'nonlinear'
  controller_type: 'lqr' | 'lmi'
}

export interface OutputSeries {
  label: string
  y: number[]
  ref: number
}

export interface ValveSeries {
  label: string
  y: number[]
}

export interface SteadyStateEntry {
  label: string
  temp_C: number
  ref_C: number
  valve: number
}

export interface SimMetrics {
  K_I: number[][]
  K_x: number[][]
  N:   number[][]
  M:   number[][]
  cl_eigenvalues: { re: number; im: number }[]
  condition_number: number
  steady_state: Record<string, SteadyStateEntry>
  actuated_ids: string[]
}

export interface SimResults {
  t: number[]
  outputs: Record<string, OutputSeries>
  valves: Record<string, ValveSeries>
  metrics: SimMetrics
  d_signal: number[]   // outdoor air temperature over time (°C)
}

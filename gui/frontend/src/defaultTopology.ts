import type { Node, Edge } from 'reactflow'
import type { HxData, DuctData, JunctionData, OutdoorAirData, FanData, ExhaustData } from './types'

const q_fresh = 2615 / 3600   // m³/s  ≈ 0.7264  (2615 m³/h)
const q_main  = 8638 / 3600   // m³/s  ≈ 2.3994  (8638 m³/h)

const coolerGeom = {
  num_segments:          5,
  num_pipes:             10,
  gamma:                 951.87,
  cross_area_water:      0.000402,
  heat_exchanger_depth:  0.12,
  heat_exchanger_width:  0.5,
  heat_exchanger_height: 0.5,
  Kvs:                   1.6471,
}

const heaterGeom = {
  num_segments:          5,
  num_pipes:             10,
  gamma:                 951.87,
  cross_area_water:      0.000201,
  heat_exchanger_depth:  0.06,
  heat_exchanger_width:  0.5,
  heat_exchanger_height: 0.5,
  Kvs:                   1.6471,
}

export const defaultNodes: Node[] = [
  {
    id: 'outdoor_air', type: 'outdoor_air',
    position: { x: -162, y: 97 },
    data: {
      label: 'Outdoor Air', T_fresh: 23,
      disturbance_type: 'constant', disturbance_amplitude: 3, disturbance_period: 86400,
    } as OutdoorAirData,
  },
  {
    id: 'fan_fresh', type: 'fan',
    position: { x: 46, y: 97 },
    data: { label: 'Fresh Fan', volume_flow_rate: q_fresh } as FanData,
  },
  {
    id: 'pre_cooler', type: 'cooler',
    position: { x: 255, y: 88 },
    data: { label: 'Pre Cooler', T_out_target: 9.9, volume_flow_rate: q_fresh, water_supply_T: 4, ...coolerGeom } as HxData,
  },
  {
    id: 'pre_heater', type: 'heater',
    position: { x: 475, y: 88 },
    data: { label: 'Pre Heater', T_out_target: 18.2, volume_flow_rate: q_fresh, water_supply_T: 66.9, ...heaterGeom } as HxData,
  },
  {
    id: 'duct_1', type: 'airduct',
    position: { x: 697, y: 96 },
    data: { label: 'Fresh Duct', volume_flow_rate: q_fresh, cross_section_area: 0.25, duct_length: 5, num_segments: 5 } as DuctData,
  },
  {
    id: 'junction_1', type: 'junction',
    position: { x: 917, y: 200 },
    data: { label: 'Junction 1', num_inputs: 2, num_outputs: 2 } as JunctionData,
  },
  {
    id: 'exhaust_1', type: 'exhaust',
    position: { x: 1135, y: 197 },
    data: { label: 'Exhaust', volume_flow_rate: 0 } as ExhaustData,
  },
  {
    id: 'duct_2', type: 'airduct',
    position: { x: 917, y: 358 },
    data: { label: 'Main Duct', volume_flow_rate: q_main, cross_section_area: 0.25, duct_length: 5, num_segments: 5, flipHandles: true } as DuctData,
  },
  {
    id: 'fan_main', type: 'fan',
    position: { x: 700, y: 357 },
    data: { label: 'Main Fan', volume_flow_rate: q_main, flipHandles: true } as FanData,
  },
  {
    id: 'cooler_main', type: 'cooler',
    position: { x: 500, y: 349 },
    data: { label: 'Main Cooler', T_out_target: 9.9, volume_flow_rate: q_main, water_supply_T: 4, ...coolerGeom, flipHandles: true } as HxData,
  },
  {
    id: 'heater_main', type: 'heater',
    position: { x: 296, y: 350 },
    data: { label: 'Main Heater', T_out_target: 21.4, volume_flow_rate: q_main, water_supply_T: 66.9, ...heaterGeom, flipHandles: true } as HxData,
  },
  {
    id: 'room', type: 'airduct',
    position: { x: 292, y: 239 },
    data: { label: 'Room', volume_flow_rate: q_main, cross_section_area: 20, duct_length: 8, num_segments: 5 } as DuctData,
  },
  {
    id: 'duct_return', type: 'airduct',
    position: { x: 500, y: 238 },
    data: { label: 'Return Duct', volume_flow_rate: q_main, cross_section_area: 0.25, duct_length: 5, num_segments: 5 } as DuctData,
  },
]

export const defaultEdges: Edge[] = [
  { id: 'e0',  source: 'outdoor_air', target: 'fan_fresh',   animated: true },
  { id: 'e1',  source: 'fan_fresh',   target: 'pre_cooler',  animated: true },
  { id: 'e2',  source: 'pre_cooler',  target: 'pre_heater',  animated: true },
  { id: 'e3',  source: 'pre_heater',  target: 'duct_1',      animated: true },
  { id: 'e4',  source: 'duct_1',      sourceHandle: 'output',   target: 'junction_1', targetHandle: 'input-0',  animated: true, data: { flow_rate: q_fresh } },
  { id: 'e5',  source: 'duct_return', sourceHandle: 'output',   target: 'junction_1', targetHandle: 'input-1',  animated: true, data: { flow_rate: q_main  } },
  { id: 'e6',  source: 'junction_1',  sourceHandle: 'output-0', target: 'exhaust_1',  targetHandle: 'input',    animated: true, data: { flow_rate: q_fresh } },
  { id: 'e7',  source: 'junction_1',  sourceHandle: 'output-1', target: 'duct_2',     targetHandle: 'input',    animated: true, data: { flow_rate: q_main  } },
  { id: 'e8',  source: 'duct_2',      target: 'fan_main',    animated: true },
  { id: 'e9',  source: 'fan_main',    target: 'cooler_main', animated: true },
  { id: 'e10', source: 'cooler_main', sourceHandle: 'output', target: 'heater_main', targetHandle: 'input', animated: true },
  { id: 'e11', source: 'heater_main', sourceHandle: 'output', target: 'room',        targetHandle: 'input', animated: true },
  { id: 'e12', source: 'room',        target: 'duct_return', animated: true },
]

export function getDefaultNodeData(type: string): Record<string, unknown> {
  if (type === 'cooler') {
    return { label: 'Cooler', T_out_target: 10.0, volume_flow_rate: 1.0, water_supply_T: 4.0, ...coolerGeom }
  }
  if (type === 'heater') {
    return { label: 'Heater', T_out_target: 22.0, volume_flow_rate: 1.0, water_supply_T: 66.9, ...heaterGeom }
  }
  if (type === 'airduct') {
    return { label: 'Duct', volume_flow_rate: 1.0, cross_section_area: 0.25, duct_length: 5.0, num_segments: 5 }
  }
  if (type === 'outdoor_air') {
    return { label: 'Outdoor Air', T_fresh: 23.0, disturbance_type: 'constant', disturbance_amplitude: 3.0, disturbance_period: 86400 }
  }
  if (type === 'fan') {
    return { label: 'Fan', volume_flow_rate: 1.0 }
  }
  if (type === 'exhaust') {
    return { label: 'Exhaust', volume_flow_rate: 0 }
  }
  return { label: 'Junction', num_inputs: 2, num_outputs: 1 }
}

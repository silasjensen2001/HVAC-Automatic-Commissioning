import type { Node, Edge } from 'reactflow'

export interface GraphWarning {
  nodeId?: string
  message: string
  severity: 'error' | 'warning'
}

export function validateGraph(nodes: Node[], edges: Edge[]): GraphWarning[] {
  const warnings: GraphWarning[] = []

  const inDegree  = new Map<string, number>()
  const outDegree = new Map<string, number>()
  for (const n of nodes) { inDegree.set(n.id, 0); outDegree.set(n.id, 0) }
  for (const e of edges) {
    inDegree.set(e.target,  (inDegree.get(e.target)  ?? 0) + 1)
    outDegree.set(e.source, (outDegree.get(e.source) ?? 0) + 1)
  }

  const hxNodes      = nodes.filter(n => n.type === 'cooler' || n.type === 'heater')
  const outdoorNodes = nodes.filter(n => n.type === 'outdoor_air')

  // Must have at least one HX to simulate
  if (hxNodes.length === 0) {
    warnings.push({
      severity: 'error',
      message: 'No cooler or heater nodes — nothing to simulate.',
    })
  }

  // Outdoor air node not present (simulation falls back to sidebar defaults)
  if (outdoorNodes.length === 0) {
    warnings.push({
      severity: 'warning',
      message: 'No Outdoor Air node on canvas — using default 23 °C constant.',
    })
  }

  // Multiple outdoor air nodes — ambiguous
  if (outdoorNodes.length > 1) {
    warnings.push({
      severity: 'warning',
      message: `${outdoorNodes.length} Outdoor Air nodes found — only the first will be used.`,
    })
  }

  for (const n of nodes) {
    const label = (n.data as Record<string, unknown>)?.label as string ?? n.id
    const ins   = inDegree.get(n.id)  ?? 0
    const outs  = outDegree.get(n.id) ?? 0

    // Nodes that need an input connection
    if (['cooler', 'heater', 'airduct', 'fan'].includes(n.type ?? '')) {
      if (ins === 0) {
        warnings.push({
          nodeId: n.id,
          severity: 'error',
          message: `"${label}" has no incoming connection.`,
        })
      }
    }

    // Nodes that need an output connection (except terminal ducts/rooms)
    if (['outdoor_air', 'fan'].includes(n.type ?? '')) {
      if (outs === 0) {
        warnings.push({
          nodeId: n.id,
          severity: 'warning',
          message: `"${label}" has no outgoing connection.`,
        })
      }
    }

    // Junction must have ≥ 2 inputs to be meaningful
    if (n.type === 'junction') {
      if (ins < 2) {
        warnings.push({
          nodeId: n.id,
          severity: 'warning',
          message: `"${label}" has only ${ins} incoming connection${ins === 1 ? '' : 's'} — junctions mix at least 2 streams.`,
        })
      }
      if (outs === 0) {
        warnings.push({
          nodeId: n.id,
          severity: 'error',
          message: `"${label}" has no outgoing connection.`,
        })
      }
    }

    // HX / duct nodes with no output are dead ends in the flow graph
    if (['cooler', 'heater', 'airduct'].includes(n.type ?? '') && outs === 0) {
      warnings.push({
        nodeId: n.id,
        severity: 'warning',
        message: `"${label}" has no outgoing connection — it is a dead end.`,
      })
    }
  }

  return warnings
}

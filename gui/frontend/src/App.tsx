import { useState, useCallback, useRef, useEffect } from 'react'
import { toPng } from 'html-to-image'
import ReactFlow, {
  Background,
  Controls,
  addEdge,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type Connection,
  type ReactFlowInstance,
} from 'reactflow'
import 'reactflow/dist/style.css'

import CoolerNode     from './components/nodes/CoolerNode'
import HeaterNode     from './components/nodes/HeaterNode'
import DuctNode       from './components/nodes/DuctNode'
import JunctionNode   from './components/nodes/JunctionNode'
import OutdoorAirNode from './components/nodes/OutdoorAirNode'
import FanNode        from './components/nodes/FanNode'
import ExhaustNode    from './components/nodes/ExhaustNode'
import NodePalette      from './components/NodePalette'
import PropertiesPanel  from './components/PropertiesPanel'
import ResultsPanel     from './components/ResultsPanel'

import axios from 'axios'

// ── Flow-propagation helpers ──────────────────────────────────────────────────

/** Can `fromId` reach `toId` by following directed edges (passes through any node type)? */
function canReach(fromId: string, toId: string, allEdges: Edge[]): boolean {
  const visited = new Set([fromId])
  const queue   = [fromId]
  while (queue.length > 0) {
    const curr = queue.shift()!
    for (const e of allEdges) {
      if (e.source !== curr || visited.has(e.target)) continue
      if (e.target === toId) return true
      visited.add(e.target)
      queue.push(e.target)
    }
  }
  return false
}

/** BFS backward from `startId` — returns the nearest fan found upstream, or null. */
function nearestUpstreamFan(startId: string, allNodes: Node[], allEdges: Edge[]): Node | null {
  const nodeMap = new Map(allNodes.map(n => [n.id, n]))
  const visited = new Set([startId])
  const queue   = [startId]
  while (queue.length > 0) {
    const curr = queue.shift()!
    for (const e of allEdges) {
      if (e.target !== curr || visited.has(e.source)) continue
      visited.add(e.source)
      const src = nodeMap.get(e.source)
      if (src?.type === 'fan') return src
      queue.push(e.source)
    }
  }
  return null
}

/** BFS forward from `startId` — returns the nearest fan found downstream, or null. */
function nearestDownstreamFan(startId: string, allNodes: Node[], allEdges: Edge[]): Node | null {
  const nodeMap = new Map(allNodes.map(n => [n.id, n]))
  const visited = new Set([startId])
  const queue   = [startId]
  while (queue.length > 0) {
    const curr = queue.shift()!
    for (const e of allEdges) {
      if (e.source !== curr || visited.has(e.target)) continue
      visited.add(e.target)
      const tgt = nodeMap.get(e.target)
      if (tgt?.type === 'fan') return tgt
      queue.push(e.target)
    }
  }
  return null
}
import { defaultNodes, defaultEdges, getDefaultNodeData } from './defaultTopology'
import { runSimulation } from './api'
import { validateGraph, type GraphWarning } from './validateGraph'
import type { SimParams, SimResults } from './types'
import './index.css'

function loadFromStorage<T>(key: string, fallback: T): T {
  try {
    const s = localStorage.getItem(key)
    if (s) return JSON.parse(s) as T
  } catch {}
  return fallback
}

const nodeTypes = {
  cooler:      CoolerNode,
  heater:      HeaterNode,
  airduct:     DuctNode,
  junction:    JunctionNode,
  outdoor_air: OutdoorAirNode,
  fan:         FanNode,
  exhaust:     ExhaustNode,
}

const defaultSimParams: SimParams = {
  t_end:           200,
  Q_scale:         5.0,
  R_scale:         800.0,
  model_mode:      'nonlinear',
  controller_type: 'lqr',
}

export default function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState(loadFromStorage('hvac_nodes', defaultNodes))
  const [edges, setEdges, onEdgesChange] = useEdgesState(loadFromStorage('hvac_edges', defaultEdges))

  const [simParams, setSimParams]       = useState<SimParams>(loadFromStorage('hvac_simParams', defaultSimParams))
  const [simResults, setSimResults]     = useState<SimResults | null>(null)
  const [isLoading, setIsLoading]       = useState(false)
  const [error, setError]               = useState<string | null>(null)
  const [showResults, setShowResults]   = useState(false)
  const [warnings, setWarnings]         = useState<GraphWarning[]>([])
  const [showWarnings, setShowWarnings] = useState(false)
  const [rightOpen, setRightOpen]       = useState(true)
  const [theme, setTheme]               = useState<'dark' | 'light'>('dark')
  const [snapOn, setSnapOn]             = useState(false)
  const [guides, setGuides]             = useState<{
    x: number[]; y: number[]
    vp: { x: number; y: number; zoom: number }
    dists: Array<{ x1: number; y1: number; x2: number; y2: number; gap: number }>
  } | null>(null)
  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark')

  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null)

  const rfWrapperRef    = useRef<HTMLDivElement>(null)
  const rfInstance      = useRef<ReactFlowInstance | null>(null)
  const abortController = useRef<AbortController | null>(null)
  const fileInputRef    = useRef<HTMLInputElement>(null)

  const selectedNode = nodes.find(n => n.id === selectedNodeId) ?? null
  const selectedEdge = edges.find(e => e.id === selectedEdgeId) ?? null

  const onConnect = useCallback(
    (params: Connection) =>
      setEdges(eds => {
        // Treat a missing targetHandle as 'input' — all our single-port nodes use that id
        const norm = (h: string | null | undefined) => h ?? 'input'
        const alreadyConnected = eds.some(
          e => e.target === params.target &&
               norm(e.targetHandle) === norm(params.targetHandle),
        )
        if (alreadyConnected) return eds
        return addEdge({ ...params, animated: true, data: { flow_rate: 1.0 } }, eds)
      }),
    [setEdges],
  )

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const type = e.dataTransfer.getData('application/reactflow')
    if (!type || !rfInstance.current || !rfWrapperRef.current) return

    const bounds = rfWrapperRef.current.getBoundingClientRect()
    const position = rfInstance.current.project({
      x: e.clientX - bounds.left,
      y: e.clientY - bounds.top,
    })

    const id = `${type}_${Date.now()}`
    const newNode: Node = { id, type, position, data: getDefaultNodeData(type) }
    setNodes(nds => [...nds, newNode])
  }, [setNodes])

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    setSelectedNodeId(node.id)
    setSelectedEdgeId(null)
  }, [])

  const onEdgeClick = useCallback((_: React.MouseEvent, edge: Edge) => {
    setSelectedEdgeId(edge.id)
    setSelectedNodeId(null)
  }, [])

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null)
    setSelectedEdgeId(null)
  }, [])

  const SNAP_THRESHOLD = 6  // screen pixels
  const onNodeDrag = useCallback((_: React.MouseEvent, dragged: Node) => {
    if (!rfInstance.current) return
    const vp  = rfInstance.current.getViewport()
    const fts = (fx: number, fy: number) => ({ sx: fx * vp.zoom + vp.x, sy: fy * vp.zoom + vp.y })
    const thresh = SNAP_THRESHOLD / vp.zoom

    const dw = dragged.width  ?? 160, dh = dragged.height ?? 80
    const dx = dragged.position.x,    dy = dragged.position.y
    const dcx = dx + dw / 2,          dcy = dy + dh / 2

    const xGuides = new Set<number>()
    const yGuides = new Set<number>()

    // nearest neighbour in each cardinal direction
    let left:   { edge: number; gap: number; cy: number } | null = null
    let right:  { edge: number; gap: number; cy: number } | null = null
    let top:    { edge: number; gap: number; cx: number } | null = null
    let bottom: { edge: number; gap: number; cx: number } | null = null

    for (const n of nodes) {
      if (n.id === dragged.id) continue
      const nw = n.width ?? 160, nh = n.height ?? 80
      const nx = n.position.x,   ny = n.position.y
      const ncx = nx + nw / 2,   ncy = ny + nh / 2

      // center-to-center alignment guides
      if (Math.abs(dcx - ncx) < thresh) xGuides.add(ncx)
      if (Math.abs(dcy - ncy) < thresh) yGuides.add(ncy)

      // left neighbour: n's right edge is left of dragged's left edge
      if (nx + nw <= dx) {
        const gap = dx - (nx + nw)
        if (!left || gap < left.gap) left = { edge: nx + nw, gap, cy: ncy }
      }
      // right neighbour: n's left edge is right of dragged's right edge
      if (nx >= dx + dw) {
        const gap = nx - (dx + dw)
        if (!right || gap < right.gap) right = { edge: nx, gap, cy: ncy }
      }
      // top neighbour: n's bottom edge is above dragged's top edge
      if (ny + nh <= dy) {
        const gap = dy - (ny + nh)
        if (!top || gap < top.gap) top = { edge: ny + nh, gap, cx: ncx }
      }
      // bottom neighbour: n's top edge is below dragged's bottom edge
      if (ny >= dy + dh) {
        const gap = ny - (dy + dh)
        if (!bottom || gap < bottom.gap) bottom = { edge: ny, gap, cx: ncx }
      }
    }

    // Snap center to guide
    const snapX = xGuides.size > 0 ? Array.from(xGuides)[0] : null
    const snapY = yGuides.size > 0 ? Array.from(yGuides)[0] : null
    if (snapX !== null || snapY !== null) {
      setNodes(nds => nds.map(n =>
        n.id !== dragged.id ? n : {
          ...n,
          position: {
            x: snapX !== null ? snapX - dw / 2 : n.position.x,
            y: snapY !== null ? snapY - dh / 2 : n.position.y,
          },
        }
      ))
    }

    // Build screen-space distance indicators
    const dists: Array<{ x1: number; y1: number; x2: number; y2: number; gap: number }> = []
    if (left) {
      const sy = fts(0, dcy).sy
      dists.push({ x1: fts(left.edge, 0).sx, y1: sy, x2: fts(dx, 0).sx, y2: sy, gap: Math.round(left.gap) })
    }
    if (right) {
      const sy = fts(0, dcy).sy
      dists.push({ x1: fts(dx + dw, 0).sx, y1: sy, x2: fts(right.edge, 0).sx, y2: sy, gap: Math.round(right.gap) })
    }
    if (top) {
      const sx = fts(dcx, 0).sx
      dists.push({ x1: sx, y1: fts(0, top.edge).sy, x2: sx, y2: fts(0, dy).sy, gap: Math.round(top.gap) })
    }
    if (bottom) {
      const sx = fts(dcx, 0).sx
      dists.push({ x1: sx, y1: fts(0, dy + dh).sy, x2: sx, y2: fts(0, bottom.edge).sy, gap: Math.round(bottom.gap) })
    }

    setGuides({ x: Array.from(xGuides), y: Array.from(yGuides), vp, dists })
  }, [nodes, setNodes])

  const onNodeDragStop = useCallback(() => setGuides(null), [])

  const updateNodeData = useCallback((id: string, updates: Record<string, unknown>) => {
    setNodes(nds => nds.map(n => n.id === id ? { ...n, data: { ...n.data, ...updates } } : n))
  }, [setNodes])

  const updateEdgeData = useCallback((id: string, updates: Record<string, unknown>) => {
    setEdges(eds => eds.map(e => e.id === id ? { ...e, data: { ...e.data, ...updates } } : e))
  }, [setEdges])

  // Propagate fan flow rates with proper junction mass balance
  useEffect(() => {
    const nodeMap = new Map(nodes.map(n => [n.id, n]))
    const edgeMap = new Map(edges.map(e => [e.id, e]))
    let nodesChanged = false
    let edgesChanged = false
    const fans = nodes.filter(n => n.type === 'fan')

    // Phase 1: Forward BFS from each fan — set downstream node flows
    for (const fan of fans) {
      const d        = fan.data as Record<string, unknown>
      const flowRate = d.volume_flow_rate as number
      const fanLabel = d.label as string
      const visited  = new Set<string>([fan.id])
      const queue    = [fan.id]

      while (queue.length > 0) {
        const curr = queue.shift()!
        for (const edge of edges) {
          if (edge.source !== curr || visited.has(edge.target)) continue
          visited.add(edge.target)
          const tNode = nodeMap.get(edge.target)
          if (!tNode) continue

          if (tNode.type === 'junction') {
            if ((edge.data as Record<string, unknown>)?.flow_rate !== flowRate) {
              edgeMap.set(edge.id, { ...edge, data: { ...edge.data, flow_rate: flowRate } })
              edgesChanged = true
            }
            continue
          }
          if (tNode.type === 'fan' || tNode.type === 'outdoor_air' || tNode.type === 'exhaust') continue

          const td = tNode.data as Record<string, unknown>
          if (td.volume_flow_rate !== flowRate || td.flowSource !== fanLabel) {
            nodeMap.set(edge.target, { ...tNode, data: { ...td, volume_flow_rate: flowRate, flowSource: fanLabel } })
            nodesChanged = true
          }
          queue.push(edge.target)
        }
      }
    }

    // Phase 2: Backward BFS from each fan — set upstream nodes (e.g. duct between junction and fan)
    for (const fan of fans) {
      const d        = fan.data as Record<string, unknown>
      const flowRate = d.volume_flow_rate as number
      const fanLabel = d.label as string
      const visited  = new Set<string>([fan.id])
      const queue    = [fan.id]

      while (queue.length > 0) {
        const curr = queue.shift()!
        for (const edge of edges) {
          if (edge.target !== curr || visited.has(edge.source)) continue
          visited.add(edge.source)
          const sNode = nodeMap.get(edge.source)
          if (!sNode) continue
          if (['junction', 'fan', 'outdoor_air', 'exhaust'].includes(sNode.type ?? '')) continue

          const sd = sNode.data as Record<string, unknown>
          if (sd.volume_flow_rate !== flowRate || sd.flowSource !== fanLabel) {
            nodeMap.set(edge.source, { ...sNode, data: { ...sd, volume_flow_rate: flowRate, flowSource: fanLabel } })
            nodesChanged = true
          }
          queue.push(edge.source)
        }
      }
    }

    // Phase 3: Junction exhaust — fresh inputs that don't "return" leave as exhaust.
    // The return path carries the FULL loop flow (q_main); the junction shows what's exhausted.
    for (const jOrig of nodes.filter(n => n.type === 'junction')) {
      const junction   = nodeMap.get(jOrig.id) ?? jOrig
      const downFan    = nearestDownstreamFan(junction.id, nodes, edges)
      if (!downFan) continue
      const outputFlow = (downFan.data as Record<string, unknown>).volume_flow_rate as number

      // Sum only "fresh" inputs (fans NOT reachable from this junction = not in the loop)
      const inEdges = [...edgeMap.values()].filter(e => e.target === junction.id)
      let exhaustFlow = 0
      for (const inEdge of inEdges) {
        const upFan = nearestUpstreamFan(inEdge.source, nodes, edges)
        if (upFan && !canReach(junction.id, upFan.id, edges)) {
          exhaustFlow += ((inEdge.data as Record<string, unknown>)?.flow_rate as number) ?? 0
        }
      }

      // Auto-set flow on any exhaust node connected as a junction output
      for (const outEdge of [...edgeMap.values()].filter(e => e.source === junction.id)) {
        const tNode = nodeMap.get(outEdge.target)
        if (tNode?.type === 'exhaust') {
          if ((outEdge.data as Record<string, unknown>)?.flow_rate !== exhaustFlow) {
            edgeMap.set(outEdge.id, { ...outEdge, data: { ...outEdge.data, flow_rate: exhaustFlow } })
            edgesChanged = true
          }
        }
      }

      // Store display values on junction
      const jNode = nodeMap.get(junction.id) ?? junction
      const jd    = jNode.data as Record<string, unknown>
      if (jd.outputFlow !== outputFlow || jd.exhaustFlow !== exhaustFlow) {
        nodeMap.set(junction.id, { ...jNode, data: { ...jd, outputFlow, exhaustFlow } })
        nodesChanged = true
      }
    }

    // Phase 4: Sync junction inputFlows from updated edgeMap
    for (const jOrig of nodes.filter(n => n.type === 'junction')) {
      const n     = nodeMap.get(jOrig.id) ?? jOrig
      const numIn = Math.max(1, (n.data as Record<string, unknown>).num_inputs as number ?? 2)
      const inputFlows = Array.from({ length: numIn }, (_, i) => {
        const edge = [...edgeMap.values()].find(
          e => e.target === n.id && e.targetHandle === `input-${i}`,
        )
        return edge ? ((edge.data as Record<string, unknown>).flow_rate as number ?? 1.0) : null
      })
      const prev = (n.data as Record<string, unknown>).inputFlows as (number | null)[] | undefined
      const same = prev?.length === inputFlows.length && prev.every((f, i) => f === inputFlows[i])
      if (!same) {
        const jNode = nodeMap.get(n.id) ?? n
        nodeMap.set(n.id, { ...jNode, data: { ...jNode.data, inputFlows } })
        nodesChanged = true
      }
    }

    // Phase 5: Sync exhaust nodes — their display flow comes from the incoming edge
    for (const n of nodes.filter(n => n.type === 'exhaust')) {
      const inEdge   = [...edgeMap.values()].find(e => e.target === n.id)
      const flowRate = inEdge ? ((inEdge.data as Record<string, unknown>)?.flow_rate as number ?? 1.0) : 0
      const cur      = nodeMap.get(n.id) ?? n
      if ((cur.data as Record<string, unknown>).volume_flow_rate !== flowRate) {
        nodeMap.set(n.id, { ...cur, data: { ...cur.data, volume_flow_rate: flowRate } })
        nodesChanged = true
      }
    }

    if (nodesChanged) setNodes(Array.from(nodeMap.values()))
    if (edgesChanged) setEdges(Array.from(edgeMap.values()))
  }, [nodes, edges, setNodes, setEdges])

  // Persist topology and sim params across page refreshes
  useEffect(() => {
    localStorage.setItem('hvac_nodes', JSON.stringify(nodes))
    localStorage.setItem('hvac_edges', JSON.stringify(edges))
  }, [nodes, edges])

  useEffect(() => {
    localStorage.setItem('hvac_simParams', JSON.stringify(simParams))
  }, [simParams])

  const handleExport = useCallback(() => {
    const blob = new Blob(
      [JSON.stringify({ nodes, edges, simParams }, null, 2)],
      { type: 'application/json' },
    )
    const url = URL.createObjectURL(blob)
    const a   = document.createElement('a')
    a.href    = url
    a.download = 'hvac_config.json'
    a.click()
    URL.revokeObjectURL(url)
  }, [nodes, edges, simParams])

  const handleImport = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = evt => {
      try {
        const data = JSON.parse(evt.target!.result as string)
        if (data.nodes)     setNodes(data.nodes)
        if (data.edges)     setEdges(data.edges)
        if (data.simParams) setSimParams(data.simParams)
      } catch { /* ignore malformed files */ }
    }
    reader.readAsText(file)
    e.target.value = ''
  }, [setNodes, setEdges, setSimParams])

  const handleSaveDefault = useCallback(() => {
    localStorage.setItem('hvac_default_nodes',     JSON.stringify(nodes))
    localStorage.setItem('hvac_default_edges',     JSON.stringify(edges))
    localStorage.setItem('hvac_default_simParams', JSON.stringify(simParams))
  }, [nodes, edges, simParams])

  const handleReset = useCallback(() => {
    if (!window.confirm('Reset to saved default layout?')) return
    setNodes(loadFromStorage('hvac_default_nodes',     defaultNodes))
    setEdges(loadFromStorage('hvac_default_edges',     defaultEdges))
    setSimParams(loadFromStorage('hvac_default_simParams', defaultSimParams))
  }, [setNodes, setEdges, setSimParams])

  const handleExportGraph = useCallback(async () => {
    if (!rfInstance.current || !rfWrapperRef.current) return

    rfInstance.current.fitView({ padding: 0.08, duration: 0 })
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))

    const DPI    = 400
    const SCREEN = 96
    const scale  = DPI / SCREEN
    const bg     = theme === 'light' ? '#f1f5f9' : '#0f172a'

    // Inline SVG stroke/fill from computed styles so html-to-image captures edge lines.
    // ReactFlow edges are styled via external CSS which doesn't survive the DOM clone.
    const SVG_PROPS = ['stroke', 'stroke-width', 'stroke-dasharray', 'stroke-linecap', 'fill', 'opacity'] as const
    const svgEls = Array.from(rfWrapperRef.current.querySelectorAll<SVGElement>('svg path, svg polyline, svg line'))
    const savedAttrs = svgEls.map(el => el.getAttribute('style'))
    svgEls.forEach(el => {
      const cs  = getComputedStyle(el)
      const extra = SVG_PROPS.map(p => `${p}:${cs.getPropertyValue(p)}`).join(';')
      el.setAttribute('style', (el.getAttribute('style') ?? '') + ';' + extra)
    })

    const opts = {
      pixelRatio:      scale,
      backgroundColor: bg,
      filter: (el: Element) => {
        const cls = (el as HTMLElement).classList
        return !cls?.contains('react-flow__controls')  &&
               !cls?.contains('react-flow__minimap')   &&
               !cls?.contains('react-flow__panel')     &&
               !cls?.contains('react-flow__background')
      },
    }
    await toPng(rfWrapperRef.current, opts)          // warm-up: inlines fonts
    const dataUrl = await toPng(rfWrapperRef.current, opts)

    // Restore original SVG styles
    svgEls.forEach((el, i) => {
      const s = savedAttrs[i]
      if (s === null) el.removeAttribute('style')
      else el.setAttribute('style', s)
    })

    // Load into a canvas for auto-cropping
    const img = await new Promise<HTMLImageElement>(resolve => {
      const i = new Image()
      i.onload = () => resolve(i)
      i.src = dataUrl
    })

    const raw = document.createElement('canvas')
    raw.width  = img.width
    raw.height = img.height
    const ctx = raw.getContext('2d')!
    ctx.drawImage(img, 0, 0)

    // Parse background RGB
    const tmp = document.createElement('canvas').getContext('2d')!
    tmp.fillStyle = bg
    tmp.fillRect(0, 0, 1, 1)
    const [bgR, bgG, bgB] = tmp.getImageData(0, 0, 1, 1).data
    const d = ctx.getImageData(0, 0, raw.width, raw.height).data
    const W = raw.width, H = raw.height

    const isBg = (i: number) =>
      Math.abs(d[i]   - bgR) < 6 &&
      Math.abs(d[i+1] - bgG) < 6 &&
      Math.abs(d[i+2] - bgB) < 6

    let top = 0, bottom = H - 1, left = 0, right = W - 1
    outer: for (let y = 0; y < H; y++) { for (let x = 0; x < W; x++) { if (!isBg((y*W+x)*4)) { top = y; break outer } } }
    outer: for (let y = H-1; y >= 0; y--) { for (let x = 0; x < W; x++) { if (!isBg((y*W+x)*4)) { bottom = y; break outer } } }
    outer: for (let x = 0; x < W; x++) { for (let y = 0; y < H; y++) { if (!isBg((y*W+x)*4)) { left = x; break outer } } }
    outer: for (let x = W-1; x >= 0; x--) { for (let y = 0; y < H; y++) { if (!isBg((y*W+x)*4)) { right = x; break outer } } }

    const PAD = Math.round(scale * 20)
    top    = Math.max(0,   top    - PAD)
    bottom = Math.min(H-1, bottom + PAD)
    left   = Math.max(0,   left   - PAD)
    right  = Math.min(W-1, right  + PAD)

    const cropped = document.createElement('canvas')
    cropped.width  = right - left + 1
    cropped.height = bottom - top + 1
    cropped.getContext('2d')!.drawImage(raw, left, top, cropped.width, cropped.height, 0, 0, cropped.width, cropped.height)

    cropped.toBlob(blob => {
      if (!blob) return
      const url = URL.createObjectURL(blob)
      const a   = document.createElement('a')
      a.href     = url
      a.download = 'hvac_layout_400dpi.png'
      a.click()
      URL.revokeObjectURL(url)
    }, 'image/png')
  }, [theme])

  const handleSimulate = async () => {
    // Validate topology first
    const w = validateGraph(nodes, edges)
    setWarnings(w)
    if (w.some(x => x.severity === 'error')) {
      setShowWarnings(true)
      return   // block on errors; warnings are shown but don't block
    }
    if (w.length > 0) setShowWarnings(true)

    abortController.current = new AbortController()
    setIsLoading(true)
    setError(null)
    try {
      const results = await runSimulation(nodes, edges, simParams, abortController.current.signal)
      setSimResults(results)
      setShowResults(true)
    } catch (err: unknown) {
      if (axios.isCancel(err)) return   // user pressed Stop — no error banner
      const msg = err instanceof Error ? err.message : String(err)
      setError(msg)
    } finally {
      setIsLoading(false)
      abortController.current = null
    }
  }

  const handleStop = () => {
    abortController.current?.abort()
  }

  return (
    <div className="app-root" data-theme={theme}>
      <header className="app-header">
        <span className="app-title">HVAC Commissioning Tool</span>
        {error && <span className="error-banner">{error}</span>}
        {isLoading && (
          <>
            <span className="loading-badge">Simulating…</span>
            <button className="stop-btn" onClick={handleStop}>■ Stop</button>
          </>
        )}
        <div className="header-actions">
          <button className="theme-toggle-btn" onClick={toggleTheme} title="Toggle light/dark mode">
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
          <button
            className={`header-btn${snapOn ? ' header-btn-active' : ''}`}
            onClick={() => setSnapOn(s => !s)}
            title="Toggle snap-to-grid (15 px)"
          >Snap to grid</button>
          <button className="header-btn" onClick={handleExportGraph} title="Export node graph as 400 DPI PNG">Export graph</button>
          <button className="header-btn" onClick={handleExport}      title="Export layout as JSON">↓ Export</button>
          <button className="header-btn" onClick={() => fileInputRef.current?.click()} title="Import layout from JSON">↑ Import</button>
          <button className="header-btn" onClick={handleSaveDefault} title="Save current layout as default">★ Set default</button>
          <button className="header-btn header-btn-danger" onClick={handleReset} title="Reset to saved default">↺ Reset</button>
          <input ref={fileInputRef} type="file" accept=".json" style={{ display: 'none' }} onChange={handleImport} />
        </div>
      </header>

      {showWarnings && warnings.length > 0 && (
        <div className="warning-panel">
          <div className="warning-panel-header">
            <span>Graph issues ({warnings.filter(w => w.severity === 'error').length} error{warnings.filter(w => w.severity === 'error').length !== 1 ? 's' : ''}, {warnings.filter(w => w.severity === 'warning').length} warning{warnings.filter(w => w.severity === 'warning').length !== 1 ? 's' : ''})</span>
            <button className="close-btn" onClick={() => setShowWarnings(false)}>✕</button>
          </div>
          <ul className="warning-list">
            {warnings.map((w, i) => (
              <li key={i} className={`warning-item warning-${w.severity}`}>
                {w.severity === 'error' ? '✖' : '⚠'} {w.message}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="main-layout">
        <NodePalette
          simParams={simParams}
          onSimParamsChange={setSimParams}
          onSimulate={handleSimulate}
          isLoading={isLoading}
        />

        <div
          className="canvas-wrapper"
          ref={rfWrapperRef}
          onDrop={onDrop}
          onDragOver={onDragOver}
        >
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onEdgeClick={onEdgeClick}
            onPaneClick={onPaneClick}
            onInit={inst => { rfInstance.current = inst }}
            onNodeDrag={onNodeDrag}
            onNodeDragStop={onNodeDragStop}
            fitView
            deleteKeyCode="Delete"
            snapToGrid={snapOn}
            snapGrid={[15, 15]}
            defaultEdgeOptions={{ type: 'smoothstep', animated: true }}
          >
            <Background color="#334155" gap={20} />
            <Controls />
          </ReactFlow>

          {/* Center-alignment guide lines */}
          {guides && guides.x.map((fx, i) => (
            <div key={`gx${i}`} style={{
              position: 'absolute', top: 0, bottom: 0,
              left: fx * guides.vp.zoom + guides.vp.x, width: 1,
              background: '#6366f1', opacity: 0.8,
              pointerEvents: 'none', zIndex: 10,
            }} />
          ))}
          {guides && guides.y.map((fy, i) => (
            <div key={`gy${i}`} style={{
              position: 'absolute', left: 0, right: 0,
              top: fy * guides.vp.zoom + guides.vp.y, height: 1,
              background: '#6366f1', opacity: 0.8,
              pointerEvents: 'none', zIndex: 10,
            }} />
          ))}

          {/* Distance indicators */}
          {guides && guides.dists.map((d, i) => {
            const isH = d.y1 === d.y2
            const cx  = (d.x1 + d.x2) / 2
            const cy  = (d.y1 + d.y2) / 2
            return (
              <div key={`d${i}`} style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none', zIndex: 11 }}>
                {/* Line */}
                <div style={{
                  position: 'absolute',
                  left: isH ? Math.min(d.x1, d.x2) : d.x1 - 0.5,
                  top:  isH ? d.y1 - 0.5           : Math.min(d.y1, d.y2),
                  width:  isH ? Math.abs(d.x2 - d.x1) : 1,
                  height: isH ? 1 : Math.abs(d.y2 - d.y1),
                  background: '#f59e0b',
                }} />
                {/* Label */}
                <div style={{
                  position: 'absolute',
                  left: cx, top: cy,
                  transform: 'translate(-50%, -50%)',
                  background: '#f59e0b', color: '#000',
                  fontSize: 10, fontWeight: 700,
                  padding: '1px 5px', borderRadius: 3,
                  whiteSpace: 'nowrap',
                }}>
                  {d.gap}
                </div>
              </div>
            )
          })}
        </div>

        {rightOpen ? (
          <PropertiesPanel
            selectedNode={selectedNode}
            selectedEdge={selectedEdge}
            onUpdateNode={updateNodeData}
            onUpdateEdge={updateEdgeData}
            onCollapse={() => setRightOpen(false)}
          />
        ) : (
          <button className="sidebar-collapsed-tab" onClick={() => setRightOpen(true)} title="Show properties">
            ‹ Properties
          </button>
        )}
      </div>

      {showResults && simResults && (
        <ResultsPanel results={simResults} onClose={() => setShowResults(false)} theme={theme} />
      )}
    </div>
  )
}

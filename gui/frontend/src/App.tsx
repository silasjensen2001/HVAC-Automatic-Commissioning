import { useState, useCallback, useRef, useEffect } from 'react'
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
    <div className="app-root">
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
            fitView
            deleteKeyCode="Delete"
            defaultEdgeOptions={{ type: 'smoothstep', animated: true }}
          >
            <Background color="#334155" gap={20} />
            <Controls />
          </ReactFlow>
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
        <ResultsPanel results={simResults} onClose={() => setShowResults(false)} />
      )}
    </div>
  )
}

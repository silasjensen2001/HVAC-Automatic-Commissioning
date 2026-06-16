import { memo, useEffect } from 'react'
import { Handle, Position, type NodeProps, useUpdateNodeInternals } from 'reactflow'
import type { JunctionData } from '../../types'

function handlePositions(count: number): number[] {
  return Array.from({ length: count }, (_, i) => ((i + 1) / (count + 1)) * 100)
}

function JunctionNode({ id, data, selected }: NodeProps<JunctionData>) {
  const updateNodeInternals = useUpdateNodeInternals()
  const numIn  = Math.max(1, data.num_inputs  ?? 2)
  const numOut = Math.max(1, data.num_outputs ?? 1)
  const inputFlows  = data.inputFlows  ?? []
  const outputFlow  = data.outputFlow  ?? null
  const exhaustFlow = data.exhaustFlow ?? null

  useEffect(() => { updateNodeInternals(id) }, [numIn, numOut, id, updateNodeInternals])

  const inPositions  = handlePositions(numIn)
  const outPositions = handlePositions(numOut)
  const bodyHeight   = Math.max(numIn, numOut) * 22 + 8

  return (
    <div className={`hx-node junction-node${selected ? ' node-selected' : ''}`}>
      {inPositions.map((pct, i) => (
        <Handle
          key={`input-${i}`}
          id={`input-${i}`}
          type="target"
          position={Position.Left}
          style={{ top: `${pct}%` }}
        />
      ))}

      <div className="node-inner">
        <div className="node-header" style={{ background: '#8b5cf6' }}>
          <span className="node-icon">⊕</span>
          <span className="node-label">{data.label}</span>
        </div>
        <div className="node-body" style={{ minHeight: bodyHeight }}>
          {Array.from({ length: numIn }, (_, i) => {
            const flow = inputFlows[i]
            return (
              <div key={i} className="node-stat" style={{ fontSize: 9 }}>
                <span style={{ color: '#475569' }}>in {i + 1}: </span>
                <strong style={{ color: flow != null ? '#93c5fd' : '#334155' }}>
                  {flow != null ? `${(flow * 3600).toFixed(0)} m³/h` : '—'}
                </strong>
              </div>
            )
          })}
          {outputFlow != null && (
            <div className="node-stat" style={{ fontSize: 9, marginTop: 3, borderTop: '1px solid #334155', paddingTop: 3 }}>
              <span style={{ color: '#475569' }}>out: </span>
              <strong style={{ color: '#6ee7b7' }}>{(outputFlow * 3600).toFixed(0)} m³/h</strong>
            </div>
          )}
          {outputFlow != null && exhaustFlow != null && outputFlow > 0 && (
            <div className="node-stat" style={{ fontSize: 9, marginTop: 3, borderTop: '1px solid #334155', paddingTop: 3 }}>
              <span style={{ color: '#475569' }}>♻ recirc: </span>
              <strong style={{ color: '#6ee7b7' }}>
                {(((outputFlow - exhaustFlow) / outputFlow) * 100).toFixed(1)}%
              </strong>
            </div>
          )}
        </div>
      </div>


      {outPositions.map((pct, i) => (
        <Handle
          key={`output-${i}`}
          id={`output-${i}`}
          type="source"
          position={Position.Right}
          style={{ top: `${pct}%` }}
        />
      ))}
    </div>
  )
}

export default memo(JunctionNode)

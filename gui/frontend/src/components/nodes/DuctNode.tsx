import { memo, useEffect } from 'react'
import { Handle, Position, type NodeProps, useUpdateNodeInternals } from 'reactflow'
import type { DuctData } from '../../types'

function DuctNode({ id, data, selected }: NodeProps<DuctData>) {
  const updateNodeInternals = useUpdateNodeInternals()
  const flip = data.flipHandles ?? false

  useEffect(() => { updateNodeInternals(id) }, [flip, id, updateNodeInternals])

  const inputPos  = flip ? Position.Right : Position.Left
  const outputPos = flip ? Position.Left  : Position.Right
  return (
    <div className={`hx-node duct-node${selected ? ' node-selected' : ''}`}>
      <Handle type="target" position={inputPos} id="input" />
      <div className="node-inner">
        <div className="node-header" style={{ background: '#6b7280' }}>
          <span className="node-icon">▶</span>
          <span className="node-label">{data.label}</span>
        </div>
        <div className="node-body">
          <div className="node-stat">Flow {(data.volume_flow_rate * 3600).toFixed(0)} m³/h</div>
          <div className="node-stat">L = {data.duct_length.toFixed(1)} m</div>
        </div>
      </div>
      <span className="io-in">{flip ? 'OUT' : 'IN'}</span>
      <span className="io-out">{flip ? 'IN' : 'OUT'}</span>
      <Handle type="source" position={outputPos} id="output" />
    </div>
  )
}

export default memo(DuctNode)

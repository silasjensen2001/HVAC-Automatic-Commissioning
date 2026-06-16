import { memo, useEffect } from 'react'
import { Handle, Position, type NodeProps, useUpdateNodeInternals } from 'reactflow'
import type { FanData } from '../../types'

function FanNode({ id, data, selected }: NodeProps<FanData>) {
  const updateNodeInternals = useUpdateNodeInternals()
  const flip = data.flipHandles ?? false

  useEffect(() => { updateNodeInternals(id) }, [flip, id, updateNodeInternals])

  const inputPos  = flip ? Position.Right : Position.Left
  const outputPos = flip ? Position.Left  : Position.Right
  return (
    <div className={`hx-node fan-node${selected ? ' node-selected' : ''}`}>
      <Handle type="target" position={inputPos} id="input" />
      <div className="node-inner">
        <div className="node-header" style={{ background: '#0d9488' }}>
          <span className="node-icon">↻</span>
          <span className="node-label">{data.label}</span>
        </div>
        <div className="node-body">
          <div className="node-stat"><strong>{(data.volume_flow_rate * 3600).toFixed(0)}</strong> m³/h</div>
          <div className="node-stat" style={{ color: '#64748b' }}>{data.volume_flow_rate.toFixed(4)} m³/s</div>
        </div>
      </div>
      <span className="io-in">{flip ? 'OUT' : 'IN'}</span>
      <span className="io-out">{flip ? 'IN' : 'OUT'}</span>
      <Handle type="source" position={outputPos} id="output" />
    </div>
  )
}

export default memo(FanNode)

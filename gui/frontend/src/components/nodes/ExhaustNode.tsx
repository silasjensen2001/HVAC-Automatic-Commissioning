import { memo } from 'react'
import { Handle, Position, type NodeProps } from 'reactflow'
import type { ExhaustData } from '../../types'

function ExhaustNode({ data, selected }: NodeProps<ExhaustData>) {
  return (
    <div className={`hx-node exhaust-node${selected ? ' node-selected' : ''}`}>
      <Handle type="target" position={Position.Left} id="input" />
      <div className="node-inner">
        <div className="node-header" style={{ background: '#92400e' }}>
          <span className="node-icon">↑</span>
          <span className="node-label">{data.label}</span>
        </div>
        <div className="node-body">
          <div className="node-stat">
            <strong>{(data.volume_flow_rate * 3600).toFixed(0)}</strong> m³/h
          </div>
          <div className="node-stat" style={{ color: '#64748b' }}>to outside</div>
        </div>
      </div>
      <span className="io-in">IN</span>
    </div>
  )
}

export default memo(ExhaustNode)

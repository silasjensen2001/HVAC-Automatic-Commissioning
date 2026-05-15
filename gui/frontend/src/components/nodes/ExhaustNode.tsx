import { memo, useEffect } from 'react'
import { Handle, Position, type NodeProps, useUpdateNodeInternals } from 'reactflow'
import type { ExhaustData } from '../../types'

function ExhaustNode({ id, data, selected }: NodeProps<ExhaustData>) {
  const updateNodeInternals = useUpdateNodeInternals()
  const inputPos = data.flipHandles ? Position.Right : Position.Left

  useEffect(() => { updateNodeInternals(id) }, [data.flipHandles, id, updateNodeInternals])

  return (
    <div className={`hx-node exhaust-node${selected ? ' node-selected' : ''}`}>
      <Handle type="target" position={inputPos} id="input" />
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
    </div>
  )
}

export default memo(ExhaustNode)

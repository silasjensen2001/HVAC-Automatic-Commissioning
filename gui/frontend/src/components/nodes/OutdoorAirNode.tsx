import { memo, useEffect } from 'react'
import { Handle, Position, type NodeProps, useUpdateNodeInternals } from 'reactflow'
import type { OutdoorAirData } from '../../types'

function OutdoorAirNode({ id, data, selected }: NodeProps<OutdoorAirData>) {
  const updateNodeInternals = useUpdateNodeInternals()
  const flip   = data.flipHandles ?? false
  const isSin  = data.disturbance_type === 'sinusoidal'
  const outPos = flip ? Position.Left : Position.Right

  useEffect(() => { updateNodeInternals(id) }, [flip, id, updateNodeInternals])

  return (
    <div className={`hx-node outdoor-node${selected ? ' node-selected' : ''}`}>
      <div className="node-inner">
        <div className="node-header" style={{ background: '#0369a1' }}>
          <span className="node-icon">☁</span>
          <span className="node-label">{data.label}</span>
        </div>
        <div className="node-body">
          <div className="node-stat">
            {isSin ? 'Mean' : 'Temp'} <strong>{data.T_fresh.toFixed(1)} °C</strong>
          </div>
          {isSin && (
            <>
              <div className="node-stat">±{data.disturbance_amplitude} °C</div>
              <div className="node-stat">{(data.disturbance_period / 3600).toFixed(1)} h period</div>
            </>
          )}
          {!isSin && <div className="node-stat" style={{ color: '#64748b' }}>Constant</div>}
        </div>
      </div>
      {/* Label positioned on whichever side the output handle is */}
      <span className={flip ? 'io-in' : 'io-out'}>OUT</span>
      <Handle type="source" position={outPos} id="output" />
    </div>
  )
}

export default memo(OutdoorAirNode)

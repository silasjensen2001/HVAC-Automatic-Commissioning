import type { SimParams } from '../types'

interface Props {
  simParams: SimParams
  onSimParamsChange: (p: SimParams) => void
  onSimulate: () => void
  isLoading: boolean
}

const PALETTE_ITEMS = [
  { type: 'outdoor_air', label: 'Outdoor Air', icon: '☁', color: '#0369a1' },
  { type: 'fan',         label: 'Fan / Pump',  icon: '↻', color: '#0d9488' },
  { type: 'cooler',      label: 'Cooler',      icon: '❄', color: '#3b82f6' },
  { type: 'heater',      label: 'Heater',      icon: '♨', color: '#ef4444' },
  { type: 'airduct',     label: 'Air Duct',    icon: '▶', color: '#6b7280' },
  { type: 'junction',    label: 'Junction',    icon: '⊕', color: '#8b5cf6' },
  { type: 'exhaust',     label: 'Exhaust',     icon: '↑', color: '#92400e' },
]

export default function NodePalette({ simParams, onSimParamsChange, onSimulate, isLoading }: Props) {
  const set = (key: keyof SimParams, value: unknown) =>
    onSimParamsChange({ ...simParams, [key]: value })

  const onDragStart = (e: React.DragEvent, nodeType: string) => {
    e.dataTransfer.setData('application/reactflow', nodeType)
    e.dataTransfer.effectAllowed = 'move'
  }

  return (
    <aside className="sidebar left-sidebar">
      <div className="sidebar-section">
        <h3 className="sidebar-title">Add Nodes</h3>
        <p className="sidebar-hint">Drag onto canvas</p>
        {PALETTE_ITEMS.map(item => (
          <div
            key={item.type}
            className="palette-item"
            draggable
            onDragStart={e => onDragStart(e, item.type)}
            style={{ borderLeftColor: item.color }}
          >
            <span style={{ color: item.color, marginRight: 8 }}>{item.icon}</span>
            {item.label}
          </div>
        ))}
      </div>

      <div className="sidebar-section">
        <h3 className="sidebar-title">Simulation</h3>

        <label className="field-label">
          Duration (s)
          <input
            type="text"
            inputMode="decimal"
            className="field-input"
            defaultValue={simParams.t_end}
            onChange={e => set('t_end', parseFloat(e.target.value.replace(',', '.')))}
          />
        </label>

        <label className="field-label">
          Q scale
          <input
            type="text"
            inputMode="decimal"
            className="field-input"
            defaultValue={simParams.Q_scale}
            onChange={e => set('Q_scale', parseFloat(e.target.value.replace(',', '.')))}
          />
        </label>

        <label className="field-label">
          R scale
          <input
            type="text"
            inputMode="decimal"
            className="field-input"
            defaultValue={simParams.R_scale}
            onChange={e => set('R_scale', parseFloat(e.target.value.replace(',', '.')))}
          />
        </label>

        <label className="field-label">
          Model
          <select
            className="field-input"
            value={simParams.model_mode}
            onChange={e => set('model_mode', e.target.value)}
          >
            <option value="nonlinear">Nonlinear</option>
            <option value="linear">Linear</option>
          </select>
        </label>

        <label className="field-label">
          Controller
          <select
            className="field-input"
            value={simParams.controller_type}
            onChange={e => set('controller_type', e.target.value)}
          >
            <option value="lqr">LQR (standard)</option>
            <option value="lmi">LMI / H∞ (disturbance rejection)</option>
          </select>
        </label>
      </div>

      <button
        className="simulate-btn"
        onClick={onSimulate}
        disabled={isLoading}
      >
        {isLoading ? (
          <><span className="spinner" /> Running…</>
        ) : (
          '▶  Simulate'
        )}
      </button>
    </aside>
  )
}

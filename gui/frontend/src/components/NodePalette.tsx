import { useState, useEffect } from 'react'
import type { SimParams, SystemInfo } from '../types'

interface Props {
  simParams: SimParams
  onSimParamsChange: (p: SimParams) => void
  onSimulate: (extraParams?: Partial<SimParams>) => void
  isLoading: boolean
  systemInfo: SystemInfo | null
  onFetchSystemInfo: () => void
  isFetchingInfo: boolean
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

function formatVec(v?: number[]): string {
  if (!v || v.length === 0) return ''
  return v.map(x => x.toExponential(2)).join(' ')
}

function parseVec(s: string): number[] | undefined {
  const vals = s.trim().split(/[\s,]+/).map(Number).filter(v => !isNaN(v))
  return vals.length > 0 ? vals : undefined
}

export default function NodePalette({
  simParams, onSimParamsChange, onSimulate, isLoading,
  systemInfo, onFetchSystemInfo, isFetchingInfo,
}: Props) {
  const set = (key: keyof SimParams, value: unknown) =>
    onSimParamsChange({ ...simParams, [key]: value })

  const onDragStart = (e: React.DragEvent, nodeType: string) => {
    e.dataTransfer.setData('application/reactflow', nodeType)
    e.dataTransfer.effectAllowed = 'move'
  }

  const [advOpen,   setAdvOpen]   = useState(false)
  const [useAdv,    setUseAdv]    = useState(simParams.use_advanced_qr ?? false)

  const DEFAULT_Q_LQR = '2.50e-1 2.50e-1 2.50e-1 2.50e-1 2.50e-1 1.00e-4 1.00e-4 1.00e-4 1.00e-4 1.00e-4 2.50e-1 2.50e-1 2.50e-1 2.50e-1 2.50e-1 1.00e-4 1.00e-4 1.00e-4 1.00e-4 1.00e-4 1.60e-1 1.60e-1'
  const DEFAULT_R_LQR = '5.00e-1 5.00e-1'
  const DEFAULT_Q_DR  = '2.50e-1 2.50e-1 2.50e-1 2.50e-1 2.50e-1 1.00e-4 1.00e-4 1.00e-4 1.00e-4 1.00e-4 2.50e-1 2.50e-1 2.50e-1 2.50e-1 2.50e-1 1.00e-4 1.00e-4 1.00e-4 1.00e-4 1.00e-4 8.00e-2 8.00e-2'
  const DEFAULT_R_DR  = '2.00e+0 2.00e+0'

  // Local string state for the four textareas
  const [qLqrText, setQLqrText] = useState(() => formatVec(simParams.Q_diag_lqr) || DEFAULT_Q_LQR)
  const [rLqrText, setRLqrText] = useState(() => formatVec(simParams.R_diag_lqr) || DEFAULT_R_LQR)
  const [qDrText,  setQDrText]  = useState(() => formatVec(simParams.Q_diag_dr)  || DEFAULT_Q_DR)
  const [rDrText,  setRDrText]  = useState(() => formatVec(simParams.R_diag_dr)  || DEFAULT_R_DR)

  // Only populate textareas from system info if they are truly empty
  useEffect(() => {
    if (!systemInfo) return
    if (!qLqrText.trim()) setQLqrText(formatVec(systemInfo.q_diag_default))
    if (!rLqrText.trim()) setRLqrText(formatVec(systemInfo.r_diag_default))
    if (!qDrText.trim())  setQDrText(formatVec(systemInfo.q_diag_default))
    if (!rDrText.trim())  setRDrText(formatVec(systemInfo.r_diag_default))
  }, [systemInfo]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleSimulateClick = () => {
    onSimulate({
      use_advanced_qr: useAdv,
      Q_diag_lqr: parseVec(qLqrText),
      R_diag_lqr: parseVec(rLqrText),
      Q_diag_dr:  parseVec(qDrText),
      R_diag_dr:  parseVec(rDrText),
    })
  }

  const lqrActive = simParams.controller_type === 'lqr'
  const drActive  = simParams.controller_type === 'lmi'

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

      {/* Advanced Q/R Configuration */}
      <div className="sidebar-section">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div
            style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', userSelect: 'none', flex: 1 }}
            onClick={() => setAdvOpen(v => !v)}
          >
            <span style={{ fontSize: 11, opacity: 0.7 }}>{advOpen ? '▼' : '▶'}</span>
            <h3 className="sidebar-title" style={{ margin: 0 }}>Advanced Q/R</h3>
          </div>
          <button
            onClick={() => setUseAdv(v => !v)}
            style={{
              fontSize: 10,
              padding: '2px 8px',
              borderRadius: 4,
              border: `1px solid ${useAdv ? 'var(--accent)' : 'var(--border)'}`,
              background: useAdv ? 'var(--accent)' : 'transparent',
              color: useAdv ? '#fff' : 'var(--fg)',
              cursor: 'pointer',
              fontWeight: 600,
              whiteSpace: 'nowrap',
            }}
          >
            {useAdv ? 'ON' : 'OFF'}
          </button>
        </div>

        {advOpen && (
          <div style={{ marginTop: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
              <button
                className="header-btn"
                onClick={onFetchSystemInfo}
                disabled={isFetchingInfo}
                style={{ flex: 1, fontSize: 11, padding: '4px 8px' }}
              >
                {isFetchingInfo ? 'Fetching…' : 'Get dimensions'}
              </button>
              {systemInfo && (
                <span style={{ fontSize: 10, opacity: 0.65 }}>
                  Q:{systemInfo.q_size}×{systemInfo.q_size}&nbsp;R:{systemInfo.r_size}×{systemInfo.r_size}
                </span>
              )}
            </div>

            {systemInfo && (
              <p style={{ fontSize: 10, opacity: 0.55, marginBottom: 8, lineHeight: 1.4 }}>
                {systemInfo.n_states} states · {systemInfo.n_outputs} outputs · {systemInfo.n_inputs} inputs
              </p>
            )}

            {/* LQR section */}
            <div style={{
              border: `1px solid ${lqrActive ? 'var(--accent)' : 'var(--border)'}`,
              borderRadius: 6,
              padding: 8,
              marginBottom: 8,
              opacity: lqrActive ? 1 : 0.55,
            }}>
              <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
                LQR
                {systemInfo && <span style={{ fontWeight: 400, opacity: 0.6 }}>Q diagonal ({systemInfo.q_size})</span>}
              </div>
              <textarea
                className="field-input"
                rows={4}
                style={{ fontFamily: 'monospace', fontSize: 11, resize: 'vertical', width: '100%' }}
                placeholder={systemInfo ? `${systemInfo.q_size} space-separated values` : 'Click "Get dimensions" first'}
                value={qLqrText}
                onChange={e => setQLqrText(e.target.value)}
              />
              <div style={{ fontSize: 11, fontWeight: 600, margin: '6px 0', display: 'flex', alignItems: 'center', gap: 6 }}>
                R diagonal
                {systemInfo && <span style={{ fontWeight: 400, opacity: 0.6 }}>({systemInfo.r_size})</span>}
              </div>
              <input
                type="text"
                className="field-input"
                style={{ fontFamily: 'monospace', fontSize: 11 }}
                placeholder={systemInfo ? `${systemInfo.r_size} values` : ''}
                value={rLqrText}
                onChange={e => setRLqrText(e.target.value)}
              />
            </div>

            {/* Disturbance Rejection section */}
            <div style={{
              border: `1px solid ${drActive ? 'var(--accent)' : 'var(--border)'}`,
              borderRadius: 6,
              padding: 8,
              opacity: drActive ? 1 : 0.55,
            }}>
              <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
                Disturbance Rejection (LMI)
                {systemInfo && <span style={{ fontWeight: 400, opacity: 0.6 }}>Q diagonal ({systemInfo.q_size})</span>}
              </div>
              <textarea
                className="field-input"
                rows={4}
                style={{ fontFamily: 'monospace', fontSize: 11, resize: 'vertical', width: '100%' }}
                placeholder={systemInfo ? `${systemInfo.q_size} space-separated values` : 'Click "Get dimensions" first'}
                value={qDrText}
                onChange={e => setQDrText(e.target.value)}
              />
              <div style={{ fontSize: 11, fontWeight: 600, margin: '6px 0', display: 'flex', alignItems: 'center', gap: 6 }}>
                R diagonal
                {systemInfo && <span style={{ fontWeight: 400, opacity: 0.6 }}>({systemInfo.r_size})</span>}
              </div>
              <input
                type="text"
                className="field-input"
                style={{ fontFamily: 'monospace', fontSize: 11 }}
                placeholder={systemInfo ? `${systemInfo.r_size} values` : ''}
                value={rDrText}
                onChange={e => setRDrText(e.target.value)}
              />
            </div>
          </div>
        )}
      </div>

      <button
        className="simulate-btn"
        onClick={handleSimulateClick}
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

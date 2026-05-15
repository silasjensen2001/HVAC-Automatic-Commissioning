import type { Node, Edge } from 'reactflow'

interface Props {
  selectedNode: Node | null
  selectedEdge: Edge | null
  onUpdateNode: (id: string, updates: Record<string, unknown>) => void
  onUpdateEdge: (id: string, updates: Record<string, unknown>) => void
}

function FieldRow({
  label, value, onChange, step = 'any', min,
}: {
  label: string
  value: number | string
  onChange: (v: string) => void
  step?: string | number
  min?: number
}) {
  const isNum = typeof value === 'number'
  return (
    <label className="field-label">
      {label}
      <input
        type="text"
        inputMode={isNum ? 'decimal' : undefined}
        className="field-input"
        defaultValue={value}
        onChange={e => onChange(e.target.value)}
      />
    </label>
  )
}

function FlowRateField({ value, onChange, readOnly = false, fanName }: { value: number; onChange: (v: number) => void; readOnly?: boolean; fanName?: string }) {
  const parse = (v: string) => parseFloat(v.replace(',', '.'))
  return (
    <div>
      <label className="field-label">
        Flow rate (m³/s){readOnly && <span style={{ color: '#475569', fontStyle: 'italic' }}> — set by {fanName ?? 'fan'}</span>}
        <input
          type="text"
          inputMode="decimal"
          className="field-input"
          {...(readOnly
            ? { value: value.toFixed(5) }
            : { defaultValue: value.toFixed(5) })}
          readOnly={readOnly}
          style={readOnly ? { color: '#64748b', cursor: 'default' } : undefined}
          onChange={readOnly ? undefined : e => onChange(parse(e.target.value))}
        />
      </label>
      <label className="field-label">
        Flow rate (m³/h)
        <input
          type="text"
          inputMode="decimal"
          className="field-input"
          {...(readOnly
            ? { value: (value * 3600).toFixed(2) }
            : { defaultValue: (value * 3600).toFixed(2) })}
          readOnly={readOnly}
          style={readOnly ? { color: '#64748b', cursor: 'default' } : undefined}
          onChange={readOnly ? undefined : e => onChange(parse(e.target.value) / 3600)}
        />
      </label>
    </div>
  )
}

function FlipButton({ flipped, onToggle }: { flipped: boolean; onToggle: () => void }) {
  return (
    <button
      className={`flip-btn${flipped ? ' flip-btn-active' : ''}`}
      onClick={onToggle}
      title="Swap which side is input vs output"
    >
      ⇄ {flipped ? 'I/O flipped' : 'Flip I/O sides'}
    </button>
  )
}

function HxForm({ data, onChange }: { data: Record<string, unknown>; onChange: (k: string, v: unknown) => void }) {
  const num = (k: string) => (v: string) => onChange(k, parseFloat(v.replace(',', '.')))
  const int = (k: string) => (v: string) => onChange(k, parseInt(v.replace(',', '.'), 10))
  const str = (k: string) => (v: string) => onChange(k, v)

  return (
    <>
      <FlipButton
        flipped={!!(data.flipHandles)}
        onToggle={() => onChange('flipHandles', !(data.flipHandles ?? false))}
      />
      <FieldRow label="Label"             value={data.label as string}          onChange={str('label')} />
      <FieldRow label="Setpoint (°C)"     value={data.T_out_target as number}   onChange={num('T_out_target')}   step={0.5} />
      <FlowRateField
        value={data.volume_flow_rate as number}
        onChange={v => onChange('volume_flow_rate', v)}
        readOnly
        fanName={data.flowSource as string | undefined}
      />
      <FieldRow label="Water supply (°C)" value={data.water_supply_T as number} onChange={num('water_supply_T')} step={0.5} />

      <details className="advanced-section">
        <summary>Advanced geometry</summary>
        <FieldRow label="Segments"     value={data.num_segments as number}          onChange={int('num_segments')}         step={1} min={1} />
        <FieldRow label="Pipes"        value={data.num_pipes as number}             onChange={int('num_pipes')}            step={1} min={1} />
        <FieldRow label="γ (W/m²K)"   value={data.gamma as number}                onChange={num('gamma')}                step={10} />
        <FieldRow label="Cross area water (m²)" value={data.cross_area_water as number} onChange={num('cross_area_water')} step={0.0001} />
        <FieldRow label="HX depth (m)"  value={data.heat_exchanger_depth as number}  onChange={num('heat_exchanger_depth')}  step={0.01} />
        <FieldRow label="HX width (m)"  value={data.heat_exchanger_width as number}  onChange={num('heat_exchanger_width')}  step={0.05} />
        <FieldRow label="HX height (m)" value={data.heat_exchanger_height as number} onChange={num('heat_exchanger_height')} step={0.05} />
        <FieldRow label="Kvs (m³/h)"    value={data.Kvs as number}                   onChange={num('Kvs')}                   step={0.01} />
      </details>
    </>
  )
}

function DuctForm({ data, onChange }: { data: Record<string, unknown>; onChange: (k: string, v: unknown) => void }) {
  const num = (k: string) => (v: string) => onChange(k, parseFloat(v.replace(',', '.')))
  const int = (k: string) => (v: string) => onChange(k, parseInt(v.replace(',', '.'), 10))
  const str = (k: string) => (v: string) => onChange(k, v)
  return (
    <>
      <FlipButton
        flipped={!!(data.flipHandles)}
        onToggle={() => onChange('flipHandles', !(data.flipHandles ?? false))}
      />
      <FieldRow label="Label"           value={data.label as string}           onChange={str('label')} />
      <FlowRateField
        value={data.volume_flow_rate as number}
        onChange={v => onChange('volume_flow_rate', v)}
        readOnly
        fanName={data.flowSource as string | undefined}
      />
      <FieldRow label="Length (m)"      value={data.duct_length as number}     onChange={num('duct_length')}      step={0.5} />
      <FieldRow label="Cross area (m²)" value={data.cross_section_area as number} onChange={num('cross_section_area')} step={0.05} />
      <FieldRow label="Segments"        value={data.num_segments as number}    onChange={int('num_segments')}     step={1} min={1} />
    </>
  )
}

function JunctionForm({ data, onChange }: { data: Record<string, unknown>; onChange: (k: string, v: unknown) => void }) {
  const int = (k: string) => (v: string) => onChange(k, Math.max(1, parseInt(v.replace(',', '.'), 10)))
  return (
    <>
      <FieldRow label="Label"   value={data.label as string}               onChange={v => onChange('label', v)} />
      <FieldRow label="Inputs"  value={(data.num_inputs  as number) ?? 2}  onChange={int('num_inputs')}  step={1} min={1} />
      <FieldRow label="Outputs" value={(data.num_outputs as number) ?? 1}  onChange={int('num_outputs')} step={1} min={1} />
      <p className="sidebar-hint" style={{ marginTop: 4 }}>
        Set flow rate on each edge entering this node.
      </p>
    </>
  )
}

function OutdoorAirForm({ data, onChange }: { data: Record<string, unknown>; onChange: (k: string, v: unknown) => void }) {
  const num = (k: string) => (v: string) => onChange(k, parseFloat(v.replace(',', '.')))
  const isSin = data.disturbance_type === 'sinusoidal'
  return (
    <>
      <FlipButton
        flipped={!!(data.flipHandles)}
        onToggle={() => onChange('flipHandles', !(data.flipHandles ?? false))}
      />
      <FieldRow label="Label" value={data.label as string} onChange={v => onChange('label', v)} />

      <label className="field-label">
        {isSin ? 'Mean temp (°C)' : 'Temperature (°C)'}
        <input type="text" inputMode="decimal" className="field-input"
          defaultValue={data.T_fresh as number}
          onChange={e => onChange('T_fresh', parseFloat(e.target.value.replace(',', '.')))} />
      </label>

      <p className="sidebar-hint" style={{ marginBottom: 4 }}>Signal type</p>
      <div className="disturbance-toggle">
        {(['constant', 'sinusoidal'] as const).map(t => (
          <button
            key={t}
            className={`toggle-btn${data.disturbance_type === t ? ' toggle-active' : ''}`}
            onClick={() => onChange('disturbance_type', t)}
          >
            {t === 'constant' ? '— Constant' : '∿ Sinusoidal'}
          </button>
        ))}
      </div>

      {isSin && (
        <>
          <FieldRow label="Amplitude (°C peak)"
            value={data.disturbance_amplitude as number}
            onChange={num('disturbance_amplitude')} step={0.5} min={0} />
          <FieldRow label="Period (s)"
            value={data.disturbance_period as number}
            onChange={num('disturbance_period')} step={3600} min={1} />
          <p className="sidebar-hint">
            {((data.disturbance_period as number) / 3600).toFixed(1)} h · ±{data.disturbance_amplitude} °C
          </p>
        </>
      )}
    </>
  )
}

function FanForm({ data, onChange }: { data: Record<string, unknown>; onChange: (k: string, v: unknown) => void }) {
  const str = (k: string) => (v: string) => onChange(k, v)
  return (
    <>
      <FlipButton
        flipped={!!(data.flipHandles)}
        onToggle={() => onChange('flipHandles', !(data.flipHandles ?? false))}
      />
      <FieldRow label="Label" value={data.label as string} onChange={str('label')} />
      <FlowRateField
        value={data.volume_flow_rate as number}
        onChange={v => onChange('volume_flow_rate', v)}
      />
      <p className="sidebar-hint" style={{ marginTop: 4 }}>
        Fan flow rate propagates to downstream nodes during simulation.
      </p>
    </>
  )
}

export default function PropertiesPanel({ selectedNode, selectedEdge, onUpdateNode, onUpdateEdge }: Props) {
  if (!selectedNode && !selectedEdge) {
    return (
      <aside className="sidebar right-sidebar empty-panel">
        <p style={{ color: '#9ca3af', fontSize: 13, textAlign: 'center', marginTop: 32 }}>
          Click a node or edge<br />to edit its properties
        </p>
      </aside>
    )
  }

  if (selectedEdge && !selectedNode) {
    const edgeData = (selectedEdge.data ?? {}) as Record<string, unknown>
    return (
      <aside className="sidebar right-sidebar">
        <h3 className="sidebar-title">Edge</h3>
        <p className="sidebar-hint">{selectedEdge.source} → {selectedEdge.target}</p>
        <FlowRateField
          value={(edgeData.flow_rate as number) ?? 1.0}
          onChange={v => onUpdateEdge(selectedEdge.id, { flow_rate: v })}
        />
        <p className="sidebar-hint" style={{ marginTop: 8 }}>
          Flow rate is only used when this edge connects to a Junction node.
        </p>
      </aside>
    )
  }

  if (!selectedNode) return null

  const data = selectedNode.data as Record<string, unknown>
  const onChange = (k: string, v: unknown) => onUpdateNode(selectedNode.id, { [k]: v })

  const typeLabel: Record<string, string> = {
    cooler:      'Cooler',
    heater:      'Heater',
    airduct:     'Air Duct',
    junction:    'Junction',
    outdoor_air: 'Outdoor Air',
    fan:         'Fan / Pump',
    exhaust:     'Exhaust',
  }

  return (
    <aside key={selectedNode.id} className="sidebar right-sidebar">
      <h3 className="sidebar-title">{typeLabel[selectedNode.type ?? ''] ?? selectedNode.type}</h3>
      <p className="sidebar-hint">ID: {selectedNode.id}</p>
      {(selectedNode.type === 'cooler' || selectedNode.type === 'heater') && (
        <HxForm data={data} onChange={onChange} />
      )}
      {selectedNode.type === 'airduct' && (
        <DuctForm data={data} onChange={onChange} />
      )}
      {selectedNode.type === 'junction' && (
        <JunctionForm data={data} onChange={onChange} />
      )}
      {selectedNode.type === 'outdoor_air' && (
        <OutdoorAirForm data={data} onChange={onChange} />
      )}
      {selectedNode.type === 'fan' && (
        <FanForm data={data} onChange={onChange} />
      )}
      {selectedNode.type === 'exhaust' && (
        <>
          <FieldRow label="Label" value={data.label as string} onChange={v => onChange('label', v)} />
          <FlowRateField
            value={data.volume_flow_rate as number}
            onChange={() => {}}
            readOnly
            fanName="edge flow"
          />
          <p className="sidebar-hint" style={{ marginTop: 4 }}>
            Set the flow on the incoming edge to control the exhaust rate.
          </p>
        </>
      )}
    </aside>
  )
}

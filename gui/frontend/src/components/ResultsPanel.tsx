import { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'
import axios from 'axios'
import type { SimResults, SimMetrics } from '../types'

const DEFAULT_HEIGHT = 380

function downloadCsv(filename: string, headers: string[], rows: (number | string)[][]) {
  const escape = (v: number | string) =>
    typeof v === 'number' ? v.toPrecision(10) : `"${String(v).replace(/"/g, '""')}"`
  const csv = [headers.join(','), ...rows.map(r => r.map(escape).join(','))].join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

function downloadGains(metrics: SimMetrics, rowLabels: string[], colLabels: string[]) {
  const payload = {
    actuated_ids:  metrics.actuated_ids,
    row_labels:    rowLabels,
    col_labels:    colLabels,
    K_x:           metrics.K_x,
    K_I:           metrics.K_I,
    N:             metrics.N,
    M_anti_windup: metrics.M,
  }
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href     = url
  a.download = 'hvac_gains.json'
  a.click()
  URL.revokeObjectURL(url)
}

type ColorScheme = 'default' | 'viridis' | 'plasma' | 'coolwarm'

// 11 anchor points (t = 0.0 … 1.0 in steps of 0.1) sampled directly from
// matplotlib's colormap tables — keeps interpolation vivid and perceptually uniform.
const COLORMAPS: Record<Exclude<ColorScheme, 'default'>, string[]> = {
  viridis: [
    '#440154', '#482475', '#414487', '#355f8d', '#2a788e',
    '#21918c', '#22a884', '#44bf70', '#7ad151', '#bddf26', '#fde725',
  ],
  plasma: [
    '#0d0887', '#41049d', '#6a00a8', '#8f0da4', '#b12a90',
    '#cc4778', '#e16462', '#f2844b', '#fca636', '#fcce25', '#f0f921',
  ],
  coolwarm: [
    '#3b4cc0', '#5d7ce6', '#88a7f5', '#b9caf5', '#dddddd',
    '#f5c4b0', '#f5a27a', '#e8765c', '#cf4446', '#b40426', '#7f0000',
  ],
}

function hexToRgb(hex: string): [number, number, number] {
  return [parseInt(hex.slice(1,3),16), parseInt(hex.slice(3,5),16), parseInt(hex.slice(5,7),16)]
}
function rgbToHex(r: number, g: number, b: number): string {
  return '#' + [r,g,b].map(v => Math.round(v).toString(16).padStart(2,'0')).join('')
}
function sampleColormap(anchors: string[], t: number): string {
  const n = anchors.length - 1
  const i = Math.min(Math.floor(t * n), n - 1)
  const f = t * n - i
  const [r1,g1,b1] = hexToRgb(anchors[i])
  const [r2,g2,b2] = hexToRgb(anchors[i+1])
  return rgbToHex(r1+(r2-r1)*f, g1+(g2-g1)*f, b1+(b2-b1)*f)
}

type NodeType = 'cooler' | 'heater' | 'other'

const TYPE_RANGES: Record<Exclude<ColorScheme, 'default'>, Record<NodeType, [number, number]>> = {
  viridis:  { cooler: [0.28, 0.45], heater: [0.80, 0.95], other: [0.55, 0.70] },
  plasma:   { cooler: [0.05, 0.18], heater: [0.80, 0.95], other: [0.42, 0.58] },
  coolwarm: { cooler: [0.05, 0.20], heater: [0.80, 0.95], other: [0.45, 0.55] },
}

function assignColors(labels: string[], scheme: ColorScheme): string[] {
  if (scheme === 'default') {
    const fallback = ['#10b981', '#f59e0b', '#8b5cf6', '#ec4899']
    let otherIdx = 0
    return labels.map(label => {
      const l = label.toLowerCase()
      if (l.includes('cool')) return '#3b82f6'
      if (l.includes('heat')) return '#ef4444'
      return fallback[(otherIdx++) % fallback.length]
    })
  }
  const groups: Record<NodeType, number[]> = { cooler: [], heater: [], other: [] }
  labels.forEach((label, i) => {
    const l = label.toLowerCase()
    if (l.includes('cool'))      groups.cooler.push(i)
    else if (l.includes('heat')) groups.heater.push(i)
    else                         groups.other.push(i)
  })
  const colors = new Array<string>(labels.length)
  const ranges = TYPE_RANGES[scheme]
  for (const type of ['cooler', 'heater', 'other'] as NodeType[]) {
    const indices = groups[type]
    const [lo, hi] = ranges[type]
    indices.forEach((globalIdx, rank) => {
      const t = indices.length === 1 ? (lo + hi) / 2 : lo + (hi - lo) * rank / (indices.length - 1)
      colors[globalIdx] = sampleColormap(COLORMAPS[scheme], t)
    })
  }
  return colors
}

interface Props {
  results: SimResults
  onClose: () => void
  theme?: 'dark' | 'light'
}

type Tab = 'temperatures' | 'valves' | 'humidity' | 'junctions' | 'metrics' | 'controller'

export default function ResultsPanel({ results, onClose, theme = 'dark' }: Props) {
  const [tab,           setTab]           = useState<Tab>('temperatures')
  const [collapsed,     setCollapsed]     = useState(false)
  const [height,        setHeight]        = useState(DEFAULT_HEIGHT)
  const [colorScheme,   setColorScheme]   = useState<ColorScheme>('default')
  const [exporting,     setExporting]     = useState(false)

  const handleMatplotlibExport = async () => {
    if (tab === 'metrics' || tab === 'controller') return
    setExporting(true)
    try {
      const resp = await axios.post('http://localhost:8000/export_plot',
        { tab, data: results, scheme: colorScheme },
        { responseType: 'blob' }
      )
      const url = URL.createObjectURL(resp.data)
      const a   = document.createElement('a')
      a.href     = url
      a.download = `hvac_${tab}.png`
      a.click()
      URL.revokeObjectURL(url)
    } finally {
      setExporting(false)
    }
  }

  const { t, outputs, valves, humidity, junctions, metrics, d_signal } = results

  // Tell Plotly to re-fit after height or collapsed state changes
  useEffect(() => {
    const id = requestAnimationFrame(() => window.dispatchEvent(new Event('resize')))
    return () => cancelAnimationFrame(id)
  }, [height, collapsed])

  // ── Drag-to-resize ────────────────────────────────────────────────────────────
  const onDragStart = (e: React.MouseEvent) => {
    e.preventDefault()
    const startY      = e.clientY
    const startHeight = height

    const onMove = (me: MouseEvent) => {
      const delta     = startY - me.clientY   // drag up → larger
      const newHeight = Math.max(120, Math.min(window.innerHeight - 120, startHeight + delta))
      setHeight(newHeight)
    }
    const onUp = () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup',   onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup',   onUp)
  }

  // ── Traces ────────────────────────────────────────────────────────────────────
  const disturbanceTrace = {
    x: t, y: d_signal,
    type: 'scatter' as const, mode: 'lines' as const,
    name: 'Outdoor air (T_fresh)',
    line: { color: '#94a3b8', width: 1.5, dash: 'dot' as const },
  }

  const outputEntries   = Object.entries(outputs)
  const valveEntries    = Object.entries(valves)
  const humidityEntries = Object.entries(humidity ?? {})

  const tempColors     = assignColors(outputEntries.map(([, s]) => s.label), colorScheme)
  const valveColors    = assignColors(valveEntries.map(([, s]) => s.label), colorScheme)
  const humidityColors = assignColors(humidityEntries.map(([, s]) => s.label), colorScheme)

  const tempTraces = outputEntries.flatMap(([, series], i) => {
    const color = tempColors[i]
    return [
      { x: t, y: series.y, type: 'scatter' as const, mode: 'lines' as const,
        name: series.label, legendgroup: series.label, line: { color, width: 2.5 } },
      { x: [t[0], t[t.length - 1]], y: [series.ref, series.ref],
        type: 'scatter' as const, mode: 'lines' as const,
        name: `${series.label} ref (${series.ref.toFixed(1)} °C)`, legendgroup: series.label,
        line: { color, width: 1.5, dash: 'dash' as const }, showlegend: true },
    ]
  })

  const valveTraces = valveEntries.map(([, series], i) => ({
    x: t, y: series.y, type: 'scatter' as const, mode: 'lines' as const,
    name: series.label, line: { color: valveColors[i], width: 2 },
  }))

  const humidityTraces = humidityEntries.map(([, series], i) => ({
    x: t, y: series.y, type: 'scatter' as const, mode: 'lines' as const,
    name: `${series.label} inlet`,
    line: { color: humidityColors[i], width: 2 },
  }))

  // ── Control-effort RMSE ───────────────────────────────────────────────────────
  // Numerically differentiate each valve signal (central differences, with
  // forward/backward at the endpoints), then compute RMS of dv/dt over the
  // full simulation window.  This quantifies valve activity: a high value means
  // the valve was moving fast / erratically; a low value means smooth control.
  const valveEffortRmse: { label: string; rmse: number }[] = valveEntries.map(([, series]) => {
    const v = series.y
    const n = v.length
    if (n < 2) return { label: series.label, rmse: 0 }

    // dv/dt via central differences
    const dvdt = new Array<number>(n)
    dvdt[0]     = (v[1]     - v[0])         / (t[1]     - t[0])
    dvdt[n - 1] = (v[n - 1] - v[n - 2])     / (t[n - 1] - t[n - 2])
    for (let i = 1; i < n - 1; i++) {
      dvdt[i] = (v[i + 1] - v[i - 1]) / (t[i + 1] - t[i - 1])
    }

    // Trapezoidal integral of (dv/dt)^2
    let integral = 0
    for (let i = 0; i < n - 1; i++) {
      integral += 0.5 * (dvdt[i] ** 2 + dvdt[i + 1] ** 2) * (t[i + 1] - t[i])
    }
    const T = t[n - 1] - t[0]
    return { label: series.label, rmse: T > 0 ? Math.sqrt(integral / T) : 0 }
  })

  // ── Metrics ───────────────────────────────────────────────────────────────────
  const ssRows    = Object.values(metrics.steady_state)
  const KI        = metrics.K_I
  const nRows     = KI.length
  const nCols     = KI[0]?.length ?? 0
  const rowLabels = metrics.actuated_ids.map((id, i) => metrics.steady_state[id]?.label ?? `u${i}`)
  const colLabels = metrics.actuated_ids.map((id, i) => metrics.steady_state[id]?.label ?? `y${i}`)
  const poleTrace = {
    x: metrics.cl_eigenvalues.map(e => e.re),
    y: metrics.cl_eigenvalues.map(e => e.im),
    type: 'scatter' as const, mode: 'markers' as const,
    name: 'CL poles', marker: { color: '#6366f1', size: 8, symbol: 'x' as const },
  }

  // ── CSV exports ───────────────────────────────────────────────────────────────
  const exportTemperatures = () => {
    const keys = Object.keys(outputs)
    const headers = ['time_s', 'd_signal_C', ...keys.map(k => `${outputs[k].label}_C`), ...keys.map(k => `${outputs[k].label}_ref_C`)]
    const rows = t.map((ti, i) => [ti, d_signal[i], ...keys.map(k => outputs[k].y[i]), ...keys.map(k => outputs[k].ref)])
    downloadCsv('hvac_temperatures.csv', headers, rows)
  }

  const exportValves = () => {
    const keys = Object.keys(valves)
    const headers = ['time_s', ...keys.map(k => `${valves[k].label}_opening`)]
    const rows = t.map((ti, i) => [ti, ...keys.map(k => valves[k].y[i])])
    downloadCsv('hvac_valves.csv', headers, rows)
  }

  const exportHumidity = () => {
    const keys = Object.keys(humidity ?? {})
    const headers = ['time_s', ...keys.map(k => humidity[k].label + '_kg_per_kg')]
    const rows = t.map((ti, i) => [ti, ...keys.map(k => humidity[k].y[i])])
    downloadCsv('hvac_humidity.csv', headers, rows)
  }

  const exportJunction = (jid: string) => {
    const jd = junctions[jid]
    const inletTempHeaders = jd.inlet_ids.map(src => `${jd.inlet_labels[jd.inlet_ids.indexOf(src)]}_T_C`)
    const inletHumHeaders  = jd.inlet_ids.map(src => `${jd.inlet_labels[jd.inlet_ids.indexOf(src)]}_hum_kg_per_kg`)
    const headers = ['time_s', ...inletTempHeaders, 'mixed_outlet_T_C', ...inletHumHeaders, 'mixed_outlet_hum_kg_per_kg']
    const rows = t.map((ti, i) => [
      ti,
      ...jd.inlet_ids.map(src => jd.inlet_temperatures[src][i]),
      jd.outlet_temperatures[i],
      ...jd.inlet_ids.map(src => jd.inlet_specific_humidities[src][i]),
      jd.outlet_specific_humidities[i],
    ])
    downloadCsv(`hvac_junction_${jd.label.replace(/\s+/g, '_')}.csv`, headers, rows)
  }

  const exportAll = () => {
    const outKeys = Object.keys(outputs)
    const valKeys = Object.keys(valves)
    const humKeys = Object.keys(humidity ?? {})
    const headers = [
      'time_s', 'd_signal_C',
      ...outKeys.map(k => `T_${outputs[k].label}_C`),
      ...outKeys.map(k => `T_${outputs[k].label}_ref_C`),
      ...valKeys.map(k => `valve_${valves[k].label}`),
      ...humKeys.map(k => `hum_${humidity[k].label}_kg_per_kg`),
    ]
    const rows = t.map((ti, i) => [
      ti, d_signal[i],
      ...outKeys.map(k => outputs[k].y[i]),
      ...outKeys.map(k => outputs[k].ref),
      ...valKeys.map(k => valves[k].y[i]),
      ...humKeys.map(k => humidity[k].y[i]),
    ])
    downloadCsv('hvac_all_data.csv', headers, rows)
    // Also export each junction if present
    Object.keys(junctions ?? {}).forEach(jid => exportJunction(jid))
  }

  const exportBtnStyle: React.CSSProperties = {
    background: 'var(--bg-deepest)', border: '1px solid var(--bg-mid)',
    color: 'var(--text-sec)', borderRadius: 4, fontSize: 11,
    padding: '4px 12px', cursor: 'pointer', alignSelf: 'flex-end',
    margin: '4px 0 2px',
  }

  const isLight = theme === 'light'
  const gridColor  = isLight ? '#dddddd' : '#334155'
  const lineColor  = isLight ? '#333333' : '#94a3b8'
  const axisStyle  = {
    gridcolor:     gridColor,
    gridwidth:     1,
    showgrid:      true,
    showline:      true,
    linecolor:     lineColor,
    linewidth:     1,
    mirror:        true,     // draws the box (all 4 spines)
    tickcolor:     lineColor,
    tickfont:      { size: 11 },
    zerolinecolor: gridColor,
    zerolinewidth: 1,
  }
  const commonLayout = {
    paper_bgcolor: isLight ? '#ffffff' : '#1e293b',
    plot_bgcolor:  isLight ? '#ffffff' : '#0f172a',
    font:    { family: 'Arial, sans-serif', color: isLight ? '#333333' : '#e2e8f0', size: 12 },
    margin:  { t: 40, r: 20, b: 55, l: 65 },
    legend:  {
      bgcolor:      isLight ? 'rgba(255,255,255,0.9)' : 'rgba(30,41,59,0.9)',
      bordercolor:  isLight ? '#cccccc' : '#475569',
      borderwidth:  1,
      font:         { size: 11 },
    },
  }

  return (
    <section
      className="results-panel"
      style={{ height: collapsed ? undefined : height }}
    >
      {/* Drag handle — only when expanded */}
      {!collapsed && (
        <div className="results-drag-handle" onMouseDown={onDragStart} title="Drag to resize" />
      )}

      <div className="results-header">
        <div className="tab-bar">
          {(['temperatures', 'valves', 'humidity', 'junctions', 'metrics', 'controller'] as Tab[]).map(tabName => (
            <button
              key={tabName}
              className={`tab-btn${tab === tabName ? ' tab-active' : ''}`}
              onClick={() => { setCollapsed(false); setTab(tabName) }}
            >
              {tabName.charAt(0).toUpperCase() + tabName.slice(1)}
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <select
            value={colorScheme}
            onChange={e => setColorScheme(e.target.value as ColorScheme)}
            title="Plot color scheme"
            style={{
              background: 'var(--bg-deepest)', border: '1px solid var(--bg-mid)',
              color: 'var(--text-sec)', borderRadius: 4, fontSize: 11, padding: '2px 6px', cursor: 'pointer',
            }}
          >
            <option value="default">Default</option>
            <option value="viridis">Viridis</option>
            <option value="plasma">Plasma</option>
            <option value="coolwarm">Coolwarm</option>
          </select>
          <button
            onClick={exportAll}
            title="Export all time-series data as CSV (+ one CSV per junction)"
            style={{
              background: 'var(--bg-deepest)', border: '1px solid var(--bg-mid)',
              color: 'var(--text-sec)', borderRadius: 4, fontSize: 11,
              padding: '2px 8px', cursor: 'pointer',
            }}
          >
            ↓ Export all CSV
          </button>
          {tab !== 'metrics' && tab !== 'controller' && (
            <button
              onClick={handleMatplotlibExport}
              disabled={exporting}
              title={`Export ${tab} as matplotlib PNG`}
              style={{
                background: 'var(--bg-deepest)', border: '1px solid var(--bg-mid)',
                color: exporting ? 'var(--text-muted)' : 'var(--text-sec)',
                borderRadius: 4, fontSize: 11, padding: '2px 8px', cursor: exporting ? 'not-allowed' : 'pointer',
              }}
            >
              {exporting ? 'Exporting…' : 'Save figure'}
            </button>
          )}
          <button
            className="close-btn"
            onClick={() => setCollapsed(c => !c)}
            title={collapsed ? 'Expand results' : 'Collapse results'}
          >
            {collapsed ? '▲' : '▼'}
          </button>
          <button className="close-btn" onClick={onClose} title="Close results">✕</button>
        </div>
      </div>

      {!collapsed && (
        <div className="results-body">
          {tab === 'temperatures' && (
            <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
              <Plot
                data={[disturbanceTrace, ...tempTraces]}
                layout={{
                  ...commonLayout,
                  xaxis: { ...axisStyle, title: { text: 'Time (s)' } },
                  yaxis: { ...axisStyle, title: { text: 'Temperature (°C)' } },
                }}
                style={{ width: '100%', flex: 1, minHeight: 0 }}
                useResizeHandler
                config={{ responsive: true }}
              />
              <button style={exportBtnStyle} onClick={exportTemperatures}>↓ Export data</button>
            </div>
          )}

          {tab === 'valves' && (
            <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
              <Plot
                data={valveTraces}
                layout={{
                  ...commonLayout,
                  xaxis: { ...axisStyle, title: { text: 'Time (s)' } },
                  yaxis: { ...axisStyle, title: { text: 'Opening (0–1)' }, range: [0, 1], autorange: false },
                }}
                style={{ width: '100%', flex: 1, minHeight: 0 }}
                useResizeHandler
                config={{ responsive: true }}
              />
              <button style={exportBtnStyle} onClick={exportValves}>↓ Export data</button>
            </div>
          )}

          {tab === 'humidity' && (
            humidityTraces.length === 0
              ? <div style={{ color: '#94a3b8', padding: 24 }}>Humidity data is only available in nonlinear mode.</div>
              : <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                  <Plot
                    data={humidityTraces}
                    layout={{
                      ...commonLayout,
                      title: { text: 'Specific humidity at heat exchanger inlets', font: { size: 13 } },
                      xaxis: { ...axisStyle, title: { text: 'Time (s)' } },
                      yaxis: { ...axisStyle, title: { text: 'Specific humidity (kg/kg dry air)' } },
                    }}
                    style={{ width: '100%', flex: 1, minHeight: 0 }}
                    useResizeHandler
                    config={{ responsive: true }}
                  />
                  <button style={exportBtnStyle} onClick={exportHumidity}>↓ Export data</button>
                </div>
          )}

          {tab === 'junctions' && (
            Object.keys(junctions).length === 0
              ? <div style={{ color: '#94a3b8', padding: 24 }}>Junction data is only available in nonlinear mode, and only when the graph contains junctions.</div>
              : <div style={{ overflowY: 'auto', height: '100%', display: 'flex', flexDirection: 'column', gap: 16, padding: 8 }}>
                  {Object.entries(junctions).map(([jid, jd]) => {
                    const inletColors = assignColors(jd.inlet_labels, colorScheme)
                    const inletColorMap: Record<string, string> = {}
                    jd.inlet_ids.forEach((src, i) => { inletColorMap[src] = inletColors[i] })
                    const outletColor = '#f59e0b'

                    const tempTraces = [
                      ...jd.inlet_ids.map(src => ({
                        x: t, y: jd.inlet_temperatures[src],
                        type: 'scatter' as const, mode: 'lines' as const,
                        name: jd.inlet_labels[jd.inlet_ids.indexOf(src)],
                        line: { color: inletColorMap[src], width: 2 },
                      })),
                      { x: t, y: jd.outlet_temperatures,
                        type: 'scatter' as const, mode: 'lines' as const,
                        name: 'Mixed outlet',
                        line: { color: outletColor, width: 2, dash: 'dash' as const } },
                    ]

                    const humTraces = [
                      ...jd.inlet_ids.map(src => ({
                        x: t, y: jd.inlet_specific_humidities[src],
                        type: 'scatter' as const, mode: 'lines' as const,
                        name: jd.inlet_labels[jd.inlet_ids.indexOf(src)],
                        line: { color: inletColorMap[src], width: 2 },
                        showlegend: false,
                      })),
                      { x: t, y: jd.outlet_specific_humidities,
                        type: 'scatter' as const, mode: 'lines' as const,
                        name: 'Mixed outlet',
                        line: { color: outletColor, width: 2, dash: 'dash' as const },
                        showlegend: false },
                    ]

                    return (
                      <div key={jid}>
                        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 4, paddingLeft: 4 }}>
                          <span style={{ color: '#e2e8f0', fontWeight: 600 }}>Junction: {jd.label}</span>
                          <button style={{ ...exportBtnStyle, margin: '0 0 0 auto' }} onClick={() => exportJunction(jid)}>↓ Export data</button>
                        </div>
                        <div style={{ display: 'flex', gap: 8 }}>
                          <Plot
                            data={tempTraces}
                            layout={{ ...commonLayout,
                              margin: { t: 24, r: 12, b: 40, l: 60 },
                              xaxis: { ...axisStyle, title: { text: 'Time (s)' } },
                              yaxis: { ...axisStyle, title: { text: 'Temperature (°C)' } },
                            }}
                            style={{ flex: 1, height: 220 }}
                            useResizeHandler
                            config={{ responsive: true }}
                          />
                          <Plot
                            data={humTraces}
                            layout={{ ...commonLayout,
                              margin: { t: 24, r: 12, b: 40, l: 60 },
                              xaxis: { ...axisStyle, title: { text: 'Time (s)' } },
                              yaxis: { ...axisStyle, title: { text: 'Specific humidity (kg/kg)' } },
                            }}
                            style={{ flex: 1, height: 220 }}
                            useResizeHandler
                            config={{ responsive: true }}
                          />
                        </div>
                      </div>
                    )
                  })}
                </div>
          )}

          {tab === 'metrics' && (
            <div className="metrics-grid">
              {/* Steady-state table */}
              <div className="metric-card">
                <h4 className="metric-title">Steady-state summary</h4>
                <table className="ss-table">
                  <thead>
                    <tr><th>Node</th><th>Final (°C)</th><th>Ref (°C)</th><th>Error (°C)</th><th>Valve</th></tr>
                  </thead>
                  <tbody>
                    {ssRows.map(row => {
                      const err = row.temp_C - row.ref_C
                      return (
                        <tr key={row.label}>
                          <td>{row.label}</td>
                          <td>{row.temp_C.toFixed(2)}</td>
                          <td>{row.ref_C.toFixed(2)}</td>
                          <td style={{ color: Math.abs(err) > 0.5 ? '#f87171' : '#34d399' }}>{err.toFixed(3)}</td>
                          <td>{row.valve.toFixed(3)}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
                <p className="metric-note">Condition number: {metrics.condition_number.toExponential(3)}</p>
              </div>

              {/* Valve control effort */}
              <div className="metric-card">
                <h4 className="metric-title">Valve control effort (RMS of dv/dt)  —  lower = smoother</h4>
                <Plot
                  data={[{
                    type: 'bar' as const,
                    orientation: 'h' as const,
                    x: valveEffortRmse.map(e => e.rmse),
                    y: valveEffortRmse.map(e => e.label),
                    text: valveEffortRmse.map(e => e.rmse.toExponential(3)),
                    textposition: 'auto' as const,
                    insidetextanchor: 'start' as const,
                    marker: { color: valveEffortRmse.map((_, i) => valveColors[i] ?? '#6366f1') },
                    hovertemplate: '%{y}: %{x:.4e} s⁻¹<extra></extra>',
                  } as never]}
                  layout={{
                    ...commonLayout,
                    margin: { t: 10, r: 120, b: 50, l: 100 },
                    xaxis: { ...axisStyle, title: { text: 'RMS of dv/dt  (s⁻¹)' } },
                    yaxis: { ...axisStyle, automargin: true, autorange: 'reversed' },
                    bargap: 0.35,
                  }}
                  style={{ width: '100%', height: 220 }}
                  useResizeHandler
                  config={{ responsive: true }}
                />
              </div>
            </div>
          )}

          {tab === 'controller' && (
            <div className="metrics-grid">
              {/* K_I heatmap */}
              <div className="metric-card">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                  <h4 className="metric-title" style={{ marginBottom: 0 }}>K_I gain matrix</h4>
                  <button
                    onClick={() => downloadGains(metrics, rowLabels, colLabels)}
                    style={{ background: '#1e3a5f', border: '1px solid #3b82f6', color: '#93c5fd', borderRadius: 4, padding: '3px 10px', fontSize: 11, cursor: 'pointer' }}
                  >
                    ↓ Save gains
                  </button>
                </div>
                <Plot
                  data={[{
                    z: KI, x: colLabels.slice(0, nCols), y: rowLabels.slice(0, nRows),
                    type: 'heatmap' as const, colorscale: 'RdBu', reversescale: true,
                    zmid: 0,
                    text: KI.map(row => row.map(v => v.toFixed(4))) as unknown as string[],
                    texttemplate: '%{text}',
                    hovertemplate: 'Input: %{y}<br>Output: %{x}<br>Gain: %{z:.4f}<extra></extra>',
                  } as never]}
                  layout={{
                    ...commonLayout,
                    margin: { t: 10, r: 20, b: 80, l: 120 },
                    xaxis: { title: { text: 'Output channel' }, tickangle: -30 },
                    yaxis: { title: { text: 'Input channel'  } },
                  }}
                  style={{ width: '100%', height: 280 }}
                  useResizeHandler
                  config={{ responsive: true }}
                />
              </div>

              {/* Pole map */}
              <div className="metric-card">
                <h4 className="metric-title">Closed-loop poles</h4>
                <Plot
                  data={[
                    poleTrace,
                    { x: [0,0], y: [-1,1].map(v => v * (Math.max(...metrics.cl_eigenvalues.map(e => Math.abs(e.im)))||1)*1.2),
                      type: 'scatter' as const, mode: 'lines' as const,
                      line: { color: '#64748b', width: 1, dash: 'dot' as const }, showlegend: false, name: 'Im axis' },
                  ]}
                  layout={{
                    ...commonLayout,
                    margin: { t: 10, r: 20, b: 50, l: 60 },
                    xaxis: { ...axisStyle, title: { text: 'Real' } },
                    yaxis: { ...axisStyle, title: { text: 'Imaginary' } },
                  }}
                  style={{ width: '100%', height: 280 }}
                  useResizeHandler
                  config={{ responsive: true }}
                />
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  )
}

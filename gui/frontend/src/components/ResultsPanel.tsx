import { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'
import type { SimResults, SimMetrics } from '../types'

const DEFAULT_HEIGHT = 380

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

const NODE_COLORS: Record<string, string> = {
  cooler:   '#3b82f6',
  heater:   '#ef4444',
  airduct:  '#6b7280',
  junction: '#8b5cf6',
}

function pickColor(label: string, idx: number): string {
  const palette = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899']
  const lower = label.toLowerCase()
  if (lower.includes('cool')) return NODE_COLORS.cooler
  if (lower.includes('heat')) return NODE_COLORS.heater
  return palette[idx % palette.length]
}

interface Props {
  results: SimResults
  onClose: () => void
}

type Tab = 'temperatures' | 'valves' | 'humidity' | 'junctions' | 'metrics'

export default function ResultsPanel({ results, onClose }: Props) {
  const [tab,       setTab]       = useState<Tab>('temperatures')
  const [collapsed, setCollapsed] = useState(false)
  const [height,    setHeight]    = useState(DEFAULT_HEIGHT)

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

  const tempTraces = Object.entries(outputs).flatMap(([, series], i) => {
    const color = pickColor(series.label, i)
    return [
      { x: t, y: series.y, type: 'scatter' as const, mode: 'lines' as const,
        name: series.label, legendgroup: series.label, line: { color, width: 2 } },
      { x: [t[0], t[t.length - 1]], y: [series.ref, series.ref],
        type: 'scatter' as const, mode: 'lines' as const,
        name: `${series.label} ref`, legendgroup: series.label,
        line: { color, width: 1, dash: 'dash' as const }, showlegend: false },
    ]
  })

  const valveTraces = Object.entries(valves).map(([, series], i) => ({
    x: t, y: series.y, type: 'scatter' as const, mode: 'lines' as const,
    name: series.label, line: { color: pickColor(series.label, i), width: 2 },
  }))

  const junctionTempTraces = Object.entries(junctions ?? {}).map(([, jd]) => ({
    x: t, y: jd.outlet_temperatures,
    type: 'scatter' as const, mode: 'lines' as const,
    name: `${jd.label} (mixed)`,
    line: { color: '#f59e0b', width: 1.5, dash: 'dot' as const },
  }))

  const junctionHumidityTraces = Object.entries(junctions ?? {}).map(([, jd]) => ({
    x: t, y: jd.outlet_specific_humidities,
    type: 'scatter' as const, mode: 'lines' as const,
    name: `${jd.label} (mixed)`,
    line: { color: '#f59e0b', width: 1.5, dash: 'dot' as const },
  }))

  const humidityTraces = Object.entries(humidity ?? {}).map(([, series], i) => ({
    x: t, y: series.y,
    type: 'scatter' as const, mode: 'lines' as const,
    name: `${series.label} inlet`,
    line: { color: pickColor(series.label, i), width: 2 },
  }))

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

  const commonLayout = {
    paper_bgcolor: '#1e293b',
    plot_bgcolor:  '#0f172a',
    font:          { color: '#e2e8f0', size: 12 },
    margin:        { t: 30, r: 20, b: 50, l: 60 },
    legend:        { bgcolor: '#1e293b', bordercolor: '#334155', borderwidth: 1 },
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
          {(['temperatures', 'valves', 'humidity', 'junctions', 'metrics'] as Tab[]).map(tabName => (
            <button
              key={tabName}
              className={`tab-btn${tab === tabName ? ' tab-active' : ''}`}
              onClick={() => { setCollapsed(false); setTab(tabName) }}
            >
              {tabName.charAt(0).toUpperCase() + tabName.slice(1)}
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
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
            <Plot
              data={[disturbanceTrace, ...tempTraces]}
              layout={{
                ...commonLayout,
                xaxis: { title: { text: 'Time (s)' },        gridcolor: '#334155', zerolinecolor: '#475569' },
                yaxis: { title: { text: 'Temperature (°C)' }, gridcolor: '#334155', zerolinecolor: '#475569' },
              }}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler
              config={{ responsive: true }}
            />
          )}

          {tab === 'valves' && (
            <Plot
              data={[
                ...valveTraces,
                { x: [t[0], t[t.length-1]], y: [1,1], type: 'scatter' as const, mode: 'lines' as const,
                  line: { color: '#f87171', dash: 'dot' as const, width: 1 }, name: 'Max', showlegend: false },
                { x: [t[0], t[t.length-1]], y: [0,0], type: 'scatter' as const, mode: 'lines' as const,
                  line: { color: '#f87171', dash: 'dot' as const, width: 1 }, name: 'Min', showlegend: false },
              ]}
              layout={{
                ...commonLayout,
                xaxis: { title: { text: 'Time (s)' },         gridcolor: '#334155', zerolinecolor: '#475569' },
                yaxis: { title: { text: 'Opening (0–1)' }, range: [-0.05, 1.05], gridcolor: '#334155', zerolinecolor: '#475569' },
              }}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler
              config={{ responsive: true }}
            />
          )}

          {tab === 'humidity' && (
            humidityTraces.length === 0
              ? <div style={{ color: '#94a3b8', padding: 24 }}>Humidity data is only available in nonlinear mode.</div>
              : <Plot
                  data={humidityTraces}
                  layout={{
                    ...commonLayout,
                    title: { text: 'Specific humidity at the inlet of each heat exchanger', font: { size: 13, color: '#94a3b8' } },
                    margin: { ...commonLayout.margin, t: 48 },
                    xaxis: { title: { text: 'Time (s)' },                          gridcolor: '#334155', zerolinecolor: '#475569' },
                    yaxis: { title: { text: 'Specific humidity (kg/kg dry air)' },  gridcolor: '#334155', zerolinecolor: '#475569' },
                  }}
                  style={{ width: '100%', height: '100%' }}
                  useResizeHandler
                  config={{ responsive: true }}
                />
          )}

          {tab === 'junctions' && (
            Object.keys(junctions).length === 0
              ? <div style={{ color: '#94a3b8', padding: 24 }}>Junction data is only available in nonlinear mode, and only when the graph contains junctions.</div>
              : <div style={{ overflowY: 'auto', height: '100%', display: 'flex', flexDirection: 'column', gap: 16, padding: 8 }}>
                  {Object.entries(junctions).map(([jid, jd]) => {
                    const inletColorMap: Record<string, string> = {}
                    jd.inlet_ids.forEach((src, i) => {
                      inletColorMap[src] = pickColor(jd.inlet_labels[i], i)
                    })
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
                        <div style={{ color: '#e2e8f0', fontWeight: 600, marginBottom: 4, paddingLeft: 4 }}>
                          Junction: {jd.label}
                        </div>
                        <div style={{ display: 'flex', gap: 8 }}>
                          <Plot
                            data={tempTraces}
                            layout={{ ...commonLayout,
                              margin: { t: 24, r: 12, b: 40, l: 60 },
                              xaxis: { title: { text: 'Time (s)' },          gridcolor: '#334155', zerolinecolor: '#475569' },
                              yaxis: { title: { text: 'Temperature (°C)' },  gridcolor: '#334155', zerolinecolor: '#475569' },
                            }}
                            style={{ flex: 1, height: 220 }}
                            useResizeHandler
                            config={{ responsive: true }}
                          />
                          <Plot
                            data={humTraces}
                            layout={{ ...commonLayout,
                              margin: { t: 24, r: 12, b: 40, l: 60 },
                              xaxis: { title: { text: 'Time (s)' },                         gridcolor: '#334155', zerolinecolor: '#475569' },
                              yaxis: { title: { text: 'Specific humidity (kg/kg)' }, gridcolor: '#334155', zerolinecolor: '#475569' },
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
                    xaxis: { title: { text: 'Real' },      gridcolor: '#334155', zerolinecolor: '#94a3b8' },
                    yaxis: { title: { text: 'Imaginary' }, gridcolor: '#334155', zerolinecolor: '#94a3b8' },
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

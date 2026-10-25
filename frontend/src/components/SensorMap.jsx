import { useEffect, useMemo, useState } from 'react'
import { CircleMarker, ImageOverlay, MapContainer, Pane, Popup, TileLayer, Tooltip, useMap } from 'react-leaflet'
import { MAP_MODELS, MODEL_COLOR, RAMP_GRADIENT, SEVERITY_COLOR, STATUS_LABEL, rampCss, rampRgb } from '../api.js'

function rasterToDataUrl(r, vmin, vmax) {
  const small = document.createElement('canvas')
  small.width = r.nx; small.height = r.ny
  const ctx = small.getContext('2d')
  const img = ctx.createImageData(r.nx, r.ny)
  const span = Math.max(vmax - vmin, 1e-6)
  r.values.forEach((v, i) => {
    const o = i * 4
    if (v == null) { img.data[o + 3] = 0; return }
    const [R, G, B] = rampRgb((v - vmin) / span)
    img.data[o] = R; img.data[o + 1] = G; img.data[o + 2] = B; img.data[o + 3] = 205
  })
  ctx.putImageData(img, 0, 0)
  const big = document.createElement('canvas')
  big.width = r.nx * 4; big.height = r.ny * 4
  const bctx = big.getContext('2d')
  bctx.imageSmoothingEnabled = true
  bctx.imageSmoothingQuality = 'high'
  bctx.drawImage(small, 0, 0, big.width, big.height)
  return big.toDataURL('image/png')
}

function FitToData({ bounds, signature }) {
  const map = useMap()
  useEffect(() => {
    if (bounds) map.fitBounds(bounds, { padding: [24, 24], maxZoom: 12 })
  }, [signature]) // eslint-disable-line react-hooks/exhaustive-deps
  return null
}

function nseRgb(nse) {
  if (nse == null || Number.isNaN(nse)) return [75, 98, 128]
  const stops = [
    [-1, [234, 88, 12]],
    [-0.2, [245, 158, 11]],
    [0, [139, 166, 198]],
    [0.5, [56, 189, 248]],
    [1, [34, 211, 238]],
  ]
  const x = Math.min(1, Math.max(-1.5, nse))
  if (x <= stops[0][0]) return stops[0][1]
  for (let i = 1; i < stops.length; i++) {
    const [t1, c1] = stops[i], [t0, c0] = stops[i - 1]
    if (x <= t1) {
      const u = (x - t0) / (t1 - t0)
      return c0.map((v, k) => Math.round(v + (c1[k] - v) * u))
    }
  }
  return stops[stops.length - 1][1]
}

const nseCss = (nse) => `rgb(${nseRgb(nse).join(',')})`

function stationFill(view, depthT, metrics, skillModel) {
  if (view === 'depth' || !metrics) return rampCss(depthT)
  if (view === 'nse') {
    const nse = metrics[skillModel]?.nse ?? MAP_MODELS.map((m) => metrics[m]?.nse).find((v) => v != null)
    return nseCss(nse)
  }
  if (view === 'skill') {
    const row = metrics[skillModel] || MAP_MODELS.map((m) => metrics[m]).find(Boolean)
    if (!row) return '#4b6280'
    return row.outcome_vs_persistence === 'win' ? '#2dd4bf' : '#f59e0b'
  }
  const winner = metrics.compare?.winner
  if (winner === 'PatchTST') return MODEL_COLOR.PatchTST
  if (winner === 'LightGBM') return MODEL_COLOR.LightGBM
  if (winner === 'tie') return '#8ba6c6'
  return '#4b6280'
}

function Legend({ view, vmin, vmax, thresholds, horizon, skillModel }) {
  if (view === 'nse') {
    return (
      <div className="map-card bottom-3 left-3 w-64">
        <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-sky">{skillModel} NSE · {horizon}</div>
        <div className="h-2.5 rounded-full" style={{ background: 'linear-gradient(90deg,#ea580c 0%,#f59e0b 32%,#8ba6c6 50%,#38bdf8 75%,#22d3ee 100%)' }} />
        <div className="mt-1 flex justify-between text-muted"><span>&lt; 0 worse than mean</span><span>1 perfect</span></div>
        <p className="mt-2 text-[11px] text-muted">Fill is station NSE for {skillModel}. Ring still shows anomaly state.</p>
      </div>
    )
  }
  if (view === 'skill') {
    return (
      <div className="map-card bottom-3 left-3 w-64">
        <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-sky">{skillModel} vs persistence · {horizon}</div>
        <div className="flex flex-wrap gap-x-3 gap-y-1">
          <span className="flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-full bg-teal" />beats persistence</span>
          <span className="flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-full bg-warn" />does not beat</span>
          <span className="flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-full bg-[#4b6280]" />no score</span>
        </div>
        <p className="mt-2 text-[11px] text-muted">Teal wells: {skillModel} RMSE lower than persistence. Grey wells were not in the stored comparison.</p>
      </div>
    )
  }
  if (view === 'compare') {
    return (
      <div className="map-card bottom-3 left-3 w-64">
        <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-sky">Lower RMSE at well · {horizon}</div>
        <div className="flex flex-wrap gap-x-3 gap-y-1">
          <span className="flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: MODEL_COLOR.PatchTST }} />PatchTST</span>
          <span className="flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: MODEL_COLOR.LightGBM }} />LightGBM</span>
          <span className="flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-full bg-muted" />tie / missing</span>
        </div>
      </div>
    )
  }
  const semi = thresholds?.SEMI_CRITICAL, crit = thresholds?.CRITICAL
  return (
    <div className="map-card bottom-3 left-3 w-64">
      <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-sky">Groundwater depth</div>
      <div className="h-2.5 rounded-full" style={{ background: RAMP_GRADIENT }} />
      <div className="mt-1 flex justify-between text-muted"><span>Shallow</span><span>Moderate</span><span>Deep</span></div>
      {vmin != null && <div className="flex justify-between text-ink"><span>{vmin.toFixed(1)} m</span><span>{vmax.toFixed(1)} m bgl</span></div>}
      <div className="mt-1 text-sky">Higher m bgl = deeper groundwater</div>
      <div className="mt-2 border-t border-rule pt-2">
        <div className="mb-1 text-muted">Station ring = anomaly state</div>
        <div className="flex flex-wrap gap-x-3 gap-y-1">
          <span className="flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-full border border-ink" />none</span>
          {Object.entries(SEVERITY_COLOR).map(([k, c]) => (
            <span key={k} className="flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: c }} />{k.toLowerCase()}</span>
          ))}
        </div>
      </div>
      <div className="mt-2 border-t border-rule pt-2 text-muted">
        Prototype classification{semi && crit ? `: moderate >= ${semi} m, deep >= ${crit} m bgl` : ''} — not an official regulatory classification.
      </div>
    </div>
  )
}

function EvalRows({ metrics, horizon }) {
  if (!metrics) return <div className="mt-2 text-muted">No station score at {horizon}.</div>
  return (
    <div className="mt-2 space-y-1">
      {MAP_MODELS.map((name) => {
        const row = metrics[name]
        if (!row) return <div key={name} className="text-muted">{name}: not scored</div>
        return (
          <div key={name}>
            <span className="font-semibold" style={{ color: MODEL_COLOR[name] }}>{name}</span>
            {' '}RMSE {row.rmse_m?.toFixed(3)} m · NSE {row.nse?.toFixed(3)} · {row.outcome_vs_persistence === 'win' ? 'beats persistence' : 'does not beat persistence'}
          </div>
        )
      })}
      {metrics.compare?.winner && metrics.compare.winner !== 'tie' && (
        <div className="text-sky">{metrics.compare.winner} lower RMSE by {metrics.compare.rmse_delta_m} m</div>
      )}
    </div>
  )
}

export default function SensorMap({ sensors, grid, selected, onSelect, stationEval, horizon = '7d', mapView = 'depth', skillModel = 'PatchTST' }) {
  const [showSurface, setShowSurface] = useState(true)
  const raster = grid?.raster
  const vmin = raster?.vmin ?? (sensors.length ? Math.min(...sensors.map((s) => s.latest_depth_m_bgl)) : null)
  const vmax = raster?.vmax ?? (sensors.length ? Math.max(...sensors.map((s) => s.latest_depth_m_bgl)) : null)
  const span = vmin != null ? Math.max(vmax - vmin, 1e-6) : 1
  const gb = grid?.properties?.bounds
  const bounds = gb ? [[gb.south, gb.west], [gb.north, gb.east]]
    : sensors.length ? [[Math.min(...sensors.map((s) => s.latitude)), Math.min(...sensors.map((s) => s.longitude))],
                         [Math.max(...sensors.map((s) => s.latitude)), Math.max(...sensors.map((s) => s.longitude))]] : null
  const center = grid?.properties?.center
    || (sensors.length ? [sensors.reduce((a, s) => a + s.latitude, 0) / sensors.length, sensors.reduce((a, s) => a + s.longitude, 0) / sensors.length] : [0, 0])
  const url = useMemo(() => (raster && vmin != null ? rasterToDataUrl(raster, vmin, vmax) : null), [raster, vmin, vmax])
  const small = sensors.length > 40
  const evalMap = stationEval?.stations || {}
  const depthMode = mapView === 'depth'
  const showIdw = showSurface && depthMode

  return (
    <div className="relative h-full min-h-[520px] overflow-hidden rounded-xl border border-rule shadow-glow">
      <MapContainer center={center} zoom={4} className="h-full w-full" scrollWheelZoom>
        <TileLayer
          attribution="&copy; OpenStreetMap contributors &copy; CARTO"
          url={`https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=${import.meta.env.VITE_CARTO_API_KEY}`}
        />
        <Pane name="surface" style={{ zIndex: 350 }}>
          {showIdw && url && bounds && gb && (
            <ImageOverlay url={url} bounds={[[gb.south, gb.west], [gb.north, gb.east]]} opacity={0.88} className="idw-surface" />
          )}
        </Pane>
        {sensors.filter((s) => s.active_anomalies > 0).map((s) => (
          <CircleMarker key={`halo-${s.sensor_id}`} center={[s.latitude, s.longitude]} radius={small ? 11 : 16} interactive={false}
            pathOptions={{ stroke: false, fillColor: SEVERITY_COLOR[s.max_active_severity] || '#f59e0b', fillOpacity: 0.22, className: 'pulse-ring' }} />
        ))}
        {sensors.map((s) => {
          const sel = s.sensor_id === selected
          const ring = SEVERITY_COLOR[s.max_active_severity] || '#e6f1ff'
          const metrics = evalMap[s.sensor_id]?.[horizon]
          const fill = stationFill(mapView, (s.latest_depth_m_bgl - vmin) / span, metrics, skillModel)
          return (
            <CircleMarker key={s.sensor_id} center={[s.latitude, s.longitude]} radius={(small ? 5 : 7) + (sel ? 3 : 0)}
              pathOptions={{ color: sel ? '#ffffff' : ring, weight: sel ? 3 : s.active_anomalies ? 2.5 : 1.5,
                             fillColor: fill, fillOpacity: 1 }}
              eventHandlers={{ click: () => onSelect(s.sensor_id) }}>
              <Tooltip>
                {s.name || s.sensor_id}: {s.latest_depth_m_bgl} m bgl
                {metrics?.[skillModel] ? ` · ${skillModel} RMSE ${metrics[skillModel].rmse_m} m` : ''}
              </Tooltip>
              <Popup>
                <div className="font-semibold text-sm">{s.name || s.sensor_id}</div>
                {s.name && <div className="text-muted">{s.sensor_id}</div>}
                <div className="mt-1">Depth <strong>{s.latest_depth_m_bgl} m bgl</strong> on {s.latest_date}</div>
                <div>Status (prototype): {STATUS_LABEL[s.status] || s.status}</div>
                <div>
                  Anomaly state:{' '}
                  {s.active_anomalies > 0
                    ? <strong style={{ color: SEVERITY_COLOR[s.max_active_severity] }}>{s.max_active_severity} ({s.active_anomalies} flagged day(s), last 30 d)</strong>
                    : 'none in the last 30 d'}
                </div>
                {s.latest_anomaly_score != null && <div>Latest anomaly score {s.latest_anomaly_score.toFixed(3)}</div>}
                <EvalRows metrics={metrics} horizon={horizon} />
                <button className="mt-2 rounded border border-aqua px-2 py-0.5 text-aqua" onClick={() => onSelect(s.sensor_id)}>Show forecast</button>
              </Popup>
            </CircleMarker>
          )
        })}
        <FitToData bounds={bounds} signature={`${sensors.length}-${grid?.properties?.as_of}-${bounds ? bounds.flat().join(',') : ''}`} />
      </MapContainer>
      {depthMode && (
        <label className="map-card right-3 top-3 flex items-center gap-2 text-ink">
          <input type="checkbox" className="accent-cyan-400" checked={showSurface} onChange={(e) => setShowSurface(e.target.checked)} />
          IDW surface
        </label>
      )}
      {!depthMode && (
        <div className="map-card right-3 top-3 max-w-[220px] text-ink">
          Map fill is {mapView === 'compare' ? 'PatchTST vs LightGBM RMSE' : mapView === 'nse' ? 'station NSE' : 'skill vs persistence'} at {horizon}.
        </div>
      )}
      <Legend view={mapView} vmin={vmin} vmax={vmax} thresholds={grid?.properties?.thresholds_m} horizon={horizon} skillModel={skillModel} />
    </div>
  )
}

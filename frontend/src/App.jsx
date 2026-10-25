import { useEffect, useMemo, useState } from 'react'
import { getAnomalies, getDataSummary, getEvaluationStations, getForecast, getHealth, getSensors, getSpatialGrid } from './api.js'
import AnomalyFeed from './components/AnomalyFeed.jsx'
import ForecastChart from './components/ForecastChart.jsx'
import Header from './components/Header.jsx'
import ModelEvaluation from './components/ModelEvaluation.jsx'
import SensorMap from './components/SensorMap.jsx'
import StationEvalPanel from './components/StationEvalPanel.jsx'
import SummaryCards from './components/SummaryCards.jsx'

const TABS = [['dashboard', 'Map & operations'], ['evaluation', 'Pooled evaluation']]
const tabFromHash = () => (window.location.hash === '#evaluation' ? 'evaluation' : 'dashboard')

export default function App() {
  const [tab, setTab] = useState(tabFromHash)
  const selectTab = (t) => { setTab(t); window.history.replaceState(null, '', t === 'evaluation' ? '#evaluation' : '#dashboard') }
  useEffect(() => {
    const onHash = () => setTab(tabFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  const [health, setHealth] = useState(null)
  const [summary, setSummary] = useState(null)
  const [sensors, setSensors] = useState([])
  const [grid, setGrid] = useState(null)
  const [anomalies, setAnomalies] = useState([])
  const [stationEval, setStationEval] = useState(null)
  const [evalError, setEvalError] = useState(null)
  const [selected, setSelected] = useState(null)
  const [forecast, setForecast] = useState(null)
  const [fcError, setFcError] = useState(null)
  const [fcLoading, setFcLoading] = useState(false)
  const [error, setError] = useState(null)
  const [onlySelected, setOnlySelected] = useState(false)
  const [horizon, setHorizon] = useState('7d')
  const [mapView, setMapView] = useState('depth')
  const [skillModel, setSkillModel] = useState('PatchTST')
  const [query, setQuery] = useState('')

  useEffect(() => {
    Promise.all([getHealth(), getDataSummary(), getSensors(), getSpatialGrid(), getAnomalies()])
      .then(([h, d, s, g, a]) => {
        setHealth(h); setSummary(d); setSensors(s); setGrid(g); setAnomalies(a)
        if (s.length) setSelected(s[0].sensor_id)
      })
      .catch((e) => setError(e.message))
    getEvaluationStations()
      .then(setStationEval)
      .catch((e) => setEvalError(e.message))
  }, [])

  useEffect(() => {
    if (!selected) return undefined
    let cancelled = false
    setFcLoading(true); setFcError(null)
    getForecast(selected)
      .then((f) => { if (!cancelled) setForecast(f) })
      .catch((e) => { if (!cancelled) { setForecast(null); setFcError(e.message) } })
      .finally(() => { if (!cancelled) setFcLoading(false) })
    return () => { cancelled = true }
  }, [selected])

  const sensor = sensors.find((s) => s.sensor_id === selected)
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return sensors
    return sensors.filter((s) => (s.sensor_id || '').toLowerCase().includes(q) || (s.name || '').toLowerCase().includes(q))
  }, [sensors, query])

  return (
    <div className="mx-auto min-h-full max-w-[1600px] space-y-4 p-4 md:p-5">
      <Header summary={summary} health={health} />

      <nav aria-label="Sections" className="flex flex-wrap items-center gap-2">
        <div className="inline-flex rounded-full border border-rule bg-deep/80 p-1">
          {TABS.map(([id, label]) => (
            <button key={id} onClick={() => selectTab(id)} aria-current={tab === id ? 'page' : undefined}
              className={`rounded-full px-4 py-1.5 text-sm font-medium ${tab === id ? 'bg-aqua/15 text-aqua' : 'text-muted hover:text-ink'}`}>
              {label}
            </button>
          ))}
        </div>
        {tab === 'dashboard' && stationEval && (
          <span className="text-xs text-muted">
            Map can colour wells by {(stationEval.models || []).join(' / ')} scores from {stationEval.source_file} ({stationEval.n_stations} evaluated stations).
          </span>
        )}
      </nav>

      {tab === 'evaluation' && <ModelEvaluation />}

      {tab === 'dashboard' && (<>
      {error && (
        <div role="alert" className="rounded-xl border border-crit/60 bg-crit/10 p-3 text-sm text-crit">
          Could not load data from the API: {error}. Run <code>python scripts/run_pipeline.py --source synthetic</code> (or <code>--source real</code>),
          then start the backend with <code>uvicorn backend.main:app --port 8000</code>.
        </div>
      )}
      {evalError && (
        <div role="status" className="rounded-xl border border-warn/40 bg-warn/5 p-3 text-sm text-ink/90">
          Map skill layers need the stored station file: {evalError}
        </div>
      )}

      <SummaryCards sensors={sensors} summary={summary} stationEval={stationEval} />

      <div className="grid gap-4 xl:grid-cols-5">
        <div className="min-h-[520px] xl:col-span-3"><SensorMap sensors={sensors} grid={grid} selected={selected} onSelect={setSelected}
          stationEval={stationEval} horizon={horizon} mapView={mapView} skillModel={skillModel} /></div>
        <div className="min-w-0 space-y-4 xl:col-span-2">
          <StationEvalPanel stationEval={stationEval} sensor={sensor} horizon={horizon} onHorizon={setHorizon}
            mapView={mapView} onMapView={setMapView} skillModel={skillModel} onSkillModel={setSkillModel} />
          <label className="block rounded-xl border border-rule bg-panel px-3 py-2 text-xs text-muted shadow-glow">
            Find a well
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="USGS id or name"
              className="mt-1 w-full rounded-md border border-rule bg-deep px-2 py-1.5 text-sm text-ink placeholder:text-muted/70" />
          </label>
          {query && (
            <ul className="scroll-dark max-h-40 overflow-y-auto rounded-xl border border-rule bg-panel text-sm shadow-glow">
              {filtered.slice(0, 20).map((s) => (
                <li key={s.sensor_id}>
                  <button className={`w-full px-3 py-2 text-left hover:bg-raised ${s.sensor_id === selected ? 'text-aqua' : ''}`}
                    onClick={() => { setSelected(s.sensor_id); setQuery('') }}>
                    {s.name || s.sensor_id} <span className="text-xs text-muted">{s.latest_depth_m_bgl} m bgl</span>
                  </button>
                </li>
              ))}
              {filtered.length === 0 && <li className="px-3 py-2 text-muted">No matching wells.</li>}
            </ul>
          )}
          <ForecastChart forecast={forecast} loading={fcLoading} error={fcError} />
        </div>
      </div>

      <AnomalyFeed anomalies={anomalies} selected={selected} onSelect={setSelected} onlySelected={onlySelected}
        setOnlySelected={setOnlySelected} source={health?.source} />
      </>)}

      <footer className="pb-4 text-center text-xs text-muted">
        AquaSense is a B.Tech prototype, not an operational groundwater-management system. Prototype depth classes are not regulatory categories.
        Map skill colours are from a finished rolling-origin experiment, not live retraining.
      </footer>
    </div>
  )
}

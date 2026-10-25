import { useEffect, useMemo, useState } from 'react'
import { Area, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

function merge(fc, historyDays, show) {
  const rows = new Map()
  const put = (date, k, v) => { if (!rows.has(date)) rows.set(date, { date }); rows.get(date)[k] = v }
  const hist = fc.history.filter((h) => h.depth_m_bgl != null).slice(-historyDays)
  hist.forEach((h) => put(h.date, 'history', h.depth_m_bgl))
  const last = hist[hist.length - 1]
  const join = (k) => last && put(last.date, k, last.depth_m_bgl)
  const seq = [
    show.patchtst && ['patchtst_7d', 'p7'],
    show.patchtst && ['patchtst_30d', 'p30'],
    show.patchtst && ['patchtst_90d', 'p90'],
    show.lstm && ['lstm_7d', 'lstm'],
  ].filter(Boolean)
  seq.forEach(([key, name]) => {
    const series = fc[key]
    if (!series) return
    join(name)
    series.forEach((p) => put(p.date, name, p.depth_m_bgl))
  })
  const bands = [
    show.lgbm && ['lgbm_30d', 'l30'],
    show.lgbm && ['lgbm_90d', 'l90'],
    show.xgb && ['xgb_30d', 'x30'],
    show.xgb && ['xgb_90d', 'x90'],
  ].filter(Boolean)
  bands.forEach(([key, name]) => {
    const f = fc[key]
    if (!f) return
    join(name)
    f.trajectory.forEach((p) => {
      put(p.date, name, p.depth_m_bgl)
      put(p.date, `${name}band`, [p.lower, p.upper])
    })
  })
  return [...rows.values()].sort((a, b) => a.date.localeCompare(b.date))
}

const Btn = ({ active, onClick, children }) => (
  <button onClick={onClick} aria-pressed={active}
    className={`rounded-md border px-2 py-1 ${active ? 'border-aqua bg-aqua/15 text-aqua' : 'border-rule bg-deep text-muted hover:text-ink'}`}>{children}</button>
)

export default function ForecastChart({ forecast, loading, error }) {
  const hasPatch = Boolean(forecast?.patchtst_7d || forecast?.patchtst_30d || forecast?.patchtst_90d)
  const hasLgbm = Boolean(forecast?.lgbm_30d || forecast?.lgbm_90d)
  const [band, setBand] = useState('l30')
  const [historyDays, setHistoryDays] = useState(120)
  const [show, setShow] = useState({ patchtst: true, lgbm: true, lstm: false, xgb: false })
  useEffect(() => {
    setShow({ patchtst: true, lgbm: true, lstm: !hasPatch, xgb: !hasLgbm })
    setBand(hasLgbm ? 'l30' : 'x30')
  }, [forecast?.sensor_id, hasPatch, hasLgbm])
  const data = useMemo(() => (forecast ? merge(forecast, historyDays, show) : []), [forecast, historyDays, show])
  const box = 'rounded-xl border border-rule bg-panel p-4 text-sm shadow-glow'
  const toggle = (k) => setShow((s) => ({ ...s, [k]: !s[k] }))

  if (error) return <div className={`${box} text-crit`}>Forecast unavailable: {error}</div>
  if (!forecast) return <div className={`${box} text-muted`}>{loading ? 'Loading forecast…' : 'Select a station on the map.'}</div>
  const l30 = forecast.lgbm_30d, l90 = forecast.lgbm_90d, x30 = forecast.xgb_30d, x90 = forecast.xgb_90d
  const bandSource = band.startsWith('l') ? (band === 'l30' ? l30 : l90) : (band === 'x30' ? x30 : x90)
  return (
    <section aria-label="Forecast" className="rounded-xl border border-rule bg-panel shadow-glow">
      <header className="flex flex-wrap items-baseline justify-between gap-2 px-4 pt-3">
        <h2 className="font-display text-lg font-semibold">
          Forecast <span className="font-sans text-xs font-normal text-muted">{forecast.sensor_id} · last observation {forecast.last_observation_date}</span>
        </h2>
        <div className="flex flex-wrap gap-1 text-xs" role="group" aria-label="Chart options">
          {[60, 120, 180].map((d) => <Btn key={d} active={historyDays === d} onClick={() => setHistoryDays(d)}>{d} d history</Btn>)}
        </div>
      </header>
      <div className="flex flex-wrap gap-1 px-4 pt-2 text-xs" role="group" aria-label="Models on chart">
        <Btn active={show.patchtst} onClick={() => toggle('patchtst')}>PatchTST</Btn>
        <Btn active={show.lgbm} onClick={() => toggle('lgbm')}>LightGBM</Btn>
        <Btn active={show.lstm} onClick={() => toggle('lstm')}>LSTM</Btn>
        <Btn active={show.xgb} onClick={() => toggle('xgb')}>XGBoost</Btn>
      </div>
      <p className="px-4 pt-2 text-xs text-sky">Higher m bgl = deeper groundwater (axis is inverted: deeper plots lower).</p>
      <div className="h-72 px-2 pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 16, bottom: 4, left: 4 }}>
            <CartesianGrid stroke="#173456" strokeDasharray="2 4" />
            <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#8ba6c6' }} stroke="#173456" minTickGap={36} />
            <YAxis reversed domain={['auto', 'auto']} tick={{ fontSize: 11, fill: '#8ba6c6' }} stroke="#173456" width={46}
              label={{ value: 'depth, m bgl', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#8ba6c6' } }} />
            <Tooltip contentStyle={{ background: '#0b1c33', border: '1px solid #173456', color: '#e6f1ff', fontSize: 12 }}
              formatter={(v) => (Array.isArray(v) ? `${v[0]} to ${v[1]} m` : `${Number(v).toFixed(2)} m bgl`)} />
            <Legend wrapperStyle={{ fontSize: 12, color: '#8ba6c6' }} />
            {show.lgbm && l30 && band === 'l30' && <Area dataKey="l30band" name="LightGBM 30 d interval" stroke="none" fill="#c4b5fd" fillOpacity={0.16} connectNulls isAnimationActive={false} />}
            {show.lgbm && l90 && band === 'l90' && <Area dataKey="l90band" name="LightGBM 90 d interval" stroke="none" fill="#c4b5fd" fillOpacity={0.16} connectNulls isAnimationActive={false} />}
            {show.xgb && x30 && band === 'x30' && <Area dataKey="x30band" name="XGBoost 30 d interval" stroke="none" fill="#38bdf8" fillOpacity={0.16} connectNulls isAnimationActive={false} />}
            {show.xgb && x90 && band === 'x90' && <Area dataKey="x90band" name="XGBoost 90 d interval" stroke="none" fill="#38bdf8" fillOpacity={0.16} connectNulls isAnimationActive={false} />}
            <Line dataKey="history" name="observed" stroke="#e6f1ff" strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />
            {show.patchtst && <Line dataKey="p7" name="PatchTST 7 d" stroke="#22d3ee" strokeWidth={3} dot={false} connectNulls isAnimationActive={false} />}
            {show.patchtst && <Line dataKey="p30" name="PatchTST 30 d" stroke="#67e8f9" strokeWidth={2} strokeDasharray="5 3" dot={false} connectNulls isAnimationActive={false} />}
            {show.patchtst && <Line dataKey="p90" name="PatchTST 90 d" stroke="#155e75" strokeWidth={2} strokeDasharray="2 3" dot={false} connectNulls isAnimationActive={false} />}
            {show.lgbm && <Line dataKey="l30" name="LightGBM 30 d" stroke="#c4b5fd" strokeWidth={2} strokeDasharray="6 3" dot={false} connectNulls isAnimationActive={false} />}
            {show.lgbm && <Line dataKey="l90" name="LightGBM 90 d" stroke="#a78bfa" strokeWidth={2} strokeDasharray="2 3" dot={false} connectNulls isAnimationActive={false} />}
            {show.lstm && <Line dataKey="lstm" name="LSTM 7 d" stroke="#818cf8" strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />}
            {show.xgb && <Line dataKey="x30" name="XGBoost 30 d" stroke="#2dd4bf" strokeWidth={2} strokeDasharray="6 3" dot={false} connectNulls isAnimationActive={false} />}
            {show.xgb && <Line dataKey="x90" name="XGBoost 90 d" stroke="#5eead4" strokeWidth={2} strokeDasharray="2 3" dot={false} connectNulls isAnimationActive={false} />}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <footer className="space-y-2 px-4 pb-3 text-xs text-muted">
        <div className="flex flex-wrap gap-1" role="group" aria-label="Interval band">
          {l30 && <Btn active={band === 'l30'} onClick={() => setBand('l30')}>LGBM 30 d band</Btn>}
          {l90 && <Btn active={band === 'l90'} onClick={() => setBand('l90')}>LGBM 90 d band</Btn>}
          {x30 && <Btn active={band === 'x30'} onClick={() => setBand('x30')}>XGB 30 d band</Btn>}
          {x90 && <Btn active={band === 'x90'} onClick={() => setBand('x90')}>XGB 90 d band</Btn>}
        </div>
        {bandSource && <div>Selected end point {bandSource.endpoint_depth_m_bgl} m ({bandSource.endpoint_lower} to {bandSource.endpoint_upper}).</div>}
        {!forecast.patchtst_7d && !forecast.patchtst_30d && <div>PatchTST trajectory not available (model file missing or PyTorch not installed). Map scores still come from the stored evaluation.</div>}
        {!forecast.lgbm_30d && <div>LightGBM forecast not available (model file missing). Map scores still come from the stored evaluation.</div>}
        <div>Direct models draw a straight line to an end-point forecast (prototype). Bands are approximate empirical 90% intervals; coverage is measured, not guaranteed.</div>
      </footer>
    </section>
  )
}

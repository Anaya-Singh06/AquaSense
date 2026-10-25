import { MAP_MODELS } from '../api.js'

function Card({ label, value, note, tone }) {
  return (
    <div className="rounded-xl border border-rule bg-panel px-4 py-3 shadow-glow">
      <div className="text-[11px] uppercase tracking-wider text-muted">{label}</div>
      <div className={`font-display text-2xl font-semibold leading-tight ${tone || 'text-ink'}`}>{value}</div>
      <div className="text-xs text-muted">{note}</div>
    </div>
  )
}

export default function SummaryCards({ sensors, summary, stationEval }) {
  const deep = sensors.filter((s) => s.status === 'CRITICAL').length
  const activeDays = sensors.reduce((a, s) => a + s.active_anomalies, 0)
  const stationsFlagged = sensors.filter((s) => s.active_anomalies > 0).length
  const avg = sensors.length ? sensors.reduce((a, s) => a + s.latest_depth_m_bgl, 0) / sensors.length : null
  const range = summary?.date_range ? `${summary.date_range[0]} to ${summary.date_range[1]}` : '—'
  const wins7 = stationEval?.summary?.['7d']
  const primary = MAP_MODELS.find((m) => wins7?.[m])
  const p7 = wins7?.[primary]
  return (
    <section aria-label="Summary" className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
      <Card label="Monitored stations" value={summary?.station_count ?? sensors.length}
        note={summary ? `${summary.observation_count.toLocaleString()} observations` : ''} />
      <Card label="Active anomalies" value={activeDays} tone={activeDays ? 'text-warn' : 'text-ink'}
        note={`${stationsFlagged} station(s), last 30 d`} />
      <Card label="Deep stations" value={deep} tone={deep ? 'text-high' : 'text-ink'} note="20 m bgl or more (prototype class)" />
      <Card label="Average depth" value={avg == null ? '—' : `${avg.toFixed(1)} m`} note="m bgl, latest; higher = deeper" />
      <Card label="Date range" value={<span className="text-base">{range}</span>}
        note={summary ? `${(summary.missing_fraction * 100).toFixed(1)}% of days not observed` : ''} />
      {p7
        ? <Card label={`${primary} 7 d wins`} value={`${p7.station_wins}/${p7.stations}`} tone="text-aqua"
            note={`${MAP_MODELS.filter((m) => m !== primary && wins7[m]).map((m) => `${m} ${wins7[m].station_wins}/${wins7[m].stations}`).join(' · ')} vs persistence`} />
        : <Card label="Data source" value={<span className="text-base text-aqua">{summary?.source_label ?? '—'}</span>}
            note={summary?.source === 'real' ? 'climate: gridded, not at the well' : summary ? 'simulated stations' : ''} />}
    </section>
  )
}

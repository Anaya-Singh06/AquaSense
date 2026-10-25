import { MAP_MODELS, MODEL_COLOR } from '../api.js'

const num = (v, d = 3) => (v == null ? '—' : Number(v).toFixed(d))
const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)
const hWord = { '7d': '7-day', '30d': '30-day', '90d': '90-day' }

function Chip({ children, tone = 'muted' }) {
  const cls = { aqua: 'border-aqua/40 bg-aqua/10 text-aqua', warn: 'border-warn/40 bg-warn/10 text-warn',
    muted: 'border-rule bg-deep text-muted', violet: 'border-indigoish/40 bg-indigoish/10 text-indigoish' }
  return <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${cls[tone]}`}>{children}</span>
}

function ModelBlock({ name, row }) {
  if (!row) return (
    <div className="rounded-lg border border-rule bg-deep/50 p-3">
      <div className="text-sm font-semibold">{name}</div>
      <p className="mt-1 text-xs text-muted">No score for this station at this horizon (not in the all-models comparison set).</p>
    </div>
  )
  const win = row.outcome_vs_persistence === 'win'
  return (
    <div className="rounded-lg border border-rule bg-deep/50 p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-sm font-semibold">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: MODEL_COLOR[name] }} />
          {name}
        </div>
        <Chip tone={win ? 'aqua' : 'warn'}>{win ? 'beats persistence' : 'loses to persistence'}</Chip>
      </div>
      <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
        <div>RMSE <span className="float-right tabular-nums text-ink">{num(row.rmse_m, 3)} m</span></div>
        <div>vs persist. <span className="float-right tabular-nums text-ink">{num(row.rmse_persistence, 3)} m</span></div>
        <div>NSE <span className="float-right tabular-nums text-ink">{num(row.nse)}</span></div>
        <div>RMSE skill <span className="float-right tabular-nums text-ink">{pct(row.rmse_skill)}</span></div>
      </dl>
    </div>
  )
}

export default function StationEvalPanel({ stationEval, sensor, horizon, onHorizon, mapView, onMapView, skillModel, onSkillModel }) {
  const sid = sensor?.sensor_id
  const block = sid && stationEval?.stations?.[sid]?.[horizon]
  const cmp = block?.compare
  const summary = stationEval?.summary?.[horizon]
  const models = stationEval?.models || MAP_MODELS
  const views = [
    ['depth', 'Depth'],
    ['nse', 'NSE'],
    ['skill', 'vs persist.'],
    ['compare', 'PatchTST vs LightGBM'],
  ]
  return (
    <section className="rounded-xl border border-rule bg-panel p-4 shadow-glow">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h2 className="font-display text-lg font-semibold leading-tight">{sensor?.name || sid || 'Select a station'}</h2>
          {sid && sensor?.name && <p className="text-xs text-muted">{sid}</p>}
          <p className="mt-1 text-xs text-muted">
            Offline rolling-origin scores for {models.join(', ')}. Colour the map by these results, or stay on groundwater depth.
          </p>
        </div>
        <Chip tone="aqua">offline</Chip>
      </div>

      <div className="mt-3 flex flex-wrap gap-1" role="group" aria-label="Forecast horizon">
        {(stationEval?.horizons || ['7d', '30d', '90d']).map((h) => (
          <button key={h} onClick={() => onHorizon(h)} aria-pressed={h === horizon}
            className={`rounded-md border px-2.5 py-1 text-xs font-medium ${h === horizon ? 'border-aqua bg-aqua/15 text-aqua' : 'border-rule bg-deep text-muted hover:text-ink'}`}>
            {hWord[h] || h}
          </button>
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-1" role="group" aria-label="Map colouring">
        {views.map(([id, label]) => (
          <button key={id} onClick={() => onMapView(id)} aria-pressed={id === mapView}
            className={`rounded-md border px-2.5 py-1 text-xs font-medium ${id === mapView ? 'border-aqua bg-aqua/15 text-aqua' : 'border-rule bg-deep text-muted hover:text-ink'}`}>
            {label}
          </button>
        ))}
      </div>
      {(mapView === 'nse' || mapView === 'skill') && (
        <div className="mt-2 flex flex-wrap gap-1" role="group" aria-label="Score model">
          {models.map((name) => (
            <button key={name} onClick={() => onSkillModel(name)} aria-pressed={name === skillModel}
              className={`rounded-md border px-2.5 py-1 text-xs font-medium ${name === skillModel ? 'border-aqua bg-aqua/15 text-aqua' : 'border-rule bg-deep text-muted hover:text-ink'}`}>
              Colour by {name}
            </button>
          ))}
        </div>
      )}

      {summary && (
        <p className="mt-3 text-xs text-muted">
          At {hWord[horizon]} on {summary.compare?.stations ?? 0} comparable wells, PatchTST had the lower RMSE at {summary.compare?.patchtst_lower_rmse ?? 0}
          {' '}and LightGBM at {summary.compare?.lightgbm_lower_rmse ?? 0}. Beat persistence:{' '}
          {models.filter((m) => summary[m]).map((m) => `${m} ${summary[m].station_wins ?? 0}/${summary[m].stations ?? 0}`).join(' · ')}.
        </p>
      )}

      {!stationEval && <p className="mt-3 text-sm text-muted">Station evaluation is not loaded.</p>}
      {stationEval && sid && !block && (
        <p className="mt-3 text-sm text-muted">This well is not in the stored model comparison (synthetic sites, or a USGS well dropped from a horizon).</p>
      )}
      {block && (
        <div className="mt-3 space-y-2">
          {cmp && (
            <div className="rounded-lg border border-aqua/30 bg-aqua/5 px-3 py-2 text-xs">
              {cmp.winner === 'tie'
                ? 'PatchTST and LightGBM have the same RMSE here.'
                : <><span className="font-semibold text-ink">{cmp.winner}</span> has the lower RMSE by {num(cmp.rmse_delta_m, 3)} m at this well.</>}
            </div>
          )}
          {models.map((name) => <ModelBlock key={name} name={name} row={block[name]} />)}
        </div>
      )}
    </section>
  )
}

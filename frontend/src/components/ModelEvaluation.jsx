import { useEffect, useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { getEvaluation, MODEL_COLOR } from '../api.js'

// All numbers come from GET /api/evaluation (one JSON source of truth). Nothing is hard-coded here except
// presentation choices (colours, model order) and general explanations of what the metrics mean.
const MODEL_ORDER = ['PatchTST', 'iTransformer', 'LSTM', 'XGBoost', 'LightGBM', 'Persistence']
const AXIS = { fontSize: 11, fill: '#8ba6c6' }
const TIP = { background: '#0b1c33', border: '1px solid #173456', color: '#e6f1ff', fontSize: 12 }
const HORIZON_WORD = { '7d': '7-day', '30d': '30-day', '90d': '90-day' }
const hWord = (h) => HORIZON_WORD[h] || h

const num = (v, d = 3) => (v == null ? '-' : Number(v).toFixed(d))
const pct = (v) => (v == null ? '-' : `${(v * 100).toFixed(1)}%`)
const metres = (v) => (v == null ? '-' : `${Number(v).toFixed(3)} m`)
const listOf = (a) => (a.length <= 1 ? a.join('') : `${a.slice(0, -1).join(', ')} and ${a[a.length - 1]}`)

function rowsFor(block) {
  const rows = Object.entries(block.models).map(([name, m]) => ({ name, baseline: false, ...m }))
  rows.push({ name: 'Persistence', baseline: true, ...block.persistence })
  const rank = (n) => (MODEL_ORDER.includes(n) ? MODEL_ORDER.indexOf(n) : 99)
  return rows.sort((a, b) => rank(a.name) - rank(b.name))
}

const Panel = ({ title, subtitle, children, className = '' }) => (
  <section className={`rounded-xl border border-rule bg-panel p-4 shadow-glow ${className}`}>
    {title && <h2 className="font-display text-lg font-semibold">{title}</h2>}
    {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
    <div className={title ? 'mt-3' : ''}>{children}</div>
  </section>
)

const Stat = ({ label, value, note, tone }) => (
  <div className="rounded-lg border border-rule bg-panel px-4 py-3 shadow-glow">
    <div className="text-[11px] uppercase tracking-wider text-muted">{label}</div>
    <div className={`font-display text-xl font-semibold leading-tight ${tone || 'text-ink'}`}>{value}</div>
    <div className="text-xs text-muted">{note}</div>
  </div>
)

function MetricChart({ title, note, data, dataKey, unit, zeroLine, best }) {
  return (
    <div className="min-w-0">
      <h3 className="text-sm font-semibold text-ink">{title}</h3>
      <p className="mb-1 text-xs text-muted">{note}</p>
      <div className="h-56" role="img" aria-label={`${title} by model`}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#173456" strokeDasharray="2 4" vertical={false} />
            <XAxis dataKey="name" tick={AXIS} stroke="#173456" interval={0} />
            <YAxis tick={AXIS} stroke="#173456" width={44} />
            <Tooltip contentStyle={TIP} cursor={{ fill: 'rgba(34,211,238,0.06)' }}
              formatter={(v) => [`${Number(v).toFixed(3)}${unit}`, title]} />
            {zeroLine && <ReferenceLine y={0} stroke="#8ba6c6" />}
            <Bar dataKey={dataKey} isAnimationActive={false} radius={[3, 3, 0, 0]}>
              {data.map((r) => (
                <Cell key={r.name} fill={MODEL_COLOR[r.name] || '#38bdf8'} fillOpacity={r.baseline ? 0.55 : 1}
                  stroke={best && r.name === best ? '#e6f1ff' : 'none'} strokeWidth={2} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function NseChart({ data }) {
  return (
    <div className="min-w-0">
      <h3 className="text-sm font-semibold text-ink">Station NSE: mean vs median</h3>
      <p className="mb-1 text-xs text-muted">Higher is better; 1 is perfect, below 0 is worse than predicting each well's average level.</p>
      <div className="h-56" role="img" aria-label="Mean and median station NSE by model">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#173456" strokeDasharray="2 4" vertical={false} />
            <XAxis dataKey="name" tick={AXIS} stroke="#173456" interval={0} />
            <YAxis tick={AXIS} stroke="#173456" width={44} />
            <Tooltip contentStyle={TIP} cursor={{ fill: 'rgba(34,211,238,0.06)' }} formatter={(v, n) => [Number(v).toFixed(3), n]} />
            <Legend wrapperStyle={{ fontSize: 12, color: '#8ba6c6' }} />
            <ReferenceLine y={0} stroke="#8ba6c6" />
            <Bar dataKey="mean_station_nse" name="mean station NSE" fill="#22d3ee" isAnimationActive={false} radius={[3, 3, 0, 0]} />
            <Bar dataKey="median_station_nse" name="median station NSE" fill="#818cf8" isAnimationActive={false} radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function ResultsTable({ rows, block }) {
  const bestOf = (k, lowest) => {
    const vals = rows.map((r) => r[k]).filter((v) => v != null)
    return vals.length ? (lowest ? Math.min(...vals) : Math.max(...vals)) : null
  }
  const best = { rmse_m: bestOf('rmse_m', true), mae_m: bestOf('mae_m', true), mean_station_nse: bestOf('mean_station_nse', false),
    median_station_nse: bestOf('median_station_nse', false), rmse_skill: bestOf('rmse_skill', false) }
  const cell = (r, k, text) => (
    <td className={`px-3 py-2 text-right tabular-nums ${r[k] != null && r[k] === best[k] ? 'font-semibold text-aqua' : ''}`}>{text}</td>
  )
  const th = 'px-3 py-2 text-right font-medium'
  return (
    <div className="scroll-dark overflow-x-auto">
      <table className="w-full min-w-[820px] text-sm">
        <caption className="sr-only">Forecast accuracy by model for the selected horizon</caption>
        <thead className="text-[11px] uppercase tracking-wider text-muted">
          <tr className="border-b border-rule">
            <th scope="col" className="px-3 py-2 text-left font-medium">Model</th>
            <th scope="col" className={th}>RMSE (m)</th>
            <th scope="col" className={th}>MAE (m)</th>
            <th scope="col" className={th}>Mean station NSE</th>
            <th scope="col" className={th}>Median station NSE</th>
            <th scope="col" className={th}>RMSE skill vs persistence</th>
            <th scope="col" className={th}>Stations better / worse</th>
            <th scope="col" className={th}>Folds better</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name} className={`border-b border-rule/60 ${r.baseline ? 'bg-deep/60 text-muted' : 'hover:bg-raised/50'}`}>
              <th scope="row" className="px-3 py-2 text-left font-medium">
                <span className="mr-2 inline-block h-2.5 w-2.5 rounded-sm align-middle" style={{ background: MODEL_COLOR[r.name] || '#38bdf8' }} />
                {r.name}{r.baseline && <span className="ml-2 text-[10px] uppercase tracking-wider">baseline</span>}
              </th>
              {cell(r, 'rmse_m', num(r.rmse_m, 4))}
              {cell(r, 'mae_m', num(r.mae_m, 4))}
              {cell(r, 'mean_station_nse', num(r.mean_station_nse))}
              {cell(r, 'median_station_nse', num(r.median_station_nse))}
              {r.baseline
                ? <td className="px-3 py-2 text-right" colSpan={3}>reference forecast (no skill, win or fold counts)</td>
                : (<>
                  {cell(r, 'rmse_skill', pct(r.rmse_skill))}
                  <td className="px-3 py-2 text-right tabular-nums">{r.station_wins} / {r.station_total - r.station_wins}
                    <span className="ml-1 text-xs text-muted">of {r.station_total}</span></td>
                  <td className="px-3 py-2 text-right tabular-nums">{r.folds_better} / {r.folds_total}</td>
                </>)}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-muted">
        Based on {block.sample_rows.toLocaleString()} forecast evaluations across {block.stations} stations. Highlighted values are the best in each column.
        &ldquo;Stations better / worse&rdquo; counts stations where the model&rsquo;s RMSE was lower / not lower than persistence.
      </p>
    </div>
  )
}

// Plain-language findings. Model names and numbers are looked up in the API payload, never typed in.
function buildFindings(data) {
  const out = []
  for (const h of data.horizons) {
    const b = data.metrics[h], d = data.derived[h]
    const top = d.best_rmse, m = b.models[top]
    const sweep = [d.best_mae, d.best_mean_nse, d.best_median_nse].every((x) => x === top)
    out.push({
      h,
      title: sweep ? `${hWord(h)}: ${top} was best on every pooled and station-level accuracy measure`
        : `${hWord(h)}: ${top} had the lowest pooled RMSE`,
      text: `${top} reached an RMSE of ${metres(m.rmse_m)} against ${metres(b.persistence.rmse_m)} for persistence (${pct(m.rmse_skill)} lower), `
        + `and was closer than persistence at ${m.station_wins} of ${m.station_total} stations.`
        + (sweep ? '' : ` It did not lead on every measure: the best mean station NSE was ${d.best_mean_nse}, the best median was ${d.best_median_nse}.`),
    })
  }
  return out
}

function stationCaveats(data) {
  const lines = []
  for (const h of data.horizons) {
    const d = data.derived[h], b = data.metrics[h]
    if (d.pooled_better_station_worse.length) {
      lines.push(`${hWord(h)}: ${listOf(d.pooled_better_station_worse)} improved pooled RMSE over persistence, yet scored a lower mean station NSE than persistence (${num(b.persistence.mean_station_nse)}).`)
    }
  }
  return lines
}

function Interpretation({ data }) {
  const findings = useMemo(() => buildFindings(data), [data])
  const caveats = useMemo(() => stationCaveats(data), [data])
  const longH = data.horizons[data.horizons.length - 1]
  const lb = data.metrics[longH], ld = data.derived[longH]
  const negative = ld.mean_nse_below_zero
  return (
    <Panel title="What the results tell us" subtitle="Plain-language reading of the completed evaluation.">
      <ul className="space-y-3 text-sm">
        {findings.map((f) => (
          <li key={f.h} className="rounded border border-rule bg-deep/60 p-3">
            <div className="font-semibold text-sky">{f.title}</div>
            <div className="mt-1 text-ink/90">{f.text}</div>
          </li>
        ))}
      </ul>
      <div className="mt-4 rounded border border-warn/40 bg-warn/5 p-3 text-sm">
        <div className="font-semibold text-warn">A better average does not mean every well benefits</div>
        <p className="mt-1 text-ink/90">
          RMSE is pooled: all errors from all stations are combined into one number, so wells with large, hard-to-predict swings dominate it. NSE is calculated per station and then
          averaged, so it shows how the typical well fares. When pooled RMSE improves but station NSE does not, the model is helping on average while some wells are forecast poorly.
        </p>
        {caveats.length > 0 && <ul className="mt-2 list-disc space-y-1 pl-5 text-ink/90">{caveats.map((c) => <li key={c}>{c}</li>)}</ul>}
        {negative.length > 0 && (
          <p className="mt-2 text-ink/90">
            At {hWord(longH)}, mean station NSE is below zero for {listOf(negative)}
            {ld.persistence_mean_nse_below_zero ? ` (and for persistence, ${num(lb.persistence.mean_station_nse)})` : ''}, meaning that at many wells the forecast is worse than simply assuming
            that well&rsquo;s average level. {ld.best_rmse} has the lowest pooled RMSE there, but its mean station NSE is {num(lb.models[ld.best_rmse].mean_station_nse)}, so long-horizon forecasts remain unreliable.
          </p>
        )}
      </div>
      <p className="mt-3 text-sm text-ink/90">
        Forecasting gets harder as the horizon grows: pooled RMSE rises for every model from the shortest to the longest horizon.
        Pick a horizon above to compare the models directly.
      </p>
    </Panel>
  )
}

const GLOSSARY = [
  ['RMSE (m)', 'Typical size of the forecast error in metres of groundwater depth. Big misses count extra. Lower is better.'],
  ['MAE (m)', 'Average absolute error in metres. Easier to read than RMSE and less sensitive to rare large misses. Lower is better.'],
  ['NSE (Nash-Sutcliffe)', 'Compares the forecast with simply predicting that well\u2019s long-run average. 1 is perfect, 0 is no better than the average, below 0 is worse. We report it per station, then take the mean and the median across stations.'],
  ['Persistence baseline', 'The simplest forecast: "the level in the future will equal today\u2019s level". A model is only useful if it beats this.'],
  ['RMSE skill vs persistence', 'How much lower the model\u2019s RMSE is than persistence. 16% means 16% less error; 0% means no improvement.'],
  ['Stations better / worse', 'Number of wells where the model\u2019s RMSE was lower (better) or not lower (worse) than persistence.'],
  ['Folds better', 'Number of the five rolling-origin test windows in which the model beat persistence. Folds share training history, so they are not independent repeats.'],
]

function Glossary() {
  return (
    <Panel title="How to read the metrics" subtitle="No machine-learning background needed.">
      <dl className="grid gap-3 md:grid-cols-2">
        {GLOSSARY.map(([t, d]) => (
          <div key={t} className="rounded border border-rule bg-deep/60 p-3">
            <dt className="text-sm font-semibold text-sky">{t}</dt>
            <dd className="mt-1 text-sm text-ink/90">{d}</dd>
          </div>
        ))}
      </dl>
    </Panel>
  )
}

function Methodology({ data }) {
  const d = data.data, e = data.evaluation
  const range = `${d.date_start} to ${d.date_end}`
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Panel title="Methodology">
        <ul className="list-disc space-y-1.5 pl-5 text-sm text-ink/90">
          <li><b>Data:</b> {d.stations_selected} real USGS groundwater monitoring stations, {d.rows.toLocaleString()} daily rows ({range}), combined with NASA POWER environmental estimates (gridded, not measured at the well).</li>
          <li><b>Rolling-origin evaluation:</b> {e.method}. The model is trained on the past, tested on the next window, then the training period is extended and the test is repeated.</li>
          <li><b>Test period:</b> {e.test_period}.</li>
          <li><b>Scoring:</b> {e.scoring}: each forecast is judged on the single day at the end of its horizon, not averaged over every day in between. Compare numbers only within the same horizon.</li>
          <li><b>Baseline:</b> persistence (the last observed level carried forward).</li>
          <li><b>Seeds:</b> {e.seed_policy}.</li>
        </ul>
      </Panel>
      <Panel title="Limitations">
        <ul className="list-disc space-y-1.5 pl-5 text-sm text-ink/90">
          <li>{e.warning ? e.warning.charAt(0).toUpperCase() + e.warning.slice(1) : 'Folds are not independent replications'}, and each model/fold used one random seed, so small differences between models may not be meaningful.</li>
          <li>Station counts differ by horizon and model comparison, so rows are comparable within a horizon only.</li>
          <li>The data contain missing observations and long gaps, and station coverage is geographically concentrated. These results do not establish performance in other regions or deployment readiness.</li>
          <li>Pooled metrics are dominated by wells with large variation; always read them together with station-level NSE, wins and losses.</li>
          <li>Long-horizon (90-day) forecasts are unreliable at many individual wells even where the pooled error is lower than persistence.</li>
          <li>Per-fold scores and training curves are not part of this summary and are not shown.</li>
        </ul>
      </Panel>
    </div>
  )
}

export default function ModelEvaluation() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [horizon, setHorizon] = useState(null)
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    let cancelled = false
    setLoading(true); setError(null)
    getEvaluation()
      .then((d) => { if (!cancelled) { setData(d); setHorizon((cur) => (cur && d.horizons.includes(cur) ? cur : d.horizons[0])) } })
      .catch((e) => { if (!cancelled) setError(e.message) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [attempt])

  if (loading && !data) {
    return <div role="status" className="rounded-lg border border-rule bg-panel p-6 text-sm text-muted shadow-glow">Loading evaluation results...</div>
  }
  if (error && !data) {
    return (
      <div role="alert" className="rounded-lg border border-crit/60 bg-crit/10 p-4 text-sm text-crit">
        Could not load the evaluation results: {error}.
        <div className="mt-1 text-xs text-ink/80">Check that the backend is running and that <code>evaluation_reference/rolling_comparison_summary.json</code> exists. Operational model files are not needed.</div>
        <button onClick={() => setAttempt((n) => n + 1)} className="mt-2 rounded border border-crit/60 px-2 py-1 text-xs text-ink hover:bg-crit/20">Retry</button>
      </div>
    )
  }
  const block = data.metrics[horizon]
  const rows = rowsFor(block)
  const d = data.derived[horizon]
  const winner = block.models[d.best_rmse]

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-aqua/40 bg-aqua/5 px-4 py-3 text-sm" role="note">
        <span className="mr-2 rounded-full border border-aqua/50 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wider text-aqua">Offline results</span>
        {data.kind_label}. These figures were produced by a finished five-fold rolling-origin experiment and are read from a stored summary file
        (<code className="text-xs">{data.source_file}</code>); nothing on this page is trained or computed live.
      </div>

      <Panel title="Evaluation overview" subtitle={data.title}>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
          <Stat label="Stations" value={data.data.stations_selected} note="real USGS groundwater wells" />
          <Stat label="Daily rows" value={data.data.rows.toLocaleString()} note={`${data.data.date_start} to ${data.data.date_end}`} />
          <Stat label="Environment data" value={<span className="text-base text-aqua">NASA POWER</span>} note="gridded climate estimates" />
          <Stat label="Method" value={<span className="text-base">5-fold rolling origin</span>} note="expanding training window" />
          <Stat label="Test period" value={<span className="text-base">{data.evaluation.test_period}</span>} note="held-out, forward in time" />
          <Stat label="Horizons" value={<span className="text-base">{data.horizons.map((h) => h.replace('d', '')).join(', ')} days</span>} note="forecast endpoint scored" />
        </div>
      </Panel>

      <Panel>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="font-display text-lg font-semibold">Results by forecast horizon</h2>
            <p className="text-xs text-muted">Choose how far ahead the models forecast.</p>
          </div>
          <div role="group" aria-label="Forecast horizon" className="flex gap-1">
            {data.horizons.map((h) => (
              <button key={h} onClick={() => setHorizon(h)} aria-pressed={h === horizon}
                className={`rounded border px-3 py-1.5 text-sm font-medium ${h === horizon ? 'border-aqua bg-aqua/15 text-aqua' : 'border-rule bg-deep text-muted hover:text-ink'}`}>
                {hWord(h)}
              </button>
            ))}
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">
          <Stat label="Sample size" value={block.sample_rows.toLocaleString()} note="forecast evaluations (station-days)" />
          <Stat label="Stations" value={block.stations} note={`in the ${hWord(horizon)} comparison`} />
          <Stat label="Lowest pooled RMSE" value={<span className="text-aqua">{d.best_rmse}</span>} note={`${metres(winner.rmse_m)} vs ${metres(block.persistence.rmse_m)} persistence`} />
          <Stat label="Best mean station NSE" value={<span className="text-aqua">{d.best_mean_nse}</span>} note={`${num(block.models[d.best_mean_nse].mean_station_nse)} (persistence ${num(block.persistence.mean_station_nse)})`} />
        </div>

        <div className="mt-5 grid gap-5 lg:grid-cols-3">
          <MetricChart title="RMSE (m)" note="Typical error size. Lower is better. White outline = best model." data={rows} dataKey="rmse_m" unit=" m" best={d.best_rmse} />
          <MetricChart title="MAE (m)" note="Average absolute error. Lower is better." data={rows} dataKey="mae_m" unit=" m" best={d.best_mae} />
          <NseChart data={rows} />
        </div>

        <div className="mt-5">
          <ResultsTable rows={rows} block={block} />
        </div>
        {d.pooled_better_station_worse.length > 0 && (
          <p className="mt-3 rounded border border-warn/40 bg-warn/5 p-2 text-xs text-ink/90">
            <b className="text-warn">Note:</b> at {hWord(horizon)}, {listOf(d.pooled_better_station_worse)} beat persistence on pooled RMSE but have a lower mean station NSE than persistence.
          </p>
        )}
      </Panel>

      <Interpretation data={data} />
      <Glossary />
      <Methodology data={data} />
    </div>
  )
}

import { SEVERITY_COLOR } from '../api.js'

export default function AnomalyFeed({ anomalies, selected, onSelect, onlySelected, setOnlySelected, source }) {
  const rows = onlySelected && selected ? anomalies.filter((a) => a.sensor_id === selected) : anomalies
  return (
    <section aria-label="Anomaly feed" className="flex min-h-0 flex-col rounded-xl border border-rule bg-panel shadow-glow">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-rule px-4 py-3">
        <h2 className="font-display text-lg font-semibold">Potential abnormal groundwater behaviour</h2>
        <label className="flex items-center gap-2 text-xs text-muted">
          <input type="checkbox" className="accent-cyan-400" checked={onlySelected} onChange={(e) => setOnlySelected(e.target.checked)} />
          selected station only
        </label>
      </header>
      <ul className="scroll-dark max-h-96 divide-y divide-rule overflow-y-auto">
        {rows.length === 0 && <li className="px-4 py-3 text-sm text-muted">No flagged days.</li>}
        {rows.slice(0, 80).map((a) => (
          <li key={`${a.sensor_id}-${a.date}`}>
            <button className="w-full px-4 py-2.5 text-left hover:bg-raised" onClick={() => onSelect(a.sensor_id)}>
              <div className="flex items-center gap-2 text-sm">
                <span className="rounded px-1.5 py-0.5 text-[10px] font-bold tracking-wider text-abyss" style={{ background: SEVERITY_COLOR[a.severity] }}>
                  {a.severity}
                </span>
                <span className="font-semibold">{a.sensor_id}</span>
                <span className="text-muted">{a.date}</span>
                <span className="ml-auto flex items-center gap-2 text-xs text-muted">
                  score {a.anomaly_score.toFixed(3)}
                  <span className="h-1.5 w-14 overflow-hidden rounded-full bg-deep">
                    <span className="block h-full" style={{ width: `${Math.round(a.anomaly_score * 100)}%`, background: SEVERITY_COLOR[a.severity] }} />
                  </span>
                </span>
              </div>
              <div className="mt-1 text-xs text-muted">{a.reason.replace('potential abnormal groundwater drawdown: ', '')}</div>
            </button>
          </li>
        ))}
      </ul>
      <footer className="space-y-1 border-t border-rule px-4 py-2 text-xs text-muted">
        <div>Statistical outliers from an unsupervised model plus hydrological rules. They are not proof of illegal extraction.</div>
        <div>
          {source === 'real'
            ? 'Real USGS data has no verified anomaly labels: counts, dates, scores and severities only.'
            : 'Synthetic demonstration: anomalies are injected on purpose; see the synthetic anomaly evaluation in the README.'}
        </div>
      </footer>
    </section>
  )
}

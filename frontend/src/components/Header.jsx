export default function Header({ summary, health }) {
  const badge = summary?.source_badge || health?.source_label || null
  const synthetic = (summary?.source || health?.source) === 'synthetic'
  const loaded = health ? Object.entries(health.models_loaded).filter(([, v]) => v).map(([k]) => k) : []
  const featured = loaded.filter((k) => k.startsWith('patchtst') || k.startsWith('lgbm'))
  return (
    <header className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-rule bg-deep/85 px-5 py-4 shadow-glow backdrop-blur">
      <div className="flex items-center gap-3">
        <svg width="38" height="38" viewBox="0 0 32 32" aria-hidden="true">
          <defs>
            <linearGradient id="drop" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stopColor="#7dd3fc" /><stop offset="1" stopColor="#0891b2" />
            </linearGradient>
          </defs>
          <path d="M16 3 C16 3 6 14 6 20 a10 10 0 0 0 20 0 C26 14 16 3 16 3Z" fill="url(#drop)" />
          <path d="M9 22 q3.5 -3 7 0 t7 0" stroke="#040b17" strokeWidth="1.6" fill="none" strokeLinecap="round" opacity="0.65" />
        </svg>
        <div>
          <h1 className="font-display text-2xl font-bold leading-tight tracking-tight">AquaSense</h1>
          <p className="text-sm text-muted">Groundwater intelligence — depth, anomalies, and forecast skill on the map</p>
        </div>
      </div>
      <div className="flex flex-col items-end gap-1.5 text-xs text-muted">
        {badge && (
          <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-[11px] font-semibold tracking-wider ${
            synthetic ? 'border-sky/40 bg-sky/10 text-sky' : 'border-aqua/50 bg-aqua/10 text-aqua'}`}>
            <span className={`h-2 w-2 rounded-full ${synthetic ? 'bg-sky' : 'bg-aqua animate-pulse'}`} />
            {badge.toUpperCase()}
          </span>
        )}
        {health && (
          <span>
            live models: {loaded.length ? loaded.join(', ') : 'none'}
            {featured.length ? '' : ' · map skill layer uses the stored offline evaluation'}
          </span>
        )}
      </div>
    </header>
  )
}

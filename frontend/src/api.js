// Thin client for the AquaSense FastAPI backend (proxied by Vite in development) + shared colour helpers.
async function get(path) {
  const res = await fetch(path)
  if (!res.ok) {
    let detail = res.statusText
    try { detail = (await res.json()).detail || detail } catch { /* ignore */ }
    throw new Error(`${res.status} ${detail}`)
  }
  return res.json()
}

export const getHealth = () => get('/api/health')
export const getSensors = () => get('/api/sensors')
export const getForecast = (id) => get(`/api/forecast/${encodeURIComponent(id)}`)
export const getAnomalies = (limit = 200) => get(`/api/anomalies?limit=${limit}`)
export const getSpatialGrid = () => get('/api/spatial-grid')
export const getDataSummary = () => get('/api/data-summary')
export const getEvaluation = () => get('/api/evaluation')
export const getEvaluationStations = () => get('/api/evaluation/stations')

export const MODEL_COLOR = {
  PatchTST: '#22d3ee', LSTM: '#818cf8', XGBoost: '#2dd4bf', LightGBM: '#c4b5fd', iTransformer: '#f472b6',
  Persistence: '#8ba6c6',
}

// Models the Map tab colours by (must match backend MAP_MODELS). Order drives legends and fallbacks.
export const MAP_MODELS = ['PatchTST', 'LightGBM', 'iTransformer']

// Depth colour ramp: shallow groundwater (low m bgl) -> deep groundwater (high m bgl).
// Water-themed (cyan -> teal -> blue -> indigo); deliberately not a red/yellow "weather" heatmap.
export const DEPTH_RAMP = [
  [0.0, [165, 243, 252]], // light cyan  - shallow
  [0.25, [34, 211, 238]], // cyan
  [0.5, [45, 140, 220]],  // blue
  [0.75, [79, 70, 229]],  // indigo
  [1.0, [88, 28, 135]],   // deep violet - deep
]

export function rampRgb(t) {
  const x = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0))
  for (let i = 1; i < DEPTH_RAMP.length; i++) {
    const [t1, c1] = DEPTH_RAMP[i]
    const [t0, c0] = DEPTH_RAMP[i - 1]
    if (x <= t1) {
      const u = (x - t0) / (t1 - t0)
      return c0.map((v, k) => Math.round(v + (c1[k] - v) * u))
    }
  }
  return DEPTH_RAMP[DEPTH_RAMP.length - 1][1]
}
export const rampCss = (t) => `rgb(${rampRgb(t).join(',')})`
export const RAMP_GRADIENT = `linear-gradient(90deg, ${DEPTH_RAMP.map(([t, c]) => `rgb(${c.join(',')}) ${t * 100}%`).join(', ')})`

// Prototype classes (NOT regulatory): thresholds come from the backend grid properties when present.
export const STATUS_LABEL = { SAFE: 'Shallow (prototype)', SEMI_CRITICAL: 'Moderate (prototype)', CRITICAL: 'Deep (prototype)' }
// Red/orange are reserved for anomalies and warnings.
export const SEVERITY_COLOR = { LOW: '#38bdf8', MEDIUM: '#f59e0b', HIGH: '#f97316', CRITICAL: '#ef4444' }
export const SEVERITY_ORDER = { LOW: 0, MEDIUM: 1, HIGH: 2, CRITICAL: 3 }

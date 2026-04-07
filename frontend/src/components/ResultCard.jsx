import { getAbsoluteUrl } from '../services/api'

function formatConfidence(value) {
  if (value == null || Number.isNaN(Number(value))) return '—'
  const n = Number(value)
  if (n <= 1) return `${(n * 100).toFixed(1)}%`
  return `${n.toFixed(1)}%`
}

function AnalysisLoadingPanel() {
  const steps = [
    'Preprocessing image',
    'Running MobileNetV2 inference',
    'Computing Grad-CAM',
    'Building PDF report',
  ]

  return (
    <section
      className="rounded-2xl border border-brand-200/80 bg-gradient-to-b from-white to-slate-50 p-6 shadow-card dark:border-brand-900/50 dark:from-slate-900 dark:to-slate-950 sm:p-8"
      aria-busy="true"
      aria-live="polite"
    >
      <div className="flex flex-col items-center text-center">
        <div className="relative h-14 w-14">
          <div className="absolute inset-0 rounded-full border-4 border-brand-200 dark:border-brand-900" />
          <div className="absolute inset-0 animate-spin rounded-full border-4 border-transparent border-t-brand-600" />
        </div>
        <h2 className="mt-5 text-lg font-semibold text-slate-900 dark:text-white">
          Running analysis
        </h2>
        <p className="mt-1 max-w-sm text-sm text-slate-500 dark:text-slate-400">
          Please wait while the model processes your image and generates the explainability map
          and report.
        </p>

        <div className="mt-6 h-1.5 w-full max-w-xs overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
          <div className="analysis-progress-bar h-full w-1/3 rounded-full bg-gradient-to-r from-brand-500 to-brand-600" />
        </div>

        <ul className="mt-8 w-full max-w-sm space-y-2.5 text-left text-sm text-slate-600 dark:text-slate-400">
          {steps.map((label) => (
            <li key={label} className="flex items-center gap-2">
              <span className="inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-brand-500 animate-pulse" />
              {label}
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}

const ResultCard = ({ result, error, loading }) => {
  if (loading) {
    return <AnalysisLoadingPanel />
  }

  if (error) {
    return (
      <section
        className="rounded-2xl border border-red-200 bg-red-50/80 p-6 shadow-card dark:border-red-900/60 dark:bg-red-950/40 sm:p-8"
        role="alert"
      >
        <h2 className="text-lg font-semibold text-red-900 dark:text-red-200">
          Something went wrong
        </h2>
        <p className="mt-2 text-sm text-red-800 dark:text-red-300">{error}</p>
      </section>
    )
  }

  if (!result) {
    return (
      <section className="rounded-2xl border border-dashed border-slate-200 bg-slate-50/50 p-8 text-center shadow-inner dark:border-slate-700 dark:bg-slate-900/40 sm:p-10">
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Results will appear here after you run an analysis.
        </p>
      </section>
    )
  }

  const heatmapSrc = getAbsoluteUrl(result.heatmap_url)
  const pdfSrc = getAbsoluteUrl(result.report_pdf_url)
  const confidencePct = formatConfidence(result.confidence)
  const risk = result.risk
  const isHighRisk = risk === 'HIGH'
  const reportId = result.report_id

  return (
    <section className="space-y-6" aria-live="polite">
      <div className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-card dark:border-slate-700/80 dark:bg-slate-900 sm:p-8">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold tracking-tight text-slate-900 dark:text-white">
              Prediction
            </h2>
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
              AI-assisted read — not a medical diagnosis.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {risk && (
              <span
                className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide ${
                  isHighRisk
                    ? 'bg-red-100 text-red-800 ring-1 ring-red-200 dark:bg-red-950 dark:text-red-200 dark:ring-red-900'
                    : 'bg-emerald-100 text-emerald-800 ring-1 ring-emerald-200 dark:bg-emerald-950 dark:text-emerald-200 dark:ring-emerald-900'
                }`}
              >
                Risk: {risk}
              </span>
            )}
            {pdfSrc && (
              <a
                href={pdfSrc}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-slate-800 transition hover:bg-slate-100 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100 dark:hover:bg-slate-700"
              >
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Download PDF report
              </a>
            )}
          </div>
        </div>

        {reportId && (
          <p className="mt-3 font-mono text-xs text-slate-400 dark:text-slate-500">
            Report ID: {reportId}
          </p>
        )}

        <div className="mt-6 rounded-xl bg-slate-50 p-5 dark:bg-slate-800/60">
          <p className="text-xs font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400">
            Predicted class
          </p>
          <p className="mt-1 text-2xl font-bold tracking-tight text-brand-800 dark:text-brand-300 sm:text-3xl">
            {result.class}
          </p>

          <div className="mt-4">
            <div className="flex justify-between text-sm">
              <span className="text-slate-600 dark:text-slate-300">Confidence</span>
              <span className="font-semibold text-slate-900 dark:text-white">{confidencePct}</span>
            </div>
            <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-600">
              <div
                className="h-full rounded-full bg-gradient-to-r from-brand-500 to-brand-600 transition-all"
                style={{
                  width: `${Math.min(100, Math.max(0, Number(result.confidence) * 100))}%`,
                }}
              />
            </div>
            {result.confidence_level && (
              <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                Level:{' '}
                <strong className="text-slate-700 dark:text-slate-200">
                  {result.confidence_level}
                </strong>
              </p>
            )}
          </div>
        </div>

        {result.warning && (
          <p className="mt-4 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-900/60 dark:bg-amber-950/50 dark:text-amber-100">
            {result.warning}
          </p>
        )}

        {result.message && (
          <p className="mt-4 text-sm text-slate-600 dark:text-slate-300">{result.message}</p>
        )}

        {result.description && (
          <div className="mt-6 border-t border-slate-100 pt-6 dark:border-slate-700">
            <h3 className="text-sm font-semibold text-slate-900 dark:text-white">Description</h3>
            <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-300">
              {result.description}
            </p>
          </div>
        )}

        {Array.isArray(result.medications) && result.medications.length > 0 && (
          <div className="mt-6 border-t border-slate-100 pt-6 dark:border-slate-700">
            <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
              Medications (general)
            </h3>
            <ul className="mt-2 list-inside list-disc space-y-1 text-sm text-slate-600 dark:text-slate-300">
              {result.medications.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        )}

        {Array.isArray(result.prevention) && result.prevention.length > 0 && (
          <div className="mt-6 border-t border-slate-100 pt-6 dark:border-slate-700">
            <h3 className="text-sm font-semibold text-slate-900 dark:text-white">Prevention</h3>
            <ul className="mt-2 list-inside list-disc space-y-1 text-sm text-slate-600 dark:text-slate-300">
              {result.prevention.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        )}

        {result.disclaimer && (
          <p className="mt-6 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-xs leading-relaxed text-slate-600 dark:border-slate-600 dark:bg-slate-800/80 dark:text-slate-300">
            <span className="font-semibold text-slate-700 dark:text-slate-200">Note: </span>
            {result.disclaimer}
          </p>
        )}
      </div>

      {heatmapSrc && (
        <div className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-card dark:border-slate-700/80 dark:bg-slate-900 sm:p-8">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
            Attention map (Grad-CAM)
          </h3>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            Highlights regions that influenced the model most. Each new analysis uses a fresh file
            (no stale cache).
          </p>
          <div className="mt-4 overflow-hidden rounded-xl border border-slate-200 bg-slate-50 dark:border-slate-600 dark:bg-slate-800/50">
            <img
              key={reportId || heatmapSrc}
              src={heatmapSrc}
              alt="Model attention heatmap overlay"
              className="mx-auto w-full max-h-80 object-contain"
            />
          </div>
        </div>
      )}
    </section>
  )
}

export default ResultCard

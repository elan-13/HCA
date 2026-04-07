import { useState } from 'react'
import Navbar from '../components/Navbar'
import UploadCard from '../components/Uploadcard'
import ResultCard from '../components/ResultCard'
import { askAiAssistant } from '../services/api'

const Home = () => {
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [assistantQuestion, setAssistantQuestion] = useState('')
  const [assistantAnswer, setAssistantAnswer] = useState('')
  const [assistantError, setAssistantError] = useState('')
  const [assistantLoading, setAssistantLoading] = useState(false)
  const [assistantOpen, setAssistantOpen] = useState(false)

  async function onAskAssistant(e) {
    e.preventDefault()
    setAssistantError('')
    setAssistantAnswer('')
    const q = assistantQuestion.trim()
    if (!q) return
    setAssistantLoading(true)
    try {
      const data = await askAiAssistant({
        question: q,
        predictedClass: result?.class ?? result?.predicted_class ?? null,
        topPredictions: result?.possible_conditions ?? result?.top_predictions ?? null,
      })
      if (data?.answer) setAssistantAnswer(data.answer)
      else setAssistantError(data?.error || 'AI request failed.')
    } catch (err) {
      const backendMsg = err?.response?.data?.error
      setAssistantError(backendMsg || err?.message || 'AI request failed.')
    } finally {
      setAssistantLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen flex-col bg-slate-50 font-sans transition-colors dark:bg-slate-950">
      <Navbar />

      <main className="flex-1">
        <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6 lg:px-8 lg:py-14">
          <div className="mx-auto max-w-2xl text-center">
            <p className="text-sm font-medium uppercase tracking-wider text-brand-700 dark:text-brand-400">
              Skin condition analysis
            </p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900 dark:text-white sm:text-4xl">
              AI skin disease classifier
            </h1>
            <p className="mt-4 text-base leading-relaxed text-slate-600 dark:text-slate-300">
              Upload a clinical-style photo to get a probability-based prediction with an
              explainability overlay and a downloadable PDF report. Always consult a qualified
              clinician for diagnosis and treatment.
            </p>
          </div>

          <div className="mt-12 grid gap-8 lg:grid-cols-2 lg:items-start lg:gap-10">
            <UploadCard
              onResult={setResult}
              onError={setError}
              loading={loading}
              setLoading={setLoading}
            />
            <ResultCard result={result} error={error} loading={loading} />
          </div>

          <footer
            id="disclaimer"
            className="mx-auto mt-16 max-w-3xl border-t border-slate-200 pt-8 text-center dark:border-slate-800"
          >
            <p className="text-xs leading-relaxed text-slate-500 dark:text-slate-400">
              This tool is for research and educational demonstration only. It does not replace
              professional medical advice, examination, or emergency care. If you have urgent
              symptoms, contact emergency services or a dermatologist.
            </p>
          </footer>
        </div>
      </main>

      <button
        type="button"
        onClick={() => setAssistantOpen((v) => !v)}
        className="fixed bottom-6 right-6 z-50 inline-flex items-center justify-center rounded-full bg-brand-600 px-5 py-3 text-sm font-semibold text-white shadow-lg transition hover:bg-brand-700"
      >
        {assistantOpen ? 'Close AI' : 'AI Assistant'}
      </button>

      {assistantOpen ? (
        <section className="fixed bottom-24 right-6 z-50 w-[min(92vw,380px)] rounded-2xl border border-slate-200 bg-white p-4 shadow-2xl dark:border-slate-800 dark:bg-slate-900">
          <h2 className="text-base font-semibold text-slate-900 dark:text-white">AI Assistant</h2>

          <form onSubmit={onAskAssistant} className="mt-3 space-y-3">
            <div>
              <label className="sr-only" htmlFor="assistantQuestion">
                Question
              </label>
              <input
                id="assistantQuestion"
                value={assistantQuestion}
                onChange={(e) => setAssistantQuestion(e.target.value)}
                placeholder="Ask anything..."
                className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 shadow-sm outline-none transition focus:border-brand-500 focus:ring-4 focus:ring-brand-200/40 dark:border-slate-700 dark:bg-slate-950 dark:text-white dark:focus:ring-brand-500/20"
              />
            </div>
            <div className="flex items-center gap-3">
              <button
                type="submit"
                disabled={assistantLoading}
                className="inline-flex items-center justify-center rounded-xl bg-brand-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {assistantLoading ? 'Asking…' : 'Ask'}
              </button>
              {assistantError ? (
                <p className="text-sm text-rose-600 dark:text-rose-400">{assistantError}</p>
              ) : null}
            </div>
          </form>

          {assistantAnswer ? (
            <div className="mt-4 max-h-60 overflow-auto whitespace-pre-wrap rounded-xl border border-slate-200 bg-slate-50 p-3 text-sm text-slate-800 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-100">
              {assistantAnswer}
            </div>
          ) : null}
        </section>
      ) : null}
    </div>
  )
}

export default Home

import { useEffect, useRef, useState } from 'react'
import { predictSkinDisease } from '../services/api'

const UploadCard = ({ onResult, onError, loading, setLoading }) => {
  const [preview, setPreview] = useState(null)
  const fileInputRef = useRef(null)

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview)
    }
  }, [preview])

  const handleChange = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (preview) URL.revokeObjectURL(preview)
    setPreview(URL.createObjectURL(file))
    onError?.('')
    onResult?.(null)
  }

  const handleSubmit = async () => {
    const file = fileInputRef.current?.files?.[0]
    if (!file) {
      onError?.('Please choose an image first.')
      return
    }

    setLoading(true)
    onError?.('')
    onResult?.(null)

    try {
      const data = await predictSkinDisease(file)
      onResult(data)
    } catch (err) {
      const message =
        err?.response?.data?.error ||
        err?.message ||
        'Prediction failed. Is the backend running?'
      onError?.(message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section
      id="analyze"
      className="rounded-2xl border border-slate-200/80 bg-white p-6 shadow-card dark:border-slate-700/80 dark:bg-slate-900 sm:p-8"
      aria-labelledby="upload-heading"
    >
      <h2
        id="upload-heading"
        className="text-lg font-semibold tracking-tight text-slate-900 dark:text-white"
      >
        Upload image
      </h2>
      <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
        Use a well-lit, in-focus photo of the affected area. JPG or PNG, under ~10&nbsp;MB.
      </p>

      <label className="mt-6 flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-200 bg-slate-50/80 px-4 py-10 transition hover:border-brand-500/50 hover:bg-brand-50/30 dark:border-slate-600 dark:bg-slate-800/50 dark:hover:border-brand-600/50 dark:hover:bg-brand-950/20">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleChange}
          disabled={loading}
          className="sr-only"
        />
        <span className="text-sm font-medium text-slate-700 dark:text-slate-200">
          Click to browse or drag a file here
        </span>
        <span className="mt-1 text-xs text-slate-400 dark:text-slate-500">
          Supports common image formats
        </span>
      </label>

      {preview && (
        <div className="mt-6 overflow-hidden rounded-xl border border-slate-200 bg-slate-50 dark:border-slate-600 dark:bg-slate-800/50">
          <img
            src={preview}
            alt="Selected image preview"
            className="mx-auto max-h-64 w-full object-contain"
          />
        </div>
      )}

      <button
        type="button"
        onClick={handleSubmit}
        disabled={loading || !preview}
        className="mt-6 w-full rounded-xl bg-gradient-to-r from-brand-600 to-brand-700 px-4 py-3 text-sm font-semibold text-white shadow-md transition hover:from-brand-700 hover:to-brand-800 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {loading ? (
          <span className="inline-flex items-center justify-center gap-2">
            <span
              className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent"
              aria-hidden
            />
            Analyzing…
          </span>
        ) : (
          'Run analysis'
        )}
      </button>
    </section>
  )
}

export default UploadCard

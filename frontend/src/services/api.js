import axios from 'axios'

const API_BASE_URL = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000').replace(/\/$/, '')

const api = axios.create({
  baseURL: API_BASE_URL,
})

export async function predictSkinDisease(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await api.post('/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })

  return response.data
}

export async function askAiAssistant({ question, predictedClass, topPredictions }) {
  const response = await api.post('/ai/ask', {
    question,
    predicted_class: predictedClass ?? null,
    top_predictions: topPredictions ?? null,
  })
  return response.data
}

/** Absolute URL for any backend path (images, PDFs, etc.). */
export function getAbsoluteUrl(pathOrUrl) {
  if (!pathOrUrl) return null
  if (pathOrUrl.startsWith('http://') || pathOrUrl.startsWith('https://')) {
    return pathOrUrl
  }
  return `${API_BASE_URL}${pathOrUrl}`
}

export const getAbsoluteImageUrl = getAbsoluteUrl

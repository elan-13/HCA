# Deployment Guide (Render + Vercel)

## 1) Deploy Backend on Render

1. Push this repository to GitHub.
2. In Render, click **New +** -> **Blueprint**.
3. Select this repo. Render will detect `render.yaml` and create the backend service.
4. In the created Render service, set secret env vars as needed:
   - `OPENAI_API_KEY` (optional)
   - `GROQ_API_KEY` (optional)
5. Deploy and copy your backend URL, for example:
   - `https://hca-backend.onrender.com`
6. Confirm health endpoint works:
   - `https://hca-backend.onrender.com/health`

## 2) Deploy Frontend on Vercel

1. In Vercel, click **Add New...** -> **Project**.
2. Import the same GitHub repo.
3. Configure project settings:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
4. Add environment variable:
   - `VITE_API_URL` = your Render backend URL
     Example: `https://hca-backend.onrender.com`
5. Deploy.

`frontend/vercel.json` already includes SPA rewrites so direct URL refreshes work.

## 3) CORS

Backend currently allows all origins via Flask-CORS, so Vercel frontend can call Render backend without extra CORS setup.

## 4) Quick Smoke Test

1. Open deployed frontend URL.
2. Upload an image and run prediction.
3. Verify response includes disease prediction, explanation, and report links.
4. If AI assistant is enabled, test one `/ai/ask` query.

## Notes

- Render free tier can sleep after inactivity; first request may be slow.
- Model/data files are read from repo paths under `saved_models` and `reports` directories.

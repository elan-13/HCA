# HCA - Skin Disease Classification and AI Assistant

Skin disease analysis web app with:
- image classification (`Acne`, `Eczema`, `Psoriasis`, `SkinCancer`, `Vitiligo`, `Warts`)
- Grad-CAM explainability heatmap
- PDF report generation
- AI assistant (Groq/OpenAI-compatible chat endpoint)

## Tech Stack

- **Frontend:** React + Vite + Tailwind
- **Backend:** Flask
- **ML:** TensorFlow / Keras (MobileNetV2 transfer learning)
- **Extras:** OpenCV, scikit-learn, FPDF

## Project Structure

`frontend/` - React UI  
`backend/` - Flask API and ML inference/training code  
`prepare_dataset.py` - prepares dataset into `Dataset/train|val|test/...`  
`saved_models/` - trained model and class mappings (local artifacts)

## Features

- Upload image and get class prediction + confidence
- Grad-CAM heatmap visualization
- Downloadable PDF report
- AI assistant button in UI corner for conversational help
- Optional context-aware AI replies using prediction output

## Prerequisites

- Python **3.11 or 3.12** (recommended; avoid 3.14 for TensorFlow compatibility)
- Node.js 18+
- npm

## Backend Setup

From project root:

```bash
cd backend
python -m venv .venv
```

Activate venv:

- PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

- CMD:

```cmd
.\.venv\Scripts\activate.bat
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Run backend:

```bash
python app.py
```

Backend runs on: `http://127.0.0.1:5000`

## Frontend Setup

From project root:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on: `http://127.0.0.1:5173` (default Vite port)

## Dataset Preparation

1. Put raw dataset under:
   - `SkinDisease/train/<ClassName>/...`
   - `SkinDisease/test/<ClassName>/...`
2. Edit path in `prepare_dataset.py` if needed (`INPUT_ROOT`)
3. Run:

```bash
python prepare_dataset.py
```

This creates:

`Dataset/train/<Class>/...`  
`Dataset/val/<Class>/...`  
`Dataset/test/<Class>/...`

## Training Model

Run:

```bash
python backend/app/train.py
```

Artifacts saved (default):
- `saved_models/skin_model.h5`
- `saved_models/class_names.json`
- `saved_models/class_indices.json`

## AI Assistant Configuration (`.env`)

Create `backend/.env`:

```env
GROQ_API_KEY=YOUR_GROQ_KEY
GROQ_MODEL=llama-3.1-8b-instant
```

Fallback supported:

```env
OPENAI_API_KEY=YOUR_OPENAI_KEY
OPENAI_MODEL=gpt-4o-mini
```

## API Endpoints

- `GET /health` - backend health check
- `POST /predict` - image prediction (multipart form-data, field: `file`)
- `POST /ai/ask` - AI assistant chat

## Notes for GitHub Upload

- Do **not** commit heavy/generated files:
  - `backend/.venv/`
  - `Dataset/`
  - `saved_models/`
  - `node_modules/`
  - `uploads/`, `reports/`
- Do **not** commit secrets:
  - `.env`
  - `backend/.env`

## Disclaimer

This project is for educational/research use only and is **not** a medical diagnosis tool.  
Always consult a qualified dermatologist for clinical decisions.

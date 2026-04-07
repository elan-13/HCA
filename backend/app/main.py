from __future__ import annotations

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .predict import load_artifacts, predict_image_bytes
from .schemas.predict import PredictResponse
from .schemas.train import TrainOptions, TrainResponse
from .train import train_model
from .utils.paths import class_names_path, dataset_dir, model_path


app = FastAPI(title="HCA Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    # Pre-load model artifacts so /predict works immediately.
    # If the model isn't trained yet, /predict will return a clear error.
    try:
        load_artifacts()
    except FileNotFoundError:
        pass


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train", response_model=TrainResponse)
def start_train(
    background_tasks: BackgroundTasks,
    options: TrainOptions | None = None,
) -> TrainResponse:
    opts = options or TrainOptions()

    background_tasks.add_task(
        train_model,
        dataset_dir=dataset_dir(),
        model_path=model_path(),
        class_names_path=class_names_path(),
        epochs=opts.epochs,
        batch_size=opts.batch_size,
        img_size=opts.img_size,
    )

    return TrainResponse(
        started=True,
        dataset_dir=str(dataset_dir()),
        model_path=str(model_path()),
        class_names_path=str(class_names_path()),
        message="Training started in background. Call /predict after it finishes.",
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    top_k: int = 1,
) -> PredictResponse:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are supported.")

    try:
        image_bytes = await file.read()
        result = predict_image_bytes(image_bytes, top_k=top_k)
        return PredictResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


from __future__ import annotations

from pydantic import BaseModel, Field


class TrainOptions(BaseModel):
    epochs: int = Field(default=10, ge=1, le=200)
    batch_size: int = Field(default=32, ge=1, le=1024)
    img_size: int = Field(default=224, ge=64, le=1024)


class TrainResponse(BaseModel):
    started: bool
    dataset_dir: str
    model_path: str
    class_names_path: str
    message: str


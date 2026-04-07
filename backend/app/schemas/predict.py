from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List


class TopPrediction(BaseModel):
    class_name: str
    probability: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    top_k: int
    top_predictions: List[TopPrediction]


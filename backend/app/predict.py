from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_preprocess_input,
)

try:
    from .utils.paths import class_names_path as default_class_names_path
    from .utils.paths import model_path as default_model_path
    from .gradcam import compute_gradcam_heatmap, heatmap_to_overlay_and_save
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from app.utils.paths import (  # type: ignore
        class_names_path as default_class_names_path,
        model_path as default_model_path,
    )
    from app.gradcam import compute_gradcam_heatmap, heatmap_to_overlay_and_save  # type: ignore


_ARTIFACTS: dict[str, Any] = {}

disease_info: dict[str, dict[str, Any]] = {
    "Acne": {
        "description": "A common skin condition causing pimples due to clogged pores.",
        "medications": ["Benzoyl peroxide", "Salicylic acid", "Adapalene"],
        "prevention": ["Keep skin clean", "Avoid oily products", "Do not squeeze pimples"],
    },
    "Eczema": {
        "description": "A condition causing itchy and inflamed skin.",
        "medications": ["Hydrocortisone cream", "Moisturizers", "Antihistamines"],
        "prevention": ["Avoid triggers", "Use mild soaps", "Keep skin hydrated"],
    },
    "Psoriasis": {
        "description": "An autoimmune disease causing red scaly patches.",
        "medications": ["Topical steroids", "Vitamin D creams", "Phototherapy"],
        "prevention": ["Reduce stress", "Avoid smoking", "Moisturize regularly"],
    },
    "Vitiligo": {
        "description": "Loss of skin pigment causing white patches.",
        "medications": ["Corticosteroids", "Light therapy"],
        "prevention": ["Protect from sun", "Avoid skin trauma"],
    },
    "Warts": {
        "description": "Small growths caused by viral infection (HPV).",
        "medications": ["Salicylic acid", "Cryotherapy"],
        "prevention": ["Avoid touching warts", "Maintain hygiene"],
    },
    "SkinCancer": {
        "description": "Abnormal growth of skin cells, potentially dangerous.",
        "medications": [],
        "prevention": ["Avoid excessive sun exposure", "Use sunscreen"],
    },
}

DISCLAIMER = (
    "This is an AI-based prediction and not a medical diagnosis. Consult a doctor."
)


def _get_model_img_size(model: tf.keras.Model) -> tuple[int, int]:
    # Model input shape is typically (None, H, W, 3).
    shape = model.input_shape
    if not isinstance(shape, tuple) or len(shape) < 3:
        raise ValueError(f"Unexpected model input shape: {shape}")
    h = int(shape[1])
    w = int(shape[2])
    return h, w


def _confidence_level(confidence: float) -> str:
    if confidence > 0.9:
        return "High"
    if confidence >= 0.7:
        return "Medium"
    return "Low"


def load_artifacts(
    *,
    model_path: Path | str | None = None,
    class_names_path: Path | str | None = None,
) -> None:
    global _ARTIFACTS

    model_path = Path(model_path) if model_path is not None else default_model_path()
    class_names_path = (
        Path(class_names_path) if class_names_path is not None else default_class_names_path()
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not class_names_path.exists():
        raise FileNotFoundError(f"Class names not found: {class_names_path}")

    model = tf.keras.models.load_model(str(model_path), compile=False)
    with class_names_path.open("r", encoding="utf-8") as f:
        class_names = json.load(f)

    if not isinstance(class_names, list) or not all(
        isinstance(x, str) for x in class_names
    ):
        raise ValueError(f"Invalid class_names.json format: {class_names_path}")

    _ARTIFACTS = {"model": model, "class_names": class_names}


def predict_image_bytes(image_bytes: bytes, *, top_k: int = 1) -> dict[str, Any]:
    if "model" not in _ARTIFACTS or "class_names" not in _ARTIFACTS:
        load_artifacts()

    model: tf.keras.Model = _ARTIFACTS["model"]
    class_names: list[str] = _ARTIFACTS["class_names"]

    if top_k < 1:
        top_k = 1
    top_k = min(top_k, len(class_names))

    h, w = _get_model_img_size(model)
    image = tf.keras.utils.load_img(io.BytesIO(image_bytes), target_size=(h, w), color_mode="rgb")
    image_array = tf.keras.utils.img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0)
    image_array = tf.cast(image_array, tf.float32)
    image_array = tf.constant(mobilenet_preprocess_input(image_array.numpy()))

    probs = model.predict(image_array, verbose=0)[0]
    probs = np.asarray(probs, dtype=np.float32)

    # Model should already end with softmax, but normalize defensively.
    sum_probs = float(np.sum(probs))
    if sum_probs > 0:
        probs = probs / sum_probs

    idxs = np.argsort(probs)[::-1][:top_k]
    top_predictions = [
        {"class_name": class_names[int(i)], "probability": float(probs[int(i)])}
        for i in idxs
    ]

    predicted_index = int(idxs[0])
    predicted_class = class_names[predicted_index]

    return {
        "predicted_class": predicted_class,
        "predicted_index": predicted_index,
        "top_k": int(top_k),
        "top_predictions": top_predictions,
    }


def _preprocess_input(
    input_data: Any,
) -> tuple[np.ndarray, list[str], tf.keras.Model, np.ndarray]:
    if "model" not in _ARTIFACTS or "class_names" not in _ARTIFACTS:
        load_artifacts()

    model: tf.keras.Model = _ARTIFACTS["model"]
    class_names: list[str] = _ARTIFACTS["class_names"]

    h, w = _get_model_img_size(model)

    if isinstance(input_data, (bytes, bytearray)):
        image = tf.keras.utils.load_img(
            io.BytesIO(bytes(input_data)), target_size=(h, w), color_mode="rgb"
        )
        arr = tf.keras.utils.img_to_array(image)
        original_rgb = arr.astype(np.uint8)
    elif isinstance(input_data, (str, Path)):
        image = tf.keras.utils.load_img(
            str(input_data), target_size=(h, w), color_mode="rgb"
        )
        arr = tf.keras.utils.img_to_array(image)
        original_rgb = arr.astype(np.uint8)
    else:
        # Assume numpy-like array. Accept either (H,W,3) or (1,H,W,3).
        arr = np.asarray(input_data)
        if arr.dtype.kind == "f" and arr.size > 0 and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=0)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError("input_data must be an image path, bytes, or an array shaped (H,W,3).")
        if arr.shape[0] != 1:
            raise ValueError("input_data batch dimension must be 1 if providing an array.")
        if arr.shape[1] != h or arr.shape[2] != w:
            arr = tf.image.resize(arr[0], (h, w)).numpy()[None, ...]
        original_rgb = np.clip(arr[0], 0, 255).astype(np.uint8)

    image_array = np.asarray(arr, dtype=np.float32)
    if image_array.ndim == 3:
        image_array = np.expand_dims(image_array, axis=0)
    if image_array.size and float(np.nanmax(image_array)) <= 1.0:
        image_array = image_array * 255.0
    image_array = mobilenet_preprocess_input(image_array)

    # Return preprocessed tensor + original RGB for Grad-CAM overlay.
    # (Keep original_rgb for overlay; it should be uint8 in RGB.)
    return (
        np.asarray(image_array, dtype=np.float32),
        class_names,
        model,
        original_rgb,
    )


def predict_with_explanations(
    input_data: Any,
    *,
    heatmap_output_path: str = "uploads/heatmap.jpg",
) -> dict[str, Any]:
    """
    Predict + Grad-CAM + enriched response.

    Always returns `disclaimer` and `heatmap_path`.
    The API layer should convert heatmap_path to heatmap_url.
    """

    image_array, class_names, model, original_rgb = _preprocess_input(input_data)

    probs = model.predict(image_array, verbose=0)[0]
    probs = np.asarray(probs, dtype=np.float32)
    sum_probs = float(np.sum(probs))
    if sum_probs > 0:
        probs = probs / sum_probs

    top_idxs = np.argsort(probs)[::-1]
    top1_idx = int(top_idxs[0])
    top2_idx = int(top_idxs[1]) if len(top_idxs) > 1 else top1_idx

    predicted_class = class_names[top1_idx]
    confidence = float(probs[top1_idx])
    confidence_level = _confidence_level(confidence)

    # Grad-CAM (save overlay)
    heatmap = compute_gradcam_heatmap(
        model,
        tf.convert_to_tensor(image_array, dtype=tf.float32),
        class_index=top1_idx,
        input_hw=(original_rgb.shape[0], original_rgb.shape[1]),
    )
    heatmap_path = heatmap_to_overlay_and_save(
        heatmap, image_rgb=original_rgb, output_path=heatmap_output_path
    )

    # Second opinion logic (top-2 close)
    possible_conditions = None
    if len(top_idxs) > 1:
        diff = float(probs[top1_idx] - probs[top2_idx])
        if diff < 0.1:
            possible_conditions = [
                {"class": class_names[top1_idx], "probability": float(probs[top1_idx])},
                {"class": class_names[top2_idx], "probability": float(probs[top2_idx])},
            ]

    base: dict[str, Any] = {
        "class": predicted_class,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "disclaimer": DISCLAIMER,
        "heatmap_path": heatmap_path,
    }
    if possible_conditions is not None:
        base["possible_conditions"] = possible_conditions

    # Confidence handling
    if confidence < 0.7:
        base["class"] = "Uncertain"
        base["message"] = (
            "Prediction is not confident. Please upload a clearer image."
        )
        return base

    # Enriched response
    if predicted_class == "SkinCancer":
        base["risk"] = "HIGH"
        base["warning"] = (
            "⚠️ Possible Skin Cancer detected. Please consult a dermatologist immediately."
        )
        return base

    info = disease_info.get(predicted_class, {})
    base["risk"] = "LOW"
    base["description"] = info.get("description", "")
    base["medications"] = info.get("medications", [])
    base["prevention"] = info.get("prevention", [])
    return base


def predict_image(image_path: str | Path) -> dict[str, Any]:
    """
    Predict from an image file path.

    Returns:
      { "class": <predicted_class>, "confidence": <max_probability>, "flag": <HIGH RISK|NORMAL> }
    """

    # Kept for backwards compatibility (path-based prediction).
    return predict_with_explanations(image_path, heatmap_output_path="uploads/heatmap.jpg")


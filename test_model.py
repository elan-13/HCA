import json
from pathlib import Path

import numpy as np
import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parent
MODEL_PATH = REPO_ROOT / "saved_models" / "skin_model.h5"
CLASS_NAMES_PATH = REPO_ROOT / "saved_models" / "class_names.json"

IMAGE_PATH = Path(r"D:\SEM-8\HCA-PROJECT\skin_disease\test\Acne\Acne_0000.jpg")
IMG_SIZE = (224, 224)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

    if CLASS_NAMES_PATH.exists():
        class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    else:
        class_names = ["Acne", "Eczema", "Psoriasis", "SkinCancer", "Vitiligo", "Warts"]

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    print(f"Using image: {IMAGE_PATH}")

    img = tf.keras.utils.load_img(str(IMAGE_PATH), target_size=IMG_SIZE, color_mode="rgb")
    arr = tf.keras.utils.img_to_array(img)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    print("Raw prediction:", preds)

    # TOP 2 predictions (IMPORTANT DEBUG)
    top2 = np.argsort(preds)[-2:][::-1]
    for i in top2:
        label = class_names[int(i)] if int(i) < len(class_names) else str(int(i))
        print(f"{label} → {preds[int(i)]:.4f}")

    pred_index = int(np.argmax(preds))
    pred_label = class_names[pred_index] if pred_index < len(class_names) else str(pred_index)
    print("\nFinal Prediction:", pred_label)
    print("Confidence:", float(preds[pred_index]))


if __name__ == "__main__":
    main()


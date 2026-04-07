from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    # When run as a module (recommended): `python -m app.train`
    from .utils.paths import class_names_path as default_class_names_path
    from .utils.paths import dataset_dir as default_dataset_dir
    from .utils.paths import model_path as default_model_path
except ImportError:
    # When run as a script: `python backend/app/train.py`
    # Add `backend/` to sys.path so `app.*` imports work.
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from app.utils.paths import (  # type: ignore
        class_names_path as default_class_names_path,
        dataset_dir as default_dataset_dir,
        model_path as default_model_path,
    )


def build_model(
    num_classes: int,
    img_size: int,
    *,
    label_smoothing: float = 0.05,
    head_learning_rate: float = 1e-3,
) -> tf.keras.Model:
    # MobileNetV2 transfer learning backbone + two-layer head.
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # freeze base layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    model.compile(
        loss=loss,
        optimizer=Adam(learning_rate=head_learning_rate),
        metrics=["accuracy"],
    )
    return model


def train_model(
    *,
    dataset_dir: Path | str | None = None,
    model_path: Path | str | None = None,
    class_names_path: Path | str | None = None,
    epochs: int = 35,
    batch_size: int = 32,
    img_size: int = 224,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir) if dataset_dir is not None else default_dataset_dir()
    model_path = Path(model_path) if model_path is not None else default_model_path()
    class_names_path = (
        Path(class_names_path) if class_names_path is not None else default_class_names_path()
    )

    classes = ["Acne", "Eczema", "Psoriasis", "SkinCancer", "Vitiligo", "Warts"]

    # Prefer repo-level dataset folders first, then fallback to `dataset_dir`.
    # New layout supports: Dataset/train, Dataset/val, Dataset/test.
    repo_root = Path(__file__).resolve().parents[3]
    preferred_roots = [repo_root / "Dataset", repo_root / "dataset"]
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    val_dir = dataset_dir / "val"

    for root in preferred_roots:
        root_train = root / "train"
        root_test = root / "test"
        if root_train.exists() and root_test.exists():
            train_dir = root_train
            test_dir = root_test
            val_dir = root / "val"
            break

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset folders: {train_dir} and {test_dir}. "
            "Create `Dataset/train` and `Dataset/test` (or set DATASET_DIR)."
        )

    # Validate class folders exist (so `classes=[...]` doesn't fail silently).
    for cls in classes:
        if not (train_dir / cls).exists():
            raise FileNotFoundError(f"Missing train class folder: {train_dir / cls}")
        if not (test_dir / cls).exists():
            raise FileNotFoundError(f"Missing test class folder: {test_dir / cls}")
        if val_dir.exists() and not (val_dir / cls).exists():
            raise FileNotFoundError(f"Missing val class folder: {val_dir / cls}")

    dataset_root = train_dir.resolve().parent
    print("Dataset paths (resolved):")
    print(f"  root (parent of train/): {dataset_root}")
    print(f"  train:  {train_dir.resolve()}")
    print(f"  val:    {val_dir.resolve()}" + (" (missing — not used)" if not val_dir.exists() else ""))
    print(f"  test:   {test_dir.resolve()}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    class_names_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # MobileNetV2 expects `preprocess_input` (scale ~[-1,1]); do not use rescale=1/255 with it.
    train_datagen = ImageDataGenerator(
        preprocessing_function=mobilenet_preprocess_input,
        rotation_range=12,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.12,
        brightness_range=(0.85, 1.15),
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(preprocessing_function=mobilenet_preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        classes=classes,
        class_mode="categorical",
        shuffle=True,
    )
    test_generator = val_datagen.flow_from_directory(
        str(test_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        classes=classes,
        class_mode="categorical",
        shuffle=False,
    )
    validation_source_dir = val_dir if val_dir.exists() else test_dir
    print(f"  validation batches from: {validation_source_dir.resolve()}")
    validation_generator = val_datagen.flow_from_directory(
        str(validation_source_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        classes=classes,
        class_mode="categorical",
        shuffle=False,
    )

    class_names = classes
    num_classes = len(class_names)

    # Print + persist class index mapping (important for consistent inference).
    print(f"Training config: epochs={epochs}, batch_size={batch_size}, img_size={img_size}")
    print("class_indices mapping (class -> index):")
    print(train_generator.class_indices)

    # Also persist index-based mappings explicitly (0..5 for 6 classes).
    index_to_class = [""] * num_classes
    for cls, idx in train_generator.class_indices.items():
        index_to_class[int(idx)] = cls
    class_indices_payload = {
        "class_to_index": dict(train_generator.class_indices),
        "index_to_class": {str(i): name for i, name in enumerate(index_to_class)},
    }
    class_indices_path = class_names_path.parent / "class_indices.json"
    with class_indices_path.open("w", encoding="utf-8") as f:
        json.dump(class_indices_payload, f, indent=2)
    print(f"Saved class index mapping to: {class_indices_path}")

    # Compute class weights to reduce SkinCancer false positives from imbalance.
    y_int = train_generator.classes
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_int,
    )
    class_weight = {i: float(w) for i, w in enumerate(class_weights_arr)}
    print("class_weight (index -> weight):")
    print(class_weight)

    label_smoothing = 0.05
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

    model = build_model(num_classes=num_classes, img_size=img_size, label_smoothing=label_smoothing)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    )

    # Stage 1: train the head with frozen backbone.
    epochs_frozen = min(10, epochs)
    early_head = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )
    history_frozen = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs_frozen,
        class_weight=class_weight,
        callbacks=[reduce_lr, early_head],
    )
    epoch_next = len(history_frozen.history.get("loss", []))

    # Stage 2: fine-tune last layers of MobileNetV2 if epochs allow.
    if epochs > epoch_next:
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and layer.name.startswith("mobilenetv2"):
                base_model = layer
                break
        if base_model is None:
            for layer in model.layers:
                if layer.name.lower().startswith("mobilenetv2"):
                    base_model = layer
                    break

        finetune_last_n = 30
        if base_model is not None:
            base_model.trainable = True
            for l in base_model.layers[:-finetune_last_n]:
                l.trainable = False
            for l in base_model.layers[-finetune_last_n:]:
                l.trainable = True

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
                loss=loss_fn,
                metrics=["accuracy"],
            )

        reduce_lr_ft = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-8,
            verbose=1,
        )
        early_ft = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        )
        model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            initial_epoch=epoch_next,
            class_weight=class_weight,
            callbacks=[reduce_lr_ft, early_ft],
        )

    # Per-class metrics on validation split (same generator order as training).
    validation_generator.reset()
    y_val = validation_generator.classes
    val_probs = model.predict(validation_generator, verbose=1)
    y_val_pred = np.argmax(val_probs, axis=1)
    print("\nValidation classification report:")
    print(classification_report(y_val, y_val_pred, target_names=class_names))
    print("Validation confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_val, y_val_pred))

    # Final unbiased evaluation on the held-out test split.
    test_generator.reset()
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")

    y_test = test_generator.classes
    test_generator.reset()
    test_probs = model.predict(test_generator, verbose=1)
    y_test_pred = np.argmax(test_probs, axis=1)
    print("\nTest classification report:")
    print(classification_report(y_test, y_test_pred, target_names=class_names))

    # Save final model to the requested location.
    model.save(str(model_path))

    with class_names_path.open("w", encoding="utf-8") as f:
        # Save as an index-ordered list to match training/inference argmax indices.
        json.dump(index_to_class, f)

    return {
        "num_classes": num_classes,
        "class_names": class_names,
        "model_path": str(model_path),
        "class_names_path": str(class_names_path),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "dataset_root": str(dataset_root),
        "train_dir": str(train_dir.resolve()),
        "val_dir": str(val_dir.resolve()),
        "test_dir": str(test_dir.resolve()),
        "validation_dir": str(validation_source_dir.resolve()),
    }


def _onehot_encoder_dense(**base_kwargs: Any) -> OneHotEncoder:
    # sklearn changed `sparse` -> `sparse_output`. Support both.
    kwargs = dict(base_kwargs)
    try:
        kwargs["sparse_output"] = False
    except TypeError:
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)


def _infer_label_column(df: pd.DataFrame) -> str:
    candidates = ["label", "target", "class", "category", "diagnosis"]
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: assume the last column is the label.
    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least 1 feature column and 1 label column.")
    return str(df.columns[-1])


def prepare_train_test_tabular(
    *,
    dataset_dir: Path | str | None = None,
    dataset_subdir: str = "dataset",
    dataset_file: str | None = None,
    label_column: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Load a tabular dataset (CSV/XLSX/Parquet) from:
      <dataset_dir>/<dataset_subdir>/

    Then:
    - handle missing values
    - normalize numeric features
    - OneHotEncode labels
    - split to train/test (80/20 by default)

    Returns:
      X_train, X_test, y_train, y_test
    """

    dataset_dir = Path(dataset_dir) if dataset_dir is not None else default_dataset_dir()
    data_root = (dataset_dir / dataset_subdir).resolve()
    if not data_root.exists():
        raise FileNotFoundError(
            f"Expected dataset folder not found: {data_root}. "
            "Create this folder or pass `dataset_dir`/`dataset_subdir`."
        )

    supported_globs = ["*.csv", "*.xlsx", "*.parquet"]

    if dataset_file is not None:
        data_path = (data_root / dataset_file).resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
    else:
        found = []
        for pattern in supported_globs:
            found.extend(sorted(data_root.glob(pattern)))
        if not found:
            raise FileNotFoundError(
                f"No dataset file found in {data_root}. "
                f"Expected one of: {', '.join(supported_globs)}"
            )
        data_path = found[0]

    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path)
    elif data_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(data_path)
    elif data_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported dataset type: {data_path}")

    if label_column is None:
        label_column = _infer_label_column(df)
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Columns: {list(df.columns)}")

    y_raw = df[label_column].astype(str).values
    X_raw_df = df.drop(columns=[label_column])

    if X_raw_df.shape[1] == 0:
        raise ValueError("No feature columns found after dropping the label column.")

    # Identify feature types for preprocessing.
    categorical_cols = X_raw_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [c for c in X_raw_df.columns.tolist() if c not in set(categorical_cols)]

    numeric_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _onehot_encoder_dense(handle_unknown="ignore")),
        ]
    )

    preprocess_X = ColumnTransformer(
        transformers=[
            ("num", numeric_preprocess, numeric_cols),
            ("cat", categorical_preprocess, categorical_cols),
        ],
        remainder="drop",
    )

    X_train_df, X_test_df, y_train_raw, y_test_raw = train_test_split(
        X_raw_df,
        y_raw,
        test_size=test_size,
        random_state=random_state,
        stratify=y_raw,
    )

    preprocess_X.fit(X_train_df)
    X_train = preprocess_X.transform(X_train_df)
    X_test = preprocess_X.transform(X_test_df)

    y_encoder = _onehot_encoder_dense(handle_unknown="ignore")
    y_train = y_encoder.fit_transform(y_train_raw.reshape(-1, 1))
    y_test = y_encoder.transform(y_test_raw.reshape(-1, 1))

    # Ensure numpy float arrays for model training.
    return (
        np.asarray(X_train, dtype=np.float32),
        np.asarray(X_test, dtype=np.float32),
        np.asarray(y_train, dtype=np.float32),
        np.asarray(y_test, dtype=np.float32),
    )


if __name__ == "__main__":
    # Convenience entrypoint for local/manual runs.
    train_model()


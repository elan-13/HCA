from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    # backend/app/utils/paths.py -> repo root is 3 levels up: utils -> app -> backend -> repo
    return Path(__file__).resolve().parents[3]


def dataset_dir() -> Path:
    # Expected dataset layout:
    #   <dataset_dir>/train/<class_name>/*
    #   <dataset_dir>/val/<class_name>/*   (optional)
    #   <dataset_dir>/test/<class_name>/*
    #
    # Priority:
    # 1) DATASET_DIR env var (if provided)
    # 2) <repo_root>/Dataset
    # 3) <repo_root>/dataset
    # 4) <repo_root>/skin_disease
    env_dataset_dir = os.getenv("DATASET_DIR")
    if env_dataset_dir:
        return Path(env_dataset_dir).resolve()

    repo_root = _repo_root()
    candidates = [
        repo_root / "Dataset",
        repo_root / "dataset",
        repo_root / "skin_disease",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Fallback to the new default even if it does not exist yet.
    return (repo_root / "Dataset").resolve()


def model_path() -> Path:
    return Path(
        os.getenv(
            "MODEL_PATH",
            str(_repo_root() / "saved_models" / "skin_model.h5"),
        )
    ).resolve()


def class_names_path() -> Path:
    return Path(
        os.getenv(
            "CLASS_NAMES_PATH",
            str(_repo_root() / "saved_models" / "class_names.json"),
        )
    ).resolve()


from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Allow imports like `from app.predict import predict` when running from `backend/`.
import sys

sys.path.append(os.path.dirname(__file__))

from app.predict import predict_with_explanations  # noqa: E402
from app.report import report_public_urls, write_prediction_pdf  # noqa: E402
from app.ai_assistant import ask_ai  # noqa: E402

_VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    repo_root = os.path.dirname(os.path.dirname(__file__))
    upload_dir = os.path.join(repo_root, "uploads")
    reports_root = os.path.join(repo_root, "reports")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(reports_root, exist_ok=True)

    @app.get("/uploads/<path:filename>")
    def serve_uploads(filename: str):
        return send_from_directory(upload_dir, filename)

    @app.get("/reports/<path:filepath>")
    def serve_reports(filepath: str):
        """Serve files under reports/<report_id>/..."""
        return send_from_directory(reports_root, filepath)

    @app.get("/")
    def home() -> dict[str, str]:
        return {
            "message": "Backend Running",
            "health": "/health",
            "predict": "/predict (POST multipart/form-data: file)",
        }

    @app.get("/favicon.ico")
    def favicon():
        return ("", 204)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict_route() -> Any:
        if "file" not in request.files:
            return jsonify({"error": "Missing form field 'file'."}), 400

        file = request.files["file"]
        raw_name = secure_filename(file.filename or "upload.jpg")
        suffix = Path(raw_name).suffix.lower()
        if suffix not in _VALID_SUFFIXES:
            suffix = ".jpg"

        report_id = uuid.uuid4().hex[:12]
        report_dir = Path(reports_root) / report_id
        report_dir.mkdir(parents=True, exist_ok=True)

        original_path = report_dir / f"original{suffix}"
        file.save(str(original_path))

        heatmap_path = report_dir / "heatmap.jpg"

        try:
            result = predict_with_explanations(
                str(original_path), heatmap_output_path=str(heatmap_path)
            )
            pdf_path = report_dir / "report.pdf"
            write_prediction_pdf(
                output_path=pdf_path,
                result=result,
                image_paths={
                    "original": original_path,
                    "heatmap": heatmap_path,
                },
            )
            urls = report_public_urls(report_id, suffix)
            result.update(urls)
            # Avoid leaking local filesystem paths to clients.
            result.pop("heatmap_path", None)
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {e}"}), 500

        return jsonify(result)

    @app.post("/ai/ask")
    def ai_ask_route() -> Any:
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        if not question:
            return jsonify({"error": "Missing field 'question'."}), 400

        predicted_class = payload.get("predicted_class")
        if predicted_class is not None:
            predicted_class = str(predicted_class)

        top_predictions = payload.get("top_predictions")
        if top_predictions is not None and not isinstance(top_predictions, list):
            top_predictions = None

        result = ask_ai(
            question=question,
            predicted_class=predicted_class,
            top_predictions=top_predictions,
        )
        status = 200 if "answer" in result else 400
        return jsonify(result), status

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

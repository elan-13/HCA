from __future__ import annotations

from pathlib import Path
from typing import Any

from fpdf import FPDF


def _safe_text(value: Any, max_len: int = 4000) -> str:
    if value is None:
        return ""
    s = str(value).replace("\r\n", "\n")
    out: list[str] = []
    for ch in s[:max_len]:
        o = ord(ch)
        if ch == "\n":
            out.append("\n")
        elif 32 <= o <= 126 or ch in "\t":
            out.append(ch)
        else:
            out.append("?")
    return "".join(out)


def write_prediction_pdf(
    *,
    output_path: Path,
    result: dict[str, Any],
    image_paths: dict[str, Path],
) -> None:
    """Write a PDF summary with text + embedded images."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "SkinAI - Analysis report", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", size=11)
    cls = _safe_text(result.get("class"))
    conf = result.get("confidence")
    conf_s = f"{float(conf):.4f}" if conf is not None else "-"
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Predicted class: {cls}", ln=True)
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 6, f"Confidence (raw): {conf_s}", ln=True)
    if result.get("confidence_level"):
        pdf.cell(
            0,
            6,
            f"Confidence level: {_safe_text(result.get('confidence_level'))}",
            ln=True,
        )
    if result.get("risk"):
        pdf.cell(0, 6, f"Risk: {_safe_text(result.get('risk'))}", ln=True)
    pdf.ln(4)

    for label in ("message", "warning", "description"):
        val = result.get(label)
        if val:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 6, label.replace("_", " ").title(), ln=True)
            pdf.set_font("Helvetica", size=10)
            pdf.multi_cell(0, 5, _safe_text(val))
            pdf.ln(2)

    meds = result.get("medications") or []
    if isinstance(meds, list) and meds:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "Medications (general)", ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 5, _safe_text("\n".join(f"- {m}" for m in meds)))
        pdf.ln(2)

    prev = result.get("prevention") or []
    if isinstance(prev, list) and prev:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "Prevention", ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 5, _safe_text("\n".join(f"- {p}" for p in prev)))
        pdf.ln(2)

    disc = result.get("disclaimer")
    if disc:
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 4, _safe_text(disc))
        pdf.ln(4)

    orig = image_paths.get("original")
    heat = image_paths.get("heatmap")
    if orig and orig.exists():
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Input image", ln=True)
        try:
            pdf.image(str(orig), x=10, w=pdf.w - 20)
        except Exception:
            pdf.set_font("Helvetica", size=10)
            pdf.multi_cell(0, 5, "(Could not embed original image.)")
    if heat and heat.exists():
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Grad-CAM heatmap", ln=True)
        try:
            pdf.image(str(heat), x=10, w=pdf.w - 20)
        except Exception:
            pdf.set_font("Helvetica", size=10)
            pdf.multi_cell(0, 5, "(Could not embed heatmap image.)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))


def report_public_urls(report_id: str, original_suffix: str) -> dict[str, str]:
    base = f"/reports/{report_id}"
    return {
        "report_id": report_id,
        "report_folder_url": f"{base}/",
        "original_image_url": f"{base}/original{original_suffix}",
        "heatmap_url": f"{base}/heatmap.jpg",
        "report_pdf_url": f"{base}/report.pdf",
    }

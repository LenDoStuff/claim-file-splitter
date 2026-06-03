from __future__ import annotations

from pathlib import Path
from typing import Any

from .customization import DOCUMENT_CATEGORIES


DOCUMENT_TYPES = tuple(DOCUMENT_CATEGORIES)
DOCUMENT_TYPE_PREFIXES = DOCUMENT_CATEGORIES


def normalize_document_type(value: str | None) -> str:
    if not value:
        return "other"

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "repair_invoice": "repair_invoices",
        "invoice": "repair_invoices",
        "invoices": "repair_invoices",
        "appraisal": "appraisals",
        "estimate": "appraisals",
        "emails": "communications",
        "email": "communications",
        "communication": "communications",
        "police": "police_reports",
        "police_report": "police_reports",
        "photo": "photos",
        "image": "photos",
        "images": "photos",
        "payment": "payments",
        "payment_documents": "payments",
        "medical_documents": "medical",
        "legal": "legal_correspondence",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in DOCUMENT_TYPES else "other"


def make_decision(
    page_number: int,
    document_type: str,
    starts_new_document: bool,
    *,
    title: str = "",
    confidence: float = 0.0,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "page_number": int(page_number),
        "document_type": normalize_document_type(document_type),
        "starts_new_document": bool(starts_new_document),
        "title": (title or "").strip(),
        "confidence": max(0.0, min(1.0, float(confidence or 0.0))),
        "reason": (reason or "").strip(),
    }


def image_prompt_metadata(image: dict[str, Any] | None) -> dict[str, Any] | None:
    if image is None:
        return None
    return {
        "page": image["page_number"],
        "mime_type": image["mime_type"],
        "width_px": image["width_px"],
        "height_px": image["height_px"],
        "byte_size": image["byte_size"],
    }


def page_image_prompt(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "page": page["page_number"],
        "rendered_image": image_prompt_metadata(page.get("image")),
    }


def page_manifest(page: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "page": page["page_number"],
        "word_count": page["word_count"],
        "char_count": page["char_count"],
        "image_count": page["image_count"],
        "is_image_only": page["is_image_only"],
        "may_require_ocr": page["may_require_ocr"],
    }
    if page.get("image") is not None:
        image = dict(image_prompt_metadata(page["image"]) or {})
        if page["image"].get("path") is not None:
            image["path"] = str(page["image"]["path"])
        payload["rendered_image"] = image
    return payload


def segment_manifest(
    segment: dict[str, Any],
    output_path: Path | None = None,
) -> dict[str, Any]:
    payload = {
        "segment_id": segment["segment_id"],
        "document_type": segment["document_type"],
        "start_page": segment["start_page"],
        "end_page": segment["end_page"],
        "page_count": segment["end_page"] - segment["start_page"] + 1,
        "title": segment["title"],
        "confidence": round(segment["confidence"], 4),
        "reasons": segment["reasons"],
    }
    if output_path is not None:
        payload["output_path"] = str(output_path)
    return payload


def result_manifest(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_pdf": str(result["source_pdf"]),
        "output_dir": str(result["output_dir"]),
        "page_count": len(result["pages"]),
        "document_count": len(result["segments"]),
        "pages": [page_manifest(page) for page in result["pages"]],
        "page_decisions": result["page_decisions"],
        "classification_batches": result["classification_batches"],
        "documents": [
            segment_manifest(written["segment"], written["output_path"])
            for written in result["written_documents"]
        ],
    }

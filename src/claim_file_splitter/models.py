from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .customization import ClaimSplitterConfig


DOCUMENT_TYPE_ALIASES = {
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


class WrittenDocument(BaseModel):
    document_id: int
    name: str
    summary: str
    path: Path
    document_type: str
    start_page: int
    end_page: int
    page_count: int
    confidence: float = Field(ge=0.0, le=1.0)


class ClaimSplitResult(BaseModel):
    source_pdf: Path
    output_dir: Path
    manifest_path: Path
    documents: list[WrittenDocument]

    @property
    def document_count(self) -> int:
        return len(self.documents)


def normalize_document_type(
    value: str | None,
    config: ClaimSplitterConfig,
) -> str:
    configured_names = {category.name for category in config.categories}
    if not value:
        return config.default_document_type

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = DOCUMENT_TYPE_ALIASES.get(normalized, normalized)
    return normalized if normalized in configured_names else config.default_document_type


def make_decision(
    page_number: int,
    document_type: str,
    starts_new_document: bool,
    *,
    config: ClaimSplitterConfig,
    title: str = "",
    confidence: float = 0.0,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "page_number": int(page_number),
        "document_type": normalize_document_type(document_type, config),
        "starts_new_document": bool(starts_new_document),
        "title": (title or "").strip(),
        "confidence": max(0.0, min(1.0, float(confidence or 0.0))),
        "reason": (reason or "").strip(),
    }


def segment_manifest(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "segment_id": segment["segment_id"],
        "document_type": segment["document_type"],
        "start_page": segment["start_page"],
        "end_page": segment["end_page"],
        "page_count": segment["end_page"] - segment["start_page"] + 1,
        "title": segment["title"],
        "summary": segment["summary"],
        "confidence": round(segment["confidence"], 4),
        "reasons": segment["reasons"],
    }


def result_manifest(result: ClaimSplitResult) -> dict[str, Any]:
    documents = []
    for document in result.documents:
        item = document.model_dump(mode="json", exclude={"path"})
        item["path"] = str(document.path)
        item["confidence"] = round(document.confidence, 4)
        documents.append(item)

    return {
        "source_pdf": str(result.source_pdf),
        "output_dir": str(result.output_dir),
        "document_count": result.document_count,
        "documents": documents,
    }

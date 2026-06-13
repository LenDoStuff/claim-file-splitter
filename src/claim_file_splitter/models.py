from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .customization import ClaimSplitterConfig, category_names


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


class PageImage(BaseModel):
    page_number: int
    mime_type: str
    width_px: int
    height_px: int
    byte_size: int
    path: Path | None = None


class PageAnalysis(BaseModel):
    page_number: int
    word_count: int
    char_count: int
    image_count: int
    is_image_only: bool
    may_require_ocr: bool
    rendered_image: PageImage | None = None


class PageDecision(BaseModel):
    page_number: int
    document_type: str
    starts_new_document: bool
    title: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class ClassificationBatch(BaseModel):
    batch_number: int
    start_page: int
    end_page: int
    page_numbers: list[int]
    rolling_context: dict[str, Any]
    reconciliation_messages: list[str] = Field(default_factory=list)


class DocumentSegment(BaseModel):
    segment_id: int
    document_type: str
    start_page: int
    end_page: int
    page_count: int
    title: str
    confidence: float
    reasons: list[str] = Field(default_factory=list)


class WrittenDocument(BaseModel):
    segment: DocumentSegment
    output_path: Path


class ClaimSplitResult(BaseModel):
    source_pdf: Path
    output_dir: Path
    manifest_path: Path
    pages: list[PageAnalysis]
    page_decisions: list[PageDecision]
    classification_batches: list[ClassificationBatch]
    segments: list[DocumentSegment]
    documents: list[WrittenDocument]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def document_count(self) -> int:
        return len(self.segments)


def normalize_document_type(
    value: str | None,
    config: ClaimSplitterConfig,
) -> str:
    configured_names = set(category_names(config))
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


def segment_manifest(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "segment_id": segment["segment_id"],
        "document_type": segment["document_type"],
        "start_page": segment["start_page"],
        "end_page": segment["end_page"],
        "page_count": segment["end_page"] - segment["start_page"] + 1,
        "title": segment["title"],
        "confidence": round(segment["confidence"], 4),
        "reasons": segment["reasons"],
    }


def typed_result(raw: dict[str, Any]) -> ClaimSplitResult:
    segments = [typed_segment(segment) for segment in raw["segments"]]
    segment_by_id = {segment.segment_id: segment for segment in segments}
    documents = [
        WrittenDocument(
            segment=segment_by_id[written["segment"]["segment_id"]],
            output_path=written["output_path"],
        )
        for written in raw["written_documents"]
    ]
    return ClaimSplitResult(
        source_pdf=raw["source_pdf"],
        output_dir=raw["output_dir"],
        manifest_path=raw["manifest_path"],
        pages=[typed_page(page) for page in raw["pages"]],
        page_decisions=[
            PageDecision.model_validate(decision)
            for decision in raw["page_decisions"]
        ],
        classification_batches=[
            ClassificationBatch.model_validate(batch)
            for batch in raw["classification_batches"]
        ],
        segments=segments,
        documents=documents,
    )


def typed_segment(segment: dict[str, Any]) -> DocumentSegment:
    return DocumentSegment(
        **segment_manifest(segment),
    )


def typed_page(page: dict[str, Any]) -> PageAnalysis:
    rendered_image = None
    if page.get("image") is not None:
        image = page["image"]
        rendered_image = PageImage(
            page_number=image["page_number"],
            mime_type=image["mime_type"],
            width_px=image["width_px"],
            height_px=image["height_px"],
            byte_size=image["byte_size"],
            path=image.get("path"),
        )
    return PageAnalysis(
        page_number=page["page_number"],
        word_count=page["word_count"],
        char_count=page["char_count"],
        image_count=page["image_count"],
        is_image_only=page["is_image_only"],
        may_require_ocr=page["may_require_ocr"],
        rendered_image=rendered_image,
    )


def result_manifest(result: ClaimSplitResult) -> dict[str, Any]:
    return {
        "source_pdf": str(result.source_pdf),
        "output_dir": str(result.output_dir),
        "page_count": result.page_count,
        "document_count": result.document_count,
        "pages": [
            page.model_dump(mode="json", exclude_none=True)
            for page in result.pages
        ],
        "page_decisions": [
            decision.model_dump(mode="json")
            for decision in result.page_decisions
        ],
        "classification_batches": [
            batch.model_dump(mode="json")
            for batch in result.classification_batches
        ],
        "documents": [
            {
                **document.segment.model_dump(mode="json"),
                "output_path": str(document.output_path),
            }
            for document in result.documents
        ],
    }
